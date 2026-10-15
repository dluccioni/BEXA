"""Engine for master-plus-detector-file layouts (ESRF BLISS).

One master HDF5 per dataset holds the motor readbacks of every scan; each
``scanNNNN`` folder holds detector files with the frames as one 3-D dataset,
possibly split over several files. Paths and key names come from the format
spec, so the same engine serves the 2024, 2025 and 2026 ID03 layouts.
"""

from __future__ import annotations

import contextlib
import re
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from bexa._log import get_logger
from bexa.core.registry import register_engine
from bexa.core.status import ScanNotReadyError, ScanStatus
from bexa.core.structure import ENERGY_DIM, Structure
from bexa.core.units import mono_angle_to_energy
from bexa.io.base import BaseSource
from bexa.io.engines.hdf5_master import (
    FALLBACK_MOTORS,
    MOTOR_KEYS,
    collapse_lockstep,
    decide,
    declared_motors,
    discover_motors,
    entry_state,
    read_motor_table,
    read_title,
)
from bexa.io.formats import FormatSpec
from bexa.io.h5files import open_h5
from bexa.io.scan_commands import parse_scan_command

log = get_logger(__name__)

__all__ = ["FALLBACK_MOTORS", "MOTOR_KEYS", "Hdf5StackSource"]

MIN_CACHE_BYTES = 8 * 1024**2
MAX_CACHE_BYTES = 512 * 1024**2
SCAN_FOLDER = re.compile(r"scan(\d+)")


def _file_indices(files: Sequence[Path]) -> list[int] | None:
    """The numbers LIMA puts in the file names (``pco_ff_0003.h5`` -> 3), or ``None``."""
    found: list[int] = []
    for path in files:
        match = re.search(r"(\d+)\.[^.]+$", path.name)
        if match is None:
            return None
        found.append(int(match.group(1)))
    return found


def _next_prime(n: int) -> int:
    def is_prime(x: int) -> bool:
        return x >= 2 and all(x % i for i in range(2, int(x**0.5) + 1))

    while not is_prime(n):
        n += 1
    return n


@register_engine("hdf5_stack")
class Hdf5StackSource(BaseSource):
    """Frames from per-scan detector files, metadata from the dataset's master file.

    Parameters
    ----------
    spec
        Format spec describing the layout.
    root
        Folder that contains the dataset folder.
    dataset
        Dataset name (folder name and master file stem).
    scan
        Scan number.
    detector
        Detector name (``pco_ff``, ``pco_nf``, ...); default: first detector of the spec.
    energy_keV
        Overrides the energy derived from the monochromator angle.
    motors
        Per-frame channels to use as the dims instead of the ones the scan
        declares (``["ccmth", "chi", "mu"]`` for the mono angle in degrees
        instead of the energy; a motor's encoder instead of the motor).
    """

    name = "hdf5_stack"
    spec: FormatSpec

    def __init__(
        self,
        spec: FormatSpec,
        root: str | Path,
        dataset: str,
        scan: int,
        detector: str | None = None,
        energy_keV: float | None = None,
        master: str | Path | None = None,
        motors: Sequence[str] | None = None,
    ) -> None:
        super().__init__(spec)
        self.root = Path(root)
        self.dataset = dataset
        self.scan = int(scan)
        self.detector = detector or next(iter(spec.detectors), "pco_ff")
        self._energy_override = energy_keV
        self._motors = [str(m) for m in motors] if motors else None
        self.master = Path(master) if master else self.root / spec.path("master", dataset=dataset)
        self.scan_folder = self.root / spec.path("scan_folder", dataset=dataset, scan=self.scan)
        self.frames_path = spec.path("frames", detector=self.detector)
        self._entry = (
            spec.path("entry", scan=self.scan) if "entry" in spec.layout else f"{self.scan}.1"
        )
        self._local = threading.local()
        self._structure: Structure | None = None
        self._status: ScanStatus | None = None
        self._problem: str | None = None
        self._layout_known = False
        self._frame_shape: tuple[int, int] = (0, 0)
        self._dtype: np.dtype = np.dtype(np.uint16)
        self._chunks: tuple[int, ...] | None = None
        self._scan_files()

    # ----------------------------------------------------------- discovery
    @classmethod
    def from_path(
        cls, spec: FormatSpec, path: str | Path, scan: int | None = None, **kwargs: Any
    ) -> Hdf5StackSource:
        """Build a source from a dataset folder, a master file or a ``scanNNNN`` folder."""
        path = Path(path)
        if path.is_file():  # the master file
            root, dataset = path.parent.parent, path.stem
        elif re.fullmatch(r"scan\d+", path.name):
            root, dataset = path.parent.parent, path.parent.name
            if scan is None:
                scan = int(path.name[4:])
        else:
            root, dataset = path.parent, path.name
        if scan is None:
            scans = cls.list_scans(spec, root, dataset, kwargs.get("detector"))
            if len(scans) != 1:
                raise ValueError(
                    f"{dataset} has scans {scans}; pass scan=<n>, a (start, end) range or a list"
                )
            scan = scans[0]
        return cls(spec, root, dataset, scan, **kwargs)

    @classmethod
    def list_scans(
        cls,
        spec: FormatSpec,
        root: str | Path,
        dataset: str,
        detector: str | None = None,
        require_files: bool = True,
    ) -> list[int]:
        """Scan numbers of the ``scanNNNN`` folders, those with files for ``detector`` by default.

        A folder named differently (``scan0007_old``) is not a scan and is left
        out; ``require_files=False`` lists a scan folder that holds no detector
        file yet (a scan that has just started).
        """
        root = Path(root)
        detector = detector or next(iter(spec.detectors), "pco_ff")
        pattern = spec.path("detector_files", detector=detector)
        found = []
        for folder in sorted((root / dataset).glob("scan*")):
            match = SCAN_FOLDER.fullmatch(folder.name)
            if match is None or not folder.is_dir():
                if folder.is_dir():
                    log.debug("%s is not a scan folder; skipped", folder)
                continue
            if require_files and not any(folder.glob(pattern)):
                continue
            found.append(int(match.group(1)))
        return found

    def _scan_files(self) -> None:
        """Find the detector files and count their frames; none yet is not an error."""
        pattern = self.spec.path("detector_files", detector=self.detector)
        files = sorted(p for p in self.scan_folder.glob(pattern) if p.is_file())
        self._layout_known = False
        meta = self._files_from_master(files)
        if meta is None:
            meta = self._files_by_opening(files)
        self._files = meta
        self._n_frames = meta[-1][2] + meta[-1][1] if meta else 0
        if not self._layout_known:  # nothing readable yet: the shape the spec gives the detector
            shape = self.spec.detector_info(self.detector).get("shape")
            self._frame_shape = (int(shape[0]), int(shape[1])) if shape else (0, 0)
            self._dtype = np.dtype(np.uint16)
            self._chunks = None

    def _frame_layout(self, ds: Any) -> None:
        self._frame_shape = (int(ds.shape[1]), int(ds.shape[2]))
        self._dtype = np.dtype(ds.dtype)
        self._chunks = tuple(ds.chunks) if ds.chunks else None
        self._layout_known = True

    def _count_frames(self, path: Path) -> int | None:
        """Frames in one detector file, or ``None`` while the file cannot be read yet."""
        try:
            with open_h5(path) as f:
                if self.frames_path not in f:
                    return None
                ds = f[self.frames_path]
                if not self._layout_known:
                    self._frame_layout(ds)
                return int(ds.shape[0])
        except OSError:
            return None

    def _files_by_opening(self, files: list[Path]) -> list[tuple[Path, int, int]]:
        """Frames per file by opening every file: the sure way, and the slow one on a share.

        A file that cannot be read yet (the one the detector is writing) ends
        the list: the frames stop at the file before it.
        """
        meta: list[tuple[Path, int, int]] = []
        offset = 0
        for path in files:
            n = self._count_frames(path)
            if n is None:
                log.info("%s cannot be read yet; the frames stop at the file before it", path)
                break
            meta.append((path, n, offset))
            offset += n
        return meta

    def _files_from_master(self, files: list[Path]) -> list[tuple[Path, int, int]] | None:
        """Frames per file from what the master records, opening one file (two without a VDS).

        LIMA writes ``saving_frame_per_file`` frames to every file but the last,
        and the master says how many frames the scan has (``frames_total``) and,
        in the autumn 2026 layout, holds the frames as one virtual dataset
        (``master_frames``: the frame shape and dtype). Only the last file is
        opened, to count its frames, which is what a scan still being written
        or an aborted one changes. Over a network share this is the difference
        between one file open and sixty-four. A scan still being written has
        fewer files than the master announces; a last file that cannot be read
        yet is left out. ``None`` when the master does not say, or when the
        files do not add up: then every file is opened.
        """
        layout = self.spec.layout
        if "frames_total" not in layout or "frames_per_file" not in layout:
            return None
        fields = {"scan": self.scan, "detector": self.detector}
        try:
            with open_h5(self.master, retry=False) as f:
                total_path = self.spec.path("frames_total", **fields)
                per_path = self.spec.path("frames_per_file", **fields)
                if total_path not in f or per_path not in f:
                    return None
                total, per_file = int(f[total_path][()]), int(f[per_path][()])
                vds_path = (
                    self.spec.path("master_frames", **fields) if "master_frames" in layout else None
                )
                if vds_path is not None and vds_path in f:
                    self._frame_layout(f[vds_path])
                    self._chunks = None
        except (OSError, KeyError, ValueError, TypeError):
            return None
        indices = _file_indices(files)
        if total <= 0 or per_file <= 0 or len(files) > -(-total // per_file):
            return None
        if indices is not None and indices != list(range(len(files))):
            return None  # a file is missing in the middle: count what is there
        if not files:
            return []
        last = self._count_frames(files[-1])  # the last file: its count is the one that varies
        if last is None:
            log.info("%s cannot be read yet; the frames stop at the file before it", files[-1])
            files = files[:-1]
            if not files:
                return []
            last = self._count_frames(files[-1])
            if last is None:
                return None
        meta: list[tuple[Path, int, int]] = []
        offset = 0
        for i, path in enumerate(files):
            n = per_file if i < len(files) - 1 else last
            meta.append((path, n, offset))
            offset += n
        return meta

    # ---------------------------------------------------------- properties
    def files(self) -> list[Path]:
        return [self.master, *(p for p, _, _ in self._files)]

    def cache_records(self) -> list[dict[str, Any]]:
        """The detector files by size and mtime, the master by the entry this scan reads.

        BLISS appends every new scan to the master file, so its size and mtime
        change all through a beamtime while this scan's entry does not; keying on
        the file itself would discard every cached result of the dataset at each
        new scan.
        """
        from bexa.core.provenance import file_records

        master = {"path": str(self.master), "entry": self._entry, "n_frames": self._n_frames}
        return [master, *file_records(p for p, _, _ in self._files)]

    @property
    def frame_shape(self) -> tuple[int, int]:
        return self._frame_shape

    @property
    def n_frames(self) -> int:
        return self._n_frames

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def chunk_frames(self) -> int:
        return int(self._chunks[0]) if self._chunks else 1

    @property
    def energy_keV(self) -> float | None:
        return self.structure().energy_keV

    # ------------------------------------------------------------- reading
    def _cache_settings(self) -> dict[str, Any]:
        if not self._chunks:
            return {}
        chunk_bytes = int(np.prod(self._chunks)) * self._dtype.itemsize
        nbytes = int(min(max(4 * chunk_bytes, MIN_CACHE_BYTES), MAX_CACHE_BYTES))
        nslots = _next_prime(100 * max(nbytes // chunk_bytes, 1))
        return {"rdcc_nbytes": nbytes, "rdcc_nslots": nslots, "rdcc_w0": 0.0}

    def _handle(self, path: Path) -> h5py.File:
        """One open file per thread; HDF5 handles are not thread-safe to share."""
        handles = getattr(self._local, "handles", None)
        if handles is None:
            handles = self._local.handles = {}
        f = handles.get(path)
        if f is None or not f.id.valid:
            f = open_h5(path, **self._cache_settings())
            handles[path] = f
        return f

    def read_frames(
        self,
        index: np.ndarray | slice,
        y: slice = slice(None),
        x: slice = slice(None),
        dtype: Any = None,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        ids = self.normalise_index(index, self._n_frames)
        h, w = self._frame_shape
        ny = len(range(*y.indices(h)))
        nx = len(range(*x.indices(w)))
        target_dtype = np.dtype(dtype) if dtype is not None else self._dtype
        if out is None:
            out = np.empty((ids.size, ny, nx), dtype=target_dtype)
        if ids.size == 0:
            return out
        pos = 0
        for path, n, offset in self._files:
            local = ids[(ids >= offset) & (ids < offset + n)] - offset
            if local.size == 0:
                continue
            ds = self._handle(path)[self.frames_path]
            for start, stop in self.contiguous_runs(local):
                count = stop - start
                if out.dtype == ds.dtype:  # straight into the output: no copy in between
                    ds.read_direct(out, np.s_[start:stop, y, x], np.s_[pos : pos + count])
                else:
                    out[pos : pos + count] = ds[start:stop, y, x]
                pos += count
        return out

    def close(self) -> None:
        handles = getattr(self._local, "handles", None) or {}
        for f in handles.values():
            with contextlib.suppress(Exception):  # already closed
                f.close()
        handles.clear()

    def refresh(self) -> Hdf5StackSource:
        """Re-read file sizes and metadata (for a scan still being written)."""
        self.close()
        self._scan_files()
        self._structure = None
        self._status = None
        self._problem = None
        return self

    # ----------------------------------------------------------- structure
    def structure(self, motors: Sequence[str] | None = None) -> Structure:
        """The scan's grid; ``motors`` names other per-frame channels as the dims (not kept)."""
        if motors is not None:
            return self._build_structure([str(m) for m in motors])
        if self._structure is None:
            self._structure = self._build_structure(self._motors)
        return self._structure

    @property
    def status(self) -> ScanStatus:
        """The state of the scan, from its master entry (see :func:`hdf5_master.decide`)."""
        if self._status is None:
            try:
                self.structure()
            except ScanNotReadyError:
                pass  # the status was settled before the error was raised
            except (OSError, KeyError, IndexError, ValueError) as exc:
                self._status = ScanStatus.UNREADABLE
                self._problem = f"{type(exc).__name__}: {exc}"
        return self._status if self._status is not None else ScanStatus.UNREADABLE

    @property
    def problem(self) -> str | None:
        """What went wrong reading the scan, when its status is unreadable."""
        return self._problem

    @classmethod
    def scan_status(
        cls,
        spec: FormatSpec,
        root: str | Path,
        dataset: str,
        scan: int,
        detector: str | None = None,
    ) -> ScanStatus:
        """The status of one scan, opening and closing its files (no source is kept)."""
        try:
            source = cls(spec, root, dataset, scan, detector)
        except (OSError, KeyError, IndexError, ValueError):
            return ScanStatus.UNREADABLE
        try:
            return source.status
        finally:
            source.close()

    def with_motors(self, motors: Sequence[str]) -> Hdf5StackSource:
        """The same scan read with ``motors`` (per-frame channels) as its dims."""
        return Hdf5StackSource(
            self.spec,
            self.root,
            self.dataset,
            self.scan,
            self.detector,
            self._energy_override,
            self.master,
            motors=motors,
        )

    def _energies(self, angles: np.ndarray) -> np.ndarray:
        """Energies in keV of per-frame monochromator angles, with the spec's crystal."""
        crystal = self.spec.energy.get("crystal", "Si111")
        d_spacing = self.spec.energy.get("d_spacing_A")
        angles = np.asarray(angles, dtype=float)
        return np.asarray(mono_angle_to_energy(angles, d_spacing or crystal), dtype=float)

    def _logical_names(
        self, channels: dict[str, np.ndarray], scalars: dict[str, float]
    ) -> dict[str, str]:
        """``{physical: logical}`` for the motors the beamline renamed, copying their values.

        A motor the beamline renamed (mu -> mu_new1) keeps its logical name
        everywhere: in the per-frame channels, the positioners and the motors
        the scan declares. A physical name the spec lists under ``known`` is a
        motor of its own (``samz`` with its legacy alias ``z1``), so only
        unknown names are renamed.
        """
        aliases = self.spec.motors.get("aliases", {}) or {}
        known = set(self.spec.known_motors())
        logical_name: dict[str, str] = {}
        for logical, physical in aliases.items():
            for name in [physical] if isinstance(physical, str) else physical:
                if name not in known:
                    logical_name[name] = logical
                if name in channels and logical not in channels:
                    channels[logical] = channels[name]
                if name in scalars and logical not in scalars:
                    scalars[logical] = scalars[name]
        return logical_name

    def _build_structure(self, motors: list[str] | None) -> Structure:
        with open_h5(self.master) as f:
            state = entry_state(f, self._entry, self.spec, self.scan, self.detector)
            if not state.present:
                self._status = ScanStatus.RUNNING
                raise ScanNotReadyError(
                    f"scan {self.scan} ({self._entry!r}) is not in {self.master} yet",
                    ScanStatus.RUNNING,
                    self.scan,
                )
            table = read_motor_table(f, self.spec, self.scan, self.detector)
            declared = declared_motors(f, self.spec, self.scan)
            title = read_title(f, self._entry)
        channels, scalars = table.channels, table.scalars
        logical_name = self._logical_names(channels, scalars)
        known = self.spec.known_motors()
        motor_names = table.motor_names | set(known) | set(FALLBACK_MOTORS)
        command = (
            parse_scan_command(title, motor_names=motor_names | set(channels)) if title else None
        )
        # the frames the master describes: while a scan runs, its master lags or leads the
        # detector files, so the readbacks of the scanned motors say how far the scan goes
        named = declared[1] if declared else (command.names if command is not None else [])
        n = self._n_frames
        for name in (logical_name.get(m, m) for m in named):
            if name in channels:
                n = min(n, len(channels[name]))
        if n < self._n_frames:
            log.info(
                "%s scan %d: the master records %d of the %d frames on disk so far",
                self.dataset,
                self.scan,
                n,
                self._n_frames,
            )
            self._n_frames = n
        per_frame = {k: v[:n] for k, v in channels.items() if len(v) >= n}
        planned: int | None = state.planned_frames
        if planned is None and declared and declared[2] and all(c > 0 for c in declared[2]):
            planned = int(np.prod(declared[2]))
        if planned is None and command is not None:
            planned = command.n_points
        if n == 0:
            self._status = decide(state, 0, planned)
            raise ScanNotReadyError(
                f"scan {self.scan} of {self.dataset} is {self._status}: no frame to read yet",
                self._status,
                self.scan,
            )
        choice = discover_motors(
            per_frame, declared, command, motor_names, known, self.spec.min_steps(), logical_name
        )
        order, scan_type, planned_shape = choice.order, choice.scan_type, choice.planned_shape
        source = self.spec.energy.get("from")
        # the monochromator angle stepped inside a scan (an energy mosa) reads as the energy in
        # keV unless the reader asks for the angle itself (and names it before the energy)
        as_energy = motors is None or (
            ENERGY_DIM in motors
            and (
                source not in motors or list(motors).index(ENERGY_DIM) < list(motors).index(source)
            )
        )
        if motors is not None:  # the reader's choice of dims
            order = [
                source if m == ENERGY_DIM and source else logical_name.get(m, m) for m in motors
            ]
            unknown = [m for m in order if m not in per_frame]
            if unknown:
                raise ValueError(
                    f"{self.dataset} scan {self.scan} records no motor {unknown}; "
                    f"choose from {choice.candidates}"
                )
            order, _ = collapse_lockstep(order, per_frame)
            planned_shape = None

        units = dict(self.spec.motors.get("units", {}) or {})
        energy = self._energy(per_frame, scalars)
        if order:
            arrays = {m: per_frame[m] for m in order}
            if source in arrays and as_energy:
                # the dim is the energy, in keV, and the angle stays a per-frame channel
                arrays = {(ENERGY_DIM if m == source else m): v for m, v in arrays.items()}
                arrays[ENERGY_DIM] = self._energies(per_frame[source])
                units[ENERGY_DIM] = "keV"
            structure = Structure.from_per_frame(
                arrays,
                self._frame_shape,
                order=None,  # the readbacks say which loop is outer; ties keep the declared order
                scan_type=scan_type,
                units=units,
                energy_keV=energy,
            )
            structure.per_frame.update(
                {k: v for k, v in per_frame.items() if k not in structure.per_frame}
            )
            structure.scalars.update(scalars)
        else:
            structure = Structure.frames_only(
                self._n_frames,
                self._frame_shape,
                per_frame=per_frame,
                scalars=scalars,
                units=units,
                energy_keV=energy,
                scan_type=scan_type if scan_type != f"fscan{len(order)}d" else "list",
            )
        structure.title = title
        structure.planned_shape = planned_shape
        structure.planned_frames = planned
        structure.motor_names = frozenset(motor_names)
        self._status = decide(state, self._n_frames, planned)
        return structure

    def _energy(self, per_frame: dict[str, np.ndarray], scalars: dict[str, float]) -> float | None:
        if self._energy_override is not None:
            return float(self._energy_override)
        source = self.spec.energy.get("from")
        crystal = self.spec.energy.get("crystal", "Si111")
        d_spacing = self.spec.energy.get("d_spacing_A")
        if not source:
            return None
        angle = None
        if source in scalars:
            angle = scalars[source]
        elif source in per_frame:
            angle = float(np.mean(per_frame[source]))
        if angle is None:
            return None
        return mono_angle_to_energy(angle, d_spacing if d_spacing else crystal)
