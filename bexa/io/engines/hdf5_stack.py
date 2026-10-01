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
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from bexa._log import get_logger
from bexa.core.registry import register_engine
from bexa.core.structure import ENERGY_DIM, Structure
from bexa.core.units import mono_angle_to_energy
from bexa.io.base import BaseSource
from bexa.io.formats import FormatSpec

log = get_logger(__name__)

# Motors v9 looked for when no scan description is available.
FALLBACK_MOTORS = ("mu", "chi", "phi", "theta", "omega", "delta", "gamma", "ux", "uy", "uz")
# the datasets of fscan_parameters that name the scanned motors, slowest loop first: BLISS
# writes ``motor`` for an fscan, ``slow_motor``/``fast_motor`` for an fscan2d and
# ``slow1_motor``/``slow2_motor``/``fast_motor`` for an fscan3d (slow1 is the outer loop);
# the ``outer``/``middle``/``inner`` names are accepted too
MOTOR_KEYS = (
    "outer_motor",
    "slow1_motor",
    "slow_motor",
    "middle_motor",
    "slow2_motor",
    "slow3_motor",
    "inner_motor",
    "fast_motor",
    "motor",
)
MIN_CACHE_BYTES = 8 * 1024**2
MAX_CACHE_BYTES = 512 * 1024**2


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray) and value.dtype.kind in "SO" and value.ndim == 0:
        return _decode(value[()])
    return value


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
    ) -> None:
        super().__init__(spec)
        self.root = Path(root)
        self.dataset = dataset
        self.scan = int(scan)
        self.detector = detector or next(iter(spec.detectors), "pco_ff")
        self._energy_override = energy_keV
        self.master = Path(master) if master else self.root / spec.path("master", dataset=dataset)
        self.scan_folder = self.root / spec.path("scan_folder", dataset=dataset, scan=self.scan)
        self.frames_path = spec.path("frames", detector=self.detector)
        self._entry = (
            spec.path("entry", scan=self.scan) if "entry" in spec.layout else f"{self.scan}.1"
        )
        self._local = threading.local()
        self._structure: Structure | None = None
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
        cls, spec: FormatSpec, root: str | Path, dataset: str, detector: str | None = None
    ) -> list[int]:
        """Scan numbers that have detector files for ``detector``."""
        root = Path(root)
        detector = detector or next(iter(spec.detectors), "pco_ff")
        pattern = spec.path("detector_files", detector=detector)
        found = []
        for folder in sorted((root / dataset).glob("scan[0-9]*")):
            if folder.is_dir() and any(folder.glob(pattern)):
                found.append(int(folder.name[4:]))
        return found

    def _scan_files(self) -> None:
        pattern = self.spec.path("detector_files", detector=self.detector)
        files = sorted(p for p in self.scan_folder.glob(pattern) if p.is_file())
        if not files:
            raise FileNotFoundError(
                f"no {self.detector} files matching {pattern!r} in {self.scan_folder}"
            )
        meta: list[tuple[Path, int, int]] = []
        offset = 0
        for path in files:
            with h5py.File(path, "r") as f:
                if self.frames_path not in f:
                    raise KeyError(f"{path} has no dataset {self.frames_path!r}")
                ds = f[self.frames_path]
                n = int(ds.shape[0])
                if not meta:
                    self._frame_shape = (int(ds.shape[1]), int(ds.shape[2]))
                    self._dtype = np.dtype(ds.dtype)
                    self._chunks = tuple(ds.chunks) if ds.chunks else None
            meta.append((path, n, offset))
            offset += n
        self._files = meta
        self._n_frames = offset

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
            f = h5py.File(path, "r", **self._cache_settings())
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
                block = ds[start:stop, y, x]
                out[pos : pos + count] = block
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
        return self

    # ----------------------------------------------------------- structure
    def structure(self) -> Structure:
        if self._structure is None:
            self._structure = self._build_structure()
        return self._structure

    def _read_motor_table(self, f: h5py.File) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """Per-frame channels (length n_frames) and scalar positioners."""
        per_frame: dict[str, np.ndarray] = {}
        scalars: dict[str, float] = {}
        meas_path = self.spec.motor_path("per_frame", scan=self.scan, motor="").rstrip("/")
        if meas_path in f:
            for key, node in f[meas_path].items():
                if isinstance(node, h5py.Dataset) and node.ndim == 1 and node.dtype.kind in "fiu":
                    values = node[()]
                    if len(values) >= self._n_frames:
                        per_frame[key] = np.asarray(values[: self._n_frames], dtype=float)
        pos_path = self.spec.motor_path("scalar", scan=self.scan, motor="").rstrip("/")
        if pos_path in f:
            for key, node in f[pos_path].items():
                if isinstance(node, h5py.Dataset) and node.ndim == 0 and node.dtype.kind in "fiu":
                    scalars[key] = float(node[()])
        return per_frame, scalars

    def _declared_motors(self, f: h5py.File) -> tuple[str, list[str], list[int]] | None:
        """(scan_type, motor names slow first, points per motor) from the scan description."""
        template = self.spec.motors.get("structure")
        if not template:
            return None
        group_path = self.spec.motor_path("structure", scan=self.scan)
        if group_path not in f:
            return None
        g = f[group_path]
        scan_type = None
        for key in ("scan_type", "scan_name"):
            if key in g:
                scan_type = str(_decode(g[key][()]))
                break
        if scan_type is None:
            return None
        names: list[str] = []
        counts: list[int] = []
        for motor_key in MOTOR_KEYS:
            if motor_key in g:
                count_key = motor_key.replace("motor", "npoints")
                names.append(str(_decode(g[motor_key][()])))
                counts.append(int(g[count_key][()]) if count_key in g else 0)
        return scan_type, names, counts

    def _energies(self, angles: np.ndarray) -> np.ndarray:
        """Energies in keV of per-frame monochromator angles, with the spec's crystal."""
        crystal = self.spec.energy.get("crystal", "Si111")
        d_spacing = self.spec.energy.get("d_spacing_A")
        angles = np.asarray(angles, dtype=float)
        return np.asarray(mono_angle_to_energy(angles, d_spacing or crystal), dtype=float)

    def _build_structure(self) -> Structure:
        with h5py.File(self.master, "r") as f:
            if self._entry not in f:
                raise KeyError(f"scan {self.scan} ({self._entry!r}) not found in {self.master}")
            per_frame, scalars = self._read_motor_table(f)
            declared = self._declared_motors(f)
            title_path = f"{self._entry}/title"
            title = str(_decode(f[title_path][()])) if title_path in f else ""

        # A motor the beamline renamed (mu -> mu_new1) keeps its logical name everywhere:
        # in the per-frame channels, the positioners and the motors the scan declares. A
        # physical name the spec lists under ``known`` is a motor of its own (``samz`` with
        # its legacy alias ``z1``), so only unknown names are renamed.
        aliases = self.spec.motors.get("aliases", {}) or {}
        known = set(self.spec.known_motors())
        logical_name: dict[str, str] = {}
        for logical, physical in aliases.items():
            for name in [physical] if isinstance(physical, str) else physical:
                if name not in known:
                    logical_name[name] = logical
                if name in per_frame and logical not in per_frame:
                    per_frame[logical] = per_frame[name]
                if name in scalars and logical not in scalars:
                    scalars[logical] = scalars[name]

        order: list[str] | None = None
        scan_type = "list"
        if declared:
            scan_type, names, _counts = declared
            names = [logical_name.get(n, n) for n in names]
            order = [n for n in names if n in per_frame]
            if len(order) != len(names):
                log.warning(
                    "%s scan %d declares motors %s but the master only records %s",
                    self.dataset,
                    self.scan,
                    names,
                    list(per_frame),
                )
        if not order:
            candidates = list(dict.fromkeys([*self.spec.known_motors(), *FALLBACK_MOTORS]))
            order = [
                m
                for m in candidates
                if m in per_frame
                and np.std(per_frame[m]) > 1e-6
                and m != self.spec.energy.get("from")
            ]
            if len(order) > 3:
                order = order[:3]
            scan_type = f"fscan{len(order)}d" if order else "list"

        units = dict(self.spec.motors.get("units", {}) or {})
        energy = self._energy(per_frame, scalars)
        if order:
            motors = {m: per_frame[m] for m in order}
            source = self.spec.energy.get("from")
            if source in motors:
                # the monochromator angle was stepped inside the scan (an energy mosa): the dim
                # is the energy, in keV, and the angle stays a per-frame channel
                motors = {(ENERGY_DIM if m == source else m): v for m, v in motors.items()}
                motors[ENERGY_DIM] = self._energies(per_frame[source])
                units[ENERGY_DIM] = "keV"
            structure = Structure.from_per_frame(
                motors,
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
            structure.title = title
            return structure
        structure = Structure.frames_only(
            self._n_frames,
            self._frame_shape,
            per_frame=per_frame,
            scalars=scalars,
            units=units,
            energy_keV=energy,
        )
        structure.title = title
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
