"""Engine for SPEC sessions with one TIFF frame per scan point (APS 7-ID, 2025).

A SPEC session is one text file: a header naming the motors (``#O`` labels,
``#o`` mnemonics) and the counters, then one block per scan with its command
(``#S 4 ascan phi 8.8 9.8 40 1``), the motor positions at its start (``#P``),
the beamline state (``#U_mono ...``), the column labels (``#L``) and one data
row per point. EPICS areaDetector saves the frame of every point,
``<file>_S004_00012.tif``: at the beamline in a folder per scan
(``images/<file>/S004/``), in a copy often next to the file; the spec's
``frames_dir`` lists the places to look. This engine reads the rows as the
per-frame motor and counter values and the TIFFs as the frames. A point whose
frame was not saved (or not copied) is marked missing and keeps its row in
:meth:`SpecTiffSource.point_table`, a scan aborted early (``#C ... Scan aborted
after N points.``) reports the points it planned, and the file is parsed once
per process (it grows while a session runs: ``refresh()``).
"""

from __future__ import annotations

import fnmatch
import os
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from bexa._log import get_logger
from bexa.core.registry import register_engine
from bexa.core.status import ScanNotReadyError, ScanStatus
from bexa.core.structure import MISSING, Structure
from bexa.io.base import BaseSource
from bexa.io.engines.hdf5_master import collapse_lockstep
from bexa.io.formats import FormatSpec, safe_format
from bexa.io.scan_commands import ScanCommand, parse_scan_command

if TYPE_CHECKING:
    import pandas as pd

log = get_logger(__name__)

__all__ = [
    "SpecFile",
    "SpecScan",
    "SpecTiffSource",
    "frame_point",
    "frame_templates",
    "frames_by_scan",
    "read_spec_file",
    "scan_frames",
    "session_frames",
    "spec_file_of",
]

SPEC_SUFFIX = ".spec"
SPEC_DATE = "%a %b %d %H:%M:%S %Y"  # the #D lines: "Wed Jul 30 18:43:28 2025"
TWO_SPACES = re.compile(r"\s{2,}")
FRAME_NAME = re.compile(r"^(?P<prefix>.+)_S(?P<scan>\d+)_(?P<point>\d+)\.tiff?$")
SCAN_FIELD = re.compile(r"\{scan(:[^{}]*)?\}")
FRAME_LEVELS = 3  # folders above a frame searched for its SPEC file: images/<file>/S004/
_CACHE: dict[Path, tuple[tuple[float, int], SpecFile]] = {}
_FRAMES: dict[tuple[Path, str], tuple[int, dict[int, dict[int, Path]]]] = {}
_CACHE_LOCK = threading.Lock()


# ------------------------------------------------------------------- the SPEC file
@dataclass
class SpecScan:
    """One scan of a SPEC file: its command, positions, beamline state and data rows."""

    number: int
    command: str
    date: str = ""
    labels: list[str] = field(default_factory=list)
    rows: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    positions: dict[str, float] = field(default_factory=dict)
    header: dict[str, dict[str, Any]] = field(default_factory=dict)
    comments: list[str] = field(default_factory=list)
    aborted: bool = False
    lines: tuple[int, int] = (0, 0)
    ended: bool = False  # SPEC closed it: #UXML or #R after the rows, or the abort comment

    @property
    def n_rows(self) -> int:
        """Data rows written: the points the scan recorded."""
        return int(self.rows.shape[0])

    def column(self, label: str) -> np.ndarray:
        """The data column under one ``#L`` label."""
        return self.rows[:, self.labels.index(label)]


@dataclass
class SpecFile:
    """A SPEC session file, parsed: the motors, the counters, the state blocks, the scans."""

    path: Path
    title: str = ""
    motor_labels: list[str] = field(default_factory=list)
    motor_names: list[str] = field(default_factory=list)
    counter_labels: list[str] = field(default_factory=list)
    counter_names: list[str] = field(default_factory=list)
    header_fields: dict[str, list[str]] = field(default_factory=dict)
    scans: dict[int, SpecScan] = field(default_factory=dict)

    @property
    def label_to_name(self) -> dict[str, str]:
        """``#O`` label -> ``#o`` mnemonic (``"Nano SY"`` -> ``"nsy"``)."""
        return dict(zip(self.motor_labels, self.motor_names, strict=False))

    def channel_names(self, scan: SpecScan) -> list[str]:
        """The per-frame channel of every data column: motors by mnemonic, the rest as labelled."""
        motors = self.label_to_name
        return [motors.get(label, label.replace(" ", "_")) for label in scan.labels]

    def table(self) -> pd.DataFrame:
        """One row per scan, read from the parse alone: the beamline log of the session.

        Columns ``scan``, ``command``, ``started`` (the ``#D`` time, the
        beamline's local time), ``points`` (data rows recorded), ``aborted``,
        then every ``#U_`` field as ``<block>_<field>`` (``mono_Energy``,
        ``ATTEN1_filter1_1stat``, ``ring_Current(mA)``).
        """
        import pandas as pd

        rows = []
        for number, scan in self.scans.items():
            row: dict[str, Any] = {
                "scan": number,
                "command": scan.command,
                "started": scan.date,
                "points": scan.n_rows,
                "aborted": scan.aborted,
            }
            for block, values in scan.header.items():
                row.update({f"{block}_{name}": value for name, value in values.items()})
            rows.append(row)
        table = pd.DataFrame(rows)
        if not table.empty:
            table["started"] = pd.to_datetime(table["started"], format=SPEC_DATE, errors="coerce")
        return table


def _values(tokens: list[str]) -> list[Any]:
    out: list[Any] = []
    for token in tokens:
        try:
            out.append(float(token))
        except ValueError:
            out.append(token)
    return out


def _labels(text: str) -> list[str]:
    """``#O``/``#L`` labels are separated by two spaces (a label may hold one space)."""
    text = text.strip()
    return [t for t in TWO_SPACES.split(text) if t] if text else []


def read_spec_file(path: str | Path) -> SpecFile:
    """Parse a SPEC session file, once per process (again when its size or mtime changed)."""
    path = Path(os.path.abspath(path))  # abspath, not resolve(): no round trip to a network share
    st = path.stat()
    key = (st.st_mtime, st.st_size)
    with _CACHE_LOCK:
        hit = _CACHE.get(path)
        if hit is not None and hit[0] == key:
            return hit[1]
    parsed = _parse(path)
    with _CACHE_LOCK:
        _CACHE[path] = (key, parsed)
    return parsed


def _parse(path: Path) -> SpecFile:
    session = SpecFile(path=path)
    scan: SpecScan | None = None
    rows: list[list[float]] = []
    positions: list[float] = []
    n_columns = 0

    def close(last_line: int) -> None:
        if scan is None:
            return
        scan.rows = np.array(rows, dtype=float) if rows else np.zeros((0, len(scan.labels)))
        scan.positions = dict(zip(session.motor_names, positions, strict=False))
        scan.lines = (scan.lines[0], last_line)
        if scan.number in session.scans:
            log.warning("%s: scan %d appears twice; the later one is kept", path.name, scan.number)
        session.scans[scan.number] = scan

    with open(path, encoding="utf-8", errors="replace") as f:
        for number, raw in enumerate(f, start=1):
            line = raw.rstrip("\r\n")
            if not line.strip():
                continue
            if line[0] != "#":
                if scan is not None:
                    values = _values(line.split())
                    if n_columns and len(values) != n_columns:
                        continue  # a broken row (the file was being written)
                    rows.append([v if isinstance(v, float) else np.nan for v in values])
                continue
            tag, _, rest = line[1:].partition(" ")
            if tag == "S":
                close(number - 1)
                num, _, command = rest.strip().partition(" ")
                scan = SpecScan(int(num), " ".join(command.split()), lines=(number, number))
                rows, positions, n_columns = [], [], 0
            elif tag == "F":
                session.title = rest.strip()
            elif tag == "D":
                if scan is not None:
                    scan.date = rest.strip()
            elif tag.startswith("O") and tag[1:].isdigit():
                session.motor_labels.extend(_labels(rest))
            elif tag.startswith("o") and tag[1:].isdigit():
                session.motor_names.extend(rest.split())
            elif tag.startswith("J") and tag[1:].isdigit():
                session.counter_labels.extend(_labels(rest))
            elif tag.startswith("j") and tag[1:].isdigit():
                session.counter_names.extend(rest.split())
            elif tag.startswith("U0_"):
                session.header_fields[tag[3:]] = rest.split()
            elif scan is None:
                continue
            elif tag.startswith("P") and tag[1:].isdigit():
                positions.extend(
                    v if isinstance(v, float) else np.nan for v in _values(rest.split())
                )
            elif tag.startswith("U_"):
                block = tag[2:]
                fields = session.header_fields.get(block, [])
                scan.header[block] = dict(zip(fields, _values(rest.split()), strict=False))
            elif tag == "N":
                n_columns = int(rest.split()[0]) if rest.split() else 0
            elif tag == "L":
                scan.labels = _labels(rest)
                if n_columns and len(scan.labels) != n_columns:
                    scan.labels = rest.split()  # labels separated by single spaces
            elif tag == "C":
                scan.comments.append(rest.strip())
                if "aborted" in rest.lower():
                    scan.aborted = scan.ended = True
            elif tag in ("R", "UXML"):
                scan.ended = True
    close(number if scan is not None else 0)
    return session


# -------------------------------------------------------------------- the frames
def frame_point(path: str | Path) -> int:
    """``<prefix>_S004_00012.tif`` -> 12 (-1 when the name does not follow the pattern)."""
    m = FRAME_NAME.match(Path(path).name)
    return int(m.group("point")) if m else -1


def frames_by_scan(folder: str | Path, prefix: str) -> dict[int, dict[int, Path]]:
    """``{scan: {point: path}}`` of every ``<prefix>_S<scan>_<point>.tif`` in ``folder``.

    One listing of the folder serves every scan of the session: the result is
    kept until the folder's modification time changes (a frame was added).
    """
    folder = Path(folder)
    try:
        stamp = folder.stat().st_mtime_ns
    except OSError:
        return {}
    key = (Path(os.path.abspath(folder)), prefix)
    with _CACHE_LOCK:
        hit = _FRAMES.get(key)
        if hit is not None and hit[0] == stamp:
            return hit[1]
    found: dict[int, dict[int, Path]] = {}
    try:
        entries = list(os.scandir(folder))
    except OSError:
        return found
    for entry in entries:
        m = FRAME_NAME.match(entry.name)
        if m is None or m.group("prefix") != prefix or not entry.is_file():
            continue
        found.setdefault(int(m.group("scan")), {})[int(m.group("point"))] = Path(entry.path)
    with _CACHE_LOCK:
        _FRAMES[key] = (stamp, found)
    return found


def frame_templates(layout: Mapping[str, Any]) -> list[str]:
    """The frame folders a spec's ``frames_dir`` names, in the order they are tried.

    One template or a list, relative to the SPEC file's folder, with the fields
    ``{prefix}`` (the file's stem) and ``{scan}``: ``"images/{prefix}/S{scan:03d}"``
    is the folder per scan of the beamline, ``"."`` the frames next to the file.
    """
    value = layout.get("frames_dir", ".")
    return [str(value)] if isinstance(value, (str, Path)) else [str(v) for v in value]


def scan_frames(spec_file: Path, scan: int, templates: Sequence[str] = (".",)) -> dict[int, Path]:
    """``{point: path}`` of one scan's frames, from the first template folder that holds any."""
    prefix = spec_file.stem
    for template in templates:
        folder = spec_file.parent / safe_format(template, prefix=prefix, scan=scan)
        found = frames_by_scan(folder, prefix).get(scan)
        if found:
            return found
    return {}


def session_frames(
    spec_file: Path, templates: Sequence[str] = (".",)
) -> dict[int, dict[int, Path]]:
    """``{scan: {point: path}}`` of every scan with frames, each frames folder listed once.

    A template with ``{scan}`` stands for every folder it matches with the scan
    number as a wildcard (``images/<file>/S*``); the first template holding
    frames of a scan wins, as in :func:`scan_frames`.
    """
    prefix = spec_file.stem
    found: dict[int, dict[int, Path]] = {}
    for template in templates:
        place = safe_format(template, prefix=prefix)
        if SCAN_FIELD.search(place):
            folders = sorted(spec_file.parent.glob(SCAN_FIELD.sub("*", place)))
        else:
            folders = [spec_file.parent / place]
        for folder in folders:
            for scan, points in frames_by_scan(folder, prefix).items():
                found.setdefault(scan, points)
    return found


def spec_file_of(path: str | Path, glob: str = "*.spec") -> Path:
    """The SPEC file behind a path: the file itself, the one in a folder, or a frame's session.

    A frame ``<prefix>_S004_00012.tif`` belongs to ``<prefix>.spec``, next to it
    or up to :data:`FRAME_LEVELS` folders above (``images/<prefix>/S004/``); for
    another file it is the one SPEC file of the file's folder.
    """
    path = Path(path)
    if path.suffix == SPEC_SUFFIX and path.is_file():
        return path
    if path.is_dir():
        return _one_spec_file(path, glob)
    m = FRAME_NAME.match(path.name)
    if m is not None:
        for folder in list(path.parents)[: FRAME_LEVELS + 1]:
            candidate = folder / f"{m.group('prefix')}{SPEC_SUFFIX}"
            if candidate.is_file():
                return candidate
    return _one_spec_file(path.parent, glob)


def _one_spec_file(folder: Path, glob: str) -> Path:
    try:  # one listing, no stat per entry
        with os.scandir(folder) as entries:
            found = sorted(
                Path(e.path) for e in entries if fnmatch.fnmatch(e.name, glob) and e.is_file()
            )
    except OSError:
        found = []
    if not found:
        raise FileNotFoundError(f"no SPEC file ({glob}) in {folder}")
    if len(found) > 1:
        names = [p.name for p in found]
        raise ValueError(f"{folder} holds several SPEC files {names}; give the file itself")
    return found[0]


def _read_frame(path: Path) -> np.ndarray:
    try:
        import tifffile

        return np.asarray(tifffile.imread(str(path)))
    except ImportError:
        from PIL import Image

        with Image.open(path) as im:
            return np.asarray(im)


def _probe_frame(path: Path) -> tuple[tuple[int, int], np.dtype]:
    """Shape and dtype of a TIFF without reading its pixels (tifffile), else by reading it."""
    try:
        import tifffile

        with tifffile.TiffFile(str(path)) as tf:
            page = tf.pages[0]
            return (int(page.shape[-2]), int(page.shape[-1])), np.dtype(page.dtype)
    except ImportError:
        frame = _read_frame(path)
        return (int(frame.shape[-2]), int(frame.shape[-1])), frame.dtype


# -------------------------------------------------------------------- the source
@register_engine("spec_tiff")
class SpecTiffSource(BaseSource):
    """One scan of a SPEC session as a frame stack, one TIFF per recorded point.

    Parameters
    ----------
    spec
        Format spec (``aps_7id_spec_2025`` or one derived from it).
    path
        The SPEC file, the folder holding it, or one of its frames.
    scan
        The scan number (``#S``).
    detector
        Name from the spec's ``detectors``; its ``shape`` is the frame shape of
        a scan without a frame yet, and pixels at its ``invalid_value`` (the
        EIGER's 2**32 - 1 for a dead or masked pixel) read as 0.
    energy_keV
        Overrides the energy the spec reads from the scan's ``#U_`` block.
    motors
        Data columns (mnemonics) to take as the dims instead of the command's.
    """

    name = "spec_tiff"
    dataset_layout = True
    spec: FormatSpec

    def __init__(
        self,
        spec: FormatSpec,
        path: str | Path,
        scan: int,
        detector: str | None = None,
        energy_keV: float | None = None,
        motors: Sequence[str] | None = None,
    ) -> None:
        super().__init__(spec)
        self.spec_file = spec_file_of(path, str(spec.layout.get("spec_glob", "*.spec")))
        self.scan = int(scan)
        self.detector = detector or next(iter(spec.detectors), "areadetector")
        invalid = spec.detector_info(self.detector).get("invalid_value")
        self._invalid = None if invalid is None else int(invalid)
        self._energy_override = energy_keV
        self._motors = [str(m) for m in motors] if motors is not None else None
        self._structure: Structure | None = None
        self._shape: tuple[int, int] | None = None
        self._dtype: np.dtype | None = None
        self._load()

    def _load(self) -> None:
        self.session = read_spec_file(self.spec_file)
        if self.scan not in self.session.scans:
            numbers = sorted(self.session.scans)
            span = f"{numbers[0]}..{numbers[-1]}" if numbers else "none"
            raise KeyError(f"no scan {self.scan} in {self.spec_file.name}; scans {span}")
        self.record = self.session.scans[self.scan]
        self.command: ScanCommand = parse_scan_command(
            self.record.command, motor_names=set(self.session.motor_names)
        )
        found = scan_frames(self.spec_file, self.scan, frame_templates(self.spec.layout))
        self.points = [p for p in sorted(found) if p < self.record.n_rows]
        self.frames = [found[p] for p in self.points]
        # while it runs, a frame is often written before SPEC writes its row
        if len(found) > len(self.points) and self.status is not ScanStatus.RUNNING:
            log.warning(
                "%s scan %d: %d frames have no data row and are left out",
                self.spec_file.name,
                self.scan,
                len(found) - len(self.points),
            )

    # ----------------------------------------------------------- discovery
    @classmethod
    def from_path(
        cls, spec: FormatSpec, path: str | Path, scan: int | None = None, **kwargs: Any
    ) -> SpecTiffSource:
        """From the SPEC file, its folder or one of its frames (whose name gives the scan)."""
        path = Path(path)
        spec_file = spec_file_of(path, str(spec.layout.get("spec_glob", "*.spec")))
        if scan is None and path != spec_file and path.is_file():
            m = FRAME_NAME.match(path.name)
            if m is None or m.group("prefix") != spec_file.stem:  # a file of another program
                raise LookupError(
                    f"{path.name} is not a frame of {spec_file.name}, whose frames are named "
                    f"{spec_file.stem}_S<scan>_<point>.tif: pass scan=<n> to open one of its "
                    'scans, or open the file alone with format="generic_stack"'
                )
            scan = int(m.group("scan"))
        if scan is None:
            numbers = sorted(read_spec_file(spec_file).scans)
            if not numbers:
                raise ValueError(f"{spec_file} holds no scan")
            if len(numbers) > 1:
                raise LookupError(
                    f"{spec_file.name} holds {len(numbers)} scans ({numbers[0]}..{numbers[-1]}): "
                    "pass scan=<n>, or open the folder with bexa.open_dataset to list them"
                )
            scan = numbers[0]
        return cls(spec, spec_file, int(scan), **kwargs)

    @classmethod
    def list_scans(
        cls,
        spec: FormatSpec,
        root: str | Path,
        dataset: str,
        detector: str | None = None,
        require_files: bool = True,
    ) -> list[int]:
        """The scan numbers of a session: ``root/dataset`` its folder, or ``root/dataset.spec``."""
        root = Path(root)
        glob = str(spec.layout.get("spec_glob", "*.spec"))
        candidate = root / dataset
        if candidate.is_dir():
            spec_file = spec_file_of(candidate, glob)
        elif (root / f"{dataset}{SPEC_SUFFIX}").is_file():
            spec_file = root / f"{dataset}{SPEC_SUFFIX}"
        else:  # bexa.list_scans names a file as <root>/<dataset>/<dataset>.h5: one level down
            below = sorted(root.glob(f"*/{dataset}{SPEC_SUFFIX}"))
            spec_file = below[0] if len(below) == 1 else spec_file_of(candidate, glob)
        session = read_spec_file(spec_file)
        numbers = sorted(session.scans)
        if require_files:
            with_frames = session_frames(spec_file, frame_templates(spec.layout))
            numbers = [n for n in numbers if with_frames.get(n)]
        return numbers

    @property
    def dataset(self) -> str:
        """The session's name, the SPEC file's stem: what ``Scan.name`` shows."""
        return self.spec_file.stem

    # ------------------------------------------------------------ the frames
    def files(self) -> list[Path]:
        return [self.spec_file, *self.frames]

    def _probe(self) -> None:
        if self.frames:
            self._shape, self._dtype = _probe_frame(self.frames[0])
            return
        info = self.spec.detector_info(self.detector)
        shape = info.get("shape")
        self._shape = (int(shape[0]), int(shape[1])) if shape else (1, 1)
        self._dtype = np.dtype(info.get("dtype", "uint32"))

    @property
    def frame_shape(self) -> tuple[int, int]:
        if self._shape is None:
            self._probe()
        assert self._shape is not None
        return self._shape

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def dtype(self) -> np.dtype:
        if self._dtype is None:
            self._probe()
        assert self._dtype is not None
        return self._dtype

    @property
    def energy_keV(self) -> float | None:
        """The energy given, else the ``#U_`` field the spec names (``energy: {header, field}``)."""
        if self._energy_override is not None:
            return float(self._energy_override)
        block = self.spec.energy.get("header")
        name = self.spec.energy.get("field")
        value = self.record.header.get(str(block), {}).get(str(name)) if block and name else None
        return float(value) if isinstance(value, (int, float)) else None

    def read_frames(
        self,
        index: np.ndarray | slice,
        y: slice = slice(None),
        x: slice = slice(None),
        dtype: Any = None,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        ids = self.normalise_index(index, self.n_frames)
        h, w = self.frame_shape
        ny = len(range(*y.indices(h)))
        nx = len(range(*x.indices(w)))
        target = np.dtype(dtype) if dtype is not None else self.dtype
        if out is None:
            out = np.empty((ids.size, ny, nx), dtype=target)
        for k, fid in enumerate(ids):
            frame = _read_frame(self.frames[fid])[y, x]
            if self._invalid is not None:  # a flagged pixel would add 4e9 counts to every sum
                frame = np.where(frame == self._invalid, 0, frame)
            out[k] = frame
        return out

    # --------------------------------------------------------- the structure
    def structure(self) -> Structure:
        if self._structure is None:
            self._structure = self._build_structure()
        return self._structure

    def _build_structure(self) -> Structure:
        record, session = self.record, self.session
        if record.n_rows == 0:
            why = ": SPEC aborted it before its first point" if record.aborted else " yet"
            raise ScanNotReadyError(
                f"scan {self.scan} of {self.spec_file.name} has no data row{why}",
                self.status,  # running when SPEC has just started it
                self.scan,
            )
        names = session.channel_names(record)
        columns = {name: record.rows[:, i] for i, name in enumerate(names)}
        chosen = self._motors or [m for m in self.command.names if m in columns]
        unknown = [m for m in chosen if m not in columns]
        if unknown:
            raise ValueError(
                f"scan {self.scan} of {self.spec_file.name} records no column {unknown}; "
                f"choose from {names}"
            )
        order, _ = collapse_lockstep(chosen, columns) if chosen else ([], {})
        used = np.asarray(self.points, dtype=np.int64)  # the rows that have a frame
        exact = self.command.exact and bool(order) and self._motors is None
        units = self.spec.motors.get("units", {}) or {}
        common: dict[str, Any] = {
            "scan_type": self.command.command or "list",  # "loopscan" for frames at one place
            "title": record.command,
            "units": {m: units[m] for m in order if m in units},
            "energy_keV": self.energy_keV,
            "scalars": dict(record.positions),
            "motor_names": frozenset(session.motor_names),
            "planned_shape": self.command.shape if exact else None,
            "planned_frames": self.command.n_points if exact else None,
            "per_frame": {name: columns[name][used] for name in names},
        }
        if order:  # the grid of every recorded point; the points without a frame stay missing
            full = Structure.from_per_frame({m: columns[m] for m in order}, self.frame_shape)
        else:
            full = Structure.frames_only(record.n_rows, self.frame_shape)
        remap = np.full(record.n_rows, MISSING, dtype=np.int64)
        remap[used] = np.arange(len(used))
        index = np.where(full.frame_index >= 0, remap[np.maximum(full.frame_index, 0)], MISSING)
        return Structure(
            motor_dims=full.motor_dims,
            motor_shape=full.motor_shape,
            coords=full.coords,
            frame_index=index,
            frame_shape=self.frame_shape,
            n_frames=len(used),
            **common,
        )

    @property
    def status(self) -> ScanStatus:
        """``running`` while SPEC writes it (the session's last scan, short of its points and
        not closed yet); ``empty`` without a frame or row; ``aborted`` when SPEC says so or
        frames are missing."""
        record = self.record
        planned = self.command.n_points if self.command.exact else None
        last = max(self.session.scans) == self.scan
        short = planned is None or record.n_rows < planned
        if last and not record.ended and short:
            return ScanStatus.RUNNING
        if record.n_rows == 0 or not self.frames:
            return ScanStatus.EMPTY
        if record.aborted or (planned is not None and record.n_rows < planned):
            return ScanStatus.ABORTED
        return ScanStatus.COMPLETE if len(self.frames) >= record.n_rows else ScanStatus.ABORTED

    def with_motors(self, motors: Sequence[str]) -> SpecTiffSource:
        return SpecTiffSource(
            self.spec,
            self.spec_file,
            self.scan,
            self.detector,
            self._energy_override,
            motors=list(motors),
        )

    def scalars(self) -> dict[str, float]:
        return dict(self.record.positions)

    def point_table(self) -> pd.DataFrame:
        """One row per point SPEC recorded, with or without its frame: every data column.

        ``point`` is the row number (the ``<point>`` of the frame's name) and
        ``frame`` the id :meth:`read_frames` reads for it, -1 when the frame was
        not saved or not copied; then the columns by channel name: the motors by
        mnemonic (``phi``, ``nsz``), the rest as labelled (``r1``, ``IC_1``,
        ``Epoch``). A session copied without its images keeps all of them.
        """
        import pandas as pd

        record = self.record
        table = pd.DataFrame(record.rows, columns=self.session.channel_names(record))
        frame = np.full(record.n_rows, -1, dtype=np.int64)
        frame[self.points] = np.arange(len(self.points))
        table.insert(0, "frame", frame)
        table.insert(0, "point", np.arange(record.n_rows))
        return table

    def aux_tables(self) -> dict[str, pd.DataFrame]:
        """``points`` (:meth:`point_table`) and ``header``: the scan's ``#U_`` state in one row.

        The header has one ``<block>_<field>`` column per field (``mono_Energy``).
        """
        import pandas as pd

        tables = {"points": self.point_table()}
        flat = {
            f"{block}_{name}": value
            for block, values in self.record.header.items()
            for name, value in values.items()
        }
        if flat:
            tables["header"] = pd.DataFrame([flat])
        return tables

    def cache_records(self) -> list[dict[str, Any]]:
        """The scan's rows in the (growing) SPEC file, and the frame files."""
        from bexa.core.provenance import file_records

        own = {
            "path": str(self.spec_file),
            "scan": self.scan,
            "rows": self.record.n_rows,
            "command": self.record.command,
        }
        return [own, *file_records(self.frames)]

    def refresh(self) -> SpecTiffSource:
        """Read the SPEC file and the frames folders again (a session still running)."""
        self._structure = None
        self._shape = self._dtype = None
        self._load()
        return self

    def close(self) -> None:
        return None
