"""Engine for the scans a Python script runs at APS 7-ID: a netCDF file per run, a TIFF per point.

A scan run from the beamline's Python scripts (a raster of the nano stages, say)
is not a SPEC scan: the script saves the run's metadata as ``run_<n>.nc`` and
the detector saves one areaDetector TIFF per point into a folder of the run,
under ``python_images/``. The netCDF file holds one array
(``xarray.load_dataset(path).to_array()``) whose dims are the motors scanned,
slow first, and ``det``, the counters read at every point; its coordinates are
the motor positions. This engine takes those dims as the grid, the counters as
per-frame channels and the TIFFs, by the point numbers of their names, as the
frames, placed in the order the points were taken: a counter that rises point by
point (``order_channel``) tells a serpentine raster from a plain one. Opened
through its frames alone a run is a series of frames; through this engine it
is the raster it was.

Nothing ties the two folders together but their names, so each is searched for
from the other: the frames' folders (``frames_dirs``) in the netCDF folder or in
a folder above it (``search_up`` levels), down to ``search_depth`` levels below
each, the nearest first; the netCDF folder from a frame the same way. A copy
that puts the TIFFs loose beside the netCDF files or the SPEC file
(``MapScan_102_1.tiff``: run 102, point 1) is read too. A run whose frames are
there but whose netCDF file is not (the script writes it at the end) is a
series of frames: running when it is the last run, aborted otherwise.
"""

from __future__ import annotations

import fnmatch
import os
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from bexa._log import get_logger
from bexa.core.registry import register_engine
from bexa.core.status import ScanStatus
from bexa.core.structure import MISSING, Structure, raster_order
from bexa.io.base import BaseSource
from bexa.io.engines.spec_tiff import TiffLayout, probe_tiff, read_tiffs
from bexa.io.formats import FolderListing, FormatSpec

if TYPE_CHECKING:
    import pandas as pd

log = get_logger(__name__)

__all__ = [
    "NcTiffSource",
    "RunGrid",
    "RunIndex",
    "RunLayout",
    "find_run_folder",
    "first_point",
    "read_run",
    "run_folder_of",
    "run_frames",
    "run_index",
    "run_of",
]

TIFF_SUFFIXES = (".tif", ".tiff")
SKIPPED = frozenset({"bexa_cache", "__pycache__"})
LAST_NUMBER = re.compile(r"(\d+)\D*$")
RUN_FIELD = re.compile(r"\{run(:[^{}]*)?\}")
MAX_FOLDERS = 64  # frame folders whose listing is kept (a run folder may hold 10 000 frames)
MAX_RUNS = 512  # netCDF files whose grid is kept (a few MB at most)
_LISTINGS: dict[Path, tuple[int, list[tuple[str, bool]]]] = {}
_FRAMES: dict[tuple[Path, int], tuple[int, dict[int, Path]]] = {}
_PLACES: dict[tuple[Path, str], list[Path]] = {}
_INDEXES: dict[tuple[Path, str], tuple[tuple[int, ...], RunIndex]] = {}
_GRIDS: dict[tuple[Path, RunLayout], tuple[tuple[int, int], RunGrid]] = {}
_LOCK = threading.Lock()


# ----------------------------------------------------------------- the layout
@dataclass(frozen=True)
class RunLayout:
    """The names of a Python-scan layout, read from a spec's ``layout``.

    ``run_file`` names a run's netCDF file (``run_{run}.nc``); ``frames_dirs``
    the folders holding a subfolder of frames per run (``python_images*``);
    ``frame_name`` a frame's name, with the groups ``run`` and ``point``;
    ``run_dir`` the run number in a frames subfolder's name. The search goes
    ``search_up`` folders above where it starts and ``search_depth`` below
    each, never into ``search_skip`` (SPEC's ``images``). ``channel_dim`` is the
    netCDF dim of the counters, named ``channel_names`` (``<dim>_<k>`` for the
    rest); ``order_channel`` the counter that rises point by point.
    """

    run_file: str
    frames_dirs: tuple[str, ...]
    frame_name: re.Pattern[str]
    run_dir: re.Pattern[str]
    search_up: int = 3
    search_depth: int = 2
    search_skip: tuple[str, ...] = ()
    channel_dim: str = "det"
    channel_names: tuple[str, ...] = ()
    order_channel: int | None = None
    first_point: int | None = None
    running_seconds: float = 120.0

    @classmethod
    def of(cls, spec: FormatSpec) -> RunLayout:
        layout = spec.layout
        order = layout.get("order_channel")
        first = layout.get("first_point")
        return cls(
            run_file=str(layout["run_file"]),
            frames_dirs=tuple(str(g) for g in layout.get("frames_dirs", ())),
            frame_name=re.compile(str(layout["frame_name"]), re.IGNORECASE),
            run_dir=re.compile(str(layout.get("run_dir", LAST_NUMBER.pattern))),
            search_up=int(layout.get("search_up", 3)),
            search_depth=int(layout.get("search_depth", 2)),
            search_skip=tuple(str(g) for g in layout.get("search_skip", ())),
            channel_dim=str(layout.get("channel_dim", "det")),
            channel_names=tuple(str(n) for n in layout.get("channel_names", ()) or ()),
            order_channel=None if order is None else int(order),
            first_point=None if first in (None, "auto") else int(first),
            running_seconds=float(layout.get("running_seconds", 120.0)),
        )

    @property
    def run_glob(self) -> str:
        """``run_*.nc``: every run's file."""
        return RUN_FIELD.sub("*", self.run_file)

    @property
    def run_pattern(self) -> re.Pattern[str]:
        """``run_<n>.nc`` with the number as the group ``run``."""
        parts = RUN_FIELD.split(self.run_file)[::2]  # split() also returns the format specs
        return re.compile("^" + r"(?P<run>\d+)".join(re.escape(p) for p in parts) + "$")

    def run_of_file(self, name: str) -> int | None:
        m = self.run_pattern.match(name)
        return int(m.group("run")) if m else None

    def is_frames_dir(self, name: str) -> bool:
        return any(fnmatch.fnmatch(name, g) for g in self.frames_dirs)

    def is_skipped(self, name: str) -> bool:
        return (
            name.startswith(".")
            or name in SKIPPED
            or any(fnmatch.fnmatch(name, g) for g in self.search_skip)
        )

    def frame_of(self, name: str) -> tuple[int | None, int | None]:
        """``(run, point)`` in a frame's name: ``MapScan_102_1.tiff`` gives ``(102, 1)``; a name
        with one number gives ``(None, point)``, a name that is not a TIFF ``(None, None)``."""
        if not name.lower().endswith(TIFF_SUFFIXES):
            return None, None
        m = self.frame_name.match(name)
        if m:
            return int(m.group("run")), int(m.group("point"))
        last = LAST_NUMBER.search(Path(name).stem)
        return None, (int(last.group(1)) if last else None)

    def run_of_dir(self, name: str) -> int | None:
        m = self.run_dir.search(name)
        return int(m.group("run")) if m else None


# --------------------------------------------------------------- the folders
def _stamp(folder: Path) -> int:
    """The modification time of a folder (it changes when an entry is added), -1 if gone."""
    try:
        return folder.stat().st_mtime_ns
    except OSError:
        return -1


def _entries(folder: Path) -> list[tuple[str, bool]]:
    """``(name, is a folder)`` of every entry, listed again only when the folder changed."""
    try:
        stamp = folder.stat().st_mtime_ns
    except OSError:
        return []
    key = Path(os.path.abspath(folder))
    with _LOCK:
        hit = _LISTINGS.get(key)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    found: list[tuple[str, bool]] = []
    try:
        with os.scandir(folder) as listing:
            for entry in listing:
                try:
                    found.append((entry.name, entry.is_dir()))
                except OSError:
                    found.append((entry.name, False))
    except OSError:
        return []
    found.sort()
    with _LOCK:
        _LISTINGS[key] = (stamp, found)
    return found


def _ancestors(folder: Path, levels: int) -> list[Path]:
    """``folder`` and the ``levels`` folders above it, the nearest first."""
    folder = Path(os.path.abspath(folder))
    return [folder, *list(folder.parents)[:levels]]


def _search(
    start: Path,
    depth: int,
    listing: FolderListing,
    wanted: Callable[[Path], bool],
    skipped: Callable[[str], bool],
    avoid: Path | None = None,
) -> list[Path]:
    """The folders ``wanted`` at the nearest level below ``start`` (itself first) that has any,
    down to ``depth`` levels; ``avoid`` (a subtree searched already) is left out."""
    level = [start]
    for d in range(depth + 1):
        found = [folder for folder in level if wanted(folder)]
        if found or d == depth:
            return found
        below: list[Path] = []
        for folder in level:
            for name, (is_dir, is_link) in sorted(listing.entries(folder).items()):
                # a linked folder is not followed: a link back up the tree would never end
                if is_dir and not is_link and not skipped(name) and folder / name != avoid:
                    below.append(folder / name)
        level = below
    return []


def find_run_folder(
    start: str | Path,
    layout: RunLayout,
    run: int | None = None,
    listing: FolderListing | None = None,
) -> Path | None:
    """The folder of the runs' netCDF files nearest ``start``: ``start`` itself, a folder below
    it, or below one of the ``search_up`` folders above it, the nearest first. With ``run``,
    a folder holding that run's file wins over the others at the same distance."""
    listing = listing if listing is not None else FolderListing()
    glob = layout.run_glob

    def holds_runs(folder: Path) -> bool:
        return bool(listing.glob(folder, glob, folders=False))

    def skipped(name: str) -> bool:
        return layout.is_skipped(name) or layout.is_frames_dir(name)

    below: Path | None = None
    for ancestor in _ancestors(Path(start), layout.search_up):
        found = _search(ancestor, layout.search_depth, listing, holds_runs, skipped, below)
        if found:
            if run is not None:
                name = layout.run_file.format(run=run)
                found.sort(key=lambda f: name not in listing.entries(f))
            return found[0]
        below = ancestor
    return None


def _frames_places(folder: Path, layout: RunLayout) -> list[Path]:
    """The folders holding the runs' frames nearest the netCDF ``folder``: frames folders
    (``python_images``) and folders with frames lying loose; kept once found."""
    key = (Path(os.path.abspath(folder)), layout.run_file)
    with _LOCK:
        hit = _PLACES.get(key)
    if hit is not None:
        return hit
    listing = FolderListing()

    def holds_frames(place: Path) -> bool:
        if layout.is_frames_dir(place.name):
            return True
        return any(
            not is_dir and layout.frame_of(name)[0] is not None
            for name, (is_dir, _) in listing.entries(place).items()
        )

    below: Path | None = None
    found: list[Path] = []
    for ancestor in _ancestors(folder, layout.search_up):
        found = _search(
            ancestor, layout.search_depth, listing, holds_frames, layout.is_skipped, below
        )
        if found:
            break
        below = ancestor
    if found:  # none found is searched again next time: the first frames may come
        with _LOCK:
            _PLACES[key] = found
    return found


def _non_empty(folder: Path) -> bool:
    try:
        with os.scandir(folder) as listing:
            return next(listing, None) is not None
    except OSError:
        return False


@dataclass
class RunIndex:
    """The runs of one netCDF folder: their files, their frames folders, their loose frames."""

    folder: Path
    files: dict[int, Path] = field(default_factory=dict)  # run -> its netCDF file
    folders: dict[int, list[Path]] = field(default_factory=dict)  # run -> its frames folders
    loose: dict[int, dict[int, Path]] = field(default_factory=dict)  # run -> {point: frame}

    def runs(self) -> list[int]:
        """Every run with a netCDF file or frames."""
        return sorted(set(self.files) | set(self.folders) | set(self.loose))

    def has_frames(self, run: int) -> bool:
        return bool(self.loose.get(run)) or any(_non_empty(f) for f in self.folders.get(run, ()))


def run_index(folder: str | Path, layout: RunLayout) -> RunIndex:
    """The runs of the netCDF ``folder`` and where their frames are (no frames folder listed).

    Kept until a folder it read changes: a dataset of a hundred runs opens each with
    the one index (read it, do not change it).
    """
    folder = Path(folder)
    places = _frames_places(folder, layout)
    key = (Path(os.path.abspath(folder)), layout.run_file)
    stamps = tuple(_stamp(f) for f in (folder, *places))
    with _LOCK:
        hit = _INDEXES.get(key)
    if hit is not None and hit[0] == stamps:
        return hit[1]
    index = RunIndex(folder)
    for name, is_dir in _entries(folder):
        run = None if is_dir else layout.run_of_file(name)
        if run is not None:
            index.files[run] = folder / name
    for place in places:
        frames_dir = layout.is_frames_dir(place.name)
        for name, is_dir in _entries(place):
            if is_dir:
                run = layout.run_of_dir(name) if frames_dir else None
                if run is not None:
                    index.folders.setdefault(run, []).append(place / name)
                continue
            run, point = layout.frame_of(name)
            if run is not None and point is not None:
                index.loose.setdefault(run, {}).setdefault(point, place / name)
    with _LOCK:
        _INDEXES[key] = (stamps, index)
    return index


def _folder_frames(folder: Path, run: int, layout: RunLayout) -> dict[int, Path]:
    """``{point: path}`` of the frames of ``run`` in its folder, listed again when it changed."""
    try:
        stamp = folder.stat().st_mtime_ns
    except OSError:
        return {}
    key = (Path(os.path.abspath(folder)), run)
    with _LOCK:
        hit = _FRAMES.get(key)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    found: dict[int, Path] = {}
    try:
        with os.scandir(folder) as listing:
            names = [entry.name for entry in listing]
    except OSError:
        return found
    for name in names:
        named_run, point = layout.frame_of(name)
        if point is not None and named_run in (None, run):  # a frame of another run is not its
            found.setdefault(point, folder / name)
    with _LOCK:
        if len(_FRAMES) >= MAX_FOLDERS:
            _FRAMES.pop(next(iter(_FRAMES)))
        _FRAMES[key] = (stamp, found)
    return found


def run_frames(index: RunIndex, run: int, layout: RunLayout) -> tuple[np.ndarray, list[Path]]:
    """The point numbers of a run's frames, sorted, and their files."""
    found: dict[int, Path] = {}
    for folder in index.folders.get(run, ()):
        for point, path in _folder_frames(folder, run, layout).items():
            found.setdefault(point, path)
    for point, path in index.loose.get(run, {}).items():
        found.setdefault(point, path)
    points = np.array(sorted(found), dtype=np.int64)
    return points, [found[int(p)] for p in points]


def first_point(points: np.ndarray, n_points: int, configured: int | None = None) -> int:
    """The number of a run's first point in its frames' names: 0, or 1 when the names count
    from 1 (no frame 0, and a frame 1 or one numbered as the last point)."""
    if configured is not None:
        return configured
    if points.size and points[0] != 0 and (points[0] == 1 or points[-1] == n_points):
        return 1
    return 0


# ------------------------------------------------------------- the netCDF file
@dataclass
class RunGrid:
    """What a run's netCDF file says: the grid and the counters of every point.

    ``dims`` are the motors, slow first, with their ``coords``; ``order`` the
    flat grid position of every point in the order they were taken (``snake``:
    the fast motor turned back on every other row); ``channels`` the counters
    of every point, in that order.
    """

    dims: tuple[str, ...]
    coords: dict[str, np.ndarray]
    order: np.ndarray
    channels: dict[str, np.ndarray]
    snake: bool = False

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(self.coords[d]) for d in self.dims)

    @property
    def n_points(self) -> int:
        return int(self.order.size)


def _rising(values: np.ndarray) -> bool:
    values = values[np.isfinite(values)]
    return bool(np.all(np.diff(values) >= 0))


def _acquisition_order(
    counter: np.ndarray | None, shape: tuple[int, ...]
) -> tuple[np.ndarray, bool]:
    """The flat grid position of every point in the order taken, and whether it is a serpentine.

    The rows of the netCDF array are in grid order; a counter that rises point by
    point rises along that order for a plain raster, along the serpentine for
    one whose fast motor turns back; when it does neither, it gives the order.
    """
    n = int(np.prod(shape)) if shape else 1
    plain = np.arange(n)
    if counter is None or _rising(counter):
        return plain, False
    if len(shape) > 1:
        snake = np.argsort(raster_order(n, shape, snake=True).reshape(-1))
        if _rising(counter[snake]):
            return snake, True
    log.warning("the points of the run were taken in no raster order: ordered by their counter")
    return np.argsort(counter, kind="stable"), False


def read_run(path: str | Path, layout: RunLayout) -> RunGrid:
    """The grid and counters of a run's netCDF file (the one array of ``to_array()``), read
    once per process (again when the file changes)."""
    path = Path(path)
    st = path.stat()
    key, stamp = (Path(os.path.abspath(path)), layout), (st.st_mtime_ns, st.st_size)
    with _LOCK:
        hit = _GRIDS.get(key)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    grid = _parse_run(path, layout)
    with _LOCK:
        if len(_GRIDS) >= MAX_RUNS:
            _GRIDS.pop(next(iter(_GRIDS)))
        _GRIDS[key] = (stamp, grid)
    return grid


def _parse_run(path: Path, layout: RunLayout) -> RunGrid:
    named = _netcdf3_array(path, layout) or _xarray_array(path, layout)
    if named is None:
        raise ValueError(f"{path.name} holds no data array")
    name, array_dims, values, coords = named
    dims = tuple(d for d in array_dims if d != layout.channel_dim)
    if layout.channel_dim in array_dims:  # the counters last, the motors in their order
        values = np.moveaxis(values, array_dims.index(layout.channel_dim), -1)
        labels = coords.get(layout.channel_dim, np.arange(values.shape[-1]))
    else:
        values, labels = values[..., None], np.array([name])
    values = np.asarray(values, dtype=np.float64)
    shape = values.shape[:-1]
    flat = values.reshape(-1, values.shape[-1])
    k = layout.order_channel
    counter = flat[:, k] if k is not None and k < flat.shape[1] else None
    order, snake = _acquisition_order(counter, shape)
    given = layout.channel_names
    channels = {
        (given[c] if c < len(given) else f"{layout.channel_dim}_{labels[c]}"): flat[order, c]
        for c in range(flat.shape[1])
    }
    grid = {d: coords.get(d, np.arange(n)) for d, n in zip(dims, shape, strict=True)}
    return RunGrid(dims, grid, order, channels, snake)


Named = tuple[str, tuple[str, ...], np.ndarray, dict[str, np.ndarray]]


def _netcdf3_array(path: Path, layout: RunLayout) -> Named | None:
    """``(name, dims, values, coordinates)`` of the data array of a netCDF-3 file, read by
    scipy alone (a quarter of xarray's time); ``None`` for another kind of file."""
    with open(path, "rb") as f:
        if f.read(3) != b"CDF":
            return None
    from scipy.io import netcdf_file

    with netcdf_file(path, "r", mmap=False) as f:  # no map: the arrays are copies, the file closes
        variables = f.variables
        arrays = [n for n, v in variables.items() if tuple(v.dimensions) != (n,)]
        if not arrays:
            return None
        chosen = next(
            (n for n in arrays if layout.channel_dim in variables[n].dimensions), arrays[0]
        )
        array = variables[chosen]
        dims = tuple(str(d) for d in array.dimensions)
        coords = {d: np.array(variables[d].data) for d in dims if d in variables}
        return chosen, dims, np.array(array.data), coords


def _xarray_array(path: Path, layout: RunLayout) -> Named | None:
    """The same through xarray, for a netCDF-4 file (``h5netcdf`` or ``netCDF4`` installed)."""
    import xarray as xr

    try:
        data = xr.load_dataset(path)
    except ValueError as exc:  # no xarray backend reads this kind of file here
        raise OSError(
            f"{path.name} could not be read ({exc}); a netCDF-4 file needs h5netcdf "
            "(pip install h5netcdf)"
        ) from exc
    arrays = [str(n) for n in data.data_vars]
    if not arrays:
        return None
    chosen = next((n for n in arrays if layout.channel_dim in data[n].dims), arrays[0])
    array = data[chosen]
    dims = tuple(str(d) for d in array.dims)
    coords = {d: np.asarray(array[d].values) for d in dims if d in array.coords}
    return chosen, dims, np.asarray(array.values), coords


def run_of(path: str | Path, layout: RunLayout) -> int | None:
    """The run a path names: a run's file, a frame (its name or its run's folder), a run folder."""
    path = Path(path)
    if path.is_dir():
        return layout.run_of_dir(path.name) if layout.is_frames_dir(path.parent.name) else None
    run = layout.run_of_file(path.name)
    if run is None:
        run = layout.frame_of(path.name)[0]
    if run is None and layout.is_frames_dir(path.parent.parent.name):
        run = layout.run_of_dir(path.parent.name)
    return run


def run_folder_of(path: str | Path, layout: RunLayout, run: int | None = None) -> Path:
    """The netCDF folder behind a path: the folder, a run's file, or found from the frames."""
    path = Path(path)
    if path.is_file() and layout.run_of_file(path.name) is not None:
        return path.parent
    start = path if path.is_dir() else path.parent
    found = find_run_folder(start, layout, run)
    if found is None:
        raise FileNotFoundError(
            f"no folder of {layout.run_glob} files in or near {path} "
            f"({layout.search_up} folders up, {layout.search_depth} down)"
        )
    return found


# ------------------------------------------------------------------ the source
@register_engine("nc_tiff")
class NcTiffSource(BaseSource):
    """One run of a Python scan as a frame stack on the grid its netCDF file gives.

    Parameters
    ----------
    spec
        Format spec (``aps_7id_python_2025`` or one derived from it).
    path
        The folder of the netCDF files, a run's file, a frames folder, a run's
        folder of frames or one of its frames: the netCDF folder is found from it.
    scan
        The run number (``run_<n>.nc``).
    detector
        Name from the spec's ``detectors``: its ``shape`` is the frame shape of a
        run without a frame yet, a pixel at its ``invalid_value`` reads as 0.
    energy_keV
        The photon energy (the netCDF file does not hold it).
    """

    name = "nc_tiff"
    dataset_layout = True
    spec: FormatSpec

    def __init__(
        self,
        spec: FormatSpec,
        path: str | Path,
        scan: int,
        detector: str | None = None,
        energy_keV: float | None = None,
    ) -> None:
        super().__init__(spec)
        self.layout = RunLayout.of(spec)
        self.scan = int(scan)
        self.folder = run_folder_of(path, self.layout, self.scan)
        self.detector = detector or next(iter(spec.detectors), "areadetector")
        invalid = spec.detector_info(self.detector).get("invalid_value")
        self._invalid = None if invalid is None else int(invalid)
        self._energy = energy_keV
        self._structure: Structure | None = None
        self._shape: tuple[int, int] | None = None
        self._dtype: np.dtype | None = None
        self._layout: TiffLayout | None = None  # the frames' layout, read without parsing
        self._load()

    def _load(self) -> None:
        index = run_index(self.folder, self.layout)
        runs = index.runs()
        if self.scan not in runs:
            span = f"{runs[0]}..{runs[-1]}" if runs else "none"
            raise KeyError(f"no run {self.scan} in {self.folder}; runs {span}")
        self.last = self.scan == runs[-1]
        self.frame_folders = list(index.folders.get(self.scan, ()))
        self.run_file = index.files.get(self.scan)
        self.grid = read_run(self.run_file, self.layout) if self.run_file is not None else None
        points, frames = run_frames(index, self.scan, self.layout)
        if self.grid is None:  # no netCDF file yet: the frames in the order of their numbers
            self.points, self.frames = points, frames
            self.positions = np.arange(len(frames), dtype=np.int64)
            return
        n = self.grid.n_points
        positions = points - first_point(points, n, self.layout.first_point)
        keep = (positions >= 0) & (positions < n)
        if not keep.all():
            log.warning(
                "%s run %d: %d frames are numbered past its %d points and are left out",
                self.folder.name,
                self.scan,
                int((~keep).sum()),
                n,
            )
        self.points = points[keep]
        self.positions = positions[keep]  # the place of each frame in the order taken
        self.frames = [f for f, k in zip(frames, keep, strict=True) if k]

    # ----------------------------------------------------------- discovery
    @classmethod
    def from_path(
        cls, spec: FormatSpec, path: str | Path, scan: int | None = None, **kwargs: Any
    ) -> NcTiffSource:
        """From the netCDF folder, a run's file, a frames folder or a frame (naming its run)."""
        layout = RunLayout.of(spec)
        path = Path(path)
        if scan is None:
            scan = run_of(path, layout)
        folder = run_folder_of(path, layout, scan)
        if scan is None:
            runs = run_index(folder, layout).runs()
            if not runs:
                raise ValueError(f"{folder} holds no run")
            if len(runs) > 1:
                raise LookupError(
                    f"{folder.name} holds {len(runs)} runs ({runs[0]}..{runs[-1]}): pass "
                    "scan=<run>, or open the folder with bexa.open_dataset to list them"
                )
            scan = runs[0]
        return cls(spec, folder, int(scan), **kwargs)

    @classmethod
    def list_scans(
        cls,
        spec: FormatSpec,
        root: str | Path,
        dataset: str,
        detector: str | None = None,
        require_files: bool = True,
    ) -> list[int]:
        """The run numbers of the netCDF folder behind ``root/dataset``, those with frames by
        default; a run with frames but no netCDF file yet counts."""
        layout = RunLayout.of(spec)
        index = run_index(run_folder_of(_dataset_path(Path(root), dataset), layout), layout)
        runs = index.runs()
        return [r for r in runs if index.has_frames(r)] if require_files else runs

    @classmethod
    def scan_status(
        cls,
        spec: FormatSpec,
        root: str | Path,
        dataset: str,
        scan: int,
        detector: str | None = None,
    ) -> ScanStatus:
        """The status of one run, for a run that cannot be opened."""
        try:
            return cls(spec, _dataset_path(Path(root), dataset), scan, detector).status
        except (OSError, KeyError, IndexError, ValueError):
            return ScanStatus.UNREADABLE

    @classmethod
    def recognises(cls, spec: FormatSpec, path: Path, is_dir: bool, listing: FolderListing) -> bool:
        """True for the netCDF folder, a run's file, and a frames folder, run folder or frame
        with a netCDF folder near it (the fingerprint's ``engine_check``)."""
        layout = RunLayout.of(spec)
        if is_dir:
            if listing.glob(path, layout.run_glob, folders=False):
                return True
            if not (layout.is_frames_dir(path.name) or layout.is_frames_dir(path.parent.name)):
                return False
            return find_run_folder(path, layout, None, listing) is not None
        if layout.run_of_file(path.name) is not None:
            return True
        if not path.name.lower().endswith(TIFF_SUFFIXES):
            return False
        # python_images/<run>/<frame>, a frame loose in python_images, or one loose elsewhere
        framed = layout.is_frames_dir(path.parent.parent.name) or layout.is_frames_dir(
            path.parent.name
        )
        run = run_of(path, layout)
        if not framed and layout.frame_of(path.name)[0] is None:
            return False
        found = find_run_folder(path.parent, layout, run, listing)
        if found is None:
            return False
        # a loose frame is a run's only when that run has its file: a name alone says little
        return framed or (
            run is not None and layout.run_file.format(run=run) in listing.entries(found)
        )

    @property
    def dataset(self) -> str:
        """The netCDF folder's name: what ``Scan.name`` shows."""
        return self.folder.name

    # ------------------------------------------------------------ the frames
    def files(self) -> list[Path]:
        return ([self.run_file] if self.run_file is not None else []) + list(self.frames)

    def _probe(self) -> None:
        if self.frames:
            self._layout = TiffLayout.probe(self.frames[0])
            if self._layout is not None:
                self._shape = self._layout.shape
                self._dtype = self._layout.dtype.newbyteorder("=")
            else:
                self._shape, self._dtype = probe_tiff(self.frames[0])
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
        return None if self._energy is None else float(self._energy)

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
        return read_tiffs(self.frames, ids, y, x, out, self._invalid, self._layout)

    # --------------------------------------------------------- the structure
    def structure(self) -> Structure:
        if self._structure is None:
            self._structure = self._build_structure()
        return self._structure

    def _build_structure(self) -> Structure:
        common: dict[str, Any] = {"energy_keV": self.energy_keV}
        grid = self.grid
        if grid is None or not grid.dims:  # frames before the netCDF file, or one point
            channels = grid.channels if grid is not None else {}
            return Structure.frames_only(
                len(self.frames),
                self.frame_shape,
                title=f"run {self.scan}: frames",
                per_frame={k: v[self.positions] for k, v in channels.items() if v.size},
                **common,
            )
        shape = grid.shape
        flat = grid.order[self.positions]  # the grid position of every frame
        index = np.full(grid.n_points, MISSING, dtype=np.int64)
        index[flat] = np.arange(len(self.frames))
        units = self.spec.motors.get("units", {}) or {}
        title = " x ".join(f"{d} {n}" for d, n in zip(grid.dims, shape, strict=True))
        return Structure(
            motor_dims=grid.dims,
            motor_shape=shape,
            coords=dict(grid.coords),
            frame_index=index.reshape(shape),
            frame_shape=self.frame_shape,
            n_frames=len(self.frames),
            per_frame={name: values[self.positions] for name, values in grid.channels.items()},
            scan_type="raster" if len(shape) > 1 else "line",
            units={d: units[d] for d in grid.dims if d in units},
            title=f"run {self.scan}: {title}" + (" serpentine" if grid.snake else ""),
            planned_shape=shape,
            planned_frames=grid.n_points,
            motor_names=frozenset(grid.dims),
            **common,
        )

    @property
    def status(self) -> ScanStatus:
        """``complete`` with a frame for every point of the netCDF file; ``running`` for the
        last run short of its points while its frames come (one written in the last
        ``running_seconds``); else ``aborted``, or ``empty`` without a frame. A run without
        its netCDF file yet is never complete."""
        if self.grid is not None and len(self.frames) >= self.grid.n_points:
            return ScanStatus.COMPLETE
        if self.last and self._recent():
            return ScanStatus.RUNNING
        return ScanStatus.ABORTED if self.frames else ScanStatus.EMPTY

    def _recent(self) -> bool:
        """A frame (or the run's file) written in the last ``running_seconds``."""
        newest = self.frames[-1] if self.frames else self.run_file
        if newest is None:
            return False
        try:
            return time.time() - newest.stat().st_mtime < self.layout.running_seconds
        except OSError:
            return False

    def point_table(self) -> pd.DataFrame:
        """One row per point of the netCDF file in the order taken, with or without its frame:
        ``point`` (its place in that order), ``frame`` (the id ``read_frames`` reads, -1 when
        there is none), the motor positions, then the counters (``det_0``, ...)."""
        import pandas as pd

        grid = self.grid
        if grid is None:
            return pd.DataFrame({"point": self.positions, "frame": np.arange(len(self.frames))})
        frame = np.full(grid.n_points, -1, dtype=np.int64)
        frame[self.positions] = np.arange(len(self.frames))
        table = {"point": np.arange(grid.n_points), "frame": frame}
        if grid.dims:
            where = np.unravel_index(grid.order, grid.shape)
            table.update({d: grid.coords[d][w] for d, w in zip(grid.dims, where, strict=True)})
        table.update(grid.channels)
        return pd.DataFrame(table)

    def aux_tables(self) -> dict[str, pd.DataFrame]:
        """``points``: :meth:`point_table`."""
        return {"points": self.point_table()}

    def cache_records(self) -> list[dict[str, Any]]:
        """The run's netCDF file, its first and last frames, how many there are and the times
        of its frames folders, which change when a frame is added: not a stat of every frame
        (ten thousand round trips on a network share at every reduction)."""
        from bexa.core.provenance import file_records

        own = {"folder": str(self.folder), "run": self.scan, "frames": len(self.frames)}
        ends = self.frames[:1] + self.frames[-1:] if len(self.frames) > 1 else self.frames
        own_file = [self.run_file] if self.run_file is not None else []
        folders = [{"path": str(f), "mtime_ns": _stamp(f)} for f in self.frame_folders]
        return [own, *file_records([*own_file, *ends]), *folders]

    def refresh(self) -> NcTiffSource:
        """Read the run's folders again (a run still being written)."""
        self._structure = None
        self._shape = self._dtype = self._layout = None
        self._load()
        return self


def _dataset_path(root: Path, dataset: str) -> Path:
    """``root/dataset``, or the folder of ``root/*/dataset.*`` (a file named by its stem)."""
    candidate = root / dataset
    if candidate.exists():
        return candidate
    below = sorted(root.glob(f"*/{dataset}.*"))
    return below[0].parent if below else candidate
