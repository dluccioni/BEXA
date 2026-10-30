"""Engine for scan packages: one scan in one HDF5 file, written by :mod:`bexa.io.package`.

A package holds the frames of a scan, the grid that places them, the per-frame motors and
counters, the positioners, the energy, the status, the detector geometry and the side
tables of the source it was written from. This engine serves all of it back; the frames are
read chunk by chunk from the file (a chunk per frame), straight into the caller's buffer when
the dtypes agree. A folder of packages is a dataset whose scans are numbered by the scan
number each package records (:func:`packages_in`).
"""

from __future__ import annotations

import contextlib
import os
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from bexa._log import get_logger
from bexa.core.registry import register_engine
from bexa.core.status import ScanStatus
from bexa.core.structure import Structure
from bexa.io.base import BaseSource
from bexa.io.cube import _read_attrs
from bexa.io.engines.hdf5_master import collapse_lockstep
from bexa.io.formats import FormatSpec
from bexa.io.h5files import open_h5
from bexa.io.package import FRAMES, PACKAGE_MARKER, PACKAGE_VERSION, _get_array, _key, _name_of

if TYPE_CHECKING:
    import pandas as pd

log = get_logger(__name__)

__all__ = ["PackageSource", "packages_in"]

PACKAGE_SUFFIXES = (".h5", ".hdf5")
# folder -> (modification time, {scan number: (package, frames)}): a folder is listed once
_LISTINGS: dict[Path, tuple[int, dict[int, tuple[Path, int]]]] = {}
_LISTINGS_LOCK = threading.Lock()


def packages_in(folder: str | Path) -> dict[int, Path]:
    """``{scan number: package}`` of the scan packages in ``folder``, by scan number.

    The number is the one each package records; packages without one take the numbers
    after the highest, in name order, and of two packages of one number the first by name
    is kept (with a warning). Other HDF5 files are left out. The listing is kept until the
    folder's modification time changes.
    """
    return {number: path for number, (path, _) in _listing(Path(folder)).items()}


def _listing(folder: Path) -> dict[int, tuple[Path, int]]:
    try:
        stamp = folder.stat().st_mtime_ns
    except OSError:
        return {}
    key = Path(os.path.abspath(folder))
    with _LISTINGS_LOCK:
        hit = _LISTINGS.get(key)
        if hit is not None and hit[0] == stamp:
            return hit[1]
    found: dict[int, tuple[Path, int]] = {}
    unnumbered: list[tuple[Path, int]] = []
    try:
        entries = sorted(os.scandir(folder), key=lambda e: e.name)
    except OSError:
        return {}
    for entry in entries:
        if not entry.name.endswith(PACKAGE_SUFFIXES) or not entry.is_file():
            continue
        try:
            with open_h5(entry.path, retry=False) as f:
                if f.attrs.get(PACKAGE_MARKER) is None:
                    continue  # another HDF5 file
                number = int(f.attrs.get("scan_number", -1))
                frames = int(f.attrs.get("frames", 0))
        except OSError:
            log.debug("%s could not be opened; not listed as a package", entry.path)
            continue
        if number < 0:
            unnumbered.append((Path(entry.path), frames))
        elif number in found:
            log.warning(
                "%s: scan %d has two packages, %s and %s; the first is used",
                folder,
                number,
                found[number][0].name,
                entry.name,
            )
        else:
            found[number] = (Path(entry.path), frames)
    after = max(found, default=0) + 1
    for offset, item in enumerate(unnumbered):
        found[after + offset] = item
    found = dict(sorted(found.items()))
    with _LISTINGS_LOCK:
        _LISTINGS[key] = (stamp, found)
    return found


@register_engine("package")
class PackageSource(BaseSource):
    """One scan package as a source of frames.

    Parameters
    ----------
    spec
        Format spec (``bexa_package``).
    path
        The package file.
    detector
        Not used: a package holds the frames of one detector, named by :attr:`detector`.
    energy_keV
        Overrides the energy the package records.
    motors
        Per-frame channels to take as the dims instead of the recorded grid.
    """

    name = "package"
    dataset_layout = True  # a folder of packages lists as a dataset of numbered scans
    spec: FormatSpec

    def __init__(
        self,
        spec: FormatSpec,
        path: str | Path,
        detector: str | None = None,
        energy_keV: float | None = None,
        motors: Sequence[str] | None = None,
    ) -> None:
        super().__init__(spec)
        self.path = Path(path)
        self._energy_override = energy_keV
        self._motors = [str(m) for m in motors] if motors is not None else None
        self._local = threading.local()
        self._handles: list[h5py.File] = []
        self._handles_lock = threading.Lock()
        self._structure: Structure | None = None
        self._tables: dict[str, pd.DataFrame] | None = None
        with open_h5(self.path) as f:
            self._read_header(f)

    def _read_header(self, f: h5py.File) -> None:
        root = _read_attrs(f)
        if root.get(PACKAGE_MARKER) != PACKAGE_VERSION:
            raise ValueError(
                f"{self.path.name} is not a bexa scan package of layout {PACKAGE_VERSION} "
                f"({PACKAGE_MARKER} = {root.get(PACKAGE_MARKER)!r})"
            )
        number = int(root.get("scan_number", -1))
        self.scan: int | None = number if number >= 0 else None
        self.dataset = str(root.get("name") or self.path.stem)  # what Scan.name shows
        frames = f[FRAMES]
        self._n_frames = int(frames.shape[0])
        self._frame_shape = (int(frames.shape[1]), int(frames.shape[2]))
        self._dtype = np.dtype(frames.dtype)
        grid = f["entry/structure"]
        self._grid = _read_attrs(grid)
        self._frame_index = np.asarray(grid["frame_index"][()], dtype=np.int64)
        self._coords = {_name_of(n): np.asarray(_get_array(n)) for n in grid["coords"].values()}
        self._per_frame = {
            _name_of(n): np.asarray(_get_array(n)) for n in f["entry/measurement"].values()
        }
        self._scalars = {
            _name_of(n): _scalar(_get_array(n)) for n in f["entry/instrument/positioners"].values()
        }
        self._status = _read_attrs(f["entry"]).get("status")
        detector = _read_attrs(f["entry/instrument/detector"])
        self.detector: str | None = detector.get("name")
        # the geometry the scan had, for bexa.open: pixel size, distance, effective pixel, ...
        self.recorded_geometry: dict[str, Any] = dict(detector.get("geometry") or {})
        self.pixel_size_um = self.recorded_geometry.get("pixel_size_um")
        self.origin: dict[str, Any] = _read_attrs(f["entry/source"])

    # ----------------------------------------------------------- discovery
    @classmethod
    def from_path(
        cls, spec: FormatSpec, path: str | Path, scan: int | None = None, **kwargs: Any
    ) -> PackageSource:
        """From a package, or from a folder of packages and a scan number."""
        path = Path(path)
        if not path.is_dir():
            source = cls(spec, path, **kwargs)
            if scan is not None and source.scan is not None and int(scan) != source.scan:
                raise KeyError(f"{path.name} holds scan {source.scan}, not scan {scan}")
            return source
        found = packages_in(path)
        if not found:
            raise FileNotFoundError(f"no scan package in {path}")
        if scan is None:
            if len(found) == 1:
                return cls(spec, next(iter(found.values())), **kwargs)
            numbers = sorted(found)
            raise LookupError(
                f"{path.name} holds {len(found)} scan packages ({numbers[0]}..{numbers[-1]}): "
                "pass scan=<n>, or open the folder with bexa.open_dataset to list them"
            )
        if int(scan) not in found:
            raise KeyError(f"no package of scan {scan} in {path}; scans {sorted(found)}")
        return cls(spec, found[int(scan)], **kwargs)

    @classmethod
    def list_scans(
        cls,
        spec: FormatSpec,
        root: str | Path,
        dataset: str,
        detector: str | None = None,
        require_files: bool = True,
    ) -> list[int]:
        """The scan numbers of the packages in ``root/dataset``, those with frames by default."""
        root = Path(root)
        folder = root / dataset
        if not folder.is_dir():  # bexa.list_scans names a file <root>/<folder>/<stem>.h5 so
            below = sorted(root.glob(f"*/{dataset}.h5"))
            folder = below[0].parent if below else folder
        found = _listing(folder)
        return [n for n, (_, frames) in found.items() if frames > 0 or not require_files]

    # ------------------------------------------------------------ the frames
    def files(self) -> list[Path]:
        return [self.path]

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
    def energy_keV(self) -> float | None:
        if self._energy_override is not None:
            return float(self._energy_override)
        value = self._grid.get("energy_keV")
        return float(value) if value is not None else None

    def _handle(self) -> h5py.File:
        f = getattr(self._local, "file", None)
        if f is None or not f.id.valid:
            f = open_h5(self.path)
            self._local.file = f
            with self._handles_lock:
                self._handles.append(f)
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
        target = np.dtype(dtype) if dtype is not None else self._dtype
        if out is None:
            out = np.empty((ids.size, ny, nx), dtype=target)
        frames = self._handle()[FRAMES]
        direct = out.dtype == self._dtype and out.flags.c_contiguous  # no copy, no cast
        pos = 0
        for start, stop in self.contiguous_runs(ids):
            count = stop - start
            if direct:
                frames.read_direct(out, np.s_[start:stop, y, x], np.s_[pos : pos + count])
            else:
                out[pos : pos + count] = frames[start:stop, y, x]
            pos += count
        return out

    # --------------------------------------------------------- the structure
    def structure(self) -> Structure:
        if self._structure is None:
            self._structure = self._build_structure()
        return self._structure

    def _build_structure(self) -> Structure:
        grid = self._grid
        common: dict[str, Any] = {
            "per_frame": dict(self._per_frame),
            "scalars": dict(self._scalars),
            "scan_type": str(grid.get("scan_type", "grid")),
            "units": dict(grid.get("units") or {}),
            "energy_keV": self.energy_keV,
            "title": str(grid.get("title", "")),
            "motor_names": frozenset(grid.get("motor_names") or ()),
        }
        if self._motors is None:  # the grid as written
            dims = tuple(str(d) for d in grid["motor_dims"])
            planned = grid.get("planned_shape")
            return Structure(
                motor_dims=dims,
                motor_shape=tuple(int(n) for n in grid["motor_shape"]),
                coords={d: self._coords[d] for d in dims},
                frame_index=self._frame_index,
                frame_shape=self._frame_shape,
                n_frames=self._n_frames,
                planned_shape=None if planned is None else tuple(int(n) for n in planned),
                planned_frames=grid.get("planned_frames"),
                **common,
            )
        unknown = [m for m in self._motors if m not in self._per_frame]
        if unknown:
            raise ValueError(
                f"{self.path.name} records no per-frame channel {unknown}; "
                f"choose from {sorted(self._per_frame)}"
            )
        if self._n_frames == 0:
            raise ValueError(
                f"{self.path.name} holds no frame to place on a grid of {self._motors}"
            )
        order, _ = collapse_lockstep(self._motors, self._per_frame)
        chosen = Structure.from_per_frame({m: self._per_frame[m] for m in order}, self._frame_shape)
        common["units"] = {m: u for m, u in common["units"].items() if m in order}
        return Structure(
            motor_dims=chosen.motor_dims,
            motor_shape=chosen.motor_shape,
            coords=chosen.coords,
            frame_index=chosen.frame_index,
            frame_shape=self._frame_shape,
            n_frames=self._n_frames,
            **common,
        )

    @property
    def status(self) -> ScanStatus:
        """The status the scan had when it was written; computed from the grid if not recorded."""
        if self._status in {s.value for s in ScanStatus}:
            return ScanStatus(self._status)
        return super().status

    def with_motors(self, motors: Sequence[str]) -> PackageSource:
        return PackageSource(
            self.spec, self.path, energy_keV=self._energy_override, motors=list(motors)
        )

    # ------------------------------------------------------------- tables
    def aux_tables(self) -> dict[str, pd.DataFrame]:
        """The side tables the scan was written with (an APS scan: ``points``, ``header``)."""
        if self._tables is None:
            import pandas as pd

            tables: dict[str, pd.DataFrame] = {}
            with open_h5(self.path) as f:
                for group in f["entry/tables"].values():
                    columns = [str(c) for c in _read_attrs(group).get("columns", [])]
                    data = {c: _get_array(group[_key(c)]) for c in columns}
                    tables[_name_of(group)] = pd.DataFrame(data, columns=columns)
            self._tables = tables
        return {name: table.copy() for name, table in self._tables.items()}

    def point_table(self) -> pd.DataFrame:
        """One row per point of the scan: the ``points`` table it was written with, else per frame.

        A package of an APS scan has every point SPEC recorded, frames or not (see
        :meth:`bexa.io.engines.spec_tiff.SpecTiffSource.point_table`); other packages give one
        row per frame of the per-frame channels, with the ``frame`` id.
        """
        import pandas as pd

        tables = self.aux_tables()
        if "points" in tables:
            return tables["points"]
        table = pd.DataFrame(dict(self._per_frame))
        table.insert(0, "frame", np.arange(self._n_frames))
        return table

    # ---------------------------------------------------------- lifecycle
    def refresh(self) -> PackageSource:
        """Read the package again (a package does not change once written)."""
        self.close()
        self._structure = None
        self._tables = None
        with open_h5(self.path) as f:
            self._read_header(f)
        return self

    def close(self) -> None:
        with self._handles_lock:
            handles, self._handles = self._handles, []
        for f in handles:
            with contextlib.suppress(Exception):  # closed already
                f.close()
        self._local = threading.local()


def _scalar(value: Any) -> Any:
    """A 0-d array as a Python number or text."""
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else array
