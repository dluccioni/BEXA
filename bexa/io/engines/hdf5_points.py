"""Engine for one-file-per-motor-point layouts (PAL-XFEL, September 2025).

Every motor point of a run has a pandas HDF table with one row per shot
(``type=measurement/run=NNN/scan=NNN/pNNNN.h5``) and a companion raw file
holding the detector frames of those shots (``type=raw/.../pNNNN.h5``). The
shots are the frames of this source and the table rows their metadata: laser
flag, I0, motor readbacks. Column names come from the format spec, so a key
renamed at the next beamtime is a one-line change in the YAML.
"""

from __future__ import annotations

import contextlib
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from bexa._log import get_logger
from bexa.core.registry import register_engine
from bexa.core.structure import Structure
from bexa.io.base import BaseSource
from bexa.io.formats import FormatSpec

if TYPE_CHECKING:
    import pandas as pd

log = get_logger(__name__)

POINT_RE = re.compile(r"p(\d+)\.h5$")
RUN_RE = re.compile(r"run=(\d+)")
SCAN_RE = re.compile(r"scan=(\d+)")
FLAG_KEYS = ("laser_flag", "raw_laser_flag")
AXIS_TOLERANCE = 1e-9


@dataclass
class PointFiles:
    """One motor point: its shot table, its raw frames and where its shots sit in the stack."""

    number: int
    table: Path
    raw: Path
    n_shots: int
    offset: int


def point_number(path: Path) -> int:
    """``pNNNN.h5`` -> NNNN (-1 when the name does not follow the pattern)."""
    m = POINT_RE.search(path.name)
    return int(m.group(1)) if m else -1


def split_run_path(path: Path) -> tuple[Path, int | None, int | None]:
    """``(raw_root, run, scan)`` from any path inside a ``type=...`` tree."""
    parts = path.parts
    for i, part in enumerate(parts):
        if part.startswith("type="):
            run = scan = None
            for later in parts[i + 1 :]:
                run_match = RUN_RE.fullmatch(later)
                scan_match = SCAN_RE.fullmatch(later)
                if run_match and run is None:
                    run = int(run_match.group(1))
                elif scan_match and scan is None:
                    scan = int(scan_match.group(1))
            return Path(*parts[:i]) if i else Path(), run, scan
    return path, None, None


@register_engine("hdf5_points")
class Hdf5PointsSource(BaseSource):
    """Shots of one run as a frame stack, grouped by motor point.

    Parameters
    ----------
    spec
        Format spec (``pal_xfel_points_2025_09`` or one derived from it).
    raw_root
        Folder that holds ``type=measurement`` and ``type=raw``.
    run, scan
        Run and scan numbers of the folder names.
    detector
        Detector name from the spec's ``detectors`` (default: the first one).
    energy_keV
        Photon energy; the point tables do not record it.
    """

    name = "hdf5_points"
    spec: FormatSpec

    def __init__(
        self,
        spec: FormatSpec,
        raw_root: str | Path,
        run: int,
        scan: int = 1,
        detector: str | None = None,
        energy_keV: float | None = None,
    ) -> None:
        super().__init__(spec)
        self.raw_root = Path(raw_root)
        self.run = int(run)
        self.scan = int(scan)
        self.detector = detector or next(iter(spec.detectors), "jungfrau2")
        self._energy_override = energy_keV
        fields = {"raw_root": str(self.raw_root), "run": self.run, "scan": self.scan}
        self.points_dir = Path(spec.path("points", **fields))
        self.frames_dir = Path(spec.path("frames", **fields))
        layout = spec.layout
        self.point_glob = str(layout.get("point_glob", "p*.h5"))
        self.table_key = str(layout.get("table_key", "measurements"))
        self.metadata_key = str(layout.get("metadata_key", "metadata"))
        info = spec.detectors.get(self.detector, {})
        frames_path = info.get("frames_path") or layout.get("frames_path")
        if not frames_path:
            raise KeyError(f"spec {spec.name} gives no frames_path for detector {self.detector!r}")
        self.frames_path = str(frames_path)
        self._local = threading.local()
        self._structure: Structure | None = None
        self._tables: dict[int, pd.DataFrame] = {}
        self._scan_files()

    # ----------------------------------------------------------- discovery
    @classmethod
    def from_path(
        cls, spec: FormatSpec, path: str | Path, scan: int | None = None, **kwargs: Any
    ) -> Hdf5PointsSource:
        """Build a source from a ``scan=NNN`` folder, a ``run=NNN`` folder or the raw root."""
        root, run, found_scan = split_run_path(Path(path))
        run = kwargs.pop("run", run)
        if run is None:
            raise ValueError(f"{path} does not name a run; pass run=<n>")
        if scan is None:
            scan = found_scan if found_scan is not None else 1
        return cls(spec, root, int(run), int(scan), **kwargs)

    @classmethod
    def list_runs(cls, spec: FormatSpec, raw_root: str | Path) -> list[int]:
        """Run numbers that have a measurement folder."""
        points = Path(spec.path("points", raw_root=str(raw_root), run=0, scan=0))
        runs_dir = points.parent.parent  # .../type=measurement
        found = []
        for folder in runs_dir.glob("run=*"):
            m = RUN_RE.fullmatch(folder.name)
            if m and folder.is_dir():
                found.append(int(m.group(1)))
        return sorted(found)

    def _scan_files(self) -> None:
        tables = sorted(self.points_dir.glob(self.point_glob), key=point_number)
        if not tables:
            raise FileNotFoundError(f"no {self.point_glob} files in {self.points_dir}")
        points: list[PointFiles] = []
        offset = 0
        for table in tables:
            raw = self.frames_dir / table.name
            if not raw.exists():
                log.warning("%s has no raw frames file yet; skipped", table.name)
                continue
            with h5py.File(raw, "r") as f:
                if self.frames_path not in f:
                    raise KeyError(f"{raw} has no dataset {self.frames_path!r}")
                ds = f[self.frames_path]
                if not points:
                    self._frame_shape = (int(ds.shape[1]), int(ds.shape[2]))
                    self._dtype = np.dtype(ds.dtype)
                    self._chunks = tuple(ds.chunks) if ds.chunks else None
                n = int(ds.shape[0])
            points.append(PointFiles(point_number(table), table, raw, n, offset))
            offset += n
        if not points:
            raise FileNotFoundError(f"no raw frame files in {self.frames_dir}")
        self.points = points
        self._n_frames = offset

    # ---------------------------------------------------------- properties
    def files(self) -> list[Path]:
        return [p.table for p in self.points] + [p.raw for p in self.points]

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
        return None if self._energy_override is None else float(self._energy_override)

    @property
    def n_points(self) -> int:
        return len(self.points)

    # ------------------------------------------------------------- reading
    def _handle(self, path: Path) -> h5py.File:
        """One open file per thread; HDF5 handles are not thread-safe to share."""
        handles = getattr(self._local, "handles", None)
        if handles is None:
            handles = self._local.handles = {}
        f = handles.get(path)
        if f is None or not f.id.valid:
            f = h5py.File(path, "r")
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
        for p in self.points:
            local = ids[(ids >= p.offset) & (ids < p.offset + p.n_shots)] - p.offset
            if local.size == 0:
                continue
            ds = self._handle(p.raw)[self.frames_path]
            for start, stop in self.contiguous_runs(local):
                count = stop - start
                out[pos : pos + count] = ds[start:stop, y, x]
                pos += count
        return out

    def read_point(self, i: int, y: slice = slice(None), x: slice = slice(None)) -> np.ndarray:
        """All shots of the ``i``-th point, cut to the pixel window."""
        p = self.points[i]
        return self.read_frames(slice(p.offset, p.offset + p.n_shots), y, x)

    def close(self) -> None:
        handles = getattr(self._local, "handles", None) or {}
        for f in handles.values():
            with contextlib.suppress(Exception):  # already closed
                f.close()
        handles.clear()

    def refresh(self) -> Hdf5PointsSource:
        """Re-scan the point files (for a run that is still being written)."""
        self.close()
        self._scan_files()
        self._tables.clear()
        self._structure = None
        return self

    # -------------------------------------------------------------- tables
    def table(self, i: int) -> pd.DataFrame:
        """Per-shot table of the ``i``-th point (cached)."""
        if i not in self._tables:
            import pandas as pd

            self._tables[i] = pd.read_hdf(self.points[i].table, key=self.table_key)
        return self._tables[i]

    def columns(self) -> dict[str, str]:
        """Logical key -> column present in the tables, resolved through the spec aliases."""
        available = list(self.table(0).columns)
        out: dict[str, str] = {}
        for key in self.spec.keys:
            column = self.spec.resolve_key(key, available)
            if column is not None:
                out[key] = column
        return out

    def shot_table(self) -> pd.DataFrame:
        """Every per-shot table in one frame, in frame order, with ``point`` and ``shot``."""
        import pandas as pd

        parts = []
        for i, p in enumerate(self.points):
            t = self.table(i).iloc[: p.n_shots].copy()
            t.insert(0, "shot", np.arange(len(t)))
            t.insert(0, "point", p.number)
            parts.append(t)
        return pd.concat(parts, ignore_index=True)

    def aux_tables(self) -> dict[str, pd.DataFrame]:
        return {"shots": self.shot_table(), "points": self.point_summary()}

    def point_summary(self) -> pd.DataFrame:
        """One row per point: shot counts and the median of every scalar column.

        The medians are what the legacy cube builder stored as the motor
        position of a point.
        """
        import pandas as pd

        columns = self.columns()
        laser = columns.get("laser_flag")
        rows = []
        for i, p in enumerate(self.points):
            t = self.table(i)
            row: dict[str, Any] = {"point": p.number, "n_shots": p.n_shots, "n_rows": len(t)}
            if laser is not None:
                row["n_on"] = int((t[laser] == 1).sum())
                row["n_off"] = int((t[laser] == 0).sum())
            for key, column in columns.items():
                if key in FLAG_KEYS or t[column].dtype.kind not in "fiu":
                    continue
                row[key] = float(t[column].median())
            rows.append(row)
        return pd.DataFrame(rows)

    def scan_axis(self) -> str:
        """First key of the spec's ``axis_priority`` whose per-point median varies."""
        summary = self.point_summary()
        for key in self.spec.axis_priority:
            if key == "index":
                break
            if key not in summary:
                continue
            values = summary[key].to_numpy(dtype=float)
            if np.isfinite(values).all() and np.ptp(values) > AXIS_TOLERANCE:
                return key
        return "index"

    # ----------------------------------------------------------- structure
    def structure(self) -> Structure:
        if self._structure is None:
            self._structure = self._build_structure()
        return self._structure

    def _build_structure(self) -> Structure:
        columns = self.columns()
        n = self._n_frames
        per_frame: dict[str, np.ndarray] = {
            key: np.full(n, np.nan) for key in ("point", "shot", *columns)
        }
        for i, p in enumerate(self.points):
            t = self.table(i)
            m = min(len(t), p.n_shots)
            block = slice(p.offset, p.offset + m)
            per_frame["point"][block] = p.number
            per_frame["shot"][block] = np.arange(m)
            for key, column in columns.items():
                values = t[column].to_numpy()
                if values.dtype.kind in "fiub":
                    per_frame[key][block] = values[:m].astype(float)
        scalars = {
            key: float(np.nanmedian(per_frame[key]))
            for key in self.spec.axis_priority
            if key in per_frame and np.isfinite(per_frame[key]).any()
        }
        return Structure.frames_only(
            n,
            self._frame_shape,
            per_frame=per_frame,
            scalars=scalars,
            scan_type="points",
            energy_keV=self.energy_keV,
        )
