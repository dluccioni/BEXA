"""Engine for LCLS smalldata HDF5 files (one file per run, one row per event).

A smalldata file holds per-event scalars (``lightStatus/laser``, ``ipm4/sum``,
``scan/...``) and, when the producer stored them, per-event detector images
or ROIs (``<detector>/ROI_0_area``), plus run-level ``Sums/`` images. The
per-event image dataset is the frame stack of this source; the scalars are
its per-frame metadata, resolved through the format spec's key aliases.
"""

from __future__ import annotations

import contextlib
import threading
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from bexa._log import get_logger
from bexa.core.registry import register_engine
from bexa.core.structure import Structure
from bexa.io.base import BaseSource
from bexa.io.formats import FormatSpec

log = get_logger(__name__)

__all__ = ["SmalldataSource"]


@register_engine("smalldata")
class SmalldataSource(BaseSource):
    """Per-event images and scalars of one smalldata run file.

    Parameters
    ----------
    spec
        Format spec (``lcls_smalldata``).
    path
        The ``<experiment>_Run<NNNN>.h5`` file.
    detector
        Detector name from the spec's ``detectors`` (its ``frames_path`` is the
        per-event image dataset); default: the first one present in the file.
    """

    name = "smalldata"
    spec: FormatSpec

    def __init__(
        self,
        spec: FormatSpec,
        path: str | Path,
        detector: str | None = None,
        energy_keV: float | None = None,
    ) -> None:
        super().__init__(spec)
        self.path = Path(path)
        self._energy_override = energy_keV
        self._local = threading.local()
        self._structure: Structure | None = None
        with h5py.File(self.path, "r") as f:
            self.detector, self.frames_path = self._pick_detector(f, detector)
            ds = f[self.frames_path]
            if ds.ndim != 3:
                raise ValueError(f"{self.frames_path} has {ds.ndim} dims; expected (event, y, x)")
            self._n_frames = int(ds.shape[0])
            self._frame_shape = (int(ds.shape[1]), int(ds.shape[2]))
            self._dtype = np.dtype(ds.dtype)
            self._chunks = tuple(ds.chunks) if ds.chunks else None
            self.columns = self._resolve_columns(f)

    def _pick_detector(self, f: h5py.File, detector: str | None) -> tuple[str, str]:
        candidates = self.spec.detectors
        if detector is not None:
            info = candidates.get(detector, {})
            frames_path = info.get("frames_path", f"{detector}/ROI_0_area")
            if frames_path not in f:
                raise KeyError(f"{self.path} has no dataset {frames_path!r}")
            return detector, str(frames_path)
        for name, info in candidates.items():
            frames_path = info.get("frames_path")
            if frames_path and frames_path in f:
                return name, str(frames_path)
        raise KeyError(
            f"{self.path} holds none of the per-event image datasets of {self.spec.name}"
        )

    def _resolve_columns(self, f: h5py.File) -> dict[str, str]:
        available: list[str] = []

        def visit(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset) and obj.ndim == 1 and obj.shape[0] == self._n_frames:
                available.append(name)

        f.visititems(visit)
        out: dict[str, str] = {}
        for key in self.spec.keys:
            column = self.spec.resolve_key(key, available)
            if column is not None:
                out[key] = column
        return out

    @classmethod
    def from_path(
        cls, spec: FormatSpec, path: str | Path, scan: int | None = None, **kwargs: Any
    ) -> SmalldataSource:
        path = Path(path)
        if path.is_dir():
            files = sorted(path.glob("*.h5"))
            if not files:
                raise FileNotFoundError(f"no .h5 file in {path}")
            path = files[0]
        return cls(spec, path, **kwargs)

    # ---------------------------------------------------------- properties
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
    def chunk_frames(self) -> int:
        return int(self._chunks[0]) if self._chunks else 1

    @property
    def energy_keV(self) -> float | None:
        if self._energy_override is not None:
            return float(self._energy_override)
        column = self.columns.get("photon_energy")
        if column is None:
            return None
        with h5py.File(self.path, "r") as f:
            values = np.asarray(f[column][()], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return None
        median = float(np.median(finite))
        return median / 1e3 if median > 1e3 else median  # eV in the file, keV here

    # ------------------------------------------------------------- reading
    def _handle(self) -> h5py.File:
        f = getattr(self._local, "handle", None)
        if f is None or not f.id.valid:
            f = self._local.handle = h5py.File(self.path, "r")
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
        ds = self._handle()[self.frames_path]
        pos = 0
        for start, stop in self.contiguous_runs(ids):
            count = stop - start
            out[pos : pos + count] = ds[start:stop, y, x]
            pos += count
        return out

    def close(self) -> None:
        f = getattr(self._local, "handle", None)
        if f is not None:
            with contextlib.suppress(Exception):
                f.close()
            self._local.handle = None

    def sums(self) -> dict[str, np.ndarray]:
        """The run-level summed images under ``Sums/``."""
        out: dict[str, np.ndarray] = {}
        with h5py.File(self.path, "r") as f:
            if "Sums" in f:
                for name, node in f["Sums"].items():
                    if isinstance(node, h5py.Dataset):
                        out[name] = np.asarray(node[()])
        return out

    def scalars_table(self) -> Any:
        """Every resolved per-event scalar as a pandas DataFrame (one row per event)."""
        import pandas as pd

        with h5py.File(self.path, "r") as f:
            data = {key: np.asarray(f[column][()]) for key, column in self.columns.items()}
        return pd.DataFrame(data)

    # ----------------------------------------------------------- structure
    def structure(self) -> Structure:
        if self._structure is None:
            with h5py.File(self.path, "r") as f:
                per_frame = {}
                for key, column in self.columns.items():
                    values = np.asarray(f[column][()])
                    if values.dtype.kind in "fiub":
                        per_frame[key] = values.astype(float)
            axis = self.spec.resolve_key("scanvar", list(per_frame)) if per_frame else None
            scan_axis = next(
                (k for k in self.spec.axis_priority if k in per_frame and np.ptp(per_frame[k]) > 0),
                axis,
            )
            self._structure = Structure.frames_only(
                self._n_frames,
                self._frame_shape,
                per_frame=per_frame,
                scan_type="events",
                energy_keV=self.energy_keV,
            )
            if scan_axis:
                self._structure.scalars["scan_axis"] = float("nan")
                self._structure.units["scan_axis"] = scan_axis
        return self._structure
