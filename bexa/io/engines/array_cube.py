"""Engine for already-reduced cubes: legacy ``runN.h5``, LCLS triples, bexa files, ``.npy``.

The cube is loaded through :mod:`bexa.io.cube` and exposed as a frame stack
whose leading dims (``laser``, the scan axis, ...) form the grid, so previews,
reductions and the plots work on reduced data exactly as on raw scans.
``dataset()`` returns the underlying xarray object for cube-level analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from bexa._log import get_logger
from bexa.core.registry import register_engine
from bexa.core.structure import Structure
from bexa.io.base import BaseSource
from bexa.io.formats import FormatSpec

log = get_logger(__name__)

LASER_LEVELS = {"off": 0.0, "on": 1.0}


def _numeric_coord(values: np.ndarray) -> np.ndarray:
    """Grid coordinates must be numbers; ``laser`` labels map to 0/1, other labels to indices."""
    if values.dtype.kind in "iufb":
        return values.astype(float)
    labels = [str(v) for v in values]
    if all(v in LASER_LEVELS for v in labels):
        return np.array([LASER_LEVELS[v] for v in labels])
    return np.arange(len(values), dtype=float)


@register_engine("array_cube")
class ArrayCubeSource(BaseSource):
    """A reduced cube as a source of frames.

    Parameters
    ----------
    spec
        Format spec (``bexa_cube_legacy``, ``lcls_xcs_cube`` or ``bexa_cube``).
    path
        Cube file, LCLS folder or one of its files, or a plain ``.npy`` stack.
    run
        Run number for LCLS folders holding several runs.
    """

    name = "array_cube"

    def __init__(
        self,
        spec: FormatSpec | None,
        path: str | Path,
        run: int | None = None,
        energy_keV: float | None = None,
    ) -> None:
        super().__init__(spec)
        self.path = Path(path)
        self.run = run
        self._energy_override = energy_keV
        self._structure: Structure | None = None
        self._load()

    @classmethod
    def from_path(
        cls, spec: FormatSpec, path: str | Path, scan: int | None = None, **kwargs: Any
    ) -> ArrayCubeSource:
        kwargs.pop("detector", None)
        return cls(spec, path, **kwargs)

    # ------------------------------------------------------------- loading
    def _load(self) -> None:
        from bexa.io.cube import load, load_lcls_cube

        p = self.path
        if p.suffix == ".npy" and not p.stem.startswith("Run"):
            data = np.load(p, mmap_mode="r")
            if data.ndim < 3:
                raise ValueError(f"{p} holds a {data.ndim}-D array; need at least 3 dims")
            dims = (*(f"dim_{i}" for i in range(data.ndim - 2)), "y", "x")
            self._dataset = xr.Dataset({"frames": xr.DataArray(data, dims=dims)})
        elif (p.is_dir() and p.suffix != ".zarr") or p.suffix in (".npy", ".csv"):
            self._dataset = load_lcls_cube(p, run=self.run)
        else:
            loaded = load(p, squeeze_single=False)
            self._dataset = loaded if isinstance(loaded, xr.Dataset) else loaded.to_dataset()
        self.frames_name = self._pick_frames_variable(self._dataset)
        frames = self._dataset[self.frames_name]
        self.lead_dims = tuple(str(d) for d in frames.dims[:-2])
        self.pixel_dims = tuple(str(d) for d in frames.dims[-2:])
        h, w = (int(frames.shape[-2]), int(frames.shape[-1]))
        self._frame_shape = (h, w)
        self._frames = np.asarray(frames.data).reshape(-1, h, w)
        self._dtype = np.dtype(self._frames.dtype)
        self._n_frames = int(self._frames.shape[0])

    @staticmethod
    def _pick_frames_variable(ds: xr.Dataset) -> str:
        if "frames" in ds.data_vars:
            return "frames"
        for name, var in ds.data_vars.items():
            if var.ndim >= 3:
                return str(name)
        raise ValueError("the cube holds no variable with at least 3 dims")

    def dataset(self) -> xr.Dataset:
        """The cube as loaded (``frames`` plus the per-point series and coordinates)."""
        return self._dataset

    # ---------------------------------------------------------- properties
    def files(self) -> list[Path]:
        if self.path.is_dir() and self.path.suffix != ".zarr":
            run = self.run
            if run is None:
                candidates = sorted(self.path.glob("Run*_onStk.npy"))
                run = int(candidates[0].stem[3:7]) if candidates else 0
            return [
                self.path / f"Run{run:04d}_{s}" for s in ("onStk.npy", "offStk.npy", "stats.csv")
            ]
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
        value = self._dataset.attrs.get("energy_keV")
        return float(value) if value is not None else None

    # ------------------------------------------------------------- reading
    def read_frames(
        self,
        index: np.ndarray | slice,
        y: slice = slice(None),
        x: slice = slice(None),
        dtype: Any = None,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        ids = self.normalise_index(index, self._n_frames)
        block = self._frames[ids][:, y, x]
        if out is None:
            return block.astype(dtype) if dtype is not None else np.array(block)
        out[...] = block
        return out

    # ----------------------------------------------------------- structure
    def structure(self) -> Structure:
        if self._structure is None:
            self._structure = self._build_structure()
        return self._structure

    def _build_structure(self) -> Structure:
        ds = self._dataset
        frames = ds[self.frames_name]
        coords: dict[str, np.ndarray] = {}
        for d in self.lead_dims:
            values = (
                np.asarray(frames.coords[d].values)
                if d in frames.coords
                else np.arange(frames.sizes[d])
            )
            coords[d] = _numeric_coord(values)
        structure = Structure.from_grid(
            self.lead_dims,
            coords,
            self._frame_shape,
            scan_type="cube",
            energy_keV=self.energy_keV,
        )
        # per-point series (I0, signal, scan variable, ...) broadcast to every frame
        grid_shape = tuple(frames.sizes[d] for d in self.lead_dims)
        for name, var in {**ds.data_vars, **ds.coords}.items():
            if name == self.frames_name or not var.dims:
                continue
            if not set(var.dims) <= set(self.lead_dims) or var.dtype.kind not in "iufb":
                continue
            expanded = var.transpose(*[d for d in self.lead_dims if d in var.dims])
            full = np.broadcast_to(
                expanded.values.reshape(
                    [frames.sizes[d] if d in var.dims else 1 for d in self.lead_dims]
                ),
                grid_shape,
            )
            structure.per_frame[str(name)] = np.asarray(full, dtype=float).reshape(-1)
        return structure
