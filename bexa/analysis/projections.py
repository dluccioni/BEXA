"""Projections, integrated maps and reciprocal-lattice point clouds from volumes in memory."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr

from bexa.core.backend import to_host
from bexa.core.roi import ROI

__all__ = ["integrated_map", "project", "rlp_points"]


def project(volume: Any, dims: str | Sequence[str] | Sequence[int], method: str = "sum") -> Any:
    """Reduce a volume over ``dims`` (names for a DataArray, axes for an array)."""
    if isinstance(volume, xr.DataArray):
        reducer = getattr(volume, method)
        return reducer([dims] if isinstance(dims, str) else list(dims))
    axes = tuple(int(d) for d in ([dims] if isinstance(dims, int) else dims))  # type: ignore[arg-type]
    return getattr(volume, method)(axis=axes)


def integrated_map(volume: Any, roi: ROI | None = None, method: str = "sum") -> Any:
    """Intensity inside the pixel ROI at every motor point (the last two axes are y, x)."""
    if roi is not None:
        ys, xs = roi.pixel_slices(volume.shape[-2:])
        volume = volume[..., ys, xs]
    if isinstance(volume, xr.DataArray):
        return getattr(volume, method)(["y", "x"])
    return getattr(volume, method)(axis=(-2, -1))


def rlp_points(
    structure: Any,
    intensity: Any,
    geometry: Any,
    crystal: Any = None,
    space: str = "angular",
    hkl_center: tuple[int, int, int] | None = None,
    threshold: float | None = None,
    **fixed_angles_deg: float,
) -> dict[str, np.ndarray]:
    """Coordinates and intensity of every motor point for a 3-D reciprocal-space scatter.

    ``intensity`` is the integrated map on the motor grid; the coordinates come
    from :func:`bexa.geometry.transforms.build_coordinate_grids` in angular,
    Q or hkl space. Points below ``threshold`` are dropped.
    """
    from bexa.geometry.transforms import build_coordinate_grids

    grids = build_coordinate_grids(
        structure, geometry, crystal=crystal, space=space, hkl_center=hkl_center, **fixed_angles_deg
    )
    values = np.asarray(to_host(intensity), dtype=float).reshape(-1)
    keep = np.isfinite(values)
    if threshold is not None:
        keep &= values >= threshold
    out = {name: np.asarray(grid, dtype=float).reshape(-1)[keep] for name, grid in grids.items()}
    out["intensity"] = values[keep]
    return out
