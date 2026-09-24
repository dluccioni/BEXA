"""One call for the common figures: ``plot(obj)`` picks the function from dims and attrs.

- a 1-D array is a curve, a 2-D array an image, a centre-of-mass or width map
  when its name or attrs say so (``com_mu``, ``width_chi``, ``argmax_mu``);
- a 3-D array is a grid of tiles along its first dim, a 4-D or larger one the
  projections panel;
- a reduce result (``xarray.Dataset``) is a panel with one map per 2-D
  variable; a cube with ``frames(laser, ...)`` is the laser on/off figure.

``kind`` overrides the guess: ``image``, ``com``, ``width``, ``curve``,
``rocking``, ``tiles``, ``browse``, ``projections``, ``volume``, ``on_off``
or ``grid``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import xarray as xr

from bexa.viz import curves, images, interactive, maps, volume

__all__ = ["KINDS", "guess_kind", "plot"]

KINDS = (
    "image",
    "com",
    "width",
    "curve",
    "rocking",
    "tiles",
    "browse",
    "projections",
    "volume",
    "on_off",
    "grid",
)
_MAP_PREFIXES = ("com_", "argmax_")
_WIDTH_PREFIXES = ("width_", "skew_", "kurtosis_")


def guess_kind(obj: Any) -> str:
    """The plot kind :func:`plot` would choose for ``obj``."""
    if isinstance(obj, xr.Dataset):
        if "frames" in obj.data_vars and "laser" in obj.dims:
            return "on_off"
        return "grid"
    ndim = int(np.ndim(obj)) if not hasattr(obj, "ndim") else int(obj.ndim)
    name = str(getattr(obj, "name", "") or "")
    attrs = getattr(obj, "attrs", {}) or {}
    if ndim <= 1:
        return "curve"
    if ndim == 2:
        if name.startswith(_WIDTH_PREFIXES):
            return "width"
        if name.startswith(_MAP_PREFIXES) or "motor" in attrs:
            return "com"
        return "image"
    if ndim == 3:
        return "tiles"
    return "projections"


def _as_dataarray(obj: Any) -> Any:
    """Give plain arrays of three or more dims the ``(..., y, x)`` names the panels expect."""
    if isinstance(obj, xr.DataArray) or np.ndim(obj) < 3:
        return obj
    data = np.asarray(obj)
    dims = [f"axis{i}" for i in range(data.ndim - 2)] + ["y", "x"]
    return xr.DataArray(data, dims=dims)


def plot(obj: Any, kind: str | None = None, **kwargs: Any) -> Any:
    """Plot ``obj`` with the function that fits it; returns what that function returns."""
    kind = kind or guess_kind(obj)
    if kind == "image":
        return images.show(obj, **kwargs)
    if kind == "com":
        return maps.com_map(obj, **kwargs)
    if kind == "width":
        return maps.com_map(obj, **{"center": "median", "cmap": "viridis", **kwargs})
    if kind == "curve":
        return curves.curve(obj, **kwargs)
    if kind == "rocking":
        return curves.rocking_curves(obj, **kwargs)
    if kind == "tiles":
        return images.tiles(obj, **kwargs)
    if kind == "browse":
        return interactive.browse(_as_dataarray(obj), **kwargs)
    if kind == "projections":
        return volume.projections_panel(_as_dataarray(obj), **kwargs)
    if kind == "volume":
        return interactive.browse_volume(_as_dataarray(obj), **kwargs)
    if kind == "on_off":
        return curves.on_off_diff(obj, **kwargs)
    if kind == "grid":
        return grid(obj, **kwargs)
    raise ValueError(f"unknown kind {kind!r}; choose from {KINDS}")


def grid(
    ds: xr.Dataset,
    per_row: int = 3,
    max_panels: int = 9,
    figsize: tuple[float, float] | None = None,
    names: list[str] | None = None,
) -> tuple[Any, np.ndarray]:
    """One panel per 2-D variable of a reduce result: maps as maps, the rest as images."""
    import matplotlib.pyplot as plt

    chosen = names or [str(k) for k in ds.data_vars if ds[k].ndim == 2][:max_panels]
    if not chosen:
        raise ValueError("the dataset has no 2-D variable to show")
    rows = math.ceil(len(chosen) / per_row)
    fig, axes = plt.subplots(
        rows, per_row, figsize=figsize or (4.4 * per_row, 3.8 * rows), squeeze=False
    )
    for ax, name in zip(axes.ravel(), chosen, strict=False):
        plot(ds[name], ax=ax, title=name)
    for ax in axes.ravel()[len(chosen) :]:
        ax.axis("off")
    return fig, axes
