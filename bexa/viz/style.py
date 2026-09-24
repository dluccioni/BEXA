"""Themes and small helpers shared by the plotting modules."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from bexa.core.backend import to_host

# The rcParams block used by the lab's XFEL scripts, kept as the "paper" theme.
PAPER: dict[str, Any] = {
    "font.size": 10,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 10,
    "ytick.major.size": 10,
    "xtick.minor.size": 5,
    "ytick.minor.size": 5,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "image.cmap": "viridis",
}

SCREEN: dict[str, Any] = {
    "font.size": 10,
    "figure.dpi": 100,
    "savefig.dpi": 150,
    "image.cmap": "viridis",
    "figure.constrained_layout.use": True,
}

THEMES = {"paper": PAPER, "screen": SCREEN}


def use(theme: str = "screen") -> None:
    """Apply a theme (``"screen"`` or ``"paper"``) to matplotlib."""
    plt.rcParams.update(THEMES[theme])


def as_array(data: Any) -> np.ndarray:
    """Plain numpy array from a DataArray, cupy array or array-like."""
    if hasattr(data, "values") and hasattr(data, "dims"):
        data = data.values
    return np.asarray(to_host(data))


def percentile_limits(data: Any, low: float = 1.0, high: float = 99.0) -> tuple[float, float]:
    """Colour limits from percentiles, ignoring NaN and infinities."""
    arr = as_array(data)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(finite, [low, high])
    if vmax <= vmin:
        vmax = vmin + 1e-12
    return float(vmin), float(vmax)


def resolve_clim(data: Any, clim: Any = ("p", 1, 99)) -> tuple[float, float] | tuple[None, None]:
    """Turn ``("p", lo, hi)``, ``(vmin, vmax)`` or ``None`` into colour limits."""
    if clim is None:
        return None, None
    if isinstance(clim, (tuple, list)) and len(clim) == 3 and clim[0] == "p":
        return percentile_limits(data, clim[1], clim[2])
    if isinstance(clim, str) and clim.startswith("p"):
        lo, hi = clim[1:].split("-")
        return percentile_limits(data, float(lo), float(hi))
    return float(clim[0]), float(clim[1])


def new_axes(
    ax: Any = None, figsize: tuple[float, float] | None = None, **kwargs: Any
) -> tuple[Any, Any]:
    """``(fig, ax)``: reuse ``ax`` when given, else make a new figure."""
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    return fig, ax


def dim_label(da: Any, dim: str) -> str:
    """Axis label with units from the DataArray's coordinate attrs when present."""
    if hasattr(da, "coords") and dim in da.coords:
        unit = da.coords[dim].attrs.get("units", "")
        return f"{dim} ({unit})" if unit else dim
    return dim


def pixel_size_nm(da: Any) -> float | None:
    """Effective pixel size recorded in the attrs, if any."""
    if hasattr(da, "attrs"):
        value = da.attrs.get("effective_pixel_nm")
        if value:
            return float(value)
    return None
