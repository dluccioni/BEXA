"""Drawing objects into a matplotlib figure: images with their coordinates, curves with errors.

Pure matplotlib, no widgets: the View tab calls these with the options of its
controls, and the tests call them directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

__all__ = [
    "CMAPS",
    "CurveOptions",
    "ImageOptions",
    "draw_curves",
    "draw_image",
    "draw_rgb",
    "extent_of",
]

CMAPS = (
    "viridis",
    "magma",
    "inferno",
    "gray",
    "cividis",
    "turbo",
    "twilight",
    "RdBu_r",
    "coolwarm",
)
MAX_SHOWN = 2048  # larger images are shown every n-th pixel (the data are not changed)
ROI_COLOURS = ("#e53935", "#1e88e5", "#43a047", "#fb8c00", "#8e24aa", "#00acc1")
PIXEL_ALIASES = {"yb": "y", "xb": "x", "y": "y", "x": "x"}


@dataclass
class ImageOptions:
    cmap: str = "viridis"
    log: bool = False
    percentiles: tuple[float, float] = (1.0, 99.5)
    vmin: float | None = None
    vmax: float | None = None
    colorbar: bool = True
    aspect: str = "equal"


@dataclass
class CurveOptions:
    log_y: bool = False
    errors: bool = True
    markers: bool = True
    normalize: bool = False
    legend: bool = True


def _coordinate(da: xr.DataArray, dim: str) -> np.ndarray | None:
    if dim not in da.coords:
        return None
    values = np.asarray(da.coords[dim].values)
    if values.dtype.kind not in "iuf" or values.size == 0:
        return None
    return values.astype(float)


def extent_of(da: xr.DataArray) -> list[float] | None:
    """The imshow extent from the coordinates of a 2-D array (``None``: plain indices)."""
    dy, dx = (str(d) for d in da.dims[-2:])
    y, x = _coordinate(da, dy), _coordinate(da, dx)
    if y is None or x is None or y.size < 2 or x.size < 2:
        return None
    sy = (y[-1] - y[0]) / (y.size - 1)
    sx = (x[-1] - x[0]) / (x.size - 1)
    if sy == 0 or sx == 0:
        return None
    return [x[0] - sx / 2, x[-1] + sx / 2, y[-1] + sy / 2, y[0] - sy / 2]


def _limits(values: np.ndarray, options: ImageOptions) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    if finite.size > 1_000_000:  # percentiles of a sample: as good, much faster
        finite = finite[:: finite.size // 1_000_000 + 1]
    lo, hi = np.percentile(finite, options.percentiles)
    lo = options.vmin if options.vmin is not None else float(lo)
    hi = options.vmax if options.vmax is not None else float(hi)
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


def draw_image(
    fig: Any,
    da: xr.DataArray,
    options: ImageOptions,
    title: str = "",
    rois: Mapping[str, Any] | None = None,
) -> Any:
    """``da`` (2-D) as an image on its coordinates; ROI boxes over the dims they limit."""
    ax = fig.add_subplot(111)
    shown = da
    step = max(1, max(da.shape) // MAX_SHOWN)
    if step > 1:
        shown = da.isel({str(d): slice(None, None, step) for d in da.dims})
    values = np.asarray(shown.values, dtype=float)
    if options.log:
        with np.errstate(invalid="ignore", divide="ignore"):
            values = np.log10(np.clip(values, 0, None) + 1.0)
    lo, hi = _limits(values, options)
    extent = extent_of(shown)
    dy, dx = (str(d) for d in da.dims[-2:])
    # square pixels for detector images; a map over two motors fills the axes
    aspect = "equal" if options.aspect == "equal" and dx in PIXEL_ALIASES else "auto"
    image = ax.imshow(
        values,
        cmap=options.cmap,
        vmin=lo,
        vmax=hi,
        extent=extent,
        interpolation="nearest",
        origin="upper",
        aspect=aspect,
    )
    ax.set_xlabel(_label(da, dx))
    ax.set_ylabel(_label(da, dy))
    if options.colorbar:
        bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
        units = da.attrs.get("units", "")
        bar.set_label(
            ("log10(1+I)" if options.log else str(da.name or "")) + (f" ({units})" if units else "")
        )
    if rois:
        _draw_rois(ax, rois, dy, dx)
    note = f" (every {step}th pixel shown)" if step > 1 else ""
    ax.set_title(title + note, fontsize=9)
    return ax


def _label(da: xr.DataArray, dim: str) -> str:
    units = da.coords[dim].attrs.get("units", "") if dim in da.coords else ""
    return f"{dim} ({units})" if units else dim


def _draw_rois(ax: Any, rois: Mapping[str, Any], dy: str, dx: str) -> None:
    from matplotlib.patches import Rectangle

    for k, (name, roi) in enumerate(rois.items()):
        # binned pixel dims carry full-resolution pixel coordinates: the y/x ranges apply
        ry = roi.ranges.get(PIXEL_ALIASES.get(dy, dy))
        rx = roi.ranges.get(PIXEL_ALIASES.get(dx, dx))
        if ry is None and rx is None:
            continue
        x0, x1 = ax.get_xlim()
        y_top, y_bottom = ax.get_ylim()[1], ax.get_ylim()[0]
        lo_x = rx[0] if rx and rx[0] is not None else min(x0, x1)
        hi_x = rx[1] if rx and rx[1] is not None else max(x0, x1)
        lo_y = ry[0] if ry and ry[0] is not None else min(y_top, y_bottom)
        hi_y = ry[1] if ry and ry[1] is not None else max(y_top, y_bottom)
        colour = ROI_COLOURS[k % len(ROI_COLOURS)]
        ax.add_patch(
            Rectangle((lo_x, lo_y), hi_x - lo_x, hi_y - lo_y, fill=False, edgecolor=colour, lw=1.2)
        )
        ax.annotate(name, (lo_x, lo_y), color=colour, fontsize=8, va="bottom")


def draw_rgb(fig: Any, da: xr.DataArray, title: str = "") -> Any:
    ax = fig.add_subplot(111)
    picture = np.clip(np.nan_to_num(np.asarray(da.values, dtype=float)), 0.0, 1.0)
    plane = da.isel(rgb=0) if "rgb" in da.dims else da
    ax.imshow(picture, extent=extent_of(plane), interpolation="nearest")
    dims = [str(d) for d in da.dims if d != "rgb"]
    if len(dims) == 2:
        ax.set_ylabel(dims[0])
        ax.set_xlabel(dims[1])
    ax.set_title(title, fontsize=9)
    return ax


def draw_curves(
    fig: Any,
    curves: Sequence[tuple[str, xr.DataArray, Any]],
    options: CurveOptions,
    fits: Sequence[tuple[str, xr.DataArray]] = (),
    title: str = "",
) -> Any:
    """Curves ``[(label, data, sigma), ...]`` on one axes, with error bars and fitted curves."""
    ax = fig.add_subplot(111)
    for label, da, sigma in curves:
        dim = str(da.dims[0])
        x = _coordinate(da, dim)
        x = np.arange(da.sizes[dim], dtype=float) if x is None else x
        y = np.asarray(da.values, dtype=float)
        err = (
            None
            if sigma is None
            else np.asarray(sigma.values if hasattr(sigma, "values") else sigma, float)
        )
        if options.normalize:
            top = np.nanmax(np.abs(y)) or 1.0
            y = y / top
            err = None if err is None else err / top
        style = {"marker": "o", "ms": 3} if options.markers else {}
        if options.errors and err is not None:
            ax.errorbar(x, y, yerr=err, lw=1, capsize=0, elinewidth=0.6, label=label, **style)
        else:
            ax.plot(x, y, lw=1, label=label, **style)
    for label, da in fits:
        dim = str(da.dims[0])
        x = _coordinate(da, dim)
        x = np.arange(da.sizes[dim], dtype=float) if x is None else x
        ax.plot(x, np.asarray(da.values, dtype=float), "--", lw=1.2, label=label)
    if options.log_y:
        ax.set_yscale("log")
    if curves:
        first = curves[0][1]
        ax.set_xlabel(_label(first, str(first.dims[0])))
        ax.set_ylabel("normalised" if options.normalize else str(first.name or "value"))
    if options.legend and (len(curves) + len(fits)) > 1:
        ax.legend(fontsize=8, loc="best")
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.25)
    return ax
