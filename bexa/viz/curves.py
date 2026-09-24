"""1-D figures: pump-probe on/off panels, moments vs a scan axis, rocking curves."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import xarray as xr

from bexa.core.roi import ROI
from bexa.viz.images import roi_overlay
from bexa.viz.style import as_array, dim_label, new_axes, resolve_clim

__all__ = ["curve", "linearity", "moments_vs_axis", "on_off_diff", "rocking_curves"]


def curve(
    da: xr.DataArray, ax: Any = None, label: str | None = None, **kwargs: Any
) -> tuple[Any, Any]:
    """Line plot of a 1-D DataArray against its coordinate."""
    fig, ax = new_axes(ax)
    dim = da.dims[0]
    x = da.coords[dim].values if dim in da.coords else np.arange(da.size)
    ax.plot(x, as_array(da), label=label or (str(da.name) if da.name else None), **kwargs)
    ax.set_xlabel(dim_label(da, dim))
    ax.set_ylabel(str(da.name or ""))
    if label or da.name:
        ax.legend(frameon=False)
    return fig, ax


def _axis_of(ds: xr.Dataset, axis: str | None) -> str:
    if axis:
        return axis
    if "scan_axis" in ds.attrs:
        return str(ds.attrs["scan_axis"])
    return next(str(d) for d in ds["frames"].dims if d not in ("laser", "y", "x"))


def on_off_diff(
    ds: xr.Dataset,
    roi: ROI | None = None,
    axis: str | None = None,
    mode: str = "diff",
    axes: Any = None,
    figsize: tuple[float, float] = (13, 4),
    title: str | None = None,
) -> tuple[Any, np.ndarray]:
    """Three panels: summed image with the ROI, on/off curves, and their difference.

    Parameters
    ----------
    ds
        Cube dataset with ``frames(laser, <axis>, y, x)`` (see :func:`bexa.load`).
    mode
        ``"diff"`` (on - off), ``"ratio"`` (on / off) or ``"percent"``.
    """
    import matplotlib.pyplot as plt

    axis = _axis_of(ds, axis)
    frames = ds["frames"]
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
    else:
        fig = axes[0].figure
    total = frames.sum(("laser", axis))
    vmin, vmax = resolve_clim(total, ("p", 1, 99))
    axes[0].imshow(as_array(total), vmin=vmin, vmax=vmax)
    if roi is not None:
        roi_overlay(axes[0], roi)
        ys, xs = roi.pixel_slices(frames.shape[-2:])
        window = frames.isel(y=ys, x=xs)
    else:
        window = frames
    on = window.sel(laser="on").sum(("y", "x"))
    off = window.sel(laser="off").sum(("y", "x"))
    x = on.coords[axis].values
    axes[1].plot(x, as_array(on), color="red", label="laser on")
    axes[1].plot(x, as_array(off), color="black", label="laser off")
    axes[1].set_xlabel(dim_label(frames, axis))
    axes[1].set_ylabel("ROI intensity")
    axes[1].legend(frameon=False)
    if mode == "ratio":
        diff, ylabel = as_array(on) / as_array(off), "on / off"
    elif mode == "percent":
        diff, ylabel = 100 * (as_array(on) - as_array(off)) / as_array(off), "(on - off) / off (%)"
    else:
        diff, ylabel = as_array(on) - as_array(off), "on - off"
    axes[2].plot(x, diff, color="black")
    axes[2].axhline(0 if mode != "ratio" else 1, color="gray", lw=0.5)
    axes[2].set_xlabel(dim_label(frames, axis))
    axes[2].set_ylabel(ylabel)
    if title:
        fig.suptitle(title)
    return fig, axes


def moments_vs_axis(
    stats: Mapping[str, xr.DataArray],
    keys: Sequence[str] | None = None,
    axes: Any = None,
    difference: bool = True,
    figsize: tuple[float, float] | None = None,
) -> tuple[Any, np.ndarray]:
    """Plot per-frame statistics (COM, variance, skew, ...) against the scan axis.

    ``stats`` maps names to 1-D DataArrays; names ending in ``_on``/``_off`` are
    paired so both curves share a panel and, with ``difference``, a third
    line shows ``on - off``.
    """
    import matplotlib.pyplot as plt

    names = list(keys) if keys else list(stats)
    bases: dict[str, dict[str, xr.DataArray]] = {}
    for name in names:
        if name.endswith("_on"):
            bases.setdefault(name[:-3], {})["on"] = stats[name]
        elif name.endswith("_off"):
            bases.setdefault(name[:-4], {})["off"] = stats[name]
        else:
            bases.setdefault(name, {})["value"] = stats[name]
    n = len(bases)
    cols = min(3, n) or 1
    rows = int(np.ceil(n / cols))
    if axes is None:
        fig, axes = plt.subplots(
            rows, cols, figsize=figsize or (4.2 * cols, 3.2 * rows), squeeze=False
        )
    else:
        fig = np.asarray(axes).ravel()[0].figure
    for ax, (base, parts) in zip(np.asarray(axes).ravel(), bases.items(), strict=False):
        for key, color in (("on", "red"), ("off", "black"), ("value", "tab:blue")):
            if key in parts:
                da = parts[key]
                dim = da.dims[0]
                x = da.coords[dim].values if dim in da.coords else np.arange(da.size)
                ax.plot(x, as_array(da), color=color, label=key if key != "value" else None)
                ax.set_xlabel(dim_label(da, dim))
        if difference and "on" in parts and "off" in parts:
            ax.plot(
                x,
                as_array(parts["on"]) - as_array(parts["off"]),
                color="tab:green",
                ls="--",
                label="on - off",
            )
        ax.set_ylabel(base)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, fontsize=8)
    for ax in np.asarray(axes).ravel()[n:]:
        ax.axis("off")
    return fig, axes


def rocking_curves(
    curves: xr.DataArray | Mapping[str, Any],
    fits: Mapping[str, Any] | None = None,
    ax: Any = None,
    normalize: bool = False,
    xlabel: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Any, Any]:
    """Rocking curves (intensity vs motor) for one or more ROIs, with optional fits.

    ``curves`` is a 1-D DataArray or ``{label: DataArray}``; ``fits`` maps the
    same labels to ``(x, y)`` arrays or fit results with ``best_fit``.
    """
    fig, ax = new_axes(ax, figsize)
    items = {"": curves} if isinstance(curves, xr.DataArray) else dict(curves)
    for label, da in items.items():
        dim = da.dims[0]
        x = np.asarray(
            da.coords[dim].values if dim in da.coords else np.arange(da.size), dtype=float
        )
        y = as_array(da).astype(float)
        if normalize and np.nanmax(y) > 0:
            y = (y - np.nanmin(y)) / (np.nanmax(y) - np.nanmin(y))
        (line,) = ax.plot(x, y, ".", label=label or None)
        if fits and label in fits:
            fit = fits[label]
            fx, fy = (fit.userkws["x"], fit.best_fit) if hasattr(fit, "best_fit") else fit
            if normalize and np.nanmax(fy) > 0:
                fy = (fy - np.nanmin(fy)) / (np.nanmax(fy) - np.nanmin(fy))
            ax.plot(fx, fy, "-", color=line.get_color())
        if xlabel is None:
            xlabel = dim_label(da, dim)
    ax.set_xlabel(xlabel or "motor")
    ax.set_ylabel("normalised intensity" if normalize else "intensity")
    if any(items):
        ax.legend(frameon=False, fontsize=8)
    return fig, ax


def linearity(
    detector: Any, i0: Any, laser: Any = None, ax: Any = None, figsize: tuple[float, float] = (5, 5)
) -> tuple[Any, Any]:
    """Detector signal against the incident monitor, split by laser state when given."""
    fig, ax = new_axes(ax, figsize)
    d = as_array(detector)
    m = as_array(i0)
    if laser is None:
        ax.plot(d, m, ".", ms=3)
    else:
        flag = as_array(laser)
        on = flag == 1
        ax.plot(d[on], m[on], ".", ms=3, color="red", label="laser on")
        ax.plot(d[~on], m[~on], "x", ms=3, color="black", label="laser off")
        ax.legend(frameon=False)
    ax.set_xlabel("detector (a.u.)")
    ax.set_ylabel("I0 (a.u.)")
    ax.set_title("linearity check")
    return fig, ax
