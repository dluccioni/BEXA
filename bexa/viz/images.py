"""Single images, tile grids, ROI overlays and scalebars."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from bexa.core.roi import ROI
from bexa.viz.style import as_array, dim_label, new_axes, pixel_size_nm, resolve_clim

__all__ = ["add_scalebar", "roi_overlay", "show", "tiles"]


def _extent(da: Any) -> list[float] | None:
    """imshow extent from the pixel coordinates (the last two dims) of a DataArray."""
    if not (hasattr(da, "coords") and hasattr(da, "dims") and len(da.dims) >= 2):
        return None
    dy_name, dx_name = da.dims[-2:]
    if dy_name not in da.coords or dx_name not in da.coords:
        return None
    y = np.asarray(da.coords[dy_name].values, dtype=float)
    x = np.asarray(da.coords[dx_name].values, dtype=float)
    if y.size < 2 or x.size < 2:
        return None
    dy = y[1] - y[0]
    dx = x[1] - x[0]
    return [x[0] - dx / 2, x[-1] + dx / 2, y[-1] + dy / 2, y[0] - dy / 2]


def add_scalebar(
    ax: Any, pixel_size_nm: float, location: str = "lower right", **kwargs: Any
) -> Any:
    """Add a scalebar for ``pixel_size_nm`` nanometres per pixel (needs matplotlib-scalebar)."""
    try:
        from matplotlib_scalebar.scalebar import ScaleBar
    except ImportError:
        return None
    bar = ScaleBar(pixel_size_nm, units="nm", location=location, **kwargs)
    ax.add_artist(bar)
    return bar


def roi_overlay(
    ax: Any, rois: ROI | dict[str, ROI] | Sequence[ROI], color: str = "r", lw: float = 1.0
) -> list[Any]:
    """Draw rectangles for the pixel ranges of one or more ROIs."""
    from matplotlib.patches import Rectangle

    if isinstance(rois, ROI):
        rois = {"roi": rois}
    if not isinstance(rois, dict):
        rois = {f"roi{i}": r for i, r in enumerate(rois)}
    patches = []
    for name, roi in rois.items():
        y = roi.get("y") or (None, None)
        x = roi.get("x") or (None, None)
        if y[0] is None or y[1] is None or x[0] is None or x[1] is None:
            continue
        rect = Rectangle((x[0], y[0]), x[1] - x[0], y[1] - y[0], fill=False, edgecolor=color, lw=lw)
        ax.add_patch(rect)
        ax.annotate(name, (x[0], y[0]), color=color, fontsize=7, va="bottom")
        patches.append(rect)
    return patches


def show(
    image: Any,
    ax: Any = None,
    clim: Any = ("p", 1, 99),
    cmap: str | None = None,
    log: bool = False,
    roi: ROI | dict[str, ROI] | None = None,
    scalebar: bool | float = True,
    title: str | None = None,
    colorbar: bool = True,
    figsize: tuple[float, float] | None = None,
    **imshow_kwargs: Any,
) -> tuple[Any, Any, Any]:
    """Display a 2-D image with percentile colour limits, optional ROI boxes and scalebar.

    Parameters
    ----------
    image
        2-D DataArray (uses its ``y``/``x`` coordinates) or array.
    clim
        ``("p", 1, 99)`` for percentiles, ``(vmin, vmax)`` or ``None``.
    log
        Show ``log10(1 + image)``.
    scalebar
        ``True`` uses ``attrs["effective_pixel_nm"]`` when present; a number
        gives nanometres per pixel; ``False`` draws none.

    Returns
    -------
    (fig, ax, image_artist)
    """
    fig, ax = new_axes(ax, figsize)
    data = as_array(image)
    if data.ndim != 2:
        raise ValueError(f"show() needs a 2-D image; got shape {data.shape}")
    if log:
        data = np.log10(1.0 + np.clip(data, 0, None))
    vmin, vmax = resolve_clim(data, clim)
    extent = _extent(image)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, **imshow_kwargs)
    ax.set_xlabel(dim_label(image, "x"))
    ax.set_ylabel(dim_label(image, "y"))
    if title is None and hasattr(image, "name") and image.name:
        title = str(image.name)
    if title:
        ax.set_title(title)
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if roi is not None:
        roi_overlay(ax, roi)
    nm = scalebar if isinstance(scalebar, (int, float)) and not isinstance(scalebar, bool) else None
    if scalebar is True:
        nm = pixel_size_nm(image)
    if nm:
        add_scalebar(ax, nm)
    return fig, ax, im


def tiles(
    volume: Any,
    axis: int | str = 0,
    n: int = 9,
    per_row: int = 3,
    indices: Sequence[int] | None = None,
    clim: Any = "shared",
    cmap: str | None = None,
    log: bool = False,
    titles: bool = True,
    figsize: tuple[float, float] | None = None,
) -> tuple[Any, np.ndarray]:
    """Grid of images taken along one axis of a stack.

    Parameters
    ----------
    volume
        3-D DataArray or array ``(axis, y, x)``; higher dims are summed away
        except ``axis``.
    axis
        Index or dim name to step along.
    clim
        ``"shared"`` (one scale for all tiles), ``"each"``, or an explicit spec.
    """
    import matplotlib.pyplot as plt

    if hasattr(volume, "dims"):
        dim = volume.dims[axis] if isinstance(axis, int) else axis
        pixels = tuple(volume.dims[-2:])
        others = [d for d in volume.dims if d not in (dim, *pixels)]
        stack = volume.sum(others) if others else volume
        stack = stack.transpose(dim, *pixels)
        coords = (
            np.asarray(stack.coords[dim].values)
            if dim in stack.coords
            else np.arange(stack.shape[0])
        )
        data = as_array(stack)
        label = dim
    else:
        data = np.moveaxis(as_array(volume), axis if isinstance(axis, int) else 0, 0)
        coords = np.arange(data.shape[0])
        label = "index"
    if log:
        data = np.log10(1.0 + np.clip(data, 0, None))
    if indices is None:
        indices = np.unique(np.linspace(0, data.shape[0] - 1, min(n, data.shape[0])).astype(int))
    rows = int(np.ceil(len(indices) / per_row))
    fig, axes = plt.subplots(
        rows, per_row, figsize=figsize or (3.2 * per_row, 3.0 * rows), squeeze=False
    )
    shared = resolve_clim(data[list(indices)], ("p", 1, 99)) if clim == "shared" else None
    for k, ax in enumerate(axes.ravel()):
        if k >= len(indices):
            ax.axis("off")
            continue
        i = int(indices[k])
        vmin, vmax = (
            shared if shared else resolve_clim(data[i], ("p", 1, 99) if clim == "each" else clim)
        )
        ax.imshow(data[i], cmap=cmap, vmin=vmin, vmax=vmax)
        if titles:
            ax.set_title(f"{label} = {coords[i]:.4g}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    return fig, axes
