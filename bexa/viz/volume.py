"""Views of N-D volumes: projection panels, slice grids, isosurfaces, 3-D RLP scatter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import xarray as xr

from bexa.viz.images import tiles
from bexa.viz.style import as_array, dim_label, resolve_clim

__all__ = ["isosurface", "projections_panel", "rlp_scatter", "slices_grid"]


def _projections_from_volume(volume: xr.DataArray, method: str) -> dict[str, xr.DataArray]:
    reducer = getattr(volume, method)
    motors = [d for d in volume.dims if d not in ("y", "x")]
    out: dict[str, xr.DataArray] = {"xy": reducer(motors) if motors else volume}
    for d in motors:
        others = [m for m in motors if m != d]
        out[f"{d}_y"] = reducer([*others, "x"])
        out[f"{d}_x"] = reducer([*others, "y"])
    for i, d0 in enumerate(motors):
        for d1 in motors[i + 1 :]:
            others = [m for m in motors if m not in (d0, d1)]
            out[f"{d0}_{d1}"] = reducer([*others, "y", "x"])
    return out


def projections_panel(
    data: xr.DataArray | Mapping[str, xr.DataArray],
    method: str = "sum",
    keys: list[str] | None = None,
    log: bool = False,
    cmap: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Any, np.ndarray]:
    """Panel of every projection of a volume: ``xy``, motor-vs-pixel and motor-vs-motor maps.

    Accepts a volume ``(motors..., y, x)`` or the result dict of the
    :class:`bexa.core.reductions.Projections` accumulator.
    """
    import matplotlib.pyplot as plt

    projections = (
        _projections_from_volume(data, method) if isinstance(data, xr.DataArray) else dict(data)
    )
    projections = {k: v for k, v in projections.items() if k != "grid" and v.ndim == 2}
    if keys:
        projections = {k: projections[k] for k in keys if k in projections}
    n = len(projections)
    cols = min(3, n) or 1
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize or (4.2 * cols, 3.6 * rows), squeeze=False)
    for ax, (name, arr) in zip(axes.ravel(), projections.items(), strict=False):
        values = as_array(arr)
        if log:
            values = np.log10(1.0 + np.clip(values, 0, None))
        vmin, vmax = resolve_clim(values, ("p", 1, 99))
        d0, d1 = arr.dims
        extent = None
        if (
            d0 in arr.coords
            and d1 in arr.coords
            and arr.coords[d0].size > 1
            and arr.coords[d1].size > 1
        ):
            c0 = np.asarray(arr.coords[d0].values, dtype=float)
            c1 = np.asarray(arr.coords[d1].values, dtype=float)
            extent = [c1[0], c1[-1], c0[-1], c0[0]]
        im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", extent=extent)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel(dim_label(arr, d1))
        ax.set_ylabel(dim_label(arr, d0))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    return fig, axes


def slices_grid(
    volume: xr.DataArray, axis: str | int = 0, n: int = 9, per_row: int = 3, **kwargs: Any
) -> tuple[Any, np.ndarray]:
    """Tiles along one motor dim, titled with the motor value (see :func:`bexa.viz.images.tiles`).

    ``n`` evenly spaced slices are shown, ``per_row`` per row.
    """
    return tiles(volume, axis=axis, n=n, per_row=per_row, **kwargs)


def isosurface(
    volume: Any,
    threshold: float | None = None,
    ax: Any = None,
    downsample: int = 1,
    color: str = "cyan",
    alpha: float = 0.6,
    figsize: tuple[float, float] = (8, 8),
) -> tuple[Any, Any]:
    """3-D isosurface of a volume (needs scikit-image).

    ``threshold`` defaults to the 90th percentile of the finite values.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    try:
        from skimage.measure import marching_cubes
    except ImportError as exc:
        raise ImportError("isosurface needs scikit-image: pip install scikit-image") from exc
    data = as_array(volume)
    if data.ndim != 3:
        data = data.reshape(-1, *data.shape[-2:])
    if downsample > 1:
        data = data[::downsample, ::downsample, ::downsample]
    finite = data[np.isfinite(data)]
    if threshold is None:
        threshold = float(np.percentile(finite, 90))
    verts, faces, _, _ = marching_cubes(np.nan_to_num(data), level=threshold)
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
    else:
        fig = ax.figure
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    mesh.set_facecolor(color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, data.shape[2])
    ax.set_ylim(0, data.shape[1])
    ax.set_zlim(0, data.shape[0])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel(str(getattr(volume, "dims", ["axis 0"])[0]))
    return fig, ax


def rlp_scatter(
    coords: Mapping[str, Any],
    intensity: Any,
    ax: Any = None,
    threshold: float | None = None,
    cmap: str = "viridis",
    point_size: float = 20.0,
    alpha: float = 0.8,
    log: bool = False,
    backend: str = "mpl",
    figsize: tuple[float, float] = (8, 8),
) -> Any:
    """Scatter of integrated intensity in 3-D reciprocal (or angular) coordinates.

    ``coords`` maps three axis names to arrays of the same shape as ``intensity``
    (see :func:`bexa.geometry.transforms.build_coordinate_grids`). With
    ``backend="plotly"`` an interactive figure is returned instead.
    """
    names = list(coords)
    if len(names) != 3:
        raise ValueError("rlp_scatter needs exactly three coordinate arrays")
    values = as_array(intensity).ravel()
    axes_data = [as_array(coords[n]).ravel() for n in names]
    keep = np.isfinite(values)
    if threshold is not None:
        keep &= values >= threshold
    if log:
        values = np.log10(1.0 + np.clip(values, 0, None))
    xs, ys, zs = (a[keep] for a in axes_data)
    c = values[keep]
    if backend == "plotly":
        import plotly.graph_objects as go

        fig = go.Figure(
            data=go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers",
                marker={"size": 3, "color": c, "colorscale": cmap, "opacity": alpha},
            )
        )
        fig.update_layout(
            scene={"xaxis_title": names[0], "yaxis_title": names[1], "zaxis_title": names[2]}
        )
        return fig
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
    else:
        fig = ax.figure
    sc = ax.scatter(xs, ys, zs, c=c, cmap=cmap, s=point_size, alpha=alpha)
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.set_zlabel(names[2])
    fig.colorbar(sc, ax=ax, shrink=0.6, label="log10 intensity" if log else "intensity")
    return fig, ax
