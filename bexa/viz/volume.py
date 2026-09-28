"""Views of N-D volumes: projection panels, slice grids, isosurfaces, 3-D RLP scatter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import xarray as xr

from bexa.viz.images import tiles
from bexa.viz.style import as_array, dim_label, resolve_clim

__all__ = ["isosurface", "projections_panel", "render", "rlp_scatter", "show_plotly", "slices_grid"]


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
    backend: str = "mpl",
) -> Any:
    """Panel of every projection of a volume: ``xy``, motor-vs-pixel and motor-vs-motor maps.

    Accepts a volume ``(motors..., y, x)`` or the result dict of the
    :class:`bexa.core.reductions.Projections` accumulator. Returns
    ``(fig, axes)``; with ``backend="plotly"`` an interactive plotly figure.
    """
    projections = (
        _projections_from_volume(data, method) if isinstance(data, xr.DataArray) else dict(data)
    )
    projections = {k: v for k, v in projections.items() if k != "grid" and v.ndim == 2}
    if keys:
        projections = {k: projections[k] for k in keys if k in projections}
    n = len(projections)
    cols = min(3, n) or 1
    rows = int(np.ceil(n / cols))
    if backend == "plotly":
        return _projections_plotly(projections, rows, cols, log, cmap)
    import matplotlib.pyplot as plt

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
        ax.set_xlabel(dim_label(arr, str(d1)))
        ax.set_ylabel(dim_label(arr, str(d0)))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    return fig, axes


def _projections_plotly(
    projections: Mapping[str, xr.DataArray], rows: int, cols: int, log: bool, cmap: str | None
) -> Any:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=list(projections))
    for i, (name, arr) in enumerate(projections.items()):
        values = as_array(arr)
        if log:
            values = np.log10(1.0 + np.clip(values, 0, None))
        vmin, vmax = resolve_clim(values, ("p", 1, 99))
        d0, d1 = arr.dims
        x = np.asarray(arr.coords[d1].values) if d1 in arr.coords else None
        y = np.asarray(arr.coords[d0].values) if d0 in arr.coords else None
        fig.add_trace(
            go.Heatmap(
                z=values,
                x=x,
                y=y,
                zmin=vmin,
                zmax=vmax,
                colorscale=cmap or "Viridis",
                name=name,
                showscale=i == 0,
            ),
            row=i // cols + 1,
            col=i % cols + 1,
        )
        row, col = i // cols + 1, i % cols + 1
        fig.update_xaxes(title_text=dim_label(arr, str(d1)), row=row, col=col)
        fig.update_yaxes(title_text=dim_label(arr, str(d0)), autorange="reversed", row=row, col=col)
    fig.update_layout(height=320 * rows, width=380 * cols)
    return fig


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
    backend: str = "mpl",
) -> Any:
    """3-D isosurface of a volume (needs scikit-image; plotly with ``backend="plotly"``).

    ``threshold`` defaults to the 90th percentile of the finite values.
    Returns ``(fig, ax)`` for matplotlib or the plotly figure.
    """
    data = as_array(volume)
    if data.ndim != 3:
        data = data.reshape(-1, *data.shape[-2:])
    if downsample > 1:
        data = data[::downsample, ::downsample, ::downsample]
    finite = data[np.isfinite(data)]
    if threshold is None:
        threshold = float(np.percentile(finite, 90))
    if backend == "plotly":
        import plotly.graph_objects as go

        z, y, x = np.mgrid[0 : data.shape[0], 0 : data.shape[1], 0 : data.shape[2]]
        return go.Figure(
            go.Isosurface(
                x=x.ravel(), y=y.ravel(), z=z.ravel(), value=np.nan_to_num(data).ravel(),
                isomin=threshold, isomax=float(finite.max()), surface_count=2,
                opacity=alpha, caps={"x_show": False, "y_show": False, "z_show": False},
            )
        )  # fmt: skip
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    try:
        from skimage.measure import marching_cubes
    except ImportError as exc:
        raise ImportError("isosurface needs scikit-image: pip install scikit-image") from exc
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


_RENDER_MODES: dict[str, dict[str, Any]] = {
    # opacityscale maps the normalised value to an opacity factor; opacity scales all of them
    "translucent": {
        "opacityscale": [[0.0, 0.0], [0.3, 0.05], [0.7, 0.4], [1.0, 1.0]],
        "opacity": 0.6,
        "surface_count": 17,
    },
    "mip": {
        "opacityscale": [[0.0, 0.0], [0.6, 0.0], [0.8, 0.5], [1.0, 1.0]],
        "opacity": 0.9,
        "surface_count": 15,
    },
    "iso": {"opacityscale": "uniform", "opacity": 0.6, "surface_count": 1},
}


def render(
    volume: Any,
    mode: str = "translucent",
    log: bool = False,
    clim: Any = ("p", 5, 99.5),
    cmap: str = "viridis",
    scale: Sequence[float] = (1.0, 1.0, 1.0),
    downsample: int | None = None,
    max_voxels: int = 300_000,
    surface_count: int | None = None,
    opacity: float | None = None,
    title: str | None = None,
) -> Any:
    """Render a ``(z, y, x)`` stack with plotly: colour and opacity follow the intensity.

    ``mode``: ``"translucent"`` fades dim voxels out and lets bright ones glow, like napari's
    translucent rendering and the Slices-to-Voxels app; ``"mip"`` keeps only the brightest
    voxels visible, like a maximum-intensity projection; ``"iso"`` draws one surface at the
    upper colour limit. ``scale`` is the voxel size ``(z, y, x)`` relative to a pixel, as
    napari's ``scale``. Stacks with more than ``max_voxels`` voxels are block-averaged in
    ``y`` and ``x`` first (``downsample`` sets the factor by hand); the browser cannot rotate
    much more than that. The figure rotates and zooms in the browser; in a notebook show it
    with :func:`show_plotly`.
    """
    import plotly.graph_objects as go

    from bexa.core.reductions import block_reduce

    if mode not in _RENDER_MODES:
        raise ValueError(f"unknown mode {mode!r}; use translucent, mip or iso")
    names = [str(d) for d in getattr(volume, "dims", ("z", "y", "x"))[-3:]]
    data = np.asarray(as_array(volume), dtype=np.float32)
    if data.ndim != 3:
        data = data.reshape(-1, *data.shape[-2:])
    factor = downsample or max(1, int(np.ceil(np.sqrt(data.size / max_voxels))))
    if factor > 1:
        data = block_reduce(data, factor, factor, "mean")
    if log:
        data = np.log10(1.0 + np.clip(data, 0, None))
    vmin, vmax = resolve_clim(data, clim)
    if vmin is None or vmax is None or not vmax > vmin:
        finite = data[np.isfinite(data)]
        vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    sz, sy, sx = (float(s) for s in scale)
    nz, ny, nx = data.shape
    zz, yy, xx = np.meshgrid(
        np.arange(nz) * sz, np.arange(ny) * sy * factor, np.arange(nx) * sx * factor, indexing="ij"
    )
    settings = dict(_RENDER_MODES[mode])
    if surface_count is not None:
        settings["surface_count"] = int(surface_count)
    if opacity is not None:
        settings["opacity"] = float(opacity)
    if mode == "iso":  # one surface, at the upper colour limit
        vmin = vmax
    label = "log10(1 + I)" if log else str(getattr(volume, "name", "") or "intensity")
    trace = go.Volume(
        x=xx.ravel(),
        y=yy.ravel(),
        z=zz.ravel(),
        value=np.nan_to_num(data, nan=vmin).ravel(),
        isomin=vmin,
        isomax=vmax,
        colorscale=cmap,
        caps={"x_show": False, "y_show": False, "z_show": False},
        colorbar={"title": {"text": label}},
        **settings,
    )
    fig = go.Figure(trace)
    fig.update_layout(
        title=title,
        margin={"l": 0, "r": 0, "t": 40 if title else 10, "b": 0},
        scene={
            "xaxis_title": names[2],
            "yaxis_title": names[1],
            "zaxis_title": names[0],
            "aspectmode": "data",
        },
    )
    return fig


def show_plotly(
    fig: Any, how: str = "auto", height: int = 600, include_plotlyjs: str | bool = "cdn"
) -> Any:
    """Show a plotly figure in a notebook without needing the plotly JupyterLab extension.

    ``how``: ``"iframe"`` embeds the figure as a self-contained inline frame, which works in
    any JupyterLab (``include_plotlyjs="cdn"`` loads plotly.js from the web, so the browser
    needs internet; ``True`` embeds the 3 MB library and works offline); ``"native"`` uses
    plotly's own display, which needs the extension; ``"browser"`` opens a browser tab;
    ``"html"`` returns the page as text. ``"auto"`` is ``"iframe"`` in IPython, ``"html"``
    when ``BEXA_HEADLESS`` is set and ``"browser"`` otherwise.
    """
    import html as html_module
    import os

    from bexa._log import in_ipython

    if how == "auto":
        if in_ipython():
            how = "iframe"
        else:
            how = "html" if os.environ.get("BEXA_HEADLESS") else "browser"
    if how == "browser":
        fig.show(renderer="browser")
        return None
    if how == "native":
        fig.show()
        return None
    if how not in ("iframe", "html"):
        raise ValueError(f"unknown how {how!r}; use auto, iframe, native, browser or html")
    page = fig.to_html(
        include_plotlyjs=include_plotlyjs,
        full_html=True,
        default_width="100%",
        default_height=f"{height}px",
    )
    if how == "html":
        return page
    from IPython.display import HTML, display

    frame = HTML(  # inside a div: IPython treats a bare iframe as a link to a page
        f'<div><iframe srcdoc="{html_module.escape(page, quote=True)}" width="100%" '
        f'height="{height + 24}" style="border:0" allowfullscreen></iframe></div>'
    )
    display(frame)
    return frame
