"""Maps of per-pixel quantities: centre of mass, bivariate blends, HSV, vector fields.

These reproduce the figures the lab drew in every notebook: the RdBu COM map
with limits ``centre +/- span/2``, the blue/red plus green/purple bivariate
blend with its inset key, the HSV mosaicity image, and quiver plots with a
colour wheel.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, hsv_to_rgb

from bexa.viz import style
from bexa.viz.images import add_scalebar, scalebar_nm
from bexa.viz.style import DEFAULT, as_array, new_axes

__all__ = [
    "bivariate",
    "cluster_map",
    "color_wheel",
    "com_map",
    "contour_quiver_overlay",
    "hsv_mosaicity",
    "vector_field",
]

BLUE_RED = LinearSegmentedColormap.from_list("blue_white_red", ["blue", "white", "red"])
GREEN_PURPLE = LinearSegmentedColormap.from_list("green_white_purple", ["green", "white", "purple"])


def _normalize(
    data: np.ndarray, limits: tuple[float, float] | None = None
) -> tuple[np.ndarray, tuple[float, float]]:
    finite = data[np.isfinite(data)]
    if limits is None:
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    else:
        lo, hi = limits
    span = hi - lo if hi > lo else 1.0
    return np.clip((np.nan_to_num(data, nan=lo) - lo) / span, 0, 1), (lo, hi)


def _scalebar(ax: Any, kind: str, scalebar: Any, source: Any) -> None:
    """Draw the scalebar the argument or the style asks for (``True``: from the attrs)."""
    nm = scalebar_nm(style.option(kind, "scalebar", scalebar, fallback=True), source)
    if nm:
        add_scalebar(ax, nm)


def com_map(
    com: Any,
    ax: Any = None,
    center: float | str = "mean",
    span: float | None = None,
    span_fraction: Any = DEFAULT,
    cmap: Any = DEFAULT,
    scalebar: Any = DEFAULT,
    title: str | None = None,
    colorbar: Any = DEFAULT,
    figsize: tuple[float, float] | None = None,
    clim: Any = None,
) -> tuple[Any, Any, Any]:
    """A centre-of-mass map with limits ``center +/- span / 2``.

    Parameters
    ----------
    center
        ``"mean"``, ``"median"`` or a value.
    span
        Width of the colour scale; default ``span_fraction`` of the map's range
        (the notebooks used a fifth of the motor range). ``style.set(com={"span": 0.1})``
        sets it for every COM map.
    cmap
        Default the style's ``diverging_cmap`` (``"RdBu"``).
    clim
        Explicit limits instead of centre and span: ``(vmin, vmax)``, ``("p", lo, hi)``
        or ``"p2-98"``.
    """
    fig, ax = new_axes(ax, style.figure_size("com", figsize))
    data = as_array(com)
    finite = data[np.isfinite(data)]
    clim = style.local_option("com", "clim", clim)
    if clim is not None:
        vmin, vmax = style.resolve_clim(data, clim)
    else:
        if isinstance(center, str):
            c = float(np.mean(finite)) if center == "mean" else float(np.median(finite))
        else:
            c = float(center)
        span = style.option("com", "span", span)
        if span is None:
            fraction = style.option("com", "span_fraction", span_fraction, fallback=0.2)
            span = fraction * float(np.ptp(finite)) if finite.size else 1.0
        vmin, vmax = c - span / 2, c + span / 2
    im = ax.imshow(data, cmap=style.colormap("com", cmap), vmin=vmin, vmax=vmax)
    unit = com.attrs.get("units", "") if hasattr(com, "attrs") else ""
    label = str(getattr(com, "name", "") or "centre of mass") + (f" ({unit})" if unit else "")
    style.colorbar(fig, im, ax, "com", colorbar, label=label)
    if title is None and hasattr(com, "name") and com.name:
        title = str(com.name)
    if title:
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    _scalebar(ax, "com", scalebar, com)
    return fig, ax, im


def bivariate(
    map_a: Any,
    map_b: Any,
    ax: Any = None,
    cmaps: tuple[Any, Any] = (BLUE_RED, GREEN_PURPLE),
    labels: tuple[str, str] | None = None,
    limits: tuple[tuple[float, float] | None, tuple[float, float] | None] = (None, None),
    key: bool = True,
    scalebar: Any = DEFAULT,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Any, Any]:
    """Blend two maps (for example COM in mu and in chi) into one RGB image with a key.

    Each map is normalised to [0, 1] (over its own range or ``limits``), mapped
    through its colormap, and the two RGB images are averaged, as in the lab's
    notebooks.
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    fig, ax = new_axes(ax, style.figure_size("bivariate", figsize))
    a, lim_a = _normalize(as_array(map_a), limits[0])
    b, lim_b = _normalize(as_array(map_b), limits[1])
    cmap_a, cmap_b = cmaps
    rgb = 0.5 * cmap_a(a)[..., :3] + 0.5 * cmap_b(b)[..., :3]
    ax.imshow(rgb)
    ax.set_xticks([])
    ax.set_yticks([])
    if labels is None:
        labels = (str(getattr(map_a, "name", "a")), str(getattr(map_b, "name", "b")))
    if title:
        ax.set_title(title)
    if key:
        n = 100
        ga, gb = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
        key_rgb = 0.5 * cmap_a(ga)[..., :3] + 0.5 * cmap_b(gb)[..., :3]
        inset = inset_axes(ax, width="25%", height="25%", loc="upper right", borderpad=1)
        inset.imshow(
            key_rgb, extent=[lim_a[0], lim_a[1], lim_b[0], lim_b[1]], origin="lower", aspect="auto"
        )
        inset.set_xlabel(labels[0], fontsize=7)
        inset.set_ylabel(labels[1], fontsize=7)
        inset.tick_params(labelsize=6)
    _scalebar(ax, "bivariate", scalebar, map_a)
    return fig, ax


def hsv_mosaicity(
    map_a: Any,
    map_b: Any,
    ax: Any = None,
    labels: tuple[str, str] | None = None,
    key: bool = True,
    scalebar: Any = DEFAULT,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Any, Any]:
    """darfix-style mosaicity image: hue from ``map_a``, saturation from ``map_b``."""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    fig, ax = new_axes(ax, style.figure_size("hsv", figsize))
    h, lim_a = _normalize(as_array(map_a))
    s, lim_b = _normalize(as_array(map_b))
    rgb = hsv_to_rgb(np.stack([h, s, np.ones_like(h)], axis=-1))
    ax.imshow(rgb)
    ax.set_xticks([])
    ax.set_yticks([])
    if labels is None:
        labels = (str(getattr(map_a, "name", "hue")), str(getattr(map_b, "name", "saturation")))
    if title:
        ax.set_title(title)
    if key:
        n = 100
        hh, ss = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
        key_rgb = hsv_to_rgb(np.stack([hh, ss, np.ones_like(hh)], axis=-1))
        inset = inset_axes(ax, width="25%", height="25%", loc="upper right", borderpad=1)
        inset.imshow(
            key_rgb, extent=[lim_a[0], lim_a[1], lim_b[0], lim_b[1]], origin="lower", aspect="auto"
        )
        inset.set_xlabel(labels[0], fontsize=7)
        inset.set_ylabel(labels[1], fontsize=7)
        inset.tick_params(labelsize=6)
    _scalebar(ax, "hsv", scalebar, map_a)
    return fig, ax


def color_wheel(ax: Any, cmap: str = "twilight", n: int = 100) -> None:
    """Draw a hue wheel keyed to angles (for vector-field figures)."""
    import matplotlib.pyplot as plt

    theta = np.linspace(-np.pi, np.pi, n)
    r = np.linspace(0.6, 1.0, 2)
    tt, rr = np.meshgrid(theta, r)
    ax.pcolormesh(tt, rr, tt, cmap=plt.get_cmap(cmap), shading="auto")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("angle", fontsize=8)


def vector_field(
    u: Any,
    v: Any,
    ax: Any = None,
    step: int = 8,
    background: Any = None,
    cmap: str = "twilight",
    wheel: bool = True,
    scale: float | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Any, Any]:
    """Quiver plot of ``(u, v)`` (for example COM in two motors) coloured by direction."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure(figsize=style.figure_size("vector_field", figsize) or (8, 6))
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = ax.figure
    U = as_array(u)
    V = as_array(v)
    if background is not None:
        ax.imshow(as_array(background), cmap="gray", alpha=0.6)
    yy, xx = np.mgrid[0 : U.shape[0] : step, 0 : U.shape[1] : step]
    us, vs = U[::step, ::step], V[::step, ::step]
    valid = np.isfinite(us) & np.isfinite(vs)
    angle = np.arctan2(vs[valid], us[valid])
    ax.quiver(
        xx[valid],
        yy[valid],
        us[valid],
        vs[valid],
        angle,
        cmap=cmap,
        scale=scale,
        clim=(-np.pi, np.pi),
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    if title:
        ax.set_title(title)
    if wheel:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        inset = inset_axes(
            ax, width="18%", height="18%", loc="lower left", borderpad=1, axes_class=plt.PolarAxes
        )
        color_wheel(inset, cmap)
    return fig, ax


def contour_quiver_overlay(
    background: Any,
    u: Any,
    v: Any,
    ax: Any = None,
    levels: int = 8,
    step: int = 8,
    cmap: str = "viridis",
    figsize: tuple[float, float] | None = None,
) -> tuple[Any, Any]:
    """Contours of ``background`` with the ``(u, v)`` field drawn on top."""
    fig, ax = new_axes(ax, figsize)
    bg = as_array(background)
    ax.imshow(bg, cmap=cmap)
    ax.contour(bg, levels=levels, colors="w", linewidths=0.5)
    U, V = as_array(u), as_array(v)
    yy, xx = np.mgrid[0 : U.shape[0] : step, 0 : U.shape[1] : step]
    ax.quiver(xx, yy, U[::step, ::step], V[::step, ::step], color="k")
    return fig, ax


def cluster_map(
    labels: Any,
    ax: Any = None,
    cmap: str = "tab10",
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    colorbar: Any = DEFAULT,
) -> tuple[Any, Any, Any]:
    """Show integer cluster labels with a discrete colormap."""
    fig, ax = new_axes(ax, style.figure_size("cluster", figsize))
    im = ax.imshow(as_array(labels), cmap=cmap, interpolation="nearest")
    style.colorbar(fig, im, ax, "cluster", colorbar, label="cluster")
    if title:
        ax.set_title(title)
    return fig, ax, im
