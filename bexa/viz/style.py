"""Plot conventions: the themes, the style defaults every figure reads, colour limits.

Every plotting function takes the same knobs (``clim``, ``cmap``, ``colorbar``,
``scalebar``, ``figsize``...) and leaves them at :data:`DEFAULT`, which means
"take it from the style defaults": :func:`set` changes those once for a whole
notebook, :func:`context` for a few cells, per plot kind or for all, and
:func:`apply` restyles a figure already drawn. Fonts and dpi stay in
matplotlib's ``rcParams`` (:func:`use` applies a theme).
"""

from __future__ import annotations

import contextlib
import copy
import importlib
import inspect
from collections.abc import Iterator, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field, fields
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from bexa.core.backend import to_host

__all__ = [
    "DEFAULT",
    "GLOBAL_OPTIONS",
    "KINDS",
    "PAPER",
    "SCREEN",
    "THEMES",
    "PlotDefaults",
    "apply",
    "as_array",
    "colorbar",
    "colormap",
    "context",
    "defaults",
    "dim_label",
    "figure_size",
    "limits",
    "local_option",
    "new_axes",
    "option",
    "percentile_limits",
    "percentiles_of",
    "pixel_size_nm",
    "reset",
    "resolve_clim",
    "set",
    "stack_mode",
    "title_font",
    "use",
]

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


# ------------------------------------------------------------------ the defaults
class _Default:
    """The marker of an argument left to the style defaults (see :func:`option`)."""

    def __repr__(self) -> str:
        return "DEFAULT"

    def __bool__(self) -> bool:
        raise TypeError("DEFAULT is a marker, not a value: resolve it with style.option()")


DEFAULT: Any = _Default()


@dataclass
class PlotDefaults:
    """What every figure uses for an argument left at :data:`DEFAULT`.

    ``kinds`` holds overrides per plot kind (``{"com": {"span": 0.1}}``): the
    keywords of that kind's function, or any global option for that kind only
    (``{"tiles": {"title_size": 7}}``); they win over the global values.
    :func:`set` fills both.
    """

    clim: Any = ("p", 1, 99)  # colour limits of every image; stacks take its percentiles
    render_clim: Any = ("p", 5, 99.5)  # plotly volume renderings
    cmap: str | None = None  # None: the theme's rcParams["image.cmap"]
    diverging_cmap: str = "RdBu"  # centre-of-mass maps
    span_fraction: float = 0.2  # a COM map's colour scale: a fifth of the map's range
    figsize: tuple[float, float] | None = None  # single-panel figures; None: the function's own
    panel_size: tuple[float, float] | None = (
        None  # inches per pane of a tiles/grid/projections panel
    )
    colorbar: bool = True
    colorbar_fraction: float = 0.046
    colorbar_pad: float = 0.04
    scalebar: bool | float = True
    title_size: float | None = 9  # pane titles; None: rcParams axes.titlesize
    kinds: dict[str, dict[str, Any]] = field(default_factory=dict)


defaults = PlotDefaults()
GLOBAL_OPTIONS = tuple(f.name for f in fields(PlotDefaults) if f.name != "kinds")
# the plot kinds, and the function or class whose keywords a kind's overrides may name
KINDS: dict[str, tuple[str, str]] = {
    "image": ("bexa.viz.images", "show"),
    "tiles": ("bexa.viz.images", "tiles"),
    "com": ("bexa.viz.maps", "com_map"),
    "width": ("bexa.viz.maps", "com_map"),
    "bivariate": ("bexa.viz.maps", "bivariate"),
    "hsv": ("bexa.viz.maps", "hsv_mosaicity"),
    "vector_field": ("bexa.viz.maps", "vector_field"),
    "cluster": ("bexa.viz.maps", "cluster_map"),
    "curve": ("bexa.viz.curves", "curve"),
    "rocking": ("bexa.viz.curves", "rocking_curves"),
    "on_off": ("bexa.viz.curves", "on_off_diff"),
    "moments": ("bexa.viz.curves", "moments_vs_axis"),
    "linearity": ("bexa.viz.curves", "linearity"),
    "projections": ("bexa.viz.volume", "projections_panel"),
    "isosurface": ("bexa.viz.volume", "isosurface"),
    "rlp": ("bexa.viz.volume", "rlp_scatter"),
    "render": ("bexa.viz.volume", "render"),
    "browse": ("bexa.viz.interactive", "Browser"),
    "volume": ("bexa.viz.interactive", "VolumeBrowser"),
    "roi_picker": ("bexa.viz.interactive", "RoiPicker"),
    "animate": ("bexa.viz.animation", "animate"),
    "grid": ("bexa.viz.auto", "grid"),
}


def _parameters(kind: str) -> AbstractSet[str] | None:
    """The keywords the function of ``kind`` takes; ``None`` when it takes any."""
    module_name, name = KINDS[kind]
    target = getattr(importlib.import_module(module_name), name)
    signature = inspect.signature(target.__init__ if inspect.isclass(target) else target)
    names = {p.name for p in signature.parameters.values() if p.name != "self"}
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return None
    return names


def set(**options: Any) -> None:
    """Change the style defaults: global names, or a plot kind with a dict of its keywords.

    ``style.set(cmap="magma", colorbar=False)`` applies to every figure;
    ``style.set(com={"span": 0.1}, tiles={"per_row": 4})`` to centre-of-mass
    maps and tile grids only (the kinds are the keys of :data:`KINDS`, the
    keywords those of their functions or a global option for that kind). An
    unknown name raises ``ValueError``.
    """
    for name, value in options.items():
        if name in KINDS:
            if not isinstance(value, Mapping):
                raise ValueError(f"{name}= takes a dict of keywords of the {name} plot kind")
            allowed = _parameters(name)
            unknown = sorted({*value} - allowed - {*GLOBAL_OPTIONS}) if allowed else []
            if unknown:
                raise ValueError(
                    f"the {name} plot kind ({KINDS[name][1]}) has no keyword {unknown}; "
                    f"it takes {sorted(allowed or [])} or a global option {GLOBAL_OPTIONS}"
                )
            defaults.kinds.setdefault(name, {}).update(value)
        elif name in GLOBAL_OPTIONS:
            setattr(defaults, name, value)
        else:
            raise ValueError(
                f"unknown style option {name!r}: the global options are {GLOBAL_OPTIONS} "
                f"and the plot kinds {tuple(KINDS)}"
            )


def reset() -> None:
    """Back to the shipped defaults (``rcParams`` are not touched)."""
    defaults.__dict__.clear()
    defaults.__dict__.update(PlotDefaults().__dict__)


@contextlib.contextmanager
def context(**options: Any) -> Iterator[PlotDefaults]:
    """A ``with`` block with other defaults: ``with style.context(clim="p2-98"): ...``."""
    saved = copy.deepcopy(defaults.__dict__)
    set(**options)
    try:
        yield defaults
    finally:
        defaults.__dict__.clear()
        defaults.__dict__.update(saved)


def option(
    kind: str,
    name: str,
    value: Any = DEFAULT,
    *,
    global_name: str | None = None,
    fallback: Any = None,
) -> Any:
    """One keyword: the argument, else the kind's override, the global, or ``fallback``.

    An argument that is :data:`DEFAULT` or ``None`` counts as not given (``clim``
    is the one keyword where ``None`` means something, and :func:`limits` handles
    it first). ``global_name`` names the global field when it differs from the
    keyword (a COM map's ``cmap`` is the ``diverging_cmap``).
    """
    if value is not DEFAULT and value is not None:
        return value
    overrides = defaults.kinds.get(kind, {})
    if name in overrides:
        return overrides[name]
    field_name = global_name or name
    if field_name in GLOBAL_OPTIONS:
        value = getattr(defaults, field_name)
        if value is not None:
            return value
    return fallback


def local_option(kind: str, name: str, value: Any = DEFAULT, fallback: Any = None) -> Any:
    """Like :func:`option` without the global: the argument, the kind's override or ``fallback``.

    For keywords whose global value would be wrong for a kind: tile grids and
    volume browsers draw no colourbar unless asked, whatever ``colorbar`` says.
    """
    if value is not DEFAULT and value is not None:
        return value
    return defaults.kinds.get(kind, {}).get(name, fallback)


# ------------------------------------------------------------------ colour limits
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
    """Turn ``("p", lo, hi)``, ``"p2-98"``, ``(vmin, vmax)`` or ``None`` into colour limits.

    ``"shared"`` and ``"each"`` are the modes of a stack of frames (tiles, the
    browsers, an animation), not limits of one image: they raise here.
    """
    if clim is None:
        return None, None
    if clim is DEFAULT:
        clim = defaults.clim
    if isinstance(clim, str) and clim in ("shared", "each"):
        raise ValueError(f"{clim!r} is a stack mode; use it with tiles, animate or the browsers")
    if isinstance(clim, (tuple, list)) and len(clim) == 3 and clim[0] in ("shared", "each"):
        raise ValueError(f"{clim!r} is a stack mode; use it with tiles, animate or the browsers")
    if isinstance(clim, (tuple, list)) and len(clim) == 3 and clim[0] == "p":
        return percentile_limits(data, clim[1], clim[2])
    if isinstance(clim, str):
        return percentile_limits(data, *percentiles_of(clim))
    return float(clim[0]), float(clim[1])


def limits(data: Any, clim: Any = DEFAULT, kind: str = "image") -> tuple[Any, Any]:
    """The ``(vmin, vmax)`` of one image: ``clim``, else the kind's or the global default spec.

    ``clim=None`` keeps matplotlib's own scaling (``(None, None)``). The
    ``render`` kind reads ``render_clim``.
    """
    if clim is None:
        return None, None
    global_name = "render_clim" if kind == "render" else "clim"
    spec = option(kind, "clim", clim, global_name=global_name, fallback=defaults.clim)
    if stack_mode(spec, kind)[0] != "fixed":  # a stack mode as the default: its percentiles
        spec = ("p", *percentiles_of(spec))
    return resolve_clim(data, spec)


def percentiles_of(spec: Any) -> tuple[float, float]:
    """The percentiles of a ``("p", lo, hi)`` or ``"p2-98"`` spec; ``(1, 99)`` for the others."""
    if isinstance(spec, (tuple, list)) and len(spec) == 3 and spec[0] in ("p", "shared", "each"):
        return float(spec[1]), float(spec[2])
    if isinstance(spec, str) and spec not in ("shared", "each"):
        if spec.startswith("p") and "-" in spec[1:]:
            lo, hi = spec[1:].split("-", 1)
            return float(lo), float(hi)
        raise ValueError(
            f"unknown colour-limit spec {spec!r}: use 'p2-98', ('p', lo, hi) or (vmin, vmax)"
        )
    return 1.0, 99.0


def stack_mode(clim: Any = DEFAULT, kind: str = "image") -> tuple[str, tuple[float, float]]:
    """How a stack is scaled: ``("shared" | "each", (lo, hi))`` or ``("fixed", (vmin, vmax))``.

    The grammar of ``clim`` for tiles, the browsers and animations: ``"shared"``
    (one scale for every frame, the default percentiles), ``("shared", lo, hi)``,
    ``"each"`` (every frame on its own scale), ``("each", lo, hi)``,
    ``("p", lo, hi)`` and ``"p2-98"`` (shared, those percentiles), ``(vmin, vmax)``
    (fixed) and ``None`` (each). Left at :data:`DEFAULT`, the kind's or the
    global ``clim`` decides (``render_clim`` for the ``render`` kind).
    """
    global_name = "render_clim" if kind == "render" else "clim"
    default_spec = getattr(defaults, global_name)
    if clim is DEFAULT:
        clim = option(kind, "clim", DEFAULT, global_name=global_name, fallback=default_spec)
    if clim is None or (isinstance(clim, str) and clim == "each"):
        return "each", percentiles_of(default_spec)
    if isinstance(clim, str) and clim == "shared":
        return "shared", percentiles_of(default_spec)
    if isinstance(clim, (tuple, list)) and len(clim) == 3 and clim[0] in ("shared", "each", "p"):
        return ("each" if clim[0] == "each" else "shared"), percentiles_of(clim)
    if isinstance(clim, str):
        return "shared", percentiles_of(clim)
    return "fixed", (float(clim[0]), float(clim[1]))


# ------------------------------------------------------------------ figure helpers
def colormap(kind: str, cmap: Any = DEFAULT) -> Any:
    """The colormap of a kind: the argument, the override, else the global ``cmap``.

    COM maps read ``diverging_cmap`` instead. ``None`` leaves matplotlib's
    ``rcParams["image.cmap"]`` in charge; plotly
    callers substitute it themselves.
    """
    global_name = "diverging_cmap" if kind == "com" else "cmap"
    return option(kind, "cmap", cmap, global_name=global_name)


def colorbar(
    fig: Any, im: Any, ax: Any, kind: str = "image", show: Any = DEFAULT, label: str | None = None
) -> Any:
    """Draw the colourbar of ``im`` when the style says so; returns it, or ``None``."""
    if not option(kind, "colorbar", show, fallback=True):
        return None
    return fig.colorbar(
        im,
        ax=ax,
        fraction=option(kind, "colorbar_fraction", fallback=0.046),
        pad=option(kind, "colorbar_pad", fallback=0.04),
        label=label or "",
    )


def figure_size(
    kind: str,
    figsize: tuple[float, float] | None = None,
    rows: int = 1,
    cols: int = 1,
    panel: tuple[float, float] | None = None,
) -> tuple[float, float] | None:
    """The figure size: the argument, else the kind's ``figsize``, else the global one or the panes.

    A single-panel figure (``panel=None``) follows the global ``figsize`` and
    ends with ``None`` (the function's own size, or ``rcParams``). A panel
    figure gives ``panel`` inches to each of its ``rows x cols`` panes unless
    the kind or the global ``panel_size`` says otherwise.
    """
    if figsize is not None:
        return float(figsize[0]), float(figsize[1])
    whole = defaults.kinds.get(kind, {}).get("figsize")
    if whole is None and panel is None:
        whole = defaults.figsize
    if whole is not None:
        return float(whole[0]), float(whole[1])
    if panel is None:
        return None
    pane = option(kind, "panel_size", fallback=panel)
    return float(pane[0]) * cols, float(pane[1]) * rows


def title_font(kind: str) -> dict[str, Any]:
    """``{"fontsize": ...}`` for a pane title, or ``{}`` to leave matplotlib's own size."""
    size = option(kind, "title_size")
    return {"fontsize": size} if size is not None else {}


def apply(
    fig: Any,
    clim: Any = None,
    cmap: Any = None,
    figsize: tuple[float, float] | None = None,
    colorbar_label: str | None = None,
) -> Any:
    """Restyle a figure already drawn: every image gets ``clim`` (a spec) and ``cmap``.

    ``clim`` is a :func:`resolve_clim` spec applied to each image's own data in
    the units shown (after ``log``), ``figsize`` the new size in inches, and
    ``colorbar_label`` the label of every colourbar. Returns the figure.
    """
    for ax in fig.axes:
        for im in ax.images:
            if clim is not None:
                im.set_clim(*resolve_clim(im.get_array(), clim))
            if cmap is not None:
                im.set_cmap(cmap)
            if colorbar_label is not None and getattr(im, "colorbar", None) is not None:
                im.colorbar.set_label(colorbar_label)
    if figsize is not None:
        fig.set_size_inches(*figsize)
    return fig


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
