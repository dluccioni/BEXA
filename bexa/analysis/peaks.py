"""Per-frame peak statistics on the detector: centre of mass, moments, Gaussian fits.

The centre of mass follows ``scipy.ndimage.center_of_mass`` (intensity-weighted
pixel coordinates, no clipping), evaluated for a whole stack at once. Moments
are the raw central moments the legacy plots used, ``sum((x - com)^n I) / sum(I)``,
computed from the two projections of every frame, which is exact and avoids
building coordinate grids.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from bexa._log import get_logger
from bexa.core.backend import array_module, to_host
from bexa.core.roi import ROI

log = get_logger(__name__)

MOMENT_NAMES = {2: "var", 3: "skew", 4: "kurt"}

__all__ = [
    "find_peaks_1d",
    "fit_stack",
    "frame_com",
    "frame_moments",
    "frame_stats",
    "gaussian2d",
    "gaussian2d_fit",
]


def _window(frames: Any, roi: ROI | None) -> Any:
    if roi is None:
        return frames
    ys, xs = roi.pixel_slices(frames.shape[-2:])
    return frames[..., ys, xs]


def _as_stack(frames: Any) -> Any:
    if frames.ndim == 2:
        return frames[None]
    if frames.ndim > 3:
        return frames.reshape(-1, *frames.shape[-2:])
    return frames


def frame_com(frames: Any, roi: ROI | None = None) -> tuple[Any, Any]:
    """``(com_y, com_x)`` of every frame in pixels of the window (like ``center_of_mass``)."""
    xp = array_module(frames)
    f = _as_stack(_window(frames, roi))
    h, w = f.shape[-2:]
    yy = xp.arange(h, dtype=np.float64)
    xx = xp.arange(w, dtype=np.float64)
    total = f.sum(axis=(1, 2), dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):  # empty frames give NaN, like scipy
        com_y = xp.tensordot(f.sum(axis=2, dtype=np.float64), yy, axes=(1, 0)) / total
        com_x = xp.tensordot(f.sum(axis=1, dtype=np.float64), xx, axes=(1, 0)) / total
    return com_y, com_x


def frame_moments(
    frames: Any,
    orders: Sequence[int] = (2, 3),
    roi: ROI | None = None,
    centre: tuple[Any, Any] | None = None,
) -> dict[str, Any]:
    """Raw central moments about the frame COM, per axis.

    Returns ``com_y``, ``com_x`` and, per order ``n``, ``var_y``/``var_x`` (2),
    ``skew_y``/``skew_x`` (3), ``kurt_y``/``kurt_x`` (4) or ``moment{n}_y``/``_x``,
    each ``sum((coord - com)^n I) / sum(I)`` as in the legacy plots (not normalised).
    """
    xp = array_module(frames)
    f = _as_stack(_window(frames, roi))
    h, w = f.shape[-2:]
    com_y, com_x = centre if centre is not None else frame_com(f)
    total = f.sum(axis=(1, 2), dtype=np.float64)
    proj_y = f.sum(axis=2, dtype=np.float64)  # (n, h)
    proj_x = f.sum(axis=1, dtype=np.float64)  # (n, w)
    dy = xp.arange(h, dtype=np.float64)[None, :] - xp.asarray(com_y)[:, None]
    dx = xp.arange(w, dtype=np.float64)[None, :] - xp.asarray(com_x)[:, None]
    out: dict[str, Any] = {"com_y": com_y, "com_x": com_x}
    for order in orders:
        name = MOMENT_NAMES.get(order, f"moment{order}")
        out[f"{name}_y"] = (dy**order * proj_y).sum(axis=1) / total
        out[f"{name}_x"] = (dx**order * proj_x).sum(axis=1) / total
    return out


def frame_stats(
    frames: Any, roi: ROI | None = None, orders: Sequence[int] = (2, 3, 4)
) -> dict[str, Any]:
    """COM, moments, total and maximum of every frame in one call."""
    f = _as_stack(_window(frames, roi))
    out = frame_moments(f, orders)
    out["total"] = f.sum(axis=(1, 2), dtype=np.float64)
    out["max"] = f.max(axis=(1, 2))
    return out


def gaussian2d(
    xy: tuple[Any, Any],
    amp: float,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
    offset: float,
) -> np.ndarray:
    """Elliptical Gaussian with an offset, flattened for ``curve_fit``."""
    x, y = xy
    g = amp * np.exp(-(((x - x0) ** 2) / (2 * sigma_x**2) + ((y - y0) ** 2) / (2 * sigma_y**2)))
    return (g + offset).ravel()


def gaussian2d_fit(
    image: Any,
    p0: Sequence[float] | None = None,
    sigma0: tuple[float, float] = (20.0, 10.0),
    **kwargs: Any,
) -> dict[str, float]:
    """Fit :func:`gaussian2d` to one image.

    The initial guess is the legacy notebook's: amplitude = max, centre = COM,
    sigmas = ``sigma0``, offset = min. Returns the parameters plus ``success``
    (0 when the fit did not converge; the parameters are then NaN).
    """
    from scipy.optimize import curve_fit

    img = np.asarray(to_host(image), dtype=float)
    h, w = img.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    if p0 is None:
        com_y, com_x = frame_com(img[None])
        p0 = [float(img.max()), float(com_x[0]), float(com_y[0]), *sigma0, float(img.min())]
    names = ("amp", "x0", "y0", "sigma_x", "sigma_y", "offset")
    failed = {**dict.fromkeys(names, np.nan), "success": 0.0}
    if not np.all(np.isfinite(p0)):  # an empty frame has no COM to start from
        return failed
    try:
        popt, _ = curve_fit(gaussian2d, (x, y), img.ravel(), p0=list(p0), **kwargs)
    except (RuntimeError, ValueError) as exc:
        log.debug("gaussian2d_fit failed: %s", exc)
        return failed
    return {**{n: float(v) for n, v in zip(names, popt, strict=True)}, "success": 1.0}


def fit_stack(frames: Any, roi: ROI | None = None, **kwargs: Any) -> dict[str, np.ndarray]:
    """:func:`gaussian2d_fit` for every frame; one array per parameter."""
    f = _as_stack(_window(np.asarray(to_host(frames)), roi))
    results = [gaussian2d_fit(img, **kwargs) for img in f]
    return {name: np.array([r[name] for r in results]) for name in results[0]}


def find_peaks_1d(curve: Any, n_peaks: int = 1, **kwargs: Any) -> np.ndarray:
    """Indices of the ``n_peaks`` most prominent peaks of a curve (``scipy.signal.find_peaks``)."""
    from scipy.signal import find_peaks

    y = np.asarray(to_host(curve), dtype=float)
    kwargs.setdefault("prominence", 0.0)
    idx, props = find_peaks(y, **kwargs)
    if idx.size == 0:
        return idx
    order = np.argsort(props["prominences"])[::-1]
    return idx[order][:n_peaks]
