"""One-dimensional curve tools with uncertainties: peaks, widths, smoothing, backgrounds.

Every function takes plain arrays ``x``, ``y`` (and optionally ``sigma``, the
standard uncertainty of every ``y``) and returns arrays; the workspace
operations of :mod:`bexa.workspace.ops` wrap them for curve objects. Smoothing
works along the last axis of any array, so a map of curves is smoothed in one
call.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from bexa.core.backend import to_host

__all__ = [
    "PEAK_METHODS",
    "SMOOTH_METHODS",
    "background",
    "cumulative_integral",
    "derivative",
    "fwhm",
    "interp_onto",
    "merge_points",
    "peak_position",
    "resample",
    "smooth",
    "stitch",
]

PEAK_METHODS = ("maximum", "refined", "com")
SMOOTH_METHODS = ("gaussian", "moving average", "savitzky-golay", "median")
BACKGROUND_METHODS = ("linear", "constant", "polynomial", "minimum")


def _arrays(x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(to_host(x), dtype=float), np.asarray(to_host(y), dtype=float)


def _inside(x: np.ndarray, x_range: Sequence[float | None] | None) -> np.ndarray:
    """Mask of the points within ``x_range`` (``None`` bounds are open)."""
    keep = np.isfinite(x)
    if x_range is not None:
        lo, hi = x_range
        if lo is not None:
            keep &= x >= lo
        if hi is not None:
            keep &= x <= hi
    return keep


def peak_position(
    x: Any, y: Any, method: str = "maximum", x_range: Sequence[float | None] | None = None
) -> float:
    """Where the curve peaks.

    ``"maximum"`` is the x of the highest point; ``"refined"`` the top of the
    parabola through that point and its two neighbours (between scan points);
    ``"com"`` the centre of mass of the points above half the maximum.
    ``x_range`` limits the search.
    """
    x, y = _arrays(x, y)
    keep = _inside(x, x_range) & np.isfinite(y)
    if not keep.any():
        raise ValueError("no finite point inside the search range")
    xs, ys = x[keep], y[keep]
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    top = int(np.argmax(ys))
    if method == "maximum":
        return float(xs[top])
    if method == "refined":
        if top == 0 or top == len(xs) - 1:
            return float(xs[top])
        x0, x1, x2 = xs[top - 1 : top + 2]
        y0, y1, y2 = ys[top - 1 : top + 2]
        denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
        a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
        b = (x2**2 * (y0 - y1) + x1**2 * (y2 - y0) + x0**2 * (y1 - y2)) / denom
        return float(-b / (2 * a)) if a < 0 else float(xs[top])
    if method == "com":
        base = float(np.min(ys))
        above = ys - base >= 0.5 * (ys[top] - base)
        weights = ys[above] - base
        return float(np.sum(xs[above] * weights) / np.sum(weights))
    raise ValueError(f"unknown peak method {method!r}; choose from {PEAK_METHODS}")


def fwhm(x: Any, y: Any, baseline: str = "min") -> tuple[float, float, float]:
    """Full width at half maximum, interpolated between points: ``(width, left, right)``.

    ``baseline="min"`` measures the half height above the curve's minimum,
    ``"zero"`` above zero. NaN when the curve does not fall below half on
    both sides of its maximum.
    """
    x, y = _arrays(x, y)
    order = np.argsort(x)
    x, y = x[order], y[order]
    base = float(np.nanmin(y)) if baseline == "min" else 0.0
    top = int(np.nanargmax(y))
    half = base + 0.5 * (y[top] - base)
    left_side = np.flatnonzero(y[:top] < half)
    right_side = np.flatnonzero(y[top:] < half)
    if left_side.size == 0 or right_side.size == 0:
        return float("nan"), float("nan"), float("nan")
    i = int(left_side[-1])
    j = int(right_side[0]) + top
    left = x[i] + (half - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
    right = x[j - 1] + (half - y[j - 1]) * (x[j] - x[j - 1]) / (y[j] - y[j - 1])
    return float(right - left), float(left), float(right)


def smooth(y: Any, method: str = "gaussian", width: float = 2.0, order: int = 2) -> np.ndarray:
    """Smooth along the last axis.

    ``"gaussian"``: ``width`` is the standard deviation in points;
    ``"moving average"``, ``"savitzky-golay"`` (polynomial ``order``) and
    ``"median"``: ``width`` is the window in points (made odd).
    """
    from scipy import ndimage, signal

    data = np.asarray(to_host(y), dtype=float)
    window = max(round(width), 1)
    if method == "gaussian":
        return ndimage.gaussian_filter1d(data, float(width), axis=-1, mode="nearest")
    if method == "moving average":
        return ndimage.uniform_filter1d(data, window, axis=-1, mode="nearest")
    window += 1 - window % 2  # odd
    if method == "savitzky-golay":
        window = max(window, order + 2 + (order + 2 + 1) % 2)
        return signal.savgol_filter(data, window, int(order), axis=-1, mode="interp")
    if method == "median":
        size = [1] * data.ndim
        size[-1] = window
        return ndimage.median_filter(data, size=size, mode="nearest")
    raise ValueError(f"unknown smoothing {method!r}; choose from {SMOOTH_METHODS}")


def derivative(x: Any, y: Any, sigma: Any = None) -> tuple[np.ndarray, np.ndarray | None]:
    """``dy/dx`` by central differences (one-sided at the ends), with its uncertainty."""
    x, y = _arrays(x, y)
    dy = np.gradient(y, x)
    if sigma is None:
        return dy, None
    s = np.asarray(to_host(sigma), dtype=float)
    ds = np.empty_like(s)
    ds[1:-1] = np.hypot(s[2:], s[:-2]) / np.abs(x[2:] - x[:-2])
    ds[0] = np.hypot(s[1], s[0]) / abs(x[1] - x[0])
    ds[-1] = np.hypot(s[-1], s[-2]) / abs(x[-1] - x[-2])
    return dy, ds


def cumulative_integral(x: Any, y: Any, sigma: Any = None) -> tuple[np.ndarray, np.ndarray | None]:
    """The running trapezoid integral from the first point, with its uncertainty."""
    x, y = _arrays(x, y)
    steps = np.diff(x)
    area = np.concatenate(([0.0], np.cumsum(0.5 * steps * (y[1:] + y[:-1]))))
    if sigma is None:
        return area, None
    s = np.asarray(to_host(sigma), dtype=float)
    weight = np.zeros_like(x)  # each point's share of the area up to the end
    weight[:-1] += 0.5 * steps
    weight[1:] += 0.5 * steps
    return area, np.sqrt(np.cumsum((weight * s) ** 2))


def background(
    x: Any,
    y: Any,
    method: str = "linear",
    regions: Sequence[tuple[float, float]] | None = None,
    order: int = 2,
) -> np.ndarray:
    """A background under the curve, fitted to its background ``regions``.

    ``regions`` are ``(low, high)`` x ranges holding background only (default:
    the first and last tenth of the points). ``"constant"``, ``"linear"`` and
    ``"polynomial"`` (of ``order``) are least-squares fits there; ``"minimum"``
    is the curve's minimum.
    """
    x, y = _arrays(x, y)
    if method == "minimum":
        return np.full_like(y, np.nanmin(y))
    if regions:
        mask = np.zeros(x.shape, dtype=bool)
        for lo, hi in regions:
            mask |= (x >= min(lo, hi)) & (x <= max(lo, hi))
    else:
        tenth = max(len(x) // 10, 2)
        order_x = np.argsort(x)
        mask = np.zeros(x.shape, dtype=bool)
        mask[order_x[:tenth]] = True
        mask[order_x[-tenth:]] = True
    mask &= np.isfinite(y)
    degree = {"constant": 0, "linear": 1, "polynomial": int(order)}.get(method)
    if degree is None:
        raise ValueError(f"unknown background {method!r}; choose from {BACKGROUND_METHODS}")
    if mask.sum() <= degree:
        raise ValueError(f"{int(mask.sum())} background points cannot fix a degree-{degree} fit")
    centre = float(np.mean(x[mask]))
    coefficients = np.polyfit(x[mask] - centre, y[mask], degree)
    return np.polyval(coefficients, x - centre)


def interp_onto(x_from: Any, y: Any, x_to: Any) -> np.ndarray:
    """``y`` given at ``x_from`` interpolated linearly at ``x_to``; NaN outside the range."""
    xf, yf = _arrays(x_from, y)
    xt = np.asarray(to_host(x_to), dtype=float)
    order = np.argsort(xf)
    return np.interp(xt, xf[order], yf[order], left=np.nan, right=np.nan)


def resample(x: Any, y: Any, new_x: Any, sigma: Any = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Values (and uncertainties) interpolated linearly onto ``new_x``."""
    values = interp_onto(x, y, new_x)
    if sigma is None:
        return values, None
    return values, interp_onto(x, sigma, new_x)


def merge_points(
    x: Any, y: Any, sigma: Any = None, tolerance: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Sort by x and average the points closer than ``tolerance`` (default 1e-9 of the span).

    Uncertainties combine as for a mean: ``sqrt(sum(s^2)) / n``.
    """
    x, y = _arrays(x, y)
    order = np.argsort(x)
    x, y = x[order], y[order]
    s = None if sigma is None else np.asarray(to_host(sigma), dtype=float)[order]
    span = float(np.ptp(x)) if x.size else 0.0
    tol = tolerance if tolerance is not None else 1e-9 * max(span, 1.0)
    group = np.concatenate(([0], np.cumsum(np.diff(x) > tol)))
    counts = np.bincount(group)
    mx = np.bincount(group, weights=x) / counts
    my = np.bincount(group, weights=y) / counts
    ms = None if s is None else np.sqrt(np.bincount(group, weights=s**2)) / counts
    return mx, my, ms


def stitch(
    curves: Sequence[tuple[Any, Any, Any]], scale: bool = True, overlap: str = "average"
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """One curve from curves of neighbouring ranges ``[(x, y, sigma or None), ...]``.

    Curves are taken in order of their first x. With ``scale`` every curve is
    multiplied so that it matches the stitched part where they overlap (the
    ratio of their means there). Where they overlap, ``overlap`` keeps the
    points of both (``"average"``: points at the same x are averaged), of the
    ``"first"`` curve or of the ``"last"``.
    """
    if not curves:
        raise ValueError("no curves to stitch")
    parts = []
    for x, y, s in curves:
        xa, ya = _arrays(x, y)
        sa = None if s is None else np.asarray(to_host(s), dtype=float)
        order = np.argsort(xa)
        parts.append((xa[order], ya[order], None if sa is None else sa[order]))
    parts.sort(key=lambda p: p[0][0])
    with_sigma = all(p[2] is not None for p in parts)
    sx, sy, ss = parts[0][0], parts[0][1], parts[0][2]
    for x, y, s in parts[1:]:
        shared = (x >= sx[0]) & (x <= sx[-1])
        if scale and shared.sum() >= 1:
            reference = interp_onto(sx, sy, x[shared])
            ok = np.isfinite(reference)
            mean_new = float(np.mean(y[shared][ok])) if ok.any() else 0.0
            if ok.any() and mean_new != 0:
                factor = float(np.mean(reference[ok])) / mean_new
                y = y * factor
                s = None if s is None else s * abs(factor)
        if overlap == "first":
            keep = ~shared
        elif overlap == "last":
            old = sx < x[0]
            sx, sy = sx[old], sy[old]
            ss = None if ss is None else ss[old]
            keep = np.ones(x.shape, dtype=bool)
        elif overlap == "average":
            keep = np.ones(x.shape, dtype=bool)
        else:
            raise ValueError(f"overlap must be average, first or last; got {overlap!r}")
        sx = np.concatenate((sx, x[keep]))
        sy = np.concatenate((sy, y[keep]))
        if with_sigma and ss is not None and s is not None:
            ss = np.concatenate((ss, s[keep]))
        sx, sy, ss = merge_points(sx, sy, ss if with_sigma else None, tolerance=None)
    return sx, sy, ss if with_sigma else None
