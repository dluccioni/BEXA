"""Per-pixel rocking-curve statistics on volumes already in memory.

The streaming accumulators of :mod:`bexa.core.reductions` do the same job for
data that does not fit in memory; these functions take a ``(motor, y, x)``
(or any) volume plus the motor values and return maps. Everything is
vectorised with ``tensordot`` (the notebook loops over pixels took hours).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bexa.core.backend import array_module, ndimage_module
from bexa.core.units import FWHM_PER_SIGMA

__all__ = [
    "argmax_motor",
    "combine_axes",
    "energy_com",
    "moments",
    "motor_com",
    "rocking_curve_stats",
    "stat_images",
    "weighted_median",
    "weighted_quantile",
]


def _prepare(
    volume: Any, values: Any, axis: int, sigma: float, clip: float | None
) -> tuple[Any, Any, Any]:
    xp = array_module(volume)
    w = xp.moveaxis(xp.asarray(volume, dtype=np.float64), axis, 0)
    if sigma:
        ndi = ndimage_module(volume)
        w = xp.stack([ndi.gaussian_filter(f, sigma) for f in w])
    if clip is not None:
        w = xp.clip(w, clip, None)
    v = xp.asarray(values, dtype=np.float64)
    if v.shape[0] != w.shape[0]:
        raise ValueError(f"{v.shape[0]} motor values for {w.shape[0]} frames")
    return xp, w, v


def motor_com(
    volume: Any, values: Any, axis: int = 0, sigma: float = 0.0, clip: float | None = 1e-10
) -> Any:
    """Intensity-weighted mean motor position per pixel (the DFXM centre-of-mass map).

    ``sigma`` smooths every frame first (v9 used 3 pixels); ``clip`` floors the
    weights so empty pixels do not divide by zero.
    """
    xp, w, v = _prepare(volume, values, axis, sigma, clip)
    return xp.tensordot(v, w, axes=(0, 0)) / w.sum(axis=0)


energy_com = motor_com


def moments(
    volume: Any, values: Any, axis: int = 0, sigma: float = 0.0, clip: float | None = 1e-10
) -> dict[str, Any]:
    """COM, standard deviation, FWHM, skewness and excess kurtosis maps (darfix-style)."""
    xp, w, v = _prepare(volume, values, axis, sigma, clip)
    total = w.sum(axis=0)
    com = xp.tensordot(v, w, axes=(0, 0)) / total
    delta = (
        v[:, None, None] - com[None]
        if w.ndim == 3
        else v.reshape((-1,) + (1,) * (w.ndim - 1)) - com[None]
    )
    m2 = (delta**2 * w).sum(axis=0) / total
    m3 = (delta**3 * w).sum(axis=0) / total
    m4 = (delta**4 * w).sum(axis=0) / total
    std = xp.sqrt(m2)
    with np.errstate(invalid="ignore", divide="ignore"):
        skew = m3 / std**3
        kurtosis = m4 / m2**2 - 3.0
    return {
        "com": com,
        "std": std,
        "fwhm": FWHM_PER_SIGMA * std,
        "skew": skew,
        "kurtosis": kurtosis,
        "total": total,
    }


def argmax_motor(volume: Any, values: Any, axis: int = 0) -> Any:
    """Motor value of the brightest frame per pixel."""
    xp = array_module(volume)
    idx = xp.argmax(xp.moveaxis(volume, axis, 0), axis=0)
    return xp.asarray(values, dtype=np.float64)[idx]


def weighted_quantile(values: Any, weights: Any, q: float = 0.5, axis: int = 0) -> Any:
    """Per-pixel weighted quantile of the motor values along ``axis``.

    Vectorised form of the notebook's ``weighted_median`` loop: the weights
    are sorted once by motor value, and the first motor value whose cumulative
    weight reaches ``q`` of the total is returned. (The notebook divided the
    total by 1.5, which is ``q = 2/3``.)
    """
    xp = array_module(weights)
    w = xp.moveaxis(xp.asarray(weights, dtype=np.float64), axis, 0)
    v = xp.asarray(values, dtype=np.float64)
    order = xp.argsort(v)
    v_sorted = v[order]
    cumulative = xp.cumsum(w[order], axis=0)
    cutoff = cumulative[-1] * q
    idx = (cumulative >= cutoff[None]).argmax(axis=0)
    return v_sorted[idx]


def weighted_median(values: Any, weights: Any, axis: int = 0) -> Any:
    return weighted_quantile(values, weights, 0.5, axis)


def combine_axes(mu: Any, phi: Any, sign: float = -1.0) -> Any:
    """The combined tilt axis ``mu + sign * phi`` used when stitching mu scans."""
    return np.asarray(mu, dtype=float) + sign * np.asarray(phi, dtype=float)


def stat_images(stack: Any, values: Any) -> dict[str, Any]:
    """Mean image, per-pixel motor value closest to the mean, and standard deviation image.

    Port of the ``Topography`` notebook's ``stat_images``.
    """
    xp = array_module(stack)
    mean = stack.mean(axis=0)
    closest = xp.abs(stack - mean[None]).argmin(axis=0)
    return {
        "mean": mean,
        "com": xp.asarray(values, dtype=np.float64)[closest],
        "std": stack.std(axis=0),
    }


def rocking_curve_stats(stack: Any, qmin: float = 0.0, qmax: float = 100.0) -> dict[str, Any]:
    """Per-frame minimum, maximum, sum and the mean of the pixels inside ``[qmin, qmax]``.

    The ``Topography`` notebook's ``rocking_curves``; its mask combined the two
    limits with ``and`` (which excludes nothing), here pixels outside the
    percentile window are left out of the mean as intended.
    """
    xp = array_module(stack)
    flat = stack.reshape(stack.shape[0], -1)
    lo = xp.percentile(flat, qmin, axis=1)
    hi = xp.percentile(flat, qmax, axis=1)
    inside = (flat >= lo[:, None]) & (flat <= hi[:, None])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (flat * inside).sum(axis=1) / inside.sum(axis=1)
    return {"min": lo, "max": hi, "mean": mean, "sum": flat.sum(axis=1)}
