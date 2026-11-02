"""Maps for dark-field microscopy and topography: peak fits per pixel, KAM, orientation, strain.

All functions are vectorised over the pixels (no Python loop per pixel) and
accept numpy or cupy arrays where they reduce a volume.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from bexa.core.backend import array_module, to_host

__all__ = [
    "gaussian_peak_maps",
    "hsv_to_rgb",
    "kam",
    "orientation_rgb",
    "strain_map",
]

FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))


def gaussian_peak_maps(
    volume: Any, values: Any, axis: int = 0, threshold: float = 0.1
) -> dict[str, Any]:
    """A Gaussian fitted to the rocking curve of every pixel, all pixels at once.

    The logarithm of a Gaussian is a parabola, so ``ln I = a + b x + c x^2`` is
    fitted by weighted least squares (weights ``I^2``, Guo's form of Caruana's
    algorithm) over the points above ``threshold`` of each pixel's maximum.
    That is a 3 x 3 linear system per pixel, solved for every pixel together.
    Returns ``center``, ``fwhm``, ``amplitude`` and ``points`` (points used);
    pixels without a peak (``c >= 0`` or fewer than three points) are NaN.
    """
    xp = array_module(volume)
    data = xp.moveaxis(xp.asarray(volume, dtype=np.float64), axis, 0)
    x = xp.asarray(values, dtype=np.float64)
    if x.shape[0] != data.shape[0]:
        raise ValueError(f"{x.shape[0]} motor values for {data.shape[0]} points")
    x0 = float(to_host(x.mean()))
    xs = x - x0  # centred for conditioning
    peak = data.max(axis=0)
    used = (data > threshold * peak[None]) & (data > 0)
    weights = xp.where(used, data * data, 0.0)
    weighted_logs = weights * xp.log(xp.where(used, data, 1.0))
    # motor-weighted sums over the first axis: tensordot, no volume-sized temporaries
    s = [xp.tensordot(xs**k, weights, axes=(0, 0)) for k in range(5)]
    t = [xp.tensordot(xs**k, weighted_logs, axes=(0, 0)) for k in range(3)]
    # Cramer's rule on [[s0 s1 s2] [s1 s2 s3] [s2 s3 s4]] (a b c) = (t0 t1 t2)
    det = (
        s[0] * (s[2] * s[4] - s[3] ** 2)
        - s[1] * (s[1] * s[4] - s[2] * s[3])
        + s[2] * (s[1] * s[3] - s[2] ** 2)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        a = (
            t[0] * (s[2] * s[4] - s[3] ** 2)
            - s[1] * (t[1] * s[4] - s[3] * t[2])
            + s[2] * (t[1] * s[3] - s[2] * t[2])
        ) / det
        b = (
            s[0] * (t[1] * s[4] - s[3] * t[2])
            - t[0] * (s[1] * s[4] - s[2] * s[3])
            + s[2] * (s[1] * t[2] - t[1] * s[2])
        ) / det
        c = (
            s[0] * (s[2] * t[2] - t[1] * s[3])
            - s[1] * (s[1] * t[2] - t[1] * s[2])
            + t[0] * (s[1] * s[3] - s[2] ** 2)
        ) / det
        good = (c < 0) & (used.sum(axis=0) >= 3)
        sigma = xp.sqrt(-1.0 / (2.0 * c))
        center = -b / (2.0 * c) + x0
        amplitude = xp.exp(a - b * b / (4.0 * c))
    nan = xp.nan
    return {
        "center": xp.where(good, center, nan),
        "fwhm": xp.where(good, FWHM_PER_SIGMA * sigma, nan),
        "amplitude": xp.where(good, amplitude, nan),
        "points": used.sum(axis=0),
    }


def kam(com_a: Any, com_b: Any = None, size: int = 3) -> np.ndarray:
    """Kernel average misorientation: the mean tilt difference to the neighbours of each pixel.

    ``com_a`` (and ``com_b`` for a second tilt axis) are centre-of-mass maps in
    degrees; the misorientation to a neighbour is ``hypot(da, db)``. The kernel
    is ``size`` x ``size`` (odd); NaN neighbours are skipped.
    """
    a = np.asarray(to_host(com_a), dtype=float)
    b = None if com_b is None else np.asarray(to_host(com_b), dtype=float)
    half = int(size) // 2
    pad = ((half, half), (half, half))
    pa = np.pad(a, pad, constant_values=np.nan)
    pb = None if b is None else np.pad(b, pad, constant_values=np.nan)
    h, w = a.shape
    total = np.zeros_like(a)
    count = np.zeros_like(a)
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            if dy == 0 and dx == 0:
                continue
            da = pa[half + dy : half + dy + h, half + dx : half + dx + w] - a
            diff = np.abs(da)
            if pb is not None and b is not None:
                db = pb[half + dy : half + dy + h, half + dx : half + dx + w] - b
                diff = np.hypot(da, db)
            ok = np.isfinite(diff)
            total += np.where(ok, diff, 0.0)
            count += ok
    with np.errstate(invalid="ignore", divide="ignore"):
        return total / count


def hsv_to_rgb(hue: Any, saturation: Any, value: Any) -> np.ndarray:
    """Hue, saturation and value in [0, 1] to an ``(..., 3)`` RGB array (numpy only)."""
    h = np.asarray(hue, dtype=float) % 1.0
    s = np.clip(np.asarray(saturation, dtype=float), 0.0, 1.0)
    v = np.clip(np.asarray(value, dtype=float), 0.0, 1.0)
    i = np.floor(h * 6.0).astype(int) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    choices_r = [v, q, p, p, t, v]
    choices_g = [t, v, v, q, p, p]
    choices_b = [p, p, t, v, v, q]
    r = np.choose(i, choices_r)
    g = np.choose(i, choices_g)
    b = np.choose(i, choices_b)
    return np.stack([r, g, b], axis=-1)


def orientation_rgb(
    com_a: Any,
    com_b: Any,
    center: tuple[float, float] | None = None,
    span: float | None = None,
    intensity: Any = None,
) -> np.ndarray:
    """Two tilt maps as one colour image: hue the direction, saturation the size of the tilt.

    The tilt is measured from ``center`` (default: the medians) and is fully
    saturated at ``span`` (default: the 95th percentile of the tilt). With
    ``intensity`` the brightness follows it (percentiles 1-99), else it is 1.
    NaN pixels are black.
    """
    a = np.asarray(to_host(com_a), dtype=float)
    b = np.asarray(to_host(com_b), dtype=float)
    ca, cb = center if center is not None else (np.nanmedian(a), np.nanmedian(b))
    da, db = a - ca, b - cb
    radius = np.hypot(da, db)
    reach = span if span is not None else float(np.nanpercentile(radius, 95)) or 1.0
    hue = (np.arctan2(db, da) / (2 * math.pi)) % 1.0
    if intensity is not None:
        weight = np.asarray(to_host(intensity), dtype=float)
        lo, hi = np.nanpercentile(weight, [1, 99])
        value = np.clip((weight - lo) / ((hi - lo) or 1.0), 0, 1)
    else:
        value = np.ones_like(a)
    rgb = hsv_to_rgb(hue, radius / reach, value)
    rgb[~np.isfinite(radius)] = 0.0
    return rgb


def strain_map(
    com: Any, reference: float | None = None, kind: str = "theta", theta_deg: float | None = None
) -> np.ndarray:
    """Lattice strain from a centre-of-mass map.

    ``kind="theta"``: ``com`` is a Bragg angle (or 2theta with
    ``kind="two_theta"``) in degrees and ``strain = -(theta - theta_0) cot
    theta_0``; ``kind="energy"``: ``com`` is an energy and ``strain = -(E -
    E_0) / E_0`` at a fixed angle. ``reference`` is the unstrained value
    (default: the median); ``theta_deg`` the Bragg angle for the cotangent
    (default: the reference itself, for angle maps).
    """
    values = np.asarray(to_host(com), dtype=float)
    ref = float(np.nanmedian(values)) if reference is None else float(reference)
    if kind == "energy":
        return -(values - ref) / ref
    if kind not in ("theta", "two_theta"):
        raise ValueError(f"kind must be theta, two_theta or energy, not {kind!r}")
    half = 0.5 if kind == "two_theta" else 1.0
    theta0 = math.radians(theta_deg if theta_deg is not None else ref * half)
    return -np.radians((values - ref) * half) / math.tan(theta0)
