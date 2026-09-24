"""Vector fields, gradients, edges and the strain integration of the ``Yue idea`` notebook.

Two centre-of-mass maps (``mu`` and ``chi``, or ``mu`` and energy) are read
as the components of a field: its angle and magnitude, the edges where the
angle changes sharply, the compatibility of the two strain components and
the displacement obtained by integrating them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bexa.core.backend import to_host

__all__ = [
    "angle_magnitude",
    "compatibility",
    "edge_mask",
    "gradients",
    "integrate_displacement",
    "integrate_second",
    "vector_field",
]


def vector_field(
    com_a: Any, com_b: Any, center: str | tuple[float, float] = "median"
) -> tuple[np.ndarray, np.ndarray]:
    """``(u, v)`` components: the two COM maps minus their centre (median, mean or given)."""
    a = np.asarray(to_host(com_a), dtype=float)
    b = np.asarray(to_host(com_b), dtype=float)
    if center == "median":
        ca, cb = np.nanmedian(a), np.nanmedian(b)
    elif center == "mean":
        ca, cb = np.nanmean(a), np.nanmean(b)
    elif center == "none":
        ca, cb = 0.0, 0.0
    else:
        ca, cb = center
    return a - ca, b - cb


def angle_magnitude(u: Any, v: Any) -> tuple[np.ndarray, np.ndarray]:
    """``atan2(v, u)`` and ``hypot(u, v)`` of a field."""
    u = np.asarray(to_host(u), dtype=float)
    v = np.asarray(to_host(v), dtype=float)
    return np.arctan2(v, u), np.hypot(u, v)


def gradients(field: Any) -> tuple[np.ndarray, np.ndarray]:
    """``(d/dy, d/dx)`` of a map with ``numpy.gradient``."""
    data = np.asarray(to_host(field), dtype=float)
    return np.gradient(data, axis=0), np.gradient(data, axis=1)


def edge_mask(
    field: Any, method: str = "sobel", sigma: float = 0.9, threshold: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Boundaries where a map changes sharply: ``(mask, edge_strength)``.

    ``method`` ``"sobel"`` uses the gradient magnitude, ``"laplace"`` the
    absolute Laplacian of Gaussian (``sigma``). NaN pixels never count as edges.
    """
    from scipy.ndimage import gaussian_laplace, sobel

    data = np.asarray(to_host(field), dtype=float)
    finite = np.isfinite(data)
    filled = np.where(finite, data, np.nanmean(data) if finite.any() else 0.0)
    if method == "sobel":
        strength = np.hypot(sobel(filled, axis=0), sobel(filled, axis=1))
    elif method == "laplace":
        strength = np.abs(gaussian_laplace(filled, sigma=sigma))
    else:
        raise ValueError(f"unknown method {method!r}; use sobel or laplace")
    strength[~finite] = np.nan
    mask = strength > threshold
    mask[~finite] = False
    return mask, strength


def compatibility(eyz: Any, ezx: Any) -> dict[str, np.ndarray]:
    """Partial derivatives of the two shear-strain maps and their compatibility term.

    Returns ``eyz_py, eyz_px, ezx_py, ezx_px`` (``numpy.gradient`` along y and
    x) and ``dev2nd = (ezx_py - eyz_px) / 2``, the quantity the notebook
    integrated to a displacement gradient.
    """
    eyz_py, eyz_px = gradients(eyz)
    ezx_py, ezx_px = gradients(ezx)
    return {
        "eyz_py": eyz_py,
        "eyz_px": eyz_px,
        "ezx_py": ezx_py,
        "ezx_px": ezx_px,
        "dev2nd": (ezx_py - eyz_px) / 2.0,
    }


def integrate_displacement(dev2nd: Any, eyz: Any) -> np.ndarray:
    """``df/dy`` by cumulative summation of ``dev2nd`` along y with the notebook's boundary rule.

    Port of the "first integration" cell: the cumulative sum along y (the
    notebook also computed a sum along x first, then overwrote it) and, row by
    row, the subtraction of the last column's value and of ``eyz`` there.
    """
    d = np.asarray(to_host(dev2nd), dtype=float)
    e = np.asarray(to_host(eyz), dtype=float)
    dfdy = np.cumsum(d, axis=0)
    return dfdy - dfdy[:, -1:] - e[:, -1:]


def integrate_second(ezx: Any, eyz: Any) -> dict[str, np.ndarray]:
    """The notebook's "second integration": ``f1`` (cumsum of ``ezx`` along y), ``f2``, ``f``.

    ``f2`` is the cumsum of ``eyz`` along x and ``f[j] = f1[j] - f1[j, 0] + f2[j, 0]``
    row by row, as written in the notebook.
    """
    a = np.asarray(to_host(ezx), dtype=float)
    b = np.asarray(to_host(eyz), dtype=float)
    f1 = np.cumsum(a, axis=0)
    f2 = np.cumsum(b, axis=1)
    f = f1 - f1[:, :1] + f2[:, :1]
    return {"f": f, "f1": f1, "f2": f2}
