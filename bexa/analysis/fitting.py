"""Rocking-curve fitting with lmfit: sums of Gaussians, Voigt profiles, reports.

Port of the ``Topography`` notebook cells: a sum of ``n`` Gaussians with an
optional background (constant, linear or a broad Gaussian), bounded centres
and widths, and a report with centres, sigmas, FWHM, amplitudes, R^2 and the
reduced chi-square.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from bexa.core.backend import to_host
from bexa.core.parallel import thread_map
from bexa.core.units import FWHM_PER_SIGMA

__all__ = [
    "FitResult",
    "edge",
    "fit_edge",
    "fit_multi_gaussian",
    "fit_report",
    "fit_rocking_curves",
    "fit_voigt",
    "gaussian",
    "multi_gaussian_model",
]


def gaussian(x: Any, amplitude: float, center: float, sigma: float) -> Any:
    """``amplitude * exp(-(x - center)^2 / (2 sigma^2))`` (peak height, not area)."""
    return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma**2))


@dataclass
class FitResult:
    """What a rocking-curve fit produced."""

    centers: np.ndarray
    sigmas: np.ndarray
    amplitudes: np.ndarray
    fwhm: np.ndarray
    r2: float
    reduced_chi2: float
    best_fit: np.ndarray
    background: dict[str, float] = field(default_factory=dict)
    success: bool = True
    result: Any = None  # the lmfit ModelResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "centers": self.centers.tolist(),
            "sigmas": self.sigmas.tolist(),
            "amplitudes": self.amplitudes.tolist(),
            "fwhm": self.fwhm.tolist(),
            "r2": self.r2,
            "reduced_chi2": self.reduced_chi2,
            "background": dict(self.background),
            "success": self.success,
        }


def multi_gaussian_model(n: int, background: str = "none", prefix: str = "g") -> Any:
    """lmfit model of ``n`` Gaussians (prefixes ``g1_``, ``g2_``, ...) plus a background.

    ``background``: ``"none"``, ``"constant"``, ``"linear"`` or ``"gaussian"``
    (a broad Gaussian with prefix ``bg_``, as the notebook used).
    """
    from lmfit import Model
    from lmfit.models import ConstantModel, LinearModel

    if n < 1:
        raise ValueError("at least one Gaussian is needed")
    model = Model(gaussian, prefix=f"{prefix}1_")
    for i in range(2, n + 1):
        model = model + Model(gaussian, prefix=f"{prefix}{i}_")
    if background == "gaussian":
        model = model + Model(gaussian, prefix="bg_")
    elif background == "constant":
        model = model + ConstantModel(prefix="bg_")
    elif background == "linear":
        model = model + LinearModel(prefix="bg_")
    elif background != "none":
        raise ValueError(f"unknown background {background!r}")
    return model


def _r2_and_chi2(y: np.ndarray, best: np.ndarray, n_params: int) -> tuple[float, float]:
    residuals = y - best
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    chi2 = float(np.sum((residuals / np.sqrt(np.abs(y) + 1e-8)) ** 2))
    dof = max(len(y) - n_params, 1)
    return r2, chi2 / dof


def fit_multi_gaussian(
    x: Any,
    y: Any,
    n: int | None = None,
    centers: Sequence[float] | None = None,
    sigmas: float | Sequence[float] = 0.2,
    amplitudes: Sequence[float] | None = None,
    background: str = "none",
    sigma_bounds: tuple[float, float] = (0.05, 5.0),
    center_bounds: tuple[float, float] | None = None,
    scale: bool = True,
    prefix: str = "g",
) -> FitResult:
    """Fit a sum of Gaussians to a curve.

    Parameters
    ----------
    n, centers
        Number of peaks, or their initial centres (``n`` is then their count).
        Without either, the most prominent peaks of the curve are used.
    sigmas, amplitudes
        Initial widths (one value or one per peak) and heights (default: 80 %
        of the maximum, as the notebook did).
    background
        See :func:`multi_gaussian_model`.
    scale
        Fit ``y / max(y)`` for numerical stability and scale the amplitudes back.
    """
    x = np.asarray(to_host(x), dtype=float)
    y = np.asarray(to_host(y), dtype=float)
    if centers is None:
        if n is None:
            n = 1
        from bexa.analysis.peaks import find_peaks_1d

        idx = find_peaks_1d(y, n_peaks=n)
        centers = (
            [float(x[i]) for i in idx] if idx.size else list(np.linspace(x[0], x[-1], n + 2)[1:-1])
        )
        while len(centers) < n:
            centers.append(float(x[len(x) // 2]))
    centers = list(centers)
    n = len(centers)
    factor = float(np.max(np.abs(y))) if scale and np.max(np.abs(y)) > 0 else 1.0
    y_fit = y / factor
    sigma_list = [float(sigmas)] * n if np.isscalar(sigmas) else [float(s) for s in sigmas]
    amp_list = (
        [float(a) / factor for a in amplitudes]
        if amplitudes is not None
        else [float(y_fit.max()) * 0.8] * n
    )
    lo, hi = center_bounds if center_bounds is not None else (float(x.min()), float(x.max()))

    model = multi_gaussian_model(n, background, prefix)
    params = model.make_params()
    for i, (c, s, a) in enumerate(zip(centers, sigma_list, amp_list, strict=True), start=1):
        params[f"{prefix}{i}_amplitude"].set(value=a, min=0)
        params[f"{prefix}{i}_center"].set(value=c, min=lo, max=hi)
        params[f"{prefix}{i}_sigma"].set(value=s, min=sigma_bounds[0], max=sigma_bounds[1])
    if background == "gaussian":
        params["bg_amplitude"].set(value=float(y_fit.min()), min=0)
        params["bg_center"].set(value=float(np.mean(x)))
        params["bg_sigma"].set(value=float(np.ptp(x)), min=sigma_bounds[1] / 2)
    elif background == "constant":
        params["bg_c"].set(value=float(y_fit.min()))
    elif background == "linear":
        params["bg_intercept"].set(value=float(y_fit.min()))
        params["bg_slope"].set(value=0.0)
    result = model.fit(y_fit, params, x=x)
    best = result.best_fit * factor
    r2, red_chi2 = _r2_and_chi2(y, best, len(params))
    fitted_sigmas = np.array([result.params[f"{prefix}{i}_sigma"].value for i in range(1, n + 1)])
    background_values = {
        name[3:]: float(
            p.value * (factor if "amplitude" in name or name in ("bg_c", "bg_intercept") else 1)
        )
        for name, p in result.params.items()
        if name.startswith("bg_")
    }
    if "slope" in background_values:
        background_values["slope"] *= factor
    return FitResult(
        centers=np.array([result.params[f"{prefix}{i}_center"].value for i in range(1, n + 1)]),
        sigmas=fitted_sigmas,
        amplitudes=np.array(
            [result.params[f"{prefix}{i}_amplitude"].value * factor for i in range(1, n + 1)]
        ),
        fwhm=FWHM_PER_SIGMA * fitted_sigmas,
        r2=r2,
        reduced_chi2=red_chi2,
        best_fit=best,
        background=background_values,
        success=bool(result.success),
        result=result,
    )


def fit_voigt(x: Any, y: Any, center: float | None = None) -> FitResult:
    """Fit one Voigt profile (lmfit's ``VoigtModel``) on top of a constant."""
    from lmfit.models import ConstantModel, VoigtModel

    x = np.asarray(to_host(x), dtype=float)
    y = np.asarray(to_host(y), dtype=float)
    model = VoigtModel(prefix="v_") + ConstantModel(prefix="bg_")
    params = model.make_params()
    params.update(VoigtModel(prefix="v_").guess(y - y.min(), x=x))
    if center is not None:
        params["v_center"].set(value=center)
    params["bg_c"].set(value=float(y.min()))
    result = model.fit(y, params, x=x)
    r2, red_chi2 = _r2_and_chi2(y, result.best_fit, len(params))
    sigma = float(result.params["v_sigma"].value)
    return FitResult(
        centers=np.array([result.params["v_center"].value]),
        sigmas=np.array([sigma]),
        amplitudes=np.array([result.params["v_height"].value]),
        fwhm=np.array([result.params["v_fwhm"].value]),
        r2=r2,
        reduced_chi2=red_chi2,
        best_fit=np.asarray(result.best_fit),
        background={"c": float(result.params["bg_c"].value)},
        success=bool(result.success),
        result=result,
    )


def fit_report(fit: FitResult) -> dict[str, Any]:
    """The numbers of a fit as a flat dict (``center_1``, ``fwhm_1``, ..., ``r2``)."""
    out: dict[str, Any] = {}
    for i, (c, s, a, f) in enumerate(
        zip(fit.centers, fit.sigmas, fit.amplitudes, fit.fwhm, strict=True), start=1
    ):
        out.update(
            {
                f"center_{i}": float(c),
                f"sigma_{i}": float(s),
                f"amplitude_{i}": float(a),
                f"fwhm_{i}": float(f),
            }
        )
    out.update({f"background_{k}": v for k, v in fit.background.items()})
    out.update({"r2": fit.r2, "reduced_chi2": fit.reduced_chi2, "success": fit.success})
    return out


def fit_rocking_curves(
    x: Any, curves: Any, workers: int | None = None, **kwargs: Any
) -> list[FitResult]:
    """:func:`fit_multi_gaussian` for every row of ``curves`` (one ROI sum per row), in threads."""
    rows = [
        np.asarray(to_host(row), dtype=float) for row in np.atleast_2d(np.asarray(to_host(curves)))
    ]
    return thread_map(lambda row: fit_multi_gaussian(x, row, **kwargs), rows, workers=workers)


def edge(x: Any, amplitude: float, center: float, sigma: float, offset: float) -> Any:
    """A step of height ``amplitude`` at ``center``, blurred by a Gaussian of width ``sigma``."""
    from scipy.special import erf

    z = (np.asarray(x, dtype=float) - center) / (np.sqrt(2.0) * sigma)
    return offset + 0.5 * amplitude * (1.0 + erf(z))


def fit_edge(x: Any, y: Any, center: float | None = None) -> FitResult:
    """Fit a blurred step (an error function) to a curve: knife-edge and sample-height scans.

    The centre is where the signal crosses half way, ``sigmas`` holds the
    Gaussian width of the blur and ``fwhm`` its full width at half maximum, so
    a beam size read from an edge scan comes out in the units of a peak width.
    A falling edge gives a negative amplitude.
    """
    from scipy.optimize import curve_fit

    x = np.asarray(to_host(x), dtype=float)
    y = np.asarray(to_host(y), dtype=float)
    span = float(x.max() - x.min()) or 1.0
    third = max(len(y) // 3, 1)
    rising = y[-third:].mean() >= y[:third].mean()
    amplitude0 = float(y.max() - y.min()) * (1.0 if rising else -1.0)
    offset0 = float(y.min() if rising else y.max())
    if center is None:
        half = offset0 + amplitude0 / 2.0
        crossings = np.flatnonzero(np.diff(np.sign(y - half)))
        center = float(x[crossings[0]]) if crossings.size else float(x.mean())
    p0 = [amplitude0, float(center), 0.1 * span, offset0]
    bounds = ([-np.inf, x.min(), 1e-3 * span, -np.inf], [np.inf, x.max(), span, np.inf])
    params, _ = curve_fit(edge, x, y, p0=p0, bounds=bounds)
    best = edge(x, *params)
    r2, red_chi2 = _r2_and_chi2(y, best, len(params))
    amplitude, centre, sigma, offset = (float(v) for v in params)
    return FitResult(
        centers=np.array([centre]),
        sigmas=np.array([sigma]),
        amplitudes=np.array([amplitude]),
        fwhm=np.array([FWHM_PER_SIGMA * sigma]),
        r2=r2,
        reduced_chi2=red_chi2,
        best_fit=best,
        background={"c": offset},
        success=True,
        result=None,
    )
