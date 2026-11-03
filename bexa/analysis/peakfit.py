"""Peak fitting: sums of peak shapes on a background, with bounds and fixed values.

Shapes are parametrised by their height (``amplitude``), ``center`` and full
width at half maximum (``fwhm``), so the numbers read straight off a plot:

- ``gaussian``, ``lorentzian``;
- ``pseudo_voigt`` (``eta``: the Lorentzian share, 0 to 1);
- ``voigt`` (``fwhm`` of the Gaussian part, ``lorentz_fwhm`` of the Lorentzian);
- ``pearson7`` (``m``: 1 is a Lorentzian, large ``m`` tends to a Gaussian);
- ``step``: an error-function edge of height ``amplitude``, ``fwhm`` the width
  of its blur (knife-edge and height scans).

Backgrounds: ``none``, ``constant``, ``linear``, ``quadratic`` (about the middle
of the fitted range) and ``exponential``. Every parameter can be given as a
number, ``(value, min, max)``, ``(value, min, max, fixed)``, a dict or a
:class:`Param`; missing ones are guessed. The fit is scipy's bounded
``least_squares``, weighted by ``1/sigma`` when uncertainties are given; the
parameter errors come from the Jacobian. Nothing here needs lmfit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from bexa.core.backend import to_host

__all__ = [
    "BACKGROUNDS",
    "SHAPES",
    "Param",
    "PeakFit",
    "fit_peaks",
    "fit_series",
    "guess_peaks",
    "model_components",
    "series_table",
    "shape_values",
]

SHAPES: dict[str, tuple[str, ...]] = {
    "gaussian": ("center", "fwhm", "amplitude"),
    "lorentzian": ("center", "fwhm", "amplitude"),
    "pseudo_voigt": ("center", "fwhm", "amplitude", "eta"),
    "voigt": ("center", "fwhm", "amplitude", "lorentz_fwhm"),
    "pearson7": ("center", "fwhm", "amplitude", "m"),
    "step": ("center", "fwhm", "amplitude"),
}
BACKGROUNDS: dict[str, tuple[str, ...]] = {
    "none": (),
    "constant": ("c0",),
    "linear": ("c0", "c1"),
    "quadratic": ("c0", "c1", "c2"),
    "exponential": ("a", "decay", "c0"),
}
# parameters that scale with the intensity: fitted on curves divided by their maximum
INTENSITY_PARAMS = frozenset({"amplitude", "c0", "c1", "c2", "a"})
GAUSS_SIGMA_PER_FWHM = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
FOUR_LN2 = 4.0 * math.log(2.0)


@dataclass
class Param:
    """One fit parameter: start value, bounds and whether it is held fixed."""

    value: float
    min: float = -np.inf
    max: float = np.inf
    fixed: bool = False

    @classmethod
    def of(cls, spec: Any) -> Param:
        """A Param from a number, ``(value, min, max[, fixed])``, a dict or a Param."""
        if isinstance(spec, Param):
            return Param(spec.value, spec.min, spec.max, spec.fixed)
        if isinstance(spec, Mapping):
            return cls(
                float(spec["value"]),
                float(spec.get("min", -np.inf) if spec.get("min") is not None else -np.inf),
                float(spec.get("max", np.inf) if spec.get("max") is not None else np.inf),
                bool(spec.get("fixed", False)),
            )
        if isinstance(spec, (tuple, list)):
            lo = -np.inf if len(spec) < 2 or spec[1] is None else float(spec[1])
            hi = np.inf if len(spec) < 3 or spec[2] is None else float(spec[2])
            return cls(float(spec[0]), lo, hi, bool(spec[3]) if len(spec) > 3 else False)
        return cls(float(spec))

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "min": self.min, "max": self.max, "fixed": self.fixed}


def shape_values(shape: str, x: np.ndarray, p: Mapping[str, float]) -> np.ndarray:
    """The values of one peak ``shape`` with parameters ``p`` at ``x``."""
    u = (x - p["center"]) / p["fwhm"]
    amplitude = p["amplitude"]
    if shape == "gaussian":
        return amplitude * np.exp(-FOUR_LN2 * u * u)
    if shape == "lorentzian":
        return amplitude / (1.0 + 4.0 * u * u)
    if shape == "pseudo_voigt":
        eta = p["eta"]
        return amplitude * (eta / (1.0 + 4.0 * u * u) + (1.0 - eta) * np.exp(-FOUR_LN2 * u * u))
    if shape == "pearson7":
        m = p["m"]
        return amplitude / (1.0 + 4.0 * (2.0 ** (1.0 / m) - 1.0) * u * u) ** m
    if shape == "voigt":
        from scipy.special import voigt_profile

        sg = p["fwhm"] * GAUSS_SIGMA_PER_FWHM
        gl = 0.5 * p["lorentz_fwhm"]
        return amplitude * voigt_profile(x - p["center"], sg, gl) / voigt_profile(0.0, sg, gl)
    if shape == "step":
        from scipy.special import erf

        z = (x - p["center"]) / (p["fwhm"] * GAUSS_SIGMA_PER_FWHM * math.sqrt(2.0))
        return 0.5 * amplitude * (1.0 + erf(z))
    raise ValueError(f"unknown shape {shape!r}; choose from {tuple(SHAPES)}")


def _background_values(kind: str, x: np.ndarray, p: Mapping[str, float], x0: float) -> np.ndarray:
    if kind == "none":
        return np.zeros_like(x)
    if kind == "exponential":
        return p["a"] * np.exp(-(x - x0) / p["decay"]) + p["c0"]
    out = np.full_like(x, p["c0"])
    if kind in ("linear", "quadratic"):
        out += p["c1"] * (x - x0)
    if kind == "quadratic":
        out += p["c2"] * (x - x0) ** 2
    return out


def _area(shape: str, p: Mapping[str, float]) -> float:
    a, w = p["amplitude"], p["fwhm"]
    gauss = a * w * math.sqrt(math.pi / FOUR_LN2)
    lorentz = a * math.pi * w / 2.0
    if shape == "gaussian":
        return gauss
    if shape == "lorentzian":
        return lorentz
    if shape == "pseudo_voigt":
        return p["eta"] * lorentz + (1.0 - p["eta"]) * gauss
    if shape == "pearson7":
        m = p["m"]
        return (
            a
            * w
            * math.sqrt(math.pi)
            * math.gamma(m - 0.5)
            / (2.0 * math.sqrt(2.0 ** (1.0 / m) - 1.0) * math.gamma(m))
            if m > 0.5
            else float("inf")
        )
    if shape == "voigt":
        from scipy.special import voigt_profile

        return a / float(voigt_profile(0.0, w * GAUSS_SIGMA_PER_FWHM, 0.5 * p["lorentz_fwhm"]))
    return float("nan")  # a step has no area


def guess_peaks(x: Any, y: Any, n: int = 1, shape: str = "gaussian") -> list[dict[str, Any]]:
    """Start values for ``n`` peaks: the most prominent maxima, their heights and widths."""
    from scipy.signal import find_peaks, peak_widths

    x = np.asarray(to_host(x), dtype=float)
    y = np.asarray(to_host(y), dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    step = float(np.median(np.diff(x))) if x.size > 1 else 1.0
    span = float(np.ptp(x)) or 1.0
    base = float(np.nanmin(y))
    if shape == "step":
        third = max(len(y) // 3, 1)
        rising = np.nanmean(y[-third:]) >= np.nanmean(y[:third])
        height = float(np.nanmax(y) - base) * (1.0 if rising else -1.0)
        half = base + abs(height) / 2.0
        crossing = np.flatnonzero(np.diff(np.sign(y - half)))
        centre = float(x[crossing[0]]) if crossing.size else float(np.mean(x))
        return [{"shape": "step", "center": centre, "fwhm": 0.1 * span, "amplitude": height}]
    found, props = find_peaks(np.nan_to_num(y - base), prominence=0.0)
    chosen = found[np.argsort(props["prominences"])[::-1][:n]] if found.size else np.array([], int)
    if chosen.size:
        widths = peak_widths(np.nan_to_num(y - base), chosen, rel_height=0.5)[0] * abs(step)
    else:
        widths = np.array([])
    peaks = []
    for index, width in sorted(zip(chosen, widths, strict=True), key=lambda pair: x[pair[0]]):
        peaks.append(
            {
                "shape": shape,
                "center": float(x[index]),
                "fwhm": max(float(width), abs(step)),
                "amplitude": float(y[index] - base),
            }
        )
    while len(peaks) < n:  # fewer maxima than asked: spread the rest over the range
        k = len(peaks) + 1
        peaks.append(
            {
                "shape": shape,
                "center": float(x[0] + span * k / (n + 1)),
                "fwhm": 0.1 * span,
                "amplitude": float(np.nanmax(y) - base) / 2.0,
            }
        )
    return peaks


def _default_param(name: str, x: np.ndarray, y: np.ndarray, guess: Mapping[str, Any]) -> Param:
    """The start value and bounds bexa uses when a parameter is not given."""
    span = float(np.ptp(x)) or 1.0
    step = float(np.min(np.abs(np.diff(x)))) if x.size > 1 else span
    value = guess.get(name)
    if name == "center":
        return Param(float(value), float(x.min()) - 0.1 * span, float(x.max()) + 0.1 * span)
    if name == "fwhm":
        return Param(float(value), max(step / 10.0, span * 1e-6), 5.0 * span)
    if name == "amplitude":
        v = float(value)
        return Param(v, 0.0, np.inf) if v >= 0 else Param(v, -np.inf, 0.0)
    if name == "eta":
        return Param(0.5, 0.0, 1.0)
    if name == "m":
        return Param(1.5, 0.501, 100.0)
    if name == "lorentz_fwhm":
        return Param(0.5 * float(guess.get("fwhm", step)), 0.0, 5.0 * span)
    raise KeyError(name)


def _background_params(kind: str, x: np.ndarray, y: np.ndarray) -> dict[str, Param]:
    if kind not in BACKGROUNDS:
        raise ValueError(f"unknown background {kind!r}; choose from {tuple(BACKGROUNDS)}")
    low = float(np.nanmin(y))
    span = float(np.ptp(x)) or 1.0
    start = {"c0": low, "c1": 0.0, "c2": 0.0, "a": 0.0, "decay": span}
    params = {name: Param(start[name]) for name in BACKGROUNDS[kind]}
    if "decay" in params:
        params["decay"] = Param(span, span * 1e-3, span * 1e3)
    return params


@dataclass
class PeakFit:
    """The result of :func:`fit_peaks`: values, errors, curves and quality of the fit."""

    x: np.ndarray
    y: np.ndarray
    shapes: list[str]
    background: str
    values: dict[str, float]
    errors: dict[str, float]
    best_fit: np.ndarray
    r2: float
    reduced_chi2: float
    success: bool
    message: str = ""
    fixed: set[str] = field(default_factory=set)
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def x0(self) -> float:
        """The reference x of the background polynomial (the middle of the fitted range)."""
        return float(0.5 * (self.x.min() + self.x.max()))

    @property
    def residual(self) -> np.ndarray:
        return self.y - self.best_fit

    def peak_params(self, k: int) -> dict[str, float]:
        """The parameters of peak ``k`` (from 1) without their prefix."""
        prefix = f"p{k}_"
        return {n[len(prefix) :]: v for n, v in self.values.items() if n.startswith(prefix)}

    def components(self, x: Any = None) -> dict[str, np.ndarray]:
        """Each peak (``p1``, ``p2``, ...) and the ``background`` evaluated at ``x``."""
        xs = self.x if x is None else np.asarray(x, dtype=float)
        out = {
            f"p{k}": shape_values(shape, xs, self.peak_params(k))
            for k, shape in enumerate(self.shapes, start=1)
        }
        bg = {n[3:]: v for n, v in self.values.items() if n.startswith("bg_")}
        out["background"] = _background_values(self.background, xs, bg, self.x0)
        return out

    def evaluate(self, x: Any = None) -> np.ndarray:
        """The fitted model (peaks plus background) at ``x`` (default: the fitted points)."""
        return np.sum(list(self.components(x).values()), axis=0)

    def peaks(self) -> list[dict[str, float]]:
        """One dict per peak: its parameters, errors (``*_err``), area and shape."""
        rows = []
        for k, shape in enumerate(self.shapes, start=1):
            p = self.peak_params(k)
            row: dict[str, Any] = {"peak": k, "shape": shape, **p}
            row.update({f"{n}_err": self.errors.get(f"p{k}_{n}", np.nan) for n in p})
            row["area"] = _area(shape, p)
            if shape == "voigt":  # the width of the whole profile (Olivero and Longbothum)
                fl, fg = p["lorentz_fwhm"], p["fwhm"]
                row["fwhm_total"] = 0.5346 * fl + math.sqrt(0.2166 * fl * fl + fg * fg)
            rows.append(row)
        return rows

    def table(self) -> pd.DataFrame:
        """:meth:`peaks` as a table, one row per peak."""
        return pd.DataFrame(self.peaks())

    def report(self) -> dict[str, Any]:
        """Every value and error as one flat dict, with ``r2`` and ``reduced_chi2``."""
        out: dict[str, Any] = {}
        for name, value in self.values.items():
            out[name] = value
            out[f"{name}_err"] = self.errors.get(name, np.nan)
        for k, row in enumerate(self.peaks(), start=1):
            out[f"p{k}_area"] = row["area"]
        out.update({"r2": self.r2, "reduced_chi2": self.reduced_chi2, "success": self.success})
        return out

    def spec(self) -> dict[str, Any]:
        """The fitted values as start values (bounds and fixed flags kept) for another fit."""
        peaks = []
        for k, shape in enumerate(self.shapes, start=1):
            peak: dict[str, Any] = {"shape": shape}
            for n, v in self.peak_params(k).items():
                lo, hi = self.bounds.get(f"p{k}_{n}", (-np.inf, np.inf))
                peak[n] = Param(v, lo, hi, f"p{k}_{n}" in self.fixed)
            peaks.append(peak)
        bg = {
            n[3:]: Param(v, *self.bounds.get(n, (-np.inf, np.inf)), n in self.fixed)
            for n, v in self.values.items()
            if n.startswith("bg_")
        }
        return {"peaks": peaks, "background": self.background, "background_params": bg}

    def to_dict(self) -> dict[str, Any]:
        """Everything but the arrays, ready for JSON."""
        return {
            "shapes": list(self.shapes),
            "background": self.background,
            "values": dict(self.values),
            "errors": {k: (None if not np.isfinite(v) else v) for k, v in self.errors.items()},
            "fixed": sorted(self.fixed),
            "r2": self.r2,
            "reduced_chi2": self.reduced_chi2,
            "success": self.success,
            "message": self.message,
            "x_range": [float(self.x.min()), float(self.x.max())],
        }


def _prepare(
    x: Any, y: Any, sigma: Any, x_range: Sequence[float | None] | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    xs = np.asarray(to_host(x), dtype=float).ravel()
    ys = np.asarray(to_host(y), dtype=float).ravel()
    ss = None if sigma is None else np.asarray(to_host(sigma), dtype=float).ravel()
    keep = np.isfinite(xs) & np.isfinite(ys)
    if ss is not None:
        keep &= np.isfinite(ss) & (ss > 0)
    if x_range is not None:
        lo, hi = x_range
        if lo is not None:
            keep &= xs >= lo
        if hi is not None:
            keep &= xs <= hi
    order = np.argsort(xs[keep])
    xs, ys = xs[keep][order], ys[keep][order]
    ss = None if ss is None else ss[keep][order]
    if xs.size < 3:
        raise ValueError(f"only {xs.size} usable points to fit")
    return xs, ys, ss


def _parameters(
    xs: np.ndarray,
    ys: np.ndarray,
    peaks: Sequence[Mapping[str, Any]],
    background: str,
    background_params: Mapping[str, Any] | None,
) -> tuple[list[str], dict[str, Param]]:
    shapes: list[str] = []
    params: dict[str, Param] = {}
    for k, peak in enumerate(peaks, start=1):
        shape = str(peak.get("shape", "gaussian"))
        if shape not in SHAPES:
            raise ValueError(f"unknown shape {shape!r}; choose from {tuple(SHAPES)}")
        shapes.append(shape)
        missing = [n for n in SHAPES[shape] if peak.get(n) is None]
        guess: dict[str, Any] = {}
        if missing:
            guess = guess_peaks(xs, ys, 1, shape)[0]
        guess.update({n: Param.of(peak[n]).value for n in SHAPES[shape] if peak.get(n) is not None})
        if peak.get("amplitude") is None and peak.get("center") is not None and shape != "step":
            # a peak placed by hand: its height is the curve's there
            guess["amplitude"] = float(np.interp(guess["center"], xs, ys) - np.nanmin(ys))
        for name in SHAPES[shape]:
            given = peak.get(name)
            default = _default_param(name, xs, ys, guess)
            if given is None:
                params[f"p{k}_{name}"] = default
                continue
            param = Param.of(given)
            if not isinstance(given, (Param, Mapping, tuple, list)):  # a bare number
                param.min, param.max = default.min, default.max
            params[f"p{k}_{name}"] = param
    bg = _background_params(background, xs, ys)
    for name, spec in (background_params or {}).items():
        if name in bg and spec is not None:
            bg[name] = Param.of(spec)
    params.update({f"bg_{n}": p for n, p in bg.items()})
    return shapes, params


def fit_peaks(
    x: Any,
    y: Any,
    peaks: Sequence[Mapping[str, Any]] | None = None,
    background: str = "linear",
    sigma: Any = None,
    x_range: Sequence[float | None] | None = None,
    n_peaks: int = 1,
    shape: str = "gaussian",
    background_params: Mapping[str, Any] | None = None,
    max_nfev: int = 5000,
) -> PeakFit:
    """Fit peaks on a background to a curve.

    Parameters
    ----------
    peaks
        One dict per peak: ``shape`` and any of its parameters (see the module
        docstring); ``None`` guesses ``n_peaks`` peaks of ``shape``.
    background
        ``none``, ``constant``, ``linear``, ``quadratic`` or ``exponential``;
        ``background_params`` sets their start values (``{"c0": 5}``).
    sigma
        Uncertainties of ``y``: the residuals are weighted by ``1/sigma`` and
        the errors are absolute. Without them the errors are scaled by the
        reduced chi-square.
    x_range
        Fit only the points between these x values.
    """
    from scipy.optimize import least_squares

    xs, ys, ss = _prepare(x, y, sigma, x_range)
    peaks = list(peaks) if peaks is not None else guess_peaks(xs, ys, n_peaks, shape)
    shapes, params = _parameters(xs, ys, peaks, background, background_params)
    names = list(params)
    scale = float(np.nanmax(np.abs(ys))) or 1.0
    factor = np.array([scale if n.split("_", 1)[1] in INTENSITY_PARAMS else 1.0 for n in names])
    start = np.array([params[n].value for n in names]) / factor
    lower = np.array([params[n].min for n in names]) / factor
    upper = np.array([params[n].max for n in names]) / factor
    free = np.array([not params[n].fixed for n in names])
    tight = lower >= upper
    free &= ~tight
    start = np.clip(start, lower, upper)
    x0 = float(0.5 * (xs.min() + xs.max()))
    weights = 1.0 / (ss / scale) if ss is not None else np.ones_like(ys)
    target = ys / scale

    def model(theta: np.ndarray) -> np.ndarray:
        full = start.copy()
        full[free] = theta
        values = dict(zip(names, full, strict=True))
        total = np.zeros_like(xs)
        for k, shp in enumerate(shapes, start=1):
            prefix = f"p{k}_"
            total += shape_values(
                shp, xs, {n[len(prefix) :]: v for n, v in values.items() if n.startswith(prefix)}
            )
        bg = {n[3:]: v for n, v in values.items() if n.startswith("bg_")}
        return total + _background_values(background, xs, bg, x0)

    def residual(theta: np.ndarray) -> np.ndarray:
        return (model(theta) - target) * weights

    success, message = True, ""
    fitted = start.copy()
    errors = np.full(len(names), np.nan)
    if free.any():
        result = least_squares(
            residual,
            start[free],
            bounds=(lower[free], upper[free]),
            x_scale="jac",
            max_nfev=max_nfev,
        )
        fitted[free] = result.x
        success, message = bool(result.success), str(result.message)
        dof = max(xs.size - int(free.sum()), 1)
        chi2_scaled = float(np.sum(result.fun**2)) / dof
        try:
            covariance = np.linalg.pinv(result.jac.T @ result.jac)
            if ss is None:
                covariance = covariance * chi2_scaled
            errors[free] = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
        except np.linalg.LinAlgError:
            pass
    best = model(fitted[free]) * scale
    values = dict(zip(names, (fitted * factor).tolist(), strict=True))
    residuals = ys - best
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residuals**2)) / ss_tot if ss_tot > 0 else float("nan")
    dof = max(xs.size - int(free.sum()), 1)
    chi = residuals / ss if ss is not None else residuals
    return PeakFit(
        x=xs,
        y=ys,
        shapes=shapes,
        background=background,
        values=values,
        errors=dict(zip(names, (errors * factor).tolist(), strict=True)),
        best_fit=best,
        r2=r2,
        reduced_chi2=float(np.sum(chi**2)) / dof,
        success=success,
        message=message,
        fixed={n for n, f in zip(names, free, strict=True) if not f},
        bounds={n: (params[n].min, params[n].max) for n in names},
    )


def model_components(summary: Mapping[str, Any], x: Any) -> dict[str, np.ndarray]:
    """Each peak (``p1``...) and the ``background`` of a fit summary at ``x``.

    ``summary`` is :meth:`PeakFit.to_dict`.

    What a fit saved in an object's metadata draws again without refitting.
    """
    xs = np.asarray(x, dtype=float)
    values = dict(summary["values"])
    lo, hi = summary["x_range"]
    out = {}
    for k, shape in enumerate(summary["shapes"], start=1):
        prefix = f"p{k}_"
        out[f"p{k}"] = shape_values(
            shape, xs, {n[len(prefix) :]: v for n, v in values.items() if n.startswith(prefix)}
        )
    bg = {n[3:]: v for n, v in values.items() if n.startswith("bg_")}
    out["background"] = _background_values(summary["background"], xs, bg, 0.5 * (lo + hi))
    return out


def fit_series(
    x: Any,
    curves: Any,
    peaks: Sequence[Mapping[str, Any]] | None = None,
    background: str = "linear",
    sigmas: Any = None,
    warm_start: bool = True,
    **kwargs: Any,
) -> list[PeakFit | None]:
    """:func:`fit_peaks` for every row of ``curves`` (one curve per row, all on ``x``).

    The curves are fitted all at once (:func:`bexa.analysis.batchfit.fit_batch`:
    the same models, start values, bounds and errors). With ``warm_start`` a
    curve fitted badly (``r2`` under 0.5) is fitted again from the last one
    before it fitted well, which follows a peak that moves slowly along the
    series. A curve that cannot be fitted gives ``None``.
    """
    from bexa.analysis.batchfit import fit_batch

    rows = np.atleast_2d(np.asarray(to_host(curves), dtype=float))
    errs = None if sigmas is None else np.atleast_2d(np.asarray(to_host(sigmas), dtype=float))
    xs = np.asarray(to_host(x), dtype=float)
    try:
        batch = fit_batch(xs, rows, peaks, background, errs, warm_start=warm_start, **kwargs)
    except ValueError:  # no curve with points enough to fit
        return [None] * rows.shape[0]
    return [batch.peak_fit(i, xs, rows[i]) for i in range(rows.shape[0])]


def series_table(
    fits: Sequence[PeakFit | None], coords: Any = None, name: str = "index"
) -> pd.DataFrame:
    """One row per fit of a series: the coordinate, every value, error, area and ``r2``."""
    values = np.arange(len(fits)) if coords is None else np.asarray(coords)
    rows = []
    for value, fit in zip(values, fits, strict=True):
        row: dict[str, Any] = {name: value}
        if fit is not None:
            row.update(fit.report())
        rows.append(row)
    return pd.DataFrame(rows)
