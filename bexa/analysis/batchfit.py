"""Many curves fitted at once: the peaks of :mod:`bexa.analysis.peakfit` on every row of an array.

:func:`bexa.analysis.peakfit.fit_peaks` runs scipy's ``least_squares`` on one
curve, a few milliseconds each, most of it Python; the rocking curve of every
pixel of a map is a quarter of a million such fits. :func:`fit_batch` fits
them together: one Levenberg-Marquardt iteration updates every curve at once,
the model and its Jacobian (forward differences, one model evaluation per
parameter for all curves) are numpy arrays of ``(curves, points)``, and each
curve keeps its own damping and stops on its own when it converges. The
models, start values, bounds, fixed parameters, weights and errors are those of
``fit_peaks``; the start values of a single peak are guessed for every curve at
once (several peaks or a step: curve by curve). A curve with fewer usable
points than free parameters is not fitted. Every iteration tells the GUI's
progress bar how many curves have converged, and Cancel stops it there.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from bexa._log import report
from bexa.analysis.peakfit import (
    BACKGROUNDS,
    INTENSITY_PARAMS,
    SHAPES,
    Param,
    PeakFit,
    _parameters,
    guess_peaks,
    shape_values,
)

__all__ = ["BatchFit", "fit_batch", "guess_single"]

FTOL = 1e-8  # a step that lowers the cost by less than this share of it ends a curve's fit
XTOL = 1e-8  # and so does one that moves its parameters by less than this share
MAX_ITERATIONS = 400
REL_STEP = math.sqrt(np.finfo(float).eps)  # the forward-difference step, as scipy's
GOOD_R2 = 0.5  # a warm start refits from the last curve fitted at least this well


@dataclass
class BatchFit:
    """:func:`fit_batch`: one fit per curve as arrays, ``(curves, parameters)`` and ``(curves,)``.

    ``names`` are the parameters (``p1_center``, ``bg_c0``...), the columns of
    ``values`` and ``errors``; ``r2``, ``reduced_chi2``, ``success`` and
    ``x0`` (the reference x of the background polynomial) have one value per
    curve. A curve that was not fitted has NaN values and ``success`` False.
    """

    shapes: list[str]
    background: str
    names: list[str]
    values: np.ndarray
    errors: np.ndarray
    r2: np.ndarray
    reduced_chi2: np.ndarray
    success: np.ndarray
    x0: np.ndarray

    def __len__(self) -> int:
        return int(self.values.shape[0])

    def column(self, name: str) -> np.ndarray:
        """The fitted values of one parameter, one per curve."""
        return self.values[:, self.names.index(name)]

    def areas(self) -> dict[int, np.ndarray]:
        """The area of every peak ``k`` (from 1) of every curve (NaN for a step)."""
        return {
            k: _areas(shape, {n: self.column(f"p{k}_{n}") for n in SHAPES[shape]})
            for k, shape in enumerate(self.shapes, start=1)
        }

    def table(self, coords: Any = None, name: str = "index") -> pd.DataFrame:
        """One row per curve, the columns of :func:`bexa.analysis.peakfit.series_table`."""
        n = len(self)
        columns: dict[str, Any] = {name: np.arange(n) if coords is None else np.asarray(coords)}
        for j, param in enumerate(self.names):
            columns[param] = self.values[:, j]
            columns[f"{param}_err"] = self.errors[:, j]
        for k, area in self.areas().items():
            columns[f"p{k}_area"] = area
        columns.update(r2=self.r2, reduced_chi2=self.reduced_chi2, success=self.success)
        return pd.DataFrame(columns)

    def peak_fit(self, i: int, x: Any, y: Any) -> PeakFit | None:
        """Curve ``i`` (its ``x`` and ``y``) as a :class:`~bexa.analysis.peakfit.PeakFit`;
        ``None`` when it was not fitted."""
        if not np.isfinite(self.values[i]).all():
            return None
        xs = np.asarray(x, dtype=float)
        xs = xs[i] if xs.ndim == 2 else xs
        ys = np.asarray(y, dtype=float)
        keep = np.isfinite(xs) & np.isfinite(ys)
        fit = PeakFit(
            x=xs[keep],
            y=ys[keep],
            shapes=list(self.shapes),
            background=self.background,
            values=dict(zip(self.names, self.values[i].tolist(), strict=True)),
            errors=dict(zip(self.names, self.errors[i].tolist(), strict=True)),
            best_fit=np.zeros(int(keep.sum())),
            r2=float(self.r2[i]),
            reduced_chi2=float(self.reduced_chi2[i]),
            success=bool(self.success[i]),
        )
        fit.best_fit = fit.evaluate(fit.x)
        return fit


# ---------------------------------------------------------------- the model
def _background(kind: str, x: np.ndarray, p: Mapping[str, np.ndarray], x0: np.ndarray) -> Any:
    """The background of every curve: ``p`` holds ``(curves, 1)`` columns, ``x0`` too."""
    if kind == "none":
        return 0.0
    if kind == "exponential":
        return p["a"] * np.exp(-(x - x0) / p["decay"]) + p["c0"]
    out = p["c0"] + 0.0 * x
    if kind in ("linear", "quadratic"):
        out = out + p["c1"] * (x - x0)
    if kind == "quadratic":
        out = out + p["c2"] * (x - x0) ** 2
    return out


class _Model:
    """The model of every curve, ``(curves, points)``, from a ``(curves, parameters)`` array."""

    def __init__(self, shapes: list[str], background: str, names: list[str]) -> None:
        self.shapes, self.background = shapes, background
        self.peaks = [
            {n: names.index(f"p{k}_{n}") for n in SHAPES[shape]}
            for k, shape in enumerate(shapes, start=1)
        ]
        self.bg = {n: names.index(f"bg_{n}") for n in BACKGROUNDS[background]}

    def __call__(self, theta: np.ndarray, x: np.ndarray, x0: np.ndarray) -> np.ndarray:
        bg = {n: theta[:, j, None] for n, j in self.bg.items()}
        total = _background(self.background, x, bg, x0)
        for shape, columns in zip(self.shapes, self.peaks, strict=True):
            total = total + shape_values(
                shape, x, {n: theta[:, j, None] for n, j in columns.items()}
            )
        return np.broadcast_to(total, (theta.shape[0], x.shape[-1]))


def _areas(shape: str, p: Mapping[str, np.ndarray]) -> np.ndarray:
    """The area of a peak of every curve at once, as :class:`PeakFit` gives it."""
    from scipy.special import gamma, voigt_profile

    a, w = p["amplitude"], p["fwhm"]
    gauss = a * w * math.sqrt(math.pi / (4.0 * math.log(2.0)))
    lorentz = a * math.pi * w / 2.0
    if shape == "gaussian":
        return gauss
    if shape == "lorentzian":
        return lorentz
    if shape == "pseudo_voigt":
        return p["eta"] * lorentz + (1.0 - p["eta"]) * gauss
    if shape == "pearson7":
        m = p["m"]
        with np.errstate(invalid="ignore", divide="ignore"):
            value = a * w * math.sqrt(math.pi) * gamma(m - 0.5)
            value = value / (2.0 * np.sqrt(2.0 ** (1.0 / m) - 1.0) * gamma(m))
        return np.where(m > 0.5, value, np.inf)
    if shape == "voigt":
        sigma = w / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        return a / voigt_profile(0.0, sigma, 0.5 * p["lorentz_fwhm"])
    return np.full(np.shape(a), np.nan)  # a step has no area


# ---------------------------------------------------------------- start values
def guess_single(x: np.ndarray, y: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    """``center``, ``fwhm`` and ``amplitude`` of one peak on every curve at once.

    The highest inner local maximum of each curve's usable points (the highest
    point when there is none), as :func:`bexa.analysis.peakfit.guess_peaks`
    takes; its height above the curve's lowest point; its width where the
    curve falls to half that height (at least one step of x).
    """
    m, n = y.shape
    xs = np.broadcast_to(x, (m, n))
    rows = np.arange(m)
    low = np.where(valid, y, np.inf).min(axis=1)
    y = np.where(valid, y, low[:, None])  # a missing point is no crossing nor a peak
    filled = np.where(valid, y, -np.inf)
    inner = np.zeros((m, n), bool)
    if n > 2:
        inner[:, 1:-1] = (filled[:, 1:-1] > filled[:, :-2]) & (filled[:, 1:-1] >= filled[:, 2:])
    candidates = np.where(inner, filled, -np.inf)
    has_inner = np.isfinite(candidates.max(axis=1))
    peak = np.where(has_inner, candidates.argmax(axis=1), filled.argmax(axis=1))
    top = filled[rows, peak]
    half = low + 0.5 * (top - low)
    index = np.arange(n)[None, :]
    below = valid & (filled < half[:, None])
    left = np.where(below & (index < peak[:, None]), index, -1).max(axis=1)
    right = np.where(below & (index > peak[:, None]), index, n).min(axis=1)

    def crossing(outer: np.ndarray, inner_: np.ndarray, missing: np.ndarray) -> np.ndarray:
        """Where a curve crosses half height between the point ``outer`` (below) and ``inner_``."""
        a, b = np.clip(outer, 0, n - 1), np.clip(inner_, 0, n - 1)
        ya, yb = y[rows, a], y[rows, b]
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.clip(np.where(yb != ya, (half - ya) / (yb - ya), 0.0), 0.0, 1.0)
        at = xs[rows, a] + t * (xs[rows, b] - xs[rows, a])
        return np.where(missing, xs[rows, peak], at)

    x_left = crossing(left, left + 1, left < 0)
    x_right = crossing(right, right - 1, right >= n)
    step = np.abs(np.diff(xs, axis=1)).min(axis=1) if n > 1 else np.ones(m)
    return {
        "center": xs[rows, peak],
        "fwhm": np.maximum(np.abs(x_right - x_left), step),
        "amplitude": top - low,
    }


def _height_at(x: np.ndarray, y: np.ndarray, at: float, low: np.ndarray) -> np.ndarray:
    """Each curve's height above its lowest point at ``x = at`` (a peak placed by hand)."""
    i = int(np.clip(np.searchsorted(x, at), 1, x.size - 1))
    t = (at - x[i - 1]) / (x[i] - x[i - 1]) if x[i] != x[i - 1] else 0.0
    return y[:, i - 1] + t * (y[:, i] - y[:, i - 1]) - low


def _starts(
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    names: list[str],
    peaks: Sequence[Mapping[str, Any]] | None,
    background: str,
    background_params: Mapping[str, Any] | None,
    n_peaks: int,
    shape: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Start values and lower and upper bounds, ``(curves, parameters)``, as ``fit_peaks``
    would give each curve."""
    m, n = y.shape
    xs = np.broadcast_to(x, (m, n))
    spec = list(peaks) if peaks is not None else []
    one = (peaks is None and n_peaks == 1 and shape != "step") or (
        len(spec) == 1 and spec[0].get("shape", "gaussian") != "step"
    )
    placed = one and spec and spec[0].get("center") is not None and spec[0].get("amplitude") is None
    if not one or (placed and x.ndim != 1):  # guessed curve by curve
        rows = []
        for i in range(m):
            keep = valid[i]
            xi, yi = xs[i, keep], y[i, keep]
            given = spec if peaks is not None else guess_peaks(xi, yi, n_peaks, shape)
            rows.append(_parameters(xi, yi, given, background, background_params)[1])
        start = np.array([[r[name].value for name in names] for r in rows])
        lower = np.array([[r[name].min for name in names] for r in rows])
        upper = np.array([[r[name].max for name in names] for r in rows])
        return start, lower, upper
    x_lo = np.where(valid, xs, np.inf).min(axis=1)
    x_hi = np.where(valid, xs, -np.inf).max(axis=1)
    span = np.where(x_hi > x_lo, x_hi - x_lo, 1.0)
    pairs = valid[:, 1:] & valid[:, :-1]
    step = np.where(pairs, np.abs(np.diff(xs, axis=1)), np.inf).min(axis=1) if n > 1 else span
    step = np.where(np.isfinite(step), step, span)
    low = np.where(valid, y, np.inf).min(axis=1)
    peak = spec[0] if spec else {}
    guess = guess_single(x, y, valid)
    for field in ("center", "fwhm", "amplitude"):
        if peak.get(field) is not None:
            guess[field] = np.full(m, Param.of(peak[field]).value)
    if placed:  # a peak placed by hand: its height is the curve's there
        guess["amplitude"] = _height_at(
            x, np.where(valid, y, low[:, None]), guess["center"][0], low
        )
    a = guess["amplitude"]
    defaults = {
        "center": (guess["center"], x_lo - 0.1 * span, x_hi + 0.1 * span),
        "fwhm": (guess["fwhm"], np.maximum(step / 10.0, span * 1e-6), 5.0 * span),
        "amplitude": (a, np.where(a >= 0, 0.0, -np.inf), np.where(a >= 0, np.inf, 0.0)),
        "eta": (0.5, 0.0, 1.0),
        "m": (1.5, 0.501, 100.0),
        "lorentz_fwhm": (0.5 * guess["fwhm"], 0.0, 5.0 * span),
        "bg_c0": (low, -np.inf, np.inf),
        "bg_c1": (0.0, -np.inf, np.inf),
        "bg_c2": (0.0, -np.inf, np.inf),
        "bg_a": (0.0, -np.inf, np.inf),
        "bg_decay": (span, span * 1e-3, span * 1e3),
    }
    start = np.empty((m, len(names)))
    lower, upper = np.empty_like(start), np.empty_like(start)
    for j, name in enumerate(names):
        kind, field = name.split("_", 1)
        key = name if kind == "bg" else field
        start[:, j], lower[:, j], upper[:, j] = defaults[key]
        given = (background_params or {}).get(field) if kind == "bg" else peak.get(field)
        if given is not None and (kind == "bg" or field not in guess):
            param = Param.of(given)
            start[:, j] = param.value
        if isinstance(given, (Param, Mapping, tuple, list)):  # a bare number keeps the bounds
            param = Param.of(given)
            lower[:, j], upper[:, j] = param.min, param.max
    return start, lower, upper


# ---------------------------------------------------------------- the solver
def _jacobian(
    model: _Model,
    theta: np.ndarray,
    f0: np.ndarray,
    x: np.ndarray,
    x0: np.ndarray,
    upper: np.ndarray,
    free: np.ndarray,
) -> np.ndarray:
    """Forward differences of the model, ``(curves, points, parameters)``: a step back at an
    upper bound, the column of a fixed parameter zero."""
    m, p = theta.shape
    jac = np.zeros((m, f0.shape[1], p))
    for j in np.flatnonzero(free):
        h = REL_STEP * np.maximum(1.0, np.abs(theta[:, j]))
        h = np.where(theta[:, j] + h > upper[:, j], -h, h)
        moved = theta.copy()
        moved[:, j] += h
        jac[:, :, j] = (model(moved, x, x0) - f0) / h[:, None]
    return jac


def _solve(
    model: _Model,
    theta: np.ndarray,
    x: np.ndarray,
    x0: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    free: np.ndarray,
    max_iterations: int,
    desc: str = "fits",
) -> tuple[np.ndarray, np.ndarray]:
    """Levenberg-Marquardt on every curve at once, each with its own damping (Nielsen's
    update), its steps kept inside the bounds: a parameter at a bound that the step would
    push out stays there for that step (an active set), the others move. Gives the
    parameters and which curves converged; ``desc`` names them to the progress bar."""
    m, p = theta.shape
    theta = np.clip(theta, lower, upper)
    residual = (model(theta, x, x0) - target) * weights
    cost = np.einsum("ij,ij->i", residual, residual)
    damping = np.full(m, 1e-3)
    growth = np.full(m, 2.0)
    active = np.isfinite(cost)
    converged = np.zeros(m, bool)
    eye = np.eye(p)
    for _ in range(max_iterations):
        rows = np.flatnonzero(active)
        report(m - rows.size, m, desc)
        if rows.size == 0:
            break
        xr = x if x.ndim == 1 else x[rows]
        th, x0r, w = theta[rows], x0[rows], weights[rows]
        f0 = model(th, xr, x0r)
        jac = _jacobian(model, th, f0, xr, x0r, upper[rows], free) * w[:, :, None]
        normal = np.einsum("knp,knq->kpq", jac, jac)
        gradient = np.einsum("knp,kn->kp", jac, residual[rows])
        lo, hi = lower[rows], upper[rows]
        pinned = ((th <= lo) & (gradient > 0)) | ((th >= hi) & (gradient < 0))
        moving = free[None, :] & ~pinned
        scaling = np.maximum(np.einsum("kpp->kp", normal), 1e-12)
        system = normal + damping[rows, None, None] * scaling[:, :, None] * eye
        both = moving[:, :, None] & moving[:, None, :]
        system = np.where(both, system, 0.0) + eye * ~moving[:, :, None]
        gradient = np.where(moving, gradient, 0.0)
        try:
            step = np.linalg.solve(system, -gradient[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:  # a singular system somewhere: least squares, curve by curve
            step = np.stack(
                [
                    np.linalg.lstsq(s, -g, rcond=None)[0]
                    for s, g in zip(system, gradient, strict=True)
                ]
            )
        step = np.where(moving, step, 0.0)
        trial = np.clip(th + step, lo, hi)
        new_residual = (model(trial, xr, x0r) - target[rows]) * w
        new_cost = np.einsum("ij,ij->i", new_residual, new_residual)
        gain = cost[rows] - new_cost
        better = np.isfinite(new_cost) & (gain > 0)
        # the gain the linear model predicted: how well the step was judged
        damped = damping[rows, None] * scaling * step
        predicted = np.einsum("kp,kp->k", step, damped - gradient)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(predicted > 0, gain / predicted, 0.0)
        small_step = np.abs(trial - th).max(axis=1) <= XTOL * (np.abs(th).max(axis=1) + XTOL)
        small_gain = gain <= FTOL * cost[rows]
        taken = rows[better]
        theta[taken] = trial[better]
        residual[taken] = new_residual[better]
        cost[taken] = new_cost[better]
        shrink = np.maximum(1.0 / 3.0, 1.0 - (2.0 * ratio - 1.0) ** 3)
        damping[rows] = np.where(
            better, np.maximum(damping[rows] * shrink, 1e-15), damping[rows] * growth[rows]
        )
        growth[rows] = np.where(better, 2.0, growth[rows] * 2.0)
        settled = better & small_gain & (ratio > 0.25)  # a gain judged well, not a damped crawl
        done = settled | small_step | (damping[rows] > 1e15)
        converged[rows[done]] = True
        active[rows[done]] = False
    return theta, converged


@dataclass
class _Problem:
    """What every curve's fit needs, in the scaled units the solver works in."""

    x: np.ndarray  # (points,) or (curves, points)
    x0: np.ndarray  # (curves, 1)
    curves: np.ndarray
    valid: np.ndarray
    sigmas: np.ndarray | None
    target: np.ndarray  # the curves over their scale, 0 where not usable
    weights: np.ndarray  # 0 where not usable
    scale: np.ndarray  # (curves,)
    factor: np.ndarray  # (curves, parameters): the scale for intensities, 1 for the rest
    lower: np.ndarray
    upper: np.ndarray
    free: np.ndarray  # (parameters,)

    def rows(self, index: np.ndarray) -> _Problem:
        return _Problem(
            self.x if self.x.ndim == 1 else self.x[index],
            self.x0[index],
            self.curves[index],
            self.valid[index],
            None if self.sigmas is None else self.sigmas[index],
            self.target[index],
            self.weights[index],
            self.scale[index],
            self.factor[index],
            self.lower[index],
            self.upper[index],
            self.free,
        )


def _run(
    model: _Model,
    problem: _Problem,
    start: np.ndarray,
    max_iterations: int,
    desc: str = "fits",
) -> tuple[np.ndarray, ...]:
    """Fit every curve of ``problem`` from ``start`` (curve units): values, errors, r2,
    reduced chi-square and success."""
    pr = problem
    theta, converged = _solve(
        model,
        start / pr.factor,
        pr.x,
        pr.x0,
        pr.target,
        pr.weights,
        pr.lower / pr.factor,
        pr.upper / pr.factor,
        pr.free,
        max_iterations,
        desc,
    )
    m, p = theta.shape
    errors = np.full((m, p), np.nan)
    r2 = np.full(m, np.nan)
    chi2 = np.full(m, np.nan)
    ok = np.isfinite(theta).all(axis=1)
    if ok.any():
        sub = pr.rows(np.flatnonzero(ok))
        th = theta[ok]
        f0 = model(th, sub.x, sub.x0)
        unbounded = np.full_like(th, np.inf)
        jac = _jacobian(model, th, f0, sub.x, sub.x0, unbounded, pr.free) * sub.weights[:, :, None]
        normal = np.einsum("knp,knq->kpq", jac, jac)[:, pr.free][:, :, pr.free]
        residual = (f0 - sub.target) * sub.weights
        dof = np.maximum(sub.valid.sum(axis=1) - int(pr.free.sum()), 1)
        covariance = np.linalg.pinv(normal)
        if pr.sigmas is None:  # no uncertainties: the errors scaled by the reduced chi-square
            covariance = (
                covariance * (np.einsum("ij,ij->i", residual, residual) / dof)[:, None, None]
            )
        part = np.full((th.shape[0], p), np.nan)
        part[:, pr.free] = np.sqrt(np.clip(np.einsum("kpp->kp", covariance), 0.0, None))
        errors[ok] = part * sub.factor
        best = f0 * sub.scale[:, None]
        resid = np.where(sub.valid, sub.curves - best, 0.0)
        count = np.maximum(sub.valid.sum(axis=1), 1)
        mean = np.where(sub.valid, sub.curves, 0.0).sum(axis=1) / count
        total = np.where(sub.valid, (sub.curves - mean[:, None]) ** 2, 0.0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            r2[ok] = np.where(total > 0, 1.0 - (resid**2).sum(axis=1) / total, np.nan)
        if sub.sigmas is not None:
            resid = np.where(sub.valid, resid / np.where(sub.valid, sub.sigmas, 1.0), 0.0)
        chi2[ok] = (resid**2).sum(axis=1) / dof
    values = theta * pr.factor
    return values, errors, r2, chi2, converged & ok


def fit_batch(
    x: Any,
    curves: Any,
    peaks: Sequence[Mapping[str, Any]] | None = None,
    background: str = "linear",
    sigmas: Any = None,
    x_range: Sequence[float | None] | None = None,
    n_peaks: int = 1,
    shape: str = "gaussian",
    background_params: Mapping[str, Any] | None = None,
    warm_start: bool = True,
    max_iterations: int = MAX_ITERATIONS,
) -> BatchFit:
    """Fit the same peaks on a background to every row of ``curves`` at once.

    ``x`` is the curves' x, one row for all or one per curve (``(curves,
    points)``); NaN points and those outside ``x_range`` are left out, curve by
    curve. ``peaks``, ``background``, ``sigmas`` (one row per curve), ``n_peaks``,
    ``shape`` and ``background_params`` are those of
    :func:`bexa.analysis.peakfit.fit_peaks`, which fits the same models with the
    same start values, bounds and errors one curve at a time. With
    ``warm_start`` the curves fitted badly (not converged, or ``r2`` under 0.5)
    are fitted again from the last curve before them fitted well, the result
    with the lower chi-square kept: what follows a peak through a noisy series.
    """
    rows = np.atleast_2d(np.asarray(curves, dtype=float))
    m, n = rows.shape
    xs = np.asarray(x, dtype=float)
    if xs.ndim == 2 and xs.shape[0] == 1:
        xs = xs[0]
    if xs.shape[-1] != n:
        raise ValueError(f"{xs.shape[-1]} x values for curves of {n} points")
    if xs.ndim == 1 and n > 1 and np.any(np.diff(xs) < 0):  # in order of x, as fit_peaks sorts
        order = np.argsort(xs, kind="stable")
        xs, rows = xs[order], rows[:, order]
        sigmas = None if sigmas is None else np.atleast_2d(np.asarray(sigmas, float))[:, order]
    errs = None if sigmas is None else np.atleast_2d(np.asarray(sigmas, dtype=float))
    grid = np.broadcast_to(xs, (m, n))
    valid = np.isfinite(rows) & np.isfinite(grid)
    if errs is not None:
        valid &= np.isfinite(errs) & (errs > 0)
    if x_range is not None:
        lo, hi = x_range
        if lo is not None:
            valid &= grid >= lo
        if hi is not None:
            valid &= grid <= hi
    usable = valid.sum(axis=1)
    first = int(np.argmax(usable)) if m else 0
    xi, yi = grid[first][valid[first]], rows[first][valid[first]]
    if xi.size < 3:
        raise ValueError(f"only {xi.size} usable points to fit")
    spec = list(peaks) if peaks is not None else guess_peaks(xi, yi, n_peaks, shape)
    shapes, params = _parameters(xi, yi, spec, background, background_params)
    names = list(params)
    free = np.array([not params[name].fixed for name in names])
    start, lower, upper = _starts(
        xs, rows, valid, names, peaks, background, background_params, n_peaks, shape
    )
    # intensity parameters are fitted on each curve divided by its largest value, as fit_peaks
    scale = np.where(valid, np.abs(rows), 0.0).max(axis=1)
    scale = np.where(scale > 0, scale, 1.0)
    intensity = np.array([name.split("_", 1)[1] in INTENSITY_PARAMS for name in names])
    factor = np.where(intensity[None, :], scale[:, None], 1.0)
    weights = valid.astype(float)
    if errs is not None:
        weights = np.where(valid, scale[:, None] / np.where(valid, errs, 1.0), 0.0)
    x0 = 0.5 * (
        np.where(valid, grid, np.inf).min(axis=1) + np.where(valid, grid, -np.inf).max(axis=1)
    )
    problem = _Problem(
        xs,
        x0[:, None],
        rows,
        valid,
        errs,
        np.where(valid, rows, 0.0) / scale[:, None],
        weights,
        scale,
        factor,
        lower,
        upper,
        free,
    )
    model = _Model(shapes, background, names)
    p = len(names)
    out = [
        np.full((m, p), np.nan),
        np.full((m, p), np.nan),
        np.full(m, np.nan),
        np.full(m, np.nan),
        np.zeros(m, bool),
    ]
    fitted = np.flatnonzero(usable >= max(3, int(free.sum()) + 1))
    if fitted.size:
        for whole, part in zip(
            out, _run(model, problem.rows(fitted), start[fitted], max_iterations), strict=True
        ):
            whole[fitted] = part

    def retry(index: np.ndarray, starts: np.ndarray) -> None:
        """Fit the curves ``index`` again from ``starts``; keep the lower chi-squares."""
        again = _run(model, problem.rows(index), starts, max_iterations, "refits")
        old = np.where(np.isfinite(out[3][index]), out[3][index], np.inf)
        better = again[4] & (np.nan_to_num(again[3], nan=np.inf) < old)
        for whole, part in zip(out, again, strict=True):
            whole[index[better]] = part[better]

    def poor() -> np.ndarray:
        return fitted[~(out[4][fitted] & (out[2][fitted] >= GOOD_R2))]

    if peaks is not None and n_peaks == 1 and shapes[0] != "step" and poor().size:
        # a peak placed by hand far from the curve's: started again from the curve's own
        own, _, _ = _starts(xs, rows, valid, names, None, background, None, 1, shapes[0])
        given = np.array([not params[n].fixed for n in names])  # fixed values stay
        own = np.where(given[None, :], own, start)
        retry(poor(), own[poor()])
    if warm_start and fitted.size > 1 and poor().size:
        good = out[4] & (out[2] >= GOOD_R2)
        last_good = np.maximum.accumulate(np.where(good, np.arange(m), -1))
        again = poor()
        again = again[last_good[again] >= 0]
        if again.size:
            retry(again, out[0][last_good[again]])
    values, errors, r2, chi2, success = out
    return BatchFit(shapes, background, names, values, errors, r2, chi2, success, x0)
