"""Fixed-omega settings at which two peaks of one family sit close in mu.

Port of ``omega_finder.py``. With chi = phi = 0 the lab Q of a reflection is
``R_mu @ R_omega @ UB @ hkl``; for a fixed omega the elastic condition fixes
mu analytically (up to two values per equivalent). Every omega of a grid is
evaluated at once, the closest pair of distinct reflections inside the motor
ranges is found per omega, the local minima of that gap are refined, and the
candidates are ranked by the gap. Each can be re-checked with the full solver.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from bexa.crystal.diffractometer import FourCircleDiffractometer
from bexa.geometry.conventions import eta_of_kf

HklTuple = tuple[int, int, int]
DEFAULT_RANGES = {"mu": (-45.0, -10.0), "eta": (0.0, 360.0), "two_theta": (0.0, 170.0)}

__all__ = ["OmegaCandidate", "PeakHit", "find_omega_offsets", "format_candidates", "peaks_at_omega"]


@dataclass
class PeakHit:
    """One equivalent reflection meeting the Ewald sphere inside the ranges."""

    hkl: HklTuple
    mu: float
    eta: float


@dataclass
class OmegaCandidate:
    """An omega at which two distinct reflections are close in mu."""

    omega: float
    gap: float
    first: PeakHit
    second: PeakHit
    two_theta: float
    solver_gap: float = float("nan")
    solver_solutions: int = -1
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def eta_difference(self) -> float:
        return abs(((self.first.eta - self.second.eta + 180) % 360) - 180)


class _Family:
    """Vectors and ranges shared by every omega evaluation."""

    def __init__(self, diffractometer: FourCircleDiffractometer, family: Any, ranges: dict) -> None:
        conv = diffractometer.convention
        beam, horizontal, vertical = (
            np.asarray(v, dtype=float) for v in (conv.beam, conv.horizontal, conv.vertical)
        )
        if np.dot(np.cross(horizontal, vertical), beam) < 0:
            raise ValueError("the convention's (beam, horizontal, vertical) frame is left-handed")
        axes = conv.motor_axes
        if not (
            np.allclose(axes.get("mu", ()), horizontal)
            and np.allclose(axes.get("omega", ()), vertical)
        ):
            raise NotImplementedError(
                "omega search needs mu about the horizontal and omega about the vertical"
            )
        self.diffractometer = diffractometer
        self.equivalents = diffractometer.crystal.equivalent_reflections(family)
        W = np.array([diffractometer.UB @ np.asarray(h, float) for h in self.equivalents])
        self.wb, self.wh, self.wv = W @ beam, W @ horizontal, W @ vertical
        self.k = diffractometer.k
        q = float(np.linalg.norm(W[0]))
        self.target = -(q**2) / (2 * self.k)  # component of Q along the beam on the Ewald sphere
        self.two_theta = float(
            np.degrees(np.arccos(np.clip((self.target + self.k) / self.k, -1, 1)))
        )
        self.ranges = {**DEFAULT_RANGES, **ranges}

    def hits(self, omega_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(omega index, equivalent index, mu, eta)`` of every hit for the omegas given."""
        om = np.radians(np.asarray(omega_deg, dtype=float))[:, None]
        vb = np.cos(om) * self.wb - np.sin(om) * self.wh  # (n_omega, n_equiv)
        vh = np.sin(om) * self.wb + np.cos(om) * self.wh
        vv = np.broadcast_to(self.wv, vb.shape)
        radius = np.hypot(vb, vv)
        with np.errstate(invalid="ignore", divide="ignore"):
            reachable = radius >= abs(self.target)
            half = np.arccos(np.clip(self.target / np.where(reachable, radius, 1.0), -1, 1))
        lift = np.arctan2(vv, vb)
        mu_lo, mu_hi = self.ranges["mu"]
        eta_lo, eta_hi = self.ranges["eta"]
        out_i, out_j, out_mu, out_eta = [], [], [], []
        for mu in (lift + half, lift - half):
            mu_deg = (np.degrees(mu) + 180) % 360 - 180
            ok = reachable & (mu_deg >= mu_lo - 1e-9) & (mu_deg <= mu_hi + 1e-9)
            mr = np.radians(mu_deg)
            kf_v = -np.sin(mr) * vb + np.cos(mr) * vv
            eta = eta_of_kf(vh, kf_v, self.diffractometer.convention)
            ok &= (eta >= eta_lo - 1e-9) & (eta <= eta_hi + 1e-9)
            i, j = np.nonzero(ok)
            out_i.append(i)
            out_j.append(j)
            out_mu.append(mu_deg[i, j])
            out_eta.append(eta[i, j])
        return (
            np.concatenate(out_i),
            np.concatenate(out_j),
            np.concatenate(out_mu),
            np.concatenate(out_eta),
        )


def peaks_at_omega(
    diffractometer: FourCircleDiffractometer, family: Any, omega: float, ranges: dict | None = None
) -> list[PeakHit]:
    """Reflections of ``family`` that meet the Ewald sphere at ``omega`` inside the ranges."""
    fam = _Family(diffractometer, family, ranges or {})
    tt_lo, tt_hi = fam.ranges["two_theta"]
    if not tt_lo <= fam.two_theta <= tt_hi:
        return []
    _, j, mu, eta = fam.hits(np.array([omega]))
    return [
        PeakHit(fam.equivalents[jj], float(m), float(e))
        for jj, m, e in zip(j, mu, eta, strict=True)
    ]


def _closest_pairs(
    fam: _Family, omegas: np.ndarray
) -> list[tuple[float, int, int, float, float, float, float] | None]:
    """Per omega: ``(gap, j_a, j_b, mu_a, eta_a, mu_b, eta_b)`` of the closest distinct pair."""
    i, j, mu, eta = fam.hits(omegas)
    order = np.argsort(i, kind="stable")
    i, j, mu, eta = i[order], j[order], mu[order], eta[order]
    starts = np.searchsorted(i, np.arange(len(omegas)))
    stops = np.searchsorted(i, np.arange(len(omegas)), side="right")
    results: list[Any] = []
    for start, stop in zip(starts, stops, strict=True):
        if stop - start < 2:
            results.append(None)
            continue
        jj, mm, ee = j[start:stop], mu[start:stop], eta[start:stop]
        gap = np.abs(mm[:, None] - mm[None, :])
        gap[jj[:, None] == jj[None, :]] = np.inf  # same reflection twice does not count
        a, b = np.unravel_index(np.argmin(gap), gap.shape)
        if not np.isfinite(gap[a, b]):
            results.append(None)
            continue
        results.append(
            (
                float(gap[a, b]),
                int(jj[a]),
                int(jj[b]),
                float(mm[a]),
                float(ee[a]),
                float(mm[b]),
                float(ee[b]),
            )
        )
    return results


def find_omega_offsets(
    diffractometer: FourCircleDiffractometer,
    family: Any,
    ranges: dict[str, tuple[float, float]] | None = None,
    omega_scan: tuple[float, float] = (-180.0, 180.0),
    step: float = 0.01,
    top_n: int = 5,
    omega_base: float = 0.0,
    verify: bool = False,
) -> list[OmegaCandidate]:
    """Omega settings (best first) where two peaks of ``family`` are closest in mu.

    Parameters
    ----------
    ranges
        Allowed ``mu``, ``eta`` and ``two_theta`` windows; missing ones default
        to :data:`DEFAULT_RANGES`.
    verify
        Re-solve every candidate with the full solver at that fixed omega and
        record the smallest mu gap it finds (slow: one family solve per candidate).
    """
    fam = _Family(diffractometer, family, ranges or {})
    tt_lo, tt_hi = fam.ranges["two_theta"]
    if not tt_lo <= fam.two_theta <= tt_hi:
        return []
    grid = np.arange(omega_scan[0], omega_scan[1], step)
    pairs = _closest_pairs(fam, grid)
    gaps = np.array([p[0] if p else np.inf for p in pairs])
    minima = [
        j
        for j in range(1, len(grid) - 1)
        if np.isfinite(gaps[j]) and gaps[j] <= gaps[j - 1] and gaps[j] <= gaps[j + 1]
    ]
    candidates: list[OmegaCandidate] = []
    for j in minima:
        fine = np.linspace(grid[j] - step, grid[j] + step, 2001)
        fine_pairs = _closest_pairs(fam, fine)
        best = min(((p[0], k) for k, p in enumerate(fine_pairs) if p is not None), default=None)
        if best is None:
            continue
        gap, k = best
        pair = fine_pairs[k]
        assert pair is not None
        _, ja, jb, mu_a, eta_a, mu_b, eta_b = pair
        candidates.append(
            OmegaCandidate(
                float(fine[k]),
                gap,
                PeakHit(fam.equivalents[ja], mu_a, eta_a),
                PeakHit(fam.equivalents[jb], mu_b, eta_b),
                fam.two_theta,
            )
        )
    # one candidate per cluster of omegas closer than 0.3 deg (keep the smallest gap)
    candidates.sort(key=lambda c: c.omega)
    distinct: list[OmegaCandidate] = []
    for c in candidates:
        if distinct and abs(c.omega - distinct[-1].omega) < 0.3:
            if c.gap < distinct[-1].gap:
                distinct[-1] = c
        else:
            distinct.append(c)
    eta_lo, eta_hi = fam.ranges["eta"]

    def margin(c: OmegaCandidate) -> float:
        return min(
            c.first.eta - eta_lo, eta_hi - c.first.eta, c.second.eta - eta_lo, eta_hi - c.second.eta
        )

    distinct.sort(key=lambda c: (round(c.gap, 4), -margin(c)))
    chosen = distinct[:top_n]
    for c in chosen:
        c.extra["omega_offset"] = c.omega - omega_base
        if verify:
            c.solver_gap, c.solver_solutions = _verify(diffractometer, family, c.omega)
    return chosen


def _verify(
    diffractometer: FourCircleDiffractometer, family: Any, omega: float
) -> tuple[float, int]:
    """Smallest mu gap among the full solver's solutions with omega fixed."""
    saved = diffractometer.motors["omega"]
    diffractometer.set_motor_range("omega", omega, omega)
    try:
        solutions = diffractometer.solve_for_peak_family(family, use_equivalents=True)
    finally:
        diffractometer.motors["omega"] = saved
    mus = sorted(s.mu for s in solutions)
    gap = min((b - a for a, b in itertools.pairwise(mus)), default=float("nan"))
    return float(gap), len(solutions)


def format_candidates(candidates: list[OmegaCandidate], omega_base: float = 0.0) -> str:
    """The table ``omega_finder.py`` printed."""
    if not candidates:
        return "No candidate omega found."
    header = (
        f"{'#':>2} {'offset':>8} {'omega':>8} {'dmu':>8} {'reflA':>11} {'muA':>8} {'etaA':>7} "
        f"{'reflB':>11} {'muB':>8} {'etaB':>7} {'d_eta':>6} {'solver':>10}"
    )
    lines = [header]
    for rank, c in enumerate(candidates, 1):
        solver = f"{c.solver_gap:>7.4f}/{c.solver_solutions}" if c.solver_solutions >= 0 else "-"
        lines.append(
            f"{rank:>2} {c.omega - omega_base:>8.3f} {c.omega:>8.3f} {c.gap:>8.4f} "
            f"{c.first.hkl!s:>11} {c.first.mu:>8.3f} {c.first.eta:>7.2f} "
            f"{c.second.hkl!s:>11} {c.second.mu:>8.3f} {c.second.eta:>7.2f} "
            f"{c.eta_difference:>6.1f} {solver:>10}"
        )
    best = candidates[0]
    lines.append(
        f"Recommended: omega_offset = {best.omega - omega_base:.3f} (omega = {best.omega:.3f}), "
        f"|delta mu| = {best.gap:.4f} deg, 2theta = {best.two_theta:.2f}"
    )
    return "\n".join(lines)
