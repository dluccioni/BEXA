"""Zone-axis solver: which crystal direction lies along the lab horizontal.

Port of ``ZoneAxis_Solver_v1.py``. Given one indexed reflection and the motor
and detector angles at which it was found, every integer direction up to a
search range is tried as the zone axis; the ones whose predicted Q vector
matches the observed one best are returned, largest d-spacing first. The
candidate scoring is vectorised over all directions at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd
from typing import Any

import numpy as np
from scipy.optimize import minimize

from bexa.core.units import HC_KEV_ANGSTROM
from bexa.crystal.diffractometer import SAMPLE_MOTORS, orientation_matrix
from bexa.crystal.lattice import BravaisLattice, Crystal, LatticeParameters
from bexa.geometry.conventions import Convention, get_convention

HklTuple = tuple[int, int, int]
ANGLE_NAMES = ("mu", "chi", "phi", "omega", "two_theta", "eta")
WRAPPING_ANGLES = ("phi", "omega", "eta")

__all__ = [
    "AngleFit",
    "ZoneAxisCandidate",
    "ZoneAxisSolver",
    "normalize_zone_axis",
    "reduce_indices",
    "solve_zone_axis",
]


@dataclass
class AngleFit:
    """Angles within the tolerances that reproduce the reflection for one zone axis."""

    angles: dict[str, float]
    residual: float
    differences: dict[str, float]


@dataclass
class ZoneAxisCandidate:
    """One possible zone axis and how well it explains the observation."""

    zone_axis: HklTuple
    residual: float
    d_spacing: float
    matched_hkl: HklTuple
    fit: AngleFit | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def within_tolerance(self) -> bool:
        return self.fit is not None


def reduce_indices(hkl: Any) -> HklTuple:
    """Divide by the common factor, keeping the sign: ``(-3, 3, 0)`` -> ``(-1, 1, 0)``."""
    h, k, l = (int(v) for v in hkl)
    g = 0
    for v in (h, k, l):
        g = gcd(g, abs(v))
    return (h // g, k // g, l // g) if g > 1 else (h, k, l)


def normalize_zone_axis(hkl: Any) -> HklTuple:
    """Reduce the indices and make the first non-zero one positive (a direction, sign ignored)."""
    h, k, l = reduce_indices(hkl)
    for v in (h, k, l):
        if v != 0:
            if v < 0:
                h, k, l = -h, -k, -l
            break
    return (h, k, l)


class ZoneAxisSolver:
    """Find the zone axis from one observed reflection.

    Parameters
    ----------
    crystal
        Lattice with its Bravais class.
    energy_keV
        X-ray energy.
    out_of_plane
        Miller indices of the surface normal (lab vertical at zero angles).
    convention, motor_order, eta_zero_offset
        As for :class:`bexa.crystal.diffractometer.FourCircleDiffractometer`.
    """

    def __init__(
        self,
        crystal: Crystal,
        energy_keV: float,
        out_of_plane: Any,
        convention: str | Convention = "pal_fourc",
        motor_order: Any = None,
        eta_zero_offset: float = 0.0,
    ) -> None:
        from dataclasses import replace

        conv = get_convention(convention)
        if motor_order is not None:
            conv = replace(conv, motor_order=tuple(motor_order))
        self.convention = conv.with_eta_offset(eta_zero_offset)
        self.crystal = crystal
        self.energy_keV = float(energy_keV)
        self.wavelength = HC_KEV_ANGSTROM / self.energy_keV
        self.k = 2 * np.pi / self.wavelength
        self.out_of_plane = tuple(int(v) for v in out_of_plane)
        self._basis = tuple(
            np.asarray(v, dtype=float)
            for v in (self.convention.beam, self.convention.horizontal, self.convention.vertical)
        )

    # ------------------------------------------------------------ geometry
    def kf_direction(self, two_theta_deg: float, eta_deg: float) -> np.ndarray:
        """Unit ``k_f`` for detector angles under this eta convention."""
        beam, horizontal, vertical = self._basis
        tt = np.radians(two_theta_deg)
        eta = np.radians(eta_deg + self.convention.eta_zero_offset)
        if self.convention.eta_mode == "sacla":
            across = -np.cos(eta) * horizontal + np.sin(eta) * vertical
        else:
            across = np.sin(eta) * horizontal + np.cos(eta) * vertical
        return np.cos(tt) * beam + np.sin(tt) * across

    def detector_to_q_lab(self, two_theta_deg: float, eta_deg: float) -> np.ndarray:
        """Lab-frame Q for a peak seen at ``two_theta`` and ``eta``."""
        return self.k * (self.kf_direction(two_theta_deg, eta_deg) - self._basis[0])

    def sample_rotation(self, mu: float, chi: float, phi: float, omega: float) -> np.ndarray:
        return self.convention.sample_rotation(mu=mu, chi=chi, phi=phi, omega=omega)

    # ------------------------------------------------------- candidate sets
    def candidate_axes(self, search_range: int) -> np.ndarray:
        """Integer directions up to ``search_range``, excluding the surface normal's direction."""
        r = np.arange(-search_range, search_range + 1)
        axes = np.stack(np.meshgrid(r, r, r, indexing="ij"), axis=-1).reshape(-1, 3)
        axes = axes[np.any(axes != 0, axis=1)]
        q = axes @ self.crystal.B.T  # (N, 3) reciprocal vectors
        q_oop = self.crystal.Q_vector(self.out_of_plane)
        cos_angle = np.abs(q @ q_oop) / (np.linalg.norm(q, axis=1) * np.linalg.norm(q_oop))
        return axes[cos_angle <= 0.99]

    def orientation_matrices(self, axes: np.ndarray) -> np.ndarray:
        """U matrices ``(N, 3, 3)`` for many zone axes at once (see :func:`orientation_matrix`)."""
        q_oop = self.crystal.Q_vector(self.out_of_plane)
        z = q_oop / np.linalg.norm(q_oop)
        q_zone = np.asarray(axes, dtype=float) @ self.crystal.B.T
        y = q_zone - (q_zone @ z)[:, None] * z
        y = y / np.linalg.norm(y, axis=1)[:, None]
        x = np.cross(y, np.broadcast_to(z, y.shape))
        return np.stack([x, y, np.broadcast_to(z, y.shape)], axis=1)

    # -------------------------------------------------------------- solving
    def find_zone_axis(
        self,
        hkl_observed: Any,
        mu: float,
        chi: float,
        phi: float,
        omega: float,
        two_theta: float,
        eta: float,
        search_range: int = 10,
        num_results: int = 1,
        angle_tolerances: dict[str, float] | None = None,
    ) -> list[ZoneAxisCandidate]:
        """Candidates sorted by d-spacing (largest first), then by residual.

        Every equivalent of ``hkl_observed`` is tried for every zone axis; the
        residual is the smallest ``|Q_predicted - Q_observed|`` (1/A). Candidates
        within ten times the best residual (at least 1/A) are kept. With
        ``angle_tolerances``, only zone axes for which motor angles within those
        tolerances reproduce the reflection survive, and their fit is attached.
        """
        hkl_observed = tuple(int(v) for v in hkl_observed)
        Q_observed = self.detector_to_q_lab(two_theta, eta)
        R = self.sample_rotation(mu, chi, phi, omega)
        family = np.array(self.crystal.equivalent_reflections(hkl_observed), dtype=float)
        axes = self.candidate_axes(search_range)
        U = self.orientation_matrices(axes)  # (N, 3, 3)
        RUB = np.einsum("ij,njk,kl->nil", R, U, self.crystal.B)  # (N, 3, 3)
        predicted = np.einsum("nil,fl->nfi", RUB, family)  # (N, F, 3)
        residuals = np.linalg.norm(predicted - Q_observed, axis=2)  # (N, F)
        best_f = residuals.argmin(axis=1)
        best_residual = residuals[np.arange(len(axes)), best_f]

        # One entry per direction, sign ignored as in the legacy (an axis and its negative are
        # 180 degrees apart about the surface normal; both index the same peak through
        # symmetry-related reflections). The stored axis keeps its sign so the matched
        # reflection stays consistent with it, and is reduced so the d-spacing ordering does
        # not depend on which multiple the search met first.
        by_direction: dict[HklTuple, ZoneAxisCandidate] = {}
        for i in range(len(axes)):
            axis = reduce_indices((int(axes[i, 0]), int(axes[i, 1]), int(axes[i, 2])))
            key = normalize_zone_axis(axis)
            residual = float(best_residual[i])
            if key not in by_direction or residual < by_direction[key].residual:
                matched = tuple(int(v) for v in family[best_f[i]])
                by_direction[key] = ZoneAxisCandidate(
                    axis,
                    residual,
                    self.crystal.d_spacing(axis),
                    (matched[0], matched[1], matched[2]),
                )
        candidates = list(by_direction.values())
        if not candidates:
            return []
        threshold = max(min(c.residual for c in candidates) * 10, 1.0)
        candidates = [c for c in candidates if c.residual <= threshold]

        observed = {
            "mu": mu, "chi": chi, "phi": phi, "omega": omega, "two_theta": two_theta, "eta": eta
        }  # fmt: skip
        details = {
            "Q_lab_observed": Q_observed,
            "Q_magnitude": float(np.linalg.norm(Q_observed)),
            "candidates_tested": len(axes),
            "search_range": search_range,
            "hkl_family_size": len(family),
            "angle_tolerances": angle_tolerances,
        }
        if angle_tolerances is not None:
            kept = []
            for c in candidates:
                c.fit = self.refine_angles(c.zone_axis, c.matched_hkl, observed, angle_tolerances)
                if c.fit is not None:
                    kept.append(c)
            candidates = kept
        candidates.sort(key=lambda c: (-c.d_spacing, c.residual))
        for c in candidates:
            c.details = details
        return candidates[:num_results]

    def refine_angles(
        self,
        zone_axis: Any,
        hkl: Any,
        observed: dict[str, float],
        tolerances: dict[str, float],
    ) -> AngleFit | None:
        """Angles within ``tolerances`` of the observed ones that put ``hkl`` on the detector.

        ``two_theta`` is fixed by Bragg's law and only checked; the other five
        angles are adjusted by a bounded local optimisation of the Q mismatch.
        Returns ``None`` when no such angles exist.
        """
        try:
            U = orientation_matrix(self.crystal, self.out_of_plane, zone_axis)
        except ValueError:
            return None
        Q_crystal = U @ self.crystal.Q_vector(hkl)
        q_mag = float(np.linalg.norm(Q_crystal))
        if q_mag < 1e-10:
            return None
        sin_theta = self.wavelength * q_mag / (4 * np.pi)
        if abs(sin_theta) > 1:
            return None
        two_theta_required = float(np.degrees(2 * np.arcsin(sin_theta)))
        two_theta_obs = observed.get("two_theta", two_theta_required)
        if abs(two_theta_required - two_theta_obs) > tolerances.get("two_theta", np.inf):
            return None

        names = ("mu", "chi", "phi", "omega", "eta")
        obs = np.array([observed.get(n, 0.0) for n in names], dtype=float)
        tol = np.array([tolerances.get(n, 180.0) for n in names], dtype=float)
        bounds = [(max(-180.0, o - t), min(180.0, o + t)) for o, t in zip(obs, tol, strict=True)]

        def objective(params: np.ndarray) -> float:
            mu, chi, phi, omega, eta = params
            Q_required = self.detector_to_q_lab(two_theta_required, eta)
            Q_calc = self.sample_rotation(mu, chi, phi, omega) @ Q_crystal
            return float(np.linalg.norm(Q_calc - Q_required))

        best = minimize(objective, obs, method="L-BFGS-B", bounds=bounds, options={"maxiter": 50})
        if best.fun > 0.01:
            effective = np.maximum(tol, 0.1)
            for offset in (
                (effective[0] / 2, 0, 0, 0, 0),
                (-effective[0] / 2, 0, 0, 0, 0),
                (0, effective[1] / 2, 0, 0, 0),
                (0, 0, 0, effective[3] / 2, 0),
                (0, 0, 0, 0, effective[4] / 2),
            ):
                start = np.clip(
                    obs + np.asarray(offset), [b[0] for b in bounds], [b[1] for b in bounds]
                )
                result = minimize(
                    objective, start, method="L-BFGS-B", bounds=bounds, options={"maxiter": 30}
                )
                if result.fun < best.fun:
                    best = result
                if best.fun < 0.01:
                    break
        if best.fun > 0.1:
            return None
        angles = dict(zip(names, (float(v) for v in best.x), strict=True))
        angles["two_theta"] = two_theta_required
        differences = {}
        for name in ANGLE_NAMES:
            diff = abs(angles[name] - observed.get(name, angles[name]))
            if name in WRAPPING_ANGLES:
                diff = min(diff, 360 - diff)
            differences[name] = float(diff)
        return AngleFit(angles, float(best.fun), differences)

    def validate(
        self,
        zone_axis: Any,
        hkl: Any,
        mu: float,
        chi: float,
        phi: float,
        omega: float,
        two_theta: float,
        eta: float,
        tolerance: float = 0.5,
    ) -> dict[str, Any]:
        """Forward check: do the observed angles reproduce ``two_theta`` and ``eta``?"""
        try:
            U = orientation_matrix(self.crystal, self.out_of_plane, zone_axis)
        except ValueError as exc:
            return {"valid": False, "error": str(exc)}
        Q_lab = self.sample_rotation(mu, chi, phi, omega) @ (U @ self.crystal.Q_vector(hkl))
        two_theta_calc, eta_calc = self.convention.detector_angles(Q_lab, self.k)
        kf = Q_lab + self.k * self._basis[0]
        two_theta_diff = abs(two_theta_calc - two_theta)
        eta_diff = abs(eta_calc - eta)
        return {
            "valid": bool(two_theta_diff < tolerance and eta_diff < tolerance),
            "two_theta_calculated": two_theta_calc,
            "two_theta_observed": two_theta,
            "two_theta_diff": two_theta_diff,
            "eta_calculated": eta_calc,
            "eta_observed": eta,
            "eta_diff": eta_diff,
            "Q_lab": Q_lab,
            "k_f": kf / np.linalg.norm(kf) * self.k,
        }


def solve_zone_axis(
    lattice: LatticeParameters | dict[str, float],
    energy_keV: float,
    out_of_plane: Any,
    hkl_observed: Any,
    angles: dict[str, float],
    bravais: str | BravaisLattice | None = None,
    convention: str | Convention = "pal_fourc",
    eta_zero_offset: float = 0.0,
    search_range: int = 10,
    num_results: int = 1,
    angle_tolerances: dict[str, float] | None = None,
) -> list[ZoneAxisCandidate]:
    """One-call zone-axis solving; ``angles`` holds mu, chi, phi, omega, two_theta and eta."""
    if isinstance(lattice, dict):
        lattice = LatticeParameters(**lattice)
    solver = ZoneAxisSolver(
        Crystal(lattice, bravais=bravais),
        energy_keV,
        out_of_plane,
        convention=convention,
        eta_zero_offset=eta_zero_offset,
    )
    sample = {m: float(angles.get(m, 0.0)) for m in SAMPLE_MOTORS}
    return solver.find_zone_axis(
        hkl_observed,
        sample["mu"],
        sample["chi"],
        sample["phi"],
        sample["omega"],
        float(angles["two_theta"]),
        float(angles["eta"]),
        search_range=search_range,
        num_results=num_results,
        angle_tolerances=angle_tolerances,
    )
