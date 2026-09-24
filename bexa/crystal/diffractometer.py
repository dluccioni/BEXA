"""Four-circle diffractometer solver: motor angles that bring a reflection onto the detector.

Port of ``four_circle_diffractometer.py`` and its ``Motor_Solver_v1`` /
``Motor_Solver_SACLA`` variants, which differ only in the eta convention. The
sample stack, beam axis and eta zero come from a
:class:`bexa.geometry.conventions.Convention` (``"pal_fourc"`` or ``"sacla"``);
the lattice, B matrix and equivalent reflections from
:class:`bexa.crystal.lattice.Crystal`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from scipy.optimize import differential_evolution, minimize

from bexa.core.units import HC_KEV_ANGSTROM
from bexa.crystal.lattice import BravaisLattice, Crystal, LatticeParameters
from bexa.geometry.conventions import Convention, get_convention

SAMPLE_MOTORS = ("mu", "chi", "phi", "omega")
DETECTOR_MOTORS = ("two_theta", "eta")
HklTuple = tuple[int, int, int]
Constraint = Literal["fixed", "free", "range"]

__all__ = [
    "FourCircleDiffractometer",
    "MotorConfig",
    "Solution",
    "create_diffractometer",
    "format_solutions",
    "orientation_matrix",
]


@dataclass
class MotorConfig:
    """Constraint of one motor: fixed at a value, free, or limited to a range."""

    name: str
    constraint: Constraint = "free"
    value: float | None = None
    range_min: float = -180.0
    range_max: float = 180.0

    @classmethod
    def fixed(cls, name: str, value: float) -> MotorConfig:
        return cls(name, "fixed", value=float(value))

    @classmethod
    def free(cls, name: str, range_min: float = -180.0, range_max: float = 180.0) -> MotorConfig:
        return cls(name, "free", range_min=float(range_min), range_max=float(range_max))

    @classmethod
    def ranged(cls, name: str, range_min: float, range_max: float) -> MotorConfig:
        return cls(name, "range", range_min=float(range_min), range_max=float(range_max))

    @property
    def is_fixed(self) -> bool:
        return self.constraint == "fixed"

    def bounds(self) -> tuple[float, float]:
        if self.is_fixed and self.value is not None:
            return (self.value, self.value)
        return (self.range_min, self.range_max)

    def default_position(self) -> float:
        """Fixed value, the middle of a range, or 0 for a free motor."""
        if self.is_fixed and self.value is not None:
            return self.value
        if self.constraint == "range":
            return 0.5 * (self.range_min + self.range_max)
        return 0.0

    def penalty(self, angle: float) -> float:
        """Squared distance of ``angle`` from the allowed values (0 when inside)."""
        if self.is_fixed and self.value is not None:
            return (angle - self.value) ** 2
        if angle < self.range_min:
            return (self.range_min - angle) ** 2
        if angle > self.range_max:
            return (angle - self.range_max) ** 2
        return 0.0

    def describe(self) -> str:
        if self.is_fixed:
            return f"{self.name}: fixed at {self.value:g} deg"
        return f"{self.name}: {self.range_min:g} to {self.range_max:g} deg ({self.constraint})"


@dataclass
class Solution:
    """Motor angles that put ``hkl`` on the detector."""

    hkl: HklTuple
    mu: float
    chi: float
    phi: float
    omega: float
    two_theta: float
    eta: float
    q_magnitude: float
    d_spacing: float
    energy_keV: float
    residual: float
    motor_order: tuple[str, ...] = SAMPLE_MOTORS

    def angles(self) -> dict[str, float]:
        return {m: float(getattr(self, m)) for m in (*SAMPLE_MOTORS, *DETECTOR_MOTORS)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "hkl": self.hkl,
            **self.angles(),
            "q_magnitude": self.q_magnitude,
            "d_spacing": self.d_spacing,
            "energy_keV": self.energy_keV,
            "residual": self.residual,
        }

    def __repr__(self) -> str:
        motors = ", ".join(f"{m}={getattr(self, m):.3f}" for m in self.motor_order)
        return (
            f"Solution(hkl={self.hkl}, {motors}, two_theta={self.two_theta:.3f}, "
            f"eta={self.eta:.3f}, d={self.d_spacing:.4f} A, residual={self.residual:.2e})"
        )


def orientation_matrix(crystal: Crystal, out_of_plane: Any, zone_axis: Any) -> np.ndarray:
    """U matrix: ``out_of_plane`` becomes the lab vertical, ``zone_axis`` the lab horizontal.

    Rows are the crystal-Cartesian directions that map onto lab x (beam), y and
    z at zero motor angles, so ``U @ q_crystal = q_lab``.
    """
    q_oop = crystal.Q_vector(out_of_plane)
    q_zone = crystal.Q_vector(zone_axis)
    z = q_oop / np.linalg.norm(q_oop)
    y = q_zone - np.dot(q_zone, z) * z
    if np.linalg.norm(y) < 1e-10:
        raise ValueError("zone axis and out-of-plane direction are parallel")
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)
    return np.array([x, y, z])


class FourCircleDiffractometer:
    """Motor solutions for a crystal on a four-circle stack.

    Parameters
    ----------
    crystal
        Lattice with its Bravais class (for equivalent reflections).
    energy_keV
        X-ray energy.
    out_of_plane, zone_axis
        Miller indices along the lab vertical and along the beam-perpendicular
        horizontal at zero motor angles.
    convention
        ``"pal_fourc"`` (eta = 0 vertical, in degrees around it) or ``"sacla"``
        (eta from the -y axis, anticlockwise looking downstream, in [0, 360)).
    motor_order
        Sample motors from the bottom of the stack up (default mu, chi, phi, omega).
    eta_zero_offset
        Shift of the eta zero in degrees, subtracted from the computed eta.
    """

    def __init__(
        self,
        crystal: Crystal,
        energy_keV: float,
        out_of_plane: Any,
        zone_axis: Any,
        convention: str | Convention = "pal_fourc",
        motor_order: Any = None,
        eta_zero_offset: float = 0.0,
    ) -> None:
        conv = get_convention(convention)
        if motor_order is not None:
            order = tuple(motor_order)
            if set(order) != set(SAMPLE_MOTORS):
                raise ValueError(f"motor_order must contain exactly {SAMPLE_MOTORS}, got {order}")
            conv = replace(conv, motor_order=order)
        self.convention = conv.with_eta_offset(eta_zero_offset)
        self.crystal = crystal
        self.energy_keV = float(energy_keV)
        self.wavelength = HC_KEV_ANGSTROM / self.energy_keV
        self.k = 2 * np.pi / self.wavelength
        self.out_of_plane = tuple(int(v) for v in out_of_plane)
        self.zone_axis = tuple(int(v) for v in zone_axis)
        self.U = orientation_matrix(crystal, out_of_plane, zone_axis)
        self.UB = self.U @ crystal.B
        eta_range = (0.0, 360.0) if self.convention.eta_mode == "sacla" else (-20.0, 20.0)
        self.motors: dict[str, MotorConfig] = {
            "mu": MotorConfig.free("mu", -10, 90),
            "chi": MotorConfig.free("chi", -5, 95),
            "phi": MotorConfig.free("phi", -180, 180),
            "omega": MotorConfig.free("omega", -180, 180),
            "two_theta": MotorConfig.free("two_theta", 0, 170),
            "eta": MotorConfig.free("eta", *eta_range),
        }

    # ------------------------------------------------------------ settings
    @property
    def motor_order(self) -> tuple[str, ...]:
        return self.convention.motor_order

    @property
    def eta_zero_offset(self) -> float:
        return self.convention.eta_zero_offset

    def set_motor(self, name: str, config: MotorConfig) -> None:
        if name not in self.motors:
            raise KeyError(f"unknown motor {name!r}; motors are {sorted(self.motors)}")
        self.motors[name] = config

    def set_motor_fixed(self, name: str, value: float) -> None:
        self.set_motor(name, MotorConfig.fixed(name, value))

    def set_motor_free(self, name: str, range_min: float = -180, range_max: float = 180) -> None:
        self.set_motor(name, MotorConfig.free(name, range_min, range_max))

    def set_motor_range(self, name: str, range_min: float, range_max: float) -> None:
        self.set_motor(name, MotorConfig.ranged(name, range_min, range_max))

    def default_positions(self) -> dict[str, float]:
        return {m: self.motors[m].default_position() for m in self.motor_order}

    def describe_motors(self) -> str:
        return "\n".join(self.motors[m].describe() for m in (*self.motor_order, *DETECTOR_MOTORS))

    # ------------------------------------------------------------ geometry
    def sample_rotation(
        self, mu: float = 0.0, chi: float = 0.0, phi: float = 0.0, omega: float = 0.0
    ) -> np.ndarray:
        return self.convention.sample_rotation(mu=mu, chi=chi, phi=phi, omega=omega)

    def q_lab(self, hkl: Any, mu: float, chi: float, phi: float, omega: float) -> np.ndarray:
        """Lab-frame Q of ``hkl`` at the given motor angles."""
        return self.sample_rotation(mu, chi, phi, omega) @ (self.UB @ np.asarray(hkl, float))

    def detector_angles(self, Q_lab: np.ndarray) -> tuple[float, float]:
        """``(two_theta, eta)`` in degrees for a lab-frame Q (projected onto the Ewald sphere)."""
        return self.convention.detector_angles(Q_lab, self.k)

    def bragg_two_theta(self, hkl: Any) -> float | None:
        """Scattering angle of ``hkl`` from Bragg's law, ``None`` when unreachable."""
        q = float(np.linalg.norm(self.crystal.Q_vector(hkl)))
        sin_theta = q * self.wavelength / (4 * np.pi)
        if sin_theta > 1:
            return None
        return float(np.degrees(2 * np.arcsin(sin_theta)))

    def residual(self, angles: Any, hkl: Any, include_detector_penalty: bool = True) -> float:
        """Squared deviation from the elastic condition, plus the detector penalties."""
        values = dict(zip(self.motor_order, angles, strict=True))
        Q = self.q_lab(hkl, values["mu"], values["chi"], values["phi"], values["omega"])
        kf = Q + self.k * np.asarray(self.convention.beam, float)
        elastic = (float(np.linalg.norm(kf)) - self.k) ** 2
        if not include_detector_penalty:
            return elastic
        two_theta, eta = self.detector_angles(Q)
        penalty = self.motors["eta"].penalty(eta) + self.motors["two_theta"].penalty(two_theta)
        return elastic + 0.01 * penalty

    # -------------------------------------------------------------- solving
    def _bounds(self) -> list[tuple[float, float]]:
        bounds = []
        for motor in self.motor_order:
            config = self.motors[motor]
            if config.is_fixed and config.value is not None:
                bounds.append((config.value - 0.001, config.value + 0.001))
            else:
                bounds.append(config.bounds())
        return bounds

    def _starting_points(
        self, current: np.ndarray, default: np.ndarray, bounds: list[tuple[float, float]]
    ) -> list[np.ndarray]:
        """The legacy search pattern: single motors, pairs, a coarse grid and random points."""
        starts = [current.copy(), default.copy()]
        n = len(self.motor_order)
        for i in range(n):
            for value in np.linspace(*bounds[i], 10):
                start = current.copy()
                start[i] = value
                starts.append(start)
        for i in range(n):
            for j in range(i + 1, n):
                for vi in np.linspace(*bounds[i], 5):
                    for vj in np.linspace(*bounds[j], 5):
                        start = current.copy()
                        start[i], start[j] = vi, vj
                        starts.append(start)
        grid = {"omega": [0.0], "mu": [0, 20, 45], "chi": [0, 45, 90], "phi": [0, 90, 180, -90]}
        for mu in grid["mu"]:
            for chi in grid["chi"]:
                for phi in grid["phi"]:
                    point = {"mu": mu, "chi": chi, "phi": phi, "omega": 0.0}
                    start = np.array([point[m] for m in self.motor_order], dtype=float)
                    for i, (lo, hi) in enumerate(bounds):
                        start[i] = np.clip(start[i], lo, hi)
                    starts.append(start)
        rng = np.random.default_rng(42)
        for _ in range(20):
            starts.append(np.array([rng.uniform(lo, hi) for lo, hi in bounds]))
        return starts

    def _eta_ok(self, eta: float, tolerance: float) -> bool:
        config = self.motors["eta"]
        if config.is_fixed and config.value is not None:
            return abs(eta - config.value) <= 0.5
        return config.range_min - tolerance <= eta <= config.range_max + tolerance

    def solve_for_reflection(
        self,
        hkl: Any,
        current_positions: dict[str, float] | None = None,
        movement_threshold: float = 0.1,
    ) -> Solution | None:
        """Motor angles for one reflection, moving as few motors as possible.

        Local optimisations start from the current positions, from every motor
        swept alone and in pairs, from a coarse grid and from random points; a
        global search runs only when none of them satisfies the diffraction
        condition. Solutions must obey the eta and two_theta constraints.
        """
        hkl = tuple(int(v) for v in hkl)
        q_mag = float(np.linalg.norm(self.crystal.Q_vector(hkl)))
        d_spacing = 2 * np.pi / q_mag
        two_theta_required = self.bragg_two_theta(hkl)
        if two_theta_required is None:
            warnings.warn(
                f"reflection {hkl} not accessible at {self.energy_keV} keV (d = {d_spacing:.4f} A)",
                stacklevel=2,
            )
            return None
        tt = self.motors["two_theta"]
        if tt.is_fixed and tt.value is not None:
            if abs(two_theta_required - tt.value) > 1.0:
                return None
        elif not tt.range_min <= two_theta_required <= tt.range_max:
            return None

        bounds = self._bounds()
        default = np.array([self.motors[m].default_position() for m in self.motor_order])
        current = default.copy()
        if current_positions is not None:
            current = np.array(
                [current_positions.get(m, default[i]) for i, m in enumerate(self.motor_order)]
            )

        def movements(angles: np.ndarray) -> int:
            return int(np.sum(np.abs(angles - current) > movement_threshold))

        def objective(angles: np.ndarray) -> float:
            value = self.residual(angles, hkl)
            if value < 0.01:  # near a solution: prefer few and small movements
                value += movements(angles) * 1e-4 + np.sum(np.abs(angles - current)) * 1e-6
            return value

        best: np.ndarray | None = None
        best_key = (np.inf, np.inf)
        for start in self._starting_points(current, default, bounds):
            try:
                result = minimize(
                    objective,
                    start,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 2000, "ftol": 1e-12},
                )
            except Exception:  # a bad start must not abort the search
                continue
            if self.residual(result.x, hkl, include_detector_penalty=False) > 1e-4:
                continue
            values = dict(zip(self.motor_order, result.x, strict=True))
            _, eta = self.detector_angles(self.q_lab(hkl, **values))
            if not self._eta_ok(eta, 0.1):
                continue
            key = (movements(result.x), result.fun)
            if key < best_key:
                best_key, best = key, result.x.copy()

        if best is None:
            try:
                result = differential_evolution(
                    objective,
                    bounds,
                    maxiter=1000,
                    seed=42,
                    tol=1e-12,
                    workers=1,
                    updating="deferred",
                    mutation=(0.5, 1.0),
                    recombination=0.7,
                )
            except Exception:
                return None
            if self.residual(result.x, hkl, include_detector_penalty=False) >= 1e-4:
                return None
            values = dict(zip(self.motor_order, result.x, strict=True))
            _, eta = self.detector_angles(self.q_lab(hkl, **values))
            if not self._eta_ok(eta, 0.0):
                return None
            best_key, best = (movements(result.x), result.fun), result.x.copy()

        values = dict(zip(self.motor_order, best, strict=True))
        two_theta, eta = self.detector_angles(self.q_lab(hkl, **values))
        return Solution(
            hkl=hkl,
            mu=float(values["mu"]),
            chi=float(values["chi"]),
            phi=float(values["phi"]),
            omega=float(values["omega"]),
            two_theta=two_theta,
            eta=eta,
            q_magnitude=q_mag,
            d_spacing=d_spacing,
            energy_keV=self.energy_keV,
            residual=float(best_key[1]),
            motor_order=self.motor_order,
        )

    def solve_for_peak_family(
        self,
        hkl: Any,
        use_equivalents: bool = True,
        current_positions: dict[str, float] | None = None,
        movement_threshold: float = 0.1,
        max_solutions: int = 50,
    ) -> list[Solution]:
        """Solutions for ``hkl`` and its equivalents, fewest motor movements first."""
        hkl = tuple(int(v) for v in hkl)
        reflections = self.crystal.equivalent_reflections(hkl) if use_equivalents else [hkl]
        current = current_positions or self.default_positions()
        solutions = []
        for refl in reflections:
            solution = self.solve_for_reflection(refl, current, movement_threshold)
            if solution is not None:
                solutions.append(solution)

        def sort_key(solution: Solution) -> tuple[int, float]:
            travel = [abs(getattr(solution, m) - current.get(m, 0.0)) for m in self.motor_order]
            return sum(t > movement_threshold for t in travel), float(sum(travel))

        solutions.sort(key=sort_key)
        return solutions[:max_solutions]

    def accessible_reflections(self, max_hkl: int = 5) -> list[HklTuple]:
        """Every reflection up to ``max_hkl`` that satisfies Bragg's law at this energy."""
        out = []
        for hkl in self.crystal.reflections(max_hkl, allowed_only=False):
            if self.bragg_two_theta(hkl) is not None:
                out.append(hkl)
        return out


def create_diffractometer(
    lattice: LatticeParameters | dict[str, float],
    energy_keV: float,
    out_of_plane: Any,
    zone_axis: Any,
    bravais: str | BravaisLattice | None = None,
    convention: str | Convention = "pal_fourc",
    motor_order: Any = None,
    eta_zero_offset: float = 0.0,
    centering: str = "P",
) -> FourCircleDiffractometer:
    """Build a diffractometer from lattice parameters (dict or object) in one call."""
    if isinstance(lattice, dict):
        lattice = LatticeParameters(**lattice)
    crystal = Crystal(lattice, bravais=bravais, centering=centering)
    return FourCircleDiffractometer(
        crystal,
        energy_keV,
        out_of_plane,
        zone_axis,
        convention=convention,
        motor_order=motor_order,
        eta_zero_offset=eta_zero_offset,
    )


def format_solutions(
    solutions: list[Solution],
    diffractometer: FourCircleDiffractometer | None = None,
    movement_threshold: float = 0.1,
    max_display: int = 10,
) -> str:
    """A table of solutions; with a diffractometer, moved motors are marked with ``*``."""
    if not solutions:
        return "No solutions found."
    defaults = diffractometer.default_positions() if diffractometer is not None else {}
    order = solutions[0].motor_order
    header = (
        f"{'hkl':>12}  "
        + "  ".join(f"{m:>8}" for m in order)
        + f"  {'2theta':>8}  {'eta':>8}  {'d(A)':>8}"
    )
    lines = [f"Found {len(solutions)} solution(s):", header, "-" * len(header)]
    for sol in solutions[:max_display]:
        cells = []
        for m in order:
            angle = getattr(sol, m)
            moved = m in defaults and abs(angle - defaults[m]) > movement_threshold
            cells.append(f"{angle:>7.2f}{'*' if moved else ' '}")
        hkl = f"({sol.hkl[0]},{sol.hkl[1]},{sol.hkl[2]})"
        lines.append(
            f"{hkl:>12}  "
            + "  ".join(cells)
            + f"  {sol.two_theta:>8.2f}  {sol.eta:>8.2f}  {sol.d_spacing:>8.4f}"
        )
    if len(solutions) > max_display:
        lines.append(f"... and {len(solutions) - max_display} more")
    return "\n".join(lines)
