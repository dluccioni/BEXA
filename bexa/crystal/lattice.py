"""Lattice parameters, the B matrix, d-spacings and equivalent reflections.

One implementation replaces the three copies in ``Motor_Solver_v1.py``,
``Motor_Solver_SACLA.py`` and ``four_circle_diffractometer (1).py``, plus the
general d-spacing formula from ``Usefulthings.ipynb``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

HklTuple = tuple[int, int, int]


class BravaisLattice(str, Enum):
    CUBIC = "cubic"
    TETRAGONAL = "tetragonal"
    ORTHORHOMBIC = "orthorhombic"
    HEXAGONAL = "hexagonal"
    TRIGONAL = "trigonal"
    MONOCLINIC = "monoclinic"
    TRICLINIC = "triclinic"


@dataclass(frozen=True)
class LatticeParameters:
    """Cell lengths in Angstrom and angles in degrees."""

    a: float
    b: float
    c: float
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0

    @classmethod
    def cubic(cls, a: float) -> LatticeParameters:
        return cls(a, a, a)

    @classmethod
    def tetragonal(cls, a: float, c: float) -> LatticeParameters:
        return cls(a, a, c)

    @classmethod
    def hexagonal(cls, a: float, c: float) -> LatticeParameters:
        return cls(a, a, c, gamma=120.0)

    @classmethod
    def orthorhombic(cls, a: float, b: float, c: float) -> LatticeParameters:
        return cls(a, b, c)

    @classmethod
    def monoclinic(cls, a: float, b: float, c: float, beta: float) -> LatticeParameters:
        return cls(a, b, c, beta=beta)

    @classmethod
    def diamond(cls) -> LatticeParameters:
        return cls.cubic(3.5668)

    @classmethod
    def silicon(cls) -> LatticeParameters:
        return cls.cubic(5.43102)

    @property
    def angles_rad(self) -> tuple[float, float, float]:
        return tuple(np.radians((self.alpha, self.beta, self.gamma)))  # type: ignore[return-value]

    @property
    def volume(self) -> float:
        """Unit-cell volume in cubic Angstrom."""
        al, be, ga = self.angles_rad
        return (
            self.a
            * self.b
            * self.c
            * np.sqrt(
                1
                - np.cos(al) ** 2
                - np.cos(be) ** 2
                - np.cos(ga) ** 2
                + 2 * np.cos(al) * np.cos(be) * np.cos(ga)
            )
        )

    def metric_tensor(self) -> np.ndarray:
        al, be, ga = self.angles_rad
        a, b, c = self.a, self.b, self.c
        return np.array(
            [
                [a * a, a * b * np.cos(ga), a * c * np.cos(be)],
                [a * b * np.cos(ga), b * b, b * c * np.cos(al)],
                [a * c * np.cos(be), b * c * np.cos(al), c * c],
            ]
        )

    def d_spacing(self, hkl: Any) -> float:
        """Interplanar spacing in Angstrom for any lattice (``1/d^2 = h G* h``)."""
        return d_spacing(self, hkl)

    def as_dict(self) -> dict[str, float]:
        return {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
        }


def d_spacing(lattice: LatticeParameters, hkl: Any) -> float:
    """d-spacing from the reciprocal metric tensor; valid for triclinic cells."""
    h = np.asarray(hkl, dtype=float)
    g_star = np.linalg.inv(lattice.metric_tensor())
    inv_d2 = float(h @ g_star @ h)
    if inv_d2 <= 0:
        return float("inf")
    return 1.0 / np.sqrt(inv_d2)


def detect_bravais(lattice: LatticeParameters, tol: float = 1e-3) -> BravaisLattice:
    """Guess the lattice system from the cell parameters alone."""
    a, b, c = lattice.a, lattice.b, lattice.c
    al, be, ga = lattice.alpha, lattice.beta, lattice.gamma
    eq = lambda x, y: abs(x - y) <= tol * max(abs(x), abs(y), 1.0)  # noqa: E731
    right = eq(al, 90) and eq(be, 90) and eq(ga, 90)
    if right and eq(a, b) and eq(b, c):
        return BravaisLattice.CUBIC
    if right and eq(a, b):
        return BravaisLattice.TETRAGONAL
    if right:
        return BravaisLattice.ORTHORHOMBIC
    if eq(a, b) and eq(al, 90) and eq(be, 90) and eq(ga, 120):
        return BravaisLattice.HEXAGONAL
    if eq(a, b) and eq(b, c) and eq(al, be) and eq(be, ga):
        return BravaisLattice.TRIGONAL
    if eq(al, 90) and eq(ga, 90):
        return BravaisLattice.MONOCLINIC
    return BravaisLattice.TRICLINIC


def centering_allowed(hkl: Any, centering: str = "P") -> bool:
    """Reflection conditions of the lattice centering (P, I, F, A, B, C, R)."""
    h, k, l = (int(v) for v in hkl)
    c = centering.upper()
    if c == "P":
        return True
    if c == "I":
        return (h + k + l) % 2 == 0
    if c == "F":
        return h % 2 == k % 2 == l % 2
    if c == "A":
        return (k + l) % 2 == 0
    if c == "B":
        return (h + l) % 2 == 0
    if c == "C":
        return (h + k) % 2 == 0
    if c == "R":
        return (-h + k + l) % 3 == 0
    raise ValueError(f"unknown centering {centering!r}")


class Crystal:
    """A lattice with its B matrix and symmetry-equivalent reflections.

    ``Q = B @ (h, k, l)`` in the Cartesian crystal frame with ``a*`` along x and
    ``b*`` in the xy plane, in inverse Angstrom (the 2 pi is included).

    Parameters
    ----------
    lattice
        Cell parameters.
    bravais
        Lattice system for equivalent reflections; guessed from the cell when omitted.
    centering
        Lattice centering letter used by :meth:`is_allowed`.
    """

    def __init__(
        self,
        lattice: LatticeParameters,
        bravais: BravaisLattice | str | None = None,
        centering: str = "P",
        name: str = "",
    ) -> None:
        self.lattice = lattice
        self.bravais = BravaisLattice(bravais) if bravais else detect_bravais(lattice)
        self.centering = centering
        self.name = name
        self.B = self._b_matrix()

    def _b_matrix(self) -> np.ndarray:
        lat = self.lattice
        a, b, c = lat.a, lat.b, lat.c
        al, be, ga = lat.angles_rad
        cos_alpha_star = (np.cos(be) * np.cos(ga) - np.cos(al)) / (np.sin(be) * np.sin(ga))
        sin_alpha_star = np.sqrt(1 - cos_alpha_star**2)
        V = lat.volume
        a_star = b * c * np.sin(al) / V
        b_star = a * c * np.sin(be) / V
        c_star = a * b * np.sin(ga) / V
        self.a_star, self.b_star, self.c_star = a_star, b_star, c_star
        return (
            2
            * np.pi
            * np.array(
                [
                    [a_star, b_star * np.cos(ga), c_star * np.cos(be)],
                    [0.0, b_star * np.sin(ga), -c_star * np.sin(be) * cos_alpha_star],
                    [0.0, 0.0, c_star * np.sin(be) * sin_alpha_star],
                ]
            )
        )

    # ---------------------------------------------------------------- Q/hkl
    def Q_vector(self, hkl: Any) -> np.ndarray:
        """Q in the Cartesian crystal frame (inverse Angstrom)."""
        return self.B @ np.asarray(hkl, dtype=float)

    def hkl_from_Q(self, Q: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Continuous Miller indices of Q vectors ``(..., 3)`` in the crystal frame."""
        Q = np.asarray(Q, dtype=float)
        hkl = Q @ np.linalg.inv(self.B).T
        return hkl[..., 0], hkl[..., 1], hkl[..., 2]

    def d_spacing(self, hkl: Any) -> float:
        q = float(np.linalg.norm(self.Q_vector(hkl)))
        return float("inf") if q == 0 else 2 * np.pi / q

    def bragg_angle_deg(self, hkl: Any, energy_keV: float) -> float:
        from bexa.core.units import bragg_angle_deg

        return bragg_angle_deg(self.d_spacing(hkl), energy_keV)

    def is_allowed(self, hkl: Any) -> bool:
        return centering_allowed(hkl, self.centering)

    # --------------------------------------------------------------- symmetry
    def equivalent_reflections(self, hkl: Any) -> list[HklTuple]:
        """Symmetry-equivalent reflections of the lattice's Laue class."""
        h, k, l = (int(v) for v in hkl)
        eq: set[HklTuple] = set()
        signs = list(itertools.product((1, -1), repeat=3))
        if self.bravais == BravaisLattice.CUBIC:
            for p in itertools.permutations((h, k, l)):
                for s in signs:
                    eq.add((s[0] * p[0], s[1] * p[1], s[2] * p[2]))
        elif self.bravais == BravaisLattice.TETRAGONAL:
            for s in signs:
                eq.add((s[0] * h, s[1] * k, s[2] * l))
                eq.add((s[0] * k, s[1] * h, s[2] * l))
        elif self.bravais == BravaisLattice.HEXAGONAL:
            i = -(h + k)
            for s in signs:
                for p in ((h, k), (k, h), (k, i), (i, h), (h, i), (i, k)):
                    eq.add((s[0] * p[0], s[1] * p[1], s[2] * l))
        elif self.bravais == BravaisLattice.ORTHORHOMBIC:
            for s in signs:
                eq.add((s[0] * h, s[1] * k, s[2] * l))
        elif self.bravais == BravaisLattice.MONOCLINIC:
            for s1, s3 in itertools.product((1, -1), repeat=2):
                eq.add((s1 * h, k, s3 * l))
                eq.add((s1 * h, -k, s3 * l))
        elif self.bravais == BravaisLattice.TRIGONAL:
            i = -(h + k)
            for s3 in (1, -1):
                for p in ((h, k), (k, h), (k, i), (i, h), (h, i), (i, k)):
                    eq.add((p[0], p[1], s3 * l))
        else:
            eq.update({(h, k, l), (-h, -k, -l)})
        eq.discard((0, 0, 0))
        return sorted(eq)

    def reflections(self, max_index: int = 5, allowed_only: bool = True) -> list[HklTuple]:
        """All (h, k, l) up to ``max_index``, optionally filtered by the centering rules."""
        out: list[HklTuple] = []
        rng = range(-max_index, max_index + 1)
        for h, k, l in itertools.product(rng, rng, rng):
            if (h, k, l) == (0, 0, 0):
                continue
            if allowed_only and not self.is_allowed((h, k, l)):
                continue
            out.append((h, k, l))
        return out

    def __repr__(self) -> str:
        lat = self.lattice
        return (
            f"Crystal({self.name or self.bravais.value}: "
            f"a={lat.a:.4f}, b={lat.b:.4f}, c={lat.c:.4f}, "
            f"alpha={lat.alpha:g}, beta={lat.beta:g}, gamma={lat.gamma:g}, {self.centering})"
        )
