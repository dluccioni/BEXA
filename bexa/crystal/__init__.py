"""Crystallography: lattices, reflections, CIF files, diffractometer solvers.

- :mod:`bexa.crystal.lattice`: cell parameters, B matrix, equivalents, extinction rules.
- :mod:`bexa.crystal.cif`: CIF reading (with pymatgen when available).
- :mod:`bexa.crystal.diffractometer`: four-circle motor solutions for a reflection.
- :mod:`bexa.crystal.zone_axis`: the zone axis from one indexed reflection.
- :mod:`bexa.crystal.omega_search`: omega settings where two peaks are close in mu.
- :mod:`bexa.crystal.sample_frame`: the sample-to-grain matrix (``Sample2Grain.m``).
"""

from bexa.crystal.lattice import (
    BravaisLattice,
    Crystal,
    LatticeParameters,
    centering_allowed,
    d_spacing,
    detect_bravais,
)

__all__ = [
    "BravaisLattice",
    "Crystal",
    "LatticeParameters",
    "centering_allowed",
    "d_spacing",
    "detect_bravais",
]
