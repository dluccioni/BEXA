"""X-ray attenuation and transmission of elements and compounds.

``xraydb`` provides the mass attenuation coefficients (Elam tables) and
densities; NIST XCOM tables written by :mod:`bexa.optics.nist` can be used
instead. Thicknesses accept a number in micrometres or a string with a unit
(``"500um"``, ``"0.3mm"``, ``"2cm"``).
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

__all__ = [
    "KNOWN_MATERIALS",
    "attenuation_length_um",
    "linear_attenuation_1_cm",
    "mass_attenuation_cm2_g",
    "material_density",
    "parse_length_um",
    "resolve_material",
    "transmission",
]

# formula and density (g/cm3) of names that xraydb does not know
KNOWN_MATERIALS: dict[str, tuple[str, float]] = {
    "diamond": ("C", 3.515),
    "c-diamond": ("C", 3.515),
    "glassy carbon": ("C", 1.5),
    "c-glassy": ("C", 1.5),
    "luag": ("Lu3Al5O12", 6.73),
    "yag": ("Y3Al5O12", 4.56),
    "gagg": ("Gd3Al2Ga3O12", 6.63),
    "lyso": ("Lu1.8Y0.2SiO5", 7.1),
    "kapton": ("C22H10N2O5", 1.42),
    "mylar": ("C10H8O4", 1.4),
    "water": ("H2O", 1.0),
    "air": ("N1.562O0.42C0.0003Ar0.0094", 0.0012),
    "sapphire": ("Al2O3", 3.98),
    "quartz": ("SiO2", 2.65),
    "wte2": ("WTe2", 9.43),
    "ws2": ("WS2", 7.5),
    "mos2": ("MoS2", 5.06),
}

_LENGTH_UNITS_UM = {"nm": 1e-3, "um": 1.0, "µm": 1.0, "mm": 1e3, "cm": 1e4, "m": 1e6, "": 1.0}


def parse_length_um(value: float | str) -> float:
    """``"500um"`` -> 500, ``"0.3mm"`` -> 300, a bare number is micrometres."""
    if isinstance(value, (int, float)):
        return float(value)
    m = re.fullmatch(r"\s*([0-9.eE+-]+)\s*([a-zA-Zµ]*)\s*", str(value))
    if not m or m.group(2) not in _LENGTH_UNITS_UM:
        raise ValueError(f"cannot parse length {value!r}; use e.g. 500um, 0.3mm, 2cm")
    return float(m.group(1)) * _LENGTH_UNITS_UM[m.group(2)]


def resolve_material(material: str, density: float | None = None) -> tuple[str, float]:
    """``(formula, density)`` for an element, a formula, or a name in :data:`KNOWN_MATERIALS`."""
    import xraydb

    key = material.strip()
    if key.lower() in KNOWN_MATERIALS:
        formula, known_density = KNOWN_MATERIALS[key.lower()]
        return formula, float(density if density is not None else known_density)
    if density is not None:
        return key, float(density)
    found = None
    try:  # xraydb's own table of materials (Si, kapton, water, ...)
        found = xraydb.get_material(key)
    except Exception:
        found = None
    if found and found[1]:
        return str(found[0]), float(found[1])
    try:  # an element symbol
        return key, float(xraydb.atomic_density(key))
    except Exception:
        raise ValueError(
            f"unknown material {material!r}; pass density= or add it to KNOWN_MATERIALS"
        ) from None


def material_density(material: str) -> float:
    return resolve_material(material)[1]


def mass_attenuation_cm2_g(material: str, energy_keV: Any, density: float | None = None) -> Any:
    """Mass attenuation coefficient (cm2/g) from the xraydb tables."""
    import xraydb

    formula, rho = resolve_material(material, density)
    energy_eV = np.asarray(energy_keV, dtype=float) * 1e3
    return np.asarray(xraydb.material_mu(formula, energy_eV, density=rho)) / rho


def linear_attenuation_1_cm(material: str, energy_keV: Any, density: float | None = None) -> Any:
    """Linear attenuation coefficient (1/cm)."""
    formula, rho = resolve_material(material, density)
    return mass_attenuation_cm2_g(formula, energy_keV, rho) * rho


def attenuation_length_um(material: str, energy_keV: Any, density: float | None = None) -> Any:
    """Thickness (um) that transmits 1/e."""
    return 1e4 / linear_attenuation_1_cm(material, energy_keV, density)


def transmission(
    material: str, thickness: float | str, energy_keV: Any, density: float | None = None
) -> Any:
    """Fraction transmitted through ``thickness`` (um or ``"500um"``) at ``energy_keV``."""
    t_cm = parse_length_um(thickness) * 1e-4
    return np.exp(-linear_attenuation_1_cm(material, energy_keV, density) * t_cm)
