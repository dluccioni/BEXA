"""Energy, wavelength, monochromator angle and photon-count conversions.

Thin, documented wrappers over :mod:`bexa.core.units` plus the
``d-spacing from two theta`` and ``pulse energy to photons`` cells of the
``Usefulthings`` notebook.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bexa.core.units import (
    HC_KEV_ANGSTROM,
    bragg_angle_deg,
    energy_to_mono_angle,
    energy_to_wavelength,
    mono_angle_to_energy,
    photons_from_pulse_energy,
    wavelength_to_energy,
    wavevector,
)

__all__ = [
    "HC_KEV_ANGSTROM",
    "MONO_D_SPACINGS",
    "bragg_angle_deg",
    "d_spacing_from_two_theta",
    "energy_to_mono_angle",
    "energy_to_wavelength",
    "mono_angle_to_energy",
    "photons_from_pulse_energy",
    "summary",
    "two_theta_from_d_spacing",
    "wavelength_to_energy",
    "wavevector",
]

# d-spacings (Angstrom) of the usual monochromator reflections
MONO_D_SPACINGS = {
    "Si111": 3.1356,
    "Si220": 1.9201,
    "Si311": 1.6375,
    "Si333": 1.0452,
    "Ge111": 3.2664,
    "Ge220": 2.0002,
    "C111": 2.0593,
}


def d_spacing_from_two_theta(two_theta_deg: Any, energy_keV: Any) -> Any:
    """Lattice spacing (A) of a reflection seen at ``two_theta`` for a given energy."""
    wavelength = energy_to_wavelength(energy_keV)
    return wavelength / (2 * np.sin(np.radians(np.asarray(two_theta_deg, dtype=float) / 2)))


def two_theta_from_d_spacing(d_spacing_A: Any, energy_keV: Any) -> Any:
    """Scattering angle (deg) of a lattice spacing at a given energy."""
    return 2 * bragg_angle_deg(d_spacing_A, energy_keV)


def summary(energy_keV: float | None = None, wavelength_A: float | None = None) -> dict[str, float]:
    """Energy, wavelength, wavevector and monochromator angles for one setting."""
    if energy_keV is None and wavelength_A is None:
        raise ValueError("give energy_keV or wavelength_A")
    if energy_keV is None:
        energy_keV = float(wavelength_to_energy(wavelength_A))
    out = {
        "energy_keV": float(energy_keV),
        "wavelength_A": float(energy_to_wavelength(energy_keV)),
        "wavevector_1_A": float(wavevector(energy_keV)),
    }
    for name, d in MONO_D_SPACINGS.items():
        sin_theta = out["wavelength_A"] / (2 * d)
        if sin_theta <= 1:
            out[f"theta_{name}_deg"] = float(np.degrees(np.arcsin(sin_theta)))
    return out
