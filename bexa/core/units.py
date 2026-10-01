"""Physical constants and unit conversions used across bexa.

Conventions: energies in keV, wavelengths and d-spacings in Angstrom, angles in
degrees at the API boundary (radians only inside numerical kernels), lengths on
the detector in millimetres, pixel sizes in micrometres.
"""

from __future__ import annotations

from typing import Any

import numpy as np

HC_KEV_ANGSTROM = 12.398419843  # h*c in keV*Angstrom
D_SI111_ANGSTROM = 3.1356  # d-spacing of Si(111), used by channel-cut monochromators
EV_PER_JOULE = 6.241509074e18
FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))  # 2.3548

MONO_D_SPACINGS_ANGSTROM: dict[str, float] = {
    "Si111": D_SI111_ANGSTROM,
    "Si220": 1.9201,
    "Si311": 1.6375,
    "Si333": D_SI111_ANGSTROM / 3.0,
    "Ge111": 3.2664,
    "C111": 2.0593,  # diamond
}


def energy_to_wavelength(energy_keV: float | np.ndarray) -> float | np.ndarray:
    """Wavelength in Angstrom for a photon energy in keV: ``lambda = hc / E``."""
    return HC_KEV_ANGSTROM / np.asarray(energy_keV, dtype=float)


def wavelength_to_energy(wavelength_A: float | np.ndarray) -> float | np.ndarray:
    """Photon energy in keV for a wavelength in Angstrom."""
    return HC_KEV_ANGSTROM / np.asarray(wavelength_A, dtype=float)


def wavevector(energy_keV: float) -> float:
    """Wavevector magnitude ``k = 2 pi / lambda`` in inverse Angstrom."""
    return 2.0 * np.pi / float(energy_to_wavelength(energy_keV))


def bragg_angle_deg(d_spacing_A: float, energy_keV: float) -> float:
    """Bragg angle theta in degrees from ``lambda = 2 d sin(theta)``.

    Raises
    ------
    ValueError
        If the reflection is not reachable at this energy (``lambda > 2d``).
    """
    ratio = float(energy_to_wavelength(energy_keV)) / (2.0 * d_spacing_A)
    if ratio > 1.0:
        raise ValueError(
            f"d = {d_spacing_A:.4f} A is not reachable at {energy_keV:.3f} keV "
            f"(lambda / 2d = {ratio:.3f} > 1)"
        )
    return float(np.degrees(np.arcsin(ratio)))


def d_spacing_from_angle(theta_deg: float, energy_keV: float) -> float:
    """d-spacing in Angstrom from a Bragg angle in degrees and an energy in keV."""
    return float(energy_to_wavelength(energy_keV)) / (2.0 * np.sin(np.radians(theta_deg)))


def mono_angle_to_energy(theta_deg: float | np.ndarray, crystal: str | float = "Si111") -> Any:
    """Energy in keV selected by a monochromator crystal at Bragg angle ``theta_deg``.

    ESRF ID03 records the channel-cut angle as ``ccmth``; with Si(111) this
    gives ``E = hc / (2 d sin(ccmth))``. ``crystal`` may be a d-spacing in
    angstrom; an array of angles gives an array of energies.
    """
    d = MONO_D_SPACINGS_ANGSTROM[crystal] if isinstance(crystal, str) else float(crystal)
    return HC_KEV_ANGSTROM / (2.0 * d * np.sin(np.radians(theta_deg)))


def energy_to_mono_angle(energy_keV: float, crystal: str = "Si111") -> float:
    """Monochromator Bragg angle in degrees that selects ``energy_keV``."""
    d = MONO_D_SPACINGS_ANGSTROM[crystal] if isinstance(crystal, str) else float(crystal)
    return bragg_angle_deg(d, energy_keV)


def photons_from_pulse_energy(pulse_energy_J: float, photon_energy_eV: float) -> float:
    """Number of photons in a pulse of ``pulse_energy_J`` joules at ``photon_energy_eV``."""
    return pulse_energy_J * EV_PER_JOULE / photon_energy_eV


def sigma_to_fwhm(sigma: float | np.ndarray) -> float | np.ndarray:
    """Full width at half maximum of a Gaussian with standard deviation ``sigma``."""
    return FWHM_PER_SIGMA * np.asarray(sigma, dtype=float)


def fwhm_to_sigma(fwhm: float | np.ndarray) -> float | np.ndarray:
    """Standard deviation of a Gaussian with full width at half maximum ``fwhm``."""
    return np.asarray(fwhm, dtype=float) / FWHM_PER_SIGMA
