"""X-ray reflectivity: momentum transfer, footprint, critical angle, film thickness.

The functions take the incidence angle theta (degrees, half the scattering
angle of a specular theta-2theta scan) or q_z, and the measured intensity.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from bexa.core.backend import to_host
from bexa.core.units import energy_to_wavelength

__all__ = [
    "CLASSICAL_ELECTRON_RADIUS_A",
    "critical_angle",
    "footprint_fraction",
    "kiessig_thickness",
    "qz",
    "theta_from_qz",
]

CLASSICAL_ELECTRON_RADIUS_A = 2.8179403262e-5


def qz(theta_deg: Any, energy_keV: float) -> np.ndarray:
    """Momentum transfer normal to the surface, ``4 pi sin(theta) / lambda`` in 1/A."""
    wavelength = float(energy_to_wavelength(energy_keV))
    return 4.0 * math.pi * np.sin(np.radians(np.asarray(theta_deg, dtype=float))) / wavelength


def theta_from_qz(q: Any, energy_keV: float) -> np.ndarray:
    """The incidence angle (degrees) of a momentum transfer ``q`` (1/A)."""
    wavelength = float(energy_to_wavelength(energy_keV))
    return np.degrees(np.arcsin(np.asarray(q, dtype=float) * wavelength / (4.0 * math.pi)))


def footprint_fraction(
    theta_deg: Any, beam_mm: float, sample_mm: float, profile: str = "gaussian"
) -> np.ndarray:
    """The share of the beam that falls on the sample at each incidence angle.

    ``beam_mm`` is the beam height (its FWHM for a Gaussian profile) and
    ``sample_mm`` the sample length along the beam. Divide the intensity by
    this to correct for the footprint.
    """
    half_length = 0.5 * sample_mm * np.sin(np.radians(np.asarray(theta_deg, dtype=float)))
    if profile == "rectangular":
        return np.clip(2.0 * half_length / beam_mm, 0.0, 1.0)
    if profile == "gaussian":
        from scipy.special import erf

        sigma = beam_mm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        return erf(half_length / (math.sqrt(2.0) * sigma))
    raise ValueError(f"profile must be gaussian or rectangular, not {profile!r}")


def critical_angle(
    x: Any, intensity: Any, energy_keV: float | None = None, unit: str = "theta"
) -> dict[str, float]:
    """The critical angle: where the logarithm of the reflectivity falls fastest.

    ``x`` is theta in degrees (``unit="theta"``) or q_z in 1/A (``"q"``). With
    the energy the critical q_z and the electron density
    ``q_c^2 / (16 pi r_e)`` (electrons per cubic Angstrom) are given too.
    """
    xs = np.asarray(to_host(x), dtype=float)
    ys = np.asarray(to_host(intensity), dtype=float)
    keep = np.isfinite(xs) & np.isfinite(ys) & (ys > 0)
    order = np.argsort(xs[keep])
    xs, ys = xs[keep][order], np.log10(ys[keep][order])
    slope = np.gradient(ys, xs)
    # look past the first points (the direct beam's edge) and before the fringes
    start = max(int(np.argmax(ys)), 1)
    end = max(start + 3, int(0.6 * len(xs)))
    index = start + int(np.argmin(slope[start:end]))
    out: dict[str, float] = {}
    if unit == "theta":
        out["theta_c_deg"] = float(xs[index])
        if energy_keV is not None:
            out["q_c"] = float(qz(xs[index], energy_keV))
    else:
        out["q_c"] = float(xs[index])
        if energy_keV is not None:
            out["theta_c_deg"] = float(theta_from_qz(xs[index], energy_keV))
    if "q_c" in out:
        out["electron_density_per_A3"] = out["q_c"] ** 2 / (
            16.0 * math.pi * CLASSICAL_ELECTRON_RADIUS_A
        )
    return out


def kiessig_thickness(
    q: Any, intensity: Any, q_range: tuple[float | None, float | None] | None = None
) -> dict[str, float]:
    """Film thickness from the spacing of the Kiessig fringes, ``2 pi / delta q``.

    The fringes are flattened (``log10(I q^4)`` minus a smooth trend),
    resampled evenly in q and Fourier transformed; the strongest frequency
    gives the thickness in Angstrom. ``q_range`` limits the fringes used.
    """
    from scipy.ndimage import gaussian_filter1d

    qs = np.asarray(to_host(q), dtype=float)
    ys = np.asarray(to_host(intensity), dtype=float)
    keep = np.isfinite(qs) & np.isfinite(ys) & (ys > 0) & (qs > 0)
    if q_range is not None:
        lo, hi = q_range
        keep &= (qs >= (lo if lo is not None else -np.inf)) & (
            qs <= (hi if hi is not None else np.inf)
        )
    order = np.argsort(qs[keep])
    qs, ys = qs[keep][order], ys[keep][order]
    if qs.size < 16:
        raise ValueError(f"only {qs.size} points in the fringe range; need at least 16")
    grid = np.linspace(qs[0], qs[-1], 4 * qs.size)
    signal = np.interp(grid, qs, np.log10(ys * qs**4))
    trend = gaussian_filter1d(signal, sigma=len(grid) / 8.0, mode="nearest")
    wiggle = (signal - trend) * np.hanning(len(grid))
    spectrum = np.abs(np.fft.rfft(wiggle, n=8 * len(grid)))
    thickness = np.fft.rfftfreq(8 * len(grid), d=grid[1] - grid[0]) * 2.0 * math.pi
    usable = thickness > 2.0 * math.pi / (qs[-1] - qs[0])  # at least one fringe in range
    best = int(np.argmax(np.where(usable, spectrum, 0.0)))
    return {
        "thickness_A": float(thickness[best]),
        "fringe_spacing_q": float(2.0 * math.pi / thickness[best]),
        "strength": float(spectrum[best] / (spectrum[usable].mean() or 1.0)),
    }
