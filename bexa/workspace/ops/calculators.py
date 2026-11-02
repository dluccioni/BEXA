"""Calculators: tables from numbers alone (energy, reflections, crystallite size, transmission)."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from bexa.workspace.objects import DataObject

__all__ = ["bragg", "energy", "photons", "reflections", "scherrer_size", "transmission"]

CENTERINGS = ("P", "I", "F", "A", "B", "C", "R")


def _table(name: str, rows: list[dict[str, object]] | pd.DataFrame, **meta: object) -> DataObject:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    return DataObject(name, frame, kind="table", meta=dict(meta))


def energy(energy_keV: float | None = 17.0, wavelength_A: float | None = None) -> DataObject:
    """Energy, wavelength, wavevector and the monochromator angles of the usual crystals."""
    from bexa.optics.conversions import summary

    found = summary(energy_keV=None if wavelength_A else energy_keV, wavelength_A=wavelength_A)
    return _table("energy", [{"quantity": k, "value": v} for k, v in found.items()])


def reflections(
    a: float = 5.43102,
    b: float | None = None,
    c: float | None = None,
    alpha: float = 90.0,
    beta: float = 90.0,
    gamma: float = 90.0,
    centering: str = "F",
    energy_keV: float = 17.0,
    max_index: int = 4,
    max_two_theta: float = 90.0,
) -> DataObject:
    """The reflections of a lattice: d-spacing, 2theta and q, one row per family (h >= k >= l
    sorted, the multiplicity counted), up to ``max_two_theta``."""
    from bexa.core.units import energy_to_wavelength
    from bexa.crystal.lattice import LatticeParameters, centering_allowed, d_spacing

    lattice = LatticeParameters(a, b or a, c or a, alpha, beta, gamma)
    wavelength = float(energy_to_wavelength(energy_keV))
    # one family per d-spacing: the largest (h, k, l) names it, the others count
    families: dict[float, tuple[tuple[int, int, int], int, float, float]] = {}
    span = range(-int(max_index), int(max_index) + 1)
    for hkl in itertools.product(span, span, span):
        if hkl == (0, 0, 0) or not centering_allowed(hkl, centering):
            continue
        d = d_spacing(lattice, hkl)
        if wavelength / (2 * d) > 1:
            continue
        two_theta = 2 * math.degrees(math.asin(wavelength / (2 * d)))
        if two_theta > max_two_theta:
            continue
        key = round(d, 6)
        name, count, _, _ = families.get(key, (hkl, 0, d, two_theta))
        families[key] = (max(name, hkl), count + 1, d, two_theta)
    rows = [
        {
            "hkl": " ".join(str(i) for i in name),
            "d_A": d,
            "two_theta_deg": two_theta,
            "q_1_A": 2 * math.pi / d,
            "multiplicity": count,
        }
        for name, count, d, two_theta in sorted(families.values(), key=lambda f: -f[2])
    ]
    return _table(
        "reflections", rows, lattice=lattice.as_dict(), centering=centering, energy_keV=energy_keV
    )


def bragg(d_spacings: Sequence[float] = (3.1356,), energy_keV: float = 17.0) -> DataObject:
    """The Bragg angle and 2theta of d-spacings at an energy."""
    from bexa.core.units import energy_to_wavelength

    wavelength = float(energy_to_wavelength(energy_keV))
    rows = []
    for d in d_spacings:
        ratio = wavelength / (2 * float(d))
        theta = math.degrees(math.asin(ratio)) if ratio <= 1 else float("nan")
        rows.append(
            {
                "d_A": float(d),
                "theta_deg": theta,
                "two_theta_deg": 2 * theta,
                "q_1_A": 2 * math.pi / float(d),
            }
        )
    return _table("bragg", rows, energy_keV=energy_keV)


def scherrer_size(
    fwhm_deg: Sequence[float] = (0.1,),
    two_theta_deg: Sequence[float] = (30.0,),
    energy_keV: float = 17.0,
    shape_factor: float = 0.9,
    instrumental_deg: float = 0.0,
) -> DataObject:
    """Crystallite sizes from peak widths (Scherrer), one row per peak."""
    from bexa.analysis.xrd import scherrer

    widths = np.asarray(fwhm_deg, dtype=float)
    angles = np.broadcast_to(np.asarray(two_theta_deg, dtype=float), widths.shape)
    sizes = scherrer(widths, angles, energy_keV, shape_factor, instrumental_deg)
    rows = [
        {"fwhm_deg": w, "two_theta_deg": t, "size_nm": s}
        for w, t, s in zip(widths, angles, sizes, strict=True)
    ]
    return _table("scherrer", rows, energy_keV=energy_keV)


def transmission(
    material: str = "Si",
    thickness: str = "500um",
    energies_keV: Sequence[float] = (8.0, 12.0, 17.0, 25.0),
) -> DataObject:
    """Transmission and attenuation length of a material at several energies (needs xraydb)."""
    from bexa.optics import attenuation

    rows = []
    for e in energies_keV:
        rows.append(
            {
                "energy_keV": float(e),
                "transmission": float(attenuation.transmission(material, thickness, e)),
                "attenuation_length_um": float(attenuation.attenuation_length_um(material, e)),
            }
        )
    return _table(f"transmission_{material}", rows, material=material, thickness=thickness)


def photons(pulse_energy_uJ: float = 100.0, energy_keV: float = 9.5) -> DataObject:
    """Photons per pulse from a pulse energy (an XFEL gas monitor reading)."""
    from bexa.core.units import photons_from_pulse_energy

    count = float(photons_from_pulse_energy(pulse_energy_uJ * 1e-6, energy_keV * 1e3))
    return _table(
        "photons",
        [{"pulse_energy_uJ": pulse_energy_uJ, "energy_keV": energy_keV, "photons": count}],
    )
