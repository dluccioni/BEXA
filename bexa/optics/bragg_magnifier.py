"""Asymmetric Bragg magnifier: rocking curves, magnification and pair efficiency.

Port of ``Bragg_Mag_Efficiency_Calcs.py``. Dynamical two-beam reflectivities
of silicon in the Bragg case (``xraydb`` form factors), the magnification of a
miscut crystal, and the throughput of two orthogonal S1/S2 pairs behind a
symmetric S0 monochromator, integrated over the beam divergence and energy
spread. The rocking-curve kernel is plain numpy and vectorised over the angle
grid; ``numba`` accelerates the efficiency grid when it is installed.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import numpy as np

URAD_TO_ARCSEC = 1e-6 * 180.0 * 3600.0 / np.pi
ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)
DEFAULT_E0 = 9251.0
DEFAULT_DE = 0.5
DEFAULT_SIGMA_THETA_URAD = 1.5
DEFAULT_SIGMA_X_UM = 35.0
DEFAULT_HKL = (2, 2, 0)
R_E = 2.8179403e-5  # classical electron radius, Angstrom
A_SI = 5.43102
V_SI = A_SI**3
B_DW = 0.4632  # Debye-Waller B of silicon, A^2

__all__ = [
    "crystal_params",
    "default_angle_grid",
    "efficiency",
    "energy_grid",
    "gaussian_profile",
    "mag_from_miscut",
    "mag_split",
    "miscut_from_mag",
    "miscuts_from_split",
    "optimal_split",
    "reflectivity",
    "rocking_curve",
    "rocking_curve_absolute",
    "si_structure_factor",
    "track_phase_space",
]


def default_angle_grid() -> np.ndarray:
    """Rocking angles in arcsec: -100 to 100 in 8001 steps, as in the script."""
    return np.linspace(-100, 100, 8001)


def si_structure_factor(h: int, k: int, l: int) -> complex:
    """Geometric structure factor of the diamond lattice (fcc times the two-atom basis)."""
    fcc = (
        1
        + np.exp(1j * np.pi * (h + k))
        + np.exp(1j * np.pi * (h + l))
        + np.exp(1j * np.pi * (k + l))
    )
    basis = 1 + np.exp(1j * np.pi * (h + k + l) / 2)
    return complex(fcc * basis)


@functools.lru_cache(maxsize=64)
def crystal_params(E_eV: float, hkl: tuple[int, int, int] = DEFAULT_HKL) -> dict[str, Any]:
    """Wavelength, spacing, Bragg angle and susceptibilities chi_0, chi_h of Si ``hkl``."""
    import xraydb

    h, k, l = hkl
    lam = 12398.42 / E_eV
    d = A_SI / np.sqrt(h**2 + k**2 + l**2)
    theta_B = float(np.arcsin(lam / (2 * d)))
    gamma = R_E * lam**2 / (np.pi * V_SI)
    q = np.sin(theta_B) / lam
    f0 = float(xraydb.f0("Si", q)[0])
    fp = float(xraydb.f1_chantler("Si", E_eV))
    fpp = float(xraydb.f2_chantler("Si", E_eV))
    debye_waller = np.exp(-B_DW * q**2)
    chi_0 = -gamma * 8 * (14.0 + fp + 1j * fpp)
    chi_h = -gamma * si_structure_factor(h, k, l) * (f0 * debye_waller + fp + 1j * fpp)
    return {"lam": lam, "d": d, "theta_B": theta_B, "chi_0": chi_0, "chi_h": chi_h, "Gamma": gamma}


def _polarisation(pol: str, theta_B: float) -> float:
    return 1.0 if pol == "sigma" else float(np.cos(2 * theta_B))


def reflectivity(
    alpha: float,
    dth: np.ndarray,
    theta_B: float,
    chi_0: complex,
    chi_h: complex,
    lam: float,
    d: float,
    C: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Bragg-case reflectivity against the incidence and exit angle offsets (arcsec).

    ``alpha`` is the miscut (rad). Returns ``(R_incidence, R_exit)`` where the
    second is the same curve resampled onto the exit-angle offsets.
    """
    k = 2.0 * np.pi / lam
    H = 2.0 * np.pi / d
    theta_inc = theta_B - alpha + dth * ARCSEC_TO_RAD
    b = -np.sin(theta_B - alpha) / np.sin(theta_B + alpha)
    k0H = np.sin(theta_inc + alpha) * H * k
    alpha_p = (H**2 / 2.0 - k0H) / k**2 + chi_0 / 2.0 * (1.0 / b - 1.0)
    delta = np.sqrt(alpha_p**2 + C**2 * chi_h**2 / b)
    ra = C * chi_h / (alpha_p + delta)
    rb = C * chi_h / (alpha_p - delta)
    R = np.minimum(np.abs(ra) ** 2, np.abs(rb) ** 2) / abs(b)
    cos_exit = np.cos(theta_inc) - 2.0 * np.sin(theta_B) * np.sin(alpha)
    dth_exit = (np.arccos(np.clip(cos_exit, -1.0, 1.0)) - (theta_B + alpha)) / ARCSEC_TO_RAD
    return R, np.interp(dth, dth_exit, R)


def rocking_curve(
    alpha_deg: float,
    dtheta_arcsec: Any,
    E_eV: float = DEFAULT_E0,
    hkl: tuple[int, int, int] = DEFAULT_HKL,
    pol: str = "sigma",
) -> dict[str, Any]:
    """Rocking curve of a crystal with miscut ``alpha_deg``; widths and peak included."""
    p = crystal_params(E_eV, hkl)
    alpha = np.radians(alpha_deg)
    dth = np.asarray(dtheta_arcsec, dtype=float)
    C = _polarisation(pol, p["theta_B"])
    R, R_exit = reflectivity(alpha, dth, p["theta_B"], p["chi_0"], p["chi_h"], p["lam"], p["d"], C)
    theta_inc = p["theta_B"] - alpha + dth * ARCSEC_TO_RAD
    cos_exit = np.cos(theta_inc) - 2 * np.sin(p["theta_B"]) * np.sin(alpha)
    theta_exit = np.arccos(np.clip(cos_exit, -1, 1))
    dth_exit = (theta_exit - (p["theta_B"] + alpha)) / ARCSEC_TO_RAD
    above = R.max() / 2 < R
    fwhm_inc = float(dth[above].max() - dth[above].min()) if above.any() else 0.0
    fwhm_exit = float(dth_exit[above].max() - dth_exit[above].min()) if above.any() else 0.0
    return {
        "R_inc": R,
        "R_exit": R_exit,
        "theta_inc_deg": np.degrees(theta_inc),
        "theta_exit_deg": np.degrees(theta_exit),
        "dtheta": dth,
        "dtheta_exit": dth_exit,
        "b": float(-np.sin(p["theta_B"] - alpha) / np.sin(p["theta_B"] + alpha)),
        "FWHM_inc": fwhm_inc,
        "FWHM_exit": fwhm_exit,
        "R_peak": float(R.max()),
    }


def rocking_curve_absolute(
    alpha_deg: float,
    angles_deg: Any,
    E_eV: float = DEFAULT_E0,
    hkl: Any = DEFAULT_HKL,
    pol: str = "sigma",
) -> dict[str, Any]:
    """:func:`rocking_curve` on absolute incidence angles (deg) instead of offsets."""
    p = crystal_params(E_eV, tuple(hkl))
    theta_inc_B = np.degrees(p["theta_B"]) - alpha_deg
    return rocking_curve(
        alpha_deg, (np.asarray(angles_deg) - theta_inc_B) * 3600, E_eV, tuple(hkl), pol
    )


def mag_from_miscut(miscut_rad: Any, E_eV: float = DEFAULT_E0, hkl: Any = DEFAULT_HKL) -> Any:
    """Magnification ``sin(theta_B + alpha) / sin(theta_B - alpha)``."""
    theta_B = crystal_params(E_eV, tuple(hkl))["theta_B"]
    return np.sin(theta_B + miscut_rad) / np.sin(theta_B - miscut_rad)


def miscut_from_mag(mag: Any, E_eV: float = DEFAULT_E0, hkl: Any = DEFAULT_HKL) -> Any:
    """Miscut (rad) that gives magnification ``mag``."""
    theta_B = crystal_params(E_eV, tuple(hkl))["theta_B"]
    return np.arctan(
        np.tan(theta_B) * (np.asarray(mag, dtype=float) - 1) / (np.asarray(mag, dtype=float) + 1)
    )


def mag_split(total_mag: float, split_ratio: float) -> tuple[float, float]:
    """Magnifications of the two crystals: ``M^s`` and ``M^(1-s)``."""
    mag1 = total_mag**split_ratio
    return float(mag1), float(total_mag / mag1)


def miscuts_from_split(
    total_mag: float, split_ratio: float, E_eV: float = DEFAULT_E0, hkl: Any = DEFAULT_HKL
) -> tuple[float, float]:
    """Miscuts (deg) of the two crystals of a pair."""
    m1, m2 = mag_split(total_mag, split_ratio)
    return float(np.degrees(miscut_from_mag(m1, E_eV, hkl))), float(
        np.degrees(miscut_from_mag(m2, E_eV, hkl))
    )


def energy_grid(
    E0: float, dE_fwhm: float, n_E: int = 21, spectrum: str | Callable[[float], float] = "gaussian"
) -> tuple[np.ndarray, np.ndarray]:
    """Energies and normalised trapezoid weights of a Gaussian, flat or custom spectrum."""
    sigma = dE_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    if spectrum == "gaussian":
        E = np.linspace(E0 - 3 * sigma, E0 + 3 * sigma, n_E)
        w = np.exp(-0.5 * ((E - E0) / sigma) ** 2)
    elif spectrum == "flat":
        E = np.linspace(E0 - dE_fwhm / 2, E0 + dE_fwhm / 2, n_E)
        w = np.ones(n_E)
    elif callable(spectrum):
        E = np.linspace(E0 - 3 * sigma, E0 + 3 * sigma, n_E)
        w = np.array([spectrum(e) for e in E])
    else:
        raise ValueError(f"unknown spectrum {spectrum!r}")
    step = np.diff(E)
    trap = np.zeros(n_E)
    trap[0], trap[-1] = step[0] / 2, step[-1] / 2
    trap[1:-1] = (step[:-1] + step[1:]) / 2
    weights = w * trap
    return E, weights / weights.sum()


def gaussian_profile(dth: np.ndarray, sigma_arcsec: float, center: float = 0.0) -> np.ndarray:
    """Unit-area Gaussian on the angle grid (the beam divergence)."""
    G = np.exp(-0.5 * ((dth - center) / sigma_arcsec) ** 2)
    area = np.trapezoid(G, dth)
    return G / area if area > 0 else G


def _params_vs_energy(E: np.ndarray, hkl: tuple[int, int, int], pol: str) -> dict[str, np.ndarray]:
    rows = [crystal_params(float(e), hkl) for e in E]
    return {
        "theta_B": np.array([r["theta_B"] for r in rows]),
        "chi_0": np.array([r["chi_0"] for r in rows]),
        "chi_h": np.array([r["chi_h"] for r in rows]),
        "lam": np.array([r["lam"] for r in rows]),
        "d": np.array([r["d"] for r in rows]),
        "C": np.array([_polarisation(pol, r["theta_B"]) for r in rows]),
    }


def _pair_alignment(
    a1: float,
    a2: float,
    dth: np.ndarray,
    p: dict[str, np.ndarray],
    ie: int,
    S0_exit: np.ndarray,
    G: np.ndarray,
    mono: bool,
) -> float:
    """Angle offset (arcsec) of the magnifier that maximises the pair throughput."""
    args = (p["theta_B"][ie], p["chi_0"][ie], p["chi_h"][ie], p["lam"][ie], p["d"][ie], p["C"][ie])
    S1_R, S1_Re = reflectivity(a1, dth, *args)
    S2_R, _ = reflectivity(a2, dth, *args)
    lo = min(dth[np.argmax(S1_R)], dth[np.argmax(S2_R)]) - 2.0
    hi = max(dth[np.argmax(S1_R)], dth[np.argmax(S2_R)]) + 2.0
    best_dphi, best_overlap = 0.0, -1.0
    for dphi in np.linspace(lo, hi, 31):
        shifted = dth + dphi
        pair = (
            np.interp(shifted, dth, S1_R)
            * np.interp(shifted, dth, S1_Re)
            * np.interp(shifted, dth, S2_R)
        )
        if mono:
            pair = S0_exit * pair
        overlap = float(np.sum(pair * pair * G))
        if overlap > best_overlap:
            best_overlap, best_dphi = overlap, float(dphi)
    return best_dphi


def _efficiency_grid(
    total_mags: np.ndarray,
    splits: np.ndarray,
    dth: np.ndarray,
    p: dict[str, np.ndarray],
    E_weights: np.ndarray,
    S0_exit: np.ndarray,
    input_area: np.ndarray,
    G: np.ndarray,
    mono: bool,
    align: bool,
) -> np.ndarray:
    """Efficiency (%) of two orthogonal S1/S2 pairs for every (magnification, split)."""
    n_E = len(E_weights)
    ie_c = n_E // 2
    theta_B0 = p["theta_B"][ie_c]
    out = np.zeros((len(total_mags), len(splits)))
    for j, total in enumerate(total_mags):
        for i, split in enumerate(splits):
            m1, m2 = mag_split(float(total), float(split))
            a1 = float(np.arctan(np.tan(theta_B0) * (m1 - 1.0) / (m1 + 1.0)))
            a2 = float(np.arctan(np.tan(theta_B0) * (m2 - 1.0) / (m2 + 1.0)))
            dphi = _pair_alignment(a1, a2, dth, p, ie_c, S0_exit[ie_c], G, mono) if align else 0.0
            shifted = dth + dphi
            eta_sum = norm_sum = 0.0
            for ie in range(n_E):
                args = (
                    p["theta_B"][ie],
                    p["chi_0"][ie],
                    p["chi_h"][ie],
                    p["lam"][ie],
                    p["d"][ie],
                    p["C"][ie],
                )
                S1_R, S1_Re = reflectivity(a1, dth, *args)
                S2_R, _ = reflectivity(a2, dth, *args)
                pair = (
                    np.interp(shifted, dth, S1_R)
                    * np.interp(shifted, dth, S1_Re)
                    * np.interp(shifted, dth, S2_R)
                )
                if mono:
                    pair = S0_exit[ie] * pair
                eta_sum += E_weights[ie] * np.trapezoid(pair * pair * G, dth)
                norm_sum += E_weights[ie] * input_area[ie]
            if norm_sum > 0:
                out[j, i] = eta_sum / norm_sum * 100.0
    return out


def efficiency(
    total_mags: Any,
    splits: Any,
    E0: float = DEFAULT_E0,
    dE: float = DEFAULT_DE,
    n_E: int = 21,
    spectrum: str | Callable[[float], float] = "gaussian",
    sigma_theta_urad: float = DEFAULT_SIGMA_THETA_URAD,
    hkl: Any = DEFAULT_HKL,
    pol: str = "sigma",
    dth: np.ndarray | None = None,
    mono: bool = True,
    align: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Throughput (%) of the magnifier for grids of total magnification and split ratio.

    Port of ``compute_efficiency_v2``: the S0 monochromator (symmetric Si
    ``hkl``) is aligned to its peak, each S1/S2 pair to its best overlap, and
    the product of the four reflectivities is weighted by the beam divergence
    (Gaussian, ``sigma_theta_urad``) and the energy spread (``dE`` eV FWHM).
    Returns the grid ``(len(total_mags), len(splits))`` and the intermediate arrays.
    """
    dth = default_angle_grid() if dth is None else np.asarray(dth, dtype=float)
    total_mags = np.asarray(total_mags, dtype=float).ravel()
    splits = np.asarray(splits, dtype=float).ravel()
    hkl = tuple(int(v) for v in hkl)
    E, E_weights = energy_grid(E0, dE, n_E, spectrum)
    p = _params_vs_energy(E, hkl, pol)
    G = gaussian_profile(dth, sigma_theta_urad * URAD_TO_ARCSEC)
    ie_c = n_E // 2

    def s0(ie: int) -> np.ndarray:
        return reflectivity(
            0.0,
            dth,
            p["theta_B"][ie],
            p["chi_0"][ie],
            p["chi_h"][ie],
            p["lam"][ie],
            p["d"][ie],
            p["C"][ie],
        )[1]

    dphi_mono = float(dth[np.argmax(s0(ie_c))]) if align else 0.0
    S0_exit = np.empty((n_E, len(dth)))
    input_area = np.empty(n_E)
    for ie in range(n_E):
        S0_exit[ie] = np.interp(dth + dphi_mono, dth, s0(ie)) if align else s0(ie)
        input_area[ie] = np.trapezoid(S0_exit[ie] ** 2 * G, dth) if mono else np.trapezoid(G, dth)
    grid = _efficiency_grid(
        total_mags, splits, dth, p, E_weights, S0_exit, input_area, G, mono, align
    )
    info = {
        "E_arr": E,
        "E_weights": E_weights,
        "theta_B_arr": p["theta_B"],
        "S0_R_exit_2d": S0_exit,
        "input_area_arr": input_area,
        "G_div": G,
        "sigma_arcsec": sigma_theta_urad * URAD_TO_ARCSEC,
        "dphi_mono": dphi_mono,
        "mono": mono,
        "align": align,
    }
    return grid, info


def optimal_split(
    total_mag: float, n_splits: int = 50, **kwargs: Any
) -> tuple[float, float, float]:
    """``(split, efficiency %, dphi arcsec)`` of the best split ratio for one magnification."""
    splits = np.linspace(0.0, 1.0, n_splits)
    grid, info = efficiency(np.array([total_mag]), splits, **kwargs)
    best = int(np.argmax(grid[0]))
    best_split = float(splits[best])
    if not kwargs.get("align", True):
        return best_split, float(grid[0, best]), 0.0
    E0 = kwargs.get("E0", DEFAULT_E0)
    hkl = tuple(kwargs.get("hkl", DEFAULT_HKL))
    pol = kwargs.get("pol", "sigma")
    dth = np.asarray(kwargs.get("dth", default_angle_grid()), dtype=float)
    p = _params_vs_energy(np.array([E0]), hkl, pol)
    m1, m2 = mag_split(total_mag, best_split)
    a1 = float(miscut_from_mag(m1, E0, hkl))
    a2 = float(miscut_from_mag(m2, E0, hkl))
    G = gaussian_profile(
        dth, kwargs.get("sigma_theta_urad", DEFAULT_SIGMA_THETA_URAD) * URAD_TO_ARCSEC
    )
    S0_exit = info["S0_R_exit_2d"][len(info["E_arr"]) // 2]
    dphi = _pair_alignment(a1, a2, dth, p, 0, S0_exit, G, True)
    return best_split, float(grid[0, best]), dphi


def track_phase_space(
    total_mag: float,
    split_ratio: float,
    sigma_x_um: float = DEFAULT_SIGMA_X_UM,
    sigma_theta_urad: float = DEFAULT_SIGMA_THETA_URAD,
    E_eV: float = DEFAULT_E0,
    hkl: Any = DEFAULT_HKL,
) -> list[tuple[str, float, float, float]]:
    """Beam size, divergence and their product after every crystal of the two pairs."""
    theta_B = crystal_params(E_eV, tuple(hkl))["theta_B"]
    m1, m2 = mag_split(total_mag, split_ratio)
    a1 = np.arctan(np.tan(theta_B) * (m1 - 1) / (m1 + 1))
    a2 = np.arctan(np.tan(theta_B) * (m2 - 1) / (m2 + 1))
    M1 = np.sin(theta_B + a1) / np.sin(theta_B - a1)
    M2 = np.sin(theta_B + a2) / np.sin(theta_B - a2)
    stages = []
    for pair, label in ((1, ""), (2, "pair 2 ")):
        sx, sth = sigma_x_um, sigma_theta_urad
        stages.append((f"{label}source", sx, sth, sx * sth))
        if pair == 1:
            stages.append(("after S0", sx, sth, sx * sth))
        sx, sth = sx * M1, sth / M1
        stages.append((f"{label}after S1", sx, sth, sx * sth))
        sx, sth = sx * M2, sth / M2
        stages.append((f"{label}after S2", sx, sth, sx * sth))
    return stages
