"""Pixels to angles to Q to hkl, with an explicit convention.

Ported from v9's ``pixels_to_angles``, ``angles_to_Q`` and ``motor_angles_to_Q``
and generalised: the beam axis and motor rotation axes come from a
:class:`bexa.geometry.conventions.Convention` instead of being hard-coded.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bexa.geometry.conventions import Convention, get_convention
from bexa.geometry.detector import DetectorGeometry


def pixels_to_angles(
    pixel_x: Any, pixel_y: Any, geometry: DetectorGeometry
) -> tuple[np.ndarray, np.ndarray]:
    """Scattering angles ``(two_theta, delta)`` in radians of detector pixels.

    ``two_theta`` grows along ``x`` away from the beam centre, ``delta`` along ``y``;
    the detector arm offsets of the geometry are added.
    """
    if geometry.distance_mm is None:
        raise ValueError("distance_mm is needed to convert pixels to angles")
    cy, cx = geometry.center()
    dx = (np.asarray(pixel_x, dtype=float) - cx) * geometry.pixel_size_mm
    dy = (np.asarray(pixel_y, dtype=float) - cy) * geometry.pixel_size_mm
    dist = geometry.distance_mm
    two_theta = np.arctan2(dx, dist) + np.radians(geometry.two_theta_offset_deg)
    delta = np.arctan2(dy, np.sqrt(dx**2 + dist**2)) + np.radians(geometry.delta_offset_deg)
    return two_theta, delta


def scattering_vector(
    two_theta_deg: Any, delta_deg: Any, k0: float, convention: str | Convention = "id03"
) -> np.ndarray:
    """Lab-frame ``Q = k_f - k_i`` (``..., 3``) for scattering angles in degrees."""
    conv = get_convention(convention)
    kf = conv.kf_direction(two_theta_deg, delta_deg)
    return k0 * (kf - np.asarray(conv.beam, dtype=float))


def q_magnitude(two_theta_deg: Any, k0: float) -> np.ndarray:
    """``|Q| = 2 k sin(theta)``."""
    return 2.0 * k0 * np.sin(np.radians(np.asarray(two_theta_deg, dtype=float)) / 2.0)


def angles_to_Q(
    two_theta: Any,
    delta: Any,
    geometry: DetectorGeometry,
    convention: str | Convention | None = None,
    **motor_angles_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample-frame ``(Qx, Qy, Qz)`` in inverse Angstrom for scattering angles in radians.

    ``Q_sample = R^T Q_lab`` where ``R`` is the sample rotation for the motor
    angles (``mu=..., chi=..., phi=...`` in degrees).
    """
    conv = get_convention(convention or geometry.convention)
    tt = np.degrees(np.asarray(two_theta, dtype=float))
    dl = np.degrees(np.asarray(delta, dtype=float))
    Q_lab = scattering_vector(tt, dl, geometry.k0, conv)
    R = conv.sample_rotation(**motor_angles_deg)
    Q = Q_lab @ R  # (R^T Q)^T for every row
    return Q[..., 0], Q[..., 1], Q[..., 2]


def motor_angles_to_Q(
    two_theta_deg: float,
    geometry: DetectorGeometry,
    convention: str | Convention | None = None,
    **motor_angles_deg: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample-frame Q of a fixed Bragg condition for arrays of motor angles.

    Used for 3-D reciprocal-lattice-point plots: every motor position of a
    mosaicity scan probes Q at a slightly different orientation.
    """
    conv = get_convention(convention or geometry.convention)
    names = list(motor_angles_deg)
    arrays = np.broadcast_arrays(*[np.asarray(v, dtype=float) for v in motor_angles_deg.values()])
    Q_lab = scattering_vector(two_theta_deg, 0.0, geometry.k0, conv)
    flat = [a.ravel() for a in arrays]
    out = np.empty((flat[0].size, 3))
    for i in range(flat[0].size):
        R = conv.sample_rotation(**{n: float(a[i]) for n, a in zip(names, flat, strict=True)})
        out[i] = R.T @ Q_lab
    shape = arrays[0].shape
    return out[:, 0].reshape(shape), out[:, 1].reshape(shape), out[:, 2].reshape(shape)


def build_coordinate_grids(
    structure: Any,
    geometry: DetectorGeometry,
    crystal: Any = None,
    space: str = "angular",
    hkl_center: tuple[int, int, int] | None = None,
    **fixed_angles_deg: float,
) -> dict[str, Any]:
    """Coordinates of every point of a scan in angular, Q or hkl space.

    Parameters
    ----------
    structure
        Scan structure (motor dims and coordinates).
    space
        ``"angular"`` (motor values), ``"Q"`` (needs a Bragg angle: ``two_theta`` in
        ``fixed_angles_deg`` or the crystal and ``hkl_center``) or ``"hkl"`` (needs
        ``crystal``).
    """
    dims = structure.motor_dims
    grids = np.meshgrid(*[structure.coords[d] for d in dims], indexing="ij")
    if space == "angular":
        return {d: g for d, g in zip(dims, grids, strict=True)}
    two_theta = fixed_angles_deg.pop("two_theta", None)
    if two_theta is None:
        if crystal is None or hkl_center is None:
            raise ValueError("Q or hkl coordinates need two_theta or a crystal and hkl_center")
        from bexa.core.units import bragg_angle_deg

        if geometry.energy_keV is None:
            raise ValueError("the geometry needs energy_keV to compute the Bragg angle")
        two_theta = 2.0 * bragg_angle_deg(crystal.d_spacing(hkl_center), geometry.energy_keV)
    angles: dict[str, Any] = {
        d: g
        for d, g in zip(dims, grids, strict=True)
        if d in get_convention(geometry.convention).motor_axes
    }
    angles.update({k: np.full(grids[0].shape, v) for k, v in fixed_angles_deg.items()})
    Qx, Qy, Qz = motor_angles_to_Q(two_theta, geometry, **angles)
    if space == "Q":
        return {"Qx": Qx, "Qy": Qy, "Qz": Qz}
    if space == "hkl":
        if crystal is None:
            raise ValueError("hkl coordinates need a crystal")
        h, k, l = crystal.hkl_from_Q(np.stack([Qx, Qy, Qz], axis=-1))
        if hkl_center is not None:
            centre = crystal.Q_vector(hkl_center)
            hc, kc, lc = crystal.hkl_from_Q(centre)
            h, k, l = h - hc + hkl_center[0], k - kc + hkl_center[1], l - lc + hkl_center[2]
        return {"h": h, "k": k, "l": l}
    raise ValueError(f"unknown space {space!r}; use angular, Q or hkl")
