"""Named diffractometer conventions: beam axis, motor rotation axes and eta zero.

The legacy code used three different conventions (v9's ID03 transform with
the beam along z, the four-circle solvers with the beam along x, and SACLA's
azimuth definition). They live here as named objects so every transform takes
a ``Convention`` argument instead of hard-coding one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def rotation_axis(axis: Any, angle_deg: float) -> np.ndarray:
    """Rotation matrix about a unit ``axis`` by ``angle_deg`` (Rodrigues' formula)."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    t = np.radians(angle_deg)
    c, s = np.cos(t), np.sin(t)
    x, y, z = a
    return np.array(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ]
    )


def rotation_x(angle_deg: float) -> np.ndarray:
    return rotation_axis((1, 0, 0), angle_deg)


def rotation_y(angle_deg: float) -> np.ndarray:
    return rotation_axis((0, 1, 0), angle_deg)


def rotation_z(angle_deg: float) -> np.ndarray:
    return rotation_axis((0, 0, 1), angle_deg)


@dataclass(frozen=True)
class Convention:
    """Lab frame and motor stack of one instrument.

    Attributes
    ----------
    beam
        Unit vector of the incident beam.
    horizontal, vertical
        Unit vectors spanning the detector plane at zero angles; ``two_theta``
        rotates ``k_f`` from ``beam`` towards ``horizontal`` and ``delta`` (or
        the vertical pixel offset) towards ``vertical``.
    motor_axes
        Nominal rotation axis of each sample motor in the lab frame at zero.
    motor_order
        Motors from the bottom of the stack to the top.
    coupled
        ``True`` when each motor's axis moves with the motors below it (a real
        goniometer stack, as in the four-circle solvers); ``False`` for rotations
        about fixed lab axes (v9's ID03 transform).
    eta_mode
        ``"vertical"``: eta = atan2(kf_v, kf_h) - eta_zero (four-circle v1);
        ``"sacla"``: eta measured from -horizontal, anticlockwise looking downstream.
    """

    name: str
    beam: tuple[float, float, float]
    horizontal: tuple[float, float, float]
    vertical: tuple[float, float, float]
    motor_axes: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    motor_order: tuple[str, ...] = ()
    coupled: bool = True
    eta_mode: str = "vertical"
    eta_zero_offset: float = 0.0
    description: str = ""

    def sample_rotation(self, **angles_deg: float) -> np.ndarray:
        """Combined rotation matrix of the sample stack for the given motor angles."""
        R = np.eye(3)
        for motor in self.motor_order:
            angle = float(angles_deg.get(motor, 0.0))
            axis = np.asarray(self.motor_axes[motor], dtype=float)
            if self.coupled:
                axis = R @ axis
            R = rotation_axis(axis, angle) @ R
        return R

    def kf_direction(self, two_theta_deg: Any, delta_deg: Any = 0.0) -> np.ndarray:
        """Unit vector(s) of ``k_f`` for scattering angles (``..., 3``)."""
        tt = np.radians(np.asarray(two_theta_deg, dtype=float))
        dl = np.radians(np.asarray(delta_deg, dtype=float))
        b, h, v = (np.asarray(x, dtype=float) for x in (self.beam, self.horizontal, self.vertical))
        return (
            (np.cos(tt) * np.cos(dl))[..., None] * b
            + (np.sin(tt) * np.cos(dl))[..., None] * h
            + np.sin(dl)[..., None] * v
        )

    def detector_angles(self, Q_lab: np.ndarray, k: float) -> tuple[float, float]:
        """``(two_theta, eta)`` in degrees for a lab-frame ``Q`` and wavevector ``k``."""
        b = np.asarray(self.beam, dtype=float)
        kf = np.asarray(Q_lab, dtype=float) + k * b
        norm = np.linalg.norm(kf)
        if abs(norm - k) > 1e-6 * k:  # not elastic: project onto the Ewald sphere
            kf = kf / norm * k
        two_theta = float(np.degrees(np.arccos(np.clip(kf @ b / k, -1, 1))))
        kf_h = float(kf @ np.asarray(self.horizontal, dtype=float))
        kf_v = float(kf @ np.asarray(self.vertical, dtype=float))
        if abs(kf_h) < 1e-10 and abs(kf_v) < 1e-10:  # on axis: eta is undefined, use the zero
            eta = -self.eta_zero_offset
            if self.eta_mode == "sacla":
                eta = float(np.mod(eta, 360.0))
        else:
            eta = float(eta_of_kf(kf_h, kf_v, self))
        return two_theta, eta

    def with_eta_offset(self, eta_zero_offset: float) -> Convention:
        from dataclasses import replace

        return replace(self, eta_zero_offset=eta_zero_offset)


def eta_of_kf(kf_h: Any, kf_v: Any, convention: Convention) -> np.ndarray:
    """Azimuth eta (deg) from the horizontal and vertical components of ``k_f``, vectorised.

    ``"vertical"`` mode: ``atan2(kf_h, kf_v) - eta_zero``; ``"sacla"`` mode:
    ``atan2(kf_v, -kf_h) - eta_zero`` wrapped to ``[0, 360)``.
    """
    kf_h = np.asarray(kf_h, dtype=float)
    kf_v = np.asarray(kf_v, dtype=float)
    if convention.eta_mode == "sacla":
        return np.mod(np.degrees(np.arctan2(kf_v, -kf_h)) - convention.eta_zero_offset, 360.0)
    return np.degrees(np.arctan2(kf_h, kf_v)) - convention.eta_zero_offset


CONVENTIONS: dict[str, Convention] = {
    # v9's ID03 transform: beam along +z, mu about y, chi about z, phi about y,
    # applied as fixed-axis rotations R = R_mu R_chi R_phi.
    "id03": Convention(
        name="id03",
        beam=(0, 0, 1),
        horizontal=(1, 0, 0),
        vertical=(0, 1, 0),
        motor_axes={"mu": (0, 1, 0), "chi": (0, 0, 1), "phi": (0, 1, 0)},
        motor_order=("mu", "chi", "phi"),
        coupled=False,
        description="ESRF ID03 dark-field microscope (v9 transform)",
    ),
    # The four-circle solvers: beam along +x, stack mu(y) -> chi(x) -> phi(y) -> omega(z),
    # each axis carried by the motors below it; eta=0 is vertical diffraction.
    "pal_fourc": Convention(
        name="pal_fourc",
        beam=(1, 0, 0),
        horizontal=(0, 1, 0),
        vertical=(0, 0, 1),
        motor_axes={"mu": (0, 1, 0), "chi": (1, 0, 0), "phi": (0, 1, 0), "omega": (0, 0, 1)},
        motor_order=("mu", "chi", "phi", "omega"),
        coupled=True,
        eta_mode="vertical",
        description="Four-circle diffractometer as in Motor_Solver_v1 / four_circle_diffractometer",
    ),
    # Same stack, SACLA azimuth: eta from -y, anticlockwise looking downstream, in [0, 360).
    "sacla": Convention(
        name="sacla",
        beam=(1, 0, 0),
        horizontal=(0, 1, 0),
        vertical=(0, 0, 1),
        motor_axes={"mu": (0, 1, 0), "chi": (1, 0, 0), "phi": (0, 1, 0), "omega": (0, 0, 1)},
        motor_order=("mu", "chi", "phi", "omega"),
        coupled=True,
        eta_mode="sacla",
        description="Four-circle stack with the SACLA eta convention (Motor_Solver_SACLA)",
    ),
}
CONVENTIONS["lcls_xcs"] = CONVENTIONS["pal_fourc"]
CONVENTIONS["euxfel_mid"] = CONVENTIONS["id03"]


def get_convention(name: str | Convention | None) -> Convention:
    """Look up a convention by name (``Convention`` objects pass through)."""
    if isinstance(name, Convention):
        return name
    if name is None:
        return CONVENTIONS["id03"]
    try:
        return CONVENTIONS[name]
    except KeyError:
        raise KeyError(f"unknown convention {name!r}; known: {sorted(CONVENTIONS)}") from None
