"""Detector geometry, diffractometer conventions and coordinate transforms."""

from bexa.geometry.conventions import (
    CONVENTIONS,
    Convention,
    rotation_axis,
    rotation_x,
    rotation_y,
    rotation_z,
)
from bexa.geometry.detector import DetectorGeometry
from bexa.geometry.transforms import (
    angles_to_Q,
    motor_angles_to_Q,
    pixels_to_angles,
    q_magnitude,
    scattering_vector,
)

__all__ = [
    "CONVENTIONS",
    "Convention",
    "DetectorGeometry",
    "angles_to_Q",
    "motor_angles_to_Q",
    "pixels_to_angles",
    "q_magnitude",
    "rotation_axis",
    "rotation_x",
    "rotation_y",
    "rotation_z",
    "scattering_vector",
]
