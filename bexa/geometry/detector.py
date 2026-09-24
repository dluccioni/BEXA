"""Detector and beam geometry for coordinate transforms, scalebars and pixel sizes."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from bexa.core.units import energy_to_wavelength, wavevector


@dataclass
class DetectorGeometry:
    """Where the detector is and what the beam is.

    Attributes
    ----------
    energy_keV
        Photon energy.
    pixel_size_um
        Physical pixel pitch of the detector.
    distance_mm
        Sample-to-detector distance.
    beam_center
        ``(y, x)`` pixel hit by the direct beam.
    two_theta_offset_deg, delta_offset_deg
        Detector arm angles added to the pixel angles.
    effective_pixel_nm
        Pixel size projected onto the sample (after the objective) for scalebars.
    detector_shape
        ``(height, width)`` in pixels.
    convention
        Name of the diffractometer convention (see :mod:`bexa.geometry.conventions`).
    """

    energy_keV: float | None = None
    pixel_size_um: float | None = None
    distance_mm: float | None = None
    beam_center: tuple[float, float] | None = None
    two_theta_offset_deg: float = 0.0
    delta_offset_deg: float = 0.0
    effective_pixel_nm: float | None = None
    detector_shape: tuple[int, int] | None = None
    convention: str = "id03"
    detector: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def wavelength_A(self) -> float:
        if self.energy_keV is None:
            raise ValueError("energy_keV is not set")
        return float(energy_to_wavelength(self.energy_keV))

    @property
    def k0(self) -> float:
        """Incident wavevector magnitude ``2 pi / lambda`` in inverse Angstrom."""
        if self.energy_keV is None:
            raise ValueError("energy_keV is not set")
        return wavevector(self.energy_keV)

    @property
    def pixel_size_mm(self) -> float:
        if self.pixel_size_um is None:
            raise ValueError("pixel_size_um is not set")
        return self.pixel_size_um / 1000.0

    def center(self) -> tuple[float, float]:
        """Beam centre, defaulting to the middle of the detector."""
        if self.beam_center is not None:
            return self.beam_center
        if self.detector_shape is not None:
            return self.detector_shape[0] / 2.0, self.detector_shape[1] / 2.0
        raise ValueError("neither beam_center nor detector_shape is set")

    def with_energy(self, energy_keV: float) -> DetectorGeometry:
        return replace(self, energy_keV=energy_keV)

    @classmethod
    def from_spec(
        cls,
        spec: Any,
        detector: str | None = None,
        energy_keV: float | None = None,
        profile: Any = None,
    ) -> DetectorGeometry:
        """Combine the spec's detector table with a profile's geometry section."""
        info = spec.detector_info(detector) if (spec is not None and detector) else {}
        geom = cls(
            energy_keV=energy_keV,
            pixel_size_um=info.get("pixel_size_um"),
            effective_pixel_nm=info.get("effective_pixel_nm"),
            detector_shape=tuple(info["shape"]) if info.get("shape") else None,
            convention=(spec.convention if spec is not None and spec.convention else "id03"),
            detector=detector,
        )
        if profile is not None:
            values = profile.model_dump() if hasattr(profile, "model_dump") else dict(profile)
            for key in (
                "energy_keV",
                "pixel_size_um",
                "distance_mm",
                "effective_pixel_nm",
                "convention",
            ):
                if values.get(key) is not None:
                    setattr(geom, key, values[key])
            if values.get("beam_center") is not None:
                geom.beam_center = tuple(values["beam_center"])
            geom.two_theta_offset_deg = float(values.get("two_theta_offset_deg", 0.0) or 0.0)
            geom.delta_offset_deg = float(values.get("delta_offset_deg", 0.0) or 0.0)
        return geom

    def __str__(self) -> str:
        parts = []
        if self.energy_keV is not None:
            parts.append(f"{self.energy_keV:.4f} keV")
        if self.pixel_size_um is not None:
            parts.append(f"{self.pixel_size_um:g} um pixels")
        if self.distance_mm is not None:
            parts.append(f"{self.distance_mm:g} mm")
        if self.effective_pixel_nm is not None:
            parts.append(f"{self.effective_pixel_nm:g} nm effective pixel")
        parts.append(f"convention {self.convention}")
        return ", ".join(parts)
