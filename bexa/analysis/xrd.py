"""Powder diffraction on an area detector: geometry, azimuthal integration, cakes, Scherrer.

A flat detector at ``distance_mm`` from the sample, hit by the direct beam at
pixel ``(center_y, center_x)``, optionally tilted by ``tilt_deg`` about an
axis in its plane that makes ``tilt_plane_deg`` with the detector's x axis.
Every pixel gets a scattering angle 2theta and an azimuth chi (0 along the
detector's +x, counter-clockwise towards +y); integration bins pixels by
2theta (or q, or d) with ``numpy.bincount``, so a 4-megapixel image integrates
in tens of milliseconds once the geometry arrays are cached.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from bexa.core.backend import to_host
from bexa.core.units import energy_to_wavelength

__all__ = [
    "UNITS",
    "PowderGeometry",
    "calibrate_from_ring",
    "integrate_1d",
    "integrate_2d",
    "scherrer",
]

UNITS = ("2theta", "q", "d")
UNIT_LABELS = {"2theta": "2theta (deg)", "q": "q (1/A)", "d": "d (A)"}


@dataclass(frozen=True)
class PowderGeometry:
    """Where the detector sits: distance, beam centre, pixel size, energy and tilt."""

    distance_mm: float
    center_y: float
    center_x: float
    pixel_um: float
    energy_keV: float
    tilt_deg: float = 0.0
    tilt_plane_deg: float = 0.0

    @property
    def wavelength_A(self) -> float:
        return float(energy_to_wavelength(self.energy_keV))

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def arrays(self, shape: tuple[int, int]) -> dict[str, np.ndarray]:
        """``two_theta`` and ``chi`` (degrees) and ``solid_angle`` (1 at the centre) per pixel."""
        return dict(_geometry_arrays(self, (int(shape[0]), int(shape[1]))))

    def axis(self, shape: tuple[int, int], unit: str = "2theta") -> np.ndarray:
        """The value of ``unit`` at every pixel."""
        two_theta = self.arrays(shape)["two_theta"]
        return convert_two_theta(two_theta, unit, self.wavelength_A)

    def polarization(self, shape: tuple[int, int], factor: float = 0.0) -> np.ndarray:
        """The polarisation factor per pixel; ``factor`` 0 unpolarised, 1 fully horizontal."""
        arrays = self.arrays(shape)
        tth = np.radians(arrays["two_theta"])
        chi = np.radians(arrays["chi"])
        cos2, sin2 = np.cos(tth) ** 2, np.sin(tth) ** 2
        return 0.5 * (1.0 + cos2 - factor * np.cos(2.0 * chi) * sin2)


@lru_cache(maxsize=8)
def _geometry_arrays(geometry: PowderGeometry, shape: tuple[int, int]) -> tuple[Any, ...]:
    """The per-pixel arrays of a geometry (cached: the same detector integrates many frames)."""
    height, width = shape
    pixel_mm = geometry.pixel_um * 1e-3
    yy, xx = np.meshgrid(
        (np.arange(height) - geometry.center_y) * pixel_mm,
        (np.arange(width) - geometry.center_x) * pixel_mm,
        indexing="ij",
    )
    points = np.stack([xx, yy, np.zeros_like(xx)], axis=-1)  # in the detector plane
    rotation = _tilt_rotation(geometry.tilt_deg, geometry.tilt_plane_deg)
    lab = points @ rotation.T
    lab[..., 2] += geometry.distance_mm
    radius = np.linalg.norm(lab, axis=-1)
    two_theta = np.degrees(np.arccos(np.clip(lab[..., 2] / radius, -1.0, 1.0)))
    chi = np.degrees(np.arctan2(lab[..., 1], lab[..., 0]))
    normal = rotation @ np.array([0.0, 0.0, 1.0])
    cos_incidence = np.abs(lab @ normal) / radius
    solid = cos_incidence * (geometry.distance_mm / radius) ** 2  # 1 at normal incidence
    return (("two_theta", two_theta), ("chi", chi), ("solid_angle", solid))


def _tilt_rotation(tilt_deg: float, plane_deg: float) -> np.ndarray:
    """Rotation by ``tilt_deg`` about the in-plane axis at ``plane_deg`` from the x axis."""
    t, p = math.radians(tilt_deg), math.radians(plane_deg)
    axis = np.array([math.cos(p), math.sin(p), 0.0])
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(t) * k + (1 - math.cos(t)) * (k @ k)


def convert_two_theta(two_theta_deg: Any, unit: str, wavelength_A: float) -> np.ndarray:
    """2theta in degrees as ``"2theta"`` (unchanged), ``"q"`` (1/A) or ``"d"`` (A)."""
    tth = np.asarray(two_theta_deg, dtype=float)
    if unit == "2theta":
        return tth
    sin_theta = np.sin(np.radians(tth) / 2.0)
    if unit == "q":
        return 4.0 * math.pi * sin_theta / wavelength_A
    if unit == "d":
        with np.errstate(divide="ignore"):
            return wavelength_A / (2.0 * sin_theta)
    raise ValueError(f"unknown unit {unit!r}; choose from {UNITS}")


def _corrected(
    image: Any,
    geometry: PowderGeometry,
    mask: Any,
    solid_angle: bool,
    polarization: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel values divided by their corrections, the corrections, and the usable pixels."""
    data = np.asarray(to_host(image), dtype=np.float64)
    shape = (data.shape[-2], data.shape[-1])
    correction = np.ones(shape)
    if solid_angle:
        correction = correction * geometry.arrays(shape)["solid_angle"]
    if polarization is not None:
        correction = correction * geometry.polarization(shape, polarization)
    usable = np.isfinite(data)
    if mask is not None:
        usable &= ~np.asarray(mask, dtype=bool)
    return data, correction, usable


def _bins(values: np.ndarray, bins: int | None, x_range: Any, shape: tuple[int, int]) -> np.ndarray:
    lo, hi = (None, None) if x_range is None else x_range
    finite = values[np.isfinite(values)]
    lo = float(finite.min()) if lo is None else float(lo)
    hi = float(finite.max()) if hi is None else float(hi)
    n = int(bins) if bins else int(math.hypot(*shape) / 2)  # about one bin per pixel of radius
    return np.linspace(lo, hi, max(n, 2) + 1)


def integrate_1d(
    image: Any,
    geometry: PowderGeometry,
    bins: int | None = None,
    unit: str = "2theta",
    x_range: tuple[float | None, float | None] | None = None,
    chi_range: tuple[float | None, float | None] | None = None,
    mask: Any = None,
    solid_angle: bool = True,
    polarization: float | None = 0.0,
) -> dict[str, np.ndarray]:
    """The mean corrected intensity in rings of equal ``unit``: a powder pattern.

    Returns ``x`` (bin centres), ``intensity``, ``sigma`` (Poisson, from the
    raw counts) and ``pixels`` (pixels per bin). ``chi_range`` limits the
    azimuth (degrees); ``mask`` is True where pixels are left out;
    ``polarization`` is the factor of :meth:`PowderGeometry.polarization`
    (``None`` for no correction).
    """
    data, correction, usable = _corrected(image, geometry, mask, solid_angle, polarization)
    shape = data.shape[-2:]
    values = geometry.axis(shape, unit)
    if chi_range is not None:
        chi = geometry.arrays(shape)["chi"]
        lo, hi = chi_range
        usable &= (chi >= (-180 if lo is None else lo)) & (chi <= (180 if hi is None else hi))
    edges = _bins(values[usable], bins, x_range, shape)
    index = np.digitize(values, edges) - 1
    inside = usable & (index >= 0) & (index < len(edges) - 1)
    k = index[inside]
    n_bins = len(edges) - 1
    pixels = np.bincount(k, minlength=n_bins).astype(float)
    corrected = (data / correction)[inside]
    total = np.bincount(k, weights=corrected, minlength=n_bins)
    variance = np.bincount(
        k, weights=np.clip(data[inside], 0, None) / correction[inside] ** 2, minlength=n_bins
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        intensity = total / pixels
        sigma = np.sqrt(variance) / pixels
    return {
        "x": 0.5 * (edges[1:] + edges[:-1]),
        "intensity": intensity,
        "sigma": sigma,
        "pixels": pixels,
    }


def integrate_2d(
    image: Any,
    geometry: PowderGeometry,
    bins: int | None = None,
    chi_bins: int = 180,
    unit: str = "2theta",
    x_range: tuple[float | None, float | None] | None = None,
    mask: Any = None,
    solid_angle: bool = True,
    polarization: float | None = 0.0,
) -> dict[str, np.ndarray]:
    """The image unrolled into azimuth against ``unit`` (a cake): rings become lines.

    Returns ``x``, ``chi`` (bin centres) and ``intensity`` of shape
    ``(chi_bins, bins)``, NaN where no pixel falls.
    """
    data, correction, usable = _corrected(image, geometry, mask, solid_angle, polarization)
    shape = data.shape[-2:]
    values = geometry.axis(shape, unit)
    chi = geometry.arrays(shape)["chi"]
    edges = _bins(values[usable], bins, x_range, shape)
    chi_edges = np.linspace(-180.0, 180.0, int(chi_bins) + 1)
    ix = np.digitize(values, edges) - 1
    ic = np.clip(np.digitize(chi, chi_edges) - 1, 0, int(chi_bins) - 1)
    inside = usable & (ix >= 0) & (ix < len(edges) - 1)
    n_x = len(edges) - 1
    flat = ic[inside] * n_x + ix[inside]
    size = int(chi_bins) * n_x
    pixels = np.bincount(flat, minlength=size).astype(float)
    total = np.bincount(flat, weights=(data / correction)[inside], minlength=size)
    with np.errstate(invalid="ignore", divide="ignore"):
        intensity = (total / pixels).reshape(int(chi_bins), n_x)
    return {
        "x": 0.5 * (edges[1:] + edges[:-1]),
        "chi": 0.5 * (chi_edges[1:] + chi_edges[:-1]),
        "intensity": intensity,
    }


def calibrate_from_ring(
    image: Any,
    d_spacing_A: float,
    energy_keV: float,
    pixel_um: float,
    percentile: float = 99.0,
    guess: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Beam centre and distance from one powder ring of known d-spacing (no tilt).

    The brightest pixels (above ``percentile``) are taken as the ring, a circle
    is fitted to them by least squares, and the distance follows from its
    radius and the ring's 2theta: ``distance = radius / tan(2theta)``. With
    ``guess`` (a rough centre), pixels far from the median radius around it
    are dropped first, which helps when other rings are bright too.
    """
    data = np.asarray(to_host(image), dtype=float)
    finite = np.isfinite(data)
    level = np.percentile(data[finite], percentile)
    ys, xs = np.nonzero(finite & (data >= level))
    if ys.size < 10:
        raise ValueError("too few bright pixels to fit a ring; lower the percentile")
    if guess is not None:
        r = np.hypot(ys - guess[0], xs - guess[1])
        near = np.abs(r - np.median(r)) < 0.1 * np.median(r)
        ys, xs = ys[near], xs[near]
    # algebraic circle fit (Kasa): x^2 + y^2 + D x + E y + F = 0
    design = np.column_stack([xs, ys, np.ones_like(xs)]).astype(float)
    rhs = -(xs.astype(float) ** 2 + ys.astype(float) ** 2)
    (d_coef, e_coef, f_coef), *_ = np.linalg.lstsq(design, rhs, rcond=None)
    cx, cy = -d_coef / 2.0, -e_coef / 2.0
    radius_px = math.sqrt(max(cx * cx + cy * cy - f_coef, 0.0))
    wavelength = float(energy_to_wavelength(energy_keV))
    two_theta = 2.0 * math.asin(wavelength / (2.0 * d_spacing_A))
    distance_mm = radius_px * pixel_um * 1e-3 / math.tan(two_theta)
    scatter = float(np.std(np.hypot(xs - cx, ys - cy) - radius_px))
    return {
        "center_y": float(cy),
        "center_x": float(cx),
        "radius_px": radius_px,
        "distance_mm": float(distance_mm),
        "two_theta_deg": math.degrees(two_theta),
        "ring_scatter_px": scatter,
        "pixels": float(ys.size),
    }


def scherrer(
    fwhm_deg: Any,
    two_theta_deg: Any,
    energy_keV: float,
    shape_factor: float = 0.9,
    instrumental_deg: float = 0.0,
) -> Any:
    """Crystallite size in nanometres from a peak width (Scherrer equation).

    ``fwhm_deg`` is the peak's full width in 2theta; ``instrumental_deg`` is
    removed in quadrature first.
    """
    width = np.sqrt(np.clip(np.asarray(fwhm_deg, dtype=float) ** 2 - instrumental_deg**2, 0, None))
    theta = np.radians(np.asarray(two_theta_deg, dtype=float) / 2.0)
    wavelength_nm = float(energy_to_wavelength(energy_keV)) / 10.0
    with np.errstate(divide="ignore"):
        return shape_factor * wavelength_nm / (np.radians(width) * np.cos(theta))
