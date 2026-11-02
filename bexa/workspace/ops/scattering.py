"""XRD and XRR: powder integration, cakes, ring calibration, reflectivity curves."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from bexa._log import progress
from bexa.analysis import xrd, xrr
from bexa.workspace.objects import DataObject
from bexa.workspace.ops._base import DETECTOR, curve, curve_xy, new, other_dims, require

__all__ = [
    "cake",
    "calibrate_ring",
    "critical_angle",
    "footprint_correction",
    "integrate_powder",
    "kiessig_thickness",
    "reflectivity",
]


def _geometry(
    obj: DataObject,
    distance_mm: float,
    center_y: float,
    center_x: float,
    pixel_um: float,
    energy_keV: float | None,
    tilt_deg: float,
    tilt_plane_deg: float,
) -> xrd.PowderGeometry:
    energy = energy_keV or obj.meta.get("energy_keV")
    if not energy:
        raise ValueError("Energy: the object does not know its energy; give it")
    return xrd.PowderGeometry(
        float(distance_mm),
        float(center_y),
        float(center_x),
        float(pixel_um),
        float(energy),
        float(tilt_deg),
        float(tilt_plane_deg),
    )


def _frames(obj: DataObject) -> xr.DataArray:
    """The detector data as ``(frames..., y, x)`` with any window offset kept in its coords."""
    require(obj, (*DETECTOR, "map"), "a powder integration")
    return obj.array


def _window_offset(da: xr.DataArray) -> tuple[float, float]:
    """The full-detector pixel of the window's first pixel (bexa writes it in the coords)."""
    dy, dx = (str(d) for d in da.dims[-2:])
    oy = float(da.coords[dy].values[0]) if dy in da.coords else 0.0
    ox = float(da.coords[dx].values[0]) if dx in da.coords else 0.0
    return oy, ox


def integrate_powder(
    obj: DataObject,
    distance_mm: float = 200.0,
    center_y: float = 0.0,
    center_x: float = 0.0,
    pixel_um: float = 75.0,
    energy_keV: float | None = None,
    unit: str = "2theta",
    bins: int | None = None,
    x_range: Sequence[float | None] | None = None,
    chi_range: Sequence[float | None] | None = None,
    tilt_deg: float = 0.0,
    tilt_plane_deg: float = 0.0,
    per_frame: bool = False,
    solid_angle: bool = True,
    polarization: float | None = 0.0,
) -> DataObject:
    """The powder pattern: mean intensity in rings of equal 2theta (or q, d), with uncertainties.

    The beam centre is in full-detector pixels; an image cut to a window keeps
    its offset. A stack gives the pattern of its summed frames, or one per
    frame (a map against the stack's motor dims) with ``per_frame``.
    """
    da = _frames(obj)
    oy, ox = _window_offset(da)
    geometry = _geometry(
        obj,
        distance_mm,
        center_y - oy,
        center_x - ox,
        pixel_um,
        energy_keV,
        tilt_deg,
        tilt_plane_deg,
    )
    options: dict[str, Any] = {
        "bins": bins,
        "unit": unit,
        "x_range": x_range,
        "chi_range": chi_range,
        "solid_angle": solid_angle,
        "polarization": polarization,
    }
    units = {"2theta": "deg", "q": "1/A", "d": "A"}[unit]
    motors = other_dims(da) if obj.kind == "stack" else ()
    if not motors or not per_frame:
        image = da.sum(list(motors)) if motors else da
        pattern = xrd.integrate_1d(np.asarray(image.values, dtype=float), geometry, **options)
        data = curve(
            pattern["x"], pattern["intensity"], unit, "intensity", coord_attrs={"units": units}
        )
        return new(
            obj,
            data,
            f"pattern_{unit}",
            sigma=pattern["sigma"],
            meta={"geometry": geometry.to_dict(), "quantity": "intensity"},
        )
    stacked = da.stack(frame=list(motors))
    summed = xrd.integrate_1d(np.asarray(stacked.sum("frame").values, float), geometry, **options)
    centres = summed["x"]
    step = float(centres[1] - centres[0]) if centres.size > 1 else 1.0
    # every frame on the bins of the summed pattern, so the patterns line up
    options.update(bins=centres.size, x_range=(centres[0] - step / 2, centres[-1] + step / 2))
    rows = np.empty((stacked.sizes["frame"], centres.size))
    errors = np.empty_like(rows)
    for i in progress(range(rows.shape[0]), total=rows.shape[0], desc="patterns", enabled=False):
        pattern = xrd.integrate_1d(
            np.asarray(stacked.isel(frame=i).values, float), geometry, **options
        )
        rows[i], errors[i] = pattern["intensity"], pattern["sigma"]
    first = {"x": centres}
    index = stacked.indexes["frame"]
    data = xr.DataArray(rows, dims=("frame", unit), coords={"frame": index, unit: first["x"]})
    data = data.unstack("frame").transpose(*motors, unit)
    sigma = xr.DataArray(errors, dims=("frame", unit), coords={"frame": index, unit: first["x"]})
    sigma = sigma.unstack("frame").transpose(*motors, unit)
    return new(obj, data, f"patterns_{unit}", sigma=sigma, meta={"geometry": geometry.to_dict()})


def cake(
    obj: DataObject,
    distance_mm: float = 200.0,
    center_y: float = 0.0,
    center_x: float = 0.0,
    pixel_um: float = 75.0,
    energy_keV: float | None = None,
    unit: str = "2theta",
    bins: int | None = None,
    chi_bins: int = 180,
    x_range: Sequence[float | None] | None = None,
    tilt_deg: float = 0.0,
    tilt_plane_deg: float = 0.0,
) -> DataObject:
    """The image unrolled into azimuth chi against 2theta (q, d): rings become lines."""
    da = _frames(obj)
    oy, ox = _window_offset(da)
    geometry = _geometry(
        obj,
        distance_mm,
        center_y - oy,
        center_x - ox,
        pixel_um,
        energy_keV,
        tilt_deg,
        tilt_plane_deg,
    )
    image = da.sum(list(other_dims(da))) if obj.kind == "stack" else da
    found = xrd.integrate_2d(
        np.asarray(image.values, dtype=float), geometry, bins, int(chi_bins), unit, x_range
    )
    data = xr.DataArray(
        found["intensity"],
        dims=("chi", unit),
        coords={"chi": found["chi"], unit: found["x"]},
        name="cake",
    )
    return new(obj, data, f"cake_{unit}", meta={"geometry": geometry.to_dict()})


def calibrate_ring(
    obj: DataObject,
    d_spacing_A: float = 3.1356,
    pixel_um: float = 75.0,
    energy_keV: float | None = None,
    percentile: float = 99.0,
) -> DataObject:
    """Beam centre and detector distance from one powder ring of known d-spacing."""
    require(obj, ("image",), "Calibrate from a Ring")
    energy = energy_keV or obj.meta.get("energy_keV")
    if not energy:
        raise ValueError("Energy: the object does not know its energy; give it")
    oy, ox = _window_offset(obj.array)
    found = xrd.calibrate_from_ring(
        obj.array.values, d_spacing_A, float(energy), pixel_um, percentile
    )
    found["center_y"] += oy
    found["center_x"] += ox
    table = pd.DataFrame([{"quantity": k, "value": v} for k, v in found.items()])
    return new(obj, table, "calibration", kind="table", meta={"calibration": found})


def reflectivity(
    obj: DataObject,
    angle: str = "theta",
    i0: float | None = None,
    beam_mm: float | None = None,
    sample_mm: float | None = None,
    profile: str = "gaussian",
    energy_keV: float | None = None,
    x: str = "q",
    theta_offset: float = 0.0,
) -> DataObject:
    """A specular reflectivity curve from a counter curve against theta (or 2theta).

    Divided by the direct beam ``i0`` if given, corrected for the footprint if
    the beam and sample sizes are given, against q_z (``x="q"``) or theta.
    """
    _, xs, ys, sigma = curve_xy(obj)
    theta = (xs / 2.0 if angle == "two_theta" else xs) - theta_offset
    values, errors = ys.copy(), None if sigma is None else sigma.copy()
    if i0:
        values, errors = values / i0, None if errors is None else errors / i0
    if beam_mm and sample_mm:
        fraction = xrr.footprint_fraction(theta, beam_mm, sample_mm, profile)
        with np.errstate(invalid="ignore", divide="ignore"):
            values = values / fraction
            errors = None if errors is None else errors / fraction
    if x == "q":
        energy = energy_keV or obj.meta.get("energy_keV")
        if not energy:
            raise ValueError("Energy: needed for q_z; give it")
        axis, coord = "qz", xrr.qz(theta, float(energy))
        attrs = {"units": "1/A"}
    else:
        axis, coord, attrs = "theta", theta, {"units": "deg"}
    data = curve(coord, values, axis, "reflectivity", coord_attrs=attrs)
    return new(obj, data, "xrr", sigma=errors, meta={"quantity": "reflectivity"})


def footprint_correction(
    obj: DataObject,
    beam_mm: float = 0.1,
    sample_mm: float = 10.0,
    profile: str = "gaussian",
    energy_keV: float | None = None,
) -> DataObject:
    """Divide a reflectivity curve (against theta or q_z) by the share of the beam on the sample."""
    dim, xs, ys, sigma = curve_xy(obj)
    if dim in ("qz", "q"):
        energy = energy_keV or obj.meta.get("energy_keV")
        if not energy:
            raise ValueError("Energy: needed to turn q_z into theta; give it")
        theta = xrr.theta_from_qz(xs, float(energy))
    else:
        theta = xs
    fraction = xrr.footprint_fraction(theta, beam_mm, sample_mm, profile)
    with np.errstate(invalid="ignore", divide="ignore"):
        data = obj.array.copy(data=ys / fraction)
        errors = None if sigma is None else sigma / fraction
    return new(
        obj, data, "footprint", sigma=errors, meta={"footprint": [beam_mm, sample_mm, profile]}
    )


def critical_angle(obj: DataObject, energy_keV: float | None = None) -> DataObject:
    """The critical angle (and q_c, electron density) of a reflectivity curve."""
    dim, xs, ys, _ = curve_xy(obj)
    energy = energy_keV or obj.meta.get("energy_keV")
    found = xrr.critical_angle(
        xs, ys, None if not energy else float(energy), unit="q" if dim in ("qz", "q") else "theta"
    )
    lines = [f"{k}: {v:.6g}" for k, v in found.items()]
    return new(
        obj,
        "critical angle of " + obj.name + "\n" + "\n".join(lines),
        "critical",
        kind="text",
        meta={"critical": found},
    )


def kiessig_thickness(
    obj: DataObject, q_low: float | None = None, q_high: float | None = None
) -> DataObject:
    """Film thickness from the spacing of the Kiessig fringes of a curve against q_z."""
    dim, xs, ys, _ = curve_xy(obj)
    if dim not in ("qz", "q"):
        raise ValueError("Kiessig thickness needs a curve against q_z (Reflectivity, x = q)")
    found = xrr.kiessig_thickness(xs, ys, (q_low, q_high))
    lines = [
        f"thickness: {found['thickness_A']:.1f} A ({found['thickness_A'] / 10:.2f} nm)",
        f"fringe spacing: {found['fringe_spacing_q']:.5g} 1/A",
        f"strength (peak over mean of the spectrum): {found['strength']:.3g}",
    ]
    return new(
        obj,
        "Kiessig fringes of " + obj.name + "\n" + "\n".join(lines),
        "thickness",
        kind="text",
        meta={"kiessig": found},
    )
