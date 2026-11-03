"""Process: one object in, a new one out. Uncertainties are propagated where they exist."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import xarray as xr

from bexa.analysis import curves
from bexa.core.roi import ROI
from bexa.workspace.objects import DataObject
from bexa.workspace.ops._base import (
    ANY,
    DETECTOR,
    IMAGES,
    curve,
    curve_xy,
    like,
    new,
    other_dims,
    pairs,
    pixel_dims,
    range_slices,
    require,
    trapezoid,
)

__all__ = [
    "FUNCTIONS",
    "align_peak",
    "apply_function",
    "bin_pixels",
    "clip_values",
    "convert_axis",
    "cumulative_integral",
    "derivative",
    "equalize_histogram",
    "flat_field",
    "mask_pixels",
    "merge_points",
    "normalize",
    "remove_hot_pixels",
    "replace_nan",
    "resample",
    "shift_axis",
    "shift_image",
    "smooth",
    "subtract_background",
    "subtract_dark",
    "transform",
]

NORMALIZE_BY = ("max", "range", "integral", "mean", "point", "number", "monitor")
TRANSFORMS = ("flip rows", "flip columns", "rotate 90", "rotate 180", "rotate 270", "transpose")
FUNCTIONS = ("log10", "ln", "exp", "sqrt", "square", "abs", "reciprocal", "negate")
AXIS_UNITS = ("2theta", "theta", "q", "d")


def _sigma(obj: DataObject) -> Any:
    return None if obj.sigma is None else obj.sigma


def normalize(
    obj: DataObject,
    by: str = "max",
    value: float | None = None,
    monitor: DataObject | None = None,
    at: float | None = None,
    to: float = 1.0,
) -> DataObject:
    """Divide by the maximum, the range (to 0..1), the integral, the mean, the value at ``at``,
    a number, or a monitor curve (relative to its mean); the result's reference becomes ``to``."""
    require(obj, ANY, "Normalize")
    da, sigma = obj.array, _sigma(obj)
    offset = 0.0
    if by == "max":
        divisor = float(np.nanmax(da.values))
    elif by == "range":
        offset = float(np.nanmin(da.values))
        divisor = float(np.nanmax(da.values)) - offset
    elif by == "mean":
        divisor = float(np.nanmean(da.values))
    elif by == "integral":
        if obj.kind == "curve":
            dim, x, y, _ = curve_xy(obj)
            keep = np.isfinite(y)
            divisor = float(abs(trapezoid(y[keep], x[keep])))
        else:
            divisor = float(np.nansum(da.values))
    elif by == "point":
        if at is None:
            raise ValueError("At: give the x value whose value becomes the reference")
        dim = str(da.dims[0])
        divisor = float(da.interp({dim: at}).values) if da.ndim == 1 else float("nan")
    elif by == "number":
        if value is None:
            raise ValueError("Value: give the number to divide by")
        divisor = float(value)
    elif by == "monitor":
        if monitor is None:
            raise ValueError("Monitor: choose the curve to divide by")
        mon = monitor.array / float(np.nanmean(monitor.array.values))
        mon = mon.interp_like(da) if set(mon.dims) <= set(da.dims) and mon.ndim == 1 else mon
        result = da / mon
        out_sigma = None if sigma is None else sigma / np.abs(mon)
        return new(
            obj,
            result * to,
            "norm",
            sigma=None if out_sigma is None else out_sigma * abs(to),
            meta={"normalized_by": f"monitor {monitor.name}"},
            parents=[obj.name, monitor.name],
        )
    else:
        raise ValueError(f"by must be one of {NORMALIZE_BY}")
    if not np.isfinite(divisor) or divisor == 0:
        raise ValueError(f"cannot normalise by {by}: the divisor is {divisor}")
    factor = to / divisor
    result = (da - offset) * factor
    out_sigma = None if sigma is None else sigma * abs(factor)
    return new(obj, result, "norm", sigma=out_sigma, meta={"normalized_by": by})


def subtract_dark(
    obj: DataObject, dark: DataObject, scale: float = 1.0, clip: bool = False
) -> DataObject:
    """Subtract a dark image (a dark stack is averaged first), scaled by ``scale``."""
    require(obj, (*DETECTOR, "map"), "Subtract Dark")
    require(dark, (*DETECTOR, "map"), "the dark")
    dark_image = dark.array
    if dark.kind == "stack":
        dark_image = dark_image.mean(other_dims(dark_image))
    dark_values = np.asarray(dark_image.values, dtype=float)
    if dark_values.shape != obj.array.shape[-2:]:
        raise ValueError(f"the dark is {dark_values.shape}, the frames {obj.array.shape[-2:]}")
    result = obj.array - scale * dark_values
    if clip:
        result = result.clip(min=0)
    sigma = obj.sigma
    if dark.sigma is not None and obj.sigma is not None and dark.kind != "stack":
        sigma = np.sqrt(obj.sigma**2 + (scale * np.asarray(dark.sigma.values)) ** 2)
    return new(
        obj, result, "dark", sigma=sigma, meta={"dark": dark.name}, parents=[obj.name, dark.name]
    )


def flat_field(
    obj: DataObject, flat: DataObject, dark: DataObject | None = None, minimum: float = 0.05
) -> DataObject:
    """Divide by a flat field normalised to a mean of 1 (after its dark); weak pixels are NaN."""
    require(obj, (*DETECTOR, "map"), "Flat Field")
    flat_image = flat.array.mean(other_dims(flat.array)) if flat.kind == "stack" else flat.array
    values = np.asarray(flat_image.values, dtype=float)
    if dark is not None:
        dark_image = dark.array.mean(other_dims(dark.array)) if dark.kind == "stack" else dark.array
        values = values - np.asarray(dark_image.values, dtype=float)
    values = values / np.nanmean(values)
    values = np.where(values >= minimum, values, np.nan)
    result = obj.array / values
    sigma = None if obj.sigma is None else obj.sigma / np.abs(values)
    parents = [obj.name, flat.name] + ([dark.name] if dark is not None else [])
    return new(obj, result, "flat", sigma=sigma, meta={"flat": flat.name}, parents=parents)


def subtract_background(
    obj: DataObject,
    method: str = "linear",
    regions: Sequence[float] | None = None,
    order: int = 2,
    value: float | None = None,
) -> DataObject:
    """Fit a background to the curve's background ``regions`` (pairs of x: 1, 2, 9, 10) and
    subtract it; ``method="value"`` subtracts a number."""
    _, x, y, sigma = curve_xy(obj)
    if method == "value":
        if value is None:
            raise ValueError("Value: give the number to subtract")
        background = np.full_like(y, float(value))
    else:
        background = curves.background(x, y, method, pairs(regions, "Regions") or None, order)
    result = like(obj.array, y - background)
    return new(obj, result, "bgsub", sigma=sigma, meta={"background": method})


def remove_hot_pixels(obj: DataObject, threshold: float = 8.0, size: int = 3) -> DataObject:
    """Replace pixels far above their neighbours (``threshold`` times the local noise) by the
    local median; every frame of a stack separately, the frames in threads."""
    require(obj, (*DETECTOR, "map"), "Remove Hot Pixels")
    from bexa.core.backend import median_frames

    data = np.asarray(obj.array.values, dtype=float)
    local = median_frames(data, int(size))
    deviation = np.abs(data - local)
    noise = median_frames(deviation, int(size)) * 1.4826 + np.sqrt(np.abs(local)) + 1e-12
    hot = (data - local) > threshold * noise
    cleaned = np.where(hot, local, data)
    return new(
        obj,
        like(obj.array, cleaned),
        "dehot",
        sigma=obj.sigma,
        meta={"hot_pixels_replaced": int(hot.sum())},
    )


def mask_pixels(
    obj: DataObject,
    low: float | None = None,
    high: float | None = None,
    rois: Mapping[str, ROI] | None = None,
    inside: bool = False,
) -> DataObject:
    """Set to NaN (left out of every later step) the values below ``low``, above ``high``,
    and the pixels outside the ROIs (``inside=True``: inside them)."""
    require(obj, ANY, "Mask Pixels")
    da = obj.array
    keep = np.isfinite(np.asarray(da.values, dtype=float))
    values = np.asarray(da.values, dtype=float)
    if low is not None:
        keep &= values >= low
    if high is not None:
        keep &= values <= high
    if rois:
        in_any = np.zeros(values.shape, dtype=bool)
        for roi in rois.values():
            slices = range_slices(da, roi)
            in_any[tuple(slices.get(str(d), slice(None)) for d in da.dims)] = True
        keep &= ~in_any if inside else in_any
    result = da.where(keep)
    sigma = None if obj.sigma is None else obj.sigma.where(keep)
    return new(obj, result, "masked", sigma=sigma, meta={"masked_pixels": int((~keep).sum())})


def smooth(
    obj: DataObject, method: str = "gaussian", width: float = 2.0, order: int = 2
) -> DataObject:
    """Smooth a curve, or an image (every frame of a stack) over its two image dims."""
    require(obj, ANY, "Smooth")
    da = obj.array
    if da.ndim == 1:
        result = curves.smooth(da.values, method, width, order)
    else:
        from scipy import ndimage

        data = np.asarray(da.values, dtype=float)
        spread = (0,) * (data.ndim - 2) + (float(width),) * 2
        window = (1,) * (data.ndim - 2) + (max(round(width), 1),) * 2
        if method == "gaussian":
            result = ndimage.gaussian_filter(data, spread, mode="nearest")
        elif method == "moving average":
            result = ndimage.uniform_filter(data, window, mode="nearest")
        elif method == "median":
            result = ndimage.median_filter(data, size=window, mode="nearest")
        else:
            raise ValueError("an image is smoothed by gaussian, moving average or median")
    return new(obj, like(da, result), "smooth", meta={"smoothed": f"{method} {width}"})


def bin_pixels(obj: DataObject, factor: int = 2, how: str = "mean") -> DataObject:
    """Combine ``factor`` x ``factor`` pixels (or points of a map) into one."""
    require(obj, (*DETECTOR, "map"), "Bin Pixels")
    da = obj.array
    dims = pixel_dims(da) or tuple(str(d) for d in da.dims[-2:])
    coarse = da.coarsen({d: int(factor) for d in dims}, boundary="trim")
    result = coarse.sum() if how == "sum" else coarse.mean()
    sigma = None
    if obj.sigma is not None:
        squares = (obj.sigma**2).coarsen({d: int(factor) for d in dims}, boundary="trim").sum()
        sigma = np.sqrt(squares) / (1 if how == "sum" else int(factor) ** 2)
    return new(obj, result, f"bin{int(factor)}", sigma=sigma)


def transform(obj: DataObject, how: str = "rotate 90") -> DataObject:
    """Flip, rotate (counter-clockwise) or transpose the last two dims."""
    require(obj, (*DETECTOR, "map", "rgb"), "Flip / Rotate")
    if how not in TRANSFORMS:
        raise ValueError(f"how must be one of {TRANSFORMS}")

    def apply(da: xr.DataArray) -> xr.DataArray:
        dims = list(da.dims)
        offset = 1 if obj.kind == "rgb" else 0
        dy, dx = dims[-2 - offset], dims[-1 - offset]
        flip_y = {dy: slice(None, None, -1)}
        flip_x = {dx: slice(None, None, -1)}
        swapped = dims[:]
        swapped[-2 - offset], swapped[-1 - offset] = dx, dy
        if how == "flip rows":
            return da.isel(flip_y)
        if how == "flip columns":
            return da.isel(flip_x)
        if how == "transpose":
            return da.transpose(*swapped)
        if how == "rotate 180":
            return da.isel({**flip_y, **flip_x})
        if how == "rotate 90":  # counter-clockwise: transpose, then flip the new rows
            return da.transpose(*swapped).isel({dx: slice(None, None, -1)})
        return da.transpose(*swapped).isel({dy: slice(None, None, -1)})

    sigma = None if obj.sigma is None else apply(obj.sigma)
    return new(obj, apply(obj.array), how.replace(" ", ""), sigma=sigma)


def apply_function(obj: DataObject, function: str = "log10") -> DataObject:
    """A function of every value, the uncertainties propagated to first order."""
    require(obj, ANY, "Function of Values")
    da = obj.array.astype(float)
    s = obj.sigma
    with np.errstate(invalid="ignore", divide="ignore"):
        table = {
            "log10": (lambda v: np.log10(v), lambda v, e: e / (np.abs(v) * np.log(10))),
            "ln": (lambda v: np.log(v), lambda v, e: e / np.abs(v)),
            "exp": (lambda v: np.exp(v), lambda v, e: np.exp(v) * e),
            "sqrt": (lambda v: np.sqrt(v), lambda v, e: e / (2 * np.sqrt(np.abs(v)))),
            "square": (lambda v: v**2, lambda v, e: 2 * np.abs(v) * e),
            "abs": (lambda v: np.abs(v), lambda v, e: e),
            "reciprocal": (lambda v: 1.0 / v, lambda v, e: e / v**2),
            "negate": (lambda v: -v, lambda v, e: e),
        }
        if function not in table:
            raise ValueError(f"function must be one of {FUNCTIONS}")
        forward, error = table[function]
        result = forward(da)
        sigma = None if s is None else error(da, s)
    return new(obj, result, function, sigma=sigma, meta={"function": function})


def clip_values(
    obj: DataObject,
    low: float | None = None,
    high: float | None = None,
    replace: float | None = None,
) -> DataObject:
    """Limit values to ``low..high``: set to the limit, or to ``replace`` (0, NaN, ...)."""
    require(obj, ANY, "Clip Values")
    da = obj.array
    if replace is None:
        result = da.clip(min=low, max=high)
    else:
        outside = xr.zeros_like(da, dtype=bool)
        if low is not None:
            outside |= da < low
        if high is not None:
            outside |= da > high
        result = da.where(~outside, float(replace))
    return new(obj, result, "clip", sigma=obj.sigma)


def replace_nan(obj: DataObject, value: float = 0.0) -> DataObject:
    """Replace NaN (masked or missing points) by ``value``."""
    require(obj, ANY, "Replace NaN")
    sigma = None if obj.sigma is None else obj.sigma.fillna(0.0)
    return new(obj, obj.array.fillna(value), "filled", sigma=sigma)


def resample(
    obj: DataObject,
    step: float | None = None,
    points: int | None = None,
    low: float | None = None,
    high: float | None = None,
) -> DataObject:
    """Interpolate a curve onto evenly spaced x (by ``step`` or a number of ``points``)."""
    dim, x, y, sigma = curve_xy(obj)
    lo = float(np.nanmin(x)) if low is None else float(low)
    hi = float(np.nanmax(x)) if high is None else float(high)
    if step:
        grid = np.arange(lo, hi + 0.5 * step, float(step))
    else:
        grid = np.linspace(lo, hi, int(points) if points else x.size)
    values, errors = curves.resample(x, y, grid, sigma)
    return new(obj, curve(grid, values, dim, str(obj.array.name or "")), "resampled", sigma=errors)


def derivative(obj: DataObject) -> DataObject:
    """``dy/dx`` of a curve, with uncertainties."""
    dim, x, y, sigma = curve_xy(obj)
    order = np.argsort(x)
    dy, ds = curves.derivative(x[order], y[order], None if sigma is None else sigma[order])
    return new(obj, curve(x[order], dy, dim, "derivative"), "deriv", sigma=ds)


def cumulative_integral(obj: DataObject) -> DataObject:
    """The integral of a curve from its first point."""
    dim, x, y, sigma = curve_xy(obj)
    order = np.argsort(x)
    area, errors = curves.cumulative_integral(
        x[order], y[order], None if sigma is None else sigma[order]
    )
    return new(obj, curve(x[order], area, dim, "integral"), "integral", sigma=errors)


def shift_axis(
    obj: DataObject,
    dim: str | None = None,
    offset: float = 0.0,
    scale: float = 1.0,
    new_name: str = "",
    units: str = "",
) -> DataObject:
    """``x -> scale * x + offset`` along a dim (default: a curve's x), optionally renamed."""
    require(obj, ANY, "Scale / Shift Axis")
    da = obj.array
    dim = dim or str(da.dims[0])
    values = obj.coord(dim).astype(float) * scale + offset
    result = da.assign_coords({dim: values})
    if units:
        result.coords[dim].attrs["units"] = units
    sigma = None if obj.sigma is None else obj.sigma.assign_coords({dim: values})
    if new_name and new_name != dim:
        result = result.rename({dim: new_name})
        sigma = None if sigma is None else sigma.rename({dim: new_name})
    return new(obj, result, "axis", sigma=sigma)


def convert_axis(
    obj: DataObject, to: str = "q", source: str = "2theta", energy_keV: float | None = None
) -> DataObject:
    """Convert a curve's x between 2theta, theta (degrees), q (1/A) and d (A)."""
    from bexa.core.units import energy_to_wavelength

    _, x, y, sigma = curve_xy(obj)
    energy = energy_keV or obj.meta.get("energy_keV")
    if energy is None:
        raise ValueError("Energy: the object does not know its energy; give it")
    wavelength = float(energy_to_wavelength(float(energy)))
    with np.errstate(invalid="ignore", divide="ignore"):
        if source == "2theta":
            theta = np.radians(x) / 2
        elif source == "theta":
            theta = np.radians(x)
        elif source == "q":
            theta = np.arcsin(x * wavelength / (4 * np.pi))
        elif source == "d":
            theta = np.arcsin(wavelength / (2 * x))
        else:
            raise ValueError(f"source must be one of {AXIS_UNITS}")
        new_x = {
            "2theta": np.degrees(2 * theta),
            "theta": np.degrees(theta),
            "q": 4 * np.pi * np.sin(theta) / wavelength,
            "d": wavelength / (2 * np.sin(theta)),
        }[to]
    units = {"2theta": "deg", "theta": "deg", "q": "1/A", "d": "A"}[to]
    data = curve(new_x, y, to, str(obj.array.name or ""), coord_attrs={"units": units})
    return new(obj, data, to, sigma=sigma, meta={"energy_keV": float(energy)})


def align_peak(
    obj: DataObject,
    method: str = "maximum",
    target: str = "zero",
    value: float | None = None,
    reference: DataObject | None = None,
    x_range: Sequence[float | None] | None = None,
) -> DataObject:
    """Move a curve along x so its peak sits at zero, at a value, or at another curve's peak."""
    dim, x, y, sigma = curve_xy(obj)
    peak = curves.peak_position(x, y, method, x_range)
    if target == "zero":
        goal = 0.0
    elif target == "a value":
        if value is None:
            raise ValueError("Value: give the x where the peak goes")
        goal = float(value)
    elif target == "another curve":
        if reference is None:
            raise ValueError("Other curve: choose the curve whose peak to match")
        _, rx, ry, _ = curve_xy(reference)
        goal = curves.peak_position(rx, ry, method, x_range)
    else:
        raise ValueError("target must be zero, a value or another curve")
    shifted = obj.array.assign_coords({dim: x + (goal - peak)})
    s = None if sigma is None else obj.sigma.assign_coords({dim: x + (goal - peak)})
    return new(obj, shifted, "aligned", sigma=s, meta={"shift": goal - peak, "peak_was": peak})


def merge_points(obj: DataObject, tolerance: float | None = None) -> DataObject:
    """Sort a curve by x and average the points at the same x."""
    dim, x, y, sigma = curve_xy(obj)
    mx, my, ms = curves.merge_points(x, y, sigma, tolerance)
    return new(obj, curve(mx, my, dim, str(obj.array.name or "")), "merged", sigma=ms)


def shift_image(obj: DataObject, rows: float = 0.0, columns: float = 0.0) -> DataObject:
    """Move an image (every frame of a stack) by fractions of a pixel."""
    require(obj, (*DETECTOR, *IMAGES), "Shift Image")
    from scipy.ndimage import shift

    data = np.asarray(obj.array.values, dtype=float)
    offsets = (0.0,) * (data.ndim - 2) + (float(rows), float(columns))
    moved = shift(data, offsets, order=1, mode="constant", cval=np.nan)
    return new(obj, like(obj.array, moved), "shifted", meta={"shift_px": [rows, columns]})


def equalize_histogram(obj: DataObject) -> DataObject:
    """Spread the values evenly over 0..1 (by rank): faint contrast becomes visible."""
    require(obj, (*IMAGES, "stack"), "Equalize Contrast")
    values = np.asarray(obj.array.values, dtype=float)
    finite = np.isfinite(values)
    ranks = np.full(values.shape, np.nan)
    order = np.argsort(values[finite], kind="stable")
    filled = np.empty(order.size)
    filled[order] = np.linspace(0.0, 1.0, order.size)
    ranks[finite] = filled
    return new(obj, like(obj.array, ranks), "equalized")
