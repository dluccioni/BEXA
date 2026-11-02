"""Extract: from scans and stacks to images, curves, maps and tables.

A scan is never loaded whole: its operations stream the frames through
:func:`bexa.reduce` (one pass, the memory budget respected), so they work on
scans of any size. Stacks already in memory are reduced with xarray.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from bexa.core.roi import ROI
from bexa.workspace.objects import DataObject
from bexa.workspace.ops._base import (
    ANY,
    IMAGES,
    REDUCE_HOWS,
    curve,
    new,
    other_dims,
    pixel_dims,
    poisson,
    reduce_over,
    require,
    select_ranges,
    sigma_over,
    windowed,
)

__all__ = [
    "com_maps",
    "counters",
    "crop",
    "frame_stats",
    "histogram",
    "line_profile",
    "point_table",
    "preview",
    "profile",
    "projections",
    "read_volume",
    "reduce_dims",
    "reduce_frames",
    "roi_curves",
    "roi_statistics",
    "select_range",
    "take_slice",
]

SCAN_HOWS = ("sum", "mean", "max", "min")


def scan_meta(scan: Any) -> dict[str, Any]:
    """What a result of a scan should remember about it."""
    s = scan.structure
    meta: dict[str, Any] = {"source": str(scan.files[0]) if scan.files else scan.name}
    number = getattr(scan.source, "scan", None)
    if isinstance(number, (int, np.integer)):
        meta["scan"] = int(number)
    meta.update({"dataset": scan.name, "scan_type": s.scan_type, "title": s.title})
    if s.energy_keV is not None:
        meta["energy_keV"] = float(s.energy_keV)
    positions = {}
    for name, value in s.scalars.items():  # the fixed motors: where the sample was
        try:
            positions[str(name)] = float(value)
        except (TypeError, ValueError):
            continue
    meta["positions"] = positions
    return meta


def _counts(scan: Any) -> bool:
    """True when the scan stores integer detector counts (Poisson uncertainties apply)."""
    return np.dtype(scan.source.dtype).kind in "iu"


def _from_scan(obj: DataObject, data: xr.DataArray, suffix: str, **kwargs: Any) -> DataObject:
    return new(obj, data, suffix, meta={**scan_meta(obj.data), **kwargs.pop("meta", {})}, **kwargs)


def channel_on_grid(scan: Any, name: str) -> xr.DataArray:
    """A per-frame channel (counter, monitor, readback) placed on the scan's motor grid."""
    s = scan.structure
    if name not in s.per_frame:
        raise ValueError(f"no per-frame channel {name!r}; the scan has {sorted(s.per_frame)}")
    values = np.asarray(s.per_frame[name], dtype=float)
    index = s.frame_index
    out = np.full(index.shape, np.nan)
    ok = (index >= 0) & (index < values.size)
    out[ok] = values[index[ok]]
    coords = {d: s.coords[d] for d in s.motor_dims}
    return xr.DataArray(out, dims=s.motor_dims, coords=coords, name=name)


# --------------------------------------------------------------------------- scans and stacks
def reduce_frames(
    obj: DataObject,
    how: str = "sum",
    roi: ROI | None = None,
    downsample: int | None = None,
) -> DataObject:
    """One image from the frames: their sum, mean, maximum or minimum (also std, median of a stack).

    ``roi`` limits the pixels and, with motor ranges, the frames; ``downsample``
    bins the pixels of a scan while it is read.
    """
    require(obj, ("scan", "stack"), "Image from Frames")
    if obj.kind == "scan":
        from bexa.core import reductions as acc

        if how not in SCAN_HOWS:
            raise ValueError(f"a scan gives {SCAN_HOWS}; load it as a stack for {how}")
        accumulator = {"sum": acc.Sum, "mean": acc.Mean, "max": acc.Max, "min": acc.Min}[how]()
        ds = None if not downsample else (1,) * len(obj.data.dims[:-2]) + (downsample,) * 2
        image = obj.data.reduce([accumulator], roi=roi, downsample=ds)[how]
        sigma = poisson(image) if how == "sum" and _counts(obj.data) and not downsample else None
        return _from_scan(obj, image, how, sigma=sigma, meta={"quantity": f"{how} of frames"})
    data, sigma = windowed(obj, roi)
    motors = other_dims(data)
    if how not in REDUCE_HOWS:
        raise ValueError(f"how must be one of {REDUCE_HOWS}")
    image = reduce_over(data, motors, how)
    n = int(np.prod([data.sizes[d] for d in motors])) if motors else 1
    return new(obj, image, how, sigma=sigma_over(sigma, motors, how, n), meta={"quantity": how})


def read_volume(
    obj: DataObject, roi: ROI | None = None, downsample: int | None = None
) -> DataObject:
    """The frames of a scan in memory as a stack ``(motors..., y, x)`` within the budget."""
    require(obj, ("scan",), "Load Frames")
    ds = None if not downsample else (1,) * len(obj.data.dims[:-2]) + (downsample,) * 2
    volume = obj.data.read(roi=roi, downsample=ds)
    return _from_scan(obj, volume, "frames", meta={"quantity": "frames"})


def preview(obj: DataObject, bin: int = 4, log: bool = False) -> DataObject:
    """The scan's frames binned ``bin`` x ``bin``: a light stack to browse (cached on disk)."""
    require(obj, ("scan",), "Preview")
    downsample = (1,) * len(obj.data.dims[:-2]) + (int(bin), int(bin))
    volume = obj.data.preview(downsample=downsample, apply_log=log)
    return _from_scan(obj, volume, "preview", meta={"quantity": "preview", "binning": int(bin)})


def roi_curves(
    obj: DataObject,
    rois: Mapping[str, ROI],
    quantity: str = "sum",
    background: ROI | None = None,
    normalize: str | None = None,
) -> list[DataObject]:
    """The intensity of each ROI at every point of the scan's grid, with uncertainties.

    A scan of one motor gives a curve per ROI (a rocking curve), a mesh a map.
    ``background`` is an ROI whose mean per pixel is subtracted from every ROI,
    frame by frame (scaled by the ROI's pixel count for ``sum``); ``normalize``
    names a per-frame channel (a monitor) to divide by.
    """
    require(obj, ("scan", "stack"), "ROI Curves")
    if not rois:
        raise ValueError("choose at least one ROI (draw one on the View tab)")
    if quantity not in ("sum", "mean", "max"):
        raise ValueError("quantity must be sum, mean or max")
    values: dict[str, xr.DataArray] = {}
    pixels: dict[str, int] = {}
    if obj.kind == "scan":
        from bexa.core.reductions import RoiIntegral

        everything = dict(rois)
        if background is not None:
            everything["__background__"] = background
        motor_roi = ROI({d: r for d, r in _motor_ranges(everything).items()})
        result = obj.data.reduce([RoiIntegral(everything, method=quantity)], roi=motor_roi)
        frame = obj.data.frame_shape
        for key, roi in everything.items():
            values[key] = result[f"roi_{key}"]
            y0, y1, x0, x1 = roi.pixel_window(frame)
            pixels[key] = (y1 - y0) * (x1 - x0)
        if background is not None:  # its mean per pixel, from the same pass
            bg_sum = values.pop("__background__")
            bg_pixels = pixels.pop("__background__")
            bg_mean = bg_sum / bg_pixels if quantity == "sum" else bg_sum
    else:
        data = obj.array
        pix = pixel_dims(data)
        for key, roi in {**rois, **({"__background__": background} if background else {})}.items():
            part = select_ranges(data, roi)
            values[key] = reduce_over(part, pix, quantity)
            pixels[key] = int(np.prod([part.sizes[d] for d in pix]))
        if background is not None:
            bg_sum = values.pop("__background__")
            bg_pixels = pixels.pop("__background__")
            bg_mean = bg_sum / bg_pixels if quantity == "sum" else bg_sum
    counts = obj.kind == "scan" and _counts(obj.data) and quantity == "sum"
    monitor = channel_on_grid(obj.data, normalize) if normalize and obj.kind == "scan" else None
    out = []
    for key, value in values.items():
        sigma = poisson(value) if counts else None
        if background is not None:
            scale = pixels[key] if quantity == "sum" else 1.0
            value = value - scale * bg_mean
            if counts:
                sigma = np.sqrt(sigma**2 + (scale / bg_pixels) ** 2 * np.abs(bg_sum))
        if monitor is not None:
            monitor_aligned = monitor.interp_like(value) if monitor.dims == value.dims else monitor
            value = value / monitor_aligned
            sigma = None if sigma is None else sigma / np.abs(monitor_aligned)
        value = value.rename(f"roi_{key}")
        meta = {"quantity": f"{quantity} in ROI {key}", "roi": rois[key].to_dict()}
        if normalize:
            meta["normalized_by"] = normalize
        out.append(new(obj, value, f"roi_{key}", sigma=sigma, meta=meta))
    return out


def _motor_ranges(rois: Mapping[str, ROI]) -> dict[str, tuple[Any, Any]]:
    """The motor ranges the ROIs agree on (pixel ranges stay with each ROI)."""
    ranges: dict[str, tuple[Any, Any]] = {}
    for roi in rois.values():
        for dim, rng in roi.ranges.items():
            if dim not in ("y", "x"):
                ranges.setdefault(dim, rng)
    return ranges


def com_maps(
    obj: DataObject,
    axes: Sequence[str] | None = None,
    sigma: float = 3.0,
    roi: ROI | None = None,
    downsample: int | None = None,
) -> list[DataObject]:
    """Centre-of-mass and width maps per motor axis, and the total intensity.

    The DFXM maps: for every pixel the intensity-weighted mean motor position
    (``com_<axis>``) and spread (``width_<axis>``). Frames are smoothed by
    ``sigma`` pixels first (scans; 0 to switch off).
    """
    require(obj, ("scan", "stack"), "COM Maps")
    if obj.kind == "scan":
        from bexa.core.reductions import MotorCOM

        ds = None if not downsample else (1,) * len(obj.data.dims[:-2]) + (downsample,) * 2
        result = obj.data.reduce(
            [MotorCOM(axes=list(axes) if axes else None, sigma=sigma)], roi=roi, downsample=ds
        )
        return [
            _from_scan(obj, result[name], name, meta={"quantity": name})
            for name in result.data_vars
        ]
    from bexa.analysis.rocking import moments

    data, _ = windowed(obj, roi)
    out = []
    for axis in axes or other_dims(data):
        others = [d for d in other_dims(data) if d != axis]
        summed = reduce_over(data, others, "sum").transpose(axis, *pixel_dims(data))
        stats = moments(summed.values, _coord_values(summed, axis), axis=0, sigma=sigma)
        template = summed.isel({axis: 0}, drop=True)
        for key, name in (("com", f"com_{axis}"), ("std", f"width_{axis}")):
            out.append(
                new(
                    obj,
                    template.copy(data=stats[key]).rename(name),
                    name,
                    meta={"quantity": name, "motor": axis},
                )
            )
    total = reduce_over(data, other_dims(data), "sum").rename("total")
    out.append(new(obj, total, "total", meta={"quantity": "total"}))
    return out


def _coord_values(da: xr.DataArray, dim: str) -> np.ndarray:
    return np.asarray(
        da.coords[dim].values if dim in da.coords else np.arange(da.sizes[dim]), float
    )


def frame_stats(obj: DataObject, roi: ROI | None = None) -> list[DataObject]:
    """Per-frame sum, mean, maximum and detector centre of mass, on the motor grid."""
    require(obj, ("scan",), "Frame Statistics")
    from bexa.core.reductions import FrameStats

    result = obj.data.reduce([FrameStats()], roi=roi)
    return [_from_scan(obj, result[n], n, meta={"quantity": n}) for n in result.data_vars]


def projections(obj: DataObject, roi: ROI | None = None) -> list[DataObject]:
    """Intensity projected on every pair of axes: the summed image, motor-pixel and motor maps."""
    require(obj, ("scan",), "Projections")
    from bexa.core.reductions import Projections

    result = obj.data.reduce([Projections()], roi=roi)
    return [_from_scan(obj, result[n], f"proj_{n}", meta={"quantity": n}) for n in result.data_vars]


def counters(obj: DataObject, channels: Sequence[str] | None = None) -> list[DataObject]:
    """Per-frame channels (counters, monitors, readbacks) on the scan's grid: curves or maps.

    Without ``channels`` every channel that is not a scanned motor is given.
    """
    require(obj, ("scan",), "Counters")
    s = obj.data.structure
    names = list(channels) if channels else [c for c in s.per_frame if c not in s.motor_dims]
    out = []
    for name in names:
        values = channel_on_grid(obj.data, name)
        out.append(_from_scan(obj, values, name, meta={"quantity": name}))
    return out


def point_table(obj: DataObject) -> list[DataObject]:
    """The scan's side tables (every SPEC row, for example), else its per-frame channels."""
    require(obj, ("scan",), "Point Table")
    tables = obj.data.source.aux_tables() if hasattr(obj.data.source, "aux_tables") else {}
    if not tables:
        s = obj.data.structure
        tables = {"points": pd.DataFrame({k: np.asarray(v) for k, v in s.per_frame.items()})}
    return [
        new(obj, table.reset_index(drop=True), name, meta=scan_meta(obj.data))
        for name, table in tables.items()
    ]


def histogram(
    obj: DataObject, bins: int = 256, low: float | None = None, high: float | None = None
) -> DataObject:
    """Histogram of the pixel values (of every frame of a scan)."""
    if obj.kind == "scan":
        from bexa.core.reductions import Histogram

        span = None if low is None or high is None else (float(low), float(high))
        counts = obj.data.reduce([Histogram(bins=int(bins), range=span)])["histogram"]
        return _from_scan(
            obj, counts, "histogram", sigma=poisson(counts), meta={"quantity": "pixels"}
        )
    require(obj, ANY, "Histogram")
    values = np.asarray(obj.array.values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    lo = float(values.min()) if low is None else float(low)
    hi = float(values.max()) if high is None else float(high)
    counts, edges = np.histogram(values, bins=int(bins), range=(lo, hi))
    data = curve(0.5 * (edges[1:] + edges[:-1]), counts, "value", "histogram")
    return new(obj, data, "histogram", sigma=np.sqrt(counts), meta={"quantity": "pixels"})


# --------------------------------------------------------------------------- arrays
def crop(obj: DataObject, roi: ROI) -> DataObject:
    """The part of an object inside an ROI (pixel ranges, and ranges of any other dim)."""
    require(obj, ANY, "Crop")
    data, sigma = windowed(obj, roi)
    return new(obj, data, "crop", sigma=sigma, meta={"roi": roi.to_dict()})


def select_range(
    obj: DataObject, dim: str, low: float | None = None, high: float | None = None
) -> DataObject:
    """Keep the points of ``dim`` between ``low`` and ``high`` (coordinate values, inclusive)."""
    require(obj, ANY, "Select Range")
    if dim not in obj.dims:
        raise ValueError(f"{obj.name} has no dim {dim!r}; its dims: {obj.dims}")
    roi = ROI({dim: (low, high)})
    roi.index_dims.discard(dim)
    if dim in ("y", "x"):  # pixel coordinates: inclusive here too
        roi.ranges[dim] = (low, None if high is None else high + 1e-9)
    data, sigma = windowed(obj, roi)
    return new(obj, data, f"{dim}range", sigma=sigma)


def take_slice(
    obj: DataObject, dim: str, index: int | None = None, value: float | None = None
) -> DataObject:
    """One slice of an object along ``dim`` (by index, or nearest to a coordinate value)."""
    require(obj, ANY, "Slice")
    if dim not in obj.dims:
        raise ValueError(f"{obj.name} has no dim {dim!r}; its dims: {obj.dims}")
    if value is not None:
        index = int(np.argmin(np.abs(obj.coord(dim).astype(float) - float(value))))
    if index is None:
        raise ValueError("give the index or the value of the slice")
    picked = obj.array.isel({dim: int(index)})
    sigma = None if obj.sigma is None else obj.sigma.isel({dim: int(index)})
    where = float(obj.coord(dim)[int(index)])
    return new(obj, picked, f"{dim}{int(index)}", sigma=sigma, meta={f"slice_{dim}": where})


def reduce_dims(obj: DataObject, dims: Sequence[str], how: str = "sum") -> DataObject:
    """Sum (or average, maximum, ...) an object over the named dims."""
    require(obj, ANY, "Reduce Dims")
    missing = [d for d in dims if d not in obj.dims]
    if missing or not dims:
        raise ValueError(f"choose dims of {obj.name} {obj.dims}; not {list(missing) or dims}")
    data = reduce_over(obj.array, dims, how)
    n = int(np.prod([obj.array.sizes[d] for d in dims]))
    sigma = sigma_over(obj.sigma, dims, how, n)
    return new(obj, data, f"{how}_{'_'.join(dims)}", sigma=sigma)


def profile(
    obj: DataObject, along: str = "x", roi: ROI | None = None, how: str = "sum"
) -> DataObject:
    """A curve along one dim, every other dim summed (or averaged) inside the ROI."""
    require(obj, ANY, "Profile")
    data, sigma = windowed(obj, roi)
    if along not in data.dims:
        raise ValueError(f"{obj.name} has no dim {along!r}; its dims: {obj.dims}")
    others = [d for d in data.dims if d != along]
    n = int(np.prod([data.sizes[d] for d in others])) if others else 1
    line = reduce_over(data, others, how)
    return new(obj, line, f"profile_{along}", sigma=sigma_over(sigma, others, how, n))


def line_profile(
    obj: DataObject,
    y0: float,
    x0: float,
    y1: float,
    x1: float,
    width: int = 1,
    points: int | None = None,
) -> DataObject:
    """The values along a straight line between two points, averaged over ``width`` pixels."""
    require(obj, IMAGES, "Line Profile")
    from scipy.ndimage import map_coordinates

    da = obj.array
    dy, dx = (str(d) for d in da.dims)
    yc, xc = obj.coord(dy).astype(float), obj.coord(dx).astype(float)
    iy0, ix0 = _fractional_index(yc, y0), _fractional_index(xc, x0)
    iy1, ix1 = _fractional_index(yc, y1), _fractional_index(xc, x1)
    length = float(np.hypot(iy1 - iy0, ix1 - ix0))
    n = int(points) if points else max(int(np.ceil(length)) + 1, 2)
    t = np.linspace(0.0, 1.0, n)
    ys, xs = iy0 + t * (iy1 - iy0), ix0 + t * (ix1 - ix0)
    normal = np.array([-(ix1 - ix0), iy1 - iy0]) / max(length, 1e-12)
    offsets = np.arange(int(width)) - (int(width) - 1) / 2.0
    values = np.mean(
        [
            map_coordinates(
                np.asarray(da.values, float),
                [ys + o * normal[1], xs + o * normal[0]],
                order=1,
                mode="nearest",
            )
            for o in offsets
        ],
        axis=0,
    )
    distance = np.hypot(
        np.interp(ys, np.arange(yc.size), yc) - y0, np.interp(xs, np.arange(xc.size), xc) - x0
    )
    data = curve(distance, values, "distance", "line_profile")
    return new(obj, data, "line", meta={"line": [y0, x0, y1, x1], "width": int(width)})


def _fractional_index(coords: np.ndarray, value: float) -> float:
    """Where ``value`` falls among increasing or decreasing ``coords``, in fractional indices."""
    index = np.arange(coords.size, dtype=float)
    if coords[-1] >= coords[0]:
        return float(np.interp(value, coords, index))
    return float(np.interp(value, coords[::-1], index[::-1]))


def roi_statistics(obj: DataObject, rois: Mapping[str, ROI]) -> DataObject:
    """Sum, mean, spread, extremes and centroid of each ROI (a stack: its summed frames)."""
    require(obj, ("image", "map", "stack"), "ROI Statistics")
    data = obj.array
    if obj.kind == "stack":
        data = reduce_over(data, other_dims(data), "sum")
    rows = []
    for name, roi in rois.items():
        part = select_ranges(data, roi)
        values = np.asarray(part.values, dtype=float)
        dy, dx = (str(d) for d in part.dims[-2:])
        yy = np.asarray(
            part.coords[dy].values if dy in part.coords else np.arange(part.sizes[dy]), float
        )
        xx = np.asarray(
            part.coords[dx].values if dx in part.coords else np.arange(part.sizes[dx]), float
        )
        total = np.nansum(values)
        rows.append(
            {
                "roi": name,
                "pixels": int(np.isfinite(values).sum()),
                "sum": total,
                "mean": np.nanmean(values),
                "std": np.nanstd(values),
                "min": np.nanmin(values),
                "max": np.nanmax(values),
                f"centroid_{dy}": float(np.nansum(np.nansum(values, axis=1) * yy) / total)
                if total
                else np.nan,
                f"centroid_{dx}": float(np.nansum(np.nansum(values, axis=0) * xx) / total)
                if total
                else np.nan,
            }
        )
    return new(obj, pd.DataFrame(rows), "roistats", kind="table")
