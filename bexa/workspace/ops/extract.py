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
    over: Sequence[str] | None = None,
    min_prominence: float = 0.0,
    widths: bool = True,
    total: bool = True,
    coms: str = "auto",
    picture_cmaps: str | Sequence[str] = "bwr, PiYG",
    picture_percentiles: Sequence[float] = (1.0, 99.0),
) -> list[DataObject]:
    """Centre-of-mass and width maps along any dims, and the total intensity.

    ``axes`` are the dims the COM is taken along, motors and pixels alike
    (default: every motor); ``over`` the dims of the maps (default: along two dims,
    the last two dims of the others, the rest summed; else the pixels for a COM
    along motors, the other motors summed: the DFXM maps; else every dim not in
    ``axes``). ``("y", "x")`` of an ``a x b`` scan gives every frame's centre on
    the detector as ``a x b`` maps. Results: ``com_<dim>`` (the intensity-weighted
    mean coordinate), ``width_<dim>`` (its spread) and ``total``. Frames are
    smoothed by ``sigma`` pixels first (0 to switch off). ``min_prominence`` leaves
    out (NaN) the points whose peak does not rise above the edges of the range (its
    highest value less its highest at either end of a dim along) by this fraction of
    the strongest point's rise: the background, whose COM is noise, and the peaks on
    an edge, whose COM the range cuts (0 keeps every point). ``widths`` and ``total``
    choose those maps (no widths: they
    are not computed). ``coms`` says how the COMs come out: ``"maps"``, one map each;
    ``"picture"``, along two dims, the two as one picture ``com_<a>_<b>`` (the first
    through the first of ``picture_cmaps``, the second through the second, averaged,
    each scaled over ``picture_percentiles``: :func:`bexa.viz.maps.bivariate_rgb`);
    ``"both"``; ``"auto"`` (the default), the picture along two dims and the maps
    otherwise. A picture needs maps over two dims; with fewer, the maps come out.
    """
    require(obj, ("scan", *ANY), "COM Maps")
    along, over = _com_dims(obj, axes, over)
    pixels = list(obj.dims[-2:]) if obj.kind == "scan" else list(pixel_dims(obj.array))
    if over != pixels or any(d in pixels for d in along) or obj.kind not in ("scan", "stack"):
        made = _com_any(obj, along, over, sigma, roi, downsample, min_prominence, widths)
    elif obj.kind == "scan":
        from bexa.core.reductions import MotorCOM

        ds = None if not downsample else (1,) * len(obj.data.dims[:-2]) + (downsample,) * 2
        com = MotorCOM(
            axes=along, sigma=sigma, moments=2 if widths else 1, min_prominence=min_prominence
        )
        result = obj.data.reduce([com], roi=roi, downsample=ds)
        made = [
            _from_scan(obj, result[name], name, meta={"quantity": name})
            for name in result.data_vars
        ]
    else:
        made = _com_stack(obj, along, sigma, roi, min_prominence, widths)
    if not total:
        made = [o for o in made if o.meta.get("quantity") != "total"]
    if coms not in COMS_AS:
        raise ValueError(f"coms is one of {COMS_AS}, not {coms!r}")
    picture = len(along) == 2 and len(over) == 2 and coms != "maps"
    if not picture:
        return made
    made.append(_bivariate(obj, made, along, picture_cmaps, picture_percentiles))
    if coms in ("auto", "picture"):  # the two COMs in the picture alone
        made = [o for o in made if o.meta.get("quantity") not in {f"com_{d}" for d in along}]
        made[-1].parents = [obj.name]
    return made


def _com_stack(
    obj: DataObject,
    along: list[str],
    sigma: float,
    roi: ROI | None,
    min_prominence: float,
    widths: bool,
) -> list[DataObject]:
    """:func:`com_maps` of a stack along its motors, one at a time, over its pixels; the
    rise of a pixel is its highest smoothed weight over the frames less its highest on the
    frames at either end of a motor along, as :class:`~bexa.core.reductions.MotorCOM`
    measures it."""
    from bexa.analysis.rocking import moments
    from bexa.core.backend import gaussian_maxima
    from bexa.core.reductions import prominent

    data, _ = windowed(obj, roi)
    pixels = pixel_dims(data)
    keep = None
    if min_prominence:  # of the clipped weights
        motors = other_dims(data)
        frames = np.asarray(data.transpose(*motors, *pixels).values, np.float32)
        edge = np.zeros([data.sizes[d] for d in motors], dtype=bool)
        for axis in along:  # the frames at either end of a motor along
            ends: list[Any] = [slice(None)] * len(motors)
            ends[motors.index(axis)] = [0, -1]
            edge[tuple(ends)] = True
        high, rim = gaussian_maxima(frames.reshape(-1, *frames.shape[-2:]), sigma, edge.ravel())
        keep = prominent(np.clip(high, 1e-10, None) - np.clip(rim, 1e-10, None), min_prominence)
    linear = _linear(data)
    out = []
    keys = (("com", "com"), ("std", "width")) if widths else (("com", "com"),)
    for axis in along:
        if linear:  # the weights summed over the other motors, then smoothed: the same sums
            sums = _linear_com_sums(data, [axis], list(pixels), sigma, floor_dims=[axis])
            mean1 = sums[(axis, 1)] / sums[None]
            width = (sums[(axis, 2)] / sums[None] - mean1**2).clip(min=0) ** 0.5
            middle = float(_coord_values(data, axis).mean())
            stats = {"com": (mean1 + middle).values, "std": width.values}
            template = sums[None]
        else:
            others = [d for d in other_dims(data) if d != axis]
            summed = reduce_over(data, others, "sum").transpose(axis, *pixels)
            stats = moments(summed.values, _coord_values(summed, axis), axis=0, sigma=sigma)
            template = summed.isel({axis: 0}, drop=True)
        for key, prefix in keys:
            name = f"{prefix}_{axis}"
            values = stats[key] if keep is None else np.where(keep, stats[key], np.nan)
            out.append(
                new(
                    obj,
                    template.copy(data=values).rename(name),
                    name,
                    meta={"quantity": name, "motor": axis},
                )
            )
    if linear:  # no NaN to skip
        total = data.sum(other_dims(data), skipna=False, keep_attrs=True).rename("total")
    else:
        total = reduce_over(data, other_dims(data), "sum").rename("total")
    out.append(new(obj, total, "total", meta={"quantity": "total"}))
    return out


def _bivariate(
    obj: DataObject,
    made: list[DataObject],
    along: list[str],
    cmaps: str | Sequence[str],
    percentiles: Sequence[float],
) -> DataObject:
    """The two COM maps of ``made`` as one RGB picture, its key in the attrs."""
    from bexa.viz.maps import bivariate_rgb

    names = [c.strip() for c in cmaps.split(",")] if isinstance(cmaps, str) else list(cmaps)
    if len(names) != 2 or len(percentiles) != 2:
        raise ValueError("give two picture colour maps and two percentiles (from, to)")
    by_name = {o.meta.get("quantity"): o for o in made}
    first, second = (by_name[f"com_{d}"].array for d in along)
    limits = []
    for values in (first.values, second.values):
        finite = values[np.isfinite(values)]
        lo, hi = np.percentile(finite, sorted(percentiles)) if finite.size else (0.0, 1.0)
        limits.append((float(lo), float(hi)))
    rgb, lim_a, lim_b = bivariate_rgb(first, second, (names[0], names[1]), (limits[0], limits[1]))
    dims = tuple(str(d) for d in first.dims)
    name = f"com_{along[0]}_{along[1]}"
    key = {
        "key_x": str(first.name),
        "key_x_range": list(lim_a),
        "key_y": str(second.name),
        "key_y_range": list(lim_b),
        "key_cmaps": names,
    }
    picture = xr.DataArray(
        rgb,
        dims=(*dims, "rgb"),
        coords={**{d: first.coords[d] for d in dims if d in first.coords}, "rgb": list("rgb")},
        name=name,
        attrs=key,
    )
    parents = [by_name[f"com_{d}"].name for d in along]
    return new(obj, picture, name, kind="rgb", parents=parents, meta={"quantity": name})


COMS_AS = ("auto", "picture", "maps", "both")  # how COM Maps gives its COMs


def _com_dims(
    obj: DataObject, axes: Sequence[str] | None, over: Sequence[str] | None
) -> tuple[list[str], list[str]]:
    """The dims of :func:`com_maps`: those the COM is along and those of the maps."""
    dims = [str(d) for d in obj.dims]
    pixels = dims[-2:] if obj.kind == "scan" else list(pixel_dims(obj.array))
    motors = [d for d in dims if d not in pixels]
    along = [str(d) for d in axes] if axes else motors
    if over:
        over = [str(d) for d in over]
    elif len(along) == 2 and len(dims) > 3:  # one picture over two dims, the rest summed
        over = [d for d in dims if d not in along][-2:]
    elif pixels and all(d in motors for d in along):  # the DFXM maps: the other motors summed
        over = pixels
    else:
        over = [d for d in dims if d not in along]
    unknown = [d for d in (*along, *over) if d not in dims]
    if unknown:
        raise ValueError(f"{obj.name} has no dim {unknown}; its dims: {dims}")
    if not along or not over or set(along) & set(over):
        raise ValueError(
            f"take the COM along some dims of {obj.name} {dims} and keep others for the maps"
        )
    return along, over


def _com_any(
    obj: DataObject,
    along: list[str],
    over: list[str],
    sigma: float,
    roi: ROI | None,
    downsample: int | None,
    min_prominence: float = 0.0,
    widths: bool = True,
) -> list[DataObject]:
    """:func:`com_maps` along any dims: a scan streamed (``DimCOM``), an array in memory."""
    if obj.kind == "scan":
        from bexa.core.reductions import DimCOM

        ds = None if not downsample else (1,) * len(obj.data.dims[:-2]) + (downsample,) * 2
        com = DimCOM(
            along, over, sigma=sigma, moments=2 if widths else 1, min_prominence=min_prominence
        )
        result = obj.data.reduce([com], roi=roi, downsample=ds)
        maps = {str(name): result[name] for name in result.data_vars}
    else:
        data, _ = windowed(obj, roi)
        maps = _com_arrays(data, along, over, sigma, min_prominence)
    kinds = ("com", "width") if widths else ("com",)
    names = [f"{k}_{d}" for d in along for k in kinds] + ["total"]
    meta = {"along": along, "over": over}
    return [
        _from_scan(obj, maps[n], n, meta={"quantity": n, **meta})
        if obj.kind == "scan"
        else new(obj, maps[n], n, meta={"quantity": n, **meta})
        for n in names
    ]


def _com_arrays(
    data: xr.DataArray,
    along: list[str],
    over: list[str],
    sigma: float,
    min_prominence: float = 0.0,
) -> dict[str, xr.DataArray]:
    """``com_<dim>``, ``width_<dim>`` and ``total`` of an array: the weights are the values,
    smoothed over the pixel dims by ``sigma`` and clipped to 1e-10, the other dims summed;
    ``min_prominence`` leaves out the points whose weights rise least above the edges
    (NaN), the rise measured as :class:`~bexa.core.reductions.DimCOM` does: the pixel
    dims neither along nor over summed, the highest weight over the rest less its highest
    at either end of a dim along. Values without a negative one or a NaN are not
    smoothed themselves (:func:`_linear_com_sums`)."""
    pixels = pixel_dims(data)
    rest = [d for d in pixels if d not in along and d not in over]  # summed before the rise
    rise = None
    if _linear(data):
        summed = [d for d in data.dims if d not in over]
        sums = _linear_com_sums(data, along, over, sigma, floor_dims=summed)
        if min_prominence:
            v = _smooth_pixels(data.sum(rest, skipna=False) if rest else data, pixels, sigma)
            rise = _rise_over_edges(v, along, over)
    else:
        if sigma and pixels:
            from bexa.core.backend import gaussian_frames

            flat = np.asarray(data.values, float).reshape(-1, *data.shape[-2:])
            data = data.copy(data=gaussian_frames(flat, sigma).reshape(data.shape))
        clipped = data.clip(min=1e-10)
        w = clipped.sum([d for d in data.dims if d not in along and d not in over])
        sums = {None: w.sum(along)}
        for d in along:
            c = xr.DataArray(_coord_values(w, d), dims=(d,))
            c = c - float(c.mean())  # about the middle: the second moment keeps its precision
            sums.update({(d, 1): (w * c).sum(along), (d, 2): (w * c**2).sum(along)})
        if min_prominence:
            rise = _rise_over_edges(clipped.sum(rest) if rest else clipped, along, over)
    total = sums[None]
    out = {"total": total.transpose(*over).rename("total")}
    keep = None
    if min_prominence:
        from bexa.core.reductions import prominent

        keep = prominent(rise, min_prominence)
    for d in along:
        mean1 = sums[(d, 1)] / total
        width = (sums[(d, 2)] / total - mean1**2).clip(min=0) ** 0.5
        middle = float(_coord_values(data, d).mean())
        if keep is not None:
            mean1, width = mean1.where(keep), width.where(keep)
        out[f"com_{d}"] = (mean1 + middle).transpose(*over).rename(f"com_{d}")
        out[f"width_{d}"] = width.transpose(*over).rename(f"width_{d}")
    return out


def _rise_over_edges(v: xr.DataArray, along: Sequence[str], over: Sequence[str]) -> xr.DataArray:
    """The rise of every point of the maps over ``over``: the highest of the weights ``v``
    over the other dims less their highest at either end of a dim of ``along`` (0 for a
    peak on an edge)."""
    spread = [d for d in v.dims if d not in over]
    rim = None
    for d in along:
        face = v.isel({d: [0, -1]}).max(spread)
        rim = face if rim is None else np.fmax(rim, face)
    return v.max(spread) - rim


def _linear(data: xr.DataArray) -> bool:
    """Whether the weight sums of ``data`` come from its values as they are: none that the
    clip changes (a negative one) or the sums skip (a NaN)."""
    return bool(data.size) and bool(np.min(data.values) >= 0)


def _linear_com_sums(
    data: xr.DataArray,
    along: Sequence[str],
    over: Sequence[str],
    sigma: float,
    floor_dims: Sequence[str],
) -> dict[Any, xr.DataArray]:
    """The weight sums of a COM of ``data`` along ``along``, as maps over ``over`` (the
    other dims summed): ``None`` the weights, ``(d, k)`` the weights times the coordinate
    of ``d`` about its middle to the power ``k`` (1, 2). The weights are the values
    smoothed over the pixel dims by ``sigma``, with a floor of 1e-10 under each value of
    ``floor_dims``, for values without a negative one or a NaN.

    The values are not smoothed: scipy's Gaussian (reflect mode) is symmetric and keeps
    sums, so the coordinates of a pixel dim summed are smoothed instead, and the pixel
    dims of the maps at the end. Each sum is one pass over the values (``einsum``).
    """
    from scipy.ndimage import gaussian_filter1d

    pixels = pixel_dims(data)
    summed = [d for d in data.dims if d not in over]
    letters = dict(zip(data.dims, "abcdefghijklmnopqrstuvwxyz", strict=False))
    kept = [d for d in data.dims if d not in summed]
    to = "".join(letters[d] for d in kept)
    coords = {d: data.coords[d] for d in kept if d in data.coords}
    floor = 1e-10 * float(np.prod([data.sizes[d] for d in floor_dims]))

    def weighted(dim: str | None, weight: Any = None) -> xr.DataArray:
        spec = "".join(letters[d] for d in data.dims)
        operands = [data.values]
        if dim is not None:
            spec += "," + letters[dim]
            operands.append(weight)
        values = np.einsum(f"{spec}->{to}", *operands, dtype=np.float64, casting="same_kind")
        return _smooth_pixels(xr.DataArray(values, dims=kept, coords=coords), pixels, sigma)

    sums = {None: weighted(None) + floor}
    for d in along:  # each among floor_dims
        c = _coord_values(data, d)
        c = c - c.mean()  # about the middle: the second moment keeps its precision
        for k in (1, 2):
            weight = gaussian_filter1d(c**k, sigma) if d in pixels and sigma else c**k
            sums[(d, k)] = weighted(d, weight) + floor * float(np.mean(c**k))
    return sums


def _smooth_pixels(da: xr.DataArray, pixels: Sequence[str], sigma: float) -> xr.DataArray:
    """``da`` smoothed by ``sigma`` along those of ``pixels`` it has, as its frames would be;
    both, the last two dims, over the compute threads."""
    kept = [d for d in pixels if d in da.dims]
    if not kept or not sigma:
        return da
    values = np.asarray(da.values, float)
    if len(kept) == 2:
        from bexa.core.backend import gaussian_frames

        flat = values.reshape(-1, *values.shape[-2:])
        return da.copy(data=gaussian_frames(flat, sigma).reshape(values.shape))
    from scipy.ndimage import gaussian_filter1d

    return da.copy(data=gaussian_filter1d(values, sigma, axis=da.get_axis_num(kept[0])))


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


def counters(
    obj: DataObject,
    channels: Sequence[str] | None = None,
    x: str | None = None,
    monitor: str | None = None,
    monitor_dark: float = 0.0,
) -> list[DataObject]:
    """Per-point channels (counters, monitors, readbacks) as curves or maps, without a frame read.

    A scan of one motor whose source keeps every recorded point (a SPEC scan,
    whose rows without a saved frame count too) gives curves against ``x``: its
    motor by default (``Time`` for a loop scan), or any recorded channel. Other
    scans give their per-frame channels on the motor grid (``x`` is refused
    there: the motors are the axes). Without ``channels``
    every channel but the scanned motors. ``monitor`` divides the other channels
    by a monitor channel less ``monitor_dark`` (its counts per second without
    beam) times the count time (``Seconds``, ``sec``...); points without beam
    become NaN.
    """
    require(obj, ("scan",), "Counters")
    scan = obj.data
    s = scan.structure
    rows = _point_rows(scan)
    if rows is not None and x is None:  # the motor; a loop scan's "frame" dim is no column
        recorded = [d for d in s.motor_dims if d in rows and d not in BOOKKEEPING]
        x = recorded[0] if recorded else ("Time" if "Time" in rows else "point")
    if rows is not None and x in rows:
        return _counter_curves(obj, rows, x, channels, monitor, monitor_dark)
    if x is not None:  # on the grid of a scan the motors are the axes: nothing to choose
        if rows is not None:
            raise ValueError(
                f"Against {x!r}: no such column; the scan records {list(rows.columns)}"
            )
        raise ValueError(
            f"Against {x!r}: only a scan of one motor with every point recorded (a SPEC scan) "
            "takes another axis; this scan gives its channels on its motor grid"
        )
    names = list(channels) if channels else [c for c in s.per_frame if c not in s.motor_dims]
    beam = None
    if monitor:
        beam = _beam(lambda c: channel_on_grid(scan, c), list(s.per_frame), monitor, monitor_dark)
    out = []
    for name in names:
        values = channel_on_grid(scan, name)
        meta: dict[str, Any] = {"quantity": name}
        if beam is not None and name != monitor:
            values = (values / beam).rename(name)
            meta.update(normalized_by=monitor, monitor_dark=float(monitor_dark))
        out.append(_from_scan(obj, values, name, meta=meta))
    return out


COUNT_TIMES = ("Seconds", "sec", "count_time", "acq_time", "exposure_time", "integration_time")
BOOKKEEPING = ("point", "frame")  # columns of a point table that number the rows


def _point_rows(scan: Any) -> pd.DataFrame | None:
    """Every point a scan of one motor recorded, frames or not (a SPEC scan), else ``None``."""
    table_of = getattr(scan.source, "point_table", None)
    if table_of is None or len(scan.structure.motor_dims) > 1:
        return None
    table = table_of()
    return table if isinstance(table, pd.DataFrame) and len(table) else None


def _beam(get: Any, names: Sequence[str], monitor: str, dark: float) -> Any:
    """``get(monitor)`` less ``dark`` counts per second times the count time; no beam is NaN."""
    if monitor not in names:
        raise ValueError(f"no channel {monitor!r} to divide by; the scan records {sorted(names)}")
    beam = get(monitor)
    if dark:
        time_name = next((n for n in COUNT_TIMES if n in names), None)
        if time_name is None:
            raise ValueError(
                f"a monitor dark rate needs the count time; the scan records none of {COUNT_TIMES}"
            )
        beam = beam - float(dark) * get(time_name)
    if isinstance(beam, xr.DataArray):
        return beam.where(beam > 0)
    return np.where(beam > 0, beam, np.nan)


def _counter_curves(
    obj: DataObject,
    rows: pd.DataFrame,
    x: str,
    channels: Sequence[str] | None,
    monitor: str | None,
    dark: float,
) -> list[DataObject]:
    """One curve per channel against ``x``, from every recorded point."""
    motors = set(obj.data.structure.motor_dims)
    skip = {*BOOKKEEPING, x, *motors}
    names = list(channels) if channels else [c for c in rows.columns if c not in skip]
    missing = [c for c in names if c not in rows]
    if missing:
        raise ValueError(f"no channel {missing}; the scan records {list(rows.columns)}")
    xs = rows[x].to_numpy(dtype=float)
    keep = np.isfinite(xs)
    beam = None
    if monitor:
        beam = _beam(lambda c: rows[c].to_numpy(dtype=float), list(rows.columns), monitor, dark)
    out = []
    for name in names:
        values = rows[name].to_numpy(dtype=float)
        meta: dict[str, Any] = {"quantity": name, "points": int(keep.sum())}
        if beam is not None and name != monitor:
            values = values / beam
            meta.update(normalized_by=monitor, monitor_dark=float(dark))
        data = curve(xs[keep], values[keep], x, name)
        out.append(_from_scan(obj, data, name, meta=meta))
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


def reduce_dims(
    obj: DataObject,
    dims: Sequence[str],
    how: str = "sum",
    roi: ROI | None = None,
    downsample: int | None = None,
) -> DataObject:
    """Sum (or average, maximum, ...) an object over any of its dims, inside an ROI.

    A scan is streamed (its pixels and motors alike: ``("mu", "y", "x")`` of a mosa
    scan gives a curve over chi); std and median read its frames into memory first.
    The ROI's ranges cut the dims it names; ``downsample`` bins a scan's pixels.
    """
    require(obj, ("scan", *ANY), "Reduce Dims")
    dims = [str(d) for d in dims or ()]
    missing = [d for d in dims if d not in obj.dims]
    if missing or not dims:
        raise ValueError(f"choose dims of {obj.name} {obj.dims}; not {list(missing) or dims}")
    if len(dims) == len(obj.dims):
        raise ValueError("keep at least one dim (ROI Statistics gives one number per ROI)")
    suffix = f"{how}_{'_'.join(dims)}"
    meta = {"roi": roi.to_dict()} if roi else {}
    if obj.kind == "scan":
        return _reduce_scan(obj, dims, how, roi, downsample, suffix, meta)
    data, sigma = windowed(obj, roi)
    n = int(np.prod([data.sizes[d] for d in dims]))
    reduced = reduce_over(data, dims, how)
    return new(obj, reduced, suffix, sigma=sigma_over(sigma, dims, how, n), meta=meta)


def _reduce_scan(
    obj: DataObject,
    dims: list[str],
    how: str,
    roi: ROI | None,
    downsample: int | None,
    suffix: str,
    meta: dict[str, Any],
) -> DataObject:
    """:func:`reduce_dims` of a scan: one streaming pass, or its frames in memory."""
    scan = obj.data
    ds = None if not downsample else (1,) * len(scan.dims[:-2]) + (downsample,) * 2
    if how not in SCAN_HOWS:  # std, median: not a running total
        if how not in REDUCE_HOWS:
            raise ValueError(f"how must be one of {REDUCE_HOWS}, not {how!r}")
        reduced = reduce_over(scan.read(roi=roi, downsample=ds), dims, how)
        return _from_scan(obj, reduced, suffix, meta=meta)
    from bexa.core.reductions import ReduceDims

    result = scan.reduce([ReduceDims(dims, how)], roi=roi, downsample=ds)
    reduced, count = result["reduce_dims"], result["reduce_dims_count"]
    sigma = None
    if how in ("sum", "mean") and _counts(scan) and not downsample:  # Poisson, from the counts
        sigma = poisson(reduced) if how == "sum" else np.sqrt(np.abs(reduced) / count)
    return _from_scan(obj, reduced, suffix, sigma=sigma, meta={"quantity": how, **meta})


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
