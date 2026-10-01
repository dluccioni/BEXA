"""Helpers for stacked volumes: per-frame statistics, the brightest condition, colour limits.

A volume stacked by :func:`bexa.stack` is ``(outer..., motors..., y, x)`` and may
live on disk (:func:`bexa.io.cube.open_lazy`). These functions never load more
than a sample of its frames: they use the per-frame statistics the stack
carries in its attrs (``block_total``, ``block_p1``, ``block_p99``, computed
scan by scan while stacking), or stream over the frames when the attrs are
missing or no longer match the array's dims (after a ``transpose``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr

from bexa.core.backend import to_host

__all__ = ["block_stats", "brightest", "is_lazy", "shared_limits", "total_volume"]

IN_MEMORY_LIMIT = 256 * 1024**2  # arrays up to this size are simply loaded


def _as_array(data: Any) -> np.ndarray:
    """A numpy array from a DataArray (loading a lazy one), a cupy array or an array-like."""
    if hasattr(data, "values") and hasattr(data, "dims"):
        data = data.values
    return np.asarray(to_host(data))


def _percentile_limits(data: Any, low: float = 1.0, high: float = 99.0) -> tuple[float, float]:
    """``(low, high)`` percentiles of the finite values; ``(0, 1)`` when there are none."""
    arr = _as_array(data)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(finite, [low, high])
    if vmax <= vmin:
        vmax = vmin + 1e-12
    return float(vmin), float(vmax)


def is_lazy(array: Any) -> bool:
    """True for a DataArray that reads from a file on demand (:func:`bexa.io.cube.open_lazy`)."""
    variable = getattr(array, "variable", None)
    return variable is not None and not getattr(variable, "_in_memory", True)


def _stats_from_attrs(volume: xr.DataArray) -> dict[str, np.ndarray] | None:
    """The per-frame statistics carried by the array, if they still match its dims."""
    attrs = volume.attrs
    stored = [str(d) for d in attrs.get("block_stats_dims", [])]
    current = [str(d) for d in volume.dims[:-2]]
    if not stored or sorted(stored) != sorted(current):
        return None
    try:
        stats = {k: np.asarray(attrs[f"block_{k}"]) for k in ("total", "p1", "p99")}
    except KeyError:
        return None
    order = [stored.index(d) for d in current]  # a transposed volume keeps its statistics
    stats = {k: np.transpose(v, order) for k, v in stats.items()}
    if any(s.shape != tuple(volume.shape[:-2]) for s in stats.values()):
        return None
    return stats


def block_stats(volume: xr.DataArray, max_frames: int | None = None) -> dict[str, np.ndarray]:
    """Per-frame total and 1st/99th percentiles over the last two dims of ``volume``.

    From the attrs when the stack carries them; otherwise computed frame by
    frame, reading a lazy array one frame at a time. With ``max_frames`` only
    that many frames, spread evenly over the leading dims, are visited and the
    others stay NaN (enough for colour limits, not for :func:`brightest`).
    """
    stats = _stats_from_attrs(volume)
    if stats is not None:
        return stats
    lead = tuple(volume.shape[:-2])
    n = int(np.prod(lead)) if lead else 1
    out = {k: np.full(lead, np.nan) for k in ("total", "p1", "p99")}
    if not is_lazy(volume) and volume.nbytes <= IN_MEMORY_LIMIT:
        data = _as_array(volume).reshape(*lead, -1) if lead else _as_array(volume).reshape(1, -1)
        with np.errstate(all="ignore"):
            out["total"] = np.nansum(data, axis=-1, dtype=np.float64).reshape(lead)
            p1, p99 = np.nanpercentile(data, [1, 99], axis=-1)
        out["p1"], out["p99"] = np.asarray(p1).reshape(lead), np.asarray(p99).reshape(lead)
        out["total"][np.all(np.isnan(data), axis=-1).reshape(lead)] = np.nan
        return out
    chosen = np.arange(n)
    if max_frames is not None and n > max_frames:
        chosen = np.linspace(0, n - 1, max_frames).round().astype(int)
    for flat in chosen:
        position = np.unravel_index(int(flat), lead) if lead else ()
        frame = _as_array(volume.isel(dict(zip(volume.dims[:-2], position, strict=True))))
        finite = frame[np.isfinite(frame)]
        if finite.size:
            out["total"][position] = float(finite.sum(dtype=np.float64))
            out["p1"][position], out["p99"][position] = np.percentile(finite, [1, 99])
    return out


def total_volume(
    volume: xr.DataArray,
    keep: str | Sequence[str] | None = None,
    downsample: int | Sequence[int] | None = None,
    method: str = "mean",
) -> xr.DataArray:
    """The intensity summed over every leading dim but ``keep``, read one block at a time.

    ``stacked["sum"]`` of a z-stack, ``(z, energy, y, x)`` say, gives with
    ``keep="z"`` the total scattered intensity per voxel ``(z, y, x)``,
    whatever was scanned; ``stacked["preview"]`` summed over the scan motors
    gives the same from the binned frames. A lazy stack on disk is read one
    kept position at a time, so the memory is one such block. ``downsample``
    bins the pixels of the result (an int or ``(by, bx)``, block ``method``
    ``"mean"``, ``"sum"`` or ``"max"``) with the full-resolution index of each
    block's first pixel as coordinates, so a volume of summed images can take
    the voxel size of the previews. Missing scans (NaN frames) count as zero.
    The result is in memory, with attrs ``summed_over`` and ``binning``.
    """
    from bexa.core.reductions import block_reduce

    lead = [str(d) for d in volume.dims[:-2]]
    kept = (
        [lead[0]]
        if keep is None and lead
        else [keep]
        if isinstance(keep, str)
        else list(keep or [])
    )
    unknown = [d for d in kept if d not in lead]
    if unknown:
        raise ValueError(f"{unknown} are not leading dims of the volume {tuple(volume.dims)}")
    kept = [d for d in lead if d in kept]  # in the volume's order
    summed = [d for d in lead if d not in kept]
    if isinstance(downsample, (int, np.integer)):
        by = bx = int(downsample)
    elif downsample is not None:
        by, bx = (int(v) for v in downsample)
    else:
        by = bx = 1
    py, px = (str(d) for d in volume.dims[-2:])
    ny, nx = volume.shape[-2] // by, volume.shape[-1] // bx
    kept_shape = tuple(volume.sizes[d] for d in kept)
    out = np.zeros((*kept_shape, ny, nx), dtype=np.float64)
    for flat in range(int(np.prod(kept_shape)) if kept else 1):
        position = np.unravel_index(flat, kept_shape) if kept else ()
        block = _as_array(volume.isel(dict(zip(kept, position, strict=True))))
        with np.errstate(all="ignore"):
            image = np.nansum(block.reshape(-1, *block.shape[-2:]), axis=0, dtype=np.float64)
        out[position] = block_reduce(image, by, bx, method)
    coords: dict[str, Any] = {d: volume.coords[d].values for d in kept if d in volume.coords}
    for name, n, factor, size in ((py, ny, by, volume.shape[-2]), (px, nx, bx, volume.shape[-1])):
        start = float(volume.coords[name].values[0]) if name in volume.coords else 0.0
        step = (
            float(volume.coords[name].values[1] - volume.coords[name].values[0])
            if name in volume.coords and size > 1
            else 1.0
        )
        coords[name] = start + step * factor * np.arange(n)
    return xr.DataArray(
        out,
        dims=(*kept, py, px),
        coords=coords,
        name=f"total_{volume.name}" if volume.name else "total",
        attrs={"summed_over": summed, "binning": [by, bx], "method": method},
    )


def brightest(volume: Any, over: Any = None, name: str | None = None) -> dict[str, int]:
    """The position of the frame with the largest total intensity, as ``{dim: index}``.

    ``volume`` is a stacked DataArray (``(outer..., y, x)``) or the Dataset a
    stack returned, with ``name`` picking the variable (default: the first).
    ``over`` lists the dims to report (default: every dim but the last two);
    totals are summed over the others first, so with ``over=vol.dims[:-3]`` the
    answer is the condition that lights up the whole ``(z, y, x)`` block, ready
    for ``browser.update(**peak)``. Uses the per-frame totals the stack carries;
    a plain array is summed frame by frame.
    """
    if isinstance(volume, xr.Dataset):
        name = name or next(
            str(n)
            for n in volume.data_vars
            if not str(n).endswith(("_block_total", "_block_p1", "_block_p99"))
        )
        volume = volume[name]
    lead = [str(d) for d in volume.dims[:-2]]
    totals = np.nan_to_num(block_stats(volume)["total"], nan=-np.inf)
    report = lead if over is None else [str(d) for d in over]
    unknown = [d for d in report if d not in lead]
    if unknown:
        raise ValueError(f"{unknown} are not leading dims of the volume {tuple(volume.dims)}")
    summed = totals
    for axis in sorted((lead.index(d) for d in lead if d not in report), reverse=True):
        summed = np.where(np.isfinite(summed), summed, 0).sum(axis=axis)
    kept = [d for d in lead if d in report]
    position = np.unravel_index(int(np.argmax(summed)), summed.shape) if summed.ndim else ()
    return {d: int(i) for d, i in zip(kept, position, strict=True)}


def shared_limits(
    volume: xr.DataArray, low: float = 1.0, high: float = 99.0, log: bool = False
) -> tuple[float, float]:
    """Colour limits shared by every frame of a volume, without loading all of it.

    The stack's per-frame percentiles give ``(min of the 1st, max of the 99th)``;
    a small in-memory array gets the exact percentiles; a large or lazy one the
    same from a sample of 64 frames. ``log`` applies ``log10(1 + I)`` to the
    limits, as the browsers show the data.
    """
    stats = _stats_from_attrs(volume)
    if stats is None and not is_lazy(volume) and volume.nbytes <= IN_MEMORY_LIMIT:
        data = _as_array(volume)
        if log:
            data = np.log10(1.0 + np.clip(data, 0, None))
        return _percentile_limits(data, low, high)
    if stats is None:
        stats = block_stats(volume, max_frames=64)
    p1, p99 = stats["p1"], stats["p99"]
    finite = np.isfinite(p1) & np.isfinite(p99)
    if not finite.any():
        return 0.0, 1.0
    vmin, vmax = float(np.min(p1[finite])), float(np.max(p99[finite]))
    if log:
        vmin, vmax = float(np.log10(1.0 + max(vmin, 0.0))), float(np.log10(1.0 + max(vmax, 0.0)))
    if vmax <= vmin:
        vmax = vmin + 1e-12
    return vmin, vmax
