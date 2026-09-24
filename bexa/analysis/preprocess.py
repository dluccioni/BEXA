"""Frame preprocessing: darks, hot pixels, clipping, normalisation, cropping, binning.

Every function takes a frame stack ``(n, y, x)`` (numpy or cupy) and returns
the same kind of array. Nothing here reads files or touches the structure.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from bexa.core.backend import array_module, ndimage_module
from bexa.core.roi import ROI

__all__ = [
    "bin_spatial",
    "box_mean",
    "clip_nonpositive",
    "crop",
    "log_scale",
    "normalize_box",
    "normalize_i0",
    "remove_hot_pixels",
    "subtract_dark",
    "threshold_floor",
]

BoxLike = ROI | tuple[tuple[int, int], tuple[int, int]] | dict[str, Any]


def subtract_dark(frames: Any, dark: Any, mode: str = "mean") -> Any:
    """Subtract a dark frame; a dark *stack* is first reduced by ``mode`` (mean or median)."""
    xp = array_module(frames)
    dark = xp.asarray(dark)
    if dark.ndim == frames.ndim:
        dark = dark.mean(axis=0) if mode == "mean" else xp.median(dark, axis=0)
    return frames - dark


def remove_hot_pixels(frames: Any, size: int | Sequence[int] = (1, 3, 3)) -> Any:
    """Median filter over the stack (``size`` per axis; the default keeps frames separate)."""
    ndi = ndimage_module(frames)
    return ndi.median_filter(frames, size=size)


def threshold_floor(frames: Any, percentile: float = 1.0) -> Any:
    """Raise every value below the ``percentile``-th percentile of the stack to that level."""
    xp = array_module(frames)
    floor = xp.percentile(frames, percentile)
    return xp.maximum(frames, floor)


def log_scale(frames: Any) -> Any:
    """``log10(1 + I)`` with negative values clipped to zero (the v9 preview scaling)."""
    xp = array_module(frames)
    return xp.log10(1.0 + xp.clip(frames, 0, None))


def bin_spatial(frames: Any, factors: int | Sequence[int], method: str = "mean") -> Any:
    """Block-reduce the last two axes by integer factors (``bexa.core.reductions.block_reduce``)."""
    from bexa.core.reductions import block_reduce

    dy, dx = (factors, factors) if isinstance(factors, int) else (int(factors[0]), int(factors[1]))
    return block_reduce(frames, dy, dx, method)


def clip_nonpositive(frames: Any, eps: float = 1e-4, copy: bool = True) -> Any:
    """Replace values ``<= 0`` by ``eps`` (the legacy cube builder's ``raw[raw <= 0] = 1e-4``)."""
    out = frames.copy() if copy else frames
    out[out <= 0] = eps
    return out


def normalize_i0(frames: Any, i0: Any) -> Any:
    """Divide every frame by its I0; frames with ``I0 == 0`` become NaN."""
    xp = array_module(frames)
    i0 = xp.asarray(i0, dtype=np.float64)
    safe = xp.where(i0 == 0, xp.nan, i0)
    return frames / safe[:, None, None]


def _box_slices(box: BoxLike, frame_shape: tuple[int, int]) -> tuple[slice, slice]:
    if isinstance(box, ROI):
        return box.pixel_slices(frame_shape)
    if isinstance(box, dict):
        return ROI(y=tuple(box["y"]), x=tuple(box["x"])).pixel_slices(frame_shape)
    (y0, y1), (x0, x1) = box
    return ROI(y=(y0, y1), x=(x0, x1)).pixel_slices(frame_shape)


def crop(frames: Any, roi: BoxLike) -> Any:
    """Cut the last two axes to the ROI."""
    ys, xs = _box_slices(roi, frames.shape[-2:])
    return frames[..., ys, xs]


def box_mean(frames: Any, box: BoxLike, robust: bool = False) -> Any:
    """Mean of a pixel box per frame (the corner background of the legacy scripts).

    With ``robust`` the box values above the 90th percentile are replaced by
    the median, then those below the 10th percentile of the modified box by its
    new median, in that order, exactly as the legacy laser-off animation did.
    """
    xp = array_module(frames)
    ys, xs = _box_slices(box, frames.shape[-2:])
    sub = frames[..., ys, xs].reshape(frames.shape[0], -1).astype(np.float64)
    if robust:
        hi = xp.percentile(sub, 90, axis=1, keepdims=True)
        sub = xp.where(sub > hi, xp.percentile(sub, 50, axis=1, keepdims=True), sub)
        lo = xp.percentile(sub, 10, axis=1, keepdims=True)
        sub = xp.where(sub < lo, xp.percentile(sub, 50, axis=1, keepdims=True), sub)
    return sub.mean(axis=1)


def normalize_box(frames: Any, box: BoxLike, robust: bool = False) -> Any:
    """Divide every frame by the mean of its background box (gain normalisation)."""
    background = box_mean(frames, box, robust=robust)
    return frames / background[:, None, None]
