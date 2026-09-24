"""Shift detection, alignment, overlap search, stitching and mosaics.

Ports of the DFXM notebooks: the ``basicDFXM`` shift detection (its
``correlate2d`` search replaced by an FFT cross-correlation, which is exact
for periodic boundaries and thousands of times faster on full frames), the
``betterCOM_stitchingMu``/``stitchingZ`` overlap search on summed curves and
their frame-wise stitch, the ``RLP_code_SpatialOverlap`` sample-position
mosaic and the layer stacking of ``Slices_to_Voxels``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from bexa.core.backend import array_module, ndimage_module, to_host

__all__ = [
    "align_stack",
    "find_overlap_1d",
    "find_overlap_by_axis",
    "mosaic_from_grid",
    "mosaic_from_positions",
    "motor_expected_shift",
    "phase_correlation_shift",
    "stack_layers",
    "stitch_along_axis",
]


# ------------------------------------------------------------------ shifts
def phase_correlation_shift(
    reference: Any, image: Any, max_shift: int | None = None
) -> tuple[int, int]:
    """Integer shift ``(dy, dx)`` that moves ``image`` onto ``reference``.

    The cross-correlation is evaluated through the FFT, so it equals the
    ``scipy.signal.correlate2d`` search of the legacy notebook with periodic
    boundaries at a fraction of the cost. Applying
    ``scipy.ndimage.shift(image, (dy, dx))`` aligns the image with the
    reference. With ``max_shift`` only ``|dy|, |dx| <= max_shift`` are considered.
    """
    xp = array_module(reference)
    ref = xp.asarray(reference, dtype=np.float64)
    img = xp.asarray(image, dtype=np.float64)
    ref = ref - ref.mean()
    img = img - img.mean()
    corr = xp.fft.irfft2(xp.fft.rfft2(ref) * xp.conj(xp.fft.rfft2(img)), s=ref.shape)
    h, w = ref.shape
    if max_shift is not None:
        offsets = xp.arange(-max_shift, max_shift + 1)
        rows = offsets % h
        cols = offsets % w
        window = corr[xp.ix_(rows, cols)]
        iy, ix = np.unravel_index(int(to_host(window.argmax())), window.shape)
        return int(offsets[iy]), int(offsets[ix])
    iy, ix = np.unravel_index(int(to_host(corr.argmax())), corr.shape)
    dy = iy - h if iy > h // 2 else iy
    dx = ix - w if ix > w // 2 else ix
    return int(dy), int(dx)


def motor_expected_shift(motor_values: Any, pixels_per_unit: float = 1.0) -> np.ndarray:
    """Pixel shift of every frame relative to the first one, from a motor readback."""
    values = np.asarray(motor_values, dtype=float)
    return (values - values[0]) * pixels_per_unit


def align_stack(
    frames: Any,
    reference: int = 0,
    max_shift: int = 5,
    expected: Any | None = None,
    order: int = 1,
) -> tuple[Any, np.ndarray]:
    """Shift every frame onto the reference frame; returns ``(aligned, shifts)``.

    Port of ``detect_and_apply_shifts``: the shift of frame ``i`` is the
    expected motor-driven shift (``(n, 2)`` array of ``(dy, dx)``, default 0)
    plus the residual found by cross-correlation with the reference frame.
    """
    xp = array_module(frames)
    ndi = ndimage_module(frames)
    n = frames.shape[0]
    shifts = np.zeros((n, 2))
    if expected is not None:
        shifts += np.asarray(expected, dtype=float).reshape(n, 2)
    aligned = xp.empty_like(frames)
    ref = frames[reference]
    for i in range(n):
        if i != reference:
            shifts[i] += phase_correlation_shift(ref, frames[i], max_shift)
        aligned[i] = (
            frames[i]
            if not shifts[i].any()
            else ndi.shift(frames[i], tuple(shifts[i]), order=order)
        )
    return aligned, shifts


# ----------------------------------------------------------------- overlaps
def find_overlap_1d(
    a: Any, b: Any, max_offset: int | None = None, min_overlap: int = 2
) -> tuple[int, dict[int, float]]:
    """Offset of curve ``b`` relative to curve ``a`` that makes them agree best.

    A positive offset ``k`` means ``b[j]`` corresponds to ``a[j + k]``; a
    negative one that ``a`` starts inside ``b``. The score of an offset is the
    mean absolute difference over the overlapping samples, as in the stitching
    notebooks. Returns the best offset and the score of every offset tried.
    """
    a = np.asarray(to_host(a), dtype=float)
    b = np.asarray(to_host(b), dtype=float)
    limit = max_offset if max_offset is not None else max(len(a), len(b)) - min_overlap
    scores: dict[int, float] = {}
    for k in range(-limit, limit + 1):
        if k >= 0:
            m = min(len(a) - k, len(b))
            if m < min_overlap:
                continue
            scores[k] = float(np.mean(np.abs(a[k : k + m] - b[:m])))
        else:
            m = min(len(b) + k, len(a))
            if m < min_overlap:
                continue
            scores[k] = float(np.mean(np.abs(b[-k : -k + m] - a[:m])))
    if not scores:
        raise ValueError("the curves are too short for the requested overlap")
    best = min(scores, key=lambda k: scores[k])
    return best, scores


def find_overlap_by_axis(axes: Sequence[Any], tolerance: float | None = None) -> list[int]:
    """Offsets (in samples) of several scans along a shared motor axis, from the axis values.

    The first axis defines index 0; every other axis is placed where its first
    value sits on the first axis' grid (nearest sample, extrapolated with the
    step size when it starts earlier or later).
    """
    first = np.asarray(axes[0], dtype=float)
    step = float(np.median(np.diff(first))) if first.size > 1 else 1.0
    offsets = []
    for axis in axes:
        values = np.asarray(axis, dtype=float)
        offset = (values[0] - first[0]) / step
        if tolerance is not None and abs(offset - round(offset)) * abs(step) > tolerance:
            raise ValueError(f"axis start {values[0]} is not on the grid of the first axis")
        offsets.append(round(offset))
    shift = min(offsets)
    return [o - shift for o in offsets]


# ---------------------------------------------------------------- stitching
def stitch_along_axis(
    volumes: Sequence[Any], offsets: Sequence[int], mode: str = "max", fill: float = 0.0
) -> Any:
    """Combine arrays that overlap along their first axis.

    ``offsets`` are the start indices of each array in the stitched result.
    ``mode``: ``"max"`` (element-wise maximum, the stitched-image rule),
    ``"brightest"`` (per index keep the array whose slice has the largest
    total, the frame rule of the mu-stitching notebook), ``"sum"`` or
    ``"mean"`` (average where arrays overlap).
    """
    if len(volumes) != len(offsets):
        raise ValueError("one offset per volume is needed")
    xp = array_module(volumes[0])
    length = max(int(o) + v.shape[0] for v, o in zip(volumes, offsets, strict=True))
    shape = (length, *volumes[0].shape[1:])
    if mode == "max":
        out = xp.full(shape, -xp.inf, dtype=np.float64)
        for v, o in zip(volumes, offsets, strict=True):
            out[o : o + v.shape[0]] = xp.maximum(out[o : o + v.shape[0]], v)
        out[~xp.isfinite(out)] = fill
        return out
    if mode in ("sum", "mean"):
        out = xp.zeros(shape, dtype=np.float64)
        count = xp.zeros(length, dtype=np.float64)
        for v, o in zip(volumes, offsets, strict=True):
            out[o : o + v.shape[0]] += v
            count[o : o + v.shape[0]] += 1
        if mode == "mean":
            with np.errstate(invalid="ignore", divide="ignore"):
                out = out / count.reshape((-1,) + (1,) * (out.ndim - 1))
            out[~xp.isfinite(out)] = fill
        return out
    if mode == "brightest":
        out = xp.full(shape, fill, dtype=np.float64)
        best = xp.full(length, -xp.inf, dtype=np.float64)
        for v, o in zip(volumes, offsets, strict=True):
            totals = v.reshape(v.shape[0], -1).sum(axis=1)
            take = totals > best[o : o + v.shape[0]]
            idx = xp.nonzero(take)[0]
            out[o + idx] = v[idx]
            best[o + idx] = totals[idx]
        return out
    raise ValueError(f"unknown mode {mode!r}; use max, brightest, sum or mean")


def mosaic_from_positions(
    tiles: Sequence[Any], positions: Any, mode: str = "max", fill: float = 0.0
) -> Any:
    """Place 2-D tiles at pixel positions ``(y, x)`` of their top-left corner."""
    positions = np.asarray(positions, dtype=int).reshape(-1, 2)
    if len(tiles) != len(positions):
        raise ValueError("one position per tile is needed")
    xp = array_module(tiles[0])
    h, w = tiles[0].shape[-2:]
    ymax = int(positions[:, 0].max()) + h
    xmax = int(positions[:, 1].max()) + w
    if mode == "max":
        out = xp.full((ymax, xmax), -xp.inf, dtype=np.float64)
    else:
        out = xp.zeros((ymax, xmax), dtype=np.float64)
    count = xp.zeros((ymax, xmax), dtype=np.float64)
    for tile, (y, x) in zip(tiles, positions, strict=True):
        window = (slice(y, y + h), slice(x, x + w))
        if mode == "max":
            out[window] = xp.maximum(out[window], tile)
        else:
            out[window] += tile
            count[window] += 1
    if mode == "max":
        out[~xp.isfinite(out)] = fill
    elif mode == "mean":
        with np.errstate(invalid="ignore", divide="ignore"):
            out = out / count
        out[~xp.isfinite(out)] = fill
    elif mode != "sum":
        raise ValueError(f"unknown mode {mode!r}; use max, sum or mean")
    return out


def mosaic_from_grid(
    tiles: Any, step_px: int | tuple[int, int], mode: str = "max", fill: float = 0.0
) -> Any:
    """Mosaic of tiles ``(ny, nx, H, W)`` acquired on a regular sample-position grid.

    Tile ``[i, j]`` is placed at ``(i * step_y, j * step_x)``; overlapping
    pixels keep the maximum (or the sum / mean). Snake-ordered scans are already
    on the grid when the tiles come from a bexa preview volume.
    """
    tiles = np.asarray(tiles) if not hasattr(tiles, "__array_function__") else tiles
    ny, nx = tiles.shape[:2]
    sy, sx = (step_px, step_px) if isinstance(step_px, int) else step_px
    positions = [(i * sy, j * sx) for i in range(ny) for j in range(nx)]
    flat = [tiles[i, j] for i in range(ny) for j in range(nx)]
    return mosaic_from_positions(flat, positions, mode=mode, fill=fill)


def stack_layers(layers: Sequence[Any], offsets: Any | None = None, fill: float = 0.0) -> Any:
    """Stack 2-D layers into a ``(z, y, x)`` volume, each moved by an integer ``(dy, dx)``.

    The per-layer offsets play the role of the ``phi_off``/``mu_off`` sliders
    of ``Slices_to_Voxels``: content is shifted inside a canvas of the same
    size and the uncovered border takes ``fill``.
    """
    xp = array_module(layers[0])
    h, w = layers[0].shape
    out = xp.full((len(layers), h, w), fill, dtype=np.float64)
    shifts = np.zeros((len(layers), 2), dtype=int)
    if offsets is not None:
        shifts[:] = np.asarray(offsets, dtype=int).reshape(len(layers), 2)
    for k, (layer, (dy, dx)) in enumerate(zip(layers, shifts, strict=True)):
        ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
        xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
        out[k, yd, xd] = layer[ys, xs]
    return out
