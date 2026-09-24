"""Accumulators and the single-pass streaming engine.

A reduction is an :class:`Accumulator` that sees the frames of a scan once, in
batches, and keeps only its running totals (a sum image, weighted motor sums,
a downsampled volume). :func:`reduce` streams the frames of a scan through
every requested accumulator in one pass, sizing the batches from the memory
budget, prefetching the next batch while the current one is processed, and
running on the GPU when asked. Results are xarray objects with coordinates
and provenance.

The formulas are written once, in plain numpy or cupy, with no loops over
pixels or frames inside a batch. Motor-weighted quantities use ``tensordot``
over the frame axis so that no array of the full stack size is ever created.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import xarray as xr

from bexa._log import get_logger, progress
from bexa.core import backend
from bexa.core.cache import Cache
from bexa.core.parallel import Prefetcher
from bexa.core.provenance import RunStats, build_attrs
from bexa.core.registry import register_accumulator
from bexa.core.roi import ROI
from bexa.core.structure import PIXEL_DIMS, Structure
from bexa.io.base import FrameBatch, Source

log = get_logger(__name__)

__all__ = [
    "Accumulator",
    "ArgmaxMotor",
    "EnergyCOM",
    "FrameStats",
    "Histogram",
    "Max",
    "Mean",
    "Min",
    "MotorCOM",
    "OnOffSplit",
    "Plan",
    "Preview",
    "Projections",
    "RoiIntegral",
    "Sum",
    "block_reduce",
    "iter_batches",
    "make_plan",
    "parse_downsample",
    "reduce",
]


# ----------------------------------------------------------------- planning
@dataclass
class Window:
    """Pixel window of a plan and its spatial downsampling."""

    y: slice
    x: slice
    ds_y: int = 1
    ds_x: int = 1

    @property
    def full_shape(self) -> tuple[int, int]:
        return self.y.stop - self.y.start, self.x.stop - self.x.start

    @property
    def shape(self) -> tuple[int, int]:
        h, w = self.full_shape
        return h // self.ds_y, w // self.ds_x

    def coords(self) -> dict[str, np.ndarray]:
        """Full-resolution index of the first pixel of every (downsampled) pixel."""
        ny, nx = self.shape
        return {
            "y": self.y.start + self.ds_y * np.arange(ny),
            "x": self.x.start + self.ds_x * np.arange(nx),
        }


@dataclass
class Plan:
    """Everything derived from a structure, an ROI and a downsample request."""

    structure: Structure  # after the motor ROI
    window: Window
    motor_factors: tuple[int, ...]
    frame_ids: np.ndarray  # frames to read, sorted
    grid_shape: tuple[int, ...]  # motor grid after downsampling
    coords: dict[str, np.ndarray] = field(default_factory=dict)  # downsampled motor coords

    @property
    def motor_dims(self) -> tuple[str, ...]:
        return self.structure.motor_dims

    @property
    def dims(self) -> tuple[str, ...]:
        return self.motor_dims + PIXEL_DIMS

    @property
    def shape(self) -> tuple[int, ...]:
        return self.grid_shape + self.window.shape

    def grid_positions(self, frame_ids: np.ndarray) -> tuple[np.ndarray, ...]:
        """Position of each frame in the downsampled motor grid."""
        positions = self.structure.grid_positions(frame_ids)
        return tuple(p // f for p, f in zip(positions, self.motor_factors, strict=True))

    def motor_values(self, name: str, frame_ids: np.ndarray) -> np.ndarray:
        return self.structure.motor_values(name, frame_ids)

    def describe(self) -> dict[str, Any]:
        w = self.window
        return {
            "window": [w.y.start, w.y.stop, w.x.start, w.x.stop],
            "downsample": [*self.motor_factors, w.ds_y, w.ds_x],
            "grid": list(self.grid_shape),
            "n_frames": int(self.frame_ids.size),
            "dims": list(self.dims),
        }

    def all_coords(self) -> dict[str, Any]:
        coords: dict[str, Any] = dict(self.coords)
        coords.update(self.window.coords())
        return coords


def parse_downsample(
    downsample: int | Sequence[int] | None, structure: Structure
) -> tuple[tuple[int, ...], tuple[int, int]]:
    """Split a downsample request into motor factors and ``(ds_y, ds_x)``.

    Accepts the forms the v9 module accepted: an int or a 2-tuple for the
    pixel axes only, or one factor per dim ``(motor0, ..., ds_y, ds_x)``.
    """
    n_motors = len(structure.motor_dims)
    if downsample is None:
        return (1,) * n_motors, (1, 1)
    if isinstance(downsample, (int, np.integer)):
        f = int(downsample)
        return (1,) * n_motors, (f, f)
    factors = tuple(int(v) for v in downsample)
    if len(factors) == 2:
        return (1,) * n_motors, (factors[0], factors[1])
    if len(factors) == n_motors + 2:
        return factors[:n_motors], (factors[-2], factors[-1])
    raise ValueError(
        f"downsample must be an int, (ds_y, ds_x) or {n_motors + 2} factors for dims "
        f"{structure.dims}; got {downsample!r}"
    )


def make_plan(
    structure: Structure, roi: ROI | None = None, downsample: int | Sequence[int] | None = None
) -> Plan:
    """Resolve ROI and downsampling against a structure."""
    roi = roi or ROI()
    region = roi.motor_region(structure)
    sub = structure.sub(region)
    motor_factors, (ds_y, ds_x) = parse_downsample(downsample, structure)
    ys, xs = roi.pixel_slices(structure.frame_shape)
    window = Window(ys, xs, ds_y, ds_x)
    if window.shape[0] < 1 or window.shape[1] < 1:
        raise ValueError(
            f"downsampling {ds_y}x{ds_x} leaves no pixels in window {window.full_shape}"
        )

    # motor downsampling keeps grid positions that are multiples of the factor
    keep = tuple(slice(None, None, f) for f in motor_factors)
    kept = sub.sub(keep)
    frame_ids = kept.frame_ids()
    coords = {d: kept.coords[d] for d in kept.motor_dims}
    return Plan(sub, window, motor_factors, frame_ids, kept.motor_shape, coords)


def block_reduce(frames: Any, dy: int, dx: int, method: str = "mean") -> Any:
    """Downsample the last two axes by integer factors (numpy or cupy)."""
    if dy == 1 and dx == 1:
        return frames
    *lead, h, w = frames.shape
    ny, nx = h // dy, w // dx
    block = frames[..., : ny * dy, : nx * dx].reshape(*lead, ny, dy, nx, dx)
    if method == "sum":
        return block.sum(axis=(-3, -1))
    if method == "max":
        return block.max(axis=(-3, -1))
    if method == "mean":
        return block.mean(axis=(-3, -1), dtype=frames.dtype if frames.dtype.kind == "f" else None)
    raise ValueError(f"unknown block reduction {method!r}; use mean, sum or max")


# ------------------------------------------------------------- accumulators
class Accumulator(ABC):
    """Base class: keep running totals over batches of frames, then build xarray results.

    Subclasses set ``name`` (the key of their result), implement ``update``
    and ``result``, and declare ``live_copies`` (batch-sized temporaries they
    allocate) so the engine can size batches.
    """

    name: str = "accumulator"
    live_copies: int = 0

    def __init__(self, **params: Any) -> None:
        self.params = params
        self.plan: Plan
        self.xp: Any = np
        self.dtype: Any = np.float32
        self.n_frames_seen = 0

    def start(self, plan: Plan, xp: Any, dtype: Any) -> None:
        self.plan = plan
        self.xp = xp
        self.dtype = dtype
        self.n_frames_seen = 0
        self._allocate()

    def _allocate(self) -> None:  # optional hook
        return None

    @abstractmethod
    def update(self, frames: Any, frame_ids: np.ndarray) -> None:
        """Fold a batch ``(n, ny, nx)`` (already windowed and downsampled) into the totals."""

    @abstractmethod
    def result(self) -> xr.DataArray | dict[str, xr.DataArray]:
        """Build the result(s) on the host."""

    def describe(self) -> dict[str, Any]:
        """Identify the computation for cache keys and provenance."""
        return {"accumulator": type(self).__name__, **self.params}

    # ---- helpers for subclasses
    def _image(self, data: Any, name: str, **attrs: Any) -> xr.DataArray:
        return xr.DataArray(
            backend.to_host(data),
            dims=PIXEL_DIMS,
            coords=self.plan.window.coords(),
            name=name,
            attrs=attrs,
        )

    def _grid(self, data: Any, name: str, **attrs: Any) -> xr.DataArray:
        return xr.DataArray(
            backend.to_host(data),
            dims=self.plan.motor_dims,
            coords=self.plan.coords,
            name=name,
            attrs=attrs,
        )

    def _volume(self, data: Any, name: str, **attrs: Any) -> xr.DataArray:
        return xr.DataArray(
            backend.to_host(data),
            dims=self.plan.dims,
            coords=self.plan.all_coords(),
            name=name,
            attrs=attrs,
        )

    def _zeros(self, shape: Sequence[int], dtype: Any = np.float64) -> Any:
        return self.xp.zeros(tuple(shape), dtype=dtype)


@register_accumulator("sum")
class Sum(Accumulator):
    """Sum of all frames: the ``(y, x)`` integrated image."""

    name = "sum"

    def _allocate(self) -> None:
        self.total = self._zeros(self.plan.window.shape)

    def update(self, frames: Any, frame_ids: np.ndarray) -> None:
        self.total += frames.sum(axis=0, dtype=self.total.dtype)
        self.n_frames_seen += len(frame_ids)

    def result(self) -> xr.DataArray:
        return self._image(self.total, self.name, n_frames=self.n_frames_seen)


@register_accumulator("mean")
class Mean(Sum):
    """Mean frame."""

    name = "mean"

    def result(self) -> xr.DataArray:
        return self._image(
            self.total / max(self.n_frames_seen, 1), self.name, n_frames=self.n_frames_seen
        )


@register_accumulator("max")
class Max(Accumulator):
    """Pixel-wise maximum over frames."""

    name = "max"

    def _allocate(self) -> None:
        self.total = self.xp.full(self.plan.window.shape, -self.xp.inf, dtype=self.dtype)

    def update(self, frames: Any, frame_ids: np.ndarray) -> None:
        self.xp.maximum(self.total, frames.max(axis=0), out=self.total)
        self.n_frames_seen += len(frame_ids)

    def result(self) -> xr.DataArray:
        return self._image(self.total, self.name, n_frames=self.n_frames_seen)


@register_accumulator("min")
class Min(Accumulator):
    """Pixel-wise minimum over frames."""

    name = "min"

    def _allocate(self) -> None:
        self.total = self.xp.full(self.plan.window.shape, self.xp.inf, dtype=self.dtype)

    def update(self, frames: Any, frame_ids: np.ndarray) -> None:
        self.xp.minimum(self.total, frames.min(axis=0), out=self.total)
        self.n_frames_seen += len(frame_ids)

    def result(self) -> xr.DataArray:
        return self._image(self.total, self.name, n_frames=self.n_frames_seen)


@register_accumulator("preview")
class Preview(Accumulator):
    """The (downsampled) volume itself: ``(motors..., y, x)`` with coordinates.

    Parameters
    ----------
    apply_log
        Store ``log10(1 + I)`` instead of ``I`` (the v9 preview did this).
    fill
        Value at grid points without a frame (partial scans).
    """

    name = "preview"
    live_copies = 1

    def __init__(self, apply_log: bool = False, fill: float = np.nan) -> None:
        super().__init__(apply_log=apply_log, fill=fill)
        self.apply_log = apply_log
        self.fill = fill

    def _allocate(self) -> None:
        self.volume = self.xp.full(self.plan.shape, self.fill, dtype=self.dtype)

    def update(self, frames: Any, frame_ids: np.ndarray) -> None:
        if self.apply_log:
            frames = self.xp.log10(1.0 + self.xp.clip(frames, 0, None))
        positions = self.plan.grid_positions(frame_ids)
        self.volume[tuple(self.xp.asarray(p) for p in positions)] = frames
        self.n_frames_seen += len(frame_ids)

    def result(self) -> xr.DataArray:
        return self._volume(self.volume, self.name, apply_log=self.apply_log)


@register_accumulator("projections")
class Projections(Accumulator):
    """Integrated intensity along every axis pair.

    Results: ``xy`` (sum image), ``<motor>_y`` and ``<motor>_x`` (intensity vs a
    motor and one pixel axis), ``grid`` (total intensity per grid point) and,
    for two or more motors, ``<motor_i>_<motor_j>`` maps.
    """

    name = "projections"

    def _allocate(self) -> None:
        ny, nx = self.plan.window.shape
        self.image = self._zeros((ny, nx))
        self.rows = self._zeros((*self.plan.grid_shape, ny))
        self.cols = self._zeros((*self.plan.grid_shape, nx))

    def update(self, frames: Any, frame_ids: np.ndarray) -> None:
        xp = self.xp
        self.image += frames.sum(axis=0, dtype=self.image.dtype)
        pos = tuple(xp.asarray(p) for p in self.plan.grid_positions(frame_ids))
        self.rows[pos] += frames.sum(axis=2, dtype=self.rows.dtype)
        self.cols[pos] += frames.sum(axis=1, dtype=self.cols.dtype)
        self.n_frames_seen += len(frame_ids)

    def result(self) -> dict[str, xr.DataArray]:
        out: dict[str, xr.DataArray] = {"xy": self._image(self.image, "xy")}
        dims = self.plan.motor_dims
        n = len(dims)
        grid_total = self.rows.sum(axis=-1)
        out["grid"] = self._grid(grid_total, "grid")
        coords = self.plan.all_coords()
        for i, d in enumerate(dims):
            other = tuple(j for j in range(n) if j != i)
            rows = self.rows.sum(axis=other) if other else self.rows
            cols = self.cols.sum(axis=other) if other else self.cols
            out[f"{d}_y"] = xr.DataArray(
                backend.to_host(rows),
                dims=(d, "y"),
                coords={d: coords[d], "y": coords["y"]},
                name=f"{d}_y",
            )
            out[f"{d}_x"] = xr.DataArray(
                backend.to_host(cols),
                dims=(d, "x"),
                coords={d: coords[d], "x": coords["x"]},
                name=f"{d}_x",
            )
        for i in range(n):
            for j in range(i + 1, n):
                other = tuple(k for k in range(n) if k not in (i, j))
                pair = grid_total.sum(axis=other) if other else grid_total
                di, dj = dims[i], dims[j]
                out[f"{di}_{dj}"] = xr.DataArray(
                    backend.to_host(pair),
                    dims=(di, dj),
                    coords={di: coords[di], dj: coords[dj]},
                    name=f"{di}_{dj}",
                )
        return out


@register_accumulator("roi_integral")
class RoiIntegral(Accumulator):
    """Integrated intensity of one or more pixel ROIs at every grid point.

    Parameters
    ----------
    rois
        ``{name: ROI}`` with pixel ranges in full-resolution coordinates, or one ROI.
    method
        ``sum``, ``mean`` or ``max`` over the ROI pixels.
    """

    name = "roi_integral"

    def __init__(self, rois: dict[str, ROI] | ROI | None = None, method: str = "sum") -> None:
        if rois is None:
            rois = {"roi": ROI()}
        if isinstance(rois, ROI):
            rois = {"roi": rois}
        super().__init__(rois={k: v.to_dict() for k, v in rois.items()}, method=method)
        self.rois = rois
        self.method = method

    def _allocate(self) -> None:
        w = self.plan.window
        self.slices: dict[str, tuple[slice, slice]] = {}
        for key, roi in self.rois.items():
            y0, y1, x0, x1 = roi.pixel_window(self.plan.structure.frame_shape)
            # translate to window-relative, downsampled pixels
            ys = slice(
                max(y0 - w.y.start, 0) // w.ds_y, max(min(y1, w.y.stop) - w.y.start, 0) // w.ds_y
            )
            xs = slice(
                max(x0 - w.x.start, 0) // w.ds_x, max(min(x1, w.x.stop) - w.x.start, 0) // w.ds_x
            )
            self.slices[key] = (ys, xs)
        self.values = {
            k: self.xp.full(self.plan.grid_shape, np.nan, dtype=np.float64) for k in self.rois
        }

    def update(self, frames: Any, frame_ids: np.ndarray) -> None:
        pos = tuple(self.xp.asarray(p) for p in self.plan.grid_positions(frame_ids))
        for key, (ys, xs) in self.slices.items():
            block = frames[:, ys, xs]
            if self.method == "sum":
                value = block.sum(axis=(1, 2), dtype=np.float64)
            elif self.method == "mean":
                value = block.mean(axis=(1, 2), dtype=np.float64)
            elif self.method == "max":
                value = block.max(axis=(1, 2))
            else:
                raise ValueError(f"unknown ROI method {self.method!r}")
            self.values[key][pos] = value
        self.n_frames_seen += len(frame_ids)

    def result(self) -> dict[str, xr.DataArray]:
        return {
            f"roi_{key}": self._grid(
                v, f"roi_{key}", method=self.method, roi=self.rois[key].to_dict()
            )
            for key, v in self.values.items()
        }


@register_accumulator("motor_com")
class MotorCOM(Accumulator):
    """Per-pixel centre of mass (and higher moments) of the rocking curve along motor axes.

    For each pixel, with weights ``w_f`` from frame ``f`` and motor value ``m_f``::

        com   = sum(w * m) / sum(w)
        width = sqrt(sum(w * m^2) / sum(w) - com^2)

    Weights are the frame intensities after optional Gaussian smoothing
    (``sigma`` pixels, as in the lab's notebooks) and clipping to ``clip``;
    ``weights="log"`` uses ``log10(1 + I)`` as v9's preview-based COM did.

    Parameters
    ----------
    axes
        Motor dims to analyse (default: all motor dims).
    sigma
        Gaussian smoothing of every frame before weighting; 0 disables it.
    moments
        1 gives ``com_<axis>``; 2 adds ``width_<axis>``; 3 adds ``skew_<axis>``;
        4 adds ``kurtosis_<axis>``. ``total`` (sum of weights) is always returned.
    """

    name = "motor_com"
    live_copies = 2

    def __init__(
        self,
        axes: Sequence[str] | None = None,
        sigma: float = 3.0,
        weights: str = "linear",
        clip: float = 1e-10,
        moments: int = 2,
    ) -> None:
        if weights not in ("linear", "log"):
            raise ValueError("weights must be 'linear' or 'log'")
        if not 1 <= moments <= 4:
            raise ValueError("moments must be between 1 and 4")
        super().__init__(
            axes=list(axes) if axes else None,
            sigma=sigma,
            weights=weights,
            clip=clip,
            moments=moments,
        )
        self.axes = tuple(axes) if axes else None
        self.sigma = float(sigma)
        self.weights = weights
        self.clip = clip
        self.moments = moments

    def _allocate(self) -> None:
        self.use_axes = self.axes or self.plan.motor_dims
        missing = [a for a in self.use_axes if a not in self.plan.motor_dims]
        if missing:
            raise ValueError(
                f"axes {missing} are not motor dims of this scan {self.plan.motor_dims}"
            )
        shape = self.plan.window.shape
        self.s0 = self._zeros(shape)
        self.sums = {a: [self._zeros(shape) for _ in range(self.moments)] for a in self.use_axes}
        self._ndi = backend.ndimage_module(self.xp) if self.sigma > 0 else None

    def _weights(self, frames: Any) -> Any:
        xp = self.xp
        w = frames if frames.dtype.kind == "f" else frames.astype(self.dtype)
        if self.weights == "log":
            w = xp.log10(1.0 + xp.clip(w, 0, None))
        if self._ndi is not None:
            w = self._ndi.gaussian_filter(w, sigma=(0, self.sigma, self.sigma))
        return xp.clip(w, self.clip, None)

    def update(self, frames: Any, frame_ids: np.ndarray) -> None:
        xp = self.xp
        w = self._weights(frames)
        self.s0 += w.sum(axis=0, dtype=self.s0.dtype)
        for axis in self.use_axes:
            m = xp.asarray(self.plan.motor_values(axis, frame_ids), dtype=np.float64)
            power = m
            for k in range(self.moments):
                self.sums[axis][k] += xp.tensordot(power, w, axes=(0, 0))
                power = power * m
        self.n_frames_seen += len(frame_ids)

    def result(self) -> dict[str, xr.DataArray]:
        xp = self.xp
        s0 = xp.clip(self.s0, self.clip, None)
        attrs = {"sigma": self.sigma, "weights": self.weights}
        out: dict[str, xr.DataArray] = {"total": self._image(self.s0, "total", **attrs)}
        for axis in self.use_axes:
            unit = self.plan.structure.units.get(axis, "")
            mean1 = self.sums[axis][0] / s0
            out[f"com_{axis}"] = self._image(mean1, f"com_{axis}", motor=axis, units=unit, **attrs)
            if self.moments >= 2:
                mean2 = self.sums[axis][1] / s0
                var = xp.clip(mean2 - mean1**2, 0, None)
                width = xp.sqrt(var)
                out[f"width_{axis}"] = self._image(
                    width, f"width_{axis}", motor=axis, units=unit, **attrs
                )
            if self.moments >= 3:
                mean3 = self.sums[axis][2] / s0
                third = mean3 - 3 * mean1 * mean2 + 2 * mean1**3
                skew = third / xp.clip(width**3, 1e-30, None)
                out[f"skew_{axis}"] = self._image(skew, f"skew_{axis}", motor=axis, **attrs)
            if self.moments >= 4:
                mean4 = self.sums[axis][3] / s0
                fourth = mean4 - 4 * mean1 * mean3 + 6 * mean1**2 * mean2 - 3 * mean1**4
                kurt = fourth / xp.clip(var**2, 1e-30, None) - 3.0
                out[f"kurtosis_{axis}"] = self._image(kurt, f"kurtosis_{axis}", motor=axis, **attrs)
        return out


@register_accumulator("energy_com")
class EnergyCOM(MotorCOM):
    """Centre of mass along the ``energy`` dim of an energy series."""

    name = "energy_com"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("sigma", 0.0)
        super().__init__(axes=("energy",), **kwargs)


@register_accumulator("argmax_motor")
class ArgmaxMotor(Accumulator):
    """Motor value of the brightest frame at each pixel (a robust COM alternative)."""

    name = "argmax_motor"

    def __init__(self, axis: str | None = None) -> None:
        super().__init__(axis=axis)
        self.axis = axis

    def _allocate(self) -> None:
        self.use_axis = self.axis or self.plan.motor_dims[-1]
        shape = self.plan.window.shape
        self.best = self.xp.full(shape, -self.xp.inf, dtype=self.dtype)
        self.value = self.xp.full(shape, np.nan, dtype=np.float64)

    def update(self, frames: Any, frame_ids: np.ndarray) -> None:
        xp = self.xp
        idx = frames.argmax(axis=0)
        batch_max = xp.take_along_axis(frames, idx[None], axis=0)[0]
        m = xp.asarray(self.plan.motor_values(self.use_axis, frame_ids), dtype=np.float64)
        better = batch_max > self.best
        self.best = xp.where(better, batch_max, self.best)
        self.value = xp.where(better, m[idx], self.value)
        self.n_frames_seen += len(frame_ids)

    def result(self) -> dict[str, xr.DataArray]:
        return {
            f"argmax_{self.use_axis}": self._image(
                self.value, f"argmax_{self.use_axis}", motor=self.use_axis
            ),
            "peak": self._image(self.best, "peak"),
        }


@register_accumulator("frame_stats")
class FrameStats(Accumulator):
    """Per-frame sum, mean, max and detector centre of mass, on the motor grid."""

    name = "frame_stats"

    def _allocate(self) -> None:
        shape = self.plan.grid_shape
        self.stats = {
            k: self.xp.full(shape, np.nan, dtype=np.float64)
            for k in ("sum", "mean", "max", "com_y", "com_x")
        }
        coords = self.plan.window.coords()
        self.yy = self.xp.asarray(coords["y"], dtype=np.float64)
        self.xx = self.xp.asarray(coords["x"], dtype=np.float64)

    def update(self, frames: Any, frame_ids: np.ndarray) -> None:
        xp = self.xp
        pos = tuple(xp.asarray(p) for p in self.plan.grid_positions(frame_ids))
        total = frames.sum(axis=(1, 2), dtype=np.float64)
        safe = xp.clip(total, 1e-30, None)
        self.stats["sum"][pos] = total
        self.stats["mean"][pos] = total / frames[0].size
        self.stats["max"][pos] = frames.max(axis=(1, 2))
        self.stats["com_y"][pos] = (
            xp.tensordot(frames.sum(axis=2, dtype=np.float64), self.yy, axes=(1, 0)) / safe
        )
        self.stats["com_x"][pos] = (
            xp.tensordot(frames.sum(axis=1, dtype=np.float64), self.xx, axes=(1, 0)) / safe
        )
        self.n_frames_seen += len(frame_ids)

    def result(self) -> dict[str, xr.DataArray]:
        return {f"frame_{k}": self._grid(v, f"frame_{k}") for k, v in self.stats.items()}


@register_accumulator("histogram")
class Histogram(Accumulator):
    """Histogram of pixel values over all frames (for colour limits and thresholds)."""

    name = "histogram"

    def __init__(self, bins: int = 256, range: tuple[float, float] | None = None) -> None:
        super().__init__(bins=bins, range=list(range) if range else None)
        self.bins = bins
        self.range = range

    def _allocate(self) -> None:
        self.edges: Any = None
        self.counts: Any = None

    def update(self, frames: Any, frame_ids: np.ndarray) -> None:
        xp = self.xp
        if self.edges is None:
            lo, hi = self.range or (float(frames.min()), float(frames.max()))
            self.edges = xp.linspace(lo, hi, self.bins + 1)
            self.counts = xp.zeros(self.bins, dtype=np.int64)
        counts, _ = xp.histogram(frames, bins=self.edges)
        self.counts += counts
        self.n_frames_seen += len(frame_ids)

    def result(self) -> xr.DataArray:
        edges = backend.to_host(self.edges)
        centers = 0.5 * (edges[1:] + edges[:-1])
        return xr.DataArray(
            backend.to_host(self.counts),
            dims=("value",),
            coords={"value": centers},
            name="histogram",
        )


class OnOffSplit(Accumulator):
    """Run an accumulator separately for laser-on and laser-off frames.

    Parameters
    ----------
    factory
        Callable returning a fresh inner accumulator (for example ``Sum``).
    flag
        Name of the per-frame channel in ``structure.per_frame`` holding 1/0.
    """

    name = "on_off"

    def __init__(self, factory: Any, flag: str = "laser") -> None:
        super().__init__(inner=getattr(factory, "__name__", str(factory)), flag=flag)
        self.factory = factory
        self.flag = flag
        self.inner = {"on": factory(), "off": factory()}

    def start(self, plan: Plan, xp: Any, dtype: Any) -> None:
        super().start(plan, xp, dtype)
        for acc in self.inner.values():
            acc.start(plan, xp, dtype)

    def update(self, frames: Any, frame_ids: np.ndarray) -> None:
        flags = np.asarray(self.plan.structure.per_frame[self.flag])[frame_ids]
        on = flags == 1
        for key, mask in (("on", on), ("off", ~on)):
            if mask.any():
                sel = self.xp.asarray(mask)
                self.inner[key].update(frames[sel], frame_ids[mask])
        self.n_frames_seen += len(frame_ids)

    def result(self) -> dict[str, xr.DataArray]:
        out: dict[str, xr.DataArray] = {}
        for key, acc in self.inner.items():
            res = acc.result()
            if isinstance(res, dict):
                out.update({f"{key}_{k}": v for k, v in res.items()})
            else:
                out[f"{key}_{res.name}"] = res
        return out


# ---------------------------------------------------------------- engine
def _batch_size(
    source: Source, plan: Plan, dtype: Any, device: str, live_copies: int, batch_frames: int | None
) -> int:
    n = int(plan.frame_ids.size)
    if n == 0:
        return 1
    if batch_frames is not None:
        return max(1, min(int(batch_frames), n))
    h, w = plan.window.full_shape
    frame_bytes = h * w * np.dtype(dtype).itemsize
    size = backend.choose_batch_frames(
        frame_bytes, live_copies=live_copies, device=device, max_frames=n
    )
    chunk = max(int(getattr(source, "chunk_frames", 1)), 1)
    if size >= chunk:
        size -= size % chunk
    return max(size, 1)


def iter_batches(
    source: Source,
    plan: Plan,
    batch_frames: int | None = None,
    dtype: Any = np.float32,
    device: str = "cpu",
    method: str = "mean",
    prefetch: bool = True,
    live_copies: int = 4,
) -> Iterator[FrameBatch]:
    """Stream the frames of ``plan`` in batches, windowed, downsampled and moved to ``device``."""
    size = _batch_size(source, plan, dtype, device, live_copies, batch_frames)
    w = plan.window
    ids_all = plan.frame_ids

    def generate() -> Iterator[FrameBatch]:
        for start in range(0, ids_all.size, size):
            ids = ids_all[start : start + size]
            frames = source.read_frames(ids, w.y, w.x, dtype=dtype)
            frames = block_reduce(frames, w.ds_y, w.ds_x, method)
            if device == "cuda":
                frames = backend.to_device(frames, "cuda")
            yield FrameBatch(frames, ids, w.y, w.x)

    if prefetch and ids_all.size > size:
        return iter(Prefetcher(generate(), depth=2))
    return generate()


def _instantiate(accumulators: Iterable[Any]) -> list[Accumulator]:
    result = []
    for acc in accumulators:
        if isinstance(acc, type):
            acc = acc()
        if not isinstance(acc, Accumulator):
            raise TypeError(f"{acc!r} is not an Accumulator")
        result.append(acc)
    return result


def _run(
    source: Source,
    plan: Plan,
    accumulators: list[Accumulator],
    device: str,
    dtype: Any,
    batch_frames: int | None,
    method: str,
    prefetch: bool,
    show_progress: bool,
) -> None:
    xp = backend.array_module(device)
    for acc in accumulators:
        acc.start(plan, xp, dtype)
    live = 4 + max((a.live_copies for a in accumulators), default=0)
    batches = iter_batches(source, plan, batch_frames, dtype, device, method, prefetch, live)
    n_batches = math.ceil(
        plan.frame_ids.size / max(_batch_size(source, plan, dtype, device, live, batch_frames), 1)
    )
    for batch in progress(batches, total=n_batches, desc="reducing", enabled=show_progress):
        for acc in accumulators:
            acc.update(batch.frames, batch.frame_ids)


def reduce(
    target: Any,
    accumulators: Iterable[Any],
    roi: ROI | None = None,
    downsample: int | Sequence[int] | None = None,
    device: str | None = "auto",
    batch_frames: int | None = None,
    dtype: Any = np.float32,
    method: str = "mean",
    cache: Cache | bool | None = None,
    prefetch: bool = True,
    show_progress: bool = False,
) -> dict[str, xr.DataArray]:
    """Run accumulators over a scan in one streaming pass.

    Parameters
    ----------
    target
        A :class:`bexa.core.scan.Scan`, or a ``(source, structure)`` pair.
    accumulators
        Accumulator instances or classes.
    roi, downsample
        Region and downsampling (see :func:`make_plan`); a ``Scan`` supplies its
        own ROI when ``roi`` is None.
    device
        ``"auto"``, ``"cpu"`` or ``"cuda"``. On an out-of-memory error the batch
        size is halved; after two GPU failures the run falls back to the CPU.
    cache
        ``True`` (the scan's cache), a :class:`Cache`, or ``None``/``False``.

    Returns
    -------
    dict
        Result name -> DataArray with coordinates and provenance attrs.
    """
    from bexa.core.scan import Scan

    if isinstance(target, Scan):
        source, structure = target.source, target.structure
        files = target.files
        if roi is None:
            roi = target.roi
        if cache is True or (
            cache is None and target.cache is not None and target.cache_reductions
        ):
            cache = target.cache
    else:
        source, structure = target
        files = source.files()
    if cache is True:
        cache = None
    plan = make_plan(structure, roi, downsample)
    accs = _instantiate(accumulators)
    resolved = backend.resolve_device(device)
    dtype = np.dtype(dtype)

    results: dict[str, xr.DataArray] = {}
    to_run: list[tuple[Accumulator, str | None]] = []
    for acc in accs:
        key = None
        if isinstance(cache, Cache):
            key = cache.key(
                files,
                {
                    "acc": acc.describe(),
                    "plan": plan.describe(),
                    "method": method,
                    "dtype": dtype.str,
                },
            )
            hit = cache.get(key)
            if hit is not None:
                results.update(_unpack(hit))
                log.debug("cache hit for %s", acc.describe()["accumulator"])
                continue
        to_run.append((acc, key))
    if not to_run:
        return results

    stats = RunStats(resolved).start()
    attempts = 0
    current_batch = batch_frames
    while True:
        try:
            _run(
                source,
                plan,
                [a for a, _ in to_run],
                resolved,
                dtype,
                current_batch,
                method,
                prefetch,
                show_progress,
            )
            break
        except Exception as exc:
            if not backend.is_oom_error(exc):
                raise
            attempts += 1
            backend.free_device_memory()
            effective = current_batch or _batch_size(source, plan, dtype, resolved, 4, None)
            if resolved == "cuda" and attempts >= 2:
                log.warning("GPU out of memory twice; falling back to the CPU")
                resolved = "cpu"
                current_batch = None
            elif effective > 1:
                current_batch = max(effective // 2, 1)
                log.warning("out of memory; retrying with batches of %d frames", current_batch)
            else:
                raise
    run_attrs = stats.stop(frames=int(plan.frame_ids.size))
    base_attrs = build_attrs(
        files, {"plan": plan.describe(), "method": method, "dtype": dtype.str}, **run_attrs
    )

    for acc, key in to_run:
        res = acc.result()
        res = _unpack(res)
        for name, arr in res.items():
            arr.attrs.update(
                {
                    **base_attrs,
                    "accumulator": acc.describe()["accumulator"],
                    "params": _json(acc.params),
                }
            )
            arr.name = name
        if key is not None and isinstance(cache, Cache):
            cache.put(key, xr.Dataset(res))
        results.update(res)
    return results


def _unpack(res: Any) -> dict[str, xr.DataArray]:
    if isinstance(res, xr.Dataset):
        return {str(k): res[k] for k in res.data_vars}
    if isinstance(res, xr.DataArray):
        return {str(res.name): res}
    return {str(k): v for k, v in res.items()}


def _json(obj: Any) -> str:
    from bexa.core.provenance import to_json

    return to_json(obj)
