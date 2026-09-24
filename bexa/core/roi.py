"""Regions of interest over pixel, motor and energy dims.

Pixel ranges (``y``, ``x``) are half-open index ranges ``[lo, hi)``. Motor and
energy ranges are inclusive physical values by default (``chi=(0, 8)`` keeps
every grid point with 0 <= chi <= 8); pass ``index=True`` to give grid indices
instead. A missing bound means "to the edge".
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from bexa.core.structure import PIXEL_DIMS, Structure

Range = tuple[float | None, float | None]


class ROI:
    """A selection over any dims of a scan.

    Examples
    --------
    >>> roi = ROI(y=(700, 1100), x=(800, 1200), chi=(0, 8))
    >>> roi.set("mu", (-0.2, 0.2))
    >>> roi.pixel_slices((2160, 2560))
    (slice(700, 1100, None), slice(800, 1200, None))
    """

    def __init__(
        self,
        ranges: Mapping[str, Sequence[float | None] | None] | None = None,
        index_dims: Iterable[str] = (),
        **kwargs: Sequence[float | None] | None,
    ) -> None:
        self.ranges: dict[str, Range] = {}
        self.index_dims: set[str] = set(index_dims) | set(PIXEL_DIMS)
        for name, rng in {**(ranges or {}), **kwargs}.items():
            self.set(name, rng)

    # ------------------------------------------------------------- editing
    def set(self, dim: str, rng: Sequence[float | None] | None, index: bool | None = None) -> ROI:
        """Set (or with ``None`` remove) the range of ``dim``; returns ``self``."""
        if rng is None:
            self.ranges.pop(dim, None)
            return self
        if len(rng) != 2:
            raise ValueError(f"range for {dim!r} must be (low, high); got {rng!r}")
        lo, hi = rng
        if lo is not None and hi is not None and hi < lo:
            raise ValueError(f"range for {dim!r} has high < low: {rng!r}")
        self.ranges[dim] = (lo, hi)
        if index is True:
            self.index_dims.add(dim)
        elif index is False:
            self.index_dims.discard(dim)
        return self

    def update(self, **kwargs: Sequence[float | None] | None) -> ROI:
        for dim, rng in kwargs.items():
            self.set(dim, rng)
        return self

    def clear(self, *dims: str) -> ROI:
        """Remove the given dims, or everything when called without arguments."""
        if not dims:
            self.ranges.clear()
        for dim in dims:
            self.ranges.pop(dim, None)
        return self

    def copy(self) -> ROI:
        return ROI(dict(self.ranges), index_dims=set(self.index_dims))

    def get(self, dim: str) -> Range | None:
        return self.ranges.get(dim)

    def __contains__(self, dim: object) -> bool:
        return dim in self.ranges

    def __bool__(self) -> bool:
        return bool(self.ranges)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ROI)
            and other.ranges == self.ranges
            and other.index_dims == self.index_dims
        )

    def __repr__(self) -> str:
        body = ", ".join(f"{d}={r}" for d, r in self.ranges.items())
        return f"ROI({body})"

    # ---------------------------------------------------------- resolution
    def pixel_slices(self, frame_shape: Sequence[int]) -> tuple[slice, slice]:
        """``(slice_y, slice_x)`` clipped to the frame."""
        h, w = int(frame_shape[0]), int(frame_shape[1])
        y0, y1 = self._clip(self.ranges.get("y"), h)
        x0, x1 = self._clip(self.ranges.get("x"), w)
        return slice(y0, y1), slice(x0, x1)

    def pixel_window(self, frame_shape: Sequence[int]) -> tuple[int, int, int, int]:
        """``(y0, y1, x0, x1)`` clipped to the frame."""
        ys, xs = self.pixel_slices(frame_shape)
        return ys.start, ys.stop, xs.start, xs.stop

    @staticmethod
    def _clip(rng: Range | None, size: int) -> tuple[int, int]:
        if rng is None:
            return 0, size
        lo = 0 if rng[0] is None else int(rng[0])
        hi = size if rng[1] is None else int(rng[1])
        lo, hi = max(lo, 0), min(hi, size)
        if hi <= lo:
            raise ValueError(f"pixel range {rng} is empty for a size of {size}")
        return lo, hi

    def motor_region(self, structure: Structure) -> tuple[slice, ...]:
        """One slice per motor dim of ``structure`` selecting the grid points inside the ROI."""
        region = []
        for dim in structure.motor_dims:
            rng = self.ranges.get(dim)
            if rng is None:
                region.append(slice(None))
                continue
            coords = np.asarray(structure.coords[dim])
            if dim in self.index_dims:
                first = 0 if rng[0] is None else int(rng[0])
                last = len(coords) if rng[1] is None else int(rng[1])
                region.append(slice(max(first, 0), min(last, len(coords))))
                continue
            lo = -np.inf if rng[0] is None else float(rng[0])
            hi = np.inf if rng[1] is None else float(rng[1])
            inside = np.flatnonzero((coords >= lo) & (coords <= hi))
            if inside.size == 0:
                raise ValueError(
                    f"no {dim} grid point inside {rng}; {dim} spans "
                    f"{coords.min():.5g} to {coords.max():.5g}"
                )
            region.append(slice(int(inside[0]), int(inside[-1]) + 1))
        return tuple(region)

    def from_downsampled(self, factors: Mapping[str, int] | Sequence[int]) -> ROI:
        """Scale index ranges given in downsampled coordinates back to full resolution.

        ``factors`` maps dims to their downsample factor, or is ``(ds_y, ds_x)``.
        """
        if not isinstance(factors, Mapping):
            factors = {"y": int(factors[0]), "x": int(factors[1])}
        scaled = self.copy()
        for dim, f in factors.items():
            rng = scaled.ranges.get(dim)
            if rng is None or dim not in scaled.index_dims or f == 1:
                continue
            lo = None if rng[0] is None else int(rng[0]) * int(f)
            hi = None if rng[1] is None else int(rng[1]) * int(f)
            scaled.ranges[dim] = (lo, hi)
        return scaled

    # ------------------------------------------------------------- export
    def to_dict(self) -> dict[str, Any]:
        return {
            "ranges": {d: list(r) for d, r in self.ranges.items()},
            "index_dims": sorted(self.index_dims - set(PIXEL_DIMS)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> ROI:
        if not data:
            return cls()
        if "ranges" in data:
            return cls(data["ranges"], index_dims=data.get("index_dims", ()))
        return cls(data)

    @classmethod
    def parse(cls, text: str | None) -> ROI:
        """``"y=700:1100,x=800:1200,chi=-0.2:0.2"`` to an ROI.

        Pixel ranges are integers, motor ranges values; a missing bound
        (``x=800:``) means "to the edge". Empty text gives an empty ROI.
        """
        roi = cls()
        for part in (text or "").split(","):
            if not part.strip():
                continue
            if "=" not in part or ":" not in part:
                raise ValueError(f"cannot parse {part!r}; expected dim=low:high")
            dim, rng = part.split("=", 1)
            lo, hi = rng.split(":", 1)
            dim = dim.strip()
            convert = int if dim in PIXEL_DIMS else float
            roi.set(dim, (convert(lo) if lo.strip() else None, convert(hi) if hi.strip() else None))
        return roi

    @classmethod
    def from_indices(
        cls,
        image: Any,
        y: Sequence[float | None] | None = None,
        x: Sequence[float | None] | None = None,
        **motors: Sequence[float | None] | None,
    ) -> ROI:
        """An ROI given in the pixel indices of ``image``, in full-resolution pixels.

        ``image`` is a preview or a windowed result whose ``y``/``x``
        coordinates are full-resolution pixel indices, as bexa writes them; the
        first coordinate and the step map the indices back. Plain arrays are
        taken as full resolution. Motor ranges pass through unchanged.
        """
        roi = cls(dict(motors))
        for dim, rng in (("y", y), ("x", x)):
            if rng is not None:
                roi.set(dim, _scale_range(image, dim, rng))
        return roi

    def to_indices(self, image: Any) -> tuple[slice, slice]:
        """``(slice_y, slice_x)`` of this ROI in the pixel indices of ``image``."""
        height, width = np.shape(image)[-2:]
        slices = []
        for dim, size in (("y", height), ("x", width)):
            rng = self.ranges.get(dim)
            if rng is None:
                slices.append(slice(0, size))
                continue
            origin, step = _axis_of(image, dim)
            lo = 0 if rng[0] is None else round((float(rng[0]) - origin) / step)
            hi = size if rng[1] is None else round((float(rng[1]) - origin) / step)
            slices.append(slice(max(lo, 0), min(hi, size)))
        return slices[0], slices[1]

    def describe(self, structure: Structure | None = None) -> str:
        if not self.ranges:
            return "ROI: everything"
        parts = []
        for dim, (lo, hi) in self.ranges.items():
            unit = "" if dim in self.index_dims else " (values)"
            parts.append(f"{dim} {lo} to {hi}{unit}")
        text = "ROI: " + "; ".join(parts)
        if structure is not None:
            try:
                region = self.motor_region(structure)
                kept = structure.sub(region)
                text += f" -> grid {kept.motor_shape}, {len(kept.frame_ids())} frames"
            except ValueError as exc:
                text += f" (invalid: {exc})"
        return text


def _axis_of(image: Any, dim: str) -> tuple[float, float]:
    """Origin and step of the ``dim`` coordinate of ``image`` (index units for plain arrays)."""
    coords = getattr(image, "coords", None)
    if coords is not None and dim in coords and len(coords[dim]) > 0:
        values = np.asarray(coords[dim].values, dtype=float)
        step = float(values[1] - values[0]) if len(values) > 1 else 1.0
        return float(values[0]), step
    return 0.0, 1.0


def _scale_range(image: Any, dim: str, rng: Sequence[float | None]) -> Range:
    origin, step = _axis_of(image, dim)
    lo = None if rng[0] is None else round(origin + step * float(rng[0]))
    hi = None if rng[1] is None else round(origin + step * float(rng[1]))
    return lo, hi
