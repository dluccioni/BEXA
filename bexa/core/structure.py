"""Structure of a scan: named motor dims, their coordinates, and which frame sits where.

A detector writes frames in acquisition order. ``Structure`` maps that flat frame
list onto a grid of motor positions (``frame_index``), so every consumer can say
"the frame at chi index 3, mu index 17" without knowing whether the scan was
slow-major, snake-ordered or cyclic. Missing frames of a partial scan hold -1.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

PIXEL_DIMS: tuple[str, str] = ("y", "x")
MISSING = -1


def _pair(frame_shape: Sequence[int]) -> tuple[int, int]:
    """Validate and normalise a ``(height, width)`` pair."""
    if len(frame_shape) != 2:
        raise ValueError(f"frame_shape must be (height, width); got {tuple(frame_shape)}")
    return int(frame_shape[0]), int(frame_shape[1])


@dataclass
class Structure:
    """Dims, coordinates and frame-to-grid mapping of one scan.

    Attributes
    ----------
    motor_dims
        Names of the scanned dims, slow first, for example ``("chi", "mu")``.
        A plain frame sequence uses ``("frame",)``.
    motor_shape
        Number of grid points along each motor dim.
    coords
        1-D coordinate values per motor dim (length ``motor_shape[i]``).
    frame_index
        Integer array of shape ``motor_shape``: the flat frame id at each grid
        position, or ``MISSING`` (-1) where a partial scan has no frame.
    frame_shape
        ``(height, width)`` of one detector frame.
    n_frames
        Number of frames the source can deliver.
    per_frame
        Per-frame channels of length ``n_frames`` (motor readbacks and any other
        per-frame value), keyed by name.
    scalars
        Values that do not change during the scan (fixed positioners, energy).
    scan_type
        ``"fscan1d"``, ``"fscan2d"``, ``"fscan3d"``, ``"list"``, ``"points"`` or ``"cube"``.
    energy_keV
        Photon energy when known.
    title
        The command as recorded by the control system (``fscan2d chi ... mu ...``).
    """

    motor_dims: tuple[str, ...]
    motor_shape: tuple[int, ...]
    coords: dict[str, np.ndarray]
    frame_index: np.ndarray
    frame_shape: tuple[int, int]
    n_frames: int
    per_frame: dict[str, np.ndarray] = field(default_factory=dict)
    scalars: dict[str, float] = field(default_factory=dict)
    scan_type: str = "grid"
    units: dict[str, str] = field(default_factory=dict)
    energy_keV: float | None = None
    title: str = ""
    _grid_of_frame: np.ndarray | None = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------ basics
    @property
    def dims(self) -> tuple[str, ...]:
        return self.motor_dims + PIXEL_DIMS

    @property
    def shape(self) -> tuple[int, ...]:
        return self.motor_shape + self.frame_shape

    @property
    def ndim(self) -> int:
        return len(self.dims)

    @property
    def n_grid(self) -> int:
        return int(np.prod(self.motor_shape)) if self.motor_shape else 1

    @property
    def is_complete(self) -> bool:
        """True when every grid position has a frame."""
        return bool((self.frame_index >= 0).all())

    @property
    def n_missing(self) -> int:
        return int((self.frame_index < 0).sum())

    def __post_init__(self) -> None:
        self.frame_index = np.asarray(self.frame_index, dtype=np.int64).reshape(self.motor_shape)
        for name in self.motor_dims:
            if name not in self.coords:
                raise ValueError(f"no coordinates for dim {name!r}")
            self.coords[name] = np.asarray(self.coords[name])
        if tuple(len(self.coords[d]) for d in self.motor_dims) != tuple(self.motor_shape):
            raise ValueError("coordinate lengths do not match motor_shape")

    # ---------------------------------------------------------------- lookups
    def grid_of_frame(self) -> np.ndarray:
        """``(n_frames, len(motor_dims))`` grid position of each frame (-1 for unused frames)."""
        if self._grid_of_frame is None:
            table = np.full((self.n_frames, len(self.motor_dims)), MISSING, dtype=np.int64)
            used = self.frame_index >= 0
            positions = np.argwhere(used)
            ids = self.frame_index[used]
            valid = ids < self.n_frames
            table[ids[valid]] = positions[valid]
            self._grid_of_frame = table
        return self._grid_of_frame

    def grid_positions(self, frame_ids: np.ndarray) -> tuple[np.ndarray, ...]:
        """Index arrays, one per motor dim, for the given flat frame ids."""
        table = self.grid_of_frame()[np.asarray(frame_ids, dtype=np.int64)]
        return tuple(table[:, i] for i in range(table.shape[1]))

    def motor_values(self, name: str, frame_ids: np.ndarray) -> np.ndarray:
        """Coordinate value of motor ``name`` for each frame id.

        Uses the per-frame readback when present (exact positions), otherwise the
        grid coordinate of the frame's position.
        """
        frame_ids = np.asarray(frame_ids, dtype=np.int64)
        if name in self.per_frame:
            return np.asarray(self.per_frame[name])[frame_ids]
        axis = self.motor_dims.index(name)
        return self.coords[name][self.grid_positions(frame_ids)[axis]]

    def frame_ids(self, region: Sequence[slice | np.ndarray] | None = None) -> np.ndarray:
        """Sorted flat frame ids inside a grid region (all frames when ``region`` is None)."""
        sub = self.frame_index if region is None else self.frame_index[tuple(region)]
        ids = sub[sub >= 0]
        return np.unique(ids)

    def used_frame_ids(self) -> np.ndarray:
        return self.frame_ids()

    # ------------------------------------------------------------- derivations
    def sub(self, region: Sequence[slice | np.ndarray]) -> Structure:
        """Restrict the grid to ``region`` (one slice or index array per motor dim)."""
        region = tuple(region)
        if len(region) != len(self.motor_dims):
            raise ValueError("one slice per motor dim is required")
        index = self.frame_index[region]
        coords = {d: self.coords[d][r] for d, r in zip(self.motor_dims, region, strict=True)}
        return replace(
            self,
            motor_shape=index.shape,
            coords=coords,
            frame_index=index,
            _grid_of_frame=None,
        )

    def with_frame_shape(self, frame_shape: Sequence[int]) -> Structure:
        return replace(self, frame_shape=_pair(frame_shape), _grid_of_frame=None)

    def describe(self) -> str:
        """Multi-line human summary used by ``bexa info``."""
        lines = [
            f"scan type {self.scan_type}: dims {self.dims}, shape {self.shape}, "
            f"{self.n_frames} frames"
            + (f", {self.n_missing} grid points missing" if self.n_missing else ""),
        ]
        if self.title:
            lines.append(f"  title: {self.title}")
        for d in self.motor_dims:
            c = self.coords[d]
            unit = self.units.get(d, "")
            if len(c) > 1:
                lines.append(
                    f"  {d}: {len(c)} points, {c.min():.5g} to {c.max():.5g} {unit}".rstrip()
                )
            else:
                lines.append(f"  {d}: {c[0]:.5g} {unit}".rstrip())
        if self.energy_keV is not None:
            lines.append(f"  energy: {self.energy_keV:.4f} keV")
        for name, value in self.scalars.items():
            if name not in self.motor_dims:
                lines.append(f"  {name} = {value:.6g}")
        return "\n".join(lines)

    # ----------------------------------------------------------- constructors
    @classmethod
    def frames_only(cls, n_frames: int, frame_shape: tuple[int, int], **kwargs: Any) -> Structure:
        """A plain sequence of frames with a ``frame`` dim."""
        return cls(
            motor_dims=("frame",),
            motor_shape=(n_frames,),
            coords={"frame": np.arange(n_frames)},
            frame_index=np.arange(n_frames),
            frame_shape=_pair(frame_shape),
            n_frames=n_frames,
            scan_type=kwargs.pop("scan_type", "list"),
            **kwargs,
        )

    @classmethod
    def from_grid(
        cls,
        motor_dims: Sequence[str],
        coords: dict[str, np.ndarray],
        frame_shape: tuple[int, int],
        n_frames: int | None = None,
        **kwargs: Any,
    ) -> Structure:
        """A regular grid acquired slow-major (first dim outermost), possibly partial."""
        motor_dims = tuple(motor_dims)
        shape = tuple(len(np.asarray(coords[d])) for d in motor_dims)
        total = int(np.prod(shape))
        if n_frames is None:
            n_frames = total
        index = np.arange(total)
        index[index >= n_frames] = MISSING
        return cls(
            motor_dims=motor_dims,
            motor_shape=shape,
            coords={d: np.asarray(coords[d]) for d in motor_dims},
            frame_index=index.reshape(shape),
            frame_shape=_pair(frame_shape),
            n_frames=n_frames,
            **kwargs,
        )

    @classmethod
    def from_per_frame(
        cls,
        motors: dict[str, np.ndarray],
        frame_shape: tuple[int, int],
        tolerance: float | dict[str, float] | None = None,
        order: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> Structure:
        """Detect the grid from per-frame motor readbacks.

        Works for slow-major, snake-ordered and cyclic acquisitions because
        frames are placed by value, not by order. The slow dim is the motor whose
        value changes least often between consecutive frames unless ``order``
        is given. Frames whose position repeats keep the first occurrence.

        Parameters
        ----------
        motors
            Per-frame readbacks, all of the same length.
        tolerance
            Values closer than this are the same grid point. Default: 1e-3 of
            the motor range (or 1e-6 for a constant motor).
        """
        names = list(motors)
        lengths = {len(np.asarray(v)) for v in motors.values()}
        if len(lengths) != 1:
            raise ValueError(f"motor arrays have different lengths: {lengths}")
        n_frames = lengths.pop()
        labels: dict[str, np.ndarray] = {}
        centers: dict[str, np.ndarray] = {}
        changes: dict[str, int] = {}
        for name in names:
            values = np.asarray(motors[name], dtype=float)
            tol = tolerance.get(name) if isinstance(tolerance, dict) else tolerance
            centers[name], labels[name] = _group_values(values, tol)
            changes[name] = int(np.count_nonzero(np.diff(labels[name])))
        if order is None:
            # slow motors change rarely; ties keep the given order
            order = sorted(names, key=lambda n: changes[n])
        motor_dims = tuple(order)
        shape = tuple(len(centers[d]) for d in motor_dims)
        index = np.full(shape, MISSING, dtype=np.int64)
        positions = tuple(labels[d] for d in motor_dims)
        # first occurrence wins: iterate in reverse so earlier frames overwrite later ones
        for fid in range(n_frames - 1, -1, -1):
            index[tuple(p[fid] for p in positions)] = fid
        scan_type = kwargs.pop("scan_type", f"fscan{len(motor_dims)}d")
        return cls(
            motor_dims=motor_dims,
            motor_shape=shape,
            coords={d: centers[d] for d in motor_dims},
            frame_index=index,
            frame_shape=_pair(frame_shape),
            n_frames=n_frames,
            per_frame={n: np.asarray(motors[n], dtype=float) for n in names},
            scan_type=scan_type,
            **kwargs,
        )

    @classmethod
    def stack(
        cls, parts: Sequence[Structure], dim: str, coords: np.ndarray | Sequence[float]
    ) -> Structure:
        """Stack scans with identical inner grids along a new leading dim (energy series)."""
        if not parts:
            raise ValueError("nothing to stack")
        first = parts[0]
        for p in parts[1:]:
            if p.motor_dims != first.motor_dims or p.motor_shape != first.motor_shape:
                raise ValueError(
                    "cannot stack scans with different grids: "
                    f"{first.motor_shape} vs {p.motor_shape}"
                )
            if p.frame_shape != first.frame_shape:
                raise ValueError("cannot stack scans with different frame shapes")
        offsets = np.cumsum([0] + [p.n_frames for p in parts[:-1]])
        index = np.stack(
            [
                np.where(p.frame_index >= 0, p.frame_index + off, MISSING)
                for p, off in zip(parts, offsets, strict=True)
            ]
        )
        per_frame: dict[str, np.ndarray] = {}
        for name in first.per_frame:
            if all(name in p.per_frame for p in parts):
                per_frame[name] = np.concatenate([p.per_frame[name] for p in parts])
        coords_all = {dim: np.asarray(coords, dtype=float)}
        coords_all.update({d: first.coords[d] for d in first.motor_dims})
        stacked = cls(
            motor_dims=(dim, *first.motor_dims),
            motor_shape=(len(parts), *first.motor_shape),
            coords=coords_all,
            frame_index=index,
            frame_shape=first.frame_shape,
            n_frames=int(sum(p.n_frames for p in parts)),
            per_frame=per_frame,
            scalars=dict(first.scalars),
            scan_type="multi",
            units=dict(first.units),
            energy_keV=first.energy_keV,
            title=first.title,
        )
        return stacked


def _group_values(values: np.ndarray, tolerance: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Cluster 1-D values into grid points; returns (sorted centers, label per value)."""
    if tolerance is None:
        span = float(np.nanmax(values) - np.nanmin(values)) if len(values) else 0.0
        tolerance = 1e-3 * span if span > 0 else 1e-6
    order = np.argsort(values, kind="stable")
    sorted_vals = values[order]
    new_group = np.empty(len(values), dtype=bool)
    new_group[0] = True
    new_group[1:] = np.diff(sorted_vals) > tolerance
    group_sorted = np.cumsum(new_group) - 1
    labels = np.empty(len(values), dtype=np.int64)
    labels[order] = group_sorted
    n_groups = int(group_sorted[-1]) + 1 if len(values) else 0
    sums = np.bincount(labels, weights=values, minlength=n_groups)
    counts = np.bincount(labels, minlength=n_groups)
    centers = sums / np.maximum(counts, 1)
    return centers, labels
