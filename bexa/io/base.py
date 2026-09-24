"""The ``Source`` protocol implemented by every engine, and shared helpers.

An engine knows how to walk one kind of file structure. Everything above it
(``Scan``, the reductions engine, the plots) only sees this interface: how many
frames there are, their shape, how to read a set of them with a pixel window,
and the per-frame and scalar metadata.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:  # avoid an import cycle at runtime
    import pandas as pd

    from bexa.core.structure import Structure
    from bexa.io.formats import FormatSpec


@dataclass
class FrameBatch:
    """A block of frames read from a source.

    Attributes
    ----------
    frames
        Array ``(n, height, width)`` on the host or on the device.
    frame_ids
        Flat frame ids of the rows of ``frames``.
    y, x
        Pixel window the frames were cut to.
    """

    frames: Any
    frame_ids: np.ndarray
    y: slice
    x: slice

    def __len__(self) -> int:
        return len(self.frame_ids)


@runtime_checkable
class Source(Protocol):
    """What ``Scan`` and the reductions engine need from a reader."""

    name: str

    def files(self) -> list[Path]: ...

    @property
    def frame_shape(self) -> tuple[int, int]: ...

    @property
    def n_frames(self) -> int: ...

    @property
    def dtype(self) -> np.dtype: ...

    def read_frames(
        self,
        index: np.ndarray | slice,
        y: slice = slice(None),
        x: slice = slice(None),
        dtype: Any = None,
        out: np.ndarray | None = None,
    ) -> np.ndarray: ...

    def structure(self) -> Structure: ...

    def scalars(self) -> dict[str, float]: ...

    def aux_tables(self) -> dict[str, pd.DataFrame]: ...

    @property
    def energy_keV(self) -> float | None: ...

    @property
    def chunk_frames(self) -> int: ...

    def refresh(self) -> Any: ...

    def close(self) -> None: ...


class BaseSource(ABC):
    """Base class with the boring parts of :class:`Source` filled in."""

    name: str = "base"

    def __init__(self, spec: FormatSpec | None = None) -> None:
        self.spec = spec

    # ---- required -------------------------------------------------------
    @abstractmethod
    def files(self) -> list[Path]:
        """Every file the source reads (for provenance and cache fingerprints)."""

    @property
    @abstractmethod
    def frame_shape(self) -> tuple[int, int]:
        """(height, width) of one frame."""

    @property
    @abstractmethod
    def n_frames(self) -> int:
        """Number of frames available."""

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """Stored dtype of the frames."""

    @abstractmethod
    def read_frames(
        self,
        index: np.ndarray | slice,
        y: slice = slice(None),
        x: slice = slice(None),
        dtype: Any = None,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Read the frames ``index`` (sorted ids or a slice) cut to the pixel window."""

    @abstractmethod
    def structure(self) -> Structure:
        """Dims, coordinates and frame mapping."""

    # ---- optional --------------------------------------------------------
    def scalars(self) -> dict[str, float]:
        return dict(self.structure().scalars)

    def aux_tables(self) -> dict[str, pd.DataFrame]:
        return {}

    @property
    def energy_keV(self) -> float | None:
        return self.structure().energy_keV

    @property
    def chunk_frames(self) -> int:
        """Frames per storage chunk; batches are multiples of this when possible."""
        return 1

    def refresh(self) -> BaseSource:
        """Re-read file metadata (for scans still being written)."""
        return self

    def close(self) -> None:
        return None

    def __enter__(self) -> BaseSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- helpers ---------------------------------------------------------
    def frame_bytes(self, y: slice = slice(None), x: slice = slice(None), dtype: Any = None) -> int:
        """Bytes of one frame inside the window at ``dtype`` (default: stored dtype)."""
        h, w = self.frame_shape
        ny = len(range(*y.indices(h)))
        nx = len(range(*x.indices(w)))
        itemsize = np.dtype(dtype).itemsize if dtype is not None else self.dtype.itemsize
        return ny * nx * itemsize

    @staticmethod
    def contiguous_runs(index: np.ndarray) -> Iterator[tuple[int, int]]:
        """Yield ``(start, stop)`` for runs of consecutive ids in a sorted index array."""
        index = np.asarray(index, dtype=np.int64)
        if index.size == 0:
            return
        breaks = np.flatnonzero(np.diff(index) != 1) + 1
        starts = np.concatenate(([0], breaks))
        stops = np.concatenate((breaks, [index.size]))
        for s, e in zip(starts, stops, strict=True):
            yield int(index[s]), int(index[e - 1]) + 1

    @staticmethod
    def normalise_index(index: np.ndarray | slice, n_frames: int) -> np.ndarray:
        """Turn a slice or id array into a sorted, unique, bounds-checked id array."""
        if isinstance(index, slice):
            return np.arange(*index.indices(n_frames), dtype=np.int64)
        ids = np.unique(np.asarray(index, dtype=np.int64))
        if ids.size and (ids[0] < 0 or ids[-1] >= n_frames):
            raise IndexError(f"frame ids must be in [0, {n_frames}); got {ids[0]}..{ids[-1]}")
        return ids
