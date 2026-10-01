"""Several scans read as one source with a leading dim (energy series, z-stacks)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from bexa.core.provenance import to_json
from bexa.core.structure import Structure
from bexa.io.base import BaseSource, Source


class MultiScanSource(BaseSource):
    """Concatenate scans with identical grids along a new dim.

    Frame ids are global: the frames of part ``i`` follow those of part ``i - 1``.

    Parameters
    ----------
    parts
        The individual sources, in the order of ``coords``.
    dim
        Name of the new leading dim (``"energy"``, ``"z"``, ``"scan"``).
    coords
        One coordinate value per part.
    """

    name = "multi"

    def __init__(self, parts: Sequence[Source], dim: str, coords: Sequence[float]) -> None:
        if not parts:
            raise ValueError("MultiScanSource needs at least one part")
        super().__init__(getattr(parts[0], "spec", None))
        self.parts = list(parts)
        self.dim = dim
        self.coords = np.asarray(coords, dtype=float)
        if len(self.coords) != len(self.parts):
            raise ValueError("one coordinate per part is required")
        self._offsets = np.cumsum([0, *(p.n_frames for p in self.parts)])
        self._structure: Structure | None = None

    def files(self) -> list[Path]:
        seen: dict[Path, None] = {}
        for p in self.parts:
            for f in p.files():
                seen.setdefault(f, None)
        return list(seen)

    def cache_records(self) -> list[dict[str, Any]]:
        """The records of every part, each once, plus the dim and its coordinates."""
        from bexa.core.provenance import file_records

        seen: dict[str, dict[str, Any]] = {}
        for p in self.parts:
            records = p.cache_records() if hasattr(p, "cache_records") else file_records(p.files())
            for record in records:
                seen.setdefault(to_json(record), record)
        stacking = {"path": "", "dim": self.dim, "coords": [float(c) for c in self.coords]}
        return [*seen.values(), stacking]

    @property
    def frame_shape(self) -> tuple[int, int]:
        return self.parts[0].frame_shape

    @property
    def n_frames(self) -> int:
        return int(self._offsets[-1])

    @property
    def dtype(self) -> np.dtype:
        return self.parts[0].dtype

    @property
    def chunk_frames(self) -> int:
        return getattr(self.parts[0], "chunk_frames", 1)

    def read_frames(
        self,
        index: np.ndarray | slice,
        y: slice = slice(None),
        x: slice = slice(None),
        dtype: Any = None,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        ids = self.normalise_index(index, self.n_frames)
        h, w = self.frame_shape
        ny = len(range(*y.indices(h)))
        nx = len(range(*x.indices(w)))
        target = np.dtype(dtype) if dtype is not None else self.dtype
        if out is None:
            out = np.empty((ids.size, ny, nx), dtype=target)
        pos = 0
        for part, start in zip(self.parts, self._offsets[:-1], strict=False):
            local = ids[(ids >= start) & (ids < start + part.n_frames)] - start
            if local.size == 0:
                continue
            part.read_frames(local, y, x, dtype=target, out=out[pos : pos + local.size])
            pos += local.size
        return out

    def structure(self) -> Structure:
        if self._structure is None:
            self._structure = Structure.stack(
                [p.structure() for p in self.parts], self.dim, self.coords
            )
        return self._structure

    def scalars(self) -> dict[str, float]:
        return dict(self.parts[0].scalars())

    def refresh(self) -> MultiScanSource:
        for p in self.parts:
            p.refresh()
        self._offsets = np.cumsum([0, *(p.n_frames for p in self.parts)])
        self._structure = None
        return self

    def close(self) -> None:
        for p in self.parts:
            p.close()
