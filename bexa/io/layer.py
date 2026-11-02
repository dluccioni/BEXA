"""A scan's frames on a grid of its own: one layer of the scan, or the scan reshaped.

A z-scan (``uz x chi x mu``) holds a mosaicity scan at every height. Its layer
at one ``uz`` point has the grid ``chi x mu`` and ``uz`` as a fixed
positioner, so the layers of one scan stack like the scans of a z-stack
(:meth:`bexa.core.scan.Scan.layers`). A frame series (``frame x y x x``) taken
as a raster is that raster reshaped (``a x b x y x x``, :meth:`bexa.core.scan.Scan.reshaped`).
Either reads the frames through the scan it comes from, and its cache records
say what it is, so no two of them share a cached result.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bexa.core.status import ScanStatus
from bexa.core.structure import Structure
from bexa.io.base import BaseSource, Source

__all__ = ["GridSource", "LayerSource", "ReshapedSource"]

# the labels of a scan that its layers keep: the scan number and the dataset name
LABELS = ("scan", "dataset")


class GridSource(BaseSource):
    """The frames of ``parent``, read through it, on the grid :meth:`regrid` makes of its own.

    Subclasses implement :meth:`regrid` (the new structure) and :meth:`record` (what
    the cache keys add to the parent's records).
    """

    name = "grid"

    def __init__(self, parent: Source) -> None:
        super().__init__(getattr(parent, "spec", None))
        self.parent = parent
        self._structure: Structure | None = None  # made by the subclass, again on refresh
        for label in LABELS:  # labelled as its scan is
            if hasattr(parent, label):
                setattr(self, label, getattr(parent, label))

    def regrid(self, whole: Structure) -> Structure:
        raise NotImplementedError

    def record(self) -> dict[str, Any]:
        raise NotImplementedError

    def files(self) -> list[Path]:
        return self.parent.files()

    def cache_records(self) -> list[dict[str, Any]]:
        """The records of the scan, then what this grid is."""
        from bexa.core.provenance import file_records

        parent = self.parent
        records = (
            parent.cache_records()
            if hasattr(parent, "cache_records")
            else file_records(parent.files())
        )
        return [*records, {"path": "", **self.record()}]

    @property
    def frame_shape(self) -> tuple[int, int]:
        return self.parent.frame_shape

    @property
    def n_frames(self) -> int:
        return self.parent.n_frames

    @property
    def dtype(self) -> np.dtype:
        return self.parent.dtype

    @property
    def chunk_frames(self) -> int:
        return getattr(self.parent, "chunk_frames", 1)

    def read_frames(
        self,
        index: np.ndarray | slice,
        y: slice = slice(None),
        x: slice = slice(None),
        dtype: Any = None,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.parent.read_frames(index, y, x, dtype=dtype, out=out)

    def structure(self) -> Structure:
        if self._structure is None:
            self._structure = self.regrid(self.parent.structure())
        return self._structure

    @property
    def status(self) -> ScanStatus:
        """The status of the scan it comes from."""
        return getattr(self.parent, "status", ScanStatus.COMPLETE)

    def aux_tables(self) -> dict[str, pd.DataFrame]:
        return self.parent.aux_tables() if hasattr(self.parent, "aux_tables") else {}

    def refresh(self) -> GridSource:
        """Re-read the scan (a scan still being written) and the grid made of it."""
        self.parent.refresh()
        self._structure = None
        return self

    def close(self) -> None:
        """Close the scan's files; they open again at the next read."""
        self.parent.close()


class LayerSource(GridSource):
    """The frames of ``parent`` at point ``index`` of its motor dim ``dim``.

    Parameters
    ----------
    parent
        The source of the whole scan; it reads the frames.
    dim
        A motor dim of the parent (``"uz"``, ``"energy"``, ``"chi"``).
    index
        The point along ``dim`` (negative counts from the end).
    """

    name = "layer"

    def __init__(self, parent: Source, dim: str, index: int) -> None:
        super().__init__(parent)
        whole = parent.structure()
        self.dim = str(dim)
        self.index = int(index) % whole.motor_shape[whole.motor_dims.index(dim)]
        self.value = float(whole.coords[dim][self.index])
        self._structure = self.regrid(whole)

    def regrid(self, whole: Structure) -> Structure:
        return whole.layer(self.dim, self.index)

    def record(self) -> dict[str, Any]:
        """The layer: the dim, the point and its value."""
        return {"layer": self.dim, "index": self.index, "value": self.value}


class ReshapedSource(GridSource):
    """The frames of ``parent`` with its motor dim ``dim`` split into new dims (a raster).

    Parameters
    ----------
    parent
        The source of the whole scan; it reads the frames.
    dim
        A motor dim of the parent (``"frame"`` for a frame series).
    axes
        The new dims, slow first, as ``(name, first, last, points)``: each runs evenly
        from ``first`` to ``last`` (see :meth:`bexa.core.structure.Structure.split`).
    snake
        The last new dim runs backwards on every other row (a serpentine raster).
    """

    name = "reshaped"

    def __init__(
        self,
        parent: Source,
        dim: str,
        axes: Sequence[tuple[str, float, float, int]],
        snake: bool = False,
    ) -> None:
        super().__init__(parent)
        self.dim = str(dim)
        self.axes = [(str(n), float(a), float(b), int(k)) for n, a, b, k in axes]
        self.snake = bool(snake)
        self._structure = self.regrid(parent.structure())

    def regrid(self, whole: Structure) -> Structure:
        values = [(name, np.linspace(a, b, k)) for name, a, b, k in self.axes]
        return whole.split(self.dim, values, self.snake)

    def record(self) -> dict[str, Any]:
        """The reshape: the dim split, the new dims and their values, the raster's order."""
        return {"reshape": self.dim, "into": [list(a) for a in self.axes], "snake": self.snake}
