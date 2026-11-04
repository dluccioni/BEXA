"""A scan's frames on a grid of its own: one layer of the scan, the scan reshaped or renamed.

A z-scan (``uz x chi x mu``) holds a mosaicity scan at every height. Its layer
at one ``uz`` point has the grid ``chi x mu`` and ``uz`` as a fixed
positioner, so the layers of one scan stack like the scans of a z-stack
(:meth:`bexa.core.scan.Scan.layers`). A frame series (``frame x y x x``) taken
as a raster is that raster reshaped (``a x b x y x x``, :meth:`bexa.core.scan.Scan.reshaped`).
A scan whose motors, counters and detectors take names of their own, and whose
values take offsets and scales (``y_sample`` for ``nsamstageY``, its zero
moved), is the scan renamed (:meth:`bexa.core.scan.Scan.renamed`). Each reads
the frames through the scan it comes from, and its cache records say what it
is, so no two of them share a cached result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bexa.core.status import ScanStatus
from bexa.core.structure import Structure
from bexa.io.base import BaseSource, Source

__all__ = ["GridSource", "LayerSource", "RenamedSource", "ReshapedSource"]

# the labels of a scan that its layers keep: the scan number, the dataset, the detector
LABELS = ("scan", "dataset", "detector")
PIXEL_DIMS = ("y", "x")


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

    def detector_names(self) -> list[str]:
        parent = self.parent
        return list(parent.detector_names()) if hasattr(parent, "detector_names") else []

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

    def file_metadata(self) -> dict[str, pd.DataFrame]:
        """What the scan's files hold: the files are the parent's."""
        hook = getattr(self.parent, "file_metadata", None)
        return hook() if callable(hook) else {}

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


class RenamedSource(GridSource):
    """The scan of ``parent`` with names of its own and its values moved, its frames the parent's.

    ``names`` maps a motor, channel, position or detector to its alias
    (``{"nsamstageY": "y_sample", "7idd:scaler1.S2": "I0"}``); ``offsets`` and
    ``scales``, by the names the files give, make a value ``scale * value + offset``
    (a motor read against another zero, a counter in other units, a sign turned).
    The detector pixels ``y`` and ``x`` keep their names. An alias that is already
    another name of the scan raises ``ValueError``.
    """

    name = "renamed"

    def __init__(
        self,
        parent: Source,
        names: Mapping[str, str] | None = None,
        offsets: Mapping[str, float] | None = None,
        scales: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__(parent)
        self.names = {
            str(k): str(v) for k, v in (names or {}).items() if str(v) and str(v) != str(k)
        }
        self.offsets = {str(k): float(v) for k, v in (offsets or {}).items() if float(v) != 0.0}
        self.scales = {str(k): float(v) for k, v in (scales or {}).items() if float(v) != 1.0}
        for pixel in PIXEL_DIMS:
            if pixel in self.names:
                raise ValueError(f"the detector pixels {pixel!r} keep their name")
        detector = getattr(parent, "detector", None)
        if isinstance(detector, str):
            self.detector = self.names.get(detector, detector)

    def alias(self, name: str) -> str:
        """The name ``name`` has here (itself without an alias)."""
        return self.names.get(name, name)

    def moved(self, name: str, values: Any) -> Any:
        """``values`` of ``name`` (as the files give them) as this scan gives them."""
        scale, offset = self.scales.get(name, 1.0), self.offsets.get(name, 0.0)
        if scale == 1.0 and offset == 0.0:
            return values
        return np.asarray(values, dtype=float) * scale + offset

    def regrid(self, whole: Structure) -> Structure:
        taken = set(whole.coords) | set(whole.per_frame) | set(whole.scalars)
        clashes = {
            a for k, a in self.names.items() if a in taken and a not in self.names and k != a
        }
        if clashes:
            raise ValueError(f"aliases already names of the scan: {sorted(clashes)}")
        return replace(
            whole,
            motor_dims=tuple(self.alias(d) for d in whole.motor_dims),
            coords={self.alias(d): self.moved(d, v) for d, v in whole.coords.items()},
            per_frame={self.alias(k): self.moved(k, v) for k, v in whole.per_frame.items()},
            scalars={self.alias(k): float(self.moved(k, v)) for k, v in whole.scalars.items()},
            units={self.alias(k): u for k, u in whole.units.items()},
            motor_names=frozenset(self.alias(n) for n in whole.motor_names),
        )

    def record(self) -> dict[str, Any]:
        return {"names": self.names, "offsets": self.offsets, "scales": self.scales}

    def detector_names(self) -> list[str]:
        return [self.alias(d) for d in super().detector_names()]

    def read_frames(
        self,
        index: np.ndarray | slice,
        y: slice = slice(None),
        x: slice = slice(None),
        dtype: Any = None,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """The parent's frames; those of a counter read as a detector (a netCDF run's: a value
        per point) moved as the counter is."""
        frames = self.parent.read_frames(index, y, x, dtype=dtype, out=out)
        name = getattr(self.parent, "detector", None)
        moved = name in self.scales or name in self.offsets
        if moved and name in self.parent.structure().per_frame:  # a counter, not a camera
            frames *= self.scales.get(name, 1.0)
            frames += self.offsets.get(name, 0.0)
        return frames

    def _table(self, table: pd.DataFrame) -> pd.DataFrame:
        out = table.copy()
        for column in table.columns:
            if column in self.offsets or column in self.scales:
                out[column] = self.moved(str(column), table[column].to_numpy())
        return out.rename(columns=self.names)

    def point_table(self) -> pd.DataFrame:
        """The parent's table of points with the names and values of this scan."""
        return self._table(self.parent.point_table())  # type: ignore[attr-defined]

    def aux_tables(self) -> dict[str, pd.DataFrame]:
        return {name: self._table(table) for name, table in self.parent.aux_tables().items()}
