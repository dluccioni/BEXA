"""One layer of a scan: its frames at one point of a motor dim, read as a scan of their own.

A z-scan (``uz x chi x mu``) holds a mosaicity scan at every height. Its layer
at one ``uz`` point has the grid ``chi x mu`` and ``uz`` as a fixed
positioner, so the layers of one scan stack like the scans of a z-stack
(:meth:`bexa.core.scan.Scan.layers`). A layer reads the frames through the
scan it comes from, and its cache records name the layer, so two layers of
one scan never share a cached result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bexa.core.status import ScanStatus
from bexa.core.structure import Structure
from bexa.io.base import BaseSource, Source

__all__ = ["LayerSource"]

# the labels of a scan that its layers keep: the scan number and the dataset name
LABELS = ("scan", "dataset")


class LayerSource(BaseSource):
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
        super().__init__(getattr(parent, "spec", None))
        whole = parent.structure()
        self._structure: Structure | None = whole.layer(dim, index)
        self.parent = parent
        self.dim = str(dim)
        self.index = int(index) % whole.motor_shape[whole.motor_dims.index(dim)]
        self.value = float(whole.coords[dim][self.index])
        for label in LABELS:  # a layer is labelled as its scan is
            if hasattr(parent, label):
                setattr(self, label, getattr(parent, label))

    def files(self) -> list[Path]:
        return self.parent.files()

    def cache_records(self) -> list[dict[str, Any]]:
        """The records of the scan, then the layer: the dim, the point and its value."""
        from bexa.core.provenance import file_records

        parent = self.parent
        records = (
            parent.cache_records()
            if hasattr(parent, "cache_records")
            else file_records(parent.files())
        )
        layer = {"path": "", "layer": self.dim, "index": self.index, "value": self.value}
        return [*records, layer]

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
            self._structure = self.parent.structure().layer(self.dim, self.index)
        return self._structure

    @property
    def status(self) -> ScanStatus:
        """The status of the scan the layer comes from."""
        return getattr(self.parent, "status", ScanStatus.COMPLETE)

    def aux_tables(self) -> dict[str, pd.DataFrame]:
        return self.parent.aux_tables() if hasattr(self.parent, "aux_tables") else {}

    def refresh(self) -> LayerSource:
        """Re-read the scan (a scan still being written) and the layer of it."""
        self.parent.refresh()
        self._structure = None
        return self

    def close(self) -> None:
        """Close the scan's files; they open again at the next read."""
        self.parent.close()
