"""Data objects: what a workspace holds and every operation makes.

An object is a scan read from files (kind ``scan``: never changed), several
scans handled together (``scanset``), or a result in memory: a ``stack`` of
detector frames over motor dims, an ``image`` (the two pixel dims), a ``map``
(two dims that are not pixels, such as a mesh scan's ROI sum), a ``curve``, an
N-dimensional ``volume``, an ``rgb`` picture, a ``table`` or a ``text`` report.
Operations never change their inputs: each makes new objects, which remember
their parents and the commands that made them.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from bexa.core.scan import Scan
from bexa.core.structure import BINNED_PIXEL_DIMS, PIXEL_DIMS

__all__ = [
    "ARRAY_KINDS",
    "KINDS",
    "DataObject",
    "ScanSet",
    "has_pixel_dims",
    "kind_of",
    "scan_label",
]

KINDS = ("scan", "scanset", "stack", "image", "map", "curve", "volume", "rgb", "table", "text")
ARRAY_KINDS = ("stack", "image", "map", "curve", "volume", "rgb")
SCAN_KINDS = ("scan", "scanset")
PIXEL_PAIRS = (PIXEL_DIMS, BINNED_PIXEL_DIMS)


def has_pixel_dims(dims: Sequence[str]) -> bool:
    """True when the last two dims are detector pixels (``y, x`` or the binned ``yb, xb``)."""
    return len(dims) >= 2 and tuple(dims[-2:]) in PIXEL_PAIRS


def kind_of(data: Any) -> str:
    """The kind of object ``data`` makes (see the module docstring)."""
    if isinstance(data, Scan):
        return "scan"
    if isinstance(data, ScanSet):
        return "scanset"
    if isinstance(data, pd.DataFrame):
        return "table"
    if isinstance(data, str):
        return "text"
    dims = tuple(getattr(data, "dims", ())) or tuple(f"dim_{i}" for i in range(np.ndim(data)))
    if dims and dims[-1] == "rgb":
        return "rgb"
    if len(dims) <= 1:
        return "curve"
    if has_pixel_dims(dims):
        return "image" if len(dims) == 2 else "stack"
    return "map" if len(dims) == 2 else "volume"


def scan_label(scan: Scan) -> str:
    """A short name for a scan: its dataset and number (``sample_1 #7``)."""
    number = getattr(scan.source, "scan", None)
    if isinstance(number, (int, np.integer)):
        return f"{scan.name} #{int(number)}"
    return str(scan.name)


class ScanSet:
    """Several scans handled as one object: the multi-scan input of the workspace.

    The scans keep their own grids; operations that take one scan run on each
    of them, and their results stack along the positioner that differs between
    the scans (``varying``) or along the scan order. :meth:`series` joins scans
    that share a grid into one scan with a new leading dim.
    """

    def __init__(self, scans: Sequence[Scan], labels: Sequence[str] | None = None) -> None:
        self.scans = list(scans)
        if not self.scans:
            raise ValueError("a scan set needs at least one scan")
        self.labels = list(labels) if labels is not None else [scan_label(s) for s in self.scans]
        if len(self.labels) != len(self.scans):
            raise ValueError(f"{len(self.labels)} labels for {len(self.scans)} scans")

    def __len__(self) -> int:
        return len(self.scans)

    def __iter__(self) -> Iterator[Scan]:
        return iter(self.scans)

    def __getitem__(self, index: int) -> Scan:
        return self.scans[index]

    def __repr__(self) -> str:
        more = ", ..." if len(self) > 4 else ""
        return f"ScanSet({len(self)} scans: {', '.join(self.labels[:4])}{more})"

    @property
    def recipes(self) -> list[dict[str, Any] | None]:
        """How to open every scan again (``bexa.open(**recipe)``)."""
        return [s.recipe for s in self.scans]

    @property
    def n_frames(self) -> int:
        return int(sum(s.n_frames for s in self.scans))

    def table(self) -> pd.DataFrame:
        """One row per scan: label, type, motors, grid, frames, energy and positioners."""
        from bexa.core.scan_table import scan_row

        rows = []
        for i, (scan, label) in enumerate(zip(self.scans, self.labels, strict=True)):
            number = getattr(scan.source, "scan", i)
            row = scan_row(scan, i, int(number) if isinstance(number, (int, np.integer)) else i)
            rows.append({"label": label, **row})
        return pd.DataFrame(rows)

    def varying(self, ignore: Sequence[str] | None = None) -> dict[str, np.ndarray]:
        """The positioners (and the energy) that differ between the scans."""
        if len(self) < 2:
            return {}
        from bexa.core.dataset import varying

        return varying(self.scans, ignore=ignore)

    def axis(self) -> tuple[str, np.ndarray]:
        """The dim results of the scans stack on: the leading varying positioner, else ``scan``."""
        found = self.varying()
        for name, values in found.items():
            values = np.asarray(values, dtype=float)
            if values.size == len(self) and np.unique(values).size == len(self):
                return name, values
        return "scan", np.arange(len(self), dtype=float)

    def series(self, dim: str | None = None) -> Scan:
        """The scans joined as one scan with a new leading dim (they must share one grid)."""
        from bexa.core.scan import combine_sources

        source = combine_sources([s.source for s in self.scans], list(range(len(self))), dim)
        first = self.scans[0]
        return Scan(
            source,
            first.spec,
            geometry=first.geometry,
            crystal=first.crystal,
            name=first.name,
            cache=first.cache,
            profile=first.profile,
            cache_reductions=first.cache_reductions,
        )

    def subset(self, indices: Sequence[int]) -> ScanSet:
        """The scans at ``indices``, in that order."""
        return ScanSet([self.scans[i] for i in indices], [self.labels[i] for i in indices])


@dataclass(eq=False)
class DataObject:
    """One object of a workspace.

    Attributes
    ----------
    name
        Unique within its workspace.
    data
        A :class:`bexa.Scan`, a :class:`ScanSet`, an ``xarray.DataArray``, a
        ``pandas.DataFrame`` or text.
    kind
        One of :data:`KINDS`; guessed from ``data`` when left empty.
    sigma
        Standard uncertainties of an array object, same shape (or ``None``).
    meta
        What the object is: energy, units, source file and scan, ROI, motor
        positions, the quantity it holds. Operations pass it on.
    history
        The commands that made the object, oldest first (its parents' too).
    parents
        The names of the objects it was made from.
    raw
        True for scans read from files: they are inputs only.
    """

    name: str
    data: Any
    kind: str = ""
    sigma: xr.DataArray | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)
    parents: list[str] = field(default_factory=list)
    notes: str = ""
    raw: bool = False
    created: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if isinstance(self.data, np.ndarray):
            self.data = xr.DataArray(self.data, dims=_default_dims(self.data.ndim))
        if isinstance(self.sigma, np.ndarray):
            self.sigma = xr.DataArray(self.sigma, dims=self.data.dims, coords=self.data.coords)
        self.kind = self.kind or kind_of(self.data)
        if self.kind not in KINDS:
            raise ValueError(f"unknown kind {self.kind!r}; choose from {KINDS}")
        if self.kind in SCAN_KINDS:
            self.raw = True

    # ------------------------------------------------------------------ what it is
    @property
    def is_array(self) -> bool:
        return self.kind in ARRAY_KINDS

    @property
    def shape(self) -> tuple[int, ...]:
        if self.kind == "scanset":
            return (len(self.data),)
        if self.kind == "text":
            return ()
        return tuple(int(n) for n in self.data.shape)

    @property
    def dims(self) -> tuple[str, ...]:
        if self.kind in ("scan", *ARRAY_KINDS):
            return tuple(str(d) for d in self.data.dims)
        return ()

    @property
    def array(self) -> xr.DataArray:
        """The data of an array object (an error for scans, tables and text)."""
        if not self.is_array:
            raise TypeError(f"{self.name} is a {self.kind}, not an array")
        return self.data

    def coord(self, dim: str) -> np.ndarray:
        """The coordinate values along ``dim`` (indices when it has none)."""
        if self.kind == "scan":
            if dim in self.data.coords:
                return np.asarray(self.data.coords[dim])
        elif dim in getattr(self.data, "coords", {}):
            return np.asarray(self.data.coords[dim].values)
        return np.arange(dict(zip(self.dims, self.shape, strict=True))[dim])

    def describe(self) -> str:
        """One line: kind, dims and sizes, dtype (``image y[512] x x[1028] float32``)."""
        if self.kind == "scan":
            s = self.data.structure
            grid = " x ".join(f"{d}[{n}]" for d, n in zip(s.dims, s.shape, strict=True))
            several = len(self.data.detectors) > 1  # the detector read, when there is a choice
            return f"scan {s.scan_type} {grid}" + (f", {self.data.detector}" if several else "")
        if self.kind == "scanset":
            return f"scan set of {len(self.data)} scans"
        if self.kind == "table":
            return f"table {self.data.shape[0]} rows x {self.data.shape[1]} columns"
        if self.kind == "text":
            first = self.data.strip().splitlines()[0] if self.data.strip() else ""
            return f"text: {first[:60]}"
        grid = " x ".join(f"{d}[{n}]" for d, n in zip(self.dims, self.shape, strict=True))
        uncertain = " +/-" if self.sigma is not None else ""
        return f"{self.kind} {grid} {self.data.dtype}{uncertain}"

    def label(self) -> str:
        """The name and :meth:`describe`, for lists."""
        return f"{self.name}  ({self.describe()})"

    def info(self) -> str:
        """Everything known about the object: description, metadata, parents, history, notes."""
        lines = [self.label()]
        if self.raw:
            lines.append("raw data: read from files, never changed by an operation")
        if self.kind == "scan":
            lines.append(self.data.info())
        elif self.kind == "scanset":
            lines.extend(f"  {label}" for label in self.data.labels)
        for key, value in self.meta.items():
            text = str(value)
            lines.append(f"{key}: {text[:200]}")
        if self.parents:
            lines.append("made from: " + ", ".join(self.parents))
        if self.history:
            lines.append("history:")
            lines.extend(f"  {step}" for step in self.history)
        if self.notes:
            lines.append("notes: " + self.notes)
        return "\n".join(lines)

    def copy(self, name: str | None = None) -> DataObject:
        """A copy with its own arrays (scans are shared: they are read-only)."""
        data = self.data
        if self.is_array or self.kind == "table":
            data = data.copy(deep=True)
        return DataObject(
            name=name or self.name,
            data=data,
            kind=self.kind,
            sigma=None if self.sigma is None else self.sigma.copy(deep=True),
            meta=copy.deepcopy(self.meta),
            history=list(self.history),
            parents=list(self.parents),
            notes=self.notes,
            raw=self.raw,
        )

    def __repr__(self) -> str:
        return f"DataObject({self.label()})"

    def _repr_html_(self) -> str:
        import html

        return f"<pre>{html.escape(self.info())}</pre>"


def _default_dims(ndim: int) -> tuple[str, ...]:
    """Dims for a plain array: the last two are pixels, the others ``dim_0``, ``dim_1``..."""
    if ndim == 1:
        return ("index",)
    if ndim == 2:
        return PIXEL_DIMS
    return tuple(f"dim_{i}" for i in range(ndim - 2)) + PIXEL_DIMS
