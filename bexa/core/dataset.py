"""A dataset folder as one object: the table of its scans, selection, and stacking.

One BLISS dataset holds many numbered scans: alignment scans, a mosaicity scan
per sample height, an energy series. :class:`Dataset` reads the metadata of
every scan once, keeps it as a table, and opens scans by number or by what
they contain. :func:`stack` reduces several scans with the same accumulators
and concatenates the results on a new dim, a positioner such as ``samz``,
which is how a z-stack becomes ``(samz, y, x)`` maps.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from bexa._log import get_logger
from bexa.core.provenance import build_attrs
from bexa.core.scan import Scan, list_scans
from bexa.core.scan import open as open_scan

log = get_logger(__name__)

__all__ = ["Dataset", "open_dataset", "stack"]

FIXED_COLUMNS = ("scan", "type", "title", "motors", "shape", "ranges", "frames", "missing")


def _scan_numbers(selection: Any, available: Sequence[int]) -> list[int]:
    """Resolve an int, an inclusive ``(start, end)`` tuple or a list against the scans on disk."""
    if selection is None:
        return list(available)
    if isinstance(selection, (int, np.integer)):
        wanted = [int(selection)]
    elif isinstance(selection, tuple) and len(selection) == 2:
        wanted = [n for n in available if int(selection[0]) <= n <= int(selection[1])]
    else:
        wanted = [int(n) for n in selection]
    missing = [n for n in wanted if n not in available]
    if missing:
        raise ValueError(f"scans {missing} are not in the dataset; available: {list(available)}")
    return wanted


class Dataset:
    """The scans of one dataset folder (this is not an ``xarray.Dataset``).

    Parameters
    ----------
    path
        The dataset folder, its master file or one of its ``scanNNNN`` folders.
    profile, sample, dataset
        Instead of a path: a beamtime profile with a sample name (or a dataset
        folder name), as for :func:`bexa.open`.
    detector, format, cache
        Passed to :func:`bexa.open` for every scan.

    Examples
    --------
    >>> ds = bexa.open_dataset(folder)
    >>> ds.table()                                   # one row per scan
    >>> ds[7]                                        # one Scan
    >>> ds.select(scans=[1, 5, 7])                   # chosen by number
    >>> ds.select(type="fscan2d", samz=(0, 0.1))     # chosen by content
    >>> ds.series(dim="energy", scans=(20, 40))      # several scans as one Scan
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        profile: Any = None,
        sample: str | None = None,
        dataset: str | None = None,
        format: str | None = None,
        detector: str | None = None,
        cache: Any = True,
        **open_kwargs: Any,
    ) -> None:
        self._open_kwargs: dict[str, Any] = dict(open_kwargs)
        if detector is not None:
            self._open_kwargs["detector"] = detector
        if profile is not None or path is None:
            from bexa.config import BeamtimeProfile, load_profile

            prof = profile if isinstance(profile, BeamtimeProfile) else load_profile(profile)
            if sample is not None:
                dataset = prof.sample(sample).dataset
            if dataset is None:
                raise ValueError("give sample=<name from the profile> or dataset=<folder name>")
            self.path = prof.data_root() / dataset
            self.name = f"{prof.name}/{dataset}"
            self._open_kwargs.update({"profile": prof, "sample": sample, "dataset": dataset})
            self._list_kwargs = {"format": prof.format, "detector": detector or prof.detector}
        else:
            folder = Path(str(path).split("::")[0])
            if folder.is_file() or (folder.name.startswith("scan") and folder.name[4:].isdigit()):
                folder = folder.parent
            self.path = folder
            self.name = folder.name
            self._open_kwargs.update({"format": format, "cache": cache})
            self._list_kwargs = {"format": format, "detector": detector}
        self._numbers: list[int] | None = None
        self._table: pd.DataFrame | None = None
        self._scans: dict[int, Scan] = {}

    # ------------------------------------------------------------- basics
    @property
    def scans(self) -> list[int]:
        """Every scan number that has detector files, in order."""
        if self._numbers is None:
            self._numbers = list_scans(self.path, **self._list_kwargs)
        return list(self._numbers)

    def refresh(self) -> Dataset:
        """Forget the cached scan list, table and handles (for a dataset still being written)."""
        for scan in self._scans.values():
            scan.close()
        self._scans.clear()
        self._numbers = None
        self._table = None
        return self

    def __len__(self) -> int:
        return len(self.scans)

    def __iter__(self) -> Iterable[int]:
        return iter(self.scans)

    def __contains__(self, number: object) -> bool:
        return number in self.scans

    def __repr__(self) -> str:
        return f"Dataset({self.name!r}, {len(self)} scans, {self.path})"

    def _open(self, scan: Any, **kwargs: Any) -> Scan:
        options = {**self._open_kwargs, **kwargs}
        if "profile" in options:
            return open_scan(scan=scan, **options)
        return open_scan(self.path, scan=scan, **options)

    def scan(self, number: int) -> Scan:
        """Open one scan (handles are kept until :meth:`refresh` or :meth:`close`)."""
        number = int(number)
        if number not in self._scans:
            if number not in self.scans:
                raise ValueError(f"scan {number} is not in the dataset; available: {self.scans}")
            self._scans[number] = self._open(number)
        return self._scans[number]

    def __getitem__(self, key: Any) -> Scan | list[Scan]:
        if isinstance(key, (int, np.integer)):
            return self.scan(int(key))
        return [self.scan(n) for n in _scan_numbers(key, self.scans)]

    def close(self) -> None:
        for scan in self._scans.values():
            scan.close()
        self._scans.clear()

    def __enter__(self) -> Dataset:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -------------------------------------------------------------- table
    def table(self, refresh: bool = False) -> pd.DataFrame:
        """One row per scan: type, title, motors, shape, ranges, frames, energy and positioners.

        Every scalar positioner of the master file becomes a column, so the
        table can be filtered by ``samz``, ``ccmth`` or any other fixed motor.
        """
        if self._table is not None and not refresh:
            return self._table.copy()
        rows = []
        for number in self.scans:
            s = self.scan(number).structure
            row: dict[str, Any] = {
                "scan": number,
                "type": s.scan_type,
                "title": s.title,
                "motors": " x ".join(s.motor_dims),
                "shape": " x ".join(str(n) for n in s.motor_shape),
                "ranges": "; ".join(
                    f"{d} {s.coords[d].min():.5g}..{s.coords[d].max():.5g}" for d in s.motor_dims
                ),
                "frames": s.n_frames,
                "missing": s.n_missing,
                "energy_keV": s.energy_keV,
            }
            row.update({k: v for k, v in s.scalars.items() if k not in row})
            rows.append(row)
        table = pd.DataFrame(rows)
        if not table.empty:
            positioners = sorted(
                c for c in table.columns if c not in (*FIXED_COLUMNS, "energy_keV")
            )
            table = table[[*FIXED_COLUMNS, "energy_keV", *positioners]]
        self._table = table
        return table.copy()

    def info(self) -> str:
        """The table as text, with the folder on top."""
        table = self.table()
        head = f"{self.name}: {len(self)} scans in {self.path}"
        return head if table.empty else f"{head}\n{table.to_string(index=False)}"

    def _repr_html_(self) -> str:
        table = self.table()
        caption = f"<b>{self.name}</b>: {len(self)} scans in <code>{self.path}</code>"
        return caption if table.empty else caption + table.to_html(index=False)

    # ---------------------------------------------------------- selection
    def select_numbers(
        self,
        scans: Any = None,
        *,
        type: str | None = None,
        motors: str | Sequence[str] | None = None,
        where: Callable[[pd.Series], bool] | None = None,
        **positioners: Any,
    ) -> list[int]:
        """Scan numbers chosen by hand and/or by content.

        Parameters
        ----------
        scans
            A number, an inclusive ``(start, end)`` tuple or a list of numbers;
            ``None`` starts from every scan. Numbers that are not on disk raise.
        type
            The scan type, for example ``"fscan2d"``.
        motors
            The scanned motors, exactly: ``"mu"`` or ``("chi", "mu")``.
        where
            A callable on a table row returning True to keep the scan.
        **positioners
            A table column with a value (``samz=0.5``, matched within 1e-9) or
            an inclusive ``(low, high)`` range; ``energy_keV`` works too.
        """
        numbers = _scan_numbers(scans, self.scans)
        if type is None and motors is None and where is None and not positioners:
            return numbers
        table = self.table()
        table = table[table["scan"].isin(numbers)]
        if type is not None:
            table = table[table["type"] == type]
        if motors is not None:
            wanted = motors if isinstance(motors, str) else " x ".join(motors)
            table = table[table["motors"] == wanted]
        for name, value in positioners.items():
            if name not in table.columns:
                raise ValueError(f"no column {name!r} in the table; columns: {list(table.columns)}")
            column = pd.to_numeric(table[name], errors="coerce")
            if isinstance(value, (tuple, list)) and len(value) == 2:
                keep = (column >= float(value[0])) & (column <= float(value[1]))
            else:
                keep = np.isclose(column, float(value), atol=1e-9, rtol=0.0)
            table = table[np.asarray(keep, dtype=bool)]
        if where is not None:
            table = table[[bool(where(row)) for _, row in table.iterrows()]]
        return [int(n) for n in table["scan"]]

    def select(self, scans: Any = None, **criteria: Any) -> list[Scan]:
        """The scans of :meth:`select_numbers`, opened."""
        return [self.scan(n) for n in self.select_numbers(scans, **criteria)]

    def series(self, scans: Any = None, *, dim: str | None = None, **criteria: Any) -> Scan:
        """Several scans opened as one :class:`Scan` with a new leading dim.

        ``dim`` is ``"energy"`` (the default when every scan has its own
        energy), ``"scan"`` (the scan numbers) or a positioner such as
        ``"samz"`` whose value is read from every scan. The scans must share
        one motor grid.
        """
        numbers = self.select_numbers(scans, **criteria)
        if not numbers:
            raise ValueError("no scan matches the selection")
        return self._open(numbers, stack_dim=dim)

    def groups(self, by: str, scans: Any = None, **criteria: Any) -> dict[float, list[int]]:
        """Scan numbers grouped by the value of a positioner column (``samz`` gives the layers)."""
        numbers = self.select_numbers(scans, **criteria)
        table = self.table()
        table = table[table["scan"].isin(numbers)]
        if by not in table.columns:
            raise ValueError(f"no column {by!r} in the table; columns: {list(table.columns)}")
        out: dict[float, list[int]] = {}
        for value, part in table.groupby(by, sort=True):
            out[float(value)] = [int(n) for n in part["scan"]]
        return out


def open_dataset(path: str | Path | None = None, **kwargs: Any) -> Dataset:
    """Open a dataset folder (or ``profile=``/``sample=``) as a :class:`Dataset`."""
    return Dataset(path, **kwargs)


def _coordinate(scan: Scan, dim: str) -> float:
    s = scan.structure
    if dim in s.scalars:
        return float(s.scalars[dim])
    if dim in s.per_frame:
        return float(np.mean(s.per_frame[dim]))
    available = sorted(s.scalars)
    raise ValueError(f"{scan.name} records no positioner {dim!r}; available: {available}")


def stack(
    scans: Iterable[Scan],
    accumulators: Iterable[Any],
    dim: str = "samz",
    *,
    coords: Sequence[float] | None = None,
    **reduce_kwargs: Any,
) -> xr.Dataset:
    """Reduce every scan with the same accumulators and stack the results on ``dim``.

    The coordinate of each scan is its positioner ``dim`` (``samz`` for a
    z-stack) unless ``coords`` gives the values. ``roi``, ``downsample``,
    ``device`` and the other keywords of :func:`bexa.reduce` apply to every
    scan, so the maps line up pixel by pixel. The result is sorted along ``dim``.
    """
    scans = list(scans)
    if not scans:
        raise ValueError("no scans to stack")
    accumulators = list(accumulators)
    values = (
        [float(v) for v in coords] if coords is not None else [_coordinate(s, dim) for s in scans]
    )
    if len(values) != len(scans):
        raise ValueError("one coordinate per scan is needed")
    parts = []
    for scan, value in zip(scans, values, strict=True):
        result = scan.reduce(accumulators, **reduce_kwargs)
        parts.append(result.expand_dims({dim: [value]}))
    out = xr.concat(parts, dim=dim, combine_attrs="drop_conflicts").sortby(dim)
    files = [str(f) for scan in scans for f in scan.files]
    names = [
        type(acc).__name__ if not isinstance(acc, type) else acc.__name__ for acc in accumulators
    ]
    out.attrs.update(
        build_attrs(
            files,
            {"dim": dim, "coords": values, "accumulators": names, "n_scans": len(scans)},
        )
    )
    out.attrs["stack_dim"] = dim
    out.attrs["scan_names"] = [scan.name for scan in scans]
    return out
