"""A dataset folder (or several) as one object: the table of its scans, selection, stacking.

One BLISS dataset holds many numbered scans: alignment scans, a mosaicity scan
per sample height, an energy series; a measurement sometimes continues in a
second folder. :class:`Dataset` reads the metadata of every scan once, keeps it
as a table, and opens scans by number or by what they contain. :func:`varying`
says which positioners (and whether the energy) differ between chosen scans,
and :func:`stack` reduces them with the same accumulators onto that grid, so a
z-stack becomes ``(samz, y, x)`` maps and a z-stack at several energies
``(samz, energy, y, x)`` maps, in memory or in one HDF5 file read on demand.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from concurrent.futures import as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from bexa._log import get_logger, interactive_session, progress
from bexa.core.backend import DEFAULT_MEMORY_FRACTION, resolve_device
from bexa.core.parallel import default_workers, process_pool
from bexa.core.provenance import build_attrs, fingerprint, to_json
from bexa.core.reductions import make_plan
from bexa.core.resources import usable_cpus
from bexa.core.scan import Scan, _resolve_cache, _resolve_spec, combine_sources, list_scans
from bexa.core.scan import open as open_scan
from bexa.core.structure import ENERGY_DIM, _group_values

log = get_logger(__name__)

__all__ = ["Dataset", "common_grid", "open_dataset", "regrid", "stack", "varying"]

LEAD_COLUMNS = ("id", "dataset")
FIXED_COLUMNS = ("scan", "type", "title", "motors", "shape", "ranges", "frames", "missing")
STATS_SUFFIXES = ("_block_total", "_block_p1", "_block_p99")
# stacks on disk are bexa's own files: lzf writes ten times faster than gzip and reads four
# times faster, which the browsers feel at every slider move
STORE_COMPRESSION = "lzf"


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


def _folder_of(path: str | Path) -> Path:
    """The dataset folder behind a folder, a master file or a ``scanNNNN`` folder."""
    folder = Path(str(path).split("::")[0])
    if folder.is_file() or (folder.name.startswith("scan") and folder.name[4:].isdigit()):
        folder = folder.parent
    return folder


@dataclass
class _Folder:
    """One dataset folder of a :class:`Dataset` and how its scans are opened."""

    path: Path
    name: str
    open_kwargs: dict[str, Any]
    list_kwargs: dict[str, Any]
    numbers: list[int] | None = field(default=None, repr=False)

    def scans(self) -> list[int]:
        if self.numbers is None:
            self.numbers = list_scans(self.path, **self.list_kwargs)
        return list(self.numbers)

    def open(self, scan: Any, **kwargs: Any) -> Scan:
        options = {**self.open_kwargs, **kwargs}
        if "profile" in options:
            return open_scan(scan=scan, **options)
        return open_scan(self.path, scan=scan, **options)


class Dataset:
    """The scans of one dataset folder, or of several (this is not an ``xarray.Dataset``).

    Every scan has an ``id``, the first column of :meth:`table`: for one folder
    the scan number, so ``ds[7]`` is scan 7; for several folders a running
    number over the combined table, next to a ``dataset`` column naming the
    folder. ``ds[id]``, :meth:`select`, :meth:`series` and :meth:`groups` all
    take ids. The scans share one result cache.

    Parameters
    ----------
    path
        The dataset folder, its master file or one of its ``scanNNNN`` folders;
        or a list of folders when a measurement continues in another dataset.
    profile, sample, dataset
        Instead of a path: a beamtime profile with a sample name (or a dataset
        folder name), as for :func:`bexa.open`.
    detector, format, cache, cache_reductions
        Passed to :func:`bexa.open` for every scan; ``cache`` may be a folder
        (one cache for the dataset there), ``True``, ``False`` or a
        :class:`bexa.core.cache.Cache`.

    Examples
    --------
    >>> ds = bexa.open_dataset(folder)
    >>> ds.table()                                   # one row per scan
    >>> ds[7]                                        # one Scan
    >>> ds.select(scans=[1, 5, 7])                   # chosen by number
    >>> ds.select(type="fscan2d", samz=(0, 0.1))     # chosen by content
    >>> ds.varying(type="fscan2d")                   # {"samz": [...], "energy": [...]}
    >>> ds.series(dim="energy", scans=(20, 40))      # several scans as one Scan
    """

    def __init__(
        self,
        path: str | Path | Sequence[str | Path] | None = None,
        *,
        profile: Any = None,
        sample: str | None = None,
        dataset: str | None = None,
        format: str | None = None,
        detector: str | None = None,
        cache: Any = True,
        cache_reductions: bool | None = None,
        **open_kwargs: Any,
    ) -> None:
        from bexa.config import cache_root

        self._open_kwargs: dict[str, Any] = dict(open_kwargs)
        if detector is not None:
            self._open_kwargs["detector"] = detector
        if cache_reductions is not None:
            self._open_kwargs["cache_reductions"] = cache_reductions
        folders: list[_Folder] = []
        if profile is not None or path is None:
            from bexa.config import BeamtimeProfile, load_profile

            prof = profile if isinstance(profile, BeamtimeProfile) else load_profile(profile)
            if sample is not None:
                dataset = prof.sample(sample).dataset
            if dataset is None:
                raise ValueError("give sample=<name from the profile> or dataset=<folder name>")
            self.cache = _resolve_cache(
                True if cache is None else cache, cache_root(prof.processed_root)
            )
            options = {
                **self._open_kwargs,
                "profile": prof,
                "sample": sample,
                "dataset": dataset,
                "cache": self.cache if self.cache is not None else False,
            }
            listing = {"format": prof.format, "detector": detector or prof.detector}
            folders.append(
                _Folder(prof.data_root() / dataset, f"{prof.name}/{dataset}", options, listing)
            )
        else:
            paths = [path] if isinstance(path, (str, Path)) else list(path)
            if not paths:
                raise ValueError("give at least one dataset folder")
            self.cache = _resolve_cache(cache, cache_root())
            for p in paths:
                folder = _folder_of(p)
                # the layout is sniffed once per folder, here: a folder that cannot be read (a
                # BLISS dataset copied without its master, say) fails where it is named, and
                # the scans are opened by the spec's name without sniffing again
                spec_name = _resolve_spec(folder, format, None).name
                options = {
                    **self._open_kwargs,
                    "format": spec_name,
                    "cache": self.cache if self.cache is not None else False,
                }
                listing = {"format": spec_name, "detector": detector}
                folders.append(_Folder(folder, folder.name, options, listing))
        self._folders = folders
        self.paths = [f.path for f in folders]
        self.path = self.paths[0]
        self.name = folders[0].name if len(folders) == 1 else " + ".join(f.name for f in folders)
        self._ids: list[int] | None = None
        self._where: dict[int, tuple[int, int]] = {}
        self._table: pd.DataFrame | None = None
        self._scans: dict[int, Scan] = {}

    # ------------------------------------------------------------- basics
    @property
    def scans(self) -> list[int]:
        """Every scan id (the scan number for one folder, a running number for several)."""
        if self._ids is None:
            where: dict[int, tuple[int, int]] = {}
            ids: list[int] = []
            for k, folder in enumerate(self._folders):
                for number in folder.scans():
                    scan_id = number if len(self._folders) == 1 else len(ids)
                    where[scan_id] = (k, number)
                    ids.append(scan_id)
            self._where, self._ids = where, ids
        return list(self._ids)

    @property
    def folders(self) -> list[Path]:
        """The dataset folders, in the order their scans are numbered."""
        return list(self.paths)

    def refresh(self) -> Dataset:
        """Forget the cached scan list, table and handles (for a dataset still being written)."""
        for scan in self._scans.values():
            scan.close()
        self._scans.clear()
        for folder in self._folders:
            folder.numbers = None
        self._ids = None
        self._table = None
        return self

    def clear_cache(self) -> int:
        """Delete every cached result of the dataset's cache; returns the number of files."""
        if self.cache is None:
            return 0
        return self.cache.clear()

    def __len__(self) -> int:
        return len(self.scans)

    def __iter__(self) -> Iterable[int]:
        return iter(self.scans)

    def __contains__(self, number: object) -> bool:
        return number in self.scans

    def __repr__(self) -> str:
        return f"Dataset({self.name!r}, {len(self)} scans, {self.path})"

    def scan(self, number: int) -> Scan:
        """Open one scan by id (handles are kept until :meth:`refresh` or :meth:`close`)."""
        number = int(number)
        if number not in self._scans:
            if number not in self.scans:
                raise ValueError(f"scan {number} is not in the dataset; available: {self.scans}")
            k, scan_number = self._where[number]
            self._scans[number] = self._folders[k].open(scan_number)
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
    def table(self, refresh: bool = False, compact: bool = False) -> pd.DataFrame:
        """One row per scan: type, title, motors, shape, ranges, frames, energy and positioners.

        Every scalar positioner of the master file becomes a column, so the
        table can be filtered by ``samz``, ``ccmth`` or any other fixed motor.
        ``compact=True`` drops the positioner columns that never change and the
        ``missing`` column when no scan has missing frames.
        """
        if self._table is None or refresh:
            rows = []
            for scan_id in self.scans:
                k, number = self._where[scan_id]
                s = self.scan(scan_id).structure
                row: dict[str, Any] = {"id": scan_id}
                if len(self._folders) > 1:
                    row["dataset"] = self._folders[k].name
                row.update(
                    {
                        "scan": number,
                        "type": s.scan_type,
                        "title": s.title,
                        "motors": " x ".join(s.motor_dims),
                        "shape": " x ".join(str(n) for n in s.motor_shape),
                        "ranges": "; ".join(
                            f"{d} {s.coords[d].min():.5g}..{s.coords[d].max():.5g}"
                            for d in s.motor_dims
                        ),
                        "frames": s.n_frames,
                        "missing": s.n_missing,
                        "energy_keV": s.energy_keV,
                    }
                )
                row.update({k: v for k, v in s.scalars.items() if k not in row})
                rows.append(row)
            table = pd.DataFrame(rows)
            if not table.empty:
                lead = [c for c in LEAD_COLUMNS if c in table.columns]
                fixed = [*lead, *FIXED_COLUMNS, "energy_keV"]
                positioners = sorted(c for c in table.columns if c not in fixed)
                table = table[[*fixed, *positioners]]
            self._table = table
        table = self._table.copy()
        if compact and not table.empty:
            positioners = [
                c for c in table.columns if c not in (*LEAD_COLUMNS, *FIXED_COLUMNS, "energy_keV")
            ]
            dropped = [c for c in positioners if table[c].nunique(dropna=True) <= 1]
            spec = self.scan(self.scans[0]).spec  # a motor kept under a legacy name too: once
            aliases = (spec.motors.get("aliases", {}) or {}) if spec is not None else {}
            for logical, physical in aliases.items():
                names = [physical] if isinstance(physical, str) else list(physical)
                if logical in positioners and any(
                    p in positioners and table[logical].equals(table[p]) for p in names
                ):
                    dropped.append(logical)
            for leader, followers in _followers(spec).items():  # samz follows uz: once too
                if leader in positioners and leader not in dropped:
                    dropped.extend(f for f in followers if f in positioners)
            if not table["missing"].any():
                dropped.append("missing")
            table = table.drop(columns=sorted(set(dropped)))
        return table

    def info(self) -> str:
        """The table as text, with the folder on top."""
        table = self.table()
        head = f"{self.name}: {len(self)} scans in {', '.join(str(p) for p in self.paths)}"
        return head if table.empty else f"{head}\n{table.to_string(index=False)}"

    def _repr_html_(self) -> str:
        table = self.table()
        where = ", ".join(f"<code>{p}</code>" for p in self.paths)
        caption = f"<b>{self.name}</b>: {len(self)} scans in {where}"
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
        """Scan ids chosen by hand and/or by content.

        Parameters
        ----------
        scans
            An id, an inclusive ``(start, end)`` tuple or a list of ids;
            ``None`` starts from every scan. Ids that are not on disk raise.
        type
            The scan type, for example ``"fscan2d"``.
        motors
            The scanned motors, exactly: ``"mu"`` or ``("chi", "mu")``.
        where
            A callable on a table row returning True to keep the scan.
        **positioners
            A table column with a value (``samz=0.5``, matched within 1e-9) or
            an inclusive ``(low, high)`` range; ``energy_keV`` and ``dataset``
            work too.
        """
        numbers = _scan_numbers(scans, self.scans)
        if type is None and motors is None and where is None and not positioners:
            return numbers
        table = self.table()
        table = table[table["id"].isin(numbers)]
        if type is not None:
            table = table[table["type"] == type]
        if motors is not None:
            wanted = motors if isinstance(motors, str) else " x ".join(motors)
            table = table[table["motors"] == wanted]
        for name, value in positioners.items():
            if name not in table.columns:
                raise ValueError(f"no column {name!r} in the table; columns: {list(table.columns)}")
            if table[name].dtype == object:  # a text column such as dataset
                table = table[table[name] == value]
                continue
            column = pd.to_numeric(table[name], errors="coerce")
            if isinstance(value, (tuple, list)) and len(value) == 2:
                keep = (column >= float(value[0])) & (column <= float(value[1]))
            else:
                keep = np.isclose(column, float(value), atol=1e-9, rtol=0.0)
            table = table[np.asarray(keep, dtype=bool)]
        if where is not None:
            table = table[[bool(where(row)) for _, row in table.iterrows()]]
        return [int(n) for n in table["id"]]

    def select(self, scans: Any = None, **criteria: Any) -> list[Scan]:
        """The scans of :meth:`select_numbers`, opened."""
        return [self.scan(n) for n in self.select_numbers(scans, **criteria)]

    def series(self, scans: Any = None, *, dim: str | None = None, **criteria: Any) -> Scan:
        """Several scans opened as one :class:`Scan` with a new leading dim.

        ``dim`` is ``"energy"`` (the default when every scan has its own
        energy), ``"scan"`` (the ids) or a positioner such as ``"samz"`` whose
        value is read from every scan. The scans must share one motor grid.
        """
        numbers = self.select_numbers(scans, **criteria)
        if not numbers:
            raise ValueError("no scan matches the selection")
        if len(self._folders) == 1:
            return self._folders[0].open([self._where[n][1] for n in numbers], stack_dim=dim)
        first = self.scan(numbers[0])
        source = combine_sources([self.scan(n).source for n in numbers], numbers, dim)
        return Scan(
            source,
            first.spec,
            geometry=first.geometry,
            crystal=first.crystal,
            name=self.name,
            cache=self.cache,
            profile=first.profile,
            cache_reductions=first.cache_reductions,
        )

    def groups(self, by: str, scans: Any = None, **criteria: Any) -> dict[float, list[int]]:
        """Scan ids grouped by the value of a positioner column (``samz`` gives the layers)."""
        numbers = self.select_numbers(scans, **criteria)
        table = self.table()
        table = table[table["id"].isin(numbers)]
        if by not in table.columns:
            raise ValueError(f"no column {by!r} in the table; columns: {list(table.columns)}")
        out: dict[float, list[int]] = {}
        for value, part in table.groupby(by, sort=True):
            out[float(value)] = [int(n) for n in part["id"]]
        return out

    def varying(
        self,
        scans: Any = None,
        axes: str | Sequence[str] | None = None,
        ignore: str | Sequence[str] | None = None,
        **criteria: Any,
    ) -> dict[str, np.ndarray]:
        """What differs between the chosen scans, or the ``axes`` named: see :func:`varying`."""
        return varying(self.select(scans, **criteria), axes=axes, ignore=ignore)


def open_dataset(path: str | Path | Sequence[str | Path] | None = None, **kwargs: Any) -> Dataset:
    """Open a dataset folder, several folders, or ``profile=``/``sample=`` as a :class:`Dataset`."""
    return Dataset(path, **kwargs)


# ------------------------------------------------------------ the outer grid
def _coordinate(scan: Scan, dim: str, index: int) -> float:
    """The value of ``dim`` for one scan: its energy, its number, or a positioner."""
    s = scan.structure
    if dim == ENERGY_DIM:
        if s.energy_keV is None:
            raise ValueError(f"{scan.name} has no energy to stack on")
        return float(s.energy_keV)
    if dim == "scan":
        return float(getattr(scan.source, "scan", index))
    if dim in s.scalars:
        return float(s.scalars[dim])
    if dim in s.per_frame:
        return float(np.mean(s.per_frame[dim]))
    available = sorted(s.scalars)
    raise ValueError(f"{scan.name} records no positioner {dim!r}; available: {available}")


def _followers(spec: Any) -> dict[str, list[str]]:
    """``{leader: [followers]}`` from the spec's ``motors.coupled`` (``uz`` drives ``samz``)."""
    coupled = (spec.motors.get("coupled", {}) or {}) if spec is not None else {}
    return {
        str(leader): [names] if isinstance(names, str) else [str(n) for n in names]
        for leader, names in coupled.items()
    }


def _same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    """True when two label arrays split the scans into the same groups (lockstep motors)."""
    pairs = set(zip(a.tolist(), b.tolist(), strict=True))
    return len(pairs) == len(set(a.tolist())) == len(set(b.tolist()))


def varying(
    scans: Iterable[Scan],
    tolerance: float | None = None,
    axes: str | Sequence[str] | None = None,
    ignore: str | Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """The positioners, and the energy, that differ between scans: ``{dim: sorted values}``.

    Scanned motors are left out, and so is the monochromator motor the format
    spec derives the energy from (``ccmth``) when the energy itself varies, so
    the same thing is not counted twice; the energy is reported as ``"energy"``
    in keV. A motor that another one drives (the spec's ``motors.coupled``:
    ``uz`` moves the stage readbacks ``samx``, ``samy``, ``samz`` at ID03) is
    not an axis of its own when its leader steps, and neither is any positioner
    that steps in lockstep with another one (the leader, then the first in the
    spec's ``motors.known`` order, is kept). ``ignore`` names positioners that
    are never axes, whatever they do (a readback that drifts between scans).
    Values closer than ``tolerance`` (default 1e-3 of their spread) count as
    one. The dims come slowest first: the one that changes least often from
    scan to scan is the outer one, as ``uz`` is for a z-stack repeated at every
    energy.

    ``axes`` overrides all of this: these names, in this order, are the dims
    (positioners, ``"energy"`` or ``"scan"``), for when other positioners drift
    or follow and the automatic choice is not the one measured.
    """
    scans = list(scans)
    if not scans:
        return {}
    if axes is not None:
        chosen = [axes] if isinstance(axes, str) else [str(a) for a in axes]
        return {
            d: _group_values(
                np.array([_coordinate(s, d, i) for i, s in enumerate(scans)], dtype=float),
                tolerance,
            )[0]
            for d in chosen
        }
    rows: list[dict[str, float]] = []
    scanned: set[str] = set()
    for scan in scans:
        s = scan.structure
        scanned |= set(s.motor_dims)
        row = {k: float(v) for k, v in s.scalars.items()}
        if s.energy_keV is not None:
            row[ENERGY_DIM] = float(s.energy_keV)
        rows.append(row)
    columns = [c for c in dict.fromkeys(k for r in rows for k in r) if c not in scanned]
    found: dict[str, tuple[np.ndarray, int, np.ndarray]] = {}
    for column in columns:
        if not all(column in r for r in rows):
            continue
        values = np.array([r[column] for r in rows], dtype=float)
        if not np.all(np.isfinite(values)):
            continue
        centers, labels = _group_values(values, tolerance)
        if len(centers) > 1:
            found[column] = (centers, int(np.count_nonzero(np.diff(labels))), labels)
    spec = scans[0].spec
    energy_from = spec.energy.get("from") if spec is not None else None
    if ENERGY_DIM in found and energy_from in found:
        del found[energy_from]
    # a motor the spec keeps under a legacy name too (samz and z1) counts once
    aliases = (spec.motors.get("aliases", {}) or {}) if spec is not None else {}
    for logical, physical in aliases.items():
        names = [physical] if isinstance(physical, str) else list(physical)
        twins = [p for p in names if p in found and logical in found]
        if twins and np.array_equal(found[logical][0], found[twins[0]][0]):
            del found[logical]
    # the motors a stepping leader drives follow it: not axes of their own
    coupled = _followers(spec)
    for leader, followers in coupled.items():
        if leader in found:
            for name in followers:
                found.pop(name, None)
    for name in [ignore] if isinstance(ignore, str) else list(ignore or []):
        found.pop(name, None)  # never an axis, by request
    # nor is anything stepping in lockstep with another positioner: the leader, the energy or
    # the first in the spec's known order is the axis
    known = list(spec.known_motors()) if spec is not None else []

    def rank(column: str) -> tuple[int, int, str]:
        first = column in coupled or column == ENERGY_DIM
        return (0 if first else 1, known.index(column) if column in known else len(known), column)

    kept: dict[str, tuple[np.ndarray, int, np.ndarray]] = {}
    for column in sorted(found, key=rank):
        if not any(_same_partition(found[column][2], other[2]) for other in kept.values()):
            kept[column] = found[column]
    order = sorted(kept, key=lambda c: kept[c][1])  # fewest changes first: the outer dims
    return {c: kept[c][0] for c in order}


STATS_SAMPLE = 1 << 18  # pixels per frame that the percentiles look at (every k-th pixel)


def _block_stats(array: np.ndarray) -> dict[str, np.ndarray]:
    """Per-frame total and 1st/99th percentiles over the last two axes (NaN frames stay NaN).

    The total is exact; the percentiles, which only set colour limits, come
    from a strided sample of large frames so that a 2048 x 2048 map costs a
    few ms rather than a sort of four million values.
    """
    lead = tuple(array.shape[:-2])
    flat = np.asarray(array).reshape((*lead, -1))
    step = max(1, flat.shape[-1] // STATS_SAMPLE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN frames
        total = np.where(
            np.all(np.isnan(flat), axis=-1), np.nan, np.nansum(flat, axis=-1, dtype=np.float64)
        )
        p1, p99 = np.nanpercentile(flat[..., ::step], [1, 99], axis=-1)
    return {"total": np.asarray(total), "p1": np.asarray(p1), "p99": np.asarray(p99)}


def _attach_stats(array: xr.DataArray, stats: dict[str, Any], dims: Sequence[str]) -> None:
    """Keep the per-frame statistics on the array, for the browsers and :func:`brightest`."""
    array.attrs["block_stats_dims"] = [str(d) for d in dims]
    for key, value in stats.items():
        array.attrs[f"block_{key}"] = np.asarray(value)


def stack(
    scans: Iterable[Scan],
    accumulators: Iterable[Any],
    dim: str | Sequence[str] = "auto",
    *,
    coords: Sequence[float] | None = None,
    store: str | Path | bool | None = None,
    dtype: Any = None,
    tolerance: float | None = None,
    workers: int | str | None = None,
    grid: str | Mapping[str, Any] | bool | None = "auto",
    ignore: str | Sequence[str] | None = None,
    **reduce_kwargs: Any,
) -> xr.Dataset:
    """Reduce every scan with the same accumulators and stack the results on a grid of scans.

    ``dim`` names the new dims: ``"auto"`` takes the positioners (and the
    energy) that differ between the scans, from :func:`varying`; one name
    (``"samz"``) or several (``["uz", "energy"]``) name them by hand, each read
    from every scan (``"energy"`` is the energy in keV, ``"scan"`` the scan
    numbers). The results form a dense grid, NaN where no scan sits at a
    combination; two scans at the same position are an error. ``roi``,
    ``downsample``, ``device`` and the other keywords of :func:`bexa.reduce`
    apply to every scan, so the maps line up pixel by pixel; the inner dims
    must have the same shape in every scan and take the first scan's
    coordinates. Each output variable is allocated once and filled scan by
    scan: peak memory is the result plus one scan, and the result is checked
    against the memory budget first.

    ``store`` writes the grid scan by scan into one HDF5 file instead and
    returns it opened with :func:`bexa.io.cube.open_lazy`, so only the blocks
    that are indexed are ever read: the way to browse previews of a dataset
    that does not fit in memory. ``True`` puts the file in the scans' cache
    (``stacks/<fingerprint>.h5``), a path names it; a file built from the same
    scans with the same request is reused. Every result also carries per-frame
    statistics (``<name>_block_total``, ``_block_p1``, ``_block_p99``) for
    :func:`bexa.viz.interactive.brightest` and the browsers' colour limits.

    ``workers`` reduces several scans at once, each in its own process (reopened
    from ``Scan.recipe``), which is how the reading and decompression of the
    frames run in parallel: h5py lets one thread into HDF5 at a time, so
    threads cannot. An int sets the number of processes; ``"auto"`` uses the
    cores of the job but one on the CPU, and ``GPU_WORKERS`` (3) when the device
    resolves to the GPU, which the workers then share (two or three keep it fed
    while the others read). Each worker gets ``BEXA_MEMORY_FRACTION`` divided by
    the number of workers and its share of the cores as ``BEXA_THREADS``.
    Results arrive as they finish and are written at once, so memory stays at
    a few scans' results.

    Parameters
    ----------
    coords
        With one dim, the coordinate of each scan given by hand.
    dtype
        Store the results in this dtype (``np.float32`` halves the maps).
    tolerance
        Values of a dim closer than this are the same grid point (default
        1e-3 of the spread).
    workers
        Scans reduced at the same time, in processes (default: one, in this
        process).
    grid
        What to do when the scans' motor grids differ (another range or step at
        some heights): ``"auto"`` interpolates every result that has motor
        dims (previews, per-frame statistics, curves) onto the grid of
        :func:`common_grid`, the full range in the finest step; a
        ``{dim: values}`` mapping names that grid; ``False`` requires one grid
        and raises otherwise. Maps without motor dims are never touched.
    ignore
        Positioners ``dim="auto"`` must not take as axes however they vary
        (see :func:`varying`).
    """
    scans = list(scans)
    if not scans:
        raise ValueError("no scans to stack")
    accumulators = list(accumulators)
    dims, centers, positions = _grid(scans, dim, coords, tolerance, ignore)
    grid_coords: dict[str, np.ndarray] | None = None
    if isinstance(grid, Mapping):
        grid_coords = {str(d): np.asarray(v, dtype=float) for d, v in grid.items()}
    elif grid == "auto" and len(scans) > 1:
        planned = _planned_grids(scans, reduce_kwargs)
        if all(g.keys() == planned[0].keys() for g in planned):
            common, differs = _merge_grids(planned)
            if differs:
                grid_coords = common
                log.info(
                    "the motor grids of the scans differ; results with motor dims are "
                    "interpolated onto the common grid %s",
                    {
                        d: f"{v.min():.5g}..{v.max():.5g} ({len(v)} points)"
                        for d, v in common.items()
                    },
                )
    names = [
        type(acc).__name__ if not isinstance(acc, type) else acc.__name__ for acc in accumulators
    ]
    files = [str(f) for scan in scans for f in scan.files]
    records = [r for scan in scans for r in _records(scan)]
    request = {
        "dims": dims,
        "centers": {d: c.tolist() for d, c in zip(dims, centers, strict=True)},
        "accumulators": [_describe(acc) for acc in accumulators],
        "reduce": {
            k: v.to_dict() if hasattr(v, "to_dict") else v
            for k, v in sorted(reduce_kwargs.items())
            if k in ("roi", "downsample", "method", "dtype")
        },
        "dtype": str(np.dtype(dtype)) if dtype is not None else None,
        "grid": None
        if grid_coords is None
        else {d: [float(v[0]), float(v[-1]), len(v)] for d, v in grid_coords.items()},
    }
    key = fingerprint(records, request)
    attrs = build_attrs(files, {**request, "n_scans": len(scans)}, stack_dims=dims, stack_key=key)
    attrs["stack_dims"] = list(dims)
    attrs["stack_dim"] = dims[0] if len(dims) == 1 else to_json(dims)
    attrs["scan_names"] = [scan.name for scan in scans]
    attrs["scan_ids"] = [int(getattr(scan.source, "scan", i)) for i, scan in enumerate(scans)]
    attrs["datasets"] = ", ".join(dict.fromkeys(scan.name for scan in scans))  # where it came from
    attrs["accumulators"] = names
    if not dims:  # one scan, nothing to stack on
        result = scans[0].reduce(accumulators, **reduce_kwargs)
        for name in list(result.data_vars):
            if result[name].ndim >= 2:
                _attach_stats(
                    result[name], _block_stats(result[name].values), result[name].dims[:-2]
                )
        return result

    grid_shape = tuple(len(c) for c in centers)
    outer_coords = {d: c for d, c in zip(dims, centers, strict=True)}
    missing = int(np.prod(grid_shape)) - len(scans)
    if missing:
        log.warning(
            "%d of %d grid positions on %s have no scan; their results are NaN",
            missing,
            int(np.prod(grid_shape)),
            dims,
        )
    if store is not None and store is not False:
        path = _store_path(store, scans[0], key)
        if path.exists() and _store_key(path) == key:
            log.info("reusing the stacked results in %s", path)
            return _open_store(path)
        return _build_store(
            path,
            scans,
            accumulators,
            positions,
            grid_shape,
            outer_coords,
            attrs,
            dtype,
            reduce_kwargs,
            workers,
            grid_coords,
        )
    return _build_in_memory(
        scans,
        accumulators,
        positions,
        grid_shape,
        outer_coords,
        attrs,
        dtype,
        reduce_kwargs,
        workers,
        grid_coords,
    )


def _planned_grids(
    scans: list[Scan], reduce_kwargs: Mapping[str, Any]
) -> list[dict[str, np.ndarray]]:
    """The motor coordinates each scan's pass will have, after the ROI and the downsampling."""
    roi = reduce_kwargs.get("roi")
    downsample = reduce_kwargs.get("downsample")
    return [
        {
            str(d): np.asarray(c, dtype=float)
            for d, c in make_plan(
                s.structure, roi if roi is not None else s.roi, downsample
            ).coords.items()
        }
        for s in scans
    ]


def _merge_grids(grids: list[dict[str, np.ndarray]]) -> tuple[dict[str, np.ndarray], bool]:
    """One grid spanning every grid, in the finest step, and whether any grid differed."""
    common: dict[str, np.ndarray] = {}
    differs = False
    for d in grids[0]:
        arrays = [g[d] for g in grids]
        first = arrays[0]
        spread = max((float(np.ptp(a)) for a in arrays if a.size), default=0.0)
        same = all(
            a.shape == first.shape and np.allclose(a, first, rtol=0, atol=1e-3 * spread + 1e-12)
            for a in arrays[1:]
        )
        if same:
            common[d] = first
            continue
        differs = True
        lo = min(float(a.min()) for a in arrays)
        hi = max(float(a.max()) for a in arrays)
        steps = [float(np.median(np.abs(np.diff(a)))) for a in arrays if a.size > 1]
        step = min((s for s in steps if s > 0), default=0.0)
        if step <= 0 or hi <= lo:
            common[d] = np.unique(np.concatenate(arrays))
        else:
            # the full range in a step no coarser than the finest; it ends on the highest value
            common[d] = np.linspace(lo, hi, int(np.ceil((hi - lo) / step - 1e-9)) + 1)
    return common, differs


def common_grid(
    scans: Iterable[Scan],
    dims: str | Sequence[str] | None = None,
    roi: Any = None,
    downsample: Any = None,
) -> dict[str, np.ndarray]:
    """One coordinate array per scanned motor spanning every scan, in the finest step used.

    For scans whose grids differ (another range or step at some heights, two
    datasets measured differently): from the lowest to the highest value of
    each motor over the scans, with the smallest median step any scan used;
    a motor whose grid is the same in every scan keeps it. ``dims`` limits the
    answer to some motors (default: every scanned motor, which every scan must
    have); ``roi`` and ``downsample``, as in :func:`stack`, restrict the ranges
    the way the pass does. :func:`regrid` puts results onto this grid, and
    :func:`stack` does so by itself (``grid="auto"``).
    """
    scans = list(scans)
    if not scans:
        return {}
    grids = _planned_grids(scans, {"roi": roi, "downsample": downsample})
    wanted = [dims] if isinstance(dims, str) else list(grids[0]) if dims is None else list(dims)
    missing = [d for d in wanted if any(d not in g for g in grids)]
    if missing:
        raise ValueError(f"{missing} are not scanned by every scan: {[tuple(g) for g in grids]}")
    common, _ = _merge_grids([{d: g[d] for d in wanted} for g in grids])
    return common


def regrid(
    result: xr.Dataset | xr.DataArray, coords: Mapping[str, Any], method: str = "linear"
) -> Any:
    """``result`` with every variable on a dim of ``coords`` interpolated onto those values.

    The result of a pass (or any xarray object): variables with one of the
    named dims are interpolated along them (``method`` as in ``DataArray.interp``,
    ``"nearest"`` for a dim with a single point), NaN outside their own range;
    variables without those dims, the maps, are returned as they are.
    """
    targets = {str(d): np.asarray(v, dtype=float) for d, v in coords.items()}

    def one(var: xr.DataArray) -> xr.DataArray:
        wanted = {d: v for d, v in targets.items() if d in var.dims}
        if not wanted or all(
            d in var.coords and np.array_equal(var.coords[d].values, v) for d, v in wanted.items()
        ):
            return var
        how: Any = method if all(var.sizes[d] > 1 for d in wanted) else "nearest"
        out = var.interp(wanted, method=how, kwargs={"bounds_error": False, "fill_value": np.nan})
        return out.assign_attrs(var.attrs)

    if isinstance(result, xr.DataArray):
        return one(result)
    return result.map(one, keep_attrs=True)


GPU_WORKERS = 3  # "auto" workers sharing one GPU: enough to keep it fed while others read


def _resolve_workers(workers: int | str | None, n_scans: int, device: str) -> int:
    if workers is None or workers is False:
        return 1
    if workers == "auto":
        count = GPU_WORKERS if device == "cuda" else default_workers("cpu")
    else:
        count = int(workers)
    return max(1, min(count, n_scans))


def _reduce_recipe(
    recipe: dict[str, Any],
    accumulators: list[Any],
    reduce_kwargs: dict[str, Any],
    cache: str | None,
    cache_reductions: bool,
) -> xr.Dataset:
    """One scan reduced in a worker process: reopened from its recipe, closed after."""
    scan = open_scan(**recipe, cache=cache or False, cache_reductions=cache_reductions)
    try:
        return scan.reduce(accumulators, **reduce_kwargs)
    finally:
        scan.close()


def _reduced(
    scans: list[Scan],
    accumulators: list[Any],
    reduce_kwargs: dict[str, Any],
    workers: Any,
    grid: dict[str, np.ndarray] | None = None,
) -> Generator[tuple[int, xr.Dataset], None, None]:
    """``(index, result)`` per scan: in this process in order, or from processes as they finish.

    With ``grid`` every result is first put onto that motor grid (:func:`regrid`).
    """
    device = resolve_device(reduce_kwargs.get("device", "auto"))  # what this process would use
    n_workers = _resolve_workers(workers, len(scans), device)
    if n_workers <= 1:
        for i, scan in enumerate(scans):
            result = scan.reduce(accumulators, **reduce_kwargs)
            yield i, regrid(result, grid) if grid else result
        return
    jobs = []
    for scan in scans:
        if scan.recipe is None:
            raise ValueError(
                f"{_label(scan)} cannot be reopened in a worker: scans opened with bexa.open or "
                "a Dataset carry the recipe that workers need"
            )
        root = scan.cache.root if scan.cache is not None else None
        jobs.append((scan.recipe, None if root is None else str(root), scan.cache_reductions))
    # one progress bar over the scans, here; the workers share the GPU when there is one
    kwargs = {**reduce_kwargs, "show_progress": False, "device": device}
    fraction = float(os.environ.get("BEXA_MEMORY_FRACTION", DEFAULT_MEMORY_FRACTION))
    cpus, _ = usable_cpus()
    env = {  # the workers share the job's memory and cores
        "BEXA_MEMORY_FRACTION": f"{fraction / n_workers:.6g}",
        "BEXA_THREADS": str(max(1, min(8, cpus // n_workers))),
    }
    log.info("reducing %d scans in %d processes", len(scans), n_workers)
    pool = process_pool(n_workers, env=env)
    try:
        futures = {
            pool.submit(_reduce_recipe, recipe, accumulators, kwargs, root, cached): i
            for i, (recipe, root, cached) in enumerate(jobs)
        }
        show = reduce_kwargs.get("show_progress") is not False and interactive_session()
        for future in progress(as_completed(futures), len(futures), desc="scans", enabled=show):
            result = future.result()
            yield futures[future], regrid(result, grid) if grid else result
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)  # scans still running finish on their own
        raise
    pool.shutdown(wait=True)


def _records(scan: Scan) -> list[dict[str, Any]]:
    source = scan.source
    if hasattr(source, "cache_records"):
        return list(source.cache_records())
    from bexa.core.provenance import file_records

    return file_records(scan.files)


def _describe(acc: Any) -> dict[str, Any]:
    if isinstance(acc, type):
        acc = acc()
    return dict(acc.describe())


def _grid(
    scans: list[Scan],
    dim: str | Sequence[str],
    coords: Sequence[float] | None,
    tolerance: float | None,
    ignore: str | Sequence[str] | None = None,
) -> tuple[list[str], list[np.ndarray], list[tuple[int, ...]]]:
    """The dims of the grid, their sorted centres, and the position of every scan on it."""
    if isinstance(dim, str) and dim == "auto":
        dims = list(varying(scans, tolerance, ignore=ignore))
        if not dims and len(scans) > 1:
            dims = ["scan"]
    else:
        dims = [dim] if isinstance(dim, str) else [str(d) for d in dim]
    if coords is not None:
        if len(dims) != 1:
            raise ValueError("coords can only be given for a single dim")
        values = [np.asarray([float(v) for v in coords], dtype=float)]
        if len(values[0]) != len(scans):
            raise ValueError("one coordinate per scan is needed")
    else:
        values = [
            np.array([_coordinate(scan, d, i) for i, scan in enumerate(scans)], dtype=float)
            for d in dims
        ]
    centers: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for d, v in zip(dims, values, strict=True):
        if d == "scan":
            centers.append(np.asarray(v))  # the given order, as a series on "scan" keeps it
            labels.append(np.arange(len(v)))
            continue
        c, lab = _group_values(v, tolerance)
        centers.append(c)
        labels.append(lab)
    positions = [tuple(int(lab[i]) for lab in labels) for i in range(len(scans))]
    seen: dict[tuple[int, ...], int] = {}
    for i, pos in enumerate(positions):
        if pos in seen:
            where = {d: float(c[p]) for d, c, p in zip(dims, centers, pos, strict=True)}
            raise ValueError(
                f"{_label(scans[seen[pos]])} and {_label(scans[i])} both sit at {where}; "
                "select one of them, or add the dim that tells them apart"
            )
        seen[pos] = i
    return dims, centers, positions


def _label(scan: Scan) -> str:
    number = getattr(scan.source, "scan", None)
    return f"{scan.name} scan {number}" if number is not None else scan.name


def _plan_variables(
    first: xr.Dataset, dtype: Any
) -> dict[str, tuple[tuple[str, ...], tuple[int, ...], np.dtype]]:
    return {
        str(name): (tuple(str(d) for d in var.dims), tuple(var.shape), np.dtype(dtype or var.dtype))
        for name, var in first.data_vars.items()
    }


def _build_in_memory(
    scans: list[Scan],
    accumulators: list[Any],
    positions: list[tuple[int, ...]],
    grid_shape: tuple[int, ...],
    outer_coords: dict[str, np.ndarray],
    attrs: dict[str, Any],
    dtype: Any,
    reduce_kwargs: dict[str, Any],
    workers: Any = None,
    grid: dict[str, np.ndarray] | None = None,
) -> xr.Dataset:
    from bexa.core.backend import check_fits

    dims = list(outer_coords)
    arrays: dict[str, np.ndarray] = {}
    stats: dict[str, dict[str, np.ndarray]] = {}
    layout: dict[str, tuple[tuple[str, ...], tuple[int, ...], np.dtype]] = {}
    inner_coords: dict[str, Any] = {}
    inner_attrs: dict[str, dict[str, Any]] = {}
    results = _reduced(scans, accumulators, reduce_kwargs, workers, grid)
    try:
        for i, result in results:
            scan, position = scans[i], positions[i]
            if not arrays:
                layout = _plan_variables(result, dtype)
                total = sum(
                    int(np.prod(grid_shape + shape)) * dt.itemsize
                    for _, shape, dt in layout.values()
                )
                check_fits(
                    total,
                    f"stacking {len(scans)} scans on {dims}",
                    reduce_kwargs.get("device", "cpu")
                    if reduce_kwargs.get("device") not in (None, "auto")
                    else "cpu",
                    hint="pass store=True to build the stack on disk, or add an ROI or a "
                    "downsample",
                )
                for name, (_, shape, dt) in layout.items():
                    arrays[name] = np.full(grid_shape + shape, np.nan, dtype=dt)
                    stats[name] = (
                        {
                            k: np.full(grid_shape + shape[:-2], np.nan)
                            for k in ("total", "p1", "p99")
                        }
                        if len(shape) >= 2
                        else {}
                    )
                    inner_coords.update(
                        {str(k): v for k, v in result[name].coords.items() if k not in inner_coords}
                    )
                    inner_attrs[name] = dict(result[name].attrs)
            for name, (_, shape, _) in layout.items():
                if name not in result or tuple(result[name].shape) != shape:
                    got = tuple(result[name].shape) if name in result else None
                    raise ValueError(
                        f"{scan.name}: {name} has shape {got}, the first scan {shape}; "
                        "stack scans with the same grid and the same ROI"
                    )
                block = result[name].values
                arrays[name][position] = block
                if stats[name]:
                    for k, v in _block_stats(block).items():
                        stats[name][k][position] = v
    finally:
        results.close()  # stops the worker pool when a result could not be placed
    data_vars = {}
    for name, (var_dims, _, _) in layout.items():
        full_dims = (*dims, *var_dims)
        coords = {
            **{d: outer_coords[d] for d in dims},
            **{d: inner_coords[d] for d in var_dims if d in inner_coords},
        }
        array = xr.DataArray(
            arrays[name], dims=full_dims, coords=coords, name=name, attrs=inner_attrs[name]
        )
        if stats[name]:
            _attach_stats(array, stats[name], full_dims[:-2])
        data_vars[name] = array
    out = xr.Dataset(data_vars, attrs=attrs)
    for d in dims:
        if d == ENERGY_DIM:
            out.coords[d].attrs["units"] = "keV"
    return out


def _store_path(store: str | Path | bool, scan: Scan, key: str) -> Path:
    if store is True:
        if scan.cache is None or scan.cache.root is None:
            raise ValueError(
                "store=True needs a disk cache: open the scans with cache=<folder> or cache=True, "
                "or give store=<file path>"
            )
        return scan.cache.root / "stacks" / f"{key}.h5"
    if isinstance(store, (str, Path)):
        return Path(store)
    raise ValueError(f"store must be True or a file path, not {store!r}")


def _store_key(path: Path) -> str | None:
    import h5py

    try:
        with h5py.File(path, "r") as f:
            key = f.attrs.get("stack_key")
    except OSError:
        return None
    return key.decode() if isinstance(key, bytes) else (str(key) if key is not None else None)


def _open_store(path: Path) -> xr.Dataset:
    """The stored stack opened lazily, its per-frame statistics moved from variables to attrs."""
    from bexa.io.cube import open_lazy

    out = open_lazy(path)
    for name in list(out.data_vars):
        if str(name).endswith(STATS_SUFFIXES) or f"{name}_block_total" not in out:
            continue
        stats = {k: out[f"{name}_block_{k}"].values for k in ("total", "p1", "p99")}
        _attach_stats(out[name], stats, out[name].dims[:-2])
    return out.drop_vars([n for n in out.data_vars if str(n).endswith(STATS_SUFFIXES)])


def _build_store(
    path: Path,
    scans: list[Scan],
    accumulators: list[Any],
    positions: list[tuple[int, ...]],
    grid_shape: tuple[int, ...],
    outer_coords: dict[str, np.ndarray],
    attrs: dict[str, Any],
    dtype: Any,
    reduce_kwargs: dict[str, Any],
    workers: Any = None,
    grid: dict[str, np.ndarray] | None = None,
) -> xr.Dataset:
    from bexa.io.cube import StoreWriter

    dims = list(outer_coords)
    writer: StoreWriter | None = None
    layout: dict[str, tuple[tuple[str, ...], tuple[int, ...], np.dtype]] = {}
    reduce_kwargs = {"cache": False, **reduce_kwargs}  # the store is the cache of these results
    results = _reduced(scans, accumulators, reduce_kwargs, workers, grid)
    try:
        for i, result in results:
            scan, position = scans[i], positions[i]
            if writer is None:
                layout = _plan_variables(result, dtype)
                writer = StoreWriter(path, attrs, compression=STORE_COMPRESSION)
                for d in dims:
                    writer.add_coord(
                        d, outer_coords[d], attrs={"units": "keV"} if d == ENERGY_DIM else None
                    )
                written: set[str] = set(dims)
                for name, (var_dims, shape, dt) in layout.items():
                    full_dims = (*dims, *var_dims)
                    writer.add_variable(
                        name, full_dims, grid_shape + shape, dt, attrs=dict(result[name].attrs)
                    )
                    if len(shape) >= 2:
                        for k in ("total", "p1", "p99"):
                            writer.add_variable(
                                f"{name}_block_{k}",
                                full_dims[:-2],
                                grid_shape + shape[:-2],
                                np.dtype(np.float64),
                            )
                    for cname, coord in result[name].coords.items():
                        if str(cname) not in written:
                            writer.add_coord(
                                str(cname), coord.values, dims=coord.dims, attrs=dict(coord.attrs)
                            )
                            written.add(str(cname))
            for name, (_, shape, _) in layout.items():
                if name not in result or tuple(result[name].shape) != shape:
                    got = tuple(result[name].shape) if name in result else None
                    raise ValueError(
                        f"{scan.name}: {name} has shape {got}, the first scan {shape}; "
                        "stack scans with the same grid and the same ROI"
                    )
                block = result[name].values
                writer.write(name, position, block)
                if len(shape) >= 2:
                    for k, v in _block_stats(block).items():
                        writer.write(f"{name}_block_{k}", position, v)
            del result
    except BaseException:
        if writer is not None:
            writer.abort()
        raise
    finally:
        results.close()  # stops the worker pool when a result could not be written
    assert writer is not None
    writer.close()
    return _open_store(path)
