"""The scans a picker lists and selects: dataset folders, filters, a selection across folders.

:class:`ScanListing` is :class:`bexa.notebook.picker.ScanPicker` without the
motors, the widgets and the state file: the dataset folders under a root, the
scans of the folders listed (several folders: one list with running ids), the
code-level filters and those of the bars, and the selection, which keeps the
scans chosen in any folder (:mod:`bexa.notebook.selection`).
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bexa._log import get_logger, progress
from bexa.core.dataset import Dataset
from bexa.core.scan_table import FIXED_COLUMNS, LEAD_COLUMNS
from bexa.core.status import ScanNotReadyError, ScanStatus, status_filter
from bexa.notebook.selection import OpenedDataset, Selection, folder_path

log = get_logger(__name__)

__all__ = ["STATUS_SHOWN", "ScanListing", "list_datasets"]

# the "show" filter of the bars: which statuses are listed
STATUS_SHOWN: dict[str, tuple[str, ...] | None] = {
    "any": None,
    "finished": ("complete", "aborted"),
    "complete": ("complete",),
}


def list_datasets(root: str | Path, depth: int = 2, format: str | None = None) -> list[Path]:
    """Dataset folders of numbered scans under ``root``, up to ``depth`` levels down.

    A folder is listed when a format spec whose folders hold numbered scans
    recognises it: a BLISS master beside ``scanNNNN`` folders, a SPEC session
    file (see :func:`bexa.io.survey.survey`). By default no HDF5 file is opened
    and the file-system part of the fingerprints decides; ``format="auto"``
    sniffs every folder fully, a spec name keeps the folders that spec
    recognises. Hidden folders, ``bexa_cache`` and ``__pycache__`` are skipped,
    a recognised folder is not entered, a dataset folder given as ``root`` is
    returned alone, and a missing root raises ``FileNotFoundError``.
    """
    from bexa.io.formats import load_spec
    from bexa.io.survey import survey

    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"no folder {root}")
    if format is None or format == "auto":
        entries = survey(root, depth=depth, quick=format is None)
    else:
        entries = survey(root, depth=depth, specs=[load_spec(format)])
    return [e.path for e in entries if e.kind == "folder" and e.opens_as_dataset]


def _common_root(folders: Sequence[Path]) -> Path:
    root = Path(os.path.commonpath([str(f.resolve()) for f in folders]))
    return root.parent if root in [f.resolve() for f in folders] else root


def _not_a_dataset(folders: Sequence[str | Path], error: Exception) -> str:
    """Why folders do not open as a dataset, naming the dataset folders under one if any."""
    for folder in map(Path, folders):
        if not folder.is_dir():
            return f"no folder {folder}"
        inside = [p for p in list_datasets(folder) if p != folder]
        if inside:
            names = ", ".join(p.relative_to(folder).as_posix() for p in inside[:5])
            more = ", ..." if len(inside) > 5 else ""
            return f"{folder} is not a dataset folder; the datasets under it: {names}{more}"
    return str(error)


def _line(row: Mapping[str, Any], lead: Sequence[str], columns: Sequence[str]) -> str:
    """A scan as one line: ``lead``, status, type, motors, shape, ``columns``, frames."""
    parts = [
        *lead,
        f"{row.get('status', 'complete')!s:<9}",
        f"{row['type']!s:<8}",
        f"{row['motors']!s:<20}",
        f"{row['shape']!s:<12}",
    ]
    for column in columns:
        value = row.get(column)
        if isinstance(value, (int, float, np.floating)) and pd.notna(value):
            parts.append(f"{value:.4g} keV" if column == "energy_keV" else f"{column} {value:.4g}")
    frames = row.get("frames")
    if frames is not None and pd.notna(frames):
        parts.append(f"{int(frames)} frames")
    return "  ".join(parts)


class ScanListing:
    """The scans of the dataset folders listed, the filters, and a selection across folders.

    The scans of the folders listed are shown (several folders: one list with
    running ids). The selection holds the scans chosen in any folder: listing
    another folder keeps them and choosing there adds to them. :attr:`ids` are
    the selected scans among those listed, :attr:`selection` every selected
    scan as ``(folder, scan number)``. The parameters are those of
    :class:`bexa.notebook.picker.ScanPicker` that decide what is listed;
    ``open_options`` go to :class:`bexa.core.dataset.Dataset`.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        folders: str | Path | Sequence[str | Path] | None = None,
        scans: Any = None,
        filters: Mapping[str, Any] | None = None,
        *,
        status: Any = "finished",
        depth: int = 2,
        **open_options: Any,
    ) -> None:
        if root is None and folders is None:
            raise ValueError("give root= (a RAW_DATA folder) or folders= (dataset folders)")
        given = [folders] if isinstance(folders, (str, Path)) else list(folders or [])
        self.root = Path(root) if root is not None else _common_root([Path(f) for f in given])
        self.depth = int(depth)
        self.datasets = (
            list_datasets(self.root, depth=self.depth)
            if root is not None
            else [Path(f) for f in given]
        )
        if not given:
            if not self.datasets:
                raise ValueError(f"no dataset folder under {self.root}")
            given = [self.datasets[0]]
            log.info("no folders given: opening %s", given[0])
        self.code_folders = [Path(f) for f in given]
        self.code_scans = scans
        self.code_filters = dict(filters or {})
        self._status = status
        self.selectable_statuses = status_filter(status)
        self.ui: dict[str, str] = {"type": "any", "motors": "", "status": "any"}
        self._listeners: list[Callable[[Any], None]] = []
        self._selection = Selection(open_options, self.selectable_statuses)
        self._listed: list[int] | None = None  # the candidates until the listing changes
        self.dataset: Dataset = None  # type: ignore[assignment]  (set by _open)
        self.folders: list[Path] = []
        self._key_of: dict[int, tuple[Path, int]] = {}
        self._open(self.code_folders)
        self._select_defaults()

    # ------------------------------------------------------------------ listing
    def _open(self, folders: Sequence[str | Path]) -> None:
        """List these folders; their dataset is opened once and kept for when they come back."""
        fresh = tuple(folder_path(f) for f in folders) not in self._selection.opened
        try:
            opened = self._selection.open(folders)
        except (OSError, ValueError) as exc:
            raise ValueError(_not_a_dataset(folders, exc)) from exc
        self._listing = opened
        self._show(opened, fresh)

    def _show(self, opened: OpenedDataset, fresh: bool = False) -> None:
        """List an opened dataset: open its scans (a progress bar when new), build the tables."""
        ds = self.dataset = opened.dataset
        self.folders = [folder_path(p) for p in ds.paths]
        self._key_of = {scan_id: key for key, scan_id in opened.ids.items()}
        bar = fresh and len(ds.scans) > 8
        for scan_id in progress(ds.scans, desc="opening scans", enabled=bar):
            with contextlib.suppress(ScanNotReadyError):
                ds.scan(scan_id)
        self.rows = ds.table()
        compact = ds.table(compact=True)
        self.varying_columns = [
            str(c) for c in compact.columns if c not in LEAD_COLUMNS and c not in FIXED_COLUMNS
        ]
        self.listed_by_code = self._listed_by_code()
        self._listed = None

    def _listed_by_code(self) -> list[int]:
        """The ids the code-level filters keep; all of them where the filters do not fit."""
        try:
            return self.dataset.select_numbers(None, status="any", **self.code_filters)
        except (KeyError, TypeError, ValueError) as exc:
            if self.folders == [folder_path(f) for f in self.code_folders]:
                raise
            log.warning("the filters do not fit %s (%s): all its scans listed", self.dataset, exc)
            return list(self.dataset.scans)

    def _select_defaults(self) -> None:
        """The code-level selection, in the code folders: ``scans`` through the filters, or all."""
        if self.code_scans is None:
            wanted = self.selectable
        else:
            wanted = self.dataset.select_numbers(  # status="any": the listing itself leaves
                self.code_scans,
                status="any",
                **self.code_filters,  # the unfinished scans out, quietly
            )
        allowed = set(self.selectable)
        self._selection.keys = {self._key_of[int(i)] for i in wanted if int(i) in allowed}

    def _changed(self) -> None:
        """Tell the observers (a subclass forgets what it derived from the selection first)."""
        for callback in self._listeners:
            callback(self)

    @property
    def candidates(self) -> list[int]:
        """The ids listed: the code-level filters, then the type, motors and status of the bars."""
        if self._listed is None:
            rows = self.rows.set_index("id")
            shown = STATUS_SHOWN[self.ui["status"]]
            kind, motors = self.ui["type"], self.ui["motors"].strip()
            keep = []
            for scan_id in self.listed_by_code:
                row = rows.loc[scan_id]
                if kind != "any" and str(row["type"]) != kind:
                    continue
                if motors and str(row["motors"]) != motors:
                    continue
                if shown is not None and str(row["status"]) not in shown:
                    continue
                keep.append(int(scan_id))
            self._listed = keep
        return list(self._listed)

    @property
    def selectable(self) -> list[int]:
        """The listed ids whose status allows them to be selected (finished scans by default)."""
        allowed = self.selectable_statuses
        if allowed is None:
            return self.candidates
        status = self.rows.set_index("id")["status"]
        return [i for i in self.candidates if ScanStatus(str(status.loc[i])) in allowed]

    def set_filters(self, **ui: str) -> None:
        """Narrow the listing: ``type="fscan3d"``, ``motors="chi x mu"``, ``status="finished"``.

        The selection stays as it is: a selected scan a filter hides is still selected.
        """
        unknown = sorted(set(ui) - set(self.ui))
        if unknown:
            raise ValueError(f"unknown filters {unknown}; the bars filter by {sorted(self.ui)}")
        if "status" in ui and ui["status"] not in STATUS_SHOWN:
            raise ValueError(f"status must be one of {sorted(STATUS_SHOWN)}")
        self.ui.update({k: str(v) for k, v in ui.items()})
        self._listed = None
        self._changed()

    def set_folders(self, folders: str | Path | Sequence[str | Path]) -> None:
        """List other dataset folder(s); the scans selected so far stay selected."""
        given = [folders] if isinstance(folders, (str, Path)) else list(folders)
        self._open(given)
        self._changed()

    def refresh(self) -> None:
        """Look at the folders again (scans finished or started since); the selection stays."""
        self._selection.refresh()
        self._show(self._listing)
        self._changed()

    def reset(self) -> None:
        """Back to the code defaults: the folders, the selection and the filters.

        The scans selected in other folders are dropped.
        """
        self.ui = {"type": "any", "motors": "", "status": "any"}
        self._open(self.code_folders)
        self._select_defaults()
        self._changed()

    def rows_of(self, ids: Iterable[int]) -> pd.DataFrame:
        """The full table rows of these listed ids, in this order."""
        return self.rows.set_index("id").loc[list(ids)].reset_index()

    # ---------------------------------------------------------------- selection
    @property
    def ids(self) -> list[int]:
        """The selected scans among those listed, by id (:attr:`selection` has them all)."""
        picked = self._selection.keys
        return [i for i in self.candidates if self._key_of[i] in picked]

    @property
    def selection(self) -> list[tuple[Path, int]]:
        """Every selected scan as ``(dataset folder, scan number)``, folder by folder."""
        return self._selection.ordered()

    def selected_in(self, folder: str | Path) -> list[int]:
        """The scan numbers selected in one dataset folder."""
        wanted = folder_path(folder)
        return [number for f, number in self.selection if f == wanted]

    def select(self, ids: Iterable[int]) -> None:
        """Select these scans among those listed, and none of the other listed ones.

        The scans selected in other folders, or hidden by a filter, stay
        selected (:meth:`clear` empties the whole selection). An id that is not
        listed or cannot be selected (a scan still running) is reported and
        left out.
        """
        wanted = [int(i) for i in ids]
        allowed = set(self.selectable)
        dropped = [i for i in wanted if i not in allowed]
        if dropped:
            log.warning("scans %s are not listed or not selectable: left out", dropped)
        listed = {self._key_of[i] for i in self.candidates}
        chosen = {self._key_of[i] for i in wanted if i in allowed}
        self._selection.keys = (self._selection.keys - listed) | chosen
        self._changed()

    def select_all(self) -> None:
        """Select every listed scan that can be selected (the other folders' scans stay)."""
        self.select(self.selectable)

    def remove(self, scans: Iterable[Any]) -> None:
        """Unselect scans: ids of the listed scans, or ``(folder, scan number)`` pairs."""
        keys = set()
        for item in scans:
            if isinstance(item, (tuple, list)):
                keys.add((folder_path(item[0]), int(item[1])))
            elif int(item) in self._key_of:
                keys.add(self._key_of[int(item)])
            else:
                raise ValueError(f"no scan {item} listed; give a listed id or (folder, number)")
        self._selection.keys = self._selection.keys - keys
        self._changed()

    def clear(self) -> None:
        """Select nothing, in every folder."""
        self._selection.keys = set()
        self._changed()

    def observe(self, callback: Callable[[Any], None]) -> None:
        """Call ``callback(listing)`` after every change of the selection or the filters."""
        self._listeners.append(callback)

    # -------------------------------------------------------------------- output
    @property
    def table(self) -> pd.DataFrame:
        """One row per selected scan, in selection order, its folder in ``dataset``.

        The positioners that are the same for every selected scan are left out,
        as in a compact :meth:`bexa.core.dataset.Dataset.table`.
        """
        return self._selection.table(self.root)

    def status_text(self) -> str:
        """``"14 of 20 scans selected"``; with scans selected elsewhere too, the folder listed
        and the whole selection: ``"2 of 6 scans selected in annealed; 10 in all, from 2
        folders"``."""
        text = f"{len(self.ids)} of {len(self.candidates)} scans selected"
        total = len(self._selection.keys)
        if total > len(self.ids):
            listed = " + ".join(folder.name for folder in self.folders)
            folders = len({folder for folder, _ in self._selection.keys})
            text += f" in {listed}; {total} in all, from {folders} folder"
            text += "s" if folders > 1 else ""
        return text

    def selection_text(self) -> str:
        """``"8 scans in 2 folders"``, or ``"nothing selected"``."""
        keys = self._selection.keys
        if not keys:
            return "nothing selected"
        folders = len({folder for folder, _ in keys})
        scans = f"{len(keys)} scan{'s' if len(keys) > 1 else ''}"
        return f"{scans} in {folders} folder{'s' if folders > 1 else ''}"

    def summary_line(self, row: Mapping[str, Any]) -> str:
        """One line of the list view: id, dataset, status, type, motors, shape, positioners."""
        lead = [f"{int(row['id']):>3}"]
        if "dataset" in row:
            lead.append(f"{row['dataset']!s:<12}")
        return _line(row, lead, self.varying_columns)

    def selection_lines(self) -> list[str]:
        """One line per selected scan (the selection panel): folder, scan number, then the rest."""
        table = self.table
        if table.empty:
            return []
        width = max(len(str(name)) for name in table["dataset"])
        columns = [str(c) for c in table.columns if c not in (*LEAD_COLUMNS, *FIXED_COLUMNS)]
        return [
            _line(row, [f"{row['dataset']!s:<{width}}", f"{int(row['scan']):>4}"], columns)
            for _, row in table.iterrows()
        ]

    def close(self) -> None:
        """Close the files of every dataset opened."""
        self._selection.close()
