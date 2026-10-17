"""Scans chosen in several dataset folders: the selection of the scan picker.

A scan is named by its dataset folder and number, ``(Path, int)``, so the
selection survives listing other folders. Each set of folders listed is
opened once as a :class:`bexa.core.dataset.Dataset` and kept, so listing it
again is instant, and a selected scan is opened through whichever of these
datasets holds it (a folder seen only in a saved state is opened on demand).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from bexa._log import get_logger
from bexa.core import scan_table
from bexa.core.dataset import Dataset
from bexa.core.scan import Scan
from bexa.core.status import ScanStatus

log = get_logger(__name__)

__all__ = ["OpenedDataset", "Selection", "folder_path"]


def folder_path(path: str | Path) -> Path:
    """A folder as the selection names it: absolute, links not followed (a mapped drive stays)."""
    return Path(os.path.abspath(path))


@dataclass
class OpenedDataset:
    """A dataset the picker opened, and the id of each of its scans by ``(folder, number)``."""

    dataset: Dataset
    ids: dict[tuple[Path, int], int] = field(default_factory=dict)

    def index(self) -> None:
        """Map every scan to its id (again after a refresh: new scans move the running ids)."""
        ids = {}
        for scan_id in self.dataset.scans:
            folder, number = self.dataset.origin(scan_id)
            ids[(folder_path(folder), int(number))] = int(scan_id)
        self.ids = ids


class Selection:
    """The scans chosen in any number of dataset folders, each as ``(folder, scan number)``.

    Parameters
    ----------
    open_options
        The keywords of :class:`bexa.core.dataset.Dataset` for every folder
        opened (``cache``, ``cache_reductions``, ...).
    statuses
        The statuses a scan must have to be selected, as
        :func:`bexa.core.status.status_filter` gives them; ``None`` allows any.
    """

    def __init__(
        self, open_options: dict[str, Any], statuses: frozenset[ScanStatus] | None
    ) -> None:
        self.open_options = dict(open_options)
        self.statuses = statuses
        self.opened: dict[tuple[Path, ...], OpenedDataset] = {}
        self.keys: set[tuple[Path, int]] = set()
        self.order: list[Path] = []  # the folders in the order they were first opened
        self._rows: dict[tuple[Path, int], dict[str, Any]] = {}

    def open(self, folders: Sequence[str | Path]) -> OpenedDataset:
        """The dataset of these folders (several: one list with running ids), opened once."""
        wanted = tuple(folder_path(f) for f in folders)
        opened = self.opened.get(wanted)
        if opened is None:
            opened = OpenedDataset(Dataset(list(wanted), **self.open_options))
            opened.index()
            self.opened[wanted] = opened
        for folder in opened.dataset.paths:
            if folder_path(folder) not in self.order:
                self.order.append(folder_path(folder))
        return opened

    def ordered(self) -> list[tuple[Path, int]]:
        """The selected scans folder by folder, in the order the folders were opened."""
        rank = {folder: k for k, folder in enumerate(self.order)}
        return sorted(self.keys, key=lambda key: (rank.get(key[0], len(rank)), str(key[0]), key[1]))

    def locate(self, key: tuple[Path, int]) -> tuple[Dataset, int]:
        """The opened dataset holding a scan and its id there; raises ``KeyError`` if none does.

        A folder not opened yet (named in a saved state) is opened alone first.
        """
        for opened in self.opened.values():
            if key in opened.ids:
                return opened.dataset, opened.ids[key]
        folder, number = key
        if (folder,) not in self.opened and folder.is_dir():
            opened = self.open([folder])
            if key in opened.ids:
                return opened.dataset, opened.ids[key]
        raise KeyError(f"no scan {number} in {folder}")

    def scan(self, key: tuple[Path, int]) -> Scan:
        """One selected scan, opened as bexa reads it by default."""
        dataset, scan_id = self.locate(key)
        return dataset.scan(scan_id)

    def allowed(self, key: tuple[Path, int]) -> bool:
        """Whether a scan can be selected: it is on disk, it opens and its status allows it."""
        try:
            status = self.scan(key).status
        except (KeyError, OSError, ValueError):  # gone, unreadable, or nothing to read yet
            return False
        return self.statuses is None or status in self.statuses

    def row(self, key: tuple[Path, int]) -> dict[str, Any]:
        """The table row of a scan, as :meth:`bexa.core.dataset.Dataset.table` builds it."""
        row = self._rows.get(key)
        if row is None:
            dataset, scan_id = self.locate(key)
            row = scan_table.scan_row(dataset.scan(scan_id), scan_id, key[1])
            self._rows[key] = row
        return row

    def label(self, folder: Path, root: Path | None = None) -> str:
        """A folder's name in the selection; its path under ``root`` when two folders share it."""
        if sum(f.name == folder.name for f in self.order) < 2:
            return folder.name
        if root is not None and folder.is_relative_to(folder_path(root)):
            return folder.relative_to(folder_path(root)).as_posix()
        return str(folder)

    def table(self, root: Path | None = None) -> pd.DataFrame:
        """One row per selected scan, in :meth:`ordered` order, its folder in ``dataset``.

        The positioners that are the same for every selected scan are left out,
        as in a compact :meth:`bexa.core.dataset.Dataset.table`; the index is
        the position of the scan in the selection.
        """
        keys = self.ordered()
        rows = []
        for key in keys:
            row = {"dataset": self.label(key[0], root), **self.row(key)}
            del row["id"]  # an id of one listing; the folder and the number name the scan
            rows.append(row)
        table = scan_table.arrange(rows)
        if table.empty:
            return table
        spec = self.scan(keys[0]).spec
        return table.drop(columns=scan_table.compact_columns(table, spec))

    def refresh(self) -> None:
        """Look at every opened folder again; selected scans gone or no longer allowed leave."""
        for opened in self.opened.values():
            opened.dataset.refresh()
            opened.index()
        self._rows.clear()
        gone = sorted(key for key in self.keys if not self.allowed(key))
        if gone:
            log.warning("%d selected scans are gone or cannot be selected any more", len(gone))
        self.keys -= set(gone)

    def close(self) -> None:
        """Close the files of every opened dataset."""
        for opened in self.opened.values():
            opened.dataset.close()
