"""Choosing scans in a notebook: the datasets under a root, the scans of one, the motors to read.

:class:`ScanPicker` holds what the notebook's data cell used to spell out by hand
(``FOLDERS``, ``SCANS``, ``FILTERS``, the stack axes): the dataset folders under a
``RAW_DATA`` root, the scans listed and selected, the motors to extract and
which of them is the stack axis. In a Jupyter kernel with ipywidgets it shows
as a widget (:mod:`bexa.notebook.picker_widgets`); anywhere else as a table,
and the same methods drive it from code.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bexa._log import get_logger, in_kernel, progress
from bexa.core.dataset import Dataset, varying
from bexa.core.scan import Scan
from bexa.core.scan_table import FIXED_COLUMNS, LEAD_COLUMNS, MotorCandidate
from bexa.core.status import ScanNotReadyError, ScanStatus, status_filter

log = get_logger(__name__)

__all__ = ["STATUS_SHOWN", "VIEWS", "ScanPicker", "list_datasets", "pick_scans"]

VIEWS = ("list", "table", "gallery")
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


def _stable(value: Any) -> Any:
    """A JSON-able stand-in for a code default; a lambda counts by its code, not its address."""
    if callable(value):
        code = getattr(value, "__code__", None)
        if code is not None:
            return ["callable", code.co_code.hex(), repr(code.co_consts), repr(code.co_names)]
        return ["callable", repr(value)]
    if isinstance(value, Mapping):
        return {str(k): _stable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stable(v) for v in value]
    return repr(value)


def _common_root(folders: Sequence[Path]) -> Path:
    root = Path(os.path.commonpath([str(f.resolve()) for f in folders]))
    return root.parent if root in [f.resolve() for f in folders] else root


class ScanPicker:
    """The scans of a beamtime to analyse: which folders, which scans, which motors.

    Parameters
    ----------
    root
        A ``RAW_DATA`` folder whose datasets are listed (:func:`list_datasets`);
        ``None`` lists the given ``folders`` only.
    folders
        The dataset folder(s) opened at first, as one :class:`bexa.core.dataset.Dataset`
        with running ids; ``None`` opens the first dataset under ``root``.
    scans, filters
        The scans selected at first, as :meth:`bexa.core.dataset.Dataset.select_numbers`
        takes them (``None``: every scan that passes ``filters``), and the
        code-level filters that decide what is listed at all (``type``,
        ``motors``, ``where``, positioner ranges).
    view
        ``"list"`` (one line per scan, the default), ``"table"`` (a checkbox per
        row, sortable columns) or ``"gallery"`` (thumbnails).
    cache, cache_reductions, **open_kwargs
        As for :func:`bexa.open_dataset`.
    interactive
        ``None``: widgets when inside a Jupyter kernel with ipywidgets; ``False``
        never; ``True`` builds them without a kernel (tests).
    state_file
        A JSON file that remembers the selection, the pressed motors, the stack
        axis and the filters across kernel restarts, as long as the code
        defaults above are unchanged (edited code wins).
    status
        Which scans may be selected: ``"finished"`` (complete and aborted, the
        default), ``"complete"`` or ``"any"``. Scans still running, empty or
        unreadable are listed greyed and never selected.
    page_size
        Rows per page of the table and gallery views.
    depth
        How many folder levels below ``root`` the datasets are looked for
        (:func:`list_datasets`): 2 is ``RAW_DATA``, sample, dataset.
    motors, stack_axis
        The motors to extract and the stack axis chosen in code instead of with
        the buttons (``None``: the defaults bexa detects).
    """

    def __init__(
        self,
        root: str | Path | None = None,
        folders: str | Path | Sequence[str | Path] | None = None,
        scans: Any = None,
        filters: Mapping[str, Any] | None = None,
        *,
        view: str = "list",
        cache: Any = True,
        cache_reductions: bool | None = None,
        interactive: bool | None = None,
        state_file: str | Path | None = None,
        status: Any = "finished",
        page_size: int = 50,
        depth: int = 2,
        motors: Sequence[str] | None = None,
        stack_axis: str | None = None,
        **open_kwargs: Any,
    ) -> None:
        if root is None and folders is None:
            raise ValueError("give root= (a RAW_DATA folder) or folders= (dataset folders)")
        if view not in VIEWS:
            raise ValueError(f"view must be one of {VIEWS}, not {view!r}")
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
        self.view = view
        self.page_size = int(page_size)
        self.state_file = Path(state_file) if state_file else None
        self._status = status
        self.selectable_statuses = status_filter(status)
        self._open_options: dict[str, Any] = {
            "cache": cache,
            "cache_reductions": cache_reductions,
            **open_kwargs,
        }
        self.ui: dict[str, str] = {"type": "any", "motors": "", "status": "any"}
        self.chosen_motors: set[str] | None = None
        self.stack_choice: str | None = None  # None: the default; "none": no stack axis; a name
        self._listeners: list[Callable[[ScanPicker], None]] = []
        self.widgets: Any = None  # the PickerWidgets when shown with ipywidgets
        self._thumbnails: dict[int, bytes] = {}
        self._candidates: tuple[tuple[int, ...], dict[str, MotorCandidate]] | None = None
        self._between: tuple[tuple[int, ...], list[str]] | None = None
        self._scans: tuple[Any, list[Scan]] | None = None
        self.dataset: Dataset = None  # type: ignore[assignment]  (set by _open)
        self.ids: list[int] = []
        self._open(self.code_folders)
        self._select_defaults()
        if motors is not None:
            self.set_motors(motors, notify=False)
        if stack_axis is not None:
            self.set_stack_axis(stack_axis, notify=False)
        self._restore_state()
        if interactive is None:
            interactive = in_kernel()
            if interactive:
                try:
                    import ipywidgets  # noqa: F401
                except ImportError:
                    log.warning("ipywidgets is not installed: the picker shows as a table")
                    interactive = False
        if interactive:
            from bexa.notebook.picker_widgets import PickerWidgets

            self.widgets = PickerWidgets(self)

    # ------------------------------------------------------------------ opening
    def _open(self, folders: Sequence[Path]) -> None:
        if self.dataset is not None:
            self.dataset.close()
        self.folders = [Path(f) for f in folders]
        self.dataset = Dataset(self.folders, **self._open_options)
        self._list()

    def _list(self) -> None:
        """Open every scan (with a progress bar), build the tables and the listing."""
        ds = self.dataset
        for scan_id in progress(ds.scans, desc="opening scans", enabled=len(ds.scans) > 8):
            with contextlib.suppress(ScanNotReadyError):
                ds.scan(scan_id)
        self.rows = ds.table()
        compact = ds.table(compact=True)
        self.varying_columns = [
            str(c) for c in compact.columns if c not in LEAD_COLUMNS and c not in FIXED_COLUMNS
        ]
        self.listed_by_code = ds.select_numbers(None, status="any", **self.code_filters)
        self._invalidate()

    def _invalidate(self) -> None:
        self._candidates = self._between = self._scans = None

    def _select_defaults(self) -> None:
        """The code-level selection: ``scans`` through the filters, or every selectable scan."""
        if self.code_scans is None:
            wanted = self.selectable
        else:
            wanted = self.dataset.select_numbers(  # status="any": the picker itself leaves the
                self.code_scans,
                status="any",
                **self.code_filters,  # unfinished scans out, quietly
            )
        allowed = set(self.selectable)
        self.ids = [int(i) for i in wanted if int(i) in allowed]

    def _changed(self) -> None:
        self._invalidate()
        self.save_state()
        for callback in self._listeners:
            callback(self)

    # ------------------------------------------------------------------ listing
    @property
    def candidates(self) -> list[int]:
        """The ids listed: the code-level filters, then the type, motors and status of the bars."""
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
        return keep

    @property
    def selectable(self) -> list[int]:
        """The listed ids whose status allows them to be selected (finished scans by default)."""
        allowed = self.selectable_statuses
        if allowed is None:
            return self.candidates
        status = self.rows.set_index("id")["status"]
        return [i for i in self.candidates if ScanStatus(str(status.loc[i])) in allowed]

    def set_filters(self, **ui: str) -> None:
        """Narrow the listing: ``type="fscan3d"``, ``motors="chi x mu"``, ``status="finished"``."""
        unknown = sorted(set(ui) - set(self.ui))
        if unknown:
            raise ValueError(f"unknown filters {unknown}; the bars filter by {sorted(self.ui)}")
        if "status" in ui and ui["status"] not in STATUS_SHOWN:
            raise ValueError(f"status must be one of {sorted(STATUS_SHOWN)}")
        self.ui.update({k: str(v) for k, v in ui.items()})
        allowed = set(self.selectable)
        self.ids = [i for i in self.ids if i in allowed]
        self._changed()

    def set_folders(self, folders: str | Path | Sequence[str | Path]) -> None:
        """Open other dataset folder(s); the code-level selection applies again to them."""
        given = [folders] if isinstance(folders, (str, Path)) else list(folders)
        self._open([Path(f) for f in given])
        self._thumbnails.clear()
        self._select_defaults()
        self._changed()

    def refresh(self) -> None:
        """Look at the folders again (scans finished or started since); the selection stays."""
        self.dataset.refresh()
        self._list()
        allowed = set(self.selectable)
        self.ids = [i for i in self.ids if i in allowed]
        self._changed()

    def reset(self) -> None:
        """Back to the code defaults: the folders, the selection, the motors and the filters."""
        self.ui = {"type": "any", "motors": "", "status": "any"}
        self.chosen_motors = None
        self.stack_choice = None
        if self.folders != self.code_folders:
            self._open(self.code_folders)
        self._select_defaults()
        self._changed()

    # ---------------------------------------------------------------- selection
    def select(self, ids: Iterable[int]) -> None:
        """Select these ids (listed and selectable ones; the others are reported and dropped)."""
        wanted = [int(i) for i in ids]
        allowed = set(self.selectable)
        dropped = [i for i in wanted if i not in allowed]
        if dropped:
            log.warning("scans %s are not listed or not selectable: left out", dropped)
        self.ids = [i for i in wanted if i in allowed]
        self._changed()

    def select_all(self) -> None:
        """Select every listed scan that can be selected."""
        self.select(self.selectable)

    def clear(self) -> None:
        """Select nothing."""
        self.select([])

    def observe(self, callback: Callable[[ScanPicker], None]) -> None:
        """Call ``callback(picker)`` after every change of the selection, motors or filters."""
        self._listeners.append(callback)

    # ------------------------------------------------------------------- motors
    def motor_candidates(self) -> dict[str, MotorCandidate]:
        """Every motor that moves within the selected scans or differs between them."""
        key = tuple(self.ids)
        if self._candidates is None or self._candidates[0] != key:
            found = self.dataset.motor_candidates(self.ids, status="any") if self.ids else {}
            self._candidates = (key, found)
        return self._candidates[1]

    @property
    def motors(self) -> list[str]:
        """The motors to extract: the pressed buttons, else the defaults bexa detects."""
        candidates = self.motor_candidates()
        if self.chosen_motors is None:
            return [name for name, c in candidates.items() if c.default]
        return [name for name in candidates if name in self.chosen_motors]

    def set_motors(self, names: Iterable[str], notify: bool = True) -> None:
        """Press exactly these motors (an unknown name raises ``ValueError``)."""
        chosen = {str(n) for n in names}
        candidates = self.motor_candidates()
        unknown = sorted(chosen - set(candidates))
        if unknown and self.ids:
            raise ValueError(
                f"no motor {unknown} in the selected scans; choose from {list(candidates)}"
            )
        self.chosen_motors = chosen
        if notify:
            self._changed()

    def toggle_motor(self, name: str) -> None:
        """Press a released motor, release a pressed one."""
        self.set_motors(set(self.motors) ^ {str(name)})

    def reset_motors(self) -> None:
        """Back to the motors and the stack axis bexa detects."""
        self.chosen_motors = None
        self.stack_choice = None
        self._changed()

    def _pressed(self, where: str) -> list[str]:
        candidates = self.motor_candidates()
        return [m for m in self.motors if candidates[m].where == where]

    def _between_scans(self) -> list[str]:
        """The pressed motors that differ between the scans, the leading one first."""
        key = tuple(self.ids)
        if self._between is None or self._between[0] != key:
            order = list(self.dataset.varying(self.ids, status="any")) if len(self.ids) > 1 else []
            self._between = (key, order)
        order = self._between[1]
        pressed = self._pressed("stack")
        return sorted(pressed, key=lambda m: (m not in order, order.index(m) if m in order else 0))

    @property
    def scans(self) -> list[Scan]:
        """The selected scans, read with the pressed motors as the dims of their grids."""
        within = self._pressed("scan")
        key = (tuple(self.ids), tuple(within))
        if self._scans is None or self._scans[0] != key:
            candidates = self.motor_candidates()
            defaults = [m for m in within if not candidates[m].default] or [
                m
                for m, c in candidates.items()
                if c.where == "scan" and c.default and m not in within
            ]
            opened = []
            for scan_id in self.ids:
                scan = self.dataset.scan(scan_id)
                if defaults and set(within) != set(scan.dims[:-2]):  # other motors than bexa's
                    scan = scan.with_dims(within)
                opened.append(scan)
            self._scans = (key, opened)
        return self._scans[1]

    @property
    def dims(self) -> list[str]:
        """The pressed motors that move within a scan, in loop order (the dims of the maps)."""
        scans = self.scans
        return [str(d) for d in scans[0].dims[:-2]] if scans else self._pressed("scan")

    @property
    def stack_axis(self) -> str | None:
        """The one motor the scans are stacked on (``uz`` for a z-stack), or ``None``."""
        between = self._between_scans()
        if self.stack_choice in between:
            return self.stack_choice
        if self.stack_choice == "none":
            return None
        return between[0] if between else None

    def set_stack_axis(self, name: str | None, notify: bool = True) -> None:
        """Set the stack axis: a pressed motor, "none" or None for none, "default" for bexa's."""
        if name is None or name == "none":
            self.stack_choice = "none"
        elif name == "default":
            self.stack_choice = None
        else:
            between = self._between_scans()
            if name not in between:
                raise ValueError(
                    f"{name!r} is not a pressed motor that differs between the scans; "
                    f"choose from {between} or press it first"
                )
            self.stack_choice = str(name)
        if notify:
            self._changed()

    @property
    def series_axes(self) -> list[str]:
        """The pressed motors that differ between the scans and are not the stack axis."""
        axis = self.stack_axis
        return [m for m in self._between_scans() if m != axis]

    @property
    def scan_axes(self) -> list[str]:
        """Every pressed motor but the stack axis: the dims within a scan, then the series."""
        return [*self.dims, *self.series_axes]

    @property
    def axes(self) -> list[str]:
        """What :func:`bexa.stack` is given as ``dim``: the stack axis, then the series axes."""
        axis = self.stack_axis
        return ([axis] if axis else []) + self.series_axes

    def varying(self) -> dict[str, np.ndarray]:
        """The values of every axis of :attr:`axes` over the selected scans."""
        return varying(self.scans, axes=self.axes) if self.axes else {}

    def roles_text(self) -> str:
        """``"scan axes: chi (9) x mu (21); stack axis: uz (5)"``."""
        candidates = self.motor_candidates()
        scan_axes = " x ".join(f"{m} ({candidates[m].n_values})" for m in self.scan_axes)
        axis = self.stack_axis
        stack = f"{axis} ({candidates[axis].n_values})" if axis else "none"
        return f"scan axes: {scan_axes or 'frames only'}; stack axis: {stack}"

    # -------------------------------------------------------------------- output
    @property
    def table(self) -> pd.DataFrame:
        """The compact table rows of the selected scans."""
        compact = self.dataset.table(compact=True)
        return compact[compact["id"].isin(self.ids)].reset_index(drop=True)

    def status_text(self) -> str:
        """``"14 of 20 scans selected"``."""
        return f"{len(self.ids)} of {len(self.candidates)} scans selected"

    def summary_line(self, row: Mapping[str, Any]) -> str:
        """One line of the list view: id, dataset, status, type, motors, shape, positioners."""
        parts = [f"{int(row['id']):>3}"]
        if "dataset" in row:
            parts.append(f"{row['dataset']!s:<12}")
        parts += [
            f"{row['status']!s:<9}",
            f"{row['type']!s:<8}",
            f"{row['motors']!s:<20}",
            f"{row['shape']!s:<12}",
        ]
        for column in self.varying_columns:
            value = row.get(column)
            if isinstance(value, (int, float, np.floating)) and pd.notna(value):
                parts.append(
                    f"{value:.4g} keV" if column == "energy_keV" else f"{column} {value:.4g}"
                )
        frames = row.get("frames")
        if frames is not None and pd.notna(frames):
            parts.append(f"{int(frames)} frames")
        return "  ".join(parts)

    def rows_of(self, ids: Iterable[int]) -> pd.DataFrame:
        """The full table rows of these ids, in this order."""
        return self.rows.set_index("id").loc[list(ids)].reset_index()

    def thumbnails(
        self, size: int = 128, images: Mapping[int, Any] | None = None
    ) -> dict[int, bytes]:
        """PNG thumbnails of the summed image of every listed scan (computed once, kept).

        Each scan is summed with a motor stride of 4 and the pixels binned to about
        ``size`` across (one pass per scan, cached by the dataset's cache); ``images``
        maps ids to summed images already at hand.
        """
        import matplotlib.image

        for scan_id in self.candidates:
            if scan_id in self._thumbnails:
                continue
            if images is not None and scan_id in images:
                total = np.asarray(images[scan_id], dtype=float)
            else:
                try:
                    scan = self.dataset.scan(scan_id)
                except ScanNotReadyError:
                    continue
                height, width = scan.frame_shape
                stride = {d: 4 for d in scan.dims[:-2]}
                stride.update({"y": max(1, height // size), "x": max(1, width // size)})
                total = np.asarray(scan.sum(downsample=stride).values, dtype=float)
            finite = total[np.isfinite(total)]
            lo, hi = np.percentile(finite, [1, 99]) if finite.size else (0.0, 1.0)
            picture = np.clip((np.nan_to_num(total, nan=lo) - lo) / max(hi - lo, 1e-9), 0, 1)
            buffer = io.BytesIO()
            matplotlib.image.imsave(buffer, picture, cmap="magma", format="png")
            self._thumbnails[scan_id] = buffer.getvalue()
        if self.widgets is not None:
            self.widgets.refresh()
        return dict(self._thumbnails)

    # --------------------------------------------------------------------- state
    def _defaults_key(self) -> str:
        parts = {
            "root": str(self.root),
            "folders": [str(f) for f in self.code_folders],
            "scans": _stable(self.code_scans),
            "filters": _stable(self.code_filters),
            "status": repr(self._status),
        }
        return hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]

    def _keys(self) -> dict[tuple[str, int], int]:
        """``{(dataset folder name, scan number): id}``: what survives a kernel restart."""
        several = "dataset" in self.rows.columns
        return {
            (str(row["dataset"]) if several else self.folders[0].name, int(row["scan"])): int(
                row["id"]
            )
            for _, row in self.rows.iterrows()
        }

    def save_state(self) -> Path | None:
        """Write the selection, motors, stack axis and filters to ``state_file`` (when set)."""
        if self.state_file is None:
            return None
        ids = {v: k for k, v in self._keys().items()}
        state = {
            "defaults": self._defaults_key(),
            "root": str(self.root),
            "folders": [str(f) for f in self.folders],
            "selected": [list(ids[i]) for i in self.ids if i in ids],
            "ui": dict(self.ui),
            "motors": sorted(self.chosen_motors) if self.chosen_motors is not None else None,
            "stack_axis": self.stack_choice,
            "view": self.view,
        }
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(state, indent=1), encoding="utf-8")
        return self.state_file

    def _restore_state(self) -> None:
        if self.state_file is None or not self.state_file.is_file():
            return
        try:
            state = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("picker state %s ignored: %s", self.state_file, exc)
            return
        if state.get("defaults") != self._defaults_key():
            log.info("picker state %s ignored: the code defaults changed", self.state_file)
            return
        folders = [Path(f) for f in state.get("folders", [])]
        if folders and not all(f.is_dir() for f in folders):
            log.warning("picker state %s ignored: a folder is gone", self.state_file)
            return
        if folders and folders != self.folders:
            self._open(folders)
        keys = self._keys()
        self.ui.update({k: str(v) for k, v in state.get("ui", {}).items() if k in self.ui})
        self.chosen_motors = (
            set(state["motors"]) if state.get("motors") is not None else self.chosen_motors
        )
        self.stack_choice = state.get("stack_axis", self.stack_choice)
        wanted = [keys[(n, int(s))] for n, s in state.get("selected", []) if (n, int(s)) in keys]
        allowed = set(self.selectable)
        self.ids = [i for i in wanted if i in allowed]
        self._invalidate()
        log.info("picker state restored from %s: %s", self.state_file, self.status_text())

    # ------------------------------------------------------------------- display
    def show(self) -> None:
        """Display the widgets (in a kernel) or the table."""
        from IPython.display import HTML, display

        display(self.widgets.box if self.widgets is not None else HTML(self._repr_html_()))

    def _ipython_display_(self) -> None:
        self.show()

    def __repr__(self) -> str:
        where = ", ".join(str(f) for f in self.folders)
        return f"ScanPicker({self.status_text()} in {where}; {self.roles_text()})"

    def _repr_html_(self) -> str:
        where = ", ".join(f"<code>{f}</code>" for f in self.folders)
        caption = (
            f"<b>{self.status_text()}</b> in {where}<br>{self.roles_text()}"
            "<br><i>no widgets here: select with picker.select([...]) or SCANS / FILTERS</i>"
        )
        compact = self.dataset.table(compact=True).set_index("id")
        rows = compact.loc[self.candidates].reset_index()
        rows.insert(1, "selected", rows["id"].isin(self.ids).map({True: "x", False: ""}))
        if "status" not in rows.columns:
            rows.insert(
                2, "status", self.rows.set_index("id").loc[self.candidates, "status"].values
            )
        drop = [c for c in rows.columns if c in ("title", "ranges", "missing")]
        return caption + rows.drop(columns=drop).to_html(index=False)

    def close(self) -> None:
        """Close the dataset's files."""
        self.dataset.close()


def pick_scans(
    root: str | Path | None = None,
    folders: str | Path | Sequence[str | Path] | None = None,
    **kwargs: Any,
) -> ScanPicker:
    """Build a :class:`ScanPicker`, show it and return it."""
    picker = ScanPicker(root, folders, **kwargs)
    picker.show()
    return picker
