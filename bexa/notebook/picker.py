"""Choosing scans in a notebook: the datasets under a root, the scans of any of them, the motors.

:class:`ScanPicker` holds what the notebook's data cell used to spell out by hand
(``FOLDERS``, ``SCANS``, ``FILTERS``, the stack axes): the dataset folders under a
``RAW_DATA`` root, the scans listed and selected (:mod:`bexa.notebook.listing`;
the selection may span folders), the motors to extract and which of them is the
stack axis. In a Jupyter kernel with ipywidgets it shows as a widget with a
panel of the selection (:mod:`bexa.notebook.picker_widgets`); anywhere else as
a table, and the same methods drive it from code.
"""

from __future__ import annotations

import io
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bexa._log import get_logger, in_kernel
from bexa.core import scan_table
from bexa.core.dataset import varying
from bexa.core.scan import Scan
from bexa.core.scan_table import MotorCandidate
from bexa.core.status import ScanNotReadyError
from bexa.notebook.listing import STATUS_SHOWN, ScanListing, list_datasets
from bexa.notebook.picker_state import PickerState, defaults_key
from bexa.notebook.selection import folder_path

log = get_logger(__name__)

__all__ = ["STATUS_SHOWN", "VIEWS", "ScanPicker", "list_datasets", "pick_scans"]

VIEWS = ("list", "table", "gallery")


class ScanPicker(ScanListing):
    """The scans of a beamtime to analyse: which folders, which scans, which motors.

    The scans of the folders listed are shown (several folders: one list with
    running ids). The selection holds the scans chosen in any folder: listing
    another folder keeps them and choosing there adds to them. :attr:`ids` are
    the selected scans among those listed, :attr:`selection` every selected
    scan as ``(folder, scan number)``, and :attr:`scans` opens them all with
    the pressed motors as their dims.

    Parameters
    ----------
    root
        A ``RAW_DATA`` folder whose datasets are listed (:func:`list_datasets`);
        ``None`` lists the given ``folders`` only.
    folders
        The dataset folder(s) listed at first, as one :class:`bexa.core.dataset.Dataset`
        with running ids; ``None`` lists the first dataset under ``root``.
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
        the buttons (``None``: the defaults bexa detects). Like the buttons, the
        motors are kept as presses and releases relative to the defaults, so
        another selection starts from its own defaults with them on top.
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
        if view not in VIEWS:
            raise ValueError(f"view must be one of {VIEWS}, not {view!r}")
        self.view = view
        self.page_size = int(page_size)
        self.state_file = Path(state_file) if state_file else None
        # the motor buttons as changes to the defaults, kept from one selection to the next
        self.pressed_extra: set[str] = set()  # pressed although bexa would leave them released
        self.released: set[str] = set()  # defaults released
        self.stack_choice: str | None = None  # None: the default; "none": no stack axis; a name
        self.widgets: Any = None  # the PickerWidgets when shown with ipywidgets
        self._thumbnails: dict[tuple[Path, int], bytes] = {}
        self._candidates: tuple[Any, dict[str, MotorCandidate]] | None = None
        self._between: tuple[Any, list[str]] | None = None
        self._base: tuple[Any, list[Scan]] | None = None
        self._scans: tuple[Any, list[Scan]] | None = None
        super().__init__(
            root,
            folders,
            scans,
            filters,
            status=status,
            depth=depth,
            cache=cache,
            cache_reductions=cache_reductions,
            **open_kwargs,
        )
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

    def _invalidate(self) -> None:
        self._candidates = self._between = self._base = self._scans = None

    def _changed(self) -> None:
        self._invalidate()
        self.save_state()
        super()._changed()

    def reset(self) -> None:
        """Back to the code defaults: the folders, the selection, the motors and the filters.

        The scans selected in other folders are dropped.
        """
        self.pressed_extra, self.released = set(), set()
        self.stack_choice = None
        super().reset()

    # ------------------------------------------------------------------- motors
    def _base_scans(self) -> list[Scan]:
        """The selected scans as bexa opens them (its own dims), in :attr:`selection` order."""
        key = tuple(self.selection)
        if self._base is None or self._base[0] != key:
            self._base = (key, [self._selection.scan(k) for k in key])
        return self._base[1]

    def motor_candidates(self) -> dict[str, MotorCandidate]:
        """Every motor that moves within the selected scans or differs between them."""
        key = tuple(self.selection)
        if self._candidates is None or self._candidates[0] != key:
            scans = self._base_scans()
            found: dict[str, MotorCandidate] = {}
            if scans:
                rows = pd.DataFrame([self._selection.row(k) for k in key])
                defaults = list(varying(scans)) if len(scans) > 1 else []
                found = scan_table.motor_candidates(scans[0], rows, defaults)
            self._candidates = (key, found)
        return self._candidates[1]

    @property
    def motors(self) -> list[str]:
        """The motors to extract: the defaults bexa detects for the selection, with the
        buttons pressed and released on top."""
        return [
            name
            for name, c in self.motor_candidates().items()
            if (c.default and name not in self.released) or name in self.pressed_extra
        ]

    def set_motors(self, names: Iterable[str], notify: bool = True) -> None:
        """Press exactly these motors (an unknown name raises ``ValueError``).

        The choice is kept as presses and releases relative to the defaults, so
        another selection starts from its own defaults with them on top: a
        z-stack selected after a button was pressed for an energy mosa still
        gets ``uz``.
        """
        chosen = {str(n) for n in names}
        candidates = self.motor_candidates()
        unknown = sorted(chosen - set(candidates))
        if unknown and self.selection:
            raise ValueError(
                f"no motor {unknown} in the selected scans; choose from {list(candidates)}"
            )
        for name, candidate in candidates.items():
            self.pressed_extra.discard(name)
            self.released.discard(name)
            if name in chosen and not candidate.default:
                self.pressed_extra.add(name)
            elif name not in chosen and candidate.default:
                self.released.add(name)
        self.pressed_extra |= set(unknown)  # nothing selected yet: pressed once they appear
        if notify:
            self._changed()

    def toggle_motor(self, name: str) -> None:
        """Press a released motor, release a pressed one."""
        self.set_motors(set(self.motors) ^ {str(name)})

    def reset_motors(self) -> None:
        """Back to the motors and the stack axis bexa detects."""
        self.pressed_extra, self.released = set(), set()
        self.stack_choice = None
        self._changed()

    def _pressed(self, where: str) -> list[str]:
        candidates = self.motor_candidates()
        return [m for m in self.motors if candidates[m].where == where]

    def _between_scans(self) -> list[str]:
        """The pressed motors that differ between the scans, the leading one first."""
        key = tuple(self.selection)
        if self._between is None or self._between[0] != key:
            scans = self._base_scans()
            self._between = (key, list(varying(scans)) if len(scans) > 1 else [])
        order = self._between[1]
        pressed = self._pressed("stack")
        return sorted(pressed, key=lambda m: (m not in order, order.index(m) if m in order else 0))

    def _layered(self) -> str | None:
        """The stack axis when it moves within the scans (they are split into its layers)."""
        axis = self.stack_axis
        return axis if axis is not None and axis in self._pressed("scan") else None

    @property
    def scans(self) -> list[Scan]:
        """The selected scans of every folder, read with the pressed motors as their dims.

        When the stack axis moves within the scans, every scan is split into its
        layers along it (:meth:`bexa.core.scan.Scan.layers`), one per point.
        """
        within = self._pressed("scan")
        layered = self._layered()
        key = (tuple(self.selection), tuple(within), layered)
        if self._scans is None or self._scans[0] != key:
            candidates = self.motor_candidates()
            changed = [m for m in within if not candidates[m].default] or [
                m
                for m, c in candidates.items()
                if c.where == "scan" and c.default and m not in within
            ]
            # the stack axis first: of motors that step together, the first named is the dim
            motors = [layered, *(m for m in within if m != layered)] if layered else within
            opened = []
            for scan in self._base_scans():
                dims = set(scan.dims[:-2])
                if (changed and set(within) != dims) or (layered and layered not in dims):
                    scan = scan.with_dims(motors)  # other motors than bexa's
                opened.extend(scan.layers(layered) if layered else [scan])
            self._scans = (key, opened)
        return self._scans[1]

    @property
    def dims(self) -> list[str]:
        """The pressed motors that move within a scan but the stack axis, in loop order.

        These are the dims of the maps.
        """
        scans = self.scans
        if scans:
            return [str(d) for d in scans[0].dims[:-2]]
        return [m for m in self._pressed("scan") if m != self._layered()]

    def stack_candidates(self) -> list[str]:
        """The pressed motors the scans can be stacked on: those that differ between the
        scans (the leading one first), then those that move within them."""
        return [*self._between_scans(), *self._pressed("scan")]

    @property
    def stack_axis(self) -> str | None:
        """The one motor the scans are stacked on (``uz`` for a z-stack), or ``None``.

        By default the leading motor that differs between the scans; any pressed
        motor when chosen (:meth:`set_stack_axis`).
        """
        if self.stack_choice in self.stack_candidates():
            return self.stack_choice
        if self.stack_choice == "none":
            return None
        between = self._between_scans()
        return between[0] if between else None

    def set_stack_axis(self, name: str | None, notify: bool = True) -> None:
        """Set the stack axis: any pressed motor, "none" or None for none, "default" for bexa's.

        A motor that moves within the scans splits every scan into its layers
        along it, which then stack the way separate scans do.
        """
        if name is None or name == "none":
            self.stack_choice = "none"
        elif name == "default":
            self.stack_choice = None
        else:
            candidates = self.stack_candidates()
            if name not in candidates:
                raise ValueError(
                    f"{name!r} is not a pressed motor; choose from {candidates} or press it first"
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
        if self._layered():
            stack += ", each scan split into its layers"
        return f"scan axes: {scan_axes or 'frames only'}; stack axis: {stack}"

    # ---------------------------------------------------------------- thumbnails
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
            key = self._key_of[scan_id]
            if key in self._thumbnails:
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
            self._thumbnails[key] = buffer.getvalue()
        if self.widgets is not None:
            self.widgets.refresh()
        return {
            i: self._thumbnails[self._key_of[i]]
            for i in self.candidates
            if self._key_of[i] in self._thumbnails
        }

    # --------------------------------------------------------------------- state
    def _defaults_key(self) -> str:
        return defaults_key(
            self.root, self.code_folders, self.code_scans, self.code_filters, self._status
        )

    def save_state(self) -> Path | None:
        """Write the folders, the selection, the motor buttons, the stack axis and the filters."""
        if self.state_file is None:
            return None
        return PickerState(
            defaults=self._defaults_key(),
            root=str(self.root),
            folders=[str(f) for f in self.folders],
            selected=[[str(folder), number] for folder, number in self.selection],
            ui=dict(self.ui),
            pressed=sorted(self.pressed_extra),
            released=sorted(self.released),
            stack_axis=self.stack_choice,
            view=self.view,
        ).write(self.state_file)

    def _restore_state(self) -> None:
        """Take the choices of the state file when it was saved under these code defaults."""
        if self.state_file is None:
            return
        state = PickerState.read(self.state_file, self._defaults_key())
        if state is None:
            return
        folders = [Path(f) for f in state.folders]
        if folders and all(f.is_dir() for f in folders):
            try:
                self._open(folders)
            except ValueError as exc:
                log.warning("picker state %s: %s; the code folders listed", self.state_file, exc)
        elif folders:
            log.warning(
                "picker state %s: a folder is gone; the code folders listed", self.state_file
            )
        self.ui.update({k: str(v) for k, v in state.ui.items() if k in self.ui})
        self._listed = None
        self.pressed_extra, self.released = set(state.pressed), set(state.released)
        self.stack_choice = state.stack_axis
        keys = []
        for folder, number in state.selected:
            path = Path(folder)
            if not path.is_absolute():  # saved before the selection spanned folders: a name
                path = next((f for f in self.folders if f.name == folder), path)
            keys.append((folder_path(path), int(number)))
        kept = {key for key in keys if self._selection.allowed(key)}
        if len(kept) < len(keys):
            log.warning(
                "picker state %s: %d selected scans are gone or cannot be selected: left out",
                self.state_file,
                len(keys) - len(kept),
            )
        self._selection.keys = kept
        self._invalidate()
        log.info("picker state restored from %s: %s", self.state_file, self.status_text())

    # ------------------------------------------------------------------- display
    def show(self) -> None:
        """Display the widgets (in a kernel) or the table."""
        from IPython.display import HTML, display

        display(self.widgets.box if self.widgets is not None else HTML(self._repr_html_()))

    def _ipython_display_(self) -> None:
        self.show()

    def selection_view(self) -> Any:
        """Another live view of the selection panel, for a cell of its own.

        In JupyterLab, "Create New View for Cell Output" on that cell gives it a
        window of its own. Without widgets: the table of the selection.
        """
        if self.widgets is None:
            return self.table
        return self.widgets.selection_view()

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
        drop = ("title", "ranges", "missing")
        html = caption + rows.drop(columns=[c for c in rows.columns if c in drop]).to_html(
            index=False
        )
        if len(self.selection) > len(self.ids):  # scans selected in other folders too
            table = self.table
            table = table.drop(columns=[c for c in table.columns if c in drop])
            html += f"<br><b>selection</b>: {self.selection_text()}" + table.to_html(index=False)
        return html


def pick_scans(
    root: str | Path | None = None,
    folders: str | Path | Sequence[str | Path] | None = None,
    **kwargs: Any,
) -> ScanPicker:
    """Build a :class:`ScanPicker`, show it and return it."""
    picker = ScanPicker(root, folders, **kwargs)
    picker.show()
    return picker
