"""The ipywidgets views of a :class:`~bexa.notebook.picker.ScanPicker`.

Shared bars (the root and the datasets under it, the type / motors / status
filters, the selection buttons, the status line), then the scan list in one of
three views (``"list"``: one line per scan; ``"table"``: a checkbox per row with
sortable columns; ``"gallery"``: thumbnails), then the motor buttons and the
roles row. Every change goes through the picker, which calls back here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from bexa.notebook.picker import STATUS_SHOWN, VIEWS, ScanPicker

__all__ = ["PickerWidgets"]

STATUS_COLOR = {
    "complete": "#1b5e20",
    "aborted": "#e65100",
    "running": "#9e9e9e",
    "empty": "#9e9e9e",
    "unreadable": "#9e9e9e",
}
MONO = "font-family: monospace; white-space: pre; font-size: 12px"
WIDTHS = {"id": 5, "dataset": 12, "status": 9, "type": 8, "motors": 20, "shape": 12, "frames": 7}
CSS = (
    "<style>.bexa-mono select, .bexa-mono option { font-family: monospace; font-size: 12px; }"
    " .bexa-head button { font-weight: bold; font-family: monospace; font-size: 12px; }</style>"
)
HINT = (
    "<span style='color:#757575;font-size:11px'>no widgets? run bin/esrf_setup.sh with "
    "BEXA_LINK_LABEXTENSIONS=1, or ScanPicker(..., interactive=False) and select in code</span>"
)


class PickerWidgets:
    """The widget box of a picker: ``box`` is what a cell displays."""

    def __init__(self, picker: ScanPicker) -> None:
        import ipywidgets as ipw

        self.picker = picker
        self.sort: tuple[str, bool] = ("id", True)
        self.page = 0
        self.checkboxes: dict[int, Any] = {}
        self._syncing = False
        m = picker
        self.root_text = ipw.Text(
            value=str(m.root), description="root", layout=ipw.Layout(width="70%")
        )
        self.rescan = ipw.Button(
            description="Rescan", icon="refresh", layout=ipw.Layout(width="100px")
        )
        self.rescan.on_click(lambda _b: self._rescan())
        self.folder_select = ipw.SelectMultiple(
            description="datasets", layout=ipw.Layout(width="70%"), rows=2
        )
        self.open_button = ipw.Button(
            description="Open", button_style="primary", layout=ipw.Layout(width="100px")
        )
        self.open_button.on_click(lambda _b: self._open())
        self._fill_datasets()
        folder_bar = ipw.VBox(
            [
                ipw.HBox([self.root_text, self.rescan]),
                ipw.HBox([self.folder_select, self.open_button]),
            ]
        )
        self.type_dd = ipw.Dropdown(description="type", layout=ipw.Layout(width="190px"))
        self.motors_text = ipw.Text(
            description="motors", placeholder="chi x mu", layout=ipw.Layout(width="230px")
        )
        self.status_dd = ipw.Dropdown(
            options=[
                ("any status", "any"),
                ("finished scans", "finished"),
                ("complete only", "complete"),
            ],
            description="show",
            layout=ipw.Layout(width="200px"),
        )
        self.view_dd = ipw.Dropdown(
            options=list(VIEWS),
            value=picker.view,
            description="view",
            layout=ipw.Layout(width="170px"),
        )
        for widget in (self.type_dd, self.motors_text, self.status_dd):
            widget.observe(lambda _ch: self._filters_changed(), names="value")
        self.view_dd.observe(lambda ch: self._set_view(ch["new"]), names="value")
        filter_bar = ipw.HBox([self.type_dd, self.motors_text, self.status_dd, self.view_dd])
        self.all_button = ipw.Button(description="All shown", layout=ipw.Layout(width="100px"))
        self.none_button = ipw.Button(description="None", layout=ipw.Layout(width="80px"))
        self.invert_button = ipw.Button(description="Invert", layout=ipw.Layout(width="80px"))
        self.reset_button = ipw.Button(
            description="Reset", icon="undo", layout=ipw.Layout(width="90px")
        )
        self.refresh_button = ipw.Button(
            description="Refresh", icon="sync", layout=ipw.Layout(width="100px")
        )
        self.all_button.on_click(lambda _b: m.select_all())
        self.none_button.on_click(lambda _b: m.clear())
        self.invert_button.on_click(
            lambda _b: m.select([i for i in m.selectable if i not in m.ids])
        )
        self.reset_button.on_click(lambda _b: m.reset())
        self.refresh_button.on_click(lambda _b: m.refresh())
        self.status = ipw.HTML()
        buttons = ipw.HBox(
            [
                self.all_button,
                self.none_button,
                self.invert_button,
                self.reset_button,
                self.refresh_button,
                self.status,
            ]
        )
        self.body = ipw.VBox()
        self.motor_row = ipw.HBox(layout=ipw.Layout(flex_flow="row wrap", align_items="center"))
        self.motor_reset = ipw.Button(
            description="Default motors", icon="undo", layout=ipw.Layout(width="140px")
        )
        self.motor_reset.on_click(lambda _b: m.reset_motors())
        self.roles_text = ipw.HTML()
        self.roles_row = ipw.HBox(layout=ipw.Layout(align_items="center"))
        motor_box = ipw.VBox(
            [
                ipw.HTML(
                    "<b>1. motors to extract</b> <span style='color:#757575'>(pressed = "
                    "extracted; the defaults are what bexa detects)</span>"
                ),
                self.motor_row,
                ipw.HTML(
                    "<b>2. roles</b> <span style='color:#757575'>(one motor is the stack axis, "
                    "every other extracted motor is a scan axis)</span>"
                ),
                self.roles_row,
            ],
            layout=ipw.Layout(border_top="1px solid #e0e0e0", padding="6px 0", margin="6px 0"),
        )
        self.box = ipw.VBox(
            [ipw.HTML(CSS), folder_bar, filter_bar, buttons, self.body, motor_box, ipw.HTML(HINT)]
        )
        self.box.add_class("bexa-mono")
        picker.observe(lambda _p: self.refresh())
        self.refresh()

    # ------------------------------------------------------------------ the bars
    def _fill_datasets(self) -> None:
        m = self.picker
        options = []
        for path in m.datasets:
            try:
                label = str(path.relative_to(m.root))
            except ValueError:
                label = str(path)
            options.append((label, str(path)))
        self.folder_select.options = options
        self.folder_select.rows = min(4, max(2, len(options)))
        self.folder_select.value = tuple(
            str(f) for f in m.folders if str(f) in dict(options).values()
        )

    def _rescan(self) -> None:
        from bexa.notebook.picker import list_datasets

        m = self.picker
        from pathlib import Path

        m.root = Path(self.root_text.value)
        m.datasets = list_datasets(m.root)
        self._fill_datasets()

    def _open(self) -> None:
        if self.folder_select.value:
            self.picker.set_folders(list(self.folder_select.value))

    def _filters_changed(self) -> None:
        if self._syncing:
            return
        self.picker.set_filters(
            type=self.type_dd.value, motors=self.motors_text.value, status=self.status_dd.value
        )

    def _set_view(self, view: str) -> None:
        self.picker.view = view
        self.page = 0
        self._build_body()

    def refresh(self) -> None:
        """Rebuild what the picker's state decides: bars, status line, body, motors, roles."""
        m = self.picker
        self._syncing = True
        try:
            types = ["any", *sorted(set(map(str, m.rows["type"].dropna())))]
            self.type_dd.options = types
            self.type_dd.value = m.ui["type"] if m.ui["type"] in types else "any"
            self.motors_text.value = m.ui["motors"]
            self.status_dd.value = m.ui["status"] if m.ui["status"] in STATUS_SHOWN else "any"
            self.view_dd.value = m.view
            self._fill_datasets()
        finally:
            self._syncing = False
        self.status.value = f"<b>&nbsp;&nbsp;{m.status_text()}</b>"
        self._build_motors()
        self._build_body()

    def _build_motors(self) -> None:
        """One toggle button per motor that moves; the defaults pressed, any of them changeable."""
        import ipywidgets as ipw

        m = self.picker
        pressed = set(m.motors)
        buttons = []
        for name, info in m.motor_candidates().items():
            where = "per scan" if info.where == "scan" else "between scans"
            button = ipw.ToggleButton(
                value=name in pressed,
                description=f"{name} ({info.n_values})",
                tooltip=f"{name}: {info.n_values} values {where}"
                + (", a default" if info.default else ""),
                button_style="primary" if name in pressed else "",
                layout=ipw.Layout(width="auto", margin="2px 4px 2px 0"),
            )
            button.add_class("bexa-motor")
            button.observe(lambda ch, n=name: self._motor_toggled(n, ch["new"]), names="value")
            buttons.append(button)
        self.motor_row.children = buttons or [ipw.HTML("<i>select a scan first</i>")]
        between = m._between_scans()
        axis = m.stack_axis
        self.stack_dropdown = ipw.Dropdown(
            options=["none", *between],
            value=axis or "none",
            description="stack axis",
            layout=ipw.Layout(width="200px"),
        )
        self.stack_dropdown.observe(lambda ch: m.set_stack_axis(ch["new"]), names="value")
        self.roles_text.value = f"&nbsp;&nbsp;{m.roles_text()}"
        self.roles_row.children = [self.stack_dropdown, self.motor_reset, self.roles_text]

    def _motor_toggled(self, name: str, pressed: bool) -> None:
        m = self.picker
        chosen = set(m.motors)
        if pressed:
            chosen.add(name)
        else:
            chosen.discard(name)
        m.set_motors(chosen)

    # ------------------------------------------------------------------ the views
    def _build_body(self) -> None:
        view = self.picker.view
        if view == "list":
            self.body.children = [self._list()]
        elif view == "table":
            self.body.children = [self._table()]
        else:
            self.body.children = [self._gallery()]

    def _page_rows(self) -> tuple[pd.DataFrame, Any]:
        """The rows of the current page of the table and gallery views, and the page control."""
        import ipywidgets as ipw

        m = self.picker
        rows = m.rows_of(m.candidates)
        column, ascending = self.sort
        if column in rows.columns:
            rows = rows.sort_values(column, ascending=ascending, kind="stable")
        pages = max(1, int(np.ceil(len(rows) / m.page_size)))
        self.page = min(self.page, pages - 1)
        control = None
        if pages > 1:
            control = ipw.Dropdown(
                options=[(f"page {k + 1} of {pages}", k) for k in range(pages)],
                value=self.page,
                layout=ipw.Layout(width="150px"),
            )
            control.observe(lambda ch: self._turn_page(ch["new"]), names="value")
        start = self.page * m.page_size
        return rows.iloc[start : start + m.page_size], control

    def _turn_page(self, page: int) -> None:
        self.page = int(page)
        self._build_body()

    def _toggle(self, scan_id: int, selected: bool) -> None:
        m = self.picker
        ids = [i for i in m.ids if i != scan_id] + ([scan_id] if selected else [])
        m.select(sorted(ids))

    def _list(self) -> Any:
        import ipywidgets as ipw

        m = self.picker
        rows = m.rows_of(m.candidates)
        options = [(m.summary_line(row), int(row["id"])) for _, row in rows.iterrows()]
        select = ipw.SelectMultiple(
            options=options,
            value=tuple(i for i in m.ids if i in m.candidates),
            rows=min(18, max(4, len(options))),
            layout=ipw.Layout(width="100%"),
        )
        select.add_class("bexa-list")

        def changed(change: Any) -> None:
            chosen = [int(i) for i in change["new"]]
            if chosen != m.ids:  # a running row snaps back: the list is rebuilt from the model
                m.select(chosen)

        select.observe(changed, names="value")
        return select

    def _table(self) -> Any:
        import ipywidgets as ipw

        m = self.picker
        rows, control = self._page_rows()
        columns = ["id", *(["dataset"] if "dataset" in rows.columns else []), "status", "type"]
        columns += ["motors", "shape", *m.varying_columns, "frames"]
        column, ascending = self.sort
        header = []
        for name in columns:
            label = name + (
                " v" if name == column and ascending else " ^" if name == column else ""
            )
            button = ipw.Button(
                description=label,
                layout=ipw.Layout(width=f"{WIDTHS.get(name, 10) * 8 + 8}px", height="24px"),
            )
            button.on_click(lambda _b, c=name: self._sort_by(c))
            header.append(button)
        head = ipw.HBox([ipw.HTML(" ", layout=ipw.Layout(width="30px")), *header])
        head.add_class("bexa-head")
        selectable = set(m.selectable)
        self.checkboxes = {}
        lines = []
        for _, row in rows.iterrows():
            scan_id = int(row["id"])
            cells = []
            for name in columns:
                value = row[name]
                if isinstance(value, (float, np.floating)):
                    value = f"{value:.4g}" if pd.notna(value) else ""
                cells.append(f"{value!s:<{WIDTHS.get(name, 10)}}")
            box = ipw.Checkbox(
                value=scan_id in m.ids,
                indent=False,
                disabled=scan_id not in selectable,
                layout=ipw.Layout(width="30px"),
            )
            box.add_class("bexa-row")
            box.observe(lambda ch, i=scan_id: self._toggle(i, ch["new"]), names="value")
            self.checkboxes[scan_id] = box
            color = STATUS_COLOR.get(str(row["status"]), "black")
            text = ipw.HTML(f"<div style='{MONO}; color:{color}'>{' '.join(cells)}</div>")
            lines.append(ipw.HBox([box, text]))
        children = [head, *lines]
        if control is not None:
            children.append(control)
        return ipw.VBox(children)

    def _sort_by(self, column: str) -> None:
        current, ascending = self.sort
        self.sort = (column, not ascending if column == current else True)
        self._build_body()

    def _gallery(self) -> Any:
        import ipywidgets as ipw

        m = self.picker
        rows, control = self._page_rows()
        self.size_dd = ipw.Dropdown(
            options=[96, 128, 192], value=128, description="size", layout=ipw.Layout(width="150px")
        )
        compute = ipw.Button(description="Compute thumbnails", icon="image", button_style="info")
        compute.on_click(lambda _b: m.thumbnails(size=int(self.size_dd.value)))
        note = ipw.Label("thumbnails are computed on demand (one pass per scan, cached afterwards)")
        selectable = set(m.selectable)
        self.checkboxes = {}
        tiles = []
        size = int(self.size_dd.value)
        height = size * 60 // 72
        for _, row in rows.iterrows():
            scan_id = int(row["id"])
            png = m._thumbnails.get(scan_id)
            if png is not None:
                picture: Any = ipw.Image(value=png, format="png", width=size, height=height)
            else:
                picture = ipw.HTML(
                    f"<div style='width:{size}px;height:{height}px;background:#eeeeee;"
                    "color:#9e9e9e;display:flex;align-items:center;justify-content:center;"
                    "font-size:11px'>no thumbnail yet</div>"
                )
            box = ipw.Checkbox(
                value=scan_id in m.ids,
                indent=False,
                disabled=scan_id not in selectable,
                description=f"{scan_id}  {row['type']}",
                layout=ipw.Layout(width="170px"),
            )
            box.observe(lambda ch, i=scan_id: self._toggle(i, ch["new"]), names="value")
            self.checkboxes[scan_id] = box
            extras = "  ".join(
                f"{c} {row[c]:.4g}"
                for c in m.varying_columns
                if isinstance(row[c], (float, np.floating)) and pd.notna(row[c])
            )
            color = STATUS_COLOR.get(str(row["status"]), "black")
            caption = ipw.HTML(
                f"<div style='{MONO}; color:{color}; font-size:11px'>{row['motors']}<br>"
                f"{extras}<br>{row['status']}</div>"
            )
            tiles.append(
                ipw.VBox(
                    [picture, box, caption],
                    layout=ipw.Layout(border="1px solid #e0e0e0", padding="3px"),
                )
            )
        grid = ipw.GridBox(
            tiles,
            layout=ipw.Layout(grid_template_columns="repeat(auto-fill, 180px)", grid_gap="6px"),
        )
        children = [ipw.HBox([self.size_dd, compute, note]), grid]
        if control is not None:
            children.append(control)
        return ipw.VBox(children)
