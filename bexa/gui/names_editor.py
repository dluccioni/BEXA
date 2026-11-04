"""A scan's names and offsets, edited: an alias, an offset and a scale for each of its motors,
counters, positions and detectors.

**Apply** makes the scan renamed (the operation Names and Offsets, logged as
``ws.run("rename", ...)``): every map, curve and table made of it then carries the
names given, and the values ``scale * value + offset``. **Remember for scans
opened later** also keeps them in the workspace (``ws.set_names``), so every scan
opened afterwards with any of these names opens renamed. The names are always
those the files give: the editor shows a renamed scan by its files' names, with
its aliases filled in.
"""

from __future__ import annotations

import html
from typing import Any

__all__ = ["NamesEditor", "mapping_text"]

MAX_ROWS = 200  # rows shown at once; the find box reaches the others
COLUMNS = "62px minmax(100px, 1.4fr) minmax(90px, 1fr) 70px 70px"  # fits the left column


def mapping_text(values: dict[str, Any]) -> str:
    """``{"a": "b", "c": 1.5}`` as ``"a=b, c=1.5"``, the text the operation's fields take."""
    return ", ".join(f"{k}={v}" for k, v in values.items())


class NamesEditor:
    """Aliases, offsets and scales for one scan object, by the names its files give."""

    def __init__(self, app: Any) -> None:
        import ipywidgets as ipw

        self.app = app
        self.name: str | None = None
        self.rows: list[tuple[str, str, str]] = []  # (kind, name, unit) of the scan
        self.values: dict[str, list[Any]] = {}  # name -> [alias, offset, scale], as typed
        self.head = ipw.HTML()
        self.find = ipw.Text(
            placeholder="find a name",
            continuous_update=False,
            layout=ipw.Layout(width="200px"),
        )
        self.positions = ipw.Checkbox(
            description="positions too", value=False, indent=False, layout=ipw.Layout(width="130px")
        )
        self.find.observe(lambda _c: self._fill(), names="value")
        self.positions.observe(lambda _c: self._fill(), names="value")
        self.grid = ipw.GridBox(
            layout=ipw.Layout(grid_template_columns=COLUMNS, grid_gap="1px 6px", width="100%")
        )
        self.remember = ipw.Checkbox(
            description="Remember for scans opened later",
            value=False,
            indent=False,
            layout=ipw.Layout(width="260px"),
        )
        apply = ipw.Button(description="Apply", icon="check", button_style="primary")
        apply.tooltip = "a new scan object with these names, offsets and scales"
        clear = ipw.Button(description="Clear", tooltip="no alias, offset or scale")
        apply.on_click(lambda _b: self.apply())
        clear.on_click(lambda _b: self._clear())
        self.note = ipw.HTML()
        self.box = ipw.VBox(
            [
                self.head,
                ipw.HBox([self.find, self.positions]),
                self.grid,
                ipw.HBox([apply, clear, self.remember], layout=ipw.Layout(flex_flow="row wrap")),
                self.note,
            ]
        )
        self.box.add_class("bexa-names")

    # ------------------------------------------------------------------ the scan
    def show(self, name: str) -> None:
        """Edit the names of the scan object ``name``: its motors, counters, detectors and (on
        request) positions, the aliases it has, else those the workspace keeps."""
        ws = self.app.ws
        scan = ws[name].data.unnamed()
        s = scan.structure
        self.name = name
        motors = set(s.motor_dims)
        self.rows = [("motor", d, s.units.get(d, "")) for d in s.motor_dims]
        self.rows += [("counter", k, s.units.get(k, "")) for k in s.per_frame if k not in motors]
        # a counter that is a detector too (a netCDF run's) has its counter's row
        self.rows += [("detector", d, "") for d in scan.detectors if d not in s.per_frame]
        listed = motors | set(s.per_frame)  # a motor's fixed position shares its row
        self.rows += [("position", k, s.units.get(k, "")) for k in s.scalars if k not in listed]
        given = ws[name].data.names
        if not any(given.values()):
            given = ws.settings.get("names") or {}
        names, offsets, scales = (given.get(k, {}) for k in ("names", "offsets", "scales"))
        self.values = {
            n: [names.get(n, ""), float(offsets.get(n, 0.0)), float(scales.get(n, 1.0))]
            for _, n, _ in self.rows
        }
        counts = {kind: sum(1 for r in self.rows if r[0] == kind) for kind in ("motor", "counter")}
        self.head.value = (
            f"<b>Names and offsets of {html.escape(name)}</b>: {counts['motor']} motors, "
            f"{counts['counter']} counters, {len(scan.detectors)} detectors; a value becomes "
            "scale x value + offset"
        )
        self.note.value = ""
        self._fill()

    def _fill(self) -> None:
        import ipywidgets as ipw

        needle = self.find.value.strip().lower()
        shown = [
            row
            for row in self.rows
            if (row[0] != "position" or self.positions.value or needle)
            and (not needle or needle in row[1].lower())
        ]
        cells: list[Any] = [
            ipw.HTML(f"<b>{h}</b>")
            for h in ("kind", "name in the files", "alias", "offset", "scale")
        ]
        for kind, name, unit in shown[:MAX_ROWS]:
            alias, offset, scale = self.values[name]
            label = html.escape(name) + (f" <i>({html.escape(unit)})</i>" if unit else "")
            alias_box = ipw.Text(value=alias, placeholder=name, layout=ipw.Layout(width="100%"))
            alias_box.observe(self._typed(name, 0), names="value")
            cells += [ipw.HTML(kind), ipw.HTML(label), alias_box]
            if kind == "detector":  # frames: a name only
                cells += [ipw.HTML(""), ipw.HTML("")]
                continue
            for k, value in ((1, offset), (2, scale)):
                box = ipw.FloatText(value=value, layout=ipw.Layout(width="100%"))
                box.observe(self._typed(name, k), names="value")
                cells.append(box)
        if len(shown) > MAX_ROWS:
            cells.append(ipw.HTML(f"<i>the first {MAX_ROWS}; find the others by name</i>"))
        self.grid.children = cells

    def _typed(self, name: str, k: int) -> Any:
        def changed(change: Any) -> None:
            self.values[name][k] = change["new"]

        return changed

    def _clear(self) -> None:
        for name in self.values:
            self.values[name] = ["", 0.0, 1.0]
        self._fill()

    # ------------------------------------------------------------------ applying
    def given(self) -> tuple[dict[str, str], dict[str, float], dict[str, float]]:
        """The aliases, offsets and scales typed: only those that change something."""
        names = {
            n: str(v[0]).strip() for n, v in self.values.items() if str(v[0]).strip() not in ("", n)
        }
        offsets = {n: float(v[1]) for n, v in self.values.items() if float(v[1] or 0.0) != 0.0}
        scales = {n: float(v[2]) for n, v in self.values.items() if float(v[2] or 1.0) != 1.0}
        return names, offsets, scales

    def apply(self) -> None:
        """A new scan object with the names, offsets and scales typed (and, with **Remember**,
        the workspace keeping them for the scans opened later)."""
        if self.name is None:
            return
        names, offsets, scales = self.given()
        if self.remember.value:
            kept = self.app.ws.settings.get("names") or {}
            mine = {n for _, n, _ in self.rows}
            merged = [
                {**{k: v for k, v in kept.get(kind, {}).items() if k not in mine}, **typed}
                for kind, typed in (("names", names), ("offsets", offsets), ("scales", scales))
            ]
            self.app.ws.set_names(*merged)
        if not (names or offsets or scales):
            self.note.value = "<i>nothing to apply: every name as the files give it</i>"
            return
        values = {
            "names": mapping_text(names),
            "offsets": mapping_text(offsets),
            "scales": mapping_text(scales),
        }
        self.note.value = ""
        self.app.run_op("rename", [self.name], values)
