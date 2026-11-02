"""The objects panel: every object of the workspace, the selection, and what each one is.

Raw scans and scan sets come first (they are read from files and never
change); results follow in the order they were made. Selecting objects here
sets the inputs of the Analysis and Fit tabs and what the View tab shows.
"""

from __future__ import annotations

import html
from typing import Any

from bexa.gui.forms import kind_label

__all__ = ["ObjectsPanel"]

KIND_FILTERS = ("all", "scans", "stack", "image", "map", "curve", "volume", "rgb", "table", "text")


class ObjectsPanel:
    """The left column of the app."""

    def __init__(self, app: Any) -> None:
        import ipywidgets as ipw

        self.app = app
        self._syncing = False
        self.filter = ipw.Dropdown(
            options=KIND_FILTERS, value="all", layout=ipw.Layout(width="110px")
        )
        self.search = ipw.Text(placeholder="filter by name", layout=ipw.Layout(width="150px"))
        self.filter.observe(lambda _c: self.refresh(), names="value")
        self.search.observe(lambda _c: self.refresh(), names="value")
        self.listing = ipw.SelectMultiple(options=[], rows=16, layout=ipw.Layout(width="100%"))
        self.listing.add_class("bexa-objects")
        self.listing.observe(self._selected, names="value")
        self.new_name = ipw.Text(placeholder="new name", layout=ipw.Layout(width="140px"))
        rename = ipw.Button(description="Rename", layout=ipw.Layout(width="80px"))
        duplicate = ipw.Button(
            description="Duplicate",
            layout=ipw.Layout(width="90px"),
            tooltip="a copy with its own data, to try something on",
        )
        self.delete = ipw.ToggleButton(
            description="Delete",
            icon="trash",
            layout=ipw.Layout(width="90px"),
            tooltip="press, then Confirm",
        )
        self.confirm = ipw.Button(
            description="Confirm",
            button_style="danger",
            layout=ipw.Layout(width="80px", display="none"),
        )
        rename.on_click(lambda _b: self._rename())
        duplicate.on_click(lambda _b: self._duplicate())
        self.delete.observe(lambda c: self._arm_delete(c["new"]), names="value")
        self.confirm.on_click(lambda _b: self._delete())
        self.notes = ipw.Textarea(
            placeholder="notes on the selected object", rows=2, layout=ipw.Layout(width="100%")
        )
        save_notes = ipw.Button(description="Save notes", layout=ipw.Layout(width="100px"))
        save_notes.on_click(lambda _b: self._save_notes())
        self.details = ipw.HTML(layout=ipw.Layout(width="100%"))
        self.rois = ipw.HTML()
        self.box = ipw.VBox(
            [
                ipw.HTML(
                    "<b>Objects</b> <span style='color:#757575;font-size:11px'>raw scans are "
                    "read-only; every operation makes a new object</span>"
                ),
                ipw.HBox([self.filter, self.search]),
                self.listing,
                ipw.HBox([self.new_name, rename]),
                ipw.HBox([duplicate, self.delete, self.confirm]),
                self.notes,
                save_notes,
                ipw.HTML("<b>Selected</b>"),
                self.details,
                ipw.HTML("<b>ROIs</b>"),
                self.rois,
            ],
            layout=ipw.Layout(width="100%", padding="0 8px 0 0"),
        )
        self.refresh()

    # ------------------------------------------------------------------ listing
    def _shown(self) -> list[Any]:
        ws = self.app.ws
        wanted = self.filter.value
        needle = self.search.value.strip().lower()
        objects = list(ws.objects.values())
        objects.sort(key=lambda o: (not o.raw, o.created))
        out = []
        for obj in objects:
            if wanted == "scans" and obj.kind not in ("scan", "scanset"):
                continue
            if wanted not in ("all", "scans") and obj.kind != wanted:
                continue
            if needle and needle not in obj.name.lower():
                continue
            out.append(obj)
        return out

    def refresh(self) -> None:
        """List the objects again (after any change of the workspace)."""
        shown = self._shown()
        options = [(kind_label(o), o.name) for o in shown]
        selected = tuple(n for n in self.app.selection if n in {o.name for o in shown})
        self._syncing = True
        try:
            self.listing.options = options
            self.listing.value = selected
        finally:
            self._syncing = False
        self._show_details()
        self._show_rois()

    def select(self, names: list[str]) -> None:
        """Select objects from code (a result just made)."""
        listed = {v for _, v in self.listing.options}
        self._syncing = True
        try:
            self.listing.value = tuple(n for n in names if n in listed)
        finally:
            self._syncing = False
        self._show_details()

    def _selected(self, change: Any) -> None:
        if self._syncing:
            return
        self.app.set_selection(list(change["new"]))

    def _show_details(self) -> None:
        ws = self.app.ws
        current = self.app.current
        if current is None or current not in ws:
            self.details.value = "<i>nothing selected</i>"
            self.notes.value = ""
            return
        obj = ws[current]
        self.notes.value = obj.notes
        text = obj.info()
        lines = text.splitlines()
        if len(lines) > 40:
            lines = [*lines[:40], f"... {len(lines) - 40} more lines"]
        self.details.value = (
            "<pre style='font-size:11px;line-height:1.25;white-space:pre-wrap;max-height:260px;"
            f"overflow:auto;margin:0'>{html.escape(chr(10).join(lines))}</pre>"
        )

    def _show_rois(self) -> None:
        rois = self.app.ws.rois
        if not rois:
            self.rois.value = "<i>none yet: draw or type one on the View tab</i>"
            return
        rows = "".join(
            f"<tr><td><b>{html.escape(n)}</b></td><td>{html.escape(_ranges(r))}</td></tr>"
            for n, r in rois.items()
        )
        self.rois.value = f"<table style='font-size:11px'>{rows}</table>"

    # ------------------------------------------------------------------ actions
    def _rename(self) -> None:
        current = self.app.current
        if current is None or not self.new_name.value.strip():
            self.app.say("select one object and type its new name", "error")
            return
        self.app.guarded(lambda: self.app.ws.rename(current, self.new_name.value.strip()))
        self.new_name.value = ""

    def _duplicate(self) -> None:
        for name in list(self.app.selection):
            self.app.guarded(lambda n=name: self.app.ws.duplicate(n))

    def _arm_delete(self, armed: bool) -> None:
        self.confirm.layout.display = "" if armed else "none"

    def _delete(self) -> None:
        names = list(self.app.selection)
        self.delete.value = False
        if names:
            self.app.guarded(lambda: self.app.ws.remove(*names))
            self.app.set_selection([])

    def _save_notes(self) -> None:
        current = self.app.current
        if current is not None:
            self.app.guarded(lambda: self.app.ws.set_notes(current, self.notes.value))


def _ranges(roi: Any) -> str:
    parts = []
    for dim, (lo, hi) in roi.ranges.items():
        lo_text = "" if lo is None else f"{lo:g}"
        hi_text = "" if hi is None else f"{hi:g}"
        parts.append(f"{dim} {lo_text}:{hi_text}")
    return ", ".join(parts) or "everything"
