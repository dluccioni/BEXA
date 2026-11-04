"""The objects panel: every object of the workspace, the selection, and what each one is.

Raw scans and scan sets come first (they are read from files and never
change); results follow in the order they were made. Selecting objects here
sets the inputs of the Analysis and Fit tabs and what the View tab shows.
The **detector** of the scan selected chooses which of its detectors it reads
(``Workspace.use_detector``: the object itself, so the View and every
operation read that one; for a scan set, every scan that holds it).
"""

from __future__ import annotations

import html
from typing import Any

from bexa.gui.forms import kind_label, set_tooltip

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
        self.listing = ipw.SelectMultiple(options=[], rows=8, layout=ipw.Layout(width="100%"))
        self.listing.add_class("bexa-objects")
        self.listing.observe(self._selected, names="value")
        self.detector = ipw.Dropdown(
            description="detector",
            layout=ipw.Layout(width="100%", display="none"),
            style={"description_width": "60px"},
        )
        self.detector.observe(lambda c: self._detector_chosen(c["new"]), names="value")
        set_tooltip(self.detector, "the detector the scan reads: in the View and every operation")
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
        self.notes = ipw.Textarea(placeholder="notes", rows=2, layout=ipw.Layout(width="100%"))
        save_notes = ipw.Button(description="Save notes", layout=ipw.Layout(width="100px"))
        save_notes.on_click(lambda _b: self._save_notes())
        self.details = ipw.HTML(layout=ipw.Layout(width="100%"))
        self.metadata = ipw.Button(
            description="Full metadata",
            icon="list",
            tooltip="everything the scan's files say, on the Browse tab",
            layout=ipw.Layout(width="130px", display="none"),
        )
        self.metadata.on_click(lambda _b: self._show_metadata())
        self.names = ipw.Button(
            description="Names and offsets",
            icon="edit",
            tooltip="aliases, offsets and scales for the scan's motors, counters, detectors",
            layout=ipw.Layout(width="150px", display="none"),
        )
        self.names.on_click(lambda _b: self._show_metadata(names=True))
        self.rois = ipw.HTML()
        # what is selected, its notes and the ROIs: folded away, the list kept short
        self.more = ipw.Accordion(
            children=[
                ipw.VBox(
                    [
                        self.details,
                        ipw.HBox([self.metadata, self.names]),
                        self.notes,
                        save_notes,
                        ipw.HTML("<b>ROIs</b>"),
                        self.rois,
                    ]
                )
            ],
            selected_index=None,
        )
        self.more.set_title(0, "Details")
        self.box = ipw.VBox(
            [
                ipw.HTML(
                    "<b title='raw scans are read-only; every operation makes a new object'>"
                    "Objects</b>"
                ),
                ipw.HBox([self.filter, self.search]),
                self.listing,
                self.detector,
                ipw.HBox(
                    [self.new_name, rename, duplicate, self.delete, self.confirm],
                    layout=ipw.Layout(flex_flow="row wrap"),
                ),
                self.more,
            ],
            layout=ipw.Layout(width="100%", padding="0 0 6px 0", flex="0 0 auto"),
        )
        self.box.add_class("bexa-objects-panel")
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
        self._detector_options()
        if current is None or current not in ws:
            self.details.value = "<i>nothing selected</i>"
            self.notes.value = ""
            self.metadata.layout.display = self.names.layout.display = "none"
            return
        obj = ws[current]
        shown = "" if obj.kind == "scan" else "none"
        self.metadata.layout.display = self.names.layout.display = shown
        self.notes.value = obj.notes
        text = obj.info()
        lines = text.splitlines()
        if len(lines) > 40:
            lines = [*lines[:40], f"... {len(lines) - 40} more lines"]
        self.details.value = (
            "<pre style='font-size:11px;line-height:1.25;white-space:pre-wrap;max-height:260px;"
            f"overflow:auto;margin:0'>{html.escape(chr(10).join(lines))}</pre>"
        )

    def _show_metadata(self, names: bool = False) -> None:
        """The selected scan's full metadata on the Browse tab; its **Names and offsets**
        unfolded with ``names``. **Details** folds, so the tab has the column."""
        current = self.app.current
        if current is None or current not in self.app.ws:
            return
        obj = self.app.ws[current]
        if obj.kind == "scan":
            self.app.browse.show_scan_metadata(obj.name, obj.data, names=names)
            self.app.show_tab("Browse")
            self.more.selected_index = None

    def _show_rois(self) -> None:
        rois = self.app.ws.rois
        if not rois:
            self.rois.value = "<i>none</i>"
            return
        rows = "".join(
            f"<tr><td><b>{html.escape(n)}</b></td><td>{html.escape(_ranges(r))}</td></tr>"
            for n, r in rois.items()
        )
        self.rois.value = f"<table style='font-size:11px'>{rows}</table>"

    # ------------------------------------------------------------------ the detector
    def _scans(self) -> list[Any]:
        """The scans of the object selected: a scan, a scan set's scans, else none."""
        current = self.app.current
        if current is None or current not in self.app.ws:
            return []
        obj = self.app.ws[current]
        if obj.kind == "scan":
            return [obj.data]
        return list(obj.data.scans) if obj.kind == "scanset" else []

    def _detector_options(self) -> None:
        """The detectors of the scan selected (a scan set: those of its scans), the one it
        reads chosen (the first scan's); hidden for other objects, fixed for one detector."""
        scans = self._scans()
        names = list(dict.fromkeys(d for scan in scans for d in scan.detectors))
        reads = scans[0].detector if scans else None
        syncing, self._syncing = self._syncing, True
        try:
            self.detector.options = names
            self.detector.value = reads if reads in names else (names[0] if names else None)
            self.detector.disabled = len(names) < 2
            self.detector.layout.display = "" if names else "none"
        finally:
            self._syncing = syncing

    def _detector_chosen(self, detector: str | None) -> None:
        """Read the scan selected (every scan of a set that holds it) with ``detector``: the
        object itself, so the View and every operation read that detector."""
        current, scans = self.app.current, self._scans()
        if self._syncing or not detector or not scans:
            return
        if all(scan.detector == detector for scan in scans if detector in scan.detectors):
            return
        # the workspace says the object changed, and the panels follow
        done = self.app.guarded(lambda: self.app.ws.use_detector(current, detector))
        if done is None:  # it failed (said in red): the field shows what is read
            self._detector_options()
            return
        self.app.say(f"{current} reads {detector}", "ok")

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
