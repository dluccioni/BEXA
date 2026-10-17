"""The selection panel of the scan picker: every selected scan, whichever folder it is in.

It sits under the scan list of :class:`~bexa.notebook.picker_widgets.PickerWidgets`;
:meth:`bexa.notebook.picker.ScanPicker.selection_view` shows another live copy
in a cell of its own, which JupyterLab's "Create New View for Cell Output" turns
into a window beside the notebook.
"""

from __future__ import annotations

from bexa.notebook.picker import ScanPicker

__all__ = ["SelectionPanel"]

CSS = (
    "<style>.bexa-selection select, .bexa-selection option "
    "{ font-family: monospace; font-size: 12px; }</style>"
)
EMPTY = "<i>nothing selected yet: choose scans in the list above, in any folder</i>"
HELP = (
    "<span style='color:#757575'>(every folder; highlight scans here to remove them "
    "or to list their folder)</span>"
)


class SelectionPanel:
    """The selected scans of every folder, with Remove, Clear all and List folder buttons."""

    def __init__(self, picker: ScanPicker) -> None:
        import ipywidgets as ipw

        self.picker = picker
        self.title = ipw.HTML()
        self.scans = ipw.SelectMultiple(layout=ipw.Layout(width="100%"))
        self.scans.add_class("bexa-selection")
        self.remove_button = ipw.Button(
            description="Remove",
            icon="minus",
            tooltip="unselect the highlighted scans",
            layout=ipw.Layout(width="100px"),
        )
        self.clear_button = ipw.Button(
            description="Clear all",
            icon="trash",
            tooltip="unselect every scan, in every folder",
            layout=ipw.Layout(width="100px"),
        )
        self.list_button = ipw.Button(
            description="List folder",
            icon="folder-open",
            tooltip="list the folders of the highlighted scans in the scan list",
            layout=ipw.Layout(width="120px"),
        )
        self.remove_button.on_click(lambda _b: self._remove())
        self.clear_button.on_click(lambda _b: picker.clear())
        self.list_button.on_click(lambda _b: self._list_folders())
        self.buttons = ipw.HBox([self.remove_button, self.clear_button, self.list_button])
        self.box = ipw.VBox(
            [ipw.HTML(CSS), self.title, self.scans, self.buttons],
            layout=ipw.Layout(border="1px solid #90caf9", padding="4px 8px", margin="6px 0"),
        )
        self.refresh()

    def refresh(self) -> None:
        """Show the picker's selection again (nothing highlighted)."""
        m = self.picker
        keys = m.selection
        self.scans.options = list(zip(m.selection_lines(), keys, strict=True))
        self.scans.rows = min(10, max(3, len(keys)))
        self.scans.layout.display = None if keys else "none"
        self.buttons.layout.display = None if keys else "none"
        if keys:
            self.title.value = f"<b>selection</b>&nbsp; {m.selection_text()} &nbsp;{HELP}"
        else:
            self.title.value = f"<b>selection</b>&nbsp; {EMPTY}"

    def _remove(self) -> None:
        if self.scans.value:
            self.picker.remove(list(self.scans.value))

    def _list_folders(self) -> None:
        folders = list(dict.fromkeys(folder for folder, _ in self.scans.value))
        if not folders:
            return
        try:
            self.picker.set_folders(folders)
        except ValueError as exc:  # a folder that no longer opens: say so here
            self.title.value = f"<b>selection</b>&nbsp; <span style='color:#b71c1c'>{exc}</span>"
