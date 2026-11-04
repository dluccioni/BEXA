"""The Browse tab: walk folders and look inside files before opening anything.

Folders are listed with their sizes and what bexa recognises in them; a file
shows its structure: an HDF5/NeXus tree with shapes, chunks, compression and
attributes, a netCDF file's dims and variables, a SPEC session's scans and
columns, a TIFF's tags, an array's shape, a text file's first lines. Any
dataset or variable, or a slice of it, loads as an object; anything bexa reads
as a scan opens from here, and **Scan metadata** shows everything its files say
(:func:`bexa.core.metadata.scan_metadata`): the dims, every channel recorded at
each point, the positions, the detectors, the files and the engine's own tables.
The objects list shows the same for a scan object, with its **Names and offsets**
(:class:`bexa.gui.names_editor.NamesEditor`): an alias, an offset and a scale for each
motor, counter, position and detector. Refresh looks again and keeps what is picked:
the folder's entries, the file picked read anew (a SPEC file being written: its new
scans and points, on the scans selected) and the opened scans' new points. Several scans
of a SPEC session open at once, as objects or as a scan set.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import numpy as np

from bexa.gui.figures import FigurePane
from bexa.gui.inspectors import H5Inspector, SpecInspector, array_preview, tiff_summary
from bexa.gui.metadata_view import MetadataView
from bexa.gui.names_editor import NamesEditor
from bexa.gui.netcdf_inspector import NetCDFInspector
from bexa.io.netcdf import NC_SUFFIXES

__all__ = ["BrowseTab"]

H5_SUFFIXES = (".h5", ".hdf5", ".nxs", ".nx", ".cxi", ".hdf")
TEXT_SUFFIXES = (
    ".txt",
    ".csv",
    ".tsv",
    ".dat",
    ".log",
    ".yaml",
    ".yml",
    ".json",
    ".md",
    ".py",
    ".cfg",
)
MAX_ENTRIES = 2000


def _size(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        return "?"
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if size < 1000 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1000
    return ""


def _is_spec(path: Path) -> bool:
    if path.suffix.lower() == ".spec":
        return True
    if path.suffix:
        return False
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"#F"
    except OSError:
        return False


class BrowseTab:
    """Folder navigation on the left, the selected file's inside on the right."""

    def __init__(self, app: Any, root: str | Path | None = None) -> None:
        import ipywidgets as ipw

        self.app = app
        self.folder = Path(root) if root else Path.cwd()
        self.inspector: Any = None
        self.path = ipw.Text(value=str(self.folder), layout=ipw.Layout(width="70%"))
        go = ipw.Button(description="Go", layout=ipw.Layout(width="50px"))
        choose = ipw.Button(
            icon="ellipsis-h",
            tooltip="choose a file",
            layout=ipw.Layout(width="40px", display="" if app.dialogs else "none"),
        )
        choose.on_click(lambda _b: app.choose_path("file", self.go, self.folder))
        up = ipw.Button(
            icon="arrow-up", tooltip="the parent folder", layout=ipw.Layout(width="40px")
        )
        refresh = ipw.Button(
            icon="refresh",
            tooltip="look again, what is picked kept: the folder, the file (a SPEC file being "
            "written: its new scans and points) and the opened scans' new points",
            layout=ipw.Layout(width="40px"),
        )
        enter = ipw.Button(
            description="Open folder", icon="folder-open", layout=ipw.Layout(width="120px")
        )
        use = ipw.Button(
            description="Use this folder",
            icon="check",
            tooltip="work in this folder (Data tab), scans or not",
            layout=ipw.Layout(width="140px"),
        )
        go.on_click(lambda _b: self.go(self.path.value))
        up.on_click(lambda _b: self.go(self.folder.parent))
        refresh.on_click(lambda _b: self.refresh())
        enter.on_click(lambda _b: self._enter())
        use.on_click(lambda _b: self._use())
        self.listing = ipw.Select(options=[], rows=8, layout=ipw.Layout(width="100%"))
        self.listing.add_class("bexa-files")
        self._relisting = False  # the listing made again: what is picked is shown after
        self.listing.observe(lambda c: self._picked(c["new"]), names="value")
        self.head = ipw.HTML()
        self.pane = FigurePane((5.4, 3.6), interactive=False)
        self.figure_box = ipw.Box([self.pane.widget], layout=ipw.Layout(display="none"))
        self.text = ipw.HTML()
        self.body = ipw.VBox()
        self.metadata = MetadataView()
        self.names = NamesEditor(app)
        self.names_fold = ipw.Accordion(children=[self.names.box], selected_index=None)
        self.names_fold.set_title(0, "Names and offsets")
        self.metadata_box = ipw.VBox()  # the metadata view, when a scan's is shown
        self._shown = 0  # counts what was shown: a job's late answer for an older one waits
        self.box = ipw.VBox(  # one column: the folder, then what is inside the entry picked
            [
                ipw.HBox([up, self.path, go, choose, refresh]),
                ipw.HBox([enter, use]),
                self.listing,
                self.head,
                self.body,
                self.figure_box,
                self.text,
                self.metadata_box,
            ]
        )
        self.box.add_class("bexa-browse")
        self.path.add_class("bexa-browse-path")
        self.go(self.folder)

    # ------------------------------------------------------------------ navigation
    def go(self, folder: str | Path) -> None:
        path = Path(str(folder).strip().strip('"'))
        if path.is_file():
            self.go(path.parent)
            self.listing.value = str(path)
            return
        if not path.is_dir():
            self.app.say(f"no folder {path}", "error")
            return
        self.folder = path
        self.path.value = str(path)
        try:
            entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError as exc:
            self.app.say(str(exc), "error")
            return
        options = [(".. (parent)", str(path.parent))]
        for entry in entries[:MAX_ENTRIES]:
            if entry.name.startswith("."):
                continue
            label = (
                f"[dir]  {entry.name}/"
                if entry.is_dir()
                else f"       {entry.name}   {_size(entry)}"
            )
            options.append((label, str(entry)))
        if len(entries) > MAX_ENTRIES:
            options.append((f"... {len(entries) - MAX_ENTRIES} more", ""))
        self.listing.options = options
        self.head.value = f"<b>{html.escape(str(path))}</b>: {len(entries)} entries"

    def refresh(self) -> None:
        """Look again, keeping what is picked: the folder's entries, the file picked read
        anew (a SPEC file being written: its new scans and points, the scans selected, the
        columns and the search kept), and the opened scans being written their new points."""
        chosen = self.listing.value
        keep = self.inspector.state() if isinstance(self.inspector, SpecInspector) else None
        self._relisting = True
        try:
            self.go(self.folder)
            listed = {value for _, value in self.listing.options}
            self.listing.value = chosen if chosen in listed else None
        finally:
            self._relisting = False
        if self.listing.value:
            self._picked(self.listing.value, keep)

        def said(grown: list[str]) -> None:
            if grown:
                self.app.say(f"read again: {', '.join(grown)}", "ok")

        reading = "reading the opened scans again"
        self.app.submit(reading, self.app.ws.refresh, done=said, quiet=True)

    def _enter(self) -> None:
        chosen = self.listing.value
        if chosen and Path(chosen).is_dir():
            self.go(chosen)

    def _use(self) -> None:
        """Work in the folder shown: the Data tab's folder, scans or not."""
        self.app.data.use_folder(self.folder)
        self.app.show_tab("Data")

    # ------------------------------------------------------------------ inspecting
    def show_figure(self) -> None:
        self.figure_box.layout.display = ""

    def show_text(self, text: str) -> None:
        self.text.value = (
            f"<pre style='font-size:11px;max-height:360px;overflow:auto;"
            f"white-space:pre-wrap'>{html.escape(text)}</pre>"
        )

    def show_html(self, markup: str) -> None:
        self.text.value = (
            f"<div style='max-height:360px;overflow:auto;font-size:11px'>{markup}</div>"
        )

    def _reset(self) -> None:
        if self.inspector is not None:
            self.inspector.close()
        self.inspector = None
        self.body.children = []
        self.text.value = ""
        self.figure_box.layout.display = "none"
        self.metadata_box.children = []
        self._shown += 1

    def _picked(self, chosen: str | None, keep: dict[str, Any] | None = None) -> None:
        """Show the entry picked; ``keep``: what was picked in it before (a refresh)."""
        if not chosen or self._relisting:
            return
        path = Path(chosen)
        self._reset()
        try:
            if path.is_dir():
                self._folder(path)
            elif path.suffix.lower() in H5_SUFFIXES:
                self.inspector = H5Inspector(self, path)
                self.body.children = [self._open_bar(path), self.inspector.box]
            elif _is_spec(path):
                self.inspector = SpecInspector(self, path, keep)
                self.body.children = [self._open_bar(path), self.inspector.box]
            elif path.suffix.lower() in NC_SUFFIXES:
                bar = self._open_bar(path)
                self.inspector = NetCDFInspector(self, path)
                self.body.children = [bar, self.inspector.box]
                if bar.children:  # a run bexa reads: its metadata at once (a few kB)
                    self.scan_metadata(path)
            elif path.suffix.lower() in (".tif", ".tiff"):
                self._tiff(path)
            elif path.suffix.lower() == ".npy":
                self._npy(path)
            elif path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                self._picture(path)
            elif path.suffix.lower() in TEXT_SUFFIXES or path.stat().st_size < 200_000:
                self._text(path)
            else:
                self.head.value = f"<b>{html.escape(path.name)}</b> ({_size(path)}): no preview"
        except Exception as exc:
            self.show_text(f"cannot look inside {path.name}: {type(exc).__name__}: {exc}")

    def _open_bar(self, path: Path) -> Any:
        """What bexa makes of the file, and buttons to open it as a scan or load it."""
        import ipywidgets as ipw

        from bexa.io.formats import match_spec

        scored = match_spec(path)
        spec = scored[0][1] if scored and scored[0][0] >= 1.0 else None
        note = f"{spec.name} ({spec.engine})" if spec is not None else "no scan layout"
        self.head.value = f"<b>{html.escape(path.name)}</b> ({_size(path)}): {note}"
        to_data = ipw.Button(
            description="Open with the Data tab",
            icon="share",
            tooltip="put the path in the Data tab",
        )
        to_data.on_click(lambda _b: self.app.data_path(path))
        metadata = ipw.Button(
            description="Scan metadata",
            icon="list",
            tooltip="everything the files say of the scan (of the scan picked, in a session)",
        )
        metadata.on_click(lambda _b: self.scan_metadata(path))
        return ipw.HBox([to_data, metadata]) if spec is not None else ipw.HBox()

    # ------------------------------------------------------------------ metadata
    def _scan_number(self) -> int | None:
        """The scan picked in the inspector shown: a SPEC session's (the one shown), or the
        scan group of an HDF5 file's tree (``/4.1/...``); ``None`` for a file of one scan."""
        if isinstance(self.inspector, SpecInspector):
            return self.inspector.shown
        if isinstance(self.inspector, H5Inspector) and self.inspector.selected:
            head = self.inspector.selected.strip("/").split("/")[0].split(".")[0]
            return int(head) if head.isdigit() else None
        return None

    def scan_metadata(self, path: str | Path) -> None:
        """Everything the files say of the scan behind ``path``, below the file's inside
        (the scan picked, for a file of several). Read in the background, no frame read."""
        import bexa

        path = Path(path)
        number = self._scan_number()
        title = path.name + (f", scan {number}" if number is not None else "")
        self._metadata_of(title, lambda: bexa.open(path, scan=number, cache=False).metadata())

    def show_scan_metadata(self, name: str, scan: Any, names: bool = False) -> None:
        """The metadata of the scan object ``name`` (the objects list's **Full metadata**),
        under its **Names and offsets**, unfolded with ``names``."""
        self._reset()
        self.head.value = f"<b>{html.escape(name)}</b>: a scan object"
        self.names.show(name)
        self.names_fold.selected_index = 0 if names else None
        self.body.children = [self.names_fold]
        self._metadata_of(name if scan.name == name else f"{name} ({scan.name})", scan.metadata)

    def _metadata_of(self, title: str, read: Any) -> None:
        shown = self._shown
        self.metadata.wait(title)
        self.metadata_box.children = [self.metadata.box]

        def done(tables: Any) -> None:
            if shown == self._shown:  # still what the tab shows
                self.metadata.show(tables, title)

        self.app.submit(f"reading the metadata of {title}", read, done=done, quiet=True)

    def _folder(self, path: Path) -> None:
        import ipywidgets as ipw

        from bexa.io.formats import match_spec

        try:
            entries = list(path.iterdir())
        except OSError as exc:
            self.show_text(str(exc))
            return
        scored = match_spec(path)
        spec = scored[0][1] if scored and scored[0][0] >= 1.0 else None
        files = sum(1 for e in entries if e.is_file())
        note = f"{spec.name} ({spec.engine})" if spec else "no scans"
        self.head.value = (
            f"<b>{html.escape(path.name)}/</b>: {len(entries) - files} folders, "
            f"{files} files. {note}"
        )
        enter = ipw.Button(description="Open folder", icon="folder-open")
        enter.on_click(lambda _b: self.go(path))
        buttons = [enter]
        if spec is not None:
            to_data = ipw.Button(description="Open with the Data tab", icon="share")
            to_data.on_click(lambda _b: self.app.data_path(path))
            buttons.append(to_data)
        self.body.children = [ipw.HBox(buttons)]

    def _tiff(self, path: Path) -> None:
        import ipywidgets as ipw

        head, image, pages = tiff_summary(path)
        self.head.value = head
        load = ipw.Button(description="Load as object", button_style="primary")
        load.on_click(
            lambda _b: self.app.submit(
                f"loading {path.name}",
                lambda: self.app.ws.open_file(str(path)),
                done=self.app.show_results,
            )
        )
        children: list[Any] = [load, self._open_bar_compact(path)]
        if pages > 1:
            page = ipw.IntSlider(min=0, max=pages - 1, description="page", continuous_update=False)
            page.observe(lambda c: self._tiff_page(path, c["new"]), names="value")
            children.insert(0, page)
        self.body.children = children
        self.show_figure()
        array_preview(self.pane, image, path.name)

    def _tiff_page(self, path: Path, page: int) -> None:
        _, image, _ = tiff_summary(path, page)
        array_preview(self.pane, image, f"{path.name} page {page}")

    def _open_bar_compact(self, path: Path) -> Any:
        import ipywidgets as ipw

        from bexa.io.engines.spec_tiff import frame_point
        from bexa.io.formats import match_spec

        scored = match_spec(path)
        spec = scored[0][1] if scored and scored[0][0] >= 1.0 else None
        # beside a SPEC file only a frame named <file>_S<scan>_<point>.tif belongs to a scan
        if spec is not None and (spec.engine != "spec_tiff" or frame_point(path) >= 0):
            button = ipw.Button(description=f"Open its scan ({spec.name})", icon="share")
            button.on_click(lambda _b: self.app.data_path(path))
            return button
        return ipw.HTML("")

    def _npy(self, path: Path) -> None:
        import ipywidgets as ipw

        values = np.load(path, mmap_mode="r")
        self.head.value = (
            f"<b>{html.escape(path.name)}</b>: {values.shape} {values.dtype}, "
            f"{values.nbytes / 1e6:.1f} MB"
        )
        index = ipw.Text(
            description="index", placeholder="0, :, :", style={"description_width": "50px"}
        )
        load = ipw.Button(description="Load as object", button_style="primary")
        load.on_click(
            lambda _b: self.app.submit(
                f"loading {path.name}",
                lambda: self.app.ws.open_file(str(path), index=index.value or None),
                done=self.app.show_results,
            )
        )
        self.body.children = [ipw.HBox([index, load])]
        self.show_figure()
        preview = values[(0,) * (values.ndim - 2)] if values.ndim > 2 else values
        array_preview(self.pane, np.asarray(preview), path.name)

    def _picture(self, path: Path) -> None:
        import ipywidgets as ipw

        self.head.value = f"<b>{html.escape(path.name)}</b> ({_size(path)})"
        image = ipw.Image(
            value=path.read_bytes(),
            format=path.suffix.lstrip(".").lower(),
            layout=ipw.Layout(max_width="100%"),
        )
        load = ipw.Button(description="Load as object", button_style="primary")
        load.on_click(
            lambda _b: self.app.submit(
                f"loading {path.name}",
                lambda: self.app.ws.open_file(str(path)),
                done=self.app.show_results,
            )
        )
        self.body.children = [load, image]

    def _text(self, path: Path) -> None:
        import ipywidgets as ipw

        with open(path, encoding="utf-8", errors="replace") as f:
            lines = [next(f, "") for _ in range(300)]
        self.head.value = f"<b>{html.escape(path.name)}</b> ({_size(path)}), the first lines:"
        if path.suffix.lower() in (".csv", ".tsv", ".dat", ".txt"):
            load = ipw.Button(description="Load as table", button_style="primary")
            load.on_click(
                lambda _b: self.app.submit(
                    f"loading {path.name}",
                    lambda: self.app.ws.open_file(str(path)),
                    done=self.app.show_results,
                )
            )
            self.body.children = [load]
        self.show_text("".join(lines))
