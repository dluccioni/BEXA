"""The Data tab: the folder to work in, and opening data from any facility bexa reads.

- the folder: any existing folder, scans or not (a beamtime folder before its first scan);
  its datasets show in the scan picker once there are any, running scans included, and its
  selection is added as scans, a scan set or a series. Refresh looks again and reads the
  new points of the opened scans still being written; auto refreshes every 10 s;
- a path (folder, master file, SPEC session, run, package, cube, frame file): sniffed
  against every format spec, opened as one scan, as a scan set, or loaded as objects;
- a beamtime profile and a sample name;
- synthetic demo data to try the tools.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from bexa.workspace.catalog import parse_ints

__all__ = ["DataTab"]

DEMO_KINDS = ("mosa", "rocking", "energy", "zstack", "alignment")
NO_SCANS = "no scans yet: Refresh when they are written"
LIVE = ("complete", "aborted", "running")  # running scans read as far as they are written
AUTO_SECONDS = 10.0


def _section(title: str) -> Any:
    import ipywidgets as ipw

    return ipw.HTML(f"<h4 style='margin:10px 0 4px 0'>{html.escape(title)}</h4>")


def _no_format(exc: Exception) -> bool:
    return isinstance(exc, (LookupError, ValueError)) and "no format spec matches" in str(exc)


class DataTab:
    """The folder to work in; open scans, scan sets and files."""

    def __init__(self, app: Any, root: str | Path | None = None) -> None:
        import ipywidgets as ipw

        self.app = app
        self.picker: Any = None
        self.datasets: list[Path] = []  # the dataset folders the picker was made with
        self._timer: Any = None  # the next automatic refresh
        wide = ipw.Layout(width="70%")
        label = {"description_width": "60px"}
        # the folder
        self.root = ipw.Text(
            value=str(root or ""),
            description="folder",
            placeholder="a data folder",
            layout=ipw.Layout(width="60%"),
            style=label,
        )
        use = ipw.Button(description="Use folder", button_style="primary", icon="folder-open")
        refresh = ipw.Button(
            icon="refresh",
            tooltip="look again: new scans, new points of the scans being written",
            layout=ipw.Layout(width="40px"),
        )
        self.auto = ipw.ToggleButton(
            description="auto",
            tooltip=f"refresh every {AUTO_SECONDS:.0f} s",
            layout=ipw.Layout(width="70px"),
        )
        use.on_click(lambda _b: self.use_folder(self.root.value))
        refresh.on_click(lambda _b: self.refresh())
        self.auto.observe(lambda change: self._auto_changed(change["new"]), names="value")
        self.folder_note = ipw.HTML()
        self.picker_box = ipw.VBox()
        add_scans = ipw.Button(description="Add as scans", button_style="primary")
        add_set = ipw.Button(description="Add as a scan set", button_style="info")
        add_series = ipw.Button(
            description="Add as one series",
            tooltip="scans that share a grid joined on a new dim (energy, a positioner)",
        )
        add_scans.on_click(lambda _b: self._from_picker("scans"))
        add_set.on_click(lambda _b: self._from_picker("set"))
        add_series.on_click(lambda _b: self._from_picker("series"))
        self.picker_buttons = ipw.HBox(
            [add_scans, add_set, add_series], layout=ipw.Layout(display="none")
        )
        # a path
        self.path = ipw.Text(
            value=str(root or ""),
            description="path",
            placeholder="a folder or file",
            layout=wide,
            style=label,
        )
        self.scan_numbers = ipw.Text(
            description="scans",
            placeholder="all, or 7, 1-5",
            layout=ipw.Layout(width="320px"),
            style=label,
        )
        self.format = ipw.Text(
            description="format", placeholder="auto", layout=ipw.Layout(width="320px"), style=label
        )
        sniff = ipw.Button(description="What is it?", icon="search")
        open_scan = ipw.Button(
            description="Open scan(s)", button_style="primary", tooltip="one object per scan"
        )
        open_set = ipw.Button(
            description="Open as scan set",
            button_style="info",
            tooltip="the scans as one object, to process together",
        )
        load = ipw.Button(description="Load file", tooltip="arrays, cubes, TIFF, CSV")
        sniff.on_click(lambda _b: self._sniff())
        open_scan.on_click(lambda _b: self._open(as_set=False))
        open_set.on_click(lambda _b: self._open(as_set=True))
        load.on_click(lambda _b: self._load())
        self.sniffed = ipw.HTML()
        # a profile, demo data
        self.profile = ipw.Text(
            description="profile",
            placeholder="name or YAML",
            layout=ipw.Layout(width="320px"),
            style=label,
        )
        self.sample = ipw.Text(description="sample", layout=ipw.Layout(width="240px"), style=label)
        self.profile_scan = ipw.Text(
            description="scan",
            layout=ipw.Layout(width="160px"),
            style={"description_width": "40px"},
        )
        open_profile = ipw.Button(description="Open", button_style="primary")
        open_profile.on_click(lambda _b: self._open_profile())
        self.demo = ipw.Dropdown(
            options=DEMO_KINDS,
            value="mosa",
            description="demo",
            layout=ipw.Layout(width="220px"),
            style=label,
        )
        make_demo = ipw.Button(description="Make demo data", icon="flask")
        make_demo.on_click(lambda _b: self._demo())
        self.box = ipw.VBox(
            [
                _section("Folder"),
                ipw.HBox([self.root, use, refresh, self.auto]),
                self.folder_note,
                self.picker_buttons,
                self.picker_box,
                _section("Open a path"),
                self.path,
                ipw.HBox([self.scan_numbers, self.format]),
                ipw.HBox([sniff, open_scan, open_set, load]),
                self.sniffed,
                _section("Beamtime profile"),
                ipw.HBox([self.profile, self.sample, self.profile_scan, open_profile]),
                _section("Demo data"),
                ipw.HBox([self.demo, make_demo]),
            ]
        )

    # ------------------------------------------------------------------ the folder
    def use_folder(self, path: str | Path) -> None:
        """Work in this folder, whatever it holds yet: the scan picker lists its datasets (or
        says there are none: Refresh later), the Browse tab and the path field start there."""
        folder = Path(str(path).strip().strip('"'))
        if not str(path).strip() or not folder.is_dir():
            self.app.say(f"no folder {folder}", "error")
            return
        self.app.folder = folder
        self.root.value = str(folder)
        self.path.value = str(folder)
        self.app.browse.go(folder)
        self.refresh()

    def refresh(self, quiet: bool = False) -> None:
        """Look again: the opened scans still being written read their new points
        (:meth:`~bexa.workspace.Workspace.refresh`), the picker lists the folder's datasets
        and their new scans. ``quiet``: a message only when a scan grew (auto)."""
        folder = self.app.folder
        picker, known = self.picker, list(self.datasets)

        def look() -> tuple[list[str], list[Path] | None, Any]:
            grown = self.app.ws.refresh()
            if folder is None:
                return grown, None, None
            from bexa.notebook.listing import list_datasets

            found = list_datasets(folder)
            if not found:
                return grown, found, None
            if picker is not None and found == known:
                picker.refresh()  # here, not in the main thread: it reads the folders again
                return grown, found, picker
            from bexa.notebook.picker import ScanPicker

            made = ScanPicker(root=folder, interactive=True, cache=self.app.ws.cache, status=LIVE)
            return grown, found, made

        def shown(result: tuple[list[str], list[Path] | None, Any]) -> None:
            grown, found, made = result
            if found is not None:
                self._list(found, made)
            if grown:
                self.app.say(f"read again: {', '.join(grown)}", "ok")
            elif not quiet:
                self.app.say("")

        where = folder.name if folder is not None else "the opened scans"
        self.app.submit(f"looking in {where}", look, done=shown, quiet=quiet)

    def _list(self, found: list[Path], picker: Any) -> None:
        """Show the folder's datasets in ``picker``, or say there are none yet."""
        self.datasets = found
        self.picker = picker
        self.picker_box.children = [picker.widgets.box] if picker is not None else []
        self.picker_buttons.layout.display = "" if picker is not None else "none"
        count = f"{len(found)} dataset{'s' if len(found) != 1 else ''}"
        self.folder_note.value = (
            f"<span style='color:#757575'>{count if found else NO_SCANS}</span>"
        )

    # ------------------------------------------------------------------ auto refresh
    def _auto_changed(self, on: bool) -> None:
        self.stop_auto()
        if on:
            self._timer = self.app.later(AUTO_SECONDS, self._tick)
            if self._timer is None:
                self.app.say("auto refresh needs the GUI in a running kernel", "error")
                self.auto.value = False

    def _tick(self) -> None:
        """One automatic refresh (skipped while another job runs), then the next one."""
        self._timer = None
        if not self.auto.value:
            return
        try:
            if not self.app.jobs.busy:
                self.refresh(quiet=True)
        finally:
            self._timer = self.app.later(AUTO_SECONDS, self._tick)

    def stop_auto(self) -> None:
        """Stop the automatic refresh (the toggle stays as it is)."""
        self.app.cancel_later(self._timer)
        self._timer = None

    # ------------------------------------------------------------------ a path
    def _path(self) -> str:
        path = self.path.value.strip().strip('"')
        if not path:
            raise ValueError("type a path first")
        return path

    def _sniff(self) -> None:
        def look() -> str:
            from bexa.io.formats import match_spec

            path = Path(self._path())
            if not path.exists():
                raise FileNotFoundError(path)
            scored = match_spec(path)
            head = f"<b>{html.escape(path.name)}</b>: "
            if scored and scored[0][0] >= 1.0:
                spec = scored[0][1]
                return head + f"{spec.name} ({spec.engine})"
            if path.is_dir():
                return head + "no scans yet (Use folder to work in it anyway)"
            return head + "no scan layout: load it as an object, or look inside on Browse"

        self.app.guarded(look, show=lambda text: setattr(self.sniffed, "value", text))

    def _open(self, as_set: bool) -> None:
        path = self._path()
        numbers = parse_ints(self.scan_numbers.value, "scans")
        fmt = self.format.value.strip() or None
        options = {"format": fmt} if fmt else {}

        def work() -> list[Any] | Path:
            try:
                return self._open_now(path, numbers, as_set, fmt, options)
            except Exception as exc:  # a folder without scans yet becomes the folder to use
                if Path(path).is_dir() and _no_format(exc):
                    return Path(path)
                raise

        def done(made: list[Any] | Path) -> None:
            if isinstance(made, Path):
                self.use_folder(made)
                return
            self.app.show_results(made)

        self.app.submit(f"opening {Path(path).name}", work, done=done)

    def _open_now(
        self,
        path: str,
        numbers: list[int] | None,
        as_set: bool,
        fmt: str | None,
        options: dict[str, Any],
    ) -> list[Any]:
        ws = self.app.ws
        if as_set or (numbers is not None and len(numbers) > 1):
            # scans named by number open whatever their status (a SPEC scan whose frames
            # were not copied still has its counters); all scans: those with data
            status: Any = "any" if numbers is not None else LIVE
            return ws.open_scans(path, numbers, as_set=as_set, status=status, **options)
        if numbers is not None:
            return [ws.open_scan(path, scan=numbers[0], **options)]
        try:  # a dataset of several scans gives one object per scan; anything else, one scan
            from bexa.core.scan import list_scans

            several = len(list_scans(path, format=fmt)) > 1
        except (LookupError, ValueError, OSError, TypeError):
            several = False
        if several:
            return ws.open_scans(path, None, as_set=False, status=LIVE, **options)
        return [ws.open_scan(path, **options)]

    def _load(self) -> None:
        path = self._path()
        self.app.submit(
            f"loading {Path(path).name}",
            lambda: self.app.ws.open_file(path),
            done=self.app.show_results,
        )

    # ------------------------------------------------------------------ the picker
    def _from_picker(self, how: str) -> None:
        picker = self.picker
        if picker is None:
            self.app.say("no scans listed: Use folder first", "error")
            return
        ws = self.app.ws

        def work() -> list[Any]:
            scans = picker.scans
            if not scans:
                raise ValueError("no scan is selected in the picker")
            if how == "scans":
                return ws.add_scans(scans)
            made = ws.add_scans(scans, as_set=True, name="selection")
            if how == "series":
                return ws.run("series", made[0], dim=picker.stack_axis or "auto")
            return made

        self.app.submit("adding the selected scans", work, done=self.app.show_results)

    # ------------------------------------------------------------------ profile, demo
    def _open_profile(self) -> None:
        profile = self.profile.value.strip()
        sample = self.sample.value.strip() or None
        numbers = parse_ints(self.profile_scan.value, "scan")
        if not profile:
            self.app.say("type a profile name or path", "error")
            return
        scan = numbers[0] if numbers and len(numbers) == 1 else numbers
        self.app.submit(
            f"opening {sample or profile}",
            lambda: [self.app.ws.open_scan(None, scan=scan, profile=profile, sample=sample)],
            done=self.app.show_results,
        )

    def _demo(self) -> None:
        kind = self.demo.value
        self.app.submit(
            f"making demo data ({kind})", lambda: self.app.ws.demo(kind), done=self.app.show_results
        )
