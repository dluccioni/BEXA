"""The Data tab: open data from any facility bexa reads.

- a path (folder, master file, SPEC session, run, package, cube, frame file):
  sniffed against every format spec, opened as one scan, as a scan set, or
  loaded as objects when it is not a scan;
- a beamtime tree in the scan picker (datasets, scan list, motor buttons, the
  selection across folders), its selection added as scans, a scan set or a series;
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


def _section(title: str, note: str = "") -> Any:
    import ipywidgets as ipw

    extra = (
        f" <span style='color:#757575;font-size:11px'>{html.escape(note)}</span>" if note else ""
    )
    return ipw.HTML(f"<h4 style='margin:10px 0 4px 0'>{html.escape(title)}{extra}</h4>")


class DataTab:
    """Open scans, scan sets and files."""

    def __init__(self, app: Any, root: str | Path | None = None) -> None:
        import ipywidgets as ipw

        self.app = app
        self.picker: Any = None
        wide = ipw.Layout(width="70%")
        self.path = ipw.Text(
            value=str(root or ""),
            description="path",
            placeholder="a folder or file",
            layout=wide,
            style={"description_width": "60px"},
        )
        self.scan_numbers = ipw.Text(
            description="scans",
            placeholder="empty: all, or 7, or 1-5, 9",
            layout=ipw.Layout(width="320px"),
            style={"description_width": "60px"},
        )
        self.format = ipw.Text(
            description="format",
            placeholder="sniffed; or a spec name",
            layout=ipw.Layout(width="320px"),
            style={"description_width": "60px"},
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
        load = ipw.Button(description="Load file as object", tooltip="arrays, cubes, TIFF, CSV")
        sniff.on_click(lambda _b: self._sniff())
        open_scan.on_click(lambda _b: self._open(as_set=False))
        open_set.on_click(lambda _b: self._open(as_set=True))
        load.on_click(lambda _b: self._load())
        self.sniffed = ipw.HTML()
        self.root = ipw.Text(
            value=str(root or ""),
            description="root",
            layout=wide,
            placeholder="a RAW_DATA folder (or any folder of datasets)",
            style={"description_width": "60px"},
        )
        show_picker = ipw.Button(description="Show the scan picker", icon="list")
        show_picker.on_click(lambda _b: self._show_picker())
        self.picker_box = ipw.VBox()
        add_scans = ipw.Button(description="Add selected as scans", button_style="primary")
        add_set = ipw.Button(description="Add selected as a scan set", button_style="info")
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
        self.profile = ipw.Text(
            description="profile",
            placeholder="name or YAML path",
            layout=ipw.Layout(width="320px"),
            style={"description_width": "60px"},
        )
        self.sample = ipw.Text(
            description="sample",
            layout=ipw.Layout(width="240px"),
            style={"description_width": "60px"},
        )
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
            style={"description_width": "60px"},
        )
        make_demo = ipw.Button(description="Make demo data", icon="flask")
        make_demo.on_click(lambda _b: self._demo())
        self.box = ipw.VBox(
            [
                _section(
                    "Open a path",
                    "ESRF BLISS, APS SPEC + TIFF, PAL, LCLS smalldata, "
                    "EuXFEL, packages, cubes: the layout is sniffed",
                ),
                self.path,
                ipw.HBox([self.scan_numbers, self.format]),
                ipw.HBox([sniff, open_scan, open_set, load]),
                self.sniffed,
                _section(
                    "Browse a beamtime",
                    "pick scans in any dataset folders; the selection "
                    "stays when another folder is listed",
                ),
                ipw.HBox([self.root, show_picker]),
                self.picker_buttons,
                self.picker_box,
                _section("Beamtime profile", "a profile names the data root, format and samples"),
                ipw.HBox([self.profile, self.sample, self.profile_scan, open_profile]),
                _section("Demo data", "synthetic scans with a known answer, to try every tool"),
                ipw.HBox([self.demo, make_demo]),
            ]
        )

    # ------------------------------------------------------------------ actions
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
            lines = [f"<b>{html.escape(str(path))}</b>"]
            if scored and scored[0][0] >= 1.0:
                spec = scored[0][1]
                lines.append(
                    f"bexa reads it as <b>{spec.name}</b> (engine {spec.engine}): "
                    f"{html.escape(spec.description or '')}"
                )
            elif scored:
                close = ", ".join(f"{s.name} {sc:.0%}" for sc, s in scored[:3])
                lines.append(
                    f"no format matches fully; closest: {html.escape(close)}. "
                    "Load it as an object, or look inside on the Browse tab."
                )
            else:
                lines.append("no format spec matches; look inside on the Browse tab")
            return "<br>".join(lines)

        self.app.guarded(look, show=lambda text: setattr(self.sniffed, "value", text))

    def _open(self, as_set: bool) -> None:
        path = self._path()
        numbers = parse_ints(self.scan_numbers.value, "scans")
        fmt = self.format.value.strip() or None
        options = {"format": fmt} if fmt else {}
        ws = self.app.ws

        def work() -> list[Any]:
            if as_set or (numbers is not None and len(numbers) > 1):
                return ws.open_scans(path, numbers, as_set=as_set, **options)
            if numbers is not None:
                return [ws.open_scan(path, scan=numbers[0], **options)]
            try:  # a dataset of several scans gives one object per scan; anything else, one scan
                from bexa.core.scan import list_scans

                several = len(list_scans(path, format=fmt)) > 1
            except (LookupError, ValueError, OSError, TypeError):
                several = False
            if several:
                return ws.open_scans(path, None, as_set=False, **options)
            return [ws.open_scan(path, **options)]

        self.app.submit(f"opening {Path(path).name}", work, done=self.app.show_results)

    def _load(self) -> None:
        path = self._path()
        self.app.submit(
            f"loading {Path(path).name}",
            lambda: self.app.ws.open_file(path),
            done=self.app.show_results,
        )

    def _show_picker(self) -> None:
        def build() -> Any:
            from bexa.notebook.picker import ScanPicker

            root = self.root.value.strip().strip('"') or None
            if root is None:
                raise ValueError("type the root folder (RAW_DATA or any folder of datasets)")
            return ScanPicker(root=root, interactive=True, cache=self.app.ws.cache)

        def shown(picker: Any) -> None:
            self.picker = picker
            self.picker_box.children = [picker.widgets.box]
            self.picker_buttons.layout.display = ""

        self.app.submit("listing the datasets", build, done=shown)

    def _from_picker(self, how: str) -> None:
        picker = self.picker
        if picker is None:
            self.app.say("show the scan picker first", "error")
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
