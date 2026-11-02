"""The Convert tab: direct conversions between formats, and exporting objects.

- a scan or scan set to a bexa scan package, a cube, NeXus HDF5, TIFF or NPY;
- a whole dataset or SPEC session, scan by scan (packages are kept while current);
- one dataset of any HDF5 file (or a slice of it) to NPY, TIFF or CSV;
- a file bexa loads (npy, TIFF, cube, CSV) to another format;
- objects of the workspace to NeXus HDF5, CSV, NPY, TIFF, PNG, text or JSON.

Every conversion runs in the background and is logged like the operations.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from bexa.gui.forms import object_options
from bexa.workspace.catalog import parse_ints
from bexa.workspace.convert import DATASET_TARGETS, SCAN_TARGETS
from bexa.workspace.files import EXPORT_FORMATS

__all__ = ["ConvertTab"]

COMPRESSIONS = ("gzip", "bitshuffle", "lzf", "none")


def _section(title: str, note: str) -> Any:
    import ipywidgets as ipw

    return ipw.HTML(
        f"<h4 style='margin:12px 0 4px 0'>{html.escape(title)} "
        f"<span style='color:#757575;font-size:11px;font-weight:normal'>"
        f"{html.escape(note)}</span></h4>"
    )


class ConvertTab:
    """Conversions and exports."""

    def __init__(self, app: Any) -> None:
        import ipywidgets as ipw

        self.app = app
        style = {"description_width": "110px"}
        wide = ipw.Layout(width="70%")
        # a scan object
        self.scan = ipw.Dropdown(description="scan", style=style, layout=wide)
        self.scan_target = ipw.Dropdown(
            options=[(f"{k}: {v}", k) for k, v in SCAN_TARGETS.items()],
            value="package",
            description="into",
            style=style,
            layout=wide,
        )
        self.scan_out = ipw.Text(
            description="file or folder",
            style=style,
            layout=wide,
            placeholder="out/scan_7.h5 (a scan set: a folder)",
        )
        self.scan_roi = ipw.Dropdown(
            description="ROI", style=style, layout=ipw.Layout(width="300px")
        )
        self.scan_bin = ipw.Text(
            description="bin pixels",
            placeholder="empty: none",
            style=style,
            layout=ipw.Layout(width="250px"),
        )
        self.compression = ipw.Dropdown(
            options=COMPRESSIONS,
            value="gzip",
            description="compression",
            style=style,
            layout=ipw.Layout(width="250px"),
        )
        self.overwrite = ipw.Checkbox(value=False, description="Overwrite", indent=False)
        go_scan = ipw.Button(description="Convert", button_style="primary", icon="exchange")
        go_scan.on_click(lambda _b: self._convert_scan())
        # a session
        self.session = ipw.Text(
            description="dataset",
            placeholder="a dataset folder or SPEC file",
            style=style,
            layout=wide,
        )
        self.session_scans = ipw.Text(
            description="scans",
            placeholder="empty: all; 1-20, 25",
            style=style,
            layout=ipw.Layout(width="350px"),
        )
        self.session_target = ipw.Dropdown(
            options=list(SCAN_TARGETS),
            value="package",
            description="into",
            style=style,
            layout=ipw.Layout(width="300px"),
        )
        self.session_out = ipw.Text(description="folder", style=style, layout=wide)
        go_session = ipw.Button(
            description="Convert every scan", button_style="primary", icon="exchange"
        )
        go_session.on_click(lambda _b: self._convert_session())
        # an HDF5 dataset
        self.h5_file = ipw.Text(description="HDF5 file", style=style, layout=wide)
        self.h5_dataset = ipw.Text(
            description="dataset", placeholder="/entry/data/frames", style=style, layout=wide
        )
        self.h5_index = ipw.Text(
            description="slice",
            placeholder="empty: all; 0:100, :, :",
            style=style,
            layout=ipw.Layout(width="350px"),
        )
        self.h5_target = ipw.Dropdown(
            options=DATASET_TARGETS,
            value="npy",
            description="into",
            style=style,
            layout=ipw.Layout(width="250px"),
        )
        self.h5_out = ipw.Text(description="file or folder", style=style, layout=wide)
        go_h5 = ipw.Button(description="Convert dataset", button_style="primary", icon="exchange")
        go_h5.on_click(lambda _b: self._convert_dataset())
        # a file
        self.file_in = ipw.Text(
            description="file",
            placeholder="frames.npy, stack.tif, table.csv, cube.h5",
            style=style,
            layout=wide,
        )
        self.file_target = ipw.Dropdown(
            options=EXPORT_FORMATS,
            value="h5",
            description="into",
            style=style,
            layout=ipw.Layout(width="250px"),
        )
        self.file_out = ipw.Text(description="file", style=style, layout=wide)
        go_file = ipw.Button(description="Convert file", button_style="primary", icon="exchange")
        go_file.on_click(lambda _b: self._convert_file())
        # objects
        self.objects = ipw.SelectMultiple(description="objects", rows=6, style=style, layout=wide)
        self.export_format = ipw.Dropdown(
            options=EXPORT_FORMATS,
            value="h5",
            description="as",
            style=style,
            layout=ipw.Layout(width="250px"),
        )
        self.export_out = ipw.Text(
            description="file or folder",
            style=style,
            layout=wide,
            placeholder="one object: a file; several: a folder (h5: one file)",
        )
        go_export = ipw.Button(description="Export", button_style="primary", icon="save")
        go_export.on_click(lambda _b: self._export())
        self.result = ipw.HTML()
        self.box = ipw.VBox(
            [
                _section(
                    "Scan or scan set",
                    "frames are streamed: any size; a package keeps everything "
                    "and reopens as the same scan",
                ),
                self.scan,
                self.scan_target,
                self.scan_out,
                ipw.HBox([self.scan_roi, self.scan_bin]),
                ipw.HBox([self.compression, self.overwrite, go_scan]),
                _section(
                    "Whole dataset or session",
                    "one file per scan; packages already current are kept",
                ),
                self.session,
                ipw.HBox([self.session_scans, self.session_target]),
                self.session_out,
                go_session,
                _section("HDF5 dataset", "any file; pick the dataset on the Browse tab"),
                self.h5_file,
                self.h5_dataset,
                ipw.HBox([self.h5_index, self.h5_target]),
                self.h5_out,
                go_h5,
                _section("File to file", "anything bexa loads as objects"),
                self.file_in,
                self.file_target,
                self.file_out,
                go_file,
                _section(
                    "Export objects",
                    "NeXus HDF5 keeps the axes, uncertainties, metadata and history",
                ),
                self.objects,
                self.export_format,
                self.export_out,
                go_export,
                self.result,
            ]
        )
        self.refresh()

    def refresh(self) -> None:
        ws = self.app.ws
        self.scan.options = object_options(ws, ("scan", "scanset")) or [("no scan yet", None)]
        self.scan_roi.options = [("none", None)] + [(n, n) for n in ws.rois]
        listed = [
            (label, name)
            for label, name in object_options(
                ws, ("stack", "image", "map", "curve", "volume", "rgb", "table", "text")
            )
        ]
        chosen = tuple(n for n in self.app.selection if n in {v for _, v in listed})
        self.objects.options = listed
        self.objects.value = chosen

    def use_dataset(self, path: Path, dataset: str, index: str = "") -> None:
        """Fill the HDF5 section (the Browse tab's "Convert this dataset")."""
        self.h5_file.value = str(path)
        self.h5_dataset.value = dataset
        self.h5_index.value = index
        self.h5_out.value = str(Path(path).with_name(Path(dataset).name or "dataset"))
        self.app.show_tab("Convert")

    # ------------------------------------------------------------------ actions
    def _done(self, files: list[Path]) -> None:
        shown = "".join(f"<li><code>{html.escape(str(f))}</code></li>" for f in files[:20])
        more = f"<li>... {len(files) - 20} more</li>" if len(files) > 20 else ""
        self.result.value = f"<b>wrote {len(files)} file(s)</b><ul>{shown}{more}</ul>"
        self.app.say(f"wrote {len(files)} file(s)", "ok")

    @staticmethod
    def _needed(value: str, what: str) -> str:
        text = value.strip().strip('"')
        if not text:
            raise ValueError(f"{what}: needed")
        return text

    def _convert_scan(self) -> None:
        def work() -> list[Path]:
            name = self.scan.value
            if not name:
                raise ValueError("open a scan first (Data tab)")
            out = self._needed(self.scan_out.value, "file or folder")
            options: dict[str, Any] = {
                "compression": self.compression.value,
                "overwrite": self.overwrite.value,
            }
            if self.scan_roi.value:
                options["roi"] = self.scan_roi.value
            if self.scan_bin.value.strip():
                options["downsample"] = int(self.scan_bin.value)
            return self.app.ws.convert(name, self.scan_target.value, out, **options)

        self.app.submit("converting the scan", work, done=self._done)

    def _convert_session(self) -> None:
        def work() -> list[Path]:
            path = self._needed(self.session.value, "dataset")
            out = self._needed(self.session_out.value, "folder")
            scans = parse_ints(self.session_scans.value, "scans")
            return self.app.ws.convert_session(
                path,
                out,
                scans=scans,
                target=self.session_target.value,
                compression=self.compression.value,
                overwrite=self.overwrite.value,
            )

        self.app.submit("converting the session", work, done=self._done)

    def _convert_dataset(self) -> None:
        def work() -> list[Path]:
            path = self._needed(self.h5_file.value, "HDF5 file")
            dataset = self._needed(self.h5_dataset.value, "dataset")
            out = self._needed(self.h5_out.value, "file or folder")
            return self.app.ws.convert_dataset(
                path,
                dataset,
                self.h5_target.value,
                out,
                index=self.h5_index.value.strip() or None,
                overwrite=self.overwrite.value,
            )

        self.app.submit("converting the dataset", work, done=self._done)

    def _convert_file(self) -> None:
        def work() -> list[Path]:
            path = self._needed(self.file_in.value, "file")
            out = self._needed(self.file_out.value, "file")
            return self.app.ws.convert_file(path, self.file_target.value, out)

        self.app.submit("converting the file", work, done=self._done)

    def _export(self) -> None:
        def work() -> list[Path]:
            names = list(self.objects.value)
            if not names:
                raise ValueError("select the objects to export")
            out = Path(self._needed(self.export_out.value, "file or folder"))
            fmt = self.export_format.value
            ws = self.app.ws
            if len(names) == 1:
                target = out if out.suffix else out.with_suffix(f".{fmt}")
                return [ws.export(names[0], target, fmt)]
            if fmt == "h5":
                return [ws.export_nexus(names, out if out.suffix else out.with_suffix(".h5"))]
            return [ws.export(n, out / f"{n}.{fmt}", fmt) for n in names]

        self.app.submit("exporting", work, done=self._done)
