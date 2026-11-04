"""The View tab: whatever is selected, drawn the way it should be.

- a scan: its raw frames, one at a time, a slider per motor (one frame read per move);
  a point detector (a counter of a netCDF run: frames of 1 x 1) against its motor, or as
  an image of its last two motors;
- a scan set: the same for any of its scans;
- the detector a scan of several reads, chosen here: the object itself reads it, so the
  operations take it too (``Workspace.use_detector``);
- a stack: a slider per motor dim, or the frames summed / maximum / mean;
- an image or map: on its coordinates, with colour map, limits, log scale and ROIs;
- curves: every selected curve overlaid, with error bars and their fits;
- a volume: two dims as the image, a slider for each other;
- a colour picture, a table, a text report.

Named ROIs are typed here or, with ipympl, drawn on the image; a line drawn on
an image makes its line profile. With ipympl the value under the mouse shows.
"""

from __future__ import annotations

import html
from typing import Any

import numpy as np
import xarray as xr

from bexa._log import get_logger
from bexa.gui.drawing import (
    ASPECTS,
    SCALES,
    CurveOptions,
    ImageOptions,
    colormap_names,
    draw_curves,
    draw_image,
    draw_rgb,
)
from bexa.gui.figures import FigurePane
from bexa.gui.forms import set_tooltip
from bexa.workspace.objects import DataObject, has_pixel_dims

log = get_logger(__name__)

__all__ = ["ViewTab"]

MODES = {
    "stack": ("one frame", "sum", "max", "mean"),
    "map": ("image", "curves"),
    "scan": ("raw frames",),
    "scanset": ("raw frames",),
}
MAX_CURVES = 40  # a map drawn as curves shows at most this many


class ViewTab:
    """Draw the selected objects; ROIs and lines."""

    def __init__(self, app: Any) -> None:
        import ipywidgets as ipw

        self.app = app
        self.pane = FigurePane((6.6, 4.5), interactive=app.interactive_figures)
        self._key: tuple[Any, ...] | None = None
        self._shown: xr.DataArray | None = None
        self._line_start: tuple[float, float] | None = None
        self._selector: Any = None
        self._points_read: tuple[Any, int, xr.DataArray] | None = None
        self.sliders: dict[str, Any] = {}
        self.hidden_rois: set[str] = set()  # the ROIs not drawn (their toggles off)
        self.title = ipw.HTML()
        self.mode = ipw.Dropdown(
            description="show",
            layout=ipw.Layout(width="220px"),
            style={"description_width": "40px"},
        )
        self.member = ipw.Dropdown(
            description="scan",
            layout=ipw.Layout(width="300px", display="none"),
            style={"description_width": "40px"},
        )
        self.detector = ipw.Dropdown(
            description="detector",
            layout=ipw.Layout(width="280px", display="none"),
            style={"description_width": "60px"},
        )
        self.detector.observe(lambda c: self._detector_chosen(c["new"]), names="value")
        set_tooltip(self.detector, "the detector the scan reads, here and in every operation")
        self.row_dim = ipw.Dropdown(
            description="rows", layout=ipw.Layout(width="180px", display="none")
        )
        self.col_dim = ipw.Dropdown(
            description="columns", layout=ipw.Layout(width="200px", display="none")
        )
        for widget in (self.mode, self.member, self.row_dim, self.col_dim):
            widget.observe(lambda _c: self._controls_changed(), names="value")
        self.slider_box = ipw.VBox()
        # image options
        self.cmap = ipw.Combobox(  # the usual maps offered, any matplotlib name typed
            options=colormap_names(),
            value="auto",
            placeholder="auto",
            ensure_option=False,
            continuous_update=False,
            description="colours",
            layout=ipw.Layout(width="190px"),
            style={"description_width": "55px"},
        )
        self.reverse = ipw.Checkbox(
            value=False, description="reversed", indent=False, layout=ipw.Layout(width="90px")
        )
        self.scale = ipw.Dropdown(
            options=SCALES,
            value="auto",
            description="scale",
            layout=ipw.Layout(width="150px"),
            style={"description_width": "40px"},
        )
        self.percent = ipw.FloatRangeSlider(
            value=(1.0, 99.5),
            min=0.0,
            max=100.0,
            step=0.5,
            description="limits %",
            continuous_update=False,
            layout=ipw.Layout(width="300px"),
        )
        self.vmin = ipw.Text(placeholder="min", layout=ipw.Layout(width="80px"))
        self.vmax = ipw.Text(placeholder="max", layout=ipw.Layout(width="80px"))
        self.colorbar = ipw.Checkbox(
            value=True, description="colour bar", indent=False, layout=ipw.Layout(width="100px")
        )
        self.show_rois = ipw.Checkbox(
            value=True, description="ROIs", indent=False, layout=ipw.Layout(width="70px")
        )
        self.aspect = ipw.Dropdown(
            options=ASPECTS,
            value="auto",
            description="axes",
            layout=ipw.Layout(width="160px"),
            style={"description_width": "35px"},
        )
        self.image_options = ipw.HBox(
            [
                self.cmap,
                self.reverse,
                self.scale,
                self.percent,
                self.vmin,
                self.vmax,
                self.colorbar,
                self.show_rois,
                self.aspect,
            ],
            layout=ipw.Layout(flex_flow="row wrap"),
        )
        # curve options
        self.log_y = ipw.Checkbox(
            value=False, description="log y", indent=False, layout=ipw.Layout(width="80px")
        )
        self.errors = ipw.Checkbox(
            value=True, description="error bars", indent=False, layout=ipw.Layout(width="100px")
        )
        self.markers = ipw.Checkbox(
            value=True, description="markers", indent=False, layout=ipw.Layout(width="90px")
        )
        self.normalize = ipw.Checkbox(
            value=False,
            description="normalise view",
            indent=False,
            layout=ipw.Layout(width="130px"),
        )
        self.fits = ipw.Checkbox(
            value=True, description="fits", indent=False, layout=ipw.Layout(width="60px")
        )
        self.curve_options = ipw.HBox(
            [self.log_y, self.errors, self.markers, self.normalize, self.fits]
        )
        for widget in (
            self.cmap,
            self.reverse,
            self.scale,
            self.percent,
            self.colorbar,
            self.show_rois,
            self.aspect,
            self.log_y,
            self.errors,
            self.markers,
            self.normalize,
            self.fits,
        ):
            widget.observe(lambda _c: self.redraw(), names="value")
        for widget in (self.vmin, self.vmax):  # redrawn when the typing is done
            widget.continuous_update = False
            widget.observe(lambda _c: self.redraw(), names="value")
        self.readout = ipw.HTML()
        self.table = ipw.HTML()
        self.figure_box = ipw.Box([self.pane.widget])
        self.box = ipw.VBox(
            [
                self.title,
                ipw.HBox([self.mode, self.member, self.detector, self.row_dim, self.col_dim]),
                self.slider_box,
                self.image_options,
                self.curve_options,
                self.figure_box,
                self.readout,
                self.table,
                self._roi_tools(),
                self._extras(),
            ]
        )
        self.pane.on_click(self._clicked)
        self.pane.on_move(self._moved)
        self.refresh()

    # ------------------------------------------------------------------ tools
    def _roi_tools(self) -> Any:
        import ipywidgets as ipw

        self.roi_list = ipw.Select(options=[], rows=4, layout=ipw.Layout(width="200px"))
        self.roi_list.observe(lambda c: self._roi_picked(c["new"]), names="value")
        self.roi_toggles = ipw.HBox(layout=ipw.Layout(flex_flow="row wrap"))
        self.roi_toggles.add_class("bexa-roi-toggles")
        self.roi_name = ipw.Text(
            value="roi1",
            description="name",
            layout=ipw.Layout(width="200px"),
            style={"description_width": "40px"},
        )
        self.roi_ranges = ipw.Text(
            description="ranges",
            placeholder="y=100:160, x=220:300, mu=-0.2:0.2",
            layout=ipw.Layout(width="440px"),
            style={"description_width": "50px"},
        )
        set_roi = ipw.Button(
            description="Set ROI", button_style="primary", layout=ipw.Layout(width="90px")
        )
        delete = ipw.Button(description="Delete", layout=ipw.Layout(width="80px"))
        set_roi.on_click(lambda _b: self._set_roi())
        delete.on_click(lambda _b: self._delete_roi())
        self.draw = ipw.ToggleButton(
            description="Draw ROI",
            icon="square-o",
            layout=ipw.Layout(width="110px"),
            tooltip="drag a rectangle on the image (needs ipympl)",
        )
        self.line = ipw.ToggleButton(
            description="Line profile",
            icon="minus",
            layout=ipw.Layout(width="120px"),
            tooltip="click two points on an image (needs ipympl)",
        )
        self.draw.observe(lambda c: self._draw_mode(c["new"]), names="value")
        self.line.observe(lambda c: self._line_mode(c["new"]), names="value")
        set_tooltip(
            self.roi_ranges,
            "pixel ranges: full-detector pixels, end excluded; motor ranges: values"
            + ("" if self.pane.interactive else " (no ipympl: type them)"),
        )
        tools = ipw.VBox(
            [
                ipw.HTML("<b>ROIs</b>"),
                ipw.HBox(
                    [
                        self.roi_list,
                        ipw.VBox(
                            [
                                ipw.HBox([self.roi_name, set_roi, delete]),
                                self.roi_ranges,
                                ipw.HBox([self.draw, self.line]),
                            ]
                        ),
                    ]
                ),
                self.roi_toggles,
            ]
        )
        tools.add_class("bexa-rois")
        return tools

    def _extras(self) -> Any:
        import ipywidgets as ipw

        self.save_path = ipw.Text(placeholder="figure.png", layout=ipw.Layout(width="300px"))
        save = ipw.Button(description="Save figure", icon="save")
        save.on_click(lambda _b: self._save())
        self.render_mode = ipw.Dropdown(
            options=("translucent", "mip", "iso"),
            value="translucent",
            layout=ipw.Layout(width="130px"),
        )
        render = ipw.Button(
            description="3-D view", icon="cube", tooltip="plotly volume of the stack"
        )
        render.on_click(lambda _b: self._render())
        self.render_out = ipw.Output()
        return ipw.VBox(
            [ipw.HBox([self.save_path, save, self.render_mode, render]), self.render_out]
        )

    # ------------------------------------------------------------------ selection
    def _objects(self) -> list[DataObject]:
        ws = self.app.ws
        return [ws[n] for n in self.app.selection if n in ws]

    def refresh(self) -> None:
        """Show the current selection (rebuild the controls when it changed)."""
        objects = self._objects()
        self.roi_list.options = list(self.app.ws.rois)
        self._roi_toggles()
        if not objects:
            self._key = None
            self.title.value = "<i>select an object</i>"
            self._visible(False, False)
            self.figure_box.layout.display = "none"
            self.table.value = ""
            return
        main = objects[0]
        key = (main.name, main.kind, main.shape, self._reads(main))
        if key != self._key:
            same = self._key is not None and self._key[:2] == key[:2]  # a scan that grew
            self._key = key
            self._build(main, keep=same)
        self.redraw()

    def _visible(self, image: bool, curves: bool) -> None:
        self.image_options.layout.display = "" if image else "none"
        self.curve_options.layout.display = "" if curves else "none"

    def _build(self, main: DataObject, keep: bool = False) -> None:
        """The controls for one object: mode, member scan, image dims, sliders.

        ``keep``: the same object again (a scan being written grew), whose mode
        and member scan stay as they were.
        """
        self._syncing = True
        try:
            modes = MODES.get(main.kind, ("default",))
            mode, member = self.mode.value, self.member.value
            self.mode.options = modes
            self.mode.value = mode if keep and mode in modes else modes[0]
            self.mode.layout.display = "" if len(modes) > 1 else "none"
            self.member.layout.display = "" if main.kind == "scanset" else "none"
            if main.kind == "scanset":
                self.member.options = [(label, k) for k, label in enumerate(main.data.labels)]
                self.member.value = member if keep and member in range(len(main.data)) else 0
            self._detector_options(main)
            volume = main.kind == "volume"
            self.row_dim.layout.display = self.col_dim.layout.display = "" if volume else "none"
            if volume:
                dims = list(main.dims)
                self.row_dim.options, self.col_dim.options = dims, dims
                self.row_dim.value, self.col_dim.value = dims[-2], dims[-1]
        finally:
            self._syncing = False
        self._make_sliders(main)

    def _slider_dims(self, main: DataObject) -> list[tuple[str, np.ndarray]]:
        if main.kind in ("scan", "scanset"):
            scan = self._scan(main)
            s = scan.structure
            # a point detector draws its last one or two motors at once
            dims = s.motor_dims[:-2] if self._points(main) else s.motor_dims
            return [(d, np.asarray(s.coords[d])) for d in dims]
        if main.kind == "stack" and self.mode.value == "one frame":
            data = main.array
            return [(str(d), main.coord(str(d))) for d in data.dims[:-2]]
        if main.kind == "volume":
            shown = {self.row_dim.value, self.col_dim.value}
            return [(d, main.coord(d)) for d in main.dims if d not in shown]
        return []

    def _make_sliders(self, main: DataObject) -> None:
        import ipywidgets as ipw

        old = {d: s.value for d, s in self.sliders.items()}
        at_end = {d for d, s in self.sliders.items() if s.max > 0 and s.value == s.max}
        self.sliders = {}
        rows = []
        for dim, values in self._slider_dims(main):
            last = max(len(values) - 1, 0)
            # a slider left at its last point follows a scan being written
            start = last if dim in at_end else min(old.get(dim, len(values) // 2), last)
            slider = ipw.IntSlider(
                min=0,
                max=last,
                value=start,
                description=dim,
                continuous_update=self.pane.interactive,
                layout=ipw.Layout(width="60%"),
            )
            label = ipw.HTML()
            slider.observe(lambda _c: self.redraw(), names="value")
            self.sliders[dim] = slider
            slider.label = label  # type: ignore[attr-defined]
            slider.values = values  # type: ignore[attr-defined]
            rows.append(ipw.HBox([slider, label]))
        self.slider_box.children = rows

    _syncing = False

    def _controls_changed(self) -> None:
        if self._syncing:
            return
        objects = self._objects()
        if objects:
            self._detector_options(objects[0])  # another scan of a set: its own
            self._make_sliders(objects[0])
        self.redraw()

    def _scan(self, main: DataObject) -> Any:
        if main.kind == "scanset":
            scans = main.data.scans
            k = int(self.member.value or 0)  # still another set's before the controls are built
            return scans[k] if k < len(scans) else scans[0]
        return main.data

    def _reads(self, main: DataObject) -> str | None:
        """The detector the scan shown reads (none for other objects)."""
        return self._scan(main).detector if main.kind in ("scan", "scanset") else None

    def _points(self, main: DataObject) -> bool:
        """True for a scan read with a point detector: a frame of 1 x 1 per point."""
        scans = main.kind in ("scan", "scanset")
        return scans and tuple(self._scan(main).frame_shape) == (1, 1)

    # ------------------------------------------------------------------ detectors
    def _detector_options(self, main: DataObject) -> None:
        """The detectors of the scan shown, the one it reads chosen; hidden for one."""
        names = self._scan(main).detectors if main.kind in ("scan", "scanset") else []
        syncing, self._syncing = self._syncing, True
        try:
            self.detector.options = names
            reads = self._reads(main)
            self.detector.value = reads if reads in names else (names[0] if names else None)
            self.detector.layout.display = "" if len(names) > 1 else "none"
        finally:
            self._syncing = syncing

    def _detector_chosen(self, detector: str | None) -> None:
        """Read the scan shown (every scan of a set that holds it) with ``detector``: the
        object itself, so every operation reads that detector too."""
        objects = self._objects()
        if self._syncing or not detector or not objects:
            return
        main = objects[0]
        if main.kind not in ("scan", "scanset") or self._reads(main) == detector:
            return
        # the workspace says the object changed, and the panels follow (this one redraws)
        done = self.app.guarded(lambda: self.app.ws.use_detector(main.name, detector))
        if done is None:  # it failed (said in red): the dropdown shows what is read
            self._detector_options(main)
            return
        self.app.say(f"{main.name} reads {detector}", "ok")

    def _index(self) -> dict[str, int]:
        out = {}
        for dim, slider in self.sliders.items():
            out[dim] = int(slider.value)
            value = slider.values[slider.value] if len(slider.values) else ""
            shown = (
                f"{value:.5g}"
                if isinstance(value, (float, np.floating, int, np.integer))
                else value
            )
            slider.label.value = f"<code>{dim} = {shown}</code>"
        return out

    # ------------------------------------------------------------------ drawing
    def redraw(self) -> None:
        """Draw the current selection with the current controls."""
        objects = self._objects()
        if not objects:
            return
        main = objects[0]
        self._shown = None
        self.table.value = ""
        self.figure_box.layout.display = ""
        fig = self.pane.clear()
        try:
            if main.kind in ("table", "text"):
                self._show_table(main)
                return
            if main.kind == "curve" or (main.kind == "map" and self.mode.value == "curves"):
                self._visible(False, True)
                self._curves(fig, objects)
            elif self._points(main):
                self._draw_points(fig, main)
            elif main.kind == "rgb":
                self._visible(True, False)  # its axes to scale or not
                draw_rgb(fig, main.array, main.name, self._image_options())
            else:
                self._visible(True, False)
                image, title = self._image(main)
                self._shown = image
                rois = self.app.ws.rois if self.show_rois.value else None
                ax = draw_image(fig, image, self._image_options(), title, rois, self.hidden_rois)
                if self.draw.value:
                    self._attach_selector(ax)
        except Exception as exc:
            fig.clf()
            fig.text(
                0.5,
                0.5,
                f"cannot draw {main.name}:\n{type(exc).__name__}: {exc}",
                ha="center",
                va="center",
                wrap=True,
            )
            log.debug("drawing %s failed", main.name, exc_info=True)
        self._title(main)
        self.pane.draw()

    def _title(self, main: DataObject) -> None:
        self.title.value = (
            f"<b>{html.escape(main.name)}</b> "
            f"<span style='color:#757575'>{html.escape(main.describe())}</span>"
        )

    def _image_options(self) -> ImageOptions:
        def number(text: str) -> float | None:
            try:
                return float(text) if text.strip() else None
            except ValueError:
                return None

        return ImageOptions(
            cmap=self.cmap.value.strip() or "auto",
            scale=self.scale.value,
            percentiles=tuple(self.percent.value),  # type: ignore[arg-type]
            vmin=number(self.vmin.value),
            vmax=number(self.vmax.value),
            colorbar=self.colorbar.value,
            aspect=self.aspect.value,
            reverse=self.reverse.value,
        )

    def _image(self, main: DataObject) -> tuple[xr.DataArray, str]:
        index = self._index()
        if main.kind in ("scan", "scanset"):
            scan = self._scan(main)
            s = scan.structure
            position = tuple(index[d] for d in s.motor_dims)
            frame_id = int(s.frame_index[position]) if s.motor_dims else 0
            where = ", ".join(f"{d}={s.coords[d][index[d]]:.5g}" for d in s.motor_dims)
            if frame_id < 0:
                h, w = s.frame_shape
                return xr.DataArray(
                    np.full((h, w), np.nan), dims=("y", "x")
                ), f"{where}: no frame (missing)"
            frame = scan.source.read_frames(np.array([frame_id]))[0]
            data = xr.DataArray(
                np.asarray(frame, dtype=float),
                dims=("y", "x"),
                coords={"y": np.arange(frame.shape[0]), "x": np.arange(frame.shape[1])},
            )
            return data, f"frame {frame_id} at {where}"
        data = main.array
        if main.kind == "stack":
            motors = [str(d) for d in data.dims[:-2]]
            if self.mode.value == "one frame":
                picked = data.isel({d: index[d] for d in motors})
                where = ", ".join(f"{d}={main.coord(d)[index[d]]:.5g}" for d in motors)
                return picked, where
            how = self.mode.value
            return getattr(data, how)(motors, skipna=True), f"{how} of the frames"
        if main.kind == "volume":
            rows, cols = self.row_dim.value, self.col_dim.value
            if rows == cols:
                raise ValueError("choose two different dims for the rows and the columns")
            picked = data.isel({d: index[d] for d in index})
            return picked.transpose(rows, cols), ", ".join(
                f"{d} index {i}" for d, i in index.items()
            )
        return data, ""

    def _point_values(self, scan: Any) -> xr.DataArray:
        """A point detector's value at every point of the scan's grid (NaN where none was
        taken), read once per scan and frame count."""
        s = scan.structure
        kept = self._points_read
        if kept is not None and kept[0] is scan and kept[1] == s.n_frames:
            return kept[2]
        values = np.asarray(scan.source.read_frames(np.arange(s.n_frames)), dtype=float)
        grid = np.full(s.frame_index.shape, np.nan)
        taken = s.frame_index >= 0
        grid[taken] = values.reshape(-1)[s.frame_index[taken]]
        coords = {d: np.asarray(s.coords[d]) for d in s.motor_dims}
        data = xr.DataArray(grid, dims=s.motor_dims, coords=coords, name=scan.detector)
        self._points_read = (scan, s.n_frames, data)
        return data

    def _draw_points(self, fig: Any, main: DataObject) -> None:
        """A point detector against the scan's motor, or as an image of its last two motors
        (a slider for each other)."""
        scan = self._scan(main)
        index = self._index()
        values = self._point_values(scan)
        picked = values.isel(index)
        where = ", ".join(f"{d}={values[d].values[i]:.5g}" for d, i in index.items())
        title = str(scan.detector) + (f" at {where}" if where else "")
        if picked.ndim == 1:
            self._visible(False, True)
            options = CurveOptions(
                self.log_y.value, self.errors.value, self.markers.value, self.normalize.value
            )
            draw_curves(fig, [(str(scan.detector), picked, None)], options, title=title)
            return
        self._visible(True, False)
        self._shown = picked
        rois = self.app.ws.rois if self.show_rois.value else None
        ax = draw_image(fig, picked, self._image_options(), title, rois, self.hidden_rois)
        if self.draw.value:
            self._attach_selector(ax)

    def _curves(self, fig: Any, objects: list[DataObject]) -> None:
        main = objects[0]
        curves: list[tuple[str, xr.DataArray, Any]] = []
        if main.kind == "map":
            rows = str(main.array.dims[0])
            step = max(1, main.array.sizes[rows] // MAX_CURVES)
            for k in range(0, main.array.sizes[rows], step):
                sigma = None if main.sigma is None else main.sigma.isel({rows: k})
                curves.append(
                    (f"{rows}={main.coord(rows)[k]:.4g}", main.array.isel({rows: k}), sigma)
                )
        else:
            curves = [(o.name, o.array, o.sigma) for o in objects if o.kind == "curve"]
        fits = []
        if self.fits.value:
            names = {c[0] for c in curves}
            for obj in self.app.ws.objects.values():
                if (
                    obj.kind == "curve"
                    and obj.meta.get("fit_of") in names
                    and obj.meta.get("quantity") == "fit"
                ):
                    fits.append((obj.name, obj.array))
        options = CurveOptions(
            self.log_y.value, self.errors.value, self.markers.value, self.normalize.value
        )
        draw_curves(fig, curves, options, fits, "" if len(curves) > 1 else main.name)

    def _show_table(self, main: DataObject) -> None:
        self._visible(False, False)
        self.figure_box.layout.display = "none"
        if main.kind == "text":
            self.table.value = (
                f"<pre style='white-space:pre-wrap'>{html.escape(str(main.data))}</pre>"
            )
        else:
            frame = main.data
            more = f"<i>the first 200 of {len(frame)} rows</i>" if len(frame) > 200 else ""
            rows = frame.head(200).to_html(float_format=lambda v: f"{v:.6g}")
            self.table.value = (
                f"<div style='max-height:480px;overflow:auto;font-size:12px'>{rows}</div>{more}"
            )
        self._title(main)

    # ------------------------------------------------------------------ the mouse
    def _value_at(self, x: float, y: float) -> str:
        image = self._shown
        if image is None:
            return f"x = {x:.5g}, y = {y:.5g}"
        dy, dx = (str(d) for d in image.dims[-2:])
        try:
            value = float(image.sel({dy: y, dx: x}, method="nearest").values)
        except Exception:
            return f"{dx} = {x:.5g}, {dy} = {y:.5g}"
        return f"{dx} = {x:.5g}, {dy} = {y:.5g}, value = {value:.6g}"

    def _moved(self, event: Any) -> None:
        if event.xdata is not None and event.ydata is not None:
            self.readout.value = (
                f"<code>{html.escape(self._value_at(event.xdata, event.ydata))}</code>"
            )

    def _clicked(self, event: Any) -> None:
        if event.xdata is None or event.ydata is None:
            return
        if self.line.value:
            point = (float(event.ydata), float(event.xdata))
            if self._line_start is None:
                self._line_start = point
                self.readout.value = "<code>line: click the end point</code>"
                return
            (y0, x0), (y1, x1) = self._line_start, point
            self._line_start = None
            objects = self._objects()
            if objects and objects[0].kind in ("image", "map"):
                self.app.run_op(
                    "line_profile", [objects[0].name], {"y0": y0, "x0": x0, "y1": y1, "x1": x1}
                )
            else:
                self.app.say(
                    "a line profile needs an image or map object (make one first)", "error"
                )
            return
        self.readout.value = f"<code>{html.escape(self._value_at(event.xdata, event.ydata))}</code>"

    def _draw_mode(self, active: bool) -> None:
        if active and not self.pane.interactive:
            self.app.say(
                "drawing needs ipympl (%matplotlib widget); type the ranges instead", "error"
            )
            self.draw.value = False
            return
        if active:
            self.line.value = False
        self.redraw()

    def _line_mode(self, active: bool) -> None:
        if active and not self.pane.interactive:
            self.app.say(
                "lines need ipympl; use the Line Profile operation on the Analysis tab", "error"
            )
            self.line.value = False
            return
        if active:
            self.draw.value = False
        self._line_start = None

    def _attach_selector(self, ax: Any) -> None:
        from matplotlib.widgets import RectangleSelector

        self._selector = RectangleSelector(
            ax, self._drawn, useblit=False, interactive=True, button=[1]
        )

    def _drawn(self, press: Any, release: Any) -> None:
        image = self._shown
        if image is None:
            return
        dy, dx = (str(d) for d in image.dims[-2:])
        y0, y1 = sorted((press.ydata, release.ydata))
        x0, x1 = sorted((press.xdata, release.xdata))
        if has_pixel_dims(image.dims):
            rows = f"{int(np.floor(y0))}:{int(np.ceil(y1))}"
            columns = f"{int(np.floor(x0))}:{int(np.ceil(x1))}"
            text = f"y={rows}, x={columns}"
        else:
            text = f"{dy}={y0:.6g}:{y1:.6g}, {dx}={x0:.6g}:{x1:.6g}"
        self.roi_ranges.value = text

    # ------------------------------------------------------------------ ROIs
    def _roi_picked(self, name: str | None) -> None:
        if not name or name not in self.app.ws.rois:
            return
        roi = self.app.ws.rois[name]
        self.roi_name.value = name
        self.roi_ranges.value = ", ".join(
            f"{d}={'' if lo is None else f'{lo:g}'}:{'' if hi is None else f'{hi:g}'}"
            for d, (lo, hi) in roi.ranges.items()
        )

    def _roi_toggles(self) -> None:
        """One toggle per ROI, on while it is drawn: made again when the ROIs change."""
        import ipywidgets as ipw

        names = list(self.app.ws.rois)
        self.hidden_rois &= set(names)  # a deleted ROI is forgotten
        if [b.description for b in self.roi_toggles.children[1:]] == names:
            return
        for button in self.roi_toggles.children[1:]:  # a widget lives until closed
            button.close()
        buttons = []
        for name in names:
            shown = name not in self.hidden_rois
            button = ipw.ToggleButton(
                value=shown,
                description=name,
                icon="eye" if shown else "eye-slash",
                tooltip="show or hide this ROI on the image",
                layout=ipw.Layout(width="auto"),
            )
            button.observe(lambda c, n=name: self._roi_shown(n, c["owner"]), names="value")
            buttons.append(button)
        label = ipw.HTML("shown:" if names else "", layout=ipw.Layout(width="50px"))
        self.roi_toggles.children = [label, *buttons]

    def _roi_shown(self, name: str, button: Any) -> None:
        if button.value:
            self.hidden_rois.discard(name)
        else:
            self.hidden_rois.add(name)
        button.icon = "eye" if button.value else "eye-slash"
        self.redraw()

    def _set_roi(self) -> None:
        from bexa.core.roi import ROI

        name = self.roi_name.value.strip() or "roi"
        text = self.roi_ranges.value.replace(" ", "")
        self.app.guarded(lambda: self.app.ws.set_roi(name, ROI.parse(text)))

    def _delete_roi(self) -> None:
        name = self.roi_list.value
        if name:
            self.app.guarded(lambda: self.app.ws.remove_roi(name))

    # ------------------------------------------------------------------ extras
    def _save(self) -> None:
        path = self.save_path.value.strip() or "figure.png"
        self.app.guarded(
            lambda: self.pane.savefig(path), show=lambda _r: self.app.say(f"saved {path}", "ok")
        )

    def _render(self) -> None:
        objects = self._objects()
        if not objects or objects[0].kind not in ("stack", "volume"):
            self.app.say("the 3-D view takes a stack or a volume", "error")
            return
        data = objects[0].array
        while data.ndim > 3:
            data = data.sum(str(data.dims[0]))

        def show() -> None:
            from bexa.viz.volume import render, show_plotly

            figure = render(data, mode=self.render_mode.value, title=objects[0].name)
            self.render_out.clear_output()
            with self.render_out:
                show_plotly(figure, height=520)

        self.app.guarded(show)
