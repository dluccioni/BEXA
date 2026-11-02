"""The View tab: whatever is selected, drawn the way it should be.

- a scan: its raw frames, one at a time, a slider per motor (one frame read per move);
- a scan set: the same for any of its scans;
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
from bexa.gui.drawing import CMAPS, CurveOptions, ImageOptions, draw_curves, draw_image, draw_rgb
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
        self.sliders: dict[str, Any] = {}
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
        self.cmap = ipw.Dropdown(
            options=CMAPS,
            value="viridis",
            description="colours",
            layout=ipw.Layout(width="190px"),
            style={"description_width": "55px"},
        )
        self.log = ipw.Checkbox(
            value=False, description="log", indent=False, layout=ipw.Layout(width="60px")
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
            options=("equal", "auto"), value="equal", layout=ipw.Layout(width="80px")
        )
        self.image_options = ipw.HBox(
            [
                self.cmap,
                self.log,
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
            self.log,
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
                ipw.HBox([self.mode, self.member, self.row_dim, self.col_dim]),
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
        if not objects:
            self._key = None
            self.title.value = "<i>select an object</i>"
            self._visible(False, False)
            self.figure_box.layout.display = "none"
            self.table.value = ""
            return
        main = objects[0]
        key = (main.name, main.kind, main.shape)
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
            return [(d, np.asarray(s.coords[d])) for d in s.motor_dims]
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
            self._make_sliders(objects[0])
        self.redraw()

    def _scan(self, main: DataObject) -> Any:
        if main.kind == "scanset":
            return main.data.scans[int(self.member.value or 0)]
        return main.data

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
            elif main.kind == "rgb":
                self._visible(False, False)
                draw_rgb(fig, main.array, main.name)
            else:
                self._visible(True, False)
                image, title = self._image(main)
                self._shown = image
                rois = self.app.ws.rois if self.show_rois.value else None
                ax = draw_image(fig, image, self._image_options(), title, rois)
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
            self.cmap.value,
            self.log.value,
            tuple(self.percent.value),  # type: ignore[arg-type]
            number(self.vmin.value),
            number(self.vmax.value),
            self.colorbar.value,
            self.aspect.value,
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
