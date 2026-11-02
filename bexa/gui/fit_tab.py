"""The Fit tab: peaks with every parameter in hand.

Choose a curve (or a map, to fit each of its curves), guess or add peaks (with
ipympl, click on the plot to place one), give any parameter its start value,
bounds, or fix it, pick a background, and Fit. The fitted values go back into
the rows, so a second Fit refines; the best fit, its components and residuals
are drawn; the best-fit curve and the parameter table become objects.
"""

from __future__ import annotations

import html
from typing import Any

import numpy as np

from bexa.analysis import peakfit
from bexa.gui.figures import FigurePane
from bexa.gui.forms import object_options

__all__ = ["FitTab", "PeakRow"]

BACKGROUND_NAMES = ("none", "constant", "linear", "quadratic", "exponential")


def _number(text: str) -> float | None:
    text = str(text).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"{text!r} is not a number") from None


def _shown(value: float | None) -> str:
    return "" if value is None or not np.isfinite(value) else f"{value:.6g}"


class ParamRow:
    """One parameter: start value, bounds, fixed."""

    def __init__(self, name: str, value: float | None = None) -> None:
        import ipywidgets as ipw

        self.name = name
        small = ipw.Layout(width="95px")
        self.value = ipw.Text(value=_shown(value), placeholder="guess", layout=small)
        self.low = ipw.Text(placeholder="min", layout=small)
        self.high = ipw.Text(placeholder="max", layout=small)
        self.fixed = ipw.Checkbox(
            value=False, indent=False, description="fix", layout=ipw.Layout(width="55px")
        )
        self.box = ipw.HBox(
            [
                ipw.HTML(f"<div style='width:95px'>{name}</div>"),
                self.value,
                self.low,
                self.high,
                self.fixed,
            ]
        )

    def spec(self) -> dict[str, Any] | None:
        value = _number(self.value.value)
        if value is None:
            return None
        return {
            "value": value,
            "min": _number(self.low.value),
            "max": _number(self.high.value),
            "fixed": bool(self.fixed.value),
        }

    def set(self, value: float) -> None:
        self.value.value = _shown(value)


class PeakRow:
    """One peak: its shape and the parameters of that shape."""

    def __init__(
        self, tab: FitTab, shape: str = "gaussian", start: dict[str, float] | None = None
    ) -> None:
        import ipywidgets as ipw

        self.tab = tab
        self.shape = ipw.Dropdown(
            options=tuple(peakfit.SHAPES), value=shape, layout=ipw.Layout(width="150px")
        )
        self.shape.observe(lambda _c: self._rebuild(), names="value")
        remove = ipw.Button(
            icon="times", tooltip="remove this peak", layout=ipw.Layout(width="40px")
        )
        remove.on_click(lambda _b: tab.remove_peak(self))
        self.head = ipw.HBox([self.shape, remove])
        self.params: dict[str, ParamRow] = {}
        self.body = ipw.VBox()
        self.box = ipw.VBox(
            [self.head, self.body],
            layout=ipw.Layout(border="1px solid #e0e0e0", padding="4px", margin="2px 0"),
        )
        self._rebuild(start or {})

    def _rebuild(self, start: dict[str, float] | None = None) -> None:
        values = {n: _number(p.value.value) for n, p in self.params.items()}
        values.update(start or {})
        self.params = {n: ParamRow(n, values.get(n)) for n in peakfit.SHAPES[self.shape.value]}
        self.body.children = [p.box for p in self.params.values()]

    def spec(self) -> dict[str, Any]:
        out: dict[str, Any] = {"shape": self.shape.value}
        for name, row in self.params.items():
            given = row.spec()
            if given is not None:
                out[name] = given
        return out

    def update(self, values: dict[str, float]) -> None:
        for name, row in self.params.items():
            if name in values:
                row.set(values[name])


class FitTab:
    """Interactive peak fitting."""

    def __init__(self, app: Any) -> None:
        import ipywidgets as ipw

        self.app = app
        self.peaks: list[PeakRow] = []
        self.last_fit: dict[str, Any] | None = None
        self._table: str | None = None  # the parameter table of the last fit
        self.pane = FigurePane((4.5, 4.3), interactive=app.interactive_figures)
        style = {"description_width": "80px"}
        self.target = ipw.Dropdown(
            description="curve", style=style, layout=ipw.Layout(width="420px")
        )
        self.target.observe(lambda _c: self._target_changed(), names="value")
        self.along = ipw.Dropdown(
            description="fit along", style=style, layout=ipw.Layout(width="260px", display="none")
        )
        self.row = ipw.IntSlider(
            description="curve no.",
            min=0,
            max=0,
            continuous_update=False,
            style=style,
            layout=ipw.Layout(width="420px", display="none"),
        )
        self.along.observe(lambda _c: self._target_changed(), names="value")
        self.row.observe(lambda _c: self.draw(), names="value")
        self.low = ipw.Text(
            placeholder="from",
            description="fit range",
            style=style,
            layout=ipw.Layout(width="200px"),
        )
        self.high = ipw.Text(placeholder="to", layout=ipw.Layout(width="110px"))
        self.use_sigma = ipw.Checkbox(
            value=True, description="weight by the uncertainties", indent=False
        )
        self.background = ipw.Dropdown(
            options=BACKGROUND_NAMES,
            value="linear",
            description="background",
            style=style,
            layout=ipw.Layout(width="250px"),
        )
        self.background.observe(lambda _c: self._background_rows(), names="value")
        self.bg_box = ipw.VBox()
        self.n_guess = ipw.BoundedIntText(
            value=1,
            min=1,
            max=20,
            description="peaks",
            style=style,
            layout=ipw.Layout(width="150px"),
        )
        self.guess_shape = ipw.Dropdown(
            options=tuple(peakfit.SHAPES), value="gaussian", layout=ipw.Layout(width="150px")
        )
        guess = ipw.Button(description="Guess", icon="magic", tooltip="the most prominent peaks")
        add = ipw.Button(description="Add peak", icon="plus")
        self.click = ipw.ToggleButton(
            description="Click to add",
            icon="mouse-pointer",
            tooltip="click on the plot to add a peak there (needs ipympl)",
        )
        draw_start = ipw.Button(description="Draw start values", icon="pencil")
        guess.on_click(lambda _b: self.guess())
        add.on_click(lambda _b: self.add_peak())
        draw_start.on_click(lambda _b: self.draw(start=True))
        fit = ipw.Button(
            description="Fit", button_style="success", icon="play", layout=ipw.Layout(width="100px")
        )
        series = ipw.Button(
            description="Fit every curve", tooltip="a map: each of its curves, warm-started"
        )
        fit.on_click(lambda _b: self.fit())
        series.on_click(lambda _b: self.fit_series())
        self.peak_box = ipw.VBox()
        self.results = ipw.HTML()
        self.export_path = ipw.Text(
            placeholder="fit.csv (or .json, .h5)", layout=ipw.Layout(width="260px")
        )
        export = ipw.Button(description="Save parameters", icon="save")
        export.on_click(lambda _b: self._export())
        self.pane.on_click(self._clicked)
        self.box = ipw.HBox(
            [
                ipw.VBox(
                    [
                        self.target,
                        self.along,
                        self.row,
                        ipw.HBox([self.low, self.high]),
                        self.use_sigma,
                        self.background,
                        self.bg_box,
                        ipw.HBox([self.n_guess, self.guess_shape, guess]),
                        ipw.HBox([add, self.click, draw_start]),
                        ipw.HTML(
                            "<b>Peaks</b> <span style='color:#757575;font-size:11px'>"
                            "value (empty: guessed), "
                            "min, max, fixed; centre and FWHM in x units, amplitude = height</span>"
                        ),
                        self.peak_box,
                        ipw.HBox([fit, series]),
                    ],
                    layout=ipw.Layout(width="44%", padding="0 10px 0 0"),
                ),
                ipw.VBox(
                    [self.pane.widget, self.results, ipw.HBox([self.export_path, export])],
                    layout=ipw.Layout(width="56%", overflow="auto"),
                ),
            ]
        )
        self._background_rows()
        self.refresh()

    # ------------------------------------------------------------------ the target
    def refresh(self) -> None:
        options = object_options(self.app.ws, ("curve", "map", "volume"))
        current = self.target.value
        self.target.options = options or [("no curve yet", None)]
        names = [v for _, v in options]
        # a newly selected curve becomes the target; fit results (just made) never do
        picked = [
            n for n in self.app.selection if n in names and "fit_of" not in self.app.ws[n].meta
        ]
        if picked:
            self.target.value = picked[0]
        else:
            self.target.value = current if current in names else (names[0] if names else None)

    def _object(self) -> Any:
        name = self.target.value
        return self.app.ws.objects.get(name) if name else None

    def _target_changed(self) -> None:
        obj = self._object()
        several = obj is not None and obj.kind in ("map", "volume")
        self.along.layout.display = "" if several else "none"
        self.row.layout.display = "" if several else "none"
        if several:
            dims = list(obj.dims)
            if self.along.options != tuple(dims):
                self.along.options = dims
                self.along.value = dims[-1]
            others = [d for d in dims if d != self.along.value]
            self.row.max = max(int(np.prod([obj.array.sizes[d] for d in others])) - 1, 0)
        self.draw()

    def data(self) -> tuple[str, np.ndarray, np.ndarray, np.ndarray | None]:
        """``(dim, x, y, sigma)`` of the curve fitted (a map: the curve picked by the slider)."""
        obj = self._object()
        if obj is None:
            raise ValueError("choose a curve")
        if obj.kind == "curve":
            dim = str(obj.dims[0])
            sigma = None if obj.sigma is None else np.asarray(obj.sigma.values, float)
            return dim, obj.coord(dim).astype(float), np.asarray(obj.array.values, float), sigma
        along = self.along.value
        others = [d for d in obj.dims if d != along]
        ordered = obj.array.transpose(*others, along)
        rows = np.asarray(ordered.values, float).reshape(-1, obj.array.sizes[along])
        sigma = None
        if obj.sigma is not None:
            sigma = np.asarray(obj.sigma.transpose(*others, along).values, float).reshape(
                rows.shape
            )[self.row.value]
        return along, obj.coord(along).astype(float), rows[self.row.value], sigma

    def x_range(self) -> tuple[float | None, float | None] | None:
        low, high = _number(self.low.value), _number(self.high.value)
        return None if low is None and high is None else (low, high)

    # ------------------------------------------------------------------ peaks
    def add_peak(self, shape: str | None = None, start: dict[str, float] | None = None) -> PeakRow:
        row = PeakRow(self, shape or self.guess_shape.value, start)
        self.peaks.append(row)
        self.peak_box.children = [p.box for p in self.peaks]
        return row

    def remove_peak(self, row: PeakRow) -> None:
        self.peaks = [p for p in self.peaks if p is not row]
        self.peak_box.children = [p.box for p in self.peaks]

    def guess(self) -> None:
        def guessed() -> None:
            _, x, y, _ = self.data()
            keep = np.isfinite(y)
            rng = self.x_range()
            if rng is not None:
                keep &= (x >= (rng[0] if rng[0] is not None else -np.inf)) & (
                    x <= (rng[1] if rng[1] is not None else np.inf)
                )
            found = peakfit.guess_peaks(
                x[keep], y[keep], int(self.n_guess.value), self.guess_shape.value
            )
            self.peaks = []
            for peak in found:
                self.add_peak(peak["shape"], {k: v for k, v in peak.items() if k != "shape"})
            self.peak_box.children = [p.box for p in self.peaks]
            self.draw(start=True)

        self.app.guarded(guessed)

    def _clicked(self, event: Any) -> None:
        if not self.click.value or event.xdata is None:
            return
        _, x, y, _ = self.data()
        height = float(event.ydata) - float(np.nanmin(y))
        width = 0.05 * float(np.ptp(x)) if x.size > 1 else 1.0
        self.add_peak(None, {"center": float(event.xdata), "amplitude": height, "fwhm": width})
        self.draw(start=True)

    def _background_rows(self) -> None:
        self.bg_rows = {n: ParamRow(n) for n in peakfit.BACKGROUNDS[self.background.value]}
        self.bg_box.children = [r.box for r in self.bg_rows.values()]

    def spec(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        peaks = [p.spec() for p in self.peaks]
        background = {n: r.spec() for n, r in self.bg_rows.items() if r.spec() is not None}
        return peaks, background

    # ------------------------------------------------------------------ drawing
    def draw(self, start: bool = False) -> None:
        """The data, and the start model (``start``) or the last fit with its components."""
        try:
            dim, x, y, sigma = self.data()
        except Exception as exc:
            self.results.value = f"<i>{html.escape(str(exc))}</i>"
            return
        fig = self.pane.clear()
        grid = fig.add_gridspec(4, 1, hspace=0.05)
        ax = fig.add_subplot(grid[:3, 0])
        rest = fig.add_subplot(grid[3, 0], sharex=ax)
        if sigma is not None:
            ax.errorbar(x, y, yerr=sigma, fmt="o", ms=3, lw=0.8, color="#424242", label="data")
        else:
            ax.plot(x, y, "o", ms=3, color="#424242", label="data")
        fine = np.linspace(np.nanmin(x), np.nanmax(x), 500)
        summary = self._summary(start)
        if summary is not None:
            parts = peakfit.model_components(summary, fine)
            total = np.sum(list(parts.values()), axis=0)
            ax.plot(fine, total, "-", color="#e53935", lw=1.5, label="start" if start else "fit")
            for key, values in parts.items():  # each peak drawn on top of the background
                shown = values + parts["background"] if key != "background" else values
                ax.plot(fine, shown, "--", lw=0.8, label=key)
            at_points = np.sum(list(peakfit.model_components(summary, x).values()), axis=0)
            rest.plot(x, y - at_points, "o", ms=2, color="#1e88e5")
        rest.axhline(0, color="#9e9e9e", lw=0.8)
        rest.set_xlabel(dim)
        rest.set_ylabel("residual", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
        rng = self.x_range()
        if rng is not None:
            for edge in rng:
                if edge is not None:
                    ax.axvline(edge, color="#fb8c00", lw=0.8)
        self.pane.draw()

    def _summary(self, start: bool) -> dict[str, Any] | None:
        """The model to draw: the start values of the rows, or the last fit of this curve."""
        if not start:
            fit = self.last_fit
            return (
                fit
                if fit is not None and fit.get("target") == (self.target.value, self.row.value)
                else None
            )
        if not self.peaks:
            return None
        try:
            peaks, background = self.spec()
            _, x, y, _ = self.data()
        except ValueError:
            return None
        values: dict[str, float] = {}
        for k, peak in enumerate(peaks, start=1):
            for name in peakfit.SHAPES[peak["shape"]]:
                given = peak.get(name)
                default = {"eta": 0.5, "m": 1.5, "lorentz_fwhm": 0.0}.get(name)
                if given is None and default is None:
                    return None
                values[f"p{k}_{name}"] = given["value"] if given else float(default)
        bg_values = {
            "c0": float(np.nanmin(y)),
            "c1": 0.0,
            "c2": 0.0,
            "a": 0.0,
            "decay": float(np.ptp(x)) or 1.0,
        }
        for name in peakfit.BACKGROUNDS[self.background.value]:
            given = background.get(name)
            values[f"bg_{name}"] = given["value"] if given else bg_values[name]
        return {
            "shapes": [p["shape"] for p in peaks],
            "background": self.background.value,
            "values": values,
            "x_range": [float(np.nanmin(x)), float(np.nanmax(x))],
        }

    # ------------------------------------------------------------------ fitting
    def fit(self) -> None:
        obj = self._object()
        if obj is None:
            self.app.say("choose a curve", "error")
            return
        if obj.kind != "curve":
            self.app.say(
                "a map: Fit every curve fits them all; to fit one, slice it first", "error"
            )
            return
        peaks, background = self.spec()
        values: dict[str, Any] = {
            "background": self.background.value,
            "use_sigma": self.use_sigma.value,
        }
        if peaks:
            values["peaks"] = peaks
        else:
            values.update(n_peaks=int(self.n_guess.value), shape=self.guess_shape.value)
        if background:
            values["background_params"] = background
        if self.x_range() is not None:
            values["x_range"] = self.x_range()
        target = (self.target.value, self.row.value)
        self.app.run_op(
            "fit_peaks", [obj.name], values, then=lambda made: self._fitted(made, target)
        )

    def _fitted(self, made: list[Any], target: tuple[Any, int]) -> None:
        fit_curve = next((o for o in made if o.kind == "curve"), None)
        table = next((o for o in made if o.kind == "table"), None)
        if fit_curve is None or "fit" not in fit_curve.meta:
            return
        summary = dict(fit_curve.meta["fit"])
        summary["target"] = target
        self.last_fit = summary
        if len(self.peaks) == len(summary["shapes"]):
            for k, row in enumerate(self.peaks, start=1):
                prefix = f"p{k}_"
                row.update(
                    {
                        n[len(prefix) :]: v
                        for n, v in summary["values"].items()
                        if n.startswith(prefix)
                    }
                )
        else:  # the peaks were guessed: show them as rows to refine
            self.peaks = []
            for k, shape in enumerate(summary["shapes"], start=1):
                prefix = f"p{k}_"
                self.add_peak(
                    shape,
                    {
                        n[len(prefix) :]: v
                        for n, v in summary["values"].items()
                        if n.startswith(prefix)
                    },
                )
        for name, row in self.bg_rows.items():
            if f"bg_{name}" in summary["values"]:
                row.set(summary["values"][f"bg_{name}"])
        quality = f"r2 = {summary['r2']:.5f}, reduced chi2 = {summary['reduced_chi2']:.4g}"
        rows = (
            table.data.to_html(float_format=lambda v: f"{v:.6g}", index=False)
            if table is not None
            else ""
        )
        self.results.value = (
            f"<b>{html.escape(quality)}</b><div style='overflow:auto;font-size:11px'>{rows}</div>"
        )
        self._table = table.name if table is not None else None
        self.draw()

    def fit_series(self) -> None:
        obj = self._object()
        if obj is None or obj.kind not in ("map", "volume"):
            self.app.say("Fit every curve takes a map (curves along one of its dims)", "error")
            return
        peaks, _ = self.spec()
        values: dict[str, Any] = {"along": self.along.value, "background": self.background.value}
        if peaks:
            values["peaks"] = peaks
        else:
            values.update(n_peaks=int(self.n_guess.value), shape=self.guess_shape.value)
        if self.x_range() is not None:
            values["x_range"] = self.x_range()
        self.app.run_op("fit_series", [obj.name], values)

    def _export(self) -> None:
        if not self._table:
            self.app.say("fit first: the parameter table is saved", "error")
            return
        path = self.export_path.value.strip() or "fit.csv"
        self.app.guarded(
            lambda: self.app.ws.export(self._table, path),
            show=lambda out: self.app.say(f"saved {out}", "ok"),
        )
