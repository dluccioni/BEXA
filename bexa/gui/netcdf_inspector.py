"""What the Browse tab shows inside a netCDF file: its dims, attributes and variables.

A variable picked shows its dims, shape, type and attributes and a picture of its
values (a curve against its coordinate; a few curves for an array whose last dim
is short, the counters of an APS run; an image otherwise); it loads as an object
with its dims and coordinates, cut by an index while it is read.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["NetCDFInspector"]

MAX_VALUES = 4_000_000  # values read for a picture; a larger variable is shown by its first plane
FEW_CURVES = 16  # a last dim this short is drawn as one curve per value


def _grid(variable: Any) -> str:
    """``nsamstageY[21] x det[3]``: a variable's dims with their lengths."""
    return " x ".join(f"{d}[{n}]" for d, n in zip(variable.dims, variable.shape, strict=False))


class NetCDFInspector:
    """A netCDF file's dims and attributes, its variables, a variable's details and picture."""

    def __init__(self, tab: Any, path: Path) -> None:
        import ipywidgets as ipw

        from bexa.io import netcdf

        self.tab = tab
        self.path = path
        self.info = netcdf.describe(path)
        info = self.info
        dims = ", ".join(f"{d} = {n}" for d, n in info.dims.items()) or "none"
        attrs = "".join(
            f"<tr><td><b>{html.escape(k)}</b></td><td><code>{html.escape(str(v)[:300])}</code></td></tr>"
            for k, v in info.attrs.items()
        )
        self.summary = ipw.HTML(
            f"<div style='font-size:12px'>{info.kind}, {len(info.variables)} variables; "
            f"dims: {html.escape(dims)}</div>"
            + (f"<table style='font-size:11px'>{attrs}</table>" if attrs else "")
        )
        options = [
            (
                f"{v.name}  {_grid(v)}  {v.dtype}{'  (coordinate)' if v.coordinate else ''}",
                v.name,
            )
            for v in info.variables
        ]
        self.variables = ipw.Select(
            options=options, rows=min(10, max(3, len(options))), layout=ipw.Layout(width="100%")
        )
        self.variables.add_class("bexa-nc-variables")
        self.details = ipw.HTML()
        self.index = ipw.Text(
            description="index",
            placeholder="e.g. :, 2 (empty: all)",
            layout=ipw.Layout(width="320px"),
            style={"description_width": "50px"},
        )
        load = ipw.Button(description="Load as object", button_style="primary")
        load.on_click(lambda _b: self._load())
        self.box = ipw.VBox(
            [self.summary, self.variables, self.details, ipw.HBox([self.index, load])]
        )
        self.variables.observe(lambda c: self._picked(c["new"]), names="value")
        first = next((v.name for v in info.variables if not v.coordinate), None)
        if first is not None:  # a Select starts on its first option: show the data at once
            self.variables.value = first
            self._picked(first)

    def close(self) -> None:
        return None

    def _picked(self, name: str | None) -> None:
        if not name:
            return
        variable = self.info.variable(name)
        facts = [
            ("variable", name),
            ("dims", _grid(variable)),
            ("dtype", variable.dtype),
            *((f"@{k}", v) for k, v in variable.attrs.items()),
        ]
        rows = "".join(
            f"<tr><td><b>{html.escape(str(k))}</b></td><td><code>{html.escape(str(v)[:300])}</code></td></tr>"
            for k, v in facts
        )
        self.details.value = f"<table style='font-size:11px'>{rows}</table>"
        self._preview(variable)

    def _preview(self, variable: Any) -> None:
        from bexa.gui.inspectors import array_preview
        from bexa.io import netcdf

        try:
            part = None
            if int(np.prod(variable.shape or (1,))) > MAX_VALUES and len(variable.shape) > 2:
                part = (0,) * (len(variable.shape) - 2)  # the first plane of a large variable
            values = netcdf.read_variable(self.path, variable.name, part)
        except Exception as exc:
            self.tab.show_text(f"cannot read {variable.name}: {type(exc).__name__}: {exc}")
            return
        if values.ndim == 0 or values.dtype.kind not in "biuf":
            self.tab.show_text(f"{variable.name} = {values.values!r}"[:20000])
            return
        self.tab.show_figure()
        if values.ndim == 2 and values.shape[1] <= FEW_CURVES < values.shape[0]:
            self._curves(values)
        elif values.ndim == 1:
            self._curves(values.expand_dims("_", axis=1))
        else:
            array_preview(self.tab.pane, values.values, variable.name)

    def _curves(self, values: Any) -> None:
        """A curve per value of the last dim against the first dim's coordinate."""
        fig = self.tab.pane.clear()
        ax = fig.add_subplot(111)
        first, last = values.dims
        x = values[first].values if first in values.coords else np.arange(values.shape[0])
        labels = values[last].values if last in values.coords else range(values.shape[1])
        for k, label in enumerate(labels):
            ax.plot(x, values.values[:, k], "o-", ms=3, lw=1, label=f"{last} {label}")
        if values.shape[1] > 1:
            ax.legend(fontsize=7)
        ax.set_xlabel(first)
        ax.set_title(str(values.name), fontsize=9)
        self.tab.pane.draw()

    def _load(self) -> None:
        name = self.variables.value
        if not name:
            return
        index = self.index.value.strip() or None
        app = self.tab.app
        app.submit(
            f"loading {name}",
            lambda: app.ws.open_file(str(self.path), dataset=name, index=index),
            done=app.show_results,
        )
