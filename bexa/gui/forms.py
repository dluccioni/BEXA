"""Forms made from the catalog: one field per setting, its help in the tooltip, values checked.

A :class:`Form` turns an :class:`bexa.workspace.catalog.Op` into widgets: text
fields for numbers (empty means "not given" for optional ones), checkboxes,
drop-downs for choices, objects, ROIs, dims and channels (refreshed from the
workspace and the first input), two fields for a range. :meth:`Form.values`
parses them with the catalog's own rules, so the GUI and ``ws.run`` agree.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from typing import Any

from bexa.workspace.catalog import Op, P
from bexa.workspace.objects import DataObject

__all__ = ["Form", "kind_label", "object_options", "set_tooltip"]

LABEL_WIDTH = "190px"
PIXEL_NAMES = ("y", "x", "yb", "xb")
KIND_TAGS = {
    "scan": "scan",
    "scanset": "set",
    "stack": "stack",
    "image": "image",
    "map": "map",
    "curve": "curve",
    "volume": "volume",
    "rgb": "rgb",
    "table": "table",
    "text": "text",
}


def kind_label(obj: DataObject) -> str:
    """``[image] name  y[512] x x[1028]``: an object as a line of a list."""
    shape = (
        " x ".join(f"{d}[{n}]" for d, n in zip(obj.dims, obj.shape, strict=False))
        if obj.dims
        else ""
    )
    if obj.kind == "scanset":
        shape = f"{len(obj.data)} scans"
    elif obj.kind == "table":
        shape = f"{obj.shape[0]} rows"
    return f"[{KIND_TAGS.get(obj.kind, obj.kind)}] {obj.name}  {shape}".rstrip()


def set_tooltip(widget: Any, text: str) -> None:
    """The text shown when the mouse rests on a widget (``tooltip`` in ipywidgets 8,
    ``description_tooltip`` in 7). A box passes it to its parts: the box views of
    ipywidgets 8 fail to draw a box with a tooltip of its own (they read its description)."""
    if "description" not in widget.keys and getattr(widget, "children", None) is not None:
        for child in widget.children:
            set_tooltip(child, text)
        return
    if "tooltip" in widget.keys:
        widget.tooltip = text
    elif "description_tooltip" in widget.keys:
        widget.description_tooltip = text


def object_options(
    ws: Any, kinds: tuple[str, ...], blank: str | None = None
) -> list[tuple[str, Any]]:
    """Drop-down options for the objects of ``kinds`` (a scan set where scans are taken)."""
    allowed = set(kinds) | ({"scanset"} if "scan" in kinds else set())
    options: list[tuple[str, Any]] = [(blank, None)] if blank is not None else []
    options += [(kind_label(o), o.name) for o in ws.objects.values() if o.kind in allowed]
    return options


def _first_channels(obj: DataObject | None) -> list[str]:
    if obj is None:
        return []
    scan = obj.data.scans[0] if obj.kind == "scanset" else obj.data if obj.kind == "scan" else None
    if scan is None:
        return []
    try:
        return sorted(scan.structure.per_frame)
    except Exception:
        return []


def _first_dims(obj: DataObject | None, pixels: bool = False) -> list[str]:
    """The dims of the first input; a scan's motors, with its pixel dims when ``pixels``."""
    if obj is None:
        return []
    if obj.kind in ("scan", "scanset"):
        dims = (obj.data.scans[0] if obj.kind == "scanset" else obj.data).dims
        return [str(d) for d in (dims if pixels else dims[:-2])]
    return list(obj.dims)


class DimToggles:
    """Any number of dims chosen with a toggle button each: the field of a ``dims`` setting.

    ``options`` and ``value`` (a tuple) behave as those of a SelectMultiple, which
    needs Ctrl+click to choose several; ``box`` is the widget shown.
    """

    def __init__(self, label: str) -> None:
        import ipywidgets as ipw

        self._options: list[tuple[str, Any]] = []
        self.buttons: list[Any] = []
        self.row = ipw.HBox(layout=ipw.Layout(flex_flow="row wrap"))
        title = ipw.HTML(
            f"<div style='width:{LABEL_WIDTH};text-align:right;padding-right:6px'>"
            f"{html.escape(label)}</div>"
        )
        self.box = ipw.HBox([title, self.row])

    @property
    def options(self) -> list[tuple[str, Any]]:
        return list(self._options)

    @options.setter
    def options(self, options: Sequence[Any]) -> None:
        import ipywidgets as ipw

        chosen = set(self.value)
        for button in self.buttons:  # ipywidgets keeps a widget until it is closed
            button.close()
        self._options = [o if isinstance(o, tuple) else (str(o), o) for o in options]
        self.buttons = [
            ipw.ToggleButton(description=label, value=v in chosen, layout=ipw.Layout(width="auto"))
            for label, v in self._options
        ]
        self.row.children = self.buttons or [ipw.HTML("<i>no dims</i>")]

    @property
    def value(self) -> tuple[Any, ...]:
        return tuple(v for (_, v), b in zip(self._options, self.buttons, strict=True) if b.value)

    @value.setter
    def value(self, values: Sequence[Any]) -> None:
        wanted = set(values)
        for (_, v), button in zip(self._options, self.buttons, strict=True):
            button.value = v in wanted


class Form:
    """The widgets of one operation's settings."""

    def __init__(self, op: Op, ws: Any) -> None:
        import ipywidgets as ipw

        self.op = op
        self.ws = ws
        self.first: DataObject | None = None
        self.fields: dict[str, Any] = {}
        rows = []
        style = {"description_width": LABEL_WIDTH}
        wide = ipw.Layout(width="96%")
        for p in op.params:
            if p.kind == "spec":
                continue
            field = self._widget(p, style, wide)
            self.fields[p.name] = field
            rows.append(getattr(field, "box", field))  # DimToggles: its box
        if not rows:
            rows.append(ipw.HTML("<i>no settings</i>"))
        self.box = ipw.VBox(rows, layout=ipw.Layout(width="100%"))
        self.refresh()

    def _widget(self, p: P, style: dict[str, str], layout: Any) -> Any:
        widget = self._make(p, style, layout)
        # the help shows when the mouse rests on it
        set_tooltip(getattr(widget, "box", widget), p.help or p.label)
        return widget

    @staticmethod
    def _make(p: P, style: dict[str, str], layout: Any) -> Any:
        import ipywidgets as ipw

        label = p.label
        common = {"description": label, "style": style, "layout": layout}
        if p.kind == "bool":
            return ipw.Checkbox(value=bool(p.default), indent=True, **common)
        if p.kind == "choice":
            return ipw.Dropdown(
                options=list(p.choices),
                value=p.default if p.default in p.choices else None,
                **common,
            )
        if p.kind in ("object", "roi", "dim", "channel"):
            return ipw.Dropdown(options=[("none", None)], value=None, **common)
        if p.kind == "dims":
            return DimToggles(label)
        if p.kind in ("objects", "rois", "channels"):
            return ipw.SelectMultiple(options=[], rows=4, **common)
        if p.kind == "range":
            low = ipw.Text(placeholder="from", layout=ipw.Layout(width="120px"))
            high = ipw.Text(placeholder="to", layout=ipw.Layout(width="120px"))
            label_html = ipw.HTML(
                f"<div style='width:{LABEL_WIDTH};text-align:right;padding-right:6px'>"
                f"{html.escape(label)}</div>"
            )
            box = ipw.HBox([label_html, low, high])
            box.low, box.high = low, high  # type: ignore[attr-defined]
            return box
        placeholder = {"numbers": "1, 2.5, 4", "ints": "0-3, 7"}.get(p.kind, "")
        if p.optional and not placeholder:
            placeholder = "optional"
        return ipw.Text(value=p.text(p.default), placeholder=placeholder, **common)

    # ------------------------------------------------------------------ choices
    def refresh(self, first: DataObject | None = None) -> None:
        """Fill the drop-downs from the workspace and the first input (its dims, channels)."""
        if first is not None:
            self.first = first
        for p in self.op.params:
            widget = self.fields.get(p.name)
            if widget is None:
                continue
            if p.kind in ("object", "objects"):
                options = object_options(self.ws, p.kinds, "none" if p.kind == "object" else None)
            elif p.kind in ("roi", "rois"):
                names = list(self.ws.rois)
                options = ([("none", None)] if p.kind == "roi" else []) + [(n, n) for n in names]
            elif p.kind in ("dim", "dims"):
                dims = _first_dims(self.first, p.pixels)
                options = ([("", None)] if p.kind == "dim" and p.optional else []) + [
                    (d, d) for d in dims
                ]
            elif p.kind in ("channel", "channels"):
                channels = _first_channels(self.first)
                options = ([("none", None)] if p.kind == "channel" else []) + [
                    (c, c) for c in channels
                ]
            else:
                continue
            self._set_options(widget, options, p)

    @staticmethod
    def _set_options(widget: Any, options: list[tuple[str, Any]], p: P) -> None:
        values = [v for _, v in options]
        current = widget.value
        widget.options = options
        if isinstance(current, tuple):  # a multiple selection keeps what is still there
            widget.value = tuple(v for v in current if v in values)
        elif current in values:
            widget.value = current
        elif p.kind == "dim" and not p.optional and values:
            # the last dim that is not a pixel: a stack's scanned motor, a map's x
            motors = [v for v in values if v not in PIXEL_NAMES]
            widget.value = motors[-1] if motors else values[-1]
        elif values:
            widget.value = values[0]

    # ------------------------------------------------------------------ values
    def values(self) -> dict[str, Any]:
        """The parsed values, the defaults of the hidden ones left out (``ValueError`` names
        the field at fault)."""
        out: dict[str, Any] = {}
        for p in self.op.params:
            widget = self.fields.get(p.name)
            if widget is None:
                continue
            if p.kind == "range":
                raw: Any = (widget.low.value, widget.high.value)
            elif p.kind in ("objects", "rois", "dims", "channels"):
                raw = list(widget.value)
            else:
                raw = widget.value
            value = p.parse(raw)
            if p.kind in ("objects", "rois", "dims", "channels") and not value:
                value = None
            if value is None and not p.optional and p.kind != "bool":
                raise ValueError(f"{p.label}: needed")
            if value != p.default:  # the command then shows what was changed
                out[p.name] = value
        return out

    def set_values(self, values: Mapping[str, Any]) -> None:
        """Put remembered values back into the fields (unknown or stale ones are skipped)."""
        for name, value in values.items():
            widget = self.fields.get(name)
            if widget is None:
                continue
            p = self.op.param(name)
            try:
                if p.kind == "range":
                    low, high = value if value else (None, None)
                    widget.low.value, widget.high.value = p.text(low), p.text(high)
                elif p.kind in ("objects", "rois", "dims", "channels"):
                    allowed = [o[1] if isinstance(o, tuple) else o for o in widget.options]
                    widget.value = tuple(v for v in (value or []) if v in allowed)
                elif p.kind in ("bool", "choice", "object", "roi", "dim", "channel"):
                    widget.value = value
                else:
                    widget.value = p.text(value)
            except Exception:  # a stale value (an object gone, a choice renamed)
                continue
