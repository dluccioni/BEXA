"""The Analysis tab: every operation of the catalog, by group, with its help and its form.

Pick a group and an operation; its inputs default to the objects selected on
the left; its settings form comes from the catalog, each field with its help.
Run sends the work to the background: the results appear as new objects, the
command goes to the log. "Run on each" repeats the operation for every
selected object (a batch); a scan set given to a scan operation runs it on
every scan and stacks the results.
"""

from __future__ import annotations

import html
from typing import Any

from bexa.gui.forms import Form, object_options, set_tooltip
from bexa.workspace.catalog import GROUPS, OPS, Op, find, for_kind
from bexa.workspace.source import jsonable

__all__ = ["AnalysisTab"]


class AnalysisTab:
    """Operations by group; inputs, settings, Run."""

    def __init__(self, app: Any) -> None:
        import ipywidgets as ipw

        self.app = app
        self.forms: dict[str, Form] = {}
        self.input_widgets: list[Any] = []
        self.group = ipw.ToggleButtons(
            options=GROUPS,
            value=GROUPS[0],
            style={"button_width": "auto"},
            layout=ipw.Layout(width="100%"),
        )
        self.only_fitting = ipw.Checkbox(
            value=False, indent=False, description="for the selection only"
        )
        self.search = ipw.Text(placeholder="search", layout=ipw.Layout(width="200px"))
        self.listing = ipw.Select(options=[], rows=9, layout=ipw.Layout(width="100%"))
        self.listing.add_class("bexa-ops")
        self.group.observe(lambda _c: self._fill(), names="value")
        self.only_fitting.observe(lambda _c: self._fill(), names="value")
        self.search.observe(lambda _c: self._fill(), names="value")
        self.listing.observe(lambda c: self._picked(c["new"]), names="value")
        self.title = ipw.HTML()
        self.help = ipw.HTML()
        self.inputs = ipw.VBox()
        self.form_box = ipw.VBox()
        self.name = ipw.Text(placeholder="result name", layout=ipw.Layout(width="260px"))
        self.each = ipw.Checkbox(
            value=False, indent=False, description="Run on each selected object"
        )
        set_tooltip(self.each, "a batch: one run per selected object")
        self.run_button = ipw.Button(
            description="Run", button_style="success", icon="play", layout=ipw.Layout(width="110px")
        )
        self.run_button.on_click(lambda _b: self._run())
        preview = ipw.Button(description="Show command", layout=ipw.Layout(width="130px"))
        preview.on_click(lambda _b: self._preview())
        self.command = ipw.HTML()
        self.box = ipw.VBox(  # one column: the list above the form of the operation picked
            [
                self.group,
                ipw.HBox([self.search, self.only_fitting], layout=ipw.Layout(flex_flow="row wrap")),
                self.listing,
                self.title,
                self.help,
                ipw.HTML("<b>Inputs</b>"),
                self.inputs,
                ipw.HTML("<b>Settings</b>"),
                self.form_box,
                ipw.HBox(
                    [self.run_button, self.name, preview], layout=ipw.Layout(flex_flow="row wrap")
                ),
                self.each,
                self.command,
            ]
        )
        self._fill()

    # ------------------------------------------------------------------ the list
    def _listed(self) -> list[Op]:
        needle = self.search.value.strip().lower()
        if needle:
            ops = [
                op
                for op in OPS
                if needle in op.label.lower() or needle in op.help.lower() or needle in op.key
            ]
        else:
            ops = [op for op in OPS if op.group == self.group.value]
        if self.only_fitting.value and self.app.current in self.app.ws:
            fitting = {op.key for op in for_kind(self.app.ws[self.app.current].kind)}
            ops = [op for op in ops if op.key in fitting or not op.inputs]
        return ops

    def _fill(self) -> None:
        ops = self._listed()
        current = self.listing.value
        self.listing.options = [
            (f"{op.label}" + (f"   [{op.group}]" if self.search.value else ""), op.key)
            for op in ops
        ]
        keys = [op.key for op in ops]
        if current in keys:
            self.listing.value = current
        elif keys:
            self.listing.value = keys[0]

    def select_op(self, key: str) -> None:
        """Show one operation (from code, or from another tab)."""
        op = find(key)
        self.search.value = ""
        self.group.value = op.group
        self.listing.value = key

    # ------------------------------------------------------------------ one op
    def _picked(self, key: str | None) -> None:
        if not key:
            return
        op = find(key)
        self.title.value = f"<h3 style='margin:4px 0'>{html.escape(op.label)}</h3>"
        self.help.value = (
            f"<div style='max-width:760px;color:#555;font-size:12px'>{html.escape(op.help)}</div>"
        )
        form = self.forms.get(key)
        if form is None:
            form = Form(op, self.app.ws)
            remembered = self.app.ws.settings.get("forms", {}).get(key)
            if remembered:
                form.set_values(remembered)
            self.forms[key] = form
        self.form_box.children = [form.box]
        self._build_inputs(op)
        self.command.value = ""

    def _build_inputs(self, op: Op) -> None:
        import ipywidgets as ipw

        self.input_widgets = []
        rows = []
        style = {"description_width": "190px"}
        selected = list(self.app.selection)
        for slot in op.inputs:
            options = object_options(self.app.ws, slot.kinds)
            names = [v for _, v in options]
            if slot.many:
                widget = ipw.SelectMultiple(
                    options=options,
                    rows=5,
                    description=slot.label,
                    style=style,
                    layout=ipw.Layout(width="96%"),
                )
                widget.value = tuple(n for n in selected if n in names)
            else:
                widget = ipw.Dropdown(
                    options=options or [("no object of this kind yet", None)],
                    description=slot.label,
                    style=style,
                    layout=ipw.Layout(width="96%"),
                )
                taken = [w.value for w in self.input_widgets if not isinstance(w.value, tuple)]
                choice = next((n for n in selected if n in names and n not in taken), None)
                widget.value = choice if choice is not None else (names[0] if names else None)
            kinds = ", ".join(slot.kinds) + (
                " (or a scan set: every scan)" if "scan" in slot.kinds else ""
            )
            set_tooltip(widget, kinds)
            widget.observe(lambda _c: self._first_changed(), names="value")
            self.input_widgets.append(widget)
            rows.append(widget)
        if not op.inputs:
            rows.append(ipw.HTML("<i>none</i>"))
        self.inputs.children = rows
        self.each.layout.display = "" if op.inputs and not op.inputs[0].many else "none"
        self._first_changed()

    def _first_changed(self) -> None:
        key = self.listing.value
        if not key or key not in self.forms:
            return
        first = None
        if self.input_widgets:
            value = self.input_widgets[0].value
            name = value[0] if isinstance(value, tuple) and value else value
            first = self.app.ws.objects.get(name) if isinstance(name, str) else None
        self.forms[key].refresh(first)

    def refresh(self) -> None:
        """The workspace or the selection changed: inputs, ROI and object choices anew."""
        key = self.listing.value
        if self.only_fitting.value:
            self._fill()
        if key:
            op = find(key)
            self._build_inputs(op)

    # ------------------------------------------------------------------ running
    def _gather(self) -> tuple[Op, list[Any], dict[str, Any]]:
        op = find(self.listing.value)
        inputs = []
        for slot, widget in zip(op.inputs, self.input_widgets, strict=True):
            value = list(widget.value) if slot.many else widget.value
            if not value:
                raise ValueError(f"{slot.label}: choose an object ({', '.join(slot.kinds)})")
            inputs.append(value)
        values = self.forms[op.key].values()
        return op, inputs, values

    def _preview(self) -> None:
        try:
            op, inputs, values = self._gather()
        except ValueError as exc:
            self.command.value = f"<span style='color:#b71c1c'>{html.escape(str(exc))}</span>"
            return
        name = self.name.value.strip() or None
        self.command.value = f"<code>{html.escape(op.command(inputs, values, name))}</code>"

    def _run(self) -> None:
        try:
            op, inputs, values = self._gather()
        except ValueError as exc:
            self.app.say(str(exc), "error")
            return
        forms = self.app.ws.settings.setdefault("forms", {})
        forms[op.key] = jsonable(values)
        name = self.name.value.strip() or None
        if self.each.value and op.inputs and not op.inputs[0].many:
            allowed = set(op.inputs[0].kinds) | (
                {"scanset"} if "scan" in op.inputs[0].kinds else set()
            )
            targets = [n for n in self.app.selection if self.app.ws[n].kind in allowed]
            if not targets:
                self.app.say(f"no selected object is a {', '.join(op.inputs[0].kinds)}", "error")
                return
            for target in targets:
                self.app.run_op(op.key, [target, *inputs[1:]], values)
            return
        self.app.run_op(op.key, inputs, values, name)
        self._preview()
