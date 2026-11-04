"""What the Browse tab shows inside a file: HDF5/NeXus trees, SPEC sessions, TIFF, NPY.

Each inspector reads only what it shows: an HDF5 group's children when it is
opened, one slice of a dataset per slider move, the first rows of a SPEC scan.
A file of hundreds of gigabytes browses as fast as a small one. Both help to find
the variables of interest: an HDF5 file is searched by name (a motor's value shown
in every scan that has it), a SPEC session lists its motors (where each was, which
moved), counters, columns and header fields, filtered by name.
"""

from __future__ import annotations

import contextlib
import html
from pathlib import Path
from typing import Any

import numpy as np

from bexa._log import get_logger

log = get_logger(__name__)

__all__ = [
    "H5Inspector",
    "SpecInspector",
    "array_preview",
    "attrs_table",
    "spec_variables",
    "tiff_summary",
]

MAX_LINES = 3000  # tree lines listed at once
MAX_FOUND = 1000  # names found in an HDF5 file, listed at once
MAX_VARIABLES = 300  # rows of a SPEC session's variables, shown at once
PREVIEW_PIXELS = 1024  # an image preview is cut to about this many pixels a side


def attrs_table(attrs: Any, limit: int = 60) -> str:
    """The attributes of an HDF5 node as an HTML table."""
    rows = []
    for k, (key, value) in enumerate(attrs.items()):
        if k >= limit:
            rows.append(f"<tr><td colspan=2>... {len(attrs) - limit} more</td></tr>")
            break
        if isinstance(value, bytes):
            value = value.decode(errors="replace")
        text = str(value)
        if len(text) > 300:
            text = text[:300] + " ..."
        rows.append(
            f"<tr><td style='padding-right:8px'><b>@{html.escape(str(key))}</b></td>"
            f"<td><code>{html.escape(text)}</code></td></tr>"
        )
    return (
        f"<table style='font-size:11px'>{''.join(rows)}</table>" if rows else "<i>no attributes</i>"
    )


def array_preview(pane: Any, values: np.ndarray, title: str = "") -> None:
    """Draw a 1-D array as a curve, a 2-D one as an image, into a :class:`FigurePane`."""
    fig = pane.clear()
    ax = fig.add_subplot(111)
    data = np.asarray(values)
    if data.dtype.kind not in "biufc":
        ax.text(0.5, 0.5, f"{data.dtype} values: no picture", ha="center", va="center")
        ax.set_axis_off()
    elif data.ndim <= 1:
        ax.plot(np.atleast_1d(data).real, lw=1)
        ax.set_xlabel("index")
    else:
        image = data.reshape(-1, *data.shape[-2:])[0] if data.ndim > 2 else data
        image = np.asarray(image, dtype=float)
        finite = image[np.isfinite(image)]
        lo, hi = np.percentile(finite, [1, 99.5]) if finite.size else (0, 1)
        shown = ax.imshow(
            image,
            cmap="viridis",
            vmin=lo,
            vmax=hi if hi > lo else lo + 1,
            interpolation="nearest",
            aspect="auto" if min(image.shape) < 8 else "equal",
        )
        fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.02)
    ax.set_title(title, fontsize=9)
    pane.draw()


def _value(node: Any) -> str:
    """`` = value`` of a dataset holding one value (a motor's position), else nothing."""
    if node.size != 1 or node.dtype.kind not in "biufSUO":
        return ""
    try:
        value = node[()] if node.shape == () else node[(0,) * node.ndim]
    except Exception:
        return ""
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    if isinstance(value, (int, float, np.number)):
        return f" = {float(value):.6g}"
    return f" = {str(value)[:40]}"


def _stride(shape: tuple[int, ...]) -> tuple[slice, ...]:
    """A strided slice that keeps about :data:`PREVIEW_PIXELS` per image axis."""
    return tuple(slice(None, None, max(1, n // PREVIEW_PIXELS)) for n in shape)


class H5Inspector:
    """An HDF5 file as an expandable tree; a dataset's details, attributes and a preview."""

    def __init__(self, tab: Any, path: Path) -> None:
        import ipywidgets as ipw

        from bexa.io.h5files import open_h5

        self.tab = tab
        self.path = path
        self.file = open_h5(path)
        self.expanded: set[str] = {"/"}
        self.selected: str | None = None
        self.sliders: list[Any] = []
        self.find = ipw.Text(
            placeholder="find a name in the file: a motor, a counter, a dataset (Enter)",
            continuous_update=False,
            layout=ipw.Layout(width="100%"),
        )
        self.find.observe(lambda _c: self._fill(), names="value")
        self.tree = ipw.Select(options=[], rows=18, layout=ipw.Layout(width="100%"))
        self.tree.add_class("bexa-h5tree")
        self.tree.observe(self._picked, names="value")
        self.info = ipw.HTML()
        self.slider_box = ipw.VBox()
        self.index = ipw.Text(
            description="index",
            placeholder="e.g. 0, :, 100:200 (empty: all)",
            layout=ipw.Layout(width="320px"),
            style={"description_width": "50px"},
        )
        load = ipw.Button(description="Load as object", button_style="primary")
        convert = ipw.Button(description="Convert this dataset...")
        load.on_click(lambda _b: self._load())
        convert.on_click(lambda _b: self._convert())
        self.actions = ipw.HBox([self.index, load, convert], layout=ipw.Layout(display="none"))
        self.box = ipw.VBox([self.find, self.tree, self.info, self.slider_box, self.actions])
        self._fill()

    def close(self) -> None:
        with contextlib.suppress(Exception):  # a file already closed, or gone
            self.file.close()

    # ------------------------------------------------------------------ the tree
    def _found(self, needle: str) -> list[tuple[str, str]]:
        """Every group and dataset whose path holds ``needle``, a scalar with its value."""
        import h5py

        found: list[tuple[str, str]] = []

        def visit(name: str, node: Any) -> bool | None:
            if len(found) >= MAX_FOUND:
                return True  # stops the walk
            if needle in name.lower():
                where = f"/{name}"
                if isinstance(node, h5py.Group):
                    found.append((f"{where}/  ({len(node)})", where))
                else:
                    found.append((f"{where}  {node.shape} {node.dtype}{_value(node)}", where))
            return None

        self.file.visititems(visit)  # links to other files are not followed
        more = " (the first ones)" if len(found) >= MAX_FOUND else ""
        return [(f"{len(found)} found for {needle!r}{more}", ""), *found]

    def _lines(self) -> list[tuple[str, str]]:
        import h5py

        needle = self.find.value.strip().lower()
        if needle:
            return self._found(needle)
        lines: list[tuple[str, str]] = []

        def walk(group: Any, prefix: str, depth: int) -> None:
            try:
                names = list(group.keys())
            except Exception as exc:
                lines.append((f"{'  ' * depth}(cannot list: {exc})", ""))
                return
            for name in names:
                if len(lines) >= MAX_LINES:
                    return
                path = f"{prefix}/{name}" if prefix != "/" else f"/{name}"
                link = group.get(name, getlink=True)
                if isinstance(link, (h5py.ExternalLink, h5py.SoftLink)):
                    target = getattr(link, "filename", "") + ":" + link.path
                    try:
                        node = group[name]
                    except Exception:
                        lines.append((f"{'  ' * depth}  {name}  -> {target} (broken link)", path))
                        continue
                else:
                    node = group[name]
                if isinstance(node, h5py.Group):
                    mark = "▾" if path in self.expanded else "▸"
                    lines.append((f"{'  ' * depth}{mark} {name}/  ({len(node)})", path))
                    if path in self.expanded:
                        walk(node, path, depth + 1)
                else:
                    lines.append((f"{'  ' * depth}  {name}  {node.shape} {node.dtype}", path))

        lines.append((f"▾ {self.path.name}", "/"))
        walk(self.file, "/", 1)
        if len(lines) >= MAX_LINES:
            lines.append((f"... the first {MAX_LINES} lines; close groups to see others", ""))
        return lines

    def _fill(self) -> None:
        options = self._lines()
        self.tree.unobserve(self._picked, names="value")
        self.tree.options = options
        if self.selected in {v for _, v in options}:
            self.tree.value = self.selected
        self.tree.observe(self._picked, names="value")

    def _picked(self, change: Any) -> None:
        import h5py

        path = change["new"]
        if not path:
            return
        self.selected = path
        try:
            node = self.file[path]
        except Exception as exc:
            self.info.value = f"<span style='color:#b71c1c'>{html.escape(str(exc))}</span>"
            return
        if isinstance(node, h5py.Group):
            if self.find.value.strip():  # a group found: the tree, opened down to it
                parts = path.strip("/").split("/")
                self.expanded |= {"/" + "/".join(parts[: k + 1]) for k in range(len(parts))}
                self.find.value = ""  # the tree again (filled by the observer)
            elif path != "/":
                self.expanded ^= {path}
                self._fill()
            nx = node.attrs.get("NX_class", b"")
            nx = nx.decode() if isinstance(nx, bytes) else str(nx)
            self.info.value = (
                f"<b>{html.escape(path)}</b> group, {len(node)} members"
                f"{' (' + html.escape(nx) + ')' if nx else ''}" + attrs_table(node.attrs)
            )
            self.actions.layout.display = "none"
            self.slider_box.children = []
            return
        self._dataset(node)

    # ------------------------------------------------------------------ a dataset
    def _dataset(self, node: Any) -> None:
        import ipywidgets as ipw

        from bexa.io.inspect import _filters

        nbytes = (
            int(np.prod(node.shape)) * node.dtype.itemsize if node.shape else node.dtype.itemsize
        )
        try:
            stored = node.id.get_storage_size()
        except Exception:
            stored = 0
        facts = [
            ("path", node.name),
            ("shape", str(node.shape)),
            ("dtype", str(node.dtype)),
            (
                "size",
                f"{nbytes / 1e6:.3f} MB"
                + (
                    f", stored {stored / 1e6:.3f} MB (ratio {nbytes / stored:.1f}x)"
                    if stored
                    else ""
                ),
            ),
            ("chunks", str(node.chunks)),
            ("filters", _filters(node)),
        ]
        if node.is_virtual:
            facts.append(("virtual", f"{len(node.virtual_sources())} source mappings"))
        rows = "".join(
            f"<tr><td><b>{k}</b></td><td><code>{html.escape(str(v))}</code></td></tr>"
            for k, v in facts
        )
        self.info.value = f"<table style='font-size:12px'>{rows}</table>" + attrs_table(node.attrs)
        self.actions.layout.display = ""
        self.sliders = []
        if node.ndim > 2 and node.dtype.kind in "biuf":
            for axis, size in enumerate(node.shape[:-2]):
                slider = ipw.IntSlider(
                    min=0,
                    max=max(size - 1, 0),
                    description=f"axis {axis}",
                    continuous_update=False,
                    layout=ipw.Layout(width="70%"),
                )
                slider.observe(lambda _c: self._preview(node), names="value")
                self.sliders.append(slider)
        self.slider_box.children = self.sliders
        self._preview(node)

    def _preview(self, node: Any) -> None:
        pane = self.tab.pane
        try:
            if node.shape == ():
                value = node[()]
                if isinstance(value, bytes):
                    value = value.decode(errors="replace")
                self.tab.show_text(f"{node.name} = {value!r}"[:20000])
                return
            if node.dtype.kind in "SUO":
                values = node[: min(node.shape[0], 200)] if node.ndim else node[()]
                text = "\n".join(
                    v.decode(errors="replace") if isinstance(v, bytes) else str(v)
                    for v in np.atleast_1d(values).ravel()[:200]
                )
                self.tab.show_text(text)
                return
            if node.dtype.names:  # a compound dataset: its first rows as a table
                import pandas as pd

                self.tab.show_html(pd.DataFrame(node[: min(node.shape[0], 50)]).to_html())
                return
            if node.ndim == 1:
                step = max(1, node.shape[0] // 200000)
                values = node[::step]
                title = f"{node.name}" + (f" (every {step}th)" if step > 1 else "")
            else:
                index = tuple(s.value for s in self.sliders)
                values = (
                    node[(*index, *_stride(node.shape[-2:]))]
                    if node.ndim > 2
                    else node[_stride(node.shape)]
                )
                where = ", ".join(str(i) for i in index)
                title = f"{node.name}[{where + ', ' if where else ''}:, :]"
                self.index.placeholder = f"{where + ', ' if where else ''}:, :"
            self.tab.show_figure()
            array_preview(pane, values, title)
        except Exception as exc:
            self.tab.show_text(f"cannot preview {node.name}: {exc}")

    def _load(self) -> None:
        if not self.selected:
            return
        index = self.index.value.strip() or None
        dataset = self.selected
        self.tab.app.submit(
            f"loading {dataset}",
            lambda: self.tab.app.ws.open_file(str(self.path), dataset=dataset, index=index),
            done=self.tab.app.show_results,
        )

    def _convert(self) -> None:
        if self.selected:
            self.tab.app.convert.use_dataset(self.path, self.selected, self.index.value.strip())


class SpecInspector:
    """A SPEC session: its scans; its variables, found by name (every motor with where it was
    in the scan picked and over the session, which moved first; the counters, the scan's
    columns and header fields); a scan's header and columns, a column against another.
    ``keep``: what was picked before (:meth:`state`), picked again where it still is."""

    def __init__(self, tab: Any, path: Path, keep: dict[str, Any] | None = None) -> None:
        import ipywidgets as ipw

        from bexa.io.engines.spec_tiff import read_spec_file

        self.tab = tab
        self.path = path
        self.spec = read_spec_file(path)
        options = [
            (f"#S {n:4d}  {s.n_rows:5d} pts  {s.command}"[:110], n)
            for n, s in self.spec.scans.items()
        ]
        self.scans = ipw.Select(options=options, rows=8, layout=ipw.Layout(width="100%"))
        self.scans.add_class("bexa-spec")
        self.scans.observe(lambda c: self._scan(c["new"]), names="value")
        self.x = ipw.Dropdown(description="x", layout=ipw.Layout(width="260px"))
        self.y = ipw.Dropdown(description="y", layout=ipw.Layout(width="260px"))
        self.x.observe(lambda _c: self._plot(), names="value")
        self.y.observe(lambda _c: self._plot(), names="value")
        open_button = ipw.Button(description="Open this scan", button_style="primary")
        open_button.on_click(lambda _b: self._open())
        self.info = ipw.HTML(
            f"<b>{html.escape(path.name)}</b>: {len(self.spec.scans)} scans, "
            f"{len(self.spec.motor_names)} motors"
        )
        self.details = ipw.HTML()
        self.moves = _motor_moves(self.spec)
        self.find = ipw.Text(
            placeholder="find a motor, counter, column or header field",
            layout=ipw.Layout(width="100%"),
        )
        self.find.observe(lambda _c: self._variables(), names="value")
        self.variables = ipw.HTML()
        self.box = ipw.VBox(
            [
                self.info,
                self.scans,
                self.find,
                self.variables,
                ipw.HBox([self.x, self.y, open_button], layout=ipw.Layout(flex_flow="row wrap")),
                self.details,
            ]
        )
        self._keep = dict(keep or {})  # the columns to plot again, for the first scan shown
        self.find.value = str(self._keep.get("find", ""))
        numbers = [n for _, n in options]
        if numbers:  # the scan picked before, else the first
            number = self._keep["scan"] if self._keep.get("scan") in numbers else numbers[0]
            if self.scans.value != number:
                self.scans.value = number  # shown as any scan picked
            else:  # a Select starts on its first option: no change says so
                self._scan(number)

    def state(self) -> dict[str, Any]:
        """What is picked: the scan, the columns plotted, the search."""
        return {
            "scan": self.scans.value,
            "x": self.x.value,
            "y": self.y.value,
            "find": self.find.value,
        }

    def close(self) -> None:
        return None

    def _scan(self, number: int | None) -> None:
        if number is None:
            return
        keep, self._keep = self._keep, {}
        scan = self.spec.scans[number]
        labels = list(scan.labels)
        self.x.options, self.y.options = labels, labels
        if labels:
            self.x.value = keep["x"] if keep.get("x") in labels else labels[0]
            self.y.value = keep["y"] if keep.get("y") in labels else labels[-1]
        header = "".join(
            f"<tr><td><b>{html.escape(block)}</b></td><td>{html.escape(str(values)[:200])}</td></tr>"
            for block, values in list(scan.header.items())[:12]
        )
        rows = ""
        if scan.n_rows:
            import pandas as pd

            table = pd.DataFrame(scan.rows[:20], columns=labels[: scan.rows.shape[1]])
            rows = table.to_html(max_cols=14, float_format=lambda v: f"{v:.5g}")
        self.details.value = (
            f"<b>#S {number}</b> {html.escape(scan.command)}<br>{html.escape(scan.date)}"
            f"{' (aborted)' if scan.aborted else ''}<table style='font-size:11px'>{header}</table>"
            f"<div style='overflow:auto;max-height:240px;font-size:11px'>{rows}</div>"
        )
        self._variables()
        self._plot()

    def _variables(self) -> None:
        """The table of variables, for the scan picked and the name typed."""
        number = self.scans.value
        scan = self.spec.scans.get(number) if number is not None else None
        needle = self.find.value.strip().lower()
        rows = [
            r
            for r in spec_variables(self.spec, scan, self.moves)
            if needle in r[1].lower() + r[2].lower()
        ]
        cells = "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
            for row in rows[:MAX_VARIABLES]
        )
        more = f" (the first {MAX_VARIABLES})" if len(rows) > MAX_VARIABLES else ""
        head = "".join(
            f"<th>{h}</th>"
            for h in ("kind", "name", "label", f"in #S {number}", "session", "moved in")
        )
        count = f"{len(rows)} variable{'' if len(rows) == 1 else 's'}{more}"
        self.variables.value = (
            f"<div style='font-size:11px'>{count}</div>"
            f"<div style='overflow:auto;max-height:260px;font-size:11px'>"
            f"<table style='border-spacing:8px 1px'><tr>{head}</tr>{cells}</table></div>"
        )

    def _plot(self) -> None:
        number = self.scans.value
        if number is None or not self.x.value or not self.y.value:
            return
        scan = self.spec.scans[number]
        fig = self.tab.pane.clear()
        ax = fig.add_subplot(111)
        if scan.n_rows:
            ax.plot(scan.column(self.x.value), scan.column(self.y.value), "o-", ms=3, lw=1)
        ax.set_xlabel(self.x.value)
        ax.set_ylabel(self.y.value)
        ax.set_title(f"#S {number}", fontsize=9)
        self.tab.show_figure()
        self.tab.pane.draw()

    def _open(self) -> None:
        number = self.scans.value
        if number is None:
            return
        self.tab.app.submit(
            f"opening scan {number}",
            lambda: [self.tab.app.ws.open_scan(str(self.path), scan=int(number))],
            done=self.tab.app.show_results,
        )


def _motor_moves(spec: Any) -> dict[str, tuple[float, float, int]]:
    """Every motor's lowest and highest position over the session's scans, and in how many
    scans it stood elsewhere than in the scan before."""
    scans = [spec.scans[n] for n in sorted(spec.scans)]
    out = {}
    for name in spec.motor_names:
        values = np.array([s.positions.get(name, np.nan) for s in scans], dtype=float)
        values = values[np.isfinite(values)]
        if not values.size:
            out[name] = (np.nan, np.nan, 0)
            continue
        tolerance = 1e-9 * max(1.0, float(np.abs(values).max()))
        moved = int((np.abs(np.diff(values)) > tolerance).sum())
        out[name] = (float(values.min()), float(values.max()), moved)
    return out


def spec_variables(
    spec: Any, scan: Any, moves: dict[str, tuple[float, float, int]] | None = None
) -> list[tuple[str, str, str, str, str, str]]:
    """The variables of a SPEC session as rows ``(kind, name, label, in the scan, over the
    session, moved in)``: its motors (the ones that moved first), counters, the columns of
    ``scan`` and its header fields. ``moves`` is :func:`_motor_moves` (made when missing)."""
    moves = moves if moves is not None else _motor_moves(spec)
    labels = dict(zip(spec.motor_names, spec.motor_labels, strict=False))
    total = len(spec.scans)
    motors = []
    for name in spec.motor_names:
        low, high, moved = moves.get(name, (np.nan, np.nan, 0))
        here = scan.positions.get(name) if scan is not None else None
        motors.append(
            (
                "motor",
                name,
                labels.get(name, ""),
                "" if here is None else f"{here:.6g}",
                "" if np.isnan(low) else f"{low:.6g} .. {high:.6g}",
                f"{moved} of {total} scans",
            )
        )
    motors.sort(key=lambda row: -int(row[5].split()[0]))  # the ones that moved first
    rows = list(motors)
    for name, label in zip(spec.counter_names, spec.counter_labels, strict=False):
        rows.append(("counter", name, label, "", "", ""))
    if scan is not None:
        for k, label in enumerate(scan.labels):
            column = scan.rows[:, k] if scan.n_rows and k < scan.rows.shape[1] else None
            span = "" if column is None else f"{np.nanmin(column):.6g} .. {np.nanmax(column):.6g}"
            rows.append(("column", label, "", span, "", ""))
        for block, fields in scan.header.items():
            for field_name, value in dict(fields).items():
                rows.append(("header", f"{block} {field_name}", "", str(value)[:40], "", ""))
    return rows


def tiff_summary(path: Path, page: int = 0) -> tuple[str, np.ndarray | None, int]:
    """The tags of a TIFF page as HTML, its image and the number of pages."""
    import tifffile

    with tifffile.TiffFile(path) as tif:
        pages = len(tif.pages)
        first = tif.pages[min(page, pages - 1)]
        rows = []
        for tag in list(first.tags.values())[:60]:
            value = str(tag.value)
            rows.append(
                f"<tr><td><b>{html.escape(str(tag.name))}</b> ({tag.code})</td>"
                f"<td><code>{html.escape(value[:160])}</code></td></tr>"
            )
        image = first.asarray()
    head = f"<b>{html.escape(path.name)}</b>: {pages} page(s), {image.shape} {image.dtype}"
    return head + f"<table style='font-size:11px'>{''.join(rows)}</table>", image, pages
