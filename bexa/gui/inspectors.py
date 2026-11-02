"""What the Browse tab shows inside a file: HDF5/NeXus trees, SPEC sessions, TIFF, NPY.

Each inspector reads only what it shows: an HDF5 group's children when it is
opened, one slice of a dataset per slider move, the first rows of a SPEC scan.
A file of hundreds of gigabytes browses as fast as a small one.
"""

from __future__ import annotations

import contextlib
import html
from pathlib import Path
from typing import Any

import numpy as np

from bexa._log import get_logger

log = get_logger(__name__)

__all__ = ["H5Inspector", "SpecInspector", "array_preview", "attrs_table", "tiff_summary"]

MAX_LINES = 3000  # tree lines listed at once
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
        self.box = ipw.VBox([self.tree, self.info, self.slider_box, self.actions])
        self._fill()

    def close(self) -> None:
        with contextlib.suppress(Exception):  # a file already closed, or gone
            self.file.close()

    # ------------------------------------------------------------------ the tree
    def _lines(self) -> list[tuple[str, str]]:
        import h5py

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
            if path != "/":
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
    """A SPEC session: its scans, a scan's header and columns, a column plotted against another."""

    def __init__(self, tab: Any, path: Path) -> None:
        import ipywidgets as ipw

        from bexa.io.engines.spec_tiff import read_spec_file

        self.tab = tab
        self.path = path
        self.spec = read_spec_file(path)
        options = [
            (f"#S {n:4d}  {s.n_rows:5d} pts  {s.command}"[:110], n)
            for n, s in self.spec.scans.items()
        ]
        self.scans = ipw.Select(options=options, rows=14, layout=ipw.Layout(width="100%"))
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
        self.box = ipw.VBox(
            [self.info, self.scans, ipw.HBox([self.x, self.y, open_button]), self.details]
        )
        if options:
            self.scans.value = options[0][1]

    def close(self) -> None:
        return None

    def _scan(self, number: int | None) -> None:
        if number is None:
            return
        scan = self.spec.scans[number]
        labels = list(scan.labels)
        self.x.options, self.y.options = labels, labels
        if labels:
            self.x.value = labels[0]
            self.y.value = labels[-1]
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
        self._plot()

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
