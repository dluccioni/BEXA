"""A scan's metadata on screen: every table of :func:`bexa.core.metadata.scan_metadata`, folded
by title and searched by name or value at once.

The Browse tab shows it under a file bexa reads as a scan (the run's netCDF file,
a SPEC scan, an HDF5 scan) and for the scan selected in the objects list.
"""

from __future__ import annotations

import html

import pandas as pd

__all__ = ["MetadataView", "tables_html"]

MAX_ROWS = 400  # rows of one table shown at once
OPEN = ("scan", "dims", "per point")  # the tables shown unfolded
STYLE = (
    "<style>.bexa-meta table{border-collapse:collapse;margin:2px 0 6px 0}"
    ".bexa-meta th,.bexa-meta td{padding:1px 8px;text-align:left;vertical-align:top}"
    ".bexa-meta tr:nth-child(even){background:rgba(127,127,127,0.08)}"
    ".bexa-meta summary{cursor:pointer;padding:2px 0}</style>"
)


def _matching(table: pd.DataFrame, needle: str) -> pd.DataFrame:
    """The rows of ``table`` with ``needle`` in any cell (or in a column's name: all rows)."""
    if not needle or any(needle in str(c).lower() for c in table.columns):
        return table
    text = table.astype(str).apply(
        lambda column: column.str.lower().str.contains(needle, regex=False)
    )
    return table[text.any(axis=1)]


def tables_html(tables: dict[str, pd.DataFrame], needle: str = "") -> str:
    """The tables as foldable HTML sections; with ``needle``, only the rows that hold it (every
    section with any unfolded, the others left out)."""
    needle = needle.strip().lower()
    parts = []
    for title, table in tables.items():
        shown = _matching(table, needle) if needle and needle not in title.lower() else table
        if needle and shown.empty:
            continue
        count = f"{len(shown)} of {len(table)}" if len(shown) != len(table) else f"{len(table)}"
        body = (
            shown.head(MAX_ROWS).to_html(index=False, border=0, escape=True, na_rep="")
            if len(shown)
            else "<i>none</i>"
        )
        more = (
            f"<i>the first {MAX_ROWS} rows; type to find the others</i>"
            if len(shown) > MAX_ROWS
            else ""
        )
        unfolded = " open" if needle or title in OPEN else ""
        parts.append(
            f"<details{unfolded}><summary><b>{html.escape(title)}</b> ({count})</summary>"
            f"{body}{more}</details>"
        )
    if not parts:
        return f"<i>nothing holds {html.escape(needle)!s}</i>" if needle else "<i>no metadata</i>"
    return STYLE + "<div class='bexa-meta'>" + "".join(parts) + "</div>"


class MetadataView:
    """A title, a find box and the tables of one scan's metadata."""

    def __init__(self) -> None:
        import ipywidgets as ipw

        self.tables: dict[str, pd.DataFrame] = {}
        self.head = ipw.HTML()
        self.find = ipw.Text(
            placeholder="find a motor, counter, detector or value in every table (Enter)",
            continuous_update=False,
            layout=ipw.Layout(width="100%"),
        )
        self.find.observe(lambda _c: self._fill(), names="value")
        self.body = ipw.HTML()
        self.box = ipw.VBox([self.head, self.find, self.body])
        self.box.add_class("bexa-metadata")

    def show(self, tables: dict[str, pd.DataFrame], title: str) -> None:
        """Show ``tables`` (:func:`~bexa.core.metadata.scan_metadata`) under ``title``."""
        self.tables = dict(tables)
        rows = sum(len(t) for t in self.tables.values())
        self.head.value = (
            f"<b>{html.escape(title)}</b>: everything its files say, "
            f"{len(self.tables)} tables, {rows} rows"
        )
        self._fill()

    def wait(self, title: str) -> None:
        self.head.value = f"<b>{html.escape(title)}</b>: reading its metadata ..."
        self.body.value = ""

    def _fill(self) -> None:
        self.body.value = (
            "<div style='font-size:11px;max-height:560px;overflow:auto'>"
            f"{tables_html(self.tables, self.find.value)}</div>"
        )
