"""The bexa GUI for Jupyter notebooks.

>>> import bexa.gui
>>> app = bexa.gui.App(root="X:/beamtime/RAW_DATA")   # the last line of a cell shows it
>>> app.ws                                             # the workspace behind it

Tabs: Data (open scans of any facility, scan sets, files, demo data), Browse
(the inside of folders and files), View (stacks, images, maps, curves, ROIs),
Analysis (every operation of :mod:`bexa.workspace.catalog`), Fit (peaks with
bounds and fixed values), Convert (formats and exports), Log (commands, the
script, workspace files). Needs ipywidgets; ipympl makes the figures
interactive (ROIs drawn with the mouse, peaks placed by a click).
"""

from __future__ import annotations

from typing import Any

__all__ = ["App", "launch"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from bexa.gui import app

        return getattr(app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
