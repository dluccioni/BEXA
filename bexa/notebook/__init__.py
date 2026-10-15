"""Helpers for Jupyter and VS Code interactive windows.

Call :func:`setup` once at the top of a notebook. It picks the matplotlib
backend, widens the output cells, and routes bexa's log messages to the cell.
"""

from __future__ import annotations

import sys
from typing import Any

from bexa._log import configure, get_logger
from bexa._log import in_ipython as _in_ipython
from bexa._log import in_kernel as _in_kernel

log = get_logger(__name__)


def in_ipython() -> bool:
    """True inside IPython, Jupyter or a VS Code interactive window."""
    return _in_ipython()


def setup(
    backend: str = "widget",
    wide: bool = True,
    autoreload: bool = False,
    log_level: str = "INFO",
    theme: str | None = None,
) -> None:
    """Configure the kernel for bexa.

    Parameters
    ----------
    backend
        ``"widget"`` (ipympl, interactive), ``"inline"`` or ``"qt"``. Falls
        back to ``inline`` when ipympl is not installed.
    wide
        Let output cells use the full window width.
    autoreload
        Reload edited bexa modules automatically (handy while developing).
    theme
        ``"screen"`` or ``"paper"``: apply that :mod:`bexa.viz.style` theme to
        matplotlib (fonts, dpi, ticks); ``None`` leaves ``rcParams`` alone.
    """
    configure(log_level, stream=sys.stdout, force=True)
    if theme is not None:
        from bexa.viz.style import use

        use(theme)
    if not in_ipython():
        log.info("not running under IPython; nothing to set up")
        return
    from IPython import get_ipython

    ip = get_ipython()
    if backend == "widget":
        try:
            import ipympl  # noqa: F401
        except ImportError:
            log.warning("ipympl is not installed; using the inline backend")
            backend = "inline"
    ip.run_line_magic("matplotlib", backend)
    if autoreload:
        ip.run_line_magic("load_ext", "autoreload")
        ip.run_line_magic("autoreload", "2")
    if wide:
        try:
            from IPython.display import HTML, display

            display(HTML("<style>.container, .jp-Cell { width: 100% !important; }</style>"))
        except Exception:  # pragma: no cover - cosmetic
            pass
    log.info("bexa notebook setup done (matplotlib backend: %s)", backend)


def show_plotly(
    fig: Any, how: str = "auto", height: int = 600, include_plotlyjs: str | bool = "cdn"
) -> Any:
    """Show a plotly figure in the notebook (see :func:`bexa.viz.volume.show_plotly`).

    The default embeds the figure as an inline frame, which works in any JupyterLab without
    the plotly extension; ``how="native"`` uses plotly's own display when the extension exists.
    """
    from bexa.viz.volume import show_plotly as _show_plotly

    return _show_plotly(fig, how=how, height=height, include_plotlyjs=include_plotlyjs)


def in_kernel() -> bool:
    """True inside a Jupyter kernel (JupyterLab, VS Code interactive window), where widgets show."""
    return _in_kernel()


_PICKER_NAMES = ("ScanPicker", "list_datasets", "pick_scans")


def __getattr__(name: str) -> Any:
    """``ScanPicker``, ``list_datasets`` and ``pick_scans`` load the picker module on demand."""
    if name in _PICKER_NAMES:
        from bexa.notebook import picker

        return getattr(picker, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ScanPicker",
    "in_ipython",
    "in_kernel",
    "list_datasets",
    "pick_scans",
    "setup",
    "show_plotly",
]
