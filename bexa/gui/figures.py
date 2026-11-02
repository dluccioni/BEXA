"""Figures inside the GUI: an ipympl canvas when it exists, a picture otherwise.

With ipympl (``%matplotlib widget``) the canvas is a widget of its own: it
zooms, pans, and reports clicks, which draw ROIs and place peaks. Without it
(the ESRF POWER nodes have no ipympl) the figure is drawn into a PNG shown in
an ``ipywidgets.Image``: every view works, only the mouse tools are replaced
by typed values.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from typing import Any

from bexa._log import get_logger

log = get_logger(__name__)

__all__ = ["FigurePane", "interactive_backend"]


def interactive_backend() -> bool:
    """True when matplotlib draws into ipympl widgets (``%matplotlib widget``)."""
    import matplotlib

    backend = matplotlib.get_backend().lower()
    return "ipympl" in backend or "widget" in backend


class FigurePane:
    """One matplotlib figure that redraws in place.

    Parameters
    ----------
    size
        ``(width, height)`` in inches.
    interactive
        ``None``: an ipympl canvas when that backend is active, else a picture.
    """

    def __init__(
        self, size: tuple[float, float] = (7.0, 4.6), interactive: bool | None = None
    ) -> None:
        import ipywidgets as ipw

        self.interactive = interactive_backend() if interactive is None else bool(interactive)
        self._click: Callable[[Any], None] | None = None
        self._move: Callable[[Any], None] | None = None
        if self.interactive:
            import matplotlib.pyplot as plt

            with plt.ioff():
                self.fig = plt.figure(figsize=size)
            canvas = self.fig.canvas
            for name, value in (
                ("header_visible", False),
                ("footer_visible", True),
                ("toolbar_position", "left"),
            ):
                if hasattr(canvas, name):
                    setattr(canvas, name, value)
            self.widget: Any = canvas
            canvas.mpl_connect("button_press_event", self._clicked)
            canvas.mpl_connect("motion_notify_event", self._moved)
        else:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure

            self.fig = Figure(figsize=size, dpi=96)
            FigureCanvasAgg(self.fig)
            self.widget = ipw.Image(
                format="png", layout=ipw.Layout(max_width="100%", height="auto")
            )
        if hasattr(self.widget, "add_class"):
            self.widget.add_class("bexa-figure")

    # ------------------------------------------------------------------ drawing
    def clear(self) -> Any:
        """An empty figure (the axes are made by the caller)."""
        self.fig.clf()
        return self.fig

    def draw(self) -> None:
        """Show what was drawn: the canvas redraws, or the picture is rendered anew."""
        try:
            if self.interactive:
                self.fig.canvas.draw_idle()
            else:
                buffer = io.BytesIO()
                self.fig.savefig(buffer, format="png", dpi=self.fig.dpi, bbox_inches="tight")
                self.widget.value = buffer.getvalue()
        except Exception:  # a figure that cannot draw must not take the GUI down
            log.exception("drawing the figure failed")

    def savefig(self, path: Any, dpi: int = 150) -> None:
        self.fig.savefig(path, dpi=dpi, bbox_inches="tight")

    def close(self) -> None:
        if self.interactive:
            import matplotlib.pyplot as plt

            plt.close(self.fig)

    # ------------------------------------------------------------------ the mouse
    def on_click(self, callback: Callable[[Any], None] | None) -> None:
        """``callback(event)`` for clicks inside the axes (ipympl only); ``None`` removes it."""
        self._click = callback

    def on_move(self, callback: Callable[[Any], None] | None) -> None:
        """``callback(event)`` when the mouse moves over the axes (ipympl only)."""
        self._move = callback

    def _clicked(self, event: Any) -> None:
        if self._click is not None and event.inaxes is not None:
            try:
                self._click(event)
            except Exception:
                log.exception("click handler failed")

    def _moved(self, event: Any) -> None:
        if self._move is not None and event.inaxes is not None:
            try:
                self._move(event)
            except Exception:
                log.debug("move handler failed", exc_info=True)
