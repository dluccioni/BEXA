"""What the slider browsers share: the indices, the two slider back ends, the colour limits.

:class:`bexa.viz.interactive.Browser`, :class:`~bexa.viz.interactive.VolumeBrowser`
and the render browser build on :class:`SliderPanel`: one index per leading dim,
an ipywidgets slider per dim in a notebook (matplotlib's own sliders in a
script), a block read per move, and colour limits that are shared, per frame
or fixed (:func:`bexa.viz.style.stack_mode`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import xarray as xr

from bexa.core.stacks import shared_limits
from bexa.viz.style import DEFAULT, as_array, stack_mode

__all__ = ["SliderPanel", "format_value", "in_notebook", "log_block", "static_backend"]


def log_block(block: np.ndarray) -> np.ndarray:
    """``log10(1 + I)`` of a block, the way the browsers show intensities."""
    return np.log10(1.0 + np.clip(block, 0, None))


def in_notebook() -> bool:
    """True inside a Jupyter kernel (where ipywidgets can show): :func:`bexa._log.in_kernel`."""
    from bexa._log import in_kernel

    return in_kernel()


def static_backend() -> bool:
    """True when figures are pictures (the inline backend): they cannot change once shown."""
    import matplotlib

    backend = matplotlib.get_backend().lower()
    return "inline" in backend or backend in ("agg", "pdf", "svg", "ps", "cairo")


def format_value(value: Any) -> str:
    """``0.1234`` for numbers, the plain text otherwise (a quantity name, for example)."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{value:.4g}"
    return str(value)


class SliderPanel:
    """One index per slider dim, the two slider back ends, and the colour limits of a stack.

    A subclass sets ``volume``, ``motor_dims``, ``index``, ``fig`` and ``title``
    before calling :meth:`_init_sliders`, and implements :meth:`redraw`. The
    data is read block by block through :meth:`_block`, so a volume opened
    with :func:`bexa.io.cube.open_lazy` is never loaded whole. ``limits`` is
    ``(vmin, vmax)`` for a shared or fixed scale, ``None`` when every frame is
    scaled on its own, with ``percentiles`` saying how.
    """

    volume: xr.DataArray
    motor_dims: list[str]
    index: dict[str, int]
    fig: Any
    title: Any
    log: bool = False
    kind: str = "browse"
    limits: tuple[float, float] | None = None
    percentiles: tuple[float, float] = (1.0, 99.0)
    _current_key: tuple[int, ...] | None = None
    _current_block: np.ndarray | None = None

    def _block(self) -> np.ndarray:
        """The numpy block at the current slider positions (after ``log``), read once per move."""
        key = tuple(self.index[d] for d in self.motor_dims)
        if key != self._current_key or self._current_block is None:
            block = as_array(self.volume.isel(dict(zip(self.motor_dims, key, strict=True))))
            self._current_block = log_block(block) if self.log else block
            self._current_key = key
        return self._current_block

    def _set_limits(self, clim: Any = DEFAULT) -> None:
        """Settle ``limits`` and ``percentiles`` from a stack ``clim`` (see ``stack_mode``)."""
        mode, values = stack_mode(clim, self.kind)
        if mode == "shared":  # from the stack's statistics when it has them: no full pass
            self.limits = shared_limits(self.volume, values[0], values[1], log=self.log)
            self.percentiles = values
        elif mode == "each":
            self.limits, self.percentiles = None, values
        else:
            self.limits = (values[0], values[1])

    def _frame_limits(self, image: np.ndarray) -> tuple[float, float]:
        """The limits of one frame: the shared or fixed ones, else its own percentiles."""
        if self.limits is not None:
            return self.limits
        from bexa.viz.style import percentile_limits

        return percentile_limits(image, *self.percentiles)

    def set_clim(self, clim: Any) -> None:
        """Change the colour limits of a browser on show (any ``stack_mode`` spec) and redraw."""
        self._set_limits(clim)
        self._apply_limits()
        if getattr(self, "_output", None) is not None:
            self._show()
        else:
            self.fig.canvas.draw_idle()

    def _apply_limits(self) -> None:
        """Put the current limits on the artists (implemented by the subclasses)."""
        raise NotImplementedError

    def _init_sliders(self, widgets: bool | None) -> None:
        self._sliders: list[Any] = []
        self._output: Any = None  # the output area that holds the figure with a static backend
        use_widgets = in_notebook() if widgets is None else widgets
        if use_widgets:
            self._ipywidgets()
        else:
            self._mpl_sliders()

    def _title(self) -> str:
        parts = []
        for d in self.motor_dims:
            value = (
                self.volume.coords[d].values[self.index[d]]
                if d in self.volume.coords
                else self.index[d]
            )
            parts.append(f"{d} = {format_value(value)}")
        return ", ".join(parts)

    def redraw(self) -> None:
        """Redraw the artists for the current indices (implemented by the subclasses)."""
        raise NotImplementedError

    _syncing = False  # True while the code moves the knobs, so they do not call back

    def update(self, **index: int) -> None:
        """Move to new indices (``update(mu=12)``), put the knobs there and redraw."""
        self.index.update({k: int(v) for k, v in index.items()})
        self._sync_sliders()
        self.redraw()
        self.title.set_text(self._title())
        if self._output is not None:
            self._show()
        else:
            self.fig.canvas.draw_idle()

    def _sync_sliders(self) -> None:
        """Move the knobs to the current indices, so dragging one away and back returns here."""
        sliders = getattr(self, "_sliders", None)
        if not sliders:
            return
        self._syncing = True
        try:
            for dim, slider in zip(self.motor_dims, sliders, strict=True):
                value = self.index[dim]
                if hasattr(slider, "set_val"):  # matplotlib
                    if int(slider.val) != value:
                        slider.set_val(value)
                elif slider.value != value:  # ipywidgets
                    slider.value = value
        finally:
            self._syncing = False

    def _moved(self, dim: str, value: float) -> None:
        """A knob moved: follow it, unless the code is moving the knobs."""
        if not self._syncing:
            self.update(**{dim: int(value)})

    def _show(self) -> None:
        """Draw the figure again into its output area (static backends)."""
        from IPython.display import display

        with self._output:
            self._output.clear_output(wait=True)
            display(self.fig)

    def _ipywidgets(self) -> None:
        try:
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError:
            self._mpl_sliders()
            return
        static = static_backend()  # a picture per move is slow: redraw on release there
        sliders = {
            d: widgets.IntSlider(
                min=0, max=self.volume.sizes[d] - 1, description=d, continuous_update=not static
            )
            for d in self.motor_dims
        }
        for name, slider in sliders.items():
            slider.observe(lambda change, n=name: self._moved(n, change["new"]), names="value")
        self._sliders = list(sliders.values())
        if static:  # inline figures are pictures: show them in an output area and redraw there
            import matplotlib.pyplot as plt

            self._output = widgets.Output()
            plt.close(self.fig)  # no second copy when the cell ends
            display(widgets.VBox([*self._sliders, self._output]))
            self._show()
        else:
            display(widgets.VBox(self._sliders))

    def _mpl_sliders(self) -> None:
        from matplotlib.widgets import Slider

        self.fig.subplots_adjust(bottom=0.12 + 0.05 * len(self.motor_dims))
        for i, d in enumerate(self.motor_dims):
            rect = self.fig.add_axes([0.2, 0.04 + 0.05 * i, 0.6, 0.03])
            slider = Slider(rect, d, 0, max(self.volume.sizes[d] - 1, 0), valinit=0, valstep=1)
            slider.on_changed(self._slider_callback(d))
            self._sliders.append(slider)

    def _slider_callback(self, dim: str) -> Callable[[float], None]:
        def moved(value: float) -> None:
            self._moved(dim, value)

        return moved
