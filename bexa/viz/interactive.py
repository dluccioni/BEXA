"""Sliders and ROI pickers that work in Jupyter, VS Code and plain scripts.

In a notebook with ipywidgets the browser gets one slider per motor dim; in a
terminal session matplotlib's own ``Slider`` widgets are used instead. Updates
call ``set_data`` on the existing image, so stepping through a preview volume
stays fast.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from bexa.core.roi import ROI
from bexa.viz.style import as_array, resolve_clim

__all__ = ["Browser", "RoiPicker", "browse", "compare", "pick_roi"]


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython

        ip = get_ipython()
        return ip is not None and "IPKernelApp" in ip.config
    except ImportError:
        return False


class Browser:
    """Step through the motor dims of a volume with sliders.

    Parameters
    ----------
    volume
        DataArray ``(motors..., y, x)``.
    clim
        ``"each"`` rescales every frame; ``"shared"`` (default) uses global
        percentiles; or an explicit ``(vmin, vmax)``.
    """

    def __init__(
        self,
        volume: xr.DataArray,
        cmap: str | None = None,
        clim: Any = "shared",
        log: bool = False,
        figsize: tuple[float, float] = (6, 5),
        widgets: bool | None = None,
    ) -> None:
        import matplotlib.pyplot as plt

        self.volume = volume
        self.motor_dims = [d for d in volume.dims if d not in ("y", "x")]
        self.index = {d: 0 for d in self.motor_dims}
        self.log = log
        self.data = as_array(volume)
        if log:
            self.data = np.log10(1.0 + np.clip(self.data, 0, None))
        if clim == "each":
            self.limits = None
        elif clim == "shared":
            self.limits = resolve_clim(self.data, ("p", 1, 99))
        else:
            self.limits = clim
        # No automatic layout: the slider axes are placed by hand below the image.
        self.fig, self.ax = plt.subplots(figsize=figsize, layout="none")
        frame = self._frame()
        vmin, vmax = self.limits or resolve_clim(frame, ("p", 1, 99))
        self.im = self.ax.imshow(frame, cmap=cmap, vmin=vmin, vmax=vmax)
        self.fig.colorbar(self.im, ax=self.ax, fraction=0.046, pad=0.04)
        self.title = self.ax.set_title(self._title())
        self._sliders: list[Any] = []
        use_widgets = _in_notebook() if widgets is None else widgets
        if use_widgets:
            self._ipywidgets()
        else:
            self._mpl_sliders()

    def _frame(self) -> np.ndarray:
        return self.data[tuple(self.index[d] for d in self.motor_dims)]

    def _title(self) -> str:
        parts = []
        for d in self.motor_dims:
            value = (
                self.volume.coords[d].values[self.index[d]]
                if d in self.volume.coords
                else self.index[d]
            )
            parts.append(f"{d} = {value:.4g}")
        return ", ".join(parts)

    def update(self, **index: int) -> None:
        """Move to new indices (``update(mu=12)``) and redraw the image only."""
        self.index.update({k: int(v) for k, v in index.items()})
        frame = self._frame()
        self.im.set_data(frame)
        if self.limits is None:
            self.im.set_clim(*resolve_clim(frame, ("p", 1, 99)))
        self.title.set_text(self._title())
        self.fig.canvas.draw_idle()

    def _ipywidgets(self) -> None:
        try:
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError:
            self._mpl_sliders()
            return
        sliders = {
            d: widgets.IntSlider(
                min=0, max=self.volume.sizes[d] - 1, description=d, continuous_update=True
            )
            for d in self.motor_dims
        }
        for name, slider in sliders.items():
            slider.observe(lambda change, n=name: self.update(**{n: change["new"]}), names="value")
        self._sliders = list(sliders.values())
        display(widgets.VBox(self._sliders))

    def _mpl_sliders(self) -> None:
        from matplotlib.widgets import Slider

        self.fig.subplots_adjust(bottom=0.12 + 0.05 * len(self.motor_dims))
        for i, d in enumerate(self.motor_dims):
            rect = self.fig.add_axes([0.2, 0.04 + 0.05 * i, 0.6, 0.03])
            slider = Slider(rect, d, 0, max(self.volume.sizes[d] - 1, 0), valinit=0, valstep=1)
            slider.on_changed(lambda value, n=d: self.update(**{n: int(value)}))
            self._sliders.append(slider)


def browse(volume: xr.DataArray, **kwargs: Any) -> Browser:
    """Open a :class:`Browser` for ``volume``."""
    return Browser(volume, **kwargs)


def compare(a: xr.DataArray, b: xr.DataArray, **kwargs: Any) -> tuple[Browser, Browser]:
    """Two browsers whose sliders move together (same dims expected)."""
    first = Browser(a, **kwargs)
    second = Browser(b, **kwargs)
    linked = first.update

    def both(**index: int) -> None:
        linked(**index)
        second.update(**index)

    first.update = both  # type: ignore[method-assign]
    return first, second


class RoiPicker:
    """Draw a rectangle on an image to define a pixel ROI.

    The current selection is in ``picker.roi`` (full-resolution pixel indices
    when the image carries ``y``/``x`` coordinates).
    """

    def __init__(
        self, image: Any, ax: Any = None, cmap: str | None = None, clim: Any = ("p", 1, 99)
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import RectangleSelector

        self.image = image
        data = as_array(image)
        if ax is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.fig, self.ax = ax.figure, ax
        vmin, vmax = resolve_clim(data, clim)
        self.ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        self.ax.set_title("drag a rectangle to select the ROI")
        self.roi: ROI | None = None
        self.selector = RectangleSelector(
            self.ax, self._on_select, useblit=True, button=[1], interactive=True
        )

    def _on_select(self, eclick: Any, erelease: Any) -> None:
        x0, x1 = sorted((eclick.xdata, erelease.xdata))
        y0, y1 = sorted((eclick.ydata, erelease.ydata))
        y = (
            self.image.coords["y"].values
            if hasattr(self.image, "coords") and "y" in self.image.coords
            else None
        )
        x = (
            self.image.coords["x"].values
            if hasattr(self.image, "coords") and "x" in self.image.coords
            else None
        )
        if y is not None and x is not None and len(y) > 1 and len(x) > 1:
            dy, dx = y[1] - y[0], x[1] - x[0]
            ry = (_snap(y[0], dy, y0), _snap(y[0], dy, y1 + 1))
            rx = (_snap(x[0], dx, x0), _snap(x[0], dx, x1 + 1))
        else:
            ry = (round(float(y0)), round(float(y1)) + 1)
            rx = (round(float(x0)), round(float(x1)) + 1)
        self.roi = ROI(y=ry, x=rx)
        self.ax.set_title(f"ROI y={ry}, x={rx}")
        self.fig.canvas.draw_idle()


def _snap(origin: Any, step: Any, value: Any) -> int:
    """Pixel index of data coordinate ``value`` on an axis starting at ``origin``."""
    return round(float(origin) + float(step) * float(value))


def pick_roi(image: Any, block: bool = False, **kwargs: Any) -> RoiPicker:
    """Interactive rectangle selection; with ``block=True`` waits for the window to close."""
    picker = RoiPicker(image, **kwargs)
    if block:
        import matplotlib.pyplot as plt

        plt.show(block=True)
    return picker
