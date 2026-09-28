"""Sliders and ROI pickers that work in Jupyter, VS Code and plain scripts.

In a notebook with ipywidgets a browser gets one slider per extra dim; in a
terminal session matplotlib's own ``Slider`` widgets are used instead.
:class:`Browser` steps through the motor dims of a volume one image at a time;
:class:`VolumeBrowser` shows a whole ``(z, y, x)`` volume, as its projections
or as an isosurface, and steps through the conditions in front of it, such as
the ``(chi, mu)`` frame or the energy taken from every layer of a z-stack.
Updates redraw only what changed, so stepping through a preview stays fast.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import xarray as xr

from bexa.core.roi import ROI
from bexa.viz.style import as_array, resolve_clim

__all__ = [
    "Browser",
    "RenderBrowser",
    "RoiPicker",
    "VolumeBrowser",
    "browse",
    "browse_render",
    "browse_volume",
    "compare",
    "pick_roi",
]


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython

        ip = get_ipython()
        return ip is not None and "IPKernelApp" in ip.config
    except ImportError:
        return False


def _static_backend() -> bool:
    """True when figures are pictures (the inline backend): they cannot change once shown."""
    import matplotlib

    backend = matplotlib.get_backend().lower()
    return "inline" in backend or backend in ("agg", "pdf", "svg", "ps", "cairo")


def _format(value: Any) -> str:
    """``0.1234`` for numbers, the plain text otherwise (a quantity name, for example)."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{value:.4g}"
    return str(value)


class _SliderPanel:
    """What the browsers share: one index per slider dim and the two slider back ends.

    A subclass sets ``volume``, ``motor_dims``, ``index``, ``fig`` and ``title``
    before calling :meth:`_init_sliders`, and implements :meth:`redraw`.
    """

    volume: xr.DataArray
    motor_dims: list[str]
    index: dict[str, int]
    fig: Any
    title: Any

    def _init_sliders(self, widgets: bool | None) -> None:
        self._sliders: list[Any] = []
        self._output: Any = None  # the output area that holds the figure with a static backend
        use_widgets = _in_notebook() if widgets is None else widgets
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
            parts.append(f"{d} = {_format(value)}")
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
        static = _static_backend()  # a picture per move is slow: redraw on release there
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


class Browser(_SliderPanel):
    """Step through the motor dims of a volume with sliders, one image at a time.

    Parameters
    ----------
    volume
        DataArray ``(motors..., y, x)``. A leading dim may also index different
        quantities (``xr.concat`` of maps with a ``quantity`` coordinate).
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
        self.motor_dims = [str(d) for d in volume.dims if d not in ("y", "x")]
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
        self._init_sliders(widgets)

    def _frame(self) -> np.ndarray:
        return self.data[tuple(self.index[d] for d in self.motor_dims)]

    def redraw(self) -> None:
        frame = self._frame()
        self.im.set_data(frame)
        if self.limits is None:
            self.im.set_clim(*resolve_clim(frame, ("p", 1, 99)))


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


class VolumeBrowser(_SliderPanel):
    """Show a ``(z, y, x)`` volume and step through the conditions in front of it.

    The last three dims of ``volume`` are the volume axes; every leading dim
    (``chi``, ``mu``, ``energy``, a ``quantity`` index, ...) gets a slider. The
    whole array is held in memory, so use it on previews and maps.

    Parameters
    ----------
    view
        ``"projections"`` shows the three maximum-intensity projections
        (``method="sum"`` for summed projections) and updates them in place;
        ``"isosurface"`` redraws a marching-cubes surface (needs scikit-image).
    threshold
        Isosurface level; by default the 90th percentile of the first volume,
        kept fixed so the surfaces of different conditions can be compared.
    clim
        ``"shared"`` (percentiles of everything), ``"each"`` (per projection
        and condition) or an explicit ``(vmin, vmax)``.
    """

    def __init__(
        self,
        volume: xr.DataArray,
        view: str = "projections",
        method: str = "max",
        log: bool = False,
        clim: Any = "shared",
        cmap: str | None = None,
        threshold: float | None = None,
        figsize: tuple[float, float] | None = None,
        widgets: bool | None = None,
    ) -> None:
        import matplotlib.pyplot as plt

        if volume.ndim < 3:
            raise ValueError(f"a volume needs at least three dims; got {tuple(volume.dims)}")
        if view not in ("projections", "isosurface"):
            raise ValueError("view must be 'projections' or 'isosurface'")
        if method not in ("max", "sum"):
            raise ValueError("method must be 'max' or 'sum'")
        self.volume = volume
        self.view, self.method, self.log = view, method, log
        self.volume_dims = [str(d) for d in volume.dims[-3:]]
        self.motor_dims = [str(d) for d in volume.dims[:-3]]
        self.index = {d: 0 for d in self.motor_dims}
        self.data = as_array(volume)
        if log:
            self.data = np.log10(1.0 + np.clip(self.data, 0, None))
        if clim == "each":
            self.limits = None
        elif clim == "shared":
            self.limits = resolve_clim(self.data, ("p", 1, 99))
        else:
            self.limits = clim
        self.threshold = threshold
        self.images: list[Any] = []
        if view == "projections":
            self.fig, self.axes = plt.subplots(1, 3, figsize=figsize or (12, 4), layout="none")
            z, y, x = self.volume_dims
            panels = [(y, x, z), (z, x, y), (z, y, x)]  # (rows, columns, reduced dim)
            for ax, image, (rows, cols, reduced) in zip(
                self.axes, self._projections(), panels, strict=True
            ):
                vmin, vmax = self.limits or resolve_clim(image, ("p", 1, 99))
                self.images.append(ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto"))
                ax.set_title(f"{method} over {reduced}", fontsize=10)
                ax.set_xlabel(cols)
                ax.set_ylabel(rows)
            self.fig.subplots_adjust(wspace=0.35)
        else:
            self.fig = plt.figure(figsize=figsize or (7, 7), layout="none")
            self.ax = self.fig.add_subplot(1, 1, 1, projection="3d")
            self._draw_isosurface()
        self.title = self.fig.suptitle(self._title())
        self._init_sliders(widgets)

    def current(self) -> np.ndarray:
        """The ``(z, y, x)`` block at the current slider positions."""
        return self.data[tuple(self.index[d] for d in self.motor_dims)]

    def _projections(self) -> list[np.ndarray]:
        block = self.current()
        reducer = np.nanmax if self.method == "max" else np.nansum
        return [reducer(block, axis=0), reducer(block, axis=1), reducer(block, axis=2)]

    def _draw_isosurface(self) -> None:
        from bexa.viz.volume import isosurface

        block = self.current()
        if self.threshold is None:
            finite = block[np.isfinite(block)]
            self.threshold = float(np.percentile(finite, 90)) if finite.size else 0.0
        self.ax.cla()
        try:
            isosurface(block, threshold=self.threshold, ax=self.ax)
        except (ValueError, RuntimeError):  # no voxel reaches the level for this condition
            self.ax.text2D(
                0.5, 0.5, "no voxels at the threshold", transform=self.ax.transAxes, ha="center"
            )
        z, y, x = self.volume_dims
        self.ax.set_xlabel(x)
        self.ax.set_ylabel(y)
        self.ax.set_zlabel(z)

    def redraw(self) -> None:
        if self.view == "projections":
            for im, image in zip(self.images, self._projections(), strict=True):
                im.set_data(image)
                if self.limits is None:
                    im.set_clim(*resolve_clim(image, ("p", 1, 99)))
        else:
            self._draw_isosurface()


def browse_volume(volume: xr.DataArray, **kwargs: Any) -> VolumeBrowser:
    """Open a :class:`VolumeBrowser` for ``volume``."""
    return VolumeBrowser(volume, **kwargs)


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
        buttons: Any = [1]  # the left button; the stub wants MouseButton, the code accepts ints
        self.selector = RectangleSelector(
            self.ax, self._on_select, useblit=True, button=buttons, interactive=True
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


class RenderBrowser(_SliderPanel):
    """Sliders for the leading dims of a volume; the ``(z, y, x)`` block is rendered on release.

    Each condition is drawn with :func:`bexa.viz.volume.render` (plotly: colour and opacity
    follow the intensity) into an output area under the sliders. Every move costs one
    rendering, so the sliders act when released. Needs ipywidgets; without it, call
    ``render(volume.isel(...))`` for one condition at a time.

    Parameters
    ----------
    volume
        DataArray ``(conditions..., z, y, x)``, for example previews stacked on ``samz``.
    height
        Height of the rendering in pixels.
    **render_kwargs
        Passed to :func:`bexa.viz.volume.render` (``mode``, ``log``, ``scale``, ...).
    """

    def __init__(self, volume: xr.DataArray, height: int = 600, **render_kwargs: Any) -> None:
        try:
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError as exc:
            raise ImportError(
                "browse_render needs ipywidgets; without it, render(volume.isel(...)) shows one "
                "condition at a time"
            ) from exc
        if volume.ndim < 3:
            raise ValueError(f"a volume needs at least three dims; got {tuple(volume.dims)}")
        self.volume = volume
        self.motor_dims = [str(d) for d in volume.dims[:-3]]
        self.index = {d: 0 for d in self.motor_dims}
        self.height = height
        self.render_kwargs = render_kwargs
        self.figure: Any = None
        self._syncing = False
        self.output = widgets.Output()
        self.sliders = {
            d: widgets.IntSlider(
                min=0, max=volume.sizes[d] - 1, description=d, continuous_update=False
            )
            for d in self.motor_dims
        }
        for name, slider in self.sliders.items():
            slider.observe(lambda change, n=name: self._moved(n, change["new"]), names="value")
        display(widgets.VBox([*self.sliders.values(), self.output]))
        self.update()

    def _moved(self, name: str, value: float) -> None:
        if not self._syncing:
            self.update(**{name: int(value)})

    def current(self) -> xr.DataArray:
        """The ``(z, y, x)`` block at the current slider positions."""
        return self.volume.isel(self.index)

    def update(self, **index: int) -> None:
        """Move to new indices (``update(chi=2, mu=5)``), render and show the block."""
        from bexa.viz.volume import render

        self.index.update({k: int(v) for k, v in index.items()})
        self._syncing = True  # the knobs follow a call from code without rendering twice
        try:
            for name, value in self.index.items():
                self.sliders[name].value = value
        finally:
            self._syncing = False
        self.figure = render(self.current(), title=self._title(), **self.render_kwargs)
        self.show(self.figure)

    def show(self, fig: Any) -> None:
        """Draw ``fig`` into the output area under the sliders."""
        from bexa.viz.volume import show_plotly

        with self.output:
            self.output.clear_output(wait=True)
            show_plotly(fig, how="iframe", height=self.height)


def browse_render(volume: xr.DataArray, **kwargs: Any) -> RenderBrowser:
    """Open a :class:`RenderBrowser` for ``volume``."""
    return RenderBrowser(volume, **kwargs)
