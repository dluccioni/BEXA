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

from typing import Any

import numpy as np
import xarray as xr

from bexa.core.roi import ROI
from bexa.core.stacks import brightest, shared_limits, total_volume
from bexa.viz import style
from bexa.viz.sliders import SliderPanel, format_value, in_notebook, log_block, static_backend
from bexa.viz.style import DEFAULT, as_array

__all__ = [
    "Browser",
    "RenderBrowser",
    "RoiPicker",
    "VolumeBrowser",
    "brightest",
    "browse",
    "browse_render",
    "browse_volume",
    "compare",
    "pick_roi",
    "shared_limits",
    "total_volume",
]


# the shared machinery lives in bexa.viz.sliders; these names stay for the callers that use them
_log_block = log_block
_in_notebook = in_notebook
_static_backend = static_backend
_format = format_value
_SliderPanel = SliderPanel


class Browser(_SliderPanel):
    """Step through the motor dims of a volume with sliders, one image at a time.

    Parameters
    ----------
    volume
        DataArray ``(motors..., y, x)``. A leading dim may also index different
        quantities (``xr.concat`` of maps with a ``quantity`` coordinate).
    clim
        ``"shared"`` (one scale for every frame, from the stack's statistics or
        a sample of frames), ``("shared", lo, hi)``, ``"each"`` (every frame on
        its own percentiles), ``("each", lo, hi)``, ``(vmin, vmax)`` or ``None``
        (each); left at ``DEFAULT`` the style defaults decide, as for ``cmap``,
        ``figsize`` and ``colorbar`` (:func:`bexa.viz.style.stack_mode`).
        :meth:`set_clim` changes it on show.
    """

    def __init__(
        self,
        volume: xr.DataArray,
        cmap: Any = DEFAULT,
        clim: Any = DEFAULT,
        log: bool = False,
        figsize: tuple[float, float] | None = None,
        widgets: bool | None = None,
        colorbar: Any = DEFAULT,
    ) -> None:
        import matplotlib.pyplot as plt

        if volume.ndim < 2:
            raise ValueError(f"a browser needs at least an image; got dims {tuple(volume.dims)}")
        self.volume = volume
        self.motor_dims = [str(d) for d in volume.dims[:-2]]
        self.index = {d: 0 for d in self.motor_dims}
        self.log = log
        self.kind = "browse"
        self._set_limits(clim)
        # No automatic layout: the slider axes are placed by hand below the image.
        figsize = style.figure_size("browse", figsize) or (6, 5)
        self.fig, self.ax = plt.subplots(figsize=figsize, layout="none")
        frame = self._frame()
        vmin, vmax = self._frame_limits(frame)
        self.im = self.ax.imshow(frame, cmap=style.colormap("browse", cmap), vmin=vmin, vmax=vmax)
        style.colorbar(self.fig, self.im, self.ax, "browse", colorbar)
        self.title = self.ax.set_title(self._title())
        self._init_sliders(widgets)

    @property
    def data(self) -> np.ndarray:
        """The frame on show (after ``log``)."""
        return self._block()

    def _frame(self) -> np.ndarray:
        return self._block()

    def redraw(self) -> None:
        frame = self._frame()
        self.im.set_data(frame)
        if self.limits is None:
            self.im.set_clim(*self._frame_limits(frame))

    def _apply_limits(self) -> None:
        self.im.set_clim(*self._frame_limits(self._frame()))


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
    (``chi``, ``mu``, ``energy``, a ``quantity`` index, ...) gets a slider. One
    block is read per move, so a stack opened from disk
    (:func:`bexa.io.cube.open_lazy`) browses without being loaded.

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
        ``"shared"`` (one scale for everything), ``("shared", lo, hi)``,
        ``"each"`` (per projection and condition), ``("each", lo, hi)``,
        ``(vmin, vmax)`` or ``None``; left at ``DEFAULT`` the style defaults
        decide (:func:`bexa.viz.style.stack_mode`). :meth:`set_clim` changes it
        on show.
    colorbar
        One per projection when ``True``; off unless asked (the sliders sit
        under the panels).
    """

    def __init__(
        self,
        volume: xr.DataArray,
        view: str = "projections",
        method: str = "max",
        log: bool = False,
        clim: Any = DEFAULT,
        cmap: Any = DEFAULT,
        threshold: float | None = None,
        figsize: tuple[float, float] | None = None,
        widgets: bool | None = None,
        colorbar: Any = DEFAULT,
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
        self.kind = "volume"
        self._set_limits(clim)
        self.threshold = threshold
        self.images: list[Any] = []
        cmap = style.colormap("volume", cmap)
        with_bar = style.local_option("volume", "colorbar", colorbar, fallback=False)
        if view == "projections":
            figsize = style.figure_size("volume", figsize, 1, 3, panel=(4, 4))
            self.fig, self.axes = plt.subplots(1, 3, figsize=figsize, layout="none")
            z, y, x = self.volume_dims
            panels = [(y, x, z), (z, x, y), (z, y, x)]  # (rows, columns, reduced dim)
            for ax, image, (rows, cols, reduced) in zip(
                self.axes, self._projections(), panels, strict=True
            ):
                vmin, vmax = self._frame_limits(image)
                self.images.append(ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto"))
                ax.set_title(f"{method} over {reduced}", **style.title_font("volume"))
                ax.set_xlabel(cols)
                ax.set_ylabel(rows)
                if with_bar:
                    style.colorbar(self.fig, self.images[-1], ax, "volume", True)
            self.fig.subplots_adjust(wspace=0.35)
        else:
            figsize = style.figure_size("volume", figsize) or (7, 7)
            self.fig = plt.figure(figsize=figsize, layout="none")
            self.ax = self.fig.add_subplot(1, 1, 1, projection="3d")
            self._draw_isosurface()
        self.title = self.fig.suptitle(self._title())
        self._init_sliders(widgets)

    def current(self) -> np.ndarray:
        """The ``(z, y, x)`` block at the current slider positions (after ``log``)."""
        return self._block()

    @property
    def data(self) -> np.ndarray:
        """The block on show, as :meth:`current`."""
        return self._block()

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
                    im.set_clim(*self._frame_limits(image))
        else:
            self._draw_isosurface()

    def _apply_limits(self) -> None:
        if self.view == "projections":
            for im, image in zip(self.images, self._projections(), strict=True):
                im.set_clim(*self._frame_limits(image))


def browse_volume(volume: xr.DataArray, **kwargs: Any) -> VolumeBrowser:
    """Open a :class:`VolumeBrowser` for ``volume``."""
    return VolumeBrowser(volume, **kwargs)


class RoiPicker:
    """Draw a rectangle on an image to define a pixel ROI.

    The current selection is in ``picker.roi`` (full-resolution pixel indices
    when the image carries ``y``/``x`` coordinates).
    """

    def __init__(
        self, image: Any, ax: Any = None, cmap: Any = DEFAULT, clim: Any = DEFAULT
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import RectangleSelector

        self.image = image
        data = as_array(image)
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=style.figure_size("roi_picker"))
        else:
            self.fig, self.ax = ax.figure, ax
        vmin, vmax = style.limits(data, clim, "roi_picker")
        self.ax.imshow(data, cmap=style.colormap("roi_picker", cmap), vmin=vmin, vmax=vmax)
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
    title
        A prefix of the title (the slider positions follow it).
    **render_kwargs
        Passed to :func:`bexa.viz.volume.render` (``mode``, ``log``, ``scale``, ``cmap``,
        ...). ``clim`` takes the stack grammar of :func:`bexa.viz.style.stack_mode`:
        ``"shared"`` (one scale for every condition: ``shared_limits`` at the style's
        ``render_clim`` percentiles, the default), ``("shared", lo, hi)``, ``"each"``,
        ``("each", lo, hi)``, ``(vmin, vmax)``; :meth:`restyle` and :meth:`set_clim`
        change them on show.
    """

    def __init__(
        self,
        volume: xr.DataArray,
        height: int = 600,
        title: str | None = None,
        **render_kwargs: Any,
    ) -> None:
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
        self.title_prefix = title
        self.render_kwargs = dict(render_kwargs)
        self._set_limits(self.render_kwargs.pop("clim", DEFAULT))
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

    def _set_limits(self, clim: Any = DEFAULT) -> None:
        """One colour scale for every condition, or each block on its own (see ``stack_mode``)."""
        mode, values = style.stack_mode(clim, "render")
        if mode == "shared":
            log = bool(self.render_kwargs.get("log", False))
            self.render_kwargs["clim"] = shared_limits(self.volume, values[0], values[1], log=log)
        elif mode == "fixed":
            self.render_kwargs["clim"] = values
        else:
            self.render_kwargs["clim"] = ("p", *values)  # render scales each block by itself

    def _moved(self, name: str, value: float) -> None:
        if not self._syncing:
            self.update(**{name: int(value)})

    def _title(self) -> str:
        text = super()._title()
        return f"{self.title_prefix}: {text}" if self.title_prefix else text

    def restyle(self, **render_kwargs: Any) -> None:
        """Render with other settings (``mode``, ``cmap``, ``clim``, ``log``...) and show again."""
        clim = render_kwargs.pop("clim", None)
        self.render_kwargs.update(render_kwargs)
        if clim is not None or "log" in render_kwargs:  # new units or a new spec: new limits
            self._set_limits(DEFAULT if clim is None else clim)
        self.update()

    def set_clim(self, clim: Any) -> None:
        """Change the colour scale (any ``stack_mode`` spec) and show the block again."""
        self.restyle(clim=clim)

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
