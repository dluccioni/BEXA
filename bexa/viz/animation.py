"""Animations of a stack along one axis, saved as GIF or MP4."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from bexa._log import get_logger
from bexa.viz import style
from bexa.viz.style import DEFAULT, as_array, percentile_limits

log = get_logger(__name__)

__all__ = ["animate", "frames_to_gif", "save_gif", "save_mp4"]


def animate(
    volume: xr.DataArray | np.ndarray,
    axis: int | str = 0,
    clim: Any = DEFAULT,
    cmap: Any = DEFAULT,
    interval: int = 200,
    title_fmt: str = "{dim} = {value:.3g}",
    normalize: Callable[[np.ndarray], np.ndarray] | None = None,
    figsize: tuple[float, float] | None = None,
    colorbar: Any = DEFAULT,
) -> Any:
    """A ``FuncAnimation`` stepping through ``axis`` with ``set_data`` and blitting.

    Parameters
    ----------
    clim
        ``"shared"`` (one scale for every frame), ``("shared", lo, hi)``,
        ``"each"`` (per frame), ``("each", lo, hi)``, ``(vmin, vmax)`` or
        ``None`` (each); left at ``DEFAULT`` the style defaults decide, as for
        ``cmap``, ``figsize`` and ``colorbar`` (see :func:`bexa.viz.style.stack_mode`).
    normalize
        Optional callable applied to every frame (for example dividing by a
        background box, as the legacy animation scripts did).
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation as mpl_animation

    if isinstance(volume, xr.DataArray):
        dim = volume.dims[axis] if isinstance(axis, int) else axis
        stack = volume.transpose(dim, ...)
        values = stack.coords[dim].values if dim in stack.coords else np.arange(stack.shape[0])
        data = as_array(stack)
    else:
        data = np.moveaxis(np.asarray(volume), axis if isinstance(axis, int) else 0, 0)
        dim, values = "index", np.arange(data.shape[0])
    if data.ndim != 3:
        data = data.reshape(data.shape[0], *data.shape[-2:])
    if normalize is not None:
        data = np.stack([normalize(f) for f in data])
    mode, percentiles = style.stack_mode(clim, "animate")
    limits: tuple[float, float] | None = None
    if mode == "shared":
        limits = percentile_limits(data, *percentiles)
    elif mode == "fixed":
        limits = percentiles

    fig, ax = plt.subplots(figsize=style.figure_size("animate", figsize) or (6, 5))
    vmin, vmax = limits or percentile_limits(data[0], *percentiles)
    im = ax.imshow(
        data[0], cmap=style.colormap("animate", cmap), vmin=vmin, vmax=vmax, animated=True
    )
    style.colorbar(fig, im, ax, "animate", colorbar)
    title = ax.set_title(title_fmt.format(dim=dim, value=values[0]))

    def update(i: int) -> tuple[Any, Any]:
        im.set_data(data[i])
        if limits is None:
            im.set_clim(*percentile_limits(data[i], *percentiles))
        title.set_text(title_fmt.format(dim=dim, value=values[i]))
        return im, title

    return mpl_animation.FuncAnimation(
        fig, update, frames=data.shape[0], interval=interval, blit=True
    )


def save_gif(anim: Any, path: str | Path, fps: int = 5) -> Path:
    """Save an animation as GIF with the Pillow writer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(path), writer="pillow", fps=fps)
    log.info("animation written to %s", path)
    return path


def save_mp4(anim: Any, path: str | Path, fps: int = 10, dpi: int = 150) -> Path:
    """Save an animation as MP4 (needs ffmpeg on the PATH)."""
    from matplotlib import animation as mpl_animation

    if not mpl_animation.writers.is_available("ffmpeg"):
        raise RuntimeError("ffmpeg is not available; install it or use save_gif")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(path), writer="ffmpeg", fps=fps, dpi=dpi)
    return path


def frames_to_gif(frames: Iterable[np.ndarray], path: str | Path, fps: int = 5) -> Path:
    """Write already-rendered RGB frames (uint8 arrays) to a GIF with imageio."""
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), list(frames), duration=1000.0 / fps, loop=0)
    return path
