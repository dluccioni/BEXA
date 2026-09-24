"""Animations of a stack along one axis, saved as GIF or MP4."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from bexa._log import get_logger
from bexa.viz.style import as_array, resolve_clim

log = get_logger(__name__)

__all__ = ["animate", "frames_to_gif", "save_gif", "save_mp4"]


def animate(
    volume: xr.DataArray | np.ndarray,
    axis: int | str = 0,
    clim: Any = "shared",
    cmap: str | None = None,
    interval: int = 200,
    title_fmt: str = "{dim} = {value:.3g}",
    normalize: Callable[[np.ndarray], np.ndarray] | None = None,
    figsize: tuple[float, float] = (6, 5),
) -> Any:
    """A ``FuncAnimation`` stepping through ``axis`` with ``set_data`` and blitting.

    Parameters
    ----------
    clim
        ``"shared"`` (global percentiles), ``"each"`` (per frame) or ``(vmin, vmax)``.
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
    limits = (
        resolve_clim(data, ("p", 1, 99)) if clim == "shared" else (None if clim == "each" else clim)
    )

    fig, ax = plt.subplots(figsize=figsize)
    vmin, vmax = limits or resolve_clim(data[0], ("p", 1, 99))
    im = ax.imshow(data[0], cmap=cmap, vmin=vmin, vmax=vmax, animated=True)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    title = ax.set_title(title_fmt.format(dim=dim, value=values[0]))

    def update(i: int) -> tuple[Any, Any]:
        im.set_data(data[i])
        if limits is None:
            im.set_clim(*resolve_clim(data[i], ("p", 1, 99)))
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
