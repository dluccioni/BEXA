"""The figure and animation set for one laser on/off cube.

Panel-for-panel port of ``auto_process/1d_plot_allscan.py`` and
``animation_allscan.py``: ROI image, on/off/difference curves, per-frame COM,
variance and skew against the scan axis, and normalised on/off GIFs. File
names follow the legacy pattern ``run42_ROI_r0_r1_c0_c1_<kind>.png`` so the
watcher recognises processed runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from bexa._log import get_logger
from bexa.analysis.peaks import frame_moments
from bexa.analysis.preprocess import box_mean
from bexa.analysis.pump_probe import scan_axis_of
from bexa.core.roi import ROI
from bexa.viz.animation import animate, save_gif
from bexa.viz.style import resolve_clim

log = get_logger(__name__)

DEFAULT_BACKGROUND_BOX = {"y": [10, 30], "x": [10, 30]}
FIGURE_KINDS = ("image", "diffplot", "COMplot", "FWHMplot", "skewplot")

__all__ = ["animations", "figure_set", "full_roi", "roi_tag"]


def full_roi(frame_shape: tuple[int, int]) -> ROI:
    return ROI(y=(0, frame_shape[0]), x=(0, frame_shape[1]))


def roi_tag(roi: ROI, frame_shape: tuple[int, int]) -> str:
    """``ROI_r0_r1_c0_c1`` as the legacy scripts named their outputs."""
    y0, y1, x0, x1 = roi.pixel_window(frame_shape)
    return f"ROI_{y0}_{y1}_{x0}_{x1}"


def _box_roi(box: Any) -> ROI:
    if isinstance(box, ROI):
        return box
    if box is None:
        box = DEFAULT_BACKGROUND_BOX
    if isinstance(box, dict):
        return ROI(y=tuple(box["y"]), x=tuple(box["x"]))
    (y0, y1), (x0, x1) = box
    return ROI(y=(y0, y1), x=(x0, x1))


def _rectangle(ax: Any, roi: ROI, frame_shape: tuple[int, int]) -> None:
    from matplotlib import patches

    y0, y1, x0, x1 = roi.pixel_window(frame_shape)
    ax.add_patch(
        patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1, edgecolor="r", facecolor="none")
    )


def _curves(ds: xr.Dataset, roi: ROI, axis: str) -> dict[str, Any]:
    frames = ds["frames"]
    shape = (int(frames.sizes["y"]), int(frames.sizes["x"]))
    ys, xs = roi.pixel_slices(shape)
    on = np.asarray(frames.sel(laser="on").values)
    off = np.asarray(frames.sel(laser="off").values)
    roi_on, roi_off = on[:, ys, xs], off[:, ys, xs]
    return {
        "x": np.asarray(ds[axis].values, dtype=float),
        "shape": shape,
        "on": on,
        "off": off,
        "roi_on": roi_on,
        "roi_off": roi_off,
        "sum_on": roi_on.sum(axis=(1, 2)),
        "sum_off": roi_off.sum(axis=(1, 2)),
        "stats_on": frame_moments(roi_on, orders=(2, 3)),
        "stats_off": frame_moments(roi_off, orders=(2, 3)),
    }


def _pair(ax: Any, x: np.ndarray, on: np.ndarray, off: np.ndarray, axis: str, ylabel: str) -> None:
    ax.plot(x, on, label="on", color="red")
    ax.plot(x, off, label="off", color="black")
    ax.set_xlabel(axis)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)


def _diff(ax: Any, x: np.ndarray, on: np.ndarray, off: np.ndarray, axis: str, ylabel: str) -> None:
    ax.plot(x, on - off, label="on-off", color="red")
    ax.set_xlabel(axis)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)


def figure_set(
    ds: xr.Dataset,
    out_dir: str | Path,
    label: str,
    roi: ROI | None = None,
    axis: str | None = None,
    background_box: Any = None,
    dpi: int = 100,
    kinds: tuple[str, ...] = FIGURE_KINDS,
) -> dict[str, Path]:
    """Write the legacy figure set; returns ``{kind: path}``."""
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    axis = axis or scan_axis_of(ds)
    if axis == "delays":
        axis = "delay"
    frames = ds["frames"]
    shape = (int(frames.sizes["y"]), int(frames.sizes["x"]))
    roi = roi if roi is not None else full_roi(shape)
    bkg = _box_roi(background_box)
    c = _curves(ds, roi, axis)
    x = c["x"]
    tag = roi_tag(roi, shape)
    written: dict[str, Path] = {}

    def save(fig: Any, kind: str) -> None:
        path = out_dir / f"{label}_{tag}_{kind}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        written[kind] = path

    if "image" in kinds:
        fig, ax = plt.subplots()
        total = c["roi_on"].sum(axis=0)
        vmin, vmax = resolve_clim(total, ("p", 1, 99))
        ax.imshow(total, vmin=vmin, vmax=vmax)
        ax.set_title(f"{label} ROI sum (laser on)")
        save(fig, "image")

    sum_image = c["on"].sum(axis=0)
    if "diffplot" in kinds:
        fig, ax = plt.subplots(1, 3, figsize=(13, 4))
        ax[0].imshow(sum_image)
        _rectangle(ax[0], roi, shape)
        _rectangle(ax[0], bkg, shape)
        _pair(ax[1], x, c["sum_on"], c["sum_off"], axis, "raw intensity /counts")
        ax[2].plot(x, c["sum_on"] - c["sum_off"], label="(on-off)", color="black")
        ax[2].set_xlabel(axis)
        ax[2].set_ylabel("differential scattering /arb(not norm).")
        ax[2].legend(frameon=False)
        fig.suptitle(label)
        save(fig, "diffplot")

    s_on, s_off = c["stats_on"], c["stats_off"]
    if "COMplot" in kinds:
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 6))
        ax[0, 0].imshow(sum_image)
        _rectangle(ax[0, 0], roi, shape)
        _rectangle(ax[0, 0], bkg, shape)
        _pair(ax[0, 1], x, s_on["com_x"], s_off["com_x"], axis, "com_x /pixel")
        _pair(ax[0, 2], x, s_on["com_y"], s_off["com_y"], axis, "com_y /pixel")
        _diff(ax[1, 0], x, s_on["com_x"], s_off["com_x"], axis, "com_x /pixel")
        _diff(ax[1, 1], x, s_on["com_y"], s_off["com_y"], axis, "com_y /pixel")
        ax[1, 2].axis("off")
        fig.suptitle(f"COM {label}", fontsize=30)
        save(fig, "COMplot")

    if "FWHMplot" in kinds:
        fig, ax = plt.subplots(1, 4, figsize=(11, 4))
        _pair(ax[0], x, s_on["var_x"], s_off["var_x"], axis, "x_variance /pixel")
        _pair(ax[1], x, s_on["var_y"], s_off["var_y"], axis, "y_variance /pixel")
        _diff(ax[2], x, s_on["var_x"], s_off["var_x"], axis, "x_variance /pixel")
        _diff(ax[3], x, s_on["var_y"], s_off["var_y"], axis, "y_variance /pixel")
        fig.suptitle(f"{label} FWHM")
        save(fig, "FWHMplot")

    if "skewplot" in kinds:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        _pair(ax[0], x, s_on["skew_x"], s_off["skew_x"], axis, "x_skew /pixel")
        _pair(ax[1], x, s_on["skew_y"], s_off["skew_y"], axis, "y_skew /pixel")
        fig.suptitle(f"{label} skew")
        save(fig, "skewplot")
    return written


def animations(
    ds: xr.Dataset,
    out_dir: str | Path,
    label: str,
    roi: ROI | None = None,
    axis: str | None = None,
    background_box: Any = None,
    fps: int = 5,
) -> dict[str, Path]:
    """Laser on and off GIFs of the ROI, each frame divided by its background box mean."""
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    axis = axis or scan_axis_of(ds)
    if axis == "delays":
        axis = "delay"
    frames = ds["frames"]
    shape = (int(frames.sizes["y"]), int(frames.sizes["x"]))
    roi = roi if roi is not None else full_roi(shape)
    bkg = _box_roi(background_box or {"y": [0, 20], "x": [0, 20]})
    ys, xs = roi.pixel_slices(shape)
    values = np.asarray(ds[axis].values, dtype=float)
    written: dict[str, Path] = {}
    for state, robust in (("on", False), ("off", True)):
        full = np.asarray(frames.sel(laser=state).values, dtype=np.float64)
        background = box_mean(full, bkg, robust=robust)
        data = full[:, ys, xs] / background[:, None, None]
        volume = xr.DataArray(data, dims=(axis, "y", "x"), coords={axis: values})
        anim = animate(
            volume,
            axis=axis,
            clim="each",
            interval=200,
            title_fmt=f"Laser {state.upper()} | {label} | {axis}={{value:.2f}}",
        )
        path = save_gif(anim, out_dir / f"{label}_animation_{state}.gif", fps=fps)
        plt.close(anim._fig)
        written[f"animation_{state}"] = path
    return written
