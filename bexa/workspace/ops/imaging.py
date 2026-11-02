"""Imaging: per-pixel maps of stacks in memory, orientation colours, KAM, strain, drift."""

from __future__ import annotations

import numpy as np
import xarray as xr

from bexa.analysis import imaging as maps
from bexa.workspace.objects import DataObject
from bexa.workspace.ops._base import IMAGES, new, other_dims, pixel_dims, reduce_over, require

__all__ = [
    "align_frames",
    "argmax_map",
    "gradient_maps",
    "kam_map",
    "moments_maps",
    "orientation_map",
    "peak_maps",
    "segment",
    "strain",
]

STRAIN_KINDS = ("theta", "two_theta", "energy")


def _along(obj: DataObject, dim: str) -> tuple[xr.DataArray, str, np.ndarray]:
    """The stack summed over every motor dim but ``dim``, with ``dim`` first, and its values."""
    require(obj, ("stack",), "a per-pixel map")
    da = obj.array
    motors = other_dims(da)
    dim = dim or (motors[-1] if motors else "")
    if dim not in motors:
        raise ValueError(f"choose one of the stack's motor dims {motors}")
    summed = reduce_over(da, [d for d in motors if d != dim], "sum").transpose(dim, *pixel_dims(da))
    return summed, dim, obj.coord(dim).astype(float)


def _image_like(summed: xr.DataArray, dim: str, values: np.ndarray, name: str) -> xr.DataArray:
    template = summed.isel({dim: 0}, drop=True)
    return template.copy(data=np.asarray(values)).rename(name)


def moments_maps(obj: DataObject, dim: str = "", sigma: float = 0.0) -> list[DataObject]:
    """Centre of mass, spread, FWHM, skewness and kurtosis of every pixel's curve along ``dim``."""
    from bexa.analysis.rocking import moments

    summed, dim, x = _along(obj, dim)
    stats = moments(np.asarray(summed.values, dtype=float), x, axis=0, sigma=sigma)
    names = {
        "com": f"com_{dim}",
        "std": f"width_{dim}",
        "fwhm": f"fwhm_{dim}",
        "skew": f"skew_{dim}",
        "kurtosis": f"kurtosis_{dim}",
        "total": "total",
    }
    return [
        new(
            obj,
            _image_like(summed, dim, stats[key], name),
            name,
            meta={"quantity": name, "motor": dim},
        )
        for key, name in names.items()
    ]


def peak_maps(obj: DataObject, dim: str = "", threshold: float = 0.1) -> list[DataObject]:
    """A Gaussian fitted to every pixel's curve along ``dim`` (all pixels at once):
    centre, FWHM and height maps."""
    summed, dim, x = _along(obj, dim)
    found = maps.gaussian_peak_maps(np.asarray(summed.values, dtype=float), x, threshold=threshold)
    out = []
    for key in ("center", "fwhm", "amplitude"):
        name = f"{key}_{dim}" if key != "amplitude" else "amplitude"
        out.append(
            new(
                obj,
                _image_like(summed, dim, found[key], name),
                f"gauss_{name}",
                meta={"quantity": name, "motor": dim},
            )
        )
    return out


def argmax_map(obj: DataObject, dim: str = "") -> DataObject:
    """The motor value of the brightest frame at every pixel."""
    from bexa.analysis.rocking import argmax_motor

    summed, dim, x = _along(obj, dim)
    values = argmax_motor(np.asarray(summed.values, dtype=float), x, axis=0)
    return new(
        obj,
        _image_like(summed, dim, values, f"argmax_{dim}"),
        f"argmax_{dim}",
        meta={"quantity": f"argmax_{dim}", "motor": dim},
    )


def orientation_map(
    a: DataObject,
    b: DataObject,
    span: float | None = None,
    intensity: DataObject | None = None,
) -> DataObject:
    """Two tilt maps as one colour picture: hue the direction, saturation the size of the tilt
    (fully saturated at ``span``), brightness the ``intensity`` map if given."""
    require(a, IMAGES, "Orientation Map")
    require(b, IMAGES, "Orientation Map")
    weight = None if intensity is None else intensity.array.values
    rgb = maps.orientation_rgb(a.array.values, b.array.values, span=span, intensity=weight)
    dims = tuple(str(d) for d in a.array.dims)
    coords = {d: a.array.coords[d] for d in dims if d in a.array.coords}
    data = xr.DataArray(
        rgb, dims=(*dims, "rgb"), coords={**coords, "rgb": ["r", "g", "b"]}, name="orientation"
    )
    parents = [a.name, b.name] + ([intensity.name] if intensity is not None else [])
    return new(
        a,
        data,
        "orientation",
        kind="rgb",
        parents=parents,
        meta={"hue": f"direction of ({a.name}, {b.name})"},
    )


def kam_map(a: DataObject, b: DataObject | None = None, size: int = 3) -> DataObject:
    """Kernel average misorientation: the mean tilt difference to the neighbours of each pixel."""
    require(a, IMAGES, "KAM")
    values = maps.kam(a.array.values, None if b is None else b.array.values, int(size))
    parents = [a.name] + ([b.name] if b is not None else [])
    return new(
        a,
        a.array.copy(data=values).rename("kam"),
        "kam",
        parents=parents,
        meta={"quantity": "kernel average misorientation", "kernel": int(size)},
    )


def strain(
    obj: DataObject,
    kind: str = "theta",
    reference: float | None = None,
    theta_deg: float | None = None,
) -> DataObject:
    """Lattice strain from an angle (theta, 2theta) or energy centre-of-mass map."""
    require(obj, IMAGES, "Strain")
    values = maps.strain_map(obj.array.values, reference, kind, theta_deg)
    return new(
        obj,
        obj.array.copy(data=values).rename("strain"),
        "strain",
        meta={"quantity": "strain", "strain_from": kind},
    )


def align_frames(obj: DataObject, dim: str = "", max_shift: int = 5) -> DataObject:
    """Drift correction: every frame along ``dim`` shifted onto the first (phase correlation)."""
    from bexa.analysis.registration import align_stack

    summed, dim, _ = _along(obj, dim)
    frames = np.nan_to_num(np.asarray(summed.values, dtype=float))
    aligned, shifts = align_stack(frames, reference=0, max_shift=int(max_shift))
    data = summed.copy(data=aligned)
    return new(obj, data, "aligned", meta={"shifts_px": shifts.tolist(), "aligned_along": dim})


def segment(a: DataObject, b: DataObject, clusters: int = 4, method: str = "kmeans") -> DataObject:
    """Grains or domains: pixels clustered by their two tilt maps (k-means or DBSCAN)."""
    require(a, IMAGES, "Segment")
    require(b, IMAGES, "Segment")
    from bexa.analysis import segmentation

    if method == "kmeans":
        labels = segmentation.kmeans_com(a.array.values, b.array.values, n_clusters=int(clusters))
    elif method == "dbscan":
        labels = segmentation.dbscan_map(a.array.values, b.array.values)
    else:
        raise ValueError("method must be kmeans or dbscan")
    return new(
        a,
        a.array.copy(data=labels).rename("labels"),
        "segments",
        parents=[a.name, b.name],
        meta={"quantity": "cluster labels", "method": method},
    )


def gradient_maps(obj: DataObject) -> list[DataObject]:
    """The derivatives of a map along its two dims (per coordinate unit)."""
    require(obj, IMAGES, "Gradients")
    da = obj.array
    dy, dx = (str(d) for d in da.dims)
    values = np.asarray(da.values, dtype=float)
    yc, xc = obj.coord(dy).astype(float), obj.coord(dx).astype(float)
    grad_y = np.gradient(values, yc, axis=0) if yc.size > 1 else np.zeros_like(values)
    grad_x = np.gradient(values, xc, axis=1) if xc.size > 1 else np.zeros_like(values)
    return [
        new(obj, da.copy(data=grad_y).rename(f"d_d{dy}"), f"d{dy}", meta={"quantity": f"d/d{dy}"}),
        new(obj, da.copy(data=grad_x).rename(f"d_d{dx}"), f"d{dx}", meta={"quantity": f"d/d{dx}"}),
    ]
