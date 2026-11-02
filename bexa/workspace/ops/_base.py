"""What the operations share: making results, checking kinds, ranges and uncertainties."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import xarray as xr

from bexa.core.roi import ROI
from bexa.core.structure import BINNED_PIXEL_DIMS, PIXEL_DIMS
from bexa.workspace.objects import DataObject, has_pixel_dims

ANY = ("stack", "image", "map", "curve", "volume")
DETECTOR = ("stack", "image")
IMAGES = ("image", "map")
SCANS = ("scan",)
REDUCE_HOWS = ("sum", "mean", "max", "min", "std", "median")


def new(
    source: DataObject,
    data: Any,
    suffix: str,
    *,
    sigma: Any = None,
    kind: str = "",
    meta: Mapping[str, Any] | None = None,
    parents: Sequence[str] | None = None,
) -> DataObject:
    """A result made from ``source``: named ``<source>_<suffix>``, its metadata passed on."""
    if isinstance(data, xr.DataArray) and isinstance(sigma, np.ndarray):
        sigma = xr.DataArray(sigma, dims=data.dims, coords=data.coords)
    inherited = {k: v for k, v in source.meta.items() if k not in ("fit", "fit_of")}
    return DataObject(
        name=f"{source.name}_{suffix}" if suffix else source.name,
        data=data,
        kind=kind,
        sigma=sigma,
        meta={**inherited, **(meta or {})},
        parents=list(parents) if parents is not None else [source.name],
    )


def require(obj: DataObject, kinds: Iterable[str], what: str = "this operation") -> None:
    """Raise a ``TypeError`` naming the kinds ``what`` takes when ``obj`` is none of them."""
    kinds = tuple(kinds)
    if obj.kind not in kinds:
        wanted = " or ".join(kinds)
        found = f"{obj.name} is {_article(obj.kind)} {obj.kind}"
        raise TypeError(f"{what} takes {_article(wanted)} {wanted}; {found}")


def _article(word: str) -> str:
    return "an" if word[:1] in "aeiou" else "a"


def pixel_dims(da: xr.DataArray) -> tuple[str, ...]:
    """The pixel dims of ``da`` (its last two when they are pixels), else ``()``."""
    return tuple(str(d) for d in da.dims[-2:]) if has_pixel_dims(da.dims) else ()


def other_dims(da: xr.DataArray) -> tuple[str, ...]:
    """The dims of ``da`` that are not pixels (motors, energy, scan...)."""
    pixels = pixel_dims(da)
    return tuple(str(d) for d in da.dims if d not in pixels)


def reduce_over(da: xr.DataArray, dims: Sequence[str], how: str = "sum") -> xr.DataArray:
    """``da`` summed (or averaged, ...) over ``dims``, NaN skipped."""
    dims = [d for d in dims if d in da.dims]
    if not dims:
        return da
    if how not in REDUCE_HOWS:
        raise ValueError(f"how must be one of {REDUCE_HOWS}, not {how!r}")
    reducer = getattr(da, how)
    return reducer(dims, skipna=True, keep_attrs=True)


def sigma_over(sigma: xr.DataArray | None, dims: Sequence[str], how: str, n: int) -> Any:
    """The uncertainty of a sum or mean over ``dims`` (independent errors), else ``None``."""
    if sigma is None or how not in ("sum", "mean"):
        return None
    dims = [d for d in dims if d in sigma.dims]
    total = np.sqrt((sigma**2).sum(dims, skipna=True))
    return total / n if how == "mean" else total


def poisson(counts: xr.DataArray) -> xr.DataArray:
    """``sqrt(|counts|)``: the uncertainty of summed detector counts."""
    return np.sqrt(np.abs(counts))


def select_ranges(da: xr.DataArray, roi: ROI | None) -> xr.DataArray:
    """``da`` cut to the ranges of ``roi`` on the dims it has.

    Pixel ranges are half-open full-resolution pixel indices (bexa's ``y``/``x``
    coordinates); other ranges are inclusive coordinate values, or indices for
    the ROI's index dims.
    """
    if roi is None or not roi:
        return da
    return da.isel(range_slices(da, roi))


def range_slices(da: xr.DataArray, roi: ROI) -> dict[str, slice]:
    """The index slice of every dim of ``da`` that ``roi`` limits (see :func:`select_ranges`)."""
    pixel_names = (*PIXEL_DIMS, *BINNED_PIXEL_DIMS)
    slices: dict[str, slice] = {}
    for dim, (lo, hi) in roi.ranges.items():
        if dim not in da.dims:
            continue
        if dim in roi.index_dims and dim not in pixel_names:
            slices[dim] = slice(lo, hi)
            continue
        if dim in da.coords:
            values = np.asarray(da.coords[dim].values, dtype=float)
        else:
            values = np.arange(da.sizes[dim], dtype=float)
        keep = np.ones(values.shape, dtype=bool)
        if lo is not None:
            keep &= values >= float(lo)
        if hi is not None:
            keep &= values < float(hi) if dim in pixel_names else values <= float(hi)
        index = np.flatnonzero(keep)
        if index.size == 0:
            raise ValueError(f"the ROI range {dim} {lo}..{hi} holds no point of {dim}")
        slices[dim] = slice(int(index[0]), int(index[-1]) + 1)
    return slices


def windowed(obj: DataObject, roi: ROI | None) -> tuple[xr.DataArray, Any]:
    """The data of an array object and its sigma, both cut to ``roi``."""
    data = select_ranges(obj.array, roi)
    sigma = None if obj.sigma is None else select_ranges(obj.sigma, roi)
    return data, sigma


def numeric(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float)


def trapezoid(y: Any, x: Any) -> float:
    """The trapezoid integral (``numpy.trapezoid``, ``numpy.trapz`` before numpy 2)."""
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(integrate(y, x))


def curve_xy(obj: DataObject) -> tuple[str, np.ndarray, np.ndarray, np.ndarray | None]:
    """``(dim, x, y, sigma)`` of a curve object."""
    require(obj, ("curve",), "a curve tool")
    da = obj.array
    dim = str(da.dims[0])
    x = numeric(obj.coord(dim))
    sigma = None if obj.sigma is None else numeric(obj.sigma.values)
    return dim, x, numeric(da.values), sigma


def curve(
    x: Any,
    y: Any,
    dim: str,
    name: str = "",
    attrs: Mapping[str, Any] | None = None,
    coord_attrs: Mapping[str, Any] | None = None,
) -> xr.DataArray:
    """A 1-D DataArray along ``dim`` with coordinate ``x``."""
    coord = xr.DataArray(numeric(x), dims=(dim,), attrs=dict(coord_attrs or {}))
    return xr.DataArray(
        numeric(y), dims=(dim,), coords={dim: coord}, name=name, attrs=dict(attrs or {})
    )


def like(da: xr.DataArray, values: Any, name: str | None = None) -> xr.DataArray:
    """``values`` with the dims, coordinates and attrs of ``da``."""
    return xr.DataArray(
        np.asarray(values),
        dims=da.dims,
        coords=da.coords,
        attrs=dict(da.attrs),
        name=name if name is not None else da.name,
    )


def pairs(values: Sequence[float] | None, what: str = "values") -> list[tuple[float, float]]:
    """``[1, 2, 9, 10]`` as ``[(1, 2), (9, 10)]``."""
    if not values:
        return []
    if len(values) % 2:
        raise ValueError(f"{what}: give pairs, low and high: 1, 2, 9, 10")
    return [(float(values[i]), float(values[i + 1])) for i in range(0, len(values), 2)]
