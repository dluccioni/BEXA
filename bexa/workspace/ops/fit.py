"""Fit: peaks on curves, series of curves, and peak finding (:mod:`bexa.analysis.peakfit`)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from bexa.analysis import peakfit
from bexa.workspace.objects import DataObject
from bexa.workspace.ops._base import curve, curve_xy, new, require

__all__ = ["find_peaks", "fit_many", "fit_peaks", "fit_series", "peak_spec"]

FIT_POINTS = 400  # the best-fit curve is drawn on this many points (at least 4 per data point)


def peak_spec(
    shape: str, centers: Sequence[float] | None, peaks: Sequence[Mapping[str, Any]] | None
) -> list[dict[str, Any]] | None:
    """The peaks to fit: given in full (the Fit tab), at ``centers``, or ``None`` (guessed)."""
    if peaks:
        return [dict(p) for p in peaks]
    if centers:
        return [{"shape": shape, "center": float(c)} for c in centers]
    return None


def _fit_objects(
    obj: DataObject, fit: peakfit.PeakFit, dim: str, components: bool
) -> list[DataObject]:
    fine = np.linspace(fit.x.min(), fit.x.max(), max(FIT_POINTS, 4 * fit.x.size))
    summary = fit.to_dict()
    best = curve(fine, fit.evaluate(fine), dim, "fit")
    out = [new(obj, best, "fit", meta={"fit": summary, "fit_of": obj.name, "quantity": "fit"})]
    table = fit.table()
    table["r2"] = fit.r2
    table["reduced_chi2"] = fit.reduced_chi2
    out.append(
        new(obj, table, "fit_params", kind="table", meta={"fit": summary, "fit_of": obj.name})
    )
    if components:
        for key, values in fit.components(fine).items():
            out.append(
                new(
                    obj,
                    curve(fine, values, dim, key),
                    f"fit_{key}",
                    meta={"fit_of": obj.name, "quantity": key},
                )
            )
    return out


def fit_peaks(
    obj: DataObject,
    shape: str = "gaussian",
    n_peaks: int = 1,
    background: str = "linear",
    centers: Sequence[float] | None = None,
    x_range: Sequence[float | None] | None = None,
    use_sigma: bool = True,
    components: bool = False,
    peaks: Sequence[Mapping[str, Any]] | None = None,
    background_params: Mapping[str, Any] | None = None,
) -> list[DataObject]:
    """Fit peaks on a background to a curve: the best fit (a curve) and its parameters (a table).

    ``centers`` places the peaks (else the ``n_peaks`` most prominent are
    guessed); ``peaks`` gives every parameter with bounds and fixed flags (the
    Fit tab). The uncertainties of the curve weight the fit when ``use_sigma``.
    """
    dim, x, y, sigma = curve_xy(obj)
    fit = peakfit.fit_peaks(
        x,
        y,
        peak_spec(shape, centers, peaks),
        background,
        sigma if use_sigma else None,
        x_range,
        n_peaks=int(n_peaks),
        shape=shape,
        background_params=background_params,
    )
    return _fit_objects(obj, fit, dim, components)


def fit_series(
    obj: DataObject,
    along: str = "",
    shape: str = "gaussian",
    n_peaks: int = 1,
    background: str = "linear",
    centers: Sequence[float] | None = None,
    x_range: Sequence[float | None] | None = None,
    warm_start: bool = True,
    peaks: Sequence[Mapping[str, Any]] | None = None,
) -> list[DataObject]:
    """Fit every curve of a map (or volume) along the dim ``along``.

    Gives a table with one row per curve and, for every peak, its centre,
    width, height and area as objects over the other dims (a curve for a
    map). Each fit starts from the previous good one with ``warm_start``.
    """
    require(obj, ("map", "volume", "stack", "image"), "Fit Series")
    da = obj.array
    along = along or str(da.dims[-1])
    if along not in da.dims:
        raise ValueError(f"{obj.name} has no dim {along!r}; its dims: {obj.dims}")
    others = [str(d) for d in da.dims if d != along]
    ordered = da.transpose(*others, along)
    rows = np.asarray(ordered.values, dtype=float).reshape(-1, da.sizes[along])
    sigmas = None
    if obj.sigma is not None:
        sigmas = np.asarray(obj.sigma.transpose(*others, along).values, float).reshape(rows.shape)
    x = obj.coord(along).astype(float)
    fits = peakfit.fit_series(
        x,
        rows,
        peak_spec(shape, centers, peaks),
        background,
        sigmas,
        warm_start,
        x_range=x_range,
        n_peaks=int(n_peaks),
        shape=shape,
    )
    index = pd.MultiIndex.from_product([obj.coord(d) for d in others], names=others)
    table = peakfit.series_table(fits)
    for level, name in enumerate(others):
        table.insert(level, name, index.get_level_values(level))
    table = table.drop(columns=["index"])
    out = [new(obj, table, f"fits_{along}", kind="table", meta={"fit_along": along})]
    shape_others = tuple(da.sizes[d] for d in others)
    coords = {d: ordered.coords[d] for d in others if d in ordered.coords}
    first = next((f for f in fits if f is not None), None)
    if first is not None:
        for k in range(1, len(first.shapes) + 1):
            for quantity in ("center", "fwhm", "amplitude", "area"):
                column = f"p{k}_{quantity}"
                if column not in table:
                    continue
                values = table[column].to_numpy(dtype=float).reshape(shape_others)
                data = xr.DataArray(values, dims=others, coords=coords, name=column)
                errors = table.get(f"{column}_err")
                sigma = None if errors is None else errors.to_numpy(float).reshape(shape_others)
                out.append(
                    new(
                        obj,
                        data,
                        column,
                        sigma=sigma,
                        meta={"quantity": column, "fit_along": along},
                    )
                )
    return out


def fit_many(
    objects: Sequence[DataObject],
    shape: str = "gaussian",
    n_peaks: int = 1,
    background: str = "linear",
    centers: Sequence[float] | None = None,
    x_range: Sequence[float | None] | None = None,
    use_sigma: bool = True,
) -> DataObject:
    """Fit the same model to several curves: one table, a row per curve."""
    rows = []
    for obj in objects:
        _, x, y, sigma = curve_xy(obj)
        try:
            fit = peakfit.fit_peaks(
                x,
                y,
                peak_spec(shape, centers, None),
                background,
                sigma if use_sigma else None,
                x_range,
                n_peaks=int(n_peaks),
                shape=shape,
            )
            rows.append({"object": obj.name, **fit.report()})
        except (ValueError, RuntimeError) as exc:
            rows.append({"object": obj.name, "success": False, "error": str(exc)})
    return new(
        objects[0], pd.DataFrame(rows), "fits", kind="table", parents=[o.name for o in objects]
    )


def find_peaks(obj: DataObject, n: int = 5, prominence: float = 0.0) -> DataObject:
    """The most prominent maxima of a curve: position, height, width (FWHM) and prominence."""
    from scipy.signal import find_peaks as scipy_find_peaks
    from scipy.signal import peak_widths

    dim, x, y, _ = curve_xy(obj)
    order = np.argsort(x)
    x, y = x[order], np.nan_to_num(y[order])
    found, props = scipy_find_peaks(y, prominence=float(prominence))
    best = found[np.argsort(props["prominences"])[::-1][: int(n)]] if found.size else found
    best = np.sort(best)
    widths = peak_widths(y, best, rel_height=0.5)[0] if best.size else np.array([])
    step = float(np.median(np.diff(x))) if x.size > 1 else 1.0
    prominences = dict(zip(found.tolist(), props["prominences"].tolist(), strict=True))
    table = pd.DataFrame(
        {
            dim: x[best],
            "height": y[best],
            "fwhm": widths * abs(step),
            "prominence": [prominences[int(i)] for i in best],
        }
    )
    return new(obj, table, "peaks", kind="table")
