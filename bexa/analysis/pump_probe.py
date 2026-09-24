"""Laser on/off analysis: splitting shots, differential signals, binning and combining runs.

Ports the scalar logic of the PAL-XFEL scripts (``Scan_Combiner.py``) and the
LCLS notebook binning (``kieran_analysis``) as vectorised functions that take
numpy or cupy arrays.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr

from bexa.core.backend import array_module
from bexa.core.provenance import build_attrs, parameters_of, to_json

LASER_COORD = np.array(["off", "on"])

__all__ = [
    "bin_by_axis",
    "bin_lcls",
    "bin_shots",
    "combine_runs",
    "differential",
    "linearity_check",
    "scan_axis_of",
    "split_on_off",
]


def split_on_off(table: Any, flag_key: str, dropna: bool = True) -> tuple[Any, Any]:
    """Rows with the laser on (flag 1) and off (flag 0); NaN flags belong to neither.

    With ``dropna`` every row holding a NaN in any column is dropped, which is
    what the legacy cube builder did before averaging the ROI/I0 signal.
    """
    flags = table[flag_key]
    on = table[flags == 1]
    off = table[flags == 0]
    if dropna:
        on, off = on.dropna(), off.dropna()
    return on, off


def differential(on: Any, off: Any, mode: str = "diff") -> Any:
    """``on - off`` (diff), ``on / off`` (ratio) or ``100 (on - off) / off`` (percent)."""
    if mode == "diff":
        return on - off
    if mode == "ratio":
        return on / off
    if mode == "percent":
        return 100.0 * (on - off) / off
    raise ValueError(f"unknown mode {mode!r}; use diff, ratio or percent")


def bin_by_axis(
    axis: Any, values: Sequence[Any], decimals: int = 2
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Average arrays over the points whose axis value agrees after rounding.

    Port of ``Scan_Combiner.py``: the bins are the unique values of
    ``round(axis, decimals)``; every output is the mean of its members along the
    leading axis (NaN propagates) and the bin coordinate is the mean of the
    member axis values.

    Returns
    -------
    centres, binned
        The bin coordinates and one binned array per input array.
    """
    axis = np.asarray(axis, dtype=float)
    rounded = np.round(axis, decimals)
    bins, inverse = np.unique(rounded, return_inverse=True)
    inverse = inverse.reshape(-1)
    members = [np.flatnonzero(inverse == i) for i in range(len(bins))]
    centres = np.array([axis[m].mean() for m in members])
    binned = []
    for array in values:
        array = np.asarray(array)
        if array.shape[0] != axis.shape[0]:
            raise ValueError(f"array has {array.shape[0]} points, the axis {axis.shape[0]}")
        binned.append(np.stack([array[m].mean(axis=0) for m in members]))
    return centres, binned


def scan_axis_of(ds: xr.Dataset) -> str:
    """The scan dimension of a cube dataset (``attrs["scan_axis"]`` or the first non-laser dim)."""
    if "scan_axis" in ds.attrs:
        return str(ds.attrs["scan_axis"])
    frames = ds["frames"]
    for d in frames.dims:
        if d not in ("laser", "y", "x"):
            return str(d)
    raise ValueError("the dataset has no scan dimension")


def combine_runs(
    datasets: Sequence[xr.Dataset], axis: str | None = None, decimals: int = 2
) -> xr.Dataset:
    """Merge cubes of several runs onto one axis, averaging points that coincide.

    Replaces ``Scan_Combiner.py``: every variable with the scan dim is binned
    with :func:`bin_by_axis`; coordinates along the dim are averaged per bin.
    """
    if not datasets:
        raise ValueError("no datasets to combine")
    dim = axis or scan_axis_of(datasets[0])
    if dim == "delays":
        dim = "delay"
    for ds in datasets:
        if dim not in ds.dims:
            raise ValueError(f"dataset has no dim {dim!r}; dims are {tuple(ds.dims)}")
    # keep only what every run has, so runs recorded with different scalars still combine
    common_vars = set.intersection(*[set(map(str, ds.data_vars)) for ds in datasets])
    common_coords = set.intersection(*[set(map(str, ds.coords)) for ds in datasets])
    trimmed = [
        ds.drop_vars([v for v in ds.variables if str(v) not in common_vars | common_coords])
        for ds in datasets
    ]
    merged = xr.concat(trimmed, dim=dim, combine_attrs="drop", coords="minimal")
    values = np.asarray(merged[dim].values, dtype=float)
    rounded = np.round(values, decimals)
    bins, inverse = np.unique(rounded, return_inverse=True)
    members = [np.flatnonzero(inverse.reshape(-1) == i) for i in range(len(bins))]

    def binned(var: xr.DataArray) -> xr.DataArray:
        pos = var.dims.index(dim)
        moved = np.moveaxis(np.asarray(var.values), pos, 0)
        out = np.stack([moved[m].mean(axis=0) for m in members])
        data = np.moveaxis(out, 0, pos)
        return xr.DataArray(data, dims=var.dims, attrs=var.attrs)

    data_vars = {}
    for name, var in merged.data_vars.items():
        data_vars[str(name)] = binned(var) if dim in var.dims else var
    coords: dict[str, Any] = {dim: np.array([values[m].mean() for m in members])}
    for name, coord in merged.coords.items():
        if name == dim:
            continue
        if dim in coord.dims and coord.dtype.kind in "iuf":
            coords[str(name)] = (coord.dims, binned(coord).values)
        elif dim not in coord.dims:
            coords[str(name)] = coord
    out = xr.Dataset(data_vars, coords=coords)
    files = [f for ds in datasets for f in _source_paths(ds)]
    out.attrs.update(
        build_attrs(
            source_files=files,
            parameters={"axis": dim, "decimals": decimals, "n_runs": len(datasets)},
        )
    )
    out.attrs["scan_axis"] = dim
    out.attrs["combined_runs"] = [ds.attrs.get("run") for ds in datasets]
    return out


def _source_paths(ds: xr.Dataset) -> list[str]:
    records = ds.attrs.get("source_files") or []
    paths = []
    for rec in records:
        if isinstance(rec, dict) and "path" in rec:
            paths.append(str(rec["path"]))
        elif isinstance(rec, str):
            paths.append(rec)
    return paths


def bin_shots(
    on: Any, off: Any, scanvar: Any, i0_on: Any, i0_off: Any
) -> tuple[np.ndarray, Any, Any, np.ndarray]:
    """Sum I0-normalised shots per unique scan value (the LCLS notebook binning).

    Shots where either I0 is zero are skipped. Returns the bin values, the
    summed on and off stacks ``(bins, y, x)`` and the number of shots per bin.
    """
    xp = array_module(on)
    scanvar = np.asarray(scanvar, dtype=float)
    i0_on_h = np.asarray(i0_on, dtype=float)
    i0_off_h = np.asarray(i0_off, dtype=float)
    bins = np.unique(scanvar)
    keep = (i0_on_h != 0) & (i0_off_h != 0)
    idx = np.minimum(np.searchsorted(bins, scanvar, side="left"), len(bins) - 1)
    shape = (len(bins), *on.shape[1:])
    sum_on = xp.zeros(shape, dtype=np.float64)
    sum_off = xp.zeros(shape, dtype=np.float64)
    counts = np.zeros(len(bins), dtype=int)
    for b in range(len(bins)):
        sel = np.flatnonzero(keep & (idx == b))
        if sel.size == 0:
            continue
        counts[b] = sel.size
        sel_dev = xp.asarray(sel)
        sum_on[b] = (on[sel_dev] / xp.asarray(i0_on_h[sel])[:, None, None]).sum(axis=0)
        sum_off[b] = (off[sel_dev] / xp.asarray(i0_off_h[sel])[:, None, None]).sum(axis=0)
    return bins, sum_on, sum_off, counts


def bin_lcls(ds: xr.Dataset, axis: str = "scanvar") -> xr.Dataset:
    """Bin an LCLS per-shot dataset (see :func:`bexa.io.cube.load_lcls_cube`) into a cube."""
    frames = ds["frames"]
    i0 = ds["i0"]
    scanvar = ds.coords[axis].values
    bins, sum_on, sum_off, counts = bin_shots(
        frames.sel(laser="on").values,
        frames.sel(laser="off").values,
        scanvar,
        i0.sel(laser="on").values,
        i0.sel(laser="off").values,
    )
    keep = (i0.sel(laser="on").values != 0) & (i0.sel(laser="off").values != 0)
    idx = np.minimum(np.searchsorted(bins, scanvar, side="left"), len(bins) - 1)
    i0_sum = np.zeros((2, len(bins)))
    for state, values in enumerate((i0.sel(laser="off").values, i0.sel(laser="on").values)):
        np.add.at(i0_sum[state], idx[keep], values[keep])
    out = xr.Dataset(
        {
            "frames": xr.DataArray(np.stack([sum_off, sum_on]), dims=("laser", axis, "y", "x")),
            "i0": xr.DataArray(i0_sum, dims=("laser", axis)),
            "n_shots": xr.DataArray(counts, dims=(axis,)),
        },
        coords={"laser": LASER_COORD, axis: bins},
    )
    out.attrs.update(ds.attrs)
    out.attrs["scan_axis"] = axis
    out.attrs["parameters"] = to_json({**parameters_of(ds.attrs), "binned_by": axis})
    return out


def linearity_check(det: Any, i0: Any, laser: Any | None = None) -> dict[str, dict[str, float]]:
    """Slope, intercept and R^2 of detector signal against I0, per laser state.

    The legacy notebooks plotted this to spot saturation and bad shots.
    """
    det = np.asarray(det, dtype=float)
    i0 = np.asarray(i0, dtype=float)
    groups: dict[str, np.ndarray] = {"all": np.ones(det.shape, dtype=bool)}
    if laser is not None:
        laser = np.asarray(laser, dtype=float)
        groups = {"on": laser == 1, "off": laser == 0}
    out: dict[str, dict[str, float]] = {}
    for name, mask in groups.items():
        mask = mask & np.isfinite(det) & np.isfinite(i0)
        if mask.sum() < 2:
            out[name] = {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "n": int(mask.sum())}
            continue
        slope, intercept = np.polyfit(i0[mask], det[mask], 1)
        predicted = slope * i0[mask] + intercept
        residual = np.sum((det[mask] - predicted) ** 2)
        total = np.sum((det[mask] - det[mask].mean()) ** 2)
        r2 = 1.0 - residual / total if total > 0 else np.nan
        out[name] = {
            "slope": float(slope),
            "intercept": float(intercept),
            "r2": float(r2),
            "n": int(mask.sum()),
        }
    return out
