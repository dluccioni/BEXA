"""Multi-scan: process many scans at once.

A scan set is the multi-scan object. :func:`stack_scans` runs one streaming
pass per scan (in parallel processes with ``workers``) and stacks the results
on the positioners that differ between the scans: the z-stacks, energy series
and temperature series of a beamtime. Any single-scan operation also runs on
every scan of a set (the workspace maps it, see :func:`stack_per_scan`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import xarray as xr

from bexa.core.roi import ROI
from bexa.workspace.objects import DataObject, ScanSet
from bexa.workspace.ops._base import new, require

__all__ = [
    "STACK_QUANTITIES",
    "layers",
    "same_grid",
    "scan_table",
    "series",
    "stack_per_scan",
    "stack_scans",
    "subset",
]

STACK_QUANTITIES = ("sum", "max", "mean", "com", "stats", "preview", "argmax", "energy_com")


def _accumulators(
    quantities: Sequence[str], sigma: float, preview_bin: int, rois: Mapping[str, ROI] | None
) -> list[Any]:
    from bexa.core import reductions as acc

    made: list[Any] = []
    for q in quantities:
        if q == "sum":
            made.append(acc.Sum())
        elif q == "max":
            made.append(acc.Max())
        elif q == "mean":
            made.append(acc.Mean())
        elif q == "com":
            made.append(acc.MotorCOM(sigma=sigma))
        elif q == "stats":
            made.append(acc.FrameStats())
        elif q == "preview":
            made.append(acc.Preview(downsample=int(preview_bin)))
        elif q == "argmax":
            made.append(acc.ArgmaxMotor())
        elif q == "energy_com":
            made.append(acc.EnergyCOM())
        else:
            raise ValueError(f"unknown quantity {q!r}; choose from {STACK_QUANTITIES}")
    if rois:
        made.append(acc.RoiIntegral(dict(rois)))
    return made


def stack_scans(
    obj: DataObject,
    quantities: Sequence[str] = ("sum", "com"),
    dim: str = "auto",
    sigma: float = 3.0,
    roi: ROI | None = None,
    downsample: int | None = None,
    preview_bin: int = 8,
    rois: Mapping[str, ROI] | None = None,
    workers: int | None = None,
) -> list[DataObject]:
    """Reduce every scan of a set the same way and stack the results on a grid of scans.

    ``quantities``: ``sum``, ``max``, ``mean`` (images), ``com`` (centre-of-mass
    and width maps per motor), ``stats`` (per-frame statistics), ``preview``
    (the frames binned ``preview_bin`` times), ``argmax``, ``energy_com``;
    ``rois`` adds every ROI's curve per scan. ``dim="auto"`` stacks on the
    positioners that differ between the scans (a height, an energy, a
    temperature); a name stacks on that positioner, ``"scan"`` on the order.
    ``workers`` reduces several scans at once in separate processes.
    """
    require(obj, ("scanset",), "Stack Scans")
    from bexa.core.dataset import stack

    if isinstance(quantities, str):  # "sum,com" from the form
        quantities = [q.strip() for q in quantities.split(",") if q.strip()]
    accumulators = _accumulators(quantities, sigma, preview_bin, rois)
    options: dict[str, Any] = {"roi": roi}
    if downsample:
        options["downsample"] = {"y": int(downsample), "x": int(downsample)}
    result = stack(obj.data.scans, accumulators, dim=dim or "auto", workers=workers, **options)
    out = []
    for name in result.data_vars:
        if "_block_" in str(name):  # per-frame statistics for the browsers, not results
            continue
        data = result[str(name)]
        out.append(
            new(
                obj,
                data,
                str(name),
                meta={
                    "quantity": str(name),
                    "stack_dims": list(result.attrs.get("stack_dims", [])),
                },
            )
        )
    return out


def series(obj: DataObject, dim: str = "auto") -> DataObject:
    """The scans of a set joined as one scan with a new leading dim (they must share a grid).

    ``dim``: ``"auto"`` (the energy when every scan has its own, else ``scan``),
    ``"energy"``, ``"scan"`` or a positioner recorded in every scan.
    """
    require(obj, ("scanset",), "Series")
    joined = obj.data.series(None if dim in ("", "auto") else dim)
    return DataObject(
        f"{obj.name}_series", joined, parents=[obj.name], meta={"scans": list(obj.data.labels)}
    )


def scan_table(obj: DataObject) -> DataObject:
    """One row per scan of a set: type, motors, grid, frames, energy and every positioner."""
    require(obj, ("scanset",), "Scan Table")
    return new(obj, obj.data.table(), "table", kind="table")


def subset(obj: DataObject, indices: Sequence[int]) -> DataObject:
    """The scans at ``indices`` (from 0, in the set's order) as a scan set of their own."""
    require(obj, ("scanset",), "Subset")
    if not indices:
        raise ValueError("Indices: give the positions of the scans to keep, from 0")
    picked = obj.data.subset([int(i) for i in indices])
    return DataObject(f"{obj.name}_sub", picked, parents=[obj.name])


def layers(obj: DataObject, dim: str) -> DataObject:
    """A scan split along one of its motor dims, one layer per point, as a scan set.

    A z-scan ``uz x chi x mu`` becomes its heights, which then stack like the
    scans of a z-stack.
    """
    require(obj, ("scan",), "Layers")
    scan = obj.data
    parts = scan.layers(dim)
    values = scan.structure.coords[dim]
    labels = [f"{obj.name} {dim}={float(v):.5g}" for v in values]
    return DataObject(f"{obj.name}_{dim}_layers", ScanSet(parts, labels), parents=[obj.name])


def stack_per_scan(
    scan_set: ScanSet, per_scan: Sequence[Sequence[DataObject]], base: str
) -> list[DataObject]:
    """The results of one operation run on every scan of a set, stacked where they can be.

    ``per_scan[i]`` holds the results for scan ``i``, named ``<base>_<suffix>``
    (the operation ran on each scan under the set's name ``base``). Result ``k``
    of every scan is stacked along the set's axis (the positioner that
    differs, else ``scan``) when they are arrays on one grid (see
    :func:`same_grid`); otherwise each scan's result is kept on its own, named
    after its scan.
    """
    axis, values = scan_set.axis()
    out: list[DataObject] = []
    counts = {len(r) for r in per_scan}
    if len(counts) != 1:  # the scans gave different numbers of results: keep them all
        for label, results in zip(scan_set.labels, per_scan, strict=True):
            out.extend(_relabelled(r, label, base) for r in results)
        return out
    for k in range(counts.pop()):
        column = [results[k] for results in per_scan]
        first = column[0]
        if not same_grid(column) or axis in first.dims:
            out.extend(
                _relabelled(c, label, base)
                for c, label in zip(column, scan_set.labels, strict=True)
            )
            continue
        data = xr.concat([c.array for c in column], dim=axis, coords="minimal", compat="override")
        data = data.assign_coords({axis: np.asarray(values, dtype=float)})
        sigma = None
        if all(c.sigma is not None for c in column):
            sigma = xr.concat(
                [c.sigma for c in column], dim=axis, coords="minimal", compat="override"
            )
            sigma = sigma.assign_coords({axis: np.asarray(values, dtype=float)})
        per_scan_keys = ("scan", "source", "positions", "scan_label")  # true of one scan only
        meta = {k2: v for k2, v in first.meta.items() if k2 not in per_scan_keys}
        meta.update({"scans": list(scan_set.labels), "stacked_on": axis})
        out.append(DataObject(first.name, data, sigma=sigma, meta=meta, parents=[base]))
    return out


def same_grid(objects: Sequence[DataObject]) -> bool:
    """True when the objects are arrays with the first one's dims, shape and coordinates.

    Numeric coordinates may differ by a quarter of their smallest step (the
    readback jitter of one grid); rocking curves over other ranges are other
    grids, which stacking would put on the first one's axis.
    """
    first = objects[0]
    if not all(o.is_array and o.dims == first.dims and o.shape == first.shape for o in objects):
        return False
    for dim in first.dims:
        if dim not in first.array.coords:
            continue
        ref = np.asarray(first.array.coords[dim].values)
        others = [
            np.asarray(o.array.coords[dim].values) for o in objects[1:] if dim in o.array.coords
        ]
        if ref.dtype.kind not in "fiu":
            if not all(np.array_equal(other, ref) for other in others):
                return False
            continue
        steps = np.abs(np.diff(ref.astype(float)))
        steps = steps[steps > 0]
        tolerance = 0.25 * float(steps.min()) if steps.size else 0.0
        for other in others:
            if not np.allclose(other.astype(float), ref.astype(float), rtol=0.0, atol=tolerance):
                return False
    return True


def _relabelled(obj: DataObject, label: str, base: str) -> DataObject:
    """A per-scan result named after its scan instead of the set."""
    suffix = obj.name[len(base) + 1 :] if obj.name.startswith(base + "_") else obj.name
    clean = label.replace(" #", "_").replace(" ", "_").replace("=", "")
    obj.name = f"{clean}_{suffix}"
    return obj
