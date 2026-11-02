"""Combine: several objects in, new ones out (arithmetic, averages, joins, stitching, mosaics)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr

from bexa.analysis import curves
from bexa.workspace.objects import DataObject, ScanSet
from bexa.workspace.ops._base import ANY, IMAGES, curve, curve_xy, new, require

__all__ = [
    "INTERP_METHODS",
    "OPERATORS",
    "align_images",
    "align_peaks",
    "arithmetic",
    "average",
    "concatenate",
    "merge_scan_sets",
    "mosaic",
    "resample_onto",
    "stack_objects",
    "stitch_curves",
]

OPERATORS = ("+", "-", "*", "/")
INTERP_METHODS = ("linear", "nearest", "cubic")


def _on_grid_of(a: xr.DataArray, b: xr.DataArray, interpolate: bool) -> xr.DataArray:
    """``b`` on the coordinates of ``a`` where they share dims (interpolated, or by position)."""
    shared = [d for d in b.dims if d in a.dims]
    if not shared:
        return b
    same = all(
        d not in a.coords
        or d not in b.coords
        or (a.sizes[d] == b.sizes[d] and np.allclose(a.coords[d].values, b.coords[d].values))
        for d in shared
    )
    if same:
        return b.assign_coords({d: a.coords[d] for d in shared if d in a.coords})
    if interpolate:
        return b.interp({d: a.coords[d] for d in shared if d in a.coords and d in b.coords})
    if all(a.sizes[d] == b.sizes[d] for d in shared):  # by position
        return b.assign_coords({d: a.coords[d] for d in shared if d in a.coords})
    raise ValueError(f"the objects differ along {shared}; tick Interpolate to put B on A's grid")


def arithmetic(
    a: DataObject,
    op: str = "-",
    b: DataObject | None = None,
    number: float | None = None,
    interpolate: bool = True,
) -> DataObject:
    """``A op B`` with uncertainties: B is another object or a number.

    Objects are matched by dim names: an image with a stack applies to every
    frame, a curve along a stack's motor dim gives a value per frame; curves
    on other x values are interpolated onto A's when ``interpolate``.
    """
    require(a, ANY, "Arithmetic")
    if op not in OPERATORS:
        raise ValueError(f"op must be one of {OPERATORS}")
    da = a.array
    sa = a.sigma
    if b is not None:
        require(b, ANY, "B")
        db = _on_grid_of(da, b.array, interpolate)
        sb = None if b.sigma is None else _on_grid_of(da, b.sigma, interpolate)
        label = b.name
    elif number is not None:
        db, sb, label = float(number), None, f"{number:g}"
    else:
        raise ValueError("B: choose an object or give a number")
    with np.errstate(invalid="ignore", divide="ignore"):
        result = {"+": da + db, "-": da - db, "*": da * db, "/": da / db}[op]
        sigma = None
        if sa is not None or sb is not None:
            ea = sa if sa is not None else 0.0
            eb = sb if sb is not None else 0.0
            if op in "+-":
                sigma = np.sqrt(ea**2 + eb**2)
            else:
                sigma = np.abs(result) * np.sqrt((ea / da) ** 2 + (eb / db) ** 2)
    if isinstance(result, xr.DataArray) and isinstance(sigma, xr.DataArray):
        sigma = sigma.broadcast_like(result).transpose(*result.dims)
    symbol = {"+": "plus", "-": "minus", "*": "times", "/": "over"}[op]
    parents = [a.name] + ([b.name] if b is not None else [])
    return new(
        a,
        result,
        f"{symbol}_{label}",
        sigma=sigma,
        meta={"operation": f"{a.name} {op} {label}"},
        parents=parents,
    )


def _aligned(objects: Sequence[DataObject], interpolate: bool = True) -> list[DataObject]:
    for obj in objects:
        require(obj, ANY, "this combination")
    first = objects[0].array
    out = []
    for obj in objects:
        data = _on_grid_of(first, obj.array, interpolate)
        sigma = None if obj.sigma is None else _on_grid_of(first, obj.sigma, interpolate)
        if data.shape != first.shape:
            raise ValueError(f"{obj.name} is {data.shape}, {objects[0].name} {first.shape}")
        out.append(DataObject(obj.name, data, sigma=sigma, meta=obj.meta))
    return out


def average(objects: Sequence[DataObject], how: str = "mean") -> DataObject:
    """The mean, sum or uncertainty-weighted mean of objects of one shape (repeated scans)."""
    if len(objects) < 2:
        raise ValueError("select two or more objects")
    parts = _aligned(objects)
    values = np.stack([np.asarray(p.array.values, dtype=float) for p in parts])
    sigmas = [p.sigma for p in parts]
    has_sigma = all(s is not None for s in sigmas)
    errors = np.stack([np.asarray(s.values, dtype=float) for s in sigmas]) if has_sigma else None
    n = values.shape[0]
    if how == "sum":
        result = np.nansum(values, axis=0)
        sigma = None if errors is None else np.sqrt(np.nansum(errors**2, axis=0))
    elif how == "mean":
        result = np.nanmean(values, axis=0)
        sigma = None if errors is None else np.sqrt(np.nansum(errors**2, axis=0)) / n
    elif how == "weighted":
        if errors is None:
            raise ValueError("a weighted mean needs the uncertainties of every object")
        weights = 1.0 / np.where(errors > 0, errors, np.inf) ** 2
        total = np.nansum(weights, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            result = np.nansum(values * weights, axis=0) / total
            sigma = 1.0 / np.sqrt(total)
    else:
        raise ValueError("how must be mean, sum or weighted")
    data = parts[0].array.copy(data=result)
    return new(
        objects[0],
        data,
        how,
        sigma=sigma,
        parents=[o.name for o in objects],
        meta={"combined": [o.name for o in objects]},
    )


def concatenate(objects: Sequence[DataObject], dim: str = "") -> DataObject:
    """Join objects along a dim they share (curves point after point, stacks frame after frame)."""
    if len(objects) < 2:
        raise ValueError("select two or more objects")
    for obj in objects:
        require(obj, ANY, "Join")
    dim = dim or str(objects[0].array.dims[0])
    joined = xr.concat([o.array for o in objects], dim=dim)
    sigma = None
    if all(o.sigma is not None for o in objects):
        sigma = xr.concat([o.sigma for o in objects], dim=dim)
    if joined.ndim == 1 and dim in joined.coords:
        order = np.argsort(joined.coords[dim].values, kind="stable")
        joined = joined.isel({dim: order})
        sigma = None if sigma is None else sigma.isel({dim: order})
    return new(objects[0], joined, "joined", sigma=sigma, parents=[o.name for o in objects])


def stack_objects(
    objects: Sequence[DataObject], dim: str = "index", values: Sequence[float] | None = None
) -> DataObject:
    """Objects of one shape stacked along a new leading dim: images to a stack, curves to a map."""
    if len(objects) < 2:
        raise ValueError("select two or more objects")
    parts = _aligned(objects)
    coords = np.asarray(values, dtype=float) if values else np.arange(len(parts), dtype=float)
    if coords.size != len(parts):
        raise ValueError(f"{coords.size} values for {len(parts)} objects")
    data = xr.concat([p.array for p in parts], dim=dim).assign_coords({dim: coords})
    sigma = None
    if all(p.sigma is not None for p in parts):
        sigma = xr.concat([p.sigma for p in parts], dim=dim).assign_coords({dim: coords})
    return new(
        objects[0],
        data,
        f"by_{dim}",
        sigma=sigma,
        parents=[o.name for o in objects],
        meta={"stacked": [o.name for o in objects]},
    )


def stitch_curves(
    objects: Sequence[DataObject], scale: bool = True, overlap: str = "average"
) -> DataObject:
    """One curve from curves of neighbouring ranges, each scaled to match where they overlap."""
    if len(objects) < 2:
        raise ValueError("select two or more curves")
    parts = [curve_xy(o) for o in objects]
    x, y, s = curves.stitch([(px, py, ps) for _, px, py, ps in parts], scale=scale, overlap=overlap)
    data = curve(x, y, parts[0][0], str(objects[0].array.name or ""))
    return new(objects[0], data, "stitched", sigma=s, parents=[o.name for o in objects])


def align_peaks(
    objects: Sequence[DataObject],
    method: str = "maximum",
    target: str = "first",
    value: float | None = None,
    x_range: Sequence[float | None] | None = None,
) -> list[DataObject]:
    """Move curves along x so their peaks coincide: at the first curve's peak, zero or a value."""
    if not objects:
        raise ValueError("select curves")
    found = []
    for obj in objects:
        dim, x, y, _ = curve_xy(obj)
        found.append(curves.peak_position(x, y, method, x_range))
    goal = {"first": found[0], "zero": 0.0}.get(target, value)
    if goal is None:
        raise ValueError("Value: give the x where the peaks go")
    out = []
    for obj, peak in zip(objects, found, strict=True):
        dim = str(obj.array.dims[0])
        x = obj.coord(dim).astype(float) + (goal - peak)
        sigma = None if obj.sigma is None else obj.sigma.assign_coords({dim: x})
        out.append(
            new(
                obj,
                obj.array.assign_coords({dim: x}),
                "aligned",
                sigma=sigma,
                meta={"shift": goal - peak},
            )
        )
    return out


def align_images(objects: Sequence[DataObject], max_shift: int | None = None) -> list[DataObject]:
    """Shift images onto the first by phase correlation (integer pixels)."""
    from scipy.ndimage import shift

    from bexa.analysis.registration import phase_correlation_shift

    for obj in objects:
        require(obj, IMAGES, "Align Images")
    reference = np.nan_to_num(np.asarray(objects[0].array.values, dtype=float))
    out = []
    for obj in objects:
        image = np.nan_to_num(np.asarray(obj.array.values, dtype=float))
        dy, dx = phase_correlation_shift(reference, image, max_shift)
        moved = shift(image, (dy, dx), order=0, mode="constant", cval=np.nan)
        out.append(
            new(
                obj,
                obj.array.copy(data=moved),
                "registered",
                meta={"shift_px": [dy, dx]},
                parents=[obj.name, objects[0].name],
            )
        )
    return out


def mosaic(
    objects: Sequence[DataObject],
    offsets: Sequence[float] | None = None,
    rows: str = "",
    columns: str = "",
    pixel_mm: float = 0.0,
    how: str = "mean",
) -> DataObject:
    """Images placed side by side: by pixel ``offsets`` (row, column per image) or by the stage
    positions in their metadata (``rows``/``columns`` motor names, ``pixel_mm`` at the sample)."""
    from bexa.analysis.registration import mosaic_from_positions

    for obj in objects:
        require(obj, IMAGES, "Mosaic")
    if offsets:
        if len(offsets) != 2 * len(objects):
            raise ValueError("Offsets: a row and a column offset per image: 0, 0, 0, 512, ...")
        positions = np.asarray(offsets, dtype=float).reshape(-1, 2)
    else:
        if not (rows and columns and pixel_mm):
            raise ValueError("give pixel offsets, or the row and column motors and the pixel size")
        positions = (
            np.array(
                [
                    [float(o.meta.get("positions", {}).get(m, np.nan)) for m in (rows, columns)]
                    for o in objects
                ]
            )
            / pixel_mm
        )
        if not np.all(np.isfinite(positions)):
            raise ValueError(f"not every image records {rows} and {columns} in its positions")
    positions = np.round(positions - positions.min(axis=0)).astype(int)
    tiles = [np.nan_to_num(np.asarray(o.array.values, dtype=float)) for o in objects]
    image = mosaic_from_positions(tiles, positions, mode=how)
    data = xr.DataArray(image, dims=("y", "x"), name="mosaic")
    return new(
        objects[0],
        data,
        "mosaic",
        parents=[o.name for o in objects],
        meta={"tiles": [o.name for o in objects], "positions_px": positions.tolist()},
    )


def resample_onto(
    obj: DataObject,
    reference: DataObject,
    dims: Sequence[str] | None = None,
    method: str = "linear",
) -> DataObject:
    """``obj`` interpolated onto the coordinates of ``reference``, along the dims they share
    (or the ``dims`` named).

    Points outside the range of ``obj`` are NaN. A scan's frames are read into memory
    first (within the memory budget); a reference scan gives only its coordinates. The
    uncertainties are interpolated as the values are (linearly for a cubic method).
    """
    require(obj, ("scan", *ANY), "Resample onto")
    require(reference, ("scan", *ANY), "Resample onto")
    if method not in INTERP_METHODS:
        raise ValueError(f"method must be one of {INTERP_METHODS}, not {method!r}")
    shared = [d for d in obj.dims if d in reference.dims and (not dims or d in dims)]
    if not shared:
        raise ValueError(
            f"{obj.name} {obj.dims} and {reference.name} {reference.dims} share no dim"
        )
    data = obj.data.read() if obj.kind == "scan" else obj.array
    sigma = None if obj.kind == "scan" else obj.sigma
    data = data.assign_coords({d: obj.coord(d) for d in shared if d not in data.coords})
    targets = {d: np.asarray(reference.coord(d), dtype=float) for d in shared}
    moved = {
        d: t
        for d, t in targets.items()
        if t.size != data.sizes[d] or not np.allclose(np.asarray(data.coords[d], float), t)
    }
    if moved:
        data = data.interp(moved, method=method)
    data = data.assign_coords(targets)  # the reference's values exactly, moved or not
    if sigma is not None:
        sigma = sigma.assign_coords({d: obj.coord(d) for d in shared if d not in sigma.coords})
        smooth = "nearest" if method == "nearest" else "linear"
        sigma = (sigma.interp(moved, method=smooth) if moved else sigma).assign_coords(targets)
    return new(
        obj,
        data,
        f"on_{reference.name}",
        sigma=sigma,
        parents=[obj.name, reference.name],
        meta={"resampled_onto": reference.name, "along": list(moved), "method": method},
    )


def merge_scan_sets(objects: Sequence[DataObject]) -> DataObject:
    """One scan set from scans and scan sets (to process them together)."""
    scans, labels = [], []
    for obj in objects:
        require(obj, ("scan", "scanset"), "Merge Scans")
        if obj.kind == "scan":
            scans.append(obj.data)
            labels.append(obj.name)
        else:
            scans.extend(obj.data.scans)
            labels.extend(obj.data.labels)
    merged = ScanSet(scans, labels)
    return DataObject(f"{objects[0].name}_set", merged, parents=[o.name for o in objects])
