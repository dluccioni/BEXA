"""Reading back the objects of a saved workspace (:func:`bexa.workspace.files.write_nexus`)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from bexa._log import get_logger
from bexa.workspace.objects import DataObject, ScanSet

log = get_logger(__name__)

__all__ = ["read_object"]


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def read_object(name: str, group: Any, cache: Any = True) -> DataObject:
    """The object stored in one group of a saved workspace.

    Scans are opened again from the recipes kept for them; one whose files
    are gone becomes a text object that says why.
    """
    kind = _text(group.attrs.get("kind", ""))
    fields = {
        "meta": json.loads(_text(group.attrs.get("meta", "{}"))),
        "history": json.loads(_text(group.attrs.get("history", "[]"))),
        "parents": json.loads(_text(group.attrs.get("parents", "[]"))),
        "notes": _text(group.attrs.get("notes", "")),
    }
    if kind in ("scan", "scanset"):
        recipes = json.loads(_text(group.attrs.get("recipes", "[]")))
        labels = json.loads(_text(group.attrs.get("labels", "null")))
        try:
            data = _reopen(kind, recipes, cache, labels)
        except Exception as exc:
            log.warning("could not open %s again: %s", name, exc)
            return DataObject(
                name, f"{name}: the {kind} could not be opened again ({exc})", kind="text", **fields
            )
        return DataObject(name, data, kind=kind, **fields)
    if kind == "table":
        columns = json.loads(_text(group.attrs["columns"]))
        values = {}
        for i, column in enumerate(columns):
            column_values = group[f"c{i:03d}"][()]
            if column_values.dtype.kind == "S":
                column_values = column_values.astype(str)
            values[column] = column_values
        return DataObject(name, pd.DataFrame(values), kind="table", **fields)
    if kind == "text":
        return DataObject(name, _text(group["text"][()]), kind="text", **fields)
    dims = [_text(d) for d in group.attrs["axes"]]
    coords = {}
    for dim in dims:
        values = group[dim][()]
        coords[dim] = values.astype(str) if values.dtype.kind == "S" else values
    values = group["data"][()]
    long_name = group["data"].attrs.get("long_name")
    data = xr.DataArray(
        values, dims=dims, coords=coords, name=_text(long_name) if long_name else None
    )
    for dim in dims:
        units = group[dim].attrs.get("units")
        if units is not None:
            data.coords[dim].attrs["units"] = _text(units)
    data.attrs.update(json.loads(_text(group.attrs.get("attrs", "{}"))))
    for point, on in json.loads(_text(group.attrs.get("coordinates", "{}"))).items():
        units = group[point].attrs.get("units")
        point_attrs = {"units": _text(units)} if units is not None else {}
        data = data.assign_coords({point: (on, group[point][()], point_attrs)})
    sigma = None
    if "errors" in group:
        sigma = xr.DataArray(np.asarray(group["errors"][()]), dims=dims, coords=coords)
    return DataObject(name, data, kind=kind, sigma=sigma, **fields)


def _reopen(kind: str, recipes: list[dict[str, Any]], cache: Any, labels: Any = None) -> Any:
    import bexa

    scans = [bexa.open(**{"cache": cache, **recipe}) for recipe in recipes]
    return scans[0] if kind == "scan" else ScanSet(scans, labels)
