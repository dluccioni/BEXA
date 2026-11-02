"""Files to objects and objects to files: arrays, cubes, images, tables, NeXus HDF5.

:func:`load_file` reads what is not a scan (a scan is opened with
:func:`bexa.open`): a bexa cube, one dataset of any HDF5 file (sliced while it
is read), ``.npy``, TIFF, PNG/JPEG, CSV/text tables. :func:`export_object`
writes an object as NeXus HDF5 (values, axes, uncertainties, metadata and
history), CSV, NPY, TIFF, PNG or text.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from bexa.workspace.objects import DataObject
from bexa.workspace.source import jsonable

__all__ = [
    "EXPORT_FORMATS",
    "H5_SUFFIXES",
    "export_object",
    "load_file",
    "parse_index",
    "write_nexus",
]

H5_SUFFIXES = (".h5", ".hdf5", ".nxs", ".nx", ".cxi", ".hdf")
TIFF_SUFFIXES = (".tif", ".tiff")
PICTURE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")
TABLE_SUFFIXES = (".csv", ".tsv", ".txt", ".dat")
EXPORT_FORMATS = ("h5", "csv", "npy", "tiff", "png", "txt", "json")


def parse_index(text: str | None) -> tuple[Any, ...] | None:
    """``"0, :, 10:20"`` as ``(0, slice(None), slice(10, 20))``; empty text gives ``None``."""
    text = (text or "").strip().strip("[]")
    if not text:
        return None
    parts: list[Any] = []
    for part in text.split(","):
        part = part.strip()
        if part in (":", ""):
            parts.append(slice(None))
        elif ":" in part:
            bits = [int(b) if b.strip() else None for b in part.split(":")]
            parts.append(slice(*bits))
        else:
            parts.append(int(part))
    return tuple(parts)


def _array_object(values: np.ndarray, name: str, meta: dict[str, Any]) -> DataObject:
    data = np.asarray(values)
    if data.ndim == 0:
        return DataObject(name, f"{name} = {data.item()!r}", kind="text", meta=meta)
    if data.ndim == 1:
        da = xr.DataArray(data, dims=("index",), coords={"index": np.arange(data.size)}, name=name)
    else:
        dims = (*(f"dim_{i}" for i in range(data.ndim - 2)), "y", "x")
        coords = {d: np.arange(n) for d, n in zip(dims, data.shape, strict=True)}
        da = xr.DataArray(data, dims=dims, coords=coords, name=name)
    return DataObject(name, da, meta=meta)


def _stem(path: Path) -> str:
    return path.stem.replace(" ", "_").replace(".", "_") or "data"


def load_file(
    path: str | Path,
    dataset: str | None = None,
    index: str | None = None,
    name: str | None = None,
) -> list[DataObject]:
    """The objects in a file that is not a scan.

    ``dataset`` (or ``file.h5::/path``) reads one dataset of an HDF5 file, cut
    by ``index`` (``"0, :, 100:200"``) while it is read, so a slice of a
    100 GB dataset costs only the slice.
    """
    text = str(path)
    if "::" in text and dataset is None:
        text, dataset = text.split("::", 1)
    file = Path(text)
    suffix = file.suffix.lower()
    meta = {"source": str(file)}
    stem = name or _stem(file)
    if suffix in H5_SUFFIXES:
        if dataset:
            return [_h5_dataset(file, dataset, index, name, meta)]
        from bexa.io.cube import is_bexa_file, load

        if not is_bexa_file(file):
            raise ValueError(
                f"{file.name} is neither a bexa cube nor a layout bexa reads; pick one of its "
                "datasets on the Browse tab"
            )
        loaded = load(file)
        if isinstance(loaded, xr.DataArray):
            return [DataObject(stem, loaded, meta=meta)]
        return [DataObject(f"{stem}_{var}", loaded[var], meta=meta) for var in loaded.data_vars]
    if suffix == ".npy":
        values = np.load(file, mmap_mode="r")
        part = parse_index(index)
        return [_array_object(np.array(values[part] if part else values), stem, meta)]
    if suffix in TIFF_SUFFIXES:
        import tifffile

        values = tifffile.imread(file)
        part = parse_index(index)
        values = values[part] if part else values
        obj = _array_object(values, stem, meta)
        if obj.is_array and obj.array.ndim == 3:
            obj = DataObject(stem, obj.array.rename({"dim_0": "frame"}), meta=meta)
        return [obj]
    if suffix in PICTURE_SUFFIXES:
        from matplotlib.image import imread

        picture = np.asarray(imread(file), dtype=float)
        if picture.ndim == 3:
            rgb = xr.DataArray(picture[..., :3], dims=("y", "x", "rgb"), name=stem)
            return [DataObject(stem, rgb, kind="rgb", meta=meta)]
        return [_array_object(picture, stem, meta)]
    if suffix in TABLE_SUFFIXES:
        table = pd.read_csv(file, sep=None, engine="python", comment="#")
        return [DataObject(stem, table, kind="table", meta=meta)]
    if suffix == ".json":
        content = json.loads(file.read_text())
        return [DataObject(stem, json.dumps(content, indent=2)[:100000], kind="text", meta=meta)]
    raise ValueError(f"bexa does not read {suffix} files as objects; open scans on the Data tab")


def _h5_dataset(
    file: Path, dataset: str, index: str | None, name: str | None, meta: dict[str, Any]
) -> DataObject:
    from bexa.io.h5files import open_h5

    part = parse_index(index)
    with open_h5(file) as f:
        if dataset not in f:
            raise KeyError(f"{file.name} has no dataset {dataset!r}")
        node = f[dataset]
        values = node[part] if part else node[()]
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in node.attrs.items()}
    if isinstance(values, bytes):
        values = values.decode(errors="replace")
    if isinstance(values, str):
        return DataObject(
            name or dataset.strip("/").replace("/", "_"), values, kind="text", meta=meta
        )
    label = name or dataset.strip("/").replace("/", "_")
    info = {**meta, "dataset": dataset, "index": index or ""}
    info.update({f"attr_{k}": jsonable(v) for k, v in list(attrs.items())[:20]})
    return _array_object(np.asarray(values), label, info)


# ------------------------------------------------------------------------------- export
def export_object(obj: DataObject, path: str | Path, format: str | None = None) -> Path:
    """Write ``obj`` to ``path``.

    ``format`` is h5, csv, npy, tiff, png, txt or json; the default is the suffix.
    """
    out = Path(path)
    kind = (format or out.suffix.lstrip(".") or "h5").lower()
    kind = {"hdf5": "h5", "nxs": "h5", "tif": "tiff", "jpg": "png"}.get(kind, kind)
    if kind not in EXPORT_FORMATS:
        raise ValueError(f"format must be one of {EXPORT_FORMATS}, not {kind!r}")
    out.parent.mkdir(parents=True, exist_ok=True)
    if kind == "h5":
        write_nexus([obj], out)
    elif kind == "csv":
        _write_csv(obj, out)
    elif kind == "txt":
        text = (
            obj.data
            if obj.kind == "text"
            else (obj.data.to_string() if obj.kind == "table" else obj.info())
        )
        out.write_text(str(text), encoding="utf-8")
    elif kind == "json":
        payload = {
            "name": obj.name,
            "kind": obj.kind,
            "meta": jsonable(obj.meta),
            "history": obj.history,
        }
        if obj.kind == "table":
            payload["table"] = json.loads(obj.data.to_json(orient="records"))
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    elif kind == "npy":
        np.save(out, np.asarray(obj.array.values))
    elif kind == "tiff":
        import tifffile

        values = np.asarray(obj.array.values)
        if values.ndim > 3:
            values = values.reshape(-1, *values.shape[-2:])
        tifffile.imwrite(out, values.astype(np.float32) if values.dtype.kind == "f" else values)
    elif kind == "png":
        _write_png(obj, out)
    return out


def _write_csv(obj: DataObject, out: Path) -> None:
    if obj.kind == "table":
        obj.data.to_csv(out, index=False)
        return
    da = obj.array
    if da.ndim == 1:
        dim = str(da.dims[0])
        frame = pd.DataFrame({dim: obj.coord(dim), str(da.name or "value"): da.values})
        if obj.sigma is not None:
            frame["sigma"] = np.asarray(obj.sigma.values)
        frame.to_csv(out, index=False)
        return
    if da.ndim == 2:
        rows, cols = (str(d) for d in da.dims)
        frame = pd.DataFrame(np.asarray(da.values), index=obj.coord(rows), columns=obj.coord(cols))
        frame.index.name = f"{rows} \\ {cols}"
        frame.to_csv(out)
        return
    flat = da.to_dataframe(name=str(da.name or "value")).reset_index()
    flat.to_csv(out, index=False)


def _write_png(obj: DataObject, out: Path) -> None:
    import matplotlib.pyplot as plt

    from bexa.viz.auto import plot

    if obj.kind == "rgb":
        fig, ax = plt.subplots()
        ax.imshow(np.clip(np.asarray(obj.array.values, dtype=float), 0, 1))
    else:
        result = plot(obj.array.squeeze())
        fig = result[0] if isinstance(result, tuple) else plt.gcf()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_nexus(objects: Sequence[DataObject], path: str | Path) -> Path:
    """Write array objects (and tables, texts) to one NeXus HDF5 file, one NXdata group each.

    Each group holds the values as ``data`` (the NeXus signal), one dataset
    per axis, ``errors`` for the uncertainties, and the metadata, parents and
    history as JSON attributes; tables become one dataset per column.
    """
    import h5py

    out = Path(path)
    with h5py.File(out, "w") as f:
        f.attrs["default"] = "entry"
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.attrs["creator"] = "bexa"
        for obj in objects:
            group = entry.create_group(obj.name.replace("/", "%2F"))
            group.attrs["kind"] = obj.kind
            group.attrs["meta"] = json.dumps(jsonable(obj.meta))
            group.attrs["history"] = json.dumps(list(obj.history))
            group.attrs["parents"] = json.dumps(list(obj.parents))
            group.attrs["notes"] = obj.notes
            if obj.is_array:
                _write_nxdata(group, obj)
            elif obj.kind == "table":
                group.attrs["NX_class"] = "NXcollection"
                group.attrs["columns"] = json.dumps([str(c) for c in obj.data.columns])
                for i, column in enumerate(obj.data.columns):
                    values = obj.data[column].to_numpy()
                    if values.dtype == object:
                        values = values.astype(str).astype("S")
                    group.create_dataset(f"c{i:03d}", data=values)
            elif obj.kind == "text":
                group.attrs["NX_class"] = "NXnote"
                group.create_dataset("text", data=str(obj.data))
            else:  # scans and scan sets are kept by how to open them again
                group.attrs["NX_class"] = "NXcollection"
                recipes = [obj.data.recipe] if obj.kind == "scan" else obj.data.recipes
                group.attrs["recipes"] = json.dumps(jsonable(recipes))
                if obj.kind == "scanset":
                    group.attrs["labels"] = json.dumps(list(obj.data.labels))
    return out


def _write_nxdata(group: Any, obj: DataObject) -> None:
    da = obj.array
    group.attrs["NX_class"] = "NXdata"
    group.attrs["signal"] = "data"
    dims = [str(d) for d in da.dims]
    group.attrs["axes"] = dims
    values = np.asarray(da.values)
    chunks = (1, *values.shape[1:]) if values.ndim >= 3 else None
    dataset = group.create_dataset(
        "data", data=values, chunks=chunks, compression="gzip" if values.size > 4096 else None
    )
    if da.name:
        dataset.attrs["long_name"] = str(da.name)
    for dim in dims:
        coord = obj.coord(dim)
        if coord.dtype.kind in "OU":  # text labels (the rgb channels): stored as bytes
            coord = coord.astype(str).astype("S")
        axis = group.create_dataset(dim, data=coord)
        units = da.coords[dim].attrs.get("units") if dim in da.coords else None
        if units:
            axis.attrs["units"] = str(units)
        group.attrs[f"{dim}_indices"] = dims.index(dim)
    if obj.sigma is not None:
        group.create_dataset(
            "errors",
            data=np.asarray(obj.sigma.values),
            compression="gzip" if values.size > 4096 else None,
        )
