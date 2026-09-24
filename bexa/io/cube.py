"""bexa's reduced-data format, and ``load``/``save`` for xarray objects.

``save`` writes a DataArray or Dataset to HDF5 (``.h5``) or zarr (``.zarr``)
with its coordinates and provenance attrs. ``load`` reads those back and also
understands the legacy PAL-XFEL cubes (``runN.h5``) and LCLS cube triples, so
old and new reduced data look the same in memory: a Dataset with
``frames(laser, <axis>, y, x)`` and ``i0(laser, <axis>)``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import xarray as xr

from bexa._log import get_logger
from bexa.core.provenance import build_attrs, to_json

log = get_logger(__name__)

FORMAT_MARKER = "bexa_format"
FORMAT_VERSION = "cube-v1"
LEGACY_AXES = ("delays", "phi", "chi", "th", "tth", "laser_v", "laser_h", "index")
LASER_COORD = np.array(["off", "on"])

__all__ = ["is_bexa_file", "load", "load_lcls_cube", "load_legacy_cube", "save"]


# ----------------------------------------------------------------------- attrs
def _encode_attr(value: Any) -> Any:
    if isinstance(value, (str, bytes, int, float, bool, np.integer, np.floating)):
        return value
    if isinstance(value, np.ndarray) and value.dtype.kind in "iufb" and value.ndim == 1:
        return value
    return "json:" + to_json(value)


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, str) and value.startswith("json:"):
        return json.loads(value[5:])
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_attrs(node: Any, attrs: dict[str, Any]) -> None:
    for key, value in attrs.items():
        if value is None:
            continue
        node.attrs[key] = _encode_attr(value)


def _read_attrs(node: Any) -> dict[str, Any]:
    return {k: _decode_attr(v) for k, v in node.attrs.items()}


# ------------------------------------------------------------------------ save
def save(
    obj: xr.DataArray | xr.Dataset | dict[str, xr.DataArray],
    path: str | Path,
    compression: str | None = "gzip",
    overwrite: bool = True,
) -> Path:
    """Write an xarray object (or a dict of DataArrays) to ``.h5`` or ``.zarr``.

    Every variable keeps its dims, coordinates and attrs; the file-level attrs
    get the provenance fields from :func:`bexa.core.provenance.build_attrs` if
    the object does not carry them already. Device arrays are copied to the host.
    """
    from bexa.core.backend import to_host

    path = Path(path)
    if isinstance(obj, dict):
        obj = xr.Dataset({k: v for k, v in obj.items()})
    if isinstance(obj, xr.DataArray):
        name = obj.name or "data"
        dataset = obj.to_dataset(name=name)
        dataset.attrs.update(obj.attrs)
    else:
        dataset = obj
    dataset = to_host(dataset)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    attrs = dict(dataset.attrs)
    if "bexa_version" not in attrs:
        attrs.update(build_attrs())

    if path.suffix == ".zarr":
        clean = dataset.copy()
        clean.attrs = {k: _encode_attr(v) for k, v in attrs.items()}
        for var in clean.variables.values():
            var.attrs = {k: _encode_attr(v) for k, v in var.attrs.items()}
        clean.to_zarr(path, mode="w")
        return path

    with h5py.File(path, "w") as f:
        f.attrs[FORMAT_MARKER] = FORMAT_VERSION
        _write_attrs(f, attrs)
        f.attrs["data_vars"] = json.dumps(list(dataset.data_vars))
        for name, var in dataset.data_vars.items():
            _write_variable(f, str(name), var, compression)
        coords_group = f.create_group("coords")
        for name, coord in dataset.coords.items():
            _write_variable(coords_group, str(name), coord, None)
    return path


def _write_variable(group: Any, name: str, var: xr.DataArray, compression: str | None) -> None:
    data = np.asarray(var.values)
    if data.dtype.kind in "UO":
        data = data.astype("S")
    kwargs: dict[str, Any] = {}
    if compression and data.ndim >= 2 and data.size > 1024:
        kwargs = {
            "compression": compression,
            "compression_opts": 4 if compression == "gzip" else None,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        kwargs["chunks"] = (1,) * (data.ndim - 2) + tuple(data.shape[-2:])
    ds = group.create_dataset(name, data=data, **kwargs)
    ds.attrs["dims"] = json.dumps(list(var.dims))
    _write_attrs(ds, dict(var.attrs))


# ------------------------------------------------------------------------ load
def is_bexa_file(path: str | Path) -> bool:
    """True for HDF5 files written by :func:`save`."""
    path = Path(path)
    if path.suffix not in (".h5", ".hdf5"):
        return False
    try:
        with h5py.File(path, "r") as f:
            return FORMAT_MARKER in f.attrs
    except OSError:
        return False


def load(path: str | Path, squeeze_single: bool = True) -> xr.Dataset | xr.DataArray:
    """Read ``.h5``/``.zarr`` written by :func:`save`, a legacy ``runN.h5``, or an LCLS cube.

    Parameters
    ----------
    squeeze_single
        Return a DataArray when the file holds exactly one data variable.
    """
    path = Path(path)
    if path.suffix == ".zarr":
        dataset = xr.open_zarr(path).load()
        dataset.attrs = {k: _decode_attr(v) for k, v in dataset.attrs.items()}
        for var in dataset.variables.values():
            var.attrs = {k: _decode_attr(v) for k, v in var.attrs.items()}
    elif path.is_dir() or path.suffix in (".npy", ".csv"):
        dataset = load_lcls_cube(path)
    elif is_bexa_file(path):
        dataset = _load_bexa_h5(path)
    else:
        dataset = load_legacy_cube(path)
    if squeeze_single and len(dataset.data_vars) == 1:
        name = next(iter(dataset.data_vars))
        array = dataset[name]
        array.attrs = {**dataset.attrs, **array.attrs}
        return array
    return dataset


def _read_variable(node: Any) -> xr.Variable:
    data = node[()]
    if isinstance(data, np.ndarray) and data.dtype.kind == "S":
        data = data.astype(str)
    dims = (
        json.loads(node.attrs["dims"])
        if "dims" in node.attrs
        else [f"dim_{i}" for i in range(np.ndim(data))]
    )
    attrs = {k: _decode_attr(v) for k, v in node.attrs.items() if k != "dims"}
    return xr.Variable(dims, data, attrs=attrs)


def _load_bexa_h5(path: Path) -> xr.Dataset:
    with h5py.File(path, "r") as f:
        attrs = _read_attrs(f)
        names = json.loads(attrs.pop("data_vars", "[]"))
        attrs.pop(FORMAT_MARKER, None)
        data_vars = {name: _read_variable(f[name]) for name in names}
        coords = (
            {name: _read_variable(f["coords"][name]) for name in f["coords"]}
            if "coords" in f
            else {}
        )
    return xr.Dataset(data_vars, coords=coords, attrs=attrs)


def load_legacy_cube(path: str | Path, axis: str | None = None) -> xr.Dataset:
    """Read a ``runN.h5`` written by the legacy PAL-XFEL cube builder.

    The scan axis is ``axis`` or the first entry of ``LEGACY_AXES`` that varies
    and matches the number of points; ``index`` is the fallback.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        if "jungfrau_on" not in f:
            raise ValueError(f"{path} is neither a bexa file nor a legacy cube (no 'jungfrau_on')")
        frames_on = f["jungfrau_on"][()]
        frames_off = f["jungfrau_off"][()]
        n = frames_on.shape[0]
        axes = {}
        for name in LEGACY_AXES:
            if name in f and f[name].shape == (n,):
                axes[name] = f[name][()]
        i0_on = f["signals_on"][()] if "signals_on" in f else np.full(n, np.nan)
        i0_off = f["signals_off"][()] if "signals_off" in f else np.full(n, np.nan)
    if axis is None:
        varying = [a for a in LEGACY_AXES if a in axes and np.ptp(axes[a]) > 0]
        axis = varying[0] if varying else "index"
    if axis not in axes:
        axes[axis] = np.arange(n, dtype=float)
    dim = "delay" if axis == "delays" else axis
    coords: dict[str, Any] = {"laser": LASER_COORD, dim: axes[axis]}
    for name, values in axes.items():
        if name != axis:
            coords[name] = (dim, values)
    frames = xr.DataArray(
        np.stack([frames_off, frames_on]),
        dims=("laser", dim, "y", "x"),
        coords=coords,
        name="frames",
    )
    i0 = xr.DataArray(
        np.stack([i0_off, i0_on]),
        dims=("laser", dim),
        coords={"laser": LASER_COORD, dim: axes[axis]},
        name="i0",
    )
    dataset = xr.Dataset({"frames": frames, "i0": i0})
    dataset.attrs.update(build_attrs(source_files=[path], parameters={"legacy_axis": axis}))
    dataset.attrs["scan_axis"] = dim
    return dataset


def load_lcls_cube(path: str | Path, run: int | None = None) -> xr.Dataset:
    """Read an LCLS XCS cube triple as ``frames(laser, shot, y, x)`` with ``scanvar`` and ``i0``."""
    path = Path(path)
    folder = path if path.is_dir() else path.parent
    if run is None:
        candidates = sorted(folder.glob("Run*_onStk.npy"))
        if path.is_file() and path.stem.startswith("Run"):
            run = int(path.stem[3:7])
        elif candidates:
            run = int(candidates[0].stem[3:7])
        else:
            raise FileNotFoundError(f"no Run*_onStk.npy in {folder}")
    on = np.load(folder / f"Run{run:04d}_onStk.npy")
    off = np.load(folder / f"Run{run:04d}_offStk.npy")
    stats = np.loadtxt(folder / f"Run{run:04d}_stats.csv", delimiter=",", skiprows=1)
    stats = np.atleast_2d(stats)
    n = on.shape[0]
    shot = np.arange(n)
    frames = xr.DataArray(
        np.stack([off, on]),
        dims=("laser", "shot", "y", "x"),
        coords={"laser": LASER_COORD, "shot": shot, "scanvar": ("shot", stats[:n, 1])},
        name="frames",
    )
    i0 = xr.DataArray(
        np.stack([stats[:n, 3], stats[:n, 2]]),
        dims=("laser", "shot"),
        coords={"laser": LASER_COORD, "shot": shot},
        name="i0",
    )
    dataset = xr.Dataset({"frames": frames, "i0": i0})
    files = [folder / f"Run{run:04d}_{s}" for s in ("onStk.npy", "offStk.npy", "stats.csv")]
    dataset.attrs.update(build_attrs(source_files=files, parameters={"run": run}))
    dataset.attrs["scan_axis"] = "scanvar"
    return dataset
