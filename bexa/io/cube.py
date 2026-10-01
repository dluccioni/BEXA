"""bexa's reduced-data format, and ``load``/``save`` for xarray objects.

``save`` writes a DataArray or Dataset to HDF5 (``.h5``) or zarr (``.zarr``)
with its coordinates and provenance attrs. ``load`` reads those back and also
understands the legacy PAL-XFEL cubes (``runN.h5``) and LCLS cube triples, so
old and new reduced data look the same in memory: a Dataset with
``frames(laser, <axis>, y, x)`` plus the per-point series along the axis
(``signal`` = ROI/I0 mean for PAL cubes, ``i0`` for LCLS shots).
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import xarray as xr
from xarray.backends import BackendArray
from xarray.core import indexing

from bexa._log import get_logger
from bexa.core.provenance import build_attrs, to_json

log = get_logger(__name__)

FORMAT_MARKER = "bexa_format"
FORMAT_VERSION = "cube-v1"
LEGACY_AXES = ("delays", "phi", "chi", "th", "tth", "laser_v", "laser_h", "index")
LASER_COORD = np.array(["off", "on"])

__all__ = [
    "StoreWriter",
    "is_bexa_file",
    "load",
    "load_lcls_cube",
    "load_legacy_cube",
    "open_lazy",
    "save",
]


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


MAX_ATTR_BYTES = 60_000  # HDF5 keeps attributes in the object header: about 64 KB in all


def _write_attrs(node: Any, attrs: dict[str, Any]) -> None:
    for key, value in attrs.items():
        if value is None:
            continue
        encoded = _encode_attr(value)
        size = encoded.nbytes if isinstance(encoded, np.ndarray) else len(str(encoded))
        if size > MAX_ATTR_BYTES:  # the per-frame statistics of a stack: derived, droppable
            log.debug(
                "attr %s is %d bytes, too large for an HDF5 attribute; not written", key, size
            )
            continue
        node.attrs[key] = encoded


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
        _write_nexus(f, dataset)
    return path


def _write_nexus(f: Any, dataset: xr.Dataset) -> None:
    """NeXus ``NXdata`` groups linking to the arrays, so h5web and silx plot the file as it is."""
    var_dims = {str(n): [str(d) for d in dataset[n].dims] for n in dataset.data_vars}
    coord_ndim = {str(n): int(c.ndim) for n, c in dataset.coords.items()}
    _link_nexus(f, var_dims, coord_ndim)


def _link_nexus(f: Any, var_dims: dict[str, list[str]], coord_ndim: dict[str, int]) -> None:
    """NeXus groups for variables with the given dims, linking the 1-D coords as axes."""
    names = list(var_dims)
    if not names:
        return
    f.attrs["NX_class"] = "NXroot"
    f.attrs["default"] = "nexus"
    entry = f.create_group("nexus")
    entry.attrs["NX_class"] = "NXentry"
    entry.attrs["default"] = names[0]
    for name in names:
        group = entry.create_group(name)
        group.attrs["NX_class"] = "NXdata"
        group.attrs["signal"] = name
        group[name] = f[name]  # a hard link: no second copy of the data
        axes = []
        for dim in var_dims[name]:
            usable = dim in f["coords"] and coord_ndim.get(dim) == 1
            if usable and dim != name:
                if dim not in group:
                    group[dim] = f["coords"][dim]
                axes.append(dim)
            else:
                axes.append(".")
        group.attrs["axes"] = axes


def _dataset_kwargs(
    shape: tuple[int, ...], dtype: np.dtype, compression: str | None
) -> dict[str, Any]:
    """Chunk and compression settings of :func:`save`: one chunk per frame, gzip."""
    size = int(np.prod(shape)) if shape else 1
    if not compression or len(shape) < 2 or size <= 1024:
        return {}
    kwargs: dict[str, Any] = {
        "compression": compression,
        "chunks": (1,) * (len(shape) - 2) + shape[-2:],
    }
    if compression == "gzip":
        kwargs["compression_opts"] = 4
    return kwargs


class StoreWriter:
    """Write a bexa HDF5 file piece by piece, for results built scan by scan.

    Declare every variable with :meth:`add_variable` (its full shape; the file
    holds NaN until written), the coordinates with :meth:`add_coord`, then
    :meth:`write` blocks into position and :meth:`close`. The file appears
    under ``path`` only when it is complete (it is built under a temporary
    name), and it reads back with :func:`load` or :func:`open_lazy`.
    """

    def __init__(
        self,
        path: str | Path,
        attrs: dict[str, Any] | None = None,
        compression: str | None = "gzip",
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp = self.path.with_name(self.path.stem + ".partial.h5")
        self._file = h5py.File(self._tmp, "w")
        self._file.attrs[FORMAT_MARKER] = FORMAT_VERSION
        file_attrs = dict(attrs or {})
        if "bexa_version" not in file_attrs:
            file_attrs.update(build_attrs())
        _write_attrs(self._file, file_attrs)
        self._file.create_group("coords")
        self._var_dims: dict[str, list[str]] = {}
        self._coord_ndim: dict[str, int] = {}
        self._compression = compression

    def add_variable(
        self,
        name: str,
        dims: Sequence[str],
        shape: Sequence[int],
        dtype: Any,
        attrs: dict[str, Any] | None = None,
        compression: str | bool | None = True,
    ) -> None:
        """Declare a variable of the given dims and shape; its values start as NaN (or 0)."""
        dtype = np.dtype(dtype)
        shape = tuple(int(n) for n in shape)
        kwargs = _dataset_kwargs(
            shape, dtype, self._compression if compression is True else (compression or None)
        )
        if dtype.kind == "f":
            kwargs["fillvalue"] = np.nan
        ds = self._file.create_dataset(name, shape=shape, dtype=dtype, **kwargs)
        ds.attrs["dims"] = json.dumps([str(d) for d in dims])
        _write_attrs(ds, dict(attrs or {}))
        self._var_dims[name] = [str(d) for d in dims]

    def add_coord(
        self,
        name: str,
        values: Any,
        dims: Sequence[str] | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        """Write a coordinate (1-D along its own dim unless ``dims`` says otherwise)."""
        data = np.asarray(values)
        if data.dtype.kind in "UO":
            data = data.astype("S")
        ds = self._file["coords"].create_dataset(name, data=data)
        ds.attrs["dims"] = json.dumps([str(d) for d in (dims or (name,))])
        _write_attrs(ds, dict(attrs or {}))
        self._coord_ndim[name] = int(data.ndim)

    def write(self, name: str, index: tuple[int, ...] | tuple[slice, ...], block: Any) -> None:
        """Put ``block`` at ``index`` (one int or slice per leading dim) of a declared variable."""
        self._file[name][tuple(index)] = np.asarray(block)

    def close(self) -> Path:
        """Finish the file: the variable list, the NeXus links, and the final name."""
        self._file.attrs["data_vars"] = json.dumps(list(self._var_dims))
        _link_nexus(self._file, self._var_dims, self._coord_ndim)
        self._file.close()
        self._tmp.replace(self.path)
        return self.path

    def abort(self) -> None:
        """Close and delete the partial file after a failure."""
        with contextlib.suppress(Exception):
            self._file.close()
        self._tmp.unlink(missing_ok=True)


def _write_variable(group: Any, name: str, var: xr.DataArray, compression: str | None) -> None:
    data = np.asarray(var.values)
    if data.dtype.kind in "UO":
        data = data.astype("S")
    kwargs = _dataset_kwargs(tuple(data.shape), data.dtype, compression)
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


def _file_contents(f: Any) -> tuple[dict[str, Any], list[str], dict[str, xr.Variable]]:
    """File attrs (without the format markers), the variable names and the coordinates."""
    attrs = _read_attrs(f)
    names = json.loads(attrs.pop("data_vars", "[]"))
    for key in (FORMAT_MARKER, "NX_class", "default"):
        attrs.pop(key, None)
    coords = (
        {name: _read_variable(f["coords"][name]) for name in f["coords"]} if "coords" in f else {}
    )
    return attrs, names, coords


def _load_bexa_h5(path: Path) -> xr.Dataset:
    from bexa.core.backend import check_fits

    with h5py.File(path, "r") as f:
        attrs, names, coords = _file_contents(f)
        nbytes = sum(int(np.prod(f[name].shape)) * f[name].dtype.itemsize for name in names)
        check_fits(nbytes, f"loading {path.name}", hint="open_lazy(path) reads blocks on demand")
        data_vars = {name: _read_variable(f[name]) for name in names}
    return xr.Dataset(data_vars, coords=coords, attrs=attrs)


class _LazyH5Array(BackendArray):
    """One dataset of an HDF5 file, read on demand the way xarray's backends do it."""

    def __init__(self, path: Path, name: str, shape: tuple[int, ...], dtype: np.dtype) -> None:
        self.path = path
        self.name = name
        self.shape = shape
        self.dtype = dtype

    def __getitem__(self, key: Any) -> np.ndarray:
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.OUTER_1VECTOR, self._read
        )

    def _read(self, key: tuple[Any, ...]) -> np.ndarray:
        with h5py.File(self.path, "r") as f:  # opened per read: nothing stays locked
            return np.asarray(f[self.name][key])


def open_lazy(path: str | Path) -> xr.Dataset:
    """Open a bexa HDF5 file so that only the part that is indexed is read.

    The variables are lazily indexed arrays: ``ds[name].isel(...)``, ``.sel``
    and ``.transpose`` read nothing, ``.values`` reads the selected block, and
    coordinates and attrs are loaded. Anything that needs the whole array (a
    sum, a plot of everything, :func:`save`) loads it, so keep to blocks. This
    is how a volume stacked with ``store=`` (:func:`bexa.stack`) is browsed
    without fitting in memory.
    """
    path = Path(path)
    if not is_bexa_file(path):
        raise ValueError(f"{path} is not a bexa HDF5 file")
    with h5py.File(path, "r") as f:
        attrs, names, coords = _file_contents(f)
        data_vars: dict[str, xr.Variable] = {}
        for name in names:
            node = f[name]
            if node.dtype.kind in "SOU" or node.ndim == 0:
                data_vars[name] = _read_variable(node)
                continue
            dims = (
                json.loads(node.attrs["dims"])
                if "dims" in node.attrs
                else [f"dim_{i}" for i in range(node.ndim)]
            )
            var_attrs = {k: _decode_attr(v) for k, v in node.attrs.items() if k != "dims"}
            lazy = indexing.LazilyIndexedArray(
                _LazyH5Array(path, name, tuple(node.shape), np.dtype(node.dtype))
            )
            data_vars[name] = xr.Variable(dims, lazy, attrs=var_attrs)
    attrs["lazy_file"] = str(path)
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
        signal_on = f["signals_on"][()] if "signals_on" in f else np.full(n, np.nan)
        signal_off = f["signals_off"][()] if "signals_off" in f else np.full(n, np.nan)
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
    signal = xr.DataArray(
        np.stack([signal_off, signal_on]),
        dims=("laser", dim),
        coords={"laser": LASER_COORD, dim: axes[axis]},
        name="signal",
        attrs={"description": "mean of roi_stat / i0 over the shots of each laser state"},
    )
    dataset = xr.Dataset({"frames": frames, "signal": signal})
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
