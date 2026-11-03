"""netCDF files looked into and read without xarray's backends: netCDF-3 by scipy, netCDF-4 by h5py.

:func:`describe` lists a file's dims, global attributes and variables (their
dims, shape, type and attributes) without reading a value; :func:`read_variable`
reads one variable, cut by an index while it is read, with the coordinate
variables of its dims, as an ``xarray.DataArray``. The Browse tab shows the
first, loading a variable as an object uses the second; the runs of the APS
7-ID Python scans are such files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr

__all__ = ["NC_SUFFIXES", "NcFile", "NcVariable", "describe", "netcdf_kind", "read_variable"]

NC_SUFFIXES = (".nc", ".cdf", ".netcdf", ".nc4")
# h5py's name for a dim that netCDF-4 stores as a dataset of no values of its own
PURE_DIM = "This is a netCDF dimension but not a netCDF variable"


@dataclass
class NcVariable:
    """One variable: its ``dims`` and ``shape``, numpy ``dtype`` and ``attrs``; ``coordinate`` when
    it is the variable of its one dim (a motor's positions)."""

    name: str
    dims: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    attrs: dict[str, Any] = field(default_factory=dict)
    coordinate: bool = False


@dataclass
class NcFile:
    """What :func:`describe` found: the file's ``kind``, its ``dims`` with their lengths, its
    global ``attrs`` and its ``variables``."""

    path: Path
    kind: str
    dims: dict[str, int]
    attrs: dict[str, Any]
    variables: list[NcVariable]

    def variable(self, name: str) -> NcVariable:
        for variable in self.variables:
            if variable.name == name:
                return variable
        raise KeyError(f"{self.path.name} has no variable {name!r}")

    def table(self) -> pd.DataFrame:
        """One row per variable: name, role (data or coordinate), dims, shape, dtype, attributes."""
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "variable": v.name,
                    "role": "coordinate" if v.coordinate else "data",
                    "dims": " x ".join(f"{d}[{self.dims.get(d, '?')}]" for d in v.dims),
                    "shape": str(v.shape),
                    "dtype": v.dtype,
                    "attributes": _attrs_text(v.attrs),
                }
                for v in self.variables
            ],
            columns=["variable", "role", "dims", "shape", "dtype", "attributes"],
        )


def _attrs_text(attrs: dict[str, Any], limit: int = 200) -> str:
    text = ", ".join(f"{k}={_plain(v)}" for k, v in attrs.items())
    return text if len(text) <= limit else text[:limit] + " ..."


def _plain(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.reshape(-1)[0].item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _dtype_name(dtype: Any) -> str:
    """``float64``, not ``>f8``: netCDF-3 stores big-endian, read in the machine's order."""
    dtype = np.dtype(dtype)
    return dtype.newbyteorder("=").name if dtype.kind in "biufc" else str(dtype)


def netcdf_kind(path: str | Path) -> str | None:
    """``"netCDF-3"`` (classic or 64-bit offset), ``"netCDF-4"`` (an HDF5 file) or ``None``,
    from the file's first bytes."""
    try:
        with open(path, "rb") as f:
            head = f.read(8)
    except OSError:
        return None
    if head[:3] == b"CDF":
        return "netCDF-3"
    if head == b"\x89HDF\r\n\x1a\n":
        return "netCDF-4"
    return None


def describe(path: str | Path) -> NcFile:
    """The dims, global attributes and variables of a netCDF file, no value read."""
    path = Path(path)
    kind = netcdf_kind(path)
    if kind == "netCDF-3":
        return _describe3(path)
    if kind == "netCDF-4":
        return _describe4(path)
    raise ValueError(f"{path.name} is not a netCDF file")


def _describe3(path: Path) -> NcFile:
    from scipy.io import netcdf_file

    with netcdf_file(path, "r", mmap=False) as f:
        dims = {str(d): int(n) if n is not None else -1 for d, n in f.dimensions.items()}
        variables = [
            NcVariable(
                str(name),
                tuple(str(d) for d in v.dimensions),
                tuple(int(n) for n in v.shape),
                _dtype_name(v.data.dtype) if v.data is not None else v.typecode(),
                {str(k): _plain(a) for k, a in v._attributes.items()},
                tuple(v.dimensions) == (name,),
            )
            for name, v in f.variables.items()
        ]
        attrs = {str(k): _plain(a) for k, a in f._attributes.items()}
    return NcFile(path, "netCDF-3", dims, attrs, variables)


def _dims_of(node: Any) -> tuple[str, ...]:
    """The dims of an h5py dataset of a netCDF-4 file: its attached dimension scales."""
    names = []
    for k, dim in enumerate(node.dims):
        scales = list(dim.values())
        names.append(scales[0].name.rsplit("/", 1)[-1] if scales else f"dim_{k}")
    return tuple(names)


def _describe4(path: Path) -> NcFile:
    import h5py

    from bexa.io.h5files import open_h5

    dims: dict[str, int] = {}
    variables: list[NcVariable] = []
    with open_h5(path) as f:
        for name, node in f.items():
            if not isinstance(node, h5py.Dataset):
                continue
            attrs = {
                str(k): _plain(v)
                for k, v in node.attrs.items()
                if k not in ("DIMENSION_LIST", "REFERENCE_LIST", "CLASS", "NAME", "_Netcdf4Dimid")
            }
            is_scale = node.attrs.get("CLASS", b"") == b"DIMENSION_SCALE"
            if is_scale:
                dims[name] = int(node.shape[0]) if node.shape else 0
                if str(_plain(node.attrs.get("NAME", b""))).startswith(PURE_DIM):
                    continue  # a dim, no variable
            dataset_dims = (name,) if is_scale else _dims_of(node)
            variables.append(
                NcVariable(
                    name,
                    dataset_dims,
                    tuple(int(n) for n in node.shape),
                    _dtype_name(node.dtype),
                    attrs,
                    is_scale,
                )
            )
        attrs = {str(k): _plain(v) for k, v in f.attrs.items() if not str(k).startswith("_NC")}
    for variable in variables:  # dims without a scale of their own
        for d, n in zip(variable.dims, variable.shape, strict=False):
            dims.setdefault(d, n)
    return NcFile(path, "netCDF-4", dims, attrs, variables)


def read_variable(
    path: str | Path, name: str, index: tuple[Any, ...] | None = None
) -> xr.DataArray:
    """The variable ``name`` as a DataArray with its dims and their coordinate variables, cut
    by ``index`` (``(0, slice(None))``) while it is read; ``_FillValue`` points are NaN."""
    import xarray as xr

    path = Path(path)
    kind = netcdf_kind(path)
    if kind is None:
        raise ValueError(f"{path.name} is not a netCDF file")
    info = describe(path)
    variable = info.variable(name)
    part = tuple(index) if index is not None else ()
    part = part + (slice(None),) * (len(variable.dims) - len(part))
    coordinates = {v.name for v in info.variables if v.coordinate}
    if kind == "netCDF-3":
        from scipy.io import netcdf_file

        with netcdf_file(path, "r", mmap=False) as f:
            values = np.array(
                f.variables[name].data[part] if variable.dims else f.variables[name].data
            )
            axes = {d: np.array(f.variables[d].data) for d in variable.dims if d in coordinates}
    else:
        from bexa.io.h5files import open_h5

        with open_h5(path) as f:
            values = np.asarray(f[name][part] if variable.dims else f[name][()])
            axes = {d: np.asarray(f[d][()]) for d in variable.dims if d in coordinates}
    if values.dtype.kind in "biufc" and not values.dtype.isnative:
        values = values.astype(values.dtype.newbyteorder("="))
    fill = variable.attrs.get("_FillValue")
    if fill is not None and values.dtype.kind == "f" and np.isfinite(fill):
        values = np.where(values == fill, np.nan, values)
    kept = [
        d for d, p in zip(variable.dims, part, strict=True) if not isinstance(p, (int, np.integer))
    ]
    coords = {}
    for d, p in zip(variable.dims, part, strict=True):
        if d in kept and d in axes:
            coords[d] = axes[d][p]
    attrs = {k: v for k, v in variable.attrs.items() if k != "_FillValue"}
    return xr.DataArray(values, dims=kept, coords=coords, name=name, attrs=attrs)
