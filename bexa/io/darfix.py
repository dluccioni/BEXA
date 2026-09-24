"""Bridges to darfix: data URLs for its ``Dataset`` and its moment maps as xarray.

darfix reads a scan through silx ``DataUrl`` strings (``file.h5::/path``);
:func:`data_urls` and :func:`metadata_url` build them from a bexa scan, and
:func:`to_darfix_dataset` creates the darfix ``Dataset`` (darfix itself is
imported only there). :func:`moments_dataset` turns darfix's ``apply_moments``
output into an xarray Dataset with the same names as bexa's own moments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from bexa.core.provenance import build_attrs

MOMENT_NAMES = ("com", "std", "skew", "kurtosis")

__all__ = ["compute_moments", "data_urls", "metadata_url", "moments_dataset", "to_darfix_dataset"]


def data_urls(scan: Any) -> list[str]:
    """``file.h5::/path`` of every detector file of a scan (the first one seeds darfix)."""
    source = scan.source
    files = [f for f in source.files() if f != getattr(source, "master", None)]
    frames_path = getattr(source, "frames_path", None)
    if frames_path is None:
        raise ValueError("darfix export needs an HDF5 frame-stack source (hdf5_stack)")
    return [f"{Path(f)}::{frames_path}" for f in files]


def metadata_url(scan: Any) -> str:
    """``master.h5::/N.1/instrument/positioners`` of the scan."""
    source = scan.source
    master = getattr(source, "master", None)
    if master is None:
        raise ValueError("darfix export needs a scan with a master file (hdf5_stack)")
    positioners = source.spec.motor_path("scalar", scan=source.scan, motor="").rstrip("/")
    return f"{Path(master)}::/{positioners.lstrip('/')}"


def to_darfix_dataset(scan: Any, work_dir: str | Path, in_memory: bool = True) -> Any:
    """A darfix ``Dataset`` over the scan's detector files (needs ``darfix``)."""
    try:
        from darfix.core.dataset import Dataset
    except ImportError as exc:
        raise ImportError("darfix is not installed; pip install darfix") from exc
    urls = data_urls(scan)
    return Dataset(
        _dir=str(work_dir),
        first_filename=urls[0],
        metadata_url=metadata_url(scan),
        isH5=True,
        in_memory=in_memory,
    )


def compute_moments(
    values: Any, frames: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(com, std, skew, kurtosis)`` maps of a frame stack along its motor values.

    The same numbers darfix's ``compute_moments`` produces (see
    :func:`bexa.analysis.rocking.moments`), without darfix.
    """
    from bexa.analysis.rocking import moments

    stats = moments(np.asarray(frames, dtype=float), np.asarray(values, dtype=float), clip=None)
    return stats["com"], stats["std"], stats["skew"], stats["kurtosis"]


def moments_dataset(
    moments: dict[str, Any] | Any, dims: list[str] | tuple[str, ...] | None = None
) -> xr.Dataset:
    """darfix ``apply_moments`` output (per motor: a ``(4, y, x)`` array) as an xarray Dataset.

    Variables are ``com_<motor>``, ``std_<motor>``, ``skew_<motor>``, ``kurtosis_<motor>``.
    A bare array (or list of arrays) is labelled with ``dims``.
    """
    if not isinstance(moments, dict):
        arrays = list(moments)
        names = list(dims) if dims else [f"dim_{i}" for i in range(len(arrays))]
        moments = dict(zip(names, arrays, strict=True))
    data_vars = {}
    for motor, block in moments.items():
        block = np.asarray(block, dtype=float)
        if block.ndim != 3 or block.shape[0] != 4:
            raise ValueError(f"moments of {motor!r} must have shape (4, y, x), got {block.shape}")
        for name, plane in zip(MOMENT_NAMES, block, strict=True):
            data_vars[f"{name}_{motor}"] = xr.DataArray(plane, dims=("y", "x"))
    ds = xr.Dataset(data_vars)
    ds.attrs.update(build_attrs(parameters={"source": "darfix.apply_moments"}))
    return ds
