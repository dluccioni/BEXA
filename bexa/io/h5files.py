"""Opening HDF5 files that another program may still be writing.

At a beamline the master file of a dataset stays open in the control system
for the whole beamtime, and a detector writes its files while a scan runs.
HDF5 refuses to open a file another process holds open unless file locking is
turned off, and a file caught in the middle of a flush can fail once and open
fine a moment later. :func:`open_h5` is ``h5py.File`` with those two things
taken care of; bexa's own result files are opened with ``h5py.File`` directly.
"""

from __future__ import annotations

import functools
import inspect
import time
from pathlib import Path
from typing import Any

import h5py

__all__ = ["RETRY_DELAY", "locking_kwargs", "open_h5"]

RETRY_DELAY = 0.5  # seconds before the one retry of an open that failed


@functools.lru_cache(maxsize=1)
def locking_kwargs() -> dict[str, Any]:
    """``{"locking": False}`` when this h5py and HDF5 take it, else ``{}``.

    HDF5 1.12.1 added the per-file switch; older libraries only know the
    ``HDF5_USE_FILE_LOCKING`` environment variable, which the library reads
    by itself and which wins over this keyword when the user sets it.
    """
    version = tuple(int(v) for v in h5py.version.hdf5_version_tuple[:3])
    takes_keyword = "locking" in inspect.signature(h5py.File.__init__).parameters
    if version >= (1, 12, 1) and takes_keyword:
        return {"locking": False}
    return {}


def open_h5(path: str | Path, mode: str = "r", retry: bool = True, **kwargs: Any) -> h5py.File:
    """``h5py.File(path, mode)`` for a file another program may be writing.

    The file lock is not taken (see :func:`locking_kwargs`). When a read-only
    open fails with ``OSError`` and ``retry`` is on, it is tried once more after
    :data:`RETRY_DELAY` seconds, then in SWMR mode (single writer, multiple
    readers, which a file written that way allows); the first error is raised
    when all three fail. Other keywords go to ``h5py.File``.
    """
    options = {**locking_kwargs(), **kwargs}
    try:
        return h5py.File(path, mode, **options)
    except OSError as first:
        if not retry or mode != "r":
            raise
        time.sleep(RETRY_DELAY)
        try:
            return h5py.File(path, mode, **options)
        except OSError:
            pass
        try:
            return h5py.File(path, mode, swmr=True, **options)
        except (OSError, ValueError):
            raise first from None
