"""Engines: one module per kind of file structure.

Importing this package registers the built-in engines with
:data:`bexa.core.registry.engines`. Engines that need an optional
dependency register themselves only when that dependency is importable.
"""

from __future__ import annotations

import contextlib

from bexa.io import generic  # noqa: F401  (registers the plain-array engine)
from bexa.io.engines import hdf5_stack  # noqa: F401  (registers hdf5_stack)

for _optional in ("hdf5_points", "spec_h5", "array_cube", "smalldata", "extra_data"):
    with contextlib.suppress(ImportError):
        __import__(f"bexa.io.engines.{_optional}")
