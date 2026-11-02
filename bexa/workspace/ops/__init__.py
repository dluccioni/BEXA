"""The operations of a workspace: functions of data objects that return new objects.

Grouped as the GUI shows them:

- :mod:`~bexa.workspace.ops.extract`: scans and stacks to images, curves, maps, tables;
- :mod:`~bexa.workspace.ops.process`: one object in, a new one out;
- :mod:`~bexa.workspace.ops.combine`: several objects in (arithmetic, averages, mosaics);
- :mod:`~bexa.workspace.ops.multiscan`: scan sets (stacks over scans, series, layers);
- :mod:`~bexa.workspace.ops.fit`: peaks, series of fits, peak finding;
- :mod:`~bexa.workspace.ops.imaging`: per-pixel maps, orientation, KAM, strain, drift;
- :mod:`~bexa.workspace.ops.scattering`: powder integration, cakes, reflectivity;
- :mod:`~bexa.workspace.ops.calculators`: tables from numbers alone.

Each can be called directly on objects (``extract.roi_curves(obj, {"peak": roi})``);
:meth:`bexa.workspace.Workspace.run` calls them by name and logs the command.
"""

from bexa.workspace.ops import (
    calculators,
    combine,
    extract,
    fit,
    imaging,
    multiscan,
    process,
    scattering,
)

__all__ = [
    "calculators",
    "combine",
    "extract",
    "fit",
    "imaging",
    "multiscan",
    "process",
    "scattering",
]
