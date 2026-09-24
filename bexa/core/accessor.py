"""``.bexa`` on xarray objects: plot, browse, ROI and save one attribute away.

The accessors are registered when :mod:`bexa.core.reductions` or
:mod:`bexa.io.cube` is imported, so every array bexa produces has them::

    res = scan.reduce([bexa.acc.MotorCOM()])
    res["com_mu"].bexa.plot()          # the COM map with the right colours
    prev.bexa.browse()                 # sliders over the motor dims
    peak = res["sum"].bexa.roi(y=(10, 40), x=(20, 60))   # in this image's pixels

The plotting lives in :mod:`bexa.viz`; the methods import it on first use,
the one place a core module reaches up into the viz layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import xarray as xr

from bexa.core.roi import ROI

__all__ = ["ArrayAccessor", "DatasetAccessor"]


@xr.register_dataarray_accessor("bexa")
class ArrayAccessor:
    """``DataArray.bexa``: see the module docstring."""

    def __init__(self, obj: xr.DataArray) -> None:
        self._obj = obj

    def plot(self, kind: str | None = None, **kwargs: Any) -> Any:
        """The figure that fits this array (see :func:`bexa.viz.auto.plot`)."""
        from bexa.viz.auto import plot

        return plot(self._obj, kind=kind, **kwargs)

    def browse(self, **kwargs: Any) -> Any:
        """Sliders over the motor dims (see :class:`bexa.viz.interactive.Browser`)."""
        from bexa.viz.interactive import browse

        return browse(self._obj, **kwargs)

    def browse_volume(self, **kwargs: Any) -> Any:
        """A 3-D view with sliders for the extra dims (see ``bexa.viz.interactive``)."""
        from bexa.viz.interactive import browse_volume

        return browse_volume(self._obj, **kwargs)

    def roi(
        self,
        y: tuple[int | None, int | None] | None = None,
        x: tuple[int | None, int | None] | None = None,
    ) -> ROI:
        """An ROI given in this array's pixel indices, expressed in full-resolution pixels."""
        return ROI.from_indices(self._obj, y=y, x=x)

    def save(self, path: str | Path, **kwargs: Any) -> Path:
        """Write the array with :func:`bexa.save`."""
        from bexa.io.cube import save

        return save(self._obj, path, **kwargs)


@xr.register_dataset_accessor("bexa")
class DatasetAccessor:
    """``Dataset.bexa``: a panel of every map, or the laser on/off figure for a cube."""

    def __init__(self, obj: xr.Dataset) -> None:
        self._obj = obj

    def plot(self, kind: str | None = None, **kwargs: Any) -> Any:
        from bexa.viz.auto import plot

        return plot(self._obj, kind=kind, **kwargs)

    def save(self, path: str | Path, **kwargs: Any) -> Path:
        from bexa.io.cube import save

        return save(self._obj, path, **kwargs)
