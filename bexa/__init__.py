"""bexa: Beamline EXperiment Analysis.

Read, reduce and visualise X-ray beamtime data (dark-field microscopy at
synchrotrons, pump-probe diffraction at free-electron lasers) from a terminal,
a notebook or a batch job, on machines with any amount of memory and with or
without a GPU.

Everyday entry points::

    import bexa
    scan = bexa.open(profile="hc6293", sample="WS2G_100", scan=7)   # or bexa.open(path, scan=7)
    prev = scan.preview(downsample=(1, 1, 4, 4))
    maps = scan.com(axes=("chi", "mu"))          # com_chi, com_mu, width_..., total
    bexa.plot(maps["com_mu"])                    # or maps["com_mu"].bexa.plot()
    ds = bexa.open_dataset(profile="hc6293", sample="WS2G_100")   # every scan of the dataset
    ds.table(); ds.select(type="fscan2d"); bexa.stack(ds[[1, 7, 13]], [bexa.acc.Sum()], dim="samz")

Submodules are imported lazily so that ``import bexa`` stays fast; heavy
optional dependencies (cupy, napari, lmfit, ...) load only inside the
functions that need them.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "1.0.0.dev0"
__all__ = [
    "ROI",
    "Dataset",
    "Scan",
    "__version__",
    "acc",
    "analysis",
    "common_grid",
    "crystal",
    "demo",
    "geometry",
    "help",
    "io",
    "list_scans",
    "load",
    "notebook",
    "open",
    "open_dataset",
    "open_profile",
    "optics",
    "pipelines",
    "plot",
    "profile",
    "reduce",
    "regrid",
    "save",
    "settings",
    "stack",
    "testing",
    "to_host",
    "viz",
]

# Public attribute -> (module, attribute) resolved on first access.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "open": ("bexa.core.scan", "open"),
    "open_profile": ("bexa.core.scan", "open_profile"),
    "Scan": ("bexa.core.scan", "Scan"),
    "list_scans": ("bexa.core.scan", "list_scans"),
    "Dataset": ("bexa.core.dataset", "Dataset"),
    "open_dataset": ("bexa.core.dataset", "open_dataset"),
    "stack": ("bexa.core.dataset", "stack"),
    "common_grid": ("bexa.core.dataset", "common_grid"),
    "regrid": ("bexa.core.dataset", "regrid"),
    "settings": ("bexa.core.settings", "show"),
    "plot": ("bexa.viz.auto", "plot"),
    "demo": ("bexa.testing.demo", "demo"),
    "ROI": ("bexa.core.roi", "ROI"),
    "reduce": ("bexa.core.reductions", "reduce"),
    "load": ("bexa.io.cube", "load"),
    "save": ("bexa.io.cube", "save"),
    "to_host": ("bexa.core.backend", "to_host"),
    "profile": ("bexa._profiling", "profile"),
    "help": ("bexa._help", "help"),
}

# Public attribute -> submodule imported on first access.
_LAZY_MODULES: dict[str, str] = {
    "acc": "bexa.core.reductions",
    "analysis": "bexa.analysis",
    "crystal": "bexa.crystal",
    "geometry": "bexa.geometry",
    "io": "bexa.io",
    "notebook": "bexa.notebook",
    "optics": "bexa.optics",
    "pipelines": "bexa.pipelines",
    "testing": "bexa.testing",
    "viz": "bexa.viz",
}

if TYPE_CHECKING:  # pragma: no cover - static analysis only
    from bexa._help import help
    from bexa._profiling import profile
    from bexa.core.backend import to_host
    from bexa.core.dataset import Dataset, common_grid, open_dataset, regrid, stack
    from bexa.core.reductions import reduce
    from bexa.core.roi import ROI
    from bexa.core.scan import Scan, list_scans, open, open_profile
    from bexa.core.settings import show as settings
    from bexa.io.cube import load, save
    from bexa.testing.demo import demo
    from bexa.viz.auto import plot


def __getattr__(name: str) -> Any:
    """Resolve public names lazily (PEP 562)."""
    if name in _LAZY_ATTRS:
        module_name, attr = _LAZY_ATTRS[name]
        value = getattr(importlib.import_module(module_name), attr)
    elif name in _LAZY_MODULES:
        value = importlib.import_module(_LAZY_MODULES[name])
    else:
        raise AttributeError(f"module 'bexa' has no attribute {name!r}")
    globals()[name] = value  # cache so the next access is a plain lookup
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
