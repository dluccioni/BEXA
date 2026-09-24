"""bexa: Beamline EXperiment Analysis.

Read, reduce and visualise X-ray beamtime data (dark-field microscopy at
synchrotrons, pump-probe diffraction at free-electron lasers) from a terminal,
a notebook or a batch job, on machines with any amount of memory and with or
without a GPU.

Everyday entry points::

    import bexa
    scan = bexa.open_profile("hc6293", sample="WS2G_100", scan=7)
    prev = scan.preview(downsample=(1, 1, 4, 4))
    res = bexa.reduce(scan, [bexa.acc.Sum(), bexa.acc.MotorCOM(axes=("chi", "mu"))])

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
    "Scan",
    "__version__",
    "acc",
    "analysis",
    "crystal",
    "geometry",
    "help",
    "io",
    "load",
    "notebook",
    "open",
    "open_profile",
    "optics",
    "pipelines",
    "profile",
    "reduce",
    "save",
    "testing",
    "to_host",
    "viz",
]

# Public attribute -> (module, attribute) resolved on first access.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "open": ("bexa.core.scan", "open"),
    "open_profile": ("bexa.core.scan", "open_profile"),
    "Scan": ("bexa.core.scan", "Scan"),
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
    from bexa.core.reductions import reduce
    from bexa.core.roi import ROI
    from bexa.core.scan import Scan, open, open_profile
    from bexa.io.cube import load, save


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
