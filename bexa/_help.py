"""``bexa.help()``: the everyday names first, the whole public API on request."""

from __future__ import annotations

import importlib
import inspect
from typing import Any

# What a notebook session uses most, in the order one meets it: (shown as, module, attribute).
_EVERYDAY: list[tuple[str, str, str]] = [
    ("bexa.open(...)", "bexa.core.scan", "open"),
    ("bexa.open_dataset(...)", "bexa.core.dataset", "open_dataset"),
    ("bexa.list_scans(path)", "bexa.core.scan", "list_scans"),
    ("bexa.demo(kind)", "bexa.testing.demo", "demo"),
    ("scan.info()", "bexa.core.scan", "Scan.info"),
    ("scan.preview(...)", "bexa.core.scan", "Scan.preview"),
    ("scan.sum(...)", "bexa.core.scan", "Scan.sum"),
    ("scan.rocking_curve(...)", "bexa.core.scan", "Scan.rocking_curve"),
    ("scan.com(...)", "bexa.core.scan", "Scan.com"),
    ("scan.energy_com(...)", "bexa.core.scan", "Scan.energy_com"),
    ("scan.stats(...)", "bexa.core.scan", "Scan.stats"),
    ("scan.reduce(...)", "bexa.core.scan", "Scan.reduce"),
    ("scan.read(...)", "bexa.core.scan", "Scan.read"),
    ("bexa.ROI(...)", "bexa.core.roi", "ROI"),
    ("bexa.ROI.parse(text)", "bexa.core.roi", "ROI.parse"),
    ("image.bexa.roi(...)", "bexa.core.roi", "ROI.from_indices"),
    ("ds.table()", "bexa.core.dataset", "Dataset.table"),
    ("ds.select(...)", "bexa.core.dataset", "Dataset.select_numbers"),
    ("ds.series(...)", "bexa.core.dataset", "Dataset.series"),
    ("ds.groups(by)", "bexa.core.dataset", "Dataset.groups"),
    ("bexa.stack(...)", "bexa.core.dataset", "stack"),
    ("bexa.plot(obj)", "bexa.viz.auto", "plot"),
    ("array.bexa.browse()", "bexa.viz.interactive", "browse"),
    ("array.bexa.browse_volume()", "bexa.viz.interactive", "browse_volume"),
    ("bexa.save(obj, path)", "bexa.io.cube", "save"),
    ("bexa.load(path)", "bexa.io.cube", "load"),
    ("bexa.pipelines.dfxm_report.report(...)", "bexa.pipelines.dfxm_report", "report"),
    ("bexa.settings()", "bexa.core.settings", "show"),
]

_SECTIONS: list[tuple[str, str]] = [
    ("Open and reduce", "bexa.core.scan"),
    ("Datasets", "bexa.core.dataset"),
    ("Accumulators (bexa.acc)", "bexa.core.reductions"),
    ("Regions of interest", "bexa.core.roi"),
    ("Saving and loading", "bexa.io.cube"),
    ("Format specs", "bexa.io.formats"),
    ("Settings", "bexa.core.settings"),
    ("Analysis: preprocessing", "bexa.analysis.preprocess"),
    ("Analysis: rocking curves", "bexa.analysis.rocking"),
    ("Analysis: peaks", "bexa.analysis.peaks"),
    ("Analysis: pump-probe", "bexa.analysis.pump_probe"),
    ("Analysis: registration", "bexa.analysis.registration"),
    ("Analysis: fitting", "bexa.analysis.fitting"),
    ("Analysis: segmentation", "bexa.analysis.segmentation"),
    ("Analysis: fields", "bexa.analysis.fields"),
    ("Analysis: projections", "bexa.analysis.projections"),
    ("Geometry", "bexa.geometry.transforms"),
    ("Crystal", "bexa.crystal.lattice"),
    ("Diffractometer", "bexa.crystal.diffractometer"),
    ("Zone axis", "bexa.crystal.zone_axis"),
    ("Omega search", "bexa.crystal.omega_search"),
    ("Optics: conversions", "bexa.optics.conversions"),
    ("Optics: attenuation", "bexa.optics.attenuation"),
    ("Optics: Bragg magnifier", "bexa.optics.bragg_magnifier"),
    ("Plots: auto", "bexa.viz.auto"),
    ("Plots: images", "bexa.viz.images"),
    ("Plots: curves", "bexa.viz.curves"),
    ("Plots: maps", "bexa.viz.maps"),
    ("Plots: volumes", "bexa.viz.volume"),
    ("Plots: interactive", "bexa.viz.interactive"),
    ("Plots: napari", "bexa.viz.napari_app"),
    ("Animation", "bexa.viz.animation"),
    ("Pipelines: DFXM report", "bexa.pipelines.dfxm_report"),
    ("Pipelines: XFEL cube", "bexa.pipelines.xfel_cube"),
    ("Pipelines: cube figures", "bexa.pipelines.cube_report"),
    ("Pipelines: watcher", "bexa.pipelines.auto_process"),
    ("Pipelines: benchmark", "bexa.pipelines.benchmark"),
    ("Synthetic data", "bexa.testing.demo"),
]


def _first_line(obj: object) -> str:
    doc = inspect.getdoc(obj) or ""
    return doc.splitlines()[0] if doc else ""


def _signature(obj: Any, limit: int = 60) -> str:
    try:
        sig = str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "(...)"
    return sig if len(sig) <= limit else sig[: limit - 3] + "..."


def _resolve(module_name: str, attribute: str) -> Any:
    obj: Any = importlib.import_module(module_name)
    for part in attribute.split("."):
        obj = getattr(obj, part)
    return obj


def _everyday() -> list[str]:
    lines = [
        "Everyday names (bexa.help('all') lists every public function, "
        "bexa.help('plots') one group)"
    ]
    for shown, module_name, attribute in _EVERYDAY:
        try:
            obj = _resolve(module_name, attribute)
        except Exception:  # an optional dependency is missing
            continue
        lines.append(f"    {shown:42s} {_first_line(obj)}")
    lines.append(
        "    bexa.acc                                   Sum, Mean, Max, Min, Preview, Projections, "
        "RoiIntegral, MotorCOM, EnergyCOM, ArgmaxMotor, FrameStats, Histogram, OnOffSplit"
    )
    lines.append("")
    return lines


def _sections(section: str | None, verbose: bool) -> list[str]:
    lines: list[str] = []
    for title, module_name in _SECTIONS:
        if section and section.lower() not in title.lower():
            continue
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # optional dependency missing, module not built yet
            lines.append(f"{title}  ({module_name}: unavailable, {exc})")
            continue
        lines.append(f"{title}  ({module_name})")
        public = getattr(module, "__all__", None) or [
            n for n in dir(module) if verbose or not n.startswith("_")
        ]
        for name in public:
            obj = getattr(module, name, None)
            if obj is None or inspect.ismodule(obj):
                continue
            if not (inspect.isfunction(obj) or inspect.isclass(obj)):
                continue
            if getattr(obj, "__module__", module_name) != module_name and not verbose:
                continue
            lines.append(f"    {name}{_signature(obj)}")
            summary = _first_line(obj)
            if summary:
                lines.append(f"        {summary}")
        lines.append("")
    return lines


def help(section: str | None = None, verbose: bool = False) -> str:
    """Return (and print) a listing of the API: the everyday names, one group, or everything.

    Parameters
    ----------
    section
        ``None`` shows the everyday names; ``"all"`` every public function of
        every module; any other text the sections whose title contains it
        (``"plots"``, ``"analysis"``, ``"optics"``), case-insensitive.
    verbose
        Include private names in the full listing.
    """
    if section is None:
        lines = _everyday()
    elif section.lower() == "all":
        lines = _sections(None, verbose)
    else:
        lines = _sections(section, verbose)
    text = "\n".join(lines)
    print(text)
    return text
