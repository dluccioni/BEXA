"""The catalog of operations: what each runs, which objects it takes, its settings and help.

The GUI builds its forms from this catalog, so every operation behaves alike:
its settings are checked, it runs in the background with progress and Cancel,
its results become objects, and the log shows the command it ran
(``ws.run("normalize", "scan_7_roi_peak", by="max")``), which does the same in
a script or a notebook cell. Nothing here imports ipywidgets.

Parameter kinds (:class:`P`):
    number, int, bool, choice, text     plain values
    numbers                             a list of numbers: "0.5, 1, 2"
    ints                                a list of whole numbers: "0, 3-5" (ranges included)
    range                               two numbers, low and high (an empty end is open)
    object, objects                     other workspace objects (``kinds`` limits which)
    dim, dims                           dims of the first input
    channel, channels                   per-frame channels of the first input (a scan)
    roi, rois                           named ROIs of the workspace
    spec                                structured values set by a tab of its own (the Fit tab's
                                        peaks); no form field
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

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
from bexa.workspace.source import source_text

__all__ = ["GROUPS", "OPS", "Input", "Op", "P", "find", "parse_ints"]

GROUPS = (
    "Extract",
    "Process",
    "Combine",
    "Multi-scan",
    "Fit",
    "Imaging",
    "XRD",
    "XRR",
    "Calculators",
)
ANY = ("stack", "image", "map", "curve", "volume")
SCAN = ("scan",)
SCANS = ("scan", "stack")
DETECTOR = ("stack", "image")
IMAGES = ("image", "map")
CURVE = ("curve",)
SHAPES = ("gaussian", "lorentzian", "pseudo_voigt", "voigt", "pearson7", "step")
BACKGROUNDS = ("none", "constant", "linear", "quadratic", "exponential")
PEAK_METHODS = ("maximum", "refined", "com")
PARSED_AS_IS = (
    "bool",
    "choice",
    "object",
    "objects",
    "dim",
    "dims",
    "channel",
    "channels",
    "roi",
    "rois",
    "spec",
)


def parse_ints(text: str, label: str = "Indices") -> list[int] | None:
    """``"0-3, 7"`` as ``[0, 1, 2, 3, 7]`` (the end of a range included); empty gives ``None``."""
    text = (text or "").strip()
    if not text:
        return None
    found: list[int] = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part[1:]:
                a, b = (int(v) for v in part.split("-", 1))
                found.extend(range(min(a, b), max(a, b) + 1))
            else:
                found.append(int(part))
        except ValueError:
            raise ValueError(f"{label}: whole numbers and ranges, such as 0-3, 7") from None
    return found


@dataclass
class P:
    """One setting of an operation."""

    name: str
    label: str = ""
    kind: str = "number"
    default: Any = None
    choices: tuple[Any, ...] = ()
    help: str = ""
    optional: bool = False
    kinds: tuple[str, ...] = ANY
    minimum: float | None = None

    def __post_init__(self) -> None:
        self.label = self.label or self.name.replace("_", " ").capitalize()
        self.choices = tuple(self.choices)

    def parse(self, raw: Any) -> Any:
        """The value of a form field (text or widget state); ``ValueError`` with a message."""
        if self.kind in PARSED_AS_IS:
            return raw if raw not in ("", None, (), []) else (None if self.optional else raw)
        text = "" if raw is None else str(raw).strip()
        if self.kind in ("int", "number"):
            if text == "":
                if self.optional:
                    return None
                raise ValueError(f"{self.label}: enter a number")
            try:
                value: float = int(float(text)) if self.kind == "int" else float(text)
            except ValueError:
                raise ValueError(f"{self.label}: {text!r} is not a number") from None
            if self.minimum is not None and value < self.minimum:
                raise ValueError(f"{self.label}: at least {self.minimum:g}")
            return value
        if self.kind == "text":
            return text
        if self.kind == "numbers":
            if not text:
                return None
            try:
                return [float(v) for v in text.replace(";", ",").split(",") if v.strip()]
            except ValueError:
                raise ValueError(f"{self.label}: numbers separated by commas") from None
        if self.kind == "ints":
            return parse_ints(text, self.label)
        if self.kind == "range":
            low, high = raw if isinstance(raw, (list, tuple)) else (raw, "")
            ends = []
            for part in (low, high):
                part = "" if part is None else str(part).strip()
                try:
                    ends.append(float(part) if part else None)
                except ValueError:
                    raise ValueError(f"{self.label}: {part!r} is not a number") from None
            if ends == [None, None]:
                return None
            if None not in ends and ends[0] > ends[1]:  # type: ignore[operator]
                ends.reverse()
            return tuple(ends)
        raise ValueError(f"unknown parameter kind {self.kind!r}")

    def text(self, value: Any) -> str:
        """The text a form field shows for ``value`` (the inverse of :meth:`parse`)."""
        if value is None:
            return ""
        if self.kind in ("numbers", "ints") and isinstance(value, (list, tuple)):
            return ", ".join(f"{v:g}" if isinstance(v, float) else str(v) for v in value)
        if isinstance(value, float):
            return f"{value:g}"
        return str(value)


@dataclass
class Input:
    """An object the operation takes: its name in the function, the kinds allowed, a label."""

    name: str
    kinds: tuple[str, ...]
    label: str = ""
    many: bool = False  # several objects, passed as a list

    def __post_init__(self) -> None:
        self.label = self.label or self.name.capitalize()


@dataclass
class Op:
    """One operation of the catalog."""

    key: str
    label: str
    group: str
    function: Callable[..., Any]
    inputs: list[Input]
    params: list[P] = field(default_factory=list)
    help: str = ""

    def param(self, name: str) -> P:
        return next(p for p in self.params if p.name == name)

    def defaults(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.params}

    def check(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """The values with the missing required settings reported (``ValueError``)."""
        out = {}
        for p in self.params:
            value = values.get(p.name, p.default)
            if value is None and not p.optional and p.kind not in ("bool",):
                raise ValueError(f"{self.label}: {p.label} is needed")
            out[p.name] = value
        unknown = set(values) - {p.name for p in self.params}
        if unknown:
            raise ValueError(f"{self.label} has no setting {sorted(unknown)}")
        return out

    def command(
        self, inputs: Sequence[Any], values: Mapping[str, Any], name: str | None = None
    ) -> str:
        """The ``ws.run(...)`` line that runs this operation (logged, and valid Python)."""
        parts = [source_text(self.key)]
        for item in inputs:
            parts.append(source_text(item))
        defaults = self.defaults()
        for key, value in values.items():
            if value is None and defaults.get(key) is None:
                continue
            parts.append(f"{key}={source_text(value)}")
        if name:
            parts.append(f"name={source_text(name)}")
        return f"ws.run({', '.join(parts)})"


def _o(*args: Any, **kwargs: Any) -> Op:
    return Op(*args, **kwargs)


ROI_HELP = "A named ROI (draw or type one on the View tab); none: the whole frame"
DOWNSAMPLE = P(
    "downsample",
    "Bin pixels while reading",
    "int",
    None,
    optional=True,
    minimum=1,
    help="Read every frame binned this many times (faster, smaller); empty: full resolution",
)

OPS: list[Op] = [
    # --- Extract ---------------------------------------------------------------------------
    _o(
        "reduce_frames",
        "Image from Frames",
        "Extract",
        extract.reduce_frames,
        [Input("obj", SCANS, "Scan or stack")],
        [
            P(
                "how",
                "Combine as",
                "choice",
                "sum",
                ("sum", "mean", "max", "min", "std", "median"),
                help="A scan streams sum, mean, max or min; std and median need a stack in memory",
            ),
            P(
                "roi",
                "ROI",
                "roi",
                optional=True,
                help=ROI_HELP + "; its motor ranges limit the frames",
            ),
            DOWNSAMPLE,
        ],
        "One image from the frames of a scan or stack: their sum (with Poisson uncertainties), "
        "mean, maximum or minimum. A scan is read in one streaming pass, whatever its size.",
    ),
    _o(
        "read_volume",
        "Load Frames",
        "Extract",
        extract.read_volume,
        [Input("obj", SCAN, "Scan")],
        [P("roi", "ROI", "roi", optional=True, help=ROI_HELP), DOWNSAMPLE],
        "The scan's frames in memory as a stack (motors..., y, x), cut to the ROI. Refused when "
        "it would not fit the memory budget: add an ROI or bin.",
    ),
    _o(
        "preview",
        "Preview Stack",
        "Extract",
        extract.preview,
        [Input("obj", SCAN, "Scan")],
        [P("bin", "Binning", "int", 4, minimum=1), P("log", "log10(1 + I)", "bool", False)],
        "The frames binned bin x bin: a light stack to browse, cached on disk for the next time.",
    ),
    _o(
        "roi_curves",
        "ROI Curves",
        "Extract",
        extract.roi_curves,
        [Input("obj", SCANS, "Scan or stack")],
        [
            P("rois", "ROIs", "rois", help="One curve (or map) per ROI, all in one pass"),
            P("quantity", "Quantity", "choice", "sum", ("sum", "mean", "max")),
            P(
                "background",
                "Background ROI",
                "roi",
                optional=True,
                help="Its mean per pixel is subtracted from every ROI, frame by frame",
            ),
            P(
                "normalize",
                "Divide by channel",
                "channel",
                optional=True,
                help="A monitor or counter recorded with every frame",
            ),
        ],
        "The intensity of each ROI at every point of the scan: a rocking curve for one motor, a "
        "map for a mesh. Uncertainties from the counts.",
    ),
    _o(
        "com_maps",
        "COM Maps",
        "Extract",
        extract.com_maps,
        [Input("obj", SCANS, "Scan or stack")],
        [
            P("axes", "Motor axes", "dims", optional=True, help="Empty: every motor of the scan"),
            P("sigma", "Smoothing (pixels)", "number", 3.0, minimum=0),
            P("roi", "ROI", "roi", optional=True),
            DOWNSAMPLE,
        ],
        "The DFXM maps: per pixel the intensity-weighted mean motor position (com_<motor>) and "
        "spread (width_<motor>), and the total intensity; one streaming pass.",
    ),
    _o(
        "frame_stats",
        "Frame Statistics",
        "Extract",
        extract.frame_stats,
        [Input("obj", SCAN, "Scan")],
        [P("roi", "ROI", "roi", optional=True)],
        "Per frame: the sum, mean, maximum and centre of mass on the detector, on the motor grid "
        "(curves for one motor, maps for two).",
    ),
    _o(
        "projections",
        "Projections",
        "Extract",
        extract.projections,
        [Input("obj", SCAN, "Scan")],
        [P("roi", "ROI", "roi", optional=True)],
        "Intensity projected on every pair of axes: the summed image, motor x pixel maps and motor "
        "x motor maps, in one pass.",
    ),
    _o(
        "counters",
        "Counters",
        "Extract",
        extract.counters,
        [Input("obj", SCAN, "Scan")],
        [
            P(
                "channels",
                "Channels",
                "channels",
                optional=True,
                help="Empty: every recorded channel",
            )
        ],
        "The per-frame channels (counters, monitors, readbacks) on the scan's grid, without "
        "reading a frame. On a scan set: one map per channel over the scans.",
    ),
    _o(
        "point_table",
        "Point Table",
        "Extract",
        extract.point_table,
        [Input("obj", SCAN, "Scan")],
        [],
        "The scan's side tables (every SPEC row of an APS scan), or its per-frame channels.",
    ),
    _o(
        "histogram",
        "Histogram",
        "Extract",
        extract.histogram,
        [Input("obj", ("scan", *ANY), "Object")],
        [
            P("bins", "Bins", "int", 256, minimum=2),
            P("low", "From", "number", optional=True),
            P("high", "To", "number", optional=True),
        ],
        "How many pixels have each value (all frames of a scan).",
    ),
    _o(
        "crop",
        "Crop to ROI",
        "Extract",
        extract.crop,
        [Input("obj", ANY, "Object")],
        [P("roi", "ROI", "roi")],
        "The part inside an ROI (pixel ranges and any other dim's range).",
    ),
    _o(
        "select_range",
        "Select Range",
        "Extract",
        extract.select_range,
        [Input("obj", ANY, "Object")],
        [
            P("dim", "Dim", "dim"),
            P("low", "From", "number", optional=True),
            P("high", "To", "number", optional=True),
        ],
        "Keep the points of one dim between two coordinate values.",
    ),
    _o(
        "take_slice",
        "Slice",
        "Extract",
        extract.take_slice,
        [Input("obj", ANY, "Object")],
        [
            P("dim", "Dim", "dim"),
            P("index", "Index", "int", optional=True, minimum=0),
            P("value", "Or nearest to value", "number", optional=True),
        ],
        "One slice along a dim: a frame of a stack, a curve of a map.",
    ),
    _o(
        "reduce_dims",
        "Reduce Dims",
        "Extract",
        extract.reduce_dims,
        [Input("obj", ANY, "Object")],
        [
            P("dims", "Dims", "dims"),
            P("how", "Combine as", "choice", "sum", ("sum", "mean", "max", "min", "std", "median")),
        ],
        "Sum (average, ...) over the chosen dims: a stack to an image, a map to a curve.",
    ),
    _o(
        "profile",
        "Profile",
        "Extract",
        extract.profile,
        [Input("obj", ANY, "Object")],
        [
            P("along", "Along", "dim"),
            P("roi", "Within ROI", "roi", optional=True),
            P("how", "Combine as", "choice", "sum", ("sum", "mean")),
        ],
        "A curve along one dim, every other dim summed inside the ROI.",
    ),
    _o(
        "line_profile",
        "Line Profile",
        "Extract",
        extract.line_profile,
        [Input("obj", IMAGES, "Image or map")],
        [
            P("y0", "From y"),
            P("x0", "From x"),
            P("y1", "To y"),
            P("x1", "To x"),
            P("width", "Width (pixels)", "int", 1, minimum=1),
            P("points", "Points", "int", optional=True, minimum=2),
        ],
        "The values along a straight line between two points (draw it on the View tab).",
    ),
    _o(
        "roi_statistics",
        "ROI Statistics",
        "Extract",
        extract.roi_statistics,
        [Input("obj", ("image", "map", "stack"), "Image or stack")],
        [P("rois", "ROIs", "rois")],
        "Sum, mean, spread, extremes and centroid of each ROI, as a table.",
    ),
    # --- Process ---------------------------------------------------------------------------
    _o(
        "normalize",
        "Normalize",
        "Process",
        process.normalize,
        [Input("obj", ANY, "Object")],
        [
            P(
                "by",
                "Divide by",
                "choice",
                "max",
                process.NORMALIZE_BY,
                help="range: (v - min) / (max - min); monitor: a curve, relative to its mean",
            ),
            P("value", "Value", "number", optional=True, help="For 'number'"),
            P("monitor", "Monitor", "object", optional=True, kinds=CURVE, help="For 'monitor'"),
            P(
                "at",
                "At x",
                "number",
                optional=True,
                help="For 'point': the x whose value becomes 'to'",
            ),
            P(
                "to",
                "Make it",
                "number",
                1.0,
                help="The value the reference becomes (100: percent)",
            ),
        ],
        "Divide by the maximum, range, integral, mean, a point, a number or a monitor curve.",
    ),
    _o(
        "subtract_dark",
        "Subtract Dark",
        "Process",
        process.subtract_dark,
        [Input("obj", (*DETECTOR, "map"), "Object")],
        [
            P("dark", "Dark", "object", kinds=(*DETECTOR, "map")),
            P("scale", "Scale", "number", 1.0),
            P("clip", "Negative values to 0", "bool", False),
        ],
        "Subtract a dark image (a dark stack is averaged first) from an image or every frame.",
    ),
    _o(
        "flat_field",
        "Flat Field",
        "Process",
        process.flat_field,
        [Input("obj", (*DETECTOR, "map"), "Object")],
        [
            P("flat", "Flat", "object", kinds=(*DETECTOR, "map")),
            P("dark", "Dark of the flat", "object", optional=True, kinds=(*DETECTOR, "map")),
            P("minimum", "Leave out below", "number", 0.05, help="Of the flat's mean"),
        ],
        "Divide by a flat field normalised to a mean of 1.",
    ),
    _o(
        "subtract_background",
        "Subtract Background",
        "Process",
        process.subtract_background,
        [Input("obj", CURVE, "Curve")],
        [
            P(
                "method",
                "Background",
                "choice",
                "linear",
                ("linear", "constant", "polynomial", "minimum", "value"),
            ),
            P(
                "regions",
                "Regions (x)",
                "numbers",
                optional=True,
                help="Pairs of x values holding background only: 1, 2, 9, 10; empty: the ends",
            ),
            P("order", "Order", "int", 2, minimum=0),
            P("value", "Value", "number", optional=True),
        ],
        "Fit a background to the background regions of a curve and subtract it.",
    ),
    _o(
        "remove_hot_pixels",
        "Remove Hot Pixels",
        "Process",
        process.remove_hot_pixels,
        [Input("obj", (*DETECTOR, "map"), "Object")],
        [
            P("threshold", "Threshold (x noise)", "number", 8.0, minimum=0),
            P("size", "Neighbourhood", "int", 3, minimum=3),
        ],
        "Replace pixels far above their neighbours by the local median (every frame of a stack).",
    ),
    _o(
        "mask_pixels",
        "Mask Pixels",
        "Process",
        process.mask_pixels,
        [Input("obj", ANY, "Object")],
        [
            P("low", "Leave out below", "number", optional=True),
            P("high", "Leave out above", "number", optional=True),
            P("rois", "Keep ROIs", "rois", optional=True),
            P("inside", "Leave out inside the ROIs instead", "bool", False),
        ],
        "Set pixels to NaN so every later step leaves them out: by value, outside or inside ROIs.",
    ),
    _o(
        "smooth",
        "Smooth",
        "Process",
        process.smooth,
        [Input("obj", ANY, "Object")],
        [
            P(
                "method",
                "Method",
                "choice",
                "gaussian",
                ("gaussian", "moving average", "savitzky-golay", "median"),
            ),
            P(
                "width",
                "Width",
                "number",
                2.0,
                minimum=0,
                help="Gaussian: sigma; others: the window",
            ),
            P("order", "Order", "int", 2, minimum=0, help="Savitzky-Golay (curves)"),
        ],
        "Smooth a curve, an image or every frame of a stack.",
    ),
    _o(
        "bin_pixels",
        "Bin Pixels",
        "Process",
        process.bin_pixels,
        [Input("obj", (*DETECTOR, "map"), "Object")],
        [
            P("factor", "Factor", "int", 2, minimum=2),
            P("how", "Combine as", "choice", "mean", ("mean", "sum")),
        ],
        "Combine factor x factor pixels into one.",
    ),
    _o(
        "transform",
        "Flip / Rotate",
        "Process",
        process.transform,
        [Input("obj", (*DETECTOR, "map", "rgb"), "Object")],
        [P("how", "Transform", "choice", "rotate 90", process.TRANSFORMS)],
        "Flip, rotate (counter-clockwise) or transpose; the coordinates follow the pixels.",
    ),
    _o(
        "apply_function",
        "Function of Values",
        "Process",
        process.apply_function,
        [Input("obj", ANY, "Object")],
        [P("function", "Function", "choice", "log10", process.FUNCTIONS)],
        "A function of every value, with the uncertainties propagated.",
    ),
    _o(
        "clip_values",
        "Clip Values",
        "Process",
        process.clip_values,
        [Input("obj", ANY, "Object")],
        [
            P("low", "Low", "number", optional=True),
            P("high", "High", "number", optional=True),
            P(
                "replace",
                "Replace with",
                "number",
                optional=True,
                help="Empty: the limit; or 0, nan",
            ),
        ],
        "Limit values to a range.",
    ),
    _o(
        "replace_nan",
        "Replace NaN",
        "Process",
        process.replace_nan,
        [Input("obj", ANY, "Object")],
        [P("value", "Value", "number", 0.0)],
        "Fill masked or missing points with a number.",
    ),
    _o(
        "resample",
        "Resample",
        "Process",
        process.resample,
        [Input("obj", CURVE, "Curve")],
        [
            P("step", "Step", "number", optional=True),
            P("points", "Points", "int", optional=True, minimum=2),
            P("low", "From x", "number", optional=True),
            P("high", "To x", "number", optional=True),
        ],
        "Interpolate a curve onto evenly spaced x.",
    ),
    _o(
        "derivative",
        "Derivative",
        "Process",
        process.derivative,
        [Input("obj", CURVE, "Curve")],
        [],
        "dy/dx with uncertainties (an edge scan becomes a peak).",
    ),
    _o(
        "cumulative_integral",
        "Running Integral",
        "Process",
        process.cumulative_integral,
        [Input("obj", CURVE, "Curve")],
        [],
        "The integral of a curve from its first point.",
    ),
    _o(
        "shift_axis",
        "Scale / Shift Axis",
        "Process",
        process.shift_axis,
        [Input("obj", ANY, "Object")],
        [
            P("dim", "Dim", "dim", optional=True, help="Empty: a curve's x"),
            P("offset", "Offset", "number", 0.0),
            P("scale", "Scale", "number", 1.0),
            P("new_name", "New name", "text", ""),
            P("units", "Units", "text", ""),
        ],
        "x -> scale x + offset along a dim, optionally renamed (a relative axis, other units).",
    ),
    _o(
        "convert_axis",
        "Convert 2theta / q / d",
        "Process",
        process.convert_axis,
        [Input("obj", CURVE, "Curve")],
        [
            P("to", "Convert to", "choice", "q", process.AXIS_UNITS),
            P("source", "Present axis", "choice", "2theta", process.AXIS_UNITS),
            P(
                "energy_keV",
                "Energy (keV)",
                "number",
                optional=True,
                help="Empty: the object's own",
            ),
        ],
        "Convert a curve's x between scattering angle, Bragg angle, momentum transfer and d.",
    ),
    _o(
        "align_peak",
        "Align Peak",
        "Process",
        process.align_peak,
        [Input("obj", CURVE, "Curve")],
        [
            P(
                "method",
                "Peak position",
                "choice",
                "maximum",
                PEAK_METHODS,
                help="refined: the top of the parabola through the highest points; "
                "com: of the top half",
            ),
            P("target", "Move the peak to", "choice", "zero", ("zero", "a value", "another curve")),
            P("value", "Value", "number", optional=True),
            P("reference", "Other curve", "object", optional=True, kinds=CURVE),
            P("x_range", "Search range", "range", optional=True),
        ],
        "Shift a curve along x so its peak sits at zero, at a value or at another curve's peak.",
    ),
    _o(
        "merge_points",
        "Merge Repeated Points",
        "Process",
        process.merge_points,
        [Input("obj", CURVE, "Curve")],
        [P("tolerance", "Tolerance (x)", "number", optional=True)],
        "Sort by x and average the points at the same x.",
    ),
    _o(
        "shift_image",
        "Shift Image",
        "Process",
        process.shift_image,
        [Input("obj", (*DETECTOR, *IMAGES), "Image")],
        [P("rows", "Rows", "number", 0.0), P("columns", "Columns", "number", 0.0)],
        "Move by fractions of a pixel.",
    ),
    _o(
        "equalize_histogram",
        "Equalize Contrast",
        "Process",
        process.equalize_histogram,
        [Input("obj", (*IMAGES, "stack"), "Image")],
        [],
        "Spread the values evenly over 0..1 by rank: faint topograph contrast becomes visible.",
    ),
    # --- Combine ---------------------------------------------------------------------------
    _o(
        "arithmetic",
        "Arithmetic",
        "Combine",
        combine.arithmetic,
        [Input("a", ANY, "A")],
        [
            P("op", "Operation", "choice", "-", combine.OPERATORS),
            P("b", "B (object)", "object", optional=True),
            P("number", "or B (number)", "number", optional=True),
            P("interpolate", "Put B on A's grid (interpolate)", "bool", True),
        ],
        "A + B, A - B, A x B, A / B with uncertainties: same shapes, an image with every frame of "
        "a stack, a curve with a value per frame, curves on other x interpolated, or a number.",
    ),
    _o(
        "average",
        "Average / Sum",
        "Combine",
        combine.average,
        [Input("objects", ANY, "Objects", many=True)],
        [P("how", "Combine as", "choice", "mean", ("mean", "sum", "weighted"))],
        "The mean, sum or uncertainty-weighted mean of the selected objects (repeated scans).",
    ),
    _o(
        "concatenate",
        "Join",
        "Combine",
        combine.concatenate,
        [Input("objects", ANY, "Objects", many=True)],
        [P("dim", "Along dim", "dim", optional=True)],
        "Join along a dim: curves point after point, stacks frame after frame.",
    ),
    _o(
        "stack_objects",
        "Stack Objects",
        "Combine",
        combine.stack_objects,
        [Input("objects", ANY, "Objects", many=True)],
        [P("dim", "New dim", "text", "index"), P("values", "Its values", "numbers", optional=True)],
        "Objects of one shape along a new dim: images to a stack, curves to a map (a waterfall).",
    ),
    _o(
        "stitch_curves",
        "Stitch Curves",
        "Combine",
        combine.stitch_curves,
        [Input("objects", CURVE, "Curves", many=True)],
        [
            P("scale", "Scale each to match", "bool", True),
            P("overlap", "Where they overlap", "choice", "average", ("average", "first", "last")),
        ],
        "One curve from curves of neighbouring ranges (XRR segments, 2theta ranges).",
    ),
    _o(
        "align_peaks",
        "Match Peaks",
        "Combine",
        combine.align_peaks,
        [Input("objects", CURVE, "Curves", many=True)],
        [
            P("method", "Peak position", "choice", "maximum", PEAK_METHODS),
            P("target", "Put the peaks at", "choice", "first", ("first", "zero", "a value")),
            P("value", "Value", "number", optional=True),
            P("x_range", "Search range", "range", optional=True),
        ],
        "Shift curves so their peaks coincide; Average / Sum can then add them.",
    ),
    _o(
        "align_images",
        "Register Images",
        "Combine",
        combine.align_images,
        [Input("objects", IMAGES, "Images", many=True)],
        [P("max_shift", "Largest shift (pixels)", "int", optional=True, minimum=0)],
        "Shift images onto the first by phase correlation.",
    ),
    _o(
        "mosaic",
        "Mosaic",
        "Combine",
        combine.mosaic,
        [Input("objects", IMAGES, "Images", many=True)],
        [
            P(
                "offsets",
                "Offsets (pixels)",
                "numbers",
                optional=True,
                help="Row, column per image",
            ),
            P("rows", "Or motor along rows", "text", ""),
            P("columns", "Motor along columns", "text", ""),
            P("pixel_mm", "Pixel at the sample (mm)", "number", 0.0),
            P("how", "Overlaps", "choice", "mean", ("mean", "max", "sum")),
        ],
        "Images placed side by side by pixel offsets or by their recorded stage positions.",
    ),
    _o(
        "merge_scan_sets",
        "Merge Scans",
        "Combine",
        combine.merge_scan_sets,
        [Input("objects", ("scan", "scanset"), "Scans", many=True)],
        [],
        "One scan set from scans and scan sets, to process them together.",
    ),
    # --- Multi-scan ------------------------------------------------------------------------
    _o(
        "stack_scans",
        "Stack Scans",
        "Multi-scan",
        multiscan.stack_scans,
        [Input("obj", ("scanset",), "Scan set")],
        [
            P(
                "quantities",
                "Quantities",
                "choice",
                "sum,com",
                (
                    "sum,com",
                    "sum",
                    "max",
                    "mean",
                    "com",
                    "stats",
                    "preview",
                    "sum,max,com,stats",
                    "argmax",
                    "energy_com",
                ),
                help="What every scan gives, stacked over the scans",
            ),
            P(
                "dim",
                "Stack on",
                "text",
                "auto",
                help="auto: the positioners that differ (height, energy, temperature); "
                "a name; scan",
            ),
            P("sigma", "COM smoothing (pixels)", "number", 3.0, minimum=0),
            P("roi", "ROI", "roi", optional=True),
            DOWNSAMPLE,
            P("preview_bin", "Preview binning", "int", 8, minimum=1),
            P("rois", "ROI curves too", "rois", optional=True),
            P("workers", "Parallel processes", "int", optional=True, minimum=1),
        ],
        "Reduce every scan of a set the same way (one streaming pass each, in parallel with "
        "workers) and stack the results on the positioners that differ: z-stacks, energy "
        "series, temperature series.",
    ),
    _o(
        "series",
        "Series (one scan)",
        "Multi-scan",
        multiscan.series,
        [Input("obj", ("scanset",), "Scan set")],
        [P("dim", "New dim", "text", "auto", help="auto, energy, scan or a positioner")],
        "Join scans that share a grid into one scan with a new leading dim: an energy series "
        "becomes energy x motors x pixels, read and reduced as one scan.",
    ),
    _o(
        "scan_table",
        "Scan Table",
        "Multi-scan",
        multiscan.scan_table,
        [Input("obj", ("scanset",), "Scan set")],
        [],
        "One row per scan: type, motors, grid, frames, energy and every positioner.",
    ),
    _o(
        "subset",
        "Subset",
        "Multi-scan",
        multiscan.subset,
        [Input("obj", ("scanset",), "Scan set")],
        [P("indices", "Positions", "ints", help="From 0, in the set's order: 0-3, 7")],
        "Some scans of a set as a set of their own.",
    ),
    _o(
        "layers",
        "Split into Layers",
        "Multi-scan",
        multiscan.layers,
        [Input("obj", SCAN, "Scan")],
        [P("dim", "Along", "dim")],
        "A scan split along one motor, one layer per point, as a scan set (a z-scan's heights).",
    ),
    # --- Fit -------------------------------------------------------------------------------
    _o(
        "fit_peaks",
        "Fit Peaks",
        "Fit",
        fit.fit_peaks,
        [Input("obj", CURVE, "Curve")],
        [
            P("shape", "Shape", "choice", "gaussian", SHAPES),
            P("n_peaks", "Peaks", "int", 1, minimum=1),
            P("background", "Background", "choice", "linear", BACKGROUNDS),
            P(
                "centers",
                "Centres",
                "numbers",
                optional=True,
                help="Empty: the most prominent peaks",
            ),
            P("x_range", "Fit range", "range", optional=True),
            P("use_sigma", "Weight by the uncertainties", "bool", True),
            P("components", "Each peak as a curve too", "bool", False),
            P("peaks", "Peaks", "spec", optional=True, help="Set on the Fit tab"),
            P(
                "background_params",
                "Background start",
                "spec",
                optional=True,
                help="Set on the Fit tab",
            ),
        ],
        "Peaks on a background: the best fit and a table of centres, widths, heights, areas with "
        "errors. The Fit tab gives every parameter with bounds and fixed values.",
    ),
    _o(
        "fit_series",
        "Fit Series",
        "Fit",
        fit.fit_series,
        [Input("obj", ("map", "volume", "stack", "image"), "Map")],
        [
            P("along", "Fit along", "dim"),
            P("shape", "Shape", "choice", "gaussian", SHAPES),
            P("n_peaks", "Peaks", "int", 1, minimum=1),
            P("background", "Background", "choice", "linear", BACKGROUNDS),
            P("centers", "Centres", "numbers", optional=True),
            P("x_range", "Fit range", "range", optional=True),
            P("warm_start", "Start from the previous fit", "bool", True),
            P("peaks", "Peaks", "spec", optional=True, help="Set on the Fit tab"),
        ],
        "Fit every curve of a map along a dim: the peak's centre, width, height and area against "
        "the other dims (a rocking curve per height, a peak per temperature).",
    ),
    _o(
        "fit_many",
        "Fit Each Curve",
        "Fit",
        fit.fit_many,
        [Input("objects", CURVE, "Curves", many=True)],
        [
            P("shape", "Shape", "choice", "gaussian", SHAPES),
            P("n_peaks", "Peaks", "int", 1, minimum=1),
            P("background", "Background", "choice", "linear", BACKGROUNDS),
            P("centers", "Centres", "numbers", optional=True),
            P("x_range", "Fit range", "range", optional=True),
            P("use_sigma", "Weight by the uncertainties", "bool", True),
        ],
        "The same model fitted to every selected curve: one table, a row per curve.",
    ),
    _o(
        "find_peaks",
        "Find Peaks",
        "Fit",
        fit.find_peaks,
        [Input("obj", CURVE, "Curve")],
        [
            P("n", "At most", "int", 5, minimum=1),
            P("prominence", "Least prominence", "number", 0.0),
        ],
        "The most prominent maxima: position, height, FWHM.",
    ),
    # --- Imaging ---------------------------------------------------------------------------
    _o(
        "moments_maps",
        "Moment Maps",
        "Imaging",
        imaging.moments_maps,
        [Input("obj", ("stack",), "Stack")],
        [P("dim", "Along", "dim"), P("sigma", "Smoothing (pixels)", "number", 0.0, minimum=0)],
        "Per pixel: centre of mass, spread, FWHM, skewness and kurtosis along one motor.",
    ),
    _o(
        "peak_maps",
        "Gaussian Peak Maps",
        "Imaging",
        imaging.peak_maps,
        [Input("obj", ("stack",), "Stack")],
        [
            P("dim", "Along", "dim"),
            P("threshold", "Points above (of the maximum)", "number", 0.1, minimum=0),
        ],
        "A Gaussian fitted to every pixel's curve, all pixels at once: centre, FWHM, height maps.",
    ),
    _o(
        "argmax_map",
        "Brightest-Frame Map",
        "Imaging",
        imaging.argmax_map,
        [Input("obj", ("stack",), "Stack")],
        [P("dim", "Along", "dim")],
        "The motor value of the brightest frame at every pixel.",
    ),
    _o(
        "orientation_map",
        "Orientation Map",
        "Imaging",
        imaging.orientation_map,
        [Input("a", IMAGES, "Tilt map A"), Input("b", IMAGES, "Tilt map B")],
        [
            P("span", "Full colour at", "number", optional=True, help="Empty: the 95th percentile"),
            P("intensity", "Brightness from", "object", optional=True, kinds=IMAGES),
        ],
        "Two tilt (COM) maps as one colour picture: hue the direction, saturation the size.",
    ),
    _o(
        "kam_map",
        "KAM",
        "Imaging",
        imaging.kam_map,
        [Input("a", IMAGES, "Tilt map A")],
        [
            P("b", "Tilt map B", "object", optional=True, kinds=IMAGES),
            P("size", "Kernel", "int", 3, minimum=3),
        ],
        "Kernel average misorientation: the mean tilt difference to each pixel's neighbours.",
    ),
    _o(
        "strain",
        "Strain",
        "Imaging",
        imaging.strain,
        [Input("obj", IMAGES, "COM map")],
        [
            P("kind", "COM of", "choice", "theta", imaging.STRAIN_KINDS),
            P("reference", "Unstrained value", "number", optional=True, help="Empty: the median"),
            P("theta_deg", "Bragg angle (deg)", "number", optional=True),
        ],
        "Lattice strain from a Bragg-angle, 2theta or energy centre-of-mass map.",
    ),
    _o(
        "align_frames",
        "Drift Correction",
        "Imaging",
        imaging.align_frames,
        [Input("obj", ("stack",), "Stack")],
        [P("dim", "Along", "dim"), P("max_shift", "Largest shift (pixels)", "int", 5, minimum=1)],
        "Shift every frame onto the first by phase correlation.",
    ),
    _o(
        "segment",
        "Segment",
        "Imaging",
        imaging.segment,
        [Input("a", IMAGES, "Tilt map A"), Input("b", IMAGES, "Tilt map B")],
        [
            P("clusters", "Clusters", "int", 4, minimum=2),
            P("method", "Method", "choice", "kmeans", ("kmeans", "dbscan")),
        ],
        "Grains or domains: pixels clustered by their two tilt maps (needs scikit-learn).",
    ),
    _o(
        "gradient_maps",
        "Gradients",
        "Imaging",
        imaging.gradient_maps,
        [Input("obj", IMAGES, "Map")],
        [],
        "The derivatives of a map along its two dims.",
    ),
    # --- XRD -------------------------------------------------------------------------------
    _o(
        "integrate_powder",
        "Azimuthal Integration",
        "XRD",
        scattering.integrate_powder,
        [Input("obj", (*DETECTOR, "map"), "Image or stack")],
        [
            P("distance_mm", "Distance (mm)", "number", 200.0, minimum=0),
            P("center_y", "Beam centre row (px)", "number", 0.0),
            P("center_x", "Beam centre column (px)", "number", 0.0),
            P("pixel_um", "Pixel (um)", "number", 75.0, minimum=0),
            P(
                "energy_keV",
                "Energy (keV)",
                "number",
                optional=True,
                help="Empty: the object's own",
            ),
            P("unit", "x axis", "choice", "2theta", ("2theta", "q", "d")),
            P("bins", "Bins", "int", optional=True, minimum=2),
            P("x_range", "Range", "range", optional=True),
            P("chi_range", "Azimuth (deg)", "range", optional=True),
            P("tilt_deg", "Detector tilt (deg)", "number", 0.0),
            P("tilt_plane_deg", "Tilt plane (deg)", "number", 0.0),
            P("per_frame", "One pattern per frame (a map)", "bool", False),
            P("solid_angle", "Solid-angle correction", "bool", True),
            P(
                "polarization",
                "Polarisation factor",
                "number",
                0.0,
                optional=True,
                help="0 unpolarised, 0.95-0.99 a synchrotron; empty: no correction",
            ),
        ],
        "The powder pattern: mean intensity in rings of equal 2theta, q or d, with uncertainties.",
    ),
    _o(
        "cake",
        "Cake (2theta-chi)",
        "XRD",
        scattering.cake,
        [Input("obj", (*DETECTOR, "map"), "Image or stack")],
        [
            P("distance_mm", "Distance (mm)", "number", 200.0, minimum=0),
            P("center_y", "Beam centre row (px)", "number", 0.0),
            P("center_x", "Beam centre column (px)", "number", 0.0),
            P("pixel_um", "Pixel (um)", "number", 75.0, minimum=0),
            P("energy_keV", "Energy (keV)", "number", optional=True),
            P("unit", "x axis", "choice", "2theta", ("2theta", "q", "d")),
            P("bins", "Bins", "int", optional=True, minimum=2),
            P("chi_bins", "Azimuth bins", "int", 180, minimum=4),
            P("x_range", "Range", "range", optional=True),
            P("tilt_deg", "Detector tilt (deg)", "number", 0.0),
            P("tilt_plane_deg", "Tilt plane (deg)", "number", 0.0),
        ],
        "The image unrolled into azimuth against 2theta: texture and strain show along chi.",
    ),
    _o(
        "calibrate_ring",
        "Calibrate from a Ring",
        "XRD",
        scattering.calibrate_ring,
        [Input("obj", ("image",), "Image")],
        [
            P("d_spacing_A", "Ring d-spacing (A)", "number", 3.1356, minimum=0),
            P("pixel_um", "Pixel (um)", "number", 75.0, minimum=0),
            P("energy_keV", "Energy (keV)", "number", optional=True),
            P("percentile", "Ring pixels above percentile", "number", 99.0),
        ],
        "Beam centre and distance from one powder ring of a known reflection (Si 111: 3.1356 A).",
    ),
    # --- XRR -------------------------------------------------------------------------------
    _o(
        "reflectivity",
        "Reflectivity",
        "XRR",
        scattering.reflectivity,
        [Input("obj", CURVE, "Counter curve")],
        [
            P("angle", "x is", "choice", "theta", ("theta", "two_theta")),
            P("i0", "Direct beam I0", "number", optional=True),
            P("beam_mm", "Beam size (mm)", "number", optional=True),
            P("sample_mm", "Sample length (mm)", "number", optional=True),
            P("profile", "Beam profile", "choice", "gaussian", ("gaussian", "rectangular")),
            P("energy_keV", "Energy (keV)", "number", optional=True),
            P("x", "Against", "choice", "q", ("q", "theta")),
            P("theta_offset", "theta offset (deg)", "number", 0.0),
        ],
        "A specular reflectivity curve: over I0, footprint-corrected, against q_z or theta.",
    ),
    _o(
        "footprint_correction",
        "Footprint Correction",
        "XRR",
        scattering.footprint_correction,
        [Input("obj", CURVE, "XRR curve")],
        [
            P("beam_mm", "Beam size (mm)", "number", 0.1),
            P("sample_mm", "Sample length (mm)", "number", 10.0),
            P("profile", "Beam profile", "choice", "gaussian", ("gaussian", "rectangular")),
            P("energy_keV", "Energy (keV)", "number", optional=True),
        ],
        "Divide by the share of the beam that falls on the sample.",
    ),
    _o(
        "critical_angle",
        "Critical Angle",
        "XRR",
        scattering.critical_angle,
        [Input("obj", CURVE, "XRR curve")],
        [P("energy_keV", "Energy (keV)", "number", optional=True)],
        "Where the reflectivity falls fastest: theta_c, q_c and the electron density.",
    ),
    _o(
        "kiessig_thickness",
        "Film Thickness",
        "XRR",
        scattering.kiessig_thickness,
        [Input("obj", CURVE, "XRR curve (q)")],
        [
            P("q_low", "Fringes from q", "number", optional=True),
            P("q_high", "to q", "number", optional=True),
        ],
        "The thickness from the spacing of the Kiessig fringes (2 pi / delta q).",
    ),
    # --- Calculators -----------------------------------------------------------------------
    _o(
        "energy",
        "Energy and Wavelength",
        "Calculators",
        calculators.energy,
        [],
        [
            P("energy_keV", "Energy (keV)", "number", 17.0, optional=True),
            P("wavelength_A", "or wavelength (A)", "number", optional=True),
        ],
        "Energy, wavelength, wavevector and the monochromator angles of Si, Ge and C crystals.",
    ),
    _o(
        "reflections",
        "Reflections of a Lattice",
        "Calculators",
        calculators.reflections,
        [],
        [
            P("a", "a (A)", "number", 5.43102),
            P("b", "b (A)", "number", optional=True),
            P("c", "c (A)", "number", optional=True),
            P("alpha", "alpha", "number", 90.0),
            P("beta", "beta", "number", 90.0),
            P("gamma", "gamma", "number", 90.0),
            P("centering", "Centering", "choice", "F", calculators.CENTERINGS),
            P("energy_keV", "Energy (keV)", "number", 17.0),
            P("max_index", "Largest index", "int", 4, minimum=1),
            P("max_two_theta", "Up to 2theta (deg)", "number", 90.0),
        ],
        "d-spacings, 2theta and q of the allowed reflections, one row per family.",
    ),
    _o(
        "bragg",
        "Bragg Angles",
        "Calculators",
        calculators.bragg,
        [],
        [
            P("d_spacings", "d-spacings (A)", "numbers", [3.1356]),
            P("energy_keV", "Energy (keV)", "number", 17.0),
        ],
        "theta and 2theta of d-spacings at an energy.",
    ),
    _o(
        "scherrer_size",
        "Crystallite Size",
        "Calculators",
        calculators.scherrer_size,
        [],
        [
            P("fwhm_deg", "Peak FWHM (deg 2theta)", "numbers", [0.1]),
            P("two_theta_deg", "Peak 2theta (deg)", "numbers", [30.0]),
            P("energy_keV", "Energy (keV)", "number", 17.0),
            P("shape_factor", "Shape factor K", "number", 0.9),
            P("instrumental_deg", "Instrumental width (deg)", "number", 0.0),
        ],
        "The Scherrer crystallite size of peaks.",
    ),
    _o(
        "transmission",
        "Transmission",
        "Calculators",
        calculators.transmission,
        [],
        [
            P(
                "material",
                "Material",
                "text",
                "Si",
                help="An element, a formula, or kapton, diamond...",
            ),
            P("thickness", "Thickness", "text", "500um"),
            P("energies_keV", "Energies (keV)", "numbers", [8.0, 12.0, 17.0, 25.0]),
        ],
        "Transmission and attenuation length (needs xraydb).",
    ),
    _o(
        "photons",
        "Photons per Pulse",
        "Calculators",
        calculators.photons,
        [],
        [
            P("pulse_energy_uJ", "Pulse energy (uJ)", "number", 100.0),
            P("energy_keV", "Energy (keV)", "number", 9.5),
        ],
        "Photons in a pulse of an XFEL.",
    ),
]

_BY_KEY = {op.key: op for op in OPS}


def find(key: str) -> Op:
    """The operation named ``key`` (``KeyError`` naming the close ones)."""
    try:
        return _BY_KEY[key]
    except KeyError:
        close = [k for k in _BY_KEY if key.lower() in k or k in key.lower()]
        raise KeyError(f"no operation {key!r}; close: {close or sorted(_BY_KEY)[:10]}") from None


def in_group(group: str) -> list[Op]:
    return [op for op in OPS if op.group == group]


def for_kind(kind: str) -> list[Op]:
    """The operations whose first input takes ``kind`` (a scan set also gets the scan ones)."""
    out = []
    for op in OPS:
        if not op.inputs:
            continue
        kinds = op.inputs[0].kinds
        if kind in kinds or (kind == "scanset" and "scan" in kinds and not op.inputs[0].many):
            out.append(op)
    return out
