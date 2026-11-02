"""Python source for values: the logged commands of a workspace are valid Python."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["call_text", "jsonable", "source_text"]

WINDOWS_PATH = re.compile(r"^(?:[A-Za-z]:\\|\\\\)")


def source_text(value: Any) -> str:
    """Python source that evaluates to ``value`` (objects and ROIs by their names)."""
    from bexa.core.roi import ROI

    if hasattr(value, "kind") and hasattr(value, "name") and hasattr(value, "data"):
        return repr(str(value.name))  # a DataObject
    if isinstance(value, ROI):
        return f"bexa.ROI({source_text(value.to_dict()['ranges'])})"
    if isinstance(value, (bool, type(None))):
        return repr(value)
    if isinstance(value, (np.integer, int)):
        return repr(int(value))
    if isinstance(value, (np.floating, float)):
        v = float(value)
        if math.isnan(v):
            return "float('nan')"
        if math.isinf(v):
            return "float('inf')" if v > 0 else "-float('inf')"
        return repr(v)
    if isinstance(value, Path):
        return repr(value.as_posix())
    if isinstance(value, str):
        if WINDOWS_PATH.match(value):  # C:\data or \\server\share: forward slashes read better
            return repr(value.replace("\\", "/"))
        return repr(value)
    if isinstance(value, np.ndarray):
        return source_text(value.tolist())
    if isinstance(value, tuple):
        inner = ", ".join(source_text(v) for v in value)
        return f"({inner},)" if len(value) == 1 else f"({inner})"
    if isinstance(value, (list, set, frozenset)):
        return "[" + ", ".join(source_text(v) for v in value) + "]"
    if isinstance(value, dict):
        return (
            "{" + ", ".join(f"{source_text(k)}: {source_text(v)}" for k, v in value.items()) + "}"
        )
    name = getattr(value, "name", None)
    if isinstance(name, str):  # a FormatSpec or a BeamtimeProfile: by name
        path = getattr(value, "source_path", None)
        return repr(str(path) if path else name)
    return repr(str(value))


def call_text(function: str, *args: Any, **kwargs: Any) -> str:
    """``function(arg, key=value)`` with every value as source; ``None`` keywords left out."""
    parts = [source_text(a) for a in args]
    parts += [f"{k}={source_text(v)}" for k, v in kwargs.items() if v is not None]
    return f"{function}({', '.join(parts)})"


def jsonable(value: Any) -> Any:
    """``value`` made of JSON types only (numpy to Python, paths and the rest to text)."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist() if value.size <= 10000 else f"<array {value.shape}>"
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if hasattr(value, "to_dict"):
        return jsonable(value.to_dict())
    return str(value)
