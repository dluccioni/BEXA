"""Everything about a scan that its files say, as tables, before a frame is read.

:func:`scan_metadata` gathers what bexa makes of a scan, whatever its format:
the scan itself (format, type, status, size, energy, detector), its dims with
their ranges and units, the channels recorded at every point (the motors,
counters and detectors of the point table) with their ranges, the scalars (where
the motors that did not move stood, the run's settings), the detectors the files
hold, the tables beside the frames and the files; then what the engine finds in
the files themselves (:meth:`bexa.io.base.BaseSource.file_metadata`: a netCDF
file's variables, a SPEC scan's header and area detectors). The Browse tab shows
it for a scan file and for the scan selected; :meth:`Scan.metadata` is the same.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from bexa._log import get_logger

if TYPE_CHECKING:
    from bexa.core.scan import Scan

log = get_logger(__name__)

__all__ = ["MAX_FILES", "scan_metadata"]

MAX_FILES = 200  # files listed one by one; the rest are counted
INDEX_COLUMNS = ("point", "frame")


def _facts(rows: list[tuple[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [(k, "" if v is None else str(v)) for k, v in rows], columns=["field", "value"]
    )


def _number(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{value:.6g}" if np.isfinite(value) else str(value)


def _span(values: np.ndarray) -> dict[str, str]:
    """first, last, min, max, step (the median one) and NaN count of a numeric column."""
    values = np.asarray(values)
    if values.dtype.kind not in "biuf" or values.size == 0:
        first = values.reshape(-1)[0] if values.size else ""
        return {"first": str(first)[:40], "last": "", "min": "", "max": "", "step": "", "NaN": ""}
    numbers = values.astype(float)
    finite = numbers[np.isfinite(numbers)]
    steps = np.diff(finite)
    return {
        "first": _number(numbers[0]),
        "last": _number(numbers[-1]),
        "min": _number(finite.min()) if finite.size else "",
        "max": _number(finite.max()) if finite.size else "",
        "step": _number(np.median(steps)) if steps.size else "",
        "NaN": str(int(numbers.size - finite.size)),
    }


def _summary(scan: Scan) -> pd.DataFrame:
    s = scan.structure
    source = scan.source
    spec = scan.spec
    dims = " x ".join(f"{d}[{n}]" for d, n in zip(s.dims, s.shape, strict=True))
    frames = f"{s.n_frames}"
    if s.planned_frames is not None and s.planned_frames != s.n_frames:
        frames += f" of {s.planned_frames} planned"
    if s.n_missing:
        frames += f", {s.n_missing} grid points without one"
    return _facts(
        [
            ("name", scan.name),
            ("format", spec.name if spec is not None else ""),
            ("engine", type(source).__name__),
            ("scan type", s.scan_type),
            ("title", s.title),
            ("status", scan.status),
            ("dims", dims),
            ("scanned motors", ", ".join(s.motor_dims)),
            ("frames", frames),
            ("frame", f"{s.frame_shape[0]} x {s.frame_shape[1]} {source.dtype}"),
            ("energy", f"{s.energy_keV:.6g} keV" if s.energy_keV is not None else "unknown"),
            ("detector read", scan.detector or ""),
            ("files", len(scan.files)),
        ]
    )


def _dims(scan: Scan) -> pd.DataFrame:
    s = scan.structure
    rows = []
    for d, n in zip(s.dims, s.shape, strict=True):
        motor = d in s.motor_dims
        values = np.asarray(s.coords[d]) if motor and d in s.coords else np.arange(n)
        rows.append(
            {
                "dim": d,
                "size": n,
                "kind": "motor" if motor else "detector pixels",
                **_span(values),
                "unit": s.units.get(d, "" if motor else "pixel"),
            }
        )
    return pd.DataFrame(rows)


def _point_table(scan: Scan) -> pd.DataFrame | None:
    """The source's table of points, else its per-frame channels."""
    source = scan.source
    table = getattr(source, "point_table", None)
    if callable(table):
        try:
            return table()
        except Exception as exc:  # a table that cannot be made is left out, not fatal
            log.debug("no point table for %s: %s", scan.name, exc)
    per_frame = scan.structure.per_frame
    return pd.DataFrame({k: np.asarray(v) for k, v in per_frame.items()}) if per_frame else None


def _channels(scan: Scan, table: pd.DataFrame | None) -> pd.DataFrame:
    s = scan.structure
    if table is None or table.empty:
        return pd.DataFrame(columns=["channel", "kind", "dtype", "points"])
    rows = []
    for name in table.columns:
        values = table[name].to_numpy()
        if name in INDEX_COLUMNS:
            kind = "index"
        elif name in s.motor_dims:
            kind = "scanned motor"
        elif name in s.motor_names:
            kind = "motor"
        else:
            kind = "counter"
        rows.append(
            {
                "channel": str(name),
                "kind": kind,
                "dtype": str(values.dtype),
                "points": int(values.size),
                **_span(values),
                "mean": _number(np.nanmean(values))
                if values.dtype.kind in "biuf" and values.size
                else "",
            }
        )
    return pd.DataFrame(rows)


def _scalars(scan: Scan) -> pd.DataFrame:
    try:
        scalars = scan.source.scalars()
    except Exception as exc:
        log.debug("no scalars for %s: %s", scan.name, exc)
        scalars = {}
    return pd.DataFrame(
        [(str(k), _number(v)) for k, v in scalars.items()], columns=["name", "value"]
    )


def _detectors(scan: Scan) -> pd.DataFrame:
    read = scan.detector
    names = scan.detectors or ([read] if read else [])
    s = scan.structure
    rows = [
        {
            "detector": name,
            "read": "yes" if name == read else "",
            "frame": f"{s.frame_shape[0]} x {s.frame_shape[1]} {scan.source.dtype}"
            if name == read
            else "",
            "frames": s.n_frames if name == read else "",
        }
        for name in names
    ]
    return pd.DataFrame(rows, columns=["detector", "read", "frame", "frames"])


def _tables(scan: Scan) -> pd.DataFrame:
    try:
        tables = scan.source.aux_tables()
    except Exception as exc:
        log.debug("no tables for %s: %s", scan.name, exc)
        tables = {}
    rows = []
    for name, table in tables.items():
        columns = [str(c) for c in table.columns]
        listed = ", ".join(columns[:40]) + (
            f", ... {len(columns) - 40} more" if len(columns) > 40 else ""
        )
        rows.append({"table": name, "rows": len(table), "columns": len(columns), "names": listed})
    return pd.DataFrame(rows, columns=["table", "rows", "columns", "names"])


def _files(scan: Scan) -> pd.DataFrame:
    files = list(scan.files)
    rows = []
    for k, path in enumerate(files[:MAX_FILES]):
        try:
            size = f"{path.stat().st_size / 1e6:.3f} MB"
        except OSError:
            size = "missing"
        rows.append({"file": k, "name": path.name, "folder": str(path.parent), "size": size})
    if len(files) > MAX_FILES:
        last = files[-1]
        rows.append(
            {
                "file": f"... {len(files) - MAX_FILES - 1} more, then {len(files) - 1}",
                "name": last.name,
                "folder": str(last.parent),
                "size": "",
            }
        )
    return pd.DataFrame(rows, columns=["file", "name", "folder", "size"])


def scan_metadata(scan: Scan) -> dict[str, pd.DataFrame]:
    """A scan's metadata as tables by title, no frame read.

    ``scan`` (what it is), ``dims``, ``per point`` (every channel of the point table:
    kind, range, NaN), ``scalars``, ``detectors``, ``tables`` (beside the frames),
    ``files`` (the first :data:`MAX_FILES` with their sizes), then the engine's own
    tables of what the files hold (:meth:`~bexa.io.base.BaseSource.file_metadata`).
    """
    out = {
        "scan": _summary(scan),
        "dims": _dims(scan),
        "per point": _channels(scan, _point_table(scan)),
        "scalars": _scalars(scan),
        "detectors": _detectors(scan),
        "tables": _tables(scan),
        "files": _files(scan),
    }
    hook = getattr(scan.source, "file_metadata", None)
    if callable(hook):
        try:
            out.update(hook())
        except Exception as exc:  # the files may be gone or half written: say so, show the rest
            out["file metadata"] = _facts([("error", f"{type(exc).__name__}: {exc}")])
    return out
