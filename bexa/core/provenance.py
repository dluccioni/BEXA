"""Provenance: attrs that record where a result came from and how fast it was made.

Every array produced by a reader, reduction or pipeline carries these attrs, and
:func:`bexa.save` writes them into the file, so a result can always be traced
back to its source files and parameters.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from bexa._log import get_logger

log = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def git_hash(root: Path | None = None) -> str | None:
    """Short git hash of the checkout, or ``None`` when git or the repo is absent."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(root or _REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def json_default(obj: Any) -> Any:
    """Make numpy scalars, arrays, paths, sets and datetimes JSON-serialisable."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, (_dt.datetime, _dt.date)):
        return obj.isoformat()
    if isinstance(obj, slice):
        return [obj.start, obj.stop, obj.step]
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return str(obj)


def to_json(obj: Any) -> str:
    """Compact, sorted JSON with the :func:`json_default` conversions."""
    return json.dumps(obj, default=json_default, sort_keys=True, separators=(",", ":"))


def file_records(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Path, size and modification time of each file (missing files get size -1)."""
    records = []
    for p in paths:
        path = Path(p)
        try:
            st = path.stat()
            records.append({"path": str(path), "size": st.st_size, "mtime": st.st_mtime})
        except OSError:
            records.append({"path": str(path), "size": -1, "mtime": 0.0})
    return records


def fingerprint(paths: Iterable[str | Path], params: Any = None) -> str:
    """SHA-1 over the file records and parameters; the key used by the cache."""
    payload = {"files": file_records(paths), "params": params}
    return hashlib.sha1(to_json(payload).encode("utf-8")).hexdigest()


def build_attrs(
    source_files: Iterable[str | Path] = (),
    parameters: Any = None,
    **extra: Any,
) -> dict[str, Any]:
    """Standard provenance attrs (all values are plain strings or numbers)."""
    from bexa import __version__

    attrs: dict[str, Any] = {
        "bexa_version": __version__,
        "git_hash": git_hash() or "",
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "source_files": to_json(file_records(source_files)),
        "parameters": to_json(parameters if parameters is not None else {}),
    }
    for key, value in extra.items():
        if value is None:
            continue
        attrs[key] = value if isinstance(value, (str, int, float, bool)) else to_json(value)
    return attrs


def parameters_of(attrs: Mapping[str, Any]) -> dict[str, Any]:
    """The ``parameters`` of provenance attrs as a dict (they are stored as JSON text)."""
    value = attrs.get("parameters")
    if isinstance(value, str):
        return dict(json.loads(value)) if value else {}
    return dict(value or {})


def attach(obj: Any, **attrs: Any) -> Any:
    """Update ``obj.attrs`` (xarray DataArray or Dataset) in place and return ``obj``."""
    obj.attrs.update({k: v for k, v in attrs.items() if v is not None})
    return obj


def peak_rss_mb() -> float:
    """Peak resident memory of this process in megabytes (Windows, Linux, macOS)."""
    proc = psutil.Process(os.getpid())
    info = proc.memory_info()
    peak = getattr(info, "peak_wset", None)  # Windows
    if peak is None and sys.platform != "win32":
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = usage * 1024 if sys.platform.startswith("linux") else usage  # KB on Linux
    if peak is None:  # pragma: no cover - other platforms
        peak = info.rss
    return float(peak) / 1e6


class RunStats:
    """Wall time, throughput and peak memory of one run, for provenance attrs.

    Examples
    --------
    >>> stats = RunStats(device="cuda").start()
    >>> ...  # do the work
    >>> attrs = stats.stop(frames=4000)
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.t0 = 0.0
        self.elapsed_s = 0.0

    def start(self) -> RunStats:
        self.t0 = time.perf_counter()
        return self

    def stop(self, frames: int | None = None) -> dict[str, Any]:
        self.elapsed_s = time.perf_counter() - self.t0
        stats: dict[str, Any] = {
            "elapsed_s": round(self.elapsed_s, 3),
            "peak_rss_mb": round(peak_rss_mb(), 1),
            "device": self.device,
        }
        if frames:
            stats["frames"] = int(frames)
            stats["frames_per_s"] = round(frames / max(self.elapsed_s, 1e-9), 1)
        return stats
