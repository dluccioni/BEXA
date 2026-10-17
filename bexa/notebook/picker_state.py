"""What a scan picker remembers across kernel restarts, and when it may use it.

The state file holds the folders listed, every selected scan by folder and
number, the filters of the bars, the pressed motors, the stack axis and the
view, with a hash of the code defaults it was saved under: a state saved
under other ``FOLDERS``, ``SCANS`` or ``FILTERS`` is ignored (edited code wins).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from bexa._log import get_logger

log = get_logger(__name__)

__all__ = ["PickerState", "defaults_key", "stable"]


def stable(value: Any) -> Any:
    """A JSON-able stand-in for a code default; a lambda counts by its code, not its address."""
    if callable(value):
        code = getattr(value, "__code__", None)
        if code is not None:
            return ["callable", code.co_code.hex(), repr(code.co_consts), repr(code.co_names)]
        return ["callable", repr(value)]
    if isinstance(value, Mapping):
        return {str(k): stable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [stable(v) for v in value]
    return repr(value)


def defaults_key(
    root: str | Path,
    folders: Sequence[str | Path],
    scans: Any,
    filters: Mapping[str, Any],
    status: Any,
) -> str:
    """A short hash of the code defaults (root, folders, scans, filters, status) of a picker."""
    parts = {
        "root": str(root),
        "folders": [str(f) for f in folders],
        "scans": stable(scans),
        "filters": stable(filters),
        "status": repr(status),
    }
    return hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class PickerState:
    """A picker's choices as its state file holds them.

    ``selected`` has one ``[folder, scan number]`` per selected scan; a state
    written before the selection could span folders names the folder alone.
    """

    defaults: str
    root: str
    folders: list[str]
    selected: list[list[Any]]
    ui: dict[str, str] = field(default_factory=dict)
    motors: list[str] | None = None
    stack_axis: str | None = None
    view: str = "list"

    def write(self, path: str | Path) -> Path:
        """Write the state as JSON (the folder is made when missing); returns the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=1), encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: str | Path, defaults: str) -> PickerState | None:
        """The state in ``path``; ``None`` if missing, unreadable or saved under other defaults."""
        path = Path(path)
        if not path.is_file():
            return None
        names = {f.name for f in fields(cls)}
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            state = cls(**{k: v for k, v in saved.items() if k in names})
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            log.warning("picker state %s ignored: %s", path, exc)
            return None
        if state.defaults != defaults:
            log.info("picker state %s ignored: the code defaults changed", path)
            return None
        return state
