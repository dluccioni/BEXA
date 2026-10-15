"""What state a scan is in: complete, still running, aborted, empty or unreadable.

During a beamtime a dataset folder holds scans in every state at once: the
control system appends a scan to the master file when it starts and writes
its end only when it finishes, while the detector files fill up in between.
Every engine reports one :class:`ScanStatus` per scan so that a table of the
dataset can list them all, a selection can keep the finished ones, and a scan
that is still being written can be read as far as it goes.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from typing import Any

__all__ = ["FINISHED", "ScanNotReadyError", "ScanStatus", "status_filter", "worst"]


class ScanStatus(str, Enum):
    """The state of one scan, as the engine reads it from the files."""

    COMPLETE = "complete"  # every planned frame is there
    RUNNING = "running"  # started and not ended: the files are still growing
    ABORTED = "aborted"  # ended early: fewer frames than planned, readable as far as it goes
    EMPTY = "empty"  # ended, or never started, with no frame at all
    UNREADABLE = "unreadable"  # the files could not be read (locked, truncated, missing)

    def __str__(self) -> str:
        return self.value


# a finished scan's files do not change any more: its table row and results can be kept
FINISHED = frozenset({ScanStatus.COMPLETE, ScanStatus.ABORTED})
# the worse a status, the less of the scan can be read
_SEVERITY = {
    ScanStatus.COMPLETE: 0,
    ScanStatus.ABORTED: 1,
    ScanStatus.RUNNING: 2,
    ScanStatus.EMPTY: 3,
    ScanStatus.UNREADABLE: 4,
}


def status_filter(value: Any) -> frozenset[ScanStatus] | None:
    """The statuses a selection keeps, or ``None`` for every status.

    ``"finished"`` keeps the complete and the aborted scans (their files do not
    change any more), ``"complete"`` only the complete ones, ``"any"`` or
    ``None`` all of them; a status name or a tuple of names keeps exactly those.
    """
    if value is None or value == "any":
        return None
    if value == "finished":
        return FINISHED
    names: Iterable[object] = [value] if isinstance(value, (str, ScanStatus)) else list(value)
    chosen: set[ScanStatus] = set()
    for name in names:
        try:
            chosen.add(ScanStatus(str(name)))
        except ValueError:
            options = ", ".join(s.value for s in ScanStatus)
            raise ValueError(
                f"unknown scan status {name!r}; use finished, any, or one of {options}"
            ) from None
    return frozenset(chosen)


def worst(statuses: Iterable[ScanStatus]) -> ScanStatus:
    """The status of a series of scans: the one that lets the least be read."""
    found = list(statuses)
    if not found:
        return ScanStatus.EMPTY
    return max(found, key=lambda s: _SEVERITY[s])


class ScanNotReadyError(ValueError):
    """A scan with nothing to read yet: running without a frame, empty, or unreadable."""

    def __init__(self, message: str, status: ScanStatus, scan: int | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.scan = scan
