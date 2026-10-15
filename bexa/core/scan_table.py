"""The rows of a dataset's scan table, and what a compact table leaves out.

:class:`bexa.core.dataset.Dataset` keeps one row per scan: its status, type,
motors, grid, frames, energy and every fixed positioner. A scan that cannot be
read yet (still running without a frame, empty, unreadable) gets a placeholder
row carrying its status, so one such scan never hides the others.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from bexa.core.scan import Scan
from bexa.core.status import ScanStatus

__all__ = [
    "FIXED_COLUMNS",
    "LEAD_COLUMNS",
    "arrange",
    "compact_columns",
    "followers",
    "min_steps",
    "placeholder_row",
    "scan_row",
]

LEAD_COLUMNS = ("id", "dataset")
FIXED_COLUMNS = (
    "scan",
    "status",
    "type",
    "title",
    "motors",
    "shape",
    "ranges",
    "frames",
    "missing",
)


def scan_row(scan: Scan, scan_id: int, number: int, dataset: str | None = None) -> dict[str, Any]:
    """The table row of an opened scan: its structure, status, energy and positioners."""
    s = scan.structure
    row: dict[str, Any] = {"id": scan_id}
    if dataset is not None:
        row["dataset"] = dataset
    row.update(
        {
            "scan": number,
            "status": str(scan.status),
            "type": s.scan_type,
            "title": s.title,
            "motors": " x ".join(s.motor_dims),
            "shape": " x ".join(str(n) for n in s.motor_shape),
            "ranges": "; ".join(
                f"{d} {s.coords[d].min():.5g}..{s.coords[d].max():.5g}" for d in s.motor_dims
            ),
            "frames": s.n_frames,
            "missing": s.n_missing,
            "energy_keV": s.energy_keV,
        }
    )
    row.update({k: v for k, v in s.scalars.items() if k not in row})
    return row


def placeholder_row(
    scan_id: int, number: int, status: ScanStatus, dataset: str | None = None
) -> dict[str, Any]:
    """The row of a scan that cannot be read yet: its status, and nothing else."""
    row: dict[str, Any] = {"id": scan_id}
    if dataset is not None:
        row["dataset"] = dataset
    row.update(
        {
            "scan": number,
            "status": str(status),
            "type": "",
            "title": "",
            "motors": "",
            "shape": "",
            "ranges": "",
            "frames": 0,
            "missing": 0,
            "energy_keV": np.nan,
        }
    )
    return row


def arrange(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """The rows as a table: id, dataset, the fixed columns, the energy, then the positioners."""
    table = pd.DataFrame(rows)
    if not table.empty:
        lead = [c for c in LEAD_COLUMNS if c in table.columns]
        fixed = [*lead, *FIXED_COLUMNS, "energy_keV"]
        positioners = sorted(c for c in table.columns if c not in fixed)
        table = table[[*fixed, *positioners]]
    return table


def min_steps(spec: Any) -> dict[str, float]:
    """``{motor: step}`` from the spec's ``motors.min_step``: smaller changes are drift."""
    if spec is None:
        return {}
    steps = spec.motors.get("min_step", {}) or {}
    return {str(name): float(step) for name, step in steps.items()}


def followers(spec: Any) -> dict[str, list[str]]:
    """``{leader: [followers]}`` from the spec's ``motors.coupled`` (``uz`` drives ``samz``)."""
    coupled = (spec.motors.get("coupled", {}) or {}) if spec is not None else {}
    return {
        str(leader): [names] if isinstance(names, str) else [str(n) for n in names]
        for leader, names in coupled.items()
    }


def compact_columns(table: pd.DataFrame, spec: Any) -> list[str]:
    """The columns a compact table drops.

    Positioners that never change, a motor kept under a legacy name too (once
    is enough), the readbacks a stepping leader drives, a drift below the
    motor's ``min_step``, ``missing`` when no scan has missing frames and
    ``status`` when every scan is complete.
    """
    positioners = [
        c for c in table.columns if c not in (*LEAD_COLUMNS, *FIXED_COLUMNS, "energy_keV")
    ]
    dropped = [c for c in positioners if table[c].nunique(dropna=True) <= 1]
    aliases = (spec.motors.get("aliases", {}) or {}) if spec is not None else {}
    for logical, physical in aliases.items():
        names = [physical] if isinstance(physical, str) else list(physical)
        if logical in positioners and any(
            p in positioners and table[logical].equals(table[p]) for p in names
        ):
            dropped.append(logical)
    for leader, names in followers(spec).items():  # samz follows uz: once too
        if leader in positioners and leader not in dropped:
            dropped.extend(f for f in names if f in positioners)
    for name, step in min_steps(spec).items():  # a drift below the motor's step
        if name in positioners and name not in dropped:
            column = pd.to_numeric(table[name], errors="coerce")
            if column.notna().any() and float(column.max() - column.min()) < step:
                dropped.append(name)
    if "missing" in table.columns and not table["missing"].any():
        dropped.append("missing")
    if "status" in table.columns and (table["status"] == ScanStatus.COMPLETE.value).all():
        dropped.append("status")
    return sorted(set(dropped))
