"""What the master file of a BLISS dataset says about one scan.

The master holds, per scan entry ``N.1``, the readback of every channel in
``measurement`` (one value per frame), the fixed positioners, the description
of a continuous scan in ``fscan_parameters``, the command line as the title,
and when and why the scan started and ended. This module reads those and
turns them into the motors a scan was made of, the grid it was meant to cover
and the state it is in; :mod:`bexa.io.engines.hdf5_stack` reads the frames.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import h5py
import numpy as np

from bexa._log import get_logger
from bexa.core.status import ScanStatus
from bexa.core.structure import _group_values, same_partition
from bexa.io.formats import FormatSpec, safe_format
from bexa.io.scan_commands import STATIC_COMMANDS, ScanCommand

log = get_logger(__name__)

__all__ = [
    "FALLBACK_MOTORS",
    "MOTOR_KEYS",
    "NOT_MOTORS",
    "EntryState",
    "MotorChoice",
    "MotorTable",
    "decide",
    "declared_motors",
    "decode",
    "discover_motors",
    "entry_state",
    "read_motor_table",
    "read_title",
]

# the datasets of fscan_parameters that name the scanned motors, slowest loop first: BLISS
# writes ``motor`` for an fscan, ``slow_motor``/``fast_motor`` for an fscan2d and
# ``slow1_motor``/``slow2_motor``/``fast_motor`` for an fscan3d (slow1 is the outer loop);
# the ``outer``/``middle``/``inner`` names are accepted too
MOTOR_KEYS = (
    "outer_motor",
    "slow1_motor",
    "slow_motor",
    "middle_motor",
    "slow2_motor",
    "slow3_motor",
    "inner_motor",
    "fast_motor",
    "motor",
)
# motors looked for when neither the file nor the command names any (the v9 list)
FALLBACK_MOTORS = ("mu", "chi", "phi", "theta", "omega", "delta", "gamma", "ux", "uy", "uz")
# per-frame channels that are never motors
NOT_MOTORS = frozenset({"elapsed_time", "epoch", "epoch_trig", "frame", "timestamp"})
MOVE_THRESHOLD = 1e-6  # a readback that changes by less than this did not move


def decode(value: Any) -> Any:
    """Bytes and 0-d string arrays as ``str``; anything else as it is."""
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray) and value.dtype.kind in "SO" and value.ndim == 0:
        return decode(value[()])
    return value


# ------------------------------------------------------------------ the motor table
@dataclass
class MotorTable:
    """The per-frame channels and fixed positioners of one scan, as recorded.

    ``channels`` keeps every 1-D numeric dataset of ``measurement`` at its full
    length (a running scan's master lags or leads the detector files; the
    engine cuts them to the frames it has); ``motor_names`` lists every name
    the file calls a motor, so a channel among them may become a dim.
    """

    channels: dict[str, np.ndarray] = field(default_factory=dict)
    scalars: dict[str, float] = field(default_factory=dict)
    motor_names: frozenset[str] = field(default_factory=frozenset)


def read_motor_table(f: h5py.File, spec: FormatSpec, scan: int, detector: str) -> MotorTable:
    """Read the channels, the positioners and the motor names of scan ``scan``."""
    table = MotorTable()
    meas_path = spec.motor_path("per_frame", scan=scan, motor="").rstrip("/")
    if meas_path in f:
        for key, node in f[meas_path].items():
            if isinstance(node, h5py.Dataset) and node.ndim == 1 and node.dtype.kind in "fiu":
                table.channels[key] = np.asarray(node[()], dtype=float)
    pos_path = spec.motor_path("scalar", scan=scan, motor="").rstrip("/")
    names: set[str] = set()
    if pos_path in f:
        for key, node in f[pos_path].items():
            names.add(str(key))
            if isinstance(node, h5py.Dataset) and node.ndim == 0 and node.dtype.kind in "fiu":
                table.scalars[key] = float(node[()])
    # BLISS also lists every motor at the start and the end of a scan
    for suffix in ("_start", "_end"):
        if pos_path + suffix in f:
            names.update(str(key) for key in f[pos_path + suffix])
    if not names:  # an older layout: the NXpositioner groups of the instrument
        names.update(_nxpositioners(f, meas_path))
    # values recorded elsewhere in the master, as positioners of the scan: the temperature
    # of a furnace sampled in the monitoring sub-scan BLISS writes next to an fscan, say
    for name, template in (spec.motors.get("scalars_from") or {}).items():
        path = safe_format(str(template), scan=scan, detector=detector)
        if path in f and isinstance(f[path], h5py.Dataset):
            values = np.asarray(f[path][()], dtype=float).ravel()
            if values.size and np.isfinite(values).any():
                table.scalars[str(name)] = float(np.nanmean(values))
    table.motor_names = frozenset(names)
    return table


def _nxpositioners(f: h5py.File, meas_path: str) -> set[str]:
    """Names of the ``NXpositioner`` groups next to the measurement group."""
    entry = meas_path.rsplit("/", 1)[0] if "/" in meas_path else ""
    instrument = f"{entry}/instrument" if entry else "instrument"
    found: set[str] = set()
    if instrument not in f:
        return found
    for key, node in f[instrument].items():
        if not isinstance(node, h5py.Group):
            continue
        if decode(node.attrs.get("NX_class", "")) == "NXpositioner":
            found.add(str(key))
    return found


def declared_motors(
    f: h5py.File, spec: FormatSpec, scan: int
) -> tuple[str, list[str], list[int]] | None:
    """``(scan_type, motor names slow first, points per motor)`` from ``fscan_parameters``."""
    template = spec.motors.get("structure")
    if not template:
        return None
    group_path = spec.motor_path("structure", scan=scan)
    if group_path not in f:
        return None
    g = f[group_path]
    scan_type = None
    for key in ("scan_type", "scan_name"):
        if key in g:
            scan_type = str(decode(g[key][()]))
            break
    if scan_type is None:
        return None
    names: list[str] = []
    counts: list[int] = []
    for motor_key in MOTOR_KEYS:
        if motor_key in g:
            count_key = motor_key.replace("motor", "npoints")
            names.append(str(decode(g[motor_key][()])))
            counts.append(int(g[count_key][()]) if count_key in g else 0)
    return scan_type, names, counts


def read_title(f: h5py.File, entry: str) -> str:
    """The command line of the scan, or an empty string."""
    path = f"{entry}/title"
    return str(decode(f[path][()])) if path in f else ""


# ------------------------------------------------------------------------ the state
@dataclass
class EntryState:
    """When and how the scan entry says it started and ended, and what it planned."""

    present: bool = True
    started: bool = False
    ended: bool = False
    end_reason: str | None = None
    writer_status: str | None = None
    planned_frames: int | None = None


def entry_state(f: h5py.File, entry: str, spec: FormatSpec, scan: int, detector: str) -> EntryState:
    """Read the times, the end reason, the writer status and the planned frame count."""
    if entry not in f:
        return EntryState(present=False)
    g = f[entry]
    state = EntryState(
        started="start_time" in g or "writer" in g,
        ended="end_time" in g,
    )
    if "end_reason" in g:
        state.end_reason = str(decode(g["end_reason"][()]))
    if "writer/status" in g:
        state.writer_status = str(decode(g["writer/status"][()]))
    if "frames_total" in spec.layout:
        path = spec.path("frames_total", scan=scan, detector=detector)
        if path in f:
            planned = int(f[path][()])
            if planned > 0:
                state.planned_frames = planned
    return state


def decide(state: EntryState, n_frames: int, planned: int | None = None) -> ScanStatus:
    """The status of a scan from its entry state and the frames found.

    First match wins: an entry the master does not hold yet, or one that
    started and has not ended, is ``running``; no frame at all is ``empty``;
    an end reason other than success, or fewer frames than planned, is
    ``aborted``; otherwise the scan is ``complete``. Masters without the time
    fields (older files) are judged by the frames alone.
    """
    if not state.present or (state.started and not state.ended):
        return ScanStatus.RUNNING
    if n_frames <= 0:
        return ScanStatus.EMPTY
    if state.end_reason is not None and state.end_reason.strip().upper() != "SUCCESS":
        return ScanStatus.ABORTED
    expected = state.planned_frames if state.planned_frames is not None else planned
    if expected is not None and n_frames < expected:
        return ScanStatus.ABORTED
    return ScanStatus.COMPLETE


# -------------------------------------------------------------------- the motors
@dataclass
class MotorChoice:
    """Which channels are the dims of a scan, and what the scan planned.

    ``order`` lists the dims slowest first as the file or the command names
    them (the readbacks settle the loop order later); ``lockstep`` maps a dim
    to the channels that step with it, kept as coordinates rather than dims;
    ``candidates`` is every channel that is a motor and moved, the choices a
    reader has.
    """

    scan_type: str
    order: list[str]
    planned_shape: tuple[int, ...] | None
    lockstep: dict[str, list[str]]
    candidates: list[str]


def moving_motors(
    channels: Mapping[str, np.ndarray],
    motor_names: Iterable[str],
    known: list[str],
    min_steps: Mapping[str, float],
) -> list[str]:
    """The per-frame channels that are motors and moved, in the spec's known order first.

    A channel is a motor when the file lists it as a positioner, or the spec
    knows it; it moved when its range exceeds the spec's ``min_step`` for it
    (else :data:`MOVE_THRESHOLD`). Time stamps and counters never qualify.
    """
    names = set(motor_names)
    moving: list[str] = []
    for name, values in channels.items():
        if name in NOT_MOTORS or not (name in names or name in known or name in FALLBACK_MOTORS):
            continue
        finite = values[np.isfinite(values)] if values.size else values
        if finite.size == 0:
            continue
        threshold = max(float(min_steps.get(name, 0.0)), MOVE_THRESHOLD)
        if float(finite.max() - finite.min()) > threshold:
            moving.append(name)

    def rank(name: str) -> tuple[int, int, str]:
        in_known = known.index(name) if name in known else len(known)
        fallback = list(FALLBACK_MOTORS)
        in_fallback = fallback.index(name) if name in fallback else len(fallback)
        return (in_known, in_fallback, name)

    return sorted(moving, key=rank)


def collapse_lockstep(
    order: list[str], channels: Mapping[str, np.ndarray]
) -> tuple[list[str], dict[str, list[str]]]:
    """Keep one of several channels that step together (``a2scan``, a motor and its encoder)."""
    kept: list[str] = []
    lockstep: dict[str, list[str]] = {}
    labels = {
        name: _group_values(np.asarray(channels[name], dtype=float), None)[1] for name in order
    }
    for name in order:
        twin = next((k for k in kept if same_partition(labels[k], labels[name])), None)
        if twin is None:
            kept.append(name)
        else:
            lockstep.setdefault(twin, []).append(name)
    return kept, lockstep


def discover_motors(
    channels: Mapping[str, np.ndarray],
    declared: tuple[str, list[str], list[int]] | None,
    command: ScanCommand | None,
    motor_names: Iterable[str],
    known: list[str],
    min_steps: Mapping[str, float],
    logical_name: Mapping[str, str] | None = None,
) -> MotorChoice:
    """Which channels are the scanned motors, in this order of trust.

    1. The description of a continuous scan (``fscan_parameters``), when the
       file has one and its motors are recorded.
    2. The command line in the title: ``amesh mu ... phi ...`` names its motors.
    3. Every recorded channel that is a motor and moved (:func:`moving_motors`);
       a command that moves no motor (``loopscan``, ``ct``) gives none.

    Channels that step together collapse to the first named. The scan type is
    the declared one, else the command word, else ``fscan<n>d``.
    """
    rename = dict(logical_name or {})
    planned: tuple[int, ...] | None = None
    order: list[str] = []
    scan_type = "list"
    if declared:
        scan_type, names, counts = declared
        names = [rename.get(n, n) for n in names]
        order = [n for n in names if n in channels]
        if len(order) != len(names):
            log.warning(
                "the scan declares motors %s but the master only records %s", names, list(channels)
            )
        elif order and all(c > 0 for c in counts):
            planned = tuple(int(c) for c in counts)
    if not order and command is not None and command.motors:
        names = [rename.get(n, n) for n in command.names]
        order = [n for n in names if n in channels]
        scan_type = command.command
        if len(order) != len(names):
            log.warning(
                "the title names motors %s but the master only records %s", names, list(channels)
            )
        elif order and command.exact:
            planned = command.shape
    candidates = moving_motors(channels, motor_names, known, min_steps)
    if not order:
        static = command is not None and command.command in STATIC_COMMANDS
        order = [] if static else list(candidates)
        if command is not None and command.command:
            scan_type = command.command
        else:
            scan_type = f"fscan{len(order)}d" if order else "list"
    for name in order:
        if name not in candidates:
            candidates.append(name)
    order, lockstep = collapse_lockstep(order, channels)
    return MotorChoice(scan_type, order, planned, lockstep, candidates)
