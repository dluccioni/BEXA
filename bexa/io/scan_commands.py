"""Scan commands as the control system records them: which motors moved, over what.

A BLISS or SPEC scan writes its command line as the title of the scan:
``ascan chi 6.7 6.9 10 0.1``, ``amesh mu 9.56 10.56 20 phi -1.08 -0.48 12 0.4``,
``fscan2d chi 4.14 0.023 31 mu 11.92 0.04 200 2 2.08``. Two grammars cover the
scans a beamline writes:

- the step scans of SPEC and BLISS, ``<command> (<motor> <start> <stop> <intervals>)...
  <count time>``, whose point count is the number of intervals plus one;
- the continuous scans of ID03, ``<command> (<motor> <start> <step> <npoints>)...``,
  whose count is the number of points.

A command that is not in either family (a beamline macro such as ``dfmesh``) is
read with the same block structure, so its motors are known even when its
point count is not trusted; the readbacks then settle the grid.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

__all__ = [
    "CONTINUOUS_COMMANDS",
    "STATIC_COMMANDS",
    "STEP_COMMANDS",
    "ScanCommand",
    "ScanMotor",
    "parse_scan_command",
]

# SPEC and BLISS step scans: the third number of a motor block counts intervals
STEP_COMMANDS = frozenset(
    {
        "ascan",
        "dscan",
        "a2scan",
        "d2scan",
        "a3scan",
        "d3scan",
        "a4scan",
        "d4scan",
        "amesh",
        "dmesh",
        "pointscan",
        "anscan",
        "dnscan",
    }
)
# the continuous scans of ID03: the third number of a motor block counts points
CONTINUOUS_COMMANDS = frozenset(
    {"fscan", "fscan2d", "fscan3d", "finterlaced", "fscan2d_cyclic", "fsweep"}
)
# commands that move no motor: a sequence of frames
STATIC_COMMANDS = frozenset({"loopscan", "timescan", "ct", "sct"})


@dataclass
class ScanMotor:
    """One motor of a scan command: its name, range and number of points."""

    name: str
    start: float
    stop: float
    npoints: int


@dataclass
class ScanCommand:
    """A parsed scan command.

    ``exact`` says whether the point counts are to be trusted: they are for the
    step and continuous families, not for an unknown command, whose counts are
    read as intervals but may be points.
    """

    command: str
    motors: list[ScanMotor]
    text: str
    exact: bool = True

    @property
    def shape(self) -> tuple[int, ...]:
        """Points per motor, in the order of the command."""
        return tuple(m.npoints for m in self.motors)

    @property
    def names(self) -> list[str]:
        return [m.name for m in self.motors]

    @property
    def n_points(self) -> int | None:
        """Frames the command plans: the product of the counts, or ``None`` when unknown."""
        if not self.motors or not self.exact:
            return None
        total = 1
        for m in self.motors:
            total *= m.npoints
        return total


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def parse_scan_command(
    text: str, motor_names: Iterable[str] | None = None, intervals: bool | None = None
) -> ScanCommand:
    """Parse a scan command line into its command word and motor blocks.

    Parameters
    ----------
    text
        The command as recorded, starting with the command word.
    motor_names
        When given, only blocks whose name is one of these motors count (the
        positioners the file knows); other words end the motor blocks.
    intervals
        Whether the third number of a block counts intervals (the step-scan
        convention: points are intervals plus one) or points. By default the
        command word decides, and an unknown command is read as a step scan
        with ``exact=False``.

    A command without motor blocks (``loopscan``, ``ct``, or ``lookupscan`` with
    its positions elsewhere) gives an empty motor list.
    """
    tokens = str(text).split()
    if not tokens:
        raise ValueError("empty scan command")
    command = tokens[0]
    known = set(motor_names) if motor_names is not None else None
    blocks: list[tuple[str, list[float]]] = []
    i = 1
    while i < len(tokens):
        token = tokens[i]
        if _is_number(token) or (known is not None and token not in known):
            break
        i += 1
        numbers: list[float] = []
        while i < len(tokens) and _is_number(tokens[i]):
            numbers.append(float(tokens[i]))
            i += 1
        if len(numbers) < 2:  # a word followed by one number or none: not a motor block
            break
        blocks.append((token, numbers))
    if intervals is None:
        intervals = command not in CONTINUOUS_COMMANDS
    exact = command in STEP_COMMANDS or command in CONTINUOUS_COMMANDS
    # a2scan-style commands share one count after the last block: fill it into the blocks
    # that only carry a range
    shared = next((n[2] for _, n in blocks if len(n) >= 3), None)
    motors: list[ScanMotor] = []
    for name, numbers in blocks:
        count = numbers[2] if len(numbers) >= 3 else shared
        if count is None:
            exact = False
            count = 0.0
        points = round(count) + (1 if intervals else 0)
        motors.append(ScanMotor(name, numbers[0], numbers[1], max(points, 0)))
    if command in STATIC_COMMANDS:
        motors = []
    return ScanCommand(command, motors, str(text), exact=exact and bool(motors))
