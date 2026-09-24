"""Profiling helpers: ``with bexa.profile(): ...`` and the CLI ``--profile`` flag."""

from __future__ import annotations

import cProfile
import io
import pstats
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from bexa._log import get_logger

log = get_logger(__name__)


@contextmanager
def profile(
    output: str | Path | None = None, top: int = 25, sort: str = "cumulative"
) -> Iterator[cProfile.Profile]:
    """Profile the enclosed block with cProfile.

    Parameters
    ----------
    output
        If given, the raw stats are written there (``.prof``, readable with
        ``snakeviz`` or ``pstats``) and a text summary next to it (``.txt``).
        Otherwise the summary is logged.
    top
        Number of functions in the summary.
    sort
        pstats sort key (``cumulative``, ``tottime``, ...).

    Examples
    --------
    >>> with bexa.profile("com_run.prof"):
    ...     bexa.reduce(scan, [bexa.acc.MotorCOM(axes=("chi", "mu"))])
    """
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        yield profiler
    finally:
        profiler.disable()
        summary = io.StringIO()
        pstats.Stats(profiler, stream=summary).sort_stats(sort).print_stats(top)
        if output is None:
            log.info("profile summary:\n%s", summary.getvalue())
        else:
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            profiler.dump_stats(str(output))
            output.with_suffix(".txt").write_text(summary.getvalue(), encoding="utf-8")
            log.info("profile written to %s (summary in %s)", output, output.with_suffix(".txt"))
