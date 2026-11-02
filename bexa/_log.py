"""Logging and progress reporting for bexa.

All modules log through ``get_logger(__name__)``; nothing in the library calls
``print``. The level comes from ``configure(level)`` or the ``BEXA_LOG_LEVEL``
environment variable and defaults to INFO.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from typing import TypeVar

ROOT_NAME = "bexa"
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"
_configured = False
_hooks = threading.local()  # the progress callback of each thread, see progress_hook

T = TypeVar("T")
ProgressCallback = Callable[[int, "int | None", str], None]


class CancelledError(Exception):
    """Raised by a progress callback to stop the work it watches (the GUI's Cancel button)."""


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the ``bexa`` logger or one of its children (``bexa.io.formats``)."""
    configure()
    if name is None or name == ROOT_NAME:
        return logging.getLogger(ROOT_NAME)
    if not name.startswith(ROOT_NAME + "."):
        name = f"{ROOT_NAME}.{name}"
    return logging.getLogger(name)


def configure(level: str | int | None = None, stream=None, force: bool = False) -> None:
    """Attach one stream handler to the root ``bexa`` logger.

    Parameters
    ----------
    level
        Logging level name or number. ``None`` reads ``BEXA_LOG_LEVEL`` and
        falls back to INFO.
    stream
        Where to write; defaults to ``sys.stderr``.
    force
        Re-create the handler even if one exists (used after a stream change).
    """
    global _configured
    logger = logging.getLogger(ROOT_NAME)
    if level is None:
        level = os.environ.get("BEXA_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
    if _configured and not force:
        logger.setLevel(level)
        return
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    _configured = True


def set_level(level: str | int) -> None:
    """Change the level of the ``bexa`` logger and its children."""
    configure(level)


def progress(
    iterable: Iterable[T],
    total: int | None = None,
    desc: str = "",
    enabled: bool = True,
) -> Iterator[T]:
    """Wrap an iterable in a tqdm progress bar when tqdm is installed and enabled.

    Falls back to the plain iterable, so callers never need to check for tqdm.
    Inside :func:`progress_hook` the thread's callback hears of every item
    instead of a bar, whatever ``enabled`` says.
    """
    hook = getattr(_hooks, "callback", None)
    if hook is not None:
        hook(0, total, desc)
        for done, item in enumerate(iterable, start=1):
            yield item
            hook(done, total, desc)
        return
    if not enabled:
        yield from iterable
        return
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - tqdm is a core dependency
        yield from iterable
        return
    yield from tqdm(iterable, total=total, desc=desc, leave=False)


@contextmanager
def progress_hook(callback: ProgressCallback | None) -> Iterator[None]:
    """Report every :func:`progress` loop of this thread to ``callback(done, total, desc)``.

    The callback is called before the first item and after each one; raising
    :class:`CancelledError` (or anything else) from it stops the loop there. Other
    threads are not affected, so a background job can watch its own work.
    """
    previous = getattr(_hooks, "callback", None)
    _hooks.callback = callback
    try:
        yield
    finally:
        _hooks.callback = previous


def in_ipython() -> bool:
    """True inside IPython, Jupyter or a VS Code interactive window."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    return get_ipython() is not None


def in_kernel() -> bool:
    """True inside a Jupyter kernel (JupyterLab, VS Code interactive window), where widgets show."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    ip = get_ipython()
    return ip is not None and "IPKernelApp" in getattr(ip, "config", {})


def interactive_session() -> bool:
    """True when someone is watching: an IPython kernel, or a terminal on stderr."""
    return in_ipython() or bool(getattr(sys.stderr, "isatty", lambda: False)())
