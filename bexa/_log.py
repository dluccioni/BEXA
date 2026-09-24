"""Logging and progress reporting for bexa.

All modules log through ``get_logger(__name__)``; nothing in the library calls
``print``. The level comes from ``configure(level)`` or the ``BEXA_LOG_LEVEL``
environment variable and defaults to INFO.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterable, Iterator
from typing import TypeVar

ROOT_NAME = "bexa"
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"
_configured = False

T = TypeVar("T")


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
    """
    if not enabled:
        yield from iterable
        return
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - tqdm is a core dependency
        yield from iterable
        return
    yield from tqdm(iterable, total=total, desc=desc, leave=False)
