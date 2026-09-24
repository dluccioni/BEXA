"""Where things are: the repo root, the config folders, the cache root, path aliases."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

CACHE_DIR_ENV = "BEXA_CACHE_DIR"
CACHE_DIR_NAME = "bexa_cache"


def repo_root() -> Path:
    """Root of the checkout (the folder that contains ``bexa/`` and ``configs/``)."""
    return Path(__file__).resolve().parents[2]


def package_configs() -> Path:
    """The ``configs/`` folder shipped with the code."""
    return repo_root() / "configs"


def user_cache_dir() -> Path:
    """Per-user cache folder used when no beamtime cache is available."""
    return Path.home() / ".cache" / "bexa"


def is_writable(path: Path) -> bool:
    """True when ``path`` exists (or its parent does) and can be written."""
    probe = path if path.exists() else path.parent
    return probe.exists() and os.access(probe, os.W_OK)


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if needed and return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_root(processed_root: str | Path | None = None) -> Path:
    """Cache folder: ``BEXA_CACHE_DIR``, else ``<processed_root>/bexa_cache``, else the user's."""
    env = os.environ.get(CACHE_DIR_ENV)
    if env:
        return Path(env)
    if processed_root is not None:
        candidate = Path(processed_root) / CACHE_DIR_NAME
        if is_writable(candidate):
            return candidate
    return user_cache_dir()


def apply_aliases(path: str | Path, aliases: Mapping[str, str] | None) -> Path:
    """Map a path recorded at the beamline onto this machine.

    ``aliases`` maps a prefix as written in the profile (``/data/visitor/hc6293``)
    to the prefix where the same data lives here (``X:/Beamtimes/hc6293``). The
    original path wins when it exists; otherwise the first alias whose target
    exists is used; otherwise the original path is returned unchanged.
    """
    original = Path(path)
    if original.exists() or not aliases:
        return original
    text = str(path).replace("\\", "/")
    for source, target in aliases.items():
        src = source.replace("\\", "/").rstrip("/")
        if text == src or text.startswith(src + "/"):
            candidate = Path(target.rstrip("/\\") + text[len(src) :])
            if candidate.exists():
                return candidate
    return original
