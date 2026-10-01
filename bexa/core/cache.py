"""Disk and memory cache for previews and reduction results.

Entries are keyed by a fingerprint of the source files (path, size, mtime) and
the parameters of the operation, so a result is reused only while the data and
the request are unchanged. Disk entries are bexa ``.h5`` files with full
provenance, so anything in the cache is also a valid product.
"""

from __future__ import annotations

import shutil
import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import xarray as xr

from bexa._log import get_logger
from bexa.config.paths import cache_root
from bexa.core.provenance import fingerprint

log = get_logger(__name__)

DEFAULT_MEMORY_BYTES = 2 * 1024**3  # the memory level never grows past this
MEMORY_SHARE = 0.1  # ... nor past this share of the memory the process may use


def default_memory_bytes() -> int:
    """Budget of the in-memory level: a tenth of the usable memory, at most 2 GiB.

    A dataset of many scans shares one cache, so the level must stay small next
    to the results themselves; on a 64 GB session it is 2 GiB, on a 4 GB laptop
    about 400 MB.
    """
    from bexa.core.resources import memory_info

    return int(min(DEFAULT_MEMORY_BYTES, MEMORY_SHARE * memory_info().available))


@dataclass
class CacheEntry:
    key: str
    path: Path
    size_bytes: int
    created: float


def _nbytes(obj: Any) -> int:
    if isinstance(obj, xr.Dataset):
        return int(sum(v.nbytes for v in obj.variables.values()))
    if isinstance(obj, xr.DataArray):
        return int(obj.nbytes)
    if isinstance(obj, dict):
        return int(sum(_nbytes(v) for v in obj.values()))
    return int(getattr(obj, "nbytes", 0))


class Cache:
    """Two-level cache: an in-memory LRU in front of a folder of ``.h5`` files.

    Parameters
    ----------
    root
        Cache folder; ``None`` disables the disk level. Defaults come from
        :func:`bexa.config.paths.cache_root`.
    memory_bytes
        Budget of the in-memory level: ``"auto"`` (:func:`default_memory_bytes`),
        a number of bytes, or 0 / ``None`` to disable it.
    """

    def __init__(self, root: str | Path | None = None, memory_bytes: int | str | None = "auto"):
        self.root = Path(root) if root is not None else None
        if memory_bytes == "auto":
            memory_bytes = default_memory_bytes()
        self.memory_bytes = int(memory_bytes or 0)
        self._memory: OrderedDict[str, Any] = OrderedDict()
        self._memory_used = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ keys
    @staticmethod
    def key(files: Sequence[str | Path], params: Any) -> str:
        """Fingerprint of ``files`` and ``params`` (:func:`bexa.core.provenance.fingerprint`)."""
        return fingerprint(files, params)

    def path_for(self, key: str) -> Path | None:
        if self.root is None:
            return None
        return self.root / key[:2] / f"{key}.h5"

    # ---------------------------------------------------------------- access
    def get(self, key: str) -> Any | None:
        """Return the cached object or ``None``; disk hits are promoted to memory."""
        with self._lock:
            if key in self._memory:
                self._memory.move_to_end(key)
                return self._memory[key]
        path = self.path_for(key)
        if path is not None and path.exists():
            from bexa.io.cube import load

            try:
                obj = load(path, squeeze_single=False)
            except MemoryError as exc:  # over the budget right now; the file stays
                log.warning("not loading cache entry %s: %s", path, exc)
                return None
            except Exception as exc:  # a half-written file; drop it
                log.warning("discarding unreadable cache entry %s (%s)", path, exc)
                path.unlink(missing_ok=True)
                return None
            self._remember(key, obj)
            return obj
        return None

    def put(self, key: str, obj: Any) -> Path | None:
        """Store ``obj`` in memory and, when a root is set, on disk."""
        self._remember(key, obj)
        path = self.path_for(key)
        if path is None:
            return None
        from bexa.io.cube import save

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.h5")
        save(obj, tmp)
        tmp.replace(path)
        return path

    def _remember(self, key: str, obj: Any) -> None:
        if self.memory_bytes <= 0:
            return
        size = _nbytes(obj)
        if size > self.memory_bytes:
            return
        with self._lock:
            if key in self._memory:
                self._memory_used -= _nbytes(self._memory.pop(key))
            self._memory[key] = obj
            self._memory_used += size
            while self._memory_used > self.memory_bytes and self._memory:
                _, old = self._memory.popitem(last=False)
                self._memory_used -= _nbytes(old)

    def __contains__(self, key: str) -> bool:
        if key in self._memory:
            return True
        path = self.path_for(key)
        return bool(path is not None and path.exists())

    # ------------------------------------------------------------ maintenance
    def entries(self) -> list[CacheEntry]:
        if self.root is None or not self.root.exists():
            return []
        found = []
        for path in sorted(self.root.glob("*/*.h5")):
            st = path.stat()
            found.append(CacheEntry(path.stem, path, st.st_size, st.st_mtime))
        return found

    def clear(self, memory: bool = True, disk: bool = True) -> int:
        """Drop entries; returns the number of disk files removed."""
        removed = 0
        if memory:
            with self._lock:
                self._memory.clear()
                self._memory_used = 0
        if disk and self.root is not None and self.root.exists():
            for entry in self.entries():
                entry.path.unlink(missing_ok=True)
                removed += 1
            for sub in self.root.iterdir():
                if sub.is_dir() and not any(sub.iterdir()):
                    shutil.rmtree(sub, ignore_errors=True)
        return removed


_default: Cache | None = None


def default_cache(processed_root: str | Path | None = None) -> Cache:
    """The process-wide cache rooted at :func:`bexa.config.paths.cache_root`."""
    global _default
    if _default is None or (
        processed_root is not None and _default.root != cache_root(processed_root)
    ):
        _default = Cache(cache_root(processed_root))
    return _default
