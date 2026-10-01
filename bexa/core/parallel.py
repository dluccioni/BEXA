"""Threads for I/O overlap, processes for per-file jobs, and HDF5-safe settings.

HDF5 reads and hdf5plugin decompression release the GIL, so threads overlap
file reading with computation (:class:`Prefetcher`). CPU-bound per-file work,
such as reducing one XFEL point file, runs in spawned processes with the BLAS
thread count pinned to one so the workers do not oversubscribe the cores.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import queue
import threading
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")

BLAS_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

_SENTINEL = object()


def pin_blas_threads(n: int = 1) -> None:
    """Limit BLAS/OpenMP threads (call before numpy is imported in a worker)."""
    for name in BLAS_THREAD_VARIABLES:
        os.environ.setdefault(name, str(n))


def compute_threads() -> int:
    """Threads for the CPU work of one process (:func:`bexa.core.backend.gaussian_frames`).

    ``BEXA_THREADS`` when set, else up to eight of the usable cores: the
    gain flattens beyond that, and worker processes get their share.
    """
    env = os.environ.get("BEXA_THREADS")
    if env:
        return max(1, int(env))
    from bexa.core.resources import usable_cpus

    cpus, _ = usable_cpus()
    return max(1, min(8, cpus))


_compute_pool: ThreadPoolExecutor | None = None
_compute_lock = threading.Lock()


def compute_pool() -> ThreadPoolExecutor:
    """One shared pool of :func:`compute_threads` threads for frame-parallel numpy work."""
    global _compute_pool
    with _compute_lock:
        if _compute_pool is None:
            _compute_pool = ThreadPoolExecutor(
                max_workers=compute_threads(), thread_name_prefix="bexa-compute"
            )
        return _compute_pool


def default_workers(kind: str = "io") -> int:
    """Sensible worker counts: 8 threads for I/O, all but one usable core for processes.

    The usable cores are the ones this process may run on (a SLURM job's allocation on a
    cluster), not the cores of the node.
    """
    from bexa.core.resources import usable_cpus

    cpus, _ = usable_cpus()
    if kind == "io":
        return max(1, min(8, cpus))
    return max(1, cpus - 1)


class Prefetcher(Generic[T]):
    """Run an iterator in a background thread so the next item is ready when asked for.

    Exceptions raised by the iterator are re-raised in the consumer. The queue
    is bounded (default: two items) so memory stays flat.

    Examples
    --------
    >>> for batch in Prefetcher(reader.batches()):
    ...     accumulate(batch)
    """

    def __init__(self, iterable: Iterable[T], depth: int = 2) -> None:
        self._source = iter(iterable)
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, depth))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="bexa-prefetch")
        self._started = False

    def _run(self) -> None:
        try:
            for item in self._source:
                if self._stop.is_set():
                    break
                self._queue.put(item)
        except BaseException as exc:  # forwarded to the consumer
            self._queue.put(_ExceptionWrapper(exc))
        finally:
            self._queue.put(_SENTINEL)

    def __iter__(self) -> Iterator[T]:
        if not self._started:
            self._started = True
            self._thread.start()
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                return
            if isinstance(item, _ExceptionWrapper):
                raise item.exc
            yield item

    def close(self) -> None:
        """Stop the producer thread early (drains one item so it can exit)."""
        self._stop.set()
        with contextlib.suppress(queue.Empty):
            self._queue.get_nowait()


class _ExceptionWrapper:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


def thread_map(fn: Callable[[T], R], items: Iterable[T], workers: int | None = None) -> list[R]:
    """Apply ``fn`` to ``items`` in threads, preserving order."""
    items = list(items)
    if not items:
        return []
    workers = min(workers or default_workers("io"), len(items))
    if workers <= 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bexa-io") as pool:
        return list(pool.map(fn, items))


def _init_worker(blas_threads: int, env: dict[str, str] | None) -> None:
    """Runs first in every spawned worker: the environment of the job, then the BLAS pinning."""
    if env:
        os.environ.update(env)
    pin_blas_threads(blas_threads)


def process_pool(
    workers: int | None = None, env: dict[str, str] | None = None
) -> ProcessPoolExecutor:
    """A spawn-based process pool with BLAS threads pinned in every worker.

    Spawn (rather than fork) keeps HDF5 and CUDA state out of the children,
    which is what the legacy XFEL cube builder learned the hard way. ``env``
    sets environment variables in every worker before it imports numpy (a
    smaller ``BEXA_MEMORY_FRACTION`` when several workers share the job's
    memory, for example).
    """
    workers = workers or default_workers("cpu")
    ctx = mp.get_context("spawn")
    return ProcessPoolExecutor(
        max_workers=workers, mp_context=ctx, initializer=_init_worker, initargs=(1, env)
    )
