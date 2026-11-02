"""Background jobs with progress and Cancel.

One worker thread runs the jobs one after another, so the notebook stays
responsive while a scan is reduced. Every :func:`bexa._log.progress` loop of
the job (reduction batches, scans of a stack, packages, fits) reports to the
job through a thread-local hook; Cancel makes the hook raise
:class:`bexa._log.CancelledError` at the next step, which stops the work cleanly.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from bexa._log import CancelledError, get_logger, progress_hook

log = get_logger(__name__)

__all__ = ["Job", "JobRunner"]

REPORT_INTERVAL_S = 0.15  # progress reaches the views at most this often


@dataclass
class Job:
    """One piece of work: a label, the callable, and what to do with its result."""

    label: str
    function: Callable[[], Any]
    done: Callable[[Any], None] | None = None
    failed: Callable[[BaseException], None] | None = None
    status: str = "queued"  # queued, running, done, failed, cancelled
    fraction: float = 0.0
    message: str = ""
    started: float = 0.0
    seconds: float = 0.0
    error: BaseException | None = None
    result: Any = None
    _loops: list[list[Any]] = field(default_factory=list)

    def report(self, done: int, total: int | None, desc: str) -> None:
        """Note the step ``done`` of ``total`` of the loop ``desc`` (nested loops compose)."""
        for loop in self._loops:
            if loop[0] == desc:
                loop[1], loop[2] = done, total
                break
        else:
            self._loops.append([desc, done, total])
        fraction, scale = 0.0, 1.0
        for _, d, t in self._loops:
            if not t:
                continue
            fraction += scale * min(d, t) / t
            scale /= t
        self.fraction = min(fraction, 1.0)
        self.message = f"{desc} {done}/{total}" if total else f"{desc} {done}"


class JobRunner:
    """Runs jobs in a background thread (or inline with ``background=False``, for tests).

    ``on_change(job)`` is called when a job starts, progresses (throttled),
    and ends; it runs in the worker thread, so it should only set widget
    values.
    """

    def __init__(
        self, background: bool = True, on_change: Callable[[Job], None] | None = None
    ) -> None:
        self.background = background
        self.on_change = on_change
        self.history: list[Job] = []
        self.current: Job | None = None
        self._cancel = threading.Event()
        self._queue: queue.Queue[Job | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._last_report = 0.0

    @property
    def busy(self) -> bool:
        return self.current is not None or not self._queue.empty()

    def submit(
        self,
        label: str,
        function: Callable[[], Any],
        done: Callable[[Any], None] | None = None,
        failed: Callable[[BaseException], None] | None = None,
    ) -> Job:
        """Queue ``function``; ``done(result)`` or ``failed(error)`` is called when it ends."""
        job = Job(label, function, done, failed)
        self.history.append(job)
        if not self.background:
            self._run(job)
            return job
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._work, name="bexa-gui-jobs", daemon=True)
            self._thread.start()
        self._queue.put(job)
        return job

    def cancel(self) -> None:
        """Stop the running job at its next step and drop the queued ones."""
        while not self._queue.empty():
            try:
                job = self._queue.get_nowait()
            except queue.Empty:
                break
            if job is not None:
                job.status = "cancelled"
                self._notify(job, force=True)
        if self.current is not None:
            self._cancel.set()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until every queued job has ended (``False`` on timeout)."""
        end = None if timeout is None else time.monotonic() + timeout
        while self.busy:
            if end is not None and time.monotonic() > end:
                return False
            time.sleep(0.02)
        return True

    def _work(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            if job.status == "cancelled":
                continue
            self._run(job)

    def _hook(self, job: Job) -> Callable[[int, int | None, str], None]:
        def hook(done: int, total: int | None, desc: str) -> None:
            if self._cancel.is_set():
                raise CancelledError(f"{job.label} cancelled")
            job.report(done, total, desc)
            self._notify(job)

        return hook

    def _run(self, job: Job) -> None:
        self.current = job
        self._cancel.clear()
        job.status, job.started = "running", time.perf_counter()
        self._notify(job, force=True)
        try:
            with progress_hook(self._hook(job)):
                job.result = job.function()
            job.status, job.fraction = "done", 1.0
        except CancelledError as exc:
            job.status, job.error = "cancelled", exc
        except Exception as exc:  # reported to the view, which shows it
            job.status, job.error = "failed", exc
            log.debug("job %s failed", job.label, exc_info=True)
        finally:
            job.seconds = time.perf_counter() - job.started
            self.current = None
            self._cancel.clear()
        try:
            if job.status == "done" and job.done is not None:
                job.done(job.result)
            elif job.status != "done" and job.failed is not None and job.error is not None:
                job.failed(job.error)
        except Exception:
            log.exception("the callback of job %s failed", job.label)
        self._notify(job, force=True)

    def _notify(self, job: Job, force: bool = False) -> None:
        now = time.monotonic()
        if self.on_change is None or (not force and now - self._last_report < REPORT_INTERVAL_S):
            return
        self._last_report = now
        try:
            self.on_change(job)
        except Exception:
            log.exception("progress view failed")
