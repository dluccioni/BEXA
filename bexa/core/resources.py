"""What this process may use: memory and CPUs, inside a SLURM job or on a whole machine.

On a cluster node psutil reports the node: hundreds of GB and dozens of cores. A SLURM job
(a Jupyter-SLURM session too) only owns part of it, and the kernel kills the job when it
goes over its memory allocation. SLURM enforces that allocation with a cgroup, so the limit
and the current use are read from the cgroup of this process (v2, or v1 on older systems),
falling back to the SLURM environment variables and finally to the machine.

Page cache counts towards a cgroup's use but is freed under pressure, and reading large
HDF5 files fills it quickly; the inactive file pages are therefore not counted as used.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import psutil

__all__ = ["MemoryInfo", "cgroup_memory", "memory_info", "usable_cpus"]

CGROUP_ROOT = Path("/sys/fs/cgroup")
PROC_CGROUP = Path("/proc/self/cgroup")
UNLIMITED = 2**60  # cgroup v1 writes a number close to 2**63 for "no limit"


@dataclass(frozen=True)
class MemoryInfo:
    """Memory this process can still use, the limit it is under, and where the limit comes from."""

    available: int
    total: int
    source: str


def _read_int(path: Path) -> int | None:
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    if text == "max":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _read_stat(path: Path) -> dict[str, int]:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return {}
    stats = {}
    for line in lines:
        key, _, value = line.partition(" ")
        if value.strip().isdigit():
            stats[key] = int(value)
    return stats


def _cgroup_paths(text: str) -> tuple[str | None, str | None]:
    """The v2 path and the v1 memory-controller path of a ``/proc/self/cgroup`` listing."""
    v2 = v1 = None
    for line in text.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, path = parts
        if hierarchy == "0" and controllers == "":
            v2 = path
        elif "memory" in controllers.split(","):
            v1 = path
    return v2, v1


def _walk_up(base: Path, path: str) -> Iterator[Path]:
    """``base/a/b/c``, ``base/a/b``, ``base/a``, ``base``: a limit may sit on any ancestor."""
    parts = [p for p in path.strip("/").split("/") if p]
    for depth in range(len(parts), -1, -1):
        yield base.joinpath(*parts[:depth])


def cgroup_memory(
    root: Path = CGROUP_ROOT, proc_cgroup: Path = PROC_CGROUP
) -> tuple[int, int, str] | None:
    """``(limit, used, version)`` of the tightest memory limit on this process, or None.

    ``used`` leaves out the inactive file cache, which the kernel reclaims before it
    kills anything. Returns None outside Linux and when no level sets a limit.
    """
    try:
        text = proc_cgroup.read_text()
    except OSError:
        return None
    v2, v1 = _cgroup_paths(text)
    best: tuple[int, int, str] | None = None

    def consider(limit: int | None, usage: int | None, stat: dict[str, int], version: str) -> None:
        nonlocal best
        if limit is None or limit >= UNLIMITED:
            return
        used = max((usage or 0) - stat.get("inactive_file", stat.get("total_inactive_file", 0)), 0)
        if best is None or limit - used < best[0] - best[1]:
            best = (limit, used, version)

    if v2 is not None:
        for folder in _walk_up(root, v2):
            consider(
                _read_int(folder / "memory.max"),
                _read_int(folder / "memory.current"),
                _read_stat(folder / "memory.stat"),
                "cgroup v2",
            )
    if best is None and v1 is not None:
        for folder in _walk_up(root / "memory", v1):
            consider(
                _read_int(folder / "memory.limit_in_bytes"),
                _read_int(folder / "memory.usage_in_bytes"),
                _read_stat(folder / "memory.stat"),
                "cgroup v1",
            )
    return best


def _slurm_memory_limit() -> int | None:
    """The job's memory in bytes from ``SLURM_MEM_PER_NODE`` or ``SLURM_MEM_PER_CPU`` (MB)."""
    env = os.environ
    per_node = env.get("SLURM_MEM_PER_NODE", "")
    if per_node.isdigit():
        return int(per_node) * 1024**2
    per_cpu = env.get("SLURM_MEM_PER_CPU", "")
    cpus = env.get("SLURM_CPUS_ON_NODE", env.get("SLURM_CPUS_PER_TASK", ""))
    if per_cpu.isdigit() and cpus.isdigit():
        return int(per_cpu) * int(cpus) * 1024**2
    return None


def memory_info(root: Path = CGROUP_ROOT, proc_cgroup: Path = PROC_CGROUP) -> MemoryInfo:
    """The memory this process can use: the job's cgroup, the SLURM allocation or the machine."""
    vm = psutil.virtual_memory()
    info = MemoryInfo(available=int(vm.available), total=int(vm.total), source="machine")
    limited = cgroup_memory(root, proc_cgroup)
    if limited is not None:
        limit, used, version = limited
        available = max(limit - used, 0)
        if available < info.available:
            info = MemoryInfo(available=available, total=limit, source=version)
    elif os.environ.get("SLURM_JOB_ID"):
        allocation = _slurm_memory_limit()
        if allocation is not None:
            available = max(allocation - psutil.Process().memory_info().rss, 0)
            if available < info.available:
                info = MemoryInfo(available=available, total=allocation, source="SLURM allocation")
    return info


def usable_cpus() -> tuple[int, str]:
    """``(cores, source)``: the cores this process may run on, not the cores of the node."""
    affinity = getattr(os, "sched_getaffinity", None)  # Linux only
    if affinity is not None:
        count, source = len(affinity(0)), "CPU affinity"
    else:
        count, source = os.cpu_count() or 1, "machine"
    per_task = os.environ.get("SLURM_CPUS_PER_TASK", "")
    if per_task.isdigit() and 0 < int(per_task) < count:
        count, source = int(per_task), "SLURM_CPUS_PER_TASK"
    return max(count, 1), source
