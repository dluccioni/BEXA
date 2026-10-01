"""Array backend selection (numpy or cupy), device transfers and memory budgeting.

Every function in :mod:`bexa.analysis` and every accumulator works on whatever
array module its input belongs to. This module answers three questions for
them: which module (``array_module``), how much memory may be used
(``memory_budget``) and how many frames fit in one batch (``choose_batch_frames``).
"""

from __future__ import annotations

import functools
import os
import types
from typing import Any, Literal

import numpy as np

from bexa._log import get_logger

log = get_logger(__name__)

Device = Literal["cpu", "cuda"]
DEFAULT_MEMORY_FRACTION = 0.5


@functools.lru_cache(maxsize=1)
def cupy_available() -> bool:
    """True when cupy imports and at least one CUDA device answers."""
    try:
        import cupy

        return int(cupy.cuda.runtime.getDeviceCount()) > 0
    except Exception:  # ImportError, CUDARuntimeError, driver problems
        return False


def resolve_device(device: str | None = "auto") -> Device:
    """Turn ``"auto" | "cpu" | "cuda" | "gpu"`` (or ``BEXA_DEVICE``) into ``"cpu"`` or ``"cuda"``.

    Raises
    ------
    RuntimeError
        If CUDA was requested explicitly but cupy or a device is missing.
    """
    name = (device or os.environ.get("BEXA_DEVICE") or "auto").lower()
    if name == "auto":
        return "cuda" if cupy_available() else "cpu"
    if name in ("cuda", "gpu"):
        if not cupy_available():
            raise RuntimeError("device='cuda' requested but cupy or a CUDA device is not available")
        return "cuda"
    if name == "cpu":
        return "cpu"
    raise ValueError(f"unknown device {device!r}; use 'auto', 'cpu' or 'cuda'")


def is_cupy_array(a: Any) -> bool:
    """True for cupy ndarrays without importing cupy."""
    return type(a).__module__.split(".")[0] == "cupy"


def device_of(a: Any) -> Device:
    """Device holding array ``a``."""
    return "cuda" if is_cupy_array(a) else "cpu"


def array_module(target: str | Any = "cpu") -> Any:
    """Return ``numpy`` or ``cupy`` for a device name or for an existing array."""
    if isinstance(target, str):
        if resolve_device(target) == "cuda":
            import cupy

            return cupy
        return np
    if is_cupy_array(target):
        import cupy

        return cupy
    return np


def ndimage_module(xp: Any) -> Any:
    """``scipy.ndimage`` for numpy, ``cupyx.scipy.ndimage`` for cupy.

    ``xp`` may be the array module or an array.
    """
    if not isinstance(xp, types.ModuleType):
        xp = array_module(xp)
    if xp is np:
        import scipy.ndimage

        return scipy.ndimage
    import cupyx.scipy.ndimage

    return cupyx.scipy.ndimage


def to_device(a: Any, device: str | None = "auto") -> Any:
    """Move an array to ``device`` (no copy when it is already there)."""
    if resolve_device(device) == "cuda":
        import cupy

        return cupy.asarray(a)
    if is_cupy_array(a):
        import cupy

        return cupy.asnumpy(a)
    return np.asarray(a)


def to_host(obj: Any) -> Any:
    """Bring arrays back to numpy, recursing through dicts, lists, tuples and xarray objects."""
    if is_cupy_array(obj):
        import cupy

        return cupy.asnumpy(obj)
    if isinstance(obj, dict):
        return {k: to_host(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_host(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_host(v) for v in obj)
    module = type(obj).__module__.split(".")[0]
    if module == "xarray":
        import xarray as xr

        if isinstance(obj, xr.DataArray):
            return obj.copy(data=to_host(obj.data)) if is_cupy_array(obj.data) else obj
        if isinstance(obj, xr.Dataset):
            if not any(is_cupy_array(v.data) for v in obj.variables.values()):
                return obj
            return obj.map(to_host, keep_attrs=True)
    return obj


def memory_budget(device: str | None = "cpu", fraction: float | None = None) -> int:
    """Bytes a single operation may use on ``device``.

    The budget is ``fraction`` of the memory this process may still use (default 0.5, or
    ``BEXA_MEMORY_FRACTION``): the SLURM job's allocation on a cluster, the free memory of
    the machine elsewhere (see :func:`bexa.core.resources.memory_info`), further limited by
    free GPU memory on CUDA.
    """
    from bexa.core.resources import memory_info

    if fraction is None:
        fraction = float(os.environ.get("BEXA_MEMORY_FRACTION", DEFAULT_MEMORY_FRACTION))
    budget = memory_info().available * fraction
    if resolve_device(device) == "cuda":
        import cupy

        free, _total = cupy.cuda.runtime.memGetInfo()
        budget = min(budget, free * fraction)
    return int(budget)


def check_fits(nbytes: int, what: str, device: str | None = "cpu", hint: str = "") -> int:
    """Raise ``MemoryError`` when ``nbytes`` exceeds the budget of ``device``; return the budget.

    The message names the size, the budget and, with ``hint``, the way out
    (an ROI, a larger downsample, ``store=``), so a result that cannot fit
    stops the call instead of the kernel.
    """
    budget = memory_budget(device)
    if nbytes > budget:
        raise MemoryError(
            f"{what} needs {nbytes / 1e9:.2f} GB but the memory budget is {budget / 1e9:.2f} GB"
            + (f"; {hint}" if hint else "")
            + " (BEXA_MEMORY_FRACTION raises the budget)"
        )
    return budget


def choose_batch_frames(
    frame_bytes: int,
    budget: int | None = None,
    live_copies: int = 4,
    min_frames: int = 1,
    max_frames: int | None = None,
    device: str | None = "cpu",
) -> int:
    """Number of frames per batch that keeps ``live_copies`` copies of the batch inside the budget.

    Parameters
    ----------
    frame_bytes
        Size of one frame after ROI and dtype conversion.
    budget
        Bytes available; ``memory_budget(device)`` when omitted.
    live_copies
        How many batch-sized arrays exist at once (the raw batch, its float
        copy, a filtered copy and accumulator work space is a typical 4).
    """
    if budget is None:
        budget = memory_budget(device)
    n = int(budget // max(frame_bytes * live_copies, 1))
    n = max(n, min_frames)
    if max_frames is not None:
        n = min(n, max_frames)
    return n


def is_oom_error(exc: BaseException) -> bool:
    """True for host ``MemoryError`` and cupy out-of-memory errors."""
    if isinstance(exc, MemoryError):
        return True
    name = type(exc).__name__
    return "OutOfMemory" in name or ("CUDARuntimeError" in name and "out of memory" in str(exc))


def free_device_memory() -> None:
    """Release cached GPU memory blocks (no-op on CPU-only machines)."""
    if cupy_available():
        import cupy

        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()


def device_info() -> dict[str, Any]:
    """Describe what this process may use: CPUs, memory (the job's inside SLURM) and the GPU."""
    from bexa.core.resources import memory_info, usable_cpus

    memory = memory_info()
    cpus, cpu_source = usable_cpus()
    info: dict[str, Any] = {
        "cpu_count": cpus,
        "cpu_source": cpu_source,
        "cpu_count_machine": os.cpu_count(),
        "ram_total_gb": round(memory.total / 1e9, 1),
        "ram_available_gb": round(memory.available / 1e9, 1),
        "memory_source": memory.source,
        "cuda": cupy_available(),
    }
    if info["cuda"]:
        import cupy

        props = cupy.cuda.runtime.getDeviceProperties(0)
        free, total = cupy.cuda.runtime.memGetInfo()
        name = props["name"]
        info["gpu_name"] = name.decode() if isinstance(name, bytes) else str(name)
        info["gpu_total_gb"] = round(total / 1e9, 1)
        info["gpu_free_gb"] = round(free / 1e9, 1)
    return info
