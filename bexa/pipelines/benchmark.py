"""Timing of readers, reductions and plots on synthetic (or real) data, with baselines.

``run_suite`` times the steps of a typical session and records throughput,
the compute-to-read ratio, peak RAM and peak GPU memory. Results are plain
dicts so they can be saved as JSON; ``compare`` flags regressions against a
stored baseline. The CLI wrapper is ``bexa bench``.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from bexa._log import get_logger
from bexa.core import backend
from bexa.core.provenance import peak_rss_mb

log = get_logger(__name__)

__all__ = ["compare", "load_baseline", "machine_id", "run_suite", "save_baseline"]


def machine_id() -> str:
    """``<hostname>-<device>`` used to name baseline files."""
    device = backend.resolve_device("auto")
    return f"{platform.node()}-{device}".replace(" ", "_")


def _timed(fn: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


def _gpu_peak_mb() -> float | None:
    if not backend.cupy_available():
        return None
    import cupy

    pool = cupy.get_default_memory_pool()
    return pool.used_bytes() / 1e6


def bench_scan(scan: Any, device: str = "auto", batch_frames: int | None = None) -> dict[str, Any]:
    """Time metadata, a full read, a preview and the standard reductions of one scan."""
    from bexa.core import reductions as rd

    results: dict[str, Any] = {}
    frame_bytes = scan.source.frame_bytes()
    n = scan.n_frames
    results["frames"] = n
    results["frame_shape"] = list(scan.frame_shape)
    results["stored_gb"] = frame_bytes * n / 1e9

    _, t_info = _timed(scan.info)
    results["info_s"] = t_info

    _, t_read = _timed(scan.source.read_frames, slice(0, n))
    results["read_s"] = t_read
    results["read_fps"] = n / t_read if t_read else float("inf")
    results["read_mb_s"] = frame_bytes * n / 1e6 / t_read if t_read else float("inf")

    _, t_preview = _timed(scan.preview, downsample=4, cache=False, device=device)
    results["preview_s"] = t_preview

    accs = [rd.Sum(), rd.Max(), rd.MotorCOM(axes=tuple(scan.structure.motor_dims), sigma=3.0)]
    res, t_reduce = _timed(
        rd.reduce, scan, accs, device=device, batch_frames=batch_frames, cache=False
    )
    results["reduce_s"] = t_reduce
    results["reduce_fps"] = n / t_reduce if t_reduce else float("inf")
    results["compute_to_read"] = t_reduce / t_read if t_read else float("nan")
    results["device"] = str(res["sum"].attrs.get("device", device))
    results["peak_rss_mb"] = peak_rss_mb()
    results["gpu_used_mb"] = _gpu_peak_mb()
    return results


def bench_plots(scan: Any) -> dict[str, float]:
    """Time the standard figure set (headless)."""
    import tempfile

    import matplotlib

    matplotlib.use("Agg")
    from bexa.pipelines.dfxm_report import report

    with tempfile.TemporaryDirectory() as tmp:
        _, t_report = _timed(report, scan, tmp, downsample=4, device="cpu")
    return {"report_s": t_report}


def run_suite(
    quick: bool = True,
    device: str = "auto",
    real_path: str | Path | None = None,
    work_dir: str | Path | None = None,
    plots: bool = True,
) -> dict[str, Any]:
    """Benchmark a synthetic ESRF scan (small when ``quick``) and optionally a real frame stack.

    ``real_path`` is any stack the generic reader opens (``.npy``, tiff,
    ``file.h5::/path``); it is benchmarked as a plain frame sequence.
    """
    import tempfile

    from bexa.core.scan import open as open_scan
    from bexa.testing import make_esrf_scan

    results: dict[str, Any] = {
        "machine": machine_id(),
        "quick": quick,
        "device": device,
        "cupy": backend.cupy_available(),
        "cpu_count": __import__("os").cpu_count(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cases": {},
    }
    tmp = tempfile.TemporaryDirectory() if work_dir is None else None
    root = Path(work_dir) if work_dir is not None else Path(tmp.name)  # type: ignore[union-attr]
    try:
        if quick:
            made = make_esrf_scan(root / "bench", dataset="quick", motors=(("chi", 4), ("mu", 12)))
        else:  # realistic frame size, few frames (the generator keeps float frames in memory)
            made = make_esrf_scan(
                root / "bench",
                dataset="full",
                motors=(("chi", 2), ("mu", 16)),
                frame_shape=(2160, 2560),
                n_files=2,
            )
        scan = open_scan(made.dataset_dir, cache=False)
        results["cases"]["esrf_synthetic"] = bench_scan(scan, device=device)
        if plots:
            results["cases"]["esrf_synthetic"].update(bench_plots(scan))
        scan.close()
        if real_path is not None:
            scan = open_scan(real_path, format="generic_stack", cache=False)
            results["cases"]["real_stack"] = bench_scan(scan, device=device)
            scan.close()
    finally:
        if tmp is not None:
            tmp.cleanup()
    return results


def save_baseline(results: dict[str, Any], folder: str | Path) -> Path:
    """Write ``<folder>/<machine>.json``."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{results['machine']}.json"
    path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return path


def load_baseline(folder: str | Path, machine: str | None = None) -> dict[str, Any] | None:
    path = Path(folder) / f"{machine or machine_id()}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


TIMED_KEYS = ("info_s", "read_s", "preview_s", "reduce_s", "report_s")


def compare(results: dict[str, Any], baseline: dict[str, Any], tolerance: float = 0.2) -> list[str]:
    """Regressions: timings slower than the baseline by more than ``tolerance`` (fraction)."""
    regressions = []
    for case, values in results.get("cases", {}).items():
        reference = baseline.get("cases", {}).get(case, {})
        for key in TIMED_KEYS:
            now, before = values.get(key), reference.get(key)
            if now is None or before is None or before <= 0:
                continue
            if now > before * (1 + tolerance):
                regressions.append(f"{case}.{key}: {now:.3f} s vs baseline {before:.3f} s")
    return regressions


def format_results(results: dict[str, Any]) -> str:
    lines = [f"bexa benchmark on {results['machine']} ({'quick' if results['quick'] else 'full'})"]
    for case, values in results.get("cases", {}).items():
        lines.append(f"  {case}: {values.get('frames')} frames of {values.get('frame_shape')}")
        for key in ("info_s", "read_s", "preview_s", "reduce_s", "report_s"):
            if key in values:
                lines.append(f"    {key:<10} {values[key]:8.3f} s")
        for key in ("read_mb_s", "reduce_fps", "compute_to_read", "peak_rss_mb", "gpu_used_mb"):
            if values.get(key) is not None:
                lines.append(f"    {key:<15} {float(np.round(values[key], 3))}")
    return "\n".join(lines)
