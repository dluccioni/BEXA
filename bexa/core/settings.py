"""What bexa would use right now, and where each value comes from.

Device, memory budget, cache folder and profile are set by four layers
(defaults, profile, environment variables, call arguments). :func:`settings`
resolves them the way the library does and names the source of each, which
is the first thing to look at when a beamtime machine behaves unexpectedly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

__all__ = ["describe", "settings", "show"]


def settings() -> dict[str, Any]:
    """Resolved settings as nested dicts (see :func:`describe` for the text form)."""
    from bexa import __version__
    from bexa.config.loader import PROFILE_DIR_ENV, PROFILE_ENV, list_profiles, profile_dirs
    from bexa.config.paths import CACHE_DIR_ENV, cache_root, repo_root
    from bexa.core.backend import (
        DEFAULT_MEMORY_FRACTION,
        device_info,
        memory_budget,
        resolve_device,
    )

    env = os.environ
    requested = env.get("BEXA_DEVICE", "auto")
    resolved = resolve_device(requested)
    info = device_info()
    fraction = float(env.get("BEXA_MEMORY_FRACTION", DEFAULT_MEMORY_FRACTION))

    profile_name = env.get(PROFILE_ENV)
    processed_root: Path | None = None
    profile_error = None
    if profile_name:
        try:
            from bexa.config import load_profile

            processed_root = load_profile(profile_name).processed_root
        except Exception as exc:  # a stale BEXA_PROFILE must not break the report
            profile_error = str(exc)
    cache_dir = cache_root(processed_root)
    if env.get(CACHE_DIR_ENV):
        cache_source = CACHE_DIR_ENV
    elif processed_root is not None and cache_dir == Path(processed_root) / "bexa_cache":
        cache_source = f"processed_root of profile {profile_name}"
    else:
        cache_source = "user cache folder"

    return {
        "version": __version__,
        "repository": str(repo_root()),
        "device": {
            "requested": requested,
            "resolved": resolved,
            "source": "BEXA_DEVICE" if "BEXA_DEVICE" in env else "default",
            "cpu_count": info.get("cpu_count"),
            "gpu": info.get("gpu_name") if info.get("cuda") else None,
        },
        "memory": {
            "fraction": fraction,
            "source": "BEXA_MEMORY_FRACTION" if "BEXA_MEMORY_FRACTION" in env else "default",
            "available_gb": info.get("ram_available_gb"),
            "gpu_free_gb": info.get("gpu_free_gb"),
            "budget_gb": round(memory_budget(resolved, fraction) / 1e9, 2),
        },
        "cache": {"dir": str(cache_dir), "source": cache_source},
        "profile": {
            "name": profile_name,
            "source": PROFILE_ENV if profile_name else "not set",
            "error": profile_error,
            "search_path": [str(d) for d in profile_dirs()],
            "profile_dir_env": env.get(PROFILE_DIR_ENV),
            "available": sorted(list_profiles()),
        },
        "log_level": env.get("BEXA_LOG_LEVEL", "INFO"),
    }


def show() -> dict[str, Any]:
    """Print :func:`describe` and return :func:`settings` (``bexa.settings()``)."""
    print(describe())
    return settings()


def describe() -> str:
    """The settings as indented text, one line per value with its source."""
    s = settings()
    dev, mem, cache, prof = s["device"], s["memory"], s["cache"], s["profile"]
    lines = [
        f"bexa {s['version']} from {s['repository']}",
        f"device: {dev['resolved']} (requested {dev['requested']}, from {dev['source']})"
        + (f", GPU {dev['gpu']}" if dev["gpu"] else f", {dev['cpu_count']} CPU cores"),
        f"memory: budget {mem['budget_gb']} GB = fraction {mem['fraction']} (from {mem['source']})"
        f" of {mem['available_gb']} GB available"
        + (f" and {mem['gpu_free_gb']} GB free on the GPU" if mem["gpu_free_gb"] else ""),
        f"cache: {cache['dir']} (from {cache['source']})",
        f"profile: {prof['name'] or 'none'} (from {prof['source']})"
        + (f"; error: {prof['error']}" if prof["error"] else ""),
        f"  search path: {', '.join(prof['search_path']) or 'none'}",
        f"  available: {', '.join(prof['available']) or 'none'}",
        f"log level: {s['log_level']}",
    ]
    return "\n".join(lines)
