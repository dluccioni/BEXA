"""Load beamtime profiles from YAML, with environment overrides and path aliases.

Loading order (later wins): package defaults, the format spec (with ``extends``),
the beamtime profile (with ``format_overrides``), environment variables
(``BEXA_PROFILE``, ``BEXA_DEVICE``, ``BEXA_MEMORY_FRACTION``, ``BEXA_CACHE_DIR``),
then explicit call arguments.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from bexa._log import get_logger
from bexa.config.models import BeamtimeProfile
from bexa.config.paths import apply_aliases, package_configs

log = get_logger(__name__)

PROFILE_ENV = "BEXA_PROFILE"
PROFILE_DIR_ENV = "BEXA_PROFILE_DIR"
PROFILE_SUFFIXES = (".yaml", ".yml")


def profile_dirs() -> list[Path]:
    """Folders searched for profiles: ``BEXA_PROFILE_DIR``, ``./configs/beamtimes``, the repo's."""
    dirs: list[Path] = []
    env = os.environ.get(PROFILE_DIR_ENV)
    if env:
        dirs.extend(Path(p) for p in env.split(os.pathsep) if p)
    dirs.append(Path.cwd() / "configs" / "beamtimes")
    dirs.append(package_configs() / "beamtimes")
    seen: set[Path] = set()
    unique = []
    for d in dirs:
        if d.exists() and d.resolve() not in seen:
            seen.add(d.resolve())
            unique.append(d)
    return unique


def list_profiles() -> dict[str, Path]:
    """Profile names available on this machine (first folder in the search path wins)."""
    found: dict[str, Path] = {}
    for folder in profile_dirs():
        for path in sorted(folder.iterdir()):
            if path.suffix in PROFILE_SUFFIXES and path.stem not in found:
                found[path.stem] = path
    return found


def _resolve_profile_path(name_or_path: str | Path) -> Path:
    candidate = Path(name_or_path)
    if candidate.suffix in PROFILE_SUFFIXES and candidate.exists():
        return candidate
    profiles = list_profiles()
    if str(name_or_path) in profiles:
        return profiles[str(name_or_path)]
    available = ", ".join(sorted(profiles)) or "none"
    raise FileNotFoundError(
        f"no beamtime profile {name_or_path!r}; available: {available} "
        f"(searched {[str(d) for d in profile_dirs()]})"
    )


def load_profile(name_or_path: str | Path | None = None, **overrides: Any) -> BeamtimeProfile:
    """Load a profile by name, by path, or from ``BEXA_PROFILE`` when ``None``.

    Keyword overrides replace top-level fields (``load_profile("hc6293", detector="pco_nf")``).
    Data roots are passed through :func:`bexa.config.paths.apply_aliases` so the same
    profile works at the beamline and on the lab PC.
    """
    if name_or_path is None:
        name_or_path = os.environ.get(PROFILE_ENV)
        if not name_or_path:
            raise ValueError("no profile given and BEXA_PROFILE is not set")
    path = _resolve_profile_path(name_or_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"profile {path} must be a YAML mapping")
    raw.setdefault("name", path.stem)
    raw.update(overrides)
    profile = BeamtimeProfile.model_validate(raw)
    profile.source_path = path
    for field in ("root", "raw_root", "processed_root"):
        value = getattr(profile, field)
        if value is not None:
            setattr(profile, field, apply_aliases(value, profile.path_aliases))
    log.debug("loaded profile %s from %s", profile.name, path)
    return profile


def env_settings() -> dict[str, Any]:
    """Settings taken from environment variables (only the ones that are set)."""
    settings: dict[str, Any] = {}
    if os.environ.get("BEXA_DEVICE"):
        settings["device"] = os.environ["BEXA_DEVICE"]
    if os.environ.get("BEXA_MEMORY_FRACTION"):
        settings["memory_fraction"] = float(os.environ["BEXA_MEMORY_FRACTION"])
    if os.environ.get("BEXA_CACHE_DIR"):
        settings["cache_dir"] = Path(os.environ["BEXA_CACHE_DIR"])
    if os.environ.get(PROFILE_ENV):
        settings["profile"] = os.environ[PROFILE_ENV]
    return settings


def profile_skeleton(name: str, format_name: str, root: str | Path | None = None) -> dict[str, Any]:
    """A starting profile with the common sections filled with placeholders."""
    return {
        "name": name,
        "format": format_name,
        "root": str(root) if root else "TODO: data root at the beamline",
        "processed_root": "TODO: where reduced results and the cache go",
        "path_aliases": {},
        "samples": {"SAMPLE": {"dataset": "TODO: dataset folder name", "cif": None}},
        "detector": None,
        "geometry": {"distance_mm": None, "effective_pixel_nm": None},
        "defaults": {"precision": "float32", "downsample": [1, 1, 4, 4], "device": "auto"},
        "rois": {},
        "format_overrides": {},
    }


def write_profile_skeleton(
    path: str | Path,
    name: str,
    format_name: str,
    root: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Write :func:`profile_skeleton` as YAML; refuses to overwrite unless ``force``."""
    out = Path(path)
    if out.exists() and not force:
        raise FileExistsError(f"{out} exists; pass force=True to overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.safe_dump(profile_skeleton(name, format_name, root), sort_keys=False),
        encoding="utf-8",
    )
    return out
