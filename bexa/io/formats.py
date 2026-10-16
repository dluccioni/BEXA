"""Format specs: one YAML document per file layout, interpreted by an engine.

A spec names the paths, keys, aliases and conventions inside a layout. Specs
inherit from a base with ``extends``, accept alias lists for keys that were
renamed between beamtimes, and carry a ``fingerprint`` so ``bexa.open(path)``
can recognise the layout. :func:`draft_spec` writes a starting spec for an
unknown layout during a beamtime.
"""

from __future__ import annotations

import copy
import fnmatch
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from bexa._log import get_logger
from bexa.config.paths import package_configs

log = get_logger(__name__)

FORMAT_DIR_ENV = "BEXA_FORMAT_DIR"
SPEC_SUFFIXES = (".yaml", ".yml")
_PLACEHOLDER = re.compile(r"\{(\w+)(:[^{}]*)?\}")
_H5_SUFFIXES = (".h5", ".hdf5", ".nxs", ".nx")


class FormatSpec(BaseModel):
    """Description of one file layout (see ``configs/formats/*.yaml``)."""

    model_config = ConfigDict(extra="allow")

    name: str
    engine: str
    extends: str | None = None
    description: str = ""
    abstract: bool = False  # a base for ``extends`` only; never matched by sniffing
    fingerprint: dict[str, Any] = Field(default_factory=dict)
    layout: dict[str, Any] = Field(default_factory=dict)
    motors: dict[str, Any] = Field(default_factory=dict)
    keys: dict[str, list[str]] = Field(default_factory=dict)
    energy: dict[str, Any] = Field(default_factory=dict)
    detectors: dict[str, dict[str, Any]] = Field(default_factory=dict)
    convention: str | None = None
    axis_priority: list[str] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)
    source_path: Path | None = None

    @field_validator("keys", mode="before")
    @classmethod
    def _alias_lists(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: ([v] if isinstance(v, str) else list(v)) for k, v in value.items()}
        return value

    # ---- key aliases -----------------------------------------------------
    def aliases(self, key: str) -> list[str]:
        """All names ever used for a logical key (first is the preferred one)."""
        return list(self.keys.get(key, [key]))

    def resolve_key(self, key: str, available: Iterable[str]) -> str | None:
        """The first alias of ``key`` present in ``available``, or ``None``."""
        names = set(available)
        for alias in self.aliases(key):
            if alias in names:
                return alias
        return None

    # ---- path templates --------------------------------------------------
    def path(self, item: str, **fields: Any) -> str:
        """Fill the layout template ``item`` with ``fields`` (missing ones stay as-is)."""
        try:
            template = self.layout[item]
        except KeyError:
            raise KeyError(f"format {self.name!r} has no layout entry {item!r}") from None
        return safe_format(str(template), **fields)

    def motor_path(self, kind: str, **fields: Any) -> str:
        """Fill the motor template ``kind`` (``per_frame``, ``scalar``, ``structure``)."""
        template = self.motors.get(kind)
        if template is None:
            raise KeyError(f"format {self.name!r} has no motor template {kind!r}")
        return safe_format(str(template), **fields)

    def known_motors(self) -> list[str]:
        return list(self.motors.get("known", []))

    def min_steps(self) -> dict[str, float]:
        """``{motor: step}`` from ``motors.min_step``: a smaller change is drift, not a move."""
        steps = self.motors.get("min_step", {}) or {}
        return {str(name): float(step) for name, step in steps.items()}

    def motor_aliases(self, name: str) -> list[str]:
        """Names to try in the file for the logical motor ``name`` (``z1`` -> ``samz``)."""
        aliases = self.motors.get("aliases", {}) or {}
        extra = aliases.get(name, [])
        return [name] + ([extra] if isinstance(extra, str) else list(extra))

    def detector_info(self, detector: str) -> dict[str, Any]:
        return dict(self.detectors.get(detector, {}))


def safe_format(template: str, **fields: Any) -> str:
    """``str.format`` that leaves unknown placeholders untouched.

    ``safe_format("{dataset}/scan{scan:04d}", scan=7)`` -> ``"{dataset}/scan0007"``.
    """

    def substitute(match: re.Match[str]) -> str:
        name, spec = match.group(1), match.group(2) or ""
        if name not in fields:
            return match.group(0)
        return format(fields[name], spec[1:]) if spec else str(fields[name])

    return _PLACEHOLDER.sub(substitute, template)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge; lists and scalars in ``override`` replace those in ``base``."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


# ------------------------------------------------------------------ loading
def format_dirs() -> list[Path]:
    """Folders searched for specs: ``BEXA_FORMAT_DIR``, ``./configs/formats``, the repo's."""
    dirs: list[Path] = []
    env = os.environ.get(FORMAT_DIR_ENV)
    if env:
        dirs.extend(Path(p) for p in env.split(os.pathsep) if p)
    dirs.append(Path.cwd() / "configs" / "formats")
    dirs.append(package_configs() / "formats")
    unique: list[Path] = []
    seen: set[Path] = set()
    for d in dirs:
        if d.exists() and d.resolve() not in seen:
            seen.add(d.resolve())
            unique.append(d)
    return unique


def list_specs() -> dict[str, Path]:
    """Spec names available on this machine (first folder in the search path wins)."""
    found: dict[str, Path] = {}
    for folder in format_dirs():
        for path in sorted(folder.iterdir()):
            if path.suffix in SPEC_SUFFIXES and path.stem not in found:
                found[path.stem] = path
    return found


def all_specs() -> list[FormatSpec]:
    """Every spec of :func:`list_specs` that loads; a broken one is logged and left out."""
    specs: list[FormatSpec] = []
    for name in list_specs():
        try:
            specs.append(load_spec(name))
        except Exception as exc:  # one broken spec must not disable sniffing
            log.warning("skipping format spec %s: %s", name, exc)
    return specs


def _read_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"format spec {path} must be a YAML mapping")
    return raw


def load_spec(
    name_or_path: str | Path,
    overrides: dict[str, Any] | None = None,
    _chain: tuple[str, ...] = (),
) -> FormatSpec:
    """Load a spec by name or path, resolving ``extends`` and applying ``overrides``."""
    candidate = Path(name_or_path)
    if candidate.suffix in SPEC_SUFFIXES and candidate.exists():
        path = candidate
    else:
        specs = list_specs()
        if str(name_or_path) not in specs:
            available = ", ".join(sorted(specs)) or "none"
            raise FileNotFoundError(f"no format spec {name_or_path!r}; available: {available}")
        path = specs[str(name_or_path)]
    raw = _read_yaml(path)
    raw.setdefault("name", path.stem)
    chain = (*_chain, raw["name"])
    if raw["name"] in _chain:
        raise ValueError(f"circular 'extends' chain: {' -> '.join(chain)}")
    parent_name = raw.get("extends")
    if parent_name:
        parent = load_spec(parent_name, _chain=chain)
        inherited = parent.model_dump(exclude={"source_path", "name", "extends", "abstract"})
        merged = deep_merge(inherited, raw)
    else:
        merged = raw
    if overrides:
        merged = deep_merge(merged, overrides)
    spec = FormatSpec.model_validate(merged)
    spec.source_path = path
    return spec


# ----------------------------------------------------------------- matching
# the fingerprint items that open HDF5 files; a quick match leaves them out
CONTENT_ITEMS = ("master_paths", "h5_paths", "h5_attrs")


def _scan_numbers(f: Any, limit: int = 3) -> list[int]:
    """The first scan numbers of an open BLISS master, from its top-level keys ``"7.1"``."""
    numbers: list[int] = []
    for key in f:
        m = re.fullmatch(r"(\d+)\.1", key)
        if m:
            numbers.append(int(m.group(1)))
    return sorted(numbers)[:limit]


def _h5_holds(path: Path, templates: Iterable[str], scans: bool = False) -> bool:
    """True when an HDF5 file holds every path of ``templates``, opened once.

    With ``scans``, the ``{scan}`` field is filled with each of the first scan
    numbers of the master in turn, and one scan holding every path is enough.
    """
    from bexa.io.h5files import open_h5

    templates = list(templates)
    try:
        with open_h5(path, retry=False) as f:
            numbers = (_scan_numbers(f) or [1]) if scans else [1]
            return any(all(safe_format(t, scan=n) in f for t in templates) for n in numbers)
    except OSError:
        return False


def _text_has_tags(path: Path, tags: Iterable[str], limit: int = 65536) -> bool:
    """True when the first ``limit`` bytes of a text file hold a line starting with each tag."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            head = f.read(limit)
    except OSError:
        return False
    lines = head.split("\n")
    return all(any(line.startswith(tag) for line in lines) for tag in tags)


class FolderListing:
    """The entries of folders, each folder listed once; one survey of a tree shares one.

    Matching a spec against a folder looks at its names (and at its parents'
    for ``dir_globs``) several times; on a network share every listing costs
    tens of milliseconds, so the listings are kept.
    """

    def __init__(self) -> None:
        self._folders: dict[Path, dict[str, tuple[bool, bool]]] = {}

    def entries(self, folder: Path) -> dict[str, tuple[bool, bool]]:
        """``{name: (is a folder, is a link)}`` of ``folder``; an unreadable folder is empty."""
        found = self._folders.get(folder)
        if found is None:
            found = {}
            try:
                with os.scandir(folder) as listing:
                    for entry in listing:
                        try:
                            found[entry.name] = (entry.is_dir(), entry.is_symlink())
                        except OSError:
                            found[entry.name] = (False, False)
            except OSError:
                pass
            self._folders[folder] = found
        return found

    def glob(self, folder: Path, pattern: str, folders: bool | None = None) -> list[Path]:
        """The sorted entries of ``folder`` matching ``pattern``; ``folders`` keeps one kind."""
        return [
            folder / name
            for name, (is_dir, _) in sorted(self.entries(folder).items())
            if fnmatch.fnmatch(name, pattern) and (folders is None or is_dir == folders)
        ]


def _dir_and_parents(folder: Path, levels: int = 3) -> list[Path]:
    out = [folder]
    for _ in range(levels):
        if folder.parent == folder:
            break
        folder = folder.parent
        out.append(folder)
    return out


def _fingerprint_checks(
    spec: FormatSpec, path: Path, is_dir: bool, listing: FolderListing, quick: bool = False
) -> list[bool]:
    """One boolean per fingerprint item; ``quick`` leaves out the items that open HDF5 files."""
    fp = spec.fingerprint
    results: list[bool] = []
    if is_dir and fnmatch.fnmatch(path.name, fp.get("scan_dir_glob", "")):
        path = path.parent  # a scanNNNN folder is recognised through its dataset folder
    folder = path if is_dir else path.parent

    masters: list[Path] = []
    if "master_glob" in fp:
        if is_dir:
            masters = listing.glob(path, fp["master_glob"], folders=False)
        elif path.suffix in _H5_SUFFIXES:
            masters = [path]
        results.append(bool(masters))
    elif not is_dir and path.suffix in _H5_SUFFIXES:
        masters = [path]

    if "master_paths" in fp and not quick:
        results.append(any(_h5_holds(m, fp["master_paths"], scans=True) for m in masters[:3]))

    if "scan_dir_glob" in fp:
        results.append(bool(listing.glob(folder, fp["scan_dir_glob"], folders=True)))

    if "dir_globs" in fp:
        found = False
        for place in _dir_and_parents(folder):
            names = list(listing.entries(place))
            if all(any(fnmatch.fnmatch(n, g) for n in names) for g in fp["dir_globs"]):
                found = True
                break
        results.append(found)

    if "file_globs" in fp:
        results.append(all(listing.glob(folder, g) for g in fp["file_globs"]))

    texts: list[Path] = []  # a text file (a SPEC session) given itself or inside the folder
    if "text_glob" in fp:
        if is_dir:
            texts = listing.glob(path, fp["text_glob"], folders=False)
        elif fnmatch.fnmatch(path.name, fp["text_glob"]):
            texts = [path]
        elif path.is_file():  # a frame next to the text file
            texts = listing.glob(folder, fp["text_glob"], folders=False)
        results.append(bool(texts))
    if "text_tags" in fp:
        results.append(any(_text_has_tags(t, fp["text_tags"]) for t in texts[:3]))

    if "h5_paths" in fp and not quick:
        candidates = listing.glob(path, "*.h5", folders=False)[:3] if is_dir else [path]
        results.append(any(_h5_holds(c, fp["h5_paths"]) for c in candidates))

    if "h5_attrs" in fp and not quick:
        ok = False
        if not is_dir and path.suffix in _H5_SUFFIXES:
            from bexa.io.h5files import open_h5

            try:
                with open_h5(path, retry=False) as f:
                    ok = all(
                        safe_format(group, scan=1) in f
                        and all(a in f[safe_format(group, scan=1)].attrs for a in attrs)
                        for group, attrs in fp["h5_attrs"].items()
                    )
            except OSError:
                ok = False
        results.append(ok)
    return results


def _specificity(fingerprint: dict[str, Any]) -> int:
    """Number of individual fingerprint items (every list entry counts one)."""
    return sum(len(v) if isinstance(v, list) else 1 for v in fingerprint.values())


def match_spec(
    path: str | Path,
    specs: Iterable[FormatSpec] | None = None,
    quick: bool = False,
    listing: FolderListing | None = None,
) -> list[tuple[float, FormatSpec]]:
    """Score every spec against ``path``; best first.

    Returns ``(score, spec)`` pairs where ``score`` is the fraction of fingerprint
    checks that passed. Specs without a fingerprint and abstract specs (bases for
    ``extends``) are never matched. Ties go to the more detailed fingerprint, then
    to the newest name (dated specs sort by year). ``quick`` opens no HDF5 file:
    only the file-system part of each fingerprint counts (file and folder
    names, the opening lines of a text file), and a spec with nothing else to
    check is left out. ``listing`` keeps the folder listings between calls (a
    survey of a tree passes one, so every folder is listed once).
    """
    path = Path(path)
    if specs is None:
        specs = all_specs()
    listing = listing if listing is not None else FolderListing()
    is_dir = path.is_dir()
    scored: list[tuple[float, int, FormatSpec]] = []
    for spec in specs:
        if spec.abstract:
            continue
        checks = _fingerprint_checks(spec, path, is_dir, listing, quick=quick)
        if not checks:
            continue
        scored.append((sum(checks) / len(checks), _specificity(spec.fingerprint), spec))
    scored.sort(key=lambda item: (item[0], item[1], item[2].name), reverse=True)
    return [(score, spec) for score, _, spec in scored]


def best_spec(path: str | Path) -> FormatSpec | None:
    """The spec whose fingerprint fully matches ``path``, or ``None``."""
    scored = match_spec(path)
    if scored and scored[0][0] >= 1.0:
        return scored[0][1]
    return None


# ------------------------------------------------------------------ drafting
def _walk_h5(path: Path, max_items: int = 5000) -> list[tuple[str, tuple[int, ...], str]]:
    """(path, shape, dtype) of every dataset in an HDF5 file."""
    import h5py

    found: list[tuple[str, tuple[int, ...], str]] = []

    def visit(name: str, obj: Any) -> Any:
        if isinstance(obj, h5py.Dataset):
            found.append((name, tuple(obj.shape), str(obj.dtype)))
        if len(found) >= max_items:
            return True
        return None

    with h5py.File(path, "r") as f:
        f.visititems(visit)
    return found


def draft_spec(path: str | Path, name: str | None = None) -> dict[str, Any]:
    """Guess a spec for an unknown layout; ``TODO`` marks what a human must fill in.

    The guess is based on the HDF5 files found at ``path`` (a file or a folder):
    the largest 3-D datasets are frame candidates, 1-D datasets whose length
    equals a frame count are motor candidates, and the group names suggest
    which engine walks the structure.
    """
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.h5")) + sorted(path.glob("*/*.h5"))
    else:
        files = [path]
    files = [f for f in files if f.suffix in _H5_SUFFIXES][:20]
    if not files:
        raise FileNotFoundError(f"no HDF5 files found at {path}")

    frames: list[tuple[str, tuple[int, ...], str]] = []
    motors: list[tuple[str, tuple[int, ...], str]] = []
    scalars: list[str] = []
    top_keys: set[str] = set()
    for f in files:
        import h5py

        with h5py.File(f, "r") as h5:
            top_keys.update(h5.keys())
        for dpath, shape, dtype in _walk_h5(f):
            if len(shape) == 3 and min(shape[1:]) >= 16:
                frames.append((f"{f.name}::{dpath}", shape, dtype))
            elif len(shape) == 1 and shape[0] > 1:
                motors.append((f"{f.name}::{dpath}", shape, dtype))
            elif len(shape) == 0:
                scalars.append(f"{f.name}::{dpath}")
    frames.sort(key=lambda item: -int(item[1][0] * item[1][1] * item[1][2]))
    # a scan split over several files: motors match the total frame count
    frame_counts = {item[1][0] for item in frames}
    totals: dict[str, int] = {}
    for fpath, shape, _dtype in frames:
        inner = fpath.split("::", 1)[1]
        totals[inner] = totals.get(inner, 0) + int(shape[0])
    frame_counts |= set(totals.values())
    motor_candidates = [m[0] for m in motors if m[1][0] in frame_counts][:20]

    if any(re.fullmatch(r"\d+\.1", k) for k in top_keys):
        engine = "hdf5_stack"
    elif "run" in top_keys:
        engine = "spec_h5"
    elif any("block0_values" in item[0] for item in frames) or "measurements" in top_keys:
        engine = "hdf5_points"
    else:
        engine = "hdf5_stack"

    stem = name or f"TODO_{path.stem}"
    return {
        "name": stem,
        "engine": engine,
        "description": f"TODO: describe the layout found at {path}",
        "fingerprint": {"h5_paths": [frames[0][0].split("::", 1)[1]] if frames else ["TODO"]},
        "layout": {
            "frames": frames[0][0].split("::", 1)[1] if frames else "TODO",
            "frame_candidates": [f"{p} shape={s} dtype={d}" for p, s, d in frames[:5]],
        },
        "motors": {
            "known": [c.split("::", 1)[1].rsplit("/", 1)[-1] for c in motor_candidates],
            "candidates": motor_candidates,
        },
        "keys": {},
        "energy": {"from": "TODO: motor or scalar holding the energy", "crystal": "Si111"},
        "detectors": {
            "TODO_detector": {
                "pixel_size_um": None,
                "shape": list(frames[0][1][1:]) if frames else None,
            }
        },
        "convention": "TODO: id03, pal_fourc, ...",
        "scalars_seen": scalars[:30],
    }


def write_draft(path: str | Path, out: str | Path, name: str | None = None) -> Path:
    """Write :func:`draft_spec` as YAML to ``out`` and return the path."""
    draft = draft_spec(path, name=name)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(draft, sort_keys=False, allow_unicode=True), encoding="utf-8")
    log.info("draft spec written to %s (fill in the TODO entries)", out)
    return out
