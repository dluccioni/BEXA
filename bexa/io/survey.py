"""Every dataset under a folder, and the format spec that recognises each one.

:func:`survey` walks a data tree (a ``RAW_DATA`` folder, a copy on the lab
share, a folder of beamline examples) and reports what it finds: the folders a
spec recognises as a whole (a BLISS dataset, a SPEC session, a PAL-XFEL run)
and the data files a spec recognises one by one (one HDF5 file per scan, saved
cubes). A recognised folder is not entered, the scan folders of a known layout
are skipped, and of the files that share a naming pattern only the first is
opened. Data no spec recognises is reported with the spec that came closest,
and :func:`write_drafts` drafts one format spec per unknown HDF5 layout rather
than one per dataset (``bexa profile sniff --recursive``). The scan picker lists
its dataset folders with the same walk.
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from bexa._log import get_logger
from bexa.io.formats import (
    FolderListing,
    FormatSpec,
    _walk_h5,
    all_specs,
    draft_spec,
    match_spec,
)

if TYPE_CHECKING:
    import pandas as pd

log = get_logger(__name__)

__all__ = [
    "DATA_SUFFIXES",
    "HDF5_SUFFIXES",
    "SKIPPED_FOLDERS",
    "SurveyEntry",
    "describe_survey",
    "survey",
    "survey_table",
    "write_drafts",
]

HDF5_SUFFIXES = (".h5", ".hdf5", ".nxs", ".nx")
# the files data is made of: a folder holding none of them only matters for its subfolders
DATA_SUFFIXES = (*HDF5_SUFFIXES, ".spec", ".tif", ".tiff", ".npy", ".npz", ".edf", ".cbf")
SKIPPED_FOLDERS = frozenset({"bexa_cache", "__pycache__"})
DIGITS = re.compile(r"\d+")


@dataclass
class SurveyEntry:
    """A dataset :func:`survey` found, or data that no format spec recognises.

    ``path`` is a dataset folder (``kind="folder"``) or the first of the data
    files that share one naming pattern (``kind="files"``: ``count`` files,
    ``pattern`` their name with every run of digits as ``#``). ``spec`` and
    ``engine`` say what recognised it; a quick survey, which opens no HDF5
    file, names the spec only when one alone fits. ``closest`` names the spec
    that came nearest for data nothing recognised, with its ``score``.
    ``opens_as_dataset`` is True for a folder of numbered scans, the kind
    :func:`bexa.open_dataset` and the scan picker open.
    """

    path: Path
    kind: str
    recognised: bool
    spec: str | None = None
    engine: str | None = None
    score: float = 0.0
    closest: str | None = None
    count: int = 1
    pattern: str | None = None
    opens_as_dataset: bool = False


def survey(
    root: str | Path,
    depth: int = 4,
    quick: bool = False,
    specs: Sequence[FormatSpec] | None = None,
) -> list[SurveyEntry]:
    """Every dataset under ``root``, up to ``depth`` folder levels down, and what recognises it.

    ``quick=True`` opens no HDF5 file: a folder is recognised when the
    file-system part of a spec's fingerprint matches (a master file beside
    ``scanNNNN`` folders, the opening lines of a SPEC file), and data files are
    not checked one by one; the scan picker lists folders this way. ``specs``
    limits the specs tried (default: every spec of
    :func:`bexa.io.formats.all_specs`). Hidden folders, ``bexa_cache`` and
    ``__pycache__`` are skipped. A file given as ``root`` is checked alone; a
    missing root raises ``FileNotFoundError``.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"no file or folder {root}")
    tried = [
        s for s in (all_specs() if specs is None else specs) if s.fingerprint and not s.abstract
    ]
    walker = _Walker(tried, int(depth), quick)
    if root.is_file():
        walker.check_files([root], None)
    else:
        walker.visit(root, 0)
    return walker.entries


class _Walker:
    """One survey: the specs tried, the folder listings, the entries found."""

    def __init__(self, specs: list[FormatSpec], depth: int, quick: bool) -> None:
        self.specs = specs
        self.depth = depth
        self.quick = quick
        self.listing = FolderListing()
        self.entries: list[SurveyEntry] = []
        globs = {s.fingerprint.get("scan_dir_glob") for s in specs}
        self.scan_globs = sorted(str(g) for g in globs if g)

    def skipped(self, name: str) -> bool:
        """Hidden folders, bexa's own, and the scan folders of a known layout."""
        if name.startswith(".") or name in SKIPPED_FOLDERS:
            return True
        return any(fnmatch.fnmatch(name, g) for g in self.scan_globs)

    def visit(self, folder: Path, level: int) -> None:
        scored = match_spec(folder, self.specs, quick=self.quick, listing=self.listing)
        if scored and scored[0][0] >= 1.0:
            self.entries.append(self.recognised(folder, "folder", scored))
            return
        entries = sorted(self.listing.entries(folder).items())
        files = [
            folder / name
            for name, (is_dir, _) in entries
            if not is_dir and Path(name).suffix.lower() in DATA_SUFFIXES
        ]
        if files and not self.quick:
            self.check_files(files, scored[0] if scored else None)
        if level >= self.depth:
            return
        for name, (is_dir, is_link) in entries:
            # a linked folder is not followed: a link back up the tree would never end
            if is_dir and not is_link and not self.skipped(name):
                self.visit(folder / name, level + 1)

    def check_files(self, files: Sequence[Path], near: tuple[float, FormatSpec] | None) -> None:
        """One entry per naming pattern; the first file of a pattern stands for all of them."""
        groups: dict[str, list[Path]] = {}
        for path in files:
            groups.setdefault(_name_pattern(path.name), []).append(path)
        for pattern, members in groups.items():
            first = members[0]
            hdf5 = first.suffix.lower() in HDF5_SUFFIXES
            scored = match_spec(first, self.specs, listing=self.listing) if hdf5 else []
            if scored and scored[0][0] >= 1.0:
                entry = self.recognised(first, "files", scored)
            else:
                closest = scored[0] if scored and scored[0][0] > 0 else near
                entry = SurveyEntry(first, "files", False)
                if closest is not None and closest[0] > 0:
                    entry.closest, entry.score = closest[1].name, float(closest[0])
            entry.count, entry.pattern = len(members), pattern
            self.entries.append(entry)

    def recognised(
        self, path: Path, kind: str, scored: list[tuple[float, FormatSpec]]
    ) -> SurveyEntry:
        fitting = [spec for score, spec in scored if score >= 1.0]
        best = fitting[0]  # the most detailed fingerprint, then the newest name
        named = not self.quick or len(fitting) == 1
        return SurveyEntry(
            path,
            kind,
            True,
            spec=best.name if named else None,
            engine=best.engine,
            score=1.0,
            opens_as_dataset=kind == "folder" and _opens_as_dataset(best.engine),
        )


def _name_pattern(name: str) -> str:
    """A file name with its runs of digits as ``#``, the suffix kept: ``p0012.h5``, ``p#.h5``."""
    path = Path(name)
    return DIGITS.sub("#", path.stem) + path.suffix


def _opens_as_dataset(engine: str) -> bool:
    """True for an engine whose folders hold numbered scans (``dataset_layout``)."""
    from bexa.core.registry import engines

    try:
        return bool(getattr(engines.get(engine), "dataset_layout", False))
    except KeyError:  # an engine whose optional package is missing here
        return False


# ------------------------------------------------------------------- output
def survey_table(entries: Iterable[SurveyEntry], root: str | Path | None = None) -> pd.DataFrame:
    """The entries as a table, paths relative to ``root`` when given."""
    import pandas as pd

    rows = []
    for entry in entries:
        row: dict[str, Any] = asdict(entry)
        row["path"] = _shown(entry.path, root)
        rows.append(row)
    return pd.DataFrame(rows, columns=[f.name for f in fields(SurveyEntry)])


def describe_survey(entries: Sequence[SurveyEntry], root: str | Path | None = None) -> str:
    """The entries as text: what is recognised, by which spec, and what is not."""
    found = [e for e in entries if e.recognised]
    unknown = [e for e in entries if not e.recognised]
    where = f" under {root}" if root is not None else ""
    lines = [f"{len(found)} recognised{where}:"]
    for e in found:
        what = e.spec or f"a {e.engine} layout"
        note = "opens with open_dataset and the scan picker" if e.opens_as_dataset else ""
        lines.append(f"  {_label(e, root):<60} {what:<26} {note}".rstrip())
    if unknown:
        lines.append(f"{len(unknown)} not recognised:")
        for e in unknown:
            near = f"closest {e.closest} ({e.score:.0%})" if e.closest else "no spec comes close"
            lines.append(f"  {_label(e, root):<60} {near}")
    return "\n".join(lines)


def _shown(path: Path, root: str | Path | None) -> str:
    if root is not None:
        try:
            return path.relative_to(Path(root)).as_posix() or "."
        except ValueError:
            pass
    return str(path)


def _label(entry: SurveyEntry, root: str | Path | None) -> str:
    shown = _shown(entry.path, root)
    if entry.kind == "files" and entry.count > 1:
        folder = _shown(entry.path.parent, root)
        return f"{folder}/{entry.pattern} ({entry.count} files)"
    return shown


# ------------------------------------------------------------------- drafts
def write_drafts(entries: Iterable[SurveyEntry], out_dir: str | Path) -> dict[Path, list[Path]]:
    """Draft one format spec per HDF5 layout that no spec recognises; ``{draft: paths}``.

    The unrecognised HDF5 data is grouped by layout: the names of its top-level
    groups and of its 3-D datasets, digits ignored. Each group gets one draft,
    made by :func:`bexa.io.formats.draft_spec` from the folder of its first
    entry and written to ``<out_dir>/TODO_<folder>.yaml`` with the paths it
    covers under ``seen_in``; a draft of that name from an earlier run is
    replaced. Data in other formats (TIFF, NumPy, SPEC) gets no draft:
    drafting reads HDF5 only.
    """
    groups: dict[tuple[str, ...], list[SurveyEntry]] = {}
    for entry in entries:
        if entry.recognised or entry.path.suffix.lower() not in HDF5_SUFFIXES:
            continue
        groups.setdefault(_layout_signature(entry.path), []).append(entry)
    out_dir = Path(out_dir)
    written: dict[Path, list[Path]] = {}
    for members in groups.values():
        draft = draft_spec(members[0].path.parent)
        draft["seen_in"] = [str(m.path) for m in members]
        target = out_dir / f"{draft['name']}.yaml"
        n = 2
        while target in written:  # two layouts found in folders of the same name
            target = out_dir / f"{draft['name']}_{n}.yaml"
            n += 1
        draft["name"] = target.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        target.write_text(
            yaml.safe_dump(draft, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        written[target] = [m.path for m in members]
        log.info("draft spec %s covers %d data paths", target, len(members))
    return written


def _layout_signature(path: Path) -> tuple[str, ...]:
    """The layout of an HDF5 file: its top-level names and its 3-D datasets, digits as ``#``."""
    import h5py

    try:
        with h5py.File(path, "r") as f:
            tops = sorted({DIGITS.sub("#", str(k)) for k in f})
        items = _walk_h5(path, max_items=2000)
    except OSError:
        return ("unreadable", str(path))
    frames = sorted({DIGITS.sub("#", name) for name, shape, _ in items if len(shape) == 3})
    return (*tops, "|", *frames)
