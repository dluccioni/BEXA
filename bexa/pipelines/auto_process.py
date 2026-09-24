"""Beamtime watcher: discover runs, build cubes, write figures and animations.

Port of ``auto_process/scan_auto_process.py`` without ``os.system``: the cube
builder and the figure functions are imported and called directly, every run
gets a log record (CSV line plus a JSON file with timings and errors), a
failed run is retried once and then added to the skip list, and ``watch``
polls the raw folder at a fixed interval. ``dry_run`` only lists what would
be processed.
"""

from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bexa._log import get_logger
from bexa.config import BeamtimeProfile, load_profile
from bexa.core.roi import ROI
from bexa.io.cube import load
from bexa.pipelines import cube_report, xfel_cube

log = get_logger(__name__)

RUN_NAME_RE = re.compile(r"run(\d+)")
LEGACY_MOTORS = ("delays", "phi", "chi", "th", "tth", "laser_v", "laser_h", "index")

__all__ = [
    "AutoConfig",
    "RoiTable",
    "RunRecord",
    "config_from_profile",
    "discover_runs",
    "pending_runs",
    "process_run",
    "processed_runs",
    "run_once",
    "watch",
]


# ------------------------------------------------------------------ ROI table
@dataclass
class RoiTable:
    """Per-run scan axis and named ROIs.

    Two file formats are read: the legacy ``ROI_dict.csv`` (one column per run,
    first row the axis name, then ``r0 r1 c0 c1`` quadruples) and a YAML mapping
    ``run42: {axis: delays, rois: {peak: [r0, r1, c0, c1]}}``.
    """

    entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    path: Path | None = None

    @classmethod
    def load(cls, path: str | Path | None) -> RoiTable:
        if path is None:
            return cls()
        path = Path(path)
        if not path.exists():
            log.info("ROI table %s does not exist; using defaults", path)
            return cls(path=path)
        if path.suffix in (".yaml", ".yml"):
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            entries = {str(k): dict(v or {}) for k, v in raw.items()}
            return cls(entries, path)
        return cls(cls._read_legacy_csv(path), path)

    @staticmethod
    def _read_legacy_csv(path: Path) -> dict[str, dict[str, Any]]:
        import pandas as pd

        df = pd.read_csv(path)
        entries: dict[str, dict[str, Any]] = {}
        for run in df.columns:
            column = df[run].tolist()
            if not column:
                continue
            axis = column[0]
            axis = None if (isinstance(axis, float) and np.isnan(axis)) else str(axis)
            numbers = [float(v) for v in column[1:] if not (isinstance(v, float) and np.isnan(v))]
            rois = {}
            for i in range(len(numbers) // 4):
                r0, r1, c0, c1 = (int(v) for v in numbers[4 * i : 4 * i + 4])
                rois[f"roi{i + 1}"] = [r0, r1, c0, c1]
            entries[str(run)] = {"axis": axis, "rois": rois}
        return entries

    def axis(self, run: str) -> str | None:
        axis = self.entries.get(run, {}).get("axis")
        return None if axis in (None, "", "index") else str(axis)

    def rois(self, run: str) -> list[tuple[str, ROI]]:
        out = []
        for name, box in (self.entries.get(run, {}).get("rois") or {}).items():
            r0, r1, c0, c1 = (int(v) for v in box)
            out.append((str(name), ROI(y=(r0, r1), x=(c0, c1))))
        return out

    def set_axis(self, run: str, axis: str) -> None:
        self.entries.setdefault(run, {})["axis"] = axis

    def save(self, path: str | Path | None = None) -> Path | None:
        """Write the table as YAML (the legacy CSV is read-only)."""
        target = Path(path) if path is not None else self.path
        if target is None:
            return None
        if target.suffix not in (".yaml", ".yml"):
            target = target.with_suffix(".yaml")
        target.write_text(yaml.safe_dump(self.entries, sort_keys=True), encoding="utf-8")
        return target


# ------------------------------------------------------------------ config
@dataclass
class AutoConfig:
    """Folders and settings of one watcher session."""

    raw_dir: Path
    cube_dir: Path
    outputs_dir: Path
    profile: BeamtimeProfile | None = None
    roi_table: RoiTable = field(default_factory=RoiTable)
    skip_list: Path | None = None
    scan: int = 1
    workers: int | None = None
    legacy_cubes: bool = False
    background_box: Any = None
    default_rois: list[tuple[str, ROI]] = field(default_factory=list)
    animations: bool = True
    fps: int = 5

    @property
    def log_dir(self) -> Path:
        return self.outputs_dir / "logs"


def config_from_profile(
    profile: str | Path | BeamtimeProfile,
    raw_dir: str | Path | None = None,
    cube_dir: str | Path | None = None,
    outputs_dir: str | Path | None = None,
    roi_table: str | Path | None = None,
    skip_list: str | Path | None = None,
    **kwargs: Any,
) -> AutoConfig:
    """Build the watcher config from a beamtime profile (explicit paths win)."""
    prof = profile if isinstance(profile, BeamtimeProfile) else load_profile(profile)
    raw_root = prof.data_root()
    processed = Path(prof.processed_root) if prof.processed_root else raw_root.parent / "reduced"
    auto = prof.auto_process
    outputs = outputs_dir or (auto.outputs if auto and auto.outputs else processed / "outputs")
    roi_path = roi_table or (auto.roi_table if auto and auto.roi_table else None)
    skip_path = skip_list or (auto.skip_list if auto and auto.skip_list else None)
    if roi_path is None and (processed / "ROI_dict.csv").exists():
        roi_path = processed / "ROI_dict.csv"
    if skip_path is None:
        skip_path = processed / "Skip_dict.csv"
    default_rois = [(name, ROI.from_dict(box)) for name, box in (prof.rois or {}).items()]
    return AutoConfig(
        raw_dir=Path(raw_dir) if raw_dir else raw_root / "type=raw",
        cube_dir=Path(cube_dir) if cube_dir else processed / "data",
        outputs_dir=_resolve(outputs, processed),
        profile=prof,
        roi_table=RoiTable.load(_resolve(roi_path, processed) if roi_path else None),
        skip_list=_resolve(skip_path, processed),
        background_box=prof.background_box,
        default_rois=default_rois,
        **kwargs,
    )


def _resolve(path: str | Path, base: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else base / p


# ------------------------------------------------------------------ discovery
def discover_runs(raw_dir: str | Path, cube_dir: str | Path | None = None) -> list[str]:
    """``runN`` for every ``run=NNN`` raw folder plus every ``*Combined*`` cube."""
    runs: list[str] = []
    raw_dir = Path(raw_dir)
    if raw_dir.exists():
        for entry in sorted(raw_dir.iterdir()):
            digits = "".join(re.findall(r"\d+", entry.name))
            if entry.is_dir() and digits:
                runs.append(f"run{int(digits)}")
    if cube_dir is not None and Path(cube_dir).exists():
        for path in sorted(Path(cube_dir).glob("*Combined*.h5")):
            runs.append(path.stem)
    return list(dict.fromkeys(runs))


def processed_runs(outputs_dir: str | Path) -> set[str]:
    """Run names that already have an output (the part of a file name before ``_``)."""
    outputs_dir = Path(outputs_dir)
    if not outputs_dir.exists():
        return set()
    return {p.name.split("_")[0] for p in outputs_dir.iterdir() if p.is_file() and "_" in p.name}


def read_skip_list(path: str | Path | None) -> list[str]:
    if path is None or not Path(path).exists():
        return []
    text = Path(path).read_text(encoding="utf-8").splitlines()
    names = []
    for line in text:
        value = line.strip().strip(",")
        if value and value != "0" and not value.isdigit():
            names.append(value)
    return list(dict.fromkeys(names))


def write_skip_list(path: str | Path, runs: list[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("0\n" + "\n".join(dict.fromkeys(runs)) + "\n", encoding="utf-8")
    return path


def pending_runs(cfg: AutoConfig, only: list[str] | None = None) -> list[str]:
    """Runs with raw data but no outputs, minus the skip list (and ``only`` when given)."""
    done = processed_runs(cfg.outputs_dir)
    skipped = set(read_skip_list(cfg.skip_list))
    runs = [
        r for r in discover_runs(cfg.raw_dir, cfg.cube_dir) if r not in done and r not in skipped
    ]
    if only is not None:
        wanted = set(only)
        runs = [r for r in runs if r in wanted]
    return runs


# ------------------------------------------------------------------ processing
@dataclass
class RunRecord:
    run: str
    status: str
    seconds: float
    outputs: dict[str, str] = field(default_factory=dict)
    error: str = ""
    attempts: int = 1


def cube_axis(ds: Any, requested: str | None) -> str:
    """The axis to plot against: the ROI table's, else the cube's varying axis."""
    if requested:
        name = "delay" if requested == "delays" else requested
        if name in ds.dims or name in ds.coords:
            return name
        log.warning("axis %r not in the cube; using %s", requested, ds.attrs.get("scan_axis"))
    return str(ds.attrs.get("scan_axis", "index"))


def _ensure_cube(run: str, cfg: AutoConfig) -> Path:
    cube = cfg.cube_dir / f"{run}.h5"
    if cube.exists():
        return cube
    if "Combined" in run:
        raise FileNotFoundError(f"combined cube {cube} does not exist")
    if cfg.profile is None:
        raise ValueError("a profile is needed to build cubes")
    number = int(RUN_NAME_RE.fullmatch(run).group(1))  # type: ignore[union-attr]
    xfel_cube.reduce_run(
        cfg.profile,
        number,
        scan=cfg.scan,
        workers=cfg.workers,
        out=cube,
        legacy=cfg.legacy_cubes,
    )
    return cube


def _process_once(run: str, cfg: AutoConfig) -> dict[str, str]:
    cube = _ensure_cube(run, cfg)
    ds = load(cube, squeeze_single=False)
    axis = cube_axis(ds, cfg.roi_table.axis(run))
    rois = cfg.roi_table.rois(run) or cfg.default_rois or [("full", None)]
    written: dict[str, str] = {}
    for name, roi in rois:
        figures = cube_report.figure_set(
            ds, cfg.outputs_dir, label=run, roi=roi, axis=axis, background_box=cfg.background_box
        )
        written.update({f"{name}:{k}": str(v) for k, v in figures.items()})
        if cfg.animations:
            gifs = cube_report.animations(
                ds, cfg.outputs_dir, label=run, roi=roi, axis=axis, fps=cfg.fps
            )
            written.update({f"{name}:{k}": str(v) for k, v in gifs.items()})
    cfg.roi_table.set_axis(run, axis)
    return written


def process_run(run: str, cfg: AutoConfig, retries: int = 1) -> RunRecord:
    """Cube, figures and animations for one run; retried once like the legacy watcher."""
    start = time.time()
    error = ""
    for attempt in range(1, retries + 2):
        try:
            outputs = _process_once(run, cfg)
            record = RunRecord(run, "ok", time.time() - start, outputs, attempts=attempt)
            break
        except Exception as exc:  # the watcher must survive one bad run
            error = f"{type(exc).__name__}: {exc}"
            log.warning("%s attempt %d failed: %s", run, attempt, error)
    else:
        record = RunRecord(run, "failed", time.time() - start, error=error, attempts=retries + 1)
    _log_record(record, cfg)
    return record


def _log_record(record: RunRecord, cfg: AutoConfig) -> None:
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cfg.log_dir / "bexa_auto.csv"
    new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new:
            writer.writerow(["time", "run", "status", "seconds", "attempts", "error"])
        writer.writerow(
            [
                time.strftime("%Y-%m-%d %H:%M:%S"),
                record.run,
                record.status,
                f"{record.seconds:.1f}",
                record.attempts,
                record.error,
            ]
        )
    (cfg.log_dir / f"{record.run}.json").write_text(
        json.dumps(record.__dict__, indent=2, default=str), encoding="utf-8"
    )


def run_once(
    cfg: AutoConfig, only: list[str] | None = None, dry_run: bool = False
) -> list[RunRecord]:
    """Process every pending run (or list them with ``dry_run``)."""
    runs = pending_runs(cfg, only)
    log.info("to process: %s", runs or "nothing")
    if dry_run:
        return [RunRecord(r, "pending", 0.0) for r in runs]
    records = []
    for run in runs:
        log.info("processing %s", run)
        records.append(process_run(run, cfg))
    failed = [r.run for r in records if r.status == "failed"]
    if failed and cfg.skip_list is not None:
        write_skip_list(cfg.skip_list, read_skip_list(cfg.skip_list) + failed)
    if cfg.roi_table.path is not None and records:
        cfg.roi_table.save()
    return records


def watch(
    cfg: AutoConfig,
    interval_s: float = 60.0,
    only: list[str] | None = None,
    dry_run: bool = False,
    max_cycles: int | None = None,
) -> list[RunRecord]:
    """Poll the raw folder every ``interval_s`` seconds and process what appears."""
    records: list[RunRecord] = []
    cycle = 0
    while True:
        records.extend(run_once(cfg, only, dry_run))
        cycle += 1
        if max_cycles is not None and cycle >= max_cycles:
            return records
        time.sleep(interval_s)
