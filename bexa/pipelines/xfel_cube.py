"""Per-point reduction of PAL-XFEL point files to a laser on/off cube.

Port of ``Aaron_allscan_cube_parallel.py``: every motor point becomes one mean
frame for laser on and one for laser off, plus the median of each motor
readback and the mean ROI/I0 signal. Points are reduced in spawned processes
with BLAS pinned to one thread, then assembled in acquisition order into the
cube dataset of :mod:`bexa.io.cube` (``frames(laser, <axis>, y, x)``).
:func:`save_legacy` writes the same numbers in the old ``runN.h5`` layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import xarray as xr

from bexa._log import get_logger, progress
from bexa.config import BeamtimeProfile, load_profile
from bexa.core.parallel import process_pool
from bexa.core.provenance import build_attrs
from bexa.core.roi import ROI
from bexa.io.cube import save
from bexa.io.engines.hdf5_points import Hdf5PointsSource
from bexa.io.formats import FormatSpec, load_spec

log = get_logger(__name__)

SCALAR_KEYS = ("delay", "phi", "chi", "th", "tth", "laser_v", "laser_h")
SIGNAL_KEYS = ("laser_flag", "i0_upstream", "i0", "roi_stat", "beam_x", "beam_y")
LEGACY_NAMES = {"delay": "delays"}
LASER_COORD = np.array(["off", "on"])
AXIS_TOLERANCE = 1e-9


@dataclass
class PointKeys:
    """Column names of one beamtime, resolved from the spec aliases."""

    laser: str | None
    i0: str | None
    roi_stat: str | None
    signal_columns: list[str]
    scalars: dict[str, str | None]

    @classmethod
    def from_spec(cls, spec: FormatSpec, columns: Any) -> PointKeys:
        available = list(columns)
        resolved = {key: spec.resolve_key(key, available) for key in SIGNAL_KEYS}
        signal_columns = [c for c in resolved.values() if c is not None]
        return cls(
            laser=resolved["laser_flag"],
            i0=resolved["i0"],
            roi_stat=resolved["roi_stat"],
            signal_columns=signal_columns,
            scalars={key: spec.resolve_key(key, available) for key in SCALAR_KEYS},
        )


@dataclass
class PointResult:
    """What one motor point reduces to."""

    number: int
    scalars: dict[str, float]
    signal_on: float
    signal_off: float
    frame_on: np.ndarray
    frame_off: np.ndarray
    n_on: int
    n_off: int
    n_shots: int
    extra: dict[str, Any] = field(default_factory=dict)


def _median(table: Any, column: str | None) -> float:
    """Median of a column, NaN when the column is missing (legacy ``_median_safe``)."""
    if column is None or column not in table.columns:
        return float("nan")
    value = table[column].median()
    return float(value) if value == value else float("nan")


def reduce_point(
    table: Any,
    frames: np.ndarray,
    keys: PointKeys,
    roi: ROI | None = None,
    clip: float | None = 1e-4,
    number: int = 0,
) -> PointResult:
    """Reduce the shots of one point exactly as the legacy ``_process_one`` did.

    Scalars are column medians; the signal is the mean of ``roi_stat / i0`` over
    the shots of each laser state after dropping rows with a NaN in any signal
    column; frames are cut to the ROI, non-positive pixels set to ``clip``,
    shots with a NaN laser flag dropped, and the remaining shots averaged per
    laser state (NaN frame when a state has no shots).
    """
    scalars = {key: _median(table, column) for key, column in keys.scalars.items()}

    signal_on = signal_off = float("nan")
    if keys.roi_stat and keys.i0 and keys.laser:
        subset = table[keys.signal_columns]
        laser = table[keys.laser]
        rows_on = subset[laser == 1].dropna()
        rows_off = subset[laser == 0].dropna()
        on_mean = (rows_on[keys.roi_stat] / rows_on[keys.i0]).mean()
        off_mean = (rows_off[keys.roi_stat] / rows_off[keys.i0]).mean()
        signal_on = float(on_mean) if on_mean == on_mean else float("nan")
        signal_off = float(off_mean) if off_mean == off_mean else float("nan")

    raw = np.asarray(frames)
    if roi is not None:
        ys, xs = roi.pixel_slices(raw.shape[1:])
        raw = raw[:, ys, xs]
    raw = np.array(raw)  # own the memory before clipping in place
    if clip is not None:
        raw[raw <= 0] = clip

    if keys.laser is not None:
        flags = np.asarray(table[keys.laser].to_numpy(), dtype=float)
    else:
        flags = np.zeros(len(table))
    keep = ~np.isnan(flags)
    n = min(raw.shape[0], keep.shape[0])
    raw_kept = raw[:n][keep[:n]]
    laser_kept = flags[:n][keep[:n]]
    shape = raw.shape[1:]
    if raw_kept.shape[0] == 0:
        frame_on = np.full(shape, np.nan, dtype=np.float64)
        frame_off = np.full(shape, np.nan, dtype=np.float64)
        n_on = n_off = 0
    else:
        on_sel = laser_kept == 1
        off_sel = laser_kept == 0
        n_on, n_off = int(on_sel.sum()), int(off_sel.sum())
        frame_on = (
            raw_kept[on_sel].mean(axis=0) if n_on else np.full(shape, np.nan, dtype=np.float64)
        )
        frame_off = (
            raw_kept[off_sel].mean(axis=0) if n_off else np.full(shape, np.nan, dtype=np.float64)
        )
    return PointResult(
        number, scalars, signal_on, signal_off, frame_on, frame_off, n_on, n_off, int(raw.shape[0])
    )


def _reduce_point_job(job: dict[str, Any]) -> PointResult:
    """Worker: read one point's table and raw frames, then :func:`reduce_point`."""
    import pandas as pd

    table = pd.read_hdf(job["table"], key=job["table_key"])
    with h5py.File(job["raw"], "r") as f:
        frames = f[job["frames_path"]][...]
    return reduce_point(table, frames, job["keys"], job["roi"], job["clip"], job["number"])


def detect_axis(results: list[PointResult], priority: list[str]) -> str:
    """First scalar of ``priority`` that is finite everywhere and varies, else ``"index"``."""
    for key in priority:
        if key == "index":
            break
        values = np.array([r.scalars.get(key, np.nan) for r in results], dtype=float)
        if values.size and np.isfinite(values).all() and np.ptp(values) > AXIS_TOLERANCE:
            return key
    return "index"


def assemble(
    results: list[PointResult],
    axis: str | None = None,
    priority: list[str] | None = None,
    attrs: dict[str, Any] | None = None,
) -> xr.Dataset:
    """Stack point results into the cube dataset."""
    results = sorted(results, key=lambda r: r.number)
    n = len(results)
    dim = axis or detect_axis(results, priority or [*SCALAR_KEYS, "index"])
    scalars = {
        key: np.array([r.scalars.get(key, np.nan) for r in results], dtype=float)
        for key in SCALAR_KEYS
    }
    index = np.linspace(0, n - 1, n) if n else np.zeros(0)
    axis_values = index if dim == "index" else scalars[dim]
    coords: dict[str, Any] = {
        "laser": LASER_COORD,
        dim: axis_values,
        "point": (dim, np.array([r.number for r in results])),
    }
    if dim != "index":
        coords["index"] = (dim, index)
    for key, values in scalars.items():
        if key != dim and np.isfinite(values).any():
            coords[key] = (dim, values)
    frames = xr.DataArray(
        np.stack(
            [np.stack([r.frame_off for r in results]), np.stack([r.frame_on for r in results])]
        ),
        dims=("laser", dim, "y", "x"),
        name="frames",
    )
    signal = xr.DataArray(
        np.array([[r.signal_off for r in results], [r.signal_on for r in results]]),
        dims=("laser", dim),
        name="signal",
        attrs={"description": "mean of roi_stat / i0 over the shots of each laser state"},
    )
    n_shots = xr.DataArray(
        np.array([[r.n_off for r in results], [r.n_on for r in results]]),
        dims=("laser", dim),
        name="n_shots",
    )
    ds = xr.Dataset({"frames": frames, "signal": signal, "n_shots": n_shots}, coords=coords)
    ds.attrs.update(attrs or {})
    ds.attrs["scan_axis"] = dim
    return ds


def build_cube(
    source: Hdf5PointsSource,
    roi: ROI | None = None,
    workers: int | None = None,
    clip: float | None = None,
    axis: str | None = None,
    show_progress: bool = False,
) -> xr.Dataset:
    """Reduce every point of a run (in processes) and assemble the cube."""
    spec = source.spec
    keys = PointKeys.from_spec(spec, source.table(0).columns)
    if clip is None:
        clip = spec.options.get("clip_nonpositive", 1e-4)
    jobs = [
        {
            "number": p.number,
            "table": p.table,
            "raw": p.raw,
            "table_key": source.table_key,
            "frames_path": source.frames_path,
            "keys": keys,
            "roi": roi,
            "clip": clip,
        }
        for p in source.points
    ]
    if (workers is not None and workers <= 1) or len(jobs) == 1:
        results = [
            _reduce_point_job(j) for j in progress(jobs, desc="points", enabled=show_progress)
        ]
    else:
        with process_pool(workers) as pool:
            results = list(
                progress(
                    pool.map(_reduce_point_job, jobs),
                    total=len(jobs),
                    desc="points",
                    enabled=show_progress,
                )
            )
    attrs = build_attrs(
        source_files=source.files(),
        parameters={
            "run": source.run,
            "scan": source.scan,
            "detector": source.detector,
            "roi": roi.to_dict() if roi is not None else None,
            "clip": clip,
            "keys": {
                "laser": keys.laser,
                "i0": keys.i0,
                "roi_stat": keys.roi_stat,
                **{k: v for k, v in keys.scalars.items() if v},
            },
        },
    )
    attrs.update({"run": source.run, "scan": source.scan, "format": spec.name})
    if source.energy_keV is not None:
        attrs["energy_keV"] = source.energy_keV
    return assemble(results, axis=axis, priority=list(spec.axis_priority), attrs=attrs)


# ------------------------------------------------------------- profile route
def cube_path(profile: BeamtimeProfile, run: int | str) -> Path:
    """Where the cube of ``run`` lives: ``<processed_root>/data/runN.h5``."""
    if profile.processed_root is None:
        raise ValueError(f"profile {profile.name!r} has no processed_root")
    name = run if isinstance(run, str) else f"run{run}"
    return Path(profile.processed_root) / "data" / f"{name}.h5"


def open_run(
    profile: str | Path | BeamtimeProfile, run: int, scan: int = 1, **kwargs: Any
) -> Hdf5PointsSource:
    """The point-file source of ``run`` described by a beamtime profile."""
    prof = profile if isinstance(profile, BeamtimeProfile) else load_profile(profile)
    spec = load_spec(prof.format, overrides=prof.format_overrides or None)
    if spec.engine != "hdf5_points":
        raise ValueError(f"profile {prof.name!r} uses engine {spec.engine!r}, not hdf5_points")
    kwargs.setdefault("detector", prof.detector)
    if prof.geometry.energy_keV is not None:
        kwargs.setdefault("energy_keV", prof.geometry.energy_keV)
    return Hdf5PointsSource(spec, prof.data_root(), run, scan, **kwargs)


def reduce_run(
    profile: str | Path | BeamtimeProfile,
    run: int,
    scan: int = 1,
    roi: ROI | None = None,
    workers: int | None = None,
    out: str | Path | None = None,
    legacy: bool = False,
    show_progress: bool = False,
    **kwargs: Any,
) -> xr.Dataset:
    """Build the cube of ``run`` and save it (bexa format, or the legacy layout with ``legacy``)."""
    prof = profile if isinstance(profile, BeamtimeProfile) else load_profile(profile)
    source = open_run(prof, run, scan)
    try:
        ds = build_cube(source, roi=roi, workers=workers, show_progress=show_progress, **kwargs)
    finally:
        source.close()
    path = Path(out) if out is not None else cube_path(prof, run)
    if legacy:
        save_legacy(ds, path)
    else:
        save(ds, path)
    log.info("cube of run %d saved to %s", run, path)
    return ds


# ------------------------------------------------------------- legacy layout
def to_legacy_arrays(ds: xr.Dataset) -> dict[str, np.ndarray]:
    """The datasets of a legacy ``runN.h5`` (points whose scalar is NaN are dropped, as before)."""
    dim = str(ds.attrs.get("scan_axis", "index"))
    n = ds.sizes[dim]
    out: dict[str, np.ndarray] = {}
    for key in SCALAR_KEYS:
        name = LEGACY_NAMES.get(key, key)
        if key == dim:
            values = np.asarray(ds[dim].values, dtype=float)
        elif key in ds.coords:
            values = np.asarray(ds.coords[key].values, dtype=float)
        else:
            values = np.full(n, np.nan)
        out[name] = values[np.isfinite(values)]
    out["index"] = np.linspace(0, n - 1, n) if n else np.zeros(0)
    out["signals_on"] = np.asarray(ds["signal"].sel(laser="on").values, dtype=float)
    out["signals_off"] = np.asarray(ds["signal"].sel(laser="off").values, dtype=float)
    out["jungfrau_on"] = np.asarray(ds["frames"].sel(laser="on").values)
    out["jungfrau_off"] = np.asarray(ds["frames"].sel(laser="off").values)
    return out


def save_legacy(ds: xr.Dataset, path: str | Path) -> Path:
    """Write the cube in the layout of ``Aaron_allscan_cube_parallel.py``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for name, data in to_legacy_arrays(ds).items():
            f.create_dataset(name, data=data)
    return path
