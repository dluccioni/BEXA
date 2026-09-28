"""Generators of small, valid files for every layout bexa reads.

Each generator reproduces the paths, dtypes, chunking, compression and metadata
of one legacy layout (see PLAN.md, section 8.3) and plants a signal with a
known answer: a Gaussian peak whose rocking-curve centre moves linearly across
the detector, so centre-of-mass maps have an analytic expectation. The
returned objects also carry the float frames and reference results computed
the way the legacy scripts computed them, for parity tests.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.special import erf

from bexa.core.units import energy_to_mono_angle

try:  # ESRF detector files are compressed with bitshuffle+lz4
    import hdf5plugin

    try:  # hdf5plugin 4.2 renamed the lz4 flag
        _ESRF_COMPRESSION: dict[str, Any] = dict(hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
    except TypeError:  # pragma: no cover - older hdf5plugin, as on some cluster images
        _ESRF_COMPRESSION = dict(hdf5plugin.Bitshuffle(nelems=0, lz4=True))
except ImportError:  # pragma: no cover
    _ESRF_COMPRESSION = {"compression": "gzip", "compression_opts": 1}

ESRF_FRAMES_PATH = "entry_0000/ESRF-ID03/{detector}/data"
PAL_FRAMES_PATH = "/detector/eh1/jungfrau2/image/block0_values"


# --------------------------------------------------------------------------- signal
def _peak_frames(
    motor_values: dict[str, np.ndarray],
    frame_shape: tuple[int, int],
    amplitude: float | np.ndarray,
    background: float,
    rng: np.random.Generator,
    noise: float,
    curve: str = "gaussian",
    centers: Mapping[str, float | np.ndarray] | None = None,
    widths: Mapping[str, float | np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Frames whose curve in each motor is a Gaussian (or a blurred step) at a per-pixel centre.

    ``centers`` gives the centre of a motor's curve, one value or an ``(H, W)`` map, and
    ``widths`` its sigma the same way; a peak narrower than half the scan step is widened to
    that and lowered to keep its area, as a continuous scan integrates a narrow curve into
    one step, so coarse scans still sample every curve. Without them the centre of the first
    motor varies linearly along x, the second along y, with a sigma of 12 % of the range.
    ``curve="edge"`` gives an error-function step instead of a peak, as a knife-edge or
    sample-height scan does. Returns the float frames ``(n, H, W)`` and the centre maps.
    """
    names = list(motor_values)
    n = len(next(iter(motor_values.values())))
    H, W = frame_shape
    yy, xx = np.mgrid[0:H, 0:W]
    truth: dict[str, np.ndarray] = {}
    weight = np.ones((n, H, W), dtype=np.float64)
    for i, name in enumerate(names):
        values = np.asarray(motor_values[name], dtype=float)
        lo, hi = float(values.min()), float(values.max())
        span = hi - lo if hi > lo else 1.0
        unique = np.unique(values)
        step = float(np.diff(unique).min()) if len(unique) > 1 else span
        if centers is not None and name in centers:
            centre = np.array(np.broadcast_to(np.asarray(centers[name], dtype=float), (H, W)))
        else:  # keep centres inside the inner 60% of the scanned range so the curves are sampled
            frac = (xx / max(W - 1, 1)) if i % 2 == 0 else (yy / max(H - 1, 1))
            centre = lo + span * (0.2 + 0.6 * frac)
        if widths is not None and name in widths:
            given = np.asarray(widths[name], dtype=float)
            sigma = np.maximum(given, 0.5 * step)
            if curve != "edge":  # a step integrates a narrower peak: same area, lower and wider
                weight *= given / sigma
        else:
            sigma = np.asarray(0.12 * span)
        truth[f"{name}_center"] = centre
        offset = (values[:, None, None] - centre[None]) / sigma
        if curve == "edge":
            weight *= 0.5 * (1.0 + erf(offset / np.sqrt(2.0)))
        else:
            weight *= np.exp(-0.5 * offset**2)
    envelope = np.exp(-0.5 * (((yy - H / 2) / (0.45 * H)) ** 2 + ((xx - W / 2) / (0.45 * W)) ** 2))
    frames = background + np.asarray(amplitude, dtype=float) * weight * envelope[None]
    if noise > 0:
        frames = frames + rng.normal(0.0, noise, size=frames.shape)
    return frames.astype(np.float32), truth


# ---------------------------------------------------------------------------- ESRF
@dataclass
class SyntheticEsrfScan:
    """What :func:`make_esrf_scan` wrote, plus the planted truth."""

    root: Path
    dataset: str
    scan: int
    detector: str
    master: Path
    detector_files: list[Path]
    motor_dims: tuple[str, ...]
    motor_shape: tuple[int, ...]
    coords: dict[str, np.ndarray]
    per_frame: dict[str, np.ndarray]
    frame_shape: tuple[int, int]
    n_frames: int
    energy_keV: float
    frames: np.ndarray
    truth: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def dataset_dir(self) -> Path:
        return self.root / self.dataset

    @property
    def scan_folder(self) -> Path:
        return self.dataset_dir / f"scan{self.scan:04d}"


def make_esrf_scan(
    root: str | Path,
    dataset: str = "synth_dfxm",
    scan: int = 1,
    detector: str = "pco_ff",
    motors: tuple[tuple[str, int], ...] = (("chi", 4), ("mu", 12)),
    ranges: dict[str, tuple[float, float]] | None = None,
    frame_shape: tuple[int, int] = (48, 64),
    n_files: int = 2,
    energy_keV: float = 17.0,
    layout: str = "2026",
    order: str = "slow_major",
    partial: int = 0,
    amplitude: float | np.ndarray = 4000.0,
    background: float = 10.0,
    noise: float = 0.0,
    seed: int = 0,
    dtype: Any = np.uint16,
    positioners: dict[str, float] | None = None,
    curve: str = "gaussian",
    centers: Mapping[str, float | np.ndarray] | None = None,
    widths: Mapping[str, float | np.ndarray] | None = None,
) -> SyntheticEsrfScan:
    """Write one ESRF BLISS scan (master file plus detector files).

    Parameters
    ----------
    motors
        ``(name, n_points)`` per scanned motor, slow first. One motor gives an
        fscan1d, two an fscan2d.
    amplitude
        Peak height: one number, or an ``(H, W)`` map (a bright grain on a dim
        background).
    positioners
        Fixed motor values written to the positioners group, for example
        ``{"samz": 0.5}`` for one layer of a z-stack.
    curve
        ``"gaussian"`` (a peak) or ``"edge"`` (a blurred step across the range,
        as a knife-edge or sample-height scan records).
    centers, widths
        Centre and sigma of the curve per motor, one value or an ``(H, W)`` map
        each, as :func:`dfxm_sample` provides (``**sample.scan_kwargs``); without
        them the centres form a smooth gradient across the field of view.
    layout
        ``"2026"``/``"2025"`` write ``fscan_parameters``; ``"2024"`` omits them
        so the grid must be detected from readbacks (and adds ``obpitch``).
    order
        ``"slow_major"`` (fast motor loops inside the slow one) or ``"snake"``
        (fast motor reverses on every other slow step).
    partial
        Number of frames missing at the end, as in an aborted scan.
    """
    rng = np.random.default_rng(seed)
    root = Path(root)
    names = [m[0] for m in motors]
    shape = tuple(m[1] for m in motors)
    default_ranges = {"mu": (-1.0, 1.0), "chi": (-0.5, 0.5), "phi": (-0.3, 0.3), "obpitch": (0, 1)}
    coords = {}
    for name, n in motors:
        lo, hi = (ranges or {}).get(name, default_ranges.get(name, (0.0, 1.0)))
        coords[name] = np.linspace(lo, hi, n)

    # acquisition order: slow-major, with optional snake on the fast axis
    grids = np.meshgrid(*[coords[n] for n in names], indexing="ij")
    per_frame_full = {n: g.reshape(-1).copy() for n, g in zip(names, grids, strict=True)}
    if order == "snake" and len(names) == 2:
        fast = per_frame_full[names[1]].reshape(shape)
        fast[1::2] = fast[1::2, ::-1]
        per_frame_full[names[1]] = fast.reshape(-1)
    n_total = int(np.prod(shape))
    n_frames = n_total - partial
    per_frame = {n: v[:n_frames] for n, v in per_frame_full.items()}

    frames, truth = _peak_frames(
        per_frame, frame_shape, amplitude, background, rng, noise, curve, centers, widths
    )
    stored = np.clip(np.rint(frames), 0, np.iinfo(dtype).max).astype(dtype)

    dataset_dir = root / dataset
    scan_folder = dataset_dir / f"scan{scan:04d}"
    scan_folder.mkdir(parents=True, exist_ok=True)
    master = dataset_dir / f"{dataset}.h5"

    # ---- detector files -------------------------------------------------
    bounds = np.linspace(0, n_frames, n_files + 1).astype(int)
    detector_files = []
    H, W = frame_shape
    for i in range(n_files):
        start, stop = int(bounds[i]), int(bounds[i + 1])
        path = scan_folder / f"{detector}_{i:04d}.h5"
        with h5py.File(path, "w") as f:
            ds = f.create_dataset(
                ESRF_FRAMES_PATH.format(detector=detector),
                data=stored[start:stop],
                chunks=(1, H, W),
                **_ESRF_COMPRESSION,
            )
            ds.attrs["interpretation"] = "image"
            f["entry_0000/measurement/data"] = h5py.SoftLink(ds.name)
            f["entry_0000"].attrs["NX_class"] = "NXentry"
        detector_files.append(path)

    # ---- master file -----------------------------------------------------
    ccmth = energy_to_mono_angle(energy_keV, "Si111")
    with h5py.File(master, "a") as f:
        entry = f.require_group(f"{scan}.1")
        if layout != "2024" and len(names) >= 1:
            fs = entry.require_group("instrument/fscan_parameters")
            scan_type = f"fscan{len(names)}d"
            fs.create_dataset("scan_type", data=scan_type)
            fs.create_dataset("scan_name", data=scan_type)
            if len(names) == 1:
                fs.create_dataset("motor", data=names[0])
                fs.create_dataset("npoints", data=shape[0])
            else:
                fs.create_dataset("slow_motor", data=names[0])
                fs.create_dataset("fast_motor", data=names[1])
                fs.create_dataset("slow_npoints", data=shape[0])
                fs.create_dataset("fast_npoints", data=shape[1])
            if len(names) == 3:
                fs.create_dataset("outer_motor", data=names[0])
                fs.create_dataset("middle_motor", data=names[1])
                fs.create_dataset("inner_motor", data=names[2])
                fs.create_dataset("outer_npoints", data=shape[0])
                fs.create_dataset("middle_npoints", data=shape[1])
                fs.create_dataset("inner_npoints", data=shape[2])
        title = " ".join(f"{n} {coords[n][0]:g} {coords[n][-1]:g} {len(coords[n])}" for n in names)
        entry.create_dataset("title", data=f"fscan{len(names)}d {title} 0.05")
        meas = entry.require_group("measurement")
        for name, values in per_frame.items():
            meas.create_dataset(name, data=values.astype(np.float64))
        meas.create_dataset("elapsed_time", data=np.arange(n_frames) * 0.05)
        pos = entry.require_group("instrument/positioners")
        scalars = {
            "mu": float(per_frame.get("mu", coords.get("mu", np.zeros(1)))[0]),
            "chi": float(per_frame.get("chi", np.zeros(1))[0]),
            "phi": 0.0,
            "samz": 0.0,
            "ffz": 100.0,
            "ffx": 0.0,
            "nfx": 0.0,
            "nfy": 0.0,
            "nfz": 0.0,
            "ccmth": float(ccmth),
        }
        if layout == "2024":
            scalars["obpitch"] = float(coords.get("obpitch", np.zeros(1))[0])
        if layout == "2026":
            scalars.update({"ux": 0.0, "uy": 0.0, "uz": 0.0})
        scalars.update(positioners or {})
        for name, value in scalars.items():
            pos.create_dataset(name, data=np.float64(value))
        for name in names:  # BLISS also records the scanned motors as arrays here
            if name in pos:
                del pos[name]
            pos.create_dataset(name, data=per_frame[name].astype(np.float64))
        entry.attrs["NX_class"] = "NXentry"

    return SyntheticEsrfScan(
        root=root,
        dataset=dataset,
        scan=scan,
        detector=detector,
        master=master,
        detector_files=detector_files,
        motor_dims=tuple(names),
        motor_shape=shape,
        coords=coords,
        per_frame=per_frame,
        frame_shape=frame_shape,
        n_frames=n_frames,
        energy_keV=energy_keV,
        frames=frames,
        truth=truth,
    )


def make_esrf_energy_series(
    root: str | Path,
    dataset: str = "synth_energy",
    scans: tuple[int, ...] = (1, 2, 3),
    energies: tuple[float, ...] = (17.00, 17.05, 17.10),
    **kwargs: Any,
) -> list[SyntheticEsrfScan]:
    """Several scans of one dataset at different ``ccmth`` energies (a multi-scan series)."""
    return [
        make_esrf_scan(root, dataset=dataset, scan=s, energy_keV=e, seed=i, **kwargs)
        for i, (s, e) in enumerate(zip(scans, energies, strict=True))
    ]


# ------------------------------------------------------------------------- PAL-XFEL
@dataclass
class SyntheticPalRun:
    """What :func:`make_pal_run` wrote, with per-point reference results."""

    root: Path
    run: int
    scan: int
    point_files: list[Path]
    raw_files: list[Path]
    laser_key: str
    frame_shape: tuple[int, int]
    delays: np.ndarray
    tables: list[Any]
    frames: list[np.ndarray]
    truth: dict[str, Any] = field(default_factory=dict)

    @property
    def measurement_dir(self) -> Path:
        return self.root / "type=measurement" / f"run={self.run:03d}" / f"scan={self.scan:03d}"

    @property
    def raw_dir(self) -> Path:
        return self.root / "type=raw" / f"run={self.run:03d}" / f"scan={self.scan:03d}"


def make_pal_run(
    root: str | Path,
    run: int = 1,
    scan: int = 1,
    n_points: int = 5,
    shots_per_point: int = 16,
    frame_shape: tuple[int, int] = (24, 32),
    laser_key: str = "event_info.THIRTY_HERTZ",
    delays: np.ndarray | None = None,
    missing_shots: int = 1,
    seed: int = 0,
) -> SyntheticPalRun:
    """Write a PAL-XFEL run in the September 2025 point-file layout.

    Each point has a pandas table (``type=measurement``) and a raw frames file
    (``type=raw``). The laser alternates shot by shot; ``missing_shots`` rows per
    point carry a NaN laser flag (dropped shots) as in real data.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    root = Path(root)
    H, W = frame_shape
    if delays is None:
        delays = np.linspace(-2.0, 8.0, n_points)
    meas_dir = root / "type=measurement" / f"run={run:03d}" / f"scan={scan:03d}"
    raw_dir = root / "type=raw" / f"run={run:03d}" / f"scan={scan:03d}"
    meas_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[0:H, 0:W]
    peak = np.exp(-0.5 * (((yy - H / 2) / 3.0) ** 2 + ((xx - W / 2) / 4.0) ** 2))

    point_files, raw_files, tables, frames_all = [], [], [], []
    ref_on, ref_off, ref_sig_on, ref_sig_off = [], [], [], []
    for p in range(n_points):
        delay = float(delays[p])
        n = shots_per_point
        laser = (np.arange(n) % 2 == 0).astype(float)
        if missing_shots:
            laser[rng.choice(n, size=min(missing_shots, n), replace=False)] = np.nan
        i0 = rng.uniform(0.8, 1.2, size=n)
        response = 1.0 + 0.3 * np.tanh(delay) * np.nan_to_num(laser)
        frames = (
            50.0 * peak[None] * response[:, None, None] * i0[:, None, None]
            + rng.normal(0.0, 1.0, size=(n, H, W))
        ).astype(np.float32)
        frames[:, 0, 0] = -1.0  # a dead pixel with non-positive values, as in real Jungfrau data
        roi_stat = frames[:, H // 4 : 3 * H // 4, W // 4 : 3 * W // 4].sum(axis=(1, 2))
        table = pd.DataFrame(
            {
                "timestamp": np.arange(n, dtype=np.int64) + p * 1000,
                "delay_input": delay + rng.normal(0.0, 1e-4, size=n),
                laser_key: laser,
                "qbpm:eh1:qbpm1:sum": i0,
                "qbpm:oh:qbpm2:sum": i0 * 1.1,
                "detector:eh1:jungfrau2:ROI1_stat.sum": roi_stat,
                "qbpm:eh1:qbpm1:pos_X": rng.normal(0.0, 0.01, size=n),
                "qbpm:eh1:qbpm1:pos_Y": rng.normal(0.0, 0.01, size=n),
                "th_value": np.full(n, 12.5),
                "phi_input": np.full(n, 0.0),
                "chi_value": np.full(n, 90.0),
                "tth_value": np.full(n, 25.0),
                "laser_v_value": np.full(n, 3.2),
                "laser_h_value": np.full(n, -1.1),
            }
        )
        point_path = meas_dir / f"p{p + 1:04d}.h5"
        raw_path = raw_dir / f"p{p + 1:04d}.h5"
        table.to_hdf(point_path, key="measurements", mode="w")
        meta = pd.DataFrame(
            {"timestamp_info.THIRTY_HERTZ": laser, "delay_stage_eh1_delay_stage_delay_time": delay}
        )
        meta.to_hdf(raw_path, key="metadata", mode="w")
        with h5py.File(raw_path, "a") as f:
            f.create_dataset(PAL_FRAMES_PATH, data=frames, chunks=(1, H, W))

        # reference results with the legacy cube builder's exact semantics
        raw = frames.copy()
        raw[raw <= 0] = 1e-4
        keep = ~np.isnan(laser)
        raw_kept, laser_kept = raw[keep], laser[keep]
        ref_on.append(raw_kept[laser_kept == 1.0].mean(axis=0))
        ref_off.append(raw_kept[laser_kept == 0.0].mean(axis=0))
        sig = table["detector:eh1:jungfrau2:ROI1_stat.sum"] / table["qbpm:eh1:qbpm1:sum"]
        ref_sig_on.append(float(sig[table[laser_key] == 1.0].mean()))
        ref_sig_off.append(float(sig[table[laser_key] == 0.0].mean()))

        point_files.append(point_path)
        raw_files.append(raw_path)
        tables.append(table)
        frames_all.append(frames)

    truth = {
        "delays": np.asarray([float(t["delay_input"].median()) for t in tables]),
        "frames_on": np.stack(ref_on),
        "frames_off": np.stack(ref_off),
        "signals_on": np.asarray(ref_sig_on),
        "signals_off": np.asarray(ref_sig_off),
    }
    return SyntheticPalRun(
        root=root,
        run=run,
        scan=scan,
        point_files=point_files,
        raw_files=raw_files,
        laser_key=laser_key,
        frame_shape=frame_shape,
        delays=np.asarray(delays),
        tables=tables,
        frames=frames_all,
        truth=truth,
    )


@dataclass
class SyntheticSpecScan:
    """What :func:`make_pal_spec_scan` wrote."""

    root: Path
    run: int
    scan: int
    shot: int
    path: Path
    motors: dict[str, np.ndarray]
    n_recorded: int
    frame_shape: tuple[int, int]
    images: np.ndarray
    i0: np.ndarray


def make_pal_spec_scan(
    root: str | Path,
    run: int = 1,
    scan: int = 1,
    shot: int = 1,
    motors: tuple[tuple[str, int, float, float], ...] = (("delay", 8, -1.0, 6.0),),
    frame_shape: tuple[int, int] = (20, 24),
    partial: int = 0,
    detector: str = "jungfrau",
    seed: int = 0,
) -> SyntheticSpecScan:
    """Write a PAL-XFEL April 2025 scan file with a spec-like ``scanHistory``.

    ``motors`` holds ``(name, n_points, start, stop)``; one entry gives an
    ``a1scan``, two an ``a2scan`` (first motor outer). ``partial`` points are
    left unrecorded, as in an aborted scan.
    """
    rng = np.random.default_rng(seed)
    root = Path(root)
    H, W = frame_shape
    path = root / "type=measurement" / f"run={run:03d}" / f"scan={scan:03d}" / f"p{shot:04d}.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    coords = {name: np.linspace(lo, hi, n) for name, n, lo, hi in motors}
    if len(motors) == 1:
        ((name, n, lo, hi),) = motors
        command = f"a1scan {name} {lo:g} {hi:g} {n} 0.1"
        n_total = n
    else:
        outer, inner = motors[0], motors[1]
        command = (
            f"a2scan {outer[0]} {outer[2]:g} {outer[3]:g} {outer[1]} "
            f"{inner[0]} {inner[2]:g} {inner[3]:g} {inner[1]} 0.1"
        )
        n_total = outer[1] * inner[1]
    n_recorded = n_total - partial
    if len(motors) == 2:  # the 2-D layout stores whole rows of the inner motor
        n_recorded -= n_recorded % motors[1][1]
    yy, xx = np.mgrid[0:H, 0:W]
    peak = np.exp(-0.5 * (((yy - H / 2) / 3.0) ** 2 + ((xx - W / 2) / 4.0) ** 2))
    i0 = rng.uniform(0.9, 1.1, size=n_recorded)
    images = (
        100.0 * peak[None] * i0[:, None, None] * (1 + 0.1 * np.arange(n_recorded))[:, None, None]
    ).astype(np.float32)
    with h5py.File(path, "w") as f:
        g = f.create_group(f"run/scan{scan:04d}")
        g.attrs["scanMode"] = "LINEAR 1D" if len(motors) == 1 else "LINEAR 2D"
        g.attrs["scanHistory"] = command
        for i, (name, _n, _lo, _hi) in enumerate(motors):
            g.create_dataset(f"motor/m{i + 1}", data=coords[name])
        det = g.create_group("det")
        if len(motors) == 1:
            det.create_dataset("ohqbpm2_totalsum/data", data=i0.reshape(n_recorded, 1))
            det.create_dataset("ohqbpm2_totalsum/rawData", data=np.repeat(i0[:, None], 4, axis=1))
            det.create_dataset(f"{detector}/data", data=images)
        else:
            rows = n_recorded // motors[1][1]
            det.create_dataset(
                "ohqbpm2_totalsum/data", data=i0[: rows * motors[1][1]].reshape(rows, motors[1][1])
            )
            det.create_dataset(
                f"{detector}/data",
                data=images[: rows * motors[1][1]].reshape(rows, motors[1][1], H, W),
            )
    return SyntheticSpecScan(
        root, run, scan, shot, path, coords, n_recorded, frame_shape, images, i0
    )


# ------------------------------------------------------------------------------ LCLS
@dataclass
class SyntheticLclsCube:
    """What :func:`make_lcls_cube` wrote, with the legacy binning as reference."""

    root: Path
    run: int
    on_path: Path
    off_path: Path
    stats_path: Path
    scanvar: np.ndarray
    i0_on: np.ndarray
    i0_off: np.ndarray
    truth: dict[str, np.ndarray]


def make_lcls_cube(
    root: str | Path,
    run: int = 296,
    n_steps: int = 6,
    repeats: int = 4,
    frame_shape: tuple[int, int] = (24, 24),
    seed: int = 0,
) -> SyntheticLclsCube:
    """Write an LCLS XCS cube triple: per-shot on/off stacks and a stats table."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    H, W = frame_shape
    scanvar = np.repeat(np.linspace(-1.0, 4.0, n_steps), repeats)
    n = len(scanvar)
    i0_on = rng.uniform(0.8, 1.2, size=n)
    i0_off = rng.uniform(0.8, 1.2, size=n)
    i0_on[0] = 0.0  # a dropped shot the legacy loop skips
    yy, xx = np.mgrid[0:H, 0:W]
    peak = np.exp(-0.5 * (((yy - H / 2) / 3.0) ** 2 + ((xx - W / 2) / 3.0) ** 2))
    on = (
        40 * peak[None] * (1 + 0.2 * np.tanh(scanvar))[:, None, None] * i0_on[:, None, None]
    ).astype(np.float32)
    off = (40 * peak[None] * i0_off[:, None, None]).astype(np.float32)
    on_path = root / f"Run{run:04d}_onStk.npy"
    off_path = root / f"Run{run:04d}_offStk.npy"
    stats_path = root / f"Run{run:04d}_stats.csv"
    np.save(on_path, on)
    np.save(off_path, off)
    table = np.column_stack([np.arange(n), scanvar, i0_on, i0_off])
    np.savetxt(stats_path, table, delimiter=",", header="shot,scanvar,OnI0,OffI0", comments="")

    # legacy binning: sum of I0-normalised images per unique scan value, shots with I0 == 0 skipped
    bins = np.unique(scanvar)
    avg_on = np.zeros((len(bins), H, W))
    avg_off = np.zeros((len(bins), H, W))
    for i in range(n):
        if i0_on[i] == 0 or i0_off[i] == 0:
            continue
        b = min(np.digitize(scanvar[i], bins, right=True), len(bins) - 1)
        avg_on[b] += on[i] / i0_on[i]
        avg_off[b] += off[i] / i0_off[i]
    truth = {"bins": bins, "avg_on": avg_on, "avg_off": avg_off}
    return SyntheticLclsCube(
        root, run, on_path, off_path, stats_path, scanvar, i0_on, i0_off, truth
    )


# ---------------------------------------------------------------------- legacy cube
def make_legacy_cube(
    path: str | Path,
    n_points: int = 6,
    frame_shape: tuple[int, int] = (20, 24),
    axis: str = "delays",
    seed: int = 0,
) -> Path:
    """Write a ``runN.h5`` cube in the layout of ``Aaron_allscan_cube_parallel.py``."""
    rng = np.random.default_rng(seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    H, W = frame_shape
    values = np.linspace(-1.0, 4.0, n_points)
    yy, xx = np.mgrid[0:H, 0:W]
    peak = np.exp(-0.5 * (((yy - H / 2) / 3.0) ** 2 + ((xx - W / 2) / 4.0) ** 2))
    on = 30 * peak[None] * (1 + 0.2 * np.tanh(values))[:, None, None] + rng.normal(
        0, 0.1, (n_points, H, W)
    )
    off = 30 * peak[None] * np.ones(n_points)[:, None, None] + rng.normal(0, 0.1, (n_points, H, W))
    with h5py.File(path, "w") as f:
        for name in ("delays", "phi", "chi", "th", "tth", "laser_v", "laser_h"):
            f.create_dataset(name, data=values if name == axis else np.full(n_points, 1.5))
        f.create_dataset("index", data=np.linspace(0, n_points - 1, n_points))
        f.create_dataset("signals_on", data=rng.uniform(0.9, 1.1, n_points))
        f.create_dataset("signals_off", data=rng.uniform(0.9, 1.1, n_points))
        f.create_dataset("jungfrau_on", data=on.astype(np.float64))
        f.create_dataset("jungfrau_off", data=off.astype(np.float64))
    return path


# ------------------------------------------------------------------------ ESRF z-stack
@dataclass(frozen=True)
class Region:
    """One diffracting region of the synthetic sample: orientation, peak widths, strain."""

    label: int
    tilt: tuple[float, float]  # (chi, mu) centre of the rocking curve, deg
    width: tuple[float, float]  # (chi, mu) sigma of the rocking curve, deg
    brightness: float  # integrated intensity of the rocking curve relative to the brightest
    energy_offset: float  # centre of the energy curve relative to the nominal energy, keV


SAMPLE_REGIONS: dict[str, Region] = {
    "matrix": Region(0, (0.0, 0.0), (0.10, 0.15), 0.05, 0.0),
    "domain_a1": Region(1, (-0.06, -0.14), (0.06, 0.09), 1.0, -0.004),
    "domain_a2": Region(2, (0.06, 0.10), (0.06, 0.09), 0.85, 0.004),
    "grain_b": Region(3, (0.22, 0.50), (0.08, 0.12), 0.7, 0.006),
}
"""The regions of :func:`dfxm_sample`: grain A split into two domains whose chi and mu differ
by one to two peak widths, grain B several widths away, and a faint matrix around them."""


@dataclass
class SampleModel:
    """A field of view of the synthetic sample, as per-pixel maps for :func:`make_esrf_scan`."""

    amplitude: np.ndarray  # peak height per pixel (counts); the brightest domain reaches ``peak``
    centers: dict[str, np.ndarray]  # chi and mu centre per pixel (deg)
    widths: dict[str, np.ndarray]  # chi and mu sigma per pixel (deg)
    energy_offset: np.ndarray  # energy centre per pixel relative to the nominal energy (keV)
    labels: np.ndarray  # the :class:`Region` label of every pixel
    grain_center: tuple[float, float]  # (row, column) of the centre of grain A

    @property
    def scan_kwargs(self) -> dict[str, Any]:
        """``amplitude``, ``centers`` and ``widths`` for :func:`make_esrf_scan`."""
        return {"amplitude": self.amplitude, "centers": self.centers, "widths": self.widths}


def _disc(yy: np.ndarray, xx: np.ndarray, cy: float, cx: float, radius: float) -> np.ndarray:
    """A disc with a soft edge about one pixel wide: 1 inside, 0 outside."""
    distance = np.hypot(yy - cy, xx - cx)
    return 0.5 * (1.0 + erf((radius - distance) / 1.1))


def dfxm_sample(
    frame_shape: tuple[int, int],
    drift: float = 0.0,
    peak: float = 4000.0,
    regions: dict[str, Region] = SAMPLE_REGIONS,
) -> SampleModel:
    """Two grains in the field of view, as per-pixel orientation, width and strain maps.

    Grain A is a disc that a wall splits into two domains whose chi and mu differ slightly
    (``domain_a1``, ``domain_a2``); grain B is a smaller disc at a clearly different
    orientation; the faint matrix around them is bent a little across the field. ``drift``
    from 0 to 1 walks through a z-stack: grain A slides across the field (a grain inclined
    to the beam) and grain B shrinks (a grain that ends). The default tilts sit well inside
    scans of chi +-0.5 and mu +-1 deg, so such scans record every region.
    """
    H, W = frame_shape
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    size = float(min(H, W))
    matrix, a1, a2, b = (regions[k] for k in ("matrix", "domain_a1", "domain_a2", "grain_b"))

    cy, cx = H * (0.62 - 0.17 * drift), W * (0.32 + 0.23 * drift)  # grain A slides up and right
    in_a = _disc(yy, xx, cy, cx, 0.22 * size)
    angle = np.radians(30.0)  # the domain wall through the centre of grain A
    wall = (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle) > 0
    in_b = _disc(yy, xx, 0.25 * H, 0.78 * W, 0.14 * size * (1.1 - 0.6 * drift))

    labels = np.full((H, W), matrix.label, dtype=np.int64)
    labels[in_a > 0.5] = np.where(wall, a2.label, a1.label)[in_a > 0.5]
    labels[in_b > 0.5] = b.label

    def per_pixel(value: Callable[[Region], float]) -> np.ndarray:
        out = np.zeros((H, W))
        for region in regions.values():
            out[labels == region.label] = value(region)
        return out

    bend_chi = 0.03 * (xx / max(W - 1, 1) - 0.5)  # the whole field is bent a little
    bend_mu = 0.04 * (yy / max(H - 1, 1) - 0.5)
    brightness = (  # integrated intensity, blended at the edges of the grains
        matrix.brightness * (1 - in_a) * (1 - in_b)
        + np.where(wall, a2.brightness, a1.brightness) * in_a * (1 - in_b)
        + b.brightness * in_b
    )
    widths = {"chi": per_pixel(lambda r: r.width[0]), "mu": per_pixel(lambda r: r.width[1])}
    # the same integrated intensity makes a lower peak where the curve is wider
    height = peak * brightness * (a1.width[0] * a1.width[1]) / (widths["chi"] * widths["mu"])
    return SampleModel(
        amplitude=height,
        centers={
            "chi": per_pixel(lambda r: r.tilt[0]) + bend_chi,
            "mu": per_pixel(lambda r: r.tilt[1]) + bend_mu,
        },
        widths=widths,
        energy_offset=per_pixel(lambda r: r.energy_offset),
        labels=labels,
        grain_center=(cy, cx),
    )


@dataclass
class SyntheticZStack:
    """What :func:`make_esrf_zstack` wrote: a mosaicity scan and an energy series per height."""

    root: Path
    dataset: str
    z_values: np.ndarray
    mosa_scans: list[int]
    energy_scans: list[tuple[int, int]]
    energies: np.ndarray
    layers: list[SyntheticEsrfScan]
    truth: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def dataset_dir(self) -> Path:
        return self.root / self.dataset


def make_esrf_zstack(
    root: str | Path,
    dataset: str = "synth_zstack",
    z_values: tuple[float, ...] = (-0.02, -0.01, 0.0, 0.01, 0.02),
    mosa: tuple[tuple[str, int], ...] = (("chi", 6), ("mu", 15)),
    energies: tuple[float, ...] = (16.98, 16.99, 17.0, 17.01, 17.02),
    energy_motor: tuple[str, int] = ("mu", 8),
    frame_shape: tuple[int, int] = (40, 48),
    first_scan: int = 1,
    n_files: int = 1,
    seed: int = 0,
    energy_scan: str = "rocking",
    **kwargs: Any,
) -> SyntheticZStack:
    """A mosaicity scan and an energy series per sample height (``samz``), in one dataset.

    Two ways of scanning the energy, chosen with ``energy_scan``:

    - ``"rocking"``: at every height one ``mosa`` scan (chi x mu) at the nominal energy,
      followed by one ``energy_motor`` scan (a mu rocking curve) per energy;
    - ``"mosa"``: at every height the full ``mosa`` scan is repeated at every energy, so
      energy, chi and mu lie on one grid and the series opens with dims
      ``(energy, chi, mu, y, x)``. The scan at the energy nearest the mean is the
      height's ``mosa_scans`` entry.

    ``bexa.open(dataset, scan=mosa_scans[k])`` and
    ``bexa.open(dataset, scan=energy_scans[k])`` open layer ``k``. The sample is
    :func:`dfxm_sample` at a drift that grows with the height: grain A, split into two
    domains, slides across the field of view and grain B shrinks. The energy centre of
    every region is offset by its strain, on top of a gradient along ``x`` and with the
    height. The planted values are in ``truth``: ``grain_center`` (row, column of grain A)
    per layer and the ``energy_center``, ``chi_center``, ``mu_center`` and ``labels`` maps.
    """
    if energy_scan not in ("rocking", "mosa"):
        raise ValueError(f"energy_scan must be 'rocking' or 'mosa', not {energy_scan!r}")
    root = Path(root)
    height, width = frame_shape
    _, xx = np.mgrid[0:height, 0:width]
    z_array = np.asarray(z_values, dtype=float)
    e_array = np.asarray(energies, dtype=float)
    n_z = len(z_array)
    e_span = float(e_array.max() - e_array.min()) or 0.04
    layers: list[SyntheticEsrfScan] = []
    mosa_scans: list[int] = []
    energy_scans: list[tuple[int, int]] = []
    grain_centres = np.zeros((n_z, 2))
    energy_centres = np.zeros((n_z, height, width))
    tilt_centres = {name: np.zeros((n_z, height, width)) for name in ("chi", "mu")}
    labels = np.zeros((n_z, height, width), dtype=np.int64)
    scan = first_scan
    for k, z in enumerate(z_array):
        frac = k / max(n_z - 1, 1)
        sample = dfxm_sample(frame_shape, drift=frac)
        grain_centres[k] = sample.grain_center
        for name in tilt_centres:
            tilt_centres[name][k] = sample.centers[name]
        labels[k] = sample.labels
        fixed = {"samz": float(z)}
        if energy_scan == "rocking":  # the mosaicity scan at the nominal energy comes first
            made = make_esrf_scan(
                root,
                dataset=dataset,
                scan=scan,
                motors=mosa,
                frame_shape=frame_shape,
                n_files=n_files,
                positioners=fixed,
                seed=seed + scan,
                **sample.scan_kwargs,
                **kwargs,
            )
            layers.append(made)
            mosa_scans.append(scan)
            scan += 1
        # the matrix is bent along x and the strain drifts with the height; each region adds its own
        centre = (
            e_array.mean()
            + e_span * (0.1 * (xx / max(width - 1, 1) - 0.5) + 0.2 * (frac - 0.5))
            + sample.energy_offset
        )
        energy_centres[k] = centre
        first = scan
        nominal = int(np.argmin(np.abs(e_array - e_array.mean())))
        for j, energy in enumerate(e_array):
            weight = np.exp(-0.5 * ((energy - centre) / (0.3 * e_span)) ** 2)
            made = make_esrf_scan(
                root,
                dataset=dataset,
                scan=scan,
                motors=mosa if energy_scan == "mosa" else (energy_motor,),
                frame_shape=frame_shape,
                n_files=n_files,
                energy_keV=float(energy),
                amplitude=sample.amplitude * weight,
                centers=sample.centers,
                widths=sample.widths,
                positioners=fixed,
                seed=seed + scan,
                **kwargs,
            )
            if energy_scan == "mosa" and j == nominal:
                layers.append(made)
                mosa_scans.append(scan)
            scan += 1
        energy_scans.append((first, scan - 1))
    return SyntheticZStack(
        root=root,
        dataset=dataset,
        z_values=z_array,
        mosa_scans=mosa_scans,
        energy_scans=energy_scans,
        energies=e_array,
        layers=layers,
        truth={
            "grain_center": grain_centres,
            "energy_center": energy_centres,
            "chi_center": tilt_centres["chi"],
            "mu_center": tilt_centres["mu"],
            "labels": labels,
        },
    )
