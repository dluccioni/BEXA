"""The ``Scan`` handle: open a measurement lazily, then read, preview or reduce it.

``bexa.open(path)`` sniffs the layout, ``bexa.open_profile(name, sample=...)``
starts from a beamtime profile. Both return a ``Scan`` that has read only
metadata; frames are touched when ``read``, ``preview`` or ``reduce`` is called.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from bexa._log import get_logger
from bexa.config import BeamtimeProfile, cache_root, load_profile
from bexa.config.paths import CACHE_DIR_ENV
from bexa.core.cache import Cache
from bexa.core.registry import engines
from bexa.core.roi import ROI
from bexa.core.structure import Structure
from bexa.io.base import Source
from bexa.io.formats import FormatSpec, best_spec, load_spec, match_spec
from bexa.io.multi import MultiScanSource

log = get_logger(__name__)

__all__ = ["Scan", "list_scans", "open", "open_profile", "open_source"]


class Scan:
    """Lazy handle to one measurement (or an energy series of measurements).

    Parameters
    ----------
    source
        The engine instance that reads frames.
    spec
        Format spec the source was built from.
    geometry
        Detector geometry (see :mod:`bexa.geometry.detector`).
    crystal
        Crystal for hkl conversions (see :mod:`bexa.crystal`).
    cache
        Result cache; ``None`` disables caching.
    """

    def __init__(
        self,
        source: Source,
        spec: FormatSpec | None = None,
        geometry: Any = None,
        crystal: Any = None,
        name: str = "",
        cache: Cache | None = None,
        profile: BeamtimeProfile | None = None,
        cache_reductions: bool = False,
        hkl_center: tuple[int, int, int] | None = None,
    ) -> None:
        self.source = source
        self.spec = spec
        self.geometry = geometry
        self.crystal = crystal
        self.hkl_center = hkl_center
        self.name = name or getattr(source, "dataset", "") or "scan"
        self.cache = cache
        self.cache_reductions = cache_reductions
        self.profile = profile
        self.roi = ROI()
        self._structure: Structure | None = None

    # ------------------------------------------------------------ metadata
    @property
    def structure(self) -> Structure:
        if self._structure is None:
            self._structure = self.source.structure()
        return self._structure

    @property
    def dims(self) -> tuple[str, ...]:
        return self.structure.dims

    @property
    def shape(self) -> tuple[int, ...]:
        return self.structure.shape

    @property
    def coords(self) -> dict[str, np.ndarray]:
        return {d: self.structure.coords[d] for d in self.structure.motor_dims}

    @property
    def n_frames(self) -> int:
        return self.source.n_frames

    @property
    def frame_shape(self) -> tuple[int, int]:
        return self.source.frame_shape

    @property
    def energy_keV(self) -> float | None:
        return self.structure.energy_keV

    @property
    def files(self) -> list[Path]:
        return self.source.files()

    @property
    def is_multi(self) -> bool:
        return isinstance(self.source, MultiScanSource)

    def info(self) -> str:
        """Human summary: files, structure, ROI, memory estimates."""
        s = self.structure
        frame_mb = s.frame_shape[0] * s.frame_shape[1] * 4 / 1e6
        lines = [
            f"{self.name}: {type(self.source).__name__}"
            + (f" ({self.spec.name})" if self.spec else ""),
            s.describe(),
            f"  frame {s.frame_shape}, stored {self.source.dtype}, "
            f"{frame_mb:.1f} MB per frame as float32, "
            f"{frame_mb * s.n_frames / 1e3:.2f} GB for the stack",
            f"  {len(self.files)} files, first: {self.files[0]}",
        ]
        if self.roi:
            lines.append("  " + self.roi.describe(s))
        if self.geometry is not None:
            lines.append(f"  geometry: {self.geometry}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Scan({self.name!r}, dims={self.dims}, shape={self.shape})"

    # ------------------------------------------------------------ reading
    def reduce(self, accumulators: Sequence[Any], **kwargs: Any) -> xr.Dataset:
        """Run accumulators over this scan (see :func:`bexa.core.reductions.reduce`)."""
        from bexa.core.reductions import reduce

        return reduce(self, accumulators, **kwargs)

    # ------------------------------------------------------------ everyday verbs
    def sum(self, roi: ROI | None = None, downsample: Any = None, **kwargs: Any) -> xr.DataArray:
        """The summed image ``(y, x)``; ``roi``, ``downsample``, ``device`` as in :meth:`reduce`."""
        from bexa.core.reductions import Sum

        return self.reduce([Sum()], roi=roi, downsample=downsample, **kwargs)["sum"]

    def max(self, roi: ROI | None = None, downsample: Any = None, **kwargs: Any) -> xr.DataArray:
        """The pixel-wise maximum over the frames."""
        from bexa.core.reductions import Max

        return self.reduce([Max()], roi=roi, downsample=downsample, **kwargs)["max"]

    def rocking_curve(
        self, roi: ROI | None = None, method: str = "sum", downsample: Any = None, **kwargs: Any
    ) -> xr.DataArray:
        """Integrated intensity of the pixel ROI at every motor point (1-D for a rocking scan).

        The whole frame is integrated when ``roi`` has no pixel ranges; motor
        ranges of the ROI limit the points.
        """
        from bexa.core.reductions import RoiIntegral

        window = roi if roi is not None else self.roi
        result = self.reduce(
            [RoiIntegral({"curve": window}, method=method)],
            roi=window,
            downsample=downsample,
            **kwargs,
        )
        return result["roi_curve"].rename("rocking_curve")

    def com(
        self,
        axes: Sequence[str] | None = None,
        sigma: float = 3.0,
        roi: ROI | None = None,
        downsample: Any = None,
        moments: int = 2,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Centre-of-mass maps per motor axis: ``com_<axis>``, ``width_<axis>`` and ``total``."""
        from bexa.core.reductions import MotorCOM

        return self.reduce(
            [MotorCOM(axes=axes, sigma=sigma, moments=moments)],
            roi=roi,
            downsample=downsample,
            **kwargs,
        )

    def energy_com(
        self, sigma: float = 0.0, roi: ROI | None = None, downsample: Any = None, **kwargs: Any
    ) -> xr.Dataset:
        """Centre of mass along the ``energy`` dim of a series: ``com_energy``, ``width_energy``."""
        from bexa.core.reductions import EnergyCOM

        return self.reduce([EnergyCOM(sigma=sigma)], roi=roi, downsample=downsample, **kwargs)

    def stats(self, roi: ROI | None = None, downsample: Any = None, **kwargs: Any) -> xr.Dataset:
        """Per-frame sum, mean, maximum and detector centre of mass on the motor grid."""
        from bexa.core.reductions import FrameStats

        return self.reduce([FrameStats()], roi=roi, downsample=downsample, **kwargs)

    def _repr_html_(self) -> str:
        """A small table for notebooks: what the scan is, before any frame is read."""
        s = self.structure
        rows: list[tuple[str, str]] = [
            ("format", type(self.source).__name__ + (f" ({self.spec.name})" if self.spec else "")),
            ("scan", s.scan_type + (f": {s.title}" if s.title else "")),
            ("dims", " x ".join(f"{d}[{n}]" for d, n in zip(s.dims, s.shape, strict=True))),
        ]
        for d in s.motor_dims:
            unit = s.units.get(d, "")
            rows.append((d, f"{s.coords[d].min():.5g} to {s.coords[d].max():.5g} {unit}".rstrip()))
        rows.append(
            ("energy", f"{s.energy_keV:.4f} keV" if s.energy_keV is not None else "unknown")
        )
        frames = f"{s.n_frames} of {s.frame_shape[0]} x {s.frame_shape[1]} {self.source.dtype}"
        if s.n_missing:
            frames += f", {s.n_missing} grid points missing"
        rows.append(("frames", frames))
        rows.append(("files", f"{len(self.files)}, first {self.files[0]}"))
        body = "".join(
            f"<tr><th style='text-align:left'>{k}</th><td>{v}</td></tr>" for k, v in rows
        )
        return f"<table><caption><b>{self.name}</b></caption>{body}</table>"

    def preview(
        self,
        downsample: int | Sequence[int] | None = 8,
        roi: ROI | None = None,
        apply_log: bool = False,
        method: str = "mean",
        device: str | None = "auto",
        cache: bool = True,
        **kwargs: Any,
    ) -> xr.DataArray:
        """The downsampled volume ``(motors..., y, x)``; cached on disk when possible."""
        from bexa.core.reductions import Preview, reduce

        use_cache = self.cache if cache else None
        res = reduce(
            self,
            [Preview(apply_log=apply_log)],
            roi=roi,
            downsample=downsample,
            device=device,
            method=method,
            cache=use_cache,
            **kwargs,
        )
        return res["preview"]

    def read(
        self,
        roi: ROI | None = None,
        downsample: int | Sequence[int] | None = None,
        device: str | None = "cpu",
        dtype: Any = np.float32,
        **kwargs: Any,
    ) -> xr.DataArray:
        """Load frames into memory as ``(motors..., y, x)`` (checks the size against the budget)."""
        from bexa.core import backend
        from bexa.core.reductions import Preview, make_plan, reduce

        plan = make_plan(self.structure, roi if roi is not None else self.roi, downsample)
        nbytes = int(np.prod(plan.shape)) * np.dtype(dtype).itemsize
        budget = backend.memory_budget(device)
        if nbytes > budget:
            raise MemoryError(
                f"reading {plan.shape} at {np.dtype(dtype)} needs {nbytes / 1e9:.1f} GB but the "
                f"budget is {budget / 1e9:.1f} GB; add an ROI or downsample, or use reduce()"
            )
        res = reduce(
            self,
            [Preview()],
            roi=roi,
            downsample=downsample,
            device=device,
            dtype=dtype,
            cache=False,
            **kwargs,
        )
        return res["preview"]

    def batches(self, roi: ROI | None = None, downsample: Any = None, **kwargs: Any):
        """Iterate over :class:`bexa.io.base.FrameBatch` objects (for custom reductions)."""
        from bexa.core.reductions import iter_batches, make_plan

        plan = make_plan(self.structure, roi if roi is not None else self.roi, downsample)
        return iter_batches(self.source, plan, **kwargs)

    def to_dask(self, chunks: int | str = "auto") -> xr.DataArray:
        """A lazy dask-backed DataArray of the full-resolution stack (needs ``dask``)."""
        import dask
        import dask.array as da

        s = self.structure
        h, w = s.frame_shape
        n = self.n_frames
        chunk = int(chunks) if isinstance(chunks, int) else max(1, int(2e8 // (h * w * 4)))
        blocks = [
            da.from_delayed(
                dask.delayed(self.source.read_frames)(
                    slice(i, min(i + chunk, n)), dtype=np.float32
                ),
                shape=(min(i + chunk, n) - i, h, w),
                dtype=np.float32,
            )
            for i in range(0, n, chunk)
        ]
        stack = da.concatenate(blocks, axis=0)
        return xr.DataArray(
            stack, dims=("frame", "y", "x"), coords={"frame": np.arange(n)}, name="frames"
        )

    def refresh(self) -> Scan:
        """Re-read metadata for a scan that is still being written."""
        self.source.refresh()
        self._structure = None
        return self

    def close(self) -> None:
        self.source.close()

    def __enter__(self) -> Scan:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ----------------------------------------------------------------- opening
def _resolve_spec(
    path: Path, format_name: str | None, overrides: dict[str, Any] | None
) -> FormatSpec:
    if format_name:
        return load_spec(format_name, overrides=overrides)
    spec = best_spec(path)
    if spec is None:
        scored = match_spec(path)
        hint = ", ".join(f"{s.name} ({sc:.0%})" for sc, s in scored[:3]) or "none"
        raise ValueError(
            f"no format spec matches {path}; closest: {hint}. "
            "Run 'bexa profile sniff' to draft one."
        )
    if overrides:
        spec = load_spec(spec.name, overrides=overrides)
    return spec


def open_source(
    spec: FormatSpec,
    path: str | Path,
    scan: int | Sequence[int] | tuple[int, int] | None = None,
    detector: str | None = None,
    stack_dim: str | None = None,
    **kwargs: Any,
) -> Source:
    """Instantiate the engine of ``spec`` for ``path``; several scans become a MultiScanSource.

    ``stack_dim`` names the new leading dim of a series: ``"energy"`` (the
    default when every scan has its own energy), ``"scan"`` (the scan numbers)
    or a positioner such as ``"samz"`` read from every scan.
    """
    engine = engines.get(spec.engine)
    path = Path(path)
    if detector is not None:
        kwargs["detector"] = detector
    if scan is None or isinstance(scan, (int, np.integer)):
        try:
            return engine.from_path(spec, path, scan=None if scan is None else int(scan), **kwargs)
        except ValueError as exc:
            if scan is not None or not hasattr(engine, "list_scans"):
                raise
            log.info("%s; opening all scans as a series", exc)
            scans = engine.list_scans(spec, path.parent, path.name, detector)
    elif isinstance(scan, tuple) and len(scan) == 2:
        scans = list(range(int(scan[0]), int(scan[1]) + 1))
    else:
        scans = [int(s) for s in scan]
    parts = [engine.from_path(spec, path, scan=s, **kwargs) for s in scans]
    energies = [p.energy_keV for p in parts]
    if stack_dim is None:
        distinct = (
            len({round(e, 6) for e in energies if e is not None})
            if all(e is not None for e in energies)
            else 0
        )
        stack_dim = "energy" if distinct == len(parts) and len(parts) > 1 else "scan"
    if stack_dim == "energy":
        if any(e is None for e in energies):
            raise ValueError("cannot stack on energy: a scan of the series has no energy")
        coords = [float(e) for e in energies]
    elif stack_dim == "scan":
        coords = [float(s) for s in scans]
    else:  # a positioner recorded in every scan, for example samz
        coords = [_positioner(p, stack_dim, s) for p, s in zip(parts, scans, strict=True)]
    order = np.arange(len(parts)) if stack_dim == "scan" else np.argsort(coords)
    parts = [parts[i] for i in order]
    coords = [coords[i] for i in order]
    return MultiScanSource(parts, stack_dim, coords)


def _positioner(source: Source, name: str, scan: int) -> float:
    """The fixed value of motor ``name`` in one scan of a series."""
    s = source.structure()
    if name in s.scalars:
        return float(s.scalars[name])
    if name in s.per_frame:
        return float(np.mean(s.per_frame[name]))
    raise ValueError(
        f"scan {scan} records no positioner {name!r} to stack on; available: {sorted(s.scalars)}"
    )


def open(
    path: str | Path | None = None,
    *,
    profile: str | Path | BeamtimeProfile | None = None,
    sample: str | None = None,
    dataset: str | None = None,
    format: str | None = None,
    scan: int | Sequence[int] | tuple[int, int] | None = None,
    detector: str | None = None,
    overrides: dict[str, Any] | None = None,
    energy_keV: float | None = None,
    geometry: Any = None,
    crystal: Any = None,
    cache: Cache | bool | None = True,
    cache_reductions: bool | None = None,
    **kwargs: Any,
) -> Scan:
    """Open a scan from a path, or from a beamtime profile and a sample name.

    Parameters
    ----------
    path
        Dataset folder, master file, ``scanNNNN`` folder, run folder or cube
        file; the format is sniffed unless ``format`` names a spec. Leave it
        out to open through a profile.
    profile, sample, dataset
        A beamtime profile (name, path or object; with no path and no profile
        the ``BEXA_PROFILE`` variable is read) and the sample, or the dataset
        folder, to open in it.
    scan
        One scan number, ``(start, end)`` or a list; ``None`` opens the only scan
        or, for a dataset with several, all of them as a series. ``stack_dim``
        names the dim of a series (``energy``, ``scan`` or a positioner).
    detector
        Detector name when the layout has several.
    energy_keV
        Override the energy derived from the monochromator.
    cache_reductions
        Keep reduction results in the disk cache like previews. Off by default
        for a path; on for a profile with a ``processed_root``.
    """
    if path is None or profile is not None:
        if path is not None:
            raise ValueError("give either a path or a profile, not both")
        if energy_keV is not None:
            kwargs["energy_keV"] = energy_keV
        return open_profile(
            profile,
            sample=sample,
            scan=scan,
            detector=detector,
            dataset=dataset,
            cache_reductions=cache_reductions,
            **kwargs,
        )
    if format is None and "::" in str(path):  # file.h5::/dataset shortcut
        format = "generic_stack"
    spec = _resolve_spec(Path(str(path).split("::")[0]), format, overrides)
    if energy_keV is not None:
        kwargs["energy_keV"] = energy_keV
    source = open_source(spec, path, scan=scan, detector=detector, **kwargs)
    if geometry is None:
        from bexa.geometry.detector import DetectorGeometry

        geometry = DetectorGeometry.from_spec(
            spec, getattr(source, "detector", detector), energy_keV=source.energy_keV
        )
    if cache is True:
        cache = Cache(cache_root())
    elif cache is False:
        cache = None
    return Scan(
        source,
        spec,
        geometry=geometry,
        crystal=crystal,
        cache=cache,
        cache_reductions=bool(cache_reductions) and cache is not None,
    )


def open_profile(
    profile: str | Path | BeamtimeProfile | None,
    sample: str | None = None,
    scan: int | Sequence[int] | tuple[int, int] | None = None,
    detector: str | None = None,
    dataset: str | None = None,
    cache_reductions: bool | None = None,
    **kwargs: Any,
) -> Scan:
    """Open a scan described by a beamtime profile (``bexa.open(profile=..., sample=...)``).

    ``profile`` may be ``None`` to use ``BEXA_PROFILE``. Reduction results are
    cached under the profile's ``processed_root`` unless ``cache_reductions``
    says otherwise.
    """
    prof = profile if isinstance(profile, BeamtimeProfile) else load_profile(profile)
    spec = load_spec(prof.format, overrides=prof.format_overrides or None)
    root = prof.data_root()
    cif = None
    hkl_center = None
    if sample is not None:
        entry = prof.sample(sample)
        dataset = entry.dataset
        detector = detector or entry.detector
        cif = entry.cif
        hkl_center = entry.hkl_center
    if dataset is None:
        raise ValueError("give sample=<name from the profile> or dataset=<folder name>")
    detector = detector or prof.detector
    path = root / dataset
    source = open_source(spec, path, scan=scan, detector=detector, **kwargs)

    from bexa.geometry.detector import DetectorGeometry

    geometry = DetectorGeometry.from_spec(
        spec,
        getattr(source, "detector", detector),
        energy_keV=source.energy_keV,
        profile=prof.geometry,
    )
    crystal = None
    if cif:
        cif_path = Path(cif)
        if not cif_path.is_absolute():
            cif_path = root / cif_path
        try:
            from bexa.crystal.cif import crystal_from_cif

            crystal = crystal_from_cif(cif_path)
        except Exception as exc:
            log.warning("could not load CIF %s: %s", cif_path, exc)
    cache = Cache(cache_root(prof.processed_root))
    if cache_reductions is None:
        cache_reductions = prof.processed_root is not None or bool(os.environ.get(CACHE_DIR_ENV))
    return Scan(
        source,
        spec,
        geometry=geometry,
        crystal=crystal,
        cache=cache,
        profile=prof,
        name=f"{prof.name}/{dataset}",
        cache_reductions=cache_reductions,
        hkl_center=None if hkl_center is None else (hkl_center[0], hkl_center[1], hkl_center[2]),
    )


def list_scans(
    path: str | Path, *, format: str | None = None, detector: str | None = None
) -> list[int]:
    """Scan numbers of a dataset that hold detector files (layouts with numbered scans).

    ``path`` is a dataset folder, its master file or one of its ``scanNNNN``
    folders. Layouts without numbered scans give an empty list.
    """
    path = Path(str(path).split("::")[0])
    spec = _resolve_spec(path, format, None)
    engine = engines.get(spec.engine)
    if not hasattr(engine, "list_scans"):
        return []
    if path.is_file():
        root, dataset = path.parent.parent, path.stem
    elif path.name.startswith("scan") and path.name[4:].isdigit():
        root, dataset = path.parent.parent, path.parent.name
    else:
        root, dataset = path.parent, path.name
    return [int(number) for number in engine.list_scans(spec, root, dataset, detector)]
