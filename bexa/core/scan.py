"""The ``Scan`` handle: open a measurement lazily, then read, preview or reduce it.

``bexa.open(path)`` sniffs the layout, ``bexa.open_profile(name, sample=...)``
starts from a beamtime profile. Both return a ``Scan`` that has read only
metadata; frames are touched when ``read``, ``preview`` or ``reduce`` is called.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from bexa._log import get_logger
from bexa.config import BeamtimeProfile, cache_root, load_profile
from bexa.core.cache import Cache
from bexa.core.registry import engines
from bexa.core.roi import ROI
from bexa.core.structure import Structure
from bexa.io.base import Source
from bexa.io.formats import FormatSpec, best_spec, load_spec, match_spec
from bexa.io.multi import MultiScanSource

log = get_logger(__name__)

__all__ = ["Scan", "open", "open_profile", "open_source"]


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
    def reduce(self, accumulators: Sequence[Any], **kwargs: Any) -> dict[str, xr.DataArray]:
        """Run accumulators over this scan (see :func:`bexa.core.reductions.reduce`)."""
        from bexa.core.reductions import reduce

        return reduce(self, accumulators, **kwargs)

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
    """Instantiate the engine of ``spec`` for ``path``; several scans become a MultiScanSource."""
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
    coords = energies if stack_dim == "energy" else scans
    order = np.argsort(coords) if stack_dim == "energy" else np.arange(len(parts))
    parts = [parts[i] for i in order]
    coords = [coords[i] for i in order]
    return MultiScanSource(parts, stack_dim, coords)


def open(
    path: str | Path,
    *,
    format: str | None = None,
    scan: int | Sequence[int] | tuple[int, int] | None = None,
    detector: str | None = None,
    overrides: dict[str, Any] | None = None,
    energy_keV: float | None = None,
    geometry: Any = None,
    crystal: Any = None,
    cache: Cache | bool | None = True,
    **kwargs: Any,
) -> Scan:
    """Open a scan from a path, sniffing the format unless ``format`` names a spec.

    Parameters
    ----------
    path
        Dataset folder, master file, ``scanNNNN`` folder, run folder or cube file.
    scan
        One scan number, ``(start, end)`` or a list; ``None`` opens the only scan
        or, for a dataset with several, all of them as a series.
    detector
        Detector name when the layout has several.
    energy_keV
        Override the energy derived from the monochromator.
    """
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
    return Scan(source, spec, geometry=geometry, crystal=crystal, cache=cache)


def open_profile(
    profile: str | Path | BeamtimeProfile,
    sample: str | None = None,
    scan: int | Sequence[int] | tuple[int, int] | None = None,
    detector: str | None = None,
    dataset: str | None = None,
    **kwargs: Any,
) -> Scan:
    """Open a scan described by a beamtime profile (the everyday entry point)."""
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
    return Scan(
        source,
        spec,
        geometry=geometry,
        crystal=crystal,
        cache=cache,
        profile=prof,
        name=f"{prof.name}/{dataset}",
        hkl_center=None if hkl_center is None else (hkl_center[0], hkl_center[1], hkl_center[2]),
    )
