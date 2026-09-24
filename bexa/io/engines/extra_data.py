"""Engine for European XFEL runs read through ``extra_data`` (optional extra).

Wraps ``extra_data.open_run`` and ``extra.components.Scan`` the way the
``Usefulthings`` notebook used them: the per-train images of an area detector
are the frames, the scanned motor (an alias such as ``ssry``) gives the
per-frame coordinate, and :meth:`ExtraDataSource.binned_steps` returns the
mean image per scan step minus a dark run. Importing this module without
``extra_data`` installed raises ``ImportError``, so the engine is simply
absent on machines without it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import extra_data
import numpy as np
import xarray as xr

from bexa._log import get_logger
from bexa.core.registry import register_engine
from bexa.core.structure import Structure
from bexa.io.base import BaseSource
from bexa.io.formats import FormatSpec

log = get_logger(__name__)

RUN_RE = re.compile(r"p?(\d+)[/:_-]r?(\d+)")

__all__ = ["ExtraDataSource", "parse_run_id"]


def parse_run_id(text: str | Path) -> tuple[int, int]:
    """``"6832/259"``, ``"p6832:r259"`` or ``"p6832_r0259"`` -> ``(proposal, run)``."""
    m = RUN_RE.search(str(text))
    if not m:
        raise ValueError(f"cannot read a proposal and run number from {text!r}")
    return int(m.group(1)), int(m.group(2))


@register_engine("extra_data")
class ExtraDataSource(BaseSource):
    """Per-train images of one EuXFEL run.

    Parameters
    ----------
    spec
        Format spec (``euxfel_extra_data``); its layout names the detector key,
        image key, scan alias and resolution, all overridable here.
    proposal, run
        The run to open (``data="all"``).
    """

    name = "extra_data"
    spec: FormatSpec

    def __init__(
        self,
        spec: FormatSpec,
        proposal: int,
        run: int,
        detector_key: str | None = None,
        image_key: str | None = None,
        scan_alias: str | None = None,
        resolution: float | None = None,
        energy_keV: float | None = None,
    ) -> None:
        super().__init__(spec)
        layout = spec.layout
        self.proposal, self.run = int(proposal), int(run)
        self.detector_key = detector_key or str(layout.get("detector_key"))
        self.image_key = image_key or str(layout.get("image_key", "data.image.pixels"))
        self.scan_alias = scan_alias or layout.get("scan_alias")
        self.resolution = resolution if resolution is not None else layout.get("scan_resolution")
        self._energy = energy_keV
        self.run_data = extra_data.open_run(proposal=self.proposal, run=self.run, data="all")
        self.images = self.run_data[self.detector_key, self.image_key]
        shape = tuple(int(v) for v in self.images.shape)
        self._n_frames = shape[0]
        self._frame_shape = (shape[-2], shape[-1])
        self._dtype = np.dtype(self.images.dtype)
        self._structure: Structure | None = None

    @classmethod
    def from_path(
        cls, spec: FormatSpec, path: str | Path, scan: int | None = None, **kwargs: Any
    ) -> ExtraDataSource:
        """``path`` names the run as ``proposal/run`` (EuXFEL folder names work too)."""
        proposal, run = parse_run_id(path)
        kwargs.pop("detector", None)
        return cls(spec, proposal, run, **kwargs)

    # ---------------------------------------------------------- properties
    def files(self) -> list[Path]:
        return [Path(f.filename) for f in getattr(self.run_data, "files", [])]

    @property
    def frame_shape(self) -> tuple[int, int]:
        return self._frame_shape

    @property
    def n_frames(self) -> int:
        return self._n_frames

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def energy_keV(self) -> float | None:
        return self._energy

    # ------------------------------------------------------------- reading
    def read_frames(
        self,
        index: np.ndarray | slice,
        y: slice = slice(None),
        x: slice = slice(None),
        dtype: Any = None,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        ids = self.normalise_index(index, self._n_frames)
        h, w = self._frame_shape
        ny = len(range(*y.indices(h)))
        nx = len(range(*x.indices(w)))
        target_dtype = np.dtype(dtype) if dtype is not None else self._dtype
        if out is None:
            out = np.empty((ids.size, ny, nx), dtype=target_dtype)
        pos = 0
        for start, stop in self.contiguous_runs(ids):
            block = self.images.select_trains(np.s_[start:stop]).ndarray()
            block = block.reshape(-1, h, w)[:, y, x]
            out[pos : pos + block.shape[0]] = block
            pos += block.shape[0]
        return out

    # ----------------------------------------------------------- structure
    def motor_values(self) -> np.ndarray | None:
        """Per-train value of the scanned motor alias (``None`` without a scan alias)."""
        if not self.scan_alias:
            return None
        values = np.asarray(self.run_data.alias[self.scan_alias].ndarray(), dtype=float).reshape(-1)
        if values.shape[0] != self._n_frames:
            log.warning(
                "%d motor values for %d frames; using the frame order",
                values.shape[0],
                self._n_frames,
            )
            return None
        return values

    def structure(self) -> Structure:
        if self._structure is None:
            values = self.motor_values()
            if values is not None:
                self._structure = Structure.from_per_frame(
                    {str(self.scan_alias): values},
                    self._frame_shape,
                    tolerance=self.resolution,
                    scan_type="scan",
                    energy_keV=self._energy,
                )
            else:
                self._structure = Structure.frames_only(
                    self._n_frames, self._frame_shape, energy_keV=self._energy
                )
        return self._structure

    def binned_steps(self, dark: Any | None = None) -> xr.DataArray:
        """Mean image per scan step minus ``dark`` (the notebook's ``obtain_data``)."""
        s = self.structure()
        if s.motor_dims == ("frame",):
            raise ValueError("no scan alias: the run has no steps to bin")
        name = s.motor_dims[0]
        steps = []
        for i in range(s.motor_shape[0]):
            ids = s.frame_ids((slice(i, i + 1),))
            frames = self.read_frames(ids, dtype=np.float64)
            steps.append(frames.mean(axis=0))
        data = np.stack(steps)
        if dark is not None:
            data = data - np.asarray(dark, dtype=float)
        return xr.DataArray(data, dims=(name, "y", "x"), coords={name: s.coords[name]})
