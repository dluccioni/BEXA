"""Pydantic models for beamtime profiles and their sub-sections.

A beamtime profile names one format spec and holds everything specific to that
beamtime: data roots, samples, detector geometry, dark references, default
processing settings and named ROIs. It never contains HDF5 paths or key names;
those belong to the format spec (:mod:`bexa.io.formats`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Geometry(BaseModel):
    """Detector and beam geometry used for coordinate transforms and scalebars."""

    model_config = ConfigDict(extra="allow")

    energy_keV: float | None = None
    pixel_size_um: float | None = None
    distance_mm: float | None = None
    effective_pixel_nm: float | None = Field(
        default=None, description="Sample-plane pixel size after the objective magnification"
    )
    beam_center: tuple[float, float] | None = Field(
        default=None, description="(y, x) pixel of the direct beam"
    )
    two_theta_offset_deg: float = 0.0
    delta_offset_deg: float = 0.0
    convention: str | None = None


class Sample(BaseModel):
    """One sample (one dataset folder) measured during the beamtime."""

    model_config = ConfigDict(extra="allow")

    dataset: str
    cif: str | None = None
    hkl_center: tuple[int, int, int] | None = None
    detector: str | None = None
    notes: str = ""


class Defaults(BaseModel):
    """Processing defaults applied when a call does not say otherwise."""

    model_config = ConfigDict(extra="allow")

    precision: Literal["float32", "float64"] = "float32"
    downsample: list[int] | int | None = None
    device: str = "auto"
    batch_scans: int = 4
    parallel: bool = True
    memory_fraction: float | None = None
    apply_log: bool = False


class DarkReference(BaseModel):
    """Where the dark frames for a detector were recorded."""

    dataset: str
    scan: int = 1


class AutoProcessSettings(BaseModel):
    """Files used by the beamtime watcher."""

    roi_table: str | None = None
    skip_list: str | None = None
    outputs: str | None = None
    interval_s: float = 60.0


class BeamtimeProfile(BaseModel):
    """Everything specific to one beamtime; see ``configs/beamtimes`` for examples."""

    model_config = ConfigDict(extra="allow")

    name: str = ""
    format: str
    root: Path | None = None
    raw_root: Path | None = None
    processed_root: Path | None = None
    path_aliases: dict[str, str] = Field(default_factory=dict)
    samples: dict[str, Sample] = Field(default_factory=dict)
    detector: str | None = None
    geometry: Geometry = Field(default_factory=Geometry)
    darks: dict[str, DarkReference] = Field(default_factory=dict)
    defaults: Defaults = Field(default_factory=Defaults)
    rois: dict[str, dict[str, Any]] = Field(default_factory=dict)
    background_box: dict[str, list[int]] | None = None
    auto_process: AutoProcessSettings | None = None
    format_overrides: dict[str, Any] = Field(default_factory=dict)
    source_path: Path | None = None

    @field_validator("samples", mode="before")
    @classmethod
    def _plain_strings_are_datasets(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: ({"dataset": v} if isinstance(v, str) else v) for k, v in value.items()}
        return value

    def sample(self, name: str) -> Sample:
        """Look up a sample by name with a helpful error."""
        try:
            return self.samples[name]
        except KeyError:
            available = ", ".join(sorted(self.samples)) or "none"
            raise KeyError(
                f"unknown sample {name!r} in profile {self.name!r}; available: {available}"
            ) from None

    def data_root(self) -> Path:
        """``root`` for frame-stack beamlines, ``raw_root`` for point-file beamlines."""
        root = self.root or self.raw_root
        if root is None:
            raise ValueError(f"profile {self.name!r} defines neither 'root' nor 'raw_root'")
        return root
