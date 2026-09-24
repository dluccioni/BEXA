"""Engine for single-file scans with spec-like scan commands (PAL-XFEL, April 2025).

One HDF5 per scan: the scan group carries ``scanMode`` and a spec-like
``scanHistory`` attribute (``a1scan``/``a2scan`` with one ``name start stop
npoints`` block per motor), ``motor/mN`` hold the motor positions, and every
``det/<name>/data`` holds one summed image per motor position. Aborted scans
are shorter than the command says; the missing points are marked in the
structure rather than padded with NaN, as ``genCubepalApril2025.py`` did.
"""

from __future__ import annotations

import contextlib
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from bexa._log import get_logger
from bexa.core.registry import register_engine
from bexa.core.structure import Structure
from bexa.io.base import BaseSource
from bexa.io.formats import FormatSpec

log = get_logger(__name__)

SCAN_GROUP_RE = re.compile(r"scan(\d+)")


@dataclass
class ScanMotor:
    """One ``name start stop npoints`` block of a scan command."""

    name: str
    start: float
    stop: float
    npoints: int


@dataclass
class ScanCommand:
    """A parsed spec-like scan command."""

    command: str
    motors: list[ScanMotor]
    text: str

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(m.npoints for m in self.motors)


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _decode(value[()])
    return str(value)


def parse_scan_command(
    text: str,
    one_motor: tuple[str, ...] = ("a1scan", "sscan"),
    two_motor: tuple[str, ...] = ("a2scan",),
) -> ScanCommand:
    """Parse ``... a2scan m1 s1 e1 n1 m2 s2 e2 n2 time`` into its motors.

    The command word may be preceded by other tokens (a scan number, a
    timestamp), so the first known command word is used.
    """
    tokens = text.split()
    known = [i for i, token in enumerate(tokens) if token in one_motor or token in two_motor]
    if not known:
        raise ValueError(f"no known scan command in {text!r}")
    i = known[0]
    n_motors = 1 if tokens[i] in one_motor else 2
    motors = []
    pos = i + 1
    for _ in range(n_motors):
        try:
            name, start, stop, npoints = tokens[pos : pos + 4]
        except ValueError:
            raise ValueError(f"incomplete motor block in {text!r}") from None
        motors.append(ScanMotor(name, float(start), float(stop), int(float(npoints))))
        pos += 4
    return ScanCommand(tokens[i], motors, text)


@register_engine("spec_h5")
class SpecH5Source(BaseSource):
    """Summed images of one spec-style scan file as a frame stack.

    Parameters
    ----------
    spec
        Format spec (``pal_xfel_spec_2025_04`` or one derived from it).
    path
        The scan file.
    scan
        Scan number of the ``run/scanNNNN`` group.
    detector
        Name of the ``det`` subgroup to read; default: the first one holding images.
    """

    name = "spec_h5"
    spec: FormatSpec

    def __init__(
        self,
        spec: FormatSpec,
        path: str | Path,
        scan: int = 1,
        detector: str | None = None,
        energy_keV: float | None = None,
    ) -> None:
        super().__init__(spec)
        self.path = Path(path)
        self.scan = int(scan)
        self._energy_override = energy_keV
        self.scan_group = spec.path("scan_group", scan=self.scan)
        self.motor_group = spec.path("motor_group", scan=self.scan)
        self.det_group = spec.path("det_group", scan=self.scan)
        self.i0_dataset = str(spec.layout.get("i0_dataset", "ohqbpm2_totalsum"))
        self._local = threading.local()
        self._structure: Structure | None = None
        with h5py.File(self.path, "r") as f:
            self._discover(f, detector)

    # ----------------------------------------------------------- discovery
    @classmethod
    def from_path(
        cls, spec: FormatSpec, path: str | Path, scan: int | None = None, **kwargs: Any
    ) -> SpecH5Source:
        """Build a source from the scan file (or a folder holding one)."""
        path = Path(path)
        if path.is_dir():
            files = sorted(path.glob("*.h5"))
            if not files:
                raise FileNotFoundError(f"no .h5 file in {path}")
            path = files[0]
        if scan is None:
            scans = cls.list_scans(spec, path)
            if not scans:
                raise ValueError(f"{path} has no scan groups")
            scan = scans[0]
        return cls(spec, path, scan, **kwargs)

    @classmethod
    def list_scans(cls, spec: FormatSpec, path: str | Path) -> list[int]:
        """Scan numbers of the ``scanNNNN`` groups in the file."""
        parent = spec.path("scan_group", scan=1).rsplit("/", 1)[0]
        with h5py.File(path, "r") as f:
            if parent not in f:
                return []
            found = []
            for key in f[parent]:
                m = SCAN_GROUP_RE.fullmatch(key)
                if m:
                    found.append(int(m.group(1)))
        return sorted(found)

    def _discover(self, f: h5py.File, detector: str | None) -> None:
        if self.scan_group not in f:
            raise KeyError(f"{self.path} has no group {self.scan_group!r}")
        group = f[self.scan_group]
        attrs = list(group.attrs)
        mode_key = self.spec.resolve_key("scan_mode", attrs)
        command_key = self.spec.resolve_key("scan_command", attrs)
        if command_key is None:
            raise KeyError(f"{self.scan_group} has none of {self.spec.aliases('scan_command')}")
        self.scan_mode = _decode(group.attrs[mode_key]) if mode_key else ""
        options = self.spec.options
        self.command = parse_scan_command(
            _decode(group.attrs[command_key]),
            tuple(options.get("one_motor_commands", ("a1scan", "sscan"))),
            tuple(options.get("two_motor_commands", ("a2scan",))),
        )
        self.coords: dict[str, np.ndarray] = {}
        for i, motor in enumerate(self.command.motors):
            key = f"{self.motor_group}/m{i + 1}"
            if key in f:
                self.coords[motor.name] = np.asarray(f[key][()], dtype=float).ravel()
            else:
                self.coords[motor.name] = np.linspace(motor.start, motor.stop, motor.npoints)
        self._n_total = int(np.prod([len(v) for v in self.coords.values()]))

        det = f[self.det_group]
        if detector is None:
            for key in det:
                node = det[key]
                is_image_group = isinstance(node, h5py.Group) and "data" in node
                if key != self.i0_dataset and is_image_group and node["data"].ndim >= 3:
                    detector = key
                    break
            if detector is None:
                raise KeyError(f"no image dataset under {self.det_group}")
        self.detector = detector
        self._frames_key = f"{self.det_group}/{detector}/data"
        ds = f[self._frames_key]
        if ds.ndim == 4:
            self._inner: int | None = int(ds.shape[1])
            n_recorded = int(ds.shape[0]) * int(ds.shape[1])
        elif ds.ndim == 3:
            self._inner = None
            n_recorded = int(ds.shape[0])
        else:
            raise ValueError(f"{self._frames_key} has {ds.ndim} dims; expected 3 or 4")
        self._frame_shape = (int(ds.shape[-2]), int(ds.shape[-1]))
        self._dtype = np.dtype(ds.dtype)
        self._n_frames = min(n_recorded, self._n_total)
        if n_recorded < self._n_total:
            log.info(
                "%s scan %d: %d of %d points recorded (partial scan)",
                self.path.name,
                self.scan,
                n_recorded,
                self._n_total,
            )
        i0_key = f"{self.det_group}/{self.i0_dataset}/data"
        i0 = np.full(self._n_frames, np.nan)
        if i0_key in f:
            values = np.asarray(f[i0_key][()], dtype=float).ravel()
            m = min(len(values), self._n_frames)
            i0[:m] = values[:m]
        self._i0 = i0

    # ---------------------------------------------------------- properties
    def files(self) -> list[Path]:
        return [self.path]

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
        return None if self._energy_override is None else float(self._energy_override)

    @property
    def is_partial(self) -> bool:
        return self._n_frames < self._n_total

    # ------------------------------------------------------------- reading
    def _handle(self) -> h5py.File:
        f = getattr(self._local, "handle", None)
        if f is None or not f.id.valid:
            f = self._local.handle = h5py.File(self.path, "r")
        return f

    def _grid_segments(self, start: int, stop: int) -> list[tuple[int, int, int]]:
        """``(row, col_start, col_stop)`` pieces of flat ids ``[start, stop)`` on the 2-D grid."""
        inner = self._inner or 1
        r0, c0 = divmod(start, inner)
        r1, c1 = divmod(stop - 1, inner)
        segments = []
        for r in range(r0, r1 + 1):
            cs = c0 if r == r0 else 0
            ce = c1 + 1 if r == r1 else inner
            segments.append((r, cs, ce))
        return segments

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
        if ids.size == 0:
            return out
        ds = self._handle()[self._frames_key]
        pos = 0
        for start, stop in self.contiguous_runs(ids):
            if self._inner is None:
                count = stop - start
                out[pos : pos + count] = ds[start:stop, y, x]
                pos += count
                continue
            for row, cs, ce in self._grid_segments(start, stop):
                count = ce - cs
                out[pos : pos + count] = ds[row, cs:ce, y, x]
                pos += count
        return out

    def close(self) -> None:
        f = getattr(self._local, "handle", None)
        if f is not None:
            with contextlib.suppress(Exception):
                f.close()
            self._local.handle = None

    def refresh(self) -> SpecH5Source:
        self.close()
        with h5py.File(self.path, "r") as f:
            self._discover(f, self.detector)
        self._structure = None
        return self

    # ----------------------------------------------------------- structure
    def structure(self) -> Structure:
        if self._structure is None:
            names = [m.name for m in self.command.motors]
            self._structure = Structure.from_grid(
                names,
                self.coords,
                self._frame_shape,
                n_frames=self._n_frames,
                per_frame={"i0": self._i0},
                scan_type=self.command.command,
                energy_keV=self.energy_keV,
            )
        return self._structure
