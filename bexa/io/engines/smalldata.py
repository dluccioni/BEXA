"""Engine for LCLS smalldata HDF5 files (one file per run, one row per event).

A smalldata file holds per-event scalars (``lightStatus/laser``, ``ipm4/sum``,
``scan/...``) and, for the detectors the producer was told to save, per-event
images: one ROI as ``<detector>/ROI_area``, several as ``<detector>/ROI_0_area``,
``ROI_1_area``, ..., the whole detector as ``<detector>/full_area``. Run sums sit
under ``Sums/`` and each detector's settings (pixel size, ROI bounds, masks)
under ``UserDataCfg/<detector>``. The per-event images of one detector are the
frames of this source; the scalars are their per-frame channels, resolved
through the format spec's key aliases. A run without images (point detectors
only) has no frames: :func:`read_scalars` reads its scalars as a table.
"""

from __future__ import annotations

import contextlib
import re
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from bexa._log import get_logger
from bexa.core.registry import register_engine
from bexa.core.structure import Structure
from bexa.io.base import BaseSource
from bexa.io.formats import FormatSpec, load_spec, safe_format
from bexa.io.h5files import open_h5

if TYPE_CHECKING:
    import pandas as pd

log = get_logger(__name__)

__all__ = [
    "DEFAULT_FRAMES",
    "GAP_BYTES",
    "SCATTERED",
    "SPAN_BYTES",
    "NoImagesError",
    "SmalldataSource",
    "read_scalars",
]

# the per-event image datasets of a detector when the spec's layout names none
DEFAULT_FRAMES = ("{detector}/ROI_0_area", "{detector}/ROI_area", "{detector}/full_area")
# reading in file order: the fraction of events stored before their predecessor
# above which the chunks count as scattered, the largest single read, and the
# gap between chunks still read through rather than skipped with a seek
SCATTERED = 0.1
SPAN_BYTES = 64 * 2**20
GAP_BYTES = 2**20
_RUN = re.compile(r"_Run(\d+)\.h5$")


class NoImagesError(LookupError):
    """A smalldata run without per-event images; :func:`read_scalars` reads what it holds."""


def _spans(rows: np.ndarray) -> Iterator[tuple[int, int, np.ndarray]]:
    """``(begin, end, rows)`` byte ranges that read the chunk ``rows`` (by offset) in few calls."""
    i = 0
    while i < len(rows):
        begin = int(rows[i, 0])
        j = i
        while (
            j + 1 < len(rows)
            and rows[j + 1, 0] - (rows[j, 0] + rows[j, 1]) < GAP_BYTES
            and rows[j + 1, 0] + rows[j + 1, 1] - begin <= SPAN_BYTES
        ):
            j += 1
        yield begin, int(rows[j, 0] + rows[j, 1]), rows[i : j + 1]
        i = j + 1


def _as_spec(spec: FormatSpec | str | None) -> FormatSpec:
    if isinstance(spec, FormatSpec):
        return spec
    return load_spec(spec or "lcls_smalldata")


def _image_templates(spec: FormatSpec, detector: str) -> list[str]:
    """Where the images of ``detector`` may be: its own ``frames_path``, else the layout's."""
    own = spec.detector_info(detector).get("frames_path")
    templates = own or spec.layout.get("frames", DEFAULT_FRAMES)
    templates = [templates] if isinstance(templates, str) else list(templates)
    return [safe_format(t, detector=detector) for t in templates]


def _image_datasets(f: h5py.File, spec: FormatSpec) -> dict[str, str]:
    """``{detector: image dataset}`` for every top-level group holding ``(event, y, x)`` images.

    The spec's ``detectors`` come first, in their order; the others follow by name.
    """
    found: dict[str, str] = {}
    for name, group in f.items():
        if not isinstance(group, h5py.Group):
            continue
        for path in _image_templates(spec, name):
            ds = f.get(path)
            if isinstance(ds, h5py.Dataset) and ds.ndim == 3:
                found[name] = path
                break
    preferred = [d for d in spec.detectors if d in found]
    return {d: found[d] for d in preferred + sorted(set(found) - set(preferred))}


def _n_events(f: h5py.File, spec: FormatSpec) -> int:
    for name in spec.aliases("event_time"):
        ds = f.get(name)
        if isinstance(ds, h5py.Dataset) and ds.ndim == 1:
            return int(ds.shape[0])
    raise ValueError(f"{f.filename} has no event_time dataset to count the events by")


def _resolve_columns(f: h5py.File, spec: FormatSpec, n_events: int) -> dict[str, str]:
    """``{logical key: dataset}`` of the spec's keys found among the 1-D per-event datasets."""
    available: list[str] = []

    def visit(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset) and obj.ndim == 1 and obj.shape[0] == n_events:
            available.append(name)

    f.visititems(visit)
    out: dict[str, str] = {}
    for key in spec.keys:
        column = spec.resolve_key(key, available)
        if column is not None:
            out[key] = column
    return out


def read_scalars(
    path: str | Path, spec: FormatSpec | str | None = None, extra: Sequence[str] = ()
) -> pd.DataFrame:
    """The per-event scalars of a smalldata run as a table, with or without images in the file.

    One row per event and one column per logical key of the spec that the file
    holds (``laser_flag``, ``i0``, ``scanvar``, ``delay``, ...), then one column per
    dataset of ``extra``, named by its path (``"snd_dio/sum"``, ``"evr/code_41"``).
    ``spec`` is a spec or its name (default ``lcls_smalldata``); a beamtime
    profile's comes from ``BeamtimeProfile.format_spec()``.
    """
    import pandas as pd

    spec = _as_spec(spec)
    with open_h5(path) as f:
        n = _n_events(f, spec)
        data = {key: np.asarray(f[col][()]) for key, col in _resolve_columns(f, spec, n).items()}
        for name in extra:
            ds = f.get(name)
            if not isinstance(ds, h5py.Dataset):
                raise KeyError(f"{Path(path).name} has no dataset {name!r}")
            if ds.shape != (n,):
                raise ValueError(f"{name} has shape {ds.shape}; a per-event scalar has ({n},)")
            data[name] = np.asarray(ds[()])
    return pd.DataFrame(data)


@register_engine("smalldata")
class SmalldataSource(BaseSource):
    """Per-event images and scalars of one smalldata run file.

    Parameters
    ----------
    spec
        Format spec (``lcls_smalldata``).
    path
        The ``<experiment>_RunNNNN.h5`` file.
    detector
        The detector whose images are the frames (``"jungfrau1M"``), or the image
        dataset itself (``"jungfrau1M/ROI_1_area"``); default: the first of
        :attr:`detectors`.
    energy_keV
        Photon energy, instead of the median ``photon_energy`` of the file.
    """

    name = "smalldata"
    spec: FormatSpec

    def __init__(
        self,
        spec: FormatSpec,
        path: str | Path,
        detector: str | None = None,
        energy_keV: float | None = None,
    ) -> None:
        super().__init__(spec)
        self.path = Path(path)
        match = _RUN.search(self.path.name)
        self.run = int(match.group(1)) if match else None
        self._energy_override = energy_keV
        self._local = threading.local()
        self._structure: Structure | None = None
        self._chunk_rows: np.ndarray | None = None  # file-order chunk table, when scattered
        self._chunk_shape = (0, 0)
        self._chunks_checked = False
        with open_h5(self.path) as f:
            self.detectors = _image_datasets(f, spec)
            self.detector, self.frames_path = self._pick_detector(f, detector)
            ds = f[self.frames_path]
            self._n_frames = int(ds.shape[0])
            self._frame_shape = (int(ds.shape[1]), int(ds.shape[2]))
            self._dtype = np.dtype(ds.dtype)
            self._chunks = tuple(ds.chunks) if ds.chunks else None
            self.columns = _resolve_columns(f, spec, self._n_frames)
            self.pixel_size_um, self.roi_bound = self._detector_config(f)

    def _pick_detector(self, f: h5py.File, detector: str | None) -> tuple[str, str]:
        if detector is not None and "/" in detector:  # the image dataset itself
            ds = f.get(detector)
            if not isinstance(ds, h5py.Dataset) or ds.ndim != 3:
                raise KeyError(f"{self.path.name} has no (event, y, x) dataset {detector!r}")
            return detector.split("/", 1)[0], detector
        if not self.detectors:
            raise NoImagesError(
                f"{self.path.name} holds no per-event images (no detector wrote ROIs or full "
                "images); bexa.io.engines.smalldata.read_scalars(path) reads its scalars"
            )
        if detector is None:
            detector = next(iter(self.detectors))
        if detector not in self.detectors:
            have = ", ".join(self.detectors)
            raise KeyError(
                f"{self.path.name} has no per-event images of {detector!r}; detectors: {have}"
            )
        return detector, self.detectors[detector]

    def _detector_config(self, f: h5py.File) -> tuple[float | None, list[list[int]] | None]:
        """The pixel size (um) and ROI bounds of the detector: the file's, else the spec's size."""
        pixel = self.spec.detector_info(self.detector).get("pixel_size_um")
        bound = None
        template = self.spec.layout.get("detector_config", "UserDataCfg/{detector}")
        group = f.get(safe_format(template, detector=self.detector))
        if isinstance(group, h5py.Group):
            if "pixelsize" in group:
                size = float(np.ravel(group["pixelsize"][()])[0])
                pixel = size * 1e6 if size < 1 else size  # smalldata_tools writes metres
            roi = self.frames_path.rsplit("/", 1)[-1].removesuffix("_area")  # ROI, ROI_0, full
            key = f"{roi}__{roi}_bound"
            if key in group:
                bound = np.asarray(group[key][()]).astype(int).tolist()
        return (None if pixel is None else float(pixel)), bound

    @classmethod
    def from_path(
        cls, spec: FormatSpec, path: str | Path, scan: int | None = None, **kwargs: Any
    ) -> SmalldataSource:
        """The run file ``path``, or run ``scan`` of the folder ``path``."""
        path = Path(path)
        if path.is_dir():
            path = cls._run_file(spec, path, scan)
        return cls(spec, path, **kwargs)

    @staticmethod
    def _run_file(spec: FormatSpec, folder: Path, run: int | None) -> Path:
        files = sorted(folder.glob("*.h5"))
        runs = sorted(int(m.group(1)) for f in files if (m := _RUN.search(f.name)))
        if run is not None:
            pattern = safe_format(spec.layout.get("run_file", "*_Run{run:04d}.h5"), run=int(run))
            matches = sorted(folder.glob(pattern))
            if not matches:
                raise FileNotFoundError(f"no run {run} in {folder}; runs there: {runs or 'none'}")
            return matches[0]
        if len(files) == 1:
            return files[0]
        if not files:
            raise FileNotFoundError(f"no .h5 file in {folder}")
        raise LookupError(f"{folder} holds {len(files)} run files (runs {runs}); give scan=<run>")

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
    def chunk_frames(self) -> int:
        return int(self._chunks[0]) if self._chunks else 1

    @property
    def energy_keV(self) -> float | None:
        if self._energy_override is not None:
            return float(self._energy_override)
        column = self.columns.get("photon_energy")
        if column is None:
            return None
        with open_h5(self.path) as f:
            values = np.asarray(f[column][()], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return None
        median = float(np.median(finite))
        return median / 1e3 if median > 1e3 else median  # eV in the file, keV here

    # ------------------------------------------------------------- reading
    def _handle(self) -> h5py.File:
        f = getattr(self._local, "handle", None)
        if f is None or not f.id.valid:
            f = self._local.handle = open_h5(self.path)
        return f

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
        ds = self._handle()[self.frames_path]
        pos = 0
        for start, stop in self.contiguous_runs(ids):
            count = stop - start
            out[pos : pos + count] = ds[start:stop, y, x]
            pos += count
        return out

    def iter_frames(
        self, index: np.ndarray | slice = slice(None), batch: int = 64
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(frame_ids, frames)`` blocks of whole frames, in the order of the file.

        MPI producers can write the chunks of a run out of event order; read by
        event, such a file is mostly seeks (100 to 200 MB/s on a network share).
        Then the chunks are read the way they lie, in long sequential spans,
        straight from the file (they are uncompressed). Images whose chunks are
        in event order, compressed or several events per chunk are read by
        frame id, which is fastest for them.
        """
        ids = self.normalise_index(index, self._n_frames)
        table = self._scattered_chunks()
        if table is None:
            yield from super().iter_frames(ids, batch)
            return
        wanted = np.zeros(self._n_frames, dtype=bool)
        wanted[ids] = True
        rows = table[wanted[table[:, 2]]]  # the chunks of the frames asked for, in file order
        _, first = np.unique(rows[:, 2], return_index=True)
        order = rows[np.sort(first), 2]  # frames by where their first chunk lies
        h, w = self._frame_shape
        cy, cx = self._chunk_shape
        slot = np.zeros(self._n_frames, dtype=np.int64)
        with open(self.path, "rb", buffering=0) as fh:
            for start in range(0, order.size, batch):
                block = order[start : start + batch]
                slot[block] = np.arange(block.size)
                frames = np.empty((block.size, h, w), dtype=self._dtype)
                for begin, end, chunk_rows in _spans(rows[np.isin(rows[:, 2], block)]):
                    fh.seek(begin)
                    buffer = fh.read(end - begin)
                    for offset, _, event, y0, x0 in chunk_rows:
                        flat = np.frombuffer(buffer, self._dtype, cy * cx, int(offset - begin))
                        tile = flat.reshape(cy, cx)[: h - y0, : w - x0]
                        frames[slot[event], y0 : y0 + tile.shape[0], x0 : x0 + tile.shape[1]] = tile
                yield block, frames

    def _scattered_chunks(self) -> np.ndarray | None:
        """``(byte offset, size, event, y0, x0)`` of the image chunks in file order, or ``None``.

        Only for images in uncompressed chunks of one event that are out of
        event order in the file (more than :data:`SCATTERED` of the events start
        before the one ahead of them); ``None`` otherwise, and with an h5py too
        old for ``chunk_iter``.
        """
        if not self._chunks_checked:
            self._chunks_checked = True
            ds = self._handle()[self.frames_path]
            chunks = ds.chunks
            raw = (
                chunks is not None
                and chunks[0] == 1
                and ds.id.get_create_plist().get_nfilters() == 0
                and ds.dtype.isnative
                and hasattr(ds.id, "chunk_iter")
            )
            rows: list[tuple[int, ...]] = []
            if raw:
                ds.id.chunk_iter(lambda c: rows.append((c.byte_offset, c.size, *c.chunk_offset)))
            per_frame = -(-ds.shape[1] // chunks[1]) * -(-ds.shape[2] // chunks[2]) if raw else 0
            if raw and len(rows) == ds.shape[0] * per_frame:
                table = np.array(rows, dtype=np.int64)
                corners = table[(table[:, 3] == 0) & (table[:, 4] == 0)]  # one per event
                corners = corners[np.argsort(corners[:, 2])]
                if np.mean(np.diff(corners[:, 0]) < 0) > SCATTERED:
                    self._chunk_rows = table[np.argsort(table[:, 0], kind="stable")]
                    self._chunk_shape = (int(chunks[1]), int(chunks[2]))
        return self._chunk_rows

    def close(self) -> None:
        f = getattr(self._local, "handle", None)
        if f is not None:
            with contextlib.suppress(Exception):
                f.close()
            self._local.handle = None

    def sums(self) -> dict[str, np.ndarray]:
        """The run-level summed images under ``Sums/``."""
        out: dict[str, np.ndarray] = {}
        with open_h5(self.path) as f:
            if "Sums" in f:
                for name, node in f["Sums"].items():
                    if isinstance(node, h5py.Dataset):
                        out[name] = np.asarray(node[()])
        return out

    def scalars_table(self, extra: Sequence[str] = ()) -> pd.DataFrame:
        """Every resolved per-event scalar, plus the datasets of ``extra``, as a DataFrame."""
        return read_scalars(self.path, self.spec, extra)

    # ----------------------------------------------------------- structure
    def structure(self) -> Structure:
        if self._structure is None:
            with open_h5(self.path) as f:
                per_frame = {}
                for key, column in self.columns.items():
                    values = np.asarray(f[column][()])
                    if values.dtype.kind in "fiub":
                        per_frame[key] = values.astype(float)
            axis = self.spec.resolve_key("scanvar", list(per_frame)) if per_frame else None
            scan_axis = next(
                (k for k in self.spec.axis_priority if k in per_frame and np.ptp(per_frame[k]) > 0),
                axis,
            )
            self._structure = Structure.frames_only(
                self._n_frames,
                self._frame_shape,
                per_frame=per_frame,
                scan_type="events",
                energy_keV=self.energy_keV,
            )
            if scan_axis:
                self._structure.scalars["scan_axis"] = float("nan")
                self._structure.units["scan_axis"] = scan_axis
        return self._structure
