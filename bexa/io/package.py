"""Scan packages: one scan in one HDF5 file, its frames with every motor and counter.

A scan read from any layout bexa knows (an APS 7-ID SPEC session with a TIFF per point, a
BLISS master with its detector files, a PAL-XFEL run, a series of scans) is written by
:func:`write_package` to one self-contained HDF5 file, which ``bexa.open`` reads back as the
same scan (spec ``bexa_package``, engine ``package``): the same dims, coordinates and frame
map, the per-frame motors and counters, the fixed positioners, the energy, the status, the
detector geometry and the side tables of the source, and the frames as the engine read
them. A folder of packages opens as a dataset. The groups are NeXus ones and the arrays
plain, so h5web, silx or h5py read a package without bexa::

    /                               attrs: bexa_package = "scan-v1", scan_number, name, frames
    /entry                          NXentry: title, scan_type, status
    /entry/data/frames              (n_frames, height, width), one chunk per frame
    /entry/measurement/<channel>    (n_frames,): the per-frame motors and counters
    /entry/structure/frame_index    the grid of frame ids (-1 where a frame is missing), with
                                    coords/<dim> and the grid's attrs (dims, units, plan, ...)
    /entry/instrument/positioners/  the motors that stayed put
    /entry/instrument/detector      the detector's name and geometry
    /entry/tables/<table>/<column>  the source's side tables: every point of an APS scan,
                                    frames or not, its beamline state; PAL-XFEL shot tables
    /entry/source                   the spec, engine, recipe and files it was read from

The frames are compressed with gzip at level 1 after a byte shuffle by default, which every
HDF5 library reads; on EIGER frames that is 10 times smaller than raw at 280 MB/s.
``"bitshuffle"`` (LZ4, the EIGER's own format) writes three times as fast and needs
``hdf5plugin`` to read, ``"lzf"`` is for h5py alone and ``None`` stores the frames raw.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from bexa._log import get_logger, interactive_session, progress
from bexa.core.provenance import build_attrs, file_records, fingerprint, to_json
from bexa.core.status import ScanNotReadyError
from bexa.io.cube import _read_attrs, _write_attrs
from bexa.io.formats import safe_format
from bexa.io.h5files import open_h5

if TYPE_CHECKING:
    import pandas as pd

    from bexa.core.dataset import Dataset
    from bexa.core.scan import Scan
    from bexa.core.structure import Structure

log = get_logger(__name__)

__all__ = [
    "BATCH_BYTES",
    "COMPRESSIONS",
    "DEFAULT_NAME",
    "FRAMES",
    "PACKAGE_MARKER",
    "PACKAGE_VERSION",
    "write_package",
    "write_packages",
]

PACKAGE_MARKER = "bexa_package"  # the root attribute that says what the file is ...
PACKAGE_VERSION = "scan-v1"  # ... and which layout
FRAMES = "entry/data/frames"
COMPRESSIONS = ("gzip", "bitshuffle", "lzf", None)
DEFAULT_NAME = "{stem}_S{scan:03d}.h5"  # write_packages: wse2_1_1_S004.h5, as the TIFFs are named
BATCH_BYTES = 64 * 2**20  # frames read per step; with two steps read ahead, under 200 MB in all


# ------------------------------------------------------------------------- writing
def write_package(
    scan: Scan,
    path: str | Path,
    *,
    compression: str | None = "gzip",
    overwrite: bool = False,
    show_progress: bool | None = None,
) -> Path:
    """Write ``scan`` to one HDF5 file that ``bexa.open`` reads back as the same scan.

    ``scan`` is any :class:`bexa.Scan`. The frames are copied as the engine reads them, in
    their stored dtype (an EIGER frame stays uint32, its flagged pixels read as 0) and in
    steps of :data:`BATCH_BYTES`, the next step read while the last one is compressed; a
    scan without frames (an APS session copied without its images) keeps its motors and
    counters in the tables. The file is written as ``<path>.partial`` and takes its name when
    complete. ``FileExistsError`` when ``path`` exists and ``overwrite`` is off; a scan with no
    data row yet raises :class:`bexa.ScanNotReadyError`.

    Parameters
    ----------
    compression
        ``"gzip"`` (level 1 after a byte shuffle: any HDF5 reader opens it), ``"bitshuffle"``
        (LZ4, three times as fast; readers need ``hdf5plugin``), ``"lzf"`` (h5py only) or
        ``None``.
    show_progress
        A progress bar over the frames; default: in an interactive session.
    """
    path = Path(path)
    filters = _frame_filters(compression)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
    structure = scan.structure  # a scan without a data row raises here, before a file exists
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    try:
        with h5py.File(partial, "w") as f:
            _write_header(f, scan, structure, compression)
            _write_frames(f, scan, filters, show_progress)
            _write_grid(f["entry"], structure)
            _write_instrument(f["entry"], scan, structure)
            _write_tables(f["entry"], scan.source)
            _write_source(f["entry"], scan)
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return path


def write_packages(
    data: Dataset | str | Path,
    out_dir: str | Path,
    scans: Any = None,
    *,
    name: str = DEFAULT_NAME,
    compression: str | None = "gzip",
    overwrite: bool = False,
    show_progress: bool | None = None,
    **open_kwargs: Any,
) -> list[Path]:
    """One package per scan of a dataset: an APS session, a BLISS dataset, any folder bexa reads.

    ``data`` is a :class:`bexa.Dataset` or what :func:`bexa.open_dataset` opens (with
    ``open_kwargs``); ``scans`` an id, an inclusive ``(first, last)`` or a list, by default
    every scan of the table. ``name`` is the file name in ``out_dir``, with the fields
    ``{stem}`` (the session: the SPEC file's stem, else the scan's name), ``{scan}`` (the scan
    number) and ``{id}`` (the table id). A package already there is kept while its scan is
    unchanged (the same files and rows), else written again, so a second run during a
    beamtime writes only the new scans and those that grew; ``overwrite=True`` writes them all.
    A scan with nothing to read (no data row) is left out with a warning. Returns the package
    of every scan, in table order.
    """
    from bexa.core.dataset import Dataset, open_dataset

    ds = data if isinstance(data, Dataset) else open_dataset(data, cache=False, **open_kwargs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids = ds.select_numbers(scans, status="any")  # frames or not: the counters are worth keeping
    shown = interactive_session() if show_progress is None else show_progress
    done: list[Path] = []
    for scan_id in progress(ids, total=len(ids), desc="packages", enabled=shown):
        try:
            scan = ds.scan(scan_id)
            number = ds.origin(scan_id)[1]
            target = out / safe_format(name, stem=_stem(scan), scan=number, id=scan_id)
            if target.exists() and not overwrite and _is_current(target, scan):
                done.append(target)
                continue
            done.append(
                write_package(
                    scan, target, compression=compression, overwrite=True, show_progress=False
                )
            )
        except ScanNotReadyError as exc:
            log.warning("scan %s has no package: %s", scan_id, exc)
    return done


def _frame_filters(compression: str | None) -> dict[str, Any]:
    """The ``create_dataset`` keywords of a compression name (see :data:`COMPRESSIONS`)."""
    if compression is None:
        return {}
    if compression == "gzip":
        return {"compression": "gzip", "compression_opts": 1, "shuffle": True}
    if compression == "lzf":
        return {"compression": "lzf", "shuffle": True}
    if compression == "bitshuffle":
        import hdf5plugin

        return dict(hdf5plugin.Bitshuffle(cname="lz4"))
    raise ValueError(f"unknown compression {compression!r}; choose from {COMPRESSIONS}")


def _write_header(f: h5py.File, scan: Scan, structure: Structure, compression: str | None) -> None:
    number = _scan_number(scan)
    _write_attrs(
        f,
        {
            PACKAGE_MARKER: PACKAGE_VERSION,
            **build_attrs(parameters={"compression": compression}),
            "scan_number": -1 if number is None else number,
            "name": scan.name,
            "title": structure.title,
            "frames": int(structure.n_frames),
            "source_fingerprint": _fingerprint(scan),  # how write_packages tells a scan changed
        },
    )
    f.attrs["NX_class"] = "NXroot"
    f.attrs["default"] = "entry"
    entry = f.create_group("entry")
    entry.attrs["NX_class"] = "NXentry"
    entry.attrs["default"] = "data"
    _write_attrs(entry, {"scan_type": structure.scan_type, "status": str(scan.status)})
    entry.create_dataset("title", data=structure.title)


def _write_frames(
    f: h5py.File, scan: Scan, filters: dict[str, Any], show_progress: bool | None
) -> None:
    from bexa.core.parallel import Prefetcher

    source = scan.source
    n, (h, w) = int(source.n_frames), source.frame_shape
    data = f.create_group("entry/data")
    data.attrs["NX_class"] = "NXdata"
    data.attrs["signal"] = "frames"
    data.attrs["axes"] = [".", ".", "."]  # acquisition order: the grid is in /entry/structure
    frames = data.create_dataset(
        "frames", (n, h, w), dtype=source.dtype, maxshape=(None, h, w), chunks=(1, h, w), **filters
    )
    frames.attrs["interpretation"] = "image"
    flagged = _flagged_value(scan)
    if flagged is not None:  # the pixels the detector flagged with it were read, and stored, as 0
        frames.attrs["flagged_pixel_value"] = flagged
    if n == 0:
        return
    step = max(1, BATCH_BYTES // max(1, h * w * np.dtype(source.dtype).itemsize))
    starts = range(0, n, step)

    def blocks() -> Iterator[tuple[int, np.ndarray]]:
        for start in starts:
            yield start, source.read_frames(slice(start, min(start + step, n)))

    shown = interactive_session() if show_progress is None else show_progress
    ahead = Prefetcher(blocks(), depth=2)  # reading the files overlaps the compression
    for start, block in progress(ahead, total=len(starts), desc="frames", enabled=shown):
        frames[start : start + len(block)] = block


def _write_grid(entry: h5py.Group, s: Structure) -> None:
    group = entry.create_group("structure")
    group.attrs["NX_class"] = "NXcollection"
    _write_attrs(
        group,
        {
            "motor_dims": list(s.motor_dims),
            "motor_shape": [int(n) for n in s.motor_shape],
            "frame_shape": [int(n) for n in s.frame_shape],
            "n_frames": int(s.n_frames),
            "scan_type": s.scan_type,
            "title": s.title,
            "units": dict(s.units),
            "energy_keV": s.energy_keV,
            "planned_shape": None if s.planned_shape is None else [int(n) for n in s.planned_shape],
            "planned_frames": s.planned_frames,
            "motor_names": sorted(s.motor_names),
        },
    )
    group.create_dataset("frame_index", data=np.asarray(s.frame_index, dtype=np.int64))
    coords = group.create_group("coords")
    for dim in s.motor_dims:
        _put_array(coords, dim, s.coords[dim])
    measurement = entry.create_group("measurement")
    measurement.attrs["NX_class"] = "NXcollection"
    for channel, values in s.per_frame.items():
        _put_array(measurement, channel, values)


def _write_instrument(entry: h5py.Group, scan: Scan, s: Structure) -> None:
    instrument = entry.create_group("instrument")
    instrument.attrs["NX_class"] = "NXinstrument"
    positioners = instrument.create_group("positioners")
    positioners.attrs["NX_class"] = "NXcollection"
    for motor, value in s.scalars.items():
        _put_array(positioners, motor, value)
    detector = instrument.create_group("detector")
    detector.attrs["NX_class"] = "NXdetector"
    geometry = scan.geometry  # a DetectorGeometry, or whatever bexa.open(geometry=...) was given
    known: dict[str, Any] = {}
    if dataclasses.is_dataclass(geometry) and not isinstance(geometry, type):
        known = dataclasses.asdict(geometry)
    fields = {k: v for k, v in known.items() if v is not None and k != "extra"}
    _write_attrs(detector, {"name": _detector_name(scan), "geometry": fields})
    if fields.get("pixel_size_um") is not None:  # NeXus readers look for these
        for axis in ("x_pixel_size", "y_pixel_size"):
            detector.create_dataset(axis, data=float(fields["pixel_size_um"])).attrs["units"] = "um"
    if fields.get("distance_mm") is not None:
        detector.create_dataset("distance", data=float(fields["distance_mm"])).attrs["units"] = "mm"


def _write_tables(entry: h5py.Group, source: Any) -> None:
    group = entry.create_group("tables")
    group.attrs["NX_class"] = "NXcollection"
    read = getattr(source, "aux_tables", None)  # a series of scans has none
    if read is None:
        return
    try:
        tables = read()
    except Exception as exc:  # an engine's side table that fails must not cost the frames
        log.warning("the side tables of the scan could not be read (%s): the package has none", exc)
        return
    for table_name, table in tables.items():
        sub = group.create_group(_key(str(table_name)))
        sub.attrs["name"] = str(table_name)
        columns = list(dict.fromkeys(str(c) for c in table.columns))  # a repeated label once
        _write_attrs(sub, {"columns": columns})
        for label in columns:
            position = [str(c) for c in table.columns].index(label)
            _put_array(sub, label, _column_values(table.iloc[:, position]))


def _write_source(entry: h5py.Group, scan: Scan) -> None:
    group = entry.create_group("source")
    group.attrs["NX_class"] = "NXcollection"
    spec = scan.spec
    _write_attrs(
        group,
        {
            "format": spec.name if spec is not None else None,
            "engine": spec.engine if spec is not None else None,
            "recipe": scan.recipe,
            "name": scan.name,
        },
    )
    # one record per file read (an APS scan has a TIFF per point): a dataset, not an attribute
    group.create_dataset("files", data=to_json(file_records(scan.source.files())))


# ------------------------------------------------------------------ what a scan is
def _scan_number(scan: Scan) -> int | None:
    value = getattr(scan.source, "scan", None)  # every engine of numbered scans sets it
    if isinstance(value, (int, np.integer)):
        return int(value)
    recorded = (scan.recipe or {}).get("scan")
    return int(recorded) if isinstance(recorded, (int, np.integer)) else None


def _stem(scan: Scan) -> str:
    """The session a scan belongs to, as a file name: the SPEC file's stem, else the scan's name."""
    spec_file = getattr(scan.source, "spec_file", None)
    stem = Path(spec_file).stem if spec_file is not None else scan.name
    return re.sub(r"[^\w.-]+", "_", stem).strip("_") or "scan"


def _detector_name(scan: Scan) -> str | None:
    name = getattr(scan.source, "detector", None)
    if name is None and scan.geometry is not None:
        name = scan.geometry.detector
    return None if name is None else str(name)


def _flagged_value(scan: Scan) -> int | None:
    name = _detector_name(scan)
    if scan.spec is None or name is None:
        return None
    value = scan.spec.detector_info(name).get("invalid_value")
    return None if value is None else int(value)


def _fingerprint(scan: Scan) -> str:
    records = getattr(scan.source, "cache_records", None)
    return fingerprint(records() if callable(records) else scan.source.files())


def _is_current(path: Path, scan: Scan) -> bool:
    """True when the package at ``path`` was written from the scan as it is now."""
    try:
        with open_h5(path, retry=False) as f:
            stored = _read_attrs(f)
    except OSError:
        return False
    same_layout = stored.get(PACKAGE_MARKER) == PACKAGE_VERSION
    return same_layout and stored.get("source_fingerprint") == _fingerprint(scan)


# ------------------------------------------------------------------------ arrays
def _key(name: str) -> str:
    """An HDF5 member name for any text: ``/`` (the path separator) and ``%`` escaped."""
    return name.replace("%", "%25").replace("/", "%2F") or "%00"


def _name(key: str) -> str:
    return "" if key == "%00" else key.replace("%2F", "/").replace("%25", "%")


def _put_array(group: h5py.Group, name: str, values: Any) -> h5py.Dataset:
    """Store ``values`` under ``name``: text as UTF-8, times as int64 nanoseconds."""
    data = np.asarray(values)
    kind = None
    if data.dtype.kind == "M":
        kind = "datetime64[ns]"
        data = data.astype("datetime64[ns]").astype(np.int64)
    if data.dtype.kind in "OUS":
        text = np.asarray([str(v) for v in data.ravel()], dtype=object).reshape(data.shape)
        node = group.create_dataset(_key(name), data=text, dtype=h5py.string_dtype())
    else:
        node = group.create_dataset(_key(name), data=data)
    if _key(name) != name:
        node.attrs["name"] = name
    if kind is not None:
        node.attrs["kind"] = kind
    return node


def _get_array(node: h5py.Dataset) -> Any:
    """What :func:`_put_array` stored: an array, or a scalar for a 0-d dataset."""
    data = node.asstr()[()] if h5py.check_string_dtype(node.dtype) is not None else node[()]
    if node.attrs.get("kind") == "datetime64[ns]":
        data = np.asarray(data).astype("datetime64[ns]")
    return data


def _name_of(node: h5py.Dataset | h5py.Group) -> str:
    stored = node.attrs.get("name")
    if stored is not None:
        return stored.decode() if isinstance(stored, bytes) else str(stored)
    return _name(node.name.rsplit("/", 1)[-1])


def _column_values(column: pd.Series) -> np.ndarray:
    """A table column as numbers, booleans, times or text (nullable numbers as floats with NaN)."""
    import pandas as pd

    if pd.api.types.is_bool_dtype(column) and not column.isna().any():
        return column.to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(column):
        values = column.to_numpy()
        return values if values.dtype.kind in "iuf" else column.to_numpy(float, na_value=np.nan)
    if pd.api.types.is_datetime64_any_dtype(column):
        return column.to_numpy(dtype="datetime64[ns]")
    return column.astype(str).to_numpy(dtype=object)
