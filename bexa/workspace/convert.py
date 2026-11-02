"""Direct conversions between data formats: scans, sessions, HDF5 datasets and files.

Frames are streamed batch by batch (never a whole scan in memory) into the
target: a bexa scan package, a bexa cube, a multi-page TIFF, a folder of TIFFs,
an NPY file or NeXus HDF5. Every frame target gets a ``*_positions.csv`` with
the frame number, its grid indices and every per-frame channel, so nothing of
the motor information is lost.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bexa._log import get_logger, progress
from bexa.core.roi import ROI

log = get_logger(__name__)

__all__ = [
    "DATASET_TARGETS",
    "SCAN_TARGETS",
    "convert_dataset",
    "convert_file",
    "convert_scan",
    "convert_session",
    "positions_table",
]

SCAN_TARGETS = {
    "package": "bexa scan package (.h5): frames, motors, counters, geometry; reopens as the scan",
    "cube": "bexa cube (.h5): the frames on their motor grid with coordinates (fits in memory)",
    "nexus": "NeXus HDF5: the frames (n, y, x) as NXdata with every motor per frame",
    "tiff_stack": "one multi-page TIFF of the frames, in acquisition order",
    "tiff_folder": "a folder with one TIFF per frame",
    "npy": "one NPY array (n, y, x) of the frames",
}
DATASET_TARGETS = ("npy", "tiff_stack", "tiff_folder", "csv")
BLOCK_BYTES = 64 * 2**20  # an HDF5 dataset is copied in blocks of about this size


def positions_table(scan: Any) -> pd.DataFrame:
    """One row per frame: its number, grid indices and every per-frame channel."""
    s = scan.structure
    table: dict[str, Any] = {"frame": np.arange(s.n_frames)}
    grid = s.grid_of_frame()
    for k, dim in enumerate(s.motor_dims):
        table[f"index_{dim}"] = grid[:, k]
    for name, values in s.per_frame.items():
        values = np.asarray(values)
        if values.shape[:1] == (s.n_frames,) and values.ndim == 1:
            table[name] = values
    return pd.DataFrame(table)


def _target_path(out: Path, suffix: str) -> Path:
    return out if out.suffix else out.with_suffix(suffix)


def _filter(compression: str | None) -> str | None:
    """The package filter of a form choice: ``"none"`` means no compression."""
    return None if compression in (None, "", "none") else compression


def convert_scan(
    scan: Any,
    target: str,
    out: str | Path,
    roi: ROI | None = None,
    downsample: int | None = None,
    compression: str = "gzip",
    overwrite: bool = False,
) -> list[Path]:
    """Write one scan as ``target`` (see :data:`SCAN_TARGETS`); returns the files written.

    ``roi`` and ``downsample`` cut and bin the frames on the way (not for a
    package, which keeps the scan whole).
    """
    if target not in SCAN_TARGETS:
        raise ValueError(f"target must be one of {list(SCAN_TARGETS)}")
    out = Path(out)
    if target == "package":
        from bexa.io.package import write_package

        return [
            write_package(
                scan,
                _target_path(out, ".h5"),
                compression=_filter(compression),
                overwrite=overwrite,
            )
        ]
    ds = None if not downsample else (1,) * len(scan.dims[:-2]) + (int(downsample),) * 2
    if target == "cube":
        from bexa.io.cube import save

        path = _target_path(out, ".h5")
        _refuse_existing(path, overwrite)
        volume = scan.read(roi=roi, downsample=ds)
        return [save(volume.to_dataset(name="frames"), path)]
    written = []
    if target == "tiff_folder":
        folder = out if not out.suffix else out.with_suffix("")
        folder.mkdir(parents=True, exist_ok=True)
        positions = folder / "positions.csv"
    else:
        suffix = {"tiff_stack": ".tif", "npy": ".npy", "nexus": ".h5"}[target]
        path = _target_path(out, suffix)
        _refuse_existing(path, overwrite)
        path.parent.mkdir(parents=True, exist_ok=True)
        positions = path.with_name(path.stem + "_positions.csv")
    batches = scan.batches(roi=roi, downsample=ds, dtype=np.dtype(scan.source.dtype))
    n = scan.n_frames
    if target == "tiff_folder":
        written = _tiff_folder(batches, folder, scan.name, n)
    elif target == "tiff_stack":
        written = [_tiff_stack(batches, path, n)]
    elif target == "npy":
        written = [_npy(batches, path, n)]
    else:
        written = [_nexus_frames(batches, path, n, scan, compression)]
    positions_table(scan).to_csv(positions, index=False)
    return [*written, positions]


def _refuse_existing(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; tick Overwrite to replace it")


def _tiff_folder(batches: Any, folder: Path, stem: str, n: int) -> list[Path]:
    import tifffile

    written = []
    name = str(stem).replace("/", "_").replace(" ", "_")
    for batch in progress(batches, desc="frames", enabled=False):
        frames = np.asarray(batch.frames)
        for frame_id, frame in zip(batch.frame_ids, frames, strict=True):
            path = folder / f"{name}_{int(frame_id):05d}.tif"
            tifffile.imwrite(path, frame)
            written.append(path)
    return written


def _tiff_stack(batches: Any, path: Path, n: int) -> Path:
    import tifffile

    with tifffile.TiffWriter(path, bigtiff=True) as writer:
        for batch in progress(batches, desc="frames", enabled=False):
            for frame in np.asarray(batch.frames):
                writer.write(frame, contiguous=True)
    return path


def _npy(batches: Any, path: Path, n: int) -> Path:
    array = None
    for batch in progress(batches, desc="frames", enabled=False):
        frames = np.asarray(batch.frames)
        if array is None:
            array = np.lib.format.open_memmap(
                path, mode="w+", dtype=frames.dtype, shape=(n, *frames.shape[1:])
            )
        array[batch.frame_ids] = frames
    if array is not None:
        array.flush()
        del array
    return path


def _nexus_frames(batches: Any, path: Path, n: int, scan: Any, compression: str) -> Path:
    import h5py

    table = positions_table(scan)
    with h5py.File(path, "w") as f:
        f.attrs["default"] = "entry"
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.attrs["title"] = scan.structure.title
        data = entry.create_group("data")
        data.attrs.update({"NX_class": "NXdata", "signal": "frames", "axes": ["frame", ".", "."]})
        dataset = None
        for batch in progress(batches, desc="frames", enabled=False):
            frames = np.asarray(batch.frames)
            if dataset is None:
                dataset = data.create_dataset(
                    "frames",
                    shape=(n, *frames.shape[1:]),
                    dtype=frames.dtype,
                    chunks=(1, *frames.shape[1:]),
                    compression=None if compression in ("none", "", None) else "gzip",
                    shuffle=compression not in ("none", "", None),
                )
            dataset[batch.frame_ids] = frames
        data.create_dataset("frame", data=table["frame"].to_numpy())
        positioners = entry.create_group("positioners")
        positioners.attrs["NX_class"] = "NXpositioner"
        for column in table.columns:
            positioners.create_dataset(column, data=table[column].to_numpy())
    return path


def convert_session(
    path: str | Path,
    out_dir: str | Path,
    scans: Sequence[int] | None = None,
    target: str = "package",
    compression: str = "gzip",
    overwrite: bool = False,
    status: str = "finished",
) -> list[Path]:
    """Every scan (or the ``scans`` named) of a dataset or session written as ``target``.

    Packages are written by :func:`bexa.io.package.write_packages`, which skips
    the scans whose package is already up to date.
    """
    from bexa.core.dataset import open_dataset

    dataset = open_dataset(path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if target == "package":
        from bexa.io.package import write_packages

        return write_packages(
            dataset, out, scans, compression=_filter(compression), overwrite=overwrite
        )
    ids = dataset.select_numbers(scans, status=status)
    files: list[Path] = []
    for scan_id in progress(ids, total=len(ids), desc="scans", enabled=False):
        scan = dataset.scan(scan_id)
        stem = f"{str(dataset.name).replace(' ', '_')}_S{int(scan_id):03d}"
        files.extend(
            convert_scan(scan, target, out / stem, compression=compression, overwrite=overwrite)
        )
    return files


def convert_dataset(
    path: str | Path,
    dataset: str,
    target: str,
    out: str | Path,
    index: str | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """One dataset of any HDF5 file written as ``target`` (npy, tiff_stack, tiff_folder, csv).

    Large datasets are copied block by block along their first axis;
    ``index`` (``"0:100, :, :"``) cuts them first.
    """
    from bexa.io.h5files import open_h5
    from bexa.workspace.files import parse_index

    if target not in DATASET_TARGETS:
        raise ValueError(f"target must be one of {DATASET_TARGETS}")
    out = Path(out)
    part = parse_index(index)
    with open_h5(path) as f:
        node = f[dataset]
        if part is not None:
            values = np.asarray(node[part])
            blocks = [values] if values.ndim else [values.reshape(1)]
            shape, dtype = values.shape, values.dtype
        else:
            shape, dtype = node.shape, node.dtype
            blocks = _blocks(node)
        if target == "csv":
            path_out = _target_path(out, ".csv")
            _refuse_existing(path_out, overwrite)
            data = np.concatenate([np.asarray(b) for b in blocks]) if part is None else blocks[0]
            if data.ndim > 2:
                raise ValueError("CSV holds 1-D or 2-D data; pick a slice (index) first")
            pd.DataFrame(np.atleast_2d(data).T if data.ndim == 1 else data).to_csv(
                path_out, index=False
            )
            return [path_out]
        if target == "npy":
            path_out = _target_path(out, ".npy")
            _refuse_existing(path_out, overwrite)
            array = np.lib.format.open_memmap(path_out, mode="w+", dtype=dtype, shape=tuple(shape))
            start = 0
            for block in blocks:
                block = np.asarray(block)
                array[start : start + len(block)] = block
                start += len(block)
            array.flush()
            del array
            return [path_out]
        import tifffile

        if target == "tiff_stack":
            path_out = _target_path(out, ".tif")
            _refuse_existing(path_out, overwrite)
            with tifffile.TiffWriter(path_out, bigtiff=True) as writer:
                for block in blocks:
                    for frame in np.asarray(block).reshape(-1, *shape[-2:]):
                        writer.write(frame, contiguous=True)
            return [path_out]
        folder = out if not out.suffix else out.with_suffix("")
        folder.mkdir(parents=True, exist_ok=True)
        written, k = [], 0
        for block in blocks:
            for frame in np.asarray(block).reshape(-1, *shape[-2:]):
                target_file = folder / f"{Path(dataset).name}_{k:05d}.tif"
                tifffile.imwrite(target_file, frame)
                written.append(target_file)
                k += 1
        return written


def _blocks(node: Any) -> Any:
    """The dataset in blocks along its first axis, about :data:`BLOCK_BYTES` each."""
    if node.ndim == 0:
        yield np.asarray(node[()]).reshape(1)
        return
    row_bytes = int(np.prod(node.shape[1:], dtype=np.int64)) * node.dtype.itemsize or 1
    step = max(1, BLOCK_BYTES // row_bytes)
    for start in progress(
        range(0, node.shape[0], step), total=-(-node.shape[0] // step), desc="blocks", enabled=False
    ):
        yield node[start : start + step]


def convert_file(path: str | Path, target: str, out: str | Path) -> list[Path]:
    """A file bexa loads as objects (npy, TIFF, cube, CSV, picture) written as ``target``
    (an export format of :func:`bexa.workspace.files.export_object`), one file per object."""
    from bexa.workspace.files import export_object, load_file

    objects = load_file(path)
    out = Path(out)
    if len(objects) == 1:
        return [
            export_object(objects[0], out if out.suffix else out.with_suffix(f".{target}"), target)
        ]
    folder = out if not out.suffix else out.with_suffix("")
    return [export_object(o, folder / f"{o.name}.{target}", target) for o in objects]
