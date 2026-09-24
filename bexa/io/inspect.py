"""Look inside files: HDF5 trees with shapes, chunks and compression, and path summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

_H5_SUFFIXES = (".h5", ".hdf5", ".nxs", ".nx")


def _filters(dataset: Any) -> str:
    parts: list[str] = []
    if dataset.compression:
        parts.append(str(dataset.compression))
    try:
        plist = dataset.id.get_create_plist()
        for i in range(plist.get_nfilters()):
            code = plist.get_filter(i)[0]
            if code == 32008:
                parts.append("bitshuffle")
            elif code == 32004:
                parts.append("lz4")
            elif code == 32001:
                parts.append("blosc")
            elif code == 32013:
                parts.append("zfp")
    except Exception:
        pass
    return "+".join(dict.fromkeys(parts)) if parts else "none"


def h5_tree(
    path: str | Path, max_depth: int | None = None, attrs: bool = False, max_items: int = 2000
) -> list[str]:
    """Lines describing every group and dataset of an HDF5 file.

    Datasets show shape, dtype, chunking, compression filters and the
    compression ratio (stored bytes vs uncompressed).
    """
    import h5py

    lines: list[str] = []

    def visit(name: str, obj: Any) -> Any:
        depth = name.count("/")
        if max_depth is not None and depth > max_depth:
            return None
        indent = "  " * depth
        label = name.rsplit("/", 1)[-1]
        if isinstance(obj, h5py.Group):
            lines.append(f"{indent}{label}/")
        else:
            nbytes = (
                int(np.prod(obj.shape)) * obj.dtype.itemsize if obj.shape else obj.dtype.itemsize
            )
            try:
                stored = obj.id.get_storage_size()
            except Exception:
                stored = 0
            ratio = f" ratio {nbytes / stored:.1f}x" if stored else ""
            chunks = f" chunks {obj.chunks}" if obj.chunks else ""
            lines.append(
                f"{indent}{label}  {obj.shape} {obj.dtype}{chunks} [{_filters(obj)}]{ratio}"
            )
        if attrs and len(obj.attrs):
            for key, value in obj.attrs.items():
                text = value.decode() if isinstance(value, bytes) else str(value)
                lines.append(f"{indent}  @{key} = {text[:80]}")
        if len(lines) >= max_items:
            lines.append("... (truncated)")
            return True
        return None

    with h5py.File(path, "r") as f:
        lines.append(f"{Path(path).name}/")
        f.visititems(visit)
    return lines


def print_h5_structure(path: str | Path, **kwargs: Any) -> None:
    """Print :func:`h5_tree` (the helper from the lab's ``scratch.md``, kept for habit)."""
    print("\n".join(h5_tree(path, **kwargs)))


def describe(path: str | Path, max_depth: int | None = 3) -> str:
    """Human summary of a path: matching format spec, folder listing or HDF5 tree."""
    from bexa.io.formats import match_spec

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    lines: list[str] = [f"path: {path}"]
    scored = match_spec(path)
    if scored and scored[0][0] >= 1.0:
        lines.append(f"format: {scored[0][1].name} (engine {scored[0][1].engine})")
    elif scored:
        best = ", ".join(f"{s.name} {sc:.0%}" for sc, s in scored[:3])
        lines.append(f"format: no full match; closest: {best}")
    else:
        lines.append("format: no spec matched")

    if path.is_dir():
        entries = sorted(path.iterdir())
        lines.append(f"folder with {len(entries)} entries:")
        for entry in entries[:40]:
            size = f" ({entry.stat().st_size / 1e6:.1f} MB)" if entry.is_file() else "/"
            lines.append(f"  {entry.name}{size}")
        if len(entries) > 40:
            lines.append(f"  ... {len(entries) - 40} more")
    elif path.suffix in _H5_SUFFIXES:
        lines.extend(h5_tree(path, max_depth=max_depth))
    elif path.suffix == ".npy":
        arr = np.load(path, mmap_mode="r")
        lines.append(f"npy array {arr.shape} {arr.dtype} ({arr.nbytes / 1e9:.2f} GB)")
    else:
        lines.append(f"file ({path.stat().st_size / 1e6:.1f} MB)")
    return "\n".join(lines)
