"""Frame stacks from plain files: ``.npy``, ``.npz``, tiff stacks and ``file.h5::/path``.

For data that has no beamline layout: a saved array, a folder of tiffs or one
HDF5 dataset. An optional coordinates file (``.csv`` with one column per
motor, ``.json`` mapping motor to values, or ``.npy``) turns the frame
sequence into a motor grid.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from bexa.core.registry import register_engine
from bexa.core.structure import Structure
from bexa.io.base import BaseSource
from bexa.io.formats import FormatSpec

TIFF_SUFFIXES = (".tif", ".tiff")
H5_SUFFIXES = (".h5", ".hdf5", ".nxs")
URL_SEPARATOR = "::"

__all__ = ["GenericStackSource", "load_array", "read_coords", "split_h5_url"]


def split_h5_url(text: str | Path) -> tuple[Path, str | None]:
    """Split ``"file.h5::/entry/data"`` into file and dataset path (``None`` without ``::``)."""
    text = str(text)
    if URL_SEPARATOR in text:
        file_part, _, data_path = text.partition(URL_SEPARATOR)
        # a pathlib round trip on Windows turns the dataset path's slashes into backslashes
        data_path = data_path.removeprefix("path=").replace("\\", "/")
        return Path(file_part), data_path
    return Path(text), None


def _read_tiff(path: Path) -> np.ndarray:
    try:
        import tifffile

        return np.asarray(tifffile.imread(str(path)))
    except ImportError:
        from PIL import Image

        with Image.open(path) as im:
            frames = []
            for i in range(getattr(im, "n_frames", 1)):
                im.seek(i)
                frames.append(np.asarray(im))
        return np.stack(frames) if len(frames) > 1 else frames[0]


def _first_h5_dataset(path: Path) -> str:
    import h5py

    found: list[tuple[int, str]] = []
    with h5py.File(path, "r") as f:

        def visit(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
                found.append((int(np.prod(obj.shape)), name))

        f.visititems(visit)
    if not found:
        raise KeyError(f"{path} holds no 2-D or 3-D dataset")
    return max(found)[1]


def load_array(source: str | Path, mmap: bool = True) -> np.ndarray:
    """Load ``.npy``/``.npz``, a tiff (stack) or folder of tiffs, or ``file.h5::/path``.

    HDF5 datasets are read into memory; ``.npy`` files are memory-mapped when
    ``mmap`` is true. A folder of tiffs becomes one stack in name order.
    """
    path, data_path = split_h5_url(source)
    if data_path is not None or path.suffix.lower() in H5_SUFFIXES:
        import h5py

        with h5py.File(path, "r") as f:
            key = data_path or _first_h5_dataset(path)
            return np.asarray(f[key][()])
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in TIFF_SUFFIXES)
        if not files:
            raise FileNotFoundError(f"no tiff files in {path}")
        return np.stack([_read_tiff(p) for p in files])
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, mmap_mode="r" if mmap else None)
    if suffix == ".npz":
        with np.load(path) as archive:
            return np.asarray(archive[archive.files[0]])
    if suffix in TIFF_SUFFIXES:
        return _read_tiff(path)
    raise ValueError(f"cannot load {path}: unknown extension {suffix!r}")


def read_coords(path: str | Path) -> dict[str, np.ndarray]:
    """Per-frame motor values from ``.csv`` (header row), ``.json`` (name -> list) or ``.npy``."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): np.asarray(v, dtype=float) for k, v in raw.items()}
    if suffix == ".npy":
        data = np.load(path)
        if data.ndim == 1:
            return {"axis": data.astype(float)}
        return {f"axis{i}": data[:, i].astype(float) for i in range(data.shape[1])}
    import pandas as pd

    table = pd.read_csv(path)
    return {str(c): table[c].to_numpy(dtype=float) for c in table.columns}


@register_engine("generic")
class GenericStackSource(BaseSource):
    """A frame stack from a plain file with optional motor coordinates.

    Parameters
    ----------
    path
        ``.npy``, ``.npz``, tiff file or folder, or ``file.h5::/dataset``.
    coords
        Coordinates file (see :func:`read_coords`) or a mapping of per-frame values.
    """

    name = "generic"

    def __init__(
        self,
        spec: FormatSpec | None,
        path: str | Path,
        coords: str | Path | dict[str, Any] | None = None,
        energy_keV: float | None = None,
    ) -> None:
        super().__init__(spec)
        self.path = Path(str(path).split(URL_SEPARATOR)[0])
        self.source = str(path)
        self._energy = energy_keV
        data = load_array(path)
        if data.ndim == 2:
            data = data[None]
        elif data.ndim > 3:
            data = data.reshape(-1, *data.shape[-2:])
        self._frames = data
        self._coords: dict[str, np.ndarray] = {}
        if isinstance(coords, dict):
            self._coords = {k: np.asarray(v, dtype=float) for k, v in coords.items()}
        elif coords is not None:
            self._coords = read_coords(coords)
        self._structure: Structure | None = None

    @classmethod
    def from_path(
        cls, spec: FormatSpec, path: str | Path, scan: int | None = None, **kwargs: Any
    ) -> GenericStackSource:
        kwargs.pop("detector", None)
        return cls(spec, path, **kwargs)

    def files(self) -> list[Path]:
        return [self.path]

    @property
    def frame_shape(self) -> tuple[int, int]:
        return (int(self._frames.shape[1]), int(self._frames.shape[2]))

    @property
    def n_frames(self) -> int:
        return int(self._frames.shape[0])

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._frames.dtype)

    @property
    def energy_keV(self) -> float | None:
        return self._energy

    def read_frames(
        self,
        index: np.ndarray | slice,
        y: slice = slice(None),
        x: slice = slice(None),
        dtype: Any = None,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        ids = self.normalise_index(index, self.n_frames)
        block = self._frames[ids][:, y, x]
        if out is None:
            return block.astype(dtype) if dtype is not None else np.array(block)
        out[...] = block
        return out

    def structure(self) -> Structure:
        if self._structure is None:
            if self._coords:
                self._structure = Structure.from_per_frame(
                    self._coords, self.frame_shape, energy_keV=self._energy
                )
            else:
                self._structure = Structure.frames_only(
                    self.n_frames, self.frame_shape, energy_keV=self._energy
                )
        return self._structure
