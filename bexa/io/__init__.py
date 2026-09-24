"""Reading and writing data.

- :mod:`bexa.io.formats`: format specs (YAML) and how a path is matched to one.
- :mod:`bexa.io.base`: the ``Source`` protocol every engine implements.
- :mod:`bexa.io.engines`: the engines, one per kind of file structure.
- :mod:`bexa.io.cube`: bexa's own reduced format, plus ``load``/``save``.
- :mod:`bexa.io.inspect`: HDF5 tree printing and file summaries.
- :mod:`bexa.io.generic`: tiff stacks, npy, ``"file.h5::/path"``.
- :mod:`bexa.io.darfix`: interoperability with the darfix package (extra).
"""

from bexa._log import get_logger

try:  # registers the Bitshuffle, LZ4 and Blosc filters of ESRF and XFEL detector files with HDF5
    import hdf5plugin  # noqa: F401
except ImportError:  # pragma: no cover - listed in requirements.txt
    get_logger(__name__).warning(
        "hdf5plugin is not installed: compressed detector frames cannot be read "
        "(pip install hdf5plugin)"
    )

from bexa.io.base import BaseSource, FrameBatch, Source
from bexa.io.formats import FormatSpec, list_specs, load_spec, match_spec

__all__ = [
    "BaseSource",
    "FormatSpec",
    "FrameBatch",
    "Source",
    "list_specs",
    "load_spec",
    "match_spec",
]
