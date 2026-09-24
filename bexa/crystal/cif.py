"""Read cell parameters and space-group information from CIF files.

pymatgen gives full symmetry when installed; without it a small parser reads
the cell and the space-group symbol, which is all hkl conversions and the
centering rules need.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from bexa._log import get_logger
from bexa.crystal.lattice import Crystal, LatticeParameters

log = get_logger(__name__)

_NUMBER = re.compile(r"^\s*(_cell_(?:length|angle)_[a-z]+)\s+([-+0-9.eE]+)")
_SYMBOL = re.compile(
    r"^\s*(_symmetry_space_group_name_H-M|_space_group_name_H-M_alt)\s+'?\"?([^'\"\n]+)"
)


@dataclass
class CifInfo:
    """What was read from the file."""

    path: Path
    lattice: LatticeParameters
    space_group: str | None = None
    centering: str = "P"
    formula: str | None = None


def read_cif(path: str | Path) -> CifInfo:
    """Cell parameters and space group of a CIF file (first data block)."""
    path = Path(path)
    values: dict[str, float] = {}
    symbol: str | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _NUMBER.match(line)
        if m:
            values[m.group(1)] = float(m.group(2).split("(")[0])
            continue
        m = _SYMBOL.match(line)
        if m and symbol is None:
            symbol = m.group(2).strip()
    needed = ("_cell_length_a", "_cell_length_b", "_cell_length_c")
    if not all(k in values for k in needed):
        raise ValueError(f"{path} has no complete _cell_length_ entries")
    lattice = LatticeParameters(
        values["_cell_length_a"],
        values["_cell_length_b"],
        values["_cell_length_c"],
        values.get("_cell_angle_alpha", 90.0),
        values.get("_cell_angle_beta", 90.0),
        values.get("_cell_angle_gamma", 90.0),
    )
    centering = symbol.strip()[0].upper() if symbol else "P"
    if centering not in "PIFABCR":
        centering = "P"
    return CifInfo(path, lattice, symbol, centering)


def crystal_from_cif(path: str | Path, use_pymatgen: bool = True) -> Crystal:
    """A :class:`Crystal` from a CIF file, via pymatgen when available."""
    path = Path(path)
    if use_pymatgen:
        try:
            from pymatgen.core import Structure
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

            structure = Structure.from_file(str(path))
            lat = structure.lattice
            lattice = LatticeParameters(lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma)
            try:
                analyzer = SpacegroupAnalyzer(structure)
                symbol = analyzer.get_space_group_symbol()
                system = analyzer.get_crystal_system()
            except Exception:
                symbol, system = None, None
            centering = symbol[0].upper() if symbol and symbol[0].upper() in "PIFABCR" else "P"
            return Crystal(lattice, bravais=system, centering=centering, name=path.stem)
        except ImportError:
            log.debug("pymatgen not installed; using the built-in CIF parser")
        except Exception as exc:
            log.warning("pymatgen could not read %s (%s); using the built-in parser", path, exc)
    info = read_cif(path)
    return Crystal(info.lattice, centering=info.centering, name=path.stem)
