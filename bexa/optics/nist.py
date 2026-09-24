"""NIST XCOM mass attenuation tables: fetching them once and reading them back.

Port of ``Database_Extraction.py``. :func:`fetch_nist_tables` downloads the
per-element tables and densities (needs ``requests`` and ``beautifulsoup4``
and the network); :func:`load_nist_tables` reads the two files it writes, and
:func:`nist_mass_attenuation` interpolates them on a log-log grid.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np

from bexa._log import get_logger

log = get_logger(__name__)

URL_ELEMENT = "https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z{Z:02d}.html"
URL_DENSITIES = "https://physics.nist.gov/PhysRefData/XrayMassCoef/tab1.html"
XCOM_FILE = "nist_xcom.dat"
DENSITY_FILE = "element_densities.dat"
SYMBOLS = [  # Z = 1 ... 92
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
    "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
    "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",
]  # fmt: skip
_DATA_LINE = re.compile(r"^[0-9.]+E[+\-][0-9]{2}\s+[0-9.]+E[+\-][0-9]{2}")

__all__ = ["fetch_nist_tables", "load_nist_tables", "nist_mass_attenuation"]


def fetch_nist_tables(out_dir: str | Path, delay_s: float = 0.2) -> tuple[Path, Path]:
    """Download the XCOM tables and densities into ``out_dir`` (one-time, network)."""
    import requests
    from bs4 import BeautifulSoup

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xcom_path, density_path = out_dir / XCOM_FILE, out_dir / DENSITY_FILE
    with density_path.open("w", encoding="utf-8") as rho:
        rho.write("# symbol  density_g_per_cm3\n")
        soup = BeautifulSoup(requests.get(URL_DENSITIES, timeout=15).text, "html.parser")
        for row in soup.find_all("tr"):
            cols = row.text.strip().split()
            if len(cols) >= 5 and cols[0].isdigit():
                rho.write(f"{cols[2].rstrip(','):<3s} {cols[4]}\n")
    with xcom_path.open("w", encoding="utf-8") as xcom:
        xcom.write("# NIST XCOM mass-attenuation coefficients\n")
        xcom.write("#   #S  Z <atomic-number>  <symbol>\n#       Energy_eV   mu_rho_cm2_per_g\n")
        for Z, symbol in enumerate(SYMBOLS, start=1):
            try:
                html = requests.get(URL_ELEMENT.format(Z=Z), timeout=15).text
            except Exception as exc:
                log.warning("Z=%d %s: %s", Z, symbol, exc)
                continue
            block = BeautifulSoup(html, "html.parser").find("pre")
            if block is None:
                log.warning("Z=%d %s: no data block", Z, symbol)
                continue
            rows = [ln.strip() for ln in block.text.splitlines() if _DATA_LINE.match(ln.strip())]
            if not rows:
                continue
            xcom.write(f"#S  Z {Z}  {symbol}\n")
            for ln in rows:
                e_mev, mu = ln.split()[:2]
                xcom.write(f"{float(e_mev) * 1e6:11.5E} {float(mu):11.3E}\n")
            time.sleep(delay_s)
    return xcom_path, density_path


def load_nist_tables(folder: str | Path) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """``({symbol: (energy_eV, mu_rho) array}, {symbol: density})`` from the two files."""
    folder = Path(folder)
    tables: dict[str, list[tuple[float, float]]] = {}
    current = None
    for line in (folder / XCOM_FILE).read_text(encoding="utf-8").splitlines():
        if line.startswith("#S"):
            current = line.split()[-1]
            tables[current] = []
        elif line and not line.startswith("#") and current is not None:
            e_ev, mu = line.split()[:2]
            tables[current].append((float(e_ev), float(mu)))
    densities: dict[str, float] = {}
    for line in (folder / DENSITY_FILE).read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        symbol, rho = line.split()[:2]
        densities[symbol] = float(rho)
    return {k: np.array(v) for k, v in tables.items()}, densities


def nist_mass_attenuation(table: np.ndarray, energy_keV: np.ndarray | float) -> np.ndarray:
    """Interpolate a ``(energy_eV, mu_rho)`` table log-log at ``energy_keV``."""
    energy_eV = np.asarray(energy_keV, dtype=float) * 1e3
    return np.exp(np.interp(np.log(energy_eV), np.log(table[:, 0]), np.log(table[:, 1])))
