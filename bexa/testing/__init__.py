"""Synthetic data for tests, examples and benchmarks.

Every format spec has a generator here that writes tiny files matching the
layout exactly, with a planted signal whose centre of mass is known.
"""

from bexa.testing.synthetic import (
    SyntheticEsrfScan,
    SyntheticLclsCube,
    SyntheticPalRun,
    SyntheticSpecScan,
    make_esrf_energy_series,
    make_esrf_scan,
    make_lcls_cube,
    make_legacy_cube,
    make_pal_run,
    make_pal_spec_scan,
)

__all__ = [
    "SyntheticEsrfScan",
    "SyntheticLclsCube",
    "SyntheticPalRun",
    "SyntheticSpecScan",
    "make_esrf_energy_series",
    "make_esrf_scan",
    "make_lcls_cube",
    "make_legacy_cube",
    "make_pal_run",
    "make_pal_spec_scan",
]
