"""Synthetic data for tests, examples and benchmarks.

Every format spec has a generator here that writes tiny files matching the
layout exactly, with a planted signal whose centre of mass is known.
"""

from bexa.testing.synthetic import (
    SAMPLE_REGIONS,
    Region,
    SampleModel,
    SyntheticEsrfScan,
    SyntheticLclsCube,
    SyntheticPalRun,
    SyntheticSpecScan,
    SyntheticZStack,
    dfxm_sample,
    make_esrf_energy_series,
    make_esrf_scan,
    make_esrf_zstack,
    make_lcls_cube,
    make_legacy_cube,
    make_pal_run,
    make_pal_spec_scan,
)

__all__ = [
    "SAMPLE_REGIONS",
    "Region",
    "SampleModel",
    "SyntheticEsrfScan",
    "SyntheticLclsCube",
    "SyntheticPalRun",
    "SyntheticSpecScan",
    "SyntheticZStack",
    "dfxm_sample",
    "make_esrf_energy_series",
    "make_esrf_scan",
    "make_esrf_zstack",
    "make_lcls_cube",
    "make_legacy_cube",
    "make_pal_run",
    "make_pal_spec_scan",
]
