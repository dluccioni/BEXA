"""Core data model and execution machinery.

- :mod:`bexa.core.scan`: the lazy ``Scan`` handle, ``open`` and ``list_scans``.
- :mod:`bexa.core.dataset`: a dataset folder as a table of scans, selection and stacking.
- :mod:`bexa.core.structure`: dims, coordinates and frame-to-grid mapping of a scan.
- :mod:`bexa.core.roi`: regions of interest over pixels, motors and energy.
- :mod:`bexa.core.reductions`: accumulators and the single-pass streaming engine.
- :mod:`bexa.core.backend`: numpy/cupy selection, memory budget, batch sizing.
- :mod:`bexa.core.resources`: the memory and CPUs this process may use (SLURM-aware).
- :mod:`bexa.core.cache`: on-disk and in-memory caching of reduction results.
- :mod:`bexa.core.parallel`: thread prefetching and process pools.
- :mod:`bexa.core.registry`: plugin registries for engines, accumulators and plots.
- :mod:`bexa.core.provenance`: attrs that record where a result came from.
- :mod:`bexa.core.units`: physical constants and unit conversions.
- :mod:`bexa.core.settings`: the resolved device, memory, cache and profile settings.
- :mod:`bexa.core.accessor`: the ``.bexa`` accessor on xarray objects.
"""
