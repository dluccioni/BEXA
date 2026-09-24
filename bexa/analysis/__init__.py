"""Pure analysis functions: numpy or cupy arrays in, the same kind out.

- :mod:`bexa.analysis.preprocess`: darks, hot pixels, thresholds, normalisation, binning.
- :mod:`bexa.analysis.rocking`: per-pixel centre of mass, moments, argmax, weighted median.
- :mod:`bexa.analysis.peaks`: per-frame detector COM and moments, 2-D Gaussian fits.
- :mod:`bexa.analysis.pump_probe`: on/off splitting, differential signals, binning runs.
- :mod:`bexa.analysis.registration`: shifts, overlaps, stitching and mosaics.
- :mod:`bexa.analysis.fitting`: lmfit models for rocking curves.
- :mod:`bexa.analysis.segmentation`: clustering and smoothness masks.
- :mod:`bexa.analysis.fields`: vector fields, gradients and strain integration.
- :mod:`bexa.analysis.projections`: projections, integrated maps, RLP point clouds.
"""
