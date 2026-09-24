"""Plotting.

Every function takes ``ax=None``, returns the figure and artists it made, and
never calls ``plt.show()``. Colour limits default to the 1st and 99th
percentiles, axes are labelled from the DataArray coordinates, and a scalebar
is drawn when the effective pixel size is known.

- :mod:`bexa.viz.images`: single images, tile grids, ROI overlays, scalebars.
- :mod:`bexa.viz.curves`: on/off/difference panels, moments vs axis, rocking curves.
- :mod:`bexa.viz.maps`: centre-of-mass maps, bivariate and HSV blends, vector fields.
- :mod:`bexa.viz.volume`: projections panel, slice grids, isosurfaces, 3-D RLP scatter.
- :mod:`bexa.viz.interactive`: sliders and ROI pickers for notebooks and scripts.
- :mod:`bexa.viz.animation`: GIF and MP4 from any stack.
- :mod:`bexa.viz.style`: screen and paper themes.
- :mod:`bexa.viz.napari_app`: optional napari launchers (imported on demand).
- :mod:`bexa.viz.auto`: ``plot(obj)`` picks the figure from the object's dims and attrs.

``projections_panel``, ``isosurface`` and ``rlp_scatter`` accept ``backend="plotly"``
for interactive figures.
"""

from bexa.viz import animation, auto, curves, images, interactive, maps, style, volume

__all__ = ["animation", "auto", "curves", "images", "interactive", "maps", "style", "volume"]
