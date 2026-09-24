"""Optional napari launchers: a volume viewer, the ROI-to-3D-RLP app and Slices-to-Voxels.

The state of each app lives in a plain Python object (:class:`Roi3dRlpSession`,
:class:`SlicesToVoxels`) that works without napari, so the logic is testable
and usable from a notebook; the ``*_app`` functions wrap it in a napari
viewer with magicgui controls. napari and magicgui are imported only there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import xarray as xr

from bexa.analysis.rocking import motor_com
from bexa.core.roi import ROI
from bexa.viz.style import as_array, resolve_clim

__all__ = [
    "LayerSpec",
    "Roi3dRlpSession",
    "SlicesToVoxels",
    "projection_layers",
    "roi_3drlp_app",
    "roi_from_shapes",
    "slices_to_voxels_app",
    "view_volume",
    "volume_layer",
]


@dataclass
class LayerSpec:
    """Arguments of one ``viewer.add_image`` call."""

    data: Any
    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    def add_to(self, viewer: Any) -> Any:
        if self.name in [layer.name for layer in viewer.layers]:
            viewer.layers.remove(self.name)
        return viewer.add_image(self.data, name=self.name, **self.kwargs)


def volume_layer(
    volume: Any,
    name: str = "volume",
    log: bool = False,
    scale: tuple[float, ...] | None = None,
    rendering: str = "mip",
    colormap: str = "inferno",
) -> LayerSpec:
    """Layer spec of a volume with percentile contrast limits (``log`` shows ``log10(1 + I)``)."""
    data = np.asarray(as_array(volume), dtype=np.float32)
    if log:
        data = np.log10(1.0 + np.clip(data, 0, None))
    kwargs: dict[str, Any] = {
        "colormap": colormap,
        "contrast_limits": [float(v) for v in resolve_clim(data, ("p", 1, 99))],
        "rendering": rendering,
        "blending": "additive",
    }
    if scale is not None:
        kwargs["scale"] = tuple(scale)
    return LayerSpec(data, name, kwargs)


def projection_layers(volume: Any, log: bool = True) -> dict[str, LayerSpec]:
    """``xy``, ``xz`` and ``yz`` sum projections of a 3-D volume ``(z, y, x)`` as layers."""
    data = np.asarray(as_array(volume), dtype=np.float64)
    if data.ndim > 3:
        data = data.reshape(-1, *data.shape[-2:])
    projections = {"xy": data.sum(axis=0), "xz": data.sum(axis=1), "yz": data.sum(axis=2)}
    out = {}
    for key, image in projections.items():
        if log:
            image = np.log10(1.0 + np.clip(image, 0, None))
        out[key] = LayerSpec(
            image.astype(np.float32),
            f"proj_{key}",
            {
                "colormap": "magma",
                "contrast_limits": [float(v) for v in resolve_clim(image, ("p", 1, 99))],
            },
        )
    return out


def roi_from_shapes(shapes: dict[str, Any], volume_shape: tuple[int, int, int]) -> ROI:
    """The ROI drawn as rectangles on the projections (``xy`` gives y, x; ``xz``/``yz`` give z).

    ``shapes`` maps a projection name to the vertices of its first rectangle
    ``(n, 2)`` in ``(row, column)`` order, as napari shapes layers store them.
    """
    nz, ny, nx = volume_shape
    z0, z1, y0, y1, x0, x1 = 0, nz, 0, ny, 0, nx

    def bounds(vertices: Any) -> tuple[int, int, int, int]:
        v = np.asarray(vertices, dtype=float)
        return int(v[:, 0].min()), int(v[:, 0].max()), int(v[:, 1].min()), int(v[:, 1].max())

    if shapes.get("xy") is not None and len(shapes["xy"]):
        y0, y1, x0, x1 = bounds(shapes["xy"])
    if shapes.get("xz") is not None and len(shapes["xz"]):
        z0, z1, x0, x1 = bounds(shapes["xz"])
    if shapes.get("yz") is not None and len(shapes["yz"]):
        z0, z1, y0, y1 = bounds(shapes["yz"])
    return ROI(
        z=(max(z0, 0), min(z1, nz)),
        y=(max(y0, 0), min(y1, ny)),
        x=(max(x0, 0), min(x1, nx)),
        index_dims=["z"],
    )


class Roi3dRlpSession:
    """State of the ROI-to-3D-RLP app: a scan, its preview volume, an ROI and coordinates.

    Parameters
    ----------
    scan
        An open :class:`bexa.core.scan.Scan`.
    downsample
        Pixel downsampling of the preview (motor dims are kept).
    """

    def __init__(self, scan: Any, downsample: int = 4, log: bool = True) -> None:
        self.scan = scan
        self.log = log
        self.downsample = downsample
        self.volume: xr.DataArray | None = None
        self.roi = ROI()
        self.space = "angular"

    @property
    def motor_dims(self) -> tuple[str, ...]:
        return tuple(self.scan.structure.motor_dims)

    def load(self, downsample: int | None = None) -> xr.DataArray:
        """Build (and cache) the preview volume ``(motors..., y, x)``."""
        if downsample is not None:
            self.downsample = downsample
        factors = (1,) * len(self.motor_dims) + (self.downsample, self.downsample)
        self.volume = self.scan.preview(downsample=factors, apply_log=self.log)
        return self.volume

    def stack(self) -> np.ndarray:
        """The preview as a 3-D stack ``(frames, y, x)`` for the viewer."""
        if self.volume is None:
            self.load()
        assert self.volume is not None
        data = np.asarray(self.volume.values)
        return data.reshape(-1, *data.shape[-2:])

    def set_roi_from_shapes(self, shapes: dict[str, Any]) -> ROI:
        self.roi = roi_from_shapes(shapes, self.stack().shape)
        return self.roi

    def roi_volume(self) -> np.ndarray:
        """The stack cut to the ROI (z is the frame index of the preview)."""
        stack = self.stack()
        z = self.roi.get("z") or (0, stack.shape[0])
        ys, xs = (
            self.roi.pixel_slices(stack.shape[1:])
            if self.roi.get("y") or self.roi.get("x")
            else (slice(None), slice(None))
        )
        return stack[int(z[0]) : int(z[1]), ys, xs]

    def coordinates(
        self, space: str = "angular", crystal: Any = None, **fixed_angles_deg: float
    ) -> dict[str, np.ndarray]:
        """Motor-point coordinates of the preview in angular, Q or hkl space."""
        from bexa.geometry.transforms import build_coordinate_grids

        self.space = space
        return build_coordinate_grids(
            self.scan.structure,
            self.scan.geometry,
            crystal=crystal or self.scan.crystal,
            space=space,
            hkl_center=getattr(self.scan, "hkl_center", None),
            **fixed_angles_deg,
        )

    def com_maps(self, sigma: float = 3.0) -> dict[str, np.ndarray]:
        """Centre-of-mass map per motor from the preview (inside the pixel ROI)."""
        if self.volume is None:
            self.load()
        assert self.volume is not None
        volume = self.volume
        if self.roi.get("y") or self.roi.get("x"):
            ys, xs = self.roi.pixel_slices(volume.shape[-2:])
            volume = volume.isel(y=ys, x=xs)
        out = {}
        for i, dim in enumerate(self.motor_dims):
            others = [d for d in self.motor_dims if d != dim]
            collapsed = volume.sum(others) if others else volume
            out[dim] = motor_com(
                np.asarray(collapsed.values),
                np.asarray(volume.coords[dim].values),
                axis=0,
                sigma=sigma,
            )
            del i
        return out


class SlicesToVoxels:
    """State of the Slices-to-Voxels app: one frame per z layer picked by (phi, mu) indices.

    Parameters
    ----------
    layers
        One 4-D array ``(phi, mu, y, x)`` (or DataArray) per z layer, or a list
        of :class:`bexa.core.scan.Scan` objects whose preview is used.
    dims
        Names of the two leading dims, used to label the sliders; taken from
        the first layer's dims (a DataArray or a Scan) when not given.
    """

    def __init__(
        self, layers: list[Any], downsample: int = 4, dims: tuple[str, str] | None = None
    ) -> None:
        self.dims = tuple(dims) if dims else self._dims_of(layers[0])
        self.layers = [self._as_grid(layer, downsample) for layer in layers]
        shape = self.layers[0].shape
        self.phi_count, self.mu_count = int(shape[0]), int(shape[1])
        self.phi_idx = 0
        self.mu_idx = 0
        self.phi_rad = 0
        self.mu_rad = 0
        self.offsets = np.zeros((len(self.layers), 2), dtype=int)  # (dphi, dmu) per layer
        self.visible = np.ones(len(self.layers), dtype=bool)

    @staticmethod
    def _dims_of(layer: Any) -> tuple[str, str]:
        """Names of the two leading dims of a layer (``phi``/``mu`` for plain arrays)."""
        if hasattr(layer, "structure"):
            motors = tuple(str(d) for d in layer.structure.motor_dims)
        elif hasattr(layer, "dims"):
            motors = tuple(str(d) for d in layer.dims if d not in ("y", "x"))
        else:
            motors = ()
        if len(motors) >= 2:
            return motors[0], motors[1]
        if len(motors) == 1:
            return motors[0], "index"
        return "phi", "mu"

    @staticmethod
    def _as_grid(layer: Any, downsample: int) -> np.ndarray:
        if hasattr(layer, "preview"):
            layer = layer.preview(downsample=(1, 1, downsample, downsample))
        data = np.asarray(as_array(layer), dtype=np.float32)
        if data.ndim == 3:
            data = data[:, None]
        if data.ndim != 4:
            raise ValueError("each layer must be (phi, mu, y, x)")
        return data

    def mean_frame(self, layer: int) -> np.ndarray:
        """Mean of the frames around ``(phi_idx, mu_idx)`` within the radii, offset per layer."""
        data = self.layers[layer]
        p0 = int(np.clip(self.phi_idx + self.offsets[layer, 0], 0, self.phi_count - 1))
        m0 = int(np.clip(self.mu_idx + self.offsets[layer, 1], 0, self.mu_count - 1))
        ps = slice(max(p0 - self.phi_rad, 0), min(p0 + self.phi_rad + 1, self.phi_count))
        ms = slice(max(m0 - self.mu_rad, 0), min(m0 + self.mu_rad + 1, self.mu_count))
        return data[ps, ms].mean(axis=(0, 1))

    def build_volume(self, log: bool = True) -> np.ndarray:
        """``(z, y, x)`` volume of the visible layers' mean frames (log10 when ``log``)."""
        frames = []
        for i in range(len(self.layers)):
            if not self.visible[i]:
                frames.append(np.zeros(self.layers[i].shape[-2:], dtype=np.float32))
            else:
                frames.append(self.mean_frame(i))
        volume = np.stack(frames).astype(np.float32)
        return np.log10(np.clip(volume, 1e-6, None)) if log else volume


# ------------------------------------------------------------- napari wrappers
def _napari() -> Any:
    try:
        import napari
    except ImportError as exc:
        raise ImportError("napari is not installed; pip install 'napari[all]'") from exc
    return napari


def view_volume(volume: Any, log: bool = False, show: bool = True, **kwargs: Any) -> Any:
    """Open a napari viewer on a volume (``(z, y, x)`` or any stack of images)."""
    napari = _napari()
    viewer = napari.Viewer(show=show)
    volume_layer(volume, log=log, **kwargs).add_to(viewer)
    viewer.dims.ndisplay = 3
    return viewer


def roi_3drlp_app(scan: Any, downsample: int = 4, show: bool = True) -> Any:
    """The ROI-to-3D-RLP viewer: draw the ROI on projections, then view it with coordinates."""
    napari = _napari()
    from magicgui import magicgui

    session = Roi3dRlpSession(scan, downsample=downsample)
    viewer = napari.Viewer(title="bexa ROI to 3D RLP", show=show)
    shapes: dict[str, Any] = {}

    def show_projections(factor: int = downsample) -> None:
        session.load(factor)
        stack = session.stack()
        for key, spec in projection_layers(stack, log=False).items():
            spec.add_to(viewer)
            if f"roi_{key}" in [layer.name for layer in viewer.layers]:
                viewer.layers.remove(f"roi_{key}")
            shapes[key] = viewer.add_shapes(
                name=f"roi_{key}", edge_color="yellow", face_color="transparent", edge_width=2
            )

    @magicgui(call_button="Show projections", factor={"choices": [1, 2, 4, 8, 16]})
    def projections_widget(factor: int = downsample) -> None:
        show_projections(factor)

    @magicgui(call_button="Apply ROI and build volume", space={"choices": ["angular", "Q", "hkl"]})
    def roi_widget(space: str = "angular") -> str:
        drawn = {key: (layer.data[0] if len(layer.data) else None) for key, layer in shapes.items()}
        roi = session.set_roi_from_shapes(drawn)
        volume = session.roi_volume()
        layer = volume_layer(volume, name=f"volume_{space}", rendering="mip").add_to(viewer)
        try:
            layer.metadata["coordinates"] = session.coordinates(space)
        except Exception as exc:  # no crystal for hkl, no Bragg angle for Q
            layer.metadata["coordinates_error"] = str(exc)
        layer.metadata["roi"] = roi.to_dict()
        viewer.dims.ndisplay = 3
        return roi.describe()

    viewer.window.add_dock_widget(projections_widget, name="Projections", area="right")
    viewer.window.add_dock_widget(roi_widget, name="ROI and coordinates", area="right")
    show_projections(downsample)
    return viewer


def slices_to_voxels_app(
    layers: list[Any],
    downsample: int = 4,
    show: bool = True,
    scale: tuple[float, float, float] = (5.0, 0.15, 0.15),
    dims: tuple[str, str] | None = None,
) -> Any:
    """Slices-to-Voxels: one frame per z layer chosen with two sliders, stacked in 3-D.

    The sliders step the two leading dims of the layers (``phi`` and ``mu`` in
    the legacy app; ``chi`` and ``mu`` for a mosaicity scan per layer,
    ``energy`` and ``mu`` for an energy series per layer). ``scale`` is the
    voxel size ``(z, y, x)`` napari uses to draw the stack.
    """
    napari = _napari()
    from magicgui import magicgui

    model = SlicesToVoxels(layers, downsample=downsample, dims=dims)
    first, second = model.dims
    viewer = napari.Viewer(title="bexa Slices to Voxels", show=show)
    layer = volume_layer(
        model.build_volume(), name="voxels", rendering="iso", colormap="viridis", scale=scale
    ).add_to(viewer)

    def rebuild() -> None:
        layer.data = model.build_volume()

    @magicgui(
        auto_call=True,
        phi_idx={"widget_type": "Slider", "max": model.phi_count - 1, "label": first},
        mu_idx={"widget_type": "Slider", "max": model.mu_count - 1, "label": second},
        phi_rad={"label": f"{first} radius"},
        mu_rad={"label": f"{second} radius"},
    )
    def global_controls(
        phi_idx: int = 0, mu_idx: int = 0, phi_rad: int = 0, mu_rad: int = 0
    ) -> None:
        model.phi_idx, model.mu_idx = phi_idx, mu_idx
        model.phi_rad, model.mu_rad = phi_rad, mu_rad
        rebuild()

    @magicgui(
        call_button="Apply to layer",
        z={"max": len(model.layers) - 1},
        dphi={"label": f"d{first}"},
        dmu={"label": f"d{second}"},
    )
    def local_controls(z: int = 0, dphi: int = 0, dmu: int = 0, visible: bool = True) -> None:
        model.offsets[z] = (dphi, dmu)
        model.visible[z] = visible
        rebuild()

    viewer.window.add_dock_widget(global_controls, name=f"GLOBAL {first}/{second}", area="right")
    viewer.window.add_dock_widget(local_controls, name="LOCAL slice offsets", area="right")
    viewer.dims.ndisplay = 3
    return viewer
