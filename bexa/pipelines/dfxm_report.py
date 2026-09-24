"""The standard figure set for one dark-field microscopy scan.

One streaming pass computes everything (sum, max, projections, motor centre of
mass and a preview volume); the figures are drawn from those results and the
arrays are saved next to them with full provenance.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import xarray as xr

from bexa._log import get_logger
from bexa.core.reductions import Max, MotorCOM, Preview, Projections, Sum
from bexa.core.roi import ROI
from bexa.core.scan import Scan
from bexa.io.cube import save
from bexa.viz import images, maps, volume

log = get_logger(__name__)

__all__ = ["report"]


def report(
    scan: Scan,
    out_dir: str | Path,
    roi: ROI | None = None,
    downsample: int | Sequence[int] | None = 4,
    sigma: float = 3.0,
    device: str | None = "auto",
    apply_log: bool = False,
    dpi: int = 150,
    prefix: str | None = None,
    show_progress: bool = False,
) -> dict[str, Path]:
    """Compute and save the standard DFXM figures and arrays for ``scan``.

    Files written to ``out_dir`` (``<prefix>_...``):
    ``sum.png``, ``max.png``, ``projections.png``, ``tiles.png``, ``com_<motor>.png``,
    ``width_<motor>.png``, ``bivariate.png`` (two motors), and ``results.h5`` with every array.

    Returns
    -------
    dict
        Name -> path of each file written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix or scan.name.replace("/", "_")
    interactive_backend = matplotlib.get_backend()
    written: dict[str, Path] = {}

    motors = [d for d in scan.structure.motor_dims if d != "frame"]
    accumulators: list[Any] = [Sum(), Max(), Projections(), Preview(apply_log=apply_log)]
    if motors:
        accumulators.append(MotorCOM(axes=motors, sigma=sigma, moments=2))
    results = scan.reduce(
        accumulators, roi=roi, downsample=downsample, device=device, show_progress=show_progress
    )

    def savefig(fig: Any, name: str) -> None:
        path = out_dir / f"{prefix}_{name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written[name] = path

    nm = scan.geometry.effective_pixel_nm if scan.geometry is not None else None
    for key in ("sum", "max"):
        fig, _, _ = images.show(results[key], scalebar=nm or False, title=f"{scan.name} {key}")
        savefig(fig, key)

    projections = {
        k: v
        for k, v in results.items()
        if v.ndim == 2 and (k == "xy" or ("_" in k and k.split("_")[0] in motors))
    }
    if projections:
        fig, _ = volume.projections_panel(projections, log=True)
        savefig(fig, "projections")

    preview = results["preview"]
    if motors:
        fig, _ = images.tiles(preview, axis=motors[0], n=9, per_row=3)
        savefig(fig, "tiles")
        for m in motors:
            fig, _, _ = maps.com_map(results[f"com_{m}"], scalebar=nm or False, title=f"COM {m}")
            savefig(fig, f"com_{m}")
            fig, _, _ = maps.com_map(
                results[f"width_{m}"],
                center="median",
                cmap="viridis",
                scalebar=nm or False,
                title=f"width {m}",
            )
            savefig(fig, f"width_{m}")
        if len(motors) >= 2:
            fig, _ = maps.bivariate(
                results[f"com_{motors[0]}"],
                results[f"com_{motors[1]}"],
                labels=(motors[0], motors[1]),
                scalebar=nm or False,
                title=f"{scan.name} centre of mass",
            )
            savefig(fig, "bivariate")

    dataset = xr.Dataset({k: v for k, v in results.items() if isinstance(v, xr.DataArray)})
    dataset.attrs.update(results["sum"].attrs)
    path = save(dataset, out_dir / f"{prefix}_results.h5")
    written["results"] = path
    matplotlib.use(interactive_backend, force=False)
    log.info("report for %s written to %s (%d files)", scan.name, out_dir, len(written))
    return written
