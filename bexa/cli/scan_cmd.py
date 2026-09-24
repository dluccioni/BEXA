"""``bexa scan ...``: frame-stack scans (ESRF and any other hdf5_stack layout)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from bexa._log import get_logger
from bexa.core.roi import ROI

log = get_logger(__name__)

scan_app = typer.Typer(help="Frame-stack scans: preview, reduce, report.", no_args_is_help=True)

PATH_HELP = "Dataset folder, master file or scanNNNN folder (or use --profile/--sample)."


def parse_roi(text: str | None) -> ROI | None:
    """``"x=800:1200,y=700:1100,chi=0:8"`` -> ROI (see :meth:`bexa.core.roi.ROI.parse`)."""
    return ROI.parse(text) if text else None


def parse_downsample(text: str | None) -> Any:
    """``"4"`` -> 4, ``"1,1,4,4"`` -> (1, 1, 4, 4)."""
    if not text:
        return None
    parts = [int(p) for p in text.split(",") if p.strip()]
    return parts[0] if len(parts) == 1 else tuple(parts)


def parse_scan(text: str | None) -> Any:
    """``"7"`` -> 7, ``"2-10"`` -> (2, 10), ``"2,3,5"`` -> [2, 3, 5]."""
    if not text:
        return None
    if "-" in text and "," not in text:
        a, b = text.split("-")
        return int(a), int(b)
    if "," in text:
        return [int(p) for p in text.split(",")]
    return int(text)


def open_from_options(
    path: Path | None,
    profile: str | None,
    sample: str | None,
    scan: str | None,
    detector: str | None,
) -> Any:
    from bexa.core.scan import open as open_scan
    from bexa.core.scan import open_profile

    scan_sel = parse_scan(scan)
    if profile:
        return open_profile(profile, sample=sample, scan=scan_sel, detector=detector)
    if path is None:
        raise typer.BadParameter("give a PATH or --profile with --sample")
    return open_scan(path, scan=scan_sel, detector=detector)


@scan_app.command("info")
def scan_info(
    path: Path | None = typer.Argument(None, help=PATH_HELP),
    profile: str | None = typer.Option(None, "-p", "--profile"),
    sample: str | None = typer.Option(None, "-s", "--sample"),
    scan: str | None = typer.Option(None, help="Scan number, range 2-10 or list 2,3,5."),
    detector: str | None = typer.Option(None),
) -> None:
    """Print dims, motors, energy and file sizes of a scan."""
    with open_from_options(path, profile, sample, scan, detector) as s:
        typer.echo(s.info())


@scan_app.command("preview")
def scan_preview(
    path: Path | None = typer.Argument(None, help=PATH_HELP),
    profile: str | None = typer.Option(None, "-p", "--profile"),
    sample: str | None = typer.Option(None, "-s", "--sample"),
    scan: str | None = typer.Option(None),
    detector: str | None = typer.Option(None),
    downsample: str = typer.Option("8", help="int or one factor per dim, e.g. 1,1,4,4"),
    roi: str | None = typer.Option(None, help="x=800:1200,y=700:1100,chi=0:8"),
    apply_log: bool = typer.Option(False, help="Store log10(1 + I)."),
    out: Path | None = typer.Option(None, "-o", "--out", help="Save the volume (.h5 or .zarr)."),
    plot: bool = typer.Option(False, help="Show the projections panel."),
) -> None:
    """Build (and cache) the downsampled preview volume."""
    from bexa.io.cube import save
    from bexa.viz.volume import projections_panel

    with open_from_options(path, profile, sample, scan, detector) as s:
        prev = s.preview(
            downsample=parse_downsample(downsample), roi=parse_roi(roi), apply_log=apply_log
        )
        typer.echo(f"preview {prev.dims} {prev.shape}")
        if out:
            typer.echo(f"saved to {save(prev, out)}")
        if plot:
            import matplotlib.pyplot as plt

            projections_panel(prev)
            plt.show()


@scan_app.command("reduce")
def scan_reduce(
    path: Path | None = typer.Argument(None, help=PATH_HELP),
    profile: str | None = typer.Option(None, "-p", "--profile"),
    sample: str | None = typer.Option(None, "-s", "--sample"),
    scan: str | None = typer.Option(None),
    detector: str | None = typer.Option(None),
    acc: str = typer.Option(
        "sum,com", help="Comma list of sum,mean,max,min,com,projections,preview,stats,argmax."
    ),
    roi: str | None = typer.Option(None, help="x=800:1200,y=700:1100,chi=0:8"),
    downsample: str | None = typer.Option(None, help="int or one factor per dim"),
    sigma: float = typer.Option(3.0, help="Gaussian smoothing for the COM weights."),
    out: Path = typer.Option(..., "-o", "--out", help="Output file (.h5 or .zarr)."),
) -> None:
    """Run accumulators in one pass and save every result."""
    import xarray as xr

    from bexa.core import reductions
    from bexa.io.cube import save

    table = {
        "sum": reductions.Sum,
        "mean": reductions.Mean,
        "max": reductions.Max,
        "min": reductions.Min,
        "projections": reductions.Projections,
        "preview": reductions.Preview,
        "stats": reductions.FrameStats,
        "argmax": reductions.ArgmaxMotor,
        "com": lambda: reductions.MotorCOM(sigma=sigma),
    }
    accs = []
    for name in acc.split(","):
        name = name.strip()
        if name not in table:
            raise typer.BadParameter(f"unknown accumulator {name!r}; choose from {sorted(table)}")
        accs.append(table[name]())
    with open_from_options(path, profile, sample, scan, detector) as s:
        results = s.reduce(
            accs, roi=parse_roi(roi), downsample=parse_downsample(downsample), show_progress=True
        )
        dataset = xr.Dataset(results)
        dataset.attrs.update(next(iter(results.values())).attrs)
        typer.echo(f"saved {sorted(results)} to {save(dataset, out)}")


@scan_app.command("report")
def scan_report(
    path: Path | None = typer.Argument(None, help=PATH_HELP),
    profile: str | None = typer.Option(None, "-p", "--profile"),
    sample: str | None = typer.Option(None, "-s", "--sample"),
    scan: str | None = typer.Option(None),
    detector: str | None = typer.Option(None),
    out: Path = typer.Option(..., "-o", "--out", help="Output folder."),
    roi: str | None = typer.Option(None),
    downsample: str = typer.Option("4"),
    sigma: float = typer.Option(3.0),
    apply_log: bool = typer.Option(False),
) -> None:
    """Write the standard DFXM figure set and the arrays behind it."""
    import matplotlib

    matplotlib.use("Agg")
    from bexa.pipelines.dfxm_report import report

    with open_from_options(path, profile, sample, scan, detector) as s:
        written = report(
            s,
            out,
            roi=parse_roi(roi),
            downsample=parse_downsample(downsample),
            sigma=sigma,
            apply_log=apply_log,
            show_progress=True,
        )
    for name, p in written.items():
        typer.echo(f"{name:14s} {p}")
