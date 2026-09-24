"""``bexa cube ...``: laser on/off cubes of XFEL runs (build, combine, plot, animate)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from bexa._log import get_logger
from bexa.cli.scan_cmd import parse_roi
from bexa.core.roi import ROI

log = get_logger(__name__)

cube_app = typer.Typer(
    help="XFEL laser on/off cubes: build, combine, plot, animate.", no_args_is_help=True
)

ROI_HELP = "y=400:550,x=200:375 or the legacy r0:r1,c0:c1 (rows then columns)."


def parse_cube_roi(text: str | None) -> ROI | None:
    """Accept ``y=..,x=..`` or the legacy ``r0:r1,c0:c1`` form."""
    if not text:
        return None
    if "=" in text:
        return parse_roi(text)
    rows, cols = text.split(",")
    r0, r1 = (int(v) for v in rows.split(":"))
    c0, c1 = (int(v) for v in cols.split(":"))
    return ROI(y=(r0, r1), x=(c0, c1))


def parse_runs(text: str) -> list[int]:
    """``"79-88"`` or ``"79,80,85"`` -> run numbers."""
    if "-" in text and "," not in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(v) for v in text.split(",") if v.strip()]


def _load_cube(path: Path) -> Any:
    from bexa.io.cube import load

    return load(path, squeeze_single=False)


def _profile(name: str | None) -> Any:
    from bexa.config import load_profile

    return load_profile(name)


@cube_app.command("build")
def cube_build(
    path: Path | None = typer.Argument(None, help="scan=NNN folder (or use --profile with --run)."),
    profile: str | None = typer.Option(None, "-p", "--profile"),
    run: int | None = typer.Option(None, help="Run number (with --profile)."),
    scan: int = typer.Option(1),
    roi: str | None = typer.Option(None, help=ROI_HELP),
    workers: int | None = typer.Option(None, help="Processes; default: all cores but one."),
    out: Path | None = typer.Option(
        None, "-o", "--out", help="Cube file; default from the profile."
    ),
    legacy: bool = typer.Option(False, help="Write the legacy runN.h5 layout."),
) -> None:
    """Reduce every motor point of a run to laser on/off mean frames."""
    from bexa.io.cube import save
    from bexa.pipelines import xfel_cube

    roi_obj = parse_cube_roi(roi)
    if profile:
        if run is None:
            raise typer.BadParameter("--run is needed with --profile")
        ds = xfel_cube.reduce_run(
            profile,
            run,
            scan=scan,
            roi=roi_obj,
            workers=workers,
            out=out,
            legacy=legacy,
            show_progress=True,
        )
        target = out or xfel_cube.cube_path(_profile(profile), run)
    else:
        if path is None or out is None:
            raise typer.BadParameter("give a PATH and -o OUT, or --profile with --run")
        from bexa.core.scan import open as open_scan

        with open_scan(path, scan=scan) as s:
            ds = xfel_cube.build_cube(s.source, roi=roi_obj, workers=workers, show_progress=True)
        target = out
        if legacy:
            xfel_cube.save_legacy(ds, target)
        else:
            save(ds, target)
    typer.echo(f"cube {dict(ds.sizes)} axis={ds.attrs.get('scan_axis')} saved to {target}")


@cube_app.command("combine")
def cube_combine(
    files: list[Path] = typer.Argument(None, help="Cube files to combine."),
    profile: str | None = typer.Option(None, "-p", "--profile"),
    runs: str | None = typer.Option(None, help="Run numbers with --profile, e.g. 79-88."),
    axis: str | None = typer.Option(None, help="Scan axis to bin on (default: the cube's)."),
    decimals: int = typer.Option(2, help="Axis values are rounded to this many decimals."),
    out: Path | None = typer.Option(None, "-o", "--out"),
) -> None:
    """Average the cubes of several runs onto one axis (replaces Scan_Combiner.py)."""
    from bexa.analysis.pump_probe import combine_runs
    from bexa.io.cube import save
    from bexa.pipelines.xfel_cube import cube_path

    paths = list(files or [])
    if profile and runs:
        prof = _profile(profile)
        numbers = parse_runs(runs)
        paths = [cube_path(prof, n) for n in numbers]
        if out is None:
            out = cube_path(prof, f"run{numbers[0]}-{numbers[-1]}Combined")
    if not paths:
        raise typer.BadParameter("give cube files or --profile with --runs")
    if out is None:
        raise typer.BadParameter("give -o OUT")
    combined = combine_runs([_load_cube(p) for p in paths], axis=axis, decimals=decimals)
    typer.echo(
        f"combined {len(paths)} cubes onto {combined.attrs['scan_axis']} -> {save(combined, out)}"
    )


@cube_app.command("plot")
def cube_plot(
    cube: Path | None = typer.Argument(None, help="Cube file (or --profile with --run)."),
    profile: str | None = typer.Option(None, "-p", "--profile"),
    run: str | None = typer.Option(None, help="Run number or name with --profile."),
    axis: str | None = typer.Option(None, help="Scan axis (default: the cube's)."),
    roi: str | None = typer.Option(None, help=ROI_HELP),
    out: Path | None = typer.Option(None, "-o", "--out", help="Output folder."),
    label: str | None = typer.Option(None, help="Prefix of the file names (default runN)."),
    show: bool = typer.Option(False, help="Open the figures instead of only saving them."),
) -> None:
    """Write the standard figure set: ROI image, on/off curves, COM, variance and skew."""
    from bexa.pipelines import cube_report

    cube_file, out_dir, name = _cube_target(cube, profile, run, out)
    ds = _load_cube(cube_file)
    written = cube_report.figure_set(
        ds, out_dir, label=label or name, roi=parse_cube_roi(roi), axis=axis
    )
    for kind, path in written.items():
        typer.echo(f"{kind:10s} {path}")
    if show:
        import matplotlib.pyplot as plt

        plt.show()


@cube_app.command("animate")
def cube_animate(
    cube: Path | None = typer.Argument(None, help="Cube file (or --profile with --run)."),
    profile: str | None = typer.Option(None, "-p", "--profile"),
    run: str | None = typer.Option(None),
    axis: str | None = typer.Option(None),
    roi: str | None = typer.Option(None, help=ROI_HELP),
    out: Path | None = typer.Option(None, "-o", "--out", help="Output folder."),
    label: str | None = typer.Option(None),
    fps: int = typer.Option(5),
) -> None:
    """Write laser on and off GIFs of the ROI, normalised by the background box."""
    from bexa.pipelines import cube_report

    cube_file, out_dir, name = _cube_target(cube, profile, run, out)
    ds = _load_cube(cube_file)
    written = cube_report.animations(
        ds, out_dir, label=label or name, roi=parse_cube_roi(roi), axis=axis, fps=fps
    )
    for kind, path in written.items():
        typer.echo(f"{kind:14s} {path}")


@cube_app.command("info")
def cube_info(cube: Path = typer.Argument(..., help="Cube file or LCLS folder.")) -> None:
    """Print the dims, coordinates and provenance of a cube."""
    ds = _load_cube(cube)
    typer.echo(f"{cube}: axis {ds.attrs.get('scan_axis')}")
    typer.echo(f"  dims {dict(ds.sizes)}")
    typer.echo(f"  variables {list(ds.data_vars)}")
    typer.echo(f"  coords {list(ds.coords)}")
    for key in ("run", "format", "bexa_version", "created"):
        if key in ds.attrs:
            typer.echo(f"  {key}: {ds.attrs[key]}")


def _cube_target(
    cube: Path | None, profile: str | None, run: str | None, out: Path | None
) -> tuple[Path, Path, str]:
    from bexa.pipelines.xfel_cube import cube_path

    if profile and run:
        prof = _profile(profile)
        name = run if run.startswith("run") else f"run{int(run)}"
        cube_file = cube_path(prof, name)
        out_dir = out or Path(prof.processed_root or cube_file.parent.parent) / "outputs"
        return cube_file, out_dir, name
    if cube is None:
        raise typer.BadParameter("give a cube file or --profile with --run")
    return cube, out or cube.parent / "outputs", cube.stem
