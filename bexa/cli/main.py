"""The ``bexa`` command-line application.

Every command is a thin wrapper over a public Python function, so anything the
terminal can do a notebook can do too. Sub-commands for scans, cubes,
crystallography and optics live in sibling modules and register here.
"""

from __future__ import annotations

import cProfile
import importlib
import os
import pstats
from dataclasses import dataclass
from pathlib import Path

import typer

from bexa import __version__
from bexa._log import configure, get_logger

log = get_logger(__name__)

app = typer.Typer(
    help="bexa: Beamline EXperiment Analysis.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode=None,
)
profile_app = typer.Typer(help="Beamtime profiles and format specs.", no_args_is_help=True)
cache_app = typer.Typer(help="Inspect or clear the result cache.", no_args_is_help=True)
app.add_typer(profile_app, name="profile")
app.add_typer(cache_app, name="cache")


@dataclass
class CliState:
    """Options shared by all commands (set by the top-level callback)."""

    device: str = "auto"
    memory_fraction: float | None = None
    headless: bool = False


@app.callback()
def main(
    ctx: typer.Context,
    device: str = typer.Option("auto", "--device", help="cpu, cuda or auto."),
    memory_fraction: float | None = typer.Option(
        None,
        "--memory-fraction",
        help="Fraction of free memory one operation may use (default 0.5).",
    ),
    headless: bool = typer.Option(
        False, "--headless", help="Never open plot windows (Agg backend)."
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="DEBUG, INFO, WARNING or ERROR."),
    profile: bool = typer.Option(
        False, "--profile", help="Write a cProfile report of the command."
    ),
) -> None:
    """Global options; they also apply to the Python API through environment variables."""
    configure(log_level)
    os.environ["BEXA_DEVICE"] = device
    if memory_fraction is not None:
        os.environ["BEXA_MEMORY_FRACTION"] = str(memory_fraction)
    if headless:
        import matplotlib

        matplotlib.use("Agg")
    ctx.obj = CliState(device=device, memory_fraction=memory_fraction, headless=headless)
    if profile:
        profiler = cProfile.Profile()
        profiler.enable()

        def dump() -> None:
            profiler.disable()
            out = Path.cwd() / f"bexa-{ctx.invoked_subcommand or 'run'}.prof"
            profiler.dump_stats(str(out))
            summary = out.with_suffix(".txt")
            with summary.open("w", encoding="utf-8") as fh:
                pstats.Stats(profiler, stream=fh).sort_stats("cumulative").print_stats(30)
            typer.echo(f"profile written to {out} (summary in {summary})")

        ctx.call_on_close(dump)


@app.command()
def version() -> None:
    """Print the bexa version and the machine's compute resources."""
    from bexa.core.backend import device_info

    typer.echo(f"bexa {__version__}")
    for key, value in device_info().items():
        typer.echo(f"  {key}: {value}")


@app.command()
def info(
    path: Path = typer.Argument(..., exists=True, help="A scan folder, dataset folder or file."),
    depth: int = typer.Option(3, help="Maximum HDF5 tree depth to print."),
    attrs: bool = typer.Option(False, help="Also print HDF5 attributes."),
    scan: int | None = typer.Option(None, help="Scan number to open when the path is a dataset."),
    detector: str | None = typer.Option(None, help="Detector name to open."),
) -> None:
    """Match the format spec, print the file structure, motors, energy and sizes."""
    from bexa.io.inspect import describe

    typer.echo(describe(path, max_depth=depth))
    try:
        from bexa.core.scan import open as open_scan

        kwargs = {k: v for k, v in {"scan": scan, "detector": detector}.items() if v is not None}
        with open_scan(path, **kwargs) as scan_obj:
            typer.echo("")
            typer.echo(scan_obj.info())
    except ImportError:
        pass
    except Exception as exc:  # the path is inspectable but not openable as a scan
        log.debug("not opened as a scan: %s", exc)


# ------------------------------------------------------------------ profiles
@profile_app.command("list")
def profile_list() -> None:
    """List beamtime profiles and format specs found on this machine."""
    from bexa.config import list_profiles
    from bexa.io.formats import list_specs

    typer.echo("beamtime profiles:")
    for name, path in list_profiles().items():
        typer.echo(f"  {name:32s} {path}")
    typer.echo("format specs:")
    for name, path in list_specs().items():
        typer.echo(f"  {name:32s} {path}")


@profile_app.command("new")
def profile_new(
    name: str = typer.Argument(..., help="Profile name, e.g. hc6293_id03_2026-01."),
    format_name: str = typer.Option(..., "--format", help="Format spec the beamtime uses."),
    root: Path | None = typer.Option(None, help="Data root at the beamline."),
    out: Path | None = typer.Option(
        None, help="Output file (default configs/beamtimes/<name>.yaml)."
    ),
    force: bool = typer.Option(False, help="Overwrite an existing file."),
) -> None:
    """Write a beamtime profile skeleton to fill in."""
    from bexa.config import write_profile_skeleton
    from bexa.config.paths import package_configs

    target = out or package_configs() / "beamtimes" / f"{name}.yaml"
    written = write_profile_skeleton(target, name, format_name, root, force=force)
    typer.echo(f"profile skeleton written to {written}")


@profile_app.command("sniff")
def profile_sniff(
    path: Path = typer.Argument(..., exists=True, help="A file or folder of an unknown layout."),
    out: Path = typer.Option(..., "-o", "--out", help="Where to write the draft format spec."),
    name: str | None = typer.Option(None, help="Spec name (default derived from the path)."),
) -> None:
    """Draft a format spec from the files found at PATH (fill in the TODO entries)."""
    from bexa.io.formats import write_draft

    written = write_draft(path, out, name=name)
    typer.echo(f"draft spec written to {written}")


# --------------------------------------------------------------------- cache
@cache_app.command("path")
def cache_path() -> None:
    """Print the cache folder in use."""
    from bexa.core.cache import default_cache

    typer.echo(str(default_cache().root))


@cache_app.command("ls")
def cache_ls() -> None:
    """List cached results with their size."""
    from bexa.core.cache import default_cache

    entries = default_cache().entries()
    if not entries:
        typer.echo("cache is empty")
    for entry in entries:
        typer.echo(f"{entry.key}  {entry.size_bytes / 1e6:8.1f} MB  {entry.path}")


@cache_app.command("clear")
def cache_clear() -> None:
    """Delete every cached result."""
    from bexa.core.cache import default_cache

    removed = default_cache().clear()
    typer.echo(f"removed {removed} cached files")


# ------------------------------------------------------------- sub-commands
def _register_subapps() -> None:
    """Attach the scan, cube, auto, crystal and optics groups when their modules exist."""
    for module_name, attr, name in (
        ("bexa.cli.scan_cmd", "scan_app", "scan"),
        ("bexa.cli.cube_cmd", "cube_app", "cube"),
        ("bexa.cli.auto_cmd", "register", None),
        ("bexa.cli.crystal_cmd", "crystal_app", "crystal"),
        ("bexa.cli.optics_cmd", "optics_app", "optics"),
        ("bexa.cli.bench_cmd", "register", None),
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        target = getattr(module, attr)
        if name is None:
            target(app)
        else:
            app.add_typer(target, name=name)


_register_subapps()
