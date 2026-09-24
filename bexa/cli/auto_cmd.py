"""``bexa auto``: the beamtime watcher (cubes, figures and animations for new runs)."""

from __future__ import annotations

from pathlib import Path

import typer

from bexa._log import get_logger

log = get_logger(__name__)


def register(app: typer.Typer) -> None:
    """Attach the ``auto`` command to the main app."""

    @app.command("auto")
    def auto(
        profile: str = typer.Option(..., "-p", "--profile", help="Beamtime profile."),
        raw_dir: Path | None = typer.Option(
            None, help="type=raw folder (default from the profile)."
        ),
        cube_dir: Path | None = typer.Option(None, help="Folder of the runN.h5 cubes."),
        out_dir: Path | None = typer.Option(None, help="Folder of the figures and GIFs."),
        roi_table: Path | None = typer.Option(None, help="ROI_dict.csv or a YAML table."),
        skip_list: Path | None = typer.Option(None, help="Skip_dict.csv."),
        runs: str | None = typer.Option(None, help="Only these runs, e.g. 10-20 or 12,15."),
        scan: int = typer.Option(1),
        workers: int | None = typer.Option(None),
        watch: bool = typer.Option(False, help="Keep polling for new runs."),
        interval: float = typer.Option(60.0, help="Seconds between polls with --watch."),
        dry_run: bool = typer.Option(False, help="List the runs that would be processed."),
        legacy: bool = typer.Option(False, help="Write cubes in the legacy runN.h5 layout."),
        no_animations: bool = typer.Option(False, help="Skip the GIFs."),
    ) -> None:
        """Process every raw run without outputs: cube, figure set, animations."""
        import matplotlib

        matplotlib.use("Agg")
        from bexa.cli.cube_cmd import parse_runs
        from bexa.pipelines import auto_process

        cfg = auto_process.config_from_profile(
            profile,
            raw_dir=raw_dir,
            cube_dir=cube_dir,
            outputs_dir=out_dir,
            roi_table=roi_table,
            skip_list=skip_list,
            scan=scan,
            workers=workers,
            legacy_cubes=legacy,
            animations=not no_animations,
        )
        only = [f"run{n}" for n in parse_runs(runs)] if runs else None
        if watch:
            records = auto_process.watch(cfg, interval_s=interval, only=only, dry_run=dry_run)
        else:
            records = auto_process.run_once(cfg, only=only, dry_run=dry_run)
        if not records:
            typer.echo("nothing to process")
        for record in records:
            typer.echo(
                f"{record.run:24s} {record.status:8s} {record.seconds:7.1f} s {record.error}"
            )
