"""``bexa bench``: time readers, reductions and plots, and compare with a baseline."""

from __future__ import annotations

from pathlib import Path

import typer


def register(app: typer.Typer) -> None:
    @app.command("bench")
    def bench(
        full: bool = typer.Option(False, help="Realistic frame size instead of the quick set."),
        device: str = typer.Option("auto"),
        real: Path | None = typer.Option(
            None, help="A real frame stack (.npy, tiff, file.h5::/path)."
        ),
        baselines: Path = typer.Option(
            Path("benchmarks/baselines"), help="Folder of baseline JSON files."
        ),
        save: bool = typer.Option(False, help="Store this run as the baseline for this machine."),
        tolerance: float = typer.Option(
            0.2, help="Allowed slowdown before a regression is reported."
        ),
        no_plots: bool = typer.Option(False, help="Skip the figure-set timing."),
    ) -> None:
        """Benchmark this machine and report regressions against the stored baseline."""
        from bexa.pipelines import benchmark

        results = benchmark.run_suite(
            quick=not full, device=device, real_path=real, plots=not no_plots
        )
        typer.echo(benchmark.format_results(results))
        baseline = benchmark.load_baseline(baselines)
        if baseline is not None:
            regressions = benchmark.compare(results, baseline, tolerance)
            if regressions:
                typer.echo("regressions:")
                for line in regressions:
                    typer.echo(f"  {line}")
            else:
                typer.echo("no regression against the baseline")
        if save:
            typer.echo(f"baseline written to {benchmark.save_baseline(results, baselines)}")
        elif baseline is None:
            typer.echo("no baseline for this machine yet; rerun with --save to store one")
        if baseline is not None and benchmark.compare(results, baseline, tolerance) and not save:
            raise typer.Exit(code=1)
