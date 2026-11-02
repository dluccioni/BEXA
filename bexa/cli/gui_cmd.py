"""``bexa gui``: the GUI in a window of its own (:mod:`bexa.gui.window`)."""

from __future__ import annotations

from pathlib import Path

import typer


def register(app: typer.Typer) -> None:
    @app.command("gui")
    def gui(
        root: Path | None = typer.Argument(
            None, help="The folder to work in; it may hold no scan yet."
        ),
        workspace: Path | None = typer.Option(
            None, "--workspace", "-w", exists=True, help="A saved workspace to continue."
        ),
        port: int = typer.Option(0, help="The port to serve on (0: a free one)."),
        window: bool = typer.Option(
            True,
            "--window/--no-window",
            help="An application window (Edge, Chrome); --no-window prints the address.",
        ),
        browser: str | None = typer.Option(
            None, help="The Chromium browser to open the window with (a path or a command)."
        ),
    ) -> None:
        """Open the bexa GUI in a window of its own; closing the window stops it (needs voila)."""
        from bexa.gui.window import serve

        try:
            serve(root, workspace, port=port, window=window, browser=browser)
        except (ImportError, RuntimeError, TimeoutError) as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from None
