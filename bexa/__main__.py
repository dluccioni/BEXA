"""Entry point for ``python -m bexa``: runs the command-line interface."""

from bexa.cli.main import app

if __name__ == "__main__":
    app(prog_name="bexa")
