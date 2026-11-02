"""The bexa GUI in a window of its own: ``bexa gui`` on the command line, or :func:`serve`.

The notebook GUI (:class:`bexa.gui.App`) without the notebook around it. Voilà
runs a one-cell notebook that makes the App and serves its widgets alone, on
this computer only (127.0.0.1, with a token), and a Chromium browser (Edge,
Chrome, Chromium) shows the page as an application window: no tabs, no address
bar, a profile of its own. Closing the window stops the server and the kernel.
Without such a browser the page opens in the default browser, and Ctrl+C in the
console stops it. Needs voila (``pip install voila``) on top of the GUI's
ipywidgets and ipympl.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bexa._log import get_logger

log = get_logger(__name__)

__all__ = ["GuiServer", "app_notebook", "find_browser", "serve", "start"]

WINDOW_SIZE = (1800, 1000)  # before it is maximized
START_TIMEOUT = 60.0  # seconds for Voilà to listen; the App itself appears ~10 s later
NOTEBOOK_NAME = "bexa.ipynb"  # Voilà titles the window after the notebook
BROWSERS = {  # executables by platform, in order of preference
    "win32": (
        r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe",
        r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe",
        r"%ProgramFiles%\Google\Chrome\Application\chrome.exe",
        r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe",
        r"%LocalAppData%\Google\Chrome\Application\chrome.exe",
        "msedge",
        "chrome",
    ),
    "darwin": (
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ),
    "linux": (
        "microsoft-edge",
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
    ),
}


def app_notebook(
    root: str | Path | None = None,
    workspace: str | Path | None = None,
    *,
    figures: str = "auto",
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    """The notebook Voilà runs: one cell that makes the :class:`~bexa.gui.App` and shows it.

    ``root`` is the folder to work in (scans or not yet), ``workspace`` a
    saved workspace to continue (``Workspace.save``), ``cwd`` the folder relative
    paths are read from (the kernel would otherwise start in the notebook's
    temporary folder). The cell puts this bexa checkout on ``sys.path``, since
    bexa is used from a checkout rather than installed.
    """
    checkout = Path(os.path.abspath(__file__)).parents[2]  # abspath: a share stays a drive letter
    folder = Path(cwd) if cwd is not None else Path.cwd()
    opened = f"Workspace.load({str(Path(workspace))!r})" if workspace is not None else "None"
    start = repr(str(Path(root))) if root is not None else "None"
    source = "\n".join(
        [
            "import os, sys",
            f"sys.path.insert(0, {str(checkout)!r})",
            f"os.chdir({str(folder)!r})",
            "import bexa",
            "from bexa.workspace import Workspace",
            'bexa.notebook.setup(log_level="WARNING")',
            f"app = bexa.gui.App(root={start}, workspace={opened}, figures={figures!r})",
            "app",
        ]
    )
    return {
        "cells": [
            {
                "cell_type": "code",
                "id": "bexa-app",
                "metadata": {},
                "source": source,
                "outputs": [],
                "execution_count": None,
            }
        ],
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def find_browser(name: str | Path | None = None) -> Path | None:
    """A Chromium browser that opens application windows (``--app``): Edge, Chrome, Chromium.

    ``name`` is a path or a command to use instead; ``None`` when none is found.
    """
    if name is not None:
        given = Path(os.path.expandvars(str(name)))
        found = given if given.is_file() else shutil.which(str(name))
        return Path(found) if found else None
    platform = "linux" if sys.platform.startswith("linux") else sys.platform
    for candidate in BROWSERS.get(platform, ()):
        path = Path(os.path.expandvars(candidate))
        if path.is_absolute() and path.is_file():
            return path
        if not path.is_absolute() and (which := shutil.which(candidate)):
            return Path(which)
    return None


def free_port() -> int:
    """A TCP port nobody listens on (the system picks it)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def voila_command(notebook: Path, port: int, token: str) -> list[str]:
    """The command that serves ``notebook`` with Voilà on 127.0.0.1:``port``."""
    return [
        sys.executable,
        "-m",
        "voila",
        str(notebook),
        "--no-browser",
        f"--port={port}",
        "--Voila.ip=127.0.0.1",
        f"--Voila.token={token}",
        "--theme=light",
        "--show_tracebacks=True",
    ]


def window_command(
    browser: Path, url: str, profile: Path, size: tuple[int, int] = WINDOW_SIZE
) -> list[str]:
    """The command that opens ``url`` as an application window of ``browser``.

    A profile of its own makes the browser start a process of its own, which
    ends when the window closes (with the user's profile it would hand the
    window to the running browser and return at once).
    """
    return [
        str(browser),
        f"--app={url}",
        f"--user-data-dir={profile}",
        f"--window-size={size[0]},{size[1]}",
        "--start-maximized",  # three panels side by side
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-sync",
    ]


@dataclass
class GuiServer:
    """A running GUI server: its page, its process and its folder (:func:`start`)."""

    url: str
    process: subprocess.Popen[Any]
    folder: Path
    window: subprocess.Popen[Any] | None = None
    log_file: Path = field(default_factory=Path)

    def open_window(
        self, browser: str | Path | None = None, size: tuple[int, int] = WINDOW_SIZE
    ) -> bool:
        """Show the page as an application window; ``False`` when it went to the default
        browser instead (no Edge, Chrome or Chromium found)."""
        found = find_browser(browser)
        if found is None:
            webbrowser.open(self.url)
            return False
        self.window = subprocess.Popen(
            window_command(found, self.url, self.folder / "window-profile", size)
        )
        return True

    def running(self) -> bool:
        return self.process.poll() is None

    def wait(self) -> None:
        """Until the window closes (or, without one, until the server stops or Ctrl+C)."""
        try:
            while self.running():
                if self.window is not None and self.window.poll() is not None:
                    return
                time.sleep(0.5)
        except KeyboardInterrupt:
            return

    def log_tail(self, lines: int = 30) -> str:
        try:
            text = self.log_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])

    def stop(self) -> None:
        """Stop the window, the server (its kernels follow) and remove the folder."""
        for process in (self.window, self.process):
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(10)
                except subprocess.TimeoutExpired:
                    process.kill()
        for _ in range(10):  # the browser lets go of its profile a moment after it exits
            shutil.rmtree(self.folder, ignore_errors=True)
            if not self.folder.exists():
                return
            time.sleep(0.5)


def start(
    root: str | Path | None = None,
    workspace: str | Path | None = None,
    *,
    port: int = 0,
    figures: str = "auto",
    command: list[str] | None = None,
) -> GuiServer:
    """Start the GUI server and return once it listens; :meth:`GuiServer.open_window` shows it.

    ``port`` 0 takes a free one. ``command`` replaces the Voilà command (tests);
    its ``{notebook}``, ``{port}`` and ``{token}`` fields are filled in.
    """
    if command is None and _missing("voila"):
        raise ImportError(
            "the GUI window needs voila: pip install voila (or use the GUI in a notebook)"
        )
    folder = Path(tempfile.mkdtemp(prefix="bexa_gui_window_"))
    notebook = folder / NOTEBOOK_NAME
    notebook.write_text(
        json.dumps(app_notebook(root, workspace, figures=figures), indent=1), encoding="utf-8"
    )
    port = int(port) or free_port()
    token = secrets.token_urlsafe(16)
    if command is None:
        argv = voila_command(notebook, port, token)
    else:
        argv = [a.format(notebook=notebook, port=port, token=token) for a in command]
    log_file = folder / "server.log"
    with open(log_file, "w", encoding="utf-8") as out:
        process = subprocess.Popen(
            argv, stdout=out, stderr=subprocess.STDOUT, cwd=folder, stdin=subprocess.DEVNULL
        )
    server = GuiServer(f"http://127.0.0.1:{port}/?token={token}", process, folder, None, log_file)
    deadline = time.monotonic() + START_TIMEOUT
    while time.monotonic() < deadline:
        if not server.running():
            tail = server.log_tail()
            server.stop()
            raise RuntimeError(f"the GUI server stopped at its start:\n{tail}")
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            return server
        except OSError:
            time.sleep(0.3)
    server.stop()
    raise TimeoutError(f"the GUI server did not listen on port {port} within {START_TIMEOUT:.0f} s")


def serve(
    root: str | Path | None = None,
    workspace: str | Path | None = None,
    *,
    port: int = 0,
    window: bool = True,
    browser: str | Path | None = None,
    figures: str = "auto",
) -> None:
    """Show the GUI in a window of its own and return when the window is closed.

    ``window=False`` only prints the address (open it in any browser on this
    computer; Ctrl+C stops). ``browser`` names the Chromium browser to use.
    """
    server = start(root, workspace, port=port, figures=figures)
    try:
        # flushed: the address must reach a log file at once (nohup, an ssh session)
        if window and server.open_window(browser):
            print(f"bexa GUI in its window ({server.url}); close the window to stop", flush=True)
        else:
            if window:
                print(
                    "no Edge, Chrome or Chromium found: the GUI opens in the default browser",
                    flush=True,
                )
            print(f"bexa GUI at {server.url} ; Ctrl+C here stops it", flush=True)
        server.wait()
    finally:
        server.stop()


def _missing(module: str) -> bool:
    from importlib.util import find_spec

    return find_spec(module) is None
