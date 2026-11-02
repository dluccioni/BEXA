"""The system's own dialogs to choose a file or a folder, for the GUI's path fields.

A browser does not give the kernel the full path of a file chosen in it, so the kernel opens
the dialog itself, in a process of its own: tkinter's (the Explorer dialogs on Windows), else
PowerShell's on Windows, AppleScript's on macOS, zenity's or kdialog's on Linux. That needs
the kernel to run on the computer in front of the user (``bexa gui``, a local JupyterLab, ssh
with X forwarding), not on a remote server: :func:`available` says whether it can.
"""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

__all__ = ["available", "choose", "dialog_command"]

TITLES = {"file": "Choose a file", "folder": "Choose a folder"}
TK_SCRIPT = """
import sys, tkinter
from tkinter import filedialog
kind, start, title = sys.argv[1:4]
root = tkinter.Tk()
root.withdraw()
root.attributes("-topmost", True)
ask = filedialog.askdirectory if kind == "folder" else filedialog.askopenfilename
path = ask(parent=root, initialdir=start or None, title=title)
sys.stdout.buffer.write(str(path or "").encode("utf-8"))
"""
POWERSHELL_SCRIPT = """
Add-Type -AssemblyName System.Windows.Forms
$owner = New-Object System.Windows.Forms.Form -Property @{ TopMost = $true }
if ($env:BEXA_DIALOG_KIND -eq 'folder') {
    $dialog = New-Object System.Windows.Forms.FolderBrowserDialog
    $dialog.Description = $env:BEXA_DIALOG_TITLE
    $dialog.SelectedPath = $env:BEXA_DIALOG_START
} else {
    $dialog = New-Object System.Windows.Forms.OpenFileDialog
    $dialog.Title = $env:BEXA_DIALOG_TITLE
    $dialog.InitialDirectory = $env:BEXA_DIALOG_START
}
if ($dialog.ShowDialog($owner) -eq 'OK') {
    $path = $dialog.FileName
    if ($env:BEXA_DIALOG_KIND -eq 'folder') { $path = $dialog.SelectedPath }
    $bytes = [Text.Encoding]::UTF8.GetBytes($path)
    [Console]::OpenStandardOutput().Write($bytes, 0, $bytes.Length)
}
$owner.Dispose()
"""


def _desktop() -> bool:
    """Whether a window opened by this process shows on the screen of the person using it."""
    if sys.platform == "win32":
        return bool(os.environ.get("SESSIONNAME"))  # an interactive logon, not a service or ssh
    if sys.platform == "darwin":
        return not os.environ.get("SSH_CONNECTION")
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def dialog_command(
    kind: str, start: str | Path | None = None, title: str | None = None
) -> tuple[list[str], dict[str, str]] | None:
    """The command that shows the dialog here and prints the path chosen, with the
    environment variables it reads; ``None`` when no dialog can be shown."""
    if kind not in TITLES:
        raise ValueError(f"kind must be 'file' or 'folder', not {kind!r}")
    if not _desktop():
        return None
    start_text = str(start) if start and Path(start).is_dir() else ""
    title = title or TITLES[kind]
    if find_spec("tkinter") is not None:
        return [sys.executable, "-c", TK_SCRIPT, kind, start_text, title], {}
    if sys.platform == "win32" and shutil.which("powershell"):
        encoded = base64.b64encode(POWERSHELL_SCRIPT.encode("utf-16-le")).decode("ascii")
        env = {"BEXA_DIALOG_KIND": kind, "BEXA_DIALOG_START": start_text}
        env["BEXA_DIALOG_TITLE"] = title
        return ["powershell", "-NoProfile", "-STA", "-EncodedCommand", encoded], env
    if sys.platform == "darwin" and shutil.which("osascript"):
        what = "folder" if kind == "folder" else "file"
        where = f' default location (POSIX file "{_quoted(start_text)}")' if start_text else ""
        script = f'POSIX path of (choose {what} with prompt "{_quoted(title)}"{where})'
        return ["osascript", "-e", script], {}
    if shutil.which("zenity"):
        argv = ["zenity", "--file-selection", f"--title={title}"]
        argv += ["--directory"] if kind == "folder" else []
        argv += [f"--filename={start_text}/"] if start_text else []
        return argv, {}
    if shutil.which("kdialog"):
        verb = "--getexistingdirectory" if kind == "folder" else "--getopenfilename"
        return ["kdialog", verb, start_text or str(Path.home()), "--title", title], {}
    return None


def available() -> bool:
    """Whether :func:`choose` can show a dialog here."""
    return dialog_command("file") is not None


def choose(
    kind: str = "file", start: str | Path | None = None, title: str | None = None
) -> str | None:
    """Show the system's dialog for a ``"file"`` or a ``"folder"``, starting in ``start``;
    the path chosen, or ``None`` on Cancel. Blocks until the dialog closes."""
    found = dialog_command(kind, start, title)
    if found is None:
        raise RuntimeError("no file dialog can be shown here (a remote kernel): type the path")
    argv, env = found
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # no console window flashing on Windows
    done = subprocess.run(
        argv, capture_output=True, env={**os.environ, **env}, creationflags=flags, check=False
    )
    path = done.stdout.decode("utf-8", errors="replace").strip()
    return str(Path(path)) if path else None  # tkinter gives C:/data: made C:\data


def _quoted(text: str) -> str:
    """Text inside an AppleScript string."""
    return text.replace("\\", "\\\\").replace('"', '\\"')
