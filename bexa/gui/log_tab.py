"""The Log tab: every command run, the script that replays them, workspace files, jobs."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

__all__ = ["LogTab"]

STATUS_COLOURS = {"ok": "#1b5e20", "failed": "#b71c1c", "cancelled": "#e65100", "note": "#757575"}
SHOWN_ENTRIES = 300


class LogTab:
    """The command log, the script, saving and opening the workspace."""

    def __init__(self, app: Any) -> None:
        import ipywidgets as ipw

        self.app = app
        self.entries = ipw.HTML()
        self.script = ipw.Textarea(rows=14, layout=ipw.Layout(width="100%"))
        show_script = ipw.Button(
            description="Show script",
            icon="code",
            tooltip="the commands that worked, as a Python script",
        )
        show_script.on_click(lambda _b: self._show_script())
        self.script_path = ipw.Text(placeholder="session.py", layout=ipw.Layout(width="300px"))
        save_script = ipw.Button(description="Save script", icon="save")
        save_script.on_click(lambda _b: self._save_script())
        self.ws_path = ipw.Text(placeholder="session.bexa.h5", layout=ipw.Layout(width="300px"))
        save_ws = ipw.Button(
            description="Save workspace",
            icon="save",
            button_style="primary",
            tooltip="objects, ROIs, settings and log in one HDF5 file; scans by recipe",
        )
        open_ws = ipw.Button(
            description="Open workspace",
            icon="folder-open",
            tooltip="replaces the objects of this app",
        )
        save_ws.on_click(lambda _b: self._save_workspace())
        open_ws.on_click(lambda _b: self._open_workspace())
        self.jobs = ipw.HTML()
        self.box = ipw.VBox(
            [
                ipw.HTML(
                    "<b>Commands</b> <span style='color:#757575;font-size:11px'>"
                    "each line runs the same "
                    "in a notebook cell, with <code>ws = app.ws</code></span>"
                ),
                self.entries,
                ipw.HBox([show_script, self.script_path, save_script]),
                self.script,
                ipw.HTML("<b>Workspace file</b>"),
                ipw.HBox([self.ws_path, save_ws, open_ws]),
                ipw.HTML("<b>Jobs</b>"),
                self.jobs,
            ]
        )
        self.refresh()

    def refresh(self) -> None:
        log = self.app.ws.log[-SHOWN_ENTRIES:]
        rows = []
        for entry in reversed(log):
            colour = STATUS_COLOURS.get(entry.status, "black")
            message = (
                f"<br><span style='color:#b71c1c'>{html.escape(entry.message)}</span>"
                if entry.message
                else ""
            )
            took = f"{entry.seconds:.2f} s" if entry.seconds else ""
            rows.append(
                f"<tr><td style='color:#757575;white-space:nowrap'>"
                f"{html.escape(entry.time[11:])}</td>"
                f"<td style='color:{colour}'>{entry.status}</td>"
                f"<td style='white-space:nowrap'>{took}</td>"
                f"<td><code>{html.escape(entry.command)}</code>{message}</td></tr>"
            )
        self.entries.value = (
            (
                "<div style='max-height:320px;overflow:auto;font-size:11px'><table>"
                + "".join(rows)
                + "</table></div>"
            )
            if rows
            else "<i>nothing run yet</i>"
        )
        self._show_jobs()

    def _show_jobs(self) -> None:
        jobs = self.app.jobs.history[-30:]
        rows = "".join(
            f"<tr><td>{html.escape(j.label)}</td>"
            f"<td style='color:{STATUS_COLOURS.get(j.status, 'black')}'>"
            f"{j.status}</td><td>{j.seconds:.2f} s</td>"
            f"<td>{html.escape(str(j.error) if j.error else '')}</td></tr>"
            for j in reversed(jobs)
        )
        self.jobs.value = (
            f"<table style='font-size:11px'>{rows}</table>" if rows else "<i>no jobs yet</i>"
        )

    def _show_script(self) -> None:
        self.script.value = self.app.ws.script()

    def _save_script(self) -> None:
        path = self.script_path.value.strip() or "session.py"
        self.app.guarded(
            lambda: self.app.ws.save_script(path),
            show=lambda out: self.app.say(f"saved {out}", "ok"),
        )

    def _save_workspace(self) -> None:
        path = self.ws_path.value.strip() or "session.bexa.h5"
        self.app.submit(
            f"saving {Path(path).name}",
            lambda: self.app.ws.save(path),
            done=lambda out: self.app.say(f"saved {out}", "ok"),
        )

    def _open_workspace(self) -> None:
        path = self.ws_path.value.strip()
        if not path:
            self.app.say("type the workspace file to open", "error")
            return

        def load() -> Any:
            from bexa.workspace import Workspace

            return Workspace.load(path, cache=self.app.ws.cache)

        self.app.submit(f"opening {Path(path).name}", load, done=self.app.set_workspace)
