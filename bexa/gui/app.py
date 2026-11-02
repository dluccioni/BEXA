"""The bexa GUI in panels: data on the left, the View in the middle, analysis on the right.

``app = bexa.gui.App(root="X:/beamtime/RAW_DATA")`` then ``app`` shows it. The
work runs in a background thread (progress and Cancel at the top), so the
notebook stays responsive; widget updates that follow a job are handed back to
the kernel's event loop. ``app.ws`` is the :class:`bexa.workspace.Workspace`:
everything the GUI does is a logged command that also works in a cell.
"""

from __future__ import annotations

import html
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from bexa._log import CancelledError, get_logger
from bexa.workspace import DataObject, Workspace
from bexa.workspace.jobs import Job, JobRunner

log = get_logger(__name__)

__all__ = ["App", "launch"]

LEFT_TABS = ("Data", "Browse", "Convert")
RIGHT_TABS = ("Analysis", "Fit")
CSS = """<style>
.bexa-objects select, .bexa-files select, .bexa-h5tree select, .bexa-spec select,
.bexa-ops select { font-family: monospace; font-size: 12px; }
.bexa-app .widget-label { font-size: 12px; }
/* a column fills the height left; its tab bar stays, the tab's contents scroll */
.bexa-panel { min-height: 480px; overflow: auto; }
.bexa-panel > .widget-tab { flex: 1 1 auto; min-height: 0; display: flex;
  flex-direction: column; }
.bexa-panel > .widget-tab > .widget-tab-contents { flex: 1 1 auto; min-height: 0;
  overflow: auto; }
.bexa-log { overflow: auto; max-height: 170px; }
</style>"""
# the three columns: how they share the width, and the width below which they wrap
COLUMNS = (("1 1 380px", "360px"), ("1.5 1 520px", "460px"), ("1 1 400px", "380px"))
# the height of the columns: the window less the header and the Log (open or folded away)
COLUMN_HEIGHTS = ("calc(100vh - 300px)", "calc(100vh - 130px)")
MESSAGE_COLOURS = {"info": "#424242", "ok": "#1b5e20", "error": "#b71c1c"}


def _kernel_loop() -> Any:
    """The kernel's event loop (callbacks there run in the main thread), or ``None``."""
    try:
        from IPython import get_ipython

        kernel = getattr(get_ipython(), "kernel", None)
        return getattr(kernel, "io_loop", None)
    except Exception:
        return None


class App:
    """The bexa notebook GUI.

    Parameters
    ----------
    root
        The folder to work in (:meth:`bexa.gui.data_tab.DataTab.use_folder`): the scan
        picker lists its datasets, the Browse tab starts there. It may hold no scan yet.
    workspace
        Continue with this workspace (``Workspace.load(...)``) instead of a new one.
    background
        Run the work in a background thread (``False``: in the calling thread, for tests).
    figures
        ``"auto"`` (interactive ipympl figures when ipympl is installed, pictures
        otherwise), ``"interactive"`` or ``"static"``.
    cache
        Passed to :func:`bexa.open` for every scan.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        workspace: Workspace | None = None,
        *,
        background: bool = True,
        figures: str = "auto",
        cache: Any = True,
    ) -> None:
        import ipywidgets as ipw

        from bexa.gui.analysis_tab import AnalysisTab
        from bexa.gui.browse_tab import BrowseTab
        from bexa.gui.convert_tab import ConvertTab
        from bexa.gui.data_tab import DataTab
        from bexa.gui.dialogs import available
        from bexa.gui.fit_tab import FitTab
        from bexa.gui.log_tab import LogTab
        from bexa.gui.objects_panel import ObjectsPanel
        from bexa.gui.view_tab import ViewTab

        self.ws = workspace if workspace is not None else Workspace(cache=cache)
        self.selection: list[str] = []
        self.folder: Path | None = None  # the folder worked in (Data tab, Use folder)
        self.dialogs = available()  # the system's file dialogs can open (a local kernel)
        self.interactive_figures = self._figures(figures)
        self._main = threading.main_thread()
        self._loop = _kernel_loop() if background else None
        self._pending = False
        self.jobs = JobRunner(background=background, on_change=self._job_changed)
        self.status = ipw.HTML()
        self.progress = ipw.FloatProgress(
            value=0.0, min=0.0, max=1.0, layout=ipw.Layout(width="220px")
        )
        self.cancel = ipw.Button(
            description="Cancel",
            icon="stop",
            button_style="warning",
            layout=ipw.Layout(width="90px", visibility="hidden"),
        )
        self.cancel.on_click(lambda _b: self.jobs.cancel())
        self.message = ipw.HTML()
        self.objects = ObjectsPanel(self)
        self.data = DataTab(self, root)
        self.browse = BrowseTab(self, root)
        self.view = ViewTab(self)
        self.analysis = AnalysisTab(self)
        self.fit = FitTab(self)
        self.convert = ConvertTab(self)
        self.log = LogTab(self)
        self.left_tabs = ipw.Tab(children=[self.data.box, self.browse.box, self.convert.box])
        self.left_tabs.titles = LEFT_TABS
        self.right_tabs = ipw.Tab(children=[self.analysis.box, self.fit.box])
        self.right_tabs.titles = RIGHT_TABS
        view = ipw.Tab(children=[self.view.box])
        view.titles = ("View",)
        self.bottom = ipw.Accordion(children=[self.log.box], selected_index=0)
        self.bottom.set_title(0, "Log")
        self.bottom.observe(lambda _c: self._fit_columns(), names="selected_index")
        self.columns: list[Any] = []  # the three panels side by side
        for parts, (flex, smallest) in zip(
            ([self.objects.box, self.left_tabs], [view], [self.right_tabs]), COLUMNS, strict=True
        ):
            column = ipw.VBox(
                parts, layout=ipw.Layout(flex=flex, min_width=smallest, height=COLUMN_HEIGHTS[0])
            )
            column.add_class("bexa-panel")  # scrolls on its own
            self.columns.append(column)
        figures_note = (
            "interactive figures (ipympl)" if self.interactive_figures else "figures as pictures"
        )
        header = ipw.HBox(
            [
                ipw.HTML(f"<b style='font-size:15px' title='{figures_note}'>bexa</b>"),
                self.progress,
                self.cancel,
                self.status,
            ],
            layout=ipw.Layout(align_items="center", justify_content="flex-start"),
        )
        self.box = ipw.VBox(
            [
                ipw.HTML(CSS),
                header,
                self.message,
                # side by side; on a narrow screen (a notebook) a column wraps below
                ipw.HBox(
                    self.columns,
                    layout=ipw.Layout(flex_flow="row wrap", align_items="flex-start"),
                ),
                self.bottom,
            ]
        )
        self.box.add_class("bexa-app")
        self.ws.observe(self._changed)
        if root is not None and Path(root).is_dir():
            self.data.use_folder(root)

    @staticmethod
    def _figures(figures: str) -> bool:
        if figures == "static":
            return False
        from importlib.util import find_spec

        from bexa._log import in_kernel

        # looked for, not imported: importing ipympl switches matplotlib to its backend
        if find_spec("ipympl") is None:
            if figures == "interactive":
                log.warning("ipympl is not installed: figures are pictures")
            return False
        if not in_kernel():
            return figures == "interactive"
        from bexa.gui.figures import interactive_backend

        if not interactive_backend():
            from IPython import get_ipython

            get_ipython().run_line_magic("matplotlib", "widget")
        return True

    # ------------------------------------------------------------------ display
    def _ipython_display_(self) -> None:
        from IPython.display import display

        display(self.box)

    def _fit_columns(self) -> None:
        """The panels take the height the Log leaves: more of it when the Log is folded."""
        height = COLUMN_HEIGHTS[0 if self.bottom.selected_index == 0 else 1]
        for column in self.columns:
            column.layout.height = height

    def show_tab(self, name: str) -> None:
        """Bring a tab forward: Data, Browse or Convert on the left, Analysis or Fit on the
        right; the View is always shown, the Log opens."""
        if name in LEFT_TABS:
            self.left_tabs.selected_index = LEFT_TABS.index(name)
        elif name in RIGHT_TABS:
            self.right_tabs.selected_index = RIGHT_TABS.index(name)
        elif name == "Log":
            self.bottom.selected_index = 0
        elif name != "View":
            raise ValueError(f"no tab {name!r}: {(*LEFT_TABS, 'View', *RIGHT_TABS, 'Log')}")

    def say(self, text: str, kind: str = "info") -> None:
        """A message under the header (errors in red)."""
        colour = MESSAGE_COLOURS.get(kind, "#424242")
        self.message.value = (
            f"<div style='color:{colour};padding:2px 0'>{html.escape(str(text))}</div>"
        )

    # ------------------------------------------------------------------ selection
    @property
    def current(self) -> str | None:
        """The first selected object."""
        return self.selection[0] if self.selection else None

    def set_selection(self, names: Sequence[str]) -> None:
        """Select objects: the View draws them, the tabs take them as inputs."""
        self.selection = [n for n in names if n in self.ws]
        self.objects.select(self.selection)
        self.view.refresh()
        self.analysis.refresh()
        self.fit.refresh()
        self.convert.refresh()

    def show_results(self, made: Any) -> None:
        """Select what an action made and draw it."""
        objects = [
            m
            for m in (made if isinstance(made, (list, tuple)) else [made])
            if isinstance(m, DataObject)
        ]
        if not objects:
            return
        self.refresh()
        self.set_selection([o.name for o in objects])
        self.show_tab("View")
        self.say(
            f"made {', '.join(o.name for o in objects[:6])}{' ...' if len(objects) > 6 else ''}",
            "ok",
        )

    def data_path(self, path: str | Path) -> None:
        """Put a path into the Data tab and show it."""
        self.data.path.value = str(path)
        self.show_tab("Data")

    def set_workspace(self, workspace: Workspace) -> None:
        """Continue with another workspace (an opened file)."""
        self.ws.unobserve(self._changed)
        self.ws.close()  # the files of the workspace replaced
        self.ws = workspace
        self.ws.observe(self._changed)
        self.analysis.forms.clear()
        self.selection = []
        self.refresh()
        self.say(f"opened the workspace {workspace.name} ({len(workspace)} objects)", "ok")

    # ------------------------------------------------------------------ work
    def guarded(
        self, function: Callable[[], Any], show: Callable[[Any], None] | None = None
    ) -> Any:
        """Run ``function`` now; a failure becomes a red message instead of a traceback."""
        try:
            result = function()
        except Exception as exc:
            self.say(f"{type(exc).__name__}: {exc}", "error")
            log.debug("action failed", exc_info=True)
            return None
        if show is not None:
            show(result)
        return result

    def submit(
        self,
        label: str,
        function: Callable[[], Any],
        done: Callable[[Any], None] | None = None,
        *,
        quiet: bool = False,
    ) -> Job:
        """Run ``function`` in the background; ``done(result)`` follows in the main thread.

        ``quiet``: no message when it starts (the header still shows the job).
        """
        if not quiet:
            self.say(f"{label} ...")

        def finished(result: Any) -> None:
            if done is not None:
                self._soon(lambda: self.guarded(lambda: done(result)))
            else:
                self._soon(lambda: self.say(f"{label}: done", "ok"))

        def failed(error: BaseException) -> None:
            kind = (
                "cancelled"
                if isinstance(error, CancelledError)
                else f"{type(error).__name__}: {error}"
            )
            self._soon(lambda: self.say(f"{label}: {kind}", "error"))

        return self.jobs.submit(label, function, finished, failed)

    def run_op(
        self,
        key: str,
        inputs: Sequence[Any],
        values: dict[str, Any],
        name: str | None = None,
        then: Callable[[list[DataObject]], None] | None = None,
    ) -> Job:
        """Run an operation of the catalog in the background; its results are selected and drawn."""
        from bexa.workspace.catalog import find

        label = find(key).label

        def done(made: list[DataObject]) -> None:
            self.show_results(made)
            if then is not None:
                then(made)

        return self.submit(label, lambda: self.ws.run(key, *inputs, name=name, **values), done)

    # ------------------------------------------------------------------ updates
    def _soon(self, function: Callable[[], None]) -> None:
        """Run ``function`` in the main thread: now when there, else through the kernel's loop."""
        if threading.current_thread() is self._main or self._loop is None:
            function()
        else:
            self._loop.add_callback(function)

    def choose_path(
        self, kind: str, then: Callable[[str], None], start: str | Path | None = None
    ) -> None:
        """The system's dialog for a ``"file"`` or a ``"folder"`` (:mod:`bexa.gui.dialogs`);
        ``then(path)`` follows in the main thread, nothing on Cancel. The dialog waits in a
        thread of its own, so the GUI keeps working while it is open."""
        from bexa.gui.dialogs import choose

        def ask() -> None:
            try:
                path = choose(kind, start or self.folder)
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                self._soon(lambda: self.say(message, "error"))
                return
            if path:
                self._soon(lambda: self.guarded(lambda: then(path)))

        if self._loop is None:
            ask()
        else:
            threading.Thread(target=ask, name="bexa-dialog", daemon=True).start()

    def later(self, seconds: float, function: Callable[[], None]) -> Any:
        """Run ``function`` in the main thread in ``seconds``; ``None`` without a kernel loop
        to wait in. :meth:`cancel_later` takes back what it returns."""
        if self._loop is None:
            return None
        return self._loop.call_later(seconds, function)

    def cancel_later(self, handle: Any) -> None:
        if handle is not None and self._loop is not None:
            self._loop.remove_timeout(handle)

    def _changed(self, event: str, info: dict[str, Any]) -> None:
        if event == "log":
            self._soon(self.log.refresh)
            return
        if self._loop is None:
            self.refresh()
            return
        if not self._pending:  # many changes in a row (an operation's results): one refresh
            self._pending = True
            self._loop.add_callback(self._refresh_pending)

    def _refresh_pending(self) -> None:
        self._pending = False
        self.refresh()

    def refresh(self) -> None:
        """Bring every panel up to date with the workspace."""
        self.selection = [n for n in self.selection if n in self.ws]
        self.objects.refresh()
        self.analysis.refresh()
        self.fit.refresh()
        self.convert.refresh()
        self.log.refresh()
        self.view.refresh()

    def close(self) -> None:
        """Stop the jobs, close the files of every scan and file shown, and close the widgets.

        ipywidgets keeps every widget until it is closed, and with it this app and its
        scans: close an app that is not needed any more to free its files and memory.
        """
        self.jobs.cancel()
        self.data.stop_auto()
        self.ws.unobserve(self._changed)
        self.ws.close()
        if self.browse.inspector is not None:
            self.browse.inspector.close()
        for pane in (self.view.pane, self.fit.pane, self.browse.pane):
            pane.close()
        _close_widgets(self.box)

    def _job_changed(self, job: Job) -> None:
        running = job.status == "running"
        self.progress.value = job.fraction if running else (1.0 if job.status == "done" else 0.0)
        self.progress.bar_style = {
            "done": "success",
            "failed": "danger",
            "cancelled": "warning",
        }.get(job.status, "")
        self.cancel.layout.visibility = "visible" if running else "hidden"
        detail = f" ({job.message})" if running and job.message else ""
        self.status.value = f"&nbsp;{html.escape(job.label)}: {job.status}{html.escape(detail)}"


def _close_widgets(widget: Any) -> None:
    """Close a widget and every widget inside it."""
    for child in getattr(widget, "children", ()):
        _close_widgets(child)
    widget.close()


def launch(root: str | Path | None = None, **kwargs: Any) -> App:
    """Make an :class:`App` and show it (the last line of a cell shows it too)."""
    app = App(root, **kwargs)
    from bexa._log import in_kernel

    if in_kernel():
        from IPython.display import display

        display(app.box)
    return app
