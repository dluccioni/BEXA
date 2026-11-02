"""The object system behind the bexa GUI, usable from scripts too.

A :class:`Workspace` holds named :class:`DataObject` s: raw scans (never
changed), scan sets (several scans processed together) and results in memory
(stacks, images, maps, curves, volumes, colour pictures, tables, text). Every
operation of :mod:`bexa.workspace.catalog` makes new objects from existing ones
and logs the command it ran, so :meth:`Workspace.script` replays a session.

>>> from bexa.workspace import Workspace
>>> ws = Workspace()
>>> scan = ws.open_scan("RAW_DATA/sample/dataset", scan=7)
>>> image = ws.run("reduce_frames", scan, how="sum")
"""

from bexa.workspace.objects import KINDS, DataObject, ScanSet, kind_of
from bexa.workspace.workspace import LogEntry, Workspace

__all__ = ["KINDS", "DataObject", "LogEntry", "ScanSet", "Workspace", "kind_of"]
