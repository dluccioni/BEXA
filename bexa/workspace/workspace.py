"""The workspace: named objects, named ROIs, the operations run on them, and their log.

Every change goes through a method that logs the Python command doing it
(``ws.open_scan(...)``, ``ws.set_roi(...)``, ``ws.run("roi_curves", ...)``), so
:meth:`Workspace.script` replays a whole session in a script or notebook, and
:meth:`Workspace.save` keeps the objects, ROIs and log in one HDF5 file.
Operations never change their inputs; raw scans stay as they were read.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from bexa._log import CancelledError, get_logger, progress
from bexa.core.roi import ROI
from bexa.workspace.objects import DataObject, ScanSet, scan_label
from bexa.workspace.source import call_text, jsonable

log = get_logger(__name__)

__all__ = ["LogEntry", "Workspace"]

WORKSPACE_MARKER = "bexa-workspace-v1"
Observer = Callable[[str, dict[str, Any]], None]


def timestamp() -> str:
    """The local time as ``2026-11-01 14:05:09``: when a command of the log ran."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class LogEntry:
    """One logged command: what ran, how long, whether it worked, what it made."""

    command: str
    seconds: float = 0.0
    status: str = "ok"  # ok, failed, cancelled, note
    message: str = ""
    results: list[str] = field(default_factory=list)
    time: str = field(default_factory=timestamp)


def _clean(name: str) -> str:
    text = re.sub(r"[^\w.+-]+", "_", str(name)).strip("_")
    return text or "object"


class Workspace:
    """Named objects and ROIs, the operations of :mod:`bexa.workspace.catalog`, and their log.

    Parameters
    ----------
    name
        Shown in the GUI and the saved file.
    cache
        Passed to :func:`bexa.open` for every scan (``True``: bexa's cache).

    Examples
    --------
    >>> ws = Workspace()
    >>> scan = ws.open_scan("RAW_DATA/sample/dataset", scan=7)
    >>> ws.set_roi("peak", y=(100, 160), x=(220, 300))
    >>> curves = ws.run("roi_curves", scan, rois=["peak"])
    >>> ws.run("fit_peaks", curves[0], shape="pseudo_voigt")
    >>> print(ws.script())
    """

    def __init__(self, name: str = "workspace", cache: Any = True) -> None:
        self.name = name
        self.cache = cache
        self.objects: dict[str, DataObject] = {}
        self.rois: dict[str, ROI] = {}
        self.log: list[LogEntry] = []
        self.settings: dict[str, Any] = {}  # remembered form values and the like (saved too)
        self._observers: list[Observer] = []

    # ------------------------------------------------------------------ events
    def observe(self, callback: Observer) -> None:
        """Call ``callback(event, info)`` on every change: added, removed, renamed, changed,
        rois, log."""
        self._observers.append(callback)

    def unobserve(self, callback: Observer) -> None:
        if callback in self._observers:
            self._observers.remove(callback)

    def _emit(self, event: str, **info: Any) -> None:
        for callback in list(self._observers):
            try:
                callback(event, info)
            except Exception:  # a broken view must not break the workspace
                log.exception("workspace observer failed on %s", event)

    # ------------------------------------------------------------------ objects
    def __getitem__(self, key: str | DataObject) -> DataObject:
        if isinstance(key, DataObject):
            return key
        try:
            return self.objects[str(key)]
        except KeyError:
            raise KeyError(f"no object {key!r}; the workspace holds {list(self.objects)}") from None

    def __contains__(self, key: object) -> bool:
        return str(key) in self.objects

    def __iter__(self) -> Iterator[str]:
        return iter(list(self.objects))

    def __len__(self) -> int:
        return len(self.objects)

    def __repr__(self) -> str:
        return f"Workspace({self.name!r}, {len(self)} objects, {len(self.rois)} ROIs)"

    def names(self, kinds: Sequence[str] | None = None) -> list[str]:
        """The object names (of the given kinds), oldest first."""
        return [n for n, o in self.objects.items() if kinds is None or o.kind in kinds]

    def unique(self, name: str) -> str:
        """``name`` made safe and unique in the workspace (``name_2``, ``name_3``...)."""
        base = _clean(name)
        if base not in self.objects:
            return base
        k = 2
        while f"{base}_{k}" in self.objects:
            k += 1
        return f"{base}_{k}"

    def _insert(self, obj: DataObject) -> DataObject:
        obj.name = self.unique(obj.name)
        self.objects[obj.name] = obj
        self._emit("added", name=obj.name)
        return obj

    def add(self, data: Any, name: str | None = None, **fields: Any) -> DataObject:
        """Add data made outside the workspace (an array, a table, a scan) as an object.

        The log gets a comment only: a script cannot remake such data. Prefer
        :meth:`open_scan`, :meth:`open_file` and :meth:`run`, which log commands.
        """
        if isinstance(data, DataObject):
            obj = data if name is None else data.copy(name)
        else:
            obj = DataObject(name or "object", data, **fields)
        self._insert(obj)
        self.record(f"# {obj.name} added from code ({obj.describe()})", status="note")
        return obj

    def remove(self, *names: str) -> None:
        """Remove objects (raw scans too: their files stay)."""
        for name in names:
            self.objects.pop(self[name].name)
            self._emit("removed", name=name)
        self.record(call_text("ws.remove", *names))

    def rename(self, old: str, new: str) -> str:
        """Rename an object; returns the name it got (made unique)."""
        obj = self[old]
        target = self.unique(new) if _clean(new) != old else old
        if target == old:
            return old
        self.objects = {(target if k == old else k): v for k, v in self.objects.items()}
        obj.name = target
        self._emit("renamed", old=old, new=target)
        self.record(call_text("ws.rename", old, target))
        return target

    def duplicate(self, name: str, new_name: str | None = None) -> DataObject:
        """A copy with its own arrays (to keep a version before trying something)."""
        copy = self[name].copy(new_name or f"{name}_copy")
        copy.raw = copy.kind in ("scan", "scanset")
        copy.parents = [name]
        self._insert(copy)
        self.record(call_text("ws.duplicate", name, new_name=copy.name), results=[copy.name])
        return copy

    def set_notes(self, name: str, text: str) -> None:
        self[name].notes = str(text)
        self._emit("changed", name=name)
        self.record(call_text("ws.set_notes", name, str(text)))

    def clear(self) -> None:
        """Remove every object and ROI (the log stays)."""
        self.objects.clear()
        self.rois.clear()
        self._emit("cleared")
        self.record("ws.clear()")

    def close(self) -> None:
        """Close the files of every scan (they reopen by themselves when read again)."""
        for obj in self.objects.values():
            scans = (
                [obj.data]
                if obj.kind == "scan"
                else obj.data.scans
                if obj.kind == "scanset"
                else []
            )
            for scan in scans:
                scan.close()

    # --------------------------------------------------------------------- ROIs
    def set_roi(self, name: str, roi: ROI | Mapping[str, Any] | None = None, **ranges: Any) -> ROI:
        """Define (or replace) a named ROI: ``set_roi("peak", y=(10, 50), x=(20, 80))``.

        Pixel ranges are half-open full-resolution indices, other ranges
        inclusive motor values (see :class:`bexa.ROI`).
        """
        if isinstance(roi, ROI):
            made = roi.copy()
        else:
            made = ROI(dict(roi or {}), **ranges)
        key = _clean(name)
        self.rois[key] = made
        self._emit("rois", name=key)
        written = {d: list(r) for d, r in made.ranges.items()}
        self.record(call_text("ws.set_roi", key, **{d: tuple(r) for d, r in written.items()}))
        return made

    def remove_roi(self, name: str) -> None:
        self.rois.pop(name, None)
        self._emit("rois", name=name)
        self.record(call_text("ws.remove_roi", name))

    def roi(self, name: str) -> ROI:
        try:
            return self.rois[name]
        except KeyError:
            raise ValueError(
                f"no ROI named {name!r}; defined: {list(self.rois) or 'none'}"
            ) from None

    # ------------------------------------------------------------------ opening
    def open_scan(
        self,
        path: str | Path | None = None,
        scan: Any = None,
        *,
        name: str | None = None,
        **options: Any,
    ) -> DataObject:
        """Open a scan of any layout bexa reads (:func:`bexa.open`) as a raw object."""
        import bexa

        options.setdefault("cache", self.cache)
        t0 = time.perf_counter()
        opened = bexa.open(path, scan=scan, **options)
        label = name or _scan_name(opened)
        obj = self._insert(DataObject(label, opened, meta=_scan_meta(opened)))
        logged = {k: v for k, v in options.items() if not (k == "cache" and v is self.cache)}
        self.record(
            call_text("ws.open_scan", path, scan=scan, name=obj.name, **logged),
            time.perf_counter() - t0,
            results=[obj.name],
        )
        return obj

    def open_scans(
        self,
        path: str | Path | Sequence[str | Path],
        scans: Any = None,
        *,
        name: str | None = None,
        as_set: bool = True,
        status: str = "finished",
        type: str | None = None,
        **options: Any,
    ) -> list[DataObject]:
        """Open several scans of a dataset (folder, master file, SPEC session): one scan set,
        or one object per scan with ``as_set=False``. ``scans``, ``status`` and ``type`` (the
        scan type, ``"fscan2d"``) choose them as ``Dataset.select_numbers`` does; the log
        keeps the scan numbers chosen."""
        import bexa

        t0 = time.perf_counter()
        dataset = bexa.open_dataset(path, cache=self.cache, **options)
        ids = dataset.select_numbers(scans, status=status, type=type)
        if not ids:
            raise ValueError(f"no scan of {dataset.name} matches {scans!r}")
        opened = [
            dataset.scan(i) for i in progress(ids, total=len(ids), desc="opening", enabled=False)
        ]
        if as_set:
            made = [
                self._insert(DataObject(name or f"{_clean(dataset.name)}_set", ScanSet(opened)))
            ]
        else:
            made = [self._insert(DataObject(_scan_name(s), s, meta=_scan_meta(s))) for s in opened]
        self.record(
            call_text(
                "ws.open_scans", path, scans=ids, name=name, as_set=as_set, status=status, **options
            ),
            time.perf_counter() - t0,
            results=[o.name for o in made],
        )
        return made

    def open_scan_set(
        self,
        recipes: Sequence[Mapping[str, Any]],
        *,
        name: str = "scans",
        labels: Sequence[str] | None = None,
    ) -> DataObject:
        """A scan set from the recipes of its scans (``bexa.open(**recipe)`` each)."""
        import bexa

        scans = [bexa.open(**{"cache": self.cache, **dict(r)}) for r in recipes]
        obj = self._insert(DataObject(name, ScanSet(scans, labels)))
        self.record(
            call_text(
                "ws.open_scan_set",
                [dict(r) for r in recipes],
                name=obj.name,
                labels=list(labels) if labels else None,
            ),
            results=[obj.name],
        )
        return obj

    def add_scans(
        self,
        scans: Sequence[Any],
        *,
        as_set: bool = False,
        name: str | None = None,
        labels: Sequence[str] | None = None,
    ) -> list[DataObject]:
        """Scans opened elsewhere (the scan picker) as objects, logged by their recipes."""
        scans = list(scans)
        if not scans:
            raise ValueError("no scans selected")
        recipes = [s.recipe for s in scans]
        if as_set:
            obj = self._insert(DataObject(name or "scans", ScanSet(scans, labels)))
            command = call_text(
                "ws.open_scan_set",
                [_recipe(r) for r in recipes],
                name=obj.name,
                labels=list(labels) if labels else None,
            )
            self.record(command, results=[obj.name])
            return [obj]
        made = []
        for scan, recipe in zip(scans, recipes, strict=True):
            obj = self._insert(DataObject(_scan_name(scan), scan, meta=_scan_meta(scan)))
            opened = _recipe(recipe)
            path = opened.pop("path", None)
            self.record(
                call_text("ws.open_scan", path, name=obj.name, **opened), results=[obj.name]
            )
            made.append(obj)
        return made

    def open_file(
        self,
        path: str | Path,
        *,
        dataset: str | None = None,
        index: str | None = None,
        name: str | None = None,
    ) -> list[DataObject]:
        """Objects from a file that is not a scan: a cube, an HDF5 dataset (sliced by ``index``),
        npy, TIFF, PNG, CSV (:func:`bexa.workspace.files.load_file`)."""
        from bexa.workspace.files import load_file

        t0 = time.perf_counter()
        made = [self._insert(o) for o in load_file(path, dataset, index, name)]
        self.record(
            call_text("ws.open_file", str(path), dataset=dataset, index=index, name=name),
            time.perf_counter() - t0,
            results=[o.name for o in made],
        )
        return made

    def demo(self, kind: str = "mosa") -> list[DataObject]:
        """Synthetic data to try things on (:func:`bexa.demo`): a scan, or a scan set."""
        from bexa.testing.demo import demo

        made_data = demo(kind)
        if hasattr(made_data, "scans") and not hasattr(made_data, "structure"):  # a Dataset
            scans = [made_data.scan(i) for i in made_data.scans]
            made = [self._insert(DataObject(f"demo_{kind}", ScanSet(scans)))]
        else:
            made = [self._insert(DataObject(f"demo_{kind}", made_data, meta=_scan_meta(made_data)))]
        self.record(call_text("ws.demo", kind), results=[o.name for o in made])
        return made

    # --------------------------------------------------------------- operations
    def run(
        self, operation: Any, /, *inputs: Any, name: str | None = None, **params: Any
    ) -> list[DataObject]:
        """Run an operation of the catalog on objects (by name or as objects); its results
        become new objects, which are returned. A scan set given to an operation of single
        scans runs it on every scan and stacks the results over the scans."""
        from bexa.workspace.catalog import Op, find

        spec = operation if isinstance(operation, Op) else find(str(operation))
        if len(inputs) != len(spec.inputs):
            wanted = ", ".join(i.label for i in spec.inputs) or "no input"
            raise TypeError(f"{spec.label} takes {len(spec.inputs)} input(s): {wanted}")
        given = [self._input(item, slot) for item, slot in zip(inputs, spec.inputs, strict=True)]
        values = spec.check(params)
        resolved = {k: self._param(spec.param(k), v) for k, v in values.items()}
        command = spec.command([_names_of(i) for i in inputs], params, name)
        t0 = time.perf_counter()
        try:
            results = self._call(spec, given, resolved)
        except Exception as exc:
            status = "cancelled" if isinstance(exc, CancelledError) else "failed"
            self.record(command, time.perf_counter() - t0, status, f"{type(exc).__name__}: {exc}")
            raise
        made = self._keep(results, given, command, name)
        self.record(command, time.perf_counter() - t0, results=[o.name for o in made])
        return made

    def _input(self, item: Any, slot: Any) -> Any:
        if slot.many:
            items = [item] if isinstance(item, (str, DataObject)) else list(item)
            objects = [self[i] for i in items]
            for obj in objects:
                if obj.kind not in slot.kinds:
                    raise TypeError(f"{slot.label}: {obj.name} is a {obj.kind}; takes {slot.kinds}")
            return objects
        obj = self[item]
        if obj.kind not in slot.kinds and not (obj.kind == "scanset" and "scan" in slot.kinds):
            raise TypeError(
                f"{slot.label}: {obj.name} is a {obj.kind}; takes {' or '.join(slot.kinds)}"
            )
        return obj

    def _param(self, p: Any, value: Any) -> Any:
        if value is None:
            return None
        if p.kind == "object":
            return self[value]
        if p.kind == "objects":
            return [self[v] for v in value]
        if p.kind == "roi":
            return value if isinstance(value, ROI) else self.roi(str(value))
        if p.kind == "rois":
            if isinstance(value, Mapping):
                return dict(value)
            names = [value] if isinstance(value, str) else list(value)
            return {n: self.roi(n) for n in names}
        return value

    @staticmethod
    def _call(spec: Any, given: list[Any], resolved: dict[str, Any]) -> list[DataObject]:
        first = given[0] if given else None
        slot = spec.inputs[0] if spec.inputs else None
        maps_over_set = (
            isinstance(first, DataObject)
            and first.kind == "scanset"
            and slot is not None
            and "scanset" not in slot.kinds
        )
        if not maps_over_set:
            return _as_list(spec.function(*given, **resolved))
        from bexa.workspace.ops.multiscan import stack_per_scan

        assert isinstance(first, DataObject)
        per_scan = []
        pairs = list(zip(first.data.scans, first.data.labels, strict=True))
        for scan, label in progress(pairs, total=len(pairs), desc="scans", enabled=False):
            stand_in = DataObject(first.name, scan, meta={"scan_label": label})
            per_scan.append(_as_list(spec.function(stand_in, *given[1:], **resolved)))
        return stack_per_scan(first.data, per_scan, first.name)

    def _keep(
        self, results: list[DataObject], given: list[Any], command: str, name: str | None
    ) -> list[DataObject]:
        first = (
            given[0][0] if given and isinstance(given[0], list) else (given[0] if given else None)
        )
        lineage = list(first.history) if isinstance(first, DataObject) else []
        base = first.name if isinstance(first, DataObject) else ""
        made = []
        for k, obj in enumerate(results):
            if name:
                if len(results) == 1:
                    obj.name = name
                elif base and obj.name.startswith(base):
                    obj.name = name + obj.name[len(base) :]
                else:
                    obj.name = f"{name}_{k + 1}"
            obj.history = [*lineage, command]
            made.append(self._insert(obj))
        return made

    # ------------------------------------------------------------------ export
    def export(self, name: str, path: str | Path, format: str | None = None) -> Path:
        """Write one object to a file (NeXus HDF5, CSV, NPY, TIFF, PNG, text, JSON)."""
        from bexa.workspace.files import export_object

        out = export_object(self[name], path, format)
        self.record(call_text("ws.export", name, str(out), format=format))
        return out

    def export_nexus(self, names: Sequence[str], path: str | Path) -> Path:
        """Write several objects to one NeXus HDF5 file, one NXdata group each."""
        from bexa.workspace.files import write_nexus

        out = write_nexus([self[n] for n in names], path)
        self.record(call_text("ws.export_nexus", list(names), str(out)))
        return out

    # -------------------------------------------------------------- conversions
    def convert(self, name: str, target: str, out: str | Path, **options: Any) -> list[Path]:
        """Write a scan (every scan of a set, into the folder ``out``) as ``target``
        (:data:`bexa.workspace.convert.SCAN_TARGETS`); ``roi`` may name an ROI."""
        from bexa.workspace.convert import convert_scan

        obj = self[name]
        if obj.kind not in ("scan", "scanset"):
            raise TypeError(f"{name} is a {obj.kind}; Export writes objects, Convert writes scans")
        roi = options.get("roi")
        given = {**options, "roi": self.roi(roi) if isinstance(roi, str) else roi}
        t0 = time.perf_counter()
        if obj.kind == "scan":
            files = convert_scan(obj.data, target, out, **given)
        else:
            files = []
            folder = Path(out)
            for scan, label in zip(obj.data.scans, obj.data.labels, strict=True):
                files += convert_scan(
                    scan, target, folder / _clean(label.replace(" #", "_S")), **given
                )
        self.record(
            call_text("ws.convert", name, target, str(out), **options), time.perf_counter() - t0
        )
        return files

    def convert_session(self, path: str | Path, out_dir: str | Path, **options: Any) -> list[Path]:
        """Every scan of a dataset or session written as a target (packages by default)."""
        from bexa.workspace.convert import convert_session

        t0 = time.perf_counter()
        files = convert_session(path, out_dir, **options)
        self.record(
            call_text("ws.convert_session", str(path), str(out_dir), **options),
            time.perf_counter() - t0,
        )
        return files

    def convert_dataset(
        self, path: str | Path, dataset: str, target: str, out: str | Path, **options: Any
    ) -> list[Path]:
        """One HDF5 dataset as npy, TIFF or CSV (:func:`bexa.workspace.convert.convert_dataset`)."""
        from bexa.workspace.convert import convert_dataset

        t0 = time.perf_counter()
        files = convert_dataset(path, dataset, target, out, **options)
        self.record(
            call_text("ws.convert_dataset", str(path), dataset, target, str(out), **options),
            time.perf_counter() - t0,
        )
        return files

    def convert_file(self, path: str | Path, target: str, out: str | Path) -> list[Path]:
        """A file bexa loads as objects (npy, TIFF, CSV, cube) written in another format."""
        from bexa.workspace.convert import convert_file

        files = convert_file(path, target, out)
        self.record(call_text("ws.convert_file", str(path), target, str(out)))
        return files

    # ------------------------------------------------------------------ the log
    def record(
        self,
        command: str,
        seconds: float = 0.0,
        status: str = "ok",
        message: str = "",
        results: Sequence[str] = (),
    ) -> LogEntry:
        """Add a line to the log (every logging method calls this)."""
        entry = LogEntry(command, float(seconds), status, message, list(results))
        self.log.append(entry)
        self._emit("log", entry=entry)
        return entry

    def script(self) -> str:
        """The commands that worked, as a Python script that rebuilds the workspace."""
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [
            f'"""Rebuilds the bexa workspace {self.name!r} (written {stamp})."""',
            "",
            "import bexa",
            "from bexa.workspace import Workspace",
            "",
            f"ws = Workspace({self.name!r})",
        ]
        lines += [e.command for e in self.log if e.status in ("ok", "note")]
        return "\n".join(lines) + "\n"

    def save_script(self, path: str | Path) -> Path:
        out = Path(path)
        out.write_text(self.script(), encoding="utf-8")
        return out

    # -------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> Path:
        """Keep the whole workspace in one HDF5 file: objects (scans by their recipes), ROIs,
        settings and the log."""
        import h5py

        from bexa.workspace.files import write_nexus

        out = Path(path)
        write_nexus(list(self.objects.values()), out)
        with h5py.File(out, "a") as f:
            f.attrs["bexa_workspace"] = WORKSPACE_MARKER
            f.attrs["name"] = self.name
            f.attrs["order"] = json.dumps(list(self.objects))
            f.attrs["rois"] = json.dumps({n: r.to_dict() for n, r in self.rois.items()})
            f.attrs["settings"] = json.dumps(jsonable(self.settings))
            f.attrs["log"] = json.dumps([asdict(e) for e in self.log])
        return out

    @classmethod
    def load(cls, path: str | Path, cache: Any = True) -> Workspace:
        """A workspace saved by :meth:`save`; scans are opened again from their files."""
        import h5py

        from bexa.workspace.storage import read_object

        ws = cls(cache=cache)
        with h5py.File(path, "r") as f:
            if f.attrs.get("bexa_workspace") != WORKSPACE_MARKER:
                raise ValueError(f"{path} is not a saved bexa workspace")
            ws.name = str(f.attrs.get("name", ws.name))
            ws.rois = {n: ROI.from_dict(d) for n, d in json.loads(f.attrs["rois"]).items()}
            ws.settings = json.loads(f.attrs.get("settings", "{}"))
            ws.log = [LogEntry(**e) for e in json.loads(f.attrs.get("log", "[]"))]
            order = json.loads(f.attrs.get("order", "[]"))
            entry = f["entry"]
            for key in order:
                group = entry[key.replace("/", "%2F")]
                ws.objects[key] = read_object(key, group, cache)
        return ws


# ------------------------------------------------------------------- helpers
def _as_list(results: Any) -> list[DataObject]:
    if results is None:
        return []
    if isinstance(results, DataObject):
        return [results]
    return list(results)


def _names_of(item: Any) -> Any:
    if isinstance(item, DataObject):
        return item.name
    if isinstance(item, (list, tuple)):
        return [_names_of(i) for i in item]
    return item


def _scan_name(scan: Any) -> str:
    return _clean(scan_label(scan).replace(" #", "_"))


def _scan_meta(scan: Any) -> dict[str, Any]:
    from bexa.workspace.ops.extract import scan_meta

    try:
        return scan_meta(scan)
    except Exception:  # a scan that cannot describe itself still opens
        return {}


def _recipe(recipe: Mapping[str, Any] | None) -> dict[str, Any]:
    """A scan's recipe without the empty entries and without the cache (the workspace's)."""
    if not recipe:
        raise ValueError("a scan without a recipe cannot be logged; open it with ws.open_scan")
    return {k: v for k, v in recipe.items() if v is not None and k != "cache"}
