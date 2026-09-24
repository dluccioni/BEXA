"""Plugin registries for engines, accumulators and plot functions.

A registry is a named dictionary with a decorator. Built-in members register
themselves when their module is imported; :func:`load_builtins` imports those
modules so that a name lookup always sees the standard set.

Examples
--------
>>> from bexa.core.registry import register_engine, engines
>>> @register_engine("my_layout")
... class MyEngine(BaseSource): ...
>>> engines.get("my_layout")
<class 'MyEngine'>
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A named collection of plugins with a registration decorator."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def add(self, name: str, obj: T, replace: bool = False) -> T:
        """Register ``obj`` under ``name``; refuse silent overwrites unless ``replace``."""
        if name in self._items and not replace and self._items[name] is not obj:
            raise KeyError(f"{self.kind} {name!r} is already registered")
        self._items[name] = obj
        return obj

    def register(self, name: str | None = None, replace: bool = False) -> Callable[[T], T]:
        """Decorator form of :meth:`add`; the object's ``__name__`` is the default name."""

        def decorator(obj: T) -> T:
            key = name or getattr(obj, "name", None) or getattr(obj, "__name__", repr(obj))
            return self.add(str(key), obj, replace=replace)

        return decorator

    def get(self, name: str) -> T:
        """Look up a plugin by name, loading the built-ins first if needed."""
        if name not in self._items:
            load_builtins()
        try:
            return self._items[name]
        except KeyError:
            available = ", ".join(sorted(self._items)) or "none"
            raise KeyError(f"unknown {self.kind} {name!r}; available: {available}") from None

    def names(self) -> list[str]:
        load_builtins()
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._items))

    def __len__(self) -> int:
        return len(self._items)


engines: Registry = Registry("engine")
accumulators: Registry = Registry("accumulator")
plots: Registry = Registry("plot")


def register_engine(name: str | None = None, replace: bool = False):
    """Register a :class:`bexa.io.base.Source` implementation under a layout name."""
    return engines.register(name, replace=replace)


def register_accumulator(name: str | None = None, replace: bool = False):
    """Register an :class:`bexa.core.reductions.Accumulator` class."""
    return accumulators.register(name, replace=replace)


def register_plot(name: str | None = None, replace: bool = False):
    """Register a plotting function so pipelines can look it up by name."""
    return plots.register(name, replace=replace)


_BUILTIN_MODULES = ("bexa.io.engines", "bexa.core.reductions")
_loaded = False


def load_builtins() -> None:
    """Import the modules that register the built-in engines and accumulators."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    for module in _BUILTIN_MODULES:
        try:
            importlib.import_module(module)
        except ImportError:  # a module is still being built or an extra is missing
            continue
