"""``bexa.demo()``: a small synthetic dataset to try the API without beamtime data."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

__all__ = ["KINDS", "demo"]

KINDS = ("mosa", "rocking", "energy", "zstack", "alignment")


def demo(kind: str = "mosa", root: str | Path | None = None, **kwargs: Any) -> Any:
    """Write a synthetic ESRF dataset and open it.

    Parameters
    ----------
    kind
        ``"mosa"`` (a chi x mu scan, opened as a :class:`bexa.Scan`),
        ``"rocking"`` (a mu scan), ``"energy"`` (an energy series opened with an
        ``energy`` dim), ``"zstack"`` (a mosaicity scan and an energy series per
        height, opened as a :class:`bexa.Dataset`) or ``"alignment"`` (four
        quick 1-D scans, as a :class:`bexa.Dataset`).
    root
        Where to write; a new temporary folder by default.
    **kwargs
        Passed to the generator (``frame_shape``, ``noise``, ...).
    """
    from bexa.core.dataset import open_dataset
    from bexa.core.scan import open as open_scan
    from bexa.testing.synthetic import make_esrf_energy_series, make_esrf_scan, make_esrf_zstack

    folder = Path(root) if root is not None else Path(tempfile.mkdtemp(prefix="bexa_demo_"))
    if kind == "mosa":
        kwargs.setdefault("motors", (("chi", 6), ("mu", 16)))
        kwargs.setdefault("frame_shape", (64, 80))
        made = make_esrf_scan(folder, dataset="demo_mosa", n_files=1, **kwargs)
        return open_scan(made.dataset_dir, cache=False)
    if kind == "rocking":
        kwargs.setdefault("motors", (("mu", 25),))
        kwargs.setdefault("frame_shape", (64, 80))
        made = make_esrf_scan(folder, dataset="demo_rocking", n_files=1, **kwargs)
        return open_scan(made.dataset_dir, cache=False)
    if kind == "energy":
        kwargs.setdefault("motors", (("chi", 3), ("mu", 10)))
        kwargs.setdefault("frame_shape", (48, 64))
        series = make_esrf_energy_series(folder, dataset="demo_energy", n_files=1, **kwargs)
        return open_scan(series[0].dataset_dir, cache=False)
    if kind == "zstack":
        made_stack = make_esrf_zstack(folder, dataset="demo_zstack", **kwargs)
        return open_dataset(made_stack.dataset_dir, cache=False)
    if kind == "alignment":
        kwargs.setdefault("frame_shape", (48, 64))
        scans = {
            1: ("mu", 25, (-0.6, 0.6), "gaussian"),
            2: ("chi", 21, (-0.4, 0.4), "gaussian"),
            3: ("samz", 16, (-0.05, 0.05), "edge"),
            4: ("ccmth", 15, (6.66, 6.70), "gaussian"),
        }
        for number, (motor, points, span, curve) in scans.items():
            make_esrf_scan(
                folder,
                dataset="demo_align",
                scan=number,
                motors=((motor, points),),
                ranges={motor: span},
                curve=curve,
                n_files=1,
                seed=number,
                **kwargs,
            )
        return open_dataset(folder / "demo_align", cache=False)
    raise ValueError(f"unknown demo {kind!r}; choose from {KINDS}")
