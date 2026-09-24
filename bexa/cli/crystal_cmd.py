"""``bexa crystal ...``: Bragg's law, motor solutions, zone axes and omega search."""

from __future__ import annotations

from typing import Any

import typer

from bexa.crystal.lattice import Crystal, LatticeParameters

crystal_app = typer.Typer(
    help="Crystallography: d-spacings, motor solutions, zone axes.", no_args_is_help=True
)

LATTICE_HELP = (
    "diamond | silicon | cubic:3.56 | tetragonal:a,c | hexagonal:a,c | "
    "orthorhombic:a,b,c | a,b,c,alpha,beta,gamma"
)
HKL_HELP = "Three Miller indices, e.g. --hkl 3 1 1"


def parse_lattice(text: str) -> LatticeParameters:
    """Lattice parameters from the short CLI forms in ``LATTICE_HELP``."""
    text = text.strip()
    if text.lower() in ("diamond", "silicon"):
        return getattr(LatticeParameters, text.lower())()
    if ":" in text:
        kind, values = text.split(":", 1)
        numbers = [float(v) for v in values.split(",")]
        return getattr(LatticeParameters, kind.lower())(*numbers)
    numbers = [float(v) for v in text.split(",")]
    if len(numbers) == 1:
        return LatticeParameters.cubic(numbers[0])
    if len(numbers) == 3:
        return LatticeParameters.orthorhombic(*numbers)
    if len(numbers) == 6:
        return LatticeParameters(*numbers)
    raise typer.BadParameter(f"cannot parse lattice {text!r}; {LATTICE_HELP}")


def parse_ranges(items: list[str] | None) -> dict[str, tuple[float, float]]:
    """``["mu=-45:-10", "eta=35:100"]`` -> ``{"mu": (-45, -10), ...}``."""
    out: dict[str, tuple[float, float]] = {}
    for item in items or []:
        name, rng = item.split("=")
        lo, hi = rng.split(":")
        out[name.strip()] = (float(lo), float(hi))
    return out


def parse_angles(text: str | None) -> dict[str, float]:
    """``"mu=-38,chi=0,two_theta=53.2,eta=-67.5"`` -> dict."""
    out: dict[str, float] = {}
    for part in (text or "").split(","):
        if part.strip():
            name, value = part.split("=")
            out[name.strip()] = float(value)
    return out


Hkl = tuple[int, int, int]


def _hkl(values: Hkl) -> Hkl:
    return (int(values[0]), int(values[1]), int(values[2]))


@crystal_app.command("dspacing")
def crystal_dspacing(
    lattice: str = typer.Option(..., help=LATTICE_HELP),
    hkl: Hkl = typer.Option(..., help=HKL_HELP),
    energy: float | None = typer.Option(None, help="Energy in keV for the Bragg angle."),
    bravais: str | None = typer.Option(None),
) -> None:
    """d-spacing, |Q| and (with --energy) the Bragg angle of a reflection."""
    crystal = Crystal(parse_lattice(lattice), bravais=bravais)
    refl = _hkl(hkl)
    d = crystal.d_spacing(refl)
    q = 2 * 3.141592653589793 / d
    typer.echo(f"{refl}: d = {d:.5f} A, |Q| = {q:.5f} 1/A, {crystal.bravais.value}")
    if energy is not None:
        theta = crystal.bragg_angle_deg(refl, energy)
        typer.echo(f"  at {energy:g} keV: theta = {theta:.4f} deg, 2theta = {2 * theta:.4f} deg")
    typer.echo(f"  {len(crystal.equivalent_reflections(refl))} equivalent reflections")


@crystal_app.command("solve")
def crystal_solve(
    lattice: str = typer.Option(..., help=LATTICE_HELP),
    energy: float = typer.Option(..., help="Energy in keV."),
    hkl: Hkl = typer.Option(..., help=HKL_HELP),
    oop: Hkl = typer.Option(..., help="Out-of-plane direction, e.g. --oop 1 1 0"),
    zone: Hkl = typer.Option(..., help="Zone axis, e.g. --zone 0 0 1"),
    bravais: str | None = typer.Option(None),
    convention: str = typer.Option("pal_fourc", help="pal_fourc or sacla"),
    eta_offset: float = typer.Option(0.0, help="eta zero offset in degrees."),
    range_: list[str] | None = typer.Option(
        None, "--range", help="Motor range, e.g. mu=-45:-10 (repeatable)."
    ),
    fixed: list[str] | None = typer.Option(None, help="Fixed motor, e.g. omega=35.5 (repeatable)."),
    no_equivalents: bool = typer.Option(False, help="Solve only the given hkl."),
    max_display: int = typer.Option(20),
) -> None:
    """Motor angles that bring a reflection family onto the detector."""
    from bexa.crystal.diffractometer import create_diffractometer, format_solutions

    diff = create_diffractometer(
        parse_lattice(lattice), energy, _hkl(oop), _hkl(zone), bravais=bravais,
        convention=convention, eta_zero_offset=eta_offset,
    )  # fmt: skip
    for name, (lo, hi) in parse_ranges(range_).items():
        diff.set_motor_range(name, lo, hi)
    for name, value in parse_angles(",".join(fixed or [])).items():
        diff.set_motor_fixed(name, value)
    typer.echo(diff.describe_motors())
    typer.echo("searching...")
    solutions = diff.solve_for_peak_family(_hkl(hkl), use_equivalents=not no_equivalents)
    typer.echo(format_solutions(solutions, diff, max_display=max_display))


@crystal_app.command("zone-axis")
def crystal_zone_axis(
    lattice: str = typer.Option(..., help=LATTICE_HELP),
    energy: float = typer.Option(...),
    hkl: Hkl = typer.Option(..., help="Observed reflection."),
    oop: Hkl = typer.Option(..., help="Out-of-plane direction."),
    angles: str = typer.Option(..., help="mu=-38,chi=0,phi=0,omega=0,two_theta=53.2,eta=-67.5"),
    tolerances: str | None = typer.Option(
        None, help="mu=10,chi=10,phi=0,omega=10,two_theta=10,eta=10"
    ),
    bravais: str | None = typer.Option(None),
    convention: str = typer.Option("pal_fourc"),
    eta_offset: float = typer.Option(0.0),
    search_range: int = typer.Option(10),
    top: int = typer.Option(5),
) -> None:
    """Zone axes consistent with one indexed reflection and its motor angles."""
    from bexa.crystal.zone_axis import solve_zone_axis

    results = solve_zone_axis(
        parse_lattice(lattice), energy, _hkl(oop), _hkl(hkl), parse_angles(angles),
        bravais=bravais, convention=convention, eta_zero_offset=eta_offset,
        search_range=search_range, num_results=top,
        angle_tolerances=parse_angles(tolerances) if tolerances else None,
    )  # fmt: skip
    if not results:
        typer.echo("No valid zone axes found.")
        return
    typer.echo(f"{'rank':<6}{'zone axis':<16}{'d (A)':<10}{'residual':<12}{'matched hkl'}")
    for i, r in enumerate(results, 1):
        typer.echo(
            f"{i:<6}{r.zone_axis!s:<16}{r.d_spacing:<10.4f}{r.residual:<12.6f}{r.matched_hkl}"
        )
        if r.fit is not None:
            fitted = ", ".join(f"{k}={v:.2f}" for k, v in r.fit.angles.items())
            typer.echo(f"      angles within tolerance: {fitted}")


@crystal_app.command("omega-search")
def crystal_omega_search(
    lattice: str = typer.Option(..., help=LATTICE_HELP),
    energy: float = typer.Option(...),
    family: Hkl = typer.Option(..., help="Reflection family, e.g. --family 3 1 1"),
    oop: Hkl = typer.Option(..., help="Out-of-plane direction."),
    zone: Hkl = typer.Option(..., help="Zone axis."),
    range_: list[str] | None = typer.Option(
        None, "--range", help="mu=-45:-10, eta=35:100, two_theta=20:70"
    ),
    bravais: str | None = typer.Option(None),
    convention: str = typer.Option("sacla"),
    eta_offset: float = typer.Option(0.0),
    omega_scan: str = typer.Option("-180:180"),
    step: float = typer.Option(0.01),
    top: int = typer.Option(5),
    verify: bool = typer.Option(False, help="Re-check each candidate with the full solver (slow)."),
) -> None:
    """Omega settings where two peaks of a family are closest in mu (phi = chi = 0)."""
    from bexa.crystal.diffractometer import create_diffractometer
    from bexa.crystal.omega_search import find_omega_offsets, format_candidates

    diff = create_diffractometer(
        parse_lattice(lattice), energy, _hkl(oop), _hkl(zone), bravais=bravais,
        convention=convention, eta_zero_offset=eta_offset,
    )  # fmt: skip
    ranges = parse_ranges(range_)
    for name in ("phi", "chi"):
        diff.set_motor_range(name, 0, 0)
    for name, (lo, hi) in ranges.items():
        if name in diff.motors:
            diff.set_motor_range(name, lo, hi)
    lo, hi = (float(v) for v in omega_scan.split(":"))
    candidates = find_omega_offsets(diff, _hkl(family), ranges, (lo, hi), step, top, verify=verify)
    typer.echo(format_candidates(candidates))


@crystal_app.command("reflections")
def crystal_reflections(
    lattice: str = typer.Option(..., help=LATTICE_HELP),
    energy: float = typer.Option(...),
    max_hkl: int = typer.Option(3),
    centering: str = typer.Option(
        "P", help="Lattice centering for the extinction rules (P, I, F, ...)."
    ),
    bravais: str | None = typer.Option(None),
) -> None:
    """Allowed reflections reachable at an energy, with d-spacing and 2theta."""
    crystal = Crystal(parse_lattice(lattice), bravais=bravais, centering=centering)
    rows: list[tuple[float, Any]] = []
    for refl in crystal.reflections(max_hkl):
        d = crystal.d_spacing(refl)
        try:
            theta = crystal.bragg_angle_deg(refl, energy)
        except ValueError:
            continue
        rows.append((d, (refl, theta)))
    typer.echo(f"{len(rows)} reflections up to |hkl| <= {max_hkl} at {energy:g} keV")
    for d, (refl, theta) in sorted(rows, key=lambda r: -r[0]):
        typer.echo(f"  {refl!s:>14}  d = {d:8.4f} A  2theta = {2 * theta:8.3f} deg")
