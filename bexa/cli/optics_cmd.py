"""``bexa optics ...``: conversions, attenuation and Bragg magnifier estimates."""

from __future__ import annotations

import numpy as np
import typer

optics_app = typer.Typer(
    help="X-ray optics: conversions, transmission, magnifiers.", no_args_is_help=True
)


def parse_energies(text: str) -> np.ndarray:
    """``"17"`` -> [17], ``"10:30:5"`` -> 10, 15, ..., 30 (keV)."""
    if ":" in text:
        parts = [float(v) for v in text.split(":")]
        lo, hi = parts[0], parts[1]
        step = parts[2] if len(parts) > 2 else 1.0
        return np.arange(lo, hi + step / 2, step)
    return np.array([float(v) for v in text.split(",")])


@optics_app.command("convert")
def optics_convert(
    energy: float | None = typer.Option(None, help="Energy in keV."),
    wavelength: float | None = typer.Option(None, help="Wavelength in Angstrom."),
) -> None:
    """Energy, wavelength, wavevector and monochromator angles."""
    from bexa.optics.conversions import summary

    for key, value in summary(energy, wavelength).items():
        typer.echo(f"{key:>18}: {value:.6g}")


@optics_app.command("transmission")
def optics_transmission(
    material: str = typer.Option(
        ..., help="Element, formula or a known name (Si, WTe2, LuAG, kapton)."
    ),
    thickness: str = typer.Option(..., help="e.g. 500um, 0.3mm, 2cm"),
    energy: str = typer.Option(..., help="keV: a value, a list 10,17, or a range 10:30:1"),
    density: float | None = typer.Option(None, help="g/cm3, overrides the table."),
) -> None:
    """Transmission and attenuation length of a material."""
    from bexa.optics.attenuation import attenuation_length_um, resolve_material, transmission

    formula, rho = resolve_material(material, density)
    energies = parse_energies(energy)
    typer.echo(f"{material} ({formula}, {rho:g} g/cm3), {thickness}")
    for e in energies:
        t = float(transmission(formula, thickness, e, rho))
        length = float(attenuation_length_um(formula, e, rho))
        typer.echo(f"  {e:8.3f} keV: transmission {t:.4%}, 1/e length {length:.2f} um")


@optics_app.command("photons")
def optics_photons(
    pulse_energy: float = typer.Option(..., help="Pulse energy in J (e.g. 5e-6)."),
    energy: float = typer.Option(..., help="Photon energy in keV."),
) -> None:
    """Photons per pulse."""
    from bexa.optics.conversions import photons_from_pulse_energy

    n = photons_from_pulse_energy(pulse_energy, energy * 1e3)
    typer.echo(f"{pulse_energy:g} J at {energy:g} keV = {n:.3e} photons")


@optics_app.command("magnifier")
def optics_magnifier(
    mag: float = typer.Option(..., help="Total magnification of one pair."),
    energy: float = typer.Option(9251.0, help="Photon energy in eV."),
    de: float = typer.Option(0.5, help="Energy spread FWHM in eV."),
    divergence: float = typer.Option(1.5, help="Beam divergence sigma in urad."),
    splits: int = typer.Option(25, help="Number of split ratios tried."),
    n_energies: int = typer.Option(11),
    points: int = typer.Option(2001, help="Points of the rocking-angle grid."),
) -> None:
    """Best split of a Bragg magnifier pair and its efficiency."""
    from bexa.optics.bragg_magnifier import miscuts_from_split, optimal_split

    dth = np.linspace(-100, 100, points)
    split, eff, dphi = optimal_split(
        mag, n_splits=splits, E0=energy, dE=de, n_E=n_energies, sigma_theta_urad=divergence, dth=dth
    )
    a1, a2 = miscuts_from_split(mag, split, energy)
    m1, m2 = mag**split, mag ** (1 - split)
    typer.echo(f"magnification {mag:g}: best split {split:.3f} (M1 = {m1:.2f}, M2 = {m2:.2f})")
    typer.echo(f"  efficiency {eff:.4f} %, pair offset {dphi:.2f} arcsec")
    typer.echo(f"  miscuts {a1:.3f} and {a2:.3f} deg")
