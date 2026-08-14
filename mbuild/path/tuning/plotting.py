"""Plots of the energies behind a tuned parameter set."""

import numpy as np

from mbuild.path.points import GAS_CONSTANT

from .stages import marginal_free_energy, orientation_averaged_pmf


def plot_pair_energy(
    separations, u_pair, temperature, radius=None, ax=None, ylim=None, **kwargs
):
    """Plot the orientation averaged pair energy against separation.

    The curve is the Boltzmann average over relative orientations, so it is a
    potential of mean force rather than a mean energy.

    Parameters
    ----------
    separations : array-like (M,), required
        Center to center separations in nm.
    u_pair : np.ndarray (n_orientations, M), required
        Pair energies referenced to large separation, in kJ/mol.
    temperature : float, required
        Temperature in Kelvin.
    radius : float, optional
        Excluded volume diameter to mark with a vertical line.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when omitted.
    ylim : tuple, optional
        Defaults to a window of 10 RT above the well depth, which keeps the
        repulsive wall from flattening the rest of the curve.
    **kwargs
        Passed to ``ax.plot``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _axes(ax)
    separations = np.asarray(separations, dtype=float)
    u_eff = orientation_averaged_pmf(u_pair, temperature)
    ax.plot(separations, u_eff, **{"label": f"{temperature:g} K", **kwargs})
    ax.axhline(0.0, color="0.7", lw=0.8, zorder=0)
    if radius is not None:
        ax.axvline(
            radius,
            color="0.4",
            ls="--",
            lw=1.0,
            label=f"radius {radius:.3f} nm",
        )
    if ylim is None:
        rt = GAS_CONSTANT * temperature
        low = min(float(np.min(u_eff)), 0.0)
        ylim = (low - 0.2 * rt - 1.0, low + 10.0 * rt)
    ax.set_ylim(ylim)
    ax.set_xlabel("separation (nm)")
    ax.set_ylabel("pair energy (kJ/mol)")
    ax.legend()
    return ax


def plot_angle_energy(theta_grid, table, temperature, ax=None, degrees=True, **kwargs):
    """Plot the free energy against bending angle, dihedral integrated out.

    Parameters
    ----------
    theta_grid : array-like (N,), required
        Bending angle bin centers in radians.
    table : np.ndarray (N, M), required
        Free energy of each cell in kJ/mol.
    temperature : float, required
        Temperature in Kelvin.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when omitted.
    degrees : bool, default True
        Label the axis in degrees.
    **kwargs
        Passed to ``ax.plot``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    energies = marginal_free_energy(table, temperature, axis=1)
    return _plot_marginal(
        theta_grid,
        energies,
        ax=ax,
        degrees=degrees,
        xlabel="bending angle",
        temperature=temperature,
        **kwargs,
    )


def plot_dihedral_energy(phi_grid, table, temperature, ax=None, degrees=True, **kwargs):
    """Plot the free energy against dihedral, bending angle integrated out.

    Parameters
    ----------
    phi_grid : array-like (M,), required
        Dihedral bin centers in radians.
    table : np.ndarray (N, M), required
        Free energy of each cell in kJ/mol.
    temperature : float, required
        Temperature in Kelvin.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when omitted.
    degrees : bool, default True
        Label the axis in degrees.
    **kwargs
        Passed to ``ax.plot``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    energies = marginal_free_energy(table, temperature, axis=0)
    return _plot_marginal(
        phi_grid,
        energies,
        ax=ax,
        degrees=degrees,
        xlabel="dihedral",
        temperature=temperature,
        **kwargs,
    )


def _plot_marginal(grid, energies, ax, degrees, xlabel, temperature, **kwargs):
    """Draw a 1D free energy curve, breaking the line at unsampled bins."""
    ax = _axes(ax)
    x = np.degrees(grid) if degrees else np.asarray(grid, dtype=float)
    # Unsampled bins are infinite, which matplotlib draws as a gap when nan
    y = np.where(np.isfinite(energies), energies, np.nan)
    ax.plot(x, y, **{"marker": "o", "ms": 3, "label": f"{temperature:g} K", **kwargs})
    ax.set_xlabel(f"{xlabel} ({'degrees' if degrees else 'radians'})")
    ax.set_ylabel("free energy (kJ/mol)")
    ax.legend()
    return ax


def _axes(ax, figsize=(5.0, 3.5)):
    """Return the given axes, or a new one on a new figure."""
    if ax is not None:
        return ax
    return _pyplot().subplots(figsize=figsize)[1]


def _pyplot():
    """Import pyplot, with a pointer to the install when it is missing."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "Plotting tuner results needs matplotlib, which is not an mbuild "
            "dependency. Install it with `conda install matplotlib`."
        ) from error
    return plt
