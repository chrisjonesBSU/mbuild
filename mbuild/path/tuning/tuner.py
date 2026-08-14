"""Turn a target chemistry into hard_sphere_random_walk parameters."""

import logging
import re

import numpy as np

from mbuild.coarse_graining import coarse_grain
from mbuild.path.points import AngleDihedralSampler

from .plotting import (
    _pyplot,
    plot_angle_energy,
    plot_dihedral_energy,
    plot_pair_energy,
)
from .stages import (
    FreeEnergyAccumulator,
    barker_henderson,
    effective_sample_size,
    marginal_free_energy,
    natural_bond_length,
    orientation_averaged_pmf,
    pair_potential,
    phi_grid,
    sample_walks,
    theta_grid,
)

logger = logging.getLogger(__name__)


class Chemistry:
    """Fragment definition and forcefield shared by every tuning stage.

    Parameters
    ----------
    cgsmiles : str, optional
        A CGsmiles fragment string such as ``"{#A=[>]CC[<]}"``. Required
        unless every fragment is supplied through ``templates``.
    templates : dict[str, mbuild.Compound], optional
        Tagged compounds defining fragments. Strongly recommended, since
        per call RDKit embedding makes the measured energies seed dependent.
    forcefield : str or None, default None
        A foyer compatible forcefield name or XML path. None uses UFF.
    bead_name : str, optional
        Which fragment to tune. Required when more than one is defined.
    center : str, default "mass"
        Where a bead sits within its fragment, either "mass" for the mass
        weighted mean of the member positions or "geometry" for the
        arithmetic mean. Every tuned parameter is defined against this
        choice, so it has to match whatever the resulting path is compared
        to. Mass weighting keeps a bead near its heavy atoms, which makes
        the coarse grained bond length about three times less sensitive to
        temperature than a geometric centroid pulled around by hydrogens.
    """

    def __init__(
        self,
        cgsmiles=None,
        templates=None,
        forcefield=None,
        bead_name=None,
        center="mass",
    ):
        if cgsmiles is None and not templates:
            raise ValueError("Pass a cgsmiles fragment string, templates, or both.")
        if center not in ("geometry", "mass"):
            raise ValueError(
                f"Argument {center=} is invalid. Pass 'geometry' or 'mass'."
            )
        self.cgsmiles = cgsmiles
        self.templates = templates
        self.forcefield = forcefield
        self.center = center
        names = _fragment_names(cgsmiles, templates)
        if bead_name is None:
            if len(names) != 1:
                raise ValueError(
                    f"Found fragments {sorted(names)}. Pass bead_name to choose "
                    "which one to tune."
                )
            bead_name = names[0]
        elif bead_name not in names:
            raise ValueError(f"Fragment {bead_name} is not one of {sorted(names)}.")
        self.bead_name = bead_name

    def backmap(self, path, seed=1):
        """Backmap a path to an atomistic Compound."""
        return path.backmap(self.cgsmiles, templates=self.templates, seed=seed)

    def coarse_grain(self, compound):
        """Coarse grain a backmapped compound back to a Path."""
        cg_path, _ = coarse_grain(compound, beads=[self.bead_name], center=self.center)
        return cg_path

    def pair_fragment(self):
        """A capped copy of the fragment used for the pair sweep.

        Bonding descriptors are stripped from the fragment SMILES, so the
        junction atoms carry hydrogens instead of chain bonds. A mid chain
        bead's sterics differ from this, see the notes on ``radius``.
        """
        import mbuild as mb

        return mb.load(_fragment_smiles(self.cgsmiles, self.bead_name), smiles=True)


class TunerResult:
    """Tuned walk parameters plus the temperature free tables behind them.

    The energy table and the pair sweep carry no temperature, so any
    temperature can be reweighted from the same result with ``at``.

    Attributes
    ----------
    bond_length : float
        Natural coarse grained bond length in nm.
    radius : float
        Barker-Henderson excluded volume diameter in nm, at ``temperature``.
    angles : mbuild.path.points.AngleDihedralSampler
        Correlated bending angle and dihedral sampler at ``temperature``.
    temperature : float
        Temperature in Kelvin these parameters were built for.
    theta_grid, phi_grid : np.ndarray
        Bin centers of the energy table, in radians.
    energies : np.ndarray (n_theta, n_phi)
        Free energy of each cell in kJ/mol. Unsampled cells are np.inf.
    ess : list of float
        Effective sample size of each refinement round.
    n_samples : int
        Walks accepted in the final round.
    """

    def __init__(
        self,
        bond_length,
        radius,
        angles,
        temperature,
        theta_grid,
        phi_grid,
        energies,
        ess,
        n_samples,
        u_pair,
        separations,
    ):
        self.bond_length = bond_length
        self.radius = radius
        self.angles = angles
        self.temperature = temperature
        self.theta_grid = theta_grid
        self.phi_grid = phi_grid
        self.energies = energies
        self.ess = ess
        self.n_samples = n_samples
        self.u_pair = u_pair
        self.separations = separations

    def at(self, temperature):
        """Return a result for another temperature from the same tables.

        Rebuilds both the sampler and the excluded volume diameter. The walks
        are not rerun, so the reference ensemble stays the one refined at
        ``self.temperature``.
        """
        return TunerResult(
            bond_length=self.bond_length,
            radius=barker_henderson(self.u_pair, self.separations, temperature),
            angles=AngleDihedralSampler(
                self.theta_grid, self.phi_grid, self.energies, temperature=temperature
            ),
            temperature=float(temperature),
            theta_grid=self.theta_grid,
            phi_grid=self.phi_grid,
            energies=self.energies,
            ess=self.ess,
            n_samples=self.n_samples,
            u_pair=self.u_pair,
            separations=self.separations,
        )

    def pmf(self, temperature=None):
        """Orientation averaged pair potential of mean force, in kJ/mol."""
        if temperature is None:
            temperature = self.temperature
        return orientation_averaged_pmf(self.u_pair, temperature)

    def angle_energy(self, temperature=None):
        """Free energy against bending angle, dihedral integrated out.

        Returns
        -------
        theta_grid, energies : np.ndarray
            Bin centers in radians and free energy in kJ/mol, offset so the
            minimum is 0. Unsampled bins are np.inf.
        """
        temperature = self.temperature if temperature is None else temperature
        return self.theta_grid, marginal_free_energy(self.energies, temperature, axis=1)

    def dihedral_energy(self, temperature=None):
        """Free energy against dihedral, bending angle integrated out.

        Returns
        -------
        phi_grid, energies : np.ndarray
            Bin centers in radians and free energy in kJ/mol, offset so the
            minimum is 0. Unsampled bins are np.inf.
        """
        temperature = self.temperature if temperature is None else temperature
        return self.phi_grid, marginal_free_energy(self.energies, temperature, axis=0)

    def plot_pair(self, temperature=None, ax=None, **kwargs):
        """Plot the orientation averaged pair energy against separation."""
        temperature = self.temperature if temperature is None else temperature
        return plot_pair_energy(
            self.separations,
            self.u_pair,
            temperature,
            radius=barker_henderson(self.u_pair, self.separations, temperature),
            ax=ax,
            **kwargs,
        )

    def plot_angles(self, temperature=None, ax=None, **kwargs):
        """Plot the free energy against bending angle."""
        temperature = self.temperature if temperature is None else temperature
        return plot_angle_energy(
            self.theta_grid, self.energies, temperature, ax=ax, **kwargs
        )

    def plot_dihedrals(self, temperature=None, ax=None, **kwargs):
        """Plot the free energy against dihedral."""
        temperature = self.temperature if temperature is None else temperature
        return plot_dihedral_energy(
            self.phi_grid, self.energies, temperature, ax=ax, **kwargs
        )

    def plot(self, temperature=None, figsize=(13.0, 3.5)):
        """Plot the pair, bending angle and dihedral energies side by side.

        Parameters
        ----------
        temperature : float or sequence of float, optional
            Defaults to the result's own temperature. A sequence overlays one
            curve per temperature, all from the same tables.
        figsize : tuple, default (13.0, 3.5)

        Returns
        -------
        matplotlib.figure.Figure, np.ndarray of matplotlib.axes.Axes
        """
        temperatures = np.atleast_1d(
            self.temperature if temperature is None else temperature
        ).astype(float)
        figure, axes = _pyplot().subplots(1, 3, figsize=figsize)
        for value in temperatures:
            self.plot_pair(value, ax=axes[0])
            self.plot_angles(value, ax=axes[1])
            self.plot_dihedrals(value, ax=axes[2])
        figure.tight_layout()
        return figure, axes

    def as_walk_kwargs(self):
        """Keyword arguments for ``hard_sphere_random_walk``.

        The sampler emits correlated angle and dihedral pairs, so it is
        passed as ``rw_angles`` and ``rw_dihedrals`` is left unset.
        """
        return {
            "bond_length": self.bond_length,
            "radius": self.radius,
            "rw_angles": self.angles,
        }

    def __repr__(self):
        return (
            f"TunerResult(bond_length={self.bond_length:.4f}, "
            f"radius={self.radius:.4f}, temperature={self.temperature:g}, "
            f"n_samples={self.n_samples}, ess={[round(e, 1) for e in self.ess]})"
        )


class Tuner:
    """Measures hard_sphere_random_walk parameters for a target chemistry.

    Runs the pair sweep and the bond length relaxation first, since the walk
    that generates conformations for the angle table needs both up front.

    Parameters
    ----------
    cgsmiles : str, optional
        A CGsmiles fragment string such as ``"{#A=[>]CC[<]}"``.
    templates : dict[str, mbuild.Compound], optional
        Tagged compounds defining fragments. Strongly recommended.
    forcefield : str or None, default None
        A foyer compatible forcefield name or XML path. None uses UFF.
    bead_name : str, optional
        Which fragment to tune. Required when more than one is defined.
    center : str, default "mass"
        Where a bead sits within its fragment, "mass" or "geometry". See
        ``Chemistry``.
    temperature : float, default 300.0
        Temperature in Kelvin used to weight the energy tables.
    n_beads : int, default 4
        Sites per sampling walk. Energy is attributed to the interior
        coordinate set only. Longer walks add sites that are not measured but
        still contribute to the energy, which collapses the effective sample
        size, so raise this only once the energy can be restricted to the
        measured beads. See ``sample_walks``.
    n_walks : int, default 500
        Walks attempted per refinement round.
    n_rounds : int, default 3
        Refinement rounds. Each round feeds its sampler back in as the
        reference distribution for the next.
    n_theta_bins, n_phi_bins : int, default 12
        Energy table resolution.
    bond_length_beads : int, default 12
        Chain length used for the unrestrained bond length relaxation.
    n_orientations : int, default 300
        Relative orientations averaged over in the pair sweep.
    separations : array-like, optional
        Center to center separations of the pair sweep, in nm.
    seed : int, default 0
        Base seed for the walks and the pair sweep.
    """

    def __init__(
        self,
        cgsmiles=None,
        templates=None,
        forcefield=None,
        bead_name=None,
        center="mass",
        temperature=300.0,
        n_beads=4,
        n_walks=500,
        n_rounds=3,
        n_theta_bins=12,
        n_phi_bins=12,
        bond_length_beads=12,
        n_orientations=300,
        separations=None,
        seed=0,
    ):
        self.chemistry = Chemistry(
            cgsmiles=cgsmiles,
            templates=templates,
            forcefield=forcefield,
            bead_name=bead_name,
            center=center,
        )
        self.temperature = float(temperature)
        self.n_beads = int(n_beads)
        self.n_walks = int(n_walks)
        self.n_rounds = int(n_rounds)
        self.n_theta_bins = int(n_theta_bins)
        self.n_phi_bins = int(n_phi_bins)
        self.bond_length_beads = int(bond_length_beads)
        self.n_orientations = int(n_orientations)
        self.separations = separations
        self.seed = int(seed)

    def bond_length(self):
        """Run the unrestrained relaxation and return the bond length in nm."""
        value = natural_bond_length(
            self.chemistry, n_beads=self.bond_length_beads, seed=self.seed + 1
        )
        logger.info(f"bond_length {value:.4f} nm")
        return value

    def pair_sweep(self):
        """Run the pair sweep and return (u_pair, separations)."""
        u_pair, separations = pair_potential(
            self.chemistry,
            separations=self.separations,
            n_orientations=self.n_orientations,
            seed=self.seed,
        )
        logger.info(
            f"pair sweep {u_pair.shape[0]} orientations over "
            f"{u_pair.shape[1]} separations"
        )
        return u_pair, separations

    def run(self):
        """Run every stage and return a TunerResult."""
        bond_length = self.bond_length()
        u_pair, separations = self.pair_sweep()
        radius = barker_henderson(u_pair, separations, self.temperature)
        logger.info(f"radius {radius:.4f} nm at {self.temperature:g} K")

        thetas = theta_grid(bond_length, radius, self.n_theta_bins)
        phis = phi_grid(self.n_phi_bins)
        # Round 0 samples uniformly over the reachable angle range
        half_bin = 0.5 * (thetas[1] - thetas[0])
        rw_angles = (thetas[0] - half_bin, thetas[-1] + half_bin)

        accumulator = FreeEnergyAccumulator(thetas, phis, self.temperature)
        ess, table, n_samples = [], None, 0
        for round_index in range(self.n_rounds):
            pairs, energies = sample_walks(
                self.chemistry,
                bond_length=bond_length,
                radius=radius,
                n_walks=self.n_walks,
                n_beads=self.n_beads,
                rw_angles=rw_angles,
                seed=self.seed + round_index * self.n_walks,
            )
            if len(energies) == 0:
                raise RuntimeError(
                    f"Round {round_index} produced no usable walks. Check the "
                    "bond length and radius against the fragment size."
                )
            accumulator.add(pairs, energies)
            table = accumulator.table()
            ess.append(effective_sample_size(energies, self.temperature))
            n_samples += len(energies)
            rw_angles = AngleDihedralSampler(
                thetas, phis, table, temperature=self.temperature
            )
            logger.info(
                f"round {round_index} {len(energies)}/{self.n_walks} walks, "
                f"ESS {ess[-1]:.1f}, {int(np.isfinite(table).sum())}/{table.size} "
                "cells filled"
            )

        return TunerResult(
            bond_length=bond_length,
            radius=radius,
            angles=rw_angles,
            temperature=self.temperature,
            theta_grid=thetas,
            phi_grid=phis,
            energies=table,
            ess=ess,
            n_samples=n_samples,
            u_pair=u_pair,
            separations=separations,
        )


def tune(
    cgsmiles=None,
    templates=None,
    forcefield=None,
    bead_name=None,
    temperature=300.0,
    **kwargs,
):
    """Measure hard_sphere_random_walk parameters for a target chemistry.

    Convenience wrapper around ``Tuner``. See ``Tuner`` for the full
    parameter list.

    Parameters
    ----------
    temperature : float or sequence of float, default 300.0
        A sequence returns one TunerResult per temperature, all built from a
        single sweep. The first entry drives the refinement rounds.

    Returns
    -------
    TunerResult, or list of TunerResult when temperature is a sequence.

    Example
    -------
    >>> result = tune("{#A=[>]CC[<]}", n_walks=200)  # doctest: +SKIP
    >>> path = hard_sphere_random_walk(
    ...     **result.as_walk_kwargs(), termination=100
    ... )  # doctest: +SKIP
    """
    temperatures = np.atleast_1d(temperature).astype(float)
    tuner = Tuner(
        cgsmiles=cgsmiles,
        templates=templates,
        forcefield=forcefield,
        bead_name=bead_name,
        temperature=float(temperatures[0]),
        **kwargs,
    )
    result = tuner.run()
    if np.isscalar(temperature) or np.ndim(temperature) == 0:
        return result
    return [result] + [result.at(t) for t in temperatures[1:]]


def _fragment_names(cgsmiles, templates):
    """Fragment names defined by a cgsmiles string and a template dict."""
    names = list(templates) if templates else []
    if cgsmiles:
        names += [n for n in re.findall(r"#([^=]+)=", cgsmiles) if n not in names]
    if not names:
        raise ValueError(f"No fragment definitions found in {cgsmiles!r}.")
    return names


def _fragment_smiles(cgsmiles, bead_name):
    """SMILES of one cgsmiles fragment with its bonding descriptors removed."""
    if not cgsmiles:
        raise ValueError(
            "A cgsmiles fragment string is needed to build the pair fragment. "
            "Pass cgsmiles, or pass a radius instead of measuring one."
        )
    match = re.search(rf"#{re.escape(bead_name)}=([^,}}]+)", cgsmiles)
    if match is None:
        raise ValueError(f"Fragment {bead_name} not found in {cgsmiles}.")
    return re.sub(r"\[[<>$!][^\]]*\]", "", match.group(1))
