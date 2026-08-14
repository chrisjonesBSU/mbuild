"""The parameter producing stages of the random walk tuner."""

import logging

import numpy as np

from mbuild.exceptions import PathConvergenceError
from mbuild.path.build import hard_sphere_random_walk, zigzag
from mbuild.path.points import GAS_CONSTANT, boltzmann_weights
from mbuild.simulation import OpenMMSimulation

from .energy import PairEnergy, restrained_energy
from .geometry import bond_lengths, interior_angle_dihedral_pair, internals

logger = logging.getLogger(__name__)

# Energies above this are minimizer blowups rather than physical states
ENERGY_CUTOFF = 1e5
# A perfectly straight chain gives a NaN bounding box in get_boundingbox
MAX_THETA_DEG = 175.0


def relaxed_bond_length(
    chemistry,
    n_beads=12,
    spacing=0.25,
    seed_angle_deg=130.0,
    seed=1,
    n_steps=5000,
    tolerance=0.5,
):
    """Coarse grained bond length after one unrestrained relaxation.

    Backmaps a planar all trans chain, relaxes it with no restraints, coarse
    grains the result and averages the distance between bonded beads. The
    relaxation is unrestrained so the atomistic geometry sets its own bead
    spacing.

    Parameters
    ----------
    chemistry : Chemistry, required
        Fragment definition and forcefield.
    n_beads : int, default 12
        Chain length.
    spacing : float, default 0.25
        Bead spacing of the starting geometry, in nm.
    seed_angle_deg : float, default 130.0
        Bending angle of the starting geometry, in degrees. Extended seeds
        avoid the strained local minima that clashed seeds relax into.
    seed : int, default 1
        Seed for backmapping and the simulation.
    n_steps : int, default 5000
        Minimization steps.
    tolerance : float, default 0.5
        Minimization tolerance in kJ/mol/nm.

    Returns
    -------
    bond_length : float
        Mean bead to bead distance in nm.
    angle : float
        Mean bending angle in degrees, used to detect a stuck minimization.
    """
    path = zigzag(
        N=n_beads,
        spacing=spacing,
        angle_deg=180.0 - seed_angle_deg,
        sites_per_segment=1,
        bead_name=chemistry.bead_name,
    )
    compound = chemistry.backmap(path, seed=seed)
    sim = OpenMMSimulation(
        compound, forcefield=chemistry.forcefield, platform="CPU", seed=seed
    )
    sim.minimize(n_steps=n_steps, tolerance=tolerance)
    coordinates = chemistry.coarse_grain(sim.compound).coordinates
    _, angles, _ = internals(coordinates)
    return float(np.mean(bond_lengths(coordinates))), float(np.degrees(np.mean(angles)))


def natural_bond_length(
    chemistry,
    n_beads=12,
    spacing=0.25,
    seed_angles_deg=(130.0, 115.0),
    rtol=0.02,
    **kwargs,
):
    """Mean coarse grained bond length the chemistry settles at.

    Relaxes from several starting geometries and averages the results. A
    minimization that sticks in a strained local minimum returns a bond
    length that is confidently wrong and nothing downstream detects it, so
    the seeds are compared against each other and disagreement is warned on.

    Parameters
    ----------
    chemistry : Chemistry, required
        Fragment definition and forcefield.
    n_beads : int, default 12
        Chain length.
    spacing : float, default 0.25
        Bead spacing of the starting geometries, in nm.
    seed_angles_deg : sequence of float, default (130.0, 115.0)
        Bending angle of each starting geometry, in degrees.
    rtol : float, default 0.02
        Relative spread between seeds above which a warning is logged.
    **kwargs
        Passed to ``relaxed_bond_length``.

    Returns
    -------
    float
        Mean bead to bead distance in nm.
    """
    results = [
        relaxed_bond_length(
            chemistry,
            n_beads=n_beads,
            spacing=spacing,
            seed_angle_deg=angle,
            **kwargs,
        )
        for angle in seed_angles_deg
    ]
    lengths = np.array([r[0] for r in results])
    mean = float(lengths.mean())
    spread = float(np.ptp(lengths)) / mean
    if spread > rtol:
        logger.warning(
            f"Relaxed bond length varies by {spread:.1%} across starting "
            f"geometries, {np.round(lengths, 4).tolist()} nm from seed angles "
            f"{list(seed_angles_deg)} degrees. At least one relaxation is stuck "
            "in a strained local minimum. Compare against an unrestrained MD "
            "run before trusting the result."
        )
    for (length, angle), seed_angle in zip(results, seed_angles_deg):
        if abs(angle - seed_angle) < 1.0:
            logger.warning(
                f"The relaxed bending angle {angle:.1f} degrees barely moved "
                f"from its starting value of {seed_angle:.1f}, so the "
                f"minimization from that seed may not have converged."
            )
    return mean


def orientation_averaged_pmf(u_pair, temperature):
    """Orientational Boltzmann average of a pair energy, in kJ/mol.

    Averages exp(-u/kT) rather than u. An arithmetic average over
    orientations is not a free energy and overestimates the repulsion.

    Parameters
    ----------
    u_pair : np.ndarray (n_orientations, n_separations), required
        Pair energies referenced to large separation, in kJ/mol.
    temperature : float, required
        Temperature in Kelvin.
    """
    rt = GAS_CONSTANT * temperature
    shift = u_pair.min(axis=0)
    return shift - rt * np.log(np.mean(np.exp(-(u_pair - shift) / rt), axis=0))


def barker_henderson(u_pair, separations, temperature):
    """Effective hard sphere diameter of an orientation averaged pair potential.

    Returns the integral of 1 - exp(-u_eff / kT) over the scanned range,
    offset by the first separation so the unscanned core counts as fully
    repulsive.
    """
    rt = GAS_CONSTANT * temperature
    u_eff = orientation_averaged_pmf(u_pair, temperature)
    integrand = 1.0 - np.exp(-np.clip(u_eff, 0.0, None) / rt)
    return float(np.trapezoid(integrand, separations) + separations[0])


def pair_potential(chemistry, separations=None, n_orientations=300, seed=0):
    """Orientation averaged pair energies of two fragment copies.

    Runs the temperature free part of the excluded volume measurement, so one
    sweep serves every temperature.

    Returns
    -------
    u_pair : np.ndarray (n_orientations, n_separations)
        Energies in kJ/mol, referenced to the largest separation.
    separations : np.ndarray (n_separations,)
        Center to center separations in nm.
    """
    if separations is None:
        separations = np.arange(0.20, 0.86, 0.02)
    separations = np.asarray(separations, dtype=float)
    pair = PairEnergy(chemistry.pair_fragment(), forcefield=chemistry.forcefield)
    energies = pair.scan(separations, n_orientations=n_orientations, seed=seed)
    # The largest separation carries the constant intramolecular part
    return energies - energies[:, -1][:, None], separations


def theta_grid(bond_length, radius, n_bins, max_theta_deg=MAX_THETA_DEG):
    """Bending angle bin centers spanning the range the walk can reach.

    The lower bound is the angle at which a new site would overlap the site
    two bonds back, since that site is not excluded from the overlap check.
    The upper bound avoids colinear beads.
    """
    low = 2.0 * np.arcsin(min(radius / (2.0 * bond_length), 1.0))
    high = np.radians(max_theta_deg)
    if low >= high:
        raise ValueError(
            f"With {bond_length=} and {radius=} no bond angle below "
            f"{max_theta_deg} degrees avoids overlapping the site two bonds "
            "back. Reduce radius or increase bond_length."
        )
    edges = np.linspace(low, high, n_bins + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def phi_grid(n_bins):
    """Dihedral bin centers spanning [-pi, pi)."""
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def sample_walks(
    chemistry,
    bond_length,
    radius,
    n_walks=500,
    n_beads=4,
    rw_angles=None,
    seed=0,
    restraint_k=None,
):
    """Generate walks and measure the restrained minimum energy of each.

    Conformations come from ``hard_sphere_random_walk`` itself, so the
    reference distribution is exactly the sampler passed in and only
    geometries the walk can reach are sampled.

    Each energy is attributed to the walk's interior coordinate set alone.

    Parameters
    ----------
    chemistry : Chemistry, required
        Fragment definition and forcefield.
    bond_length, radius : float, required
        Walk parameters, from the bond length and pair sweep stages.
    n_walks : int, default 500
        Number of walks attempted.
    n_beads : int, default 4
        Sites per walk. Longer walks put chain rather than terminal caps on
        the measured beads, which is worth about 6 degrees of bending angle
        for polyethylene under UFF. Realizing that needs an energy restricted
        to the measured beads, because a whole chain energy also carries the
        sites that are not measured, and the walk places those anywhere. Each
        added site therefore widens the energy spread and collapses the
        effective sample size.
    rw_angles : sampler, optional
        Passed straight to the walk as its angle sampler.
    seed : int, default 0
        Base seed. Walk i uses seed + i.
    restraint_k : float, optional
        Restraint force constant in kJ/mol/nm^2.

    Returns
    -------
    pairs : list of np.ndarray (K, 2)
        The (theta, phi) pairs of each accepted walk.
    energies : np.ndarray (M,)
        Force field energy of each accepted walk, in kJ/mol.
    """
    kwargs = {} if restraint_k is None else {"k": restraint_k}
    pairs, energies = [], []
    for i in range(n_walks):
        try:
            path = hard_sphere_random_walk(
                bead_name=chemistry.bead_name,
                bond_length=bond_length,
                radius=radius,
                rw_angles=rw_angles,
                termination=n_beads,
                seed=seed + i,
            )
        except PathConvergenceError:
            continue
        if len(path.coordinates) < n_beads:
            continue
        _, angles, dihedrals = internals(path.coordinates)
        compound = chemistry.backmap(path)
        energy = restrained_energy(
            compound,
            path.coordinates,
            forcefield=chemistry.forcefield,
            **kwargs,
        )
        if not np.isfinite(energy) or energy > ENERGY_CUTOFF:
            continue
        pairs.append(interior_angle_dihedral_pair(angles, dihedrals))
        energies.append(energy)
    return pairs, np.asarray(energies)


def sample_thermal(
    chemistry,
    bond_length,
    radius,
    n_beads=12,
    n_starts=1,
    equilibrate_ps=100.0,
    production_ps=200.0,
    n_frames=40,
    temperature=300.0,
    friction=0.5,
    dt=0.0005,
    seed=0,
):
    """Coarse grained internals from unconstrained MD seeded by random walks.

    Each start is an independent walk, backmapped and run without restraints.
    Frames are coarse grained and their internal coordinates returned, so the
    distribution is histogrammed directly with no restrained minimum, no
    reweighting and no energy attribution.

    Dynamics are stepped continuously rather than through repeated ``nvt``
    calls, which redraw velocities and need calls of a few times
    ``1 / friction`` to hold temperature.

    Parameters
    ----------
    chemistry : Chemistry, required
        Fragment definition and forcefield.
    bond_length, radius : float, required
        Walk parameters used to build the starting conformations. The
        relaxation is unrestrained, so these seed the run rather than
        constrain the result.
    n_beads : int, default 12
        Sites per chain. Longer chains give more internal coordinates per
        frame and dilute the influence of the ends.
    n_starts : int, default 1
        Independent walks, each its own trajectory. A single trajectory
        cannot detect its own trapping, so more than one turns replica
        spread into a convergence diagnostic.
    equilibrate_ps : float, default 100.0
        Discarded before frames are kept, so each replica forgets its
        starting conformation. Raise it when replicas disagree.
    production_ps : float, default 200.0
        Sampled per replica.
    n_frames : int, default 40
        Frames kept per replica.
    temperature : float, default 300.0
        Temperature in Kelvin.
    friction : float, default 0.5
        Langevin friction in 1/ps. Below the OpenMM default, which sits on
        the high friction side of the Kramers turnover, so barriers are
        crossed about four times faster here.
    dt : float, default 0.0005
        Timestep in picoseconds.
    seed : int, default 0
        Base seed. Replica i uses seed + i.

    Returns
    -------
    replicas : list of list of np.ndarray
        Coarse grained coordinates, grouped by replica so spread across them
        can be measured.
    """
    import openmm.unit as u
    from openmm.openmm import LangevinIntegrator

    replicas = []
    for index in range(n_starts):
        walk_seed = seed + index
        try:
            path = hard_sphere_random_walk(
                bead_name=chemistry.bead_name,
                bond_length=bond_length,
                radius=radius,
                termination=n_beads,
                seed=walk_seed,
            )
        except PathConvergenceError:
            logger.warning(f"Walk {walk_seed} did not converge, skipping this start.")
            continue
        compound = chemistry.backmap(path)
        sim = OpenMMSimulation(
            compound, forcefield=chemistry.forcefield, platform="CPU", seed=walk_seed
        )
        sim.minimize(n_steps=4000, tolerance=1.0)
        integrator = LangevinIntegrator(
            temperature * u.kelvin, friction / u.picosecond, dt * u.picoseconds
        )
        integrator.setRandomNumberSeed(walk_seed)
        sim._create_simulation(integrator)
        sim.simulation.context.setVelocitiesToTemperature(temperature, walk_seed)
        sim.simulation.step(int(equilibrate_ps / dt))
        frames, per_frame = [], max(1, int(production_ps / dt / n_frames))
        for _ in range(n_frames):
            sim.simulation.step(per_frame)
            sim._update_compound_positions()
            frames.append(chemistry.coarse_grain(sim.compound).coordinates.copy())
        replicas.append(frames)
        logger.info(
            f"replica {index} done, {len(frames)} frames over {production_ps:g} ps"
        )
    if not replicas:
        raise RuntimeError("No starting conformation produced a usable trajectory.")
    return replicas


def thermal_internals(replicas):
    """Bond lengths, bending angles and dihedrals pooled over replicas.

    Every internal coordinate of every frame counts, so a longer chain
    yields more samples per frame.

    Returns
    -------
    bonds, angles, dihedrals : np.ndarray
        Angles and dihedrals in radians, dihedrals signed over [-pi, pi].
    """
    frames = [frame for replica in replicas for frame in replica]
    measured = [internals(frame) for frame in frames]
    return tuple(np.concatenate([m[i] for m in measured]) for i in range(3))


def replica_spread(replicas):
    """Peak to peak variation of per replica means, as a convergence check.

    A single trajectory cannot detect its own trapping. Spread comparable to
    the statistical scatter of one replica means the replicas agree.

    Returns
    -------
    dict
        Spread in ``bond`` (nm), ``angle`` (degrees) and ``trans``, the
        fraction of dihedrals beyond 160 degrees. Zeros for one replica,
        which carries no information about its own convergence.
    """
    if len(replicas) < 2:
        return {"bond": 0.0, "angle": 0.0, "trans": 0.0}
    rows = []
    for replica in replicas:
        bonds, angles, dihedrals = thermal_internals([replica])
        rows.append(
            [
                bonds.mean(),
                np.degrees(angles).mean(),
                float(np.mean(np.abs(np.degrees(dihedrals)) > 160.0)),
            ]
        )
    spread = np.ptp(np.array(rows), axis=0)
    return {"bond": spread[0], "angle": spread[1], "trans": spread[2]}


def energy_table_from_samples(angles, dihedrals, thetas, phis, temperature):
    """Boltzmann invert a sampled (theta, phi) histogram into an energy table.

    The histogram of a thermal ensemble is a probability density, and
    ``random_coordinate`` places a site at polar angle theta, so the table
    the sampler consumes has to be the constrained potential rather than the
    density. Dividing out the angular measure before inverting does that,
    matching the convention ``angle_table_from_sampler`` assumes in the other
    direction. Skipping it biases the walk toward extended chains.

    Each bending angle bin is divided by its exact measure,
    ``cos(low) - cos(high)``, rather than by ``sin`` at the bin center. The
    two agree to well under a percent for the bin widths used here, so this
    is exactness rather than a correction.

    Parameters
    ----------
    angles, dihedrals : np.ndarray, required
        Sampled values in radians, of any length and not necessarily paired.
        Pairs are formed by flanking, as in ``angle_dihedral_pairs``.
    thetas, phis : np.ndarray, required
        Bin centers in radians.
    temperature : float, required
        Temperature in Kelvin.

    Returns
    -------
    np.ndarray (len(thetas), len(phis))
        Energy of each cell in kJ/mol, offset so the minimum is 0. Cells with
        no samples are np.inf.
    """
    theta_edges = _edges(thetas)
    counts, _, _ = np.histogram2d(angles, dihedrals, bins=[theta_edges, _edges(phis)])
    # Exact angular measure of each bending angle bin
    measure = np.cos(theta_edges[:-1]) - np.cos(theta_edges[1:])
    density = counts / np.clip(measure, 1e-12, None)[:, None]
    table = np.full(density.shape, np.inf)
    sampled = density > 0
    if not sampled.any():
        raise ValueError("No samples fell inside the grid.")
    rt = GAS_CONSTANT * temperature
    table[sampled] = -rt * np.log(density[sampled])
    table[sampled] -= table[sampled].min()
    return table


class FreeEnergyAccumulator:
    """Pools Boltzmann weighted samples across refinement rounds.

    Builds F = -RT ln <exp(-U / RT)> over the samples landing in a cell. The
    average over the ensemble is what supplies the entropy, since each sample
    carries different values of every other coordinate.

    Each round draws from a different reference distribution, but the average
    within a cell does not depend on that reference as long as the reference
    is close to constant across the cell. Pooling keeps a cell that later
    rounds stop visiting from reverting to np.inf, which would otherwise let
    a sharpening sampler shrink its own support round on round.

    Parameters
    ----------
    thetas, phis : array-like, required
        Bin centers in radians.
    temperature : float, required
        Temperature in Kelvin.
    """

    def __init__(self, thetas, phis, temperature):
        self.thetas = np.asarray(thetas, dtype=float)
        self.phis = np.asarray(phis, dtype=float)
        self.temperature = float(temperature)
        self._theta_edges = _edges(self.thetas)
        self._phi_edges = _edges(self.phis)
        self.weight_sum = np.zeros((self.thetas.size, self.phis.size))
        self.counts = np.zeros((self.thetas.size, self.phis.size), dtype=int)
        self.shift = None

    def add(self, pairs, energies):
        """Add one round of samples.

        Parameters
        ----------
        pairs : list of np.ndarray (K, 2), required
            Per walk (theta, phi) pairs, from ``sample_walks``.
        energies : np.ndarray (M,), required
            Per walk energy in kJ/mol.
        """
        if len(energies) == 0:
            return
        rt = GAS_CONSTANT * self.temperature
        lowest = float(np.min(energies))
        if self.shift is None:
            self.shift = lowest
        elif lowest < self.shift:
            # Rescale earlier rounds onto the new reference energy
            self.weight_sum *= np.exp((lowest - self.shift) / rt)
            self.shift = lowest
        for pair, energy in zip(pairs, energies):
            weight = np.exp(-(energy - self.shift) / rt)
            i = np.clip(
                np.digitize(pair[:, 0], self._theta_edges) - 1, 0, self.thetas.size - 1
            )
            j = np.clip(
                np.digitize(pair[:, 1], self._phi_edges) - 1, 0, self.phis.size - 1
            )
            np.add.at(self.weight_sum, (i, j), weight)
            np.add.at(self.counts, (i, j), 1)

    def table(self):
        """Free energy of each cell in kJ/mol. Unsampled cells are np.inf."""
        if self.shift is None:
            raise ValueError("No samples to build a table from.")
        rt = GAS_CONSTANT * self.temperature
        table = np.full(self.weight_sum.shape, np.inf)
        filled = self.counts > 0
        table[filled] = self.shift - rt * np.log(
            self.weight_sum[filled] / self.counts[filled]
        )
        return table


def free_energy_table(pairs, energies, thetas, phis, temperature):
    """Free energy of each (theta, phi) cell from one set of samples, in kJ/mol.

    See ``FreeEnergyAccumulator``, which this wraps, for pooling several
    rounds of samples into one table.
    """
    accumulator = FreeEnergyAccumulator(thetas, phis, temperature)
    accumulator.add(pairs, energies)
    return accumulator.table()


def marginal_free_energy(table, temperature, axis=1):
    """Free energy along one coordinate of a 2D table, in kJ/mol.

    Integrates the other coordinate out with
    F(x) = -RT ln sum(exp(-F(x, y) / RT)), which keeps the entropy of the
    coordinate being removed. A minimum along the axis would discard it.

    Parameters
    ----------
    table : np.ndarray (N, M), required
        Free energy of each cell in kJ/mol. Cells set to np.inf carry no
        weight.
    temperature : float, required
        Temperature in Kelvin.
    axis : int, default 1
        Axis summed over. Use 1 to leave the bending angle and 0 to leave
        the dihedral.

    Returns
    -------
    np.ndarray
        Free energy along the remaining coordinate, offset so its minimum is
        0. Entries whose cells are all np.inf stay np.inf.
    """
    finite = np.isfinite(table)
    if not finite.any():
        raise ValueError("table has no finite cells to marginalize.")
    rt = GAS_CONSTANT * temperature
    shift = table[finite].min()
    weights = np.where(finite, np.exp(-(table - shift) / rt), 0.0)
    total = weights.sum(axis=axis)
    marginal = np.full(total.shape, np.inf)
    sampled = total > 0
    marginal[sampled] = -rt * np.log(total[sampled])
    marginal[sampled] -= marginal[sampled].min()
    return marginal


def effective_sample_size(energies, temperature):
    """Number of effective samples after Boltzmann reweighting the ensemble."""
    weights = boltzmann_weights(energies, temperature)
    return float(1.0 / np.sum(weights**2))


def _edges(grid):
    """Bin edges from bin centers, using midpoints between neighbors."""
    edges = np.empty(grid.size + 1)
    edges[1:-1] = 0.5 * (grid[:-1] + grid[1:])
    edges[0] = grid[0] - (edges[1] - grid[0])
    edges[-1] = grid[-1] + (grid[-1] - edges[-2])
    return edges
