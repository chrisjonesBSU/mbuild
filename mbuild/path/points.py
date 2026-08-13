import logging

import numpy as np

from mbuild.exceptions import PathConvergenceError
from mbuild.path.constraints import CuboidConstraint, CylinderConstraint

logger = logging.getLogger(__name__)

# Molar gas constant in kJ/mol/K
GAS_CONSTANT = 8.314462618e-3


def get_second_point(state, existing_points, beads, check_path, next_step):
    """Generate a secound point from the given first point using RandomWalkState.

    Candidates are generated around the chain tip using a batch of trial angles
    and vectors. If the walk uses ``link-linear`` connectivity and a previous
    direction is available, candidates are angle-constrained relative to that
    direction; otherwise a sphere of candidates is generated around the tip.

    Parameters
    ----------
    state : RandomWalkState
        The current state of the random walk, containing radius, bond_length,
        tolerance, connectivity, volume_constraint, bias, and include_compound.
    existing_points : np.ndarray, shape (N, 3)
        Live coordinates accepted so far, sliced to ``coordinates[:state.count]``.
        Must contain at least one point (the initial point).
    beads : np.ndarray of str, shape (N,)
        Bead names corresponding to ``existing_points``, sliced to the same
        live length.
    check_path : callable
        Overlap-check function with signature
        ``check_path(existing_points, new_point, radius, tolerance) -> bool``.
    next_step : callable
        Coordinate-generation function with signature
        ``next_step(pos1, pos2, bond_length, thetas, r_vectors) -> np.ndarray``.
        Pass ``pos1=None`` to generate a sphere around ``pos2``.

    Returns
    -------
    np.ndarray, shape (3,) or None
        The accepted second coordinate, or ``None`` if no valid candidate was
        found within the trial batch.

    """
    batch_angles, _, batch_vectors = generate_trials(state)
    # If this RW is using link linear, pos2 = last site of last
    # Set pos1 and pos2 before checking include compound and combining coordinates
    if state.connectivity == "link-linear" and len(existing_points) > 1:
        pos1 = existing_points[-1]
        pos2 = existing_points[-2]
    else:
        pos1 = None
        pos2 = existing_points[-1]
    # Update existing points and beads to include those in the compound.
    if state.include_compound:
        compound_xyz = state.include_compound.xyz
        compound_names = np.array([p.name for p in state.include_compound.particles()])
        existing_points = np.concat((existing_points, compound_xyz))
        beads = np.concat((beads, compound_names))

    xyzs = next_step(
        pos1=pos1,
        pos2=pos2,
        bond_length=state.bond_length,
        thetas=batch_angles,
        r_vectors=batch_vectors,
    )
    if state.volume_constraint:
        is_inside_mask = state.volume_constraint.is_inside(
            points=xyzs, buffer=state.radius
        )
        xyzs = xyzs[is_inside_mask]

    if state.bias:
        xyzs = state.bias(candidates=xyzs, coordinates=existing_points, names=beads)

    for xyz in xyzs:
        if check_path(
            existing_points=existing_points,
            new_point=xyz,
            radius=state.radius,
            tolerance=state.tolerance,
        ):
            return xyz
    return None


def get_initial_point(state, existing_points, beads, check_path, next_step):
    """Generate a starting pointt from a RandomWalkState.

    The strategy for choosing a starting point depends on ``state.initial_point``:

    - A coordinate: the array is used directly as the starting coordinate.
    - A site index, indicated by ``state.starting_from_site``: indexes into
      ``existing_points``; a new point is generated in a sphere around that
      coordinate, filtered by volume constraint and bias, and checked for
      overlaps.
    - None with volume_constraint: candidates are sampled from the volume
      constraint's low-density regions and checked for overlaps.
    - None without volume_constraint: candidates are drawn uniformly at
      random within the bounding box of ``existing_points`` (or a unit sphere
      around the origin if no points exist yet) and checked for overlaps.

    If ``state.include_compound`` is set, its coordinates are appended to
    ``existing_points`` before overlap checks so that the new point does not
    clash with the included compound's atoms.

    Parameters
    ----------
    state : RandomWalkState
        The current state of the random walk, containing initial_point, radius,
        bond_length, tolerance, connectivity, volume_constraint, bias,
        include_compound, and the trial-generation parameters.
    existing_points : np.ndarray, shape (N, 3)
        Live coordinates accepted so far, sliced to ``coordinates[:state.count]``.
        May be empty at the start of the first walk.
    beads : np.ndarray of str, shape (N,)
        Bead names corresponding to ``existing_points``, sliced to the same
        live length. Passed to the bias for sequence-aware scoring.
    check_path : callable
        Overlap-check function with signature
        ``check_path(existing_points, new_point, radius, tolerance) -> bool``.
    next_step : callable
        Coordinate-generation function with signature
        ``next_step(pos1, pos2, bond_length, thetas, r_vectors) -> np.ndarray``.
        Used only when ``state.initial_point`` is a site index.

    Returns
    -------
    np.ndarray, shape (3,)
        The accepted starting coordinate.

    Raises
    ------
    ValueError
        If ``state.initial_point`` is a site index that is out of bounds for
        ``existing_points``.
    PathConvergenceError
        If no valid starting point can be found within the trial batch,
        regardless of which strategy is used.
    """
    # Get len of existing points before adding coords from include compound
    n_walk_points = len(existing_points)
    # Include a Compound's coordinates and bead names
    if state.include_compound:
        compound_xyz = state.include_compound.xyz
        compound_names = np.array([p.name for p in state.include_compound.particles()])
        existing_points = np.concat((existing_points, compound_xyz))
        beads = np.concat((beads, compound_names))

    # An initial point was manually given in hard_sphere_random_walk, use that.
    # Check if this point causes any overlaps, if so, raise error.
    if state.initial_point is not None and not state.starting_from_site:
        if check_path(
            existing_points=existing_points,
            new_point=state.initial_point,
            radius=state.radius,
            tolerance=state.tolerance,
        ):
            return state.initial_point
        raise PathConvergenceError(
            f"The provided initial_point {state.initial_point} overlaps with "
            "existing particles. Try a different starting point."
        )

    # Passing in an index to specify an initial point from already defined set of coordinates
    elif state.starting_from_site:
        if state.initial_point >= n_walk_points:
            raise ValueError(
                f"You passed a starting index of {state.initial_point} "
                f"but there are only {n_walk_points} existing points in the path."
            )
        # generate point off of current path coordinates
        starting_xyz = existing_points[state.initial_point]
        batch_angles, _, batch_vectors = generate_trials(state)
        # TODO: If building from a path with coordinates, can we try to get both pos1 and pos2?
        xyzs = next_step(
            pos1=None,  # will generate sphere of points around pos2
            pos2=starting_xyz,
            bond_length=state.bond_length,
            thetas=batch_angles,
            r_vectors=batch_vectors,
        )
        if state.volume_constraint:
            is_inside_mask = state.volume_constraint.is_inside(
                points=xyzs, buffer=state.radius
            )
            xyzs = xyzs[is_inside_mask]

        if state.bias:
            xyzs = state.bias(candidates=xyzs, coordinates=existing_points, names=beads)

        # Set up PBC info from volume constraints
        if isinstance(state.volume_constraint, CuboidConstraint):
            pbc = state.volume_constraint.pbc
            box_lengths = state.volume_constraint.box_lengths.astype(np.float32)
        elif isinstance(state.volume_constraint, CylinderConstraint):
            pbc = (False, False, state.volume_constraint.periodic_height)
            box_lengths = np.array(
                [
                    state.volume_constraint.radius * 2,
                    state.volume_constraint.radius * 2,
                    state.volume_constraint.height,
                ]
            ).astype(np.float32)
        else:
            pbc = (None, None, None)
            box_lengths = (None, None, None)

        for i in range(len(xyzs)):
            xyz = xyzs[i]
            if any(pbc):
                xyz = state.volume_constraint.mins + np.mod(
                    xyz - state.volume_constraint.mins, box_lengths
                )
            if check_path(  # check for overlaps
                existing_points=existing_points,
                new_point=xyz,
                radius=state.radius,
                tolerance=state.tolerance,
            ):
                return xyz
        raise PathConvergenceError(
            f"Unable to find a starting point at {starting_xyz} "
            "without overlapping particles. "
            "Check your `initial_point` argument."
        )
    # No initial point given, but there is a volume constraint to sample starting points from
    # TODO: Use find_low_density_point here instead?
    elif state.volume_constraint:
        xyzs = state.volume_constraint.sample_candidates(
            points=existing_points,
            n_candidates=300,
            buffer=state.radius + 0.1,
            rng=state.rng,
        )
        for xyz in xyzs:
            if check_path(
                existing_points=existing_points,
                new_point=xyz,
                radius=state.radius,
                tolerance=state.tolerance,
            ):
                return xyz
        raise PathConvergenceError(
            "Unable to find a starting point without overlapping particles. "
            "The density of the volume constraint may be too high."
        )
    else:  # completely random point
        if len(existing_points) == 0:
            max_dist = state.radius
            min_dist = -1 * state.radius
        else:
            # TODO: Update seed based on initial_point
            padding = state.bond_length + state.tolerance
            max_dist = np.max(existing_points, axis=0) + padding
            min_dist = np.min(existing_points, axis=0) - padding
        xyzs = state.rng.uniform(low=min_dist, high=max_dist, size=(300, 3))
        for xyz in xyzs:
            if check_path(
                existing_points=existing_points,
                new_point=xyz,
                radius=state.radius,
                tolerance=state.tolerance,
            ):
                return xyz
        raise PathConvergenceError(
            "Unable to find a starting point without overlapping particles. "
            "The density of the volume constraint may be too high."
        )


class AnglesSampler:
    """
    TODO:Allow for passing a specific 2D weighted pre-defined sample, as opposed to a numpy distribution.
    This would use np.random.choice instead as the sample method.
    NOTES
    -----
    "uniform" distribution should use 'low' and 'high' as kwargs.
    "normal" distribution should use 'loc' and 'scale' as kwargs.
    """

    def __init__(self, distributionStr, kwargs, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng()
        distribution = distributionStr.lower()
        if distribution == "uniform":
            assert "low" in kwargs
            assert "high" in kwargs
        elif distribution == "normal":
            assert "loc" in kwargs
            assert "scale" in kwargs
        elif distribution == "choice":
            assert "a" in kwargs  # p is not required
        else:
            raise NotImplementedError(
                f"Sample Distribution {distributionStr} not supported."
            )
        self.distribution = distribution
        self.kwargs = kwargs

    @classmethod
    def from_energies(cls, angles, energies, temperature=300.0, rng=None):
        """Create a sampler that draws angles weighted by a tabulated energy.

        Parameters
        ----------
        angles : array-like (N,), required
            Angle values in radians.
        energies : array-like (N,), required
            Energy of each angle in kJ/mol. Values of np.inf are never sampled.
        temperature : float, default 300.0
            Temperature in Kelvin used to weight the energies.
        rng : numpy.random.Generator, optional
            Defaults to numpy.random.default_rng().

        Returns
        -------
        AnglesSampler
            A sampler over the "choice" distribution, drawing values from
            `angles`. Only the given angle values are returned, so the grid
            sets the resolution of the sampled distribution.
        """
        angles = np.asarray(angles, dtype=float)
        if angles.ndim != 1:
            raise ValueError("angles must be 1D.")
        weights = boltzmann_weights(energies, temperature)
        if weights.shape != angles.shape:
            raise ValueError(
                f"energies has shape {weights.shape}, expected {angles.shape} "
                "to match angles."
            )
        return cls("choice", {"a": angles, "p": weights}, rng=rng)

    def sample(self, size=None):
        # Resolve against self.rng at call time so the rng can be swapped in.
        return getattr(self.rng, self.distribution)(size=size, **self.kwargs)


def boltzmann_weights(energies, temperature):
    """Normalized Boltzmann weights for a tabulated energy.

    Parameters
    ----------
    energies : array-like, required
        Energies in kJ/mol. Values of np.inf are given zero weight.
    temperature : float, required
        Temperature in Kelvin.

    Returns
    -------
    numpy.ndarray
        Weights summing to 1, with the same shape as `energies`.
    """
    energies = np.asarray(energies, dtype=float)
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError(f"{temperature=} must be greater than 0.")
    finite = np.isfinite(energies)
    if not finite.any():
        raise ValueError("energies has no finite values to sample from.")
    shifted = energies - energies[finite].min()
    weights = np.where(finite, np.exp(-shifted / (GAS_CONSTANT * temperature)), 0.0)
    total = weights.sum()
    if total <= 0:
        raise ValueError(
            "All weights underflowed to 0. The energy table spans too large a "
            "range for this temperature."
        )
    return weights / total


class AngleDihedralSampler:
    """Samples bending angle and dihedral pairs from a 2D energy table.

    Draws a cell from a 2D grid of bending angles and dihedrals with
    probability proportional to exp(-E / RT), then returns an angle pair
    from within that cell. Correlation between the two comes from the
    energy table, so a table that is additive in angle and dihedral
    samples them independently.

    Parameters
    ----------
    theta_grid : array-like (N,), required
        Bending angle bin centers in radians.
    phi_grid : array-like (M,), required
        Dihedral bin centers in radians, spanning [-pi, pi).
    energies : array-like (N, M), required
        Energy of each (theta, phi) cell in kJ/mol. Cells set to np.inf
        are never sampled.
    temperature : float, default 300.0
        Temperature in Kelvin used to weight the energies.
    jitter : bool, default True
        Draw uniformly within the selected cell. When False, the bin
        centers are returned.
    rng : numpy.random.Generator, optional
        Defaults to numpy.random.default_rng().

    Notes
    -----
    Bin edges are the midpoints between neighboring grid values, so
    non-uniform grids are supported.
    """

    def __init__(
        self,
        theta_grid,
        phi_grid,
        energies,
        temperature=300.0,
        jitter=True,
        rng=None,
    ):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.theta_grid = np.asarray(theta_grid, dtype=float)
        self.phi_grid = np.asarray(phi_grid, dtype=float)
        self.energies = np.asarray(energies, dtype=float)
        if self.theta_grid.ndim != 1 or self.phi_grid.ndim != 1:
            raise ValueError("theta_grid and phi_grid must be 1D.")
        if self.theta_grid.size < 2 or self.phi_grid.size < 2:
            raise ValueError("theta_grid and phi_grid need at least 2 values each.")
        expected = (self.theta_grid.size, self.phi_grid.size)
        if self.energies.shape != expected:
            raise ValueError(
                f"energies has shape {self.energies.shape}, expected {expected} "
                "to match (theta_grid, phi_grid)."
            )
        self._theta_edges = _bin_edges(self.theta_grid)
        self._phi_edges = _bin_edges(self.phi_grid)
        self.jitter = bool(jitter)
        self.temperature = float(temperature)
        self.weights = boltzmann_weights(self.energies, self.temperature)
        self._flat_weights = self.weights.ravel()

    def sample(self, size=None):
        """Return (thetas, phis) drawn from the weighted energy table."""
        n = 1 if size is None else int(size)
        flat = self.rng.choice(self._flat_weights.size, size=n, p=self._flat_weights)
        i, j = np.unravel_index(flat, self.energies.shape)
        if self.jitter:
            thetas = self.rng.uniform(self._theta_edges[i], self._theta_edges[i + 1])
            phis = self.rng.uniform(self._phi_edges[j], self._phi_edges[j + 1])
        else:
            thetas = self.theta_grid[i]
            phis = self.phi_grid[j]
        if size is None:
            return thetas[0], phis[0]
        return thetas, phis


def _bin_edges(grid):
    """Bin edges from grid centers, using midpoints between neighbors."""
    edges = np.empty(grid.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (grid[:-1] + grid[1:])
    edges[0] = grid[0] - (edges[1] - grid[0])
    edges[-1] = grid[-1] + (grid[-1] - edges[-2])
    return edges


def generate_trials(state):
    """Use normal or uniform sampling on angles, isotropic sampling on radius.

    Returns
    -------
    thetas, phis, r_vectors
        `phis` is None when the walk has no dihedral sampler, which leaves
        the rotation about the last bond set by `r_vectors`.
    """
    if state.joint_angles is not None:
        thetas, phis = state.joint_angles.sample(size=state.trial_batch_size)
        thetas = thetas.astype(np.float32)
        phis = phis.astype(np.float32)
    else:
        thetas = state.angles.sample(size=state.trial_batch_size).astype(np.float32)
        if state.dihedrals is not None:
            phis = state.dihedrals.sample(size=state.trial_batch_size).astype(
                np.float32
            )
        else:
            phis = None
    r = state.rng.normal(size=(state.trial_batch_size, 3)).astype(np.float32)
    return thetas, phis, r


def generate_crosslink_init_points():
    pass
