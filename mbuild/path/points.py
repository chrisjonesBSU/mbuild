import logging

import numpy as np

from mbuild.exceptions import PathConvergenceError

logger = logging.getLogger(__name__)


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
    batch_angles, batch_vectors, _ = generate_trials(state)
    # If this RW links to an existing path, the first site of this walk is bonded
    # to state.attach_index, so that site sets the angle reference for the second.
    # Set pos1 and pos2 before checking include compound and combining coordinates
    if state.connectivity == "link-linear" and state.attach_index >= 0:
        pos1 = existing_points[-1]
        pos2 = existing_points[state.attach_index]
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
        pos3=None,
        phis=None,
    )
    if state.volume_constraint:
        is_inside_mask = state.volume_constraint.is_inside(
            points=xyzs, buffer=state.radius
        )
        xyzs = xyzs[is_inside_mask]

    if state.bias:
        xyzs = state.bias(candidates=xyzs, coordinates=existing_points, names=beads)

    if any(state.pbc):
        xyzs = (
            state.volume_constraint.mins
            + np.mod(xyzs - state.volume_constraint.mins, state.box_lengths)
        ).astype(np.float32)

    excluded_indices = state.excluded_indices()
    for xyz in xyzs:
        if check_path(
            existing_points=existing_points,
            new_point=xyz,
            radius=state.radius,
            tolerance=state.tolerance,
            pbc=state.pbc,
            box_lengths=state.box_lengths,
            excluded_indices=excluded_indices,
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

    excluded_indices = state.excluded_indices()

    # An initial point was manually given in hard_sphere_random_walk, use that.
    # Check if this point causes any overlaps, if so, raise error.
    if state.initial_point is not None and not state.starting_from_site:
        if check_path(
            existing_points=existing_points,
            new_point=state.initial_point,
            radius=state.radius,
            tolerance=state.tolerance,
            pbc=state.pbc,
            box_lengths=state.box_lengths,
            excluded_indices=excluded_indices,
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
        batch_angles, batch_vectors, _ = generate_trials(state)
        # TODO: If building from a path with coordinates, can we try to get both pos1 and pos2?
        xyzs = next_step(
            pos1=None,  # will generate sphere of points around pos2
            pos2=starting_xyz,
            bond_length=state.initial_point_distance,
            thetas=batch_angles,
            r_vectors=batch_vectors,
            pos3=None,
            phis=None,
        )
        if state.volume_constraint:
            is_inside_mask = state.volume_constraint.is_inside(
                points=xyzs, buffer=state.radius
            )
            xyzs = xyzs[is_inside_mask]

        if state.bias:
            bias_coords = np.concat((existing_points, starting_xyz[None, :]))
            bias_names = np.concat(
                (beads, beads[state.initial_point : state.initial_point + 1])
            )
            xyzs = state.bias(
                candidates=xyzs, coordinates=bias_coords, names=bias_names
            )

        for i in range(len(xyzs)):
            xyz = xyzs[i]
            if any(state.pbc):
                xyz = state.volume_constraint.mins + np.mod(
                    xyz - state.volume_constraint.mins, state.box_lengths
                )
            if check_path(  # check for overlaps
                existing_points=existing_points,
                new_point=xyz,
                radius=state.radius,
                tolerance=state.tolerance,
                pbc=state.pbc,
                box_lengths=state.box_lengths,
                excluded_indices=excluded_indices,
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
                pbc=state.pbc,
                box_lengths=state.box_lengths,
                excluded_indices=excluded_indices,
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
                pbc=state.pbc,
                box_lengths=state.box_lengths,
                excluded_indices=excluded_indices,
            ):
                return xyz
        raise PathConvergenceError(
            "Unable to find a starting point without overlapping particles. "
            "The density of the volume constraint may be too high."
        )


class AnglesSampler:
    """Sample angles (or dihedrals) from a named distribution.

    The ``"uniform"`` and ``"normal"`` distributions are minimal, out-of-the-box
    samplers. The ``"tabulated"`` distribution samples a value y from an arbitrary
    tabulated distribution P(y) (e.g. a target ``P(theta)`` / ``P(phi)`` measured
    from an MD simulation or built from a table potential).

    NOTES
    -----
    - ``"uniform"`` distribution should use ``'low'`` and ``'high'`` as kwargs.
    - ``"normal"`` distribution should use ``'loc'`` and ``'scale'`` as kwargs.
    - ``"choice"`` distribution should use ``'a'`` (and optionally ``'p'``).
    - ``"tabulated"`` distribution should use ``'values'`` (the y grid) and
      ``'probabilities'`` (P(y), un-normalized weights are fine — they are
      normalized internally). Optional ``'interpolate'`` (default ``True``)
      selects continuous inverse-CDF sampling; ``False`` samples discretely at
      the grid ``values``. The grid need not be sorted.
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
        elif distribution == "tabulated":
            self._init_tabulated(kwargs)
        else:
            raise NotImplementedError(
                f"Sample Distribution {distributionStr} not supported."
            )
        self.distribution = distribution
        self.kwargs = kwargs

    def _init_tabulated(self, kwargs):
        """Precompute the (normalized) CDF for inverse-transform sampling."""
        if "values" not in kwargs or "probabilities" not in kwargs:
            raise ValueError(
                "The 'tabulated' distribution requires 'values' and "
                "'probabilities' kwargs."
            )
        values = np.asarray(kwargs["values"], dtype=float)
        probs = np.asarray(kwargs["probabilities"], dtype=float)
        if values.shape != probs.shape or values.ndim != 1:
            raise ValueError(
                "'values' and 'probabilities' must be 1D arrays of equal length."
            )
        if np.any(probs < 0):
            raise ValueError("'probabilities' must be non-negative.")
        total = probs.sum()
        if total <= 0:
            raise ValueError("'probabilities' must sum to a positive value.")
        # Sort by value so the CDF is monotonic.
        order = np.argsort(values)
        self._values = values[order]
        self._probs = probs[order] / total
        self.interpolate = bool(kwargs.get("interpolate", True))
        # CDF on grid points, prepended with 0 so the inversion covers [0, 1).
        self._cdf = np.concatenate([[0.0], np.cumsum(self._probs)])
        self._cdf_values = np.concatenate([[self._values[0]], self._values])

    def sample(self, size=None):
        if self.distribution == "tabulated":
            n = 1 if size is None else size
            u = self.rng.uniform(size=n)
            if self.interpolate:
                # Continuous inverse-CDF (linear interpolation between grid points).
                out = np.interp(u, self._cdf, self._cdf_values)
            else:
                # Discrete: return the grid value of the bin u falls into.
                idx = np.clip(
                    np.searchsorted(self._cdf, u, side="right") - 1,
                    0,
                    len(self._values) - 1,
                )
                out = self._values[idx]
            return out if size is not None else out[0]
        # Resolve against self.rng at call time so the rng can be swapped in.
        return getattr(self.rng, self.distribution)(size=size, **self.kwargs)


def generate_trials(state):
    """Sample a batch of trial bond angles, random vectors, and (optionally) dihedrals.

    Angles are sampled from ``state.angles`` and the random vectors set the
    azimuth when no dihedral control is used. If a dihedral sampler was passed
    to ``hard_sphere_random_walk`` (``state.dihedrals`` is set), a batch of
    dihedral angles ``phis`` is also sampled; otherwise ``phis`` is ``None``.

    Returns
    -------
    thetas : np.ndarray (batch,)
    r_vectors : np.ndarray (batch, 3)
    phis : np.ndarray (batch,) or None
    """
    thetas = state.angles.sample(size=state.trial_batch_size).astype(np.float32)
    # Only the direction of r is used; normal sampling is isotropic, giving a
    # uniform azimuth around the chain direction.
    r = state.rng.normal(size=(state.trial_batch_size, 3)).astype(np.float32)
    phis = None
    if getattr(state, "dihedrals", None) is not None:
        phis = state.dihedrals.sample(size=state.trial_batch_size).astype(np.float32)
    return thetas, r, phis


def generate_crosslink_init_points():
    pass
