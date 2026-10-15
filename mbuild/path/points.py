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
    batch_angles, batch_vectors = generate_trials(state)
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
        batch_angles, batch_vectors = generate_trials(state)
        # TODO: If building from a path with coordinates, can we try to get both pos1 and pos2?
        xyzs = next_step(
            pos1=None,  # will generate sphere of points around pos2
            pos2=starting_xyz,
            bond_length=state.initial_point_distance,
            thetas=batch_angles,
            r_vectors=batch_vectors,
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
    """Sample bond angles, and optionally dihedral angles, for a random walk.

    Parameters
    ----------
    distributionStr : str
        The numpy Generator method used to sample bond angles (radians):
        "uniform", "normal" or "choice".
    kwargs : dict
        Keyword arguments for that method.
    rng : np.random.Generator, optional
        Random number generator. A random walk replaces it with its own.
    dihedrals : AnglesSampler, optional
        Sampler for the dihedral angle (radians) formed by the new site and
        the three sites before it, using the IUPAC convention (trans is pi,
        gauche is about +-pi/3). When None (default), the new site is placed at
        a uniform random azimuth around the last bond. Dihedrals are applied
        once three bonded sites precede the new site.

    Notes
    -----
    "uniform" distribution should use 'low' and 'high' as kwargs.
    "normal" distribution should use 'loc' and 'scale' as kwargs.
    "choice" distribution should use 'a' and optionally 'p' as kwargs.

    Examples
    --------
    Trans and gauche states with 60/40 trans/gauche populations:

    >>> dihedrals = AnglesSampler(
    ...     "choice", {"a": [np.pi, np.pi / 3, -np.pi / 3], "p": [0.6, 0.2, 0.2]}
    ... )
    >>> sampler = AnglesSampler("normal", {"loc": 1.98, "scale": 0.05}, dihedrals=dihedrals)
    """

    def __init__(self, distributionStr, kwargs, rng=None, dihedrals=None):
        if dihedrals is not None and not isinstance(dihedrals, AnglesSampler):
            raise TypeError(
                f"dihedrals must be an AnglesSampler or None, not {type(dihedrals)}."
            )
        if dihedrals is not None and dihedrals.dihedrals is not None:
            raise ValueError("The dihedrals sampler cannot have its own dihedrals.")
        self.dihedrals = dihedrals
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

    @property
    def rng(self):
        return self._rng

    @rng.setter
    def rng(self, rng):
        # The dihedrals sampler shares the rng so one seed drives the walk.
        self._rng = rng
        if self.dihedrals is not None:
            self.dihedrals.rng = rng

    def sample(self, size=None):
        # Resolve against self.rng at call time so the rng can be swapped in.
        return getattr(self.rng, self.distribution)(size=size, **self.kwargs)


def dihedral_vectors(sites, phis):
    """Return unit vectors that place a new site at the given dihedrals.

    The vectors are perpendicular to the last bond, ``sites[2] - sites[1]``.
    A new site placed off ``sites[2]`` along the last bond direction rotated
    toward one of these vectors forms the dihedral ``phi`` with ``sites``,
    whatever the bond angle.

    Parameters
    ----------
    sites : np.ndarray, shape (3, 3)
        The three sites before the new site in bonded order, oldest first,
        unwrapped so that consecutive sites are a bond apart.
    phis : np.ndarray, shape (N,)
        Dihedral angles in radians (IUPAC convention, trans is pi).

    Returns
    -------
    np.ndarray, shape (N, 3) or None
        The vectors, or None if the three sites are collinear and the
        dihedral is undefined.
    """
    sites = np.asarray(sites, dtype=np.float64)
    axis = sites[2] - sites[1]
    axis /= np.linalg.norm(axis)
    # Component of the first bond perpendicular to the last bond, pointing
    # from the axis toward the oldest site: the cis direction (phi = 0).
    cis = sites[0] - sites[1]
    cis -= np.dot(cis, axis) * axis
    cis_norm = np.linalg.norm(cis)
    if cis_norm < 1e-6:
        return None
    cis /= cis_norm
    side = np.cross(axis, cis)
    phis = np.asarray(phis, dtype=np.float64)
    return (np.cos(phis)[:, None] * cis + np.sin(phis)[:, None] * side).astype(
        np.float32
    )


def generate_trials(state, dihedral_sites=None):
    """Sample trial bond angles and the trial vectors that orient them.

    Parameters
    ----------
    state : RandomWalkState
        The random walk state, which holds the angles sampler and the rng.
    dihedral_sites : np.ndarray, shape (3, 3), optional
        The three sites before the new site in bonded order, oldest first and
        unwrapped. Used only when the angles sampler has a dihedrals sampler;
        otherwise, or when None, trial vectors have a uniform random azimuth.
    """
    thetas = state.angles.sample(size=state.trial_batch_size).astype(np.float32)
    dihedrals = state.angles.dihedrals
    if dihedrals is not None and dihedral_sites is not None:
        phis = dihedrals.sample(size=state.trial_batch_size)
        r = dihedral_vectors(dihedral_sites, phis)
        if r is not None:
            return thetas, r
    # Only the direction of r is used; normal sampling is isotropic, giving a
    # uniform azimuth around the chain direction.
    r = state.rng.normal(size=(state.trial_batch_size, 3)).astype(np.float32)
    return thetas, r


def generate_crosslink_init_points():
    pass
