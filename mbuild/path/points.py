import logging

import numpy as np

from mbuild.exceptions import PathConvergenceError
from mbuild.path.constraints import CuboidConstraint, CylinderConstraint

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
        batch_angles, batch_vectors = generate_trials(state)
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

    def sample(self, size=None):
        # Resolve against self.rng at call time so the rng can be swapped in.
        return getattr(self.rng, self.distribution)(size=size, **self.kwargs)


def min_bond_angle(bond_length, radius):
    """Return the smallest bond angle that avoids a 1-3 overlap, in radians.

    Sites two bonds apart are separated by ``2 bond_length sin(theta / 2)``,
    so `check_path` rejects any bond angle below
    ``2 arcsin(radius / (2 bond_length))``. Overlaps with sites further along
    the chain are not accounted for.

    Parameters
    ----------
    bond_length : float, required
        Bond length of the walk, in nm.
    radius : float, required
        Hard-sphere exclusion distance, in nm. This is the center-to-center
        minimum separation `check_path` enforces, not half of it.

    Returns
    -------
    float
        Minimum bond angle in radians.

    Raises
    ------
    ValueError
        If ``radius`` is at least twice ``bond_length``, where no bond angle
        avoids the overlap.
    """
    ratio = float(radius) / (2.0 * float(bond_length))
    if ratio >= 1.0:
        raise ValueError(
            f"{radius=} is at least twice {bond_length=}, so no bond angle "
            "avoids an overlap with the site two bonds back."
        )
    return float(2.0 * np.arcsin(ratio))


def _cosine_table(kappa, n_bins, min_angle):
    """Return (bin centers, probabilities) for the cosine bending potential.

    Bins are equal width over ``[min_angle, pi]`` and each is weighted by
    ``sin(theta) exp(-kappa (1 + cos(theta)))``, normalized to sum to 1.
    """
    edges = np.linspace(float(min_angle), np.pi, int(n_bins) + 1)
    angles = 0.5 * (edges[:-1] + edges[1:])
    # p(theta) ~ sin(theta) exp(-U(theta) / kBT), in logs to survive large kappa.
    # In logs so large kappa does not overflow.
    log_weights = np.log(np.sin(angles)) - float(kappa) * (1.0 + np.cos(angles))
    weights = np.exp(log_weights - log_weights.max())
    return angles, weights / weights.sum()


def _table_mean_cos(angles, probabilities):
    """Mean cosine of the deflection angle for a tabulated bond angle."""
    return float(-np.sum(probabilities * np.cos(angles)))


def _ratio_from_mean_cos(mean_cos):
    """Characteristic ratio from a mean deflection cosine."""
    if mean_cos >= 1.0:
        return np.inf
    return float((1.0 + mean_cos) / (1.0 - mean_cos))


class CosineAnglesSampler(AnglesSampler):
    """Samples bond angles from the Faller-Muller-Plathe bending potential.

    Builds a tabulated angle distribution and draws from it with
    `numpy.random.Generator.choice`, so sampled angles are bin centers.

    The potential is ``U(Theta) = kappa kBT (1 - cos Theta)``, where ``Theta``
    is the angle between subsequent bonds. `random_coordinate` measures the
    interior bond angle ``theta = pi - Theta`` and draws the azimuth
    uniformly, so the table includes the ``sin(theta)`` measure of the sphere:

        p(theta) ~ sin(theta) exp(-kappa (1 + cos(theta)))

    ``kappa`` is dimensionless, already reduced by kBT, so no temperature is
    needed.

    Parameters
    ----------
    kappa : float, required
        Bending stiffness in units of kBT. Positive favors extended chains,
        negative favors folded ones, and 0 gives a freely jointed chain.
    n_bins : int, default 360
        Number of equal-width bins spanning ``[min_angle, pi]``. Sets the
        angular resolution of the sampled angles.
    min_angle : float, default 0.0
        Lower edge of the table in radians. Angles below it are never drawn.
        See `min_bond_angle` for the value a hard-sphere walk rejects anyway.
    rng : numpy.random.Generator, optional
        Defaults to numpy.random.default_rng(). `RandomWalkState` replaces
        this with the walk's generator when the sampler is passed to
        `hard_sphere_random_walk`.

    Attributes
    ----------
    kappa, n_bins, min_angle : as passed in.

    Notes
    -----
    `characteristic_ratio` and `kuhn_length` describe an ideal chain. A walk
    with a finite ``radius`` rejects low-angle candidates and measures larger
    values than both.

    Examples
    --------
    >>> sampler = CosineAnglesSampler(kappa=1.5)
    >>> round(sampler.characteristic_ratio, 3)
    2.56

    >>> sampler = CosineAnglesSampler(
    ...     kappa=1.5, min_angle=min_bond_angle(bond_length=0.25, radius=0.22)
    ... )
    """

    def __init__(self, kappa, n_bins=360, min_angle=0.0, rng=None):
        n_bins = int(n_bins)
        min_angle = float(min_angle)
        if n_bins < 2:
            raise ValueError(f"{n_bins=} must be at least 2.")
        if not 0.0 <= min_angle < np.pi:
            raise ValueError(f"{min_angle=} must be in [0, pi) radians.")
        self.kappa = float(kappa)
        self.n_bins = n_bins
        self.min_angle = min_angle
        angles, probabilities = _cosine_table(self.kappa, n_bins, min_angle)
        super().__init__("choice", {"a": angles, "p": probabilities}, rng=rng)

    @property
    def bin_angles(self):
        """Bond angles the sampler can return, in radians."""
        return self.kwargs["a"]

    @property
    def probabilities(self):
        """Probability of each entry in `bin_angles`, summing to 1."""
        return self.kwargs["p"]

    @property
    def table(self):
        """The (2, n_bins) angle and probability array, as `rw_angles` takes."""
        return np.vstack((self.bin_angles, self.probabilities))

    @property
    def mean_cos_deflection(self):
        """Mean cosine of the deflection angle, ``<cos Theta>``.

        Computed from the table, so ``min_angle`` and ``n_bins`` are
        reflected. With ``min_angle`` of 0 this converges to the Langevin
        function ``coth(kappa) - 1 / kappa``.
        """
        return _table_mean_cos(self.bin_angles, self.probabilities)

    @property
    def characteristic_ratio(self):
        """Ideal characteristic ratio, ``(1 + <cos Theta>) / (1 - <cos Theta>)``."""
        return _ratio_from_mean_cos(self.mean_cos_deflection)

    def kuhn_length(self, bond_length):
        """Return the ideal Kuhn length, ``characteristic_ratio * bond_length``.

        Parameters
        ----------
        bond_length : float, required
            Bond length of the walk, in nm.

        Returns
        -------
        float
            Kuhn length in the units of ``bond_length``.
        """
        return float(bond_length) * self.characteristic_ratio

    @classmethod
    def from_characteristic_ratio(cls, c_infinity, n_bins=360, min_angle=0.0, rng=None):
        """Create a sampler with a given ideal characteristic ratio.

        Solves ``characteristic_ratio == c_infinity`` for ``kappa`` against
        the table that ``n_bins`` and ``min_angle`` produce.

        Parameters
        ----------
        c_infinity : float, required
            Target characteristic ratio. Values below 1 give a negative
            ``kappa``.
        n_bins : int, default 360
            Passed to the constructor and used during the solve.
        min_angle : float, default 0.0
            Passed to the constructor and used during the solve.
        rng : numpy.random.Generator, optional

        Returns
        -------
        CosineAnglesSampler

        Raises
        ------
        ValueError
            If ``c_infinity`` is not positive, or if no ``kappa`` reaches it
            for the given ``n_bins`` and ``min_angle``.
        """
        from scipy.optimize import brentq

        c_infinity = float(c_infinity)
        if c_infinity <= 0.0:
            raise ValueError(f"{c_infinity=} must be greater than 0.")
        target = (c_infinity - 1.0) / (c_infinity + 1.0)

        def mean_cos(kappa):
            return _table_mean_cos(*_cosine_table(kappa, n_bins, min_angle))

        # mean_cos rises monotonically in kappa, so the bracket ends bound it.
        low, high = -700.0, 700.0
        low_cos, high_cos = mean_cos(low), mean_cos(high)
        if not low_cos <= target <= high_cos:
            raise ValueError(
                f"{c_infinity=} is outside the "
                f"[{_ratio_from_mean_cos(low_cos):.4g}, "
                f"{_ratio_from_mean_cos(high_cos):.4g}] this table can reach "
                f"with {n_bins=} and {min_angle=}. Lower min_angle to reach a "
                "floppier chain, or raise n_bins to reach a stiffer one."
            )
        kappa = brentq(lambda value: mean_cos(value) - target, low, high)
        return cls(kappa, n_bins=n_bins, min_angle=min_angle, rng=rng)

    @classmethod
    def from_kuhn_length(
        cls, kuhn_length, bond_length, n_bins=360, min_angle=0.0, rng=None
    ):
        """Create a sampler with a given ideal Kuhn length.

        Divides by ``bond_length`` and defers to `from_characteristic_ratio`.

        Parameters
        ----------
        kuhn_length : float, required
            Target Kuhn length, in the same units as ``bond_length``.
        bond_length : float, required
            Bond length of the walk, in nm.
        n_bins : int, default 360
            Passed to `from_characteristic_ratio`.
        min_angle : float, default 0.0
            Passed to `from_characteristic_ratio`.
        rng : numpy.random.Generator, optional

        Returns
        -------
        CosineAnglesSampler

        Raises
        ------
        ValueError
            If ``bond_length`` is not positive, or as raised by
            `from_characteristic_ratio`.
        """
        bond_length = float(bond_length)
        if bond_length <= 0.0:
            raise ValueError(f"{bond_length=} must be greater than 0.")
        return cls.from_characteristic_ratio(
            float(kuhn_length) / bond_length,
            n_bins=n_bins,
            min_angle=min_angle,
            rng=rng,
        )

    def __repr__(self):
        return (
            f"CosineAnglesSampler(kappa={self.kappa:.4g}, "
            f"n_bins={self.n_bins}, min_angle={self.min_angle:.4g}, "
            f"c_infinity={self.characteristic_ratio:.4g})"
        )


def generate_trials(state):
    """Use normal or uniform sampling on angles, uniform sampling on radius."""
    thetas = state.angles.sample(size=state.trial_batch_size).astype(np.float32)
    r = state.rng.uniform(-0.5, 0.5, size=(state.trial_batch_size, 3)).astype(
        np.float32
    )
    return thetas, r


def generate_crosslink_init_points():
    pass
