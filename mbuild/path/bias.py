"""Contains biases that can be included in mbuild.path.HardSphereRandomWalk."""

import numpy as np

from mbuild.path.path_utils import (
    nematic_q_director,
    order_alignment_scores,
)
from mbuild.path.path_utils import target_density as _target_density_cpu
from mbuild.path.path_utils import (
    target_sq_distances as _target_sq_distances_cpu,
)

try:
    from numba import cuda

    _CUDA_AVAILABLE = cuda.is_available()
    if _CUDA_AVAILABLE:
        from mbuild.path.path_utils_gpu import (
            target_density as _target_density_gpu,
        )
        from mbuild.path.path_utils_gpu import (
            target_sq_distances as _target_sq_distances_gpu,
        )
    else:
        _target_density_gpu = None
        _target_sq_distances_gpu = None

except Exception:  # pragma: no cover - CUDA stack not importable or GPU utils failed
    _CUDA_AVAILABLE = False
    _target_density_gpu = None
    _target_sq_distances_gpu = None


class Bias:
    def __init__(self, weight):
        if weight <= 0 or weight > 1:
            raise ValueError(
                "weight should be larger than 0 and smaller than or equal to 1."
            )
        self.weight = weight
        # Large beta diminishes the effect of noise
        self.beta = self.weight / max(1e-6, (1.0 - self.weight))
        self.noise_scale = 1 - self.weight

    def _attach_path(self, path, state):
        """Create access Path and RandomWalkState used by hard_sphere_random_walk."""
        self.path = path
        self.state = state
        # Inherit rng from the path for use in Bias classes
        self.rng = self.state.rng
        # Decide which path_utils implementations to use based on the path's device.
        use_gpu = getattr(state, "run_on_gpu", False)
        if use_gpu and _CUDA_AVAILABLE and _target_sq_distances_gpu is not None:
            self._target_sq_distances = _target_sq_distances_gpu
            self._target_density = _target_density_gpu
        else:
            self._target_sq_distances = _target_sq_distances_cpu
            self._target_density = _target_density_cpu

    def _clean(self):
        self.path = None
        self.state = None
        self.rng = None
        self._target_sq_distances = None
        self._target_density = None

    def __call__(self, candidates, coordinates, names):
        """Implemented in sub classes of Bias."""
        raise NotImplementedError


class TargetCoordinate(Bias):
    """Bias next-moves so that ones moving closer to a target coordinate are more likely to be accepted."""

    def __init__(self, target_coordinate, weight):
        self.target_coordinate = np.asarray(target_coordinate)
        super(TargetCoordinate, self).__init__(weight=weight)

    def __call__(self, candidates, coordinates, names):
        """Sorts a set of candidate coordinates according to the bias.

        Parameters
        ----------
        candidates : np.ndarray (N,3), required
            Array of coordinates candidate sites for the next site in a random walk.
        coordinates : np.ndarray (N,3), required
            Array of coordinates for all previously accepted sites in a random walk.
        names : np.ndarray (N), required
            Array of site names for all previously accepted sites in a random walk.

        Returns
        -------
        candidates : np.ndarray (N,3)
            Returns the original candidate array, sorted according to the bias.
        """
        sq_distances = self._target_sq_distances(self.target_coordinate, candidates)
        noise = self.rng.normal(0, self.noise_scale, size=sq_distances.shape)
        scores = self.beta * sq_distances + noise
        # Target coordinate should favor short distances, sort in ascending order (np default)
        sort_idx = np.argsort(scores)
        return candidates[sort_idx]


class AvoidCoordinate(Bias):
    """Bias next-moves so that ones moving further from a specific coordinate are more likely to be accepted."""

    def __init__(self, avoid_coordinate, weight):
        self.avoid_coordinate = np.asarray(avoid_coordinate)
        super(AvoidCoordinate, self).__init__(weight=weight)

    def __call__(self, candidates, coordinates, names):
        """Sorts a set of candidate coordinates according to the bias.

        Parameters
        ----------
        candidates : np.ndarray (N,3), required
            Array of coordinates candidate sites for the next site in a random walk.
        coordinates : np.ndarray (N,3), required
            Array of coordinates for all previously accepted sites in a random walk.
        names : np.ndarray (N), required
            Array of site names for all previously accepted sites in a random walk.

        Returns
        -------
        candidates : np.ndarray (N,3)
            Returns the original candidate array, sorted according to the bias.
        """
        sq_distances = self._target_sq_distances(self.avoid_coordinate, candidates)
        noise = self.rng.normal(0, self.noise_scale, size=sq_distances.shape)
        scores = self.beta * sq_distances + noise
        # Avoid cooardinate should favor larger distances, sort in descending order
        sort_idx = np.argsort(scores)[::-1]
        return candidates[sort_idx]


class TargetType(Bias):
    """Bias next-moves so that ones increasing local density relative to a site type are more likely to be accepted."""

    def __init__(self, target_type, weight, r_cut):
        self.target_type = target_type
        self.r_cut = float(r_cut)
        super().__init__(weight=weight)

    def __call__(self, candidates, coordinates, names):
        """Sorts a set of candidate coordinates according to the bias.

        Parameters
        ----------
        candidates : np.ndarray (N,3), required
            Array of coordinates candidate sites for the next site in a random walk.
        coordinates : np.ndarray (N,3), required
            Array of coordinates for all previously accepted sites in a random walk.
        names : np.ndarray (N), required
            Array of site names for all previously accepted sites in a random walk.

        Returns
        -------
        candidates : np.ndarray (N,3)
            Returns the original candidate array, sorted according to the bias.
        """
        # Only get coordinates of sites where the site name == target type
        target_coords = coordinates[names == self.target_type]
        densities = self._target_density(
            candidates=candidates, target_coords=target_coords, r_cut=self.r_cut
        )
        noise = self.rng.normal(0, self.noise_scale, size=densities.shape)
        scores = self.beta * densities + noise
        # Target type should favor larger densities, sort in descending order
        sort_idx = np.argsort(scores)[::-1]
        return candidates[sort_idx]


class AvoidType(Bias):
    """Bias next-moves so that ones decreasing local density relative to a site type are more likely to be accepted."""

    def __init__(self, avoid_type, weight, r_cut):
        self.avoid_type = avoid_type
        self.r_cut = r_cut
        super().__init__(weight=weight)

    def __call__(self, candidates, coordinates, names):
        """Sorts a set of candidate coordinates according to the bias.

        Parameters
        ----------
        candidates : np.ndarray (N,3), required
            Array of coordinates candidate sites for the next site in a random walk.
        coordinates : np.ndarray (N,3), required
            Array of coordinates for all previously accepted sites in a random walk.
        names : np.ndarray (N), required
            Array of site names for all previously accepted sites in a random walk.

        Returns
        -------
        candidates : np.ndarray (N,3)
            Returns the original candidate array, sorted according to the bias.
        """
        target_coords = coordinates[names == self.avoid_type]
        densities = self._target_density(
            candidates=candidates, target_coords=target_coords, r_cut=self.r_cut
        )
        noise = self.rng.normal(0, self.noise_scale, size=densities.shape)
        scores = self.beta * densities + noise
        # Avoid type should favor smaller densities, sort in ascending order (np default)
        sort_idx = np.argsort(scores)
        return candidates[sort_idx]


class TargetDirection(Bias):
    """Bias next-moves so that ones moving along a target direction are more likely to be accepted."""

    def __init__(self, direction, weight):
        direction = np.asarray(direction, dtype=np.float32)
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            raise ValueError("Direction vector must be non-zero.")
        self.direction = direction / norm
        super().__init__(weight=weight)

    def __call__(self, candidates, coordinates, names):
        """Sorts a set of candidate coordinates according to the bias.

        Parameters
        ----------
        candidates : np.ndarray (N,3), required
            Array of coordinates candidate sites for the next site in a random walk.
        coordinates : np.ndarray (N,3), required
            Array of coordinates for all previously accepted sites in a random walk.
        names : np.ndarray (N), required
            Array of site names for all previously accepted sites in a random walk.

        Returns
        -------
        candidates : np.ndarray (N,3)
            Returns the original candidate array, sorted according to the bias.
        """
        last_step_pos = coordinates[-1]
        next_step_vectors = candidates - last_step_pos
        norms = np.linalg.norm(next_step_vectors, axis=1, keepdims=True)
        # Avoid divisions by zero
        norms = np.clip(norms, 1e-12, None)
        next_step_unit_vectors = next_step_vectors / norms
        # Alignment score: dot product with target direction
        alignment = np.dot(next_step_unit_vectors, self.direction)
        noise = self.rng.normal(0.0, self.noise_scale, size=alignment.shape)
        scores = self.beta * alignment + noise
        # Larger dot product = better alignment with target, sort descending
        sort_idx = np.argsort(scores)[::-1]
        return candidates[sort_idx]


class AvoidDirection(Bias):
    """Bias next-moves so that ones not moving along a certain direction are more likely to be accepted."""

    def __init__(self, direction, weight):
        self.direction = direction
        super().__init__(weight=weight)

    def __call__(self, candidates, coordinates, names):
        """Sorts a set of candidate coordinates according to the bias.

        Parameters
        ----------
        candidates : np.ndarray (N,3), required
            Array of coordinates candidate sites for the next site in a random walk.
        coordinates : np.ndarray (N,3), required
            Array of coordinates for all previously accepted sites in a random walk.
        names : np.ndarray (N), required
            Array of site names for all previously accepted sites in a random walk.

        Returns
        -------
        candidates : np.ndarray (N,3)
            Returns the original candidate array, sorted according to the bias.
        """
        last_step_pos = coordinates[-1]
        next_step_vectors = candidates - last_step_pos
        norms = np.linalg.norm(next_step_vectors, axis=1, keepdims=True)
        # Avoid divisions by zero
        norms = np.clip(norms, 1e-12, None)
        next_step_unit_vectors = next_step_vectors / norms
        # Alignment score: dot product with target direction
        alignment = np.dot(next_step_unit_vectors, self.direction)
        noise = self.rng.normal(0.0, self.noise_scale, size=alignment.shape)
        scores = self.beta * alignment + noise
        # Larger dot product = better alignment with target, sort ascending (np default)
        sort_idx = np.argsort(scores)
        return candidates[sort_idx]


class Ordering(Bias):
    """Bias next-moves toward increasing local (rod-like) nematic order.

    Favors candidates whose trial bond aligns with the local nematic director,
    building crystalline-*like* (semicrystalline) rod/extended-chain alignment
    with tunable imperfections rather than a clean lattice. The amount of
    disorder is set by ``weight`` (signal vs noise) and capped by
    ``target_order`` (how good a domain is allowed to get before the bias
    releases), so defects survive by design. For a perfect lattice use
    ``path.build.lamellar`` (or ``spherulite``) instead.

    Parameters
    ----------
    weight : float
        Bias strength in (0, 1], as for all ``Bias`` subclasses. Sets the
        signal/noise balance via the inherited ``beta`` and ``noise_scale``.
        Lower weight -> more amorphous/defective; higher -> crisper alignment.
    r_cut : float
        Locality cutoff. A bond contributes to the local order tensor if its
        midpoint lies within ``r_cut`` of the current chain tip. Same length
        units as the walk.
    target_order : float, optional, default 0.7
        Target scalar order S* in (0, 1]. The alignment reward is scaled by
        max(0, 1 - S_local / S*), so order is self-limiting: once a neighborhood
        reaches S*, the bias switches off there and the walk wanders (amorphous).
        Lower S* -> looser structure; higher S* -> more strongly aligned domains.
    min_local_bonds : int, optional, default 3
        Minimum number of in-cutoff bonds required to define a meaningful
        director. Below this the bias is off (noise-only sort) for that step.
    """

    def __init__(self, weight, r_cut, target_order=0.7, min_local_bonds=3):
        if r_cut <= 0:
            raise ValueError("r_cut should be positive.")
        if not (0.0 < target_order <= 1.0):
            raise ValueError("target_order should be in (0, 1].")
        self.r_cut = float(r_cut)
        self.target_order = float(target_order)
        self.min_local_bonds = int(min_local_bonds)
        self._static_bonds = None
        super().__init__(weight=weight)

    def _attach_path(self, path, state):
        """Attach path/state and cache previous chains' bonds as an int array.

        Edges for all already-completed chains live in ``path.bond_graph`` (each
        prior walk reconciled its graph at termination). We extract them once
        here, restricted to sites below ``init_count`` (the start of the current
        walk), into an (E, 2) int32 array. This avoids touching networkx -- which
        is not numba-friendly -- on every step. The current walk's own bonds are
        rebuilt cheaply each step from consecutive indices, since the graph is
        not yet populated for the in-progress chain.
        """
        super()._attach_path(path, state)
        init_count = int(getattr(state, "init_count", 0))
        edges = []
        bg = getattr(path, "bond_graph", None)
        if bg is not None and init_count > 0:
            for u, v in bg.edges():
                # Keep only edges fully within the previously-placed sites.
                if u < init_count and v < init_count:
                    edges.append((int(u), int(v)))
        if len(edges) > 0:
            self._static_bonds = np.asarray(edges, dtype=np.int32)
        else:
            self._static_bonds = np.empty((0, 2), dtype=np.int32)

    def _clean(self):
        super()._clean()
        self._static_bonds = None

    def _all_bonds(self, n_sites, init_count):
        """Combine cached previous-chain bonds with the current chain's
        consecutive-pair bonds (init_count .. n_sites-1). Never bonds across the
        init_count seam."""
        # Current chain: consecutive pairs (init_count, init_count+1), ...
        n_cur = n_sites - init_count
        if n_cur >= 2:
            starts = np.arange(init_count, n_sites - 1, dtype=np.int32)
            cur = np.empty((starts.shape[0], 2), dtype=np.int32)
            cur[:, 0] = starts
            cur[:, 1] = starts + 1
        else:
            cur = np.empty((0, 2), dtype=np.int32)

        if self._static_bonds is None or self._static_bonds.shape[0] == 0:
            return cur
        if cur.shape[0] == 0:
            return self._static_bonds
        return np.concatenate([self._static_bonds, cur], axis=0)

    def __call__(self, candidates, coordinates, names):
        """Sort candidate coordinates to favor increasing local nematic order.

        Parameters
        ----------
        candidates : np.ndarray (N, 3)
            Candidate sites for the next step.
        coordinates : np.ndarray (M, 3)
            All placed sites in the system (already trimmed to count by the
            walker), spanning every chain built so far.
        names : np.ndarray (M,)
            Site names (unused here, kept for interface).

        Returns
        -------
        np.ndarray (N, 3)
            The candidate array, sorted most-favored first.
        """
        n = candidates.shape[0]
        if n == 0:
            return candidates

        m = coordinates.shape[0]
        init_count = (
            int(getattr(self.state, "init_count", 0)) if self.state is not None else 0
        )
        if init_count < 0 or init_count > m:
            init_count = 0

        coords32 = coordinates.astype(np.float32)
        tip = coords32[-1]
        bonds = self._all_bonds(m, init_count)

        # Not enough bonds anywhere to define locality -> bias off.
        if bonds.shape[0] < self.min_local_bonds:
            noise = self.rng.normal(0.0, 1.0, size=n)
            return candidates[np.argsort(noise)]

        n_local, s_param, director = nematic_q_director(
            bonds, coords32, tip, self.r_cut
        )

        # Too few local bonds, or local order already saturated -> bias off.
        if n_local < self.min_local_bonds or s_param >= self.target_order:
            noise = self.rng.normal(0.0, 1.0, size=n)
            return candidates[np.argsort(noise)]

        scores = order_alignment_scores(
            candidates.astype(np.float32),
            tip,
            director,
            float(s_param),
            self.target_order,
        )
        noise = self.rng.normal(0.0, self.noise_scale, size=scores.shape)
        biased = self.beta * scores + noise
        sort_idx = np.argsort(biased)[::-1]
        return candidates[sort_idx]
