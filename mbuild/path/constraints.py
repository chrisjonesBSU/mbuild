import numpy as np
from numba import njit
from scipy.spatial import cKDTree


class Constraint:
    """
    Defines a volume that acts as a constraint in mbuild.path.HardSphereRandomWalk.
    This is the base class from which all constraints inherit from.

    Notes
    -----
    Design and implement your own volume constraint by inheriting from this class
    and implementing the constraint check in an `is_inside` method.

    `is_inside` is expected to return a mask of booleans.

    The existing contraints of CuboidConstraint, SphereConstraint, and CylinderConstraint
    call numba methods from their respectice `is_inside` method, but that is not a required
    implementation to design your own constraint.
    """

    def __init__(self):
        pass

    def is_inside(self, points, buffer):
        """Return True if point satisfies constraint (inside), else False."""
        raise NotImplementedError("Must be implemented in subclasses")

    def sample_candidates(self, points, n_candiates, buffer):
        """Sample the volume for canidate points sorted by lowest local density."""
        raise NotImplementedError("Must be implemented in subclasses")

    def find_low_density_points(self, points, n_candidates, buffer, k=10):
        low_density_points = self.sample_candidates(
            points=points, n_candidates=n_candidates, buffer=buffer, k=k
        )
        return low_density_points


class CuboidConstraint(Constraint):
    """Creates a cuboid constraint.

    Parameters
    ----------
    Lx : float, required
        The length of the volume along the x-axis.
    Ly : float, required
        The length of the volume along the y-axis.
    Lz : float, required
        The length of the volume along the z-axis.
    center : array-like (1,3), default = (0, 0, 0)
        Defines the center of the volume.
    pbc : array-like of bool, shape (3,), optional
        Periodic boundary flags for each spatial dimension ``(x, y, z)``.
        A value of ``True`` indicates that the corresponding boundary is
        treated as periodic. When periodicity is enabled along a given axis,
        points are considered automatically inside the constraint along that
        dimension, regardless of their coordinate. The default is
        ``(False, False, False)``, meaning no periodic boundaries.
    """

    def __init__(
        self, Lx, Ly=None, Lz=None, center=(0, 0, 0), pbc=(False, False, False)
    ):
        self.center = np.asarray(center)
        if Ly is None and Lz is None and Lx:
            Ly = Lz = Lx
        self.mins = self.center - np.array([Lx / 2, Ly / 2, Lz / 2])
        self.maxs = self.center + np.array([Lx / 2, Ly / 2, Lz / 2])
        self.box_lengths = np.array([Lx, Ly, Lz]).astype(np.float32)
        self.pbc = np.asarray(pbc, dtype=np.bool_)

    @classmethod
    def from_array(cls, box, center=(0, 0, 0), pbc=(False, False, False)):
        """Create a cuboid box from a 3D array."""
        return cls(box[0], box[1], box[2], center=center, pbc=pbc)

    def is_inside(self, points, buffer):
        """Check a set of coordinates against the volume constraint.

        Parameters
        ----------
        points : ndarray (N, 3), required
            The set of points to check against the volume constraint.
        buffer : float, required
            Buffer used for rounding

        Returns
        -------
        Mask of booleans of length N corresponding to each point.
        """
        return is_inside_cuboid(
            mins=self.mins,
            maxs=self.maxs,
            points=points,
            buffer=buffer,
            pbc=self.pbc,
        )

    def sample_candidates(self, points, n_candidates, buffer, k=10):
        """Generate candidate points uniformly distributed inside the box,
        optionally ranked by lowest local density around existing points.

        Parameters
        ----------
        points : ndarray, shape (N, 3) or None
            Existing points inside the volume. If provided, the candidate
            points will be sorted so that points in regions of lowest local
            density appear first.
            If None or empty, candidates are returned in random order.
        n_candidates : int
            Number of candidate points to sample uniformly inside the sphere.
        k : int, optional, default 10
            Number of neighbors to use for local density.
        buffer : float
            Edge buffer to subtract from the box edge, ensuring sampled
            points remain at least `buffer` distance away from the boundary.

        Returns
        -------
        candidates : ndarray, shape (n_candidates, 3)
            Candidate points inside the volume. If `points` is given,
            the array is sorted so that points with the greatest distance to
            the nearest neighbor (lowest local density) appear first.
        """
        # Create random candidates inside the box to test and sample from
        candidates = np.random.uniform(
            self.mins + buffer, self.maxs - buffer, size=(n_candidates, 3)
        )
        if points is None or len(points) == 0:
            return candidates
        # Existing points given, sort candidates by local density
        points = np.asarray(points)
        points = points[np.isfinite(points).all(axis=1)]  # Filter out np.inf values
        tree = cKDTree(points)
        dists, _ = tree.query(candidates, k=k)
        if dists.ndim == 1:
            density_metric = dists
        else:
            density_metric = dists[:, -1]
        sorted_order = np.argsort(-density_metric)  # negative = biggest first
        return candidates[sorted_order]


class SphereConstraint(Constraint):
    """Creates a spherical constraint.

    Parameters
    ----------
    radius : float, required
        The radius of the sphere
    center : array-like (1,3), default = (0, 0, 0)
        Defines the center point of the sphere.
    """

    def __init__(self, center, radius):
        self.center = np.array(center)
        self.radius = radius
        self.mins = self.center - self.radius
        self.maxs = self.center + self.radius

    def is_inside(self, points, buffer):
        """Check a set of coordinates against the volume constraint.

        Parameters
        ----------
        points : ndarray (N, 3), required
            The set of points to check against the volume constraint.
        buffer : float, required
            Buffer used for rounding

        Returns
        -------
        Mask of booleans of length N corresponding to each point.
        """
        return is_inside_sphere(points=points, sphere_radius=self.radius, buffer=buffer)

    def sample_candidates(self, points, n_candidates, buffer, k=10):
        """Generate candidate points uniformly distributed inside the sphere,
        optionally ranked by lowest local density around existing points.

        Parameters
        ----------
        points : ndarray, shape (N, 3) or None
            Existing points inside the volume. If provided, the candidate
            points will be sorted so that points in regions of lowest local
            density appear first.
            If None or empty, candidates are returned in random order.
        n_candidates : int
            Number of candidate points to sample uniformly inside the sphere.
        k : int, optional, default 10
            Number of neighbors to use for local density.
        buffer : float
            Radial buffer to subtract from the sphere radius, ensuring sampled
            points remain at least `buffer` distance away from the boundary.

        Returns
        -------
        candidates : ndarray, shape (n_candidates, 3)
            Candidate points inside the volume. If `points` is given,
            the array is sorted so that points with the greatest distance to
            the nearest neighbor (lowest local density) appear first.
        """
        effective_radius = self.radius - buffer
        dirs = np.random.normal(size=(n_candidates, 3))
        dirs /= np.linalg.norm(dirs, axis=1)[:, None]
        u = np.random.random(size=n_candidates)
        radii = effective_radius * (u ** (1 / 3))
        candidates = self.center + dirs * radii[:, None]
        # If just sampling from the volume, no KDTree needed
        if points is None or len(points) == 0:
            return candidates
        # Want to sample around existing points, sort by local density
        points = np.asarray(points)
        tree = cKDTree(points)
        dists, _ = tree.query(candidates, k=k)
        if dists.ndim == 1:
            density_metric = dists
        else:
            density_metric = dists[:, -1]
        sorted_order = np.argsort(-density_metric)  # negative = biggest first
        return candidates[sorted_order]


class CylinderConstraint(Constraint):
    """Creates a cylindrical constraint.

    Parameters
    ----------
    radius : float, required
        The radius of the cylinder
    height : float, required
        The height  of the cylinder
    center : array-like (1,3), default = (0, 0, 0)
        Defines the center point of the sphere.
    periodic_height : bool, default False
        If True, then treat the bounding along the Z-axis (height)
        as periodic.
    """

    def __init__(self, radius, height, center=(0, 0, 0), periodic_height=False):
        self.center = np.array(center)
        self.height = height
        self.radius = radius
        self.periodic_height = periodic_height
        self.mins = np.array(
            [
                self.center[0] - self.radius,
                self.center[1] - self.radius,
                self.center[2] - self.height / 2,
            ]
        )
        self.maxs = np.array(
            [
                self.center[0] + self.radius,
                self.center[1] + self.radius,
                self.center[2] + self.height / 2,
            ]
        )

    def is_inside(self, points, buffer):
        """Check a set of coordinates against the volume constraint.

        Parameters
        ----------
        points : ndarray (N, 3), required
            The set of points to check against the volume constraint.
        buffer : float, required
            Buffer used for rounding

        Returns
        -------
        Mask of booleans of length N corresponding to each point.
        """
        return is_inside_cylinder(
            points=points,
            center=self.center,
            cylinder_radius=self.radius,
            height=self.height,
            buffer=buffer,
            periodic=self.periodic_height,
        )

    def sample_candidates(self, points, n_candidates, buffer, k=10):
        """Generate candidate points uniformly distributed inside the cylinder,
        optionally ranked by lowest local density around existing points.

        Parameters
        ----------
        points : ndarray, shape (N, 3) or None
            Existing points inside the volume. If provided, the candidate
            points will be sorted so that points in regions of lowest local
            density appear first.
            If None or empty, candidates are returned in random order.
        n_candidates : int
            Number of candidate points to sample uniformly.
        k : int, optional, default 10
            Number of neighbors to use for local density.
        buffer : float
            Radial buffer to subtract from the radius and height, ensuring sampled
            points remain at least `buffer` distance away from the boundary.

        Returns
        -------
        candidates : ndarray, shape (n_candidates, 3)
            Candidate points inside the volume. If `points` is given,
            the array is sorted so that points with the greatest distance to
            the nearest neighbor (lowest local density) appear first.
        """
        eff_radius = max(self.radius - buffer, 0.0)
        eff_half_height = max(self.height * 0.5 - buffer, 0.0)
        z = self.center[2] + np.random.uniform(
            -eff_half_height, eff_half_height, size=n_candidates
        )
        r = eff_radius * np.sqrt(np.random.random(size=n_candidates))
        theta = 2 * np.pi * np.random.random(size=n_candidates)
        x = self.center[0] + r * np.cos(theta)
        y = self.center[1] + r * np.sin(theta)
        candidates = np.column_stack((x, y, z))
        if points is None or len(points) == 0:
            return candidates
        # Existing points given, rank by lowest local density
        points = np.asarray(points)
        tree = cKDTree(points)
        dists, _ = tree.query(candidates, k=k)
        if dists.ndim == 1:
            density_metric = dists
        else:
            density_metric = dists[:, -1]
        sorted_order = np.argsort(-density_metric)  # negative = biggest first
        return candidates[sorted_order]


@njit(cache=True, fastmath=True)
def is_inside_cylinder(points, center, cylinder_radius, height, buffer, periodic):
    n_points = points.shape[0]
    results = np.empty(n_points, dtype=np.bool_)
    max_r = cylinder_radius - buffer
    max_r_sq = max_r * max_r
    half_height = height / 2.0
    max_z = half_height - buffer
    for i in range(n_points):
        dx = points[i, 0] - center[0]
        dy = points[i, 1] - center[1]
        dz = points[i, 2] - center[2]
        r_sq = dx * dx + dy * dy
        inside_radial = r_sq <= max_r_sq
        if periodic:
            inside_z = True
        else:
            inside_z = abs(dz) <= max_z
        results[i] = inside_radial and inside_z
    return results


@njit(cache=True, fastmath=True)
def is_inside_sphere(sphere_radius, points, buffer):
    n_points = points.shape[0]
    results = np.empty(n_points, dtype=np.bool_)
    max_distance = sphere_radius - buffer
    max_distance_sq = max_distance * max_distance
    for i in range(n_points):
        dist_from_center_sq = 0.0
        for j in range(3):
            dist_from_center_sq += points[i, j] * points[i, j]
        results[i] = dist_from_center_sq < max_distance_sq
    return results


@njit(cache=True, fastmath=True)
def is_inside_cuboid(mins, maxs, points, buffer, pbc):
    n_points = points.shape[0]
    results = np.empty(n_points, dtype=np.bool_)
    for i in range(n_points):
        inside = True
        for j in range(3):
            if not pbc[j]:
                if points[i, j] - buffer < mins[j] or points[i, j] + buffer > maxs[j]:
                    inside = False
                    break
        results[i] = inside
    return results


class PipeConstraint(Constraint):
    """A hollow cylinder (pipe): polymer packs in the annular shell, leaving
    an open central channel running along the z-axis.

    A point is "inside" when it lies in the annulus between ``inner_radius``
    and ``outer_radius`` and within the cylinder's height. The central cylinder
    (radius < ``inner_radius``) is left empty -- the channel you might later
    fill with solvent or small molecules.

    Parameters
    ----------
    outer_radius : float, required
        Outer wall radius of the shell.
    inner_radius : float, required
        Radius of the empty central channel. Must be smaller than
        ``outer_radius``.
    height : float, required
        Extent of the cylinder along the z-axis.
    center : array-like (3,), default (0, 0, 0)
        Center of the cylinder.
    """

    def __init__(self, outer_radius, inner_radius, height, center=(0, 0, 0)):
        if inner_radius >= outer_radius:
            raise ValueError("inner_radius must be smaller than outer_radius.")
        self.outer_radius = float(outer_radius)
        self.inner_radius = float(inner_radius)
        self.height = float(height)
        self.center = np.asarray(center, dtype=float)
        # Axis-aligned bounding box, handy for any caller that wants it.
        self.mins = self.center - np.array(
            [self.outer_radius, self.outer_radius, self.height / 2.0]
        )
        self.maxs = self.center + np.array(
            [self.outer_radius, self.outer_radius, self.height / 2.0]
        )

    def is_inside(self, points, buffer):
        """Return a boolean mask: True where a point is inside the shell.

        Parameters
        ----------
        points : ndarray (N, 3), required
        buffer : float, required
            Keep points at least this far from every wall (inner, outer, caps).

        Returns
        -------
        ndarray of bool, shape (N,)
        """
        d = np.asarray(points, dtype=float) - self.center
        r = np.hypot(d[:, 0], d[:, 1])
        inside_radial = (r >= self.inner_radius + buffer) & (
            r <= self.outer_radius - buffer
        )
        inside_z = np.abs(d[:, 2]) <= (self.height / 2.0 - buffer)
        return inside_radial & inside_z

    def sample_candidates(self, points, n_candidates, buffer, k=10):
        """Sample candidate points inside the annular shell, low-density-first.

        Follows the same contract as the built-in constraints: if existing
        ``points`` are given, candidates are sorted so the most "open" spots
        (largest distance to their k-th nearest neighbor) come first.
        """
        r_in = self.inner_radius + buffer
        r_out = max(self.outer_radius - buffer, r_in)
        half_h = max(self.height / 2.0 - buffer, 0.0)

        theta = np.random.uniform(0, 2 * np.pi, size=n_candidates)
        # sqrt-sampling on the radius gives a uniform areal density in the annulus.
        u = np.random.random(size=n_candidates)
        r = np.sqrt(u * (r_out**2 - r_in**2) + r_in**2)
        z = np.random.uniform(-half_h, half_h, size=n_candidates)
        x = self.center[0] + r * np.cos(theta)
        y = self.center[1] + r * np.sin(theta)
        candidates = np.column_stack((x, y, self.center[2] + z))

        if points is None or len(points) == 0:
            return candidates
        points = np.asarray(points)
        points = points[np.isfinite(points).all(axis=1)]
        tree = cKDTree(points)
        dists, _ = tree.query(candidates, k=k)
        density_metric = dists if dists.ndim == 1 else dists[:, -1]
        return candidates[np.argsort(-density_metric)]  # most open first
