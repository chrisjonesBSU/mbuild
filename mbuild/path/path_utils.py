"""Utility functions (mostly numba) for mbuild path generation.
These methods are primarly used by other functions and classes in mbuild.Path and
should rarely need to be called directly by users.
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def random_coordinate(
    pos1,
    pos2,
    bond_length,
    thetas,
    r_vectors,
):
    """Default next_step method for HardSphereRandomWalk.
    This method takes in a a batch of thetas and vectors
    and creates a batch of random coordinates.

    Parameters
    ----------
    pos1 : np.ndarray (1,3), required
        The coordinate of the last accepted site.
    pos2 : np.ndarray (1,3), required
        The coordinate of the second to last accepted site.
    bond_length : float, required
        The fixed bond length between pos1 and the new coordinates.
    thetas : array-like (N, 1), required
        A set of possible angles used to determine new
        coordinates relative to pos2-pos1-new angle
    r_vectors : array-like (N, 3), required
        A set of normal vectors used perform rotations around.
    """

    if pos1 is None:  # pick random point in sphere.
        r_norm = compute_norms(r_vectors)
        return (pos2 + r_vectors / r_norm * bond_length).astype(np.float32)
    # pos1 and pos2 are defined, use available angles to sample new coordinates
    v1 = pos2 - pos1
    v1_norm = v1 / norm(v1)
    dot_products = (r_vectors * v1_norm).sum(axis=1)
    r_perp = r_vectors - dot_products[:, None] * v1_norm
    norms = np.sqrt((r_perp * r_perp).sum(axis=1))
    # Handle rare cases where rprep vectors approach zero
    norms = np.where(norms < 1e-6, 1.0, norms)
    r_perp_norm = r_perp / norms[:, None]
    # Batch of trial next-step vectors using angles and r_norms
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    v2s = cos_thetas[:, None] * v1_norm + sin_thetas[:, None] * r_perp_norm
    # Batch of trial positions
    next_positions = pos1 + v2s * bond_length
    return next_positions.astype(np.float32)


@njit(cache=True, fastmath=True)
def check_path(existing_points, new_point, radius, tolerance):
    """Default check path method for HardSphereRandomWalk.

    Parameters
    ----------
    existing_points : np.ndarray (N, 3), required
        Array of all fixed points in the system.
        This includes accepted sites from previous random walk steps
        and coordinates from included compounds.
    new_point : np.ndarray (1,3), required
        A candidate point for the next step.
    radius : float, required
        The radius used for hard-sphere overlap checks.
    tolerance : float, required
        Tolerance in center-to-center distances, allowing for rounding errors.
    """
    if existing_points is None or existing_points.size == 0:
        return True
    min_sq_dist = (radius - tolerance) ** 2
    for i in range(existing_points.shape[0]):
        dist_sq = 0.0
        for j in range(existing_points.shape[1]):
            diff = existing_points[i, j] - new_point[j]
            dist_sq += diff * diff
        if dist_sq < min_sq_dist:
            return False
    return True


@njit(cache=True, fastmath=True)
def target_sq_distances(
    target_coordinate,
    new_points,
    pbc=np.array([False, False, False], dtype=bool),
    box_lengths=np.array([np.inf, np.inf, np.inf], dtype=np.float32),
):
    """Return squared distances from target_coordinate to new_points."""
    n_points = new_points.shape[0]
    sq_distances = np.empty(n_points, dtype=np.float32)
    for i in range(n_points):
        dx = target_coordinate[0] - new_points[i, 0]
        dy = target_coordinate[1] - new_points[i, 1]
        dz = target_coordinate[2] - new_points[i, 2]
        # Apply PBC per-axis
        if pbc[0]:
            dx -= np.round(dx / box_lengths[0]) * box_lengths[0]
        if pbc[1]:
            dy -= np.round(dy / box_lengths[1]) * box_lengths[1]
        if pbc[2]:
            dz -= np.round(dz / box_lengths[2]) * box_lengths[2]
        sq_distances[i] = dx * dx + dy * dy + dz * dz
    return sq_distances


@njit(cache=True, fastmath=True)
def local_density(candidate, target_coords, r_cut):
    """Return number of target-type sites within r_cut of candidate."""
    r2_cut = r_cut * r_cut
    density = 0
    for i in range(target_coords.shape[0]):
        dx = candidate[0] - target_coords[i, 0]
        dy = candidate[1] - target_coords[i, 1]
        dz = candidate[2] - target_coords[i, 2]
        dist2 = dx * dx + dy * dy + dz * dz
        if dist2 < r2_cut:
            density += 1
    return density


@njit(cache=True, fastmath=True)
def target_density(candidates, target_coords, r_cut):
    """For a batch of candidate sites, calculate local density of target site-types."""
    n = candidates.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = local_density(candidates[i], target_coords, r_cut)
    return out


@njit(cache=True, fastmath=True)
def norm(vec):
    """Use in place of np.linalg.norm inside of numba functions."""
    s = 0.0
    for i in range(vec.shape[0]):
        s += vec[i] * vec[i]
    return np.sqrt(s)


@njit
def compute_norms(vec):
    """Compute norms for multiple vectors."""
    n = vec.shape[0]
    r_norm = np.zeros((n, 1))
    for i in range(n):
        r_norm[i, 0] = norm(vec[i])  # Call your norm function
    return r_norm


@njit(cache=True, fastmath=True)
def rotate_vector(v, axis, theta):
    """Rotate vector v around a normalized axis by angle theta using Rodrigues' formula."""
    c = np.cos(theta)
    s = np.sin(theta)
    k = axis
    k_dot_v = k[0] * v[0] + k[1] * v[1] + k[2] * v[2]
    cross = np.zeros(3)
    cross[0] = k[1] * v[2] - k[2] * v[1]
    cross[1] = k[2] * v[0] - k[0] * v[2]
    cross[2] = k[0] * v[1] - k[1] * v[0]
    rotated = np.zeros(3)
    for i in range(3):
        rotated[i] = v[i] * c + cross[i] * s + k[i] * k_dot_v * (1 - c)
    return rotated


@njit(cache=True, fastmath=True)
def calculate_sq_distances(
    target_coordinate,
    new_points,
    pbc=np.array([False, False, False], dtype=bool),
    box_lengths=np.array([np.inf, np.inf, np.inf], dtype=np.float32),
):
    """
    Return squared distances from target_coordinate to new_points.

    Parameters
    ----------
    target_coordinate : np.ndarray
        Single coordinate [x, y, z]
    new_points : np.ndarray
        Array of coordinates with shape (n, 3)
    pbc : np.ndarray
        Boolean array indicating periodic boundary conditions for each axis
    box_lengths : np.ndarray
        Box lengths for periodic boundary conditions

    Returns
    -------
    np.ndarray
        Squared distances from target to each point in new_points
    """
    n_points = new_points.shape[0]
    sq_distances = np.empty(n_points, dtype=np.float32)
    for i in range(n_points):
        dx = target_coordinate[0] - new_points[i, 0]
        dy = target_coordinate[1] - new_points[i, 1]
        dz = target_coordinate[2] - new_points[i, 2]
        # Apply PBC per-axis
        if pbc[0]:
            dx -= np.round(dx / box_lengths[0]) * box_lengths[0]
        if pbc[1]:
            dy -= np.round(dy / box_lengths[1]) * box_lengths[1]
        if pbc[2]:
            dz -= np.round(dz / box_lengths[2]) * box_lengths[2]
        sq_distances[i] = dx * dx + dy * dy + dz * dz
    return sq_distances


@njit(cache=True, fastmath=True)
def find_candidates_within_radius(
    target_coordinate,
    candidate_points,
    radius,
    pbc=np.array([False, False, False], dtype=bool),
    box_lengths=np.array([np.inf, np.inf, np.inf], dtype=np.float32),
):
    """
    Find indices of candidate points within radius of target coordinate.

    Parameters
    ----------
    target_coordinate : np.ndarray
        Single coordinate [x, y, z]
    candidate_points : np.ndarray
        Array of coordinates with shape (n, 3)
    radius : float
        Search radius
    pbc : np.ndarray
        Boolean array indicating periodic boundary conditions for each axis
    box_lengths : np.ndarray
        Box lengths for periodic boundary conditions

    Returns
    -------
    np.ndarray
        Boolean mask of points within radius
    """
    n_points = candidate_points.shape[0]
    r2_cut = radius * radius
    within_radius = np.empty(n_points, dtype=np.bool_)

    for i in range(n_points):
        dx = target_coordinate[0] - candidate_points[i, 0]
        dy = target_coordinate[1] - candidate_points[i, 1]
        dz = target_coordinate[2] - candidate_points[i, 2]
        # Apply PBC per-axis
        if pbc[0]:
            dx -= np.round(dx / box_lengths[0]) * box_lengths[0]
        if pbc[1]:
            dy -= np.round(dy / box_lengths[1]) * box_lengths[1]
        if pbc[2]:
            dz -= np.round(dz / box_lengths[2]) * box_lengths[2]

        dist2 = dx * dx + dy * dy + dz * dz
        within_radius[i] = dist2 <= r2_cut

    return within_radius


@njit(cache=True, fastmath=True)
def _sym_eig_3x3(a00, a01, a02, a11, a12, a22):
    """Eigen-decomposition of a symmetric 3x3 matrix via cyclic Jacobi rotations.

    Returns eigenvalues sorted ascending together with the matching eigenvectors
    as the *columns* of V. Self-contained (no LAPACK), so it stays numba
    nopython friendly. A handful of sweeps is plenty for a 3x3.

    Parameters
    ----------
    a00, a01, a02, a11, a12, a22 : float
        The six unique entries of the symmetric matrix.

    Returns
    -------
    ev : np.ndarray (3,) float64
        Eigenvalues, ascending.
    V : np.ndarray (3, 3) float64
        Eigenvectors; column k matches ev[k].
    """
    A = np.empty((3, 3))
    A[0, 0] = a00
    A[0, 1] = a01
    A[0, 2] = a02
    A[1, 0] = a01
    A[1, 1] = a11
    A[1, 2] = a12
    A[2, 0] = a02
    A[2, 1] = a12
    A[2, 2] = a22
    V = np.zeros((3, 3))
    V[0, 0] = 1.0
    V[1, 1] = 1.0
    V[2, 2] = 1.0
    for _ in range(50):
        off = abs(A[0, 1]) + abs(A[0, 2]) + abs(A[1, 2])
        if off < 1e-14:
            break
        for p in range(2):
            for q in range(p + 1, 3):
                apq = A[p, q]
                if abs(apq) < 1e-30:
                    continue
                # Jacobi rotation angle that zeros out A[p, q].
                theta = (A[q, q] - A[p, p]) / (2.0 * apq)
                t = 1.0 / (abs(theta) + np.sqrt(theta * theta + 1.0))
                if theta < 0.0:
                    t = -t
                c = 1.0 / np.sqrt(t * t + 1.0)
                s = t * c
                # A <- J^T A J: rotate columns p, q then rows p, q.
                for k in range(3):
                    akp = A[k, p]
                    akq = A[k, q]
                    A[k, p] = c * akp - s * akq
                    A[k, q] = s * akp + c * akq
                for k in range(3):
                    apk = A[p, k]
                    aqk = A[q, k]
                    A[p, k] = c * apk - s * aqk
                    A[q, k] = s * apk + c * aqk
                # Accumulate the rotation into the eigenvector matrix.
                for k in range(3):
                    vkp = V[k, p]
                    vkq = V[k, q]
                    V[k, p] = c * vkp - s * vkq
                    V[k, q] = s * vkp + c * vkq
    ev = np.empty(3)
    ev[0] = A[0, 0]
    ev[1] = A[1, 1]
    ev[2] = A[2, 2]
    # Sort ascending and carry the eigenvector columns along.
    order = np.argsort(ev)
    ev_sorted = np.empty(3)
    V_sorted = np.empty((3, 3))
    for new_i in range(3):
        src = order[new_i]
        ev_sorted[new_i] = ev[src]
        for r in range(3):
            V_sorted[r, new_i] = V[r, src]
    return ev_sorted, V_sorted


@njit(cache=True, fastmath=True)
def nematic_q_director(bonds, coordinates, center, r_cut):
    """Local nematic order parameter and director around a point.

    Gathers all bonds whose midpoint lies within ``r_cut`` of ``center``, builds
    the nematic order tensor Q = <(3/2) u_outer_u - (1/2) I> over their unit
    vectors, and returns its scalar order S (the largest eigenvalue) and the
    director (its eigenvector). Because Q is built from u (x) u, parallel and
    antiparallel bonds contribute identically -- the correct symmetry for
    nematic/lamellar order. The dominant eigenpair is taken from a full
    symmetric eigendecomposition (``_sym_eig_3x3``), which avoids the
    largest-|eigenvalue| sign trap of power iteration on a traceless tensor.

    Parameters
    ----------
    bonds : np.ndarray (E, 2) int
        Index pairs into ``coordinates`` defining bonded sites.
    coordinates : np.ndarray (M, 3) float32
        All placed site coordinates.
    center : np.ndarray (3,) float32
        Point about which locality is measured (the current chain tip).
    r_cut : float
        Locality cutoff; a bond counts if its midpoint is within r_cut of center.

    Returns
    -------
    n_local : int
        Number of bonds found in the neighborhood.
    s_param : float
        Scalar nematic order parameter S (largest eigenvalue; 0 if no bonds).
    director : np.ndarray (3,) float32
        Unit director (dominant alignment axis); zeros if undefined.
    """
    r2 = r_cut * r_cut
    qxx = 0.0
    qxy = 0.0
    qxz = 0.0
    qyy = 0.0
    qyz = 0.0
    qzz = 0.0
    n_local = 0
    for e in range(bonds.shape[0]):
        i = bonds[e, 0]
        j = bonds[e, 1]
        # Bond midpoint
        mx = 0.5 * (coordinates[i, 0] + coordinates[j, 0])
        my = 0.5 * (coordinates[i, 1] + coordinates[j, 1])
        mz = 0.5 * (coordinates[i, 2] + coordinates[j, 2])
        dx = mx - center[0]
        dy = my - center[1]
        dz = mz - center[2]
        if dx * dx + dy * dy + dz * dz > r2:
            continue
        # Unit bond vector
        ux = coordinates[j, 0] - coordinates[i, 0]
        uy = coordinates[j, 1] - coordinates[i, 1]
        uz = coordinates[j, 2] - coordinates[i, 2]
        un = np.sqrt(ux * ux + uy * uy + uz * uz)
        if un < 1e-12:
            continue
        ux /= un
        uy /= un
        uz /= un
        # Accumulate (3/2) u_outer_u - (1/2) I
        qxx += 1.5 * ux * ux - 0.5
        qyy += 1.5 * uy * uy - 0.5
        qzz += 1.5 * uz * uz - 0.5
        qxy += 1.5 * ux * uy
        qxz += 1.5 * ux * uz
        qyz += 1.5 * uy * uz
        n_local += 1

    director = np.zeros(3, dtype=np.float32)
    if n_local < 1:
        return n_local, 0.0, director

    inv = 1.0 / n_local
    qxx *= inv
    qxy *= inv
    qxz *= inv
    qyy *= inv
    qyz *= inv
    qzz *= inv

    # Dominant eigenpair from a full symmetric eigendecomposition (eigenvalues
    # ascending). The largest eigenvalue is the uniaxial order S; its eigenvector
    # is the director.
    ev, V = _sym_eig_3x3(qxx, qxy, qxz, qyy, qyz, qzz)
    s_param = float(ev[2])
    director[0] = np.float32(V[0, 2])
    director[1] = np.float32(V[1, 2])
    director[2] = np.float32(V[2, 2])
    return n_local, s_param, director


@njit(cache=True, fastmath=True)
def order_alignment_scores(candidates, tip, director, s_param, target_s):
    """Score candidates by how their trial bond aligns with the local director.

    For each candidate the trial bond is u = (candidate - tip), normalized. The
    nematic alignment with the director is (u . director)**2 (squared so parallel
    and antiparallel score equally). It is scaled by a self-limiting factor
    max(0, 1 - S / target_s): once the neighborhood reaches the target order the
    reward vanishes and the walk reverts to noise (amorphous defects).

    Parameters
    ----------
    candidates : np.ndarray (N, 3) float32
    tip : np.ndarray (3,) float32
        Current chain tip; trial bonds originate here.
    director : np.ndarray (3,) float32
        Local director from nematic_q_director.
    s_param : float
        Local scalar order parameter S.
    target_s : float
        Target order S* governing the self-limiting penalty.

    Returns
    -------
    np.ndarray (N,) float32
        Per-candidate order-increase scores.
    """
    n = candidates.shape[0]
    out = np.empty(n, dtype=np.float32)
    limit = 1.0 - s_param / target_s
    if limit < 0.0:
        limit = 0.0
    for k in range(n):
        dx = candidates[k, 0] - tip[0]
        dy = candidates[k, 1] - tip[1]
        dz = candidates[k, 2] - tip[2]
        dn = np.sqrt(dx * dx + dy * dy + dz * dz)
        if dn < 1e-12:
            out[k] = 0.0
            continue
        dot = (dx * director[0] + dy * director[1] + dz * director[2]) / dn
        out[k] = dot * dot * limit
    return out
