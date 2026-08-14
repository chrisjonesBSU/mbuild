"""Internal coordinates measured off a coarse grained path."""

import numpy as np


def bond_lengths(coordinates):
    """Distances between consecutive sites of a linear chain."""
    return np.linalg.norm(np.diff(coordinates, axis=0), axis=1)


def internals(coordinates):
    """Bond lengths, bending angles and signed dihedrals of a linear chain.

    Parameters
    ----------
    coordinates : np.ndarray (N, 3), required
        Site coordinates in path order.

    Returns
    -------
    bonds, angles, dihedrals : np.ndarray
        Arrays of length N-1, N-2 and N-3. Angles and dihedrals are in
        radians, with dihedrals signed over [-pi, pi].
    """
    vectors = np.diff(coordinates, axis=0)
    bonds = np.linalg.norm(vectors, axis=1)
    units = vectors / bonds[:, None]
    angles = np.arccos(np.clip(-(units[:-1] * units[1:]).sum(axis=1), -1.0, 1.0))
    normals = np.cross(units[:-1], units[1:])
    normals /= np.clip(np.linalg.norm(normals, axis=1), 1e-9, None)[:, None]
    cross = np.cross(normals[:-1], normals[1:])
    dihedrals = np.arctan2(
        (cross * units[1:-1]).sum(axis=1), (normals[:-1] * normals[1:]).sum(axis=1)
    )
    return bonds, angles, dihedrals


def interior_angle_dihedral_pair(angles, dihedrals):
    """Return the (theta, phi) pair at the center of a chain.

    Pairs the middle dihedral with the mean of the two bending angles
    flanking it, leaving any outer sites as context.

    At 6 sites or more this keeps terminal caps off the measured beads,
    worth about 6 degrees of bending angle for polyethylene under UFF.
    Realizing that gain needs an energy restricted to the measured beads,
    since a whole chain energy is dominated by the coordinates this
    discards. At 4 sites the measured beads still carry caps and only the
    attribution differs.

    Parameters
    ----------
    angles : np.ndarray (N-2,), required
        Bending angles in path order.
    dihedrals : np.ndarray (N-3,), required
        Signed dihedrals in path order.

    Returns
    -------
    np.ndarray (1, 2)
        Columns are (theta, phi) in radians. Empty for chains under 4 sites.
    """
    if len(dihedrals) == 0:
        return np.empty((0, 2))
    middle = (len(dihedrals) - 1) // 2
    theta = 0.5 * (angles[middle] + angles[middle + 1])
    return np.array([[theta, dihedrals[middle]]])


def angle_dihedral_pairs(angles, dihedrals):
    """Pair each dihedral with the two bending angles that flank it.

    Attributes one energy to every coordinate set a chain contains. Spreads
    a single energy over coordinates of differing quality, which biases the
    bending angle about 2 degrees wide of
    ``interior_angle_dihedral_pair``.

    Parameters
    ----------
    angles : np.ndarray (N-2,), required
        Bending angles in path order.
    dihedrals : np.ndarray (N-3,), required
        Signed dihedrals in path order.

    Returns
    -------
    np.ndarray (2 * len(dihedrals), 2)
        Columns are (theta, phi) in radians.
    """
    if len(dihedrals) == 0:
        return np.empty((0, 2))
    theta = np.concatenate([angles[:-1], angles[1:]])
    phi = np.concatenate([dihedrals, dihedrals])
    return np.column_stack([theta, phi])
