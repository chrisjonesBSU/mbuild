"""Simple random walk algorithm for generating polymer chains."""

import math

import numpy as np


def lamellae(num_layers, layer_separation, layer_length, bond_L):
    """Generate monomer coordinates of a lamellar structure.

    Parameters
    ----------
    num_layers : int, required
        The number of parallel layers in the structure.
    layer_separation : float, (nm), required.
        The distance, in nanometers, between parallel layers.
    layer_length : float, (nm), required.
        The length, in nanometers, of each layer.
    bond_L : float, (nm), required.
        The monomer-monomer bond length of the backbone.
    """
    layer_spacing = np.arange(0, layer_length, bond_L)
    # Info for generating coords of the curves between layers
    r = layer_separation / 2
    arc_length = r * np.pi
    arc_num_points = math.floor(arc_length / bond_L)
    arc_angle = np.pi / (arc_num_points + 1)  # incremental angle
    arc_angles = np.linspace(arc_angle, np.pi, arc_num_points, endpoint=False)
    coordinates = []
    for i in range(num_layers):
        if i % 2 == 0:  # Even layer; build from left to right
            layer = [np.array([layer_separation * i, y, 0]) for y in layer_spacing]
            # Mid-point between this and next layer; use to get curve coords.
            origin = layer[-1] + np.array([r, 0, 0])
            arc = [
                origin + np.array([-np.cos(theta), np.sin(theta), 0]) * r
                for theta in arc_angles
            ]
        else:  # Odd layer; build from right to left
            layer = [
                np.array([layer_separation * i, y, 0]) for y in layer_spacing[::-1]
            ]
            # Mid-point between this and next layer; use to get curve coords.
            origin = layer[-1] + np.array([r, 0, 0])
            arc = [
                origin + np.array([-np.cos(theta), -np.sin(theta), 0]) * r
                for theta in arc_angles
            ]
        if i != num_layers - 1:
            coordinates.extend(layer + arc)
        else:
            coordinates.extend(layer)
    return coordinates


def random_walk(N, bond_L, radius, min_angle, max_angle, max_tries=1000, seed=24):
    """Generate chain coordinates resulting from a simple self-avoiding random walk.

    Parameters
    ----------
    N : int, required
        The number of particles in the random walk.
    bond_L : float, nm, required
        The fixed bond distance between consecutive sites.
    radius : float, nm, required
        Defines the monomer radius used when checking for overlapping sites.
    min_angle : float, radians, required
        The minimum allowed angle between 3 consecutive sites.
    max_angle : float, radians, required
        The maximum allowed angle between 3 consecutive sites.
    max_tries : int, default 1000
        The maximum number of attemps to complete the random walk.
    seed : int, default 24
        Random seed used during random walk.

    Returns
    -------
    coordinates : np.ndarray, shape=(N, 3)
        Final set of coordinates from random walk.

    """
    np.random.seed(seed)
    # First coord is always [0,0,0], next pos is always accepted.
    coordinates = np.zeros((N, 3))
    coordinates[1] = _next_coordinate(pos1=coordinates[0], bond_L=bond_L)
    tries = 0
    count = 1  # Start at 1; we already have 2 accepted moves
    while count < N - 1:
        new_xyz = _next_coordinate(
            pos1=coordinates[count],
            pos2=coordinates[count - 1],
            min_angle=min_angle,
            max_angle=max_angle,
            bond_L=bond_L,
        )
        coordinates[count + 1] = new_xyz

        if _check_system(
            system_coordinates=coordinates, radius=radius, count=count + 1
        ):
            count += 1
            tries += 1
        else:  # Next step failed. Set next coordinate back to (0,0,0).
            coordinates[count + 1] = np.zeros(3)
            tries += 1
        if tries == max_tries and count < N:
            raise RuntimeError(
                "The maximum number attempts allowed have passed, and only ",
                f"{count} sucsessful attempts were completed.",
                "Try changing the parameters and running again.",
            )

    return coordinates


def _next_coordinate(bond_L, pos1, pos2=None, min_angle=None, max_angle=None):
    if pos2 is None:
        phi = np.random.uniform(0, 2 * np.pi)
        theta = np.random.uniform(0, np.pi)
        next_pos = np.array(
            [
                bond_L * np.sin(theta) * np.cos(phi),
                bond_L * np.sin(theta) * np.sin(phi),
                bond_L * np.cos(theta),
            ]
        )
    else:  # Get the last bond vector, use angle range with last 2 coords.
        v1 = pos2 - pos1
        v1_norm = v1 / np.linalg.norm(v1)
        theta = np.random.uniform(min_angle, max_angle)
        r = np.random.rand(3) - 0.5  # Random vector
        r_perp = r - np.dot(r, v1_norm) * v1_norm
        r_perp_norm = r_perp / np.linalg.norm(r_perp)
        v2 = np.cos(theta) * v1_norm + np.sin(theta) * r_perp_norm
        next_pos = v2 * bond_L

    return pos1 + next_pos


def _check_system(system_coordinates, radius, count):
    # Count is the last particle, iterate through all others
    # Skip the particle bonded to the current one
    current_xyz = system_coordinates[count]
    for xyz in system_coordinates[: count - 1]:
        d = np.linalg.norm(xyz - current_xyz)
        if d < radius:
            return False
    return True
