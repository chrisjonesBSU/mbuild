"""OpenMM energy evaluations used by the tuning stages."""

import numpy as np

from mbuild.simulation import OpenMMSimulation

RESTRAINT_GROUP = 1
RESTRAINT_K = 2.0e6


def add_centroid_restraints(sim, compound, targets, k=RESTRAINT_K):
    """Restrain each bead centroid to a target coordinate.

    The restraint occupies its own force group so its energy can be
    subtracted from the total.

    Parameters
    ----------
    sim : mbuild.simulation.OpenMMSimulation, required
        Simulation whose system the restraint is added to.
    compound : mbuild.Compound, required
        Backmapped compound whose children are the beads.
    targets : np.ndarray (N, 3), required
        Target centroid coordinate of each bead, in compound child order.
    k : float, default 2e6
        Restraint force constant in kJ/mol/nm^2.
    """
    import openmm

    force = openmm.CustomCentroidBondForce(1, "0.5*k*((x1-px)^2+(y1-py)^2+(z1-pz)^2)")
    for name in ("k", "px", "py", "pz"):
        force.addPerBondParameter(name)
    index_of = {p: i for i, p in enumerate(compound.particles())}
    for bead, target in zip(compound.children, targets):
        group = force.addGroup([index_of[p] for p in bead.particles()])
        force.addBond([group], [k, *target])
    force.setForceGroup(RESTRAINT_GROUP)
    sim.system.addForce(force)
    return force


def restrained_energy(
    compound,
    targets,
    forcefield=None,
    k=RESTRAINT_K,
    seed=1,
    n_steps=3000,
    tolerance=1.0,
):
    """Force field energy at the centroid restrained minimum, in kJ/mol.

    Minimizes with every bead centroid restrained to its target coordinate,
    then subtracts the restraint group energy so the returned value carries
    no restraint contribution.
    """
    import openmm.unit as u

    sim = OpenMMSimulation(compound, forcefield=forcefield, platform="CPU", seed=seed)
    add_centroid_restraints(sim, compound, targets, k=k)
    sim.minimize(n_steps=n_steps, tolerance=tolerance)
    context = sim.simulation.context
    total = context.getState(getEnergy=True).getPotentialEnergy() / u.kilojoule_per_mole
    restraint = (
        context.getState(getEnergy=True, groups={RESTRAINT_GROUP}).getPotentialEnergy()
        / u.kilojoule_per_mole
    )
    return total - restraint


class PairEnergy:
    """Rigid energy of two fragment copies at a fixed center to center separation.

    Builds the OpenMM system once and only replaces coordinates, so each
    evaluation is a single energy call with no minimization. Both fragments
    are held at their input geometry.

    Parameters
    ----------
    fragment : mbuild.Compound, required
        The fragment standing in for one coarse grained bead.
    forcefield : str or None, default None
        Passed to OpenMMSimulation. None uses UFF.
    seed : int, default 1
        Seed for the coordinate kick.
    """

    def __init__(self, fragment, forcefield=None, seed=1):
        import openmm.unit as u
        from openmm.openmm import LangevinIntegrator

        import mbuild as mb

        first, second = mb.clone(fragment), mb.clone(fragment)
        system = mb.Compound()
        system.add(first)
        system.add(second)
        self.n_atoms = first.n_particles
        sim = OpenMMSimulation(
            system, forcefield=forcefield, platform="CPU", seed=seed, kick=False
        )
        integrator = LangevinIntegrator(
            298 * u.kelvin, 1.0 / u.picosecond, 0.002 * u.picoseconds
        )
        sim._create_simulation(integrator)
        self._context = sim.simulation.context
        xyz = system.xyz.copy()
        self.xyz_a = xyz[: self.n_atoms] - xyz[: self.n_atoms].mean(axis=0)
        self.xyz_b = xyz[self.n_atoms :] - xyz[self.n_atoms :].mean(axis=0)

    def energy(self, xyz):
        """Potential energy of the pair at the given coordinates, in kJ/mol."""
        import openmm.unit as u

        self._context.setPositions(xyz * u.nanometer)
        return (
            self._context.getState(getEnergy=True).getPotentialEnergy()
            / u.kilojoule_per_mole
        )

    def scan(self, separations, n_orientations=300, seed=0):
        """Energy at every separation for a set of random relative orientations.

        Orientations are drawn with ``scipy.spatial.transform.Rotation.random``,
        which samples SO(3) uniformly. Uniformly sampled Euler angles are not
        uniform rotations and would bias the average.

        Returns
        -------
        np.ndarray (n_orientations, len(separations))
            Energies in kJ/mol.
        """
        from scipy.spatial.transform import Rotation

        separations = np.asarray(separations, dtype=float)
        rotations = Rotation.random(2 * n_orientations, random_state=seed)
        rot_a, rot_b = rotations[:n_orientations], rotations[n_orientations:]
        energies = np.empty((n_orientations, separations.size))
        for i in range(n_orientations):
            xyz_a = rot_a[i].apply(self.xyz_a)
            xyz_b = rot_b[i].apply(self.xyz_b)
            for j, separation in enumerate(separations):
                offset = np.array([separation, 0.0, 0.0])
                energies[i, j] = self.energy(np.vstack([xyz_a, xyz_b + offset]))
        return energies
