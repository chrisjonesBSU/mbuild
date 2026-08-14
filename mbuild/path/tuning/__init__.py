"""Tune hard_sphere_random_walk parameters to a target chemistry.

Measures the bond length, excluded volume radius, and correlated angle and
dihedral distribution a chemistry wants, so large multi-chain morphologies
inherit chemistry specific local structure instead of hard sphere defaults.

>>> from mbuild.path.tuning import tune
>>> result = tune("{#A=[>]CC[<]}")  # doctest: +SKIP
>>> path = hard_sphere_random_walk(
...     **result.as_walk_kwargs(), termination=100
... )  # doctest: +SKIP
"""

# ruff: noqa: F401
from .energy import PairEnergy, add_centroid_restraints, restrained_energy
from .geometry import (
    angle_dihedral_pairs,
    bond_lengths,
    interior_angle_dihedral_pair,
    internals,
)
from .plotting import (
    plot_angle_energy,
    plot_dihedral_energy,
    plot_pair_energy,
)
from .stages import (
    FreeEnergyAccumulator,
    barker_henderson,
    effective_sample_size,
    energy_table_from_samples,
    free_energy_table,
    marginal_free_energy,
    natural_bond_length,
    orientation_averaged_pmf,
    pair_potential,
    phi_grid,
    relaxed_bond_length,
    replica_spread,
    sample_thermal,
    sample_walks,
    thermal_internals,
    theta_grid,
)
from .tuner import Chemistry, Tuner, TunerResult, tune

__all__ = [
    "Chemistry",
    "FreeEnergyAccumulator",
    "PairEnergy",
    "Tuner",
    "TunerResult",
    "add_centroid_restraints",
    "angle_dihedral_pairs",
    "barker_henderson",
    "bond_lengths",
    "effective_sample_size",
    "energy_table_from_samples",
    "free_energy_table",
    "internals",
    "interior_angle_dihedral_pair",
    "marginal_free_energy",
    "natural_bond_length",
    "orientation_averaged_pmf",
    "pair_potential",
    "phi_grid",
    "plot_angle_energy",
    "plot_dihedral_energy",
    "plot_pair_energy",
    "relaxed_bond_length",
    "replica_spread",
    "restrained_energy",
    "sample_thermal",
    "sample_walks",
    "theta_grid",
    "thermal_internals",
    "tune",
]
