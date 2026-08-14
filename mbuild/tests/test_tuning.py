import numpy as np
import pytest

from mbuild.path.points import AngleDihedralSampler
from mbuild.path.tuning import (
    Chemistry,
    FreeEnergyAccumulator,
    Tuner,
    TunerResult,
    angle_dihedral_pairs,
    barker_henderson,
    bond_lengths,
    effective_sample_size,
    free_energy_table,
    internals,
    interior_angle_dihedral_pair,
    marginal_free_energy,
    phi_grid,
    theta_grid,
)
from mbuild.path.tuning.tuner import _fragment_names, _fragment_smiles
from mbuild.tests.base_test import BaseTest


def four_bead(bond, theta, phi):
    """Four beads at a known bond length, bending angle and dihedral."""
    p0 = np.zeros(3)
    p1 = p0 + np.array([bond, 0.0, 0.0])
    p2 = p1 + bond * np.array([-np.cos(theta), np.sin(theta), 0.0])
    b1, b2 = p1 - p0, p2 - p1
    n = np.cross(b1, b2)
    n /= np.linalg.norm(n)
    d2 = b2 / np.linalg.norm(b2)
    m = np.cross(n, d2)
    p3 = p2 + bond * (
        -np.cos(theta) * d2 + np.sin(theta) * (np.cos(phi) * m + np.sin(phi) * n)
    )
    return np.array([p0, p1, p2, p3])


class TestGeometry(BaseTest):
    @pytest.mark.parametrize("theta_deg, phi_deg", [(110, 60), (95, 180), (140, -75)])
    def test_internals_round_trip(self, theta_deg, phi_deg):
        coords = four_bead(0.285, np.radians(theta_deg), np.radians(phi_deg))
        bonds, angles, dihedrals = internals(coords)
        assert np.allclose(bonds, 0.285)
        assert np.allclose(np.degrees(angles), theta_deg)
        assert np.allclose(np.degrees(dihedrals), phi_deg)

    def test_bond_lengths(self):
        coords = np.array([[0.0, 0, 0], [0.3, 0, 0], [0.3, 0.4, 0]])
        assert np.allclose(bond_lengths(coords), [0.3, 0.4])

    def test_angle_dihedral_pairs_flanking(self):
        angles = np.array([1.0, 2.0, 3.0])
        dihedrals = np.array([10.0, 20.0])
        pairs = angle_dihedral_pairs(angles, dihedrals)
        # Each dihedral pairs with the angle before and the angle after it
        assert pairs.shape == (4, 2)
        assert sorted(map(tuple, pairs)) == [
            (1.0, 10.0),
            (2.0, 10.0),
            (2.0, 20.0),
            (3.0, 20.0),
        ]

    def test_angle_dihedral_pairs_too_short(self):
        assert angle_dihedral_pairs(np.array([1.0]), np.array([])).shape == (0, 2)

    def test_interior_pair_of_a_6mer_is_the_middle(self):
        # 6 sites give 4 angles and 3 dihedrals; the middle dihedral is
        # index 1, flanked by angles 1 and 2
        angles = np.array([1.0, 2.0, 4.0, 8.0])
        dihedrals = np.array([10.0, 20.0, 30.0])
        pair = interior_angle_dihedral_pair(angles, dihedrals)
        assert pair.shape == (1, 2)
        assert pair[0, 0] == pytest.approx(3.0)
        assert pair[0, 1] == pytest.approx(20.0)

    def test_interior_pair_of_a_4mer_uses_the_only_dihedral(self):
        pair = interior_angle_dihedral_pair(np.array([1.0, 3.0]), np.array([10.0]))
        assert pair[0, 0] == pytest.approx(2.0)
        assert pair[0, 1] == pytest.approx(10.0)

    def test_interior_pair_too_short(self):
        assert interior_angle_dihedral_pair(
            np.array([1.0]), np.array([])
        ).shape == (0, 2)

    def test_interior_pair_tracks_a_real_chain(self):
        # Measured off real coordinates rather than synthetic index arrays
        coords = four_bead(0.285, np.radians(110), np.radians(60))
        _, angles, dihedrals = internals(coords)
        pair = interior_angle_dihedral_pair(angles, dihedrals)
        assert np.degrees(pair[0, 0]) == pytest.approx(110.0, abs=1e-3)
        assert np.degrees(pair[0, 1]) == pytest.approx(60.0, abs=1e-3)


class TestGrids(BaseTest):
    def test_theta_grid_clears_the_critical_angle(self):
        bond_length, radius = 0.285, 0.392
        grid = theta_grid(bond_length, radius, 12)
        critical = 2.0 * np.arcsin(radius / (2.0 * bond_length))
        assert grid.size == 12
        assert grid[0] > critical
        assert grid[-1] < np.radians(175.0)

    def test_theta_grid_no_room(self):
        with pytest.raises(ValueError):
            theta_grid(0.1, 0.25, 12)

    def test_phi_grid_spans_the_period(self):
        grid = phi_grid(12)
        assert grid.size == 12
        assert grid[0] > -np.pi and grid[-1] < np.pi
        assert np.allclose(np.diff(grid), 2 * np.pi / 12)


class TestFreeEnergy(BaseTest):
    def _pairs(self, values):
        return [np.array([[t, p]]) for t, p in values]

    def test_table_recovers_a_flat_energy(self):
        thetas, phis = theta_grid(0.285, 0.392, 4), phi_grid(4)
        pairs = self._pairs([(t, p) for t in thetas for p in phis])
        energies = np.zeros(len(pairs))
        table = free_energy_table(pairs, energies, thetas, phis, 300.0)
        assert np.allclose(table, 0.0)

    def test_unsampled_cells_are_infinite(self):
        thetas, phis = theta_grid(0.285, 0.392, 4), phi_grid(4)
        pairs = self._pairs([(thetas[0], phis[0])])
        table = free_energy_table(pairs, np.array([0.0]), thetas, phis, 300.0)
        assert np.isfinite(table[0, 0])
        assert np.isinf(table).sum() == table.size - 1

    def test_pooling_keeps_cells_a_later_round_misses(self):
        thetas, phis = theta_grid(0.285, 0.392, 4), phi_grid(4)
        accumulator = FreeEnergyAccumulator(thetas, phis, 300.0)
        accumulator.add(self._pairs([(thetas[0], phis[0])]), np.array([5.0]))
        accumulator.add(self._pairs([(thetas[2], phis[2])]), np.array([5.0]))
        table = accumulator.table()
        assert np.isfinite(table[0, 0])
        assert np.isfinite(table[2, 2])

    def test_pooling_matches_a_single_batch(self):
        thetas, phis = theta_grid(0.285, 0.392, 4), phi_grid(4)
        values = [(thetas[1], phis[1]), (thetas[1], phis[1]), (thetas[3], phis[0])]
        energies = np.array([1.0, 9.0, 4.0])
        pooled = FreeEnergyAccumulator(thetas, phis, 300.0)
        # Second round carries the lower energy, forcing a shift rescale
        pooled.add(self._pairs(values[1:]), energies[1:])
        pooled.add(self._pairs(values[:1]), energies[:1])
        one_shot = free_energy_table(
            self._pairs(values), energies, thetas, phis, 300.0
        )
        assert np.allclose(pooled.table(), one_shot, equal_nan=True)

    def test_table_is_relative_not_absolute(self):
        # Shifting every energy by a constant shifts the table by the same amount
        thetas, phis = theta_grid(0.285, 0.392, 4), phi_grid(4)
        values = [(thetas[0], phis[0]), (thetas[1], phis[1])]
        base = free_energy_table(
            self._pairs(values), np.array([0.0, 3.0]), thetas, phis, 300.0
        )
        shifted = free_energy_table(
            self._pairs(values), np.array([10.0, 13.0]), thetas, phis, 300.0
        )
        finite = np.isfinite(base)
        assert np.allclose(shifted[finite] - base[finite], 10.0)

    def test_empty_accumulator_raises(self):
        accumulator = FreeEnergyAccumulator(theta_grid(0.285, 0.392, 4), phi_grid(4), 300.0)
        with pytest.raises(ValueError):
            accumulator.table()

    def test_effective_sample_size_bounds(self):
        assert effective_sample_size(np.zeros(10), 300.0) == pytest.approx(10.0)
        # One dominant sample collapses the effective count toward 1
        assert effective_sample_size(np.array([0.0, 200.0, 200.0]), 300.0) < 1.01


class TestMarginalFreeEnergy(BaseTest):
    def test_flat_table_is_flat(self):
        marginal = marginal_free_energy(np.zeros((4, 6)), 300.0, axis=1)
        assert marginal.shape == (4,)
        assert np.allclose(marginal, 0.0)

    def test_offset_to_zero_minimum(self):
        table = np.array([[0.0, 0.0], [5.0, 5.0], [20.0, 20.0]])
        marginal = marginal_free_energy(table, 300.0, axis=1)
        assert marginal.min() == pytest.approx(0.0)
        assert np.argmin(marginal) == 0
        assert marginal[1] == pytest.approx(5.0)

    def test_integrates_rather_than_minimizes(self):
        # Two rows share a minimum cell, but one has more low energy cells and
        # so must come out lower once the other axis is integrated out
        table = np.array([[0.0, 100.0, 100.0], [0.0, 0.0, 0.0]])
        marginal = marginal_free_energy(table, 300.0, axis=1)
        assert marginal[1] < marginal[0]
        assert np.allclose(np.min(table, axis=1), 0.0)

    def test_axis_selects_the_surviving_coordinate(self):
        table = np.zeros((3, 7))
        assert marginal_free_energy(table, 300.0, axis=1).shape == (3,)
        assert marginal_free_energy(table, 300.0, axis=0).shape == (7,)

    def test_all_infinite_row_stays_infinite(self):
        table = np.array([[0.0, 1.0], [np.inf, np.inf]])
        marginal = marginal_free_energy(table, 300.0, axis=1)
        assert np.isfinite(marginal[0])
        assert np.isinf(marginal[1])

    def test_empty_table_raises(self):
        with pytest.raises(ValueError):
            marginal_free_energy(np.full((2, 2), np.inf), 300.0)


class TestBarkerHenderson(BaseTest):
    def test_recovers_a_hard_sphere_diameter(self):
        separations = np.linspace(0.2, 0.9, 400)
        diameter = 0.5
        # A true hard sphere is infinitely repulsive inside its diameter
        u_pair = np.where(separations < diameter, 1e6, 0.0)[None, :]
        assert barker_henderson(u_pair, separations, 300.0) == pytest.approx(
            diameter, abs=0.01
        )

    def test_softer_potential_shrinks_with_temperature(self):
        separations = np.linspace(0.2, 0.9, 200)
        u_pair = (50.0 * (0.4 / separations) ** 12)[None, :]
        cold = barker_henderson(u_pair, separations, 200.0)
        hot = barker_henderson(u_pair, separations, 800.0)
        assert cold > hot


class TestFragmentParsing(BaseTest):
    def test_single_fragment(self):
        assert _fragment_names("{#A=[>]CC[<]}", None) == ["A"]
        assert _fragment_smiles("{#A=[>]CC[<]}", "A") == "CC"

    def test_multiple_fragments(self):
        cgsmiles = "{#A=[>]CC[<],#B=[>]COC[<]}"
        assert _fragment_names(cgsmiles, None) == ["A", "B"]
        assert _fragment_smiles(cgsmiles, "B") == "COC"

    def test_templates_contribute_names(self):
        assert _fragment_names(None, {"PEO": None}) == ["PEO"]

    def test_unknown_fragment(self):
        with pytest.raises(ValueError):
            _fragment_smiles("{#A=[>]CC[<]}", "B")


class TestChemistryCenter(BaseTest):
    CGSMILES = "{#A=[>]CC[<]}"

    def test_defaults_to_geometry(self):
        assert Chemistry(self.CGSMILES).center == "geometry"

    @pytest.mark.parametrize("center", ["geometry", "mass"])
    def test_center_is_kept(self, center):
        assert Chemistry(self.CGSMILES, center=center).center == center

    def test_invalid_center_raises_before_any_work(self):
        with pytest.raises(ValueError, match="geometry"):
            Chemistry(self.CGSMILES, center="centroid")

    def test_tuner_passes_center_through(self):
        tuner = Tuner(self.CGSMILES, center="mass")
        assert tuner.chemistry.center == "mass"

    def test_center_reaches_coarse_grain(self):
        from mbuild.path import Path

        # A mass weighted bead sits nearer the carbons, a geometric one is
        # pulled toward the hydrogens, so the two mappings disagree
        chemistry = Chemistry(self.CGSMILES)
        path = Path(
            coordinates=np.array([[0.0, 0, 0], [0.28, 0, 0], [0.42, 0.24, 0]]),
            bead_name="A",
        )
        path.form_linear_bond_graph()
        compound = chemistry.backmap(path)
        geometric = chemistry.coarse_grain(compound).coordinates
        chemistry.center = "mass"
        weighted = chemistry.coarse_grain(compound).coordinates
        assert geometric.shape == weighted.shape
        assert not np.allclose(geometric, weighted, atol=1e-6)


class TestTunerResult(BaseTest):
    def _result(self):
        thetas, phis = theta_grid(0.285, 0.392, 4), phi_grid(4)
        energies = np.zeros((4, 4))
        separations = np.linspace(0.2, 0.9, 100)
        u_pair = np.where(separations < 0.4, 1e6, 0.0)[None, :]
        return TunerResult(
            bond_length=0.285,
            radius=0.392,
            angles=AngleDihedralSampler(thetas, phis, energies, temperature=300.0),
            temperature=300.0,
            theta_grid=thetas,
            phi_grid=phis,
            energies=energies,
            ess=[10.0],
            n_samples=100,
            u_pair=u_pair,
            separations=separations,
        )

    def test_as_walk_kwargs_drives_a_walk(self):
        from mbuild.path import hard_sphere_random_walk

        kwargs = self._result().as_walk_kwargs()
        assert set(kwargs) == {"bond_length", "radius", "rw_angles"}
        path = hard_sphere_random_walk(**kwargs, termination=25, seed=3)
        assert len(path.coordinates) == 25
        assert np.allclose(bond_lengths(path.coordinates), 0.285, atol=1e-5)

    def test_at_rebuilds_for_another_temperature(self):
        result = self._result().at(600.0)
        assert result.temperature == 600.0
        assert result.angles.temperature == 600.0
        assert result.bond_length == 0.285
        # The table is temperature free, so it carries over untouched
        assert np.allclose(result.energies, 0.0)

    def test_marginal_energies_match_the_grids(self):
        result = self._result()
        thetas, angle_energies = result.angle_energy()
        phis, dihedral_energies = result.dihedral_energy()
        assert thetas.shape == angle_energies.shape
        assert phis.shape == dihedral_energies.shape


class TestPlotting(BaseTest):
    @pytest.fixture(autouse=True)
    def _agg_backend(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        yield
        matplotlib.pyplot.close("all")

    def _result(self):
        thetas, phis = theta_grid(0.285, 0.392, 6), phi_grid(6)
        # A table with real structure, so the curves are not degenerate
        energies = 20.0 * (1.0 - np.cos(phis)[None, :]) + (
            50.0 * (thetas - 2.0) ** 2
        )[:, None]
        energies[0, 0] = np.inf
        separations = np.linspace(0.2, 0.9, 60)
        u_pair = (10.0 * ((0.4 / separations) ** 12 - (0.4 / separations) ** 6))[None, :]
        return TunerResult(
            bond_length=0.285,
            radius=0.392,
            angles=AngleDihedralSampler(thetas, phis, energies, temperature=300.0),
            temperature=300.0,
            theta_grid=thetas,
            phi_grid=phis,
            energies=energies,
            ess=[10.0],
            n_samples=100,
            u_pair=u_pair,
            separations=separations,
        )

    @pytest.mark.parametrize("method", ["plot_pair", "plot_angles", "plot_dihedrals"])
    def test_single_plots_return_axes(self, method):
        import matplotlib.pyplot as plt

        result = self._result()
        ax = getattr(result, method)()
        assert ax.get_xlabel() and ax.get_ylabel()
        assert len(ax.lines) > 0
        # An explicit axes is drawn on rather than replaced
        _, given = plt.subplots()
        assert getattr(result, method)(ax=given) is given

    def test_summary_plot_has_three_panels(self):
        figure, axes = self._result().plot()
        assert len(axes) == 3
        assert figure.axes == list(axes)

    def test_summary_plot_overlays_temperatures(self):
        _, axes = self._result().plot(temperature=[200.0, 300.0, 600.0])
        # One curve per temperature on the angle panel
        assert len(axes[1].lines) == 3

    def test_unsampled_bins_break_the_line(self):
        result = self._result()
        result.energies = np.full_like(result.energies, np.inf)
        result.energies[:, 0] = 0.0
        ax = result.plot_dihedrals()
        y = ax.lines[0].get_ydata()
        assert np.isnan(y[1:]).all()
        assert np.isfinite(y[0])
