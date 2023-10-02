import mbuild as mb
from mbuild.lattice import Lattice


class Graphene(Compound):
    def __init__(self, x_repeat, y_repeat, n_layers, periodic=True):
        if periodic:
            periodicity = (True, True, False)
        else:
            periodicity = (False, False, False)
        super(Graphene, self).__init__(periodicity=periodicity)
        spacings = [0.425, 0.246, 0.35]
        points = [[1/6, 0, 0], [1/2, 0, 0], [0, 0.5, 0], [2/3, 1/2, 0]]
        lattice = mb.Lattice
                spacings=spacings,
                angles=[90,90,90],
                lattice_points={"A": points}
        )
        carbon = mb.Compound(name="C")
        layers = lattice.populate(
                compound_dict={"A": carbon}, x=x_repeat, y=y_repeat, z=n_layers
        )
        self.add(layers)
        self.freud_generate_bonds("C", "C", dmin=0.14, dmax=0.145)
