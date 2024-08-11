import mbuild as mb
from mbuild import Compound


def load(string):
    """"""
    reactive_indices = []
    site_types = []
    skip_strings = 0
    for idx, string in enumerate(smirks):
        if not string.isalpha():
            if string == ".":
                reactive_indices.append(idx-(1 + skip_strings))
                site_types.append("polymer")
            if string == ",":
                site_types.append("branch")
            skip_strings += 1
    smiles = smirks.replace(".", "")
    smiles = smiles.replace(",", "")
    mb_comp = mb.load(smiles, smiles=True)
    mb_comp.name = name
    for idx, _type in zip(reactive_indices, site_types):
        mb_comp[idx].set_reactive(reaction_type=_type)
    return mb_comp


class Reactant
    """"""
    def __init__(
        self,
        subcompounds=None,
        name=None,
        pos=None,
        mass=None,
        charge=None,
        periodicity=None,
        box=None,
        element=None,
        port_particle=False,
    ):


class React:
    """"""
    def __init__(self, reactants, sequence, n_repeats=1):
        self.reactants = reactants
        self.sequence = sequence
        self.n_repeats = n_repeats

    def _parse(self):
        """Generate connection algorithm given the sequence"""
        pass

    def _react(self):
        """After parsing, do all the mBuild book keep stuff"""
        pass

    def coarse_grain(self):
        """For each atomistic chunk of self.sequence, get CoM, place beads there"""
        pass
