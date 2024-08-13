import re

import mbuild as mb
from mbuild import Compound

# from mbuild.react.library import react_library


def load_reactant(smirks, name=None):
    """"""
    reactive_indices = []
    site_types = []
    skip_strings = 0
    for idx, string in enumerate(smirks):
        if not string.isalpha():
            if string == ".":
                reactive_indices.append(idx - (1 + skip_strings))
                site_types.append("polymer")
            if string == ",":
                reactive_indices.append(idx - (1 + skip_strings))
                site_types.append("branch")
            skip_strings += 1
    smiles = smirks.replace(".", "")
    smiles = smiles.replace(",", "")
    # Load using the actual SMILES string now
    mb_comp = mb.load(
        filename_or_object=smiles, smiles=True, compound=Reactant(name=name)
    )
    for idx, _type in zip(reactive_indices, site_types):
        mb_comp[idx].set_reactive(reaction_type=_type)
    return mb_comp


class Reactant(Compound):
    """"""

    def __init__(self, name):
        super(Reactant, self).__init__(name=name)

    def branch_sites(self):
        """"""
        sites = []
        for p in self.particles():
            if p.element.atomic_number == 1:
                continue
            if p.reactive and p.reaction_type == "branch":
                sites.append(
                    [p]
                    + [
                        i
                        for i in p.direct_bonds()
                        if i.element.atomic_number == 1
                    ]
                )
        return sites

    def polymer_sites(self):
        """"""
        sites = []
        for p in self.particles():
            if p.element.atomic_number == 1:
                continue
            if p.reactive and p.reaction_type == "polymer":
                sites.append(
                    [p]
                    + [
                        i
                        for i in p.direct_bonds()
                        if i.element.atomic_number == 1
                    ]
                )
        return sites

    def set_reactive(self, reaction_type):
        """"""
        if self._contains_only_ports():
            self.reactive = True
            self.reaction_type = reaction_type
            h_bonds = []
            for p in self.direct_bonds():
                if p.element.atomic_number == 1:
                    p.reactive = True
                    p.reaction_type = reaction_type
                    h_bonds.append(p.xyz - self.xyz)
            # Make port
            print("adding port")
            port = mb.port.Port(
                    anchor=self,
                    length=0.10,
                    orientation=np.mean(h_bonds, axis=0)
            )
            self.add(port)

        else:
            raise AttributeError(
                "Reaction types are immutable for Compounds that are "
                "not at the bottom of the containment hierarchy."
            )


class React:
    """"""

    def __init__(
        self,
        reactants=None,
        sequence=None,
        n_repeats=1,
        library=None,
        name=None,
    ):
        self.name = name
        self.reactants = reactants
        self.sequence = sequence
        self.n_repeats = n_repeats
        self._library = library
        self._parse()
        self._components = self._parse()

    @classmethod
    def from_sequence(self, sequence, n_repeats):
        """"""
        pass

    @classmethod
    def from_reactants(self, reactants, sequence, n_repeats):
        """"""
        pass

    def _parse(self):
        """Generate connection algorithm given the sequence"""
        components = []
        component_str = re.split(r"(-|=)", self.sequence)
        return component_str

    def _react(site1, site2, bon_order=1):
        """After parsing, do all the mBuild book keeping stuff"""
        pass

    def coarse_grain(self):
        """For each atomistic chunk of self.sequence, get CoM, place beads there"""
        pass
