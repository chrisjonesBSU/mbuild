import re

import mbuild as mb
from mbuild import Compound, Port
# from mbuild.library.react import react_library


def load_reactant(string, name=None):
    """Load a compound using SMILES with reaction notation.
    
    Parameters
    ----------
    string : str, required
        SMILES string of the compound to load with the
        reaction notation included.
    """
    reactive_indices = []
    site_types = []
    skip_strings = 0
    for idx, s in enumerate(string):
        if not s.isalpha():
            if s == "*":
                reactive_indices.append(idx - (1 + skip_strings))
                site_types.append("polymer")
            elif s == "^":
                reactive_indices.append(idx - (1 + skip_strings))
                site_types.append("branch")
            skip_strings += 1
    # Get the actual SMILES string
    smiles = string.replace("*", "")
    smiles = smiles.replace("^", "")
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

    def reaction_sites(self, reaction_type):
        """"""
        #TODO: Make this a dict?
        sites = []
        for p in self.particles():
            if p.element.atomic_number == 1:
                continue
            if p.reactive and p.reaction_type == reaction_type:
                sites.append(
                    [p]
                    + [
                        i
                        for i in p.direct_bonds()
                        if i.element.atomic_number == 1
                    ]
                    + [port for port in self.available_ports()
                       if port.anchor is p]
                )
        return sites

    def set_reactive(self, reaction_type):
        """"""
        if not self._contains_only_ports():
            raise AttributeError(
                "Reaction types are immutable for Compounds that are "
                "not at the bottom of the containment hierarchy."
                "Setting reactive sites must be done at the particle level."
            )
        self._reactive = True
        self.reaction_type = reaction_type
        h_bonds = [] 
        # Set all directly bonded hydrogens to reactive as well
        for p in self.direct_bonds():
            if p.element.atomic_number == 1:
                p._reactive = True
                p.reaction_type = reaction_type
                h_bonds.append(p.xyz[0] - self.xyz[0])
        # Make port
        orientation = np.mean(h_bonds, axis=0)
        port = Port(
                anchor=self,
                separation=0.10,
                orientation=orientation
        )
        self.parent.add(port)


class React(Compound):
    """"""
    def __init__(
        self,
        reactants=None,
        sequence=None,
        n_repeats=1,
        library=None,
        name=None,
    ):
        super(React, self).__init__(name=name)
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

    def react(self, site1, site2, bond_distance, bond_order=1):
        # Get reaction site ports and anchors
        site1_port = [p for p in site1 if isinstance(p, mb.port.Port)][0]
        site1_port.update_separation(bond_distance / 2)
        site1_anchor = site1_port.anchor
        site2_port = [p for p in site2 if isinstance(p, mb.port.Port)][0]
        site2_port.update_separation(bond_distance / 2)
        site2_anchor = site2_port.anchor
        # Remove hydrogens based on bond order
        for site in [site1_anchor, site2_anchor]:
            hs = [p for p in site.direct_bonds() if p.element.atomic_number == 1]
            if bond_order == 1:
                site.parent.remove(hs[0]) 
            elif bond_order == 2:
                site.parent.remove(hs[0])
                site.parent.remove(hs[1])
            elif bond_order == 3:
                site.parent.remove(hs[0])
                site.parent.remove(hs[1])
                site.parent.remove(hs[2])
        # Force overlap
        mb.force_overlap(
            move_this=site2_anchor.parent,
            from_positions=site2_port,
            to_positions=site1_port,
        )
        self.add(site1_anchor.parent)
        self.add(site2_anchor.parent)

    def coarse_grain(self):
        """For each atomistic chunk of self.sequence, get CoM, place beads there"""
        pass
