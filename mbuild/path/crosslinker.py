"""Classes to configure cross-linked CG configurations."""

import logging
import types

import networkx as nx
import numpy as np

import mbuild as mb
from mbuild.path import HardSphereRandomWalk
from mbuild.path.bias import TargetType
from mbuild.path.termination import NumAttempts, NumSites, Termination
from mbuild.simulation import energy_minimize_path
from mbuild.utils.io import import_
from mbuild.utils.volumes import CuboidConstraint

freud = import_("freud")
logger = logging.getLogger(__name__)


class CrossLinkedPolymer:
    """The overall class that contains all valid steps for performing actions to random walk generate a crosslinked polymer.

    Parameters
    ----------
    targets : list of classes
        The methods assessed at each random step to build up a system.
    box : mb.box.Box
        mBuild box used to fill the simulation.
        Set periodic x boundaries with box.periodicity = [True, False, False]
        # NOTE: could replace this with an optional constraint class.
    max_attempts : int, default 100, optional
       Maximum steps to take for many target classes.
    bead_radius : float, default 1, optional
        The radius in nm for beads.
        # NOTE: This should maybe be part of the targets class instead
    seed : int, default 1
        The value used to control the random number generator. Deterministic.
    logfile : str, default None, optional
        Absolute path to write info about builder. Important for debugging. No
        file is written if None is passed.

    """

    def __init__(
        self,
        targets,
        box,
        max_attempts=100,
        bead_radius=1,
        seed=1,
        logfile=None,
    ):
        self.path = None
        self.box = box
        self.shape_constraint = CuboidConstraint(
            Lx=box.Lx, Ly=box.Ly, Lz=box.Lz, pbc=box.periodicity
        )
        self.targets = targets
        self.npolymers = 0
        self.max_attempts = max_attempts
        self.bead_radius = bead_radius  # this could be dynamic, growing and shrinking, and hooked to relax chain
        self.bond_length = bead_radius * 1.12
        self.step = 0
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)
        self.failed_steps = 0
        if logfile:
            # TODO: Validate path
            self.logfile = logfile
            header = (
                "Running CrossLinkedPolymer with\n"
                + f"N Polymers:\t {targets['NPolymers']}\n"
                + f"Polymer Length:\t {targets['chainlengths'][0]}\n"
                + f"Crosslinks:\t {targets['NCrosslinks']}\n"
                + f"Box:\t {self.box}\n"
                + "#" * 30
                + "\n"
            )
            with open(self.log_file, "w") as f:
                f.write(header)
        else:
            self.logfile = None

    def add_methods(self, buildList):
        """Bind all methods directly to this instance with their original names"""
        if not isinstance(buildList, (list, tuple)):
            buildList = tuple(buildList)
        for buildClass in buildList:
            for method_name in ["method", "weight"]:
                if hasattr(buildClass.__class__, method_name):
                    bound_method = types.MethodType(
                        getattr(buildClass.__class__, method_name), self
                    )
                    wrapped_method = MethodWrapper(bound_method, buildClass.__class__)
                    setattr(self, method_name, wrapped_method)
                for other_attrs in [
                    "nCrosslinks",
                    "ff",
                    "blacklist_crosslinks",
                    "hit_maximum_sites",
                ]:
                    # TODO: This could be more dynamic, instead of manually looking for attributs to add.
                    if hasattr(
                        buildClass, other_attrs
                    ):  # inject other useful attributes
                        setattr(self, other_attrs, getattr(buildClass, other_attrs))

            # Also store them in lists for the run method
            if not hasattr(self, "methods"):
                self.methods = [self.method]
                self.weights = [self.weight]
                self.method_classes = [buildClass.__class__.__name__]
            else:
                self.methods.append(self.method)
                self.weights.append(self.weight)
                self.method_classes.append(buildClass.__class__.__name__)

    def generate_weights(self):
        """Iterate through all weights and methods and pair these values at runtime."""
        if not self.weights:
            raise ValueError("Missing Weights. Please add something via add_methods")
        return [
            (weightFunction(), method)
            for weightFunction, method in zip(self.weights, self.methods)
        ]

    def update_progress(self, selected_method):
        """Method to print to screen current state of the builder."""
        msg1 = f"-------Generated {self.npolymers} polymers--------"
        msg2 = f"-------Created {getattr(self, 'nCrosslinks', None)} crosslinks--------"
        msg3 = f"At step {self.step}: {selected_method.get_original_class_name()}: Failed {getattr(self, 'failed_steps', 0)} consecutively"

        if not hasattr(self, "_progress_initialized"):
            # First time: just print both lines
            print(f"{msg1}\n{msg2}\n{msg3}", flush=True)
            self._progress_initialized = True
        else:
            # Subsequent times: move up 2 lines and overwrite
            print(f"\033[3A\r{msg1}\033[K\n{msg2}\033[K\n{msg3}\033[K", flush=True)
            # print(msg1, msg2, msg3)
        if self.logfile:
            with open(self.log_file, "a") as f:
                f.write(msg3 + "\n")

    def run(self):
        """Central function to run the builder based on all methods added via self.add_methods."""
        for _ in range(self.max_attempts):
            self.step += 1
            current_weights = self.generate_weights()
            filtered_weights = [tup for tup in current_weights if tup[0] > 0]
            if not filtered_weights:  # exit here on completion
                print("\nCompleted Cycle")
                break
            weights, available_methods = zip(*filtered_weights)
            selected_method = self.rng.choice(
                available_methods,
                size=None,
                p=np.array(weights) / sum(weights),
                replace=True,
            )
            # selected_method = random.choices(available_methods, weights, k=1)[0]
            if selected_method():  # run next cycle step
                if selected_method.get_original_class_name() == "RelaxChain":
                    self.failed_steps += 1
                else:
                    self.failed_steps = 0  # successful step
            else:
                self.failed_steps += 1
            self.update_progress(selected_method)
            if self.failed_steps and not self.failed_steps % 3:
                # TODO: make an actual method with a weight to scale
                self.bead_radius *= 0.9  # slowly decrease bead size

    def assess_crosslinks(self, reassess=True):
        """Go through bondgraph, identify _L particles, and get their adjacency."""
        if not reassess and getattr(self, "crosslink_distribution", None):
            return self.crosslink_distribution
        adjacencyList = [0, 0, 0, 0]
        # Iterate through all nodes and their attributes
        for node, attributes in self.path.bond_graph.nodes(data=True):
            # Check if the node has the attribute 'node_type' and its value is '_L'
            if attributes.get("name") == "_L":
                # Get the neighbors (adjacency) of the current node
                neighbors = list(self.path.bond_graph.neighbors(node))
                adjacencyList[len(neighbors) - 1] += 1
        self.crosslink_distribution = adjacencyList
        return self.crosslink_distribution


class RandomChainBuilder:
    """Class to build chains based on self.targets['chainlengths']"""

    def __init__(self):
        pass

    def method(self):
        completed = False
        for i in range(10):
            try:
                self.path = HardSphereRandomWalk(
                    termination=Termination(
                        [
                            NumSites(self.targets["chainlengths"][self.npolymers]),
                            NumAttempts(self.max_attempts),
                        ]
                    ),
                    radius=self.bead_radius,
                    volume_constraint=self.shape_constraint,
                    bond_length=self.bond_length,
                    min_angle=np.pi / 2,
                    max_angle=np.pi,
                    start_from_path=getattr(self, "path", None),
                    attach_paths=False,
                    seed=self.seed + i,
                    trial_batch_size=100,
                    bead_name="_A",
                )
                self.npolymers += 1
                completed = True
                break  # Completed
            except RuntimeError:
                pass
        return completed

    def weight(self):
        """10x the difference between target and current number of polymers."""
        return 10 * (self.targets["NPolymers"] - self.npolymers)


class RandomCrossLinker:
    """Class to randomly crosslink chains based on self.targets['NumCrosslinks'] with a single bead."""

    def __init__(self):
        self.nCrosslinks = 0
        self.blacklist_crosslinks = []
        self.hit_maximum_sites = False

    def method(self):
        if not getattr(self, "path", None):
            return  # unable to create crosslink without a backbone
        backbones = [
            node
            for node, attrs in self.path.bond_graph.nodes(data=True)
            if attrs.get("name") == "_A" and node not in self.blacklist_crosslinks
        ]
        # random_starts = np.random.choice(backbones, len(backbones), replace=False)
        if len(backbones) == 0:
            self.hit_maximum_sites = True

        completed = False
        for start_index in self.rng.choice(
            backbones, size=min(len(backbones), 3), replace=False
        ):
            try:
                self.path = HardSphereRandomWalk(
                    termination=Termination([NumSites(1), NumAttempts(200)]),
                    radius=self.bead_radius,  # TODO: This should be determined when class is created.
                    volume_constraint=self.shape_constraint,
                    bead_name="_L",  # TODO: Should be passable.
                    bond_length=self.bond_length,
                    bias=TargetType(target_type="_A", r_cut=0.5, weight=1.0),
                    min_angle=np.pi / 3,
                    max_angle=np.pi,
                    start_from_path=self.path,
                    start_from_path_index=int(start_index),
                    attach_paths=True,
                    seed=self.seed,
                    trial_batch_size=500,
                )
                self.nCrosslinks += 1
                # try to attach to a different chain
                try:
                    # TODO: This method could also be configurable.
                    self.form_crosslink(
                        path=self.path,
                        particle_index=self.path.N - 1,
                        allowed_bond_types=["_A"],
                        max_distance=0.38,
                        excluded_bond_depth=4,
                        n_allowed_connections=2,
                    )
                except RuntimeError:
                    logger.debug(
                        f"Failed to form crosslinker number {self.nCrosslinks} across two backbones."
                    )
                lastNode = len(self.path.bond_graph) - 1
                blacklist_crosslinks = list(self.path.bond_graph.neighbors(lastNode))
                self.blacklist_crosslinks.extend(blacklist_crosslinks)
                completed = True
                break
            except RuntimeError:
                logger.debug(
                    f"Failed to add crosslinker number {self.nCrosslinks} to system."
                )
        return completed

    def weight(self):
        """Ratio of (distance to target crosslinks)/(half distance to target polymers)"""
        if self.hit_maximum_sites:
            return 0
        return (self.targets["NCrosslinks"] - self.nCrosslinks) / (
            (self.targets["NPolymers"] - self.npolymers + 1) / 2
        )

    def form_crosslink(
        self,
        path,
        particle_index,
        max_distance,
        allowed_bond_types=None,
        excluded_bond_depth=3,
        n_allowed_connections=2,
    ):
        """Take a single bead and try to form a bond with neighboring backbone."""
        particle_types = np.array(
            [attrs["name"] for _, attrs in path.bond_graph.nodes(data=True)]
        )
        pos, box = path.to_freud()
        ref_pos = pos[particle_index]
        if allowed_bond_types:
            keep_indices = np.where(np.isin(particle_types, allowed_bond_types))[0]
            # pos = pos[keep_indices]

        aq = freud.locality.AABBQuery(box, pos)
        aq_query = aq.query(
            query_points=ref_pos,
            query_args=dict(r_min=0.01, r_max=max_distance, exclude_ii=True),
        )
        nlist = aq_query.toNeighborList()
        neighbors = []
        for i, j in nlist:
            # Exclude incorrect bond types
            if allowed_bond_types and j not in keep_indices:
                continue
            # Exclude bonded neighbors that are within min distance
            if excluded_bond_depth > 0:
                i_bonds = nx.ego_graph(
                    path.bond_graph, particle_index, radius=excluded_bond_depth
                )
                # i_bonds = path.bond_graph.neighbors(particle_index).direct_bonds(graph_depth=excluded_bond_depth)
                if path.bond_graph[int(j)] not in i_bonds:
                    neighbors.append(int(j))
            else:
                neighbors.append(int(j))
        neighbors = [
            node
            for node in neighbors
            if path.bond_graph.degree(node) <= n_allowed_connections
        ]
        if neighbors:
            partnerIndex = sorted(
                neighbors, key=lambda x: np.linalg.norm(pos[x] - ref_pos)
            )[0]
            path.bond_graph.add_edge(path.N - 1, int(partnerIndex))
        return nlist


class RelaxChain:
    """Class to relax overall system before trying another random walk."""

    def __init__(self):
        pass

    def method(self):
        if not getattr(self, "path", None):
            return False
        energy_minimize_path(self.path, self.bead_radius)
        return True

    def weight(self):
        """Triggered after other methods fail three times in a row"""
        return 1000 if not (self.failed_steps + 1) % 3 else 0


# TODO: Set specific percentage of crystalline portion
# class Crystalline

# TODO: Method to densify chains to form higher crosslinking
# class Densify


class MethodWrapper:
    def __init__(self, method, original_class):
        self.method = method
        self.original_class_name = original_class.__name__
        self.original_class = original_class

    def __call__(self, *args, **kwargs):
        return self.method(*args, **kwargs)

    def get_original_class_name(self):
        return self.original_class_name


if __name__ == "__main__":
    # A test script to validate compatibility
    from datetime import datetime

    import unyt as u

    radius = 0.25  # nm
    density = 0.5 * u.g / u.cm**3  # g/cm**3
    n_mers, n_polymers, n_crosslinks = 50, 50, 500  # test case takes ~5s
    box_length = (
        ((n_mers * n_polymers + n_crosslinks) * 14.01 * u.amu / density)
        .in_units("nm**3")
        .value
    ) ** (1 / 3)
    crosslinked_polymer = CrossLinkedPolymer(
        {
            "NPolymers": n_polymers,
            "NCrosslinks": n_crosslinks,
            "chainlengths": [n_mers] * n_polymers,
        },
        bead_radius=radius,
        box=mb.Box([box_length] * 3, periodicity=[True] * 3),
        max_attempts=500,
        seed=2,
    )
    build = RandomChainBuilder()
    crosslink = RandomCrossLinker()
    relaxer = RelaxChain()
    crosslinked_polymer.add_methods([build, crosslink, relaxer])
    start = datetime.now()
    crosslinked_polymer.run()
    print(f"Took {(datetime.now() - start).total_seconds() / 60:.2f} minutes")
    clinks = crosslinked_polymer.assess_crosslinks()
    outmsgList = [
        f"{i + 1} bonded crosslinkers count:\t {cl}" for i, cl in enumerate(clinks)
    ]
    print("\n".join(outmsgList))
    print(f"Actual Crosslinked at {100 * clinks[1] / (n_polymers * n_mers):.2f}%")
    print(
        f"Attempted Crosslinked at {100 * crosslinked_polymer.targets['NCrosslinks'] / (n_polymers * n_mers):.2f}%"
    )
