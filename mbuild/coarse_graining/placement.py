"""Atom placement strategies for backmapped fragments.

Given a resolved molecular graph and the CG bead anchors, compute a
position for every atom: local fragment geometry comes from a user
template or an RDKit embedding (cached per unique fragment). Fragments
are rotated so their junction atoms point toward the neighboring beads.
"""

import logging

import numpy as np

from mbuild.utils.geometry import kabsch
from mbuild.utils.io import import_

logger = logging.getLogger(__name__)


def generate_positions(
    molecule,
    node_to_beads,
    anchors,
    bead_fragnames,
    templates,
    seed,
    all_atom=True,
    n_twists=24,
    twist_cutoff=0.6,
):
    """Compute a position for every atom in the resolved molecule.

    For each bead, local fragment geometry is taken from a user template
    (if one covers the bead's fragment) or an RDKit embedding of the
    fragment (cached per unique fragment). Fragments are rotated so the
    atoms bonded to other beads point toward those beads, then centered
    on the bead anchor.

    Clashes resulting from back-mapping are further resolved by scanning
    ``n_twists`` rotations and keeping the one that overlaps the already
    placed atoms least. Beads are placed in sorted order, so each fragment
    is oriented against its predecessors. Pass ``n_twists=0`` to leave the
    twist as the orientation fit returns it.

    With ``all_atom=False`` the resolved nodes are finer coarse-grained
    beads rather than atoms. Local geometry then comes only from
    templates; a fragment without one raises ``ValueError``.

    Parameters
    ----------
    n_twists : int, default 24
        Rotations scanned when a fragment's twist is undetermined. Below 2
        skips the scan.
    twist_cutoff : float (nm), default 0.6
        Placed atoms beyond this distance from a bead are ignored when
        scoring its twist.
    """
    # Group atoms by primary bead
    bead_to_atoms = {}
    for node in molecule.nodes:
        bead_to_atoms.setdefault(node_to_beads[node][0], []).append(node)

    embed_cache = {}
    positions = {}
    for bead, atoms in sorted(bead_to_atoms.items()):
        atoms = sorted(atoms)
        atom_set = set(atoms)
        rel_index = {node: i for i, node in enumerate(atoms)}
        fragname = bead_fragnames.get(bead)

        if templates is not None and fragname in templates:
            local_xyz = _template_local_coords(
                molecule, atoms, templates[fragname], fragname, all_atom=all_atom
            )
        elif not all_atom:
            # a coarse-grained endpoint has no atomistic detail to embed
            raise ValueError(
                f"No template was provided for coarse-grained fragment "
                f"'{fragname}'. Resolving to a coarse-grained resolution "
                "has no atomistic geometry to embed, so each such fragment "
                "needs a template compound whose particles are its "
                "sub-beads, in the same order as the fragment definition: "
                f"templates={{'{fragname}': compound}}."
            )
        else:
            import_("rdkit")
            try:
                local_xyz = _embedded_fragment_coords(
                    molecule, atoms, rel_index, embed_cache, seed
                )
            except Exception as error:
                raise ValueError(
                    f"RDKit embedding failed for fragment '{fragname}': "
                    f"{error}. Provide a template for this fragment "
                    "instead."
                ) from error

        group_anchor = np.mean([anchors[node] for node in atoms], axis=0)

        # Rotate the fragment so its junction atoms (those bonded to
        # atoms in other beads) point toward the neighboring beads
        sources = []
        targets = []
        for node in atoms:
            for neighbor in molecule.neighbors(node):
                if neighbor in atom_set:
                    continue
                source = local_xyz[rel_index[node]]
                target = anchors[neighbor] - group_anchor
                source_norm = np.linalg.norm(source)
                target_norm = np.linalg.norm(target)
                if source_norm > 1e-8 and target_norm > 1e-8:
                    sources.append(source / source_norm)
                    targets.append(target / target_norm)
        if sources:
            sources = np.array(sources)
            targets = np.array(targets)
            local_xyz = local_xyz @ kabsch(sources, targets).T
            axis = _underdetermined_axis(sources, targets)
            if axis is not None:
                local_xyz = _resolve_twist(
                    local_xyz,
                    axis,
                    group_anchor,
                    positions,
                    n_twists=n_twists,
                    cutoff=twist_cutoff,
                )

        for node in atoms:
            positions[node] = group_anchor + local_xyz[rel_index[node]]
    return positions


def _underdetermined_axis(sources, targets, tol=1e-6):
    """Return the axis a fragment remains free to rotate about, or None.

    The orientation fit pins the one direction the junction vectors span
    onto its image among the neighbor directions, which is the leading
    right singular vector of the same covariance ``kabsch`` decomposes. For
    a fragment with two junctions that direction is the local chain
    tangent. Its sign is arbitrary and does not matter, since a full turn
    is scanned about it.

    Parameters
    ----------
    sources : np.ndarray (N, 3), required
        Unit junction directions in the fragment's local frame.
    targets : np.ndarray (N, 3), required
        Unit directions from the fragment toward the neighboring beads.
    tol : float, default 1e-6
        Threshold on the singular values of the fit, relative to the
        largest.

    Returns
    -------
    np.ndarray (3,) or None
        Unit axis when the fit leaves a rotation free. None when the fit
        fixes the orientation, and also when it fixes no direction at all,
        which leaves no single axis to scan about.
    """
    singular_values, right_vectors = np.linalg.svd(sources.T @ targets)[1:]
    if singular_values[0] <= tol:
        return None
    if singular_values[1] > tol * singular_values[0]:
        return None
    return right_vectors[0]


def _resolve_twist(local_xyz, axis, group_anchor, positions, n_twists=24, cutoff=0.6):
    """Rotate a fragment about an axis to its least overlapping orientation.

    Scans ``n_twists`` evenly spaced rotations about ``axis`` and keeps the
    one minimizing a soft repulsion against the atoms in ``positions`` that
    lie within ``cutoff`` of ``group_anchor``.

    Parameters
    ----------
    local_xyz : np.ndarray (M, 3), required
        Fragment coordinates relative to its own centroid.
    axis : np.ndarray (3,), required
        Unit axis to rotate about.
    group_anchor : np.ndarray (3,), required
        Coordinate the fragment will be centered on.
    positions : dict, required
        Node to coordinate mapping of the atoms already placed.
    n_twists : int, default 24
        Rotations scanned over a full turn. Below 2 leaves the fragment
        untouched.
    cutoff : float (nm), default 0.6
        Placed atoms beyond this distance from ``group_anchor`` are ignored.

    Returns
    -------
    np.ndarray (M, 3)
        The selected coordinates, unchanged when no placed atom lies within
        ``cutoff``.
    """
    if n_twists < 2 or not positions:
        return local_xyz
    placed = np.array(list(positions.values()))
    offsets = placed - group_anchor
    near = placed[np.einsum("ij,ij->i", offsets, offsets) < cutoff * cutoff]
    if near.size == 0:
        return local_xyz

    neighbors = near - group_anchor
    best_xyz, best_score = local_xyz, np.inf
    for angle in np.linspace(0.0, 2.0 * np.pi, n_twists, endpoint=False):
        candidate = local_xyz @ _axis_rotation(axis, angle).T
        deltas = candidate[:, None, :] - neighbors[None, :, :]
        sq_distances = np.einsum("ijk,ijk->ij", deltas, deltas)
        # Soft repulsion, so every close contact counts rather than the
        # single closest one
        score = float(np.sum(1.0 / np.maximum(sq_distances, 1e-12) ** 3))
        if score < best_score:
            best_xyz, best_score = candidate, score
    return best_xyz


def _axis_rotation(axis, angle):
    """Rotation matrix for a right handed rotation about a unit axis."""
    x, y, z = axis
    cos, sin = np.cos(angle), np.sin(angle)
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return cos * np.eye(3) + sin * cross + (1.0 - cos) * np.outer(axis, axis)


def _embedded_fragment_coords(molecule, atoms, rel_index, embed_cache, seed):
    """Local fragment coordinates from a cached RDKit embedding.

    Fragment instances created from the same CGsmiles fragment have
    their atoms in the same relative node order.
    """
    elements = tuple(molecule.nodes[n].get("element", "*") for n in atoms)
    charges = tuple(molecule.nodes[n].get("charge", 0) for n in atoms)
    edges = tuple(
        sorted(
            (
                min(rel_index[u], rel_index[v]),
                max(rel_index[u], rel_index[v]),
                data.get("order", 1),
            )
            for u, v, data in molecule.subgraph(atoms).edges(data=True)
        )
    )
    key = (elements, charges, edges)
    if key not in embed_cache:
        embed_cache[key] = _embed_fragment(elements, charges, edges, seed)
    return embed_cache[key]


def _tag_as_index(particle):
    """Interpret a particle's tag as a fragment atom index, if possible.

    Non-integer tags (e.g. the "head"/"tail" tags used by the Polymer
    workflow) are ignored so templates carrying them still match by
    atom order.
    """
    tag = particle.particle_tag
    if tag is None:
        return None
    try:
        return int(str(tag))
    except ValueError:
        return None


def _template_local_coords(molecule, atoms, template, fragname, all_atom=True):
    """Local coordinates for a bead's atoms taken from an mBuild template.

    Heavy atoms in the resolved molecule are matched to the template's
    heavy particles either explicitly, with every heavy particle tagged
    by its fragment atom index, e.g. ``mb.load("O{1}(C{0})C{2}",
    smiles=True)``, or, for untagged templates, by fragment atom order
    (the CGsmiles ``mapping`` node attribute), in which case the
    template's heavy atoms must appear in the same order as in the
    fragment SMILES. Each hydrogen is assigned the position of an unused
    hydrogen bonded to its parent's matched template particle; surplus
    template hydrogens (e.g. saturating a junction) are ignored. Returns
    coordinates in nm, centered on the centroid of the used atoms.

    With ``all_atom=False`` the resolved fragment is coarse-grained: its
    nodes are sub-beads with no element and no hydrogens, so each is
    matched to the template particle at the same fragment index (see
    ``_template_local_coords_cg``).
    """
    if not all_atom:
        return _template_local_coords_cg(molecule, atoms, template, fragname)

    def symbol(particle):
        """Element symbol of a particle, falling back to its name when no
        element is set.
        """
        if particle.element is not None:
            return particle.element.symbol
        return particle.name

    heavy_nodes = []
    for node in atoms:
        if molecule.nodes[node].get("element") == "H":
            continue
        mapping = molecule.nodes[node].get("mapping")
        if not mapping or len(mapping) != 1:
            raise ValueError(
                f"Cannot use a template for fragment '{fragname}': atom "
                f"{node} has no unique fragment mapping (squashed fragments "
                "are not supported by template placement)."
            )
        heavy_nodes.append(node)
    heavy_nodes.sort(key=lambda n: molecule.nodes[n]["mapping"][0][1])

    template_heavy = [p for p in template.particles() if symbol(p) != "H"]
    if len(template_heavy) != len(heavy_nodes):
        raise ValueError(
            f"Template for fragment '{fragname}' has {len(template_heavy)} "
            f"heavy atoms but the resolved fragment has {len(heavy_nodes)}."
        )

    # Explicit correspondence via particle tags: heavy atoms tagged with
    # their fragment atom index (e.g. mb.load("O{1}(C{0})C{2}")) are
    # matched by tag; untagged templates are matched by atom order.
    tag_indices = [_tag_as_index(particle) for particle in template_heavy]
    n_tagged = sum(1 for tag in tag_indices if tag is not None)
    if 0 < n_tagged < len(template_heavy):
        raise ValueError(
            f"Template for fragment '{fragname}' has fragment-index tags on "
            f"{n_tagged} of {len(template_heavy)} heavy atoms; tag all heavy "
            "atoms or none."
        )
    if n_tagged:
        if len(set(tag_indices)) != len(tag_indices):
            raise ValueError(
                f"Template for fragment '{fragname}' has duplicate "
                "fragment-index tags on its heavy atoms."
            )
        particle_by_index = dict(zip(tag_indices, template_heavy))
        pairs = []
        for node in heavy_nodes:
            local_index = molecule.nodes[node]["mapping"][0][1]
            particle = particle_by_index.get(local_index)
            if particle is None:
                raise ValueError(
                    f"Template for fragment '{fragname}' has no heavy atom "
                    f"tagged {{{local_index}}}; the fragment uses atom "
                    f"indices {sorted(molecule.nodes[n]['mapping'][0][1] for n in heavy_nodes)}."
                )
            pairs.append((node, particle))
    else:
        pairs = list(zip(heavy_nodes, template_heavy))

    coords = {}
    matched = {}
    for position, (node, particle) in enumerate(pairs):
        element = molecule.nodes[node]["element"]
        if element != symbol(particle):
            hint = (
                "Check the fragment-index tags."
                if n_tagged
                else "Template heavy atoms must appear in the same order as "
                "in the fragment SMILES, or be tagged with fragment atom "
                'indices (e.g. mb.load("O{1}(C{0})C{2}", smiles=True)).'
            )
            raise ValueError(
                f"Template for fragment '{fragname}' does not match: heavy "
                f"atom {position} is '{symbol(particle)}' but the fragment "
                f"expects '{element}'. {hint}"
            )
        coords[node] = np.asarray(particle.pos, dtype=float)
        matched[node] = particle

    # hydrogens available on each matched template heavy atom
    available_hydrogens = {particle: [] for particle in template_heavy}
    for particle_a, particle_b in template.bonds():
        if symbol(particle_a) == "H" and particle_b in available_hydrogens:
            available_hydrogens[particle_b].append(particle_a)
        elif symbol(particle_b) == "H" and particle_a in available_hydrogens:
            available_hydrogens[particle_a].append(particle_b)

    atom_set = set(atoms)
    for node in atoms:
        if molecule.nodes[node].get("element") != "H":
            continue
        parents = [
            neighbor
            for neighbor in molecule.neighbors(node)
            if neighbor in atom_set and neighbor in matched
        ]
        if not parents:
            raise ValueError(
                f"Cannot use a template for fragment '{fragname}': hydrogen "
                f"atom {node} is not bonded to a heavy atom in the fragment."
            )
        pool = available_hydrogens[matched[parents[0]]]
        if not pool:
            raise ValueError(
                f"Template for fragment '{fragname}' has too few hydrogens "
                f"on heavy atom {heavy_nodes.index(parents[0])}; the template "
                "must be at least as saturated as the resolved fragment."
            )
        coords[node] = np.asarray(pool.pop(0).pos, dtype=float)

    local_xyz = np.array([coords[node] for node in atoms])
    return local_xyz - local_xyz.mean(axis=0)


def _template_local_coords_cg(molecule, atoms, template, fragname):
    """Local coordinates for a coarse-grained fragment from a template.

    The resolved fragment's nodes are sub-beads (no element, no
    hydrogens). Each is matched to the template particle at the same
    fragment index, which is the ``mapping`` node attribute CGsmiles
    records; the template beads must therefore appear in the same order
    as in the fragment definition. Returns coordinates in nm, centered
    on the centroid of the fragment.
    """
    template_particles = list(template.particles())
    if len(atoms) != len(template_particles):
        raise ValueError(
            f"Template for fragment '{fragname}' has {len(template_particles)} "
            f"bead(s) but the resolved fragment has {len(atoms)}."
        )

    coords = {}
    for node in atoms:
        mapping = molecule.nodes[node].get("mapping")
        if not mapping or len(mapping) != 1:
            raise ValueError(
                f"Cannot use a template for fragment '{fragname}': bead "
                f"{node} has no unique fragment mapping (squashed fragments "
                "are not supported by template placement)."
            )
        index = mapping[0][1]
        if not 0 <= index < len(template_particles):
            raise ValueError(
                f"Template for fragment '{fragname}' has no bead at fragment "
                f"index {index}."
            )
        particle = template_particles[index]
        name = molecule.nodes[node].get("atomname")
        if name is not None and str(particle.name) != str(name):
            raise ValueError(
                f"Template for fragment '{fragname}' does not match: bead at "
                f"fragment index {index} is '{particle.name}' but the fragment "
                f"expects '{name}'. Template beads must appear in the same "
                "order as in the fragment definition."
            )
        coords[node] = np.asarray(particle.pos, dtype=float)

    local_xyz = np.array([coords[node] for node in atoms])
    return local_xyz - local_xyz.mean(axis=0)


def _embed_fragment(elements, charges, edges, seed):
    """Embed a single fragment with RDKit.

    The fragment is described by per-atom element symbols and formal
    charges plus (index, index, order) edge tuples. Atoms bonded to
    other fragments keep an open valence; RDKit treats them as radicals,
    which does not disturb the embedded geometry. Returns coordinates in
    nm, centered on the fragment centroid.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    bond_type_map = {
        0: Chem.BondType.ZERO,
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.QUADRUPLE,
        1.5: Chem.BondType.AROMATIC,
    }

    fragment = Chem.RWMol()
    for element, charge in zip(elements, charges):
        atom = Chem.Atom(element)
        atom.SetFormalCharge(charge)
        atom.SetNoImplicit(True)  # all H atoms are explicit after resolving
        fragment.AddAtom(atom)
    for u, v, order in edges:
        fragment.AddBond(u, v, bond_type_map.get(order, Chem.BondType.SINGLE))
        if order == 1.5:
            fragment.GetAtomWithIdx(u).SetIsAromatic(True)
            fragment.GetAtomWithIdx(v).SetIsAromatic(True)

    fragment = fragment.GetMol()
    Chem.SanitizeMol(fragment)
    if AllChem.EmbedMolecule(fragment, randomSeed=seed) != 0:
        if AllChem.EmbedMolecule(fragment, randomSeed=seed, useRandomCoords=True) != 0:
            raise ValueError(f"RDKit failed to embed fragment {elements}.")
    try:
        AllChem.UFFOptimizeMolecule(fragment)
    except Exception:  # UFF parameters can be missing; keep raw DG coordinates
        pass

    # Angstrom -> nm
    xyz = fragment.GetConformer().GetPositions() / 10.0
    return xyz - xyz.mean(axis=0)
