"""
Streamlit App for Molecule Decomposition

This app takes a SMILES string as input, decomposes it into ring and non-ring fragments,
displays the fragments with their 2D structures, and allows selection of one fragment.
"""

import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor, AllChem, rdFingerprintGenerator, QED, Descriptors, Crippen, rdDistGeom, rdFMCS
from rdkit import DataStructs, RDLogger
from typing import List, Dict, Set, Tuple
from itertools import permutations
from streamlit_ketcher import st_ketcher
import io
import base64
import pandas as pd
import mols2grid

# Page configuration
st.set_page_config(
    page_title="Molecule Decomposition",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# UNDESIRABLE SMARTS PATTERNS (Structural Alerts)
# ============================================================================
UNDESIRABLE_PATTERNS = [
    # Radioactive isotopes
    ('[18F]', 'Fluorine-18 (radioactive)'),
    ('[123I]', 'Iodine-123 (radioactive)'),
    # Peroxides and related
    ('[O]-[O]', 'Peroxide'),
    ('[O]-[O]-[O]', 'Ozonide'),
    ('C(=O)O[O]', 'Peroxycarboxylate'),
    ('C(=O)OO', 'Peroxyacid'),
    # Nitrogen-nitrogen bonds
    ('[n]-[N]', 'Connected Ring Nitrogens'), # [N] specifies a nitrogen atom. R0 is a SMARTS primitive that requires the atom to be in zero rings of a smallest set of smallest rings (SSSR) definition, which effectively ensures the bond connecting the two nitrogens is an exocyclic (non-ring) bond
    ('[N]-[N]', 'Hydrazine (N-N)'),
    ('[N]=[N]-[N]', 'Azide (N=N-N)'),
    # Disulfide
    ('[S]-[S]', 'Disulfide (S-S)'),
    # N-O bonds
    ('[n]-[O]','n-O bond'),
    ('[O]-[N]', 'O-N bond'),
    ('[N]-[O]', 'N-O bond'),
    # Acyl halides
    ('C(=O)Cl', 'Acyl Chloride'),
    ('C(=O)Br', 'Acyl Bromide'),
    ('C(=O)F', 'Acyl Fluoride'),
    # Sulfonyl chloride
    ('[S](=O)(=O)Cl', 'Sulfonyl Chloride'),
    # Phosphorus chlorides
    ('[P]Cl', 'Phosphorus Chloride'),
    ('P(=O)(Cl)(Cl)', 'Phosphoryl Dichloride'),
    ('P(Cl)(Cl)(Cl)', 'Phosphorus Trichloride'),
    # Mixed anhydride / acyl-O-alkyl with adjacent acyl
    ('C(=O)OC(=O)', 'Anhydride'),
    ('C(=O)O[C;!$(C=O)]', 'Acyl-O-alkyl (ester)'),
    # Aldehydes
    ('[CH]=O', 'Aldehyde'),
    ('[CX3H1](=O)[#6]', 'Aldehyde'),
    # Nitro groups
    ('[N+](=O)[O-]', 'Nitro group'),
    ('[NX3](=O)=O', 'Aromatic Nitro'),
    # Nitro adjacent to carbonyl
    ('[N+](=O)[O-]C(=O)', 'Nitro adjacent to carbonyl'),
    ('C(=O)C[N+](=O)[O-]', 'Carbonyl adjacent to nitro'),
    # Isocyanate and Isothiocyanate
    ('N=C=O', 'Isocyanate'),
    ('N=C=S', 'Isothiocyanate'),
    # Thiol
    ('[SH]', 'Thiol'),
    # Cyanohydrin motif (carbon with both OH and CN)
    ('[CH]([OH])(C#N)', 'Cyanohydrin'),
    ('C([OH])(C#N)', 'Cyanohydrin'),
    # Phenol
    ('c[OH]', 'Phenol'),
    ('[cH]O', 'Phenol'),
    # Michael acceptors (alpha,beta-unsaturated carbonyl)
    ('[#6]=[#6]-C(=O)', 'Michael acceptor'),
    ('C=CC(=O)', 'Michael acceptor (enone)'),
    ('C=CC(=O)[O,N]', 'Michael acceptor (acrylate/acrylamide)'),
    # Quinone-like (redox active)
    ('c1cc(=O)cc(=O)c1', 'Quinone (redox active)'),
    ('C1=CC(=O)C=CC1=O', 'Benzoquinone'),
    # # Catechol (redox active)
    # ('c1cc(O)c(O)cc1', 'Catechol (redox active)'),
    # ('c1ccc(O)c(O)c1', 'Catechol'),
    # Rhodanine-ish (PAINS)
    ('O=C1NC(=O)C=C1', 'Rhodanine-like (PAINS)'),
    ('O=C1NC(=S)SC1', 'Rhodanine'),
    ('O=C1NC(=O)SC1', 'Thiazolidinedione')
]

# ============================================================================
# DECOMPOSITION FUNCTIONS (from fragmentation.ipynb)
# ============================================================================

from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from typing import List, Dict, Set, Tuple
from IPython.display import display

def decompose_molecule_with_wildcards(mol: Chem.Mol, include_terminal_substituents: bool = True, 
                                       preserve_fused_rings: bool = True,
                                       max_terminal_atoms: int = 3) -> Dict[str, List[Dict]]:
    """
    Decompose a molecule into its individual rings (or fused ring systems) AND non-ring fragments,
    adding numbered wildcard dummy atoms ([*:1], [*:2], etc.) at each attachment point.
    
    Connected fragments will have matching dummy atom numbers indicating which pieces
    connect to each other.
    
    Terminal substituents (e.g., methyl groups, ethyl groups) that are only attached to the ring
    and not to any other functional groups can optionally be included as part of the ring,
    up to a specified number of heavy atoms.
    
    Fused/bicyclic ring systems can be preserved as single units.

    Args:
        mol: RDKit Mol object
        include_terminal_substituents: If True, include terminal groups as part of the ring
        preserve_fused_rings: If True, keep fused/bicyclic rings together as single units
        max_terminal_atoms: Maximum number of heavy atoms in terminal substituents to include (default: 3)

    Returns:
        Dict with two keys:
            - 'rings': List of ring fragment dicts
            - 'non_rings': List of non-ring fragment dicts
        
        Each dict contains:
            - 'base_smiles': SMILES of the fragment without wildcards
            - 'wildcard_smiles': SMILES with numbered [*:n] at attachment points (RDKit-readable)
            - 'frag_mol': RDKit Mol of the fragment with wildcards (for depiction)
            - 'atom_indices': tuple of atom indices in parent molecule
            - 'attachment_atoms': list of parent atom indices that are attachment points
            - 'size': number of heavy atoms (excluding wildcards)
            - 'hetero_count': number of heteroatoms
            - 'frag_type': 'ring', 'fused_ring', 'linker', 'terminal', etc.
    """
    if mol is None:
        return {'rings': [], 'non_rings': []}

    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    
    # Get all atoms that are part of any ring
    all_ring_atoms = set()
    for ring in atom_rings:
        all_ring_atoms.update(ring)
    
    # ============== PART 1: Process Ring Systems ==============
    
    # Group fused rings together if requested
    if preserve_fused_rings and atom_rings:
        ring_sets = [set(ring) for ring in atom_rings]
        
        merged = True
        while merged:
            merged = False
            new_ring_sets = []
            used = [False] * len(ring_sets)
            
            for i in range(len(ring_sets)):
                if used[i]:
                    continue
                current = ring_sets[i].copy()
                used[i] = True
                
                for j in range(i + 1, len(ring_sets)):
                    if used[j]:
                        continue
                    if current & ring_sets[j]:
                        current |= ring_sets[j]
                        used[j] = True
                        merged = True
                
                new_ring_sets.append(current)
            
            ring_sets = new_ring_sets
        
        ring_systems = [tuple(sorted(rs)) for rs in ring_sets]
    else:
        ring_systems = [tuple(ring) for ring in atom_rings] if atom_rings else []
    
    # First pass: collect all fragments and their atoms
    all_fragments = []  # List of (frag_type_category, ring_system_or_none, atom_set, is_fused)
    atoms_assigned_to_rings = set()

    for ring_system in ring_systems:
        ring_atoms = set(ring_system)
        is_fused = preserve_fused_rings and len(ring_system) > 6
        
        # Expand ring atoms to include terminal substituents if requested
        if include_terminal_substituents:
            expanded_atoms = set(ring_atoms)
            
            def get_terminal_chain(start_idx: int, from_atoms: Set[int], max_atoms: int) -> Set[int]:
                """
                Find a terminal chain starting from start_idx that:
                1. Does not connect to any ring atoms (other than through from_atoms)
                2. Does not branch into chains longer than max_atoms
                3. Has at most max_atoms heavy atoms total
                
                Returns set of atom indices in the terminal chain, or empty set if not terminal.
                """
                if start_idx in all_ring_atoms:
                    return set()
                
                # BFS to collect the entire connected component of non-ring atoms
                # reachable from start_idx without going through from_atoms
                chain = set()
                queue = [start_idx]
                visited = set(from_atoms)  # Don't revisit atoms we came from
                
                while queue:
                    current = queue.pop(0)
                    if current in visited:
                        continue
                    if current in all_ring_atoms:
                        # This chain connects to another ring - not terminal
                        return set()
                    
                    visited.add(current)
                    chain.add(current)
                    
                    # Check if we've exceeded max atoms
                    if len(chain) > max_atoms:
                        return set()
                    
                    atom = mol.GetAtomWithIdx(current)
                    for neighbor in atom.GetNeighbors():
                        nb_idx = neighbor.GetIdx()
                        if nb_idx not in visited:
                            queue.append(nb_idx)
                
                # Verify the chain is truly terminal (only connects back to from_atoms, not to other rings)
                for atom_idx in chain:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    for neighbor in atom.GetNeighbors():
                        nb_idx = neighbor.GetIdx()
                        if nb_idx not in chain and nb_idx not in from_atoms:
                            # Connected to something outside the chain and not the ring
                            if nb_idx in all_ring_atoms:
                                return set()  # Connects to another ring
                
                return chain
            
            # Check each atom adjacent to the ring for terminal substituents
            for ring_atom_idx in list(ring_atoms):
                atom = mol.GetAtomWithIdx(ring_atom_idx)
                for neighbor in atom.GetNeighbors():
                    nb_idx = neighbor.GetIdx()
                    if nb_idx in expanded_atoms:
                        continue
                    if nb_idx in all_ring_atoms:
                        continue
                    
                    # Try to get terminal chain starting from this neighbor
                    terminal_chain = get_terminal_chain(nb_idx, expanded_atoms, max_terminal_atoms)
                    if terminal_chain:
                        expanded_atoms.update(terminal_chain)
            
            ring_atoms_list = list(expanded_atoms)
        else:
            ring_atoms_list = list(ring_atoms)
        
        atoms_assigned_to_rings.update(ring_atoms_list)
        all_fragments.append(('ring', ring_system, set(ring_atoms_list), is_fused))
    
    # Get non-ring atoms
    all_atoms = set(range(mol.GetNumAtoms()))
    non_ring_atoms = all_atoms - atoms_assigned_to_rings
    
    # Find connected components among non-ring atoms
    if non_ring_atoms:
        visited = set()
        
        for start_atom in non_ring_atoms:
            if start_atom in visited:
                continue
            
            component = set()
            queue = [start_atom]
            
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                if current not in non_ring_atoms:
                    continue
                    
                visited.add(current)
                component.add(current)
                
                atom = mol.GetAtomWithIdx(current)
                for neighbor in atom.GetNeighbors():
                    nb_idx = neighbor.GetIdx()
                    if nb_idx in non_ring_atoms and nb_idx not in visited:
                        queue.append(nb_idx)
            
            if component:
                all_fragments.append(('non_ring', None, component, False))
    
    # ============== Build Bond-to-Number Mapping ==============
    # Find all bonds between different fragments and assign numbers
    
    bond_number_map = {}  # (atom1_idx, atom2_idx) -> number (with atom1 < atom2)
    current_bond_number = 1
    
    for i, (cat1, rs1, atoms1, _) in enumerate(all_fragments):
        for j, (cat2, rs2, atoms2, _) in enumerate(all_fragments):
            if i >= j:
                continue
            
            # Find bonds between fragment i and fragment j
            for a1 in atoms1:
                atom = mol.GetAtomWithIdx(a1)
                for neighbor in atom.GetNeighbors():
                    a2 = neighbor.GetIdx()
                    if a2 in atoms2:
                        bond_key = (min(a1, a2), max(a1, a2))
                        if bond_key not in bond_number_map:
                            bond_number_map[bond_key] = current_bond_number
                            current_bond_number += 1
    
    # ============== Process Each Fragment with Numbered Dummies ==============
    
    def process_fragment(atom_set: Set[int], frag_category: str, ring_system: Tuple = None, 
                        is_fused: bool = False) -> Dict:
        """Process a single fragment and return its info dict with numbered dummy atoms."""
        atom_list = list(atom_set)
        
        # Identify attachment bonds and their numbers
        attachment_info = []  # List of (internal_atom, external_atom, bond_number)
        for a_idx in atom_list:
            atom = mol.GetAtomWithIdx(a_idx)
            for neighbor in atom.GetNeighbors():
                nb_idx = neighbor.GetIdx()
                if nb_idx not in atom_set:
                    bond_key = (min(a_idx, nb_idx), max(a_idx, nb_idx))
                    bond_num = bond_number_map.get(bond_key, 0)
                    attachment_info.append((a_idx, nb_idx, bond_num))
        
        attachment_atoms = sorted(set(a for a, _, _ in attachment_info))
        
        # Determine fragment type for non-rings
        if frag_category == 'non_ring':
            if len(attachment_atoms) == 0:
                frag_type = 'isolated'
            elif len(attachment_atoms) == 1:
                frag_type = 'terminal'
            else:
                frag_type = 'linker'
        else:
            frag_type = 'fused_ring' if is_fused else 'ring'
        
        base_smi = Chem.MolFragmentToSmiles(mol, atom_list, canonical=True)
        
        # Get bonds to break with their dummy labels
        bonds_to_break = []
        dummy_labels = []  # List of (bond_idx, (label_for_begin, label_for_end))
        
        for a_idx, nb_idx, bond_num in attachment_info:
            bond = mol.GetBondBetweenAtoms(a_idx, nb_idx)
            if bond is not None:
                bond_idx = bond.GetIdx()
                if bond_idx not in [b for b, _ in dummy_labels]:
                    # Determine which end is inside our fragment
                    begin_idx = bond.GetBeginAtomIdx()
                    end_idx = bond.GetEndAtomIdx()
                    
                    if begin_idx in atom_set:
                        # Begin is inside, end is outside
                        # The dummy attached to begin gets the label
                        dummy_labels.append((bond_idx, (bond_num, bond_num)))
                    else:
                        # End is inside, begin is outside
                        dummy_labels.append((bond_idx, (bond_num, bond_num)))
                    
                    bonds_to_break.append(bond_idx)
        
        frag_mol = None
        wildcard_smi = None
        
        if bonds_to_break:
            try:
                # Create dummy labels list in bond order
                dummy_label_list = []
                for bond_idx in bonds_to_break:
                    for bi, (l1, l2) in dummy_labels:
                        if bi == bond_idx:
                            dummy_label_list.append((l1, l2))
                            break
                
                frag_mol_temp = Chem.FragmentOnBonds(mol, bonds_to_break, addDummies=True, 
                                                      dummyLabels=dummy_label_list)
                frags = Chem.GetMolFrags(frag_mol_temp, asMols=True, sanitizeFrags=False)
                frag_atom_lists = Chem.GetMolFrags(frag_mol_temp, asMols=False)
                
                target_frag = None
                for frag, frag_atoms in zip(frags, frag_atom_lists):
                    frag_atoms_set = set(frag_atoms)
                    
                    # Check if this fragment contains atoms from our atom_set
                    check_atoms = ring_system if ring_system else atom_list
                    if any(a_idx in frag_atoms_set for a_idx in check_atoms):
                        non_dummy_count = sum(1 for a in frag.GetAtoms() if a.GetAtomicNum() != 0)
                        if non_dummy_count == len(atom_list):
                            target_frag = frag
                            break
                
                if target_frag is not None:
                    frag_mol = target_frag
                    
                    # Convert isotope labels to atom map numbers for [*:n] format
                    rw = Chem.RWMol(frag_mol)
                    for atom in rw.GetAtoms():
                        if atom.GetAtomicNum() == 0:
                            isotope = atom.GetIsotope()
                            if isotope > 0:
                                atom.SetAtomMapNum(isotope)
                            atom.SetIsotope(0)
                    frag_mol = rw.GetMol()
                    
                    try:
                        Chem.SanitizeMol(frag_mol)
                    except:
                        try:
                            for atom in frag_mol.GetAtoms():
                                atom.SetIsAromatic(False)
                            for bond in frag_mol.GetBonds():
                                bond.SetIsAromatic(False)
                            Chem.SanitizeMol(frag_mol)
                        except:
                            frag_mol = None
            except Exception as e:
                frag_mol = None
        
        if frag_mol is None:
            try:
                frag_mol = Chem.MolFromSmiles(base_smi)
                wildcard_smi = base_smi
            except:
                return None
        
        if frag_mol is None:
            return None

        try:
            rdDepictor.Compute2DCoords(frag_mol)
        except:
            pass

        if wildcard_smi is None:
            try:
                wildcard_smi = Chem.MolToSmiles(frag_mol, canonical=True)
            except:
                wildcard_smi = base_smi
        
        test_mol = Chem.MolFromSmiles(wildcard_smi)
        if test_mol is None:
            wildcard_smi = base_smi

        hetero_count = sum(
            1 for a in frag_mol.GetAtoms()
            if a.GetAtomicNum() not in (0, 1, 6)
        )
        
        result = {
            'base_smiles': base_smi,
            'wildcard_smiles': wildcard_smi,
            'frag_mol': frag_mol,
            'atom_indices': tuple(atom_list),
            'attachment_atoms': attachment_atoms,
            'size': len(ring_system) if ring_system else len(atom_list),
            'hetero_count': hetero_count,
            'frag_type': frag_type
        }
        
        if ring_system:
            result['core_ring_atoms'] = ring_system
            result['total_atoms'] = len(atom_list)
        
        return result
    
    # Process all fragments
    ring_results = []
    non_ring_results = []
    seen_wildcard_smiles = set()
    
    for cat, ring_sys, atoms, is_fused in all_fragments:
        result = process_fragment(atoms, cat, ring_sys, is_fused)
        if result is None:
            continue
        
        if result['wildcard_smiles'] in seen_wildcard_smiles:
            continue
        seen_wildcard_smiles.add(result['wildcard_smiles'])
        
        if cat == 'ring':
            ring_results.append(result)
        else:
            non_ring_results.append(result)

    return {'rings': ring_results, 'non_rings': non_ring_results}


def decompose_to_smiles(mol: Chem.Mol, 
                        include_terminal_substituents: bool = True,
                        preserve_fused_rings: bool = True) -> List[str]:
    """
    Decompose a molecule into a list of fragment SMILES with numbered dummy atoms.
    """
    decomposition = decompose_molecule_with_wildcards(
        mol, 
        include_terminal_substituents=include_terminal_substituents,
        preserve_fused_rings=preserve_fused_rings
    )
    
    smiles_list = []
    
    for frag in decomposition.get('rings', []):
        smiles_list.append(frag['wildcard_smiles'])
    
    for frag in decomposition.get('non_rings', []):
        smiles_list.append(frag['wildcard_smiles'])
    
    return smiles_list


# ============================================================================
# FRAGMENT SIMILARITY SEARCH (from fragmentation.ipynb)
# ============================================================================

def find_similar_fragments(query_smiles: str, 
                           fragments_file: str,
                           similarity_threshold: float = 0.3,
                           top_n: int = 50) -> List[Tuple[str, float, int]]:
    """
    Find fragments similar to a query SMILES with the same number of attachment points.
    
    For queries with multiple attachment points, also filters for fragments with the same
    pairwise distances between attachment points. The returned fragments have their dummy 
    atoms renumbered to match the numbering scheme of the query molecule.
    
    Args:
        query_smiles: SMILES string of the query fragment (with [*:n] wildcards)
        fragments_file: Path to file containing fragment SMILES (one per line)
        similarity_threshold: Minimum Tanimoto similarity (0-1)
        top_n: Maximum number of results to return
    
    Returns:
        List of tuples: (smiles, similarity_score, num_attachments)
        Sorted by similarity score (highest first)
        SMILES will have renumbered dummy atoms matching query's numbering scheme
    """
    # Suppress RDKit warnings (including kekulization warnings)
    RDLogger.DisableLog('rdApp.*')
    
    def get_attachment_info(mol: Chem.Mol) -> List[Tuple[int, int, int]]:
        """Get information about attachment points."""
        info = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:  # Dummy atom
                dummy_idx = atom.GetIdx()
                map_num = atom.GetAtomMapNum()
                neighbors = atom.GetNeighbors()
                if neighbors:
                    neighbor_idx = neighbors[0].GetIdx()
                    info.append((dummy_idx, neighbor_idx, map_num))
        return info
    
    def get_distance_matrix(mol: Chem.Mol, attachment_info: List[Tuple[int, int, int]]) -> Dict[Tuple[int, int], int]:
        """Calculate pairwise distances between attachment points."""
        distances = {}
        n = len(attachment_info)
        for i in range(n):
            for j in range(i + 1, n):
                _, neighbor_i, map_i = attachment_info[i]
                _, neighbor_j, map_j = attachment_info[j]
                try:
                    path = Chem.GetShortestPath(mol, neighbor_i, neighbor_j)
                    if path:
                        dist = len(path) - 1
                        distances[(map_i, map_j)] = dist
                        distances[(map_j, map_i)] = dist
                except:
                    pass
        return distances
    
    def get_sorted_distances(mol: Chem.Mol) -> Tuple[int, ...]:
        """Calculate ALL pairwise distances between attachment points."""
        attachment_info = get_attachment_info(mol)
        if len(attachment_info) < 2:
            return ()
        
        distances = []
        for i in range(len(attachment_info)):
            for j in range(i + 1, len(attachment_info)):
                _, neighbor_i, _ = attachment_info[i]
                _, neighbor_j, _ = attachment_info[j]
                try:
                    path = Chem.GetShortestPath(mol, neighbor_i, neighbor_j)
                    if path:
                        distances.append(len(path) - 1)
                except:
                    pass
        return tuple(sorted(distances))
    
    def find_mapping(query_info: List[Tuple[int, int, int]], 
                     query_distances: Dict[Tuple[int, int], int],
                     frag_info: List[Tuple[int, int, int]], 
                     frag_mol: Chem.Mol) -> Dict[int, int]:
        """Find a mapping from fragment dummy map numbers to query dummy map numbers."""
        query_map_nums = [info[2] for info in query_info]
        frag_map_nums = [info[2] for info in frag_info]
        
        frag_distances = {}
        for i in range(len(frag_info)):
            for j in range(i + 1, len(frag_info)):
                _, neighbor_i, map_i = frag_info[i]
                _, neighbor_j, map_j = frag_info[j]
                try:
                    path = Chem.GetShortestPath(frag_mol, neighbor_i, neighbor_j)
                    if path:
                        dist = len(path) - 1
                        frag_distances[(map_i, map_j)] = dist
                        frag_distances[(map_j, map_i)] = dist
                except:
                    pass
        
        for perm in permutations(query_map_nums):
            mapping = dict(zip(frag_map_nums, perm))
            valid = True
            for i in range(len(frag_map_nums)):
                for j in range(i + 1, len(frag_map_nums)):
                    frag_m1, frag_m2 = frag_map_nums[i], frag_map_nums[j]
                    query_m1, query_m2 = mapping[frag_m1], mapping[frag_m2]
                    frag_dist = frag_distances.get((frag_m1, frag_m2))
                    query_dist = query_distances.get((query_m1, query_m2))
                    if frag_dist != query_dist:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return mapping
        return {}
    
    def renumber_fragment(smiles: str, mapping: Dict[int, int]) -> str:
        """Renumber the dummy atoms in a fragment SMILES according to the mapping."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        rw = Chem.RWMol(mol)
        for atom in rw.GetAtoms():
            if atom.GetAtomicNum() == 0:
                old_map = atom.GetAtomMapNum()
                if old_map in mapping:
                    atom.SetAtomMapNum(mapping[old_map])
        return Chem.MolToSmiles(rw.GetMol(), canonical=True)
    
    # Parse query molecule
    query_mol = Chem.MolFromSmiles(query_smiles)
    if query_mol is None:
        return []
    
    query_attachment_info = get_attachment_info(query_mol)
    query_attachments = len(query_attachment_info)
    query_sorted_distances = get_sorted_distances(query_mol) if query_attachments > 1 else ()
    query_distance_matrix = get_distance_matrix(query_mol, query_attachment_info) if query_attachments >= 3 else {}
    
    # Generate fingerprint for query (remove wildcards for fingerprint)
    query_mol_no_dummy = Chem.RWMol(query_mol)
    atoms_to_remove = [a.GetIdx() for a in query_mol_no_dummy.GetAtoms() if a.GetAtomicNum() == 0]
    for idx in sorted(atoms_to_remove, reverse=True):
        query_mol_no_dummy.RemoveAtom(idx)
    
    try:
        query_mol_clean = query_mol_no_dummy.GetMol()
        Chem.SanitizeMol(query_mol_clean)
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        query_fp = fpgen.GetFingerprint(query_mol_clean)
    except:
        return []
    
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    similar_fragments = []
    seen_canonical_smiles = set()
    
    with open(fragments_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            frag_mol = Chem.MolFromSmiles(line)
            if frag_mol is None:
                continue
            
            frag_attachment_info = get_attachment_info(frag_mol)
            num_attachments = len(frag_attachment_info)
            
            if num_attachments != query_attachments:
                continue
            
            if query_attachments > 1:
                frag_sorted_distances = get_sorted_distances(frag_mol)
                if frag_sorted_distances != query_sorted_distances:
                    continue
            
            frag_mol_no_dummy = Chem.RWMol(frag_mol)
            atoms_to_remove = [a.GetIdx() for a in frag_mol_no_dummy.GetAtoms() if a.GetAtomicNum() == 0]
            for idx in sorted(atoms_to_remove, reverse=True):
                frag_mol_no_dummy.RemoveAtom(idx)
            
            try:
                frag_mol_clean = frag_mol_no_dummy.GetMol()
                try:
                    Chem.SanitizeMol(frag_mol_clean)
                except:
                    try:
                        Chem.SanitizeMol(frag_mol_clean, catchErrors=True)
                    except:
                        continue
                
                canonical_smi = Chem.MolToSmiles(frag_mol_clean, canonical=True)
                if canonical_smi in seen_canonical_smiles:
                    continue
                
                frag_fp = fpgen.GetFingerprint(frag_mol_clean)
                similarity = DataStructs.TanimotoSimilarity(query_fp, frag_fp)
                
                if similarity >= similarity_threshold and similarity < 1.0:
                    output_smiles = line
                    
                    if query_attachments == 1:
                        query_map_num = query_attachment_info[0][2]
                        frag_map_num = frag_attachment_info[0][2]
                        if frag_map_num != query_map_num:
                            mapping = {frag_map_num: query_map_num}
                            output_smiles = renumber_fragment(line, mapping)
                    elif query_attachments == 2:
                        query_map_nums = [info[2] for info in query_attachment_info]
                        frag_map_nums = [info[2] for info in frag_attachment_info]
                        mapping = dict(zip(frag_map_nums, query_map_nums))
                        output_smiles = renumber_fragment(line, mapping)
                    elif query_attachments >= 3:
                        mapping = find_mapping(query_attachment_info, query_distance_matrix,
                                              frag_attachment_info, frag_mol)
                        if mapping:
                            output_smiles = renumber_fragment(line, mapping)
                    
                    similar_fragments.append((output_smiles, similarity, num_attachments))
                    seen_canonical_smiles.add(canonical_smi)
            except:
                continue
    
    RDLogger.EnableLog('rdApp.*')
    similar_fragments.sort(key=lambda x: x[1], reverse=True)
    return similar_fragments[:top_n]


# ============================================================================
# REASSEMBLY FUNCTION (from fragmentation.ipynb)
# ============================================================================

def reassemble_from_smiles(smiles_list: List[str]) -> Chem.Mol:
    """
    Reassemble a molecule from fragment SMILES with numbered dummy atoms.
    
    Fragments are connected by matching their dummy atom numbers.
    For example, [*:1] in one fragment connects to [*:1] in another fragment.
    
    Args:
        smiles_list: List of SMILES strings with [*:n] wildcard notation
    
    Returns:
        RDKit Mol object of the reassembled molecule, or None if failed
    """
    if not smiles_list:
        return None
    
    # Parse all SMILES into molecules
    mols = []
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            mols.append(m)
    
    if not mols:
        return None
    
    # Single fragment - just remove dummy atoms
    if len(mols) == 1:
        rw = Chem.RWMol(mols[0])
        atoms_to_remove = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]
        for idx in sorted(atoms_to_remove, reverse=True):
            rw.RemoveAtom(idx)
        mol = rw.GetMol()
        try:
            Chem.SanitizeMol(mol)
        except:
            pass
        return mol
    
    # Combine all molecules into one disconnected molecule
    combined = mols[0]
    for m in mols[1:]:
        combined = Chem.CombineMols(combined, m)
    
    rw = Chem.RWMol(combined)
    
    # Find all dummy atoms and group by their map number
    dummy_map = {}  # map_num -> [(dummy_idx, neighbor_idx, bond_type)]
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() == 0:  # Dummy atom (*)
            map_num = atom.GetAtomMapNum()
            if map_num > 0:
                neighbors = atom.GetNeighbors()
                if neighbors:
                    neighbor_idx = neighbors[0].GetIdx()
                    bond = rw.GetBondBetweenAtoms(atom.GetIdx(), neighbor_idx)
                    bond_type = bond.GetBondType() if bond else Chem.BondType.SINGLE
                    
                    if map_num not in dummy_map:
                        dummy_map[map_num] = []
                    dummy_map[map_num].append((atom.GetIdx(), neighbor_idx, bond_type))
    
    # Connect fragments by joining atoms with matching dummy numbers
    atoms_to_remove = set()
    
    for map_num, dummy_list in dummy_map.items():
        if len(dummy_list) >= 2:
            # Connect the two fragments with this map number
            dummy1_idx, real1_idx, bond_type = dummy_list[0]
            dummy2_idx, real2_idx, _ = dummy_list[1]
            
            # Create bond between the real atoms
            if rw.GetBondBetweenAtoms(real1_idx, real2_idx) is None:
                rw.AddBond(real1_idx, real2_idx, bond_type)
            
            # Mark dummies for removal
            atoms_to_remove.add(dummy1_idx)
            atoms_to_remove.add(dummy2_idx)
    
    # Remove matched dummy atoms
    for idx in sorted(atoms_to_remove, reverse=True):
        rw.RemoveAtom(idx)
    
    # Remove any remaining unmatched dummies
    remaining = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]
    for idx in sorted(remaining, reverse=True):
        rw.RemoveAtom(idx)
    
    mol = rw.GetMol()
    
    # Sanitize the molecule
    try:
        Chem.SanitizeMol(mol)
    except:
        try:
            for atom in mol.GetAtoms():
                atom.SetIsAromatic(False)
            for bond in mol.GetBonds():
                bond.SetIsAromatic(False)
            Chem.SanitizeMol(mol)
        except:
            pass
    
    # Generate 2D coordinates
    try:
        rdDepictor.Compute2DCoords(mol)
    except:
        pass
    
    return mol


# ============================================================================
# STREAMLIT APP
# ============================================================================

st.title("💧 Morpheus: A bioisostere and R-group replacement tool")
st.markdown("Decompose molecules into ring and non-ring fragments with wildcard attachment points.")

# Example molecules
examples = {
    "-- Select an example --": "",
    "Imatinib": "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
    "Gefitinib": "COc1cc2ncnc(c2cc1OCCCN1CCOCC1)Nc1ccc(c(c1)Cl)F",
    "Dasatinib": "CC1=C(C(=CC=C1)Cl)NC(=O)C2=CN=C(S2)NC3=CC(=NC(=N3)C)N4CCN(CC4)CCO",
    "Maraviroc": "CC1=NN=C(N1C2C[C@H]3CC[C@@H](C2)N3CC[C@@H](C4=CC=CC=C4)NC(=O)C5CCC(CC5)(F)F)C(C)C",
    "Roniciclib": "C[C@H]([C@@H](C)OC1=NC(=NC=C1C(F)(F)F)NC2=CC=C(C=C2)[S@](=N)(=O)C3CC3)O",
    "GV134": "FC1(F)CN(C1)C(=O)C=2N(C)c3cc(ccc3C2)c4nccc(n4)N5CC[C@@H](C5)C=6C=NNC6"
}

# Default options (enabled by default)
include_terminal = True
preserve_fused = True

# Initialize session state
if 'smiles_input' not in st.session_state:
    st.session_state.smiles_input = ""
if 'selected_idx' not in st.session_state:
    st.session_state.selected_idx = 0
if 'last_smiles' not in st.session_state:
    st.session_state.last_smiles = ""
if 'similar_fragments' not in st.session_state:
    st.session_state.similar_fragments = None
if 'last_selected_for_replace' not in st.session_state:
    st.session_state.last_selected_for_replace = None

# Main input with example dropdown
col_input, col_example = st.columns([3, 1])

def on_example_change():
    """Callback when example dropdown changes."""
    selected = st.session_state.example_dropdown
    if selected != "-- Select an example --" and examples.get(selected):
        st.session_state.smiles_text_input = examples[selected]
        st.session_state.smiles_input = examples[selected]
        st.session_state.last_smiles = ""  # Force reset on next check

with col_example:
    selected_example = st.selectbox(
        "Examples:",
        options=list(examples.keys()),
        index=0,
        key="example_dropdown",
        on_change=on_example_change
    )

with col_input:
    smiles_input = st.text_input(
        "Enter SMILES string:",
        placeholder="",
        key="smiles_text_input"
    )

# Ketcher molecule sketcher expander (always visible)
with st.expander("✏️ Draw Molecule (Ketcher)", expanded=False):
    st.markdown("*Draw a molecule using the Ketcher editor. The SMILES will appear below - copy it to the input field above:*")
    
    # Display ketcher with current SMILES as default (or empty)
    ketcher_smiles = st_ketcher(smiles_input if smiles_input else "", height=500)
    
    # Display the SMILES from the sketcher
    if ketcher_smiles:
        st.code(ketcher_smiles, language="text")
        st.caption("👆 Copy this SMILES and paste it into the input field above")

# Sync text input with session state
if smiles_input:
    st.session_state.smiles_input = smiles_input

# Reset selection only if the actual molecule changed (not just on every rerun)
if smiles_input and smiles_input != st.session_state.last_smiles:
    st.session_state.selected_idx = 0
    st.session_state.last_smiles = smiles_input
    st.session_state.similar_fragments = None
    st.session_state.last_selected_for_replace = None

if smiles_input:
    # Parse molecule
    mol = Chem.MolFromSmiles(smiles_input)
    
    if mol is None:
        st.error("❌ Invalid SMILES string. Please enter a valid molecule.")
    else:
        # Display input molecule
        st.subheader("Input Molecule")
        col1, col2, col3 = st.columns([1.5, 2, 1], gap="medium")
        
        with col1:
            img = Draw.MolToImage(mol, size=(600, 600))
            st.markdown("**2D Structure:**")
            st.image(img, caption="")
        
        with col3:
            st.markdown("**Molecule Info:**")
            #st.markdown(f"**SMILES:** `{smiles_input}`")
            st.markdown(f"**Canonical SMILES:** `{Chem.MolToSmiles(mol, canonical=True)}`")
            st.markdown(f"**Heavy atoms:** {mol.GetNumHeavyAtoms()}")
            st.markdown(f"**Rings:** {mol.GetRingInfo().NumRings()}")
            st.markdown(f"**QED:** {QED.qed(mol):.3f}")
            st.markdown(f"**TPSA:** {Descriptors.TPSA(mol):.2f} Å²")
            st.markdown(f"**cLogP:** {Crippen.MolLogP(mol):.2f}")
            
            # Check for undesirable substructures in input molecule
            input_has_undesirable = False
            input_undesirable_patterns = []
            for smarts, name in UNDESIRABLE_PATTERNS:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern and mol.HasSubstructMatch(pattern):
                    input_has_undesirable = True
                    input_undesirable_patterns.append(name)
            
            if input_has_undesirable:
                st.warning(f"⚠️ **Warning:** Input molecule contains undesirable substructure(s): {', '.join(set(input_undesirable_patterns))}")
        
        with col2:
            # Generate 3D structure
            mol_3d = Chem.AddHs(mol)
            try:
                # Generate 3D conformer
                result = rdDistGeom.EmbedMolecule(mol_3d, randomSeed=42)
                if result == 0:  # Success
                    # Optimize the conformer
                    AllChem.MMFFOptimizeMolecule(mol_3d)
                    
                    # Generate 3D view using py3Dmol
                    st.markdown("**3D Structure:**")
                    
                    # Get molecule block and escape for JavaScript
                    mol_block = Chem.MolToMolBlock(mol_3d)
                    # Escape backticks and special characters
                    mol_block_escaped = mol_block.replace('`', '\\`').replace('$', '\\$')
                    
                    # Create py3Dmol viewer HTML
                    viewer_html = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
                    </head>
                    <body>
                        <div id="3dmol_viewer" style="width: 100%; height: 400px; position: relative;"></div>
                        <script>
                            var element = document.getElementById('3dmol_viewer');
                            var viewer = $3Dmol.createViewer(element, {{backgroundColor: 'white'}});
                            var moldata = `{mol_block_escaped}`;
                            viewer.addModel(moldata, "sdf");
                            viewer.setStyle({{}}, {{stick: {{radius: 0.2}}}});
                            viewer.zoomTo();
                            viewer.render();
                        </script>
                    </body>
                    </html>
                    """
                    components.html(viewer_html, height=420, scrolling=False)
                else:
                    st.info("3D conformer generation failed.")
            except Exception as e:
                st.info(f"3D structure not available: {str(e)}")
        
        st.markdown("---")
        
        # Decompose molecule
        decomposition = decompose_molecule_with_wildcards(
            mol, 
            include_terminal_substituents=include_terminal,
            preserve_fused_rings=preserve_fused
        )
        
        fragments = decompose_to_smiles(mol, include_terminal, preserve_fused)
        
        if not fragments:
            st.warning("No fragments found. The molecule may be too simple to decompose.")
        else:
            st.subheader(f"Fragments ({len(fragments)} total)")
            st.markdown("**Select the fragment that you wish to replace-**")
            
            # Create fragment data
            all_frags = decomposition['rings'] + decomposition['non_rings']
            
            # Ensure selected index is valid
            if st.session_state.selected_idx >= len(all_frags):
                st.session_state.selected_idx = 0
            
            # Display fragments in grid with clickable buttons
            cols_per_row = 6
            for row_start in range(0, len(all_frags), cols_per_row):
                cols = st.columns(cols_per_row)
                for col_idx, frag_idx in enumerate(range(row_start, min(row_start + cols_per_row, len(all_frags)))):
                    frag = all_frags[frag_idx]
                    with cols[col_idx]:
                        # Check if this fragment is selected
                        is_selected = (frag_idx == st.session_state.selected_idx)
                        
                        # Create image
                        frag_mol = Chem.MolFromSmiles(frag['wildcard_smiles'])
                        if frag_mol:
                            img = Draw.MolToImage(frag_mol, size=(300, 300))
                            
                            # Convert image to bytes for button
                            img_buffer = io.BytesIO()
                            img.save(img_buffer, format='PNG')
                            img_bytes = img_buffer.getvalue()
                            
                            # Style for selected vs unselected
                            if is_selected:
                                st.markdown(
                                    f"""<div style="border: 4px solid #1f77b4; border-radius: 10px; 
                                    padding: 5px; background-color: rgba(31, 119, 180, 0.2);">
                                    <p style="text-align: center; margin: 0; font-weight: bold; color: #1f77b4;">
                                    ✓ Selected</p></div>""",
                                    unsafe_allow_html=True
                                )
                            
                        # Clickable image button
                        if st.button(
                            f"Fragment {frag_idx + 1}",
                            key=f"frag_btn_{frag_idx}",
                            width='stretch',
                            type="primary" if is_selected else "secondary"
                        ):
                            st.session_state.selected_idx = frag_idx
                            st.rerun()
                        
                        # Display image
                        st.image(img, width='stretch')
                        
                        # Fragment info
                        #st.caption(f"**{frag['frag_type']}**")
                        st.code(frag['wildcard_smiles'], language=None)
            
            # Get the selected index
            selected = st.session_state.selected_idx
            
            #st.markdown("---")
            
            # Show selected fragment details
            
            selected_frag = all_frags[selected]
                
            # Replace button
            st.markdown("")
            if st.button("🔄 Replace", type="primary", width='stretch'):
                with st.spinner("Searching for similar fragments..."):
                    similar = find_similar_fragments(
                        selected_frag['wildcard_smiles'],
                        "data/kinase_inhibitors_fragments.txt",
                        similarity_threshold=0.2,
                        top_n=50
                    )
                    st.session_state.similar_fragments = similar
                    st.session_state.last_selected_for_replace = selected
            
            # Reset similar fragments if selected fragment changed
            if st.session_state.last_selected_for_replace is not None and st.session_state.last_selected_for_replace != selected:
                st.session_state.similar_fragments = None
                st.session_state.last_selected_for_replace = None
            
            # Display similar fragments if available
            if st.session_state.similar_fragments is not None:
                st.markdown("---")
                
                similar_frags = st.session_state.similar_fragments
                
                if not similar_frags:
                    st.warning("No similar fragments found with matching attachment points and distances.")
                else:
                    # Similar fragments in collapsed expander
                    with st.expander(f"🔍 Similar Replacement Fragments ({len(similar_frags)} found)", expanded=False):
                        # Display similar fragments in a grid
                        cols_per_row = 6
                        for row_start in range(0, len(similar_frags), cols_per_row):
                            cols = st.columns(cols_per_row)
                            for col_idx, frag_idx in enumerate(range(row_start, min(row_start + cols_per_row, len(similar_frags)))):
                                smiles, similarity, n_attach = similar_frags[frag_idx]
                                with cols[col_idx]:
                                    sim_mol = Chem.MolFromSmiles(smiles)
                                    if sim_mol:
                                        img = Draw.MolToImage(sim_mol, size=(300, 300))
                                        st.image(img, width='stretch')
                                        st.markdown(f"**Similarity:** {similarity:.3f}")
                                        st.code(smiles, language=None)
                    
                    # Reassemble molecules with replacement fragments
                    #st.markdown("---")
                    st.subheader("🧬 New Molecules")
                    st.markdown("*Original molecule with selected fragment replaced by each similar fragment:*")
                    
                    # Get the list of all fragment SMILES
                    all_frag_smiles = [f['wildcard_smiles'] for f in all_frags]
                    
                    # Prepare the reference molecule with 2D coordinates for alignment
                    ref_mol = Chem.MolFromSmiles(smiles_input)
                    AllChem.Compute2DCoords(ref_mol)
                    
                    # Generate fingerprint for the input molecule for similarity comparison
                    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
                    ref_fp = fpgen.GetFingerprint(ref_mol)
                    
                    # Check if input molecule has undesirable patterns
                    input_has_undesirable = False
                    for smarts, name in UNDESIRABLE_PATTERNS:
                        pattern = Chem.MolFromSmarts(smarts)
                        if pattern and ref_mol.HasSubstructMatch(pattern):
                            input_has_undesirable = True
                            break
                    
                    # Pre-compile SMARTS patterns (using global UNDESIRABLE_PATTERNS)
                    compiled_patterns = []
                    for smarts, name in UNDESIRABLE_PATTERNS:
                        pattern = Chem.MolFromSmarts(smarts)
                        if pattern:
                            compiled_patterns.append((pattern, name))
                    
                    def has_undesirable_substructure(mol):
                        """Check if molecule contains any undesirable substructures."""
                        for pattern, name in compiled_patterns:
                            if mol.HasSubstructMatch(pattern):
                                return True, name
                        return False, None
                    
                    # Reassemble molecules with each replacement
                    reassembled_mols = []
                    reassembled_info = []
                    filtered_info = []  # Track filtered molecules
                    seen_smiles = set()  # Track unique molecules
                    
                    for sim_smiles, similarity, _ in similar_frags:
                        # Create new fragment list with replacement
                        new_frag_list = all_frag_smiles.copy()
                        new_frag_list[selected] = sim_smiles
                        
                        # Reassemble
                        new_mol = reassemble_from_smiles(new_frag_list)
                        if new_mol is not None:
                            # Get canonical SMILES for duplicate check
                            new_smiles = Chem.MolToSmiles(new_mol, canonical=True)
                            
                            # Skip if we've already seen this molecule
                            if new_smiles in seen_smiles:
                                continue
                            seen_smiles.add(new_smiles)
                            
                            # Check for undesirable substructures (only filter if input doesn't have them)
                            if not input_has_undesirable:
                                has_bad, bad_pattern_name = has_undesirable_substructure(new_mol)
                                if has_bad:
                                    # Store filtered molecule info
                                    filtered_info.append({
                                        'mol': new_mol,
                                        'smiles': new_smiles,
                                        'replacement_frag': sim_smiles,
                                        'filter_reason': bad_pattern_name
                                    })
                                    continue  # Skip this molecule
                            
                            # Calculate Tanimoto similarity to input molecule
                            try:
                                new_mol_fp = fpgen.GetFingerprint(new_mol)
                                mol_similarity = DataStructs.TanimotoSimilarity(ref_fp, new_mol_fp)
                            except:
                                mol_similarity = None
                            
                            # Align the reassembled molecule to match the input molecule's orientation
                            try:
                                # GenerateDepictionMatching2DStructure(mol_to_align, reference_mol)
                                AllChem.GenerateDepictionMatching2DStructure(new_mol,ref_mol)
                            except:
                                # If alignment fails, just compute regular 2D coords
                                try:
                                    AllChem.Compute2DCoords(new_mol)
                                except:
                                    pass
                            
                            reassembled_mols.append(new_mol)
                            reassembled_info.append({
                                'mol': new_mol,
                                'smiles': new_smiles,
                                'replacement_frag': sim_smiles,
                                'frag_similarity': similarity,
                                'mol_similarity': mol_similarity
                            })
                    
                    if reassembled_mols:
                        st.success(f"Replacement successful! ({len(reassembled_mols)} molecules, {len(filtered_info)} filtered out)")
                        
                        # Sort reassembled_info by Tanimoto similarity (descending)
                        reassembled_info.sort(key=lambda x: x['mol_similarity'] if x['mol_similarity'] is not None else -1, reverse=True)
                        
                        # Create DataFrame for mols2grid
                        df_data = []
                        
                        for mol_idx, info in enumerate(reassembled_info):
                            mol = info['mol']
                            replacement_smiles = info['replacement_frag']
                            
                            # Calculate QED score
                            try:
                                qed_score = QED.qed(mol)
                            except:
                                qed_score = None
                            
                            # Calculate TPSA
                            try:
                                tpsa = Descriptors.TPSA(mol)
                            except:
                                tpsa = None
                            
                            df_data.append({
                                'SMILES': info['smiles'],
                                'Tanimoto_Sim': f"{info['mol_similarity']:.3f}" if info['mol_similarity'] is not None else "N/A",
                                'QED': f"{qed_score:.3f}" if qed_score is not None else "N/A",
                                'TPSA': f"{tpsa:.2f}" if tpsa is not None else "N/A",
                                'Replacement_Fragment': replacement_smiles,
                                'ID': mol_idx + 1
                            })
                        
                        df = pd.DataFrame(df_data)
                        
                        # Display new molecules in an expander
                        with st.expander(f"🧬 New Molecules ({len(reassembled_mols)} molecules)", expanded=True):
                            # Property selection
                            available_properties = ['ID', 'SMILES', 'Tanimoto_Sim', 'QED', 'TPSA']
                            selected_properties = st.multiselect(
                                "Select properties to display:",
                                options=available_properties,
                                default=['ID', 'Tanimoto_Sim'],
                                key="property_selector"
                            )
                            
                            # Ensure at least one property is selected
                            if not selected_properties:
                                st.warning("Please select at least one property to display.")
                                selected_properties = ['ID', 'Tanimoto_Sim']
                            
                            # Build subset list (always include 'img')
                            subset = ['img'] + selected_properties
                            
                            # Build tooltip list (all properties)
                            tooltip_properties = ['ID', 'SMILES', 'Tanimoto_Sim', 'QED', 'TPSA', 'Replacement_Fragment']
                            
                            # Only transform properties that are in the subset (displayed as legends)
                            transform_dict = {}
                            if 'ID' in selected_properties:
                                transform_dict['ID'] = lambda x: f"ID: {x}"
                            if 'SMILES' in selected_properties:
                                transform_dict['SMILES'] = lambda x: f"SMILES: <span style='font-size:9px;word-break:break-all;'>{x}</span>"
                            if 'Tanimoto_Sim' in selected_properties:
                                transform_dict['Tanimoto_Sim'] = lambda x: f"Tanimoto Sim: {x}"
                            if 'QED' in selected_properties:
                                transform_dict['QED'] = lambda x: f"QED: {x}"
                            if 'TPSA' in selected_properties:
                                transform_dict['TPSA'] = lambda x: f"TPSA: {x} Å²"
                            
                            # Display using mols2grid
                            raw_html = mols2grid.display(
                                df,
                                smiles_col='SMILES',
                                subset=subset,
                                tooltip=tooltip_properties,
                                size=(300, 300),
                                n_items_per_page=20,
                                prerender=True,
                                transform=transform_dict if transform_dict else None
                            )._repr_html_()
                            
                            # Dynamic height based on number of molecules
                            num_mols = len(reassembled_mols)
                            if num_mols <= 8:
                                grid_height = 800
                            elif num_mols > 16:
                                grid_height = 1900
                            else:
                                grid_height = 1200  # Middle ground for 9-16 molecules
                            
                            # Render in Streamlit
                            components.html(raw_html, height=grid_height, scrolling=True)
                        
                        # 3D Comparison Viewer
                        with st.expander("🔬 3D Structure Comparison", expanded=False):
                            st.markdown("*Select a reassembled molecule to compare its 3D structure with the input molecule:*")
                            
                            # Initialize session state for selected 3D molecule
                            if 'selected_3d_mol_idx' not in st.session_state:
                                st.session_state.selected_3d_mol_idx = 0
                            
                            # Create two columns - small scrollable list on left, 3D viewer on right
                            col_list, col_viewer = st.columns([1, 3])
                            
                            with col_list:
                                st.markdown("**Select molecule:**")
                                # Create a scrollable container with molecule buttons
                                # Use custom CSS to make the container scrollable with fixed height matching viewer
                                st.markdown("""
                                    <style>
                                    div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"]:has(> div.scrollable-mol-list) {
                                        max-height: 550px;
                                        overflow-y: auto;
                                        padding-right: 10px;
                                    }
                                    </style>
                                """, unsafe_allow_html=True)
                                
                                # Wrap buttons in a div with marker class
                                st.markdown('<div class="scrollable-mol-list"></div>', unsafe_allow_html=True)
                                
                                with st.container(height=550):
                                    for idx, info in enumerate(reassembled_info):
                                        # Create button for each molecule
                                        is_selected = (idx == st.session_state.selected_3d_mol_idx)
                                        if st.button(
                                            f"Mol {idx + 1} (Sim: {info['mol_similarity']:.3f})" if info['mol_similarity'] else f"Mol {idx + 1}",
                                            key=f"mol3d_btn_{idx}",
                                            type="primary" if is_selected else "secondary",
                                            width='stretch'
                                        ):
                                            st.session_state.selected_3d_mol_idx = idx
                                            st.rerun()
                                        
                                        # Display 2D structure of the replacement fragment below the button
                                        replacement_mol = Chem.MolFromSmiles(info['replacement_frag'])
                                        if replacement_mol:
                                            frag_img = Draw.MolToImage(replacement_mol, size=(250, 250))
                                            st.image(frag_img, width='stretch')
                                            st.markdown("----")
                            
                            with col_viewer:
                                selected_3d_idx = st.session_state.selected_3d_mol_idx
                                if selected_3d_idx < len(reassembled_info):
                                    selected_reassembled = reassembled_info[selected_3d_idx]
                                    selected_reassembled_mol = selected_reassembled['mol']
                                    
                                    try:
                                        # Number of conformers to sample for better alignment
                                        num_conformers = 25
                                        
                                        # Generate multiple 3D conformers for both molecules
                                        # Input molecule
                                        mol_input_3d = Chem.AddHs(Chem.MolFromSmiles(smiles_input))
                                        params_input = rdDistGeom.ETKDGv3()
                                        params_input.randomSeed = 42
                                        params_input.numThreads = 0  # Use all available threads
                                        params_input.useSmallRingTorsions = True
                                        params_input.useMacrocycleTorsions = True
                                        cids_input = rdDistGeom.EmbedMultipleConfs(mol_input_3d, numConfs=num_conformers, params=params_input)
                                        
                                        # Optimize all conformers
                                        if len(cids_input) > 0:
                                            for cid in cids_input:
                                                AllChem.MMFFOptimizeMolecule(mol_input_3d, confId=cid, maxIters=50)
                                        
                                        # Reassembled molecule
                                        mol_reassembled_3d = Chem.AddHs(Chem.MolFromSmiles(selected_reassembled['smiles']))
                                        params_reassembled = rdDistGeom.ETKDGv3()
                                        params_reassembled.randomSeed = 42
                                        params_reassembled.numThreads = 0
                                        params_reassembled.useSmallRingTorsions = True
                                        params_reassembled.useMacrocycleTorsions = True
                                        cids_reassembled = rdDistGeom.EmbedMultipleConfs(mol_reassembled_3d, numConfs=num_conformers, params=params_reassembled)
                                        
                                        # Optimize all conformers
                                        if len(cids_reassembled) > 0:
                                            for cid in cids_reassembled:
                                                AllChem.MMFFOptimizeMolecule(mol_reassembled_3d, confId=cid, maxIters=50)
                                        
                                        if len(cids_input) > 0 and len(cids_reassembled) > 0:
                                            # Find MCS for alignment
                                            mcs_result = rdFMCS.FindMCS([Chem.RemoveHs(mol_input_3d), Chem.RemoveHs(mol_reassembled_3d)],
                                                                        timeout=10,
                                                                        completeRingsOnly=True,
                                                                        bondCompare=rdFMCS.BondCompare.CompareAny,
                                                                        atomCompare=rdFMCS.AtomCompare.CompareAny)
                                            
                                            best_rmsd = float('inf')
                                            best_input_conf = 0
                                            best_reassembled_conf = 0
                                            atom_map = None
                                            
                                            if mcs_result.numAtoms > 0:
                                                # Get MCS as SMARTS and create pattern
                                                mcs_smarts = mcs_result.smartsString
                                                mcs_mol = Chem.MolFromSmarts(mcs_smarts)
                                                
                                                if mcs_mol:
                                                    # Get atom mappings for alignment (using heavy atom indices)
                                                    mol_input_noH = Chem.RemoveHs(mol_input_3d)
                                                    mol_reassembled_noH = Chem.RemoveHs(mol_reassembled_3d)
                                                    
                                                    match_input = mol_input_noH.GetSubstructMatch(mcs_mol)
                                                    match_reassembled = mol_reassembled_noH.GetSubstructMatch(mcs_mol)
                                                    
                                                    if match_input and match_reassembled:
                                                        # Map heavy atom indices to indices in H-added molecule
                                                        # Build mapping from heavy atom index to full molecule index
                                                        def get_heavy_to_full_map(mol_with_H):
                                                            heavy_to_full = {}
                                                            heavy_idx = 0
                                                            for atom in mol_with_H.GetAtoms():
                                                                if atom.GetAtomicNum() != 1:  # Not hydrogen
                                                                    heavy_to_full[heavy_idx] = atom.GetIdx()
                                                                    heavy_idx += 1
                                                            return heavy_to_full
                                                        
                                                        h2f_input = get_heavy_to_full_map(mol_input_3d)
                                                        h2f_reassembled = get_heavy_to_full_map(mol_reassembled_3d)
                                                        
                                                        # Create atom map using full molecule indices
                                                        atom_map = []
                                                        for i, (idx_r, idx_i) in enumerate(zip(match_reassembled, match_input)):
                                                            full_idx_r = h2f_reassembled.get(idx_r, idx_r)
                                                            full_idx_i = h2f_input.get(idx_i, idx_i)
                                                            atom_map.append((full_idx_r, full_idx_i))
                                                        
                                                        # Find the best conformer pair with lowest RMSD
                                                        for cid_input in cids_input:
                                                            for cid_reassembled in cids_reassembled:
                                                                try:
                                                                    rmsd = AllChem.AlignMol(mol_reassembled_3d, mol_input_3d,
                                                                                           prbCid=cid_reassembled,
                                                                                           refCid=cid_input,
                                                                                           atomMap=atom_map)
                                                                    if rmsd < best_rmsd:
                                                                        best_rmsd = rmsd
                                                                        best_input_conf = cid_input
                                                                        best_reassembled_conf = cid_reassembled
                                                                except:
                                                                    continue
                                                        
                                                        # Final alignment with best conformers
                                                        if atom_map and best_rmsd < float('inf'):
                                                            AllChem.AlignMol(mol_reassembled_3d, mol_input_3d,
                                                                           prbCid=best_reassembled_conf,
                                                                           refCid=best_input_conf,
                                                                           atomMap=atom_map)
                                            
                                            # Get mol blocks for the best conformers
                                            mol_block_input = Chem.MolToMolBlock(mol_input_3d, confId=best_input_conf)
                                            mol_block_reassembled = Chem.MolToMolBlock(mol_reassembled_3d, confId=best_reassembled_conf)
                                            
                                            # Escape for JavaScript
                                            mol_block_input_escaped = mol_block_input.replace('`', '\\`').replace('$', '\\$')
                                            mol_block_reassembled_escaped = mol_block_reassembled.replace('`', '\\`').replace('$', '\\$')
                                            # Create dual molecule 3D viewer
                                            viewer_html = f"""
                                            <!DOCTYPE html>
                                            <html>
                                            <head>
                                                <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
                                            </head>
                                            <body>
                                                <div style="margin-bottom: 10px; color: #333;">
                                                    <span style="display: inline-block; width: 15px; height: 15px; background-color: #3498db; border: 2px solid #3498db; margin-right: 5px;"></span>
                                                    <span style="margin-right: 20px; color: #3498db; font-weight: bold;">Input Molecule (blue carbons)</span>
                                                    <span style="display: inline-block; width: 15px; height: 15px; background-color: #e74c3c; border: 2px solid #e74c3c; margin-right: 5px;"></span>
                                                    <span style="color: #e74c3c; font-weight: bold;">Analog Molecule (red carbons) (ID: {selected_3d_idx + 1})</span>
                                                </div>
                                                <div id="3dmol_comparison" style="width: 100%; height: 500px; position: relative;"></div>
                                                <script>
                                                    var element = document.getElementById('3dmol_comparison');
                                                    var viewer = $3Dmol.createViewer(element, {{backgroundColor: 'white'}});
                                                    
                                                    // Custom color scheme for input molecule (blue carbons)
                                                    var inputColorScheme = {{
                                                        'C': '#3498db',  // Blue for carbon
                                                        'N': '#3050F8',  // Standard nitrogen blue
                                                        'O': '#FF0D0D',  // Standard oxygen red
                                                        'S': '#FFFF30',  // Standard sulfur yellow
                                                        'F': '#90E050',  // Standard fluorine green
                                                        'Cl': '#1FF01F', // Standard chlorine green
                                                        'Br': '#A62929', // Standard bromine
                                                        'I': '#940094',  // Standard iodine
                                                        'H': '#FFFFFF',  // White hydrogen
                                                        'P': '#FF8000'   // Standard phosphorus
                                                    }};
                                                    
                                                    // Custom color scheme for reassembled molecule (red carbons)
                                                    var reassembledColorScheme = {{
                                                        'C': '#e74c3c',  // Red for carbon
                                                        'N': '#3050F8',  // Standard nitrogen blue
                                                        'O': '#FF0D0D',  // Standard oxygen red
                                                        'S': '#FFFF30',  // Standard sulfur yellow
                                                        'F': '#90E050',  // Standard fluorine green
                                                        'Cl': '#1FF01F', // Standard chlorine green
                                                        'Br': '#A62929', // Standard bromine
                                                        'I': '#940094',  // Standard iodine
                                                        'H': '#FFFFFF',  // White hydrogen
                                                        'P': '#FF8000'   // Standard phosphorus
                                                    }};
                                                    
                                                    // Add input molecule (blue carbons)
                                                    var moldata_input = `{mol_block_input_escaped}`;
                                                    viewer.addModel(moldata_input, "sdf");
                                                    viewer.setStyle({{model: 0}}, {{stick: {{radius: 0.2, colorscheme: {{prop: 'elem', map: inputColorScheme}}}}}});
                                                    
                                                    // Add reassembled molecule (red carbons)
                                                    var moldata_reassembled = `{mol_block_reassembled_escaped}`;
                                                    viewer.addModel(moldata_reassembled, "sdf");
                                                    viewer.setStyle({{model: 1}}, {{stick: {{radius: 0.15, colorscheme: {{prop: 'elem', map: reassembledColorScheme}}}}}});
                                                    
                                                    viewer.zoomTo();
                                                    viewer.render();
                                                </script>
                                            </body>
                                            </html>
                                            """
                                            
                                            st.markdown(f"**Comparing:** Input molecule vs Analog Molecule {selected_3d_idx + 1}")
                                            st.markdown(f"**MCS atoms:** {mcs_result.numAtoms if mcs_result else 'N/A'} | **Best RMSD:** {best_rmsd:.3f} Å | **Conformers sampled:** {len(cids_input)} × {len(cids_reassembled)}")
                                            
                                            # Initialize session state for toggle if not exists
                                            if 'show_input_mol_3d' not in st.session_state:
                                                st.session_state.show_input_mol_3d = True
                                            
                                            # Toggle to show/hide input molecule - use session state key directly
                                            show_input_mol = st.checkbox("Show input molecule", key="show_input_mol_3d")
                                            
                                            # Create viewer HTML with conditional input molecule display
                                            if show_input_mol:
                                                viewer_html_final = viewer_html
                                            else:
                                                # Create viewer with only reassembled molecule
                                                viewer_html_final = f"""
                                                <!DOCTYPE html>
                                                <html>
                                                <head>
                                                    <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
                                                </head>
                                                <body>
                                                    <div style="margin-bottom: 10px; color: #333;">
                                                        <span style="display: inline-block; width: 15px; height: 15px; background-color: #e74c3c; border: 2px solid #e74c3c; margin-right: 5px;"></span>
                                                        <span style="color: #e74c3c; font-weight: bold;">Reassembled Molecule (ID: {selected_3d_idx + 1})</span>
                                                    </div>
                                                    <div id="3dmol_comparison" style="width: 100%; height: 500px; position: relative;"></div>
                                                    <script>
                                                        var element = document.getElementById('3dmol_comparison');
                                                        var viewer = $3Dmol.createViewer(element, {{backgroundColor: 'white'}});
                                                        
                                                        // Custom color scheme for analog molecule (red carbons)
                                                        var reassembledColorScheme = {{
                                                            'C': '#e74c3c',
                                                            'N': '#3050F8',
                                                            'O': '#FF0D0D',
                                                            'S': '#FFFF30',
                                                            'F': '#90E050',
                                                            'Cl': '#1FF01F',
                                                            'Br': '#A62929',
                                                            'I': '#940094',
                                                            'H': '#FFFFFF',
                                                            'P': '#FF8000'
                                                        }};
                                                        
                                                        // Add only reassembled molecule
                                                        var moldata_reassembled = `{mol_block_reassembled_escaped}`;
                                                        viewer.addModel(moldata_reassembled, "sdf");
                                                        viewer.setStyle({{model: 0}}, {{stick: {{radius: 0.2, colorscheme: {{prop: 'elem', map: reassembledColorScheme}}}}}});
                                                        
                                                        viewer.zoomTo();
                                                        viewer.render();
                                                    </script>
                                                </body>
                                                </html>
                                                """
                                            
                                            components.html(viewer_html_final, height=550, scrolling=False)
                                        else:
                                            st.warning("Could not generate 3D conformers for comparison.")
                                    except Exception as e:
                                        st.error(f"Error generating 3D comparison: {str(e)}")
                        
                        # Show filtered molecules in an expander
                        if filtered_info:
                            with st.expander(f"🚫 Filtered Molecules ({len(filtered_info)} molecules with undesirable substructures)", expanded=False):
                                st.markdown("*These molecules were filtered out due to containing structural alerts:*")
                                
                                # Build HTML grid for filtered molecules
                                filtered_html_parts = ['<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; padding: 10px;">']
                                
                                for idx, filt_info in enumerate(filtered_info):
                                    filt_mol = filt_info['mol']
                                    try:
                                        AllChem.GenerateDepictionMatching2DStructure(filt_mol, ref_mol)
                                    except:
                                        try:
                                            AllChem.Compute2DCoords(filt_mol)
                                        except:
                                            pass
                                    
                                    img = Draw.MolToImage(filt_mol, size=(400, 400))
                                    img_buffer = io.BytesIO()
                                    img.save(img_buffer, format='PNG')
                                    img_b64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                                    
                                    filtered_html_parts.append(f'''
                                    <div style="border: 1px solid #ffcccc; border-radius: 8px; padding: 10px; background: #fff5f5; text-align: center;">
                                        <img src="data:image/png;base64,{img_b64}" style="width: 100%; max-width: 300px;">
                                        <div style="margin-top: 8px;">
                                            <b style="color: #cc0000;">⚠️ {filt_info['filter_reason']}</b><br>
                                            <div style="font-size: 9px; word-break: break-all; margin-top: 5px;">
                                                <b>SMILES:</b> {filt_info['smiles']}
                                            </div>
                                        </div>
                                    </div>
                                    ''')
                                
                                filtered_html_parts.append('</div>')
                                filtered_html = ''.join(filtered_html_parts)
                                components.html(filtered_html, height=600, scrolling=True)
                    else:
                        st.warning("Could not reassemble any molecules with the replacement fragments. This many be due to problematic substructure/s in the input molecule or the fragment to be replaced is too small.")

else:
    st.info("👆 Enter a SMILES string above or select an example from the drop down menu OR draw a molecule")

# ============================================================================
# DISPLAY UNDESIRABLE PATTERNS AT BOTTOM
# ============================================================================
st.markdown("---")

with st.expander("⚠️ Undesirable Substructure Patterns (Structural Alerts)", expanded=False):
    st.markdown("*These SMARTS patterns are used to filter out molecules with potentially problematic substructures:*")
    
    # Build HTML grid for pattern structures
    pattern_html_parts = ['<div style="display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; padding: 10px;">']
    
    for smarts, name in UNDESIRABLE_PATTERNS:
        pattern_mol = Chem.MolFromSmarts(smarts)
        if pattern_mol:
            try:
                # Generate 2D coordinates for the pattern
                rdDepictor.Compute2DCoords(pattern_mol)
                img = Draw.MolToImage(pattern_mol, size=(200, 200))
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='PNG')
                img_b64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                
                pattern_html_parts.append(f'''
                <div style="border: 1px solid #ffcccc; border-radius: 8px; padding: 8px; background: #fff8f8; text-align: center;">
                    <img src="data:image/png;base64,{img_b64}" style="width: 100%; max-width: 150px;">
                    <div style="margin-top: 5px; font-size: 11px;">
                        <b style="color: #cc0000;">{name}</b><br>
                        <code style="font-size: 9px; word-break: break-all;">{smarts}</code>
                    </div>
                </div>
                ''')
            except:
                # If image generation fails, just show text
                pattern_html_parts.append(f'''
                <div style="border: 1px solid #ffcccc; border-radius: 8px; padding: 8px; background: #fff8f8; text-align: center;">
                    <div style="height: 100px; display: flex; align-items: center; justify-content: center; color: #999;">
                        [No structure]
                    </div>
                    <div style="margin-top: 5px; font-size: 11px;">
                        <b style="color: #cc0000;">{name}</b><br>
                        <code style="font-size: 9px; word-break: break-all;">{smarts}</code>
                    </div>
                </div>
                ''')
        else:
            # Pattern couldn't be parsed
            pattern_html_parts.append(f'''
            <div style="border: 1px solid #ffcccc; border-radius: 8px; padding: 8px; background: #fff8f8; text-align: center;">
                <div style="height: 100px; display: flex; align-items: center; justify-content: center; color: #999;">
                    [Invalid SMARTS]
                </div>
                <div style="margin-top: 5px; font-size: 11px;">
                    <b style="color: #cc0000;">{name}</b><br>
                    <code style="font-size: 9px; word-break: break-all;">{smarts}</code>
                </div>
            </div>
            ''')
    
    pattern_html_parts.append('</div>')
    pattern_html = ''.join(pattern_html_parts)
    
    components.html(pattern_html, height=800, scrolling=True)

# Footer
# st.markdown("---")
# st.markdown(
#     "<div style='text-align: center; color: gray;'>"
#     "Built with Streamlit and RDKit | Molecule Decomposition Tool"
#     "</div>",
#     unsafe_allow_html=True
# )
