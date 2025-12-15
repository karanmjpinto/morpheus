"""
Interactive Substructure Selection App
Select atoms in a 2D molecule structure and get the SMILES with dummy atoms at cut points.
"""

import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import json
from streamlit.components.v1 import html

st.set_page_config(page_title="Substructure Selector", layout="wide")

st.title("🧪 Interactive Substructure Selector")
st.markdown("Click on atoms to select them, then extract the substructure SMILES with dummy atoms at cut points.")

# Session state for selected atoms
if 'selected_atoms' not in st.session_state:
    st.session_state.selected_atoms = set()

# Input SMILES
default_smiles = "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"
smiles_input = st.text_input("Enter SMILES:", value=default_smiles)

# Parse molecule
mol = Chem.MolFromSmiles(smiles_input)

if mol is None:
    st.error("Invalid SMILES. Please enter a valid molecular structure.")
    st.stop()

# Generate 2D coordinates
rdDepictor.Compute2DCoords(mol)

# Get atom positions for the interactive SVG
conf = mol.GetConformer()
atom_positions = []
for i in range(mol.GetNumAtoms()):
    pos = conf.GetAtomPosition(i)
    atom = mol.GetAtomWithIdx(i)
    atom_positions.append({
        'idx': i,
        'x': pos.x,
        'y': pos.y,
        'symbol': atom.GetSymbol(),
        'atomic_num': atom.GetAtomicNum()
    })

# Calculate bounds for SVG scaling
min_x = min(p['x'] for p in atom_positions)
max_x = max(p['x'] for p in atom_positions)
min_y = min(p['y'] for p in atom_positions)
max_y = max(p['y'] for p in atom_positions)

# Add padding
padding = 1.5
min_x -= padding
max_x += padding
min_y -= padding
max_y += padding

# Get bond information
bonds = []
for bond in mol.GetBonds():
    bonds.append({
        'begin': bond.GetBeginAtomIdx(),
        'end': bond.GetEndAtomIdx(),
        'type': str(bond.GetBondType())
    })

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Click atoms to select/deselect")
    
    # Create interactive HTML/SVG component
    html_content = f"""
    <style>
        .mol-container {{
            border: 2px solid #ddd;
            border-radius: 10px;
            background: white;
            padding: 10px;
        }}
        .atom {{
            cursor: pointer;
            transition: all 0.2s;
        }}
        .atom:hover {{
            stroke-width: 3;
            stroke: #007bff;
        }}
        .atom.selected {{
            fill: #ff6b6b !important;
            stroke: #c92a2a;
            stroke-width: 2;
        }}
        .bond {{
            stroke: #333;
            stroke-width: 2;
        }}
        .atom-label {{
            font-family: Arial, sans-serif;
            font-size: 12px;
            pointer-events: none;
            user-select: none;
        }}
        #selection-info {{
            margin-top: 10px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
            font-family: monospace;
        }}
    </style>
    
    <div class="mol-container">
        <svg id="mol-svg" width="600" height="500" viewBox="{min_x} {min_y} {max_x - min_x} {max_y - min_y}">
            <g id="bonds"></g>
            <g id="atoms"></g>
        </svg>
    </div>
    
    <div id="selection-info">
        <strong>Selected atoms:</strong> <span id="selected-display">None</span>
    </div>
    
    <div style="margin-top: 10px;">
        <button onclick="clearSelection()" style="padding: 8px 16px; margin-right: 10px; cursor: pointer;">Clear Selection</button>
        <button onclick="sendSelection()" style="padding: 8px 16px; background: #28a745; color: white; border: none; cursor: pointer; border-radius: 4px;">Extract Substructure</button>
    </div>
    
    <input type="hidden" id="selected-atoms-input" name="selected_atoms" value="">
    
    <script>
        const atomPositions = {json.dumps(atom_positions)};
        const bonds = {json.dumps(bonds)};
        let selectedAtoms = new Set();
        
        const svg = document.getElementById('mol-svg');
        const bondsGroup = document.getElementById('bonds');
        const atomsGroup = document.getElementById('atoms');
        
        // Scale factor for display
        const scale = 1;
        
        // Draw bonds
        bonds.forEach(bond => {{
            const atom1 = atomPositions[bond.begin];
            const atom2 = atomPositions[bond.end];
            
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', atom1.x * scale);
            line.setAttribute('y1', -atom1.y * scale);  // Flip Y
            line.setAttribute('x2', atom2.x * scale);
            line.setAttribute('y2', -atom2.y * scale);
            line.setAttribute('class', 'bond');
            
            // Handle double/triple bonds
            if (bond.type === 'DOUBLE') {{
                line.setAttribute('stroke-width', '4');
            }} else if (bond.type === 'TRIPLE') {{
                line.setAttribute('stroke-width', '6');
            }} else if (bond.type === 'AROMATIC') {{
                line.setAttribute('stroke-dasharray', '5,3');
            }}
            
            bondsGroup.appendChild(line);
        }});
        
        // Draw atoms
        atomPositions.forEach(atom => {{
            const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            g.setAttribute('class', 'atom');
            g.setAttribute('data-idx', atom.idx);
            
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', atom.x * scale);
            circle.setAttribute('cy', -atom.y * scale);  // Flip Y
            circle.setAttribute('r', 0.35);
            
            // Color by atom type
            let color = '#666';  // Carbon default
            if (atom.symbol === 'N') color = '#3498db';
            else if (atom.symbol === 'O') color = '#e74c3c';
            else if (atom.symbol === 'S') color = '#f1c40f';
            else if (atom.symbol === 'F' || atom.symbol === 'Cl' || atom.symbol === 'Br') color = '#27ae60';
            else if (atom.symbol === 'C') color = '#555';
            
            circle.setAttribute('fill', color);
            g.appendChild(circle);
            
            // Add label for non-carbon atoms
            if (atom.symbol !== 'C') {{
                const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                text.setAttribute('x', atom.x * scale);
                text.setAttribute('y', -atom.y * scale + 0.05);
                text.setAttribute('text-anchor', 'middle');
                text.setAttribute('dominant-baseline', 'middle');
                text.setAttribute('class', 'atom-label');
                text.setAttribute('fill', 'white');
                text.setAttribute('font-size', '0.3');
                text.textContent = atom.symbol;
                g.appendChild(text);
            }}
            
            g.addEventListener('click', () => toggleAtom(atom.idx));
            atomsGroup.appendChild(g);
        }});
        
        function toggleAtom(idx) {{
            if (selectedAtoms.has(idx)) {{
                selectedAtoms.delete(idx);
            }} else {{
                selectedAtoms.add(idx);
            }}
            updateDisplay();
        }}
        
        function updateDisplay() {{
            // Update visual selection
            document.querySelectorAll('.atom').forEach(g => {{
                const idx = parseInt(g.getAttribute('data-idx'));
                if (selectedAtoms.has(idx)) {{
                    g.classList.add('selected');
                }} else {{
                    g.classList.remove('selected');
                }}
            }});
            
            // Update text display
            const display = document.getElementById('selected-display');
            if (selectedAtoms.size === 0) {{
                display.textContent = 'None';
            }} else {{
                display.textContent = Array.from(selectedAtoms).sort((a,b) => a-b).join(', ');
            }}
            
            // Update hidden input
            document.getElementById('selected-atoms-input').value = JSON.stringify(Array.from(selectedAtoms));
        }}
        
        function clearSelection() {{
            selectedAtoms.clear();
            updateDisplay();
        }}
        
        function sendSelection() {{
            const atoms = Array.from(selectedAtoms);
            if (atoms.length === 0) {{
                alert('Please select at least one atom');
                return;
            }}
            // Send to Streamlit via query params workaround
            const params = new URLSearchParams(window.location.search);
            params.set('selected', atoms.join(','));
            window.parent.postMessage({{
                type: 'streamlit:setComponentValue',
                value: atoms
            }}, '*');
            
            // Store in sessionStorage for Streamlit to read
            sessionStorage.setItem('selectedAtoms', JSON.stringify(atoms));
            
            // Trigger form submission by updating URL
            const newUrl = window.location.pathname + '?selected=' + atoms.join(',');
            window.location.href = newUrl;
        }}
    </script>
    """
    
    html(html_content, height=650)

with col2:
    st.subheader("Selected Substructure")
    
    # Get selected atoms from query params
    query_params = st.query_params
    selected_str = query_params.get('selected', '')
    
    if selected_str:
        try:
            selected_atoms = [int(x) for x in selected_str.split(',') if x.strip()]
            
            if selected_atoms:
                st.write(f"**Selected atoms:** {selected_atoms}")
                
                # Find bonds to break (bonds between selected and non-selected atoms)
                selected_set = set(selected_atoms)
                bonds_to_break = []
                
                for bond in mol.GetBonds():
                    begin_idx = bond.GetBeginAtomIdx()
                    end_idx = bond.GetEndAtomIdx()
                    
                    # Bond crosses selection boundary
                    if (begin_idx in selected_set) != (end_idx in selected_set):
                        bonds_to_break.append(bond.GetIdx())
                
                if bonds_to_break:
                    # Fragment at the boundary bonds
                    try:
                        frag_mol = Chem.FragmentOnBonds(mol, bonds_to_break, addDummies=True)
                        frags = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=False)
                        frag_atom_lists = Chem.GetMolFrags(frag_mol, asMols=False)
                        
                        # Find the fragment containing our selected atoms
                        target_frag = None
                        for frag, frag_atoms in zip(frags, frag_atom_lists):
                            frag_atoms_set = set(frag_atoms)
                            if any(a in frag_atoms_set for a in selected_atoms):
                                # Count non-dummy atoms
                                non_dummy = sum(1 for a in frag.GetAtoms() if a.GetAtomicNum() != 0)
                                if non_dummy == len(selected_atoms):
                                    target_frag = frag
                                    break
                        
                        if target_frag is not None:
                            # Clean up isotope labels on dummy atoms
                            rw = Chem.RWMol(target_frag)
                            for atom in rw.GetAtoms():
                                if atom.GetAtomicNum() == 0:
                                    atom.SetIsotope(0)
                            target_frag = rw.GetMol()
                            
                            # Try to sanitize
                            try:
                                Chem.SanitizeMol(target_frag)
                            except:
                                try:
                                    for atom in target_frag.GetAtoms():
                                        atom.SetIsAromatic(False)
                                    for bond in target_frag.GetBonds():
                                        bond.SetIsAromatic(False)
                                    Chem.SanitizeMol(target_frag)
                                except:
                                    pass
                            
                            # Generate SMILES
                            sub_smiles = Chem.MolToSmiles(target_frag, canonical=True)
                            
                            st.success("Substructure extracted!")
                            st.code(sub_smiles, language=None)
                            
                            # Check if SMILES is valid
                            test_mol = Chem.MolFromSmiles(sub_smiles)
                            if test_mol:
                                st.write("✅ Valid SMILES")
                            else:
                                st.write("⚠️ SMILES may have parsing issues")
                            
                            # Display the substructure
                            try:
                                rdDepictor.Compute2DCoords(target_frag)
                                img = Draw.MolToImage(target_frag, size=(300, 300))
                                st.image(img, caption="Selected Substructure")
                            except:
                                st.warning("Could not render substructure image")
                        else:
                            st.warning("Could not extract the exact substructure")
                    except Exception as e:
                        st.error(f"Error fragmenting molecule: {e}")
                else:
                    # No bonds to break - entire selection is isolated or complete molecule
                    try:
                        sub_smiles = Chem.MolFragmentToSmiles(mol, selected_atoms, canonical=True)
                        st.success("Substructure extracted (no cut points)!")
                        st.code(sub_smiles, language=None)
                        
                        # Create and display mol
                        sub_mol = Chem.MolFromSmiles(sub_smiles)
                        if sub_mol:
                            img = Draw.MolToImage(sub_mol, size=(300, 300))
                            st.image(img, caption="Selected Substructure")
                    except Exception as e:
                        st.error(f"Error: {e}")
        except Exception as e:
            st.error(f"Error parsing selection: {e}")
    else:
        st.info("👈 Click atoms in the molecule to select them, then click 'Extract Substructure'")
        
        # Show the original molecule
        st.write("**Original molecule:**")
        img = Draw.MolToImage(mol, size=(300, 300))
        st.image(img)

# Instructions
with st.expander("ℹ️ How to use"):
    st.markdown("""
    1. **Enter a SMILES** string in the input box above
    2. **Click on atoms** in the 2D structure to select them (they will turn red)
    3. Click an atom again to **deselect** it
    4. Click **'Extract Substructure'** to get the SMILES of your selection
    5. The resulting SMILES will have **[*] dummy atoms** at points where bonds were cut
    6. Use **'Clear Selection'** to start over
    
    **Tips:**
    - Select contiguous atoms for best results
    - The dummy atoms ([*]) indicate where the substructure was connected to the rest of the molecule
    """)
