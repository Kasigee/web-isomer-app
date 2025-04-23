import os
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

# Try to import py3Dmol for 3D viewer
try:
    import py3Dmol
    HAVE_VIEWER = True
except ImportError:
    HAVE_VIEWER = False

# ---------- Page Configuration ----------
st.set_page_config(page_title="Isomerization Energy", layout="centered")

# Optional: UNE branding colors
UNE_GREEN = "#00693e"
UNE_GOLD = "#ffd400"
st.markdown(f"""<style>
.stApp {{ background-color:{UNE_GREEN}10; }}
.css-1d391kg {{ color:{UNE_GREEN}; }}
.css-1v3fvcr {{ background:{UNE_GOLD}33; }}
</style>""", unsafe_allow_html=True)

# ---------- Geometry Helpers ----------
def load_molecule_from_xyz(xyz_text):
    mol = Chem.MolFromXYZBlock(xyz_text)
    if mol is None:
        st.error("Could not parse XYZ file.")
        return None
    rdDetermineBonds.DetermineBonds(mol)
    return mol


def calculate_bond_angle(p1, p2, p3):
    v1, v2 = p1 - p2, p3 - p2
    v1 /= np.linalg.norm(v1); v2 /= np.linalg.norm(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))


def calculate_dihedral(p1, p2, p3, p4):
    b1, b2, b3 = p2 - p1, p3 - p2, p4 - p3
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    n1 /= np.linalg.norm(n1); n2 /= np.linalg.norm(n2)
    return np.degrees(np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0)))


def find_bonded_carbons(mol):
    pairs = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if mol.GetAtomWithIdx(i).GetAtomicNum() == 6 and mol.GetAtomWithIdx(j).GetAtomicNum() == 6:
            pairs.append((i, j))
    return pairs


def analyze_geometry(mol):
    conf = mol.GetConformer()
    bonded = find_bonded_carbons(mol)
    neighbors = {idx: [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(idx).GetNeighbors()]
                 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6
                 for idx in [atom.GetIdx()]}
    angles = []
    for a1, a2 in bonded:
        for a3 in neighbors.get(a2, []):
            if a3 == a1: continue
            p1 = np.array(conf.GetAtomPosition(a1))
            p2 = np.array(conf.GetAtomPosition(a2))
            p3 = np.array(conf.GetAtomPosition(a3))
            angles.append(calculate_bond_angle(p1, p2, p3))
    sum_dev = sum(abs(120 - a) for a in angles)
    rmsd = np.sqrt(np.mean([(120 - a)**2 for a in angles])) if angles else 0.0
    return sum_dev, rmsd


def homa_aromatic_rings(mol, alpha=257.7, R_opt=1.388):
    rings = mol.GetRingInfo().AtomRings()
    conf = mol.GetConformer()
    homas = []
    for ring in rings:
        if all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 and mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            lengths = [
                np.linalg.norm(np.array(conf.GetAtomPosition(ring[i])) -
                               np.array(conf.GetAtomPosition(ring[(i+1) % len(ring)])))
                for i in range(len(ring))
            ]
            n = len(lengths)
            sum_sq = sum((L - R_opt)**2 for L in lengths)
            homas.append(1 - (alpha/n)*sum_sq)
    return (np.nan, []) if not homas else (float(np.mean(homas)), homas)

# ---------- Database Lookup ----------
def find_database_energy(mol):
    primary = "COMPAS_XTB_MS_WEBAPP_DATA.csv"
    secondary = "compas-3D.csv"
    smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
    if os.path.exists(primary):
        df = pd.read_csv(primary)
        m = df.loc[df.smiles == smi]
        if not m.empty:
            return float(m.D4_rel_energy.iloc[0]), True, "PBE0-D4/6-31G(2df,p)", smi
    if os.path.exists(secondary):
        df2 = pd.read_csv(secondary)
        m2 = df2.loc[df2.smiles == smi]
        if not m2.empty and "Erel_eV" in df2.columns:
            e_ev = float(m2.Erel_eV.iloc[0])
            return e_ev * 96.485, True, \
                   "CAM-B3LYP-D3BJ/cc-pvdz//CAM-B3LYP-D3BJ/def2-SVP", smi
    if os.path.exists(primary) or os.path.exists(secondary):
        return None, True, None, smi
    return None, False, None, smi

# ---------- Prediction Models ----------
def model_dhr(sum_dev, homa, rmsd):
    A_d, A_h, A_r, C = 0.02082790, -340.97268109, 16.64640654, 236.14120030
    eq = "E = 0.02082790·ΣDihedral - 340.9727·HOMA + 16.6464·θRMSD + 236.1412"
    return A_d*sum_dev + A_h*homa + A_r*rmsd + C, None, None, eq

def model_dihedral_only(sum_dev):
    A, B = 0.01506654, 5.26542057
    eq = "E = 0.01506654·ΣDihedral + 5.26542057"
    return A*sum_dev + B, None, None, eq

def model_homa_only(homa):
    A, B = -83.95374901, 81.47198711
    eq = "E = -83.9537·HOMA + 81.4720"
    return A*homa + B, None, None, eq

def model_dh_only(sum_dev, homa):
    A_d, A_h, B = 0.02230771, -361.11764751, 275.86109778
    eq = "E = 0.02230771·ΣDihedral - 361.1176·HOMA + 275.8611"
    return A_d*sum_dev + A_h*homa + B, None, None, eq

# ---------- Streamlit App ----------
def main():
    st.title("Isomerization Energy Predictor")
    if 'xyz_text' not in st.session_state:
        st.session_state.xyz_text = None
        st.session_state.prev_text = None
    uploaded = st.file_uploader("Upload XYZ file", type="xyz")
    if uploaded:
        txt = uploaded.read().decode()
        if txt != st.session_state.prev_text:
            st.session_state.xyz_text = txt
            st.session_state.prev_text = txt
    models = [
        "Dihedral + HOMA + θRMSD",
        "Dihedral-only",
        "HOMA-only",
        "Dihedral + HOMA"
    ]
    choice = st.sidebar.selectbox("Prediction model:", models, index=0)
    trigger = (st.session_state.xyz_text and
               st.session_state.xyz_text != st.session_state.get('computed_for'))
    if st.sidebar.button("Calculate"): trigger = True
    if trigger:
        st.session_state.computed_for = st.session_state.xyz_text
        mol = load_molecule_from_xyz(st.session_state.xyz_text)
        if not mol: return
        smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
        st.markdown(f"**SMILES:** {smi}")
        # 3D viewer if available
        if HAVE_VIEWER:
            view = py3Dmol.view(width=400, height=300)
            view.addModel(st.session_state.xyz_text, 'xyz')
            view.setStyle({'stick':{}})
            view.setBackgroundColor('0xeeeeee')
            view.rotate({'x':90, 'y':0, 'z':0})
            view.zoomTo()
            components.html(view._make_html(), height=300, width=400)
        else:
            st.info("Install 'py3Dmol' in requirements.txt to enable 3D viewer.")
        sum_dev, rmsd = analyze_geometry(mol)
        homa_avg, _ = homa_aromatic_rings(mol)
        homa_avg = 0.0 if np.isnan(homa_avg) else homa_avg
        if choice == "Dihedral + HOMA + θRMSD":
            E, low, high, eq = model_dhr(sum_dev, homa_avg, rmsd)
        elif choice == "Dihedral-only":
            E, low, high, eq = model_dihedral_only(sum_dev)
        elif choice == "HOMA-only":
            E, low, high, eq = model_homa_only(homa_avg)
        elif choice == "Dihedral + HOMA":
            E, low, high, eq = model_dh_only(sum_dev, homa_avg)
        db_energy, db_avail, db_source, smi_out = find_database_energy(mol)
        st.markdown(f"**Equation:** `{eq}`")
        st.markdown(f"**ΣDihedral:** {sum_dev:.3f}   **HOMA:** {homa_avg:.3f}   **θRMSD:** {rmsd:.3f}")
        st.markdown(f"## Predicted ΔE = {E:.3f} kJ/mol")
        if low is not None and high is not None:
            st.markdown(f"_Range: {low:.3f} – {high:.3f} kJ/mol_")
        if not db_avail:
            st.warning("No database files found. Add CSVs to enable lookup.")
        elif db_energy is None:
            st.info(f"SMILES: {smi_out} — no database match found.")
        else:
            st.success(f"SMILES: {smi_out} — Database ΔE: {db_energy:.3f} kJ/mol ({db_source})")

if __name__ == '__main__':
    main()
