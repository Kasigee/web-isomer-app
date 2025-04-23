import streamlit as st
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

# ---------- Geometry & Analysis Helpers ----------

def load_molecule_from_xyz(xyz_text):
    mol = Chem.MolFromXYZBlock(xyz_text)
    if mol is None:
        st.error("Could not parse XYZ file.")
        return None
    rdDetermineBonds.DetermineBonds(mol)
    return mol


def calculate_bond_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def calculate_dihedral(p1, p2, p3, p4):
    b1 = p2 - p1; b2 = p3 - p2; b3 = p4 - p3
    n1 = np.cross(b1, b2); n2 = np.cross(b2, b3)
    n1 /= np.linalg.norm(n1); n2 /= np.linalg.norm(n2)
    cos_theta = np.clip(np.dot(n1, n2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def find_bonded_carbons(mol):
    pairs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx(); j = bond.GetEndAtomIdx()
        if mol.GetAtomWithIdx(i).GetAtomicNum()==6 and mol.GetAtomWithIdx(j).GetAtomicNum()==6:
            pairs.append((i, j))
    return pairs


def analyze_geometry(mol):
    conf = mol.GetConformer()
    bonded = find_bonded_carbons(mol)
    neighbors = {idx: [nb.GetIdx() for nb in mol.GetAtomWithIdx(idx).GetNeighbors()]
                 for atom in mol.GetAtoms() if atom.GetAtomicNum()==6 for idx in [atom.GetIdx()]}
    angles = []
    for a1, a2 in bonded:
        for a3 in neighbors.get(a2, []):
            if a3 == a1: continue
            p1 = np.array(conf.GetAtomPosition(a1))
            p2 = np.array(conf.GetAtomPosition(a2))
            p3 = np.array(conf.GetAtomPosition(a3))
            angles.append(calculate_bond_angle(p1,p2,p3))
    sum_dev = sum(abs(120 - a) for a in angles)
    rmsd = np.sqrt(np.mean([(120 - a)**2 for a in angles])) if angles else 0.0
    return sum_dev, rmsd


def homa_aromatic_rings(mol, alpha=257.7, R_opt=1.388):
    ri = mol.GetRingInfo().AtomRings(); conf = mol.GetConformer(); homas=[]
    for ring in ri:
        if all(mol.GetAtomWithIdx(i).GetAtomicNum()==6 and mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            lengths=[np.linalg.norm(np.array(conf.GetAtomPosition(ring[i]))-
                                    np.array(conf.GetAtomPosition(ring[(i+1)%len(ring)])))
                     for i in range(len(ring))]
            n=len(lengths); sum_sq=sum((L-R_opt)**2 for L in lengths)
            homas.append(1-(alpha/n)*sum_sq)
    return (np.nan,[]) if not homas else (float(np.mean(homas)), homas)


def find_database_energy(mol, csv_file="COMPAS_XTB_MS_WEBAPP_DATA.csv"):
    m0=Chem.RemoveHs(mol); smi=Chem.MolToSmiles(m0)
    try: df=pd.read_csv(csv_file)
    except: return None
    match=df.loc[df.smiles==smi]
    return float(match.D4_rel_energy.iloc[0]) if not match.empty else None

# ---------- Prediction Models ----------

def model_dhr(sum_dev, homa, rmsd):
    A_d,A_h,A_r,C=0.02082790,-340.97268109,16.64640654,236.14120030
    return A_d*sum_dev + A_h*homa + A_r*rmsd + C, None, None,
    "E = 0.02082790·ΣDihedral - 340.9727·HOMA + 16.6464·θRMSD + 236.1412"
# (Define other models similarly...)

# ---------- Streamlit App ----------

def main():
    st.set_page_config(page_title="Isomerization Energy", layout="centered")
    st.title("Isomerization Energy Predictor")

    # File upload persists in session_state
    if 'xyz_text' not in st.session_state:
        st.session_state.xyz_text = None
    uploaded = st.file_uploader("Upload XYZ file", type="xyz")
    if uploaded:
        st.session_state.xyz_text = uploaded.read().decode()

    # Sidebar options
    models = ["Dihedral + HOMA + θRMSD",
              "Dihedral-only","HOMA-only","Dihedral + HOMA","HOMA + XTB","XTB"]
    model_choice = st.sidebar.selectbox("Prediction model:", models, index=0)
    xtb_val = st.sidebar.number_input("XTB energy:", value=0.0)

    if st.sidebar.button("Calculate"):
        if not st.session_state.xyz_text:
            st.sidebar.warning("Please upload an XYZ file first.")
            return
        mol = load_molecule_from_xyz(st.session_state.xyz_text)
        if not mol: return
        sum_dev, rmsd = analyze_geometry(mol)
        homa_avg, _ = homa_aromatic_rings(mol)
        homa_avg = 0.0 if np.isnan(homa_avg) else homa_avg

        # Compute predicted energy
        if model_choice == "Dihedral + HOMA + θRMSD":
            E, low, high, eq = model_dhr(sum_dev, homa_avg, rmsd)
        # elif other models...

        # Always lookup database
        db = find_database_energy(mol)

        # Display
        st.markdown(f"**Equation:** `{eq}`")
        st.markdown(f"**ΣDihedral:** {sum_dev:.3f}   **HOMA:** {homa_avg:.3f}   **θRMSD:** {rmsd:.3f}")
        st.markdown(f"## Predicted ΔE = {E:.3f} kJ/mol")
        if low and high:
            st.markdown(f"_Range: {low:.3f} – {high:.3f} kJ/mol_")
        if db is not None:
            st.success(f"Database ΔE: {db:.3f} kJ/mol (PBE0-D4/6-31G(2df,p))")

if __name__=='__main__': main()
