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
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)
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
    # bond-angle deviations
    sum_dev = 0.0
    count = 0
    angles = []
    dihedrals = []
    for a1,a2 in bonded:
        for a3 in [x for x in mol.GetBondedAtoms(mol.GetAtomWithIdx(a2)) if x.GetIdx()!=a1]:
            p1 = np.array(conf.GetAtomPosition(a1))
            p2 = np.array(conf.GetAtomPosition(a2))
            p3 = np.array(conf.GetAtomPosition(a3.GetIdx()))
            ang = calculate_bond_angle(p1,p2,p3)
            sum_dev += abs(120 - ang)
            count += 1
            angles.append(ang)
            for a4 in [x for x in mol.GetBondedAtoms(mol.GetAtomWithIdx(a3.GetIdx())) if x.GetIdx() not in (a2,a1)]:
                p4 = np.array(conf.GetAtomPosition(a4.GetIdx()))
                dih = calculate_dihedral(p1,p2,p3,p4)
                dihedrals.append(dih)
    # RMSD of angles
    if count>0:
        rmsd = np.sqrt(np.mean([(120 - a)**2 for a in angles]))
    else:
        rmsd = 0.0
    return sum_dev, rmsd


def homa_aromatic_rings(mol, alpha=257.7, R_opt=1.388):
    ri = mol.GetRingInfo().AtomRings()
    conf = mol.GetConformer()
    homas = []
    for ring in ri:
        # check aromatic C
        if all(mol.GetAtomWithIdx(i).GetAtomicNum()==6 and mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            lengths = []
            for i in range(len(ring)):
                p1 = np.array(conf.GetAtomPosition(ring[i]))
                p2 = np.array(conf.GetAtomPosition(ring[(i+1)%len(ring)]))
                lengths.append(np.linalg.norm(p1-p2))
            n = len(lengths)
            sum_sq = sum((L - R_opt)**2 for L in lengths)
            homas.append(1 - (alpha/n)*sum_sq)
    return (np.nan, []) if not homas else (float(np.mean(homas)), homas)


def find_database_energy(mol, csv_file="analysis_results.C44H24.csv"):
    # match by SMILES
    m0 = Chem.RemoveHs(mol)
    smi = Chem.MolToSmiles(m0)
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        return None
    match = df.loc[df.smiles==smi]
    if not match.empty:
        return float(match.D4_rel_energy.iloc[0])
    return None

# ---------- Prediction Models ----------

def model_dihedral(sum_dev):
    A, B, MAD = 0.01498026, 5.01448849, 3.618
    E = A*sum_dev + B; return E, E-MAD, E+MAD, "E = 0.01498026·ΣDihedral + 5.01448849"

def model_xtb(sum_dev, xtb):
    A, B, C, MAD = 0.00678795, 1.07126936, 3.49502511, 2.07925630
    E = A*sum_dev + B*xtb + C; return E, E-MAD, E+MAD, "E = 0.00678795·ΣDihedral + 1.07126936·XTB + 3.49502511"

def model_homa(homa):
    E = -40.71451171*homa + 49.03884678; return E, None, None, "E = -40.7145·HOMA + 49.0388"

def model_dh(sum_dev, homa):
    A, B, C, MAD = 0.01985355, -314.14544891, 241.66896314, 2.53312351
    E = A*sum_dev + B*homa + C; return E, E-MAD, E+MAD, "E = 0.01985355·ΣDihedral - 314.1454·HOMA + 241.6690"

def model_ht(homa, xtb):
    A, B, C, MAD = 29.41245664, 1.37801523, -15.68808658, 2.57228594
    E = A*homa + B*xtb + C; return E, E-MAD, E+MAD, "E = 29.4125·HOMA + 1.3780·XTB - 15.6881"

def model_dhr(sum_dev, homa, rmsd):
    A_d, A_h, A_r, C = 0.02082790, -340.97268109, 16.64640654, 236.14120030
    E = A_d*sum_dev + A_h*homa + A_r*rmsd + C; return E, None, None, "E = 0.02082790·ΣDihedral - 340.9727·HOMA + 16.6464·θRMSD + 236.1412"

# ---------- Streamlit App ----------

def main():
    st.set_page_config(page_title="Isomerization Energy", layout="centered")
    st.title("Isomerization Energy Predictor")
    st.sidebar.header("Options")

    model = st.sidebar.selectbox(
        "Select prediction model:",
        [
            "Dihedral-only",
            "XTB (requires XTB energy)",
            "HOMA-only",
            "Dihedral + HOMA",
            "HOMA + XTB",
            "Dihedral + HOMA + XTB",
            "Database lookup"
        ]
    )
    xtb_val = 0.0
    if "XTB" in model:
        xtb_val = st.sidebar.number_input("XTB energy value:", value=0.0)

    xyz_file = st.file_uploader("Upload XYZ file", type="xyz")
    if not xyz_file:
        return
    xyz_text = xyz_file.read().decode("utf-8")
    mol = load_molecule_from_xyz(xyz_text)
    if mol is None:
        return

    sum_dev, rmsd = analyze_geometry(mol)
    homa_avg, _ = homa_aromatic_rings(mol)
    if np.isnan(homa_avg): homa_avg = 0.0
    db_energy = None
    if model=="Database lookup":
        db_energy = find_database_energy(mol)

    # Compute
    if model=="Dihedral-only":
        E, low, high, eq = model_dihedral(sum_dev)
    elif model.startswith("XTB"):
        E, low, high, eq = model_xtb(sum_dev, xtb_val)
    elif model=="HOMA-only":
        E, low, high, eq = model_homa(homa_avg)
    elif model=="Dihedral + HOMA":
        E, low, high, eq = model_dh(sum_dev, homa_avg)
    elif model=="HOMA + XTB":
        E, low, high, eq = model_ht(homa_avg, xtb_val)
    elif model=="Dihedral + HOMA + XTB":
        # fallback to D+H+R model for this choice
        E, low, high, eq = model_dhr(sum_dev, homa_avg, rmsd)
    elif model=="Database lookup":
        if db_energy is not None:
            st.success(f"Database ΔE: {db_energy:.3f} kJ/mol")
        else:
            st.warning("No database match found for this isomer.")
        return

    # Display results
    st.markdown(f"**Equation:** `{eq}`")
    st.markdown(f"**ΣDihedral:** {sum_dev:.3f}   **HOMA:** {homa_avg:.3f}   **θRMSD:** {rmsd:.3f}")
    st.markdown(f"## Predicted ΔE = {E:.3f} kJ/mol")
    if low is not None and high is not None:
        st.markdown(f"_Range: {low:.3f} – {high:.3f} kJ/mol_")

if __name__ == '__main__':
    main()
