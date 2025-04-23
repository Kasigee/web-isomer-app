import streamlit as st
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

###############################################################################
# Geometry helpers (condensed from your analysis pipeline)
###############################################################################

def load_molecule_from_xyz(xyz_text):
    """Return an RDKit Mol with bonds perceived from XYZ block."""
    mol = Chem.MolFromXYZBlock(xyz_text)
    if mol is None:
        raise ValueError("Could not parse XYZ; check format.")
    rdDetermineBonds.DetermineBonds(mol)
    if mol.GetNumConformers() == 0:
        raise ValueError("XYZ has no coordinates.")
    return mol

# --- small vector helpers ---------------------------------------------------

def _angle(v1, v2):
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))

def _dihedral(p1, p2, p3, p4):
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)
    return np.degrees(np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0)))

# --- main geometry extraction ----------------------------------------------

def bonded_carbons(mol):
    return [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()
            if mol.GetAtomWithIdx(b.GetBeginAtomIdx()).GetAtomicNum()==6 and
               mol.GetAtomWithIdx(b.GetEndAtomIdx()).GetAtomicNum()==6]

def dihedral_and_angle_stats(mol):
    bc = bonded_carbons(mol)
    nbrs = {a: [] for pair in bc for a in pair}
    for a1, a2 in bc:
        nbrs[a1].append(a2)
        nbrs[a2].append(a1)
    conf = mol.GetConformer()
    sum_abs_120, count_ang, sum_sq_dev = 0.0, 0, 0.0
    sum_less90, sum_gt90 = 0.0, 0.0
    dihed_set = set()
    for a1 in nbrs:
        for a2 in nbrs[a1]:
            p1 = np.array(conf.GetAtomPosition(a1))
            p2 = np.array(conf.GetAtomPosition(a2))
            for a3 in nbrs[a2]:
                if a3 == a1:
                    continue
                p3 = np.array(conf.GetAtomPosition(a3))
                ang = _angle(p1-p2, p3-p2)
                sum_abs_120 += abs(120-ang)
                sum_sq_dev += (120-ang)**2
                count_ang += 1
                for a4 in nbrs[a3]:
                    if a4 in (a2, a1):
                        continue
                    dih_tuple = tuple(sorted((a1,a2,a3,a4)))
                    if dih_tuple in dihed_set:
                        continue
                    dihed_set.add(dih_tuple)
                    p4 = np.array(conf.GetAtomPosition(a4))
                    dih = _dihedral(p1, p2, p3, p4)
                    if dih < 90:
                        sum_less90 += dih
                    else:
                        sum_gt90 += 180 - dih
    rmsd_bond_angle = np.sqrt(sum_sq_dev/count_ang) if count_ang else 0.0
    sum_dev = sum_less90 + sum_gt90
    return sum_dev, sum_abs_120, rmsd_bond_angle

# --- HOMA (unchanged) -------------------------------------------------------

def homa_aromatic_rings(mol, alpha=257.7, R_opt=1.388):
    ri = mol.GetRingInfo()
    conf = mol.GetConformer()
    ring_homas = []
    for ring in ri.AtomRings():
        if any(mol.GetAtomWithIdx(idx).GetAtomicNum()!=6 or
               not mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            continue
        bl = []
        for i in range(len(ring)):
            a1 = ring[i]
            a2 = ring[(i+1)%len(ring)]
            p1 = np.array(conf.GetAtomPosition(a1))
            p2 = np.array(conf.GetAtomPosition(a2))
            bl.append(np.linalg.norm(p1-p2))
        n=len(bl)
        homa = 1.0 - (alpha/n)*sum((R-R_opt)**2 for R in bl)
        ring_homas.append(homa)
    if not ring_homas:
        return float('nan')
    return sum(ring_homas)/len(ring_homas)

###############################################################################
# Energy model (C40H22 fit)
###############################################################################

def energy_c40_fit(sum_dev, avg_homa, rmsd):
    A_dih = 0.02082790
    A_homa = -340.97268109
    A_rmsd = 16.64640654
    C = 236.14120030
    return A_dih*sum_dev + A_homa*avg_homa + A_rmsd*rmsd + C

###############################################################################
# Streamlit UI
###############################################################################

st.set_page_config(page_title="PAH Isomer Energy Estimator", page_icon="⚛️")

st.title("⚛️ PAH Isomer Energy Estimator (C₄₀H₂₂-trained)")
st.markdown("Upload an **XYZ** file of your polycyclic aromatic hydrocarbon. The app\ncomputes key geometric descriptors (ΣΔφ, HOMA, RMSD₍angle₎) and estimates the\nisomerisation energy using the regression fitted to the C₄₀H₂₂ data set.")

uploaded = st.file_uploader("XYZ file", type=["xyz"])

if uploaded is not None:
    xyz_bytes = uploaded.read()
    try:
        xyz_text = xyz_bytes.decode("utf-8")
    except UnicodeDecodeError:
        st.error("File is not UTF‑8 encoded.")
        st.stop()
    with st.spinner("Parsing and analysing …"):
        try:
            mol = load_molecule_from_xyz(xyz_text)
            sum_dev, sum_abs_120, rmsd = dihedral_and_angle_stats(mol)
            avg_homa = homa_aromatic_rings(mol)
            est_energy = energy_c40_fit(sum_dev, avg_homa, rmsd)
        except Exception as e:
            st.exception(e)
            st.stop()
    st.success("Analysis complete!")
    st.subheader("Geometry metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("ΣΔφ (°)", f"{sum_dev:,.1f}")
    col2.metric("Avg HOMA", f"{avg_homa:0.3f}")
    col3.metric("RMSDᵦ (°)", f"{rmsd:0.2f}")

    st.subheader("Predicted ΔE (kcal mol⁻¹)")
    st.latex(r"E = 0.02082790\,\Sigma\lvert120-\varphi\rvert - 340.97268\,\overline{\text{HOMA}} + 16.6464\,\text{RMSD}_{\angle} + 236.141")
    st.write(f"**{est_energy:,.2f} kcal mol⁻¹**")

else:
    st.info("↖️ Upload an XYZ file to begin.")
