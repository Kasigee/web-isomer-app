import streamlit as st
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

###############################################################################
# Helper functions (ported from miller_PAH_analysis_wDPO4.py)
###############################################################################

def load_molecule_from_xyz(xyz_data: str):
    """Return an RDKit Mol with 3‑D coordinates from an XYZ string."""
    mol = Chem.MolFromXYZBlock(xyz_data)
    if mol is None:
        raise ValueError("Could not parse XYZ text into an RDKit molecule.«)"""
    rdDetermineBonds.DetermineBonds(mol)
    return mol

# --- geometry helpers -------------------------------------------------------

def calculate_bond_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    cos_theta = np.dot(v1, v2)
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

def calculate_dihedral(p1, p2, p3, p4):
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)
    cos_theta = np.dot(n1, n2)
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

def find_bonded_carbons(mol):
    return [(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
            for b in mol.GetBonds()
            if mol.GetAtomWithIdx(b.GetBeginAtomIdx()).GetAtomicNum() == 6 and
               mol.GetAtomWithIdx(b.GetEndAtomIdx()).GetAtomicNum() == 6]

def find_dihedrals_and_bond_angles(mol):
    bonded = find_bonded_carbons(mol)
    bonded_dict = {a: [] for pair in bonded for a in pair}
    for a1, a2 in bonded:
        bonded_dict[a1].append(a2)
        bonded_dict[a2].append(a1)

    conf = mol.GetConformer()
    sum_abs_120 = 0.0
    count_angles = 0
    rmsd_sq = 0.0

    for a1 in bonded_dict:
        for a2 in bonded_dict[a1]:
            for a3 in bonded_dict[a2]:
                if a3 != a1:
                    p1, p2, p3 = map(lambda idx: np.array(conf.GetAtomPosition(idx)), (a1, a2, a3))
                    angle = calculate_bond_angle(p1, p2, p3)
                    sum_abs_120 += abs(120.0 - angle)
                    rmsd_sq += (120.0 - angle) ** 2
                    count_angles += 1
    rmsd_bond_angle = np.sqrt(rmsd_sq / count_angles) if count_angles else 0.0
    return sum_abs_120, rmsd_bond_angle

# --- HOMA --------------------------------------------------------------------

def homa_aromatic_rings(mol, alpha=257.7, R_opt=1.388):
    ri = mol.GetRingInfo()
    conf = mol.GetConformer()
    vals = []
    for ring in ri.AtomRings():
        if not all(mol.GetAtomWithIdx(idx).GetIsAromatic() and mol.GetAtomWithIdx(idx).GetAtomicNum() == 6 for idx in ring):
            continue
        bond_lengths = []
        for i in range(len(ring)):
            a1, a2 = ring[i], ring[(i + 1) % len(ring)]
            p1, p2 = map(lambda idx: np.array(conf.GetAtomPosition(idx)), (a1, a2))
            bond_lengths.append(np.linalg.norm(p1 - p2))
        n = len(bond_lengths)
        if n:
            vals.append(1.0 - (alpha / n) * sum((R - R_opt) ** 2 for R in bond_lengths))
    return (float('nan') if not vals else np.mean(vals))

###############################################################################
# Energy model (C40H22 multicomponent fit)
###############################################################################

def predict_isomerization_energy(sum_abs_120, avg_homa, rmsd):
    return (
        0.02082790 * sum_abs_120 +
        -340.97268109 * avg_homa +
        16.64640654 * rmsd +
        236.14120030
    )

###############################################################################
# Streamlit UI ----------------------------------------------------------------
###############################################################################

st.set_page_config(page_title="PAH Isomer Energy Estimator", page_icon="⚛", layout="centered")

st.title("PAH Isomer Energy Estimator")

st.markdown(
    """Upload a 3‑D **XYZ** file of your polycyclic aromatic hydrocarbon and
    we will estimate the relative ΔE based on dihedral flexibility, aromaticity (HOMA) and RMSD of bond angles
    using a linear model trained on C₄₀H₂₂ isomers.
    """
)

uploaded = st.file_uploader("Drag‑and‑drop XYZ here", type="xyz")

if uploaded:
    xyz_text = uploaded.read().decode()
    try:
        mol = load_molecule_from_xyz(xyz_text)
        sum_abs_120, rmsd_ba = find_dihedrals_and_bond_angles(mol)
        avg_homa = homa_aromatic_rings(mol)

        est_E = predict_isomerization_energy(sum_abs_120, avg_homa, rmsd_ba)

        st.subheader("Results")
        st.write(f"**Σ|120°–θ|:** {sum_abs_120:,.2f} deg")
        st.write(f"**RMSD(θ):** {rmsd_ba:,.2f} deg")
        st.write(f"**Avg HOMA:** {avg_homa:.4f}")
        st.success(f"**Predicted ΔE₍D4₎:** {est_E:,.2f} kcal·mol⁻¹")

        with st.expander("Model details"):
            st.code(
                "ΔE = 0.02082790·Σ|120–θ| – 340.97268109·HOMA + 16.64640654·RMSD(θ) + 236.14120030",
                language="python",
            )
            st.caption("Fitted to C₄₀H₂₂ isomer data; R² ≈ 0.88; MAD ≈ 2.0 kcal mol⁻¹.")
    except Exception as e:
        st.error(f"Could not process file: {e}")
else:
    st.info("⬆ Upload an XYZ file to start.")

###############################################################################
# Footer ----------------------------------------------------------------------
###############################################################################

st.write("""<sub>Made at the University of New England.  RDKit 2024.09 • Streamlit 1.33</sub>""", unsafe_allow_html=True)
