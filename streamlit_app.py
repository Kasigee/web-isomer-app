import os
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, AllChem
from rdkit.Chem.rdmolfiles import MolToXYZBlock

# ---------- Page Configuration ----------
st.set_page_config(page_title="Isomerization Energy", layout="centered")
UNE_GREEN, UNE_GOLD = "#00693e", "#ffd400"
st.markdown(f"""<style>
.stApp {{ background-color:{UNE_GREEN}10; }}
.css-1d391kg {{ color:{UNE_GREEN}; }}
.css-1v3fvcr {{ background:{UNE_GOLD}33; }}
</style>""", unsafe_allow_html=True)

# ---------- Helpers ----------
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
    n1 = np.cross(b1, b2); n2 = np.cross(b2, b3)
    n1 /= np.linalg.norm(n1); n2 /= np.linalg.norm(n2)
    return np.degrees(np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0)))


def analyze_geometry(mol):
    from collections import defaultdict
    bonded = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()
               if mol.GetAtomWithIdx(b.GetBeginAtomIdx()).GetAtomicNum() == 6
               and mol.GetAtomWithIdx(b.GetEndAtomIdx()).GetAtomicNum() == 6]
    neighbors = defaultdict(list)
    for a, b in bonded:
        neighbors[a].append(b); neighbors[b].append(a)
    conf = mol.GetConformer()
    sum_less = sum_greater = sum_sq = 0.0
    count = 0
    seen = set()
    for a1 in neighbors:
        for a2 in neighbors[a1]:
            for a3 in neighbors[a2]:
                if a3 == a1: continue
                p1 = np.array(conf.GetAtomPosition(a1))
                p2 = np.array(conf.GetAtomPosition(a2))
                p3 = np.array(conf.GetAtomPosition(a3))
                angle = calculate_bond_angle(p1, p2, p3)
                sum_sq += (120 - angle) ** 2; count += 1
                for a4 in neighbors[a3]:
                    if a4 in (a1, a2): continue
                    key = tuple(sorted((a1, a2, a3, a4)))
                    if key in seen: continue
                    seen.add(key)
                    p4 = np.array(conf.GetAtomPosition(a4))
                    dih = calculate_dihedral(p1, p2, p3, p4)
                    if dih < 90:
                        sum_less += dih
                    else:
                        sum_greater += (180 - dih)
    sum_dev = sum_less + sum_greater
    rmsd = np.sqrt(sum_sq / count) if count > 0 else 0.0
    return sum_dev, rmsd


def homa_aromatic_rings(mol, alpha=257.7, R_opt=1.388):
    rings = mol.GetRingInfo().AtomRings()
    conf = mol.GetConformer()
    homas = []
    for ring in rings:
        if all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 and mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            lengths = [np.linalg.norm(np.array(conf.GetAtomPosition(ring[i])) -
                                      np.array(conf.GetAtomPosition(ring[(i+1) % len(ring)])))
                       for i in range(len(ring))]
            n = len(lengths)
            sum_sq = sum((L - R_opt)**2 for L in lengths)
            homas.append(1 - (alpha / n) * sum_sq)
    return float(np.mean(homas)) if homas else 0.0


def get_db_energies(smi):
    specs = [
        ("COMPAS_XTB_MS_WEBAPP_DATA.csv", "D4_rel_energy", "PBE0-D4/6-31G(2df,p)//GFN2-xTB"),
        ("compas-3D.csv", "Erel_eV", "CAM-B3LYP-D3BJ/cc-pvdz//CAM-B3LYP-D3BJ/def2-SVP"),
        ("compas-3x.csv", "Erel_eV", "GFN2-xTB")
    ]
    out = []
    for fn, col, label in specs:
        if os.path.exists(fn):
            df = pd.read_csv(fn)
            if col in df.columns:
                match = df.loc[df.smiles == smi]
                if not match.empty:
                    val = float(match.iloc[0][col])
                    if col == "Erel_eV":
                        val *= 96.485
                    out.append((label, val))
                    continue
        out.append((label, None))
    return out

# ---------- Models ----------
def model_dhr(sum_dev, homa, rmsd):
    return (0.02082790*sum_dev - 340.97268109*homa + 16.64640654*rmsd + 236.14120030,
            "E=0.0208·ΣDihedral-340.97·HOMA+16.65·θRMSD+236.14")

def model_dihedral(sum_dev):
    return (0.01506654*sum_dev + 5.26542057, "E=0.01507·ΣDihedral+5.2654")

def model_homa(homa):
    return (-83.95374901*homa + 81.47198711, "E=-83.95·HOMA+81.47")

def model_dh(sum_dev, homa):
    return (0.02230771*sum_dev - 361.11764751*homa + 275.86109778,
            "E=0.02231·ΣDihedral-361.12·HOMA+275.86")

def model_xtb(xtb):
    return (1.43801209*xtb + 0.57949987, "E=1.4380·XTB+0.5795")

def model_dx(sum_dev, xtb):
    return (0.00678795*sum_dev + 1.07126936*xtb + 3.49502511,
            "E=0.00679·ΣDihedral+1.0713·XTB+3.4950")


def smiles_to_xyz(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3(); params.randomSeed = 42
    AllChem.EmbedMolecule(mol, params)
    AllChem.MMFFOptimizeMolecule(mol)
    return MolToXYZBlock(mol)

# ---------- App ----------
def main():
    st.title("Isomerization Energy Predictor")

    # Session state initialization
    for key in ('xyz', 'prev', 'last_comp'):
        st.session_state.setdefault(key, None)

    # File upload
    uploaded = st.file_uploader("Upload XYZ file", type="xyz")
    if uploaded:
        xyz_text = uploaded.read().decode()
        if xyz_text != st.session_state.prev:
            st.session_state.xyz = xyz_text
            st.session_state.prev = xyz_text

    # Sidebar model selector
    models_list = [
        "Dihedral+HOMA+θRMSD",
        "Dihedral-only",
        "HOMA-only",
        "Dihedral+HOMA",
        "XTB-only",
        "Dihedral+XTB"
    ]
    choice = st.sidebar.selectbox("Select model:", models_list, key='model')

    # XTB input when needed
    xtb_val = None
    if choice in ("XTB-only", "Dihedral+XTB"):
        xtb_val = st.sidebar.number_input("XTB energy (kJ/mol)", value=0.0, key='xtb_val')

    # Trigger computation
    new_xyz = st.session_state.xyz and st.session_state.xyz != st.session_state.last_comp
    if st.sidebar.button("Calculate") or new_xyz:
        st.session_state.last_comp = st.session_state.xyz
    else:
        return

    # Load and analyze
    mol = load_molecule_from_xyz(st.session_state.xyz)
    if mol is None:
        return

    smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
    st.markdown(f"**SMILES:** {smi}")

    sum_dev, rmsd = analyze_geometry(mol)
    homa = homa_aromatic_rings(mol)

    # Prediction dispatch
    dispatch = {
        "Dihedral+HOMA+θRMSD": (model_dhr, sum_dev, homa, rmsd),
        "Dihedral-only":        (model_dihedral, sum_dev),
        "HOMA-only":            (model_homa, homa),
        "Dihedral+HOMA":        (model_dh, sum_dev, homa),
        "XTB-only":             (model_xtb, xtb_val),
        "Dihedral+XTB":         (model_dx, sum_dev, xtb_val)
    }
    fn, *args = dispatch[choice]
    E, eq = fn(*args)

    # Display results
    st.markdown(f"**Equation:** {eq}")
    metrics = [f"ΣDihedral: {sum_dev:.3f}", f"HOMA: {homa:.3f}", f"θRMSD: {rmsd:.3f}"]
    if xtb_val is not None:
        metrics.append(f"XTB: {xtb_val:.3f}")
    st.markdown("   ".join([f"**{m}**" for m in metrics]))
    st.markdown(f"## Predicted ΔE = {E:.3f} kJ/mol")

    # Database lookups
    st.subheader("Database Energies")
    for label, val in get_db_energies(smi):
        if val is None:
            st.info(f"No match in {label}")
        else:
            st.markdown(f"**{label}: {val:.3f} kJ/mol**")

    # Original XYZ viewer
    html1 = (
        "<div id='v1' style='width:400px;height:300px'></div>"
        "<script src='https://3Dmol.org/build/3Dmol.js'></script>"
        "<script>"
        "var v = $3Dmol.createViewer('v1',{backgroundColor:'0xeeeeee'});"
        f"v.addModel(`{st.session_state.xyz}`,'xyz');"
        "v.setStyle({stick:{}});v.rotate(1,90);v.zoomTo();v.render();"
        "</script>"
    )
    components.html(html1, height=320)

        # Compare to lowest‐energy isomer from PBE0 database
    try:
        df_db = pd.read_csv("COMPAS_XTB_MS_WEBAPP_DATA.csv")
        # get uploaded record to infer prefix
        uploaded_file = df_db.loc[df_db.smiles == smi, 'file'].iloc[0]
        # prefix is first two segments
        parts = uploaded_file.split('_')
        prefix = '_'.join(parts[:2])
        # find all matching isomers
        df_pref = df_db[df_db.file.str.startswith(prefix)]
        if not df_pref.empty:
            # lowest-energy isomer
            row_min = df_pref.loc[df_pref.D4_rel_energy.idxmin()]
            low_fname = row_min.file + '.xyz'
            if os.path.exists(low_fname):
                st.subheader(f"Compared to lowest energy isomer below (with SMILES string {row_min.smiles}):")
                with open(low_fname) as f:
                    xyz_low = f.read()
                html2 = (
                    "<div id='v2' style='width:400px;height:300px'></div>"
                    "<script src='https://3Dmol.org/build/3Dmol.js'></script>"
                    "<script>"
                    "var v2 = $3Dmol.createViewer('v2',{backgroundColor:'0xeeeeee'});"
                    f"v2.addModel(`{xyz_low}`,'xyz');"
                    "v2.setStyle({stick:{}});v2.rotate(1,90);v2.zoomTo();v2.render();"
                    "</script>"
                )
                components.html(html2, height=320)
            else:
                st.info(f"XYZ file for lowest isomer '{low_fname}' not found.")
        else:
            st.info(f"No other isomers found with prefix '{prefix}'.")
    except Exception as e:
        st.warning("Could not load lowest-energy isomer structure: " + str(e))

if __name__ == '__main__':
    main()
