import os
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

# ---------- Page Configuration ----------
st.set_page_config(page_title="Isomerization Energy", layout="centered")
# UNE branding colors
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
    from collections import defaultdict
    bonded = find_bonded_carbons(mol)
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
                p1, p2, p3 = np.array(conf.GetAtomPosition(a1)), np.array(conf.GetAtomPosition(a2)), np.array(conf.GetAtomPosition(a3))
                angle = calculate_bond_angle(p1, p2, p3)
                sum_sq += (120 - angle)**2; count += 1
                for a4 in neighbors[a3]:
                    if a4 in (a1, a2): continue
                    key = tuple(sorted((a1,a2,a3,a4)))
                    if key in seen: continue
                    seen.add(key)
                    p4 = np.array(conf.GetAtomPosition(a4))
                    dih = calculate_dihedral(p1,p2,p3,p4)
                    if dih < 90: sum_less += dih
                    else: sum_greater += (180 - dih)
    sum_dev = sum_less + sum_greater
    rmsd = np.sqrt(sum_sq/count) if count>0 else 0.0
    return sum_dev, rmsd


def homa_aromatic_rings(mol, alpha=257.7, R_opt=1.388):
    rings = mol.GetRingInfo().AtomRings(); conf = mol.GetConformer(); homas=[]
    for ring in rings:
        if all(mol.GetAtomWithIdx(i).GetAtomicNum()==6 and mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            lengths=[ np.linalg.norm(np.array(conf.GetAtomPosition(ring[i]))-np.array(conf.GetAtomPosition(ring[(i+1)%len(ring)]))) for i in range(len(ring)) ]
            n=len(lengths); sum_sq=sum((L-R_opt)**2 for L in lengths)
            homas.append(1-(alpha/n)*sum_sq)
    return float(np.mean(homas)) if homas else 0.0

# ---------- Database Energies ----------
def get_db_energies(smi):
    specs = [
        ("COMPAS_XTB_MS_WEBAPP_DATA.csv", "D4_rel_energy", "PBE0-D4/6-31G(2df,p)//GFN2-xTB"),
        ("compas-3D.csv", "Erel_eV", "CAM-B3LYP-D3BJ/cc-pvdz//CAM-B3LYP-D3BJ/def2-SVP"),
        ("compas-3x.csv", "xtb_iso_energy", "GFN2-xTB")
    ]
    out=[]
    for fn, col, label in specs:
        if not os.path.exists(fn):
            out.append((label, None))
            continue
        df = pd.read_csv(fn)
        if col not in df.columns:
            out.append((label, None))
            continue
        m = df.loc[df.smiles == smi]
        if m.empty:
            out.append((label, None))
        else:
            try:
                val = float(m.iloc[0][col])
                if col == "Erel_eV":
                    val *= 96.485
                out.append((label, val))
            except Exception:
                out.append((label, None))
    return out

# ---------- Models ----------
def model_dhr(sum_dev,homa,rmsd): A_d,A_h,A_r,C=0.02082790,-340.97268109,16.64640654,236.14120030; eq="E=0.0208279·ΣDihedral-340.9727·HOMA+16.6464·θRMSD+236.1412"; return A_d*sum_dev+A_h*homa+A_r*rmsd+C,eq

def model_dihedral(sum_dev): A,B=0.01506654,5.26542057; eq="E=0.01506654·ΣDihedral+5.2654"; return A*sum_dev+B,eq

def model_homa(homa): A,B=-83.95374901,81.47198711; eq="E=-83.9537·HOMA+81.4720"; return A*homa+B,eq

def model_dh(sum_dev,homa): A_d,A_h,B=0.02230771,-361.11764751,275.86109778; eq="E=0.0223077·ΣDihedral-361.1176·HOMA+275.8611"; return A_d*sum_dev+A_h*homa+B,eq

def model_xtb(xtb): A,B=1.43801209,0.57949987; eq="E=1.4380·XTB+0.5795"; return A*xtb+B,eq

def model_dx(sum_dev,xtb): A_d,B_x,C=0.00678795,1.07126936,3.49502511; eq="E=0.006788·ΣDihedral+1.0713·XTB+3.4950"; return A_d*sum_dev+B_x*xtb+C,eq

# ---------- Main ----------
def main():
    st.title("Isomerization Energy Predictor")

    # Upload widget
    if 'xyz' not in st.session_state:
        st.session_state.xyz = None
        st.session_state.prev = None
        st.session_state.last_comp = None
    uploaded = st.file_uploader("Upload XYZ file", type="xyz")
    if uploaded:
        txt = uploaded.read().decode()
        if txt != st.session_state.prev:
            st.session_state.xyz = txt
            st.session_state.prev = txt

    # Sidebar inputs
    models = [
        "Dihedral+HOMA+θRMSD",
        "Dihedral-only",
        "HOMA-only",
        "Dihedral+HOMA",
        "XTB-only",
        "Dihedral+XTB"
    ]
    choice = st.sidebar.selectbox("Model:", models, key='model')
    xtb_val = None
    if choice in ("XTB-only", "Dihedral+XTB"):
        xtb_val = st.sidebar.number_input("XTB energy (kJ/mol)", value=0.0, key='xtb_energy')

    compute = False
    # auto-run on new upload
    if st.session_state.xyz and st.session_state.xyz != st.session_state.last_comp:
        compute = True
    # manual trigger
    if st.sidebar.button("Calculate"):
        compute = True

    if not compute:
        return

    # mark computed
    st.session_state.last_comp = st.session_state.xyz

    # Load molecule
    mol = load_molecule_from_xyz(st.session_state.xyz)
    if not mol:
        return

    # Compute metrics
    smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
    st.markdown(f"**SMILES:** {smi}")
    sum_dev, rmsd = analyze_geometry(mol)
    homa = homa_aromatic_rings(mol)

    # Prediction
    if choice == "Dihedral+HOMA+θRMSD":
        E, eq = model_dhr(sum_dev, homa, rmsd)
    elif choice == "Dihedral-only":
        E, eq = model_dihedral(sum_dev)
    elif choice == "HOMA-only":
        E, eq = model_homa(homa)
    elif choice == "Dihedral+HOMA":
        E, eq = model_dh(sum_dev, homa)
    elif choice == "XTB-only":
        E, eq = model_xtb(xtb_val)
    else:  # Dihedral+XTB
        E, eq = model_dx(sum_dev, xtb_val)

    st.markdown(f"**Equation:** {eq}")
    parts = [f"ΣDihedral: {sum_dev:.3f}", f"HOMA: {homa:.3f}", f"θRMSD: {rmsd:.3f}"]
    if xtb_val is not None:
        parts.append(f"XTB: {xtb_val:.3f}")
    st.markdown("   ".join([f"**{p}**" for p in parts]))
    st.markdown(f"## Predicted ΔE = {E:.3f} kJ/mol")

    # Database Energies
    st.subheader("Database Energies")
    for label, val in get_db_energies(smi):
        if val is None:
            st.info(f"No match in {label}")
        else:
            st.markdown(f"**{label}: {val:.3f} kJ/mol**")

    # 3D viewer
    xyz_txt = st.session_state.xyz
    html = (
        "<div id='view' style='width:400px;height:300px'></div>"
        "<script src='https://3Dmol.org/build/3Dmol.js'></script>"
        "<script>"
        "var v = $3Dmol.createViewer('view',{backgroundColor:'0xeeeeee'});"
        "v.addModel(`" + xyz_txt + "`,'xyz');"
        "v.setStyle({stick:{}});"
        "v.rotate(1,90);v.zoomTo();v.render();"
        "</script>"
    )
    components.html(html, height=320)

    # 2D depiction
    from rdkit.Chem import Draw
    mol2d = Chem.MolFromSmiles(smi)
    if mol2d:
        img = Draw.MolToImage(mol2d, size=(300, 300))
        st.subheader("2D Structure (from SMILES)")
        st.image(img, use_column_width=False)

if __name__ == '__main__':
    main()
