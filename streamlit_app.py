import os
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdDetermineBonds, AllChem
from rdkit.Chem.rdmolfiles import MolToXYZBlock

# ---------- Page Configuration ----------
st.set_page_config(page_title="Isomerization Energy", layout="centered")
# UNE branding colors
UNE_GREEN = "#00693e"
UNE_GOLD  = "#ffd400"
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
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))


def calculate_dihedral(p1, p2, p3, p4):
    b1, b2, b3 = p2 - p1, p3 - p2, p4 - p3
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)
    return np.degrees(np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0)))


def analyze_geometry(mol):
    # Find all bonded carbon pairs and build neighbor dict
    from collections import defaultdict
    neighbors = defaultdict(list)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if mol.GetAtomWithIdx(i).GetAtomicNum() == 6 and mol.GetAtomWithIdx(j).GetAtomicNum() == 6:
            neighbors[i].append(j)
            neighbors[j].append(i)
    conf = mol.GetConformer()
    sum_less = sum_greater = sum_sq = 0.0
    count = 0
    seen_dihedrals = set()
    for a1, nbrs in neighbors.items():
        for a2 in nbrs:
            for a3 in neighbors[a2]:
                if a3 == a1:
                    continue
                # Bond angle
                p1 = np.array(conf.GetAtomPosition(a1))
                p2 = np.array(conf.GetAtomPosition(a2))
                p3 = np.array(conf.GetAtomPosition(a3))
                angle = calculate_bond_angle(p1, p2, p3)
                sum_sq += (120 - angle) ** 2
                count += 1
                # Dihedral
                for a4 in neighbors[a3]:
                    if a4 in (a1, a2):
                        continue
                    key = tuple(sorted((a1, a2, a3, a4)))
                    if key in seen_dihedrals:
                        continue
                    seen_dihedrals.add(key)
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
    homa_vals = []
    for ring in rings:
        # only aromatic carbon rings
        if all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 and mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            lengths = []
            for idx in ring:
                nxt = ring[(ring.index(idx)+1) % len(ring)]
                p1 = np.array(conf.GetAtomPosition(idx))
                p2 = np.array(conf.GetAtomPosition(nxt))
                lengths.append(np.linalg.norm(p1 - p2))
            n = len(lengths)
            sum_sq = sum((L - R_opt) ** 2 for L in lengths)
            homa_vals.append(1.0 - (alpha / n) * sum_sq)
    return float(np.mean(homa_vals)) if homa_vals else 0.0

# ---------- Database Loading ----------
DB_SPECS = [
    ('COMPAS_XTB_MS_WEBAPP_DATA.csv', 'D4_rel_energy', 'PBE0-D4/6-31G(2df,p)'),
    ('compas-3D.csv',                'Erel_eV',      'CAM-B3LYP-D3BJ/...'),
    ('compas-3x.csv',               'Erel_eV','GFN2-xTB')
]


def load_db(path, energy_col):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    # compute formula column
    df['formula'] = df['smiles'].apply(lambda s: rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(s)))
    return df

# Load all DBS once
DB_DFS = [(load_db(fn, col), col, label) for fn, col, label in DB_SPECS]

# ---------- Models ----------
MODEL_FUNCS = {
    'Dihedral-only':       (lambda sd,h,r,x: 0.01506654*sd + 5.26542057, 'E = 0.01506654·ΣDihedral + 5.26542057'),
    'HOMA-only':           (lambda sd,h,r,x: -83.95374901*h + 81.47198711, 'E = -83.9537·HOMA + 81.4720'),
    'Dihedral + HOMA':     (lambda sd,h,r,x: 0.02230771*sd -361.11764751*h + 275.86109778, 'E = 0.02230771·ΣDihedral -361.1176·HOMA +275.8611'),
    'Dihedral + XTB':      (lambda sd,h,r,x: 0.00311220*sd +1.31031476*x -0.41296296, 'E = 0.00311220·ΣDihedral +1.31031476·XTB -0.41296296'),
    'HOMA + XTB':          (lambda sd,h,r,x: 29.41245664*h +1.37801523*x -15.68808658,'E = 29.4125·HOMA +1.3780·XTB -15.6881'),
    'D + H + XTB':         (lambda sd,h,r,x: 0.00680599*sd -0.58577103*h +1.07051358*x +3.93466134,'E = 0.006806·ΣDihedral -0.5858·HOMA +1.0705·XTB +3.9347'),
    'D + H + θRMSD':       (lambda sd,h,r,x: 0.02082790*sd -340.97268109*h +16.64640654*r +236.14120030,'E = 0.0208279·ΣDihedral -340.9727·HOMA +16.6464·θRMSD +236.1412')
}

# ---------- Streamlit App ----------

def main():
    st.title("Isomerization Energy Predictor")

    uploaded = st.file_uploader("Upload XYZ file", type="xyz")
    if not uploaded:
        return

    xyz_txt = uploaded.read().decode()
    mol = load_molecule_from_xyz(xyz_txt)
    if mol is None:
        return

    # Sidebar
    choice = st.sidebar.selectbox("Model:", list(MODEL_FUNCS.keys()), index=6)
    xtb_val = None
    if 'XTB' in choice:
        xtb_val = st.sidebar.number_input("XTB energy (kJ/mol)", value=0.0)

    # Compute geometry & HOMA
    sum_dev, rmsd = analyze_geometry(mol)
    homa_avg = homa_aromatic_rings(mol)

    # Predict
    func, eq = MODEL_FUNCS[choice]
    E_pred = func(sum_dev, homa_avg, rmsd, xtb_val or 0.0)

    st.markdown(f"**Equation:** `{eq}`")
    st.markdown(f"**ΣDihedral:** {sum_dev:.3f}  **HOMA:** {homa_avg:.3f}  **θRMSD:** {rmsd:.3f}  **XTB:** {xtb_val or '–'}")
    st.markdown(f"## Predicted ΔE = {E_pred:.3f} kJ/mol")

    # Database matches for uploaded
    st.subheader("Database Matches (uploaded)")
    base_smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
    base_formula = rdMolDescriptors.CalcMolFormula(mol)
    for df, col, label in DB_DFS:
        if df.empty:
            st.warning(f"Missing DB: {label}")
        else:
            match = df[df.smiles == base_smiles]
            if not match.empty:
                val = float(match.iloc[0][col])
                if col == 'Erel_eV':
                    val *= 96.485
                st.markdown(f"**{label}:** {val:.3f} kJ/mol")
            else:
                st.info(f"No match in {label}")

    # Compare lowest-energy isomers
    st.subheader("Lowest-Energy Isomer per DB")
    cols = st.columns(len(DB_DFS) + 1)
    # show uploaded
    with cols[0]:
        st.markdown("**Uploaded**")
        st.code(base_smiles)
        html0 = (
            "<div id='v0' style='width:200px;height:200px'></div>"
            "<script src='https://3Dmol.org/build/3Dmol.js'></script>"
            f"<script>v=$3Dmol.createViewer('v0',{{backgroundColor:'0xeeeeee'}});"
            f"v.addModel(`{xyz_txt}`,'xyz');"
            "v.setStyle({stick:{}});v.rotate(1,90);v.zoomTo();v.render();</script>"
        )
        components.html(html0, height=200)

    for i, (df, col, label) in enumerate(DB_DFS, start=1):
        with cols[i]:
            if df.empty:
                st.warning(label)
                continue
            subset = df[df.formula == base_formula]
            if subset.empty:
                st.info(f"No {label}")
                continue
            # pick lowest
            if col == 'Erel_eV':
                subset = subset.copy()
                subset['kj'] = subset[col].astype(float) * 96.485
                idx = subset['kj'].idxmin()
                energy = subset.loc[idx, 'kj']
            else:
                idx = subset[col].astype(float).idxmin()
                energy = subset.loc[idx, col]
            smi_low = subset.loc[idx, 'smiles']
            # embed geometry
            mlow = Chem.AddHs(Chem.MolFromSmiles(smi_low))
            AllChem.EmbedMolecule(mlow, randomSeed=0xf00d)
            xyz_low = MolToXYZBlock(mlow)
            st.markdown(f"**{label}**
{smi_low}
{energy:.3f} kJ/mol")
            html = (
                "<div id='v" + str(i) + "' style='width:200px;height:200px'></div>"
                "<script src='https://3Dmol.org/build/3Dmol.js'></script>"
                f"<script>v=$3Dmol.createViewer('v{i}',{{backgroundColor:'0xeeeeee'}});"
                f"v.addModel(`{xyz_low}`,'xyz');"
                "v.setStyle({stick:{}});v.rotate(1,90);v.zoomTo();v.render();</script>"
            )
            components.html(html, height=200)

if __name__ == '__main__':
    main()
