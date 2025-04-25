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

# Optional: UNE branding
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
    v1, v2 = p1-p2, p3-p2
    v1 /= np.linalg.norm(v1); v2 /= np.linalg.norm(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))


def calculate_dihedral(p1, p2, p3, p4):
    b1, b2, b3 = p2-p1, p3-p2, p4-p3
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    n1 /= np.linalg.norm(n1); n2 /= np.linalg.norm(n2)
    return np.degrees(np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0)))


def analyze_geometry(mol):
    from collections import defaultdict
    # build neighbor list for C-C bonds
    neighbors = defaultdict(list)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if mol.GetAtomWithIdx(i).GetAtomicNum()==6 and mol.GetAtomWithIdx(j).GetAtomicNum()==6:
            neighbors[i].append(j); neighbors[j].append(i)
    conf = mol.GetConformer()
    sum_less=0.0; sum_greater=0.0; sum_sq=0.0; count=0; dihed_set=set()
    for a1, nbrs in neighbors.items():
        for a2 in nbrs:
            for a3 in neighbors[a2]:
                if a3==a1: continue
                p1, p2, p3 = map(lambda i: np.array(conf.GetAtomPosition(i)), (a1,a2,a3))
                ang = calculate_bond_angle(p1,p2,p3)
                sum_sq += (120-ang)**2; count+=1
                for a4 in neighbors[a3]:
                    if a4 in (a1,a2): continue
                    key = tuple(sorted((a1,a2,a3,a4)))
                    if key in dihed_set: continue
                    dihed_set.add(key)
                    p4 = np.array(conf.GetAtomPosition(a4))
                    dih = calculate_dihedral(p1,p2,p3,p4)
                    if dih < 90: sum_less += dih
                    else:        sum_greater += (180-dih)
    sum_dev = sum_less + sum_greater
    rmsd    = np.sqrt(sum_sq/count) if count>0 else 0.0
    return sum_dev, rmsd


def homa_aromatic_rings(mol, alpha=257.7, R_opt=1.388):
    rings = mol.GetRingInfo().AtomRings(); conf = mol.GetConformer(); homas = []
    for ring in rings:
        if all(mol.GetAtomWithIdx(i).GetAtomicNum()==6 and mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            lengths = [np.linalg.norm(np.array(conf.GetAtomPosition(ring[i]))-
                        np.array(conf.GetAtomPosition(ring[(i+1)%len(ring)]))) for i in range(len(ring))]
            n = len(lengths); sum_sq = sum((L-R_opt)**2 for L in lengths)
            homas.append(1 - (alpha/n)*sum_sq)
    avg = float(np.mean(homas)) if homas else 0.0
    return avg

# ---------- Database Lookup (three sources) ----------
def load_db(path, energy_col):
    df = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    if not df.empty:
        # add formula column
        df['formula'] = df['smiles'].apply(lambda s: Chem.CalcMolFormula(Chem.MolFromSmiles(s)))
    return df

DB_SPECS = [
    ('COMPAS_XTB_MS_WEBAPP_DATA.csv', 'D4_rel_energy', 'PBE0-D4/6-31G(2df,p)'),
    ('compas-3D.csv',          'Erel_eV',         'CAM-B3LYP-D3BJ/cc-pvdz//CAM-B3LYP-D3BJ/def2-SVP'),
    ('compas-3x.csv',         'Erel_eV',  'GFN2-xTB')
]

# load all DBs
DB_DFS = []
for fn, col, label in DB_SPECS:
    df = load_db(fn, col)
    DB_DFS.append((df, col, label))

# ---------- Models ----------
MODEL_FUNCS = {
    'Dihedral-only':       (lambda sum_dev,homa,rmsd,xtb: (0.01506654*sum_dev + 5.26542057),
                             'E = 0.01506654·ΣDihedral + 5.26542057'),
    'HOMA-only':           (lambda sum_dev,homa,rmsd,xtb: (-83.95374901*homa + 81.47198711),
                             'E = -83.9537·HOMA + 81.4720'),
    'Dihedral + HOMA':     (lambda sum_dev,homa,rmsd,xtb: (0.02230771*sum_dev -361.11764751*homa + 275.86109778),
                             'E = 0.02230771·ΣDihedral - 361.1176·HOMA + 275.8611'),
    'Dihedral + XTB':      (lambda sum_dev,homa,rmsd,xtb: (0.00311220*sum_dev + 1.31031476*xtb -0.41296296),
                             'E = 0.00311220·ΣDihedral + 1.31031476·XTB - 0.41296296'),
    'HOMA + XTB':          (lambda sum_dev,homa,rmsd,xtb: (29.41245664*homa +1.37801523*xtb -15.68808658),
                             'E = 29.4125·HOMA + 1.3780·XTB - 15.6881'),
    'D + H + XTB':         (lambda sum_dev,homa,rmsd,xtb: (0.00680599*sum_dev -0.58577103*homa +1.07051358*xtb +3.93466134),
                             'E = 0.006806·ΣDihedral -0.5858·HOMA + 1.0705·XTB + 3.9347'),
    'D + H + θRMSD':       (lambda sum_dev,homa,rmsd,xtb: (0.02082790*sum_dev -340.97268109*homa +16.64640654*rmsd +236.14120030),
                             'E = 0.0208279·ΣDihedral -340.9727·HOMA +16.6464·θRMSD +236.1412')
}

# ---------- Streamlit App ----------

def main():
    st.title("Isomerization Energy Predictor")

    # Uploader
    xyz_txt = None
    uploaded = st.file_uploader("Upload XYZ file", type="xyz")
    if uploaded:
        xyz_txt = uploaded.read().decode()
    if not xyz_txt:
        return
    mol = load_molecule_from_xyz(xyz_txt)
    if not mol:
        return

    # Sidebar inputs
    choice = st.sidebar.selectbox("Model:", list(MODEL_FUNCS.keys()), index=6)
    xtb_val = None
    if 'XTB' in choice:
        xtb_val = st.sidebar.number_input("XTB energy (kJ/mol):", value=0.0)

    # Compute geometry metrics
    sum_dev, rmsd = analyze_geometry(mol)
    homa_avg     = homa_aromatic_rings(mol)
    homa_avg     = 0.0 if np.isnan(homa_avg) else homa_avg

    # Prediction
    func, eq = MODEL_FUNCS[choice]
    E_pred = func(sum_dev, homa_avg, rmsd, xtb_val or 0.0)

    st.markdown(f"**Equation:** `{eq}`")
    st.markdown(f"**ΣDihedral:** {sum_dev:.3f}  **HOMA:** {homa_avg:.3f}  **θRMSD:** {rmsd:.3f}  **XTB:** {xtb_val or '–'}")
    st.markdown(f"## Predicted ΔE = {E_pred:.3f} kJ/mol")

    # Database matches for uploaded isomer
    st.subheader("Database Matches (uploaded isomer)")
    base_smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
    base_formula= Chem.CalcMolFormula(mol)
    for df, col, label in DB_DFS:
        if df.empty:
            st.warning(f"Database file missing for {label}")
            continue
        match = df.loc[df.smiles == base_smiles]
        if not match.empty:
            val = float(match[col].iloc[0]); unit = 'eV' if col=='Erel_eV' else 'kJ/mol'
            if col=='Erel_eV': val *= 96.485; unit='kJ/mol'
            st.markdown(f"**{label}:** {val:.3f} {unit}")
        else:
            st.info(f"No match in {label} for this isomer.")

    # Find and display lowest-energy isomers per DB
    st.subheader("Lowest‑Energy Isomers (base formula)")
    cols = st.columns(len(DB_DFS)+1)
    cols[0].markdown("**Uploaded**")
    cols[0].code(base_smiles)
    # display uploaded viewer
    html0 = f"<div id='v0' style='width:200px;height:200px;'></div>" + \
            "<script src='https://3Dmol.org/build/3Dmol.js'></script>" + \
            f"<script>v=$3Dmol.createViewer('v0',{{backgroundColor:'0xeeeeee'}});" + \
            f"v.addModel(`{xyz_txt}`,'xyz');v.setStyle({{stick:{{}}}});v.rotate(1,90);v.zoomTo();v.render();</script>"
    cols[0].components.html(html0, height=200)

    for i,(df,col,label) in enumerate(DB_DFS, start=1):
        if df.empty or 'formula' not in df.columns:
            cols[i].warning(f"Missing data for {label}")
            continue
        subset = df.loc[df.formula == base_formula]
        if subset.empty:
            cols[i].info(f"No {label} for formula {base_formula}")
            continue
        # find lowest energy
        if col=='Erel_eV':
            subset['val_kj']= subset[col].astype(float)*96.485
            idx = subset['val_kj'].idxmin(); energy = subset.loc[idx,'val_kj']
        else:
            idx = subset[col].astype(float).idxmin(); energy = subset.loc[idx,col]
        smi_low = subset.loc[idx,'smiles']
        # attempt to get XYZ: embed from SMILES
        mol_low = Chem.MolFromSmiles(smi_low)
        mol_low = Chem.AddHs(mol_low)
        AllChem.EmbedMolecule(mol_low,randomSeed=0xf00d)
        xyz_low = MolToXYZBlock(mol_low)
        cols[i].markdown(f"**{label}**\n{smi_low}\nEnergy: {energy:.3f} kJ/mol")
        html = f"<div id='v{i}' style='width:200px;height:200px;'></div>" + \
               "<script src='https://3Dmol.org/build/3Dmol.js'></script>" + \
               f"<script>v=$3Dmol.createViewer('v{i}',{{backgroundColor:'0xeeeeee'}});" + \
               f"v.addModel(`{xyz_low}`,'xyz');v.setStyle({{stick:{{}}}});v.rotate(1,90);v.zoomTo();v.render();</script>"
        cols[i].components.html(html, height=200)

if __name__=='__main__':
    main()
