import os
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
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))


def calculate_dihedral(p1, p2, p3, p4):
    b1, b2, b3 = p2 - p1, p3 - p2, p4 - p3
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    n1 /= np.linalg.norm(n1);
    n2 /= np.linalg.norm(n2)
    return np.degrees(np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0)))


def find_bonded_carbons(mol):
    pairs=[]
    for bond in mol.GetBonds():
        i,j=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
        if mol.GetAtomWithIdx(i).GetAtomicNum()==6 and mol.GetAtomWithIdx(j).GetAtomicNum()==6:
            pairs.append((i,j))
    return pairs


def analyze_geometry(mol):
    conf=mol.GetConformer()
    bonded=find_bonded_carbons(mol)
    neighbors={idx:[nb.GetIdx() for nb in mol.GetAtomWithIdx(idx).GetNeighbors()]
               for atom in mol.GetAtoms() if atom.GetAtomicNum()==6
               for idx in [atom.GetIdx()]}
    angles=[]
    for a1,a2 in bonded:
        for a3 in neighbors.get(a2,[]):
            if a3==a1: continue
            p1,p2,p3=(np.array(conf.GetAtomPosition(i)) for i in (a1,a2,a3))
            angles.append(calculate_bond_angle(p1,p2,p3))
    sum_dev=sum(abs(120-a) for a in angles)
    rmsd=np.sqrt(np.mean([(120-a)**2 for a in angles])) if angles else 0.0
    return sum_dev,rmsd


def homa_aromatic_rings(mol,alpha=257.7,R_opt=1.388):
    rings=mol.GetRingInfo().AtomRings();conf=mol.GetConformer();homas=[]
    for ring in rings:
        if all(mol.GetAtomWithIdx(i).GetAtomicNum()==6 and mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            lengths=[np.linalg.norm(np.array(conf.GetAtomPosition(ring[i]))-
                                     np.array(conf.GetAtomPosition(ring[(i+1)%len(ring)])))
                     for i in range(len(ring))]
            n=len(lengths);sum_sq=sum((L-R_opt)**2 for L in lengths)
            homas.append(1-(alpha/n)*sum_sq)
    return (np.nan,[]) if not homas else (float(np.mean(homas)),homas)


def find_database_energy(mol,csv_file="COMPAS_XTB_MS_WEBAPP_DATA.csv"):
    if not os.path.exists(csv_file): return None,False
    smi=Chem.MolToSmiles(Chem.RemoveHs(mol))
    df=pd.read_csv(csv_file)
    match=df.loc[df.smiles==smi]
    if not match.empty: return float(match.D4_rel_energy.iloc[0]),True
    return None,True

# ---------- Prediction Models ----------

def model_dhr(sum_dev,homa,rmsd):
    A_d,A_h,A_r,C=0.02082790,-340.97268109,16.64640654,236.14120030
    eq="E = 0.02082790·ΣDihedral - 340.9727·HOMA + 16.6464·θRMSD + 236.1412"
    return (A_d*sum_dev + A_h*homa + A_r*rmsd + C),None,None,eq
# ...define other models as needed

# ---------- Streamlit App ----------

def main():
    st.set_page_config(page_title="Isomerization Energy",layout="centered")
    st.title("Isomerization Energy Predictor")

    # Persist last uploaded for auto-calc
    if 'xyz_text' not in st.session_state:
        st.session_state.xyz_text=None
        st.session_state.prev_text=None
    uploaded=st.file_uploader("Upload XYZ file",type="xyz")
    if uploaded:
        txt=uploaded.read().decode()
        if txt!=st.session_state.prev_text:
            st.session_state.xyz_text=txt
            st.session_state.prev_text=txt
    
    # Sidebar model selector
    models=["Dihedral + HOMA + θRMSD","Dihedral-only","HOMA-only","Dihedral + HOMA","HOMA + XTB","XTB-only"]
    choice=st.sidebar.selectbox("Prediction model:",models,index=0)
    if "XTB" in choice:
        xtb_val=st.sidebar.number_input("XTB energy:",value=0.0)
    else:
        xtb_val=None
    
    # Auto or manual trigger
    trigger=(st.session_state.xyz_text and st.session_state.xyz_text!=st.session_state.get('computed_for'))
    if st.sidebar.button("Calculate"): trigger=True

    if trigger:
        st.session_state.computed_for=st.session_state.xyz_text
        mol=load_molecule_from_xyz(st.session_state.xyz_text)
        if not mol: return
        sum_dev,rmsd=analyze_geometry(mol)
        homa_avg,_=homa_aromatic_rings(mol)
        homa_avg=0.0 if np.isnan(homa_avg) else homa_avg
        db_energy,db_avail=find_database_energy(mol)

        if choice=="Dihedral + HOMA + θRMSD":
            E,low,high,eq=model_dhr(sum_dev,homa_avg,rmsd)
        # ...elif other models

        st.markdown(f"**Equation:** `{eq}`")
        st.markdown(f"**ΣDihedral:** {sum_dev:.3f}  **HOMA:** {homa_avg:.3f}  **θRMSD:** {rmsd:.3f}")
        st.markdown(f"## Predicted ΔE = {E:.3f} kJ/mol")
        if low is not None and high is not None:
            st.markdown(f"_Range: {low:.3f} – {high:.3f} kJ/mol_")

        if not db_avail:
            st.warning(f"Database CSV 'COMPAS_XTB_MS_WEBAPP_DATA.csv' not found.")
        elif db_energy is not None:
            st.success(f"Database ΔE: {db_energy:.3f} kJ/mol (PBE0-D4/6-31G(2df,p))")
        else:
            st.info("No database match found for this isomer.")

if __name__=='__main__': main()
