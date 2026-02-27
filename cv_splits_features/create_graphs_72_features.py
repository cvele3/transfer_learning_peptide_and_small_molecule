"""
Create graphs with 72-element vocabulary and molecular fingerprints.

This script processes both MolToxPredDataset.xlsx and ToxinSequenceSMILES.xlsx
to create StellarGraph objects with:
- 72-element one-hot encoding for atoms (instead of 27)
- Node features: one-hot element + atomic_number, degree, formal_charge, hybrid_value, aromatic
- Edge features: bond type one-hot + conjugation, in_ring, stereo
- Graph-level features: Morgan fingerprints (ECFP) - equivalent to DPC for SMILES
"""

import os
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from stellargraph import StellarGraph

# ========================== KONFIGURACIJA ==========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Output files
PROCESSED_DATA_FILE_PEPTIDE = "overtrained_peptide_graphs_72_features.pkl"
PROCESSED_DATA_FILE_SMALL_MOL = "overtrained_small_molecule_graphs_72_features.pkl"

# Input datasets
DATASET_PEPTIDE = os.path.join(SCRIPT_DIR, "..", "datasets", "ToxinSequenceSMILES.xlsx")
DATASET_SMALL_MOL = os.path.join(SCRIPT_DIR, "..", "datasets", "MolToxPredDataset.xlsx")

# 72-element vocabulary (from find_all_elements.py)
ELEMENT_TO_INDEX = {
    "*": 0, "Ac": 1, "Ag": 2, "Al": 3, "Am": 4, "Ar": 5, "As": 6, "Au": 7,
    "B": 8, "Ba": 9, "Be": 10, "Bi": 11, "Br": 12, "C": 13, "Ca": 14, "Cd": 15,
    "Ce": 16, "Cl": 17, "Co": 18, "Cr": 19, "Cs": 20, "Cu": 21, "F": 22, "Fe": 23,
    "Ga": 24, "Gd": 25, "Ge": 26, "H": 27, "Hg": 28, "I": 29, "In": 30, "K": 31,
    "La": 32, "Li": 33, "Mg": 34, "Mn": 35, "Mo": 36, "N": 37, "Na": 38, "Nb": 39,
    "Ni": 40, "Np": 41, "O": 42, "P": 43, "Pb": 44, "Pd": 45, "Po": 46, "Pt": 47,
    "Pu": 48, "Ra": 49, "Rb": 50, "Re": 51, "Rn": 52, "S": 53, "Sb": 54, "Se": 55,
    "Si": 56, "Sn": 57, "Sr": 58, "Ta": 59, "Tb": 60, "Te": 61, "Th": 62, "Ti": 63,
    "Tl": 64, "U": 65, "V": 66, "W": 67, "Y": 68, "Yb": 69, "Zn": 70, "Zr": 71,
}
NUM_FEATURES = len(ELEMENT_TO_INDEX)

# Morgan fingerprint parameters
MORGAN_RADIUS = 2
MORGAN_NBITS = 1024  # Standard size for ECFP fingerprints


# ========================== POMOĆNE FUNKCIJE ==========================

def get_morgan_fingerprint(smiles, radius=MORGAN_RADIUS, nBits=MORGAN_NBITS):
    """
    Generira Morgan fingerprint (ECFP) za SMILES string.
    
    Args:
        smiles: SMILES string
        radius: Radius za Morgan fingerprint (default 2)
        nBits: Broj bitova u fingerprintu (default 1024)
    
    Returns:
        numpy array s fingerprintom ili None ako SMILES nije validan
    """
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
        return np.array(fp)
    except Exception:
        return None


def process_molecule(smileString, smileLabel, element_to_index, num_features):
    """
    Procesira jednu molekulu i vraća StellarGraph objekt + fingerprint.
    
    Args:
        smileString: SMILES string
        smileLabel: Label (0 ili 1)
        element_to_index: Dictionary za one-hot encoding elemenata
        num_features: Broj elemenata u vokabularu
    
    Returns:
        Tuple (StellarGraph, label, fingerprint) ili None ako nije validan
    """
    mol = Chem.MolFromSmiles(smileString)
    if mol is None:
        return None
    
    # Skip molecules with isolated hydrogen atoms (for small molecules)
    has_isolated_H = any(atom.GetAtomicNum() == 1 and atom.GetDegree() == 0 for atom in mol.GetAtoms())
    if has_isolated_H:
        return None
    
    # ----- Process Edge (Bond) Features -----
    edges = []
    edge_features_list = []
    for bond in mol.GetBonds():
        source = bond.GetBeginAtomIdx()
        target = bond.GetEndAtomIdx()
        
        # One-hot encoding for bond type: [SINGLE, DOUBLE, TRIPLE, AROMATIC]
        bond_type = bond.GetBondType()
        bond_type_onehot = [0, 0, 0, 0]
        if bond_type == Chem.rdchem.BondType.SINGLE:
            bond_type_onehot[0] = 1
        elif bond_type == Chem.rdchem.BondType.DOUBLE:
            bond_type_onehot[1] = 1
        elif bond_type == Chem.rdchem.BondType.TRIPLE:
            bond_type_onehot[2] = 1
        elif bond_type == Chem.rdchem.BondType.AROMATIC:
            bond_type_onehot[3] = 1
        
        # Additional bond features
        conjugation = int(bond.GetIsConjugated())
        in_ring = int(bond.IsInRing())
        stereo = int(bond.GetStereo())
        
        # Combine the bond features into a single vector
        edge_feature_vector = bond_type_onehot + [conjugation, in_ring, stereo]
        
        # Add the edge in both directions to make the graph undirected
        edges.append((source, target))
        edge_features_list.append(edge_feature_vector)
        edges.append((target, source))
        edge_features_list.append(edge_feature_vector)
    
    # ----- Process Node (Atom) Features -----
    node_features = []
    for atom in mol.GetAtoms():
        # One-hot encoding for element (72 features)
        onehot = [0] * num_features
        elem = atom.GetSymbol()
        if elem in element_to_index:
            onehot[element_to_index[elem]] = 1
        
        # Additional features:
        atomic_number = atom.GetAtomicNum()
        degree = atom.GetDegree()
        formal_charge = atom.GetFormalCharge()
        # Map hybridization to a numerical value
        hybridization_mapping = {
            Chem.rdchem.HybridizationType.SP: 1,
            Chem.rdchem.HybridizationType.SP2: 2,
            Chem.rdchem.HybridizationType.SP3: 3,
        }
        hybrid_value = hybridization_mapping.get(atom.GetHybridization(), 0)
        aromatic = 1 if atom.GetIsAromatic() else 0
        
        # Concatenate one-hot vector with additional features
        features = onehot + [atomic_number, degree, formal_charge, hybrid_value, aromatic]
        node_features.append(features)
    
    # Convert node features to a NumPy array
    node_features = np.array(node_features)
    
    # Create DataFrames for nodes and edges
    nodes_df = pd.DataFrame(node_features)
    edges_df = pd.DataFrame(edges, columns=["source", "target"])
    edge_features_df = pd.DataFrame(edge_features_list, columns=[
        "bond_single", "bond_double", "bond_triple", "bond_aromatic",
        "conjugated", "in_ring", "stereo"
    ])
    # Combine the edge endpoints with their features
    edges_df = pd.concat([edges_df, edge_features_df], axis=1)
    
    # Create the StellarGraph object with both node and edge features
    G = StellarGraph(nodes=nodes_df, edges=edges_df)
    
    # Generate Morgan fingerprint (ECFP) - equivalent to DPC for SMILES
    fingerprint = get_morgan_fingerprint(smileString, radius=MORGAN_RADIUS, nBits=MORGAN_NBITS)
    
    return (G, smileLabel, fingerprint)


def process_dataset(filepath, smiles_col, label_col, dataset_name):
    """
    Procesira jedan dataset i vraća liste grafova, labela i fingerprintova.
    
    Args:
        filepath: Putanja do Excel datoteke
        smiles_col: Ime stupca s SMILES stringovima
        label_col: Ime stupca s labelama
        dataset_name: Ime dataseta (za printanje)
    
    Returns:
        Tuple (graphs, labels, fingerprints, graph_labels)
    """
    print(f"\n{'=' * 70}")
    print(f"  Procesiranje dataseta: {dataset_name}")
    print(f"{'=' * 70}")
    
    if not os.path.exists(filepath):
        print(f"  GRESKA: Datoteka '{filepath}' nije pronadjena!")
        return None, None, None, None
    
    print(f"  Učitavanje: {filepath}")
    data_file = pd.read_excel(filepath, header=0, usecols=[smiles_col, label_col])
    
    listOfTuples = []
    for index, row in data_file.iterrows():
        smiles = row[smiles_col]
        label = row[label_col]
        listOfTuples.append((smiles, label))
    
    print(f"  Učitano {len(listOfTuples)} zapisa")
    
    # Procesiranje molekula
    graphs = []
    labels = []
    fingerprints = []
    skipped_invalid = 0
    skipped_fingerprint = 0
    
    for smileString, smileLabel in listOfTuples:
        result = process_molecule(smileString, smileLabel, ELEMENT_TO_INDEX, NUM_FEATURES)
        if result is None:
            skipped_invalid += 1
            continue
        
        G, label, fingerprint = result
        if fingerprint is None:
            skipped_fingerprint += 1
            # Still add the graph even if fingerprint failed
            fingerprint = np.zeros(MORGAN_NBITS)
        
        graphs.append(G)
        labels.append(label)
        fingerprints.append(fingerprint)
    
    print(f"  Uspješno procesirano: {len(graphs)}")
    print(f"  Preskočeno (nevalidni SMILES): {skipped_invalid}")
    print(f"  Preskočeno (fingerprint greška): {skipped_fingerprint}")
    
    graph_labels = pd.Series(labels)
    print(f"\n  Raspodjela labela:")
    print(f"    Label 0: {(graph_labels == 0).sum()}")
    print(f"    Label 1: {(graph_labels == 1).sum()}")
    
    return graphs, labels, fingerprints, graph_labels


# ========================== GLAVNI TOK ==========================

def main():
    print("=" * 70)
    print("  KREIRANJE GRAFOVA S 72-ELEMENT VOKABULAROM I FINGERPRINTIMA")
    print("=" * 70)
    
    print(f"\n  Vokabular elemenata: {NUM_FEATURES} elemenata")
    print(f"  Morgan fingerprint: radius={MORGAN_RADIUS}, nBits={MORGAN_NBITS}")
    
    # Procesiranje peptide dataseta
    graphs_peptide, labels_peptide, fingerprints_peptide, graph_labels_peptide = process_dataset(
        DATASET_PEPTIDE,
        smiles_col="SMILES",
        label_col="TOXICITY",
        dataset_name="ToxinSequenceSMILES (Peptide)"
    )
    
    # Procesiranje small molecule dataseta
    graphs_small_mol, labels_small_mol, fingerprints_small_mol, graph_labels_small_mol = process_dataset(
        DATASET_SMALL_MOL,
        smiles_col="SMILES",
        label_col="Toxicity",
        dataset_name="MolToxPredDataset (Small Molecules)"
    )
    
    # Spremanje pojedinačnih datasetova
    if graphs_peptide is not None:
        processed_data_peptide = {
            "graphs": graphs_peptide,
            "labels": labels_peptide,
            "graph_labels": graph_labels_peptide,
            "element_to_index": ELEMENT_TO_INDEX,
            "fingerprints": fingerprints_peptide,  # Morgan fingerprints
            "fingerprint_params": {"radius": MORGAN_RADIUS, "nBits": MORGAN_NBITS}
        }
        output_path_peptide = os.path.join(SCRIPT_DIR, PROCESSED_DATA_FILE_PEPTIDE)
        with open(output_path_peptide, "wb") as f:
            pickle.dump(processed_data_peptide, f)
        print(f"\n  Peptide dataset spremljen u: {output_path_peptide}")
    
    if graphs_small_mol is not None:
        processed_data_small_mol = {
            "graphs": graphs_small_mol,
            "labels": labels_small_mol,
            "graph_labels": graph_labels_small_mol,
            "element_to_index": ELEMENT_TO_INDEX,
            "fingerprints": fingerprints_small_mol,  # Morgan fingerprints
            "fingerprint_params": {"radius": MORGAN_RADIUS, "nBits": MORGAN_NBITS}
        }
        output_path_small_mol = os.path.join(SCRIPT_DIR, PROCESSED_DATA_FILE_SMALL_MOL)
        with open(output_path_small_mol, "wb") as f:
            pickle.dump(processed_data_small_mol, f)
        print(f"  Small molecule dataset spremljen u: {output_path_small_mol}")
    
    print("\n[DONE]")


if __name__ == "__main__":
    main()
