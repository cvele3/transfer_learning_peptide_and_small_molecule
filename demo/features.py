import pandas as pd
import numpy as np
from rdkit import Chem
from stellargraph import StellarGraph

import os
import math
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Import necessary libraries from RDKit and StellarGraph
from rdkit import Chem
from rdkit.Chem import Draw
from stellargraph import StellarGraph
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import DeepGraphCNN
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import LambdaCallback, EarlyStopping
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Read in the dataset
filepath_raw = '../datasets/ToxinSequenceSMILES.xlsx'
data_file = pd.read_excel(filepath_raw, header=0, usecols=["SEQUENCE", "TOXICITY", "SMILES"])

# Create list of (SMILES, label) tuples
listOfTuples = []
data_file.reset_index(drop=True, inplace=True)
for index, row in data_file.iterrows():
    smiles = row['SMILES']
    label = row["TOXICITY"]
    listOfTuples.append((smiles, label))

# Define fixed vocabulary for atomic one-hot encoding (27 elements)
element_to_index = {
    "N": 0, "C": 1, "O": 2, "F": 3, "Cl": 4, "S": 5, "Na": 6, "Br": 7,
    "Se": 8, "I": 9, "Pt": 10, "P": 11, "Mg": 12, "K": 13, "Au": 14,
    "Ir": 15, "Cu": 16, "B": 17, "Zn": 18, "Re": 19, "Ca": 20, "As": 21,
    "Hg": 22, "Ru": 23, "Pd": 24, "Cs": 25, "Si": 26,
}
NUM_FEATURES = len(element_to_index)
print("\nFixed vocabulary (27 elements) =", element_to_index)

# List to hold the StellarGraph objects and corresponding labels
stellarGraphAllList = []

for molecule in listOfTuples:
    smileString = molecule[0]
    smileLabel = molecule[1]
    mol = Chem.MolFromSmiles(smileString)
    if mol is None:
        continue  # Skip invalid SMILES

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
        # One-hot encoding for element
        onehot = [0] * NUM_FEATURES
        elem = atom.GetSymbol()
        if elem in element_to_index:
            onehot[element_to_index[elem]] = 1

        # Additional features:
        atomic_number = atom.GetAtomicNum()            # e.g. 6 for Carbon
        degree = atom.GetDegree()                       # number of bonds
        formal_charge = atom.GetFormalCharge()          # e.g. 0, +1, -1
        # Map hybridization to a numerical value; adjust as needed:
        hybridization_mapping = {
            Chem.rdchem.HybridizationType.SP: 1,
            Chem.rdchem.HybridizationType.SP2: 2,
            Chem.rdchem.HybridizationType.SP3: 3,
        }
        hybrid_value = hybridization_mapping.get(atom.GetHybridization(), 0)
        aromatic = 1 if atom.GetIsAromatic() else 0     # aromaticity flag

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
    stellarGraphAllList.append((G, smileLabel))


# Pick the first graph from your list:
G, label = stellarGraphAllList[0]
print("Graph label:", label)

# Print summary info about the graph:
print(G.info())

# --- Inspect Node Features ---
# Get the list of node IDs
node_ids = list(G.nodes())
print("Node IDs:", node_ids[:5])

# Get node features as a NumPy array (these are in the same order as node_ids)
features_array = G.node_features()
print("Shape of node features array:", features_array.shape)
print("Features for the first node (ID={}): {}".format(node_ids[0], features_array[0]))

# Alternatively, build a DataFrame manually:
nodes_df = pd.DataFrame(features_array, index=node_ids)
print("First 5 nodes and their features:")
print(nodes_df.head())

# --- Inspect Edge Features ---

# Get the edges as a list with their data, if available.
# StellarGraph stores edge features in its edges DataFrame, but if there's no helper method,
# you can access the underlying DataFrame by retrieving it with:
print("First 5 edges and their features:")
print(edges_df.head())