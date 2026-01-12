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

import io
import sys
from contextlib import redirect_stderr

from data_splits import generate_cv_splits, load_cv_splits

######################################
# 1. Load or Generate Preprocessed Data
######################################
# PROCESSED_DATA_FILE = "overtrained_small_molecule_graphs.pkl"
# PROCESSED_DATA_FILE = "overtrained_peptide_graphs.pkl"
PROCESSED_DATA_FILE = "overtrained_small_molecule_mol_tox_pred_graphs_obr.pkl"

if os.path.exists(PROCESSED_DATA_FILE):
    print("Učitavam prethodno spremljene podatke iz:", PROCESSED_DATA_FILE)
    with open(PROCESSED_DATA_FILE, "rb") as f:
        processed_data = pickle.load(f)
    graphs = processed_data["graphs"]
    labels = processed_data["labels"]
    graph_labels = processed_data["graph_labels"]
    element_to_index = processed_data["element_to_index"]
else:
    print("Ne postoji spremljena datoteka. Pokrećem preprocesiranje podataka...")
    # Read data from Excel
    # filepath_raw = '../datasets/out.xlsx'
    # data_file = pd.read_excel(filepath_raw, header=0, usecols=["SMILES", "HEK"])

    # # Initialize an empty list to store tuples
    # listOfTuples = []

    # # Iterate through each row to extract the SMILES and HEK columns
    # for index, row in data_file.iterrows():
    #     molecule = (row["SMILES"], row["HEK"])
    #     listOfTuples.append(molecule)

    filepath_raw = '../datasets/MolToxPredDataset.xlsx'
    data_file = pd.read_excel(filepath_raw, header=0, usecols=["SMILES", "Toxicity"])

    # Initialize an empty list to store tuples
    listOfTuples = []

    # Iterate through each row to extract the SMILES and HEK columns
    for index, row in data_file.iterrows():
        molecule = (row["SMILES"], row["Toxicity"])
        listOfTuples.append(molecule)

    # filepath_raw = '../datasets/ToxinSequenceSMILES.xlsx'
    # data_file = pd.read_excel(filepath_raw, header=0, usecols=["SEQUENCE", "TOXICITY", "SMILES"])
    
    # listOfTuples = []
    # data_file.reset_index(drop=True, inplace=True)
    # for index, row in data_file.iterrows():
    #     smiles = row['SMILES']
    #     label = row["TOXICITY"]
    #     molecule = (smiles, label)
    #     listOfTuples.append(molecule)


    # Definiraj fiksni vokabular s 27 elemenata
    element_to_index = {
        "*": 0, "Ac": 1, "Ag": 2, "Al": 3, "Am": 4, "Ar": 5, "As": 6, "Au": 7,
        "B": 8, "Ba": 9, "Be": 10, "Bi": 11, "Br": 12, "C": 13, "Ca": 14,
        "Cd": 15, "Ce": 16, "Cl": 17, "Co": 18, "Cr": 19, "Cs": 20, "Cu": 21,
        "F": 22, "Fe": 23, "Ga": 24, "Gd": 25, "Ge": 26, "H": 27, "Hg": 28,
        "I": 29, "In": 30, "K": 31, "La": 32, "Li": 33, "Mg": 34, "Mn": 35,
        "Mo": 36, "N": 37, "Na": 38, "Nb": 39, "Ni": 40, "Np": 41, "O": 42,
        "P": 43, "Pb": 44, "Pd": 45, "Po": 46, "Pt": 47, "Pu": 48, "Ra": 49,
        "Rb": 50, "Re": 51, "Rn": 52, "S": 53, "Sb": 54, "Se": 55, "Si": 56,
        "Sn": 57, "Sr": 58, "Ta": 59, "Tb": 60, "Te": 61, "Th": 62, "Ti": 63,
        "Tl": 64, "U": 65, "V": 66, "W": 67, "Y": 68, "Yb": 69, "Zn": 70,
        "Zr": 71,
    }
    NUM_FEATURES = len(element_to_index)
    print("\nFiksni vokabular (27 elemenata) =", element_to_index)

    # (Ostale varijable za normalizaciju se mogu izostaviti ako nisu potrebne.)
    
    # Convert each SMILES into a StellarGraph object
    stellarGraphAllList = []
    ZeroActivity = 0
    OneActivity = 0
    
    for molecule in listOfTuples:
        smileString = molecule[0]
        smileLabel = molecule[1]
        mol = Chem.MolFromSmiles(smileString)

        if mol is None:
            continue  # Skip invalid SMILES
  
        has_isolated_H = any(atom.GetAtomicNum() == 1 and atom.GetDegree() == 0 for atom in mol.GetAtoms())
        if has_isolated_H:
            print("Skipped")
            continue 

        # Create edge list (both directions)
        edges = []
        for bond in mol.GetBonds():
            edges.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            edges.append((bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()))
        
        # Create node features (one-hot encoding using the fixed vocabulary)
        node_features = []
        for atom in mol.GetAtoms():
            elem = atom.GetSymbol()
            if elem not in element_to_index:
                onehot = [0] * NUM_FEATURES
            else:
                onehot = [0] * NUM_FEATURES
                onehot[element_to_index[elem]] = 1
            node_features.append(onehot)
        node_features = np.array(node_features)
        
        # Save edges into a DataFrame (StellarGraph requires a DataFrame for edges)
        edges_df = pd.DataFrame(edges, columns=["source", "target"])
        
        # Create the StellarGraph object
        G = StellarGraph(nodes=node_features, edges=edges_df)
        
        # Optionally, add all examples
        if smileLabel == 1:
            OneActivity += 1
            stellarGraphAllList.append((G, smileLabel))
        elif smileLabel == 0:
            ZeroActivity += 1
            stellarGraphAllList.append((G, smileLabel))
    
    print("Broj primjera za label 0:", ZeroActivity)
    print("Broj primjera za label 1:", OneActivity)
    print("Ukupno primjera:", len(stellarGraphAllList))
    
    # Extract lists of graphs and labels
    graphs = [item[0] for item in stellarGraphAllList]
    labels = [item[1] for item in stellarGraphAllList]
    graph_labels = pd.Series(labels)
    print("Raspodjela labela:")
    print(graph_labels.value_counts().to_frame())
    
    # Save the processed data to a file for future use
    processed_data = {
        "graphs": graphs,
        "labels": labels,
        "graph_labels": graph_labels,
        "element_to_index": element_to_index
    }
    with open(PROCESSED_DATA_FILE, "wb") as f:
        pickle.dump(processed_data, f)
    print("Podaci su spremljeni u", PROCESSED_DATA_FILE)

# Initialize the graph generator using all graphs
generator = PaddedGraphGenerator(graphs=graphs)

######################################
# 2. Generate CV Splits
######################################
# SPLITS_FILE = "cv_splits_small.pkl"
# SPLITS_FILE = "cv_splits_peptide.pkl"
SPLITS_FILE = "cv_splits_small_mol_tox_pred_obr.pkl"
if not os.path.exists(SPLITS_FILE):
    cv_splits = generate_cv_splits(graph_labels.values, n_splits=10, val_split=0.2, random_state=42, save_path=SPLITS_FILE)