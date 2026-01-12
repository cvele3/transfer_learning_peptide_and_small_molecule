# train_model.py

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

######################################
# 1. Load or Generate Preprocessed Data
######################################
# PROCESSED_DATA_FILE = "overtrained_peptide_graphs.pkl"
PROCESSED_DATA_FILE = "large_layers_overtrained_peptide.pkl" 

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
    filepath_raw = '../datasets/ToxinSequenceSMILES.xlsx'
    data_file = pd.read_excel(filepath_raw, header=0, usecols=["SEQUENCE", "TOXICITY", "SMILES"])
    
    listOfTuples = []
    data_file.reset_index(drop=True, inplace=True)
    for index, row in data_file.iterrows():
        smiles = row['SMILES']
        label = row["TOXICITY"]
        molecule = (smiles, label)
        listOfTuples.append(molecule)
    
    # Definiramo fiksni vokabular s 27 elemenata
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
    
    # Nema potrebe računati all_elements, lst_degree, lst_hybridization itd.
    
    # Konvertiramo svaki SMILES u StellarGraph objekt koristeći fiksni vokabular
    stellarGraphAllList = []
    ZeroActivity = 0
    OneActivity = 0
    
    for molecule in listOfTuples:
        smileString = molecule[0]
        smileLabel = molecule[1]
        mol = Chem.MolFromSmiles(smileString)
        if mol is None:
            continue  # Preskoči nevalidne SMILES
        # Kreiraj listu bridova (oba smjera)
        edges = []
        for bond in mol.GetBonds():
            edges.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            edges.append((bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()))
        
        # Kreiraj node features (one-hot kodiranje koristeći element_to_index)
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
        
        edges_df = pd.DataFrame(edges, columns=["source", "target"])
        G = StellarGraph(nodes=node_features, edges=edges_df)
        
        if smileLabel == 1:
            OneActivity += 1
            stellarGraphAllList.append((G, smileLabel))
        elif smileLabel == 0:
            ZeroActivity += 1
            stellarGraphAllList.append((G, smileLabel))
    
    print("Broj primjera za label 0:", ZeroActivity)
    print("Broj primjera za label 1:", OneActivity)
    print("Ukupno primjera:", len(stellarGraphAllList))
    
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



epochs = 10000
k = 25
layer_sizes = [25, 25, 25, 1]
filter1 = 16
filter2 = 32
filter3 = 128

# Create the DeepGraphCNN model using the same generator
dgcnn_model = DeepGraphCNN(
    layer_sizes=layer_sizes,
    activations=["tanh", "tanh", "tanh", "tanh"],
    k=k,
    bias=False,
    generator=generator,
)
x_inp, x_out = dgcnn_model.in_out_tensors()

x_out = Conv1D(filters=filter1, kernel_size=sum(layer_sizes), strides=sum(layer_sizes))(x_out)
x_out = MaxPool1D(pool_size=2)(x_out)
x_out = Conv1D(filters=filter2, kernel_size=5, strides=1)(x_out)
x_out = Flatten()(x_out)
x_out = Dense(units=filter3, activation="relu")(x_out)
x_out = Dropout(rate=0.2)(x_out)
predictions = Dense(units=1, activation="sigmoid")(x_out)

model = Model(inputs=x_inp, outputs=predictions)
model.compile(optimizer=Adam(lr=0.0001), loss=binary_crossentropy, metrics=["acc"])

# Early stopping callback
callback = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

indices = np.arange(len(labels))
train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)

# Define a batch size
batch_size = 32

# Build generators using indices
train_gen = generator.flow(train_idx, targets=np.array(labels)[train_idx], batch_size=batch_size)
val_gen = generator.flow(val_idx, targets=np.array(labels)[val_idx], batch_size=batch_size)

# Train the model using these generators
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    verbose=1,
    shuffle=True,
    callbacks=[callback]
)

model.save('extra_features_overtrained_peptide_model.h5')