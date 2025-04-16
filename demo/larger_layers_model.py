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

graphs = [item[0] for item in stellarGraphAllList]
labels = [item[1] for item in stellarGraphAllList]
graph_labels = pd.Series(labels)

# Initialize the graph generator using all graphs
generator = PaddedGraphGenerator(graphs=graphs)



epochs = 10000
k = 25
layer_sizes = [125, 125, 125, 1]
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

model.save('larger_layers_peptide_model.h5')