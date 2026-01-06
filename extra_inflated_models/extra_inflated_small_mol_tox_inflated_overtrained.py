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
PROCESSED_DATA_FILE = "large_layers_overtrained_small_mol_tox_pred.pkl"

if os.path.exists(PROCESSED_DATA_FILE):
    print("Učitavam prethodno spremljene podatke iz:", PROCESSED_DATA_FILE)
    with open(PROCESSED_DATA_FILE, "rb") as f:
        processed_data = pickle.load(f)
    graphs = processed_data["graphs"]
    labels = processed_data["labels"]
    graph_labels = processed_data["graph_labels"]
    element_to_index = processed_data["element_to_index"]

# Initialize the graph generator using all graphs
generator = PaddedGraphGenerator(graphs=graphs)



epochs = 10000
k = 25
layer_sizes = [1024, 512, 256, 1]
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
callback = EarlyStopping(monitor='val_loss', patience=13, restore_best_weights=True)

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

model.save('extra_inflated_small_mol_tox_model.h5')