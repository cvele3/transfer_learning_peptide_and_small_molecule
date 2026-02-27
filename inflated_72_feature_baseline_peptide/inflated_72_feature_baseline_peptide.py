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
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Uvoz funkcija za generiranje/učitavanje CV splitova (cv_splits.pkl)
from cv_splits.data_splits import generate_cv_splits, load_cv_splits

######################################
# 1. Load or Generate Preprocessed Data
######################################
PROCESSED_DATA_FILE = "../cv_splits_features/overtrained_peptide_graphs_72_features.pkl"

if os.path.exists(PROCESSED_DATA_FILE):
    print("Učitavam prethodno spremljene podatke iz:", PROCESSED_DATA_FILE)
    with open(PROCESSED_DATA_FILE, "rb") as f:
        processed_data = pickle.load(f)
    graphs = processed_data["graphs"]
    labels = processed_data["labels"]
    graph_labels = processed_data["graph_labels"]
    element_to_index = processed_data["element_to_index"]
    print(f"Vokabular elemenata: {len(element_to_index)} elemenata")
else:
    print("Ne postoji spremljena datoteka. Do sad bi trebala vec postojat")
    print("Pokrenite prvo: cv_splits_features/create_graphs_72_features.py")

# Initialize the graph generator using all graphs
generator = PaddedGraphGenerator(graphs=graphs)

######################################
# 2. Generate CV Splits
######################################
SPLITS_FILE = "../cv_splits_features/cv_splits_peptide_72_features.pkl"
cv_splits = load_cv_splits(SPLITS_FILE)

######################################
# 3. Model Parameters and Build Function
######################################
# Model parameters
epochs = 10000
k = 25
layer_sizes = [512, 256, 128, 1]
filter1 = 16
filter2 = 32
filter3 = 128

def build_model(generator):
    # Create the DeepGraphCNN model using the provided generator
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
    return model

# Early stopping callback
callback = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

######################################
# 4. Define Evaluation Functions
######################################
def roc_auc_metric(y_true, y_pred):
    roc_auc = roc_auc_score(y_true, y_pred)
    print("ROC AUC:", roc_auc)
    return roc_auc

def rest_of_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn + 1e-8)
    tnr = tn / (tn + fp + 1e-8)
    gm = math.sqrt(tpr * tnr)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print("GM:", gm)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    return gm, precision, recall, f1

def mcc_metric(y_true, y_pred):
    mcc = matthews_corrcoef(y_true, y_pred)
    print("MCC:", mcc)
    return mcc

######################################
# 5. Cross Validation Training
######################################
histories = []
mcc_values = []
gm_values = []
precision_values = []
recall_values = []
f1_values = []
roc_auc_values = []

# Convert the list of graphs to a numpy array for easier indexing
graphs_arr = np.array(graphs)

all_y_test = []
all_y_pred_probs = []

for fold, split in enumerate(cv_splits):
    print(f"\n--- Fold {fold+1} ---")
    train_idx = split["train_idx"]
    val_idx   = split["val_idx"]
    test_idx  = split["test_idx"]
    
    X_train = graphs_arr[train_idx]
    X_val   = graphs_arr[val_idx]
    X_test  = graphs_arr[test_idx]
    y_train = graph_labels.iloc[train_idx]
    y_val   = graph_labels.iloc[val_idx]
    y_test  = graph_labels.iloc[test_idx]
    
    # Create new generators for this fold
    train_gen = generator.flow(
        X_train,
        targets=y_train,
        batch_size=32,
        symmetric_normalization=False,
    )
    val_gen = generator.flow(
        X_val,
        targets=y_val,
        batch_size=50,
        symmetric_normalization=False,
    )
    test_gen = generator.flow(
        X_test,
        targets=y_test,
        batch_size=50,
        symmetric_normalization=False,
    )
    
    # Reinitialize the model for this fold
    model = build_model(generator)
    
    # Train the model
    history = model.fit(
        train_gen,
        epochs=epochs,
        verbose=1,
        validation_data=val_gen,
        shuffle=True,
        callbacks=[callback]
    )
    histories.append(history)
    
    # Evaluate on the test set
    y_pred_probs = model.predict(test_gen)
    y_pred_probs = np.reshape(y_pred_probs, (-1,))
    roc_auc = roc_auc_metric(y_test, y_pred_probs)
    roc_auc_values.append(roc_auc)
    
    # Save results for later analysis
    all_y_test.extend(y_test.to_numpy())
    all_y_pred_probs.extend(y_pred_probs)

    # Use threshold 0.5 for binary predictions
    y_pred = [0 if prob < 0.5 else 1 for prob in y_pred_probs]
    gm, precision, recall, f1 = rest_of_metrics(y_test.to_numpy(), np.array(y_pred))
    mcc = mcc_metric(y_test.to_numpy(), np.array(y_pred))
    
    # Save the model for the current fold
    model.save(f"inflated_72_feature_baseline_peptide_fold_{fold+1}.h5")
