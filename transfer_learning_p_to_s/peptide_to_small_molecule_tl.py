import math
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import rdkit.Chem
import networkx as nx
import matplotlib.pyplot as plt
import stellargraph as sg

from rdkit import Chem
from rdkit.Chem import Draw

from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import matthews_corrcoef, precision_recall_curve

from stellargraph import StellarGraph, datasets
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import PaddedGraphGenerator, GraphSAGELinkGenerator
from stellargraph.layer import DeepGraphCNN, GraphSAGE, link_classification

from tensorflow.keras import Model, optimizers, losses, metrics
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.callbacks import LambdaCallback, EarlyStopping
from tensorflow.keras.utils import Sequence
import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPool1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from sklearn.model_selection import train_test_split

# Ako je potrebno za custom slojeve iz StellarGrapha:
from stellargraph.layer import DeepGraphCNN
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph import StellarGraph

from rdkit import Chem

# Uvoz ovih slojeva ako ih treba prilikom load_model
from stellargraph.layer import (
    DeepGraphCNN,
    GCNSupervisedGraphClassification,
    SortPooling,
    GraphConvolution,
)
from stellargraph.layer.graph_classification import SortPooling

from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Uvoz funkcija za generiranje/učitavanje CV splitova (cv_splits.pkl)
from cv_splits.data_splits import generate_cv_splits, load_cv_splits

##############################################################
# 1. Učitavanje podataka i generiranje StellarGraph objekata
##############################################################
PROCESSED_DATA_FILE = "../overtrained_models/overtrained_small_molecule_graphs.pkl"

if os.path.exists(PROCESSED_DATA_FILE):
    print("Učitavam prethodno spremljene podatke iz:", PROCESSED_DATA_FILE)
    with open(PROCESSED_DATA_FILE, "rb") as f:
        processed_data = pickle.load(f)
    graphs = processed_data["graphs"]
    labels = processed_data["labels"]
    graph_labels = processed_data["graph_labels"]
    element_to_index = processed_data["element_to_index"]
else:
    print("At this point there is no way you don't have that file")


generator = PaddedGraphGenerator(graphs=graphs)

##############################################################
# 2. Učitavanje CV splitova (cv_splits.pkl)
##############################################################
SPLITS_FILE = "../cv_splits/cv_splits_small.pkl"
cv_splits = load_cv_splits(SPLITS_FILE)

##############################################################
# 3. Transfer Learning – Učitavanje baseline modela i dotreniranje
##############################################################
PRETRAINED_MODEL_PATH = "../overtrained_models/overtrained_peptide_model.h5"

# Funkcija za učitavanje baseline modela
def load_pretrained_model():
    model_loaded = load_model(
        PRETRAINED_MODEL_PATH,
        custom_objects={
            "DeepGraphCNN": DeepGraphCNN,
            "GCNSupervisedGraphClassification": GCNSupervisedGraphClassification,
            "SortPooling": SortPooling,
            "GraphConvolution": GraphConvolution,
        }
    )
    return model_loaded

# Funkcije za evaluaciju
def roc_auc_metric(y_true, y_pred):
    roc_auc = roc_auc_score(y_true, y_pred)
    print("ROC AUC:", roc_auc)
    return roc_auc

def rest_of_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, (y_pred >= 0.5).astype(int)).ravel()
    tpr = tp / (tp + fn + 1e-8)
    tnr = tn / (tn + fp + 1e-8)
    gm = math.sqrt(tpr * tnr)
    precision = precision_score(y_true, (y_pred >= 0.5).astype(int))
    recall = recall_score(y_true, (y_pred >= 0.5).astype(int))
    f1 = f1_score(y_true, (y_pred >= 0.5).astype(int))
    print("GM:", gm)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    return gm, precision, recall, f1

def mcc_metric(y_true, y_pred):
    mcc = matthews_corrcoef(y_true, (y_pred >= 0.5).astype(int))
    print("MCC:", mcc)
    return mcc


epochs = 10000
# Set EarlyStopping callback
callback = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)


##############################################################
# 4. Transfer Learning
##############################################################
# Za svaki CV split koristimo iste podjele (učitane iz cv_splits.pkl)
# Metoda 1: Zamrzavanje GNN slojeva
print("\n=== METODA 1: Zamrzavanje GNN slojeva")
fold_index = 0
for split in cv_splits:
    fold_index += 1
    print(f"\n--- Fold {fold_index} ---")
    model1 = load_pretrained_model()
    # Zamrzavanje slojeva koji sadrže "deep_graph_cnn" ili "graph_conv"
    for layer in model1.layers:
        if "deep_graph_cnn" in layer.name or "graph_conv" in layer.name:
            layer.trainable = False
        else:
            layer.trainable = True
    model1.compile(optimizer=Adam(learning_rate=1e-4), loss=binary_crossentropy, metrics=["accuracy"])
    
    graphs_arr = np.array(graphs)
    labels_arr = np.array(graph_labels)
    X_train = graphs_arr[split["train_idx"]]
    X_val   = graphs_arr[split["val_idx"]]
    X_test  = graphs_arr[split["test_idx"]]
    y_train = labels_arr[split["train_idx"]]
    y_val   = labels_arr[split["val_idx"]]
    y_test  = labels_arr[split["test_idx"]]
    
    # Dodatno razdvajanje trening skupa na trening i validaciju (ako je potrebno)
    X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=42)
    y_train, y_val = train_test_split(y_train, test_size=0.2, random_state=42)
    
    gen_fold = PaddedGraphGenerator(graphs=graphs_arr)
    train_gen = gen_fold.flow(X_train, y_train, batch_size=32, shuffle=True)
    val_gen   = gen_fold.flow(X_val,   y_val,   batch_size=32, shuffle=False)
    test_gen  = gen_fold.flow(X_test,  y_test,  batch_size=32, shuffle=False)
    
    history1 = model1.fit(train_gen, validation_data=val_gen, epochs=epochs, verbose=1, shuffle=True, callbacks=[callback])
    
    y_pred = model1.predict(test_gen)
    y_pred = np.reshape(y_pred, (-1,))
    roc_auc_metric(y_test, y_pred)
    rest_of_metrics(y_test, y_pred)
    mcc_metric(y_test, y_pred)
    
    # Save the model for this fold with a unique filename
    model_save_path = f"freeze_gnn_peptide_to_smile_fold_{fold_index}.h5"
    model1.save(model_save_path)
    print(f"Model saved as {model_save_path}")


print("\n=== METODA 2: Zamrzavanje READOUT/dense slojeva + 10-fold CV ===")
fold_index = 0
for split in cv_splits:
    fold_index += 1
    print(f"\n--- Fold {fold_index} ---")
    model2 = load_pretrained_model()
    # Freeze layers that contain "dense", "dropout", "flatten" or "readout" in their name
    for layer in model2.layers:
        if any(x in layer.name for x in ["dense", "dropout", "flatten", "readout"]):
            layer.trainable = False
        else:
            layer.trainable = True
    model2.compile(optimizer=Adam(learning_rate=1e-5), loss=binary_crossentropy, metrics=["accuracy"])
    
    graphs_arr = np.array(graphs)
    labels_arr = np.array(graph_labels)
    X_train = graphs_arr[split["train_idx"]]
    X_val   = graphs_arr[split["val_idx"]]
    X_test  = graphs_arr[split["test_idx"]]
    y_train = labels_arr[split["train_idx"]]
    y_val   = labels_arr[split["val_idx"]]
    y_test  = labels_arr[split["test_idx"]]
    
    # Optionally perform an extra train-validation split if needed
    X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=42)
    y_train, y_val = train_test_split(y_train, test_size=0.2, random_state=42)
    
    gen_fold = PaddedGraphGenerator(graphs=graphs_arr)
    train_gen = gen_fold.flow(X_train, y_train, batch_size=32, shuffle=True)
    val_gen   = gen_fold.flow(X_val,   y_val,   batch_size=32, shuffle=False)
    test_gen  = gen_fold.flow(X_test,  y_test,  batch_size=32, shuffle=False)
    
    history2 = model2.fit(train_gen, validation_data=val_gen, epochs=epochs,
                          verbose=1, shuffle=True, callbacks=[callback])
    
    y_pred = model2.predict(test_gen)
    y_pred = np.reshape(y_pred, (-1,))
    roc_auc_metric(y_test, y_pred)
    rest_of_metrics(y_test, y_pred)
    mcc_metric(y_test, y_pred)
    
    # Save the model for this fold with a unique filename
    model_save_path = f"freeze_readout_peptide_to_smile_fold_{fold_index}.h5"
    model2.save(model_save_path)
    print(f"Model saved as {model_save_path}")

print("\n=== METODA 3: Zamrzavanje svih slojeva + novi izlazni sloj + 10-fold CV ===")
fold_index = 0
for split in cv_splits:
    fold_index += 1
    print(f"\n--- Fold {fold_index} ---")
    base_model = load_pretrained_model()
    # Freeze all layers in the baseline model
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add a new output layer using the second-to-last layer's output as input
    intermediate_output = base_model.layers[-2].output
    new_output = Dense(1, activation="sigmoid", name="new_output")(intermediate_output)
    model3 = Model(inputs=base_model.input, outputs=new_output)
    model3.compile(optimizer=Adam(learning_rate=1e-4), loss=binary_crossentropy, metrics=["accuracy"])
    
    graphs_arr = np.array(graphs)
    labels_arr = np.array(graph_labels)
    X_train = graphs_arr[split["train_idx"]]
    X_val   = graphs_arr[split["val_idx"]]
    X_test  = graphs_arr[split["test_idx"]]
    y_train = labels_arr[split["train_idx"]]
    y_val   = labels_arr[split["val_idx"]]
    y_test  = labels_arr[split["test_idx"]]
    
    # Optionally perform an extra train-validation split if needed
    X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=42)
    y_train, y_val = train_test_split(y_train, test_size=0.2, random_state=42)
    
    gen_fold = PaddedGraphGenerator(graphs=graphs_arr)
    train_gen = gen_fold.flow(X_train, y_train, batch_size=32, shuffle=True)
    val_gen   = gen_fold.flow(X_val,   y_val,   batch_size=32, shuffle=False)
    test_gen  = gen_fold.flow(X_test,  y_test,  batch_size=32, shuffle=False)
    
    history3 = model3.fit(train_gen, validation_data=val_gen, epochs=epochs, verbose=1, shuffle=True, callbacks=[callback])
    
    y_pred = model3.predict(test_gen)
    y_pred = np.reshape(y_pred, (-1,))
    roc_auc_metric(y_test, y_pred)
    rest_of_metrics(y_test, y_pred)
    mcc_metric(y_test, y_pred)
    
    # Save the model for this fold with a unique filename
    model_save_path = f"freeze_all_peptide_to_smile_fold_{fold_index}.h5"
    model3.save(model_save_path)
    print(f"Model saved as {model_save_path}")

    

print("\n=== METODA 4: Gradual unfreezing + discriminative fine-tuning + 10-fold CV ===")
fold_index = 0
for split in cv_splits:
    fold_index += 1
    print(f"\n--- Fold {fold_index} ---")
    
    model4 = load_pretrained_model()
    
    # Define groups of layers:
    gnn_layers = [layer for layer in model4.layers if "deep_graph_cnn" in layer.name or "graph_conv" in layer.name]
    readout_layers = [layer for layer in model4.layers if any(x in layer.name for x in ["dense", "dropout", "flatten", "readout"])]
    # Final layers: all layers which are not in gnn or readout groups
    final_layers = [layer for layer in model4.layers if layer.name not in [l.name for l in (gnn_layers + readout_layers)]]
    
    # Initially freeze all layers
    for layer in model4.layers:
        layer.trainable = False
        
    # Compile with highest learning rate (for final layers)
    model4.compile(optimizer=Adam(learning_rate=1e-3), loss=binary_crossentropy, metrics=["accuracy"])
    
    graphs_arr = np.array(graphs)
    labels_arr = np.array(graph_labels)
    X_train = graphs_arr[split["train_idx"]]
    X_val   = graphs_arr[split["val_idx"]]
    X_test  = graphs_arr[split["test_idx"]]
    y_train = labels_arr[split["train_idx"]]
    y_val   = labels_arr[split["val_idx"]]
    y_test  = labels_arr[split["test_idx"]]
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    gen_fold = PaddedGraphGenerator(graphs=graphs_arr)
    train_gen = gen_fold.flow(X_train, y_train, batch_size=32, shuffle=True)
    val_gen   = gen_fold.flow(X_val,   y_val,   batch_size=32, shuffle=False)
    test_gen  = gen_fold.flow(X_test,  y_test,  batch_size=32, shuffle=False)
    
    epochs_per_phase = 10  # Adjust number of epochs per phase as needed
    
    # --- Phase 1: Train only final layers ---
    for layer in final_layers:
        layer.trainable = True
    model4.compile(optimizer=Adam(learning_rate=1e-3), loss=binary_crossentropy, metrics=["accuracy"])
    model4.fit(train_gen, validation_data=val_gen, epochs=epochs_per_phase,
               verbose=1, callbacks=[callback])
    
    # --- Phase 2: Unfreeze readout layers (with lower learning rate) ---
    for layer in readout_layers:
        layer.trainable = True
    model4.compile(optimizer=Adam(learning_rate=1e-4), loss=binary_crossentropy, metrics=["accuracy"])
    model4.fit(train_gen, validation_data=val_gen, epochs=epochs_per_phase,
               verbose=1, callbacks=[callback])
    
    # --- Phase 3: Unfreeze GNN layers (lowest learning rate) ---
    for layer in gnn_layers:
        layer.trainable = True
    model4.compile(optimizer=Adam(learning_rate=1e-5), loss=binary_crossentropy, metrics=["accuracy"])
    model4.fit(train_gen, validation_data=val_gen, epochs=epochs_per_phase,
               verbose=1, callbacks=[callback])
    
    # Evaluate after gradual unfreezing
    y_pred = model4.predict(test_gen).reshape(-1,)
    roc_auc_metric(y_test, y_pred)
    rest_of_metrics(y_test, y_pred)
    mcc_metric(y_test, y_pred)
    
    # Save the model for this fold with a unique filename
    model_save_path = f"gradual_unfreezing_peptide_to_smile_fold_{fold_index}.h5"
    model4.save(model_save_path)
    print(f"Model saved as {model_save_path}")