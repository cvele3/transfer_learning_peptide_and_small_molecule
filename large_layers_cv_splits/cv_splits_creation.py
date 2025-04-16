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
# PROCESSED_DATA_FILE = "large_layers_overtrained_peptide.pkl"
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

######################################
# 2. Generate CV Splits
######################################
# SPLITS_FILE = "large_layers_cv_splits_peptide.pkl"
SPLITS_FILE = "large_layers_cv_splits_small_mol_tox_pred.pkl"
if not os.path.exists(SPLITS_FILE):
    cv_splits = generate_cv_splits(graph_labels.values, n_splits=10, val_split=0.2, random_state=42, save_path=SPLITS_FILE)