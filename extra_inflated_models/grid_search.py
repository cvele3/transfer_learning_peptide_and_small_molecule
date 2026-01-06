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
from tensorflow.keras.callbacks import EarlyStopping # Removed LambdaCallback as it's not used
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
# import numpy as np # already imported
# import matplotlib.pyplot as plt # already imported

######################################
# 1. Load or Generate Preprocessed Data
######################################
PROCESSED_DATA_FILE = "large_layers_overtrained_peptide.pkl"

if os.path.exists(PROCESSED_DATA_FILE):
    print("Učitavam prethodno spremljene podatke iz:", PROCESSED_DATA_FILE)
    with open(PROCESSED_DATA_FILE, "rb") as f:
        processed_data = pickle.load(f)
    graphs = processed_data["graphs"]
    labels = processed_data["labels"]
    # graph_labels = processed_data["graph_labels"] # Not used directly in this script's training loop logic
    # element_to_index = processed_data["element_to_index"] # Not used directly
else:
    print(f"Error: Preprocessed data file not found at {PROCESSED_DATA_FILE}")
    exit()

# Initialize the graph generator using all graphs
generator = PaddedGraphGenerator(graphs=graphs)

######################################
# Grid Search Parameters
######################################
layer_sizes_combinations = [
    [256, 128, 64, 1], [512, 256, 128, 1], [512, 512, 256, 1], [512, 256, 64, 1],
    [768, 384, 192, 1], [768, 512, 256, 1], [1024, 512, 256, 1], [1024, 256, 128, 1],
    [128, 128, 128, 1], [256, 256, 256, 1], [512, 128, 64, 1], [256, 128, 32, 1],
    [512, 384, 128, 1], [768, 256, 64, 1], [1024, 768, 512, 1], [1024, 1024, 1024, 1],
    [64, 32, 16, 1], [128, 64, 32, 1], [256, 64, 32, 1], [512, 128, 32, 1]
]

grid_search_results = []

# Fixed parameters for all models in the grid search
epochs = 10000  # Max epochs, early stopping will likely trigger sooner
k_param = 25
filter1 = 16
filter2 = 32
filter3 = 128
batch_size = 32
learning_rate = 0.0001

# Prepare data split once
indices = np.arange(len(labels))
# Ensure labels are a numpy array for consistent indexing
np_labels = np.array(labels)
train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42, stratify=np_labels[indices] if len(np.unique(np_labels)) > 1 else None)

######################################
# Grid Search Loop
######################################
for i, current_layer_sizes in enumerate(layer_sizes_combinations):
    print(f"\n--- Training Model {i+1}/{len(layer_sizes_combinations)} with layer_sizes: {current_layer_sizes} ---")
    
    # Create the DeepGraphCNN model
    dgcnn_model = DeepGraphCNN(
        layer_sizes=current_layer_sizes,
        activations=["tanh"] * len(current_layer_sizes), # Ensure activations match layer_sizes length
        k=k_param,
        bias=False,
        generator=generator,
    )
    x_inp, x_out = dgcnn_model.in_out_tensors()

    x_out = Conv1D(filters=filter1, kernel_size=sum(current_layer_sizes), strides=sum(current_layer_sizes))(x_out)
    x_out = MaxPool1D(pool_size=2)(x_out)
    x_out = Conv1D(filters=filter2, kernel_size=5, strides=1)(x_out)
    x_out = Flatten()(x_out)
    x_out = Dense(units=filter3, activation="relu")(x_out)
    x_out = Dropout(rate=0.2)(x_out)
    predictions = Dense(units=1, activation="sigmoid")(x_out)

    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(optimizer=Adam(lr=learning_rate), loss=binary_crossentropy, metrics=["acc"])

    # Early stopping callback
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1) # Increased patience slightly

    # Build generators for this specific training run
    train_gen = generator.flow(train_idx, targets=np_labels[train_idx], batch_size=batch_size)
    val_gen = generator.flow(val_idx, targets=np_labels[val_idx], batch_size=batch_size)
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        verbose=1, # 0 for silent, 1 for progress bar, 2 for one line per epoch
        shuffle=True,
        callbacks=[early_stopping_callback]
    )
    
    # Evaluate the model with restored best weights on the validation set
    eval_loss, eval_acc = model.evaluate(val_gen, verbose=0)
    
    grid_search_results.append({
        'layer_sizes': current_layer_sizes,
        'val_accuracy': eval_acc,
        'val_loss': eval_loss,
        'epochs_trained': len(history.history['loss']) # Actual epochs run due to early stopping
    })
    print(f"Layer sizes: {current_layer_sizes}, Validation Accuracy: {eval_acc:.4f}, Validation Loss: {eval_loss:.4f}, Epochs: {len(history.history['loss'])}")
    
    # Clear TensorFlow session to free memory (important for long loops)
    tf.keras.backend.clear_session()

######################################
# Process and Display Results
######################################
results_df = pd.DataFrame(grid_search_results)
results_df = results_df.sort_values(by='val_accuracy', ascending=False)

print("\n--- Grid Search Results (Sorted by Validation Accuracy) ---")
print(results_df.to_string())