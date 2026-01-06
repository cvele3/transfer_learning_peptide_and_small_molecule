import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from stellargraph.mapper import PaddedGraphGenerator
# Import necessary custom layers from StellarGraph for loading
from stellargraph.layer import DeepGraphCNN, GCNSupervisedGraphClassification, SortPooling, GraphConvolution
from sklearn.model_selection import train_test_split

# 1. Load the same data used for training
PROCESSED_DATA_FILE = "large_layers_overtrained_small_mol_tox_pred.pkl"

if os.path.exists(PROCESSED_DATA_FILE):
    print("Loading data...")
    with open(PROCESSED_DATA_FILE, "rb") as f:
        processed_data = pickle.load(f)
    graphs = processed_data["graphs"]
    labels = processed_data["labels"]
else:
    print(f"Error: Data file {PROCESSED_DATA_FILE} not found.")
    exit()

# 2. Recreate the exact same data split
# It is CRITICAL to use the same random_state=42 as in your training script
indices = np.arange(len(labels))
train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)

# 3. Create the generator for the validation set
generator = PaddedGraphGenerator(graphs=graphs)
val_gen = generator.flow(val_idx, targets=np.array(labels)[val_idx], batch_size=32)

# 4. Load the saved model
# You must pass the custom StellarGraph objects to load_model
custom_objects = {
    "DeepGraphCNN": DeepGraphCNN,
    "GCNSupervisedGraphClassification": GCNSupervisedGraphClassification,
    "SortPooling": SortPooling,
    "GraphConvolution": GraphConvolution,
}

model_path = 'inflated_small_moll_tox_model.h5'

if os.path.exists(model_path):
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, custom_objects=custom_objects)
    
    # 5. Evaluate the model
    print("Evaluating model on validation set...")
    results = model.evaluate(val_gen, verbose=1)
    
    # Print results
    # results[0] is loss, results[1] is accuracy (based on your compile metrics=["acc"])
    print("\n--- Evaluation Results ---")
    print(f"Validation Loss:     {results[0]:.4f}")
    print(f"Validation Accuracy: {results[1]:.4f}")
else:
    print(f"Error: Model file {model_path} not found.")
