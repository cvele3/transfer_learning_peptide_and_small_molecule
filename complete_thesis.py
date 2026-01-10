import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.manifold import TSNE
from tensorflow.keras.models import load_model, Model
from stellargraph.mapper import PaddedGraphGenerator
# Import custom layers needed for loading
from stellargraph.layer import DeepGraphCNN, GCNSupervisedGraphClassification, SortPooling, GraphConvolution

# ==========================================
# CONFIGURATION REGISTRY
# ==========================================
# Dictionary defining all the experiments we want to plot.
# Each entry contains the paths to the Data, Model, and CV Splits.

CONFIGS = {
    # ---------------------------------------------------------
    # 1. NORMAL (OVERTRAINED) MODELS
    # ---------------------------------------------------------
    "Normal_Peptide": {
        "data_path": "overtrained_models/overtrained_peptide_graphs.pkl",
        "model_path": "overtrained_models/overtrained_peptide_model.h5",
        "splits_path": "cv_splits/cv_splits_peptide.pkl",
        "fold": 1
    },
    "Normal_SmallMolTox": {
        "data_path": "overtrained_models/overtrained_small_molecule_mol_tox_pred_graphs.pkl",
        "model_path": "overtrained_models/overtrained_small_molecule_mol_tox_model.h5",
        "splits_path": "cv_splits/cv_splits_small_mol_tox_pred_.pkl",
        "fold": 1
    },

    # ---------------------------------------------------------
    # 2. INFLATED MODELS
    # ---------------------------------------------------------
    # Note: Inflated models often use the "large_layers" data/splits because they have expanded feature sets.
    "Inflated_Peptide": {
        "data_path": "inflated_models/large_layers_overtrained_peptide.pkl",
        "model_path": "inflated_models/inflated_peptide_model.h5",
        "splits_path": "large_layers_cv_splits/large_layers_cv_splits_peptide.pkl",
        "fold": 1
    },
    "Inflated_SmallMolTox": {
        "data_path": "inflated_models/large_layers_overtrained_small_mol_tox_pred.pkl",
        "model_path": "inflated_models/inflated_small_moll_tox_model.h5",
        "splits_path": "large_layers_cv_splits/large_layers_cv_splits_small_mol_tox_pred.pkl",
        "fold": 1
    },

    # ---------------------------------------------------------
    # 3. LARGE LAYERS MODELS
    # ---------------------------------------------------------
    "LargeLayers_Peptide": {
        "data_path": "large_layers_models/large_layers_overtrained_peptide.pkl",
        "model_path": "large_layers_models/larger_layers_peptide_model.h5",
        "splits_path": "large_layers_cv_splits/large_layers_cv_splits_peptide.pkl",
        "fold": 1
    },
    "LargeLayers_SmallMolTox": {
        "data_path": "large_layers_models/large_layers_overtrained_small_mol_tox_pred.pkl",
        "model_path": "large_layers_models/larger_layers_small_moll_tox_model.h5",
        "splits_path": "large_layers_cv_splits/large_layers_cv_splits_small_mol_tox_pred.pkl",
        "fold": 1
    },
}

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def load_data_and_model(config_name, config):
    """Loads data, splits, and model for a given configuration."""
    print(f"\n[{config_name}] Loading resources...")
    
    # 1. Check file existence
    if not os.path.exists(config["data_path"]):
        raise FileNotFoundError(f"Data file not found: {config['data_path']}")
    if not os.path.exists(config["model_path"]):
        raise FileNotFoundError(f"Model file not found: {config['model_path']}")
    if not os.path.exists(config["splits_path"]):
        raise FileNotFoundError(f"Splits file not found: {config['splits_path']}")

    # 2. Load Data
    with open(config["data_path"], "rb") as f:
        data = pickle.load(f)
    graphs = data["graphs"]
    labels = np.array(data["labels"])
    
    # 3. Load Splits to get the correct Test Set for this fold
    with open(config["splits_path"], "rb") as f:
        splits = pickle.load(f)
    
    # Adjust for 0-based index if needed. Fold 1 -> Index 0.
    fold_idx = config["fold"] - 1
    if fold_idx >= len(splits):
         raise ValueError(f"Fold {config['fold']} out of range (max {len(splits)})")
         
    test_idx = splits[fold_idx]["test_idx"]
    
    X_test = [graphs[i] for i in test_idx]
    y_test = labels[test_idx]
    
    print(f"[{config_name}] Data loaded. Test set size: {len(X_test)}")

    # 4. Load Model
    custom_objects = {
        "DeepGraphCNN": DeepGraphCNN,
        "GCNSupervisedGraphClassification": GCNSupervisedGraphClassification,
        "SortPooling": SortPooling,
        "GraphConvolution": GraphConvolution,
    }
    model = load_model(config["model_path"], custom_objects=custom_objects)
    print(f"[{config_name}] Model loaded successfully.")
    
    return model, X_test, y_test

def generate_plots(config_name, model, X_test, y_test):
    """Generates and displays plots for a single experiment configuration."""
    
    # Prepare Generator
    generator = PaddedGraphGenerator(graphs=X_test)
    test_gen = generator.flow(list(range(len(X_test))), targets=y_test, batch_size=32, shuffle=False)
    
    # --- PREDICTIONS ---
    print(f"[{config_name}] Running predictions...")
    y_pred_probs = model.predict(test_gen).flatten()
    y_pred_binary = (y_pred_probs > 0.5).astype(int)

    # --- FIGURE 1: CONFUSION MATRIX ---
    cm = confusion_matrix(y_test, y_pred_binary)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={'size': 14})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'{config_name} - Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # --- FIGURE 2: ROC CURVE ---
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{config_name} - ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- FIGURE 3: t-SNE EMBEDDINGS ---
    # We want the output of the layer BEFORE the final classification
    print(f"[{config_name}] Generating t-SNE embeddings...")
    
    # Attempt to find the penultimate layer (usually Dense before the final Dense)
    # Strategy: Look for the last Dense layer that is NOT the output layer.
    # If the last layer is Dense (1 unit), we want the one before it.
    
    try:
        # Simple heuristic: -2 usually works for standard Keras Sequential/Functional models 
        # where the last layers are Dense -> Dense(output)
        penultimate_layer_output = model.layers[-2].output 
        embedding_model = Model(inputs=model.input, outputs=penultimate_layer_output)
        
        embeddings = embedding_model.predict(test_gen)
        
        # Run t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
        tsne_results = tsne.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                             c=y_test, cmap='coolwarm', alpha=0.7)
        cbar = plt.colorbar(scatter)
        cbar.set_label('Toxicity (0=Non-toxic, 1=Toxic)', rotation=270, labelpad=15)
        plt.title(f'{config_name} - t-SNE Visualization')
        plt.xlabel('t-SNE dimension 1')
        plt.ylabel('t-SNE dimension 2')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"[{config_name}] Could not generate t-SNE plot. Error: {e}")

# ==========================================
# MAIN EXECUTION LOOP
# ==========================================

if __name__ == "__main__":
    print("Starting Thesis Plots Generation...")
    
    for name, config in CONFIGS.items():
        try:
            print(f"\n{'='*50}")
            print(f"PROCESSING: {name}")
            print(f"{'='*50}")
            
            model, X_test, y_test = load_data_and_model(name, config)
            generate_plots(name, model, X_test, y_test)
            
        except Exception as e:
            print(f"\n!!! SKIPPING {name} due to error: {e}")
            continue
            
    print("\nAll processing complete.")
