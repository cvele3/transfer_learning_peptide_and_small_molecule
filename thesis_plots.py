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

# --- CONFIGURATION ---
# Path to your data (using the one inside inflated_models)
DATA_PATH = "inflated_models/large_layers_overtrained_peptide.pkl" 
# Path to the model
MODEL_PATH = "inflated_models/inflated_peptide_model.h5" 
# Path to the corresponding CV splits (assuming they are in the standard location or inflated_models if specific ones exist)
# Note: Check if you have specific splits for large layers. Based on your file structure, you have 'large_layers_cv_splits' folder.
SPLITS_PATH = "large_layers_cv_splits/large_layers_cv_splits_peptide.pkl"
FOLD_TO_PLOT = 1  # 1-based index to match your filenames

def load_data_and_model():
    # 1. Load Data
    print("Loading data...")
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)
    graphs = data["graphs"]
    labels = np.array(data["labels"])
    
    # 2. Load Splits to get the correct Test Set for this fold
    with open(SPLITS_PATH, "rb") as f:
        splits = pickle.load(f)
    
    # Adjust for 0-based index (Fold 1 is index 0)
    test_idx = splits[FOLD_TO_PLOT - 1]["test_idx"]
    
    X_test = [graphs[i] for i in test_idx]
    y_test = labels[test_idx]

    # 3. Load Model
    print(f"Loading model: {MODEL_PATH}...")
    custom_objects = {
        "DeepGraphCNN": DeepGraphCNN,
        "GCNSupervisedGraphClassification": GCNSupervisedGraphClassification,
        "SortPooling": SortPooling,
        "GraphConvolution": GraphConvolution,
    }
    model = load_model(MODEL_PATH, custom_objects=custom_objects)
    
    return model, X_test, y_test

def plot_thesis_figures():
    model, X_test, y_test = load_data_and_model()
    
    # Prepare Generator
    generator = PaddedGraphGenerator(graphs=X_test)
    test_gen = generator.flow(list(range(len(X_test))), targets=y_test, batch_size=32, shuffle=False)
    
    # --- 1. PREDICTIONS ---
    print("Running predictions...")
    y_pred_probs = model.predict(test_gen).flatten()
    y_pred_binary = (y_pred_probs > 0.5).astype(int)

    # --- FIGURE 1: CONFUSION MATRIX ---
    cm = confusion_matrix(y_test, y_pred_binary)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={'size': 14})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Confusion Matrix (Fold {FOLD_TO_PLOT})', fontsize=14)
    plt.show()
    
    # --- FIGURE 2: ROC CURVE ---
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

    # --- FIGURE 3: t-SNE EMBEDDINGS ---
    # We want the output of the layer BEFORE the final classification
    # Usually the second to last layer (Dense)
    print("Generating t-SNE embeddings (this might take a moment)...")
    
    # Create a new model that outputs the penultimate layer
    # Note: 'dense' might be named differently (e.g. dense_1). 
    # Check model.summary() if this fails, or use index -2
    penultimate_layer_output = model.layers[-2].output 
    embedding_model = Model(inputs=model.input, outputs=penultimate_layer_output)
    
    embeddings = embedding_model.predict(test_gen)
    
    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                         c=y_test, cmap='coolwarm', alpha=0.7)
    plt.colorbar(scatter, label='Toxicity (0=Non-toxic, 1=Toxic)')
    plt.title('t-SNE Visualization of Learned Features')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.show()

if __name__ == "__main__":
    plot_thesis_figures()