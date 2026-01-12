import glob
import pickle
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, ConfusionMatrixDisplay
from scipy import interp
from tensorflow.keras.models import load_model
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import DeepGraphCNN, GCNSupervisedGraphClassification, SortPooling, GraphConvolution

# Append parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cv_splits.data_splits import load_cv_splits

# 1. Configuration
# Adjusted for extra_features_eval_small_mol_tox
PROCESSED_FILE = "../inflated_models/large_layers_overtrained_small_coadd.pkl"
SPLITS_FILE = "../large_layers_cv_splits/large_layers_cv_splits_small_mol_tox.pkl"

# 2. Load Data
print("Loading data...")
if not os.path.exists(PROCESSED_FILE):
    print(f"File not found: {PROCESSED_FILE}")
    sys.exit(1)

with open(PROCESSED_FILE, "rb") as f:
    processed_data = pickle.load(f)
graphs = np.array(processed_data["graphs"])
labels = np.array(processed_data["labels"])
splits = load_cv_splits(SPLITS_FILE)

# 3. Model Loading Helper
def load_models_by_pattern(pattern):
    print(f"Loading models: {pattern}")
    paths = glob.glob(pattern)
    models = []
    for path in paths:
        try:
            # Assume format ..._fold_X.h5
            fold_num = int(path.split("_fold_")[-1].split(".")[0])
            model = load_model(path, compile=False, custom_objects={
                "DeepGraphCNN": DeepGraphCNN, "GCNSupervisedGraphClassification": GCNSupervisedGraphClassification,
                "SortPooling": SortPooling, "GraphConvolution": GraphConvolution
            })
            models.append((fold_num, model))
        except Exception as e:
            print(f"Error loading {path}: {e}")
    # Sort by fold number
    return sorted(models, key=lambda x: x[0])

# Load all model groups
# Adjusted paths for extra_features_eval_small_mol_tox
model_groups = {
    "Baseline": "../inflated_baseline_small_mol_tox/inflated_baseline_small_mol_tox_fold_*.h5",
    "Method 1 (Freeze GNN)": "../inflated_transfer_learning_p_to_smt/inflated_freeze_gnn_peptide_to_small_mol_tox_fold_*.h5",
    "Method 2 (Freeze Readout)": "../inflated_transfer_learning_p_to_smt/inflated_freeze_readout_peptide_to_small_mol_tox_fold_*.h5",
    "Method 3 (Freeze All)": "../inflated_transfer_learning_p_to_smt/inflated_freeze_all_peptide_to_small_mol_tox_fold_*.h5",
    "Method 4 (Gradual Unfreezing)": "../inflated_transfer_learning_p_to_smt/inflated_gradual_unfreezing_peptide_to_small_mol_tox_fold_*.h5"
}

# Pre-load all models to memory
loaded_models = {}
for name, pattern in model_groups.items():
    loaded_models[name] = load_models_by_pattern(pattern)

gen = PaddedGraphGenerator(graphs=graphs)

# ==========================================
# PLOT 1: ROC Curve Comparison
# ==========================================
def plot_roc_comparison():
    print("\nGenerating ROC Curve Comparison...")
    plt.figure(figsize=(10, 8))
    
    # Common X-axis for interpolation
    mean_fpr = np.linspace(0, 1, 100)

    for group_name, models_list in loaded_models.items():
        tprs = []
        aucs = []
        
        # Iterate over folds
        for fold_idx, fold in enumerate(splits, start=1):
            # Find model for this fold
            model = next((m for f, m in models_list if f == fold_idx), None)
            if not model: continue

            # Get test data
            test_idx = fold["test_idx"]
            X_test = graphs[test_idx]
            y_test = labels[test_idx]
            test_gen = gen.flow(X_test, y_test, batch_size=32, shuffle=False)
            
            # Predict
            y_pred = model.predict(test_gen, verbose=0).ravel()
            
            # Compute ROC
            fpr, tpr, _ = roc_curve(y_test, y_pred)
            
            # Interpolate TPR
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc(fpr, tpr))

        # Mean ROC for this group
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        plt.plot(mean_fpr, mean_tpr, 
                 label=f'{group_name} (AUC = {mean_auc:.3f} $\pm$ {std_auc:.3f})',
                 linewidth=2, alpha=0.8)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', label='Chance', alpha=0.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Comparison of ROC Curves (Averaged over 10 Folds)', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig("comparison_roc_curves.png", dpi=300)
    print("Saved comparison_roc_curves.png")
    plt.close()

# ==========================================
# PLOT 2: Precision-Recall Curve Comparison
# ==========================================
def plot_pr_comparison():
    print("\nGenerating Precision-Recall Comparison...")
    plt.figure(figsize=(10, 8))
    
    # Common X-axis (Recall) for interpolation
    mean_recall = np.linspace(0, 1, 100)

    for group_name, models_list in loaded_models.items():
        precisions = []
        aps = [] # Average Precision scores
        
        for fold_idx, fold in enumerate(splits, start=1):
            model = next((m for f, m in models_list if f == fold_idx), None)
            if not model: continue

            test_idx = fold["test_idx"]
            X_test = graphs[test_idx]
            y_test = labels[test_idx]
            test_gen = gen.flow(X_test, y_test, batch_size=32, shuffle=False)
            y_pred = model.predict(test_gen, verbose=0).ravel()
            
            # Compute PR
            precision, recall, _ = precision_recall_curve(y_test, y_pred)
            
            # Reversing for interpolation (recall must be increasing)
            precision, recall = precision[::-1], recall[::-1]
            
            # Interpolate Precision
            interp_prec = np.interp(mean_recall, recall, precision)
            precisions.append(interp_prec)
            aps.append(average_precision_score(y_test, y_pred))

        mean_precision = np.mean(precisions, axis=0)
        mean_ap = np.mean(aps)
        std_ap = np.std(aps)

        plt.plot(mean_recall, mean_precision, 
                 label=f'{group_name} (AP = {mean_ap:.3f} $\pm$ {std_ap:.3f})',
                 linewidth=2, alpha=0.8)

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Comparison of Precision-Recall Curves (Averaged over 10 Folds)', fontsize=14)
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.savefig("comparison_pr_curves.png", dpi=300)
    print("Saved comparison_pr_curves.png")
    plt.close()

# ==========================================
# PLOT 3: Aggregated Confusion Matrices
# ==========================================
def plot_confusion_matrices():
    print("\nGenerating Confusion Matrices...")
    
    for group_name, models_list in loaded_models.items():
        # Accumulate CM over all folds
        total_cm = np.zeros((2, 2), dtype=int)
        
        for fold_idx, fold in enumerate(splits, start=1):
            model = next((m for f, m in models_list if f == fold_idx), None)
            if not model: continue

            test_idx = fold["test_idx"]
            X_test = graphs[test_idx]
            y_test = labels[test_idx]
            test_gen = gen.flow(X_test, y_test, batch_size=32, shuffle=False)
            
            y_pred_prob = model.predict(test_gen, verbose=0).ravel()
            y_pred = (y_pred_prob >= 0.5).astype(int)
            
            cm = confusion_matrix(y_test, y_pred)
            total_cm += cm

        # Plot
        disp = ConfusionMatrixDisplay(confusion_matrix=total_cm, display_labels=["Non-Toxic", "Toxic"])
        
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
        plt.title(f'Confusion Matrix - {group_name}\n(Summed over 10 folds)')
        
        safe_name = group_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        filename = f"confusion_matrix_{safe_name}.png"
        plt.savefig(filename, dpi=300)
        print(f"Saved {filename}")
        plt.close()

# Run all
if __name__ == "__main__":
    plot_roc_comparison()
    plot_pr_comparison()
    plot_confusion_matrices()
    print("\nAll extra plots generated successfully!")

