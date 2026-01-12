import glob
import pickle
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from math import pi
from sklearn.calibration import calibration_curve
from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from tensorflow.keras.models import load_model
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import DeepGraphCNN, GCNSupervisedGraphClassification, SortPooling, GraphConvolution

# Append parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cv_splits.data_splits import load_cv_splits

# 1. Configuration
# Adjusted for large_layers_eval_peptide
PROCESSED_FILE = "../large_layers_models/large_layers_overtrained_peptide.pkl"
SPLITS_FILE = "../large_layers_cv_splits/large_layers_cv_splits_peptide.pkl"

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
gen = PaddedGraphGenerator(graphs=graphs)

# 3. Model Loading
# Adjusted paths for large_layers_eval_peptide
model_groups = {
    "Baseline": "../large_layers_baseline_peptide/LL_baseline_peptide_fold_*.h5",
    "Method 1": "../large_layers_transfer_learning_smt_to_p/LL_freeze_gnn_smile_mol_tox_to_peptide_fold_*.h5",
    "Method 2": "../large_layers_transfer_learning_smt_to_p/LL_freeze_readout_smile_mol_tox_to_peptide_fold_*.h5",
    "Method 3": "../large_layers_transfer_learning_smt_to_p/LL_freeze_all_smile_mol_tox_to_peptide_fold_*.h5",
    "Method 4": "../large_layers_transfer_learning_smt_to_p/LL_gradual_unfreezing_smile_mol_tox_to_peptide_fold_*.h5"
}

def load_models_for_group(pattern):
    paths = glob.glob(pattern)
    models = []
    for path in paths:
        try:
            fold_num = int(path.split("_fold_")[-1].split(".")[0])
            model = load_model(path, compile=False, custom_objects={
                "DeepGraphCNN": DeepGraphCNN, "GCNSupervisedGraphClassification": GCNSupervisedGraphClassification,
                "SortPooling": SortPooling, "GraphConvolution": GraphConvolution
            })
            models.append((fold_num, model))
        except: pass
    return sorted(models, key=lambda x: x[0])

loaded_models = {k: load_models_for_group(v) for k, v in model_groups.items()}

# 4. Gather Data
print("\nGathering predictions across folds...")
results_df = [] # For Violin Plots
aggregated_probs = {k: {"y_true": [], "y_prob": []} for k in model_groups.keys()} # For Calibration

metrics_summary = {k: {"MCC": [], "F1": [], "AUC": [], "Precision": [], "Recall": [], "Accuracy": []} for k in model_groups.keys()}

for fold_idx, fold in enumerate(splits, start=1):
    test_idx = fold["test_idx"]
    X_test = graphs[test_idx]
    y_test = labels[test_idx]
    test_gen = gen.flow(X_test, y_test, batch_size=32, shuffle=False)
    
    for group_name, models_list in loaded_models.items():
        model = next((m for f, m in models_list if f == fold_idx), None)
        if not model: continue
        
        y_prob = model.predict(test_gen, verbose=0).ravel()
        y_pred = (y_prob >= 0.5).astype(int)
        
        # Collect for Calibration
        aggregated_probs[group_name]["y_true"].extend(y_test)
        aggregated_probs[group_name]["y_prob"].extend(y_prob)
        
        # Collect for Violin/Radar
        mcc = matthews_corrcoef(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        
        metrics_summary[group_name]["MCC"].append(mcc)
        metrics_summary[group_name]["F1"].append(f1)
        metrics_summary[group_name]["AUC"].append(auc)
        metrics_summary[group_name]["Precision"].append(prec)
        metrics_summary[group_name]["Recall"].append(rec)
        metrics_summary[group_name]["Accuracy"].append(acc)
        
        results_df.append({
            "Model": group_name,
            "MCC": mcc,
            "F1": f1,
            "AUC": auc
        })

df = pd.DataFrame(results_df)

# ==========================================
# PLOT 1: Radar Chart
# ==========================================
def plot_radar_chart():
    print("Generating Radar Chart...")
    
    # Calculate means
    categories = ['MCC', 'F1', 'AUC', 'Precision', 'Recall', 'Accuracy']
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1] # Close the loop
    
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    
    # Draw one axe per variable + labels
    plt.xticks(angles[:-1], categories)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=7)
    plt.ylim(0, 1)
    
    # Plot each model
    colors = plt.cm.get_cmap("tab10", len(model_groups))
    
    for i, (group_name, metrics) in enumerate(metrics_summary.items()):
        values = [np.mean(metrics[cat]) for cat in categories]
        values += values[:1] # Close loop
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=group_name, color=colors(i))
        ax.fill(angles, values, color=colors(i), alpha=0.1)
        
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Model Performance Fingerprint (Mean Metrics)", y=1.08)
    plt.savefig("advanced_radar_chart.png", dpi=300)
    plt.close()

# ==========================================
# PLOT 2: Reliability Diagram (Calibration)
# ==========================================
def plot_calibration_curve():
    print("Generating Calibration Curve...")
    plt.figure(figsize=(10, 8))
    
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    
    for group_name, data in aggregated_probs.items():
        if len(data["y_true"]) == 0: continue
        
        prob_true, prob_pred = calibration_curve(data["y_true"], data["y_prob"], n_bins=10)
        plt.plot(prob_pred, prob_true, "s-", label=group_name)
        
    plt.ylabel("Fraction of Positives")
    plt.xlabel("Mean Predicted Probability")
    plt.ylim([-0.05, 1.05])
    plt.legend(loc="lower right")
    plt.title("Reliability Diagram (Calibration Curve)")
    plt.grid(alpha=0.3)
    plt.savefig("advanced_calibration_curve.png", dpi=300)
    plt.close()

# ==========================================
# PLOT 3: Violin Plots
# ==========================================
def plot_violin_distributions():
    print("Generating Violin Plots...")
    
    metrics_to_plot = ["MCC", "F1", "AUC"]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, metric in enumerate(metrics_to_plot):
        sns.violinplot(x="Model", y=metric, data=df, ax=axes[i], inner="point")
        axes[i].set_title(f"{metric} Distribution")
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(axis='y', alpha=0.3)
        
    plt.tight_layout()
    plt.savefig("advanced_violin_plots.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_radar_chart()
    plot_calibration_curve()
    plot_violin_distributions()
    print("\nAdvanced plots generated successfully.")

