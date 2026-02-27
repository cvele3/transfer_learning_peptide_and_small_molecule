import glob
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from stellargraph.layer import DeepGraphCNN, GCNSupervisedGraphClassification, SortPooling, GraphConvolution
import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare
from stellargraph.mapper import PaddedGraphGenerator
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Uvoz funkcija za generiranje/učitavanje CV splitova (cv_splits.pkl)
from cv_splits.data_splits import generate_cv_splits, load_cv_splits

PROCESSED_FILE = "../cv_splits_features/overtrained_peptide_graphs_72_features.pkl"
with open(PROCESSED_FILE, "rb") as f:
    processed_data = pickle.load(f)

# Assume processed_data contains: "graphs", "labels", "graph_labels"
graphs = np.array(processed_data["graphs"])
labels = np.array(processed_data["labels"])
graph_labels = processed_data["graph_labels"]

# Load the CV splits file (generated using data_splits.py)
SPLITS_FILE = "../cv_splits_features/cv_splits_peptide_72_features.pkl"
splits = load_cv_splits(SPLITS_FILE) 

def roc_auc_metric(y_true, y_pred):
    roc_auc = roc_auc_score(y_true, y_pred)
    # print("ROC AUC:", roc_auc)
    return roc_auc

def rest_of_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn + 1e-8)
    tnr = tn / (tn + fp + 1e-8)
    gm = math.sqrt(tpr * tnr)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    # print("GM:", gm)
    # print("Precision:", precision)
    # print("Recall:", recall)
    # print("F1:", f1)
    return gm, precision, recall, f1

def mcc_metric(y_true, y_pred):
    mcc = matthews_corrcoef(y_true, y_pred)
    # print("MCC:", mcc)
    return mcc


def extract_fold_number(filename):
    # Assuming the pattern has '_fold_' and the fold number comes after it.
    return int(filename.split("_fold_")[-1].split(".")[0])


def load_models_by_pattern(pattern):
    model_paths = glob.glob(pattern)
    models = []
    for path in model_paths:
        model = load_model(
            path,
            custom_objects={
                "DeepGraphCNN": DeepGraphCNN,
                "GCNSupervisedGraphClassification": GCNSupervisedGraphClassification,
                "SortPooling": SortPooling,
                "GraphConvolution": GraphConvolution,
            }
        )
        # Append tuple (filename, model)
        models.append((os.path.basename(path), model))
    return models

# Load the 10 baseline models
baseline_models = load_models_by_pattern("../inflated_72_feature_baseline_peptide/inflated_72_feature_baseline_peptide_fold_*.h5")
# Load the 10 models per transfer learning method
method1_models = load_models_by_pattern("../inflated_72_feature_smt_to_p_tl/inflated_72_feature_freeze_gnn_smile_mol_tox_to_peptide_fold_*.h5")
method2_models = load_models_by_pattern("../inflated_72_feature_smt_to_p_tl/inflated_72_feature_freeze_readout_smile_mol_tox_to_peptide_fold_*.h5")
method3_models = load_models_by_pattern("../inflated_72_feature_smt_to_p_tl/inflated_72_feature_freeze_all_smile_mol_tox_to_peptide_fold_*.h5")
method4_models = load_models_by_pattern("../inflated_72_feature_smt_to_p_tl/inflated_72_feature_gradual_unfreezing_smile_mol_tox_to_peptide_fold_*.h5")
# Print the sizes of all model lists
print(f"Number of baseline models loaded: {len(baseline_models)}")
print(f"Number of method1 models loaded: {len(method1_models)}")
print(f"Number of method2 models loaded: {len(method2_models)}")
print(f"Number of method3 models loaded: {len(method3_models)}")
print(f"Number of method4 models loaded: {len(method4_models)}")

baseline_models = sorted(baseline_models, key=lambda x: extract_fold_number(x[0]))
method1_models = sorted(method1_models, key=lambda x: extract_fold_number(x[0]))
method2_models = sorted(method2_models, key=lambda x: extract_fold_number(x[0]))
method3_models = sorted(method3_models, key=lambda x: extract_fold_number(x[0]))
method4_models = sorted(method4_models, key=lambda x: extract_fold_number(x[0]))

gen = PaddedGraphGenerator(graphs=graphs)

# Dictionary to store metrics for each fold and each model group.
results = {}

# Loop over each fold in the CV splits (assuming 10 folds)
for fold_idx, fold in enumerate(splits, start=1):
    test_idx = fold["test_idx"]
    X_test = graphs[test_idx]
    y_test = labels[test_idx]
    
    # Create a test generator for this fold
    test_gen = gen.flow(X_test, y_test, batch_size=32, shuffle=False)
    
    # Create a sub-dictionary for this fold
    results[fold_idx] = {}
    
    # Dictionary mapping group names to the corresponding sorted list of models.
    model_groups = {
        "baseline": baseline_models,
        "method1": method1_models,
        "method2": method2_models,
        "method3": method3_models,
        "method4": method4_models,
    }
    
    # For each model group, pick the model corresponding to the current fold
    for group_name, models_list in model_groups.items():
        # Assuming models_list is sorted by fold number,
        # the model for the current fold is located at index fold_idx-1.
        filename, model = models_list[fold_idx - 1]
        
        # Predict probabilities on the test set
        y_pred_probs = model.predict(test_gen)
        y_pred_probs = np.reshape(y_pred_probs, (-1,))
        
        # Calculate metrics (using threshold 0.5 for binary predictions)
        roc_auc = roc_auc_metric(y_test, y_pred_probs)
        # Convert probability to binary predictions
        y_pred = (y_pred_probs >= 0.5).astype(int)
        gm, precision, recall, f1 = rest_of_metrics(y_test, y_pred)
        mcc = mcc_metric(y_test, y_pred)
        
        # Store results for this model
        results[fold_idx][group_name] = {
            "filename": filename,
            "ROC_AUC": roc_auc,
            "GM": gm,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "MCC": mcc,
        }
        
        # print(f"Fold {fold_idx} - {group_name} - {filename}:")
        # print(f" ROC AUC: {roc_auc}")
        # print(f" GM: {gm}")
        # print(f" Precision: {precision}")
        # print(f" Recall: {recall}")
        # print(f" F1: {f1}")
        # print(f" MCC: {mcc}\n")

# At this point, the dictionary 'results' holds the metrics for each model, by fold and by group.
# You can then save or process 'results' as needed.

import numpy as np

# Dictionary to store summary stats for each model group
metrics_summary = {}

model_groups = ["baseline", "method1", "method2", "method3", "method4"]
metrics_list = ["ROC_AUC", "GM", "Precision", "Recall", "F1", "MCC"]

for group in model_groups:
    metrics_summary[group] = {}
    # Gather metric values from every fold for this model group
    for metric in metrics_list:
        values = [results[fold][group][metric] for fold in results]
        metrics_summary[group][metric] = {
            "mean": np.mean(values),
            "std": np.std(values)
        }

print("\nSummary of Metrics Across Folds:")
for group in model_groups:
    print(f"\nGroup: {group}")
    for metric in metrics_list:
        mean_val = metrics_summary[group][metric]["mean"]
        std_val = metrics_summary[group][metric]["std"]
        print(f" {metric}: Mean = {mean_val:.4f}, Std = {std_val:.4f}")

# Define model groups and metric names
model_groups = ["baseline", "method1", "method2", "method3", "method4"]
metric_names = ["ROC_AUC", "GM", "Precision", "Recall", "F1", "MCC"]

# Build a fold_results dictionary keyed by model group
fold_results = {group: {metric: [] for metric in metric_names} for group in model_groups}
for fold in sorted(results.keys()):
    for group in model_groups:
        for metric in metric_names:
            fold_results[group][metric].append(results[fold][group][metric])

# Print average performance per model group over the 10 folds
print("Evaluation metrics (averaged over 10 folds):")
for model in model_groups:
    print(f"\nModel: {model}")
    for metric in metric_names:
        values = fold_results[model][metric]
        # print(f"  {metric}: mean = {np.mean(values):.4f}, std = {np.std(values):.4f}")


friedman_results = {}

print("\nFriedman Test Results and Post Hoc Analysis:")
for metric in metric_names:
    # Kreiraj DataFrame gdje je svaki stupac model, a retci su vrijednosti po foldovima
    data_for_metric = {model: fold_results[model][metric] for model in model_groups}
    df_metric = pd.DataFrame(data_for_metric)
    
    # Provedi Friedman test (svaki stupac su ponovljene mjere preko foldova)
    data_list = [df_metric[col].values for col in df_metric.columns]
    stat, p_value = friedmanchisquare(*data_list)
    friedman_results[metric] = {"statistic": stat, "p_value": p_value}
    
    print(f"\nMetric: {metric}")
    print(f"  Friedman chi-square = {stat:.4f}, p-value = {p_value:.4f}")
    
    # Kreiraj boxplot za vizualizaciju
    df_melted = df_metric.melt(var_name="Model", value_name=metric)
    plt.figure(figsize=(10, 6))
    # Dodan parametar 'palette' za različite boje i 'hue' da se izbjegne upozorenje
    sns.boxplot(x="Model", y=metric, data=df_melted, hue="Model", palette="Set2", legend=False)
    plt.title(f"Box-and-Whisker Plot for {metric}")
    plt.xlabel("Model", fontsize=10)
    plt.ylabel(metric, fontsize=10)
    
    # Save the boxplot automatically
    box_plot_filename = f"box_whisker_{metric.lower()}_plot.png"
    plt.savefig(box_plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved boxplot to {box_plot_filename}")
    plt.close() # Close plot to free memory
    # plt.show() # Removed show() to focus on saving
    
    # Ako je Friedman test značajan, provedi Nemenyi post hoc test
    if p_value < 0.05:
        nemenyi_results = sp.posthoc_nemenyi_friedman(df_metric)
        print(f"\nPost Hoc Nemenyi Test Results for {metric}:")
        print(pd.DataFrame(nemenyi_results, index=df_metric.columns, columns=df_metric.columns))
        
        # Vizualizacija Nemenyi p-vrijednosti pomoću Heatmap-a
        plt.figure(figsize=(10, 8))
        # Koristimo masku da sakrijemo gornji trokut (jer je simetrično)
        mask = np.triu(np.ones_like(nemenyi_results, dtype=bool))
        
        # heatmap: p < 0.05 je značajno.
        # cmap="Reds_r": Tamno crveno za niske vrijednosti (značajna razlika), bijelo za visoke (nema razlike)
        sns.heatmap(nemenyi_results, annot=True, fmt=".3f", cmap="Reds_r", 
                    mask=mask, vmin=0, vmax=0.1, cbar_kws={'label': 'p-value (0=Significant)'})
        
        plt.title(f"Nemenyi Post-hoc P-values for {metric}\n(Darker Red = Significant Difference p<0.05)")
        plt.yticks(rotation=0) 
        plt.tight_layout()
        
        # Save the heatmap automatically
        heatmap_filename = f"heatmap_{metric.lower()}_plot.png"
        plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to {heatmap_filename}")
        plt.close() # Close plot
        # plt.show()
        
        # Odredi najbolji model prema srednjoj vrijednosti
        means = {model: np.mean(fold_results[model][metric]) for model in model_groups}
        best_model = max(means, key=means.get)
        best_mean = means[best_model]
        
        # Provjeri je li najbolji model statistički bolji od svih ostalih
        is_significantly_better = True
        for model in model_groups:
            if model == best_model:
                continue
            p_val_comparison = nemenyi_results.loc[best_model, model]
            if p_val_comparison >= 0.05:
                is_significantly_better = False
                break
        
        if is_significantly_better:
            print(f"\nFor metric '{metric}', the model '{best_model}' is significantly better than all others (mean = {best_mean:.4f}).")
        else:
            print(f"\nFor metric '{metric}', although '{best_model}' has the best mean (mean = {best_mean:.4f}), it is not significantly better than every other model.")
    else:
        print(f"No significant difference for {metric} (p-value >= 0.05); post hoc test not performed.")
