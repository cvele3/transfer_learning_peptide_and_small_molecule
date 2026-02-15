"""
Script to create radar plots for peptide models across three GNN layer configurations.
Each configuration gets its own radar plot showing all models and metrics.
Uses MEDIAN instead of mean for aggregating metrics across folds.
"""

import glob
import pickle
import numpy as np
import pandas as pd
import math
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# TensorFlow/Keras imports
from tensorflow.keras.models import load_model
from stellargraph.layer import DeepGraphCNN, GCNSupervisedGraphClassification, SortPooling, GraphConvolution
from stellargraph.mapper import PaddedGraphGenerator

# Sklearn metrics
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Add parent directory for imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from cv_splits.data_splits import load_cv_splits


# ============== METRIC FUNCTIONS ==============
def roc_auc_metric(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

def rest_of_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn + 1e-8)
    tnr = tn / (tn + fp + 1e-8)
    gm = math.sqrt(tpr * tnr)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return gm, precision, recall, f1

def mcc_metric(y_true, y_pred):
    return matthews_corrcoef(y_true, y_pred)

def extract_fold_number(filename):
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
        models.append((os.path.basename(path), model))
    return models


# ============== EVALUATION FUNCTION ==============
def evaluate_configuration(config_name, processed_file, splits_file, model_patterns):
    """Evaluate a single GNN layer configuration and return median metrics."""
    print(f"\n{'='*60}")
    print(f"Evaluating configuration: {config_name}")
    print(f"{'='*60}")
    
    # Load data
    with open(processed_file, "rb") as f:
        processed_data = pickle.load(f)
    
    graphs = np.array(processed_data["graphs"])
    labels = np.array(processed_data["labels"])
    splits = load_cv_splits(splits_file)
    
    # Load models
    model_groups_loaded = {}
    for group_name, pattern in model_patterns.items():
        models = load_models_by_pattern(pattern)
        if len(models) == 0:
            print(f"  WARNING: No models found for {group_name} with pattern {pattern}")
        else:
            print(f"  Loaded {len(models)} models for {group_name}")
        model_groups_loaded[group_name] = sorted(models, key=lambda x: extract_fold_number(x[0]))
    
    gen = PaddedGraphGenerator(graphs=graphs)
    
    # Evaluate each fold
    results = {}
    for fold_idx, fold in enumerate(splits, start=1):
        test_idx = fold["test_idx"]
        X_test = graphs[test_idx]
        y_test = labels[test_idx]
        test_gen = gen.flow(X_test, y_test, batch_size=32, shuffle=False)
        
        results[fold_idx] = {}
        
        for group_name, models_list in model_groups_loaded.items():
            if len(models_list) < fold_idx:
                continue
            filename, model = models_list[fold_idx - 1]
            
            y_pred_probs = model.predict(test_gen, verbose=0)
            y_pred_probs = np.reshape(y_pred_probs, (-1,))
            
            roc_auc = roc_auc_metric(y_test, y_pred_probs)
            y_pred = (y_pred_probs >= 0.5).astype(int)
            gm, precision, recall, f1 = rest_of_metrics(y_test, y_pred)
            mcc = mcc_metric(y_test, y_pred)
            
            results[fold_idx][group_name] = {
                "ROC_AUC": roc_auc,
                "GM": gm,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "MCC": mcc,
            }
    
    # Compute median metrics for each model
    model_groups = ["baseline", "method1", "method2", "method3", "method4"]
    metric_names = ["ROC_AUC", "GM", "Precision", "Recall", "F1", "MCC"]
    
    metrics_summary = {}
    for group in model_groups:
        metrics_summary[group] = {}
        for metric in metric_names:
            values = [results[fold][group][metric] for fold in results if group in results[fold]]
            if len(values) > 0:
                metrics_summary[group][metric] = np.median(values)
            else:
                metrics_summary[group][metric] = 0.0
    
    return metrics_summary


def create_radar_plot(ax, metrics_summary, config_name, metric_names, model_groups, model_colors, model_labels):
    """Create a single radar plot on the given axes."""
    
    # Number of metrics
    num_metrics = len(metric_names)
    
    # Compute angle for each metric
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    # Plot each model
    for group in model_groups:
        values = [metrics_summary[group][metric] for metric in metric_names]
        values += values[:1]  # Complete the loop
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model_labels[group], color=model_colors[group])
        ax.fill(angles, values, alpha=0.1, color=model_colors[group])
    
    # Set the labels for each metric
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, size=10)
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)
    
    # Add title
    ax.set_title(config_name, size=14, fontweight='bold', pad=20)


def main():
    # Define configurations
    # Configuration 1: [25, 25, 25, 1] - Standard/Extra Features
    config1 = {
        "name": "[25, 25, 25, 1]",
        "processed_file": "inflated_models/large_layers_overtrained_peptide.pkl",
        "splits_file": "large_layers_cv_splits/large_layers_cv_splits_peptide.pkl",
        "model_patterns": {
            "baseline": "extra_features_baseline_peptide/baseline_peptide_fold_*.h5",
            "method1": "transfer_learning_extra_features_smt_to_p/freeze_gnn_small_mol_tox_to_peptide_fold_*.h5",
            "method2": "transfer_learning_extra_features_smt_to_p/freeze_readout_small_mol_tox_to_peptide_fold_*.h5",
            "method3": "transfer_learning_extra_features_smt_to_p/freeze_all_small_mol_tox_to_peptide_fold_*.h5",
            "method4": "transfer_learning_extra_features_smt_to_p/gradual_unfreezing_small_mol_tox_to_peptide_fold_*.h5",
        }
    }
    
    # Configuration 2: [125, 125, 125, 1] - Large Layers
    config2 = {
        "name": "[125, 125, 125, 1]",
        "processed_file": "large_layers_models/large_layers_overtrained_peptide.pkl",
        "splits_file": "large_layers_cv_splits/large_layers_cv_splits_peptide.pkl",
        "model_patterns": {
            "baseline": "large_layers_baseline_peptide/large_layers_baseline_peptide_fold_*.h5",
            "method1": "large_layers_transfer_learning_smt_to_p/LL_freeze_gnn_smile_mol_tox_to_peptide_fold_*.h5",
            "method2": "large_layers_transfer_learning_smt_to_p/LL_freeze_readout_smile_mol_tox_to_peptide_fold_*.h5",
            "method3": "large_layers_transfer_learning_smt_to_p/LL_freeze_all_smile_mol_tox_to_peptide_fold_*.h5",
            "method4": "large_layers_transfer_learning_smt_to_p/LL_gradual_unfreezing_smile_mol_tox_to_peptide_fold_*.h5",
        }
    }
    
    # Configuration 3: [512, 256, 128, 1] - Inflated
    config3 = {
        "name": "[512, 256, 128, 1]",
        "processed_file": "inflated_models/large_layers_overtrained_peptide.pkl",
        "splits_file": "large_layers_cv_splits/large_layers_cv_splits_peptide.pkl",
        "model_patterns": {
            "baseline": "inflated_baseline_peptide/inflated_baseline_peptide_fold_*.h5",
            "method1": "inflated_transfer_learning_smt_to_p/inflated_freeze_gnn_smile_mol_tox_to_peptide_fold_*.h5",
            "method2": "inflated_transfer_learning_smt_to_p/inflated_freeze_readout_smile_mol_tox_to_peptide_fold_*.h5",
            "method3": "inflated_transfer_learning_smt_to_p/inflated_freeze_all_smile_mol_tox_to_peptide_fold_*.h5",
            "method4": "inflated_transfer_learning_smt_to_p/inflated_gradual_unfreezing_smile_mol_tox_to_peptide_fold_*.h5",
        }
    }
    
    configurations = [config1, config2, config3]
    
    # Model display settings
    model_groups = ["baseline", "method1", "method2", "method3", "method4"]
    model_labels = {
        "baseline": "Baseline",
        "method1": "Method 1 - Freeze GNN",
        "method2": "Method 2 - Freeze Readout",
        "method3": "Method 3 - Freeze All",
        "method4": "Method 4 - Gradual Unfreezing",
    }
    model_colors = {
        "baseline": "#1f77b4",  # Blue
        "method1": "#ff7f0e",   # Orange
        "method2": "#2ca02c",   # Green
        "method3": "#d62728",   # Red
        "method4": "#9467bd",   # Purple
    }
    
    metric_names = ["ROC_AUC", "GM", "Precision", "Recall", "F1", "MCC"]
    
    # Collect all results
    all_results = {}
    for config in configurations:
        metrics_summary = evaluate_configuration(
            config["name"],
            config["processed_file"],
            config["splits_file"],
            config["model_patterns"]
        )
        all_results[config["name"]] = metrics_summary
    
    # Create figure with 3 radar plots - increased height for title and legend
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), subplot_kw=dict(projection='polar'))
    
    # Adjust subplot positions to make room for title and legend
    plt.subplots_adjust(top=0.85, bottom=0.15, wspace=0.3)
    
    for idx, config in enumerate(configurations):
        config_name = config["name"]
        metrics_summary = all_results[config_name]
        create_radar_plot(
            axes[idx], 
            metrics_summary, 
            config_name, 
            metric_names, 
            model_groups, 
            model_colors, 
            model_labels
        )
    
    # Add main title
    fig.suptitle('Peptide Models - Performance Comparison Across Layer Configurations (Median)', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Add a single legend for all subplots
    handles = [Patch(facecolor=model_colors[group], label=model_labels[group], alpha=0.7) 
               for group in model_groups]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=10, 
               bbox_to_anchor=(0.5, 0.02))
    
    # Save the figure
    output_file = "radar_plots_peptide_median.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nRadar plots saved to: {output_file}")
    
    plt.show()


if __name__ == "__main__":
    main()
