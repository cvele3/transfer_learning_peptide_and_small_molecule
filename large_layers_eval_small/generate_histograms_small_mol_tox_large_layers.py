import glob
import pickle
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import DeepGraphCNN, GCNSupervisedGraphClassification, SortPooling, GraphConvolution
import sys

# Append parent directory to path to import local modules if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import splits loader
from cv_splits.data_splits import load_cv_splits

# 1. Configuration and Constants
PROCESSED_FILE = "../large_layers_models/large_layers_overtrained_small_mol_tox_pred.pkl"
SPLITS_FILE = "../large_layers_cv_splits/large_layers_cv_splits_small_mol_tox_pred.pkl"

# 2. Load Data and Splits
print("Loading processed data...")
if not os.path.exists(PROCESSED_FILE):
    print(f"Error: Processed data file not found at {PROCESSED_FILE}")
    sys.exit(1)

with open(PROCESSED_FILE, "rb") as f:
    processed_data = pickle.load(f)

graphs = np.array(processed_data["graphs"])
labels = np.array(processed_data["labels"])
# graph_labels = processed_data["graph_labels"]

print("Loading CV splits...")
if not os.path.exists(SPLITS_FILE):
    print(f"Error: Splits file not found at {SPLITS_FILE}")
    sys.exit(1)

splits = load_cv_splits(SPLITS_FILE)

# 3. Helper Functions
def extract_fold_number(filename):
    try:
        return int(filename.split("_fold_")[-1].split(".")[0])
    except:
        return 0

def load_models_by_pattern(pattern):
    print(f"Loading models matching: {pattern}")
    model_paths = glob.glob(pattern)
    models = []
    for path in model_paths:
        try:
            model = load_model(
                path,
                custom_objects={
                    "DeepGraphCNN": DeepGraphCNN,
                    "GCNSupervisedGraphClassification": GCNSupervisedGraphClassification,
                    "SortPooling": SortPooling,
                    "GraphConvolution": GraphConvolution,
                },
                compile=False 
            )
            models.append((os.path.basename(path), model))
        except Exception as e:
            print(f"Failed to load model {path}: {e}")
            
    models = sorted(models, key=lambda x: extract_fold_number(x[0]))
    return models

def generate_histogram(model_name, y_true_all, y_pred_prob_all):
    correct_probs = []
    incorrect_probs = []
    
    threshold = 0.5
    
    for true_label, pred_prob in zip(y_true_all, y_pred_prob_all):
        pred_label = 1 if pred_prob >= threshold else 0
        if pred_label == true_label:
            correct_probs.append(pred_prob)
        else:
            incorrect_probs.append(pred_prob)
            
    bins = np.arange(0.0, 1.05, 0.05)
    
    plt.figure(figsize=(10, 6))
    
    n_correct, bins_correct, patches_correct = plt.hist(
        correct_probs, 
        bins=bins, 
        color='blue', 
        alpha=0.5, 
        label='Correctly Classified', 
        edgecolor='black',
        linewidth=0.5
    )
    
    n_incorrect, bins_incorrect, patches_incorrect = plt.hist(
        incorrect_probs, 
        bins=bins, 
        color='red', 
        alpha=0.5, 
        label='Incorrectly Classified', 
        edgecolor='black',
        linewidth=0.5
    )
    
    def add_labels(counts, bins, color, vertical_offset=5):
        for count, x in zip(counts, bins):
            if count > 0:
                plt.text(
                    x + 0.025,
                    count + vertical_offset, 
                    str(int(count)), 
                    ha='center', 
                    va='bottom', 
                    fontsize=9, 
                    color=color,
                    fontweight='bold'
                )

    # Label positioning logic to avoid overlap
    for i in range(len(bins_correct) - 1):
        count_correct = n_correct[i]
        count_incorrect = n_incorrect[i]
        x_pos = bins_correct[i] + 0.025
        
        offset_correct = 5
        offset_incorrect = 5
        
        vertical_dist = abs(count_correct - count_incorrect)
        min_dist = 40 
        
        pos_correct = count_correct + offset_correct
        pos_incorrect = count_incorrect + offset_incorrect
        
        if vertical_dist < min_dist:
             if count_correct >= count_incorrect:
                 pos_correct += 25 
             else:
                 pos_incorrect += 25
                 
        if count_correct > 0:
            plt.text(x_pos, pos_correct, str(int(count_correct)), ha='center', va='bottom', fontsize=9, color='blue', fontweight='bold')
            
        if count_incorrect > 0:
            plt.text(x_pos, pos_incorrect, str(int(count_incorrect)), ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.xlabel('Predicted Toxicity Probability', fontsize=12)
    plt.ylabel('Number of Peptides/Molecules', fontsize=12)
    plt.title(f'Prediction Distribution - {model_name}', fontsize=14)
    plt.xlim(0, 1.0)
    plt.xticks(np.arange(0, 1.1, 0.1))
    
    plt.legend()
    
    output_filename = f"histogram_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved histogram to {output_filename}")
    plt.close()

# 4. Load Models
print("Loading models...")
baseline_models = load_models_by_pattern("../large_layers_baseline_small_mol_tox/large_layers_baseline_small_mol_tox_fold_*.h5")
method1_models = load_models_by_pattern("../large_layers_transfer_learning_p_to_smt/LL_freeze_gnn_peptide_to_smile_mol_tox_fold_*.h5")
method2_models = load_models_by_pattern("../large_layers_transfer_learning_p_to_smt/LL_freeze_readout_peptide_to_smile_mol_tox_fold_*.h5")
method3_models = load_models_by_pattern("../large_layers_transfer_learning_p_to_smt/LL_freeze_all_peptide_to_smile_mol_tox_fold_*.h5")
method4_models = load_models_by_pattern("../large_layers_transfer_learning_p_to_smt/LL_gradual_unfreezing_peptide_to_smile_mol_tox_fold_*.h5")

model_groups = {
    "Baseline": baseline_models,
    "Method 1 (Freeze GNN)": method1_models,
    "Method 2 (Freeze Readout)": method2_models,
    "Method 3 (Freeze All)": method3_models,
    "Method 4 (Gradual Unfreezing)": method4_models
}

# 5. Process Each Model Group
gen = PaddedGraphGenerator(graphs=graphs)

for group_name, models_list in model_groups.items():
    print(f"\nProcessing {group_name}...")
    
    if len(models_list) != 10:
        print(f"Warning: Expected 10 models for {group_name}, found {len(models_list)}. Checking folds...")
    
    all_y_true = []
    all_y_pred_prob = []
    
    for fold_idx, fold in enumerate(splits, start=1):
        model = None
        for name, m in models_list:
            if extract_fold_number(name) == fold_idx:
                model = m
                break
        
        if model is None:
            print(f"  Warning: No model found for Fold {fold_idx} in {group_name}. Skipping fold.")
            continue
            
        test_idx = fold["test_idx"]
        X_test = graphs[test_idx]
        y_test = labels[test_idx]
        
        test_gen = gen.flow(X_test, y_test, batch_size=32, shuffle=False)
        
        y_pred = model.predict(test_gen, verbose=0)
        y_pred = np.reshape(y_pred, (-1,))
        
        all_y_true.extend(y_test)
        all_y_pred_prob.extend(y_pred)
        
    all_y_true = np.array(all_y_true)
    all_y_pred_prob = np.array(all_y_pred_prob)
    
    print(f"  Aggregated {len(all_y_true)} predictions.")
    generate_histogram(group_name, all_y_true, all_y_pred_prob)

print("\nDone. All histograms generated.")
