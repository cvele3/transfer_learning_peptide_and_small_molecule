import os
import math
import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "peptide_data.pkl")

SVM_KERNEL = "rbf"
SVM_C = 5
SVM_GAMMA = 0.001
SVM_THRESHOLD = -0.60
RANDOM_STATE = 42

AMINO_ACIDS = sorted(list("ACDEFGHIKLMNPQRSTVWY"))
DIPEPTIDES = [a1 + a2 for a1 in AMINO_ACIDS for a2 in AMINO_ACIDS]
DIPEPTIDE_INDEX = {dp: i for i, dp in enumerate(DIPEPTIDES)}


def compute_dpc(sequence):
    sequence = sequence.upper().strip()
    L = len(sequence)
    if L < 2:
        return np.zeros(400)
    counts = np.zeros(400)
    total_dipeptides = L - 1
    for i in range(total_dipeptides):
        dp = sequence[i:i + 2]
        if dp in DIPEPTIDE_INDEX:
            counts[DIPEPTIDE_INDEX[dp]] += 1
    if total_dipeptides > 0:
        counts = counts / total_dipeptides
    return counts


def compute_metrics(y_true, y_pred, y_pred_proba=None):
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    gm = math.sqrt(sensitivity * specificity)
    metrics = {
        "Accuracy": acc,
        "MCC": mcc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "GM": gm,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
    }
    if y_pred_proba is not None:
        auc = roc_auc_score(y_true, y_pred_proba)
        metrics["AUC"] = auc
    return metrics


def main():
    print("=" * 70)
    print("  DPC (Dipeptide Composition) + SVM")
    print("=" * 70)
    
    print(f"\n  Učitavanje: {DATA_FILE}")
    if not os.path.exists(DATA_FILE):
        print(f"  GRESKA: Datoteka '{DATA_FILE}' nije pronadjena!")
        return
    
    with open(DATA_FILE, "rb") as f:
        data = pickle.load(f)
    
    fasta_sequences = data["fasta_sequences"]
    labels = data["labels"]
    cv_splits = data["cv_splits"]
    n_folds = data["n_folds"]
    
    print(f"  Učitano sekvenci: {len(fasta_sequences)}")
    print(f"  Broj foldova: {n_folds}")
    
    print(f"\n  Računanje DPC značajki...")
    X_all = np.array([compute_dpc(seq) for seq in fasta_sequences])
    y_all = labels
    print(f"  Matrica značajki: {X_all.shape}")
    
    print(f"\n  Pokretanje {n_folds}-fold unakrsne validacije...")
    fold_metrics = []
    models_dir = os.path.join(SCRIPT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    for fold_idx, split in enumerate(cv_splits, 1):
        train_idx = split["train_idx"]
        test_idx = split["test_idx"]
        
        X_train = X_all[train_idx]
        X_test = X_all[test_idx]
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        svm = SVC(
            kernel=SVM_KERNEL,
            C=SVM_C,
            gamma=SVM_GAMMA,
            class_weight='balanced',
            probability=True,
            random_state=RANDOM_STATE + fold_idx,
        )
        svm.fit(X_train_scaled, y_train)
        
        decision_scores = svm.decision_function(X_test_scaled)
        y_pred = (decision_scores > SVM_THRESHOLD).astype(int)
        y_pred_proba = svm.predict_proba(X_test_scaled)[:, 1]
        
        metrics = compute_metrics(y_test, y_pred, y_pred_proba)
        fold_metrics.append(metrics)
        
        model_path = os.path.join(models_dir, f"dpc_svm_fold_{fold_idx}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({"model": svm, "scaler": scaler}, f)
        
        print(f"  Fold {fold_idx:2d}: AUC={metrics['AUC']:.4f}  "
              f"GM={metrics['GM']:.4f}  Precision={metrics['Precision']:.4f}  "
              f"Recall={metrics['Recall']:.4f}  F1={metrics['F1']:.4f}  "
              f"MCC={metrics['MCC']:.4f}")
    
    print(f"\n{'=' * 70}")
    print("  SUMARNI REZULTATI")
    print(f"{'=' * 70}")
    
    metric_names = ["AUC", "GM", "Precision", "Recall", "F1", "MCC"]
    
    summary = {}
    for metric_name in metric_names:
        values = [m[metric_name] for m in fold_metrics]
        mean_val = np.mean(values)
        std_val = np.std(values)
        summary[metric_name] = {"mean": mean_val, "std": std_val}
        print(f"  {metric_name:15s}:  {mean_val:.4f} +/- {std_val:.4f}")
    
    results_file = os.path.join(SCRIPT_DIR, "DPC_SVM_results.xlsx")
    results_data = []
    for fold_idx, metrics in enumerate(fold_metrics, 1):
        row = {"Fold": fold_idx}
        for metric_name in metric_names:
            row[metric_name] = metrics[metric_name]
        results_data.append(row)
    
    mean_row = {"Fold": "Mean"}
    std_row = {"Fold": "Std"}
    for metric_name in metric_names:
        values = [m[metric_name] for m in fold_metrics]
        mean_row[metric_name] = np.mean(values)
        std_row[metric_name] = np.std(values)
    results_data.append(mean_row)
    results_data.append(std_row)
    
    results_df = pd.DataFrame(results_data)
    results_df.to_excel(results_file, index=False)
    print(f"\n  Rezultati spremljeni u: {results_file}")
    print(f"  Spremljeno {n_folds} SVM modela u: {models_dir}/")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
