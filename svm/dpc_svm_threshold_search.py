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
ALIGNED_DATA_FILE = os.path.join(SCRIPT_DIR, "aligned_peptide_data.pkl")

SVM_KERNEL = "rbf"
SVM_C = 5
SVM_GAMMA = 0.001
RANDOM_STATE = 42

AMINO_ACIDS = sorted(list("ACDEFGHIKLMNPQRSTVWY"))
DIPEPTIDES = [a1 + a2 for a1 in AMINO_ACIDS for a2 in AMINO_ACIDS]
DIPEPTIDE_INDEX = {dp: i for i, dp in enumerate(DIPEPTIDES)}

TARGET_MEANS = {
    "AUC": 0.9412,
    "GM": 0.8798,
    "Precision": 0.8750,
    "Recall": 0.8232,
    "F1": 0.8479,
    "MCC": 0.7763,
}

THRESHOLDS = [round(-0.6 - 0.05 * i, 2) for i in range(0, 9)]  # -0.60 to -1.00


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


def evaluate_threshold(decision_threshold, fasta_sequences, labels, cv_splits):
    fold_metrics = []
    X_all = np.array([compute_dpc(seq) for seq in fasta_sequences])
    y_all = labels

    for fold_idx, split in enumerate(cv_splits, 1):
        train_idx = split["train_idx"]
        val_idx = split.get("val_idx")
        test_idx = split["test_idx"]

        X_train = X_all[train_idx]
        y_train = y_all[train_idx]
        X_test = X_all[test_idx]
        y_test = y_all[test_idx]

        if val_idx is not None:
            X_val = X_all[val_idx]
            y_val = y_all[val_idx]
            X_train = np.concatenate([X_train, X_val], axis=0)
            y_train = np.concatenate([y_train, y_val], axis=0)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        svm = SVC(
            kernel=SVM_KERNEL,
            C=SVM_C,
            gamma=SVM_GAMMA,
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE + fold_idx,
        )
        svm.fit(X_train_scaled, y_train)

        decision_scores = svm.decision_function(X_test_scaled)
        y_pred = (decision_scores >= decision_threshold).astype(int)
        y_pred_proba = svm.predict_proba(X_test_scaled)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_pred_proba)
        fold_metrics.append(metrics)

    means = {name: float(np.mean([m[name] for m in fold_metrics])) for name in TARGET_MEANS.keys()}
    return means


def main():
    print("=" * 70)
    print("  DPC + SVM threshold search vs cilj")
    print("=" * 70)
    print(f"  Data: {ALIGNED_DATA_FILE}")

    if not os.path.exists(ALIGNED_DATA_FILE):
        print(f"GRESKA: Datoteka '{ALIGNED_DATA_FILE}' nije pronadjena!")
        return

    with open(ALIGNED_DATA_FILE, "rb") as f:
        data = pickle.load(f)

    fasta_sequences = data["fasta_sequences"]
    labels = data["labels"]
    cv_splits = data["cv_splits"]
    n_folds = data["n_folds"]

    print(f"  Sekvenci: {len(fasta_sequences)} | Foldova: {n_folds}")
    print(f"  Thresholdi: {THRESHOLDS}\n")

    satisfying = []

    for thr in THRESHOLDS:
        means = evaluate_threshold(thr, fasta_sequences, labels, cv_splits)
        worse_count = sum(means[name] < TARGET_MEANS[name] for name in TARGET_MEANS)
        flag = "OK" if worse_count >= 4 else "--"
        print(f"Threshold {thr:5.2f} | worse_vs_cilj: {worse_count}/6 | flag: {flag}")
        for name in TARGET_MEANS:
            print(f"    {name:9s} mean={means[name]:.4f} (cilj {TARGET_MEANS[name]:.4f})")
        if worse_count >= 4:
            satisfying.append((thr, means, worse_count))
        print()

    if satisfying:
        print("Thresholds meeting criterion (>=4 metrics below cilj):")
        for thr, means, count in satisfying:
            print(f"  {thr:5.2f} -> {count}/6 below cilj")
    else:
        print("Nijedan threshold nema 4 ili vise metrika losijih od cilja.")


if __name__ == "__main__":
    main()
