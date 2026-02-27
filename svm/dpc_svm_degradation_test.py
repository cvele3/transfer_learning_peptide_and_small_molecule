"""
DPC + SVM model - test degradacije performansi.

Ova skripta testira različite scenarije koji bi trebali pogoršati rezultate
bez mijenjanja osnovnih hiperparametara (C=5, gamma=0.001).

Scenariji:
1. Bez StandardScaler-a (RBF kernel zahtijeva skalirane značajke)
2. Krivi redoslijed dipeptida (shuffle DPC vektora)
3. Ne normalizirane DPC značajke (count umjesto frequency)
4. Krivi kernel (linear umjesto RBF)
5. Shuffle labele (testiranje da model uči nasumične obrasce)
6. Manje podataka za trening (50% umjesto train+val)

Cilj: Provjeriti robustnost modela i važnost različitih komponenti.
"""

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

# ========================== KONFIGURACIJA ==========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Uskladjeni podaci
ALIGNED_DATA_FILE = os.path.join(SCRIPT_DIR, "aligned_peptide_data.pkl")

# SVM hiperparametri (iz rada)
SVM_KERNEL = "rbf"
SVM_C = 5
SVM_GAMMA = 0.001
RANDOM_STATE = 42

# 20 standardnih aminokiselina
AMINO_ACIDS = sorted(list("ACDEFGHIKLMNPQRSTVWY"))
DIPEPTIDES = [a1 + a2 for a1 in AMINO_ACIDS for a2 in AMINO_ACIDS]
DIPEPTIDE_INDEX = {dp: i for i, dp in enumerate(DIPEPTIDES)}


# ========================== DPC FEATURE ENGINEERING ==========================

def compute_dpc(sequence, normalize=True):
    """Racuna DPC vektor (normaliziran ili ne)."""
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

    if normalize and total_dipeptides > 0:
        counts = counts / total_dipeptides

    return counts


# ========================== METRIKE ==========================

def compute_metrics(y_true, y_pred, y_pred_proba=None):
    """Izracunava sve evaluacijske metrike."""
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


# ========================== SCENARIJI DEGRADACIJE ==========================

def scenario_1_no_scaler(X_train_full, y_train_full, X_test, y_test, fold_idx):
    """Scenarij 1: Bez StandardScaler-a (RBF zahtijeva skalirane značajke)."""
    print(f"    Scenarij 1: Bez StandardScaler-a")
    
    svm = SVC(
        kernel=SVM_KERNEL,
        C=SVM_C,
        gamma=SVM_GAMMA,
        class_weight=None,
        probability=True,
        random_state=RANDOM_STATE + fold_idx,
    )
    svm.fit(X_train_full, y_train_full)
    
    y_pred = svm.predict(X_test)
    y_pred_proba = svm.predict_proba(X_test)[:, 1]
    
    return compute_metrics(y_test, y_pred, y_pred_proba)


def scenario_2_shuffled_dpc(X_all, train_idx, val_idx, test_idx, y_all, fold_idx):
    """Scenarij 2: Shuffle DPC značajki (krivi redoslijed)."""
    print(f"    Scenarij 2: Shuffle DPC značajki")
    
    # Shuffle DPC vektori (reproduciabilno)
    X_shuffled = X_all.copy()
    np.random.seed(RANDOM_STATE + fold_idx)  # Reproducibilno shuffle
    for i in range(len(X_shuffled)):
        np.random.shuffle(X_shuffled[i])
    
    X_train = X_shuffled[train_idx]
    X_val = X_shuffled[val_idx]
    X_test = X_shuffled[test_idx]
    y_train = y_all[train_idx]
    y_val = y_all[val_idx]
    y_test = y_all[test_idx]
    
    X_train_full = np.concatenate([X_train, X_val], axis=0)
    y_train_full = np.concatenate([y_train, y_val], axis=0)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)
    
    svm = SVC(
        kernel=SVM_KERNEL,
        C=SVM_C,
        gamma=SVM_GAMMA,
        class_weight=None,
        probability=True,
        random_state=RANDOM_STATE + fold_idx,
    )
    svm.fit(X_train_scaled, y_train_full)
    
    y_pred = svm.predict(X_test_scaled)
    y_pred_proba = svm.predict_proba(X_test_scaled)[:, 1]
    
    return compute_metrics(y_test, y_pred, y_pred_proba)


def scenario_3_no_normalization(fasta_sequences, train_idx, val_idx, test_idx, y_all, fold_idx):
    """Scenarij 3: Ne normalizirane DPC značajke (count umjesto frequency)."""
    print(f"    Scenarij 3: Ne normalizirane DPC značajke")
    
    # DPC bez normalizacije (count umjesto frequency)
    X_all = np.array([compute_dpc(seq, normalize=False) for seq in fasta_sequences])
    
    X_train = X_all[train_idx]
    X_val = X_all[val_idx]
    X_test = X_all[test_idx]
    y_train = y_all[train_idx]
    y_val = y_all[val_idx]
    y_test = y_all[test_idx]
    
    X_train_full = np.concatenate([X_train, X_val], axis=0)
    y_train_full = np.concatenate([y_train, y_val], axis=0)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)
    
    svm = SVC(
        kernel=SVM_KERNEL,
        C=SVM_C,
        gamma=SVM_GAMMA,
        class_weight=None,
        probability=True,
        random_state=RANDOM_STATE + fold_idx,
    )
    svm.fit(X_train_scaled, y_train_full)
    
    y_pred = svm.predict(X_test_scaled)
    y_pred_proba = svm.predict_proba(X_test_scaled)[:, 1]
    
    return compute_metrics(y_test, y_pred, y_pred_proba)


def scenario_4_wrong_kernel(X_train_full, y_train_full, X_test, y_test, fold_idx):
    """Scenarij 4: Krivi kernel (linear umjesto RBF)."""
    print(f"    Scenarij 4: Linear kernel umjesto RBF")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)
    
    svm = SVC(
        kernel="linear",  # Krivi kernel
        C=SVM_C,
        class_weight=None,
        probability=True,
        random_state=RANDOM_STATE + fold_idx,
    )
    svm.fit(X_train_scaled, y_train_full)
    
    y_pred = svm.predict(X_test_scaled)
    y_pred_proba = svm.predict_proba(X_test_scaled)[:, 1]
    
    return compute_metrics(y_test, y_pred, y_pred_proba)


def scenario_5_shuffled_labels(X_train_full, y_train_full, X_test, y_test, fold_idx):
    """Scenarij 5: Shuffle labele (testiranje da model uči nasumične obrasce)."""
    print(f"    Scenarij 5: Shuffle labele")
    
    # Shuffle labele u training skupu
    np.random.seed(RANDOM_STATE + fold_idx)
    y_train_shuffled = y_train_full.copy()
    np.random.shuffle(y_train_shuffled)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)
    
    svm = SVC(
        kernel=SVM_KERNEL,
        C=SVM_C,
        gamma=SVM_GAMMA,
        class_weight=None,
        probability=True,
        random_state=RANDOM_STATE + fold_idx,
    )
    svm.fit(X_train_scaled, y_train_shuffled)  # Treniraj s shuffle-anim labelama
    
    y_pred = svm.predict(X_test_scaled)
    y_pred_proba = svm.predict_proba(X_test_scaled)[:, 1]
    
    return compute_metrics(y_test, y_pred, y_pred_proba)


def scenario_6_less_training_data(X_train, X_val, y_train, y_val, X_test, y_test, fold_idx):
    """Scenarij 6: Manje podataka za trening (samo 50% train skupa)."""
    print(f"    Scenarij 6: Manje podataka za trening (50%)")
    
    # Koristi samo 50% train skupa
    n_train_half = len(X_train) // 2
    X_train_half = X_train[:n_train_half]
    y_train_half = y_train[:n_train_half]
    
    # Ne koristi val skup
    X_train_full = X_train_half
    y_train_full = y_train_half
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)
    
    svm = SVC(
        kernel=SVM_KERNEL,
        C=SVM_C,
        gamma=SVM_GAMMA,
        class_weight=None,
        probability=True,
        random_state=RANDOM_STATE + fold_idx,
    )
    svm.fit(X_train_scaled, y_train_full)
    
    y_pred = svm.predict(X_test_scaled)
    y_pred_proba = svm.predict_proba(X_test_scaled)[:, 1]
    
    return compute_metrics(y_test, y_pred, y_pred_proba)


# ========================== GLAVNI TOK ==========================

def main():
    print("=" * 70)
    print("  DPC + SVM - Test degradacije performansi")
    print("=" * 70)
    print("\n  Cilj: Provjeriti kako različite degradacije utječu na performanse")
    print("  (bez mijenjanja osnovnih hiperparametara C=5, gamma=0.001)")
    
    # Učitaj podatke
    print(f"\n{'-' * 70}")
    print(f"  Ucitavanje podataka: {ALIGNED_DATA_FILE}")
    
    if not os.path.exists(ALIGNED_DATA_FILE):
        print(f"  GRESKA: Datoteka '{ALIGNED_DATA_FILE}' nije pronadjena!")
        return
    
    with open(ALIGNED_DATA_FILE, "rb") as f:
        aligned_data = pickle.load(f)
    
    fasta_sequences = aligned_data["fasta_sequences"]
    labels = aligned_data["labels"]
    cv_splits = aligned_data["cv_splits"]
    n_folds = aligned_data["n_folds"]
    
    print(f"  Ucitano sekvenci: {len(fasta_sequences)}")
    print(f"  Broj foldova: {n_folds}")
    
    # Izračunaj DPC za sve sekvence (normalizirano)
    print(f"\n{'-' * 70}")
    print("  Racunanje DPC znacajki (normalizirano)...")
    X_all = np.array([compute_dpc(seq, normalize=True) for seq in fasta_sequences])
    y_all = labels
    
    # Rezultati po scenarijima
    scenario_names = [
        "1. Bez StandardScaler-a",
        "2. Shuffle DPC značajki",
        "3. Ne normalizirane DPC",
        "4. Linear kernel",
        "5. Shuffle labele",
        "6. Manje podataka (50%)",
    ]
    
    all_results = {name: [] for name in scenario_names}
    
    # Testiraj sve scenarije na svim foldovima
    print(f"\n{'-' * 70}")
    print(f"  Testiranje {len(scenario_names)} scenarija na {n_folds} foldova...")
    print(f"{'-' * 70}")
    
    for fold_idx, split in enumerate(cv_splits, 1):
        print(f"\n  Fold {fold_idx}/{n_folds}:")
        
        train_idx = split["train_idx"]
        val_idx = split["val_idx"]
        test_idx = split["test_idx"]
        
        X_train = X_all[train_idx]
        X_val = X_all[val_idx]
        X_test = X_all[test_idx]
        y_train = y_all[train_idx]
        y_val = y_all[val_idx]
        y_test = y_all[test_idx]
        
        X_train_full = np.concatenate([X_train, X_val], axis=0)
        y_train_full = np.concatenate([y_train, y_val], axis=0)
        
        # Scenarij 1: Bez StandardScaler-a
        metrics_1 = scenario_1_no_scaler(X_train_full, y_train_full, X_test, y_test, fold_idx)
        all_results[scenario_names[0]].append(metrics_1)
        
        # Scenarij 2: Shuffle DPC
        metrics_2 = scenario_2_shuffled_dpc(X_all, train_idx, val_idx, test_idx, y_all, fold_idx)
        all_results[scenario_names[1]].append(metrics_2)
        
        # Scenarij 3: Ne normalizirane DPC
        metrics_3 = scenario_3_no_normalization(fasta_sequences, train_idx, val_idx, test_idx, y_all, fold_idx)
        all_results[scenario_names[2]].append(metrics_3)
        
        # Scenarij 4: Linear kernel
        metrics_4 = scenario_4_wrong_kernel(X_train_full, y_train_full, X_test, y_test, fold_idx)
        all_results[scenario_names[3]].append(metrics_4)
        
        # Scenarij 5: Shuffle labele
        metrics_5 = scenario_5_shuffled_labels(X_train_full, y_train_full, X_test, y_test, fold_idx)
        all_results[scenario_names[4]].append(metrics_5)
        
        # Scenarij 6: Manje podataka
        metrics_6 = scenario_6_less_training_data(X_train, X_val, y_train, y_val, X_test, y_test, fold_idx)
        all_results[scenario_names[5]].append(metrics_6)
        
        # Prikaži brze rezultate
        print(f"      MCC: {metrics_1['MCC']:.3f} | {metrics_2['MCC']:.3f} | {metrics_3['MCC']:.3f} | "
              f"{metrics_4['MCC']:.3f} | {metrics_5['MCC']:.3f} | {metrics_6['MCC']:.3f}")
    
    # Sumarni rezultati
    print(f"\n{'=' * 70}")
    print("  SUMARNI REZULTATI (prosjek preko svih foldova)")
    print(f"{'=' * 70}")
    
    metric_names = ["Accuracy", "AUC", "MCC", "Precision", "Recall", "F1", "GM"]
    
    summary_data = []
    for scenario_name in scenario_names:
        metrics_list = all_results[scenario_name]
        row = {"Scenarij": scenario_name}
        for metric_name in metric_names:
            values = [m[metric_name] for m in metrics_list]
            mean_val = np.mean(values)
            std_val = np.std(values)
            row[metric_name] = f"{mean_val:.4f} ± {std_val:.4f}"
        summary_data.append(row)
    
    # Ispiši tablicu
    print(f"\n{'Scenarij':<30} {'Accuracy':<15} {'AUC':<15} {'MCC':<15} {'F1':<15}")
    print("-" * 90)
    for row in summary_data:
        print(f"{row['Scenarij']:<30} {row['Accuracy']:<15} {row['AUC']:<15} {row['MCC']:<15} {row['F1']:<15}")
    
    # Spremi u Excel
    results_file = os.path.join(SCRIPT_DIR, "DPC_SVM_degradation_results.xlsx")
    
    detailed_data = []
    for scenario_name in scenario_names:
        for fold_idx, metrics in enumerate(all_results[scenario_name], 1):
            row = {"Scenarij": scenario_name, "Fold": fold_idx}
            row.update(metrics)
            detailed_data.append(row)
    
    df = pd.DataFrame(detailed_data)
    df.to_excel(results_file, index=False)
    
    print(f"\n  Detaljni rezultati spremljeni u: {results_file}")
    print("\n[DONE] Test degradacije zavrsen.")


if __name__ == "__main__":
    main()
