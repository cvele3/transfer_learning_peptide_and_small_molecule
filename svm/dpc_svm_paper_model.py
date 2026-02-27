"""
DPC + SVM model -- verzija uskladjena s ToxinPred radom (Gupta et al., 2013).

Cilj: Sto vjernije replicirati DPC SVM model iz rada, uz prilagodbu
      na 10-fold CV radi konzistentnosti s GNN modelima.

Postavke iz rada:
  - DPC feature vektor dimenzije 400 (20x20 dipeptida)
  - SVM s RBF jezgrom
  - Cost factor = 1 (class_weight=None)
  - Hiperparametri (C, gamma) optimizirani na validacijskom skupu
    (SVMlight koristi internu optimizaciju parametara)

Prilagodbe:
  - 10-fold CV (umjesto 5-fold) radi konzistentnosti s GNN modelima
  - StandardScaler (potreban za scikit-learn SVC s RBF kernelom;
    SVMlight moze interno drugacije rukovati skaliranjem)
  - Grid search za C i gamma na validacijskom skupu
    (zamjena za SVMlight-ovu internu optimizaciju)

Koristi uskladjene splitove iz aligned_peptide_data.pkl
(generira ih create_aligned_splits.py).
"""

import os
import math
import time
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

# Uskladjeni podaci (generirani s create_aligned_splits.py)
ALIGNED_DATA_FILE = os.path.join(SCRIPT_DIR, "aligned_peptide_data.pkl")

# SVM fiksne postavke iz rada
SVM_KERNEL = "rbf"
COST_FACTOR = None  # class_weight=None => cost factor = 1 (kao u radu)

# Grid search raspon za C i gamma
# (SVMlight interno optimizira; mi koristimo grid search na val skupu)
C_GRID = [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
GAMMA_GRID = [0.0001, 0.001, 0.01, 0.1, "scale", "auto"]

# Metrika za odabir najboljeg modela na validacijskom skupu
SELECTION_METRIC = "MCC"

# Random state (za reproducibilnost)
RANDOM_STATE = 42

# 20 standardnih aminokiselina (abecedni redoslijed)
AMINO_ACIDS = sorted(list("ACDEFGHIKLMNPQRSTVWY"))

# 400 dipeptidnih kombinacija u fiksnom (leksikografskom) redoslijedu
DIPEPTIDES = [a1 + a2 for a1 in AMINO_ACIDS for a2 in AMINO_ACIDS]
DIPEPTIDE_INDEX = {dp: i for i, dp in enumerate(DIPEPTIDES)}


# ========================== DPC FEATURE ENGINEERING ==========================

def compute_dpc(sequence):
    """
    Racuna Dipeptide Composition (DPC) feature vektor za peptidnu sekvencu.

    Za sekvencu duljine L:
      - Ukupan broj dipeptida = L - 1
      - DPC(i) = count(dipeptide_i) / (L - 1)

    Vraca 400-dimenzionalni normalizirani vektor frekvencija.
    """
    sequence = sequence.upper().strip()
    L = len(sequence)

    if L < 2:
        return np.zeros(400)

    counts = np.zeros(400)
    total_dipeptides = L - 1

    for i in range(total_dipeptides):
        dp = sequence[i : i + 2]
        if dp in DIPEPTIDE_INDEX:
            counts[DIPEPTIDE_INDEX[dp]] += 1

    # Normalizacija
    if total_dipeptides > 0:
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
    sensitivity = tp / (tp + fn + 1e-8)  # TPR
    specificity = tn / (tn + fp + 1e-8)  # TNR
    gm = math.sqrt(sensitivity * specificity)  # Geometrijska sredina

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


# ========================== GRID SEARCH ==========================


def grid_search_on_val(X_train, y_train, X_val, y_val):
    """
    Pretrazuje mreze hiperparametara (C, gamma) na validacijskom skupu.
    Odabire kombinaciju s najboljim MCC-om (SELECTION_METRIC).

    Ovo zamjenjuje SVMlight-ovu internu optimizaciju parametara.
    Koristi class_weight=None (cost factor = 1, kao u radu).

    Vraca: (best_C, best_gamma, best_score, results_log)
    """
    best_score = -999.0
    best_C = None
    best_gamma = None
    results_log = []

    for c_val in C_GRID:
        for g_val in GAMMA_GRID:
            svm = SVC(
                kernel=SVM_KERNEL,
                C=c_val,
                gamma=g_val,
                class_weight=COST_FACTOR,  # None => cost factor = 1
                probability=False,  # brze bez probability za grid search
                random_state=RANDOM_STATE,
            )
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_val)

            mcc = matthews_corrcoef(y_val, y_pred)
            acc = accuracy_score(y_val, y_pred)

            results_log.append(
                {"C": c_val, "gamma": g_val, "MCC": mcc, "Accuracy": acc}
            )

            if mcc > best_score:
                best_score = mcc
                best_C = c_val
                best_gamma = g_val

    return best_C, best_gamma, best_score, results_log


# ========================== GLAVNI TOK ==========================


def main():
    print("=" * 70)
    print("  DPC + SVM model (verzija uskladjena s ToxinPred radom)")
    print("=" * 70)
    print(f"\n  Postavke iz rada (Gupta et al., 2013):")
    print(f"    Kernel:             {SVM_KERNEL}")
    print(f"    Cost factor:        1 (class_weight=None)")
    print(f"    Feature scaling:    StandardScaler (per fold)")
    print(f"    Dimenzija znacajki: 400 (20x20 dipeptida)")
    print(f"    Odabir parametara:  Grid search na val skupu (C, gamma)")
    print(f"    C grid:             {C_GRID}")
    print(f"    Gamma grid:         {GAMMA_GRID}")
    print(f"    Selekcijska metrika: {SELECTION_METRIC}")

    # ------------------------------------------------------------------
    # 1. Ucitavanje uskladjenih podataka
    # ------------------------------------------------------------------
    print(f"\n{'-' * 70}")
    print(f"  Ucitavanje uskladjenih podataka: {ALIGNED_DATA_FILE}")

    if not os.path.exists(ALIGNED_DATA_FILE):
        print(f"\n  GRESKA: Datoteka '{ALIGNED_DATA_FILE}' nije pronadjena!")
        print("  Pokrenite najprije create_aligned_splits.py")
        return

    with open(ALIGNED_DATA_FILE, "rb") as f:
        aligned_data = pickle.load(f)

    fasta_sequences = aligned_data["fasta_sequences"]
    labels = aligned_data["labels"]
    cv_splits = aligned_data["cv_splits"]
    n_folds = aligned_data["n_folds"]

    n_toxic = (labels == 1).sum()
    n_nontoxic = (labels == 0).sum()

    print(f"  Ucitano sekvenci:  {len(fasta_sequences)}")
    print(f"    Toksicni  (1):   {n_toxic}")
    print(f"    Netoksicni (0):  {n_nontoxic}")
    print(f"    Omjer:           1:{n_nontoxic / n_toxic:.2f}")
    print(f"  Broj foldova:      {n_folds}")

    # ------------------------------------------------------------------
    # 2. Racunanje DPC znacajki za SVE sekvence
    # ------------------------------------------------------------------
    print(f"\n{'-' * 70}")
    print("  Racunanje DPC znacajki za sve sekvence...")

    X_all = np.array([compute_dpc(seq) for seq in fasta_sequences])
    y_all = labels

    print(f"  Matrica znacajki: {X_all.shape}")
    print(f"  Vektor oznaka:    {y_all.shape}")

    # ------------------------------------------------------------------
    # 3. 10-fold unakrsna validacija (isti foldovi kao GNN)
    # ------------------------------------------------------------------
    print(f"\n{'-' * 70}")
    print(f"  Pokretanje {n_folds}-fold unakrsne validacije (isti foldovi kao GNN)...")
    print(f"  Za svaki fold: grid search na val skupu -> trening na train+val -> test")
    print(f"{'-' * 70}")

    fold_metrics = []
    fold_best_params = []
    models_dir = os.path.join(SCRIPT_DIR, "models_paper")
    os.makedirs(models_dir, exist_ok=True)

    total_start = time.time()

    for fold_idx, split in enumerate(cv_splits, 1):
        fold_start = time.time()

        train_idx = split["train_idx"]
        val_idx = split["val_idx"]
        test_idx = split["test_idx"]

        X_train = X_all[train_idx]
        X_val = X_all[val_idx]
        X_test = X_all[test_idx]
        y_train = y_all[train_idx]
        y_val = y_all[val_idx]
        y_test = y_all[test_idx]

        # StandardScaler - fit na train, transform na train, val i test
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # ----- Grid search na validacijskom skupu -----
        best_C, best_gamma, best_val_score, gs_log = grid_search_on_val(
            X_train_scaled, y_train, X_val_scaled, y_val
        )

        fold_best_params.append(
            {"fold": fold_idx, "C": best_C, "gamma": best_gamma, "val_MCC": best_val_score}
        )

        # ----- Finalni trening na train + val s najboljim parametrima -----
        X_train_full = np.concatenate([X_train_scaled, X_val_scaled], axis=0)
        y_train_full = np.concatenate([y_train, y_val], axis=0)

        svm_final = SVC(
            kernel=SVM_KERNEL,
            C=best_C,
            gamma=best_gamma,
            class_weight=COST_FACTOR,  # None => cost factor = 1
            probability=True,
            random_state=RANDOM_STATE,
        )
        svm_final.fit(X_train_full, y_train_full)

        # ----- Predikcija na test skupu -----
        y_pred = svm_final.predict(X_test_scaled)
        y_pred_proba = svm_final.predict_proba(X_test_scaled)[:, 1]

        # Metrike
        metrics = compute_metrics(y_test, y_pred, y_pred_proba)
        fold_metrics.append(metrics)

        # Spremi model + scaler + parametre za ovaj fold
        model_path = os.path.join(models_dir, f"dpc_svm_paper_fold_{fold_idx}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "model": svm_final,
                    "scaler": scaler,
                    "best_C": best_C,
                    "best_gamma": best_gamma,
                },
                f,
            )

        fold_time = time.time() - fold_start
        print(
            f"  Fold {fold_idx:2d}:  "
            f"C={best_C:<6}  gamma={str(best_gamma):<8}  "
            f"Acc={metrics['Accuracy']:.4f}  "
            f"AUC={metrics['AUC']:.4f}  "
            f"MCC={metrics['MCC']:.4f}  "
            f"F1={metrics['F1']:.4f}  "
            f"GM={metrics['GM']:.4f}  "
            f"({fold_time:.1f}s)"
        )

    total_time = time.time() - total_start
    print(f"\n  Ukupno vrijeme: {total_time:.1f}s")

    # ------------------------------------------------------------------
    # 4. Odabrani hiperparametri po foldovima
    # ------------------------------------------------------------------
    print(f"\n{'-' * 70}")
    print("  Odabrani hiperparametri po foldovima (grid search na val skupu):")
    print(f"{'-' * 70}")

    for p in fold_best_params:
        print(
            f"    Fold {p['fold']:2d}:  C={p['C']:<6}  "
            f"gamma={str(p['gamma']):<8}  val_MCC={p['val_MCC']:.4f}"
        )

    # ------------------------------------------------------------------
    # 5. Sumarni rezultati
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("  SUMARNI REZULTATI UNAKRSNE VALIDACIJE")
    print(f"  (DPC + SVM, cost factor = 1, grid search na val skupu)")
    print(f"{'=' * 70}")

    metric_names = [
        "Accuracy",
        "AUC",
        "MCC",
        "Precision",
        "Recall",
        "F1",
        "GM",
        "Sensitivity",
        "Specificity",
    ]

    summary = {}
    for metric_name in metric_names:
        values = [m[metric_name] for m in fold_metrics]
        mean_val = np.mean(values)
        std_val = np.std(values)
        summary[metric_name] = {"mean": mean_val, "std": std_val}
        print(f"  {metric_name:15s}:  {mean_val:.4f} +/- {std_val:.4f}")

    # ------------------------------------------------------------------
    # 6. Usporedba s radom
    # ------------------------------------------------------------------
    print(f"\n{'-' * 70}")
    print("  Usporedba s rezultatima iz rada (Gupta et al., 2013):")
    print(f"{'-' * 70}")
    print(f"    Rad (5-fold, nebalansirani):  Acc=0.9450  MCC=0.8800")
    print(f"    Rad (5-fold, balansirani):    Acc=0.9388  MCC=0.8800")
    print(
        f"    Nasa impl. ({n_folds}-fold):          "
        f"Acc={summary['Accuracy']['mean']:.4f}  "
        f"MCC={summary['MCC']['mean']:.4f}"
    )

    # ------------------------------------------------------------------
    # 7. Spremanje rezultata u XLSX
    # ------------------------------------------------------------------
    results_file = os.path.join(SCRIPT_DIR, "DPC_SVM_paper_results.xlsx")

    results_data = []
    for fold_idx, metrics in enumerate(fold_metrics, 1):
        row = {"Fold": fold_idx}
        row.update(metrics)
        # Dodaj parametre
        row["Best_C"] = fold_best_params[fold_idx - 1]["C"]
        row["Best_gamma"] = str(fold_best_params[fold_idx - 1]["gamma"])
        results_data.append(row)

    # Srednja vrijednost i standardna devijacija
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

    # ------------------------------------------------------------------
    # 8. Spremanje rezultata po foldovima u pickle (za evaluacijske skripte)
    # ------------------------------------------------------------------
    fold_results_file = os.path.join(SCRIPT_DIR, "dpc_svm_paper_fold_results.pkl")

    fold_results_data = {
        "fold_metrics": fold_metrics,
        "metric_names": metric_names,
        "summary": summary,
        "n_folds": n_folds,
        "fold_best_params": fold_best_params,
        "svm_params": {
            "kernel": SVM_KERNEL,
            "class_weight": COST_FACTOR,
            "C_grid": C_GRID,
            "gamma_grid": GAMMA_GRID,
            "selection_metric": SELECTION_METRIC,
        },
    }
    with open(fold_results_file, "wb") as f:
        pickle.dump(fold_results_data, f)

    print(f"  Rezultati po foldovima: {fold_results_file}")
    print(f"\n  Spremljeno {n_folds} SVM modela u: {models_dir}/")
    print(f"    dpc_svm_paper_fold_1.pkl ... dpc_svm_paper_fold_{n_folds}.pkl")

    print("\n[DONE] DPC + SVM (paper verzija) evaluacija zavrsena.")


if __name__ == "__main__":
    main()
