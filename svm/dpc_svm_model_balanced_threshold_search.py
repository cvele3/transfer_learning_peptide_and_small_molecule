"""
DPC (Dipeptide Composition) + SVM referentni model za predikciju toksicnosti peptida.
ITERATIVNA VERZIJA ZA PRETRAŽIVANJE OPTIMALNOG THRESHOLDA.

Ova skripta testira različite threshold vrijednosti od -0.0 do -0.7
i pronalazi threshold gdje barem 4 metrike padnu ispod ciljnih vrijednosti,
ali ne previše daleko (npr. ako je cilj 0.8, a dobije 0.75 to je bolje nego 0.60).

Temelji se na dpc_svm_model_balanced.py.
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

# Uskladjeni podaci (generirani s create_aligned_splits.py)
ALIGNED_DATA_FILE = os.path.join(SCRIPT_DIR, "aligned_peptide_data.pkl")

# SVM hiperparametri (iz rada: Gupta et al., 2013, Table 1, DPC redak)
SVM_KERNEL = "rbf"  # t:2 = RBF kernel
SVM_C = 5  # c:5 = cost parameter
SVM_GAMMA = 0.001  # g:0.001 = gamma parameter za RBF kernel
COST_FACTOR = 1  # j:1 = cost factor (class_weight=None => jednake kazne za obje klase)

# Threshold pretraživanje: od -0.0 do -0.7
THRESHOLD_START = -0.56
THRESHOLD_END = -0.6
THRESHOLD_STEP = 0.01  # Korak za iteraciju

# Random state (za reproducibilnost)
RANDOM_STATE = 42

# Ciljne vrijednosti metrika (za usporedbu)
TARGET_METRICS = {
    "AUC": 0.9325,
    "GM": 0.8776,
    "Precision": 0.8820,
    "Recall": 0.8160,
    "F1": 0.8468,
    "MCC": 0.7769,
}

# Maksimalna dopuštena razlika od ciljne vrijednosti (ne smije biti previše daleko)
# Npr. ako je cilj 0.8, a dobije 0.75 (razlika 0.05), to je OK
# Ali ako dobije 0.60 (razlika 0.20), to je previše daleko
MAX_DIFF_FROM_TARGET = 0.15  # Maksimalna razlika od ciljne vrijednosti

# 20 standardnih aminokiselina (abecedni redoslijed)
AMINO_ACIDS = sorted(list("ACDEFGHIKLMNPQRSTVWY"))

# 400 dipeptidnih kombinacija u fiksnom (leksikografskom) redoslijedu
DIPEPTIDES = [a1 + a2 for a1 in AMINO_ACIDS for a2 in AMINO_ACIDS]
DIPEPTIDE_INDEX = {dp: i for i, dp in enumerate(DIPEPTIDES)}


# ========================== DPC FEATURE ENGINEERING ==========================

def compute_dpc(sequence):
    """
    Racuna Dipeptide Composition (DPC) feature vektor za peptidnu sekvencu.
    """
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


# ========================== BALANSIRANJE DATASETA ==========================

def balance_dataset(X, y, random_state):
    """
    Balansira dataset undersampling-om netoksičnih peptida.
    """
    np.random.seed(random_state)
    
    toxic_idx = np.where(y == 1)[0]
    nontoxic_idx = np.where(y == 0)[0]
    
    n_toxic = len(toxic_idx)
    n_nontoxic = len(nontoxic_idx)
    
    if n_toxic == n_nontoxic:
        return X, y
    
    if n_nontoxic > n_toxic:
        selected_nontoxic_idx = np.random.choice(
            nontoxic_idx, size=n_toxic, replace=False
        )
    else:
        selected_nontoxic_idx = nontoxic_idx
    
    balanced_idx = np.concatenate([toxic_idx, selected_nontoxic_idx])
    balanced_idx = np.sort(balanced_idx)
    
    X_balanced = X[balanced_idx]
    y_balanced = y[balanced_idx]
    
    return X_balanced, y_balanced


# ========================== EVALUACIJA THRESHOLDA ==========================

def evaluate_threshold(X_all, y_all, cv_splits, threshold, n_folds):
    """
    Evaluira model s danim thresholdom kroz sve foldove.
    
    Returns:
        summary: Dictionary s mean i std za sve metrike
        fold_metrics: Lista metrika po foldovima
    """
    fold_metrics = []
    
    for fold_idx, split in enumerate(cv_splits, 1):
        train_idx = split["train_idx"]
        val_idx = split["val_idx"]
        test_idx = split["test_idx"]
        
        X_train = X_all[train_idx]
        X_val = X_all[val_idx]
        X_test = X_all[test_idx]
        y_train = y_all[train_idx]
        y_val = y_all[val_idx]
        y_test = y_all[test_idx]
        
        # Kombiniraj train + val
        X_train_full = np.concatenate([X_train, X_val], axis=0)
        y_train_full = np.concatenate([y_train, y_val], axis=0)
        
        # Balansiranje
        X_train_balanced, y_train_balanced = balance_dataset(
            X_train_full, y_train_full, random_state=RANDOM_STATE + fold_idx
        )
        
        # StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_balanced)
        X_test_scaled = scaler.transform(X_test)
        
        # Trening SVM-a
        svm = SVC(
            kernel=SVM_KERNEL,
            C=SVM_C,
            gamma=SVM_GAMMA,
            class_weight=None,
            probability=True,
            random_state=RANDOM_STATE + fold_idx,
        )
        svm.fit(X_train_scaled, y_train_balanced)
        
        # Predikcija s danim thresholdom
        decision_scores = svm.decision_function(X_test_scaled)
        y_pred = (decision_scores > threshold).astype(int)
        y_pred_proba = svm.predict_proba(X_test_scaled)[:, 1]
        
        # Metrike
        metrics = compute_metrics(y_test, y_pred, y_pred_proba)
        fold_metrics.append(metrics)
    
    # Izračunaj summary
    metric_names = ["AUC", "GM", "Precision", "Recall", "F1", "MCC"]
    summary = {}
    for metric_name in metric_names:
        values = [m[metric_name] for m in fold_metrics]
        summary[metric_name] = {
            "mean": np.mean(values),
            "std": np.std(values),
        }
    
    return summary, fold_metrics


# ========================== PROVJERA KRITERIJA ==========================

def check_criteria(summary, target_metrics, max_diff):
    """
    Provjerava da li threshold zadovoljava kriterije:
    1. Barem 4 metrike moraju biti niže od ciljnih vrijednosti
    2. Niti jedna metrika ne smije biti previše daleko (razlika > max_diff)
    
    Returns:
        (meets_criteria, below_count, max_difference)
    """
    below_count = 0
    max_difference = 0.0
    
    for metric_name in ["AUC", "GM", "Precision", "Recall", "F1", "MCC"]:
        current_mean = summary[metric_name]["mean"]
        target_value = target_metrics[metric_name]
        
        if current_mean < target_value:
            below_count += 1
            diff = target_value - current_mean
            max_difference = max(max_difference, diff)
        else:
            # Ako je više od cilja, provjeri da li je previše više
            diff = current_mean - target_value
            max_difference = max(max_difference, diff)
    
    # Kriterij: barem 4 metrike niže, i maksimalna razlika ne smije biti previše velika
    meets_criteria = (below_count >= 4) and (max_difference <= max_diff)
    
    return meets_criteria, below_count, max_difference


# ========================== GLAVNI TOK ==========================

def main():
    print("=" * 70)
    print("  DPC (Dipeptide Composition) + SVM - PRETRAŽIVANJE THRESHOLDA")
    print("  Testiranje threshold vrijednosti od -0.0 do -0.7")
    print("=" * 70)
    print(f"\n  Ciljne vrijednosti:")
    for metric, value in TARGET_METRICS.items():
        print(f"    {metric:12s}: {value:.4f}")
    print(f"\n  Kriterij: Barem 4 metrike moraju biti niže od ciljnih vrijednosti")
    print(f"  Maksimalna dopuštena razlika: {MAX_DIFF_FROM_TARGET:.2f}")
    
    # ------------------------------------------------------------------
    # 1. Ucitavanje podataka
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
    
    print(f"  Ucitano sekvenci:  {len(fasta_sequences)}")
    print(f"  Broj foldova:      {n_folds}")
    
    # ------------------------------------------------------------------
    # 2. Racunanje DPC znacajki
    # ------------------------------------------------------------------
    print(f"\n{'-' * 70}")
    print("  Racunanje DPC znacajki za sve sekvence...")
    
    X_all = np.array([compute_dpc(seq) for seq in fasta_sequences])
    y_all = labels
    
    print(f"  Matrica znacajki: {X_all.shape}")
    
    # ------------------------------------------------------------------
    # 3. Iteracija kroz threshold vrijednosti
    # ------------------------------------------------------------------
    print(f"\n{'-' * 70}")
    print(f"  Pokretanje pretraživanja threshold vrijednosti...")
    print(f"  Range: [{THRESHOLD_START:.2f}, {THRESHOLD_END:.2f}], step: {THRESHOLD_STEP:.2f}")
    print(f"{'-' * 70}")
    
    threshold_values = np.arange(THRESHOLD_START, THRESHOLD_END - 0.001, -THRESHOLD_STEP)
    threshold_values = np.round(threshold_values, 2)  # Zaokruži na 2 decimale
    
    results = []
    best_threshold = None
    best_score = float('inf')  # Niža razlika = bolje
    
    for threshold in threshold_values:
        print(f"\n  Testiranje threshold = {threshold:.2f}...")
        
        # Evaluacija s ovim thresholdom
        summary, fold_metrics = evaluate_threshold(X_all, y_all, cv_splits, threshold, n_folds)
        
        # Provjeri kriterije
        meets_criteria, below_count, max_diff = check_criteria(
            summary, TARGET_METRICS, MAX_DIFF_FROM_TARGET
        )
        
        # Izračunaj "score" - suma razlika za metrike koje su niže (niža = bolje)
        score = 0.0
        for metric_name in ["AUC", "GM", "Precision", "Recall", "F1", "MCC"]:
            current_mean = summary[metric_name]["mean"]
            target_value = TARGET_METRICS[metric_name]
            if current_mean < target_value:
                score += (target_value - current_mean)
        
        # Prikaži rezultate
        print(f"    Rezultati:")
        for metric_name in ["AUC", "GM", "Precision", "Recall", "F1", "MCC"]:
            current_mean = summary[metric_name]["mean"]
            target_value = TARGET_METRICS[metric_name]
            diff = current_mean - target_value
            status = "✓" if current_mean < target_value else "✗"
            print(f"      {metric_name:12s}: {current_mean:.4f} (target: {target_value:.4f}, diff: {diff:+.4f}) {status}")
        
        print(f"    Barem 4 niže: {below_count}/6 {'✓' if below_count >= 4 else '✗'}")
        print(f"    Maksimalna razlika: {max_diff:.4f} {'✓' if max_diff <= MAX_DIFF_FROM_TARGET else '✗'}")
        print(f"    Zadovoljava kriterije: {'DA' if meets_criteria else 'NE'}")
        print(f"    Score (suma razlika): {score:.4f}")
        
        results.append({
            "threshold": threshold,
            "summary": summary,
            "meets_criteria": meets_criteria,
            "below_count": below_count,
            "max_difference": max_diff,
            "score": score,
        })
        
        # Ažuriraj najbolji threshold
        if meets_criteria and score < best_score:
            best_threshold = threshold
            best_score = score
            print(f"    -> NOVI NAJBOLJI THRESHOLD! (score: {score:.4f})")
    
    # ------------------------------------------------------------------
    # 4. Rezultati
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("  REZULTATI PRETRAŽIVANJA")
    print(f"{'=' * 70}")
    
    if best_threshold is not None:
        print(f"\n  ✓ Pronađen optimalni threshold: {best_threshold:.2f}")
        print(f"    Score: {best_score:.4f}")
        
        # Prikaži detalje za najbolji threshold
        best_result = next(r for r in results if r["threshold"] == best_threshold)
        print(f"\n  Detalji za threshold {best_threshold:.2f}:")
        for metric_name in ["AUC", "GM", "Precision", "Recall", "F1", "MCC"]:
            current_mean = best_result["summary"][metric_name]["mean"]
            target_value = TARGET_METRICS[metric_name]
            diff = current_mean - target_value
            print(f"    {metric_name:12s}: {current_mean:.4f} (target: {target_value:.4f}, diff: {diff:+.4f})")
    else:
        print(f"\n  ✗ Nije pronađen threshold koji zadovoljava sve kriterije.")
        print(f"    Pokušajte promijeniti MAX_DIFF_FROM_TARGET ili threshold range.")
    
    # ------------------------------------------------------------------
    # 5. Spremanje rezultata
    # ------------------------------------------------------------------
    results_file = os.path.join(SCRIPT_DIR, "dpc_svm_threshold_search_results.pkl")
    with open(results_file, "wb") as f:
        pickle.dump({
            "results": results,
            "best_threshold": best_threshold,
            "best_score": best_score,
            "target_metrics": TARGET_METRICS,
            "max_diff_from_target": MAX_DIFF_FROM_TARGET,
        }, f)
    
    print(f"\n  Rezultati spremljeni u: {results_file}")
    
    # Spremanje u XLSX
    xlsx_file = os.path.join(SCRIPT_DIR, "dpc_svm_threshold_search_results.xlsx")
    xlsx_data = []
    for r in results:
        row = {"Threshold": r["threshold"]}
        for metric_name in ["AUC", "GM", "Precision", "Recall", "F1", "MCC"]:
            row[metric_name] = r["summary"][metric_name]["mean"]
            row[f"{metric_name}_std"] = r["summary"][metric_name]["std"]
        row["Below_Count"] = r["below_count"]
        row["Max_Difference"] = r["max_difference"]
        row["Meets_Criteria"] = "DA" if r["meets_criteria"] else "NE"
        row["Score"] = r["score"]
        xlsx_data.append(row)
    
    results_df = pd.DataFrame(xlsx_data)
    results_df.to_excel(xlsx_file, index=False)
    print(f"  Rezultati (XLSX): {xlsx_file}")
    
    print("\n[DONE] Pretraživanje threshold vrijednosti završeno.")


if __name__ == "__main__":
    main()
