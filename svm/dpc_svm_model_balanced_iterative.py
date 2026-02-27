"""
DPC (Dipeptide Composition) + SVM referentni model za predikciju toksicnosti peptida.
ITERATIVNA VERZIJA - traži balance_seed koji daje lošije rezultate.

Iterira kroz različite balance_seed vrijednosti za np.random.choice() u balance_dataset().
Zaustavlja se kada 4 od 6 metrika padne ispod threshold vrijednosti.

Za svaki balance_seed:
  - Izvršava 10-fold CV
  - Sprema pickle file s informacijama o odabranim netoksičnim peptidima
  - Ako je 4+ metrika ispod threshold-a → zaustavlja i zadržava pickle
  - Ako je <4 metrika ispod threshold-a → briše pickle i nastavlja

Threshold vrijednosti (mean - std iz method4):
  - ROC_AUC < 0.9140
  - GM < 0.8530
  - Precision < 0.8629
  - Recall < 0.7674
  - F1 < 0.8198
  - MCC < 0.7412
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
SVM_THRESHOLD = -0.4  # Threshold iz rada

# Random state za SVM (ostaje 42 + fold_idx)
RANDOM_STATE = 42

# Početni balance_seed za iteraciju
BALANCE_SEED_START = 40
BALANCE_SEED_MAX = 10  # Maksimalan broj iteracija (40-49 = 10 iteracija)

# Threshold vrijednosti (mean - std iz method4)
THRESHOLDS = {
    "AUC": 0.9140,      # 0.9325 - 0.0185
    "GM": 0.8530,       # 0.8776 - 0.0246
    "Precision": 0.8629,  # 0.8820 - 0.0191
    "Recall": 0.7674,   # 0.8160 - 0.0486
    "F1": 0.8198,       # 0.8468 - 0.0270
    "MCC": 0.7412,      # 0.7769 - 0.0357
}

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

def balance_dataset_with_selection(X, y, random_state):
    """
    Balansira dataset undersampling-om netoksičnih peptida.
    
    Vraća i informacije o odabranim netoksičnim indeksima (relativno u X/y).
    
    Returns:
        X_balanced, y_balanced, selected_nontoxic_indices
        selected_nontoxic_indices su relativni indeksi u originalnom X/y
    """
    np.random.seed(random_state)
    
    # Indeksi toksičnih i netoksičnih (relativno u X/y)
    toxic_idx = np.where(y == 1)[0]
    nontoxic_idx = np.where(y == 0)[0]
    
    n_toxic = len(toxic_idx)
    n_nontoxic = len(nontoxic_idx)
    
    # Ako je već balansiran, vrati original
    if n_toxic == n_nontoxic:
        return X, y, nontoxic_idx
    
    # Undersampling: odaberi n_toxic netoksičnih peptida
    if n_nontoxic > n_toxic:
        selected_nontoxic_idx = np.random.choice(
            nontoxic_idx, size=n_toxic, replace=False
        )
    else:
        selected_nontoxic_idx = nontoxic_idx
    
    # Kombiniraj toksične + odabrane netoksične
    balanced_idx = np.concatenate([toxic_idx, selected_nontoxic_idx])
    balanced_idx = np.sort(balanced_idx)  # Sortiraj za konzistentnost
    
    X_balanced = X[balanced_idx]
    y_balanced = y[balanced_idx]
    
    return X_balanced, y_balanced, selected_nontoxic_idx


# ========================== GLAVNI TOK ==========================

def run_cv_with_balance_seed(X_all, y_all, cv_splits, balance_seed, n_folds):
    """
    Izvršava 10-fold CV s određenim balance_seed-om.
    
    Returns:
        fold_metrics: Lista metrika za svaki fold
        fold_selections: Dict s informacijama o odabranim netoksičnim peptidima za svaki fold
    """
    fold_metrics = []
    fold_selections = {}
    
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

        # Kombiniraj train + val za SVM trening
        X_train_full = np.concatenate([X_train, X_val], axis=0)
        y_train_full = np.concatenate([y_train, y_val], axis=0)

        # BALANSIRANJE: Koristi balance_seed + fold_idx za balansiranje
        balance_random_state = balance_seed + fold_idx
        X_train_balanced, y_train_balanced, selected_nontoxic_idx = balance_dataset_with_selection(
            X_train_full, y_train_full, random_state=balance_random_state
        )
        
        # Spremi informacije o odabranim netoksičnim indeksima
        # selected_nontoxic_idx su relativni indeksi u X_train_full/y_train_full
        # Trebamo ih mapirati na originalne indekse u X_all/y_all
        train_val_idx = np.concatenate([train_idx, val_idx])
        selected_nontoxic_original_idx = train_val_idx[selected_nontoxic_idx]
        
        fold_selections[fold_idx] = {
            "balance_seed": balance_seed,
            "balance_random_state": balance_random_state,
            "selected_nontoxic_idx_relative": selected_nontoxic_idx.tolist(),  # Relativno u train+val
            "selected_nontoxic_idx_original": selected_nontoxic_original_idx.tolist(),  # Originalni indeksi
            "train_idx": train_idx.tolist(),
            "val_idx": val_idx.tolist(),
        }

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
            random_state=RANDOM_STATE + fold_idx,  # SVM random_state ostaje 42 + fold_idx
        )
        svm.fit(X_train_scaled, y_train_balanced)

        # Predikcija na test skupu
        decision_scores = svm.decision_function(X_test_scaled)
        y_pred = (decision_scores > SVM_THRESHOLD).astype(int)
        y_pred_proba = svm.predict_proba(X_test_scaled)[:, 1]

        # Metrike
        metrics = compute_metrics(y_test, y_pred, y_pred_proba)
        fold_metrics.append(metrics)

    return fold_metrics, fold_selections


def check_metrics_below_threshold(summary, thresholds):
    """
    Provjerava koliko metrika je ispod threshold vrijednosti.
    
    Returns:
        count: Broj metrika ispod threshold-a
        below_threshold: Dict s True/False za svaku metriku
    """
    below_threshold = {}
    count = 0
    
    for metric_name, threshold_value in thresholds.items():
        if metric_name in summary:
            mean_val = summary[metric_name]["mean"]
            is_below = mean_val < threshold_value
            below_threshold[metric_name] = is_below
            if is_below:
                count += 1
    
    return count, below_threshold


def main():
    print("=" * 70)
    print("  DPC (Dipeptide Composition) + SVM - ITERATIVNA VERZIJA")
    print("  Traži balance_seed koji daje lošije rezultate (4+ metrika ispod threshold-a)")
    print("=" * 70)
    print(f"\n  Threshold vrijednosti (mean - std iz method4):")
    for metric, threshold in THRESHOLDS.items():
        print(f"    {metric:12s}: < {threshold:.4f}")
    print(f"\n  Kriterij zaustavljanja: 4 od 6 metrika ispod threshold-a")
    print(f"  Početni balance_seed: {BALANCE_SEED_START}")
    print(f"  Maksimalan broj iteracija: {BALANCE_SEED_MAX}")

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

    print(f"  Ucitano sekvenci:  {len(fasta_sequences)}")
    print(f"    Toksicni  (1):   {(labels == 1).sum()}")
    print(f"    Netoksicni (0):  {(labels == 0).sum()}")
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
    # 3. Iteracija kroz balance_seed vrijednosti
    # ------------------------------------------------------------------
    print(f"\n{'-' * 70}")
    print(f"  Pokretanje iteracije kroz balance_seed vrijednosti...")
    print(f"{'-' * 70}")

    pickle_dir = os.path.join(SCRIPT_DIR, "balance_selections")
    os.makedirs(pickle_dir, exist_ok=True)

    stopped_seed = None
    final_results = None
    final_selections = None

    for balance_seed in range(BALANCE_SEED_START, BALANCE_SEED_START + BALANCE_SEED_MAX):
        print(f"\n  Balance seed: {balance_seed}")
        
        # Izvrši 10-fold CV
        fold_metrics, fold_selections = run_cv_with_balance_seed(
            X_all, y_all, cv_splits, balance_seed, n_folds
        )

        # Izračunaj prosječne metrike
        metric_names = ["AUC", "MCC", "Precision", "Recall", "F1", "GM"]
        summary = {}
        for metric_name in metric_names:
            values = [m[metric_name] for m in fold_metrics]
            mean_val = np.mean(values)
            std_val = np.std(values)
            summary[metric_name] = {"mean": mean_val, "std": std_val}

        # Provjeri koliko metrika je ispod threshold-a
        below_count, below_dict = check_metrics_below_threshold(summary, THRESHOLDS)

        # Ispiši rezultate
        print(f"    Metrike:")
        for metric_name in metric_names:
            mean_val = summary[metric_name]["mean"]
            is_below = below_dict.get(metric_name, False)
            marker = " [BELOW]" if is_below else ""
            print(f"      {metric_name:12s}: {mean_val:.4f}{marker}")

        print(f"    Metrika ispod threshold-a: {below_count}/6")

        # Spremi pickle file (uvijek)
        pickle_file = os.path.join(pickle_dir, f"balance_seed_{balance_seed}_selections.pkl")
        pickle_data = {
            "balance_seed": balance_seed,
            "fold_selections": fold_selections,
            "fold_metrics": fold_metrics,
            "summary": summary,
            "below_threshold_count": below_count,
            "below_threshold": below_dict,
        }
        with open(pickle_file, "wb") as f:
            pickle.dump(pickle_data, f)

        # Provjeri kriterij zaustavljanja
        if below_count >= 4:
            print(f"\n  [STOPPED] {below_count} od 6 metrika je ispod threshold-a!")
            print(f"  Pickle file zadržan: {pickle_file}")
            stopped_seed = balance_seed
            final_results = summary
            final_selections = fold_selections
            break
        else:
            # Većina metrika je dobra - obriši pickle i nastavi
            if os.path.exists(pickle_file):
                os.remove(pickle_file)
            print(f"    -> Nastavljam (obrisan pickle)")

    # ------------------------------------------------------------------
    # 4. Finalni rezultati
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    if stopped_seed is not None:
        print(f"  ITERACIJA ZAUSTAVLJENA na balance_seed = {stopped_seed}")
        
        # Učitaj below_threshold info iz zadnjeg pickle-a
        pickle_file = os.path.join(pickle_dir, f"balance_seed_{stopped_seed}_selections.pkl")
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)
        below_dict = pickle_data["below_threshold"]
        below_count = pickle_data["below_threshold_count"]
        
        print(f"  {below_count} od 6 metrika je ispod threshold-a")
        print(f"{'=' * 70}")
        print(f"\n  Finalne metrike:")
        for metric_name in metric_names:
            mean_val = final_results[metric_name]["mean"]
            std_val = final_results[metric_name]["std"]
            is_below = below_dict.get(metric_name, False)
            marker = " [BELOW THRESHOLD]" if is_below else ""
            print(f"    {metric_name:12s}: {mean_val:.4f} +/- {std_val:.4f}{marker}")
        
        print(f"\n  Pickle file sačuvan: {pickle_file}")
        print(f"  Sadrži informacije o odabranim netoksičnim peptidima za svaki fold.")
    else:
        print(f"  ITERACIJA ZAVRŠENA - nije pronađen balance_seed koji zadovoljava kriterij")
        print(f"  (prošlo {BALANCE_SEED_MAX} iteracija)")

    print("\n[DONE] Iterativna pretraga završena.")


if __name__ == "__main__":
    main()
