"""
DPC (Dipeptide Composition) + SVM referentni model za predikciju toksicnosti peptida.
VERZIJA S BALANSIRANIM DATASETOM (kao u radu).

Feature engineering:
  - Za svaku sekvencu racuna se DPC vektor dimenzije 400.
  - Klizni prozor velicine 2 broji sve susjedne dipeptidne kombinacije.
  - Normalizirana frekvencija: DPC(i) = count(i) / (L - 1)

Model:
  - SVM s RBF jezgrom
  - Parametri iz rada (Gupta et al., 2013, Table 1, DPC):
    * t:2 (RBF kernel)
    * g:0.001 (gamma)
    * c:5 (C/cost)
    * j:1 (cost factor = 1, class_weight=None)
    * Threshold: -0.4 (decision_function > -0.4 => klasa 1)
  - StandardScaler za skaliranje znacajki (fit na train+val, transform na test)
  - random_state = 42 + fold_idx (varijabilnost izmedju foldova uz reproducibilnost)
  - BALANSIRAN DATASET: Undersampling netoksičnih peptida na broj toksičnih (1:1 omjer)

Evaluacija:
  - 10-fold unakrsna validacija (isti foldovi kao GNN modeli)
  - Metrike: Accuracy, AUC, MCC, Precision, Recall, F1, GM, Sensitivity, Specificity

Koristi uskladjene splitove iz aligned_peptide_data.pkl
(generira ih create_aligned_splits.py).

Napomene:
  - random_state se postavlja na RANDOM_STATE + fold_idx za svaki fold
    radi osiguravanja varijabilnosti izmedju foldova uz zadrzavanje reproducibilnosti
  - class_weight=None (jednake kazne za obje klase) - kao u radu na balansiranom datasetu
  - StandardScaler se fit-uje samo na train+val skupu, test se samo transformira
    (osigurava da nema data leakage-a)
  - BALANSIRANJE: Za svaki fold, u train+val skupu se nasumično odabire jednako
    netoksičnih peptida koliko ima toksičnih (undersampling, kao u radu)
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
# Parametri iz rada: t:2 (RBF kernel), g:0.001 (gamma), c:5 (C), j:1 (cost factor = 1)
SVM_KERNEL = "rbf"  # t:2 = RBF kernel
SVM_C = 5  # c:5 = cost parameter
SVM_GAMMA = 0.001  # g:0.001 = gamma parameter za RBF kernel
COST_FACTOR = 1  # j:1 = cost factor (class_weight=None => jednake kazne za obje klase)

# Threshold za predikciju (iz rada: Gupta et al., 2013, Table 1, DPC)
# Threshold: -0.4 (umjesto defaultnog 0.0)
SVM_THRESHOLD = -0.60

# Random state (za reproducibilnost)
# Za svaki fold koristi se RANDOM_STATE + fold_idx radi varijabilnosti
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
    
    Nasumično odabire jednako netoksičnih peptida koliko ima toksičnih.
    Kao u radu: "we randomly picked up equal number of (1805) non-toxic peptides"
    
    Args:
        X: Feature matrica
        y: Labele (0 = netoksični, 1 = toksični)
        random_state: Seed za reproducibilnost
    
    Returns:
        X_balanced, y_balanced: Balansirani dataset (1:1 omjer)
    """
    np.random.seed(random_state)
    
    # Indeksi toksičnih i netoksičnih
    toxic_idx = np.where(y == 1)[0]
    nontoxic_idx = np.where(y == 0)[0]
    
    n_toxic = len(toxic_idx)
    n_nontoxic = len(nontoxic_idx)
    
    # Ako je već balansiran, vrati original
    if n_toxic == n_nontoxic:
        return X, y
    
    # Undersampling: odaberi n_toxic netoksičnih peptida
    if n_nontoxic > n_toxic:
        selected_nontoxic_idx = np.random.choice(
            nontoxic_idx, size=n_toxic, replace=False
        )
    else:
        # Ako ima manje netoksičnih (ne bi trebalo biti), koristi sve
        selected_nontoxic_idx = nontoxic_idx
    
    # Kombiniraj toksične + odabrane netoksične
    balanced_idx = np.concatenate([toxic_idx, selected_nontoxic_idx])
    balanced_idx = np.sort(balanced_idx)  # Sortiraj za konzistentnost
    
    X_balanced = X[balanced_idx]
    y_balanced = y[balanced_idx]
    
    return X_balanced, y_balanced


# ========================== ANALIZA REZULTATA ==========================

def analyze_results(summary, target_metrics):
    """
    Analizira dobivene rezultate u usporedbi s ciljnim vrijednostima.
    
    Returns:
        analysis: Dictionary s analizom (below_count, above_count, metrics_comparison)
    """
    metrics_to_compare = ["AUC", "GM", "Precision", "Recall", "F1", "MCC"]
    
    below_count = 0
    above_count = 0
    metrics_comparison = {}
    
    for metric_name in metrics_to_compare:
        current_mean = summary[metric_name]["mean"]
        target_value = target_metrics[metric_name]
        diff = current_mean - target_value
        
        if current_mean < target_value:
            below_count += 1
            status = "NIŽE"
        else:
            above_count += 1
            status = "VIŠE"
        
        metrics_comparison[metric_name] = {
            "current": current_mean,
            "target": target_value,
            "diff": diff,
            "status": status,
        }
    
    analysis = {
        "below_count": below_count,
        "above_count": above_count,
        "metrics_comparison": metrics_comparison,
        "meets_criteria": below_count >= 4,  # Barem 4 metrike moraju biti niže
    }
    
    return analysis


# ========================== GLAVNI TOK ==========================

def main():
    print("=" * 70)
    print("  DPC (Dipeptide Composition) + SVM referentni model")
    print("  VERZIJA S BALANSIRANIM DATASETOM (1:1 omjer, kao u radu)")
    print("=" * 70)
    print(f"\n  Parametri:")
    print(f"    Parametri iz rada (Table 1, DPC):")
    print(f"      t:2 (kernel)        -> {SVM_KERNEL.upper()}")
    print(f"      g:0.001 (gamma)    -> {SVM_GAMMA}")
    print(f"      c:5 (cost)         -> {SVM_C}")
    print(f"      j:1 (cost factor)  -> {COST_FACTOR} (class_weight=None)")
    print(f"      Threshold         -> {SVM_THRESHOLD} (iz rada)")
    print(f"    Feature scaling:    StandardScaler (per fold)")
    print(f"    Dataset balancing:  Undersampling netoksičnih na broj toksičnih (1:1)")
    print(f"    Random state:       {RANDOM_STATE} + fold_idx (varijabilnost po foldovima)")
    print(f"    Dimenzija znacajki: 400 (20x20 dipeptida)")

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

    n_toxic_original = (labels == 1).sum()
    n_nontoxic_original = (labels == 0).sum()
    
    print(f"  Ucitano sekvenci:  {len(fasta_sequences)}")
    print(f"    Toksicni  (1):   {n_toxic_original}")
    print(f"    Netoksicni (0):  {n_nontoxic_original}")
    print(f"    Omjer (original): 1:{n_nontoxic_original / n_toxic_original:.2f}")
    print(f"  Broj foldova:      {n_folds}")
    print(f"\n  Napomena: Balansiranje se izvodi unutar svakog folda (train+val skup)")
    print(f"  tako da svaki fold ima 1:1 omjer toksičnih:netoksičnih u trening skupu.")

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
    print(f"  S BALANSIRANIM DATASETOM (undersampling netoksičnih na 1:1 omjer)")
    print(f"{'-' * 70}")

    fold_metrics = []
    models_dir = os.path.join(SCRIPT_DIR, "models_balanced")
    os.makedirs(models_dir, exist_ok=True)

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
        # (SVM nema early stopping, pa koristimo i validacijski skup za trening)
        X_train_full = np.concatenate([X_train, X_val], axis=0)
        y_train_full = np.concatenate([y_train, y_val], axis=0)

        # BALANSIRANJE: Undersampling netoksičnih na broj toksičnih (1:1 omjer)
        # Kao u radu: "we randomly picked up equal number of (1805) non-toxic peptides"
        # Koristimo random_state + fold_idx za reproducibilnost
        X_train_balanced, y_train_balanced = balance_dataset(
            X_train_full, y_train_full, random_state=RANDOM_STATE + fold_idx
        )
        
        n_toxic_balanced = (y_train_balanced == 1).sum()
        n_nontoxic_balanced = (y_train_balanced == 0).sum()
        
        print(f"  Fold {fold_idx:2d}:  Original train+val: {len(y_train_full)} "
              f"(toksični: {n_toxic_balanced}, netoksični: {n_nontoxic_balanced}) -> "
              f"Balansiran: {len(y_train_balanced)} (1:1)")

        # StandardScaler - fit na balansiranom train+val, transform na train i test
        # (SVM s RBF jezgrom zahtijeva skalirane znacajke)
        # VAZNO: fit-uje se samo na balansiranom train+val, test se samo transformira
        # (osigurava da nema data leakage-a)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_balanced)
        X_test_scaled = scaler.transform(X_test)

        # Trening SVM-a
        # Parametri iz rada (Gupta et al., 2013, Table 1, DPC):
        #   t:2 -> RBF kernel
        #   g:0.001 -> gamma
        #   c:5 -> C (cost)
        #   j:1 -> cost factor = 1 (class_weight=None => jednake kazne za obje klase)
        # Napomena: random_state=RANDOM_STATE + fold_idx osigurava malu varijaciju
        # izmedju foldova uz zadrzavanje reproducibilnosti (42+1, 42+2, ..., 42+10)
        svm = SVC(
            kernel=SVM_KERNEL,  # t:2 = RBF
            C=SVM_C,  # c:5
            gamma=SVM_GAMMA,  # g:0.001
            class_weight=None,  # j:1 = cost factor = 1 (jednake kazne - kao u radu na balansiranom datasetu)
            probability=True,
            random_state=RANDOM_STATE + fold_idx,  # Varijabilnost po foldovima
        )
        svm.fit(X_train_scaled, y_train_balanced)

        # Predikcija na test skupu
        # Koristimo decision_function s threshold -0.4 (iz rada)
        # decision_function > -0.4 => klasa 1 (toksicni)
        # decision_function <= -0.4 => klasa 0 (netoksicni)
        decision_scores = svm.decision_function(X_test_scaled)
        y_pred = (decision_scores > SVM_THRESHOLD).astype(int)
        y_pred_proba = svm.predict_proba(X_test_scaled)[:, 1]

        # Metrike
        metrics = compute_metrics(y_test, y_pred, y_pred_proba)
        fold_metrics.append(metrics)

        # Spremi model + scaler za ovaj fold
        model_path = os.path.join(models_dir, f"dpc_svm_balanced_fold_{fold_idx}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({"model": svm, "scaler": scaler}, f)

        print(
            f"           Acc={metrics['Accuracy']:.4f}  "
            f"AUC={metrics['AUC']:.4f}  "
            f"MCC={metrics['MCC']:.4f}  "
            f"F1={metrics['F1']:.4f}  "
            f"GM={metrics['GM']:.4f}  "
            f"-> Saved {model_path}"
        )

    # ------------------------------------------------------------------
    # 4. Sumarni rezultati
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("  SUMARNI REZULTATI UNAKRSNE VALIDACIJE")
    print("  (BALANSIRAN DATASET - 1:1 omjer toksični:netoksični u trening skupu)")
    print(f"{'=' * 70}")

    metric_names = [
        "Accuracy", "AUC", "MCC", "Precision", "Recall",
        "F1", "GM", "Sensitivity", "Specificity",
    ]

    summary = {}
    for metric_name in metric_names:
        values = [m[metric_name] for m in fold_metrics]
        mean_val = np.mean(values)
        std_val = np.std(values)
        summary[metric_name] = {"mean": mean_val, "std": std_val}
        print(f"  {metric_name:15s}:  {mean_val:.4f} +/- {std_val:.4f}")

    # ------------------------------------------------------------------
    # 5. Analiza rezultata u usporedbi s ciljnim vrijednostima
    # ------------------------------------------------------------------
    print(f"\n{'-' * 70}")
    print("  ANALIZA REZULTATA - Usporedba s ciljnim vrijednostima")
    print(f"{'-' * 70}")
    
    analysis = analyze_results(summary, TARGET_METRICS)
    
    print(f"\n  Usporedba metrika:")
    print(f"    {'Metrika':<12s}  {'Dobiveno':<10s}  {'Cilj':<10s}  {'Razlika':<10s}  {'Status':<6s}")
    print(f"    {'-' * 12}  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 6}")
    
    for metric_name in ["AUC", "GM", "Precision", "Recall", "F1", "MCC"]:
        comp = analysis["metrics_comparison"][metric_name]
        diff_str = f"{comp['diff']:+.4f}"
        print(f"    {metric_name:<12s}  {comp['current']:<10.4f}  {comp['target']:<10.4f}  {diff_str:<10s}  {comp['status']:<6s}")
    
    print(f"\n  Sažetak:")
    print(f"    Metrike niže od cilja:  {analysis['below_count']}/6")
    print(f"    Metrike više od cilja:  {analysis['above_count']}/6")
    print(f"    Kriterij (barem 4 niže): {'✓ ZADOVOLJENO' if analysis['meets_criteria'] else '✗ NIJE ZADOVOLJENO'}")
    
    # ------------------------------------------------------------------
    # 6. Usporedba s radom
    # ------------------------------------------------------------------
    print(f"\n{'-' * 70}")
    print("  Usporedba s rezultatima iz rada (Gupta et al., 2013):")
    print(f"{'-' * 70}")
    print(f"    Rad (5-fold, balansirani dataset):  Acc=0.9388  MCC=0.8800")
    print(
        f"    Nasa impl. ({n_folds}-fold, balansirani):  "
        f"Acc={summary['Accuracy']['mean']:.4f}  "
        f"MCC={summary['MCC']['mean']:.4f}"
    )

    # ------------------------------------------------------------------
    # 7. Spremanje rezultata u XLSX
    # ------------------------------------------------------------------
    results_file = os.path.join(SCRIPT_DIR, "DPC_SVM_results_balanced.xlsx")

    results_data = []
    for fold_idx, metrics in enumerate(fold_metrics, 1):
        row = {"Fold": fold_idx}
        row.update(metrics)
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
    fold_results_file = os.path.join(SCRIPT_DIR, "dpc_svm_fold_results_balanced.pkl")

    fold_results_data = {
        "fold_metrics": fold_metrics,
        "metric_names": metric_names,
        "summary": summary,
        "n_folds": n_folds,
        "svm_params": {
            "kernel": SVM_KERNEL,  # t:2 = RBF (iz rada)
            "C": SVM_C,  # c:5 (iz rada)
            "gamma": SVM_GAMMA,  # g:0.001 (iz rada)
            "class_weight": None,  # j:1 = cost factor = 1 (iz rada)
            "cost_factor": COST_FACTOR,  # j:1 (iz rada)
            "random_state_base": RANDOM_STATE,
            "random_state_strategy": "RANDOM_STATE + fold_idx",
            "source": "Gupta et al., 2013, Table 1, DPC row",
            "dataset_balancing": "Undersampling netoksičnih na broj toksičnih (1:1 omjer)",
            "balancing_method": "Per fold, in train+val set (before training)",
            "threshold": SVM_THRESHOLD,  # -0.4 (iz rada, Table 1, DPC)
            "target_metrics": TARGET_METRICS,
            "results_analysis": analysis,
        },
    }
    with open(fold_results_file, "wb") as f:
        pickle.dump(fold_results_data, f)

    print(f"  Rezultati po foldovima: {fold_results_file}")
    print(f"\n  Spremljeno {n_folds} SVM modela u: {models_dir}/")
    print(f"    dpc_svm_balanced_fold_1.pkl ... dpc_svm_balanced_fold_{n_folds}.pkl")
    print(f"\n  Napomena: Modeli su trenirani na balansiranom datasetu (1:1 omjer)")
    print(f"  kao u originalnom radu (Gupta et al., 2013).")

    print("\n[DONE] DPC + SVM evaluacija zavrsena (balansirani dataset).")


if __name__ == "__main__":
    main()
