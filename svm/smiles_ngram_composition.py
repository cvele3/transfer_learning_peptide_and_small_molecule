"""
SMILES N-gram Composition Feature Extraction

This module implements n-gram composition for SMILES strings, similar to 
Dipeptide Composition (DPC) for peptide sequences.

The function counts the frequency of character n-grams (typically bigrams/2-grams)
in SMILES strings and normalizes them to create fixed-length feature vectors.

Example:
    For SMILES "CCO" with n=2:
    - Bigrams: "CC", "CO"
    - Counts: {"CC": 1, "CO": 1}
    - Normalized: frequencies divided by total bigrams (2)
"""

import os
import pickle
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import pandas as pd
import math

# ========================== KONFIGURACIJA ==========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "peptide_data.pkl")
EXCEL_FILE = os.path.join(SCRIPT_DIR, "..", "datasets", "ToxinSequenceSMILES.xlsx")

# SVM parametri
SVM_KERNEL = "rbf"
SVM_C = 5
SVM_GAMMA = 0.001
SVM_THRESHOLD = -0.60
RANDOM_STATE = 42

# N-gram parametri
NGRAM_SIZE = 2  # 2 = bigrams (like DPC), 3 = trigrams, etc.

# ========================== SMILES N-GRAM COMPOSITION ==========================

def get_all_smiles_ngrams(smiles_list, n=2):
    """
    Ekstrahira sve moguće n-grame iz liste SMILES stringova.
    
    Args:
        smiles_list: Lista SMILES stringova
        n: Veličina n-grama (default 2 za bigrame, kao DPC)
    
    Returns:
        Sorted list svih jedinstvenih n-grama
    """
    all_ngrams = set()
    for smiles in smiles_list:
        if smiles and len(smiles) >= n:
            for i in range(len(smiles) - n + 1):
                ngram = smiles[i:i+n]
                all_ngrams.add(ngram)
    return sorted(list(all_ngrams))


def compute_smiles_ngram_composition(smiles, ngram_vocab, n=2):
    """
    Računa N-gram Composition feature vektor za SMILES string.
    
    Slično kao DPC za peptide sekvence:
      - Za SMILES duljine L: ukupan broj n-grama = L - n + 1
      - N-gram Composition(i) = count(ngram_i) / (L - n + 1)
    
    Args:
        smiles: SMILES string
        ngram_vocab: Lista svih mogućih n-grama (vokabular)
        n: Veličina n-grama (default 2)
    
    Returns:
        Normalizirani vektor frekvencija n-grama
    """
    if not smiles or len(smiles) < n:
        return np.zeros(len(ngram_vocab))
    
    # Brojanje n-grama
    counts = np.zeros(len(ngram_vocab))
    ngram_to_index = {ng: i for i, ng in enumerate(ngram_vocab)}
    
    total_ngrams = len(smiles) - n + 1
    for i in range(total_ngrams):
        ngram = smiles[i:i+n]
        if ngram in ngram_to_index:
            counts[ngram_to_index[ngram]] += 1
    
    # Normalizacija
    if total_ngrams > 0:
        counts = counts / total_ngrams
    
    return counts


# ========================== METRIKE ==========================

def compute_metrics(y_true, y_pred, y_pred_proba=None):
    """Izračunava sve evaluacijske metrike."""
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


# ========================== GLAVNI TOK ==========================

def main():
    print("=" * 70)
    print(f"  SMILES {NGRAM_SIZE}-gram Composition + SVM")
    print("=" * 70)
    
    print(f"\n  Učitavanje: {DATA_FILE}")
    if not os.path.exists(DATA_FILE):
        print(f"  GRESKA: Datoteka '{DATA_FILE}' nije pronadjena!")
        print("  Napomena: Ova skripta očekuje peptide_data.pkl s SMILES stupcem")
        return
    
    # Pokušaj učitati iz pickle datoteke
    smiles_list = None
    labels = None
    cv_splits = None
    n_folds = None
    
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "rb") as f:
            data = pickle.load(f)
        
        # Provjeri ima li SMILES podataka
        if "smiles" in data:
            smiles_list = data["smiles"]
        elif "SMILES" in data:
            smiles_list = data["SMILES"]
        
        if "labels" in data:
            labels = data["labels"]
        if "cv_splits" in data:
            cv_splits = data["cv_splits"]
        if "n_folds" in data:
            n_folds = data["n_folds"]
    
    # Ako nema SMILES u pickle, pokušaj učitati iz Excel datoteke
    if smiles_list is None:
        print(f"  Nema SMILES podataka u pickle datoteci.")
        print(f"  Pokušavam učitati iz Excel datoteke: {EXCEL_FILE}")
        
        if not os.path.exists(EXCEL_FILE):
            print(f"  GRESKA: Excel datoteka '{EXCEL_FILE}' nije pronadjena!")
            print("\n  Napomena: Trebate dodati SMILES podatke u peptide_data.pkl")
            print("  ili osigurati da ToxinSequenceSMILES.xlsx postoji u datasets/ folderu.")
            return
        
        try:
            df = pd.read_excel(EXCEL_FILE, usecols=["SMILES", "TOXICITY"])
            smiles_list = df["SMILES"].astype(str).tolist()
            labels = df["TOXICITY"].values
            
            print(f"  Učitano {len(smiles_list)} SMILES stringova iz Excel datoteke.")
            
            # Ako nema CV splitova, koristi jednostavnu podjelu (upozorenje)
            if cv_splits is None:
                print("\n  UPOZORENJE: Nema CV splitova u podacima!")
                print("  Koristit će se jednostavna train/test podjela (80/20).")
                print("  Za pravu CV validaciju, trebate koristiti peptide_data.pkl s CV splitovima.")
                
                from sklearn.model_selection import train_test_split
                indices = np.arange(len(smiles_list))
                train_idx, test_idx = train_test_split(
                    indices, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
                )
                cv_splits = [{"train_idx": train_idx, "test_idx": test_idx}]
                n_folds = 1
        except Exception as e:
            print(f"  GRESKA pri učitavanju Excel datoteke: {e}")
            return
    
    if smiles_list is None or labels is None:
        print("  GRESKA: Nije moguće učitati SMILES podatke!")
        return
    
    if cv_splits is None:
        print("  GRESKA: Nema CV splitova! Potrebni su za evaluaciju.")
        return
    
    if n_folds is None:
        n_folds = len(cv_splits)
    
    print(f"  Učitano SMILES stringova: {len(smiles_list)}")
    print(f"  Broj foldova: {n_folds}")
    
    # Ekstrahiraj sve moguće n-grame iz svih SMILES stringova
    print(f"\n  Ekstrahiranje {NGRAM_SIZE}-gram vokabulara iz svih SMILES stringova...")
    ngram_vocab = get_all_smiles_ngrams(smiles_list, n=NGRAM_SIZE)
    print(f"  Pronađeno jedinstvenih {NGRAM_SIZE}-grama: {len(ngram_vocab)}")
    
    # Računanje n-gram composition značajki
    print(f"\n  Računanje {NGRAM_SIZE}-gram composition značajki...")
    X_all = np.array([
        compute_smiles_ngram_composition(smiles, ngram_vocab, n=NGRAM_SIZE)
        for smiles in smiles_list
    ])
    y_all = labels
    print(f"  Matrica značajki: {X_all.shape}")
    
    print(f"\n  Pokretanje {n_folds}-fold unakrsne validacije...")
    fold_metrics = []
    models_dir = os.path.join(SCRIPT_DIR, "models_smiles_ngram")
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
        
        model_path = os.path.join(models_dir, f"smiles_{NGRAM_SIZE}gram_svm_fold_{fold_idx}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": svm,
                "scaler": scaler,
                "ngram_vocab": ngram_vocab,
                "ngram_size": NGRAM_SIZE
            }, f)
        
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
    
    results_file = os.path.join(SCRIPT_DIR, f"SMILES_{NGRAM_SIZE}gram_SVM_results.xlsx")
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
    print(f"  N-gram vokabular veličine: {len(ngram_vocab)}")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
