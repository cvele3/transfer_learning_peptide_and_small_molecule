"""
Compare CV splits between old and new 72-feature versions.

This script compares:
1. cv_splits_peptide_with_features.pkl vs cv_splits_peptide_72_features.pkl
2. cv_splits_small_mol_tox_pred_with_features.pkl vs cv_splits_small_mol_72_features.pkl

It checks if the same indices (and thus same molecules) are in the same folds.
"""

import os
import pickle
import numpy as np
from collections import Counter

# ========================== KONFIGURACIJA ==========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# CV split files to compare
PEPTIDE_OLD = "cv_splits_peptide_with_features.pkl"
PEPTIDE_NEW = "cv_splits_peptide_72_features.pkl"

SMALL_MOL_OLD = "cv_splits_small_mol_tox_pred_with_features.pkl"
SMALL_MOL_NEW = "cv_splits_small_mol_72_features.pkl"

# Optional: Load underlying data to verify molecules match
PEPTIDE_DATA_OLD = "overtrained_peptide_graphs_with_features.pkl"
PEPTIDE_DATA_NEW = "overtrained_peptide_graphs_72_features.pkl"

SMALL_MOL_DATA_OLD = "overtrained_small_molecule_mol_tox_pred_graphs_with_features.pkl"
SMALL_MOL_DATA_NEW = "overtrained_small_molecule_graphs_72_features.pkl"


# ========================== POMOĆNE FUNKCIJE ==========================

def load_splits(filepath):
    """Učitava CV splitove iz pickle datoteke."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, "rb") as f:
        splits = pickle.load(f)
    return splits


def compare_splits(splits1, splits2, name1, name2):
    """
    Uspoređuje dva CV split fajla i provjerava jesu li isti.
    
    Args:
        splits1: Prvi CV split (lista dict-ova)
        splits2: Drugi CV split (lista dict-ova)
        name1: Ime prvog split fajla
        name2: Ime drugog split fajla
    
    Returns:
        Dictionary s rezultatima usporedbe
    """
    print(f"\n{'=' * 70}")
    print(f"  Usporedba: {name1} vs {name2}")
    print(f"{'=' * 70}")
    
    if splits1 is None:
        print(f"  GRESKA: {name1} nije pronadjena!")
        return None
    
    if splits2 is None:
        print(f"  GRESKA: {name2} nije pronadjena!")
        return None
    
    # Provjeri broj foldova
    n_folds1 = len(splits1)
    n_folds2 = len(splits2)
    
    print(f"\n  Broj foldova:")
    print(f"    {name1}: {n_folds1}")
    print(f"    {name2}: {n_folds2}")
    
    if n_folds1 != n_folds2:
        print(f"  GRESKA: Različit broj foldova!")
        return None
    
    # Usporedi svaki fold
    results = {
        "n_folds": n_folds1,
        "folds_match": True,
        "fold_details": []
    }
    
    all_match = True
    
    for fold_idx in range(n_folds1):
        fold1 = splits1[fold_idx]
        fold2 = splits2[fold_idx]
        
        # Sortiraj indekse za usporedbu
        train1 = np.sort(fold1["train_idx"])
        val1 = np.sort(fold1["val_idx"])
        test1 = np.sort(fold1["test_idx"])
        
        train2 = np.sort(fold2["train_idx"])
        val2 = np.sort(fold2["val_idx"])
        test2 = np.sort(fold2["test_idx"])
        
        # Provjeri jesu li isti
        train_match = np.array_equal(train1, train2)
        val_match = np.array_equal(val1, val2)
        test_match = np.array_equal(test1, test2)
        
        fold_match = train_match and val_match and test_match
        
        fold_detail = {
            "fold": fold_idx + 1,
            "train_match": train_match,
            "val_match": val_match,
            "test_match": test_match,
            "train_size_1": len(train1),
            "train_size_2": len(train2),
            "val_size_1": len(val1),
            "val_size_2": len(val2),
            "test_size_1": len(test1),
            "test_size_2": len(test2),
        }
        
        if not fold_match:
            all_match = False
            # Pronađi razlike
            train_diff = np.setdiff1d(train1, train2)
            val_diff = np.setdiff1d(val1, val2)
            test_diff = np.setdiff1d(test1, test2)
            
            fold_detail["train_diff"] = train_diff.tolist()
            fold_detail["val_diff"] = val_diff.tolist()
            fold_detail["test_diff"] = test_diff.tolist()
        
        results["fold_details"].append(fold_detail)
        
        status = "[OK]" if fold_match else "[DIFF]"
        print(f"\n  Fold {fold_idx + 1}: {status}")
        print(f"    Train: {'MATCH' if train_match else 'DIFFERENT'} "
              f"({len(train1)} vs {len(train2)})")
        print(f"    Val:   {'MATCH' if val_match else 'DIFFERENT'} "
              f"({len(val1)} vs {len(val2)})")
        print(f"    Test:  {'MATCH' if test_match else 'DIFFERENT'} "
              f"({len(test1)} vs {len(test2)})")
        
        if not fold_match:
            if len(fold_detail.get("train_diff", [])) > 0:
                print(f"    Train razlike: {len(fold_detail['train_diff'])} indeksa")
            if len(fold_detail.get("val_diff", [])) > 0:
                print(f"    Val razlike: {len(fold_detail['val_diff'])} indeksa")
            if len(fold_detail.get("test_diff", [])) > 0:
                print(f"    Test razlike: {len(fold_detail['test_diff'])} indeksa")
    
    results["folds_match"] = all_match
    
    print(f"\n  {'=' * 70}")
    if all_match:
        print(f"  [OK] SVI FOLDOVI SE PODUDARAJU!")
    else:
        print(f"  [DIFF] POSTOJE RAZLIKE IZMEDJU SPLITOVA!")
    print(f"  {'=' * 70}")
    
    return results


def verify_data_consistency(data_file1, data_file2, splits1, splits2, name1, name2):
    """
    Provjerava jesu li isti molekuli na istim indeksima u oba dataseta.
    Ovo je dodatna provjera - uspoređuje SMILES stringove ako su dostupni.
    """
    print(f"\n{'=' * 70}")
    print(f"  Provjera konzistentnosti podataka")
    print(f"{'=' * 70}")
    
    if not os.path.exists(data_file1) or not os.path.exists(data_file2):
        print(f"  Preskačem - nedostaju data fajlovi")
        return None
    
    try:
        with open(data_file1, "rb") as f:
            data1 = pickle.load(f)
        with open(data_file2, "rb") as f:
            data2 = pickle.load(f)
        
        labels1 = data1.get("labels", [])
        labels2 = data2.get("labels", [])
        
        print(f"\n  Veličine datasetova:")
        print(f"    {name1}: {len(labels1)} primjera")
        print(f"    {name2}: {len(labels2)} primjera")
        
        if len(labels1) != len(labels2):
            print(f"  UPOZORENJE: Različite veličine datasetova!")
            return None
        
        # Provjeri jesu li labeli isti na istim indeksima
        labels_match = np.array_equal(np.array(labels1), np.array(labels2))
        
        print(f"\n  Labeli na istim indeksima: {'MATCH' if labels_match else 'DIFFERENT'}")
        
        if not labels_match:
            diff_indices = np.where(np.array(labels1) != np.array(labels2))[0]
            print(f"  Razlike na {len(diff_indices)} indeksima (prvih 10): {diff_indices[:10]}")
        
        # Provjeri jesu li isti labeli u svakom foldu
        print(f"\n  Provjera labela po foldovima:")
        for fold_idx in range(min(len(splits1), len(splits2))):
            test_idx1 = splits1[fold_idx]["test_idx"]
            test_idx2 = splits2[fold_idx]["test_idx"]
            
            labels_fold1 = [labels1[i] for i in test_idx1]
            labels_fold2 = [labels2[i] for i in test_idx2]
            
            # Provjeri distribuciju labela
            count1 = Counter(labels_fold1)
            count2 = Counter(labels_fold2)
            
            labels_dist_match = count1 == count2
            
            status = "[OK]" if labels_dist_match else "[DIFF]"
            print(f"    Fold {fold_idx + 1}: {status} "
                  f"(Label 0: {count1.get(0, 0)} vs {count2.get(0, 0)}, "
                  f"Label 1: {count1.get(1, 0)} vs {count2.get(1, 0)})")
        
        return labels_match
        
    except Exception as e:
        print(f"  GRESKA pri učitavanju podataka: {e}")
        return None


# ========================== GLAVNI TOK ==========================

def main():
    print("=" * 70)
    print("  USPOREDBA CV SPLITOVA - STARE VS NOVE 72-FEATURE VERZIJE")
    print("=" * 70)
    
    # Učitaj peptide splitove
    peptide_old_path = os.path.join(SCRIPT_DIR, PEPTIDE_OLD)
    peptide_new_path = os.path.join(SCRIPT_DIR, PEPTIDE_NEW)
    
    peptide_splits_old = load_splits(peptide_old_path)
    peptide_splits_new = load_splits(peptide_new_path)
    
    # Učitaj small molecule splitove
    small_mol_old_path = os.path.join(SCRIPT_DIR, SMALL_MOL_OLD)
    small_mol_new_path = os.path.join(SCRIPT_DIR, SMALL_MOL_NEW)
    
    small_mol_splits_old = load_splits(small_mol_old_path)
    small_mol_splits_new = load_splits(small_mol_new_path)
    
    # Usporedi peptide splitove
    peptide_results = compare_splits(
        peptide_splits_old,
        peptide_splits_new,
        PEPTIDE_OLD,
        PEPTIDE_NEW
    )
    
    # Usporedi small molecule splitove
    small_mol_results = compare_splits(
        small_mol_splits_old,
        small_mol_splits_new,
        SMALL_MOL_OLD,
        SMALL_MOL_NEW
    )
    
    # Opcionalna provjera konzistentnosti podataka
    if peptide_results and peptide_results["folds_match"]:
        peptide_data_old_path = os.path.join(SCRIPT_DIR, PEPTIDE_DATA_OLD)
        peptide_data_new_path = os.path.join(SCRIPT_DIR, PEPTIDE_DATA_NEW)
        verify_data_consistency(
            peptide_data_old_path,
            peptide_data_new_path,
            peptide_splits_old,
            peptide_splits_new,
            PEPTIDE_DATA_OLD,
            PEPTIDE_DATA_NEW
        )
    
    if small_mol_results and small_mol_results["folds_match"]:
        small_mol_data_old_path = os.path.join(SCRIPT_DIR, SMALL_MOL_DATA_OLD)
        small_mol_data_new_path = os.path.join(SCRIPT_DIR, SMALL_MOL_DATA_NEW)
        verify_data_consistency(
            small_mol_data_old_path,
            small_mol_data_new_path,
            small_mol_splits_old,
            small_mol_splits_new,
            SMALL_MOL_DATA_OLD,
            SMALL_MOL_DATA_NEW
        )
    
    # Sažetak
    print(f"\n{'=' * 70}")
    print("  SAŽETAK")
    print(f"{'=' * 70}")
    
    if peptide_results:
        status = "[OK] PODUDARAJU SE" if peptide_results["folds_match"] else "[DIFF] RAZLIKE"
        print(f"  Peptide splitovi: {status}")
    else:
        print(f"  Peptide splitovi: N/A (nedostaju fajlovi)")
    
    if small_mol_results:
        status = "[OK] PODUDARAJU SE" if small_mol_results["folds_match"] else "[DIFF] RAZLIKE"
        print(f"  Small molecule splitovi: {status}")
    else:
        print(f"  Small molecule splitovi: N/A (nedostaju fajlovi)")
    
    print("\n[DONE]")


if __name__ == "__main__":
    main()
