"""
Kreiranje balansiranih splitova za DPC SVM model.

Ova skripta učitava originalne splitove iz aligned_peptide_data.pkl,
balansira train+val skup za svaki fold (1:1 omjer toksični:netoksični),
i sprema balansirane splitove u novi pickle file.
"""

import os
import pickle
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ALIGNED_DATA_FILE = os.path.join(SCRIPT_DIR, "aligned_peptide_data.pkl")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "balanced_peptide_data.pkl")
RANDOM_STATE = 42


def balance_dataset_indices(original_indices, y, random_state):
    """Balansira dataset i vraća odabrane indekse iz originalnih indeksa."""
    np.random.seed(random_state)
    
    toxic_mask = (y == 1)
    nontoxic_mask = (y == 0)
    
    toxic_indices = original_indices[toxic_mask]
    nontoxic_indices = original_indices[nontoxic_mask]
    
    n_toxic = len(toxic_indices)
    n_nontoxic = len(nontoxic_indices)
    
    if n_toxic == n_nontoxic:
        return np.concatenate([toxic_indices, nontoxic_indices])
    
    if n_nontoxic > n_toxic:
        selected_nontoxic_idx = np.random.choice(
            len(nontoxic_indices), size=n_toxic, replace=False
        )
        selected_nontoxic_indices = nontoxic_indices[selected_nontoxic_idx]
    else:
        selected_nontoxic_indices = nontoxic_indices
    
    balanced_idx = np.concatenate([toxic_indices, selected_nontoxic_indices])
    return np.sort(balanced_idx)


def main():
    print("=" * 70)
    print("  KREIRANJE BALANSIRANIH SPLITOVA")
    print("=" * 70)
    
    print(f"\n  Učitavanje: {ALIGNED_DATA_FILE}")
    if not os.path.exists(ALIGNED_DATA_FILE):
        print(f"  GRESKA: Datoteka '{ALIGNED_DATA_FILE}' nije pronadjena!")
        print("  Pokrenite najprije create_aligned_splits.py")
        return
    
    with open(ALIGNED_DATA_FILE, "rb") as f:
        aligned_data = pickle.load(f)
    
    fasta_sequences = aligned_data["fasta_sequences"]
    labels = aligned_data["labels"]
    cv_splits = aligned_data["cv_splits"]
    n_folds = aligned_data["n_folds"]
    
    print(f"  Učitano sekvenci: {len(fasta_sequences)}")
    print(f"  Broj foldova: {n_folds}")
    
    balanced_splits = []
    
    for fold_idx, split in enumerate(cv_splits, 1):
        train_idx = split["train_idx"]
        val_idx = split["val_idx"]
        test_idx = split["test_idx"]
        
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        
        train_full_idx = np.concatenate([train_idx, val_idx])
        train_full_labels = np.concatenate([train_labels, val_labels])
        
        balanced_train_full_idx = balance_dataset_indices(
            train_full_idx, train_full_labels, random_state=RANDOM_STATE + fold_idx
        )
        
        balanced_train_labels = labels[balanced_train_full_idx]
        n_toxic_train = (balanced_train_labels == 1).sum()
        n_nontoxic_train = (balanced_train_labels == 0).sum()
        
        print(f"  Fold {fold_idx:2d}: Balansiran train+val: {len(balanced_train_full_idx)} "
              f"(toksični: {n_toxic_train}, netoksični: {n_nontoxic_train})")
        
        balanced_splits.append({
            "train_idx": balanced_train_full_idx,
            "val_idx": np.array([]),
            "test_idx": test_idx,
        })
    
    balanced_data = {
        "fasta_sequences": fasta_sequences,
        "labels": labels,
        "cv_splits": balanced_splits,
        "n_folds": n_folds,
        "balanced": True,
        "balance_ratio": "1:1",
        "random_state_base": RANDOM_STATE,
    }
    
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(balanced_data, f)
    
    print(f"\n  Balansirani splitovi spremljeni u: {OUTPUT_FILE}")
    print("  [DONE]")


if __name__ == "__main__":
    main()
