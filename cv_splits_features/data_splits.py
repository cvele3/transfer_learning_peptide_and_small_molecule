# data_splits.py
import os
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

def generate_cv_splits(labels, n_splits=10, val_split=0.2, random_state=42, save_path="cv_splits.pkl"):
    """
    Za dani niz labela (npr. 0/1) generira n_splits cross validation podjela.
    Za svaki fold:
      - Prvo se podaci dijele u (train+val) i test (prema StratifiedKFold),
      - Zatim se unutar train+val skupa radi dodatna podjela na trening i validaciju.
    Sprema se u pickle datoteku.
    """
    labels = np.array(labels)
    all_indices = np.arange(len(labels))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_splits = []
    
    for train_val_idx, test_idx in skf.split(all_indices, labels):
        # Daljnja podjela train_val skupa na trening i validaciju
        train_idx, val_idx, _, _ = train_test_split(
            train_val_idx,
            labels[train_val_idx],
            test_size=val_split,
            random_state=random_state,
            stratify=labels[train_val_idx]
        )
        cv_splits.append({
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx
        })
    
    with open(save_path, "wb") as f:
        pickle.dump(cv_splits, f)
    print(f"CV podjele su spremljene u '{save_path}'.")
    return cv_splits

def load_cv_splits(save_path="cv_splits.pkl"):
    """
    Učitava spremljene cross validation podjele.
    Ako datoteka ne postoji, bacit će se greška – pa je dobro prvo generirati splitove.
    """
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Datoteka '{save_path}' ne postoji. Prvo generirajte splitove!")
    with open(save_path, "rb") as f:
        cv_splits = pickle.load(f)
    print(f"Učitani CV splitovi iz '{save_path}'.")
    return cv_splits

# New part, jet to be used
def split_train_eval_indices(labels, eval_ratio=0.2, random_state=42):
    """
    Splits all indices based on labels into a training candidate set and an evaluation set.
    """
    labels = np.array(labels)
    all_indices = np.arange(len(labels))
    train_idx, eval_idx = train_test_split(
        all_indices,
        test_size=eval_ratio,
        random_state=random_state,
        stratify=labels
    )
    return train_idx, eval_idx

def generate_cv_splits_for_subset(subset_indices, labels, n_splits=10, val_split=0.2, random_state=42):
    """
    From a given subset of indices, generate CV splits.
    Each split contains a further split of the training+validation set into training and validation.
    Returns a list of dictionaries with keys: "train_idx", "val_idx", and "test_idx".
    """
    subset_indices = np.array(subset_indices)
    subset_labels = np.array(labels)[subset_indices]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_splits = []
    
    # We use np.arange(len(subset_indices)) for indexing within the subset.
    for train_val_local, test_local in skf.split(np.arange(len(subset_indices)), subset_labels):
        # Convert local indices to global indices from subset_indices
        train_val_global = subset_indices[train_val_local]
        test_global = subset_indices[test_local]
        
        # Further split train_val_global into actual training and validation sets.
        train_idx, val_idx, _, _ = train_test_split(
            train_val_global,
            np.array(labels)[train_val_global],
            test_size=val_split,
            random_state=random_state,
            stratify=np.array(labels)[train_val_global]
        )
        
        cv_splits.append({
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_global
        })
    
    return cv_splits

def generate_train_and_eval_cv_splits(labels, eval_ratio=0.2, n_splits=10, val_split=0.2, random_state=42,
                                      train_save_path="train_cv_splits.pkl", eval_save_path="eval_cv_splits.pkl"):
    """
    Splits the data into two parts:
    1. A training candidate set to be used for model training (with its own CV splits).
    2. A separate evaluation set (with its own CV splits).
    The respective splits are then saved in separate pickle files.
    """
    train_idx, eval_idx = split_train_eval_indices(labels, eval_ratio, random_state)
    
    # Generate CV splits for the training candidate data (for training and validation during model development)
    train_cv_splits = generate_cv_splits_for_subset(train_idx, labels, n_splits, val_split, random_state)
    with open(train_save_path, "wb") as f:
        pickle.dump(train_cv_splits, f)
    print(f"Train CV splits saved in '{train_save_path}'.")
    
    # Generate CV splits for the evaluation set (for unbiased final model evaluation)
    eval_cv_splits = generate_cv_splits_for_subset(eval_idx, labels, n_splits, val_split, random_state)
    with open(eval_save_path, "wb") as f:
        pickle.dump(eval_cv_splits, f)
    print(f"Eval CV splits saved in '{eval_save_path}'.")
    
    return train_cv_splits, eval_cv_splits