"""
Create CV splits for datasets with 72-element vocabulary.

This script loads the pickle files created by create_graphs_72_features.py
and generates cross-validation splits for each dataset.
"""

import os
import pickle
import numpy as np
from data_splits import generate_cv_splits

# ========================== KONFIGURACIJA ==========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Input pickle files (created by create_graphs_72_features.py)
PROCESSED_DATA_FILE_PEPTIDE = "overtrained_peptide_graphs_72_features.pkl"
PROCESSED_DATA_FILE_SMALL_MOL = "overtrained_small_molecule_graphs_72_features.pkl"

# Output CV split files
CV_SPLITS_FILE_PEPTIDE = "cv_splits_peptide_72_features.pkl"
CV_SPLITS_FILE_SMALL_MOL = "cv_splits_small_mol_72_features.pkl"

# CV parametri
N_SPLITS = 10
VAL_SPLIT = 0.2
RANDOM_STATE = 42


# ========================== POMOĆNE FUNKCIJE ==========================

def create_cv_splits_for_dataset(pickle_file, splits_file, dataset_name):
    """
    Učitava pickle datoteku s grafovima i kreira CV splitove.
    
    Args:
        pickle_file: Putanja do pickle datoteke s grafovima
        splits_file: Putanja gdje će se spremiti CV splitovi
        dataset_name: Ime dataseta (za printanje)
    
    Returns:
        True ako je uspješno, False inače
    """
    print(f"\n{'=' * 70}")
    print(f"  Kreiranje CV splitova za: {dataset_name}")
    print(f"{'=' * 70}")
    
    pickle_path = os.path.join(SCRIPT_DIR, pickle_file)
    splits_path = os.path.join(SCRIPT_DIR, splits_file)
    
    if not os.path.exists(pickle_path):
        print(f"  GRESKA: Datoteka '{pickle_path}' nije pronadjena!")
        print(f"  Prvo pokrenite create_graphs_72_features.py")
        return False
    
    print(f"  Učitavanje: {pickle_path}")
    with open(pickle_path, "rb") as f:
        processed_data = pickle.load(f)
    
    graphs = processed_data["graphs"]
    labels = processed_data["labels"]
    graph_labels = processed_data["graph_labels"]
    element_to_index = processed_data["element_to_index"]
    
    print(f"  Učitano grafova: {len(graphs)}")
    print(f"  Vokabular elemenata: {len(element_to_index)} elemenata")
    
    if "fingerprints" in processed_data:
        fingerprints = processed_data["fingerprints"]
        print(f"  Fingerprintova: {len(fingerprints)}")
        if len(fingerprints) > 0:
            print(f"  Dimenzija fingerprinta: {len(fingerprints[0])}")
    
    print(f"\n  Raspodjela labela:")
    print(f"    Label 0: {(graph_labels == 0).sum()}")
    print(f"    Label 1: {(graph_labels == 1).sum()}")
    
    # Provjeri već postoje li splitovi
    if os.path.exists(splits_path):
        print(f"\n  UPOZORENJE: Datoteka '{splits_path}' već postoji!")
        response = input("  Prepisati postojeće splitove? (y/n): ")
        if response.lower() != 'y':
            print("  Preskačem...")
            return False
    
    # Generiranje CV splitova
    print(f"\n  Generiranje {N_SPLITS}-fold CV splitova...")
    print(f"    Val split: {VAL_SPLIT}")
    print(f"    Random state: {RANDOM_STATE}")
    
    cv_splits = generate_cv_splits(
        graph_labels.values,
        n_splits=N_SPLITS,
        val_split=VAL_SPLIT,
        random_state=RANDOM_STATE,
        save_path=splits_path
    )
    
    print(f"\n  CV splitovi spremljeni u: {splits_path}")
    print(f"  Broj foldova: {len(cv_splits)}")
    
    # Prikaz statistike za prvi fold
    if len(cv_splits) > 0:
        first_fold = cv_splits[0]
        print(f"\n  Primjer (Fold 1):")
        print(f"    Train: {len(first_fold['train_idx'])} primjera")
        print(f"    Val:   {len(first_fold['val_idx'])} primjera")
        print(f"    Test:  {len(first_fold['test_idx'])} primjera")
    
    return True


# ========================== GLAVNI TOK ==========================

def main():
    print("=" * 70)
    print("  KREIRANJE CV SPLITOVA ZA DATASETE S 72-ELEMENT VOKABULAROM")
    print("=" * 70)
    
    datasets = [
        (PROCESSED_DATA_FILE_PEPTIDE, CV_SPLITS_FILE_PEPTIDE, "Peptide Dataset"),
        (PROCESSED_DATA_FILE_SMALL_MOL, CV_SPLITS_FILE_SMALL_MOL, "Small Molecule Dataset"),
    ]
    
    success_count = 0
    for pickle_file, splits_file, dataset_name in datasets:
        if create_cv_splits_for_dataset(pickle_file, splits_file, dataset_name):
            success_count += 1
    
    print(f"\n{'=' * 70}")
    print(f"  ZAVRŠENO: {success_count}/{len(datasets)} datasetova uspješno procesirano")
    print(f"{'=' * 70}")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
