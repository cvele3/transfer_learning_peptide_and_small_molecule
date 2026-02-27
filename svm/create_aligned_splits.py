"""
Kreiranje uskladjenih CV splitova za DPC SVM model.

Cilj: Osigurati da SVM model koristi ISTE foldove (iste peptide u train/val/test)
kao i GNN modeli, samo s FASTA sekvencama umjesto SMILES grafova.

Postupak:
  1. Ucitava ToxinSequenceSMILES.xlsx na isti nacin kao GNN pipeline
     (isti redoslijed, isto filtriranje nevalidnih SMILES-a).
  2. Cuva stupac SEQUENCE (= FASTA) za svaki validni unos.
  3. Ucitava GNN CV splitove (large_layers_cv_splits_peptide.pkl).
  4. Sprema uskladjeni pickle s FASTA sekvencama, labelama i CV splitovima.

Rezultat: svm/aligned_peptide_data.pkl
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem

# ========================== KONFIGURACIJA ==========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# Izvorni peptide dataset (isti koji koristi GNN pipeline)
EXCEL_FILE = os.path.join(PROJECT_DIR, "datasets", "ToxinSequenceSMILES.xlsx")

# GNN CV splitovi (isti koji koriste svi GNN modeli)
GNN_SPLITS_FILE = os.path.join(
    PROJECT_DIR, "large_layers_cv_splits", "large_layers_cv_splits_peptide.pkl"
)

# Izlazna datoteka
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "aligned_peptide_data.pkl")


# ========================== GLAVNI TOK ==========================

def main():
    print("=" * 70)
    print("  Kreiranje uskladjenih CV splitova za DPC SVM model")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Ucitavanje Excel datoteke (isti pristup kao GNN pipeline)
    # ------------------------------------------------------------------
    print(f"\n  Ucitavanje: {EXCEL_FILE}")

    if not os.path.exists(EXCEL_FILE):
        print(f"  GRESKA: Datoteka '{EXCEL_FILE}' nije pronadjena!")
        sys.exit(1)

    data_file = pd.read_excel(
        EXCEL_FILE, header=0, usecols=["SEQUENCE", "TOXICITY", "SMILES"]
    )
    data_file.reset_index(drop=True, inplace=True)
    print(f"  Ukupno redaka u Excel datoteci: {len(data_file)}")

    # ------------------------------------------------------------------
    # 2. Filtriranje na isti nacin kao GNN pipeline
    #    (overtrained_models/peptide_overtrained.py linije 95-127)
    #    - Preskoce se redci gdje MolFromSmiles vraca None
    #    - Zadrze se samo labeli 0 i 1
    # ------------------------------------------------------------------
    print("\n  Filtriranje (isti postupak kao GNN pipeline)...")

    valid_sequences = []  # FASTA sekvence
    valid_labels = []     # TOXICITY labele
    valid_smiles = []     # originalni SMILES (za verifikaciju)
    skipped = 0

    for _, row in data_file.iterrows():
        smiles = row["SMILES"]
        label = row["TOXICITY"]
        sequence = row["SEQUENCE"]

        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            skipped += 1
            continue

        if label == 1:
            valid_sequences.append(sequence)
            valid_labels.append(label)
            valid_smiles.append(smiles)
        elif label == 0:
            valid_sequences.append(sequence)
            valid_labels.append(label)
            valid_smiles.append(smiles)
        else:
            skipped += 1

    valid_sequences = np.array(valid_sequences)
    valid_labels = np.array(valid_labels)
    valid_smiles = np.array(valid_smiles)

    n_toxic = (valid_labels == 1).sum()
    n_nontoxic = (valid_labels == 0).sum()

    print(f"  Validnih unosa:     {len(valid_sequences)}")
    print(f"    Toksicni  (1):    {n_toxic}")
    print(f"    Netoksicni (0):   {n_nontoxic}")
    print(f"  Preskoceno:         {skipped}")

    # ------------------------------------------------------------------
    # 3. Verifikacija protiv GNN pickle-a
    # ------------------------------------------------------------------
    gnn_pickle_file = os.path.join(
        PROJECT_DIR, "inflated_models", "large_layers_overtrained_peptide.pkl"
    )

    if os.path.exists(gnn_pickle_file):
        print(f"\n  Verifikacija s GNN pickle-om: {gnn_pickle_file}")
        with open(gnn_pickle_file, "rb") as f:
            gnn_data = pickle.load(f)
        gnn_labels = np.array(gnn_data["labels"])
        print(f"    GNN pickle ima {len(gnn_labels)} unosa")
        print(f"    SVM pipeline ima {len(valid_labels)} unosa")

        if len(gnn_labels) == len(valid_labels):
            if np.array_equal(gnn_labels, valid_labels):
                print("    [OK] Labele se potpuno podudaraju!")
            else:
                print("    [UPOZORENJE] Labele se NE podudaraju! Provjerite pipeline.")
        else:
            print("    [UPOZORENJE] Razlicit broj unosa! Provjerite pipeline.")
    else:
        print(f"\n  [INFO] GNN pickle nije pronadjen za verifikaciju: {gnn_pickle_file}")

    # ------------------------------------------------------------------
    # 4. Ucitavanje GNN CV splitova
    # ------------------------------------------------------------------
    print(f"\n  Ucitavanje GNN CV splitova: {GNN_SPLITS_FILE}")

    if not os.path.exists(GNN_SPLITS_FILE):
        print(f"  GRESKA: Datoteka '{GNN_SPLITS_FILE}' nije pronadjena!")
        sys.exit(1)

    with open(GNN_SPLITS_FILE, "rb") as f:
        cv_splits = pickle.load(f)

    print(f"  Ucitano {len(cv_splits)} foldova")

    # Prikaz velicina foldova
    for i, split in enumerate(cv_splits, 1):
        train_n = len(split["train_idx"])
        val_n = len(split["val_idx"])
        test_n = len(split["test_idx"])
        print(f"    Fold {i:2d}: train={train_n}, val={val_n}, test={test_n}")

    # Provjera da indeksi ne prelaze velicinu dataseta
    max_idx = max(
        max(split["train_idx"].max(), split["val_idx"].max(), split["test_idx"].max())
        for split in cv_splits
    )
    print(f"\n  Maksimalni indeks u splitovima: {max_idx}")
    print(f"  Velicina dataseta:              {len(valid_sequences)}")

    if max_idx >= len(valid_sequences):
        print("  [GRESKA] Indeksi prelaze velicinu dataseta!")
        sys.exit(1)
    else:
        print("  [OK] Svi indeksi su unutar granica.")

    # ------------------------------------------------------------------
    # 5. Spremanje uskladjenih podataka
    # ------------------------------------------------------------------
    aligned_data = {
        "fasta_sequences": valid_sequences,
        "labels": valid_labels,
        "smiles": valid_smiles,
        "cv_splits": cv_splits,
        "n_folds": len(cv_splits),
    }

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(aligned_data, f)

    print(f"\n  Uskladjeni podaci spremljeni u: {OUTPUT_FILE}")
    print(f"  Sadrzaj:")
    print(f"    fasta_sequences: {valid_sequences.shape}")
    print(f"    labels:          {valid_labels.shape}")
    print(f"    smiles:          {valid_smiles.shape}")
    print(f"    cv_splits:       {len(cv_splits)} foldova")

    # Prikaz primjera
    print(f"\n  Primjeri (prvih 3):")
    for i in range(min(3, len(valid_sequences))):
        print(f"    [{i}] FASTA={valid_sequences[i][:30]}...  LABEL={valid_labels[i]}")

    print("\n[DONE] Uskladjeni splitovi kreirani uspjesno.")


if __name__ == "__main__":
    main()
