"""
Priprema DPC-SVM dataseta.

Koraci:
  1. Preuzima FASTA sekvence s ToxinPred (pozitivni i negativni skup).
  2. Parsira FASTA zapise i izdvaja sekvence.
  3. Pretvara svaku sekvencu u SMILES koristeći RDKit (MolFromSequence).
  4. Kanonikalizira SMILES.
  5. Učitava postojeći ToxinSequenceSMILES.xlsx i kanonikalizira njegove SMILES-e.
  6. Filtrira: zadržava samo zapise s podudaranjem kanonskog SMILES-a.
  7. Uklanja duplikate.
  8. Sprema novu XLSX datoteku sa stupcima: FASTA, SMILES, ACTIVITY.
"""

import os
import sys
import requests
import pandas as pd
from rdkit import Chem

# ========================== KONFIGURACIJA ==========================

POS_URL = "https://webs.iiitd.edu.in/raghava/toxinpred/datasets/pos-maindataset-1"
NEG_URL = "https://webs.iiitd.edu.in/raghava/toxinpred/datasets/neg-maindataset-1"

EXISTING_DATASET = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "ToxinSequenceSMILES.xlsx"
)
OUTPUT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "DPC_SVM_dataset.xlsx"
)

# 20 standardnih aminokiselina (jednoslovne oznake)
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


# ========================== POMOĆNE FUNKCIJE ==========================

def download_fasta(url):
    """
    Preuzima podatke s URL-a i parsira FASTA format.
    Podržava:
      - Čisti FASTA (linije s '>' su zaglavlja)
      - Običan tekst (jedna sekvenca po retku)
    Vraća listu sekvenci.
    """
    print(f"  Preuzimanje s {url} ...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    raw_text = response.text.strip()
    lines = raw_text.split("\n")

    sequences = []
    current_seq = ""
    has_headers = any(line.strip().startswith(">") for line in lines)

    if has_headers:
        # FASTA format
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_seq:
                    sequences.append(current_seq)
                    current_seq = ""
            else:
                current_seq += line.upper()
        if current_seq:
            sequences.append(current_seq)
    else:
        # Obični tekst – jedna sekvenca po retku
        for line in lines:
            line = line.strip().upper()
            if line:
                sequences.append(line)

    print(f"    Parsirano {len(sequences)} sekvenci.")
    return sequences


def is_valid_peptide(sequence):
    """Provjerava sadrži li sekvenca samo 20 standardnih aminokiselina."""
    return all(aa in STANDARD_AA for aa in sequence) and len(sequence) >= 2


def sequence_to_canonical_smiles(sequence):
    """
    Pretvara peptidnu sekvencu u kanonikalizirani SMILES
    koristeći RDKit-ov MolFromSequence (flavor=0 = L-aminokiseline).
    """
    try:
        mol = Chem.MolFromSequence(sequence, sanitize=True, flavor=0)
        if mol is None:
            return None
        smiles = Chem.MolToSmiles(mol)
        return smiles
    except Exception:
        return None


def canonicalize_smiles(smiles):
    """Kanonikalizira zadani SMILES string."""
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


# ========================== GLAVNI TOK ==========================

def main():
    # ------------------------------------------------------------------
    # Korak 1: Preuzimanje FASTA sekvenci
    # ------------------------------------------------------------------
    print("=" * 65)
    print("Korak 1: Preuzimanje FASTA sekvenci s ToxinPred")
    print("=" * 65)

    pos_sequences = download_fasta(POS_URL)
    neg_sequences = download_fasta(NEG_URL)

    print(f"\n  Pozitivnih (toksičnih) sekvenci:   {len(pos_sequences)}")
    print(f"  Negativnih (netoksičnih) sekvenci: {len(neg_sequences)}")

    # ------------------------------------------------------------------
    # Korak 2: Filtriranje nestandardnih sekvenci i konverzija u SMILES
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Korak 2: Filtriranje i konverzija FASTA -> SMILES")
    print("=" * 65)

    records = []

    def process_sequences(seqs, activity_label, label_name):
        converted = 0
        skipped_invalid_aa = 0
        skipped_conversion = 0

        for seq in seqs:
            if not is_valid_peptide(seq):
                skipped_invalid_aa += 1
                continue
            smiles = sequence_to_canonical_smiles(seq)
            if smiles is None:
                skipped_conversion += 1
                continue
            records.append({
                "FASTA": seq,
                "SMILES": smiles,
                "ACTIVITY": activity_label,
            })
            converted += 1

        print(f"  {label_name}:")
        print(f"    Konvertirano:                 {converted}")
        print(f"    Preskočeno (nestandardne AA): {skipped_invalid_aa}")
        print(f"    Preskočeno (konverzija):      {skipped_conversion}")

    process_sequences(pos_sequences, activity_label=1, label_name="Pozitivni (toksični)")
    process_sequences(neg_sequences, activity_label=0, label_name="Negativni (netoksični)")

    if not records:
        print("\nGREŠKA: Nijedna sekvenca nije uspješno konvertirana!")
        sys.exit(1)

    converted_df = pd.DataFrame(records)
    print(f"\n  Ukupno konvertiranih zapisa: {len(converted_df)}")

    # ------------------------------------------------------------------
    # Korak 3: Mapiranje i filtriranje prema postojećem datasetu
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Korak 3: Usporedba s postojećim peptide datasetom")
    print("=" * 65)

    if not os.path.exists(EXISTING_DATASET):
        print(f"  UPOZORENJE: Datoteka '{EXISTING_DATASET}' nije pronađena!")
        print("  Spremam sve konvertirane zapise bez filtriranja ...")
        output_df = converted_df[["FASTA", "SMILES", "ACTIVITY"]].copy()
    else:
        existing_df = pd.read_excel(EXISTING_DATASET)
        print(f"  Učitano zapisa iz postojećeg dataseta: {len(existing_df)}")
        print(f"  Stupci: {list(existing_df.columns)}")

        # Kanonikalizacija SMILES-a u postojećem skupu
        print("  Kanonikalizacija SMILES-a u postojećem skupu...")
        existing_df["CANONICAL_SMILES"] = existing_df["SMILES"].apply(canonicalize_smiles)
        existing_df = existing_df.dropna(subset=["CANONICAL_SMILES"])

        existing_canonical_set = set(existing_df["CANONICAL_SMILES"].values)
        print(f"  Jedinstvenih kanonskih SMILES-a u postojećem skupu: {len(existing_canonical_set)}")

        # Filtriranje: zadržavamo samo zapise s podudarajućim SMILES-om
        mask = converted_df["SMILES"].isin(existing_canonical_set)
        matched_df = converted_df[mask].copy()

        print(f"\n  Podudrani zapisi:    {matched_df.shape[0]}")
        print(f"    - Toksični  (1):   {(matched_df['ACTIVITY'] == 1).sum()}")
        print(f"    - Netoksični (0):  {(matched_df['ACTIVITY'] == 0).sum()}")
        print(f"  Nepodudrani zapisi:  {(~mask).sum()}")

        output_df = matched_df[["FASTA", "SMILES", "ACTIVITY"]].copy()

    # ------------------------------------------------------------------
    # Korak 4: Uklanjanje duplikata
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Korak 4: Uklanjanje duplikata")
    print("=" * 65)

    before_dedup = len(output_df)
    output_df = output_df.drop_duplicates(subset=["SMILES"])
    after_dedup = len(output_df)
    print(f"  Prije:   {before_dedup}")
    print(f"  Poslije: {after_dedup}")
    print(f"  Uklonjeno duplikata: {before_dedup - after_dedup}")

    # ------------------------------------------------------------------
    # Korak 5: Spremanje XLSX datoteke
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Korak 5: Spremanje konačnog dataseta")
    print("=" * 65)

    output_df.reset_index(drop=True, inplace=True)
    output_df.to_excel(OUTPUT_FILE, index=False)

    print(f"  Datoteka:  {OUTPUT_FILE}")
    print(f"  Ukupno zapisa:       {len(output_df)}")
    print(f"    ACTIVITY=1 (toksično):    {(output_df['ACTIVITY'] == 1).sum()}")
    print(f"    ACTIVITY=0 (netoksično):  {(output_df['ACTIVITY'] == 0).sum()}")

    print("\n  Prvih 5 zapisa:")
    print(output_df.head(5).to_string(index=False))

    print("\n[DONE] Priprema dataseta zavrsena.")


if __name__ == "__main__":
    main()
