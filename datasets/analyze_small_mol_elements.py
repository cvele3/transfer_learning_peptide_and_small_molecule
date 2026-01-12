import pandas as pd
from rdkit import Chem
import os
from collections import Counter

# Naziv datoteke - PROVJERI JE LI OVO TOČAN NAZIV TVOJE DATOTEKE
# U nekim mapama se zvala 'out.xlsx', u nekima 'MolToxPredDataset.xlsx'
FILENAME = "MolToxPredDataset.xlsx" 

def analyze_elements():
    if not os.path.exists(FILENAME):
        # Fallback provjera za out.xlsx ako primarna ne postoji
        if os.path.exists("out.xlsx"):
            print(f"Datoteka '{FILENAME}' nije nađena, ali 'out.xlsx' jest. Koristim nju.")
            actual_filename = "out.xlsx"
        else:
            print(f"Greška: Datoteka '{FILENAME}' (ni 'out.xlsx') nije pronađena u mapi.")
            return
    else:
        actual_filename = FILENAME

    print(f"Učitavam {actual_filename}...")
    try:
        df = pd.read_excel(actual_filename)
        
        # Provjera imena stupca
        if "SMILES" in df.columns:
            target_col = "SMILES"
        elif "smiles" in df.columns:
            target_col = "smiles"
        else:
            print(f"Greška: Stupac 'SMILES' nije pronađen. Dostupni stupci: {list(df.columns)}")
            return

        element_counts = Counter()
        count_valid = 0
        count_invalid = 0

        print("Analiziram elemente...")
        
        for smiles in df[target_col]:
            smiles_str = str(smiles)
            mol = Chem.MolFromSmiles(smiles_str)
            
            if mol is None:
                count_invalid += 1
                continue
            
            count_valid += 1
            for atom in mol.GetAtoms():
                element_counts[atom.GetSymbol()] += 1

        sorted_elements = sorted(list(element_counts.keys()))
        element_to_index = {elem: i for i, elem in enumerate(sorted_elements)}

        print("-" * 50)
        print(f"Analiza završena.")
        print(f"Validnih SMILES-a: {count_valid}")
        print(f"Nevalidnih SMILES-a: {count_invalid}")
        print("-" * 50)
        print("Frekvencija elemenata:")
        for elem in sorted_elements:
            print(f"{elem}: {element_counts[elem]}")
        print("-" * 50)
        print(f"Pronađeno je {len(sorted_elements)} jedinstvenih elemenata.")
        print("\nKopiraj ovaj rječnik u svoj kod:")
        print("element_to_index = {")
        for elem, idx in element_to_index.items():
            print(f'    "{elem}": {idx},')
        print("}")
        print(f"NUM_FEATURES = {len(sorted_elements)}")

    except Exception as e:
        print(f"Došlo je do greške prilikom obrade: {e}")

if __name__ == "__main__":
    analyze_elements()