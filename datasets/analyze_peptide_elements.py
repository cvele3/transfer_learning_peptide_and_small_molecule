import pandas as pd
from rdkit import Chem
import os

# Naziv datoteke
FILENAME = "ToxinSequenceSMILES.xlsx"

def analyze_elements():
    if not os.path.exists(FILENAME):
        print(f"Greška: Datoteka '{FILENAME}' nije pronađena u mapi.")
        return

    print(f"Učitavam {FILENAME}...")
    try:
        # Učitavanje podataka (pretpostavka da je header u prvom retku)
        df = pd.read_excel(FILENAME)
        
        if "SMILES" not in df.columns:
            print("Greška: Stupac 'SMILES' nije pronađen u Excel datoteci.")
            return

        unique_elements = set()
        count_valid = 0
        count_invalid = 0

        print("Analiziram elemente u SMILES stupcu...")
        
        for smiles in df["SMILES"]:
            # Osiguraj da je string
            smiles_str = str(smiles)
            mol = Chem.MolFromSmiles(smiles_str)
            
            if mol is None:
                count_invalid += 1
                continue
            
            count_valid += 1
            for atom in mol.GetAtoms():
                unique_elements.add(atom.GetSymbol())

        # Sortiranje elemenata (abecedno ili prema atomskom broju, ovdje abecedno radi konzistencije rječnika)
        sorted_elements = sorted(list(unique_elements))
        
        # Kreiranje rječnika u stilu tvog koda
        element_to_index = {elem: i for i, elem in enumerate(sorted_elements)}

        print("-" * 50)
        print(f"Analiza završena.")
        print(f"Validnih SMILES-a: {count_valid}")
        print(f"Nevalidnih SMILES-a: {count_invalid}")
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