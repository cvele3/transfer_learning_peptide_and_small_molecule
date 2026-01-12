# filepath: c:\_OTHERS_\faks\4. semestar\git diplomski\transfer_learning_peptide_and_small_molecule\datasets\find_all_elements.py
import pandas as pd
from rdkit import Chem

files = ["ToxinSequenceSMILES.xlsx", "MolToxPredDataset.xlsx"] # Ili "out.xlsx"
all_elements = set()

for f in files:
    try:
        df = pd.read_excel(f)
        col = "SMILES" if "SMILES" in df.columns else "smiles"
        print(f"Obrađujem {f}...")
        for s in df[col]:
            mol = Chem.MolFromSmiles(str(s))
            if mol:
                for atom in mol.GetAtoms():
                    all_elements.add(atom.GetSymbol())
    except Exception as e:
        print(f"Greška s {f}: {e}")
        
print("-" * 30)
print("KONAČNI RJEČNIK ZA OBA SKUPA PODATAKA:")
sorted_elems = sorted(list(all_elements))
print("{")
for i, el in enumerate(sorted_elems):
    print(f'    "{el}": {i},')
print("}")
print(f"NUM_FEATURES = {len(sorted_elems)}")