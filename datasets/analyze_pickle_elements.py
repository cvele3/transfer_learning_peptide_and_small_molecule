import pickle
import numpy as np
import os
import sys

# Pokušaj importa StellarGraph biblioteke
try:
    from stellargraph import StellarGraph
except ImportError:
    print("Greška: StellarGraph biblioteka nije pronađena.")
    sys.exit(1)

##############################################################
# 1. Definiranje putanje do pickle datoteke
# (Prilagođeno da radi iz 'datasets' mape prema 'large_layers_models')
##############################################################
PROCESSED_DATA_FILE = "../large_layers_models/large_layers_overtrained_small_mol_tox_pred.pkl"

def main():
    if os.path.exists(PROCESSED_DATA_FILE):
        print("Učitavam prethodno spremljene podatke iz:", PROCESSED_DATA_FILE)
        try:
            with open(PROCESSED_DATA_FILE, "rb") as f:
                processed_data = pickle.load(f)
            
            # Dohvaćanje ključnih podataka iz pickle rječnika
            graphs = processed_data["graphs"]
            element_to_index = processed_data["element_to_index"]
            # graphs_labels = processed_data["graph_labels"] # Nije obavezno za ovu analizu

            print("Podaci su uspješno učitani.")
            print(f"Broj učitanih grafova: {len(graphs)}")
            
            # --- POČETAK PROVJERE ELEMENATA ---
            vocab = list(element_to_index.keys())
            vocab_size = len(vocab) # npr. 27 ili 72
            print("-" * 50)
            print(f"Korišteni vokabular ({vocab_size} elemenata):")
            print(sorted(vocab))
            print("-" * 50)

            print("Analiziram čvorove grafova...")
            
            missing_atom_count = 0
            affected_graphs_count = 0
            
            for idx, g in enumerate(graphs):
                # Dohvati matrice značajki za sve čvorove u grafu
                node_features = g.node_features()
                
                # Značajke su obično [one-hot elementi, ostali deskriptori...]
                # Ovdje provjeravamo samo prvih 'vocab_size' stupaca koji odgovaraju elementima
                
                # Sigurnosna provjera dimenzija
                if node_features.shape[1] < vocab_size:
                    print(f"GRAF {idx}: Greška u dimenzijama! Ima {node_features.shape[1]} značajki, a očekuje se bar {vocab_size}.")
                    continue
                
                element_features = node_features[:, :vocab_size]
                
                # Suma po retku (za svaki atom). Ako je suma 0, znači da niti jedan bit za element nije aktivan.
                atom_sums = np.sum(element_features, axis=1)
                
                # Broj atoma gdje je suma 0 (tzv. "nula-vektori" u one-hot dijelu)
                zeros = np.sum(atom_sums == 0)
                
                if zeros > 0:
                    missing_atom_count += zeros
                    affected_graphs_count += 1
                    if affected_graphs_count <= 5: # Ispis samo prvih par primjera
                         print(f" -> Graf {idx} ima {zeros} atoma bez mapiranog elementa.")

            print("-" * 50)
            print("REZULTAT ANALIZE:")
            print("-" * 50)
            
            if missing_atom_count > 0:
                print(f"❌ DETEKTIRAN PROBLEM!")
                print(f"Ukupno {missing_atom_count} atoma nema postavljen one-hot indeks.")
                print(f"Problem se pojavljuje u {affected_graphs_count} molekula.")
                print("To znači da te molekule sadrže elemente koji NISU u tvom 'element_to_index' rječniku.")
                print("RJEŠENJE: Ažuriraj skriptu za kreiranje pickle-a s potpunim rječnikom elemenata.")
            else:
                print(f"✅ SVE U REDU.")
                print("Svaki atom u svim grafovima ima ispravno mapiran element.")

        except Exception as e:
            print(f"Došlo je do greške prilikom čitanja pickle datoteke: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Fallback poruka ako file ne postoji
        print(f"Datoteka ne postoji: {PROCESSED_DATA_FILE}")
        print("Provjeri nalaziš li se u mapi 'datasets' i je li putanja ispravna.")

if __name__ == "__main__":
    main()