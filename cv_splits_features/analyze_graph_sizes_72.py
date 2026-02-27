"""
Analiza veličina grafova u overtrained 72 features datotekama.

Za peptide i male molekule izračunava:
  - Najveću veličinu grafa (broj čvorova)
  - Prosječnu veličinu grafa
  - Najmanju veličinu grafa

Sprema rezultate u Excel i Markdown format.
"""

import os
import pickle
import numpy as np
import pandas as pd
from stellargraph import StellarGraph

# ========================== KONFIGURACIJA ==========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Input files
PEPTIDE_FILE = os.path.join(SCRIPT_DIR, "overtrained_peptide_graphs_72_features.pkl")
SMALL_MOL_FILE = os.path.join(SCRIPT_DIR, "overtrained_small_molecule_graphs_72_features.pkl")


# ========================== ANALIZA ==========================

def analyze_graph_sizes(filepath, dataset_name):
    """
    Analizira veličine grafova u datoteci.
    
    Args:
        filepath: Putanja do pkl datoteke
        dataset_name: Ime dataseta (za printanje)
    
    Returns:
        Dictionary s statistikama ili None ako datoteka ne postoji
    """
    if not os.path.exists(filepath):
        print(f"  GRESKA: Datoteka '{filepath}' nije pronadjena!")
        return None
    
    print(f"\n{'-' * 70}")
    print(f"  Analiza: {dataset_name}")
    print(f"  Datoteka: {os.path.basename(filepath)}")
    print(f"{'-' * 70}")
    
    # Učitavanje podataka
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    
    graphs = data["graphs"]
    print(f"  Ukupan broj grafova: {len(graphs)}")
    
    # Izračunavanje veličina (broj čvorova)
    graph_sizes = []
    for graph in graphs:
        num_nodes = graph.number_of_nodes()
        graph_sizes.append(num_nodes)
    
    graph_sizes = np.array(graph_sizes)
    
    # Statistike
    max_size = np.max(graph_sizes)
    avg_size = np.mean(graph_sizes)
    min_size = np.min(graph_sizes)
    median_size = np.median(graph_sizes)
    std_size = np.std(graph_sizes)
    
    stats = {
        "max": max_size,
        "avg": avg_size,
        "min": min_size,
        "median": median_size,
        "std": std_size,
        "total_graphs": len(graphs)
    }
    
    print(f"\n  Statistike veličina grafova (broj čvorova):")
    print(f"    Najveća:     {max_size:.1f}")
    print(f"    Prosječna:  {avg_size:.2f}")
    print(f"    Medijan:    {median_size:.1f}")
    print(f"    Najmanja:   {min_size:.1f}")
    print(f"    Std dev:    {std_size:.2f}")
    
    return stats


# ========================== SPREMANJE REZULTATA ==========================

def save_results_to_excel(peptide_stats, small_mol_stats, output_file):
    """
    Sprema rezultate u Excel datoteku.
    
    Args:
        peptide_stats: Dictionary s statistikama za peptide
        small_mol_stats: Dictionary s statistikama za small molecules
        output_file: Putanja do Excel datoteke
    """
    data = []
    
    if peptide_stats:
        data.append({
            "Dataset": "Peptide graphs (72 features)",
            "Total Graphs": int(peptide_stats['total_graphs']),
            "Max Graph Size": f"{peptide_stats['max']:.1f}",
            "Avg Graph Size": f"{peptide_stats['avg']:.2f}",
            "Min Graph Size": f"{peptide_stats['min']:.1f}",
            "Median Graph Size": f"{peptide_stats['median']:.1f}",
            "Std Dev": f"{peptide_stats['std']:.2f}"
        })
    
    if small_mol_stats:
        data.append({
            "Dataset": "Small molecule graphs (72 features)",
            "Total Graphs": int(small_mol_stats['total_graphs']),
            "Max Graph Size": f"{small_mol_stats['max']:.1f}",
            "Avg Graph Size": f"{small_mol_stats['avg']:.2f}",
            "Min Graph Size": f"{small_mol_stats['min']:.1f}",
            "Median Graph Size": f"{small_mol_stats['median']:.1f}",
            "Std Dev": f"{small_mol_stats['std']:.2f}"
        })
    
    df = pd.DataFrame(data)
    df.to_excel(output_file, index=False)
    print(f"\n  Rezultati spremljeni u Excel: {output_file}")


def save_results_to_markdown(peptide_stats, small_mol_stats, output_file):
    """
    Generira Markdown datoteku s rezultatima.
    
    Args:
        peptide_stats: Dictionary s statistikama za peptide
        small_mol_stats: Dictionary s statistikama za small molecules
        output_file: Putanja do Markdown datoteke
    """
    md_content = """# Graph Size Analysis - 72 Features

## Summary Table

"""
    
    # Kreiraj tablicu
    headers = ["Dataset", "Total Graphs", "Max Graph Size", "Avg Graph Size", "Min Graph Size", "Median Graph Size", "Std Dev"]
    md_content += "| " + " | ".join(headers) + " |\n"
    md_content += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    
    if peptide_stats:
        row = [
            "Peptide graphs (72 features)",
            str(int(peptide_stats['total_graphs'])),
            f"{peptide_stats['max']:.1f}",
            f"{peptide_stats['avg']:.2f}",
            f"{peptide_stats['min']:.1f}",
            f"{peptide_stats['median']:.1f}",
            f"{peptide_stats['std']:.2f}"
        ]
        md_content += "| " + " | ".join(row) + " |\n"
    
    if small_mol_stats:
        row = [
            "Small molecule graphs (72 features)",
            str(int(small_mol_stats['total_graphs'])),
            f"{small_mol_stats['max']:.1f}",
            f"{small_mol_stats['avg']:.2f}",
            f"{small_mol_stats['min']:.1f}",
            f"{small_mol_stats['median']:.1f}",
            f"{small_mol_stats['std']:.2f}"
        ]
        md_content += "| " + " | ".join(row) + " |\n"
    
    # Dodaj opise
    md_content += """
## Column Descriptions

| Column | Description |
|--------|-------------|
| Dataset | Name of the dataset (72 features version) |
| Total Graphs | Total number of graphs in the dataset |
| Max Graph Size | Largest graph size (number of nodes/atoms) |
| Avg Graph Size | Average graph size (number of nodes/atoms) |
| Min Graph Size | Smallest graph size (number of nodes/atoms) |
| Median Graph Size | Median graph size (number of nodes/atoms) |
| Std Dev | Standard deviation of graph sizes |

## Notes

- **Graph Size** refers to the number of nodes (atoms) in the molecular graph representation
- Each atom in a molecule becomes a node in the graph
- Each chemical bond becomes an edge (in both directions for undirected graphs)
- Analysis performed on overtrained graphs with 72-element vocabulary
"""
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    print(f"  Rezultati spremljeni u Markdown: {output_file}")


# ========================== GLAVNI TOK ==========================

def main():
    print("=" * 70)
    print("  ANALIZA VELIČINA GRAFOVA - 72 FEATURES")
    print("=" * 70)
    
    # Analiza peptide grafova
    peptide_stats = analyze_graph_sizes(
        PEPTIDE_FILE,
        "Peptide graphs"
    )
    
    # Analiza small molecule grafova
    small_mol_stats = analyze_graph_sizes(
        SMALL_MOL_FILE,
        "Small molecule graphs"
    )
    
    # Sumarni prikaz
    print(f"\n{'=' * 70}")
    print("  SUMARNI PRIKAZ")
    print(f"{'=' * 70}")
    
    if peptide_stats:
        print(f"\n  Peptide graphs:")
        print(f"    Najveća:    {peptide_stats['max']:.1f} čvorova")
        print(f"    Prosječna:  {peptide_stats['avg']:.2f} čvorova")
        print(f"    Najmanja:   {peptide_stats['min']:.1f} čvorova")
    
    if small_mol_stats:
        print(f"\n  Small molecule graphs:")
        print(f"    Najveća:    {small_mol_stats['max']:.1f} čvorova")
        print(f"    Prosječna:  {small_mol_stats['avg']:.2f} čvorova")
        print(f"    Najmanja:   {small_mol_stats['min']:.1f} čvorova")
    
    # Spremanje rezultata u Excel i Markdown
    excel_file = os.path.join(SCRIPT_DIR, "graph_sizes_72_analysis.xlsx")
    md_file = os.path.join(SCRIPT_DIR, "graph_sizes_72_analysis.md")
    
    if peptide_stats or small_mol_stats:
        save_results_to_excel(peptide_stats, small_mol_stats, excel_file)
        save_results_to_markdown(peptide_stats, small_mol_stats, md_file)
    
    print("\n[DONE]")


if __name__ == "__main__":
    main()
