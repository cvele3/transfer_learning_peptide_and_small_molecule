"""
Dataset Analysis Script

Analyzes MolToxPredDataset.xlsx and ToxinSequenceSMILES.xlsx datasets,
producing a summary table with toxicity counts, unique elements, and graph statistics.
"""

import pandas as pd
from rdkit import Chem
import os

def analyze_dataset(filepath, smiles_col, label_col, dataset_name):
    """
    Analyze a single dataset and return statistics.
    
    Parameters:
        filepath: Path to the Excel file
        smiles_col: Name of the SMILES column
        label_col: Name of the label column
        dataset_name: Name to identify the dataset
    
    Returns:
        Dictionary with dataset statistics
    """
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return None
    
    print(f"Loading {filepath}...")
    df = pd.read_excel(filepath, header=0, usecols=[smiles_col, label_col])
    
    # Initialize counters
    toxic_count = 0
    non_toxic_count = 0
    unique_elements = set()
    graph_lengths = []  # Number of atoms (nodes) per molecule
    valid_smiles = 0
    invalid_smiles = 0
    
    print(f"Analyzing {dataset_name}...")
    
    for _, row in df.iterrows():
        smiles = str(row[smiles_col])
        label = row[label_col]
        
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            invalid_smiles += 1
            continue
        
        valid_smiles += 1
        
        # Count toxicity labels
        if label == 1:
            toxic_count += 1
        elif label == 0:
            non_toxic_count += 1
        
        # Get graph length (number of atoms/nodes)
        num_atoms = mol.GetNumAtoms()
        graph_lengths.append(num_atoms)
        
        # Collect unique elements
        for atom in mol.GetAtoms():
            unique_elements.add(atom.GetSymbol())
    
    # Calculate statistics
    if graph_lengths:
        avg_graph_length = sum(graph_lengths) / len(graph_lengths)
        min_graph_length = min(graph_lengths)
        max_graph_length = max(graph_lengths)
    else:
        avg_graph_length = 0
        min_graph_length = 0
        max_graph_length = 0
    
    # Calculate percentages for ratio
    total = toxic_count + non_toxic_count
    toxic_pct = (toxic_count / total * 100) if total > 0 else 0
    non_toxic_pct = (non_toxic_count / total * 100) if total > 0 else 0
    ratio_str = f"{toxic_pct:.0f}% / {non_toxic_pct:.0f}%"
    
    print(f"  Valid SMILES: {valid_smiles}, Invalid SMILES: {invalid_smiles}")
    
    return {
        "Dataset": dataset_name,
        "Toxic (1)": toxic_count,
        "Non-Toxic (0)": non_toxic_count,
        "Ratio (T/NT)": ratio_str,
        "Unique Elements": len(unique_elements),
        "Avg Graph Len": round(avg_graph_length, 2),
        "Min Graph Len": min_graph_length,
        "Max Graph Len": max_graph_length
    }


def main():
    print("=" * 80)
    print("Dataset Analysis")
    print("=" * 80)
    
    # Define datasets to analyze
    datasets = [
        {
            "filepath": "MolToxPredDataset.xlsx",
            "smiles_col": "SMILES",
            "label_col": "Toxicity",
            "name": "MolToxPredDataset (Small Molecules)"
        },
        {
            "filepath": "ToxinSequenceSMILES.xlsx",
            "smiles_col": "SMILES",
            "label_col": "TOXICITY",
            "name": "ToxinSequenceSMILES (Peptides)"
        }
    ]
    
    results = []
    
    for ds in datasets:
        print("\n" + "-" * 50)
        stats = analyze_dataset(
            filepath=ds["filepath"],
            smiles_col=ds["smiles_col"],
            label_col=ds["label_col"],
            dataset_name=ds["name"]
        )
        if stats:
            results.append(stats)
    
    # Create summary DataFrame
    if results:
        print("\n" + "=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80)
        
        summary_df = pd.DataFrame(results)
        
        # Display the table with nice formatting
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        
        print("\n" + summary_df.to_string(index=False))
        
        # Also print as markdown table for easy copy-paste
        print("\n" + "-" * 80)
        print("MARKDOWN TABLE FORMAT:")
        print("-" * 80)
        try:
            print(summary_df.to_markdown(index=False))
        except ImportError:
            # tabulate not installed - print simple alternative
            print("| " + " | ".join(summary_df.columns) + " |")
            print("| " + " | ".join(["---"] * len(summary_df.columns)) + " |")
            for _, row in summary_df.iterrows():
                print("| " + " | ".join(str(v) for v in row) + " |")
        
        # Save to Excel for reference
        output_file = "dataset_analysis_results.xlsx"
        summary_df.to_excel(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
    else:
        print("\nNo datasets were successfully analyzed.")


if __name__ == "__main__":
    main()
