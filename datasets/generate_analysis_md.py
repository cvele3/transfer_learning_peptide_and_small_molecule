"""
Generate Markdown file from dataset analysis results.

Reads dataset_analysis_results.xlsx and creates a formatted markdown file.
"""

import pandas as pd
import os

def main():
    input_file = "dataset_analysis_results.xlsx"
    output_file = "dataset_analysis_results.md"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run dataset_analysis.py first.")
        return
    
    print(f"Reading {input_file}...")
    df = pd.read_excel(input_file)
    
    # Build markdown content
    md_content = """# Dataset Analysis Results

## Summary Table

"""
    
    # Create markdown table header
    headers = list(df.columns)
    md_content += "| " + " | ".join(headers) + " |\n"
    md_content += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    
    # Add data rows
    for _, row in df.iterrows():
        md_content += "| " + " | ".join(str(v) for v in row) + " |\n"
    
    # Add additional info
    md_content += """
## Column Descriptions

| Column | Description |
|--------|-------------|
| Dataset | Name of the dataset |
| Toxic (1) | Number of toxic samples (label = 1) |
| Non-Toxic (0) | Number of non-toxic samples (label = 0) |
| Ratio (T/NT) | Percentage of toxic vs non-toxic samples |
| Unique Elements | Number of unique chemical elements in the dataset |
| Avg Graph Len | Average number of atoms (nodes) per molecule |
| Min Graph Len | Smallest molecule size (number of atoms) |
| Max Graph Len | Largest molecule size (number of atoms) |

## Notes

- **Graph Length** refers to the number of atoms (nodes) in the molecular graph representation
- Each atom in a molecule becomes a node in the graph
- Each chemical bond becomes an edge (in both directions for undirected graphs)
"""
    
    # Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    print(f"Markdown file saved to: {output_file}")
    print("\n--- Generated Content ---")
    print(md_content)


if __name__ == "__main__":
    main()
