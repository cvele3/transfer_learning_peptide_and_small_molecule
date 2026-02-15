# Dataset Analysis Results

## Summary Table

| Dataset | Toxic (1) | Non-Toxic (0) | Ratio (T/NT) | Unique Elements | Avg Graph Len | Min Graph Len | Max Graph Len |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MolToxPredDataset (Small Molecules) | 4616 | 5833 | 44% / 56% | 72 | 21.1 | 1 | 293 |
| ToxinSequenceSMILES (Peptides) | 1805 | 3593 | 33% / 67% | 4 | 164.33 | 26 | 333 |

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
