# Graph Size Analysis - 72 Features

## Summary Table

| Dataset | Total Graphs | Max Graph Size | Avg Graph Size | Min Graph Size | Median Graph Size | Std Dev |
| --- | --- | --- | --- | --- | --- | --- |
| Peptide graphs (72 features) | 5398 | 333.0 | 164.33 | 26.0 | 163.0 | 67.37 |
| Small molecule graphs (72 features) | 10446 | 293.0 | 21.11 | 1.0 | 19.0 | 14.37 |

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
