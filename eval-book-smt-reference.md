# Eval Book SMT - Reference Table

Note: Layer size naming convention:
- [25, 25, 25, 1] = Standard Overtrained Models (Extra Features)
- [512, 256, 128, 1] = Inflated Models
- [125, 125, 125, 1] = Large Layers Models

| Veličine GNN slojeva | Modeli | ROC-AUC | GM | Precision | Recall | F1 | MCC |
|----------------------|--------|---------|-----|-----------|--------|-----|-----|
| [25, 25, 25, 1] | Baseline | | | | | | |
| | Method 1 - freeze gnn | | | | | | |
| | Method 2 - freeze readout | | | | | | |
| | Method 3 - freeze all | | | | | | |
| | Method 4 - gradual unfreezing | | | | | | |
| | Friedman p-value | | | | | | |
| [125, 125, 125, 1] | Baseline | | | | | | |
| | Method 1 - freeze gnn | | | | | | |
| | Method 2 - freeze readout | | | | | | |
| | Method 3 - freeze all | | | | | | |
| | Method 4 - gradual unfreezing | | | | | | |
| | Friedman p-value | | | | | | |
| [512, 256, 128, 1] | Baseline | | | | | | |
| | Method 1 - freeze gnn | | | | | | |
| | Method 2 - freeze readout | | | | | | |
| | Method 3 - freeze all | | | | | | |
| | Method 4 - gradual unfreezing | | | | | | |
| | Friedman p-value | | | | | | |
