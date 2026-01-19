# Tables and Plots for Scientific Paper
## Transfer Learning from Peptide to Small Molecule Toxicity Prediction (and Vice Versa)

---

## SECTION 1: TABLES

### Table 1: Dataset Characteristics
**Purpose:** Describe the two datasets used in the study
| Dataset | Domain | #Samples | #Positive (Toxic) | #Negative (Non-toxic) | Class Ratio | Source |
|---------|--------|----------|-------------------|----------------------|-------------|--------|
| ToxinSequenceSMILES | Peptide Toxicity | ? | ? | ? | ? | Reference |
| MolToxPredDataset | Small Molecule Toxicity | ? | ? | ? | ? | Reference |

*Fill in from your data. Include molecular representation (SMILES→Graph), unique elements found, average graph size.*

---

### Table 2: Model Architecture Configurations
**Purpose:** Detail the different model sizes tested for transfer learning
| Configuration | k (SortPooling) | Layer Sizes | Filter Sizes | Total Parameters |
|--------------|-----------------|-------------|--------------|------------------|
| Standard Overtrained | 25 | [25, 25, 25, 1] | [16, 32, 128] | ? |
| Large Layers | 25 | [125, 125, 125, 1] | [16, 32, 128] | ? |
| Inflated | 25 | [512, 256, 128, 1] | [16, 32, 128] | ? |
| Extra Inflated | 25 | [1024, 512, 256, 1] | [16, 32, 128] | ? |

---

### Table 3: Transfer Learning Methods Description
**Purpose:** Explain each transfer learning strategy tested
| Method | Strategy | Frozen Layers | Trainable Layers | Learning Rate |
|--------|----------|---------------|------------------|---------------|
| Baseline | No transfer (trained from scratch) | None | All | 1e-4 |
| Method 1 (Freeze GNN) | Freeze graph convolution layers | DeepGraphCNN, GraphConv | Dense, Conv1D, Dropout | 1e-4 |
| Method 2 (Freeze Readout) | Freeze classification head | Dense, Dropout, Flatten | GNN layers | 1e-5 |
| Method 3 (Freeze All) | Freeze all + new output layer | All original layers | New Dense output | 1e-4 |
| Method 4 (Gradual Unfreezing) | Progressive unfreezing with discriminative LR | Phased unfreezing | Phase 1→2→3 | 1e-3→1e-4→1e-5 |

---

### Table 4: Performance Comparison - Small Molecule → Peptide Transfer
**Purpose:** Main results table for SMT→Peptide direction
| Model | ROC-AUC (mean±std) | MCC (mean±std) | F1 (mean±std) | GM (mean±std) | Precision (mean±std) | Recall (mean±std) |
|-------|-------------------|----------------|---------------|---------------|---------------------|------------------|
| Baseline | | | | | | |
| Freeze GNN | | | | | | |
| Freeze Readout | | | | | | |
| Freeze All | | | | | | |
| Gradual Unfreezing | | | | | | |

*Highlight best performing method. Use bold for statistically significant improvements.*

---

### Table 5: Performance Comparison - Peptide → Small Molecule Transfer
**Purpose:** Main results table for Peptide→SMT direction
| Model | ROC-AUC (mean±std) | MCC (mean±std) | F1 (mean±std) | GM (mean±std) | Precision (mean±std) | Recall (mean±std) |
|-------|-------------------|----------------|---------------|---------------|---------------------|------------------|
| Baseline | | | | | | |
| Freeze GNN | | | | | | |
| Freeze Readout | | | | | | |
| Freeze All | | | | | | |
| Gradual Unfreezing | | | | | | |

---

### Table 6: Impact of Model Capacity on Transfer Learning (Peptide Target)
**Purpose:** Compare transfer effectiveness across model sizes
| Model Size | Baseline ROC-AUC | Best TL Method | Best TL ROC-AUC | Δ ROC-AUC | Transfer Effect |
|------------|------------------|----------------|-----------------|-----------|-----------------|
| Standard | | | | | Positive/Negative/Neutral |
| Large Layers | | | | | |
| Inflated | | | | | |
| Extra Inflated | | | | | |

---

### Table 7: Impact of Model Capacity on Transfer Learning (Small Molecule Target)
**Purpose:** Compare transfer effectiveness across model sizes for SMT domain
| Model Size | Baseline ROC-AUC | Best TL Method | Best TL ROC-AUC | Δ ROC-AUC | Transfer Effect |
|------------|------------------|----------------|-----------------|-----------|-----------------|
| Standard | | | | | Positive/Negative/Neutral |
| Large Layers | | | | | |
| Inflated | | | | | |
| Extra Inflated | | | | | |

---

### Table 8: Statistical Significance - Friedman Test Results
**Purpose:** Show which metrics showed statistically significant differences
| Metric | Friedman χ² | p-value | Significant (α=0.05) |
|--------|-------------|---------|---------------------|
| ROC-AUC | | | Yes/No |
| MCC | | | |
| F1 | | | |
| GM | | | |
| Precision | | | |
| Recall | | | |

---

### Table 9: Nemenyi Post-hoc Test Results (for significant metrics)
**Purpose:** Pairwise comparisons between methods
| Comparison | ROC-AUC p-value | MCC p-value | F1 p-value |
|------------|----------------|-------------|------------|
| Baseline vs Freeze GNN | | | |
| Baseline vs Freeze Readout | | | |
| Baseline vs Freeze All | | | |
| Baseline vs Gradual Unfreezing | | | |
| Freeze GNN vs Freeze Readout | | | |
| ... | | | |

*Bold p-values < 0.05*

---

### Table 10: Summary of Transfer Learning Effectiveness
**Purpose:** High-level summary table for conclusions
| Transfer Direction | Best Method | Avg. Improvement | Recommendation |
|-------------------|-------------|------------------|----------------|
| Small Molecule → Peptide | | | Effective/Ineffective |
| Peptide → Small Molecule | | | Effective/Ineffective |

---

## SECTION 2: FIGURES/PLOTS

### Figure 1: Molecular Graph Representation
**Purpose:** Illustrate how molecules are converted to graphs
- **Content:** Side-by-side showing:
  - (a) Example SMILES string
  - (b) 2D molecular structure (RDKit Draw)
  - (c) Graph representation with nodes (atoms) and edges (bonds)
  - (d) Node feature matrix visualization
- **Generation:** Use RDKit for molecule drawing + NetworkX for graph visualization

---

### Figure 2: DeepGraphCNN Architecture Diagram
**Purpose:** Explain the model architecture
- **Content:** Neural network diagram showing:
  - Input (graph with node features)
  - Graph convolution layers
  - SortPooling layer
  - 1D CNN layers
  - Dense layers
  - Output (binary classification)
- **Generation:** Create with draw.io, Lucidchart, or TikZ

---

### Figure 3: Transfer Learning Strategies Visualization
**Purpose:** Visual explanation of the 4 transfer methods
- **Content:** Four diagrams showing which layers are frozen (blue) vs trainable (orange):
  - (a) Freeze GNN
  - (b) Freeze Readout
  - (c) Freeze All + New Head
  - (d) Gradual Unfreezing (3 phases)
- **Generation:** Create with diagram software

---

### Figure 4: Dataset Class Distribution
**Purpose:** Show class balance in both datasets
- **Content:** Bar charts showing:
  - (a) Peptide dataset: Toxic vs Non-toxic counts
  - (b) Small molecule dataset: Toxic vs Non-toxic counts
- **Note:** Important for understanding potential class imbalance issues

---

### Figure 5: Box-and-Whisker Plots - Metric Comparison (SMT → Peptide)
**Purpose:** Visualize metric distributions across methods
- **Content:** 6 subplots (one per metric: ROC-AUC, MCC, F1, GM, Precision, Recall)
- **Already generated:** `box_whisker_*.png` files in eval directories
- **Note:** Consider creating a combined figure with all metrics

---

### Figure 6: Box-and-Whisker Plots - Metric Comparison (Peptide → SMT)
**Purpose:** Same as Figure 5 but for opposite transfer direction

---

### Figure 7: ROC Curve Comparison
**Purpose:** Compare classification performance visually
- **Content:** Multiple ROC curves on same plot:
  - All 5 methods with mean AUC ± std in legend
  - Diagonal reference line (random classifier)
- **Already generated:** `comparison_roc_curves.png`
- **Generate for:** Both transfer directions

---

### Figure 8: Precision-Recall Curve Comparison
**Purpose:** Important for imbalanced datasets
- **Content:** PR curves for all methods with Average Precision scores
- **Already generated:** `comparison_pr_curves.png`
- **Generate for:** Both transfer directions

---

### Figure 9: Confusion Matrices Grid
**Purpose:** Show prediction patterns for each method
- **Content:** 5 confusion matrices (one per method) arranged in grid
- **Already generated:** Individual `confusion_matrix_*.png` files
- **Recommendation:** Combine into single figure with subplots

---

### Figure 10: Nemenyi Post-hoc Test Heatmaps
**Purpose:** Visualize pairwise statistical significance
- **Content:** Heatmap showing p-values between method pairs
- **Already generated:** `heatmap_*.png` files
- **Select:** Most important metrics (ROC-AUC, MCC, F1)

---

### Figure 11: Radar/Spider Chart - Model Performance Fingerprint
**Purpose:** Multi-metric comparison in single view
- **Content:** Radar chart with axes for each metric
- **Already generated:** `advanced_radar_chart.png`
- **Generate for:** Both transfer directions

---

### Figure 12: Model Capacity vs Transfer Effectiveness
**Purpose:** Key finding - how model size affects transfer
- **Content:** Line plot showing:
  - X-axis: Model capacity (Standard → Large → Inflated → Extra Inflated)
  - Y-axis: Δ Performance (Transfer - Baseline)
  - Multiple lines for different metrics or methods
- **Important:** This could be a key finding figure

---

### Figure 13: Violin Plots - Performance Distribution
**Purpose:** Show distribution shape, not just summary statistics
- **Already generated:** `advanced_violin_plots.png`
- **Better than boxplots:** Shows multimodality if present

---

### Figure 14: Calibration Curves (Reliability Diagram)
**Purpose:** Assess prediction confidence calibration
- **Already generated:** `advanced_calibration_curve.png`
- **Important for:** Understanding if predicted probabilities are reliable

---

### Figure 15: Training History Comparison
**Purpose:** Show convergence patterns
- **Content:** 
  - (a) Loss curves during training
  - (b) Validation loss comparison between methods
- **Generate from:** Training histories (if saved)

---

### Figure 16: t-SNE Embedding Visualization
**Purpose:** Visualize learned representations
- **Content:** 2D scatter plot of penultimate layer activations
- **Color by:** True toxicity label
- **Compare:** Baseline vs best transfer method
- **Code exists:** `thesis_plots.py`

---

### Figure 17: Feature Importance / Attention Visualization (Optional)
**Purpose:** Interpretability - what features matter
- **If applicable:** GNN attention weights on molecular graph

---

### Figure 18: Cross-Domain Transfer Comparison Summary
**Purpose:** Final summary visualization
- **Content:** Grouped bar chart comparing:
  - SMT → Peptide improvements
  - Peptide → SMT improvements
  - Across all model sizes and methods

---

## SECTION 3: SUPPLEMENTARY MATERIALS

### Supplementary Table S1: Per-Fold Results
**Purpose:** Full detailed results for reproducibility
- All metrics for all 10 folds, all methods, all model sizes

### Supplementary Table S2: Hyperparameters
**Purpose:** Reproducibility
- Learning rates, batch sizes, epochs, early stopping patience

### Supplementary Table S3: Element/Atom Types in Datasets
**Purpose:** Dataset characteristics
- Unique elements found in each dataset (from your analyze_*_elements.py)

### Supplementary Figure S1: All Box-Whisker Plots (Full Collection)
### Supplementary Figure S2: All Heatmaps (Full Collection)
### Supplementary Figure S3: Histograms of Predictions

---

## GENERATION PRIORITY

### High Priority (Main Paper):
1. Table 1 (Dataset characteristics)
2. Table 3 (Transfer methods description)
3. Table 4 & 5 (Main performance results)
4. Table 8 & 9 (Statistical tests)
5. Figure 2 (Architecture diagram)
6. Figure 3 (Transfer strategies diagram)
7. Figure 5 & 6 (Box plots)
8. Figure 7 (ROC curves)
9. Figure 12 (Model capacity effect - KEY FINDING)

### Medium Priority (Results section):
1. Table 6 & 7 (Model capacity comparison)
2. Figure 9 (Confusion matrices)
3. Figure 10 (Nemenyi heatmaps)
4. Figure 11 (Radar charts)

### Lower Priority (Supplementary):
1. Tables S1-S3
2. Figure 14-17

---

## NOTES

1. **Existing plots location:**
   - `extra_features_eval_peptide/` - Peptide evaluation plots
   - `extra_features_eval_small_mol_tox/` - Small molecule toxicity evaluation plots
   - `inflated_eval_peptide/` - Inflated model peptide evaluation plots
   - `inflated_eval_small_mol_tox/` - Inflated model SMT evaluation plots
   - `large_layers_eval_peptide/` - Large layers peptide evaluation plots

2. **Scripts to generate plots:**
   - `generate_advanced_plots.py` - Radar, calibration, violin
   - `generate_extra_plots.py` - ROC, PR curves, confusion matrices
   - `thesis_plots.py` - t-SNE, basic plots
   - Evaluation scripts contain boxplot and heatmap generation

3. **Key narrative for paper:**
   - Research question: Does transfer learning help across molecular domains (peptide ↔ small molecule)?
   - What model capacity is needed for effective transfer?
   - Which transfer strategy works best?
   - Is transfer symmetric (A→B same as B→A)?

4. **Statistical rigor:**
   - 10-fold cross-validation
   - Friedman test for overall significance
   - Nemenyi post-hoc for pairwise comparisons
   - Report mean ± std for all metrics
