# Tables and Figures for Scientific Paper
## Transfer Learning Between Peptide and Small Molecule Toxicity Prediction

---

## ESSENTIAL TABLES (3 tables)

### Table 1: Dataset and Model Overview
**Purpose:** Combine dataset info + model configurations in one compact table

| Property | Peptide Dataset | Small Molecule Dataset |
|----------|-----------------|------------------------|
| Source | ToxinSequenceSMILES | MolToxPredDataset |
| Samples | ? | ? |
| Toxic / Non-toxic | ? / ? | ? / ? |
| Unique Elements | ? | ? |

| Model Size | Layer Sizes | Parameters |
|------------|-------------|------------|
| Standard | [25, 25, 25, 1] | ? |
| Large Layers | [125, 125, 125, 1] | ? |
| Inflated | [512, 256, 128, 1] | ? |
| Extra Inflated | [1024, 512, 256, 1] | ? |

*Note: All models use k=25 (SortPooling) and filters [16, 32, 128]*

---

### Table 2: Transfer Learning Performance Results
**Purpose:** THE main results table - shows all methods, both directions, key metrics

| Direction | Method | ROC-AUC | MCC | F1 |
|-----------|--------|---------|-----|-----|
| **SMT → Peptide** | Baseline | | | |
| | Freeze GNN | | | |
| | Freeze Readout | | | |
| | Freeze All | | | |
| | Gradual Unfreezing | | | |
| **Peptide → SMT** | Baseline | | | |
| | Freeze GNN | | | |
| | Freeze Readout | | | |
| | Freeze All | | | |
| | Gradual Unfreezing | | | |

*Values shown as mean±std over 10-fold CV. Bold = best per direction. * = p<0.05 vs baseline (Nemenyi test)*

---

### Table 3: Model Capacity Effect on Transfer
**Purpose:** Key finding - does bigger model = better transfer?

| Model Size | SMT→Peptide Δ | Peptide→SMT Δ | Transfer Effective? |
|------------|---------------|---------------|---------------------|
| Standard | | | Yes/No |
| Large Layers | | | Yes/No |
| Inflated | | | Yes/No |
| Extra Inflated | | | Yes/No |

*Δ = Best TL method ROC-AUC minus Baseline ROC-AUC*

---

## ESSENTIAL FIGURES (4 figures)

### Figure 1: Method Overview (Architecture + Transfer Strategies)
**Purpose:** Single figure explaining the approach
- **(a)** DeepGraphCNN architecture diagram (simplified)
- **(b)** Four transfer strategies shown as layer diagrams (frozen vs trainable)

*Create with draw.io or similar. This replaces detailed text explanation.*

---

### Figure 2: Box Plots - Main Performance Comparison
**Purpose:** Visual comparison of all methods
- **Content:** 2 subplots (one per transfer direction)
- **Metric:** ROC-AUC (or MCC - pick one primary metric)
- **Already exists:** `box_whisker_roc_auc_plot.png` in eval directories

*Combine both directions into single figure with (a) and (b) panels*

---

### Figure 3: ROC Curves Comparison
**Purpose:** Standard classification performance visualization
- **Content:** 2 subplots (one per transfer direction)
- **Shows:** All 5 methods with AUC in legend
- **Already exists:** `comparison_roc_curves.png`

---

### Figure 4: Model Capacity vs Transfer Effectiveness
**Purpose:** KEY FINDING visualization
- **Content:** Line/bar chart showing Δ performance across model sizes
- **X-axis:** Model capacity (Standard → Extra Inflated)
- **Y-axis:** Performance improvement from transfer
- **Lines/Bars:** Different transfer methods or directions

*This is your main contribution visualization - create this carefully*

---

## OPTIONAL (if space permits)

### Optional Table: Friedman Test Summary
*Only if reviewers ask for more statistical detail*
| Metric | χ² | p-value |
|--------|-----|---------|
| ROC-AUC | | |
| MCC | | |

### Optional Figure: Confusion Matrix or Nemenyi Heatmap
*Move to supplementary materials if needed*

---

## WHAT TO PUT IN TEXT (not tables/figures)

1. **Transfer methods description** - Brief paragraph explaining the 4 strategies
2. **Hyperparameters** - One sentence: "Models trained for max 10000 epochs with early stopping (patience=7), Adam optimizer (lr=1e-4), batch size 32"
3. **Statistical tests** - "Friedman test with Nemenyi post-hoc (α=0.05)"
4. **Detailed per-fold results** - Reference to supplementary materials

---

## LAYOUT SUGGESTION FOR 6-PAGE PAPER

| Section | Content | Space |
|---------|---------|-------|
| Abstract | - | 0.25 page |
| Introduction | Background, motivation, contribution | 0.75 page |
| Related Work | Brief literature review | 0.5 page |
| Methods | Figure 1 + brief text | 1 page |
| Experiments | Table 1 (datasets/models) | 0.5 page |
| Results | Table 2 + Figure 2 + Figure 3 | 1.5 pages |
| Discussion | Table 3 + Figure 4 (key finding) | 1 page |
| Conclusion | Summary | 0.25 page |
| References | - | 0.25 page |

---

## GENERATION CHECKLIST

- [ ] Fill Table 1 with actual dataset statistics
- [ ] Run evaluations and fill Table 2 with results
- [ ] Calculate Δ values for Table 3
- [ ] Create Figure 1 (architecture diagram) - manual creation needed
- [ ] Combine existing box plots into Figure 2
- [ ] Combine existing ROC curves into Figure 3  
- [ ] Create Figure 4 (capacity effect) - new plot needed

---

## KEY NARRATIVE (for writing)

**Research Question:** Can transfer learning improve toxicity prediction across molecular domains (peptide ↔ small molecule)?

**Main Findings to Highlight:**
1. Which transfer method works best for each direction
2. Whether transfer is symmetric (A→B vs B→A)
3. How model capacity affects transfer effectiveness
4. Statistical significance of improvements

**One-sentence conclusion template:**
"Transfer learning from [source] to [target] using [best method] achieved [X]% improvement in ROC-AUC (p<0.05), with [larger/smaller] models showing [more/less] effective transfer."
