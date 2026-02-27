# Transfer Learning Project Flow (Inflated, SMT → Peptide)

## 1) Project overview (paper-introduction level)

This project studies **cross-domain transfer learning for molecular toxicity classification** using graph neural networks.
The work is organized as **two case studies**:

1. **Small Molecule Toxicity (SMT) → Peptide Toxicity**
2. **Peptide Toxicity → Small Molecule Toxicity**

In both directions, the same scientific question is tested:

> Can a model pretrained on one molecular domain transfer useful toxicity knowledge to another domain, and outperform a target-domain baseline trained from scratch?

The target variable is binary in both domains:

- **Toxic** (1)
- **Non-toxic** (0)

Although the project supports three model-size families (normal, large-layers, inflated), this document focuses on the **inflated pipeline** and especially **SMT → Peptide transfer**.

---

## 2) Datasets and task framing

The project uses two principal XLSX datasets:

- **Peptide dataset** (`datasets/ToxinSequenceSMILES.xlsx`)
  - key columns: `SMILES`, `TOXICITY`
- **Small-molecule dataset** (`datasets/MolToxPredDataset.xlsx`)
  - key columns: `SMILES`, `Toxicity`

Both become graph-classification problems where each molecule is represented as a graph and each graph has one binary toxicity label.

---

## 3) Data preprocessing pipeline (start of the ML lifecycle)

### 3.1 Raw input

The pipeline starts from SMILES strings in Excel files.

### 3.2 Chemistry parsing

- RDKit parses each SMILES string into a molecular object.
- Invalid SMILES are skipped.

### 3.3 Graph construction

For each valid molecule:

- **Nodes** = atoms
- **Edges** = bonds (typically duplicated in both directions for undirected message passing)

### 3.4 Node features

Node-level feature vectors are built per atom. The project documentation and scripts describe one-hot element identity + extra atom descriptors (such as atom number/degree/hybridization/aromaticity style fields).

> Important writing preference for this project document:
> Use **72 one-hot element dimensions** as the effective element-space description.

### 3.5 Edge features

Bond-level features include bond-type category and additional structural indicators (e.g., ring/conjugation/stereo style descriptors), depending on script variant.

### 3.6 Persisted artifacts

Processed datasets are serialized as `.pkl` files, typically storing:

- graph objects
- labels
- graph label series
- element/index mapping metadata

These cached artifacts are reused by training/evaluation scripts to avoid repeating heavy preprocessing.

---

## 4) Core model concept (DeepGraphCNN family)

All experiments are built around a shared graph-classification backbone (DeepGraphCNN + readout/classifier head).

High-level decomposition:

1. **Graph feature extractor** (stacked graph convolutions + pooling)
2. **Readout/classifier** (1D conv/dense/dropout/sigmoid output pattern)

For the inflated setting, the GNN block uses the high-capacity layer configuration.

---

## 5) Full experimental lifecycle (end-to-end flow)

The project follows a staged process that is repeated for each direction and model-size family.

## Phase A — Build source knowledge (overtrained model)

1. Load source-domain processed graphs.
2. Build inflated graph model.
3. Train with large epoch ceiling + early stopping.
4. Save source model weights (`.h5`).

Purpose:

- Not primary fair benchmarking.
- Main goal is to pack transferable domain knowledge into weights before transfer.

## Phase B — Create shared CV splits

1. Use stratified K-fold logic (10-fold design).
2. Inside each fold, keep train/validation/test indexing logic.
3. Save split definitions to `.pkl`.
4. Reuse the same split definitions across baseline and transfer methods.

Purpose:

- Fold-level fairness.
- Paired comparison between methods on identical test partitions.

## Phase C — Train baseline on target domain (no transfer)

For each fold:

1. Initialize a new inflated model from scratch.
2. Train on target fold-train, monitor fold-validation.
3. Evaluate on fold-test.
4. Save fold model file.

Purpose:

- Defines the target-domain no-transfer reference.

## Phase D — Transfer learning training (four methods)

For each fold and method:

1. Load source pretrained model.
2. Apply a layer-freezing/unfreezing strategy.
3. Fine-tune on target fold train/val.
4. Evaluate on fold test.
5. Save fold model.

Methods used conceptually in the project:

- **Method 1**: Freeze graph extractor, tune readout/classifier.
- **Method 2**: Freeze readout, tune graph extractor.
- **Method 3**: Freeze all transferred layers, attach/train new output head.
- **Method 4**: Gradual unfreezing with staged learning-rate control.

## Phase E — Evaluation + statistical analysis

Across all folds and methods:

1. Load fold-matched models.
2. Generate probabilities on fold test sets.
3. Compute binary and threshold-free metrics.
4. Aggregate mean/std across folds.
5. Run significance testing.
6. Generate plots/tables.

Typical metrics tracked in this repo:

- ROC-AUC
- G-Mean (GM)
- Precision
- Recall
- F1
- MCC

Statistical layer in project docs/scripts includes Friedman + post-hoc comparison workflow.

---

## 6) Focused case study walkthrough: SMT → Peptide (Inflated)

This section describes the preferred direction in this project.

## Step 1 — Source domain preparation (SMT)

- Build/load processed small-molecule graphs.
- Train inflated source model on SMT.
- Save pretrained SMT model weights.

## Step 2 — Target domain preparation (Peptide)

- Build/load processed peptide graphs.
- Load peptide CV split definitions.

## Step 3 — Baseline training on peptide (inflated)

- Train fold-specific peptide baselines from random initialization.
- Save 10 fold checkpoints.

## Step 4 — Transfer fine-tuning from SMT to peptide

For each fold and method:

- Load SMT pretrained weights.
- Apply method-specific freeze policy.
- Fine-tune on peptide train/val subset.
- Evaluate on peptide test subset.
- Save method+fold checkpoints.

## Step 5 — Fold-wise comparison and reporting

- Compare each TL method against peptide baseline on identical test folds.
- Summarize distribution across folds.
- Use statistical testing to identify significant differences.

Interpretation target:

- Whether source-domain small-molecule signal improves peptide toxicity classification.
- Which transfer strategy is most stable/effective in this direction.

---

## 7) Project outputs and artifacts (what is produced)

The pipeline produces:

- **Processed graph caches** (`.pkl`)
- **CV split files** (`.pkl`)
- **Saved fold models** (`.h5`) for baseline and TL methods
- **Evaluation summaries** (console + saved summaries)
- **Figures** (box plots, histograms, heatmaps, radar-like comparison images)
- **Spreadsheet result books** (`eval-book-*.xlsx`)

This artifact layout supports reproducible paper tables/figures from fold-based evidence.

---

## 8) Draw.io architecture guidance (how diagrams map to pipeline)

The `draw_io/` and `models_architecture/` documentation aligns with this flow:

1. Data sources
2. Preprocessing
3. Source model training
4. Target baseline + transfer methods
5. 10-fold CV evaluation
6. Statistical testing and visualization

For paper figures, this enables:

- one high-level pipeline diagram
- one method-comparison diagram (freeze strategies)
- one focused SMT→Peptide workflow panel

---

## 9) Suggested paper chapter blueprint (aligned to your project)

A practical chapter order for writing:

1. **Introduction**
   - motivation for toxicity transfer learning across molecular domains
2. **Related Work**
   - molecular toxicity prediction + transfer learning in chem/bio tasks
3. **Problem Definition**
   - two domains, two transfer directions, binary labels
4. **Data and Graph Representation**
   - preprocessing and feature engineering overview
5. **Methodology**
   - model family + baseline + four TL methods
6. **Experimental Protocol**
   - fold design, training protocol, metrics, statistics
7. **Results**
   - direction-wise comparisons (focus SMT→Peptide first)
8. **Discussion**
   - transferability behavior, implications, failure modes
9. **Conclusion and Future Work**

---

## 10) Practical interpretation notes for your study

- The project is not only “which model gets best metric”, but also **which transfer hypothesis holds** (what should be frozen vs adapted).
- Bidirectional experiments test asymmetry: transfer may help more in one direction than the other.
- Fold-consistent baseline/TL comparisons are key for robust claims in the final manuscript.
- Statistics (not only means) should be central to final conclusions.

---

## 11) One-paragraph abstract-style summary (general)

This project investigates cross-domain transfer learning for molecular toxicity prediction by treating peptides and small molecules as graph-based binary classification domains. Molecules are converted from SMILES to graph representations with structured node and bond features, then modeled using a common DeepGraphCNN-style framework. A source model is first pretrained, after which multiple transfer strategies are evaluated against a target-domain baseline under shared stratified cross-validation splits. The full workflow includes preprocessing, source training, baseline benchmarking, transfer fine-tuning, fold-wise evaluation across multiple metrics, and statistical significance testing, producing a reproducible evidence pipeline for scientific reporting.

