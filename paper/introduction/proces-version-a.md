# End-to-End Project Flow (Paper-Oriented): Transfer Learning for Toxicity Prediction

## 1) Project Context and Core Research Design

This repository studies **cross-domain transfer learning** for molecular toxicity classification.

The project is built around **two case studies**:

1. **Small Molecule Toxicity (SMT) → Peptide toxicity**
2. **Peptide toxicity → Small Molecule Toxicity (SMT)**

Both are binary classification tasks (toxic vs non-toxic), and both represent molecules as graphs.

For paper writing, the central framing is:

- Can learned molecular toxicity knowledge transfer across related but distinct chemical domains?
- Which transfer strategy works best vs training from scratch?
- Is the transfer direction important?

---

## 2) Datasets and Labeling

Two XLSX datasets are used throughout the project:

- `datasets/MolToxPredDataset.xlsx` (small molecules)
- `datasets/ToxinSequenceSMILES.xlsx` (peptides)

Columns used are SMILES + toxicity label (0/1), with column names adjusted per dataset.

### Class balance and composition (as documented in repo outputs)

From repository analysis outputs:

- **SMT dataset**: 4,616 toxic and 5,833 non-toxic samples
- **Peptide dataset**: 1,805 toxic and 3,593 non-toxic samples

This class imbalance is one reason the project uses multiple metrics (not only accuracy).

---

## 3) Important Correction for Paper Consistency: Feature Vocabulary Size

Some older scripts/docs mention a fixed **27-element one-hot vocabulary**.

For your paper-level narrative, the corrected and currently relevant statement should be:

- The broader chemistry scope across datasets supports a **72-element space** (as reported in repository dataset analysis for the small-molecule domain).
- Therefore, in the manuscript narrative, reference the feature space as aligned to the full observed chemistry support (72), not the older reduced 27-element statement.

This avoids a mismatch between early prototype assumptions and expanded dataset analysis.

---

## 4) Data Preprocessing Pipeline (SMILES → Graph)

The preprocessing stage transforms each molecule from text notation to graph data structures that DeepGraphCNN can consume.

## 4.1 Input parsing

1. Read rows from XLSX.
2. Parse each SMILES string with RDKit (`Chem.MolFromSmiles`).
3. Skip invalid molecules.

## 4.2 Graph construction

For each valid molecule:

- **Nodes** = atoms
- **Edges** = bonds in both directions (undirected representation encoded as directed pairs)

## 4.3 Node features

Each atom receives a feature vector made from:

- one-hot element indicator
- additional chemical descriptors (e.g., atomic number, degree, charge, hybridization, aromatic flag)

## 4.4 Edge features

Each bond receives feature information such as:

- bond type category (single/double/triple/aromatic)
- conjugation
- ring membership
- stereochemistry encoding

## 4.5 Object packaging

Node/edge tables are wrapped into `StellarGraph` objects and saved into `.pkl` files for reuse.

This caching is critical because graph construction is expensive and reused by overtraining, baseline, TL, and evaluation scripts.

---

## 5) Model Family and Why Focus on Inflated

The repository implements three size families:

- Normal (`[25,25,25,1]`)
- Large layers (`[125,125,125,1]`)
- Inflated (`[512,256,128,1]`)

For your requested focus, the paper narrative centers on the **inflated family**, which is the highest-capacity variant in your experiments.

At architecture level, all families follow the same conceptual design:

1. **DeepGraphCNN block** (graph feature extraction)
2. **Readout/classifier head** (Conv1D + pooling + dense layers + sigmoid output)

So the scientific question is not “different architecture type,” but “same architecture paradigm with different capacity and transfer strategy.”

---

## 6) Complete Training/Evaluation Lifecycle

The project lifecycle can be described in four phases.

## Phase A — Source overtraining (knowledge acquisition)

Goal:
Create a source model that captures as much domain signal as possible before transfer.

Process:

1. Load all source-domain graphs.
2. Split train/validation.
3. Train with early stopping and high epoch ceiling.
4. Save source model weights (`.h5`).

Interpretation for paper:
This is **representation acquisition**, not final fairness benchmarking.

## Phase B — Baseline (target-domain from scratch)

Goal:
Establish no-transfer reference performance on target dataset.

Process:

1. Build fresh target model with random initialization.
2. Use stratified 10-fold splits.
3. Train/evaluate per fold.
4. Save per-fold baseline models.

Interpretation:
Baseline quantifies what target data alone can do without transferred knowledge.

## Phase C — Transfer learning methods (target-domain fine-tuning)

Goal:
Compare alternative hypotheses on what should transfer from source to target.

Methods implemented in repository design:

1. **Freeze GNN, train readout**
2. **Freeze readout, train GNN**
3. **Freeze all, add new output layer**
4. **Gradual unfreezing in phases**

All methods use the same CV splits as baseline for fair comparisons.

## Phase D — Evaluation, statistics, and reporting

Goal:
Assess performance robustness and significance.

Per fold, each model variant is scored with:

- ROC-AUC
- Geometric Mean (GM)
- Precision
- Recall
- F1
- MCC

Then across folds:

- Friedman test checks if model ranks differ significantly.
- Nemenyi post-hoc identifies pairwise differences.

Outputs include summary tables and visualizations (boxplots, heatmaps, histograms, radar-style comparisons).

---

## 7) Focused Walkthrough: SMT → Peptide (Inflated)

This is the preferred case-study direction for your manuscript narrative.

## 7.1 Source side (SMT)

- Build/load SMT graph dataset.
- Train inflated source model on SMT.
- Save source checkpoint for transfer.

## 7.2 Target side (Peptide)

- Load peptide graph dataset.
- Load peptide CV splits.
- Train peptide baseline from scratch (fold-wise).

## 7.3 Transfer stage

- Load SMT inflated pretrained model.
- Apply selected TL strategy.
- Fine-tune on peptide train folds.
- Evaluate on peptide test folds.
- Save TL fold models.

## 7.4 Comparison logic

For each fold, compare:

- baseline vs each TL method

Then aggregate across folds and perform statistical testing.

This produces the evidence needed for paper claims about whether transfer helps, by how much, and under which strategy.

---

## 8) Cross-Validation Design and Fairness Guarantees

The repository design uses stratified 10-fold CV with train/validation/test partition logic per fold.

Why this matters scientifically:

1. **Class balance preservation** across folds.
2. **Repeated evaluation** instead of one split.
3. **Shared splits** across baseline and TL methods (apples-to-apples comparison).

Without shared folds, method differences could reflect data-split luck rather than transfer effects.

---

## 9) What the Draw.io and Architecture Docs Add

The `models_architecture/` and `draw_io/` materials provide two benefits:

1. **Conceptual decomposition** of pipeline stages (data, source training, transfer, evaluation).
2. **Visual communication templates** suitable for thesis/paper figures.

In manuscript writing, these assets can support:

- one high-level pipeline figure
- one architecture figure
- one TL-method comparison panel

This strongly improves readability and reproducibility.

---

## 10) Suggested Paper Framing (Single-Author Voice)

A concise problem statement style aligned with your goals:

> This study investigates whether toxicity-related knowledge learned in one molecular domain can improve prediction in another domain through graph-based transfer learning. Two directional case studies are evaluated: small-molecule to peptide transfer and peptide to small-molecule transfer. The pipeline standardizes molecular representation, applies controlled cross-validated training protocols, and compares baseline learning against multiple transfer strategies under identical splits and statistical testing.

---

## 11) Recommended Chapter Skeleton for Your Manuscript

1. **Introduction**
   - Motivation, problem, contribution scope (two case studies).
2. **Related Work**
   - Toxicity prediction and transfer learning context.
3. **Materials and Methods**
   - Datasets, graph preprocessing, model family, transfer strategies, CV protocol, metrics/statistics.
4. **Results**
   - SMT→Peptide and Peptide→SMT, baseline vs TL, fold-aggregated metrics, significance tests.
5. **Discussion**
   - Transfer direction effects, strategy behavior, implications.
6. **Limitations and Future Work**
   - Domain shift boundaries, data size, feature-space harmonization.
7. **Conclusion**
   - Main scientific takeaways.

---

## 12) Practical Notes for Future Repository Hygiene

To prevent confusion between historical and current settings:

- Explicitly version feature-space definitions (e.g., `FEATURE_SCHEMA_VERSION`).
- Add one canonical config file for element vocabulary size and mapping.
- Update old comments/docs that still state 27 if current intended narrative is 72.

This will make the paper-methods section cleaner and reduce reviewer concerns about reproducibility consistency.

---

## 13) One-Page Process Summary (Quick Read)

- Start from two labeled SMILES datasets.
- Convert each molecule into graph + node/edge features.
- Train source model (overtraining) to learn transferable representations.
- Train target baseline from scratch with 10-fold CV.
- Transfer source model to target and fine-tune using four TL strategies.
- Evaluate all models on the same folds with robust metrics.
- Apply Friedman + Nemenyi tests for significance.
- Compare both transfer directions.
- Report findings with plots/tables and clear methodological controls.

