# End-to-End Project Flow (Inflated Focus, SMT → Peptide)

## 1) Project purpose and scientific framing

This project is a transfer-learning study in molecular toxicity prediction with two case studies:

1. **Small molecule → peptide transfer learning** (preferred focus in this document)
2. **Peptide → small molecule transfer learning**

In both cases, the task is **binary graph classification**: toxic (`1`) vs non-toxic (`0`). The overall goal is to test whether learned toxicity-relevant graph representations from one molecular domain improve performance in the other domain.

---

## 2) Problem setup in practical terms

- Input data in both domains starts as **SMILES strings** plus toxicity labels.
- Each molecule is transformed into a graph where:
  - atoms = nodes
  - bonds = edges
- A shared modeling pipeline is used so that baseline and transfer-learning variants can be compared fairly under the same split protocol and metrics.

---

## 3) Data assets used by the project

Core datasets:

- `datasets/MolToxPredDataset.xlsx` (small molecules)
- `datasets/ToxinSequenceSMILES.xlsx` (peptides)

The repository also includes dataset-analysis outputs indicating class distributions, graph lengths, and unique element counts. In that analysis summary, small molecules report **72 unique elements**.

> **Important note for this document:** this write-up uses **72** as the active one-hot vocabulary size for your paper-facing explanation, per your instruction.

---

## 4) Preprocessing pipeline (SMILES → graph tensors)

### Step 4.1: Read and validate molecular records

- Read tabular data from Excel.
- Extract `(SMILES, label)` pairs.
- Parse each SMILES with RDKit.
- Skip invalid parses.

### Step 4.2: Build graph topology

For each valid molecule:

- Enumerate bonds.
- Add edges in both directions (to model undirected chemistry in directed edge lists).
- Keep bond-level descriptors (bond type + additional bond properties where used).

### Step 4.3: Build node features

For each atom:

- Build one-hot element encoding using shared vocabulary (**paper-facing: 72**).
- Concatenate extra atom descriptors used by the scripts (atomic number, degree, formal charge, hybridization mapping, aromatic flag).

### Step 4.4: Assemble graph objects

- Create node feature matrix and edge table.
- Build `StellarGraph` object per molecule.
- Save processed artifacts (`.pkl`) containing:
  - graphs
  - labels
  - graph_labels
  - element-to-index mapping

This stage creates reusable processed data for downstream overtraining, baseline, TL, and evaluation scripts.

---

## 5) Shared representation bridge for transfer learning

Transfer learning only works reliably if source and target models consume compatible input feature spaces. The project enforces this through a shared element-index mapping reused in both domains.

For your manuscript wording, this should be described as:

- “A single shared atom vocabulary is defined across domains and used consistently in graph construction, enabling dimensional compatibility of transferred weights.”

---

## 6) Core model family and the inflated focus

All experiments share the same conceptual architecture family:

1. Graph convolution block (feature extraction from molecular graph structure)
2. Sort-pooling to fixed-size representation
3. 1D-conv/readout block
4. Binary output head (sigmoid)

The project has multiple capacity variants; this document focuses on the **inflated** line for manuscript narrative and results interpretation.

---

## 7) Training flow from start to finish

## Phase A: Source-model overtraining

For SMT → peptide TL, the source side is small molecules:

1. Load processed small-molecule graphs.
2. Train source model with early stopping and high epoch cap.
3. Save pretrained source weights/model (`.h5`).

Purpose:

- maximize source-domain knowledge extraction
- provide initialization for transfer strategies

## Phase B: Cross-validation split generation

- Generate and save stratified folds (train/val/test index sets).
- Reuse the *same fold definitions* across baseline and TL methods for fair comparison.

This is critical because fold mismatch would invalidate method comparisons.

## Phase C: Target baseline (no transfer)

For peptide target:

1. Initialize a fresh model (random weights).
2. Train on peptide train split.
3. Monitor validation split for early stopping.
4. Evaluate on held-out test split.
5. Repeat per fold and save fold-specific models.

Baseline answers: “What performance is achievable without transfer?”

## Phase D: Transfer learning (SMT → peptide)

For each fold:

1. Load pretrained SMT source model.
2. Apply a transfer strategy (layer freezing/unfreezing policy).
3. Fine-tune on peptide train/val of that fold.
4. Evaluate on peptide test split.
5. Save fold model by strategy.

Conceptually, project documentation defines four method families:

- Method 1: freeze graph encoder, tune higher layers
- Method 2: freeze readout/classifier block, tune graph encoder
- Method 3: freeze almost all and learn new output mapping
- Method 4: gradual unfreezing in phases with decaying learning rate

---

## 8) Evaluation and statistical testing pipeline

Per fold and per model group (baseline + TL methods):

1. Load fold model
2. Predict probabilities on test fold
3. Threshold to labels (0.5)
4. Compute metrics:
   - ROC-AUC
   - Geometric Mean (GM)
   - Precision
   - Recall
   - F1
   - MCC

Across folds:

- Aggregate mean/std metrics
- Run non-parametric significance tests (Friedman)
- Run post-hoc pairwise analysis (Nemenyi) when appropriate
- Export plots/tables (box-whisker, heatmaps, radar-style summaries)

This provides both practical performance ranking and statistical confidence for manuscript claims.

---

## 9) End-to-end reproducibility map (inflated SMT → peptide)

Recommended order to explain in the paper and to run in practice:

1. **Analyze datasets** and define shared representation policy.
2. **Build graph pickles** with consistent feature schema.
3. **Train overtrained source model** on SMT.
4. **Generate stratified CV splits** for peptide target.
5. **Train peptide baseline** across folds.
6. **Train SMT→peptide TL models** across same folds/methods.
7. **Evaluate all models** on held-out fold tests.
8. **Run statistical comparisons** and produce final visuals/tables.

---

## 10) How to describe this in Introduction (general, non-overly-technical)

A manuscript-safe high-level phrasing:

- “This work studies cross-domain transfer learning for toxicity classification by modeling molecules as graphs and transferring knowledge between small-molecule and peptide domains. The experimental design compares transfer strategies against non-transfer baselines under matched stratified cross-validation, and evaluates improvements through multi-metric and statistical analysis.”

This keeps the introduction conceptual while deferring architecture specifics to Methods.

---

## 11) Draw.io project diagrams and documentation usage

The repository includes architecture/process diagrams and guides in:

- `draw_io/`
- `models_architecture/`

These should be treated as:

- visual summary for chapter-level flow
- support material for explaining stage transitions (data → source model → transfer → evaluation)

---

## 12) Known consistency note to keep in writing

Some internal project documentation text may mention a smaller shared one-hot vocabulary value. For your requested manuscript-facing description in this iteration, maintain the statement that the active one-hot element vocabulary is **72**.

