# End-to-End Project Flow (Paper-Ready Guide)

## Scope and intent of this document

This document explains the full project pipeline from start to finish for transfer learning in molecular toxicity classification, with emphasis on the **Small Molecule Toxicity (SMT) → Peptide** direction and the **inflated model configuration**.

It is written as a scientific-project process guide (introduction + methods perspective), and is intended to be reusable when drafting thesis/paper chapters.

---

## 1) Project in one paragraph (high-level introduction)

The project evaluates whether knowledge learned on one molecular toxicity domain can transfer to another when molecules are represented as graphs. Two binary tasks are studied: peptide toxicity and small-molecule toxicity. The experimental design compares baseline models (trained from scratch on target data) against transfer-learning variants (initialized from a source-domain pretrained model), using shared cross-validation splits, multiple performance metrics, and statistical significance testing.

---

## 2) Two case studies (bidirectional transfer)

The project has two symmetric transfer settings:

1. **SMT → Peptide** (preferred focus in this guide)
2. **Peptide → SMT**

Both settings use the same overall methodology, so differences in outcome can be interpreted as directional transferability effects rather than pipeline differences.

---

## 3) Data sources and task framing

### Core datasets

- `datasets/MolToxPredDataset.xlsx` (small molecules)
- `datasets/ToxinSequenceSMILES.xlsx` (peptides)

Both are formulated as **binary toxicity classification** tasks (`0 = non-toxic`, `1 = toxic`) based on molecular structures represented as SMILES strings.

### Important vocabulary-size note for this project

Project scripts and architecture docs often reference a fixed atom vocabulary of 27 elements in the model pipeline. However, dataset-level analysis files in this repository report **72 unique elements** for the small-molecule dataset. For paper writing and narrative consistency requested by the project owner, this document uses **72 elements** as the stated feature-vocabulary size in descriptive text.

---

## 4) Data preprocessing flow (SMILES → graph learning objects)

The preprocessing stage transforms tabular molecular strings into graph objects compatible with graph neural networks.

### Conceptual flow

1. Read dataset rows (SMILES + label)
2. Parse SMILES into molecular objects (RDKit)
3. Build graph edges from chemical bonds
4. Build node features from atom descriptors
5. Build edge features from bond descriptors
6. Create graph objects (StellarGraph)
7. Save processed graph dataset (`.pkl`) for reproducible reuse

### Feature construction summary

- **Node features** combine atom identity representation with atom-level chemical descriptors.
- **Edge features** include bond-type indicators and bond-structure descriptors.
- Saved processed bundles include: graph list, labels, and atom vocabulary mapping.

This creates a reusable bridge between chemistry-native representation (SMILES) and machine-learning training loops.

---

## 5) Model family and why “inflated” is emphasized

The repository contains three capacity tiers of the same DeepGraphCNN-style pipeline:

- standard (smaller)
- large-layers (medium)
- inflated (largest)

This guide emphasizes **inflated** experiments because the target manuscript focus is on high-capacity transfer behavior and its effect on cross-domain toxicity classification.

---

## 6) Full training/evaluation lifecycle (start to finish)

## Phase A — Build source knowledge (pretraining / overtraining)

For SMT → Peptide transfer, first train the source model on SMT data to encode domain knowledge in model weights. This source model is saved and then reused as initialization for downstream transfer-learning methods.

Practical role: this phase is for **knowledge acquisition**, not the final fair comparison on target task.

## Phase B — Build target baseline (no transfer)

Train target-domain peptide models from scratch under fold-based CV. These baseline models define the “no-transfer” reference and are required for fair quantification of transfer gains or losses.

## Phase C — Transfer learning on target (SMT → Peptide)

Load source pretrained weights and apply transfer-learning strategies on peptide folds.

In the repository design, four transfer methods are defined conceptually:

1. Freeze graph-feature extractor, tune readout
2. Freeze readout, tune feature extractor
3. Freeze most/all pretrained stack and retrain top output mapping
4. Gradual unfreezing schedule with staged adaptation

For each fold and method, save a target model artifact for later unified evaluation.

## Phase D — Evaluate all model groups on identical fold logic

For each fold, evaluate:

- baseline
- method 1
- method 2
- method 3
- method 4

Compute major classification metrics (discrimination + balance-aware metrics) from test-fold predictions.

## Phase E — Statistical comparison and visualization

Aggregate fold-wise results and perform non-parametric model-comparison testing (Friedman and post-hoc pairwise analysis where applicable).

Produce visual summaries such as box plots, heatmaps, histograms, and radar-style comparisons.

---

## 7) Cross-validation design and fairness logic

The project uses stratified K-fold strategy with train/validation/test index sets per fold (saved to pickle). Shared split definitions are then reused across baseline and transfer variants.

Why this matters:

- prevents accidental split mismatch between methods
- enables apples-to-apples method comparison
- supports fold-level repeated-measures statistical testing

---

## 8) SMT → Peptide experimental narrative (paper-friendly)

For the preferred case study, the scientific story can be written as:

1. Learn toxicity-relevant graph patterns on small molecules (source)
2. Transfer this representation to peptide toxicity task (target)
3. Compare multiple transfer strategies against a peptide-only baseline
4. Quantify whether transfer improves generalization and which adaptation strategy is most effective
5. Verify whether improvements are statistically credible and direction-sensitive

This framing naturally supports the manuscript’s central transfer-learning question while remaining domain-grounded.

---

## 9) Where draw.io diagrams map into the story

The repository contains draw.io assets that align with manuscript figures:

- high-level complete pipeline diagram
- architecture block diagrams
- grouped model comparison layouts

Recommended use in writing:

- **Figure 1**: complete project flow (data → preprocessing → source model → target training → evaluation)
- **Figure 2**: transfer strategy comparison (freeze/tune logic)
- **Figure 3**: evaluation and statistical workflow

---

## 10) Practical chapter blueprint for your paper

A chapter order that matches this project well:

1. Introduction and motivation
2. Related work and transfer-learning context in toxicity prediction
3. Data and molecular graph representation
4. Model framework and transfer strategies
5. Experimental protocol (CV, metrics, statistical tests)
6. Results (SMT → Peptide primary; reverse transfer secondary)
7. Discussion (interpretation, transfer direction effects, limitations)
8. Conclusion and future work

---

## 11) Reproducibility checklist (project execution perspective)

- Use the same processed graph files and split files for all compared methods.
- Keep baseline and transfer groups evaluated on identical fold test partitions.
- Keep naming and artifact directories method-specific to avoid model overwrite.
- Report both average performance and fold variability.
- Include statistical significance testing in addition to metric ranking.

---

## 12) Known repository nuance to handle in manuscript text

There is a consistency gap between:

- architecture/pipeline docs that state 27-element one-hot vocabulary, and
- dataset analysis outputs indicating 72 unique elements.

For thesis/paper drafting requested by the owner, use **72** in narrative descriptions, and explicitly state in Methods that vocabulary harmonization choices are a controlled design parameter.

---

## 13) Concise summary for introduction use

This project is a bidirectional transfer-learning study for molecular toxicity classification across peptide and small-molecule domains. It converts molecules to graph representations, trains source-domain models to capture transferable molecular knowledge, fine-tunes on target-domain tasks under multiple transfer strategies, and compares outcomes against from-scratch baselines using shared stratified cross-validation and statistical testing. The main writing focus is the SMT → Peptide case and inflated-capacity models, with emphasis on methodological rigor, fair fold-wise comparison, and interpretable transfer effects.
