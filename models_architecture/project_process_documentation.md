# Transfer Learning for Molecular Toxicity Prediction — Full Project Documentation

## Table of Contents

1. [Project Goal & High-Level Idea](#1-project-goal--high-level-idea)
2. [Datasets](#2-datasets)
3. [Data Preprocessing Pipeline — From XLSX to Graphs](#3-data-preprocessing-pipeline--from-xlsx-to-graphs)
4. [Shared Atom Vocabulary](#4-shared-atom-vocabulary)
5. [The Neural Network Architecture — DeepGraphCNN](#5-the-neural-network-architecture--deepgraphcnn)
6. [Phase 1: Overtraining the Source Model](#6-phase-1-overtraining-the-source-model)
7. [Cross-Validation Split Generation](#7-cross-validation-split-generation)
8. [Phase 2: Baseline Model Training (No Transfer Learning)](#8-phase-2-baseline-model-training-no-transfer-learning)
9. [Phase 3: Transfer Learning — Four Methods](#9-phase-3-transfer-learning--four-methods)
10. [Bidirectional Transfer](#10-bidirectional-transfer)
11. [Phase 4: Model Evaluation](#11-phase-4-model-evaluation)
12. [Statistical Analysis](#12-statistical-analysis)
13. [Model Size Configurations](#13-model-size-configurations)
14. [Complete Pipeline Summary](#14-complete-pipeline-summary)
15. [Project Directory Map](#15-project-directory-map)

---

## 1. Project Goal & High-Level Idea

The central question of this project is: **Can knowledge learned from one molecular toxicity classification task be transferred to improve performance on a different, but related, molecular toxicity classification task?**

This is a **transfer learning (TL)** study applied to **graph-based molecular representations**. Two distinct molecular domains are considered:

- **Peptides** — larger biological molecules (antimicrobial peptide toxicity)
- **Small molecules** — smaller chemical compounds (general molecular toxicity)

Both tasks share the same fundamental goal — **binary classification** (toxic = 1 vs. non-toxic = 0) — and both types of molecules can be represented as **molecular graphs**. The hypothesis is that a graph neural network trained on one domain learns generalizable patterns about molecular structure and toxicity that can benefit prediction on the other domain.

The project trains a **"source" model** on one dataset, then **transfers its learned weights** to be fine-tuned on the other dataset using four different transfer learning strategies. These are compared against a **baseline** model trained from scratch to determine whether transfer learning provides any benefit.

---

## 2. Datasets

### 2.1 Peptide Dataset — `ToxinSequenceSMILES.xlsx`

| Property | Detail |
|----------|--------|
| File | `datasets/ToxinSequenceSMILES.xlsx` |
| Columns used | `SMILES`, `TOXICITY` |
| Task | Antimicrobial peptide toxicity prediction |
| Labels | 0 (non-toxic), 1 (toxic) |
| Molecule type | Peptides represented in SMILES notation |

### 2.2 Small Molecule Toxicity Dataset — `MolToxPredDataset.xlsx`

| Property | Detail |
|----------|--------|
| File | `datasets/MolToxPredDataset.xlsx` |
| Columns used | `SMILES`, `Toxicity` |
| Task | Small molecule toxicity prediction |
| Labels | 0 (non-toxic), 1 (toxic) |
| Molecule type | Small chemical compounds in SMILES notation |

### Why these two datasets?

Both datasets describe molecular toxicity prediction as a **binary graph classification problem**. Although peptides and small molecules differ in size, complexity, and biological mechanism, they share **the same underlying chemistry** — they are both composed of atoms (C, N, O, S, etc.) connected by bonds. This common atomic-level structure makes it plausible that a GNN trained on one can learn features transferable to the other.

---

## 3. Data Preprocessing Pipeline — From XLSX to Graphs

The preprocessing converts raw SMILES strings into **graph objects** that the neural network can consume. This is a critical step where chemistry meets machine learning.

### Step-by-step process:

```
XLSX File (SMILES + Label)
        │
        ▼
   ┌─────────────┐
   │  Read Excel  │   pandas reads the XLSX file
   │  (pandas)    │   extracts SMILES string + activity label (0/1)
   └─────────────┘
        │
        ▼
   ┌─────────────────────┐
   │  Parse SMILES        │   RDKit converts SMILES string → Mol object
   │  (RDKit)             │   e.g. "CCO" → ethanol molecule object
   │  Chem.MolFromSmiles() │   Invalid SMILES are skipped (returns None)
   └─────────────────────┘
        │
        ▼
   ┌──────────────────────┐
   │  Extract Edges        │   For each bond in the molecule:
   │  (Bond list)          │   - Get begin atom index & end atom index
   │                       │   - Add both directions (undirected graph)
   │                       │   → edges = [(0,1), (1,0), (1,2), (2,1), ...]
   └──────────────────────┘
        │
        ▼
   ┌──────────────────────────────┐
   │  Create Node Features         │   For each atom in the molecule:
   │  (One-hot encoding)           │   - Look up its element symbol (C, N, O, ...)
   │                               │   - Create a one-hot vector of length 27
   │                               │     using the shared element_to_index vocabulary
   │                               │   e.g. Carbon → [0, 1, 0, 0, ..., 0]
   └──────────────────────────────┘
        │
        ▼
   ┌──────────────────────────────┐
   │  Build StellarGraph Object    │   StellarGraph(nodes=features, edges=edges_df)
   │                               │   This is the graph representation the GNN
   │                               │   will operate on
   └──────────────────────────────┘
        │
        ▼
   ┌──────────────────────┐
   │  Store as Pickle      │   Save list of (StellarGraph, label) pairs
   │  (.pkl file)          │   for reuse without re-parsing
   └──────────────────────┘
```

### What does a molecular graph look like?

Consider ethanol (SMILES: `CCO`):
- **Nodes (atoms):** C, C, O → three nodes, each represented by a 27-dimensional one-hot vector
- **Edges (bonds):** C—C and C—O → stored as bidirectional pairs in an edge DataFrame
- **Graph:** A StellarGraph object containing the node feature matrix and edge list

The resulting graph is an **undirected, attributed graph** where node attributes encode atom identity.

### Filtering

During preprocessing, molecules are also filtered for quality:
- Molecules with invalid SMILES (RDKit returns `None`) are skipped
- Molecules with isolated hydrogen atoms (degree = 0) are skipped (small molecule dataset)

### Output

Each dataset's preprocessing produces a `.pkl` file containing:
- `graphs` — list of StellarGraph objects
- `labels` — list of integer labels (0 or 1)
- `graph_labels` — pandas Series of labels
- `element_to_index` — the atom vocabulary dictionary

---

## 4. Shared Atom Vocabulary

A crucial design decision: **both datasets use the same fixed atom vocabulary**. This is what makes transfer learning possible at the architecture level.

The vocabulary was determined by scanning **both** datasets to find all unique chemical elements:

```python
element_to_index = {
    "N": 0,  "C": 1,  "O": 2,  "F": 3,  "Cl": 4,  "S": 5,  "Na": 6,
    "Br": 7, "Se": 8, "I": 9,  "Pt": 10, "P": 11, "Mg": 12, "K": 13,
    "Au": 14, "Ir": 15, "Cu": 16, "B": 17, "Zn": 18, "Re": 19,
    "Ca": 20, "As": 21, "Hg": 22, "Ru": 23, "Pd": 24, "Cs": 25, "Si": 26,
}
# NUM_FEATURES = 27
```

**Why a shared vocabulary is essential:**
- The GNN's first layer expects input node features of a fixed dimension (27)
- If each dataset had a different vocabulary, the pretrained model's first-layer weights wouldn't align with the target dataset's feature space
- By using the same 27-element vocabulary for both datasets, every layer's weights are directly compatible between the source and target domains

This is the **bridge** that makes weight transfer possible — atoms are represented identically regardless of whether they appear in a peptide or a small molecule.

---

## 5. The Neural Network Architecture — DeepGraphCNN

All models in this project share the same **DeepGraphCNN** architecture from the StellarGraph library. The model has two conceptual blocks:

### Block 1: GNN Layers (Graph Feature Extraction)

```
Input: Graph (node features + adjacency matrix)
    │
    ▼
GraphConvolution Layer 1  (tanh activation)
    │
    ▼
GraphConvolution Layer 2  (tanh activation)
    │
    ▼
GraphConvolution Layer 3  (tanh activation)
    │
    ▼
GraphConvolution Layer 4  (tanh activation)
    │
    ▼
SortPooling (k=25)
```

**What the GNN layers do:**
- Each `GraphConvolution` layer aggregates information from a node's neighbors, producing new node-level embeddings
- Stacking 4 layers means each node's final embedding captures information from its **4-hop neighborhood**
- The `SortPooling` layer converts the variable-size graph into a **fixed-size representation** by sorting nodes by their last-layer features and selecting the top-k (k=25) nodes
- This makes the output a fixed-size tensor regardless of how many atoms the molecule has

**Intuition:** The GNN layers learn **what molecular substructures are important** — they recognize patterns like aromatic rings, functional groups, charge distributions, etc.

### Block 2: Readout Layers (Classification)

```
SortPooled output (fixed-size tensor)
    │
    ▼
Conv1D (16 filters, kernel=sum(layer_sizes))
    │
    ▼
MaxPool1D (pool_size=2)
    │
    ▼
Conv1D (32 filters, kernel=5, stride=1)
    │
    ▼
Flatten
    │
    ▼
Dense (128 units, ReLU)
    │
    ▼
Dropout (rate=0.2)
    │
    ▼
Dense (1 unit, Sigmoid) → Output probability [0, 1]
```

**What the readout layers do:**
- The 1D convolutions process the sorted node embeddings as a sequence, detecting ordering-based patterns
- The dense layer maps the extracted features to the final classification decision
- The sigmoid output produces a probability of toxicity
- A threshold of 0.5 is used to convert the probability to a binary prediction

**Intuition:** The readout layers learn **how to make a toxicity decision** from the molecular features extracted by the GNN.

### Why this two-block mental model matters

The separation into "GNN block" and "Readout block" is **the foundation of the transfer learning strategy**. Different TL methods freeze different blocks, based on different hypotheses about what knowledge is transferable:
- GNN layers → domain-general molecular pattern recognition
- Readout layers → task-specific classification logic

---

## 6. Phase 1: Overtraining the Source Model

### Purpose

Before any transfer learning can happen, we need a **pretrained source model** whose weights encode useful knowledge about molecular toxicity. This model is called the "overtrained" model.

### Process

1. **Load the entire source dataset** (all graphs and labels)
2. **Split into 90% train / 10% validation** (using `train_test_split` with `random_state=42`)
3. **Build the DeepGraphCNN model** with random weight initialization
4. **Train until convergence** using:
   - Optimizer: Adam (learning rate = 0.0001)
   - Loss: Binary crossentropy
   - Early stopping: patience=7 on validation loss, restoring best weights
   - Max epochs: 10,000 (effectively infinite, early stopping will trigger first)
5. **Save the trained model** as an `.h5` file

### Key insight: Why "overtrained"?

The name "overtrained" refers to the fact that this model is trained on the **entire source dataset** without cross-validation — it sees all available data. The goal is not to evaluate this model's generalization, but to **pack as much learned knowledge as possible** into its weights so that the transfer learning methods have the richest possible starting point.

### Two source models are created:

| Source Model | Trained on | Saved as |
|-------------|-----------|---------|
| Peptide overtrained | `ToxinSequenceSMILES.xlsx` | `overtrained_peptide_model.h5` |
| SMT overtrained | `MolToxPredDataset.xlsx` | `overtrained_small_molecule_mol_tox_model.h5` |

### What the saved model contains

The `.h5` file stores:
- The complete model architecture (all layers, their shapes, activations)
- All trained weight values (the "knowledge" from the source domain)
- The optimizer state

---

## 7. Cross-Validation Split Generation

### Why cross-validation?

Unlike the overtrained model which uses all data, the **baseline and transfer learning models** must be evaluated fairly. The project uses **stratified 10-fold cross-validation** to:
- Ensure every data point is used for testing exactly once
- Preserve class balance (0/1 ratio) in each fold
- Provide 10 independent performance measurements for statistical analysis

### How splits are generated

```
Full dataset (N samples)
        │
        ▼
Stratified K-Fold (K=10, shuffle=True, random_state=42)
        │
        ├── Fold 1: train+val indices  │  test indices
        ├── Fold 2: train+val indices  │  test indices
        ├── ...
        └── Fold 10: train+val indices │  test indices
        
For each fold:
    train+val indices
        │
        ▼
    train_test_split (val_split=0.2)
        │
        ├── train indices (80% of train+val)
        └── val indices   (20% of train+val)
```

### The split structure stored per fold:

```python
{
    "train_idx": array([...]),  # Indices for training
    "val_idx":   array([...]),  # Indices for validation (early stopping)
    "test_idx":  array([...]),  # Indices for testing (held-out evaluation)
}
```

### Critical design choice: Shared splits

The **same CV splits** are used for both the baseline and all transfer learning methods. This ensures a **fair apples-to-apples comparison** — every model sees exactly the same training data and is tested on exactly the same test data for each fold.

Splits are saved as `.pkl` files and loaded by every training script:
- `cv_splits_peptide.pkl` — for peptide target experiments
- `cv_splits_small_mol_tox_pred_.pkl` — for SMT target experiments

---

## 8. Phase 2: Baseline Model Training (No Transfer Learning)

### Purpose

The baseline model establishes the **performance floor** — what you can achieve without any transfer learning. It answers: "How well does a freshly initialized model perform when trained from scratch on the target dataset?"

### Process (for each fold):

1. **Build a brand-new model** with the same architecture as the overtrained model, but with **randomly initialized weights**
2. **Train on the fold's training set** with validation monitoring
3. **Evaluate on the fold's test set** and compute metrics
4. **Save the model** as `baseline_*_fold_X.h5`

### What makes it baseline:

| Aspect | Baseline | Transfer Learning |
|--------|----------|-------------------|
| Initial weights | Random | Pretrained from source domain |
| Prior knowledge | None | Source domain patterns |
| Training | From scratch | Fine-tuning existing weights |

### Baseline models created:

- 10 models for peptide: `baseline_peptide/baseline_peptide_fold_{1-10}.h5`
- 10 models for SMT: `baseline_small_mol_tox/baseline_small_mol_tox_fold_{1-10}.h5`

---

## 9. Phase 3: Transfer Learning — Four Methods

All four methods follow the same high-level process:

1. **Load the pretrained (overtrained) model** from the source domain
2. **Apply a freezing strategy** (which layers to freeze/unfreeze)
3. **Fine-tune on the target dataset** using the same CV splits as the baseline
4. **Evaluate and save** each fold's model

The key difference between methods is **which layers are frozen** and **how training proceeds**.

---

### Method 1: Freeze GNN Layers

**Hypothesis:** The GNN layers have learned general-purpose molecular pattern recognition that applies across both domains. Only the classification head needs to adapt.

**What happens:**
```
┌──────────────────────────┐
│  GNN Layers              │  🔒 FROZEN — keep pretrained weights exactly
│  (4 × GraphConvolution   │       These already know how to read
│   + SortPooling)         │       molecular structures
└──────────────────────────┘
            │
            ▼
┌──────────────────────────┐
│  Readout Layers          │  🔓 TRAINABLE — fine-tuned on target data
│  (Conv1D + Dense + etc.) │       Learning new classification rules
└──────────────────────────┘
```

**Training:** Learning rate = 1e-4, Adam optimizer, early stopping patience=7

**When this works well:** When both domains have similar graph structures (both are molecules), and only the decision boundary needs to shift.

---

### Method 2: Freeze Readout Layers

**Hypothesis:** The readout/classification layers have learned a good **decision-making strategy** from the source domain. The GNN layers need to adapt their feature extraction to the new molecular domain.

**What happens:**
```
┌──────────────────────────┐
│  GNN Layers              │  🔓 TRAINABLE — can learn new graph patterns
│  (4 × GraphConvolution   │       Adapting feature extraction to new
│   + SortPooling)         │       molecular structures
└──────────────────────────┘
            │
            ▼
┌──────────────────────────┐
│  Readout Layers          │  🔒 FROZEN — keep pretrained classification
│  (Conv1D + Dense + etc.) │       logic unchanged
└──────────────────────────┘
```

**Training:** Learning rate = 1e-5 (lower, to prevent catastrophic forgetting of GNN weights), Adam optimizer, early stopping patience=7

**When this works well:** When the classification logic is similar between domains, but the raw molecular features differ significantly.

---

### Method 3: Freeze All Layers + New Output

**Hypothesis:** The **entire pretrained model** has learned useful feature representations. We use it as a pure **feature extractor** and only train a new output layer.

**What happens:**
```
┌──────────────────────────┐
│  GNN Layers              │  🔒 FROZEN
└──────────────────────────┘
            │
            ▼
┌──────────────────────────┐
│  Readout Layers          │  🔒 FROZEN
│  (up to Dropout layer)   │
└──────────────────────────┘
            │
            ▼
┌──────────────────────────┐
│  NEW Dense(1, sigmoid)   │  🔓 TRAINABLE — brand new layer added
│  (replaces original      │       Only ~129 parameters to learn
│   output layer)          │
└──────────────────────────┘
```

**Implementation detail:** The original output Dense layer is removed. A new Dense(1, sigmoid) layer is connected to the output of the Dropout layer (second-to-last layer of the original model).

**Training:** Learning rate = 1e-4, Adam optimizer, early stopping patience=7

**This is the most conservative approach** — it assumes the entire pretrained pipeline produces good features, and only needs a new linear classifier on top. Very fast training, minimal overfitting risk, but also least flexible.

---

### Method 4: Gradual Unfreezing + Discriminative Learning Rates

**Hypothesis:** All layers contain useful pretrained knowledge, but deeper layers (GNN) should be adjusted more carefully than later layers (readout/output).

**What happens — Three sequential training phases:**

**Phase 1: Train final layers only (LR = 1e-3)**
```
GNN Layers     → 🔒 FROZEN
Readout Layers → 🔒 FROZEN
Final Layers   → 🔓 TRAINABLE   ← Highest learning rate
```
Quick adaptation of the output decision to the new task.

**Phase 2: Unfreeze readout layers (LR = 1e-4)**
```
GNN Layers     → 🔒 FROZEN
Readout Layers → 🔓 TRAINABLE   ← Now learning too
Final Layers   → 🔓 TRAINABLE
```
The readout layers begin adapting their feature processing.

**Phase 3: Unfreeze GNN layers (LR = 1e-5)**
```
GNN Layers     → 🔓 TRAINABLE   ← Finally unfrozen, lowest LR
Readout Layers → 🔓 TRAINABLE
Final Layers   → 🔓 TRAINABLE
```
Very careful fine-tuning of the deep graph representations.

**Training:** 10 epochs per phase, early stopping within each phase, decreasing learning rates (1e-3 → 1e-4 → 1e-5)

**This is the most sophisticated approach.** The decreasing learning rate schedule prevents **catastrophic forgetting** — the phenomenon where fine-tuning destroys useful pretrained knowledge by updating weights too aggressively. By starting from the top (output) and slowly working down to the bottom (GNN), the model gradually adapts while preserving foundational representations.

---

### Summary: Comparing the Four Methods

```
               Method 1         Method 2         Method 3         Method 4
              Freeze GNN      Freeze Readout    Freeze All      Gradual Unfreeze
              ──────────       ──────────────    ──────────      ────────────────
GNN:          🔒 Frozen        🔓 Fine-tuned     🔒 Frozen       🔓 Phase 3 (LR=1e-5)
Readout:      🔓 Fine-tuned    🔒 Frozen         🔒 Frozen       🔓 Phase 2 (LR=1e-4)
Output:       🔓 Fine-tuned    🔒 Frozen         🔓 NEW layer    🔓 Phase 1 (LR=1e-3)
              
Flexibility:  Low-Medium       Medium            Very Low         High
Speed:        Fast             Medium            Very Fast        Slow (3 phases)
Risk of       
forgetting:   None (GNN)       Low (slow LR)     None             Very Low (gradual)
```

---

## 10. Bidirectional Transfer

Transfer learning is applied in **both directions**, creating a total of 8 experimental conditions (4 methods × 2 directions):

### Direction 1: Peptide → Small Molecule Toxicity

```
Source: Peptide overtrained model
        ↓ (transfer weights)
Target: Small Molecule Toxicity dataset
        ↓ (fine-tune with 4 methods)
Output: 4 × 10 fold models in transfer_learning_p_to_smt/
```

File naming: `{method}_peptide_to_smile_mol_tox_fold_{1-10}.h5`

### Direction 2: Small Molecule Toxicity → Peptide

```
Source: SMT overtrained model
        ↓ (transfer weights)
Target: Peptide dataset
        ↓ (fine-tune with 4 methods)
Output: 4 × 10 fold models in transfer_learning_smt_to_p/
```

File naming: `{method}_small_mol_tox_to_peptide_fold_{1-10}.h5`

### Why bidirectional?

Transfer learning is not always symmetric. Knowledge from peptides may help predict small molecule toxicity, but the reverse may not hold (or may hold to a different degree). Testing both directions provides a complete picture of cross-domain knowledge transferability.

---

## 11. Phase 4: Model Evaluation

### Per-fold evaluation

For each of the 10 folds, and for each model (baseline + 4 methods), the evaluation process:

1. **Load the trained model** (with StellarGraph custom layers)
2. **Load the test set** for that fold (from the shared CV splits)
3. **Generate predictions** (probability values between 0 and 1)
4. **Compute metrics** using threshold = 0.5 for binary conversion

### Metrics computed:

| Metric | What it measures | Formula/Notes |
|--------|-----------------|---------------|
| **ROC-AUC** | Overall discrimination ability | Area under the ROC curve. Computed on raw probabilities (before thresholding) |
| **GM** | Geometric Mean | √(TPR × TNR) — balanced metric for imbalanced datasets |
| **Precision** | Prediction accuracy for positives | TP / (TP + FP) |
| **Recall** | Coverage of actual positives | TP / (TP + FN) |
| **F1** | Balance of precision and recall | 2 × (Precision × Recall) / (Precision + Recall) |
| **MCC** | Overall correlation quality | Matthews Correlation Coefficient — best single metric for binary classification |

### Why these specific metrics?

Toxicity datasets are often **imbalanced** (different numbers of toxic vs. non-toxic samples). Simple accuracy can be misleading. MCC and GM are particularly valuable because they account for all four quadrants of the confusion matrix and are robust to class imbalance.

---

## 12. Statistical Analysis

### Friedman Test

Since we have 5 models (baseline + 4 methods) evaluated across 10 folds (repeated measures), the **Friedman test** is used to determine if there are statistically significant differences between models.

- **Null hypothesis:** All models perform equally
- **If p-value < 0.05:** There are significant differences → proceed to post-hoc analysis

### Nemenyi Post-Hoc Test

When the Friedman test is significant, the **Nemenyi test** is used to determine which specific pairs of models differ significantly. This produces a pairwise comparison matrix showing which models are statistically distinguishable.

### Visualization

- **Box-and-whisker plots** show the distribution of each metric across the 10 folds for each model
- **Radar plots** provide a multi-metric visual comparison across all models and configurations

---

## 13. Model Size Configurations

The project also explores whether model capacity affects transfer learning effectiveness by testing three GNN layer size configurations:

| Configuration | GNN Layer Sizes | Total GNN Parameters | Description |
|---------------|----------------|---------------------|-------------|
| Standard | [25, 25, 25, 1] | Small | Compact model |
| Large Layers | [125, 125, 125, 1] | Medium | Increased capacity |
| Inflated | [512, 256, 128, 1] | Large | Maximum capacity |

Each configuration goes through the complete pipeline independently:
- Its own overtrained source model
- Its own baseline models
- Its own transfer learning models (all 4 methods, both directions)
- Its own evaluation

This allows analyzing whether **larger models benefit more from transfer learning** (because they have more capacity to store transferable representations) or whether **smaller models benefit more** (because they need external knowledge more).

---

## 14. Complete Pipeline Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA PREPARATION                             │
│                                                                     │
│  XLSX Files (SMILES + Labels)                                       │
│       │                                                             │
│       ▼                                                             │
│  Element Discovery ──► Shared 27-atom vocabulary                    │
│       │                                                             │
│       ▼                                                             │
│  SMILES → RDKit Mol → Edges + One-hot Features → StellarGraph      │
│       │                                                             │
│       ▼                                                             │
│  Pickle files (.pkl) — preprocessed graph datasets                  │
│       │                                                             │
│       ▼                                                             │
│  Stratified 10-fold CV splits (saved for consistency)               │
└─────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     SOURCE MODEL TRAINING                           │
│                                                                     │
│  For EACH dataset (Peptide & SMT):                                  │
│    Build DeepGraphCNN → Train on 100% data → Save as .h5            │
│    (This is the "overtrained" model with transferable weights)      │
└─────────────────────────────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌──────────────────┐  ┌──────────────────────────────────────────────┐
│  BASELINE MODELS │  │         TRANSFER LEARNING MODELS             │
│                  │  │                                              │
│  For each fold:  │  │  For each fold, for each method:            │
│   New model      │  │   Load pretrained source model               │
│   Random init    │  │   Apply freeze strategy (M1/M2/M3/M4)      │
│   Train from     │  │   Fine-tune on target dataset               │
│   scratch        │  │   Save model per fold                       │
│   Save per fold  │  │                                              │
└──────────────────┘  │  Direction 1: Peptide → SMT                 │
                      │  Direction 2: SMT → Peptide                  │
                      └──────────────────────────────────────────────┘
        │                       │
        └───────────┬───────────┘
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         EVALUATION                                  │
│                                                                     │
│  For each fold, for each model (baseline + 4 methods):             │
│    Load model → Predict on test set → Compute 6 metrics            │
│                                                                     │
│  Aggregate across folds → Mean ± Std per metric per model          │
│                                                                     │
│  Statistical tests:                                                 │
│    Friedman test → Nemenyi post-hoc → Significance conclusions     │
│                                                                     │
│  Visualization:                                                     │
│    Box plots, Radar plots                                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 15. Project Directory Map

```
transfer_project/
│
├── datasets/                          # Raw XLSX data + analysis scripts
│   ├── ToxinSequenceSMILES.xlsx       # Peptide dataset
│   ├── MolToxPredDataset.xlsx         # Small molecule toxicity dataset
│   ├── find_all_elements.py           # Discovers shared atom vocabulary
│   ├── analyze_peptide_elements.py    # Analyzes peptide elements
│   └── analyze_small_mol_elements.py  # Analyzes SMT elements
│
├── overtrained_models/                # Source models (pretrained)
│   ├── peptide_overtrained.py         # Script to train peptide source model
│   ├── small_molecule_mol_tox_overtrained.py  # Script to train SMT source model
│   ├── overtrained_peptide_model.h5   # Saved peptide source model
│   ├── overtrained_small_molecule_mol_tox_model.h5  # Saved SMT source model
│   └── *.pkl                          # Preprocessed graph datasets
│
├── cv_splits/                         # Cross-validation split definitions
│   ├── data_splits.py                 # Functions to generate/load CV splits
│   ├── cv_splits_creation.py          # Script to create splits
│   ├── cv_splits_peptide.pkl          # 10-fold splits for peptide data
│   └── cv_splits_small_mol_tox_pred_.pkl  # 10-fold splits for SMT data
│
├── baseline_peptide/                  # Baseline models — peptide target
│   ├── baseline_peptide.py            # Training script
│   └── baseline_peptide_fold_{1-10}.h5  # 10 trained models
│
├── baseline_small_mol_tox/            # Baseline models — SMT target
│   ├── baseline_small_mol_tox.py      # Training script
│   └── baseline_small_mol_tox_fold_{1-10}.h5  # 10 trained models
│
├── transfer_learning_p_to_smt/        # TL: Peptide → Small Molecule Tox
│   ├── peptide_to_small_molecule_tl.py  # All 4 methods in one script
│   ├── freeze_gnn_peptide_to_smile_mol_tox_fold_{1-10}.h5
│   ├── freeze_readout_peptide_to_smile_mol_tox_fold_{1-10}.h5
│   ├── freeze_all_peptide_to_smile_mol_tox_fold_{1-10}.h5
│   └── gradual_unfreezing_peptide_to_smile_mol_tox_fold_{1-10}.h5
│
├── transfer_learning_smt_to_p/        # TL: Small Molecule Tox → Peptide
│   ├── small_molecule_to_peptide_tl.py  # All 4 methods in one script
│   ├── freeze_gnn_small_mol_tox_to_peptide_fold_{1-10}.h5
│   ├── freeze_readout_small_mol_tox_to_peptide_fold_{1-10}.h5
│   ├── freeze_all_small_mol_tox_to_peptide_fold_{1-10}.h5
│   └── gradual_unfreezing_small_mol_tox_to_peptide_fold_{1-10}.h5
│
├── eval_peptide/                      # Evaluation — peptide as target
│   └── evaluate_peptide.py
│
├── eval_small_mol_tox/                # Evaluation — SMT as target
│   └── evaluate_small_mol_tox_models.py
│
├── models_architecture/               # Architecture documentation (this folder)
│   ├── README.md
│   ├── overtrained_model.md
│   ├── baseline_model.md
│   ├── method1_freeze_gnn.md
│   ├── method2_freeze_readout.md
│   ├── method3_freeze_all.md
│   └── method4_gradual_unfreezing.md
│
├── large_layers_*/                    # [125,125,125,1] configuration
├── inflated_*/                        # [512,256,128,1] configuration
├── extra_features_*/                  # [25,25,25,1] configuration
│
└── radar_plots_*.py                   # Multi-config radar plot generation
```

---

## Key Takeaways

1. **Molecules become graphs** — SMILES strings are converted to StellarGraph objects where atoms are nodes and bonds are edges
2. **A shared atom vocabulary (27 elements)** ensures the neural network's input dimensions match between source and target domains
3. **DeepGraphCNN** extracts hierarchical molecular features through graph convolutions, then classifies using a readout network
4. **The overtrained model** packs maximum knowledge from the source domain into its weights
5. **Four TL methods** explore different hypotheses about what knowledge transfers: graph patterns (M1), classification logic (M2), entire feature pipeline (M3), or everything gradually (M4)
6. **10-fold stratified CV** with shared splits ensures fair comparison
7. **Six metrics** provide a comprehensive view of model performance, especially for imbalanced datasets
8. **Friedman + Nemenyi tests** determine statistical significance of differences between methods
9. **Three model size configurations** reveal how network capacity interacts with transfer learning effectiveness
