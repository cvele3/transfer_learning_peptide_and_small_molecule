# Complete Project Pipeline & Draw.io Construction Blueprint

> **Purpose of this document:** A single, unified reference that (1) explains **every stage** of the Transfer Learning for Molecular Toxicity Prediction project and (2) provides **exact draw.io instructions** — shapes, icons, colors, connectors, labels, and layout — so you can build a publication-ready diagram directly from this guide.

---

## Table of Contents

1. [Visual Style Reference (Legend)](#1-visual-style-reference-legend)
2. [Stage A — Raw Data: XLSX Datasets with SMILES](#2-stage-a--raw-data-xlsx-datasets-with-smiles)
3. [Stage B — Preprocessing: SMILES → Molecular Graphs](#3-stage-b--preprocessing-smiles--molecular-graphs)
4. [Stage C — Shared Atom Vocabulary (The Bridge)](#4-stage-c--shared-atom-vocabulary-the-bridge)
5. [Stage D — Cross-Validation Splits](#5-stage-d--cross-validation-splits)
6. [Stage E — Overtraining: Source Model Creation (90/10 Split)](#6-stage-e--overtraining-source-model-creation-9010-split)
7. [Stage F — Baseline Models (Trained from Scratch)](#7-stage-f--baseline-models-trained-from-scratch)
8. [Stage G — Transfer Learning: 4 Methods](#8-stage-g--transfer-learning-4-methods)
9. [Stage H — Evaluation & Statistical Analysis](#9-stage-h--evaluation--statistical-analysis)
10. [Full Single-Page Diagram: Assembling Everything](#10-full-single-page-diagram-assembling-everything)
11. [Shape & Icon Quick-Reference Table](#11-shape--icon-quick-reference-table)

---

## 1. Visual Style Reference (Legend)

Before building anything in draw.io, set up these conventions. This legend should also appear **inside your final diagram** as a small box in the bottom-right corner.

### 1.1 Color Palette

| Role | Hex Color | Fill Example | When to Use |
|------|-----------|--------------|-------------|
| **Data / Raw Input** | `#DAE8FC` | Light blue | XLSX files, SMILES data, raw datasets |
| **Processing Step** | `#D5E8D4` | Light green | RDKit parsing, feature engineering, any transformation |
| **Neural Network / Model** | `#FFE6CC` | Light orange | DeepGraphCNN architecture, model blocks |
| **Frozen Layer** | `#F8CECC` | Light red | Any layer whose weights are locked during TL |
| **Trainable Layer** | `#D5E8D4` | Light green | Any layer that is being fine-tuned |
| **Saved Artifact / File** | `#FFF2CC` | Light yellow | `.h5` model files, `.pkl` pickle files |
| **Output / Metric** | `#E1D5E7` | Light purple | Predictions, metrics, evaluation results |
| **Stage Container** | `#F5F5F5` | Light gray | Background rectangles grouping an entire stage |
| **Phase (Method 4)** | `#FFF2CC` | Light yellow | Layers unfrozen in later gradual-unfreezing phases |

### 1.2 Shape Dictionary

| Concept | draw.io Shape | Where to Find |
|---------|--------------|---------------|
| XLSX / data file | **Document** shape (paper with folded corner) | General → Document |
| Pickle / H5 file | **Cylinder** (database) | General → Cylinder |
| Processing / computation | **Rounded Rectangle** | General → Rounded Rectangle |
| NN Layer | **Rectangle** (straight corners) | General → Rectangle |
| Complete Model | **Rectangle** with **thick 3pt border** | Style: strokeWidth=3 |
| Decision point | **Diamond** | General → Diamond |
| Metric / result | **Hexagon** | Advanced → Hexagon |
| Loop indicator | **Circular arrow** icon | Search "loop" or "refresh" in shape search |
| Frozen indicator | **Lock** icon (🔒) | Search "lock" in shape panel |
| Trainable indicator | **Checkmark** icon (✓) or **Unlock** 🔓 | Search "unlock" or "check" |
| Stage group | **Container** (dashed border) | Insert → Advanced → Container |
| Annotation / note | **Callout** (speech bubble) | General → Callout |
| SMILES label | **Text box** with monospace font | Double-click canvas → type text |
| Molecule drawing | **Ellipses** (atoms) + **Lines** (bonds) | General → Ellipse + Line connector |

### 1.3 Connectors

| Type | Style | Usage |
|------|-------|-------|
| Main flow | **Solid black**, 2pt, filled arrowhead | Primary pipeline direction (top→bottom) |
| Data movement | **Dashed blue**, 1.5pt, open arrowhead | Data being loaded/read by a step |
| Weight transfer | **Thick dashed orange**, 2pt, filled arrowhead | Pretrained weights flowing from source to target model |
| Annotation pointer | **Dotted gray**, 1pt, no arrowhead | Connecting a note/callout to a shape |
| Branch / split | **Solid black**, 2pt, two output arrowheads | When the pipeline forks (e.g., into baseline + TL) |

### 1.4 Font Conventions

- **Stage titles:** Arial 16pt Bold, centered inside container header
- **Shape labels:** Arial 11pt Regular, centered
- **Annotations / notes:** Arial 9pt Italic, color `#666666` (gray)
- **Code snippets inside shapes:** Courier New 9pt, left-aligned

---

## 2. Stage A — Raw Data: XLSX Datasets with SMILES

### What happens in the project

The project begins with **two Excel files** (`.xlsx`), each containing molecular data in **SMILES** (Simplified Molecular Input Line Entry System) format alongside a binary activity label:

| Dataset | File | SMILES Column | Label Column | Label Meaning |
|---------|------|---------------|--------------|---------------|
| **Peptide toxicity** | `ToxinSequenceSMILES.xlsx` | `SMILES` | `TOXICITY` | 0 = non-toxic, 1 = toxic |
| **Small molecule toxicity** | `MolToxPredDataset.xlsx` | `SMILES` | `Toxicity` | 0 = non-toxic, 1 = toxic |

**SMILES** is a text-based notation that describes a molecule's structure as a string. For example:
- `CCO` = ethanol (two carbons bonded, plus an oxygen)
- `c1ccccc1` = benzene (aromatic ring)
- Peptides have much longer SMILES strings because they are larger molecules

The activity label marks whether the molecule is **toxic (1)** or **non-toxic (0)**, making this a **binary classification** problem.

### draw.io Construction — Stage A

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE A: RAW DATA                                                          │
│  (Container: dashed border, fill #F5F5F5, title in header)                  │
│                                                                             │
│   ┌──────────────────┐              ┌──────────────────┐                    │
│   │ 📄 Document Shape │              │ 📄 Document Shape │                    │
│   │ Fill: #DAE8FC     │              │ Fill: #DAE8FC     │                    │
│   │                   │              │                   │                    │
│   │ Label:            │              │ Label:            │                    │
│   │ "ToxinSequence    │              │ "MolToxPred       │                    │
│   │  SMILES.xlsx"     │              │  Dataset.xlsx"    │                    │
│   │                   │              │                   │                    │
│   │ [Small icon below │              │ [Small icon below │                    │
│   │  or badge:]       │              │  or badge:]       │                    │
│   │  "SMILES + Label" │              │  "SMILES + Label" │                    │
│   └────────┬─────────┘              └────────┬─────────┘                    │
│            │                                  │                              │
│   ┌────────▼─────────┐              ┌────────▼─────────┐                    │
│   │  Text Annotation  │              │  Text Annotation  │                    │
│   │  (Italic 9pt)     │              │  (Italic 9pt)     │                    │
│   │  "Peptide          │              │  "Small molecule   │                    │
│   │   molecules in     │              │   compounds in     │                    │
│   │   SMILES format"   │              │   SMILES format"   │                    │
│   │                    │              │                    │                    │
│   │  Example row:      │              │  Example row:      │                    │
│   │  SMILES: CC(=O)N.. │              │  SMILES: CCO       │                    │
│   │  TOXICITY: 1       │              │  Toxicity: 0       │                    │
│   └────────────────────┘              └────────────────────┘                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Exact draw.io steps:**

1. **Create a Container** (Insert → Advanced → Container): width ~800px, height ~300px. Set fill to `#F5F5F5`, border dashed, label "STAGE A: RAW DATA" in the header bar (16pt Bold).

2. **Place two Document shapes** side by side inside the container:
   - Find "Document" in the General shape library (it looks like a piece of paper with a curled bottom edge)
   - Size: ~160×100px each
   - Fill: `#DAE8FC` (light blue)
   - Left document label: **"ToxinSequenceSMILES.xlsx"** (11pt)
   - Right document label: **"MolToxPredDataset.xlsx"** (11pt)

3. **Add a small badge/label** below each document (or overlapping the bottom edge):
   - Use a small rounded rectangle (~120×25px), fill `#DAE8FC` darker shade (`#B0C4DE`)
   - Label: **"SMILES + Label (0/1)"** in 9pt

4. **Add annotation callouts** beneath each document:
   - Use a **callout/speech-bubble shape** or a plain text box with dotted border
   - Include an example data row in `Courier New` font:
     - Left: `SMILES: CC(=O)NC...  TOXICITY: 1`
     - Right: `SMILES: CCO  Toxicity: 0`

5. **Optional molecule icon:** Place a tiny **hexagon** (representing a benzene ring) or a small cluster of 3 connected circles next to each document to visually suggest "molecule data". This is purely decorative but immediately communicates the chemistry context.

---

## 3. Stage B — Preprocessing: SMILES → Molecular Graphs

### What happens in the project

Each SMILES string is converted into a **graph object** that the neural network can process:

1. **Read the XLSX** using pandas → extract each SMILES string and its label
2. **Parse SMILES** using RDKit (`Chem.MolFromSmiles()`) → get a Mol object representing the molecule
3. **Extract edges** (bonds) → for each bond, record `(begin_atom_idx, end_atom_idx)` in both directions (undirected graph)
4. **Create node features** → for each atom, create a **27-dimensional one-hot vector** using the shared element vocabulary (e.g., Carbon → `[0,1,0,...,0]`)
5. **Build a StellarGraph object** from the node feature matrix + edge DataFrame
6. **Filter invalid molecules** — skip any SMILES that RDKit cannot parse, or molecules with isolated hydrogen atoms
7. **Save as `.pkl`** — the complete list of (graph, label) pairs is pickled for reuse

The key output: each molecule is now a **graph** where:
- **Nodes** = atoms (each with a 27-dim feature vector)
- **Edges** = bonds (undirected, stored as bidirectional pairs)

### draw.io Construction — Stage B

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  STAGE B: PREPROCESSING — SMILES → GRAPHS                                    │
│  (Container: dashed border, fill #F5F5F5)                                    │
│                                                                              │
│  [Document: XLSX]─────►[Rounded Rect: pandas.read_excel()]                   │
│  (from Stage A)         Fill: #D5E8D4                                        │
│                                │                                              │
│                                ▼                                              │
│                    [Rounded Rect: RDKit]                                      │
│                    "Chem.MolFromSmiles()"                                     │
│                    Fill: #D5E8D4                                              │
│                    [Small RDKit logo or                                       │
│                     chemistry flask icon]                                     │
│                                │                                              │
│                    ┌───────────┴───────────┐                                  │
│                    ▼                       ▼                                   │
│          [Rounded Rect:            [Rounded Rect:                             │
│           "Extract Bonds"           "One-Hot Encode Atoms"                    │
│           Fill: #D5E8D4]            Fill: #D5E8D4]                            │
│                                                                              │
│          Annotation below:         Annotation below:                         │
│          "edges = [(0,1),          "C→[0,1,0,...,0]                           │
│           (1,0),(1,2),              O→[0,0,1,...,0]                           │
│           (2,1)]"                   27-dim vector"                            │
│                    │                       │                                   │
│                    └───────────┬───────────┘                                  │
│                                ▼                                              │
│                    [Rounded Rect: StellarGraph()]                             │
│                    "Build graph object"                                        │
│                    Fill: #D5E8D4                                              │
│                                │                                              │
│                    ┌───────────┴───────────┐                                  │
│                    ▼                       ▼                                   │
│           [Cylinder:               [Cylinder:                                 │
│            "peptide_graphs.pkl"     "smt_graphs.pkl"                          │
│            Fill: #FFF2CC]           Fill: #FFF2CC]                            │
│                                                                              │
│  ── VISUAL EXAMPLE (bottom of container) ──────────────────────────          │
│  │                                                                │          │
│  │  "CCO" ──RDKit──►  (C)──(C)──(O)  ──StellarGraph──► 📊 Graph  │          │
│  │  string            3 nodes, 2 bonds          object            │          │
│  │                                                                │          │
│  └────────────────────────────────────────────────────────────────┘          │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Exact draw.io steps:**

1. **Container:** Same style as Stage A. Label: "STAGE B: PREPROCESSING".

2. **Input arrow from Stage A:** Draw a **solid black arrow** (2pt) from the bottom of both Stage A documents merging into one arrow that enters the top of Stage B.

3. **Processing chain** — place 4 rounded rectangles vertically:
   - **"pandas.read_excel()"** — fill `#D5E8D4`, size ~200×50px
   - **"RDKit: Chem.MolFromSmiles()"** — fill `#D5E8D4`, size ~220×50px
     - *Optional:* Search draw.io for a "flask" or "beaker" icon (Insert → Search → type "flask") and place it as a small 20×20px overlay in the corner of this shape to suggest chemistry processing
   - Two parallel rounded rectangles branching from RDKit:
     - **"Extract Bonds (Edge List)"** — fill `#D5E8D4`
     - **"One-Hot Encode Atoms (27-dim)"** — fill `#D5E8D4`
   - **"StellarGraph()"** — fill `#D5E8D4`, both branches merge into this

4. **Branch to two cylinders** at the bottom:
   - **Cylinder** (General → Cylinder): "peptide_graphs.pkl" — fill `#FFF2CC`
   - **Cylinder**: "smt_graphs.pkl" — fill `#FFF2CC`

5. **Molecule visualization example** (the "CCO" example):
   - At the bottom of the container, create a small horizontal inset box (dashed border, no fill)
   - Inside: place a **text box** `"CCO"` → draw a right-arrow → place **3 ellipses** (representing C, C, O atoms) connected with lines → draw another right-arrow → place a small **cylinder** icon labeled "Graph Object"
   - Color the atom ellipses: first two gray (Carbon), last one red (Oxygen)
   - This gives the viewer an immediate intuition of what SMILES→Graph means

---

## 4. Stage C — Shared Atom Vocabulary (The Bridge)

### What happens in the project

Before preprocessing either dataset, **both XLSX files are scanned** to collect every unique chemical element that appears in any molecule. This produces a **shared dictionary** mapping element symbols to integer indices:

```python
element_to_index = {
    "N": 0,  "C": 1,  "O": 2,  "F": 3,  "Cl": 4,  "S": 5,  "Na": 6,
    "Br": 7, "Se": 8, "I": 9,  "Pt": 10, "P": 11, "Mg": 12, "K": 13,
    "Au": 14, "Ir": 15, "Cu": 16, "B": 17, "Zn": 18, "Re": 19,
    "Ca": 20, "As": 21, "Hg": 22, "Ru": 23, "Pd": 24, "Cs": 25, "Si": 26,
}
# → 27 unique elements → one-hot vector length = 27
```

**Why this is critical for transfer learning:** Both datasets produce node features of the **exact same dimensionality** (27). This means the first layer of the GNN (which reads node features) has the same input shape regardless of which dataset trained it. When we load pretrained weights from a source model trained on one dataset, every weight matrix is **dimensionally compatible** with the target dataset. Without this shared vocabulary, weight transfer would be impossible.

### draw.io Construction — Stage C

This is best shown as a **side element** connected to Stage B, not its own full stage container:

```
     [Document: Peptide XLSX]──┐
                                ├──►[Rounded Rect: Scan All Elements]
     [Document: SMT XLSX]──────┘         Fill: #D5E8D4
                                              │
                                              ▼
                                  ┌─────────────────────────┐
                                  │  TABLE SHAPE             │
                                  │  "Shared Atom Vocabulary" │
                                  │  Fill: #FFF2CC            │
                                  │                           │
                                  │  N→0  C→1  O→2  F→3  ... │
                                  │  ...  Si→26               │
                                  │  Total: 27 elements       │
                                  └────────────┬──────────────┘
                                               │
                                    dashed blue arrow down to
                                    Stage B's "One-Hot Encode" step
                                               │
                                  [Callout annotation:]
                                  "THIS makes transfer learning
                                   possible — identical feature
                                   space in both domains"
```

**Exact draw.io steps:**

1. **Do NOT create a separate full-width container.** Instead, place this as a **floating group** to the **right side** of Stage B.

2. **Two dashed blue arrows** from the Stage A documents feed into a **rounded rectangle** labeled "Scan All Elements" (fill `#D5E8D4`).

3. Below it, place a **Table** (Insert → Table, 4 columns × 7 rows) or a **rectangle with internal text**:
   - Fill: `#FFF2CC` (yellow — it's a saved artifact / reference)
   - Label: **"Shared Atom Vocabulary"** at top (bold)
   - Inside: show element→index mapping (at least a few representative entries)
   - Bottom text: **"27 unique elements = 27-dim feature vector"**

4. **Dashed blue arrow** from this vocabulary table pointing down-left into Stage B's "One-Hot Encode Atoms" step — labeled **"Used for encoding"** on the arrow.

5. **Important callout:** Place a **callout/bubble shape** (General → Callout) next to the vocabulary table. Fill it with `#FFFACD` (very pale yellow). Text: *"Critical: Identical vocabulary for both datasets → weight dimensions match → transfer learning is possible"*. Use 9pt italic gray text.

---

## 5. Stage D — Cross-Validation Splits

### What happens in the project

A **stratified 10-fold cross-validation** scheme is generated **once** and saved, then reused by every training script to ensure fair comparison:

1. The full dataset (all graphs + labels) is divided into **10 folds** using `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`
2. Each fold defines:
   - **Test set** (~10% of data) — held out, never seen during training
   - **Training + Validation** (~90% of data) — further split 80/20:
     - **Training set** (80% of the 90%) — used for gradient updates
     - **Validation set** (20% of the 90%) — used for early stopping monitoring
3. Splits are saved as `.pkl` files: `cv_splits_peptide.pkl` and `cv_splits_small_mol_tox_pred_.pkl`

**Key design choice:** The **baseline model** and **all 4 transfer learning methods** use the **exact same 10 folds**. This guarantees that any performance difference is due to the learning strategy, not different data splits.

### draw.io Construction — Stage D

```
┌───────────────────────────────────────────────────────────────────────────┐
│  STAGE D: CROSS-VALIDATION SPLITS                                         │
│  (Container: dashed, #F5F5F5)                                             │
│                                                                           │
│  [Cylinder: peptide_graphs.pkl]  [Cylinder: smt_graphs.pkl]               │
│  (from Stage B)                  (from Stage B)                           │
│         │                               │                                 │
│         └──────────┬────────────────────┘                                 │
│                    ▼                                                       │
│         [Rounded Rect: StratifiedKFold]                                   │
│         "K=10, shuffle=True"                                              │
│         Fill: #D5E8D4                                                     │
│         [Small loop/recycle icon overlay]                                  │
│                    │                                                       │
│    ┌───┬───┬───┬──┴──┬───┬───┬───┬───┬───┬───┐                           │
│    ▼   ▼   ▼   ▼     ▼   ▼   ▼   ▼   ▼   ▼   │                           │
│   F1  F2  F3  F4    F5  F6  F7  F8  F9  F10   │                           │
│   (10 small rectangles in a row, each labeled) │                           │
│                                                                           │
│   [Annotation: Each fold → Train (72%) | Val (18%) | Test (10%)]          │
│                                                                           │
│         ┌──────────┬──────────┐                                           │
│         ▼          ▼          ▼                                            │
│   [Cylinder:   [Cylinder:                                                 │
│    cv_splits    cv_splits    [Callout: "Shared by ALL                     │
│    _peptide     _smt          models for fair                             │
│    .pkl]        .pkl]         comparison"]                                 │
│   Fill: #FFF2CC  Fill: #FFF2CC                                            │
└───────────────────────────────────────────────────────────────────────────┘
```

**Exact draw.io steps:**

1. **Container** as before. Label: "STAGE D: CROSS-VALIDATION SPLITS".

2. **Arrows from Stage B cylinders** enter from the top.

3. **Rounded rectangle** "StratifiedKFold(K=10)" — fill `#D5E8D4`. Place a small **circular arrow icon** (search "refresh" or "loop" in draw.io) as a 20×20px overlay at the top-right corner to indicate repeated folding.

4. **Fan-out to 10 small rectangles** in a horizontal row:
   - Each rectangle ~40×30px, fill `#DAE8FC` (light blue)
   - Labels: F1, F2, ..., F10
   - Connect with 10 thin arrows from the StratifiedKFold box
   - *Tip:* Group these 10 as a single grouped element so they move together

5. **Below the fold row**, add an **annotation text box**:
   - "Each fold: **Train 72%** | **Val 18%** | **Test 10%**"
   - Use a small stacked-bar visual: a rectangle divided into 3 colored sections (green = train, yellow = val, blue = test) — create this by placing 3 thin rectangles side by side

6. **Two output cylinders** at the bottom: `cv_splits_peptide.pkl` and `cv_splits_smt.pkl`, both fill `#FFF2CC`.

7. **Callout bubble** (fill `#FFFACD`): *"These exact splits are reused by baseline AND all 4 TL methods — guaranteeing fair comparison"*

---

## 6. Stage E — Overtraining: Source Model Creation (90/10 Split)

### What happens in the project

For **each dataset independently**, an **overtrained source model** is created:

1. Load the **full preprocessed dataset** (all graphs from the `.pkl`)
2. Split into **90% training / 10% validation** (`train_test_split`, `random_state=42`)
3. Build a **DeepGraphCNN** model with random weight initialization:
   - **GNN Block:** 4 × GraphConvolution layers (tanh) → SortPooling (k=25)
   - **Readout Block:** Conv1D → MaxPool → Conv1D → Flatten → Dense(128, ReLU) → Dropout(0.2) → Dense(1, sigmoid)
4. Train with:
   - **Adam optimizer**, learning rate = 0.0001
   - **Binary crossentropy** loss
   - **Early stopping:** patience=7, monitoring `val_loss`, restore best weights
   - Max epochs = 10,000 (early stopping fires long before)
5. Save the converged model as `.h5`

**Why "overtrained"?** This model is trained on **100% of the source data** (no CV holdout). The purpose is NOT to evaluate this model. The purpose is to **maximize knowledge absorption** — pack as much molecular toxicity knowledge as possible into the weights — so that transfer learning has the richest starting point.

**Two overtrained models are produced:**

| Model | Source Data | Saved File |
|-------|-----------|------------|
| Peptide source | ToxinSequenceSMILES.xlsx | `overtrained_peptide_model.h5` |
| SMT source | MolToxPredDataset.xlsx | `overtrained_small_molecule_mol_tox_model.h5` |

### draw.io Construction — Stage E

This is one of the most important stages visually. It should clearly show the **90/10 split** and the **training loop**.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE E: OVERTRAINING — SOURCE MODEL CREATION                               │
│  (Container: dashed, #F5F5F5)                                                │
│                                                                              │
│  ┌─── Peptide Path ─────────────────────┐ ┌─── SMT Path ──────────────────┐ │
│  │                                       │ │                                │ │
│  │ [Cylinder: peptide_graphs.pkl]        │ │ [Cylinder: smt_graphs.pkl]     │ │
│  │        │                              │ │        │                       │ │
│  │        ▼                              │ │        ▼                       │ │
│  │ [Rounded Rect:                        │ │ [Rounded Rect:                 │ │
│  │  "90% / 10% Split"]                   │ │  "90% / 10% Split"]           │ │
│  │  Fill: #D5E8D4                        │ │  Fill: #D5E8D4                 │ │
│  │        │                              │ │        │                       │ │
│  │   ┌────┴────┐                         │ │   ┌────┴────┐                  │ │
│  │   ▼         ▼                         │ │   ▼         ▼                  │ │
│  │ [Train    [Val                        │ │ [Train    [Val                 │ │
│  │  90%]      10%]                       │ │  90%]      10%]               │ │
│  │ #D5E8D4   #DAE8FC                     │ │ #D5E8D4   #DAE8FC             │ │
│  │   │         │                         │ │   │         │                  │ │
│  │   └────┬────┘                         │ │   └────┬────┘                  │ │
│  │        ▼                              │ │        ▼                       │ │
│  │ ┌──────────────────────────┐          │ │ ┌──────────────────────────┐   │ │
│  │ │  DeepGraphCNN Model      │          │ │ │  DeepGraphCNN Model      │   │ │
│  │ │  (Random Init → Train)   │          │ │ │  (Random Init → Train)   │   │ │
│  │ │  Fill: #FFE6CC            │          │ │ │  Fill: #FFE6CC            │   │ │
│  │ │  Border: 3pt solid       │          │ │ │  Border: 3pt solid       │   │ │
│  │ │                          │          │ │ │                          │   │ │
│  │ │  ┌──────────────────┐   │          │ │ │  ┌──────────────────┐   │   │ │
│  │ │  │ GNN Block        │   │          │ │ │  │ GNN Block        │   │   │ │
│  │ │  │ 4×GraphConv+Sort │   │          │ │ │  │ 4×GraphConv+Sort │   │   │ │
│  │ │  └──────────────────┘   │          │ │ │  └──────────────────┘   │   │ │
│  │ │  ┌──────────────────┐   │          │ │ │  ┌──────────────────┐   │   │ │
│  │ │  │ Readout Block    │   │          │ │ │  │ Readout Block    │   │   │ │
│  │ │  │ Conv+Dense+Sig   │   │          │ │ │  │ Conv+Dense+Sig   │   │   │ │
│  │ │  └──────────────────┘   │          │ │ │  └──────────────────┘   │   │ │
│  │ └──────────────────────────┘          │ │ └──────────────────────────┘   │ │
│  │        │                              │ │        │                       │ │
│  │  [Annotation: Adam LR=1e-4           │ │  [Same annotation]             │ │
│  │   EarlyStopping patience=7            │ │                                │ │
│  │   Restore best weights]               │ │                                │ │
│  │        │                              │ │        │                       │ │
│  │        ▼                              │ │        ▼                       │ │
│  │ [Cylinder w/ star icon:               │ │ [Cylinder w/ star icon:        │ │
│  │  "overtrained_peptide                 │ │  "overtrained_smt              │ │
│  │   _model.h5"]                         │ │   _model.h5"]                  │ │
│  │  Fill: #FFF2CC                        │ │  Fill: #FFF2CC                 │ │
│  │  Border: thick (key artifact!)        │ │  Border: thick                 │ │
│  └───────────────────────────────────────┘ └────────────────────────────────┘ │
│                                                                              │
│  [Callout at bottom center:]                                                 │
│  "These models are NOT evaluated — they exist solely to                      │
│   provide pretrained weights for transfer learning"                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Exact draw.io steps:**

1. **Container** with two internal **sub-containers** (or two grouped areas) side by side — one for the Peptide path, one for the SMT path. Use a faint vertical dashed line to separate them.

2. **90/10 split visualization:**
   - A rounded rectangle labeled "train_test_split (90% / 10%)"
   - Below it: two small rectangles side by side:
     - Left: **"Train 90%"** — fill `#D5E8D4`, width proportionally ~4.5× larger
     - Right: **"Val 10%"** — fill `#DAE8FC`, width proportionally ~0.5× smaller
   - This proportional sizing **visually communicates** the 90/10 ratio

3. **Model block** — a large rectangle with thick 3pt border:
   - Fill: `#FFE6CC` (orange)
   - Label: **"DeepGraphCNN"** (bold 14pt)
   - **Inside the model block**, place two smaller stacked rectangles:
     - Top inner: "GNN Block (4×GraphConv + SortPooling)" — fill slightly darker orange `#FFD699`
     - Bottom inner: "Readout Block (Conv1D + Dense + Sigmoid)" — fill `#FFE6CC`
   - This foreshadows the block separation used later in TL methods

4. **Training annotations** — to the right of the model block, place a text box:
   - `Adam optimizer, LR = 0.0001`
   - `Loss: Binary Crossentropy`
   - `EarlyStopping: patience=7`
   - `Restore best weights`
   - Use 9pt italic gray text

5. **Output cylinder** — larger than normal (e.g., 140×80px) with fill `#FFF2CC`:
   - Label: filename of the `.h5` model
   - Place a small **star icon** (search "star" in draw.io) at the top-right corner to visually mark this as a key artifact
   - Or use a thick gold border (`#B8860B`) to make it stand out

6. **Bottom callout** spanning both paths:
   - Callout shape, fill `#FFF0F0` (very light red for emphasis)
   - Text: *"⚠ These models are NOT evaluated — their sole purpose is to provide pretrained weights for transfer learning"*

---

## 7. Stage F — Baseline Models (Trained from Scratch)

### What happens in the project

The baseline establishes the **performance floor** — what a model achieves with NO transfer learning:

1. For each of the 10 CV folds:
   - Build a **brand new DeepGraphCNN** with the **same architecture** as the overtrained model but with **completely random weights** (no pretrained knowledge)
   - Train on the fold's **training set**, monitor the **validation set** for early stopping
   - Save the model as `baseline_*_fold_X.h5`

The baseline uses the **same hyperparameters** (Adam, LR=1e-4, patience=7) and the **same CV splits** as all TL methods — the only difference is that it starts from scratch.

**Baseline models produced:**
- 10 models for peptide target: `baseline_peptide_fold_{1-10}.h5`
- 10 models for SMT target: `baseline_small_mol_tox_fold_{1-10}.h5`

### draw.io Construction — Stage F

```
┌───────────────────────────────────────────────┐
│  STAGE F: BASELINE (No Transfer Learning)      │
│  (Container: dashed, #F5F5F5)                  │
│                                                │
│  [Cylinder: CV Splits .pkl]                    │
│  (from Stage D)                                │
│        │                                       │
│        ▼                                       │
│  [Loop icon] For fold k = 1..10:               │
│  ┌─────────────────────────────────────┐       │
│  │ [Rounded Rect: Build NEW Model]     │       │
│  │ "Random weight initialization"      │       │
│  │ Fill: #D5E8D4                       │       │
│  │        │                            │       │
│  │        ▼                            │       │
│  │ [Rect: DeepGraphCNN]               │       │
│  │ Fill: #FFE6CC, thick border        │       │
│  │ ALL layers GREEN (trainable)       │       │
│  │ [Inside: GNN Block ✓ Readout ✓]   │       │
│  │        │                            │       │
│  │        ▼                            │       │
│  │ [Rounded Rect: Train on fold k]    │       │
│  │ "Adam LR=1e-4, patience=7"        │       │
│  │ Fill: #D5E8D4                       │       │
│  └─────────────────────────────────────┘       │
│        │                                       │
│        ▼                                       │
│  [Stack of 10 small cylinders:]                │
│  "baseline_fold_1.h5 ... fold_10.h5"          │
│  Fill: #FFF2CC                                 │
│                                                │
│  [Annotation: "NO pretrained knowledge —       │
│   this is the control experiment"]             │
│                                                │
└───────────────────────────────────────────────┘
```

**Exact draw.io steps:**

1. **Container** — narrower than Stages B/E (about half the width), because this will sit **side by side** with Stage G (Transfer Learning) in the final layout.

2. **Loop indicator:** Use a **circular arrow icon** placed at the top of the inner processing area, with text "For k = 1..10" next to it.

3. **Model rectangle** — same style as the overtrained model, but with a **key visual difference**: ALL layers inside should have a **green fill/border** (`#D5E8D4`) to indicate everything is trainable with random initialization. No lock icons anywhere. Place a **checkmark icon** (✓) on both the GNN and Readout blocks.

4. **Output:** Show **10 small cylinders** fanned out horizontally (or as a single cylinder with "×10" label badge), fill `#FFF2CC`.

5. **Important annotation** below the output: *"No pretrained knowledge — this is the control experiment (performance floor)"*

---

## 8. Stage G — Transfer Learning: 4 Methods

### What happens in the project

For each of the 10 CV folds, the **pretrained overtrained model** is loaded and fine-tuned on the target dataset using one of four freezing strategies. This is done in **both directions** (Peptide→SMT and SMT→Peptide).

### Method 1: Freeze GNN Layers

- **Load** the overtrained source model (`.h5`)
- **Freeze** the GNN block (4 GraphConv layers + SortPooling) → weights stay exactly as pretrained
- **Keep trainable** the Readout block (Conv1D, Dense layers) → these adapt to the new domain
- **Train** with LR=1e-4, patience=7

**Hypothesis:** The GNN has learned universal molecular pattern recognition; only the classification head needs domain-specific adjustment.

### Method 2: Freeze Readout Layers

- **Load** the overtrained source model
- **Freeze** the Readout block → classification logic stays as pretrained
- **Keep trainable** the GNN block → graph feature extraction adapts to new molecule types
- **Train** with LR=1e-5 (lower LR to protect pretrained GNN knowledge)

**Hypothesis:** The classification logic transfers across domains; the molecular feature extraction needs updating.

### Method 3: Freeze All + New Output

- **Load** the overtrained source model
- **Freeze ALL layers** (GNN + Readout) — entire model becomes a fixed feature extractor
- **Remove** the original final Dense(1, sigmoid) layer
- **Add a brand new** Dense(1, sigmoid) layer on top of the Dropout layer
- **Train** only this new layer (LR=1e-4)

**Hypothesis:** The entire pretrained pipeline produces good features; only a fresh linear classifier is needed. The most conservative approach — fewest trainable parameters (~129).

### Method 4: Gradual Unfreezing + Discriminative Learning Rates

Three sequential training phases, each unfreezing more layers with progressively smaller learning rates:

| Phase | Trainable | Frozen | LR | Epochs |
|-------|-----------|--------|-----|--------|
| **Phase 1** | Final layers only | GNN + Readout | 1e-3 | 10 |
| **Phase 2** | Readout + Final | GNN only | 1e-4 | 10 |
| **Phase 3** | ALL layers | Nothing | 1e-5 | 10 |

**Hypothesis:** Start by adapting the output, then gradually allow deeper layers to fine-tune with smaller learning rates. This prevents **catastrophic forgetting** — the risk of destroying useful pretrained representations through aggressive weight updates.

### Bidirectional Transfer

All 4 methods are applied in **both directions**:

```
Direction 1: Peptide → SMT
  Source = overtrained_peptide_model.h5
  Target = Small Molecule Toxicity dataset
  → 4 methods × 10 folds = 40 models

Direction 2: SMT → Peptide  
  Source = overtrained_smt_model.h5
  Target = Peptide dataset
  → 4 methods × 10 folds = 40 models
```

### draw.io Construction — Stage G

This is the **visual centerpiece** of the entire diagram. It should be placed **side by side** with Stage F.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  STAGE G: TRANSFER LEARNING — 4 METHODS                                      │
│  (Container: dashed, #F5F5F5)                                                │
│                                                                              │
│  [Cylinder w/ star: overtrained_model.h5]                                    │
│  (from Stage E)                                                              │
│  Fill: #FFF2CC                                                               │
│            │                                                                  │
│            │ THICK DASHED ORANGE ARROW                                        │
│            │ labeled "Load pretrained weights"                                │
│            ▼                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐     │
│  │  [Rounded Rect: "Load Source Model"]                                 │     │
│  │  Fill: #D5E8D4                                                       │     │
│  │            │                                                          │     │
│  │     ┌──────┼──────┬──────────┐                                       │     │
│  │     ▼      ▼      ▼          ▼                                        │     │
│  │                                                                       │     │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌──────────┐                               │     │
│  │  │ M1  │ │ M2  │ │ M3  │ │ M4       │                               │     │
│  │  │     │ │     │ │     │ │          │                               │     │
│  │  │ GNN │ │ GNN │ │ GNN │ │ Phase 1: │                               │     │
│  │  │ 🔒  │ │ 🔓  │ │ 🔒  │ │ Final🔓  │                               │     │
│  │  │ RED │ │ GRN │ │ RED │ │ GRN     │                               │     │
│  │  │─────│ │─────│ │─────│ │─────────│                               │     │
│  │  │Read │ │Read │ │Read │ │ Phase 2: │                               │     │
│  │  │ 🔓  │ │ 🔒  │ │ 🔒  │ │ Read 🔓  │                               │     │
│  │  │ GRN │ │ RED │ │ RED │ │ ORANGE  │                               │     │
│  │  │─────│ │─────│ │─────│ │─────────│                               │     │
│  │  │Out  │ │Out  │ │NEW  │ │ Phase 3: │                               │     │
│  │  │ 🔓  │ │ 🔒  │ │Dense│ │ GNN 🔓   │                               │     │
│  │  │ GRN │ │ RED │ │ 🔓  │ │ YELLOW  │                               │     │
│  │  │     │ │     │ │ GRN │ │         │                               │     │
│  │  └──┬──┘ └──┬──┘ └──┬──┘ └────┬────┘                               │     │
│  │     │       │       │         │                                      │     │
│  │     └───────┴───────┴─────────┘                                      │     │
│  │                 │                                                     │     │
│  │                 ▼                                                      │     │
│  │     [Loop icon] For fold k = 1..10:                                  │     │
│  │     [Rounded Rect: Fine-tune on target data]                         │     │
│  │     Fill: #D5E8D4                                                     │     │
│  └──────────────────────────────────────────────────────────────────────┘     │
│            │                                                                  │
│            ▼                                                                  │
│  [Stack: 4 × 10 = 40 cylinders]                                             │
│  "method_fold_{1-10}.h5"                                                     │
│  (shown as 4 groups of cylinder stacks)                                      │
│  Fill: #FFF2CC                                                               │
│                                                                              │
│  ── DIRECTION LABELS ──                                                      │
│  [Left arrow label: "Peptide → SMT"]                                         │
│  [Right arrow label: "SMT → Peptide"]                                        │
│  (Or use two horizontal arrows side by side to show bidirectionality)        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Exact draw.io steps:**

1. **Container** — wider than Stage F, same height. Place them **side by side** with Stage F to the left and Stage G to the right. Draw a horizontal bracket or line above both labeled "STAGE F + G: MODEL TRAINING" if desired.

2. **Overtrained model cylinder** at the top, connected with a **thick dashed orange arrow** (the weight transfer arrow — this is the most important visual connector in the entire diagram):
   - Arrow style: dashed, color `#FF8C00` (dark orange), width 3pt, filled arrowhead
   - Label on arrow: **"Transfer pretrained weights"**

3. **Four method columns** inside the container, each as a narrow vertical stack:

   **Method 1 — "Freeze GNN":**
   - Top sub-block (GNN): fill `#F8CECC` (red), 🔒 lock icon overlay
   - Bottom sub-block (Readout): fill `#D5E8D4` (green), 🔓 unlock icon overlay
   - Small text below: "LR: 1e-4"

   **Method 2 — "Freeze Readout":**
   - Top sub-block (GNN): fill `#D5E8D4` (green), 🔓 unlock
   - Bottom sub-block (Readout): fill `#F8CECC` (red), 🔒 lock
   - Small text below: "LR: 1e-5"

   **Method 3 — "Freeze All + New Output":**
   - Top sub-block (GNN): fill `#F8CECC` (red), 🔒 lock
   - Middle sub-block (Readout): fill `#F8CECC` (red), 🔒 lock
   - Bottom sub-block: fill `#D5E8D4` (green), **dashed border** (to indicate it's a NEW layer), label "NEW Dense(1)", 🔓 unlock
   - Small text: "LR: 1e-4"

   **Method 4 — "Gradual Unfreezing":**
   - This column should be **slightly wider** than the others
   - Show it as **3 horizontal slices** with phase labels:
     - Top slice (GNN): fill `#FFF2CC` (yellow), label "Phase 3, LR=1e-5"
     - Middle slice (Readout): fill `#FFE6CC` (orange), label "Phase 2, LR=1e-4"
     - Bottom slice (Final): fill `#D5E8D4` (green), label "Phase 1, LR=1e-3"
   - Add a small **downward arrow** along the right side with "1e-3 → 1e-4 → 1e-5" showing the LR schedule

4. **Legend box** below the four columns:

   | Color | Meaning |
   |-------|---------|
   | 🟢 Green `#D5E8D4` | Trainable — weights updated during fine-tuning |
   | 🔴 Red `#F8CECC` | Frozen — weights locked from source model |
   | 🟡 Yellow `#FFF2CC` | Unfrozen later (Phase 3) |
   | 🟠 Orange `#FFE6CC` | Unfrozen in middle phase (Phase 2) |

   Create this as a small draw.io **Table** (Insert → Table).

5. **Output models** at the bottom: Show **4 small groups** of cylinder stacks (one per method), each labeled with the method name and "×10 folds". All fill `#FFF2CC`.

6. **Bidirectional arrows** at the very bottom:
   - Place two thick horizontal arrows (one pointing right, one pointing left)
   - Right arrow: "Peptide → SMT" (label above)
   - Left arrow: "SMT → Peptide" (label below)
   - Or use a **double-headed arrow** labeled "Bidirectional Transfer"

---

## 9. Stage H — Evaluation & Statistical Analysis

### What happens in the project

All trained models (baseline + 4 TL methods × 10 folds = **50 models per direction**) are evaluated:

1. **For each fold k** (k = 1..10):
   - Load all 5 models for fold k (1 baseline + 4 TL methods)
   - Load the **test set** for fold k from the saved CV splits
   - Each model predicts **P(toxic)** for each molecule in the test set
   - Apply threshold 0.5 to convert probabilities → binary predictions
   - Compute **6 metrics** per model:

| Metric | Full Name | What It Captures |
|--------|-----------|-----------------|
| **ROC-AUC** | Area Under ROC Curve | Overall ranking quality (threshold-independent) |
| **GM** | Geometric Mean | √(Sensitivity × Specificity) — good for imbalanced data |
| **Precision** | Positive Predictive Value | Of those predicted toxic, how many truly are |
| **Recall** | Sensitivity / TPR | Of truly toxic, how many we catch |
| **F1** | Harmonic Mean of P & R | Balance between precision and recall |
| **MCC** | Matthews Correlation Coefficient | Overall binary classification quality [-1, +1] |

2. **Aggregate across folds:** Mean ± Standard Deviation for each metric per model type

3. **Statistical testing:**
   - **Friedman test:** Non-parametric test — are there significant differences between the 5 models?
   - If p < 0.05 → **Nemenyi post-hoc test:** Which specific pairs of models differ significantly?

4. **Visualization:**
   - **Box-and-whisker plots:** Distribution of each metric across 10 folds for each model
   - **Radar plots:** Multi-metric comparison on a single chart across all models and configurations

### draw.io Construction — Stage H

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  STAGE H: EVALUATION & STATISTICAL ANALYSIS                                   │
│  (Container: dashed, #F5F5F5)                                                │
│                                                                              │
│  ── INPUTS ──                                                                │
│  [5 cylinder stacks in a row, each a different color:]                       │
│   Baseline(×10)  M1(×10)  M2(×10)  M3(×10)  M4(×10)                        │
│   blue           red      green    red      yellow                           │
│                                                                              │
│  [Cylinder: cv_splits.pkl] ──── dashed blue arrow into loop                  │
│                                                                              │
│  ── EVALUATION LOOP ──                                                       │
│  ┌──────────────────────────────────────────────────────────────┐            │
│  │  [Loop icon] For each fold k = 1..10                         │            │
│  │                                                               │            │
│  │  [Rounded Rect: Load 5 models for fold k]                    │            │
│  │  [Rounded Rect: Load test set for fold k]                    │            │
│  │  [Rounded Rect: Predict → P(toxic) → threshold 0.5]         │            │
│  │  [Hexagon: Compute 6 Metrics]  Fill: #E1D5E7                │            │
│  │  Inside: ROC-AUC | GM | Precision | Recall | F1 | MCC       │            │
│  │                                                               │            │
│  └──────────────────────────────────────────────────────────────┘            │
│            │                                                                  │
│            ▼                                                                  │
│  [Rounded Rect: Aggregate — Mean ± Std per metric, per model]                │
│  Fill: #D5E8D4                                                               │
│            │                                                                  │
│      ┌─────┴─────────────────────┐                                           │
│      ▼                           ▼                                            │
│  ┌──────────────┐       ┌────────────────────────┐                           │
│  │ STATISTICAL  │       │ VISUALIZATION           │                           │
│  │ TESTING      │       │                         │                           │
│  │              │       │ [Rect: Box Plots]       │                           │
│  │ [Rect:       │       │  (placeholder image)    │                           │
│  │  Friedman]   │       │                         │                           │
│  │      │       │       │ [Rect: Radar Plots]     │                           │
│  │      ▼       │       │  (placeholder image)    │                           │
│  │ [Diamond:    │       │                         │                           │
│  │  p < 0.05?] │       │ Fill: #E1D5E7           │                           │
│  │  │Yes  │No  │       └────────────────────────┘                           │
│  │  ▼     ▼    │                                                             │
│  │ [Rect: [End]│                                                             │
│  │ Nemenyi]    │                                                             │
│  │      │      │                                                             │
│  │      ▼      │                                                             │
│  │ [Hexagon:   │                                                             │
│  │  Signif.    │                                                             │
│  │  Matrix]    │                                                             │
│  │ Fill:#E1D5E7│                                                             │
│  └──────────────┘                                                            │
│                                                                              │
│  ── EXAMPLE OUTPUT TABLE ──                                                  │
│  ┌────────────┬──────────┬──────────┬───────┐                                │
│  │            │ ROC-AUC  │ GM       │ MCC   │                                │
│  ├────────────┼──────────┼──────────┼───────┤                                │
│  │ Baseline   │ 0.82±.03 │ 0.75±.04 │ ...   │                                │
│  │ Method 1   │ 0.85±.02 │ 0.78±.03 │ ...   │                                │
│  │ ...        │ ...      │ ...      │ ...   │                                │
│  └────────────┴──────────┴──────────┴───────┘                                │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Exact draw.io steps:**

1. **Container** — full width, taller than previous stages.

2. **Model inputs at top:** Place **5 cylinder stacks** in a horizontal row. Each stack represents one model type across 10 folds. Give each a **distinct color accent**:
   - Baseline: blue accent (`#DAE8FC`)
   - Method 1: red-green split
   - Method 2: green-red split
   - Method 3: all-red with green tip
   - Method 4: gradient yellow-orange-green
   - OR simply use the model's primary color and add a text label below

3. **Evaluation loop:**
   - Use a **rounded-corner container** with a **circular arrow icon** at the top-right corner
   - Inside: 4 rounded rectangles stacked vertically, connected by arrows
   - The **Hexagon** (for metrics) should be prominent — fill `#E1D5E7` (purple), with the 6 metric names inside it

4. **Fork into two branches:**
   - **Left branch: Statistical Testing**
     - Rectangle "Friedman Test" → Diamond "p < 0.05?" → if Yes → Rectangle "Nemenyi Post-Hoc" → Hexagon "Significance Matrix"
     - if No → small rectangle "No significant difference" (end)
   - **Right branch: Visualization**
     - Two rectangles representing plots (box plots and radar plots)
     - Fill `#E1D5E7` (purple)
     - *Optional:* Insert actual placeholder images of box plots/radar charts

5. **Example output table** at the bottom:
   - Use draw.io's Table shape (Insert → Table): 5 rows × 4+ columns
   - Fill header row with `#E1D5E7`
   - This gives the viewer a concrete idea of what the final output looks like

---

## 10. Full Single-Page Diagram: Assembling Everything

### Master Layout

The complete diagram flows **top to bottom** with these spatial relationships:

```
                    ┌───────────────────────────────┐
                    │        STAGE A: RAW DATA       │
                    │  📄 Peptide XLSX    📄 SMT XLSX │
                    │  (SMILES + Labels)              │
                    └───────────────┬───────────────┘
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                    ┌────▼─────┐    ┌──────────▼───────────────┐
                    │ STAGE C: │    │     STAGE B:              │
                    │ Shared   │───►│     PREPROCESSING         │
                    │ Vocab(27)│    │  SMILES → RDKit → Graph   │
                    └──────────┘    │  → StellarGraph → .pkl    │
                                    └──────────┬───────────────┘
                                               │
                                    ┌──────────▼───────────────┐
                                    │     STAGE D:              │
                                    │  10-Fold CV Splits        │
                                    │  → cv_splits.pkl          │
                                    └──────────┬───────────────┘
                                               │
                                    ┌──────────▼───────────────┐
                                    │     STAGE E:              │
                                    │  OVERTRAINED MODELS       │
                                    │  90/10 Split → Train      │
                                    │  → .h5 source models      │
                                    └──────────┬───────────────┘
                                               │
                              ┌────────────────┴─────────────────┐
                              │                                   │
                   ┌──────────▼──────────┐         ┌──────────────▼──────────────┐
                   │    STAGE F:          │         │    STAGE G:                  │
                   │    BASELINE          │         │    TRANSFER LEARNING         │
                   │    (from scratch)    │         │    4 Methods × 2 Directions  │
                   │    10 models         │         │    80 models                 │
                   └──────────┬──────────┘         └──────────────┬──────────────┘
                              │                                   │
                              └────────────────┬──────────────────┘
                                               │
                                    ┌──────────▼───────────────┐
                                    │     STAGE H:              │
                                    │     EVALUATION            │
                                    │  Metrics + Friedman +     │
                                    │  Nemenyi + Plots          │
                                    └───────────────────────────┘
```

**Exact draw.io assembly steps:**

1. **Page setup:** File → Page Setup → set to **A2 landscape** (or custom 2000×3000px) for adequate space.

2. **Use Layers** (View → Layers):
   - **Layer 0: Background** — all gray containers/stage boxes
   - **Layer 1: Shapes** — all shapes, icons, text within containers
   - **Layer 2: Connectors** — all arrows and lines
   - This lets you easily select and move elements without accidentally grabbing the wrong layer

3. **Place stages top to bottom**, with generous vertical spacing (~80px between stages).

4. **Stages F and G sit side by side** — they are at the same vertical level. Draw a horizontal dashed line or bracket above them labeled "MODEL TRAINING" to visually group them.

5. **Key connector styles:**
   - Stages A→B: solid black arrow (data flows)
   - Stage C→B: dashed blue arrow from the side (vocabulary feeds into preprocessing)
   - Stages B→D: solid black arrow (graphs feed into splitter)
   - Stages D→E: solid black arrow (split data used for overtraining context, even though overtraining uses all data — the splits are generated from the same preprocessed data)
   - Stage E→G: **THICK DASHED ORANGE ARROW** ← this is the most important visual connection — it represents the weight transfer. Make this arrow **3pt width, dashed, color #FF8C00**, with a label "Pretrained weights"
   - Stage E→F: **NO arrow** (baseline doesn't use pretrained weights). The absence of this connection is visually meaningful
   - Stages F+G→H: solid black arrows merging into one

6. **Legend box** in the bottom-right corner:
   - Small rectangle with the color palette, shape dictionary, and connector type reference
   - Include lock/unlock symbols with explanations

---

## 11. Shape & Icon Quick-Reference Table

A consolidated cheat-sheet for building the diagram:

| Element | draw.io Shape | Fill Color | Icon Overlay | Label Format |
|---------|--------------|------------|--------------|-------------|
| XLSX dataset file | **Document** (folded corner) | `#DAE8FC` | None | Filename, 11pt |
| SMILES badge | Small **rounded rect** | `#B0C4DE` | None | "SMILES + Label", 9pt |
| Molecule example | 3 **Ellipses** + **Lines** | Gray/Red | None | Atom symbols (C, O, N) |
| Processing step | **Rounded Rectangle** | `#D5E8D4` | None | Action name, 11pt |
| Vocabulary table | **Table** or **Rectangle** | `#FFF2CC` | None | Element→Index map |
| CV fold | Small **Rectangle** | `#DAE8FC` | None | "F1", "F2", ..., 9pt |
| Split bar (90/10) | Two adjacent **Rectangles** | Green/Blue | None | "90%" / "10%" |
| Neural network model | **Rectangle** thick border (3pt) | `#FFE6CC` | None | "DeepGraphCNN", 14pt Bold |
| GNN block (inside model) | **Rectangle** | `#FFD699` | None | "GNN Block", 11pt |
| Readout block (inside model) | **Rectangle** | `#FFE6CC` | None | "Readout Block", 11pt |
| Frozen layer | **Rectangle** | `#F8CECC` | 🔒 Lock icon | Layer name, 10pt |
| Trainable layer | **Rectangle** | `#D5E8D4` | 🔓 Unlock icon | Layer name, 10pt |
| New layer (Method 3) | **Rectangle** dashed border | `#D5E8D4` | "NEW" badge | "Dense(1, sigmoid)", 10pt |
| Saved model (.h5) | **Cylinder** | `#FFF2CC` | ⭐ Star (for overtrained) | Filename, 10pt |
| Saved data (.pkl) | **Cylinder** | `#FFF2CC` | None | Filename, 10pt |
| Metric result | **Hexagon** | `#E1D5E7` | None | Metric name, 10pt |
| Decision point | **Diamond** | White | None | Question, 10pt |
| Loop indicator | **Circular arrow** icon | None | Overlaid on container corner | "k=1..10" |
| Weight transfer arrow | **Connector** (dashed, thick) | `#FF8C00` | Filled arrowhead | "Pretrained weights" |
| Annotation/note | **Callout** bubble | `#FFFACD` | None | Italic 9pt gray |
| Stage container | **Container** (dashed border) | `#F5F5F5` | None | Stage title in header, 16pt Bold |
| Result table | **Table** shape | `#E1D5E7` header | None | Metric values |
| Box plot placeholder | **Rectangle** | `#E1D5E7` | None | "Box Plot" |
| Radar plot placeholder | **Circle** or **Polygon** | `#E1D5E7` | None | "Radar Plot" |

---

## Summary: The Complete Story in One Sentence Per Stage

| Stage | One-Line Summary |
|-------|-----------------|
| **A** | Two XLSX files contain molecules as SMILES strings with toxic/non-toxic labels |
| **B** | SMILES are parsed by RDKit into molecular graphs (atoms=nodes, bonds=edges) with 27-dim one-hot node features |
| **C** | A shared 27-element atom vocabulary ensures identical feature spaces — the bridge for weight transfer |
| **D** | Stratified 10-fold CV splits are generated once and shared by ALL models for fair comparison |
| **E** | Overtrained source models are trained on 100% of each dataset (90/10 train-val) to maximize transferred knowledge |
| **F** | Baseline models train from random initialization — the control experiment (performance floor) |
| **G** | Four TL methods (freeze GNN / freeze readout / freeze all+new / gradual unfreeze) fine-tune the source model on the target domain, in both directions |
| **H** | All 50 models per direction are evaluated with 6 metrics, Friedman + Nemenyi statistical tests, and visualized with box/radar plots |
