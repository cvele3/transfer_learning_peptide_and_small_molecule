# Draw.io Diagram Design Guide — Transfer Learning Project Schematics

This document describes how to create a series of **draw.io diagrams** that visually communicate the transfer learning pipeline. Each section specifies the shapes, icons, colors, connectors, and layout to use.

---

## Table of Contents

1. [Diagram 1: Complete Project Pipeline (High-Level Overview)](#diagram-1-complete-project-pipeline-high-level-overview)
2. [Diagram 2: Data Preprocessing — SMILES to Graph Conversion](#diagram-2-data-preprocessing--smiles-to-graph-conversion)
3. [Diagram 3: DeepGraphCNN Architecture](#diagram-3-deepgraphcnn-architecture)
4. [Diagram 4: Overtraining Phase](#diagram-4-overtraining-phase)
5. [Diagram 5: Transfer Learning Methods — Comparative View](#diagram-5-transfer-learning-methods--comparative-view)
6. [Diagram 6: Gradual Unfreezing Timeline](#diagram-6-gradual-unfreezing-timeline)
7. [Diagram 7: Evaluation Pipeline](#diagram-7-evaluation-pipeline)
8. [Diagram 8: Bidirectional Transfer & Model Configurations](#diagram-8-bidirectional-transfer--model-configurations)
9. [General Style Guide](#general-style-guide)

---

## General Style Guide

### Color Palette

| Purpose | Color (Hex) | Usage |
|---------|-------------|-------|
| Data / Input | `#DAE8FC` (light blue) | Datasets, XLSX files, raw data |
| Processing / Transform | `#D5E8D4` (light green) | Preprocessing steps, RDKit, feature creation |
| Model / Neural Network | `#FFE6CC` (light orange) | Model blocks, layers, architecture |
| Frozen Layer | `#F8CECC` (light red) | Frozen/locked layers in TL |
| Trainable Layer | `#D5E8D4` (light green) | Unfrozen/trainable layers in TL |
| Output / Result | `#E1D5E7` (light purple) | Predictions, metrics, evaluation |
| Storage / File | `#FFF2CC` (light yellow) | .h5 files, .pkl files, saved artifacts |
| Phase / Stage | `#F5F5F5` (light gray) | Background containers grouping steps |

### Font and Text

- **Title font:** 16-18pt, Bold
- **Shape labels:** 11-12pt, Regular
- **Annotation text:** 9-10pt, Italic, Gray
- **Use font:** Arial or Helvetica

### Connectors

- **Flow arrows:** Solid, black, 2pt width, filled arrowhead
- **Data flow:** Dashed, blue, 1pt width
- **Annotation lines:** Dotted, gray, 1pt width
- **Decision/branch arrows:** Standard solid with labels on the arrow

### Shape Conventions

| Concept | draw.io Shape | Notes |
|---------|--------------|-------|
| Data file (XLSX/PKL) | **Cylinder** (database shape) or **Document** shape | Light blue fill |
| Processing step | **Rounded Rectangle** | Light green fill |
| Neural network layer | **Rectangle** with rounded corners | Orange fill, stacked when multiple |
| Model (complete) | **Rectangle** with thick border | Orange fill, bold label |
| Frozen indicator | **Lock icon** (🔒) or red border/fill | Place next to frozen layers |
| Trainable indicator | **Unlock icon** (🔓) or green border/fill | Place next to trainable layers |
| Decision point | **Diamond** | Standard flowchart decision |
| Grouping container | **Dashed rectangle** or **swimlane** | Light gray fill, groups related items |
| Metric / Result | **Hexagon** or **parallelogram** | Purple fill |
| Loop (10 folds) | **Circular arrow** annotation or **loop symbol** | Gray, placed near CV loop |

---

## Diagram 1: Complete Project Pipeline (High-Level Overview)

**Purpose:** One-page overview showing all phases from raw data to final evaluation.

**Layout:** Top-to-bottom vertical flow, 5 main stages in a swimlane-like layout.

### Structure:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        STAGE 1: DATA PREPARATION                       │
│  (Gray background container)                                           │
│                                                                        │
│  [Cylinder: XLSX Peptide]    [Cylinder: XLSX SMT]                      │
│        │                           │                                   │
│        └─────────┬─────────────────┘                                   │
│                  ▼                                                      │
│  [Rounded Rect: Element Discovery] ──► [Rect: Shared Vocabulary (27)]  │
│                  │                                                      │
│                  ▼                                                      │
│  [Rounded Rect: SMILES → Graph Conversion (RDKit + StellarGraph)]      │
│                  │                                                      │
│        ┌────────┴────────┐                                             │
│        ▼                 ▼                                              │
│  [Cylinder: Peptide      [Cylinder: SMT                                │
│   Graphs .pkl]            Graphs .pkl]                                  │
│                                                                        │
│  [Rounded Rect: Generate 10-Fold CV Splits]                            │
│        │                 │                                              │
│        ▼                 ▼                                              │
│  [Cylinder: CV Splits    [Cylinder: CV Splits                          │
│   Peptide .pkl]           SMT .pkl]                                     │
└────────────────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     STAGE 2: SOURCE MODEL TRAINING                     │
│  (Gray background container)                                           │
│                                                                        │
│  [Cylinder: Peptide Graphs]  ──►  [Rect: Train DeepGraphCNN           │
│                                     100% data, EarlyStopping]          │
│                                     ──►  [Cylinder: Peptide .h5]       │
│                                                                        │
│  [Cylinder: SMT Graphs]      ──►  [Rect: Train DeepGraphCNN           │
│                                     100% data, EarlyStopping]          │
│                                     ──►  [Cylinder: SMT .h5]           │
└────────────────────────────────────────────────────────────────────────┘
                  │
        ┌─────────┴──────────┐
        ▼                    ▼
┌───────────────┐  ┌───────────────────────────────────────────────────┐
│  STAGE 3a:    │  │  STAGE 3b: TRANSFER LEARNING                     │
│  BASELINE     │  │  (Gray container)                                 │
│               │  │                                                   │
│  [Rect: New   │  │  [Cylinder: Source .h5] ──► [Rect: Load Model]   │
│   Random      │  │                                   │               │
│   Model]      │  │        ┌──────┬──────┬──────┐    │               │
│      │        │  │        ▼      ▼      ▼      ▼    │               │
│      ▼        │  │  [M1:Freeze [M2:Freeze [M3:Freeze [M4:Gradual   │
│  [Rect: Train │  │   GNN]    Readout]  All+New]  Unfreeze]          │
│   10-fold CV] │  │        │      │      │      │                     │
│      │        │  │        └──────┴──────┴──────┘                     │
│      ▼        │  │               │                                   │
│  [Cylinder:   │  │               ▼                                   │
│   10 .h5      │  │  [Rect: Fine-tune 10-fold CV]                    │
│   models]     │  │               │                                   │
│               │  │               ▼                                   │
│               │  │  [Cylinder: 4×10 = 40 .h5 models per direction]  │
└───────────────┘  └───────────────────────────────────────────────────┘
        │                    │
        └─────────┬──────────┘
                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        STAGE 4: EVALUATION                             │
│  (Gray background container)                                           │
│                                                                        │
│  [Rect: Load all 50 models (1 baseline + 4 methods) × 10 folds]       │
│                  │                                                      │
│                  ▼                                                      │
│  [Rect: Predict on test set per fold]                                  │
│                  │                                                      │
│                  ▼                                                      │
│  [Hexagon: 6 Metrics] ──► ROC-AUC, GM, Precision, Recall, F1, MCC    │
│                  │                                                      │
│                  ▼                                                      │
│  [Rect: Friedman Test + Nemenyi Post-Hoc]                              │
│                  │                                                      │
│                  ▼                                                      │
│  [Parallelogram: Box Plots + Radar Plots + Significance Tables]        │
└────────────────────────────────────────────────────────────────────────┘
```

### draw.io construction notes:

1. Create a **vertical flow** from top to bottom
2. Use **swimlane containers** (Insert → Advanced → Container) for each stage with a gray background (`#F5F5F5`)
3. Give each stage container a **bold title label** at the top
4. Use **cylinder** shapes from General shape library for all data files (`.xlsx`, `.pkl`, `.h5`)
5. Use **rounded rectangles** for processing/computation steps
6. The branching from Stage 2 into Stages 3a and 3b should use **two parallel flow arrows**
7. Add small **annotation text** boxes (italic, gray) near key elements explaining "why"

---

## Diagram 2: Data Preprocessing — SMILES to Graph Conversion

**Purpose:** Detailed view of how a single molecule goes from SMILES to a graph object.

**Layout:** Horizontal flow, left-to-right, with a concrete molecule example.

### Structure:

```
LEFT SIDE (Input):
  [Document shape: XLSX File]
  Label: "MolToxPredDataset.xlsx"
  Annotation below: "Columns: SMILES, Toxicity"
        │
        ▼
  [Rectangle: Example Row]
  Label: SMILES="CCO", Toxicity=1
        │
        ▼

MIDDLE (Processing):
  [Rounded Rect with Green fill: RDKit Parser]
  Label: "Chem.MolFromSmiles('CCO')"
  Annotation: "Converts string → Mol object"
        │
        ├─────────────────────────────────┐
        ▼                                 ▼
  [Rounded Rect: Extract Bonds]     [Rounded Rect: Extract Atoms]
  Label: "Get edges (both          Label: "One-hot encode each
  directions)"                     atom using vocabulary"
  Show example:                    Show example:
  "[(0,1),(1,0),(1,2),(2,1)]"     "C → [0,1,0,...,0]"
                                   "O → [0,0,1,...,0]"
        │                                 │
        └────────────┬────────────────────┘
                     ▼
  [Rounded Rect: Build StellarGraph]
  Label: "StellarGraph(nodes=features, edges=edges_df)"
        │
        ▼

RIGHT SIDE (Output):
  [Custom drawing: A small graph visualization]
  Three circles (nodes) labeled C, C, O
  Connected with lines (edges)
  Below the graph: annotation "StellarGraph object"
        │
        ▼
  [Cylinder: saved as .pkl]
```

### draw.io construction notes:

1. Place a **small molecule structural drawing** (3 circles connected) in the right section — use **ellipse shapes** for atoms and **lines** for bonds
2. Color the atom circles: Carbon = gray, Oxygen = red (chemistry convention)
3. Next to each atom circle, show the one-hot vector as a small **table/matrix** annotation
4. Use a **curly brace** shape (from the General library) to group "Node Features Matrix" and "Edge DataFrame" as the two components feeding into StellarGraph
5. Add a **small info box** (rounded rect, blue fill) explaining: "The same 27-element vocabulary is used for BOTH peptide and small molecule datasets — this is what makes weight transfer possible"

---

## Diagram 3: DeepGraphCNN Architecture

**Purpose:** Show the complete model architecture in a clean neural-network style diagram.

**Layout:** Vertical, top-to-bottom, with two visually distinct blocks.

### Structure:

```
TOP: Input
  [Parallelogram: Graph Input]
  Label: "Molecular Graph (node features + adjacency)"
  Color: Light blue
        │
        ▼

BLOCK 1: GNN LAYERS (orange dashed container)
  Title: "Graph Feature Extraction (GNN Block)"
  
  [Rectangle: GraphConv Layer 1] ─── tanh
        │
        ▼
  [Rectangle: GraphConv Layer 2] ─── tanh
        │
        ▼
  [Rectangle: GraphConv Layer 3] ─── tanh
        │
        ▼
  [Rectangle: GraphConv Layer 4] ─── tanh
        │
        ▼
  [Rectangle: SortPooling (k=25)]
  
  Side annotation (italic): "Each node aggregates info from neighbors.
  4 layers = 4-hop neighborhood. SortPooling creates fixed-size output."
        │
        ▼

BLOCK 2: READOUT LAYERS (orange dashed container)
  Title: "Classification Head (Readout Block)"
  
  [Rectangle: Conv1D (16 filters)]
        │
        ▼
  [Rectangle: MaxPool1D]
        │
        ▼
  [Rectangle: Conv1D (32 filters)]
        │
        ▼
  [Rectangle: Flatten]
        │
        ▼
  [Rectangle: Dense (128, ReLU)]
        │
        ▼
  [Rectangle: Dropout (0.2)]
        │
        ▼
  [Rectangle: Dense (1, Sigmoid)]
  
  Side annotation (italic): "Processes the sorted node embeddings,
  extracts patterns, produces toxicity probability."

BOTTOM: Output
  [Parallelogram: Output]
  Label: "P(toxic) ∈ [0, 1]"
  Color: Light purple
```

### draw.io construction notes:

1. Make the **two blocks** clearly distinct — use **dashed-border containers** with rounded corners
2. Label Block 1 as "GNN Block" and Block 2 as "Readout Block" — these names will be referenced in the TL diagrams
3. Each layer rectangle should be the same width, stacked vertically with small gaps
4. Use a **gradient shade** going from darker orange at the bottom (closer to input) to lighter orange at top for GNN layers — this visually communicates "depth"
5. Add **dimension annotations** on the right side of each layer:
   - After GraphConv 1: `[N, layer_sizes[0]]`
   - After SortPooling: `[k, sum(layer_sizes)]`
   - After Flatten: `[1D vector]`
   - After final Dense: `[1]`
6. Consider adding a **small bracket** on the left side labeling the entire GNN block as "Layers that learn WHAT patterns matter" and the Readout block as "Layers that learn HOW to classify"

---

## Diagram 4: Overtraining Phase

**Purpose:** Show how the source model is created from raw data.

**Layout:** Horizontal flow with training loop visualization.

### Structure:

```
[Cylinder: Peptide/SMT Graph .pkl]
        │
        ▼
[Rounded Rect: 90/10 Train-Val Split]
        │
        ├──────────────────────┐
        ▼                      ▼
[Rect: Training Set (90%)]  [Rect: Validation Set (10%)]
        │                      │
        └───────┬──────────────┘
                ▼
[Large Rect (orange): DeepGraphCNN Model]
Label: "Random initialization → Adam (LR=0.0001)"
        │
        ▼
[Rounded Rect with circular arrow: Training Loop]
Inside the loop:
  "Forward pass → Loss (Binary Crossentropy) → Backprop → Update weights"
  "Monitor val_loss, patience=7"
Side annotation: "max 10,000 epochs, early stopping triggers much sooner"
        │
        ▼
[Diamond: Early stopping triggered?]
        │ Yes
        ▼
[Rect: Restore best weights]
        │
        ▼
[Cylinder (yellow): Saved Model .h5]
Label: "overtrained_peptide_model.h5"
Annotation: "Contains: architecture + all trained weights"
```

### draw.io construction notes:

1. Use a **loop/cycle icon** (circular arrow from Arrows library) next to the training rectangle to show iterative training
2. Show the **early stopping** mechanism as a diamond decision shape breaking out of the loop
3. The final `.h5` file should be visually prominent — perhaps a **larger cylinder** with a **star/highlight icon** to indicate this is the key artifact
4. Add a **text annotation** at the bottom: "This model is NOT evaluated — its sole purpose is to provide pretrained weights for transfer learning"

---

## Diagram 5: Transfer Learning Methods — Comparative View

**Purpose:** Side-by-side comparison of all 4 methods + baseline, showing which layers are frozen/trainable.

**Layout:** 5 columns (one per method), each showing the same architecture with different coloring.

### Structure:

```
COLUMN 1: BASELINE          COLUMN 2: METHOD 1        COLUMN 3: METHOD 2
───────────────────         ──────────────────        ──────────────────
Title: "No Transfer"        Title: "Freeze GNN"       Title: "Freeze Readout"

[GCN 1] 🟢 Random          [GCN 1] 🔴 Frozen        [GCN 1] 🟢 Trainable
[GCN 2] 🟢 Random          [GCN 2] 🔴 Frozen        [GCN 2] 🟢 Trainable
[GCN 3] 🟢 Random          [GCN 3] 🔴 Frozen        [GCN 3] 🟢 Trainable
[GCN 4] 🟢 Random          [GCN 4] 🔴 Frozen        [GCN 4] 🟢 Trainable
[Sort]  🟢 Random          [Sort]  🔴 Frozen         [Sort]  🟢 Trainable
──────────────────          ──────────────────        ──────────────────
[Conv1D] 🟢 Random         [Conv1D] 🟢 Trainable    [Conv1D] 🔴 Frozen
[MaxP]   🟢 Random         [MaxP]   🟢 Trainable    [MaxP]   🔴 Frozen
[Conv1D] 🟢 Random         [Conv1D] 🟢 Trainable    [Conv1D] 🔴 Frozen
[Flat]   🟢 Random         [Flat]   🟢 Trainable    [Flat]   🔴 Frozen
[Dense]  🟢 Random         [Dense]  🟢 Trainable    [Dense]  🔴 Frozen
[Drop]   🟢 Random         [Drop]   🟢 Trainable    [Drop]   🔴 Frozen
[Dense]  🟢 Random         [Dense]  🟢 Trainable    [Dense]  🔴 Frozen
──────────────────          ──────────────────        ──────────────────
LR: 1e-4                   LR: 1e-4                  LR: 1e-5


COLUMN 4: METHOD 3          COLUMN 5: METHOD 4
──────────────────          ──────────────────
Title: "Freeze All"         Title: "Gradual Unfreeze"

[GCN 1] 🔴 Frozen          [GCN 1] 🟡 Phase 3 (1e-5)
[GCN 2] 🔴 Frozen          [GCN 2] 🟡 Phase 3 (1e-5)
[GCN 3] 🔴 Frozen          [GCN 3] 🟡 Phase 3 (1e-5)
[GCN 4] 🔴 Frozen          [GCN 4] 🟡 Phase 3 (1e-5)
[Sort]  🔴 Frozen           [Sort]  🟡 Phase 3 (1e-5)
──────────────────          ──────────────────
[Conv1D] 🔴 Frozen          [Conv1D] 🟠 Phase 2 (1e-4)
[MaxP]   🔴 Frozen          [MaxP]   🟠 Phase 2 (1e-4)
[Conv1D] 🔴 Frozen          [Conv1D] 🟠 Phase 2 (1e-4)
[Flat]   🔴 Frozen          [Flat]   🟠 Phase 2 (1e-4)
[Dense]  🔴 Frozen          [Dense]  🟠 Phase 2 (1e-4)
[Drop]   🔴 Frozen          [Drop]   🟠 Phase 2 (1e-4)
──────────────────          ──────────────────
[NEW Dense] 🟢 Trainable    [Dense]  🟢 Phase 1 (1e-3)
──────────────────          ──────────────────
LR: 1e-4                   LR: 1e-3 → 1e-4 → 1e-5
```

### draw.io construction notes:

1. Create **5 vertical columns** using narrow containers side by side
2. Each column has the same layer stack, but with different fill colors:
   - **Green fill** (`#D5E8D4`) = Trainable / Random init
   - **Red fill** (`#F8CECC`) = Frozen
   - **Yellow fill** (`#FFF2CC`) = Unfrozen in later phase (for Method 4)
   - **Orange fill** (`#FFE6CC`) = Unfrozen in middle phase (for Method 4)
3. Use a **horizontal dashed line** to separate GNN block from Readout block within each column
4. Add a **lock icon** (🔒) on frozen layers and **unlock icon** (🔓) on trainable layers — these can be small overlay shapes
5. At the bottom of each column, add a **text box** with the learning rate
6. For Method 3, show the **new Dense layer** as a **distinctly shaped rectangle** (perhaps with a dashed border or a "NEW" badge) to make it visually clear it replaces the original
7. For Method 4, use a **color gradient** from Phase 1 (green/warm) to Phase 3 (yellow/cool) to show the progression
8. Add a **title row** at the top with method names
9. Add a **legend** at the bottom explaining the color coding

### Alternative layout — Horizontal with icons:

Instead of 5 full-stack columns, consider a **table layout**:

```
             │ GNN Block │ Readout Block │ Output │ LR
─────────────┼───────────┼───────────────┼────────┼──────
Baseline     │ 🟢 Random │ 🟢 Random     │ 🟢 Rand│ 1e-4
Method 1     │ 🔴 Frozen │ 🟢 Trainable  │ 🟢 Train│ 1e-4
Method 2     │ 🟢 Train  │ 🔴 Frozen     │ 🔴 Froz │ 1e-5
Method 3     │ 🔴 Frozen │ 🔴 Frozen     │ 🟢 NEW │ 1e-4
Method 4     │ 🟡 Ph.3   │ 🟠 Ph.2       │ 🟢 Ph.1│ Sched.
```

This works well as a **compact summary table** within a larger diagram.

---

## Diagram 6: Gradual Unfreezing Timeline

**Purpose:** Detailed visualization of Method 4's three-phase process.

**Layout:** Horizontal timeline (left to right = time), with vertical layer stack at each phase.

### Structure:

```
TIME AXIS (horizontal arrow at top):
──────────Phase 1──────────|──────Phase 2──────────|──────Phase 3──────────►

BELOW THE TIMELINE, THREE SNAPSHOTS OF THE MODEL:

PHASE 1:                    PHASE 2:                    PHASE 3:
┌─────────────┐            ┌─────────────┐             ┌─────────────┐
│ GNN Block   │ 🔴 FROZEN  │ GNN Block   │ 🔴 FROZEN   │ GNN Block   │ 🟢 TRAIN
│             │            │             │              │             │ LR=1e-5
├─────────────┤            ├─────────────┤             ├─────────────┤
│ Readout     │ 🔴 FROZEN  │ Readout     │ 🟢 TRAIN    │ Readout     │ 🟢 TRAIN
│ Block       │            │ Block       │ LR=1e-4     │ Block       │ LR=1e-5
├─────────────┤            ├─────────────┤             ├─────────────┤
│ Final       │ 🟢 TRAIN   │ Final       │ 🟢 TRAIN    │ Final       │ 🟢 TRAIN
│ Layers      │ LR=1e-3   │ Layers      │ LR=1e-4     │ Layers      │ LR=1e-5
└─────────────┘            └─────────────┘             └─────────────┘
10 epochs                   10 epochs                   10 epochs

BELOW: Learning Rate Graph
    LR
    │
1e-3├────────┐
    │        │
1e-4├────────┼────────┐
    │        │        │
1e-5├────────┼────────┼────────┐
    │        │        │        │
    └────────┴────────┴────────┴──► Time
     Phase1   Phase2   Phase3
```

### draw.io construction notes:

1. Use a **horizontal arrow** at the top as the time axis — label "Training Progress →"
2. Place **three model diagrams** (simplified stacks of 3 blocks) at equal intervals along the timeline
3. Use **transition arrows** between the three snapshots with labels "Unfreeze Readout" and "Unfreeze GNN"
4. Below the timeline, draw a **step-function graph** showing the learning rate dropping from 1e-3 → 1e-4 → 1e-5. Use the built-in draw.io **line/polyline** tool
5. Color the learning rate graph line in **blue** with labeled horizontal segments
6. Add **shading** to the learning rate graph matching the frozen/trainable colors above
7. Include a **callout annotation** (cloud shape or callout bubble): "Key insight: Deeper layers are adjusted more carefully (lower LR) because they contain more fundamental, generalizable representations"

---

## Diagram 7: Evaluation Pipeline

**Purpose:** Show how trained models are evaluated with statistical testing.

**Layout:** Top-down flow with a loop and statistical analysis branch.

### Structure:

```
TOP: Inputs
  [Multiple Cylinders in a row:]
  [Baseline ×10]  [M1 ×10]  [M2 ×10]  [M3 ×10]  [M4 ×10]
  Annotation: "50 trained models (5 groups × 10 folds)"
        │
        ▼
  [Cylinder: CV Splits .pkl]
  Annotation: "Same 10-fold splits used during training"
        │
        ▼

LOOP (show as a rounded container with a circular arrow icon):
  Label: "For each fold k = 1, 2, ..., 10"
  
  Inside the loop:
    [Rounded Rect: Load model k from each group (5 models)]
            │
            ▼
    [Rounded Rect: Load test set for fold k]
            │
            ▼
    [Rounded Rect: Generate predictions → P(toxic)]
            │
            ▼
    [Rounded Rect: Compute 6 metrics]
    Inside: ROC-AUC, GM, Precision, Recall, F1, MCC
    Annotation: "threshold = 0.5 for binary conversion"
            │
            ▼
    [Rect: Store results[fold_k][group] = metrics dict]

END OF LOOP
        │
        ▼
[Rounded Rect: Aggregate — Mean ± Std per metric per model]
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
[Rect: Friedman Test]              [Rect: Visualization]
Label: "Non-parametric              Label: "Box-and-Whisker
 test for K related                  plots per metric,
 samples"                            Radar plots"
        │                                  │
        ▼                                  ▼
[Diamond: p < 0.05?]              [Parallelogram: Plots .png]
        │ Yes
        ▼
[Rect: Nemenyi Post-Hoc Test]
Label: "Pairwise comparison
 of all model pairs"
        │
        ▼
[Parallelogram: Significance Matrix]
Label: "Which models are
 statistically different?"
```

### draw.io construction notes:

1. The **50 models** at the top can be shown as **5 small stacks of cylinders** (10 each), with color-coded labels matching the method colors from the style guide
2. The **loop** should be a rounded-corner container with a **circular arrow overlay** or a recycle icon in the top corner
3. The **Friedman test decision diamond** is critical — show the "Yes" path continuing to Nemenyi, and the "No" path ending with "No significant difference"
4. For the **visualization outputs**, consider placing small **thumbnail-style placeholder boxes** suggesting box plots and radar plots
5. Add a **small table** at the bottom showing example output format:

```
         | ROC-AUC      | GM           | ...
Baseline | 0.82 ± 0.03  | 0.75 ± 0.04 | ...
Method1  | 0.85 ± 0.02  | 0.78 ± 0.03 | ...
...
```

---

## Diagram 8: Bidirectional Transfer & Model Configurations

**Purpose:** Show the full experimental matrix — 2 directions × 3 configurations × 4 methods.

**Layout:** Matrix/grid layout.

### Structure:

```
LEFT COLUMN: Transfer Directions
  
  [Rounded Rect (Blue): Peptide Dataset]
         │
         │  Arrow labeled "Source"     Arrow labeled "Target"
         │         │                        ▲
         ▼         ▼                        │
  [Large Rect: DeepGraphCNN]  ──────►  [Large Rect: DeepGraphCNN]
  "Overtrained on Peptide"              "Fine-tuned on SMT"
         
         ═══════════ AND ═══════════
         
  [Rounded Rect (Green): SMT Dataset]
         │
         │  Arrow labeled "Source"     Arrow labeled "Target"
         │         │                        ▲
         ▼         ▼                        │
  [Large Rect: DeepGraphCNN]  ──────►  [Large Rect: DeepGraphCNN]
  "Overtrained on SMT"                 "Fine-tuned on Peptide"


RIGHT SIDE: Configurations Matrix

              │ [25,25,25,1] │ [125,125,125,1] │ [512,256,128,1]
──────────────┼──────────────┼─────────────────┼────────────────
Baseline      │      ✓       │        ✓        │       ✓
Method 1      │      ✓       │        ✓        │       ✓
Method 2      │      ✓       │        ✓        │       ✓
Method 3      │      ✓       │        ✓        │       ✓
Method 4      │      ✓       │        ✓        │       ✓
──────────────┼──────────────┼─────────────────┼────────────────
Total models  │  5×10 = 50   │    5×10 = 50    │   5×10 = 50
per direction │              │                 │
```

### draw.io construction notes:

1. Use **two large horizontal arrows** (one Peptide→SMT, one SMT→Peptide) as the central visual metaphor
2. Place the **source dataset icon** on the left and **target dataset icon** on the right, with the model in between
3. Show the arrow passing **through the model** to symbolize knowledge transfer
4. For the configurations matrix, use draw.io's **table shape** (Insert → Table)
5. Color the three configuration columns in progressively darker shades to indicate increasing model size
6. Add a **total count summary**: "Grand total: 2 directions × 3 configurations × 5 models × 10 folds = **300 trained models**"

---

## Recommended Diagram Creation Order

1. **Start with Diagram 1** (overview) — this orients the viewer
2. **Then Diagram 3** (architecture) — establishes what the model looks like
3. **Then Diagram 5** (methods comparison) — the heart of the project
4. **Then Diagrams 2, 4, 6, 7** (detail views) as needed
5. **Diagram 8** (matrix) last — summarizes the full experimental scope

## Tips for draw.io

- Use **File → Page Setup** to set a large canvas (A3 or larger) for Diagrams 1 and 5
- Use **Layers** (View → Layers) to separate the background containers from the foreground shapes
- **Group related shapes** (Ctrl+G) to move them together
- Export as **PNG at 300 DPI** for thesis inclusion, or as **SVG** for web
- Use **Edit → Find/Replace** to quickly fix labels across the diagram
- The **Container** shapes (Insert → Advanced) are excellent for the stage groupings
- For lock/unlock icons: search "lock" in the draw.io search panel (left sidebar) to find built-in shapes
