# Concise Project Pipeline Diagram — Draw.io Guide

> **Goal:** One clean, horizontal, left-to-right diagram that any person can glance at and immediately understand what this project does. Inspired by the reference image style — simple icons, short labels, minimal clutter.

---

## Design Principles

- **Horizontal flow**, left → right, one single row of major stages
- **Maximum 5–6 columns** (one per stage)
- **Small icons** with short 2–4 word labels underneath — no paragraphs inside shapes
- **Thin vertical separator lines** between stages with a stage title at the top of each column
- **Light muted colors** — no heavy fills, mostly white shapes with subtle colored accents
- **One optional secondary row** at the bottom (for the transfer learning detail or bidirectional info)
- **Total size:** fits comfortably on a single A4/letter landscape page

---

## The 5 Columns (Left to Right)

```
  Datasets        Preprocessing       Source Models       Target Training        Evaluation
     │                  │                   │                    │                    │
     ▼                  ▼                   ▼                    ▼                    ▼
 ┌────────┐       ┌──────────┐        ┌──────────┐        ┌──────────┐         ┌──────────┐
 │  🗄️🗄️  │       │  ⬡→📊   │        │  🧠🧠   │        │ 🧠×5     │         │  📋      │
 │        │──────►│          │───────►│          │───────►│          │────────►│          │
 │ 2 XLSX │       │ SMILES   │        │ Overtrain│        │ Baseline │         │ 6 Metrics│
 │ files  │       │ to Graph │        │ on 100%  │        │ + 4 TL   │         │ + Stats  │
 └────────┘       │          │        │          │        │ methods  │         └──────────┘
                 │ Node +   │        └──────────┘        │ (10-fold │
                 │ Edge     │                             │   CV)    │
                 │ Features │                             └──────────┘
                 │ (72 vocab)│
                 └──────────┘
```

---

## Column-by-Column Specification

### Column 1: Datasets

**What the viewer should understand:** "The project starts with two Excel files containing molecule data."

| Property | Value |
|----------|-------|
| **Shapes** | 2 × **Cylinder** (database icon), stacked vertically |
| **Top cylinder label** | "Peptide dataset" |
| **Bottom cylinder label** | "Small molecule dataset" |
| **Small shared annotation** | A tiny text below both: "SMILES + Activity (0/1)" |
| **Fill** | White with light blue border (`#DAE8FC`) |
| **Size per cylinder** | ~60×40px |

**Column title above:** "Datasets" (bold, 12pt)

---

### Column 2: Preprocessing

**What the viewer should understand:** "Molecules are converted from text (SMILES) into graphs with node and edge features."

| Property | Value |
|----------|-------|
| **Shape** | 1 × **Rounded rectangle** with a small molecule-to-graph icon inside |
| **Icon idea** | A tiny hexagon (molecule) with an arrow pointing to a tiny node-edge sketch (graph) — or simply use draw.io's "hierarchy" / "network" icon |
| **Label** | "SMILES → Graph" |
| **Small annotation below** | "RDKit + StellarGraph" |
| **Second small element** | A tiny rounded rect below: "One-hot encoding (72-element vocab)" |
| **Third small element** | A tiny rounded rect below: "Node features: element (72) + atomic props (5)" |
| **Fourth small element** | A tiny rounded rect below: "Edge features: bond type (4) + bond props (3)" |
| **Fill** | White with light green border (`#D5E8D4`) |
| **Size** | ~100×80px main shape (taller to accommodate multiple annotations) |

**Column title above:** "Preprocessing" (bold, 12pt)

---

### Column 3: Source Models

**What the viewer should understand:** "A model is trained on all the data from each dataset to create a knowledge-packed starting point."

| Property | Value |
|----------|-------|
| **Shape** | 2 × **Rectangle with thick border** (model icon), stacked vertically |
| **Top model label** | "Peptide model" |
| **Bottom model label** | "Small mol. model" |
| **Shared annotation** | "Trained on 100% data (90/10 train-val)" |
| **Fill** | White with light orange border (`#FFE6CC`) |
| **Size per model** | ~70×35px |
| **Optional icon** | A small brain or neural-net icon (search "neural" in draw.io) |

**Column title above:** "Source Models" (bold, 12pt)

---

### Column 4: Target Training

**What the viewer should understand:** "We train 5 model variants — 1 baseline from scratch + 4 using transferred weights. Each model is trained using 10-fold cross-validation for fair comparison."

| Property | Value |
|----------|-------|
| **Shapes** | 5 small **rectangles** in a vertical stack, inside a light container |
| **Labels (top to bottom)** | "Baseline (from scratch)" / "M1: Freeze GNN" / "M2: Freeze Readout" / "M3: Freeze All + New" / "M4: Gradual Unfreeze" |
| **Baseline fill** | White with blue border |
| **M1–M4 fill** | White with orange border |
| **Incoming dashed orange arrow** | From Column 3 to M1–M4 only (NOT to Baseline) — labeled "Transfer weights" |
| **Size per method rect** | ~120×20px |
| **Container** | Light gray dashed box around all 5 |
| **Small annotation below container** | "10-fold stratified CV (shared splits)" — italic, gray, 9pt |

**Column title above:** "Target Training" (bold, 12pt)

**Key visual detail:** The **baseline** has NO incoming arrow from Source Models — only M1–M4 do. This visually communicates that baseline starts from scratch while TL methods receive pretrained weights. The annotation below indicates that all models use the same 10-fold cross-validation splits for fair comparison.

---

### Column 5: Evaluation

**What the viewer should understand:** "Models are compared using metrics and statistical tests."

| Property | Value |
|----------|-------|
| **Shape** | 1 × **Rectangle** or **list/document** shape |
| **Label — list of items** | "ROC-AUC" / "G-Mean" / "Precision" / "Recall" / "F1" / "MCC" |
| **Second shape below** | Small rectangle: "Friedman + Nemenyi tests" |
| **Third shape (optional)** | Tiny icons suggesting box plot + radar plot |
| **Fill** | White with light purple border (`#E1D5E7`) |
| **Size** | ~90×80px |

**Column title above:** "Evaluation" (bold, 12pt)

---

## Secondary Row (Bottom): Bidirectional Transfer

Below the main row, add a **thin horizontal strip** showing that the whole pipeline runs in two directions:

```
───────────────────────────────────────────────────────────────────────────
                        Bidirectional Transfer
                              
  Peptide ──► Small Molecule        Small Molecule ──► Peptide
  (source)    (target)              (source)           (target)

  × 3 model size configs: [25,25,25,1] │ [125,125,125,1] │ [512,256,128,1]
───────────────────────────────────────────────────────────────────────────
```

| Property | Value |
|----------|-------|
| **Shape** | A wide **rounded rectangle** spanning the full width |
| **Inside** | Two horizontal arrows: one left→right labeled "Peptide → Small Mol.", one right→left labeled "Small Mol. → Peptide" |
| **Below arrows** | Text: "× 3 model size configurations" |
| **Fill** | Very light gray (`#FAFAFA`), thin border |
| **Height** | ~60px, same width as the 5 columns together |

---

## Complete Layout (ASCII Preview)

This is exactly what the final draw.io diagram should look like:

```
  Datasets       Preprocessing     Source Models     Target Training      Evaluation
     │                │                  │                  │                 │
     ▼                ▼                  ▼                  ▼                 ▼
 ┌────────┐     ┌───────────┐     ┌───────────┐     ┌──────────────┐     ┌──────────┐
 │        │     │           │     │           │     │              │     │ ROC-AUC  │
 │ 🗄️     │     │ SMILES    │     │ Peptide   │     │ Baseline     │     │ G-Mean   │
 │Peptide │     │   →       │     │ source    │     │ ─────────── │     │ Precis.  │
 │ .xlsx  │────►│ Graph     │────►│ model     │  ┌─►│ M1: Fr. GNN │────►│ Recall   │
 │        │     │           │     │           │  │  │ M2: Fr. Read│     │ F1       │
 │ 🗄️     │     │ (RDKit +  │     │ Small mol │  │  │ M3: Fr. All │     │ MCC      │
 │SmallMol│     │  Stellar  │     │ source    │──┘  │ M4: Gradual │     │──────────│
 │ .xlsx  │     │  Graph)   │     │ model     │     │              │     │ Friedman │
 │        │     │           │     │           │     │              │     │ Nemenyi  │
 └────────┘     │ Node+Edge │     │(100% data │     │ 10-fold CV   │     └──────────┘
                │ Features  │     │ 90/10)    │     │ (shared)     │           ▲
                │ (72 vocab)│     └───────────┘     └──────────────┘           │
                └───────────┘
                                                          ▲                    │
                                                    dashed orange          Results
                                                    arrow = weight         aggregation
                                                    transfer (M1-M4
                                                    only, NOT baseline)

 ┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                              Bidirectional Transfer                                                │
 │     Peptide ──────────────────► Small Molecule       Small Molecule ──────────────────► Peptide    │
 │     (source)                    (target)             (source)                           (target)   │
 │                          × 3 model size configurations                                             │
 └────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Exact draw.io Build Steps (Quick Version)

1. **Canvas:** File → Page Setup → **A4 Landscape** or **Letter Landscape**

2. **Draw 5 thin vertical dashed lines** spaced equally to create 5 columns. Add column titles at the very top in **bold 12pt**.

3. **Column 1 — Datasets:**
   - Place 2 cylinders (General → Cylinder) stacked vertically, light blue border
   - Label them "Peptide .xlsx" and "Small Mol. .xlsx"
   - Tiny text below: "SMILES + Activity (0/1)"

4. **Column 2 — Preprocessing:**
   - One rounded rectangle, light green border
   - Label: "SMILES → Graph"
   - Tiny text: "RDKit + StellarGraph"
   - Below: small rounded rect "One-hot encoding (72-element vocab)"
   - Below: small rounded rect "Node features: element (72) + atomic props (5)"
   - Below: small rounded rect "Edge features: bond type (4) + bond props (3)"

5. **Column 3 — Source Models:**
   - 2 rectangles with thick border, light orange border
   - Labels: "Peptide model", "Small mol. model"
   - Tiny text: "Trained on 100% data"

6. **Column 4 — Target Training:**
   - Light gray dashed container
   - Inside: 5 small rectangles stacked
   - Top one (Baseline): blue border, label "Baseline"
   - Bottom four (M1–M4): orange border, short method names
   - **Dashed orange arrow** from Column 3 entering ONLY the M1–M4 group (skipping Baseline)
   - **Small annotation below container** (italic, gray, 9pt): "10-fold stratified CV (shared splits)"

7. **Column 5 — Evaluation:**
   - Rectangle listing the 6 metrics
   - Small rectangle below: "Friedman + Nemenyi"

9. **Connectors between columns:**
   - Simple solid arrows (1.5pt, black) from column to column
   - The special **dashed orange arrow** from Column 3 → Column 4's M1-M4 group

10. **Bottom row — Bidirectional:**
    - Wide rounded rectangle spanning full width
    - Two arrows inside showing both transfer directions
    - Small text: "× 3 model size configurations"

11. **Legend (tiny, bottom-right):**
    - 🗄️ Cylinder = dataset
    - ▭ Rectangle = model/process
    - Dashed orange arrow = weight transfer
    - Solid arrow = pipeline flow

---

## What This Achieves

An outsider looking at this diagram will immediately understand:

1. **"There are two datasets"** (two cylinders on the left)
2. **"Molecules get converted into graphs with rich features"** (preprocessing column — shows node features with 72-element vocabulary + atomic properties, and edge features with bond types + bond properties)
3. **"A model is first trained on all data"** (source models)
4. **"Then 5 different approaches are tried, each using 10-fold cross-validation"** (target training — baseline + 4 methods, with CV annotation)
5. **"The orange arrow means some methods reuse knowledge from step 3"** (transfer learning)
6. **"Results are measured and compared"** (evaluation metrics)
7. **"This is done in both directions"** (bottom row)

No deep knowledge of machine learning or chemistry is required to get this overview.
