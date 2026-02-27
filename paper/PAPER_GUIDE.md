# PAPER_GUIDE.md — Master Reference for Writing the Scientific Paper

> **Purpose:** This file is a comprehensive reference for an LLM agent (or human) assisting with writing the scientific paper on cross-domain transfer learning for molecular toxicity prediction. It documents what exists, what needs to be done, project conventions, writing style guidance, and maps every asset to its role in the manuscript.

---

## 1. PROJECT IDENTITY

- **Topic:** Cross-domain transfer learning for molecular toxicity classification using graph neural networks
- **Official thesis task:** "Klasifikacija toksičnosti malih molekula i peptida" / "Toxicity Classification for Small Molecules and Peptides" (see `paper/official-task/`)
- **Author:** Jakov Cvetko (solo author)
- **Affiliation:** Faculty of Engineering (RITEH), University of Rijeka, Rijeka, Croatia
- **Study programme:** University graduate study of Computing, Software Engineering module
- **Mentor:** prof. Goran Mauša | **Co-mentor:** dr. sc. Erik Otović
- **Target journal:** Nature Machine Intelligence
- **Paper type:** Scientific article (Springer Nature template — `sn-jnl.cls`, Nature Portfolio reference style `sn-nature.bst`)
- **Page limit:** **8 pages maximum** — every sentence must be concise, informative, and earn its place. No filler, no redundancy. Prefer short paragraphs with high information density.
- **Core question:** Can toxicity knowledge learned in one molecular domain (small molecules or peptides) transfer to another domain via GNN-based transfer learning, and which strategy works best?
- **Additional angle:** A classical ML baseline (DPC + SVM, replicating the ToxinPred paper) is included for comparison on the peptide domain

---

## 2. FOLDER MAP — WHAT LIVES WHERE

### 2.1 Paper-writing assets (`paper/`)

| Path | Contents | Role |
|------|----------|------|
| `paper/Jakov_Cvetko___Transfer_Learning/` | Overleaf Springer Nature template files (`sn-article.tex`, `sn-jnl.cls`, `sn-bibliography.bib`, BST styles, sample eps/pdf) + **`paper_draft.tex`** (working draft with Introduction written, Nature Machine Intelligence format) | **THE manuscript** — `paper_draft.tex` is the active writing file; `sn-article.tex` is the original template reference |
| `paper/introduction/` | 4 markdown drafts (`proces-version-a.md` through `d.md`) documenting the project pipeline end-to-end | Reference material for writing Introduction and Methods sections |
| `paper/comparison/` | `toxinPred-v1.pdf` — the ToxinPred scientific paper by Gupta et al. (2013) | **Writing style reference** — demonstrates scientific, concise, informative but not overly detailed writing |
| `paper/example-works/` | **⚠️ CRITICAL STYLE REFERENCE** — Two published Nature Machine Intelligence papers (see Section 2.1.1 below) | **PRIMARY writing-style and structure reference** — these papers define how every section should be written, structured, and formatted |
| `paper/images/` | Pre-collected images and data for the paper: `project-diagram.png`, `side-by-side-models-boxes.png`, `radar_plots_peptide_median.png`, `radar_plots_smt_median.png`, `eval-book-peptide-results.xlsx`, `eval-book-smt-results.xlsx`, `dataset_analysis_results.md` | **Ready-to-use paper assets** — curated subset of project outputs gathered in one place for convenience |
| `paper/questions/` | `questions-to-be-asnwerd-in-paper.txt` — the 6 formal research questions (in Croatian) that the paper must answer | **Defines the paper's scope** — every section should contribute to answering these questions (see Section 3.1) |
| `paper/chapters/` | `chapters_plan.txt` — detailed section-by-section plan for Nature Machine Intelligence format, figure/table plans, writing rules | **Blueprint** for the paper structure, compared against ToxinPred format |
| `paper/official-task/` | `task-zavrsni-Jakov-Cvetko.pdf` — official diploma thesis task assignment from the University of Rijeka (dated 14.03.2025) | **Scope anchor** — defines the formal task: investigate toxicity of small molecules and peptides, optimise a neural network architecture, analyse cross-domain transfer learning, evaluate and compare with literature. Mentor: prof. Goran Mauša, Co-mentor: dr. sc. Erik Otović |
| `paper/PAPER_GUIDE.md` | This file | Master context file for LLM-assisted paper writing |

#### 2.1.1 Example papers from Nature Machine Intelligence (`paper/example-works/`) — ⚠️ MUST READ BEFORE WRITING

This folder contains **two published Nature Machine Intelligence papers** that serve as the **definitive style, structure, and tone reference** for writing this manuscript. Every section of `paper_draft.tex` should be written as if these papers are the template.

| File | Citation | DOI | Pages | Relevance |
|------|----------|-----|-------|-----------|
| `Reshaping the discovery of self-assembling.pdf` | Njirjak, Žužić, Babić, Janković, **Otović**, Kalafatovic & **Mauša** (2024) *Nat. Mach. Intell.* **6**, 1487–1500 | `10.1038/s42256-024-00928-1` | 19 | **HIGHEST PRIORITY** — co-authored by the thesis mentor (Mauša) and co-mentor (Otović). Defines the expected structure and writing style |
| `Tailored structured peptide design with a.pdf` | Leyva, Torres, Oliva, de la Fuente-Nunez & Brizuela (2025) *Nat. Mach. Intell.* **7**, 1685–1697 | `10.1038/s42256-025-01119-2` | 16 | Confirms the same Nature MI article format; additional style reference |

**Observed section structure from these papers:**

| Paper | Main sections | Notes |
|-------|--------------|-------|
| Njirjak et al. (2024) | Introduction → **Results and discussion** → Conclusion → Methods | Results and Discussion **combined** into one section; Conclusion present; Methods at end |
| Leyva et al. (2025) | Introduction → Results → Discussion → Methods | Separate Results and Discussion; no Conclusion; Methods at end |

**Key structural patterns to follow:**
- **Introduction** has no explicit `\section` heading in Nature MI — it flows directly after the abstract. However, in the Springer Nature template (`sn-jnl.cls`) it is fine to use `\section{Introduction}`.
- **NO numbered subsections** (no `\subsection{1.1 ...}`). Use short **descriptive subheadings** within sections instead.
- **Methods placed at the END**, after Discussion/Conclusion — this is mandatory for Nature format.
- **Very lean top-level structure** — only 3–4 major `\section` headings in the entire paper.
- **Figures and tables are interspersed** with the text where they are first referenced.
- **Voice:** Both papers use "we" (multi-author). Since this paper has a solo author, use passive voice or "this study" / "the present work" instead.

**For this paper (8-page limit), adopt the Njirjak et al. (2024) structure:**
1. Introduction
2. Results and discussion (combined)
3. Conclusion
4. Methods

### 2.2 Key project-level documentation files

| File | Purpose |
|------|---------|
| `tables-final.md` | Template tables for the paper (Croatian labels), with layout for 8+ tables covering datasets, architecture, methods, results, statistical tests |
| `tables-detailed.md` | Extended version — 13 detailed table templates including compact alternatives, hyperparameters, Nemenyi matrices |
| `table-and-plots-general.md` | Concise 3-table + 4-figure essential layout for a short paper |
| `tabkes-and-plots.md` | Comprehensive 10-table + 18-figure catalog with generation priority and existing plot locations |
| `eval-book-peptide-reference.md` | Empty result grid for peptide target (3 model sizes × 5 methods × 6 metrics + Friedman) |
| `eval-book-smt-reference.md` | Same grid for SMT target |
| `eval-book-peptide-results.xlsx` | Actual populated results (peptide target, mean aggregation) |
| `eval-book-peptide-results-median.xlsx` | Same with median aggregation |
| `eval-book-smt-results.xlsx` | Actual populated results (SMT target, mean aggregation) |
| `eval-book-smt-results-median.xlsx` | Same with median aggregation |

### 2.3 Datasets (`datasets/`)

| File | Domain | Toxic | Non-toxic | Unique Elements | Avg Graph Size |
|------|--------|-------|-----------|-----------------|----------------|
| `MolToxPredDataset.xlsx` | Small molecules | 4,616 | 5,833 | 72 | 21.1 atoms |
| `ToxinSequenceSMILES.xlsx` | Peptides | 1,805 | 3,593 | 4 | 164.3 atoms |

- `dataset_analysis_results.md` and `.xlsx` contain the summary statistics above
- Analysis scripts: `dataset_analysis.py`, `analyze_*_elements.py`, `find_all_elements.py`

### 2.4 Model architecture documentation (`models_architecture/`) — **PRIMARY TECHNICAL REFERENCE**

This folder is the **single most important technical reference** for writing the Methods section. It contains layer-by-layer architecture details, exact frozen/trainable layer specifications, code snippets, and ASCII architecture diagrams for every model variant.

| File | Content | Use when writing... |
|------|---------|---------------------|
| `README.md` | Architecture overview with ASCII diagrams and TL method comparison table | Quick reference for all methods at a glance |
| `overtrained_model.md` | Source model: full layer diagram, training params (90/10 split, patience=7), purpose as knowledge acquisition | Methods → Source pretraining subsection |
| `baseline_model.md` | Baseline: identical architecture, random init, 10-fold CV, serves as no-transfer benchmark | Methods → Baseline description |
| `method1_freeze_gnn.md` | M1: GNN layers 🔒, readout 🔓, LR=1e-4, rationale (GNN learns domain-agnostic structure) | Methods → Transfer strategies |
| `method2_freeze_readout.md` | M2: Readout 🔒, GNN 🔓, LR=1e-5 (lower to prevent catastrophic forgetting), rationale | Methods → Transfer strategies |
| `method3_freeze_all.md` | M3: All 🔒 + new Dense(1) output 🔓, LR=1e-4, ~129 trainable params, most conservative | Methods → Transfer strategies |
| `method4_gradual_unfreezing.md` | M4: 3-phase unfreezing (final→readout→GNN), LR 1e-3→1e-4→1e-5, 10 epochs/phase | Methods → Transfer strategies |
| `project_process_documentation.md` | **761-line comprehensive pipeline narrative**: preprocessing details, shared atom vocabulary (72-element one-hot), StellarGraph construction, full training flow, evaluation. **Note:** this file contains a typo stating 27 elements — the correct number is **72** | Methods section (all subsections) — most detailed single reference |
| `complete_pipeline_and_drawio_blueprint.md` | Draw.io diagram mapping guide (1024 lines) | Figures preparation |
| `concise_diagram_guide.md` | Concise 5-column pipeline diagram specification with exact draw.io build steps | Figures preparation |
| `drawio_diagram_guide.md` | Extended diagram guide | Figures preparation |

#### Key architectural details from these docs (for quick paper-writing reference)

**Full layer stack (all models share this, only GNN layer sizes differ):**
```
Input (Graph features + adjacency) → GraphConv1 (tanh) → GraphConv2 (tanh) → 
GraphConv3 (tanh) → GraphConv4 (tanh) → SortPooling (k=25) → 
Conv1D (16 filters, kernel=sum(layer_sizes)) → MaxPool1D (2) → 
Conv1D (32 filters, kernel=5, stride=1) → Flatten → 
Dense (128, ReLU) → Dropout (0.2) → Dense (1, Sigmoid) → Output
```

**Shared atom vocabulary** (from `project_process_documentation.md`):
The shared one-hot atom vocabulary used across all experiments contains **72 elements**, creating a 72-dimensional node feature vector. This shared vocabulary is what enables cross-domain weight transfer — both peptide and small molecule graphs are encoded into the same 72-dimensional feature space. **Note:** Some older project docs (e.g. `project_process_documentation.md`) incorrectly list 27 elements — this is a typo; the correct number is **72**.

### 2.5 Visual assets (`draw_io/`)

| File | Content |
|------|---------|
| `project diagram/concise_project_representation.drawio` | High-level pipeline diagram |
| `project diagram/project-diagram.png` | Exported PNG of project pipeline |
| `arhitecture diagram/models_grouped_boxes.drawio` | Architecture diagram with grouped model boxes |
| `arhitecture diagram/side-by-side-models-boxes.png` | Exported PNG showing all models side-by-side |

### 2.6 Experiment folders (code + artifacts)

#### Model size families (GNN layer configurations)

| Label | GNN Layers | Folders prefix |
|-------|-----------|----------------|
| Standard | `[25, 25, 25, 1]` | `overtrained_models/`, `baseline_*`, `transfer_learning_*`, `eval_*` (no prefix) + `extra_features_*` |
| Large Layers | `[125, 125, 125, 1]` | `large_layers_*` |
| Inflated | `[512, 256, 128, 1]` | `inflated_*` |
| Extra Inflated | `[1024, 512, 256, 1]` | `extra_inflated_*` |

#### Experiment directories pattern (per model size)

| Folder pattern | Phase | Content |
|---------------|-------|---------|
| `*_models/` | A — Source overtraining | Pretrained source `.h5` models + processed `.pkl` graph data |
| `cv_splits/` or `*_cv_splits/` | B — Split generation | Stratified 10-fold split definitions (`.pkl`) |
| `*_baseline_peptide/` | C — Peptide baseline | 10 fold models trained from scratch on peptide data |
| `*_baseline_small_mol_tox/` | C — SMT baseline | 10 fold models trained from scratch on SMT data |
| `*_transfer_learning_smt_to_p/` | D — TL (SMT→Peptide) | 4 methods × 10 folds of transfer-learned models |
| `*_transfer_learning_p_to_smt/` | D — TL (Peptide→SMT) | 4 methods × 10 folds of transfer-learned models |
| `*_eval_peptide/` | E — Evaluation | Evaluation outputs: plots (boxplots, heatmaps, ROC, PR curves, confusion matrices, radar charts, violin plots, calibration curves), summary xlsx |
| `*_eval_small_mol_tox/` or `*_eval_small/` | E — Evaluation | Same for SMT target |

### 2.7 SVM baseline (`svm/`)

A DPC (Dipeptide Composition) + SVM model replicating the ToxinPred paper (Gupta et al. 2013) methodology:

| File | Role |
|------|------|
| `DPC_SVM_dokumentacija.md` | Full documentation: original paper method, our implementation, differences |
| `ToxinPred_DPC_SVM_analiza_rada.md` | Detailed analysis of the ToxinPred paper results and methods |
| `DPC_formulas.md` | Mathematical formulas for DPC computation |
| `dpc_svm_paper_model.py` | The clean paper-ready SVM model script |
| `dpc_svm_model_balanced_clean.py` | Balanced variant of SVM model |
| `create_aligned_splits.py` | Creates CV splits aligned with GNN splits for fair comparison |
| `DPC_SVM_results*.xlsx` | Various result files from different SVM configurations |
| `models*/` | Saved SVM fold models |
| `toxinPred-v1.pdf` | Copy of comparison paper |

Key SVM implementation details:
- **Features:** 400-dimensional DPC vector (20×20 amino acid dipeptides, normalized)
- **Model:** scikit-learn SVC with RBF kernel, C=5, gamma=0.001, class_weight="balanced"
- **Evaluation:** 10-fold CV using same splits as GNN models (aligned for fair comparison)
- **Differences from original paper:** 10-fold (vs 5-fold), scikit-learn (vs SVMlight), balanced class weights, StandardScaler per fold

### 2.8 Utility scripts (root level)

| Script | Purpose |
|--------|---------|
| `complete_thesis.py` | Likely orchestrates the full pipeline |
| `populate_peptide_excel.py` / `populate_smt_excel.py` | Fill in eval-book xlsx files with results (mean aggregation) |
| `populate_peptide_excel_median.py` / `populate_smt_excel_median.py` | Same with median aggregation |
| `radar_plots_peptide.py` / `radar_plots_smt.py` | Generate radar comparison plots |
| `radar_plots_*_median.py` | Median-based radar plots |
| `thesis_plots.py` | Generate thesis-level plots (t-SNE, etc.) |

### 2.9 Environment (`provision/`, `venv/`)

- Python 3.8 (Windows)
- TensorFlow 2.11.0
- StellarGraph 1.2.1
- RDKit 2022.9.5
- scikit-learn (for SVM baseline)

---

## 3. THE SCIENTIFIC STORY

### 3.1 Research questions

The paper must answer the following 6 formal research questions (from `paper/questions/questions-to-be-asnwerd-in-paper.txt`):

| # | Question (original Croatian) | English translation | Paper section(s) |
|---|------------------------------|---------------------|-------------------|
| **RQ1** | Koliko učinkovito možemo primijeniti metode TL između dviju različitih domena toksikoloških podataka koristeći DGCNN arhitekturu? | **How effectively can we apply TL methods between two different toxicological data domains (peptides and small molecules) using the DGCNN architecture?** | Results, Discussion |
| **RQ2** | Koje od četiri istražene metode TL pokazuju najbolje performanse? | **Which of the four investigated TL methods shows the best performance when transferring a model pretrained on one domain to another?** | Results (main tables) |
| **RQ3** | Postoji li statistički značajna razlika u performansama modela koji koriste TL u odnosu na baseline? | **Is there a statistically significant difference in performance between TL models and baseline models trained exclusively within the same domain?** | Results (Friedman + Nemenyi), Discussion |
| **RQ4** | Utječe li smjer TL na učinkovitost i rezultate modela? | **Does the direction of transfer learning (peptide→SMT vs SMT→peptide) affect the effectiveness and results?** | Results (both directions), Discussion |
| **RQ5** | Koje su moguće implikacije korištenja TL pristupa u području predviđanja toksičnosti molekula? | **What are the possible implications of using TL approaches in the field of molecular toxicity prediction?** | Discussion, Conclusion |
| **RQ6** | Usporedba dobivenih modela sa modelima iz znanstvenih radova temeljenih na istim domenama? | **How do the obtained models compare with models from scientific papers based on the same domains?** | Discussion (SVM/ToxinPred comparison) |

These questions shape the entire paper narrative. Every section should explicitly contribute to answering one or more of them.

### 3.2 Central framing (one paragraph)

> This study investigates cross-domain transfer learning for molecular toxicity classification using graph neural networks. Molecules from two domains — small molecules and peptides — are represented as graphs with atom-level and bond-level features. A DeepGraphCNN model is pretrained on a source domain, then transferred to a target domain using four distinct strategies: freezing the graph encoder, freezing the readout/classifier, freezing all layers with a new output head, and gradual unfreezing. Transfer experiments are conducted bidirectionally across four model capacities. All methods are evaluated against from-scratch baselines under identical stratified 10-fold cross-validation splits, with Friedman and Nemenyi post-hoc tests for statistical rigor. Additionally, a DPC+SVM baseline replicating the ToxinPred methodology is included for the peptide domain.

### 3.3 What makes this study novel

- **Bidirectional cross-domain transfer** between chemically distinct molecular families (peptides vs small molecules)
- **Systematic comparison** of 4 transfer strategies × 4 model capacities × 2 directions
- **Shared graph representation** bridging two domains via a common atom vocabulary
- **Rigorous evaluation** with matched fold splits, 6 metrics, and non-parametric statistical tests
- **Classical ML comparison** (DPC+SVM) providing a reference point

---

## 4. MANUSCRIPT STRUCTURE AND SECTION MAPPING

The working draft is `paper/Jakov_Cvetko___Transfer_Learning/paper_draft.tex` using Nature Portfolio format (`sn-nature.bst`). The original template `sn-article.tex` is kept as reference.

### Structure (Nature Machine Intelligence format — modelled after Njirjak et al. 2024)

Based on the two example papers in `paper/example-works/`, the paper uses a **lean 4-section structure** with descriptive (not numbered) subheadings. No `\subsection{}` commands — use `\paragraph{}` or bold text for sub-topics within a section.

| Section | LaTeX label | Content | Status |
|---------|------------|---------|--------|
| **Title** | `\title` | "Toxicity Classification for Small Molecules and Peptides" (matches official thesis task) | DONE |
| **Abstract** | `\abstract` | ~150–200 words: problem, approach, key findings, significance. No citations | TODO |
| **Keywords** | `\keywords` | transfer learning, graph neural networks, molecular toxicity, DeepGraphCNN, cross-domain classification, peptide toxicity | DONE |
| **1. Introduction** | `\section{Introduction}` | Motivation, problem statement, contribution summary (3 concise paragraphs) | DONE |
| **2. Results and discussion** | `\section{Results and discussion}` | Combined: tables, figures, interpretation, transfer direction effects, strategy behaviour, capacity influence, SVM comparison, statistical tests, answer RQ1–RQ6 | TODO |
| **3. Conclusion** | `\section{Conclusion}` | Short (1 paragraph): key takeaways, practical implications, future directions | TODO |
| **4. Methods** | `\section{Methods}` | Datasets, graph construction, architecture, TL strategies, SVM baseline, CV protocol, metrics, statistical tests (placed at end per Nature format) | TODO |
| **Backmatter** | `\bmhead` | Acknowledgements, Declarations (funding, competing interests, data/code availability, author contributions) | Scaffold in place |
| **References** | `\bibliography` | `sn-bibliography.bib` — populated with 20 BibTeX entries | DONE |

**Key rule:** Keep it to exactly **4 `\section{}` commands** in the body (Introduction, Results and discussion, Conclusion, Methods). Use descriptive subheadings within sections only when needed — prefer flowing prose. Follow the example papers in `paper/example-works/`.

---

## 5. KEY NUMBERS AND FACTS (for quick reference when writing)

### 5.1 Dataset statistics

| Property | Small Molecules (SMT) | Peptides |
|----------|-----------------------|----------|
| Source file | `MolToxPredDataset.xlsx` | `ToxinSequenceSMILES.xlsx` |
| Total samples | 10,449 | 5,398 |
| Toxic (class 1) | 4,616 (44%) | 1,805 (33%) |
| Non-toxic (class 0) | 5,833 (56%) | 3,593 (67%) |
| Unique elements | 72 | 4 |
| Avg atoms per molecule | 21.1 | 164.3 |
| Min / Max atoms | 1 / 293 | 26 / 333 |
| Representation | SMILES → molecular graph | SMILES → molecular graph |

### 5.2 Model configurations

| Size | GNN Layers | k (SortPooling) | Conv1D-1 kernel | Conv1D-2 | Dense | Node features |
|------|-----------|-----------------|-----------------|----------|-------|---------------|
| Standard | [25, 25, 25, 1] | 25 | 16 filters, kernel=76 | 32 filters, kernel=5 | 128 (ReLU) → 1 (Sigmoid) | 72 |
| Large Layers | [125, 125, 125, 1] | 25 | 16 filters, kernel=376 | 32 filters, kernel=5 | 128 (ReLU) → 1 (Sigmoid) | 72 |
| Inflated | [512, 256, 128, 1] | 25 | 16 filters, kernel=897 | 32 filters, kernel=5 | 128 (ReLU) → 1 (Sigmoid) | 72 |
| Extra Inflated | [1024, 512, 256, 1] | 25 | 16 filters, kernel=1793 | 32 filters, kernel=5 | 128 (ReLU) → 1 (Sigmoid) | 72 |

Common: GNN activations all tanh, bias=False, MaxPool1D(2) between Conv1D layers, Dropout(0.2), Adam optimizer, binary cross-entropy loss.

**Note:** The first Conv1D kernel size = `sum(layer_sizes)`. This means larger GNN models also have larger Conv1D kernels, which affects the total parameter count significantly.

### 5.3 Transfer learning methods

| Method | What's Frozen | What's Trained | Learning Rate | Phases |
|--------|--------------|----------------|---------------|--------|
| Baseline | Nothing (random init) | All layers | 1e-4 | 1 |
| M1: Freeze GNN | DeepGraphCNN + GraphConv layers | Conv1D, Dense, Dropout | 1e-4 | 1 |
| M2: Freeze Readout | Dense, Dropout, Flatten | GNN layers | 1e-5 | 1 |
| M3: Freeze All | All original layers (original output removed) | New Dense(1) output (~129 params) | 1e-4 | 1 |
| M4: Gradual Unfreezing | Progressive (Phase 1→2→3) | Phase 1: final only → Phase 2: +readout → Phase 3: +GNN | 1e-3→1e-4→1e-5 | 3 (10 epochs each) |

### 5.4 Training hyperparameters

| Parameter | Source pretraining | Baseline & M1–M3 | M4 (Gradual Unfreezing) |
|-----------|-------------------|-------------------|-------------------------|
| Optimizer | Adam | Adam | Adam |
| Loss | Binary cross-entropy | Binary cross-entropy | Binary cross-entropy |
| Batch size | 32 | 32 | 32 |
| Max epochs | 10,000 | 10,000 | 10 per phase (×3 phases) |
| Early stopping patience | 7 | 7 | N/A (fixed epochs per phase) |
| Data split | 90% train / 10% val | 10-fold stratified CV | 10-fold stratified CV |
| GNN bias | False | False | False |
| GNN activations | tanh (all 4 layers) | tanh (all 4 layers) | tanh (all 4 layers) |

**Source pretraining note:** The source ("overtrained") model is trained on 100% of the source dataset (with a 90/10 train/val internal split, `random_state=42`) — NOT using cross-validation. Its purpose is knowledge acquisition, not fair benchmarking.

### 5.5 Evaluation metrics

| Metric | Type | Why included |
|--------|------|-------------|
| ROC-AUC | Threshold-free, discrimination | Primary metric, class-imbalance robust |
| G-Mean (GM) | Balance-aware | Geometric mean of sensitivity and specificity |
| Precision | Threshold-based | Positive predictive value |
| Recall (Sensitivity) | Threshold-based | True positive rate |
| F1 | Threshold-based | Harmonic mean of precision and recall |
| MCC | Threshold-based, balance-aware | Most informative single metric for binary classification |

### 5.6 Statistical tests

| Test | Purpose | When used |
|------|---------|-----------|
| Friedman test | Non-parametric test for differences among k related groups | First: are methods significantly different across folds? |
| Nemenyi post-hoc | Pairwise comparison after significant Friedman | If Friedman is significant: which specific pairs differ? |

### 5.7 SVM baseline summary

| Property | Value |
|----------|-------|
| Features | DPC (Dipeptide Composition), 400 dimensions |
| Model | SVM with RBF kernel (scikit-learn SVC) |
| Hyperparameters | C=5, gamma=0.001, class_weight="balanced" |
| Scaling | StandardScaler per fold |
| CV | 10-fold, same splits as GNN models |
| Reference paper | Gupta et al. (2013), PLoS ONE, ToxinPred |
| Reference paper result | Accuracy 94.50%, MCC 0.88 (5-fold, SVMlight) |

---

## 6. WRITING STYLE GUIDELINES

### 6.0 Page budget (8 pages total)

The paper must not exceed **8 pages**. Approximate budget (following Njirjak et al. 2024 style):

| Section | Target length |
|---------|--------------|
| Abstract | ~150–200 words |
| Introduction | ~0.75 pages (3 short paragraphs) |
| Results and discussion | ~3–3.5 pages (3 tables + 4 figures + interpretation woven together) |
| Conclusion | ~0.25–0.5 pages (1 paragraph) |
| Methods | ~2–2.5 pages (descriptive subheadings, no numbered subsections) |
| References + Declarations | ~1 page |

**Rule:** If a sentence can be cut without losing information, cut it. Merge related points. Avoid restating what tables and figures already show — reference them and state only the insight. Combining Results and Discussion into one section (as Njirjak et al. 2024 does) saves space by avoiding repetition.

### 6.1 Style reference

**PRIMARY references:** The two Nature Machine Intelligence papers in `paper/example-works/` (see Section 2.1.1):
- **Njirjak et al. (2024)** — co-authored by thesis mentor (Mauša) and co-mentor (Otović). **This is the #1 template.** Consult it before writing any section.
- **Leyva et al. (2025)** — confirms the same format; additional style reference.

**Secondary reference:** `paper/comparison/toxinPred-v1.pdf` (Gupta et al., ToxinPred, PLoS ONE 2013) — useful for scientific tone but uses a different journal format.

Key style principles from the example papers:
- **Scientific and concise** — every sentence carries information
- **Informative but not overly detailed** — describe what was done and why, not every implementation detail
- **Clear methodology flow** — dataset → features → model → evaluation → results
- **Quantitative claims backed by numbers** — always include metric values when making claims
- **Balanced discussion** — acknowledge limitations alongside findings
- **Figures and tables integrated into narrative** — not dumped in a block; each is referenced and discussed where relevant
- **Descriptive subheadings** within sections (not numbered) — e.g. "Building and fine-tuning the neural network models", "Testing and benchmarking the models"

### 6.2 Language conventions for this paper

- Use **English** for the paper text (the project docs have some Croatian — translate as needed)
- Use present tense for describing methods ("We train...", "The model uses...")
- Use past tense for describing specific experimental results ("Transfer achieved...")
- Write in **first-person plural** ("We investigate...", "Our approach...") or passive voice as the journal requires
- Be explicit about **which direction** (SMT→Peptide or Peptide→SMT) whenever stating results
- Always qualify metrics with **mean ± std** across folds

### 6.3 Key terminology (use consistently)

| Term | Meaning | Don't use |
|------|---------|-----------|
| Small molecule toxicity (SMT) | The MolToxPredDataset domain | "small mol", "drug molecules" |
| Peptide toxicity | The ToxinSequenceSMILES domain | "protein toxicity" |
| Graph convolution / GNN | The feature extraction backbone | "GCN" alone (ambiguous) |
| DeepGraphCNN | The specific architecture used | "DGCNN" (unless defined) |
| Readout | The Conv1D + Dense classifier head | "classifier", "output block" |
| Transfer method / strategy | One of the 4 TL approaches | "technique" |
| Baseline | Model trained from scratch, no transfer | "control", "reference" (less clear) |
| Fold | One of 10 CV partitions | "split" (which is broader) |

### 6.4 Important consistency note — Feature vocabulary size is 72

The shared one-hot atom vocabulary size is **72 elements**. This is the GNN input feature dimension used across all experiments.

| Number | What it means |
|--------|--------------|
| **72** | The **shared atom vocabulary** — one-hot encoding dimension for node features across all models and both domains |
| **4** | Unique elements actually found in the peptide dataset (C, N, O, S) — a small subset of the 72, but all 72 dimensions are still used |

⚠️ **Known typo in project docs:** Some older files (notably `models_architecture/project_process_documentation.md`) incorrectly state the vocabulary has **27 elements** with `NUM_FEATURES = 27`. This is a typo — the correct value is **72**. If you encounter "27" as a feature/vocabulary size anywhere in the project, treat it as an error.

**Key point for the paper:** The shared 72-element vocabulary is what makes cross-domain weight transfer architecturally possible — both domains' molecules are encoded into the same feature space, enabling direct weight compatibility between source and target models.

---

## 7. OVERLEAF TEMPLATE SPECIFICS

### 7.1 Template files (`paper/Jakov_Cvetko___Transfer_Learning/`)

| File | Purpose | Action needed |
|------|---------|---------------|
| `sn-article.tex` | Main manuscript file | **Replace boilerplate with actual paper content** |
| `sn-bibliography.bib` | Bibliography database | **Replace sample refs with real references** |
| `sn-jnl.cls` | Journal class file | Do not modify |
| `sn-mathphys-num.bst` | Bibliography style (active) | Do not modify |
| `fig.eps` | Sample figure | Replace with actual figures |
| `empty.eps` | Placeholder | Remove when not needed |
| Other `.bst` files | Alternative bibliography styles | Inactive, keep for reference |

### 7.2 Currently active document class

```latex
\documentclass[pdflatex,sn-nature]{sn-jnl}
```

This selects Nature Portfolio Numbered Reference Style (`sn-nature.bst`).

### 7.3 Working draft sections (`paper_draft.tex`)

The working draft `paper_draft.tex` (NOT `sn-article.tex`) has 4 body sections:
- `\section{Introduction}` — DONE (3 paragraphs)
- `\section{Results and discussion}` — TODO
- `\section{Conclusion}` — TODO
- `\section{Methods}` — TODO

Plus: `\abstract`, `\keywords` (done), backmatter declarations (scaffold), `\bibliography{sn-bibliography}` (populated with 20 entries).

The original `sn-article.tex` template is kept for reference only.

---

## 8. WHAT NEEDS TO BE DONE — TASK TRACKER

### 8.1 Content writing tasks

| # | Task | Status | Priority | Notes |
|---|------|--------|----------|-------|
| 1 | Write Title | DONE | High | "Toxicity Classification for Small Molecules and Peptides" |
| 2 | Write Abstract (~200 words) | TODO | High | Problem → approach → key findings |
| 3 | Write Introduction | DONE | High | 3 concise paragraphs (~300 words), solo author voice |
| 4 | Write Results and discussion | TODO | High | Combined section (Njirjak et al. 2024 style): 3 tables, 4 figures, interpretation, RQ1–RQ6 |
| 5 | Write Conclusion | TODO | Medium | 1 paragraph: key takeaways, implications, future work |
| 6 | Write Methods | TODO | High | Datasets, architecture, TL strategies, SVM baseline, CV, metrics, stats |
| 7 | Populate `sn-bibliography.bib` | DONE | High | 20 real BibTeX entries — 5 cited in Intro + 15 for future sections |

### 8.2 Figures — FINAL LIST (only PNGs from `paper/images/`)

All figures for the paper come exclusively from `paper/images/`. No additional figures need to be collected or generated.

| # | Figure | File in `paper/images/` | Description |
|---|--------|------------------------|-------------|
| Fig. 1 | Pipeline overview diagram | `project-diagram.png` | High-level project pipeline |
| Fig. 2 | Model architectures | `side-by-side-models-boxes.png` | All model configurations side-by-side |
| Fig. 3 | Radar chart — Peptide target | `radar_plots_peptide_median.png` | Method comparison across metrics (median), peptide target |
| Fig. 4 | Radar chart — SMT target | `radar_plots_smt_median.png` | Method comparison across metrics (median), SMT target |

### 8.3 Tables — FINAL LIST (exactly 3 tables)

All table data comes from files in `paper/images/`.

| # | Table | Data source | Content |
|---|-------|-------------|---------|
| Table 1 | Dataset characteristics | `paper/images/dataset_analysis_results.md` | Summary statistics for both datasets (samples, class ratio, elements, graph sizes) |
| Table 2 | Peptide target results | `paper/images/eval-book-peptide-results.xlsx` | All methods × all metrics for SMT→Peptide transfer direction |
| Table 3 | SMT target results | `paper/images/eval-book-smt-results.xlsx` | All methods × all metrics for Peptide→SMT transfer direction |

---

## 9. RESULTS DATA LOCATION MAP

When populating results, get numbers from these sources:

| What | Where to find it |
|------|------------------|
| Peptide target results (mean) | `eval-book-peptide-results.xlsx` |
| Peptide target results (median) | `eval-book-peptide-results-median.xlsx` |
| SMT target results (mean) | `eval-book-smt-results.xlsx` |
| SMT target results (median) | `eval-book-smt-results-median.xlsx` |
| Per-fold detailed results | Inside each `*_eval_*` folder |
| Box-whisker plots | `*_eval_*/box_whisker_*.png` |
| ROC curves | `*_eval_*/comparison_roc_curves.png` |
| PR curves | `*_eval_*/comparison_pr_curves.png` |
| Confusion matrices | `*_eval_*/confusion_matrix_*.png` |
| Heatmaps (Nemenyi) | `*_eval_*/heatmap_*.png` |
| Radar charts | Root: `radar_plots_*.png` |
| Violin plots | `*_eval_*/advanced_violin_plots.png` |
| Calibration curves | `*_eval_*/advanced_calibration_curve.png` |
| SVM results | `svm/DPC_SVM_results_balanced_clean.xlsx` (recommended) |

### Eval folder naming convention

| Model size | Peptide eval folder | SMT eval folder |
|------------|--------------------|-----------------------|
| Standard | `extra_features_eval_peptide/` | `extra_features_eval_small_mol_tox/` |
| Large Layers | `large_layers_eval_peptide/` | `large_layers_eval_small/` |
| Inflated | `inflated_eval_peptide/` | `inflated_eval_small_mol_tox/` |
| Extra Inflated | (inside `extra_inflated_*` folders) | (inside `extra_inflated_*` folders) |
| 72-feature variant | `inflated_72_feature_eval_peptide/` | — |

---

## 10. CROSS-VALIDATION DESIGN (critical for paper Methods section)

The fairness of method comparisons rests on these design choices:

1. **Stratified 10-fold CV** — class proportions preserved in each fold
2. **Shared fold definitions** — baseline and all TL methods use identical train/val/test splits per fold
3. **Fold files saved as `.pkl`** — in `cv_splits/` and `*_cv_splits/` folders
4. **Within each fold:** ~64% train, ~16% validation (for early stopping), ~20% test
5. **SVM baseline uses the same splits** — created via `svm/create_aligned_splits.py`

This enables paired fold-level comparisons and is required for Friedman test validity.

---

## 11. CONVENTIONS AND PITFALLS

### 11.1 Naming conventions

| In code/folders | In the paper |
|----------------|-------------|
| "overtrained" | "pretrained" or "source model" |
| "large_layers" | "Large Layers" configuration |
| "inflated" | "Inflated" configuration |
| "extra_inflated" | "Extra Inflated" configuration |
| "freeze gnn" | "Method 1" or "Freeze GNN" |
| "freeze readout" | "Method 2" or "Freeze Readout" |
| "freeze all" | "Method 3" or "Freeze All" |
| "gradual unfreezing" | "Method 4" or "Gradual Unfreezing" |
| "eval-book" | Results summary tables |

### 11.2 Known issues to handle

1. **Feature vocabulary size is 72** — The shared one-hot atom vocabulary has **72 elements** (see Section 6.4). Use 72 in Methods when describing the GNN input feature dimension. Some older project docs (e.g. `project_process_documentation.md`) incorrectly state 27 — this is a typo.
2. **"Overtrained" terminology** — In the paper, call this "pretrained source model" (overtraining is intentional to maximize knowledge extraction, but the word carries negative connotations in ML)
3. **Extra Inflated [1024, 512, 256, 1]** — Present in some experiments but the architecture README only lists 3 sizes (Standard, Large Layers, Inflated). Verify which sizes have complete results for both directions before including
4. **Mean vs Median aggregation** — Both xlsx files exist. Choose one consistently for the paper (mean ± std is conventional). The `paper/images/` folder contains median radar plots, suggesting median may be preferred for visualization
5. **Croatian in documentation** — Some docs (`tables-final.md`, `tables-detailed.md`, SVM docs, `questions-to-be-asnwerd-in-paper.txt`) are in Croatian. Content is valid but needs translation for the English paper
6. **Early stopping patience** — Architecture docs (`method*.md`) consistently state patience=7 for M1, M2, and M3. Method 4 (Gradual Unfreezing) uses fixed 10 epochs per phase instead of early stopping. The earlier version of this guide incorrectly stated patience=3 for TL methods
7. **Conv1D kernel size depends on model size** — The first Conv1D layer uses `kernel_size = sum(layer_sizes)`. For Standard [25,25,25,1] that's 76; for Inflated [512,256,128,1] that's 897. This is a detail worth mentioning in Methods

### 11.3 What the paper should NOT include

- Implementation-level details (specific Python scripts, file paths, pickle formats)
- Raw code snippets (unless describing an algorithm pseudocode-style)
- Intermediate debug outputs
- Speculation beyond what the data supports

---

## 12. BIBLIOGRAPHY — REFERENCES IN `sn-bibliography.bib`

The `sn-bibliography.bib` has been populated with **20 real BibTeX entries** (sample entries removed). The file uses `sn-nature.bst` (Nature Portfolio numbered reference style).

### 12.1 References currently cited in the Introduction (5)

| # | BibTeX key | Reference | Where cited |
|---|-----------|-----------|-------------|
| 1 | `gupta2013toxinpred` | Gupta et al. (2013) "In Silico Approach for Predicting Toxicity of Peptides and Proteins" *PLoS ONE* 8(9):e73957 | Intro ¶1, ¶3 |
| 2 | `gilmer2017mpnn` | Gilmer et al. (2017) "Neural Message Passing for Quantum Chemistry" *ICML*, PMLR 70:1263–1272 | Intro ¶1 |
| 3 | `zhang2018deepgraphcnn` | Zhang et al. (2018) "An End-to-End Deep Learning Architecture for Graph Classification" *AAAI* 32(1):4438–4445 | Intro ¶1 |
| 4 | `pan2010transfer` | Pan & Yang (2010) "A Survey on Transfer Learning" *IEEE TKDE* 22(10):1345–1359 | Intro ¶2 |
| 5 | `demsar2006statistical` | Demšar (2006) "Statistical Comparisons of Classifiers over Multiple Data Sets" *JMLR* 7:1–30 | Intro ¶3 |

### 12.2 Additional references for Methods / Results / Discussion (15)

| # | BibTeX key | Reference | Use in paper |
|---|-----------|-----------|--------------|
| 6 | `weininger1988smiles` | Weininger (1988) "SMILES, a Chemical Language…" *J. Chem. Inf. Comput. Sci.* 28(1):31–36 | Methods — molecular representation |
| 7 | `rdkit` | RDKit: Open-Source Cheminformatics (v2022.9.5) | Methods — graph construction |
| 8 | `kipf2017gcn` | Kipf & Welling (2017) "Semi-Supervised Classification with Graph Convolutional Networks" *ICLR* | Methods — GNN background |
| 9 | `hu2020pretraining` | Hu et al. (2020) "Strategies for Pre-Training Graph Neural Networks" *ICLR* | Discussion — TL on graphs context |
| 10 | `stellargraph` | Data61/CSIRO StellarGraph Library (v1.2.1) | Methods — implementation |
| 11 | `wu2018moleculenet` | Wu et al. (2018) "MoleculeNet: A Benchmark for Molecular Machine Learning" *Chem. Sci.* 9:513–530 | Introduction / Discussion — molecular ML context |
| 12 | `yosinski2014transferable` | Yosinski et al. (2014) "How Transferable Are Features in Deep Neural Networks?" *NeurIPS* 27:3320–3328 | Discussion — layer transferability |
| 13 | `tensorflow` | Abadi et al. (2016) "TensorFlow: A System for Large-Scale Machine Learning" *OSDI*:265–283 | Methods — implementation |
| 14 | `kingma2015adam` | Kingma & Ba (2015) "Adam: A Method for Stochastic Optimization" *ICLR* | Methods — optimizer |
| 15 | `matthews1975mcc` | Matthews (1975) "Comparison of the Predicted and Observed Secondary Structure of T4 Phage Lysozyme" *BBA* 405(2):442–451 | Methods — metrics |
| 16 | `chicco2020mcc` | Chicco & Jurman (2020) "The Advantages of the MCC over F1 Score and Accuracy…" *BMC Genomics* 21:6 | Methods / Discussion — why MCC |
| 17 | `friedman1937` | Friedman (1937) "The Use of Ranks…" *JASA* 32(200):675–701 | Methods — statistical testing |
| 18 | `pedregosa2011sklearn` | Pedregosa et al. (2011) "Scikit-learn: Machine Learning in Python" *JMLR* 12:2825–2830 | Methods — SVM implementation |
| 19 | `cortes1995svm` | Cortes & Vapnik (1995) "Support-Vector Networks" *Machine Learning* 20(3):273–297 | Methods — SVM background |
| 20 | `joachims1999svm` | Joachims (1999) "Making Large-Scale SVM Learning Practical" *Advances in Kernel Methods*, MIT Press | Discussion — ToxinPred comparison |

### 12.3 Additional references to consider adding

| Topic | Candidate | BibTeX key (to add) |
|-------|-----------|---------------------|
| Small molecule dataset source | Original source of MolToxPredDataset | Need to identify |
| Peptide dataset source | Original sources (ATDB, Arachno-Server, ConoServer, etc.) from ToxinPred | Covered by `gupta2013toxinpred` |
| Tox21 challenge | Huang et al. (2016) *Front. Environ. Sci.* | `huang2016tox21` (already in .bib) |
| Therapeutic peptides review | Fosgerau & Hoffmann (2015) *Drug Discovery Today* | `fosgerau2015peptide` (already in .bib) |

---

## 13. QUICK-START FOR LLM AGENT

When asked to help write a specific paper section:

1. **Read this file first** for context on what exists and conventions
2. **Consult `paper/example-works/`** — read the corresponding section in Njirjak et al. (2024) and Leyva et al. (2025) to match tone, structure, and depth. **This is the #1 priority before writing anything.**
3. **Check `paper/images/eval-book-*.xlsx`** (or root-level copies) for actual numerical results
4. **Read `models_architecture/` docs** for exact technical details — especially `project_process_documentation.md` (761 lines of detailed pipeline) and the individual `method*.md` files for layer-level specifics
5. **Use `introduction/` drafts** as detailed process knowledge (not as text to copy — they're too verbose for a paper)
6. **Consult `paper/questions/`** to ensure the section answers one or more of the 6 formal research questions (RQ1–RQ6)
7. **Output LaTeX** that can be directly pasted into `paper_draft.tex` on Overleaf
8. **Use `\cite{key}` format** for references, with keys from `sn-bibliography.bib`
9. **Tables in LaTeX** should use `booktabs` package (`\toprule`, `\midrule`, `\botrule`)
10. **Figures** from `paper/images/` should be referenced as `\includegraphics` — PNG format is fine for Overleaf
11. **Always specify transfer direction** when discussing results (SMT→Peptide or Peptide→SMT)
12. **Report metrics as mean ± std** (3 decimal places for mean, 2-3 for std)
13. **Use 72 for the shared atom vocabulary size** in Methods — any doc stating 27 is a typo (see Section 6.4)
14. **Call it "pretrained source model"**, never "overtrained model" in the paper text
15. **Keep to 4 sections only**: Introduction, Results and discussion, Conclusion, Methods — no extra sections
16. **Use descriptive subheadings** (not numbered subsections) within sections when needed

---

*Last updated: 2026-02-27*
*This file should be updated as the paper progresses and new results/decisions are made.*
