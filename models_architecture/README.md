# Model Architectures Overview

This folder contains documentation for all model architectures used in the transfer learning experiments.

## Base Architecture: DeepGraphCNN

All models share the same base architecture - a **Deep Graph Convolutional Neural Network (DeepGraphCNN)** for graph classification.

### Architecture Summary

```
Input Graph
    │
    ▼
┌──────────────────┐
│  DeepGraphCNN    │  ← Graph feature extraction
│  (4 GCN layers)  │
│  + SortPooling   │
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  Readout Layers  │  ← Conv1D + MaxPool + Dense
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  Output Layer    │  ← Binary classification
│  (Sigmoid)       │
└──────────────────┘
```

## Model Configurations

| Configuration | GNN Layer Sizes | Description |
|--------------|-----------------|-------------|
| Standard | [25, 25, 25, 1] | Smallest model |
| Large Layers | [125, 125, 125, 1] | Medium model |
| Inflated | [512, 256, 128, 1] | Largest model |

## Models Documentation

| File | Description |
|------|-------------|
| [overtrained_model.md](overtrained_model.md) | Source model trained on peptide/SMT dataset |
| [baseline_model.md](baseline_model.md) | Model trained from scratch (benchmark) |
| [method1_freeze_gnn.md](method1_freeze_gnn.md) | TL: Freeze GNN, train readout |
| [method2_freeze_readout.md](method2_freeze_readout.md) | TL: Freeze readout, train GNN |
| [method3_freeze_all.md](method3_freeze_all.md) | TL: Freeze all, add new output |
| [method4_gradual_unfreezing.md](method4_gradual_unfreezing.md) | TL: Gradual unfreezing |

## Transfer Learning Methods Comparison

| Method | Frozen Layers | Trainable Layers | Learning Rate | Complexity |
|--------|---------------|------------------|---------------|------------|
| Baseline | None | All (random init) | 1e-4 | Medium |
| Method 1 | GNN | Readout | 1e-4 | Low |
| Method 2 | Readout | GNN | 1e-5 | Medium |
| Method 3 | All | New output only | 1e-4 | Very Low |
| Method 4 | Progressive | Progressive | 1e-3 → 1e-5 | High |

## Visual Summary

```
                    BASELINE              METHOD 1            METHOD 2            METHOD 3            METHOD 4
                    (No TL)            (Freeze GNN)      (Freeze Readout)     (Freeze All)      (Gradual Unfreeze)
                    
GNN Layers:         🔓 Random           🔒 Pretrained      🔓 Pretrained       🔒 Pretrained       🔓 Phase 3
                                                           (Fine-tuned)
                    
Readout Layers:     🔓 Random           🔓 Pretrained      🔒 Pretrained       🔒 Pretrained       🔓 Phase 2
                                        (Fine-tuned)
                    
Output Layer:       🔓 Random           🔓 Pretrained      🔒 Pretrained       🔓 NEW Layer        🔓 Phase 1
                                        (Fine-tuned)                           (Trained)

Legend: 🔒 = Frozen, 🔓 = Trainable
```

## Datasets

The models are used for binary classification on two datasets:

1. **Peptide Dataset**: Antimicrobial peptide toxicity prediction
2. **SMT Dataset**: Small molecule toxicity prediction

Transfer learning is performed bidirectionally:
- Peptide → SMT
- SMT → Peptide

## Evaluation Metrics

All models are evaluated using:
- **ROC-AUC**: Area Under ROC Curve
- **GM**: Geometric Mean of sensitivity and specificity
- **Precision**: True positives / Predicted positives
- **Recall**: True positives / Actual positives  
- **F1**: Harmonic mean of precision and recall
- **MCC**: Matthews Correlation Coefficient
