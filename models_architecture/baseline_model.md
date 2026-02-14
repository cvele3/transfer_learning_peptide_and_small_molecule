# Baseline Model Architecture

## Overview
The baseline model is a **freshly initialized model** trained from scratch on the target dataset. It serves as the comparison benchmark for evaluating transfer learning methods. The architecture is identical to the overtrained model, but weights are randomly initialized.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| k (SortPooling) | 25 |
| Layer Sizes | Varies by configuration |
| Activations | tanh, tanh, tanh, tanh |
| Bias | False |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Loss Function | Binary Crossentropy |
| Early Stopping Patience | 7 epochs |

## Layer Size Configurations

| Configuration Name | GNN Layer Sizes |
|-------------------|-----------------|
| Standard (Extra Features) | [25, 25, 25, 1] |
| Large Layers | [125, 125, 125, 1] |
| Inflated | [512, 256, 128, 1] |

## Architecture Diagram

```
Input (Graph Features + Adjacency Matrix)
    │
    ▼
┌─────────────────────────────────────┐
│       DeepGraphCNN Block            │
│  ┌─────────────────────────────┐    │
│  │ GraphConvolution Layer 1    │    │
│  │ (layer_sizes[0], tanh)      │    │
│  └─────────────────────────────┘    │
│              │                      │
│  ┌─────────────────────────────┐    │
│  │ GraphConvolution Layer 2    │    │
│  │ (layer_sizes[1], tanh)      │    │
│  └─────────────────────────────┘    │
│              │                      │
│  ┌─────────────────────────────┐    │
│  │ GraphConvolution Layer 3    │    │
│  │ (layer_sizes[2], tanh)      │    │
│  └─────────────────────────────┘    │
│              │                      │
│  ┌─────────────────────────────┐    │
│  │ GraphConvolution Layer 4    │    │
│  │ (layer_sizes[3], tanh)      │    │
│  └─────────────────────────────┘    │
│              │                      │
│  ┌─────────────────────────────┐    │
│  │ SortPooling (k=25)          │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│     Readout / Classification        │
│  ┌─────────────────────────────┐    │
│  │ Conv1D (16 filters)         │    │
│  │ kernel=sum(layer_sizes)     │    │
│  └─────────────────────────────┘    │
│              │                      │
│  ┌─────────────────────────────┐    │
│  │ MaxPool1D (pool_size=2)     │    │
│  └─────────────────────────────┘    │
│              │                      │
│  ┌─────────────────────────────┐    │
│  │ Conv1D (32 filters)         │    │
│  │ kernel=5, stride=1          │    │
│  └─────────────────────────────┘    │
│              │                      │
│  ┌─────────────────────────────┐    │
│  │ Flatten                     │    │
│  └─────────────────────────────┘    │
│              │                      │
│  ┌─────────────────────────────┐    │
│  │ Dense (128 units, ReLU)     │    │
│  └─────────────────────────────┘    │
│              │                      │
│  ┌─────────────────────────────┐    │
│  │ Dropout (rate=0.2)          │    │
│  └─────────────────────────────┘    │
│              │                      │
│  ┌─────────────────────────────┐    │
│  │ Dense (1 unit, Sigmoid)     │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
    │
    ▼
Output (Binary Classification: 0 or 1)
```

## Training Process

1. **Data Loading**: Load preprocessed graph data for target dataset
2. **10-Fold Cross Validation**: Use predefined CV splits
3. **For each fold**:
   - Initialize fresh model with random weights
   - Train on training set with validation monitoring
   - Early stopping (patience=7)
   - Evaluate on test set
   - Save model as `baseline_*_fold_X.h5`

## Key Difference from Transfer Learning

| Aspect | Baseline | Transfer Learning |
|--------|----------|-------------------|
| Initial Weights | Random | Pre-trained |
| Prior Knowledge | None | Source domain knowledge |
| Training | From scratch | Fine-tuning |

## Purpose

The baseline model establishes the **performance benchmark** without transfer learning. Comparing transfer learning methods against baseline shows whether knowledge transfer provides any benefit.
