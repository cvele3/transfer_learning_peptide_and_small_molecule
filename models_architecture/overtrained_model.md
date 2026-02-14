# Overtrained Model Architecture

## Overview
The overtrained model is the **source model** used for transfer learning. It is trained on one dataset (either peptide or small molecule toxicity) until convergence, and then its weights are transferred to the target domain.

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

1. **Data Loading**: Load preprocessed graph data (StellarGraph objects)
2. **Train/Val Split**: 90% training, 10% validation
3. **Training**: Train until early stopping triggers (patience=7)
4. **Save Model**: Save as `.h5` file for transfer learning

## Purpose

The overtrained model serves as the **pre-trained model** that captures domain-specific knowledge from the source dataset. Its learned weights are then transferred to the target domain through various transfer learning methods.
