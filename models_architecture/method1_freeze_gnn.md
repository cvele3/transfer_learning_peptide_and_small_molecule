# Method 1: Freeze GNN Layers

## Overview
In this transfer learning approach, the **GNN (Graph Neural Network) layers are frozen** while the readout/classification layers are fine-tuned on the target dataset. This preserves the learned graph feature extraction capabilities from the source domain.

## Concept

The idea is that the GNN layers learn **general graph structure representations** that are transferable across domains. By freezing these layers, we:
- Preserve learned graph convolution filters
- Only adapt the classification head to the new task
- Reduce training time and prevent overfitting

## Frozen vs Trainable Layers

```
┌─────────────────────────────────────┐
│       DeepGraphCNN Block            │
│  ┌─────────────────────────────┐    │
│  │ GraphConvolution Layer 1    │ 🔒 │  FROZEN
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │ GraphConvolution Layer 2    │ 🔒 │  FROZEN
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │ GraphConvolution Layer 3    │ 🔒 │  FROZEN
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │ GraphConvolution Layer 4    │ 🔒 │  FROZEN
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │ SortPooling (k=25)          │ 🔒 │  FROZEN
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│     Readout / Classification        │
│  ┌─────────────────────────────┐    │
│  │ Conv1D (16 filters)         │ 🔓 │  TRAINABLE
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │ MaxPool1D (pool_size=2)     │ 🔓 │  TRAINABLE
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │ Conv1D (32 filters)         │ 🔓 │  TRAINABLE
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │ Flatten                     │ 🔓 │  TRAINABLE
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │ Dense (128 units, ReLU)     │ 🔓 │  TRAINABLE
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │ Dropout (rate=0.2)          │ 🔓 │  TRAINABLE
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │ Dense (1 unit, Sigmoid)     │ 🔓 │  TRAINABLE
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
```

## Implementation

```python
# Load pretrained model
model = load_pretrained_model()

# Freeze GNN layers (layers containing "deep_graph_cnn" or "graph_conv")
for layer in model.layers:
    if "deep_graph_cnn" in layer.name or "graph_conv" in layer.name:
        layer.trainable = False
    else:
        layer.trainable = True

# Compile with learning rate 1e-4
model.compile(
    optimizer=Adam(learning_rate=1e-4), 
    loss=binary_crossentropy, 
    metrics=["accuracy"]
)
```

## Training Parameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-4 |
| Optimizer | Adam |
| Loss | Binary Crossentropy |
| Early Stopping | patience=7 |
| Batch Size | 32 |

## Rationale

- **GNN layers** capture graph topology and node feature relationships → **domain-agnostic**
- **Readout layers** map graph representations to task-specific predictions → **task-specific**
- By freezing GNN, we assume graph structural patterns are similar across domains

## When to Use

✅ When source and target domains have similar graph structures
✅ When target dataset is small (prevents overfitting)
✅ When computational resources are limited

## Output Files

Models are saved as:
- `freeze_gnn_*_to_*_fold_X.h5`
