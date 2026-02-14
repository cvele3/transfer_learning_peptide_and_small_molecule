# Method 2: Freeze Readout Layers

## Overview
In this transfer learning approach, the **readout/classification layers are frozen** while the GNN layers are fine-tuned on the target dataset. This allows the model to learn new graph feature representations while keeping the classification decision boundary fixed.

## Concept

The idea is that the readout layers have learned a **good classification decision boundary** that can be reused. By freezing these layers, we:
- Preserve the learned classification patterns
- Allow GNN to adapt its feature extraction to the new domain
- Fine-tune low-level graph representations

## Frozen vs Trainable Layers

```
┌─────────────────────────────────────┐
│       DeepGraphCNN Block            │
│  ┌─────────────────────────────┐    │
│  │ GraphConvolution Layer 1    │ 🔓 │  TRAINABLE
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │ GraphConvolution Layer 2    │ 🔓 │  TRAINABLE
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │ GraphConvolution Layer 3    │ 🔓 │  TRAINABLE
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │ GraphConvolution Layer 4    │ 🔓 │  TRAINABLE
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │ SortPooling (k=25)          │ 🔓 │  TRAINABLE
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│     Readout / Classification        │
│  ┌─────────────────────────────┐    │
│  │ Conv1D (16 filters)         │ 🔒 │  FROZEN
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │ MaxPool1D (pool_size=2)     │ 🔒 │  FROZEN
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │ Conv1D (32 filters)         │ 🔒 │  FROZEN
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │ Flatten                     │ 🔒 │  FROZEN
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │ Dense (128 units, ReLU)     │ 🔒 │  FROZEN
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │ Dropout (rate=0.2)          │ 🔒 │  FROZEN
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │ Dense (1 unit, Sigmoid)     │ 🔒 │  FROZEN
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
```

## Implementation

```python
# Load pretrained model
model = load_pretrained_model()

# Freeze readout layers (dense, dropout, flatten, readout)
for layer in model.layers:
    if any(x in layer.name for x in ["dense", "dropout", "flatten", "readout"]):
        layer.trainable = False
    else:
        layer.trainable = True

# Compile with lower learning rate 1e-5
model.compile(
    optimizer=Adam(learning_rate=1e-5), 
    loss=binary_crossentropy, 
    metrics=["accuracy"]
)
```

## Training Parameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-5 (lower to prevent catastrophic forgetting) |
| Optimizer | Adam |
| Loss | Binary Crossentropy |
| Early Stopping | patience=7 |
| Batch Size | 32 |

## Rationale

- **Readout layers** have learned how to make predictions from graph embeddings → **transferable classification logic**
- **GNN layers** need to adapt to different molecular structures → **domain-specific**
- Lower learning rate prevents destroying the pretrained GNN weights too quickly

## When to Use

✅ When classification tasks are similar between domains
✅ When graph structures differ significantly between domains
✅ When you want to preserve the learned classification patterns

## Output Files

Models are saved as:
- `freeze_readout_*_to_*_fold_X.h5`
