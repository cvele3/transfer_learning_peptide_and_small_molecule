# Method 4: Gradual Unfreezing + Discriminative Fine-Tuning

## Overview
This is the most sophisticated transfer learning approach, combining **gradual unfreezing** with **discriminative learning rates**. Layers are unfrozen progressively from top to bottom, with decreasing learning rates for deeper layers.

## Concept

The approach is based on the observation that:
- **Earlier layers** learn general features → should be fine-tuned carefully
- **Later layers** learn task-specific features → can be trained more aggressively
- **Gradual unfreezing** prevents catastrophic forgetting

## Three-Phase Training Process

### Phase 1: Train Final Layers Only
```
GNN Layers     → 🔒 FROZEN
Readout Layers → 🔒 FROZEN
Final Layers   → 🔓 TRAINABLE (LR: 1e-3)
```

### Phase 2: Unfreeze Readout Layers
```
GNN Layers     → 🔒 FROZEN
Readout Layers → 🔓 TRAINABLE (LR: 1e-4)
Final Layers   → 🔓 TRAINABLE (LR: 1e-4)
```

### Phase 3: Unfreeze GNN Layers
```
GNN Layers     → 🔓 TRAINABLE (LR: 1e-5)
Readout Layers → 🔓 TRAINABLE (LR: 1e-5)
Final Layers   → 🔓 TRAINABLE (LR: 1e-5)
```

## Architecture with Layer Groups

```
┌─────────────────────────────────────────────────────────┐
│                    GNN LAYERS                           │
│  ┌─────────────────────────────┐                        │
│  │ GraphConvolution Layer 1    │  Phase 3: LR=1e-5     │
│  └─────────────────────────────┘                        │
│  ┌─────────────────────────────┐                        │
│  │ GraphConvolution Layer 2    │  Phase 3: LR=1e-5     │
│  └─────────────────────────────┘                        │
│  ┌─────────────────────────────┐                        │
│  │ GraphConvolution Layer 3    │  Phase 3: LR=1e-5     │
│  └─────────────────────────────┘                        │
│  ┌─────────────────────────────┐                        │
│  │ GraphConvolution Layer 4    │  Phase 3: LR=1e-5     │
│  └─────────────────────────────┘                        │
│  ┌─────────────────────────────┐                        │
│  │ SortPooling (k=25)          │  Phase 3: LR=1e-5     │
│  └─────────────────────────────┘                        │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                  READOUT LAYERS                         │
│  ┌─────────────────────────────┐                        │
│  │ Conv1D (16 filters)         │  Phase 2: LR=1e-4     │
│  └─────────────────────────────┘                        │
│  ┌─────────────────────────────┐                        │
│  │ MaxPool1D                   │  Phase 2: LR=1e-4     │
│  └─────────────────────────────┘                        │
│  ┌─────────────────────────────┐                        │
│  │ Conv1D (32 filters)         │  Phase 2: LR=1e-4     │
│  └─────────────────────────────┘                        │
│  ┌─────────────────────────────┐                        │
│  │ Flatten                     │  Phase 2: LR=1e-4     │
│  └─────────────────────────────┘                        │
│  ┌─────────────────────────────┐                        │
│  │ Dense (128, ReLU)           │  Phase 2: LR=1e-4     │
│  └─────────────────────────────┘                        │
│  ┌─────────────────────────────┐                        │
│  │ Dropout (0.2)               │  Phase 2: LR=1e-4     │
│  └─────────────────────────────┘                        │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                   FINAL LAYERS                          │
│  ┌─────────────────────────────┐                        │
│  │ Dense (1, Sigmoid)          │  Phase 1: LR=1e-3     │
│  └─────────────────────────────┘                        │
└─────────────────────────────────────────────────────────┘
```

## Implementation

```python
# Load pretrained model
model = load_pretrained_model()

# Define layer groups
gnn_layers = [layer for layer in model.layers 
              if "deep_graph_cnn" in layer.name or "graph_conv" in layer.name]
readout_layers = [layer for layer in model.layers 
                  if any(x in layer.name for x in ["dense", "dropout", "flatten", "readout"])]
final_layers = [layer for layer in model.layers 
                if layer.name not in [l.name for l in (gnn_layers + readout_layers)]]

# Initially freeze all layers
for layer in model.layers:
    layer.trainable = False

epochs_per_phase = 10

# --- Phase 1: Train only final layers ---
for layer in final_layers:
    layer.trainable = True
model.compile(optimizer=Adam(learning_rate=1e-3), loss=binary_crossentropy, metrics=["accuracy"])
model.fit(train_gen, validation_data=val_gen, epochs=epochs_per_phase, callbacks=[callback])

# --- Phase 2: Unfreeze readout layers ---
for layer in readout_layers:
    layer.trainable = True
model.compile(optimizer=Adam(learning_rate=1e-4), loss=binary_crossentropy, metrics=["accuracy"])
model.fit(train_gen, validation_data=val_gen, epochs=epochs_per_phase, callbacks=[callback])

# --- Phase 3: Unfreeze GNN layers ---
for layer in gnn_layers:
    layer.trainable = True
model.compile(optimizer=Adam(learning_rate=1e-5), loss=binary_crossentropy, metrics=["accuracy"])
model.fit(train_gen, validation_data=val_gen, epochs=epochs_per_phase, callbacks=[callback])
```

## Training Parameters

| Phase | Layers Trained | Learning Rate | Epochs |
|-------|----------------|---------------|--------|
| 1 | Final only | 1e-3 | 10 |
| 2 | Final + Readout | 1e-4 | 10 |
| 3 | All layers | 1e-5 | 10 |

## Learning Rate Schedule

```
LR
│
1e-3 ├────────┐
│              │
1e-4 ├─────────────────┐
│                       │
1e-5 ├──────────────────────────┐
│                                │
└────┴────────┴─────────┴────────┴──► Phase
     Phase 1   Phase 2   Phase 3
```

## Rationale

1. **Start with final layers**: Quick adaptation to new task
2. **Add readout layers**: Learn task-specific feature processing
3. **Finally GNN layers**: Careful fine-tuning of foundational representations
4. **Decreasing learning rates**: Preserve pretrained knowledge in deeper layers

## When to Use

✅ When you want the best possible performance
✅ When source and target domains are moderately different
✅ When you have sufficient training data
✅ When training time is not a constraint

## Advantages

- Most flexible approach
- Prevents catastrophic forgetting
- Allows domain adaptation at all levels
- Often achieves best results

## Output Files

Models are saved as:
- `gradual_unfreezing_*_to_*_fold_X.h5`
