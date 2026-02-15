# Diagram Redesign Plan

## Goal
Compact side-by-side view with **3 grouped boxes** per model column.

## Box Structure (per model)

```
┌─────────────────────┐
│ GNN                 │  
│ • GraphConv 1-4     │
│ • SortPooling       │
└─────────────────────┘
         ↓
┌─────────────────────┐
│ Readout (CNN)       │
│ • Conv1D → Pool     │
│ • Conv1D → Flatten  │
└─────────────────────┘
         ↓
┌─────────────────────┐
│ Head                │
│ • Dense 128 + Drop  │
│ • Dense 1 (output)  │
└─────────────────────┘
```

## Freeze States by Method

| Block   | Baseline | M1 | M2 | M3 | M4 |
|---------|----------|----|----|----|----|
| GNN     | 🟢 Train | 🔵 Freeze | 🟢 Train | 🔵 Freeze | ⚪→🟢 |
| Readout | 🟢 Train | 🟢 Train | 🔵 Freeze | 🔵 Freeze | ⚪→🟢 |
| Head    | 🟢 Train | 🟢 Train | 🔵 Freeze | 🟢 NEW | 🟢 Train |

## Draw.io Implementation

### Colors
- 🟢 Green `#d5e8d4` / `#82b366` = Trainable
- 🔵 Blue `#dae8fc` / `#6c8ebf` = Frozen
- ⚪ Gray `#f5f5f5` / dashed = Replaced
- Gradient (blue→green) = Gradual unfreeze

### Layout
- **5 columns** (220px each, 40px gap) = ~1300px total
- **3 rows** per column (container boxes)
- Container height: ~100-120px each (with 3-4 internal layers)

### Container Box Style
```
style="rounded=1;strokeWidth=2;dashed=0;container=1;collapsible=0;"
```

### Internal Layer Style (smaller, no stroke)
```
style="rounded=1;strokeWidth=0;fontSize=9;fontStyle=0;"
```

### Structure per Column
```xml
<!-- Container: GNN -->
<mxCell value="GNN" style="swimlane;...fillColor=COLOR;">
  <!-- Child layers inside -->
  <mxCell value="GraphConv 1-4" .../>
  <mxCell value="SortPooling" .../>
</mxCell>

<!-- Container: Readout -->
<mxCell value="Readout" style="swimlane;...">
  <mxCell value="Conv1D→Pool→Conv1D" .../>
  <mxCell value="Flatten" .../>
</mxCell>

<!-- Container: Head -->
<mxCell value="Head" style="swimlane;...">
  <mxCell value="Dense 128 + Dropout" .../>
  <mxCell value="Dense 1 (Sigmoid)" .../>
</mxCell>
```

### Dimensions
| Element | Width | Height |
|---------|-------|--------|
| Container | 180px | 80px |
| Internal layer | 160px | 20px |
| Column spacing | 200px | - |
| Total width | ~1100px | ~350px |

## Simplified Legend
```
[🟢 Trainable] [🔵 Frozen] [⚪ Replaced] [Gradient = Gradual]
```

## M4 Phases (small table, right side)
| Phase | GNN | Readout | Head |
|-------|-----|---------|------|
| 1 | 🔵 | 🔵 | 🟢 |
| 2 | 🔵 | 🟢 | 🟢 |
| 3 | 🟢 | 🟢 | 🟢 |
