# Single TSO Period Enforcement - Implementation Guide

## 🎯 Problem Statement

**Observation**: Model predicts multiple fragmented TSO periods per night
**Reality**: People typically have one main TSO (time-segment-of-interest) period per night
**Solution**: Hybrid approach using continuity loss during training + post-processing during inference

---

## 📚 Implementation Overview

### **Three New Functions Added to `Helpers/DL_helpers.py`:**

1. **`tso_continuity_loss()`** - Regularization loss for training
2. **`measure_loss_tso_with_continuity()`** - Combined loss function
3. **`enforce_single_tso_period()`** - Post-processing for inference
4. **`batch_enforce_single_tso()`** - Batch post-processing with sequence lengths

---

## 🔧 Usage Guide

### **Phase 1: Training with Continuity Loss**

#### **Option A: Modify existing training script (Recommended)**

**File**: `predict_TSO_segment_patch_h5.py` (or your training script)

```python
# At the top, update import
from Helpers.DL_helpers import (
    measure_loss_tso,
    measure_loss_tso_with_continuity,  # ✓ Add this
    enforce_single_tso_period,          # ✓ Add this
    batch_enforce_single_tso,           # ✓ Add this
    ...
)

# In best_params or config, add continuity weight
best_params = {
    'batch_size': 16,
    'lr': 1e-3,
    ...
    'continuity_weight': 0.1,  # ✓ Add this (0.0 = disabled, 0.1-0.5 = reasonable range)
}

# In training loop, replace loss calculation:
# OLD:
# total_loss = measure_loss_tso(outputs, pad_Y, x_lens)

# NEW:
if best_params.get('continuity_weight', 0) > 0:
    total_loss, class_loss, cont_loss = measure_loss_tso_with_continuity(
        outputs, pad_Y, x_lens,
        continuity_weight=best_params['continuity_weight']
    )

    # Optional: Log individual losses for monitoring
    if verbose and batches % 100 == 0:
        print(f"  Class loss: {class_loss:.4f}, Continuity loss: {cont_loss:.4f}")
else:
    # Fallback to standard loss (backward compatible)
    total_loss = measure_loss_tso(outputs, pad_Y, x_lens)

# Rest of training code remains the same
total_loss.backward()
optimizer.step()
...
```

#### **Option B: Quick A/B test without modifying code**

Just change one line:
```python
# Change this:
total_loss = measure_loss_tso(outputs, pad_Y, x_lens)

# To this:
total_loss, _, _ = measure_loss_tso_with_continuity(outputs, pad_Y, x_lens, continuity_weight=0.1)
```

---

### **Phase 2: Inference with Post-Processing**

#### **During Validation/Testing**

Add post-processing after getting predictions:

```python
# In evaluation loop
model.eval()
with torch.no_grad():
    for batch in dataloader:
        # Get model predictions
        outputs = model(pad_X, x_lens)  # [batch, seq_len, 3]

        # Get predicted classes
        predictions = torch.argmax(outputs, dim=-1)  # [batch, seq_len]

        # ✓ NEW: Post-process to enforce single TSO period
        predictions = batch_enforce_single_tso(
            predictions,
            x_lens,
            min_gap_minutes=30,      # Merge TSO segments within 30 min
            min_duration_minutes=10  # Filter out TSO < 10 min (noise)
        )

        # Calculate metrics with processed predictions
        # ... rest of evaluation code
```

#### **Example: Complete Evaluation Function**

```python
def evaluate_with_single_tso(model, dataset, device, enforce_single_period=True):
    """
    Evaluate model with optional single TSO period enforcement.

    Args:
        enforce_single_period: If True, post-process to enforce single TSO
    """
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_indices in batch_generator_h5(dataset, batch_size=16):
            # Prepare batch
            pad_X, pad_Y, x_lens, _ = add_padding_tso_patch_h5(
                dataset, batch_indices, device,
                num_channels=dataset.num_channels
            )

            # Forward pass
            outputs = model(pad_X, x_lens)
            predictions = torch.argmax(outputs, dim=-1)

            # ✓ Post-process if enabled
            if enforce_single_period:
                predictions = batch_enforce_single_tso(
                    predictions, x_lens,
                    min_gap_minutes=30,
                    min_duration_minutes=10
                )

            # Collect predictions and labels
            for i in range(len(x_lens)):
                valid_len = int(x_lens[i])
                all_predictions.extend(predictions[i, :valid_len].cpu().numpy())
                all_labels.extend(pad_Y[i, :valid_len].cpu().numpy())

    # Calculate F1 scores
    from sklearn.metrics import f1_score, classification_report

    f1_micro = f1_score(all_labels, all_predictions, average='micro')
    f1_macro = f1_score(all_labels, all_predictions, average='macro')
    f1_per_class = f1_score(all_labels, all_predictions, average=None)

    print(f"\nF1 Scores:")
    print(f"  Micro: {f1_micro:.4f}")
    print(f"  Macro: {f1_macro:.4f}")
    print(f"  Per-class: {f1_per_class}")
    print(f"  TSO (class 2): {f1_per_class[2]:.4f}")

    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_predictions,
                                target_names=['Other', 'Non-wear', 'PredictTSO']))

    return f1_per_class[2]  # Return TSO F1 score
```

---

## 🔬 Experimental Setup

### **Recommended Experiment Progression:**

#### **Experiment 1: Baseline (No changes)**
```python
continuity_weight = 0.0
enforce_single_period = False
```
**Purpose**: Establish baseline performance

#### **Experiment 2: Post-processing only**
```python
continuity_weight = 0.0  # No training changes
enforce_single_period = True
```
**Purpose**: Test if post-processing alone improves metrics

#### **Experiment 3: Continuity loss only**
```python
continuity_weight = 0.1  # Add during training
enforce_single_period = False
```
**Purpose**: Test if training guidance improves raw predictions

#### **Experiment 4: Hybrid (Full approach)**
```python
continuity_weight = 0.1
enforce_single_period = True
```
**Purpose**: Test combined effect (expected best performance)

#### **Experiment 5: Hyperparameter tuning**
```python
# Try different weights
for weight in [0.05, 0.1, 0.2, 0.5]:
    continuity_weight = weight
    enforce_single_period = True
    # Train and evaluate...
```

---

## 📊 Expected Results

### **Baseline (no changes)**
```
TSO F1: 0.65
Issues: Multiple fragmented TSO predictions per night
```

### **With post-processing**
```
TSO F1: 0.68-0.72 (+3-7% improvement)
Benefit: Guaranteed single period, cleaner predictions
```

### **With continuity loss**
```
TSO F1: 0.69-0.73 (+4-8% improvement)
Benefit: Model learns to predict continuous periods
```

### **Hybrid (both)**
```
TSO F1: 0.72-0.77 (+7-12% improvement)
Benefit: Best of both - training guidance + guaranteed enforcement
```

---

## ⚙️ Configuration Parameters

### **Continuity Loss Parameters**

```python
continuity_weight: float (default: 0.1)
    - 0.0: Disabled (standard training)
    - 0.05-0.1: Light regularization (recommended start)
    - 0.2-0.5: Strong regularization (if fragmentation is severe)
    - > 0.5: May hurt performance (too restrictive)
```

**How it works**:
- Measures "jumpiness" in TSO probability over time
- Penalizes: `0 → 1 → 0 → 1 → 0` (fragmented)
- Rewards: `0 → 0.3 → 0.8 → 1.0 → 0.8 → 0.3 → 0` (smooth)

### **Post-Processing Parameters**

```python
min_gap_minutes: int (default: 30)
    - TSO segments within this gap are merged
    - Example: TSO at 22:00-23:00 and 23:20-01:00 → merged if gap=30
    - Typical range: 20-60 minutes

min_duration_minutes: int (default: 10)
    - TSO segments shorter than this are filtered as noise
    - Example: 5-minute TSO burst → removed if min_duration=10
    - Typical range: 5-20 minutes
```

---

## 🐛 Troubleshooting

### **Issue: Training loss increases with continuity loss**
**Solution**: Reduce `continuity_weight` (try 0.05 or 0.02)

### **Issue: Model predicts no TSO periods**
**Solution**:
- Check `min_duration_minutes` isn't too high
- Reduce `continuity_weight` during training

### **Issue: Still seeing multiple TSO periods**
**Solution**:
- Ensure `enforce_single_period=True` in evaluation
- Check `min_gap_minutes` - increase to merge more aggressively

### **Issue: TSO F1 score decreased**
**Possible causes**:
1. `continuity_weight` too high → reduce it
2. `min_duration_minutes` filtering out real TSO → lower it
3. Labels are genuinely multi-period → post-processing may harm
4. Need more training epochs for continuity loss to take effect

---

## 📈 Monitoring Training

Track these metrics during training:

```python
# After each epoch
print(f"Epoch {epoch}:")
print(f"  Total loss: {total_loss:.4f}")
print(f"  Classification loss: {class_loss:.4f}")
print(f"  Continuity loss: {cont_loss:.4f}")

# Ratio should be reasonable (continuity shouldn't dominate)
# Good: class_loss=0.5, cont_loss=0.05 (10% of class loss)
# Bad: class_loss=0.5, cont_loss=0.8 (too high - reduce weight)
```

---

## 🎬 Quick Start Example

**Minimal code change to try hybrid approach:**

```python
# 1. Update imports
from Helpers.DL_helpers import measure_loss_tso_with_continuity, batch_enforce_single_tso

# 2. In training loop
total_loss, class_loss, cont_loss = measure_loss_tso_with_continuity(
    outputs, pad_Y, x_lens, continuity_weight=0.1
)

# 3. In evaluation loop
predictions = torch.argmax(outputs, dim=-1)
predictions = batch_enforce_single_tso(predictions, x_lens)

# That's it! Compare F1 scores before/after
```

---

## 📚 Function Reference

### `tso_continuity_loss(outputs, x_lengths, alpha=0.1)`
Calculates continuity penalty for fragmented predictions.

### `measure_loss_tso_with_continuity(outputs, labels, x_lengths, continuity_weight=0.1)`
Combined classification + continuity loss for training.
Returns: `(total_loss, class_loss, cont_loss)`

### `enforce_single_tso_period(predictions, min_gap_minutes=30, min_duration_minutes=10)`
Post-processes predictions to keep only longest TSO period.
Works on single sequences or batches.

### `batch_enforce_single_tso(predictions, x_lengths, min_gap_minutes=30, min_duration_minutes=10)`
Batch version with sequence length handling.
Returns tensor with same shape as input.

---

## 🚀 Next Steps

1. ✅ **Test post-processing first** (no retraining needed)
   ```bash
   python predict_TSO_segment_patch_h5.py --h5_file data.h5 --test_only
   # Add enforce_single_tso in evaluation code
   ```

2. ✅ **Retrain with continuity loss**
   ```bash
   python predict_TSO_segment_patch_h5.py --h5_file data.h5 --epochs 50
   # With continuity_weight=0.1 in best_params
   ```

3. ✅ **Compare F1 scores** across all 4 experiments

4. ✅ **Tune hyperparameters** if needed

---

## 💡 Alternative: If Hybrid Doesn't Work

If the hybrid approach doesn't improve performance, consider:

1. **Labels might be genuinely multi-period** - some nights have 2+ TSO periods
2. **Try BIO tagging** (Begin-Inside-Outside) for more principled sequence labeling
3. **Add regression branch** (your original idea) as a last resort
4. **Analyze failure cases** - visualize predictions vs ground truth

Good luck! 🎉
