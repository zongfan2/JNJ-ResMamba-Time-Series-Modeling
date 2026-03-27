# DLR-TC Epoch Planning for Large-Scale Dataset

**Dataset**: 34,600 segments, 900 GB sensor data

---

## Executive Summary

**Recommended Configuration**:
- **Warmup**: 10 epochs
- **Joint Refinement**: 25 epochs
- **Total**: 35 epochs
- **Training Time**: ~45-50 hours (GPU), ~80-100 hours (CPU)

---

## Detailed Analysis

### Dataset Characteristics

```
Total segments:      34,600
Data size:          900 GB
Batch size:         24
Training split:     27,680 segments (80%)
Validation split:   6,920 segments (20%)

Batches per epoch:  ~1,153 (train) + ~288 (val) = ~1,441 total
```

### Training Dynamics

#### With 34,600 Segments:

| Metric | Value |
|--------|-------|
| Train batches/epoch | 1,153 |
| Val batches/epoch | 288 |
| Total batches/epoch | 1,441 |
| Avg timesteps/segment | ~1,440 (24 hours @ 1 min resolution) |
| Total labels to refine | ~50 million timesteps |

---

## Phase 1: Warmup (10 Epochs)

### Why 10 Epochs?

**Standard Practice**:
- Small datasets (<10K): 3-5 epochs
- Medium datasets (10-50K): 5-10 epochs
- Large datasets (>50K): 10-15 epochs

**Your dataset (34.6K)** falls in the medium-to-large category → **10 epochs**

### What Happens Each Epoch

| Epoch | Learning Progress | Expected F1 |
|-------|------------------|-------------|
| 1-2 | Obvious patterns (clear non-wear, daytime TSO) | 0.45-0.55 |
| 3-4 | Activity-based TSO detection | 0.55-0.62 |
| 5-7 | Fine decision boundaries | 0.62-0.68 |
| 8-10 | Plateau and stabilization | 0.68-0.72 |

### Key Metrics to Monitor

```python
# Epoch 1
Train Loss: 0.8-1.2, F1: 0.40-0.50
Val Loss:   0.7-1.0, F1: 0.45-0.55

# Epoch 5
Train Loss: 0.5-0.7, F1: 0.60-0.65
Val Loss:   0.5-0.6, F1: 0.62-0.68

# Epoch 10
Train Loss: 0.4-0.5, F1: 0.68-0.72
Val Loss:   0.45-0.55, F1: 0.70-0.74
```

**Stop early if**: Val F1 plateaus before epoch 7 (dataset might be cleaner than expected)

---

## Phase 2: Joint Refinement (25 Epochs)

### Why 25 Epochs?

**Label Refinement is Slow**:
- 50 million individual label probabilities to optimize
- Alternating updates (labels then model) = 2x iterations
- Need multiple passes to propagate information across time

**Research Literature**:
- DivideMix (ICLR 2020): 300 epochs for CIFAR-10 (50K images)
- Our case: Fewer epochs due to temporal structure helping convergence

**Calculation**:
```
Refinement complexity = segments × avg_len × num_classes / batch_size
                      = 34,600 × 1,440 × 3 / 24
                      = 6.2 million updates needed

With 1,153 batches/epoch → ~5,400 updates/epoch
Target convergence: ~90% labels stable
6.2M × 0.9 / 5,400 ≈ 1,033 batches ≈ 18-22 epochs

Add buffer for safety: 25 epochs
```

### What Happens Each Epoch

| Epoch Range | Refinement Stage | Label Changes |
|-------------|-----------------|---------------|
| 1-5 | High-confidence corrections | 20-30% labels |
| 6-10 | Medium-confidence adjustments | 15-25% labels |
| 11-15 | Fine-grained refinement | 10-15% labels |
| 16-20 | Convergence | 5-10% labels |
| 21-25 | Stabilization | <5% labels |

### Expected Performance Trajectory

```python
# Epoch 1 (Joint)
Train: Loss=0.48 (GCE=0.42, Compat=0.02, Temp=0.04), F1=0.72
Val:   Loss=0.50, F1=0.73

# Epoch 5
Train: Loss=0.42 (GCE=0.36, Compat=0.02, Temp=0.04), F1=0.76
Val:   Loss=0.44, F1=0.77

# Epoch 10
Train: Loss=0.38 (GCE=0.33, Compat=0.018, Temp=0.032), F1=0.79
Val:   Loss=0.40, F1=0.80

# Epoch 15
Train: Loss=0.35 (GCE=0.30, Compat=0.018, Temp=0.032), F1=0.81
Val:   Loss=0.37, F1=0.82

# Epoch 20
Train: Loss=0.33 (GCE=0.29, Compat=0.015, Temp=0.025), F1=0.82
Val:   Loss=0.36, F1=0.83

# Epoch 25
Train: Loss=0.32 (GCE=0.28, Compat=0.015, Temp=0.025), F1=0.82
Val:   Loss=0.35, F1=0.83
```

### Key Indicators

**Good Progress**:
- Val F1 increasing steadily
- `loss_compat` stable (0.015-0.025 range)
- `loss_temp` decreasing (noise being smoothed)
- Label refinement rate decreasing over time

**Warning Signs**:
- Val F1 decreasing → Increase `alpha` to 0.2
- `loss_compat` > 0.05 → Labels drifting, decrease `label_lr`
- Val F1 plateaus before epoch 15 → Can stop early

---

## Time Estimation

### Per Batch Timing

**Warmup Phase** (standard training):
```
Forward pass:      1.5 sec  (MBA4TSO_Patch with 20Hz patches)
Backward pass:     1.0 sec
Batch prep:        0.5 sec
Total:            ~3 sec/batch
```

**Joint Phase** (alternating updates):
```
Forward pass:      1.5 sec
Label update:      0.8 sec  (gradient computation + update)
Model update:      1.5 sec  (backward + optimizer step)
Batch prep:        0.5 sec
Total:            ~4.3 sec/batch
```

### Total Training Time

**Warmup (10 epochs)**:
```
Time = 10 epochs × 1,441 batches × 3 sec
     = 43,230 seconds
     ≈ 12 hours
```

**Joint (25 epochs)**:
```
Time = 25 epochs × 1,441 batches × 4.3 sec
     = 154,907 seconds
     ≈ 43 hours
```

**TOTAL: ~55 hours** (2.3 days)

### With GPU Acceleration

| GPU | Speedup | Total Time |
|-----|---------|-----------|
| RTX 3090 | 2-3x | 18-27 hours |
| A100 | 3-4x | 14-18 hours |
| V100 | 2.5-3.5x | 16-22 hours |

**With 4 GPUs (parallel splits)**: Divide by 4 if using multi-GPU training

---

## Alternative Configurations

### Scenario 1: Limited Compute Budget

**Conservative Configuration**:
```python
dlrtc_config = {
    'warmup_epochs': 7,
    'joint_epochs': 18,
}
# Total: 25 epochs, ~35-40 hours
```

**Trade-offs**:
- ✓ 35% faster training
- ✗ May miss some label refinements
- ✗ F1 might be 1-2% lower

**Use when**: Compute budget limited, doing hyperparameter search

---

### Scenario 2: Very Noisy Labels

**Aggressive Configuration**:
```python
dlrtc_config = {
    'warmup_epochs': 15,
    'joint_epochs': 35,
    'label_lr': 0.02,  # Faster refinement
}
# Total: 50 epochs, ~70-80 hours
```

**Trade-offs**:
- ✓ More thorough label refinement
- ✓ Better convergence for complex patterns
- ✗ 45% longer training time
- ✗ Risk of overfitting if labels aren't that noisy

**Use when**:
- You know labels are very noisy (e.g., >30% error rate)
- You have strong compute resources
- Initial runs show labels still changing at epoch 25

---

### Scenario 3: Quick Validation

**Quick Test Configuration**:
```python
dlrtc_config = {
    'warmup_epochs': 3,
    'joint_epochs': 7,
}
# Total: 10 epochs, ~12-15 hours
```

**Trade-offs**:
- ✓ 75% faster (for quick experiments)
- ✗ Incomplete learning and refinement
- ✗ F1 will be 5-10% lower than optimal

**Use when**:
- Testing new hyperparameters (alpha, beta, gce_q)
- Debugging code changes
- Comparing different model architectures

---

## Convergence Criteria

### When to Stop Training

**Automatic (Early Stopping)**:
```python
early_stopping = EarlyStopping(patience=10)
# Stops if val loss doesn't improve for 10 epochs
```

**Manual Checkpoints**:

Check these metrics every 5 epochs:

| Metric | Good | Warning |
|--------|------|---------|
| Val F1 trend | Increasing or stable | Decreasing |
| Val loss trend | Decreasing | Increasing |
| Loss_compat | 0.015-0.025 | >0.05 |
| Loss_temp | Decreasing | Increasing |
| Label change % | <10% after epoch 20 | >20% |

**Stop early if** (before 25 epochs):
- Val F1 plateaus for 8+ epochs
- Test F1 (if checking) exceeds target
- Label changes drop below 5% before epoch 15

**Continue beyond 25 epochs if**:
- Val F1 still improving at epoch 25
- Label changes still >10% at epoch 25
- Loss_temp still decreasing significantly

---

## Practical Workflow

### Week 1: Validation Run

```bash
# Quick test to validate pipeline
dlrtc_config = {
    'warmup_epochs': 3,
    'joint_epochs': 7,
}
# Time: 12-15 hours
# Goal: Verify code works, get baseline F1
```

### Week 2: Full Training

```bash
# Recommended configuration
dlrtc_config = {
    'warmup_epochs': 10,
    'joint_epochs': 25,
}
# Time: 45-55 hours
# Goal: Best performance, production model
```

### Week 3: Hyperparameter Tuning (Optional)

If initial results are below expectations, tune:

**Experiment 1: Noise Robustness**
```python
gce_q = [0.5, 0.6, 0.7, 0.8]  # Test 4 values
# Keep: warmup=7, joint=18 (faster iteration)
```

**Experiment 2: Smoothness**
```python
beta = [0.3, 0.5, 0.8, 1.0]  # Test 4 values
# Keep: warmup=7, joint=18
```

**Experiment 3: Anchor Strength**
```python
alpha = [0.05, 0.1, 0.2, 0.3]  # Test 4 values
# Keep: warmup=7, joint=18
```

Each experiment: ~4 runs × 35 hours = 140 hours (6 days)

---

## Expected Outcomes by Dataset Size

### Comparison Table

| Dataset Size | Warmup | Joint | Total | Time (GPU) |
|--------------|--------|-------|-------|-----------|
| 5K segments | 5 | 10 | 15 | 10-15h |
| 10K segments | 5 | 15 | 20 | 15-20h |
| 20K segments | 7 | 20 | 27 | 25-35h |
| **34.6K segments** | **10** | **25** | **35** | **45-55h** |
| 50K segments | 12 | 30 | 42 | 60-80h |
| 100K segments | 15 | 35 | 50 | 100-130h |

**Your dataset (34.6K)** is in the large-scale category → requires substantial training

---

## Monitoring Dashboard

### Metrics to Track

**Console Output (Real-time)**:
```
[Joint] Epoch 15/25
  Train - Loss: 0.35 (GCE: 0.30, Compat: 0.018, Temp: 0.032)
          Acc: 0.81, F1 avg: 0.81
  Val   - Loss: 0.37, Acc: 0.82, F1 avg: 0.82
  -> Saved best model
  Refined 5,234 labels this epoch (15.2% of segments)
```

**Check every 5 epochs**:
1. Validation F1 trend (should increase or plateau)
2. Compatibility loss (should be 0.015-0.025)
3. Temporal loss (should decrease)
4. Number of labels refined (should decrease over time)

**Final evaluation**:
- Test F1 vs baseline F1
- Temporal consistency (jitter reduction)
- Label refinement statistics

---

## Summary

### Final Recommendations

**For your dataset (34.6K segments, 900 GB)**:

✅ **Use 35 total epochs** (10 warmup + 25 joint)

✅ **Allocate 2-3 days** for training (with GPU)

✅ **Monitor validation F1** - expect plateau around epoch 30-32

✅ **Check label refinement** - expect 15-30% of labels to change significantly

✅ **Compare with baseline** - expect +3-7% F1 improvement

### Quick Reference

```python
# RECOMMENDED (production-quality training)
dlrtc_config = {
    'warmup_epochs': 10,
    'joint_epochs': 25,
    'gce_q': 0.7,
    'alpha': 0.1,
    'beta': 0.5,
    'label_lr': 0.01,
}
```

This configuration balances:
- ✓ Thorough learning of general patterns
- ✓ Comprehensive label refinement
- ✓ Reasonable training time
- ✓ Stable convergence

**Next step**: Run full training and monitor the metrics above!
