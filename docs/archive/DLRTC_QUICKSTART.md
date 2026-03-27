# DLR-TC Quick Start Guide

**5-Minute Setup | Get Running Immediately**

---

## Step 1: Validate Installation (1 minute)

```bash
cd /Users/zongfan/Documents/MY/JNJ
python test_dlrtc_implementation.py
```

**Expected Output**:
```
============================================================
ALL TESTS PASSED ✓
============================================================
```

If all tests pass, you're ready to train! If not, check error messages.

---

## Step 2: Run Your First DLR-TC Training (1-2 hours)

### Option A: Use Your Existing Data Path

```bash
python predict_TSO_segment_patch_dlrtc.py \
    --input_data_folder /mnt/data/GENEActive-featurized/UKB_v2/raw \
    --output TSO_dlrtc_first_run \
    --model mba4tso_patch \
    --epochs 20 \
    --num_gpu 0 \
    --testing LOFO
```

### Option B: Quick Test on Small Subset

For faster testing, create a small data subset first:
```bash
# (Assuming you have a way to create a small test subset)
python predict_TSO_segment_patch_dlrtc.py \
    --input_data_folder /path/to/small_test_data/raw \
    --output TSO_dlrtc_test \
    --model mba4tso_patch \
    --epochs 10 \
    --num_gpu 0 \
    --testing LOFO
```

---

## Step 3: Monitor Training

Watch the console output for these phases:

### Phase 1: Warmup (First 5 epochs)
```
============================================================
PHASE 1: WARMUP TRAINING (5 epochs)
============================================================
Purpose: Learn easy patterns with standard Cross Entropy

[Warmup] Epoch 1/5
  Train - Loss: 0.6234, Acc: 0.6512, F1 avg: 0.5834
  Val   - Loss: 0.5891, Acc: 0.6723, F1 avg: 0.6012
```

**What to expect**: Standard supervised learning, F1 around 0.55-0.65

### Phase 2: Joint Refinement (Next 15 epochs)
```
============================================================
PHASE 2: JOINT LABEL-MODEL REFINEMENT (15 epochs)
============================================================
Purpose: Refine noisy labels while training model

[Joint] Epoch 1/15
  Train - Loss: 0.5234 (GCE: 0.4523, Compat: 0.0234, Temp: 0.0477)
          Acc: 0.6834, F1 avg: 0.6245
  Val   - Loss: 0.5123, Acc: 0.6945, F1 avg: 0.6312
```

**What to expect**: F1 improves to 0.65-0.75, surpassing baseline around epoch 5-10

---

## Step 4: Check Results

After training completes, you'll find:

### Output Folder Structure
```
/mnt/data/GENEActive-featurized/results/DL/UKB_v2/TSO_dlrtc_first_run/training/
├── model_weights/
│   ├── best_model_split_0_iter_0.pt           # Best model checkpoint
│   └── soft_labels_split_0_iter_0.pt          # Refined labels
├── predictions/
│   └── dlrtc_results_split_0_iter_0.joblib    # Full results
├── learning_plots/
│   └── dlrtc_training_history_split_0_iter_0.png  # Training curves
└── training_logs/
```

### Quick Performance Check

```python
import joblib

# Load results
results = joblib.load('results/dlrtc_results_split_0_iter_0.joblib')

# Check test performance
test = results['test_metrics']
print(f"Test Accuracy: {test['accuracy']:.4f}")
print(f"Test F1 (avg): {test['f1_avg']:.4f}")
print(f"Test F1 (TSO): {test['f1_tso']:.4f}")

# Check label refinement
print(f"\nRefined {results['num_refined_labels']} segment labels")
```

---

## Step 5: Compare with Baseline

### Run Baseline Training

```bash
python predict_TSO_segment_patch.py \
    --input_data_folder /mnt/data/GENEActive-featurized/UKB_v2/raw \
    --output TSO_baseline_comparison \
    --model mba4tso_patch \
    --epochs 20 \
    --num_gpu 0 \
    --testing LOFO
```

### Compare Results

```python
import joblib

# Load both results
dlrtc_results = joblib.load('results/dlrtc_results_split_0_iter_0.joblib')
baseline_results = joblib.load('results/baseline_results_split_0_iter_0.joblib')

# Compare
dlrtc_f1 = dlrtc_results['test_metrics']['f1_avg']
baseline_f1 = baseline_results['test_metrics']['f1_avg']

improvement = (dlrtc_f1 - baseline_f1) / baseline_f1 * 100

print(f"DLR-TC F1:  {dlrtc_f1:.4f}")
print(f"Baseline F1: {baseline_f1:.4f}")
print(f"Improvement: {improvement:+.2f}%")
```

**Expected**: +3-7% F1 improvement

---

## Tuning Guide (If Needed)

### Problem: Labels Not Changing

**Symptom**: `loss_compat` near zero

**Solution**: Edit `predict_TSO_segment_patch_dlrtc.py` around line 530:
```python
dlrtc_config = {
    'warmup_epochs': 5,
    'joint_epochs': 15,
    'gce_q': 0.7,
    'alpha': 0.05,        # DECREASE (was 0.1)
    'beta': 0.5,
    'label_lr': 0.05,     # INCREASE (was 0.01)
}
```

### Problem: Predictions Too Smooth

**Symptom**: Missing short TSO events

**Solution**:
```python
dlrtc_config = {
    'warmup_epochs': 5,
    'joint_epochs': 15,
    'gce_q': 0.7,
    'alpha': 0.1,
    'beta': 0.2,          # DECREASE (was 0.5)
    'label_lr': 0.01,
}
```

### Problem: Predictions Too Jittery

**Symptom**: Rapid oscillations between classes

**Solution**:
```python
dlrtc_config = {
    'warmup_epochs': 5,
    'joint_epochs': 15,
    'gce_q': 0.7,
    'alpha': 0.1,
    'beta': 0.8,          # INCREASE (was 0.5)
    'label_lr': 0.01,
}
```

### Problem: Validation Performance Drops

**Symptom**: Val F1 decreases during joint phase

**Solution**:
```python
dlrtc_config = {
    'warmup_epochs': 5,
    'joint_epochs': 15,
    'gce_q': 0.7,
    'alpha': 0.3,         # INCREASE (was 0.1)
    'beta': 0.5,
    'label_lr': 0.005,    # DECREASE (was 0.01)
}
```

---

## Understanding the Output

### Training Console Output

**Good Progress Indicators**:
```
[Joint] Epoch 10/15
  Train - Loss: 0.4523 (GCE: 0.3891, Compat: 0.0189, Temp: 0.0443)
          Acc: 0.7245, F1 avg: 0.6834
  Val   - Loss: 0.4712, Acc: 0.7123, F1 avg: 0.6723
  -> Saved best model
```

✓ GCE loss decreasing (model learning patterns)
✓ Compat loss stable and small (labels anchored)
✓ Temp loss decreasing (predictions smoothing)
✓ Validation F1 improving

**Warning Indicators**:
```
[Joint] Epoch 10/15
  Train - Loss: 0.5234 (GCE: 0.4123, Compat: 0.0823, Temp: 0.0288)
          Acc: 0.6845, F1 avg: 0.6245
  Val   - Loss: 0.5891, Acc: 0.6512, F1 avg: 0.5923
```

⚠ Compat loss increasing (labels drifting) → increase `alpha`
⚠ Val F1 decreasing → reduce `label_lr` or increase `alpha`

### Loss Component Interpretation

| Component | Range | Meaning |
|-----------|-------|---------|
| GCE | 0.3-0.6 | Prediction error (lower = better) |
| Compat | 0.01-0.1 | Label drift (too low = not refining, too high = drifting) |
| Temp | 0.02-0.1 | Temporal jitter (lower = smoother) |

---

## Advanced: Inspect Refined Labels

```python
import torch
import matplotlib.pyplot as plt
from Helpers.dlrtc_losses import SoftLabelManager

# Load soft labels
manager = SoftLabelManager(num_classes=3, device='cpu')
manager.load_state('model_weights/soft_labels_split_0_iter_0.pt')

print(f"Total segments refined: {manager.get_num_segments()}")

# Pick a segment to visualize
segment_id = list(manager.soft_labels_dict.keys())[0]
soft_labels = manager.soft_labels_dict[segment_id].detach().cpu().numpy()

# Plot
plt.figure(figsize=(14, 5))
plt.plot(soft_labels[:, 0], label='other', alpha=0.7, linewidth=2)
plt.plot(soft_labels[:, 1], label='non-wear', alpha=0.7, linewidth=2)
plt.plot(soft_labels[:, 2], label='predictTSO', alpha=0.7, linewidth=2)
plt.xlabel('Time (minutes)', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.title(f'Refined Soft Labels: {segment_id}', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('refined_labels_example.png', dpi=150)
print("Saved visualization to refined_labels_example.png")
```

---

## Checklist

- [ ] **Tested**: Ran `test_dlrtc_implementation.py` successfully
- [ ] **Trained**: Completed first DLR-TC training run
- [ ] **Baseline**: Trained baseline for comparison
- [ ] **Compared**: DLR-TC shows improvement over baseline
- [ ] **Inspected**: Checked refined labels and training curves
- [ ] **Tuned** (if needed): Adjusted hyperparameters based on results

---

## Common Questions

**Q: How long does training take?**
A: Similar to baseline (~1-2 hours for full dataset, 10-30 minutes for small subset)

**Q: Do I need to change my data?**
A: No! Uses the same data format as `predict_TSO_segment_patch.py`

**Q: Can I use with other models?**
A: Yes! Compatible with any model that outputs [batch, seq_len, num_classes]

**Q: What if I don't see improvement?**
A: Try tuning hyperparameters (see Tuning Guide above). Also check if your labels are actually noisy - clean labels won't benefit from refinement.

**Q: Can I resume training from checkpoint?**
A: Yes, but you'll need to load both model weights and soft labels. See full documentation in DLRTC_README.md

**Q: How do I know if labels were actually noisy?**
A: Check how many labels changed significantly:
```python
# Count significant changes
significant_changes = 0
for seg_id, soft in manager.soft_labels_dict.items():
    max_prob = soft.max(dim=-1)[0]
    if (max_prob < 0.9).sum() > len(soft) * 0.1:  # >10% timesteps uncertain
        significant_changes += 1
print(f"{significant_changes} segments with significant label refinement")
```

---

## Next Steps

1. **Read full documentation**: [DLRTC_README.md](DLRTC_README.md)
2. **Understand implementation**: [DLRTC_IMPLEMENTATION_SUMMARY.md](DLRTC_IMPLEMENTATION_SUMMARY.md)
3. **Experiment with hyperparameters**: See tuning guide in README
4. **Analyze label evolution**: Use visualization code above
5. **Run ablation studies**: Test individual components

---

**You're ready to go! Start with Step 1 and work through the checklist.**

If you encounter issues, check:
1. Test suite output (`test_dlrtc_implementation.py`)
2. Console training logs for warning indicators
3. Full documentation in `DLRTC_README.md`

Good luck with your experiments! 🚀
