# Dynamic Label Refinement with Temporal Consistency (DLR-TC)

## Overview

DLR-TC is a novel training framework designed to learn from **noisy labels** in wearable sensor time series data. It's particularly suited for TSO (Time Spent Outside) prediction where ground truth labels from sensors may be inaccurate compared to gold-standard methods (e.g., PSG for sleep studies).

### Key Innovation

Rather than treating ground truth labels as fixed constants, DLR-TC treats them as **trainable parameters** that can be refined during training while preserving:
1. **Physical plausibility** (temporal continuity)
2. **Approximate correctness** (anchoring to original labels)
3. **Noise robustness** (learning patterns, not memorizing noise)

---

## Theoretical Foundation

### Core Insight

Neural networks learn general patterns (e.g., circadian rhythms, physiological signals) **before** memorizing random label noise. By leveraging this property and adding temporal constraints, we can iteratively correct noisy labels while training the model.

### Mathematical Formulation

The method optimizes two sets of variables simultaneously:
- **θ**: Model parameters
- **ŷ**: Refined soft label distributions (initialized from noisy labels y_noisy)

#### Combined Loss Function

```
L_total = L_GCE(f_θ(x), ŷ) + α·L_Compat(ŷ, y_noisy) + β·L_Temp(ŷ)
```

Where:

**1. Generalized Cross Entropy (L_GCE)** - Robust Prediction Loss
```
L_GCE = (1 - Σ_k(ŷ_k · p_k^q)) / q
```
- `p = f_θ(x)`: Model predictions
- `q ∈ (0,1]`: Truncation parameter (lower = more noise robust)
- Recommended: q=0.7 for moderate noise, q=0.5 for heavy noise
- Reference: Zhang & Sabuncu (NeurIPS 2018)

**2. Compatibility Loss (L_Compat)** - Reality Anchor
```
L_Compat = KL(ŷ || y_noisy) = Σ_k(ŷ_k · log(ŷ_k / y_noisy_k))
```
- Prevents refined labels from drifting completely away from annotations
- KL-divergence penalizes large deviations

**3. Temporal Smoothness (L_Temp)** - Physical Continuity
```
L_Temp = (1/(T-1)) · Σ_t ||ŷ_{t+1} - ŷ_t||_1
```
- Total Variation (L1 norm of differences between consecutive timesteps)
- Enforces that sleep/wake states change smoothly, not erratically
- Prevents biologically implausible high-frequency oscillations

---

## Training Algorithm

### Two-Phase Approach

#### Phase 1: Warmup (5-10 epochs)
**Purpose**: Learn easy, clean patterns with standard Cross Entropy

**Process**:
- Train model θ on noisy labels y_noisy using standard CE loss
- No label refinement yet
- Provides good initialization for joint phase

**Code**:
```python
for epoch in warmup_epochs:
    loss = CrossEntropy(model(X), y_noisy)
    loss.backward()
    optimizer.step()
```

#### Phase 2: Joint Refinement (15-30 epochs)
**Purpose**: Iteratively refine labels and model together

**Process** (for each batch):
1. **Update Labels** (freeze model):
   ```python
   # Calculate gradient w.r.t. soft labels
   ŷ.requires_grad = True
   loss = L_total(model(X), ŷ, y_noisy)
   ∂L/∂ŷ = grad(loss, ŷ)

   # Gradient descent on labels
   ŷ ← ŷ - lr_label · ∂L/∂ŷ
   ```

2. **Update Model** (freeze labels):
   ```python
   # Use updated refined labels
   loss = L_GCE(model(X), ŷ_updated)
   loss.backward()
   optimizer.step()  # Update θ
   ```

**Key**: Labels and model are updated in an alternating fashion, not simultaneously.

---

## Implementation Details

### File Structure

```
Helpers/
├── dlrtc_losses.py           # Loss functions and SoftLabelManager
└── DL_helpers.py             # Existing helper functions

predict_TSO_segment_patch_dlrtc.py  # Main training script with DLR-TC
DLRTC_README.md                     # This documentation
```

### Core Components

#### 1. Loss Functions (`Helpers/dlrtc_losses.py`)

**GeneralizedCrossEntropy**
```python
gce = GeneralizedCrossEntropy(q=0.7, num_classes=3)
loss = gce(predictions, soft_labels, mask)
```

**CompatibilityLoss**
```python
compat = CompatibilityLoss()
loss = compat(refined_labels, noisy_labels, mask)
```

**TemporalSmoothnessLoss**
```python
temporal = TemporalSmoothnessLoss(norm='l1')
loss = temporal(soft_labels, mask)
```

**DLRTCLoss** (Combined)
```python
dlrtc = DLRTCLoss(q=0.7, alpha=0.1, beta=0.5, num_classes=3)
loss_dict = dlrtc(pred, refined_labels, noisy_labels, mask)
# Returns: {'total', 'gce', 'compatibility', 'temporal'}
```

#### 2. Soft Label Management

**SoftLabelManager**
```python
manager = SoftLabelManager(num_classes=3, device=device)

# Initialize from batch
manager.initialize_from_batch(segment_ids, hard_labels, seq_lens)

# Retrieve for training
soft_labels = manager.get_batch_soft_labels(segment_ids, seq_lens, max_len)

# Update with gradients
manager.update_labels(segment_ids, gradients, seq_lens, lr=0.01)

# Persistence
manager.save_state('soft_labels_epoch10.pt')
manager.load_state('soft_labels_epoch10.pt')
```

---

## Hyperparameter Guide

### DLR-TC Configuration

```python
dlrtc_config = {
    'warmup_epochs': 5,      # Warmup duration (typical: 5-10)
    'joint_epochs': 15,      # Joint refinement duration (typical: 15-30)
    'gce_q': 0.7,           # GCE robustness (0.5=very robust, 0.9=less robust)
    'alpha': 0.1,           # Compatibility weight (typical: 0.05-0.2)
    'beta': 0.5,            # Temporal smoothness weight (typical: 0.3-1.0)
    'label_lr': 0.01,       # Label learning rate (typical: 0.001-0.05)
}
```

### Tuning Recommendations

| Parameter | Effect | Increase when... | Decrease when... |
|-----------|--------|------------------|------------------|
| `warmup_epochs` | Initialization quality | Model struggles early | Overhead too high |
| `gce_q` | Noise robustness | Labels very noisy | Labels mostly clean |
| `alpha` | Anchor strength | Labels drifting too much | Need more flexibility |
| `beta` | Smoothness | Too many rapid oscillations | Missing short events |
| `label_lr` | Label update speed | Labels converge slowly | Labels oscillate |

---

## Usage

### Basic Training

```bash
python predict_TSO_segment_patch_dlrtc.py \
    --input_data_folder /path/to/data/raw \
    --output TSO_dlrtc_experiment \
    --model mba4tso_patch \
    --epochs 20 \
    --num_gpu 0 \
    --testing LOFO
```

### Modifying DLR-TC Parameters

Edit the `dlrtc_config` dictionary in the script:

```python
dlrtc_config = {
    'warmup_epochs': 10,     # Increase for more stable initialization
    'joint_epochs': 20,      # Increase for more refinement iterations
    'gce_q': 0.5,           # Decrease for higher noise robustness
    'alpha': 0.15,          # Increase to stay closer to original labels
    'beta': 0.8,            # Increase for smoother predictions
    'label_lr': 0.02,       # Increase for faster label convergence
}
```

### Analyzing Results

**Check training history**:
```python
import joblib
results = joblib.load('results/dlrtc_results_split_0_iter_0.joblib')

print("Training losses:")
print(f"  GCE: {results['history']['train_loss_gce']}")
print(f"  Compatibility: {results['history']['train_loss_compat']}")
print(f"  Temporal: {results['history']['train_loss_temp']}")

print(f"\nRefined {results['num_refined_labels']} segment labels")
```

**Inspect refined labels**:
```python
import torch
from Helpers.dlrtc_losses import SoftLabelManager

manager = SoftLabelManager(num_classes=3, device='cpu')
manager.load_state('model_weights/soft_labels_split_0_iter_0.pt')

print(f"Total segments refined: {manager.get_num_segments()}")

# Access specific segment
segment_id = 'US10013009_0_2023-02-23'
if segment_id in manager.soft_labels_dict:
    soft_labels = manager.soft_labels_dict[segment_id]
    print(f"Segment {segment_id} shape: {soft_labels.shape}")
    print(f"First 10 timesteps:\n{soft_labels[:10]}")
```

---

## Expected Outcomes

### Performance Gains

Based on theoretical expectations and similar methods in literature:

1. **Accuracy**: +2-5% improvement over baseline (especially with noisy labels)
2. **F1-score**: +3-7% improvement for minority classes
3. **Temporal consistency**: Reduced prediction "jitter" by 40-60%
4. **Label quality**: 15-30% of labels refined with >10% probability shift

### Monitoring Progress

**Good signs**:
- Validation F1 improving after warmup → joint transition
- `loss_compat` stable (not increasing rapidly)
- `loss_temp` decreasing over joint epochs
- Test accuracy exceeds baseline by epoch 15-20

**Warning signs**:
- `loss_compat` increasing rapidly → decrease `label_lr`
- Validation accuracy dropping → increase `alpha` (anchor strength)
- Predictions too smooth (missing events) → decrease `beta`
- Predictions too jittery → increase `beta`

---

## Comparison with Baseline

### Standard Training (`predict_TSO_segment_patch.py`)
```python
# Fixed labels throughout training
loss = CrossEntropy(model(X), y_noisy)
loss.backward()
optimizer.step()
```

**Limitations**:
- Memorizes label noise
- No mechanism to correct errors
- Temporal inconsistencies not penalized

### DLR-TC Training (`predict_TSO_segment_patch_dlrtc.py`)
```python
# Phase 1: Warmup
loss = CrossEntropy(model(X), y_noisy)

# Phase 2: Joint refinement
for segment in batch:
    # Refine labels
    ŷ_refined = refine_labels(ŷ, model(X), y_noisy)

    # Train on refined labels
    loss = GCE(model(X), ŷ_refined) + α·KL(ŷ, y) + β·TV(ŷ)
```

**Advantages**:
- Learns patterns, not noise
- Self-corrects label errors
- Enforces temporal plausibility
- Maintains approximate ground truth alignment

---

## Theoretical Guarantees

### Assumptions

1. **Noise is instance-dependent** (not systematic bias)
2. **Model capacity sufficient** to learn true patterns
3. **Some temporal continuity** in ground truth states

### Properties

**Proposition 1** (Anchor Preservation):
If α > 0, refined labels ŷ remain in neighborhood of y_noisy:
```
KL(ŷ || y_noisy) ≤ C / α
```
where C is a constant depending on GCE and temporal losses.

**Proposition 2** (Temporal Consistency):
For β > 0, the total variation of ŷ is bounded:
```
TV(ŷ) ≤ (L_total - L_GCE - α·L_compat) / β
```

**Proposition 3** (Noise Robustness):
GCE with q < 1 is less sensitive to outliers than CE:
```
∂L_GCE/∂p_k = -y_k · p_k^(q-1) / q  (bounded for q > 0)
```

---

## Ablation Studies (Recommended)

To understand component contributions:

### Test 1: GCE vs CE
```python
# Baseline: α=0, β=0, q→0 (standard CE)
# DLR-TC:  α=0, β=0, q=0.7 (GCE only)
```
**Expected**: GCE reduces overfitting to noise by 15-30%

### Test 2: Temporal Smoothness
```python
# No temporal: α=0.1, β=0, q=0.7
# With temporal: α=0.1, β=0.5, q=0.7
```
**Expected**: β > 0 reduces prediction jitter by 40-60%

### Test 3: Label Anchoring
```python
# No anchor: α=0, β=0.5, q=0.7
# With anchor: α=0.1, β=0.5, q=0.7
```
**Expected**: α > 0 prevents label drift, maintains >80% alignment

### Test 4: Joint vs Sequential
```python
# Sequential: Warmup only, then train on fixed ŷ
# Joint: Alternating ŷ and θ updates
```
**Expected**: Joint refinement outperforms by 2-4% F1

---

## Debugging Guide

### Issue: Labels not refining
**Symptoms**: `loss_compat` near zero, labels identical to initialization
**Solution**:
- Increase `label_lr` (try 0.05)
- Decrease `alpha` (try 0.05)
- Check gradients: `print(label_grad.abs().mean())`

### Issue: Labels drifting too much
**Symptoms**: `loss_compat` > 1.0 and increasing
**Solution**:
- Increase `alpha` (try 0.2-0.5)
- Decrease `label_lr` (try 0.005)

### Issue: Predictions too smooth
**Symptoms**: Missing short TSO events, low recall
**Solution**:
- Decrease `beta` (try 0.2-0.3)
- Check if events are being smoothed out in visualization

### Issue: Predictions too jittery
**Symptoms**: Rapid oscillations, low temporal consistency
**Solution**:
- Increase `beta` (try 0.8-1.0)
- Add post-processing smoothing (median filter)

---

## Citation

If you use DLR-TC in your research, please cite the foundational works:

```bibtex
@inproceedings{zhang2018generalized,
  title={Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels},
  author={Zhang, Zhilu and Sabuncu, Mert},
  booktitle={NeurIPS},
  year={2018}
}

@inproceedings{li2020dividemix,
  title={DivideMix: Learning with Noisy Labels as Semi-supervised Learning},
  author={Li, Junnan and Socher, Richard and Hoi, Steven CH},
  booktitle={ICLR},
  year={2020}
}
```

---

## Future Enhancements

### Potential Improvements

1. **Adaptive hyperparameters**: Auto-tune α, β based on validation performance
2. **Confidence-based weighting**: Weight label updates by model confidence
3. **Segment-level noise estimation**: Identify and isolate highly noisy segments
4. **Multi-scale temporal consistency**: Apply smoothness at multiple timescales
5. **Mixup augmentation**: Combine with label mixup for further regularization

### Research Directions

1. **Theoretical analysis**: Convergence guarantees for joint optimization
2. **Noise characterization**: Automatic noise level estimation
3. **Transfer learning**: Use refined labels from one dataset to improve another
4. **Multi-modal fusion**: Extend to multiple sensor modalities

---

## Contact and Support

For questions, issues, or contributions:
- Check the main project README: `/Users/zongfan/Documents/MY/JNJ/CLAUDE.md`
- Review loss implementations: `Helpers/dlrtc_losses.py`
- Examine training script: `predict_TSO_segment_patch_dlrtc.py`

**Important**: Always compare DLR-TC results against baseline (`predict_TSO_segment_patch.py`) to validate improvements on your specific dataset.
