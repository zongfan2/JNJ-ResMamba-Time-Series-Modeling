# Post-Processing Algorithms Guide

## Table of Contents
1. [Prediction Smoothing](#prediction-smoothing)
2. [Single TSO Period Enforcement](#single-tso-period-enforcement)
3. [Implementation Details](#implementation-details)
4. [Practical Recommendations](#practical-recommendations)

---

## Prediction Smoothing

### Problem Statement

Model predictions sometimes produce unstable outputs that rapidly switch between classes:

```
Time:    0   1   2   3   4   5   6   7   8   9
Pred: [ 0 → 1 → 0 → 2 → 0 → 1 → 0 → 2 → 1 → 0 ]  ← Jittery, unrealistic
```

This instability arises from:
- Noisy sensor measurements at specific time points
- Model uncertainty near decision boundaries
- Over-emphasis of short, transient events

**Solution**: Post-process predictions to smooth class assignments while preserving meaningful transitions.

### Available Methods

#### 1. Majority Vote (Recommended) ⭐

**Algorithm**: Sliding window majority voting on class labels

**How it works**:
```python
for time_t:
    window = predictions[t-2:t+3]  # 5-minute window (±2 minutes)
    predicted_class = argmax(count(window))
    smoothed[t] = predicted_class
```

**Example**:
```
Raw predictions:   [ 0 1 0 2 0 1 0 2 1 0 ]
Window @ t=4:      [ 0 2 0 ... ] → count: {0:2, 2:1} → vote: 0
Smoothed:          [ 0 0 0 1 0 1 1 1 1 0 ]
```

**Parameters**:
- `window_size` (default: 5 minutes)
  - 3 min: Light smoothing, fast transitions preserved
  - 5 min: Balanced (recommended)
  - 7-9 min: Aggressive, may miss short events

**When to use**: General-purpose smoothing, clinical applications

**Expected improvement**: +2-5% accuracy

#### 2. Median Filter

**Algorithm**: Median of predictions in sliding window

**How it works**:
```python
smoothed = scipy.signal.medfilt(predictions, kernel_size=5)
```

**Advantages**:
- Removes isolated spikes
- Preserves sharp transitions
- Less aggressive than majority vote

**Best for**: Data with occasional prediction spikes

**Expected improvement**: +1-3% accuracy

#### 3. Moving Average

**Algorithm**: Moving average on class probabilities → argmax

**How it works**:
```python
smooth_probs = np.convolve(probabilities, window, mode='same')
smoothed = argmax(smooth_probs, axis=-1)
```

**Advantages**:
- Preserves probability information
- Smooth gradual transitions
- Good for confidence-based decisions

**Best for**: Probability interpretation important

**Expected improvement**: +1-4% accuracy

#### 4. Gaussian Weighted Average

**Algorithm**: Gaussian-weighted moving average on probabilities

**How it works**:
```python
gaussian_weight = exp(-(x^2) / (2 * sigma^2))
smooth_probs = sum(probs * gaussian_weight) / sum(gaussian_weight)
smoothed = argmax(smooth_probs)
```

**Advantages**:
- Nearby points have more influence
- Smooth, realistic transitions
- Mathematically principled

**Parameters**:
- `window_size`: Determines σ = window_size / 4

**Best for**: Scientific/medical applications

**Expected improvement**: +2-5% accuracy

#### 5. Minimum Segment Length

**Algorithm**: Remove contiguous segments shorter than threshold

**How it works**:
```python
# Find contiguous regions of same class
segments = find_contiguous_segments(predictions)

for segment in segments:
    if len(segment) < min_length:
        # Merge with longer neighbor
        merge_with_adjacent(segment)

return merged_predictions
```

**Example**:
```
Raw:      [ 0 0 1 1 1 0 0 0 2 2 0 0 0 0 ]
Segments: [ 0(2) 1(3) 0(3) 2(2) 0(4) ]
Filter (min_len=3): Remove 2(2), merge with neighbors
Result:   [ 0 0 1 1 1 0 0 0 0 0 0 0 0 0 ]
```

**Parameters**:
- `min_segment_length` (default: 3 minutes)
  - 2-3 min: Removes brief events
  - 5-10 min: Aggressive filtering
  - Clinical decision required

**Best for**: Post-processing with known minimum event duration

**Expected improvement**: +1-3% accuracy

### Combining Methods

For best results, apply multiple smoothing methods sequentially:

```python
from Helpers.DL_helpers import smooth_predictions_combined

# Pipeline: majority vote → minimum segment filtering
smoothed = smooth_predictions_combined(
    predictions,
    methods=['majority_vote', 'min_segment'],
    window_size=5,
    min_segment_length=3
)
```

**Recommended pipeline**:
1. First: `majority_vote` (preserve important transitions)
2. Then: `min_segment` (remove noise)

**Expected combined improvement**: +3-8% accuracy

### Command Line Usage

```bash
# Majority vote smoothing
python predict_TSO_segment_patch.py \
    --input_data_folder /path/to/data \
    --output results \
    --smooth_predictions \
    --smooth_method majority_vote \
    --smooth_window 5

# Minimum segment filtering
python predict_TSO_segment_patch.py \
    --input_data_folder /path/to/data \
    --output results \
    --smooth_predictions \
    --smooth_method min_segment \
    --smooth_window 3  # min_segment_length in minutes
```

### Performance Visualization

When smoothing is enabled, visualizations show:

**5-panel layout**:
1. Accelerometer (X, Y, Z)
2. Temperature & time features
3. Ground truth labels
4. **Raw predictions** (before smoothing) with accuracy
5. **Smoothed predictions** (after smoothing) with improvement Δ

**Example output in title**:
```
Raw Accuracy: 87.3% | Smoothed Accuracy: 92.1% (Δ: +4.8%)
```

**Low accuracy folder**:
Predictions < 90% accuracy are saved to `{split}/low_acc/` subdirectory for review.

---

## Single TSO Period Enforcement

### Problem Statement

**Observation**: Neural network predicts multiple fragmented TSO periods per night

**Reality**: People typically have one continuous sleep/rest period per night

**Example**:
```
Predicted TSO periods:
  22:00-23:00 (TSO)
  23:30-01:00 (gap: 30 min)
  01:45-03:00 (TSO)  ← Should be merged
  06:00-06:30 (TSO)  ← Should be filtered (too short)

Corrected:
  22:00-03:00 (single continuous period)
  [ignore 6:00-6:30 as noise]
```

### Solution: Hybrid Approach

**Two-phase strategy**:

1. **Phase 1: Training Time** - Continuity loss guides learning
2. **Phase 2: Inference Time** - Post-processing enforces single period

### Implementation: Training with Continuity Loss

#### Continuity Loss Function

**Purpose**: Penalizes fragmented predictions during training

**Mathematical formulation**:

```
L_continuity = (1 / T) * Σ_t |p_tso[t+1] - p_tso[t]|
```

Where `p_tso[t]` is TSO probability at time t.

**Effect**:
- Rewards smooth TSO probability curves: `0 → 0.3 → 0.8 → 1.0 → 0.3 → 0`
- Penalizes jumpy patterns: `0 → 1 → 0 → 1 → 0` (fragmented)

#### Usage During Training

```python
# Import functions
from Helpers.DL_helpers import measure_loss_tso_with_continuity

# Configure
best_params = {
    'batch_size': 24,
    'lr': 0.001,
    'continuity_weight': 0.1,  # Add this parameter
}

# In training loop
total_loss, class_loss, cont_loss = measure_loss_tso_with_continuity(
    outputs,
    labels,
    x_lens,
    continuity_weight=best_params['continuity_weight']
)

# Monitor both components
if batches % 100 == 0:
    print(f"Classification loss: {class_loss:.4f}")
    print(f"Continuity loss: {cont_loss:.4f}")
    print(f"Ratio: {cont_loss/class_loss:.2f}")

# Backprop as normal
total_loss.backward()
optimizer.step()
```

#### Hyperparameter Tuning

| Weight | Effect | Use Case |
|--------|--------|----------|
| 0.0 | Disabled (standard training) | Baseline comparison |
| 0.05 | Light regularization | Testing phase |
| 0.1 | Recommended default | Most scenarios |
| 0.2-0.5 | Strong regularization | Very fragmented labels |
| > 0.5 | Too aggressive | Usually hurts performance |

**Monitoring during training**:
- Good: `cont_loss = 0.05`, `class_loss = 0.5` (ratio 1:10)
- Warning: `cont_loss = 0.8`, `class_loss = 0.5` (ratio 1.6:1, too high)

### Implementation: Post-Processing Inference

#### Single TSO Period Enforcement Algorithm

**Input**: Raw predicted class sequence (0, 1, 2)

**Process**:
1. Identify all contiguous TSO segments (class 2)
2. Merge segments within `min_gap_minutes`
3. Remove merged segment if shorter than `min_duration_minutes`
4. Keep only longest remaining segment

**Output**: Single TSO period per sequence

#### Usage During Inference

```python
from Helpers.DL_helpers import batch_enforce_single_tso

# In evaluation loop
model.eval()
with torch.no_grad():
    for batch in dataloader:
        # Forward pass
        outputs = model(batch_X, x_lens)
        predictions = torch.argmax(outputs, dim=-1)

        # ✓ Post-process to enforce single TSO period
        predictions = batch_enforce_single_tso(
            predictions,
            x_lens,
            min_gap_minutes=30,      # Merge segments within 30 min
            min_duration_minutes=10  # Filter out TSO < 10 min
        )

        # Calculate metrics with processed predictions
        f1 = calculate_f1(predictions, labels)
```

#### Parameters

**`min_gap_minutes` (default: 30)**
- Gap between TSO segments to trigger merging
- Range: 15-60 minutes (typical sleep fragmentation patterns)
- Higher value: More aggressive merging
- Lower value: Preserve genuine multi-period events

**Examples**:
```
Segments: [ 22:00-23:00, 23:25-01:30, 06:00-06:20 ]

gap=15:  Merge 1&2 (gap 25>15) → Single long period
gap=30:  Merge 1&2 (gap 25<30, within threshold)
gap=60:  Merge 1&2, potentially merge with 3 (gap 240>60)
```

**`min_duration_minutes` (default: 10)**
- Minimum length to keep merged TSO segment
- Range: 5-30 minutes
- Higher value: Remove more brief events
- Lower value: Keep even short TSO periods

**Examples**:
```
Segments: [ 22:00-23:00 (60 min), 06:00-06:05 (5 min) ]

min_dur=5:  Keep both segments
min_dur=10: Remove 5-min segment, keep 60-min
```

### Experimental Setup

#### Ablation Study Protocol

Recommended progression to understand component contributions:

**Experiment 1: Baseline**
```python
continuity_weight = 0.0
enforce_single_period = False
```
Establishes baseline F1 performance

**Experiment 2: Post-processing Only**
```python
continuity_weight = 0.0  # No training changes
enforce_single_period = True
params: min_gap=30, min_dur=10
```
Tests if post-processing alone helps

**Experiment 3: Continuity Loss Only**
```python
continuity_weight = 0.1
enforce_single_period = False
```
Tests if training guidance helps raw predictions

**Experiment 4: Hybrid (Full Approach)**
```python
continuity_weight = 0.1
enforce_single_period = True
```
Tests combined effect (expected best)

**Experiment 5: Hyperparameter Tuning**
```python
for gap in [15, 30, 45, 60]:
    for min_dur in [5, 10, 15]:
        for weight in [0.05, 0.1, 0.2]:
            # Train and evaluate
```
Optimizes parameters for your data

### Expected Performance

#### Baseline (no changes)
```
TSO F1: 0.65
Issues: Multiple fragmented TSO predictions per night
Example: 3-4 separate TSO events predicted instead of 1-2
```

#### Post-processing Only
```
TSO F1: 0.68-0.72 (+3-7% improvement)
Benefit: Guaranteed single period, cleaner predictions
No retraining required
```

#### Continuity Loss Only
```
TSO F1: 0.69-0.73 (+4-8% improvement)
Benefit: Model learns to prefer continuous predictions
Requires retraining
```

#### Hybrid (Both)
```
TSO F1: 0.72-0.77 (+7-12% improvement)
Benefit: Training guidance + guaranteed enforcement
Best results but requires retraining
```

### Visualization of Effects

**Visual progression** (conceptual):

```
Baseline:
XXXXXXXXXXXXX________XXXXXXXXXX__________XXXXXXXXX
    ↑TSO             ↑TSO              ↑TSO (fragmented)

Post-processing (gap=30):
XXXXXXXXXXXXXXXXXXXXXXXXXX________XXXXXXXXXXXXXXXXXXXXX
         ↑ Single TSO period               ↑ Short event filtered

Post-processing (min_dur=10):
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
                    ↑ Single clean TSO period
```

---

## Implementation Details

### Core Functions Reference

#### Prediction Smoothing

**File**: `Helpers/DL_helpers.py` (lines ~45-180)

```python
def smooth_predictions(predictions, method='majority_vote', window_size=5):
    """
    Apply smoothing to predictions using specified method.

    Args:
        predictions: [seq_len] or [batch, seq_len] class indices
        method: 'majority_vote', 'median_filter', 'moving_average',
                'gaussian', 'min_segment'
        window_size: Window size in timesteps or minutes

    Returns:
        smoothed_predictions: Same shape as input
    """
```

**Available methods**:
- `majority_vote`: Recommended for general use
- `median_filter`: Good for spike removal
- `moving_average`: Preserves probabilities
- `gaussian`: Smooth weighted average
- `min_segment`: Removes short segments

#### Single TSO Period Enforcement

**File**: `Helpers/DL_helpers.py` (lines ~300-450)

```python
def measure_loss_tso_with_continuity(outputs, labels, x_lengths,
                                     continuity_weight=0.1):
    """
    Calculate classification loss + continuity regularization.

    Args:
        outputs: [batch, seq_len, 3] model predictions
        labels: [batch, seq_len] ground truth
        x_lengths: [batch] sequence lengths
        continuity_weight: λ for continuity term

    Returns:
        total_loss, classification_loss, continuity_loss
    """

def batch_enforce_single_tso(predictions, x_lengths,
                             min_gap_minutes=30,
                             min_duration_minutes=10):
    """
    Post-process predictions to enforce single TSO period per sample.

    Args:
        predictions: [batch, seq_len] class indices (0,1,2)
        x_lengths: [batch] actual sequence lengths
        min_gap_minutes: Merge TSO segments within this gap
        min_duration_minutes: Filter TSO segments shorter than this

    Returns:
        processed_predictions: [batch, seq_len] with single TSO
    """
```

### Mask and Padding Handling

Both algorithms properly handle:
- **Padding**: Ignore padded regions (padding_value = -999 or similar)
- **Sequence lengths**: Respect actual sequence lengths per sample
- **Batching**: Process all samples independently and in parallel

---

## Practical Recommendations

### Quick Decision Tree

**Q1: Do predictions have jitter/instability?**
- Yes → Apply majority vote smoothing (window_size=5)
- No → Skip smoothing, go to Q2

**Q2: Are there multiple TSO periods per night?**
- Yes → Use single TSO enforcement
  - **First try**: Post-processing only (no retraining)
  - **If insufficient**: Retrain with continuity_weight=0.1
- No → Done, use processed predictions

**Q3: Metrics still below target?**
- Yes → Try hyperparameter tuning
  - Smoothing: Adjust window_size (3, 5, 7, 9 minutes)
  - Single TSO: Adjust min_gap_minutes (15-60) and min_duration_minutes (5-20)
  - Continuity: Tune weight (0.05-0.5)
- No → Use current configuration

### Use Case Recommendations

**Clinical/Medical Applications**
- Smoothing method: `majority_vote` (interpretable, conservative)
- Single TSO: Always enable (clinically realistic)
- Window size: 5 minutes (balanced)
- Continuity weight: 0.1 (moderate)

**Research/Analysis**
- Smoothing method: `moving_average` or `gaussian` (preserves probabilities)
- Single TSO: Enable if biologically justified
- Window size: 3-7 minutes (flexible)
- Continuity weight: 0.05-0.2 (tunable)

**Production/Real-time**
- Smoothing method: `majority_vote` (fast, deterministic)
- Single TSO: Enable (practical requirement)
- Window size: 5 minutes (standard)
- Continuity weight: 0.1 (proven default)

### A/B Testing Framework

**Minimal viable test**:

```python
# 1. Baseline (no post-processing)
baseline_f1 = evaluate(model, test_data)

# 2. Apply best-practice post-processing
predictions_smooth = smooth_predictions_combined(
    predictions,
    methods=['majority_vote', 'min_segment']
)
predictions_final = batch_enforce_single_tso(predictions_smooth)
postprocessed_f1 = evaluate_custom(predictions_final, test_data)

# 3. Compare
improvement = (postprocessed_f1 - baseline_f1) / baseline_f1 * 100
print(f"Improvement: {improvement:+.1f}%")

# 4. If > 3%, use post-processing in production
# If < 3%, investigate why (data quality? model issues?)
```

### Troubleshooting

**Issue: Smoothing removes legitimate short events**

Solutions:
1. Decrease `window_size` (e.g., 3 min instead of 5)
2. Use `min_segment` filtering instead (less aggressive)
3. Combine with confidence-weighted voting (advanced)

**Issue: Single TSO enforcement breaks multi-period ground truth**

Solutions:
1. Check if ground truth truly has multiple periods
2. Increase `min_gap_minutes` to keep closer events separate
3. Skip enforcement if data authentically multi-period

**Issue: Continuity loss makes predictions too smooth**

Solutions:
1. Decrease `continuity_weight` (0.05 or lower)
2. Combine with post-processing instead of relying on loss alone
3. Verify `min_duration_minutes` isn't filtering real events

---

## Summary

### Quick Reference

**Smoothing**:
```bash
# Command line
--smooth_predictions --smooth_method majority_vote --smooth_window 5

# Python
smooth_predictions_combined(pred, ['majority_vote', 'min_segment'])
```

**Single TSO**:
```python
# Training (optional)
measure_loss_tso_with_continuity(outputs, labels, x_lens, 0.1)

# Inference (recommended)
batch_enforce_single_tso(predictions, x_lens, 30, 10)
```

**Expected improvements**:
- Smoothing alone: +2-5% accuracy
- Single TSO alone: +3-7% TSO F1
- Combined: +5-12% overall F1

**Best practice**:
1. Start with majority vote smoothing (window=5)
2. Add single TSO enforcement post-processing
3. Monitor clinical validity (don't over-smooth real events)
4. If still below target, retrain with continuity loss (0.1)

See `project_overview.md` for model details and `deployment.md` for scaling.
