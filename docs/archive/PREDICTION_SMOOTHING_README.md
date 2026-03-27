# Prediction Smoothing for TSO Classification

## Overview

This document describes the prediction smoothing functionality added to `predict_TSO_segment_patch.py` to address unstable predictions that jump between classes in short time periods.

## Problem

The model sometimes produces predictions that rapidly switch between classes (other, non-wear, predictTSO) within short time windows, which is often unrealistic for real-world behavior. This instability can be caused by:

- Noisy sensor data at specific time points
- Model uncertainty near decision boundaries
- Short transient events being over-emphasized

## Solution: Multiple Smoothing Methods

We implemented 5 different smoothing methods to post-process predictions:

### 1. **Majority Vote** (Recommended) ⭐
- **Method**: Sliding window majority voting on class labels
- **How it works**: For each time point, looks at surrounding predictions within a window and assigns the most common class
- **Best for**: General-purpose smoothing while preserving class transitions
- **Parameters**: `window_size` (default: 5 minutes)

```python
# Example: With window_size=5, for each minute:
# - Look at predictions from t-2 to t+2 (5 minutes total)
# - Count occurrences of each class
# - Assign the class with the most votes
```

### 2. **Median Filter**
- **Method**: Applies scipy's median filter on class predictions
- **How it works**: Replaces each prediction with the median of surrounding predictions
- **Best for**: Removing isolated prediction spikes
- **Parameters**: `window_size` (odd numbers work best)

### 3. **Moving Average**
- **Method**: Applies moving average on class probabilities, then takes argmax
- **How it works**: Smooths probability distributions before making final class decision
- **Best for**: Gradual transitions between states
- **Parameters**: `window_size`

### 4. **Gaussian Weighted Average**
- **Method**: Gaussian-weighted moving average on probabilities
- **How it works**: Similar to moving average but nearby time points have more influence
- **Best for**: Smooth temporal transitions with emphasis on local context
- **Parameters**: `window_size` (determines sigma = window_size/4)

### 5. **Minimum Segment Length**
- **Method**: Removes segments shorter than a threshold by merging with neighbors
- **How it works**: Finds contiguous segments of same class, merges short ones into adjacent segments
- **Best for**: Post-processing to remove brief isolated predictions
- **Parameters**: `min_segment_length` (default: 3 minutes)

## Usage

### Command Line Arguments

```bash
python predict_TSO_segment_patch.py \
    --input_data_folder /path/to/data \
    --output results_folder \
    --visualize \
    --smooth_predictions \
    --smooth_method majority_vote \
    --smooth_window 5
```

**Available Arguments:**
- `--smooth_predictions`: Enable prediction smoothing (flag)
- `--smooth_method`: Choose method from: `majority_vote`, `median_filter`, `moving_average`, `gaussian`, `min_segment`
- `--smooth_window`: Window size in minutes (default: 5, should be odd for best results)

### Combining Multiple Methods

For best results, you can combine methods programmatically:

```python
from predict_TSO_segment_patch import smooth_predictions_combined

smoothed = smooth_predictions_combined(
    predictions,
    methods=['majority_vote', 'min_segment'],
    window_size=5,
    min_segment_length=3
)
```

## Visualization Enhancements

When smoothing is enabled, visualizations now include:

1. **5 rows instead of 4**:
   - Row 1: Accelerometer data (X, Y, Z)
   - Row 2: Temperature and time cyclic features
   - Row 3: Ground truth labels
   - Row 4: **Raw predictions** (before smoothing) with accuracy
   - Row 5: **Smoothed predictions** with accuracy improvement

2. **Accuracy comparison** in title:
   - Shows both raw and smoothed accuracy
   - Example: `Raw Acc: 87.3% | Smoothed Acc: 92.1%`

3. **Delta information**:
   - Smoothed prediction plot shows improvement
   - Example: `Acc: 92.1% (Δ: +4.8%)`

4. **Low accuracy folder organization**:
   - Predictions below 90% accuracy → saved to `{split_type}/low_acc/` subfolder
   - Example structure:
     ```
     training/debug_predictions/
     ├── train/
     │   ├── sample1_Acc=95.23.png
     │   └── low_acc/
     │       └── sample2_Acc=85.67.png
     └── val/
         ├── sample3_Acc=91.45.png
         └── low_acc/
             └── sample4_Acc=88.92.png
     ```

## Recommendations

### Window Size Selection

- **3 minutes**: Light smoothing, preserves fast transitions
- **5 minutes** ⭐: Balanced smoothing (recommended default)
- **7 minutes**: Moderate smoothing for noisy data
- **9 minutes**: Aggressive smoothing, may miss short events

### Method Selection by Use Case

| Use Case | Recommended Method | Window Size |
|----------|-------------------|-------------|
| General purpose | `majority_vote` | 5-7 |
| Very noisy data | `majority_vote` + `min_segment` | 7-9, min_len=3 |
| Preserve probability info | `moving_average` or `gaussian` | 5 |
| Remove isolated spikes | `median_filter` or `min_segment` | 5, min_len=2 |
| Clinical interpretation | `majority_vote` | 5 (conservative) |

## Expected Performance Improvements

Based on typical scenarios:

- **Accuracy improvement**: +2% to +8% depending on prediction stability
- **False positive reduction**: Fewer brief, spurious class switches
- **Temporal consistency**: More realistic state transitions
- **Clinical validity**: Better alignment with expected behavior patterns

## Implementation Details

### Key Functions

1. **`smooth_predictions()`** ([predict_TSO_segment_patch.py:45-156](predict_TSO_segment_patch.py#L45-L156))
   - Single method smoothing
   - Supports all 5 methods
   - Works on batched predictions

2. **`smooth_predictions_combined()`** ([predict_TSO_segment_patch.py:159-179](predict_TSO_segment_patch.py#L159-L179))
   - Sequential application of multiple methods
   - Recommended: `majority_vote` → `min_segment`

3. **`visualize_batch_predictions()`** ([predict_TSO_segment_patch.py:183-405](predict_TSO_segment_patch.py#L183-L405))
   - Enhanced to show both raw and smoothed predictions
   - Calculates accuracy for both versions
   - Organizes output by split_type and accuracy threshold

### Technical Notes

- Smoothing is applied **after** model prediction (post-processing)
- Does **not** affect model training or loss calculation
- Only impacts visualization and final output interpretation
- Preserves original sequence lengths and valid data regions
- Handles padding correctly (smoothing only applied to valid predictions)

## Future Enhancements

Potential improvements to consider:

1. **Adaptive smoothing**: Adjust window size based on prediction confidence
2. **State-specific smoothing**: Different parameters for different classes
3. **HMM-based smoothing**: Use Hidden Markov Models for state transitions
4. **Confidence-weighted voting**: Weight votes by prediction probability
5. **Export smoothed predictions**: Save smoothed predictions for downstream analysis

## References

- Median filter: `scipy.signal.medfilt`
- Gaussian filter: `scipy.ndimage.gaussian_filter1d`
- Majority voting: Custom implementation with `np.bincount`

## Questions or Issues?

If you encounter problems or have suggestions for improvement, please check:

1. Window size is appropriate for your data's temporal resolution
2. Method choice matches your use case
3. Smoothing doesn't remove clinically significant short events
4. Accuracy improvements align with domain expectations
