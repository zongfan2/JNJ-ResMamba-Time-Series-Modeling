# -*- coding: utf-8 -*-
"""
Split from DL_helpers.py - Modular code structure
@author: MBoukhec (original)
"""

import numpy as np
import pandas as pd
import torch

def smooth_binary_series(binary_series, window_size, threshold):
    """
    Smooths a binary time series by rejecting short events (less than a threshold length).

    Parameters:
    - binary_series: The input binary time series (list or numpy array of 0s and 1s)
    - window_size: The size of the sliding window for smoothing (e.g., 5)
    - threshold: The minimum length of an event to keep (events shorter than this are removed)

    Returns:
    - smoothed_series: The smoothed binary time series
    """
    smoothed_series = np.zeros_like(binary_series)

    # Iterate through the series with a sliding window
    for i in range(len(binary_series)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(binary_series), i + window_size // 2 + 1)

        window = binary_series[start_idx:end_idx]
        event_length = np.sum(window)

        # If event is longer than threshold, keep it, otherwise remove it
        if event_length >= threshold:
            smoothed_series[i] = 1 if np.sum(window) > 0 else 0
        else:
            smoothed_series[i] = 0

    return smoothed_series


def seq_to_seconds(batch_y,y_lens):
    """
    Changes a batch of sequences of single observations to seconds. Spliting each sequence seperatly then concatenating to avoid having 1sec windows running on multiple segments.

    Parameters:
    - batch_y: batch sequence of predicted or ground truth labels
    - y_lens: An array containing original length of each sequence 

    Returns:
    - s:Second level sequence
    """
    i=0
    s=np.array([],dtype=int)
    for l in y_lens:
        s=np.concatenate((s,block_reduce(batch_y[i:i+l], block_size=20, func=np.any, cval=0)))
    return s


def remove_padding(batch_y,y_lens):
    """
    Remove padding from a padded batch of sequence.

    Parameters:
    - batch_y: batch sequence of predicted or ground truth labels
    - y_lens: An array containing original length of each sequence 
    - batch_size: Size of the batch

    Returns:
    - Sequences without padding concatenated in 1D array
    """
    o=batch_y.reshape(-1,max(y_lens))
    s=np.array([],dtype=int)
    for i in range(o.shape[0]):
        s=np.concatenate((s,o[i][:y_lens[i]]))
    return s


def enforce_single_tso_period(predictions, min_gap_minutes=30, min_duration_minutes=10):
    """
    Post-process predictions to enforce single, continuous TSO period.

    Finds all TSO segments, merges close ones, and keeps only the longest period.
    This guarantees that the final prediction has at most one TSO period per night.

    Args:
        predictions: [seq_len] or [batch, seq_len] - predicted class labels (0, 1, 2)
                    0 = other, 1 = non-wear, 2 = predictTSO
        min_gap_minutes: Minimum gap to consider as separate TSO periods (default: 30)
                        Segments closer than this will be merged
        min_duration_minutes: Minimum duration to keep a TSO period (default: 10)
                             Shorter segments are filtered out as noise

    Returns:
        processed_predictions: Same shape as input, with only longest TSO period retained

    Example:
        Input:  [0, 0, 2, 2, 0, 0, 2, 2, 2, 0, ...]  (multiple TSO periods)
        Output: [0, 0, 0, 0, 0, 0, 2, 2, 2, 0, ...]  (only longest kept)
    """
    import numpy as np

    # Handle batch dimension
    if predictions.ndim == 2:
        # Batch processing
        processed_batch = []
        for i in range(predictions.shape[0]):
            processed = enforce_single_tso_period(
                predictions[i],
                min_gap_minutes=min_gap_minutes,
                min_duration_minutes=min_duration_minutes
            )
            processed_batch.append(processed)
        return np.stack(processed_batch)

    # Single sequence processing
    predictions = np.array(predictions)  # Ensure numpy array
    tso_class = 1 if predictions.max() <= 1 else 2  # binary vs 3-class
    tso_mask = (predictions == tso_class)

    if not tso_mask.any():
        return predictions  # No TSO predicted, return as-is

    # Find all continuous TSO segments
    segments = []
    in_segment = False
    start = 0

    for i in range(len(tso_mask)):
        if tso_mask[i] and not in_segment:
            # Start of new TSO segment
            start = i
            in_segment = True
        elif not tso_mask[i] and in_segment:
            # End of TSO segment
            segments.append((start, i - 1))
            in_segment = False

    # Handle case where TSO extends to end of sequence
    if in_segment:
        segments.append((start, len(tso_mask) - 1))

    if len(segments) == 0:
        return predictions  # No valid segments

    # Filter out very short segments (likely noise)
    segments = [(s, e) for s, e in segments if (e - s + 1) >= min_duration_minutes]

    if len(segments) == 0:
        # All segments were too short - remove all TSO predictions
        processed_predictions = predictions.copy()
        processed_predictions[tso_mask] = 0  # Change to "other"
        return processed_predictions

    # Merge segments that are close together (within min_gap_minutes)
    merged_segments = []
    for start, end in segments:
        if merged_segments and (start - merged_segments[-1][1] - 1) <= min_gap_minutes:
            # Gap is small, merge with previous segment
            merged_segments[-1] = (merged_segments[-1][0], end)
        else:
            # Gap is large or first segment
            merged_segments.append((start, end))

    # Keep only the longest merged segment
    if len(merged_segments) > 1:
        # Find longest segment
        longest_segment = max(merged_segments, key=lambda x: x[1] - x[0])

        # Create processed predictions with only longest segment
        processed_predictions = predictions.copy()

        # Remove all TSO predictions first
        processed_predictions[tso_mask] = 0  # Change to "other"

        # Restore only the longest segment
        processed_predictions[longest_segment[0]:longest_segment[1]+1] = tso_class

        return processed_predictions
    else:
        # Only one segment, return as-is
        return predictions



def batch_enforce_single_tso(predictions, x_lengths, min_gap_minutes=30, min_duration_minutes=10):
    """
    Batch version of enforce_single_tso_period with sequence length handling.

    Args:
        predictions: [batch_size, seq_len] - predicted class labels
        x_lengths: [batch_size] - valid sequence lengths
        min_gap_minutes: Minimum gap to merge TSO segments
        min_duration_minutes: Minimum TSO duration to keep

    Returns:
        processed_predictions: [batch_size, seq_len] - with single TSO period per sample
    """
    import numpy as np

    predictions_np = predictions.cpu().numpy() if torch.is_tensor(predictions) else np.array(predictions)
    x_lengths_np = x_lengths.cpu().numpy() if torch.is_tensor(x_lengths) else np.array(x_lengths)

    processed_batch = []

    for i in range(len(predictions_np)):
        valid_len = int(x_lengths_np[i])

        # Process only valid part
        pred_seq = predictions_np[i, :valid_len].copy()
        processed_seq = enforce_single_tso_period(
            pred_seq,
            min_gap_minutes=min_gap_minutes,
            min_duration_minutes=min_duration_minutes
        )

        # Reconstruct full sequence (with padding)
        full_seq = predictions_np[i].copy()
        full_seq[:valid_len] = processed_seq

        processed_batch.append(full_seq)

    result = np.stack(processed_batch)

    # Convert back to tensor if input was tensor
    if torch.is_tensor(predictions):
        return torch.from_numpy(result).to(predictions.device)
    else:
        return result



def smooth_predictions(predictions, method='majority_vote', window_size=5, min_segment_length=3):
    """
    Smooth predictions to reduce unstable jumping between classes.

    Args:
        predictions: [batch_size, seq_len, 3] - model predictions (logits or probabilities)
        method: str - smoothing method:
            - 'majority_vote': Sliding window majority voting (recommended for classification)
            - 'median_filter': Median filter on class predictions
            - 'moving_average': Moving average on probabilities then argmax
            - 'gaussian': Gaussian weighted moving average
            - 'min_segment': Remove segments shorter than min_segment_length
        window_size: int - size of smoothing window (should be odd, e.g., 5, 7, 9)
        min_segment_length: int - minimum segment length for 'min_segment' method

    Returns:
        smoothed_predictions: [batch_size, seq_len, 3] - smoothed predictions
    """
    import torch
    import numpy as np
    from scipy.ndimage import median_filter, gaussian_filter1d
    from scipy.signal import medfilt

    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()

    batch_size, seq_len, num_classes = predictions.shape
    smoothed = predictions.copy()

    # Convert to probabilities if needed (assume sigmoid already applied)
    probs = predictions  # Already probabilities from sigmoid

    for i in range(batch_size):
        if method == 'majority_vote':
            # Sliding window majority voting on class labels
            class_preds = np.argmax(probs[i], axis=-1)  # [seq_len]
            smoothed_classes = np.zeros_like(class_preds)

            half_window = window_size // 2
            for t in range(seq_len):
                # Get window bounds
                start = max(0, t - half_window)
                end = min(seq_len, t + half_window + 1)
                window = class_preds[start:end]

                # Majority vote
                counts = np.bincount(window, minlength=num_classes)
                smoothed_classes[t] = np.argmax(counts)

            # Convert back to one-hot style probabilities
            smoothed[i] = np.eye(num_classes)[smoothed_classes]

        elif method == 'median_filter':
            # Median filter on class predictions
            class_preds = np.argmax(probs[i], axis=-1)
            # Median filter requires odd kernel size
            kernel_size = window_size if window_size % 2 == 1 else window_size + 1
            smoothed_classes = medfilt(class_preds.astype(float), kernel_size=kernel_size).astype(int)
            smoothed_classes = np.clip(smoothed_classes, 0, num_classes - 1)
            smoothed[i] = np.eye(num_classes)[smoothed_classes]

        elif method == 'moving_average':
            # Moving average on probabilities
            for c in range(num_classes):
                smoothed[i, :, c] = np.convolve(probs[i, :, c],
                                                np.ones(window_size)/window_size,
                                                mode='same')

        elif method == 'gaussian':
            # Gaussian weighted moving average
            sigma = window_size / 4  # Standard deviation
            for c in range(num_classes):
                smoothed[i, :, c] = gaussian_filter1d(probs[i, :, c], sigma=sigma, mode='nearest')

        elif method == 'min_segment':
            # Remove short segments by merging with neighbors
            class_preds = np.argmax(probs[i], axis=-1)
            smoothed_classes = class_preds.copy()

            # Find segment boundaries
            changes = np.where(np.diff(class_preds) != 0)[0] + 1
            segments = np.split(np.arange(seq_len), changes)
            segment_labels = [class_preds[seg[0]] for seg in segments]

            # Merge short segments
            merged_labels = []
            merged_segments = []

            for idx, (seg, label) in enumerate(zip(segments, segment_labels)):
                if len(seg) < min_segment_length:
                    # Merge with previous or next segment
                    if merged_labels:
                        # Merge with previous
                        merged_segments[-1] = np.concatenate([merged_segments[-1], seg])
                    elif idx + 1 < len(segments):
                        # Merge with next
                        continue
                    else:
                        # Keep as is (edge case)
                        merged_segments.append(seg)
                        merged_labels.append(label)
                else:
                    merged_segments.append(seg)
                    merged_labels.append(label)

            # Apply merged labels
            for seg, label in zip(merged_segments, merged_labels):
                smoothed_classes[seg] = label

            smoothed[i] = np.eye(num_classes)[smoothed_classes]

    return smoothed



def smooth_predictions_combined(predictions, methods=['majority_vote', 'min_segment'],
                                window_size=5, min_segment_length=3):
    """
    Apply multiple smoothing methods sequentially.
    Recommended: majority_vote -> min_segment for best results.

    Args:
        predictions: [batch_size, seq_len, 3] - model predictions
        methods: list of str - smoothing methods to apply in order
        window_size: int - window size for voting/averaging methods
        min_segment_length: int - minimum segment length

    Returns:
        smoothed_predictions: [batch_size, seq_len, 3] - smoothed predictions
    """
    smoothed = predictions
    for method in methods:
        smoothed = smooth_predictions(smoothed, method=method,
                                     window_size=window_size,
                                     min_segment_length=min_segment_length)
    return smoothed


# ==================== Learning Curves Plotting Function ====================

