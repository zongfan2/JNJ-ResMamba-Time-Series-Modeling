# -*- coding: utf-8 -*-
"""
Split from DL_helpers.py - Modular code structure
@author: MBoukhec (original)
"""

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

def add_padding(batch,device,seg_column='segment', max_seq_len=None, random_start=False, padding_value=0.0):
    X_sequences=[]
    Y_sequences=[]
    x_lens=[]
    label1=[]
    label3=[]
    if max_seq_len is not None:
        max_seq_template = torch.ones(max_seq_len, 3).to(device)
        X_sequences.append(max_seq_template)
        # x_lens.append(max_seq_len)
    for index,seq in batch.groupby(seg_column, sort=False, observed=True):
        # X_arr=seq.loc[:, ['x', 'y', 'z','angle','wrist','position_segment','position_segmentr','position_TSO','position_TSOr', *seq.loc[:, 'stft_0':'stft_32'].columns]].to_numpy()
        # X_arr=seq.loc[:, ['x', 'y', 'z','angle','position_segment','position_segmentr','position_TSO','position_TSOr']].to_numpy()
        # X_arr=seq.loc[:, ['x', 'y', 'z','angle']].to_numpy()
        xyz = seq
        xyz_p=np.random.permutation(['x','y'])
        xyz=xyz.rename(columns={'x': xyz_p[0], 'y': xyz_p[1]})
        # xyz_p=np.random.permutation(['x','y','z'])
        # seq=seq.rename(columns={'x': xyz_p[0], 'y': xyz_p[1], 'z': xyz_p[2]})
        X_arr=seq.loc[:, ['x', 'y', 'z']].to_numpy()
        # X_arr=seq.loc[:, ['x', 'y', 'z','angle']].to_numpy()
        Y_arr=np.array([seq['scratch'].values]).T
        X_sequences.append(torch.tensor(X_arr,dtype=torch.float32,device=device))#,device=device
        # Y_sequences.append(torch.tensor(Y_arr,dtype=torch.long,device=device))#,device=device
        Y_sequences=np.concatenate((Y_sequences,seq['scratch'].values))
        x_lens.append(len(X_arr))
        label1.append(seq['segment_scratch'].any()*1)
        label3.append(seq['scratch_duration'].max())

    pad_X=pad_sequence(X_sequences, batch_first=True, padding_value=padding_value) #,padding_value=-999
    #label2=pad_sequence(Y_sequences, batch_first=True,padding_value=-999)
    if max_seq_len:
        # discard the max seq template now
        pad_X = pad_X[1:, :, :]
    label2=torch.tensor(Y_sequences,device=device)
    return pad_X, torch.tensor(label1,device=device), label2, torch.tensor(label3,device=device), x_lens



def add_padding_with_position(
    batch,
    device,
    seg_column='segment',
    max_seq_len=None,
    padding_position="tail",  # Options: "tail", "random"
    is_train=True,
    prob=0.5,
    padding_value=0
):
    """
    Add padding to sequences with support for both tail padding and random position padding.
    
    Args:
        batch: DataFrame containing the batch data
        device: Device to place tensors on
        seg_column: Column name for grouping sequences
        max_seq_len: Maximum sequence length after padding. If None, uses max length in batch
        padding_position: Where to place padding - "tail" (default) or "random"
        is_train: Whether this is training mode (random padding only applies during training)
        prob: Probability threshold for applying random padding
        padding_value: Value to use for padding (default: 0)
        
    Returns:
        pad_X: Padded input features tensor of shape [B, L, D]
        label1: Binary classification labels tensor of shape [B]
        label2: Binary sequence classification labels tensor of shape [B, L]
        label3: Regression target tensor of shape [B]
        x_lens: List of original sequence lengths
        seq_start_idx: List of sequence start indices
    """
    X_sequences = []
    Y_sequences = []
    x_lens = []
    label1 = []
    label3 = []
    seq_start_idx = []
    
    # First pass: collect all sequences and their lengths to determine max_seq_len if not provided
    sequences_data = []
    for _, seq in batch.groupby(seg_column, sort=False, observed=True):
        # Randomly permute x/y/z columns
        # TODO: disabled if position embedding is included
        xyz_p = np.random.permutation(['x', 'y'])
        seq = seq.rename(columns={'x': xyz_p[0], 'y': xyz_p[1]})

        X_arr = seq.loc[:, ['x', 'y', 'z']].to_numpy()
        Y_arr = seq['scratch'].values.astype(np.int64)
        seq_len = len(X_arr)
        
        sequences_data.append({
            'X_arr': X_arr,
            'Y_arr': Y_arr,
            'seq_len': seq_len,
            'seq': seq
        })
    
    # Determine max_seq_len if not provided
    if max_seq_len is None:
        max_seq_len = max(seq_data['seq_len'] for seq_data in sequences_data)
    
    # Second pass: create padded sequences
    for seq_data in sequences_data:
        X_arr = seq_data['X_arr']
        Y_arr = seq_data['Y_arr']
        seq_len = seq_data['seq_len']
        seq = seq_data['seq']
        
        # Initialize tensors with zeros (padding)
        if padding_value != 0:
            pad_tensor = torch.ones(max_seq_len, 3, dtype=torch.float32, device=device) * padding_value
        else:
            pad_tensor = torch.zeros(max_seq_len, 3, dtype=torch.float32, device=device)
        label_pad = torch.zeros(max_seq_len, dtype=torch.long, device=device)
        
        # Determine start index based on padding position
        use_random = (padding_position == "random" and 
                     seq_len < max_seq_len and 
                     torch.rand(1) > prob) 
                    #  and is_train)
        
        if use_random:
            # Random padding position
            start_idx = np.random.randint(0, max_seq_len - seq_len + 1)
        else:
            # Default: tail padding (sequence starts at the beginning)
            start_idx = 0
            
        seq_start_idx.append(start_idx)
        
        # If sequence is too long, crop it
        actual_seq_len = min(seq_len, max_seq_len)
        X_tensor = torch.tensor(X_arr[:actual_seq_len], dtype=torch.float32, device=device)
        Y_tensor = torch.tensor(Y_arr[:actual_seq_len], dtype=torch.long, device=device)
        
        # Place sequence at the determined start position
        pad_tensor[start_idx:start_idx+actual_seq_len, :] = X_tensor
        label_pad[start_idx:start_idx+actual_seq_len] = Y_tensor
        
        X_sequences.append(pad_tensor)
        Y_sequences.append(label_pad)
        x_lens.append(min(seq_len, max_seq_len))  # Store original length before any cropping
        
        label1.append(int(seq['segment_scratch'].any()))
        label3.append(float(seq['scratch_duration'].max()))

    # Stack as batch
    pad_X = torch.stack(X_sequences, dim=0)
    label2 = torch.stack(Y_sequences, dim=0)
    label1 = torch.tensor(label1, dtype=torch.long, device=device)
    label3 = torch.tensor(label3, dtype=torch.float, device=device)

    return pad_X, label1, label2, label3, x_lens, seq_start_idx


def add_padding_pretrain(batch,device,seg_column='segment',mask_rate=0.3, max_seq_len=None, padding_value=0.0):
    X_sequences=[]
    Y_sequences=[]
    x_lens=[]
    label1=[]
    label3=[]
    if max_seq_len is not None:
        max_seq_template = torch.ones(max_seq_len, 3).to(device)
        X_sequences.append(max_seq_template)
        # x_lens.append(max_seq_len)
    for index,seq in batch.groupby(seg_column, sort=False, observed=True):
        # X_arr=seq.loc[:, ['x', 'y', 'z','angle','wrist','position_segment','position_segmentr','position_TSO','position_TSOr', *seq.loc[:, 'stft_0':'stft_32'].columns]].to_numpy()
        # X_arr=seq.loc[:, ['x', 'y', 'z','angle','position_segment','position_segmentr','position_TSO','position_TSOr']].to_numpy()
        # X_arr=seq.loc[:, ['x', 'y', 'z','angle']].to_numpy()
        xyz = seq
        xyz_p=np.random.permutation(['x','y'])
        xyz=xyz.rename(columns={'x': xyz_p[0], 'y': xyz_p[1]})
        # xyz_p=np.random.permutation(['x','y','z'])
        # seq=seq.rename(columns={'x': xyz_p[0], 'y': xyz_p[1], 'z': xyz_p[2]})
        X_arr=seq.loc[:, ['x', 'y', 'z']].to_numpy()
        # Hard clip: truncate segment to max_seq_len BEFORE tensor creation
        # to prevent pad_sequence from allocating a massive tensor on GPU.
        if max_seq_len is not None and len(X_arr) > max_seq_len:
            X_arr = X_arr[:max_seq_len]
        X_sequences.append(torch.tensor(X_arr,dtype=torch.float32,device=device))
        x_lens.append(len(X_arr))
        # label1.append(seq['segment_scratch'].any()*1)
        # label3.append(seq['scratch_duration'].max())

    pad_X=pad_sequence(X_sequences, batch_first=True, padding_value=padding_value) #,padding_value=-999
    #label2=pad_sequence(Y_sequences, batch_first=True,padding_value=-999)
    if max_seq_len:
        # discard the max seq template now
        pad_X = pad_X[1:, :, :]
    # label2=torch.tensor(Y_sequences,device=device)
    return pad_X, torch.tensor(x_lens, dtype=torch.long, device=device)



def add_padding_TSO(batch, device, seg_column='segment', max_seq_len=None, padding_value=0.0):
    """
    Prepare batch data for status prediction task with minute-level aggregated features.

    Args:
        batch: DataFrame with 28 feature columns from load_data_tso:
               - time_cyclic, is_night, temperature (minute-level aggregated base values - 4 features)
               - x_mean, x_std, x_min, x_max, x_q25, x_q75, x_skew, x_kurt, x_cv (9 features)
               - y_mean, y_std, y_min, y_max, y_q25, y_q75, y_skew, y_kurt, y_cv (9 features)
               - z_mean, z_std, z_min, z_max, z_q25, z_q75, z_skew, z_kurt, z_cv (9 features)
               - non-wear, predictTSO (labels)
        device: torch device
        seg_column: column name for segment identifier
        max_seq_len: maximum sequence length for padding (e.g., 1440 for 24h at minute level)
        padding_value: value to use for padding

    Returns:
        pad_X: [batch_size, seq_len, 28] - input features (28 channels)
        labels: [batch_size, 3] - segment-level class labels (other, non-wear, predictTSO)
        x_lens: [batch_size] - original sequence lengths
    """
    # Define all 28 feature columns in order (4 base + 27 statistical)
    feature_cols = (
        ['temperature', 'time_cyclic'] +  # 4 base aggregated values
        # x features (9)
        ['x_mean', 'x_std', 'x_min', 'x_max', 'x_q25', 'x_q75', 'x_skew', 'x_kurt', 'x_cv'] +
        # y features (9)
        ['y_mean', 'y_std', 'y_min', 'y_max', 'y_q25', 'y_q75', 'y_skew', 'y_kurt', 'y_cv'] +
        # z features (9)
        ['z_mean', 'z_std', 'z_min', 'z_max', 'z_q25', 'z_q75', 'z_skew', 'z_kurt', 'z_cv']
    )

    num_features = len(feature_cols)  # Should be 28

    X_sequences = []
    Y_sequences = []  # Minute-level labels
    x_lens = []

    if max_seq_len is not None:
        # Template for padding to max_seq_len with 30 channels
        max_seq_template_X = torch.ones(max_seq_len, num_features).to(device)
        max_seq_template_Y = torch.full((max_seq_len,), -100, dtype=torch.long, device=device)  # Padding label
        X_sequences.append(max_seq_template_X)
        Y_sequences.append(max_seq_template_Y)

    for index, seq in batch.groupby(seg_column, sort=False, observed=True):
        # Filter to only include minute_id < max_seq_len (first 24 hours) and sort by minute_id
        # if 'minute_id' in seq.columns and max_seq_len is not None:
        #     seq = seq[seq['minute_id'] < max_seq_len].sort_values('minute_id')

        # Sanity check: warn if sequence still exceeds max_seq_len
        if max_seq_len is not None and len(seq) > max_seq_len:
            # print(f"WARNING: Segment {index} has {len(seq)} rows after filtering, truncating to {max_seq_len}")
            seq = seq.iloc[:max_seq_len]

        # Input: 30 feature channels
        X_arr = seq.loc[:, feature_cols].to_numpy()

        # Create minute-level labels: 0=other, 1=non-wear, 2=predictTSO
        # Priority: predictTSO > non-wear > other (for each timestep)
        Y_arr = np.zeros(len(seq), dtype=np.int64)

        # Check each minute
        for i in range(len(seq)):
            if seq.iloc[i]['predictTSO']:
                Y_arr[i] = 2
            elif seq.iloc[i]['non-wear'] == 1:
                Y_arr[i] = 1
            # else: remains 0 (other)

        X_sequences.append(torch.tensor(X_arr, dtype=torch.float32, device=device))
        Y_sequences.append(torch.tensor(Y_arr, dtype=torch.long, device=device))
        x_lens.append(len(X_arr))

    # Pad sequences
    pad_X = pad_sequence(X_sequences, batch_first=True, padding_value=padding_value)
    pad_Y = pad_sequence(Y_sequences, batch_first=True, padding_value=-100)  # -100 is ignore_index for CE loss

    if max_seq_len:
        # Discard the max seq template now
        pad_X = pad_X[1:, :, :]
        pad_Y = pad_Y[1:, :]

    return pad_X, pad_Y, torch.tensor(x_lens, dtype=torch.long, device=device)



def generate_time_cyclic(timestamps, use_sincos=True):
    """
    Generate cyclic time encoding from timestamps.

    Args:
        timestamps: Pandas datetime series
        use_sincos: If True, return (sin, cos) pair for unique encoding.
                   If False, return only sin (backward compatible).

    Returns:
        If use_sincos=True: (time_sin, time_cos) - 2D encoding
        If use_sincos=False: time_sin only - 1D encoding (original behavior)

    Note:
        Sin+Cos encoding provides unique representation for all times:
        - 00:00 (midnight) -> (sin=0.0, cos=1.0)
        - 06:00 (morning)  -> (sin=1.0, cos=0.0)
        - 12:00 (noon)     -> (sin=0.0, cos=-1.0)  [Different from midnight!]
        - 18:00 (evening)  -> (sin=-1.0, cos=0.0)
    """
    if not pd.api.types.is_datetime64_any_dtype(timestamps):
        timestamps = pd.to_datetime(timestamps)

    hour = timestamps.dt.hour
    minute = timestamps.dt.minute

    # Time as fraction of day [0, 1)
    time_fraction = (hour + minute / 60) / 24

    # Generate sin component
    time_sin = np.sin(2 * np.pi * time_fraction)

    if use_sincos:
        # Generate cos component for unique 2D encoding
        time_cos = np.cos(2 * np.pi * time_fraction)
        return time_sin.values, time_cos.values
    else:
        # Backward compatible: return only sin
        return time_sin.values


def add_padding_tso_patch(batch, device, seg_column='segment', max_seq_len=1440,
                          patch_size=1200, sampling_rate=20, padding_value=0.0,
                          use_sincos=False, scaler=None):
    """
    Prepare batch data for TSO prediction with patched raw sensor data.

    Each minute is represented as a patch of raw 20Hz samples.
    Uses data from load_data() which contains raw x, y, z accelerometer data.

    IMPORTANT - Scaling Strategy:
    - If using load_data_tso_patch(use_scaler=True): Data is already scaled in cache.
      Set scaler=None here to avoid double-scaling.
    - If using raw unscaled data: Pass scaler parameter to scale per-batch.

    Args:
        batch: DataFrame with columns from load_data:
               - x, y, z: raw accelerometer data at 20Hz (scaled or unscaled)
               - temperature: temperature sensor data
               - timestamp: datetime for each sample
               - non-wear, predictTSO: labels
               - segment: segment identifier
               - minute_id: minute index within segment (0-1439)
        device: torch device
        seg_column: column name for segment identifier
        max_seq_len: maximum sequence length in minutes (default: 1440 = 24h)
        patch_size: samples per minute patch (default: 1200 = 60 seconds * 20Hz)
        sampling_rate: sensor sampling rate in Hz (default: 20)
        padding_value: value to use for padding
        use_sincos: If True, use sin+cos time encoding (6 channels: x,y,z,temp,time_sin,time_cos).
                   If False, use sin only (5 channels: x,y,z,temp,time_sin). Default: False
        scaler: Optional StandardScaler to apply to x, y, z columns.
               - Use ONLY if data is NOT already scaled in load_data_tso_patch()
               - Set to None if load_data_tso_patch(use_scaler=True) was used
               - WARNING: Applying scaler to already-scaled data will corrupt results!

    Returns:
        pad_X: [batch_size, seq_len, patch_size, num_channels] - patched input features
               - If use_sincos=False (default): 5 channels [x, y, z, temperature, time_sin]
               - If use_sincos=True: 6 channels [x, y, z, temperature, time_sin, time_cos]
        pad_Y: [batch_size, seq_len] - minute-level class labels (0=other, 1=non-wear, 2=predictTSO)
        x_lens: [batch_size] - original sequence lengths in minutes

    GPU Memory Estimate:
        - Input shape (5ch): (batch_size, 1440, 1200, 5) - ~33 MB/sample, batch_16 = ~528 MB
        - Input shape (6ch): (batch_size, 1440, 1200, 6) - ~40 MB/sample, batch_16 = ~640 MB

    Examples:
        # Example 1: Data already scaled during loading (RECOMMENDED)
        df = load_data_tso_patch(folder, use_scaler=True)  # Scale once during caching
        pad_X, pad_Y, x_lens = add_padding_tso_patch(
            batch, device, use_sincos=True, scaler=None  # Don't scale again!
        )

        # Example 2: Scale per-batch (for raw unscaled data)
        df = load_data_tso_patch(folder, use_scaler=False)  # Raw data
        scaler = joblib.load('scaler.joblib')
        pad_X, pad_Y, x_lens = add_padding_tso_patch(
            batch, device, use_sincos=True, scaler=scaler  # Scale per-batch
        )
    """
    num_channels = 6 if use_sincos else 5

    X_sequences = []  # Will store [seq_len_minutes, patch_size, channels]
    Y_sequences = []  # Will store [seq_len_minutes] minute-level labels
    x_lens = []  # Length in minutes

    if max_seq_len is not None:
        # Template for padding to max_seq_len minutes
        max_seq_template_X = torch.ones(max_seq_len, patch_size, num_channels, device=device) * padding_value
        max_seq_template_Y = torch.full((max_seq_len,), -100, dtype=torch.long, device=device)
        X_sequences.append(max_seq_template_X)
        Y_sequences.append(max_seq_template_Y)

    for _, seg_metadata in batch.groupby(seg_column, sort=False, observed=True):
        # Load raw data on-demand from cache_file
        if 'cache_file' in seg_metadata.columns:
            # Lazy loading mode: load from cache file
            cache_file = seg_metadata['cache_file'].iloc[0]
            seg_data = pd.read_parquet(cache_file)
        elif 'file_path' in seg_metadata.columns:
            # Legacy: support old file_path column name
            file_path = seg_metadata['file_path'].iloc[0]
            seg_data = pd.read_parquet(file_path)
        else:
            # Direct mode: data already in batch
            seg_data = seg_metadata

        # Sort by timestamp to ensure temporal order
        seg_data = seg_data.sort_values('timestamp').reset_index(drop=True)

        # Apply scaler to x, y, z if provided (IMPORTANT: must match training scaler!)
        if scaler is not None:
            columns_to_scale = ['x', 'y', 'z']
            cols_present = [col for col in columns_to_scale if col in seg_data.columns]
            if cols_present:
                seg_data[cols_present] = scaler.transform(seg_data[cols_present])

        # Limit to max_seq_len minutes if specified
        if max_seq_len is not None and 'minute_id' in seg_data.columns:
            seg_data = seg_data[seg_data['minute_id'] < max_seq_len]

        # Group by minute_id to create patches
        if 'minute_id' not in seg_data.columns:
            # Create minute_id if not present
            seg_data['minute_id'] = seg_data.groupby(seg_column).cumcount() // patch_size

        minute_patches = []
        minute_labels = []

        for _, minute_data in seg_data.groupby('minute_id', sort=True):
            # Extract raw sensor data for this minute
            x_vals = minute_data['x'].values
            y_vals = minute_data['y'].values
            z_vals = minute_data['z'].values
            temp_vals = minute_data['temperature'].values if 'temperature' in minute_data.columns else np.zeros(len(minute_data))

            # Generate time encoding (sin or sin+cos)
            if 'timestamp' in minute_data.columns and len(minute_data) > 0:
                time_encoding = generate_time_cyclic(minute_data['timestamp'], use_sincos=use_sincos)
                if use_sincos:
                    time_sin, time_cos = time_encoding
                else:
                    time_sin = time_encoding
                    time_cos = None
            else:
                # Fallback if no timestamp
                time_sin = np.zeros(len(minute_data))
                time_cos = np.zeros(len(minute_data)) if use_sincos else None

            # Stack into patch based on time encoding method
            if use_sincos:
                # 6 channels: [x, y, z, temperature, time_sin, time_cos]
                patch = np.stack([x_vals, y_vals, z_vals, temp_vals, time_sin, time_cos], axis=1)
            else:
                # 5 channels: [x, y, z, temperature, time_sin]
                patch = np.stack([x_vals, y_vals, z_vals, temp_vals, time_sin], axis=1)

            # Pad or truncate to exactly patch_size samples
            if len(patch) < patch_size:
                # Pad with padding_value
                pad_length = patch_size - len(patch)
                padding = np.full((pad_length, num_channels), padding_value, dtype=np.float32)
                patch = np.vstack([patch, padding])
            elif len(patch) > patch_size:
                # Truncate to patch_size
                patch = patch[:patch_size]

            minute_patches.append(patch)

            # Determine minute-level label (priority: predictTSO > non-wear > other)
            if 'predictTSO' in minute_data.columns and minute_data['predictTSO'].any():
                label = 2  # predictTSO
            elif 'non-wear' in minute_data.columns and (minute_data['non-wear'] == 1).any():
                label = 1  # non-wear
            else:
                label = 0  # other

            minute_labels.append(label)

        # Convert to tensors
        if len(minute_patches) > 0:
            X_arr = np.stack(minute_patches, axis=0)  # [num_minutes, patch_size, 5]
            Y_arr = np.array(minute_labels, dtype=np.int64)  # [num_minutes]

            X_sequences.append(torch.tensor(X_arr, dtype=torch.float32, device=device))
            Y_sequences.append(torch.tensor(Y_arr, dtype=torch.long, device=device))
            x_lens.append(len(minute_patches))

    # Pad sequences to max_seq_len
    # Note: pad_sequence expects list of [seq_len, ...] tensors
    pad_X = torch.nn.utils.rnn.pad_sequence(X_sequences, batch_first=True, padding_value=padding_value)
    pad_Y = torch.nn.utils.rnn.pad_sequence(Y_sequences, batch_first=True, padding_value=-100)

    if max_seq_len is not None:
        # Discard the max seq template
        pad_X = pad_X[1:, :, :, :]
        pad_Y = pad_Y[1:, :]

    return pad_X, pad_Y, torch.tensor(x_lens, dtype=torch.long, device=device)



def random_patch_masking(segments, masking_ratio=0.3, patch_size=10):
    segment_length, num_features = segments.shape
    segments_masked = segments.copy()  # Create a copy to apply masking

    # Calculate the total number of elements to mask based on the masking ratio
    total_elements_to_mask = int(masking_ratio * segment_length)
    
    patch_size= min(patch_size,total_elements_to_mask)
    
    masked_elements = 0
    while masked_elements < total_elements_to_mask:
        # Select random starting index for the patch
        start_index = np.random.randint(0, segment_length - patch_size + 1)
        
        # Apply masking to the selected patch if not already masked
        if np.all(segments_masked[ start_index:start_index + patch_size] != 0):
            segments_masked[ start_index:start_index + patch_size] = 0  # Masking the patch
            masked_elements += patch_size

    return segments_masked


def add_padding_tso_patch_h5(dataset, batch_indices, device, max_seq_len=1440,
                     patch_size=1200, padding_value=0.0, num_channels=None):
    """
    Prepare batch from H5 dataset for TSO prediction.

    This function is analogous to add_padding_tso_patch but works with
    preprocessed H5 data instead of HuggingFace datasets.

    Args:
        dataset: H5Dataset instance (with X, Y, seq_lengths attributes)
        batch_indices: Indices of samples in batch
        device: torch device
        max_seq_len: Maximum sequence length in minutes (default: 1440 = 24h)
        patch_size: Samples per minute (default: 1200 = 60s * 20Hz)
        padding_value: Padding value for X data (default: 0.0)
        num_channels: Number of input channels (5 or 6). If None, read from dataset metadata.

    Returns:
        pad_X: [batch_size, seq_len, patch_size, num_channels] - raw sensor patches
               Channels (5): [x, y, z, temperature, time_sin] OR
               Channels (6): [x, y, z, temperature, time_sin, time_cos]
        pad_Y: [batch_size, seq_len] - minute-level labels (0=other, 1=non-wear, 2=predictTSO)
        x_lens: [batch_size] - sequence lengths in minutes
        batch_samples: List of dicts with segment info

    Note:
        Label aggregation uses np.any() logic to match add_padding_tso_patch (line 3536-3541):
        - If ANY sample in minute has predictTSO=1, label entire minute as predictTSO (2)
        - Else if ANY sample has non-wear=1, label entire minute as non-wear (1)
        - Otherwise label as other (0)
    """

    batch_size = len(batch_indices)
    samples_per_minute = patch_size  # 1200 samples = 1 minute @ 20Hz

    # Get num_channels from dataset metadata if not provided
    if num_channels is None:
        num_channels = dataset.num_channels

    # Get data for all samples in batch
    X_batch = []
    Y_batch = []
    lens_batch = []
    segments_batch = []

    for idx in batch_indices:
        sample = dataset[idx]
        X_batch.append(sample['X'])  # [max_len, num_channels]
        Y_batch.append(sample['Y'])  # [max_len, 2]
        lens_batch.append(sample['seq_length'])
        segments_batch.append({'segment': sample['segment']})

    X_batch = np.stack(X_batch)  # [batch_size, max_len, num_channels]
    Y_batch = np.stack(Y_batch)  # [batch_size, max_len, 2]
    lens_batch = np.array(lens_batch)

    # Convert to patches (minute-level aggregation)
    # X_batch: [batch_size, max_len, num_channels] -> [batch_size, num_minutes, patch_size, num_channels]
    num_minutes_max = min(max_seq_len, X_batch.shape[1] // samples_per_minute)

    pad_X = np.full((batch_size, num_minutes_max, patch_size, num_channels),
                    padding_value, dtype=np.float32)
    pad_Y = np.full((batch_size, num_minutes_max), -100, dtype=np.int64)  # -100 for ignore
    x_lens = np.zeros(batch_size, dtype=np.int64)

    for i in range(batch_size):
        seq_len_samples = lens_batch[i]
        num_minutes = min(num_minutes_max, seq_len_samples // samples_per_minute)

        if num_minutes > 0:
            x_lens[i] = num_minutes

            # Reshape to patches: [seq_len_samples, num_channels] -> [num_minutes, patch_size, num_channels]
            samples_to_use = num_minutes * samples_per_minute
            X_reshaped = X_batch[i, :samples_to_use, :].reshape(num_minutes, patch_size, num_channels)
            pad_X[i, :num_minutes, :, :] = X_reshaped

            # Aggregate labels to minute-level (using np.any logic to match line 3536-3541)
            # Y_batch: [max_len, 2] -> predictTSO and non_wear
            Y_reshaped = Y_batch[i, :samples_to_use, :].reshape(num_minutes, patch_size, 2)

            # For each minute, determine label (priority: predictTSO > non_wear > other)
            # IMPORTANT: Use np.any() to match add_padding_tso_patch behavior
            for m in range(num_minutes):
                minute_predictTSO = Y_reshaped[m, :, 0]  # [patch_size]
                minute_nonwear = Y_reshaped[m, :, 1]  # [patch_size]

                # Use np.any() - if ANY sample in minute has label=1, use that label
                if np.any(minute_predictTSO):
                    pad_Y[i, m] = 2  # predictTSO
                elif np.any(minute_nonwear):
                    pad_Y[i, m] = 1  # non-wear
                else:
                    pad_Y[i, m] = 0  # other

    # Convert to torch tensors
    pad_X = torch.from_numpy(pad_X).to(device)
    pad_Y = torch.from_numpy(pad_Y).to(device)
    x_lens = torch.from_numpy(x_lens).to(device)

    return pad_X, pad_Y, x_lens, segments_batch


# ==================== Prediction Smoothing Functions ====================

