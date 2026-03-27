# Project Overview: Deep Learning for TSO and Scratch Detection

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Core Components](#core-components)
4. [Data Pipeline](#data-pipeline)
5. [Model Architecture](#model-architecture)
6. [Training Framework](#training-framework)
7. [Getting Started](#getting-started)
8. [Advanced Topics](#advanced-topics)

---

## Introduction

This project implements state-of-the-art deep learning models for:
- **TSO (Time Spent Outside) prediction** from wearable accelerometer data
- **Scratch detection** from 3-channel sensor signals (x, y, z)

The framework supports both traditional supervised learning and advanced techniques like Dynamic Label Refinement with Temporal Consistency (DLR-TC) for learning from noisy labels.

### Key Features

- **Multiple model architectures**: MBA4TSO, PatchTST, EfficientUNet, Mamba-based models
- **Flexible data handling**: Parquet and H5 formats with memory-mapped access
- **Label refinement**: DLR-TC framework for handling noisy sensor annotations
- **Post-processing**: Prediction smoothing and single TSO period enforcement
- **Scalable training**: Single GPU to multi-node distributed training

---

## Project Structure

```
JNJ/
├── Helpers/
│   ├── DL_models.py              # Model architectures
│   ├── DL_helpers.py             # Utilities: data loading, loss functions
│   ├── dlrtc_losses.py           # DLR-TC loss functions
│   ├── net/
│   │   ├── pretrainer.py         # Self-supervised pretraining (DINO, RelCon)
│   │   └── prepare_data.py       # Data preparation and CSV generation
│
├── Training Scripts
│   ├── predict_TSO_segment_patch.py          # Baseline TSO training
│   ├── predict_TSO_segment_patch_dlrtc.py    # TSO training with DLR-TC
│   ├── predict_scratch_segment_h5.py         # Scratch detection with H5 data
│   ├── predict_scratch_segment.py            # Scratch detection (variable-length)
│   ├── test_dlrtc_implementation.py          # DLR-TC validation tests
│
├── Documentation
│   ├── CLAUDE.md                  # Project guidelines and philosophy
│   └── docs/
│       ├── project_overview.md    # This file
│       ├── deployment.md          # Deployment and distributed training
│       ├── algorithms.md          # Post-processing algorithms
│       └── archive/               # Original detailed docs
│
└── Configuration
    ├── DLRTC_README.md (archived)
    ├── DLRTC_QUICKSTART.md (archived)
    └── ... (other reference docs)
```

---

## Core Components

### 1. Data Pipeline

#### Input Formats

**Parquet Format (Raw Sensor Data)**
- Directory structure: `/raw/` folders containing CSV or parquet files
- Format: Time-indexed accelerometer data (x, y, z channels)
- Preprocessing: On-the-fly standardization with sklearn StandardScaler

**H5 Format (Preprocessed Data)**
- Compressed HDF5 files with metadata
- Pre-scaling: Z-score normalization included
- Memory-mapped access: Efficient loading of 900GB+ datasets
- Metadata: Sampling rate, channel count, sequence lengths

#### Data Preparation

```python
# Convert raw parquet to H5 (one-time)
python Helpers/net/prepare_data.py \
    --input_folder /path/to/raw/ \
    --output_h5 /path/to/data.h5 \
    --use_sincos  # Optional: 6 channels with sin+cos encoding

# Or use in training directly
python predict_TSO_segment_patch.py \
    --input_data_folder /path/to/raw/ \
    --output results/
```

#### Preprocessing Steps

1. **Data loading**: From parquet or H5 files
2. **Standardization**: Z-score normalization using StandardScaler
3. **Padding**: Fixed-length sequences with configurable padding values (-999, -10, or 0)
4. **Time encoding**: Optional sin/cos cyclic features for temporal information
5. **Augmentation**: Mixup for latent space regularization

### 2. Core Models

#### Primary Architectures

**MBA_tsm (Mamba Time Series Model)**
- Variable-length sequence processing
- State Space Model (Mamba) blocks for efficient sequence modeling
- Attention pooling for feature aggregation
- Multi-task outputs: classification + sequence labeling + severity

**MBA_tsm_with_padding**
- Fixed-length sequence variant
- Contrastive learning with projection head
- Supervised contrastive loss with mixup
- Mask handling for padded sequences

**PatchTST (Transformer-based)**
- Patch-based time series transformer
- Self-supervised pre-training capability
- Attention mechanisms for long-range dependencies

**EfficientUNet**
- U-Net architecture for sequence-to-sequence tasks
- Encoder-decoder with skip connections
- Efficient convolution blocks

#### Model Components

- **FeatureExtractor**: TCN/ResTCN with BatchNorm and ReLU
- **AttModule**: Attention blocks for feature importance
- **Mamba blocks**: State space models for sequential processing
- **Multi-head attention pooling**: Weighted sequence aggregation
- **Projection heads**: For contrastive learning

### 3. Loss Functions and Training

#### Standard Training Loss

```python
# Classification cross entropy
CE(predictions, labels)

# With temporal smoothness
CE(predictions, labels) + β·TV(predictions)
```

#### DLR-TC Framework (Dynamic Label Refinement with Temporal Consistency)

**Three-component loss function**:

```
L_total = L_GCE(f_θ(x), ŷ) + α·L_Compat(ŷ, y_noisy) + β·L_Temp(ŷ)
```

Where:
- **L_GCE**: Generalized Cross Entropy (robust to noisy labels, q=0.7 default)
- **L_Compat**: KL-divergence anchor to original labels (prevents drift)
- **L_Temp**: Total Variation regularization (enforces temporal smoothness)

**Two-Phase Training**:
1. **Warmup (5-10 epochs)**: Standard cross entropy on noisy labels
2. **Joint Refinement (15-30 epochs)**: Alternating label and model updates

---

## Data Pipeline

### Input Data Format

```python
# For parquet mode
/raw/
├── segment_id_1.parquet  # [time_steps, 3] or [time_steps, 5-6]
├── segment_id_2.parquet
└── ...

# For H5 mode
data.h5:
├── /segments/segment_id_1  # [time_steps, num_channels]
├── /segments/segment_id_2
└── metadata/
    ├── sampling_rate: 20 Hz
    ├── num_channels: 5
    └── max_sequence_length: 1440
```

### Padding Strategy

**Critical Implementation Detail**: The project uses extreme padding values (like -999) for clear detection but handles them carefully:

```python
# In feature extraction
padding_mask = ~torch.any(x_original == padding_value, dim=-1)

# Replace extreme values before BatchNorm (sensitive to outliers)
if abs(padding_value) > 100:
    x_for_features = torch.where(padding_mask_for_replacement, 0.0, x_for_features)
```

### Data Augmentation

- **Mixup**: Latent space mixing for regularization
- **Time warping**: Optional temporal stretching
- **Noise injection**: Gaussian noise for robustness
- **Contrastive learning**: Supervised contrastive loss with positive/negative pairs

---

## Model Architecture

### Forward Pass Pipeline

```
Input [batch, seq_len, channels]
    ↓
Positional Encoding (optional)
    ↓
Feature Extraction (TCN/ResTCN)
    [Sequence length may change]
    ↓
Mask Interpolation (resize to new length)
    ↓
Mamba/Attention Blocks
    ↓
Attention-based Pooling
    ↓
Projection Head (for contrastive learning)
    ↓
Classification Head
    ↓
Output [batch, num_classes]
```

### Multi-Task Learning

Models support multiple objectives:

```python
task_weights = {
    'label1': 1.0,  # Binary scratch/no-scratch classification
    'label2': 0.5,  # Per-timestep scratch intensity
    'label3': 0.3,  # Scratch event severity/type
}

total_loss = Σ weight_i * loss_i(output_i, label_i)
```

---

## Training Framework

### Standard Training Loop

```python
# Single GPU
python predict_TSO_segment_patch.py \
    --input_data_folder /path/to/data/raw \
    --output results_folder \
    --model mba4tso_patch \
    --epochs 50 \
    --num_gpu 0

# Multi-GPU (DataParallel)
python predict_TSO_segment_patch.py \
    --input_data_folder /path/to/data/raw \
    --output results_folder \
    --num_gpu 0,1,2,3 \
    --multi_gpu
```

### DLR-TC Training

```python
# H5 mode (recommended for large datasets)
python predict_TSO_segment_patch_dlrtc.py \
    --input_h5 /path/to/data.h5 \
    --output dlrtc_results \
    --model mba4tso_patch

# Parquet mode with cross-validation
python predict_TSO_segment_patch_dlrtc.py \
    --input_data_folder /path/to/data/raw \
    --output dlrtc_results \
    --testing LOFO  # or LOSO
```

### Hyperparameter Configuration

**DLR-TC Parameters**:

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| warmup_epochs | 5-10 | 3-15 | Initialization quality |
| joint_epochs | 15-30 | 10-50 | Label refinement iterations |
| gce_q | 0.7 | 0.5-0.9 | Noise robustness (lower=more robust) |
| alpha | 0.1 | 0.05-0.5 | Anchor strength to original labels |
| beta | 0.5 | 0.2-1.0 | Temporal smoothness |
| label_lr | 0.01 | 0.001-0.05 | Label update speed |

**Model Parameters**:

```python
best_params = {
    'batch_size': 24,
    'lr': 0.001,
    'optimizer': 'rmsprop',  # or 'adamw', 'radam'
    'scheduler': 'cosine_annealing_warm_restarts',
    'n_resnet_blocks': 4,
    'd_model': 128,
    'n_mamba_layers': 4,
    'dropout': 0.2,
    'patch_size': 1200,  # 60 seconds at 20Hz
    'use_sincos': False,  # 5 channels (x,y,z,temp,time)
}
```

### Epoch Planning for Large Datasets

**For 34.6K segments (900GB)**:

| Phase | Duration | Purpose | Expected Progress |
|-------|----------|---------|-------------------|
| **Warmup** | 10 epochs | Learn clean patterns | F1: 0.68-0.72 |
| **Joint** | 25 epochs | Refine noisy labels | F1: 0.72-0.82 |
| **Total** | 35 epochs | ~45-55 hours (GPU) | Production-ready model |

---

## Getting Started

### Quick Start: 5 Minute Setup

```bash
# 1. Validate installation
cd /path/to/JNJ
python test_dlrtc_implementation.py

# 2. Run first training (basic)
python predict_TSO_segment_patch_dlrtc.py \
    --input_h5 /path/to/data.h5 \
    --output first_run \
    --model mba4tso_patch

# 3. Check results
import joblib
results = joblib.load('predictions/dlrtc_results_split_0_iter_0.joblib')
print(f"Test F1: {results['test_metrics']['f1_avg']:.4f}")
```

### Understanding Output Structure

```
results/DL/UKB_v2/experiment_name/training/
├── model_weights/
│   ├── best_model_split_0_iter_0.pt
│   └── soft_labels_split_0_iter_0.pt
├── predictions/
│   ├── dlrtc_results_split_0_iter_0.joblib
│   └── dlrtc_results_split_0_iter_0.json
├── learning_plots/
│   └── dlrtc_training_history_split_0_iter_0.png
└── debug_predictions/
    ├── train/
    └── val/
```

### Results Format

Results are saved in both JSON (human-readable) and Joblib (Python-efficient) formats:

```json
{
  "test_metrics": {
    "accuracy": 0.8234,
    "f1_avg": 0.8156,
    "f1_tso": 0.8501
  },
  "history": {
    "train_loss": [...],
    "val_loss": [...],
    "train_loss_gce": [...],
    "train_loss_compat": [...],
    "train_loss_temp": [...]
  },
  "dlrtc_config": {
    "warmup_epochs": 10,
    "joint_epochs": 25,
    "gce_q": 0.7,
    "alpha": 0.1,
    "beta": 0.5,
    "label_lr": 0.01
  },
  "num_refined_labels": 27680
}
```

---

## Advanced Topics

### Post-Processing: Prediction Smoothing

**5 Methods Available**:

1. **Majority Vote** (Recommended): Sliding window voting
2. **Median Filter**: Remove isolated spikes
3. **Moving Average**: Smooth probability distributions
4. **Gaussian Weighted**: Emphasize nearby predictions
5. **Minimum Segment Length**: Filter out brief predictions

```bash
python predict_TSO_segment_patch.py \
    --input_data_folder /path/to/data \
    --output results \
    --smooth_predictions \
    --smooth_method majority_vote \
    --smooth_window 5  # minutes
```

**Expected improvement**: +2-8% accuracy depending on prediction stability

### Single TSO Period Enforcement

**Problem**: Model predicts multiple fragmented TSO periods per night

**Solution**: Hybrid approach with training + post-processing

```python
# 1. Train with continuity loss
best_params['continuity_weight'] = 0.1

# 2. Post-process during inference
predictions = batch_enforce_single_tso(
    predictions,
    x_lens,
    min_gap_minutes=30,
    min_duration_minutes=10
)
```

**Expected improvement**: +7-12% F1 for TSO prediction

### Self-Supervised Pretraining

**Available methods**:
- **DINO** (Knowledge Distillation without Labels)
- **RelCon** (Relative Contrastive Learning)

```bash
python predict_scratch_segment_h5.py \
    --pretraining True \
    --pretraining_method DINO \
    --data /path/to/data.h5
```

### Cross-Validation Strategies

**LOFO** (Leave-One-Fold-Out): k-fold splits

```bash
python predict_TSO_segment_patch.py \
    --input_data_folder /path/to/data \
    --testing LOFO \
    --num_folds 5
```

**LOSO** (Leave-One-Subject-Out): Subject-independent evaluation

```bash
python predict_TSO_segment_patch.py \
    --input_data_folder /path/to/data \
    --testing LOSO
```

---

## Common Issues and Solutions

### Issue: Labels not refining (DLR-TC)

**Symptoms**: `loss_compat` near zero, labels unchanged

**Solutions**:
1. Increase `label_lr` (try 0.05)
2. Decrease `alpha` (try 0.05)
3. Check gradient magnitudes: `print(label_grad.abs().mean())`

### Issue: Predictions too smooth

**Symptoms**: Missing short TSO events, low recall

**Solutions**:
1. Decrease `beta` (try 0.2-0.3)
2. Reduce `smooth_window` for post-processing
3. Lower `min_duration_minutes` in single TSO enforcement

### Issue: Memory errors with large datasets

**Solutions**:
1. Use H5 format (50% less RAM than parquet)
2. Reduce batch size
3. Use distributed training (see deployment guide)

### Issue: Training stalls or converges slowly

**Solutions**:
1. Verify data format consistency
2. Check padding value handling
3. Adjust learning rate scheduler
4. Ensure no GPU memory leaks

---

## Performance Benchmarks

### Expected Results

**Baseline (standard training)**:
- Accuracy: ~82-84%
- F1 score (average): ~0.80-0.82
- F1 (TSO): ~0.65-0.70
- Training time: 50-60 hours (single GPU)

**With DLR-TC** (on noisy labels):
- Accuracy: +2-5% improvement
- F1 score: +3-7% improvement
- Temporal consistency: 40-60% jitter reduction
- Additional training time: ~5-10 hours

**With post-processing**:
- Accuracy: +2-8% improvement (on top of base)
- Temporal consistency: Significantly improved

### Scaling Performance

| Setup | GPUs | Time/Epoch | Time/35epochs | Speedup |
|-------|------|-----------|---------------|---------|
| Single GPU | 1 | ~45 min | ~26 hours | 1.0x |
| Multi-GPU | 4 | ~13 min | ~7.6 hours | 3.4x |
| DataParallel | 4 | ~12 min | ~7 hours | 3.7x |
| Ray (2 nodes) | 8 | ~7 min | ~4 hours | 6.5x |
| Ray (4 nodes) | 16 | ~4 min | ~2.3 hours | 11.3x |

---

## References and Citation

If using this framework in research, please cite foundational works:

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

## Next Steps

1. **Quick validation**: Run `test_dlrtc_implementation.py`
2. **First training**: Start with H5 mode for efficiency
3. **Compare approaches**: Test baseline vs DLR-TC vs post-processing
4. **Hyperparameter tuning**: Use smaller epoch configs for rapid iteration
5. **Production deployment**: Use distributed training for large-scale runs

See `deployment.md` for multi-GPU and cloud deployment instructions.
