# Scratch Detection with Wearable Sensors - Project Overview

This project implements deep learning models for detecting scratching behavior from 3-channel wearable device signals (accelerometer data: x, y, z). The main focus is on using Mamba (State Space Models) architecture for time series analysis. The codebase has been reorganized into a modular structure supporting two primary research papers: **Deep Scratch** and **Deep TSO**.

# Code Style and Philosophy (Inspired by The Zen of Python)

- **Readability Counts:** Prioritize clear, self-documenting code. Use descriptive variable and function names.
- **Simple is better than complex:** Strive for the simplest solution that effectively addresses the problem. Avoid over-engineering.
- **Explicit is better than implicit:** Make intentions clear in the code. Avoid relying on hidden assumptions or side effects.
- **There should be one-- and preferably only one --obvious way to do it:** Promote consistency in coding patterns and approaches within the project.
- **Errors should never pass silently:** Implement robust error handling and logging to ensure issues are identified and addressed.
- **Beautiful is better than ugly:** Aim for aesthetically pleasing and well-structured code.

## Workflow and Best Practices

- **Testing:** When making changes, prioritize running relevant unit tests. For larger changes, consider the impact on integration tests.
- **Documentation:** Ensure new functionalities or complex logic are adequately documented with comments or docstrings.
- **Refactoring:** Regularly look for opportunities to refactor code for improved clarity, maintainability, and efficiency.
- **Commit Messages:** Write clear and concise commit messages summarizing the changes made.

## Specific Instructions for Claude

- **Refactoring Tasks:** When refactoring, prioritize maintaining existing functionality and test coverage.
- **New Feature Development:** When adding new features, ensure they align with the project's overall architecture and design principles.
- **Debugging:** When debugging, methodically analyze error messages and logs to pinpoint the root cause of issues.

## Project Structure

### Directory Organization

```
project/
├── models/              # All model architectures (split from Helpers/DL_models.py)
│   ├── components.py    # TCN, ResTCN, feature extractors, positional encoding
│   ├── attention.py     # Attention modules, pooling mechanisms
│   ├── mamba_blocks.py  # Mamba state-space blocks and core MBA module
│   ├── resmamba.py      # Primary models: MBA_tsm, MBA_tsm_with_padding, MBA4TSO
│   ├── encoder_decoder.py  # Encoder-decoder variants
│   ├── pretraining.py   # MBATSMForPretraining model
│   ├── pretrainer.py    # Self-supervised pretraining logic (MAE, DINO, RelCon)
│   ├── baselines.py     # RNN, LSTM, GRU, CNN, TCN baselines
│   ├── resnet.py        # 1D ResNet blocks
│   ├── unet.py          # U-Net and EfficientUNet architectures
│   ├── patchtst.py      # PatchTST implementation
│   ├── vit1d.py         # 1D Vision Transformer
│   ├── spectral.py      # Multiscale spectral networks
│   ├── embed.py         # Patch embeddings
│   ├── specialized.py   # PatchTSTHead, ViT, SwinTransformer wrappers
│   ├── conv1d.py        # 1D convolution models
│   └── setup.py         # setup_model() factory function
│
├── data/                # Data loading and processing (split from Helpers/DL_helpers.py)
│   ├── loading.py       # load_folder(), load_csv(), load_h5(), load_parquet()
│   ├── preprocessing.py # Filtering, cleaning, feature extraction
│   ├── padding.py       # Padding and masking functions
│   ├── augmentation.py  # Data augmentation (mixup, jittering, etc.)
│   ├── batching.py      # Batch generation and sampling
│   ├── prepare_data.py  # CSV generation from raw data
│   ├── convert_h5.py    # Parquet to H5 conversion
│   └── transform.py     # Data format transforms
│
├── training/            # Training scripts and configurations (evolved from predict_*.py)
│   ├── train_scratch.py    # Deep Scratch training script
│   ├── train_tso.py        # TSO with aggregated features
│   ├── train_tso_patch.py  # TSO with raw patches
│   ├── train_tso_patch_h5.py # TSO with H5 data (fastest)
│   ├── train_tso_dlrtc.py  # TSO with DLRTC losses
│   ├── pretrain.py         # Self-supervised pretraining script
│   └── configs/            # YAML experiment configurations
│
├── evaluation/          # Metrics and analysis (split from Helpers/DL_helpers.py + test-tools/)
│   ├── metrics.py       # F1, AUC, confusion matrix, learning curves
│   ├── postprocessing.py # Smoothing, TSO enforcement, padding removal
│   ├── prediction_analysis.py # Per-sample and per-subject analysis
│   ├── tso_analysis.py  # TSO-specific analysis
│   └── compare_models.py # Side-by-side model comparison
│
├── losses/              # Loss functions (split from Helpers/DL_helpers.py + dlrtc_losses.py)
│   ├── standard.py      # FocalLoss, GCE, SupCon, multi-task losses
│   └── dlrtc.py         # DLRTC-specific losses
│
├── papers/              # Paper-specific code and writing
│   ├── deep_scratch/    # IMWUT paper 1 experiments and results
│   └── deep_tso/        # IMWUT paper 2 experiments and results
│
├── experiments/         # Harness engineering: configuration management and logging
│   ├── configs/         # YAML experiment definitions
│   └── logs/            # Experiment artifacts and results
│
├── docs/                # Documentation
│   ├── project_overview.md # Consolidated architecture and design
│   ├── deployment.md    # Multi-GPU, distributed training, Domino setup
│   ├── algorithms.md    # Post-processing algorithms
│   └── archive/         # Original markdown files
│
├── utils/               # Shared utilities
│   └── common.py        # EarlyStopping, learning rate schedulers, file utilities
│
├── Helpers/             # Legacy backward-compatibility layer
│   ├── DL_models.py     # Original monolithic file (preserved for reference)
│   ├── DL_helpers.py    # Original monolithic file (preserved for reference)
│   ├── DL_models_shim.py  # Re-exports from models/ for backward compatibility
│   └── DL_helpers_shim.py # Re-exports from data/losses/evaluation/ for backward compatibility
│
├── tests/               # Unit and integration tests
├── requirement-ml.txt   # Python dependencies
└── CLAUDE.md            # This file
```

### Core Training Entry Points

- **`training/train_scratch.py`**: Main training script for Deep Scratch models (replaces `predict_scratch_segment_h5.py`)
- **`training/train_tso_patch_h5.py`**: Optimized TSO training with H5 data (fastest variant)
- **`training/pretrain.py`**: Self-supervised pretraining (DINO, MAE, RelCon)

### Key Model Architectures

#### Primary Models

1. **MBA_tsm**: Original variable-length sequence model with Mamba blocks
2. **MBA_tsm_with_padding**: Fixed-length padded sequences with contrastive learning
3. **MBA4TSO**: Mamba model specialized for Time-Segment-Of-Interest tasks
4. **PatchTST**: Transformer-based time series model with patch embeddings
5. **EfficientUNet**: U-Net architecture for sequence-to-sequence tasks
6. **MBA_encoder_decoder**: Encoder-decoder architecture for latent-space learning

#### Model Components (in `models/`)

- **FeatureExtractor** (`components.py`): TCN/ResTCN-based feature extraction with positional encoding
- **MambaBlock** (`mamba_blocks.py`): Mamba state-space module with selective scanning
- **AttModule** (`attention.py`): Multi-head attention mechanisms
- **MultiHeadAttentionPooling** (`attention.py`): Sequence aggregation via attention
- **PatchEmbedding** (`embed.py`): Converts raw signals to patch embeddings
- **Positional Encoding** (`components.py`): Learnable or sinusoidal position information

## Data Pipeline

### Input Data Format

- **Raw signals**: 3-channel accelerometer data (x, y, z)
- **Sampling rate**: Variable, typically high-frequency wearable sensor data
- **Labels**:
  - `label1`: Binary scratch/no-scratch classification
  - `label2`: Per-timestep scratch intensity (0-1)
  - `label3`: Scratch event severity/type classification

### Data Loading Methods

Use functions from `data/loading.py`:

1. **Folder-based**: `load_folder()` - Direct loading from folder structure
2. **CSV-based**: `load_csv()` - Pre-processed data from `prepare_data.py`
3. **H5 format**: `load_h5()` - Compressed data storage with metadata (fastest)
4. **Parquet format**: `load_parquet()` - Arrow-based columnar storage

**Recommendation**: Use H5 format (`load_h5()`) for training experiments on large datasets.

### Preprocessing Pipeline

Functions from `data/preprocessing.py`:

1. **Standardization**: Z-score normalization using sklearn StandardScaler
2. **Filtering**: Band-pass or low-pass filtering for noise reduction
3. **Feature extraction**: Optional derived signal features (magnitude, derivatives)
4. **TSO filtering**: Time-segment-of-interest selection and filtering

### Padding and Masking

From `data/padding.py`:

1. **Fixed-length padding**: Configurable padding values (-999, -10, 0)
2. **Padding strategies**: Random vs tail padding position
3. **Mask generation**: Create attention masks from padding indicators
4. **Mask interpolation**: Resize masks to match feature-extracted sequences

### Data Augmentation

From `data/augmentation.py`:

1. **Mixup**: Latent space mixing for regularization
2. **Time warping**: Stretch/compress time dimension
3. **Jittering**: Add small Gaussian noise
4. **Rotation**: Rotate signal in 3D accelerometer space

## Critical Implementation Details

### Padding Value Handling

**IMPORTANT**: The project uses extreme padding values (like -999) for clear padding detection, but these must be handled carefully:

```python
# In models/resmamba.py MBA_tsm_with_padding.forward():
# 1. Use original data for mask creation
padding_mask = ~torch.any(x_original == padding_value, dim=-1)

# 2. Replace extreme padding values before BatchNorm layers
if abs(padding_value) > 100:
    x_for_features = torch.where(padding_mask_for_replacement, 0.0, x_for_features)
```

**Reason**: BatchNorm1d layers are sensitive to outliers. Extreme padding values corrupt batch statistics and degrade performance significantly.

### Sequence Length Management

Models handle variable sequence lengths through:

1. **Feature extraction changes**: TCN layers in `components.py` may change sequence length via dilations
2. **Mask interpolation**: `data/padding.py` provides utilities to resize masks to feature-extracted sequences
3. **Output interpolation**: Interpolate back to original length for consistency with label dimensions

### Model Training Configuration

#### Key Parameters

- **Batch size**: Typically 16-64 (adjust for GPU memory)
- **Learning rates**: 1e-4 to 1e-3 with cosine annealing
- **Optimizers**: AdamW, RMSprop, RAdan
- **Loss weights**: Multi-task learning with configurable weights (wl1, wl2, wl3)
- **Padding value**: -999 (default) or 0 depending on standardization

#### Training Strategies

- **LOFO (Leave-One-Fold-Out)**: Cross-validation strategy
- **LOSO (Leave-One-Subject-Out)**: Subject-independent evaluation
- **Mixup augmentation**: Latent space mixup for regularization
- **Contrastive learning**: Self-supervised pretraining and supervised contrastive loss
- **Early stopping**: Monitor validation metrics with patience

## Harness Engineering and Experiment Management

### Experiment Configuration System

Located in `experiments/configs/`, YAML files define reproducible experiments:

```yaml
experiment_name: "scratch_mba_with_padding_v1"
model:
  architecture: "mba_with_padding"
  feature_extractor_type: "tcn"
  padding_value: -999
  contrastive_loss_weight: 0.1

training:
  batch_size: 32
  learning_rate: 1e-3
  epochs: 100
  optimizer: "adamw"

data:
  format: "h5"
  path: "/data/scratch_data.h5"
  split_strategy: "lofo"
```

### Reproducibility and Logging

- **Experiment tracking**: All runs logged to `experiments/logs/`
- **Config snapshots**: YAML config saved with each run
- **Metrics tracking**: Per-epoch F1, accuracy, validation loss
- **Checkpointing**: Best model weights saved based on validation metric

### Multi-GPU and Distributed Training

Supported through `training/train_scratch.py`:
- Automatic distributed data parallel (DDP) on multi-GPU setups
- Batch size scaling with number of GPUs
- Gradient accumulation for effective larger batches
- See `docs/deployment.md` for Domino and distributed setup

## Model Performance Considerations

### Common Issues and Solutions

1. **Padding value inconsistency**

   - **Problem**: Different padding values (-999 vs 0) cause large performance gaps
   - **Solution**: Use consistent padding values throughout pipeline and `data/padding.py`
2. **Mask generation errors**

   - **Problem**: Feature extractor changes sequence length but masks don't match
   - **Solution**: Use `data/padding.py` mask interpolation utilities
3. **BatchNorm corruption**

   - **Problem**: Extreme padding values (-999) corrupt normalization statistics
   - **Solution**: Replace with neutral values (0) before feature extraction while preserving mask
4. **Model signature mismatches**

   - **Problem**: Different models expect different parameter signatures
   - **Solution**: Use `models/setup.py` factory function `setup_model()` for consistency

### Performance Optimization

1. **Data format**: Use H5 (fastest) > CSV > Folder
2. **Batch size**: Larger batches (32-64) for better gradient estimates, if GPU memory allows
3. **Sequence length**: Shorter sequences (256-512) train faster than very long sequences (2000+)
4. **Mixed precision**: Enable automatic mixed precision (AMP) for ~2x speedup
5. **Gradient checkpointing**: Enable for large models to reduce memory usage

## Development Workflow

### Testing and Validation

1. **Unit tests**: `tests/` directory for component-level tests
2. **Parameter tuning**: Use Optuna for hyperparameter optimization
3. **Cross-validation**: LOFO/LOSO strategies for robust evaluation
4. **Metrics tracking**: F1-score, AUC, precision, recall (in `evaluation/metrics.py`)
5. **Attention visualization**: Analyze model attention patterns (in `evaluation/`)

### Common Commands

```bash
# Training Deep Scratch with H5 data
python training/train_scratch.py --config experiments/configs/scratch_mba_v1.yaml

# TSO training with patches
python training/train_tso_patch_h5.py --data /path/to/data.h5 --model mba4tso

# Self-supervised pretraining
python training/pretrain.py --config experiments/configs/pretrain_dino.yaml

# Hyperparameter tuning
python training/train_scratch.py --config experiments/configs/scratch_mba_v1.yaml --tune

# Evaluation and analysis
python -c "from evaluation.metrics import *; analyze_results('results/experiment_1')"
```

### Import Patterns for Modular Code

**New modular imports** (preferred):

```python
from models.resmamba import MBA_tsm_with_padding
from models.setup import setup_model
from data.loading import load_h5
from data.padding import create_padding_mask
from losses.standard import SupConLoss
from evaluation.metrics import compute_f1
from utils.common import EarlyStopping
```

**Legacy imports** (still supported via shims):

```python
# These still work via Helpers/DL_models_shim.py and DL_helpers_shim.py
from Helpers.DL_models import MBA_tsm_with_padding
from Helpers.DL_helpers import load_h5, compute_f1
```

## Migration Guide: Old → New Import Mappings

### Model Imports

| Old (Helpers/DL_models.py) | New (models/) |
|---|---|
| `from Helpers.DL_models import MBA_tsm` | `from models.resmamba import MBA_tsm` |
| `from Helpers.DL_models import MBA_tsm_with_padding` | `from models.resmamba import MBA_tsm_with_padding` |
| `from Helpers.DL_models import EfficientUNet` | `from models.unet import EfficientUNet` |
| `from Helpers.DL_models import PatchTST` | `from models.patchtst import PatchTST` |
| `from Helpers.DL_models import FeatureExtractor` | `from models.components import FeatureExtractor` |
| `from Helpers.DL_models import MambaBlock` | `from models.mamba_blocks import MambaBlock` |
| All models | `from models.setup import setup_model` (factory) |

### Data and Helper Imports

| Old (Helpers/DL_helpers.py) | New (data/, losses/, evaluation/, utils/) |
|---|---|
| `from Helpers.DL_helpers import load_h5` | `from data.loading import load_h5` |
| `from Helpers.DL_helpers import load_csv` | `from data.loading import load_csv` |
| `from Helpers.DL_helpers import create_padding_mask` | `from data.padding import create_padding_mask` |
| `from Helpers.DL_helpers import apply_mixup` | `from data.augmentation import mixup` |
| `from Helpers.DL_helpers import SupConLoss` | `from losses.standard import SupConLoss` |
| `from Helpers.DL_helpers import compute_f1` | `from evaluation.metrics import compute_f1` |
| `from Helpers.DL_helpers import EarlyStopping` | `from utils.common import EarlyStopping` |
| `from Helpers.net.prepare_data import prepare_data` | `from data.prepare_data import prepare_data` |
| `from Helpers.net.pretrainer import MAEPretrainer` | `from models.pretrainer import MAEPretrainer` |

### Training Script Evolution

| Old | New |
|---|---|
| `predict_scratch_segment_h5.py` | `training/train_scratch.py` |
| `predict_scratch_segment.py` | `training/train_scratch.py` (with `--format csv`) |
| `Helpers/net/prepare_data.py` | `data/prepare_data.py` |
| Inline config dicts | `experiments/configs/` YAML files |

## Troubleshooting Guide

### Performance Issues

- **Low F1 scores**: Check padding value consistency using `data/padding.py`, verify mask generation
- **Training instability**: Verify BatchNorm isn't corrupted by extreme values (see Critical Implementation Details)
- **Memory issues**: Reduce batch size or sequence length in experiment config
- **Slow training**: Use H5 data format and enable mixed precision (AMP)

### Common Errors

- **Shape mismatches**: Usually related to sequence length changes in `models/components.py` feature extraction
- **Parameter errors**: Check model signature via `models/setup.py` factory function
- **Data loading failures**: Verify file paths and use appropriate loader from `data/loading.py`
- **Import errors**: Use modular imports from `models/`, `data/`, `losses/`, etc. (see Import Patterns)

### Best Practices

1. **Always use consistent padding values** throughout pipeline (set once in config)
2. **Monitor attention weights** to verify model is focusing on relevant regions (visualization in `evaluation/`)
3. **Use contrastive learning** for better representations (pretraining via `training/pretrain.py`)
4. **Apply proper data augmentation** to prevent overfitting (from `data/augmentation.py`)
5. **Validate on multiple subjects** to ensure generalization (LOSO cross-validation)
6. **Version experiments** using `experiments/configs/` and track in logs
7. **Use H5 data format** for faster I/O on large datasets

## Model Architecture Notes

### MBA_tsm_with_padding Key Features

- **Fixed-length sequences**: Handles padded input uniformly via `data/padding.py`
- **Padding-aware processing**: Replaces extreme padding values before BatchNorm (see Critical Implementation Details)
- **Contrastive learning**: Supervised contrastive loss from `losses/standard.py` with projection head
- **Mixup augmentation**: Latent space mixing from `data/augmentation.py` for regularization
- **Multi-task outputs**: Classification + sequence labeling + severity prediction via multiple heads
- **Attention pooling**: Multi-head self-attention from `attention.py` for sequence aggregation

### Feature Extraction Pipeline

Implemented in `models/components.py` and used by primary models:

1. **Input**: `[Batch, SeqLen, Channels]` → permute to `[Batch, Channels, SeqLen]`
2. **Padding mask generation**: `data/padding.py` creates attention masks from padding indicators
3. **TCN/ResTCN layers** (`components.py`): Dilated convolutions with BatchNorm and ReLU
4. **Positional encoding** (`components.py`): Optional learnable or sinusoidal position information
5. **Mamba/Attention blocks** (`mamba_blocks.py`, `attention.py`): State space or attention mechanisms
6. **Mask interpolation**: `data/padding.py` resizes masks if feature extraction changes sequence length
7. **Pooling** (`attention.py`): Attention-based or CLS token pooling
8. **Output heads**: Multiple prediction heads for different tasks

### Post-Processing and TSO Enforcement

Functions in `evaluation/postprocessing.py`:

- **Smoothing**: Apply temporal smoothing to predictions
- **TSO enforcement**: Ensure time-segment-of-interest constraints
- **Padding removal**: Strip padded regions from final outputs
- **Aggregation**: Combine per-timestep predictions into segment-level scores

## Paper-Specific Experiments

### Deep Scratch (IMWUT Paper 1)

Located in `papers/deep_scratch/`, focuses on:
- Binary scratch classification across subjects
- Multi-task learning with intensity and severity labels
- Contrastive pretraining strategies
- Comparison with RNN, CNN, and Transformer baselines

### Deep TSO (IMWUT Paper 2)

Located in `papers/deep_tso/`, extends Deep Scratch with:
- Time-Segment-Of-Interest (TSO) constraints
- Hierarchical loss functions for temporal coherence
- Cross-dataset evaluation
- Real-world deployment considerations

---

This documentation should be updated as the project evolves and new insights are gained. When in doubt, refer to specific module docstrings and examples in `papers/` directories.
