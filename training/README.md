# Training Scripts

This directory contains all training and data preprocessing scripts for scratch detection and TSO (Time Segment of Interest) prediction models.

## Directory Structure

```
training/
├── __init__.py                    # Package initialization
├── README.md                      # This file
├── configs/
│   └── sample_config.yaml        # Sample configuration with all hyperparameters
├── train_scratch.py              # Scratch detection training (variable-length sequences)
├── train_scratch_h5.py           # Scratch detection training (H5 preprocessed data)
├── train_tso.py                  # TSO status prediction (aggregated features)
├── train_tso_patch.py            # TSO prediction (raw sensor patches, parquet)
├── train_tso_patch_h5.py         # TSO prediction (raw sensor patches, H5)
├── train_tso_dlrtc.py            # TSO with Dynamic Label Refinement (DLR-TC)
├── pretrain.py                   # Self-supervised pretraining (DINO, AIM, RelCon, SimCLR)
└── convert_h5.py                 # Convert parquet files to H5 format
```

## Scripts Overview

### Data Preprocessing

#### `convert_h5.py`
Converts raw parquet files to preprocessed H5 format for fast training.

**Features:**
- Applies StandardScaler normalization
- Generates time-cyclic encoding (sin/cos)
- Supports single or dual-folder data loading
- Optional folder balancing for equal representation
- Memory-efficient incremental processing

**Usage:**
```bash
python training/convert_h5.py \
    --input_folder /path/to/parquet \
    --output_h5 /path/to/data.h5 \
    --scaler_path /path/to/scaler.joblib \
    --max_seq_length 86400 \
    --use_sincos True
```

### Scratch Detection Models

#### `train_scratch.py`
Main training script for scratch detection using variable-length sequences.

**Features:**
- Supports variable-length input sequences
- Multi-task learning: binary classification + per-timestep intensity
- Configurable model architectures (MBA_tsm, PatchTST, EfficientUNet)
- LOFO/LOSO cross-validation strategies
- Contrastive learning support
- Hyperparameter tuning with Optuna

**Usage:**
```bash
python training/train_scratch.py \
    --input_data_folder /path/to/data \
    --model MBA_tsm_with_padding \
    --output results/ \
    --epochs 200 \
    --batch_size 32 \
    --execution_mode train
```

**Key Arguments:**
- `--execution_mode`: train, tune, or test
- `--model`: Model architecture to use
- `--testing`: LOFO or LOSO cross-validation
- `--pretraining`: Enable self-supervised pretraining
- `--pretrained_feature_extractor_path`: Path to pretrained weights

#### `train_scratch_h5.py`
Scratch detection training with H5 preprocessed data (10-100x faster).

**Advantages over parquet:**
- No per-batch file I/O or decompression
- Repeated preprocessing done once during H5 creation
- Direct memory-mapped access to preprocessed data
- Significantly faster training iterations

**Usage:**
```bash
python training/train_scratch_h5.py \
    --input_h5_file /path/to/data.h5 \
    --model MBA_tsm_with_padding \
    --output results/ \
    --epochs 200
```

### TSO (Time Segment of Interest) Prediction

#### `train_tso.py`
TSO status prediction using aggregated feature vectors.

**Predicts 3-class status:**
- `0`: other (normal activity)
- `1`: non-wear (device not worn)
- `2`: predictTSO (time segment of interest)

**Features:**
- Per-minute predictions from 29-D feature vectors
- Configurable TSO models (MBA4TSO variants)
- Batch generation with TSO undersampling
- Continuity-aware loss functions

**Usage:**
```bash
python training/train_tso.py \
    --input_data_folder /path/to/data \
    --model MBA4TSO \
    --output results/ \
    --epochs 200
```

#### `train_tso_patch.py`
TSO prediction using raw sensor patches (20Hz accelerometer data).

**Input Format:**
- Raw 3-axis accelerometer at 20Hz
- Aggregated into 1-minute patches
- Additional channels: temperature, time encoding

**Usage:**
```bash
python training/train_tso_patch.py \
    --input_data_folder /path/to/parquet \
    --model MBA4TSO_Patch \
    --output results/ \
    --epochs 200
```

#### `train_tso_patch_h5.py`
Fast TSO training with H5 preprocessed data (recommended).

**Key Benefits:**
- 10-100x faster data loading
- All preprocessing done once during H5 creation
- Optimized memory usage
- Direct segment access without file scanning

**Usage:**
```bash
python training/train_tso_patch_h5.py \
    --input_h5_file /path/to/data.h5 \
    --model MBA4TSO_Patch \
    --output results/ \
    --epochs 200 \
    --batch_size 32
```

#### `train_tso_dlrtc.py`
TSO prediction with Dynamic Label Refinement using Temporal Consistency (DLR-TC).

**Advanced Features:**
- Robust learning from noisy labels
- Two-phase training: warmup → joint refinement
- Generalized Cross Entropy (GCE) loss
- Temporal smoothness constraints
- Soft label evolution during training

**Useful for:**
- Datasets with label noise or uncertainty
- Requiring physically plausible predictions
- Multi-rater disagreement scenarios

**Usage:**
```bash
python training/train_tso_dlrtc.py \
    --input_h5_file /path/to/data.h5 \
    --model MBA4TSO_Patch \
    --output results/ \
    --epochs 200 \
    --warmup_epochs 50
```

**DLR-TC Specific Arguments:**
- `--warmup_epochs`: Initial training phase duration
- `--label_refine_freq`: Update soft labels every N epochs
- `--temporal_weight`: Weight of temporal smoothness loss
- `--gce_q`: Parameter q for Generalized Cross Entropy

### Self-Supervised Pretraining

#### `pretrain.py`
Pretraining feature extractors using self-supervised learning.

**Supported Methods:**
1. **DINO** (Distillation without Labels)
   - Vision Transformer-style distillation
   - Best for general representations

2. **AIM** (Appearance is Motion)
   - Contrastive learning from motion augmentations
   - Specializes in detecting dynamic patterns

3. **RelCon** (Relative Contrastive Learning)
   - Learning relative relationships between samples
   - Good for time series dynamics

4. **SimCLR** (Simple Contrastive Learning)
   - Standard contrastive framework
   - Works well with augmentation strategies

**Usage:**
```bash
python training/pretrain.py \
    --input_data_folder /path/to/data \
    --model MBA_tsm_with_padding \
    --pretraining_method DINO \
    --output results/pretrained/ \
    --pretraining_epochs 100
```

**Pretraining Arguments:**
- `--pretraining_method`: AIM, DINO, RelCon, or SimCLR
- `--pretraining_epochs`: Number of epochs
- `--pretraining_batch_size`: Batch size for pretraining
- `--temperature`: Temperature parameter for contrastive loss
- `--momentum`: Momentum coefficient for momentum encoders

**Using Pretrained Weights:**
After pretraining, use the saved weights in training scripts:
```bash
python training/train_scratch.py \
    --input_data_folder /path/to/data \
    --model MBA_tsm_with_padding \
    --pretrained_feature_extractor_path /path/to/pretrained.pth \
    --freeze_encoder False  # Fine-tune or train from pretrained
```

## Configuration Management

### Using YAML Configuration Files

All scripts support loading hyperparameters from YAML config files:

```bash
# Create a custom config based on sample
cp training/configs/sample_config.yaml training/configs/my_config.yaml
# Edit my_config.yaml with your parameters
```

### Configuration Sections

**Data Configuration:**
- Input paths and formats
- Preprocessing options
- Scaler paths

**Model Configuration:**
- Architecture selection
- Hidden dimensions
- Attention parameters

**Training Configuration:**
- Learning rates and optimizers
- Batch sizes and epochs
- Loss function weights

**Validation & Testing:**
- Cross-validation strategy
- Metrics to track
- Evaluation configuration

**Task-Specific:**
- TSO-specific parameters (class mapping, smoothing)
- Scratch detection settings
- Pretraining configuration

## Import Mapping

Scripts use the following module imports (auto-configured for new structure):

```python
# Model architectures
from models import setup_model, MBA_tsm_with_padding, MBA4TSO_Patch, ...

# Data loading and preprocessing
from data import load_data_tso, batch_generator, add_padding_TSO, ...

# Loss functions
from losses import measure_loss_tso, TemporalSmoothnessLoss, ...
from losses.dlrtc import GeneralizedCrossEntropy, DLRTCLoss, ...

# Utilities and evaluation
from utils import EarlyStopping, smooth_predictions, calculate_metrics_nn, ...
from evaluation import plot_tso_learning_curves, visualize_attention, ...

# Pretraining methods
from models.pretraining import AIMPretrainer, DINOPretrainer, RelConPretrainer, ...
```

## Common Workflows

### 1. Quick Start with H5 Data

```bash
# Step 1: Convert parquet to H5 (one-time)
python training/convert_h5.py \
    --input_folder data/parquet \
    --output_h5 data/preprocessed.h5 \
    --scaler_path data/scaler.joblib

# Step 2: Train scratch detection
python training/train_scratch_h5.py \
    --input_h5_file data/preprocessed.h5 \
    --model MBA_tsm_with_padding \
    --output results/scratch_detection

# Step 3: Train TSO prediction
python training/train_tso_patch_h5.py \
    --input_h5_file data/preprocessed.h5 \
    --model MBA4TSO_Patch \
    --output results/tso_prediction
```

### 2. Pretraining + Fine-tuning

```bash
# Step 1: Pretrain with DINO
python training/pretrain.py \
    --input_data_folder data/ \
    --model MBA_tsm_with_padding \
    --pretraining_method DINO \
    --pretraining_epochs 100 \
    --output results/pretrained/

# Step 2: Fine-tune on target task
python training/train_scratch_h5.py \
    --input_h5_file data/preprocessed.h5 \
    --model MBA_tsm_with_padding \
    --pretrained_feature_extractor_path results/pretrained/feature_extractor.pth \
    --freeze_encoder False \
    --output results/scratch_finetuned
```

### 3. Robust Training with Label Noise

```bash
python training/train_tso_dlrtc.py \
    --input_h5_file data/preprocessed.h5 \
    --model MBA4TSO_Patch \
    --warmup_epochs 50 \
    --label_refine_freq 10 \
    --temporal_weight 0.1 \
    --output results/tso_robust
```

### 4. Hyperparameter Tuning

```bash
python training/train_scratch.py \
    --input_data_folder data/ \
    --model MBA_tsm_with_padding \
    --execution_mode tune \
    --output results/tuning \
    --n_trials 100 \
    --n_jobs 4
```

## Expected Output Structure

```
results/
├── scratch_detection/
│   ├── best_model.pth
│   ├── training_log.csv
│   ├── metrics_fold_0.json
│   ├── metrics_fold_1.json
│   └── ...
├── tso_prediction/
│   ├── best_model.pth
│   ├── training_log.csv
│   ├── predictions_test.csv
│   └── ...
└── pretrained/
    ├── feature_extractor.pth
    ├── pretraining_log.csv
    └── ...
```

## Troubleshooting

### Memory Issues
- Reduce batch size with `--batch_size`
- Use H5 format instead of parquet
- Reduce sequence length with `--max_seq_length`

### Slow Training
- Use H5 preprocessed data (10-100x faster)
- Reduce number of workers
- Enable mixed precision with `--mixed_precision True`

### Poor Performance
- Check padding value consistency
- Verify scaler path matches training data
- Ensure sufficient pretraining if using `--pretrained_feature_extractor_path`
- Review loss weight configuration

### Data Loading Issues
- Verify file paths and formats
- Check scaler path if provided
- Ensure H5 file has correct structure

## References

- **Project Documentation:** `/sessions/bold-dreamy-dijkstra/mnt/JNJ/CLAUDE.md`
- **Model Architectures:** `/sessions/bold-dreamy-dijkstra/mnt/JNJ/models/`
- **Data Pipeline:** `/sessions/bold-dreamy-dijkstra/mnt/JNJ/data/`
- **Loss Functions:** `/sessions/bold-dreamy-dijkstra/mnt/JNJ/losses/`
