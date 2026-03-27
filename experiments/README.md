# Experiments: Harness Engineering Framework

## Overview

This directory follows a **harness engineering** approach to reproducible, structured ML experiments:

```
Planning (Define)  →  Generation (Execute)  →  Evaluation (Assess)
    ↓                        ↓                        ↓
  YAML Config          Training Script         Metrics & Results
```

The framework separates concerns to ensure:
- **Reproducibility**: Configs define exact experiment parameters
- **Auditability**: All decisions tracked in version control
- **Scalability**: Easy to run 10s or 100s of experiments
- **Debugging**: Results are structured and comparable

---

## Components

### 1. Planner: Experiment Config (YAML)

**Role**: Define what to test (the contract)

**Files**: `configs/*.yaml`

**Content**:
- Experiment metadata (name, description)
- Data configuration (format, sampling rate)
- Model architecture (type, hyperparameters)
- Training parameters (optimizer, learning rate, epochs)
- Evaluation strategy (cross-validation method, metrics)
- Loss weights and regularization

**Example**:
```yaml
experiment:
  name: scratch_baseline
  description: "Deep Scratch LOSO baseline with ResMamba"

model:
  architecture: mba_tsm_with_padding
  n_resnet_blocks: 4
  d_model: 128
  n_mamba_layers: 4
  dropout: 0.2

training:
  batch_size: 32
  epochs: 50
  lr: 0.001
  scheduler: cosine_annealing_warm_restarts

evaluation:
  strategy: loso
  metrics: [f1, precision, recall, auc]
```

### 2. Generator: Training Script

**Role**: Execute experiment as defined by config

**Files**: `predict_scratch_segment_h5.py`, `predict_TSO_segment_patch_dlrtc.py`, etc.

**Process**:
1. Read YAML config file
2. Load and prepare data
3. Initialize model with config parameters
4. Train with specified hyperparameters
5. Evaluate on test set
6. Save results

**Integration**:
```python
# In training script
config = yaml.load_file('configs/scratch_baseline.yaml')

# Use config throughout
model = init_model(config['model'])
train_loader = get_data(config['data'])
optimizer = get_optimizer(config['training'])
results = train_and_evaluate(model, train_loader, config)

# Save results with config reference
results['config'] = config
joblib.dump(results, f"results/{config['experiment']['name']}.joblib")
```

### 3. Evaluator: Results and Metrics

**Role**: Assess results independently

**Files**: `results/*.joblib`, metrics reports

**Process**:
1. Load results from training script
2. Calculate metrics (F1, precision, recall, AUC)
3. Compare with baseline
4. Generate visualizations
5. Document findings

**Outputs**:
```
results/
├── scratch_baseline.joblib
├── scratch_baseline_metrics.json
├── scratch_baseline_plots/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── training_history.png
└── scratch_baseline_report.md
```

---

## End-to-End Workflow

### Step 1: Define Experiment (Planning)

Create a new YAML config in `configs/`:

```yaml
# configs/scratch_baseline.yaml
experiment:
  name: scratch_baseline
  description: "Deep Scratch LOSO baseline with ResMamba"
  task: scratch_detection

data:
  format: parquet  # or h5
  channels: [x, y, z]
  sampling_rate: 20
  padding_value: -999

model:
  name: mba_tsm_with_padding
  n_resnet_blocks: 4
  d_model: 128
  n_mamba_layers: 4
  dropout: 0.2

training:
  optimizer: rmsprop
  lr: 0.001
  scheduler: cosine_annealing_warm_restarts
  batch_size: 32
  epochs: 50
  early_stopping_patience: 10

loss:
  lambda_cls: 1.0
  lambda_reg: 0.5
  lambda_mask: 0.3

evaluation:
  strategy: loso
  metrics: [f1, precision, recall, auc]
```

### Step 2: Generate Results (Generation)

Run the training script with config:

```bash
python predict_scratch_segment_h5.py \
    --config configs/scratch_baseline.yaml \
    --output_dir results/
```

**Script behavior**:
1. Loads config from YAML
2. Validates config (checks required fields)
3. Initializes reproducibility (seeds)
4. Trains model
5. Evaluates on test set
6. Saves results + config reference

### Step 3: Evaluate and Compare (Evaluation)

**Option A: Python script**

```python
import joblib
import json

# Load results
results = joblib.load('results/scratch_baseline.joblib')

# Access metrics
print(f"Test F1: {results['test_metrics']['f1']:.4f}")
print(f"Test Precision: {results['test_metrics']['precision']:.4f}")
print(f"Test Recall: {results['test_metrics']['recall']:.4f}")

# Save human-readable metrics
metrics = {
    'experiment': results['config']['experiment']['name'],
    'f1': results['test_metrics']['f1'],
    'precision': results['test_metrics']['precision'],
    'recall': results['test_metrics']['recall'],
}
with open('results/scratch_baseline_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
```

**Option B: Comparison script**

```python
import glob
import joblib
import pandas as pd

# Load all results
experiments = glob.glob('results/*.joblib')
comparison = []

for exp_file in experiments:
    results = joblib.load(exp_file)
    comparison.append({
        'name': results['config']['experiment']['name'],
        'f1': results['test_metrics']['f1'],
        'precision': results['test_metrics']['precision'],
        'recall': results['test_metrics']['recall'],
    })

# Create comparison table
df = pd.DataFrame(comparison).sort_values('f1', ascending=False)
print(df.to_string(index=False))
```

---

## Config Examples

### Example 1: Scratch Detection Baseline

```yaml
# configs/scratch_baseline.yaml
experiment:
  name: scratch_baseline
  description: "Deep Scratch LOSO baseline with ResMamba"
  task: scratch_detection

data:
  format: h5
  sampling_rate: 20
  channels: [x, y, z]
  padding_value: -999

model:
  architecture: mba_tsm_with_padding
  n_resnet_blocks: 4
  d_model: 128
  n_mamba_layers: 4
  dropout: 0.2
  attention_heads: 4

training:
  optimizer: rmsprop
  lr: 0.001
  scheduler: cosine_annealing_warm_restarts
  batch_size: 32
  epochs: 50
  early_stopping_patience: 10
  warmup_epochs: 5

loss:
  type: cross_entropy
  lambda_cls: 1.0
  lambda_reg: 0.5
  lambda_mask: 0.3

evaluation:
  strategy: loso
  metrics: [f1, precision, recall, auc, accuracy]
  val_split: 0.2
```

### Example 2: TSO with DLR-TC

```yaml
# configs/tso_dlrtc.yaml
experiment:
  name: tso_dlrtc
  description: "TSO with Dynamic Label Refinement"
  task: tso_detection

data:
  format: h5
  sampling_rate: 20
  channels: [x, y, z, temp, time_sin, time_cos]
  padding_value: -999

model:
  architecture: mba4tso_patch
  n_resnet_blocks: 4
  d_model: 128
  patch_size: 1200
  dropout: 0.2

training:
  optimizer: rmsprop
  lr: 0.001
  batch_size: 24
  epochs: 35

training_strategy: dlrtc
dlrtc:
  warmup_epochs: 10
  joint_epochs: 25
  gce_q: 0.7
  alpha: 0.1
  beta: 0.5
  label_lr: 0.01

loss:
  type: dlrtc

evaluation:
  strategy: lofo
  metrics: [f1, accuracy, precision, recall]
```

### Example 3: Scratch with Contrastive Learning

```yaml
# configs/scratch_contrastive.yaml
experiment:
  name: scratch_contrastive
  description: "Scratch detection with self-supervised pretraining"
  task: scratch_detection

data:
  format: h5
  sampling_rate: 20
  channels: [x, y, z]

pretraining:
  enabled: true
  method: dino  # or relcon
  epochs: 20

model:
  architecture: mba_tsm_with_padding
  n_resnet_blocks: 4
  d_model: 128
  n_mamba_layers: 4
  projection_dim: 256  # For contrastive loss

training:
  optimizer: adamw
  lr: 0.001
  scheduler: cosine_annealing
  batch_size: 32
  epochs: 50

loss:
  type: supervised_contrastive
  temperature: 0.07

evaluation:
  strategy: loso
  metrics: [f1, precision, recall, auc]
```

---

## Advanced Usage

### Hyperparameter Search

Create multiple configs for grid search:

```bash
# Generate configs for hyperparameter search
for lr in 0.0001 0.0005 0.001 0.005; do
    for batch_size in 16 32 48; do
        cat > configs/scratch_hp_search_lr${lr}_bs${batch_size}.yaml << EOF
experiment:
  name: "scratch_hp_search_lr${lr}_bs${batch_size}"
  description: "HP search: lr=$lr, batch_size=$batch_size"

model:
  architecture: mba_tsm_with_padding

training:
  lr: $lr
  batch_size: $batch_size
  # ... rest of config
EOF
    done
done

# Run all experiments
for config in configs/scratch_hp_search_*.yaml; do
    python predict_scratch_segment_h5.py --config $config &
done
wait
```

### Parallel Experiment Runs

```bash
#!/bin/bash
# run_experiments.sh

configs=(
    "configs/scratch_baseline.yaml"
    "configs/scratch_contrastive.yaml"
    "configs/tso_dlrtc.yaml"
)

for config in "${configs[@]}"; do
    echo "Running $config"
    python predict_scratch_segment_h5.py --config "$config" &
done

wait
echo "All experiments completed"
```

### Results Aggregation

```python
# aggregate_results.py
import glob
import joblib
import pandas as pd
import json

# Collect all results
results_list = []
for result_file in glob.glob('results/*.joblib'):
    try:
        results = joblib.load(result_file)
        results_list.append({
            'experiment': results['config']['experiment']['name'],
            'description': results['config']['experiment']['description'],
            'f1': results['test_metrics'].get('f1', None),
            'precision': results['test_metrics'].get('precision', None),
            'recall': results['test_metrics'].get('recall', None),
            'accuracy': results['test_metrics'].get('accuracy', None),
        })
    except Exception as e:
        print(f"Error loading {result_file}: {e}")

# Create comparison dataframe
df = pd.DataFrame(results_list).sort_values('f1', ascending=False)

# Save summary
print("\n" + "="*80)
print("EXPERIMENT SUMMARY")
print("="*80)
print(df.to_string(index=False))

# Save to CSV
df.to_csv('results/experiment_summary.csv', index=False)

# Save to JSON for programmatic access
df.to_json('results/experiment_summary.json', orient='records', indent=2)

print("\nResults saved to:")
print("  - results/experiment_summary.csv")
print("  - results/experiment_summary.json")
```

---

## Best Practices

### 1. Config Versioning

Keep configs in git with meaningful names:

```bash
# ✓ Good names
configs/scratch_baseline_v1.yaml
configs/scratch_contrastive_attention_heads_8.yaml
configs/tso_dlrtc_warm10_joint25.yaml

# ✗ Bad names
configs/experiment.yaml
configs/test123.yaml
configs/old_config.yaml
```

### 2. Reproducibility

Each config should specify random seeds:

```yaml
reproducibility:
  random_seed: 42
  torch_seed: 42
  numpy_seed: 42
  use_deterministic_algorithms: true
```

### 3. Experiment Documentation

Add notes to configs:

```yaml
experiment:
  name: scratch_baseline
  description: "Baseline model for scratch detection"
  notes: |
    This is the baseline model without any advanced techniques.
    Used as reference for comparing:
    - Contrastive learning
    - DLR-TC
    - Post-processing improvements
  author: "your_name"
  date: "2026-03-26"
```

### 4. Results Organization

Keep results organized by task and date:

```
results/
├── 2026-03-26/
│   ├── scratch_baseline.joblib
│   ├── scratch_contrastive.joblib
│   └── summary.csv
├── 2026-03-27/
│   ├── tso_dlrtc.joblib
│   └── summary.csv
└── latest_best_f1.joblib  # Symbolic link to best result
```

---

## Integration with Training Scripts

### Minimal Script Changes Required

**Before**:
```python
# Hard-coded parameters
batch_size = 32
lr = 0.001
epochs = 50
```

**After**:
```python
import yaml

# Load from config
config = yaml.safe_load(open(args.config))
batch_size = config['training']['batch_size']
lr = config['training']['lr']
epochs = config['training']['epochs']
```

### Config Validation

```python
import yaml
from schema import Schema, And, Or

# Define schema
config_schema = Schema({
    'experiment': {
        'name': str,
        'description': str,
        'task': Or('scratch_detection', 'tso_detection'),
    },
    'model': {
        'architecture': str,
        'n_resnet_blocks': int,
        'd_model': int,
    },
    'training': {
        'batch_size': And(int, lambda x: 8 <= x <= 256),
        'lr': And(float, lambda x: 1e-6 < x < 1),
        'epochs': And(int, lambda x: 1 <= x <= 1000),
    },
})

# Validate
try:
    config_schema.validate(config)
    print("✓ Config is valid")
except Exception as e:
    print(f"✗ Config validation failed: {e}")
```

---

## Troubleshooting

### Issue: Config not found

```
FileNotFoundError: configs/experiment.yaml
```

**Solution**: Ensure config file exists in `configs/` directory

### Issue: Config missing required field

```
KeyError: 'training'
```

**Solution**: Add missing sections to YAML, use validation schema

### Issue: Results not comparable

**Solution**: Ensure all experiments:
1. Use same data split
2. Use same evaluation metrics
3. Use same random seeds
4. Document any differences in config notes

---

## Summary

**Three-phase harness engineering**:

1. **Planner (YAML)**: Define experiment parameters
2. **Generator (Script)**: Execute experiment exactly as specified
3. **Evaluator (Results)**: Compare and assess outcomes

**Benefits**:
- ✅ Reproducible: Configs are self-contained
- ✅ Scalable: Easy to run 10s or 100s of experiments
- ✅ Comparable: Structured results enable analysis
- ✅ Auditable: All decisions tracked in git

**Quick start**:
```bash
# 1. Create config
cp configs/scratch_baseline.yaml configs/my_experiment.yaml
# (edit as needed)

# 2. Run training
python predict_scratch_segment_h5.py --config configs/my_experiment.yaml

# 3. Evaluate
python scripts/aggregate_results.py  # Shows comparison table
```

See `../docs/project_overview.md` for model details and training concepts.
