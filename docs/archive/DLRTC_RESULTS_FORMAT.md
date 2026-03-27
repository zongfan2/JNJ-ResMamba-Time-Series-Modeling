# DLR-TC Results Format Guide

## Overview

DLR-TC saves training results in **two formats**:
1. **JSON** (.json) - Human-readable, portable, easy to inspect
2. **Joblib** (.joblib) - Python-specific, efficient for large arrays

---

## JSON Results File

### Location
```
results/DL/UKB_v2/your_experiment/training/predictions/
├── dlrtc_results_split_0_iter_0.json      # Human-readable
└── dlrtc_results_split_0_iter_0.joblib    # Python arrays
```

### Structure

```json
{
  "split_id": "0",
  "iteration": 0,
  "timestamp": "2026-02-03 14:30:45",
  "model": "mba4tso_patch",
  "total_epochs": 35,

  "test_metrics": {
    "loss": 0.3456,
    "accuracy": 0.8234,
    "f1_avg": 0.8156,
    "f1_other": 0.7845,
    "f1_nonwear": 0.8123,
    "f1_tso": 0.8501
  },

  "history": {
    "train_loss": [0.8234, 0.6543, ..., 0.3456],
    "train_accuracy": [0.6012, 0.6534, ..., 0.8234],
    "train_f1_avg": [0.5823, 0.6345, ..., 0.8156],
    "train_f1_other": [...],
    "train_f1_nonwear": [...],
    "train_f1_tso": [...],

    "val_loss": [0.7891, 0.6234, ..., 0.3567],
    "val_accuracy": [0.6234, 0.6745, ..., 0.8345],
    "val_f1_avg": [0.6012, 0.6534, ..., 0.8267],
    "val_f1_other": [...],
    "val_f1_nonwear": [...],
    "val_f1_tso": [...],

    "train_loss_gce": [0.4123, 0.3845, ..., 0.2856],
    "train_loss_compat": [0.0234, 0.0198, ..., 0.0156],
    "train_loss_temp": [0.0456, 0.0389, ..., 0.0267]
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

## Reading Results

### 1. Quick Inspection (Command Line)

```bash
# Pretty-print JSON
cat dlrtc_results_split_0_iter_0.json | python -m json.tool | less

# Get test accuracy
cat dlrtc_results_split_0_iter_0.json | jq '.test_metrics.accuracy'

# Get F1 scores
cat dlrtc_results_split_0_iter_0.json | jq '.test_metrics | {f1_avg, f1_tso}'
```

### 2. Python - Load JSON

```python
import json

# Load results
with open('dlrtc_results_split_0_iter_0.json', 'r') as f:
    results = json.load(f)

# Access metrics
print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
print(f"Test F1 (avg): {results['test_metrics']['f1_avg']:.4f}")
print(f"Test F1 (TSO): {results['test_metrics']['f1_tso']:.4f}")

# Access training history
train_loss = results['history']['train_loss']
val_loss = results['history']['val_loss']

print(f"Training converged at epoch {len(train_loss)}")
print(f"Best val loss: {min(val_loss):.4f}")

# Access DLR-TC config
config = results['dlrtc_config']
print(f"Warmup epochs: {config['warmup_epochs']}")
print(f"Joint epochs: {config['joint_epochs']}")
print(f"GCE q: {config['gce_q']}")

# Check label refinement
print(f"Refined {results['num_refined_labels']} segment labels")
```

### 3. Python - Load Joblib (Alternative)

```python
import joblib

# Load results (includes full numpy arrays)
results = joblib.load('dlrtc_results_split_0_iter_0.joblib')

# Same access as JSON
print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
```

---

## Comparing Results

### Compare Multiple Experiments

```python
import json
import glob

# Load all results
experiment_folders = [
    'results/DL/UKB_v2/dlrtc_alpha_0.05/training/predictions/',
    'results/DL/UKB_v2/dlrtc_alpha_0.10/training/predictions/',
    'results/DL/UKB_v2/dlrtc_alpha_0.20/training/predictions/',
]

comparison = []
for folder in experiment_folders:
    json_file = glob.glob(f"{folder}/dlrtc_results_*.json")[0]
    with open(json_file, 'r') as f:
        results = json.load(f)

    comparison.append({
        'experiment': folder.split('/')[-3],
        'alpha': results['dlrtc_config']['alpha'],
        'test_f1': results['test_metrics']['f1_avg'],
        'test_acc': results['test_metrics']['accuracy'],
    })

# Print comparison table
import pandas as pd
df = pd.DataFrame(comparison)
print(df.to_string(index=False))
```

### Compare with Baseline

```python
import json

# Load DLR-TC results
with open('dlrtc_results_split_0_iter_0.json', 'r') as f:
    dlrtc = json.load(f)

# Load baseline results (from predict_TSO_segment_patch.py)
import joblib
baseline = joblib.load('baseline_results_split_0_iter_0.joblib')

# Compare
dlrtc_f1 = dlrtc['test_metrics']['f1_avg']
baseline_f1 = baseline['test_metrics']['f1_avg']

improvement = (dlrtc_f1 - baseline_f1) / baseline_f1 * 100

print(f"DLR-TC F1:   {dlrtc_f1:.4f}")
print(f"Baseline F1: {baseline_f1:.4f}")
print(f"Improvement: {improvement:+.2f}%")
```

---

## Visualize Training History

### Plot Learning Curves

```python
import json
import matplotlib.pyplot as plt

# Load results
with open('dlrtc_results_split_0_iter_0.json', 'r') as f:
    results = json.load(f)

history = results['history']
config = results['dlrtc_config']
warmup_end = config['warmup_epochs']

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Loss
axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
axes[0, 0].plot(history['val_loss'], label='Val', linewidth=2)
axes[0, 0].axvline(warmup_end, color='red', linestyle='--', alpha=0.5, label='Joint starts')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Total Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: F1 Score
axes[0, 1].plot(history['train_f1_avg'], label='Train', linewidth=2)
axes[0, 1].plot(history['val_f1_avg'], label='Val', linewidth=2)
axes[0, 1].axvline(warmup_end, color='red', linestyle='--', alpha=0.5, label='Joint starts')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('F1 Score')
axes[0, 1].set_title('F1 Score (Average)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Loss Components (joint phase only)
if 'train_loss_gce' in history:
    joint_epochs = range(warmup_end, len(history['train_loss']))
    axes[1, 0].plot(joint_epochs, history['train_loss_gce'][warmup_end:],
                    label='GCE', linewidth=2)
    axes[1, 0].plot(joint_epochs, history['train_loss_compat'][warmup_end:],
                    label='Compatibility', linewidth=2)
    axes[1, 0].plot(joint_epochs, history['train_loss_temp'][warmup_end:],
                    label='Temporal', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('DLR-TC Loss Components (Joint Phase)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Per-class F1
axes[1, 1].plot(history['val_f1_other'], label='Other', linewidth=2)
axes[1, 1].plot(history['val_f1_nonwear'], label='Non-wear', linewidth=2)
axes[1, 1].plot(history['val_f1_tso'], label='TSO', linewidth=2)
axes[1, 1].axvline(warmup_end, color='red', linestyle='--', alpha=0.5, label='Joint starts')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('F1 Score')
axes[1, 1].set_title('Per-Class F1 (Validation)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dlrtc_training_analysis.png', dpi=150)
print("Saved training analysis plot")
```

---

## Export Summary Report

### Generate Markdown Report

```python
import json

# Load results
with open('dlrtc_results_split_0_iter_0.json', 'r') as f:
    results = json.load(f)

# Generate markdown report
report = f"""# DLR-TC Training Report

## Experiment Info
- **Timestamp**: {results['timestamp']}
- **Model**: {results['model']}
- **Split ID**: {results['split_id']}
- **Iteration**: {results['iteration']}
- **Total Epochs**: {results['total_epochs']}

## DLR-TC Configuration
- **Warmup Epochs**: {results['dlrtc_config']['warmup_epochs']}
- **Joint Epochs**: {results['dlrtc_config']['joint_epochs']}
- **GCE q**: {results['dlrtc_config']['gce_q']}
- **Alpha (Compatibility)**: {results['dlrtc_config']['alpha']}
- **Beta (Temporal)**: {results['dlrtc_config']['beta']}
- **Label Learning Rate**: {results['dlrtc_config']['label_lr']}

## Test Performance
- **Accuracy**: {results['test_metrics']['accuracy']:.4f}
- **F1 (Average)**: {results['test_metrics']['f1_avg']:.4f}
- **F1 (Other)**: {results['test_metrics']['f1_other']:.4f}
- **F1 (Non-wear)**: {results['test_metrics']['f1_nonwear']:.4f}
- **F1 (TSO)**: {results['test_metrics']['f1_tso']:.4f}
- **Test Loss**: {results['test_metrics']['loss']:.4f}

## Label Refinement
- **Segments Refined**: {results['num_refined_labels']:,}

## Training Convergence
- **Final Train Loss**: {results['history']['train_loss'][-1]:.4f}
- **Final Val Loss**: {results['history']['val_loss'][-1]:.4f}
- **Best Val F1**: {max(results['history']['val_f1_avg']):.4f}
"""

# Save report
with open('dlrtc_report.md', 'w') as f:
    f.write(report)

print("Report saved to dlrtc_report.md")
```

---

## JSON vs Joblib: When to Use Each

### Use JSON When:
✅ **Human inspection** - Easy to read in text editor
✅ **Version control** - Git-friendly, diff-able
✅ **Cross-platform** - Works everywhere (Python, JavaScript, etc.)
✅ **Sharing results** - Email, documentation, web apps
✅ **Quick access** - Command-line tools (jq, grep, etc.)

### Use Joblib When:
✅ **Large arrays** - Efficient storage for training history
✅ **Python scripts** - Programmatic analysis
✅ **Complex objects** - Numpy arrays, custom classes
✅ **Performance** - Faster load/save for big data

---

## Best Practice

**Recommended workflow**:
1. **Training**: Script saves both JSON + Joblib automatically
2. **Quick check**: Open JSON in text editor or use `jq`
3. **Analysis**: Load Joblib in Python for plotting/comparison
4. **Reporting**: Use JSON for documentation and version control

---

## Summary

**DLR-TC Results Files**:
- ✅ **JSON**: Human-readable, portable, Git-friendly
- ✅ **Joblib**: Efficient, Python-native, large arrays

**Both formats contain**:
- Test metrics (accuracy, F1 scores)
- Full training history (all epochs)
- DLR-TC configuration
- Metadata (timestamp, model, etc.)

**Access pattern**:
```python
# Quick inspection
with open('results.json', 'r') as f:
    results = json.load(f)

# Detailed analysis
results = joblib.load('results.joblib')
```

You now have the best of both worlds! 🎉
