# Deployment Guide: Local Multi-GPU, Distributed, and Cloud Training

## Table of Contents
1. [Overview](#overview)
2. [Local Multi-GPU Training](#local-multi-gpu-training)
3. [Distributed Training (Multiple Nodes)](#distributed-training-multiple-nodes)
4. [Domino Cloud Deployment](#domino-cloud-deployment)
5. [Performance Tuning](#performance-tuning)
6. [Monitoring and Debugging](#monitoring-and-debugging)

---

## Overview

This guide covers three deployment scenarios:

| Scenario | Use Case | Hardware | Setup Time | Training Speed |
|----------|----------|----------|-----------|-----------------|
| **Local Multi-GPU** | Development, quick experiments | 1 machine, 2-8 GPUs | 5 min | 3-7x faster |
| **Distributed** | Large datasets, multi-day training | Multiple nodes/machines | 30 min | 10-20x faster |
| **Cloud (Domino)** | Production, managed infrastructure | Cloud VMs with auto-scaling | 15 min | Unlimited scale |

---

## Local Multi-GPU Training

### Quick Start

```bash
# Single GPU (baseline)
python predict_TSO_segment_patch_dlrtc.py \
    --input_h5 /path/to/data.h5 \
    --output experiment_name \
    --num_gpu 0

# Multi-GPU with DataParallel (2-4 GPUs)
python predict_TSO_segment_patch_dlrtc.py \
    --input_h5 /path/to/data.h5 \
    --output experiment_name \
    --num_gpu 0,1,2,3 \
    --multi_gpu
```

### Configuration

#### Effective Batch Size

When using multiple GPUs, the **effective batch size** increases:

```
effective_batch_size = batch_size × num_gpus
```

**Example**: With `batch_size=24` and 4 GPUs:
- Each GPU processes: 24 samples
- Effective batch size: 96 samples per iteration
- Total throughput: ~4x single GPU

#### Batch Size and Learning Rate Scaling

```python
# In best_params or config
best_params = {
    'batch_size': 24,  # Per-GPU batch size
    'num_gpus': 4,
}

# Option 1: Linear learning rate scaling
import math
scaled_lr = base_lr * math.sqrt(best_params['num_gpus'])
best_params['lr'] = scaled_lr  # More aggressive, faster convergence

# Option 2: Conservative (safe for stability)
best_params['lr'] = base_lr  # Keep same learning rate
```

### GPU Utilization Monitoring

Monitor GPU usage while training:

```bash
# Real-time monitoring (update every 1 second)
watch -n 1 nvidia-smi

# Detailed metrics
watch -n 1 'nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv'
```

**Expected output for 4 GPUs**:
```
GPU  Utilization  Memory Used/Total
0    94%          11234 / 16384 MiB
1    92%          11234 / 16384 MiB
2    93%          11234 / 16384 MiB
3    95%          11234 / 16384 MiB
```

**Good signs**:
- ✅ All GPUs > 85% utilized
- ✅ Similar memory usage across GPUs
- ✅ GPU 0 may be slightly higher (gradient aggregation)

**Warning signs**:
- ❌ GPU 0 at 100%, others < 50% (bottleneck at main GPU)
- ❌ Any GPU idle (check `--num_gpu` argument)
- ❌ Low utilization on all GPUs (increase batch size)

### Performance Scaling

**Expected speedups** (on 34.6K segments dataset):

| GPUs | Batch/GPU | Eff. Batch | Time/35 Epochs | Speedup |
|------|-----------|-----------|-----------------|---------|
| 1 | 24 | 24 | ~45 hours | 1.0x |
| 2 | 24 | 48 | ~24 hours | 1.9x |
| 4 | 24 | 96 | ~13 hours | 3.5x |
| 8 | 16 | 128 | ~8 hours | 5.6x |

**Note**: Speedup is sublinear due to communication overhead (gathering gradients to GPU 0).

### Troubleshooting Multi-GPU

**Issue 1: "CUDA out of memory"**

```python
# Solution: Reduce per-GPU batch size
best_params['batch_size'] = 16  # From 24
# Total effective batch size: 16 × 4 = 64
```

**Issue 2: Only GPU 0 working**

```bash
# Wrong (missing --multi_gpu flag)
python predict_TSO_segment_patch_dlrtc.py --num_gpu 0,1,2,3

# Correct
python predict_TSO_segment_patch_dlrtc.py --num_gpu 0,1,2,3 --multi_gpu
```

**Issue 3: No speedup (4 GPUs same speed as 1)**

Causes and solutions:
1. **Batch size too small**: Increase to 32-48 per GPU
2. **I/O bottleneck**: Use H5 format instead of parquet
3. **Model too small**: Check parameter count > 10M

---

## Distributed Training (Multiple Nodes)

Use Ray for training across multiple machines. Ray provides:
- Automatic worker management
- Gradient synchronization across nodes
- Fault tolerance and checkpointing
- 10-20x speedup with proper setup

### Installation

```bash
# Create requirements file
cat > requirements_ray.txt << EOF
ray[train]==2.9.0
ray[tune]==2.9.0
torch>=2.0.0
pandas
numpy
scikit-learn
joblib
h5py
EOF

# Install
pip install -r requirements_ray.txt
```

### Single-Node Setup (4 GPUs on 1 machine)

```bash
python predict_TSO_segment_patch_dlrtc.py \
    --input_h5 /path/to/data.h5 \
    --output ray_single_node \
    --num_workers 4 \
    --gpus_per_worker 1 \
    --cpus_per_worker 8
```

### Multi-Node Cluster Setup

#### Step 1: Start Ray Head Node

On the first machine:

```bash
ray start --head \
    --port=6379 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --num-cpus=32 \
    --num-gpus=4
```

Note the **Head Node IP** displayed in output (e.g., `192.168.1.100`)

#### Step 2: Start Ray Worker Nodes

On each additional machine:

```bash
ray start \
    --address='<HEAD_NODE_IP>:6379' \
    --num-cpus=32 \
    --num-gpus=4
```

Repeat for each worker machine.

#### Step 3: Run Training

```bash
python predict_TSO_segment_patch_dlrtc.py \
    --input_h5 /path/to/data.h5 \
    --output ray_multinode \
    --num_workers 16 \
    --gpus_per_worker 1 \
    --cpus_per_worker 8 \
    --ray_address "ray://<HEAD_NODE_IP>:10001"
```

### Performance Scaling (Multi-Node)

**For 34.6K segments dataset, 35 epochs**:

| Nodes | Total GPUs | Time/Epoch | Total Time | Speedup |
|-------|-----------|-----------|-----------|---------|
| 1 | 4 | ~13 min | ~7.6 hours | 3.5x |
| 2 | 8 | ~7 min | ~4.1 hours | **6.5x** |
| 4 | 16 | ~4 min | ~2.3 hours | **11.3x** |
| 8 | 32 | ~2.5 min | ~1.5 hours | **18x** 🚀 |

### Optimal Worker Configuration

| Total GPUs | num_workers | GPUs/worker | CPUs/worker | Batch/worker |
|-----------|------------|------------|------------|--------------|
| 4 | 4 | 1 | 8 | 16 |
| 8 | 8 | 1 | 8 | 16 |
| 16 | 16 | 1 | 4-8 | 16 |
| 32 | 32 | 1 | 4 | 12-16 |

### Ray Dashboard Monitoring

Access Ray dashboard at `http://<HEAD_NODE_IP>:8265`

**Dashboard shows**:
- Worker status and resource usage
- Task timeline and execution
- Memory utilization
- GPU utilization per worker

---

## Domino Cloud Deployment

### Setup

#### 1. Install Dependencies

Create `requirements_ray.txt` in your Domino project:

```txt
ray[train]==2.9.0
ray[tune]==2.9.0
torch>=2.0.0
pandas>=1.0.0
numpy
scikit-learn
joblib
h5py
```

#### 2. Configure Compute Environment

In Domino's environment management:

**Hardware Tier**:
- Select GPU tier (e.g., "GPU - Medium" with 4 GPUs)
- Use consistent hardware across all nodes

**Docker Image**:
- CUDA-enabled PyTorch: `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime`

### Single-Node on Domino

```bash
# Run as Domino Job with GPU hardware tier
python predict_TSO_segment_patch_dlrtc.py \
    --input_h5 /mnt/data/data.h5 \
    --output domino_single_node \
    --num_workers 4 \
    --gpus_per_worker 1 \
    --cpus_per_worker 8
```

### Multi-Node on Domino (Recommended for Production)

#### Step 1: Create Head Node Job

Create a new Domino Job:
- **Name**: `ray-head-tso-training`
- **Hardware**: GPU tier with 4 GPUs
- **Command**:

```bash
ray start --head \
    --port=6379 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --num-gpus=4

# Keep job running
sleep infinity
```

**After job starts**, note the **Job ID** or use Domino's networking to get internal IP.

#### Step 2: Create Worker Node Jobs

For each worker, create similar Domino jobs:
- **Name**: `ray-worker-1`, `ray-worker-2`, etc.
- **Hardware**: Same GPU tier as head node
- **Command**:

```bash
# Get head node IP from Domino networking
# Format: <job-id>.internal:6379 (Domino-specific)
# or <actual-IP>:6379

ray start \
    --address='<HEAD_NODE_IP>:6379' \
    --num-gpus=4

# Keep job running
sleep infinity
```

#### Step 3: Submit Training Job

Once all nodes are ready, create training job:
- **Name**: `tso-distributed-training`
- **Hardware**: Single GPU or CPU tier (will connect to Ray cluster)
- **Command**:

```bash
python predict_TSO_segment_patch_dlrtc.py \
    --input_h5 /mnt/data/data.h5 \
    --output domino_distributed_v1 \
    --model mba4tso_patch \
    --epochs 35 \
    --num_workers 16 \
    --gpus_per_worker 1 \
    --cpus_per_worker 8 \
    --ray_address "ray://<HEAD_NODE_IP>:10001"
```

### Domino-Specific Networking

**For inter-job communication in Domino**:

1. **Internal addresses**: Jobs can reach each other via:
   ```
   <source-job-id>.internal:6379
   ```

2. **Get job IDs**: In Domino, job IDs are visible in job settings

3. **Alternative (External IPs)**: Some Domino setups allow external cluster IPs

### Cost Optimization on Domino

**Strategies to minimize cloud spend**:

1. **Right-size workers**:
   - Don't over-provision CPU if data loading isn't bottleneck
   - Use smallest GPU tier that fits model + batch size

2. **Spot instances** (if available):
   - Use spot/preemptible instances for workers only
   - Keep head node on stable instances

3. **Schedule off-peak**:
   - Run large training jobs during off-peak hours
   - Domino may offer time-based pricing discounts

4. **Checkpoint frequently**:
   - Enable Ray checkpointing every 5-10 epochs
   - Allows resuming from failures without restarting

### Example: Cost Comparison (Hypothetical)

| Setup | Runtime | Cost/GPU-hr | Total Cost | Speed Benefit |
|-------|---------|-----------|-----------|--------------|
| 1 GPU × 75h | 75 hrs | $2.50 | **$187.50** | Baseline |
| 4 GPUs × 20h | 80 GPU-hrs | $2.50 | $200.00 | 3.75x faster |
| 8 GPUs × 12h | 96 GPU-hrs | $2.50 | $240.00 | 6.5x faster |
| 16 GPUs × 7h | 112 GPU-hrs | $2.50 | $280.00 | 10.7x faster |

**Analysis**: While multi-GPU increases total GPU-hours, **wall-clock time reduction** may justify cost for:
- Time-sensitive projects
- Rapid hyperparameter iteration
- Training multiple models in parallel

---

## Performance Tuning

### Batch Size Optimization

**Finding optimal batch size**:

```bash
# Start with conservative value
batch_size=16

# Increase until CUDA OOM
for bs in 16 24 32 40 48; do
    echo "Testing batch_size=$bs"
    python predict_TSO_segment_patch_dlrtc.py \
        --batch_size $bs \
        --input_h5 /path/to/data.h5 \
        --output test_bs_${bs}
    # Monitor GPU memory
done
```

**Target**: ~80-90% GPU memory utilization

**Guidelines by GPU**:
- RTX 3060 (12GB): batch_size=16-24
- RTX 3090 (24GB): batch_size=32-48
- A100 (80GB): batch_size=64-128

### Learning Rate Scaling

When effective batch size increases, scale learning rate:

```python
base_lr = 0.001
base_batch_size = 24

# Option 1: Sqrt scaling (recommended)
import math
effective_batch = 24 * num_gpus
scaled_lr = base_lr * math.sqrt(effective_batch / base_batch_size)

# Option 2: Linear scaling (more aggressive)
scaled_lr = base_lr * (effective_batch / base_batch_size)

# Option 3: No scaling (conservative)
scaled_lr = base_lr
```

**Empirical results**:
- Sqrt scaling: Best convergence and generalization
- Linear scaling: Faster convergence, risk of instability
- No scaling: Slower convergence but most stable

### Mixed Precision Training (Advanced)

Enable AMP (Automatic Mixed Precision) for 2x speedup:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop
with autocast():
    outputs = model(batch_X)
    loss = criterion(outputs, batch_Y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Benefits**:
- 1.5-2x speedup on V100/A100 GPUs
- ~30% less memory usage
- No accuracy loss if epsilon clamping is proper

---

## Monitoring and Debugging

### Console Output Interpretation

**During DLR-TC training**:

```
[Warmup] Epoch 1/10
  Train - Loss: 0.6234, Acc: 0.6512, F1 avg: 0.5834
  Val   - Loss: 0.5891, Acc: 0.6723, F1 avg: 0.6012
  Refined 0 labels (warmup phase)

[Joint] Epoch 1/25
  Train - Loss: 0.5234 (GCE: 0.4523, Compat: 0.0234, Temp: 0.0477)
          Acc: 0.6834, F1 avg: 0.6245
  Val   - Loss: 0.5123, Acc: 0.6945, F1 avg: 0.6312
  Refined 5,234 labels (19.1% of segments)
```

**Good progress indicators**:
- ✅ Val F1 increasing steadily
- ✅ `loss_compat` stable (0.015-0.025 range)
- ✅ `loss_temp` decreasing (smoothing predictions)
- ✅ Label refinement rate decreasing over epochs

**Warning signs**:
- ⚠️ Val F1 plateaus before epoch 20 → decrease `beta`
- ⚠️ `loss_compat` > 0.05 → labels drifting, increase `alpha`
- ⚠️ Val F1 decreasing → reduce `label_lr`
- ⚠️ Predictions too smooth → decrease `beta`

### TensorBoard Logging (Ray)

Add TensorBoard support for monitoring:

```python
from torch.utils.tensorboard import SummaryWriter

# In Ray training function
if rank == 0:  # Log from main worker only
    writer = SummaryWriter(log_dir=f"{output_folder}/tensorboard")
    writer.add_scalar('train/loss', train_loss, epoch)
    writer.add_scalar('val/f1', val_f1, epoch)
    writer.flush()
```

Launch TensorBoard:

```bash
tensorboard --logdir=/path/to/output_folder/tensorboard --port=6006
```

Access at `http://localhost:6006`

### Common Issues

#### Issue: Training stalls on multi-node

**Symptoms**: Hanging, no output for 10+ minutes

**Causes**:
- Network connectivity between nodes
- Ray worker crashes
- Distributed data loading issue

**Solutions**:

```bash
# Check Ray cluster status
ray status

# Check worker logs
ray logs worker <worker-id>

# Restart cluster
ray stop
ray start --head
```

#### Issue: Slow training on multi-node

**Symptoms**: Multi-node slower than single-node

**Causes**:
1. Data loading bottleneck (parquet on network drive)
2. Small batch size (communication overhead dominates)
3. Network latency between nodes

**Solutions**:
1. Use H5 format on fast local storage
2. Increase batch size per worker (24-32)
3. Verify network bandwidth: `iperf3 <node1> <node2>`

#### Issue: Inconsistent results across runs

**Causes**: Non-deterministic randomness

**Solutions**:

```python
import torch
import numpy as np
import random

# Set all seeds
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
```

---

## Quick Reference

### Local Multi-GPU
```bash
python predict_TSO_segment_patch_dlrtc.py \
    --input_h5 /path/data.h5 \
    --num_gpu 0,1,2,3 \
    --multi_gpu
# Speedup: 3-4x | Time: 10-15 hours
```

### Distributed (Ray on Multiple Nodes)
```bash
# Head: ray start --head --num-gpus=4
# Workers: ray start --address='<IP>:6379' --num-gpus=4
# Training:
python predict_TSO_segment_patch_dlrtc.py \
    --input_h5 /path/data.h5 \
    --num_workers 16 \
    --gpus_per_worker 1 \
    --ray_address "ray://<IP>:10001"
# Speedup: 10-20x | Time: 2-4 hours
```

### Domino Cloud
```bash
# 1. Create head node job: ray start --head --num-gpus=4
# 2. Create worker node jobs: ray start --address='<IP>:6379'
# 3. Submit training job with --ray_address and --num_workers
# Speedup: 10-18x | Time: 2-5 hours
```

---

## Troubleshooting Summary

| Problem | Symptom | Solution |
|---------|---------|----------|
| Low GPU util | < 50% on nvidia-smi | Increase batch size |
| OOM errors | CUDA out of memory | Reduce batch size |
| Slow training | Single GPU faster | Check data loading, increase batch size |
| Worker crashes | Hanging, no output | Check network, restart Ray cluster |
| Model diverges | Loss increasing | Reduce learning rate, check data |
| Inconsistent results | Different outputs | Set random seeds |

---

## Next Steps

1. **Start simple**: Single GPU, then multi-GPU, then distributed
2. **Monitor closely**: Watch GPU util and loss curves
3. **Scale gradually**: Add nodes incrementally
4. **Benchmark**: Compare wall-clock times at each scale
5. **Production**: Use Domino for managed, auto-scaling training

See `project_overview.md` for training concepts and `algorithms.md` for post-processing.
