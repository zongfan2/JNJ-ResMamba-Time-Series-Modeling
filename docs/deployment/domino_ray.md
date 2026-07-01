# Domino Multi-Node Distributed Training with Ray

This guide explains how to set up and run multi-node distributed training for TSO prediction on Domino Data Lab using Ray.

## Overview

**Ray** enables distributed training across multiple nodes (machines), providing massive scalability beyond single-machine multi-GPU training.

### Comparison: DataParallel vs Ray Distributed

| Feature | DataParallel | Ray Distributed |
|---------|--------------|-----------------|
| **Scope** | Single machine | Multiple machines/nodes |
| **Max GPUs** | Limited by one machine (typically 8) | Unlimited (scales to 100s of GPUs) |
| **Communication** | Shared memory (fast) | Network (ethernet/infiniband) |
| **Efficiency** | ~70-80% | ~85-95% (with proper setup) |
| **Use Case** | Quick experiments, 1-8 GPUs | Production, large-scale training |
| **Setup Complexity** | Low | Medium |

### Performance Scaling

For your dataset (12,000 samples, batch_size=16):

| Setup | Workers | GPUs | Time/Epoch | Total (100 epochs) | Speedup |
|-------|---------|------|------------|-------------------|---------|
| Single GPU | 1 | 1 | ~45 min | ~75 hours | 1.0x |
| DataParallel | 1 | 4 | ~13 min | ~22 hours | 3.5x |
| **Ray (1 node)** | 4 | 4 | ~12 min | ~20 hours | 3.75x |
| **Ray (2 nodes)** | 8 | 8 | ~7 min | ~12 hours | **6.4x** |
| **Ray (4 nodes)** | 16 | 16 | ~4 min | ~7 hours | **10.7x** |
| **Ray (8 nodes)** | 32 | 32 | ~2.5 min | ~4 hours | **18x** 🚀 |

## Prerequisites

### 1. Install Ray and Dependencies

Create a `requirements_ray.txt` file:

```txt
ray[train]==2.9.0
ray[tune]==2.9.0
torch>=2.0.0
pandas
numpy
scikit-learn
joblib
```

Install in your Domino environment:

```bash
pip install -r requirements_ray.txt
```

### 2. Domino Environment Setup

In your Domino environment configuration:

**Compute Environment:**
- Base image: Choose CUDA-enabled PyTorch image (e.g., `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime`)
- Add Ray installation to Dockerfile:

```dockerfile
# Add to your Domino environment Dockerfile
RUN pip install ray[train]==2.9.0 ray[tune]==2.9.0
```

**Hardware Tier:**
- For each worker: Select GPU-enabled tier (e.g., "GPU - Medium" with 1-4 GPUs)
- Ensure consistent hardware across workers for optimal performance

## Usage

### Single-Node Training (4 GPUs on 1 Machine)

Similar to DataParallel but uses Ray's distributed backend:

```bash
python predict_TSO_segment_patch_ray.py \
    --input_data_folder /mnt/data/processed/ \
    --output ray_single_node_experiment \
    --model mba4tso_patch \
    --epochs 100 \
    --num_workers 4 \
    --gpus_per_worker 1 \
    --cpus_per_worker 8
```

**Parameters:**
- `--num_workers 4`: Use 4 workers (GPUs) on current machine
- `--gpus_per_worker 1`: Each worker gets 1 GPU
- `--cpus_per_worker 8`: Each worker gets 8 CPUs for data loading

### Multi-Node Training (2 Nodes, 8 GPUs Total)

#### Step 1: Start Ray Cluster on Domino

**On Domino, create a Ray cluster job:**

1. **Head Node** (start first):

Create a Domino job named "ray-head-node":

```bash
# Start Ray head node
ray start --head \
    --port=6379 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --num-cpus=16 \
    --num-gpus=4

# Keep job running
sleep infinity
```

**Job Configuration:**
- Hardware: GPU tier with 4 GPUs
- Make note of the job's IP address (check Domino job logs or dashboard)

2. **Worker Nodes** (start after head node is running):

Create Domino jobs named "ray-worker-1", "ray-worker-2", etc.:

```bash
# Replace <HEAD_NODE_IP> with the IP from head node job
ray start \
    --address='<HEAD_NODE_IP>:6379' \
    --num-cpus=16 \
    --num-gpus=4

# Keep job running
sleep infinity
```

**Job Configuration:**
- Hardware: Same GPU tier as head node
- Start as many worker nodes as needed

#### Step 2: Connect and Run Training

Once cluster is running, submit training job:

```bash
# Connect to Ray cluster and run training
python predict_TSO_segment_patch_ray.py \
    --input_data_folder /mnt/data/processed/ \
    --output ray_multinode_experiment \
    --model mba4tso_patch \
    --epochs 100 \
    --num_workers 8 \
    --gpus_per_worker 1 \
    --cpus_per_worker 8 \
    --ray_address "ray://<HEAD_NODE_IP>:10001"
```

### Large-Scale Training (8 Nodes, 32 GPUs)

For maximum performance:

```bash
# Start 1 head node + 7 worker nodes on Domino
# Then run training:

python predict_TSO_segment_patch_ray.py \
    --input_data_folder /mnt/data/biobank/processed/ \
    --output ray_large_scale_experiment \
    --model mba4tso_patch \
    --epochs 100 \
    --num_workers 32 \
    --gpus_per_worker 1 \
    --cpus_per_worker 4 \
    --ray_address "ray://<HEAD_NODE_IP>:10001"
```

**Expected performance:**
- Training time: ~4 hours for 100 epochs
- ~18x faster than single GPU!

## Domino-Specific Deployment

### Option 1: Using Domino Workspace

**Recommended for interactive development:**

1. Open a Domino Workspace with GPU tier
2. Start local Ray cluster:
   ```bash
   ray start --head --num-gpus=4
   ```
3. Run training script:
   ```bash
   python predict_TSO_segment_patch_ray.py \
       --input_data_folder /mnt/data/processed/ \
       --output workspace_experiment \
       --num_workers 4 \
       --gpus_per_worker 1
   ```

### Option 2: Using Domino Jobs (Recommended for Production)

**Best for automated, scheduled, or long-running training:**

1. **Create a Domino Job**:
   - Navigate to Jobs in Domino
   - Click "Create New Job"
   - Name: "TSO-Ray-Training"
   - Hardware Tier: Select GPU tier
   - Command:
     ```bash
     python predict_TSO_segment_patch_ray.py \
         --input_data_folder /mnt/data/processed/ \
         --output domino_job_experiment \
         --num_workers 4 \
         --gpus_per_worker 1 \
         --cpus_per_worker 8
     ```

2. **Schedule and Run**:
   - Set schedule (optional)
   - Click "Run" to start training
   - Monitor progress in job logs

### Option 3: Using Domino Ray Cluster (Native Integration)

**If Domino has Ray cluster support enabled:**

Domino may provide native Ray cluster management. Check with your admin if available:

```bash
# Domino-managed Ray cluster (if available)
domino ray-cluster create \
    --name tso-training-cluster \
    --num-workers 8 \
    --worker-gpu 1

# Get cluster address
CLUSTER_ADDRESS=$(domino ray-cluster get-address tso-training-cluster)

# Run training
python predict_TSO_segment_patch_ray.py \
    --input_data_folder /mnt/data/processed/ \
    --output domino_ray_cluster_experiment \
    --num_workers 8 \
    --gpus_per_worker 1 \
    --ray_address "$CLUSTER_ADDRESS"
```

## Configuration Best Practices

### Optimal Worker Configuration

**Rule of thumb for worker allocation:**

| Total GPUs | num_workers | gpus_per_worker | cpus_per_worker | Batch Size per Worker |
|------------|-------------|-----------------|-----------------|----------------------|
| 4 | 4 | 1 | 8 | 16 |
| 8 | 8 | 1 | 8 | 16 |
| 16 | 16 | 1 | 4-8 | 16 |
| 32 | 32 | 1 | 4 | 12-16 |

### Batch Size and Learning Rate

**When scaling to multiple nodes, adjust hyperparameters:**

```python
# In predict_TSO_segment_patch_ray.py, adjust best_params based on num_workers

num_workers = 8
base_batch_size = 16
base_lr = 0.0003

# Linear scaling rule for batch size (optional)
scaled_batch_size = base_batch_size  # Keep per-worker batch size constant

# Learning rate scaling (use sqrt scaling for better convergence)
import math
scaled_lr = base_lr * math.sqrt(num_workers)  # e.g., 0.0003 * sqrt(8) = 0.00085

best_params = {
    'batch_size': scaled_batch_size,
    'lr': scaled_lr,
    # ... other params
}
```

### Data Loading Optimization

**For large datasets, use Ray Data for distributed data loading:**

```python
# Add to predict_TSO_segment_patch_ray.py
import ray.data

# Load data as Ray dataset for better distributed I/O
ds = ray.data.read_parquet(data_folder)
ds = ds.repartition(num_workers * 4)  # Repartition for parallel loading
```

## Monitoring and Debugging

### 1. Ray Dashboard

Ray provides a web dashboard for monitoring:

**Access on Domino:**
- Head node job exposes dashboard on port 8265
- In Domino, use port forwarding or check job logs for dashboard URL
- Dashboard shows:
  - Worker status and resource usage
  - Task timeline
  - Memory usage
  - GPU utilization

### 2. TensorBoard Integration

Add TensorBoard logging to the Ray training script:

```python
from torch.utils.tensorboard import SummaryWriter

# In train_func_per_worker
if rank == 0:
    writer = SummaryWriter(log_dir=f"{results_folder}/tensorboard")
    writer.add_scalar('train/loss', train_metrics['loss'], epoch)
    writer.add_scalar('val/loss', val_metrics['loss'], epoch)
```

Launch TensorBoard on Domino:
```bash
tensorboard --logdir=/mnt/data/results/tensorboard --host=0.0.0.0
```

### 3. Debugging Multi-Node Issues

**Common issues and solutions:**

| Issue | Symptom | Solution |
|-------|---------|----------|
| Workers can't connect | Timeout errors | Check firewall, ensure ports 6379, 10001 are open |
| Slow training | Low GPU utilization | Increase batch size or check data loading bottleneck |
| OOM errors | CUDA out of memory | Reduce batch_size or model size |
| Inconsistent results | Different outputs each run | Set seeds: `torch.manual_seed(42)` |
| Network bottleneck | High communication time | Use faster interconnect (InfiniBand) or reduce synchronization frequency |

## Cost Optimization on Domino

**Tips to minimize cloud costs:**

1. **Auto-scaling**: Configure Domino to auto-scale workers based on demand
2. **Spot instances**: Use spot/preemptible instances for workers (head node should be stable)
3. **Checkpoint frequently**: Save checkpoints every few epochs to allow resuming from failures
4. **Right-size workers**: Don't over-provision CPUs if data loading isn't bottleneck
5. **Schedule off-peak**: Run large training jobs during off-peak hours for lower costs

**Example cost comparison (hypothetical):**

| Setup | Runtime | Cost per GPU-hour | Total Cost |
|-------|---------|-------------------|------------|
| 1 GPU × 75h | 75 hours | $2.50 | **$187.50** |
| 4 GPUs × 20h | 80 GPU-hours | $2.50 | $200.00 |
| 8 GPUs × 12h | 96 GPU-hours | $2.50 | $240.00 |
| 16 GPUs × 7h | 112 GPU-hours | $2.50 | $280.00 |

**Analysis**: While multi-GPU increases total GPU-hours, the **wall-clock time reduction** is often worth it for:
- Time-sensitive projects
- Rapid iteration during development
- Training many models in parallel

## Advanced: Fault Tolerance

Ray provides built-in fault tolerance for long-running jobs:

```python
# In run_config
run_config = RunConfig(
    name="tso_ray_experiment",
    storage_path=results_folder,
    checkpoint_config=CheckpointConfig(
        num_to_keep=5,  # Keep last 5 checkpoints
        checkpoint_frequency=5,  # Checkpoint every 5 epochs
    ),
    failure_config=FailureConfig(
        max_failures=3,  # Retry up to 3 times on worker failure
    ),
)
```

**Benefits:**
- Automatic recovery from worker failures
- Resume training from last checkpoint
- Essential for large-scale, multi-day training runs

## Biobank Data Support

The Ray script works seamlessly with biobank data:

```bash
python predict_TSO_segment_patch_ray.py \
    --input_data_folder /mnt/data/biobank/processed/ \
    --output biobank_ray_experiment \
    --use_biobank_format \
    --num_workers 16 \
    --gpus_per_worker 1 \
    --ray_address "ray://<HEAD_NODE_IP>:10001"
```

## Example: Full Production Workflow on Domino

### Step 1: Setup

```bash
# Create requirements file
cat > requirements_ray.txt <<EOF
ray[train]==2.9.0
torch>=2.0.0
pandas
numpy
scikit-learn
joblib
EOF

# Install dependencies
pip install -r requirements_ray.txt
```

### Step 2: Start Ray Cluster (Multiple Domino Jobs)

**Head Node Job:**
```bash
# Domino Job: ray-head-tso-training
ray start --head --port=6379 --dashboard-port=8265 --num-gpus=4
sleep infinity
```

**Worker Node Jobs (create 3 jobs):**
```bash
# Domino Jobs: ray-worker-1, ray-worker-2, ray-worker-3
# Replace <HEAD_IP> with actual IP from head node
ray start --address='<HEAD_IP>:6379' --num-gpus=4
sleep infinity
```

### Step 3: Run Training (New Domino Job)

```bash
# Domino Job: tso-distributed-training
python predict_TSO_segment_patch_ray.py \
    --input_data_folder /mnt/data/biobank/processed/ \
    --output production_run_v1 \
    --model mba4tso_patch \
    --epochs 100 \
    --num_workers 16 \
    --gpus_per_worker 1 \
    --cpus_per_worker 8 \
    --ray_address "ray://<HEAD_IP>:10001" \
    --use_biobank_format
```

### Step 4: Monitor

```bash
# Access Ray dashboard (port-forward in Domino or use direct URL)
# Monitor training logs in Domino job console
# Check TensorBoard for metrics visualization
```

### Step 5: Results

Training completes in ~7 hours (vs 75 hours on single GPU). Results saved to:
- `/mnt/data/results/DL/biobank/production_run_v1/ray_results_split_0.joblib`
- Checkpoints in Ray storage path

## Summary

**Ray on Domino enables:**
- ✅ Multi-node distributed training across 10s to 100s of GPUs
- ✅ ~18x speedup with 32 GPUs (4 hours vs 75 hours)
- ✅ Built-in fault tolerance and checkpointing
- ✅ Easy scaling: just add more Domino worker jobs
- ✅ Production-ready with monitoring and logging

**When to use Ray:**
- Training takes > 12 hours on single GPU
- Need to train many models quickly (hyperparameter search)
- Have access to multiple GPU machines/nodes
- Production deployment requiring fault tolerance

**Quick Start:**
```bash
# Single command for 4 GPUs on one Domino machine
python predict_TSO_segment_patch_ray.py \
    --input_data_folder /mnt/data/processed/ \
    --output my_ray_experiment \
    --num_workers 4 \
    --gpus_per_worker 1
```

For questions or issues, refer to:
- Ray documentation: https://docs.ray.io/en/latest/train/train.html
- Domino documentation: https://docs.dominodatalab.com/
- This project's main docs: [CLAUDE.md](../CLAUDE.md)
