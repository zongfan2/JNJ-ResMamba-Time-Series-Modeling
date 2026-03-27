# DLR-TC Multi-GPU Training Guide

## Overview

DLR-TC now supports **multi-GPU training** using PyTorch's `DataParallel` for faster training on machines with multiple GPUs.

---

## Quick Start

### Single GPU (Default)
```bash
python predict_TSO_segment_patch_dlrtc.py \
    --input_h5 /path/to/data.h5 \
    --output TSO_dlrtc \
    --num_gpu 0  # Use GPU 0
```

### Multi-GPU
```bash
python predict_TSO_segment_patch_dlrtc.py \
    --input_h5 /path/to/data.h5 \
    --output TSO_dlrtc \
    --num_gpu 0,1,2,3 \  # Use GPUs 0, 1, 2, 3
    --multi_gpu           # Enable DataParallel
```

---

## Performance Scaling

### Expected Speedup

| GPUs | Speedup | Training Time (35 epochs) | Notes |
|------|---------|---------------------------|-------|
| 1 | 1.0x | ~52 hours (H5) | Baseline |
| 2 | 1.7-1.8x | ~29 hours | 70-80% efficiency |
| 4 | 3.2-3.5x | ~15 hours | 80-87% efficiency |
| 8 | 5.5-6.5x | ~8 hours | 70-80% efficiency |

**Why not linear?**: DataParallel has overhead from:
- Broadcasting model to all GPUs
- Gathering gradients back to main GPU
- Synchronization between GPUs

### Factors Affecting Speedup

**Good scalability** (near-linear speedup):
✅ Large batch size (24-32 per GPU)
✅ Large model (MBA4TSO_Patch with 128+ filters)
✅ Long sequences (1440 timesteps)
✅ Complex operations (Mamba blocks, attention)

**Poor scalability** (sublinear speedup):
❌ Small batch size (<16 per GPU)
❌ Small model (<10M parameters)
❌ Short sequences (<500 timesteps)
❌ Simple operations (linear layers only)

---

## Configuration

### Batch Size Adjustment

**Important**: Effective batch size = `batch_size` × `num_gpus`

```python
# In best_params
best_params = {
    'batch_size': 24,  # Per-GPU batch size with 4 GPUs
    # Effective batch size = 24 × 4 = 96
}
```

**Recommendations**:

| Scenario | batch_size | num_gpus | Effective | GPU VRAM/GPU |
|----------|-----------|----------|-----------|--------------|
| 1 GPU | 24 | 1 | 24 | ~12 GB |
| 2 GPUs | 24 | 2 | 48 | ~12 GB |
| 4 GPUs | 24 | 4 | 96 | ~12 GB |
| 4 GPUs | 32 | 4 | 128 | ~16 GB |
| 8 GPUs | 24 | 8 | 192 | ~12 GB |

### Learning Rate Scaling

**Rule of thumb**: Scale learning rate with effective batch size

```python
# Single GPU (batch=24)
best_params['lr'] = 0.001

# 4 GPUs (effective batch=96)
best_params['lr'] = 0.001 * sqrt(96/24) = 0.001 * 2 = 0.002
```

**Linear scaling** (conservative):
```python
lr = base_lr * (effective_batch / base_batch)
```

**Square root scaling** (recommended):
```python
lr = base_lr * sqrt(effective_batch / base_batch)
```

---

## Usage Examples

### Example 1: 4 GPUs with H5 Data

```bash
python predict_TSO_segment_patch_dlrtc.py \
    --input_h5 /path/to/data.h5 \
    --output TSO_dlrtc_4gpu \
    --model mba4tso_patch \
    --num_gpu 0,1,2,3 \
    --multi_gpu
```

**Expected**:
- 4 GPUs utilized
- ~15 hours training (vs ~52 hours single GPU)
- 3.5x speedup

### Example 2: 2 GPUs with Parquet Data

```bash
python predict_TSO_segment_patch_dlrtc.py \
    --input_data_folder /path/to/parquet/raw \
    --output TSO_dlrtc_2gpu \
    --model mba4tso_patch \
    --num_gpu 0,1 \
    --multi_gpu \
    --testing LOFO
```

**Expected**:
- 2 GPUs utilized
- ~38 hours training (vs ~70 hours single GPU)
- 1.8x speedup

### Example 3: Specific GPU Selection

If you have 8 GPUs but only want to use GPUs 2, 3, 5:

```bash
python predict_TSO_segment_patch_dlrtc.py \
    --input_h5 /path/to/data.h5 \
    --output TSO_dlrtc_custom \
    --num_gpu 2,3,5 \
    --multi_gpu
```

---

## Monitoring Multi-GPU Training

### Check GPU Utilization

**While training is running**, open a new terminal:

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Or for more detailed info
watch -n 1 'nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv'
```

**Expected output**:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.60.13    Driver Version: 525.60.13    CUDA Version: 12.0   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  On   | 00000000:00:1E.0 Off |                    0 |
| N/A   45C    P0   220W / 300W |  11234MiB / 16384MiB |     95%      Default |
|   1  Tesla V100-SXM2...  On   | 00000000:00:1F.0 Off |                    0 |
| N/A   46C    P0   218W / 300W |  11234MiB / 16384MiB |     93%      Default |
|   2  Tesla V100-SXM2...  On   | 00000000:00:20.0 Off |                    0 |
| N/A   44C    P0   221W / 300W |  11234MiB / 16384MiB |     94%      Default |
|   3  Tesla V100-SXM2...  On   | 00000000:00:21.0 Off |                    0 |
| N/A   45C    P0   219W / 300W |  11234MiB / 16384MiB |     96%      Default |
+-----------------------------------------------------------------------------+
```

**Good signs**:
- All GPUs show similar memory usage
- GPU utilization >85% on all GPUs
- Temperature stable (not thermal throttling)

**Warning signs**:
- GPU 0 much higher memory than others → Bottleneck at main GPU
- GPU utilization <50% → Batch size too small
- One GPU idle → Check `--num_gpu` argument

---

## Troubleshooting

### Issue 1: "CUDA out of memory"

**Error**:
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX MiB
```

**Cause**: Per-GPU batch size too large

**Solution**: Reduce batch size
```python
best_params['batch_size'] = 16  # Reduce from 24
```

---

### Issue 2: Only GPU 0 Working

**Symptom**: `nvidia-smi` shows only GPU 0 active

**Causes**:
1. Forgot `--multi_gpu` flag
2. Only one GPU specified in `--num_gpu`

**Solution**:
```bash
# Correct
python predict_TSO_segment_patch_dlrtc.py ... --num_gpu 0,1,2,3 --multi_gpu

# Wrong (missing --multi_gpu)
python predict_TSO_segment_patch_dlrtc.py ... --num_gpu 0,1,2,3
```

---

### Issue 3: No Speedup

**Symptom**: 4 GPUs same speed as 1 GPU

**Causes**:
1. Batch size too small (overhead dominates)
2. Model too small (not enough work to parallelize)
3. I/O bottleneck (data loading slower than compute)

**Solution**:
```python
# 1. Increase batch size
best_params['batch_size'] = 32  # From 24

# 2. Use H5 data (faster loading)
--input_h5 /path/to/data.h5  # Instead of parquet

# 3. Increase num_workers for data loading
# (Not applicable for DLR-TC - uses batch_generator)
```

---

### Issue 4: "Address already in use"

**Error**:
```
RuntimeError: Address already in use
```

**Cause**: Previous training crashed, GPU memory not freed

**Solution**:
```bash
# Kill any zombie Python processes
pkill -9 python

# Wait a few seconds
sleep 5

# Restart training
python predict_TSO_segment_patch_dlrtc.py ...
```

---

### Issue 5: Unbalanced GPU Memory

**Symptom**: GPU 0 uses 15 GB, others use 10 GB

**Cause**: DataParallel bottleneck (main GPU holds more state)

**Solution**: This is normal for DataParallel. Main GPU (GPU 0) handles:
- Model master copy
- Gradient aggregation
- Optimizer state

**Workaround** (advanced): Use `DistributedDataParallel` instead (requires code changes)

---

## Best Practices

### 1. Start with Single GPU

**Always validate with 1 GPU first**:
```bash
# First run - single GPU
python predict_TSO_segment_patch_dlrtc.py \
    --input_h5 /path/to/data.h5 \
    --output TSO_dlrtc_test \
    --num_gpu 0

# Verify training works, then scale up
python predict_TSO_segment_patch_dlrtc.py \
    --input_h5 /path/to/data.h5 \
    --output TSO_dlrtc_multigpu \
    --num_gpu 0,1,2,3 \
    --multi_gpu
```

### 2. Use H5 Data

**Parquet**: Data loading can bottleneck multi-GPU
**H5**: Faster loading, better for multi-GPU

```bash
# Preprocess once
python convert_parquet_to_h5.py --input_folder raw/ --output_h5 data.h5

# Train with multi-GPU (H5 is faster)
python predict_TSO_segment_patch_dlrtc.py --input_h5 data.h5 --multi_gpu --num_gpu 0,1,2,3
```

### 3. Monitor First Epoch

**Watch GPU utilization during first epoch**:
- If all GPUs active → Good!
- If only GPU 0 active → Check `--multi_gpu` flag
- If low utilization → Increase batch size

### 4. Optimal GPU Count

**Rule of thumb**: Diminishing returns after 4-8 GPUs

| GPU Count | Efficiency | Recommended |
|-----------|-----------|-------------|
| 1-2 | 85-90% | ✅ Always good |
| 2-4 | 80-87% | ✅ Sweet spot |
| 4-8 | 70-80% | ⚠️ Check bottlenecks |
| 8+ | 50-70% | ❌ Use DistributedDataParallel |

---

## Performance Tips

### 1. Batch Size

**Maximize GPU utilization without OOM**:

```python
# Start conservative
best_params['batch_size'] = 16

# Increase until OOM
best_params['batch_size'] = 24  # Good for 16GB GPUs
best_params['batch_size'] = 32  # Good for 32GB GPUs
```

### 2. Mixed Precision Training

**Enable AMP (Automatic Mixed Precision) for 2x speedup**:

```python
# In training loop (requires code modification)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# Forward pass with autocast
with autocast():
    outputs = model(X)
    loss = criterion(outputs, y)

# Backward with scaler
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Expected**: 1.5-2x speedup, ~30% less memory

### 3. Pin Memory

**For faster CPU→GPU transfer** (minor improvement):

```python
# In data loading
pad_X = pad_X.to(device, non_blocking=True)
pad_Y = pad_Y.to(device, non_blocking=True)
```

---

## Summary

### Quick Reference

**Single GPU**:
```bash
python predict_TSO_segment_patch_dlrtc.py --input_h5 data.h5 --output exp --num_gpu 0
```

**Multi-GPU (4 GPUs)**:
```bash
python predict_TSO_segment_patch_dlrtc.py --input_h5 data.h5 --output exp --num_gpu 0,1,2,3 --multi_gpu
```

**Expected speedup**:
- 2 GPUs: 1.7-1.8x faster
- 4 GPUs: 3.2-3.5x faster
- 8 GPUs: 5.5-6.5x faster

**For 34.6K segments, 35 epochs**:
- 1 GPU: ~52 hours
- 4 GPUs: ~15 hours
- 8 GPUs: ~8 hours

**Best practices**:
1. ✅ Use H5 data for faster loading
2. ✅ Start with single GPU to validate
3. ✅ Monitor `nvidia-smi` during first epoch
4. ✅ Keep batch size per GPU consistent (16-32)
5. ✅ Scale learning rate with effective batch size

---

**Multi-GPU training is now ready to use!** 🚀
