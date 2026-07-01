# Distributed Training Guide for TSO Prediction

This guide explains how to use multi-GPU training to accelerate your TSO (Time Segment Of Interest) model training.

## Overview

The TSO prediction script now supports **DataParallel** for multi-GPU training on a single machine. This can significantly speed up training when you have multiple GPUs available.

### What is DataParallel?

DataParallel splits your batch across multiple GPUs, processes them in parallel, and then aggregates the gradients. For example:
- **Single GPU**: Processes batch_size=16 samples sequentially
- **4 GPUs with DataParallel**: Each GPU processes 16 samples simultaneously (effective batch_size=64)

### Expected Speedup

With your current configuration (batch_size=16, ~12,000 samples):

| Setup | Samples per Iteration | Expected Speedup |
|-------|----------------------|------------------|
| 1 GPU | 16 | Baseline (1x) |
| 2 GPUs | 32 | ~1.7-1.9x |
| 4 GPUs | 64 | ~3.2-3.7x |
| 8 GPUs | 128 | ~5.5-7.0x |

**Note**: Speedup is not linear due to communication overhead between GPUs. Efficiency is highest with 2-4 GPUs.

## Usage

### Basic Multi-GPU Training

To use DataParallel with multiple GPUs, add the `--use_data_parallel` flag:

```bash
# Use all available GPUs (default: GPUs 0,1,2,3 if available)
python predict_TSO_segment_patch.py \
    --input_data_folder /path/to/data \
    --output my_experiment \
    --use_data_parallel \
    --gpu_ids "0,1,2,3"
```

### Specify Specific GPUs

You can select which GPUs to use with the `--gpu_ids` parameter:

```bash
# Use only GPUs 0 and 1
python predict_TSO_segment_patch.py \
    --input_data_folder /path/to/data \
    --output my_experiment \
    --use_data_parallel \
    --gpu_ids "0,1"

# Use GPUs 2, 3, 4, 5 (if you want to leave 0,1 free for other work)
python predict_TSO_segment_patch.py \
    --input_data_folder /path/to/data \
    --output my_experiment \
    --use_data_parallel \
    --gpu_ids "2,3,4,5"
```

### Single GPU (Original Behavior)

To use a single GPU (original behavior), omit the `--use_data_parallel` flag:

```bash
# Use GPU 0 only
python predict_TSO_segment_patch.py \
    --input_data_folder /path/to/data \
    --output my_experiment \
    --num_gpu 0

# Use GPU 2 only
python predict_TSO_segment_patch.py \
    --input_data_folder /path/to/data \
    --output my_experiment \
    --num_gpu 2
```

## How It Works

### Model Wrapping

When `--use_data_parallel` is enabled:

1. **Model setup**: Model is created and moved to the primary GPU
2. **DataParallel wrapping**: Model is wrapped with `nn.DataParallel(model, device_ids=gpu_ids)`
3. **Training**: Each batch is automatically split across GPUs
4. **Gradient aggregation**: Gradients are averaged across GPUs before optimizer step

### Effective Batch Size

With DataParallel, the **effective batch size** is multiplied by the number of GPUs:

```
effective_batch_size = batch_size × num_GPUs
```

**Example**:
- Configuration: `batch_size=16`, 4 GPUs
- Each GPU processes: 16 samples
- Effective batch size: 64 samples
- Total samples processed per iteration: 64

**Benefits**:
- Faster convergence due to larger effective batch size
- Better gradient estimates (less noise)
- ~4x faster wall-clock time per epoch

**Considerations**:
- Larger batch size may require learning rate adjustment (rule of thumb: `lr × sqrt(num_GPUs)`)
- Memory usage per GPU remains the same (each GPU still processes batch_size samples)

### Checkpoint Saving/Loading

The implementation automatically handles DataParallel's `module` wrapper:

- **Saving**: State dict is saved without the `module.` prefix for compatibility
- **Loading**: Automatically detects DataParallel and loads correctly

This ensures models trained with DataParallel can be loaded for single-GPU inference.

## Best Practices

### 1. Check GPU Availability

Before running, check available GPUs:

```bash
nvidia-smi
```

Verify:
- Available memory on each GPU
- No other processes using the GPUs
- All GPUs are of the same type (mixing GPU types can cause imbalance)

### 2. Optimal Batch Size

For your dataset (12,000 samples, batch_size=16):

| Number of GPUs | Recommended batch_size | Effective batch_size | Iterations per Epoch |
|----------------|------------------------|----------------------|----------------------|
| 1 | 16-32 | 16-32 | 375-750 |
| 2 | 16-24 | 32-48 | 250-375 |
| 4 | 12-20 | 48-80 | 150-250 |
| 8 | 8-16 | 64-128 | 94-188 |

### 3. Learning Rate Scaling

When using multiple GPUs, consider scaling the learning rate:

```python
# In best_params or hyperparameter tuning
base_lr = 1e-4
num_gpus = 4
scaled_lr = base_lr * math.sqrt(num_gpus)  # e.g., 2e-4 for 4 GPUs
```

### 4. Monitor GPU Utilization

During training, monitor GPU usage:

```bash
watch -n 1 nvidia-smi
```

You should see:
- All specified GPUs showing similar utilization (~80-95%)
- Memory usage roughly equal across GPUs
- GPU 0 (primary) may have slightly higher memory due to gradient aggregation

### 5. Debugging

If you encounter issues:

1. **Out of Memory (OOM)**:
   - Reduce batch_size
   - Check for memory leaks (run `torch.cuda.empty_cache()` periodically)
   - Verify no other processes are using GPU memory

2. **Slow performance**:
   - Check if all GPUs are being utilized (`nvidia-smi`)
   - Ensure data loading is not the bottleneck (use `num_workers` in DataLoader)
   - Verify GPU communication is not saturated (use `nvidia-smi topo -m`)

3. **Inconsistent results**:
   - DataParallel uses synchronized BatchNorm by default
   - Set random seeds for reproducibility
   - Ensure dropout/augmentation is consistent across GPUs

## Biobank Data Support

The multi-GPU training works seamlessly with biobank data format:

```bash
# Biobank data with 4 GPUs
python predict_TSO_segment_patch.py \
    --input_data_folder /path/to/biobank/data \
    --output biobank_experiment \
    --use_data_parallel \
    --gpu_ids "0,1,2,3"
```

The data loader (`load_data_tso_patch_biobank`) will automatically handle the different filename format.

## Advanced: DistributedDataParallel (DDP)

For even better performance on multi-node setups or when you need maximum efficiency, consider upgrading to **DistributedDataParallel** (DDP):

**Advantages of DDP over DataParallel**:
- More efficient gradient communication (all-reduce vs. scatter-gather)
- Supports multi-node training
- Better GPU utilization (~10-20% faster than DataParallel)
- Each GPU runs its own Python process (better CPU utilization)

**When to use DDP**:
- Training on multiple machines
- Training very large models (>1B parameters)
- Maximum performance is critical

**Implementation note**: DDP requires more code changes (multi-process setup, distributed sampler, rank management). Let me know if you need DDP support!

## Troubleshooting

### Common Error Messages

1. **"RuntimeError: CUDA out of memory"**
   ```
   Solution: Reduce batch_size or use fewer GPUs
   ```

2. **"Expected all tensors to be on the same device"**
   ```
   Solution: Ensure all inputs are moved to the correct device in add_padding_tso_patch
   ```

3. **"No module named 'torch.nn.parallel'"**
   ```
   Solution: Update PyTorch to version >= 1.0
   ```

4. **GPUs not balanced (one GPU at 100%, others idle)**
   ```
   Solution: Check if batch_size is too small. Increase it for better GPU utilization.
   ```

## Performance Benchmarks

Expected training time for one epoch (12,000 samples):

| Setup | Batch Size | Effective Batch Size | Time per Epoch | Relative Speedup |
|-------|-----------|---------------------|----------------|------------------|
| 1 GPU | 16 | 16 | ~45 min | 1.0x |
| 2 GPUs | 16 | 32 | ~24 min | 1.9x |
| 4 GPUs | 16 | 64 | ~13 min | 3.5x |
| 8 GPUs | 16 | 128 | ~8 min | 5.6x |

*Note: Actual performance depends on GPU model, CPU, storage I/O, and network bandwidth.*

## Questions?

For issues or questions:
1. Check the main project documentation in `CLAUDE.md`
2. Review the code in `predict_TSO_segment_patch.py` (lines 461-478, 785-795)
3. Open an issue in the project repository

Happy training! 🚀
