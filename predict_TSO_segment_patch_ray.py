# -*- coding: utf-8 -*-
"""
TSO Status Prediction Script - Multi-Node Distributed Training with Ray
Predicts 3-class status: 'other', 'non-wear', 'predictTSO'
Uses Ray Train for distributed training across multiple nodes/GPUs
Designed for Domino cloud environment
"""
import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from sklearn.metrics import f1_score
import ray
from ray import train
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
import tempfile

from Helpers.DL_models import setup_model
from Helpers.DL_helpers import (
    load_data_tso_patch,
    load_data_tso_patch_biobank,
    batch_generator,
    add_padding_tso_patch,
    measure_loss_tso,
    EarlyStopping,
    get_nb_steps
)

# ==================== Distributed Training Function ====================
def train_func_per_worker(config):
    """
    Training function that runs on each worker (GPU/node).
    This function is replicated across all workers in the Ray cluster.

    Args:
        config: Dictionary containing hyperparameters and data paths
    """
    import torch.distributed as dist

    # Get distributed training info
    rank = train.get_context().get_world_rank()
    local_rank = train.get_context().get_local_rank()
    world_size = train.get_context().get_world_size()

    # Set device for this worker
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(f"="*80)
        print(f"Distributed Training Setup:")
        print(f"  Total workers (world_size): {world_size}")
        print(f"  Current worker rank: {rank}")
        print(f"  Local GPU rank: {local_rank}")
        print(f"  Device: {device}")
        print(f"="*80)

    # Load data on rank 0 and share metadata
    # In production, you might want to use Ray datasets for better data parallelism
    if rank == 0:
        print("Loading data on rank 0...")

        # Choose data loader based on data format
        if config['use_biobank_format']:
            df_metadata = load_data_tso_patch_biobank(
                config['input_data_folder'],
                include_subject_filter=config.get('include_subject_filter'),
                remove_subject_filter=config.get('remove_subject_filter'),
                max_seq_length=config.get('max_seq_length', 86400)
            )
        else:
            df_metadata = load_data_tso_patch(
                config['input_data_folder'],
                include_subject_filter=config.get('include_subject_filter'),
                remove_subject_filter=config.get('remove_subject_filter'),
                max_seq_length=config.get('max_seq_length', 86400)
            )

        # Split data (LOFO or LOSO)
        split_id = config['split_id']
        testing_strategy = config['testing']

        if testing_strategy == "LOSO":
            PID_name = "PID"
        else:  # LOFO
            PID_name = "FOLD"

        splits = df_metadata[PID_name].unique()
        print(f"Total splits: {len(splits)}, Current split: {split_id}")

        # Create train/val/test splits
        test_mask = df_metadata[PID_name] == splits[split_id]
        train_mask_full = ~test_mask

        # Get train segments and split into train/val
        train_segments = df_metadata[train_mask_full]['segment'].unique()
        np.random.seed(42)
        np.random.shuffle(train_segments)
        split_point = int(len(train_segments) * 0.8)

        train_segments_final = train_segments[:split_point]
        val_segments = train_segments[split_point:]

        train_mask = df_metadata['segment'].isin(train_segments_final)
        val_mask = df_metadata['segment'].isin(val_segments)

        df_train = df_metadata[train_mask].reset_index(drop=True)
        df_val = df_metadata[val_mask].reset_index(drop=True)
        df_test = df_metadata[test_mask].reset_index(drop=True)

        print(f"Data split - Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

        # Store in shared memory or pass to other ranks
        # For simplicity, we'll use Ray's object store
        data_ref = ray.put({
            'train': df_train,
            'val': df_val,
            'test': df_test
        })
    else:
        data_ref = None

    # Broadcast data reference to all workers
    if world_size > 1:
        data_refs = [data_ref]
        dist.broadcast_object_list(data_refs, src=0)
        data_ref = data_refs[0]

    # Get data from Ray object store
    data_dict = ray.get(data_ref)
    df_train = data_dict['train']
    df_val = data_dict['val']
    df_test = data_dict['test']

    # Setup model
    best_params = config['best_params']
    max_seq_len = config.get('max_seq_len', 1440)

    model = setup_model(
        config['model'],
        None,
        max_seq_len,
        best_params,
        pretraining=False,
        num_classes=3
    )
    model = model.to(device)

    # Wrap with DistributedDataParallel
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False
        )
        if rank == 0:
            print("Model wrapped with DistributedDataParallel")

    # Setup optimizer
    optimizer = optim.RMSprop(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=best_params['lr']
    )

    # Setup scheduler
    nb_steps = get_nb_steps(
        df=df_train,
        batch_size=best_params['batch_size'],
        stratify='undersample_TSO',
        shuffle=True
    )
    # Adjust steps for distributed training
    nb_steps = nb_steps // world_size

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=nb_steps,
        T_mult=1
    )

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}, Trainable: {trainable_params:,}")
        print(f"Steps per epoch per worker: {nb_steps}")

    # Training loop
    early_stopping = EarlyStopping(patience=15, verbose=(rank == 0))

    for epoch in range(config['epochs']):
        if rank == 0:
            print(f"\nEpoch {epoch+1}/{config['epochs']}")

        # Train
        model.train()
        train_metrics = train_one_epoch(
            model, df_train, best_params, device, optimizer, scheduler,
            config, rank, world_size
        )

        if rank == 0:
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, F1 avg: {train_metrics['f1_avg']:.4f}")

        # Validation (only on rank 0 to avoid redundant computation)
        if epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                val_metrics = validate_one_epoch(
                    model, df_val, best_params, device, config, rank, world_size
                )

            if rank == 0:
                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, F1 avg: {val_metrics['f1_avg']:.4f}")

                # Early stopping check
                early_stopping(val_metrics['loss'], model)

                # Report metrics to Ray Train
                train.report({
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'train_f1_avg': train_metrics['f1_avg'],
                    'val_loss': val_metrics['loss'],
                    'val_f1_avg': val_metrics['f1_avg'],
                })

                # Save checkpoint
                checkpoint_dir = tempfile.mkdtemp()
                checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")

                # Save without DDP wrapper
                model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_f1_avg': val_metrics['f1_avg'],
                }, checkpoint_path)

                train.report({}, checkpoint=train.Checkpoint.from_directory(checkpoint_dir))

                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    break

        # Synchronize all workers
        if world_size > 1:
            dist.barrier()

    # Final evaluation on test set (rank 0 only)
    if rank == 0:
        print(f"\nEvaluating on test set...")
        model.eval()
        with torch.no_grad():
            test_metrics = validate_one_epoch(
                model, df_test, best_params, device, config, rank, world_size
            )

        print(f"\nTest Results:")
        print(f"  Loss: {test_metrics['loss']:.4f}")
        print(f"  F1 avg: {test_metrics['f1_avg']:.4f}")
        print(f"  F1 other: {test_metrics['f1_other']:.4f}")
        print(f"  F1 nonwear: {test_metrics['f1_nonwear']:.4f}")
        print(f"  F1 TSO: {test_metrics['f1_tso']:.4f}")

        train.report({
            'test_loss': test_metrics['loss'],
            'test_f1_avg': test_metrics['f1_avg'],
            'test_f1_other': test_metrics['f1_other'],
            'test_f1_nonwear': test_metrics['f1_nonwear'],
            'test_f1_tso': test_metrics['f1_tso'],
        })


def train_one_epoch(model, df, best_params, device, optimizer, scheduler, config, rank, world_size):
    """Train for one epoch with distributed data parallelism."""
    epoch_loss = 0.0
    batches = 0

    all_preds_other, all_preds_nonwear, all_preds_tso = [], [], []
    all_labels_other, all_labels_nonwear, all_labels_tso = [], [], []

    seg_column = 'segment'

    # Each worker processes a subset of batches
    for batch_idx, batch in enumerate(batch_generator(
        df=df,
        batch_size=best_params['batch_size'],
        stratify=False,
        shuffle=True,
        seg_column=seg_column
    )):
        # Distributed sampling: each worker processes every world_size-th batch
        if batch_idx % world_size != rank:
            continue

        # Prepare batch
        pad_X, pad_Y, x_lens = add_padding_tso_patch(
            batch,
            device=device,
            seg_column=seg_column,
            max_seq_len=config.get('max_seq_len', 1440),
            patch_size=best_params['patch_size'],
            padding_value=best_params['padding_value']
        )

        # Forward pass
        outputs = model(pad_X, x_lens)

        # Calculate loss
        total_loss = measure_loss_tso(outputs, pad_Y, x_lens)

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Record loss
        epoch_loss += total_loss.item()
        batches += 1

        # Collect predictions
        preds = torch.sigmoid(outputs).cpu().detach().numpy()
        labels = pad_Y.cpu().numpy()

        for i in range(len(x_lens)):
            valid_len = int(x_lens[i])
            valid_preds = preds[i, :valid_len, :]
            valid_labels = labels[i, :valid_len]

            labels_onehot = np.zeros((valid_len, 3))
            for j, lbl in enumerate(valid_labels):
                if lbl >= 0 and lbl < 3:
                    labels_onehot[j, lbl] = 1

            all_preds_other.extend(valid_preds[:, 0].tolist())
            all_preds_nonwear.extend(valid_preds[:, 1].tolist())
            all_preds_tso.extend(valid_preds[:, 2].tolist())
            all_labels_other.extend(labels_onehot[:, 0].tolist())
            all_labels_nonwear.extend(labels_onehot[:, 1].tolist())
            all_labels_tso.extend(labels_onehot[:, 2].tolist())

    # Calculate metrics
    avg_loss = epoch_loss / batches if batches > 0 else 0

    all_preds_other = np.array(all_preds_other)
    all_preds_nonwear = np.array(all_preds_nonwear)
    all_preds_tso = np.array(all_preds_tso)
    all_labels_other = np.array(all_labels_other)
    all_labels_nonwear = np.array(all_labels_nonwear)
    all_labels_tso = np.array(all_labels_tso)

    f1_other = f1_score(all_labels_other, (all_preds_other > 0.5).astype(int), zero_division=0)
    f1_nonwear = f1_score(all_labels_nonwear, (all_preds_nonwear > 0.5).astype(int), zero_division=0)
    f1_tso = f1_score(all_labels_tso, (all_preds_tso > 0.5).astype(int), zero_division=0)
    f1_avg = (f1_other + f1_nonwear + f1_tso) / 3

    return {
        'loss': avg_loss,
        'f1_avg': f1_avg,
        'f1_other': f1_other,
        'f1_nonwear': f1_nonwear,
        'f1_tso': f1_tso
    }


def validate_one_epoch(model, df, best_params, device, config, rank, world_size):
    """Validate for one epoch."""
    epoch_loss = 0.0
    batches = 0

    all_preds_other, all_preds_nonwear, all_preds_tso = [], [], []
    all_labels_other, all_labels_nonwear, all_labels_tso = [], [], []

    seg_column = 'segment'

    for batch in batch_generator(
        df=df,
        batch_size=best_params['batch_size'],
        stratify=False,
        shuffle=False,
        seg_column=seg_column
    ):
        # Prepare batch
        pad_X, pad_Y, x_lens = add_padding_tso_patch(
            batch,
            device=device,
            seg_column=seg_column,
            max_seq_len=config.get('max_seq_len', 1440),
            patch_size=best_params['patch_size'],
            padding_value=best_params['padding_value']
        )

        # Forward pass
        outputs = model(pad_X, x_lens)

        # Calculate loss
        total_loss = measure_loss_tso(outputs, pad_Y, x_lens)

        # Record loss
        epoch_loss += total_loss.item()
        batches += 1

        # Collect predictions
        preds = torch.sigmoid(outputs).cpu().detach().numpy()
        labels = pad_Y.cpu().numpy()

        for i in range(len(x_lens)):
            valid_len = int(x_lens[i])
            valid_preds = preds[i, :valid_len, :]
            valid_labels = labels[i, :valid_len]

            labels_onehot = np.zeros((valid_len, 3))
            for j, lbl in enumerate(valid_labels):
                if lbl >= 0 and lbl < 3:
                    labels_onehot[j, lbl] = 1

            all_preds_other.extend(valid_preds[:, 0].tolist())
            all_preds_nonwear.extend(valid_preds[:, 1].tolist())
            all_preds_tso.extend(valid_preds[:, 2].tolist())
            all_labels_other.extend(labels_onehot[:, 0].tolist())
            all_labels_nonwear.extend(labels_onehot[:, 1].tolist())
            all_labels_tso.extend(labels_onehot[:, 2].tolist())

    # Calculate metrics
    avg_loss = epoch_loss / batches if batches > 0 else 0

    all_preds_other = np.array(all_preds_other)
    all_preds_nonwear = np.array(all_preds_nonwear)
    all_preds_tso = np.array(all_preds_tso)
    all_labels_other = np.array(all_labels_other)
    all_labels_nonwear = np.array(all_labels_nonwear)
    all_labels_tso = np.array(all_labels_tso)

    f1_other = f1_score(all_labels_other, (all_preds_other > 0.5).astype(int), zero_division=0)
    f1_nonwear = f1_score(all_labels_nonwear, (all_preds_nonwear > 0.5).astype(int), zero_division=0)
    f1_tso = f1_score(all_labels_tso, (all_preds_tso > 0.5).astype(int), zero_division=0)
    f1_avg = (f1_other + f1_nonwear + f1_tso) / 3

    return {
        'loss': avg_loss,
        'f1_avg': f1_avg,
        'f1_other': f1_other,
        'f1_nonwear': f1_nonwear,
        'f1_tso': f1_tso
    }


# ==================== Main Entry Point ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TSO Distributed Training with Ray')
    parser.add_argument('--input_data_folder', type=str, required=True, help='Path to input data folder')
    parser.add_argument('--output', type=str, required=True, help='Output folder name')
    parser.add_argument('--model', type=str, default="mba4tso_patch", help='Model name')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--testing', type=str, default="LOFO", help='Testing strategy: LOFO or LOSO')
    parser.add_argument('--split_id', type=int, default=0, help='Split ID for LOFO/LOSO')
    parser.add_argument('--use_biobank_format', action='store_true', help='Use biobank data format')

    # Ray/Distributed training arguments
    parser.add_argument('--num_workers', type=int, default=4, help='Number of distributed workers (GPUs/nodes)')
    parser.add_argument('--cpus_per_worker', type=int, default=8, help='CPUs per worker')
    parser.add_argument('--gpus_per_worker', type=int, default=1, help='GPUs per worker')
    parser.add_argument('--ray_address', type=str, default=None,
                       help='Ray cluster address (e.g., "ray://localhost:10001"). If None, starts local cluster.')

    args = parser.parse_args()

    # Initialize Ray
    if args.ray_address:
        print(f"Connecting to Ray cluster at: {args.ray_address}")
        ray.init(address=args.ray_address)
    else:
        print("Starting local Ray cluster")
        ray.init()

    print(f"Ray cluster resources: {ray.cluster_resources()}")

    # Create output folders
    dataset_name = os.path.basename(args.input_data_folder.rstrip("/raw"))
    results_folder = f"/mnt/data/GENEActive-featurized/results/DL/{dataset_name}/{args.output}/"
    training_output_folder = os.path.join(results_folder, "training/")
    predictions_output_folder = os.path.join(training_output_folder, "predictions/")

    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(training_output_folder, exist_ok=True)
    os.makedirs(predictions_output_folder, exist_ok=True)

    print(f"Results will be saved to: {results_folder}")

    # Model hyperparameters
    best_params = {
        'batch_size': 16,
        'num_filters': 128,
        'dropout': 0.3,
        'droppath': 0.3,
        'kernel_f': 3,
        'kernel_MBA': 7,
        'num_feature_layers': 6,
        'blocks_MBA': 6,
        'featurelayer': 'ResNet',
        'lr': 0.0003,
        'w_other': 1.0,
        'w_nonwear': 1.0,
        'w_tso': 1.0,
        'padding_value': 0.0,
        'patch_size': 1200,
    }

    # Training configuration
    train_config = {
        'input_data_folder': args.input_data_folder,
        'output': args.output,
        'model': args.model,
        'epochs': args.epochs,
        'testing': args.testing,
        'split_id': args.split_id,
        'use_biobank_format': args.use_biobank_format,
        'best_params': best_params,
        'max_seq_len': 1440,
    }

    # Ray Train scaling configuration
    scaling_config = ScalingConfig(
        num_workers=args.num_workers,
        use_gpu=args.gpus_per_worker > 0,
        resources_per_worker={
            "CPU": args.cpus_per_worker,
            "GPU": args.gpus_per_worker
        },
    )

    # Ray Train run configuration
    run_config = RunConfig(
        name=f"tso_ray_{args.output}_split{args.split_id}",
        storage_path=results_folder,
        checkpoint_config=CheckpointConfig(
            num_to_keep=3,
            checkpoint_score_attribute="val_f1_avg",
            checkpoint_score_order="max",
        ),
    )

    # Create Ray TorchTrainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config=train_config,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    print(f"\n{'='*80}")
    print(f"Starting distributed training with Ray")
    print(f"  Workers: {args.num_workers}")
    print(f"  GPUs per worker: {args.gpus_per_worker}")
    print(f"  CPUs per worker: {args.cpus_per_worker}")
    print(f"  Total GPUs: {args.num_workers * args.gpus_per_worker}")
    print(f"{'='*80}\n")

    # Run training
    result = trainer.fit()

    print(f"\n{'='*80}")
    print("Training complete!")
    print(f"Best checkpoint: {result.checkpoint}")
    print(f"Final metrics: {result.metrics}")
    print(f"{'='*80}")

    # Save final results
    results_file = os.path.join(results_folder, f"ray_results_split_{args.split_id}.joblib")
    joblib.dump({
        'result': result,
        'config': train_config,
        'args': vars(args)
    }, results_file)
    print(f"Results saved to: {results_file}")

    # Extract and save test metrics to CSV
    if result.metrics:
        test_metrics_dict = {
            'split_id': args.split_id,
            'test_loss': result.metrics.get('test_loss', None),
            'test_f1_avg': result.metrics.get('test_f1_avg', None),
            'test_f1_other': result.metrics.get('test_f1_other', None),
            'test_f1_nonwear': result.metrics.get('test_f1_nonwear', None),
            'test_f1_tso': result.metrics.get('test_f1_tso', None),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model': args.model,
            'batch_size': train_config['batch_size'],
            'lr': train_config['lr'],
            'num_workers': args.num_workers,
            'gpus_per_worker': args.gpus_per_worker,
        }

        # Save individual split metrics as CSV
        metrics_csv_file = os.path.join(predictions_output_folder, f"final_metrics_split_{args.split_id}.csv")
        pd.DataFrame([test_metrics_dict]).to_csv(metrics_csv_file, index=False, encoding='utf-8')
        print(f"Test metrics saved to: {metrics_csv_file}")

        # Append to consolidated metrics file (for cross-validation summary)
        consolidated_metrics_file = os.path.join(predictions_output_folder, "all_splits_metrics.csv")
        if os.path.exists(consolidated_metrics_file):
            # Append to existing file
            existing_df = pd.read_csv(consolidated_metrics_file)
            updated_df = pd.concat([existing_df, pd.DataFrame([test_metrics_dict])], ignore_index=True)
            updated_df.to_csv(consolidated_metrics_file, index=False, encoding='utf-8')
        else:
            # Create new file
            pd.DataFrame([test_metrics_dict]).to_csv(consolidated_metrics_file, index=False, encoding='utf-8')
        print(f"Consolidated metrics updated: {consolidated_metrics_file}")

    ray.shutdown()
