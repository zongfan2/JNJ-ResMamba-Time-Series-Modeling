
import os
import sys
import torch
import numpy as np
import pandas as pd
import h5py
import argparse
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from data.loading import *


def prepare_df_for_h5(df):
    df_copy = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_dtype(df[col]):
            df_copy[col] = df[col].astype(str)
        elif df[col].dtype == 'object' and not pd.api.types.is_string_dtype(df[col]):
            try:
                df_copy[col] = df_copy[col].astype(str)
            except:
                print(f"Warning: Column {col} has complex object that may not save properly")
    return df_copy


def main():
    parser = argparse.ArgumentParser(description='Prepare and save data for model training.')
    parser.add_argument('--input_data_folder', type=str, required=True, help='Path to the input data file')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save processed data')
    parser.add_argument('--format', type=str, default="csv", help="Saved data format")
    parser.add_argument('--time_length', type=int, default=60, help='Time sequence length')
    parser.add_argument('--testing', type=str, default="LOFO", help='Testing mechanism (LOSO, LOFO, production)')
    parser.add_argument('--data_augmentation', type=int, default=0, help='Number of augmentation iterations')
    parser.add_argument('--norm_scratch_length', type=bool, default=True, help='Normalize scratch length')
    parser.add_argument('--use_TSO', type=bool, default=True, help='Use TSO data filtering')
    parser.add_argument('--energy_th', type=float, default=1.0, help='Energy threshold for filtering')

    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_folder, exist_ok=True)
    
    print(f"Loading data from {args.input_data_folder} at {datetime.now()}")
    
    # Load data based on dataset type
    if 'nsucl' in args.input_data_folder:
        df = load_data_nsucl(args.input_data_folder)
    else:
        df = load_data(args.input_data_folder, max_seq_length=args.time_length, 
                      norm_scratch_length=args.norm_scratch_length, use_TSO=args.use_TSO, energy_th=args.energy_th)
    
    # Calculate max sequence length for metadata
    max_seq_len = df.groupby('segment').count().max(1).max()+1
    print(f"Max sequence length: {max_seq_len}")
    
    # Define PID name based on testing strategy
    if args.testing == "production":
        df["FOLD"] = "All"
        PID_name = "FOLD"
        splits = ["All"]
    elif args.testing == "LOSO":
        PID_name = "PID"
        splits = df[PID_name].unique()
    else:  # LOFO
        PID_name = "FOLD"
        splits = df[PID_name].unique()
    
    # Process each split
    for split_id in splits:
        print(f"Processing split: {split_id}")
        
        # Create train/test split
        if args.testing == "production":
            df_test = df
            df_train = df
        else:
            df_test = df[df[PID_name] == split_id]
            df_train = df[df[PID_name] != split_id]
            
            if args.use_TSO:
                df_train = df_train[df_train.inTSO==True]
                df_test = df_test[df_test.inTSO==True]
        
        # Create validation split
        random_indices = df_subset_segments(df_train, 0.2, False)
        df_val = df_train.iloc[df_train.index.isin(random_indices)]
        df_train = df_train.iloc[~df_train.index.isin(random_indices)]
        
        if args.data_augmentation > 0:
            print(f"Augmenting training data with {args.data_augmentation} iterations. {datetime.now()}")
            intitial_size=df_train.size
            if 'nsucl' in args.input_data_folder:
                df_train=augment_dataset(df_train, args.data_augmentation, verbose=False)  
            else:
                df_train=augment_dataset(df_train, args.data_augmentation, interchange=False, verbose=False, norm_scratch_length=args.norm_scratch_length)
            max_seq_len=df_train.groupby('segment').count().max(1).max()+1
            print(f"Training data augmented from {intitial_size} to {df_train.size}. Max sequence length: {max_seq_len}. {datetime.now()}") 

        # Scale features
        scaler = StandardScaler()
        c_to_scale = ['x', 'y', 'z']
        df_train.loc[:, c_to_scale] = scaler.fit_transform(df_train[c_to_scale])
        df_val.loc[:, c_to_scale] = scaler.transform(df_val[c_to_scale])
        df_test.loc[:, c_to_scale] = scaler.transform(df_test[c_to_scale])

        if args.format == "h5":
            df_train_h5 = prepare_df_for_h5(df_train)
            df_val_h5 = prepare_df_for_h5(df_val)
            df_test_h5 = prepare_df_for_h5(df_test)
            
            # Save to H5 file
            h5_file_path = os.path.join(args.output_folder, f"{split_id}_data.h5")
            with h5py.File(h5_file_path, 'w') as f:
                # Save metadata
                meta = f.create_group('metadata')
                meta.attrs['max_seq_len'] = max_seq_len
                meta.attrs['split_id'] = str(split_id)
                meta.attrs['testing_method'] = args.testing
                
                # Save scaler parameters
                scaler_group = f.create_group('scaler')
                scaler_group.create_dataset('mean', data=scaler.mean_)
                scaler_group.create_dataset('scale', data=scaler.scale_)

                dtype_group = f.create_group('dtype_info')
                dtype_info = {col: str(df_train[col].dtype) for col in df_train.columns}
                for col, dtype in dtype_info.items():
                    dtype_group.attrs[col] = dtype
                
                # Save train data
                train_group = f.create_group('train')
                for col in df_train_h5.columns:
                    # Handle different data types appropriately
                    data = df_train_h5[col].values
                    if pd.api.types.is_string_dtype(data):
                        dt = h5py.special_dtype(vlen=str)
                        train_group.create_dataset(col, data=data, dtype=dt)
                    else:
                        train_group.create_dataset(col, data=data)
                
                # Save validation data
                val_group = f.create_group('val')
                for col in df_val_h5.columns:
                    data = df_val_h5[col].values
                    if pd.api.types.is_string_dtype(data):
                        dt = h5py.special_dtype(vlen=str)
                        val_group.create_dataset(col, data=data, dtype=dt)
                    else:
                        val_group.create_dataset(col, data=data)
                
                # Save test data
                test_group = f.create_group('test')
                for col in df_test_h5.columns:
                    data = df_test_h5[col].values
                    if pd.api.types.is_string_dtype(data):
                        dt = h5py.special_dtype(vlen=str)
                        test_group.create_dataset(col, data=data, dtype=dt)
                    else:
                        test_group.create_dataset(col, data=data)
            print(f"Saved data to {h5_file_path}")
        elif args.format == "csv":
            # Save to CSV files
            df_train.to_csv(os.path.join(args.output_folder, f"{split_id}_train.csv"), index=False)
            df_test.to_csv(os.path.join(args.output_folder, f"{split_id}_test.csv"), index=False)
            df_val.to_csv(os.path.join(args.output_folder, f"{split_id}_val.csv"), index=False)
        else:
            print(f"Unsupported format: {args.format}")

        
    
if __name__ == "__main__":
    main()
    