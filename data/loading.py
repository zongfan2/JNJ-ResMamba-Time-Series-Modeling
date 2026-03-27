# -*- coding: utf-8 -*-
"""
Split from DL_helpers.py - Modular code structure
@author: MBoukhec (original)
"""

import glob
import os
import random
import re
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import h5py
import torch

def load_sequence_data(path, remove_subject_filter,include_subject_filter, list_features, motion_filter,group=True,only_positive=False,max_seq_length=None,sf=20):
    start_time = datetime.now()
    print("---------------> Starting load_test_data() function:  ", start_time)
    files = sorted(glob.glob(os.path.join(path, f"*.parquet.gzip")))
    if remove_subject_filter is not None:
        files = [file for file in files if all(PID not in file for PID in remove_subject_filter)]
    
    if include_subject_filter is not None:
        files = [file for file in files if include_subject_filter in file]
    df_motion_list = []
    df_nomotion_list = []
    df=pd.DataFrame()
    print(f"Total Files: {len(files)}, path={path}")
    for counter, file in enumerate(files):#files[:50]
        current_subject = re.findall(r"US\d+", file)[0]
        day=os.path.basename(file).rstrip(".parquet.gzip").split('_')[-1]
        if counter % 50 == 0: 
            print(f"---->Processing file {counter} out of {len(files)} at {datetime.now()}") 
        df_temp = pd.read_parquet(file)
        # only use data in TSO (scratch) or predicted TSO (sleep)
        df_temp=df_temp[(df_temp.inTSO==True) | (df_temp.predictTSO==True) ]
        if df_temp.empty or len(df_temp.columns) < 2:
            custom_print(f"Warning: Data file empty. \nFile: {file}")
            continue
            
        df_temp['wrist']=np.where(df_temp.wrist.str.lower()=="left",1,0)
        df_temp['SUBJECT'] = current_subject
        df_temp['DAY'] = day
        
        if motion_filter:
            df_temp= df_temp[df_temp['stationary']==False]
        if only_positive:
            df_temp['segment'] = (df_temp['SUBJECT']+'_'+df_temp['wrist'].astype(str)+'_'+df_temp['DAY']+'_'+df_temp['segment'].astype(str)).values 
            df_temp['segment_scratch']=df_temp.groupby('segment')['scratch'].transform('max')
            df_temp=df_temp[df_temp.segment_scratch==True] # only focus on False negatives
        if  not df_temp.empty:
            df_motion_list.append(df_temp[[col for col in df_temp.columns if col not in ['index','diff','diff_flag','diff_batch','change']]])
    # -------------------------
    # Clean & Transform Dataset
    # -------------------------
    
    del df_temp
    print(f"---------------> Starting dataframes concatenation..... Time spent so far, from start:   {datetime.now() - start_time}")
    df_motion = pd.concat(df_motion_list, ignore_index=True)
    del df_motion_list
    if not only_positive:
        df_motion['segment'] = (df_motion['SUBJECT']+'_'+df_motion['wrist'].astype(str)+'_'+df_motion['DAY']+'_'+df_motion['segment'].astype(str)).values 
        
    print(f"Concatenated Dataframe Shape: {df_motion.shape}, timestamp: {datetime.now()}")
    
    
    print(f"Length of training/testing set with motion: {len(df_motion)}")    
    df_motion = filter_segment_inbed(df_motion)
    df_motion["PID"]=df_motion["SUBJECT"]
    if group:
        df_motion=add_groups(df_motion)
  
    return df_motion


def load_data(input_data_folder, motion_filter=True, max_seq_length=60, energy_th=5, remove_outbed=False, filter_type='motion', norm_scratch_length=True,):
    #input_data_files ='/mnt/data/Nocturnal-scratch/geneactive_20hz_2s'
    sf=int(re.findall(r"\d+hz", input_data_folder)[0].replace("hz",''))
    df = load_sequence_data(path=input_data_folder, remove_subject_filter=['US10013008'],include_subject_filter=None,list_features=True,motion_filter=motion_filter,max_seq_length=max_seq_length,sf=sf)
    df["angle"] = np.arctan(df["z"] / ((df["x"] ** 2 + df["y"] ** 2) ** 0.5)) * (180.0 / np.pi)
    #add signal energy
    df.loc[:,'row_energy'] = df.x**2 + df.y**2 + df.z**2
    df.loc[:,'energy']=df.groupby('segment')['row_energy'].transform('sum')
    
    if remove_outbed:
        df=df[df.bed==0]
        #Filter only observations in bed. We can test removing the whole motion segments that include partial data out of bed
        #     s_out=df.groupby('segment')['bed'].any().reset_index()
        #     s_out=s_out[s_out.bed==True]['segment']
        #     df=df[~df.segment.isin(s_out)]
        print("Load data: df.shape after removing not INBED: ", df.shape)
    
    #add scratch duration
    df['segment_scratch']=df.groupby('segment')['scratch'].transform('max')
    df['scratch_duration']=df.groupby('segment')['scratch'].transform('sum')
    df['scratch_count']=df.groupby('segment')['scratch'].transform('count')
    # df['scratch_duration']=df['scratch_duration']/df['scratch_count']
    if norm_scratch_length:
        df['scratch_duration'] /= df['scratch_count']
    else:
        # norm to seconds
        df['scratch_duration'] /= 20

    df['ADT']=(df['TSOEND']+ pd.Timedelta(hours=10, minutes=0, seconds=0)).dt.date.astype(str)
    #add positions
    df['position_segment']=df.groupby('segment')['timestamp'].rank(method='first')
    df['position_segmentr']=df.groupby('segment')['timestamp'].rank(method='first')/df.scratch_count
    # df['position_TSO']=(df.timestamp-df.TSOSTART).dt.seconds
    # df['position_TSOr']=(df.timestamp-df.TSOSTART)/(df.TSOEND-df.TSOSTART)
    
    # df['predictTSOSTART']=df[df.predictTSO].groupby(['PID','wrist','DAY'])['timestamp'].transform('min')
    # df['predictTSOEND']=df[df.predictTSO].groupby(['PID','wrist','DAY'])['timestamp'].transform('max')
    df=df[(df.SUBJECT!='US10008015')| (df.ADT!="2023-03-15")] # No Offset was found for this subject/date. Excluding it from the dataset.
    df=df[(df.SUBJECT!='US10008007')| (df.ADT!="2022-12-24")] #Subject removed the edvices during this night
    
    df['skinimpact']=np.where((df.SkinImpactLabeler1=='Yes')&(df.SkinImpactLabeler2=='Yes'),1,0)
    df['skinimpact_u']=np.where((df.SkinImpactLabeler1=='Yes')|(df.SkinImpactLabeler2=='Yes'),1,0)

    #Filter the segments having less than the energy threshold
    if filter_type=='motion':
        if energy_th is not None:
            print("Load data: df.shape after removing low energy segment: ", df.shape)
            df=df[df.scratch_count>energy_th*sf]
    elif filter_type=='non-motion':
        if energy_th is not None:
            print("Load data: df.shape of low energy segments only: ", df.shape)
            df=df[df.scratch_count<=energy_th*sf]
    else:
        pass
    
    print("---------------------------")
    df_segment=df.groupby('segment').max(1).reset_index()
    print(f"Segment prevalence: {df_segment[(df_segment.segment_scratch==True)].shape[0]/df_segment.shape[0]:.4f}, count: {df_segment.shape[0]}, average duration: {df_segment.groupby('segment')['scratch_count'].max(1).mean()/sf} s")
    return df



def load_data_pretrain(input_data_folder,include_subject_filter=None,remove_subject_filter=None,motion_filter=True,segment_th=5,remove_outbed=False,filter_type='motion',group=True):
    #input_data_files ='/mnt/data/Nocturnal-scratch/geneactive_20hz_2s'
    sf=int(re.findall(r"\d+hz", input_data_folder)[0].replace("hz",''))

    start_time = datetime.now()
    print("---------------> Starting load_test_data() function:  ", start_time)
    files = sorted(glob.glob(os.path.join(input_data_folder, f"*.parquet.gzip")))
    if remove_subject_filter is not None:
        files = [file for file in files if all(PID not in file for PID in remove_subject_filter)]
    
    if include_subject_filter is not None:
        files = [file for file in files if include_subject_filter in file]
    df_motion_list = []
    print(f"Total Files: {len(files)}, path={input_data_folder}")
    for counter, file in enumerate(files):#files[:50]
        current_subject = re.findall(r"US\d+", file)[0]
        day=os.path.basename(file).rstrip(".parquet.gzip").split('_')[-1]
        if counter % 50 == 0: 
            print(f"---->Processing file {counter} out of {len(files)} at {datetime.now()}") 
        df_temp = pd.read_parquet(file)
        df_temp=df_temp[(df_temp.predictTSO==True)]
        if df_temp.empty:
            custom_print(f"Warning: Data file empty. \nFile: {file}")
            continue
            
        df_temp['wrist']=np.where(df_temp.wrist.str.lower()=="left",1,0)
        df_temp['SUBJECT'] = current_subject
        df_temp['DAY'] = day
        
        if motion_filter:
            df_temp= df_temp[df_temp['stationary']==False]
        
        if  not df_temp.empty:
            df_motion_list.append(df_temp[[col for col in df_temp.columns if col not in ['index','diff','diff_flag','diff_batch','change']]])
    # -------------------------
    # Clean & Transform Dataset
    # -------------------------
    # no wear & predictTSO
    del df_temp
    print(f"---------------> Starting dataframes concatenation..... Time spent so far, from start:   {datetime.now() - start_time}")
    df_motion = pd.concat(df_motion_list, ignore_index=True)
    del df_motion_list
    df_motion['segment'] = (df_motion['SUBJECT']+'_'+df_motion['wrist'].astype(str)+'_'+df_motion['DAY']+'_'+df_motion['segment'].astype(str)).values 
        
    print(f"Concatenated Dataframe Shape: {df_motion.shape}, timestamp: {datetime.now()}")
    
    print(f"Length of training/testing set with motion: {len(df_motion)}")    
    df_motion["PID"]=df_motion["SUBJECT"]
    if group:
        df_motion=add_groups(df_motion)
    #Filter the segments having less than the energy threshold
    if filter_type=='motion':
        if segment_th is not None:
            df_motion=df_motion[df_motion.segment_duration>segment_th*sf]
            print(f"Load data: shape after removing segments less than threshold: {df_motion.shape}")
    elif filter_type=='non-motion':
        if segment_th is not None:
            df_motion=df_motion[df_motion.segment_duration<=segment_th*sf]
            print(f"Load data: shape of short segments only:{df_motion.shape} ")
    else:
        pass
    
    print("---------------------------")
    df_segment=df_motion.groupby('segment').max(1).reset_index()
    print(f"count: {df_segment.shape[0]}, average duration: {df_segment['segment_duration'].mean()/sf} s")
    
    return df_motion



def load_data_tso_patch(input_data_folder, include_subject_filter=None, remove_subject_filter=None, max_seq_length=86400, scaler=None):
    """
    Load raw sensor data for TSO prediction with patched input.

    Unlike load_data_tso, this function does NOT aggregate to minute-level.
    Returns raw 20Hz sensor data that will be patched in add_padding_tso_patch.

    Args:
        input_data_folder: Path to folder containing parquet.gzip files
        include_subject_filter: Subject ID to include (optional)
        remove_subject_filter: List of subject IDs to exclude (optional)
        max_seq_length: Maximum sequence length in seconds (default: 86400 = 24 hours)
        scaler: Optional StandardScaler object to apply to x, y, z columns during caching.
               If provided, data will be scaled once during cache creation for efficiency.
               If None, raw unscaled data will be cached.

    Returns:
        DataFrame with raw 20Hz sensor data for each 24-hour segment
        Columns: SUBJECT, DAY, wrist, timestamp, x, y, z, temperature,
                 non-wear, predictTSO, segment, PID, FOLD (if group=True)

    Example:
        # Load with scaling (recommended - cached for efficiency)
        scaler = joblib.load('/path/to/scaler.joblib')
        df = load_data_tso_patch(folder, scaler=scaler)

        # Load without scaling (raw data)
        df = load_data_tso_patch(folder, scaler=None)
    """
    # Create cache folder
    cache_folder = os.path.join(input_data_folder.rstrip("/raw"), "processed_patch/")
    os.makedirs(cache_folder, exist_ok=True)

    # Create cache filename
    final_cache_filename = f"tso_patch_raw_maxseq{max_seq_length}"
    if include_subject_filter:
        final_cache_filename += f"_include{include_subject_filter}"
    if remove_subject_filter:
        remove_str = "_".join(remove_subject_filter) if isinstance(remove_subject_filter, list) else str(remove_subject_filter)
        final_cache_filename += f"_remove{remove_str}"
    final_cache_filename += ".parquet.gzip"
    final_cache_path = os.path.join(cache_folder, final_cache_filename)

    # Check if metadata cache exists (new lazy loading approach)
    metadata_cache = final_cache_path.replace('.parquet.gzip', '_metadata.parquet.gzip')
    if os.path.exists(metadata_cache):
        print(f"===============> Loading from METADATA cache: {metadata_cache}")
        start_time = datetime.now()
        df_metadata = pd.read_parquet(metadata_cache)

        print(f"✓ Metadata cache loaded in {datetime.now() - start_time}")
        print(f"✓ Total segments: {len(df_metadata)}")
        print(f"✓ Metadata memory: {df_metadata.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"✓ Data will be loaded lazily during training (memory-efficient mode)")
        return df_metadata

    # Check if old final cache exists (legacy mode - loads all data)
    if os.path.exists(final_cache_path):
        print(f"===============> Loading from FINAL cache (legacy mode): {final_cache_path}")
        print(f"WARNING: This loads all data into memory. Consider regenerating cache for lazy loading.")
        start_time = datetime.now()
        df_final = pd.read_parquet(final_cache_path)

        print(f"✓ Final cache loaded in {datetime.now() - start_time}")
        print(f"✓ Final raw data shape: {df_final.shape}")
        print(f"✓ Number of unique segments: {df_final['segment'].nunique()}")
        print(f"✓ Memory usage: ~{df_final.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
        return df_final

    # Check if cached segment files already exist
    print(f"===============> No metadata cache found. Checking for cached segment files...")
    cached_segment_files = sorted(glob.glob(os.path.join(cache_folder, "tso_patch_*.parquet.gzip")))

    start_time = datetime.now()
    if len(cached_segment_files) > 0:
        print(f"✓ Found {len(cached_segment_files)} cached segment files. Building metadata from cached files...")
        processed_files = cached_segment_files
    else:
        # Load raw data files
        print(f"===============> No cached files found. Processing raw files...")
        files = sorted(glob.glob(os.path.join(input_data_folder, "*.parquet.gzip")))

        # Apply filters
        if remove_subject_filter is not None:
            files = [f for f in files if all(PID not in f for PID in remove_subject_filter)]
        if include_subject_filter is not None:
            files = [f for f in files if include_subject_filter in f]

        print(f"Total files to process: {len(files)}")

        processed_files = []
        samples_per_second = 20  # 20Hz sampling rate
        max_samples = max_seq_length * samples_per_second

        for counter, file in enumerate(files):
            # skip if cache file exists
            current_subject = re.findall(r"US\d+", file)[0]
            # day = os.path.basename(file).rstrip(".parquet.gzip").split('_')[-1]

            basename = os.path.basename(file).rstrip(".parquet.gzip")

            # Extract wrist from filename (e.g., Processed_US10001001_left_wrist_2022-04-14.parquet.gzip)
            if 'left' in basename.lower():
                wrist_str = 'left'
            elif 'right' in basename.lower():
                wrist_str = 'right'
            else:
                wrist_str = 'unknown'

            day = basename.split('_')[-1]
            if counter % 50 == 0:
                print(f"  Processing file {counter}/{len(files)}: {current_subject}_{day}")

            cache_path = os.path.join(cache_folder, f"tso_patch_{current_subject}_{wrist_str}_{day}.parquet.gzip")

            if os.path.exists(cache_path):
                processed_files.append(cache_path)
                if counter % 50 == 0:
                    print(f"  Cache file exists for {current_subject}_{day}. Using cached version...")
                continue

            # Read file
            df_temp = pd.read_parquet(file)

            # Add metadata
            df_temp['wrist'] = np.where(df_temp.wrist.str.lower() == "left", 1, 0)
            df_temp['SUBJECT'] = current_subject
            df_temp['DAY'] = day
            df_temp['PID'] = current_subject
            wrist_value = df_temp['wrist'].iloc[0]
            df_temp['segment'] = current_subject + '_' + str(wrist_value) + '_' + day

            # Filter to max sequence length (keep first 24 hours)
            if len(df_temp) > max_samples:
                df_temp = df_temp.iloc[:max_samples].copy()

            # Keep only necessary columns for patched input
            required_cols = ['timestamp', 'x', 'y', 'z',
                            'temperature', 'segment']

            # Apply scaler if provided (scale once during cache creation)
            if scaler is not None:
                columns_to_scale = ['x', 'y', 'z']
                df_temp[columns_to_scale] = scaler.transform(df_temp[columns_to_scale])

            # Add labels if available
            if 'non-wear' in df_temp.columns:
                required_cols.append('non-wear')
            if 'predictTSO' in df_temp.columns:
                required_cols.append('predictTSO')

            # Select columns
            df_temp = df_temp[required_cols].copy()

            # # Optimize dtypes to reduce memory usage (float64 -> float32 saves 50% memory)
            # # Accelerometer data doesn't need float64 precision
            # for col in ['x', 'y', 'z', 'temperature']:
            #     if col in df_temp.columns:
            #         df_temp[col] = df_temp[col].astype('float32')

            # # Boolean/categorical columns can use smaller types
            # if 'non-wear' in df_temp.columns:
            #     df_temp['non-wear'] = df_temp['non-wear'].astype('int8')
            # if 'predictTSO' in df_temp.columns:
            #     df_temp['predictTSO'] = df_temp['predictTSO'].astype('int8')

            # Cache this file
            df_temp.to_parquet(cache_path, compression='gzip')
            processed_files.append(cache_path)

            del df_temp

    # Instead of concatenating into one large DataFrame (causes OOM),
    # create metadata-only DataFrame for lazy loading
    print(f"\n===============> Building metadata index from {len(processed_files)} cached segments...")

    metadata_list = []
    total_predictTSO = 0
    total_nonwear = 0
    total_samples = 0

    for i, cache_path in enumerate(processed_files):
        if i % 100 == 0:
            print(f"    Processing metadata {i}/{len(processed_files)}")

        # Extract segment info from filename
        basename = os.path.basename(cache_path)
        parts = basename.replace('tso_patch_', '').replace('.parquet.gzip', '').split('_')

        if len(parts) >= 3:
            subject = parts[0]
            wrist_str = parts[1]
            day = parts[2]
        else:
            subject = parts[0] if len(parts) > 0 else 'unknown'
            wrist_str = 'unknown'
            day = parts[1] if len(parts) > 1 else 'unknown'

        wrist = 1 if wrist_str.lower() == 'left' else 0

        # Read just the first row to get segment name and count rows
        df_sample = pd.read_parquet(cache_path)
        segment_name = df_sample['segment'].iloc[0] if 'segment' in df_sample.columns else f"{subject}_{wrist_str}_{day}"
        num_samples = len(df_sample)

        # Count labels
        if 'predictTSO' in df_sample.columns:
            total_predictTSO += df_sample['predictTSO'].sum()
        if 'non-wear' in df_sample.columns:
            total_nonwear += df_sample['non-wear'].sum()
        total_samples += num_samples

        # Store metadata
        metadata_list.append({
            'segment': segment_name,
            'cache_file': cache_path,
            'SUBJECT': subject,
            'DAY': day,
            'wrist': wrist,
            'PID': subject,
            'num_samples': num_samples
        })

        del df_sample

    # Create metadata DataFrame (very small - just paths and segment info)
    df_metadata = pd.DataFrame(metadata_list)

    print(f"\n✓ Metadata index created:")
    print(f"  Total segments: {len(df_metadata)}")
    print(f"  Total samples: {total_samples:,}")
    if total_predictTSO > 0:
        print(f"  predictTSO samples: {total_predictTSO:,}/{total_samples:,} ({100*total_predictTSO/total_samples:.1f}%)")
    if total_nonwear > 0:
        print(f"  non-wear samples: {total_nonwear:,}/{total_samples:,} ({100*total_nonwear/total_samples:.1f}%)")
    print(f"  Metadata memory: {df_metadata.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Save metadata cache
    metadata_cache = final_cache_path.replace('.parquet.gzip', '_metadata.parquet.gzip')
    df_metadata.to_parquet(metadata_cache, compression='gzip')
    print(f"✓ Metadata saved to: {metadata_cache}")

    print(f"Total processing time: {datetime.now() - start_time}")

    return df_metadata



def load_data_tso_patch_biobank(input_data_folder, include_subject_filter=None, remove_subject_filter=None, max_seq_length=86400):
    """
    Load raw sensor data for TSO prediction with patched input from UK Biobank format.

    This function handles biobank file naming convention:
    - Original JNJ format: Processed_US10008015_right_2023-03-09.parquet.gzip
    - Biobank format: Processed_1012030_1012030_90001_0_0_2015-08-31.parquet.gzip

    Key differences from load_data_tso_patch:
    - Subject ID is numeric (e.g., 1012030) instead of US\d+ pattern
    - No wrist information in filename (no 'left'/'right')
    - Filename structure: Processed_{eid}_{eid}_{instance}_{array_idx}_{field}_{date}.parquet.gzip

    Args:
        input_data_folder: Path to folder containing parquet.gzip files
        include_subject_filter: Subject ID to include (optional, string or int)
        remove_subject_filter: List of subject IDs to exclude (optional)
        max_seq_length: Maximum sequence length in seconds (default: 86400 = 24 hours)

    Returns:
        DataFrame with metadata for lazy loading of raw 20Hz sensor data
        Columns: segment, cache_file, SUBJECT, DAY, wrist, PID, num_samples
    """
    # Create cache folder
    cache_folder = os.path.join(input_data_folder.rstrip("/raw"), "processed_patch_biobank/")
    os.makedirs(cache_folder, exist_ok=True)

    # Create cache filename
    final_cache_filename = f"tso_patch_biobank_raw_maxseq{max_seq_length}"
    if include_subject_filter:
        final_cache_filename += f"_include{include_subject_filter}"
    if remove_subject_filter:
        remove_str = "_".join(str(s) for s in remove_subject_filter) if isinstance(remove_subject_filter, list) else str(remove_subject_filter)
        final_cache_filename += f"_remove{remove_str}"
    final_cache_filename += ".parquet.gzip"
    final_cache_path = os.path.join(cache_folder, final_cache_filename)

    # Check if metadata cache exists (lazy loading approach)
    metadata_cache = final_cache_path.replace('.parquet.gzip', '_metadata.parquet.gzip')
    if os.path.exists(metadata_cache):
        print(f"===============> Loading from METADATA cache: {metadata_cache}")
        start_time = datetime.now()
        df_metadata = pd.read_parquet(metadata_cache)

        print(f"Metadata cache loaded in {datetime.now() - start_time}")
        print(f"Total segments: {len(df_metadata)}")
        print(f"Metadata memory: {df_metadata.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"Data will be loaded lazily during training (memory-efficient mode)")
        return df_metadata

    # Check if old final cache exists (legacy mode)
    if os.path.exists(final_cache_path):
        print(f"===============> Loading from FINAL cache (legacy mode): {final_cache_path}")
        print(f"WARNING: This loads all data into memory. Consider regenerating cache for lazy loading.")
        start_time = datetime.now()
        df_final = pd.read_parquet(final_cache_path)

        print(f"Final cache loaded in {datetime.now() - start_time}")
        print(f"Final raw data shape: {df_final.shape}")
        print(f"Number of unique segments: {df_final['segment'].nunique()}")
        print(f"Memory usage: ~{df_final.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
        return df_final

    # Check if cached segment files already exist
    print(f"===============> No metadata cache found. Checking for cached segment files...")
    cached_segment_files = sorted(glob.glob(os.path.join(cache_folder, "tso_patch_biobank_*.parquet.gzip")))

    start_time = datetime.now()
    if len(cached_segment_files) > 0:
        print(f"Found {len(cached_segment_files)} cached segment files. Building metadata from cached files...")
        processed_files = cached_segment_files
    else:
        # Load raw data files
        print(f"===============> No cached files found. Processing raw files...")
        files = sorted(glob.glob(os.path.join(input_data_folder, "*.parquet.gzip")))

        # Apply filters - convert filters to strings for comparison
        if remove_subject_filter is not None:
            remove_list = [str(s) for s in remove_subject_filter] if isinstance(remove_subject_filter, list) else [str(remove_subject_filter)]
            files = [f for f in files if all(pid not in os.path.basename(f) for pid in remove_list)]
        if include_subject_filter is not None:
            include_str = str(include_subject_filter)
            files = [f for f in files if include_str in os.path.basename(f)]

        print(f"Total files to process: {len(files)}")

        processed_files = []
        samples_per_second = 20  # 20Hz sampling rate
        max_samples = max_seq_length * samples_per_second

        for counter, file in enumerate(files):
            basename = os.path.basename(file).replace('.parquet.gzip', '')

            # Parse biobank filename: Processed_{eid}_{eid}_{instance}_{array_idx}_{field}_{date}
            # Example: Processed_1012030_1012030_90001_0_0_2015-08-31
            parts = basename.split('_')

            if len(parts) >= 7 and parts[0] == 'Processed':
                # Biobank format: subject is the second element (eid)
                current_subject = parts[1]
                # Date is the last element (YYYY-MM-DD format)
                day = parts[-1]
                # Create a unique identifier from the middle parts
                instance_info = '_'.join(parts[2:-1])
            else:
                # Fallback: try to extract what we can
                current_subject = parts[1] if len(parts) > 1 else 'unknown'
                day = parts[-1] if len(parts) > 0 else 'unknown'
                instance_info = '_'.join(parts[2:-1]) if len(parts) > 3 else ''

            if counter % 50 == 0:
                print(f"  Processing file {counter}/{len(files)}: {current_subject}_{day}")

            # Cache path includes instance info to handle multiple recordings per subject per day
            cache_path = os.path.join(cache_folder, f"tso_patch_biobank_{current_subject}_{instance_info}_{day}.parquet.gzip")

            if os.path.exists(cache_path):
                processed_files.append(cache_path)
                if counter % 50 == 0:
                    print(f"  Cache file exists for {current_subject}_{day}. Using cached version...")
                continue

            # Read file
            df_temp = pd.read_parquet(file)

            # Add metadata - no wrist info in biobank, default to 0 (unknown/right)
            # Check if wrist column exists in data
            if 'wrist' in df_temp.columns:
                df_temp['wrist'] = np.where(df_temp['wrist'].astype(str).str.lower() == "left", 1, 0)
            else:
                df_temp['wrist'] = 0  # Default: unknown wrist

            df_temp['SUBJECT'] = current_subject
            df_temp['DAY'] = day
            df_temp['PID'] = current_subject

            # Create unique segment identifier
            wrist_value = df_temp['wrist'].iloc[0] if 'wrist' in df_temp.columns else 0
            df_temp['segment'] = f"{current_subject}_{wrist_value}_{instance_info}_{day}"

            # Filter to max sequence length (keep first 24 hours)
            if len(df_temp) > max_samples:
                df_temp = df_temp.iloc[:max_samples].copy()

            # Keep only necessary columns for patched input
            required_cols = ['timestamp', 'x', 'y', 'z', 'segment']

            # Add optional columns if available
            if 'temperature' in df_temp.columns:
                required_cols.append('temperature')
            if 'non-wear' in df_temp.columns:
                required_cols.append('non-wear')
            if 'predictTSO' in df_temp.columns:
                required_cols.append('predictTSO')

            # Select only columns that exist
            available_cols = [col for col in required_cols if col in df_temp.columns]
            df_temp = df_temp[available_cols].copy()

            # Cache this file
            df_temp.to_parquet(cache_path, compression='gzip')
            processed_files.append(cache_path)

            del df_temp

    # Build metadata index for lazy loading
    print(f"\n===============> Building metadata index from {len(processed_files)} cached segments...")

    metadata_list = []
    total_predictTSO = 0
    total_nonwear = 0
    total_samples = 0

    for i, cache_path in enumerate(processed_files):
        if i % 100 == 0:
            print(f"    Processing metadata {i}/{len(processed_files)}")

        # Extract segment info from filename
        # Format: tso_patch_biobank_{subject}_{instance_info}_{date}.parquet.gzip
        basename = os.path.basename(cache_path)
        parts = basename.replace('tso_patch_biobank_', '').replace('.parquet.gzip', '').split('_')

        # Subject is first part, date is last part (YYYY-MM-DD)
        subject = parts[0] if len(parts) > 0 else 'unknown'
        day = parts[-1] if len(parts) > 0 else 'unknown'

        # Read the cached file to get accurate segment name and stats
        df_sample = pd.read_parquet(cache_path)
        segment_name = df_sample['segment'].iloc[0] if 'segment' in df_sample.columns else f"{subject}_{day}"
        num_samples = len(df_sample)

        # Determine wrist value (default 0 for biobank)
        wrist = 0

        # Count labels
        if 'predictTSO' in df_sample.columns:
            total_predictTSO += df_sample['predictTSO'].sum()
        if 'non-wear' in df_sample.columns:
            total_nonwear += df_sample['non-wear'].sum()
        total_samples += num_samples

        # Store metadata
        metadata_list.append({
            'segment': segment_name,
            'cache_file': cache_path,
            'SUBJECT': subject,
            'DAY': day,
            'wrist': wrist,
            'PID': subject,
            'num_samples': num_samples
        })

        del df_sample

    # Create metadata DataFrame
    df_metadata = pd.DataFrame(metadata_list)

    print(f"\nMetadata index created:")
    print(f"  Total segments: {len(df_metadata)}")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Unique subjects: {df_metadata['SUBJECT'].nunique()}")
    if total_predictTSO > 0:
        print(f"  predictTSO samples: {total_predictTSO:,}/{total_samples:,} ({100*total_predictTSO/total_samples:.1f}%)")
    if total_nonwear > 0:
        print(f"  non-wear samples: {total_nonwear:,}/{total_samples:,} ({100*total_nonwear/total_samples:.1f}%)")
    print(f"  Metadata memory: {df_metadata.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Save metadata cache
    df_metadata.to_parquet(metadata_cache, compression='gzip')
    print(f"Metadata saved to: {metadata_cache}")

    print(f"Total processing time: {datetime.now() - start_time}")

    return df_metadata



def load_data_tso(input_data_folder, include_subject_filter=None, remove_subject_filter=None, max_seq_length=86400, group=True):
    """
    Load data for TSO prediction with 24-hour windows and compute minute-level statistical features.

    For each 24h window, aggregate raw 20Hz data into minute-level features.
    Raw data: (24h × 3600s × 20Hz) = 1,728,000 samples
    Output: (24 × 60, 28) where 28 features = 4 base means + 27 statistical features

    Features per minute:
    - Base aggregated: x, y, z, temperature (4 features - means over 60s)
    - Statistical features for x, y, z: mean, std, min, max, q25, q75, skew, kurt, cv (9 × 3 = 27 features)
    - Temperature: only mean is used (already in base aggregated)

    Args:
        input_data_folder: Path to folder containing parquet.gzip files
        include_subject_filter: Subject ID to include (optional)
        remove_subject_filter: List of subject IDs to exclude (optional)
        max_seq_length: Maximum sequence length in seconds (default: 86400 = 24 hours)
        group: Whether to add groups/folds (default: True)

    Returns:
        DataFrame with minute-level features for each 24-hour segment
        Each row contains: segment identifier, minute_id (0-1439), and 28 statistical features
    """
    # Create cache folder
    cache_folder = os.path.join(input_data_folder.rstrip("/raw/"), "processed/")
    os.makedirs(cache_folder, exist_ok=True)

    # Create final cache filename based on parameters
    final_cache_filename = f"tso_final_maxseq{max_seq_length}_group{group}"
    if include_subject_filter:
        final_cache_filename += f"_include{include_subject_filter}"
    if remove_subject_filter:
        remove_str = "_".join(remove_subject_filter) if isinstance(remove_subject_filter, list) else str(remove_subject_filter)
        final_cache_filename += f"_remove{remove_str}"
    final_cache_filename += ".parquet.gzip"
    final_cache_path = os.path.join(cache_folder, final_cache_filename)

    # # Check if final processed cache exists
    if os.path.exists(final_cache_path):
        
        # os.remove(final_cache_path)
        print(f"===============> Loading from FINAL cache: {final_cache_path}")
        start_time = datetime.now()
        df_final = pd.read_parquet(final_cache_path)
        print(f"Final cache loaded in {datetime.now() - start_time}")
        print(f"Final feature dataframe shape: {df_final.shape}")
        print(f"Number of unique segments: {df_final['segment'].nunique()}")
        print(f"Number of minute-level records: {len(df_final)}")
        return df_final

    # Final cache not found - need to process data
    print(f"===============> Final cache not found. Processing data...")

    # Extract sampling frequency
    sf = int(re.findall(r"\d+hz", input_data_folder)[0].replace("hz", ''))
    window_samples = max_seq_length * sf  # 24h * 3600s * 20Hz = 1,728,000 samples
    samples_per_minute = 60 * sf  # 60s * 20Hz = 1200 samples per minute

    start_time = datetime.now()
    print(f"---------------> Starting load_data_tso() function: {start_time}")
    print(f"Window size: {max_seq_length}s ({max_seq_length/3600:.1f} hours) = {window_samples:,} samples at {sf}Hz")
    print(f"Minute-level aggregation: {samples_per_minute} samples per minute -> {max_seq_length//60} minutes per window")

    # Load files
    files = sorted(glob.glob(os.path.join(input_data_folder, f"*.parquet.gzip")))
    if remove_subject_filter is not None:
        files = [file for file in files if all(PID not in file for PID in remove_subject_filter)]

    if include_subject_filter is not None:
        files = [file for file in files if include_subject_filter in file]

    processed_files = []
    print(f"Total Files: {len(files)}, path={input_data_folder}")

    # Process each file individually to save memory
    for counter, file in enumerate(files):
        current_subject = re.findall(r"US\d+", file)[0]
        basename = os.path.basename(file).rstrip(".parquet.gzip")

        # Extract wrist from filename (e.g., Processed_US10001001_left_wrist_2022-04-14.parquet.gzip)
        if 'left' in basename.lower():
            wrist_str = 'left'
        elif 'right' in basename.lower():
            wrist_str = 'right'
        else:
            wrist_str = 'unknown'

        day = basename.split('_')[-1]

        if counter % 10 == 0:
            print(f"---->Processing file {counter} out of {len(files)} at {datetime.now()}")

        # Create cache filename for fully processed features (include wrist)
        cache_filename = f"tso_features_{current_subject}_{wrist_str}_{day}_maxseq{max_seq_length}.parquet.gzip"
        cache_path = os.path.join(cache_folder, cache_filename)

        # Check if cached version exists
        if os.path.exists(cache_path):
            if counter % 10 == 0:
                print(f"    -> Found in cache: {cache_filename}")
            processed_files.append(cache_path)
            continue

        # Process from raw data
        df_temp = pd.read_parquet(file)

        if df_temp.empty:
            print(f"Warning: Data file empty. File: {file}")
            continue

        # Add metadata
        df_temp['wrist'] = np.where(df_temp.wrist.str.lower() == "left", 1, 0)
        df_temp['SUBJECT'] = current_subject
        df_temp['DAY'] = day

        # Create 24-hour windows using row-based indexing
        # This splits files with >24h data into multiple segments
        df_temp['row_idx'] = df_temp.groupby(['SUBJECT', 'wrist', 'DAY']).cumcount()
        # df_temp['window_id'] = df_temp['row_idx'] // window_samples  # 24h window index (0, 1, 2, ...)
        df_temp['minute_id'] = (df_temp['row_idx'] % window_samples) // samples_per_minute  # Minute within window (0-1439)

        # Create segment ID that includes window_id to handle files with >24h data
        df_temp['segment'] = (df_temp['SUBJECT'] + '_' +
                             df_temp['wrist'].astype(str) + '_' +
                             df_temp['DAY']
                            #  df_temp['window_id'].astype(str)
                             )

        groupby_cols = ['segment', 'minute_id', 'SUBJECT', 'wrist', 'DAY']

        # Build comprehensive aggregation dictionary for single-pass aggregation
        # Using consistent format: key is output column name, value is (source_column, aggfunc) tuple
        agg_dict = {
            'temperature': ('temperature', 'mean')
        }

        # Preserve original timestamp for time embedding extraction
        if 'timestamp' in df_temp.columns:
            agg_dict['timestamp'] = ('timestamp', 'first')

        # Add metadata aggregations
        if 'ENMO' in df_temp.columns:
            agg_dict['ENMO'] = ('ENMO', 'mean')
        if 'non-wear' in df_temp.columns:
            agg_dict['non-wear'] = ('non-wear', lambda x: (x == 1).any() * 1)
        if 'predictTSO' in df_temp.columns:
            agg_dict['predictTSO'] = ('predictTSO', lambda x: (x == True).any())

        # Add statistical features for x, y, z channels
        for channel in ['x', 'y', 'z']:
            if channel in df_temp.columns:
                agg_dict[f'{channel}_mean'] = (channel, 'mean')
                agg_dict[f'{channel}_std'] = (channel, 'std')
                agg_dict[f'{channel}_min'] = (channel, 'min')
                agg_dict[f'{channel}_max'] = (channel, 'max')
                agg_dict[f'{channel}_q25'] = (channel, lambda x: x.quantile(0.25))
                agg_dict[f'{channel}_q75'] = (channel, lambda x: x.quantile(0.75))
                agg_dict[f'{channel}_skew'] = (channel, lambda x: x.skew())
                agg_dict[f'{channel}_kurt'] = (channel, lambda x: x.kurt())
                agg_dict[f'{channel}_cv'] = (channel, lambda x: x.std() / (x.mean() + 1e-8))

        # Perform all aggregations in a single pass
        df_temp = df_temp.groupby(groupby_cols).agg(**agg_dict).reset_index()

        
        # check the minute length per segment: if len of df_temp is equal to 1440
        actual_minutes = df_temp['minute_id'].nunique()

        wrist = df_temp['wrist'].iloc[0]
        
        if actual_minutes != 1440:
            print(f"Warning: Segment {current_subject}_{wrist}_{day} does not have 1440 minutes, actual: {actual_minutes}")

        # Add metadata (segment already exists from before aggregation)
        df_temp['PID'] = current_subject

        # Extract SUBJECT, wrist, DAY from segment for downstream use
        # segment_parts = df_temp['segment'].str.split('_', expand=True)
        # df_temp['SUBJECT'] = segment_parts[0]
        # df_temp['wrist'] = segment_parts[1].astype(int)
        # df_temp['DAY'] = segment_parts[2]

        # Extract unified time embedding from timestamp if it exists
        if 'timestamp' in df_temp.columns:
            hour = df_temp['timestamp'].dt.hour
            minute = df_temp['timestamp'].dt.minute
            # day_of_week = df_temp['timestamp'].dt.dayofweek

            # Create single composite time embedding combining all temporal information
            # Range: 0-1 representing position in week
            # Formula: (day_of_week * 24 + hour + minute/60) / (7 * 24)
            df_temp['time_embedding'] = (hour + minute / 60) / 24

            # Add cyclic encoding (captures day/night cycle smoothly)
            # This wraps around smoothly from Sunday night to Monday morning
            df_temp['time_cyclic'] = np.sin(2 * np.pi * df_temp['time_embedding'])

            # Add binary nighttime indicator (critical for scratch detection)
            # Nighttime: 10 PM (22:00) to 7 AM (07:00)
            # df_temp['is_night'] = ((hour >= 22) | (hour < 7)).astype(int)

        # Add position (rank within segment based on minute_id, 1-indexed)
        df_temp = df_temp.sort_values('minute_id')
        df_temp['position'] = df_temp.groupby('segment').cumcount() + 1

        # Save processed file
        df_temp.to_parquet(cache_path, compression='gzip')
        processed_files.append(cache_path)

        # Clean up to free memory
        del df_temp

    # Load all processed files and concatenate
    print(f"---------------> Loading and concatenating {len(processed_files)} processed files...")
    df_chunks = []
    for i, cache_path in enumerate(processed_files):
        if i % 50 == 0:
            print(f"    Loading file {i}/{len(processed_files)}")
        df_chunks.append(pd.read_parquet(cache_path))

    df_motion = pd.concat(df_chunks, ignore_index=True)
    del df_chunks

    print(f"Concatenated Dataframe Shape: {df_motion.shape}, timestamp: {datetime.now()}")

    # Load all processed files and concatenate
    print(f"\nLoading {len(processed_files)} processed files...")
    df_list = []
    for cache_file in processed_files:
        df_temp = pd.read_parquet(cache_file)
        df_list.append(df_temp)
        del df_temp

    df_final = pd.concat(df_list, ignore_index=True)
    del df_list  # Free memory

    # Add PID if not present
    if "PID" not in df_final.columns:
        df_final["PID"] = df_final["SUBJECT"]

    # # Add group information if requested
    # if group:
    #     # Extract unique segments for grouping
    #     unique_segments = df_final[['segment', 'SUBJECT', 'wrist', 'DAY', 'PID']].drop_duplicates('segment')
    #     # Create a temporary dataframe with segment as index for adding groups
    #     temp_df = unique_segments.set_index('segment')
    #     temp_df = add_groups(temp_df.reset_index())
    #     # Merge FOLD information back to df_final
    #     df_final = df_final.merge(temp_df[['segment', 'FOLD']], on='segment', how='left')

    # Ensure labels are properly formatted
    if 'predictTSO_<lambda>' in df_final.columns:
        df_final['predictTSO'] = df_final['predictTSO_<lambda>'].astype(bool)
        df_final = df_final.drop('predictTSO_<lambda>', axis=1)

    if 'non-wear_<lambda>' in df_final.columns:
        df_final['non-wear'] = df_final['non-wear_<lambda>'].astype(int)
        df_final = df_final.drop('non-wear_<lambda>', axis=1)

    print(f"\nFinal feature dataframe shape: {df_final.shape}")
    print(f"Number of unique segments: {df_final['segment'].nunique()}")
    print(f"Number of minute-level records: {len(df_final)}")


    # Print label statistics
    if 'predictTSO' in df_final.columns:
        print(f"  predictTSO minutes: {df_final['predictTSO'].sum()}/{len(df_final)} ({100*df_final['predictTSO'].sum()/len(df_final):.1f}%)")

    if 'non-wear' in df_final.columns:
        print(f"  non-wear minutes: {df_final['non-wear'].sum()}/{len(df_final)} ({100*df_final['non-wear'].sum()/len(df_final):.1f}%)")

    # Calculate 'other' minutes
    if 'predictTSO' in df_final.columns and 'non-wear' in df_final.columns:
        other_mask = (~df_final['predictTSO']) & (df_final['non-wear'] == 0)
        other_count = other_mask.sum()
        print(f"  other minutes: {other_count}/{len(df_final)} ({100*other_count/len(df_final):.1f}%)")

    print(f"Total processing time: {datetime.now() - start_time}")

    # Save final processed data to cache for future runs
    final_cache_filename = f"tso_final_maxseq{max_seq_length}_group{group}"
    if include_subject_filter:
        final_cache_filename += f"_include{include_subject_filter}"
    if remove_subject_filter:
        remove_str = "_".join(remove_subject_filter) if isinstance(remove_subject_filter, list) else str(remove_subject_filter)
        final_cache_filename += f"_remove{remove_str}"
    final_cache_filename += ".parquet.gzip"
    final_cache_path = os.path.join(cache_folder, final_cache_filename)

    print(f"Saving final processed data to cache: {final_cache_path}")
    df_final.to_parquet(final_cache_path, compression='gzip')
    print(f"Cache saved successfully")

    return df_final



def load_sequence_data_nsucl(path, remove_subject_filter,include_subject_filter, list_features, motion_filter,group=True,only_positive=False,max_seq_length=60,sf=20):
    start_time = datetime.now()
    print("---------------> Starting load_test_data() function:  ", start_time)
    files = sorted(glob.glob(os.path.join(path, f"*.parquet.gzip")))
    if remove_subject_filter is not None:
        files = [file for file in files if all(PID not in file for PID in remove_subject_filter)]
    
    if include_subject_filter is not None:
        files = [file for file in files if include_subject_filter in file]
    df_motion_list = []
    df_nomotion_list = []
    df=pd.DataFrame()
    print(f"Total Files: {len(files)}, path={path}")
    for counter, file in enumerate(files):#files[:50]
        current_subject = re.findall(r"Processed_S_\d+", file)[0].replace("Processed_","")
        if counter % 50 == 0: 
            print(f"---->Processing file {counter} out of {len(files)} at {datetime.now()}") 
        df_temp = pd.read_parquet(file)
        if df_temp.empty or len(df_temp.columns) < 2:
            custom_print(f"Warning: Data file empty. \nFile: {file}")
            continue
        
        df_temp['wrist']=df_temp.wrist.str.lower() 
        # df_temp['wrist']=np.where(df_temp.wrist.str.lower()=="left",1,0)
        df_temp['SUBJECT'] = current_subject
        
        if motion_filter:
            df_temp= df_temp[df_temp['stationary']==False]
        if only_positive:
            df_temp['segment'] = (df_temp['SUBJECT']+'_'+df_temp['wrist'].astype(str)+'_'+df_temp['segment'].astype(str)).values 
            #If maximum sequence length is set, then update the segment ID with a new one by adding an index for the mini segments.
            if max_seq_length is not None:
                length_min_segments=max_seq_length*sf
                df_temp = df_temp.groupby('segment', group_keys=False).apply(lambda x: assign_incremental_numbers(x, length_min_segments))
                df_temp['segment']=df_temp['segment'].astype(str) + "_" + df_temp['mini_segment'].astype(str)
            df_temp=df_temp.merge(df_temp.groupby('segment')['scratch'].any().reset_index().rename(columns={'scratch':'segment_scratch'}))
            df_temp=df_temp[df_temp.segment_scratch==True] # only focus on False negatives
        if  not df_temp.empty:
            #remove when cannot determine scratch is reported
#             df_temp=df_temp.loc[(df_temp.UsageLabeler1 != "CANNOT DETERMINE SCRATCH") & (df_temp.UsageLabeler2 != "CANNOT DETERMINE SCRATCH")]
#             #remove when cannot determine wrist is reported
#             df_temp=df_temp.loc[(df_temp.UsageLabeler1 != "CANNOT DETERMINE WRIST") & (df_temp.UsageLabeler2 != "CANNOT DETERMINE WRIST")]
            df_motion_list.append(df_temp)
    # -------------------------
    # Clean & Transform Dataset
    # -------------------------
    print(f"---------------> Starting dataframes concatenation..... Time spent so far, from start:   {datetime.now() - start_time}")
    df_motion = pd.concat(df_motion_list, ignore_index=True)
    if not only_positive:
        df_motion['segment'] = (df_motion['SUBJECT']+'_'+df_motion['wrist'].astype(str)+'_'+df_motion['segment'].astype(str)).values 
        
        #If maximum sequence length is set, then update the segment ID with a new one by adding an index for the mini segments.
        if max_seq_length is not None:
            length_min_segments=max_seq_length*sf
            df_motion = df_motion.groupby('segment', group_keys=False).apply(lambda x: assign_incremental_numbers(x, length_min_segments))
            df_motion['segment']=df_motion['segment'].astype(str) + "_" + df_motion['mini_segment'].astype(str)
        
        #add segment level scratch label
        df_motion=df_motion.merge(df_motion.groupby('segment')['scratch'].any().reset_index().rename(columns={'scratch':'segment_scratch'}))
        df_motion=df_motion.groupby('segment', group_keys=False).apply(lambda x: add_change_point(x))
    #df_motion = remove_missmatch_labels(df_motion)
    print(f"Concatenated Dataframe Shape: {df_motion.shape}, timestamp: {datetime.now()}")
    del df_temp

    df_motion["PID"]=df_motion["SUBJECT"]
    if group:
        df_motion=add_groups(df_motion,"nsucl")
  
    return df_motion


def load_data_nsucl(input_data_folder,motion_filter=True,max_seq_length=60):
    #input_data_files ='/mnt/data/Nocturnal-scratch/geneactive_20hz_2s'
    
    sf=int(re.findall(r"\d+hz", input_data_folder)[0].replace("hz",''))
    df = load_sequence_data_nsucl(path=input_data_folder, remove_subject_filter=['US10013008'],include_subject_filter=None,list_features=True,motion_filter=motion_filter,max_seq_length=max_seq_length,sf=sf)
    df["angle"] = np.arctan(df["z"] / ((df["x"] ** 2 + df["y"] ** 2) ** 0.5)) * (180.0 / np.pi)
    #add signal energy
    df.loc[:,'row_energy'] = df.x**2 + df.y**2 + df.z**2
    df.loc[:,'energy']=df.groupby('segment')['row_energy'].transform('sum')
    
    
    #add scratch duration
    scratch_duration=df.groupby('segment').agg({'scratch' :['sum','count']}).reset_index()
    scratch_duration.columns=scratch_duration.columns.map('_'.join).str.strip('_')
    scratch_duration['scratch_duration']=scratch_duration.scratch_sum/ scratch_duration.scratch_count #(max_seq_length*20)
    df=df.merge(scratch_duration)

    
    #add positions
    df['position_segment']=df.groupby('segment')['timestamp'].rank(method='first')
    df['position_segmentr']=df.groupby('segment')['timestamp'].rank(method='first')/df.scratch_count
    
    
    print("Load data: df.shape after removing not INBED: ", df.shape)
    #Filter the segments having less than 5 in energy
    df=df[df.energy>3]
    print("Load data: df.shape after removing low energy segment: ", df.shape)
    print("---------------------------")
    # df_segment=df.groupby('segment').max(1).reset_index()
    df_segment=df.drop_duplicates(subset='segment') 
    print(f"Segment prevalence: {df_segment[(df_segment.segment_scratch==True)].shape[0]/df_segment.shape[0]:.4f}, count: {df_segment.shape[0]}, average duration: {df_segment.groupby('segment')['scratch_count'].max(1).mean()/sf} s")
    return df


def load_data_raw(path, remove_subject_filter,include_subject_filter, list_features, motion_filter, motion_threshold,group=True):
    start_time = datetime.now()
    print("---------------> Starting load_test_data() function:  ", start_time)
    files = sorted(glob.glob(os.path.join(path, f"*.parquet.gzip")))
    if remove_subject_filter is not None:
        files = [file for file in files if all(PID not in file for PID in remove_subject_filter)]
    
    if include_subject_filter is not None:
        files = [file for file in files if include_subject_filter in file]
    df_motion_list = []
    df_nomotion_list = []
    df=pd.DataFrame()
    print(f"Total Files: {len(files)}, path={path}")
    for counter, file in enumerate(files):#files[:50]
        current_subject = re.findall(r"US\d+", file)[0]
        if counter % 50 == 0: 
            print(f"---->Processing file {counter} out of {len(files)} at {datetime.now()}") 
        df_temp = pd.read_parquet(file)
        if df_temp.empty or len(df_temp.columns) < 2:
            custom_print(f"Warning: Data file empty. \nFile: {file}")
            continue
        #Remove when the two labelers disagree on the first question
        df_temp=df_temp.loc[((df.ScratchedUsingUpperExtremityLabeler1=='YES') & (df.ScratchedUsingUpperExtremityLabeler2=='YES')) | ((df.ScratchedUsingUpperExtremityLabeler1=='NO') & (df.ScratchedUsingUpperExtremityLabeler2=='NO'))]
        #remove when cannot determine scratch is reported
        df_temp=df_temp.loc[(df_temp.UsageLabeler1 != "CANNOT DETERMINE SCRATCH") & (df_temp.UsageLabeler2 != "CANNOT DETERMINE SCRATCH")]
        #remove when cannot determine wrist is reported
        df_temp=df_temp.loc[(df_temp.UsageLabeler1 != "CANNOT DETERMINE WRIST") & (df_temp.UsageLabeler2 != "CANNOT DETERMINE WRIST")]
        #df_temp=df_temp[df_temp['non-wear']==0]
        df_temp["max_xyz"]=df_temp[['x_std', 'y_std','z_std']].values.max(1)
        df_temp['wrist']=np.where(df_temp.wrist=="left",1,0)

        if motion_filter:
            df_nomotion_temp=df_temp[df_temp['max_xyz']<=motion_threshold][['SCRATCHEND', 'TSOSTART_philips', 'TSOEND_philips',"timestamp","scratch","scratch_u",'wrist','overlap','SCRATCHSTART','max_xyz']] #ENMO_std==0.0917 is the 95th percentile of ENMO std during non motion episodes as detected from the videos
            df_nomotion_temp['SUBJECT'] = current_subject
            df_nomotion_list.append(df_nomotion_temp)
            df_temp= df_temp[df_temp['max_xyz']>motion_threshold]
        
#         #Force scratch to false when a given window contains scratch of less than 1s
#         df_temp.loc[df_temp.overlap <1,'scratch']=False
        
        if list_features:
            #df_temp=(df_temp.loc[:, ['SCRATCHEND', 'TSOSTART_philips', 'TSOEND_philips',"timestamp","scratch","scratch_u",'wrist','overlap','SCRATCHSTART','max_xyz', *df_temp.loc[:, 'x_list':'PC2_ac_list'].columns]])#'non-wear',
            df_temp=(df_temp.loc[:, [ 'SCRATCHEND', 'TSOSTART_philips', 'TSOEND_philips',"timestamp","scratch","scratch_u",'wrist','overlap','SCRATCHSTART','max_xyz', 'x_list','y_list','z_list','ENMO_list']])#'non-wear',
        else:
            df_temp=(df_temp.loc[:, [ 'SCRATCHEND', 'TSOSTART_philips', 'TSOEND_philips',"timestamp","scratch","scratch_u",'wrist','overlap','SCRATCHSTART','max_xyz', *df_temp.loc[:, 'x_amax':'rqa_ratio_laminarity_determinism'].columns]]).drop(columns=['timestamp_end'])#'non-wear',
        
        df_temp['SUBJECT'] = current_subject
        df_motion_list.append(df_temp)
        # if df.empty:
        #     df=df_temp
        # else:
        #     df = pd.concat([df,df_temp], ignore_index=True)   

    # -------------------------
    # Clean & Transform Dataset
    # -------------------------
    current_time = datetime.now()
    print("---------------> Starting dataframes concatenation..... Time spent so far, from start:  ",
          current_time - start_time)

    df_motion = pd.concat(df_motion_list, ignore_index=True)
    df_motion = remove_missmatch_labels(df_motion)

    if len(df_nomotion_list)>0:
        df_nomotion=pd.concat(df_nomotion_list, ignore_index=True)
        df_nomotion=remove_missmatch_labels(df_nomotion)
    else:
        df_nomotion = pd.DataFrame()
    if list_features:
        X = np.concatenate([
            df_motion["x_list"].tolist(),
            df_motion["y_list"].tolist(),
            df_motion["z_list"].tolist(),
            df_motion["ENMO_list"].tolist()#,
#              df["ENMO_ac_list"].tolist(),
#              df["PC1_list"].tolist(),
#              df["PC2_list"].tolist(),
#              df["PC1_ac_list"].tolist(),
#              df["PC2_ac_list"].tolist()
        ], axis=1).astype('float32')

        current_time = datetime.now()
        print("---------------> Concatenation finished.... Time spend so far, from start:  ", current_time - start_time)
        X = pd.DataFrame(X)
        
#         X= pd.concat([X, df.drop(['x_list', 'y_list','z_list','ENMO_list'], axis=1)], axis=1)# 
        X['SUBJECT'] = df_motion['SUBJECT'].values
        X['overlap'] = df_motion['overlap'].values
        X['scratch'] = df_motion['scratch'].values
        X['scratch_u'] = df_motion['scratch_u'].values
        X['max_xyz'] = df_motion['max_xyz'].values
        X['timestamp'] = df_motion['timestamp'].values
        X['SCRATCHSTART'] = df_motion['SCRATCHSTART'].values
        X['SCRATCHEND'] = df_motion['SCRATCHEND'].values
        X['TSOSTART_philips'] = df_motion['TSOSTART_philips'].values
        X['TSOEND_philips'] = df_motion['TSOEND_philips'].values
        X['wrist'] = df_motion['wrist'].values
        #X['non-wear']= df['non-wear'].values
        df_motion = X

    
    print(f"Concatenated Dataframe Shape: {df_motion.shape}, timestamp: {datetime.now()}")
    del df_temp, X
    
    print(f"Length of training/testing set with motion: {len(df_motion)}")
    print(f"Length of training/testing set with no-motion: {len(df_nomotion)}")
    
    df_motion = filter_inbed(df_motion)
    df_nomotion = filter_inbed(df_nomotion)
    print("df.shape after removing not INBED: ", df_motion.shape)
    print("df_nomotion.shape after removing not INBED: ", df_nomotion.shape)

    df_motion["PID"]=df_motion["SUBJECT"]
    df_nomotion["PID"]=df_nomotion["SUBJECT"]
    if group:
        df_motion=add_groups(df_motion)
        df_nomotion=add_groups(df_nomotion)
    

    
    return df_motion,df_nomotion


def load_data_from_h5(file_path):
    """Load data from H5 file into pandas DataFrames"""

    def _convert_to_original_types(df, dtype_info):
        for col, dtype_str in dtype_info.items():
            if col in df.columns:
                if 'datetime' in dtype_str:
                    if df[col].dtype == object:
                        df[col] = df[col].apply(
                            lambda x: x.decode('utf-8') if isinstance(x, bytes) else x
                        )
                    df[col] = pd.to_datetime(df[col])
        return df

    with h5py.File(file_path, 'r') as f:
        # Load metadata
        metadata = {}
        for key, value in f['metadata'].attrs.items():
            metadata[key] = value
        
        dtype_info = {}
        for key, value in f['dtype_info'].attrs.items():
            dtype_info[key] = value
        
        # Load scaler
        scaler = StandardScaler()
        scaler.mean_ = f['scaler']['mean'][()]
        scaler.scale_ = f['scaler']['scale'][()]
        
        # Load train data
        train_data = {}
        for col in f['train'].keys():
            train_data[col] = f['train'][col][()]
        df_train = pd.DataFrame(train_data)
        
        # Load validation data
        val_data = {}
        for col in f['val'].keys():
            val_data[col] = f['val'][col][()]
        df_val = pd.DataFrame(val_data)
        
        # Load test data
        test_data = {}
        for col in f['test'].keys():
            test_data[col] = f['test'][col][()]
        df_test = pd.DataFrame(test_data)

        df_train = _convert_to_original_types(df_train, dtype_info)
        df_val = _convert_to_original_types(df_val, dtype_info)
        df_test = _convert_to_original_types(df_test, dtype_info)

    return df_train, df_val, df_test, metadata, scaler


# ==================== H5 Dataset Helper Functions ====================

