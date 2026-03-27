# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 08:15:17 2024

@author: MBoukhec
"""

import subprocess

shell_script = '''
sudo python3.11 -m pip install -r munge/predictive_modeling/requirements-ml.txt
sudo python3.11 -m pip install -e .
sudo python3.11 -m pip install optuna
'''
result = subprocess.run(shell_script, shell=True, capture_output=True, text=True)

import glob
import os
import random
import re
import shutil
import logging
import traceback
from pprint import pprint
from datetime import datetime
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from scipy.signal import butter, filtfilt
from scipy.signal import stft
from sklearn.metrics import roc_curve, auc,precision_recall_curve, average_precision_score
import concurrent.futures

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import h5py
import joblib


# Global setup
error_logs = []
processing_logs = []

def df_subset(_df,ratio,stratified):
    random_indices=[]
    # indices = np.random.choice(df.index, int(len(df[(df.SUBJECT==PID) & (df.scratch == False)])*ratio), replace=False)
    # random_indices=np.append(random_indices,indices)
    if stratified:
        #we take random ratio from all subjects and same distrbutino of scratch/noscratch
        for PID in _df.SUBJECT.unique():
            indices = np.random.choice(_df[(_df.SUBJECT==PID) & (_df.scratch == True)].index, int(len(_df[(_df.SUBJECT==PID) & (_df.scratch == True)])*ratio), replace=False)
            random_indices=np.append(random_indices,indices)
            indices = np.random.choice(_df[(_df.SUBJECT==PID) & (_df.scratch == False)].index, int(len(_df[(_df.SUBJECT==PID) & (_df.scratch == False)])*ratio), replace=False)
            random_indices=np.append(random_indices,indices)
    else:
        #we take random ratio with the same distrbutino of scratch/noscratch
        indices = np.random.choice( _df[_df.scratch == True].index, int(len(_df[_df.scratch == True])*ratio), replace=False)
        random_indices=np.append(random_indices,indices)
        indices = np.random.choice( _df[_df.scratch == False].index, int(len(_df[_df.scratch == False])*ratio), replace=False)
        random_indices=np.append(random_indices,indices)
    
    return random_indices

def generate_spaced_integers(a, b, d, n):
    # Check if it's possible to generate n integers with the spacing condition
    if b - a < (n - 1) * d:
        raise ValueError("Not enough space to generate the required number of integers with the specified minimum difference.")

    # Create a list of n random integers spaced by at least d
    start = a

    # Generate the numbers while ensuring the minimum spacing
    numbers = []
    for i in range(n):
        num = random.randint(start, b - (n - i - 1) * d)  # Adjust max based on remaining slots
        numbers.append(num)
        start = num + d  # Move the start position for next number

    return sorted(numbers)

def get_subsegment(data,reduce=False):
    boundaries=data.groupby('change').agg({'timestamp':['min','max']}).reset_index()#[data.Activity!='Other']
    boundaries.columns=boundaries.columns.map('_'.join).str.strip('_')
    if boundaries.empty:
        chosen_data = pd.DataFrame()
    else:
        chosen_activity = random.randint(0,boundaries.shape[0]-1) # In case there are many non 'other' activities, pick one randomly
        chosen_data = data[data['timestamp'].between((boundaries.timestamp_min[chosen_activity] - pd.Timedelta('1s')),(boundaries.timestamp_max[chosen_activity])+ pd.Timedelta('1s'))]
        if reduce:
            sf=10
            min_length=1*sf
            s,e=generate_spaced_integers(chosen_data.index.min()+min_length, chosen_data.index.max(), min_length, 2)
            chosen_data=chosen_data[(chosen_data.index<s) | (chosen_data.index>e) ] # remove a random block from the chosen activity
    return chosen_data

def augment_iteration(args):
    df, i, existing_segments,interchange,verbose = args  # Unpack the arguments
    new_df_i = []
    s = 0
    
    for seg, data in df.groupby('segment'):
        if (s % 500 == 0)&(verbose): 
            print(f"---->Iteration {i}. Processing {s} out of {len(existing_segments)}. Length: {len(new_df_i)}. Time: {datetime.now()} ") 
        if interchange:
            data_to_replace = get_subsegment(data, reduce=False)
            random_seg = random.choice(existing_segments)
            new_segment = pd.concat([
                data[data.index < data_to_replace.index.min()],
                get_subsegment(df[df.segment == random_seg], reduce=False),
                data[data.index > data_to_replace.index.max()]
            ])
        else:
            new_segment=data        
        new_segment['segment'] = f"{seg}_A{str(i)}"
        
        #Shuffling x,y,z
        xyz_p=np.random.permutation(['x','y'])
        new_segment=new_segment.rename(columns={'x': xyz_p[0], 'y': xyz_p[1]})
        
        #Randomly reversing the segment
        if random.choice([True, False]):
            new_segment=new_segment.iloc[::-1].reset_index(drop=True)
        new_df_i.append(new_segment)
        s += 1
    del new_segment
    new_df_i=pd.concat(new_df_i, ignore_index=True)
    return new_df_i

def augment_dataset(df, num_iterations,interchange=True, verbose=True, norm_scratch_length=True ):
    new_df = []
    existing_segments = df.segment.unique()
    new_df.append(df)

    iterations = list(range(num_iterations + 1))
    args = [(df, i, existing_segments,interchange,verbose) for i in iterations]  # Prepare arguments for each iteration

    with concurrent.futures.ProcessPoolExecutor() as executor: # ProcessPoolExecutor()
        results = executor.map(augment_iteration, args)  # Pass the arguments list to map
    
    # Collecting results
    for result in results:
        new_df.append(result)  # Extend the new_df with the results of each iteration
    
    print(f"concatenating augmented dataframes {datetime.now()}")
    new_df = pd.concat(new_df, ignore_index=True)
    scratch_duration = new_df.groupby('segment').agg({'scratch': ['sum', 'count']}).reset_index()
    scratch_duration.columns = scratch_duration.columns.map('_'.join).str.strip('_')
    scratch_duration['scratch_duration'] = scratch_duration.scratch_sum
    if norm_scratch_length:
        scratch_duration['scratch_duration'] /= scratch_duration.scratch_count
    else:
        scratch_duration['scratch_duration'] /= 20.0
    new_df = new_df.drop(['scratch_sum', 'scratch_count', 'scratch_duration'], axis=1).merge(scratch_duration)
    new_df = new_df.drop(['segment_scratch'], axis=1).merge(new_df.groupby('segment')['scratch'].any().reset_index().rename(columns={'scratch':'segment_scratch'}))
    df_segment=new_df.groupby('segment').max(1).reset_index()
    print(f"Segment prevalence after augmentation: {df_segment[(df_segment.segment_scratch==True)].shape[0]/df_segment.shape[0]:.4f}, count: {df_segment.shape[0]}, average duration: {df_segment.groupby('segment')['scratch_count'].max(1).mean()/20} s")
    return new_df

def augment_iteration_stream(args):
    df, i, existing_segments, interchange, verbose = args  # Unpack the arguments
    s = 0
    
    # Instead of a list, we can iterate and yield segments directly
    for seg, data in df.groupby('segment'):
        if (s % 500 == 0) & (verbose):
            print(f"---->Iteration {i}. Processing {s} out of {len(existing_segments)}. Time: {datetime.now()}") 
        
        if interchange:
            data_to_replace = get_subsegment(data, reduce=False)
            random_seg = random.choice(existing_segments)
            
            # Create new segment with concatenation
            new_segment = pd.concat([
                data[data.index < data_to_replace.index.min()],
                get_subsegment(df[df.segment == random_seg], reduce=True),
                data[data.index > data_to_replace.index.max()]
            ])
        else:
            new_segment = data.copy()  # Make a copy to avoid changing the original data
        
        new_segment['segment'] = f"{seg}_A{str(i)}"
        
        # Shuffling x, y, z
        xyz_p = np.random.permutation(['x', 'y', 'z'])
        new_segment.rename(columns={'x': xyz_p[0], 'y': xyz_p[1], 'z': xyz_p[2]}, inplace=True)
        
        # Randomly reversing the segment if required
        if random.choice([True, False]):
            new_segment = new_segment.iloc[::-1].reset_index(drop=True)

        # Instead of appending to a list, we yield the new_segment directly
        yield new_segment
        
        s += 1



def augment_dataset_stream(df, num_iterations, interchange=True, verbose=True):
    existing_segments = df.segment.unique()

    # Using a generator for results to avoid building a large list in memory
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(augment_iteration_stream, [(df, i, existing_segments, interchange, verbose) for i in range(num_iterations + 1)])
        
        # Collecting results directly into the dataframe
        new_df = pd.concat(results, ignore_index=True)
    
    print(f"Concatenating augmented dataframes {datetime.now()}")
    
    scratch_duration = new_df.groupby('segment').agg(
        scratch_sum=('scratch', 'sum'),
        scratch_count=('scratch', 'count')
    ).reset_index()
    
    # Calculate scratch duration
    scratch_duration['scratch_duration'] = scratch_duration['scratch_sum'] / scratch_duration['scratch_count']
    
    # Merge and clean up
    new_df = new_df.drop(['scratch_sum', 'scratch_count'], axis=1)\
                   .merge(scratch_duration, on='segment', how='left')
    
    df_segment = new_df.groupby('segment').max().reset_index()
    
    # Log segment prevalence
    segment_prevalence = df_segment[df_segment.segment_scratch == True].shape[0] / df_segment.shape[0]
    average_duration = df_segment.groupby('segment')['scratch_count'].max().mean() / 20
    print(f"Segment prevalence after augmentation: {segment_prevalence:.4f}, count: {df_segment.shape[0]}, average duration: {average_duration} s")
    
    return new_df


def remove_missmatch_labels(df):
    #remove when there is mismatching betweet left and right labelers
    missmatch=len(df)        
    # gt_error=pd.read_csv("/mnt/data/Ground-truth/scratch_GT_offset_no agreement.csv")
    gt_error=pd.read_csv("/mnt/data/Ground-truth/scratch_GT_offset_no agreement.csv")
    gt_error_L=gt_error[['ParticipantIdentifier','Start_tz_leftoffset']].dropna()
    gt_error_R=gt_error[['ParticipantIdentifier','Start_tz_rightoffset']].dropna()
    # gt_error['StartTimestamp']=gt_error['StartTimestamp'].str[:-6]
    df['SCRATCHSTART']=pd.to_datetime(df['SCRATCHSTART'],format='mixed').dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    df=pd.merge(df, gt_error_L, left_on=['SUBJECT','SCRATCHSTART'],right_on=['ParticipantIdentifier','Start_tz_leftoffset'], how='left')
    df=pd.merge(df, gt_error_R, left_on=['SUBJECT','SCRATCHSTART'],right_on=['ParticipantIdentifier','Start_tz_rightoffset'], how='left')
    df=df[~(df['Start_tz_leftoffset'].notnull() | df['Start_tz_rightoffset'].notnull())]
    df=df.drop(['ParticipantIdentifier_x','ParticipantIdentifier_y', 'Start_tz_rightoffset','Start_tz_leftoffset'], axis=1)
    # df.drop(['ParticipantIdentifier_x','ParticipantIdentifier_x' 'SCRATCHSTART','StartTimestamp'], axis=1)
    missmatch=missmatch-len(df)
    print(f"-----> lenghth of removed instances because of label missmatching = {str(missmatch)}")
    return df

def filter_inbed(df): #Loos like there is a bug in this code in how the timestamp is merged.Not used anymore. Refer to filter_segment_inbed instead.
    # bed= pd.read_csv("/mnt/data/Ground-truth/bed_in_out_sec.csv")
    bed= pd.read_csv("/mnt/data/Ground-truth/bed_in_out_sec.csv")
    bed["timestamp"]=pd.to_datetime(bed["Timestamp_tz"].str[:-13],format='mixed')
    df2=pd.merge(
        left=df, 
        right=bed[['SUBJECT','BedStatusConsensus','timestamp']],
        how='left',
        left_on=['SUBJECT','timestamp'],
        right_on=['SUBJECT','timestamp'],
    )
    #df2["day"]=df2['timestamp'].astype(str).str[:10]
    df2['BedStatusConsensus']=df2['BedStatusConsensus'].fillna('Notfound')
    df2['bed']=0
    df2.loc[df2['BedStatusConsensus']=="INBED",'bed']=1
    df2.loc[df2['BedStatusConsensus']=="OUTBED",'bed']=2
    print(df2.BedStatusConsensus.value_counts(dropna=False))
    #df2=df2[df2.bed==1]
    #df2.drop(['BedStatusConsensus','bed'], axis=1, inplace=True)
    df2.drop(['BedStatusConsensus','timestamp_s'], axis=1, inplace=True)
    return df2

def filter_segment_inbed(df):
    df['timestamp_s']=df.timestamp.dt.floor('s')
    # bed= pd.read_csv("/mnt/data/Ground-truth/bed_in_out_sec.csv")
    bed= pd.read_csv("/mnt/data/Ground-truth/bed_in_out_sec.csv")
    bed["timestamp_s"]=pd.to_datetime(bed["Timestamp_tz"].str[:-13],format='mixed')
    df2=pd.merge(
        left=df, 
        right=bed[['SUBJECT','BedStatusConsensus','timestamp_s']],
        how='left',
        left_on=['SUBJECT','timestamp_s'],
        right_on=['SUBJECT','timestamp_s'],
    )
    #df2["day"]=df2['timestamp'].astype(str).str[:10]
    df2['BedStatusConsensus']=df2['BedStatusConsensus'].fillna('Notfound')
    df2['bed']=0
    #df2.loc[df2['BedStatusConsensus']=="INBED",'bed']=0
    #df2.loc[df2['BedStatusConsensus']!="INBED",'bed']=1#1
    df2.loc[~df2.BedStatusConsensus.isin(["INBED","OUTBED"]),'bed']=1
    
    print(df2.BedStatusConsensus.value_counts(dropna=False))
    #df2=df2[df2.bed==1]
    df2.drop(['timestamp_s'], axis=1, inplace=True)#'bed', 'BedStatusConsensus',
    return df2

def add_groups(df,dataset='NOPROD'):
    # Create 3 groups of participants
    if dataset=="nsucl":
        participant_groups = [['S_08161', 'S_44188', 'S_41046', 'S_19944', 'S_91241', 'S_08360', 'S_32948'], 
                               ['S_78670', 'S_56937', 'S_52658', 'S_67690', 'S_64825', 'S_54977'], 
                              ['S_47352', 'S_07303', 'S_64686', 'S_19366', 'S_14892', 'S_90366'],
                             ['S_37294', 'S_45677', 'S_17753', 'S_01481', 'S_27362', 'S_58621']]
        
    else:
        participant_groups = [['US10001002', 'US10001004', 'US10007001', 'US10008003', 'US10008004', 'US10008005', 'US10008006'], 
                              ['US10008009', 'US10008010', 'US10008011', 'US10008014', 'US10008015', 'US10012001', 'US10012002'], 
                              ['US10012005', 'US10012006', 'US10012007', 'US10012008', 'US10012009', 'US10013002', 'US10013003', 'US10013005'],
                             ['US10001001','US10013009','US10013006','US10012004','US10008008','US10008007','US10012003']]
    for fold_idx, group in enumerate(participant_groups, 1):
        df.loc[df['SUBJECT'].isin(group), 'FOLD'] = f'FOLD{fold_idx}'
    return df

# def add_groups(df):
#     # Sample list of participants
#     participants = df['SUBJECT'].unique().tolist()

#     # Set the seed for reproducibility
#     random.seed(42)  

#     # Shuffle the participants
#     random.shuffle(participants)

#     # Split the shuffled list into 4 groups
#     group_size = len(participants) // 4
#     groups = [participants[i * group_size:(i + 1) * group_size] for i in range(4)]

#     # Handle remaining participants if the number of participants is not divisible by 4
#     remaining_participants = participants[group_size * 4:]

#     # If there are remaining participants, distribute them evenly among the groups
#     for i, participant in enumerate(remaining_participants):
#         groups[i % 4].append(participant)
        
#     for fold_idx, group in enumerate(groups, 1):
#         df.loc[df['SUBJECT'].isin(group), 'FOLD'] = f'FOLD{fold_idx}'
#     return df



def add_spectrum(df):
    X=[]
    for index,seq in df.groupby("segment"):
        STFTx=stft(seq.x.values, fs=20, nperseg=20, noverlap=19,window="hamming")[2].astype(np.float32)
        STFTy=stft(seq.y.values, fs=20, nperseg=20, noverlap=19,window="hamming")[2].astype(np.float32)
        STFTz=stft(seq.z.values, fs=20, nperseg=20, noverlap=19,window="hamming")[2].astype(np.float32)
        X_arr=np.concatenate([STFTx.T[:len(seq)],STFTy.T[:len(seq)],STFTz.T[:len(seq)]],axis=1)
        X.append(pd.DataFrame(X_arr))   
    X=pd.concat(X, ignore_index=True)
    X.columns = ["stft_"+str(n) for n in X.columns]
    X.reset_index(drop=True,inplace=True)
    df.reset_index(drop=True,inplace=True)
    df=pd.concat([df, X], axis=1)
    return df

def get_dominant_hand():
    # Read non-dominant hand information
    hands = pd.read_excel('/mnt/data/Ground-truth/SDI.xlsx')
    hands = hands[1:]  # Skip header row if necessary
    hands = hands[['Patient_ID', 'SDI_SDI_QS2.1']]
    hands.columns = ['PID', 'wrist']
    # Invert the dominant hand to get the non-dominant hand
    hands['wrist'] = np.where(hands['wrist'] == 'Right', 'left', 'right')
    hands['PID'] = hands['PID'].str.split('-').str[1]
    return hands


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

def assign_incremental_numbers(group, n):
    group['mini_segment'] = ((group.index-min(group.index)) // n) + 1
    return group


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

def high_pass_filter(data, cutoff_frequency, sample_rate):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(4, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

def extract_gravity(acc_data, alpha=0.1):
    """
    Extract gravity vector from accelerometer signals using a low-pass filter.
    :param acc_data: numpy array of accelerometer readings (Nx3)
    :param alpha: low-pass filter coefficient (between 0 and 1)
    :return: gravity vector extracted from the accelerometer data
    """
    gravity = np.zeros(1)
    gravity_values = []

    for t in range(acc_data.shape[0]):
        # Low-pass filter to extract gravity
        gravity = alpha * acc_data[t] + (1 - alpha) * gravity
        gravity_values.append(gravity)

    return np.array(gravity_values)



class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, verbose=False, restore_best_weights=True):
        """
        Early stopping to halt training when validation loss does not improve.
        
        Parameters:
        patience (int): How many epochs to wait after the last improvement before stopping.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        verbose (bool): Whether to print messages when the validation loss improves or not.
        restore_best_weights (bool): Whether to restore model weights from the epoch with the best validation loss.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        # If this is the first validation loss, initialize best_score
        if self.best_score is None:
            self.best_score = val_loss
            self.best_weights = model.state_dict()
            return False
        
        # If the validation loss improved, reset counter and update best score
        if val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.best_weights = model.state_dict()
            self.counter = 0
            if self.verbose:
                print(f"Validation loss improved: {val_loss:.6f}")
            return False
        
        # If no improvement, increment the counter
        self.counter += 1
        if self.verbose:
            print(f"Validation loss did not improve for {self.counter} epochs.")
        
        # If patience is exceeded, return True to stop training
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        
        return False
    
    def restore(self, model):
        """Restore model weights from the best epoch."""
        if self.restore_best_weights:
            model.load_state_dict(self.best_weights)
            
            
class Retraining:
    def __init__(self, verbose=False):
        """
        Early stopping to halt training when validation loss does not improve.
        
        Parameters:
        patience (int): How many epochs to wait after the last improvement before stopping.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        verbose (bool): Whether to print messages when the validation loss improves or not.
        restore_best_weights (bool): Whether to restore model weights from the epoch with the best validation loss.
        """
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_weights = None
        self.best_index = None
    
    def check_performance(self, val_loss, model):
        # If this is the first validation loss, initialize best_score
        self.counter += 1
        if self.best_score is None:
            self.best_score = val_loss
            self.best_weights = model.state_dict()
            self.best_index = self.counter
        else:
            # If the validation loss improved, reset counter and update best score
            if val_loss < self.best_score :
                self.best_score = val_loss
                self.best_weights = model.state_dict()
                self.best_index = self.counter 
                if self.verbose:
                    print(f"Validation loss improved. Best model is from training {self.best_index}")
            else:
                # If no improvement, increment the counter
                if self.verbose:
                    print(f"Validation loss did not improve. Best model is from training {self.best_index}")   
    
    def restore(self, model):
        """Restore model weights from the best epoch."""
        if self.verbose:
            print(f"Best model restored from training {self.best_index}")
        model.load_state_dict(self.best_weights)
        return model
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss).mean()
        return loss


    
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Focal Loss for binary classification.
        
        Parameters:
        - inputs: Predicted probabilities (after sigmoid), shape (batch_size,)
        - targets: Ground truth labels, shape (batch_size,)
        
        Returns:
        - loss: Computed focal loss
        """
        # Ensure inputs are between 0 and 1 (for probabilities)
        inputs = torch.sigmoid(inputs)

        # Compute the binary cross entropy loss (log loss)
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        # Compute p_t (the predicted probability for the true class)
        p_t = inputs * targets + (1 - inputs) * (1 - targets)

        # Compute the focal loss component
        loss = self.alpha * (1 - p_t) ** self.gamma * BCE_loss

        # Apply reduction method
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class GCELoss(nn.Module):
    def __init__(self, num_classes=2, q=0.9):
        super(GCELoss, self).__init__()
        self.q = q
        self.num_classes = num_classes
        self.eps = 1e-9  # Define eps here

    def forward(self, pred, labels):
        # Convert logits to probabilities
        pred = torch.sigmoid(pred)  # Use sigmoid for binary classification
        pred = torch.stack([1 - pred, pred], dim=1)  # Shape [batch_size, 2]
        pred = torch.clamp(pred, min=self.eps, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return loss.mean()


def plot_confusion_metrics(y_test, y_pred, output_filepath):
    try:
        # Calculate the confusion matrix
        #labels = sorted(list(set(y_test)))
        labels=[False,True]
        cm = metrics.confusion_matrix(y_test, y_pred, labels=labels)

        # Convert the confusion matrix into a pandas DataFrame
        df_cm = pd.DataFrame(cm, index=labels, columns=labels)

        # Plot the confusion matrix using seaborn
        plt.figure(figsize=(10, 7))
        sns.heatmap(df_cm, annot=True, fmt="g", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(output_filepath, dpi=300)
        plt.show()
        # Optionally, you can print the classification report
        #print(classification_report(y_test, y_pred))
    except Exception as e:
        raise Exception(f"Failed to plot confusion matrix. Exception: {e}")
    

def plot_learning_curves(learning_metrics, output_filepath):
    # Extract the metrics
    train_losses = np.array(learning_metrics.get('train_losses', []), dtype=float)
    train_accuracies = np.array(learning_metrics.get('train_accuracies', []), dtype=float)
    train_F1s=np.array(learning_metrics.get('train_F1s', []), dtype=float)
    val_losses = np.array(learning_metrics.get('val_losses', []), dtype=float)
    val_accuracies = np.array(learning_metrics.get('val_accuracies', []), dtype=float)
    val_F1s=np.array(learning_metrics.get('val_F1s', []), dtype=float)

    epochs = range(1, len(train_losses) + 1)

    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))  # Changed to 1 row, 2 columns

    # Plot training and validation loss
    ax1.plot(epochs, np.array(train_losses, dtype=float), 'b-', label='Training loss')
    ax1.plot(epochs, np.array(val_losses, dtype=float), 'r-', label='Validation loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot training and validation accuracy
    ax2.plot(epochs, np.array(train_accuracies, dtype=float), 'b-', label='Training accuracy')
    ax2.plot(epochs, np.array(val_accuracies, dtype=float), 'r-', label='Validation accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    # Plot validation F1
    ax3.plot(epochs, np.array(train_F1s, dtype=float), 'b-', label='Train F1')
    ax3.plot(epochs, np.array(val_F1s, dtype=float), 'r-', label='Validation F1')
    ax3.set_title('Train F1')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('F1')
    ax3.legend()

    # Layout adjustment and saving the figure
    plt.tight_layout()
    plt.savefig(output_filepath, dpi=300)
    plt.show()
    plt.close()

def plot_roc_precision_recall_auc(y_test, y_score, output_filepath):
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(14, 5))

    # Subplot 1: ROC curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'(AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.text(0.6, 0.2, f'Counts: {len(y_test)}', fontsize=12)

    # Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    average_precision = average_precision_score(y_test, y_score)

    # Subplot 2: Precision-Recall curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f'AP = {average_precision:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.text(0.6, 0.2, f'Counts: {len(y_test)}', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_filepath, dpi=300)
    plt.show()

def calculate_metrics_nn(actuals, predictions, classification=True):
    if classification:
        # Ensure predictions are integers
        predictions = predictions.astype(int)
        actuals = actuals.astype(int)
        # Calculating various classification metrics
        metrics_values = {
            'accuracy': metrics.accuracy_score(actuals, predictions),
            'balanced_accuracy': metrics.balanced_accuracy_score(actuals, predictions),
            # 'precision': metrics.precision_score(actuals, predictions, zero_division=0),
            'precision_macro': metrics.precision_score(actuals, predictions, average='macro', zero_division=0),
            'precision_micro': metrics.precision_score(actuals, predictions, average='micro', zero_division=0),
            'precision_weighted': metrics.precision_score(actuals, predictions, average='weighted', zero_division=0),
            # 'recall': metrics.recall_score(actuals, predictions, zero_division=0),
            'recall_macro': metrics.recall_score(actuals, predictions, average='macro', zero_division=0),
            'recall_micro': metrics.recall_score(actuals, predictions, average='micro', zero_division=0),
            'recall_weighted': metrics.recall_score(actuals, predictions, average='weighted', zero_division=0),
            # 'f1_score': metrics.f1_score(actuals, predictions, zero_division=0),
            'f1_score_macro': metrics.f1_score(actuals, predictions, average='macro', zero_division=0),
            'f1_score_micro': metrics.f1_score(actuals, predictions, average='micro', zero_division=0),
            'f1_score_weighted': metrics.f1_score(actuals, predictions, average='weighted', zero_division=0),
            'roc_auc': metrics.roc_auc_score(actuals, predictions) if len(set(actuals)) == 2 else None,
            'r2': metrics.r2_score(actuals, predictions),
            'mse': metrics.mean_squared_error(actuals, predictions),
            'classification_report': metrics.classification_report(actuals, predictions, output_dict=True, zero_division=0)
        }
    else:
        metrics_values = {
        'explained_variance_score': metrics.explained_variance_score(actuals, predictions),
        'd2_absolute_error_score': metrics.d2_absolute_error_score(actuals, predictions),
        'r2': metrics.r2_score(actuals, predictions),
        'mse': metrics.mean_squared_error(actuals, predictions)}
            
    return metrics_values

def write_txt_to_file(file_path, text):
    with open(file_path, "w") as file:
        file.write(str(text))


def custom_print(text):
    processing_logs.append(text)
    pprint(text)


def create_folder(list_of_folder_paths):
    for folder_path in list_of_folder_paths:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
############################################################            
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="3"
from torch.nn.parallel import DistributedDataParallel
from sklearn import metrics
# from torcheval.metrics.functional import multiclass_f1_score,binary_f1_score
from datetime import datetime
from torch.nn.utils.rnn import pad_sequence
from sklearn.utils.class_weight import compute_class_weight
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from skimage.measure import block_reduce
from sklearn.preprocessing import StandardScaler

torch.cuda.empty_cache()

def batch_generator(df,batch_size=128,stratify=False,shuffle=True,seg_column='segment'):
    steps=0
    if stratify=='oversample': 
        neg=pos=0 
        global_neg_segs=df[df.segment_scratch==False][seg_column].unique()
        neg_segs_map={}
        pos_segs_map={}
        percentiles=range(10,110,10)
        for perc in percentiles:
            low=np.percentile(df.segment_duration,perc-10)
            high=np.percentile(df.segment_duration,perc)
            if perc==100:
                neg_segs_map[perc]=df[(df.segment_scratch==False) & (df.segment_duration>=low)& (df[seg_column].isin(global_neg_segs))][seg_column].unique()
                pos_segs_map[perc]=df[(df.segment_scratch==True)& (df.segment_duration>=low)][seg_column].unique()
            else:
                neg_segs_map[perc]=df[(df.segment_scratch==False) & (df.segment_duration>=low)& (df.segment_duration<high)& (df[seg_column].isin(global_neg_segs))][seg_column].unique()
                pos_segs_map[perc]=df[(df.segment_scratch==True)& (df.segment_duration>=low)& (df.segment_duration<high)][seg_column].unique()

        if shuffle:
            while len(global_neg_segs)>0:
                perc=np.random.choice(percentiles)
                if len(neg_segs_map[perc])>0:
                    n_choice=np.min([int(batch_size/2),len(neg_segs_map[perc])])
                    pos_choice = np.random.choice(pos_segs_map[perc], n_choice, replace=True) # Replace set to True to retake same samples if we don't have enough pos samples in this bin of percentiles
                    neg_choice = np.random.choice(neg_segs_map[perc],n_choice , replace=False)
                    neg_segs_map[perc]=[i for i in neg_segs_map[perc] if i not in neg_choice]
                    global_neg_segs=[i for i in global_neg_segs if i not in neg_choice]
                    steps +=1
                    yield df[df[seg_column].isin(np.concatenate([pos_choice,neg_choice]))]
                    
        else:
            for perc in percentiles:
                while len(neg_segs_map[perc])>0:       
                    n_choice=np.min([int(batch_size/2),len(neg_segs_map[perc])])
                    pos_choice = np.random.choice(pos_segs_map[perc], n_choice, replace=True) # Replace set to True to retake same samples if we don't have enough pos samples in this bin of percentiles
                    neg_choice = np.random.choice(neg_segs_map[perc],n_choice , replace=False)
                    neg_segs_map[perc]=[i for i in neg_segs_map[perc] if i not in neg_choice]
                    global_neg_segs=[i for i in global_neg_segs if i not in neg_choice]
                    steps +=1
                    yield df[df[seg_column].isin(np.concatenate([pos_choice,neg_choice]))]
                    
    elif stratify =='undersample':
        global_neg_segs=df[df.segment_scratch==False][seg_column].unique()
        global_pos_segs=df[df.segment_scratch==True][seg_column].unique()
        while len(global_pos_segs)>0:
            n_choice=np.min([int(batch_size/2),len(global_pos_segs)])
            pos_choice = np.random.choice(global_pos_segs, n_choice, replace=False) 
            neg_choice = np.random.choice(global_neg_segs,n_choice , replace=False)
            
            global_neg_segs=[i for i in global_neg_segs if i not in neg_choice]
            global_pos_segs=[i for i in global_pos_segs if i not in pos_choice]
            steps +=1
            yield df[df[seg_column].isin(np.concatenate([pos_choice,neg_choice]))]

    elif stratify == 'progressive_undersample':
        # Progressive sampling that adapts based on training progress
        global_neg_segs = df[df.segment_scratch==False][seg_column].unique()
        global_pos_segs = df[df.segment_scratch==True][seg_column].unique()

        # Start with balanced ratio, gradually increase negative ratio
        epoch_progress = steps / max(1, len(global_pos_segs) // (batch_size // 2))
        dynamic_ratio = 1.0 + 2.0 * np.tanh(epoch_progress * 0.5)  # 1:1 -> 3:1 progression

        print(f"Progressive undersampling: Step {steps}, Dynamic ratio {dynamic_ratio:.2f}:1")

        while len(global_pos_segs) > 0:
            pos_fraction = 1.0 / (1+dynamic_ratio)
            n_pos = max(1, min(int(batch_size * pos_fraction), len(global_pos_segs)))
            n_neg = min(batch_size - n_pos, len(global_neg_segs))

            pos_choice = np.random.choice(global_pos_segs, n_pos, replace=False)
            neg_choice = np.random.choice(global_neg_segs, n_neg, replace=False)

            global_neg_segs = [i for i in global_neg_segs if i not in neg_choice]
            global_pos_segs = [i for i in global_pos_segs if i not in pos_choice]
            steps += 1

            yield df[df[seg_column].isin(np.concatenate([pos_choice, neg_choice]))]


    elif stratify == 'complexity_undersample':
        # Complexity-based sampling for balanced FP/FN rates
        global_neg_segs = df[df.segment_scratch==False][seg_column].unique()
        global_pos_segs = df[df.segment_scratch==True][seg_column].unique()

        # Compute complexity scores for all segments
        def compute_complexity_scores(df_seg):
            motion_features = df_seg.groupby(seg_column)[['x', 'y', 'z']].agg(['std', 'mean', 'max', 'min'])
            motion_features.columns = ['_'.join(col).strip() for col in motion_features.columns]

            # Motion complexity score
            motion_complexity = (motion_features['x_std'] + motion_features['y_std'] + motion_features['z_std']) / 3

            # Signal range features
            signal_range = ((motion_features['x_max'] - motion_features['x_min']) +
                           (motion_features['y_max'] - motion_features['y_min']) +
                           (motion_features['z_max'] - motion_features['z_min'])) / 3

            scratch_ratio = df_seg.groupby(seg_column)['segment_scratch'].mean()
            segment_lengths = df_seg.groupby(seg_column).size()

            # Complex negative score: high motion + low scratch + moderate length
            complex_neg_score = motion_complexity * (1 - scratch_ratio) * np.log(segment_lengths + 1)

            # Complex positive score: scratching segments with unusual motion patterns
            complex_pos_score = motion_complexity * scratch_ratio * signal_range

            return complex_neg_score, complex_pos_score

        complex_neg_scores, complex_pos_scores = compute_complexity_scores(df)

        # Target ratio: slightly favor negatives to reduce FP
        target_ratio = 2.0  # 2:1 negative:positive ratio

        print(f"Complexity undersampling: Target ratio {target_ratio:.1f}:1")

        while len(global_pos_segs) > 0:
            # Calculate batch composition
            pos_fraction = 1.0 / (1 + target_ratio)
            n_pos = max(1, min(int(batch_size * pos_fraction), len(global_pos_segs)))
            n_neg = min(batch_size - n_pos, len(global_neg_segs))

            # Sample positives with complexity weighting (favor complex scratching patterns)
            pos_weights = complex_pos_scores.reindex(global_pos_segs, fill_value=0.1).values
            pos_weights = pos_weights / pos_weights.sum() if pos_weights.sum() > 0 else np.ones(len(pos_weights)) / len(pos_weights)

            # Sample negatives with complexity weighting (favor hard negatives)
            neg_weights = complex_neg_scores.reindex(global_neg_segs, fill_value=0.1).values
            neg_weights = neg_weights / neg_weights.sum() if neg_weights.sum() > 0 else np.ones(len(neg_weights)) / len(neg_weights)

            # Weighted sampling
            pos_choice = np.random.choice(global_pos_segs, n_pos, replace=False, p=pos_weights)
            neg_choice = np.random.choice(global_neg_segs, n_neg, replace=False, p=neg_weights)

            # Remove selected segments
            global_neg_segs = [i for i in global_neg_segs if i not in neg_choice]
            global_pos_segs = [i for i in global_pos_segs if i not in pos_choice]

            steps += 1

            yield df[df[seg_column].isin(np.concatenate([pos_choice, neg_choice]))]

    else:
        segs = df[seg_column].unique()
        if shuffle:
            np.random.shuffle(segs)
        for i in range(0,len(segs),batch_size):
            seg_choice=segs[i:i+batch_size]
            steps +=1
            yield df[df[seg_column].isin(seg_choice)]
            
            
            
def get_nb_steps(df,batch_size=128,stratify=False,shuffle=True,seg_column='segment'):
    steps=0
    if stratify=='oversample': 
        neg=pos=0 
        global_neg_segs=df[df.segment_scratch==False][seg_column].unique()
        neg_segs_map={}
        pos_segs_map={}
        percentiles=range(10,110,10)
        for perc in percentiles:
            low=np.percentile(df.segment_duration,perc-10)
            high=np.percentile(df.segment_duration,perc)
            if perc==100:
                neg_segs_map[perc]=df[(df.segment_scratch==False) & (df.segment_duration>=low)& (df[seg_column].isin(global_neg_segs))][seg_column].unique()
                pos_segs_map[perc]=df[(df.segment_scratch==True)& (df.segment_duration>=low)][seg_column].unique()
            else:
                neg_segs_map[perc]=df[(df.segment_scratch==False) & (df.segment_duration>=low)& (df.segment_duration<high)& (df[seg_column].isin(global_neg_segs))][seg_column].unique()
                pos_segs_map[perc]=df[(df.segment_scratch==True)& (df.segment_duration>=low)& (df.segment_duration<high)][seg_column].unique()

        if shuffle:
            while len(global_neg_segs)>0:
                perc=np.random.choice(percentiles)
                if len(neg_segs_map[perc])>0:
                    n_choice=np.min([int(batch_size/2),len(neg_segs_map[perc])])
                    pos_choice = np.random.choice(pos_segs_map[perc], n_choice, replace=True) # Replace set to True to retake same samples if we don't have enough pos samples in this bin of percentiles
                    neg_choice = np.random.choice(neg_segs_map[perc],n_choice , replace=False)
                    neg_segs_map[perc]=[i for i in neg_segs_map[perc] if i not in neg_choice]
                    global_neg_segs=[i for i in global_neg_segs if i not in neg_choice]
                    steps +=1
                    
        else:
            for perc in percentiles:
                while len(neg_segs_map[perc])>0:       
                    n_choice=np.min([int(batch_size/2),len(neg_segs_map[perc])])
                    pos_choice = np.random.choice(pos_segs_map[perc], n_choice, replace=True) # Replace set to True to retake same samples if we don't have enough pos samples in this bin of percentiles
                    neg_choice = np.random.choice(neg_segs_map[perc],n_choice , replace=False)
                    neg_segs_map[perc]=[i for i in neg_segs_map[perc] if i not in neg_choice]
                    global_neg_segs=[i for i in global_neg_segs if i not in neg_choice]
                    steps +=1
                    
    elif stratify =='undersample':
        global_neg_segs=df[df.segment_scratch==False][seg_column].unique()
        global_pos_segs=df[df.segment_scratch==True][seg_column].unique()
        while len(global_pos_segs)>0:
            n_choice=np.min([int(batch_size/2),len(global_pos_segs)])
            pos_choice = np.random.choice(global_pos_segs, n_choice, replace=False) 
            neg_choice = np.random.choice(global_neg_segs,n_choice , replace=False)
            
            global_neg_segs=[i for i in global_neg_segs if i not in neg_choice]
            global_pos_segs=[i for i in global_pos_segs if i not in pos_choice]
            steps +=1
    
    elif stratify == 'progressive_undersample':
        global_neg_segs = df[df.segment_scratch==False][seg_column].unique()
        global_pos_segs = df[df.segment_scratch==True][seg_column].unique()

        while len(global_pos_segs) > 0:
            # Progressive ratio calculation (matches batch_generator logic)
            epoch_progress = steps / max(1, len(global_pos_segs) // (batch_size // 2))
            dynamic_ratio = 1.0 + 2.0 * np.tanh(epoch_progress * 0.5)

            pos_fraction = 1.0 / (1+dynamic_ratio)
            n_pos = max(1, min(int(batch_size * pos_fraction), len(global_pos_segs)))
            n_neg = min(batch_size - n_pos, len(global_neg_segs))

            global_neg_segs = global_neg_segs[n_neg:]
            global_pos_segs = global_pos_segs[n_pos:]
            steps += 1

    elif stratify == 'complexity_undersample':
        global_neg_segs = df[df.segment_scratch==False][seg_column].unique()
        global_pos_segs = df[df.segment_scratch==True][seg_column].unique()

        # Fixed ratio for complexity approach
        target_ratio = 2.0

        while len(global_pos_segs) > 0:
            pos_fraction = 1.0 / (1 + target_ratio)
            n_pos = max(1, min(int(batch_size * pos_fraction), len(global_pos_segs)))
            n_neg = min(batch_size - n_pos, len(global_neg_segs))

            global_neg_segs = global_neg_segs[n_neg:]
            global_pos_segs = global_pos_segs[n_pos:]
            steps += 1

    else:
        segs = df[seg_column].unique()
        if shuffle:
            np.random.shuffle(segs)
        for i in range(0,len(segs),batch_size):
            seg_choice=segs[i:i+batch_size]
            steps +=1
    return steps
    
            
def df_subset_segments(_df,ratio,stratified):
    random_indices=[]
    if stratified: # TODO: To change to account for segments
        #we take random ratio from all subjects and same distrbutino of scratch/noscratch
        for PID in _df.SUBJECT.unique():
            indices = np.random.choice(_df[(_df.SUBJECT==PID) & (_df.scratch == True)].index, int(len(_df[(_df.SUBJECT==PID) & (_df.scratch == True)])*ratio), replace=False)
            random_indices=np.append(random_indices,indices)
            indices = np.random.choice(_df[(_df.SUBJECT==PID) & (_df.scratch == False)].index, int(len(_df[(_df.SUBJECT==PID) & (_df.scratch == False)])*ratio), replace=False)
            random_indices=np.append(random_indices,indices)
    else:
        #we take random ratio with the same distrbutino of scratch/noscratch segments
        neg_segs=_df[(_df.segment_scratch==False)]['segment'].unique()
        pos_segs=_df[(_df.segment_scratch==True)]['segment'].unique()
        pos_choise = np.random.choice(pos_segs, int(len(pos_segs)*ratio), replace=False) 
        neg_choise = np.random.choice(neg_segs, int(len(neg_segs)*ratio), replace=False)
        random_indices=_df[_df.segment.isin(np.concatenate([pos_choise,neg_choise]))].index
    
    return random_indices
            
            
def add_padding(batch,device,seg_column='segment', max_seq_len=None, random_start=False, padding_value=0.0):
    X_sequences=[]
    Y_sequences=[]
    x_lens=[]
    label1=[]
    label3=[]
    if max_seq_len is not None:
        max_seq_template = torch.ones(max_seq_len, 3).to(device)
        X_sequences.append(max_seq_template)
        # x_lens.append(max_seq_len)
    for index,seq in batch.groupby(seg_column, sort=False):
        # X_arr=seq.loc[:, ['x', 'y', 'z','angle','wrist','position_segment','position_segmentr','position_TSO','position_TSOr', *seq.loc[:, 'stft_0':'stft_32'].columns]].to_numpy()
        # X_arr=seq.loc[:, ['x', 'y', 'z','angle','position_segment','position_segmentr','position_TSO','position_TSOr']].to_numpy()
        # X_arr=seq.loc[:, ['x', 'y', 'z','angle']].to_numpy()
        xyz = seq
        xyz_p=np.random.permutation(['x','y'])
        xyz=xyz.rename(columns={'x': xyz_p[0], 'y': xyz_p[1]})
        # xyz_p=np.random.permutation(['x','y','z'])
        # seq=seq.rename(columns={'x': xyz_p[0], 'y': xyz_p[1], 'z': xyz_p[2]})
        X_arr=seq.loc[:, ['x', 'y', 'z']].to_numpy()
        # X_arr=seq.loc[:, ['x', 'y', 'z','angle']].to_numpy()
        Y_arr=np.array([seq['scratch'].values]).T
        X_sequences.append(torch.tensor(X_arr,dtype=torch.float32,device=device))#,device=device
        # Y_sequences.append(torch.tensor(Y_arr,dtype=torch.long,device=device))#,device=device
        Y_sequences=np.concatenate((Y_sequences,seq['scratch'].values))
        x_lens.append(len(X_arr))
        label1.append(seq['segment_scratch'].any()*1)
        label3.append(seq['scratch_duration'].max())

    pad_X=pad_sequence(X_sequences, batch_first=True, padding_value=padding_value) #,padding_value=-999
    #label2=pad_sequence(Y_sequences, batch_first=True,padding_value=-999)
    if max_seq_len:
        # discard the max seq template now
        pad_X = pad_X[1:, :, :]
    label2=torch.tensor(Y_sequences,device=device)
    return pad_X, torch.tensor(label1,device=device), label2, torch.tensor(label3,device=device), x_lens


def add_padding_with_position(
    batch,
    device,
    seg_column='segment',
    max_seq_len=None,
    padding_position="tail",  # Options: "tail", "random"
    is_train=True,
    prob=0.5,
    padding_value=0
):
    """
    Add padding to sequences with support for both tail padding and random position padding.
    
    Args:
        batch: DataFrame containing the batch data
        device: Device to place tensors on
        seg_column: Column name for grouping sequences
        max_seq_len: Maximum sequence length after padding. If None, uses max length in batch
        padding_position: Where to place padding - "tail" (default) or "random"
        is_train: Whether this is training mode (random padding only applies during training)
        prob: Probability threshold for applying random padding
        padding_value: Value to use for padding (default: 0)
        
    Returns:
        pad_X: Padded input features tensor of shape [B, L, D]
        label1: Binary classification labels tensor of shape [B]
        label2: Binary sequence classification labels tensor of shape [B, L]
        label3: Regression target tensor of shape [B]
        x_lens: List of original sequence lengths
        seq_start_idx: List of sequence start indices
    """
    X_sequences = []
    Y_sequences = []
    x_lens = []
    label1 = []
    label3 = []
    seq_start_idx = []
    
    # First pass: collect all sequences and their lengths to determine max_seq_len if not provided
    sequences_data = []
    for _, seq in batch.groupby(seg_column, sort=False):
        # Randomly permute x/y/z columns
        # TODO: disabled if position embedding is included
        xyz_p = np.random.permutation(['x', 'y'])
        seq = seq.rename(columns={'x': xyz_p[0], 'y': xyz_p[1]})

        X_arr = seq.loc[:, ['x', 'y', 'z']].to_numpy()
        Y_arr = seq['scratch'].values.astype(np.int64)
        seq_len = len(X_arr)
        
        sequences_data.append({
            'X_arr': X_arr,
            'Y_arr': Y_arr,
            'seq_len': seq_len,
            'seq': seq
        })
    
    # Determine max_seq_len if not provided
    if max_seq_len is None:
        max_seq_len = max(seq_data['seq_len'] for seq_data in sequences_data)
    
    # Second pass: create padded sequences
    for seq_data in sequences_data:
        X_arr = seq_data['X_arr']
        Y_arr = seq_data['Y_arr']
        seq_len = seq_data['seq_len']
        seq = seq_data['seq']
        
        # Initialize tensors with zeros (padding)
        if padding_value != 0:
            pad_tensor = torch.ones(max_seq_len, 3, dtype=torch.float32, device=device) * padding_value
        else:
            pad_tensor = torch.zeros(max_seq_len, 3, dtype=torch.float32, device=device)
        label_pad = torch.zeros(max_seq_len, dtype=torch.long, device=device)
        
        # Determine start index based on padding position
        use_random = (padding_position == "random" and 
                     seq_len < max_seq_len and 
                     torch.rand(1) > prob) 
                    #  and is_train)
        
        if use_random:
            # Random padding position
            start_idx = np.random.randint(0, max_seq_len - seq_len + 1)
        else:
            # Default: tail padding (sequence starts at the beginning)
            start_idx = 0
            
        seq_start_idx.append(start_idx)
        
        # If sequence is too long, crop it
        actual_seq_len = min(seq_len, max_seq_len)
        X_tensor = torch.tensor(X_arr[:actual_seq_len], dtype=torch.float32, device=device)
        Y_tensor = torch.tensor(Y_arr[:actual_seq_len], dtype=torch.long, device=device)
        
        # Place sequence at the determined start position
        pad_tensor[start_idx:start_idx+actual_seq_len, :] = X_tensor
        label_pad[start_idx:start_idx+actual_seq_len] = Y_tensor
        
        X_sequences.append(pad_tensor)
        Y_sequences.append(label_pad)
        x_lens.append(min(seq_len, max_seq_len))  # Store original length before any cropping
        
        label1.append(int(seq['segment_scratch'].any()))
        label3.append(float(seq['scratch_duration'].max()))

    # Stack as batch
    pad_X = torch.stack(X_sequences, dim=0)
    label2 = torch.stack(Y_sequences, dim=0)
    label1 = torch.tensor(label1, dtype=torch.long, device=device)
    label3 = torch.tensor(label3, dtype=torch.float, device=device)

    return pad_X, label1, label2, label3, x_lens, seq_start_idx

def add_padding_pretrain(batch,device,seg_column='segment',mask_rate=0.3, max_seq_len=None, padding_value=0.0):
    X_sequences=[]
    Y_sequences=[]
    x_lens=[]
    label1=[]
    label3=[]
    if max_seq_len is not None:
        max_seq_template = torch.ones(max_seq_len, 3).to(device)
        X_sequences.append(max_seq_template)
        # x_lens.append(max_seq_len)
    for index,seq in batch.groupby(seg_column, sort=False):
        # X_arr=seq.loc[:, ['x', 'y', 'z','angle','wrist','position_segment','position_segmentr','position_TSO','position_TSOr', *seq.loc[:, 'stft_0':'stft_32'].columns]].to_numpy()
        # X_arr=seq.loc[:, ['x', 'y', 'z','angle','position_segment','position_segmentr','position_TSO','position_TSOr']].to_numpy()
        # X_arr=seq.loc[:, ['x', 'y', 'z','angle']].to_numpy()
        xyz = seq
        xyz_p=np.random.permutation(['x','y'])
        xyz=xyz.rename(columns={'x': xyz_p[0], 'y': xyz_p[1]})
        # xyz_p=np.random.permutation(['x','y','z'])
        # seq=seq.rename(columns={'x': xyz_p[0], 'y': xyz_p[1], 'z': xyz_p[2]})
        X_arr=seq.loc[:, ['x', 'y', 'z']].to_numpy()
        # X_arr=seq.loc[:, ['x', 'y', 'z','angle']].to_numpy()
        # Y_arr=np.array([seq['scratch'].values]).T
        X_sequences.append(torch.tensor(X_arr,dtype=torch.float32,device=device))#,device=device
        # Y_sequences.append(torch.tensor(Y_arr,dtype=torch.long,device=device))#,device=device
        # Y_sequences=np.concatenate((Y_sequences,seq['scratch'].values))
        x_lens.append(len(X_arr))
        # label1.append(seq['segment_scratch'].any()*1)
        # label3.append(seq['scratch_duration'].max())

    pad_X=pad_sequence(X_sequences, batch_first=True, padding_value=padding_value) #,padding_value=-999
    #label2=pad_sequence(Y_sequences, batch_first=True,padding_value=-999)
    if max_seq_len:
        # discard the max seq template now
        pad_X = pad_X[1:, :, :]
    # label2=torch.tensor(Y_sequences,device=device)
    return pad_X, torch.tensor(x_lens, dtype=torch.long, device=device)


def add_padding_TSO(batch, device, seg_column='segment', max_seq_len=None, padding_value=0.0):
    """
    Prepare batch data for status prediction task with minute-level aggregated features.

    Args:
        batch: DataFrame with 28 feature columns from load_data_tso:
               - time_cyclic, is_night, temperature (minute-level aggregated base values - 4 features)
               - x_mean, x_std, x_min, x_max, x_q25, x_q75, x_skew, x_kurt, x_cv (9 features)
               - y_mean, y_std, y_min, y_max, y_q25, y_q75, y_skew, y_kurt, y_cv (9 features)
               - z_mean, z_std, z_min, z_max, z_q25, z_q75, z_skew, z_kurt, z_cv (9 features)
               - non-wear, predictTSO (labels)
        device: torch device
        seg_column: column name for segment identifier
        max_seq_len: maximum sequence length for padding (e.g., 1440 for 24h at minute level)
        padding_value: value to use for padding

    Returns:
        pad_X: [batch_size, seq_len, 28] - input features (28 channels)
        labels: [batch_size, 3] - segment-level class labels (other, non-wear, predictTSO)
        x_lens: [batch_size] - original sequence lengths
    """
    # Define all 28 feature columns in order (4 base + 27 statistical)
    feature_cols = (
        ['temperature', 'time_cyclic'] +  # 4 base aggregated values
        # x features (9)
        ['x_mean', 'x_std', 'x_min', 'x_max', 'x_q25', 'x_q75', 'x_skew', 'x_kurt', 'x_cv'] +
        # y features (9)
        ['y_mean', 'y_std', 'y_min', 'y_max', 'y_q25', 'y_q75', 'y_skew', 'y_kurt', 'y_cv'] +
        # z features (9)
        ['z_mean', 'z_std', 'z_min', 'z_max', 'z_q25', 'z_q75', 'z_skew', 'z_kurt', 'z_cv']
    )

    num_features = len(feature_cols)  # Should be 28

    X_sequences = []
    Y_sequences = []  # Minute-level labels
    x_lens = []

    if max_seq_len is not None:
        # Template for padding to max_seq_len with 30 channels
        max_seq_template_X = torch.ones(max_seq_len, num_features).to(device)
        max_seq_template_Y = torch.full((max_seq_len,), -100, dtype=torch.long, device=device)  # Padding label
        X_sequences.append(max_seq_template_X)
        Y_sequences.append(max_seq_template_Y)

    for index, seq in batch.groupby(seg_column, sort=False):
        # Filter to only include minute_id < max_seq_len (first 24 hours) and sort by minute_id
        # if 'minute_id' in seq.columns and max_seq_len is not None:
        #     seq = seq[seq['minute_id'] < max_seq_len].sort_values('minute_id')

        # Sanity check: warn if sequence still exceeds max_seq_len
        if max_seq_len is not None and len(seq) > max_seq_len:
            # print(f"WARNING: Segment {index} has {len(seq)} rows after filtering, truncating to {max_seq_len}")
            seq = seq.iloc[:max_seq_len]

        # Input: 30 feature channels
        X_arr = seq.loc[:, feature_cols].to_numpy()

        # Create minute-level labels: 0=other, 1=non-wear, 2=predictTSO
        # Priority: predictTSO > non-wear > other (for each timestep)
        Y_arr = np.zeros(len(seq), dtype=np.int64)

        # Check each minute
        for i in range(len(seq)):
            if seq.iloc[i]['predictTSO']:
                Y_arr[i] = 2
            elif seq.iloc[i]['non-wear'] == 1:
                Y_arr[i] = 1
            # else: remains 0 (other)

        X_sequences.append(torch.tensor(X_arr, dtype=torch.float32, device=device))
        Y_sequences.append(torch.tensor(Y_arr, dtype=torch.long, device=device))
        x_lens.append(len(X_arr))

    # Pad sequences
    pad_X = pad_sequence(X_sequences, batch_first=True, padding_value=padding_value)
    pad_Y = pad_sequence(Y_sequences, batch_first=True, padding_value=-100)  # -100 is ignore_index for CE loss

    if max_seq_len:
        # Discard the max seq template now
        pad_X = pad_X[1:, :, :]
        pad_Y = pad_Y[1:, :]

    return pad_X, pad_Y, torch.tensor(x_lens, dtype=torch.long, device=device)


def generate_time_cyclic(timestamps, use_sincos=True):
    """
    Generate cyclic time encoding from timestamps.

    Args:
        timestamps: Pandas datetime series
        use_sincos: If True, return (sin, cos) pair for unique encoding.
                   If False, return only sin (backward compatible).

    Returns:
        If use_sincos=True: (time_sin, time_cos) - 2D encoding
        If use_sincos=False: time_sin only - 1D encoding (original behavior)

    Note:
        Sin+Cos encoding provides unique representation for all times:
        - 00:00 (midnight) -> (sin=0.0, cos=1.0)
        - 06:00 (morning)  -> (sin=1.0, cos=0.0)
        - 12:00 (noon)     -> (sin=0.0, cos=-1.0)  [Different from midnight!]
        - 18:00 (evening)  -> (sin=-1.0, cos=0.0)
    """
    if not pd.api.types.is_datetime64_any_dtype(timestamps):
        timestamps = pd.to_datetime(timestamps)

    hour = timestamps.dt.hour
    minute = timestamps.dt.minute

    # Time as fraction of day [0, 1)
    time_fraction = (hour + minute / 60) / 24

    # Generate sin component
    time_sin = np.sin(2 * np.pi * time_fraction)

    if use_sincos:
        # Generate cos component for unique 2D encoding
        time_cos = np.cos(2 * np.pi * time_fraction)
        return time_sin.values, time_cos.values
    else:
        # Backward compatible: return only sin
        return time_sin.values

def add_padding_tso_patch(batch, device, seg_column='segment', max_seq_len=1440,
                          patch_size=1200, sampling_rate=20, padding_value=0.0,
                          use_sincos=False, scaler=None):
    """
    Prepare batch data for TSO prediction with patched raw sensor data.

    Each minute is represented as a patch of raw 20Hz samples.
    Uses data from load_data() which contains raw x, y, z accelerometer data.

    IMPORTANT - Scaling Strategy:
    - If using load_data_tso_patch(use_scaler=True): Data is already scaled in cache.
      Set scaler=None here to avoid double-scaling.
    - If using raw unscaled data: Pass scaler parameter to scale per-batch.

    Args:
        batch: DataFrame with columns from load_data:
               - x, y, z: raw accelerometer data at 20Hz (scaled or unscaled)
               - temperature: temperature sensor data
               - timestamp: datetime for each sample
               - non-wear, predictTSO: labels
               - segment: segment identifier
               - minute_id: minute index within segment (0-1439)
        device: torch device
        seg_column: column name for segment identifier
        max_seq_len: maximum sequence length in minutes (default: 1440 = 24h)
        patch_size: samples per minute patch (default: 1200 = 60 seconds * 20Hz)
        sampling_rate: sensor sampling rate in Hz (default: 20)
        padding_value: value to use for padding
        use_sincos: If True, use sin+cos time encoding (6 channels: x,y,z,temp,time_sin,time_cos).
                   If False, use sin only (5 channels: x,y,z,temp,time_sin). Default: False
        scaler: Optional StandardScaler to apply to x, y, z columns.
               - Use ONLY if data is NOT already scaled in load_data_tso_patch()
               - Set to None if load_data_tso_patch(use_scaler=True) was used
               - WARNING: Applying scaler to already-scaled data will corrupt results!

    Returns:
        pad_X: [batch_size, seq_len, patch_size, num_channels] - patched input features
               - If use_sincos=False (default): 5 channels [x, y, z, temperature, time_sin]
               - If use_sincos=True: 6 channels [x, y, z, temperature, time_sin, time_cos]
        pad_Y: [batch_size, seq_len] - minute-level class labels (0=other, 1=non-wear, 2=predictTSO)
        x_lens: [batch_size] - original sequence lengths in minutes

    GPU Memory Estimate:
        - Input shape (5ch): (batch_size, 1440, 1200, 5) - ~33 MB/sample, batch_16 = ~528 MB
        - Input shape (6ch): (batch_size, 1440, 1200, 6) - ~40 MB/sample, batch_16 = ~640 MB

    Examples:
        # Example 1: Data already scaled during loading (RECOMMENDED)
        df = load_data_tso_patch(folder, use_scaler=True)  # Scale once during caching
        pad_X, pad_Y, x_lens = add_padding_tso_patch(
            batch, device, use_sincos=True, scaler=None  # Don't scale again!
        )

        # Example 2: Scale per-batch (for raw unscaled data)
        df = load_data_tso_patch(folder, use_scaler=False)  # Raw data
        scaler = joblib.load('scaler.joblib')
        pad_X, pad_Y, x_lens = add_padding_tso_patch(
            batch, device, use_sincos=True, scaler=scaler  # Scale per-batch
        )
    """
    num_channels = 6 if use_sincos else 5

    X_sequences = []  # Will store [seq_len_minutes, patch_size, channels]
    Y_sequences = []  # Will store [seq_len_minutes] minute-level labels
    x_lens = []  # Length in minutes

    if max_seq_len is not None:
        # Template for padding to max_seq_len minutes
        max_seq_template_X = torch.ones(max_seq_len, patch_size, num_channels, device=device) * padding_value
        max_seq_template_Y = torch.full((max_seq_len,), -100, dtype=torch.long, device=device)
        X_sequences.append(max_seq_template_X)
        Y_sequences.append(max_seq_template_Y)

    for _, seg_metadata in batch.groupby(seg_column, sort=False):
        # Load raw data on-demand from cache_file
        if 'cache_file' in seg_metadata.columns:
            # Lazy loading mode: load from cache file
            cache_file = seg_metadata['cache_file'].iloc[0]
            seg_data = pd.read_parquet(cache_file)
        elif 'file_path' in seg_metadata.columns:
            # Legacy: support old file_path column name
            file_path = seg_metadata['file_path'].iloc[0]
            seg_data = pd.read_parquet(file_path)
        else:
            # Direct mode: data already in batch
            seg_data = seg_metadata

        # Sort by timestamp to ensure temporal order
        seg_data = seg_data.sort_values('timestamp').reset_index(drop=True)

        # Apply scaler to x, y, z if provided (IMPORTANT: must match training scaler!)
        if scaler is not None:
            columns_to_scale = ['x', 'y', 'z']
            cols_present = [col for col in columns_to_scale if col in seg_data.columns]
            if cols_present:
                seg_data[cols_present] = scaler.transform(seg_data[cols_present])

        # Limit to max_seq_len minutes if specified
        if max_seq_len is not None and 'minute_id' in seg_data.columns:
            seg_data = seg_data[seg_data['minute_id'] < max_seq_len]

        # Group by minute_id to create patches
        if 'minute_id' not in seg_data.columns:
            # Create minute_id if not present
            seg_data['minute_id'] = seg_data.groupby(seg_column).cumcount() // patch_size

        minute_patches = []
        minute_labels = []

        for _, minute_data in seg_data.groupby('minute_id', sort=True):
            # Extract raw sensor data for this minute
            x_vals = minute_data['x'].values
            y_vals = minute_data['y'].values
            z_vals = minute_data['z'].values
            temp_vals = minute_data['temperature'].values if 'temperature' in minute_data.columns else np.zeros(len(minute_data))

            # Generate time encoding (sin or sin+cos)
            if 'timestamp' in minute_data.columns and len(minute_data) > 0:
                time_encoding = generate_time_cyclic(minute_data['timestamp'], use_sincos=use_sincos)
                if use_sincos:
                    time_sin, time_cos = time_encoding
                else:
                    time_sin = time_encoding
                    time_cos = None
            else:
                # Fallback if no timestamp
                time_sin = np.zeros(len(minute_data))
                time_cos = np.zeros(len(minute_data)) if use_sincos else None

            # Stack into patch based on time encoding method
            if use_sincos:
                # 6 channels: [x, y, z, temperature, time_sin, time_cos]
                patch = np.stack([x_vals, y_vals, z_vals, temp_vals, time_sin, time_cos], axis=1)
            else:
                # 5 channels: [x, y, z, temperature, time_sin]
                patch = np.stack([x_vals, y_vals, z_vals, temp_vals, time_sin], axis=1)

            # Pad or truncate to exactly patch_size samples
            if len(patch) < patch_size:
                # Pad with padding_value
                pad_length = patch_size - len(patch)
                padding = np.full((pad_length, num_channels), padding_value, dtype=np.float32)
                patch = np.vstack([patch, padding])
            elif len(patch) > patch_size:
                # Truncate to patch_size
                patch = patch[:patch_size]

            minute_patches.append(patch)

            # Determine minute-level label (priority: predictTSO > non-wear > other)
            if 'predictTSO' in minute_data.columns and minute_data['predictTSO'].any():
                label = 2  # predictTSO
            elif 'non-wear' in minute_data.columns and (minute_data['non-wear'] == 1).any():
                label = 1  # non-wear
            else:
                label = 0  # other

            minute_labels.append(label)

        # Convert to tensors
        if len(minute_patches) > 0:
            X_arr = np.stack(minute_patches, axis=0)  # [num_minutes, patch_size, 5]
            Y_arr = np.array(minute_labels, dtype=np.int64)  # [num_minutes]

            X_sequences.append(torch.tensor(X_arr, dtype=torch.float32, device=device))
            Y_sequences.append(torch.tensor(Y_arr, dtype=torch.long, device=device))
            x_lens.append(len(minute_patches))

    # Pad sequences to max_seq_len
    # Note: pad_sequence expects list of [seq_len, ...] tensors
    pad_X = torch.nn.utils.rnn.pad_sequence(X_sequences, batch_first=True, padding_value=padding_value)
    pad_Y = torch.nn.utils.rnn.pad_sequence(Y_sequences, batch_first=True, padding_value=-100)

    if max_seq_len is not None:
        # Discard the max seq template
        pad_X = pad_X[1:, :, :, :]
        pad_Y = pad_Y[1:, :]

    return pad_X, pad_Y, torch.tensor(x_lens, dtype=torch.long, device=device)


def random_patch_masking(segments, masking_ratio=0.3, patch_size=10):
    segment_length, num_features = segments.shape
    segments_masked = segments.copy()  # Create a copy to apply masking

    # Calculate the total number of elements to mask based on the masking ratio
    total_elements_to_mask = int(masking_ratio * segment_length)
    
    patch_size= min(patch_size,total_elements_to_mask)
    
    masked_elements = 0
    while masked_elements < total_elements_to_mask:
        # Select random starting index for the patch
        start_index = np.random.randint(0, segment_length - patch_size + 1)
        
        # Apply masking to the selected patch if not already masked
        if np.all(segments_masked[ start_index:start_index + patch_size] != 0):
            segments_masked[ start_index:start_index + patch_size] = 0  # Masking the patch
            masked_elements += patch_size

    return segments_masked

def smooth_binary_series(binary_series, window_size, threshold):
    """
    Smooths a binary time series by rejecting short events (less than a threshold length).

    Parameters:
    - binary_series: The input binary time series (list or numpy array of 0s and 1s)
    - window_size: The size of the sliding window for smoothing (e.g., 5)
    - threshold: The minimum length of an event to keep (events shorter than this are removed)

    Returns:
    - smoothed_series: The smoothed binary time series
    """
    smoothed_series = np.zeros_like(binary_series)

    # Iterate through the series with a sliding window
    for i in range(len(binary_series)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(binary_series), i + window_size // 2 + 1)

        window = binary_series[start_idx:end_idx]
        event_length = np.sum(window)

        # If event is longer than threshold, keep it, otherwise remove it
        if event_length >= threshold:
            smoothed_series[i] = 1 if np.sum(window) > 0 else 0
        else:
            smoothed_series[i] = 0

    return smoothed_series

def seq_to_seconds(batch_y,y_lens):
    """
    Changes a batch of sequences of single observations to seconds. Spliting each sequence seperatly then concatenating to avoid having 1sec windows running on multiple segments.

    Parameters:
    - batch_y: batch sequence of predicted or ground truth labels
    - y_lens: An array containing original length of each sequence 

    Returns:
    - s:Second level sequence
    """
    i=0
    s=np.array([],dtype=int)
    for l in y_lens:
        s=np.concatenate((s,block_reduce(batch_y[i:i+l], block_size=20, func=np.any, cval=0)))
    return s

def remove_padding(batch_y,y_lens):
    """
    Remove padding from a padded batch of sequence.

    Parameters:
    - batch_y: batch sequence of predicted or ground truth labels
    - y_lens: An array containing original length of each sequence 
    - batch_size: Size of the batch

    Returns:
    - Sequences without padding concatenated in 1D array
    """
    o=batch_y.reshape(-1,max(y_lens))
    s=np.array([],dtype=int)
    for i in range(o.shape[0]):
        s=np.concatenate((s,o[i][:y_lens[i]]))
    return s

class GCELoss2(nn.Module):
    def __init__(self, num_classes=2, q=0.9):
        super(GCELoss2, self).__init__()
        self.q = q
        self.num_classes = num_classes
        self.eps = 1e-9  # Define eps here

    def forward(self, pred, labels):
        # Convert logits to probabilities
        pred = torch.sigmoid(pred)  # Use sigmoid for binary classification
        pred = torch.stack([1 - pred, pred], dim=1)  # Shape [batch_size, 2]
        pred = torch.clamp(pred, min=self.eps, max=1.0)
        #label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = pred
        loss = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return loss

def measure_loss(outputs,batch_labels,pos_weight):
    """
    Measure loss without reduction and mask the padded values to prevent loss calculation from padded values.
    """
    #loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight,reduction='none')(outputs, batch_labels.view(-1).float())
    criterion = GCELoss2(q=0.9)
    loss = criterion(outputs, batch_labels.view(-1).float())
    loss_mask = batch_labels != 2
    loss_masked = loss.where(loss_mask, torch.tensor(0.0))
    loss= loss_masked.sum() / loss_mask.sum()  # tensor(0.3802)
    return loss

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
    
def measure_loss_pretrain(outputs1,outputs2,outputs3,label1,label2,label3,wl1=1.0, wl2=1, wl3=0.1, pos_weight_l1 = False,pos_weight_l2 = False):
    mask = (label1 != -999) #detect the padded values    
    
    label1,label2,label3 = label1[mask],label2[mask],label3[mask]
    
    loss1 = F.mse_loss(outputs1,label1.float()) 
    loss2 = F.mse_loss(outputs2,label2.float())
    loss3 = F.mse_loss(outputs3,label3.float())
    
    
    return loss1+loss2+loss3,loss1,loss2,loss3,label1,label2,label3


def create_mask(original_lengths, max_length, batch_size, device):
    # same as DL_models
    mask = torch.arange(max_length,device=device).unsqueeze(0).expand(batch_size, -1)
    mask = (mask < original_lengths.unsqueeze(1)).long()
    return mask


def split_and_pad(x, x_lengths, max_seq_len, pad_value=0):
    """
    Split the x into sub-sequences based on lengths and pad them to max_seq_len.
    
    Args:
        x (torch.Tensor): Input vector of shape (N,).
        x_lengths (list): List of sub-sequence lengths [n1, n2, ...].
        max_seq_len (int): The fixed length to pad each sub-sequence.
        pad_value (int, optional): Value to pad shorter sub-sequences. Defaults to 0.
    
    Returns:
        torch.Tensor: Padded sub-sequences matrix of shape (len(x_lengths), max_seq_len).
    """

    start_idx = 0  # Keep track of the slicing index    
    xs = torch.split(x, x_lengths)
    pad_xs = []
    for cur_x in xs:
        pad_cur_x = torch.cat([cur_x, torch.full((max_seq_len - len(cur_x),), pad_value).to(x.device)])
        pad_xs.append(pad_cur_x)
    return torch.stack(pad_xs)


def measure_loss_multitask(outputs1,
                           outputs2,
                           outputs3,
                           label1,
                           label2,
                           label3,
                           wl1=1.0, 
                           wl2=1, 
                           wl3=0.1, 
                           contrastive_embedding=None,
                           mixup_label1=None,
                           mixup_label3=None,
                           mixup_lambda=1.0,
                           pos_weight_l1=False,
                           pos_weight_l2=False, 
                           ):
    """
    Measure loss without reduction and mask the padded values to prevent loss calculation from padded values.
    """
    weight1, weight2 = None, None
    
    if pos_weight_l1:
        weight1 = torch.ones([len(label1)],device=outputs1.device) * 3

    if mixup_label1 is not None and mixup_lambda < 1.0:
        loss1_orig = nn.BCEWithLogitsLoss(pos_weight=weight1, reduction='none')(outputs1, label1.float())
        loss1_mixup = nn.BCEWithLogitsLoss(pos_weight=weight1, reduction='none')(outputs1, mixup_label1.float())
        loss1 = (mixup_lambda * loss1_orig + (1 - mixup_lambda) * loss1_mixup).mean()
    else:
        loss1 = nn.BCEWithLogitsLoss(pos_weight=weight1)(outputs1, label1.float())

    if contrastive_embedding is not None:
        supcon_loss = SupConLoss(temperature=0.07)(contrastive_embedding, label1.float())
        loss1 = 0.5*loss1 + 0.5 * supcon_loss
    
    if not pos_weight_l2:
        weight2 = torch.ones([len(label2)],device=outputs1.device)*10
    loss2 = nn.BCEWithLogitsLoss(pos_weight=weight2)(outputs2,label2.view(-1).float())

    if mixup_label1 is not None and mixup_lambda < 1.0:
        loss3_orig = F.mse_loss(outputs3, label3.float())
        loss3_mixup = F.mse_loss(outputs3, mixup_label3.float())
        loss3 = (mixup_lambda * loss3_orig + (1 - mixup_lambda) * loss3_mixup).mean()
    else:
        loss3 = F.mse_loss(outputs3, label3.float())
    return wl1*loss1+wl2*loss2+wl3*loss3, loss1, loss2, loss3


def measure_loss_reg(outputs3, label1, label3, wl1=1.0, wl3=1.0, pos_thres=1.0):
    # measure classification and regression losses. The classification labels are generated from regression outputs. 
    loss3 = F.mse_loss(outputs3, label3.float())
    # if outputs3 > pos_thres, treate it as positive
    outputs1 = (outputs3 > pos_thres).float()
    loss1 = nn.BCEWithLogitsLoss()(outputs1, label1.float())
    # loss1_orig = nn.BCEWithLogitsLoss(reduction='none')(outputs1, label1.float())
    # loss1_mixup = nn.BCEWithLogitsLoss(reduction='none')(outputs1, mixup_label1.float())
    # loss1 = (mixup_lambda * loss1_orig + (1 - mixup_lambda) * loss1_mixup).mean()
    loss = wl1 * loss1 + loss3 * wl3
    return loss, loss1, loss3


def measure_loss_cls(outputs1,label1,pos_weight_l1=False):
    """
    Measure loss without reduction and mask the padded values to prevent loss calculation from padded values.
    """
    weight1 = torch.ones([len(label1)],device=outputs1.device)*3

    if pos_weight_l1:
        loss1 = nn.BCEWithLogitsLoss(pos_weight=weight1)(outputs1,label1.float()) #pos_weight=weight1
    else:
        loss1 = nn.BCEWithLogitsLoss()(outputs1,label1.float()) #pos_weight=weight1

    return loss1


def measure_loss_tso(outputs, labels, x_lengths):
    """
    Measure loss for TSO prediction. Supports binary (output_channels=1) and
    3-class (output_channels=3) modes based on output shape.

    Binary mode: maps labels to TSO (class 2) -> 1, other/non-wear -> 0, uses BCE.
    3-class mode: uses cross-entropy with ignore_index=-100 for padding.

    Args:
        outputs: [batch_size, seq_len, C] - model predictions (logits), C=1 or 3
        labels: [batch_size, seq_len] - ground truth class indices (0/1/2, -100=padding)
        x_lengths: [batch_size] - original sequence lengths before padding

    Returns:
        total_loss: scalar tensor
    """
    batch_size, seq_len, num_classes = outputs.size()

    if num_classes == 1:
        # Binary mode: TSO (2) -> 1, everything else -> 0
        ignore_mask = (labels == -100)
        binary_labels = (labels == 2).float()
        outputs_flat = outputs.reshape(-1)
        labels_flat = binary_labels.reshape(-1)
        ignore_flat = ignore_mask.reshape(-1)
        valid_labels = labels_flat[~ignore_flat]
        # Compute pos_weight from this batch to counteract class imbalance.
        # pos_weight = #negatives / #positives (clamped to [1, 50] for stability).
        n_pos = valid_labels.sum().clamp(min=1)
        n_neg = (1 - valid_labels).sum().clamp(min=1)
        pos_weight = (n_neg / n_pos).clamp(max=50.0)
        loss_per_token = F.binary_cross_entropy_with_logits(
            outputs_flat, labels_flat, reduction='none',
            pos_weight=pos_weight.expand_as(outputs_flat)
        )
        loss_per_token = loss_per_token * (~ignore_flat).float()
        return loss_per_token.sum() / (~ignore_flat).float().sum().clamp(min=1)
    else:
        # 3-class mode: cross-entropy, padding positions (label=-100) are ignored
        outputs_flat = outputs.reshape(-1, num_classes)
        labels_flat = labels.reshape(-1)
        return F.cross_entropy(outputs_flat, labels_flat, ignore_index=-100)


def tso_continuity_loss(outputs, x_lengths, alpha=0.1):
    """
    Continuity regularization for TSO predictions to encourage single, continuous TSO period.

    Penalizes fragmented TSO predictions by measuring the total variation (switches)
    in TSO probability over time. Encourages smooth, continuous TSO segments.

    OPTIMIZED: Fully vectorized implementation for GPU efficiency.

    Optimizations:
    1. Fully vectorized - no Python loops
    2. Uses masking instead of slicing for valid sequences
    3. Single GPU kernel for all operations
    4. Eliminates intermediate list allocations

    Args:
        outputs: [batch_size, seq_len, 3] - model predictions (logits)
        x_lengths: [batch_size] - valid sequence lengths
        alpha: Weight for continuity loss (default: 0.1)

    Returns:
        continuity_loss: Scalar tensor - penalty for fragmented TSO predictions

    Intuition:
        - If TSO probability smoothly goes 0 -> 0.8 -> 1.0 -> 0.8 -> 0: Low loss (continuous)
        - If TSO probability jumps 0 -> 1 -> 0 -> 1 -> 0: High loss (fragmented)
    """
    batch_size, seq_len, _ = outputs.shape
    device = outputs.device

    # Get TSO probabilities
    num_classes = outputs.shape[-1]
    if num_classes == 1:
        tso_probs = torch.sigmoid(outputs[:, :, 0])   # [batch_size, seq_len]
    else:
        tso_probs = torch.softmax(outputs, dim=-1)[:, :, 2]  # [batch_size, seq_len]

    # Create mask for valid positions: [batch_size, seq_len]
    positions = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
    valid_mask = positions < x_lengths.unsqueeze(1)  # [batch_size, seq_len]

    # Create mask for valid transitions (exclude last timestep and invalid regions)
    # Transition from t to t+1 is valid if both t and t+1 are valid
    valid_transition_mask = valid_mask[:, :-1] & valid_mask[:, 1:]  # [batch_size, seq_len-1]

    # Calculate differences between consecutive timesteps (vectorized)
    diffs = torch.abs(tso_probs[:, 1:] - tso_probs[:, :-1])  # [batch_size, seq_len-1]

    # Apply mask and sum per sequence
    masked_diffs = diffs * valid_transition_mask  # Zero out invalid transitions
    switches_per_seq = masked_diffs.sum(dim=1)  # [batch_size]

    # Normalize by valid transition count (valid_len - 1)
    # Clamp to avoid division by zero for sequences with length <= 1
    valid_transition_counts = valid_transition_mask.sum(dim=1).clamp(min=1)  # [batch_size]
    normalized_switches = switches_per_seq / valid_transition_counts  # [batch_size]

    # Return mean across batch
    return alpha * normalized_switches.mean()


def measure_loss_tso_with_continuity(outputs, labels, x_lengths, continuity_weight=0.1):
    """
    TSO loss with continuity regularization for single-period prediction.

    Combines standard cross-entropy loss with continuity regularization to encourage
    the model to predict single, continuous TSO periods rather than fragmented periods.

    Args:
        outputs: [batch_size, seq_len, 3] - model predictions (logits)
        labels: [batch_size, seq_len] - ground truth labels
        x_lengths: [batch_size] - valid sequence lengths
        continuity_weight: Weight for continuity loss (default: 0.1)
                          Higher = stronger enforcement of continuity
                          0.0 = no continuity regularization (standard CE only)

    Returns:
        total_loss: Combined classification + continuity loss
        class_loss: Standard cross-entropy loss
        cont_loss: Continuity regularization loss
    """
    # Standard classification loss
    class_loss = measure_loss_tso(outputs, labels, x_lengths)

    # Continuity regularization (encourage smooth, continuous TSO predictions)
    if continuity_weight > 0:
        cont_loss = tso_continuity_loss(outputs, x_lengths, alpha=1.0)
        total_loss = class_loss + continuity_weight * cont_loss
    else:
        cont_loss = torch.tensor(0.0, device=outputs.device)
        total_loss = class_loss

    return total_loss, class_loss, cont_loss


def enforce_single_tso_period(predictions, min_gap_minutes=30, min_duration_minutes=10):
    """
    Post-process predictions to enforce single, continuous TSO period.

    Finds all TSO segments, merges close ones, and keeps only the longest period.
    This guarantees that the final prediction has at most one TSO period per night.

    Args:
        predictions: [seq_len] or [batch, seq_len] - predicted class labels (0, 1, 2)
                    0 = other, 1 = non-wear, 2 = predictTSO
        min_gap_minutes: Minimum gap to consider as separate TSO periods (default: 30)
                        Segments closer than this will be merged
        min_duration_minutes: Minimum duration to keep a TSO period (default: 10)
                             Shorter segments are filtered out as noise

    Returns:
        processed_predictions: Same shape as input, with only longest TSO period retained

    Example:
        Input:  [0, 0, 2, 2, 0, 0, 2, 2, 2, 0, ...]  (multiple TSO periods)
        Output: [0, 0, 0, 0, 0, 0, 2, 2, 2, 0, ...]  (only longest kept)
    """
    import numpy as np

    # Handle batch dimension
    if predictions.ndim == 2:
        # Batch processing
        processed_batch = []
        for i in range(predictions.shape[0]):
            processed = enforce_single_tso_period(
                predictions[i],
                min_gap_minutes=min_gap_minutes,
                min_duration_minutes=min_duration_minutes
            )
            processed_batch.append(processed)
        return np.stack(processed_batch)

    # Single sequence processing
    predictions = np.array(predictions)  # Ensure numpy array
    tso_class = 1 if predictions.max() <= 1 else 2  # binary vs 3-class
    tso_mask = (predictions == tso_class)

    if not tso_mask.any():
        return predictions  # No TSO predicted, return as-is

    # Find all continuous TSO segments
    segments = []
    in_segment = False
    start = 0

    for i in range(len(tso_mask)):
        if tso_mask[i] and not in_segment:
            # Start of new TSO segment
            start = i
            in_segment = True
        elif not tso_mask[i] and in_segment:
            # End of TSO segment
            segments.append((start, i - 1))
            in_segment = False

    # Handle case where TSO extends to end of sequence
    if in_segment:
        segments.append((start, len(tso_mask) - 1))

    if len(segments) == 0:
        return predictions  # No valid segments

    # Filter out very short segments (likely noise)
    segments = [(s, e) for s, e in segments if (e - s + 1) >= min_duration_minutes]

    if len(segments) == 0:
        # All segments were too short - remove all TSO predictions
        processed_predictions = predictions.copy()
        processed_predictions[tso_mask] = 0  # Change to "other"
        return processed_predictions

    # Merge segments that are close together (within min_gap_minutes)
    merged_segments = []
    for start, end in segments:
        if merged_segments and (start - merged_segments[-1][1] - 1) <= min_gap_minutes:
            # Gap is small, merge with previous segment
            merged_segments[-1] = (merged_segments[-1][0], end)
        else:
            # Gap is large or first segment
            merged_segments.append((start, end))

    # Keep only the longest merged segment
    if len(merged_segments) > 1:
        # Find longest segment
        longest_segment = max(merged_segments, key=lambda x: x[1] - x[0])

        # Create processed predictions with only longest segment
        processed_predictions = predictions.copy()

        # Remove all TSO predictions first
        processed_predictions[tso_mask] = 0  # Change to "other"

        # Restore only the longest segment
        processed_predictions[longest_segment[0]:longest_segment[1]+1] = tso_class

        return processed_predictions
    else:
        # Only one segment, return as-is
        return predictions


def batch_enforce_single_tso(predictions, x_lengths, min_gap_minutes=30, min_duration_minutes=10):
    """
    Batch version of enforce_single_tso_period with sequence length handling.

    Args:
        predictions: [batch_size, seq_len] - predicted class labels
        x_lengths: [batch_size] - valid sequence lengths
        min_gap_minutes: Minimum gap to merge TSO segments
        min_duration_minutes: Minimum TSO duration to keep

    Returns:
        processed_predictions: [batch_size, seq_len] - with single TSO period per sample
    """
    import numpy as np

    predictions_np = predictions.cpu().numpy() if torch.is_tensor(predictions) else np.array(predictions)
    x_lengths_np = x_lengths.cpu().numpy() if torch.is_tensor(x_lengths) else np.array(x_lengths)

    processed_batch = []

    for i in range(len(predictions_np)):
        valid_len = int(x_lengths_np[i])

        # Process only valid part
        pred_seq = predictions_np[i, :valid_len].copy()
        processed_seq = enforce_single_tso_period(
            pred_seq,
            min_gap_minutes=min_gap_minutes,
            min_duration_minutes=min_duration_minutes
        )

        # Reconstruct full sequence (with padding)
        full_seq = predictions_np[i].copy()
        full_seq[:valid_len] = processed_seq

        processed_batch.append(full_seq)

    result = np.stack(processed_batch)

    # Convert back to tensor if input was tensor
    if torch.is_tensor(predictions):
        return torch.from_numpy(result).to(predictions.device)
    else:
        return result


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss

class SupConLossV2(nn.Module):
    """Supervised Contrastive Learning loss"""
    # No multi-view requirement. Contrast between different samples in the batch
    # Fixed implementation based on https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features, labels):
        """
        Args:
            features: hidden vector of shape [batch_size, embed_dim]
            labels: ground truth of shape [batch_size]
        """
        batch_size = features.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
            
        labels = labels.contiguous().view(-1, 1)
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Mask of positive pairs (same class, excluding self)
        mask = torch.eq(labels, labels.T).float()
        
        # Compute similarity matrix
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Create mask to exclude self-contrasts (diagonal)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size, device=features.device).view(-1, 1),
            0
        )
        
        # Mask out self-contrast cases  
        mask = mask * logits_mask
        
        # Compute log_prob: log(exp(sim_i_j) / sum_k(exp(sim_i_k))) where k != i
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # Compute mean of log-likelihood over positive pairs
        # Only consider samples that have positive pairs
        mask_sum = mask.sum(1)
        valid_samples = mask_sum > 0
        
        if valid_samples.sum() == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        mean_log_prob_pos = (mask * log_prob).sum(1)[valid_samples] / mask_sum[valid_samples]
        
        # Loss: negative mean of log-likelihood
        loss = -mean_log_prob_pos.mean()
        return loss

def measure_loss_multitask_with_padding(outputs1,
                           outputs2,
                           outputs3,
                           label1,
                           label2,
                           label3,
                           wl1=1.0, 
                           wl2=1, 
                           wl3=0.1, 
                           contrastive_embedding=None,
                           mixup_label1=None,
                           mixup_label3=None,
                           mixup_lambda=1.0,
                           padding_position="tail",
                           pos_weight_l1=False,
                           pos_weight_l2=False,
                           x_lengths=[],
                           seq_start_idx=[],
                           average_mask=False,
                           average_window_size=20,
                           ignore_padding_in_mask_loss=True):
    """
    Measure loss without reduction and mask the padded values to prevent loss calculation from padded values.
    
    Args:
        outputs1, outputs2, outputs3: Model outputs for the three tasks
        label1, label2, label3: Ground truth labels
        wl1, wl2, wl3: Loss weights for the three tasks
        padding_position: "tail" or "random" - specifies where padding was applied
        pos_weight_l1, pos_weight_l2: Whether to use positive class weighting
        x_lengths: Original sequence lengths (before padding)
        seq_start_idx: Start indices for sequences (when using random padding)
        average_mask: Whether to apply window averaging to downsample masks
        average_window_size: Size of window for averaging
        ignore_padding_in_loss: Whether to ignore padding areas when calculating loss2
    """
    weight1, weight2 = None, None
    if pos_weight_l1:
        weight1 = torch.ones([len(label1)], device=outputs1.device) * 2

    # Task 1: Binary classification loss
    if mixup_label1 is not None and mixup_lambda < 1.0:
        loss1_orig = nn.BCEWithLogitsLoss(pos_weight=weight1, reduction='none')(outputs1, label1.float())
        loss1_mixup = nn.BCEWithLogitsLoss(pos_weight=weight1, reduction='none')(outputs1, mixup_label1.float())
        loss1 = (mixup_lambda * loss1_orig + (1 - mixup_lambda) * loss1_mixup).mean()
    else:
        loss1 = nn.BCEWithLogitsLoss(pos_weight=weight1)(outputs1, label1.float())

    if contrastive_embedding is not None:
        supcon_loss = SupConLossV2(temperature=0.07)(contrastive_embedding, label1.float())
        # loss1 += supcon_loss
        # loss1 = supcon_loss
        loss1 = 0.5 * loss1 + 0.5 * supcon_loss
        

    # Handle mask averaging if enabled
    if average_mask:
        B, L = label2.shape
        target_length = L // average_window_size

        # Use F.interpolate with nearest mode for binary labels (better than averaging + threshold)
        label2_float = label2.float().unsqueeze(1)  # Add channel dimension: [B, 1, L]
        label2_interpolated = F.interpolate(label2_float, size=target_length, mode='nearest')
        label2 = label2_interpolated.squeeze(1).long()  # Remove channel dim and convert back to long

        # CRITICAL: Recalculate x_lengths and seq_start_idx for the new target_length
        # This is essential for proper mask calculation after averaging
        if len(x_lengths) > 0:
            # Simply divide by average_window_size
            x_lengths = [max(1, length // average_window_size) for length in x_lengths]

            # Scale seq_start_idx by dividing by average_window_size
            if len(seq_start_idx) > 0:
                seq_start_idx = [idx // average_window_size for idx in seq_start_idx]
    # Task 2: Binary sequence classification loss
    if pos_weight_l2:
        weight2 = torch.ones([label2.shape[1]], device=outputs2.device) * 10
    
    # Create padding mask if needed
    if ignore_padding_in_mask_loss:
        B, L = label2.shape
        non_padding = torch.zeros((B, L), dtype=torch.bool, device=outputs2.device)
        
        # Fill mask based on padding position
        if padding_position == "tail":
            # For tail padding, only the first x_lengths elements are valid
            for i in range(len(x_lengths)):
                non_padding[i, :x_lengths[i]] = True
        else:  # "random" padding
            # For random padding, valid elements depend on start indices
            for i in range(len(x_lengths)):
                non_padding[i, seq_start_idx[i]:seq_start_idx[i]+x_lengths[i]] = True
        
        # Calculate loss with reduction='none' to apply mask
        element_losses = nn.BCEWithLogitsLoss(pos_weight=weight2, reduction='none')(outputs2, label2.float())
        
        # Apply mask and calculate mean over non-padded elements only
        loss2 = (element_losses * non_padding.float()).sum() / non_padding.sum()
    else:
        # Standard loss calculation without ignoring padding
        loss2 = nn.BCEWithLogitsLoss(pos_weight=weight2)(outputs2, label2.float())
    
    # Task 3: Regression loss for scratch duration
    if mixup_label1 is not None and mixup_lambda < 1.0:
        loss3_orig = F.mse_loss(outputs3, label3.float())
        loss3_mixup = F.mse_loss(outputs3, mixup_label3.float())
        loss3 = (mixup_lambda * loss3_orig + (1 - mixup_lambda) * loss3_mixup).mean()
    else:
        loss3 = F.mse_loss(outputs3, label3.float())

    # Extract unpadded areas for evaluation metrics
    if ignore_padding_in_mask_loss:
        if padding_position == "tail":
            # For tail padding, just take the first x_lengths elements
            unpad_outputs2 = [
                outputs2[i, :x_lengths[i]]
                for i in range(len(x_lengths))
            ]
            unpad_label2 = [
                label2[i, :x_lengths[i]]
                for i in range(len(x_lengths))
            ]
        else:  # "random" padding
            # For random padding, use the sequence start indices
            unpad_outputs2 = [
                outputs2[i, seq_start_idx[i]:seq_start_idx[i]+x_lengths[i]]
                for i in range(len(x_lengths))
            ]
            unpad_label2 = [
                label2[i, seq_start_idx[i]:seq_start_idx[i]+x_lengths[i]]
                for i in range(len(x_lengths))
            ]

        # Concatenate the unpadded sequences
        unpad_outputs2 = torch.cat(unpad_outputs2, dim=0)
        unpad_label2 = torch.cat(unpad_label2, dim=0)
    else:
        # Fallback: if no x_lengths provided, just flatten everything
        unpad_outputs2 = outputs2.view(-1)
        unpad_label2 = label2.view(-1)
    
    # Calculate weighted total loss
    total_loss = wl1*loss1 + wl2*loss2 + wl3*loss3
    # total_loss = loss1
    # if torch.isnan(total_loss):
    #     print(f"loss1: {loss1.item()}, loss2: {loss2.item()}, loss3: {loss3.item()}")
    
    return total_loss, loss1, loss2, loss3, unpad_outputs2, unpad_label2


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


# def load_hf_dataset_tso_patch(dataset_path, include_subject_filter=None, remove_subject_filter=None,
#                               keep_in_memory=False):
#     """
#     Load TSO patch dataset from HuggingFace dataset format.

#     Args:
#         dataset_path: Path to saved HuggingFace dataset
#         include_subject_filter: Subject ID to include (optional)
#         remove_subject_filter: List of subject IDs to exclude (optional)
#         keep_in_memory: If True, load entire dataset into RAM for faster access (like pandas)
#                        If False, use memory-mapping (slower but uses less RAM)

#     Returns:
#         HuggingFace Dataset object

#     Example:
#         # Load dataset (memory efficient - uses memory mapping)
#         dataset = load_hf_dataset_tso_patch("/path/to/hf_dataset")

#         # Access individual samples (lazy loading)
#         sample = dataset[0]
#         print(sample.keys())  # dict_keys(['segment', 'subject', 'x', 'y', 'z', ...])

#         # Filter dataset
#         filtered = dataset.filter(lambda x: x['subject'] == 'US10008015')

#         # Use in training loop
#         for i in range(len(dataset)):
#             sample = dataset[i]
#             x_data = np.array(sample['x'])  # Loaded only when accessed
#             # ... process sample
#     """
#     try:
#         from datasets import load_from_disk
#     except ImportError:
#         raise ImportError("Please install datasets: pip install datasets")

#     print("="*80)
#     print(f"Loading HuggingFace Dataset from: {dataset_path}")

#     start_time = datetime.now()

#     # Load dataset
#     dataset = load_from_disk(dataset_path, keep_in_memory=keep_in_memory)

#     print(f"  Total segments: {len(dataset)}")
#     print(f"  Features: {list(dataset.features.keys())}")

#     # Apply filters if provided
#     if remove_subject_filter is not None:
#         remove_list = [str(s) for s in remove_subject_filter] if isinstance(remove_subject_filter, list) else [str(remove_subject_filter)]
#         print(f"Filtering out subjects: {remove_list}")
#         dataset = dataset.filter(lambda x: x['subject'] not in remove_list)
#         print(f"  After removal: {len(dataset)} segments")

#     if include_subject_filter is not None:
#         include_str = str(include_subject_filter)
#         print(f"Including only subject: {include_str}")
#         dataset = dataset.filter(lambda x: x['subject'] == include_str)
#         print(f"  After filtering: {len(dataset)} segments")

#     print(f"{'='*80}\n")

#     return dataset


def get_folder_size(folder_path):
    """Get total size of folder in GB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024**3)  # Convert to GB


def batch_generator_hf(dataset, batch_size=16, shuffle=True):
    """
    Batch generator for HuggingFace datasets (memory-efficient).

    This generator yields batches of segment indices from the dataset.
    Each batch contains indices that can be used to access the dataset.

    Args:
        dataset: HuggingFace Dataset object (from load_hf_dataset_tso_patch)
        batch_size: Number of segments per batch
        shuffle: Whether to shuffle segments

    Yields:
        list: List of indices for the current batch

    Example:
        from Helpers.DL_helpers import load_hf_dataset_tso_patch, batch_generator_hf

        # Load dataset (memory-mapped)
        dataset = load_hf_dataset_tso_patch("/path/to/hf_dataset")

        # Create batch generator
        for batch_indices in batch_generator_hf(dataset, batch_size=16, shuffle=True):
            # Get samples for this batch (loaded lazily)
            batch_samples = [dataset[i] for i in batch_indices]

            # Or access dataset with select for efficiency
            batch_data = dataset.select(batch_indices)

            # Process batch...
    """

    # Get total number of segments
    n_samples = len(dataset)

    # Create array of indices
    indices = np.arange(n_samples)

    # Shuffle if requested
    if shuffle:
        np.random.shuffle(indices)

    # Yield batches of indices
    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i+batch_size].tolist()
        yield batch_indices


def get_nb_steps_hf(dataset, batch_size=16):
    """
    Calculate number of batches for HuggingFace dataset.

    Args:
        dataset: HuggingFace Dataset object
        batch_size: Number of segments per batch

    Returns:
        int: Number of batches

    Example:
        dataset = load_hf_dataset_tso_patch("/path/to/hf_dataset")
        nb_steps = get_nb_steps_hf(dataset, batch_size=16)
        print(f"Total batches per epoch: {nb_steps}")
    """
    n_samples = len(dataset)
    nb_steps = int(np.ceil(n_samples / batch_size))

    return nb_steps


def add_padding_tso_patch_hf(batch_samples, device, max_seq_len=1440,
                             patch_size=1200, sampling_rate=20, padding_value=0.0):
    """
    Prepare batch data for TSO prediction from HuggingFace dataset samples.

    Works with HuggingFace dataset samples instead of pandas DataFrames.
    Each sample contains raw sensor data as lists, which are processed into minute-level patches.

    Processing details (matching add_padding_tso_patch):
    - Uses pre-computed time_cyclic from load_parquet_as_hf_dataset
    - Derives "other" label from predictTSO and non-wear (not loaded from data)
    - Label priority: predictTSO > non-wear > other

    Args:
        batch_samples: List of HF dataset samples, where each sample is a dict with keys:
                      - 'segment': segment identifier (str)
                      - 'x', 'y', 'z': lists of raw accelerometer data at 20Hz
                      - 'time_cyclic': list of pre-computed time_cyclic values
                      - 'temperature': list of temperature values (optional)
                      - 'predictTSO': list of binary labels (0/1)
                      - 'non-wear': list of binary labels (0/1)
                      - 'num_samples': number of samples in the segment
        device: torch device
        max_seq_len: maximum sequence length in minutes (default: 1440 = 24h)
        patch_size: samples per minute patch (default: 1200 = 60 seconds * 20Hz)
        sampling_rate: sensor sampling rate in Hz (default: 20)
        padding_value: value to use for padding

    Returns:
        pad_X: [batch_size, seq_len, patch_size, 5] - patched input features
               5 channels: [x, y, z, temperature, time_cyclic]
        pad_Y: [batch_size, seq_len] - minute-level class labels (0=other, 1=non-wear, 2=predictTSO)
        x_lens: [batch_size] - original sequence lengths in minutes

    Example:
        from Helpers.DL_helpers import load_parquet_as_hf_dataset, batch_generator_hf, add_padding_tso_patch_hf

        dataset = load_parquet_as_hf_dataset("/path/to/parquet_folder")
        for batch_indices in batch_generator_hf(dataset, batch_size=16):
            batch_samples = [dataset[i] for i in batch_indices]
            pad_X, pad_Y, x_lens = add_padding_tso_patch_hf(
                batch_samples, device, max_seq_len=1440, patch_size=1200
            )
            # Use pad_X, pad_Y for model forward pass
    """
    num_channels = 5  # x, y, z, temperature, time_cyclic

    X_sequences = []
    Y_sequences = []
    x_lens = []

    for sample in batch_samples:
        # Extract data from HF sample (all are lists)
        x_vals = np.array(sample['x'], dtype=np.float32)
        y_vals = np.array(sample['y'], dtype=np.float32)
        z_vals = np.array(sample['z'], dtype=np.float32)
        temp_vals = np.array(sample['temperature'], dtype=np.float32) if 'temperature' in sample else np.zeros(len(sample['x']), dtype=np.float32)
        time_vals = np.array(sample['time_cyclic'], dtype=np.float32) if 'time_cyclic' in sample else np.zeros(len(sample['x']), dtype=np.float32)

        # Labels (lists of 0/1 for binary labels)
        predictTSO_vals = np.array(sample['predictTSO'], dtype=np.int32) if 'predictTSO' in sample else np.zeros(len(sample['x']), dtype=np.int32)
        nonwear_vals = np.array(sample['non-wear'], dtype=np.int32) if 'non-wear' in sample else np.zeros(len(sample['x']), dtype=np.int32)

        # Determine total number of samples
        num_samples = len(x_vals)

        # Calculate number of complete minutes (patches)
        num_minutes = num_samples // patch_size

        # Limit to max_seq_len if specified
        if max_seq_len is not None and num_minutes > max_seq_len:
            num_minutes = max_seq_len
            max_samples_truncate = num_minutes * patch_size
            x_vals = x_vals[:max_samples_truncate]
            y_vals = y_vals[:max_samples_truncate]
            z_vals = z_vals[:max_samples_truncate]
            temp_vals = temp_vals[:max_samples_truncate]
            time_vals = time_vals[:max_samples_truncate]
            predictTSO_vals = predictTSO_vals[:max_samples_truncate]
            nonwear_vals = nonwear_vals[:max_samples_truncate]

        minute_patches = []
        minute_labels = []

        # Process each minute patch
        for minute_idx in range(num_minutes):
            start_idx = minute_idx * patch_size
            end_idx = start_idx + patch_size

            # Extract patch data for this minute (use pre-computed time_cyclic)
            x_patch = x_vals[start_idx:end_idx]
            y_patch = y_vals[start_idx:end_idx]
            z_patch = z_vals[start_idx:end_idx]
            temp_patch = temp_vals[start_idx:end_idx]
            time_patch = time_vals[start_idx:end_idx]

            # Pad if needed (in case last minute is incomplete)
            actual_len = len(x_patch)
            if actual_len < patch_size:
                pad_len = patch_size - actual_len
                x_patch = np.pad(x_patch, (0, pad_len), constant_values=padding_value)
                y_patch = np.pad(y_patch, (0, pad_len), constant_values=padding_value)
                z_patch = np.pad(z_patch, (0, pad_len), constant_values=padding_value)
                temp_patch = np.pad(temp_patch, (0, pad_len), constant_values=padding_value)
                time_patch = np.pad(time_patch, (0, pad_len), constant_values=padding_value)

            # Stack channels: [patch_size, 5]
            patch = np.stack([x_patch, y_patch, z_patch, temp_patch, time_patch], axis=1)
            minute_patches.append(patch)

            # Determine minute-level label (like add_padding_tso_patch lines 2595-2601)
            # Priority: predictTSO > non-wear > other
            predictTSO_minute = predictTSO_vals[start_idx:min(end_idx, len(predictTSO_vals))]
            nonwear_minute = nonwear_vals[start_idx:min(end_idx, len(nonwear_vals))]

            # Derive "other" = not(predictTSO) and not(non-wear)
            if np.any(predictTSO_minute):
                label = 2  # predictTSO
            elif np.any(nonwear_minute):
                label = 1  # non-wear
            else:
                label = 0  # other (derived, not loaded)

            minute_labels.append(label)

        # Convert to tensors
        if len(minute_patches) > 0:
            seg_X = torch.FloatTensor(np.stack(minute_patches))  # [num_minutes, patch_size, 5]
            seg_Y = torch.LongTensor(minute_labels)  # [num_minutes]

            # Pad to max_seq_len if needed
            if max_seq_len is not None and len(seg_X) < max_seq_len:
                pad_len = max_seq_len - len(seg_X)
                pad_X_template = torch.ones(pad_len, patch_size, num_channels) * padding_value
                pad_Y_template = torch.full((pad_len,), -100, dtype=torch.long)
                seg_X = torch.cat([seg_X, pad_X_template], dim=0)
                seg_Y = torch.cat([seg_Y, pad_Y_template], dim=0)

            X_sequences.append(seg_X)
            Y_sequences.append(seg_Y)
            x_lens.append(num_minutes)
        else:
            # Empty segment - create padded template
            seg_X = torch.ones(max_seq_len, patch_size, num_channels) * padding_value
            seg_Y = torch.full((max_seq_len,), -100, dtype=torch.long)
            X_sequences.append(seg_X)
            Y_sequences.append(seg_Y)
            x_lens.append(0)

    # Stack all sequences
    pad_X = torch.stack(X_sequences).to(device)  # [batch_size, max_seq_len, patch_size, 5]
    pad_Y = torch.stack(Y_sequences).to(device)  # [batch_size, max_seq_len]
    x_lens = torch.LongTensor(x_lens).to(device)  # [batch_size]

    return pad_X, pad_Y, x_lens


def load_parquet_as_hf_dataset(parquet_folder, max_seq_length=86400,
                                keep_in_memory=False,
                                split_dataset=False, val_size=0.2, split_seed=42,
                                shuffle_split=False,
                                use_scaler=False, scaler_path=None,
                                additional_folder=None, balance_folders=True):
    """
    Load raw biobank parquet files directly as HuggingFace dataset (segment-level).

    Each parquet file = one segment/example in the dataset.
    All rows within a file are concatenated into lists for each column.

    Resulting dataset structure:
    - N samples (where N = number of parquet files)
    - Each sample contains:
      - 'x', 'y', 'z': lists of ~1.7M float values (24h @ 20Hz)
      - 'time_cyclic': list of pre-computed cyclic time encoding (sine wave)
      - 'temperature': list of temperature values (optional)
      - 'predictTSO', 'non-wear': lists of binary labels (0/1)
      - 'segment': string (segment identifier: {subject}_{wrist}_{instance_info}_{date})
      - 'subject', 'wrist', 'day': metadata strings
      - 'num_samples': int (number of timesteps)

    Note: 'other' label is derived later in add_padding_tso_patch_hf (not stored to save memory)

    Handles biobank filename convention for filtering:
    - Format: Processed_{eid}_{eid}_{instance}_{array_idx}_{field}_{date}.parquet.gzip
    - Example: Processed_1012030_1012030_90001_0_0_2015-08-31.parquet.gzip

    Key features:
    - Uses generator with incremental writes (avoids OOM)
    - Loads columns: x, y, z, temperature, predictTSO, non-wear
    - Generates time_cyclic from timestamp during loading (pre-computed, not per-batch)
    - Does NOT load: 'other' label (derived later in add_padding_tso_patch_hf)
    - Creates segment identifier matching load_data_tso_patch_biobank format: {subject}_{wrist}_{instance_info}_{date}
    - Supports loading from multiple folders with balanced sampling (AD + non-AD patients)
    - Supports random train/val splitting with optional stratification
    - Compatible with batch_generator_hf and add_padding_tso_patch_hf

    Args:
        parquet_folder: Path to folder containing parquet.gzip files (e.g., AD patients)
        max_seq_length: Maximum sequence length in seconds (default: 86400 = 24 hours)
        keep_in_memory: If True, load entire dataset into RAM for faster access
                       If False, use memory-mapped files (slower access, less RAM)
        split_dataset: If True, split dataset into train/val sets (default: False)
        val_size: Validation set size as fraction of total (default: 0.2 = 20%)
        split_seed: Random seed for train/val split (default: 42)
        shuffle_split: If True, shuffle before splitting. If False, preserve file order (default: False)
        use_scaler: If True, load and apply pretrained scaler to x, y, z columns (default: False)
        scaler_path: Path to pretrained scaler file (.joblib). Required if use_scaler=True
        additional_folder: Path to second folder (e.g., non-AD patients) for balanced sampling (optional)
        balance_folders: If True and additional_folder is provided, balance file counts between folders (default: True)

    Returns:
        If split_dataset=False: HuggingFace Dataset with row-level data
        If split_dataset=True: Dict with keys 'train' and 'val', each containing a Dataset

    Examples:
        # Load full dataset without splitting
        dataset = load_parquet_as_hf_dataset('/path/to/parquet_folder')

        # Load and split into train/val (80/20)
        datasets = load_parquet_as_hf_dataset('/path/to/parquet_folder', split_dataset=True)
        train_dataset = datasets['train']
        val_dataset = datasets['val']

        # Load from two folders (AD + non-AD) with balanced sampling
        dataset = load_parquet_as_hf_dataset(
            '/path/to/AD_patients',
            additional_folder='/path/to/non_AD_patients',
            balance_folders=True  # Use same number of files from each folder
        )

        # Load with scaler and balanced folders
        dataset = load_parquet_as_hf_dataset(
            '/path/to/AD_patients',
            additional_folder='/path/to/non_AD_patients',
            balance_folders=True,
            use_scaler=True,
            scaler_path='/path/to/scaler.joblib'
        )

    Note:
        This function loads segment-level data. Each sample = one parquet file with all
        rows concatenated into lists. Compatible with the existing training pipeline.
    """
    from datasets import load_dataset, concatenate_datasets
    from datetime import datetime

    # Load pretrained scaler if requested
    scaler = None
    if use_scaler:
        if scaler_path is None:
            raise ValueError("scaler_path must be provided when use_scaler=True")
        import joblib
        print(f"Loading pretrained scaler from: {scaler_path}")
        scaler = joblib.load(scaler_path)
        print(f"Scaler loaded successfully: {type(scaler).__name__}")

    # Find all parquet files from primary folder
    parquet_files_folder1 = sorted(glob.glob(os.path.join(parquet_folder, "*.parquet*")))

    if len(parquet_files_folder1) == 0:
        raise ValueError(f"No parquet files found in {parquet_folder}")

    print(f"Found {len(parquet_files_folder1)} parquet files in {parquet_folder}")

    # Load from additional folder if provided (e.g., non-AD patients)
    parquet_files = parquet_files_folder1
    if additional_folder is not None:
        parquet_files_folder2 = sorted(glob.glob(os.path.join(additional_folder, "*.parquet*")))

        if len(parquet_files_folder2) == 0:
            print(f"WARNING: No parquet files found in additional folder: {additional_folder}")
        else:
            print(f"Found {len(parquet_files_folder2)} parquet files in {additional_folder}")

            # Balance file counts if requested
            if balance_folders:
                min_count = min(len(parquet_files_folder1), len(parquet_files_folder2))
                print(f"\nBalancing folders: Using {min_count} files from each folder")

                # Randomly sample to balance (use seed for reproducibility)
                import random
                rng = random.Random(split_seed)
                parquet_files_folder1 = rng.sample(parquet_files_folder1, min_count) if len(parquet_files_folder1) > min_count else parquet_files_folder1
                parquet_files_folder2 = rng.sample(parquet_files_folder2, min_count) if len(parquet_files_folder2) > min_count else parquet_files_folder2

                print(f"  Folder 1 ({os.path.basename(parquet_folder)}): {len(parquet_files_folder1)} files")
                print(f"  Folder 2 ({os.path.basename(additional_folder)}): {len(parquet_files_folder2)} files")

            # Combine files from both folders
            parquet_files = parquet_files_folder1 + parquet_files_folder2
            print(f"Total files after combining: {len(parquet_files)}")
    
    # parquet_files = parquet_files[:50]

    samples_per_second = 20  # 20Hz sampling rate
    max_samples = max_seq_length * samples_per_second

    print(f"\nLoading parquet files directly with HuggingFace load_dataset...")
    print(f"Max sequence length: {max_seq_length}s = {max_samples} samples @ 20Hz")
    print(f"Memory mode: {'RAM (keep_in_memory=True)' if keep_in_memory else 'Memory-mapped (keep_in_memory=False)'}")
    if use_scaler:
        print(f"Scaler: Will apply pretrained scaler to x, y, z columns")

    start_time = datetime.now()

    # Load each parquet file as one segment (all rows concatenated into lists)
    print(f"Loading {len(parquet_files)} parquet files (each file = one segment)...")
    print(f"Each segment will have ~{max_samples:,} timesteps @ 20Hz")

    # Keep same columns as load_data_tso_patch_biobank (no 'other')
    # time_cyclic will be generated from timestamp during loading (not saved in required_cols list yet)
    base_cols = ['x', 'y', 'z', 'temperature', 'predictTSO', 'non-wear']

    # Generator that yields one segment per file
    def segment_generator():
        for file_idx, file in enumerate(parquet_files):
            if file_idx % 100 == 0:
                print(f"  Processing file {file_idx+1}/{len(parquet_files)}: {os.path.basename(file)}")

            try:
                # Parse biobank filename for segment metadata
                basename = os.path.basename(file).replace('.parquet.gzip', '')
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

                # Load parquet file with pandas
                df = pd.read_parquet(file)

                # Truncate if needed
                if len(df) > max_samples:
                    df = df.iloc[:max_samples]

                # Apply scaler to x, y, z columns if requested
                if scaler is not None:
                    columns_to_scale = ['x', 'y', 'z']
                    # Check which columns exist in the dataframe
                    cols_present = [col for col in columns_to_scale if col in df.columns]
                    if cols_present:
                        # Apply scaler transformation
                        df[cols_present] = scaler.transform(df[cols_present])

                # Generate time_cyclic from timestamp (pre-compute to avoid per-batch overhead)
                if 'timestamp' in df.columns:
                    # Convert timestamp to datetime if needed
                    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                        df['timestamp'] = pd.to_datetime(df['timestamp'])

                    # Extract hour and minute
                    hour = df['timestamp'].dt.hour
                    minute = df['timestamp'].dt.minute

                    # Create time embedding (0 to 1 representing time of day)
                    time_embedding = (hour + minute / 60) / 24

                    # Create cyclic encoding using sine wave
                    df['time_cyclic'] = np.sin(2 * np.pi * time_embedding)

                # Determine wrist value (default 0 for biobank if not specified)
                if 'wrist' in df.columns:
                    wrist_value = int(df['wrist'].iloc[0]) if df['wrist'].iloc[0] in [0, 1] else 0
                else:
                    wrist_value = 0  # Default: unknown wrist

                # Create unique segment identifier (matching load_data_tso_patch_biobank format)
                segment_name = f"{current_subject}_{wrist_value}_{instance_info}_{day}"

                # Select required columns (base_cols + time_cyclic)
                output_cols = base_cols + ['time_cyclic']
                existing_cols = [col for col in output_cols if col in df.columns]
                df = df[existing_cols]

                # Convert each column to list (all rows concatenated)
                segment_dict = {}
                for col in existing_cols:
                    segment_dict[col] = df[col].tolist()

                # Add metadata
                segment_dict['segment'] = segment_name
                # segment_dict['subject'] = current_subject
                # segment_dict['wrist'] = 'unknown'  # Biobank doesn't specify wrist
                # segment_dict['day'] = day
                segment_dict['num_samples'] = len(df)

                yield segment_dict
                del df

            except Exception as e:
                print(f"    Error loading {file}: {e}")
                continue

    # Create HuggingFace dataset from generator (incremental writes, no OOM)
    print("Creating HuggingFace dataset from generator (incremental writes)...")
    from datasets import Dataset
    full_dataset = Dataset.from_generator(
        segment_generator,
        writer_batch_size=24,  # Write every 10 segments (adjust if needed)
        keep_in_memory=keep_in_memory
    )

    print(f"\nDataset created:")
    print(f"  Total segments (files): {len(full_dataset)}")
    print(f"  Features: {full_dataset.features}")

    if keep_in_memory:
        # Load into RAM for faster access
        print("Loading dataset into RAM...")
        full_dataset = full_dataset.with_format('numpy')  # Convert to numpy for faster access
        full_dataset.set_format(None)  # Reset format to allow normal access

    load_time = datetime.now() - start_time
    print(f"\nDataset created in {load_time}")
    print(f"  Total rows: {len(full_dataset)}")
    print(f"  Features: {full_dataset.features}")
    print(f"  Memory mode: {'In RAM' if keep_in_memory else 'Memory-mapped'}")

    # Split into train/val if requested
    if split_dataset:
        print(f"\nSplitting dataset into train/val (val_size={val_size}, seed={split_seed})...")


        if shuffle_split:
            print(f"  Using random shuffled split")
        else:
            print(f"  Using ordered split (preserving file order)")
        split_datasets = full_dataset.train_test_split(
            test_size=val_size,
            seed=split_seed,
            shuffle=shuffle_split
        )

        # Rename 'test' to 'val' for clarity
        result = {
            'train': split_datasets['train'],
            'val': split_datasets['test']
        }

        print(f"  Train samples: {len(result['train'])}")
        print(f"  Val samples: {len(result['val'])}")

        return result
    else:
        return full_dataset


# ==================== H5 Dataset Helper Functions ====================
def add_padding_tso_patch_h5(dataset, batch_indices, device, max_seq_len=1440,
                     patch_size=1200, padding_value=0.0, num_channels=None):
    """
    Prepare batch from H5 dataset for TSO prediction.

    This function is analogous to add_padding_tso_patch_hf but works with
    preprocessed H5 data instead of HuggingFace datasets.

    Args:
        dataset: H5Dataset instance (with X, Y, seq_lengths attributes)
        batch_indices: Indices of samples in batch
        device: torch device
        max_seq_len: Maximum sequence length in minutes (default: 1440 = 24h)
        patch_size: Samples per minute (default: 1200 = 60s * 20Hz)
        padding_value: Padding value for X data (default: 0.0)
        num_channels: Number of input channels (5 or 6). If None, read from dataset metadata.

    Returns:
        pad_X: [batch_size, seq_len, patch_size, num_channels] - raw sensor patches
               Channels (5): [x, y, z, temperature, time_sin] OR
               Channels (6): [x, y, z, temperature, time_sin, time_cos]
        pad_Y: [batch_size, seq_len] - minute-level labels (0=other, 1=non-wear, 2=predictTSO)
        x_lens: [batch_size] - sequence lengths in minutes
        batch_samples: List of dicts with segment info

    Note:
        Label aggregation uses np.any() logic to match add_padding_tso_patch_hf (line 3536-3541):
        - If ANY sample in minute has predictTSO=1, label entire minute as predictTSO (2)
        - Else if ANY sample has non-wear=1, label entire minute as non-wear (1)
        - Otherwise label as other (0)
    """

    batch_size = len(batch_indices)
    samples_per_minute = patch_size  # 1200 samples = 1 minute @ 20Hz

    # Get num_channels from dataset metadata if not provided
    if num_channels is None:
        num_channels = dataset.num_channels

    # Get data for all samples in batch
    X_batch = []
    Y_batch = []
    lens_batch = []
    segments_batch = []

    for idx in batch_indices:
        sample = dataset[idx]
        X_batch.append(sample['X'])  # [max_len, num_channels]
        Y_batch.append(sample['Y'])  # [max_len, 2]
        lens_batch.append(sample['seq_length'])
        segments_batch.append({'segment': sample['segment']})

    X_batch = np.stack(X_batch)  # [batch_size, max_len, num_channels]
    Y_batch = np.stack(Y_batch)  # [batch_size, max_len, 2]
    lens_batch = np.array(lens_batch)

    # Convert to patches (minute-level aggregation)
    # X_batch: [batch_size, max_len, num_channels] -> [batch_size, num_minutes, patch_size, num_channels]
    num_minutes_max = min(max_seq_len, X_batch.shape[1] // samples_per_minute)

    pad_X = np.full((batch_size, num_minutes_max, patch_size, num_channels),
                    padding_value, dtype=np.float32)
    pad_Y = np.full((batch_size, num_minutes_max), -100, dtype=np.int64)  # -100 for ignore
    x_lens = np.zeros(batch_size, dtype=np.int64)

    for i in range(batch_size):
        seq_len_samples = lens_batch[i]
        num_minutes = min(num_minutes_max, seq_len_samples // samples_per_minute)

        if num_minutes > 0:
            x_lens[i] = num_minutes

            # Reshape to patches: [seq_len_samples, num_channels] -> [num_minutes, patch_size, num_channels]
            samples_to_use = num_minutes * samples_per_minute
            X_reshaped = X_batch[i, :samples_to_use, :].reshape(num_minutes, patch_size, num_channels)
            pad_X[i, :num_minutes, :, :] = X_reshaped

            # Aggregate labels to minute-level (using np.any logic to match line 3536-3541)
            # Y_batch: [max_len, 2] -> predictTSO and non_wear
            Y_reshaped = Y_batch[i, :samples_to_use, :].reshape(num_minutes, patch_size, 2)

            # For each minute, determine label (priority: predictTSO > non_wear > other)
            # IMPORTANT: Use np.any() to match add_padding_tso_patch_hf behavior
            for m in range(num_minutes):
                minute_predictTSO = Y_reshaped[m, :, 0]  # [patch_size]
                minute_nonwear = Y_reshaped[m, :, 1]  # [patch_size]

                # Use np.any() - if ANY sample in minute has label=1, use that label
                if np.any(minute_predictTSO):
                    pad_Y[i, m] = 2  # predictTSO
                elif np.any(minute_nonwear):
                    pad_Y[i, m] = 1  # non-wear
                else:
                    pad_Y[i, m] = 0  # other

    # Convert to torch tensors
    pad_X = torch.from_numpy(pad_X).to(device)
    pad_Y = torch.from_numpy(pad_Y).to(device)
    x_lens = torch.from_numpy(x_lens).to(device)

    return pad_X, pad_Y, x_lens, segments_batch


# ==================== Prediction Smoothing Functions ====================
def smooth_predictions(predictions, method='majority_vote', window_size=5, min_segment_length=3):
    """
    Smooth predictions to reduce unstable jumping between classes.

    Args:
        predictions: [batch_size, seq_len, 3] - model predictions (logits or probabilities)
        method: str - smoothing method:
            - 'majority_vote': Sliding window majority voting (recommended for classification)
            - 'median_filter': Median filter on class predictions
            - 'moving_average': Moving average on probabilities then argmax
            - 'gaussian': Gaussian weighted moving average
            - 'min_segment': Remove segments shorter than min_segment_length
        window_size: int - size of smoothing window (should be odd, e.g., 5, 7, 9)
        min_segment_length: int - minimum segment length for 'min_segment' method

    Returns:
        smoothed_predictions: [batch_size, seq_len, 3] - smoothed predictions
    """
    import torch
    import numpy as np
    from scipy.ndimage import median_filter, gaussian_filter1d
    from scipy.signal import medfilt

    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()

    batch_size, seq_len, num_classes = predictions.shape
    smoothed = predictions.copy()

    # Convert to probabilities if needed (assume sigmoid already applied)
    probs = predictions  # Already probabilities from sigmoid

    for i in range(batch_size):
        if method == 'majority_vote':
            # Sliding window majority voting on class labels
            class_preds = np.argmax(probs[i], axis=-1)  # [seq_len]
            smoothed_classes = np.zeros_like(class_preds)

            half_window = window_size // 2
            for t in range(seq_len):
                # Get window bounds
                start = max(0, t - half_window)
                end = min(seq_len, t + half_window + 1)
                window = class_preds[start:end]

                # Majority vote
                counts = np.bincount(window, minlength=num_classes)
                smoothed_classes[t] = np.argmax(counts)

            # Convert back to one-hot style probabilities
            smoothed[i] = np.eye(num_classes)[smoothed_classes]

        elif method == 'median_filter':
            # Median filter on class predictions
            class_preds = np.argmax(probs[i], axis=-1)
            # Median filter requires odd kernel size
            kernel_size = window_size if window_size % 2 == 1 else window_size + 1
            smoothed_classes = medfilt(class_preds.astype(float), kernel_size=kernel_size).astype(int)
            smoothed_classes = np.clip(smoothed_classes, 0, num_classes - 1)
            smoothed[i] = np.eye(num_classes)[smoothed_classes]

        elif method == 'moving_average':
            # Moving average on probabilities
            for c in range(num_classes):
                smoothed[i, :, c] = np.convolve(probs[i, :, c],
                                                np.ones(window_size)/window_size,
                                                mode='same')

        elif method == 'gaussian':
            # Gaussian weighted moving average
            sigma = window_size / 4  # Standard deviation
            for c in range(num_classes):
                smoothed[i, :, c] = gaussian_filter1d(probs[i, :, c], sigma=sigma, mode='nearest')

        elif method == 'min_segment':
            # Remove short segments by merging with neighbors
            class_preds = np.argmax(probs[i], axis=-1)
            smoothed_classes = class_preds.copy()

            # Find segment boundaries
            changes = np.where(np.diff(class_preds) != 0)[0] + 1
            segments = np.split(np.arange(seq_len), changes)
            segment_labels = [class_preds[seg[0]] for seg in segments]

            # Merge short segments
            merged_labels = []
            merged_segments = []

            for idx, (seg, label) in enumerate(zip(segments, segment_labels)):
                if len(seg) < min_segment_length:
                    # Merge with previous or next segment
                    if merged_labels:
                        # Merge with previous
                        merged_segments[-1] = np.concatenate([merged_segments[-1], seg])
                    elif idx + 1 < len(segments):
                        # Merge with next
                        continue
                    else:
                        # Keep as is (edge case)
                        merged_segments.append(seg)
                        merged_labels.append(label)
                else:
                    merged_segments.append(seg)
                    merged_labels.append(label)

            # Apply merged labels
            for seg, label in zip(merged_segments, merged_labels):
                smoothed_classes[seg] = label

            smoothed[i] = np.eye(num_classes)[smoothed_classes]

    return smoothed


def smooth_predictions_combined(predictions, methods=['majority_vote', 'min_segment'],
                                window_size=5, min_segment_length=3):
    """
    Apply multiple smoothing methods sequentially.
    Recommended: majority_vote -> min_segment for best results.

    Args:
        predictions: [batch_size, seq_len, 3] - model predictions
        methods: list of str - smoothing methods to apply in order
        window_size: int - window size for voting/averaging methods
        min_segment_length: int - minimum segment length

    Returns:
        smoothed_predictions: [batch_size, seq_len, 3] - smoothed predictions
    """
    smoothed = predictions
    for method in methods:
        smoothed = smooth_predictions(smoothed, method=method,
                                     window_size=window_size,
                                     min_segment_length=min_segment_length)
    return smoothed


# ==================== Learning Curves Plotting Function ====================
def plot_tso_learning_curves(history, output_filepath):
    """
    Plot comprehensive training history for TSO prediction task.

    Args:
        history: Dictionary containing training history with keys:
            - train_loss, val_loss
            - train_accuracy, val_accuracy
            - train_f1_avg, val_f1_avg
            - train_f1_other, train_f1_nonwear, train_f1_tso
            - val_f1_other, val_f1_nonwear, val_f1_tso
        output_filepath: Path to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    epochs = range(1, len(history['train_loss']) + 1)

    # Total loss
    axes[0, 0].plot(epochs, history['train_loss'], label='Train', linewidth=2, color='#1f77b4')
    axes[0, 0].plot(epochs, history['val_loss'], label='Val', linewidth=2, color='#ff7f0e')
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Total Loss', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(epochs, history['train_accuracy'], label='Train', linewidth=2, color='#1f77b4')
    axes[0, 1].plot(epochs, history['val_accuracy'], label='Val', linewidth=2, color='#ff7f0e')
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Accuracy', fontsize=11)
    axes[0, 1].set_title('Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # Average F1
    axes[0, 2].plot(epochs, history['train_f1_avg'], label='Train', linewidth=2, color='#1f77b4')
    axes[0, 2].plot(epochs, history['val_f1_avg'], label='Val', linewidth=2, color='#ff7f0e')
    axes[0, 2].set_xlabel('Epoch', fontsize=11)
    axes[0, 2].set_ylabel('F1 Score', fontsize=11)
    axes[0, 2].set_title('Average F1 Score', fontsize=12, fontweight='bold')
    axes[0, 2].legend(fontsize=10)
    axes[0, 2].grid(True, alpha=0.3)

    # F1 per class - Train
    if 'train_f1_other' in history:
        axes[1, 0].plot(epochs, history['train_f1_other'], label='Other', linewidth=2, color='#2ca02c')
        axes[1, 0].plot(epochs, history['train_f1_nonwear'], label='Non-wear', linewidth=2, color='#d62728')
    axes[1, 0].plot(epochs, history['train_f1_tso'], label='TSO', linewidth=2, color='#9467bd')
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('F1 Score', fontsize=11)
    axes[1, 0].set_title('Train F1 per Class', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # F1 per class - Val
    if 'val_f1_other' in history:
        axes[1, 1].plot(epochs, history['val_f1_other'], label='Other', linewidth=2, color='#2ca02c')
        axes[1, 1].plot(epochs, history['val_f1_nonwear'], label='Non-wear', linewidth=2, color='#d62728')
    axes[1, 1].plot(epochs, history['val_f1_tso'], label='TSO', linewidth=2, color='#9467bd')
    axes[1, 1].set_xlabel('Epoch', fontsize=11)
    axes[1, 1].set_ylabel('F1 Score', fontsize=11)
    axes[1, 1].set_title('Val F1 per Class', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    # Combined F1 vs Accuracy comparison
    axes[1, 2].plot(epochs, history['train_f1_avg'], label='Train F1', linewidth=2,
                   linestyle='--', color='#1f77b4', alpha=0.7)
    axes[1, 2].plot(epochs, history['val_f1_avg'], label='Val F1', linewidth=2,
                   linestyle='--', color='#ff7f0e', alpha=0.7)
    axes[1, 2].plot(epochs, history['train_accuracy'], label='Train Acc', linewidth=2, color='#1f77b4')
    axes[1, 2].plot(epochs, history['val_accuracy'], label='Val Acc', linewidth=2, color='#ff7f0e')
    axes[1, 2].set_xlabel('Epoch', fontsize=11)
    axes[1, 2].set_ylabel('Score', fontsize=11)
    axes[1, 2].set_title('F1 vs Accuracy', fontsize=12, fontweight='bold')
    axes[1, 2].legend(fontsize=9)
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_filepath, dpi=150, bbox_inches='tight')
    plt.close()