# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 08:15:17 2024

@author: MBoukhec
"""


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



# Global setup
error_logs = []
processing_logs = []

def df_subset(_df,ratio,stratified):
    random_indices=[]
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

def augment_dataset(df, num_iterations,interchange=True, verbose=True):
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
    scratch_duration['scratch_duration'] = scratch_duration.scratch_sum / scratch_duration.scratch_count
    new_df = new_df.drop(['scratch_sum', 'scratch_count', 'scratch_duration'], axis=1).merge(scratch_duration)
    new_df = new_df.drop(['segment_scratch'], axis=1).merge(new_df.groupby('segment')['scratch'].any().reset_index().rename(columns={'scratch':'segment_scratch'}))
    df_segment=new_df.groupby('segment').max(1).reset_index()
    print(f"Segment prevalence after augmentation: {df_segment[(df_segment.segment_scratch==True)].shape[0]/df_segment.shape[0]:.4f}, count: {df_segment.shape[0]}, average duration: {df_segment.groupby('segment')['scratch_count'].max(1).mean()/20} s")
    return new_df

def remove_missmatch_labels(df):
    #remove when there is mismatching betweet left and right labelers
    missmatch=len(df)        
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


# def add_change_point(s,binary=False):
#     a=s[['x','y','z']].reset_index(drop=True)
#     algo=rpt.Pelt(model='rbf').fit(a)
#     results=[]
#     a.loc[:,'change']=0
#     try:
#         results=algo.predict(pen=10)
#         results = [row -1 for row in results]
#         a.loc[results,'change']=1
#         a['change']= a.change.cumsum()
#     except Exception as e:
#         print("Error:", e)
 
#     if binary:
#         a['change']= (a['change'] %2!=0).astype(int)
#     s['change']=a['change'].values
#     return s


def load_data(input_data_folder,motion_filter=True,max_seq_length=60,energy_th=5,remove_outbed=False,filter_type='motion'):
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
    df['scratch_duration']=df['scratch_duration']/df['scratch_count']

    df['ADT']=(df['TSOEND']+ pd.Timedelta(hours=10, minutes=0, seconds=0)).dt.date.astype(str)
    #add positions
    df['position_segment']=df.groupby('segment')['timestamp'].rank(method='first')
    df['position_segmentr']=df.groupby('segment')['timestamp'].rank(method='first')/df.scratch_count
#     df['position_TSO']=(df.timestamp-df.TSOSTART).dt.seconds
#     df['position_TSOr']=(df.timestamp-df.TSOSTART)/(df.TSOEND-df.TSOSTART)

#     df['predictTSOSTART']=df[df.predictTSO].groupby(['PID','wrist','DAY'])['timestamp'].transform('min')
#     df['predictTSOEND']=df[df.predictTSO].groupby(['PID','wrist','DAY'])['timestamp'].transform('max')
    df=df[(df.SUBJECT!='US10008015')| (df.ADT!="2023-03-15")] # No Offset was found for this subject/date. Excluding it from the dataset.
    df=df[(df.SUBJECT!='US10008007')| (df.ADT!="2022-12-24")] #Subject removed the edvices during this night
    
    df['skinimpact']=np.where((df.SkinImpactLabeler1=='Yes')&(df.SkinImpactLabeler2=='Yes'),1,0)
    df['skinimpact_u']=np.where((df.SkinImpactLabeler1=='Yes')|(df.SkinImpactLabeler2=='Yes'),1,0)

    #Filter the segments having less than the energy threshold
    if filter_type=='motion':
        if energy_th is not None:
            print("Load data: df.shape after removing segments less than threshold: ", df.shape)
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
            
        df_temp['wrist']=df_temp.wrist.str.lower()#np.where(df_temp.wrist.str.lower()=="left",1,0)
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
    df_segment=df.drop_duplicates(subset='segment')
    print(f"Segment prevalence: {df_segment[(df_segment.segment_scratch==True)].shape[0]/df_segment.shape[0]:.4f}, count: {df_segment.shape[0]}, average duration: {df_segment.groupby('segment')['scratch_count'].max(1).mean()/sf} s")
    return df


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




def calculate_metrics_nn(actuals, predictions, classification=True):
    if classification:
        # Ensure predictions are integers
        predictions = predictions.astype(int)
        actuals = actuals.astype(int)
        # Calculating various classification metrics
        metrics_values = {
            'accuracy': metrics.accuracy_score(actuals, predictions),
            'balanced_accuracy': metrics.balanced_accuracy_score(actuals, predictions),
            'precision': metrics.precision_score(actuals, predictions, zero_division=0),
            'precision_macro': metrics.precision_score(actuals, predictions, average='macro', zero_division=0),
            'precision_micro': metrics.precision_score(actuals, predictions, average='micro', zero_division=0),
            'precision_weighted': metrics.precision_score(actuals, predictions, average='weighted', zero_division=0),
            'recall': metrics.recall_score(actuals, predictions, zero_division=0),
            'recall_macro': metrics.recall_score(actuals, predictions, average='macro', zero_division=0),
            'recall_micro': metrics.recall_score(actuals, predictions, average='micro', zero_division=0),
            'recall_weighted': metrics.recall_score(actuals, predictions, average='weighted', zero_division=0),
            'f1_score': metrics.f1_score(actuals, predictions, zero_division=0),
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
from torcheval.metrics.functional import multiclass_f1_score,binary_f1_score
from datetime import datetime
from torch.nn.utils.rnn import pad_sequence
from sklearn.utils.class_weight import compute_class_weight
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
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

    else:
        segs = df[seg_column].unique()
        if shuffle:
            np.random.shuffle(segs)
        for i in range(0,len(segs),batch_size):
            seg_choice=segs[i:i+batch_size]
            steps +=1
    return steps

def df_subset_segments(_df,ratio,stratified,seed):
    random_indices=[]
    rng = np.random.default_rng(seed) # Create a Generator instance with a seed
    
    if stratified: # TODO: To change to account for segments
        #we take random ratio from all subjects and same distrbutino of scratch/noscratch
        for PID in _df.SUBJECT.unique():
            indices = rng.choice(_df[(_df.SUBJECT==PID) & (_df.scratch == True)].index, int(len(_df[(_df.SUBJECT==PID) & (_df.scratch == True)])*ratio), replace=False)
            random_indices=np.append(random_indices,indices)
            indices = rng.choice(_df[(_df.SUBJECT==PID) & (_df.scratch == False)].index, int(len(_df[(_df.SUBJECT==PID) & (_df.scratch == False)])*ratio), replace=False)
            random_indices=np.append(random_indices,indices)
    else:
        #we take random ratio with the same distrbutino of scratch/noscratch segments
        neg_segs=_df[(_df.segment_scratch==False)]['segment'].unique()
        pos_segs=_df[(_df.segment_scratch==True)]['segment'].unique()
        pos_choise = rng.choice(pos_segs, int(len(pos_segs)*ratio), replace=False) 
        neg_choise = rng.choice(neg_segs, int(len(neg_segs)*ratio), replace=False)
        random_indices=_df[_df.segment.isin(np.concatenate([pos_choise,neg_choise]))].index
    
    return random_indices

            

def add_padding(batch,device,seg_column='segment',max_seq_len=None,training=True):
    X_sequences=[]
    Y_sequences=[]
    x_lens=[]
    label1=[]
    label3=[]
    if max_seq_len is not None:
        max_seq_template = torch.ones(max_seq_len, 3).to(device)
        X_sequences.append(max_seq_template)
    for index,seq in batch.groupby(seg_column, sort=False):
        xyz=seq
        if training:

#             Correct for potential difference in x,y labeling 
            xyz_p=np.random.permutation(['x','y'])#,'z'
            xyz=xyz.rename(columns={'x': xyz_p[0], 'y': xyz_p[1]})#, 'z': xyz_p[2] ,'anglex': f'angle{xyz_p[0]}', 'angley': f'angle{xyz_p[1]}'
#             Correct for wearing the device under the wrist
            flip_z=np.random.choice([-1, 1])
            xyz['x']=xyz['x']*flip_z
            xyz['z']=xyz['z']*flip_z
            
#             correct for differences between left vs right wrist
            flip_wirst=np.random.choice([-1, 1])
            xyz['y']=xyz['y']*np.random.choice([-1, 1])
            
#         X_arr=xyz.loc[:, ['x', 'y', 'z','change']].to_numpy()
        X_arr=seq.loc[:, ['x', 'y', 'z']].to_numpy() #,'anglex','angley','anglez','temperature'
        Y_arr=np.array([seq['scratch'].values]).T
        X_sequences.append(torch.tensor(X_arr,dtype=torch.float32,device=device))#,device=device
#         Y_sequences.append(torch.tensor(Y_arr,dtype=torch.long,device=device))#,device=device
        Y_sequences=np.concatenate((Y_sequences,seq['scratch'].values))
        x_lens.append(len(X_arr))
        label1.append(seq['segment_scratch'].any()*1)
        label3.append(seq['scratch_duration'].max())

    pad_X=pad_sequence(X_sequences, batch_first=True) #,padding_value=-999
    #label2=pad_sequence(Y_sequences, batch_first=True,padding_value=-999)
    if max_seq_len:
        # discard the max seq template now
        pad_X = pad_X[1:, :, :]
    
    label2=torch.tensor(Y_sequences,device=device)
    
    
    return pad_X,torch.tensor(label1,device=device),label2,torch.tensor(label3,device=device),x_lens

def add_padding_pretrain(batch,device,seg_column='segment',mask_rate=0.3):
    X_sequences=[]
    X_sequences_masked=[]
    x_lens=[]
    for index,seq in batch.groupby(seg_column, sort=False):
        #Shuffling x,y,z
        xyz_p=np.random.permutation(['x','y','z'])
        seq=seq.rename(columns={'x': xyz_p[0], 'y': xyz_p[1], 'z': xyz_p[2]})
        X_arr=seq.loc[:, ['x', 'y', 'z']].to_numpy()
        X_arr_masked= random_patch_masking(X_arr,masking_ratio=0.3,patch_size=10)
        X_sequences.append(torch.tensor(X_arr,dtype=torch.float32,device=device))
        X_sequences_masked.append(torch.tensor(X_arr_masked,dtype=torch.float32,device=device)) 
        x_lens.append(len(X_arr))
#     y_lens = [len(y) for y in X_sequences]
    pad_X=pad_sequence(X_sequences, batch_first=True,padding_value=-999) #,padding_value=-999
    pad_X_masked=pad_sequence(X_sequences_masked, batch_first=True)
    return pad_X_masked,pad_X,x_lens

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

class SequenceDiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(SequenceDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        intersection = (preds * targets).sum()
        return 1 - (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)

class SequenceJaccardLoss(nn.Module):
    def __init__(self, smooth=1):
        super(SequenceJaccardLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum() - intersection
        return 1 - (intersection + self.smooth) / (union + self.smooth)

def measure_loss_multitask(outputs1,outputs2,outputs3,label1,label2,label3,wl1=1.0, wl2=1, wl3=0.1, pos_weight_l1=False,pos_weight_l2=False):
    """
    Measure loss without reduction and mask the padded values to prevent loss calculation from padded values.
    """
    weight1, weight2 = None, None
    
    if pos_weight_l1:
        weight1 = torch.ones([len(label1)],device=outputs1.device) * 3
    
    if pos_weight_l1:
        weight2 = torch.ones([len(label2)],device=outputs1.device)*10

    loss1 = nn.BCEWithLogitsLoss(pos_weight=weight1)(outputs1,label1.float()) 
    loss2 = nn.BCEWithLogitsLoss(pos_weight=weight2)(outputs2,label2.view(-1).float()) 
    loss3 = F.mse_loss(outputs3,label3.float())
    
    return wl1*loss1+wl2*loss2+wl3*loss3,loss1,loss2,loss3#+ 0.1*(loss2+loss3)#+loss3#loss1 + loss2 + loss3


# ==================== TSO Patch Data Helpers ====================

def add_padding_tso_patch_h5(dataset, batch_indices, device, max_seq_len=1440,
                              patch_size=1200, padding_value=0.0, num_channels=None):
    """
    Prepare a batch from an H5 dataset for TSO patch model inference.

    Args:
        dataset: H5Dataset instance with X, Y, seq_lengths attributes.
        batch_indices: List of sample indices in the batch.
        device: torch device.
        max_seq_len: Maximum sequence length in minutes (default: 1440 = 24 h).
        patch_size: Samples per minute (default: 1200 = 60 s * 20 Hz).
        padding_value: Padding fill value for X (default: 0.0).
        num_channels: Number of input channels. If None, read from dataset metadata.

    Returns:
        pad_X:  [batch_size, max_seq_len, patch_size, num_channels]
        pad_Y:  [batch_size, max_seq_len]  minute-level labels (0=other, 1=non-wear, 2=TSO)
        x_lens: [batch_size]               sequence lengths in minutes  (int64)
        segments_batch: list of dicts with segment info
    """
    import torch
    import numpy as np

    batch_size = len(batch_indices)
    samples_per_minute = patch_size

    if num_channels is None:
        num_channels = dataset.num_channels

    X_batch, Y_batch, lens_batch, segments_batch = [], [], [], []
    for idx in batch_indices:
        sample = dataset[idx]
        X_batch.append(sample['X'])          # [max_len, num_channels]
        Y_batch.append(sample['Y'])          # [max_len, 2]
        lens_batch.append(sample['seq_length'])
        segments_batch.append({'segment': sample['segment']})

    X_batch = np.stack(X_batch)              # [B, max_len, num_channels]
    Y_batch = np.stack(Y_batch)              # [B, max_len, 2]
    lens_batch = np.array(lens_batch)

    num_minutes_max = min(max_seq_len, X_batch.shape[1] // samples_per_minute)

    pad_X = np.full((batch_size, num_minutes_max, patch_size, num_channels),
                    padding_value, dtype=np.float32)
    pad_Y = np.full((batch_size, num_minutes_max), -100, dtype=np.int64)
    x_lens = np.zeros(batch_size, dtype=np.int64)

    for i in range(batch_size):
        seq_len_samples = lens_batch[i]
        num_minutes = min(num_minutes_max, seq_len_samples // samples_per_minute)

        if num_minutes > 0:
            x_lens[i] = num_minutes

            samples_to_use = num_minutes * samples_per_minute
            X_reshaped = X_batch[i, :samples_to_use, :].reshape(
                num_minutes, patch_size, num_channels)
            pad_X[i, :num_minutes, :, :] = X_reshaped

            # Aggregate labels to minute level (priority: TSO > non-wear > other)
            Y_reshaped = Y_batch[i, :samples_to_use, :].reshape(num_minutes, patch_size, 2)
            for m in range(num_minutes):
                minute_tso = Y_reshaped[m, :, 0]
                minute_nonwear = Y_reshaped[m, :, 1]
                if np.any(minute_tso):
                    pad_Y[i, m] = 2
                elif np.any(minute_nonwear):
                    pad_Y[i, m] = 1
                else:
                    pad_Y[i, m] = 0

    pad_X = torch.from_numpy(pad_X).to(device)
    pad_Y = torch.from_numpy(pad_Y).to(device)
    x_lens = torch.from_numpy(x_lens).to(device)

    return pad_X, pad_Y, x_lens, segments_batch


# ==================== Prediction Smoothing Functions ====================

def smooth_predictions(predictions, method='majority_vote', window_size=5, min_segment_length=3):
    """
    Smooth predictions to reduce unstable class transitions.

    Args:
        predictions: [batch_size, seq_len, num_classes] — probabilities or logits.
        method: 'majority_vote' | 'median_filter' | 'moving_average' | 'gaussian' | 'min_segment'
        window_size: Smoothing window (should be odd).
        min_segment_length: Minimum segment length for 'min_segment' method.

    Returns:
        smoothed: [batch_size, seq_len, num_classes]
    """
    import numpy as np
    import torch
    from scipy.signal import medfilt
    from scipy.ndimage import gaussian_filter1d

    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()

    batch_size, seq_len, num_classes = predictions.shape
    smoothed = predictions.copy()
    probs = predictions

    for i in range(batch_size):
        if method == 'majority_vote':
            class_preds = np.argmax(probs[i], axis=-1)
            smoothed_classes = np.zeros_like(class_preds)
            half_window = window_size // 2
            for t in range(seq_len):
                start = max(0, t - half_window)
                end = min(seq_len, t + half_window + 1)
                window = class_preds[start:end]
                counts = np.bincount(window, minlength=num_classes)
                smoothed_classes[t] = np.argmax(counts)
            smoothed[i] = np.eye(num_classes)[smoothed_classes]

        elif method == 'median_filter':
            class_preds = np.argmax(probs[i], axis=-1)
            kernel_size = window_size if window_size % 2 == 1 else window_size + 1
            smoothed_classes = medfilt(class_preds.astype(float),
                                       kernel_size=kernel_size).astype(int)
            smoothed_classes = np.clip(smoothed_classes, 0, num_classes - 1)
            smoothed[i] = np.eye(num_classes)[smoothed_classes]

        elif method == 'moving_average':
            for c in range(num_classes):
                smoothed[i, :, c] = np.convolve(probs[i, :, c],
                                                 np.ones(window_size) / window_size,
                                                 mode='same')

        elif method == 'gaussian':
            sigma = window_size / 4
            for c in range(num_classes):
                smoothed[i, :, c] = gaussian_filter1d(probs[i, :, c],
                                                       sigma=sigma, mode='nearest')

        elif method == 'min_segment':
            class_preds = np.argmax(probs[i], axis=-1)
            smoothed_classes = class_preds.copy()
            changes = np.where(np.diff(class_preds) != 0)[0] + 1
            segments = np.split(np.arange(seq_len), changes)
            segment_labels = [class_preds[seg[0]] for seg in segments]

            merged_labels, merged_segments = [], []
            for idx, (seg, label) in enumerate(zip(segments, segment_labels)):
                if len(seg) < min_segment_length:
                    if merged_labels:
                        merged_segments[-1] = np.concatenate([merged_segments[-1], seg])
                    elif idx + 1 < len(segments):
                        continue
                    else:
                        merged_segments.append(seg)
                        merged_labels.append(label)
                else:
                    merged_segments.append(seg)
                    merged_labels.append(label)

            for seg, label in zip(merged_segments, merged_labels):
                smoothed_classes[seg] = label
            smoothed[i] = np.eye(num_classes)[smoothed_classes]

    return smoothed


def smooth_predictions_combined(predictions, methods=None, window_size=5, min_segment_length=3):
    """
    Apply multiple smoothing methods sequentially.
    Recommended: ['majority_vote', 'min_segment'].

    Args:
        predictions: [batch_size, seq_len, num_classes]
        methods: list of method names (default: ['majority_vote', 'min_segment'])
        window_size: Window size for voting/averaging methods.
        min_segment_length: Minimum segment length.

    Returns:
        smoothed: [batch_size, seq_len, num_classes]
    """
    if methods is None:
        methods = ['majority_vote', 'min_segment']
    smoothed = predictions
    for method in methods:
        smoothed = smooth_predictions(smoothed, method=method,
                                      window_size=window_size,
                                      min_segment_length=min_segment_length)
    return smoothed

