# -*- coding: utf-8 -*-
"""
Split from DL_helpers.py - Modular code structure
@author: MBoukhec (original)
"""

import numpy as np
import pandas as pd
import re
from datetime import datetime
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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



def assign_incremental_numbers(group, n):
    group['mini_segment'] = ((group.index-min(group.index)) // n) + 1
    return group



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
            
            

