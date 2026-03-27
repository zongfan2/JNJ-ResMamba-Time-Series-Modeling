# -*- coding: utf-8 -*-
"""
Split from DL_helpers.py - Modular code structure
@author: MBoukhec (original)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

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

