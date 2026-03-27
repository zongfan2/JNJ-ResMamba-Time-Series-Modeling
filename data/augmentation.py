# -*- coding: utf-8 -*-
"""
Split from DL_helpers.py - Modular code structure
@author: MBoukhec (original)
"""

import numpy as np
import pandas as pd
import random
from datetime import datetime
import concurrent.futures

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



