# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 19:02:22 2023

@author: MBoukhec
"""

import subprocess
shell_script = '''
sudo python3.11 -m pip install pandas==2.1.4 numpy==1.26.2 scikit-learn==1.3.2 joblib==1.3.2
sudo python3.11 -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 
sudo python3.11 -m pip install transformers==4.40.2
sudo python3.11 -m pip install statsmodels 
sudo python3.11 -m pip install multiprocess matplotlib
sudo python3.11 -m pip install torcheval seaborn  mamba-ssm[causal-conv1d]==2.2.2 
'''
result = subprocess.run(shell_script, shell=True, capture_output=True, text=True)



import os
import shutil
import csv
import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
import pytz
from random import shuffle
import preprocessing as p 

from  Helpers.helpers import *
from Helpers.classical_backend import ClassicalBackend
import argparse
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
import joblib
import logging
import gc
import concurrent.futures
import ctypes
import multiprocessing
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import warnings
# Set the maximum number of open figures  to 0 to avoid warning
mpl.rcParams['figure.max_open_warning'] = 0 
script_dir = os.path.dirname(os.path.abspath(__file__))

import sklearn

# ---------------
# Parse Arguments
# ---------------
parser = argparse.ArgumentParser(description='Run Scratch Detectection Pipeline. Example : /n python /mnt/code/munge/preprocessing/geneactiv_data_preprocessing.py --target_folder /mnt/data/Nocturnal-scratch/geneactive_20hz_2s_b1s_imerit_sleep_nonwearOR_5smerge_nograv_start00h --day_start 0',formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("--data_folder",type=str, required=True, help="Path to the folder storing the data.")
parser.add_argument("--TZ_file",type=str, required=False,default="", help="Path to the file containing subjects' timezone.")
parser.add_argument("--day_start",type=int,default=18,  required=False, help="Time of the day (0 to 23) used to split the data to daily files.")
parser.add_argument("--source",type=str, required=False,default='NOPROD', help="The type of the device that generated this data. Currently accepting : [NOPROD,Leap,UKB]")
parser.add_argument("--target_folder",type=str, required=True, help="Path to the folder in which results will be stored.")
parser.add_argument("--TSO_algo",type=str, choices=['van_old', 'van_new', 'dl'],default='van_new', help="The algorithm to be used for measuring TSO. 'van_old'/'van_new': Van Hees implementations. 'dl': deep learning TSO_MBA model.")
parser.add_argument("--tso_model_path", type=str, default=None, required=False, help="Path to the trained TSO_MBA model checkpoint (.pth). Required when --TSO_algo dl is used.")
parser.add_argument("--tso_model_name", type=str, default='mba4tso_patch', required=False, help="Model architecture name for TSO_MBA. Default: mba4tso_patch.")
parser.add_argument("--tso_device", type=str, default='cuda:0', required=False, help="Device for TSO_MBA inference (e.g. 'cuda:0', 'cpu'). Default: cuda:0.")
parser.add_argument("--nonwear_algo",type=str, choices=['JJ', 'sleepy','detach','zhou'],default='JJ', help="The algorithm to be used for detecting non-wear periods.")
parser.add_argument('--skip_gravity_calibration', action='store_false', required=False, help='Calibrate gravity by default. Skip calibration if this argument is added.')
parser.add_argument("--sf",type=int,default=20, help="The sampling frequency of the accelerometer sensor.")
parser.add_argument("--resample",type=str,default='False', help="Choose either [False, None, Uniform, target frequency (int)]. If uniform, use the original frequency to fix any device sampling errors. Pass None or False to disable resampling. Pass the target frequency (Hz) to resample the signal. Defaults to False.")

parser.add_argument("--lowpass",type=int,default=None,  required=False, help="Cutoff (Hz) for the low pass filter. Defaults to None.")
parser.add_argument("--highpass",type=int,default=0.5,  required=False, help="Cutoff (Hz) for the high pass filter. Defaults to 0.5HZ to remove gravity.")
parser.add_argument('--staud', action='store_true', required=False, help='Wether to extract Staudentmayer activity levels. False if this argument is not added.')

parser.add_argument('--production', action='store_true', required=False, help='Wether to run the pipeline in production model. This will prevent adding annotations. False if this argument is not added.')
parser.add_argument("--min_sleep",type=int,default=0.5,  required=False, help="The minimum number of sleep hours required for a night to be considered valid. Default is 0.5 hours.")
parser.add_argument("--min_data",type=int,default=12,  required=False, help="The minimum duration of accelerometer data (in hours) required for a day to be deemed valid. Default is 12 hours.")

parser.add_argument('--scratch', action='store_true', required=False, help='Wether to run the scratch detection model. False if this argument is not added.')
parser.add_argument('--scratch_sleep_only', action='store_true', required=False, help='Only run the scratch detection model during the predicted TSO. False if this argument is not added.')
parser.add_argument('--scratch_model', type=str,default='mbav1', required=False, help='The name of the scratch model used to predict scratch.')
parser.add_argument("--num_gpu",type=int,default=0, required=False, help="Which gpu to use")

parser.add_argument('--write_raw', action='store_true', required=False, help='Wether to write the intermediate raw data or not. False if this argument is not added.')
parser.add_argument('--write_motion_only', action='store_true', required=False, help='Wether to write to only write motion segments (removing stationary segments). False if this argument is not added.')

parser.add_argument('--plot', action='store_true', required=False, help='Wether to plot the data. False if this argument is not added.')
parser.add_argument('--clear_tracker', action='store_true', required=False, help='Wether to delete subject trackers. This will reset any already processed files.')
# Examples
# NOPROD

# python '/mnt/code/munge/preprocessing/NS-pipeline.py' --data_folder '/mnt/imported/data/NOPRODNA0029/for_s3' --target_folder '/mnt/data/Nocturnal-scratch/geneactive_20hz_3s_b1s_production_test_scratch' --calibrate_gravity --production --scratch --plot 

# python '/mnt/imported/code/nocturnal-scratch-algorithm/NS/production/NS-pipeline.py' --data_folder '/mnt/imported/data/NOPRODNA0029/for_s3' --target_folder '/mnt/data/Nocturnal-scratch/geneactive_20hz_3s_b1s_production_train'  --plot --write_raw  

# python '/mnt/imported/code/nocturnal-scratch-algorithm/NS/production/NS-pipeline.py' --data_folder '/mnt/imported/data/NOPRODNA0029/for_s3' --target_folder '/mnt/data/Nocturnal-scratch/geneactive_20hz_3s_b1s_production_test_scratch' --production --scratch --plot --scratch_sleep_only --write_raw  --write_motion_only

#DUPLEX

# python '/mnt/code/munge/preprocessing/NS-pipeline.py' --data_folder '/mnt/data/Duplex_AD_UAT' --target_folder '/mnt/data/Nocturnal-scratch/DUPLEX_UAT_test_scratch' --calibrate_gravity --production --source DUPLEX --sf 32 --resample 20 --scratch --plot  

# python '/mnt/imported/code/nocturnal-scratch-algorithm/NS/production/NS-pipeline.py' --data_folder '/mnt/data/Duplex_AD_UAT' --target_folder '/mnt/data/Nocturnal-scratch/DUPLEX_UAT_test_scratch' --production --source DUPLEX --sf 32 --resample 20 --scratch --plot --scratch_sleep_only --num_gpu 3

# DECODE
# python '/mnt/imported/code/nocturnal-scratch-algorithm/NS/production/NS-pipeline.py' --data_folder '/mnt/data/DECODE-Analytical-Validation/data/leap/Ametris/data/raw' --TZ_file '/mnt/data/DECODE-Analytical-Validation/data/leap/Aug_2025/ActiGraph LEAP 2/Outcomes Data/Milestone 4 - 26Aug25/DECODE_Subject_Details 26Aug25.csv' --target_folder '/mnt/data/DECODE-Analytical-Validation/DeepScratch_output' --production --source DECODE --sf 32 --resample 20 --scratch --plot --scratch_sleep_only --num_gpu 0

args = parser.parse_args()
target_folder = args.target_folder
GT= "/mnt/data/Ground-truth/scratch_GT_offset_imerit.csv"
i=0

target_folder_raw= os.path.join(target_folder, "raw/")
target_folder_analytics= os.path.join(target_folder, "analytics/")
target_folder_analytics_sleep= os.path.join(target_folder, "analytics/sleep/")
target_folder_analytics_activity= os.path.join(target_folder, "analytics/activity/")
target_folder_analytics_scratch= os.path.join(target_folder, "analytics/scratch/")
target_folder_analytics_plots= os.path.join(target_folder, "analytics/plots/")
target_folder_metadata= os.path.join(target_folder, "metadata/")
target_folder_tracking= os.path.join(target_folder, "metadata/job_tracking/")




create_folder([target_folder_raw,target_folder_analytics,target_folder_metadata,target_folder_tracking,target_folder_analytics_sleep,target_folder_analytics_activity,target_folder_analytics_plots,target_folder_analytics_scratch])


logger=create_logger('NS_logger',os.path.join(target_folder_metadata, f"logger_{str(datetime.now())}.log"))

    
Start_time = datetime.now()

# =============================================================================
# Process Geneactive files to seperate data by days and store the processed data incrementally
# =============================================================================


def get_scratch_gt(GT): # TODO: add this function to helpers
    #Load scratch events
    GT_df = pd.read_csv(GT,encoding='utf-8')   
    GT_df[['Start_tz_leftoffset','End_tz_leftoffset','Start_tz_rightoffset','End_tz_rightoffset','TSOSTART','TSOEND']]=GT_df[['Start_tz_leftoffset','End_tz_leftoffset','Start_tz_rightoffset','End_tz_rightoffset','TSOSTART','TSOEND']].apply(lambda x: pd.to_datetime(x))
    return GT_df

def plot_nights(df,DATE,args,sample_rate):
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig, axs = plt.subplots(4, 1, figsize=(12, 8),num=1,clear=True) #num=1,
        fig.suptitle(f"Sleep data from {DATE}", fontsize=10)
        down=sample_rate*60
        idx=df.timestamp[::down]
        axs[0].plot(idx,df['x'][::down],label='x')
        axs[0].plot(idx,df['y'][::down],label='y')
        axs[0].plot(idx,df['z'][::down],label='z')
        axs[1].plot(idx,df['temperature'][::down],label='Temperature',color='black',linewidth=2.0)
        axs[2].plot(idx,df['non-wear'][::down],label='Non-wear',color='black',linewidth=2.0)
        axs[3].plot(idx,df['predictTSO'][::down],label='predicted TSO',color='orange',linewidth=3.0)
        if not args.production:
            axs[3].plot(idx,df['inTSO'][::down],label='Reference TSO',color='black',linewidth=3.0, linestyle='dotted',)

        for ax in axs:
            ax.label_outer()
            ax.legend(loc="right",fontsize=8)
            ax.xaxis.set_major_locator(mdates.HourLocator()) 

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S')) 
        #plt.gca().xaxis.set_tick_params(rotation = 90)
        fig.autofmt_xdate()
    return fig

def convert_float32(obj):
    if isinstance(obj, list):
        return [convert_float32(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_float32(v) for k, v in obj.items()}
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj
    
# @profile
def add_GT_imerit(data,GT_PID,wrist):
#     _data = _data.copy(deep=True)
    data.loc[:,'inTSO']=False
    data.loc[:,['SCRATCHSTART','SCRATCHEND','ScratchedUsingUpperExtremity','UsageLabeler','MotionPeriodIdentifier','SkinImpactLabeler1','SkinImpactLabeler2','LowConfLabeler1','LowConfLabeler2','TSOSTART','TSOEND']]=None
    #Filter data inTSO only
    for index, row in GT_PID[['TSOSTART','TSOEND']].drop_duplicates().iterrows():
        idx2 = data.index[(data.index >=row['TSOSTART'])& (data.index <= row['TSOEND'])]
        data.loc[idx2,"inTSO"]=True
        data.loc[idx2,['TSOSTART','TSOEND']]=[row['TSOSTART'],row['TSOEND']]

    if len(data)>0:
        for index, row in GT_PID.iterrows():
            if wrist=="right":
                idx = data.index[(data.index >= row['Start_tz_rightoffset'])& (data.index <=row['End_tz_rightoffset'])]
                UsageLabeler='RUEUsage'
                LowConfLabeler1='RUEUsageLowConfLabeler1'
                LowConfLabeler2='RUEUsageLowConfLabeler2'
                startscratch='Start_tz_rightoffset'
                endscratch='End_tz_rightoffset'
            else:
                idx = data.index[(data.index >=row['Start_tz_leftoffset'])& (data.index <= row['End_tz_leftoffset'])]
                UsageLabeler='LUEUsage'
                LowConfLabeler1='LUEUsageLowConfLabeler1'
                LowConfLabeler2='LUEUsageLowConfLabeler2'
                startscratch='Start_tz_leftoffset'
                endscratch='End_tz_leftoffset'

            data.loc[idx,['SCRATCHSTART','SCRATCHEND','ScratchedUsingUpperExtremity','UsageLabeler','MotionPeriodIdentifier','SkinImpactLabeler1','SkinImpactLabeler2','LowConfLabeler1','LowConfLabeler2']]=[row[startscratch],row[endscratch],row['ScratchedUsingUpperExtremity'],row[UsageLabeler],row['MotionPeriodIdentifier'],row['SkinImpactLabeler1'],row['SkinImpactLabeler2'],row[LowConfLabeler1],row[LowConfLabeler2]]

        data.loc[:,"scratch"]=False
        data.loc[data["UsageLabeler"]=="WRIST","scratch"]=True

    else:
        print_status("File Skipped because file is empty after filteing inTSO.",2,logger)
        data= pd.dataFrame() 
#     return data
        

# @profile
def NORPOD_processor(file,PID,wrist,i,size,GT_df,args,model,device,batch_size):
    #We change type of x,y,z, and temperature to float32 to save memory
    data=pd.read_csv(
        file,
        usecols=["timestamp", "x", "y", "z", "temperature"],dtype={
                "timestamp": np.float64,
                "x": np.float32,
                "y": np.float32,
                "z": np.float32,
                "temperature": np.float32
            }) 

    data["timestamp"]=pd.to_datetime(data["timestamp"],unit='s')#.dt.tz_localize('UTC')      
    data.set_index('timestamp',inplace=True)
    data.sort_index(inplace=True)

    process_file(data,PID,wrist,i,size,GT_df,args,model,device,batch_size)
    del data
    gc.collect()
    
def DUPLEX_processor(group,site,PID,wrist,TZ,i,size,GT_df,args,model,device,batch_size):
    acc_data_list=[]
    temperature_data_list=[]
    for f in group.file_path:
        if args.source=="DECODE":
            if 'temperature' in f:
                temperature_data_list.append(pd.read_csv(f,usecols=["Timestamp",'TemperatureCelsius',"MonitorSerial"]))
            else:
                acc_data_list.append(pd.read_csv(f,usecols=["Timestamp",'SampleOrder', "X", "Y", "Z","MonitorSerial"]))
        else:
            if 'temperature' in f:
                temperature_data_list.append(pd.read_csv(f,usecols=["Timestamp",'TemperatureCelsius',"MonitorSerial"]))
            else:
                acc_data_list.append(pd.read_csv(f,usecols=["Timestamp",'SampleOrder', "X", "Y", "Z","MonitorSerial"]))

    acc_data=pd.concat(acc_data_list,ignore_index=True)
    temperature_data=pd.concat(temperature_data_list,ignore_index=True)

    data=acc_data.merge(temperature_data,how='left')
    data['timestamp']= pd.to_datetime(data['Timestamp'], unit='s').dt.tz_localize('UTC')+ pd.to_timedelta((data.SampleOrder * (1/args.sf)), unit='s')
    #change TZ
    data['timestamp'] = data['timestamp'].dt.tz_convert(TZ)
    #Drop the TZ part of the timestamp, otherwise the time daily grouper would have an expected behavior (grouping using UTC instead of local timezone)
    data['timestamp'] = data['timestamp'].dt.tz_localize(None)
    data.drop(['SampleOrder','Timestamp'], axis=1,inplace=True)
    data.rename(columns={'X':'x','Y':'y','Z':'z','MonitorSerial':'deviceID','TemperatureCelsius':'temperature'},inplace=True)
    data.temperature.bfill(inplace=True)
    data.temperature.ffill(inplace=True)
    data.set_index('timestamp',inplace=True)
    data.sort_index(inplace=True)
    
    PID=site+PID
    process_file(data,PID,wrist,i,size,GT_df,args,model,device,batch_size)
    
    del acc_data,temperature_data,acc_data_list,temperature_data_list
    gc.collect()

def DECODE_processor(group,site,PID,wrist,TZ,i,size,GT_df,args,model,device,batch_size):
    acc_data_list=[]
    temperature_data_list=[]
    for f in group.file_path:
        if 'temperature' in f:
            temperature_data_list.append(pd.read_csv(f,usecols=["Timestamp Unix",'TemperatureCelsius',"MonitorSerial"]))
        else:
            acc_data_list.append(pd.read_csv(f,usecols=["Timestamp Unix",'SampleOrder', "X", "Y", "Z","MonitorSerial"]))
            
    acc_data=pd.concat(acc_data_list,ignore_index=True)
    temperature_data=pd.concat(temperature_data_list,ignore_index=True)

    data=acc_data.merge(temperature_data,how='left')
    data['timestamp']= pd.to_datetime(data['Timestamp Unix'], unit='s').dt.tz_localize('UTC')+ pd.to_timedelta((data.SampleOrder * (1/args.sf)), unit='s')
    #change TZ
    data['timestamp'] = data['timestamp'].dt.tz_convert(TZ)
    #Drop the TZ part of the timestamp, otherwise the time daily grouper would have an expected behavior (grouping using UTC instead of local timezone)
    data['timestamp'] = data['timestamp'].dt.tz_localize(None)
    data.drop(['SampleOrder','Timestamp Unix'], axis=1,inplace=True)
    data.rename(columns={'X':'x','Y':'y','Z':'z','MonitorSerial':'deviceID','TemperatureCelsius':'temperature'},inplace=True)
    data.temperature.bfill(inplace=True)
    data.temperature.ffill(inplace=True)
    data.set_index('timestamp',inplace=True)
    data.sort_index(inplace=True)
    
    PID=site+PID
    process_file(data,PID,wrist,i,size,GT_df,args,model,device,batch_size)
    
    del acc_data,temperature_data,acc_data_list,temperature_data_list
    gc.collect()
    
# @profile
def main_processor(GT_df,args,model,device,batch_size,day_gaps=3):
    # =============================================================================
    # NOPROD
    # =============================================================================
    if args.source == "NOPROD":
        filenames = sorted(os.path.join(dp, f) for dp, dn, filenames in os.walk(args.data_folder) for f in filenames if f.endswith('.csv'))
        size=len(filenames)
        i=0
        shuffle(filenames)
        for file in filenames:
            i+=1
            file_name=os.path.basename(file).replace(".csv","")
            PID=file_name.split("_")[0]
            wrist = file_name.split("_")[1]
            in_progress_subjects = list(os.listdir(target_folder_tracking))
            tracker=PID+"_"+wrist
            #TODO: skip for missing acc or temp files
            if (tracker in in_progress_subjects):
                m="File " + str(i) + " out of " + str(size) + ': Processed subject '+str(tracker) + ')' 
                print_status(f"{m} Skipped because it has already been processed or is being currently processed by another job.",2,logger)
            else:
                with open(os.path.join(target_folder_tracking, tracker), "w") as f:
                    f.write("-")  # placeholder file
                print_status(f"Processing {tracker} (progress: {str(i)}/{size}) :",1,logger)
                p = multiprocessing.Process(target=NORPOD_processor, args=(file,PID,wrist,i,size,GT_df,args,model,device,batch_size))
                p.start()
                p.join()
                gc.collect()
        del filenames
        gc.collect()

        
    # =============================================================================
    # DUPLEX
    # =============================================================================    
    elif args.source=="DUPLEX":
        filenames = sorted(os.path.join(dp, f) for dp, dn, filenames in os.walk(args.data_folder) for f in filenames if (f.endswith('.csv')) and (('raw-accel' in f) or ( 'temperature' in f )) and ('checkpoint' not in f))
        if args.TZ_file=="":
            subjects_file = sorted(os.path.join(dp, f) for dp, dn, filenames in os.walk(args.data_folder) for f in filenames if (f.endswith('subjects.csv')) and ('checkpoint' not in f))
            TZ_df=pd.read_csv(subjects_file[0], dtype=str)
        else:
            TZ_df=pd.read_csv(args.TZ_file, dtype=str)
            
        file_df=[]
        # Regex pattern to extract wrist and ID
        pattern = r'DUPLEX - (.+?) Wrist.*?/([^/]+)/([^/]+)/[^/]*?(\d{4}-\d{2}-\d{2})_'

        for file_path in filenames:
            match = re.search(pattern, file_path)
            if match:
                wrist = match.group(1) 
                site = match.group(2).split(' - ')[0] # remove the country part from site ID
                ID = match.group(3)       
                date=match.group(4)
                file_df.append({'wrist':wrist,'site':site,'ID': ID,'date': date, 'file_path': file_path})

        file_df=pd.DataFrame(file_df)
        file_df['date']=pd.to_datetime(file_df['date'], errors='coerce')
        file_df = file_df.sort_values(['site','ID','wrist','date'])
        groups=file_df.groupby(['site','ID','wrist'])
        i=0
        size=len(groups)
        for idx,group in groups:
            site=idx[0]
            PID=idx[1]
            wrist=idx[2]
            print(site,PID,wrist)
            TZ=TZ_df.loc[(TZ_df.siteIdentifier==site)&(TZ_df.subjectIdentifier.astype(str)==PID),'timezone']
            if len(TZ)==0:
                print_status(f"Timezone record not found: {site} ,{PID},{wrist}",2,logger)
            else:
                TZ=TZ.values[0]
                in_progress_subjects = list(os.listdir(target_folder_tracking))
                tracker=site+"_"+PID+"_"+wrist
                i+=1
                if (tracker in in_progress_subjects):
                    m="File " + str(i) + " out of " + str(size) + ': Processed subject '+str(tracker) + ')' 
                    print_status(f"{m} Skipped because it has already been processed or is being currently processed by another job.",2,logger)
                else:
                    with open(os.path.join(target_folder_tracking, tracker), "w") as f:
                        f.write("-")  # placeholder file

                    print_status(f"Processing {tracker} (progress: {str(i)}/{str(size)}) :",1,logger)

                    days=group.drop_duplicates(subset=['site','wrist','ID','date']).copy()
                    diff=days['date'].diff().dt.days
                    diff = (diff > day_gaps).cumsum()
                    days.loc[:,'diff']=diff
                    days=days[['site','wrist','ID','date','diff']]
                    group_diff=group.merge(days,how='left')
                    for idx,mini_group in group_diff.groupby(['site','ID','wrist','diff']):
                        print_status(f"Processing {tracker}. Data batch between {mini_group.date.dt.date.min()} and {mini_group.date.dt.date.max()}",2,logger)
                        mini_group.loc[:,'count']=mini_group.groupby(['wrist','ID','date'])['date'].transform('count')
                        if len(mini_group[mini_group['count']==1])>0:
                            print_status(f"The following  days are excluded from processing because they are missing either accelerometer or temperature files {mini_group[mini_group['count']==1].date.dt.date.astype(str).values}",2,logger)

                        p = multiprocessing.Process(target=DUPLEX_processor, args=(mini_group[mini_group['count']>1],site,PID,wrist,TZ,i,size,GT_df,args,model,device,batch_size))
                        p.start()
                        p.join()
        del filenames
        gc.collect()
        
    # =============================================================================
    # DECODE
    # =============================================================================    
    elif args.source=="DECODE":
        filenames = sorted(os.path.join(dp, f) for dp, dn, filenames in os.walk(args.data_folder) for f in filenames if (f.endswith('.csv')) and ('subject.' in dp) and (('raw-accel' in f) or ( 'temperature' in f )) and ('checkpoint' not in f))
        if args.TZ_file=="":
            subjects_file = sorted(os.path.join(dp, f) for dp, dn, filenames in os.walk(args.data_folder) for f in filenames if (f.endswith('subjects.csv')) and ('checkpoint' not in f))
            TZ_df=pd.read_csv(subjects_file[0], dtype=str)
        else:
            TZ_df=pd.read_csv(args.TZ_file, dtype=str)


        file_df=[]
        for file_path in filenames:
            try:
                date=file_path.split('/')[-1].split('_')[1]
                ID=file_path.split('/')[-2]
                wrist=file_path.split('/')[-2].split('_')[1]
                site=""
                file_df.append({'wrist':wrist,'site':site,'ID': ID,'date': date, 'file_path': file_path})
            except:
                pass
        file_df=pd.DataFrame(file_df)
        file_df['date']=pd.to_datetime(file_df['date'], errors='coerce')
        file_df = file_df.sort_values(['site','ID','wrist','date'])
        groups=file_df.groupby(['site','ID','wrist'])
        i=0
        size=len(groups)
        for idx,group in groups:
            site=idx[0]
            PID=idx[1]
            wrist=idx[2]
            TZ=TZ_df.loc[(TZ_df.subjectIdentifier.astype(str)==PID),'timezone']
            if len(TZ)==0:
                print_status(f"Timezone record not found: {site} ,{PID},{wrist}",2,logger)
            else:
                TZ=TZ.values[0]
                in_progress_subjects = list(os.listdir(target_folder_tracking))
                tracker=site+"_"+PID+"_"+wrist
                i+=1
                if (tracker in in_progress_subjects):
                    m="File " + str(i) + " out of " + str(size) + ': Processed subject '+str(tracker) + ')' 
                    print_status(f"{m} Skipped because it has already been processed or is being currently processed by another job.",2,logger)
                else:
                    with open(os.path.join(target_folder_tracking, tracker), "w") as f:
                        f.write("-")  # placeholder file

                    print_status(f"Processing {tracker} (progress: {str(i)}/{str(size)}) :",1,logger)

                    days=group.drop_duplicates(subset=['site','wrist','ID','date']).copy()
                    diff=days['date'].diff().dt.days
                    diff = (diff > day_gaps).cumsum()
                    days.loc[:,'diff']=diff
                    days=days[['site','wrist','ID','date','diff']]
                    group_diff=group.merge(days,how='left')
                    for idx,mini_group in group_diff.groupby(['site','ID','wrist','diff']):
                        print_status(f"Processing {tracker}. Data batch between {mini_group.date.dt.date.min()} and {mini_group.date.dt.date.max()}",2,logger)
                        mini_group.loc[:,'count']=mini_group.groupby(['wrist','ID','date'])['date'].transform('count')
                        if len(mini_group[mini_group['count']==1])>0:
                            print_status(f"The following  days are excluded from processing because they are missing either accelerometer or temperature files {mini_group[mini_group['count']==1].date.dt.date.astype(str).values}",2,logger)

                        p = multiprocessing.Process(target=DECODE_processor, args=(mini_group[mini_group['count']>1],site,PID,wrist,TZ,i,size,GT_df,args,model,device,batch_size))
                        p.start()
                        p.join()

        del filenames
        gc.collect()

    else:
        print_status(f"Data source type not found.",1,logger)

# =============================================================================
# Process the data to calibrate it, filter it, resample it (if needed), and extract sleep and motion periods.
# =============================================================================
# @profile
def process_file(data,PID,wrist,i,size,GT_df,args,model,device,batch_size):
    tracker=PID+"_"+wrist
    Start_time_file = datetime.now()
    sample_rate=args.sf
    print_status(f"Reading the data...",2,logger)
    if not args.production:
        GT_PID =GT_df.loc[(GT_df["ParticipantIdentifier"]==PID) & ~(GT_df.TSOSTART.isna())]
        if len (GT_PID)==0:
            print_status(f"{tracker} skipped because no GT was found for this subject {PID}.",2,logger)
            return
    #Check and remove duplicated index
    if data.index.has_duplicates:
        print_status(f"{tracker} has duplicated index. Removing duplicates.",2,logger)
        data=data[~data.index.duplicated(keep='first')]
    # Instantiate the data preprocessor class with parameters
    data_processor = p.DataPreprocessor(
        sample_rate=sample_rate,
        filter_cutoff=(args.highpass, args.lowpass),
        calibrate_gravity=args.skip_gravity_calibration,
        detect_nonwear=True,
        resample_hz=args.resample,
        method='FFT',
        detect_TSO=True,
        tso_algorithm=args.TSO_algo,
        nonwear_algorithm=args.nonwear_algo,
        day_start=args.day_start
    )
    if args.TSO_algo == 'dl':
        if args.tso_model_path is None:
            raise ValueError("--tso_model_path is required when --TSO_algo dl is used.")
        data_processor.tso_model_path = args.tso_model_path
        data_processor.tso_model_name = args.tso_model_name
        data_processor.tso_device = args.tso_device
        tso_scaler_file = os.path.join(script_dir, 'model', f'{args.tso_model_name}_scaler.joblib')
        data_processor.tso_scaler_path = tso_scaler_file if os.path.exists(tso_scaler_file) else None
    # Prepare the info dictionary and call the process method. We also get back the sample rate in case it was resampled
    info = {}
    data,info=data_processor.process(data, info, logger=logger)
    sample_rate = data_processor.get_samplerate()
    with open(os.path.join(target_folder_metadata, "Processed_" + tracker+"_info.json"), 'w') as json_file:
        json.dump(convert_float32(info), json_file, indent=4)
    
    # Initiate the file that will host all the nightly plots
    if args.plot:
        plots_pdf = PdfPages(os.path.join(target_folder_analytics_plots, f"Sleep_TSO_{PID}_{wrist}.pdf"))

    # If dev mode, add ground truth
    if not args.production:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") 
            print_status("Adding ground truth....",2,logger) 
            add_GT_imerit(data,GT_PID,wrist)
    data['wrist']=wrist
    data['SUBJECT']=PID
    # Add columns that will be populated when splitting the data. Adding before the loop otherwise it would force pandas to copy the data.
    if args.write_raw:
        data.loc[:,'predictTSOSTART']=None
        data.loc[:,'predictTSOEND']=None
    if args.write_raw or args.scratch:
        data.loc[:,'position']=None
        data.loc[:,'SEGMENT_END']=None
        #TODO check
        if 'deviceID' not in data.columns:
            data.loc[:,'deviceID']=None
    if args.scratch:
        data.loc[:,'intensity']=0.0
        data['DATE']=pd.NaT
        
        
    # =============================================================================
    # Process the data per night
    # =============================================================================
    print_status("Spliting data to days....",2,logger)   
    if len(data)>0:
        data.reset_index(inplace=True)
        #Group by night using the offset specified in args. Default is 18h. 
        for d,df in data.groupby(pd.Grouper(key="timestamp",freq="d",offset=f"{args.day_start}h00min")): 
            #DATE is the end date.
            DATE=(d+pd.Timedelta(days=1)).date()
            tso_condition=False
            if len(df)==0: # TODO: to remove
                print_status(f"Empty data {str(DATE)}",2,logger)
            else:
                if (not args.production):
                    #If in dev mode, process the data if there is ground truth sleep in this day.
                    if True in df["inTSO"].values:
                        tso_condition=True
                    else:
                        print_status(f"No Ground truth in {str(d)}",2,logger)     
                else:
                    #In production mode, we process the data only if there is data is more than the minimum threshold, non-wear is more than the minimum sleep, and sleep is detected in this night.
                    if ((df.timestamp.max() - df.timestamp.min()).total_seconds()/3600) >= args.min_data:
                        if len(df[~df['non-wear']])>=(args.min_sleep*(60*60*sample_rate)):     
                            # if (True in df["predictTSO"].values):
                            tso_condition=True
                            
                        else:
                            ALG_NOTE=f"Wear time is less than the minimum {args.min_sleep} hours sleep threshold on date {str(DATE)} Data."
                            print_status(ALG_NOTE,2,logger)
                            pd.DataFrame({'SUBJECT':[PID],'DATE':[DATE],'WRIST':[wrist],'ALG_TSO_START':[np.nan],'ALG_TSO_END':[np.nan],'ALG_TSO_DURATION':[np.nan],'ALG_NOTE':[ALG_NOTE]}).to_csv(os.path.join(target_folder_analytics_sleep, f"Sleep_TSO_{PID}_{wrist}_{DATE}.csv"),index=False)
                            if args.scratch:
                                pd.DataFrame({'SUBJECT':[PID],'DATE':[DATE],'WRIST':[wrist],'DEVICE_ID':str(df.deviceID.values[0]),'ALG_TSO_START':[np.nan],'ALG_TSO_END':[np.nan],'ALG_TSO_DURATION':[np.nan],'ALG_SCRATCH_FREQUENCY':[np.nan],'ALG_SCRATCH_DURATION':[np.nan],'ALG_NOTE':[ALG_NOTE]}).to_csv(os.path.join(target_folder_analytics_scratch, f"Scratch_Daily_Summary_{PID}_{wrist}_{DATE}.csv"),index=False)
                            
                    else:
                        #Not enough data in this day, skip and write empty sleep measures while adding a note.
                        ALG_NOTE=f"Data is less than the minimum {args.min_data} hours threshold on date {str(DATE)}."
                        pd.DataFrame({'SUBJECT':[PID],'DATE':[DATE],'WRIST':[wrist],'ALG_TSO_START':[np.nan],'ALG_TSO_END':[np.nan],'ALG_TSO_DURATION':[np.nan],'ALG_NOTE':[ALG_NOTE]}).to_csv(os.path.join(target_folder_analytics_sleep, f"Sleep_TSO_{PID}_{wrist}_{DATE}.csv"),index=False)
                        print_status(ALG_NOTE,2,logger)
                        if args.scratch:
                            pd.DataFrame({'SUBJECT':[PID],'DATE':[DATE],'WRIST':[wrist],'DEVICE_ID':str(df.deviceID.values[0]),'ALG_TSO_START':[np.nan],'ALG_TSO_END':[np.nan],'ALG_TSO_DURATION':[np.nan],'ALG_SCRATCH_FREQUENCY':[np.nan],'ALG_SCRATCH_DURATION':[np.nan],'ALG_NOTE':[ALG_NOTE]}).to_csv(os.path.join(target_folder_analytics_scratch, f"Scratch_Daily_Summary_{PID}_{wrist}_{DATE}.csv"),index=False)
                    
                    
                if tso_condition:  
                    # =============================================================================
                    # Sleep periods
                    # =============================================================================
                    if True in df["predictTSO"].values:
                        sleep_periods=(df["predictTSO"] != df["predictTSO"].shift())
                        sleep_periods=(sleep_periods).cumsum()
                        TSO_IDs = sleep_periods[df["predictTSO"].values]
                        TSO_IDs = TSO_IDs[TSO_IDs==TSO_IDs.value_counts().idxmax()]
                        df.loc[:,'predictTSO']=False
                        df.loc[(df.index>= TSO_IDs.index.min())&(df.index<= TSO_IDs.index.max()),'predictTSO']  = True #TODO:to check if we should use timestamp instead
                        df.loc[:,'predictTSO'] = df.groupby('segment')['predictTSO'].transform('all') # A segment is considered in TSO only if it is completly included in the TSO
                        ALG_TSO_START=df.loc[df.predictTSO].timestamp.min()
                        ALG_TSO_END=df.loc[df.predictTSO].timestamp.max()
                        ALG_TSO_DURATION= (ALG_TSO_END-ALG_TSO_START).total_seconds() / 3600
                        ALG_NOTE=np.nan
                        del sleep_periods,TSO_IDs
                        gc.collect()
                    else:
                        ALG_TSO_START=np.nan
                        ALG_TSO_END= np.nan
                        ALG_TSO_DURATION= np.nan
                        ALG_NOTE=f"No detected total sleep opportunity window on date {str(DATE)}"
                        print_status(ALG_NOTE,2,logger)
                        if args.scratch:
                            pd.DataFrame({'SUBJECT':[PID],'DATE':[DATE],'WRIST':[wrist],'DEVICE_ID':str(df.deviceID.values[0]),'ALG_TSO_START':[np.nan],'ALG_TSO_END':[np.nan],'ALG_TSO_DURATION':[np.nan],'ALG_SCRATCH_FREQUENCY':[np.nan],'ALG_SCRATCH_DURATION':[np.nan],'ALG_NOTE':[ALG_NOTE]}).to_csv(os.path.join(target_folder_analytics_scratch, f"Scratch_Daily_Summary_{PID}_{wrist}_{DATE}.csv"),index=False)
                    
                    df_sleep_TSO={'SUBJECT':[PID],'DATE':[DATE],'WRIST':[wrist],'ALG_TSO_START':[ALG_TSO_START],'ALG_TSO_END':[ALG_TSO_END],'ALG_TSO_DURATION':[ALG_TSO_DURATION],'ALG_NOTE':[ALG_NOTE]}
                    #If in dev mode, add ground truth TSO
                    if not args.production:
                        df_sleep_TSO.update({'TSOSTART':[df.loc[df['inTSO'] == True].timestamp.min()],'TSOEND':[df.loc[df['inTSO'] == True].timestamp.max()]})

                    pd.DataFrame(df_sleep_TSO).to_csv(os.path.join(target_folder_analytics_sleep, f"Sleep_TSO_{PID}_{wrist}_{DATE}.csv"),index=False)
                    
                    #Add segment and position columns.
                    if args.write_raw or args.scratch:
                        df.loc[:,'segment'] = (PID+'_'+df['wrist'].astype(str)+'_'+str(DATE)+'_'+df['segment'].astype(str)).values 
                        df.loc[:,'position'] = df.groupby('segment').cumcount() + 1
                        

                    #Save the data to parquet.
                    if args.write_raw:
                        df.loc[:,'predictTSOSTART']=ALG_TSO_START
                        df.loc[:,'predictTSOEND']=ALG_TSO_END
                        print(f"[PRE-SAVE DEBUG] tracker={tracker} date={str(DATE).split(' ')[0]} "
                              f"rows={len(df)} "
                              f"x mean={df['x'].mean():.3f} std={df['x'].std():.3f} "
                              f"temp range=[{df['temperature'].min():.2f},{df['temperature'].max():.2f}] "
                              f"predictTSO={df['predictTSO'].sum()} samples True")
                        processed_file=os.path.join(target_folder_raw, "Processed_" + tracker +"_"+str(DATE).split(" ")[0]+".parquet.gzip")
                        
                        #If write_motoin_only is selected, write motion data only, otherwise write all data
                        if args.write_motion_only:
                            if (not args.production):
                                df.loc[(df.stationary==False) & (df.predictTSO | df.inTSO), df.columns != 'predictTSO_group'].to_parquet(processed_file, index=True)
                            else:
                                df.loc[df.stationary==False, df.columns != 'predictTSO_group'].to_parquet(processed_file, index=True)
                        else:
                            df.loc[:, df.columns != 'predictTSO_group'].to_parquet(processed_file, index=False)
                    # =============================================================================df
                    # Naucturnal Scratch
                    # ============================================================================= 
                    if (args.scratch) & (True in df["predictTSO"].values):
                        print_status(f"Predicting scratch for {str(DATE)}",2,logger)   
                        #Measure vectore magnitude (VM) before scaling the data. VM is used to measure scratch intensity.

                        df.loc[:,'intensity']= np.sqrt(df.x**2 + df.y**2 + df.z**2)
                        # The DL ResMamba models expect z-scored x/y/z (their
                        # training pipeline ran StandardScaler before the
                        # mixed-frequency feature extractor).  Classical
                        # baselines run their own HPF + per-window feature
                        # extraction inside the ClassicalBackend; pre-scaling
                        # raw inputs would corrupt the FFT-derived features
                        # and the variance-sensitive mean-cross-rate / SPARC
                        # statistics.  Apply the scaler only on the DL path.
                        if not isinstance(model, ClassicalBackend):
                            pre_saved_scaler = joblib.load(os.path.join(script_dir, 'model', f'{args.scratch_model}_scaler.joblib'))
                            c_to_scale=['x', 'y', 'z']
                            df.loc[:, c_to_scale] = pre_saved_scaler.transform(df[c_to_scale])
                        #change to seconds
                        df.loc[:,'segment_duration']=df.segment_duration/sample_rate
                        with torch.no_grad():
                            model.eval()
                            if args.scratch_sleep_only:
                                predictions=run_model(model,df.loc[(df.stationary==False)&(df.segment_duration>5)&(df.predictTSO==True)],batch_size,device,stratify=False)
                                df_predict=pd.merge(df[df.predictTSO], predictions, on=['segment','position'],how="left")
                            else:
                                predictions_inTSO=run_model(model,df.loc[(df.stationary==False)&(df.segment_duration>5)&(df.predictTSO==True)],batch_size,device,stratify=False)
                                predictions_outTSO=run_model(model,df.loc[(df.stationary==False)&(df.segment_duration>5)&(df.predictTSO==False)],batch_size,device,stratify=False)
                                predictions=pd.concat([predictions_inTSO, predictions_outTSO], ignore_index=True)
                                df_predict=pd.merge(df, predictions, on=['segment','position'],how="left")
                        
                        df_predict.loc[:,['pr1','pr2','pr3','pr1_probs','pr2_probs']]=df_predict[['pr1','pr2','pr3','pr1_probs','pr2_probs']].fillna(0)
                        df_predict.loc[df_predict['pr1_probs']<0.6,['pr1']]=0
                        

                        df_predict.loc[df_predict.pr1==0,['pr3','pr2']]=0 # If the predicted scratch is False then scratch duration should be zero
                        df_predict.loc[df_predict.pr3<0,'pr3']=0 # If the predicted duration is negative, set it to zero
                        df_predict.loc[(df_predict.pr1==0)|(df_predict.pr2==0),'intensity']=0 #change intensity to 0 when there is no predict scratch or when mask is 0.
                        
                        #Transform prediction of duration to seconds
                        df_predict.loc[:,'pr3']=df_predict['pr3']*df_predict['segment_duration']
                        df_predict.loc[:,'intensity']=df_predict.groupby('segment')['intensity'].transform('sum') #sum of SVM, only when pr1 and pr2 are 1
                        df_predict.loc[df_predict['intensity']==0,['pr1','pr3']]=0 #If intensity is 0, meaning no mask is predicted, make pr1 and pr3 0
                        
                        sum_mask=df_predict.groupby('segment',sort=False)['pr2'].transform('sum')/sample_rate #duration of predicted scratch from the mask
                        df_predict.loc[:,'intensity']=df_predict['intensity']/sum_mask
                        df_predict['intensity'] = df_predict['intensity'].replace([np.inf, -np.inf], np.nan).fillna(0) #inensity can be na if no mask is predicted. Replace with 0
                        
                        
                        df_predict['predictTSOSTART']=ALG_TSO_START
                        df_predict['predictTSOEND']=ALG_TSO_END
                        df_predict['predictTSO_duration']=ALG_TSO_DURATION
                        
                        segments_df=df_predict.drop_duplicates(subset='segment',keep='first')
                        
                        segments_df.loc[:,'DATE']=DATE
                        segments_df.loc[:,'SEGMENT_END']=segments_df.timestamp+pd.to_timedelta((segments_df.segment_duration),unit='s')
                        segments_df.loc[:, segments_df.select_dtypes(include=['float']).columns] = segments_df.select_dtypes(include=['float']).round(4)
                        SCRATCH_BOUT_df=segments_df[['SUBJECT','DATE','wrist','predictTSOSTART', 'predictTSOEND','predictTSO_duration','predictTSO','segment','timestamp','SEGMENT_END','pr1','pr3','intensity','pr1_probs', 'segment_duration','stationary','deviceID']]
                        SCRATCH_BOUT_df.columns=['SUBJECT','DATE','WRIST','ALG_TSO_START', 'ALG_TSO_END','ALG_TSO_DURATION','ALG_inTSO','SEGMENT_ID','SEGMENT_START','SEGMENT_END','ALG_SCRATCH','ALG_SCRATCH_DURATION','ALG_SCRATCH_INTENSITY','ALG_SCRATCH_PROBABILITY','SEGMENT_DURATION','STATIONARY','DEVICE_ID']
                            
                        SCRATCH_BOUT_df.to_csv(os.path.join(target_folder_analytics_scratch, f"Scratch_Bout_{PID}_{wrist}_{DATE}.csv"),index=False)

                        scratch_frequency_nightly=segments_df[segments_df["predictTSO"]==True].pr1.sum()/(ALG_TSO_DURATION)
                        scratch_duration_nightly=segments_df[segments_df["predictTSO"]==True].pr3.sum()/(ALG_TSO_DURATION) #Seconds per hour


                        pd.DataFrame({'SUBJECT':[PID],'DATE':[str(segments_df.DATE[0])],'WRIST':[str(segments_df.wrist[0])],'DEVICE_ID':str(segments_df.deviceID[0]),'ALG_TSO_START':[ALG_TSO_START],'ALG_TSO_END':[ALG_TSO_END],'ALG_TSO_DURATION':[ALG_TSO_DURATION],'ALG_SCRATCH_FREQUENCY':[scratch_frequency_nightly],'ALG_SCRATCH_DURATION':[scratch_duration_nightly],'ALG_NOTE':[np.nan]}).round(4).to_csv(os.path.join(target_folder_analytics_scratch, f"Scratch_Daily_Summary_{PID}_{wrist}_{DATE}.csv"),index=False)
                                                 
                        del SCRATCH_BOUT_df,segments_df,df_predict,predictions
                        gc.collect()
                    
                #Plot the data. The data is downsampled to minute level to reduce overhead when plotting using Matplotlib. We are plotting the data regardless of condition.
                if args.plot:
                    plots_pdf.savefig(plot_nights(df,DATE,args,sample_rate))

            # =============================================================================
            # Staudenmayer activity levels
            # ============================================================================= 
            if args.staud:
                print_status("Adding Staudenmayer activity levels....",2,logger)
                staud= p.Staudenmayer (df,20)
                staud.set_index('timestamp',inplace=True)
                staud=pd.get_dummies(staud[['Activity','Sedentary','Locomotion']], columns=['Activity','Sedentary','Locomotion'],prefix='remove')
                staud.columns = staud.columns.str.replace('remove_', '', regex=False)
                staud=staud.astype(int)
                staud=staud.resample(f'60s').agg('sum')
                staud=staud*15
                for c in ['light', 'moderate','vigorous', 'nonSedentary','sedentary', 'nonLocomotion','locomotion']:
                    if c not in staud.columns:
                        staud[c]=0
                staud['mvpa']= staud.moderate+staud.vigorous 

                processed_file=os.path.join(target_folder_analytics, "Minute_Summary_" + tracker+"_"+str(DATE).split(" ")[0]+".parquet.gzip")
                staud.to_parquet(processed_file,index=False)

            del df
            gc.collect()
            libc = ctypes.CDLL("libc.so.6") # clearing cache 
            libc.malloc_trim(0)

        if args.plot:
            plots_pdf.close()
            del plots_pdf
            gc.collect()
        Excution_time = (datetime.now()-Start_time_file).total_seconds()
        m="File " + str(i) + " out of " + str(size) + ': Processed subject '+str(PID) + ')' 
        print_status(f"{m} in: {Excution_time:.03f}s",2,logger)

    else:
        print_status(f"No daily data generated for {tracker} because no overlapping GT was found.",2,logger)

    del data
    gc.collect()
    libc = ctypes.CDLL("libc.so.6") # clearing cache 
    libc.malloc_trim(0)

        
    
        
        
# =============================================================================
# Main
# =============================================================================      
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)#'spawn' creates a fresh Python process, avoiding issues with CUDA contexts that happen with 'fork'. This is the recommended approach when using CUDA with multiprocessing.
    if args.clear_tracker and os.path.exists(target_folder_tracking):
        print_status("Trackers deleted. Processing files from scratch...",1,logger)
        shutil.rmtree(target_folder_tracking, ignore_errors=True)
        create_folder([target_folder_tracking])
    with open(os.path.join(args.target_folder, "user_arguments.json"), "w") as json_file:
            json.dump(str(args), json_file, indent=4)

    GT_df=None
    if not args.production:
        GT_df = get_scratch_gt(GT)

    model=device=batch_size=None
    if args.scratch:
        model,device,batch_size=get_scratch_model(args)
        
    main_processor(GT_df,args,model,device,batch_size)     

    # =============================================================================
    # Finished
    # =============================================================================
    End_time=datetime.now()
    Excution_time = (End_time-Start_time).total_seconds()
    print_status(f"Processing finished. Total excution time : {Excution_time:.03f}s",1,logger)

