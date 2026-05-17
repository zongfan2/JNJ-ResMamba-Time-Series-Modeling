import os
import shutil
import csv
import pandas as pd
import numpy as np
import preprocessing as p 
import datetime as dt
from  Helpers.helpers import * 
import argparse


parser = argparse.ArgumentParser(description='Run the Audit of DeepScratch Output. Example : /n python audit.py --input_folder /mnt/data/duplex_ad_dev/Nocturnal Scratch 95475939ADM2001 --result_folder /mnt/data/duplex_ad_dev/nocturnal_scratch_output',formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("--input_folder",type=str, required=False,default='/mnt/data/duplex_ad_dev/Nocturnal Scratch 95475939ADM2001', help="Path to the folder storing the raw data.")
parser.add_argument("--result_folder",type=str, required=False,default='/mnt/data/duplex_ad_dev/nocturnal_scratch_output', help="Path to the folder storing the DeepScratch result data.")
parser.add_argument("--subject",type=str, required=False,default='', help="List of subjects to process seperated by comma. E.g. CV8-AR100030001 or CV8-AR100030001,CV8-AR100030002")

args = parser.parse_args()
input_folder=args.input_folder
result_folder=args.result_folder
create_folder([os.path.join(result_folder,'audit/')])

def DUPLEX_processor_1d(group,site,PID,wrist,TZ):
    acc_data_list=[]
    temperature_data_list=[]
    for f in group.file_path:
        if 'temperature' in f:
            temperature_data_list.append(pd.read_csv(f,usecols=["Timestamp",'TemperatureCelsius',"MonitorSerial"]))
        else:
            acc_data_list.append(pd.read_csv(f,usecols=["Timestamp",'SampleOrder', "X", "Y", "Z","MonitorSerial"]))

    acc_data=pd.concat(acc_data_list,ignore_index=True)
    temperature_data=pd.concat(temperature_data_list,ignore_index=True)

    data=acc_data.merge(temperature_data,how='left')
    data['timestamp']= pd.to_datetime(data['Timestamp'], unit='s').dt.tz_localize('UTC')+ pd.to_timedelta((data.SampleOrder * (1/32)), unit='s')
    #change TZ
    data['timestamp'] = data['timestamp'].dt.tz_convert(TZ)
    #Drop the TZ part of the timestamp, otherwise the time daily grouper would have an expected behavior (grouping using UTC instead of local timezone)
    data['timestamp'] = data['timestamp'].dt.tz_localize(None)
    data.drop(['SampleOrder','Timestamp'], axis=1,inplace=True)
    data.rename(columns={'X':'x','Y':'y','Z':'z','MonitorSerial':'deviceID','TemperatureCelsius':'temperature'},inplace=True)
    data.temperature.bfill(inplace=True)
    data.temperature.ffill(inplace=True)

    return data.timestamp.min(),data.timestamp.max()



#Get result files
filenames =sorted(os.path.join(dp, f) for dp, dn, filenames in os.walk(result_folder) for f in filenames if (f.endswith('.csv')) and (('Scratch_Daily_Summary' in f)) and ('checkpoint' not in f))
pattern = r"Scratch_Daily_Summary_([^_]+)_([^_]+)_([0-9]{4}-[0-9]{2}-[0-9]{2})\.csv$"
file_df=[]
for file_path in filenames:
    match = re.search(pattern, file_path)
    if match:
        PID = match.group(1)   
        wrist = match.group(2) 
        date=match.group(3)
        file_df.append({'wrist':wrist,'PID': PID,'date': date, 'file_path': file_path})
    else:
        print(f"failed {file_path}")
result_file_df=pd.DataFrame(file_df)


#Loop throgh Leap data and check if result found for min max timestamps
filenames = sorted(os.path.join(dp, f) for dp, dn, filenames in os.walk(input_folder) for f in filenames if (f.endswith('.csv')) and (('raw-accel' in f) or ( 'temperature' in f )) and ('checkpoint' not in f))
subjects_file = sorted(os.path.join(dp, f) for dp, dn, filenames in os.walk(input_folder) for f in filenames if (f.endswith('subjects.csv')) and ('checkpoint' not in f))
TZ_df=pd.read_csv(subjects_file[0], dtype=str)
# Regex pattern to extract wrist and ID
pattern = r'DUPLEX - (.+?) Wrist.*?/([^/]+)/([^/]+)/[^/]*?(\d{4}-\d{2}-\d{2})_'
file_df=[]
for file_path in filenames:
    match = re.search(pattern, file_path)
    if match:
        wrist = match.group(1) 
        site = match.group(2).split(' - ')[0] # remove the country part from site ID
        ID = match.group(3)       
        date=match.group(4)
        if (args.subject == ''):
            file_df.append({'wrist':wrist,'site':site,'ID': ID,'date': date, 'file_path': file_path})
        else:
            if site+ID in args.subject.split(','):
                file_df.append({'wrist':wrist,'site':site,'ID': ID,'date': date, 'file_path': file_path})
file_df=pd.DataFrame(file_df)
file_df['date']=pd.to_datetime(file_df['date'], errors='coerce')
file_df = file_df.sort_values(['site','ID','wrist','date'])
file_df.loc[:,'count']=file_df.groupby(['wrist','site','ID','date'])['date'].transform('count')
file_df[file_df['count']==1].to_csv(os.path.join(result_folder,'audit/audit_partial_days.csv'), index=False)
groups=file_df.groupby(['site','ID','wrist'])
i=0
size=len(groups)
audit_df=[]
for idx,group in groups:
    site=idx[0]
    PID=idx[1]
    wrist=idx[2]
    TZ=TZ_df.loc[(TZ_df.siteIdentifier==site)&(TZ_df.subjectIdentifier.astype(str)==PID),'timezone'].values[0]
    tracker=site+"_"+PID+"_"+wrist
    i+=1
    # if (tracker == "CV8-AR10003_0001_Dominant"):
    print(f"{str(pd.Timestamp.now())}  ----> Processing {tracker} (progress: {str(i)}/{str(size)})")
    if len(group[group['count']==1])>0:
        print(f"{str(pd.Timestamp.now())}  ----> The following  days are excluded from processing because they are missing either accelerometer or temperature files {group[group['count']==1].date.dt.date.astype(str).values}")
    
    for idx,d in group[group['count']>1].groupby(['wrist','ID','date']):
        try:
            min_data,max_data=DUPLEX_processor_1d(d,site,PID,wrist,TZ)
            if min_data.time()< dt.time(18, 0, 0):
                day_min=min_data.date()
            else:
                day_min=(min_data+pd.Timedelta(days=1)).date()
    
            if max_data.time()< dt.time(18, 0, 0):
                day_max=max_data.date()
            else:
                day_max=(max_data+pd.Timedelta(days=1)).date()
    
            min_result=result_file_df[(result_file_df.PID==(str(site+PID)))&(result_file_df.wrist==wrist)&(result_file_df.date==str(day_min))]
            max_result=result_file_df[(result_file_df.PID==(str(site+PID)))&(result_file_df.wrist==wrist)&(result_file_df.date==str(day_max))]
            if len(min_result)>0:
                min_path=min_result.iloc[0].file_path
            else:
                min_path=''
            if len(max_result)>0:
                max_path=max_result.iloc[0].file_path
            else:
                max_path=''             
            audit_df.append({'PID': str(site+PID),'wrist':wrist,'date': str(idx[2].date()),'min_found':len(min_result)>0,'min_path':min_path, 'max_found':len(max_result)>0,'max_path':max_path})
        except Exception as e:
            print(f"{str(pd.Timestamp.now())}  ----> An unexpected error occurred: {e}")

audit_df=pd.DataFrame(audit_df)
if args.subject == "":
    audit_df.to_csv(os.path.join(result_folder,'audit/audit.csv'), index=False)
else:
    audit_df.to_csv(os.path.join(result_folder,f"audit/audit_{args.subject.replace(',','_')}.csv"), index=False)
print(f"{str(pd.Timestamp.now())}  ----> Audit finished successfully")