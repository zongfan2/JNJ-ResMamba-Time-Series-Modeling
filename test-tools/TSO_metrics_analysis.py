import subprocess

shell_script = '''
sudo python3.11 -m pip install openpyxl seaborn matplotlib
'''
result = subprocess.run(shell_script, shell=True, capture_output=True, text=True)
import pandas as pd


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta


def compute_tst(df, column_name):
    """Compute Total Sleep Time (TST) in hours."""
    tso_active = df[df[column_name]]
    if tso_active.empty:
        return 0
    return tso_active['timestamp'].diff().dt.total_seconds().sum() / 3600

def get_formatted_algorithm_name(data_folder):
    """Get formatted algorithm name from data folder path"""
    # Extract the suffix after 'predictTSO_'
    suffix = data_folder.split('predictTSO_')[-1]
    
    # Define mapping for special cases
    name_mapping = {
        'or': 'Original J&J +van Hees',
        'sleepy': 'Sleepy (ZW Redeploy)+van Hees',
        'sleepy_original_TSO': 'Sleepy+Sleepy',
        'sleepy_original': 'Sleepy Original Code+van Hees',
        'detach': 'Detach (ZW Redeploy)+van Hees',
        'detach_original': 'Detach Original Code+van Hees'
    }
    
    # Return mapped name if it exists, otherwise use default formatting
    return name_mapping.get(suffix, suffix)

def compute_tso_start_end(df, column):
    """Compute TSO start and end times"""
    if not df[column].any():
        return None, None
    
    # Find first and last True values
    true_indices = df.index[df[column]].tolist()
    if not true_indices:
        return None, None
    
    start_idx = true_indices[0]
    end_idx = true_indices[-1]
    
    return df.loc[start_idx, 'timestamp'], df.loc[end_idx, 'timestamp']

def convert_to_hours_from_midnight(dt, adjust_negative=True):
    """Convert datetime to hours from midnight"""
    hours = dt.hour + dt.minute/60 + dt.second/3600
    if adjust_negative and hours >= 18:  # After 19:00
        hours = hours - 24  # Convert to negative hours
    return hours

def average_time_across_dates(time1, time2):
    """Calculate average time accounting for day boundaries"""
    # Convert both times to the same date to calculate difference
    time1_hour = convert_to_hours_from_midnight(time1, adjust_negative=True)
    time2_hour = convert_to_hours_from_midnight(time2, adjust_negative=True)
    avg_hour = (time1_hour + time2_hour) / 2
    
    # Create a reference datetime at midnight
    ref_date = datetime.combine(time1.date(), datetime.min.time())
    
    # Add the average hours
    if avg_hour < 0:
        avg_hour += 24
    return ref_date + timedelta(hours=avg_hour)

def calculate_stats(diff_array):
    bias = np.mean(diff_array)
    std_diff = np.std(diff_array)
    lloa = bias - 1.96 * std_diff
    uloa = bias + 1.96 * std_diff
    return bias, lloa, uloa

# Set seaborn theme
sns.set_theme(style="whitegrid")

def create_bland_altman_plots(mean_TST_array, diff_TST_array, 
                            start_time_mean_array, start_time_diff_array,
                            end_time_mean_array, end_time_diff_array,
                            n_participants, algo_info):
    """Create three Bland-Altman plots"""
    colors = sns.color_palette("tab10",1)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    
    # Plot 1: TSO Duration
    bias1, lloa1, uloa1 = calculate_stats(diff_TST_array)
    ax1.scatter(mean_TST_array, diff_TST_array, alpha=0.7, c=colors)
    ax1.axhline(y=bias1, color='black', linestyle='-')
    ax1.axhline(y=lloa1, color='red', linestyle='--')
    ax1.axhline(y=uloa1, color='red', linestyle='--')
    ax1.axhline(y=0, color='gray', linestyle=':')
    
    # Plot 2: TSO Start
    bias2, lloa2, uloa2 = calculate_stats(start_time_diff_array)
    ax2.scatter(start_time_mean_array, start_time_diff_array, alpha=0.7, c=colors)
    ax2.axhline(y=bias2, color='black', linestyle='-')
    ax2.axhline(y=lloa2, color='red', linestyle='--')
    ax2.axhline(y=uloa2, color='red', linestyle='--')
    ax2.axhline(y=0, color='gray', linestyle=':')
    
    # Plot 3: TSO End
    bias3, lloa3, uloa3 = calculate_stats(end_time_diff_array)
    ax3.scatter(end_time_mean_array, end_time_diff_array, alpha=0.7, c=colors)
    ax3.axhline(y=bias3, color='black', linestyle='-')
    ax3.axhline(y=lloa3, color='red', linestyle='--')
    ax3.axhline(y=uloa3, color='red', linestyle='--')
    ax3.axhline(y=0, color='gray', linestyle=':')
    
    # Customize plots
    plots = [(ax1, 'Duration', bias1, lloa1, uloa1),
             (ax2, 'Start', bias2, lloa2, uloa2),
             (ax3, 'End', bias3, lloa3, uloa3)]
    
    for ax, title, bias, lloa, uloa in plots:
        ax.set_title(f'Bland-Altman Plot: {algo_info}\nTSO {title}')
        
        stats_text = f'Bias = {bias:.2f}\nLLOA = {lloa:.2f}\nULOA = {uloa:.2f}'
        stats_text += f'\nNumber of Participants = {n_participants}'
        stats_text += f'\nNumber of Participant-Nights = {len(mean_TST_array)}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set specific labels and limits
    ax1.set_xlabel('Mean of Predicted and Reference\nTSO Duration [hours]')
    ax1.set_ylabel('Difference of Predicted and Reference\nTSO Duration [hours]')
    
    ax2.set_xlabel('Mean of Predicted and Reference\nTSO Start [hours from midnight]')
    ax2.set_ylabel('Difference of Predicted and Reference\nTSO Start [hours]')
    # ax2.set_xlim(-5, 19)
    
    ax3.set_xlabel('Mean of Predicted and Reference\nTSO End [hours from midnight]')
    ax3.set_ylabel('Difference of Predicted and Reference\nTSO End [hours]')
    # ax3.set_xlim(-5, 19)
    
    plt.tight_layout()
    return fig


ns_dir = '/mnt/data/Nocturnal-scratch/geneactive_20hz_3s_b1s_production_test_scratch'
folders = ['DeepTSO-JNJ', 'DeepTSO-JNJ-ch=5', 'DeepTSO-UKB-ch=6-dlrtc']


hands = pd.read_excel('/mnt/data/Ground-truth/SDI.xlsx')
hands = hands[1:]  # Skip header row if necessary
hands = hands[['Patient_ID', 'SDI_SDI_QS2.1']]
hands.columns = ['subject', 'ND-hand']
# Invert the dominant hand to get the non-dominant hand
hands['ND-hand'] = np.where(hands['ND-hand'] == 'Right', 'left', 'right')
hands['subject'] = hands['subject'].str.split('-').str[1]



res_stat = []
for folder_name in folders:
    data_folder = os.path.join(ns_dir, folder_name)
    if not os.path.exists(data_folder):
        print(f"Skipping {folder_name} - folder not found")
        continue
        
    algo_info = folder_name
    day_start = folder_name
    
    TST_predicted_list = []
    TST_ground_truth_list = []
    mean_TST_list = []
    diff_TST_list = []
    start_time_diff_list = []
    end_time_diff_list = []
    start_time_mean_list = []
    end_time_mean_list = []
    participant_ids = set()

    tso_folder = f'{data_folder}/analytics/sleep'
    filenames = [f for f in os.listdir(tso_folder) if f.endswith('.csv')]
    filenames.sort()
    
    
    for file in filenames:
        participant_id = file.split('_')[2]
        wrist = file.split('_')[3]

        if participant_id not in hands['subject'].values:
            print(f"Participant {participant_id} not in hands data")
            continue

        nd_hand = hands[hands['subject'] == participant_id]['ND-hand'].values[0]

#         if wrist== nd_hand:
#             # Skip this file as it's not from ND-hand
#             continue

        participant_ids.add(participant_id)
        
        file_path = os.path.join(tso_folder, file)
        df = pd.read_csv(file_path)
        if df[['ALG_TSO_START','ALG_TSO_END']].isna().any().any():
            print(df)
            continue
        df.fillna(0, inplace=True)
        # Compute start and end times
        predicted_start, predicted_end = pd.to_datetime(df['ALG_TSO_START'].values[0]), pd.to_datetime(df['ALG_TSO_END'].values[0])
        ground_truth_start, ground_truth_end = pd.to_datetime(df['TSOSTART'].values[0]), pd.to_datetime(df['TSOEND'].values[0])
        
        # Compute TST
        TST_predicted = (predicted_end - predicted_start).total_seconds() / 3600
        TST_ground_truth = (ground_truth_end - ground_truth_start).total_seconds() / 3600
        
        if predicted_start is None or ground_truth_start is None or predicted_end is None or ground_truth_end is None:
            continue
        
        start_time_diff = (predicted_start - ground_truth_start).total_seconds() / 3600
        end_time_diff = (predicted_end - ground_truth_end).total_seconds() / 3600
        
        # Compute average times
        mean_start_time = average_time_across_dates(predicted_start, ground_truth_start)
        mean_end_time = average_time_across_dates(predicted_end, ground_truth_end)
        
        # Store results
        TST_predicted_list.append(TST_predicted)
        TST_ground_truth_list.append(TST_ground_truth)
        
        mean_TST_list.append(np.mean([TST_predicted, TST_ground_truth]))
        diff_TST_list.append(TST_predicted - TST_ground_truth)
        
        start_time_diff_list.append(start_time_diff)
        end_time_diff_list.append(end_time_diff)
        
        # Convert to hours from midnight
        start_time_mean_list.append(convert_to_hours_from_midnight(mean_start_time, adjust_negative=True))
        end_time_mean_list.append(convert_to_hours_from_midnight(mean_end_time, adjust_negative=True))
        
    
    # Convert lists to numpy arrays
    mean_TST_array = np.array(mean_TST_list)
    diff_TST_array = np.array(diff_TST_list)
    start_time_diff_array = np.array(start_time_diff_list)
    end_time_diff_array = np.array(end_time_diff_list)
    start_time_mean_array = np.array(start_time_mean_list)
    end_time_mean_array = np.array(end_time_mean_list)
    
    # Number of unique participants
    n_participants = len(participant_ids)
    
    # Create Bland-Altman plots
    fig = create_bland_altman_plots(
        mean_TST_array, diff_TST_array,
        start_time_mean_array, start_time_diff_array,
        end_time_mean_array, end_time_diff_array,
        n_participants, algo_info
    )

    bias_tso, lloa_tso, uloa_tso = calculate_stats(diff_TST_array)
    bias_start, lloa_start, uloa_start = calculate_stats(start_time_diff_array)
    bias_end, lloa_end, uloa_end = calculate_stats(end_time_diff_array)
    
    res_stat.append(['TSO Duration', algo_info, day_start, bias_tso, lloa_tso, uloa_tso])
    res_stat.append(['TSO Start', algo_info, day_start, bias_start, lloa_start, uloa_start])
    res_stat.append(['TSO End', algo_info, day_start, bias_end, lloa_end, uloa_end])
    
    
    plt.savefig(f'bland_altman_{folder_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved bland_altman_{folder_name}.png")

res_stat_df = pd.DataFrame(res_stat, columns = ['Metric', 'Folder','Day_start','bias','lloa','uloa'])
print(res_stat_df)
