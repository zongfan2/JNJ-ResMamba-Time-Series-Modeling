import os
import tempfile
import numpy as np
import pandas as pd
import scipy.signal as signal
import statsmodels.api as sm
import warnings
import os
import time
import struct
import shutil
import tempfile
import zipfile
import gzip
import pathlib
import numpy as np
import pandas as pd
import subprocess
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
# from tso import *
# from nonwear import *
from  Helpers.helpers import *
import gc



# =============================================================================
# Data preprocessing class
# =============================================================================
class DataPreprocessor:
    def __init__(self, 
                 sample_rate, 
                 filter_cutoff=(0.25, None), 
                 calibrate_gravity=True, 
                 detect_nonwear=True, 
                 resample_hz='uniform', 
                 method='FFT', 
                 detect_motion=True, 
                 detect_TSO=True, 
                 tso_algorithm='van', 
                 nonwear_algorithm='JJ', 
                 day_start=10,
                 verbose=True):
        self.sample_rate = sample_rate
        self.filter_cutoff = filter_cutoff
        self.calibrate_gravity = calibrate_gravity
        self.detect_nonwear = detect_nonwear
        self.resample_hz = resample_hz
        self.method = method
        self.detect_motion = detect_motion
        self.detect_TSO = detect_TSO
        self.tso_algorithm = tso_algorithm
        self.nonwear_algorithm = nonwear_algorithm
        self.day_start=day_start
        self.verbose = verbose

        """
        A class to process a pandas.DataFrame of acceleration time-series. Data is modified directly and no copies are created.

        :param data: A pandas.DataFrame of acceleration time-series. It must contain
            at least columns `x,y,z` and the index must be a DateTimeIndex.
        :type data: pandas.DataFrame.
        :param sample_rate: The data's sample rate (Hz).
        :type sample_rate: int or float
        :param filter_cuttoff: Cutoff (Hz) for the pass filter. The tupple of the format (high pass cutoff, low pass cuttoff), can be (high pass cutoff, None) for highpass filtering, (None, low pass cuttoff) for low pass filtering or (high pass cutoff, low pass cuttoff) for bandwidth filtering. Pass
            None or False to disable.
        :type filter_cuttoff: tuple, optional
        :param calibrate_gravity: Whether to perform gravity calibration. Defaults to True.
        :type calibrate_gravity: bool, optional
        :param detect_nonwear: Whether to perform non-wear detection. Defaults to True.
        :type detect_nonwear: bool, optional
        :param resample_hz: Target frequency (Hz) to resample the signal. If
            "uniform", use the implied frequency (use this option to fix any device
            sampling errors). Pass None to disable. Defaults to "uniform".
        :type resample_hz: str or int, optional
        :param verbose: Verbosity, defaults to True.
        :type verbose: bool, optional
        """
    # =============================================================================
    # Main method for processing the data
    # =============================================================================
    def process(self, data, info, logger):
        """
        Process a pandas.DataFrame of acceleration time-series.
        """
        # Calibrate gravity
        if self.calibrate_gravity:
            print_status("Gravity calibration...", 3, logger)
            self.calibrate(data, info)
        
        # Interpolation will replace x,y,z,, and temperature by 0. This needs to be done after gravity calibration and before resampling.
        data=self.interpolate_data(data.copy())
        
        # Resample the data
        if self.resample_hz.lower() not in ('(none,none)','(false,false)', 'none','false'):
            print_status("Resampling...", 3, logger)
            if self.resample_hz in ('uniform', True):
                data,info=self.resample(data.copy(), info, self.sample_rate, self.sample_rate, self.method)
            else:
                # If downsampling, apply an anti aliasing filter first (low-pass filter) with a cutoff frequency equal to half of the new sampling rate. 
                if int(self.resample_hz)<self.sample_rate:
                    self.passfilter(data, info, self.sample_rate, cutoff_rate=(None,int(int(self.resample_hz)/2)))
                    
                data,info=self.resample(data.copy(), info, self.sample_rate, int(self.resample_hz), self.method)
                self.sample_rate = int(self.resample_hz)
        
        # Detect non-wear
        if self.detect_nonwear:
            print_status("Nonwear detection...", 3, logger)
            if self.nonwear_algorithm == 'JJ':
                self.nonwear_JJ(data, self.sample_rate, info)
            elif self.nonwear_algorithm == 'sleepy':
                nonwear_sleepy(data, self.sample_rate)
            elif self.nonwear_algorithm == 'detach':
                nonwear_detach(data, self.sample_rate)
            elif self.nonwear_algorithm == 'zhou':
                nonwear_zhou(data, self.sample_rate)
            else:
                raise ValueError(f"Unknown nonwear detection algorithm: {self.nonwear_algorithm}")
        
        # Detect sleep periods
        if self.detect_TSO:
            print_status("TSO detection...", 3, logger)
            if self.tso_algorithm in ['van_old','van_new']:
                self.TSO_VanHees(data, info, fs=self.sample_rate, implementation=self.tso_algorithm)
            elif self.tso_algorithm == 'dl':
                # DL TSO model was trained on high-pass filtered data.
                # Run TSO on a filtered copy so the original data is untouched;
                # only the resulting predictTSO column is copied back.
                if not hasattr(self, 'tso_model_path') or self.tso_model_path is None:
                    raise ValueError("tso_model_path must be set when using tso_algorithm='dl'")
                data_cp = data.copy()
                if self.filter_cutoff not in (None, False):
                    self.passfilter(data_cp, {}, self.sample_rate, self.filter_cutoff)
                self.TSO_MBA(
                    data_cp, info, fs=self.sample_rate,
                    model_path=self.tso_model_path,
                    model_name=getattr(self, 'tso_model_name', 'mba4tso_patch'),
                    device=getattr(self, 'tso_device', 'cuda:0'),
                    scaler_path=getattr(self, 'tso_scaler_path', None)
                )
                data['predictTSO'] = data_cp['predictTSO']
                del data_cp
            else:
                raise ValueError(f"Unknown TSO algorithm: {self.tso_algorithm}")

        # Pass filter
        if self.filter_cutoff not in (None, False):
            print_status("Pass filtering...", 3, logger)
            self.passfilter(data, info, self.sample_rate, self.filter_cutoff)
        
        # Detect motion periods
        if self.detect_motion:
            print_status("Motion segments detection...", 3, logger)
            self.motion_segment_detection(data, self.sample_rate)
            
        return data,info
    
    def get_samplerate(self):
        return self.sample_rate
    
    def interpolate_data(self,data):
        t0, tf = data.index[0], data.index[-1]
        nt = int(np.around((tf - t0).total_seconds() * self.sample_rate)) + 1  # integer number of ticks we need

        t = pd.date_range(
            start=t0,
            end=tf,
            periods=nt,
            name=data.index.name,
        )
        data = data.reindex(t, method='nearest', tolerance=pd.Timedelta('1s'))
        # Fill missing data with zeros
        data.loc[:,['x', 'y', 'z', 'temperature']] =data[['x','y','z','temperature']].fillna(0)
        if 'deviceID' in data.columns:
            data.loc[:,'deviceID']=data['deviceID'].bfill()
            data.loc[:,'deviceID']=data['deviceID'].ffill()
            
        return data
            
        
    # @profile
    def resample(self,data,info,original_sample_rate, sample_rate,method):
        """
        FFT resampling for x,y,z, and nearest neighbor resampling. 

        :param data: A pandas.DataFrame of acceleration time-series. The index must be a DateTimeIndex.
        :type data: pandas.DataFrame.
        :param originial_sample_rate: Original sample rate (Hz).
        :type sample_rate: int or float
        :param sample_rate: Target sample rate (Hz) to achieve.
        :type sample_rate: int or float
        :param method: The method used to fill the gaps. Available options: None, 'backfill’/’bfill', 'pad’/’ffill', 'nearest'', 'FFT', 
        :type method: string
        :param dropna: Whether to drop NaN values after resampling. Defaults to False.
        :type dropna: bool, optional
        :param chunksize: Chunk size for chunked processing. Defaults to 1_000_000 rows.
        :type chunksize: int, optional
        :return: Processed data and processing info.
        :rtype: (pandas.DataFrame, dict)
        """

    #     info = {}

        if np.isclose(
            1 / sample_rate,
            pd.Timedelta(pd.infer_freq(data.index)).total_seconds(),
        ):
            print_status(f"Skipping resample: Rate {sample_rate} already achieved",2)
            return data, info

        info['ResampleRate'] = sample_rate

        t0, tf = data.index[0], data.index[-1]
        nt = int(np.around((tf - t0).total_seconds() * sample_rate)) + 1  # integer number of ticks we need

        t = pd.date_range(
            start=t0,
            end=tf,
            periods=nt,
            name=data.index.name,
        )
        if method=="FFT":
            chunk = data.drop(['x', 'y','z'], axis=1).reindex(t, method='nearest', tolerance=pd.Timedelta('1s'))
            x=signal.resample(data.x, len(t))#, window=original_sample_rate*60
            y=signal.resample(data.y, len(t))
            z=signal.resample(data.z, len(t))

            chunk['x']=x
            chunk['y']=y
            chunk['z']=z
        else:
            chunk = data.reindex(t, method=method, tolerance=pd.Timedelta('1s'))
        data=chunk

        del chunk

        info['NumTicksAfterResample'] = len(data)
        
        return data,info

    def passfilter(self,data,info, data_sample_rate,cutoff_rate=(0.25,None), chunksize=1_000_000):
        """
        Apply Butterworth low-pass filter and detrending x,y,z.

        :param data: A pandas.DataFrame of acceleration time-series. The index must be a DateTimeIndex.
        :type data: pandas.DataFrame.
        :param data_sample_rate: The data's original sample rate.
        :type data_sample_rate: int or float
        :param cutoff_rate: Cutoff (Hz) for the pass filter. The tupple of the format (high pass cutoff, low pass cuttoff), can be (high pass cutoff, None) for highpass filtering, (None, low pass cuttoff) for low pass filtering or (high pass cutoff, low pass cuttoff) for bandwidth filtering.
        :type cutoff_rate: tuple, optional
        :param chunksize: Chunk size for chunked processing. Defaults to 1_000_000 rows.
        :type chunksize: int, optional
        :return: Processed data and processing info.
        :rtype: (pandas.DataFrame, dict)
        """

    #     info = {}

        # Skip this if the Nyquist freq is too low
        hicut, lowcut = cutoff_rate
        if (hicut is None) and (lowcut is None):
            print_status(f"Skipping lowpass filter: low and high cutoffs are both None",2)
            return
            
        if lowcut is not None:
            if data_sample_rate / 2 <= lowcut:
                print_status(f"Skipping lowpass filter: data sample rate {data_sample_rate} too low for cutoff rate {cutoff_rate}",2)
                info['LowpassOK'] = 0
                return

        n = len(data)
        leeway = 100  # used to minimize edge effects
        data_list=[]
        for i in range(0, n, chunksize):

            leeway0 = min(i, leeway)
            istart = i - leeway0
            istop = i + chunksize + leeway
            xyz = data.iloc[istart : istop][['x', 'y', 'z']].to_numpy()
            na = np.isnan(xyz).any(1)
            xyz[na] = 0.0  # temporarily replace nans with 0s for butterfilt
            xyz = butterfilt(xyz, cutoff_rate, fs=data_sample_rate, axis=0)
            xyz[na] = np.nan  # restore nans
            xyz = xyz[leeway0 : leeway0 + chunksize]  # trim leeway
            data_list.append(xyz)
        xyz=np.concatenate(data_list, axis=0)
        data.loc[:,['x','y','z']] = xyz
        del data_list,xyz


        data.loc[:,'x']=signal.detrend(data['x'])
        data.loc[:,'y']=signal.detrend(data['y'])
        data.loc[:,'z']=signal.detrend(data['z'])

        info['LowpassOK'] = 1
        info['LowpassCutoff(Hz)'] = str(cutoff_rate)   
        

    def calibrate(self, data, info, calib_cube=0.3, min_stationary_samples=50, window='10s',std_threshold=13/1000,max_iterations=1000,improvement_tolerance=1e-4,error_tolerance=0.01,best_error=1e16):
        """
        Gravity calibration based on van Hees et al. 2014.

        :param data: pandas.DataFrame with columns 'x', 'y', 'z', optionally 'temperature', index as DateTime.
        :param info: dict to store calibration info.
        :param calib_cube: float, calibration cube criteria.
        :param min_stationary_samples: int, minimum stationary samples.
        :param window: str, resampling window.
        :param std_threshold: float, std threshold for stationarity.
        :param return_coeffs: bool, whether to return calibration coefficients.
        :param chunksize: int, chunk size for processing.
        :return: tuple of (calibrated DataFrame, info dict).
        """

        def extract_stationary_xyz(df, stationary_mask):
            """Extract mean stationary xyz vectors based on mask."""
            xyz_mean = df[['x', 'y', 'z']].resample(window, origin='start').mean()
            xyz_stationary = xyz_mean[stationary_mask].dropna().to_numpy()
            non_zero_mask = np.linalg.norm(xyz_stationary, axis=1) > 1e-8
            return xyz_stationary[non_zero_mask]

        def extract_temperature(df, stationary_mask):
            """Extract mean temperature during stationary periods."""
            T_mean = df['temperature'].resample(window, origin='start').mean()
            T_stationary = T_mean[stationary_mask].dropna().to_numpy()
            return T_stationary

        # Step 1: Detect stationary periods
        stationary_mask = (
            data['x'].resample(window, origin='start').std() < std_threshold
        ) & (
            data['y'].resample(window, origin='start').std() < std_threshold
        ) & (
            data['z'].resample(window, origin='start').std() < std_threshold
        )

        # Step 2: Extract stationary xyz vectors
        xyz_vectors = extract_stationary_xyz(data, stationary_mask)
        T_values = extract_temperature(data, stationary_mask) 

        info['num_stationary_samples'] = len(xyz_vectors)

        # Check if enough samples are available
        if len(xyz_vectors) < min_stationary_samples:
            info.update({'calibration_error_before_mg': np.nan,
                         'calibration_error_after_mg': np.nan,
                         'calibration_success': 0})
            warnings.warn(f"Skipping calibration: insufficient stationary samples ({len(xyz_vectors)} < {min_stationary_samples})")
            return data, info

        # Initialize calibration parameters
        intercepts = np.zeros(3)
        slopes = np.ones(3)
        slopes_T = np.zeros(3)

        # Normalize xyz vectors to unit vectors
        current_estimate = xyz_vectors
        target_vectors = current_estimate / np.linalg.norm(current_estimate, axis=1, keepdims=True)

        # Compute initial errors
        residuals = np.linalg.norm(current_estimate - target_vectors, axis=1)
        initial_error = np.mean(residuals)
        info['calibration_error_before_mg'] = initial_error * 1000

        # Check if calibration is necessary
        max_xyz = np.max(xyz_vectors, axis=0)
        min_xyz = np.min(xyz_vectors, axis=0)
        if (max_xyz < calib_cube).any() or (min_xyz > -calib_cube).any():
            info.update({'calibration_error_after_mg': initial_error * 1000,
                         'calibration_num_iterations': 0,
                         'calibration_success': 0})
            return data, info

        # If already within acceptable error
        if initial_error < 0.01:
            info.update({'calibration_error_after_mg': initial_error * 1000,
                         'calibration_num_iterations': 0,
                         'calibration_success': 1})
        else:

            for iteration in range(max_iterations):
                # Outlier weighting
                max_error_quantile = np.quantile(residuals, 0.995)
                weights = np.maximum(1 - residuals / max_error_quantile, 0)

                # Fit for each axis
                for axis_idx in range(3):
                    input_vals = current_estimate[:, axis_idx]
                    target_vals = target_vectors[:, axis_idx]
                    input_vals = np.column_stack((input_vals, T_values))
                    input_with_const = sm.add_constant(input_vals, prepend=True)
                    model = sm.WLS(target_vals, input_with_const, weights=weights).fit()
                    params = model.params

                    # Update intercept and slope
                    intercepts[axis_idx] = params[0] + intercepts[axis_idx] * params[1]
                    slopes[axis_idx] = params[1] * slopes[axis_idx]
                    slopes_T[axis_idx] = params[2] + slopes_T[axis_idx] * params[1]

                # Update current estimate
                current_estimate = intercepts + (xyz_vectors * slopes)
                current_estimate += T_values[:, None] * slopes_T
                target_vectors = current_estimate / np.linalg.norm(current_estimate, axis=1, keepdims=True)

                # Calculate errors
                residuals = np.linalg.norm(current_estimate - target_vectors, axis=1)
                current_error = np.mean(residuals)

                # Track best solution
                if current_error < best_error:
                    best_error = current_error
                    best_intercepts = intercepts.copy()
                    best_slopes = slopes.copy()
                    best_slopes_T = slopes_T.copy()
                # Check convergence
                if (current_error < error_tolerance) or ((best_error - current_error) / best_error < improvement_tolerance):
                    break

            # Save final error
            info['calibration_error_after_mg'] = best_error * 1000
            info['calibration_num_iterations'] = iteration + 1

            # Final check if calibration was successful
            if (best_error >= error_tolerance) or (iteration + 1 >= max_iterations):
                info['calibration_success'] = 0
            else:
                # Apply calibration to data
                data.loc[:, ['x', 'y', 'z']] = (best_intercepts + best_slopes * data[['x', 'y', 'z']].to_numpy())
                data.loc[:, ['x', 'y', 'z']] += data['temperature'].to_numpy()[:, None] * best_slopes_T
                info['calibration_success'] = 1
                # Store calibration coefficients if requested
                info.update({
                    'calib_x_intercept': best_intercepts[0],
                    'calib_y_intercept': best_intercepts[1],
                    'calib_z_intercept': best_intercepts[2],
                    'calib_x_slope': best_slopes[0],
                    'calib_y_slope': best_slopes[1],
                    'calib_z_slope': best_slopes[2],
                    'calib_x_slope_T': best_slopes_T[0],
                    'calib_y_slope_T': best_slopes_T[1],
                    'calib_z_slope_T': best_slopes_T[2],
                })

    # @profile
    def nonwear_JJ(self,data,sf,info, patience='60m', window='60s',window_temperature='60s',motion_buffer='1s', stdtol=15/1000,rangetol=0.15,temperature=25):
        """
        Detect nonwear episodes based on long periods of no movement.

        :param pandas.DataFrame data: A pandas.DataFrame of acceleration time-series. The index must be a DateTimeIndex.
        :type data: pandas.DataFrame.
        :param patience: Minimum length of the stationary period to be flagged as non-wear. Defaults to 90 minutes ("90m").
        :type patience: str, optional
        :param window: Rolling window to use to check for stationary periods. Defaults to 10 seconds ("10s").
        :type window: str, optional
        :param stdtol: Standard deviation under which the window is considered stationary. Defaults to 15 milligravity (0.015).
        :type stdtol: float, optional
        :return: Processed data and processing info.
        :rtype: (pandas.DataFrame, dict)
        """
        info = {}

        def compute_range(series):
            return series.max() - series.min()
        def compute_std_range(group):
            std_indicator = (group.x.std()>=stdtol) & (group.y.std()>=stdtol) & (group.z.std()>=stdtol)
            range_indicator = ((group.x.max()-group.x.min())>=rangetol) & ((group.y.max()-group.y.min())>=rangetol) & ((group.y.max()-group.y.min())>=rangetol)
            return std_indicator | range_indicator


        x_std=data['x'].resample(window, origin='start').std()
        y_std=data['y'].resample(window, origin='start').std()
        z_std=data['z'].resample(window, origin='start').std()

        x_range=data['x'].resample(window, origin='start').max()-data['x'].resample(window, origin='start').min()
        y_range=data['y'].resample(window, origin='start').max()-data['y'].resample(window, origin='start').min()
        z_range=data['z'].resample(window, origin='start').max()-data['z'].resample(window, origin='start').min()

        stationary_indicator_std = (x_std.lt(stdtol) & x_std.lt(stdtol) & x_std.lt(stdtol))
        stationary_indicator_range = x_range.lt(rangetol) & y_range.lt(rangetol) & z_range.lt(rangetol)  
        stationary_indicator = stationary_indicator_std | stationary_indicator_range

        temperature_indicator = (data['temperature'].resample(window_temperature, origin='start').median().lt(temperature)).reindex(stationary_indicator.index, method='ffill')
        nonwear_indicator= stationary_indicator | temperature_indicator 

        nonwear_segment_edges = (nonwear_indicator != nonwear_indicator.shift(1))
        nonwear_segment_edges.iloc[0] = True  # first edge is always True 
        nonwear_segment_ids = nonwear_segment_edges.cumsum()
        nonwear_segment_ids = nonwear_segment_ids[nonwear_indicator]
        nonwear_segment_lengths = (
            nonwear_segment_ids
            .groupby(nonwear_segment_ids)
            .agg(
                start_time=lambda x: x.index[0],
                length=lambda x: x.index[-1] - x.index[0]
            )
            .set_index('start_time')
            .squeeze(axis=1)
            # dtype defaults to int64 when series is empty, so
            # astype('timedelta64[ns]') makes sure it's always a timedelta,
            # otherwise comparison with Timedelta(patience) below will fail
            .astype('timedelta64[ns]')
        ) 


        nonwear_segment_lengths = nonwear_segment_lengths[nonwear_segment_lengths > pd.Timedelta(patience)]

        count_nonwear = len(nonwear_segment_lengths)
        total_nonwear = nonwear_segment_lengths.sum().total_seconds()
        total_wear = (
            data.index.to_series().diff()
            .pipe(lambda x: x[x < pd.Timedelta('1s')].sum())
            .total_seconds()
        ) - total_nonwear

        info['WearTime(days)'] = total_wear / (60 * 60 * 24)
        info['NonwearTime(days)'] = total_nonwear / (60 * 60 * 24)
        info['NumNonwearEpisodes'] = count_nonwear

        # Flag nonwear segments
        #data = data.copy(deep=True)  # copy to avoid modifying original data
        data.loc[:,'non-wear']=False
        last_end=None
        for start_time, length in nonwear_segment_lengths.items():
            #data.loc[start_time:start_time + length] = np.nan
            data.loc[start_time-pd.Timedelta('1s'):start_time +pd.Timedelta('1s')+ length,"non-wear"]=True

    #         #Check if wear time between two non-wear is too small (less than 10 min) and consider it non-wear.
    #         if last_end != None:
    #             delta=(start_time-last_end).total_seconds()/60
    #             if delta<10:
    #                 data.loc[last_end:start_time,"non-wear"]=True

    #         last_end=start_time+length
        info.update(info)
    #     return data, info
    
    
        # @profile
    def TSO_VanHees(self,df,info, fs, spt_block_size=30, spt_max_gap=60, HDCZA_threshold=None, min_sleep_duration=30,nonwear_gap=120,implementation='van_new'):
        """
        Python implementation of the HDCZA sleep period detection algorithm with support for 
        separate consideration of non-contiguous sequences based on 'non-wear' column.

        Parameters:
        :param data: pandas DataFrame containing 'timestamp', 'x', 'y', 'z', and 'non-wear' columns
        :param fs: Sampling frequency (Hz), default is 20Hz
        :param spt_block_size: Minimum duration (in minutes) for a block to be considered as initial sleep period, default is 30 minutes
        :param spt_max_gap: Maximum allowed gap (in minutes) when merging sleep periods, default is 60 minutes
        :param HDCZA_threshold: Threshold parameters list; if None, default values [10, 15] are used
        :param min_sleep_duration: Minimum duration (in minutes) for a sleep period to be returned, default is 30 minutes


        :return A dictionary containing all sleep periods' start and end indices and timestamps, and other information
        """
        # Check for the 'non-wear' column
        if 'non-wear' not in df.columns:
            raise ValueError("Input data must contain a 'non-wear' column to identify wearable/non-wearable segments.")
        df.loc[:,'predictTSO']=False
        df.loc[:,'group']=0
        
        all_sleep=[]
        for d,data in df.groupby(pd.Grouper(freq="d",offset=f"{self.day_start}h00min")): 
            rescore=True
            VanHees_result={}
            while rescore:
                if implementation=='van_new':
                    VanHees_threshold=get_Van_new_Threshold(data[data['non-wear'] == False],fs)
#                     VanHees_threshold=get_Van_new_Threshold(data,fs)
                else:
                    VanHees_threshold=get_Van_old_Threshold(data[data['non-wear'] == False],fs)
#                     VanHees_threshold=get_Van_old_Threshold(data,fs)
                #Regroup data based on continous non-wear
                data.loc[:,'group'] = (data['non-wear'] != data['non-wear'].shift()).cumsum()
                valid_groups = data.loc[data['non-wear'] == False, 'group'].unique()
                data.loc[:,'predictTSO']=False
                day_sleep_periods=[]
                
                for group in valid_groups:
                    # Extract the sub-dataframe for the current group
                    sub_data = data[data['group'] == group]#.reset_index(drop=True)
                    # Apply the HASPT logic to this sub-dataframe
                #         VanHees_result = TSO_VanHees_single(sub_data, fs, spt_block_size, spt_max_gap, HDCZA_threshold, min_sleep_duration)
                    if implementation=='van_new':
                        VanHees_result = self.TSO_Van_new(sub_data, fs,min_rest_block=spt_block_size,allowed_rest_break=spt_max_gap,min_sleep_duration=min_sleep_duration,threshold=VanHees_threshold)
                    else:
                        VanHees_result = self.TSO_Van_old(sub_data, fs, spt_block_size, spt_max_gap, min_sleep_duration,HDCZA_threshold=VanHees_threshold)
#                         VanHees_result = self.TSO_Van_2015(sub_data, fs,min_rest_block=spt_block_size,allowed_rest_break=spt_max_gap,min_sleep_duration=min_sleep_duration)
                    # Append results
                    if len(VanHees_result)>0:
                        threshold=VanHees_result['tib_threshold']
                        for period in VanHees_result['sleep_periods']:
            #                 data.loc[data["timestamp"].between(period['start_time'], period['end_time']),'predictTSO']=True
                            df.loc[(df.index >= period['start_time']) & (df.index <= period['end_time']),'predictTSO'] = True

                            period['start_time'] = str(period['start_time'])
                            period['end_time'] = str(period['end_time'])
                            day_sleep_periods.append(period)

                    del sub_data
                    gc.collect()

                nb_nonwear_gap=0
            #     Loop through the non-wear group, change non-group to sleep for all nonwear periods having a duration < nonwear_gap and which are adjacent to sleep periods
                for idx,group in data[data['non-wear'] == True].groupby('group'):
                # Extract the sub-dataframe for the current group
                    s=group.index.min()
                    e=group.index.max()
                    length=(e-s).total_seconds()/60
                    sleep_is_after=df.loc[(df.index >= e) & (df.index <= (e+pd.Timedelta('10min'))),'predictTSO'].any()
                    if (length<nonwear_gap)&((sleep_is_after)): #|(sleep_is_before)
                        data.loc[(data.index >= s) & (data.index <= e),'non-wear']=False
                        nb_nonwear_gap+=1
                if nb_nonwear_gap==0:
                    rescore=False
            if len(VanHees_result)>0:
                all_sleep.append({'sleep_day':str((d+pd.Timedelta(days=1)).date()),'threshold': str(threshold),'sleep_periods': day_sleep_periods})
        info.update({'all_sleep_periods':all_sleep})

    def TSO_Van_old(self,data, fs, spt_block_size, spt_max_gap, min_sleep_duration,HDCZA_threshold):
        """
        Core logic of the HASPT algorithm, adapted for single contiguous sequences.

        :param data: pandas DataFrame containing 'timestamp', 'x', 'y', 'z' columns
        :param fs: Sampling frequency (Hz)
        :param spt_block_size: Minimum duration (in minutes) for a block to be considered as initial sleep period
        :param spt_max_gap: Maximum allowed gap (in minutes) when merging sleep periods
        :param HDCZA_threshold: Threshold parameters list
        :param min_sleep_duration: Minimum duration (in minutes) for a sleep period to be returned

        :return A dictionary containing sleep periods, thresholds, and in_spt_time array
        """
#         # Extract accelerometer data from the DataFrame
        x = data['x'].values
        y = data['y'].values
        z = data['z'].values
#         # Rolling median over 5 seconds
#         window_size = int(5 * fs)
#         x,y,z= data[['x', 'y', 'z']].rolling(window=window_size,center=True,min_periods=1).median()
#         angle = np.arctan(z / ((x ** 2 + y ** 2) ** 0.5)) * (180.0 / np.pi) 

#         angle = np.arctan(z / ((x ** 2 + y ** 2) ** 0.5)) * (180.0 / np.pi) 

#         # Compute the total acceleration
#         total_acc = np.sqrt(x**2 + y**2 + z**2)
#         # Avoid division by zero
#         total_acc[total_acc == 0] = np.nan

#         # Calculate the device's tilt angle (in degrees)
#         angle = np.arccos(z / total_acc) * (180 / np.pi)
#         # Handle NaN values
#         angle = np.nan_to_num(angle)
        
        angle = np.arctan(z / ((x ** 2 + y ** 2) ** 0.5)) * (180.0 / np.pi) 

        # Calculate the absolute change in angle
        angle_diff = np.abs(np.diff(angle))
        # To keep the same length as the original data, prepend a zero
        angle_diff = np.insert(angle_diff, 0, 0)

        # Calculate the rolling window size (in samples)
        k1 = int(5 * 60 * fs)  # 5-minute window

        # Convert angle differences to a pandas Series for rolling computation
        angle_diff_series = pd.Series(angle_diff)

        # Compute the median absolute deviation (MAD) within the rolling window
        x_mad = angle_diff_series.rolling(window=k1, center=True, min_periods=1).median().values

#         # Determine the threshold
#         if HDCZA_threshold is None:
#             HDCZA_threshold = [10, 15]  # Default values

#         if len(HDCZA_threshold) == 2:
#             percentile = HDCZA_threshold[0]
#             multiplier = HDCZA_threshold[1]
#             threshold = np.nanquantile(x_mad, percentile / 100.0) * multiplier
#             threshold = max(0.13, min(threshold, 0.5))  # Limit threshold between 0.13 and 0.5
#         else:
#             threshold = HDCZA_threshold
        threshold = HDCZA_threshold
        # Identify periods of no movement (angle changes less than threshold)
        nomov = np.zeros(len(x_mad), dtype=int)
        nomov[x_mad < threshold] = 1

        # Add 0 at the beginning and end to facilitate change point detection
        nomov_padded = np.concatenate(([0], nomov, [0]))

        # Identify start and end indices of stationary blocks
        diff_nomov = np.diff(nomov_padded)
        s1 = np.where(diff_nomov == 1)[0]  # Start indices of stationary blocks
        e1 = np.where(diff_nomov == -1)[0]  # End indices of stationary blocks

        # Calculate the duration of each stationary block (in samples)
        block_durations = e1 - s1

        # Convert spt_block_size to samples
        spt_block_size_samples = spt_block_size * 60 * fs

        # Select stationary blocks longer than spt_block_size
        spt_block_indices = np.where(block_durations > spt_block_size_samples)[0]
        s2 = s1[spt_block_indices]
        e2 = e1[spt_block_indices]

        # Initialize the sleep period time indicator
        in_spt_time = np.zeros(len(x_mad), dtype=int)
        for start, end in zip(s2, e2):
            in_spt_time[start:end] = 1

        # Identify gaps between sleep periods
        out_of_spt = (in_spt_time == 0).astype(int)
        out_of_spt_padded = np.concatenate(([0], out_of_spt, [0]))
        diff_out_of_spt = np.diff(out_of_spt_padded)
        s3 = np.where(diff_out_of_spt == 1)[0]  # Start indices of gaps
        e3 = np.where(diff_out_of_spt == -1)[0]  # End indices of gaps

        # Calculate the duration of each gap (in samples)
        gap_durations = e3 - s3
        spt_max_gap_samples = spt_max_gap * 60 * fs

        # Fill gaps shorter than spt_max_gap
        out_of_spt_block_indices = np.where(gap_durations < spt_max_gap_samples)[0]
        s4 = s3[out_of_spt_block_indices]
        e4 = e3[out_of_spt_block_indices]

        for start, end in zip(s4, e4):
            in_spt_time[start:end] = 1

        # Identify continuous sleep periods
        in_spt_time_padded = np.concatenate(([0], in_spt_time, [0]))
        diff_in_spt_time = np.diff(in_spt_time_padded)
        s5 = np.where(diff_in_spt_time == 1)[0]  # Start indices of sleep periods
        e5 = np.where(diff_in_spt_time == -1)[0]  # End indices of sleep periods

        # Calculate the duration of each sleep period (in samples)
        in_spt_durations = e5 - s5

        # Convert min_sleep_duration to samples
        min_sleep_samples = min_sleep_duration * 60 * fs

        # Select sleep periods longer than min_sleep_duration
        valid_sleep_indices = np.where(in_spt_durations >= min_sleep_samples)[0]
        s5 = s5[valid_sleep_indices]
        e5 = e5[valid_sleep_indices]
        in_spt_durations = in_spt_durations[valid_sleep_indices]

        # If timestamps are available, convert indices to times
        sleep_periods = []
    #     if 'timestamp' in data.columns:

        for start_idx, end_idx in zip(s5, e5):
            sleep_period = {
                'start_idx': int(start_idx),
                'end_idx': int(end_idx),
                'start_time': data.index[start_idx],
                'end_time': data.index[end_idx - 1],
                'duration_minutes': (end_idx - start_idx) / (60 * fs)
            }
            sleep_periods.append(sleep_period)
        # Return results
        return {
            'sleep_periods': sleep_periods,
            'tib_threshold': threshold,
            'in_spt_time': in_spt_time
        }
    
    def TSO_Van_new(
        self,
        data,
        fs,
        min_rest_block=30,
        allowed_rest_break=60,
        min_sleep_duration=30,
        minimum_rest_threshold=0.13,
        maximum_rest_threshold=0.5,
        threshold=None
        ):
        """
        TSO algorithm to detect major rest periods based on SleepPy's major_rest_period function.

        :param data: pandas DataFrame containing 'timestamp', 'x', 'y', 'z' columns.
        :param fs: Sampling frequency (Hz).
        :param min_rest_block: Minimum duration (in minutes) for a rest block to be valid.
        :param allowed_rest_break: Maximum allowed duration (in minutes) for interruptions in rest periods.
        :param min_sleep_duration: Minimum duration (in minutes) for a sleep period to be returned.
        :param minimum_rest_threshold: Minimum allowed threshold for determining major rest period.
        :param maximum_rest_threshold: Maximum allowed threshold for determining major rest period.

        :return: A dictionary containing sleep periods, thresholds, and in_spt_time array.
        :rtype: (pandas.DataFrame, dict)
        """


    #     # Ensure 'timestamp' is in datetime format
    #     data['timestamp'] = pd.to_datetime(data['timestamp'])

    #     # Create a copy of data to avoid changing the original data
        df = data.copy()
        # Rolling median over 5 seconds
        window_size = int(5 * fs)
        df[['x_med', 'y_med', 'z_med']] = df[['x', 'y', 'z']].rolling(
            window=window_size,
            center=True,
            min_periods=1
        ).median()

        # Calculate the angle in degrees
        df['angle'] = np.arctan(
            df['z_med'] / np.sqrt(df['x_med'] ** 2 + df['y_med'] ** 2)
        ) * (180.0 / np.pi)

        # Resample to 5-second intervals, taking the mean
        df_resampled = df[['angle']].resample('5s').mean().fillna(0)

        # Calculate the absolute difference in angle
        df_resampled['angle_diff'] = np.abs(
            df_resampled['angle'] - df_resampled['angle'].shift(1)
        ).fillna(0)

        # Rolling median over 5 minutes
        window_size = int((5 * 60) / 5)  # Number of 5-second intervals in 5 minutes
        df_resampled['angle_diff_median'] = df_resampled['angle_diff'].rolling(
            window=window_size,
            center=True,
            min_periods=1
        ).median()

        # Calculate the threshold
        angle_diff_values = df_resampled['angle_diff_median'].dropna().values
#         if threshold is None:
#             if angle_diff_values.size > 0:
#                 threshold = min(
#                     max(
#                         np.percentile(angle_diff_values, 10) * 15.0,
#                         minimum_rest_threshold
#                     ),
#                     maximum_rest_threshold
#                 )
#             else:
#                 threshold = maximum_rest_threshold

        # Classify rest and active periods
        df_resampled['rest'] = 1  # Initialize as active
        df_resampled.loc[df_resampled['angle_diff_median'] < threshold, 'rest'] = 0  # Rest if below threshold

        # Remove rest blocks shorter than min_rest_block minutes
        samples_in_min_rest_block = int((min_rest_block * 60) / 5)
        df_resampled['block'] = (df_resampled['rest'].diff().ne(0)).cumsum()
        groups = df_resampled.groupby('block')

        total_groups = df_resampled['block'].nunique()
        iter_count = 0

        for name, group in groups:
            iter_count += 1
            if iter_count == 1 or iter_count == total_groups:
                continue
            if group['rest'].iloc[0] == 0 and len(group) < samples_in_min_rest_block:
                # Rest block shorter than min_rest_block
                df_resampled.loc[group.index, 'rest'] = 1  # Set to active

        # Remove active blocks shorter than allowed_rest_break minutes
        samples_in_allowed_rest_break = int((allowed_rest_break * 60) / 5)
        df_resampled['block'] = (df_resampled['rest'].diff().ne(0)).cumsum()
        groups = df_resampled.groupby('block')

        iter_count = 0
        for name, group in groups:
            iter_count += 1
            if iter_count == 1 or iter_count == total_groups:
                continue
            if group['rest'].iloc[0] == 1 and len(group) < samples_in_allowed_rest_break:
                # Active block shorter than allowed_rest_break
                df_resampled.loc[group.index, 'rest'] = 0  # Set to rest

        # Identify the longest rest block
        df_resampled['block'] = (df_resampled['rest'].diff().ne(0)).cumsum()
        rest_blocks = df_resampled[df_resampled['rest'] == 0].groupby('block')

        longest_block_length = 0
        longest_block = None

    #     for name, group in rest_blocks:
    #         block_length = len(group)
    #         if block_length > longest_block_length:
    #             longest_block_length = block_length
    #             longest_block = group

        # Build the in_spt_time array
        in_spt_time = np.zeros(len(data), dtype=int)
        sleep_periods = []
#         df.reset_index(inplace=True)
        for name,group in rest_blocks:
            rest_start_time = group.index[0]
            rest_end_time = group.index[-1] + pd.Timedelta(seconds=5)
            rest_duration_minutes = (rest_end_time - rest_start_time).total_seconds() / 60.0

            if rest_duration_minutes >= min_sleep_duration:

                sleep_period = {
                    'start_time': rest_start_time,
                    'end_time': rest_end_time,
                    'duration_minutes': rest_duration_minutes
                }
                sleep_periods.append(sleep_period)
            else:
                # Rest period is shorter than minimum sleep duration
                pass  # in_spt_time remains zeros
        else:
            # No rest period found
            pass  # in_spt_time remains zeros
        del df
        return {
            'sleep_periods': sleep_periods,
            'tib_threshold': threshold
        }
    
    def TSO_Van_new2(
        self,
        data,
        fs,
        min_rest_block=30,
        allowed_rest_break=60,
        min_sleep_duration=30,
        minimum_rest_threshold=0.13,
        maximum_rest_threshold=0.5,
        threshold=None
        ):
        """
        TSO algorithm to detect major rest periods based on SleepPy's major_rest_period function.

        :param data: pandas DataFrame containing 'timestamp', 'x', 'y', 'z' columns.
        :param fs: Sampling frequency (Hz).
        :param min_rest_block: Minimum duration (in minutes) for a rest block to be valid.
        :param allowed_rest_break: Maximum allowed duration (in minutes) for interruptions in rest periods.
        :param min_sleep_duration: Minimum duration (in minutes) for a sleep period to be returned.
        :param minimum_rest_threshold: Minimum allowed threshold for determining major rest period.
        :param maximum_rest_threshold: Maximum allowed threshold for determining major rest period.

        :return: A dictionary containing sleep periods, thresholds, and in_spt_time array.
        :rtype: (pandas.DataFrame, dict)
        """


    #     # Ensure 'timestamp' is in datetime format
    #     data['timestamp'] = pd.to_datetime(data['timestamp'])

    #     # Create a copy of data to avoid changing the original data
        df = data.copy()
        # Rolling median over 5 seconds
        window_size = int(5 * fs)
        df[['x_med', 'y_med', 'z_med']] = df[['x', 'y', 'z']].rolling(
            window=window_size,
            center=True,
            min_periods=1
        ).median()

        # Calculate the angle in degrees
        df['angle'] = np.arctan(
            df['z_med'] / np.sqrt(df['x_med'] ** 2 + df['y_med'] ** 2)
        ) * (180.0 / np.pi)


        # Calculate the absolute difference in angle
        df['angle_diff'] = np.abs(
            df['angle'] - df['angle'].shift(1)
        ).fillna(0)

        # Rolling median over 5 minutes
        df_resampled=df
        window_size = int(5 * 60 * fs)
        df_resampled['angle_diff_median'] = df_resampled['angle_diff'].rolling(
            window=window_size,
            center=True,
            min_periods=1
        ).median()

        # Calculate the threshold
        angle_diff_values = df_resampled['angle_diff_median'].dropna().values
#         if threshold is None:
#             if angle_diff_values.size > 0:
#                 threshold = min(
#                     max(
#                         np.percentile(angle_diff_values, 10) * 15.0,
#                         minimum_rest_threshold
#                     ),
#                     maximum_rest_threshold
#                 )
#             else:
#                 threshold = maximum_rest_threshold

        # Classify rest and active periods
        df_resampled['rest'] = 1  # Initialize as active
        df_resampled.loc[df_resampled['angle_diff_median'] < threshold, 'rest'] = 0  # Rest if below threshold

        # Remove rest blocks shorter than min_rest_block minutes
        samples_in_min_rest_block = int((min_rest_block * 60 *fs) )
        df_resampled['block'] = (df_resampled['rest'].diff().ne(0)).cumsum()
        groups = df_resampled.groupby('block')

        total_groups = df_resampled['block'].nunique()
        iter_count = 0

        for name, group in groups:
            iter_count += 1
            if iter_count == 1 or iter_count == total_groups:
                continue
            if group['rest'].iloc[0] == 0 and len(group) < samples_in_min_rest_block:
                # Rest block shorter than min_rest_block
                df_resampled.loc[group.index, 'rest'] = 1  # Set to active

        # Remove active blocks shorter than allowed_rest_break minutes
        samples_in_allowed_rest_break = int(allowed_rest_break * 60*fs)
        df_resampled['block'] = (df_resampled['rest'].diff().ne(0)).cumsum()
        groups = df_resampled.groupby('block')

        iter_count = 0
        for name, group in groups:
            iter_count += 1
            if iter_count == 1 or iter_count == total_groups:
                continue
            if group['rest'].iloc[0] == 1 and len(group) < samples_in_allowed_rest_break:
                # Active block shorter than allowed_rest_break
                df_resampled.loc[group.index, 'rest'] = 0  # Set to rest

        # Identify the longest rest block
        df_resampled['block'] = (df_resampled['rest'].diff().ne(0)).cumsum()
        rest_blocks = df_resampled[df_resampled['rest'] == 0].groupby('block')

        longest_block_length = 0
        longest_block = None


        sleep_periods = []
#         df.reset_index(inplace=True)
        for name,group in rest_blocks:
            rest_start_time = group.index[0]
            rest_end_time = group.index[-1] + pd.Timedelta(seconds=5)
            rest_duration_minutes = (rest_end_time - rest_start_time).total_seconds() / 60.0

            if rest_duration_minutes >= min_sleep_duration:

                sleep_period = {
                    'start_time': rest_start_time,
                    'end_time': rest_end_time,
                    'duration_minutes': rest_duration_minutes
                }
                sleep_periods.append(sleep_period)
            else:
                # Rest period is shorter than minimum sleep duration
                pass  # in_spt_time remains zeros
        else:
            # No rest period found
            pass  # in_spt_time remains zeros
        del df
        return {
            'sleep_periods': sleep_periods,
            'tib_threshold': threshold
        }

    def TSO_MBA(self, df, info, fs, model_path, model_name='mba4tso_patch',
                        min_tso_duration=30, max_seq_len=1440, patch_size=1200,
                        padding_value=0.0, device='cuda:0', scaler_path=None):
        """
        Deep learning-based TSO detection using trained model from predict_TSO_segment_patch_h5.py

        Parameters:
        -----------
        df : pandas.DataFrame
            Input data with columns: 'x', 'y', 'z', 'non-wear'
            Index must be DatetimeIndex
        info : dict
            Metadata dictionary to update with TSO periods
        fs : int
            Sampling frequency (Hz), typically 20Hz
        model_path : str
            Path to saved model checkpoint (.pt file)
        model_name : str, default='MBA_tsm'
            Model architecture name
        min_tso_duration : int, default=30
            Minimum TSO duration in minutes
        max_seq_len : int, default=1440
            Maximum sequence length in minutes (24 hours)
        patch_size : int, default=1200
            Patch size for model input
        padding_value : float, default=0.0
            Padding value for sequences
        device : str, default='cuda:0'
            Device for inference ('cuda:0', 'cuda:1', or 'cpu')

        Returns:
        --------
        None (modifies df and info in-place)
        """
        import torch
        import torch.nn as nn
        import joblib
        from Helpers.DL_models import setup_model
        from Helpers.DL_helpers import add_padding_tso_patch_h5, smooth_predictions_combined

        # Initialize TSO column
        df.loc[:, 'predictTSO'] = False

        # Set device
        device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Model caching: only load model once per preprocessor instance
        cache_key = f"{model_path}_{model_name}_{device}"
        if not hasattr(self, '_tso_model_cache'):
            self._tso_model_cache = {}

        if cache_key not in self._tso_model_cache:
            print(f"Loading TSO model from: {model_path}")
            print(f"Device: {device}")

            checkpoint = torch.load(model_path, map_location=device)

            # Extract model parameters from checkpoint
            if 'best_params' in checkpoint:
                best_params = checkpoint['best_params']
            else:
                # Default parameters (borrowed from predict_TSO_segment_patch_h5.py)
                best_params = {
                    'batch_size': 24,
                    'num_filters': 128,
                    'dropout': 0.3,
                    'droppath': 0.3,
                    'kernel_f': 3,
                    'kernel_MBA': 7,
                    'num_feature_layers': 6,
                    'blocks_MBA': 5,
                    'featurelayer': 'ResNet',
                    'lr': 0.001,
                    'w_other': 1.0,
                    'w_nonwear': 1.0,
                    'w_tso': 1.0,
                    'padding_value': 0.0,
                    'patch_size': 1200,
                    'patch_channels': 5,
                    'norm1': 'BN',
                    'norm2': 'GN',
                    'output_channels': 1,
                    'skip_connect': True,
                    'skip_cross_attention': True,
                    'use_sincos': True  # Time encoding with sin+cos
                }

            # Setup model
            output_channels = best_params.get('output_channels', 3)
            model = setup_model(model_name, None, max_seq_len, best_params,
                              pretraining=False, num_classes=output_channels)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()

            # Load scaler for x, y, z (must match the scaler used during training)
            scaler_path = "/mnt/data/GENEActive-featurized/results/DL/UKB_v2/mbav1_scaler.joblib"
            scaler = joblib.load(scaler_path)

            # Cache the model and params
            self._tso_model_cache[cache_key] = {
                'model': model,
                'best_params': best_params,
                'scaler': scaler
            }
            print(f"TSO model loaded and cached successfully")
        else:
            print(f"Using cached TSO model")

        # Retrieve cached model
        model = self._tso_model_cache[cache_key]['model']
        best_params = self._tso_model_cache[cache_key]['best_params']
        scaler = self._tso_model_cache[cache_key]['scaler']

        # Process data day by day
        # The model classifies all three states (other/non-wear/TSO) directly,
        # so no need to pre-filter by non-wear — run inference on the full day.
        all_tso_periods = []

        for day_start_time, day_data in df.groupby(pd.Grouper(freq="d", offset=f"{self.day_start}h00min")):
            if len(day_data) < fs * 60:  # Skip if < 1 minute of data
                continue

            day_tso_periods = []

            with torch.no_grad():
                # Extract raw accelerometer data and scale x, y, z
                xyz = day_data[['x', 'y', 'z']].copy()
                print(f"[TSO_MBA DEBUG] x raw mean={xyz['x'].mean():.3f} std={xyz['x'].std():.3f}")
                if scaler is not None:
                    xyz[['x', 'y', 'z']] = scaler.transform(xyz[['x', 'y', 'z']])
                    print(f"[TSO_MBA DEBUG] x after scaling mean={xyz['x'].mean():.3f} std={xyz['x'].std():.3f}")
                else:
                    print(f"[TSO_MBA DEBUG] no scaler applied")
                X_acc = xyz.values  # [seq_len, 3]

                # Extract temperature (required by model)
                if 'temperature' in day_data.columns:
                    X_temp = day_data[['temperature']].values  # [seq_len, 1]
                else:
                    print("Warning: 'temperature' column not found, using zeros")
                    X_temp = np.zeros((len(day_data), 1))

                # Add time-of-day encoding (must match convert_parquet_to_h5.py exactly)
                # time_fraction = (hour + minute/60) / 24  →  absolute position in 24h day
                hour = day_data.index.hour.values
                minute = day_data.index.minute.values
                time_fraction = (hour + minute / 60.0) / 24.0
                time_sin = np.sin(2 * np.pi * time_fraction).reshape(-1, 1)
                time_cos = np.cos(2 * np.pi * time_fraction).reshape(-1, 1)

                # Concatenate all features: [x, y, z, temperature, time_sin, time_cos]
                if best_params.get('use_sincos', True):
                    X_full = np.concatenate([X_acc, X_temp, time_sin, time_cos], axis=1)  # [seq_len, 6]
                else:
                    X_full = np.concatenate([X_acc, X_temp, time_sin], axis=1)  # [seq_len, 5]

                num_channels = X_full.shape[1]

                # Reshape raw signal into minute-level patches
                # Model expects [batch, max_seq_len, patch_size, channels] (same as add_padding_tso_patch_h5)
                num_minutes = min(len(X_full) // patch_size, max_seq_len)
                if num_minutes == 0:
                    continue
                samples_to_use = num_minutes * patch_size
                X_patches = X_full[:samples_to_use].reshape(num_minutes, patch_size, num_channels)

                # Pad to max_seq_len (matching training-time padding)
                pad_X = np.full((1, max_seq_len, patch_size, num_channels), padding_value, dtype=np.float32)
                pad_X[0, :num_minutes] = X_patches

                X_tensor = torch.from_numpy(pad_X).float().to(device)  # [1, max_seq_len, patch_size, channels]
                x_lens = torch.tensor([num_minutes], dtype=torch.int64).to(device)

                # Debug: inspect inputs
                print(f"[TSO_MBA DEBUG] day={day_start_time.date()} num_minutes={num_minutes} "
                      f"X_tensor shape={X_tensor.shape} "
                      f"x[0,0,:3]={X_tensor[0,0,0,:3].tolist()} "
                      f"time_sin range=[{time_sin.min():.3f},{time_sin.max():.3f}] "
                      f"temp range=[{X_temp.min():.2f},{X_temp.max():.2f}]")

                # Run inference — model outputs [1, max_seq_len, C] logits (C=1 or 3)
                outputs = model(X_tensor, x_lens)
                output_channels = outputs.shape[-1]

                # Sigmoid per channel, slice to actual minutes
                preds = torch.sigmoid(outputs).cpu().detach().numpy()[0]  # [max_seq_len, C]
                preds = preds[:num_minutes]  # [num_minutes, C]

                # Debug: inspect outputs
                if output_channels == 1:
                    print(f"[TSO_MBA DEBUG] binary mode preds shape={preds.shape} "
                          f"TSO mean={preds[:,0].mean():.3f} max={preds[:,0].max():.3f}")
                else:
                    print(f"[TSO_MBA DEBUG] 3-class mode preds shape={preds.shape} "
                          f"class0(other) mean={preds[:,0].mean():.3f} "
                          f"class1(nonwear) mean={preds[:,1].mean():.3f} "
                          f"class2(TSO) mean={preds[:,2].mean():.3f} "
                          f"max TSO prob={preds[:,2].max():.3f}")

                # Apply combined smoothing (majority_vote + min_segment)
                # smooth_predictions_combined uses argmax internally, so binary needs 2-channel input
                if output_channels == 1:
                    preds_2ch = np.concatenate([1 - preds, preds], axis=-1)  # [num_minutes, 2]
                    preds_3d = preds_2ch[np.newaxis, :, :]  # [1, num_minutes, 2]
                else:
                    preds_3d = preds[np.newaxis, :, :]  # [1, num_minutes, 3]

                preds_smoothed = smooth_predictions_combined(
                    preds_3d,
                    methods=['majority_vote', 'min_segment'],
                    window_size=15,
                    min_segment_length=5
                )  # [1, num_minutes, C]

                # Extract TSO probabilities: channel 0 in binary, channel 2 in 3-class
                if output_channels == 1:
                    tso_probs = preds_smoothed[0, :, 1]  # channel 1 = TSO in 2-channel representation
                else:
                    tso_probs = preds_smoothed[0, :, 2]  # channel 2 = TSO in 3-class

                tso_pred = (tso_probs > 0.5).astype(int)
                print(f"[TSO_MBA DEBUG] tso_pred sum={tso_pred.sum()} / {len(tso_pred)} minutes")

                # Build a minute-to-sample mapping using actual timestamps
                minute_start_indices = [m * patch_size for m in range(num_minutes)]
                minute_end_indices = [min((m + 1) * patch_size, len(day_data)) - 1 for m in range(num_minutes)]

                # Find continuous TSO periods at minute level
                tso_padded = np.concatenate(([0], tso_pred, [0]))
                diff_tso = np.diff(tso_padded)
                starts = np.where(diff_tso == 1)[0]
                ends = np.where(diff_tso == -1)[0]

                # Upsample: map minute-level boundaries back to exact sample timestamps
                for start_min, end_min in zip(starts, ends):
                    duration_minutes = end_min - start_min

                    if duration_minutes >= min_tso_duration:
                        start_time = day_data.index[minute_start_indices[start_min]]
                        end_time = day_data.index[minute_end_indices[end_min - 1]]

                        df.loc[(df.index >= start_time) & (df.index <= end_time), 'predictTSO'] = True

                        day_tso_periods.append({
                            'start_time': str(start_time),
                            'end_time': str(end_time),
                            'duration_minutes': duration_minutes,
                            'confidence': float(np.mean(tso_probs[start_min:end_min]))
                        })

            # Add day results
            if day_tso_periods:
                all_tso_periods.append({
                    'sleep_day': str((day_start_time + pd.Timedelta(days=1)).date()),
                    'model': model_name,
                    'tso_periods': day_tso_periods
                })

        # Update info
        info.update({'all_tso_periods_dl': all_tso_periods})
        print(f"Deep learning TSO detection complete. Found {len(all_tso_periods)} days with TSO periods")

    # @profile
    def motion_segment_detection(self,data,sf, window='1s',motion_buffer='1s', stdtol=15/1000,rangetol=0.15,max_seq_length=60,segment_type='variable'):
        """
        Detect nonwear episodes based on long periods of no movement.

        :param pandas.DataFrame data: A pandas.DataFrame of acceleration time-series. The index must be a DateTimeIndex.
        :type data: pandas.DataFrame.
        :param patience: Minimum length of the stationary period to be flagged as non-wear. Defaults to 90 minutes ("90m").
        :type patience: str, optional
        :param window: Rolling window to use to check for stationary periods. Defaults to 10 seconds ("10s").
        :type window: str, optional
        :param stdtol: Standard deviation under which the window is considered stationary. Defaults to 15 milligravity (0.015).
        :type stdtol: float, optional
        :return: Processed data and processing info.
        :rtype: (pandas.DataFrame, dict)
        """

        info = {}
        def compute_range(series):
            return series.max() - series.min()
        def compute_std_range(group):
            std_indicator = (group.x.std()>=stdtol) & (group.y.std()>=stdtol) & (group.z.std()>=stdtol)
            range_indicator = ((group.x.max()-group.x.min())>=rangetol) & ((group.y.max()-group.y.min())>=rangetol) & ((group.y.max()-group.y.min())>=rangetol)
            return std_indicator | range_indicator
        def assign_incremental_numbers(x, n):
            return pd.Series([(i // n) + 1 for i in range(len(x))], index=x.index)
        stationary_indicator_std = (  # this is more memory friendly than data[['x', 'y', 'z']].std() 
            data['x'].resample(window, origin='start').std().lt(stdtol)
            & data['y'].resample(window, origin='start').std().lt(stdtol)
            & data['z'].resample(window, origin='start').std().lt(stdtol)
        )

        stationary_indicator_range = ( 
            (data['x'].resample(window, origin='start').max()-data['x'].resample(window, origin='start').min()).lt(rangetol)
            & (data['y'].resample(window, origin='start').max()-data['y'].resample(window, origin='start').min()).lt(rangetol)
            & (data['z'].resample(window, origin='start').max()-data['z'].resample(window, origin='start').min()).lt(rangetol)
        )

        stationary_indicator = stationary_indicator_std | stationary_indicator_range

        segment_edges = (stationary_indicator != stationary_indicator.shift(1))
        segment_ids = segment_edges.cumsum()
        stationary_segment_ids = segment_ids[stationary_indicator]
        stationary_segment_lengths = (
            stationary_segment_ids
            .groupby(stationary_segment_ids)
            .agg(
                start_time=lambda x: x.index[0],
                length=lambda x: x.index[-1] - x.index[0]
            )
            .set_index('start_time')
            .squeeze(axis=1)
            # dtype defaults to int64 when series is empty, 
            # astype('timedelta64[ns]') makes sure it's always a timedelta,
            # otherwise comparison with Timedelta(patience) below will fail
            .astype('timedelta64[ns]')
        )
        

        data.loc[:,"stationary"]=False
        if segment_type=='variable':
            buffer=pd.Timedelta(motion_buffer)
            for start_time, length in stationary_segment_lengths.items():
                #data.loc[start_time:start_time + length] = np.nan
                if length > pd.Timedelta('3s'): #Only focus on stationary periods of more than this threshold. Meaning a motion period can inlcude up to few seconds of non-motion at a time
                    data.loc[start_time+buffer:start_time + length-buffer,"stationary"]=True
            
            s_edges = (data.stationary != data.stationary.shift(1))
            #     s_edges.iloc[0] = True  # first edge is always True 
            data.loc[:,'segment']=s_edges.cumsum()

            #Split th emotion segments
            if max_seq_length is not None:
                length_min_segments=max_seq_length*sf
                data.loc[:,'mini_segment'] = data.groupby('segment').cumcount() // length_min_segments + 1
                data.loc[data['mini_segment']==0,'mini_segment'] = 1
                data.loc[:,'segment_duration']=data.groupby(['segment','mini_segment'])['x'].transform('count')
                data.loc[(data.segment_duration<sf)&(data.mini_segment>1),'mini_segment']=data.loc[(data.segment_duration<sf)&(data.mini_segment>1),'mini_segment']-1 #If a motion segment's duration is less then 1s, and it's a split motion, asign it to the previous segment.
                data.loc[:,'segment'] = data['segment'].astype(str,copy=False) +"_"+data['mini_segment'].astype(str,copy=False)
            data.loc[:,'segment_duration']=data.groupby(['segment'])['x'].transform('count')
            del s_edges
        else:
            #TODO: double check 
            for start_time, length in stationary_segment_lengths.items():
                #data.loc[start_time:start_time + length] = np.nan
                data.loc[start_time:start_time ,"stationary"]=True
            
            # Generate incremental numbers for each unique minute
            data['segment'] =  pd.Categorical(data.index.floor('T')).codes+1
            data.loc[:,'segment_duration']=data.groupby(['segment'])['x'].transform('count')
            data.loc[:,'stationary']=data.groupby('segment')['stationary'].transform('all')

        del stationary_indicator,segment_ids,stationary_segment_lengths,stationary_segment_ids,stationary_indicator_range,stationary_indicator_std



def butterfilt(x, cutoffs, fs, order=5, axis=0):
    """ Butterworth filter. """
    nyq = 0.5 * fs

    hicut, lowcut = cutoffs
    if hicut is not None:
        if lowcut is not None:
            btype = 'bandpass'
            Wn = (hicut / nyq, lowcut / nyq)
        else:
            btype = 'highpass'
            Wn = hicut / nyq
    else:

        if lowcut is not None:
            btype = 'lowpass'
            Wn = lowcut / nyq
        else:
            print("(None,None) not supported in passfilter")


    sos = signal.butter(order, Wn, btype=btype, analog=False, output='sos')
    y = signal.sosfiltfilt(sos, x, axis=axis)
    y = y.astype(x.dtype, copy=False)

    return y

def Staudenmayer (self,data,sf):

    def compute_features(df, interval_sec=15, sampling_rate=100):
        features = []
        # Resample data into 15-second intervals
        resampled_df = df.resample(f'{interval_sec}s',origin='start').agg(list)

        for index, row in resampled_df.iterrows():
            # Flatten the lists for x, y, z
            if len(row['x']) > sampling_rate*2:
                try:
                    x = np.array(row['x'])
                    y = np.array(row['y'])
                    z = np.array(row['z'])

                    # Vector Magnitude
                    vm = np.sqrt(x**2 + y**2 + z**2)

                    # 1. Mean of vector magnitude
                    mvm = np.mean(vm)

                    # 2. SD of vector magnitude
                    sdvm = np.std(vm)

                    # 3. Power percentage in 0.6–2.5 Hz
                    freqs = fftfreq(len(vm), 1/sampling_rate)
                    vm_fft = np.abs(fft(signal.detrend(vm)))

                    # Power in the specified frequency range
                    power_06_to_25 = np.sum(vm_fft[(freqs >= 0.6) & (freqs <= 2.5)])
                    total_power = np.sum(vm_fft)
                    p625 = power_06_to_25 / total_power if total_power > 0 else 0

                    # 4. Dominant frequency
                    dominant_freq_idx = np.argmax(vm_fft)
                    df = freqs[dominant_freq_idx]

                    # 5. Fraction of power at dominant frequency
                    fpdf = vm_fft[dominant_freq_idx] / total_power if total_power > 0 else 0

                    # 6. Mean angle of acceleration relative to vertical
                    angles = 90 * np.arcsin(y / vm) * (1 / (np.pi / 2))
                    mangle = np.mean(angles)

                    # 7. SD of the angle of acceleration relative to vertical
                    sdangle = np.std(angles)

                    # Store the computed features
                    features.append({
                        'timestamp': index,
                        'vm': mvm,
                        'sdvm': sdvm,
                        'p625': p625,
                        'df': df,
                        'fpdf': fpdf,
                        'mangle': mangle,
                        'sdangle': sdangle,
                    })
                except Exception as e:
                    print_status(f"An unexpected error occurred. Details: {e}",2)

        return pd.DataFrame(features)

    data.set_index('timestamp',inplace=True)
    features_df = compute_features(data,sampling_rate=sf)
    #Activity
    features_df.loc[(features_df.sdvm <=0.26) & (features_df.mangle>-52),'Activity']='Light'
    features_df.loc[(features_df.sdvm <=0.26) & (features_df.mangle<=-52),'Activity']='Moderate'
    features_df.loc[(0.26 < features_df.sdvm )&(features_df.sdvm <=0.79)  & (features_df.mangle>-53),'Activity']='Moderate'
    features_df.loc[(0.26 < features_df.sdvm )&(features_df.sdvm <=0.79)  & (features_df.mangle<=-53),'Activity']='Vigorous'
    features_df.loc[(features_df.sdvm >0.79),'Activity']='Vigorous'
    #Sedentary
    features_df.loc[(features_df.sdvm <=0.098) & (features_df.p625<=0.138),'Sedentary']='Sedentary'
    features_df.loc[(features_df.sdvm <=0.062) & (features_df.p625>0.138),'Sedentary']='Sedentary'
    features_df.loc[(features_df.sdvm >0.062) & (features_df.sdvm <=0.098) & (features_df.p625>0.138),'Sedentary']='Nonsedentary'
    features_df.loc[(features_df.sdvm >0.098) & (features_df.sdvm <=0.148) & (features_df.p625<=0.118),'Sedentary']='Sedentary'
    features_df.loc[(features_df.sdvm >0.098) & (features_df.sdvm <=0.148) & (features_df.p625>0.118),'Sedentary']='Nonsedentary'
    features_df.loc[(features_df.sdvm >0.148),'Sedentary']='Nonsedentary'
    #Locomotion
    features_df.loc[(features_df.fpdf <=0.039) & (features_df.mangle>-53),'Locomotion']='Nonlocomotion'
    features_df.loc[(features_df.fpdf <=0.020) & (features_df.mangle<=-53),'Locomotion']='Nonlocomotion'
    features_df.loc[(features_df.fpdf >0.020) & (features_df.fpdf <=0.039) & (features_df.mangle<=-53),'Locomotion']='Locomotion'
    features_df.loc[(features_df.fpdf >0.039) & (features_df.mangle<=-62),'Locomotion']='Locomotion'
    features_df.loc[(features_df.fpdf >0.039) & (features_df.fpdf <=0.060) & (features_df.mangle>-62),'Locomotion']='Nonlocomotion'
    features_df.loc[(features_df.fpdf >0.060) & (features_df.mangle>-62),'Locomotion']='Locomotion'
    return features_df

def get_Van_new_Threshold(data,fs=20,minimum_rest_threshold=0.13,maximum_rest_threshold=0.5):
        if len(data)>0:
            df = data.copy()
            # Rolling median over 5 seconds
            window_size = int(5 * fs)
            df[['x_med', 'y_med', 'z_med']] = df[['x', 'y', 'z']].rolling(
                window=window_size,
                center=True,
                min_periods=1
            ).median()

            # Calculate the angle in degrees
            df['angle'] = np.arctan(
                df['z_med'] / np.sqrt(df['x_med'] ** 2 + df['y_med'] ** 2)
            ) * (180.0 / np.pi)

            # Resample to 5-second intervals, taking the mean
            df_resampled = df[['angle']].resample('5s').mean().fillna(0)

            # Calculate the absolute difference in angle
            df_resampled['angle_diff'] = np.abs(
                df_resampled['angle'] - df_resampled['angle'].shift(1)
            ).fillna(0)

            # Rolling median over 5 minutes
            window_size = int((5 * 60) / 5)  # Number of 5-second intervals in 5 minutes
            df_resampled['angle_diff_median'] = df_resampled['angle_diff'].rolling(
                window=window_size,
                center=True,
                min_periods=1
            ).median()

            # Calculate the threshold
            angle_diff_values = df_resampled['angle_diff_median'].dropna().values
            if angle_diff_values.size > 0:
                threshold = min(
                    max(
                        np.percentile(angle_diff_values, 10) * 15.0,
                        minimum_rest_threshold
                    ),
                    maximum_rest_threshold
                )
#                 threshold=max(
#                         np.percentile(angle_diff_values, 10) * 15.0,
#                         minimum_rest_threshold
#                     )
            else:
                threshold = maximum_rest_threshold
            del df
        else:
            threshold=0
        return threshold
    
def get_Van_new2_Threshold(data,fs=20,minimum_rest_threshold=0.13,maximum_rest_threshold=0.5):
        if len(data)>0:
            df = data.copy()
            # Rolling median over 5 seconds
            window_size = int(5 * fs)
            df[['x_med', 'y_med', 'z_med']] = df[['x', 'y', 'z']].rolling(
                window=window_size,
                center=True,
                min_periods=1
            ).median()

            # Calculate the angle in degrees
            df['angle'] = np.arctan(
                df['z_med'] / np.sqrt(df['x_med'] ** 2 + df['y_med'] ** 2)
            ) * (180.0 / np.pi)


            # Calculate the absolute difference in angle
            df['angle_diff'] = np.abs(
                df['angle'] - df['angle'].shift(1)
            ).fillna(0)

            # Rolling median over 5 minutes
            window_size = int(5 * 60 * fs)
            df['angle_diff_median'] = df['angle_diff'].rolling(
                window=window_size,
                center=True,
                min_periods=1
            ).median()

            # Calculate the threshold
            angle_diff_values = df['angle_diff_median'].dropna().values
            if angle_diff_values.size > 0:
                threshold = min(
                    max(
                        np.percentile(angle_diff_values, 10) * 15.0,
                        minimum_rest_threshold
                    ),
                    maximum_rest_threshold
                )
            else:
                threshold = maximum_rest_threshold
            del df
        else:
            threshold=0
        return threshold
def get_Van_old_Threshold(data,fs,minimum_rest_threshold=0.13,maximum_rest_threshold=0.5):
    if len(data)>0:
        x = data['x'].values
        y = data['y'].values
        z = data['z'].values
        # Compute the total acceleration
        total_acc = np.sqrt(x**2 + y**2 + z**2)
        # Avoid division by zero
        total_acc[total_acc == 0] = np.nan

        # Calculate the device's tilt angle (in degrees)
        angle = np.arctan(z / ((x ** 2 + y ** 2) ** 0.5)) * (180.0 / np.pi) 
        # Handle NaN values
        angle = np.nan_to_num(angle)

        # Calculate the absolute change in angle
        angle_diff = np.abs(np.diff(angle))
        # To keep the same length as the original data, prepend a zero
        angle_diff = np.insert(angle_diff, 0, 0)

        # Calculate the rolling window size (in samples)
        k1 = int(5 * 60 * fs)  # 5-minute window

        # Convert angle differences to a pandas Series for rolling computation
        angle_diff_series = pd.Series(angle_diff)

        # Compute the median absolute deviation (MAD) within the rolling window
        x_mad = angle_diff_series.rolling(window=k1, center=True, min_periods=1).median().values

        # Determine the threshold
        HDCZA_threshold = [10, 15]  # Default values
        percentile = HDCZA_threshold[0]
        multiplier = HDCZA_threshold[1]
        threshold = np.nanquantile(x_mad, percentile / 100.0) * multiplier
        threshold = max(minimum_rest_threshold, min(threshold,maximum_rest_threshold))  # Limit threshold between 0.13 and 0.5
        return threshold
    else:
        return 0
