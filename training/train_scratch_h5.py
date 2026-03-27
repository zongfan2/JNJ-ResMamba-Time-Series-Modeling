# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:19:50 2024

@author: MBoukhec
"""
import subprocess

import joblib

shell_script = '''
sudo python3.11 -m pip install -r munge/predictive_modeling/requirements-ml.txt
sudo python3.11 -m pip install -e .
sudo python3.11 -m pip install optuna seaborn ray TensorboardX torcheval mamba-ssm[causal-conv1d]==2.2.2 mlflow pyts
'''
result = subprocess.run(shell_script, shell=True, capture_output=True, text=True, check=True)

import os
import joblib
import argparse
import mlflow

starting_domino_user = os.environ["DOMINO_STARTING_USERNAME"]
experiment_name = f"NS_Detect{starting_domino_user}"
# mlflow.set_experiment(experiment_name=experiment_name)
# ---------------
# Parse Arguments
# ---------------
parser = argparse.ArgumentParser(description='Run Predictive Modeling Pipeline.')
parser.add_argument('--input_data_folder', type=str, required=True, help='Path to the input data file')
parser.add_argument('--clear_tracker', type=bool, default=False, required=False, help='Clear Job Tracking Folder')
parser.add_argument('--model', type=str, default="MLP", required=True, help='Define the model to be used')
parser.add_argument('--output', type=str, required=True, help='Define the name of the output folder')
parser.add_argument('--execution_mode', type=str, default="train", required=False, help='Define if the script should train or tune the model')
parser.add_argument('--features_type', type=str, default="raw", required=False, help='Select (raw/hand_crafted) features.')
parser.add_argument('--motion_filter', type=bool, default=True, required=False, help='Motion Filter')
parser.add_argument("--motion_threshold",type=float,default=0.05,  required=False, help="Threshold of Max(x_std,y_std,z_std) to be applied. Default is 0.013")
parser.add_argument("--data_augmentation",type=int,default=0,  required=False, help="Number of iterations used to augment the the data. Default is 0 for no augmentation.")
parser.add_argument("--epochs",type=int,default=200,  required=False, help="Number of epochs to be used in training")
parser.add_argument('--tune_path', type=str, required=False,default="", help='Path to the input data file')
parser.add_argument("--balanced",type=str,default="False", required=False, help="If to balance data prior to training and testing")
parser.add_argument("--num_gpu",type=str,default="NA", required=False, help="Which gpu to use")
parser.add_argument("--remove_hnv",type=bool,default=False, required=False, help="If remove HNV from training. Default to True")
parser.add_argument("--testing",type=str,default="LOFO", required=False, help="Testing mechanism. Either LOSO (Leave One Subject Out) or LOFO (Leave One Fold Out). Default is LOFO and folds are hard coded to ensure consistency accross experiments.")
parser.add_argument("--pretraining",type=str,default="False", required=False, help="Wether to run the model in a pretraining mode using self-supervised learning.")
parser.add_argument("--training_iterations",type=int,default=1, required=False, help="The number of training iterations. Default is 1.")
parser.add_argument("--pretrained_feature_extractor_path", type=str, default="", required=False, help="Path to load pretrained feature extractor for mamba models. Default is False.")



# parser.add_argument('--list_features', type=int, default=5, required=False, help='Selected list features.')
# parser.add_argument('--multi_channel', type=bool, default=False, required=False, help='1D / multi channel data.')


args = parser.parse_args()


# scaler_path = "/mnt/data/GENEActive-featurized/results/DL/time_windows_results_3s_1s/std_scaler_3s.bin"
scaler_path = "/domino/datasets/GENEActive-featurized/results/DL/time_windows_results_3s_1s/std_scaler_3s.bin"
pre_saved_scaler = joblib.load(scaler_path)

# os.environ["CUDA_VISIBLE_DEVICES"]=num_gpu


import ast

import random
import shutil
from pprint import pprint
from datetime import datetime

import pandas as pd
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import tempfile
from torch import nn
from ray.train import Checkpoint, CheckpointConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
#from tsai.models.TCN import TCN

import ray
from ray import tune
from ray import train
from ray.tune.search.optuna import OptunaSearch
import torch
from ray.train.context import TrainContext
from torch.nn.parallel import DistributedDataParallel
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import glob

import re
import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from functools import partial
import os,gc
import tempfile
import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import ray
from ray.train import Checkpoint, CheckpointConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from models import *
from data import *
from losses import *
from evaluation import *
from utils import *

torch.cuda.empty_cache()
gc.collect()

# Global setup
objective_config = dict()
error_logs = []
processing_logs = []
current_gpu_id = 1
start_time = datetime.now()
df = pd.DataFrame()

print(f"Starting Training Now: {datetime.now()}")




X_train_tensor=y_train_tensor=X_valid_tensor=y_valid_tensor=y_train_sampler=None

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



def handle_infinity_float_exception(df):
    # Handle: ValueError: Input X contains infinity or a value too large for dtype('float32').
    df = df.astype(np.float32)  # Convert all columns to float32 in one go

    # Create a mask for inf and too large values
    inf_mask = np.isinf(df)
    too_large_mask = np.abs(df) > np.finfo(np.float32).max

    # Combine both masks
    mask = inf_mask | too_large_mask

    # Replace values with 0 using the mask
    df[mask] = 0

    return df


def get_sampler(y):
    if type(y) == torch.Tensor:
        y = pd.Series(y.numpy())

    class_counts = np.bincount(y.astype(int))
    weights = 1. / class_counts
    sample_weights = weights[y.astype(int)]
    # multiplier =df_train['ENMO_std']/max(df_train['ENMO_std'])
    # sample_weights = sample_weights * multiplier
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    return sampler


def read_param(path):
    best_param_file_path = os.path.join(optuna_best_param_path, "best_trial_params.txt")
    optuna_logs_file_path = os.path.join(optuna_best_param_path, "optuna_trial_processing_logs.log")
    try:
        best_parameters = dict()
        with open(best_param_file_path, "r") as file_h:
            best_parameters = file_h.read()
            best_parameters = ast.literal_eval(best_parameters)

        # with open(optuna_logs_file_path, "r") as file:
        #     data = file.read()
        #     test_subject = re.findall(r"US\d+", data)[0]

        best_parameters['test_subject_id'] = "US10001001"  # test_subject
        return best_parameters
    except Exception as e:
        raise Exception(f"Failed to Load Best Params. File: {best_param_file_path}, exception: {e}")

def load_optuna_pretrained_best_params(optuna_best_param_path):
    best_param_file_path = os.path.join(optuna_best_param_path, "optuna_trial_best_params.txt")
    if os.path.exists(best_param_file_path):
        return read_param(best_param_file_path)
    else:
        params_list={'1':{'batch_size': 16, 'lr': 0.0001, 'dropout': 0.1, 'droppath': 0.2, 'kernel_f': 7, 'kernel_MBA': 5, 'num_feature_layers': 5, 'blocks_MBA': 5, 'MBA_encoder': True, 'wl1': 0.8, 'wl2': 0.4, 'wl3': 0.9, 'optim': 'RAdam', 'pos_weight_l1': True, 'pos_weight_l2': False, 'cls_token': False, 'featurelayer': 'ResTCN', 'num_heads': 2, 'stratify': 'undersample'},
                     '2': {'batch_size': 16, 'lr': 0.001, 'dropout': 0.5, 'droppath': 0.1, 'kernel_f': 7, 'kernel_MBA': 15, 'num_feature_layers': 12, 'blocks_MBA': 7, 'MBA_encoder': True, 'wl1': 0.7, 'wl2': 0.8, 'wl3': 0.3, 'optim': 'AdamW', 'pos_weight_l1': False, 'pos_weight_l2': True, 'cls_token': False, 'featurelayer': 'ResTCN', 'num_heads': 4, 'stratify': 'undersample'},
                     '3': {'batch_size': 16, 'lr': 0.001, 'dropout': 0.4, 'droppath': 0.2, 'kernel_f': 7, 'kernel_MBA': 3, 'num_feature_layers': 11, 'blocks_MBA': 4, 'MBA_encoder': True, 'wl1': 0.1, 'wl2': 0.6, 'wl3': 0.4, 'optim': 'RAdam', 'pos_weight_l1': True, 'pos_weight_l2': False, 'cls_token': False, 'featurelayer': 'TCN', 'num_heads': 4, 'stratify': 'undersample'},
                     '4': {'batch_size': 64, 'lr': 0.001, 'dropout': 0.3, 'droppath': 0.2, 'kernel_f': 15, 'kernel_MBA': 13, 'num_feature_layers': 6, 'blocks_MBA': 5, 'MBA_encoder': True, 'wl1': 0.3, 'wl2': 0.4, 'wl3': 0.2, 'optim': 'RMSprop', 'pos_weight_l1': False, 'pos_weight_l2': True, 'cls_token': False, 'featurelayer': 'TCN', 'num_heads': 2, 'stratify': 'undersample'},
                     '5': {'batch_size': 16,'num_filters': 256, 'lr': 0.001, 'dropout': 0.3, 'droppath': 0.2, 'kernel_f': 15, 'kernel_MBA': 13, 'num_feature_layers': 6, 'blocks_MBA': 5, 'MBA_encoder': True, 'wl1': 0.3, 'wl2': 0.4, 'wl3': 0.2, 'optim': 'RMSprop', 'pos_weight_l1': False, 'pos_weight_l2': True, 'cls_token': False, 'featurelayer': 'TCN', 'num_heads': 2, 'stratify': 'undersample'},
                     '6': {'batch_size': 16, 'num_filters': 256, 'lr': 0.0001, 'dropout': 0.3, 'droppath': 0.2, 'kernel_f': 15, 'kernel_MBA': 9, 'num_feature_layers': 7, 'blocks_MBA': 4, 'MBA_encoder': True, 'wl1': 0.8, 'wl2': 0.1, 'wl3': 0.9, 'optim': 'RMSprop', 'pos_weight_l1': False, 'pos_weight_l2': False, 'cls_token': False, 'featurelayer': 'TCN', 'num_heads': 2, 'stratify': 'undersample'},
                     '7': {'batch_size': 16, 'num_filters': 256, 'lr': 0.0001, 'dropout': 0.3, 'droppath': 0.2, 'kernel_f': 15, 'kernel_MBA': 9, 'num_feature_layers': 7, 'blocks_MBA': 4, 'MBA_encoder': True, 'wl1': 0.8, 'wl2': 0.1, 'wl3': 0.9, 'optim': 'RMSprop', 'pos_weight_l1': False, 'pos_weight_l2': False, 'cls_token': False, 'featurelayer': 'ResTCN', 'num_heads': 2, 'stratify': 'undersample'},
                     '8': {'batch_size': 16, 'num_filters': 256, 'lr': 0.0001, 'dropout': 0.3, 'droppath': 0.2, 'kernel_f': 15, 'kernel_MBA': 9, 'num_feature_layers': 7, 'blocks_MBA': 4, 'MBA_encoder': True, 'wl1': 0.8, 'wl2': 0.1, 'wl3': 0.9, 'optim': 'RMSprop', 'pos_weight_l1': False, 'pos_weight_l2': True, 'cls_token': False, 'featurelayer': 'TCN', 'num_heads': 2, 'stratify': 'undersample'},
                     '9': {'batch_size': 16, 'num_filters': 256, 'lr': 0.0001, 'dropout': 0.3, 'droppath': 0.2, 'kernel_f': 15, 'kernel_MBA': 9, 'num_feature_layers': 7, 'blocks_MBA': 4, 'MBA_encoder': True, 'wl1': 0.8, 'wl2': 0.1, 'wl3': 0.9, 'optim': 'RMSprop', 'pos_weight_l1': False, 'pos_weight_l2': True, 'cls_token': True, 'featurelayer': 'TCN', 'num_heads': 2, 'stratify': 'undersample'},
                     '10': {'batch_size': 16, 'num_filters': 64, 'lr': 0.001, 'dropout': 0.0, 'droppath': 0.1, 'kernel_f': 9, 'kernel_MBA': 3, 'num_feature_layers': 10, 'blocks_MBA': 7, 'MBA_encoder': True, 'wl1': 0.7000000000000001, 'wl2': 0.6000000000000001, 'wl3': 0.5, 'optim': 'AdamW', 'pos_weight_l1': False, 'pos_weight_l2': True, 'cls_token': False, 'featurelayer': 'TCN', 'num_heads': 2, 'stratify': 'undersample'},
                     '11': {'batch_size': 16, 'num_filters': 128, 'lr': 0.001, 'dropout': 0.2, 'droppath': 0.4, 'kernel_f': 13, 'kernel_MBA': 11, 'num_feature_layers': 7, 'blocks_MBA': 7, 'MBA_encoder': True, 'wl1': 0.5, 'wl2': 0.5, 'wl3': 0.9, 'optim': 'RMSprop', 'pos_weight_l1': False, 'pos_weight_l2': True, 'cls_token': True, 'featurelayer': 'ResTCN', 'num_heads': 2, 'stratify': 'undersample'},
                     '12': {'batch_size': 16, 'num_filters': 64, 'lr': 0.001, 'dropout': 0.2, 'droppath': 0.4, 'kernel_f': 13, 'kernel_MBA': 11, 'num_feature_layers': 7, 'blocks_MBA': 7, 'MBA_encoder': True, 'wl1': 0.5, 'wl2': 0.5, 'wl3': 0.9, 'optim': 'RMSprop', 'pos_weight_l1': False, 'pos_weight_l2': True, 'cls_token': False, 'featurelayer': 'ResTCN', 'num_heads': 4, 'stratify': 'undersample'},
                     '13': {'batch_size': 32, 'num_filters': 64, 'lr': 0.001, 'dropout': 0.2, 'droppath': 0.4, 'kernel_f': 13, 'kernel_MBA': 11, 'num_feature_layers': 7, 'blocks_MBA': 7, 'MBA_encoder': True, 'wl1': 0.9, 'wl2': 0.5, 'wl3': 12, 'optim': 'RMSprop', 'pos_weight_l1': False, 'pos_weight_l2': True, 'cls_token': True, 'featurelayer': 'ResTCN', 'num_heads': 4, 'stratify': 'adaptive_undersample', "random_padding": False, "average_mask": False, "average_window_size": 20, "supcon_loss": False, "mixup_alpha": 0.1, "mixup_prob": 0.7, "reg_only": False, "norm_scratch_length": True, "padding_value": -999.0, "dynamic_batch_padding": False, "use_feature_extractor": True},
                     '14': {'batch_size': 16, 'lr': 0.0005, 'dropout': 0.2, 'droppath': 0.3, 'kernel_MBA': 5, 'blocks_MBA': 4, 'MBA_encoder': True, 'wl1': 1.2, 'wl2': 0.3, 'wl3': 8.0, 'optim': 'AdamW', 'pos_weight_l1': False, 'pos_weight_l2': True, 'cls_token': True, 'num_heads': 4, 'stratify': 'undersample', "average_mask": False, "average_window_size": 20, "supcon_loss": True, "mixup_alpha": 0.15, "mixup_prob": 0.7, "reg_only": False, "norm_scratch_length": True, "embed_dim": 192, "use_embedding": False, "window_size": 60, "stride": 20},

                    }

        patchtst_params_list = {
            '1': {'batch_size': 16, 'lr': 0.001, 'dropout': 0.2, 'wl1': 0.5, 'wl2': 0.5, 'wl3': 0.9, 'optim': 'AdamW', 'pos_weight_l1': False, 'pos_weight_l2': True, 'stratify': 'undersample', "norm_scratch_length": False, "reg_head": True, }
        }
        
        efficient_unet_params_list = {
            '1': {'batch_size': 16, 'd_model': 1024, 'tcn_num_filters': 3, 'tcn_kernel_size': 7, 'tcn_num_blocks': 6, 'lr': 0.001, 'wl1': 2, 'wl2': 0.5, 'wl3': 0.005, 'optim': 'AdamW', 'pos_weight_l1': False, 'pos_weight_l2': True, 'featurelayer': 'Conv2d', 'stratify': 'undersample', "target_shape": 64, "norm_scratch_length": False, "reg_head": True},
            '2': {'batch_size': 16, 'd_model': 1024, 'tcn_num_filters': 3, 'tcn_kernel_size': 7, 'tcn_num_blocks': 8, 'lr': 0.001, 'wl1': 0.5, 'wl2': 0.5, 'wl3': 0.9, 'optim': 'AdamW', 'pos_weight_l1': False, 'pos_weight_l2': True, 'featurelayer': 'Conv2d', 'stratify': 'undersample', "target_shape": 64, "norm_scratch_length": False, "reg_head": False, "mask_padding": False},
            '3': {'batch_size': 16, 'd_model': 1024, 'tcn_num_filters': 3, 'tcn_kernel_size': 7, 'tcn_num_blocks': 8, 'lr': 0.001, 'wl1': 0.5, 'wl2': 0.2, 'wl3': 0.9, 'optim': 'AdamW', 'pos_weight_l1': False, 'pos_weight_l2': True, 'featurelayer': 'Conv2d', 'stratify': 'undersample', "target_shape": 64, "norm_scratch_length": True, "reg_head": True, "mask_padding": False},
            '4': {'batch_size': 16, 'd_model': 1024, 'tcn_num_filters': 3, 'tcn_kernel_size': 7, 'tcn_num_blocks': 8, 'lr': 0.001, 'wl1': 0.5, 'wl2': 0.5, 'wl3': 0.9, 'optim': 'AdamW', 'pos_weight_l1': False, 'pos_weight_l2': True, 'featurelayer': 'Conv2d', 'stratify': 'undersample', "target_shape": 64, "norm_scratch_length": True, "reg_head": True, "mask_padding": False, "random_padding": True, "average_mask": True, "average_window_size": 20},
        }

        vit_params_list = {
            '1': {'batch_size': 16, 'd_model': 1024, 'tcn_num_filters': 64, 'tcn_kernel_size': 7, 'tcn_num_blocks': 6, 'lr': 0.001, 'wl1': 0.5, 'wl2': 0.3, 'wl3': 0.9, 'optim': 'AdamW', 'pos_weight_l1': False, 'pos_weight_l2': True, 'featurelayer': 'Conv2d', 'stratify': 'undersample', "target_shape": 128, "norm_scratch_length": True, "reg_head": True},
        }

        swin_params_list = {
            '1': {'batch_size': 16,  'tcn_num_filters': 64, 'tcn_kernel_size': 7, 'tcn_num_blocks': 6, 'lr': 0.001, 'wl1': 0.5, 'wl2': 0.3, 'wl3': 0.9, 'optim': 'AdamW', 'pos_weight_l1': False, 'pos_weight_l2': True, 'featurelayer': 'Conv2d', 'stratify': 'undersample', "target_shape": 128, "norm_scratch_length": False, "reg_head": False},
        }

        lsm2_params_list = {
            '1': {'batch_size': 16, 'patch_size': 200, 'stride': 50, 'encoder_layers': 6, 'decoder_layers': 0, 'd_model': 192, 'num_heads': 4, 'lr': 3e-4, 'wl1': 0.8, 'wl2': 0.5, 'wl3': 10, 'optim': 'AdamW', 'pos_weight_l1': False, 'pos_weight_l2': True, 'stratify': "undersample", "norm_scratch_length": True, "reg_head": True, "random_padding": False, "average_mask": False, "average_window_size": 20, "padding_value": -10.0, "supcon_loss": True,  "mixup_alpha": 0.1, "mixup_prob": 0.7},
        }

        conv1d_params_list = {
            '1': {'batch_size': 16, 'num_layers': 5, 'd_model': 64, 'lr': 3e-4, 'wl1': 0.9, 'wl2': 0.5, 'wl3': 10, 'optim': 'AdamW', 'pos_weight_l1': False, 'pos_weight_l2': True, 'stratify': "undersample", "norm_scratch_length": True, "reg_head": True, "random_padding": False, "average_mask": False, "padding_value": 0.0, "supcon_loss": True, "mixup_alpha": 0.1, "mixup_prob": 0.7, "reg_only": False,},
        }

        resnet_1d_params_list = {
            '1': {'batch_size': 16, 'lr': 3e-4, 'wl1': 0.9, 'wl2': 0.5, 'wl3': 0.1, 'optim': 'AdamW', 'pos_weight_l1': False, 'pos_weight_l2': True, 'stratify': "undersample", "norm_scratch_length": False, "random_padding": False, "average_mask": False, "padding_value": -10, "supcon_loss": True, "mixup_alpha": 0.1, "mixup_prob": 0.7},
        }


        mba_patch = {
            '1': {'batch_size': 16, "window_size": 60, "stride": 20, "embed_dim": 128, "use_embedding": True, 'lr': 0.001, 'dropout': 0.2, 'droppath': 0.3, 'kernel_MBA': 7, 'blocks_MBA': 7, 'MBA_encoder': True, 'wl1': 0.9, 'wl2': 0.5, 'wl3': 12.0, 'optim': 'AdamW', 'pos_weight_l1': False, 'pos_weight_l2': True, 'cls_token': True, 'num_heads': 4, 'stratify': 'undersample', "average_mask": False, "average_window_size": 20, "supcon_loss": True, "mixup_alpha": 0.1, "mixup_prob": 0.7, "reg_only": False, "norm_scratch_length": True, },
        }

        # return patchtst_params_list['1']
        return params_list['13']
        # return efficient_unet_params_list['4']
        # return vit_params_list['1']
        # return swin_params_list['1']
        # return lsm2_params_list['1']
        # return conv1d_params_list['1']
        # return resnet_1d_params_list['1']
        # return mba_patch['1']

#         return {'batch_size': 16, 'lr': 0.002, 'dropout': 0.2, 'droppath': 0.3, 'kernel_f': 7, 'kernel_MBA': 11, 'blocks_MBA': 0, 'wl1': 0.6, 'wl2': 0.9, 'wl3': 0.7, 'optim': 'AdamW', 'pos_weight_l1': False, 'pos_weight_l2': True, 'stratify': 'undersample'}
#         return {'batch_size': 16, 'lr': 0.000646662682506144, 'dropout': 0.3, 'droppath': 0.5, 'kernel_f': 11, 'kernel_MBA': 9, 'blocks_MBA': 5, 'wl1': 0.7, 'wl2': 0.8, 'wl3': 0.3, 'optim': 'RMSprop', 'pos_weight_l1': False, 'pos_weight_l2': False, 'stratify':'undersample'}
    
    


def generate_unique_random(min_val, max_val, excluded):
    possible_numbers = set(range(min_val, max_val + 1)) - set(excluded)
    if not possible_numbers:
        return None  # No available number exists
    return random.choice(list(possible_numbers))

# ---------------------
# TRAIN FUNCTION RAY
# ---------------------


def load_model_from_checkpoint(model, checkpoint_dir):
    """
    Load the trained model from the last checkpoint in a directory.

    Args:
    checkpoint_dir (str): Directory containing the model checkpoint.

    Returns:
    torch.nn.Module: Loaded model.
    """
    checkpoint_path = os.path.join(checkpoint_dir, "model.pt")
    device=torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path,map_location=device)
    model_state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(model_state_dict)
    return model


def run_model(model,
              df,
              batch_size,
              train_mode,
              device,
              optimizer,
              scheduler,
              stratify=False,
              get_predictions=True,
              use_TSO=True,
              wl1=1,
              wl2=2,
              wl3=0.1,
              verbose=True,
              pos_weight_l1=False,
              pos_weight_l2=False,
              pretraining=False, 
              max_seq_len=None, 
              random_padding=False,
              average_mask=False,
              average_window_size=20,
              padding_value=0.0,
              ignore_padding_in_mask_loss=False,
              mixup_prob=0.0,
              mixup_alpha=0.2,
              ):

    epoch_loss = batches  = 0.0
    gt1s = gt2s = gt3s = lens = inTSOs = predictTSOs = []
    pr1s = pr1_probs = pr2s = pr2_probs = pr3s = attentions = []
    loss1_s=[]
    loss2_s=[]
    loss3_s=[]
    seg_column='segment'
    segments1=segments2=positions=[]

    # if train_mode:
    #     model.train()
    # else:
    #     model.eval()

    # total_params = 0
    # trainable_params = 0
    # for name, param in model.named_parameters():
    #     total_params += param.numel()
    #     if param.requires_grad:
    #         print(f"Parameters trainabale {name}")
    #         trainable_params += param.numel()
    #     else:
    #         print(f"Parameters not trainable: {name}")
    # print(f"Percent trainable: {trainable_params}/{total_params}")

    for batch in batch_generator(df=df,batch_size=batch_size,stratify=stratify,shuffle=train_mode,seg_column=seg_column):    
        
        padding_position = "random" if random_padding else "tail"
        batch_data, label1, label2, label3, x_lens, seq_start_idx = add_padding_with_position(batch, device, 
                                                                            seg_column=seg_column, 
                                                                            max_seq_len=max_seq_len, 
                                                                            padding_position=padding_position,
                                                                            is_train=train_mode,
                                                                            prob=0.5,
                                                                            padding_value=padding_value)
        apply_mixup = train_mode and torch.rand(1).item() < mixup_prob
        outputs1, outputs2, outputs3, att, con_embed, mixup_info = model(batch_data, x_lens=x_lens, apply_mask=False, labels=label1, apply_mixup=apply_mixup, mixup_alpha=mixup_alpha, padding_value=padding_value)
        if mixup_info is not None:
            mixup_label1, mixup_label3, mixup_lambda = mixup_info
        else:
            mixup_label1 = None 
            mixup_label3 = None
            mixup_lambda = 1.0
        outputs1 = outputs1.view(-1)  
        # outputs2 = outputs2.reshape(-1)
        outputs3 = outputs3.view(-1)

        loss, loss1, loss2, loss3, outputs2, label2 = measure_loss_multitask_with_padding(outputs1,
                                                                    outputs2,
                                                                    outputs3,
                                                                    label1,
                                                                    label2,
                                                                    label3,
                                                                    wl1,
                                                                    wl2,
                                                                    wl3,
                                                                    contrastive_embedding=con_embed,
                                                                    mixup_label1=mixup_label1,
                                                                    mixup_label3=mixup_label3,
                                                                    mixup_lambda=mixup_lambda,
                                                                    padding_position=padding_position,
                                                                    pos_weight_l1=pos_weight_l1,
                                                                    pos_weight_l2=pos_weight_l2,
                                                                    x_lengths=x_lens,
                                                                    seq_start_idx=seq_start_idx,
                                                                    average_mask=average_mask,
                                                                    average_window_size=average_window_size,
                                                                    ignore_padding_in_mask_loss=ignore_padding_in_mask_loss)
        
        loss1_s.append(loss1.item())
        loss2_s.append(loss2.item())
        loss3_s.append(loss3.item())
        
        # Backward pass and optimization if in training mode
        if train_mode:
            # print(optimizer.param_groups[0]['lr'], loss)
            optimizer.zero_grad()
            loss.backward()
            # clip gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
        epoch_loss += loss.item()
        batches += 1
        #Measure performance using the unpadded data
        pr1_prob = torch.sigmoid(outputs1)
        pr1 = torch.round(pr1_prob)
        pr1 = torch.nan_to_num(pr1)
        pr1s = np.concatenate((pr1s,pr1.cpu().detach().numpy()))
        pr1_probs = np.concatenate((pr1_probs, pr1_prob.cpu().detach().numpy()))
        gt1s = np.concatenate((gt1s, label1.cpu().detach().numpy()))
        
        pr2_prob = torch.sigmoid(outputs2)
        pr2 = torch.round(pr2_prob)  # threshold of 0.5 for binary classification #TODO: check NAs
        pr2 = pr2.cpu().detach().numpy()
        pr3 = outputs3
        
        #label2 = batch.scratch.values * 1
        if np.isnan(pr2).any():
            if verbose:
                print("Predictions contain NA. Replacing by 0")
            print(pr2.shape)
            print(label2.shape)
            print(pr2)
            print(label2)
            pr2 = np.nan_to_num(pr2)
            pr3 = torch.nan_to_num(pr3)
            # remove padding from the step level predictions
            #  pr2 = remove_padding(pr2,x_lens)
            #  pr2_prob= remove_padding(pr2_prob.cpu().detach().numpy(),x_lens)
            #  att=remove_padding(att.cpu().detach().numpy(),x_lens)
        pr2s = np.concatenate((pr2s, pr2))
        pr2_probs = np.concatenate((pr2_probs, pr2_prob.cpu().detach().numpy()))
        # attentions=np.concatenate((attentions,att.cpu().detach().numpy()))
        # pr2s_2=np.concatenate((pr2s_2, seq_to_seconds(pr2,y_lens)))
        # pr2s_3=np.concatenate((pr2s_3,seq_to_seconds(smooth_binary_series(pr2,20,20),y_lens)))
        gt2s=np.concatenate((gt2s, label2.cpu().detach().numpy()))
        # gt2s_2=np.concatenate((gt2s_2, seq_to_seconds(label2,x_lens)))

        pr3s = np.concatenate((pr3s, pr3.cpu().detach().numpy()))
        gt3s = np.concatenate((gt3s, label3.cpu().detach().numpy()))

        segments1=np.concatenate((segments1, batch[seg_column].unique()))
        segments2=np.concatenate((segments2, batch[seg_column].values))
        positions=np.concatenate((positions, batch.position_segment.values))
        if use_TSO:
            inTSOs=np.concatenate((inTSOs, batch.inTSO.values)) 
            predictTSOs=np.concatenate((predictTSOs, batch.predictTSO.values)) 
    
    pr1s=np.nan_to_num(pr1s, nan=0, posinf=0, neginf=0)
    pr2s=np.nan_to_num(pr2s, nan=0, posinf=0, neginf=0)
    pr3s=np.nan_to_num(pr3s, nan=0, posinf=0, neginf=0)
    
    
    # print(segments2.shape, gt3s.shape, gt2s.shape, pr2s.shape, positions.shape)
    if get_predictions:
        content1 = {"segment":segments1,
                    "gt1": gt1s,
                    "pr1": pr1s,
                    "pr1_probs": pr1_probs}
        content2 = {"segment": segments2,
                    "position": positions, }
        content1["gt3"] = gt3s
        content1["pr3"] = pr3s
        gt2s_len = gt2s.size
        # size is different with segments2 values
        padding = (0, max(segments2.size - gt2s.size, 0))
        # print(f"Padding length: {padding}; gt2s length: {gt2s_len}")
            
        content2["gt2"] = np.pad(gt2s, padding).astype(int)
        content2["pr2"] = np.pad(pr2s, padding).astype(int)
        content2["pr2_probs"] = np.pad(pr2_probs, padding)
            
        # content2["attentions"] = attentions
        
        predictions1 = pd.DataFrame(content1)
        predictions2 = pd.DataFrame(content2)

        if use_TSO:
            predictions2["inTSO"] = inTSOs
        predictions = predictions2.merge(predictions1)
    else:
        predictions=pd.dataFrame()
    if use_TSO:
        predictions_inTSO=predictions[predictions.inTSO==True]
        # print(f"Train={train_mode}-GT: pos={predictions_inTSO.gt1.sum()}/predict: pos={predictions_inTSO.pr1.sum()}/total={len(predictions_inTSO.gt1)}")
        if not train_mode:
            print("Output1 confusion matrix: ", metrics.confusion_matrix(predictions_inTSO.gt1, predictions_inTSO.pr1))
        eval_metrics={'F1_1': "%.2f" % (metrics.f1_score(predictions_inTSO.gt1, predictions_inTSO.pr1,zero_division=0)*100),
                        # 'F1_2_2': "%.2f" % (metrics.f1_score(gt2s_2, pr2s_2,zero_division=0)*100),
                        # 'F1_2_3': "%.2f" % (metrics.f1_score(gt2s_2, pr2s_3,zero_division=0)*100),
                        'Accuracy1': "%.2f" % (metrics.accuracy_score(predictions_inTSO.gt1, predictions_inTSO.pr1)*100)
                        }
    else:
        eval_metrics={'F1_1': "%.3f" % (metrics.f1_score(gt1s, pr1s, zero_division=0)*100),
                    'Accuracy1': "%.2f" % (metrics.accuracy_score(gt1s, pr1s)*100)
                    }
    eval_metrics.update({'loss': "%.4f" % (epoch_loss/ batches),
                         'steps':batches})

    # print(np.unique(predictions_inTSO.gt2[:gt2s_len]))
    # print(np.unique(predictions_inTSO.pr2[:gt2s_len]))
    if use_TSO:
        eval_metrics.update({
            'F1_2': "%.2f" % (metrics.f1_score(predictions_inTSO.gt2[:gt2s_len], predictions_inTSO.pr2[:gt2s_len], zero_division=0)*100),
            'Accuracy2': "%.2f" % (metrics.accuracy_score(predictions_inTSO.gt2[:gt2s_len], predictions_inTSO.pr2[:gt2s_len])*100),
            'R2' : "%.2f" % (metrics.r2_score(predictions_inTSO.gt3, predictions_inTSO.pr3)),
        })
    else:
        eval_metrics.update({
                'F1_2': "%.2f" % (metrics.f1_score(gt2s[:gt2s_len], pr2s[:gt2s_len],zero_division=0)*100),
                # 'F1_2_2': "%.2f" % (metrics.f1_score(gt2s_2, pr2s_2,zero_division=0)*100),
                # 'F1_2_3': "%.2f" % (metrics.f1_score(gt2s_2, pr2s_3,zero_division=0)*100),
                'Accuracy2': "%.2f" % (metrics.accuracy_score(gt2s[:gt2s_len], pr2s[:gt2s_len])*100),
                # 'MSE': "%.2f" % (metrics.mean_squared_error(gt3s, pr3s)),
                'R2' : "%.4f" % (metrics.r2_score(predictions.gt3, predictions.pr3)),
                })
    eval_metrics.update({
                'MSE': "%.2f" % (metrics.mean_squared_error(predictions.gt3, predictions.pr3)),
                'loss1': "%.2f" % (np.mean(loss1_s)),
                'loss2': "%.2f" % (np.mean(loss2_s)),
                'loss3': "%.2f" % (np.mean(loss3_s)),
    })
                
    return model,eval_metrics,predictions

def DA_Permute(df):
    _df=df
    _df.x,_df.y,_df.z=df.y,df.z,df.y
    return _df

def DA_Scaling(df, sigma=0.1):
    _df=df
    xNoise = np.matmul(np.ones(df.x.shape[0]), np.random.normal(loc=1.0, scale=sigma, size=df.x.shape[0]))
    yNoise = np.matmul(np.ones(df.y.shape[0]), np.random.normal(loc=1.0, scale=sigma, size=df.y.shape[0]))
    zNoise = np.matmul(np.ones(df.z.shape[0]), np.random.normal(loc=1.0, scale=sigma, size=df.z.shape[0]))
    _df.x,_df.y,_df.z=_df.x*xNoise,_df.y*yNoise,_df.z*zNoise
    return _df           

def train_pipeline(args):
    global df, pre_saved_scaler, scaler_path, execution_mode,X_train_tensor,y_train_tensor,X_valid_tensor,y_valid_tensor,y_train_sampler  # todo: remove/debug
    # setup mlflow logging
    # mlflow.autolog()

    clear_tracker_flag = str(args.clear_tracker)
    model_name = args.model
    execution_mode = args.execution_mode #
    input_data_folder = str(args.input_data_folder)
    input_features_type = args.features_type
    results_folder_name = args.output
    motion_filter = args.motion_filter
    motion_threshold=args.motion_threshold
    data_augmentation=args.data_augmentation
    tune_path=str(args.tune_path)
    epochs=args.epochs
    balanced = str(args.balanced)
    num_gpu = str(args.num_gpu)
    remove_hnv=str(args.remove_hnv)
    testing=args.testing
    pretraining= (str(args.pretraining).lower()=="true")
    training_iterations=args.training_iterations
    
    if "/raw" in input_data_folder:
        dataset_type = os.path.basename(input_data_folder.rstrip("/").replace("/raw", ""))
    elif "data_temp" in input_data_folder:
        # dataset_type = "geneactive_20hz_2s_b1s_imerit_sleep_nonwearOR"
        # dataset_type = "geneactive_20hz_2s_b1s_imerit_sleep_nonwearOR_5smerge_nograv_change"
        dataset_type = "geneactive_20hz_2s_b1s_imerit_sleep_nonwearOR_5smerge_nograv_nochange"
    else:
        dataset_type = os.path.basename(input_data_folder.rstrip("/"))
    # results_folder = f"/mnt/data/GENEActive-featurized/results/DL/{dataset_type}/{results_folder_name}/"
    results_folder = f"/domino/datasets/GENEActive-featurized/results/DL/{dataset_type}/{results_folder_name}/"
    if tune_path=="":
        param_tuning_output_folder = os.path.join(results_folder, "tuning/")
    else:
        param_tuning_output_folder = tune_path
    training_output_folder = os.path.join(results_folder, "training/")
    job_tracking_folder = os.path.join(training_output_folder, "job_tracking_folder")
    predictions_output_folder = os.path.join(training_output_folder, "predictions/")
    confusion_matrix_plots_folder = os.path.join(training_output_folder, "confusion_matrix_plots/")
    learning_plots_output_folder = os.path.join(training_output_folder, "learning_plots/")
    roc_auc_plots_output_folder = os.path.join(training_output_folder, "roc_auc_plots/")
    train_models_folder = os.path.join(training_output_folder, "model_weights/")
    train_models_folder_joblib = os.path.join(training_output_folder, "model_weights_joblib/")
    checkpoint_folder=os.path.join(training_output_folder, "checkpoints/")
    training_logs_folder=os.path.join(training_output_folder, "training_logs/")

    if clear_tracker_flag.lower()=="true" and os.path.exists(job_tracking_folder):
        shutil.rmtree(job_tracking_folder, ignore_errors=True)

    create_folder([results_folder, training_output_folder, job_tracking_folder, predictions_output_folder,
                   confusion_matrix_plots_folder, learning_plots_output_folder, roc_auc_plots_output_folder,
                   train_models_folder, train_models_folder_joblib,training_logs_folder])

    # Write user provided arguments
    with open(os.path.join(results_folder, "user_arguments.txt"), "w") as file_handler:
        file_handler.write(str(args))

    folder_paths = {
        "input_data_folder": input_data_folder,
        "results_folder": results_folder,
        "training_output_folder": training_output_folder,
        "job_tracking_folder": job_tracking_folder,
        "predictions_output_folder": predictions_output_folder,
        "confusion_matrix_plots_folder": confusion_matrix_plots_folder,
        "learning_plots_output_folder": learning_plots_output_folder,
        "roc_auc_plots_output_folder": roc_auc_plots_output_folder,
        "train_models_folder": train_models_folder,
        "train_models_folder_joblib": train_models_folder_joblib,
        "checkpoint_folder":checkpoint_folder,
        "training_logs_folder":training_logs_folder
    }

    param_tuning_subject_id = "" #"US10001001"  # manual feed
    best_params = load_optuna_pretrained_best_params(optuna_best_param_path=param_tuning_output_folder)  # check params
    print("Best params selected: ", best_params)
    
    
    # Setup device
    if args.num_gpu == "NA":
        device_ids = [x for x in range(0, torch.cuda.device_count(), 1)]
        np.random.shuffle(device_ids)
        num_gpu = device_ids[0]
    else:
        num_gpu = args.num_gpu
    device = torch.device(f"cuda:{num_gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Process each data file (split)

    # Detect data loading method: check if CSV files exist
    csv_files = [f for f in os.listdir(args.input_data_folder) if f.endswith('_train.csv')]
    use_csv_files = len(csv_files) > 0
    
    if use_csv_files:
        print("Found CSV files. Loading from pre-processed CSV files.")
        splits = [f.replace('_train.csv', '') for f in csv_files]
    else:
        print("No CSV files found. Loading from folder structure.")
        # Original folder-based loading approach
        use_TSO = False
        if "nonwearOR" in args.input_data_folder:
            use_TSO = True
        
        # Load data based on dataset type
        if 'nsucl' in args.input_data_folder:
            df = load_data_nsucl(args.input_data_folder)
        else:
            df = load_data(args.input_data_folder, max_seq_length=None, norm_scratch_length=best_params.get("norm_scratch_length", True), use_TSO=use_TSO, energy_th=1)

        # Calculate max sequence length for metadata
        max_seq_len = df.groupby('segment').count().max(1).max()+1
        print(f"Max sequence length: {max_seq_len}")
        
        # Define PID name based on testing strategy
        if args.testing == "production":
            df["FOLD"] = "All"
            PID_name = "FOLD"
            splits = ["All"]
        elif args.testing == "LOSO":
            PID_name = "PID"
            splits = df[PID_name].unique()
        else:  # LOFO
            PID_name = "FOLD"
            splits = df[PID_name].unique()
    
    # Process each split
    for split_id in splits:
        print(f"Processing split: {split_id}")
        # Skip if already processed
        if os.path.exists(os.path.join(job_tracking_folder, split_id)):
            print(f"Split {split_id} already processed. Skipping.")
            continue
        # Mark as in progress
        with open(os.path.join(job_tracking_folder, split_id), "w") as file:
            file.write("-")  # placeholder file
        
        if use_csv_files:
            # Load from CSV files (already pre-processed and split)
            print(f"Loading pre-processed CSV files for split: {split_id}")
            df_train = pd.read_csv(os.path.join(args.input_data_folder, f"{split_id}_train.csv"), low_memory=False)
            df_val = pd.read_csv(os.path.join(args.input_data_folder, f"{split_id}_val.csv"), low_memory=False)
            df_test = pd.read_csv(os.path.join(args.input_data_folder, f"{split_id}_test.csv"), low_memory=False)
            max_seq_len = df_train.groupby('segment').count().max(1).max()+1
            print(f"Loaded CSV data. Max sequence length: {max_seq_len}")
        else:
            # Original folder-based approach with train/test splitting
            # Create train/test split
            if args.testing == "production":
                df_test = df
                df_train = df
            else:
                df_test = df[df[PID_name] == split_id]
                df_train = df[df[PID_name] != split_id]
                
                if use_TSO:
                    df_train = df_train[df_train.inTSO==True]
                    df_test = df_test[df_test.inTSO==True]
            
            # Create validation split
            random_indices = df_subset_segments(df_train, 0.2, False)
            df_val = df_train.iloc[df_train.index.isin(random_indices)]
            df_train = df_train.iloc[~df_train.index.isin(random_indices)]
            
            if data_augmentation > 0:
                print(f"Augmenting training data with {data_augmentation} iterations. {datetime.now()}")
                intitial_size=df_train.size
                if 'nsucl' in dataset_type:
                    df_train=augment_dataset(df_train,data_augmentation,verbose=False)  
                else:
                    df_train=augment_dataset(df_train,data_augmentation,interchange=False,verbose=False, norm_scratch_length=best_params.get("norm_scratch_length", True))
                max_seq_len=df_train.groupby('segment').count().max(1).max()+1
                print(f"Training data augmented from {intitial_size} to {df_train.size}. Max sequence length: {max_seq_len}. {datetime.now()}") 

            # Scale features
            scaler = StandardScaler()
            c_to_scale = ['x', 'y', 'z']
            df_train.loc[:, c_to_scale] = scaler.fit_transform(df_train[c_to_scale])
            df_val.loc[:, c_to_scale] = scaler.transform(df_val[c_to_scale])
            df_test.loc[:, c_to_scale] = scaler.transform(df_test[c_to_scale])

        # Get max sequence length from metadata
        # max_seq_len = metadata['max_seq_len']
        max_seq_len = df_train.groupby('segment').count().max(1).max()+1

        # Training parameters
        batch_size = best_params['batch_size']
        wl1 = best_params["wl1"]
        wl2 = best_params["wl2"]
        wl3 = best_params["wl3"]
        pos_weight_l1 = best_params['pos_weight_l1']
        pos_weight_l2 = best_params['pos_weight_l2']
        stratify = best_params['stratify']

        # Training loop
        retrain = Retraining(verbose=True)
        
        for i in range(args.training_iterations):
            print(f"Training iteration {i+1}/{args.training_iterations}")
            
            # Setup model
            model = setup_model(
                model_name=args.model,
                input_tensor_size=3,
                max_seq_len=max_seq_len,
                best_params=best_params,
                pretraining=pretraining,
                num_classes=1
            )
            model = model.to(device)
            print(model)

            total_params = 0
            for name, param in model.named_parameters():
                total_params += param.numel()
            print(f"Total number of parameters: {total_params}")

            if args.pretrained_feature_extractor_path:
                # Use the dedicated feature extractor loading method for mamba models
                if hasattr(model, 'load_pretrained_feature_extractor'):
                    print("Loading pretrained feature extractor for Mamba model...")
                    model.load_pretrained_feature_extractor(
                        args.pretrained_feature_extractor_path, 
                        freeze=True
                    )
                    print(f"Feature extractor loaded and frozen")
                    
                    # Count trainable parameters after freezing
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
                    print(f"Trainable parameters: {trainable_params:,}")
                    print(f"Frozen parameters: {frozen_params:,}")
                    print(f"Percentage trainable: {100 * trainable_params / (trainable_params + frozen_params):.2f}%")
                else:
                    print("Warning: Model does not support load_pretrained_feature_extractor method")
            

            # Setup optimizer
            lr = best_params['lr']
            
            # Adjust learning rate if feature extractor is frozen
            if args.pretrained_feature_extractor_path and hasattr(model, 'load_pretrained_feature_extractor'):
                # Get fraction of trainable parameters
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in model.parameters())
                trainable_fraction = trainable_params / total_params
                
                # If less than 50% parameters are trainable, scale up learning rate
                if trainable_fraction < 0.5:
                    lr_scale = 1.0 / max(0.1, trainable_fraction)  # Scale inversely with trainable fraction
                    lr = lr * min(lr_scale, 5.0)  # Cap the scaling at 5x
                    print(f"Scaled learning rate from {best_params['lr']:.6f} to {lr:.6f} (scale: {lr_scale:.2f}x)")
                else:
                    print(f"Using original learning rate: {lr:.6f}")
            optimizer_name = best_params['optim']
            optimizer = {
                'AdamW': optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4, betas=(0.9, 0.95)),
                'SGD': optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9),
                'RMSprop': optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=lr),
                'RAdam': optim.RAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr),
            }[optimizer_name]

            
            # Setup scheduler
            nb_steps = get_nb_steps(df=df_train, batch_size=batch_size, stratify=stratify, shuffle=True)
            cycle = 1
            print(f"Number of steps: {nb_steps}")
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cycle*nb_steps, T_mult=1)
            # scheduler = optim.lr_scheduler.OneCycleLR(
            #     optimizer,
            #     max_lr=lr,
            #     total_steps=nb_steps * args.epochs,
            #     pct_start=0.1,
            #     div_factor=10.0,
            #     final_div_factor=1000.0,
            # )

            # Training metrics tracking
            train_accuracies = []
            train_losses = []
            train_F1s = []
            val_accuracies = []
            val_losses = []
            val_F1s = []
            
            # Early stopping
            early_stopping = EarlyStopping(patience=10, verbose=True)
            torch.cuda.empty_cache()
            
            # Training loop
            for epoch in range(args.epochs):
                # Train
                model.train()
                model, train_metrics, _ = run_model(
                    model, df_train, batch_size, True, device, optimizer, scheduler,
                    stratify=stratify,
                    wl1=wl1, wl2=wl2, wl3=wl3,
                    verbose=False,
                    pos_weight_l1=pos_weight_l1,
                    pos_weight_l2=pos_weight_l2,
                    pretraining=pretraining,
                    max_seq_len=max_seq_len,
                    use_TSO=True,
                    random_padding=best_params.get("random_padding", False),
                    average_mask=best_params.get("average_mask", False),
                    average_window_size=best_params.get("average_window_size", 20),
                    padding_value=best_params.get("padding_value", 0.0),
                    ignore_padding_in_mask_loss=True,
                    mixup_alpha=best_params.get("mixup_alpha", 0.2),
                    mixup_prob=best_params.get("mixup_prob", 0.7),
                )
                
                # Record metrics
                # Replace first epoch loss with 1.0 to avoid skewing visualization
                if epoch == 0:
                    train_losses.append(1.0)
                else:
                    train_losses.append(train_metrics['loss'])
                train_F1s.append(train_metrics['F1_1'])
                train_accuracies.append(train_metrics['Accuracy1'])
                
                # Validate
                with torch.no_grad():
                    model.eval()
                    _, val_metrics, val_predictions = run_model(
                        model, df_val, batch_size, False, device, optimizer, scheduler,
                        stratify=False,
                        wl1=wl1, wl2=wl2, wl3=wl3,
                        verbose=False,
                        pos_weight_l1=pos_weight_l1,
                        pos_weight_l2=pos_weight_l2,
                        pretraining=pretraining,
                        max_seq_len=max_seq_len,
                        use_TSO=True,
                        random_padding=best_params.get("random_padding", False),
                        average_mask=best_params.get("average_mask", False),
                        average_window_size=best_params.get("average_window_size", 20),
                        padding_value=best_params.get("padding_value", 0.0),
                        ignore_padding_in_mask_loss=True
                    )
                    
                    # Record metrics
                    val_losses.append(val_metrics['loss'])
                    val_F1s.append(val_metrics['F1_1'])
                    val_accuracies.append(val_metrics['Accuracy1'])
                    
                    # Check early stopping
                    m = float(val_metrics['loss'])
                    if early_stopping(m, model):
                        print(f"Early stopping at epoch {epoch+1}")
                        early_stopping.restore(model)
                        break
                
                print(f"Epoch [{epoch+1}/{args.epochs}]: Time: {datetime.now()}")
                print(f"  Training: {train_metrics}")
                print(f"  Validation: {val_metrics}")
            
            # Save learning curves
            learning_metrics = {
                "train_losses": train_losses,
                "train_accuracies": train_accuracies,
                "train_F1s": train_F1s,
                "val_losses": val_losses,
                "val_accuracies": val_accuracies,
                "val_F1s": val_F1s
            }
            
            # Save training metrics
            pd.DataFrame(learning_metrics).to_csv(
                os.path.join(training_logs_folder, f"training{i}_losses_accuracies_test_subject_{split_id}.csv"),
                encoding="utf-8", index=False
            )
            
            # Plot learning curves
            plot_learning_curves(
                learning_metrics=learning_metrics,
                output_filepath=os.path.join(learning_plots_output_folder, 
                                           f"training{i}_losses_accuracies_test_subject_{split_id}.jpg")
            )
            
            # Check if this is the best model so far
            retrain.check_performance(min(val_losses), model)
        
        # Restore best model
        model = retrain.restore(model)
        
        # Test model
        with torch.no_grad():
            model.eval()
            _, test_metrics, test_predictions = run_model(
                model, df_test, batch_size, False, device, optimizer, scheduler,
                stratify=False,
                wl1=wl1, wl2=wl2, wl3=wl3,
                verbose=False,
                pos_weight_l1=pos_weight_l1,
                pos_weight_l2=pos_weight_l2,
                pretraining=pretraining,
                max_seq_len=max_seq_len,
                use_TSO=True,
                random_padding=best_params.get("random_padding", False),
                average_mask=best_params.get("average_mask", False),
                average_window_size=best_params.get("average_window_size", 20),
                padding_value=best_params.get("padding_value", 0.0),
                ignore_padding_in_mask_loss=True

            )
        
        print(f"Testing results for {split_id}: {test_metrics}")
        
        # Save model
        torch.save(
            model.state_dict(),
            os.path.join(train_models_folder, f"{args.model}_test_subject_{split_id}_weights.pth")
        )
        
        # Save predictions and metrics
        if pretraining:
            pd.DataFrame(pd.json_normalize(test_metrics)).to_csv(
                os.path.join(predictions_output_folder, f"final_metrics_sheet_test_subject_{split_id}.csv"),
                encoding="utf-8", index=False
            )
            test_predictions.to_csv(
                os.path.join(predictions_output_folder, f"Y_test_Y_pred_test_subject_{split_id}.csv"),
                encoding="utf-8", index=False
            )
        else:
            # segment_level_predictions = test_predictions.groupby('segment').max(1)
            segment_level_predictions=test_predictions[test_predictions.inTSO==True].groupby('segment').max(1)
            
            # Generate confusion matrix plot
            plot_confusion_metrics(
                y_pred=segment_level_predictions.pr1,
                y_test=segment_level_predictions.gt1,
                output_filepath=os.path.join(confusion_matrix_plots_folder, 
                                           f"confusion_matrix_test_subject_{split_id}.jpg")
            )
            
            # Calculate final metrics
            final_metrics = {'pr1': calculate_metrics_nn(segment_level_predictions.gt1, segment_level_predictions.pr1),
                'pr2': calculate_metrics_nn(test_predictions.gt2, test_predictions.pr2),
                'pr3': calculate_metrics_nn(segment_level_predictions.gt3, segment_level_predictions.pr3, False)
            }
            
            # Save final metrics
            pd.DataFrame(pd.json_normalize(final_metrics)).to_csv(
                os.path.join(predictions_output_folder, f"final_metrics_sheet_test_subject_{split_id}.csv"),
                encoding="utf-8", index=False
            )
            
            # Save predictions
            test_predictions.to_csv(
                os.path.join(predictions_output_folder, f"Y_test_Y_pred_test_subject_{split_id}.csv"),
                encoding="utf-8", index=False
            )
            
            print(f"Final metrics: {final_metrics}")

        custom_print("Job finished successfully...")
        custom_print(f"Finished Training Now: {datetime.now()}")
    # break  # todo: this experiment only runs 1 subject split test (US10008003)
    # mlflow.end_run()
    return 1

def objective(trial, df_train, df_val, df_test, param_tuning_output_trials, num_gpu, max_seq_len): 
    # TODO: add number of layers in feature extractor to the search space

    if num_gpu=="NA":
        num_gpu=trial.number%4
    device = torch.device(f"cuda:{num_gpu}" if torch.cuda.is_available() else "cpu")
    # batch_sizes = [16, 64]
    batch_sizes = [64]
    batch_size = trial.suggest_categorical ('batch_size',batch_sizes)
    # num_filters = trial.suggest_categorical ('num_filters',[3, 32, 64])
    lr = 3e-4
    # lr = trial.suggest_categorical('lr', [0.001, 0.0001])
    # lr = trial.suggest_categorical('lr', lr)
    # dropout_rate = trial.suggest_float('dropout', 0, 0.5, step=0.1)
    # drop_path_rate = trial.suggest_float('droppath', 0, 0.5,step=0.1)
    # kernel_size_feature =  trial.suggest_int("kernel_f", 3, 15, step=2)
    # kernel_size_mba = trial.suggest_int("kernel_MBA", 3, 15, step=2)
    # num_feature_layers = trial.suggest_int("num_feature_layers", 3, 9, step=1)
    # num_MBA_blocks = trial.suggest_int("blocks_MBA", 0, 8, step=1)
    # MBA_encoder = trial.suggest_categorical("MBA_encoder", [True])#,False finetuning transformer fails due to memory.
    wl1  = trial.suggest_float("wl1", 0, 1,step=0.1)
    wl2  = trial.suggest_float("wl2", 0, 1,step=0.1)
    wl3  = trial.suggest_float("wl3", 0, 1,step=0.1)
    optimizer= trial.suggest_categorical ('optim',['AdamW','RMSprop','RAdam'])#, 'SGD', 'RMSprop'
    pos_weight_l1 = trial.suggest_categorical ('pos_weight_l1',[True,False])
    pos_weight_l2 = trial.suggest_categorical ('pos_weight_l2',[True,False])
    # cls_token = trial.suggest_categorical ('cls_token',[True,False])
    # featurelayer=trial.suggest_categorical ('featurelayer',["ResTCN"])
    # num_heads = trial.suggest_categorical ('num_heads',[2,4])
    #stratify=trial.suggest_categorical ('stratify',['oversample','undersample',False])
    # stratify=trial.suggest_categorical ('stratify',['undersample'])
    model_name = "efficientnet_v2_m"
    d_model = trial.suggest_categorical('d_model',[512, 1024])
    target_shape = trial.suggest_categorical("target_shape",[64, 128, 256])
    featurelayer = "Conv2d"
    reg_head=True
    tcn_params = {
                    "tcn_num_filters":  trial.suggest_categorical('tcn_num_filters', [3, 32, 64]), 
                    "tcn_kernel_size": 7, 
                    "tcn_num_blocks": trial.suggest_int('tcn_num_blocks', 4, 8,  step=1),
                    }
    nb_steps = get_nb_steps(df=df_train,batch_size=batch_size,stratify='undersample',shuffle=True)
    print(f"Trial number: {trial.number} running on {device}. Time: {datetime.now()}. {trial.params}")
    epochs = 200
    input_size = 3
    train_accuracies = []
    train_losses = [] 
    train_F1s = []
    val_accuracies = []
    val_losses = [] 
    val_F1s = []
    retrain = Retraining (verbose=False)

            
    try:
        for i in range (1):
            #print(f"Training number {i}")
            # model = MBA_tsm(input_size,num_feature_layers=num_feature_layers,num_encoder_layers=num_MBA_blocks,drop_path_rate =drop_path_rate ,kernel_size_feature=kernel_size_feature,kernel_size_mba=kernel_size_mba,dropout_rate=dropout_rate, max_seq_len= max_seq_len,cls_token=cls_token,mba=MBA_encoder,num_heads=num_heads,featurelayer=featurelayer,num_filters=num_filters)
            model =  EfficientUNet(prediction_length=max_seq_len, 
                                        model_name=model_name, 
                                        d_model=d_model, 
                                        tsm_horizon=64, 
                                        input_dim=3, 
                                        pos_embed_dim=16, 
                                        num_filters=64,
                                        kernel_size_feature=13,
                                        num_feature_layers=num_feature_layers,
                                        tsm=False,
                                        featurelayer=featurelayer,
                                        target_shape=target_shape,
                                        tcn_params=tcn_params,
                                        reg_head=reg_head)
            model = model.to(device)

            optimizer = {'AdamW': optim.AdamW(model.parameters(), lr=lr), #,weight_decay=wd
                # 'SGD': optim.SGD(model.parameters(), lr=lr, momentum=0.9),
                'RMSprop': optim.RMSprop(model.parameters(), lr=lr),
                'RAdam': optim.RAdam(model.parameters(), lr=lr),
            }[optimizer]

    #         scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min', factor=0.8, patience=2, verbose=True)
            cycle=6
            scheduler =  optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cycle*nb_steps, T_mult=1)
            early_stopping = EarlyStopping(patience=cycle*3, min_delta=1e-10, verbose=False)
            for epoch in range(epochs):
                model.train()
                # model,train_metrics,_=run_model(model,df_train,batch_size,True,device,optimizer,scheduler,stratify=stratify,wl1=wl1,wl2=wl2,wl3=wl3,verbose=False,pos_weight_l1=pos_weight_l1,pos_weight_l2=pos_weight_l2)
                model,train_metrics,_=run_model(model,df_train,batch_size,True,device,optimizer,scheduler,
                                                stratify=stratify,
                                                wl1=wl1,
                                                wl2=wl2,
                                                wl3=wl3,
                                                verbose=False,
                                                pos_weight_l1=pos_weight_l1,
                                                pos_weight_l2=pos_weight_l2,
                                                pretraining=pretraining, 
                                                max_seq_len=max_seq_len, 
                                                use_TSO=False)
                with torch.no_grad():
                    model.eval()
                    # _,val_metrics,val_predictions=run_model(model,df_val,batch_size,False,device,optimizer,scheduler,stratify=False,wl1=wl1,wl2=wl2,wl3=wl3,verbose=True,pos_weight_l1=pos_weight_l1,pos_weight_l2=pos_weight_l2)
                    _,val_metrics,val_predictions=run_model(model,df_val,batch_size,False,device,optimizer,scheduler,
                                                stratify=stratify,
                                                wl1=wl1,
                                                wl2=wl2,
                                                wl3=wl3,
                                                verbose=False,
                                                pos_weight_l1=pos_weight_l1,
                                                pos_weight_l2=pos_weight_l2,
                                                pretraining=pretraining, 
                                                max_seq_len=max_seq_len, 
                                                use_TSO=False)
                    val_losses.append(val_metrics['loss'])

                    if early_stopping(float(val_metrics['loss']), model):
    #                     print(f"Early stopping at epoch {epoch+1}")
                        early_stopping.restore(model)
                        break
    #             print(f'Trial number: {trial.number} Epoch [{epoch+1}/{epochs}]: Time: {datetime.now()} \n   training: {train_metrics} ,\n   validation:{val_metrics}')
            retrain.check_performance(min(val_losses),model)

        model=retrain.restore(model)   
        with torch.no_grad():
            model.eval()
            # _,test_metrics,test_predictions=run_model(model,df_test,batch_size,False,device,optimizer,scheduler,stratify=False,wl1=wl1,wl2=wl2,wl3=wl3,verbose=False,pos_weight_l1=pos_weight_l1,pos_weight_l2=pos_weight_l2)
            _,test_metrics,test_predictions=run_model(model,df_test,batch_size,False,device,optimizer,scheduler,
                                                stratify=stratify,
                                                wl1=wl1,
                                                wl2=wl2,
                                                wl3=wl3,
                                                verbose=False,
                                                pos_weight_l1=pos_weight_l1,
                                                pos_weight_l2=pos_weight_l2,
                                                pretraining=pretraining, 
                                                max_seq_len=max_seq_len, 
                                                use_TSO=False)
        print("-----------------------")
        print(f'Trial number: {trial.number} finished. Testing:{test_metrics}, Time: {datetime.now()}')


        trial_summary = {
            "training_metrics": test_metrics,
            "current_run_params": trial.params,
        }
        write_txt_to_file(
            os.path.join(param_tuning_output_trials, f"tuning_metrics_{trial.number}.txt"),
            str(trial_summary))

        return  float(test_metrics["F1_1"]),float(test_metrics["R2"]) #float(test_metrics["F1_2"]),
      
    except Exception as e: 
        print(e)
        return 0,0

    

def tune_pipeline(args):
    custom_print(f"Start tuning: {datetime.now()}")   
    model_name = args.model
    input_data_folder = str(args.input_data_folder)
    results_folder_name = args.output
    num_gpu = str(args.num_gpu)
    study_name=str(args.output)
    data_augmentation=args.data_augmentation
    dataset_type = os.path.basename(input_data_folder.rstrip("/"))
    # results_folder = f"/mnt/data/GENEActive-featurized/results/DL/{dataset_type}/{results_folder_name}/"
    results_folder = f"/domino/datasets/GENEActive-featurized/results/DL/{dataset_type}/{results_folder_name}/"
    param_tuning_output_folder = os.path.join(results_folder, "tuning/")
    param_tuning_output_trials = os.path.join(param_tuning_output_folder, "trials/")
    create_folder([param_tuning_output_folder,param_tuning_output_trials])

    # Write user provided arguments
    with open(os.path.join(param_tuning_output_folder, "user_arguments.txt"), "w") as file_handler:
        file_handler.write(str(args))
    
    if 'nsucl' in dataset_type:
        df= load_data_nsucl(input_data_folder)#load_data
    else:
        df= load_data(input_data_folder, max_seq_length=60, norm_scratch_length=False)#load_data
    
    df_train=df[df.FOLD!='FOLD1']
    df_test=df[df.FOLD=='FOLD1']
    
    max_seq_len= df.groupby('segment').count().max(1).max()+1
    
    random_indices= df_subset_segments(df_train, 0.2,False)
    df_val=df_train.iloc[df_train.index.isin(random_indices)]      
    df_train=df_train.iloc[~df_train.index.isin(random_indices)]
    scaler = StandardScaler()
    c_to_scale=['x', 'y', 'z']#, *df_train.loc[:, 'stft_0':'stft_32'].columns
    
    
    if data_augmentation > 0:
        print(f"Augmenting training data with {data_augmentation} iterations. {datetime.now()}")
        intitial_size=df_train.size
        df_train=augment_dataset(df_train,data_augmentation,verbose=False)
        print(f"Training data augmented from {intitial_size} to {df_train.size}. {datetime.now()}")
    
    df_train.loc[:, c_to_scale] = scaler.fit_transform(df_train[c_to_scale])
    df_val.loc[:, c_to_scale] = scaler.transform(df_val[c_to_scale])
    df_test.loc[:, c_to_scale] = scaler.transform(df_test[c_to_scale])
    
    study = optuna.create_study(directions=['maximize','maximize'],study_name=study_name,storage=f'sqlite:///{results_folder}{model_name}.db',load_if_exists=True)#'maximize',
    tune_objective=partial(objective, df_train=df_train, df_val=df_val,df_test=df_test, param_tuning_output_trials=param_tuning_output_trials,num_gpu=num_gpu, max_seq_len= max_seq_len)
    #with parallel_backend("multiprocessing", n_jobs=1):
    study.optimize(tune_objective,callbacks=[MaxTrialsCallback(300, states=(TrialState.COMPLETE,))])

    print('Best hyperparameters: ', study.best_trials[0]) # Selecting the first Pareto-optimal trial

    write_txt_to_file(file_path=os.path.join(param_tuning_output_folder, f"best_trial_params.txt"),
                      text=str(study.best_trials[0].params))
    
    custom_print("Job finished successfully...")
    custom_print(f"Finished Tuning Now: {datetime.now()}")


    return 1


# Examples: 

### NOPROD

## Training
# python 'munge/predictive_modeling/predict_scratch_segment.py' --input_data_folder '/mnt/data/Nocturnal-scratch/geneactive_20hz_2s_b1s_imerit/' --model mbatsm --output MBA_param11 --execution_mode train --num_gpu 0

# python 'munge/predictive_modeling/predict_scratch_segment.py' --input_data_folder '/mnt/data/Nocturnal-scratch/geneactive_20hz_2s_b1s_imerit/' --model mbatsm --output MBA_param11 --execution_mode train --testing LOSO 

## Training with data augmentation
# python 'munge/predictive_modeling/predict_scratch_segment.py' --input_data_folder '/mnt/data/Nocturnal-scratch/geneactive_20hz_2s_b1s_std/' --model mbatsm --output MBA_param4_pretraining --execution_mode train --data_augmentation 10 --pretraining True --testing production

## Pretraining
# python 'munge/predictive_modeling/predict_scratch_segment.py' --input_data_folder '/mnt/data/Nocturnal-scratch/geneactive_20hz_2s_b1s_imerit/' --model mbatsm --output MBA_pretraining_param11 --execution_mode train --pretraining True --testing production --num_gpu 1



# python 'munge/predictive_modeling/predict_scratch_segment.py' --input_data_folder '/mnt/data/Nocturnal-scratch/nsucl_leap_20hz_2s_b1s/' --model mbatsm --output MBA_DA10_pretraining --execution_mode train --data_augmentation 10 --pretraining True --testing production

#python 'munge/predictive_modeling/predict_scratch_segment.py' --input_data_folder '/mnt/data/Nocturnal-scratch/geneactive_20hz_2s_b1s_std/'  --model mbatsm --output MBA_3ch_production2 --execution_mode train --testing production --training_iterations 5 --num_gpu 0
# python 'munge/predictive_modeling/predict_scratch_segment.py' --input_data_folder '/mnt/data/Nocturnal-scratch/geneactive_20hz_2s_b1s_std/'  --model mbatsm --output base_model2 --execution_mode tune --num_gpu 2

#python 'munge/predictive_modeling/predict_scratch_segment.py' --input_data_folder '/mnt/data/Nocturnal-scratch/geneactive_20hz_2s_b1s_std/'  --model mbatsm --output MBA_DA-NSUCL --execution_mode train --data_augmentation 10 --num_gpu 0

# python 'munge/predictive_modeling/predict_scratch_segment.py' --input_data_folder '/mnt/data/Nocturnal-scratch/geneactive_20hz_2s_b1s_imerit/' --model mbatsm --output MBA_shufflexyz --execution_mode tune --num_gpu 3



# NSUCL

## Training
# python 'munge/predictive_modeling/predict_scratch_segment.py' --input_data_folder '/mnt/data/Nocturnal-scratch/nsucl_leap_20hz_2s_b1s/' --model mbatsm --output MBA_param11 --execution_mode train --data_augmentation 10 --num_gpu 2

## Pretraining
# python 'munge/predictive_modeling/predict_scratch_segment.py' --input_data_folder '/mnt/data/Nocturnal-scratch/nsucl_leap_20hz_2s_b1s/' --model mbatsm --output MBA_pretraining_param11 --execution_mode train --data_augmentation 10 --pretraining True --testing production --num_gpu 3


# python 'munge/predictive_modeling/predict_scratch_segment.py' --input_data_folder '/mnt/data/Nocturnal-scratch/nsucl_leap_20hz_2s_b1s/'  --model mbatsm --output MBA_3ch --execution_mode train --testing LOSO --num_gpu 1
#python 'munge/predictive_modeling/predict_scratch_segment.py' --input_data_folder '/mnt/data/Nocturnal-scratch/nsucl_leap_20hz_2s_b1s/'  --model mbatsm --output base_model --execution_mode tune --data_augmentation 5 --num_gpu 0
# python 'munge/predictive_modeling/predict_scratch_segment.py' --input_data_folder '/mnt/data/Nocturnal-scratch/nsucl_leap_20hz_2s_b1s/'  --model mbatsm --output feature_only --execution_mode train   --num_gpu 0

#python 'munge/predictive_modeling/predict_scratch_segment.py' --input_data_folder '/mnt/data/Nocturnal-scratch/nsucl_leap_20hz_2s_b1s/'  --model mbatsm --output MBA_DA --execution_mode train --data_augmentation 1 --num_gpu 0
#python 'munge/predictive_modeling/predict_scratch_segment.py' --input_data_folder '/mnt/data/Nocturnal-scratch/nsucl_leap_20hz_2s_b1s/'  --model mbatsm --output MBA_DA10_shufflexyz_production --execution_mode train --data_augmentation 10 --testing production --num_gpu 2



# #Debug: Manual Input
# clear_tracker_flag = True
# model_name = "ResCNN"
# execution_mode = "tune"
# input_data_folder = "mnt/data/GENEActive-featurized/inTSO/time_windows_results_3s_1s"
# input_features_type = "raw"
# results_folder_name = "test_model_rnn"
# list_features = 4
# multi_channel = False
# scaler_path = "/domino/datasets/GENEActive-featurized-consolidated/results/leave-1-out-cv/time_windows_results_2s_1s/standard_scaler_2s.joblib"
# pre_saved_scaler = joblib.load(scaler_path)



# ray.init(num_gpus=4, _system_config={
#     "max_direct_call_object_size": 2 * 1024 * 1024 * 1024,  # 2 GB
# })   # handle ray's RPC memory error


if args.execution_mode == "tune":
    output = tune_pipeline(args=args)
elif args.execution_mode == "train":
    output = train_pipeline(args=args)
else:
    raise Exception(f"Script execution mode should be train/tune. Received: {execution_mode}")

print("Training Job Finished")

