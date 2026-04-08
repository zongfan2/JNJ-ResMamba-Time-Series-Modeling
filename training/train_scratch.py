# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:19:50 2024

@author: MBoukhec
"""
import subprocess
import sys
import os

# Add project root to sys.path so that data/, models/, etc. are importable
# regardless of which directory the script is launched from.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import joblib

shell_script = '''
cd munge/predictive_modeling
sudo python3.11 -m pip install -r requirement-ml.txt
sudo python3.11 -m pip install -e .
sudo python3.11 -m pip install optuna==4.3.0 seaborn ray TensorboardX torcheval ruptures mamba-ssm[causal-conv1d]==2.2.2
'''
result = subprocess.run(shell_script, shell=True, capture_output=True, text=True)

import os
import joblib
import argparse
# ---------------
# Parse Arguments
# ---------------
parser = argparse.ArgumentParser(description='Run Predictive Modeling Pipeline.')
parser.add_argument('--config', type=str, required=False, default="", help='Path to YAML config file. When provided, config values override defaults but CLI args still take precedence.')
parser.add_argument('--input_data_folder', type=str, default=None, help='Path to the input data file')
parser.add_argument('--clear_tracker', default=None, required=False, help='Clear Job Tracking Folder')
parser.add_argument('--model', type=str, default=None, help='Define the model to be used')
parser.add_argument('--output', type=str, default=None, help='Define the name of the output folder')
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
parser.add_argument("--pretrained_model_path",type=str,default="", required=False, help="Path to pretrained MBATSMForPretraining model weights for finetuning.")
parser.add_argument("--freeze_encoder",type=str,default="False", required=False, help="Whether to freeze encoder during finetuning.")
parser.add_argument('--scaler_path', type=str, default="", required=False, help='Path to the saved scaler. If not provided, a new scaler will be fitted.')


args = parser.parse_args()

# ---------------
# YAML Config Loading (--config flag)
# ---------------
# Priority: CLI args > YAML config > argparse defaults
if args.config:
    import yaml
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Flatten nested YAML into a mapping to argparse args
    yaml_to_args = {
        # data section
        'input_data_folder': cfg.get('data', {}).get('input_folder'),
        'features_type':     cfg.get('data', {}).get('format'),
        'scaler_path':       cfg.get('data', {}).get('scaler_path'),
        # model section
        'model':             cfg.get('model', {}).get('architecture'),
        # training section
        'output':            cfg.get('training', {}).get('output'),
        'execution_mode':    cfg.get('training', {}).get('execution_mode'),
        'epochs':            cfg.get('training', {}).get('epochs'),
        'num_gpu':           cfg.get('training', {}).get('num_gpu'),
        'clear_tracker':     cfg.get('training', {}).get('clear_tracker'),
        'data_augmentation': cfg.get('training', {}).get('data_augmentation'),
        # transfer_learning section
        'pretrained_model_path': cfg.get('transfer_learning', {}).get('pretrained_model_path'),
        'freeze_encoder':        cfg.get('transfer_learning', {}).get('freeze_encoder'),
    }

    # Apply YAML values where CLI didn't explicitly override
    for key, yaml_val in yaml_to_args.items():
        if yaml_val is not None:
            current = getattr(args, key, None)
            # Only override if the arg is still at its default (None or argparse default)
            if current is None or (key == 'num_gpu' and current == 'NA'):
                setattr(args, key, yaml_val)

    print(f"Loaded config from: {args.config}")

# Validate required args
if not args.input_data_folder:
    parser.error("--input_data_folder is required (or set data.input_folder in YAML config)")
if not args.model:
    parser.error("--model is required (or set model.architecture in YAML config)")
if not args.output:
    parser.error("--output is required (or set training.output in YAML config)")


scaler_path = args.scaler_path if args.scaler_path else "/mnt/code/munge/predictive_modeling/std_scaler_3s.bin"
pre_saved_scaler = joblib.load(scaler_path)

# os.environ["CUDA_VISIBLE_DEVICES"]=num_gpu


import ast

import random
import shutil
from pprint import pprint
from datetime import datetime
import json
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
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from joblib import parallel_backend
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
from data import *
from losses import *
from evaluation import *
from utils import *
from models import *


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
                     '11_1': {'batch_size': 16, 'num_filters': 64, 'lr': 0.001, 'dropout': 0.2, 'droppath': 0.4, 'kernel_f': 13, 'kernel_MBA': 11, 'num_feature_layers': 7, 'blocks_MBA': 7,'encoder_bottelneck':16, 'MBA_encoder': True, 'wl1': 0.5, 'wl2': 0.5, 'wl3': 0.9, 'optim': 'RMSprop', 'pos_weight_l1': False, 'pos_weight_l2': True, 'cls_token': True, 'featurelayer': 'ResTCN', 'num_heads': 2, 'stratify': 'undersample'},
                     '11_2': {'batch_size': 16, 'num_filters': 64, 'lr': 0.001, 'dropout': 0.2, 'droppath': 0.4, 'kernel_f': 13, 'kernel_MBA': 11, 'num_feature_layers': 7, 'blocks_MBA': 7,'encoder_bottelneck':16, 'MBA_encoder': True, 'wl1': 0.5, 'wl2': 0.5, 'wl3': 0.9, 'optim': 'RMSprop', 'pos_weight_l1': False, 'pos_weight_l2': True, 'cls_token': True, 'featurelayer': 'TCN', 'num_heads': 2, 'stratify': 'undersample'},
                     '12': {'batch_size': 16, 'num_filters': 64, 'lr': 0.001, 'dropout': 0.2, 'droppath': 0.2,'channel_masking_rate': 0.1, 'kernel_f': 13,'kernel_MBA': 11, 'kernel_MBA1': 11, 'kernel_MBA2': 11, 'num_feature_layers': 7, 'blocks_MBA': 7,'blocks_MBA1': 7, 'blocks_MBA2': 0,'dilation1': True, 'dilation2': True,'encoder_bottelneck':1, 'MBA_encoder': True, 'wl1': 0.5, 'wl2': 0.5, 'wl3': 0.9, 'optim': 'RMSprop', 'pos_weight_l1': False, 'pos_weight_l2': True, 'cls_token': True, 'featurelayer': 'ResTCN', 'num_heads': 4, 'stratify': 'undersample','norm':'GN'},
                     '12_BN': {'batch_size': 16, 'num_filters': 64, 'lr': 0.001, 'dropout': 0.2, 'droppath': 0.2,'channel_masking_rate': 0.1, 'kernel_f': 13,'kernel_MBA': 11, 'kernel_MBA1': 11, 'kernel_MBA2': 11, 'num_feature_layers': 7, 'blocks_MBA': 7,'blocks_MBA1': 7, 'blocks_MBA2': 0,'dilation1': True, 'dilation2': True,'encoder_bottelneck':1, 'MBA_encoder': True, 'wl1': 0.5, 'wl2': 0.5, 'wl3': 0.9, 'optim': 'RMSprop', 'pos_weight_l1': False, 'pos_weight_l2': True, 'cls_token': True, 'featurelayer': 'ResTCN', 'num_heads': 4, 'stratify': 'undersample','norm':'BN'},
                     '12_0channelmasking': {'batch_size': 16, 'num_filters': 64, 'lr': 0.001, 'dropout': 0.2, 'droppath': 0.2,'channel_masking_rate': 0, 'kernel_f': 13,'kernel_MBA': 11, 'kernel_MBA1': 11, 'kernel_MBA2': 11, 'num_feature_layers': 7, 'blocks_MBA': 7,'blocks_MBA1': 7, 'blocks_MBA2': 0,'dilation1': True, 'dilation2': True,'encoder_bottelneck':1, 'MBA_encoder': True, 'wl1': 0.5, 'wl2': 0.5, 'wl3': 0.9, 'optim': 'RMSprop', 'pos_weight_l1': False, 'pos_weight_l2': True, 'cls_token': True, 'featurelayer': 'ResTCN', 'num_heads': 4, 'stratify': 'undersample','norm':'GN'},
                     '12_weight': {'batch_size': 16, 'num_filters': 64, 'lr': 0.001, 'dropout': 0.2, 'droppath': 0.2,'channel_masking_rate': 0, 'kernel_f': 13,'kernel_MBA': 11, 'kernel_MBA1': 11, 'kernel_MBA2': 11, 'num_feature_layers': 7, 'blocks_MBA': 7,'blocks_MBA1': 7, 'blocks_MBA2': 0,'dilation1': True, 'dilation2': True,'encoder_bottelneck':1, 'MBA_encoder': True, 'wl1': 3, 'wl2': 5, 'wl3': 8, 'optim': 'RMSprop', 'pos_weight_l1': False, 'pos_weight_l2': True, 'cls_token': True, 'featurelayer': 'ResTCN', 'num_heads': 4, 'stratify': 'undersample','norm':'GN'},
                     '12_1FL': {'batch_size': 16, 'num_filters': 64, 'lr': 0.001, 'dropout': 0.2, 'droppath': 0.2,'channel_masking_rate': 0, 'kernel_f': 13,'kernel_MBA': 11, 'kernel_MBA1': 11, 'kernel_MBA2': 11, 'num_feature_layers': 1, 'blocks_MBA': 7,'blocks_MBA1': 7, 'blocks_MBA2': 0,'dilation1': True, 'dilation2': True,'encoder_bottelneck':1, 'MBA_encoder': True, 'wl1': 0.5, 'wl2': 0.5, 'wl3': 0.9, 'optim': 'RMSprop', 'pos_weight_l1': False, 'pos_weight_l2': True, 'cls_token': True, 'featurelayer': 'ResTCN', 'num_heads': 4, 'stratify': 'undersample','norm':'GN'},
                     '13_0FL_1ENC': {'batch_size': 16, 'num_filters': 64, 'lr': 0.001, 'dropout': 0.2, 'droppath': 0.2,'channel_masking_rate': 0, 'kernel_f': 13,'kernel_MBA': 11, 'kernel_MBA1': 11, 'kernel_MBA2': 11, 'num_feature_layers': 0, 'blocks_MBA': 7,'blocks_MBA1': 7, 'blocks_MBA2': 0,'dilation1': True, 'dilation2': True,'encoder_bottelneck':1, 'MBA_encoder': True, 'wl1': 0.5, 'wl2': 0.5, 'wl3': 0.9, 'optim': 'RMSprop', 'pos_weight_l1': False, 'pos_weight_l2': True, 'cls_token': True, 'featurelayer': 'ResTCN', 'num_heads': 4, 'stratify': 'undersample','norm':'GN'},
                     '14': {'batch_size': 16, 'num_filters': 64, 'lr': 0.001, 'dropout': 0.3, 'droppath': 0.5,'channel_masking_rate': 0.1, 'kernel_f': 13,'kernel_MBA': 11, 'kernel_MBA1': 3, 'kernel_MBA2': 11, 'num_feature_layers': 0, 'blocks_MBA': 7,'blocks_MBA1': 1, 'blocks_MBA2': 4,'dilation1': False, 'dilation2': False,'encoder_bottelneck':1, 'MBA_encoder': True, 'wl1': 3, 'wl2': 5, 'wl3': 8, 'optim': 'RMSprop', 'pos_weight_l1': False, 'pos_weight_l2': True, 'cls_token': True, 'featurelayer': 'ResTCN', 'num_heads': 4, 'stratify': 'undersample','norm':'IN'},
                     '13': {'batch_size': 16, 'num_filters': 64, 'lr': 0.001, 'dropout': 0.2, 'droppath': 0.2,'channel_masking_rate': 0, 'kernel_f': 13,'kernel_MBA': 11, 'kernel_MBA1': 11, 'kernel_MBA2': 11, 'num_feature_layers': 7, 'blocks_MBA': 7,'blocks_MBA1': 7, 'blocks_MBA2': 0,'dilation1': True, 'dilation2': True,'encoder_bottelneck':1, 'MBA_encoder': True, 'wl1': 3, 'wl2': 5, 'wl3': 8, 'optim': 'RMSprop', 'pos_weight_l1': False, 'pos_weight_l2': True, 'cls_token': True, 'featurelayer': 'ResTCN', 'num_heads': 4, 'stratify': 'undersample','norm':'GN'},
                     '13_0FL_2ENC': {'batch_size': 16, 'num_filters': 64, 'lr': 0.001, 'dropout': 0.2, 'droppath': 0.2,'channel_masking_rate': 0, 'kernel_f': 13,'kernel_MBA': 11, 'kernel_MBA1': 11, 'kernel_MBA2': 11, 'num_feature_layers': 0, 'blocks_MBA': 7,'blocks_MBA1': 7, 'blocks_MBA2': 7,'dilation1': True, 'dilation2': True,'encoder_bottelneck':1, 'MBA_encoder': True, 'wl1': 3, 'wl2': 5, 'wl3': 8, 'optim': 'RMSprop', 'pos_weight_l1': False, 'pos_weight_l2': True, 'cls_token': True, 'featurelayer': 'ResTCN', 'num_heads': 4, 'stratify': 'undersample','norm':'GN'},
                     '13_1': {'batch_size': 16, 'num_filters': 64, 'lr': 0.001, 'dropout': 0.2, 'droppath': 0.2,'channel_masking_rate': 0, 'kernel_f': 13,'kernel_MBA': 11, 'kernel_MBA1': 11, 'kernel_MBA2': 11, 'num_feature_layers': 7, 'blocks_MBA': 7,'blocks_MBA1': 7, 'blocks_MBA2': 0,'dilation1': True, 'dilation2': True,'encoder_bottelneck':1, 'MBA_encoder': True, 'wl1': 0.5, 'wl2': 0.5, 'wl3': 0.9, 'optim': 'RMSprop', 'pos_weight_l1': False, 'pos_weight_l2': True, 'cls_token': True, 'featurelayer': 'TCN', 'num_heads': 4, 'stratify': 'undersample','norm':'GN'},
                     '13_2': {'batch_size': 16, 'num_filters': 64, 'lr': 0.001, 'dropout': 0.2, 'droppath': 0.2,'channel_masking_rate': 0, 'kernel_f': 13,'kernel_MBA': 11, 'kernel_MBA1': 11, 'kernel_MBA2': 11, 'num_feature_layers': 7, 'blocks_MBA': 7,'blocks_MBA1': 7, 'blocks_MBA2': 0,'dilation1': True, 'dilation2': True,'encoder_bottelneck':1, 'MBA_encoder': True, 'wl1': 0.5, 'wl2': 0.5, 'wl3': 0.9, 'optim': 'RMSprop', 'pos_weight_l1': False, 'pos_weight_l2': True, 'cls_token': True, 'featurelayer': 'ResTCN', 'num_heads': 4, 'stratify': 'progressive_undersample','norm':'IN'},
                     '13_2_ResNet': {'batch_size': 16, 'num_filters': 64, 'lr': 0.001, 'dropout': 0.2, 'droppath': 0.2,'channel_masking_rate': 0, 'kernel_f': 13,'kernel_MBA': 11, 'kernel_MBA1': 11, 'kernel_MBA2': 11, 'num_feature_layers': 7, 'blocks_MBA': 7,'blocks_MBA1': 7, 'blocks_MBA2': 0,'dilation1': True, 'dilation2': True,'encoder_bottelneck':1, 'MBA_encoder': True, 'wl1': 0.5, 'wl2': 0.5, 'wl3': 0.9, 'optim': 'RMSprop', 'pos_weight_l1': False, 'pos_weight_l2': True, 'cls_token': True, 'featurelayer': 'ResNet', 'num_heads': 4, 'stratify': 'undersample','norm':'IN'},
                     '13_2_ResNet_noMamba': {'batch_size': 16, 'num_filters': 64, 'lr': 0.001, 'dropout': 0.2, 'droppath': 0.2,'channel_masking_rate': 0, 'kernel_f': 13,'kernel_MBA': 11, 'kernel_MBA1': 11, 'kernel_MBA2': 11, 'num_feature_layers': 7, 'blocks_MBA': 7,'blocks_MBA1': 0, 'blocks_MBA2': 0,'dilation1': True, 'dilation2': True,'encoder_bottelneck':1, 'MBA_encoder': True, 'wl1': 0.5, 'wl2': 0.5, 'wl3': 0.9, 'optim': 'RMSprop', 'pos_weight_l1': False, 'pos_weight_l2': True, 'cls_token': True, 'featurelayer': 'ResNet', 'num_heads': 4, 'stratify': 'undersample','norm':'IN'},
                     '13_2_ResNet_noMamba_64': {'batch_size': 64, 'num_filters': 64, 'lr': 0.001, 'dropout': 0.2, 'droppath': 0.2,'channel_masking_rate': 0, 'kernel_f': 13,'kernel_MBA': 11, 'kernel_MBA1': 11, 'kernel_MBA2': 11, 'num_feature_layers': 7, 'blocks_MBA': 7,'blocks_MBA1': 0, 'blocks_MBA2': 0,'dilation1': True, 'dilation2': True,'encoder_bottelneck':1, 'MBA_encoder': True, 'wl1': 0.5, 'wl2': 0.5, 'wl3': 0.9, 'optim': 'RMSprop', 'pos_weight_l1': False, 'pos_weight_l2': True, 'cls_token': True, 'featurelayer': 'ResNet', 'num_heads': 4, 'stratify': 'undersample','norm':'IN'},
                     '13_2_ResTCN': {'batch_size': 16, 'num_filters': 64, 'lr': 0.001, 'dropout': 0.2, 'droppath': 0.2,'channel_masking_rate': 0, 'kernel_f': 13,'kernel_MBA': 11, 'kernel_MBA1': 11, 'kernel_MBA2': 11, 'num_feature_layers': 7, 'blocks_MBA': 7,'blocks_MBA1': 7, 'blocks_MBA2': 0,'dilation1': True, 'dilation2': True,'encoder_bottelneck':1, 'MBA_encoder': True, 'wl1': 0.5, 'wl2': 0.5, 'wl3': 0.9, 'optim': 'RMSprop', 'pos_weight_l1': False, 'pos_weight_l2': True, 'cls_token': True, 'featurelayer': 'ResTCN', 'num_heads': 4, 'stratify': 'undersample','norm':'IN'},
                     '13_2_128': {'batch_size': 16, 'num_filters': 128, 'lr': 0.001, 'dropout': 0.2, 'droppath': 0.2,'channel_masking_rate': 0, 'kernel_f': 13,'kernel_MBA': 11, 'kernel_MBA1': 11, 'kernel_MBA2': 11, 'num_feature_layers': 7, 'blocks_MBA': 7,'blocks_MBA1': 7, 'blocks_MBA2': 0,'dilation1': True, 'dilation2': True,'encoder_bottelneck':1, 'MBA_encoder': True, 'wl1': 0.5, 'wl2': 0.5, 'wl3': 0.9, 'optim': 'RMSprop', 'pos_weight_l1': False, 'pos_weight_l2': True, 'cls_token': True, 'featurelayer': 'TCN', 'num_heads': 4, 'stratify': 'undersample','norm':'IN'},
                     '13_2_1out': {'batch_size': 16, 'num_filters': 64, 'lr': 0.001, 'dropout': 0.2, 'droppath': 0.2,'channel_masking_rate': 0, 'kernel_f': 13,'kernel_MBA': 11, 'kernel_MBA1': 11, 'kernel_MBA2': 11, 'num_feature_layers': 7, 'blocks_MBA': 7,'blocks_MBA1': 7, 'blocks_MBA2': 0,'dilation1': True, 'dilation2': True,'encoder_bottelneck':1, 'MBA_encoder': True, 'wl1': 0.5, 'wl2': 0.5, 'wl3': 0.9, 'optim': 'RMSprop', 'pos_weight_l1': False, 'pos_weight_l2': True, 'cls_token': True, 'featurelayer': 'TCN', 'num_heads': 4, 'stratify': 'undersample','norm':'IN'},
                     '13_2_batch64': {'batch_size': 64, 'num_filters': 64, 'lr': 0.001, 'dropout': 0.2, 'droppath': 0.2,'channel_masking_rate': 0, 'kernel_f': 13,'kernel_MBA': 11, 'kernel_MBA1': 11, 'kernel_MBA2': 11, 'num_feature_layers': 7, 'blocks_MBA': 7,'blocks_MBA1': 7, 'blocks_MBA2': 0,'dilation1': True, 'dilation2': True,'encoder_bottelneck':1, 'MBA_encoder': True, 'wl1': 0.5, 'wl2': 0.5, 'wl3': 0.9, 'optim': 'RMSprop', 'pos_weight_l1': False, 'pos_weight_l2': True, 'cls_token': True, 'featurelayer': 'TCN', 'num_heads': 4, 'stratify': 'undersample','norm':'IN'},
                     'TCNx2': {'batch_size': 16, 'num_filters': 64, 'lr': 0.001, 'dropout': 0.2, 'droppath': 0.2,'channel_masking_rate': 0, 'kernel_f': 13,'kernel_MBA': 11, 'kernel_MBA1': 11, 'kernel_MBA2': 11, 'num_feature_layers': 7, 'blocks_MBA': 7,'blocks_MBA1': 0, 'blocks_MBA2': 0,'dilation1': True, 'dilation2': True,'encoder_bottelneck':1, 'MBA_encoder': True, 'wl1': 0.5, 'wl2': 0.5, 'wl3': 0.9, 'optim': 'RMSprop', 'pos_weight_l1': False, 'pos_weight_l2': True, 'cls_token': True, 'featurelayer': 'TCN', 'num_heads': 4, 'stratify': 'undersample','norm':'IN'},
                     '14': {'batch_size': 32, 'num_filters': 64, 'lr': 1e-4, 'dropout': 0.2, 'droppath': 0.2,'channel_masking_rate': 0, 'kernel_f': 13,'kernel_MBA': 11, 'kernel_MBA1': 11, 'kernel_MBA2': 11, 'num_feature_layers': 0, 'blocks_MBA': 7,'blocks_MBA1': 7, 'blocks_MBA2': 0,'dilation1': True, 'dilation2': True,'encoder_bottelneck':1, 'MBA_encoder': True, 'wl1': 0.5, 'wl2': 0.5, 'wl3': 0.9, 'optim': 'RMSprop', 'pos_weight_l1': False, 'pos_weight_l2': True, 'cls_token': True, 'featurelayer': 'TCN', 'num_heads': 4, 'stratify': 'undersample','norm':'GN', 'use_cls_token': True, 'add_positional_encoding': True},
                     'param_16': {'batch_size': 32, 'num_filters': 128, 'dropout': 0.5, 'droppath': 0.3, 'kernel_f': 13, 'kernel_MBA': 9, 'num_feature_layers': 9, 'blocks_MBA': 8, 'wl1': 0.5, 'wl2': 0.9, 'wl3': 0.5, 'optim': 'RMSprop', 'pos_weight_l1': False, 'pos_weight_l2': True, 'featurelayer': 'ResNet', 'norm1': 'BN', 'norm2': 'GN', 'norm3': 'BN','lr': 0.001,'channel_masking_rate': 0,'cls_token': True, 'pooling_type': 'avg_only', 'num_heads': 4, 'stratify': 'undersample', 'skip_connect': True, 'skip_cross_attention': True, },
                     'param_mba_v1': {'batch_size': 32, 'num_filters': 128, 'dropout': 0.5, 'droppath': 0.3, 'kernel_f': 13, 'kernel_MBA1': 9, 'num_feature_layers': 9, 'blocks_MBA1': 8, 'wl1': 0.5, 'wl2': 0.9, 'wl3': 0.5, 'optim': 'RMSprop', 'pos_weight_l1': False, 'pos_weight_l2': True, 'featurelayer': 'ResNet', 'norm1': 'BN', 'norm2': 'GN', 'norm3': 'BN', 'lr': 0.001, 'channel_masking_rate': 0, 'cls_token': True, 'stratify': 'undersample'},
                    }

        resmbaunet_params_list = {
            '1': {'batch_size': 16, 'num_filters': 64, 'lr': 0.001, 'dropout': 0.2, 'droppath': 0.2,'channel_masking_rate': 0, 'kernel_f': 13,'kernel_MBA': 11, 'kernel_MBA1': 11, 'kernel_MBA2': 11, 'num_feature_layers': 7, 'blocks_MBA': 7,'blocks_MBA1': 7, 'blocks_MBA2': 0,'dilation1': True, 'dilation2': True,'encoder_bottelneck':1, 'MBA_encoder': True, 'wl1': 0.5, 'wl2': 0.5, 'wl3': 0.9, 'optim': 'RMSprop', 'pos_weight_l1': False, 'pos_weight_l2': True, 'cls_token': True, 'featurelayer': 'TCN', 'num_heads': 4, 'stratify': 'undersample','norm':'IN'},
        }
        # return resmbaunet_params_list['1']
        # return params_list['13_2']
        return params_list['param_mba_v1']


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


def run_model(model,df,batch_size,train_mode,device,optimizer,scheduler,stratify=False,get_predictions=True,wl1=1,wl2=2,wl3=0.1,verbose=True,pos_weight_l1=False,pos_weight_l2=False,pretraining=False,mixup_alpha=0.1,mixup_prob=0,max_seq_len=None):

    epoch_loss = batches  = 0.0
    gt1s = gt2s = gt3s = lens = inTSOs = predictTSOs=  []
    pr1s = pr1_probs =pr2s= pr2_probs = pr3s = attentions=[]
    loss1_s=[]
    loss2_s=[]
    loss3_s=[]
    seg_column='segment'
    segments1=segments2=positions=[]
#     for batch in batch_generator(df=df,batch_size=batch_size,stratify=train_mode,shuffle=True,seg_column=seg_column):#stratify=train_mode
    for batch in batch_generator(df=df,batch_size=batch_size,stratify=stratify,shuffle=train_mode,seg_column=seg_column):    
        if pretraining:
            batch_data,labels,x_lens=add_padding_pretrain(batch,device,seg_column) #TCN doesn't work with pytorch pack_pad.
            outputs1,outputs2,outputs3,att = model(batch_data,x_lens)
            label1=labels[:,:,0].view(-1)
            label2=labels[:,:,1].view(-1)
            label3=labels[:,:,2].view(-1)

#             outputs1 = remove_padding(outputs1.squeeze(1).cpu().detach().numpy(),x_lens)
#             outputs2 = remove_padding(outputs2.squeeze(1).cpu().detach().numpy(),x_lens)
#             outputs3 = remove_padding(outputs3.squeeze(1).cpu().detach().numpy(),x_lens)
#             label1=remove_padding(labels[:,:,0].cpu().detach().numpy(),x_lens)
#             label2=remove_padding(labels[:,:,1].cpu().detach().numpy(),x_lens)
#             label3=remove_padding(labels[:,:,2].cpu().detach().numpy(),x_lens)

            loss,loss1,loss2,loss3,label1,label2,label3 = measure_loss_pretrain(outputs1,outputs2,outputs3,label1,label2,label3)
            loss1_s.append(loss1.item())
            loss2_s.append(loss2.item())
            loss3_s.append(loss3.item())
            # Backward pass and optimization if in training mode
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            epoch_loss += loss.item()
            batches += 1
            pr1s=np.concatenate((pr1s,outputs1.cpu().detach().numpy()))
            pr2s=np.concatenate((pr2s,outputs2.cpu().detach().numpy()))
            pr3s=np.concatenate((pr3s,outputs3.cpu().detach().numpy()))
            
            gt1s=np.concatenate((gt1s,label1.cpu().detach().numpy()))
            gt2s=np.concatenate((gt2s,label2.cpu().detach().numpy()))
            gt3s=np.concatenate((gt3s,label3.cpu().detach().numpy()))
            positions=np.concatenate((positions,batch.position_segment.values))
            segments2=np.concatenate((segments2,batch[seg_column].values))
        else:
            batch_data,label1,label2,label3,x_lens=add_padding(batch,device,seg_column,max_seq_len=max_seq_len) #TCN doesn't work with pytorch pack_pad.
            apply_mixup = train_mode and torch.rand(1).item() < mixup_prob
            outputs1, outputs2, outputs3, att, contrastive_embedding, mixup_info = model(batch_data, x_lens, labels1=label1, labels3=label3, apply_mixup=apply_mixup, mixup_alpha=mixup_alpha)
            if mixup_info is not None:
                mixup_label1, mixup_label3, mixup_lambda = mixup_info
            else:
                mixup_label1 = None 
                mixup_label3 = None
                mixup_lambda = 1.0
            #outputs1,outputs2,outputs3,att = model(batch_data,x_lens)
            outputs1 = outputs1.view(-1)  
#             outputs2 = outputs2.reshape(-1)
            outputs3 = outputs3.view(-1)
#             att=att.reshape(-1)
    #         print(att.shape)
    #         print(outputs2 .shape)
#             label1=torch.tensor(batch.groupby(seg_column, sort=False)['segment_scratch'].any().reset_index()['segment_scratch'].values*1,device=device) #Make sure to prevent pandas from sorting during group_by otherwise it will missalign the labels.
#             label2 = label2.view(-1)
#             label3=torch.tensor(batch.groupby(seg_column, sort=False)['scratch_duration'].max(1).values,device=device)
    #         label3=torch.tensor(batch.groupby(seg_column, sort=False)['scratch_sum'].max(1).values,device=device)/1200

            #Measure loss
            loss,loss1,loss2,loss3 = measure_loss_multitask(outputs1,outputs2,outputs3,label1,label2,label3,wl1,wl2,wl3,mixup_label1=mixup_label1, mixup_lambda=mixup_lambda,pos_weight_l1=pos_weight_l1,pos_weight_l2=pos_weight_l2)
            loss1_s.append(loss1.item())
            loss2_s.append(loss2.item())
            loss3_s.append(loss3.item())
            # Backward pass and optimization if in training mode
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
            epoch_loss += loss.item()
            batches += 1
            #Measure performance using the unpadded data
            pr1_prob=torch.sigmoid(outputs1)
            pr2_prob=torch.sigmoid(outputs2)
            pr1 = torch.round(pr1_prob)
            pr2 = torch.round(pr2_prob)  # threshold of 0.5 for binary classification #TODO: check NAs
            pr3 = outputs3

            pr2=pr2.cpu().detach().numpy()
            #label2 = batch.scratch.values * 1
            if np.isnan(pr2).any():
                if verbose:
                    print("Predictions contain NA. Replacing by 0")
                pr1=torch.nan_to_num(pr1)
                pr2=np.nan_to_num(pr2)
                pr3=torch.nan_to_num(pr3)
            #remove padding from the step level predictions
#             pr2 = remove_padding(pr2,x_lens)
#             pr2_prob= remove_padding(pr2_prob.cpu().detach().numpy(),x_lens)
#             att=remove_padding(att.cpu().detach().numpy(),x_lens)

            pr1s=np.concatenate((pr1s,pr1.cpu().detach().numpy()))
            pr1_probs=np.concatenate((pr1_probs,pr1_prob.cpu().detach().numpy()))


            pr2s=np.concatenate((pr2s,pr2))
            pr2_probs=np.concatenate((pr2_probs,pr2_prob.cpu().detach().numpy()))
            if att is not None:
                attentions=np.concatenate((attentions,att.cpu().detach().numpy()))
    #         pr2s_2=np.concatenate((pr2s_2, seq_to_seconds(pr2,y_lens)))
    #         pr2s_3=np.concatenate((pr2s_3,seq_to_seconds(smooth_binary_series(pr2,20,20),y_lens)))

            pr3s=np.concatenate((pr3s,pr3.cpu().detach().numpy()))


            gt1s=np.concatenate((gt1s,label1.cpu().detach().numpy()))
            gt2s=np.concatenate((gt2s,label2.cpu().detach().numpy()))
    #         gt2s_2=np.concatenate((gt2s_2, seq_to_seconds(label2,x_lens)))
            gt3s=np.concatenate((gt3s,label3.cpu().detach().numpy()))

             # TODO: smooth each segment seperatly
            lens=np.concatenate((lens,x_lens))
            segments1=np.concatenate((segments1,batch[seg_column].unique()))
            segments2=np.concatenate((segments2,batch[seg_column].values))
            positions=np.concatenate((positions,batch.position_segment.values))
            inTSOs=np.concatenate((inTSOs,batch.inTSO.values)) 
            predictTSOs=np.concatenate((predictTSOs,batch.predictTSO.values)) 
    
    pr1s=np.nan_to_num(pr1s, nan=0, posinf=0, neginf=0)
    pr2s=np.nan_to_num(pr2s, nan=0, posinf=0, neginf=0)
    pr3s=np.nan_to_num(pr3s, nan=0, posinf=0, neginf=0)
    
    if pretraining:
        if get_predictions:
            predictions=pd.DataFrame()
            predictions=pd.DataFrame({#"segment":segments2,
                                       "position":positions,
                                       "gt1": gt1s,
                                      "gt2": gt2s,
                                       "gt3": gt3s,
                                       "pr1": pr1s,
                                      "pr2": pr2s,
                                      "pr3": pr3s})
        gt_args = (gt1s, gt2s, gt3s)    
        pr_args = (pr1s, pr2s, pr3s)    
        eval_metrics={'MSE': "%.2f" % ((metrics.mean_squared_error(np.concatenate(gt_args), np.concatenate(pr_args)))),
                      'R2' : "%.2f" % ((metrics.r2_score(np.concatenate(gt_args), np.concatenate(pr_args)))),
                      'loss': "%.4f" % (epoch_loss/ batches),
                      'loss1': "%.2f" % (np.mean(loss1_s)),
                      'loss2': "%.2f" % (np.mean(loss2_s)),
                      'loss3': "%.2f" % (np.mean(loss3_s)),
                      'steps':batches}
    else:
        if get_predictions:
            predictions1=pd.DataFrame({"segment":segments1,
                                       "gt1": gt1s,
                                       "gt3": gt3s,
                                       "pr1": pr1s,
                                       "pr1_probs": pr1_probs,
                                       "pr3": pr3s})

            pred2_dict = {"segment":segments2,
                                       "position":positions,
                                       "inTSO":inTSOs,
                                       "predictTSO":predictTSOs,
                                       "gt2": gt2s,
                                       "pr2": pr2s,
                                       "pr2_probs": pr2_probs}
            if len(attentions) == len(segments2):
                pred2_dict["attentions"] = attentions
            predictions2=pd.DataFrame(pred2_dict)

            predictions=predictions2.merge(predictions1)
        else:
            predictions=pd.dataFrame()

        predictions_inTSO=predictions[predictions.inTSO==True]
        predictions_segment_inTSO=predictions_inTSO.groupby('segment').max(1)
        eval_metrics={'F1_1': "%.2f" % (metrics.f1_score(predictions_segment_inTSO.gt1, predictions_segment_inTSO.pr1,zero_division=0)*100),
              'F1_2': "%.2f" % (metrics.f1_score(predictions_inTSO.gt2, predictions_inTSO.pr2,zero_division=0)*100),
#                   'F1_2_2': "%.2f" % (metrics.f1_score(gt2s_2, pr2s_2,zero_division=0)*100),
#                   'F1_2_3': "%.2f" % (metrics.f1_score(gt2s_2, pr2s_3,zero_division=0)*100),
              'Accuracy1': "%.2f" % (metrics.accuracy_score(predictions_segment_inTSO.gt1, predictions_segment_inTSO.pr1)*100),
              'Accuracy2': "%.2f" % (metrics.accuracy_score(predictions_inTSO.gt2, predictions_inTSO.pr2)*100),
              'MSE': "%.2f" % (metrics.mean_squared_error(predictions_segment_inTSO.gt3, predictions_segment_inTSO.pr3)),
              'R2' : "%.2f" % (metrics.r2_score(predictions_segment_inTSO.gt3, predictions_segment_inTSO.pr3)),
              'loss': "%.4f" % (epoch_loss/ batches),
              'loss1': "%.2f" % (np.mean(loss1_s)),
              'loss2': "%.2f" % (np.mean(loss2_s)),
              'loss3': "%.2f" % (np.mean(loss3_s)),
              'steps':batches}
                
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
    pretrained_model_path=str(args.pretrained_model_path)
    freeze_encoder=(str(args.freeze_encoder)).lower()=="true"
    scaler_path=str(args.scaler_path)
    
    dataset_name = os.path.basename(input_data_folder.rstrip("/raw"))
    results_folder = f"/mnt/data/GENEActive-featurized/results/DL/{dataset_name}/{results_folder_name}/"
    # results_folder = f"/mnt/data/GENEActive-featurized/results/DL/{dataset_name}/{results_folder_name}/"
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
    processed_data_folder= os.path.join(input_data_folder.rstrip("/raw"), "processed/")

    if clear_tracker_flag.lower()=="true" and os.path.exists(job_tracking_folder):
        shutil.rmtree(job_tracking_folder, ignore_errors=True)

    create_folder([results_folder, training_output_folder, job_tracking_folder, predictions_output_folder,
                   confusion_matrix_plots_folder, learning_plots_output_folder, roc_auc_plots_output_folder,
                   train_models_folder, train_models_folder_joblib,training_logs_folder,processed_data_folder])

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
    
    if 'nsucl' in dataset_name:
        df= load_data_nsucl(input_data_folder)
    else:
        energy_threshold=5
        data_image_path=os.path.join(processed_data_folder, f"{dataset_name}_df_th{energy_threshold}.parquet.gzip")
        if os.path.exists(data_image_path):
            print("Data image file found! Using existing image.")
            df=pd.read_parquet(data_image_path)
        else:
            print("Data image file does not exist. Creating a new one..")
            df= load_data(input_data_folder,energy_th=energy_threshold,remove_outbed=False)
            df.to_parquet(data_image_path,index=False)
    
    max_seq_len=df.groupby('segment').count().max(1).max()+1
    #max_seq_len=1200*2
    #Manual scaling to bring data to -1,1 . The data is in 8g but found that the min max can go up to -/+8.6g so deviding by 9
#     c_to_scale=['x', 'y', 'z']
#     df.loc[:, c_to_scale]=df.loc[:, c_to_scale]/9

    batch_size=best_params['batch_size']

    if num_gpu=="NA":
        device_ids=[x for x in range(0,torch.cuda.device_count(),1)]
        np.random.shuffle(device_ids) 
        num_gpu=device_ids[0]
    
    device_ids=[x for x in range(0,torch.cuda.device_count(),1)]
    np.random.shuffle(device_ids) 
    device = torch.device(f"cuda:{num_gpu}" if torch.cuda.is_available() else "cpu")
    
#     if torch.cuda.device_count() > 1:
#         model = nn.DataParallel(model,device_ids=device_ids) 
#     device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)

    print(device)

    custom_print(f"Using device: {device}")
    if "cpu" in str(device):
        raise Exception("CPU Selected")
    print(input_data_folder)
    
    #input_subjects = df['SUBJECT'].unique()  # debug: with DF's subjects
    if testing =="production":
        df["FOLD"]="All"
        PID_name="FOLD"
    elif testing=="LOSO":
        PID_name="PID"
    else:
        PID_name="FOLD"
    input_subjects = df[PID_name].unique()  # debug: with DF's subjects
    #input_subjects= ['FOLD1'] # Test only on Fold1. No cross validation
    print(input_subjects)
    for i in range(len(input_subjects)):
        torch.cuda.empty_cache()
        
        
        in_progress_subjects = list(os.listdir(folder_paths["job_tracking_folder"]))
        available_subjects = sorted([subject for subject in input_subjects if subject not in in_progress_subjects])

        if len(available_subjects) == 0:
            custom_print(f"No subject available for processing, in_progress/processed_subjects={in_progress_subjects}")
            return

        test_subject_id = available_subjects[0]#random.choice(available_subjects)
        #test_subject_id = input_subjects[i]
        # test_subject_id = "US10008003"  # todo: Manual Input, Single Subject Test
        custom_print(f"Starting training with subject-out={test_subject_id}")
        
        df.columns = df.columns.astype(str)   # sklearn error
        if testing =="production":
            df_test = df
            df_train = df
        else:
            df_test = df[df[PID_name] == test_subject_id]
            df_train = df[df[PID_name] != test_subject_id]
            
        if remove_hnv.lower()=="true":
            df_train=df_train[~df_train.PID.isin(['US10001001', 'US10001002', 'US10012004', 'US10012006', 'US10013002'])] #remove HNV from training
        
        df_train=df_train[df_train.inTSO==True] #Only train on inTSO data
        with open(os.path.join(folder_paths["job_tracking_folder"], test_subject_id), "w") as file:
            file.write("-")  # placeholder file

        # -------------------------------------
        # Prepare train, valid and test dataset
        # -------------------------------------
        # Feature Scaling
        random_indices= df_subset_segments(df_train, 0.2,False)    
        df_val=df_train.iloc[df_train.index.isin(random_indices)]      
        df_train=df_train.iloc[~df_train.index.isin(random_indices)]
        
        if data_augmentation > 0:
            print(f"Augmenting training data with {data_augmentation} iterations. {datetime.now()}")
            intitial_size=df_train.size
            if 'nsucl' in dataset_name:
                df_train=augment_dataset(df_train,data_augmentation,verbose=False)  
            else:
                #df_augmented= load_data_nsucl('/mnt/data/Nocturnal-scratch/nsucl_leap_20hz_2s_b1s/')
                #df_augmented=augment_dataset(df_augmented,data_augmentation,verbose=False)
                #df_train=pd.concat([df_train,df_augmented], ignore_index=True)
                df_train=augment_dataset(df_train,data_augmentation,interchange=True,verbose=False)
            max_seq_len=df_train.groupby('segment').count().max(1).max()+1
            print(f"Training data augmented from {intitial_size} to {df_train.size}. Max sequence length: {max_seq_len}. {datetime.now()}") 

           
        #scale the data
        if pretrained_model_path and os.path.exists(scaler_path):
            # Load existing scaler from pretraining
            print(f"Loading pretrained scaler from: {scaler_path}")
            scaler = joblib.load(scaler_path)
        else:
            # Create and fit new scaler
            print("Creating new scaler and fitting on training data")
            scaler = StandardScaler()
            #scaler = MinMaxScaler()
        
        c_to_scale=['x', 'y', 'z']
        
        if pretrained_model_path and os.path.exists(scaler_path):
            # Apply pretrained scaler to all datasets
            df_train.loc[:, c_to_scale] = scaler.transform(df_train[c_to_scale])
            df_val.loc[:, c_to_scale] = scaler.transform(df_val[c_to_scale])
            df_test.loc[:, c_to_scale] = scaler.transform(df_test[c_to_scale])
        else:
            # Fit scaler on training data and transform all datasets
            df_train.loc[:, c_to_scale] = scaler.fit_transform(df_train[c_to_scale])
            df_val.loc[:, c_to_scale] = scaler.transform(df_val[c_to_scale])
            df_test.loc[:, c_to_scale] = scaler.transform(df_test[c_to_scale])
        
#         optimizer = optim.AdamW(model.parameters(), lr=1e-4)#

        wl1  = best_params["wl1"]
        wl2  = best_params["wl2"]
        wl3  = best_params["wl3"]
        pos_weight_l1 = best_params['pos_weight_l1']
        pos_weight_l2 = best_params['pos_weight_l2']
        stratify = best_params['stratify']
        

        nb_steps = get_nb_steps(df=df_train,batch_size=batch_size,stratify=stratify,shuffle=True)
        # Training loop
        retrain = Retraining (verbose=True)
#         max_seq_len=2093 # TODO: change max seq to a fixed number =1200 * 2
#         max_seq_len=1221
        for i in range (training_iterations):
            model = setup_model(
                model_name=model_name,
                input_tensor_size=3, #42 9 
                max_seq_len=max_seq_len, #df.position_segment.max()+1
                best_params=best_params,
                pretraining=pretraining,
                num_classes=1
            )
#             model.load_state_dict(torch.load('/mnt/data/GENEActive-featurized/results/DL/nsucl_leap_20hz_2s_b1s/MBA_pretraining_param11/training/model_weights/mbatsm_test_subject_All_weights.pth', map_location=device))
#             model.load_state_dict(torch.load('/mnt/data/GENEActive-featurized/results/DL/geneactive_20hz_2s_b1s_imerit/MBA_pretraining_param11/training/model_weights/mbatsm_test_subject_All_weights.pth', map_location=device))

            # # model.out1_conv=model.out2_conv=model.out3_conv= nn.Identity()
            model = model.to(device)

            # Load pretrained weights if provided
            if os.path.exists(pretrained_model_path):
                if model_name == "mba_encoder":
                    # Handle MBATSMForPretraining model
                    print(f"\n Loading pretrained weights for MBATSMForPretraining finetuning...")
                    model.load_pretrained_encoder(pretrained_model_path)
                    model.set_mode(pretraining=False)  # Switch to fine-tuning mode
                    if freeze_encoder:
                        model.freeze_encoder()
                        print("Encoder frozen for fine-tuning.")
                    print("Pretrained weights loaded and switched to fine-tuning mode!")

            total_params = 0
            trainable_params = 0
            print("\nModel Parameters by Layer:")
            print("="*80)
            for name, param in model.named_parameters():
                num_params = param.numel()
                total_params += num_params
                if param.requires_grad:
                    trainable_params += num_params
                print(f"  {name:60s} | {num_params:>12,} | {'Trainable' if param.requires_grad else 'Frozen'}")
            print("="*80)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Frozen parameters: {total_params - trainable_params:,}")

#             model.out1_pretrain=model.out2_pretrain=model.out3_pretrain= nn.Identity()
#             model.pretraining = False
            
#             for param in model.feature_extractor.parameters():
#                 param.requires_grad = False
#             for param in model.encoder.parameters():
#                 param.requires_grad = True

            lr=best_params['lr']
            #lr=0.001
#             optimizer = {
#                 'AdamW': optim.AdamW(model.parameters(), lr=lr), #,weight_decay=wd
#                 'SGD': optim.SGD(model.parameters(), lr=lr, momentum=0.9),
#                 'RMSprop': optim.RMSprop(model.parameters(), lr=lr),
#                 'RAdam': optim.RAdam(model.parameters(), lr=lr),
#             }[best_params['optim']]
            optimizer = {
                'AdamW': optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr), #,weight_decay=wd
                'SGD': optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9),
                'RMSprop': optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=lr),
                'RAdam': optim.RAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr),
            }[best_params['optim']]

            
    #         optimizer = optim.SGD(model.parameters(), lr=1e-2,momentum=0.9, nesterov=True)#
            #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min', factor=0.8, patience=2, verbose=True)
            #scheduler  = CosineWarmupScheduler(optimizer=optimizer, warmup=round(3*nb_steps), max_iters=nb_steps)
            
            cycle=1
            #optimizer = optim.Adam(model.parameters(), lr=1e-4)
            #scheduler  = CosineWarmupScheduler(optimizer=optimizer, warmup=round(0.05*nb_steps), max_iters=nb_steps)
            scheduler =  optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cycle*nb_steps, T_mult=1)

            # -------------
            # Run Training
            # -------------
            train_accuracies = []
            train_losses = [] 
            train_F1s = []
            val_accuracies = []
            val_losses = [] 
            val_F1s = []
        
            print(f"training number {i} started at {datetime.now()}")  
            early_stopping = EarlyStopping(patience=10, verbose=True)
            torch.cuda.empty_cache()
            for epoch in range(epochs):
                model.train()
                model,train_metrics,_=run_model(model,df_train,batch_size,True,device,optimizer,scheduler,stratify=stratify,wl1=wl1,wl2=wl2,wl3=wl3,verbose=False,pos_weight_l1=pos_weight_l1,pos_weight_l2=pos_weight_l2,pretraining=pretraining,max_seq_len=None)
                train_losses.append(train_metrics['loss'])
                if not pretraining:
                    train_F1s.append(train_metrics['F1_1'])
                    train_accuracies.append(train_metrics['Accuracy1'])
                else:
                    train_F1s.append(train_metrics['MSE'])
                    train_accuracies.append(train_metrics['R2'])
                    
                with torch.no_grad():
                    model.eval()
                    _,val_metrics,val_predictions=run_model(model,df_val,batch_size,False,device,optimizer,scheduler,stratify=False,wl1=wl1,wl2=wl2,wl3=wl3,verbose=False,pos_weight_l1=pos_weight_l1,pos_weight_l2=pos_weight_l2,pretraining=pretraining,max_seq_len=None)
                    val_losses.append(val_metrics['loss'])
                    if not pretraining:
                        val_F1s.append(val_metrics['F1_1'])
                        val_accuracies.append(val_metrics['Accuracy1'])
                    else:
                        val_F1s.append(val_metrics['MSE'])
                        val_accuracies.append(val_metrics['R2'])
                    #scheduler.step(val_metrics['loss'])
#                     m=-1*(float(val_metrics['F1_1'])/100)+float(val_metrics['R2'])
                    m=float(val_metrics['loss'])
                    if early_stopping(m, model):
                        print(f"Early stopping at epoch {epoch+1}")
                        early_stopping.restore(model)
                        break 
                print("-----------------------")
                print(f'Iteration {i}. Epoch [{epoch+1}/{epochs}]: Time: {datetime.now()} \n   training: {train_metrics} ,\n   validation:{val_metrics}')
                
                #todo: Fix this to record training metrics of the best training iteration
            learning_metrics={
                        "train_losses": train_losses,
                        "train_accuracies": train_accuracies,
                        "train_F1s": train_F1s,
                        "val_losses": val_losses,
                        "val_accuracies": val_accuracies,
                        "val_F1s": val_F1s
                    }
            try:
                pd.DataFrame(learning_metrics).to_csv(os.path.join(folder_paths["training_logs_folder"],f"training{i}_losses_accuracies_test_subject_{test_subject_id}.csv"), encoding="utf-8",index=False)
            except Exception as e:
                print(f"Failed to Write file. Exception: {e}")
            try:
                plot_learning_curves(
                    learning_metrics=learning_metrics,
                    output_filepath=os.path.join(folder_paths["learning_plots_output_folder"],f"training{i}_losses_accuracies_test_subject_{test_subject_id}.jpg")
                )
            except Exception as e:
                print(f"Failed to plot learning curves. Exception: {e}")
                
                
            retrain.check_performance(min(val_losses),model)
        
        #Get bets model if trained more than once
        model=retrain.restore(model)
        
            
        with torch.no_grad():
            model.eval()
            _,test_metrics,test_predictions=run_model(model,df_test,batch_size,False,device,optimizer,scheduler,stratify=False,wl1=wl1,wl2=wl2,wl3=wl3,verbose=False,pos_weight_l1=pos_weight_l1,pos_weight_l2=pos_weight_l2,pretraining=pretraining,max_seq_len=None)

        print("-----------------------")
        print(f'Testing:{test_metrics}, Time: {datetime.now()}')

        # ------------------
        # Run Test & Metrics
        # ------------------


        # . ---------
        # Save Model
        # . ---------
        #         joblib.dump(model, os.path.join(train_models_folder_joblib, f"{model_name}_test_subject_{test_subject_id}_joblib_dump.pth"))
        torch.save(model.state_dict(),
                   os.path.join(folder_paths["train_models_folder"], f"{model_name}_test_subject_{test_subject_id}_weights.pth"))
        #save scaler
        joblib.dump(scaler, os.path.join(folder_paths["train_models_folder"], f"{model_name}_test_subject_{test_subject_id}_scaler.joblib"))
                                         
        write_txt_to_file(
            file_path=os.path.join(folder_paths["training_logs_folder"], f"processing_logs_test_subject_{test_subject_id}.log"),
            text=str(processing_logs)
        )
        
        if pretraining:
            pd.DataFrame(pd.json_normalize(test_metrics)).to_csv(
                os.path.join(folder_paths["predictions_output_folder"], f"final_metrics_sheet_test_subject_{test_subject_id}.csv"),
                encoding="utf-8",
                index=False
            )
            test_predictions.to_csv(
                os.path.join(folder_paths["predictions_output_folder"], f"Y_test_Y_pred_test_subject_{test_subject_id}.csv"),
                encoding="utf-8",
                index=False
            )
        else:
            test_predictions.to_csv(
                os.path.join(folder_paths["predictions_output_folder"], f"Y_test_Y_pred_test_subject_{test_subject_id}.csv"),
                encoding="utf-8",
                index=False
            )
            segment_level_predictions=test_predictions[test_predictions.inTSO==True].groupby('segment').max(1)
            # Test the model         
            final_metrics = {'pr1':calculate_metrics_nn(segment_level_predictions.gt1, segment_level_predictions.pr1),
                             'pr2':calculate_metrics_nn(test_predictions[test_predictions.inTSO==True].gt2, test_predictions[test_predictions.inTSO==True].pr2),
                             'pr3':calculate_metrics_nn(segment_level_predictions.gt3, segment_level_predictions.pr3,False)
                            } 
            
            plot_confusion_metrics(
            y_pred=segment_level_predictions.pr1,
            y_test=segment_level_predictions.gt1,
            output_filepath=os.path.join(folder_paths["confusion_matrix_plots_folder"], f"confusion_matrix_test_subject_{test_subject_id}.jpg")
            )
            
#             # Save Predictions
#             pd.DataFrame(pd.json_normalize(final_metrics)).to_csv(
#                 os.path.join(folder_paths["predictions_output_folder"], f"final_metrics_sheet_test_subject_{test_subject_id}.csv"),
#                 encoding="utf-8",
#                 index=False
#             )
            
            with open(os.path.join(folder_paths["predictions_output_folder"], f"final_metrics_sheet_test_subject_{test_subject_id}.json"), 'w') as json_file:
                json.dump(final_metrics, json_file, indent=4)
            


            custom_print(final_metrics)
        
        # todo: pass y_score/probabilities instead of y_pred
#         plot_roc_precision_recall_auc(
#             y_test=y_test,
#             y_score=y_pred,
#             output_filepath=os.path.join(folder_paths["roc_auc_plots_output_folder"],
#                                          f"roc_auc_plot_test_subject_id_{test_subject_id}.jpg")
#         )

        custom_print("Job finished successfully...")
        custom_print(f"Finished Training Now: {datetime.now()}")
        # break  # todo: this experiment only runs 1 subject split test (US10008003)

    return 1

def objective(trial,df_train, df_val,df_test, param_tuning_output_trials,num_gpu, max_seq_len): #TODO: add number of layers in feature extractor to the search space

    if num_gpu=="NA":
        num_gpu=trial.number%4
    device = torch.device(f"cuda:{num_gpu}" if torch.cuda.is_available() else "cpu")
    batch_sizes = [16,32]
    batch_size = 16#trial.suggest_categorical ('batch_size',[16])
    num_filters = trial.suggest_categorical ('num_filters',[32,64,128])
    lr = trial.suggest_categorical('lr', [0.001,0.0001])
#     wd = trial.suggest_float('wd', 1e-6, 1e-1, log=True)
    dropout_rate = trial.suggest_float('dropout', 0, 0.5,step=0.1)
    drop_path_rate = trial.suggest_float('droppath', 0, 0.5,step=0.1)
    channel_masking_rate = trial.suggest_float('channel_masking_rate', 0, 0.5,step=0.1)
#     kernel_size_feature =  trial.suggest_int("kernel_f", 3, 15, step=2)
    kernel_size_mba1 = trial.suggest_int("kernel_MBA1", 3, 15, step=2)
    kernel_size_mba2 = trial.suggest_int("kernel_MBA2", 3, 15, step=2)
#     num_feature_layers = trial.suggest_int("num_feature_layers", 1, 10, step=1)
    num_MBA_blocks1 = trial.suggest_int("blocks_MBA1", 0, 8, step=1)
    num_MBA_blocks2 = trial.suggest_int("blocks_MBA2", 0, 8, step=1)
    dilation1=trial.suggest_categorical ('dilation1',[True,False])
    dilation2=trial.suggest_categorical ('dilation2',[True,False])
    MBA_encoder = True #trial.suggest_categorical("MBA_encoder", [True])#,False finetuning transformer fails due to memory.
    wl1  = trial.suggest_float("wl1", 0, 10,step=1)
    wl2  = trial.suggest_float("wl2", 0, 10,step=1)
    wl3  = trial.suggest_float("wl3", 0, 10,step=1)
    optimizer= trial.suggest_categorical ('optim',['AdamW','RMSprop','RAdam'])#, 'SGD', 'RMSprop'
    pos_weight_l1 = trial.suggest_categorical ('pos_weight_l1',[True,False])
    pos_weight_l2 = trial.suggest_categorical ('pos_weight_l2',[True,False])
    cls_token = trial.suggest_categorical ('cls_token',[True,False])
    featurelayer=trial.suggest_categorical ('featurelayer',["TCN","ResTCN"])
    num_heads = trial.suggest_categorical ('num_heads',[2,4])
    #stratify=trial.suggest_categorical ('stratify',['oversample','undersample',False])
    stratify='undersample'#trial.suggest_categorical ('stratify',['undersample'])
    nb_steps = get_nb_steps(df=df_train,batch_size=batch_size,stratify='undersample',shuffle=True)
    print(f"Trial number: {trial.number} running on {device}. Time: {datetime.now()}. {trial.params}")
    epochs = 300
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
            model = MBA_tsm(input_size,num_encoder1_layers=num_MBA_blocks1,num_encoder2_layers=num_MBA_blocks2,dilation1=dilation1,dilation2=dilation1,drop_path_rate =drop_path_rate ,kernel_size_mba1=kernel_size_mba1,kernel_size_mba2=kernel_size_mba2,dropout_rate=dropout_rate, max_seq_len= max_seq_len,cls_token=cls_token,mba=MBA_encoder,num_heads=num_heads,num_filters=num_filters,channel_masking_rate=channel_masking_rate)
            

            model = model.to(device)

            optimizer = {'AdamW': optim.AdamW(model.parameters(), lr=lr), #,weight_decay=wd
                'SGD': optim.SGD(model.parameters(), lr=lr, momentum=0.9),
                'RMSprop': optim.RMSprop(model.parameters(), lr=lr),
                'RAdam': optim.RAdam(model.parameters(), lr=lr),
            }[optimizer]

    #         scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min', factor=0.8, patience=2, verbose=True)
            cycle=1
            scheduler =  optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cycle*nb_steps, T_mult=1)
            early_stopping = EarlyStopping(patience=10, verbose=False)
            for epoch in range(epochs):
                model.train()
                model,train_metrics,_=run_model(model,df_train,batch_size,True,device,optimizer,scheduler,stratify=stratify,wl1=wl1,wl2=wl2,wl3=wl3,verbose=False,pos_weight_l1=pos_weight_l1,pos_weight_l2=pos_weight_l2,pretraining=False,max_seq_len=None)
                with torch.no_grad():
                    model.eval()
                    _,val_metrics,val_predictions=run_model(model,df_val,batch_size,False,device,optimizer,scheduler,stratify=False,wl1=wl1,wl2=wl2,wl3=wl3,verbose=False,pos_weight_l1=pos_weight_l1,pos_weight_l2=pos_weight_l2,pretraining=False,max_seq_len=None)
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
            _,test_metrics,test_predictions=run_model(model,df_test,batch_size,False,device,optimizer,scheduler,stratify=False,wl1=wl1,wl2=wl2,wl3=wl3,verbose=False,pos_weight_l1=pos_weight_l1,pos_weight_l2=pos_weight_l2,pretraining=False,max_seq_len=None)

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
    dataset_name = os.path.basename(input_data_folder.rstrip("/raw"))
    results_folder = f"/mnt/data/GENEActive-featurized/results/DL/{dataset_name}/{results_folder_name}/"
    param_tuning_output_folder = os.path.join(results_folder, "tuning/")
    param_tuning_output_trials = os.path.join(param_tuning_output_folder, "trials/")
    processed_data_folder= os.path.join(input_data_folder.rstrip("/raw"), "processed/")
    create_folder([param_tuning_output_folder,param_tuning_output_trials, processed_data_folder])

    # Write user provided arguments
    with open(os.path.join(param_tuning_output_folder, "user_arguments.txt"), "w") as file_handler:
        file_handler.write(str(args))
    
    if 'nsucl' in dataset_name:
        df= load_data_nsucl(input_data_folder)#load_data
    else:
        energy_threshold=5
        data_image_path=os.path.join(processed_data_folder, f"{dataset_name}_df_th{energy_threshold}.parquet.gzip")
        if os.path.exists(data_image_path):
            print("Data image file found! Using existing image.")
            df=pd.read_parquet(data_image_path)
        else:
            print("Data image file does not exist. Creating a new one..")
            df= load_data(input_data_folder,energy_th=energy_threshold,remove_outbed=False)
            df.to_parquet(data_image_path,index=False)
    
    df=df[df.inTSO==True]
    df_train=df[df.FOLD!='FOLD4']
    df_test=df[df.FOLD=='FOLD4']
    
    max_seq_len= df.groupby('segment').count().max(1).max()+1
    
    from sklearn.model_selection import train_test_split

    # Assuming df is your DataFrame
    train_segments, validation_segments = train_test_split(df_train.segment.unique(), test_size=0.2, random_state=42)
    df_val=df_train[df_train.segment.isin(validation_segments)]      
    df_train=df_train[~df_train.segment.isin(train_segments)]
#     random_indices= df_subset_segments(df_train, 0.2,False)
#     df_val=df_train.iloc[df_train.index.isin(random_indices)]      
#     df_train=df_train.iloc[~df_train.index.isin(random_indices)]
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
    
    
    storage = JournalStorage(JournalFileBackend(f"{results_folder}{model_name}.log"))
    
    study = optuna.create_study(directions=['maximize','maximize'],study_name=study_name,storage=storage,load_if_exists=True)#'maximize',
    tune_objective=partial(objective, df_train=df_train, df_val=df_val,df_test=df_test, param_tuning_output_trials=param_tuning_output_trials,num_gpu=num_gpu, max_seq_len= max_seq_len)
#     with parallel_backend("multiprocessing", n_jobs=4):
    study.optimize(tune_objective,callbacks=[MaxTrialsCallback(600, states=(TrialState.COMPLETE,))],n_jobs=4)

    print('Best hyperparameters: ', study.best_trials[0]) # Selecting the first Pareto-optimal trial

    write_txt_to_file(file_path=os.path.join(param_tuning_output_folder, f"best_trial_params.txt"),
                      text=str(study.best_trials[0].params))
    
    custom_print("Job finished successfully...")
    custom_print(f"Finished Tuning Now: {datetime.now()}")


    return 1


# Examples: 

### NOPROD

## Training

# python 'munge/predictive_modeling/predict_scratch_segment.py' --input_data_folder '/mnt/data/Nocturnal-scratch/geneactive_20hz_3s_b1s_grav_stdrange_start18h_sleepy_original_nonwear60_temp25_sleep60_win/raw/' --model mbatsm --output MBA_p13_0FL_2ENC --execution_mode train  --num_gpu 3 --training_iterations 2

 # python 'munge/predictive_modeling/predict_scratch_segment.py' --input_data_folder '/mnt/data/Nocturnal-scratch/geneactive_20hz_3s_b1s_grav_stdrange_start18h_sleepy_original_nonwear60_temp25_sleep60_win/raw/' --model mbatsm_ED --output MBA_ED_param11_1 --execution_mode train  --num_gpu 1

# # python 'munge/predictive_modeling/predict_scratch_segment.py' --input_data_folder '/mnt/data/Nocturnal-scratch/geneactive_20hz_3s_b1s_grav_stdrange_start18h_sleepy_original_nonwear60_temp25_sleep60_win/raw/' --model mbatsm --output MBA_param16 --execution_mode train --training_iterations 2  --num_gpu 1

# python 'munge/predictive_modeling/predict_scratch_segment.py' --input_data_folder '/mnt/data/Nocturnal-scratch/geneactive_20hz_2s_b0s_nograv_start19h_van_original/raw/' --model mba_ed --output MBA_param11_mbaED --execution_mode train  --num_gpu 0

# python 'munge/predictive_modeling/predict_scratch_segment.py' --input_data_folder '/mnt/data/Nocturnal-scratch/geneactive_20hz_2s_b0s_nograv_start19h_van_original/raw/' --model cnnautoencoder --output MBA_param11_cnnautoencoder --execution_mode train  --num_gpu 0

# python 'munge/predictive_modeling/predict_scratch_segment.py' --input_data_folder '/mnt/data/Nocturnal-scratch/geneactive_20hz_2s_b1s_imerit_sleep_nonwearOR_5smerge_nograv_start19h/raw/' --model mbatsm --output MBA_param11_sleep --execution_mode train  --num_gpu 0 --training_iterations 2

# python 'munge/predictive_modeling/predict_scratch_segment.py' --input_data_folder '/mnt/data/Nocturnal-scratch/geneactive_20hz_2s_b1s_imerit_sleep_nonwearOR_5smerge_nograv_nochange/raw/' --model mbatsm --output MBA_param11_energy1 --execution_mode train --num_gpu 0
                                         
# python 'munge/predictive_modeling/predict_scratch_segment.py' --input_data_folder '/mnt/data/Nocturnal-scratch/geneactive_20hz_2s_b1s_imerit_sleep_nonwearOR/raw/' --model mbatsm --output MBA_param11_energy1_itr2_LOSO --execution_mode train --training_iterations 2 --testing LOSO --num_gpu 0

# python 'munge/predictive_modeling/predict_scratch_segment.py' --input_data_folder '/mnt/data/Nocturnal-scratch/geneactive_20hz_2s_b1s_imerit_sleep_nonwearOR_5smerge_nograv_change/raw/' --model mbatsm --output MBA_param11_energy1_DA1 --execution_mode train --data_augmentation 1 --num_gpu 0

# python 'munge/predictive_modeling/predict_scratch_segment.py' --input_data_folder '/mnt/data/Nocturnal-scratch/geneactive_20hz_2s_b1s_imerit_sleep_nonwearOR_5smerge/raw/' --model mbatsm --output MBA_param11_energy1_itr3_nostratify_nobed --execution_mode train --training_iterations 3 --num_gpu 0

# python 'munge/predictive_modeling/predict_scratch_segment.py' --input_data_folder '/mnt/data/Nocturnal-scratch/geneactive_20hz_2s_b1s_imerit_sleep_nonwearZhou/raw/' --model mbatsm --output MBA_param11_sleep --execution_mode train



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

##Tuning

#python 'munge/predictive_modeling/predict_scratch_segment.py' --input_data_folder '/mnt/data/Nocturnal-scratch/geneactive_20hz_3s_b1s_grav_stdrange_start18h_sleepy_original_nonwear60_temp25_sleep60_win/raw/' --model mbatsm --output MBA_tune --execution_mode tune  --num_gpu 0


# python 'munge/predictive_modeling/predict_scratch_segment.py' --input_data_folder '/mnt/data/Nocturnal-scratch/geneactive_20hz_2s_b1s_std/'  --model mbatsm --output base_model2 --execution_mode tune --num_gpu 2

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
# scaler_path = "/mnt/data/GENEActive-featurized-consolidated/results/leave-1-out-cv/time_windows_results_2s_1s/standard_scaler_2s.joblib"
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
