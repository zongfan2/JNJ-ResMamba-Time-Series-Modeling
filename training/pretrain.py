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
# import mlflow

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
parser.add_argument("--training_iterations",type=int,default=1, required=False, help="The number of training iterations. Default is 1.")
parser.add_argument("--pretraining_method",type=str,default="AIM", required=False, help="Pretraining method: AIM, DINO, or RelCon. Default is AIM.")
parser.add_argument("--pretrained_weights_path",type=str,default="", required=False, help="Path to pretrained weights file (.pth) from a pretrainer. If provided, loads these weights into the base model before training.")



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
import glob
import re
from functools import partial
import gc
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
import matplotlib.pyplot as plt
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils.rnn import pad_sequence

from models import *
from data import *
from losses import *
from evaluation import *
from utils import *

from models.pretraining import AIMPretrainer, DINOPretrainer, RelConPretrainer, SimCLRPretrainer
from models.pretraining.dataparallel import create_dataparallel_pretrainer

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

def parse_gpu_config(num_gpu_arg):
    """
    Parse num_gpu argument and return GPU configuration.
    
    Args:
        num_gpu_arg: String argument from --num_gpu
        
    Returns:
        tuple: (use_dataparallel: bool, device_ids: list, primary_device: str)
        
    Examples:
        "NA" or "0" -> (False, None, "cuda:0")
        "1" -> (False, None, "cuda:1") 
        "4" -> (True, [0,1,2,3], "cuda:0")
        "0,1,2" -> (True, [0,1,2], "cuda:0")
    """
    if num_gpu_arg == "NA":
        return False, None, "cuda:0"
    
    # Check if it's a comma-separated list of specific GPUs
    if "," in num_gpu_arg:
        device_ids = [int(x.strip()) for x in num_gpu_arg.split(",")]
        if len(device_ids) > 1:
            return True, device_ids, f"cuda:{device_ids[0]}"
        else:
            return False, None, f"cuda:{device_ids[0]}"
    
    # Single number - could be single GPU ID or number of GPUs to use
    try:
        num = int(num_gpu_arg)
        if num <= 1:
            # Single GPU
            return False, None, f"cuda:{max(0, num)}"
        else:
            # Multiple GPUs: use first N GPUs (0, 1, 2, ..., N-1)
            device_ids = list(range(num))
            return True, device_ids, "cuda:0"
    except ValueError:
        print(f"Warning: Invalid num_gpu value '{num_gpu_arg}', defaulting to cuda:0")
        return False, None, "cuda:0"

def load_optuna_pretrained_best_params(optuna_best_param_path):
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
                    '13': {'batch_size': 16, 'num_filters': 64, 'lr': 0.001, 'dropout': 0.2, 'droppath': 0.4, 'kernel_f': 13, 'kernel_MBA': 11, 'num_feature_layers': 7, 'blocks_MBA': 7, 'MBA_encoder': True, 'wl1': 0.5, 'wl2': 0.5, 'wl3': 12, 'optim': 'RMSprop', 'pos_weight_l1': False, 'pos_weight_l2': True, 'cls_token': False, 'featurelayer': 'ResTCN', 'num_heads': 4, 'stratify': 'undersample', "random_padding": True, "average_mask": True, "average_window_size": 20},
                    '14': {'batch_size': 16, 'num_filters': 64, 'lr': 0.001, 'dropout': 0.2, 'droppath': 0.2,'channel_masking_rate': 0, 'kernel_f': 13,'kernel_MBA': 11, 'kernel_MBA1': 11, 'kernel_MBA2': 11, 'num_feature_layers': 7, 'blocks_MBA': 7,'blocks_MBA1': 0, 'blocks_MBA2': 0,'dilation1': True, 'dilation2': True,'encoder_bottelneck':1, 'MBA_encoder': True, 'wl1': 0.5, 'wl2': 0.5, 'wl3': 0.9, 'optim': 'RMSprop', 'pos_weight_l1': False, 'pos_weight_l2': True, 'cls_token': True, 'featurelayer': 'ResNet', 'num_heads': 4, 'stratify': 'undersample','norm':'GN', 'use_cls_token': False},
                    '15': {'batch_size': 16, 'num_filters': 64, 'lr': 0.001, 'dropout': 0.2, 'droppath': 0.2,'channel_masking_rate': 0, 'kernel_f': 13,'kernel_MBA': 11, 'kernel_MBA1': 11, 'kernel_MBA2': 11, 'num_feature_layers': 0, 'blocks_MBA': 7,'blocks_MBA1': 7, 'blocks_MBA2': 0,'dilation1': True, 'dilation2': True,'encoder_bottelneck':1, 'MBA_encoder': True, 'wl1': 0.5, 'wl2': 0.5, 'wl3': 0.9, 'optim': 'RMSprop', 'pos_weight_l1': False, 'pos_weight_l2': True, 'cls_token': True, 'featurelayer': 'TCN', 'num_heads': 4, 'stratify': 'undersample','norm':'GN', 'use_cls_token': True, 'add_positional_encoding': True},
                }

    lsm2_params_list = {
        '1': {'batch_size': 64, 'patch_size': 200, 'stride': 20, 'encoder_layers': 4, 'decoder_layers': 2, 'd_model': 192, 'num_heads': 4, 'lr': 1.5e-3,  'optim': 'AdamW', 'stratify': "undersample",  "random_padding": False,  "padding_value": -999.0,},
    }

    conv1d_params_list = {
        '1': {'batch_size': 16, 'num_layers': 5, 'd_model': 64, 'lr': 3e-4, 'wl1': 0.8, 'wl2': 0.5, 'wl3': 12, 'optim': 'AdamW', 'pos_weight_l1': False, 'pos_weight_l2': True, 'stratify': "undersample", "norm_scratch_length": True, "reg_head": True, "random_padding": False, "average_mask": False, "padding_value": -10, "supcon_loss": True, "mixup_alpha": 0.1, "mixup_prob": 0.5},
    }

    # return patchtst_params_list['1']
    return params_list['15']
    # return efficient_unet_params_list['4']
    # return vit_params_list['1']
    # return swin_params_list['1']
    # return lsm2_params_list['1']
    # return conv1d_params_list['1']

    


def generate_unique_random(min_val, max_val, excluded):
    possible_numbers = set(range(min_val, max_val + 1)) - set(excluded)
    if not possible_numbers:
        return None  # No available number exists
    return random.choice(list(possible_numbers))


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


def load_pretrained_weights_from_pretrainer(base_model, pretrainer_weights_path, pretraining_method="AIM", device=None):
    """
    Load pretrained weights from a saved pretrainer into a base model.
    
    Args:
        base_model: The base model to load weights into
        pretrainer_weights_path: Path to the saved pretrainer weights (.pth file)
        pretraining_method: The pretraining method used ("AIM", "DINO", "RelCon")
        device: Device to load weights on
        
    Returns:
        base_model: Model with loaded pretrained weights
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading pretrained weights from: {pretrainer_weights_path}")
    
    # Load the pretrainer state dict
    pretrainer_state_dict = torch.load(pretrainer_weights_path, map_location=device)
    
    # Extract base model weights based on pretraining method
    base_model_state_dict = {}
    
    if pretraining_method.upper() == "AIM":
        # For AIM, the base model is wrapped as 'vit'
        prefix = "vit."
        for key, value in pretrainer_state_dict.items():
            if key.startswith(prefix):
                # Remove the 'vit.' prefix to get the base model parameter name
                base_key = key[len(prefix):]
                base_model_state_dict[base_key] = value
                
    elif pretraining_method.upper() in ["DINO", "RELCON"]:
        # For DINO and RelCon, the base model is wrapped as 'student' or 'backbone'
        prefixes = ["student.", "backbone."]
        for prefix in prefixes:
            for key, value in pretrainer_state_dict.items():
                if key.startswith(prefix):
                    # Remove the prefix to get the base model parameter name
                    base_key = key[len(prefix):]
                    base_model_state_dict[base_key] = value
                    break
    
    # Filter out parameters that don't exist in the base model
    base_model_keys = set(base_model.state_dict().keys())
    filtered_state_dict = {k: v for k, v in base_model_state_dict.items() if k in base_model_keys}
    
    # Load the filtered weights into the base model
    missing_keys, unexpected_keys = base_model.load_state_dict(filtered_state_dict, strict=False)
    
    print(f"Loaded {len(filtered_state_dict)} pretrained parameters")
    if missing_keys:
        print(f"Missing keys (will use random initialization): {len(missing_keys)} parameters")
    if unexpected_keys:
        print(f"Unexpected keys (ignored): {len(unexpected_keys)} parameters")
    
    return base_model


def mae_pretrain(pretrainer,
              df,
              batch_size,
              train_mode,
              device,
              optimizer,
              scheduler,
              stratify=False,
              use_TSO=True,
              max_seq_len=None, 
              random_padding=False,
              padding_value=-999.0,
              ):

    epoch_loss = batches  = 0.0
    gt1s = gt2s = gt3s = lens = inTSOs = predictTSOs = []
    pr1s = pr1_probs = pr2s = pr2_probs = pr3s = attentions = []
    loss1_s=[]
    loss2_s=[]
    loss3_s=[]
    seg_column='segment'
    segments1=segments2=positions=[]

    for batch in batch_generator(df=df, batch_size=batch_size, stratify=stratify, shuffle=train_mode, seg_column=seg_column):    
        
        padding_position = "random" if random_padding else "tail"
        batch_data, label1, label2, label3, x_lens, seq_start_idx = add_padding_with_position(batch, device, 
                                                                            seg_column=seg_column, 
                                                                            max_seq_len=max_seq_len, 
                                                                            padding_position=padding_position,
                                                                            is_train=train_mode,
                                                                            prob=0.5,
                                                                            padding_value=padding_value)
        # permute
        batch_data = batch_data.permute(0, 2, 1)
        res = pretrainer(batch_data)
        loss = res["loss"]
        # reconstructed = res["reconstructed"]

        # Backward pass and optimization if in training mode
        if train_mode:
            # Check for NaN loss before backward pass
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss detected: {loss.item()}. Skipping batch.")
                continue
                
            optimizer.zero_grad()
            loss.backward()
            
            # Check for NaN gradients
            nan_grads = False
            for name, param in pretrainer.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    print(f"Warning: NaN/Inf gradient in {name}. Skipping batch.")
                    nan_grads = True
                    break
            
            if nan_grads:
                optimizer.zero_grad()
                continue
                
            # Aggressive gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(pretrainer.parameters(), max_norm=0.5)
            optimizer.step()
            scheduler.step()
            
        # Add loss with additional checks
        if not (torch.isnan(loss) or torch.isinf(loss)):
            epoch_loss += loss.item()
        batches += 1
        #Measure performance using the unpadded data
    eval_metrics = {'loss': "%.4f" % (epoch_loss/ batches),
                    'steps':batches}
    return pretrainer,eval_metrics

  
def contrastive_pretrain(pretrainer,
                         df,
                         batch_size,
                         train_mode,
                         device,
                         optimizer,
                         scheduler,
                         stratify=False,
                         use_TSO=True,
                         max_seq_len=None,
                         gradient_clipping=True,
                         max_grad_norm=1.0,
                         include_view_metrics=True
                         ):
    """
    Unified pretraining function for contrastive learning methods (DINO, SimCLR, etc.).
    
    Args:
        pretrainer: The contrastive pretrainer (DINOPretrainer, SimCLRPretrainer, etc.)
        gradient_clipping: Whether to apply gradient clipping (default: True)
        max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
        include_view_metrics: Whether to include view/crop metrics in output (default: True)
    """
    epoch_loss = batches = 0.0
    seg_column='segment'

    for batch in batch_generator(df=df, batch_size=batch_size, stratify=False, shuffle=train_mode, seg_column=seg_column):    
        
        batch_data, x_lens = add_padding_pretrain(batch, 
                                                                device, 
                                                                seg_column=seg_column, 
                                                                max_seq_len=max_seq_len
                                                                )
        # Contrastive pretrainer expects (x, x_lengths)
        res = pretrainer(batch_data, x_lens)
        loss = res["loss"]

        # Backward pass and optimization if in training mode
        if train_mode:
            # Check for NaN loss before backward pass
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss detected: {loss.item()}. Skipping batch.")
                continue
                
            optimizer.zero_grad()
            loss.backward()
            
            # Check for NaN gradients
            nan_grads = False
            for name, param in pretrainer.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    print(f"Warning: NaN/Inf gradient in {name}. Skipping batch.")
                    nan_grads = True
                    break
            
            if nan_grads:
                optimizer.zero_grad()
                continue
                
            # Optional gradient clipping for stability
            if gradient_clipping:
                torch.nn.utils.clip_grad_norm_(pretrainer.parameters(), max_norm=max_grad_norm)
                
            optimizer.step()
            scheduler.step()
            
        # Add loss with additional checks
        if not (torch.isnan(loss) or torch.isinf(loss)):
            epoch_loss += loss.item()
        batches += 1
        
    # Build metrics dictionary
    eval_metrics = {
        'loss': "%.4f" % (epoch_loss/ batches),
        'steps': batches
    }
    
    # Add view/crop metrics if requested
    if include_view_metrics and batches > 0:
        eval_metrics['n_views'] = res.get("n_crops", res.get("n_views", "N/A"))
        
    return pretrainer, eval_metrics

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
    global df, pre_saved_scaler, scaler_path,X_train_tensor,y_train_tensor,X_valid_tensor,y_valid_tensor,y_train_sampler  # todo: remove/debug

    clear_tracker_flag = str(args.clear_tracker)
    model_name = args.model
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
    training_iterations=args.training_iterations
    pretraining_method=args.pretraining_method
    pretrained_weights_path=args.pretrained_weights_path
    
    dataset_name = os.path.basename(input_data_folder.rstrip("/raw"))
    results_folder = f"/domino/datasets/GENEActive-featurized/results/DL/{dataset_name}/{results_folder_name}/"
    # results_folder = f"/domino/datasets/GENEActive-featurized/results/DL/{dataset_name}/{results_folder_name}/"
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
    
    # input_data_folder ='/domino/datasets/Nocturnal-scratch/geneactive_20hz_3s_b1s_production/raw'
    # dataset_name = os.path.basename(input_data_folder.rstrip("/raw"))
    # processed_data_folder= os.path.join(input_data_folder.rstrip("/raw"), "processed/")
    energy_threshold=5
    create_folder([processed_data_folder])
    data_image_path=os.path.join(processed_data_folder, f"{dataset_name}_pretrain_df_th{energy_threshold}.parquet.gzip")
    if os.path.exists(data_image_path):
        print("Data image file found! Using existing image.")
        df=pd.read_parquet(data_image_path)
    else:
        print("Data image file does not exist. Creating a new one..")
        df=load_data_pretrain(input_data_folder,segment_th=energy_threshold)
        df.to_parquet(data_image_path,index=False)
    # if 'nsucl' in dataset_name:
    #     df= load_data_nsucl(input_data_folder)
    # else:
    #     energy_threshold=5
    #     data_image_path=os.path.join(processed_data_folder, f"{dataset_name}_df_th{energy_threshold}.parquet.gzip")
    #     if os.path.exists(data_image_path):
    #         print("Data image file found! Using existing image.")
    #         df=pd.read_parquet(data_image_path)
    #     else:
    #         print("Data image file does not exist. Creating a new one..")
    #         df= load_data(input_data_folder,energy_th=energy_threshold,remove_outbed=False)
    #         df.to_parquet(data_image_path,index=False)
    
    max_seq_len=df.groupby('segment').count().max(1).max()+1

    batch_size=best_params['batch_size']

    if num_gpu=="NA":
        device_ids=[x for x in range(0,torch.cuda.device_count(),1)]
        np.random.shuffle(device_ids) 
        num_gpu=device_ids[0]
    
    device_ids=[x for x in range(0,torch.cuda.device_count(),1)]
    np.random.shuffle(device_ids) 
    device = torch.device(f"cuda:{num_gpu}" if torch.cuda.is_available() else "cpu")


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
        
        # df_train=df_train[df_train.inTSO==True] #Only train on inTSO data
        df_train=df_train[df_train.predictTSO==True] #Only train on inTSO data
        # print("--------------------")
        # print("TEST SEG: ", len(df_train.segment.unique()))

        with open(os.path.join(folder_paths["job_tracking_folder"], test_subject_id), "w") as file:
            file.write("-")  # placeholder file

        # -------------------------------------
        # Prepare train, valid and test dataset
        # -------------------------------------
        # Feature Scaling
          
        train_segments, validation_segments = train_test_split(df_train.segment.unique(), test_size=0.1, random_state=42)
        df_val=df_train[df_train.segment.isin(validation_segments)]      
        df_train=df_train[df_train.segment.isin(train_segments)]
        # print("--------------------")
        # print("TEST SEG: ", len(df_train.segment.unique()))
        # print("TEST SEG: ", len(df_val.segment.unique()))

        # random_indices= df_subset_segments(df_train, 0.2, False)  
        # df_val=df_train.iloc[df_train.index.isin(random_indices)]      
        # df_train=df_train.iloc[~df_train.index.isin(random_indices)]
        
        if data_augmentation > 0:
            print(f"Augmenting training data with {data_augmentation} iterations. {datetime.now()}")
            intitial_size=df_train.size
            if 'nsucl' in dataset_name:
                df_train=augment_dataset(df_train,data_augmentation,verbose=False)  
            else:
                df_train=augment_dataset(df_train,data_augmentation,interchange=True,verbose=False)
            max_seq_len=df_train.groupby('segment').count().max(1).max()+1
            print(f"Training data augmented from {intitial_size} to {df_train.size}. Max sequence length: {max_seq_len}. {datetime.now()}") 

        
        scaler = StandardScaler()
        c_to_scale=['x', 'y', 'z']
        df_train.loc[:, c_to_scale] = scaler.fit_transform(df_train[c_to_scale])
        df_val.loc[:, c_to_scale] = scaler.transform(df_val[c_to_scale])
        df_test.loc[:, c_to_scale] = scaler.transform(df_test[c_to_scale])
        

        wl1  = best_params["wl1"]
        wl2  = best_params["wl2"]
        wl3  = best_params["wl3"]
        pos_weight_l1 = best_params['pos_weight_l1']
        pos_weight_l2 = best_params['pos_weight_l2']
        stratify = best_params['stratify']
        

        nb_steps = get_nb_steps(df=df_train,batch_size=batch_size,stratify=False,shuffle=True)
        # Training loop
        retrain = Retraining (verbose=True)
        
        # Training loop
        for i in range(args.training_iterations):
            print(f"Training iteration {i+1}/{args.training_iterations}")
            
            use_dataparallel, device_ids, device = parse_gpu_config(args.num_gpu)
            print(f"GPU Configuration: use_dataparallel={use_dataparallel}, device_ids={device_ids}, device={device}")

            # Setup model
            base_model = setup_model(
                model_name=args.model,
                input_tensor_size=3,
                max_seq_len=max_seq_len,
                best_params=best_params,
                pretraining=True,
                num_classes=1
            )
            
            print(base_model)

            # Load pretrained weights if provided
            if pretrained_weights_path and os.path.exists(pretrained_weights_path):
                print(f"Loading pretrained weights from: {pretrained_weights_path}")
                base_model = load_pretrained_weights_from_pretrainer(
                    base_model, 
                    pretrained_weights_path, 
                    pretraining_method, 
                    device
                )
                print("Successfully loaded pretrained weights into base model")
            elif pretrained_weights_path:
                print(f"Warning: Pretrained weights path provided but file not found: {pretrained_weights_path}")

            # Setup pretrainer based on method
            if pretraining_method.upper() == "AIM":
                model = AIMPretrainer(
                    vit_model=base_model,
                    mask_ratio=0.75,
                    dropout_rate=0.25,
                    norm_target=True,
                    missing_value=best_params.get('padding_value', -999.0),
                    missing_patch_thres=0.5
                )
            elif pretraining_method.upper() == "DINO":
                if use_dataparallel:
                    # Use DataParallel DINO pretrainer
                    model = create_dataparallel_pretrainer(
                        pretrainer_type='dino',
                        base_model=base_model,
                        device_ids=device_ids,
                        feature_dim=best_params.get('num_filters', 64),
                        projection_dim=best_params.get('proj_dim', 128),
                        projection_hidden=best_params.get('num_filters', 64),
                        momentum=0.996,
                        temperature_student=0.1,
                        temperature_teacher=0.04,
                        center_momentum=0.9,
                        global_crops=2,
                        local_crops=6
                    )
                else:
                    # Use single-GPU DINO pretrainer
                    model = DINOPretrainer(
                        base_model=base_model,
                        feature_dim=best_params.get('num_filters', 64),
                        projection_dim=best_params.get('proj_dim', 128),
                        projection_hidden=best_params.get('num_filters', 64),
                        momentum=0.996,
                        temperature_student=0.1,
                        temperature_teacher=0.04,
                        center_momentum=0.9,
                        global_crops=2,
                        local_crops=6
                    )
            elif pretraining_method.upper() == "RELCON":
                model = RelConPretrainer(
                    base_model=base_model,
                    feature_dim=best_params.get("num_filters", 64),
                    projection_dim=128,
                    projection_hidden=256,
                    temperature=0.1,
                    lambda_temporal=5.0,
                    num_candidates=8,
                    within_subject_ratio=0.5
                )
            elif pretraining_method.upper() == "SIMCLR":
                if use_dataparallel:
                    # Use DataParallel SimCLR pretrainer
                    model = create_dataparallel_pretrainer(
                        pretrainer_type='simclr',
                        base_model=base_model,
                        device_ids=device_ids,
                        feature_dim=best_params.get('num_filters', 64),
                        projection_dim=best_params.get('proj_dim', 128),
                        projection_hidden=best_params.get('num_filters', 64),
                        temperature=0.1,
                        num_views=2
                    )
                else:
                    # Use single-GPU SimCLR pretrainer
                    model = SimCLRPretrainer(
                        base_model=base_model,
                        projection_dim=best_params.get('proj_dim', 128),
                        projection_hidden=best_params.get('num_filters', 64),
                        temperature=0.1,
                        num_views=2
                    )
            else:
                raise ValueError(f"Unknown pretraining method: {pretraining_method}")
            
            if not use_dataparallel:
                model = model.to(device)

            # Print GPU configuration info
            if use_dataparallel and device_ids:
                print(f"Using DataParallel with GPUs: {device_ids}")
                print(f"Effective batch size: {batch_size} (split across {len(device_ids)} GPUs)")
            else:
                print(f"Using single GPU: {device}")
                print(f"Batch size: {batch_size}")

            total_params = 0
            for _, param in model.named_parameters():
                total_params += param.numel()
            print(f"Total number of parameters: {total_params}")
            

            # Setup optimizer
            lr = best_params['lr']
            optimizer_name = best_params['optim']
            optimizer = {
                'AdamW': optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4, betas=(0.9, 0.95)),
                'SGD': optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9),
                'RMSprop': optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=lr),
                'RAdam': optim.RAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr),
            }[optimizer_name]

            
            # Setup scheduler - use ALL data for step calculation
            print(f"Number of steps per epoch (using ALL data): {nb_steps}")
            
            # Use more conservative learning rate scheduling for stability
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr * 0.1,  # Reduce max learning rate for stability
                total_steps=nb_steps * args.epochs,
                anneal_strategy='cos',
                pct_start=0.1,  # Longer warmup for stability
                cycle_momentum=True,
                base_momentum=0.85,
                max_momentum=0.95,
                div_factor=10.0,  # More conservative div_factor
                final_div_factor=1e3,  # Less aggressive final reduction
            )

            # Training metrics tracking
            train_losses = []
            val_losses = []
            
            # No early stopping needed for pretraining - let it run full epochs
            early_stopping = EarlyStopping(patience=10, verbose=True)
            torch.cuda.empty_cache()
            
            # Training loop
            for epoch in range(args.epochs):
                # Train on ALL available data
                model.train()
                
                # Select training function based on pretraining method
                if pretraining_method.upper() == "AIM":
                    train_fn = mae_pretrain
                elif pretraining_method.upper() in ["DINO", "SIMCLR", "RELCON"]:
                    train_fn = contrastive_pretrain
                else:
                    train_fn = mae_pretrain  # Default fallback
                
                # Use ALL data for training - no validation split needed for pretraining
                model, train_metrics = train_fn(
                    model, df_train, batch_size, True, device, optimizer, scheduler,
                    stratify=stratify,
                    max_seq_len=max_seq_len,
                    use_TSO=True,
                )
                
                # Record metrics
                train_losses.append(train_metrics['loss'])
                
                print(f"Epoch [{epoch+1}/{args.epochs}]: Time: {datetime.now()}")
                print(f"Pretraining Loss: {train_metrics['loss']} | Steps: {train_metrics['steps']}")
                
                # Optional: Compute validation loss occasionally for monitoring (but don't use for early stopping)

                with torch.no_grad():
                    model.eval()
                    _, val_metrics = train_fn(
                        model, df_val, batch_size, False, device, optimizer, scheduler,
                        stratify=False,
                        max_seq_len=max_seq_len,
                        use_TSO=True,
                    )
                    val_losses.append(val_metrics['loss'])
                    m=float(val_metrics['loss'])
                    if early_stopping(m, model):
                        print(f"Early stopping at epoch {epoch+1}")
                        early_stopping.restore(model)
                        break 
                print("-----------------------")
                print(f'Iteration {i}. Epoch [{epoch+1}/{epochs}]: Time: {datetime.now()} \n   training: {train_metrics} ,\n   validation:{val_metrics}')
            # Save learning curves
            learning_metrics = {
                "train_losses": train_losses,
                "val_losses": val_losses,
            }
            
            # Save training metrics
            pd.DataFrame(learning_metrics).to_csv(os.path.join(folder_paths["training_logs_folder"],f"training{i}_losses_accuracies_test_subject_{test_subject_id}.csv"), encoding="utf-8",index=False)
            
            retrain.check_performance(min(val_losses),model)

            #Get bets model if trained more than once
            model=retrain.restore(model)
            epochs = range(1, len(train_losses) + 1)

            # Create subplots
            fig, ax = plt.subplots()

            # Plot training loss only (no validation in pretraining)
            ax.plot(epochs, np.array(train_losses, dtype=float), 'b-', label='Training loss')
            ax.plot(epochs, np.array(val_losses, dtype=float), 'r-', label='Validation loss')
            ax.set_title('Pretraining Loss')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            ax.legend()
            curve_path = os.path.join(learning_plots_output_folder, 
                                            f"training{i}_losses_accuracies_test_subject_{test_subject_id}.jpg")
            plt.tight_layout()
            plt.savefig(curve_path, dpi=300)
            plt.close()
            
            
            # Test model
            with torch.no_grad():
                model.eval()
                    
                _, test_metrics = train_fn(
                    model, df_val, batch_size, False, device, optimizer, scheduler,
                    stratify=False,
                    max_seq_len=max_seq_len,
                    use_TSO=True,
                )
            
            print(f"Testing results for {test_subject_id}: {test_metrics}")
            
            # Save model
            torch.save(
                model.state_dict(),
                os.path.join(train_models_folder, f"{args.model}_{pretraining_method}_test_subject_{test_subject_id}_weights.pth")
            )
            
            # Save predictions and metrics
            pd.DataFrame(pd.json_normalize(test_metrics)).to_csv(
                os.path.join(predictions_output_folder, f"final_metrics_sheet_test_subject_{test_subject_id}.csv"),
                encoding="utf-8", index=False
            )

            custom_print("Job finished successfully...")
            custom_print(f"Finished Training Now: {datetime.now()}")
    # break  # todo: this experiment only runs 1 subject split test (US10008003)
    # mlflow.end_run()
    return 1


if __name__ == "__main__":
    output = train_pipeline(args=args)