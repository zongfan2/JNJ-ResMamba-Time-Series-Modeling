import os
import shutil
import csv
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import pytz
# import matplotlib.pyplot as plt
from random import shuffle
import argparse
import logging
from Helpers.DL_models import *
from Helpers.DL_helpers import *
from Helpers.classical_backend import ClassicalBackend, is_classical_arch
script_dir = os.path.dirname(os.path.abspath(__file__))


def print_status(message,level,logger):
    time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    l='-'*level*2
    logger.info(f"{time} {l}> {message}")

def create_logger(name,location):
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create handlers
    console_handler = logging.StreamHandler()  # Console output
    file_handler = logging.FileHandler(location)  # Log to file

    # Set level for handlers
    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

def create_folder(list_of_folder_paths):
    for folder_path in list_of_folder_paths:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
def add_change_point(s,m,binary=False):
#     s['change']=0
    change = np.zeros(len(s), dtype=int)
    try:
        a=s[['x','y','z']].reset_index(drop=True)
        algo=rpt.Pelt(model='rbf').fit(a)
        results=algo.predict(pen=10)
        results = [row -1 for row in results[:-1]]
        change[results] = 1
        change= np.cumsum(change)
        if binary:
            change= (change %2!=0).astype(int) 
    except Exception as e:
        print("Error:", e)
 
    return pd.Series(change, index=s.index)

def add_padding_prod(batch,device,seg_column='segment'):
    X_sequences=[]
    Y_sequences=[]
    x_lens=[]
    for index,seq in batch.groupby(seg_column, sort=False):
        xyz=seq
#         xyz_p=np.random.permutation(['x','y'])#,'z'
#         xyz=xyz.rename(columns={'x': xyz_p[0], 'y': xyz_p[1]})#, 'z': xyz_p[2]
        X_arr=seq.loc[:, ['x', 'y', 'z']].to_numpy()
        X_sequences.append(torch.tensor(X_arr,dtype=torch.float32,device=device))#,device=device
        x_lens.append(len(X_arr))

    pad_X=pad_sequence(X_sequences, batch_first=True) #,padding_value=-999

    return pad_X,x_lens

def run_model(model,df,batch_size,device,stratify=False,verbose=True):

    # Classical baselines (mahadevan / ji / mdpi*) are not torch modules and
    # don't go through the batched-padded DL inference loop below.  Dispatch
    # to the ClassicalBackend wrapper, which returns the same per-frame
    # prediction DataFrame schema expected by NS-pipeline's downstream merge.
    if isinstance(model, ClassicalBackend):
        return model.predict(df)

    epoch_loss = batches  = 0.0
    pr1s = pr1_probs =pr2s= pr2_probs = pr3s =[]
    loss1_s=[]
    loss2_s=[]
    loss3_s=[]
    seg_column='segment'
    segments1=segments2=positions=[]
#     for batch in batch_generator(df=df,batch_size=batch_size,stratify=train_mode,shuffle=True,seg_column=seg_column):#stratify=train_mode
    for batch in batch_generator(df=df,batch_size=batch_size,stratify=stratify,shuffle=False,seg_column=seg_column):    
        
        batch_data,x_lens=add_padding_prod(batch,device,seg_column) #TCN doesn't work with pytorch pack_pad.
        outputs1,outputs2,outputs3= model(batch_data,x_lens)
        outputs1 = outputs1.view(-1)  
        outputs3 = outputs3.view(-1)
        
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


        pr1s=np.concatenate((pr1s,pr1.cpu().detach().numpy()))
        pr1_probs=np.concatenate((pr1_probs,pr1_prob.cpu().detach().numpy()))


        pr2s=np.concatenate((pr2s,pr2))
        pr2_probs=np.concatenate((pr2_probs,pr2_prob.cpu().detach().numpy()))
        pr3s=np.concatenate((pr3s,pr3.cpu().detach().numpy()))

        segments1=np.concatenate((segments1,batch[seg_column].unique()))
        segments2=np.concatenate((segments2,batch[seg_column].values))
        positions=np.concatenate((positions,batch.position.values))
    
    pr1s=np.nan_to_num(pr1s, nan=0, posinf=0, neginf=0)
    pr2s=np.nan_to_num(pr2s, nan=0, posinf=0, neginf=0)
    pr3s=np.nan_to_num(pr3s, nan=0, posinf=0, neginf=0)
    
    
    predictions1=pd.DataFrame({"segment":segments1,
                               "pr1": pr1s,
                               "pr1_probs": pr1_probs,
                               "pr3": pr3s})

    predictions2=pd.DataFrame({"segment":segments2,
                               "position":positions, 
                               "pr2": pr2s,
                               "pr2_probs": pr2_probs})

    predictions=predictions2.merge(predictions1)

                
    return predictions

def load_optuna_pretrained_best_params(optuna_best_param_path):
    best_param_file_path = os.path.join(optuna_best_param_path, "optuna_trial_best_params.txt")
    if os.path.exists(best_param_file_path):
        return read_param(best_param_file_path)
    else:
        params_list={'param_mba_v1':{'batch_size': 32, 'num_filters': 128, 'dropout': 0.5, 'droppath': 0.3, 'kernel_f': 13, 'kernel_MBA1': 9, 'num_feature_layers': 9, 'blocks_MBA1': 8, 'wl1': 0.5, 'wl2': 0.9, 'wl3': 0.5, 'optim': 'RMSprop', 'pos_weight_l1': False, 'pos_weight_l2': True, 'featurelayer': 'ResNet', 'norm1': 'BN', 'norm2': 'IN', 'norm3': 'BN','lr': 0.001,'channel_masking_rate': 0,'cls_token': True,'stratify': 'undersample' },
                    } 

        return params_list['param_mba_v1']


def get_scratch_model(args):
    model_name = args.scratch_model

    # ------------------------------------------------------------------
    # Classical baselines (Mahadevan 2021 / Ji 2023 / MDPI 2024) — packaged
    # as a single joblib bundle from training/train_classical.py.  The
    # presence of ``<model_name>_weights.joblib`` under ./model selects
    # this path; the file's ``arch`` field still has to be one of the
    # recognised CLASSICAL_ARCHS for ClassicalBackend to load it.
    # ------------------------------------------------------------------
    classical_bundle = os.path.join(
        os.path.dirname(script_dir), 'model', f'{model_name}_weights.joblib'
    )
    if os.path.exists(classical_bundle):
        backend = ClassicalBackend(classical_bundle)
        # device/batch_size returned for API parity with the DL path; both
        # are unused by run_model when ``model`` is a ClassicalBackend.
        return backend, torch.device("cpu"), None

    # ------------------------------------------------------------------
    # DL baseline (ResMamba family) — the original path.
    # ------------------------------------------------------------------
    best_params = load_optuna_pretrained_best_params(optuna_best_param_path='model/')  # check params
    max_seq_len=1221
    pretraining=False
    batch_size=best_params['batch_size']
    device = torch.device(f"cuda:{args.num_gpu}" if torch.cuda.is_available() else "cpu")
    model = setup_model(
                    model_name=model_name,
                    input_tensor_size=3, #42 9
                    max_seq_len=max_seq_len, #df.position_segment.max()+1
                    best_params=best_params,
                    pretraining=pretraining,
                    num_classes=1
                )
    model = model.to(device)

    model_path = os.path.join(os.path.dirname(script_dir), 'model', f'{model_name}_weights.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model,device,batch_size

