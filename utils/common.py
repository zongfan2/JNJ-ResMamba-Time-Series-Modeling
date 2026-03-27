# -*- coding: utf-8 -*-
"""
Split from DL_helpers.py - Modular code structure
@author: MBoukhec (original)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import logging

class Retraining:
    def __init__(self, verbose=False):
        """
        Early stopping to halt training when validation loss does not improve.
        
        Parameters:
        patience (int): How many epochs to wait after the last improvement before stopping.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        verbose (bool): Whether to print messages when the validation loss improves or not.
        restore_best_weights (bool): Whether to restore model weights from the epoch with the best validation loss.
        """
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_weights = None
        self.best_index = None
    
    def check_performance(self, val_loss, model):
        # If this is the first validation loss, initialize best_score
        self.counter += 1
        if self.best_score is None:
            self.best_score = val_loss
            self.best_weights = model.state_dict()
            self.best_index = self.counter
        else:
            # If the validation loss improved, reset counter and update best score
            if val_loss < self.best_score :
                self.best_score = val_loss
                self.best_weights = model.state_dict()
                self.best_index = self.counter 
                if self.verbose:
                    print(f"Validation loss improved. Best model is from training {self.best_index}")
            else:
                # If no improvement, increment the counter
                if self.verbose:
                    print(f"Validation loss did not improve. Best model is from training {self.best_index}")   
    
    def restore(self, model):
        """Restore model weights from the best epoch."""
        if self.verbose:
            print(f"Best model restored from training {self.best_index}")
        model.load_state_dict(self.best_weights)
        return model
    

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
            
############################################################            
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="3"
from torch.nn.parallel import DistributedDataParallel
from sklearn import metrics
# from torcheval.metrics.functional import multiclass_f1_score,binary_f1_score
from datetime import datetime
from torch.nn.utils.rnn import pad_sequence
from sklearn.utils.class_weight import compute_class_weight
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from skimage.measure import block_reduce
from sklearn.preprocessing import StandardScaler

torch.cuda.empty_cache()


def create_mask(original_lengths, max_length, batch_size, device):
    # same as DL_models
    mask = torch.arange(max_length,device=device).unsqueeze(0).expand(batch_size, -1)
    mask = (mask < original_lengths.unsqueeze(1)).long()
    return mask



def split_and_pad(x, x_lengths, max_seq_len, pad_value=0):
    """
    Split the x into sub-sequences based on lengths and pad them to max_seq_len.
    
    Args:
        x (torch.Tensor): Input vector of shape (N,).
        x_lengths (list): List of sub-sequence lengths [n1, n2, ...].
        max_seq_len (int): The fixed length to pad each sub-sequence.
        pad_value (int, optional): Value to pad shorter sub-sequences. Defaults to 0.
    
    Returns:
        torch.Tensor: Padded sub-sequences matrix of shape (len(x_lengths), max_seq_len).
    """

    start_idx = 0  # Keep track of the slicing index    
    xs = torch.split(x, x_lengths)
    pad_xs = []
    for cur_x in xs:
        pad_cur_x = torch.cat([cur_x, torch.full((max_seq_len - len(cur_x),), pad_value).to(x.device)])
        pad_xs.append(pad_cur_x)
    return torch.stack(pad_xs)



def get_folder_size(folder_path):
    """Get total size of folder in GB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024**3)  # Convert to GB



