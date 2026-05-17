# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 08:15:39 2024

@author: MBoukhec
"""
import glob
import os
import random
import re
import shutil
import logging
import traceback
import math
from pprint import pprint
from datetime import datetime
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from mamba_ssm import Mamba


def setup_model(model_name, input_tensor_size,max_seq_len, best_params, pretraining,  num_classes=1):
    """
        This function is responsible for initializing the model class from best_params
        Return: model object to be trained
    """
#     print("Params Received in `setup_model`: ",
#           [model_name, input_tensor_size, best_params, pretraining, num_classes])
    model = None
    model_name = str(model_name).lower()
    match model_name:
        case "bimamba":
            model = BiMamba(
                    d_model=input_tensor_size,
                    n_state=64)
        case "mtcn":
            model = MTCN(
                    input_channels=input_tensor_size)
        case "mtcnp":
            model = MTCNP(
                    input_channels=input_tensor_size,
                max_seq_len=max_seq_len)
        case "mtcna2":
            model = MTCNA2(
                    input_channels=input_tensor_size,
                max_seq_len=max_seq_len)
        case "mba":
            model = MBA(
                    input_dim=input_tensor_size,
                max_seq_len=max_seq_len)
        case "mbanew":
            model = MBAnew(
                    input_dim=input_tensor_size,
                max_seq_len=max_seq_len)
        case "cnnautoencoder":
            num_filters = best_params['num_filters']
            model = CNNSeqAutoencoder(input_tensor_size, num_filters, 128, max_seq_len)
        case "mbav1":
            dropout_rate = best_params['dropout']
            drop_path_rate = best_params['droppath']
            kernel_size_encoder =  best_params["kernel_f"]
            kernel_size_decoder = best_params["kernel_MBA1"]          
            num_decoder_layers = best_params["blocks_MBA1"]
            num_encoder_layers =best_params["num_feature_layers"]
            cls_token=best_params["cls_token"]
            featurelayer=best_params["featurelayer"]
            num_filters = best_params['num_filters']
            norm1=best_params['norm1']
            norm2=best_params['norm2']
            norm3=best_params['norm3']
            
            model = MBA_v1(input_tensor_size,num_encoder_layers=num_encoder_layers,num_decoder_layers=num_decoder_layers,drop_path_rate =drop_path_rate ,kernel_size_encoder=kernel_size_encoder,kernel_size_decoder=kernel_size_decoder,dropout_rate=dropout_rate, max_seq_len= max_seq_len,num_filters=num_filters,norm1=norm1,norm2=norm2,norm3=norm3)

            
        case "resnetunet":
            model = ResNetUNet(in_channels=input_tensor_size)
            
        case "mbatsm_old":
            dropout_rate = best_params['dropout']
            drop_path_rate = best_params['droppath']
            kernel_size_feature =  best_params["kernel_f"]
            kernel_size_mba = best_params["kernel_MBA"]
            num_MBA_blocks = best_params["blocks_MBA"]
            num_feature_layers =best_params["num_feature_layers"]
            cls_token=best_params["cls_token"]
            MBA_encoder=best_params["MBA_encoder"]
            num_heads=best_params["num_heads"]
            featurelayer=best_params["featurelayer"]
            num_filters = best_params['num_filters']
            model = MBA_tsm_old(input_tensor_size,num_feature_layers=num_feature_layers,num_encoder_layers=num_MBA_blocks,drop_path_rate =drop_path_rate ,kernel_size_feature=kernel_size_feature,kernel_size_mba=kernel_size_mba,dropout_rate=dropout_rate, max_seq_len= max_seq_len,cls_token=cls_token,mba=MBA_encoder,num_heads=num_heads,featurelayer=featurelayer,num_filters=num_filters,pretraining=pretraining)
        
        case "mbatsm_ed":
            dropout_rate = best_params['dropout']
            drop_path_rate = best_params['droppath']
            kernel_size_feature =  best_params["kernel_f"]
            kernel_size_mba = best_params["kernel_MBA"]

            kernel_size_mba1 = best_params["kernel_MBA"]
            kernel_size_mba2 = best_params["kernel_MBA"]
        #     num_feature_layers = trial.suggest_int("num_feature_layers", 1, 10, step=1)
            
            num_MBA_blocks = best_params["blocks_MBA"]
            num_MBA_blocks1 = best_params["blocks_MBA"]
            num_MBA_blocks2 = best_params["blocks_MBA"]
            dilation1=best_params["dilation1"]
            dilation2=best_params["dilation2"]
            
            num_feature_layers =best_params["num_feature_layers"]
            cls_token=best_params["cls_token"]
            MBA_encoder=best_params["MBA_encoder"]
            num_heads=best_params["num_heads"]
            featurelayer=best_params["featurelayer"]
            num_filters = best_params['num_filters']
            encoder_bottelneck = best_params['encoder_bottelneck']
            norm=best_params['norm']
            
            
            model = MBA_tsm_ED(input_tensor_size,num_feature_layers=num_feature_layers,num_encoder_layers=num_MBA_blocks,drop_path_rate =drop_path_rate ,kernel_size_feature=kernel_size_feature,kernel_size_mba=kernel_size_mba,dropout_rate=dropout_rate, max_seq_len= max_seq_len,cls_token=cls_token,mba=MBA_encoder,num_heads=num_heads,featurelayer=featurelayer,num_filters=num_filters,pretraining=pretraining,encoder_bottelneck=encoder_bottelneck)
            
            
        case "mba_ed":
            dropout_rate = best_params['dropout']
            drop_path_rate = best_params['droppath']
            kernel_size_feature =  best_params["kernel_f"]
            kernel_size_mba = best_params["kernel_MBA"]
            num_MBA_blocks = best_params["blocks_MBA"]
            num_feature_layers =best_params["num_feature_layers"]
            cls_token=best_params["cls_token"]
            MBA_encoder=best_params["MBA_encoder"]
            num_heads=best_params["num_heads"]
            featurelayer=best_params["featurelayer"]
            num_filters = best_params['num_filters']
            model = MBA_encoder_decoder(input_tensor_size,num_feature_layers=1,num_encoder_layers=num_MBA_blocks,drop_path_rate =drop_path_rate ,kernel_size_feature=kernel_size_feature,kernel_size_mba=kernel_size_mba,dropout_rate=dropout_rate, max_seq_len= max_seq_len,cls_token=cls_token,mba=MBA_encoder,num_heads=num_heads,featurelayer=featurelayer,num_filters=num_filters,pretraining=pretraining)
            
        case "mba_encoder_decoder":
            dropout_rate = best_params['dropout']
            drop_path_rate = best_params['droppath']
            kernel_size_mba = best_params["kernel_MBA"]
            num_MBA_blocks = best_params["blocks_MBA"]
            cls_token=best_params["cls_token"]
            MBA_encoder=best_params["MBA_encoder"]
            num_heads=best_params["num_heads"]
            num_filters = best_params['num_filters']
            average_mask = best_params.get("average_mask", False)
            average_window_size = best_params.get("average_window_size", 20)
            supcon_loss = best_params.get("supcon_loss", True)
            cls_src = best_params.get("cls_src", "encoder")

            # Split layers between encoder and decoder  
            num_encoder_layers = max(1, num_MBA_blocks // 2)
            if num_MBA_blocks / 2 > num_encoder_layers:
                num_encoder_layers += 1
            num_decoder_layers = max(1, num_MBA_blocks - num_encoder_layers)

            net_type = best_params.get("net_type", "normal")
            if net_type == "progressive_with_skip_connection":
                net = MBA_tsm_encoder_decoder_progressive_with_skip_connection
            elif net_type == "progressive":
                net = MBA_tsm_encoder_decoder_progressive
            elif net_type == "normal":
                net = MBA_tsm_encoder_decoder

            model = net(input_tensor_size,
                            num_encoder_layers=num_encoder_layers,
                            num_decoder_layers=num_decoder_layers,
                            drop_path_rate =drop_path_rate,
                            kernel_size_mba=kernel_size_mba,
                            dropout_rate=dropout_rate, 
                            max_seq_len= max_seq_len,
                            cls_token=cls_token,
                            mba=MBA_encoder,
                            num_heads=num_heads,
                            num_filters=num_filters,
                            average_mask=average_mask,
                            average_window_size=average_window_size,
                            supcon_loss=supcon_loss,
                            cls_src=cls_src)
        case "hybrid":
            model = hybrid(
                    input_dim=input_tensor_size)
        case "bitcnt":
            model = BiTCNT(
                    input_dim=input_tensor_size)

        case "mba4tso_patch":
            dropout_rate = best_params.get('dropout', 0.2)
            drop_path_rate = best_params.get('droppath', 0.3)
            kernel_size_feature = best_params.get("kernel_f", 3)
            kernel_size_mba = best_params.get("kernel_MBA", 7)
            num_encoder_layers = best_params.get("blocks_MBA", 3)
            num_feature_layers = best_params.get("num_feature_layers", 3)
            featurelayer = best_params.get("featurelayer", "TCN")
            num_filters = best_params.get('num_filters', 64)
            patch_size = best_params.get("patch_size", 1200)
            patch_channels = best_params.get("patch_channels", 5)
            norm1 = best_params.get("norm1", "BN")
            norm2 = best_params.get("norm2", "IN")
            output_channels = best_params.get("output_channels", 3)
            model = MBA4TSO_Patch(
                patch_size=patch_size,
                patch_channels=patch_channels,
                num_filters=num_filters,
                num_feature_layers=num_feature_layers,
                num_encoder_layers=num_encoder_layers,
                drop_path_rate=drop_path_rate,
                kernel_size_feature=kernel_size_feature,
                kernel_size_mba=kernel_size_mba,
                dropout_rate=dropout_rate,
                max_seq_len=max_seq_len,
                featurelayer=featurelayer,
                norm1=norm1,
                norm2=norm2,
                output_channels=output_channels
            )

        case model_name:
            raise Exception(f"Model name not matched in setup_model: {model_name}")

    return model


# Define RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate, num_classes=2):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.rnn = nn.RNN(input_size, hidden_sizes[-1], num_layers=self.num_layers,
                          batch_first=True, dropout=dropout_rate if self.num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_sizes[-1]).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

    # Define RNN model
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate, num_classes=2):
        super(BiRNN, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.rnn = nn.RNN(input_size, hidden_sizes[-1], num_layers=self.num_layers,batch_first=True, dropout=dropout_rate if self.num_layers > 1 else 0,bidirectional=True)
        self.lstm = nn.LSTM(input_size, hidden_sizes[-1], num_layers=self.num_layers,batch_first=True, dropout=dropout_rate if self.num_layers > 1 else 0,bidirectional=True)
        self.fc = nn.Linear(hidden_sizes[-1]*2, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_sizes[-1]).to(x.device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_sizes[-1]).to(x.device)
        out, (h_n, c_n) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

 


class CNN(nn.Module):
    def __init__(self, input_dim, num_filters, kernel_size,dropout_rate):
        super(CNN, self).__init__()
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=kernel_size)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters * 2, kernel_size=kernel_size)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(in_channels=num_filters* 2, out_channels=num_filters * 4, kernel_size=kernel_size)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv1d(in_channels=num_filters * 4, out_channels=num_filters * 4, kernel_size=kernel_size)
        self.relu4 = nn.ReLU()

        # Calculate the size after convolutions
        conv_output_size = self.calculate_conv_output_size(input_dim, kernel_size, 3) # 3 conv layers

        # Adjust the fully connected layer
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate)
        self.batchnorm = nn.BatchNorm1d(num_filters * 4 * conv_output_size)
        self.fc1 = nn.Linear(num_filters * 4 * conv_output_size, 1) # Adjusted size
        self.sigmoid = nn.Sigmoid()
        
    def calculate_conv_output_size(self, input_size, kernel_size, num_convs):
        output_size = input_size
        for _ in range(num_convs):
            output_size = output_size - kernel_size + 1
        return output_size

    def forward(self, x):
        # Forward pass
        x = x.reshape(x.shape[0], 1, -1)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
#         x = self.relu4(self.conv4(x))
        x = self.flatten(x)
        #output = self.sigmoid(self.fc1(x))
        x = self.dropout(x)
        output = self.fc1(x)
        return output


import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a residual block
class ResidualBlock(nn.Module):
    def __init__(self, num_filters, kernel_size, dropout_rate):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(num_filters, num_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(num_filters)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out += x  # Residual connection
        return F.relu(out)


# Define ResCNN model
class ResCNN_old(nn.Module):
    def __init__(self, input_dim, num_filters, kernel_size, dropout_rate):
        super(ResCNN, self).__init__()
        self.initial_conv = nn.Conv1d(1, num_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.initial_bn = nn.BatchNorm1d(num_filters)
        self.block1 = ResidualBlock(num_filters, kernel_size, dropout_rate)
        self.block2 = ResidualBlock(num_filters, kernel_size, dropout_rate)
        self.block3 = ResidualBlock(num_filters, kernel_size, dropout_rate)
        self.block4 = ResidualBlock(num_filters, kernel_size, dropout_rate)
        self.fc = nn.Linear(input_dim * num_filters, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        out = self.initial_conv(x)
        out = self.initial_bn(out)
        out = F.relu(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = out.view(out.size(0), -1)  # Flatten
        return self.fc(out)

# Define MLP model
class MLP(nn.Module):
    def __init__(self, input_tensor_size, hidden_layer_sizes, dropout_rate):
        super(MLP, self).__init__()
        layers = []
        prev_size = input_tensor_size
        for size in hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = size
        # layers.append(nn.BatchNorm1d(hidden_layer_sizes[-1]))
        layers.append(nn.Linear(hidden_layer_sizes[-1], 1))  # Output for two classes
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)
# Define ResCNN model
class ResCNN(nn.Module):
    def __init__(self, input_dim, num_filters, kernel_size, dropout_rate, num_blocks):
        super(ResCNN, self).__init__()
        self.initial_conv = nn.Conv1d(1, num_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.initial_bn = nn.BatchNorm1d(num_filters)
        self.blocks=nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append( ResidualBlock(num_filters, kernel_size, dropout_rate))
        self.fc = nn.Linear(input_dim * num_filters, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        out = self.initial_conv(x)
        out = self.initial_bn(out)
        out = F.relu(out)
        for block in self.blocks:
            out = block(out)
        out = out.view(out.size(0), -1)  # Flatten
        return self.fc(out)

    



# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.num_layers = 2
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(input_size, hidden_size,num_layers=self.num_layers, batch_first=True,dropout=0.2 if self.num_layers > 1 else 0,bidirectional=True)
        #self.fc = nn.Linear(hidden_size*2, output_size)
        self.fc = nn.Sequential(nn.Linear(hidden_size*2, 1024),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.2),
                                nn.Linear(1024, 128),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.2),
                                nn.Linear(128, output_size)
                               )

    def forward(self, x, x_lengths):
        x = pack_padded_sequence(x, x_lengths,batch_first=True, enforce_sorted=False)
        lstm_out, (hn, cn) = self.lstm(x)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        out = self.fc(lstm_out)
        return out

class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate, output_size=1):
        super(BiRNN, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.rnn = nn.RNN(input_size, hidden_sizes[-1], num_layers=self.num_layers,batch_first=True, dropout=dropout_rate if self.num_layers > 1 else 0,bidirectional=True)
        self.lstm = nn.LSTM(input_size, hidden_sizes[-1], num_layers=self.num_layers,batch_first=True, dropout=dropout_rate if self.num_layers > 1 else 0,bidirectional=True)
        self.fc = nn.Linear(hidden_sizes[-1]*2, output_size)

    def forward(self, x, x_lengths):
        #x = x.unsqueeze(1)
        d=x.device
        x = pack_padded_sequence(x, x_lengths,batch_first=True, enforce_sorted=False)
        h0 = torch.zeros(self.num_layers*2, x.data.size(0), self.hidden_sizes[-1]).to(d)#x.size(0)
        c0 = torch.zeros(self.num_layers*2, x.data.size(0), self.hidden_sizes[-1]).to(d)#x.size(0)
        out, (h_n, c_n) = self.lstm(x, (h0, c0))
        out, _ = pad_packed_sequence(out, batch_first=True)
        out = self.fc(out)
        return out
################# TCN    

    


# Define a Bidirectional Temporal Convolutional Network (Bi-TCN) block
class BiTCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(BiTCNLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Forward pass for both directions
        x1 = self.relu(self.conv1(x))  # Forward pass
        x2 = self.relu(self.conv2(torch.flip(x, dims=[-1])))  # Backward pass
        
        # Combine the forward and backward features (concatenate)
        x = torch.cat([x1, torch.flip(x2, dims=[-1])], dim=1)
        x = self.dropout(x)
        return x

class MTCN(nn.Module): #Multi Task TCN
    def __init__(self, input_channels, num_classes=1, num_layers=5, num_filters=64, kernel_size=3, max_dilation=16, dropout=0.2):
        super(MTCN, self).__init__()

        layers = []
        dilation = 1
        in_channels = input_channels
        
        for i in range(num_layers):
            out_channels = num_filters
            padding = (kernel_size - 1) * dilation // 2  # To maintain same length
            #layers.append(TCNLayer(in_channels, out_channels, kernel_size, dilation, padding))
            layers.append(TCNLayer(num_filters, out_channels, kernel_size, dilation, padding))
            in_channels = out_channels
            dilation *= 2  # Increase dilation exponentially
        
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_channels, num_filters, kernel_size=7, padding=3),  # Initial Conv
            nn.ReLU(),
            nn.BatchNorm1d(num_filters),
            nn.Dropout(dropout),
        )
        self.tcn = nn.Sequential(*layers)
        self.fc1 = nn.Linear(num_filters, 1)
        self.fc2 = nn.Conv1d(num_filters, num_classes, kernel_size=1)  # Output the class probabilities for each time step
        self.fr_duration = nn.Linear(num_filters, 1)
        
    def forward(self, x, x_lengths):
        # x: [batch_size, num_channels, sequence_length]
        x=x.permute(0, 2, 1)
        x = self.feature_extractor(x)
        x = self.tcn(x)
        x1 = self.fc1(x.mean(dim=-1))
        x2 = self.fc2(x)
        x3= self.fr_duration(x.mean(dim=-1))
        
        x2 = x2.permute(0, 2, 1) # change shape to batch,sequence,output
        
        #Ensures that if x1 is zero, x2 and x3 will be set to zero as well. This approach doesn't require modifying the loss function, but it directly imposes the condition in the model's behavior.
#         mask_x1 =torch.round(torch.sigmoid(x1))
#         x3 = x3 * (mask_x1 != 0).float() 
#         x2 = x2 * (mask_x1.unsqueeze(-1).expand(-1, x2.shape[1],-1) != 0).float() # expand the mask_x1 to have similar shape as x2 (from batch,output to batch,sequence,output)
        return x1,x2,x3

class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, embedding_dim):
        """
        Positional Embedding for sequences of varying lengths.
        
        Args:
            max_seq_len (int): Maximum sequence length.
            embedding_dim (int): The dimensionality of the positional embeddings.
        """
        super(PositionalEmbedding, self).__init__()
        
        # Create learnable positional embeddings of shape (max_seq_len, embedding_dim)
        self.positional_embeddings = nn.Embedding(max_seq_len, embedding_dim)

    def forward(self, seq_len, device):
        """
        Get positional embeddings for a given sequence length.
        
        Args:
            seq_len (int): The actual length of the input sequence.
            device (torch.device): The device on which the embeddings should be created.
        
        Returns:
            Tensor: The positional embeddings for the sequence.
        """
        # Generate positions (0, 1, 2, ..., seq_len-1)
        positions = torch.arange(0, seq_len, dtype=torch.long, device=device)
        # Look up the positional embeddings for the given positions
        return self.positional_embeddings(positions)  # (seq_len, embedding_dim)

class MTCNA2(nn.Module): #Multi Task TCN
    def __init__(self, input_channels,max_seq_len, embedding_dim=8,embed_dim=128,  num_classes=1, num_layers=5, num_filters=64, kernel_size=3, max_dilation=16, dropout=0.2,BI=False):
        super(MTCNA2, self).__init__()

        layers = []
        dilation = 1
        in_channels = input_channels
        self.pos_embedding = PositionalEmbedding(max_seq_len, embedding_dim)
        self.embedding = nn.Linear(input_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.GatedAttentionMILPooling =GatedAttentionPoolingMIL(input_dim=num_filters, hidden_dim=16)
        in_size=embed_dim + embedding_dim
        for i in range(num_layers):
            padding = (kernel_size - 1) * dilation // 2  # To maintain same length
            layers.append(TCNLayer(in_size,num_filters, kernel_size, dilation, padding))
            in_size=num_filters
            dilation *= 2  # Increase dilation exponentially
        
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_channels, num_filters, kernel_size=7, padding=3),  # Initial Conv
            nn.SELU(),
            nn.BatchNorm1d(num_filters),
            nn.Dropout(dropout),
        )
        self.tcn = nn.Sequential(*layers)
        self.out1 = nn.Linear(num_filters, 1)
        self.out2 = nn.Conv1d(num_filters, num_classes, kernel_size=1)  # Output the class probabilities for each time step
        self.out3 = nn.Linear(num_filters, 1)
        
    def forward(self, x, x_lengths): 
        batch_size, seq_len, _ = x.size()
        x=self.embedding(x)
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # Shape: (batch_size, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # Shape: (batch_size, seq_length + 1, embed_dim)
        # Get positional embeddings for the current sequence length
        pos_embed = self.pos_embedding(seq_len+1, x.device)  # (seq_len, embedding_dim)
        # Expand positional embeddings to match the batch size
        pos_embed = pos_embed.unsqueeze(0).expand(batch_size, seq_len+1, -1)  # (batch_size, seq_len, embedding_dim)
        # Concatenate the positional embeddings with the original input
        x = torch.cat([x, pos_embed], dim=-1)  # (batch_size, seq_len, input_size + embedding_dim)
        x=x.permute(0, 2, 1) # Convert to (batch_size, input_dim, seq_len) for TCN processing
#         x = self.feature_extractor(x)
        x = self.tcn(x)
#         mask = create_mask(torch.tensor(x_lengths,device=x.device), seq_len+1,batch_size,x.device)
#         masked=masked_avg_pool(x.permute(0,2,1), mask)

#         cls_output = x[:, :, 0]  # [batch_size, num_filters]
#         x1 = self.fc1(masked)#self.fc1(cls_output)
#         x2 = self.fc2(x)
#         x3= self.fr_duration(masked)#self.fr_duration(cls_output)

        mask = create_mask(torch.tensor(x_lengths,device=x.device), seq_len+1,batch_size,x.device)
        masked= self.GatedAttentionMILPooling(x.permute(0, 2, 1), mask)
        x1 = self.out1(masked)
        x2 = self.out2(x) #use .permute(0, 2, 1) if we want to use conv1d in output
        x3= self.out3(masked)
        
        
        #if x1 is 0 then duration should be 0
#         mask_x1 =torch.round(torch.sigmoid(x1))
#         x3 = x3 * (mask_x1 != 0).float()
        
        x2=x2[:, :, 1:]# Output sequence excludes the classification token, we return from 1 onwards
        x2 = x2.permute(0, 2, 1) # change shape to batch,sequence,output
        return x1,x2,x3 # Output sequence excludes the classification token, we return from 1 onwards


########################## Bidirectional TCN
class BiMTCN(nn.Module): #Multi Task TCN
    def __init__(self, input_channels, num_classes=1, num_layers=5, num_filters=64, kernel_size=3, max_dilation=16, dropout=0.2):
        super(BiMTCN, self).__init__()

        layers = []
        dilation = 1
        in_channels = input_channels
        
        for i in range(num_layers):
            out_channels = num_filters
            padding = (kernel_size - 1) * dilation // 2  # To maintain same length
            #layers.append(TCNLayer(in_channels, out_channels, kernel_size, dilation, padding))
            layers.append(TCNLayer(num_filters, out_channels, kernel_size, dilation, padding))
            in_channels = out_channels
            dilation *= 2  # Increase dilation exponentially
        
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_channels, num_filters, kernel_size=7, padding=3),  # Initial Conv
            nn.ReLU(),
            nn.BatchNorm1d(num_filters),
            nn.Dropout(dropout),
        )
        self.forward_tcn = nn.Sequential(*layers)
        self.backward_tcn = nn.Sequential(*layers)
        self.fc1 = nn.Linear(num_filters*2, 1)
        self.fc2 = nn.Conv1d(num_filters*2, num_classes, kernel_size=1)  # Output the class probabilities for each time step
        self.fr_duration = nn.Linear(num_filters*2, 1)
        
    def forward(self, x, x_lengths):
        # x: [batch_size, num_channels, sequence_length]
        x=x.permute(0, 2, 1)
        x = self.feature_extractor(x)
        #x = self.tcn(x)
        forward_out = self.forward_tcn(x)
        # Reverse the input sequence for backward pass
        backward_out = self.backward_tcn(torch.flip(x, dims=[-1]))
        # Reverse the output of the backward pass to match the original sequence order
        backward_out = torch.flip(backward_out, dims=[-1])
        # Concatenate forward and backward outputs
        x = torch.cat([forward_out, backward_out], dim=1)
        x1 = self.fc1(x.mean(dim=-1))
        x2 = self.fc2(x)
        x3= self.fr_duration(x.mean(dim=-1))
        return x1,x2.permute(0, 2, 1),x3




from mamba_ssm import Mamba
class BiMambaEncoder(nn.Module):
    def __init__(self, d_model, n_state,BI):
        super(BiMambaEncoder, self).__init__()
        self.d_model = d_model   
        self.BI=BI
        self.mamba = Mamba(d_model=d_model, d_state=n_state,d_conv=12)
        self.dropout = nn.Dropout()
        # Norm and feed-forward network layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.feed_forward2 = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SELU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, x):
        x,mask=x[0],x[1]
        # Residual connection of the original input
        residual = x
        # Forward Mamba
        x_norm = self.norm1(x)
        mamba_out_forward = self.mamba(x_norm)
        
        # Backward Mamba
        if self.BI:
            x_flip = torch.flip(x_norm, dims=[1])  # Flip Sequence
            mamba_out_backward = self.mamba(x_flip)
            mamba_out_backward = torch.flip(mamba_out_backward, dims=[1])  # Flip back
            # Combining forward and backward
            mamba_out = mamba_out_forward + mamba_out_backward
            if mask is not None:
                mamba_out = mamba_out*mask.expand(mamba_out.shape[2], -1, -1).permute(1, 2, 0)
            mamba_out = self.norm2(mamba_out)
            ff_out = self.feed_forward(mamba_out)
        else:
            mamba_out = self.norm2(mamba_out_forward)
            ff_out = mamba_out_forward
            ff_out = self.feed_forward2(mamba_out)
            
        ff_out = self.dropout(ff_out)
        output = ff_out + residual
        return output

class GatedAttentionPoolingMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        Gated Attention Pooling for MIL.
        Args:
        - input_dim: The size of the feature dimension for each instance.
        - hidden_dim: The size of the hidden dimension for the attention mechanism.
        """
        super(GatedAttentionPoolingMIL, self).__init__()
        # Attention mechanism
        self.attention_fc = nn.Linear(input_dim, hidden_dim)  # Attention network
        self.attention_score_fc = nn.Linear(hidden_dim, 1)  # Output a scalar score for each instance

    def forward(self, bag_features,mask):
        """
        Forward pass through the gated attention pooling layer.
        
        Args:
        - bag_features: Tensor of shape (batch_size, bag_size, input_dim)
        
        Returns:
        - pooled_features: Tensor of shape (batch_size, input_dim)
        """
        # Compute attention scores for each instance
        attention_scores = self.attention_fc(bag_features)  # Shape: (batch_size, bag_size, hidden_dim)
        attention_scores = torch.tanh(attention_scores)  # Activation
        attention_scores = self.attention_score_fc(attention_scores)  # Shape: (batch_size, bag_size, 1)
        # Apply sigmoid to get the attention weights between 0 and 1
        attention_weights = torch.sigmoid(attention_scores)  # Shape: (batch_size, bag_size, 1)
        # Multiply attention weights with the bag features to scale the instances
        weighted_bag_features = bag_features * attention_weights  # Shape: (batch_size, bag_size, input_dim)
        # Aggregate the weighted features (e.g., summing or averaging)
        #pooled_features = weighted_bag_features.sum(dim=1)  # Summing along the bag_size dimension
        
        sum_x = (weighted_bag_features * mask.unsqueeze(-1)).sum(dim=1)  # Shape: (batch_size, num_filters)
        valid_count = mask.sum(dim=1)  # Shape: (batch_size)
        avg_x = sum_x / valid_count.unsqueeze(1)  # Normalize by number of valid elements
        return avg_x,attention_weights

class MultiHeadSelfAttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        """
        Multi-Head Self-Attention Pooling for MIL.
        Args:
        - input_dim: The size of the feature dimension for each instance.
        - hidden_dim: The size of the hidden dimension for the attention mechanism.
        - num_heads: The number of attention heads.
        """
        super(MultiHeadSelfAttentionPooling, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Linear transformations for self-attention
        self.W_Q = nn.Linear(input_dim, hidden_dim)  # Weights for Queries
        self.W_K = nn.Linear(input_dim, hidden_dim)  # Weights for Keys
        self.W_V = nn.Linear(input_dim, hidden_dim)  # Weights for Values
        
        self.output_layer = nn.Linear(hidden_dim, input_dim)  # Final linear layer to project back to input_dim

    def forward(self, bag_features, mask):
        """
        Forward pass through the multi-head self-attention pooling layer.
        
        Args:
        - bag_features: Tensor of shape (batch_size, bag_size, input_dim)
        - mask: Tensor with shape (batch_size, bag_size) to mask invalid instances
        
        Returns:
        - pooled_features: Tensor of shape (batch_size, input_dim)
        - attention_weights: Tensor of shape (batch_size, num_heads, bag_size, bag_size)
        """
        batch_size, bag_size, _ = bag_features.size()

        # Compute Queries, Keys, and Values
        Q = self.W_Q(bag_features).view(batch_size, bag_size, self.num_heads, self.head_dim)  # Shape: (batch_size, bag_size, num_heads, head_dim)
        K = self.W_K(bag_features).view(batch_size, bag_size, self.num_heads, self.head_dim)  # Shape: (batch_size, bag_size, num_heads, head_dim)
        V = self.W_V(bag_features).view(batch_size, bag_size, self.num_heads, self.head_dim)  # Shape: (batch_size, bag_size, num_heads, head_dim)

        # Transpose Q, K, V to (batch_size, num_heads, bag_size, head_dim)
        Q = Q.transpose(1, 2)  # Shape: (batch_size, num_heads, bag_size, head_dim)
        K = K.transpose(1, 2)  # Shape: (batch_size, num_heads, bag_size, head_dim)
        V = V.transpose(1, 2)  # Shape: (batch_size, num_heads, bag_size, head_dim)

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # Shape: (batch_size, num_heads, bag_size, bag_size)

        # Apply softmax to compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # Shape: (batch_size, num_heads, bag_size, bag_size)

        # Compute the weighted sum of values based on attention weights
        context = torch.matmul(attention_weights, V)  # Shape: (batch_size, num_heads, bag_size, head_dim)

        # Transpose to get back to (batch_size, bag_size, hidden_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, bag_size, -1)  # Shape: (batch_size, bag_size, hidden_dim)



        # Project back to input dimension
        context = self.output_layer(context)  # Shape: (batch_size, input_dim)
        
        # Average the context features
        sum_x = (context * mask.unsqueeze(-1)).sum(dim=1)  # Shape: (batch_size, hidden_dim)
        valid_count = mask.sum(dim=1).unsqueeze(1)  # Shape: (batch_size, 1)
        pooled_features = sum_x / valid_count  # Normalize by number of valid elements
        
        # Aggregate attention weights across all heads by averaging
        aggregated_weights = attention_weights.mean(dim=1)  # Shape: (batch_size,seq_length, seq_length)

        # Compute attention scores for each element (i.e., sum all columns to get the importance for each element)
        importance_scores = aggregated_weights.mean(dim=2)  # Shape: (batch_size,seq_length,)
        
        return pooled_features,context, importance_scores

def masked_avg_pool(x, mask):
    # Sum the activations only for the valid (non-padded) positions
    sum_x = (x * mask.unsqueeze(-1)).sum(dim=1)  # Shape: (batch_size, num_filters)
    valid_count = mask.sum(dim=1)  # Shape: (batch_size)
    avg_x = sum_x / valid_count.unsqueeze(1)  # Normalize by number of valid elements
    return avg_x

def create_mask(original_lengths, max_length,batch_size,device):
    """
    Creates a binary mask based on the original sequence lengths.
    
    Args:
    - original_lengths (Tensor): The original (un-padded) lengths of each sequence in the batch.
    - max_length (int): The length of the padded sequences (typically the maximum sequence length in the batch).
    
    Returns:
    - mask (Tensor): A binary mask where 1 indicates valid data and 0 indicates padding.
    """
    # Create a tensor of shape (batch_size, max_length) where each row contains 
    # indices from 0 to max_length-1, representing each time step in the sequence.
    mask = torch.arange(max_length,device=device).unsqueeze(0).expand(batch_size, -1)
    # The mask will have 1s where the sequence is valid and 0s where padding occurs
    mask = (mask < original_lengths.unsqueeze(1)).long()
    
    return mask


class MBA(nn.Module):
    def __init__(self,input_dim,max_seq_len, n_state=16,pos_embed_dim=8,lin_embed_dim=64,BI=True,num_layers=2):
        super(MBA, self).__init__()

#         self.pos_embedding = PositionalEmbedding(max_seq_len, pos_embed_dim)
        self.embedding = nn.Linear(input_dim, lin_embed_dim)
        num_filters=lin_embed_dim#+pos_embed_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, num_filters))
        self.GatedAttentionMILPooling =GatedAttentionPoolingMIL(input_dim=num_filters, hidden_dim=16)
        # A Mamba model is composed of a series of MambaBlocks interleaved
        # with normalization layers (e.g. RMSNorm)
#         self.layers = nn.ModuleList([
#             nn.ModuleList(
#                 [
#                     MambaBlock(**mamba_par),
#                     RMSNorm(d_input)
#                 ]
#             )
#             for _ in range(num_layers)
#         ])
#         layers=[]
#         for i in range(num_layers):
#             layers.append(BiMambaEncoder(num_filters, n_state,BI))
#         self.mambablocks_seq = nn.Sequential(*layers)
        self.mambablocks1 =BiMambaEncoder(num_filters, n_state,BI)
        self.mambablocks2 =BiMambaEncoder(num_filters, n_state,BI)
        self.mambablocks3 =BiMambaEncoder(num_filters, n_state,BI)
        self.mambablocks4 =BiMambaEncoder(num_filters, n_state,BI)
        self.mambablocks5 =BiMambaEncoder(num_filters, n_state,BI)
        self.mambablocks6 =BiMambaEncoder(num_filters, n_state,BI)
        self.mambablocks7 =BiMambaEncoder(num_filters, n_state,BI)
        self.mambablocks8 =BiMambaEncoder(num_filters, n_state,BI)
        self.out1 = nn.Linear(num_filters, 1)
        self.out2 = nn.Linear(num_filters, 1) 
        self.out3 = nn.Linear(num_filters, 1)

    def forward(self, x, x_lengths):
        batch_size, seq_len, _ = x.size()
        x=self.embedding(x)
        # Get positional embeddings for the current sequence length
#         pos_embed = self.pos_embedding(seq_len, x.device)  # (seq_len, embedding_dim)
#         # Expand positional embeddings to match the batch size
#         pos_embed = pos_embed.unsqueeze(0).expand(batch_size, seq_len, -1)  # (batch_size, seq_len, embedding_dim)
#         # Concatenate the positional embeddings with the original input
#         x = torch.cat([x, pos_embed], dim=-1) 
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # Shape: (batch_size, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)
        mask = create_mask(torch.tensor(x_lengths,device=x.device), seq_len+1,batch_size,x.device)
#         x = self.mambablocks1({0:x,1:mask})
#         x = self.mambablocks2({0:x,1:mask})
#         x = self.mambablocks3({0:x,1:mask})
#         x = self.mambablocks4({0:x,1:mask})
#         x = self.mambablocks5({0:x,1:mask})
#         x = self.mambablocks6({0:x,1:mask})
        x = self.mambablocks7({0:x,1:mask})
        mamba_out = self.mambablocks8({0:x,1:mask})
#         mamba_out = self.mambablocks_seq({0:x,1:mask})
#         masked=masked_avg_pool(mamba_out, mask)
        masked= self.GatedAttentionMILPooling(mamba_out, mask)
        x1 = self.out1(masked)
        x2 = self.out2(mamba_out[:, 1:, :]) #use .permute(0, 2, 1) if we want to use conv1d in output
        x3= self.out3(masked)
        #if x1 is 0 then duration should be 0
#         mask_x1 =torch.round(torch.sigmoid(x1))
#         x3 = x3 * (mask_x1 != 0).float()
        
        return x1,x2,x3


class ConvFeedForward2(nn.Module):
    def __init__(self, dilation, in_channels, out_channels,kernel_size,norm='IN'):
        super(ConvFeedForward2, self).__init__()
        padding = (kernel_size - 1) * dilation // 2 
        self.conv=nn.Conv1d(in_channels, out_channels, padding=padding, dilation=dilation, kernel_size=kernel_size)
        self.activation = nn.SiLU(inplace=True)
        if norm=='GN':
            self.norm = nn.GroupNorm(num_groups=4, num_channels=in_channels)
        elif norm=='BN':
            self.norm = nn.BatchNorm1d(in_channels)
        else:
            self.norm = nn.InstanceNorm1d(in_channels,track_running_stats=False)


    def forward(self, x):
        out=self.conv(x)
        out=self.norm(out)
        out=self.activation(out)
        return out

class ConvFeedForward(nn.Module):
    def __init__(self, dilation, in_channels, out_channels,kernel_size):
        super(ConvFeedForward, self).__init__()
        padding = (kernel_size - 1) * dilation // 2 
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, padding=padding, dilation=dilation, kernel_size=kernel_size),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)
# The follow code is modified from
# https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
def drop_path(x, drop_prob=0.0, training=False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output

class AffineDropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks) with a per channel scaling factor (and zero init)
    See: https://arxiv.org/pdf/2103.17239.pdf
    """

    def __init__(self, num_dim, drop_prob=0.0, init_scale_value=1e-4):
        super().__init__()
        self.scale = nn.Parameter(
            init_scale_value * torch.ones((1, num_dim, 1)),
            requires_grad=True
        )
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(self.scale * x, self.drop_prob, self.training)





class AttentionHelper(nn.Module):
    def __init__(self):
        super(AttentionHelper, self).__init__()
        self.softmax = nn.Softmax(dim=-1)


    def scalar_dot_att(self, proj_query, proj_key, proj_val, padding_mask):
        '''
        scalar dot attention.
        :param proj_query: shape of (B, C, L) => (Batch_Size, Feature_Dimension, Length)
        :param proj_key: shape of (B, C, L)
        :param proj_val: shape of (B, C, L)
        :param padding_mask: shape of (B, C, L)
        :return: attention value of shape (B, C, L)
        '''
        m, c1, l1 = proj_query.shape
        m, c2, l2 = proj_key.shape
        
        assert c1 == c2
        
        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)  # out of shape (B, L1, L2)
        attention = energy / np.sqrt(c1)
        attention = attention + torch.log(padding_mask + 1e-6) # mask the zero paddings. log(1e-6) for zero paddings
        attention = self.softmax(attention) 
        attention = attention * padding_mask
        attention = attention.permute(0,2,1)
        out = torch.bmm(proj_val, attention)
        return out, attention


class AttLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type): # r1 = r2
        super(AttLayer, self).__init__()
        
        self.query_conv = nn.Conv1d(in_channels=q_dim, out_channels=q_dim // r1, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=k_dim, out_channels=k_dim // r2, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=v_dim, out_channels=v_dim // r3, kernel_size=1)
        
        self.conv_out = nn.Conv1d(in_channels=v_dim // r3, out_channels=v_dim, kernel_size=1)

        self.bl = bl
        self.stage = stage
        self.att_type = att_type
        assert self.att_type in ['normal_att', 'block_att', 'sliding_att']
        assert self.stage in ['encoder','decoder']
        
        self.att_helper = AttentionHelper()

    def forward(self, x1, x2, mask):
        # x1 from the encoder
        # x2 from the decoder
        
        query = self.query_conv(x1)
        key = self.key_conv(x1)
         
        if self.stage == 'decoder':
            assert x2 is not None
            value = self.value_conv(x2)
        else:
            value = self.value_conv(x1)
            
        if self.att_type == 'normal_att':
            return self._normal_self_att(query, key, value, mask)
        elif self.att_type == 'block_att':
            return self._block_wise_self_att(query, key, value, mask)
        elif self.att_type == 'sliding_att':
            return self._sliding_window_self_att(query, key, value, mask)

    
    def _normal_self_att(self,q,k,v, mask):
        m_batchsize, c1, L = q.size()
        _,c2,L = k.size()
        _,c3,L = v.size()
        padding_mask = mask.unsqueeze(1)#torch.ones((m_batchsize, 1, L)).to(device) * mask[:,0:1,:]
        output, attentions = self.att_helper.scalar_dot_att(q, k, v, padding_mask)
        output = self.conv_out(F.relu(output))
        output = output[:, :, 0:L]
        return output * mask.unsqueeze(1)#mask[:, 0:1, :]  
        
    def _block_wise_self_att(self, q,k,v, mask):
        m_batchsize, c1, L = q.size()
        _,c2,L = k.size()
        _,c3,L = v.size()
        
        nb = L // self.bl
        if L % self.bl != 0:
            q = torch.cat([q, torch.zeros((m_batchsize, c1, self.bl - L % self.bl)).to(device)], dim=-1)
            k = torch.cat([k, torch.zeros((m_batchsize, c2, self.bl - L % self.bl)).to(device)], dim=-1)
            v = torch.cat([v, torch.zeros((m_batchsize, c3, self.bl - L % self.bl)).to(device)], dim=-1)
            nb += 1

        padding_mask = torch.cat([torch.ones((m_batchsize, 1, L)).to(device) * mask[:,0:1,:], torch.zeros((m_batchsize, 1, self.bl * nb - L)).to(device)],dim=-1)

        q = q.reshape(m_batchsize, c1, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c1, self.bl)
        padding_mask = padding_mask.reshape(m_batchsize, 1, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb,1, self.bl)
        k = k.reshape(m_batchsize, c2, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c2, self.bl)
        v = v.reshape(m_batchsize, c3, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c3, self.bl)
        
        output, attentions = self.att_helper.scalar_dot_att(q, k, v, padding_mask)
        output = self.conv_out(F.relu(output))
        
        output = output.reshape(m_batchsize, nb, c3, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize, c3, nb * self.bl)
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]     

    



class Encoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, output_dim, channel_masking_rate, att_type, alpha, drop_path_rate=0.3,encoder_only=False):
        super(Encoder, self).__init__()
        self.encoder_only = encoder_only
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1) # fc layer
        self.layers = nn.ModuleList([AttModule_mamba(2 ** i, num_f_maps, num_f_maps, r1, r2, 'sliding_att', 'encoder', 1, drop_path_rate=drop_path_rate) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, output_dim, 1)
        self.dropout = nn.Dropout2d(p=channel_masking_rate)
        self.channel_masking_rate = channel_masking_rate

    def forward(self, x, mask):
        if self.channel_masking_rate > 0:
            x = x.unsqueeze(2)
            x = self.dropout(x)
            x = x.squeeze(2)
        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, mask)
        if self.encoder_only:
            out = feature
        else:
            out = self.conv_out(feature) * mask.unsqueeze(1)#* mask[:, 0:1, :]
        return out

class Decoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, out_dim, att_type, alpha, drop_path_rate=0.3):
        super(Decoder, self).__init__()#         self.position_en = PositionalEncoding(d_model=num_f_maps)
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)

        self.layers = nn.ModuleList([AttModule_mamba(2 ** i, num_f_maps, num_f_maps, r1, r2, att_type, 'decoder', alpha, drop_path_rate=drop_path_rate) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, out_dim, 1)

    def forward(self, x,  mask):
        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, mask)
        out = self.conv_out(feature) * mask.unsqueeze(1)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *(-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).permute(0,2,1) # of shape (1, d_model, l)
        self.pe = nn.Parameter(pe, requires_grad=True)
#         self.register_buffer('pe', pe)
        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
#         self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        return x + self.pe[:, :, 0:x.shape[2]] 

def latent_mixup(features, labels1=None, labels3=None, alpha=0.2):
    """
    Applies mixup to features and corresponding labels for classification and reg head only.
    
    Args:
        features: Feature tensor [batch_size, feature_dim]
        labels: Classification labels [batch_size]
        alpha: Beta distribution parameter for mixing strength
        
    Returns:
        mixed_features: Mixed feature tensor
        mixed_labels1: Mixed classification labels with lambda weights
        mixed_labels3: Mixed regression labels with lambda weights
        mixup_lambda: Mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = features.shape[0]
    
    # Create permutation indices for mixing
    indices = torch.randperm(batch_size, device=features.device)
    
    # Mix features in latent space
    mixed_features = lam * features + (1 - lam) * features[indices]
    
    if labels1 is not None:
        mixed_labels1 = labels1[indices]
    else:
        mixed_labels1 = None

    if labels3 is not None:
        mixed_labels3 = lam * labels3 + (1 - lam) * labels3[indices]
    else:
        mixed_labels3 = None
    # Return original labels, permuted labels, and lambda for loss calculation
    return mixed_features, mixed_labels1, mixed_labels3, lam



class FeatureExtractor(nn.Module):
    def __init__(self,in_channels,norm,kernel_size=3,num_feature_layers=5,num_filters=64,tsm=False,tsm_horizon=64,featurelayer="TCN"):
        super(FeatureExtractor, self).__init__()

        layers=[]
        dilation = 1
        in_size=in_channels
        for i in range(num_feature_layers):
            padding = (kernel_size - 1) * dilation // 2  # To maintain same length
            if featurelayer=="TCN":
                layers.append(TCNLayer(in_size,num_filters, kernel_size, dilation, padding,norm))
            elif featurelayer=="ResNet":
                padding = (kernel_size - 1) * 1 // 2
                layers.append(ResTCNLayer(in_size,num_filters, kernel_size, 1, padding,norm))
            else:
                layers.append(ResTCNLayer(in_size,num_filters, kernel_size, dilation, padding,norm))
            in_size=num_filters
            dilation *= 2  # Increase dilation exponentially
        self.layers=nn.ModuleList(layers)
        self.tsm=tsm
        self.tsm_horizon=tsm_horizon

    def forward(self, x, mask, return_intermediates=False):
        features = []
        for layer in self.layers:
            x = layer(x, mask)
            features.append(x.clone())
        if self.tsm:
            x = self.compute_crosscorrelation_batch(x)
        if return_intermediates:
            return x, features
        return x
    
    def compute_crosscorrelation_batch(self, features):
        """
        Compute the autocorrelation of the feature map.
        """ 
        batch_size, channels, length = features.size()
        unit=self.tsm_horizon*16
        h=self.tsm_horizon
        device=features.device
        TSM = torch.zeros(batch_size, h, length,device=device)
        for i in range(batch_size):
            tsm_list=[]
            for b in range (0,length,unit):
                autocorr = torch.zeros(1,unit+h, unit+h)
                b_size=min(length-(b),unit+h)
                autocorr[0,:b_size,:b_size]=torch.corrcoef(features[i, :, b:b+b_size].T)[:,:]        
                mat=autocorr[0,:b_size,:b_size].squeeze(-0)
                mask_value=-1111
                row_indices = torch.arange(b_size+h).unsqueeze(1).repeat(1, b_size+h)  
                col_indices = torch.arange(b_size+h).unsqueeze(0).repeat(b_size+h, 1) 
                condition = ((row_indices) < (col_indices) ) & ( col_indices -row_indices<=(h))
                similar_matrix = torch.cat([torch.cat([mat,torch.zeros((b_size, h), dtype=torch.int)],dim=1),torch.zeros((h, b_size+h), dtype=torch.int)],dim=0)
                similar_matrix[~condition] = mask_value
                tsm=similar_matrix[:b_size,:][similar_matrix[:b_size,:] != mask_value].view(b_size,h)
                tsm_list.append(tsm[:min(length-(b),unit),:])
            TSM[i, :, :] =torch.cat(tsm_list,dim=0).unsqueeze(0).permute(0,2,1)
        return TSM.to(features.device) 
    
    def compute_crosscorrelation_batch_GPU(self, features):
        """
        Compute the autocorrelation of the feature map.
        """ 
        batch_size, channels, length = features.size()
        unit=self.tsm_horizon*8
        h=self.tsm_horizon
        device=features.device
        TSM = torch.zeros(batch_size, h, length,device=device)
        for i in range(batch_size):
            tsm_list=[]
            for b in range (0,length,unit):
                autocorr = torch.zeros(1,unit+h, unit+h,device=device)
                b_size=min(length-(b),unit+h)
                autocorr[0,:b_size,:b_size]=torch.corrcoef(features[i, :, b:b+b_size].T)[:,:]        
                mat=autocorr[0,:b_size,:b_size].squeeze(-0)
                mask_value=-1111
                row_indices = torch.arange(b_size+h,device=device).unsqueeze(1).repeat(1, b_size+h)  
                col_indices = torch.arange(b_size+h,device=device).unsqueeze(0).repeat(b_size+h, 1) 
                condition = ((row_indices) < (col_indices) ) & ( col_indices -row_indices<=(h))
                similar_matrix = torch.cat([torch.cat([mat,torch.zeros((b_size, h), dtype=torch.int,device=device)],dim=1),torch.zeros((h, b_size+h), dtype=torch.int,device=device)],dim=0)
                similar_matrix[~condition] = mask_value
                tsm=similar_matrix[:b_size,:][similar_matrix[:b_size,:] != mask_value].view(b_size,h)
                tsm_list.append(tsm[:min(length-(b),unit),:])
            TSM[i, :, :] =torch.cat(tsm_list,dim=0).unsqueeze(0).permute(0,2,1)
        return TSM#.to(features.device)  
    
    def compute_crosscorrelation(self, features):
        """
        Compute the autocorrelation of the feature map.
        """ 
        batch_size, channels, length = features.size()
        tsm = torch.zeros(batch_size, self.tsm_horizon, length)
        for i in range(batch_size):
            tsm_list=[]
            autocorr = torch.corrcoef(features[i, :, :].T)
            #autocorr=fast_corrcoef(features[i, :, :].T)
            for j in range(length):
                b_size=min(length-(j),self.tsm_horizon)
                tsm[i, 0:b_size, j] =autocorr[j:j+b_size,j]
        return tsm.to(features.device) 

class ConvProjection(nn.Module):
    """
    A single conv block with batch norm and dilated convolutions.
    """
    def __init__(self, in_channels, out_channels,norm,kernel_size=1,dropout_rate=0.2):
        super(ConvProjection, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels,kernel_size=kernel_size)
        if norm=='GN':
            self.norm = nn.GroupNorm(num_groups=4, num_channels=out_channels)
        elif norm=='BN':
            self.norm = nn.BatchNorm1d(out_channels)
        else:
            self.norm = nn.InstanceNorm1d(out_channels,track_running_stats=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class TCNLayer(nn.Module):
    """
    A single TCN layer with dilated convolutions.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding,norm,dropout_rate=0.2):
        super(TCNLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                               dilation=dilation, padding=padding)
        if norm=='GN':
            self.norm = nn.GroupNorm(num_groups=4, num_channels=out_channels)
        elif norm=='BN':
            self.norm = nn.BatchNorm1d(out_channels)
        else:
            self.norm = nn.InstanceNorm1d(out_channels,track_running_stats=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x,mask):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x* mask.unsqueeze(1)

class ResTCNLayer(nn.Module):
    """
    A single TCN layer with dilated convolutions.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding,norm,dropout_rate=0.2):
        super(ResTCNLayer, self).__init__()
#         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding)
#         if norm=='GN':
#             self.norm = nn.GroupNorm(num_groups=4, num_channels=out_channels)
#         elif norm=='BN':
#             self.norm = nn.BatchNorm1d(out_channels)
#         else:
#             self.norm = nn.InstanceNorm1d(out_channels,track_running_stats=False)
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x,mask):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual  # residual connection
        out = self.relu(out)
        return out * mask.unsqueeze(1)

class OutModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,norm,method):
        super(OutModule, self).__init__()
        if norm=='GN':
            self.norm = nn.GroupNorm(num_groups=4, num_channels=hidden_dim)
        elif norm=='BN':
            self.norm = nn.BatchNorm1d(hidden_dim)
        else:
            self.norm = nn.InstanceNorm1d(hidden_dim,track_running_stats=False)
            
        if method=='FC':
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim,hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, output_dim)
        else:
            self.fc1 = nn.Conv1d(input_dim, hidden_dim,1)
            self.fc2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
            self.fc3 = nn.Conv1d(hidden_dim,output_dim, 1)
        
    def forward(self, x):
        if self.norm is not None:
            x=self.norm(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

class GatedMLP(nn.Module):
    def __init__(self,input_dim, hidden_dim, output_dim,norm,method):
        super().__init__()

        self.activation = nn.SiLU()

#         if method=='FC':
        self.fc1 = nn.Linear(input_dim, 2 *hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
#         else:
#             self.fc1 = nn.Conv1d(input_dim, 2 *hidden_dim,1)
#             self.fc2 = nn.Conv1d(hidden_dim,output_dim, 1)

    def forward(self, x):
        y = self.fc1(x)
        y, gate = y.chunk(2, dim=-1)
        y = y * self.activation(gate)
        y = self.fc2(y)
        return y



class MBA_v1(nn.Module):
    def __init__(self,input_dim,num_filters=64,num_encoder_layers=7,num_decoder_layers=3,drop_path_rate=0.3,kernel_size_encoder=3,kernel_size_decoder=7,dropout_rate=0.2,max_seq_len=1220,encoderlayer="TCN",norm1='IN',norm2='IN',norm3='BN'):
        super(MBA_v1, self).__init__()
        
        self.input_projection = ConvProjection(input_dim, num_filters,norm1)# Input projection to transform raw signals to model dimension
        self.cls_token = nn.Parameter(torch.randn(1, num_filters,1))
        self.num_encoder_layers = num_encoder_layers
        self.encoder = FeatureExtractor(num_filters,num_filters=num_filters,kernel_size=kernel_size_encoder,num_feature_layers=num_encoder_layers,tsm=False,featurelayer=encoderlayer,norm=norm1)#
        self.positional_encoding = PositionalEncoding(d_model=num_filters,max_len=max_seq_len)

        self.decoder = nn.ModuleList([AttModule_mamba_cross(2 ** i, num_filters, num_filters, 1, 1, 'sliding_att', 'encoder', 1, drop_path_rate=drop_path_rate, kernel_size=kernel_size_decoder, dropout_rate=dropout_rate, norm=norm2) for i in range(num_decoder_layers)])
        
        self.out1 = OutModule(num_filters,num_filters, 1,norm3,'FC')  
        self.out2 = OutModule(num_filters,num_filters, 1,norm3,'Conv')
        self.out3 = OutModule(num_filters,num_filters, 1,norm3,'FC')
        
        

    def forward(self, x, x_lengths,labels1=None, labels3=None, apply_mixup=False, mixup_alpha=0.2):
        #global xx,mask,mask2,lengths
        
        batch_size, seq_len, channels = x.size()       

        
        # Create padding mask from original input
        mask = create_mask(torch.tensor(x_lengths,device=x.device), seq_len,batch_size,x.device)
        
        x = self.input_projection(x.permute(0, 2, 1))# Transpose for conv layers: [B, seq_len, num_filters] -> [B, num_filters, seq_len]
        
        token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((token, x), dim=2)
        mask=torch.cat((torch.ones(batch_size,1,device=x.device),mask), dim=1)
        
        
        if self.num_encoder_layers>0:
#             x=self.encoder(x,mask)
            x, feature_maps = self.encoder(x, mask, return_intermediates=True)

        x = self.positional_encoding(x)
            
            
        # Apply encoder layers with UNet-style skip connections
        if self.num_encoder_layers > 0:
            # UNet-style: connect each encoder layer with reversed feature layer (U-shape)
            # First encoder layer gets last feature map, last encoder layer gets first feature map
            for i, layer in enumerate(self.decoder):
                # Use cross-attention to integrate skip connections
                if i < len(feature_maps):
                    skip_idx = len(feature_maps) - 1 - i  # Reverse the index for U-Net structure
                    encoder_states = feature_maps[skip_idx]  # Skip connection as encoder states
                    x = layer(x, encoder_states, mask)  # Cross-attention integration
                else:
                    x = layer(x, None, mask)  # No skip connection available

        else:
            # Standard forward without skip connections
            for layer in self.decoder:
                x = layer(x, None, mask)


        pooled_features = x[:,:,0]

        x1 = self.out1(pooled_features)
        x3= self.out3(pooled_features)
        x2 = self.out2(x[:,:,1:]) 

        mask=mask[:,1:].reshape(-1).bool()
        x2=x2.view(-1)
        x2=x2[mask]

        return x1,x2,x3







# CNN Encoder
class CNNEncoder(nn.Module):
    def __init__(self, input_channels, hidden_channels, latent_dim):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, hidden_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Pool over sequence length
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(hidden_channels, latent_dim)
 
    def forward(self, x, lengths=None):
        # x: (batch_size, seq_len, features)
        x = x.transpose(1, 2)  # (batch_size, features, seq_len)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
 
        # Global pooling over sequence length
        x = self.global_avg_pool(x)  # (batch_size, channels, 1)
        x = self.flatten(x)  # (batch_size, channels)
        latent = self.linear(x)  # (batch_size, latent_dim)
        return latent

# CNN Decoder
class CNNDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_channels, output_channels, seq_len):
        super(CNNDecoder, self).__init__()
        self.linear = nn.Linear(latent_dim, hidden_channels * seq_len)
        self.seq_len = seq_len
        self.conv_transpose1 = nn.ConvTranspose1d(hidden_channels, hidden_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.conv_transpose2 = nn.ConvTranspose1d(hidden_channels, hidden_channels, kernel_size=5, padding=2)
 
    def forward(self, latent):
        # Expand latent to feature maps
        x = self.linear(latent)  # (batch_size, hidden_channels * seq_len)
        x = x.view(-1, self.conv_transpose1.in_channels, self.seq_len)  # (batch, channels, seq_len)
        x = self.conv_transpose1(x)
        x = self.relu(x)
        x = self.conv_transpose2(x)
        x = x.transpose(1, 2)  # (batch_size, seq_len, features)
        return x

# Complete Model
class CNNSeqAutoencoder(nn.Module):
    def __init__(self, input_channels, hidden_channels, latent_dim, max_seq_len):
        super(CNNSeqAutoencoder, self).__init__()
        self.encoder = CNNEncoder(input_channels, hidden_channels, latent_dim)
        self.decoder = CNNDecoder(latent_dim, hidden_channels, input_channels, max_seq_len)
        
        self.out1 = nn.Linear(hidden_channels, 1)  
        self.out2 = nn.Conv1d(hidden_channels, 1, 1)
        self.out3 = nn.Linear(hidden_channels, 1)
    def forward(self, x,x_lengths=None,labels1=None, labels3=None, apply_mixup=False, mixup_alpha=0.2):
        #batch,seq,channels
        batch_size, seq_len, channels = x.size()   
        latent = self.encoder(x)
        recon = self.decoder(latent)
        attention_weights = torch.zeros(batch_size, seq_len,1,device=x.device)
        pooled_features = recon[:,0,:]
        mask = create_mask(torch.tensor(x_lengths,device=x.device), seq_len,batch_size,x.device)
        x1 = self.out1(pooled_features)
        x2 = self.out2(recon.permute(0, 2, 1))
        x3 = self.out3(pooled_features)


        mask=mask.reshape(-1).bool()
        x2=x2.view(-1)
        x2=x2[mask]
        attention_weights=attention_weights.reshape(-1)
        attention_weights=attention_weights[mask]
        return x1,x2,x3,attention_weights,None,None

class AttModule(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, r1, r2, att_type, stage, alpha,kernel_size=7,dropout_rate=0.2):
        super(AttModule, self).__init__()
        self.feed_forward = ConvFeedForward(dilation, in_channels, out_channels,kernel_size=kernel_size)
        self.instance_norm = nn.InstanceNorm1d(in_channels, track_running_stats=False)
        self.att_layer = AttLayer(in_channels, in_channels, out_channels, r1, r1, r2, dilation, att_type=att_type, stage=stage) # dilation
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.alpha = alpha
        
    def forward(self, x,f,  mask):
        out = self.feed_forward(x)
        out = self.alpha * self.att_layer(self.instance_norm(out),f, mask) + out
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask.unsqueeze(1)#[:, 0:1, :]


class ResTCNLayer(nn.Module):
    """
    A single TCN layer with dilated convolutions.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding,norm,dropout_rate=0.2):
        super(ResTCNLayer, self).__init__()
#         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding)
#         if norm=='GN':
#             self.norm = nn.GroupNorm(num_groups=4, num_channels=out_channels)
#         elif norm=='BN':
#             self.norm = nn.BatchNorm1d(out_channels)
#         else:
#             self.norm = nn.InstanceNorm1d(out_channels,track_running_stats=False)
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x,mask):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual  # residual connection
        out = self.relu(out)
        return out * mask.unsqueeze(1)

class MaskMambaBlock2(nn.Module):
    def __init__(
        self,        
        n_embd,                # dimension of the input features
        kernel_size=4,         # conv kernel size
        n_ds_stride=1,         # downsampling stride for the current layer
        drop_path_rate=0.3,         # drop path rate
    ) -> None:
        super().__init__()
        #self.mamba = ViM(n_embd, d_conv=kernel_size, use_fast_path=True, bimamba_type='v2')
        self.mamba = Mamba(n_embd,d_conv=kernel_size,use_fast_path=True)
        if n_ds_stride > 1:
            self.downsample = MaxPooler(kernel_size=3, stride=2, padding=1)
        else:
            self.downsample = None    
        self.norm = nn.LayerNorm(n_embd)
                
        # drop path
        if drop_path_rate > 0.0:
            self.drop_path = AffineDropPath(n_embd, drop_prob=drop_path_rate)
        else:
            self.drop_path = nn.Identity()

    def forward(self, x, mask):
        res = x
        x_ = x.transpose(1,2)
#         x_ = self.norm(x_)
        x_ = self.mamba(x_).transpose(1, 2)
        x = x_ * mask.unsqueeze(1)
#         x  = res + self.drop_path(x)
#         x  =  self.drop_path(x)
        if self.downsample is not None:
            x, mask = self.downsample(x, mask)

        return  x

class MaskMambaBlock(nn.Module):
    def __init__(
        self,        
        n_embd,                # dimension of the input features
        kernel_size=4,         # conv kernel size
        n_ds_stride=1,         # downsampling stride for the current layer
        drop_path_rate=0.3,         # drop path rate
    ) -> None:
        super().__init__()
        #self.mamba = ViM(n_embd, d_conv=kernel_size, use_fast_path=True, bimamba_type='v2')
        self.mamba = Mamba(n_embd,d_conv=kernel_size,use_fast_path=True)
        if n_ds_stride > 1:
            self.downsample = MaxPooler(kernel_size=3, stride=2, padding=1)
        else:
            self.downsample = None    
        self.norm = nn.LayerNorm(n_embd)
                
        # drop path
        if drop_path_rate > 0.0:
            self.drop_path = AffineDropPath(n_embd, drop_prob=drop_path_rate)
        else:
            self.drop_path = nn.Identity()

    def forward(self, x, mask):
        res = x
        x_ = x.transpose(1,2)
        x_ = self.norm(x_)
        x_ = self.mamba(x_).transpose(1, 2)
        x = x_ * mask.unsqueeze(1)
        x  = res + self.drop_path(x)
        if self.downsample is not None:
            x, mask = self.downsample(x, mask)

        return  x
class AttModule_mamba(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, att_type, stage, alpha,norm, drop_path_rate=0.3,kernel_size=7,dropout_rate=0.2):
        super(AttModule_mamba, self).__init__()
        self.feed_forward = ConvFeedForward(dilation, in_channels, out_channels,kernel_size=kernel_size)
        if norm=='GN':
            self.norm = nn.GroupNorm(num_groups=4, num_channels=out_channels)
        elif norm=='BN':
            self.norm = nn.BatchNorm1d(out_channels)
        else:
            self.norm = nn.InstanceNorm1d(out_channels,track_running_stats=False)

        self.att_layer = MaskMambaBlock(in_channels, drop_path_rate=drop_path_rate,kernel_size=kernel_size) # dilation
        #self.att_layer = BiMambaEncoder(64, 16,True)
        # self.att_layer = MaskMambaBlock_DBM(in_channels, drop_path_rate=drop_path_rate) # dilation
        self.conv_1x1 = nn.Conv1d(in_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.alpha = alpha
        
    def forward(self, x,f, mask):
        m_batchsize, c1,L = x.size()        
        #padding_mask = torch.ones((m_batchsize, 1, L)).to(device) * mask[:,0:1,:]
        padding_mask = mask
        out = self.feed_forward(x)
        out = self.alpha * self.att_layer(self.norm(out), padding_mask) + out
        # out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask.unsqueeze(1)

class AttModule_mamba2(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, att_type, stage, alpha,norm, drop_path_rate=0.3,kernel_size=7,dropout_rate=0.2):
        super(AttModule_mamba2, self).__init__()
        self.feed_forward = ConvFeedForward(dilation, in_channels, out_channels,kernel_size=kernel_size,norm=norm)

        if norm=='GN':
            self.norm = nn.GroupNorm(num_groups=4, num_channels=in_channels)
        elif norm=='BN':
            self.norm = nn.BatchNorm1d(in_channels)
        else:
            self.norm = nn.InstanceNorm1d(in_channels,track_running_stats=False)
        
        self.att_layer = MaskMambaBlock(in_channels, drop_path_rate=drop_path_rate,kernel_size=kernel_size) # dilation
        self.conv_1x1 = nn.Conv1d(in_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.alpha = alpha
        
    def forward(self, x,f, mask):
        m_batchsize, c1,L = x.size()        
        #padding_mask = torch.ones((m_batchsize, 1, L)).to(device) * mask[:,0:1,:]
        padding_mask = mask
#         out = self.feed_forward(x)
        out = self.feed_forward(x)
        out = self.dropout(out)
        out = self.alpha * self.norm(self.att_layer(out, padding_mask)) * out
        # out = self.conv_1x1(out)
        return (x+out) * mask.unsqueeze(1)

class AttModule_mamba_cross(nn.Module):
    """Cross-attention module for encoder-decoder architectures."""
    def __init__(self, dilation, in_channels, out_channels, r1, r2, att_type, stage, alpha, drop_path_rate=0.3, kernel_size=7, dropout_rate=0.2, norm="IN"):
        super(AttModule_mamba_cross, self).__init__()
        self.feed_forward = ConvFeedForward(dilation, in_channels, out_channels, kernel_size=kernel_size)
        # self.instance_norm = nn.InstanceNorm1d(in_channels, track_running_stats=False)
        if norm=='GN':
            self.norm = nn.GroupNorm(num_groups=4, num_channels=out_channels)
        elif norm=='BN':
            self.norm = nn.BatchNorm1d(out_channels)
        else:
            self.norm = nn.InstanceNorm1d(out_channels, track_running_stats=False)
        self.self_att_layer = MaskMambaBlock(out_channels, drop_path_rate=drop_path_rate, kernel_size=kernel_size)
        self.cross_att_layer = MaskMambaBlock(out_channels, drop_path_rate=drop_path_rate, kernel_size=kernel_size)  
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.alpha = alpha
        
    def forward(self, x, encoder_states, padding_mask):
        m_batchsize, c1, L = x.size()        
        
        out = self.feed_forward(x)
        
        # Self-attention on decoder states
        out = self.alpha * self.self_att_layer(self.norm(out), padding_mask) + out
        
        # Cross-attention to encoder states (if provided)
        if encoder_states is not None:
            # Ensure encoder_states and out have compatible dimensions
            if encoder_states.shape[2] == out.shape[2]:  # Same sequence length
                cross_out = self.alpha * self.cross_att_layer(self.norm(encoder_states), padding_mask)
                out = out + cross_out
            else:
                # Interpolate encoder states to match decoder length
                encoder_states_resized = F.interpolate(encoder_states, size=L, mode='linear', align_corners=False)
                cross_out = self.alpha * self.cross_att_layer(self.norm(encoder_states_resized), padding_mask)
                out = out + cross_out
        
        out = self.dropout(out)
        
        if padding_mask is not None:
            return (x + out) * padding_mask.unsqueeze(1)
        else:
            return x + out


class AttModule_cross(nn.Module):
    """Cross-attention module for non-Mamba encoder-decoder architectures.""" 
    def __init__(self, dilation, in_channels, out_channels, r1, r2, att_type, stage, alpha, kernel_size=7, dropout_rate=0.2):
        super(AttModule_cross, self).__init__()
        self.feed_forward = ConvFeedForward(dilation, in_channels, out_channels, kernel_size=kernel_size)
        self.instance_norm = nn.InstanceNorm1d(in_channels, track_running_stats=False)
        self.self_att_layer = AttLayer(in_channels, in_channels // r1, in_channels // r2, att_type, stage)
        self.cross_att_layer = AttLayer(in_channels, in_channels // r1, in_channels // r2, att_type, stage)
        self.conv_1x1 = nn.Conv1d(in_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.alpha = alpha
        
    def forward(self, x, encoder_states, padding_mask):
        out = self.feed_forward(x)
        
        # Self-attention
        out = self.alpha * self.self_att_layer(self.instance_norm(out), None) + out
        
        # Cross-attention to encoder states
        if encoder_states is not None:
            if encoder_states.shape[2] == out.shape[2]:
                cross_out = self.alpha * self.cross_att_layer(self.instance_norm(encoder_states), None)
                out = out + cross_out
            else:
                encoder_states_resized = F.interpolate(encoder_states, size=out.shape[2], mode='linear', align_corners=False)
                cross_out = self.alpha * self.cross_att_layer(self.instance_norm(encoder_states_resized), None)
                out = out + cross_out
        
        out = self.dropout(out)
        
        if padding_mask is not None:
            return (x + out) * padding_mask.unsqueeze(1)
        else:
            return x + out


class MBA_tsm_encoder_decoder_progressive_with_skip_connection(nn.Module):
    """
    Pure encoder-decoder architecture for multi-task scratch detection.
    Encoder processes raw accelerometer signals directly without FeatureExtractor.
    Decoder generates outputs with cross-attention to encoder states.
    """
    def __init__(self, input_dim, 
                 n_state=16,
                 pos_embed_dim=16,
                 num_filters=64,
                 BI=True,
                 num_encoder_layers=4,
                 num_decoder_layers=3,
                 drop_path_rate=0.3,
                 kernel_size_mba=7,
                 max_seq_len=1220,
                 cls_token=False,
                 add_positional_encoding=True,
                 num_heads=4,
                 dropout_rate=0.2,
                 mba=True,
                 average_mask=False,
                 average_window_size=20,
                 supcon_loss=False,
                 cls_src="encoder"):
        super(MBA_tsm_encoder_decoder_progressive_with_skip_connection, self).__init__()
        
        self.add_positional_encoding = add_positional_encoding
        self.max_seq_len = max_seq_len
        self.CLS = cls_token
        self.average_mask = average_mask
        self.average_window_size = average_window_size
        self.supcon_loss = supcon_loss
        self.input_dim = input_dim
        self.num_filters = num_filters
        self.cls_src = cls_src

        # Input projection to transform raw signals to model dimension
        self.input_projection = nn.Linear(input_dim, num_filters)
        
        # Progressive depth reduction (U-Net style) architecture
        # Define progressive feature dimensions FIRST
        self.encoder_dims = [num_filters * (2 ** i) for i in range(num_encoder_layers)]  # [64, 128, 256, 512]
        self.decoder_dims = [self.encoder_dims[-(i+1)] for i in range(num_decoder_layers)]  # [512, 256, 128] (reverse)
        
        # Optional positional encoding
        if add_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_model=num_filters, max_len=max_seq_len)
        
        # Pooling layer for final representations (use final encoder dimension)
        self.MultiHeadSelfAttentionPooling = MultiHeadSelfAttentionPooling(
            input_dim=self.decoder_dims[-1], hidden_dim=self.decoder_dims[-1], num_heads=num_heads
        )
        
        # Add dimension projection layers for progressive feature expansion/reduction
        self.encoder_projections = nn.ModuleList([
            nn.Conv1d(self.encoder_dims[i-1] if i > 0 else num_filters, self.encoder_dims[i], 1)
            for i in range(num_encoder_layers)
        ])
        
        self.decoder_projections = nn.ModuleList([
            nn.Conv1d(self.decoder_dims[i-1] if i > 0 else self.encoder_dims[-1], self.decoder_dims[i], 1)
            for i in range(num_decoder_layers)
        ])
        
        # Skip connection projection layers to match dimensions
        self.skip_projections = nn.ModuleList([
            nn.Conv1d(self.encoder_dims[-(i+1)], self.decoder_dims[i], 1)
            for i in range(min(num_decoder_layers, num_encoder_layers))
        ])

        # Encoder layers with progressive feature dimensions
        if not mba:
            self.encoder = nn.ModuleList([
                AttModule(2 ** i, self.encoder_dims[i], self.encoder_dims[i], 2, 2, 'normal_att', 'encoder', alpha=1, 
                         kernel_size=kernel_size_mba, dropout_rate=dropout_rate) 
                for i in range(num_encoder_layers)
            ])
        else:
            self.encoder = nn.ModuleList([
                AttModule_mamba(2 ** i, self.encoder_dims[i], self.encoder_dims[i], 2, 2, 'sliding_att', 'encoder', 1, 
                               drop_path_rate=drop_path_rate, dropout_rate=dropout_rate, 
                               kernel_size=kernel_size_mba) 
                for i in range(num_encoder_layers)
            ])

        # Decoder layers with progressive feature dimensions
        if not mba:
            self.decoder = nn.ModuleList([
                AttModule_cross(2 ** i, self.decoder_dims[i], self.decoder_dims[i], 2, 2, 'cross_att', 'decoder', alpha=1, 
                               kernel_size=kernel_size_mba, dropout_rate=dropout_rate) 
                for i in range(num_decoder_layers)
            ])
        else:
            self.decoder = nn.ModuleList([
                AttModule_mamba_cross(2 ** i, self.decoder_dims[i], self.decoder_dims[i], 2, 2, 'cross_att', 'decoder', 1, 
                                     drop_path_rate=drop_path_rate, dropout_rate=dropout_rate, 
                                     kernel_size=kernel_size_mba) 
                for i in range(num_decoder_layers)
            ])
        
        # Task-specific output heads (use final encoder/decoder dimensions)
        if self.cls_src == "encoder":
            self.out1 = nn.Linear(self.encoder_dims[-1], 1)  # Classification head (from encoder CLS)
            self.out3 = nn.Linear(self.encoder_dims[-1], 1)  # Regression head (from encoder CLS)
        else:
            self.out1 = nn.Linear(self.decoder_dims[-1], 1)
            self.out3 = nn.Linear(self.decoder_dims[-1], 1)
        self.out2 = nn.Conv1d(self.decoder_dims[-1], 1, 1)  # Sequence labeling head (from decoder)


        if self.average_mask:
            self.scale_out2 = nn.Linear(max_seq_len, max_seq_len//average_window_size)

        # Remove projection layers - use raw cls_token directly for contrastive learning

    def forward(self, x,x_lengths, labels1=None, labels3=None, apply_mixup=False, mixup_alpha=0.2,  **kwargs):
        batch_size, seq_len, channels = x.size()       

        
        # Create padding mask from original input
        mask = create_mask(torch.tensor(x_lengths,device=x.device), seq_len,batch_size,x.device)
        
        # Project input to model dimension: [B, seq_len, input_dim] -> [B, seq_len, num_filters]
        x = self.input_projection(x)
        
        # Transpose for conv layers: [B, seq_len, num_filters] -> [B, num_filters, seq_len]
        x = x.permute(0, 2, 1)
        
        # Add positional encoding if enabled
        if self.add_positional_encoding:
            x = self.positional_encoding(x)
        
        # Progressive encoder pass with dimension expansion
        encoder_states = []
        encoder_out = x  # Start with [B, num_filters, seq_len]
        
        for i, (proj, layer) in enumerate(zip(self.encoder_projections, self.encoder)):
            # Progressive dimension expansion: num_filters -> encoder_dims[i]
            encoder_out = proj(encoder_out)  # [B, encoder_dims[i], seq_len]
            encoder_out = layer(encoder_out, None, mask)
            encoder_states.append(encoder_out)
        
        # Extract CLS token from encoder output for information bottleneck
        # encoder_out shape: [B, encoder_dims[-1], seq_len] -> permute to [B, seq_len, encoder_dims[-1]]
        encoder_features = encoder_out.permute(0, 2, 1)  # [B, seq_len, encoder_dims[-1]]
        
        
        
        # SEQUENCE PREDICTION: Choose between encoder-only or encoder-decoder approach
        # Option 1: Use decoder (comment out for encoder-only)
        # DECODER PASS: Use full encoder features instead of CLS token bottleneck
        # Use encoder output directly as decoder input (preserves spatial information)
        decoder_input = encoder_out  # [B, num_filters, seq_len] - already in correct format
        
        
        # Progressive decoder pass with dimension reduction and skip connections
        decoder_out = decoder_input  # Initialize decoder with encoder features [B, encoder_dims[-1], seq_len]
        
        for i, (proj, layer) in enumerate(zip(self.decoder_projections, self.decoder)):
            # Progressive dimension reduction: decoder_dims[i-1] -> decoder_dims[i]
            decoder_out = proj(decoder_out)  # [B, decoder_dims[i], seq_len]
            
            # Skip connection from corresponding encoder layer
            if i < len(encoder_states) and i < len(self.skip_projections):
                # U-Net style: reverse order (last encoder -> first decoder)
                encoder_skip = encoder_states[-(i+1)]  # [B, encoder_dims[-(i+1)], seq_len]
                # Project skip connection to match decoder dimension
                encoder_skip_proj = self.skip_projections[i](encoder_skip)  # [B, decoder_dims[i], seq_len]
            else:
                encoder_skip_proj = encoder_states[-1]  # Fallback to final encoder state
            
            # Decoder processes with cross-attention to encoder skip connection
            decoder_out = layer(decoder_out, encoder_skip_proj, mask)
        
        if self.cls_src == "encoder":
            features = encoder_out
        else:
            features = decoder_out
        # Get CLS token (first position) from encoder
        if self.CLS:           
            cls_feature = features[:,:,0]
            attention_weights = torch.zeros(batch_size, seq_len, 1, device=x.device)
        else:
            # Multi-head attention pooling to get global representation
            cls_feature, weighted_features, attention_weights = self.MultiHeadSelfAttentionPooling(features.permute(0, 2, 1), mask)
        
        # Apply mixup augmentation to CLS token if enabled
        if apply_mixup and self.training:
            mixed_cls_features, mixed_labels1, mixed_labels3, mixup_lambda = latent_mixup(
                cls_feature, labels1, labels3, alpha=mixup_alpha
            )
            mixup_info = (mixed_labels1, mixed_labels3, mixup_lambda)
        else:
            mixed_cls_features = cls_feature
            mixup_info = None
  
        # ENCODER OUTPUTS: Classification and Regression from CLS token
        x1 = self.out1(mixed_cls_features)  # Classification from encoder CLS: [B, 1]
        x3 = self.out3(mixed_cls_features)  # Regression from encoder CLS: [B, 1]

        # DECODER OUTPUT: Sequence mask prediction
        x2 = self.out2(decoder_out)     # Sequence labeling from decoder: [B, 1, seq_len]
        
        # Option 2: Use encoder output directly (uncomment for encoder-only)
        # x2 = self.out2(encoder_out)    # Sequence labeling from encoder: [B, 1, seq_len] 
        # x2 = x2.squeeze(1)             # [B, seq_len]

        # Apply mask averaging if enabled
        if self.average_mask:
            x2 = x2.view(batch_size, seq_len//self.average_window_size, self.average_window_size).mean(dim=2)

        # Contrastive embedding from encoder CLS token
        if self.supcon_loss:
            contrastive_embedding = cls_feature  # Use raw CLS token directly
        else:
            contrastive_embedding = None
        
        #Remove padded values
        mask=mask.reshape(-1).bool()
        x2=x2.view(-1)
        x2=x2[mask]
        attention_weights=attention_weights.reshape(-1)
        attention_weights=attention_weights[mask]

        return x1, x2, x3, attention_weights, contrastive_embedding, mixup_info


class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels,kernel_size, stride, padding)
#         self.bn1 = nn.InstanceNorm1d(out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels,kernel_size, 1, padding)
#         self.bn2 = nn.InstanceNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1, stride)
        else:
            self.shortcut = nn.Identity()
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class EncoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, downsample):
        super().__init__()
        blocks = []
        blocks.append(ResBlock1D(in_channels, out_channels, stride=2 if downsample else 1))
        for _ in range(1, num_blocks):
            blocks.append(ResBlock1D(out_channels, out_channels, stride=1))
        self.stage = nn.Sequential(*blocks)
        sew.downsample=downsample
    def forward(self, x,mask=None):
        out=self.stage(x)
        if mask is not None:
            if self.downsample:
                mask = mask[:, ::2]
            out=out*mask.unsqueeze(1)
        return out,mask

class DecoderBlock(nn.Module):
    def __init__(self, up_in_channels, skip_channels, out_channels,MBA=True):
        super().__init__()
        self.up = nn.ConvTranspose1d(up_in_channels, skip_channels, kernel_size=2, stride=2)
        if MBA:
            self.resblock = ResBlock1D(skip_channels * 2, out_channels)
        else:
            self.resblock = AttModule_mamba(1, skip_channels * 2, out_channels, 'sliding_att', 'encoder', 1, drop_path_rate=0.1,dropout_rate=0.2,kernel_size=3,norm='IN')
            
    def forward(self, x, skip,mask=None):
        x = self.up(x)
        # Align lengths for concat
        if x.size(-1) < skip.size(-1):
            x = F.pad(x, (0, skip.size(-1) - x.size(-1)))
        elif x.size(-1) > skip.size(-1):
            x = x[..., :skip.size(-1)]
        x = torch.cat([x, skip], dim=1)
        x = self.resblock(x)
        if mask is not None: 
            mask = mask[:, :out.shape[2]*2:2]
            if mask.size(-1) < x.size(-1):
                mask = F.pad(mask, (0, x.size(-1) - mask.size(-1)))
            x=x*mask.unsqueeze(1)
        return x,mask

class ResNetUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, features=[64, 128, 256, 512], blocks_per_stage=[2, 2, 2, 2]):
        super().__init__()
        self.init_conv = nn.Conv1d(in_channels, features[0], kernel_size=7, stride=2, padding=3)
#         self.init_bn = nn.InstanceNorm1d(features[0])
        self.init_bn = nn.BatchNorm1d(features[0])
        
        self.init_relu = nn.ReLU(inplace=True)
        self.init_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        # Encoder
        self.encoder_stages = nn.ModuleList()
        prev_channels = features[0]
        for idx, (feat, n_blocks) in enumerate(zip(features, blocks_per_stage)):
            self.encoder_stages.append(
                EncoderStage(prev_channels, feat, num_blocks=n_blocks, downsample=(idx != 0))
            )
            prev_channels = feat
        # Bottleneck
        self.bottleneck = ResBlock1D(features[-1], features[-1]*2, stride=1)
        # Decoder
        rev_features = features[::-1]
        up_in_channels = features[-1]*2
        self.decoder_blocks = nn.ModuleList()
        for skip_channels in rev_features:
            self.decoder_blocks.append(
                DecoderBlock(up_in_channels, skip_channels, skip_channels)
            )
            up_in_channels = skip_channels

        self.out1 = OutModule2(features[0],features[0], num_classes,'IN','FC')  
        self.out2 = OutModule2(features[0],features[0], num_classes,'IN','Conv')
        self.out3 = OutModule2(features[0],features[0], num_classes,'IN','FC')
        
    def forward(self, x, x_lengths,labels1=None, labels3=None, apply_mixup=False, mixup_alpha=0.2):
        batch_size, seq_len, channels = x.size()
        mask = create_mask(torch.tensor(x_lengths,device=x.device), seq_len,batch_size,x.device)
#         x = self.input_projection(x.permute(0, 2, 1))
        contrastive_embedding=None
        x=x.permute(0, 2, 1)
        input_length = x.size(-1)
        x = self.init_conv(x)
        x = self.init_bn(x)
        x = self.init_relu(x)
        x = self.init_pool(x)
        skips = []
        for enc in self.encoder_stages:
            x = enc(x)
            skips.append(x)
        x = self.bottleneck(x)
        for dec, skip in zip(self.decoder_blocks, reversed(skips)):
            x = dec(x, skip)
        #x = self.final_conv(x)
        # Ensure output length matches input length
        if x.size(-1) < input_length:
            x = F.pad(x, (0, input_length - x.size(-1)))
        elif x.size(-1) > input_length:
            x = x[..., :input_length]
        
        attention_weights = torch.zeros(batch_size, seq_len,1,device=x.device)
        pooled_features = x[:,:,0]
        mixed_features = pooled_features
        mixup_info = None
        x1 = self.out1(mixed_features)
        x2 = self.out2(x) 
        x3= self.out3(mixed_features)


        mask=mask.reshape(-1).bool()
        x2=x2.view(-1)
        x2=x2[mask]
        attention_weights=attention_weights.reshape(-1)
        attention_weights=attention_weights[mask]

        return x1,x2,x3,attention_weights,contrastive_embedding, mixup_info

    

    

class MBA_tsm_ED_Encoder(nn.Module):
    def __init__(self, num_layers, num_f_maps, input_dim, output_dim,  att_type, alpha, mamba=True, drop_path_rate=0.3,channel_masking_rate=0.3):
        super(MBA_tsm_ED_Encoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1) # fc layer

        self.layers = nn.ModuleList([AttModule_mamba(2 ** i, num_f_maps, num_f_maps, att_type, 'encoder', alpha, drop_path_rate=drop_path_rate) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, output_dim, 1)
        self.dropout = nn.Dropout2d(p=channel_masking_rate)
        self.channel_masking_rate = channel_masking_rate

    def forward(self, x, mask,positional_encoding):
        '''
        :param x: (N, C, L)
        :param mask:
        :return:
        '''
        if self.channel_masking_rate > 0:
            x = x.unsqueeze(2)
            x = self.dropout(x)
            x = x.squeeze(2)
    
        feature = self.conv_1x1(x)
#         feature = positional_encoding(feature)
        for layer in self.layers:
            feature = layer(feature, None, mask)
        
        out = self.conv_out(feature) * mask.unsqueeze(1)#* mask[:, 0:1, :]

        return out, feature


class MBA_tsm_ED_Decoder(nn.Module):
    def __init__(self, num_layers, num_f_maps, input_dim, output_dim, att_type, alpha, mamba=True, drop_path_rate=0.3):
        super(MBA_tsm_ED_Decoder, self).__init__()#         self.position_en = PositionalEncoding(d_model=num_f_maps)
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        self.layers = nn.ModuleList([AttModule_mamba(2 ** i, num_f_maps, num_f_maps, att_type, 'decoder', alpha, drop_path_rate=drop_path_rate) for i in range(num_layers)])
        #self.conv_out = nn.Conv1d(num_f_maps, output_dim, 1)

    def forward(self, x, fencoder, mask,positional_encoding):

        feature = self.conv_1x1(x)
#         feature = positional_encoding(feature)
        for layer in self.layers:
            feature = layer(feature, fencoder, mask)

        #out = self.conv_out(feature) * mask.unsqueeze(1)#* mask[:, 0:1, :]

        return feature

class MBA_tsm_ED(nn.Module):
    def __init__(self,input_dim, n_state=16,pos_embed_dim=16,num_filters=64,BI=True,num_feature_layers=5,num_encoder_layers=3,drop_path_rate=0.3,tsm_horizon=64,kernel_size_feature=3,kernel_size_mba=7,dropout_rate=0.2,tsm=False,mba=True,add_positional_encoding=True,max_seq_len=2400,pretraining=False,cls_token=False,num_heads=4,featurelayer="TCN",supcon_loss=False,encoder_bottelneck=1,num_classes=1):
        super(MBA_tsm_ED, self).__init__()
        self.add_positional_encoding=add_positional_encoding
        self.pretraining = pretraining
        self.cls_token = nn.Parameter(torch.randn(1, num_filters,1))
        self.pooling=cls_token
        self.feature_extractor = FeatureExtractor(tsm_horizon,input_dim,pos_embed_dim,num_filters=num_filters,kernel_size=kernel_size_feature,num_feature_layers=num_feature_layers,tsm=tsm,featurelayer=featurelayer)
        self.positional_encoding = PositionalEncoding(d_model=num_filters,max_len=max_seq_len)
        self.GatedAttentionMILPooling =GatedAttentionPoolingMIL(input_dim=num_filters, hidden_dim=16)   
        self.MultiHeadSelfAttentionPooling = MultiHeadSelfAttentionPooling(input_dim=num_filters, hidden_dim=num_filters, num_heads=num_heads)
        self.supcon_loss = supcon_loss
        

        self.encoder=MBA_tsm_ED_Encoder( num_encoder_layers,  num_filters, input_dim, encoder_bottelneck,'sliding_att', 1, drop_path_rate=drop_path_rate)
        self.decoder=MBA_tsm_ED_Decoder( num_encoder_layers, num_filters, encoder_bottelneck, num_filters,'sliding_att', 1, drop_path_rate=drop_path_rate)
#         self.decoders = nn.ModuleList([copy.deepcopy(Decoder(num_layers, r1, r2, num_f_maps, num_classes, num_classes, att_type='sliding_att', alpha=exponential_descrease(s))) for s in range(num_decoders)]) # num_decoders
        
        self.out1 = nn.Linear(num_filters, 1)  
        self.out2 = nn.Conv1d(num_filters, 1, 1)
        self.out3 = nn.Linear(num_filters, 1)
        
        self.out1_pretrain = nn.Conv1d(num_filters, 1, 1)
        self.out2_pretrain = nn.Conv1d(num_filters, 1, 1)
        self.out3_pretrain = nn.Conv1d(num_filters, 1, 1)
        
        if self.supcon_loss:
            self.proj = nn.Sequential(
                nn.Linear(num_filters, num_filters),
                nn.ReLU(),
                nn.Linear(num_filters, 128)
            )

    def forward(self, x, x_lengths,labels1=None, labels3=None, apply_mixup=False, mixup_alpha=0.2):
        #global xx,mask,mask2,lengths
        batch_size, seq_len, channels = x.size()       
        #x=self.feature_extractor(x.permute(0, 2, 1))
        contrastive_embedding = None
        if self.pretraining: #If pretraining, then predict x,y,z
            if self.add_positional_encoding:
                x = self.positional_encoding(x)
            mask = create_mask(torch.tensor(x_lengths,device=x.device), seq_len,batch_size,x.device)
            for layer in self.encoder:
                x = layer(x,None, mask)
            pooled_features,weighted_features,attention_weights=self.MultiHeadSelfAttentionPooling(x.permute(0, 2, 1), mask)
            mask=mask.reshape(-1).bool()
            x1 = self.out1_pretrain(x).view(-1)[mask]
            x2 = self.out2_pretrain(x).view(-1)[mask]
            x3= self.out3_pretrain(x).view(-1)[mask]
        else: 
            mask = create_mask(torch.tensor(x_lengths,device=x.device), seq_len,batch_size,x.device)
            
            out_encoder,feature_encoder = self.encoder(x.permute(0, 2, 1), mask,self.positional_encoding)
            feature_decoder= self.decoder(torch.sigmoid(out_encoder)* mask.unsqueeze(1), feature_encoder * mask.unsqueeze(1), mask,self.positional_encoding)

            if self.pooling :
                attention_weights = torch.zeros(batch_size, seq_len,1,device=x.device)
                pooled_features = feature_decoder[:,:,0]
            else:
                pooled_features,weighted_features,attention_weights=self.MultiHeadSelfAttentionPooling(feature_decoder.permute(0, 2, 1), mask)      
            if apply_mixup:
                mixed_features, mixed_labels1, mixed_labels3, mixup_lambda = latent_mixup(
                    pooled_features, labels1, labels3, alpha=mixup_alpha
                )
                mixup_info = (mixed_labels1, mixed_labels3, mixup_lambda)
            else:
                mixed_features = pooled_features
                mixup_info = None

            x1 = self.out1(mixed_features)
            x2 = self.out2(feature_decoder) 
            x3= self.out3(mixed_features)
            if self.supcon_loss:
                contrastive_embedding = self.proj(mixed_features)
            mask=mask.reshape(-1).bool()
            x2=x2.view(-1)
            x2=x2[mask]
            attention_weights=attention_weights.reshape(-1)
            attention_weights=attention_weights[mask]
        return x1,x2,x3,attention_weights,contrastive_embedding, mixup_info


# ==================== TSO Patch Models ====================
# These classes match the key names in checkpoints trained with the dev codebase.
# ConvFeedForwardDev uses self.conv / self.norm (not self.layer) to match those keys.

class ConvFeedForwardDev(nn.Module):
    """ConvFeedForward matching dev checkpoint key structure: .conv and .norm attributes."""
    def __init__(self, dilation, in_channels, out_channels, kernel_size, norm="BN"):
        super(ConvFeedForwardDev, self).__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_channels, out_channels, padding=padding,
                              dilation=dilation, kernel_size=kernel_size)
        self.activation = nn.ReLU(inplace=True)
        if norm == 'GN':
            self.norm = nn.GroupNorm(num_groups=4, num_channels=in_channels)
        elif norm == 'BN':
            self.norm = nn.BatchNorm1d(in_channels)
        elif norm == 'IN':
            self.norm = nn.InstanceNorm1d(in_channels, track_running_stats=False)
        else:
            self.norm = None

    def forward(self, x):
        out = self.conv(x)
        if self.norm is not None:
            out = self.norm(out)
        out = self.activation(out)
        return out


class AttModule_mamba_dev(nn.Module):
    """AttModule_mamba using ConvFeedForwardDev — matches dev checkpoint key structure."""
    def __init__(self, dilation, in_channels, out_channels, att_type, stage, alpha,
                 drop_path_rate=0.3, kernel_size=7, dropout_rate=0.2, norm="BN"):
        super(AttModule_mamba_dev, self).__init__()
        self.feed_forward = ConvFeedForwardDev(dilation, in_channels, out_channels,
                                               kernel_size=kernel_size, norm=norm)
        self.att_layer = MaskMambaBlock(out_channels, drop_path_rate=drop_path_rate,
                                        kernel_size=kernel_size)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        if norm == 'GN':
            self.norm = nn.GroupNorm(num_groups=4, num_channels=out_channels)
        elif norm == 'BN':
            self.norm = nn.BatchNorm1d(out_channels)
        else:
            self.norm = nn.InstanceNorm1d(out_channels, track_running_stats=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.alpha = alpha

    def forward(self, x, f, mask):
        out = self.feed_forward(x)
        out = self.alpha * self.att_layer(self.norm(out), mask) + out
        out = self.dropout(out)
        return (x + out) * mask.unsqueeze(1)


class PatchEmbedding(nn.Module):
    """
    Simple and lightweight patch embedding for raw sensor data.
    Maps [batch_size, seq_len, patch_size, channels] -> [batch_size, seq_len, num_filters]

    Uses 1D convolution + pooling for efficiency.
    Memory efficient: processes patches independently.
    """
    def __init__(self, patch_size=1200, in_channels=5, num_filters=64, reduction_factor=4):
        super(PatchEmbedding, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, num_filters // 2,
                               kernel_size=7, stride=reduction_factor, padding=3)
        self.bn1 = nn.BatchNorm1d(num_filters // 2)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(num_filters // 2, num_filters,
                               kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(num_filters)
        self.act2 = nn.ReLU(inplace=True)

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, patch_size, channels]
        Returns:
            out: [batch_size, seq_len, num_filters]
        """
        batch_size, seq_len, patch_size, channels = x.size()

        x = x.view(batch_size * seq_len, patch_size, channels)
        x = x.permute(0, 2, 1)  # [B*seq_len, channels, patch_size]

        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))

        x = self.pool(x).squeeze(-1)  # [B*seq_len, num_filters]

        x = x.view(batch_size, seq_len, -1)  # [B, seq_len, num_filters]
        return x


class MBA4TSO_Patch(nn.Module):
    """
    MBA4TSO variant for patched raw sensor input.

    Input: [batch_size, seq_len, patch_size, channels]
           - seq_len: number of minutes (e.g., 1440 for 24h)
           - patch_size: samples per minute (e.g., 1200 = 60s * 20Hz)
           - channels: 5 or 6 (x, y, z, temperature, time_sin[, time_cos])

    Output: [batch_size, seq_len, 3] (other, non-wear, predictTSO logits)

    Architecture:
    1. PatchEmbedding: patches -> embeddings
    2. FeatureExtractor: lightweight TCN/ResNet feature extraction
    3. Mamba Encoder: temporal modeling
    4. Output projection: 3-class prediction
    """
    def __init__(self,
                 patch_size=1200,
                 patch_channels=5,
                 num_filters=64,
                 num_feature_layers=3,
                 num_encoder_layers=3,
                 drop_path_rate=0.3,
                 kernel_size_feature=3,
                 kernel_size_mba=7,
                 dropout_rate=0.2,
                 add_positional_encoding=True,
                 max_seq_len=1440,
                 featurelayer="ResNet",
                 skip_connect=True,
                 skip_cross_attention=False,
                 norm1='BN',
                 norm2='IN',
                 output_channels=3):
        super(MBA4TSO_Patch, self).__init__()

        self.num_feature_layers = num_feature_layers
        self.add_positional_encoding = add_positional_encoding
        self.output_channels = output_channels
        self.skip_connect = skip_connect
        self.skip_cross_attention = skip_cross_attention

        # Patch embedding: [B, L, patch_size, channels] -> [B, L, num_filters]
        self.patch_embedding = PatchEmbedding(
            patch_size=patch_size,
            in_channels=patch_channels,
            num_filters=num_filters
        )

        # Lightweight feature extraction
        if num_feature_layers > 0:
            self.feature_extractor = FeatureExtractor(
                in_channels=num_filters,
                norm=norm1,
                kernel_size=kernel_size_feature,
                num_feature_layers=num_feature_layers,
                num_filters=num_filters,
                tsm=False,
                tsm_horizon=64,
                featurelayer=featurelayer,
            )
        else:
            self.feature_extractor = None

        # Positional encoding
        if add_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_model=num_filters, max_len=max_seq_len)

        # Mamba encoder layers
        if self.skip_connect and self.skip_cross_attention:
            self.encoder = nn.ModuleList([
                AttModule_mamba_cross(
                    2 ** i, num_filters, num_filters, 1, 1,
                    'sliding_att', 'encoder', 1,
                    drop_path_rate=drop_path_rate,
                    kernel_size=kernel_size_mba,
                    dropout_rate=dropout_rate
                ) for i in range(num_encoder_layers)
            ])
        else:
            self.encoder = nn.ModuleList([
                AttModule_mamba_dev(
                    2 ** i, num_filters, num_filters,
                    'sliding_att', 'encoder', 1,
                    drop_path_rate=drop_path_rate,
                    dropout_rate=dropout_rate,
                    kernel_size=kernel_size_mba,
                    norm=norm2
                ) for i in range(num_encoder_layers)
            ])

        # Output projection
        self.output_projection = nn.Conv1d(num_filters, output_channels, kernel_size=1)

    def forward(self, x, x_lengths):
        """
        Args:
            x: [batch_size, seq_len, patch_size, channels] patched input
            x_lengths: [batch_size] original sequence lengths (in minutes)

        Returns:
            output: [batch_size, seq_len, 3] class logits per minute
        """
        batch_size, seq_len, patch_size, channels = x.size()

        # Create padding mask
        if isinstance(x_lengths, torch.Tensor):
            x_lengths_tensor = x_lengths.clone().detach()
        else:
            x_lengths_tensor = torch.tensor(x_lengths, device=x.device)
        mask = create_mask(x_lengths_tensor, seq_len, batch_size, x.device)

        # Patch embedding: [B, L, patch_size, channels] -> [B, L, num_filters]
        x = self.patch_embedding(x)  # [B, seq_len, num_filters]

        # Permute for conv layers: [B, num_filters, seq_len]
        x = x.permute(0, 2, 1)

        # Feature extraction with optional skip connections
        if self.feature_extractor is not None:
            if self.skip_connect:
                x, feature_maps = self.feature_extractor(x, mask, return_intermediates=True)
            else:
                x = self.feature_extractor(x, mask)

            # Update mask if sequence length changed
            if x.size(2) != mask.size(1):
                new_seq_len = x.size(2)
                mask = F.interpolate(
                    mask.unsqueeze(1).float(),
                    size=new_seq_len,
                    mode='nearest'
                ).squeeze(1).bool()

        # Positional encoding
        if self.add_positional_encoding:
            x = self.positional_encoding(x)

        # Apply encoder layers with optional UNet-style skip connections
        if self.skip_connect and self.num_feature_layers > 0:
            for i, layer in enumerate(self.encoder):
                if self.skip_cross_attention:
                    if i < len(feature_maps):
                        skip_idx = len(feature_maps) - 1 - i
                        encoder_states = feature_maps[skip_idx]
                        x = layer(x, encoder_states, mask)
                    else:
                        x = layer(x, None, mask)
                else:
                    x = layer(x, x, mask)
                    if i < len(feature_maps):
                        skip_idx = len(feature_maps) - 1 - i
                        if x.size(2) != feature_maps[skip_idx].size(2):
                            skip_resized = F.interpolate(
                                feature_maps[skip_idx],
                                size=x.size(2),
                                mode='linear',
                                align_corners=False
                            )
                            x = x + skip_resized
                        else:
                            x = x + feature_maps[skip_idx]
        else:
            for layer in self.encoder:
                x = layer(x, x, mask)

        # Output projection
        output = self.output_projection(x)  # [B, output_channels, L]

        # Interpolate back to original sequence length if needed
        if output.size(2) != seq_len:
            output = F.interpolate(output, size=seq_len, mode='linear', align_corners=False)

        # Permute to [B, L, 3]
        output = output.permute(0, 2, 1)

        return output