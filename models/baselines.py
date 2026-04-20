# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from typing import Optional, Tuple
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np


def create_mask(original_lengths, max_length, batch_size, device):
    """Creates a binary mask: 1 for valid data, 0 for padding."""
    mask = torch.arange(max_length, device=device).unsqueeze(0).expand(batch_size, -1)
    mask = (mask < original_lengths.unsqueeze(1)).float()
    return mask


# Late import to avoid circular dependency (components → baselines → attention)
def _get_GatedAttentionPoolingMIL():
    from .attention import GatedAttentionPoolingMIL
    return GatedAttentionPoolingMIL


# Baseline and legacy models

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
class TCNLayer(nn.Module):
    """
    A single TCN layer with dilated convolutions.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding):
        super(TCNLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                               dilation=dilation, padding=padding)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x

# class ResTCNLayer(nn.Module):
#     """
#     A single TCN layer with dilated convolutions.
#     """
#     def __init__(self, in_channels, out_channels, kernel_size, dilation, padding,dropout_rate=0.2):
#         super(ResTCNLayer, self).__init__()
#         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding)
#         # self.norm = nn.BatchNorm1d(out_channels)
#         self.norm = nn.InstanceNorm1d(out_channels)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout_rate)

#     def forward(self, x):
#         residual = x
#         x = self.conv(x)
#         x = self.norm(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         if residual.shape == x.shape:
#             x = x + residual # Residual connection
#         return x
    
# class ResTCNLayer(nn.Module):
#     """
#     A single TCN layer with dilated convolutions.
#     """
#     def __init__(self, in_channels, out_channels, kernel_size, dilation, padding, norm="IN", dropout_rate=0.2):
#         super(ResTCNLayer, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
#         self.bn1 = nn.BatchNorm1d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
#         self.bn2 = nn.BatchNorm1d(out_channels)
#         # self.dropout = nn.Dropout(dropout_rate)

#     def forward(self, x):
#         residual = x
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         if residual.shape == out.shape:
#             out += residual  # residual connection
#         out = self.relu(out)
#         return out

class ResTCNLayer(nn.Module):
    """
    A single TCN layer with dilated convolutions.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding, norm="BN", dropout_rate=0.2):
        super(ResTCNLayer, self).__init__()
        # self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding)
        # if norm=='GN':
        #     self.norm = nn.GroupNorm(num_groups=4, num_channels=out_channels)
        # elif norm=='BN':
        #     self.norm = nn.BatchNorm1d(out_channels)
        # else:
        #     self.norm = nn.InstanceNorm1d(out_channels,track_running_stats=False)
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual  # residual connection
        out = self.relu(out)
        return out * mask.unsqueeze(1)
    
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
        self.GatedAttentionMILPooling =_get_GatedAttentionPoolingMIL()(input_dim=num_filters, hidden_dim=16)
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
        
    def forward(self, x, x_lengths, labels1=None, labels3=None,
                apply_mixup=False, mixup_alpha=0.2):
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
        x = self.tcn(x)

        mask = create_mask(torch.tensor(x_lengths,device=x.device), seq_len+1,batch_size,x.device)
        masked= self.GatedAttentionMILPooling(x.permute(0, 2, 1), mask)
        x1 = self.out1(masked)
        x2 = self.out2(x) #use .permute(0, 2, 1) if we want to use conv1d in output
        x3= self.out3(masked)

        x2=x2[:, :, 1:]# Output sequence excludes the classification token
        x2 = x2.permute(0, 2, 1) # [B, seq_len, 1]

        # Flatten and keep only valid positions (matching MBA_v1 convention)
        mask_flat = mask[:, 1:].reshape(-1).bool()
        x2 = x2.reshape(-1)[mask_flat]

        return x1, x2, x3, None, None, None

    
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

class tcn_learner(nn.Module): #Multi Task TCN
    def __init__(self, input_channels, embedding_dim=8,lin_embed_dim=64,  num_classes=1, num_layers=5, num_filters=64, kernel_size=3, max_dilation=16, dropout=0.2,BI=False):
        super(tcn_learner, self).__init__()
        layers = []
        dilation = 1
        self.GatedAttentionMILPooling =_get_GatedAttentionPoolingMIL()(input_dim=num_filters, hidden_dim=16)
        in_size=lin_embed_dim #+ embedding_dim
        for i in range(num_layers):
            padding = (kernel_size - 1) * dilation // 2  # To maintain same length
            layers.append(TCNLayer(in_size,num_filters, kernel_size, dilation, padding))
            in_size=num_filters
            dilation *= 2  # Increase dilation exponentially
        self.tcn_layers = nn.Sequential(*layers)
        
    def forward(self, x, x_lengths): 
        batch_size, seq_len, _ = x.size()
        x=x.permute(0, 2, 1) # Convert to (batch_size, input_dim, seq_len) for TCN processing
        x = self.tcn_layers(x)

        return x.permute(0, 2, 1) # change shape to batch,sequence,output



class mba_learner(nn.Module):
    def __init__(self,input_dim, n_state=16,pos_embed_dim=8,lin_embed_dim=64,BI=True,num_layers=2):
        super(mba_learner, self).__init__()
        
#         self.pos_embedding = PositionalEmbedding(max_seq_len, pos_embed_dim)
        self.embedding = nn.Linear(input_dim, lin_embed_dim)
        num_filters=lin_embed_dim#+pos_embed_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, num_filters))
        self.GatedAttentionMILPooling =_get_GatedAttentionPoolingMIL()(input_dim=num_filters, hidden_dim=16)
        self.mambablocks1 =BiMambaEncoder(num_filters, n_state,BI)
        self.mambablocks2 =BiMambaEncoder(num_filters, n_state,BI)
        self.mambablocks3 =BiMambaEncoder(num_filters, n_state,BI)
        self.mambablocks4 =BiMambaEncoder(num_filters, n_state,BI)

    def forward(self, x, x_lengths, max_seq_len=None):
        batch_size, seq_len, _ = x.size()
        mask = create_mask(torch.tensor(x_lengths,device=x.device), seq_len,batch_size,x.device)
        x = self.mambablocks3({0:x,1:mask})
        mamba_out = self.mambablocks4({0:x,1:mask})
        
        return mamba_out

class hybrid(nn.Module):
    def __init__(self,input_dim, n_state=16,pos_embed_dim=8,lin_embed_dim=64,BI=True,num_layers=2):
        super(hybrid, self).__init__()
        
#         self.pos_embedding = PositionalEmbedding(max_seq_len, pos_embed_dim)
        self.embedding = nn.Linear(input_dim, lin_embed_dim)
        num_filters=lin_embed_dim#+pos_embed_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, num_filters))
        self.GatedAttentionMILPooling =_get_GatedAttentionPoolingMIL()(input_dim=num_filters*2, hidden_dim=16)
        self.mba_learner =mba_learner(input_dim)
        self.tcn_learner =tcn_learner(input_dim)
        
        self.out1 = nn.Linear(num_filters*2, 1)
        self.out2 = nn.Linear(num_filters*2, 1) 
        self.out3 = nn.Linear(num_filters*2, 1)

    def forward(self, x, x_lengths, max_seq_len=None):
        batch_size, seq_len, _ = x.size()
        x=self.embedding(x)
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # Shape: (batch_size, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)
        x_mba=self.mba_learner(x, x_lengths)
        x_tcn=self.tcn_learner(x, x_lengths)# Convert to (batch_size, input_dim, seq_len) for TCN processing
        
        x_out=torch.cat((x_mba, x_tcn), dim=2)
        
        mask = create_mask(torch.tensor(x_lengths,device=x.device), seq_len+1,batch_size,x.device)
        masked= self.GatedAttentionMILPooling(x_out, mask)
        
        
        x1 = self.out1(masked)
        x2 = self.out2(x_out[:, 1:, :]) #use .permute(0, 2, 1) if we want to use conv1d in output
        x3= self.out3(masked)
        return x1,x2,x3


class BiLSTMMultiTask(nn.Module):
    """
    Bidirectional LSTM baseline with multi-task output heads.
    Matches the 6-value return interface expected by train_scratch.py:
        (out1, out2, out3, att, contrastive_embedding, mixup_info)
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
        max_seq_len: int = 1221,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        hidden_dim = hidden_size * 2  # bidirectional

        self.lstm = nn.LSTM(
            in_channels, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        # Attention pooling over time (CLS-like aggregation)
        self.attn_pool = nn.Linear(hidden_dim, 1)

        # Task heads
        self.out1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.out3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # Per-timestep mask prediction
        self.out2 = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def forward(self, x, x_lengths, labels1=None, labels3=None,
                apply_mixup=False, mixup_alpha=0.2):
        """
        Args:
            x: [B, L, C] padded input
            x_lengths: original sequence lengths (list / ndarray / tensor)
        Returns:
            out1: [B, 1] classification logit
            out2: [N_valid] flattened mask predictions for valid positions
            out3: [B, 1] regression value
            att, contrastive_embedding, mixup_info: None (unused)
        """
        batch_size, seq_len, _ = x.size()

        # Build validity mask  [B, L]
        lens = torch.as_tensor(x_lengths, dtype=torch.long, device=x.device)
        mask = torch.arange(seq_len, device=x.device).unsqueeze(0) < lens.unsqueeze(1)

        # Pack → LSTM → unpack
        packed = pack_padded_sequence(
            x, lens.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(
            lstm_out, batch_first=True, total_length=seq_len
        )
        # lstm_out: [B, L, hidden_dim]

        # Attention pooling for classification / regression
        attn_scores = self.attn_pool(lstm_out).squeeze(-1)        # [B, L]
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(1) # [B, 1, L]
        pooled = torch.bmm(attn_weights, lstm_out).squeeze(1)     # [B, hidden_dim]

        out1 = self.out1(pooled)   # [B, 1]
        out3 = self.out3(pooled)   # [B, 1]

        # Per-timestep mask prediction
        out2 = self.out2(lstm_out.permute(0, 2, 1)).squeeze(1)    # [B, L]

        # Flatten and keep only valid positions (matches MBA_v1 convention)
        out2 = out2.view(-1)[mask.view(-1)]

        return out1, out2, out3, None, None, None


