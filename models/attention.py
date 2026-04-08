# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from typing import Optional, Tuple
import numpy as np
from .mamba_blocks import ConvFeedForward, MaskMambaBlock


# Attention and pooling modules

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
        out = self.alpha * self.att_layer(self.norm(out),f, mask) + out
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask.unsqueeze(1)#[:, 0:1, :]

class AttModule_mamba(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, att_type, stage, alpha, drop_path_rate=0.3, kernel_size=7, dropout_rate=0.2, norm="BN"):
        super(AttModule_mamba, self).__init__()
        # old version without norm
        self.feed_forward = ConvFeedForward(dilation, in_channels, out_channels,kernel_size=kernel_size, norm=norm)
        self.att_layer = MaskMambaBlock(out_channels, drop_path_rate=drop_path_rate,kernel_size=kernel_size) # dilation
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        if norm=='GN':
            self.norm = nn.GroupNorm(num_groups=4, num_channels=out_channels)
        elif norm=='BN':
            self.norm = nn.BatchNorm1d(out_channels)
        else:
            self.norm = nn.InstanceNorm1d(out_channels, track_running_stats=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.alpha = alpha

    def forward(self, x, f, mask):
        m_batchsize, c1, L = x.size()        
        #padding_mask = torch.ones((m_batchsize, 1, L)).to(device) * mask[:,0:1,:]
        padding_mask = mask
        # out = self.feed_forward(x)
        out = self.feed_forward(x)
        # out = self.dropout(out)
        out = self.alpha * self.att_layer(self.norm(out), padding_mask) + out
        # out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask.unsqueeze(1)



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


class MaskedMaxAvgPooling(nn.Module):
    def __init__(self, input_dim, combination_type='concat'):
        """
        Efficient pooling that combines masked average and max pooling.
        Much more efficient than attention pooling and handles padding properly.

        Args:
            input_dim: The dimensionality of the input embeddings
            combination_type: How to combine max and avg pooling:
                - 'concat': Concatenate max and avg features (output dim = 2 * input_dim)
                - 'weighted': Weighted combination (output dim = input_dim)
                - 'avg_only': Only masked average pooling (output dim = input_dim)
                - 'max_only': Only max pooling (output dim = input_dim)
        """
        super(MaskedMaxAvgPooling, self).__init__()
        self.input_dim = input_dim
        self.combination_type = combination_type

        if combination_type == 'concat':
            self.output_dim = 2 * input_dim
        elif combination_type == 'weighted':
            self.output_dim = input_dim
            # Learnable weights for combining max and avg
            self.alpha = nn.Parameter(torch.tensor(0.5))  # Weight for max pooling
        else:
            self.output_dim = input_dim

    def forward(self, x, mask):
        """
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Padding mask [batch_size, seq_len] where 1 = valid, 0 = padded

        Returns:
            pooled_features: [batch_size, output_dim]
            weighted_features: x (unchanged for compatibility)
            attention_weights: mask (for compatibility)
        """
        batch_size, seq_len, input_dim = x.shape
        device = x.device

        # Ensure mask is float and has correct shape
        if mask.dim() == 2:
            mask_float = mask.float()  # [batch_size, seq_len]
        else:
            mask_float = mask.squeeze(-1).float()  # Remove last dim if present

        mask_expanded = mask_float.unsqueeze(-1)  # [batch_size, seq_len, 1]

        # Masked Average Pooling
        if self.combination_type in ['concat', 'weighted', 'avg_only']:
            # Zero out padded positions
            masked_x = x * mask_expanded  # [batch_size, seq_len, input_dim]
            # Sum over sequence length
            sum_features = masked_x.sum(dim=1)  # [batch_size, input_dim]
            # Count valid positions per batch
            valid_lengths = mask_float.sum(dim=1, keepdim=True)  # [batch_size, 1]
            # Avoid division by zero
            valid_lengths = torch.clamp(valid_lengths, min=1.0)
            # Compute average
            avg_features = sum_features / valid_lengths  # [batch_size, input_dim]

        # Max Pooling
        if self.combination_type in ['concat', 'weighted', 'max_only']:
            # Set padded positions to very negative values
            masked_x_max = x.clone()
            masked_x_max[mask_expanded.squeeze(-1) == 0] = -1e9
            # Max pooling
            max_features, _ = masked_x_max.max(dim=1)  # [batch_size, input_dim]

        # Combine pooling results
        if self.combination_type == 'concat':
            pooled_features = torch.cat([avg_features, max_features], dim=-1)
        elif self.combination_type == 'weighted':
            pooled_features = self.alpha * max_features + (1 - self.alpha) * avg_features
        elif self.combination_type == 'avg_only':
            pooled_features = avg_features
        elif self.combination_type == 'max_only':
            pooled_features = max_features
        else:
            raise ValueError(f"Unknown combination_type: {self.combination_type}")


        return pooled_features


class AttModule_mamba_causal(nn.Module):
    """Causal attention module for encoder-decoder architectures."""
    def __init__(self, dilation, in_channels, out_channels, att_type, stage, alpha, drop_path_rate=0.3, kernel_size=7, dropout_rate=0.2):
        super(AttModule_mamba_causal, self).__init__()
        self.feed_forward = ConvFeedForward(dilation, in_channels, out_channels, kernel_size=kernel_size)
        self.instance_norm = nn.InstanceNorm1d(in_channels, track_running_stats=False)
        self.att_layer = MaskMambaBlock(in_channels, drop_path_rate=drop_path_rate, kernel_size=kernel_size)
        self.conv_1x1 = nn.Conv1d(in_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.alpha = alpha

    def forward(self, x, causal_mask, padding_mask):
        m_batchsize, c1, L = x.size()
        if causal_mask is not None and padding_mask is not None:
            expanded_causal_mask = causal_mask.unsqueeze(0).expand(m_batchsize, -1, -1)
            combined_mask = padding_mask.unsqueeze(-1) * expanded_causal_mask
            mask = combined_mask[:, :, -1:].permute(0, 2, 1)
        else:
            mask = padding_mask.unsqueeze(1) if padding_mask is not None else None

        out = self.feed_forward(x)
        out = self.alpha * self.att_layer(self.norm(out), mask) + out
        out = self.dropout(out)

        if mask is not None:
            return (x + out) * mask
        else:
            return x + out


class AttModule_mamba_cross(nn.Module):
    """Cross-attention module for encoder-decoder architectures using Mamba blocks."""
    def __init__(self, dilation, in_channels, out_channels, r1, r2, att_type, stage, alpha, drop_path_rate=0.3, kernel_size=7, dropout_rate=0.2, norm="IN"):
        super(AttModule_mamba_cross, self).__init__()
        self.feed_forward = ConvFeedForward(dilation, in_channels, out_channels, kernel_size=kernel_size)
        if norm == 'GN':
            self.norm = nn.GroupNorm(num_groups=4, num_channels=out_channels)
        elif norm == 'BN':
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
        out = self.alpha * self.self_att_layer(self.norm(out), padding_mask) + out

        if encoder_states is not None:
            if encoder_states.shape[2] == out.shape[2]:
                cross_out = self.alpha * self.cross_att_layer(self.norm(encoder_states), padding_mask)
                out = out + cross_out
            else:
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
        out = self.alpha * self.self_att_layer(self.instance_norm(out), None) + out

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
