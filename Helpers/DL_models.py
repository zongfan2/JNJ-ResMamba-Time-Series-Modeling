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
from transformers import PatchTSTConfig, PatchTSTModel, ViTModel, ViTConfig, Swinv2Model, Swinv2Config
from torchvision import models
# from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils import weight_norm
# from pyts.image import GramianAngularField
from typing import Optional, Tuple

try:
    from .net.vit1d import ViT1D
except:
    from net.vit1d import ViT1D

def setup_model(model_name, input_tensor_size,max_seq_len, best_params, pretraining,  num_classes=1):
    """
        This function is responsible for initializing the model class from best_params
        Return: model object to be trained
    """
    print("Params Received in `setup_model`: ",
          [model_name, input_tensor_size, best_params, pretraining, num_classes])
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
        case "mbatsm":
            # dropout_rate = best_params['dropout']
            # drop_path_rate = best_params['droppath']
            # kernel_size_feature =  best_params["kernel_f"]
            # kernel_size_mba = best_params["kernel_MBA"]
            # num_MBA_blocks = best_params["blocks_MBA"]
            # num_feature_layers =best_params["num_feature_layers"]
            # cls_token=best_params["cls_token"]
            # MBA_encoder=best_params["MBA_encoder"]
            # num_heads=best_params["num_heads"]
            # featurelayer=best_params["featurelayer"]
            # num_filters = best_params['num_filters']
            # random_padding = best_params.get("random_padding", False)
            # average_mask = best_params.get("average_mask", False)
            # average_window_size = best_params.get("average_window_size", 20)
            # supcon_loss = best_params.get("supcon_loss", True)
            # reg_only = best_params.get("reg_only", False)
            # model = MBA_tsm(input_tensor_size,
            #                 num_feature_layers=num_feature_layers,
            #                 num_encoder_layers=num_MBA_blocks,
            #                 drop_path_rate =drop_path_rate,
            #                 kernel_size_feature=kernel_size_feature,
            #                 kernel_size_mba=kernel_size_mba,
            #                 dropout_rate=dropout_rate, 
            #                 max_seq_len= max_seq_len,
            #                 cls_token=cls_token,
            #                 mba=MBA_encoder,
            #                 num_heads=num_heads,
            #                 featurelayer=featurelayer,
            #                 num_filters=num_filters,
            #                 pretraining=pretraining, 
            #                 random_padding=random_padding,
            #                 average_mask=average_mask,
            #                 average_window_size=average_window_size,
            #                 supcon_loss=supcon_loss,
            #                 reg_only=reg_only)
            dropout_rate = best_params['dropout']
            drop_path_rate = best_params['droppath']
            kernel_size_feature =  best_params["kernel_f"]
            kernel_size_mba = best_params["kernel_MBA"]

            kernel_size_mba1 = best_params["kernel_MBA"]
            kernel_size_mba2 = best_params["kernel_MBA"]
            # num_feature_layers = trial.suggest_int("num_feature_layers", 1, 10, step=1)
            
            num_MBA_blocks = best_params["blocks_MBA"]
            num_MBA_blocks1 = best_params["blocks_MBA"]
            num_MBA_blocks2 = best_params["blocks_MBA"]
            dilation1=True
            dilation2=True
            
            num_feature_layers =best_params["num_feature_layers"]
            cls_token=best_params["cls_token"]
            MBA_encoder=True
            num_heads=best_params["num_heads"]
            featurelayer=best_params["featurelayer"]
            num_filters = best_params['num_filters']
            # norm=best_params['norm']
            channel_masking_rate=best_params['channel_masking_rate']
            skip_connect = best_params['skip_connect']
            skip_cross_attention = best_params.get('skip_connect_attention', False)
            norm1 = best_params.get("norm1", "IN")
            norm2 = best_params.get("norm2", "IN")
            norm3 = best_params.get("norm3", "BN")
            pooling_type = best_params.get("pooling_type", "attention")

            model = MBA_tsm(input_tensor_size,
                            num_feature_layers=num_feature_layers,
                            num_encoder1_layers=num_MBA_blocks1,
                            num_encoder2_layers=num_MBA_blocks2,
                            dilation1=dilation1,
                            dilation2=dilation2,
                            drop_path_rate =drop_path_rate,
                            kernel_size_feature=kernel_size_feature,
                            kernel_size_mba1=kernel_size_mba1,
                            kernel_size_mba2=kernel_size_mba2,
                            dropout_rate=dropout_rate, 
                            max_seq_len= max_seq_len,
                            cls_token=cls_token,
                            mba=MBA_encoder,
                            num_heads=num_heads,
                            num_filters=num_filters,
                            pretraining=pretraining,
                            channel_masking_rate=channel_masking_rate,
                            featurelayer=featurelayer,
                            skip_connect=skip_connect,
                            skip_cross_attention=skip_cross_attention,
                            norm1=norm1,
                            norm2=norm2,
                            norm3=norm3,
                            pooling_type=pooling_type,
                            )
        case "mba_padding":
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
            average_mask = best_params.get("average_mask", False)
            average_window_size = best_params.get("average_window_size", 20)
            supcon_loss = best_params.get("supcon_loss", True)
            reg_only = best_params.get("reg_only", False)
            use_feature_extractor = best_params.get("use_feature_extractor", True)
            model = MBA_tsm_with_padding(input_tensor_size,
                            num_feature_layers=num_feature_layers,
                            num_encoder_layers=num_MBA_blocks,
                            drop_path_rate =drop_path_rate,
                            kernel_size_feature=kernel_size_feature,
                            kernel_size_mba=kernel_size_mba,
                            dropout_rate=dropout_rate, 
                            max_seq_len= max_seq_len,
                            cls_token=cls_token,
                            mba=MBA_encoder,
                            num_heads=num_heads,
                            featurelayer=featurelayer,
                            num_filters=num_filters,
                            average_mask=average_mask,
                            average_window_size=average_window_size,
                            supcon_loss=supcon_loss,
                            reg_only=reg_only,
                            use_feature_extractor=use_feature_extractor)
        case "mba4tso":
            dropout_rate = best_params.get('dropout', 0.2)
            drop_path_rate = best_params.get('droppath', 0.3)
            kernel_size_feature = best_params.get("kernel_f", 3)
            kernel_size_mba = best_params.get("kernel_MBA", 7)
            num_encoder_layers = best_params.get("blocks_MBA", 3)
            num_feature_layers = best_params.get("num_feature_layers", 7)
            featurelayer = best_params.get("featurelayer", "TCN")
            num_filters = best_params.get('num_filters', 64)
            skip_connect = best_params.get("skip_connect", True)
            skip_cross_attention = best_params.get("skip_cross_attention", False)
            norm1 = best_params.get("norm1", "IN")
            norm2 = best_params.get("norm2", "IN")
            output_channels = best_params.get("output_channels", 3)
            model = MBA4TSO(
                input_dim=input_tensor_size,
                num_feature_layers=num_feature_layers,
                num_encoder_layers=num_encoder_layers,
                drop_path_rate=drop_path_rate,
                kernel_size_feature=kernel_size_feature,
                kernel_size_mba=kernel_size_mba,
                dropout_rate=dropout_rate,
                max_seq_len=max_seq_len,
                featurelayer=featurelayer,
                num_filters=num_filters,
                skip_connect=skip_connect,
                skip_cross_attention=skip_cross_attention,
                norm1=norm1,
                norm2=norm2,
                output_channels=output_channels
            )
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
        case "mbatsmed":
            model = MBA_tsm_ED(
                    input_dim=input_tensor_size) 
        case "hybrid":
            model = hybrid(
                    input_dim=input_tensor_size)
        case "bitcnt":
            model = BiTCNT(
                    input_dim=input_tensor_size)
        case "patchtst":
            config = PatchTSTConfig(
                cls_only=False,
                num_input_channels=3,
                context_length=max_seq_len,
                # PatchTST arguments
                patch_length=32,
                patch_stride=16,
                # Transformer architecture configuration
                num_hidden_layers=10,
                d_model=1024,
                num_attention_heads=8,
                share_embedding=True,
                channel_attention=False,
                ffn_dim=768,
                norm_type="batchnorm",
                norm_eps=1e-05,
                attention_dropout=0.0,
                positional_dropout=0.0,
                path_dropout=0.0,
                ff_dropout=0.0,
                bias=True,
                activation_function="gelu",
                positional_encoding_type="sincos",
                use_cls_token=False,
                scaling=None,
                # mask pretraining
                # do_mask_input=None,
                # mask_type="random",
                # random_mask_ratio=0.5,
                # mask_value=0,
                # head
                pooling_type="mean",
                head_dropout=0.1,
                prediction_length=max_seq_len,
            )
            model = PatchTSTNS(config, pretrained=False)
        case "efficient_unet":
            # num_feature_layers =best_params["num_feature_layers"]
            # featurelayer=best_params["featurelayer"]
            # num_filters = best_params.get('num_filters', "")
            model_name = "efficientnet_v2_m"
            # tsm = best_params["tsm"]
            d_model = best_params.get("d_model", 1024)
            target_shape = best_params.get("target_shape", 128)
            reg_head = best_params.get("reg_head", True)
            tcn_params = {
                          "tcn_num_filters": best_params.get("tcn_num_filters", 64), 
                          "tcn_kernel_size": best_params.get("tcn_kernel_size", 7), 
                          "tcn_num_blocks": best_params.get("tcn_num_blocks", 6),
                          }
            mask_padding = best_params.get("mask_padding", True)
            average_mask = best_params.get("average_mask", False)
            average_window_size = best_params.get("average_window_size", 20)
            model =  EfficientUNet(prediction_length=max_seq_len, 
                                        model_name=model_name, 
                                        d_model=d_model, 
                                        input_dim=3, 
                                        # tsm_horizon=64, 
                                        # pos_embed_dim=16, 
                                        # num_filters=num_filters,
                                        # kernel_size_feature=13,
                                        # num_feature_layers=num_feature_layers,
                                        # tsm=tsm,
                                        # featurelayer=featurelayer,
                                        target_shape=target_shape,
                                        tcn_params=tcn_params,
                                        reg_head=reg_head,
                                        mask_padding=mask_padding,
                                        average_mask=average_mask,
                                        average_window_size=average_window_size)
        case "vit":
            prediction_length = max_seq_len
            input_dim = 3 
            d_model = best_params.get("d_model", 1024)
            hidden_dropout_prob = 0.1
            load_pretrained = False
            target_shape = best_params.get("target_shape", 128)
            tcn_params = {
                          "tcn_num_filters": best_params.get("tcn_num_filters", 64), 
                          "tcn_kernel_size": best_params.get("tcn_kernel_size", 7), 
                          "tcn_num_blocks": best_params.get("tcn_num_blocks", 6),
                          }
            reg_head = False
            mask_padding = best_params.get("mask_padding", True)
            average_mask = best_params.get("average_mask", False)
            average_window_size = best_params.get("average_window_size", 20)

            model = ViT(prediction_length=prediction_length, 
                        input_dim=input_dim, 
                        d_model=d_model, 
                        hidden_dropout_prob=0.1,
                        load_pretrained=False,
                        target_shape=target_shape,
                        tcn_params=tcn_params,
                        reg_head=reg_head,
                        mask_padding=mask_padding,
                        average_mask=average_mask,
                        average_window_size=average_window_size)
        case "swin":
            prediction_length = max_seq_len
            input_dim = 3
            hidden_dropout_prob = 0.1
            load_pretrained = False
            target_shape = best_params.get("target_shape", 128)
            tcn_params = {
                          "tcn_num_filters": best_params.get("tcn_num_filters", 64), 
                          "tcn_kernel_size": best_params.get("tcn_kernel_size", 7), 
                          "tcn_num_blocks": best_params.get("tcn_num_blocks", 6),
                          }
            reg_head = False
            mask_padding = best_params.get("mask_padding", True)
            average_mask = best_params.get("average_mask", False)
            average_window_size = best_params.get("average_window_size", 20)

            model = SwinT(prediction_length=prediction_length, 
                        input_dim=input_dim,  
                        hidden_dropout_prob=0.1,
                        load_pretrained=False,
                        target_shape=target_shape,
                        tcn_params=tcn_params,
                        reg_head=reg_head,
                        mask_padding=mask_padding,
                        average_mask=average_mask,
                        average_window_size=average_window_size)
        case "vit1d":
            prediction_length = max_seq_len
            input_dim = 3
            patch_size = best_params.get("patch_size", 60)
            stride = best_params.get("stride", 10)
            embed_dim = best_params.get("embed_dim", None)
            encoder_layers = best_params.get("encoder_layers", 4)
            decoder_layers = best_params.get("decoder_layers", 0)
            num_heads = best_params.get("num_heads", 4)
            mlp_ratio = 4
            padding_value = -999.0
            padding_threshold = 0.3
            hidden_dropout_prob = 0.1
            # load_weights = best_params.get("load_weights", False)
            average_mask = best_params.get("average_mask", False)
            supcon_loss = best_params.get("supcon_loss", False)

            model = ViT1D(time_length=prediction_length, 
                          patch_size=patch_size,
                          stride=stride,
                          in_channels=input_dim,
                          embed_dim=embed_dim,
                          encoder_layers=encoder_layers,
                          decoder_layers=decoder_layers,
                          num_heads=num_heads,
                          mlp_ratio=mlp_ratio,
                          dropout=hidden_dropout_prob,
                          padding_value=padding_value,
                          padding_threshold=padding_threshold,
                          average_mask=average_mask,
                          supcon_loss=supcon_loss,
                          use_embedding=True
                          )
        case "conv1dts":
            prediction_length = max_seq_len
            input_dim = 3
            num_filters = best_params.get("d_model", 64)
            # kernel_size = 15
            kernel_size = 7
            num_layers = best_params.get("num_layers", 4)
            second_level_mask = best_params.get("average_mask", False)
            supcon_loss = best_params.get("supcon_loss", False)
            model = Conv1DTS(
                time_length=prediction_length,
                in_channels=input_dim,
                base_filters=num_filters,
                num_layers=num_layers,
                kernel_size=kernel_size,
                dropout=0.1,
                second_level_mask=second_level_mask,
                supcon_loss=supcon_loss
            )
        
        case "resnet18_1d":
            prediction_length = max_seq_len
            input_dim = 3
            second_level_mask = best_params.get("average_mask", False)
            supcon_loss = best_params.get("supcon_loss", False)
            layers = [2, 2, 2, 2]
            # layers = [3, 4, 6]
            block_type = 'basic'
            model = ResNet1D(time_length=prediction_length, 
                    in_channels=input_dim, 
                    kernel_size=7,
                    block_type=block_type, 
                    layers=layers,
                    num_classes=num_classes, 
                    second_level_mask=second_level_mask,
                    supcon_loss = supcon_loss,
                    )
        case "resnet50_1d":
            prediction_length = max_seq_len
            input_dim = 3
            second_level_mask = best_params.get("average_mask", False)
            supcon_loss = best_params.get("supcon_loss", False)
            layers = [3, 4, 6, 3]
            block_type = 'bottleneck'
            model = ResNet1D(time_length=prediction_length, 
                    in_channels=input_dim, 
                    kernel_size=7,
                    block_type=block_type, 
                    layers=layers,
                    num_classes=num_classes, 
                    second_level_mask=second_level_mask,
                    supcon_loss = supcon_loss,
                    )
        case "feature_extractor":
            # Extract parameters for FeatureExtractor setup
            input_dim = input_tensor_size if input_tensor_size else 3
            num_filters = best_params.get("num_filters", 64)
            num_feature_layers = best_params.get("num_feature_layers", 7)
            kernel_size = best_params.get("kernel_f", 13)
            tsm_horizon = best_params.get("tsm_horizon", 64)
            featurelayer = best_params.get("featurelayer", "ResNet")
            tsm = best_params.get("tsm", False)
            pos_embed_dim = best_params.get("pos_embed_dim", 16)
            use_cls_token = best_params.get("use_cls_token", True)
            
            # Create FeatureExtractorForPretraining for contrastive pretraining
            model = FeatureExtractorForPretraining(
                tsm_horizon=tsm_horizon,
                in_channels=input_dim,
                pos_embed_dim=pos_embed_dim,
                kernel_size=kernel_size,
                num_feature_layers=num_feature_layers,
                num_filters=num_filters,
                tsm=tsm,
                featurelayer=featurelayer,
                use_cls_token=use_cls_token,
            )
        case "mba_encoder_decoder":
            dropout_rate = best_params['dropout']
            drop_path_rate = best_params['droppath']
            kernel_size_mba = best_params["kernel_MBA"]
            num_MBA_blocks = best_params["blocks_MBA"]
            cls_token=best_params["cls_token"]
            MBA_encoder=best_params["MBA_encoder"]
            num_heads=best_params["num_heads"]
            num_filters = best_params['num_filters']
            supcon_loss = best_params.get("supcon_loss", True)
            cls_src = best_params.get("cls_src", "encoder")
            # seq_reduction = best_params.get("seq_reduction", 8)

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
            elif net_type == "ch_bottleneck":
                net = MBA_tsm_encoder_decoder_ch_bottleneck
            elif net_type == "seq_bottleneck":
                net = MBA_tsm_encoder_decoder_seq_bottleneck
            else:
                assert "net_type must be 'normal', 'progressive', or 'progressive_with_skip_connection' when using mba_encoder_decoder"

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
                            supcon_loss=supcon_loss,
                            cls_src=cls_src)
        case "mba_patch":
            dropout_rate = best_params['dropout']
            drop_path_rate = best_params['droppath']
            kernel_size_mba = best_params["kernel_MBA"]
            num_MBA_blocks = best_params["blocks_MBA"]
            cls_token = best_params["cls_token"]
            num_heads = best_params["num_heads"]
            average_window_size = best_params.get("average_window_size", 20)
            supcon_loss = best_params.get("supcon_loss", False)
            reg_only = best_params.get("reg_only", False)
            window_size = best_params.get("window_size", 60)
            stride = best_params.get("stride", 20)
            use_embedding = best_params.get("use_embedding", True)
            embed_dim = best_params.get("embed_dim", 192)
            
            model = MBA_patch(
                in_channels=input_tensor_size,
                max_seq_len=max_seq_len,
                window_size=window_size,
                stride=stride,
                embed_dim=embed_dim,
                use_embedding=use_embedding,
                num_encoder_layers=num_MBA_blocks,
                drop_path_rate=drop_path_rate,
                kernel_size_mba=kernel_size_mba,
                dropout_rate=dropout_rate,
                cls_token=cls_token,
                num_heads=num_heads,
                supcon_loss=supcon_loss,
                reg_only=reg_only
            )
        
        case "resnetunet":
            model = ResNetUNet(in_channels=input_tensor_size)
        case "resnetmambaunet":
            model = ResNetMambaUNet(in_channels=input_tensor_size, use_mask_in_decoder=True)
        case "mba_encoder":
            dropout_rate = best_params['dropout']
            drop_path_rate = best_params['droppath']
            kernel_size_mba = best_params["kernel_MBA"]
            num_MBA_blocks = best_params["blocks_MBA"]
            num_heads = best_params["num_heads"]
            num_filters = best_params['num_filters']
            use_cls_token = best_params.get("use_cls_token", True)
            # projection_dim = best_params.get("projection_dim", 128)
            add_positional_encoding = best_params.get("add_positional_encoding", True)
            norm = best_params.get("norm", "BN")
            
            model = MBATSMForPretraining(
                input_dim=input_tensor_size,
                num_filters=num_filters,
                num_encoder_layers=num_MBA_blocks,
                drop_path_rate=drop_path_rate,
                kernel_size_mba=kernel_size_mba,
                dropout_rate=dropout_rate,
                mba=True,  # Always use Mamba for pretraining
                add_positional_encoding=add_positional_encoding,
                max_seq_len=max_seq_len,
                num_heads=num_heads,
                use_cls_token=use_cls_token,
                pretraining=pretraining,  # Pass pretraining mode
                # projection_dim=projection_dim,
                norm=norm
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

    
class ConvFeedForward(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, kernel_size, norm="BN"):
        super(ConvFeedForward, self).__init__()
        padding = (kernel_size - 1) * dilation // 2 
        self.conv=nn.Conv1d(in_channels, out_channels, padding=padding, dilation=dilation, kernel_size=kernel_size)
        self.activation = nn.ReLU(inplace=True)
        if norm=='GN':
            self.norm = nn.GroupNorm(num_groups=4, num_channels=in_channels)
        elif norm=='BN':
            self.norm = nn.BatchNorm1d(in_channels)
        elif norm=='IN':
            self.norm = nn.InstanceNorm1d(in_channels,track_running_stats=False)
        else:
            self.norm = None

    def forward(self, x):
        out = self.conv(x)
        if self.norm is not None:
            out = self.norm(out)
        out = self.activation(out)
        return out

class ConvFeedForward_cls(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(ConvFeedForward_cls, self).__init__() 
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation),
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
        x = x_ * mask.unsqueeze(1).to(x.dtype)
        x  = res + self.drop_path(x)
        if self.downsample is not None:
            x, mask = self.downsample(x, mask)

        return  x

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        if self.chomp_size == 0:
            return x
        else:
            return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.comp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.comp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.comp1, self.relu1, self.dropout1, 
                                 self.conv2, self.comp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        # self.relu = nn.ReLU()

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        out = self.net(x)
        return out + res


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.module = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            # nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(),
        )
        
    def forward(self, x):
        return self.module(x)



class FeatureExtractorConv2d(nn.Module):
    def __init__(self, in_channels, seq_len, target_shape, tcn_num_filters=64, tcn_kernel_size=7, tcn_num_blocks=4, **kwargs):
        # transform an input from (N, in_channels, seq_len) to (N, 3, target_shape, target_shape)
        super(FeatureExtractorConv2d, self).__init__()
        self.target_shape = target_shape
        tcn_layers = []
        n_inputs = in_channels
        for i in range(tcn_num_blocks):
            dilation = 2 ** i
            tcn_layers.append(TemporalBlock(n_inputs, tcn_num_filters, tcn_kernel_size, stride=1, 
                                            dilation=dilation, padding=(tcn_kernel_size-1)*dilation))
            n_inputs = tcn_num_filters
        self.tcn_encoder = nn.Sequential(*tcn_layers)
        
        
        # NOTE: flatten + linear discard the temporal information, replace it with pooling
        # in_channels = tcn_num_filters
        # OPTION1: linear + upsample layers
        # self.linear = nn.Sequential(
        #     nn.Linear(in_channels*seq_len, 128*8*8),
        #     nn.ReLU(),
        #     # nn.Dropout(0.2)
        #     )
        
        # OPTION2: 1D-to-2D reshape 
        # self.project = nn.Conv1d(tcn_num_filters, target_shape, kernel_size=1)
        self.project_only = (tcn_num_filters == in_channels)
        if self.project_only: 
            # direct upsample to target input size
            self.project = nn.Sequential(
                nn.Upsample(size=(target_shape, target_shape), mode="bilinear", align_corners=False),
                nn.Conv2d(tcn_num_filters, 3, kernel_size=3, padding=1),
                nn.BatchNorm2d(3))
        else:
            self.project = nn.Upsample(size=(32, 32), mode="bilinear", align_corners=False)
            if target_shape == 64:
                block_channels = [64]
            elif target_shape == 128:
                block_channels = [64, 32]
            elif target_shape == 256:
                block_channels = [64, 32, 16]
            else:
                raise ValueError("Target shape must be 64, 128 or 256")

            upsample_blocks = []
            for i in range(len(block_channels)-1):
                # upsample_blocks.append(nn.ConvTranspose2d(block_channels[i], block_channels[i+1], kernel_size=4, stride=2, padding=1, bias=False, ))
                # upsample_blocks.append(nn.BatchNorm2d(block_channels[i+1]))
                # upsample_blocks.append(nn.ReLU())
                upsample_blocks.append(UpsampleBlock(block_channels[i], block_channels[i+1]))
            upsample_blocks.append(nn.ConvTranspose2d(block_channels[-1], 3, kernel_size=4, stride=2, padding=1, bias=False))
            # use tanh
            # upsample_blocks.append(nn.Tanh())
            self.upsample = nn.Sequential(*upsample_blocks)

    def forward(self, x, x_lengths=None):
        x = self.tcn_encoder(x) # (N, tcn_num_filters, 1221)
        x = x.view(x.size(0), x.size(1), 33, 37) # 37*37=1221
        x = self.project(x)     # (N, tcn_num_filters, 32, 32)
        if not self.project_only:
            x = self.upsample(x)
        return x 

class FeatureExtractor(nn.Module):
    def __init__(self,
                 tsm_horizon,
                 in_channels,
                 pos_embed_dim,
                 norm="BN",
                 num_pos_channels=4,
                 kernel_size=3,
                 num_feature_layers=5,
                 num_filters=64,
                 tsm=False,
                 featurelayer="TCN"):
        super(FeatureExtractor, self).__init__()

        if pos_embed_dim != num_pos_channels:
            self.extract_pos_embed = True 
        else:
            self.extract_pos_embed = False
        layers=[]
        dilation = 1
        in_size=in_channels
        for i in range(num_feature_layers):
            # if i < 4:
            #     cur_num_filters = num_filters // 2
            # else:
            cur_num_filters = num_filters
            padding = (kernel_size - 1) * dilation // 2  # To maintain same length
            if featurelayer=="TCN":
                layers.append(TCNLayer(in_size, cur_num_filters, kernel_size, dilation, padding, norm))
            elif featurelayer=="ResNet":
                padding = (kernel_size - 1) * 1 // 2
                layers.append(ResTCNLayer(in_size, cur_num_filters, kernel_size, 1, padding, norm))
            elif featurelayer=="ResTCN":
                layers.append(ResTCNLayer(in_size, cur_num_filters, kernel_size, dilation, padding, norm))
            in_size = cur_num_filters
            dilation *= 2  # Increase dilation exponentially
        self.layers = nn.ModuleList(layers)
        self.tsm = tsm
        self.tsm_horizon = tsm_horizon

    def forward(self, x, mask, return_intermediates=False):
        """
        Forward pass with optional intermediate feature maps for UNet-style skip connections.
        
        Args:
            x: Input tensor
            mask: Input mask
            return_intermediates: If True, returns (x, intermediate_features), 
                                 else returns x only
        """
        intermediate_features = []
        
        for i, layer in enumerate(self.layers):
            x = layer(x, mask)
            intermediate_features.append(x.clone())
        
        if self.tsm:
            x = self.compute_crosscorrelation_batch(x)
        if return_intermediates:
            return x, intermediate_features
        return x    

    
    def compute_crosscorrelation_batch(self, features):
        """
        Compute the autocorrelation of the feature map.
        """ 
        batch_size, channels, length = features.size()
        unit = self.tsm_horizon * 16
        h = self.tsm_horizon
        device = features.device
        TSM = torch.zeros(batch_size, h, length, device=device)
        for i in range(batch_size):
            tsm_list=[]
            for b in range(0, length, unit):
                autocorr = torch.zeros(1, unit + h, unit + h, device=device)
                b_size = min(length - (b), unit + h)
                autocorr[0, :b_size, :b_size] = torch.corrcoef(features[i, :, b:b + b_size].T)[:, :]        
                mat = autocorr[0, :b_size, :b_size].squeeze(-0)
                mask_value = -1111
                row_indices = torch.arange(b_size + h, device=device).unsqueeze(1).repeat(1, b_size + h)  
                col_indices = torch.arange(b_size + h, device=device).unsqueeze(0).repeat(b_size + h, 1) 
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

class FeatureExtractorForPretraining(nn.Module):
    """
    Wrapper class that combines FeatureExtractor with specific configurations for 
    contrastive self-supervised pretraining (SimCLR, DINO, etc.). This class ensures 
    the feature extractor can be frozen during downstream fine-tuning while maintaining compatibility.
    """
    def __init__(self,
                 tsm_horizon=64,
                 in_channels=3,
                 pos_embed_dim=16,
                 norm="BN",
                 num_pos_channels=4,
                 kernel_size=3,
                 num_feature_layers=5,
                 num_filters=64,
                 tsm=False,
                 featurelayer="ResNet",
                 use_cls_token=True,
                #  projection_dim=128
                 ):
        super().__init__()
        
        self.input_projection = ConvProjection(in_channels, num_filters, norm)
        # Create the feature extractor with updated signature
        self.feature_extractor = FeatureExtractor(
            tsm_horizon=tsm_horizon,
            in_channels=num_filters,
            pos_embed_dim=pos_embed_dim,
            norm=norm,
            num_pos_channels=num_pos_channels,
            kernel_size=kernel_size,
            num_feature_layers=num_feature_layers,
            num_filters=num_filters,
            tsm=tsm,
            featurelayer=featurelayer
        )
        
        self.use_cls_token = use_cls_token
        self.num_filters = num_filters
        # self.projection_dim = projection_dim
        
        # Always create the pooling layer
        self.pooling = MultiHeadSelfAttentionPooling(
            input_dim=num_filters, hidden_dim=num_filters, num_heads=8
        )
        
        if self.use_cls_token:
            # CLS token with num_filters dimensions (after input_projection)
            self.cls_token = nn.Parameter(torch.randn(1, 1, num_filters))
        else:
            self.cls_token = None
        
        # Projection head for contrastive learning
        # self.projection_head = nn.Sequential(
        #     nn.Linear(num_filters, num_filters),
        #     nn.ReLU(),
        #     nn.Linear(num_filters, projection_dim)
        # )
        
        # Store configuration for downstream model initialization
        self.config = {
            'tsm_horizon': tsm_horizon,
            'in_channels': in_channels,
            'pos_embed_dim': pos_embed_dim,
            'norm': norm,
            'num_pos_channels': num_pos_channels,
            'kernel_size': kernel_size,
            'num_feature_layers': num_feature_layers,
            'num_filters': num_filters,
            'tsm': tsm,
            'featurelayer': featurelayer,
            'use_cls_token': use_cls_token,
            # 'projection_dim': projection_dim
        }
    
    def forward(self, x, x_lengths, **kwargs):
        """
        Forward pass for contrastive pretraining.
        
        Args:
            x: Input tensor [batch_size, seq_len, channels]
            x_lengths: Original lengths of each sequence in the batch (before padding)
        
        Returns:
            If return_features and return_projection: (sequence_features, pooled_features, projected_features)
        """

        batch_size, seq_len, _ = x.size()
        # Create mask based on x_lengths
        mask = create_mask(torch.tensor(x_lengths, device=x.device), seq_len, batch_size, x.device)
        
        # Apply input projection first
        x_projected = self.input_projection(x.permute(0, 2, 1))  # [batch, num_filters, seq]
        
        # Add CLS token after input projection (when we have richer num_filters dimensions)
        if self.use_cls_token:
            # Add CLS token after input_projection (better than 3-channel raw input)
            batch_size, num_filters, seq_len = x_projected.size()
            
            # Add CLS token with num_filters dimensions
            cls_tokens = self.cls_token.expand(batch_size, -1, -1).permute(0, 2, 1)  # [batch, num_filters, 1]
            x_with_cls = torch.cat([cls_tokens, x_projected], dim=2)  # [batch, num_filters, seq+1]
            
            # Create extended mask for CLS token + sequence (CLS token is always valid)
            cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=mask.device)
            extended_mask = torch.cat([cls_mask, mask], dim=1)  # [batch, seq+1]
            
            # Extract features using the feature extractor (CLS token learns through the network)
            features = self.feature_extractor(x_with_cls, extended_mask)
            # Extract the CLS token embedding (position 0, has learned through feature extractor)
            cls_embedding = features[:, :, 0]  # [batch, num_filters]
        else:
            # Standard approach without CLS token
            features = self.feature_extractor(x_projected, mask)
            
            # Apply MultiHeadSelfAttentionPooling
            features_transposed = features.permute(0, 2, 1)  # [batch, seq, num_filters]
            pooled_features, weighted_features, attention_weights = self.pooling(
                features_transposed, mask
            )
            cls_embedding = pooled_features  # Use pooled features as representation
        
        # Apply projection head for contrastive learning on CLS embedding
        # proj_features = self.projection_head(cls_embedding)  # (batch_size, projection_dim)
        # return proj_features
        return cls_embedding

    def freeze_features(self):
        """Freeze the feature extractor parameters for downstream fine-tuning."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def unfreeze_features(self):
        """Unfreeze the feature extractor parameters."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
    
    def load_pretrained_features(self, checkpoint_path):
        """Load pretrained feature extractor weights."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'feature_extractor' in checkpoint:
            self.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        elif 'backbone' in checkpoint:
            if isinstance(checkpoint['backbone'], dict):
                if 'feature_extractor' in checkpoint['backbone']:
                    self.feature_extractor.load_state_dict(checkpoint['backbone']['feature_extractor'])
                else:
                    self.feature_extractor.load_state_dict(checkpoint['backbone'])
            else:
                self.feature_extractor.load_state_dict(checkpoint['backbone'].state_dict())
        elif 'model' in checkpoint:
            # Handle cases where the entire model is saved
            if 'feature_extractor' in checkpoint['model']:
                self.feature_extractor.load_state_dict(checkpoint['model']['feature_extractor'])
            else:
                # Try to load compatible weights
                self.load_state_dict(checkpoint['model'], strict=False)
        else:
            # Assume the entire checkpoint is the feature extractor state dict
            self.feature_extractor.load_state_dict(checkpoint)
    

class MBATSMForPretraining(nn.Module):
    """
    MBA-TSM variant designed specifically for pretraining the Mamba encoders.
    This class removes the feature extractor and focuses on pretraining the Mamba blocks directly.
    When use_cls_token is True, uses the first token instead of adding a new CLS token.
    """
    def __init__(self,
                 input_dim=3,
                 num_filters=64,
                 num_encoder_layers=3,
                 drop_path_rate=0.3,
                 kernel_size_mba=7,
                 dropout_rate=0.2,
                 mba=True,
                 add_positional_encoding=True,
                 max_seq_len=2400,
                 num_heads=4,
                 use_cls_token=True,
                 pretraining=True,
                #  projection_dim=128,
                 norm="BN"):
        super(MBATSMForPretraining, self).__init__()
        
        self.add_positional_encoding = add_positional_encoding
        self.use_cls_token = use_cls_token
        self.num_filters = num_filters
        self.pretraining = pretraining
        # self.projection_dim = projection_dim
        
        # Input projection to convert raw input to num_filters dimensions
        self.input_projection = ConvProjection(input_dim, num_filters, norm)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model=num_filters, max_len=max_seq_len)
        
        # Mamba encoder layers - this is what we want to pretrain
        if not mba:
            self.encoder = nn.ModuleList([
                AttModule(2 ** i, num_filters, num_filters, 2, 2, 'normal_att', 'encoder', alpha=1,
                         kernel_size=kernel_size_mba, dropout_rate=dropout_rate) 
                for i in range(num_encoder_layers)
            ])
        else:
            self.encoder = nn.ModuleList([
                AttModule_mamba(2 ** i, num_filters, num_filters, 'sliding_att', 'encoder', 1, 
                               drop_path_rate=drop_path_rate, dropout_rate=dropout_rate, 
                               kernel_size=kernel_size_mba, norm=norm) 
                for i in range(num_encoder_layers)
            ])
        
        # Pooling for sequence-level representation
        self.pooling = MultiHeadSelfAttentionPooling(
            input_dim=num_filters, hidden_dim=num_filters, num_heads=num_heads
        )
        
        # Prediction heads for fine-tuning (following MBA_tsm design)
        if not pretraining:
            self.out1 = OutModule2(num_filters, num_filters, 1, norm, 'FC')  # Binary classification
            self.out2 = OutModule2(num_filters, num_filters, 1, norm, 'Conv')  # Sequence prediction
            self.out3 = OutModule2(num_filters, num_filters, 1, norm, 'FC')  # Regression/severity
        
        # Projection head for contrastive learning
        # self.projection_head = nn.Sequential(
        #     nn.Linear(num_filters, num_filters),
        #     nn.ReLU(),
        #     nn.Linear(num_filters, projection_dim)
        # )
        
        # Store configuration for downstream model initialization
        self.config = {
            'input_dim': input_dim,
            'num_filters': num_filters,
            'num_encoder_layers': num_encoder_layers,
            'drop_path_rate': drop_path_rate,
            'kernel_size_mba': kernel_size_mba,
            'dropout_rate': dropout_rate,
            'mba': mba,
            'add_positional_encoding': add_positional_encoding,
            'max_seq_len': max_seq_len,
            'num_heads': num_heads,
            'use_cls_token': use_cls_token,
            # 'projection_dim': projection_dim,
            'norm': norm
        }
    
    def forward(self, x, x_lengths, **kwargs):
        """
        Forward pass for both pretraining and fine-tuning.
        
        Args:
            x: Input tensor [batch_size, seq_len, channels]
            x_lengths: Original lengths of each sequence in the batch (before padding)
        
        Returns:
            If pretraining=True: pooled_features (for contrastive learning)
            If pretraining=False: (x1, x2, x3, attention_weights, None, None) following MBA_tsm format
        """
        batch_size, seq_len, _ = x.size()
        
        # Create mask based on x_lengths
        mask = create_mask(torch.tensor(x_lengths, device=x.device), seq_len, batch_size, x.device)
        
        # Apply input projection: [batch, seq, input_dim] -> [batch, input_dim, seq] -> [batch, num_filters, seq]
        x_projected = self.input_projection(x.permute(0, 2, 1))  # [batch, num_filters, seq]
        
        # Add positional encoding if enabled
        if self.add_positional_encoding:
            x_projected = self.positional_encoding(x_projected)
        
        # Apply Mamba encoder layers
        features = x_projected
        for layer in self.encoder:
            features = layer(features, None, mask)  # [batch, num_filters, seq]
        
        # Convert to [batch, seq, num_filters] for pooling
        features_transposed = features.permute(0, 2, 1)
        
        if self.pretraining:
            # Pretraining mode: return pooled features for contrastive learning
            if self.use_cls_token:
                # Use the first token as the CLS token (already learned through the network)
                cls_embedding = features_transposed[:, 0, :]  # [batch, num_filters]
                pooled_features = cls_embedding
            else:
                # Use attention pooling for sequence-level representation
                pooled_features, weighted_features, attention_weights = self.pooling(
                    features_transposed, mask
                )
            return pooled_features
        else:
            # Fine-tuning mode: return predictions following MBA_tsm format
            if self.use_cls_token:
                # Extract CLS token for classification tasks
                cls_embedding = features_transposed[:, 0, :]  # [batch, num_filters]
                  # [batch, seq-1, num_filters]
                pooled_features = cls_embedding
                attention_weights = torch.zeros(batch_size, seq_len, 1, device=x.device)  # Placeholder
            else:
                pooled_features, weighted_features, attention_weights = self.pooling(
                    features_transposed, mask
                )
            
            # Generate outputs
            x1 = self.out1(pooled_features)  # Binary classification
            x3 = self.out3(pooled_features)  # Regression/severity
            
            # For sequence-level predictions
            sequence_features_conv = features_transposed.permute(0, 2, 1)  # [batch, num_filters, seq]
            x2 = self.out2(sequence_features_conv)  # [batch, 1, seq]
            
            # Apply masking to x2 outputs
            mask_flat = mask.reshape(-1).bool()
            x2_flat = x2.view(-1)
            x2_masked = x2_flat[mask_flat]
            attention_weights_flat = attention_weights.reshape(-1)
            attention_weights_masked = attention_weights_flat[mask_flat]
            
            return x1, x2_masked, x3, attention_weights_masked, None, None
    
    def get_encoder_weights(self):
        """
        Get the encoder weights for transfer to downstream models.
        """
        return {
            'input_projection': self.input_projection.state_dict(),
            'encoder': self.encoder.state_dict(),
            'positional_encoding': self.positional_encoding.state_dict()
        }
    
    def load_pretrained_encoder(self, checkpoint_path):
        """
        Load pretrained encoder weights from checkpoint.
        Supports both direct encoder checkpoints and DINO/contrastive learning checkpoints.
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'encoder_weights' in checkpoint:
            # Direct encoder weights saved during pretraining
            weights = checkpoint['encoder_weights']
            self.input_projection.load_state_dict(weights['input_projection'])
            self.encoder.load_state_dict(weights['encoder'])
            self.positional_encoding.load_state_dict(weights['positional_encoding'])
        elif 'student.input_projection.conv.weight' in checkpoint:
            # DINO or other contrastive learning checkpoint - extract student weights
            student_weights = {}
            for key, value in checkpoint.items():
                if key.startswith('student.'):
                    # Remove 'student.' prefix
                    new_key = key[8:]  # len('student.') = 8
                    student_weights[new_key] = value
            
            # Extract encoder components
            input_proj_weights = {k.replace('input_projection.', ''): v 
                                for k, v in student_weights.items() 
                                if k.startswith('input_projection.')}
            encoder_weights = {k.replace('encoder.', ''): v 
                             for k, v in student_weights.items() 
                             if k.startswith('encoder.')}
            pos_enc_weights = {k.replace('positional_encoding.', ''): v 
                             for k, v in student_weights.items() 
                             if k.startswith('positional_encoding.')}
            
            # Load the weights
            self.input_projection.load_state_dict(input_proj_weights)
            self.encoder.load_state_dict(encoder_weights)
            self.positional_encoding.load_state_dict(pos_enc_weights)
        else:
            # Try to load compatible weights (fallback)
            self.load_state_dict(checkpoint, strict=False)
    
    def set_mode(self, pretraining=True):
        """
        Switch between pretraining and fine-tuning modes.
        
        Args:
            pretraining: If True, model is in pretraining mode. If False, fine-tuning mode.
        """
        self.pretraining = pretraining
        
        # Add prediction heads if switching to fine-tuning mode
        if not pretraining and not hasattr(self, 'out1'):
            norm = self.config.get('norm', 'BN')
            self.out1 = OutModule2(self.num_filters, self.num_filters, 1, norm, 'FC')
            self.out2 = OutModule2(self.num_filters, self.num_filters, 1, norm, 'Conv')
            self.out3 = OutModule2(self.num_filters, self.num_filters, 1, norm, 'FC')
    
    def freeze_encoder(self):
        """
        Freeze the encoder parameters including input_projection and positional_encoding.
        Useful for fine-tuning scenarios where you want to keep the pretrained encoder frozen.
        """
        # Freeze input projection
        for param in self.input_projection.parameters():
            param.requires_grad = False
        
        # Freeze positional encoding
        for param in self.positional_encoding.parameters():
            param.requires_grad = False
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """
        Unfreeze the encoder parameters including input_projection and positional_encoding.
        Allows all encoder components to be trainable.
        """
        # Unfreeze input projection
        for param in self.input_projection.parameters():
            param.requires_grad = True
        
        # Unfreeze positional encoding
        for param in self.positional_encoding.parameters():
            param.requires_grad = True
        
        # Unfreeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = True
    


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
        out = self.alpha * self.att_layer(self.instance_norm(out),f, mask) + out
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

class AttModule_mamba_cls(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, r1, r2, att_type, stage, alpha, drop_path_rate=0.3,kernel_size=4,dropout_rate=0.2):
        super(AttModule_mamba_cls, self).__init__()
        self.feed_forward = ConvFeedForward_cls(dilation, in_channels, out_channels)
        self.instance_norm = nn.InstanceNorm1d(in_channels, track_running_stats=False)
        self.att_layer = MaskMambaBlock(in_channels, drop_path_rate=drop_path_rate, kernel_size=kernel_size) # dilation
        #self.att_layer = BiMambaEncoder(64, 16,True)
        # self.att_layer = MaskMambaBlock_DBM(in_channels, drop_path_rate=drop_path_rate) # dilation
        self.conv_1x1 = nn.Conv1d(in_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.alpha = alpha
        
    def forward(self, x, f, mask):
        m_batchsize, c1,L = x.size()        
        # padding_mask = torch.ones((m_batchsize, 1, L)).to(x.device) * mask[:,0:1,:]
        padding_mask = mask
        out = self.feed_forward(x)
        out = self.alpha * self.att_layer(self.instance_norm(out), padding_mask) + out
        # out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * padding_mask.unsqueeze(1)
    
class Encoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, output_dim, channel_masking_rate, att_type, alpha, drop_path_rate=0.3,encoder_only=False):
        super(Encoder, self).__init__()
        self.encoder_only = encoder_only
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1) # fc layer
        self.layers = nn.ModuleList([AttModule_mamba(2 ** i, num_f_maps, num_f_maps, 'sliding_att', 'encoder', 1, drop_path_rate=drop_path_rate) for i in range(num_layers)])
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

        self.layers = nn.ModuleList([AttModule_mamba(2 ** i, num_f_maps, num_f_maps, att_type, 'decoder', alpha, drop_path_rate=drop_path_rate) for i in range(num_layers)])
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
        print(d_model)
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
    
# class MBA_tsm(nn.Module):
#     def __init__(self,input_dim, 
#                       n_state=16,
#                       pos_embed_dim=16,
#                       num_filters=64,
#                       BI=True,
#                       num_feature_layers=5,
#                       num_encoder_layers=3,
#                       drop_path_rate=0.3,
#                       tsm_horizon=64,
#                       kernel_size_feature=3,
#                       kernel_size_mba=7,
#                       dropout_rate=0.2,
#                       tsm=False,
#                       mba=True,
#                       add_positional_encoding=True,
#                       max_seq_len=2400,
#                       pretraining=False,
#                       cls_token=False,
#                       num_heads=4,
#                       featurelayer="TCN",
#                       random_padding=False,
#                       average_mask=False,
#                       average_window_size=20,
#                       supcon_loss=False,
#                       reg_only=False,
#                       norm="BN"):
#         super(MBA_tsm, self).__init__()
#         self.add_positional_encoding=add_positional_encoding
#         self.pretraining = pretraining
#         self.cls_token = nn.Parameter(torch.randn(1, num_filters,1))
#         self.pooling=cls_token
#         self.feature_extractor = FeatureExtractor(tsm_horizon,input_dim,pos_embed_dim,num_filters=num_filters,kernel_size=kernel_size_feature,num_feature_layers=num_feature_layers,tsm=tsm,featurelayer=featurelayer)
#         self.positional_encoding = PositionalEncoding(d_model=num_filters,max_len=max_seq_len)
#         self.GatedAttentionMILPooling =GatedAttentionPoolingMIL(input_dim=num_filters, hidden_dim=16)   
#         self.MultiHeadSelfAttentionPooling = MultiHeadSelfAttentionPooling(input_dim=num_filters, hidden_dim=num_filters, num_heads=num_heads)
#         self.random_padding = random_padding
#         self.average_mask = average_mask
#         self.average_window_size = average_window_size
#         self.supcon_loss = supcon_loss
#         self.reg_only = reg_only

#         if not mba:
#             self.encoder = nn.ModuleList([AttModule(2 ** i, num_filters, num_filters, 2, 2,'normal_att', 'encoder', alpha=1,kernel_size=kernel_size_mba,dropout_rate=dropout_rate) for i in range(num_encoder_layers)])
#         else:
#             self.encoder = nn.ModuleList([AttModule_mamba(2 ** i, num_filters, num_filters, 'sliding_att', 'encoder', 1, drop_path_rate=drop_path_rate,dropout_rate=dropout_rate,kernel_size=kernel_size_mba, norm=norm) for i in range(num_encoder_layers)])
        
#         if not reg_only:
#             self.out1 = nn.Linear(num_filters, 1)  
#             self.out2 = nn.Conv1d(num_filters, 1, 1)
#         self.out3 = nn.Linear(num_filters, 1)

#         if self.average_mask:
#             self.scale_out2 = nn.Linear(max_seq_len, max_seq_len//average_window_size)
        
#         self.out1_pretrain = nn.Conv1d(num_filters, 1, 1)
#         self.out2_pretrain = nn.Conv1d(num_filters, 1, 1)
#         self.out3_pretrain = nn.Conv1d(num_filters, 1, 1)

#         if self.supcon_loss:
#             self.proj = nn.Sequential(
#                 nn.Linear(num_filters, num_filters),
#                 nn.ReLU(),
#                 nn.Linear(num_filters, 128)
#             )

#     def forward(self, x, x_lengths, labels1=None, labels3=None, apply_mixup=False, mixup_alpha=0.2):
#         #global xx,mask,mask2,lengths
#         batch_size, seq_len, channels = x.size()       
#         x=self.feature_extractor(x.permute(0, 2, 1))
         
#         if self.pretraining: #If pretraining, then predict x,y,z
#             if self.add_positional_encoding:
#                 x = self.positional_encoding(x)
#             mask = create_mask(torch.tensor(x_lengths,device=x.device), seq_len,batch_size,x.device)
#             if len(self.encoder)>0:
#                 for layer in self.encoder:
#                     x = layer(x,None, mask)
#             # pooled_features,attention_weights= self.GatedAttentionMILPooling(x.permute(0, 2, 1), mask)
#             # pooled_features,weighted_features,attention_weights=self.MultiHeadSelfAttentionPooling(x[:,:,1:].permute(0, 2, 1), mask[:,1:])
#             pooled_features,weighted_features,attention_weights=self.MultiHeadSelfAttentionPooling(x.permute(0, 2, 1), mask)
#             mask=mask.reshape(-1).bool()
#             x1 = self.out1_pretrain(x).view(-1)[mask]
#             x2 = self.out2_pretrain(x).view(-1)[mask]
#             x3= self.out3_pretrain(x).view(-1)[mask]
#         else: # else, predict scratch, scratch sequence, and duration
#             # token = self.cls_token.expand(batch_size, -1, -1)
#             # x = torch.cat((token, x), dim=2)
#             if self.add_positional_encoding:
#                 x = self.positional_encoding(x)

#             mask = create_mask(torch.tensor(x_lengths,device=x.device), seq_len,batch_size,x.device)
#             # mask=torch.cat((torch.ones(batch_size,1,device=x.device),mask), dim=1)
#             if len(self.encoder)>0:
#                 for layer in self.encoder:
#                     x = layer(x,None, mask)
#             # pooled_features,attention_weights= self.GatedAttentionMILPooling(x.permute(0, 2, 1), mask)
#             # pooled_features,weighted_features,attention_weights=self.MultiHeadSelfAttentionPooling(x[:,:,1:].permute(0, 2, 1), mask[:,1:])
#             # mask=mask[:,1:].reshape(-1).bool()
            
#             if self.pooling :
#                 attention_weights = torch.zeros(batch_size, seq_len,1,device=x.device)
#                 pooled_features = x[:, :, 0]
#             else:
#                 pooled_features,weighted_features,attention_weights=self.MultiHeadSelfAttentionPooling(x.permute(0, 2, 1), mask)
            
#             if apply_mixup and self.training:
#                 mixed_features, mixed_labels1, mixed_labels3, mixup_lambda = latent_mixup(
#                     pooled_features, labels1, labels3, alpha=mixup_alpha
#                 )
#                 mixup_info = (mixed_labels1, mixed_labels3, mixup_lambda)
#             else:
#                 mixed_features = pooled_features
#                 mixup_info = None
#             if not self.reg_only:
#                 x1 = self.out1(mixed_features)
#                 x2 = self.out2(x) 

#                 if self.supcon_loss:
#                     contrastive_embedding = self.proj(pooled_features)
#                 else:
#                     contrastive_embedding = None

#                 # if random padding
#                 mask=mask.reshape(-1).bool()
#                 if not self.random_padding:
#                     x2=x2.view(-1)
#                     x2=x2[mask]
#                 else:
#                     x2 = x2.squeeze(1)
#                     if self.average_mask:
#                         x2 = self.scale_out2(x2)
#                 attention_weights=attention_weights.reshape(-1)
#                 attention_weights=attention_weights[mask]
#             # x3= self.out3(torch.cat((x[:,:,0], pooled_features), dim=1))
#             else:
#                 x1, x2, attention_weights, contrastive_embedding = None, None, None, None
#             x3= self.out3(mixed_features)
#         return x1, x2, x3, attention_weights, contrastive_embedding, mixup_info


# class MBA_tsm(nn.Module):
#     def __init__(self,
#                  input_dim, 
#                  n_state=16,
#                  pos_embed_dim=16,
#                  num_filters=64,
#                  BI=True,
#                  num_feature_layers=7,
#                  num_encoder1_layers=3,
#                  num_encoder2_layers=3,
#                  dilation1=True,
#                  dilation2=True,
#                  drop_path_rate=0.3,
#                  tsm_horizon=64,
#                  kernel_size_feature=3,
#                  kernel_size_mba1=7,
#                  kernel_size_mba2=7,
#                  channel_masking_rate=0,
#                  dropout_rate=0.2,
#                  tsm=False,
#                  mba=True,
#                  add_positional_encoding=True,
#                  max_seq_len=2400,
#                  pretraining=False,
#                  use_cls_token=True,
#                  num_heads=4,
#                  featurelayer="TCN",
#                  supcon_loss=False,
#                  skip_connect=False,
#                  skip_cross_attention=False,
#                  norm1='IN',
#                  norm2='IN',
#                  norm3='BN',
#                  pooling_type='attention'):
#         super(MBA_tsm, self).__init__()
        
#         self.input_projection = ConvProjection(input_dim, num_filters,norm1)# Input projection to transform raw signals to model dimension
#         self.add_positional_encoding=add_positional_encoding
#         self.pretraining = pretraining
#         self.use_cls_token = use_cls_token
#         self.cls_token = nn.Parameter(torch.randn(1, num_filters,1))
#         self.num_feature_layers = num_feature_layers
#         self.skip_connect = skip_connect
#         self.skip_cross_attention = skip_cross_attention
#         self.feature_extractor = FeatureExtractor(tsm_horizon,num_filters,pos_embed_dim,num_filters=num_filters,kernel_size=kernel_size_feature,num_feature_layers=num_feature_layers,tsm=tsm,featurelayer=featurelayer,norm=norm1)
#         # self.feature_extractor2 = FeatureExtractor(tsm_horizon,num_filters,pos_embed_dim,num_filters=num_filters,kernel_size=kernel_size_feature,num_feature_layers=num_feature_layers,tsm=tsm,featurelayer='ResTCN',norm=norm1)
#         self.positional_encoding = PositionalEncoding(d_model=num_filters,max_len=max_seq_len)
#         self.GatedAttentionMILPooling =GatedAttentionPoolingMIL(input_dim=num_filters, hidden_dim=16)

#         # Choose pooling type
#         if pooling_type == 'concat':
#             self.pooling = MaskedMaxAvgPooling(input_dim=num_filters, combination_type='concat')
#             self.pooling_output_dim = 2 * num_filters  # Concatenated max and avg
#         elif pooling_type == 'weighted':
#             self.pooling = MaskedMaxAvgPooling(input_dim=num_filters, combination_type='weighted')
#             self.pooling_output_dim = num_filters
#         elif pooling_type == 'masked_avg':
#             self.pooling = MaskedMaxAvgPooling(input_dim=num_filters, combination_type='avg_only')
#             self.pooling_output_dim = num_filters
#         elif pooling_type == 'masked_max':
#             self.pooling = MaskedMaxAvgPooling(input_dim=num_filters, combination_type='max_only')
#             self.pooling_output_dim = num_filters
#         elif pooling_type == 'attention':
#             self.pooling = MultiHeadSelfAttentionPooling(input_dim=num_filters, hidden_dim=num_filters, num_heads=num_heads)
#             self.pooling_output_dim = num_filters
#         else:
#             raise ValueError(f"Unknown pooling_type: {pooling_type}. Choose from ['masked_max_avg', 'masked_avg', 'masked_max', 'attention']")

#         self.pooling_type = pooling_type
#         self.supcon_loss = supcon_loss
#         self.channel_masking_rate = channel_masking_rate
#         self.dropout = nn.Dropout2d(p=channel_masking_rate)


#         # Choose encoder type based on skip connection configuration
#         if self.skip_connect and self.skip_cross_attention:
#             # Use cross-attention modules for skip connections
#             self.encoder1 = nn.ModuleList([AttModule_mamba_cross(2 ** i, num_filters, num_filters, 1, 1, 'sliding_att', 'encoder', 1, drop_path_rate=drop_path_rate, kernel_size=kernel_size_mba1, dropout_rate=dropout_rate) for i in range(num_encoder1_layers)])
#         else:
#             # Use standard attention modules
#             self.encoder1 = nn.ModuleList([AttModule_mamba(2 ** i, num_filters, num_filters, 'sliding_att', 'encoder', 1, drop_path_rate=drop_path_rate,dropout_rate=dropout_rate,kernel_size=kernel_size_mba1,norm=norm2) for i in range(num_encoder1_layers)])

#         # Output layers - adjust input dimension based on pooling type
#         self.out1 = OutModule2(self.pooling_output_dim, num_filters, 1, norm3, 'FC')
#         self.out2 = OutModule2(num_filters, num_filters, 1, norm3, 'Conv')  # Uses conv features, not pooled
#         self.out3 = OutModule2(self.pooling_output_dim, num_filters, 1, norm3, 'FC')
        
#         if pretraining:
#             self.out1_pretrain = nn.Conv1d(num_filters,1, 1)
#             self.out2_pretrain = nn.Conv1d(num_filters,1, 1)
#             self.out3_pretrain = nn.Conv1d(num_filters,1, 1)
        
#         if self.supcon_loss:
#             self.proj = nn.Sequential(
#                 nn.Linear(num_filters, num_filters),
#                 nn.ReLU(),
#                 nn.Linear(num_filters, 128)
#             )

#     def forward(self, x, x_lengths,labels1=None, labels3=None, apply_mixup=False, mixup_alpha=0.2):

#         batch_size, seq_len, channels = x.size()       
#         # Create padding mask from original input
#         mask = create_mask(torch.tensor(x_lengths,device=x.device), seq_len,batch_size,x.device)
        
#         x = self.input_projection(x.permute(0, 2, 1))

#         token = self.cls_token.expand(batch_size, -1, -1)
#         x = torch.cat((token, x), dim=2)
#         mask=torch.cat((torch.ones(batch_size,1,device=x.device), mask), dim=1)
        
#         if self.num_feature_layers>0:
#             # Store intermediate feature extractor outputs for UNet-style skip connections
#             if self.skip_connect:
#                 # Get intermediate features from each layer of feature extractor
#                 x, feature_maps = self.feature_extractor(x, mask, return_intermediates=True)
#             else:
#                 x = self.feature_extractor(x, mask)
            
#             # If in pretraining mode, return pooled features from feature extractor
#             if self.pretraining:
#                 # Choose pooling method based on use_cls_token
#                 if self.use_cls_token:
#                     # Use CLS token (first position) for pooling
#                     pooled_features = x[:, :, 0]
#                 else:
#                     # Use chosen pooling method, exclude CLS token (position 0)
#                     pooled_features, _, _ = self.pooling(x[:, :, 1:].permute(0, 2, 1), mask[:, 1:])
#                 return pooled_features
            
#         if self.add_positional_encoding:
#             x = self.positional_encoding(x)
        
#             if self.channel_masking_rate > 0:
#                 x = x.unsqueeze(2)
#                 x = self.dropout(x)
#                 x = x.squeeze(2)
                
#         # Apply encoder layers with UNet-style skip connections
#         if self.skip_connect and self.num_feature_layers > 0:
#             # UNet-style: connect each encoder layer with reversed feature layer (U-shape)
#             # First encoder layer gets last feature map, last encoder layer gets first feature map
#             for i, layer in enumerate(self.encoder1):
#                 if self.skip_cross_attention:
#                     # Use cross-attention to integrate skip connections
#                     if i < len(feature_maps):
#                         skip_idx = len(feature_maps) - 1 - i  # Reverse the index for U-Net structure
#                         encoder_states = feature_maps[skip_idx]  # Skip connection as encoder states
#                         x = layer(x, encoder_states, mask)  # Cross-attention integration
#                     else:
#                         x = layer(x, None, mask)  # No skip connection available
#                 else:
#                     # Use simple addition for skip connections
#                     x = layer(x, None, mask)
#                     if i < len(feature_maps):
#                         skip_idx = len(feature_maps) - 1 - i  # Reverse the index for U-Net structure
#                         x = x + feature_maps[skip_idx]  # UNet-style skip connection
#         else:
#             # Standard forward without skip connections
#             for layer in self.encoder1:
#                 x = layer(x, None, mask)
        
#         # Choose pooling method based on use_cls_token
#         if self.use_cls_token:
#             # Use CLS token (first position) for pooling
#             pooled_features = x[:, :, 0]
#         else:
#             # Use chosen pooling method, exclude CLS token (position 0)
#             pooled_features, _, _ = self.pooling(x[:, :, 1:].permute(0, 2, 1), mask[:, 1:])
        
#         # If in pretraining mode, return pooled features for contrastive learning
#         if self.pretraining:
#             return pooled_features
        
#         attention_weights = torch.zeros(batch_size, seq_len, 1, device=x.device)
        
#         x1 = self.out1(pooled_features)
#         x3 = self.out3(pooled_features)
#         x2 = self.out2(x[:, :, 1:]) 

#         mask = mask[:, 1:].reshape(-1).bool()
#         x2 = x2.view(-1)
#         x2 = x2[mask]
#         attention_weights = attention_weights.reshape(-1)
#         attention_weights = attention_weights[mask]

#         return x1, x2, x3, attention_weights, None, None
    
#     def set_mode(self, pretraining=True):
#         """
#         Switch between pretraining and fine-tuning modes.
        
#         Args:
#             pretraining: If True, model is in pretraining mode. If False, fine-tuning mode.
#         """
#         self.pretraining = pretraining
    
    
#     def freeze_encoder(self):
#         """
#         Freeze the encoder parameters including input_projection, feature_extractor, 
#         positional_encoding, and encoder1 layers.
#         """
#         # Freeze input projection
#         for param in self.input_projection.parameters():
#             param.requires_grad = False
        
#         # Freeze feature extractor
#         for param in self.feature_extractor.parameters():
#             param.requires_grad = False
        
#         # Freeze positional encoding
#         for param in self.positional_encoding.parameters():
#             param.requires_grad = False
        
#         # Freeze encoder1 layers
#         for layer in self.encoder1:
#             for param in layer.parameters():
#                 param.requires_grad = False
    
#     def unfreeze_encoder(self):
#         """
#         Unfreeze the encoder parameters.
#         """
#         # Unfreeze input projection
#         for param in self.input_projection.parameters():
#             param.requires_grad = True
        
#         # Unfreeze feature extractor
#         for param in self.feature_extractor.parameters():
#             param.requires_grad = True
        
#         # Unfreeze positional encoding
#         for param in self.positional_encoding.parameters():
#             param.requires_grad = True
        
#         # Unfreeze encoder1 layers
#         for layer in self.encoder1:
#             for param in layer.parameters():
#                 param.requires_grad = True
    
#     def freeze_feature_extractor(self):
#         """
#         Freeze only the feature extractor for hybrid pretraining approach.
#         This allows fine-tuning of Mamba encoder while keeping pretrained features frozen.
#         """
#         # Freeze input projection
#         for param in self.input_projection.parameters():
#             param.requires_grad = False
        
#         # Freeze feature extractor
#         for param in self.feature_extractor.parameters():
#             param.requires_grad = False
        
#         # Keep positional encoding and encoder1 layers trainable
#         print("Frozen feature extractor (input_projection + feature_extractor)")
#         print("Mamba encoder layers remain trainable for fine-tuning")
    
#     def unfreeze_feature_extractor(self):
#         """
#         Unfreeze the feature extractor for end-to-end training.
#         """
#         # Unfreeze input projection
#         for param in self.input_projection.parameters():
#             param.requires_grad = True
        
#         # Unfreeze feature extractor
#         for param in self.feature_extractor.parameters():
#             param.requires_grad = True
    
#     def load_pretrained_weights(self, checkpoint_path):
#         """
#         Load pretrained encoder weights from checkpoint for fine-tuning.
#         Supports both direct encoder checkpoints and contrastive learning checkpoints.
#         """
#         checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
#         if 'encoder_weights' in checkpoint:
#             # Direct encoder weights saved during pretraining
#             weights = checkpoint['encoder_weights']
#             self.input_projection.load_state_dict(weights['input_projection'])
#             self.feature_extractor.load_state_dict(weights['feature_extractor'])
#             self.positional_encoding.load_state_dict(weights['positional_encoding'])
#             for i, layer_weights in enumerate(weights['encoder1']):
#                 self.encoder1[i].load_state_dict(layer_weights)
#         elif 'student.input_projection.conv.weight' in checkpoint:
#             # DINO or other contrastive learning checkpoint - extract student weights
#             student_weights = {}
#             for key, value in checkpoint.items():
#                 if key.startswith('student.'):
#                     # Remove 'student.' prefix
#                     new_key = key[8:]  # len('student.') = 8
#                     student_weights[new_key] = value
            
#             # Extract encoder components
#             input_proj_weights = {k.replace('input_projection.', ''): v 
#                                 for k, v in student_weights.items() 
#                                 if k.startswith('input_projection.')}
#             feature_ext_weights = {k.replace('feature_extractor.', ''): v 
#                                  for k, v in student_weights.items() 
#                                  if k.startswith('feature_extractor.')}
#             pos_enc_weights = {k.replace('positional_encoding.', ''): v 
#                              for k, v in student_weights.items() 
#                              if k.startswith('positional_encoding.')}
#             encoder1_weights = {k.replace('encoder1.', ''): v 
#                               for k, v in student_weights.items() 
#                               if k.startswith('encoder1.')}
            
#             # Load the weights
#             self.input_projection.load_state_dict(input_proj_weights)
#             self.feature_extractor.load_state_dict(feature_ext_weights)
#             self.positional_encoding.load_state_dict(pos_enc_weights)
            
#             # Load encoder1 layer weights
#             encoder1_layers = {}
#             for key, value in encoder1_weights.items():
#                 layer_idx = int(key.split('.')[0])
#                 param_key = '.'.join(key.split('.')[1:])
#                 if layer_idx not in encoder1_layers:
#                     encoder1_layers[layer_idx] = {}
#                 encoder1_layers[layer_idx][param_key] = value
            
#             for layer_idx, layer_weights in encoder1_layers.items():
#                 self.encoder1[layer_idx].load_state_dict(layer_weights)
#         else:
#             # Try to load compatible weights (fallback)
#             self.load_state_dict(checkpoint, strict=False)
        
#         print(f"Loaded pretrained weights from {checkpoint_path}")


class MBA_tsm(nn.Module):
    def __init__(self,
                 input_dim, 
                 n_state=16,
                 pos_embed_dim=16,
                 num_filters=64,
                 BI=True,
                 num_feature_layers=7,
                 num_encoder1_layers=3,
                 num_encoder2_layers=3,
                 dilation1=True,
                 dilation2=True,
                 drop_path_rate=0.3,
                 tsm_horizon=64,
                 kernel_size_feature=3,
                 kernel_size_mba1=7,
                 kernel_size_mba2=7,
                 channel_masking_rate=0,
                 dropout_rate=0.2,
                 tsm=False,
                 mba=True,
                 add_positional_encoding=True,
                 max_seq_len=2400,
                 pretraining=False,
                 cls_token=True,
                 num_heads=4,
                 featurelayer="TCN",
                 supcon_loss=False,
                 skip_connect=False,
                 skip_cross_attention=False,
                 norm1='IN',
                 norm2='IN',
                 norm3='BN',
                 pooling_type="attention"):
        super(MBA_tsm, self).__init__()
        
        self.input_projection = ConvProjection(input_dim, num_filters,norm1)# Input projection to transform raw signals to model dimension
        self.add_positional_encoding=add_positional_encoding
        self.pretraining = pretraining
        self.use_cls_token = cls_token
        self.cls_token = nn.Parameter(torch.randn(1, num_filters,1))
        self.num_feature_layers = num_feature_layers
        self.skip_connect = skip_connect
        self.skip_cross_attention = skip_cross_attention
        self.feature_extractor = FeatureExtractor(tsm_horizon,num_filters,pos_embed_dim,num_filters=num_filters,kernel_size=kernel_size_feature,num_feature_layers=num_feature_layers,tsm=tsm,featurelayer=featurelayer,norm=norm1)
        # self.feature_extractor2 = FeatureExtractor(tsm_horizon,num_filters,pos_embed_dim,num_filters=num_filters,kernel_size=kernel_size_feature,num_feature_layers=num_feature_layers,tsm=tsm,featurelayer='ResTCN',norm=norm1)
        # self.positional_encoding = PositionalEncoding(d_model=num_filters,max_len=max_seq_len+1 if self.use_cls_token else max_seq_len)
        self.positional_encoding = PositionalEncoding(d_model=num_filters,max_len=max_seq_len+1)

        self.GatedAttentionMILPooling =GatedAttentionPoolingMIL(input_dim=num_filters, hidden_dim=16)   
        # Choose pooling type
        if pooling_type == 'concat':
            self.pooling = MaskedMaxAvgPooling(input_dim=num_filters, combination_type='concat')
            self.pooling_output_dim = 2 * num_filters  # Concatenated max and avg
        elif pooling_type == 'weighted':
            self.pooling = MaskedMaxAvgPooling(input_dim=num_filters, combination_type='weighted')
            self.pooling_output_dim = num_filters
        elif pooling_type == 'avg_only':
            self.pooling = MaskedMaxAvgPooling(input_dim=num_filters, combination_type='avg_only')
            self.pooling_output_dim = num_filters
        elif pooling_type == 'max_only':
            self.pooling = MaskedMaxAvgPooling(input_dim=num_filters, combination_type='max_only')
            self.pooling_output_dim = num_filters
        elif pooling_type == 'attention':
            self.pooling = MultiHeadSelfAttentionPooling(input_dim=num_filters, hidden_dim=num_filters, num_heads=num_heads)
            self.pooling_output_dim = num_filters
        else:
            raise ValueError(f"Unknown pooling_type: {pooling_type}. Choose from ['weighted', 'avg_only', 'max_only', 'attention']")

        self.pooling_type = pooling_type
        self.supcon_loss = supcon_loss
        self.channel_masking_rate = channel_masking_rate
        self.dropout = nn.Dropout2d(p=channel_masking_rate)


        # Choose encoder type based on skip connection configuration
        if self.skip_connect and self.skip_cross_attention:
            # Use cross-attention modules for skip connections
            self.encoder1 = nn.ModuleList([AttModule_mamba_cross(2 ** i, num_filters, num_filters, 1, 1, 'sliding_att', 'encoder', 1, drop_path_rate=drop_path_rate, kernel_size=kernel_size_mba1, dropout_rate=dropout_rate, norm="IN") for i in range(num_encoder1_layers)])
        else:
            # Use standard attention modules
            self.encoder1 = nn.ModuleList([AttModule_mamba(2 ** i, num_filters, num_filters, 'sliding_att', 'encoder', 1, drop_path_rate=drop_path_rate,dropout_rate=dropout_rate,kernel_size=kernel_size_mba1, norm=norm2) for i in range(num_encoder1_layers)])

        self.out1 = OutModule2(self.pooling_output_dim,num_filters, 1, norm3, 'FC')  
        self.out2 = OutModule2(num_filters,num_filters, 1, norm3, 'Conv')
        self.out3 = OutModule2(self.pooling_output_dim,num_filters, 1, norm3, 'FC')
        
        # if pretraining:
        #     self.out1_pretrain = nn.Conv1d(num_filters,1, 1)
        #     self.out2_pretrain = nn.Conv1d(num_filters,1, 1)
        #     self.out3_pretrain = nn.Conv1d(num_filters,1, 1)
        
        if self.supcon_loss:
            self.proj = nn.Sequential(
                nn.Linear(num_filters, num_filters),
                nn.ReLU(),
                nn.Linear(num_filters, 128)
            )

    def forward(self, x, x_lengths,labels1=None, labels3=None, apply_mixup=False, mixup_alpha=0.2):

        batch_size, seq_len, channels = x.size()       
        # Create padding mask from original input
        if isinstance(x_lengths, torch.Tensor):
            x_lengths = x_lengths.to(x.device)
        else:
            x_lengths = torch.tensor(x_lengths,device=x.device)
        mask = create_mask(x_lengths, seq_len,batch_size,x.device)
        
        x = self.input_projection(x.permute(0, 2, 1))

        token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((token, x), dim=2)
        mask=torch.cat((torch.ones(batch_size,1,device=x.device), mask), dim=1)
        
        if self.num_feature_layers>0:
            # Store intermediate feature extractor outputs for UNet-style skip connections
            if self.skip_connect:
                # Get intermediate features from each layer of feature extractor
                x, feature_maps = self.feature_extractor(x, mask, return_intermediates=True)
            else:
                x = self.feature_extractor(x, mask)
            
            # # If in pretraining mode, return pooled features from feature extractor
            # if self.pretraining:
            #     # Choose pooling method based on use_cls_token
            #     if self.use_cls_token:
            #         # Use CLS token (first position) for pooling
            #         pooled_features = x[:, :, 0]
            #     else:
            #         # Use multi-head self-attention pooling
            #         pooled_features, _, _ = self.pooling(x[:, :, 1:].permute(0, 2, 1), mask[:, 1:])
            #     return pooled_features
            
        if self.add_positional_encoding:
            x = self.positional_encoding(x)
        
            if self.channel_masking_rate > 0:
                x = x.unsqueeze(2)
                x = self.dropout(x)
                x = x.squeeze(2)
                
        # Apply encoder layers with UNet-style skip connections
        if self.skip_connect and self.num_feature_layers > 0:
            # UNet-style: connect each encoder layer with reversed feature layer (U-shape)
            # First encoder layer gets last feature map, last encoder layer gets first feature map
            for i, layer in enumerate(self.encoder1):
                if self.skip_cross_attention:
                    # Use cross-attention to integrate skip connections
                    if i < len(feature_maps):
                        skip_idx = len(feature_maps) - 1 - i  # Reverse the index for U-Net structure
                        encoder_states = feature_maps[skip_idx]  # Skip connection as encoder states
                        x = layer(x, encoder_states, mask)  # Cross-attention integration
                    else:
                        x = layer(x, None, mask)  # No skip connection available
                else:
                    # Use simple addition for skip connections
                    x = layer(x, None, mask)
                    if i < len(feature_maps):
                        skip_idx = len(feature_maps) - 1 - i  # Reverse the index for U-Net structure
                        x = x + feature_maps[skip_idx]  # UNet-style skip connection
        else:
            # Standard forward without skip connections
            for layer in self.encoder1:
                x = layer(x, None, mask)
        
        # Choose pooling method based on use_cls_token
        if self.use_cls_token:
            # Use CLS token (first position) for pooling
            pooled_features = x[:, :, 0]
        else:
            # Use multi-head self-attention pooling
            pooled_features, _, _ = self.pooling(x[:, :, 1:].permute(0, 2, 1), mask[:, 1:])

        if self.pretraining:
            return pooled_features

        attention_weights = torch.zeros(batch_size, seq_len, 1, device=x.device)
        
        x1 = self.out1(pooled_features)
        x3 = self.out3(pooled_features)
        x2 = self.out2(x[:, :, 1:]) 

        mask = mask[:, 1:].reshape(-1).bool()
        x2 = x2.view(-1)
        x2 = x2[mask]
        attention_weights = attention_weights.reshape(-1)
        attention_weights = attention_weights[mask]

        return x1, x2, x3, attention_weights, None, None
    
    def set_mode(self, pretraining=True):
        """
        Switch between pretraining and fine-tuning modes.
        
        Args:
            pretraining: If True, model is in pretraining mode. If False, fine-tuning mode.
        """
        self.pretraining = pretraining
    
    
    def freeze_encoder(self):
        """
        Freeze the encoder parameters including input_projection, feature_extractor, 
        positional_encoding, and encoder1 layers.
        """
        # Freeze input projection
        for param in self.input_projection.parameters():
            param.requires_grad = False
        
        # Freeze feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # Freeze positional encoding
        for param in self.positional_encoding.parameters():
            param.requires_grad = False
        
        # Freeze encoder1 layers
        for layer in self.encoder1:
            for param in layer.parameters():
                param.requires_grad = False
    
    def unfreeze_encoder(self):
        """
        Unfreeze the encoder parameters.
        """
        # Unfreeze input projection
        for param in self.input_projection.parameters():
            param.requires_grad = True
        
        # Unfreeze feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        
        # Unfreeze positional encoding
        for param in self.positional_encoding.parameters():
            param.requires_grad = True
        
        # Unfreeze encoder1 layers
        for layer in self.encoder1:
            for param in layer.parameters():
                param.requires_grad = True
    
    def freeze_feature_extractor(self):
        """
        Freeze only the feature extractor for hybrid pretraining approach.
        This allows fine-tuning of Mamba encoder while keeping pretrained features frozen.
        """
        # Freeze input projection
        for param in self.input_projection.parameters():
            param.requires_grad = False
        
        # Freeze feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # Keep positional encoding and encoder1 layers trainable
        print("Frozen feature extractor (input_projection + feature_extractor)")
        print("Mamba encoder layers remain trainable for fine-tuning")
    
    def unfreeze_feature_extractor(self):
        """
        Unfreeze the feature extractor for end-to-end training.
        """
        # Unfreeze input projection
        for param in self.input_projection.parameters():
            param.requires_grad = True
        
        # Unfreeze feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
    
    def load_pretrained_weights(self, checkpoint_path):
        """
        Load pretrained encoder weights from checkpoint for fine-tuning.
        Supports both direct encoder checkpoints and contrastive learning checkpoints.
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'encoder_weights' in checkpoint:
            # Direct encoder weights saved during pretraining
            weights = checkpoint['encoder_weights']
            self.input_projection.load_state_dict(weights['input_projection'])
            self.feature_extractor.load_state_dict(weights['feature_extractor'])
            self.positional_encoding.load_state_dict(weights['positional_encoding'])
            for i, layer_weights in enumerate(weights['encoder1']):
                self.encoder1[i].load_state_dict(layer_weights)
        elif 'student.input_projection.conv.weight' in checkpoint:
            # DINO or other contrastive learning checkpoint - extract student weights
            student_weights = {}
            for key, value in checkpoint.items():
                if key.startswith('student.'):
                    # Remove 'student.' prefix
                    new_key = key[8:]  # len('student.') = 8
                    student_weights[new_key] = value
            
            # Extract encoder components
            input_proj_weights = {k.replace('input_projection.', ''): v 
                                for k, v in student_weights.items() 
                                if k.startswith('input_projection.')}
            feature_ext_weights = {k.replace('feature_extractor.', ''): v 
                                 for k, v in student_weights.items() 
                                 if k.startswith('feature_extractor.')}
            pos_enc_weights = {k.replace('positional_encoding.', ''): v 
                             for k, v in student_weights.items() 
                             if k.startswith('positional_encoding.')}
            encoder1_weights = {k.replace('encoder1.', ''): v 
                              for k, v in student_weights.items() 
                              if k.startswith('encoder1.')}
            
            # Load the weights
            self.input_projection.load_state_dict(input_proj_weights)
            self.feature_extractor.load_state_dict(feature_ext_weights)
            self.positional_encoding.load_state_dict(pos_enc_weights)
            
            # Load encoder1 layer weights
            encoder1_layers = {}
            for key, value in encoder1_weights.items():
                layer_idx = int(key.split('.')[0])
                param_key = '.'.join(key.split('.')[1:])
                if layer_idx not in encoder1_layers:
                    encoder1_layers[layer_idx] = {}
                encoder1_layers[layer_idx][param_key] = value
            
            for layer_idx, layer_weights in encoder1_layers.items():
                self.encoder1[layer_idx].load_state_dict(layer_weights)
        else:
            # Try to load compatible weights (fallback)
            self.load_state_dict(checkpoint, strict=False)
        
        print(f"Loaded pretrained weights from {checkpoint_path}")

class MBA4TSO(nn.Module):
    """
    MBA model for status prediction with 4-channel input and 3-class sequence-level output.
    Input: [batch_size, seq_len, 4] (x, y, z, temperature)
    Output: [batch_size, seq_len, 3] (other, non-wear, predictTSO masks)
    Supports UNet-style skip connections for better segmentation.
    """
    def __init__(self,
                 input_dim=4,
                 n_state=16,
                 pos_embed_dim=16,
                 num_filters=64,
                 num_feature_layers=7,
                 num_encoder_layers=3,
                 drop_path_rate=0.3,
                 tsm_horizon=64,
                 kernel_size_feature=3,
                 kernel_size_mba=7,
                 dropout_rate=0.2,
                 tsm=False,
                 add_positional_encoding=True,
                 max_seq_len=2400,
                 num_heads=4,
                 featurelayer="TCN",
                 skip_connect=True,
                 skip_cross_attention=False,
                 norm1='IN',
                 norm2='IN',
                 output_channels=3):
        super(MBA4TSO, self).__init__()

        self.input_projection = ConvProjection(input_dim, num_filters, norm1)
        self.add_positional_encoding = add_positional_encoding
        self.num_feature_layers = num_feature_layers
        self.output_channels = output_channels
        self.skip_connect = skip_connect
        self.skip_cross_attention = skip_cross_attention

        # Feature extraction
        self.feature_extractor = FeatureExtractor(
            tsm_horizon, num_filters, pos_embed_dim,
            num_filters=num_filters,
            kernel_size=kernel_size_feature,
            num_feature_layers=num_feature_layers,
            tsm=tsm,
            featurelayer=featurelayer,
            norm=norm1
        )

        self.positional_encoding = PositionalEncoding(d_model=num_filters, max_len=max_seq_len)

        # Choose encoder type based on skip connection configuration
        if self.skip_connect and self.skip_cross_attention:
            # Use cross-attention modules for skip connections
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
            # Use standard attention modules
            self.encoder = nn.ModuleList([
                AttModule_mamba(
                    2 ** i, num_filters, num_filters,
                    'sliding_att', 'encoder', 1,
                    drop_path_rate=drop_path_rate,
                    dropout_rate=dropout_rate,
                    kernel_size=kernel_size_mba,
                    norm=norm2
                ) for i in range(num_encoder_layers)
            ])

        # Output projection: num_filters -> output_channels (3)
        self.output_projection = nn.Conv1d(num_filters, output_channels, kernel_size=1)

    def forward(self, x, x_lengths):
        """
        Args:
            x: [batch_size, seq_len, 4] input tensor
            x_lengths: [batch_size] original sequence lengths

        Returns:
            output: [batch_size, seq_len, 3] class logits for each timestep
        """
        batch_size, seq_len, channels = x.size()

        # Create padding mask
        if isinstance(x_lengths, torch.Tensor):
            x_lengths_tensor = x_lengths.clone().detach()
        else:
            x_lengths_tensor = torch.tensor(x_lengths, device=x.device)
        mask = create_mask(x_lengths_tensor, seq_len, batch_size, x.device)

        # Input projection: [B, 4, L] -> [B, num_filters, L]
        x = self.input_projection(x.permute(0, 2, 1))

        # Feature extraction with optional skip connections
        if self.num_feature_layers > 0:
            if self.skip_connect:
                # Get intermediate features from each layer of feature extractor
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

        # Apply encoder layers with UNet-style skip connections
        if self.skip_connect and self.num_feature_layers > 0:
            # UNet-style: connect each encoder layer with reversed feature layer (U-shape)
            # First encoder layer gets last feature map, last encoder layer gets first feature map
            for i, layer in enumerate(self.encoder):
                if self.skip_cross_attention:
                    # Use cross-attention to integrate skip connections
                    if i < len(feature_maps):
                        skip_idx = len(feature_maps) - 1 - i  # Reverse the index for U-Net structure
                        encoder_states = feature_maps[skip_idx]  # Skip connection as encoder states
                        x = layer(x, encoder_states, mask)  # Cross-attention integration
                    else:
                        x = layer(x, None, mask)  # No skip connection available
                else:
                    # Use simple addition for skip connections
                    x = layer(x, None, mask)
                    if i < len(feature_maps):
                        skip_idx = len(feature_maps) - 1 - i  # Reverse the index for U-Net structure
                        # Interpolate skip connection if sizes don't match
                        if x.size(2) != feature_maps[skip_idx].size(2):
                            skip_resized = F.interpolate(
                                feature_maps[skip_idx],
                                size=x.size(2),
                                mode='linear',
                                align_corners=False
                            )
                            x = x + skip_resized
                        else:
                            x = x + feature_maps[skip_idx]  # UNet-style skip connection
        else:
            # Standard forward without skip connections
            for layer in self.encoder:
                x = layer(x, None, mask)

        # Output projection: [B, num_filters, L] -> [B, output_channels, L]
        output = self.output_projection(x)

        # Interpolate back to original sequence length if needed
        if output.size(2) != seq_len:
            output = F.interpolate(output, size=seq_len, mode='linear', align_corners=False)

        # Permute to [B, L, 3]
        output = output.permute(0, 2, 1)

        return output


class PatchEmbedding(nn.Module):
    """
    Simple and lightweight patch embedding for raw sensor data.
    Maps [batch_size, seq_len, patch_size, channels] -> [batch_size, seq_len, num_filters]

    Uses 1D convolution + pooling for efficiency.
    Memory efficient: processes patches independently.
    """
    def __init__(self, patch_size=1200, in_channels=5, num_filters=64, reduction_factor=4):
        super(PatchEmbedding, self).__init__()

        # Simple 1D conv to extract features from each patch
        # Stride reduces patch dimension by reduction_factor
        self.conv1 = nn.Conv1d(in_channels, num_filters // 2,
                               kernel_size=7, stride=reduction_factor, padding=3)
        self.bn1 = nn.BatchNorm1d(num_filters // 2)
        self.act1 = nn.ReLU(inplace=True)

        # Second conv to further process
        self.conv2 = nn.Conv1d(num_filters // 2, num_filters,
                               kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(num_filters)
        self.act2 = nn.ReLU(inplace=True)

        # Global average pooling to get single vector per patch
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, patch_size, channels]
        Returns:
            out: [batch_size, seq_len, num_filters]
        """
        batch_size, seq_len, patch_size, channels = x.size()

        # Reshape: [B*seq_len, channels, patch_size]
        x = x.view(batch_size * seq_len, patch_size, channels)
        x = x.permute(0, 2, 1)  # [B*seq_len, channels, patch_size]

        # Apply convolutions
        x = self.act1(self.bn1(self.conv1(x)))  # [B*seq_len, num_filters//2, reduced]
        x = self.act2(self.bn2(self.conv2(x)))  # [B*seq_len, num_filters, reduced]

        # Global pooling
        x = self.pool(x).squeeze(-1)  # [B*seq_len, num_filters]

        # Reshape back
        x = x.view(batch_size, seq_len, -1)  # [B, seq_len, num_filters]

        return x


class MBA4TSO_Patch(nn.Module):
    """
    MBA4TSO variant for patched raw sensor input.

    Input: [batch_size, seq_len, patch_size, channels]
           - seq_len: number of minutes (e.g., 1440 for 24h)
           - patch_size: samples per minute (e.g., 1200 = 60s * 20Hz)
           - channels: 5 (x, y, z, temperature, time_cyclic)

    Output: [batch_size, seq_len, 3] (other, non-wear, predictTSO logits)

    Architecture:
    1. SimplePatchEmbedding: patches -> embeddings
    2. Reduced FeatureExtractor: lightweight feature extraction
    3. Mamba Encoder: temporal modeling
    4. Output projection: 3-class prediction
    """
    def __init__(self,
                 patch_size=1200,
                 patch_channels=5,
                 num_filters=64,
                 num_feature_layers=3,  # Reduced from 7
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

        # Patch embedding: [B, L, patch_size, 5] -> [B, L, num_filters]
        self.patch_embedding = PatchEmbedding(
            patch_size=patch_size,
            in_channels=patch_channels,
            num_filters=num_filters
        )

        # Lightweight feature extraction (reduced layers)
        if num_feature_layers > 0:
            self.feature_extractor = FeatureExtractor(
                tsm_horizon=64,
                in_channels=num_filters,
                pos_embed_dim=16,
                num_filters=num_filters,
                kernel_size=kernel_size_feature,
                num_feature_layers=num_feature_layers,
                tsm=False,
                featurelayer=featurelayer,
                norm=norm1
            )
        else:
            self.feature_extractor = None

        # Positional encoding
        if add_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_model=num_filters, max_len=max_seq_len)

        # Mamba encoder layers with optional cross-attention for skip connections
        if self.skip_connect and self.skip_cross_attention:
            # Use cross-attention modules for skip connections
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
            # Use standard attention modules
            self.encoder = nn.ModuleList([
                AttModule_mamba(
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

        # Patch embedding: [B, L, patch_size, 5] -> [B, L, num_filters]
        x = self.patch_embedding(x)  # [B, seq_len, num_filters]

        # Permute for conv layers: [B, num_filters, seq_len]
        x = x.permute(0, 2, 1)

        # Feature extraction with optional skip connections
        if self.feature_extractor is not None:
            if self.skip_connect:
                # Get intermediate features from each layer of feature extractor
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

        # Apply encoder layers with UNet-style skip connections
        if self.skip_connect and self.num_feature_layers > 0:
            # UNet-style: connect each encoder layer with reversed feature layer (U-shape)
            # First encoder layer gets last feature map, last encoder layer gets first feature map
            for i, layer in enumerate(self.encoder):
                if self.skip_cross_attention:
                    # Use cross-attention to integrate skip connections
                    if i < len(feature_maps):
                        skip_idx = len(feature_maps) - 1 - i  # Reverse the index for U-Net structure
                        encoder_states = feature_maps[skip_idx]  # Skip connection as encoder states
                        x = layer(x, encoder_states, mask)  # Cross-attention integration
                    else:
                        x = layer(x, None, mask)  # No skip connection available
                else:
                    # Use simple addition for skip connections
                    x = layer(x, x, mask)  # AttModule_mamba expects (x, f, mask)
                    if i < len(feature_maps):
                        skip_idx = len(feature_maps) - 1 - i  # Reverse the index for U-Net structure
                        # Interpolate skip connection if sizes don't match
                        if x.size(2) != feature_maps[skip_idx].size(2):
                            skip_resized = F.interpolate(
                                feature_maps[skip_idx],
                                size=x.size(2),
                                mode='linear',
                                align_corners=False
                            )
                            x = x + skip_resized
                        else:
                            x = x + feature_maps[skip_idx]  # UNet-style skip connection
        else:
            # Standard forward without skip connections
            for layer in self.encoder:
                x = layer(x, x, mask)  # AttModule_mamba expects (x, f, mask)

        # Output projection
        output = self.output_projection(x)  # [B, output_channels, L]

        # Interpolate back to original sequence length if needed
        if output.size(2) != seq_len:
            output = F.interpolate(output, size=seq_len, mode='linear', align_corners=False)

        # Permute to [B, L, 3]
        output = output.permute(0, 2, 1)

        return output


class MBA_tsm_with_padding(nn.Module):
    def __init__(self,input_dim, 
                      n_state=16,
                      pos_embed_dim=16,
                      num_filters=64,
                      BI=True,
                      num_feature_layers=5,
                      num_encoder_layers=3,
                      drop_path_rate=0.3,
                      tsm_horizon=64,
                      kernel_size_feature=3,
                      kernel_size_mba=7,
                      dropout_rate=0.2,
                      tsm=False,
                      mba=True,
                      add_positional_encoding=True,
                      max_seq_len=2400,
                      cls_token=False,
                      num_heads=4,
                      featurelayer="TCN",
                      average_mask=False,
                      average_window_size=20,
                      supcon_loss=False,
                      reg_only=False,
                      use_feature_extractor=True,
                      norm="BN"):
        super(MBA_tsm_with_padding, self).__init__()
        self.add_positional_encoding=add_positional_encoding
        self.cls_token = nn.Parameter(torch.randn(1, num_filters,1))
        self.pooling=cls_token
        self.use_feature_extractor = use_feature_extractor
        if self.use_feature_extractor:
            self.feature_extractor = FeatureExtractor(tsm_horizon,
                                                      input_dim,
                                                      pos_embed_dim,
                                                      num_filters=num_filters,
                                                      kernel_size=kernel_size_feature,
                                                      num_feature_layers=num_feature_layers,
                                                      tsm=tsm,
                                                      featurelayer=featurelayer)
        else:
            self.input_projection = nn.Conv1d(input_dim, num_filters, 1)
            self.norm = nn.LayerNorm(num_filters)
            # BN
            # self.norm = nn.BatchNorm1d(num_filters)
            # IN
            # self.norm = nn.InstanceNorm1d(num_filters)
        self.positional_encoding = PositionalEncoding(d_model=num_filters,max_len=max_seq_len)
        self.GatedAttentionMILPooling =GatedAttentionPoolingMIL(input_dim=num_filters, hidden_dim=16)   
        self.MultiHeadSelfAttentionPooling = MultiHeadSelfAttentionPooling(input_dim=num_filters, hidden_dim=num_filters, num_heads=num_heads)
        self.average_mask = average_mask
        self.average_window_size = average_window_size
        self.supcon_loss = supcon_loss
        self.reg_only = reg_only

        if not mba:
            self.encoder = nn.ModuleList([AttModule(2 ** i, num_filters, num_filters, 2, 2,'normal_att', 'encoder', alpha=1,kernel_size=kernel_size_mba,dropout_rate=dropout_rate) for i in range(num_encoder_layers)])
        else:
            self.encoder = nn.ModuleList([AttModule_mamba(2 ** i, num_filters, num_filters, 'sliding_att', 'encoder', 1, drop_path_rate=drop_path_rate,dropout_rate=dropout_rate,kernel_size=kernel_size_mba, norm=norm) for i in range(num_encoder_layers)])
        
        if not reg_only:
            self.out1 = nn.Linear(num_filters, 1)  
            self.out2 = nn.Conv1d(num_filters, 1, 1)
        self.out3 = nn.Linear(num_filters, 1)

        if self.average_mask:
            self.scale_out2 = nn.Linear(max_seq_len, max_seq_len//average_window_size)
        

    def forward(self, x, labels1=None, labels3=None, apply_mixup=False, mixup_alpha=0.2, padding_value=-999.0, replaced_padding_value=-0.5, **kwargs):
        batch_size, seq_len, channels = x.size()    
        
        x_orig = x
        x_for_features = x.clone()
        padding_mask_for_replacement = torch.any(x == padding_value, dim=-1, keepdim=True)
        x_for_features = torch.where(padding_mask_for_replacement, replaced_padding_value, x_for_features)
        if self.use_feature_extractor:
            x = x_for_features.permute(0, 2, 1)
            x = self.feature_extractor(x)
        else:
            # Project input to model dimension: [B, seq_len, input_dim] -> [B, seq_len, num_filters]
            x = self.input_projection(x_for_features.permute(0, 2, 1))
            # Transpose for conv layers: [B, seq_len, num_filters] -> [B, num_filters, seq_len]
            x = self.norm(x.permute(0, 2, 1))
            x = x.permute(0, 2, 1)
        
        if self.add_positional_encoding:
            x = self.positional_encoding(x)

        # padding_mask = create_mask(torch.tensor(x_lens, device=x.device), seq_len, batch_size, x.device).float()
        # check padding value for each position
        padding_mask = ~torch.any(x_orig == padding_value, dim=-1)
        # padding_mask = padding_mask.float()
        cur_seq_len = x.shape[2]
        if cur_seq_len != seq_len:
            padding_mask = F.interpolate(
                padding_mask.float().unsqueeze(1), 
                size=cur_seq_len,
                mode="nearest" 
            ).squeeze(1).bool().float()
        else:
            padding_mask = padding_mask.float()

        # No masking for fixed length padded inputs
        if len(self.encoder)>0:
            for layer in self.encoder:
                x = layer(x, None, padding_mask)  # use padding mask
        
        if self.pooling :
            attention_weights = torch.zeros(batch_size, cur_seq_len, 1, device=x.device)
            pooled_features = x[:, :, 0]
        else:
            # Use padding mask for attention pooling
            pooled_features,weighted_features,attention_weights=self.MultiHeadSelfAttentionPooling(x.permute(0, 2, 1), padding_mask)
        
        if apply_mixup and self.training:
            mixed_features, mixed_labels1, mixed_labels3, mixup_lambda = latent_mixup(
                pooled_features, labels1, labels3, alpha=mixup_alpha
            )
            mixup_info = (mixed_labels1, mixed_labels3, mixup_lambda)
        else:
            mixed_features = pooled_features
            mixup_info = None
            
        if not self.reg_only:
            x1 = self.out1(mixed_features)
            x2 = self.out2(x) 

            if self.supcon_loss:
                contrastive_embedding = pooled_features
            else:
                contrastive_embedding = None

            # Keep padding - do not remove any values from x2
            x2 = x2.squeeze(1)
            if self.average_mask:
                x2 = self.scale_out2(x2)
            # Return all attention weights without masking
            attention_weights = attention_weights.reshape(-1)
        else:
            x1, x2, attention_weights, contrastive_embedding = None, None, None, None
            
        x3= self.out3(mixed_features)
        return x1, x2, x3, attention_weights, contrastive_embedding, mixup_info
    
    def load_pretrained_feature_extractor(self, checkpoint_path, freeze=True):
        """
        Load a pretrained FeatureExtractor into this MBA_tsm model.
        
        Args:
            checkpoint_path: Path to the pretrained feature extractor checkpoint
            freeze: Whether to freeze the feature extractor parameters
        """
        import torch
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        feature_extractor_state_dict = {}
        
        for key, value in checkpoint.items():
            if key.startswith('backbone.feature_extractor.'):
                # Remove 'backbone.feature_extractor.' prefix
                new_key = key[len('backbone.feature_extractor.'):]
                feature_extractor_state_dict[new_key] = value
        
        # If no backbone keys found, assume entire checkpoint is feature extractor
        if not feature_extractor_state_dict:
            feature_extractor_state_dict = checkpoint
        
        # Load the state dict
        try:
            self.feature_extractor.load_state_dict(feature_extractor_state_dict, strict=True)
            print(f"Successfully loaded feature extractor weights with {len(feature_extractor_state_dict)} parameters")
        except RuntimeError as e:
            print("Attempting to load with strict=False...")
            missing_keys, unexpected_keys = self.feature_extractor.load_state_dict(feature_extractor_state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
        
        if freeze:
            # Freeze feature extractor parameters
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            # set feature extractor to eval mode to freeze BatchNorm
            self.feature_extractor.eval()
            print("FeatureExtractor parameters frozen for fine-tuning")
        else:
            print("FeatureExtractor parameters remain trainable")
    
    def unfreeze_feature_extractor(self):
        """Unfreeze the feature extractor parameters."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        print("FeatureExtractor parameters unfrozen")
    
    def train(self, mode=True):
        """Override train method to keep frozen feature extractor in eval mode."""
        super().train(mode)
        # If feature extractor is frozen (no gradients), keep it in eval mode
        # if hasattr(self, 'feature_extractor'):
        #     frozen_params = [p for p in self.feature_extractor.parameters() if not p.requires_grad]
        #     if len(frozen_params) > 0:
        #         self.feature_extractor.eval()
        return self

  
class MBA_patch(nn.Module):
    """
    Mamba model using patch embeddings from PatchEmbedWithPos1D.
    Similar to MBA_tsm_with_padding but uses patch-based input processing like ViT1D.
    """
    
    def __init__(
        self,
        max_seq_len: int = 1221,            # Maximum sequence length
        window_size: int = 60,              # Patch window size
        stride: int = 20,                   # Patch stride
        in_channels: int = 3,               # Number of input channels
        embed_dim = None,                   # Embedding dimension (defaults to patch_dim)
        use_embedding: bool = False,        # Whether to apply linear embedding to patches
        num_encoder_layers: int = 6,        # Number of Mamba encoder layers
        drop_path_rate: float = 0.3,        # Drop path rate for Mamba blocks
        kernel_size_mba: int = 7,           # Kernel size for Mamba blocks
        dropout_rate: float = 0.2,          # Dropout rate
        padding_value: float = -999.0,      # Padding value
        padding_threshold: float = 0.5,     # Padding threshold
        num_heads: int = 4,                 # Number of attention heads for pooling
        supcon_loss: bool = False,          # Whether to return contrastive embeddings
        reg_only: bool = False,             # Whether to do regression only
        cls_token: bool = False,            # Whether to use CLS token pooling
    ):
        super(MBA_patch, self).__init__()
        
        self.max_seq_len = max_seq_len
        self.window_size = window_size
        self.stride = stride
        self.in_channels = in_channels
        self.use_embedding = use_embedding
        self.num_encoder_layers = num_encoder_layers
        self.supcon_loss = supcon_loss
        self.reg_only = reg_only
        self.pooling = cls_token
        
        # Import PatchEmbedWithPos1D locally to avoid circular imports
        try:
            from Helpers.net.embed import PatchEmbedWithPos1D
        except:
            from net.embed import PatchEmbedWithPos1D
        
        # Patch embedding with positional encoding and CLS token
        self.patch_embed = PatchEmbedWithPos1D(
            max_seq_len=max_seq_len,
            window_size=window_size,
            stride=stride,
            in_channels=in_channels,
            embed_dim=embed_dim,
            padding_value=padding_value,
            padding_threshold=padding_threshold,
            use_embedding=use_embedding,
            pos_learnable=False,
            include_cls=True,
            dropout=dropout_rate
        )
        
        # Get actual embedding dimension from patch embedding
        self.embed_dim = self.patch_embed.patch_embed.embed_dim
        
        # Multi-head self-attention pooling
        self.MultiHeadSelfAttentionPooling = MultiHeadSelfAttentionPooling(
            input_dim=self.embed_dim, 
            hidden_dim=self.embed_dim, 
            num_heads=num_heads
        )
        
        # Mamba encoder layers using existing AttModule_mamba
        if num_encoder_layers > 0:
            self.encoder = nn.ModuleList([
                AttModule_mamba_cls(
                    dilation=min(16, 2**i), #2**i
                    in_channels=self.embed_dim, 
                    out_channels=self.embed_dim, 
                    r1=2, 
                    r2=2,
                    att_type='sliding_att', 
                    stage='encoder', 
                    alpha=1, 
                    drop_path_rate=drop_path_rate,
                    dropout_rate=dropout_rate,
                    kernel_size=kernel_size_mba
                ) for i in range(num_encoder_layers)
            ])
        else:
            self.encoder = nn.ModuleList([])
        
        # Output heads
        if not reg_only:
            self.out1 = nn.Linear(self.embed_dim, 1)  # Classification
            self.out2 = nn.Conv1d(self.embed_dim, 1, 1)  # Mask prediction
            expected_patches = (max_seq_len - window_size) // stride + 1
            target_length = max_seq_len // stride 
            self.scale_out2 = nn.Linear(expected_patches, target_length)
        
        self.out3 = nn.Linear(self.embed_dim, 1)  # Regression
    
    def forward(
        self, 
        x,
        labels1=None,
        labels3=None,
        apply_mixup=False,
        mixup_alpha=0.2,
        **kwargs
    ):
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, L, C]
            labels1: Classification labels
            labels3: Regression labels
            apply_mixup: Whether to apply mixup
            mixup_alpha: Mixup alpha parameter
            **kwargs: Additional keyword arguments
            
        Returns:
            x1: Classification output [B, 1]
            x2: Mask prediction output [B, seq_len] 
            x3: Regression output [B, 1]
            attention_weights: Attention weights from pooling
            contrastive_embedding: Pooled features if supcon_loss=True
            mixup_info: Mixup information if apply_mixup=True
        """
        assert x.dim() == 3, "Input must be of shape [B, L, C]"
        batch_size, seq_len, channels = x.shape
        
        # Convert to [B, C, L] for patch embedding
        x = x.permute(0, 2, 1)  # [B, C, L]

        # Get patch embeddings with CLS token and positional encoding
        x, padding_mask = self.patch_embed(x)  # [B, 1+num_patches, embed_dim]
        
        # Convert to [B, embed_dim, seq_len] format for Mamba blocks
        current_seq_len = x.shape[1]  # 1 + num_patches
        x = x.permute(0, 2, 1)  # [B, embed_dim, 1+num_patches]
        
        # Create padding mask in float format for AttModule_mamba
        if padding_mask is not None:
            padding_mask_float = (~padding_mask).float()  # [B, 1+num_patches], True for valid tokens
            # padding_mask_float = (padding_mask).float()
        else:
            padding_mask_float = torch.ones(batch_size, current_seq_len, device=x.device)
        
        # Apply Mamba encoder layers
        if len(self.encoder) > 0:
            for layer in self.encoder:
                x = layer(x, None, padding_mask_float)  # AttModule_mamba(x, f, mask)
        
        # Convert back to [B, seq_len, embed_dim] for pooling
        x = x.permute(0, 2, 1)  # [B, 1+num_patches, embed_dim]
        
        # Pool features
        if self.pooling:  # Use CLS token
            pooled_features = x[:, 0]  # [B, embed_dim] - CLS token
            attention_weights = torch.zeros(batch_size, current_seq_len, 1, device=x.device)
        else:
            # Use multi-head attention pooling
            pooled_features, weighted_features, attention_weights = self.MultiHeadSelfAttentionPooling(
                x, padding_mask_float
            )
        
        # Apply mixup if requested
        if apply_mixup and self.training and labels1 is not None and labels3 is not None:
            mixed_features, mixed_labels1, mixed_labels3, mixup_lambda = latent_mixup(
                pooled_features, labels1, labels3, alpha=mixup_alpha
            )
            mixup_info = (mixed_labels1, mixed_labels3, mixup_lambda)
        else:
            mixed_features = pooled_features
            mixup_info = None
        
        # Generate outputs
        if not self.reg_only:
            x1 = self.out1(mixed_features)  # [B, 1] Classification
            
            # For mask prediction, convert back to conv format
            # x_conv = x.permute(0, 2, 1)  # [B, embed_dim, seq_len]
            # x2 = self.out2(x_conv)  # [B, 1, seq_len]
            # x2 = x2.squeeze(1)  # [B, seq_len]
            patch_tokens = x[:, 1:, :]
            x2 = self.out2(patch_tokens.permute(0, 2, 1)).squeeze(1)
            
            x2 = self.scale_out2(x2)  # [B, seq_len // average_window_size]
            
            if self.supcon_loss:
                contrastive_embedding = pooled_features  # Use pooled features directly
            else:
                contrastive_embedding = None
        else:
            x1, x2, attention_weights, contrastive_embedding = None, None, None, None
        
        x3 = self.out3(mixed_features)  # [B, 1] Regression
        
        # Reshape attention weights to match expected format
        if attention_weights is not None:
            attention_weights = attention_weights.reshape(-1)
        
        return x1, x2, x3, attention_weights, contrastive_embedding, mixup_info

class tcn_learner(nn.Module): #Multi Task TCN
    def __init__(self, input_channels, embedding_dim=8,lin_embed_dim=64,  num_classes=1, num_layers=5, num_filters=64, kernel_size=3, max_dilation=16, dropout=0.2,BI=False):
        super(tcn_learner, self).__init__()
        layers = []
        dilation = 1
        self.GatedAttentionMILPooling =GatedAttentionPoolingMIL(input_dim=num_filters, hidden_dim=16)
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



class AttModule_mamba_causal(nn.Module):
    """Causal attention module for decoder-only architectures."""
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
        # Apply causal mask to padding mask for causal attention
        if causal_mask is not None and padding_mask is not None:
            # Expand causal mask to batch dimension and ensure it doesn't exceed valid positions
            expanded_causal_mask = causal_mask.unsqueeze(0).expand(m_batchsize, -1, -1)  # [B, L, L]
            # Apply padding mask - zero out padded positions  
            combined_mask = padding_mask.unsqueeze(-1) * expanded_causal_mask  # [B, L, L]
            # Take the last valid position for each sequence (causal)
            mask = combined_mask[:, :, -1:].permute(0, 2, 1)  # [B, 1, L]
        else:
            mask = padding_mask.unsqueeze(1) if padding_mask is not None else None
            
        out = self.feed_forward(x)
        out = self.alpha * self.att_layer(self.instance_norm(out), mask) + out
        out = self.dropout(out)
        
        if mask is not None:
            return (x + out) * mask
        else:
            return x + out


class AttModule_mamba_cross(nn.Module):
    """Cross-attention module for encoder-decoder architectures."""
    def __init__(self, dilation, in_channels, out_channels, r1, r2, att_type, stage, alpha, drop_path_rate=0.3, kernel_size=7, dropout_rate=0.2):
        super(AttModule_mamba_cross, self).__init__()
        self.feed_forward = ConvFeedForward(dilation, in_channels, out_channels, kernel_size=kernel_size)
        self.instance_norm = nn.InstanceNorm1d(in_channels, track_running_stats=False)
        self.self_att_layer = MaskMambaBlock(in_channels, drop_path_rate=drop_path_rate, kernel_size=kernel_size)
        self.cross_att_layer = MaskMambaBlock(in_channels, drop_path_rate=drop_path_rate, kernel_size=kernel_size)  
        self.conv_1x1 = nn.Conv1d(in_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.alpha = alpha
        
    def forward(self, x, encoder_states, padding_mask):
        m_batchsize, c1, L = x.size()        
        
        out = self.feed_forward(x)
        
        # Self-attention on decoder states
        out = self.alpha * self.self_att_layer(self.instance_norm(out), padding_mask) + out
        
        # Cross-attention to encoder states (if provided)
        if encoder_states is not None:
            # Ensure encoder_states and out have compatible dimensions
            if encoder_states.shape[2] == out.shape[2]:  # Same sequence length
                cross_out = self.alpha * self.cross_att_layer(self.instance_norm(encoder_states), padding_mask)
                out = out + cross_out
            else:
                # Interpolate encoder states to match decoder length
                encoder_states_resized = F.interpolate(encoder_states, size=L, mode='linear', align_corners=False)
                cross_out = self.alpha * self.cross_att_layer(self.instance_norm(encoder_states_resized), padding_mask)
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

class MBA_tsm_encoder_decoder_ch_bottleneck(nn.Module):
    """
    Encoder-decoder architecture for multi-task scratch detection.
    Encoder processes input, decoder generates outputs with cross-attention.
    """
    def __init__(self, input_dim, 
                 num_filters=64,
                 num_encoder_layers=2,
                 num_decoder_layers=1,  # Single decoder for one class
                 drop_path_rate=0.3,
                 kernel_size_mba=7,
                 max_seq_len=256,
                 cls_token=False,
                 add_positional_encoding=True,
                 num_heads=4,
                 dropout_rate=0.2,
                 mba=True,
                 supcon_loss=False,
                 cls_src="encoder"):  # "encoder" or "decoder" for classification source
        super(MBA_tsm_encoder_decoder_ch_bottleneck, self).__init__()
        
        self.add_positional_encoding = add_positional_encoding
        self.max_seq_len = max_seq_len
        self.pooling = cls_token
        self.supcon_loss = supcon_loss
        self.cls_src = cls_src
        self.num_filters = num_filters
    
        self.positional_encoding = PositionalEncoding(d_model=num_filters, max_len=max_seq_len)  # +1 for CLS token
        self.MultiHeadSelfAttentionPooling = MultiHeadSelfAttentionPooling(input_dim=num_filters, hidden_dim=num_filters, num_heads=num_heads)

        # Encoder layers (bidirectional)
        if not mba:
            self.encoder = nn.ModuleList([AttModule(2 ** i, num_filters, num_filters, 2, 2, 'normal_att', 'encoder', alpha=1, 
                                                  kernel_size=kernel_size_mba, dropout_rate=dropout_rate) for i in range(num_encoder_layers)])
        else:
            self.encoder = nn.ModuleList([AttModule_mamba_causal(2 ** i, num_filters, num_filters, 'sliding_att', 'encoder', 1, 
                                                        drop_path_rate=drop_path_rate, dropout_rate=dropout_rate, 
                                                        kernel_size=kernel_size_mba,
                                                        ) for i in range(num_encoder_layers)])

        # Decoder layers (cross-attention following MyTransformer approach)
        if not mba:
            self.decoder = nn.ModuleList([AttModule_cross(2 ** i, num_filters, num_filters, 2, 2, 'cross_att', 'decoder', alpha=1, 
                                                        kernel_size=kernel_size_mba, dropout_rate=dropout_rate) for i in range(num_decoder_layers)])
        else:
            self.decoder = nn.ModuleList([AttModule_mamba_cross(2 ** i, num_filters, num_filters, 2, 2, 'cross_att', 'decoder', 1, 
                                                              drop_path_rate=drop_path_rate, dropout_rate=dropout_rate, 
                                                              kernel_size=kernel_size_mba) for i in range(num_decoder_layers)])
        
        self.encoder_expand = nn.Conv1d(input_dim, num_filters, 1)

        # Compression layer after encoder: [B, num_filters, seq_len] -> [B, 1, seq_len]
        self.encoder_compress = nn.Conv1d(num_filters, 1, 1)
        
        # Expansion layer before decoder: [B, 1, seq_len] -> [B, num_filters, seq_len]
        self.decoder_expand = nn.Conv1d(1, num_filters, 1)
        self.encoder_norm = nn.LayerNorm(num_filters)
        # BN
        # self.encoder_norm = nn.BatchNorm1d(num_filters)
        # IN
        # self.encoder_norm = nn.InstanceNorm1d(num_filters)
        self.decoder_compress = nn.Conv1d(num_filters, 1, 1) # Sequence labeling head
        
        # All three output heads use pooled sequence (max_seq_len dimension)
        # self.out1 = nn.Linear(max_seq_len, 1)   # Classification head   
        # self.out3 = nn.Linear(max_seq_len, 1)   # Regression head
        self.out1 = nn.Linear(num_filters, 1)
        self.out3 = nn.Linear(num_filters, 1)


    def forward(self, x, padding_value=-999.0, **kwargs):
        batch_size, seq_len, channels = x.size()       
        x_original = x
        padding_mask_original = ~torch.any(x_original == padding_value, dim=-1)  # [B, seq_len]

        # Handle extreme padding values
        x_for_features = x.clone()

        if abs(padding_value) > 100:
            padding_mask_for_replacement = torch.any(x == padding_value, dim=-1, keepdim=True)
            x_for_features = torch.where(padding_mask_for_replacement, 0.0, x_for_features)

        # Transpose for conv layers: [B, seq_len, num_filters] -> [B, num_filters, seq_len]
        x = x_for_features.permute(0, 2, 1) 
        # Project input to model dimension: [B, input_dim, seq_len] -> [B, num_filters, seq_len]
        x = self.encoder_expand(x)
        x = self.encoder_norm(x.permute(0, 2, 1)).permute(0, 2, 1)
         
        if self.add_positional_encoding:
            x = self.positional_encoding(x)
        
        current_seq_len = x.shape[2]  # seq_len (includes CLS token)
        padding_mask = padding_mask_original.float()
        
        # Encoder pass - no skip connections needed
        encoder_out = x  # [B, num_filters, seq_len]
        for layer in self.encoder:
            encoder_out = layer(encoder_out, None, padding_mask)
        
        # Compress encoder output: [B, num_filters, seq_len] -> [B, 1, seq_len]
        compressed_encoder = self.encoder_compress(encoder_out)  # [B, 1, seq_len]
        
        # Apply softmax and mask for information bottleneck
        # compressed_bottleneck = F.softmax(compressed_encoder, dim=1)  # [B, 1, seq_len] - probability bottleneck
        compressed_bottleneck = torch.sigmoid(compressed_encoder)
        decoder_mask_expanded = padding_mask.unsqueeze(1)  # [B, 1, seq_len]
        compressed_bottleneck = compressed_bottleneck * decoder_mask_expanded
        
        # Expand for decoder input: [B, 1, seq_len] -> [B, num_filters, seq_len]
        decoder_out = self.decoder_expand(compressed_bottleneck)  # [B, num_filters, seq_len]
    
        # Following video-mamba-suite MyTransformer: use compressed bottleneck and encoder features both masked
        masked_encoder_out = encoder_out * decoder_mask_expanded  # [B, num_filters, seq_len]
        # Apply decoder layers following MyTransformer approach
        for layer in self.decoder:
            # Pass both masked inputs to decoder layer (following MyTransformer pattern)
            decoder_out = layer(decoder_out, masked_encoder_out, padding_mask)
        
        # Always use CLS token (position 0) for task predictions
        attention_weights = torch.zeros(batch_size, current_seq_len, 1, device=x.device)
        # pooled_features = self.decoder_compress(decoder_out).squeeze(1)  # [B, seq_len]
        
        # # Multi-task outputs using from specified source
        # x1 = self.out1(pooled_features)  # Classification from CLS token
        # x3 = self.out3(pooled_features)  # Regression from CLS token
        # x2 = pooled_features
        decoder_out_transposed = decoder_out.permute(0, 2, 1)  # [B, seq_len, num_filters]
        
        # Use multi-head attention pooling for global representation
        cls_token, weighted_features, attention_weights = self.MultiHeadSelfAttentionPooling(
            decoder_out_transposed, padding_mask_original
        )  # [B, num_filters], [B, seq_len, 1]
        # Multi-task outputs
        x1 = self.out1(cls_token)  # Classification from attention pooled features
        x3 = self.out3(cls_token)  # Regression from attention pooled features
        x2 = self.decoder_compress(decoder_out).squeeze(1)  # [B, seq_len] - Sequence labeling

        if self.supcon_loss:
            contrastive_embedding = pooled_features 
        else:
            contrastive_embedding = None
        mixup_info = None
        return x1, x2, x3, attention_weights, contrastive_embedding, mixup_info


class MBA_tsm_encoder_decoder_seq_bottleneck(nn.Module):
    """
    Encoder-decoder architecture with sequence-level information bottleneck.
    Compresses sequence length from seq_len to seq_len/reduction while keeping num_filters channels.
    """
    def __init__(self, input_dim, 
                 num_filters=64,
                 num_encoder_layers=2,
                 num_decoder_layers=1,
                 drop_path_rate=0.3,
                 kernel_size_mba=7,
                 max_seq_len=256,
                 cls_token=False,
                 add_positional_encoding=True,
                 num_heads=4,
                 dropout_rate=0.2,
                 mba=True,
                 supcon_loss=False,
                 cls_src="encoder",
                 seq_reduction=8):  # Sequence compression factor
        super(MBA_tsm_encoder_decoder_seq_bottleneck, self).__init__()
        
        self.add_positional_encoding = add_positional_encoding
        self.max_seq_len = max_seq_len
        self.pooling = cls_token
        self.supcon_loss = supcon_loss
        self.cls_src = cls_src
        self.num_filters = num_filters
        self.seq_reduction = seq_reduction
        self.compressed_seq_len = max_seq_len // seq_reduction
        
        self.positional_encoding = PositionalEncoding(d_model=num_filters, max_len=max_seq_len)
        self.MultiHeadSelfAttentionPooling = MultiHeadSelfAttentionPooling(input_dim=num_filters, hidden_dim=num_filters, num_heads=num_heads)

        # Encoder layers
        if not mba:
            self.encoder = nn.ModuleList([AttModule(2 ** i, num_filters, num_filters, 2, 2, 'normal_att', 'encoder', alpha=1, 
                                                  kernel_size=kernel_size_mba, dropout_rate=dropout_rate) for i in range(num_encoder_layers)])
        else:
            self.encoder = nn.ModuleList([AttModule_mamba_causal(2 ** i, num_filters, num_filters, 'sliding_att', 'encoder', 1, 
                                                        drop_path_rate=drop_path_rate, dropout_rate=dropout_rate, 
                                                        kernel_size=kernel_size_mba) for i in range(num_encoder_layers)])
        
        # Decoder layers (cross-attention)
        if not mba:
            self.decoder = nn.ModuleList([AttModule_cross(2 ** i, num_filters, num_filters, 2, 2, 'cross_att', 'decoder', alpha=1, 
                                                        kernel_size=kernel_size_mba, dropout_rate=dropout_rate) for i in range(num_decoder_layers)])
        else:
            self.decoder = nn.ModuleList([AttModule_mamba_cross(2 ** i, num_filters, num_filters, 2, 2, 'cross_att', 'decoder', 1, 
                                                              drop_path_rate=drop_path_rate, dropout_rate=dropout_rate, 
                                                              kernel_size=kernel_size_mba) for i in range(num_decoder_layers)])
        
        # Sequence compression/expansion layers
        self.encoder_expand = nn.Conv1d(input_dim, num_filters, 1)
        self.encoder_norm = nn.LayerNorm(num_filters)
        # BN
        # self.encoder_norm = nn.BatchNorm1d(num_filters)
        # IN
        # self.encoder_norm = nn.InstanceNorm1d(num_filters)

        # Sequence compression: [B, num_filters, seq_len] -> [B, num_filters, seq_len/reduction] 
        self.seq_compress = nn.Conv1d(num_filters, num_filters, kernel_size=seq_reduction, stride=seq_reduction, padding=0)
        
        # Sequence expansion: [B, num_filters, seq_len/reduction] -> [B, num_filters, seq_len]
        self.seq_expand = nn.ConvTranspose1d(num_filters, num_filters, kernel_size=seq_reduction, stride=seq_reduction, padding=0)
        
        # Output heads for sequence predictions  
        self.decoder_compress = nn.Conv1d(num_filters, 1, 1)  # Sequence labeling head
        
        # Task-specific heads using attention pooling
        self.out1 = nn.Linear(num_filters, 1)   # Classification from attention pooled features   
        self.out3 = nn.Linear(num_filters, 1)   # Regression from attention pooled features

    def forward(self, x, padding_value=-999.0, **kwargs):
        batch_size, seq_len, channels = x.size()       
        x_original = x
        padding_mask_original = ~torch.any(x_original == padding_value, dim=-1)  # [B, seq_len]

        # Handle extreme padding values
        x_for_features = x.clone()
        if abs(padding_value) > 100:
            padding_mask_for_replacement = torch.any(x == padding_value, dim=-1, keepdim=True)
            x_for_features = torch.where(padding_mask_for_replacement, 0.0, x_for_features)

        # Transpose for conv layers: [B, seq_len, channels] -> [B, channels, seq_len]
        x_for_features = x_for_features.permute(0, 2, 1) 
        # Project input to model dimension: [B, channels, seq_len] -> [B, num_filters, seq_len]
        x = self.encoder_expand(x_for_features)
        x = self.norm(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        
        if self.add_positional_encoding:
            x = self.positional_encoding(x)
        
        current_seq_len = x.shape[2]  # seq_len
        padding_mask = padding_mask_original.float()
        
        # Encoder pass
        encoder_out = x  # [B, num_filters, seq_len]
        for layer in self.encoder:
            encoder_out = layer(encoder_out, None, padding_mask)
        
        # Sequence compression: [B, num_filters, seq_len] -> [B, num_filters, seq_len/reduction]
        compressed_encoder = self.seq_compress(encoder_out)  # [B, num_filters, compressed_seq_len]
        compressed_seq_len = compressed_encoder.shape[2]
        
        # Create compressed padding mask
        compressed_padding_mask = F.max_pool1d(
            padding_mask.unsqueeze(1).float(), 
            kernel_size=self.seq_reduction, 
            stride=self.seq_reduction
        ).squeeze(1)  # [B, compressed_seq_len]
        
        # Apply sigmoid activation for information bottleneck
        compressed_bottleneck = torch.sigmoid(compressed_encoder)  # [B, num_filters, compressed_seq_len]
        compressed_bottleneck = compressed_bottleneck * compressed_padding_mask.unsqueeze(1)
        
        # Expand back to original sequence length: [B, num_filters, compressed_seq_len] -> [B, num_filters, seq_len]
        decoder_out = self.seq_expand(compressed_bottleneck)  # [B, num_filters, seq_len]
        
        # Ensure decoder output matches original sequence length
        if decoder_out.shape[2] != current_seq_len:
            decoder_out = F.interpolate(decoder_out, size=current_seq_len, mode='linear', align_corners=False)
        
        # Masked encoder output for cross-attention
        padding_mask_expanded = padding_mask.unsqueeze(1)  # [B, 1, seq_len]
        masked_encoder = encoder_out * padding_mask_expanded  # [B, num_filters, seq_len]
        
        # Apply decoder layers with cross-attention
        for layer in self.decoder:
            decoder_out = layer(decoder_out, masked_encoder, padding_mask)
        
        # Generate outputs using attention pooling
        # Convert to [B, seq_len, num_filters] for attention pooling
        decoder_out_transposed = decoder_out.permute(0, 2, 1)  # [B, seq_len, num_filters]
        
        # Use multi-head attention pooling for global representation
        cls_token, weighted_features, attention_weights = self.MultiHeadSelfAttentionPooling(
            decoder_out_transposed, padding_mask_original
        )  # [B, num_filters], [B, seq_len, 1]
        # Multi-task outputs
        x1 = self.out1(cls_token)  # Classification from attention pooled features
        x3 = self.out3(cls_token)  # Regression from attention pooled features
        x2 = self.decoder_compress(decoder_out).squeeze(1)  # [B, seq_len] - Sequence labeling
        
        if self.supcon_loss:
            contrastive_embedding = cls_token  # Use attention pooled features for contrastive learning
        else:
            contrastive_embedding = None
        
        mixup_info = None
        return x1, x2, x3, attention_weights, contrastive_embedding, mixup_info

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
                 cls_token=True,
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
        self.pooling = cls_token
        self.average_mask = average_mask
        self.average_window_size = average_window_size
        self.supcon_loss = supcon_loss
        self.input_dim = input_dim
        self.num_filters = num_filters
        self.cls_src = cls_src

        # Input projection to transform raw signals to model dimension
        self.input_projection = nn.Conv1d(input_dim, num_filters, 1)
        self.norm = nn.LayerNorm(num_filters)
        # BN
        # self.norm = nn.BatchNorm1d(num_filters)
        # IN
        # self.norm = nn.InstanceNorm1d(num_filters)
        
        # Progressive depth reduction (U-Net style) architecture
        # Define progressive feature dimensions FIRST
        self.encoder_dims = [num_filters * (2 ** i) for i in range(num_encoder_layers)]  # [64, 128, 256, 512]
        self.decoder_dims = [self.encoder_dims[-(i+1)] for i in range(num_decoder_layers)]  # [512, 256, 128] (reverse)
        
        # Optional positional encoding
        if add_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_model=num_filters, max_len=max_seq_len)
        
        # Pooling layer for final representations (use final encoder dimension)
        self.MultiHeadSelfAttentionPooling = MultiHeadSelfAttentionPooling(
            input_dim=self.encoder_dims[-1], hidden_dim=self.encoder_dims[-1], num_heads=num_heads
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
                AttModule_mamba_causal(2 ** i, self.encoder_dims[i], self.encoder_dims[i], 'sliding_att', 'encoder', 1, 
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

    def forward(self, x, labels1=None, labels3=None, apply_mixup=False, mixup_alpha=0.2, padding_value=-999.0,  **kwargs):
        batch_size, seq_len, channels = x.size()       
        x_original = x
        
        # Create padding mask from original input
        padding_mask_original = ~torch.any(x_original == padding_value, dim=-1)  # [B, seq_len]
        
        # Handle extreme padding values for processing
        x_processed = x.clone()
        if abs(padding_value) > 100:
            padding_mask_for_replacement = torch.any(x == padding_value, dim=-1, keepdim=True)
            x_processed = torch.where(padding_mask_for_replacement, 0.0, x_processed)
        
        # Project input to model dimension: [B, seq_len, input_dim] -> [B, seq_len, num_filters]
        x = self.input_projection(x_processed.permute(0, 2, 1))
        
        # Transpose for conv layers: [B, seq_len, num_filters] -> [B, num_filters, seq_len]
        x = self.norm(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        
        # Add positional encoding if enabled
        if self.add_positional_encoding:
            x = self.positional_encoding(x)

        # Use original padding mask (sequence length unchanged)
        padding_mask = padding_mask_original.float()
        current_seq_len = seq_len  # No change in sequence length
        
        # Progressive encoder pass with dimension expansion
        encoder_states = []
        encoder_out = x  # Start with [B, num_filters, seq_len]
        
        for i, (proj, layer) in enumerate(zip(self.encoder_projections, self.encoder)):
            # Progressive dimension expansion: num_filters -> encoder_dims[i]
            encoder_out = proj(encoder_out)  # [B, encoder_dims[i], seq_len]
            encoder_out = layer(encoder_out, None, padding_mask)
            encoder_states.append(encoder_out)
        
        # Extract CLS token from encoder output for information bottleneck
        # encoder_out shape: [B, encoder_dims[-1], seq_len] -> permute to [B, seq_len, encoder_dims[-1]]
        encoder_features = encoder_out.permute(0, 2, 1)  # [B, seq_len, encoder_dims[-1]]
        
        # Multi-head attention pooling to get global representation
        cls_token, weighted_features, attention_weights = self.MultiHeadSelfAttentionPooling(
            encoder_features, padding_mask
        )
        
        # SEQUENCE PREDICTION: Choose between encoder-only or encoder-decoder approach
        # Option 1: Use decoder (comment out for encoder-only)
        # DECODER PASS: Use full encoder features instead of CLS token bottleneck
        # Use encoder output directly as decoder input (preserves spatial information)
        decoder_input = encoder_out  # [B, num_filters, seq_len] - already in correct format
        
        # Use the original padding mask for decoder (respects actual sequence lengths)
        decoder_mask = padding_mask  # Use same mask as encoder
        
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
            decoder_out = layer(decoder_out, encoder_skip_proj, decoder_mask)
        
        # Apply mixup augmentation to CLS token if enabled
        if self.cls_src == "encoder":
            cls_feature = cls_token
        else:
            cls_feature = decoder_out[:, :, 0]
            # cls_feature = decoder_out[:, :, 1:].mean(dim=-1)

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
        x2 = x2.squeeze(1)              # [B, seq_len]
        
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

        return x1, x2, x3, attention_weights, contrastive_embedding, mixup_info

    def train(self, mode=True):
        """Standard train method - no feature extractor to handle specially."""
        return super().train(mode)


class MBA_tsm_encoder_decoder_progressive(nn.Module):
    """
    Simplified progressive encoder-decoder architecture without skip connections.
    Uses progressive dimension expansion/reduction for efficient information flow.
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
                 max_seq_len=256,
                 cls_token=False,
                 add_positional_encoding=True,
                 num_heads=4,
                 dropout_rate=0.2,
                 mba=True,
                 average_mask=False,
                 average_window_size=20,
                 supcon_loss=False,
                 reg_only=False):
        super(MBA_tsm_encoder_decoder_progressive, self).__init__()
        
        self.add_positional_encoding = add_positional_encoding
        self.max_seq_len = max_seq_len
        self.pooling = cls_token
        self.average_mask = average_mask
        self.average_window_size = average_window_size
        self.supcon_loss = supcon_loss
        self.reg_only = reg_only
        self.input_dim = input_dim
        self.num_filters = num_filters

        # Input projection to transform raw signals to model dimension
        self.input_projection = nn.Linear(input_dim, num_filters)
        
        # Progressive dimension architecture (simplified - no skip connections)
        # Define progressive feature dimensions
        self.encoder_dims = [num_filters * (2 ** i) for i in range(num_encoder_layers)]  # [64, 128, 256, 512]
        self.decoder_dims = [self.encoder_dims[-(i+1)] for i in range(num_decoder_layers)]  # [512, 256, 128] (reverse)
        
        # Optional positional encoding
        if add_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_model=num_filters, max_len=max_seq_len)
        
        # Pooling layer for final representations (use final encoder dimension)
        self.MultiHeadSelfAttentionPooling = MultiHeadSelfAttentionPooling(
            input_dim=self.encoder_dims[-1], hidden_dim=self.encoder_dims[-1], num_heads=num_heads
        )
        
        # Dimension projection layers for progressive feature expansion/reduction
        self.encoder_projections = nn.ModuleList([
            nn.Conv1d(self.encoder_dims[i-1] if i > 0 else num_filters, self.encoder_dims[i], 1)
            for i in range(num_encoder_layers)
        ])
        
        self.decoder_projections = nn.ModuleList([
            nn.Conv1d(self.decoder_dims[i-1] if i > 0 else self.encoder_dims[-1], self.decoder_dims[i], 1)
            for i in range(num_decoder_layers)
        ])
        
        # NO skip connection projections - simplified architecture

        # Encoder layers with progressive feature dimensions
        if not mba:
            self.encoder = nn.ModuleList([
                AttModule(2 ** i, self.encoder_dims[i], self.encoder_dims[i], 2, 2, 'normal_att', 'encoder', alpha=1, 
                         kernel_size=kernel_size_mba, dropout_rate=dropout_rate) 
                for i in range(num_encoder_layers)
            ])
        else:
            self.encoder = nn.ModuleList([
                AttModule_mamba_causal(2 ** i, self.encoder_dims[i], self.encoder_dims[i], 'sliding_att', 'encoder', 1, 
                               drop_path_rate=drop_path_rate, dropout_rate=dropout_rate, 
                               kernel_size=kernel_size_mba) 
                for i in range(num_encoder_layers)
            ])

        # Decoder layers with self-attention only (no cross-attention)
        if not mba:
            self.decoder = nn.ModuleList([
                AttModule(2 ** i, self.decoder_dims[i], self.decoder_dims[i], 2, 2, 'normal_att', 'decoder', alpha=1, 
                         kernel_size=kernel_size_mba, dropout_rate=dropout_rate) 
                for i in range(num_decoder_layers)
            ])
        else:
            self.decoder = nn.ModuleList([
                AttModule_mamba(2 ** i, self.decoder_dims[i], self.decoder_dims[i], 'sliding_att', 'decoder', 1, 
                               drop_path_rate=drop_path_rate, dropout_rate=dropout_rate, 
                               kernel_size=kernel_size_mba) 
                for i in range(num_decoder_layers)
            ])
        
        # Task-specific output heads (use final encoder/decoder dimensions)
        if not reg_only:
            self.out1 = nn.Linear(self.encoder_dims[-1], 1)  # Classification head (from encoder CLS)
            self.out2 = nn.Conv1d(self.decoder_dims[-1], 1, 1)  # Sequence labeling head (from decoder)
        self.out3 = nn.Linear(self.encoder_dims[-1], 1)  # Regression head (from encoder CLS)

        if self.average_mask:
            self.scale_out2 = nn.Linear(max_seq_len, max_seq_len//average_window_size)

    def forward(self, x, labels1=None, labels3=None, apply_mixup=False, mixup_alpha=0.2, padding_value=-999.0, **kwargs):
        batch_size, seq_len, channels = x.size()       
        x_original = x
        
        # Create padding mask from original input
        padding_mask_original = ~torch.any(x_original == padding_value, dim=-1)  # [B, seq_len]
        
        # Handle extreme padding values for processing
        x_processed = x.clone()
        if abs(padding_value) > 100:
            padding_mask_for_replacement = torch.any(x == padding_value, dim=-1, keepdim=True)
            x_processed = torch.where(padding_mask_for_replacement, 0.0, x_processed)
        
        # Project input to model dimension: [B, seq_len, input_dim] -> [B, seq_len, num_filters]
        x = self.input_projection(x_processed)
        
        # Transpose for conv layers: [B, seq_len, num_filters] -> [B, num_filters, seq_len]
        x = x.permute(0, 2, 1)
        
        # Add positional encoding if enabled
        if self.add_positional_encoding:
            x = self.positional_encoding(x)

        # Use original padding mask (sequence length unchanged)
        padding_mask = padding_mask_original.float()
        current_seq_len = seq_len  # No change in sequence length
        
        # Progressive encoder pass with dimension expansion (no skip connection storage needed)
        encoder_out = x  # Start with [B, num_filters, seq_len]
        
        for i, (proj, layer) in enumerate(zip(self.encoder_projections, self.encoder)):
            # Progressive dimension expansion: num_filters -> encoder_dims[i]
            encoder_out = proj(encoder_out)  # [B, encoder_dims[i], seq_len]
            encoder_out = layer(encoder_out, None, padding_mask)  # Self-attention only
        
        # Extract CLS token from encoder output
        # encoder_out shape: [B, encoder_dims[-1], seq_len] -> permute to [B, seq_len, encoder_dims[-1]]
        encoder_features = encoder_out.permute(0, 2, 1)  # [B, seq_len, encoder_dims[-1]]
        
        # Get CLS token (first position) from encoder
        if self.pooling:
            cls_token = encoder_features[:, 1:, :].mean(dim=1)  # [B, encoder_dims[-1]] - average pool positions 1:
            attention_weights = torch.zeros(batch_size, current_seq_len, 1, device=x.device)
        else:
            # Multi-head attention pooling to get global representation
            cls_token, weighted_features, attention_weights = self.MultiHeadSelfAttentionPooling(
                encoder_features, padding_mask
            )
        
        # Apply mixup augmentation to CLS token if enabled
        if apply_mixup and self.training:
            mixed_cls_features, mixed_labels1, mixed_labels3, mixup_lambda = latent_mixup(
                cls_token, labels1, labels3, alpha=mixup_alpha
            )
            mixup_info = (mixed_labels1, mixed_labels3, mixup_lambda)
        else:
            mixed_cls_features = cls_token
            mixup_info = None
        
        # ENCODER OUTPUTS: Classification and Regression from CLS token
        if not self.reg_only:
            x1 = self.out1(mixed_cls_features)  # Classification from encoder CLS: [B, 1]
            x3 = self.out3(mixed_cls_features)  # Regression from encoder CLS: [B, 1]
        else:
            x1, x3 = None, self.out3(mixed_cls_features)
        
        # SEQUENCE PREDICTION: Simplified decoder (no skip connections)
        if not self.reg_only:
            # Use original padding mask for decoder
            decoder_mask = padding_mask
            
            # Progressive decoder pass with dimension reduction (NO skip connections)
            decoder_out = encoder_out  # Initialize decoder with encoder features [B, encoder_dims[-1], seq_len]
            
            for i, (proj, layer) in enumerate(zip(self.decoder_projections, self.decoder)):
                # Progressive dimension reduction: decoder_dims[i-1] -> decoder_dims[i]
                decoder_out = proj(decoder_out)  # [B, decoder_dims[i], seq_len]
                
                # Decoder processes with SELF-ATTENTION ONLY (no cross-attention/skip connections)
                decoder_out = layer(decoder_out, None, decoder_mask)
            
            # DECODER OUTPUT: Sequence mask prediction
            x2 = self.out2(decoder_out)     # Sequence labeling from decoder: [B, 1, seq_len]
            x2 = x2.squeeze(1)              # [B, seq_len]

            # Apply mask averaging if enabled
            if self.average_mask:
                x2 = x2.view(batch_size, seq_len//self.average_window_size, self.average_window_size).mean(dim=2)

            # Contrastive embedding from encoder CLS token
            if self.supcon_loss:
                contrastive_embedding = cls_token  # Use raw CLS token directly
            else:
                contrastive_embedding = None

            return x1, x2, x3, attention_weights, contrastive_embedding, mixup_info
        else:
            return None, None, x3, attention_weights, None, mixup_info

    def train(self, mode=True):
        """Standard train method - no feature extractor to handle specially."""
        return super().train(mode)

class mba_learner(nn.Module):
    def __init__(self,input_dim, n_state=16,pos_embed_dim=8,lin_embed_dim=64,BI=True,num_layers=2):
        super(mba_learner, self).__init__()
        
#         self.pos_embedding = PositionalEmbedding(max_seq_len, pos_embed_dim)
        self.embedding = nn.Linear(input_dim, lin_embed_dim)
        num_filters=lin_embed_dim#+pos_embed_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, num_filters))
        self.GatedAttentionMILPooling =GatedAttentionPoolingMIL(input_dim=num_filters, hidden_dim=16)
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
        self.GatedAttentionMILPooling =GatedAttentionPoolingMIL(input_dim=num_filters*2, hidden_dim=16)
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


class PatchTSTHead(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super(PatchTSTHead, self).__init__()
        self.use_cls_token = config.use_cls_token
        self.pooling_type = config.pooling_type
        self.cls_only = config.cls_only
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity()
        # self.linear = nn.Linear(config.num_input_channels * config.d_model, config.num_targets)
        # self.out1 = nn.Linear(config.num_input_channels * config.d_model, 1)
        self.conv = nn.Conv1d(in_channels=config.num_input_channels, out_channels=1, kernel_size=1)
        self.out1 = nn.Linear(config.d_model, 1)
        if not self.cls_only:
            # self.out2 = nn.Linear(config.num_input_channels * config.d_model, config.prediction_length)
            # self.out3 = nn.Linear(config.num_input_channels * config.d_model, 1)
            self.out2 = nn.Linear(config.d_model, config.prediction_length)
            self.out3 = nn.Linear(config.d_model, 1)

    def forward(self, embedding: torch.Tensor, x_lengths: list, max_seq_len=None):
        """
        Parameters:
            embedding (`torch.Tensor` of shape `(bs, num_channels, num_patches, d_model)` or
                     `(bs, num_channels, num_patches+1, d_model)` if `cls_token` is set to True, *required*):
                Embedding from the model
        Returns:
            `torch.Tensor` of shape `(bs, num_targets)`

        """
        if self.use_cls_token:
            # use the first output token, pooled_embedding: bs x num_channels x d_model
            pooled_embedding = embedding[:, :, 0, :]
        elif self.pooling_type == "mean":
            # pooled_embedding: [bs x num_channels x d_model]
            pooled_embedding = embedding.mean(dim=2)
        elif self.pooling_type == "max":
            # pooled_embedding: [bs x num_channels x d_model]
            pooled_embedding = embedding.max(dim=2).values
        else:
            raise ValueError(f"pooling operator {self.pooling_type} is not implemented yet")
        # pooled_embedding: bs x num_channels * d_model
        # pooled_embedding = self.flatten(pooled_embedding)
        pooled_embedding = self.conv(pooled_embedding)
        pooled_embedding = pooled_embedding.squeeze(1)
        # output: bs x n_classes
        # output = self.linear(self.dropout(pooled_embedding))
        output1 = self.out1(self.dropout(pooled_embedding))
        if not self.cls_only:
            output2 = self.out2(self.dropout(pooled_embedding))
            output3 = self.out3(self.dropout(pooled_embedding))

            mask = create_mask(torch.tensor(x_lengths, device=pooled_embedding.device), max_seq_len, len(x_lengths), pooled_embedding.device)
            mask = mask.reshape(-1).bool()
            output2 = output2.view(-1)
            output2 = output2[mask]
            attention_weights = torch.zeros(pooled_embedding.shape[0], pooled_embedding.shape[1], 1,device=pooled_embedding.device)
            return output1, output2, output3, attention_weights
        else:
            return output1

class PatchTSTNS(nn.Module):
    def __init__(self, config: PatchTSTConfig, pretrained: bool=False):
        super(PatchTSTNS, self).__init__()
        
        self.model = PatchTSTModel(config)
        # TODO: CONNECTION ERROR DUE TO FIREWALL
        if pretrained: 
            self.model.from_pretrained("namctin/patchtst_etth1_pretrain")
        self.head = PatchTSTHead(config)
    
    def forward(self, x, x_lengths, max_seq_len):
        x = self.model(x)
        return self.head(x["last_hidden_state"], x_lengths, max_seq_len)


class PredictHead(nn.Module):
    def __init__(self, prediction_length, 
                       d_model, 
                       reg_head=True, 
                       mask_padding=True,
                       average_mask=False,
                       average_window_size=20):
        super(PredictHead, self).__init__()
        self.reg_head = reg_head
        self.mask_padding = mask_padding
        self.out1 = nn.Linear(d_model, 1)
        self.average_mask = average_mask
        if self.average_mask:
            self.out2 = nn.Linear(d_model, prediction_length//average_window_size)
        else:
            self.out2 = nn.Linear(d_model, prediction_length)
        if self.reg_head:
            self.out3 = nn.Linear(d_model, 1)
    
    def forward(self, x, x_lengths=None, max_seq_len=None):
        # multi-task output
        out1 = self.out1(x)
        out2 = self.out2(x)
        out3 = None
        if self.reg_head:
            out3 = self.out3(x)
        attn = torch.zeros(x.shape[0], 1).to(x.device)

        if self.mask_padding and not self.average_mask:
            # mask padding area and concat all batchs into a ginlge vector 
            # TODO: concat might be wrong, since default BCELoss use mean reduction
            assert x_lengths is not None, "If mask_padding is True, x_lengths must be given."
            mask = create_mask(torch.tensor(x_lengths, device=x.device), max_seq_len, len(x_lengths), x.device)
            mask = mask.reshape(-1).bool()
            out2 = out2.view(-1)
            out2 = out2[mask]
        return out1, out2, out3


class EfficientUNet(nn.Module):
    def __init__(self, prediction_length, 
                       model_name="efficientnet_v2_m", 
                       d_model=1024, 
                       input_dim=3,
                    #    tsm_horizon=64, 
                    #    pos_embed_dim=16, 
                    #    num_filters=64,
                    #    kernel_size_feature=13,
                    #    num_feature_layers=7,
                    #    tsm=False,
                    #    featurelayer="ResTCN",
                       target_shape=128,
                       tcn_params={},
                       reg_head=True,
                       mask_padding=True,
                       average_mask=False,
                       average_window_size=20):
        super(EfficientUNet, self).__init__()

        # self.featurelayer = featurelayer
        # if use out3 layer to predict scratch duration
        self.reg_head = reg_head
        self.feature_extractor = FeatureExtractorConv2d(input_dim, prediction_length, target_shape, **tcn_params)
        # if self.featurelayer == "Conv2d":
        #     self.feature_extractor = FeatureExtractorConv2d(input_dim, prediction_length, target_shape, **tcn_params)
        # else:
        #     self.feature_extractor = FeatureExtractor(tsm_horizon,
        #                                             input_dim,
        #                                             pos_embed_dim,
        #                                             num_filters=num_filters,
        #                                             kernel_size=kernel_size_feature,
        #                                             num_feature_layers=num_feature_layers,
        #                                             tsm=tsm,
        #                                             featurelayer=featurelayer)
        
        # Encoder: EfficientNet-B3 from torchvision
        if model_name == "efficientnet_v2_m":
            self.encoder = models.efficientnet_v2_m(pretrained=True)  
            self.cs = [80, 176, 1280]
        elif model_name == "efficientnet_b3":
            self.encoder = models.efficientnet_b3(pretrained=True)  
            self.cs = [48, 136, 1536]
        # Load EfficientNet-B3
        self.encoder_blocks = [
            self.encoder.features[0:4],  # First block (stem + initial conv)
            self.encoder.features[4:6],  # Second block
            self.encoder.features[6:],    # Remaining blocks
        ]
        # for v2_m: 80, 176, 1280
        # for b3: 48, 136, 1536
        # Decoding layers
        if model_name == "efficientnet_v2_m":
            self.decoder1 = nn.ConvTranspose2d(1280, 176, kernel_size=3, stride=2, padding=1)  
            self.decoder2 = nn.ConvTranspose2d(176, 80, kernel_size=3, stride=2, padding=1)
            # self.decoder3 = nn.ConvTranspose2d(80, 1, kernel_size=3, stride=2, padding=1)   
            # Batch Normalization layers
            self.bn1 = nn.BatchNorm2d(176)
            self.bn2 = nn.BatchNorm2d(80)
            self.conv = nn.Conv2d(80, d_model, kernel_size=3, stride=2, padding=1)
        else:
            self.decoder1 = nn.ConvTranspose2d(1536, 136, kernel_size=3, stride=2, padding=1)  
            self.decoder2 = nn.ConvTranspose2d(136, 48, kernel_size=3, stride=2, padding=1)
            # self.decoder3 = nn.ConvTranspose2d(48, 1, kernel_size=3, stride=2, padding=1)  
            self.bn1 = nn.BatchNorm2d(136)
            self.bn2 = nn.BatchNorm2d(48)
            self.conv = nn.Conv2d(48, d_model, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(d_model)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = PredictHead(prediction_length, 
                                d_model, 
                                reg_head=reg_head, 
                                mask_padding=mask_padding, 
                                average_mask=average_mask,
                                average_window_size=average_window_size)

    def forward(self, x, x_lengths=None, max_seq_len=None):
        x = self.feature_extractor(x.permute(0, 2, 1), x_lengths) # [batch size, num_filters, pred_len]
        # if self.featurelayer != "Conv2d":
        #     # expand dims at 1
        #     x = torch.unsqueeze(x, 1) 
        #     # Expand the channel dimension to size 3
        #     x = x.expand(-1, 3, -1, -1) 

        # Encoder Stage
        features = []  # Container for saving encoder outputs
        for block in self.encoder_blocks:
            x = block(x)  # Forward pass through each encoder block
            features.append(x)
        # Decoder Stage with Skip Connections
        x = self.decoder1(x)  # First Decoder
        x = features[1] + nn.functional.interpolate(x, size=features[1].shape[2:], mode='nearest')  # Skip connection
        x = self.bn1(x)
        x = torch.relu(x)

        x = self.decoder2(x)  # Second Decoder
        x = features[0] + nn.functional.interpolate(x, size=features[0].shape[2:], mode='nearest')  # Skip connection
        x = self.bn2(x)
        x = torch.relu(x)

        # conv
        x = self.conv(x)
        # TODO: add bn and relu
        x = self.bn3(x)
        x = torch.relu(x)
        # pool
        x = self.pool(x)
        x = x.squeeze()

        # multi-task output
        attn = torch.zeros(x.shape[0], 1).to(x.device)

        out1, out2, out3 = self.head(x, x_lengths, max_seq_len)
        return out1, out2, out3, attn

class GASFFeatureExtractor(nn.Module):
    def __init__(self, image_size):
        super(GASFFeatureExtractor, self).__init__()
        self.sum_feature_extractor = GramianAngularField(image_size=image_size, method="summation")


class ViT(nn.Module):
    def __init__(self, prediction_length, 
                 input_dim=3, 
                 d_model=1024, 
                 hidden_dropout_prob=0.1,
                 load_pretrained=False,
                 target_shape=128,
                 tcn_params={},
                 reg_head=True,
                 mask_padding=True,
                 average_mask=False,
                 average_window_size=20):
        super(ViT, self).__init__()

        self.reg_head = reg_head

        # use Conv2d FeatureExtractor
        self.feature_extractor = FeatureExtractorConv2d(input_dim, prediction_length, target_shape, **tcn_params)

        # ViT configuration
        self.config = ViTConfig(
            image_size=target_shape,  # Assuming target shape is 128
            patch_size=target_shape // 16,   # Example patch size
            num_channels=input_dim,
            hidden_size=d_model, # default 768
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=2048,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=hidden_dropout_prob,
        )

        # Load the ViT model
        self.vit = ViTModel(self.config)

        if load_pretrained:
            self.vit = self.vit.from_pretrained("google/vit-base-patch16-224-in21k")

        # Output layers
        self.pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.head = PredictHead(prediction_length, 
                                d_model, 
                                reg_head=reg_head, 
                                mask_padding=mask_padding,
                                average_mask=average_mask,
                                average_window_size=average_window_size)

    def forward(self, x, x_lengths=None, max_seq_len=None):
        # Forward pass through the ViT model
        x = self.feature_extractor(x.permute(0, 2, 1), x_lengths)

        outputs = self.vit(x).last_hidden_state  # Get the last hidden state
        x = self.pool(outputs.transpose(1, 2)).squeeze(-1)  # Global Average Pooling

        # Multi-task outputs
        out1, out2, out3 = self.head(x, x_lengths, max_seq_len)
        attn = torch.zeros(x.shape[0], 1).to(x.device)  # Placeholder for attention

        return out1, out2, out3, attn



class SwinT(nn.Module):
    def __init__(self, prediction_length, 
                 input_dim=3, 
                 hidden_dropout_prob=0.1,
                 load_pretrained=False,
                 target_shape=128,
                 tcn_params={},
                 reg_head=True,
                 mask_padding=True,
                 average_mask=False,
                 average_window_size=20
                 ):
        super(SwinT, self).__init__()

        self.reg_head = reg_head

        # use Conv2d FeatureExtractor
        self.feature_extractor = FeatureExtractorConv2d(input_dim, prediction_length, target_shape, **tcn_params)

        # SwinT configuration
        self.config = Swinv2Config(
            image_size=target_shape,  # Assuming target shape is 128
            patch_size=4,    # Patch size (for example)
            num_channels=input_dim,
            embed_dim=96,
            depths=[2, 2, 6, 2],  # Example depth configuration for each stage
            num_heads=[3, 6, 12, 24],  # Number of attention heads
            window_size=4,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            dropout_rate=hidden_dropout_prob,
            attention_probs_dropout_rate=hidden_dropout_prob,
            encoder_stride=16,
        )

        # Load the SwinT model
        self.swin = Swinv2Model(self.config)

        # if load_pretrained:
        #     self.swin = self.swin.from_pretrained("google/vit-base-patch16-224-in21k")

        # Output layers
        self.pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.head = PredictHead(prediction_length, 
                                d_model, 
                                reg_head=reg_head, 
                                mask_padding=mask_padding,
                                average_mask=average_mask,
                                average_window_size=average_window_size)

    def forward(self, x, x_lengths=None, max_seq_len=None):
        # Forward pass through the SwinT model
        x = self.feature_extractor(x.permute(0, 2, 1), x_lengths)

        outputs = self.swin(x).last_hidden_state  # Get the last hidden state
        x = self.pool(outputs.transpose(1, 2)).squeeze(-1)  # Global Average Pooling

        # Multi-task outputs
        
        attn = torch.zeros(x.shape[0], 1).to(x.device)  # Placeholder for attention
        out1, out2, out3 = self.head(x, x_lengths, max_seq_len)
        return out1, out2, out3, attn


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

class Conv1DBlock(nn.Module):
    """Convolutional block with residual connection and normalization"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 15,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Ensure padding preserves sequence length with dilation
        padding = (dilation * (kernel_size - 1)) // 2
        
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection if dimensions match
        self.residual = (in_channels == out_channels)
        
        # Projection for residual if dimensions don't match
        if not self.residual:
            self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, L]
        Returns:
            [B, C_out, L]
        """
        residual = x
        
        # Apply convolution
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Add residual connection if possible
        if self.residual:
            x = x + residual
        else:
            x = x + self.proj(residual)
            
        return x

class Conv1DTS(nn.Module):
    """
    Convolutional 1D network for time series data.
    Replaces ViT1D with a simpler convolutional architecture.
    """
    def __init__(
        self,
        time_length: int = 1221,        # Total length of input sequence
        in_channels: int = 3,           # Number of input channels
        base_filters: int = 64,         # Base number of convolutional filters
        num_layers: int = 5,            # Number of convolutional blocks
        kernel_size: int = 15,          # Kernel size for convolutions
        dropout: float = 0.1,           # Dropout rate
        second_level_mask: bool = False,          # Predict mask at second level
        supcon_loss: bool = False,      # Whether to return contrastive embedding for for supcon loss
        reg_only: bool = False,         # Whether to use regression head only
    ):
        super().__init__()
        self.time_length = time_length
        self.in_channels = in_channels
        self.base_filters = base_filters
        self.return_contrastive_embedding = supcon_loss
        self.reg_only = reg_only

        # Initial projection to embed input
        self.input_projection = nn.Conv1d(in_channels, base_filters, kernel_size=1)
        
        # Convolutional blocks with increasing dilation
        self.conv_blocks = nn.ModuleList()
        for i in range(num_layers):
            in_channels = base_filters * (2**i) if i > 0 else base_filters
            out_channels = base_filters * (2**(i+1)) if i < num_layers-1 else base_filters * (2**i)
            dilation = 2**i
            
            self.conv_blocks.append(
                Conv1DBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        
        # Global average pooling for classification and regression
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final projection for different tasks
        final_dim = base_filters * (2**(num_layers-1))

        self.regression_head = nn.Linear(final_dim, 1)
        if not self.reg_only:
            self.classification_head = nn.Linear(final_dim, 1)
            
            # Mask prediction head with a separate pathway
            if second_level_mask:
                mask_length = time_length // 20
            else:
                mask_length = time_length
            
            self.mask_prediction = nn.Sequential(
                nn.Conv1d(final_dim, final_dim // 2, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(final_dim // 2, 1, kernel_size=1),
                nn.AdaptiveAvgPool1d(mask_length)
            )
    
    def forward(self, x: torch.Tensor, labels1: Optional[torch.Tensor]=None, labels3: Optional[torch.Tensor]=None, apply_mixup: bool=False, mixup_alpha: float=0.2, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Conv1D network.
        
        Args:
            x: Input tensor of shape [B, C, L]
            
        Returns:
            Tuple of (classification_output, regression_output, mask_prediction)
        """
        # Initial projection
        x = x.permute(0, 2, 1)

        x = self.input_projection(x)
        
        # Apply convolutional blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        # Global pooling for classification and regression
        pooled = self.global_avg_pool(x).squeeze(-1)
        
        # Task-specific outputs
        if apply_mixup and self.training:
            mixed_features, mixed_labels1, mixed_labels3, mixup_lambda = latent_mixup(
                pooled, labels1, labels3, alpha=mixup_alpha
            )
            mixup_info = (mixed_labels1, mixed_labels3, mixup_lambda)
        else:
            mixed_features = pooled
            mixup_info = None

        regression_output = self.regression_head(mixed_features)

        if not self.reg_only:
            classification_output = self.classification_head(mixed_features)

            if self.return_contrastive_embedding:
                con_embed = pooled
            else:
                con_embed = None
            
            # Mask prediction
            mask_output = self.mask_prediction(x).squeeze(1)
            # else:
            #     classification_output, mask_output, con_embed, mixup_info = None, None, None, None
        else:
            classification_output, mask_output, con_embed = None, None, None

        attn = None
        return classification_output,  mask_output, regression_output, attn, con_embed, mixup_info

    
    def _init_weights(self):
        """Initialize weights for better training"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class BasicBlock1D(nn.Module):
    """Basic 1D ResNet block with two convolutional layers and residual connection"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, kernel_size=15):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                               stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, 
                               stride=1, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck1D(nn.Module):
    """Bottleneck 1D ResNet block with three convolutional layers and residual connection"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, kernel_size=15):
        super(Bottleneck1D, self).__init__()
        width = out_channels
        self.conv1 = nn.Conv1d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(width)
        self.conv2 = nn.Conv1d(width, width, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(width)
        self.conv3 = nn.Conv1d(width, out_channels * self.expansion, kernel_size=1,
                               stride=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet1D(nn.Module):
    """
    ResNet architecture for 1D time series data.
    Can be configured for different depths (18, 34, 50, etc.) by specifying block type and layers.
    """
    def __init__(
        self,
        time_length: int = 1221,        # Total length of input sequence
        in_channels: int = 3,           # Number of input channels
        base_filters: int = 64,         # Base number of filters
        block_type: str = 'basic',      # 'basic' or 'bottleneck'
        layers: list = [2, 2, 2, 2],    # Number of blocks in each layer (ResNet18 style)
        kernel_size: int = 15,          # Kernel size for convolutions
        dropout: float = 0.1,           # Dropout rate
        num_classes: int = 1,           # Number of output classes
        use_cls_token: bool = True,      # Whether to use a CLS token for classification
        second_level_mask: bool = False,          # Predict mask at second level
        supcon_loss: bool = False,      # Whether to return contrastive embedding for for supcon loss
    ):
        super(ResNet1D, self).__init__()
        
        # Initialize parameters
        self.time_length = time_length
        self.in_channels = in_channels
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.num_classes = num_classes
        self.use_cls_token = use_cls_token
        self.return_contrastive_embedding = supcon_loss
        
        # Select block type
        if block_type == 'basic':
            block = BasicBlock1D
        elif block_type == 'bottleneck':
            block = Bottleneck1D
        else:
            raise ValueError(f"Unknown block type: {block_type}")
        
        self.expansion = block.expansion
        self.inplanes = base_filters
        
        # Initial convolution and pooling
        self.input_projection = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=kernel_size, 
                      stride=2, padding=kernel_size//2, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # ResNet layers
        self.layer1 = self._make_layer(block, base_filters, layers[0], stride=1, kernel_size=3)
        self.layer2 = self._make_layer(block, base_filters*2, layers[1], stride=2, kernel_size=3)
        self.layer3 = self._make_layer(block, base_filters*4, layers[2], stride=2, kernel_size=3)
        self.layer4 = self._make_layer(block, base_filters*8, layers[3], stride=2, kernel_size=3)
        
        # In relcon, 3 layers with: 
        # 1: n=3, c_in=64, base_filters=64, k=3, s=1
        # 2: n=4, c_in=64, base_filters=128, k=4, s=1
        # 3: n=6, c_in=128, base_filters=256, k=4, s=1
        # self.layer1 = self._make_layer(block, base_filters, layers[0], stride=1, kernel_size=3)
        # self.layer2 = self._make_layer(block, base_filters*2, layers[1], stride=1, kernel_size=4)
        # self.layer3 = self._make_layer(block, base_filters*4, layers[2], stride=1, kernel_size=4)
        


        # Store all ResNet blocks for consistent interface
        self.conv_blocks = [
            self.layer1, self.layer2, self.layer3, self.layer4
        ]
        # self.conv_blocks = [
        #     self.layer1, self.layer2, self.layer3
        # ]
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Feature dimension after encoding
        self.embed_dim = base_filters * 8 * block.expansion
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output heads
        self.classification_head = nn.Linear(self.embed_dim, num_classes)
        self.regression_head = nn.Linear(self.embed_dim, 1)
        
        # Mask prediction head with a separate pathway
        if second_level_mask:
            mask_length = time_length // 20
        else:
            mask_length = time_length
        
        self.mask_prediction_head = nn.Linear(self.embed_dim, mask_length)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, kernel_size=15):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, kernel_size=kernel_size))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=kernel_size))

        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, labels1: Optional[torch.Tensor]=None, labels3: Optional[torch.Tensor]=None, apply_mixup: bool=False, mixup_alpha: float=0.2, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through ResNet1D
        
        Args:
            x: Input tensor of shape [B, C, L]
                B = batch size, C = channels, L = sequence length
                
        Returns:
            classification_output: Classification output
            mask_prediction_output: Mask prediction output
            regression_output: Regression output
        """
        x = x.permute(0, 2, 1)
        # Input projection
        x = self.input_projection(x)
        
        # Process through ResNet blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Get pooled features
        pooled = self.global_avg_pool(x).squeeze(-1)
        # features = self.dropout(features)
        # Task-specific outputs

        if apply_mixup and self.training:
            mixed_features, mixed_labels1, mixed_labels3, mixup_lambda = latent_mixup(
                pooled, labels1, labels3, alpha=mixup_alpha
            )
            mixup_info = (mixed_labels1, mixed_labels3, mixup_lambda)
        else:
            mixed_features = pooled
            mixup_info = None
        
        # Classification output
        classification_output = self.classification_head(mixed_features)
        if self.return_contrastive_embedding:
            con_embed = self.pooled
        else:
            con_embed = None
        
        # Mask prediction output
        mask_output = self.mask_prediction_head(mixed_features)
        
        # Regression output
        regression_output = self.regression_head(mixed_features)
        
        attn = None
        return classification_output,  mask_output, regression_output, attn, con_embed, mixup_info


class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels,kernel_size, stride, padding)
        self.bn1 = nn.InstanceNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels,kernel_size, 1, padding)
        self.bn2 = nn.InstanceNorm1d(out_channels)
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
    def forward(self, x):
        return self.stage(x)

class DecoderBlock(nn.Module):
    def __init__(self, up_in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose1d(up_in_channels, skip_channels, kernel_size=2, stride=2)
        self.resblock = ResBlock1D(skip_channels * 2, out_channels)
    def forward(self, x, skip):
        x = self.up(x)
        # Align lengths for concat
        if x.size(-1) < skip.size(-1):
            x = F.pad(x, (0, skip.size(-1) - x.size(-1)))
        elif x.size(-1) > skip.size(-1):
            x = x[..., :skip.size(-1)]
        x = torch.cat([x, skip], dim=1)
        x = self.resblock(x)
        return x

class OutModule2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, norm, method):
        super(OutModule2, self).__init__()
        if norm=='GN':
            self.norm = nn.GroupNorm(num_groups=4, num_channels=hidden_dim)
        elif norm=='BN':
            self.norm = nn.BatchNorm1d(hidden_dim)
        else:
            self.norm = nn.InstanceNorm1d(hidden_dim,track_running_stats=False)
            
        if method=='FC':
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            # Second layer: hidden to output
            self.fc2 = nn.Linear(hidden_dim,hidden_dim)
            # self.fc3 = nn.Linear(hidden_dim,hidden_dim)
            # self.fc4 = nn.Linear(hidden_dim,hidden_dim)
            self.fc5 = nn.Linear(hidden_dim, output_dim)
        else:
            self.fc1 = nn.Conv1d(input_dim, hidden_dim,1)
            # Second layer: hidden to output
            self.fc2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
            # self.fc3 = nn.Conv1d(hidden_dim, hidden_dim, 1)
            # self.fc4 = nn.Conv1d(hidden_dim, hidden_dim, 1)
            self.fc5 = nn.Conv1d(hidden_dim,output_dim, 1)
        
    def forward(self, x):
        if self.norm is not None:
            x=self.norm(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        # x = self.fc3(x)
        # x = F.relu(x)
        # x = self.fc4(x)
        # x = F.relu(x)
        x = self.fc5(x)
        return x

class ResNetUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, features=[64, 128, 256, 512], blocks_per_stage=[2, 2, 2, 2]):
        super().__init__()
        self.init_conv = nn.Conv1d(in_channels, features[0], kernel_size=7, stride=2, padding=3)
        self.init_bn = nn.InstanceNorm1d(features[0])
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

class MambaDecoderBlock(nn.Module):
    """
    Mamba-based decoder block for symmetric design with ResNet encoder.
    Replaces ResBlock1D with Mamba processing while maintaining upsampling.
    """
    def __init__(self, up_in_channels, skip_channels, out_channels, kernel_size_mba=7, drop_path_rate=0.3):
        super().__init__()
        # Upsampling layer
        # self.up = nn.ConvTranspose1d(up_in_channels, skip_channels, kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.channel_align = nn.Conv1d(up_in_channels+skip_channels, out_channels, 1)

        # Channel alignment before Mamba processing
        # self.channel_align = nn.Conv1d(skip_channels * 2, out_channels, 1)
        # out_channels = skip_channels * 2
        self.norm = nn.InstanceNorm1d(out_channels)

        # Mamba block for processing concatenated features
        self.mamba_block = AttModule_mamba_causal(
            dilation=1,  # No dilation needed for decoder
            in_channels=out_channels,
            out_channels=out_channels,
            att_type='sliding_att',
            stage='decoder',
            alpha=1,
            drop_path_rate=drop_path_rate,
            kernel_size=kernel_size_mba
        )

    def forward(self, x, skip, mask=None):
        # Upsample decoder features
        x = self.up(x)

        # Align sequence lengths for concatenation

        if x.size(-1) < skip.size(-1):
            x = F.pad(x, (0, skip.size(-1) - x.size(-1)))
        elif x.size(-1) > skip.size(-1):
            x = x[:, :, :skip.size(-1)]

        # Concatenate upsampled features with skip connection
        x = torch.cat([x, skip], dim=1)  # [B, skip_channels*2, seq_len]

        # Channel alignment and normalization
        x = self.channel_align(x)  # [B, out_channels, seq_len]
        x = self.norm(x)

        if mask is None:
            batch_size, seq_len = x.size(0), x.size(2)
            mask = torch.ones(batch_size, seq_len, device=x.device, dtype=torch.float32)
        # Mamba processing with optional mask
        x = self.mamba_block(x, None, mask)

        return x

class MambaDecoderStage(nn.Module):
    """
    Multi-block Mamba decoder stage to match encoder complexity.
    """
    def __init__(self, up_in_channels, skip_channels, out_channels, num_blocks=2,
                 kernel_size_mba=7, drop_path_rate=0.3):
        super().__init__()

        # First block handles upsampling and skip connection
        self.first_block = MambaDecoderBlock(
            up_in_channels, skip_channels, out_channels,
            kernel_size_mba, drop_path_rate
        )

        # Additional Mamba blocks for processing
        # self.additional_blocks = nn.ModuleList([
        #     AttModule_mamba(
        #         dilation=1,
        #         in_channels=out_channels,
        #         out_channels=out_channels,
        #         r1=2, r2=2,
        #         att_type='sliding_att',
        #         stage='decoder',
        #         alpha=1,
        #         drop_path_rate=drop_path_rate,
        #         kernel_size=kernel_size_mba
        #     ) for _ in range(num_blocks - 1)
        # ])

    def forward(self, x, skip, mask=None):
        # First block with skip connection
        x = self.first_block(x, skip, mask)

        # Additional processing blocks

        # for block in self.additional_blocks:
        #     if mask is None:
        #         batch_size, seq_len = x.size(0), x.size(2)
        #         block_mask = torch.ones(batch_size, seq_len, device=x.device, dtype=torch.float32)
        #     else:
        #         block_mask = mask
        #     x = block(x, None, block_mask)

        return x

class ResNetMambaUNet(nn.Module):
    """
    Hybrid encoder-decoder with ResNet encoder and Mamba decoder.
    Maintains symmetric design with equivalent complexity at each stage.
    """
    def __init__(self, in_channels=3, num_classes=1,
                 features=[64, 128, 256, 512],
                 blocks_per_stage=[1, 1, 1, 1],  # Reduced from [2,2,2,2] to prevent complexity
                 kernel_size_mba=7,
                 drop_path_rate=0.2,  # Reduced from 0.3
                 dropout_rate=0.2,
                 use_mask_in_decoder=True):
        super().__init__()

        self.use_mask_in_decoder = use_mask_in_decoder

        # Initial convolution (same as ResNetUNet)
        self.init_conv = nn.Conv1d(in_channels, features[0], kernel_size=7, stride=2, padding=3)
        self.init_bn = nn.InstanceNorm1d(features[0])
        self.init_relu = nn.ReLU(inplace=True)
        self.init_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet Encoder (same as ResNetUNet)
        self.encoder_stages = nn.ModuleList()
        prev_channels = features[0]
        for idx, (feat, n_blocks) in enumerate(zip(features, blocks_per_stage)):
            self.encoder_stages.append(
                EncoderStage(prev_channels, feat, num_blocks=n_blocks, downsample=(idx != 0))
            )
            prev_channels = feat

        # Bottleneck with Mamba processing - reduced alpha for stability
        self.bottleneck_conv = ResBlock1D(features[-1], features[-1]*2, stride=1)
        self.bottleneck_mamba = MaskMambaBlock(
            features[-1]*2, 
            drop_path_rate=min(drop_path_rate, 0.1),  # Very conservative drop path
            kernel_size=kernel_size_mba
        )

        # Mamba Decoder - use 3 stages to match encoder skip connections
        # Skip the deepest level since bottleneck already processed it
        rev_features = features[-2::-1]  # Exclude features[-1], reverse the rest
        rev_blocks = blocks_per_stage[-2::-1]  # Exclude blocks_per_stage[-1], reverse
        up_in_channels = features[-1]*2
        self.decoder_stages = nn.ModuleList()

        for skip_channels, n_blocks in zip(rev_features, rev_blocks):
            self.decoder_stages.append(
                MambaDecoderStage(
                    up_in_channels, skip_channels, skip_channels,
                    num_blocks=n_blocks,
                    kernel_size_mba=kernel_size_mba,
                    drop_path_rate=min(drop_path_rate, 0.2)  # Cap drop path rate for stability
                )
            )
            up_in_channels = skip_channels

        # Multi-task output heads (same as ResNetUNet)
        self.out1 = OutModule2(features[0], features[0], num_classes, 'IN', 'FC')
        self.out2 = OutModule2(features[0], features[0], num_classes, 'IN', 'Conv')
        self.out3 = OutModule2(features[0], features[0], num_classes, 'IN', 'FC')
        

    def forward(self, x, x_lengths, labels1=None, labels3=None, apply_mixup=False, mixup_alpha=0.2):
        batch_size, seq_len, channels = x.size()
        mask = create_mask(torch.tensor(x_lengths, device=x.device), seq_len, batch_size, x.device)

        contrastive_embedding = None
        x = x.permute(0, 2, 1)  # [B, C, L]
        input_length = x.size(-1)

        # Initial processing
        x = self.init_conv(x)
        x = self.init_bn(x)
        x = self.init_relu(x)
        x = self.init_pool(x)

        # Encoder with skip connections
        skips = []
        for stage in self.encoder_stages:
            x = stage(x)
            skips.append(x)

        # Bottleneck
        x = self.bottleneck_conv(x)

        # Create mask for current sequence length
        current_length = x.size(-1)
        current_mask = F.interpolate(
            mask.unsqueeze(1).float(),
            size=current_length,
            mode='nearest'
        ).squeeze(1)

        # Use mask in bottleneck only if decoder masking is enabled
        bottleneck_mask = current_mask if self.use_mask_in_decoder else None
        x = self.bottleneck_mamba(x, bottleneck_mask)

        # Mamba decoder with skip connections
        # Skip the deepest encoder output since bottleneck processed it
        for stage, skip in zip(self.decoder_stages, skips[-2::-1]):
            if self.use_mask_in_decoder:
                # Interpolate mask to match skip connection sequence length
                # The decoder stage will upsample x to match skip.size(-1)
                current_length = skip.size(-1)
                decoder_mask = F.interpolate(
                    mask.unsqueeze(1).float(),
                    size=current_length,
                    mode='nearest'
                ).squeeze(1)
            else:
                decoder_mask = None

            x = stage(x, skip, decoder_mask)

        # Final upsampling to match input length
        if x.size(-1) < input_length:
            x = F.pad(x, (0, input_length - x.size(-1)))
        elif x.size(-1) > input_length:
            x = x[..., :input_length]

        # Match ResNetUNet output format
        attention_weights = torch.zeros(batch_size, seq_len, 1, device=x.device)
        pooled_features = x[:, :, 0]  # Use first position features for classification/regression
        mixed_features = pooled_features
        mixup_info = None

        # Multi-task outputs (matching ResNetUNet)
        x1 = self.out1(mixed_features)  # Classification from pooled features
        x2 = self.out2(x)               # Sequence labeling from full sequence
        x3 = self.out3(mixed_features)  # Regression from pooled features

        # Apply mask to sequence output (matching ResNetUNet)
        mask = mask.reshape(-1).bool()
        x2 = x2.view(-1)
        x2 = x2[mask]
        attention_weights = attention_weights.reshape(-1)
        attention_weights = attention_weights[mask]

        return x1, x2, x3, attention_weights, contrastive_embedding, mixup_info
    

if __name__ == "__main__":
    print("Load success!")