# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from typing import Optional, Tuple
import numpy as np
from transformers import PatchTSTConfig, PatchTSTModel


# Import all models from submodules
from .baselines import (
    RNN, BiRNN, CNN, ResidualBlock, ResCNN_old, MLP, ResCNN, 
    LSTMModel, TCNLayer, ResTCNLayer, BiTCNLayer, MTCN, 
    PositionalEmbedding, MTCNA2, BiMTCN, BiMambaEncoder,
    tcn_learner, mba_learner, hybrid
)
from .mamba_blocks import MBA, ConvFeedForward, AffineDropPath, MaskMambaBlock, drop_path
from .attention import (
    AttentionHelper, AttLayer, AttModule, AttModule_mamba,
    GatedAttentionPoolingMIL,
    MultiHeadSelfAttentionPooling, MaskedMaxAvgPooling
)
from .components import FeatureExtractor, FeatureExtractorConv2d, FeatureExtractorForPretraining
from .resmamba import MBA_tsm, MBA_tsm_with_padding, MBA_patch, MBA4TSO, latent_mixup, masked_avg_pool, create_mask
from .encoder_decoder import (
    MBA_encoder_decoder,
    MBA_tsm_encoder_decoder_ch_bottleneck,
    MBA_tsm_encoder_decoder_seq_bottleneck,
    MBA_tsm_encoder_decoder_progressive_with_skip_connection,
    MBA_tsm_encoder_decoder_progressive,
    Encoder, Decoder, PositionalEncoding
)
from .pretraining import MBATSMForPretraining
from .resnet import ResNet1D, ResBlock1D, BasicBlock1D, Bottleneck1D
from .unet import EfficientUNet, ResNetUNet, ResNetMambaUNet, EncoderStage, DecoderBlock, OutModule2
from .specialized import PatchTSTHead, PatchTSTNS, PredictHead, GASFFeatureExtractor, ViT, SwinT, PatchEmbedding
from .conv1d import Conv1DBlock, Conv1DTS

try:
    from .vit1d import ViT1D
except ImportError:
    ViT1D = None

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

            net_type = best_params.get("net_type", "progressive_skip")
            # Map legacy net_type names to bottleneck_type parameter
            bottleneck_map = {
                "progressive_with_skip_connection": "progressive_skip",
                "progressive": "progressive",
                "ch_bottleneck": "channel",
                "seq_bottleneck": "sequence",
            }
            bottleneck_type = bottleneck_map.get(net_type, net_type)
            if bottleneck_type not in ("channel", "sequence", "progressive_skip", "progressive"):
                raise ValueError(f"net_type must be one of {list(bottleneck_map.keys())} when using mba_encoder_decoder, got '{net_type}'")

            model = MBA_encoder_decoder(input_tensor_size,
                            bottleneck_type=bottleneck_type,
                            num_encoder_layers=num_encoder_layers,
                            num_decoder_layers=num_decoder_layers,
                            drop_path_rate=drop_path_rate,
                            kernel_size_mba=kernel_size_mba,
                            dropout_rate=dropout_rate,
                            max_seq_len=max_seq_len,
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
