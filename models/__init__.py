# -*- coding: utf-8 -*-
"""
Models module for scratch detection with wearable sensors.

This module contains all deep learning model architectures organized into
separate submodules for better maintainability and clarity.

Submodules:
-----------
- components: Shared building blocks (TCN, feature extractors, etc.)
- mamba_blocks: Mamba-specific blocks and the core MBA module
- attention: Attention and pooling modules
- resmamba: Primary ResMamba models (MBA_tsm, MBA_tsm_with_padding, etc.)
- encoder_decoder: Encoder-decoder and sequence-to-sequence models
- pretraining: Models for self-supervised pretraining
- baselines: Baseline and legacy models (RNN, CNN, MTCN, etc.)
- resnet: 1D ResNet architectures
- unet: U-Net and decoder-based models
- specialized: Specialized architectures (PatchTST, ViT, Swin, etc.)
- conv1d: 1D convolution-based models
- setup: Model factory function for instantiating models
"""

# Import all classes and functions for convenient access
from .components import (
    Chomp1d,
    TemporalBlock,
    UpsampleBlock,
    FeatureExtractorConv2d,
    FeatureExtractor,
    ConvProjection,
    FeatureExtractorForPretraining,
)

from .mamba_blocks import (
    ConvFeedForward,
    drop_path,
    AffineDropPath,
    MaskMambaBlock,
    MBA,
)

from .attention import (
    AttentionHelper,
    AttLayer,
    AttModule,
    AttModule_mamba,
    GatedAttentionPoolingMIL,
    MultiHeadSelfAttentionPooling,
    MaskedMaxAvgPooling,
)

from .resmamba import (
    MBA_tsm,
    MBA_tsm_with_padding,
    MBA_patch,
    MBA4TSO,
    MBA_v1,
    MBA_v1_ForPretraining,
    OutModule,
    latent_mixup,
    masked_avg_pool,
    create_mask,
)

from .encoder_decoder import (
    PositionalEncoding,
    Encoder,
    Decoder,
    MBA_encoder_decoder,
    MBA_tsm_encoder_decoder_ch_bottleneck,
    MBA_tsm_encoder_decoder_seq_bottleneck,
    MBA_tsm_encoder_decoder_progressive_with_skip_connection,
    MBA_tsm_encoder_decoder_progressive,
)

from .pretraining import (
    MBATSMForPretraining,
)

from .baselines import (
    RNN,
    BiRNN,
    CNN,
    ResidualBlock,
    ResCNN_old,
    MLP,
    ResCNN,
    LSTMModel,
    TCNLayer,
    ResTCNLayer,
    BiTCNLayer,
    MTCN,
    PositionalEmbedding,
    MTCNA2,
    BiMTCN,
    BiMambaEncoder,
    tcn_learner,
    mba_learner,
    hybrid,
)

from .resnet import (
    BasicBlock1D,
    Bottleneck1D,
    ResNet1D,
    ResBlock1D,
)

from .unet import (
    EncoderStage,
    DecoderBlock,
    OutModule2,
    ResNetUNet,
    MambaDecoderBlock,
    MambaDecoderStage,
    ResNetMambaUNet,
    EfficientUNet,
)

from .specialized import (
    PatchEmbedding,
    PatchTSTHead,
    PatchTSTNS,
    PredictHead,
    GASFFeatureExtractor,
    ViT,
    SwinT,
)

from .conv1d import (
    Conv1DBlock,
    Conv1DTS,
)

from .setup import (
    setup_model,
)

from .vit1d import ViT1D

__all__ = [
    # Components
    'Chomp1d',
    'TemporalBlock',
    'UpsampleBlock',
    'FeatureExtractorConv2d',
    'FeatureExtractor',
    'ConvProjection',
    'FeatureExtractorForPretraining',
    # Mamba blocks
    'ConvFeedForward',
    'drop_path',
    'AffineDropPath',
    'MaskMambaBlock',
    'MBA',
    # Attention
    'AttentionHelper',
    'AttLayer',
    'AttModule',
    'AttModule_mamba',
    'GatedAttentionPoolingMIL',
    'MultiHeadSelfAttentionPooling',
    'MaskedMaxAvgPooling',
    # ResMamba
    'MBA_tsm',
    'MBA_tsm_with_padding',
    'MBA_patch',
    'MBA4TSO',
    'MBA_v1',
    'MBA_v1_ForPretraining',
    'OutModule',
    'latent_mixup',
    'masked_avg_pool',
    'create_mask',
    # Encoder-Decoder
    'PositionalEncoding',
    'Encoder',
    'Decoder',
    'MBA_encoder_decoder',
    'MBA_tsm_encoder_decoder_ch_bottleneck',
    'MBA_tsm_encoder_decoder_seq_bottleneck',
    'MBA_tsm_encoder_decoder_progressive_with_skip_connection',
    'MBA_tsm_encoder_decoder_progressive',
    # Pretraining
    'MBATSMForPretraining',
    # Baselines
    'RNN',
    'BiRNN',
    'CNN',
    'ResidualBlock',
    'ResCNN_old',
    'MLP',
    'ResCNN',
    'LSTMModel',
    'TCNLayer',
    'ResTCNLayer',
    'BiTCNLayer',
    'MTCN',
    'PositionalEmbedding',
    'MTCNA2',
    'BiMTCN',
    'BiMambaEncoder',
    'tcn_learner',
    'mba_learner',
    'hybrid',
    # ResNet
    'BasicBlock1D',
    'Bottleneck1D',
    'ResNet1D',
    'ResBlock1D',
    # U-Net
    'EncoderStage',
    'DecoderBlock',
    'OutModule2',
    'ResNetUNet',
    'MambaDecoderBlock',
    'MambaDecoderStage',
    'ResNetMambaUNet',
    'EfficientUNet',
    # Specialized
    'PatchEmbedding',
    'PatchTSTHead',
    'PatchTSTNS',
    'PredictHead',
    'GASFFeatureExtractor',
    'ViT',
    'SwinT',
    'ViT1D',
    # Conv1D
    'Conv1DBlock',
    'Conv1DTS',
    # Setup/Factory
    'setup_model',
]
