# -*- coding: utf-8 -*-
"""
Data module - Loading, preprocessing, padding, augmentation, and batching
"""

# Data loading
from .loading import (
    load_sequence_data,
    load_data,
    load_data_pretrain,
    load_data_tso_patch,
    load_data_tso_patch_biobank,
    load_data_tso,
    load_sequence_data_nsucl,
    load_data_nsucl,
    load_data_raw,
    load_data_from_h5,
)

# UKB data loading (new Dataset/DataLoader pipeline — see module docstring)
from .loading_ukb_h5 import (
    UKBPretrainDataset,
    collate_ukb_batch,
    make_ukb_collate_fn,
    read_ukb_h5_metadata,
    list_ukb_segment_indices,
    split_indices_by_segment,
)

# Preprocessing
from .preprocessing import (
    df_subset,
    df_subset_segments,
    remove_missmatch_labels,
    filter_inbed,
    filter_segment_inbed,
    add_groups,
    get_dominant_hand,
    add_spectrum,
    high_pass_filter,
    extract_gravity,
    assign_incremental_numbers,
    generate_spaced_integers,
)

# Padding
from .padding import (
    add_padding,
    add_padding_with_position,
    add_padding_pretrain,
    add_padding_TSO,
    add_padding_tso_patch,
    add_padding_tso_patch_h5,
    random_patch_masking,
    generate_time_cyclic,
)

# Augmentation
from .augmentation import (
    augment_dataset,
    augment_iteration,
    augment_dataset_stream,
    augment_iteration_stream,
    get_subsegment,
)

# Batching
from .batching import (
    batch_generator,
    get_nb_steps,
)

__all__ = [
    # Loading
    'load_sequence_data',
    'load_data',
    'load_data_pretrain',
    'load_data_tso_patch',
    'load_data_tso_patch_biobank',
    'load_data_tso',
    'load_sequence_data_nsucl',
    'load_data_nsucl',
    'load_data_raw',
    'load_data_from_h5',
    # Preprocessing
    'df_subset',
    'df_subset_segments',
    'remove_missmatch_labels',
    'filter_inbed',
    'filter_segment_inbed',
    'add_groups',
    'get_dominant_hand',
    'add_spectrum',
    'high_pass_filter',
    'extract_gravity',
    'assign_incremental_numbers',
    'generate_spaced_integers',
    # Padding
    'add_padding',
    'add_padding_with_position',
    'add_padding_pretrain',
    'add_padding_TSO',
    'add_padding_tso_patch',
    'add_padding_tso_patch_h5',
    'random_patch_masking',
    'generate_time_cyclic',
    # Augmentation
    'augment_dataset',
    'augment_iteration',
    'augment_dataset_stream',
    'augment_iteration_stream',
    'get_subsegment',
    # Batching
    'batch_generator',
    'get_nb_steps',
]
