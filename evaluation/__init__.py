# -*- coding: utf-8 -*-
"""
Evaluation module - Metrics, visualization, and post-processing
"""

from .metrics import (
    calculate_metrics_nn,
    plot_confusion_metrics,
    plot_learning_curves,
    plot_roc_precision_recall_auc,
    plot_tso_learning_curves,
)

from .postprocessing import (
    smooth_predictions,
    smooth_predictions_combined,
    smooth_binary_series,
    enforce_single_tso_period,
    batch_enforce_single_tso,
    remove_padding,
    seq_to_seconds,
)

__all__ = [
    # Metrics
    'calculate_metrics_nn',
    'plot_confusion_metrics',
    'plot_learning_curves',
    'plot_roc_precision_recall_auc',
    'plot_tso_learning_curves',
    # Post-processing
    'smooth_predictions',
    'smooth_predictions_combined',
    'smooth_binary_series',
    'enforce_single_tso_period',
    'batch_enforce_single_tso',
    'remove_padding',
    'seq_to_seconds',
]
