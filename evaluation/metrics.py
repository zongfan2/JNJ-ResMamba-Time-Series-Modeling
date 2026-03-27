# -*- coding: utf-8 -*-
"""
Split from DL_helpers.py - Modular code structure
@author: MBoukhec (original)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import torch

def plot_confusion_metrics(y_test, y_pred, output_filepath):
    try:
        # Calculate the confusion matrix
        #labels = sorted(list(set(y_test)))
        labels=[False,True]
        cm = metrics.confusion_matrix(y_test, y_pred, labels=labels)

        # Convert the confusion matrix into a pandas DataFrame
        df_cm = pd.DataFrame(cm, index=labels, columns=labels)

        # Plot the confusion matrix using seaborn
        plt.figure(figsize=(10, 7))
        sns.heatmap(df_cm, annot=True, fmt="g", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(output_filepath, dpi=300)
        plt.show()
        # Optionally, you can print the classification report
        #print(classification_report(y_test, y_pred))
    except Exception as e:
        raise Exception(f"Failed to plot confusion matrix. Exception: {e}")
    


def plot_learning_curves(learning_metrics, output_filepath):
    # Extract the metrics
    train_losses = np.array(learning_metrics.get('train_losses', []), dtype=float)
    train_accuracies = np.array(learning_metrics.get('train_accuracies', []), dtype=float)
    train_F1s=np.array(learning_metrics.get('train_F1s', []), dtype=float)
    val_losses = np.array(learning_metrics.get('val_losses', []), dtype=float)
    val_accuracies = np.array(learning_metrics.get('val_accuracies', []), dtype=float)
    val_F1s=np.array(learning_metrics.get('val_F1s', []), dtype=float)

    epochs = range(1, len(train_losses) + 1)

    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))  # Changed to 1 row, 2 columns

    # Plot training and validation loss
    ax1.plot(epochs, np.array(train_losses, dtype=float), 'b-', label='Training loss')
    ax1.plot(epochs, np.array(val_losses, dtype=float), 'r-', label='Validation loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot training and validation accuracy
    ax2.plot(epochs, np.array(train_accuracies, dtype=float), 'b-', label='Training accuracy')
    ax2.plot(epochs, np.array(val_accuracies, dtype=float), 'r-', label='Validation accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    # Plot validation F1
    ax3.plot(epochs, np.array(train_F1s, dtype=float), 'b-', label='Train F1')
    ax3.plot(epochs, np.array(val_F1s, dtype=float), 'r-', label='Validation F1')
    ax3.set_title('Train F1')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('F1')
    ax3.legend()

    # Layout adjustment and saving the figure
    plt.tight_layout()
    plt.savefig(output_filepath, dpi=300)
    plt.show()
    plt.close()


def plot_roc_precision_recall_auc(y_test, y_score, output_filepath):
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(14, 5))

    # Subplot 1: ROC curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'(AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.text(0.6, 0.2, f'Counts: {len(y_test)}', fontsize=12)

    # Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    average_precision = average_precision_score(y_test, y_score)

    # Subplot 2: Precision-Recall curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f'AP = {average_precision:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.text(0.6, 0.2, f'Counts: {len(y_test)}', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_filepath, dpi=300)
    plt.show()


def calculate_metrics_nn(actuals, predictions, classification=True):
    if classification:
        # Ensure predictions are integers
        predictions = predictions.astype(int)
        actuals = actuals.astype(int)
        # Calculating various classification metrics
        metrics_values = {
            'accuracy': metrics.accuracy_score(actuals, predictions),
            'balanced_accuracy': metrics.balanced_accuracy_score(actuals, predictions),
            # 'precision': metrics.precision_score(actuals, predictions, zero_division=0),
            'precision_macro': metrics.precision_score(actuals, predictions, average='macro', zero_division=0),
            'precision_micro': metrics.precision_score(actuals, predictions, average='micro', zero_division=0),
            'precision_weighted': metrics.precision_score(actuals, predictions, average='weighted', zero_division=0),
            # 'recall': metrics.recall_score(actuals, predictions, zero_division=0),
            'recall_macro': metrics.recall_score(actuals, predictions, average='macro', zero_division=0),
            'recall_micro': metrics.recall_score(actuals, predictions, average='micro', zero_division=0),
            'recall_weighted': metrics.recall_score(actuals, predictions, average='weighted', zero_division=0),
            # 'f1_score': metrics.f1_score(actuals, predictions, zero_division=0),
            'f1_score_macro': metrics.f1_score(actuals, predictions, average='macro', zero_division=0),
            'f1_score_micro': metrics.f1_score(actuals, predictions, average='micro', zero_division=0),
            'f1_score_weighted': metrics.f1_score(actuals, predictions, average='weighted', zero_division=0),
            'roc_auc': metrics.roc_auc_score(actuals, predictions) if len(set(actuals)) == 2 else None,
            'r2': metrics.r2_score(actuals, predictions),
            'mse': metrics.mean_squared_error(actuals, predictions),
            'classification_report': metrics.classification_report(actuals, predictions, output_dict=True, zero_division=0)
        }
    else:
        metrics_values = {
        'explained_variance_score': metrics.explained_variance_score(actuals, predictions),
        'd2_absolute_error_score': metrics.d2_absolute_error_score(actuals, predictions),
        'r2': metrics.r2_score(actuals, predictions),
        'mse': metrics.mean_squared_error(actuals, predictions)}
            
    return metrics_values


def plot_tso_learning_curves(history, output_filepath):
    """
    Plot comprehensive training history for TSO prediction task.

    Args:
        history: Dictionary containing training history with keys:
            - train_loss, val_loss
            - train_accuracy, val_accuracy
            - train_f1_avg, val_f1_avg
            - train_f1_other, train_f1_nonwear, train_f1_tso
            - val_f1_other, val_f1_nonwear, val_f1_tso
        output_filepath: Path to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    epochs = range(1, len(history['train_loss']) + 1)

    # Total loss
    axes[0, 0].plot(epochs, history['train_loss'], label='Train', linewidth=2, color='#1f77b4')
    axes[0, 0].plot(epochs, history['val_loss'], label='Val', linewidth=2, color='#ff7f0e')
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Total Loss', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(epochs, history['train_accuracy'], label='Train', linewidth=2, color='#1f77b4')
    axes[0, 1].plot(epochs, history['val_accuracy'], label='Val', linewidth=2, color='#ff7f0e')
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Accuracy', fontsize=11)
    axes[0, 1].set_title('Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # Average F1
    axes[0, 2].plot(epochs, history['train_f1_avg'], label='Train', linewidth=2, color='#1f77b4')
    axes[0, 2].plot(epochs, history['val_f1_avg'], label='Val', linewidth=2, color='#ff7f0e')
    axes[0, 2].set_xlabel('Epoch', fontsize=11)
    axes[0, 2].set_ylabel('F1 Score', fontsize=11)
    axes[0, 2].set_title('Average F1 Score', fontsize=12, fontweight='bold')
    axes[0, 2].legend(fontsize=10)
    axes[0, 2].grid(True, alpha=0.3)

    # F1 per class - Train
    if 'train_f1_other' in history:
        axes[1, 0].plot(epochs, history['train_f1_other'], label='Other', linewidth=2, color='#2ca02c')
        axes[1, 0].plot(epochs, history['train_f1_nonwear'], label='Non-wear', linewidth=2, color='#d62728')
    axes[1, 0].plot(epochs, history['train_f1_tso'], label='TSO', linewidth=2, color='#9467bd')
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('F1 Score', fontsize=11)
    axes[1, 0].set_title('Train F1 per Class', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # F1 per class - Val
    if 'val_f1_other' in history:
        axes[1, 1].plot(epochs, history['val_f1_other'], label='Other', linewidth=2, color='#2ca02c')
        axes[1, 1].plot(epochs, history['val_f1_nonwear'], label='Non-wear', linewidth=2, color='#d62728')
    axes[1, 1].plot(epochs, history['val_f1_tso'], label='TSO', linewidth=2, color='#9467bd')
    axes[1, 1].set_xlabel('Epoch', fontsize=11)
    axes[1, 1].set_ylabel('F1 Score', fontsize=11)
    axes[1, 1].set_title('Val F1 per Class', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    # Combined F1 vs Accuracy comparison
    axes[1, 2].plot(epochs, history['train_f1_avg'], label='Train F1', linewidth=2,
                   linestyle='--', color='#1f77b4', alpha=0.7)
    axes[1, 2].plot(epochs, history['val_f1_avg'], label='Val F1', linewidth=2,
                   linestyle='--', color='#ff7f0e', alpha=0.7)
    axes[1, 2].plot(epochs, history['train_accuracy'], label='Train Acc', linewidth=2, color='#1f77b4')
    axes[1, 2].plot(epochs, history['val_accuracy'], label='Val Acc', linewidth=2, color='#ff7f0e')
    axes[1, 2].set_xlabel('Epoch', fontsize=11)
    axes[1, 2].set_ylabel('Score', fontsize=11)
    axes[1, 2].set_title('F1 vs Accuracy', fontsize=12, fontweight='bold')
    axes[1, 2].legend(fontsize=9)
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_filepath, dpi=150, bbox_inches='tight')
    plt.close()

