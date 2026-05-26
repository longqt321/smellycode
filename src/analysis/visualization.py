import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, labels: list):
    """Plot ROC curve for multi-label classification."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for i, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_scores[:, i])
        ax.plot(fpr, tpr, label=f'{label} (AUC={np.trapezoid(tpr, fpr):.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_precision_recall_curve(y_true: np.ndarray, y_scores: np.ndarray, labels: list):
    """Plot Precision-Recall curve for multi-label classification."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for i, label in enumerate(labels):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_scores[:, i])
        avg_precision = average_precision_score(y_true[:, i], y_scores[:, i])
        ax.plot(recall, precision, label=f'{label} (AP={avg_precision:.3f})')
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    return fig
