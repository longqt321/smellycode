"""Advanced threshold optimization using Bayesian optimization and ROC analysis."""
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from scipy.optimize import minimize
from typing import List, Optional, Tuple


def tune_thresholds_bayesian(
    probs: np.ndarray,
    labels: np.ndarray,
    metric: str = 'f1_macro',
    n_iterations: int = 50
) -> List[float]:
    """
    Find optimal thresholds using Bayesian-inspired search with local refinement.
    
    This method optimizes thresholds by:
    1. Starting from best grid-search thresholds as initial point
    2. Using gradient-free optimization to refine thresholds
    3. Supporting multiple metrics (F1 macro, F1 micro, balanced accuracy)
    
    Args:
        probs: Predicted probabilities of shape (N, num_classes)
        labels: Ground truth labels of shape (N, num_classes)
        metric: Metric to optimize ('f1_macro', 'f1_micro', 'balanced_acc')
        n_iterations: Number of optimization iterations
        
    Returns:
        List of optimal thresholds for each class
    """
    num_classes = labels.shape[1]
    
    def objective(thresholds):
        """Negative metric value to minimize."""
        preds = np.zeros_like(probs)
        for i in range(num_classes):
            preds[:, i] = (probs[:, i] >= thresholds[i]).astype(int)
        
        if metric == 'f1_macro':
            score = f1_score(labels, preds, average='macro', zero_division=0)
        elif metric == 'f1_micro':
            score = f1_score(labels, preds, average='micro', zero_division=0)
        elif metric == 'balanced_acc':
            from sklearn.metrics import balanced_accuracy_score
            score = balanced_accuracy_score(labels, preds, sample_weight=None)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return -score  # Minimize negative = maximize metric
    
    # Get good initial thresholds via coarse grid search
    candidates = np.arange(0.1, 0.9, 0.1)
    init_thresholds = []
    for i in range(num_classes):
        best_t, best_f1 = 0.5, 0.0
        for t in candidates:
            f1 = f1_score(labels[:, i], (probs[:, i] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        init_thresholds.append(best_t)
    
    init_thresholds = np.array(init_thresholds)
    
    # Bounds for thresholds
    bounds = [(0.05, 0.95)] * num_classes
    
    # Use Nelder-Mead simplex method (gradient-free)
    result = minimize(
        objective,
        init_thresholds,
        method='Nelder-Mead',
        options={'maxiter': n_iterations, 'xatol': 0.01}
    )
    
    # Clip to valid range
    optimal_thresholds = np.clip(result.x, 0.05, 0.95).tolist()
    
    return optimal_thresholds


def tune_thresholds_roc_based(
    probs: np.ndarray,
    labels: np.ndarray,
    criterion: str = 'youden'
) -> List[float]:
    """
    Find optimal thresholds using ROC curve analysis.
    
    Args:
        probs: Predicted probabilities of shape (N, num_classes)
        labels: Ground truth labels of shape (N, num_classes)
        criterion: Criterion for threshold selection:
                   - 'youden': Maximize Youden's J statistic (sensitivity + specificity - 1)
                   - 'closest_topleft': Minimize distance to top-left corner
                   
    Returns:
        List of optimal thresholds for each class
    """
    from sklearn.metrics import roc_curve
    
    thresholds = []
    for i in range(labels.shape[1]):
        fpr, tpr, thresh = roc_curve(labels[:, i], probs[:, i])
        
        if criterion == 'youden':
            youden_j = tpr - fpr
            best_idx = np.argmax(youden_j)
        elif criterion == 'closest_topleft':
            distances = np.sqrt((1 - tpr)**2 + fpr**2)
            best_idx = np.argmin(distances)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        
        thresholds.append(float(thresh[best_idx]))
    
    return thresholds


def tune_thresholds_grid(
    probs: np.ndarray,
    labels: np.ndarray,
    candidates: Optional[np.ndarray] = None,
    metric: str = 'f1'
) -> List[float]:
    """
    Traditional grid search for threshold tuning (baseline).
    
    Args:
        probs: Predicted probabilities of shape (N, num_classes)
        labels: Ground truth labels of shape (N, num_classes)
        candidates: Threshold candidates to try
        metric: Metric to optimize per class ('f1', 'precision', 'recall')
        
    Returns:
        List of optimal thresholds for each class
    """
    if candidates is None:
        candidates = np.arange(0.05, 0.95, 0.05)
    
    thresholds = []
    for i in range(labels.shape[1]):
        best_t, best_score = 0.5, 0.0
        for t in candidates:
            preds = (probs[:, i] >= t).astype(int)
            if metric == 'f1':
                score = f1_score(labels[:, i], preds, zero_division=0)
            else:
                raise ValueError("Only 'f1' metric supported for per-class tuning")
            if score > best_score:
                best_score, best_t = score, t
        thresholds.append(float(best_t))
    
    return thresholds


def evaluate_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: List[float]
) -> dict:
    """
    Evaluate performance with given thresholds.
    
    Args:
        probs: Predicted probabilities
        labels: Ground truth labels
        thresholds: Thresholds to evaluate
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import precision_score, recall_score, confusion_matrix
    
    preds = np.zeros_like(probs)
    for i in range(len(thresholds)):
        preds[:, i] = (probs[:, i] >= thresholds[i]).astype(int)
    
    return {
        'f1_micro': f1_score(labels, preds, average='micro', zero_division=0),
        'f1_macro': f1_score(labels, preds, average='macro', zero_division=0),
        'precision_macro': precision_score(labels, preds, average='macro', zero_division=0),
        'recall_macro': recall_score(labels, preds, average='macro', zero_division=0),
        'auc_macro': roc_auc_score(labels, probs, average='macro'),
    }


def compare_threshold_methods(
    probs: np.ndarray,
    labels: np.ndarray,
    val_probs: np.ndarray,
    val_labels: np.ndarray
) -> dict:
    """
    Compare different threshold optimization methods.
    
    Args:
        probs: Test set probabilities
        labels: Test set labels
        val_probs: Validation set probabilities (for tuning)
        val_labels: Validation set labels
        
    Returns:
        Dictionary comparing methods
    """
    results = {}
    
    # Grid search baseline
    grid_thresh = tune_thresholds_grid(val_probs, val_labels)
    results['grid_search'] = {
        'thresholds': grid_thresh,
        'metrics': evaluate_thresholds(probs, labels, grid_thresh)
    }
    
    # Bayesian optimization
    bayes_thresh = tune_thresholds_bayesian(val_probs, val_labels, n_iterations=30)
    results['bayesian'] = {
        'thresholds': bayes_thresh,
        'metrics': evaluate_thresholds(probs, labels, bayes_thresh)
    }
    
    # ROC-based (Youden)
    roc_thresh = tune_thresholds_roc_based(val_probs, val_labels, criterion='youden')
    results['roc_youden'] = {
        'thresholds': roc_thresh,
        'metrics': evaluate_thresholds(probs, labels, roc_thresh)
    }
    
    return results
