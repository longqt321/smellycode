import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

def evaluate_per_label(probs: np.ndarray, labels: np.ndarray, label_names: list, thresholds: list = None):
    """Evaluate per-label metrics given probabilities and optional thresholds."""
    if thresholds is None:
        thresholds = [0.5] * len(label_names)
    print(f"\n{'Label':<20} {'Threshold':>10} {'AUC':>8} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 68)
    for i, name in enumerate(label_names):
        preds = (probs[:, i] >= thresholds[i]).astype(int)
        auc = roc_auc_score(labels[:, i], probs[:, i])
        f1 = f1_score(labels[:, i], preds, zero_division=0)
        prec = precision_score(labels[:, i], preds, zero_division=0)
        rec = recall_score(labels[:, i], preds, zero_division=0)
        print(f"  {name:<20} {thresholds[i]:>10.2f} {auc:>8.3f} {f1:>8.3f} {prec:>10.3f} {rec:>8.3f}")

def tune_thresholds(probs: np.ndarray, labels: np.ndarray, candidates=None) -> list:
    """Find best threshold per label by maximizing F1 on validation set."""
    if candidates is None:
        candidates = np.arange(0.05, 0.95, 0.05)
    thresholds = []
    for i in range(labels.shape[1]):
        best_t, best_f1 = 0.5, 0.0
        for t in candidates:
            f1 = f1_score(labels[:, i], (probs[:, i] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds.append(float(best_t))
    return thresholds
