import numpy as np

def feature_stats(X: np.ndarray) -> dict:
    """Returns basic stats for the feature matrix."""
    return {
        "shape": X.shape,
        "mean": float(X.mean()),
        "std": float(X.std()),
        "has_nan": bool(np.isnan(X).any()),
        "has_inf": bool(np.isinf(X).any()),
    }

def print_feature_stats(stats: dict):
    for k, v in stats.items():
        print(f"  {k}: {v}")
