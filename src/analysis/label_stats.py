import numpy as np

def label_distribution(y: np.ndarray, label_names: list) -> dict:
    """Returns positive count, negative count, and ratio for each label."""
    stats = {}
    for i, name in enumerate(label_names):
        pos = int(y[:, i].sum())
        neg = int((1 - y[:, i]).sum())
        stats[name] = {"positive": pos, "negative": neg, "ratio": pos / (pos + neg)}
    return stats

def print_label_distribution(stats: dict):
    print(f"{'Label':<20} {'Positive':>10} {'Negative':>10} {'Ratio':>8}")
    print("-" * 52)
    for name, s in stats.items():
        print(f"{name:<20} {s['positive']:>10} {s['negative']:>10} {s['ratio']:>8.3f}")
