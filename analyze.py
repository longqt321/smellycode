"""
Analyzes the training split for label distribution and feature quality.
Runs on Modal via: modal run analyze.py
"""
import sys
sys.path.insert(0, '/app/src')

from config import LABEL_COLUMNS, DATASET_PATH
from data import load_and_prepare
from analysis.label_stats import label_distribution, print_label_distribution
from analysis.feature_stats import feature_stats, print_feature_stats

def main():
    ds_train, _, _, _, pos_weight = load_and_prepare(DATASET_PATH)
    X_train = ds_train.features.numpy()
    y_train = ds_train.labels.numpy()

    print("=== Label Distribution (Train) ===")
    print_label_distribution(label_distribution(y_train, LABEL_COLUMNS))

    print("\n=== pos_weight ===")
    for name, w in zip(LABEL_COLUMNS, pos_weight.tolist()):
        print(f"  {name:<20} {w:.2f}")

    print("\n=== Feature Stats (Train) ===")
    print_feature_stats(feature_stats(X_train))

if __name__ == '__main__':
    main()
