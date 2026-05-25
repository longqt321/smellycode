import os
import polars as pl
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from config import LABEL_COLUMNS, SEED, DATASET_PATH

class CodeSmellDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_and_prepare(data_path, tiny=False, tiny_size=1000):
    df = pl.read_csv(data_path)
    if tiny:
        df = df.head(tiny_size)
    drop_cols = ['File', 'Project', 'Class', 'Code']
    keep_cols = [c for c in df.columns if c not in drop_cols + LABEL_COLUMNS]
    numeric_cols = [c for c in keep_cols if df[c].dtype in [pl.Float64, pl.Int64, pl.Float32, pl.Int32]]
    X = df.select(numeric_cols).to_numpy()
    y = df.select(LABEL_COLUMNS).to_numpy()
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, temp_idx = next(msss.split(X, y))
    X_train, X_temp, y_train, y_temp = X[train_idx], X[temp_idx], y[train_idx], y[temp_idx]

    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    val_idx, test_idx = next(msss2.split(X_temp, y_temp))
    X_val, X_test = X_temp[val_idx], X_temp[test_idx]
    y_val, y_test = y_temp[val_idx], y_temp[test_idx]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    pos_weight = torch.tensor((1 - y_train).sum(axis=0) / (y_train.sum(axis=0) + 1e-6), dtype=torch.float32)
    pos_weight = torch.sqrt(pos_weight)

    return (CodeSmellDataset(X_train, y_train),
            CodeSmellDataset(X_val, y_val),
            CodeSmellDataset(X_test, y_test),
            scaler, pos_weight)

class CodeSmellFusionDataset(Dataset):
    """Dataset that returns (numeric features, code string, labels)."""
    def __init__(self, features: np.ndarray, codes: list, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.codes = codes
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.codes[idx], self.labels[idx]


def load_and_prepare_fusion(data_path, tiny=False, tiny_size=1000):
    df = pl.read_csv(data_path)
    if tiny:
        df = df.head(tiny_size)
    drop_cols = ['File', 'Project', 'Class']
    keep_cols = [c for c in df.columns if c not in drop_cols + LABEL_COLUMNS + ['Code']]
    numeric_cols = [c for c in keep_cols if df[c].dtype in [pl.Float64, pl.Int64, pl.Float32, pl.Int32]]
    X = df.select(numeric_cols).to_numpy()
    codes = df['Code'].fill_null("").to_list()
    y = df.select(LABEL_COLUMNS).to_numpy()

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, temp_idx = next(msss.split(X, y))
    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    val_idx, test_idx = next(msss2.split(X[temp_idx], y[temp_idx]))

    def split(arr, idx): return arr[idx] if isinstance(arr, np.ndarray) else [arr[i] for i in idx]

    X_train, y_train = X[train_idx], y[train_idx]
    X_temp, y_temp = X[temp_idx], y[temp_idx]
    codes_train = split(codes, train_idx)
    codes_temp  = split(codes, temp_idx)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_temp[val_idx])
    X_test  = scaler.transform(X_temp[test_idx])

    pos_weight = torch.tensor((1 - y_train).sum(axis=0) / (y_train.sum(axis=0) + 1e-6), dtype=torch.float32)
    pos_weight = torch.sqrt(pos_weight)

    return (
        CodeSmellFusionDataset(X_train, codes_train, y_train),
        CodeSmellFusionDataset(X_val,   split(codes_temp, val_idx),  y_temp[val_idx]),
        CodeSmellFusionDataset(X_test,  split(codes_temp, test_idx), y_temp[test_idx]),
        scaler, pos_weight
    )


def get_loaders(batch_size, num_workers=4, tiny=False):
    local_path = os.path.join(os.path.dirname(__file__), '..', '.env', 'dataset.csv')
    data_path = DATASET_PATH if os.path.exists(DATASET_PATH) else local_path
    ds_train, ds_val, ds_test, _, pos_weight = load_and_prepare(data_path, tiny=tiny)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, pos_weight


def get_fusion_loaders(batch_size, num_workers=4, tiny=False):
    local_path = os.path.join(os.path.dirname(__file__), '..', '.env', 'dataset.csv')
    data_path = DATASET_PATH if os.path.exists(DATASET_PATH) else local_path
    ds_train, ds_val, ds_test, _, pos_weight = load_and_prepare_fusion(data_path, tiny=tiny)
    return (
        DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=num_workers),
        DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=num_workers),
        pos_weight
    )

