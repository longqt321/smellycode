"""Data loading utilities for code smell detection."""
import os
from typing import List, Optional, Tuple
import polars as pl
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from config import LABEL_COLUMNS, SEED, DATASET_PATH


class CodeSmellDataset(Dataset):
    """Dataset for tabular code smell features.
    
    Args:
        features: Feature matrix of shape (N, num_features)
        labels: Label matrix of shape (N, num_classes)
    """
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def _split_data(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = SEED,
    test_size: float = 0.2,
    val_split: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train/val/test with multilabel stratification.
    
    Args:
        X: Feature matrix
        y: Label matrix
        seed: Random seed for reproducibility
        test_size: Proportion of data for testing
        val_split: Proportion of remaining data for validation
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Train/test split
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, temp_idx = next(msss.split(X, y))
    X_train, X_temp = X[train_idx], X[temp_idx]
    y_train, y_temp = y[train_idx], y[temp_idx]
    
    # Val/test split (50-50 of remaining)
    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    val_idx, test_idx = next(msss2.split(X_temp, y_temp))
    X_val, X_test = X_temp[val_idx], X_temp[test_idx]
    y_val, y_test = y_temp[val_idx], y_temp[test_idx]
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def _compute_pos_weight(y_train: np.ndarray) -> torch.Tensor:
    """Compute class weights for imbalanced datasets.
    
    Args:
        y_train: Training labels
        
    Returns:
        Tensor of per-class weights
    """
    pos_weight = torch.tensor(
        (1 - y_train).sum(axis=0) / (y_train.sum(axis=0) + 1e-6),
        dtype=torch.float32
    )
    return torch.sqrt(pos_weight)


def load_and_prepare(
    data_path: str,
    tiny: bool = False,
    tiny_size: int = 1000
) -> Tuple[CodeSmellDataset, CodeSmellDataset, CodeSmellDataset, StandardScaler, torch.Tensor]:
    """Load and prepare the code smell dataset.
    
    Args:
        data_path: Path to CSV file
        tiny: If True, use only a subset of data for debugging
        tiny_size: Number of samples to use if tiny=True
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, scaler, pos_weight)
    """
    df = pl.read_csv(data_path)
    
    if tiny:
        df = df.head(tiny_size)
    
    # Select columns
    drop_cols = ['File', 'Project', 'Class', 'Code']
    keep_cols = [c for c in df.columns if c not in drop_cols + LABEL_COLUMNS]
    numeric_cols = [
        c for c in keep_cols 
        if df[c].dtype in [pl.Float64, pl.Int64, pl.Float32, pl.Int32]
    ]
    
    X = df.select(numeric_cols).to_numpy()
    y = df.select(LABEL_COLUMNS).to_numpy()
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = _split_data(X, y)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Compute class weights
    pos_weight = _compute_pos_weight(y_train)
    
    return (
        CodeSmellDataset(X_train, y_train),
        CodeSmellDataset(X_val, y_val),
        CodeSmellDataset(X_test, y_test),
        scaler,
        pos_weight
    )


class CodeSmellFusionDataset(Dataset):
    """Dataset for multimodal fusion with numeric features and code strings.
    
    Args:
        features: Feature matrix of shape (N, num_features)
        codes: List of code strings
        labels: Label matrix of shape (N, num_classes)
    """
    
    def __init__(self, features: np.ndarray, codes: List[str], labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.codes = codes
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, torch.Tensor]:
        return self.features[idx], self.codes[idx], self.labels[idx]


def load_and_prepare_fusion(
    data_path: str,
    tiny: bool = False,
    tiny_size: int = 1000
) -> Tuple[CodeSmellFusionDataset, CodeSmellFusionDataset, CodeSmellFusionDataset, StandardScaler, torch.Tensor]:
    """Load and prepare the fusion dataset with code strings.
    
    Args:
        data_path: Path to CSV file
        tiny: If True, use only a subset of data for debugging
        tiny_size: Number of samples to use if tiny=True
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, scaler, pos_weight)
    """
    df = pl.read_csv(data_path)
    
    if tiny:
        df = df.head(tiny_size)
    
    # Select columns
    drop_cols = ['File', 'Project', 'Class']
    keep_cols = [c for c in df.columns if c not in drop_cols + LABEL_COLUMNS + ['Code']]
    numeric_cols = [
        c for c in keep_cols 
        if df[c].dtype in [pl.Float64, pl.Int64, pl.Float32, pl.Int32]
    ]
    
    X = df.select(numeric_cols).to_numpy()
    codes = df['Code'].fill_null("").to_list()
    y = df.select(LABEL_COLUMNS).to_numpy()
    
    # Split data
    train_idx, temp_idx = next(
        MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(X, y)
    )
    val_idx, test_idx = next(
        MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED).split(X[temp_idx], y[temp_idx])
    )
    
    def split_data(arr, idx):
        return arr[idx] if isinstance(arr, np.ndarray) else [arr[i] for i in idx]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_temp, y_temp = X[temp_idx], y[temp_idx]
    codes_train = split_data(codes, train_idx)
    codes_temp = split_data(codes, temp_idx)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_temp[val_idx])
    X_test = scaler.transform(X_temp[test_idx])
    
    # Compute class weights
    pos_weight = _compute_pos_weight(y_train)
    
    return (
        CodeSmellFusionDataset(X_train, codes_train, y_train),
        CodeSmellFusionDataset(X_val, split_data(codes_temp, val_idx), y_temp[val_idx]),
        CodeSmellFusionDataset(X_test, split_data(codes_temp, test_idx), y_temp[test_idx]),
        scaler,
        pos_weight
    )


def get_loaders(
    batch_size: int,
    num_workers: int = 4,
    tiny: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """Get data loaders for standard training.
    
    Args:
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        tiny: If True, use only a subset of data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, pos_weight)
    """
    local_path = os.path.join(os.path.dirname(__file__), '..', '.env', 'dataset.csv')
    data_path = DATASET_PATH if os.path.exists(DATASET_PATH) else local_path
    
    ds_train, ds_val, ds_test, _, pos_weight = load_and_prepare(data_path, tiny=tiny)
    
    return (
        DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        pos_weight
    )


def get_fusion_loaders(
    batch_size: int,
    num_workers: int = 4,
    tiny: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """Get data loaders for fusion model training.
    
    Args:
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        tiny: If True, use only a subset of data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, pos_weight)
    """
    local_path = os.path.join(os.path.dirname(__file__), '..', '.env', 'dataset.csv')
    data_path = DATASET_PATH if os.path.exists(DATASET_PATH) else local_path
    
    ds_train, ds_val, ds_test, _, pos_weight = load_and_prepare_fusion(data_path, tiny=tiny)
    
    return (
        DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        pos_weight
    )

