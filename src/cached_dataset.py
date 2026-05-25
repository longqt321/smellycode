"""
Cached dataset that loads pre-computed embeddings from disk.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple


class CachedFusionDataset(Dataset):
    """
    Dataset that loads pre-computed GraphCodeBERT embeddings and features from disk.
    Much faster than computing embeddings on-the-fly during training.
    """
    
    def __init__(self, cache_path: str):
        """
        Load cached data from .pt file.
        
        Args:
            cache_path: Path to cached .pt file containing 'embeddings', 'features', 'labels'
        """
        data = torch.load(cache_path, map_location='cpu', weights_only=True)
        self.embeddings = data['embeddings']  # (N, 768)
        self.features = data['features']      # (N, num_features)
        self.labels = data['labels']          # (N, num_classes)
        
        assert len(self.embeddings) == len(self.features) == len(self.labels), \
            "Mismatched dimensions in cached data"
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (features, embeddings, labels)"""
        return self.features[idx], self.embeddings[idx], self.labels[idx]


def get_cached_loaders(cache_dir: str, batch_size: int = 2048, num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders from cached embeddings.
    
    Args:
        cache_dir: Directory containing train_cached.pt, val_cached.pt, test_cached.pt
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    import os
    
    train_path = os.path.join(cache_dir, 'train_cached.pt')
    val_path = os.path.join(cache_dir, 'val_cached.pt')
    test_path = os.path.join(cache_dir, 'test_cached.pt')
    
    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cached file not found: {path}")
    
    train_ds = CachedFusionDataset(train_path)
    val_ds = CachedFusionDataset(val_path)
    test_ds = CachedFusionDataset(test_path)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
