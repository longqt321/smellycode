"""
Pre-compute GraphCodeBERT embeddings and cache to disk.
This script extracts embeddings once and saves them, so training doesn't need to re-compute.
"""
import os
import torch
import polars as pl
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from config import DATASET_PATH, SEED
from src.data import LABEL_COLUMNS
from sklearn.preprocessing import StandardScaler
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import numpy as np


class CodeDataset(Dataset):
    """Simple dataset for code strings."""
    def __init__(self, codes: list):
        self.codes = codes
    
    def __len__(self):
        return len(self.codes)
    
    def __getitem__(self, idx):
        return self.codes[idx]


def split_data(df, seed=SEED):
    """Split data into train/val/test with multilabel stratification."""
    drop_cols = ['File', 'Project', 'Class']
    keep_cols = [c for c in df.columns if c not in drop_cols + LABEL_COLUMNS + ['Code']]
    numeric_cols = [c for c in keep_cols if df[c].dtype in [pl.Float64, pl.Int64, pl.Float32, pl.Int32]]
    X = df.select(numeric_cols).to_numpy()
    y = df.select(LABEL_COLUMNS).to_numpy()
    codes = df['Code'].fill_null("").to_list()
    
    # Train/test split
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, temp_idx = next(msss.split(X, y))
    
    # Val/test split (50-50 of remaining)
    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_idx, test_idx = next(msss2.split(X[temp_idx], y[temp_idx]))
    
    def split_arr(arr, idx):
        return arr[idx] if isinstance(arr, np.ndarray) else [arr[i] for i in idx]
    
    return {
        'train': {'codes': split_arr(codes, train_idx), 'features': X[train_idx], 'labels': y[train_idx]},
        'val': {'codes': split_arr(codes, temp_idx[val_idx]), 'features': X[temp_idx][val_idx], 'labels': y[temp_idx][val_idx]},
        'test': {'codes': split_arr(codes, temp_idx[test_idx]), 'features': X[temp_idx][test_idx], 'labels': y[temp_idx][test_idx]}
    }


@torch.no_grad()
def extract_embeddings(codes: list, tokenizer, model, device, batch_size=64, max_length=512):
    """
    Extract frozen GraphCodeBERT [CLS] embeddings with FP16 autocast.
    Returns CPU tensor of shape (N, 768).
    """
    model.eval()
    all_embeds = []
    dataset = CodeDataset(codes)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Extracting embeddings for {len(codes)} samples...")
    for i, batch_codes in enumerate(loader):
        if isinstance(batch_codes, str):
            batch_codes = [batch_codes]
        
        enc = tokenizer(
            batch_codes, 
            padding=True, 
            truncation=True,
            max_length=max_length, 
            return_tensors='pt'
        ).to(device)
        
        # FP16 autocast for faster inference
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            out = model(**enc)
            cls_embed = out.last_hidden_state[:, 0].float().cpu()
        
        all_embeds.append(cls_embed)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {min((i+1)*batch_size, len(codes))}/{len(codes)} samples")
    
    return torch.cat(all_embeds, dim=0)


def save_embeddings(embeddings: torch.Tensor, features: np.ndarray, labels: np.ndarray, 
                   scaler: StandardScaler, save_path: str):
    """Save pre-computed embeddings and features to a single .pt file."""
    # Scale features
    scaled_features = scaler.transform(features)
    
    data = {
        'embeddings': embeddings,  # (N, 768)
        'features': torch.tensor(scaled_features, dtype=torch.float32),  # (N, num_features)
        'labels': torch.tensor(labels, dtype=torch.float32),  # (N, 4)
    }
    torch.save(data, save_path)
    print(f"Saved to {save_path}")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Features shape: {data['features'].shape}")
    print(f"  Labels shape: {data['labels'].shape}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    local_path = os.path.join(os.path.dirname(__file__), '.env', 'dataset.csv')
    data_path = DATASET_PATH if os.path.exists(DATASET_PATH) else local_path
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    print(f"Loading data from {data_path}...")
    df = pl.read_csv(data_path)
    print(f"Total samples: {len(df)}")
    
    # Split data
    splits = split_data(df)
    print(f"Train: {len(splits['train']['codes'])}, Val: {len(splits['val']['codes'])}, Test: {len(splits['test']['codes'])}")
    
    # Load model and tokenizer
    model_name = "microsoft/graphcodebert-base"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    # Freeze model completely
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    
    # Create cache directory
    cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Extract and save for each split
    scaler = None
    for split_name in ['train', 'val', 'test']:
        print(f"\n{'='*50}")
        print(f"Processing {split_name.upper()} split...")
        print('='*50)
        
        codes = splits[split_name]['codes']
        features = splits[split_name]['features']
        labels = splits[split_name]['labels']
        
        # Fit scaler on train, transform for all
        if split_name == 'train':
            scaler = StandardScaler().fit(features)
        
        # Extract embeddings with FP16
        embeddings = extract_embeddings(
            codes, tokenizer, model, device, 
            batch_size=args.batch_size, max_length=args.max_length
        )
        
        # Save to disk
        save_path = os.path.join(cache_dir, f'{split_name}_cached.pt')
        save_embeddings(embeddings, features, labels, scaler, save_path)
    
    print(f"\n{'='*50}")
    print("✅ Pre-computation complete!")
    print(f"Cached files saved to: {cache_dir}")
    print('='*50)


if __name__ == '__main__':
    main()
