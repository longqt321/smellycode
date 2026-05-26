"""Configuration module for code smell detection training."""
import os
from dataclasses import dataclass, field
from typing import List, Optional
import torch.nn as nn


@dataclass
class TrainingConfig:
    """Training hyperparameters and settings."""
    seed: int = 1206
    test_size: float = 0.20
    val_size: float = 0.20
    train_batch_size: int = 1024
    val_batch_size: int = 1024
    num_workers: int = 4
    epochs: int = 50
    learning_rate: float = 1e-3


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    label_columns: List[str] = field(default_factory=lambda: [
        "Long method", "God class", "Feature envy", "Data class"
    ])
    volume_mount_path: str = "/mnt/data"
    
    @property
    def dataset_path(self) -> str:
        return os.path.join(self.volume_mount_path, "dataset.csv")
    
    @property
    def artifacts_dir(self) -> str:
        return os.path.join(self.volume_mount_path, "artifacts")


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    use_relu: bool = True
    
    @property
    def activation(self) -> type:
        return nn.ReLU if self.use_relu else nn.Mish


# Global default instances
training_config = TrainingConfig()
dataset_config = DatasetConfig()
model_config = ModelConfig()

# Backward compatibility aliases
SEED = training_config.seed
TEST_SIZE = training_config.test_size
VAL_SIZE = training_config.val_size
TRAIN_BATCH_SIZE = training_config.train_batch_size
VAL_BATCH_SIZE = training_config.val_batch_size
NUM_WORKERS = training_config.num_workers
LABEL_COLUMNS = dataset_config.label_columns
VOLUME_MOUNT_PATH = dataset_config.volume_mount_path
DATASET_PATH = dataset_config.dataset_path
ARTIFACTS_DIR = dataset_config.artifacts_dir
USE_RELU = model_config.use_relu
ACTIVATION = model_config.activation