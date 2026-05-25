# smellycode

A deep learning project for automated code smell detection using Deep & Cross Networks (DCNv2).

## Description

This project implements a neural network architecture designed to detect common code smells in software projects:

- **Long method**: Methods that are too long and should be refactored
- **God class**: Classes that do too much and violate single responsibility principle
- **Feature envy**: Methods that seem more interested in other classes than their own
- **Data class**: Classes that only contain data fields with no behavior

The model uses a Deep & Cross Network (DCNv2) architecture that combines:
- Cross layers for explicit feature interaction learning
- Deep layers with bottleneck structures for implicit feature learning
- Multi-label classification for detecting multiple code smells simultaneously

## Installation

This project uses `uv` for dependency management and requires Python 3.13+.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

### Dependencies

- `modal>=1.4.3` - For cloud deployment and distributed computing
- `transformers>=5.9.0` - For transformer-based models
- `torch` - PyTorch for deep learning
- `polars` - Fast DataFrame library
- `numpy` - Numerical computing

## Project Structure

```
smellycode/
├── src/
│   ├── __init__.py
│   ├── data.py            # Dataset and DataLoader utilities
│   ├── data_utils.py      # Additional data utilities
│   ├── cached_dataset.py  # Cached dataset for pre-computed embeddings
│   ├── analysis/          # Evaluation and analysis tools
│   │   ├── evaluation.py
│   │   ├── feature_stats.py
│   │   ├── label_stats.py
│   │   └── model_summary.py
│   ├── layers/
│   │   └── layers.py      # Custom neural network layers (CrossLayer, Bottleneck)
│   ├── losses/
│   │   └── focal_loss.py  # Focal loss implementation
│   └── networks/
│       ├── dcn.py         # DCNv2 model architecture
│       └── fusion.py      # Fusion network architecture (DCNv2 + GraphCodeBERT)
├── config.py              # Training and model configurations
├── train.py               # Main training script
├── precompute_embeddings.py  # Pre-compute GraphCodeBERT embeddings
├── modal_train.py         # Modal cloud training script
├── modal_precompute.py    # Modal cloud pre-computation script
├── analyze.py             # Analysis script
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies
└── README.md              # This file
```

## Usage

### Configuration

Edit `config.py` to customize:

```python
# Training config
SEED = 1206
TEST_SIZE = 0.20
VAL_SIZE = 0.20
TRAIN_BATCH_SIZE = 1024
VAL_BATCH_SIZE = 1024
NUM_WORKERS = 4

# Model config
USE_RELU = True  # Set to False to use Mish activation
```

### Model Architecture

The DCNv2 model consists of:
1. **Projection layer**: Projects input to a lower dimensional space
2. **Cross layers**: Learn explicit feature interactions
3. **Deep layers**: Bottleneck-style residual blocks for deep feature learning
4. **Embedding layer**: Combines cross and deep features
5. **Classifier**: Multi-label output layer

```python
from src.networks.dcn import DCNv2

model = DCNv2(
    input_dim=your_input_dimension,
    projection_dim=128,
    cross_layers=4,
    deep_layers=(256, 128),
    embed_dim=128,
    num_classes=4  # Number of code smell types
)
```

### Training

#### Step 1: Pre-compute Embeddings (Required for Fusion Model)

Before training the fusion model, you must pre-compute GraphCodeBERT embeddings and cache them to disk. This is a one-time operation that significantly speeds up training.

**Local execution:**
```bash
python precompute_embeddings.py --max_length 512 --batch_size 64
```

**Cloud execution (Modal):**
```bash
modal run modal_precompute.py
```

This will:
- Load GraphCodeBERT-base and freeze it completely
- Extract [CLS] embeddings using FP16 autocast for faster inference
- Cache embeddings to `cache/train_cached.pt`, `cache/val_cached.pt`, `cache/test_cached.pt`

#### Step 2: Train the Model

**Train DCNv2 only (metrics-based):**
```bash
python train.py
```

**Train Fusion model (metrics + semantic embeddings):**
```bash
python train.py --use_semantic --embed_dim 128 --max_length 512
```

**Cloud training with Modal:**
```bash
# DCNv2 only
modal run modal_train.py

# Fusion model
modal run modal_train.py --use-semantic --embed-dim 128
```

**Training options:**
```bash
python train.py --help
  --cross_type {standard,gated}   # Cross layer type
  --deep_type {bottleneck,moe}    # Deep layer type
  --use_semantic                  # Enable fusion with GraphCodeBERT
  --embed_dim INT                 # Embedding dimension (default: 128)
  --loss {bce,focal,asl}          # Loss function
  --epochs INT                    # Number of epochs
  --batch_size INT                # Batch size
  --lr FLOAT                      # Learning rate
```

### Analysis

Analyze model performance and dataset statistics:

```bash
python analyze.py
```

### Cloud Deployment

This project is configured for deployment on Modal cloud infrastructure. The dataset and artifacts are mounted at `/mnt/data`:

- `DATASET_PATH`: Path to the training dataset (`dataset.csv`)
- `ARTIFACTS_DIR`: Directory for saving model checkpoints and logs

## License

MIT License