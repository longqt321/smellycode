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
│   ├── config.py          # Training and model configurations
│   ├── data_utils.py      # Dataset and DataLoader utilities
│   ├── layers/
│   │   └── layers.py      # Custom neural network layers (CrossLayer, Bottleneck)
│   └── networks/
│       └── dcn.py         # DCNv2 model architecture
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies
└── README.md              # This file
```

## Usage

### Configuration

Edit `src/config.py` to customize:

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

### Cloud Deployment

This project is configured for deployment on Modal cloud infrastructure. The dataset and artifacts are mounted at `/mnt/data`:

- `DATASET_PATH`: Path to the training dataset (`dataset.csv`)
- `ARTIFACTS_DIR`: Directory for saving model checkpoints and logs

## License

MIT License