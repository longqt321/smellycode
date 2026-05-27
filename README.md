# smellycode

A deep learning project for automated code smell detection using Deep & Cross Networks (DCNv2) and GraphCodeBERT fusion models.

## Description

This project implements a neural network architecture designed to detect common code smells in software projects:

- **Long method**: Methods that are too long and should be refactored
- **God class**: Classes that do too much and violate single responsibility principle
- **Feature envy**: Methods that seem more interested in other classes than their own
- **Data class**: Classes that only contain data fields with no behavior

The model uses two architectures:

### DCNv2 (Metrics-based)
- Cross layers for explicit feature interaction learning
- Deep layers with bottleneck structures for implicit feature learning
- Multi-label classification for detecting multiple code smells simultaneously

### GatedFusionModel (Metrics + Semantic)
- Combines traditional code metrics with GraphCodeBERT embeddings
- Gated fusion mechanism to dynamically weight numeric vs semantic features
- Pre-computed BERT embeddings for faster training
- Advanced visualization of gate behavior

## Key Features

### ✨ Advanced Threshold Optimization
Three methods for optimizing decision thresholds:
- **Grid Search**: Traditional exhaustive search over threshold candidates
- **Bayesian Optimization**: Gradient-free optimization with local refinement (Nelder-Mead)
- **ROC-based**: Youden's J statistic or closest-to-top-left criteria

### ⚖️ Class Imbalance Handling
Multiple strategies without resampling (preserves true distribution):
- **Focal Loss**: Focuses on hard examples
- **Asymmetric Loss (ASL)**: Separate focusing for positive/negative samples
- **Class-Balanced Loss**: Effective number of samples framework
- **CB-Focal Loss**: Combines CB weighting with focal loss

### 🎯 Gate Visualization (Fusion Models Only)
Comprehensive analysis of the gated fusion mechanism:
- Distribution analysis of gate values
- Per-dimension statistics
- Decision breakdown (numeric vs text preference)
- Gate behavior by class
- Human-readable interpretation reports

### 📦 ONNX Export
Export trained models for production deployment:
- DCNv2 model export (single input)
- Fusion model export (dual input: features + BERT embeddings)
- Automatic verification with PyTorch comparison
- Benchmarking utilities for performance comparison

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
│   │   ├── model_summary.py
│   │   ├── visualization.py
│   │   ├── threshold_optimization.py  # Advanced threshold tuning (Bayesian, ROC)
│   │   └── gate_visualization.py      # Gate mechanism analysis
│   ├── layers/
│   │   └── layers.py      # Custom neural network layers (CrossLayer, Bottleneck)
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── focal_loss.py              # Focal Loss & Asymmetric Loss
│   │   └── class_balanced_loss.py     # Class-Balanced Loss (Effective Number)
│   ├── networks/
│   │   ├── dcn.py         # DCNv2 model architecture
│   │   └── fusion.py      # Fusion network architecture (GatedFusionModel)
│   └── utils/
│       ├── __init__.py
│       └── onnx_export.py # ONNX export utilities
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
  # Model architecture
  --cross_type {standard,gated}   # Cross layer type
  --deep_type {bottleneck,moe}    # Deep layer type
  --use_semantic                  # Enable fusion with GraphCodeBERT
  --fusion_type {gated,late_mlp}  # Fusion model type (default: gated)
  --embed_dim INT                 # Embedding dimension (default: 128)
  
  # Loss functions for imbalanced data
  --loss {bce,focal,asl,cb_focal,cb}  # Loss function
  --focal_gamma FLOAT             # Focal loss gamma (default: 2.0)
  --asl_gamma_neg FLOAT           # ASL negative gamma (default: 4.0)
  --asl_gamma_pos FLOAT           # ASL positive gamma (default: 1.0)
  --cb_beta FLOAT                 # Class-balanced beta (default: 0.9999)
  
  # Threshold optimization
  --threshold_method {grid,bayesian,roc}  # Method for threshold tuning
                                          # grid: Traditional grid search
                                          # bayesian: Nelder-Mead optimization
                                          # roc: Youden's J statistic
  
  # Analysis & Export
  --gate_analysis                 # Generate gate visualization report (fusion only)
  --export_onnx                   # Export trained model to ONNX format
  
  # Training hyperparameters
  --epochs INT                    # Number of epochs
  --batch_size INT                # Batch size
  --lr FLOAT                      # Learning rate
  --seed INT+                     # Random seed(s) for multi-run experiments
```

### Advanced Usage Examples

#### Train with Class-Balanced Focal Loss and Bayesian Threshold Optimization
```bash
python train.py --use_semantic --loss cb_focal --threshold_method bayesian --gate_analysis
```

#### Train with Asymmetric Loss and Export to ONNX
```bash
python train.py --use_semantic --loss asl --export_onnx --epochs 100
```

#### Multi-seed Experiment with Gate Analysis
```bash
python train.py --use_semantic --loss cb_focal --gate_analysis --seed 1206 42 123
```

### Analysis

Analyze model performance and dataset statistics:

```bash
python analyze.py
```

**Gate Visualization Report** (Fusion models only):
After training with `--gate_analysis`, you'll get:
- `artifacts/gate_analysis_seed{N}/gate_distribution.png`: Gate value distributions
- `artifacts/gate_analysis_seed{N}/gate_by_class.png`: Gate behavior per class
- Console output with human-readable interpretation

**ONNX Export**:
After training with `--export_onnx`, you'll get:
- `artifacts/onnx_export_seed{N}/fusion_model.onnx` or `dcn_model.onnx`
- Automatic verification comparing PyTorch vs ONNX outputs
- Performance metrics logged to W&B

### Cloud Deployment

This project is configured for deployment on Modal cloud infrastructure. The dataset and artifacts are mounted at `/mnt/data`:

- `DATASET_PATH`: Path to the training dataset (`dataset.csv`)
- `ARTIFACTS_DIR`: Directory for saving model checkpoints and logs

### Implementation Details

#### Threshold Optimization Methods

The project implements three threshold optimization strategies:

1. **Grid Search** (`--threshold_method grid`): Traditional exhaustive search over threshold candidates (0.05 to 0.95)

2. **Bayesian Optimization** (`--threshold_method bayesian`): 
   - Uses Nelder-Mead simplex method for gradient-free optimization
   - Starts from best grid-search thresholds as initial point
   - Optimizes F1-macro, F1-micro, or balanced accuracy

3. **ROC-based** (`--threshold_method roc`):
   - Youden's J statistic: maximizes (sensitivity + specificity - 1)
   - Closest-to-top-left: minimizes distance to perfect classifier

#### Class Imbalance Handling

All methods work by reweighting the loss function, NOT by resampling data:

- **Focal Loss**: Down-weights easy examples, focuses on hard ones
- **Asymmetric Loss**: Different focusing parameters for positive/negative samples
- **Class-Balanced Loss**: Uses effective number of samples framework
  - Formula: EN(n) = (1 - β^n) / (1 - β)
  - Preserves true data distribution while adjusting loss weights

#### Gate Mechanism (Fusion Models)

The gated fusion mechanism learns to dynamically weight numeric vs semantic features:
- Gate value close to 1: relies more on numeric metrics
- Gate value close to 0: relies more on text embeddings
- Gate value around 0.5: balanced blending

Use `--gate_analysis` to generate comprehensive reports including:
- Distribution histograms
- Per-dimension statistics
- Decision breakdown analysis
- Interpretation in natural language

## License

MIT License