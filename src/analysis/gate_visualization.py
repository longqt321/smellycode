"""Gate mechanism visualization for GatedFusionModel."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List
import torch


def plot_gate_distribution(
    gate_values: np.ndarray,
    label_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot distribution of gate values across dimensions.
    
    Args:
        gate_values: Gate activations of shape (N, embed_dim) or (embed_dim,)
        label_names: Optional names for each dimension
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure
    """
    if gate_values.ndim == 1:
        gate_values = gate_values.reshape(1, -1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Histogram of all gate values
    ax = axes[0, 0]
    ax.hist(gate_values.flatten(), bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0.5, color='r', linestyle='--', label='Neutral (0.5)')
    ax.set_xlabel('Gate Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Gate Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Per-dimension statistics
    ax = axes[0, 1]
    dim_means = gate_values.mean(axis=0)
    dim_stds = gate_values.std(axis=0)
    x = np.arange(len(dim_means))
    ax.bar(x, dim_means, yerr=dim_stds, capsize=3, alpha=0.7)
    ax.axhline(y=0.5, color='r', linestyle='--', label='Neutral (0.5)')
    ax.set_xlabel('Embedding Dimension')
    ax.set_ylabel('Mean Gate Value')
    ax.set_title('Per-Dimension Gate Statistics')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Heatmap of gate values (if enough samples)
    ax = axes[1, 0]
    if gate_values.shape[0] >= 10:
        # Sample if too many
        sample_size = min(100, gate_values.shape[0])
        indices = np.random.choice(gate_values.shape[0], sample_size, replace=False)
        sampled = gate_values[indices]
        im = ax.imshow(sampled.T, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label='Gate Value')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Embedding Dimension')
        ax.set_title(f'Gate Activation Heatmap (n={sample_size})')
    else:
        ax.text(0.5, 0.5, 'Not enough samples\nfor heatmap', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    # 4. Decision analysis
    ax = axes[1, 1]
    near_zero = (gate_values < 0.1).mean(axis=0)
    near_one = (gate_values > 0.9).mean(axis=0)
    neutral = ((gate_values >= 0.4) & (gate_values <= 0.6)).mean(axis=0)
    
    x = np.arange(len(dim_means))
    width = 0.25
    ax.bar(x - width, near_zero, width, label='Near 0 (Text)', color='blue', alpha=0.7)
    ax.bar(x, neutral, width, label='Neutral', color='gray', alpha=0.7)
    ax.bar(x + width, near_one, width, label='Near 1 (Numeric)', color='red', alpha=0.7)
    ax.set_xlabel('Embedding Dimension')
    ax.set_ylabel('Proportion')
    ax.set_title('Gate Decision Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_gate_by_class(
    gate_values: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Analyze gate behavior per predicted/true class.
    
    Args:
        gate_values: Gate activations of shape (N, embed_dim)
        predictions: Predicted probabilities or binary predictions
        labels: Ground truth binary labels
        class_names: Names of the classes
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure
    """
    num_classes = len(class_names)
    embed_dim = gate_values.shape[1]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Gate statistics by true positive/negative
    ax = axes[0]
    tp_gates = []
    tn_gates = []
    
    for i in range(num_classes):
        mask_tp = labels[:, i] == 1
        mask_tn = labels[:, i] == 0
        
        if mask_tp.sum() > 0:
            tp_gates.append(gate_values[mask_tp].mean())
        if mask_tn.sum() > 0:
            tn_gates.append(gate_values[mask_tn].mean())
    
    x = np.arange(num_classes)
    width = 0.35
    ax.bar(x - width/2, tp_gates, width, label='True Positive', color='green', alpha=0.7)
    ax.bar(x + width/2, tn_gates, width, label='True Negative', color='red', alpha=0.7)
    ax.axhline(y=0.5, color='gray', linestyle='--', label='Neutral')
    ax.set_xlabel('Class')
    ax.set_ylabel('Mean Gate Value')
    ax.set_title('Gate Behavior by True Label')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Gate vs prediction confidence
    ax = axes[1]
    
    # Compute average gate value per sample
    avg_gates = gate_values.mean(axis=1)
    
    # Compute max prediction probability per sample
    max_probs = predictions.max(axis=1)
    
    scatter = ax.scatter(max_probs, avg_gates, alpha=0.3, s=10, c='blue')
    ax.axhline(y=0.5, color='r', linestyle='--', label='Neutral Gate')
    ax.set_xlabel('Max Prediction Probability')
    ax.set_ylabel('Average Gate Value')
    ax.set_title('Gate Value vs Prediction Confidence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_gate_report(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_samples: int = 1000,
    save_dir: Optional[str] = None
) -> Dict:
    """
    Generate comprehensive gate analysis report.
    
    Args:
        model: Trained GatedFusionModel
        dataloader: Data loader for evaluation
        device: Device to run inference on
        num_samples: Number of samples to analyze
        save_dir: Optional directory to save figures
        
    Returns:
        Dictionary with gate statistics
    """
    from src.networks.fusion import GatedFusionModel
    
    if not isinstance(model, GatedFusionModel):
        raise ValueError("Model must be a GatedFusionModel")
    
    model.eval()
    all_gates = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:  # Fusion dataset
                features, bert_embed, labels = batch
                features = features.to(device)
                bert_embed = bert_embed.to(device)
            else:
                continue  # Skip non-fusion datasets
            
            _ = model(features, bert_embed)
            gate = model.last_gate.cpu().numpy()
            preds = torch.sigmoid(model(features, bert_embed)).cpu().numpy()
            
            all_gates.append(gate)
            all_preds.append(preds)
            all_labels.append(labels.numpy())
            
            if sum(len(g) for g in all_gates) >= num_samples:
                break
    
    all_gates = np.concatenate(all_gates, axis=0)[:num_samples]
    all_preds = np.concatenate(all_preds, axis=0)[:num_samples]
    all_labels = np.concatenate(all_labels, axis=0)[:num_samples]
    
    # Compute statistics
    stats = {
        'mean': float(all_gates.mean()),
        'std': float(all_gates.std()),
        'min': float(all_gates.min()),
        'max': float(all_gates.max()),
        'near_zero': float((all_gates < 0.1).mean()),
        'near_one': float((all_gates > 0.9).mean()),
        'neutral': float(((all_gates >= 0.4) & (all_gates <= 0.6)).mean()),
        'per_dim_mean': all_gates.mean(axis=0).tolist(),
        'per_dim_std': all_gates.std(axis=0).tolist(),
    }
    
    # Generate visualizations
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        fig1 = plot_gate_distribution(all_gates, save_path=os.path.join(save_dir, 'gate_distribution.png'))
        plt.close(fig1)
        
        fig2 = plot_gate_by_class(all_gates, all_preds, all_labels, 
                                   ['Long method', 'God class', 'Feature envy', 'Data class'],
                                   save_path=os.path.join(save_dir, 'gate_by_class.png'))
        plt.close(fig2)
    
    return stats


def interpret_gate_behavior(stats: Dict) -> str:
    """
    Generate human-readable interpretation of gate behavior.
    
    Args:
        stats: Dictionary from create_gate_report
        
    Returns:
        Interpretation string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("GATE MECHANISM INTERPRETATION")
    lines.append("=" * 60)
    
    mean = stats['mean']
    if mean > 0.6:
        lines.append(f"→ Model relies MORE on NUMERIC features (mean={mean:.3f})")
    elif mean < 0.4:
        lines.append(f"→ Model relies MORE on TEXT features (mean={mean:.3f})")
    else:
        lines.append(f"→ Model BALANCES both feature types (mean={mean:.3f})")
    
    lines.append("")
    lines.append(f"Gate variance: {stats['std']:.3f}")
    if stats['std'] > 0.3:
        lines.append("  → High variance: Gate makes strong decisions per sample")
    else:
        lines.append("  → Low variance: Gate tends toward neutral blending")
    
    lines.append("")
    lines.append("Decision breakdown:")
    lines.append(f"  - Strongly numeric (>0.9): {stats['near_one']*100:.1f}%")
    lines.append(f"  - Strongly text (<0.1):    {stats['near_zero']*100:.1f}%")
    lines.append(f"  - Neutral (0.4-0.6):       {stats['neutral']*100:.1f}%")
    
    if stats['neutral'] > 0.5:
        lines.append("\n→ Gate mostly performs soft averaging rather than hard selection")
    elif stats['near_one'] + stats['near_zero'] > 0.5:
        lines.append("\n→ Gate acts as a strong selector between modalities")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)
