"""Class-Balanced Loss for handling imbalanced datasets without resampling.

This implements the Class-Balanced Loss based on effective number of samples,
which reweights the loss to account for class imbalance without modifying
the training data distribution. This approach:
- Does NOT change the data distribution (unlike oversampling/undersampling)
- Provides theoretical grounding via effective number concept
- Works well with multi-label classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def compute_effective_number_weights(
    labels: torch.Tensor,
    beta: float = 0.9999,
    epsilon: float = 1e-4
) -> torch.Tensor:
    """
    Compute class-balanced weights using effective number of samples.
    
    The effective number framework assigns higher weights to rare classes
    based on the formula: EN(n) = (1 - beta^n) / (1 - beta)
    where n is the number of samples for each class.
    
    Args:
        labels: Binary labels tensor of shape (N, num_classes)
        beta: Hyperparameter controlling the weighting strength.
              Higher beta = more aggressive rebalancing.
              Typical values: 0.99 to 0.9999
        epsilon: Small constant for numerical stability
        
    Returns:
        Per-class weights tensor of shape (num_classes,)
    """
    # Count positive samples per class
    pos_counts = labels.sum(dim=0)  # Shape: (num_classes,)
    total_samples = labels.shape[0]
    neg_counts = total_samples - pos_counts
    
    # Compute effective numbers
    # For positive class
    en_pos = (1 - beta ** pos_counts.clamp(min=epsilon)) / (1 - beta)
    # For negative class  
    en_neg = (1 - beta ** neg_counts.clamp(min=epsilon)) / (1 - beta)
    
    # Normalize to get weights (inverse of effective number)
    # Higher weight for classes with fewer samples
    max_en_pos = en_pos.max()
    max_en_neg = en_neg.max()
    
    weights_pos = max_en_pos / en_pos.clamp(min=epsilon)
    weights_neg = max_en_neg / en_neg.clamp(min=epsilon)
    
    return weights_pos, weights_neg


class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss combining CB weighting with Focal Loss.
    
    This loss combines:
    1. Class-Balanced weighting based on effective number of samples
    2. Focal Loss focusing mechanism for hard examples
    
    Args:
        beta: CB weighting hyperparameter (default 0.9999)
        gamma: Focal loss focusing parameter (default 2.0)
        reduction: 'mean' | 'sum' | 'none'
    """
    
    def __init__(
        self,
        beta: float = 0.9999,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer('pos_weights', None)
        self.register_buffer('neg_weights', None)
    
    def update_weights(self, labels: torch.Tensor):
        """Update class weights based on current batch labels."""
        if labels.dim() == 1:
            # Convert binary labels to one-hot for single-label case
            num_classes = int(labels.max()) + 1
            labels_onehot = torch.zeros(labels.shape[0], num_classes, device=labels.device)
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
            labels = labels_onehot
        
        pos_w, neg_w = compute_effective_number_weights(labels, self.beta)
        self.pos_weights = pos_w
        self.neg_weights = neg_w
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        use_dynamic_weights: bool = True
    ) -> torch.Tensor:
        """
        Compute Class-Balanced Focal Loss.
        
        Args:
            logits: Model predictions of shape (B, num_classes)
            targets: Binary targets of shape (B, num_classes)
            use_dynamic_weights: If True, compute weights from targets
            
        Returns:
            Loss value (scalar if reduction='mean')
        """
        if use_dynamic_weights:
            pos_w, neg_w = compute_effective_number_weights(targets, self.beta)
        else:
            pos_w = self.pos_weights
            neg_w = self.neg_weights
            if pos_w is None:
                # Fall back to uniform weights
                pos_w = torch.ones(targets.shape[1], device=targets.device)
                neg_w = torch.ones(targets.shape[1], device=targets.device)
        
        probs = torch.sigmoid(logits)
        
        # Compute BCE loss per element
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Focal weight: focus on hard examples
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply class-balanced weights
        # Use pos_w for positive samples, neg_w for negative samples
        cb_weight = pos_w * targets + neg_w * (1 - targets)
        
        loss = cb_weight * focal_weight * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class ClassBalancedLoss(nn.Module):
    """
    Pure Class-Balanced Loss (without focal weighting).
    
    Simpler variant that only applies CB reweighting to standard BCE loss.
    
    Args:
        beta: CB weighting hyperparameter (default 0.9999)
        reduction: 'mean' | 'sum' | 'none'
    """
    
    def __init__(self, beta: float = 0.9999, reduction: str = 'mean'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_w, neg_w = compute_effective_number_weights(targets, self.beta)
        
        # Standard BCE with class-balanced weights
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Apply CB weights
        cb_weight = pos_w * targets + neg_w * (1 - targets)
        loss = cb_weight * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def get_class_balanced_weights_from_dataloader(
    loader,
    beta: float = 0.9999,
    device: torch.device = None
) -> tuple:
    """
    Pre-compute class-balanced weights from entire dataset.
    
    This is useful when you want to compute weights once before training
    rather than dynamically during training.
    
    Args:
        loader: DataLoader containing the training dataset
        beta: CB hyperparameter
        device: Device to store weights on
        
    Returns:
        Tuple of (pos_weights, neg_weights) tensors
    """
    if device is None:
        device = torch.device('cpu')
    
    all_labels = []
    for batch in loader:
        if len(batch) == 3:  # Fusion dataset
            _, _, labels = batch
        else:
            _, labels = batch
        all_labels.append(labels)
    
    all_labels = torch.cat(all_labels, dim=0).to(device)
    pos_w, neg_w = compute_effective_number_weights(all_labels, beta)
    
    return pos_w, neg_w
