import torch
import torch.nn as nn
import torch.nn.functional as F


class MultilabelFocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.
    Treats each label as an independent binary problem.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: focusing parameter. 0 = standard BCE. Higher = more focus on hard examples.
        alpha: per-class positive weight tensor, shape (num_classes,).
               Analogous to pos_weight in BCEWithLogitsLoss.
        reduction: 'mean' | 'sum' | 'none'
    """
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer('alpha', alpha if alpha is not None else torch.tensor(1.0))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Numerically stable: compute BCE per-element then apply focal weight
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        # p_t: probability of the true class
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * bce

        # Apply alpha weighting for positive class
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - targets)
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for multi-label classification.
    Uses separate gamma for positive/negative samples + clipping to suppress easy negatives.

    Args:
        gamma_neg: focusing for negative samples (default 4). Higher = more suppression of easy negatives.
        gamma_pos: focusing for positive samples (default 1).
        clip: shifts p_neg up by this value to zero-out trivially easy negatives.
        alpha: per-class positive weight tensor, shape (num_classes,).
        reduction: 'mean' | 'sum' | 'none'
    """
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, clip: float = 0.01,
                 alpha: torch.Tensor = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction
        self.register_buffer('alpha', alpha if alpha is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        probs = torch.sigmoid(logits)
        p_pos = probs
        p_neg = 1.0 - probs

        if self.clip is not None and self.clip > 0:
            p_neg = (p_neg + self.clip).clamp(max=1.0)

        focal_weight_pos = (1.0 - p_pos) ** self.gamma_pos
        focal_weight_neg = (1.0 - p_neg) ** self.gamma_neg

        loss = (focal_weight_pos * (-targets * F.logsigmoid(logits)) +
                focal_weight_neg * (-(1.0 - targets) * F.logsigmoid(-logits)))

        if self.alpha is not None:
            loss = (self.alpha * targets + (1.0 - targets)) * loss

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss
