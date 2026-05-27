"""Deep & Cross Network v2 architecture for tabular data."""
import torch
import torch.nn as nn
from typing import Literal, Optional
from src.layers.layers import CrossLayer, GatedCrossLayer, Bottleneck, MoEBottleneck
from config import ACTIVATION


class DCNv2(nn.Module):
    """Deep & Cross Network v2 with gated fusion.
    
    Combines parallel cross and deep networks with learnable gating mechanism.
    
    Args:
        input_dim: Dimension of input features
        projection_dim: Hidden dimension for projections
        cross_layers: Number of cross layers
        num_deep_layers: Number of deep bottleneck layers
        embed_dim: Final embedding dimension
        num_classes: Number of output classes (optional)
        cross_type: Type of cross layer ('standard' or 'gated')
        deep_type: Type of deep layer ('bottleneck' or 'moe')
    """
    
    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 128,
        cross_layers: int = 4,
        num_deep_layers: int = 2,
        embed_dim: int = 128,
        num_classes: Optional[int] = None,
        cross_type: Literal['standard', 'gated'] = 'standard',
        deep_type: Literal['bottleneck', 'moe'] = 'bottleneck',
    ):
        super().__init__()
        
        # Input projection
        self.proj = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            ACTIVATION(),
        )
        
        # Cross network
        CrossClass = CrossLayer if cross_type == 'standard' else GatedCrossLayer
        self.cross_layers = nn.ModuleList([
            CrossClass(projection_dim) for _ in range(cross_layers)
        ])
        
        # Deep network
        DeepClass = Bottleneck if deep_type == 'bottleneck' else MoEBottleneck
        self.deep = nn.Sequential(*[
            DeepClass(projection_dim) for _ in range(num_deep_layers)
        ])
        
        # Embedding and gating
        self.embedding = nn.Sequential(
            nn.Linear(projection_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.gate_layer = nn.Linear(projection_dim, projection_dim)
        
        # Optional classifier head
        self.num_classes = num_classes
        if num_classes is not None:
            self.classifier = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x: torch.Tensor):
        # Project input
        x0 = self.proj(x)
        
        # Apply cross layers
        xc = x0.clone()
        for layer in self.cross_layers:
            xc = layer(x0, xc)
        
        # Apply deep layers
        xd = self.deep(x0)
        
        # Gated fusion of cross and deep outputs
        gate = torch.sigmoid(self.gate_layer(xd))
        h = xd + (gate * xc)
        
        # Generate embedding
        embed = self.embedding(h)
        
        # Return classification logits if classifier exists
        if self.num_classes is not None:
            return self.classifier(embed), embed
        return None, embed
