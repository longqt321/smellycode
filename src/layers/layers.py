"""Custom neural network layers for DCN v2 architecture."""
import torch
import torch.nn as nn
from typing import List, Optional
from config import ACTIVATION


class CrossLayer(nn.Module):
    """Standard Cross Layer for Deep & Cross Network.
    
    Implements: x_{l+1} = x_0 * (W_l * x_l + b_l) + x_l
    Uses low-rank decomposition for efficiency.
    """
    
    def __init__(self, dim: int, rank: int = 32):
        super().__init__()
        self.U = nn.Linear(dim, rank, bias=False)
        self.V = nn.Linear(rank, dim, bias=True)
    
    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        return x0 * self.V(self.U(xl)) + xl


class Bottleneck(nn.Module):
    """Bottleneck residual block with multiple compression stages.
    
    Args:
        dim: Input/output dimension
        ratios: Compression ratios for bottleneck layers (must divide dim)
    """
    
    def __init__(self, dim: int, ratios: List[int] = None):
        super().__init__()
        if ratios is None:
            ratios = [2, 4]
        
        for r in ratios:
            if dim % r != 0:
                raise ValueError(f"Dimension {dim} must be divisible by ratio {r}")
        
        bottle_dims = [dim // r for r in ratios]
        
        self.btn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, bottle_dims[0]),
            ACTIVATION(),
            
            nn.Linear(bottle_dims[0], bottle_dims[0]),
            nn.LayerNorm(bottle_dims[0]),
            ACTIVATION(),
            
            nn.Linear(bottle_dims[0], bottle_dims[1]),
            nn.LayerNorm(bottle_dims[1]),
            ACTIVATION(),
            
            nn.Linear(bottle_dims[1], bottle_dims[1]),
            nn.LayerNorm(bottle_dims[1]),
            ACTIVATION(),
            
            nn.Linear(bottle_dims[1], dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.btn(x) + x


class GatedCrossLayer(nn.Module):
    """Gated Cross Layer with learnable gating mechanism.
    
    Adds a sigmoid gate to control information flow from cross connection.
    """
    
    def __init__(self, dim: int, rank: int = 16):
        super().__init__()
        self.U = nn.Linear(dim, rank, bias=False)
        self.V = nn.Linear(rank, dim, bias=True)
        self.gate_layer = nn.Linear(dim, dim)
        
        # Initialize gate to favor identity mapping at start
        nn.init.zeros_(self.gate_layer.bias)
        nn.init.xavier_uniform_(self.gate_layer.weight, gain=0.01)
    
    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_layer(xl))
        cross = x0 * self.V(self.U(xl))
        return gate * cross + xl


class MoEBottleneck(nn.Module):
    """Mixture of Experts Bottleneck layer.
    
    Routes input through top-k experts and combines outputs weighted by gate values.
    
    Args:
        dim: Input/output dimension
        num_experts: Number of expert networks
        top_k: Number of experts to use per sample
        dropout: Dropout rate (currently unused, reserved for future)
    """
    
    def __init__(self, dim: int, num_experts: int = 4, top_k: int = 2, dropout: float = 0.2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.GELU(),
                nn.Linear(dim // 4, dim)
            )
            for _ in range(num_experts)
        ])
        self.gate = nn.Linear(dim, num_experts)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        gate_logits = self.gate(x)
        top_k_vals, top_k_idx = torch.topk(gate_logits, self.top_k, dim=-1)
        weights = torch.softmax(top_k_vals, dim=-1)
        
        out = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = top_k_idx[:, k]
            expert_weight = weights[:, k].unsqueeze(1)
            
            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_out = self.experts[e](x[mask])
                    out[mask] += expert_weight[mask] * expert_out
        
        return self.norm(out + x)
