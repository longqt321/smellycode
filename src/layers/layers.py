import torch
import torch.nn as nn
from config import *

class CrossLayer(nn.Module):
    def __init__(self,dim,rank=32):
        super().__init__()
        self.U = nn.Linear(dim,rank,bias=False)
        self.V = nn.Linear(rank,dim,bias=True)
    def forward(self,x0,xl):
        return x0 * self.V(self.U(xl)) + xl
    

class Bottleneck(nn.Module):
    def __init__(self,dim,ratio=4):
        super().__init__()
        assert dim % ratio == 0, "Dim must be divisible by ratio"
        bottleneck_dim = dim // ratio
        
        self.btn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim,bottleneck_dim),
            ACTIVATION(),
            
            nn.Linear(bottleneck_dim,bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            ACTIVATION(),
            
            nn.Linear(bottleneck_dim,dim),
        )
    def forward(self,x):
        return self.btn(x) + x

class GatedCrossLayer(nn.Module):
    def __init__(self, dim, rank=32):
        super().__init__()
        self.U = nn.Linear(dim, rank, bias=False)
        self.V = nn.Linear(rank, dim, bias=True)
        self.gate_layer = nn.Linear(dim, dim)
        
        nn.init.zeros_(self.gate_layer.bias)
        nn.init.xavier_uniform_(self.gate_layer.weight, gain=0.01)
    def forward(self, x0, xl):
        gate = torch.sigmoid(self.gate_layer(xl))
        cross = x0 * self.V(self.U(xl))
        return gate * cross + xl

class MoEBottleneck(nn.Module):
    def __init__(self, dim, num_experts=4, top_k=2, dropout=0.2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(dim, dim//4),
            nn.GELU(),
            nn.Linear(dim//4, dim)
        ) for _ in range(num_experts)])
        self.gate = nn.Linear(dim, num_experts)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        batch_size, d = x.shape
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
