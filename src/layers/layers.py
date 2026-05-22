import torch
import torch.nn as nn
from config import *

class CrossLayer(nn.Module):
    def __init__(self,dim,rank=16):
        super().__init__()
        self.U = nn.Linear(dim,rank,bias=False)
        self.V = nn.Linear(rank,dim,bias=True)
    def forward(self,x0,xl):
        return x0 * self.V(self.U(xl)) + xl
    

class Bottleneck(nn.Module):
    def __init__(self,dim,ratio=4,dropout=0.2):
        super().__init__()
        assert dim % ratio == 0, "Dim must be divisible by ratio"
        bottleneck_dim = dim // ratio
        
        self.btn = nn.Sequential(
            nn.Linear(dim,bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            ACTIVATION,
            
            nn.Linear(bottleneck_dim,bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            ACTIVATION,
            
            nn.Linear(bottleneck_dim,dim),
            nn.LayerNorm(dim),            
        )
        self.act = ACTIVATION
    def forward(self,x):
        return self.act(self.btn(x) + x)
