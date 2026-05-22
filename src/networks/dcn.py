import torch
import torch.nn as nn
from src.layers.layers import CrossLayer,Bottleneck
from config import *

class DCNv2(nn.Module):
    def __init__(self,input_dim,projection_dim=128,cross_layers=4,deep_layers=(256,128),embed_dim=128,num_classes=4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim,projection_dim),
            nn.LayerNorm(projection_dim),
            ACTIVATION,
        )
        
        self.cross_layers= nn.ModuleList([
            CrossLayer(projection_dim) for _ in range(cross_layers)
        ])
        
        deep_modules=[]
        dim = projection_dim
        for d in deep_layers:
            deep_modules.extend([
                Bottleneck(d)
            ])
            dim=d
        self.deep = nn.Sequential(*deep_modules)
        
        self.embedding = nn.Sequential(
            nn.Linear(projection_dim + dim,embed_dim),
            nn.LayerNorm(embed_dim),
            ACTIVATION
        )
        self.classifier=nn.Linear(embed_dim,num_classes)
    def forward(self,x):
        x0 = self.proj(x)
        xc = x0
        for layer in self.cross_layers:
            xc = layer(x0,xc)
        xd = self.deep(x0)
        h = torch.cat([xc,xd],dim=1)
        embed=self.embedding(h)
        return self.classifier(embed),embed
            