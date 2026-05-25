import torch
import torch.nn as nn
from src.layers.layers import CrossLayer, GatedCrossLayer, Bottleneck, MoEBottleneck
from config import ACTIVATION

class DCNv2(nn.Module):
    def __init__(self, input_dim, projection_dim=128, cross_layers=4, num_deep_layers=2,
                 embed_dim=128, num_classes=4, cross_type='standard', deep_type='bottleneck'):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            ACTIVATION(),
        )
        CrossClass = CrossLayer if cross_type == 'standard' else GatedCrossLayer
        self.cross_layers = nn.ModuleList([CrossClass(projection_dim) for _ in range(cross_layers)])
        DeepClass = Bottleneck if deep_type == 'bottleneck' else MoEBottleneck
        self.deep = nn.Sequential(*[DeepClass(projection_dim) for _ in range(num_deep_layers)])
        self.embedding = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.gate_layer = nn.Linear(embed_dim,embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        x0 = self.proj(x)
        xc = x0.clone()
        for layer in self.cross_layers:
            xc = layer(x0, xc)
        xd = self.deep(x0)
        
        gate = torch.sigmoid(self.gate_layer(xd))
        
        h = xd + (gate*xc)
        
        embed = self.embedding(h)
        return self.classifier(embed), embed
