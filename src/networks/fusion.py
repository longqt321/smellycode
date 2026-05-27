"""Gated fusion model combining GraphCodeBERT embeddings with DCNv2."""
import torch
import torch.nn as nn
from typing import Literal, Optional
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
from src.networks.dcn import DCNv2


def precompute_bert_embeddings(
    codes: list,
    tokenizer: AutoTokenizer,
    bert: nn.Module,
    device: torch.device,
    batch_size: int = 64,
    max_length: int = 512
) -> torch.Tensor:
    """Pre-compute frozen BERT [CLS] embeddings for all code strings.
    
    Args:
        codes: List of code strings to embed
        tokenizer: Tokenizer for the BERT model
        bert: BERT model for embedding extraction
        device: Device to run inference on
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        
    Returns:
        Tensor of shape (N, BERT_DIM) on CPU
    """
    bert.eval()
    all_embeds = []
    
    with torch.no_grad():
        for i in range(0, len(codes), batch_size):
            batch = codes[i:i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == 'cuda'
            ):
                out = bert(**enc)
            
            all_embeds.append(out.last_hidden_state[:, 0].float().cpu())
    
    return torch.cat(all_embeds, dim=0)


class GatedFusionModel(nn.Module):
    """Late Gated Fusion of GraphCodeBERT and DCNv2.
    
    Architecture:
        - numeric_branch: DCNv2 -> embed_dim
        - text_branch: GraphCodeBERT [CLS] (frozen) -> project -> embed_dim
        - gate: sigmoid(W * [h_num; h_txt]) -> (0,1)^embed_dim
        - fused: gate * h_num + (1 - gate) * h_txt
 clear
 - classifier: Linear(embed_dim, num_classes)
    
    Attributes:
        BERT_MODEL: Name of the pretrained GraphCodeBERT model
        BERT_DIM: Dimension of BERT embeddings (768 for base model)
    """
    
    BERT_MODEL = "microsoft/graphcodebert-base"
    BERT_DIM = 768
    
    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 128,
        num_classes: int = 4,
        cross_type: Literal['standard', 'gated'] = 'standard',
        deep_type: Literal['bottleneck', 'moe'] = 'bottleneck',
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Numeric feature branch
        self.numeric = DCNv2(
            input_dim=input_dim,
            embed_dim=embed_dim,
            cross_type=cross_type,
            deep_type=deep_type,
        )
        self.last_gate = None
        
        # Text embedding projection
        self.text_proj = nn.Sequential(
            nn.Linear(self.BERT_DIM, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )
        
        # Gating mechanism
        self.gate_layer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
        
        # Classification head
        self.classifier = nn.Linear(embed_dim, num_classes)
    
    def forward(
        self,
        features: torch.Tensor,
        bert_embed: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with gated fusion.
        
        Args:
            features: Numeric features of shape (B, input_dim)
            bert_embed: Pre-computed BERT embeddings of shape (B, BERT_DIM)
            
        Returns:
            Classification logits of shape (B, num_classes)
        """
        _, h_num = self.numeric(features)
        h_txt = self.text_proj(bert_embed.to(features.device))
        
        gate = self.gate_layer(torch.cat([h_num, h_txt], dim=1))
        self.last_gate = gate.detach()
        fused_features = gate * h_num + (1.0 - gate) * h_txt
        
        return self.classifier(fused_features)

    def gate_stats(self) -> dict:
        """Return summary statistics for the most recent fusion gate."""
        if self.last_gate is None:
            return {}
        gate = self.last_gate.float()
        return {
            'gate/mean': gate.mean().item(),
            'gate/std': gate.std(unbiased=False).item(),
            'gate/min': gate.min().item(),
            'gate/max': gate.max().item(),
            'gate/near_zero': (gate < 0.05).float().mean().item(),
            'gate/near_one': (gate > 0.95).float().mean().item(),
        }


class LateFusionMLPModel(nn.Module):
    """Late fusion baseline using concat([numeric, text]) followed by an MLP head."""

    BERT_MODEL = GatedFusionModel.BERT_MODEL
    BERT_DIM = GatedFusionModel.BERT_DIM

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 128,
        num_classes: int = 4,
        cross_type: Literal['standard', 'gated'] = 'standard',
        deep_type: Literal['bottleneck', 'moe'] = 'bottleneck',
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim * 2

        self.numeric = DCNv2(
            input_dim=input_dim,
            embed_dim=embed_dim,
            cross_type=cross_type,
            deep_type=deep_type,
        )
        self.text_proj = nn.Sequential(
            nn.Linear(self.BERT_DIM, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, features: torch.Tensor, bert_embed: torch.Tensor) -> torch.Tensor:
        _, h_num = self.numeric(features)
        h_txt = self.text_proj(bert_embed.to(features.device))
        fused_features = self.fusion_mlp(torch.cat([h_num, h_txt], dim=1))
        return self.classifier(fused_features)


def get_tokenizer() -> AutoTokenizer:
    """Load the GraphCodeBERT tokenizer."""
    return AutoTokenizer.from_pretrained(GatedFusionModel.BERT_MODEL)
