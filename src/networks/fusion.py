import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
from src.networks.dcn import DCNv2


def precompute_bert_embeddings(codes: list, tokenizer, bert: nn.Module,
                                device, batch_size: int = 64, max_length: int = 512) -> torch.Tensor:
    """
    Pre-compute frozen BERT [CLS] embeddings for all code strings.
    Returns tensor of shape (N, BERT_DIM) on CPU.
    """
    bert.eval()
    all_embeds = []
    with torch.no_grad():
        for i in range(0, len(codes), batch_size):
            batch = codes[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=max_length, return_tensors='pt')
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == 'cuda'):
                out = bert(**enc)
            all_embeds.append(out.last_hidden_state[:, 0].float().cpu())
    return torch.cat(all_embeds, dim=0)  # (N, 768)


class GatedFusionModel(nn.Module):
    """
    Late Gated Fusion of GraphCodeBERT (frozen, feature extractor) and DCNv2 (numeric branch).

    Architecture:
        numeric_branch : DCNv2  -> embed_dim
        text_branch    : GraphCodeBERT [CLS] (frozen) -> project -> embed_dim
        gate           : sigmoid(W * [h_num; h_txt]) -> (0,1)^embed_dim
        fused          : gate * h_num + (1 - gate) * h_txt
        classifier     : Linear(embed_dim, num_classes)
    """
    BERT_MODEL = "microsoft/graphcodebert-base"
    BERT_DIM = 768

    def __init__(self, input_dim: int, embed_dim: int = 128, num_classes: int = 4,
                 cross_type: str = 'standard', deep_type: str = 'bottleneck'):
        super().__init__()
        self.embed_dim = embed_dim
        self.numeric = DCNv2(input_dim=input_dim, embed_dim=embed_dim,
                             num_classes=num_classes, cross_type=cross_type, deep_type=deep_type)
        self.bert = AutoModel.from_pretrained(self.BERT_MODEL)
        for p in self.bert.parameters():
            p.requires_grad = False

        self.text_proj = nn.Sequential(
            nn.Linear(self.BERT_DIM, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim*2),
            nn.Sigmoid(),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, features: torch.Tensor, bert_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features:   (B, input_dim) numeric features
            bert_embed: (B, BERT_DIM) pre-computed frozen BERT [CLS] embeddings
        """
        _, h_num = self.numeric(features)
        h_txt = self.text_proj(bert_embed.to(features.device))
        gates = self.gate(torch.cat([h_num, h_txt], dim=1))
        gate_num = gates[:, :self.embed_dim]
        gate_txt = gates[:, self.embed_dim:]
        return self.classifier(gate_num * h_num + gate_txt * h_txt)


def get_tokenizer():
    return AutoTokenizer.from_pretrained(GatedFusionModel.BERT_MODEL)
