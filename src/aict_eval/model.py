from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from transformers import AutoModel

from .config import AICTConfig


class LocalTextEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.dropout(self.norm(pooled))


class MultiModalEvaluator(nn.Module):
    def __init__(self, config: AICTConfig, tabular_dim: int) -> None:
        super().__init__()
        self.config = config

        self.use_transformer_text = True
        try:
            self.text_encoder = AutoModel.from_pretrained(
                config.model.text_model_name,
                local_files_only=not config.model.allow_online_model_download,
            )
        except Exception:
            self.use_transformer_text = False
            self.text_encoder = LocalTextEncoder(
                vocab_size=config.model.local_text_vocab_size,
                hidden_size=config.model.text_hidden_size,
                dropout=config.model.dropout,
            )
        self.text_projector = nn.Sequential(
            nn.Linear(config.model.text_hidden_size, config.model.fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
        )

        try:
            backbone = (
                resnet18(weights=ResNet18_Weights.DEFAULT)
                if config.model.allow_online_model_download
                else resnet18(weights=None)
            )
        except Exception:
            backbone = resnet18(weights=None)
        image_hidden = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.image_encoder = backbone
        self.image_projector = nn.Sequential(
            nn.Linear(image_hidden, config.model.fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
        )

        self.tabular_projector = nn.Sequential(
            nn.Linear(tabular_dim, config.model.tabular_hidden_size),
            nn.ReLU(),
            nn.Linear(config.model.tabular_hidden_size, config.model.fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.model.fusion_hidden_size,
            num_heads=config.model.num_attention_heads,
            dropout=config.model.dropout,
            batch_first=True,
        )

        self.regressor = nn.Sequential(
            nn.Linear(config.model.fusion_hidden_size * 3, config.model.fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.fusion_hidden_size, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image: torch.Tensor,
        tabular: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_transformer_text:
            text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_cls = text_output.last_hidden_state[:, 0, :]
        else:
            text_cls = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = self.text_projector(text_cls)

        image_features = self.image_projector(self.image_encoder(image))
        tabular_features = self.tabular_projector(tabular)

        fusion_tokens = torch.stack([text_features, image_features, tabular_features], dim=1)
        attended_tokens, _ = self.cross_attention(fusion_tokens, fusion_tokens, fusion_tokens)
        fused = attended_tokens.reshape(attended_tokens.size(0), -1)
        return self.regressor(fused).squeeze(-1)
