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


class CrossModalBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attn_text = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_image = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_tabular = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm_attn = nn.LayerNorm(hidden_size)
        self.norm_ffn = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_size, hidden_size),
        )

    def forward(self, tokens: torch.Tensor, return_attention: bool = False):
        normed = self.norm_attn(tokens)
        text_out, text_attn = self.attn_text(
            normed[:, 0:1, :],
            normed,
            normed,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        image_out, image_attn = self.attn_image(
            normed[:, 1:2, :],
            normed,
            normed,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        tab_out, tab_attn = self.attn_tabular(
            normed[:, 2:3, :],
            normed,
            normed,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        attended = torch.cat([text_out, image_out, tab_out], dim=1)
        tokens = tokens + self.dropout(attended)
        ffn_out = self.ffn(self.norm_ffn(tokens))
        tokens = tokens + self.dropout(ffn_out)
        if not return_attention:
            return tokens, None
        return (
            tokens,
            {
                "text": text_attn,
                "image": image_attn,
                "tabular": tab_attn,
            },
        )


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

        self.modality_gating = (
            nn.Sequential(
                nn.Linear(config.model.fusion_hidden_size * 3, config.model.fusion_hidden_size),
                nn.ReLU(),
                nn.Dropout(config.model.dropout),
                nn.Linear(config.model.fusion_hidden_size, 3),
            )
            if config.model.use_modality_gating
            else None
        )
        self.fusion_blocks = nn.ModuleList(
            [
                CrossModalBlock(
                    hidden_size=config.model.fusion_hidden_size,
                    num_heads=config.model.num_attention_heads,
                    ffn_size=config.model.fusion_ffn_size,
                    dropout=config.model.dropout,
                )
                for _ in range(max(int(config.model.fusion_layers), 1))
            ]
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
        return_attention: bool = False,
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
        gate_weights = None
        if self.modality_gating is not None:
            logits = self.modality_gating(fusion_tokens.reshape(fusion_tokens.size(0), -1))
            gate_weights = torch.softmax(logits, dim=-1)
            fusion_tokens = fusion_tokens * gate_weights.unsqueeze(-1)

        attentions = [] if return_attention else None
        for block in self.fusion_blocks:
            fusion_tokens, attn = block(fusion_tokens, return_attention=return_attention)
            if return_attention:
                attentions.append(attn)

        fused = fusion_tokens.reshape(fusion_tokens.size(0), -1)
        preds = self.regressor(fused).squeeze(-1)
        if not return_attention:
            return preds
        return preds, {"gates": gate_weights, "attentions": attentions}
