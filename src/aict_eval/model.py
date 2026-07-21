from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

try:
    from transformers import AutoModel
except Exception:
    AutoModel = None

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
        modality_names: list[str],
    ) -> None:
        super().__init__()
        self.modality_names = list(modality_names)
        self.attn_layers = nn.ModuleDict(
            {
                name: nn.MultiheadAttention(
                    embed_dim=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for name in self.modality_names
            }
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
        outputs = []
        attentions = {}
        for idx, name in enumerate(self.modality_names):
            out, attn = self.attn_layers[name](
                normed[:, idx : idx + 1, :],
                normed,
                normed,
                need_weights=return_attention,
                average_attn_weights=False,
            )
            outputs.append(out)
            if return_attention:
                attentions[name] = attn
        attended = torch.cat(outputs, dim=1)
        tokens = tokens + self.dropout(attended)
        ffn_out = self.ffn(self.norm_ffn(tokens))
        tokens = tokens + self.dropout(ffn_out)
        if not return_attention:
            return tokens, None
        return tokens, attentions


class StatsAudioEncoder(nn.Module):
    def __init__(self, config: AICTConfig) -> None:
        super().__init__()
        self.n_fft = int(config.model.audio_n_fft)
        self.hop_length = int(config.model.audio_hop_length)
        stats_dim = (self.n_fft // 2 + 1) * 2
        self.projector = nn.Sequential(
            nn.Linear(stats_dim, config.model.audio_hidden_size),
            nn.ReLU(),
            nn.LayerNorm(config.model.audio_hidden_size),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.audio_hidden_size, config.model.fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        window = torch.hann_window(self.n_fft, device=waveform.device)
        spectrum = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
        )
        magnitude = torch.log1p(spectrum.abs())
        stats = torch.cat(
            [
                magnitude.mean(dim=-1),
                magnitude.std(dim=-1),
            ],
            dim=-1,
        )
        return self.projector(stats)


class PretrainedAudioEncoder(nn.Module):
    def __init__(self, config: AICTConfig) -> None:
        super().__init__()
        self.backbone_type = config.model.audio_backbone_type.lower()
        self.use_pretrained = False
        self.fallback_encoder = StatsAudioEncoder(config)
        self.backbone = None
        hidden_size = config.model.audio_hidden_size

        if AutoModel is not None and self.backbone_type != "stats":
            try:
                self.backbone = AutoModel.from_pretrained(
                    config.model.audio_model_name,
                    local_files_only=not config.model.allow_online_model_download,
                )
                self.use_pretrained = True
                hidden_size = int(
                    getattr(self.backbone.config, "hidden_size", 0)
                    or getattr(self.backbone.config, "d_model", 0)
                    or getattr(self.backbone.config, "classifier_proj_size", 0)
                    or config.model.audio_hidden_size
                )
            except Exception:
                self.backbone = None
                self.use_pretrained = False

        self.projector = nn.Sequential(
            nn.Linear(hidden_size, config.model.audio_hidden_size),
            nn.ReLU(),
            nn.LayerNorm(config.model.audio_hidden_size),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.audio_hidden_size, config.model.fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
        )

    def _masked_mean_pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        mask = attention_mask.to(hidden_states.device).float()
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    def forward(self, audio_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        waveform = audio_inputs.get("waveform")
        if waveform is None:
            raise ValueError("音频输入缺少 waveform，无法执行语音编码。")
        if not self.use_pretrained or self.backbone is None:
            return self.fallback_encoder(waveform)

        if self.backbone_type == "whisper":
            input_features = audio_inputs.get("input_features")
            if input_features is None:
                return self.fallback_encoder(waveform)
            outputs = self.backbone(input_features=input_features)
            pooled = outputs.last_hidden_state.mean(dim=1)
            return self.projector(pooled)

        input_values = audio_inputs.get("input_values")
        if input_values is None:
            return self.fallback_encoder(waveform)
        attention_mask = audio_inputs.get("attention_mask")
        model_kwargs = {"input_values": input_values}
        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask
        outputs = self.backbone(**model_kwargs)
        pooled = self._masked_mean_pool(outputs.last_hidden_state, attention_mask)
        return self.projector(pooled)


class MultiModalEvaluator(nn.Module):
    def __init__(self, config: AICTConfig, tabular_dim: int) -> None:
        super().__init__()
        self.config = config
        self.use_audio = bool(config.train.audio_column)
        self.modality_names = ["text", "image"]
        if self.use_audio:
            self.modality_names.append("audio")
        self.modality_names.append("tabular")

        self.use_transformer_text = True
        if AutoModel is None:
            self.use_transformer_text = False
            self.text_encoder = LocalTextEncoder(
                vocab_size=config.model.local_text_vocab_size,
                hidden_size=config.model.text_hidden_size,
                dropout=config.model.dropout,
            )
        else:
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
        self.audio_encoder = PretrainedAudioEncoder(config) if self.use_audio else None

        self.tabular_projector = nn.Sequential(
            nn.Linear(tabular_dim, config.model.tabular_hidden_size),
            nn.ReLU(),
            nn.Linear(config.model.tabular_hidden_size, config.model.fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
        )

        self.modality_gating = (
            nn.Sequential(
                nn.Linear(
                    config.model.fusion_hidden_size * len(self.modality_names),
                    config.model.fusion_hidden_size,
                ),
                nn.ReLU(),
                nn.Dropout(config.model.dropout),
                nn.Linear(config.model.fusion_hidden_size, len(self.modality_names)),
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
                    modality_names=self.modality_names,
                )
                for _ in range(max(int(config.model.fusion_layers), 1))
            ]
        )

        self.regressor = nn.Sequential(
            nn.Linear(
                config.model.fusion_hidden_size * len(self.modality_names),
                config.model.fusion_hidden_size,
            ),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.fusion_hidden_size, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image: torch.Tensor,
        audio_inputs: dict[str, torch.Tensor] | None,
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
        feature_tokens = [text_features, image_features]
        if self.use_audio:
            if audio_inputs is None:
                raise ValueError("当前模型已启用语音模态，但未收到 audio_inputs 输入。")
            feature_tokens.append(self.audio_encoder(audio_inputs))
        tabular_features = self.tabular_projector(tabular)
        feature_tokens.append(tabular_features)
        fusion_tokens = torch.stack(feature_tokens, dim=1)
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
        return preds, {"gates": gate_weights, "attentions": attentions, "modality_names": self.modality_names}
