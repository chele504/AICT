from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50

try:
    from transformers import AutoModel
except Exception:
    AutoModel = None

from .config import AICTConfig


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class LocalTextEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if 2 > 1 else 0.0,
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).float()
        lstm_out, _ = self.lstm(embedded)
        attn_scores = self.attn(lstm_out).squeeze(-1)
        attn_scores = attn_scores.masked_fill(attention_mask == 0, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1).unsqueeze(-1)
        pooled = (lstm_out * attn_weights).sum(dim=1)
        mean_pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        combined = pooled + mean_pooled
        return self.dropout(self.norm(combined))


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
        self.num_modalities = len(self.modality_names)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        self.q_projs = nn.ModuleDict({name: nn.Linear(hidden_size, hidden_size) for name in self.modality_names})
        self.k_projs = nn.ModuleDict({name: nn.Linear(hidden_size, hidden_size) for name in self.modality_names})
        self.v_projs = nn.ModuleDict({name: nn.Linear(hidden_size, hidden_size) for name in self.modality_names})
        self.out_projs = nn.ModuleDict({name: nn.Linear(hidden_size, hidden_size) for name in self.modality_names})

        self.modality_coeff = nn.Parameter(torch.ones(self.num_modalities, self.num_modalities) / self.num_modalities)
        self.dropout = nn.Dropout(dropout)
        self.norm_attn = nn.LayerNorm(hidden_size)
        self.norm_ffn = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_size, hidden_size),
            nn.Dropout(dropout),
        )

    def _reshape_for_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        x = x.view(b, n, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, tokens: torch.Tensor, return_attention: bool = False):
        normed = self.norm_attn(tokens)
        outputs = []
        attentions = {}
        coeff = torch.softmax(self.modality_coeff, dim=-1)

        for idx_q, q_name in enumerate(self.modality_names):
            q = self.q_projs[q_name](normed[:, idx_q : idx_q + 1, :])
            q = self._reshape_for_heads(q)
            attended_sum = 0.0
            attn_map = {}
            for idx_k, k_name in enumerate(self.modality_names):
                k = self.k_projs[k_name](normed)
                v = self.v_projs[k_name](normed)
                k = self._reshape_for_heads(k)
                v = self._reshape_for_heads(v)
                scale = 1.0 / math.sqrt(self.head_dim)
                scores = torch.matmul(q, k.transpose(-2, -1)) * scale
                attn = torch.softmax(scores, dim=-1)
                attn_dropped = self.dropout(attn)
                if return_attention:
                    attn_map[f"to_{k_name}"] = attn.detach().cpu()
                attended = torch.matmul(attn_dropped, v)
                attended_sum = attended_sum + coeff[idx_q, idx_k] * attended
            attended_sum = attended_sum.transpose(1, 2).contiguous()
            b, _, _, _ = attended_sum.shape
            attended_sum = attended_sum.view(b, 1, self.hidden_size)
            out = self.out_projs[q_name](attended_sum)
            outputs.append(out)
            if return_attention:
                attentions[q_name] = attn_map

        attended = torch.cat(outputs, dim=1)
        tokens = tokens + self.dropout(attended)
        ffn_out = self.ffn(self.norm_ffn(tokens))
        tokens = tokens + ffn_out
        if not return_attention:
            return tokens, None
        return tokens, attentions


class StatsAudioEncoder(nn.Module):
    def __init__(self, config: AICTConfig) -> None:
        super().__init__()
        self.n_fft = int(config.model.audio_n_fft)
        self.hop_length = int(config.model.audio_hop_length)
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)
        n_mels = min(64, self.n_fft // 2 + 1)
        self.n_mels = n_mels
        stats_dim = n_mels * 6 + 8
        self.projector = nn.Sequential(
            nn.Linear(stats_dim, config.model.audio_hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.model.audio_hidden_size),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.audio_hidden_size, config.model.audio_hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.model.audio_hidden_size),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.audio_hidden_size, config.model.fusion_hidden_size),
        )
        mel_scale = torch.nn.functional.pad(
            torch.linspace(0, self.n_fft // 2, n_mels).unsqueeze(0),
            (0, 0, 0, self.n_fft // 2 + 1 - n_mels),
        )
        dummy = torch.zeros(1, self.n_fft // 2 + 1)
        self.register_buffer("_mel_init", dummy, persistent=False)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        window = self.window.to(waveform.device)
        spectrum = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
        )
        magnitude = spectrum.abs()
        log_mag = torch.log1p(magnitude)
        power = magnitude.pow(2)
        power_sum = power.sum(dim=1, keepdim=True).clamp(min=1e-8)
        freqs = torch.linspace(0, 1, magnitude.shape[1], device=waveform.device).view(1, -1, 1)
        spectral_centroid = (freqs * power).sum(dim=1) / power_sum.squeeze(1)
        spectral_energy_cum = torch.cumsum(power, dim=1) / power_sum
        rolloff_mask = (spectral_energy_cum >= 0.85).float()
        spectral_rolloff = (freqs * rolloff_mask).sum(dim=1) / rolloff_mask.sum(dim=1).clamp(min=1.0)
        spectral_bandwidth = torch.sqrt(((freqs - spectral_centroid.unsqueeze(1)).pow(2) * power).sum(dim=1) / power_sum.squeeze(1))
        spectral_flatness = torch.exp(log_mag.mean(dim=1)) / (magnitude.mean(dim=1).clamp(min=1e-8))
        zcr = torch.abs(waveform[:, 1:] - waveform[:, :-1]).gt(0).float().mean(dim=-1, keepdim=True)
        rms = torch.sqrt(waveform.pow(2).mean(dim=-1, keepdim=True).clamp(min=1e-8))
        waveform_std = waveform.std(dim=-1, keepdim=True)
        waveform_skew = (waveform - waveform.mean(dim=-1, keepdim=True)).pow(3).mean(dim=-1, keepdim=True) / (waveform_std.pow(3).clamp(min=1e-8))
        waveform_kurt = (waveform - waveform.mean(dim=-1, keepdim=True)).pow(4).mean(dim=-1, keepdim=True) / (waveform_std.pow(4).clamp(min=1e-8))
        n_mels = self.n_mels
        time_size = magnitude.shape[-1]
        if magnitude.shape[1] >= n_mels:
            idx = torch.linspace(0, magnitude.shape[1] - 1, n_mels, dtype=torch.long, device=waveform.device)
            mag_down = magnitude[:, idx, :]
            log_down = log_mag[:, idx, :]
        else:
            pad_size = n_mels - magnitude.shape[1]
            mag_down = F.pad(magnitude, (0, 0, 0, pad_size))
            log_down = F.pad(log_mag, (0, 0, 0, pad_size))
        stats = torch.cat(
            [
                log_down.mean(dim=-1),
                log_down.std(dim=-1),
                log_down.max(dim=-1).values,
                mag_down.mean(dim=-1),
                mag_down.std(dim=-1),
                mag_down.max(dim=-1).values,
                spectral_centroid.mean(dim=-1, keepdim=True),
                spectral_rolloff.mean(dim=-1, keepdim=True),
                spectral_bandwidth.mean(dim=-1, keepdim=True),
                spectral_flatness.mean(dim=-1, keepdim=True),
                zcr,
                rms,
                waveform_skew,
                waveform_kurt,
            ],
            dim=-1,
        )
        stats = torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)
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
            nn.GELU(),
            nn.LayerNorm(config.model.audio_hidden_size),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.audio_hidden_size, config.model.audio_hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.model.audio_hidden_size),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.audio_hidden_size, config.model.fusion_hidden_size),
        )
        self._hidden_size = hidden_size

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

        text_out_dim = config.model.text_hidden_size
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
                text_out_dim = int(
                    getattr(self.text_encoder.config, "hidden_size", config.model.text_hidden_size)
                )
            except Exception:
                self.use_transformer_text = False
                self.text_encoder = LocalTextEncoder(
                    vocab_size=config.model.local_text_vocab_size,
                    hidden_size=config.model.text_hidden_size,
                    dropout=config.model.dropout,
                )
        self.text_projector = nn.Sequential(
            nn.Linear(text_out_dim, config.model.fusion_hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.model.fusion_hidden_size),
            nn.Dropout(config.model.dropout),
        )

        image_backbone = config.model.image_model_name.lower()
        try:
            if "resnet50" in image_backbone:
                backbone = (
                    resnet50(weights=ResNet50_Weights.DEFAULT)
                    if config.model.allow_online_model_download
                    else resnet50(weights=None)
                )
            else:
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
        self.image_se = SEBlock(image_hidden, reduction=max(8, image_hidden // 64))
        self.image_projector = nn.Sequential(
            nn.Linear(image_hidden, config.model.fusion_hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.model.fusion_hidden_size),
            nn.Dropout(config.model.dropout),
        )
        self.audio_encoder = PretrainedAudioEncoder(config) if self.use_audio else None

        self.tabular_projector = nn.Sequential(
            nn.Linear(tabular_dim, config.model.tabular_hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.model.tabular_hidden_size),
            nn.Linear(config.model.tabular_hidden_size, config.model.tabular_hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.model.tabular_hidden_size),
            nn.Linear(config.model.tabular_hidden_size, config.model.fusion_hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.model.fusion_hidden_size),
            nn.Dropout(config.model.dropout),
        )

        fusion_dim = config.model.fusion_hidden_size
        self.modality_gating = (
            nn.Sequential(
                nn.Linear(fusion_dim * len(self.modality_names), fusion_dim),
                nn.GELU(),
                nn.LayerNorm(fusion_dim),
                nn.Dropout(config.model.dropout),
                nn.Linear(fusion_dim, fusion_dim // 2),
                nn.GELU(),
                nn.Linear(fusion_dim // 2, len(self.modality_names)),
            )
            if config.model.use_modality_gating
            else None
        )
        self.fusion_blocks = nn.ModuleList(
            [
                CrossModalBlock(
                    hidden_size=fusion_dim,
                    num_heads=config.model.num_attention_heads,
                    ffn_size=config.model.fusion_ffn_size,
                    dropout=config.model.dropout,
                    modality_names=self.modality_names,
                )
                for _ in range(max(int(config.model.fusion_layers), 1))
            ]
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.cls_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=config.model.num_attention_heads,
            dropout=config.model.dropout,
            batch_first=True,
        )
        self.cls_norm = nn.LayerNorm(fusion_dim)

        reg_in_dim = fusion_dim * (len(self.modality_names) + 1)
        self.regressor = nn.Sequential(
            nn.Linear(reg_in_dim, fusion_dim * 2),
            nn.GELU(),
            nn.LayerNorm(fusion_dim * 2),
            nn.Dropout(config.model.dropout + 0.1),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
            nn.Dropout(config.model.dropout),
            nn.Linear(fusion_dim, 1),
        )
        self.aux_regressor = (
            nn.ModuleDict(
                {name: nn.Linear(fusion_dim, 1) for name in self.modality_names}
            )
            if config.model.use_auxiliary_loss
            else None
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
            last_hidden = text_output.last_hidden_state
            text_cls = last_hidden[:, 0, :]
            mask = attention_mask.unsqueeze(-1).float()
            text_mean = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            text_cls = text_cls + text_mean
        else:
            text_cls = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = self.text_projector(text_cls)

        image_feat_map = self.image_encoder(image)
        if image_feat_map.dim() == 4:
            image_se = self.image_se(image_feat_map)
            image_pooled = F.adaptive_avg_pool2d(image_se, (1, 1)).flatten(1)
            image_max = F.adaptive_max_pool2d(image_feat_map, (1, 1)).flatten(1)
            image_encoded = image_pooled + 0.5 * image_max
        else:
            image_encoded = image_feat_map
        image_features = self.image_projector(image_encoded)

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

        bsz = fusion_tokens.size(0)
        cls_token = self.cls_token.expand(bsz, -1, -1)
        cls_input = torch.cat([cls_token, fusion_tokens], dim=1)
        cls_normed = self.cls_norm(cls_input)
        cls_attended, _ = self.cls_attn(
            cls_normed[:, 0:1, :],
            cls_normed[:, 1:, :],
            cls_normed[:, 1:, :],
            need_weights=False,
        )
        cls_out = cls_token + cls_attended

        fused = torch.cat([cls_out.squeeze(1), fusion_tokens.reshape(bsz, -1)], dim=-1)
        preds = self.regressor(fused).squeeze(-1)

        if self.aux_regressor is not None and self.training:
            aux_preds = {}
            for i, name in enumerate(self.modality_names):
                aux_preds[name] = self.aux_regressor[name](fusion_tokens[:, i, :]).squeeze(-1)
            if not return_attention:
                return preds, aux_preds
            return preds, {"gates": gate_weights, "attentions": attentions, "modality_names": self.modality_names, "aux": aux_preds}

        if not return_attention:
            return preds
        return preds, {"gates": gate_weights, "attentions": attentions, "modality_names": self.modality_names}
