from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    text_model_name: str = "bert-base-chinese"
    image_model_name: str = "resnet18"
    audio_backbone_type: str = "stats"
    audio_model_name: str = "facebook/wav2vec2-base-960h"
    text_hidden_size: int = 768
    image_hidden_size: int = 512
    audio_hidden_size: int = 256
    tabular_hidden_size: int = 128
    fusion_hidden_size: int = 256
    num_attention_heads: int = 4
    fusion_layers: int = 3
    fusion_ffn_size: int = 768
    use_modality_gating: bool = True
    use_auxiliary_loss: bool = True
    auxiliary_loss_weight: float = 0.15
    dropout: float = 0.1
    max_text_length: int = 256
    local_text_vocab_size: int = 4096
    allow_online_model_download: bool = False
    audio_sample_rate: int = 16000
    audio_duration_seconds: float = 2.0
    audio_n_fft: int = 400
    audio_hop_length: int = 160


@dataclass
class TrainConfig:
    batch_size: int = 8
    epochs: int = 10
    learning_rate: float = 2e-4
    backbone_learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    max_grad_norm: Optional[float] = 1.0
    val_ratio: float = 0.2
    random_seed: int = 42
    target_column: str = "target_score"
    image_column: str = "image_path"
    audio_column: Optional[str] = None
    text_column: str = "review_text"
    output_dir: str = "outputs"
    device: Optional[str] = None
    indicator_weight_alpha: float = 0.5
    auto_indicator_weight_alpha: bool = True
    scene_column: Optional[str] = None
    scene_alpha_map: dict[str, float] = field(default_factory=dict)
    denoise_enabled: bool = False
    denoise_method: str = "kalman"
    denoise_group_column: Optional[str] = None
    denoise_sort_column: Optional[str] = None
    kalman_process_variance: float = 1e-4
    kalman_measurement_variance: float = 1e-2
    ema_alpha: float = 0.25
    ema_min_alpha: float = 0.05
    ema_max_alpha: float = 0.6
    ema_window: int = 5
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True
    dataloader_prefetch_factor: Optional[int] = 2
    cache_preprocessed_inputs: bool = True
    cache_max_size: int = 10000
    mixed_precision: bool = True
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4
    freeze_text_encoder: bool = False
    freeze_image_encoder: bool = False
    freeze_audio_encoder: bool = False
    gradient_accumulation_steps: int = 1
    lr_scheduler_type: str = "cosine_with_warmup"
    warmup_epochs: int = 2
    warmup_ratio: Optional[float] = None
    min_lr_ratio: float = 0.05
    label_smoothing: float = 0.0
    loss_type: str = "huber"
    huber_delta: float = 0.5
    enable_augmentation: bool = True


@dataclass
class ExplainConfig:
    background_size: int = 32
    sample_size: int = 16


@dataclass
class ReportConfig:
    enabled: bool = True
    top_k_features: int = 12
    write_markdown: bool = True
    attention_summary_max_batches: int = 8


@dataclass
class AICTConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    explain: ExplainConfig = field(default_factory=ExplainConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
