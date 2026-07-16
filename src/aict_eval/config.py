from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    text_model_name: str = "bert-base-chinese"
    image_model_name: str = "resnet18"
    text_hidden_size: int = 768
    image_hidden_size: int = 512
    audio_hidden_size: int = 256
    tabular_hidden_size: int = 64
    fusion_hidden_size: int = 256
    num_attention_heads: int = 4
    fusion_layers: int = 2
    fusion_ffn_size: int = 512
    use_modality_gating: bool = True
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
    epochs: int = 5
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
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


@dataclass
class ExplainConfig:
    background_size: int = 32
    sample_size: int = 16


@dataclass
class ReportConfig:
    enabled: bool = True
    top_k_features: int = 12
    write_markdown: bool = True


@dataclass
class AICTConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    explain: ExplainConfig = field(default_factory=ExplainConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
