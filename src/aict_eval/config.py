from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    text_model_name: str = "bert-base-chinese"
    image_model_name: str = "resnet18"
    text_hidden_size: int = 768
    image_hidden_size: int = 512
    tabular_hidden_size: int = 64
    fusion_hidden_size: int = 256
    num_attention_heads: int = 4
    dropout: float = 0.1
    max_text_length: int = 256
    local_text_vocab_size: int = 4096
    allow_online_model_download: bool = False


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
    text_column: str = "review_text"
    output_dir: str = "outputs"
    device: Optional[str] = None


@dataclass
class ExplainConfig:
    background_size: int = 32
    sample_size: int = 16


@dataclass
class AICTConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    explain: ExplainConfig = field(default_factory=ExplainConfig)
