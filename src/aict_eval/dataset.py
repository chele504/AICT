from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence
import wave

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torchvision import transforms

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

try:
    from transformers import AutoFeatureExtractor
except Exception:
    AutoFeatureExtractor = None

from .config import AICTConfig
from .filters import DenoiseParams, denoise_dataframe


@dataclass
class SplitBundle:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    tabular_columns: List[str]
    scaler: StandardScaler


class HashTokenizer:
    """
    在无法下载预训练分词器时的本地兜底实现。
    """

    def __init__(self, vocab_size: int, max_length: int) -> None:
        self.vocab_size = vocab_size
        self.max_length = max_length

    def __call__(
        self,
        text: str,
        max_length: int,
        padding: str,
        truncation: bool,
        return_tensors: str,
    ) -> dict:
        del padding, truncation, return_tensors
        chars = list(text)[:max_length]
        token_ids = [2 + (hash(ch) % (self.vocab_size - 2)) for ch in chars]
        attention = [1] * len(token_ids)
        pad_len = max(0, max_length - len(token_ids))
        token_ids.extend([0] * pad_len)
        attention.extend([0] * pad_len)
        return {
            "input_ids": torch.tensor([token_ids], dtype=torch.long),
            "attention_mask": torch.tensor([attention], dtype=torch.long),
        }


def build_tokenizer(config: AICTConfig):
    if AutoTokenizer is None:
        return HashTokenizer(
            vocab_size=config.model.local_text_vocab_size,
            max_length=config.model.max_text_length,
        )
    try:
        return AutoTokenizer.from_pretrained(
            config.model.text_model_name,
            local_files_only=not config.model.allow_online_model_download,
        )
    except Exception:
        return HashTokenizer(
            vocab_size=config.model.local_text_vocab_size,
            max_length=config.model.max_text_length,
        )


def build_audio_feature_extractor(config: AICTConfig):
    if not config.train.audio_column:
        return None
    if config.model.audio_backbone_type.lower() == "stats":
        return None
    if AutoFeatureExtractor is None:
        return None
    try:
        return AutoFeatureExtractor.from_pretrained(
            config.model.audio_model_name,
            local_files_only=not config.model.allow_online_model_download,
        )
    except Exception:
        return None


def discover_tabular_columns(
    df: pd.DataFrame,
    target_column: str,
    text_column: str,
    image_column: str,
    audio_column: str | None = None,
) -> List[str]:
    ignored = {target_column, text_column, image_column}
    if audio_column:
        ignored.add(audio_column)
    columns = [
        col for col in df.columns if col not in ignored and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not columns:
        raise ValueError("未发现可用的结构化数值特征列。")
    return columns


def prepare_splits(df: pd.DataFrame, config: AICTConfig) -> SplitBundle:
    required_columns = [
        config.train.target_column,
        config.train.text_column,
        config.train.image_column,
    ]
    if config.train.audio_column:
        required_columns.append(config.train.audio_column)
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            "训练数据缺少必要列: "
            + ", ".join(missing_columns)
            + "。若暂不使用语音模态，请将 train.audio_column 设为 null。"
        )

    train_df, val_df = train_test_split(
        df,
        test_size=config.train.val_ratio,
        random_state=config.train.random_seed,
    )
    tabular_columns = discover_tabular_columns(
        df,
        target_column=config.train.target_column,
        text_column=config.train.text_column,
        image_column=config.train.image_column,
        audio_column=config.train.audio_column,
    )
    if config.train.denoise_enabled:
        params = DenoiseParams(
            method=config.train.denoise_method,
            kalman_process_variance=config.train.kalman_process_variance,
            kalman_measurement_variance=config.train.kalman_measurement_variance,
            ema_alpha=config.train.ema_alpha,
            ema_min_alpha=config.train.ema_min_alpha,
            ema_max_alpha=config.train.ema_max_alpha,
            ema_window=config.train.ema_window,
        )
        train_df = denoise_dataframe(
            train_df,
            tabular_columns,
            params,
            group_column=config.train.denoise_group_column,
            sort_column=config.train.denoise_sort_column,
        )
        val_df = denoise_dataframe(
            val_df,
            tabular_columns,
            params,
            group_column=config.train.denoise_group_column,
            sort_column=config.train.denoise_sort_column,
        )
    scaler = StandardScaler()
    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df[tabular_columns] = scaler.fit_transform(train_df[tabular_columns])
    val_df[tabular_columns] = scaler.transform(val_df[tabular_columns])
    return SplitBundle(train_df=train_df, val_df=val_df, tabular_columns=tabular_columns, scaler=scaler)


class AICTDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        config: AICTConfig,
        tabular_columns: Sequence[str],
        tabular_weights: np.ndarray | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.config = config
        self.tabular_columns = list(tabular_columns)
        self.tabular_weights = (
            np.asarray(tabular_weights, dtype=np.float32)
            if tabular_weights is not None
            else None
        )
        self.tokenizer = build_tokenizer(config)
        self.audio_column = config.train.audio_column
        self.audio_sample_rate = int(config.model.audio_sample_rate)
        self.audio_num_samples = max(
            int(config.model.audio_sample_rate * config.model.audio_duration_seconds),
            1,
        )
        self.audio_backbone_type = config.model.audio_backbone_type.lower()
        self.audio_feature_extractor = build_audio_feature_extractor(config)
        self.use_pretrained_audio = self.audio_feature_extractor is not None
        self.cache_preprocessed_inputs = bool(config.train.cache_preprocessed_inputs)
        self._text_cache: dict[str, dict[str, torch.Tensor]] = {}
        self._image_cache: dict[str, torch.Tensor] = {}
        self._audio_cache: dict[str, dict[str, torch.Tensor]] = {}
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, image_path: str) -> torch.Tensor:
        if self.cache_preprocessed_inputs and image_path in self._image_cache:
            return self._image_cache[image_path].clone()
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"找不到图像文件: {image_path}")
        with Image.open(path) as image:
            image_tensor = self.image_transform(image.convert("RGB"))
        if self.cache_preprocessed_inputs:
            self._image_cache[image_path] = image_tensor
        return image_tensor.clone() if self.cache_preprocessed_inputs else image_tensor

    def _resample_audio(self, samples: np.ndarray, source_rate: int) -> np.ndarray:
        if source_rate == self.audio_sample_rate:
            return samples.astype(np.float32, copy=False)
        if samples.size == 0:
            return np.zeros(self.audio_num_samples, dtype=np.float32)
        target_length = max(int(round(samples.shape[0] * self.audio_sample_rate / source_rate)), 1)
        source_index = np.arange(samples.shape[0], dtype=np.float32)
        target_index = np.linspace(0, samples.shape[0] - 1, num=target_length, dtype=np.float32)
        return np.interp(target_index, source_index, samples).astype(np.float32)

    def _normalize_audio_length(self, samples: np.ndarray) -> np.ndarray:
        if samples.shape[0] >= self.audio_num_samples:
            return samples[: self.audio_num_samples]
        padded = np.zeros(self.audio_num_samples, dtype=np.float32)
        padded[: samples.shape[0]] = samples
        return padded

    def _clone_audio_inputs(self, audio_inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.clone() for key, value in audio_inputs.items()}

    def _build_audio_inputs(self, samples: np.ndarray) -> dict[str, torch.Tensor]:
        normalized = self._normalize_audio_length(samples.astype(np.float32, copy=False))
        waveform = torch.tensor(normalized, dtype=torch.float32)
        if not self.use_pretrained_audio:
            return {
                "waveform": waveform,
                "attention_mask": torch.ones_like(waveform, dtype=torch.long),
            }
        extracted = self.audio_feature_extractor(
            normalized,
            sampling_rate=self.audio_sample_rate,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.audio_num_samples,
            return_attention_mask=True,
        )
        audio_inputs: dict[str, torch.Tensor] = {"waveform": waveform}
        if "input_values" in extracted:
            audio_inputs["input_values"] = extracted["input_values"].squeeze(0).to(torch.float32)
        if "input_features" in extracted:
            audio_inputs["input_features"] = extracted["input_features"].squeeze(0).to(torch.float32)
        if "attention_mask" in extracted:
            audio_inputs["attention_mask"] = extracted["attention_mask"].squeeze(0).to(torch.long)
        elif "input_values" in audio_inputs:
            audio_inputs["attention_mask"] = torch.ones_like(audio_inputs["input_values"], dtype=torch.long)
        elif "input_features" in audio_inputs:
            audio_inputs["attention_mask"] = torch.ones(
                audio_inputs["input_features"].shape[-1],
                dtype=torch.long,
            )
        else:
            audio_inputs["attention_mask"] = torch.ones_like(waveform, dtype=torch.long)
        return audio_inputs

    def _load_audio(self, audio_path: str | None) -> dict[str, torch.Tensor]:
        if not self.audio_column:
            return self._build_audio_inputs(np.zeros(self.audio_num_samples, dtype=np.float32))
        if audio_path is None or not str(audio_path).strip():
            return self._build_audio_inputs(np.zeros(self.audio_num_samples, dtype=np.float32))
        audio_key = str(audio_path)
        if self.cache_preprocessed_inputs and audio_key in self._audio_cache:
            return self._clone_audio_inputs(self._audio_cache[audio_key])

        path = Path(audio_key)
        if not path.exists():
            raise FileNotFoundError(f"找不到音频文件: {audio_key}")
        if path.suffix.lower() != ".wav":
            raise ValueError(f"当前仅支持 WAV 音频文件: {audio_key}")

        with wave.open(str(path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            frames = wav_file.readframes(frame_count)

        if sample_width == 1:
            audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
            audio = (audio - 128.0) / 128.0
        elif sample_width == 2:
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"暂不支持 {sample_width * 8} bit 的 WAV 音频: {audio_path}")

        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        audio = self._resample_audio(audio, sample_rate)
        audio_inputs = self._build_audio_inputs(audio)
        if self.cache_preprocessed_inputs:
            self._audio_cache[audio_key] = audio_inputs
        return self._clone_audio_inputs(audio_inputs) if self.cache_preprocessed_inputs else audio_inputs

    def _encode_text(self, text: str) -> dict[str, torch.Tensor]:
        if self.cache_preprocessed_inputs and text in self._text_cache:
            cached = self._text_cache[text]
            return {
                "input_ids": cached["input_ids"].clone(),
                "attention_mask": cached["attention_mask"].clone(),
            }
        encoded = self.tokenizer(
            text,
            max_length=self.config.model.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        packed = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
        if self.cache_preprocessed_inputs:
            self._text_cache[text] = packed
        return {
            "input_ids": packed["input_ids"].clone() if self.cache_preprocessed_inputs else packed["input_ids"],
            "attention_mask": packed["attention_mask"].clone() if self.cache_preprocessed_inputs else packed["attention_mask"],
        }

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        text = str(row[self.config.train.text_column])
        encoded = self._encode_text(text)
        item = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "image": self._load_image(str(row[self.config.train.image_column])),
            "audio_inputs": self._load_audio(row[self.audio_column] if self.audio_column else None),
            "tabular": torch.tensor(self._build_tabular(row), dtype=torch.float32),
            "target": torch.tensor(float(row[self.config.train.target_column]), dtype=torch.float32),
        }
        return item

    def _build_tabular(self, row: pd.Series) -> np.ndarray:
        vector = row[self.tabular_columns].to_numpy(dtype=np.float32)
        if self.tabular_weights is None:
            return vector
        if vector.shape[0] != self.tabular_weights.shape[0]:
            raise ValueError("tabular_weights 维度与 tabular_columns 不一致。")
        return vector * self.tabular_weights
