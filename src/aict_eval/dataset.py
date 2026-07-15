from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer

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


def discover_tabular_columns(
    df: pd.DataFrame,
    target_column: str,
    text_column: str,
    image_column: str,
) -> List[str]:
    ignored = {target_column, text_column, image_column}
    columns = [
        col for col in df.columns if col not in ignored and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not columns:
        raise ValueError("未发现可用的结构化数值特征列。")
    return columns


def prepare_splits(df: pd.DataFrame, config: AICTConfig) -> SplitBundle:
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
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"找不到图像文件: {image_path}")
        image = Image.open(path).convert("RGB")
        return self.image_transform(image)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        text = str(row[self.config.train.text_column])
        encoded = self.tokenizer(
            text,
            max_length=self.config.model.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "image": self._load_image(str(row[self.config.train.image_column])),
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
