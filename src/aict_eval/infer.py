from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .dataset import AICTDataset
from .model import MultiModalEvaluator
from .train import build_dataloader, load_config, move_batch_to_device, resolve_device


def load_preprocess_artifacts(model_dir: Path) -> dict:
    artifact_path = model_dir / "preprocess_artifacts.json"
    if not artifact_path.exists():
        raise FileNotFoundError(f"找不到预处理工件文件: {artifact_path}")
    with open(artifact_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_indicator_weights(model_dir: Path, tabular_columns: list[str]) -> np.ndarray:
    weight_path = model_dir / "indicator_weights.json"
    if not weight_path.exists():
        raise FileNotFoundError(f"找不到指标权重文件: {weight_path}")
    with open(weight_path, "r", encoding="utf-8") as file:
        weight_map = json.load(file)
    return np.asarray([float(weight_map[col]) for col in tabular_columns], dtype=np.float32)


def apply_saved_scaler(df: pd.DataFrame, tabular_columns: list[str], artifacts: dict) -> pd.DataFrame:
    scaled = df.copy()
    mean = np.asarray(artifacts["scaler_mean"], dtype=np.float32)
    scale = np.asarray(artifacts["scaler_scale"], dtype=np.float32)
    scale = np.where(scale == 0.0, 1.0, scale)
    scaled[tabular_columns] = (scaled[tabular_columns].to_numpy(dtype=np.float32) - mean) / scale
    return scaled


def validate_and_prepare_columns(df: pd.DataFrame, config, tabular_columns: list[str]) -> pd.DataFrame:
    prepared = df.copy()
    required_columns = [config.train.text_column, config.train.image_column]
    missing_columns = [col for col in required_columns if col not in prepared.columns]
    if missing_columns:
        raise ValueError(f"预测数据缺少必要列: {', '.join(missing_columns)}")
    missing_tabular = [col for col in tabular_columns if col not in prepared.columns]
    if missing_tabular:
        raise ValueError(f"预测数据缺少结构化特征列: {', '.join(missing_tabular)}")
    if config.train.audio_column and config.train.audio_column not in prepared.columns:
        prepared[config.train.audio_column] = ""
    if config.train.target_column not in prepared.columns:
        prepared[config.train.target_column] = 0.0
    return prepared


@torch.no_grad()
def predict(model: MultiModalEvaluator, loader, device: torch.device) -> list[float]:
    model.eval()
    outputs: list[float] = []
    for batch in tqdm(loader, desc="infer", leave=False):
        batch = move_batch_to_device(batch, device)
        preds = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            image=batch["image"],
            audio_inputs=batch["audio_inputs"],
            tabular=batch["tabular"],
        )
        outputs.extend(preds.detach().cpu().numpy().astype(float).tolist())
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="使用已训练的 AI+文旅多模态模型执行预测")
    parser.add_argument("--data", required=True, help="待预测数据 CSV 路径")
    parser.add_argument("--model-dir", default="outputs", help="训练输出目录，默认 outputs")
    parser.add_argument("--config", required=False, help="YAML 配置路径")
    parser.add_argument("--output", required=False, help="预测结果输出 CSV 路径")
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config)
    model_dir = Path(args.model_dir)
    preprocess_artifacts = load_preprocess_artifacts(model_dir)
    tabular_columns = list(preprocess_artifacts["tabular_columns"])
    indicator_weights = load_indicator_weights(model_dir, tabular_columns)

    raw_df = pd.read_csv(args.data)
    prepared_df = validate_and_prepare_columns(raw_df, config, tabular_columns)
    scaled_df = apply_saved_scaler(prepared_df, tabular_columns, preprocess_artifacts)

    dataset = AICTDataset(
        scaled_df,
        config,
        tabular_columns,
        tabular_weights=indicator_weights,
    )
    loader = build_dataloader(dataset, config, shuffle=False)

    model = MultiModalEvaluator(config, tabular_dim=len(tabular_columns)).to(device)
    state_dict = torch.load(model_dir / "multimodal_evaluator.pt", map_location=device)
    model.load_state_dict(state_dict)

    predictions = predict(model, loader, device)
    result_df = raw_df.copy()
    result_df["predicted_score"] = predictions

    output_path = Path(args.output) if args.output else model_dir / "predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"预测完成，结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
