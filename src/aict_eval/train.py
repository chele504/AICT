from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import AICTConfig
from .dataset import AICTDataset, prepare_splits
from .explain import export_shap_report, fit_tabular_surrogate
from .model import MultiModalEvaluator
from .weights import combine_gra_cv_weights


def load_config(config_path: str | None) -> AICTConfig:
    if not config_path:
        return AICTConfig()
    with open(config_path, "r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    config = AICTConfig()
    for section_name, section_values in raw.items():
        section = getattr(config, section_name)
        for key, value in section_values.items():
            setattr(section, key, value)
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(config: AICTConfig) -> torch.device:
    if config.train.device:
        return torch.device(config.train.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses = []
    for batch in tqdm(loader, desc="train", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        preds = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            image=batch["image"],
            tabular=batch["tabular"],
        )
        loss = criterion(preds, batch["target"])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    all_targets = []
    all_preds = []
    for batch in tqdm(loader, desc="eval", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        preds = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            image=batch["image"],
            tabular=batch["tabular"],
        )
        loss = criterion(preds, batch["target"])
        losses.append(loss.item())
        all_targets.extend(batch["target"].cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    metrics = {
        "loss": float(np.mean(losses)),
        "mae": float(mean_absolute_error(all_targets, all_preds)),
        "rmse": float(np.sqrt(mean_squared_error(all_targets, all_preds))),
        "r2": float(r2_score(all_targets, all_preds)),
    }
    return metrics


def save_training_artifacts(
    model: MultiModalEvaluator,
    config: AICTConfig,
    weights: np.ndarray,
    tabular_columns: list[str],
    metrics: dict,
) -> None:
    output_dir = Path(config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "multimodal_evaluator.pt")
    with open(output_dir / "indicator_weights.json", "w", encoding="utf-8") as file:
        json.dump(dict(zip(tabular_columns, weights.tolist())), file, ensure_ascii=False, indent=2)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 AI+文旅多模态成效评价模型")
    parser.add_argument("--data", required=True, help="训练数据 CSV 路径")
    parser.add_argument("--config", required=False, help="YAML 配置路径")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.train.random_seed)
    device = resolve_device(config)

    data_frame = pd.read_csv(args.data)
    splits = prepare_splits(data_frame, config)
    indicator_weights = combine_gra_cv_weights(
        splits.train_df[splits.tabular_columns].to_numpy(),
        splits.train_df[config.train.target_column].to_numpy(),
    )

    train_dataset = AICTDataset(splits.train_df, config, splits.tabular_columns)
    val_dataset = AICTDataset(splits.val_df, config, splits.tabular_columns)
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False)

    model = MultiModalEvaluator(config, tabular_dim=len(splits.tabular_columns)).to(device)
    criterion = nn.MSELoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )

    best_metrics = None
    best_state = None
    for epoch in range(config.train.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, val_loader, criterion, device)
        metrics["train_loss"] = train_loss
        metrics["epoch"] = epoch + 1
        print(f"epoch={epoch + 1} metrics={metrics}")
        if best_metrics is None or metrics["rmse"] < best_metrics["rmse"]:
            best_metrics = metrics
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None or best_metrics is None:
        raise RuntimeError("训练失败，未产生有效模型。")

    model.load_state_dict(best_state)
    save_training_artifacts(model, config, indicator_weights, splits.tabular_columns, best_metrics)

    surrogate = fit_tabular_surrogate(
        splits.train_df,
        splits.tabular_columns,
        config.train.target_column,
    )
    sample_frame = splits.val_df.head(config.explain.sample_size)
    export_shap_report(
        surrogate,
        splits.train_df.head(max(config.explain.background_size, 64)),
        sample_frame,
        splits.tabular_columns,
        str(Path(config.train.output_dir) / "shap_feature_importance.csv"),
    )


if __name__ == "__main__":
    main()
