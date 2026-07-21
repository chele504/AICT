from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import AICTConfig
from .dataset import AICTDataset, prepare_splits
from .explain import export_shap_report, fit_tabular_surrogate
from .model import MultiModalEvaluator
from .report import summarize_attention, write_diagnostic_report
from .weights import combine_gra_cv_weights, estimate_gra_cv_alpha


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


def build_dataloader(dataset, config: AICTConfig, shuffle: bool) -> DataLoader:
    num_workers = max(int(config.train.dataloader_num_workers), 0)
    pin_memory = bool(config.train.dataloader_pin_memory and torch.cuda.is_available())
    persistent_workers = bool(config.train.dataloader_persistent_workers and num_workers > 0)
    loader_kwargs = {
        "batch_size": config.train.batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0 and config.train.dataloader_prefetch_factor:
        loader_kwargs["prefetch_factor"] = int(config.train.dataloader_prefetch_factor)
    return DataLoader(dataset, **loader_kwargs)


def move_batch_to_device(batch, device: torch.device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    return batch


def maybe_freeze_backbones(model: MultiModalEvaluator, config: AICTConfig) -> None:
    if config.train.freeze_text_encoder:
        for param in model.text_encoder.parameters():
            param.requires_grad = False
    if config.train.freeze_image_encoder:
        for param in model.image_encoder.parameters():
            param.requires_grad = False
    if config.train.freeze_audio_encoder and model.audio_encoder is not None:
        backbone = getattr(model.audio_encoder, "backbone", None)
        if backbone is not None:
            for param in backbone.parameters():
                param.requires_grad = False


def train_epoch(model, loader, optimizer, criterion, device, scaler, config: AICTConfig):
    model.train()
    losses = []
    autocast_enabled = bool(config.train.mixed_precision and device.type == "cuda")
    for batch in tqdm(loader, desc="train", leave=False):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if autocast_enabled
            else nullcontext()
        ):
            preds = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                image=batch["image"],
                audio_inputs=batch["audio_inputs"],
                tabular=batch["tabular"],
            )
            loss = criterion(preds, batch["target"])
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if config.train.max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.train.max_grad_norm))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config.train.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.train.max_grad_norm))
            optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader, criterion, device, config: AICTConfig):
    model.eval()
    losses = []
    all_targets = []
    all_preds = []
    autocast_enabled = bool(config.train.mixed_precision and device.type == "cuda")
    for batch in tqdm(loader, desc="eval", leave=False):
        batch = move_batch_to_device(batch, device)
        with (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if autocast_enabled
            else nullcontext()
        ):
            preds = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                image=batch["image"],
                audio_inputs=batch["audio_inputs"],
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
    scaler,
) -> None:
    output_dir = Path(config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "multimodal_evaluator.pt")
    with open(output_dir / "indicator_weights.json", "w", encoding="utf-8") as file:
        json.dump(dict(zip(tabular_columns, weights.tolist())), file, ensure_ascii=False, indent=2)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    with open(output_dir / "preprocess_artifacts.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "tabular_columns": tabular_columns,
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "audio_backbone_type": config.model.audio_backbone_type,
                "audio_model_name": config.model.audio_model_name,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )


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
    indicator_alpha = float(config.train.indicator_weight_alpha)
    if config.train.auto_indicator_weight_alpha:
        indicator_alpha = estimate_gra_cv_alpha(
            splits.train_df[splits.tabular_columns].to_numpy(),
            splits.train_df[config.train.target_column].to_numpy(),
        )
    indicator_weights = combine_gra_cv_weights(
        splits.train_df[splits.tabular_columns].to_numpy(),
        splits.train_df[config.train.target_column].to_numpy(),
        alpha=indicator_alpha,
    )

    train_dataset = AICTDataset(
        splits.train_df,
        config,
        splits.tabular_columns,
        tabular_weights=indicator_weights,
    )
    val_dataset = AICTDataset(
        splits.val_df,
        config,
        splits.tabular_columns,
        tabular_weights=indicator_weights,
    )
    train_loader = build_dataloader(train_dataset, config, shuffle=True)
    val_loader = build_dataloader(val_dataset, config, shuffle=False)

    model = MultiModalEvaluator(config, tabular_dim=len(splits.tabular_columns)).to(device)
    maybe_freeze_backbones(model, config)
    criterion = nn.MSELoss()
    optimizer = AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    scaler = GradScaler("cuda", enabled=bool(config.train.mixed_precision and device.type == "cuda"))

    best_metrics = None
    best_state = None
    epochs_without_improvement = 0
    for epoch in range(config.train.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler, config)
        metrics = evaluate(model, val_loader, criterion, device, config)
        metrics["train_loss"] = train_loss
        metrics["epoch"] = epoch + 1
        print(f"epoch={epoch + 1} metrics={metrics}")
        improved = (
            best_metrics is None
            or metrics["rmse"] < best_metrics["rmse"] - float(config.train.early_stopping_min_delta)
        )
        if improved:
            best_metrics = metrics
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if (
                config.train.early_stopping_patience > 0
                and epochs_without_improvement >= int(config.train.early_stopping_patience)
            ):
                print(f"early_stopping at epoch={epoch + 1}")
                break

    if best_state is None or best_metrics is None:
        raise RuntimeError("训练失败，未产生有效模型。")

    model.load_state_dict(best_state)
    save_training_artifacts(
        model,
        config,
        indicator_weights,
        splits.tabular_columns,
        best_metrics,
        splits.scaler,
    )

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
    if config.report.enabled:
        attention_summary = summarize_attention(
            model,
            val_loader,
            device,
            max_batches=config.report.attention_summary_max_batches,
        )
        write_diagnostic_report(
            config=config,
            metrics=best_metrics,
            tabular_columns=splits.tabular_columns,
            indicator_weights=indicator_weights,
            indicator_alpha=indicator_alpha,
            attention_summary=attention_summary,
            shap_path=Path(config.train.output_dir) / "shap_feature_importance.csv",
        )


if __name__ == "__main__":
    main()
