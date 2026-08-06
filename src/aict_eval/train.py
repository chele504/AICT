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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
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


def build_param_groups(model: MultiModalEvaluator, config: AICTConfig):
    base_lr = float(config.train.learning_rate)
    backbone_lr = float(config.train.backbone_learning_rate)
    weight_decay = float(config.train.weight_decay)
    no_decay = {"bias", "norm", "LayerNorm", "layer_norm", "ln_"}

    def is_backbone(name: str) -> bool:
        backbone_prefixes = (
            "text_encoder.",
            "image_encoder.",
            "audio_encoder.backbone.",
        )
        return name.startswith(backbone_prefixes) and "projector" not in name and "se" not in name

    def get_decay_and_lr(name: str, param):
        use_wd = not any(nd in name for nd in no_decay) and param.ndim >= 2
        lr = backbone_lr if is_backbone(name) else base_lr
        return float(weight_decay) if use_wd else 0.0, lr

    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        wd, lr = get_decay_and_lr(name, param)
        key = (round(wd, 10), round(lr, 15))
        if key not in groups:
            groups[key] = {"params": [], "weight_decay": wd, "lr": lr}
        groups[key]["params"].append(param)
    return list(groups.values())


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


def build_criterion(config: AICTConfig):
    loss_name = str(config.train.loss_type).lower()
    delta = float(config.train.huber_delta)
    if loss_name == "huber":
        base = nn.HuberLoss(delta=delta)
    elif loss_name == "mae" or loss_name == "l1":
        base = nn.L1Loss()
    else:
        base = nn.MSELoss()
    label_smoothing = float(config.train.label_smoothing)
    aux_weight = float(config.model.auxiliary_loss_weight) if config.model.use_auxiliary_loss else 0.0

    def criterion(preds, target, aux_preds=None):
        if label_smoothing > 0:
            target_mean = target.mean()
            smoothed = target * (1.0 - label_smoothing) + target_mean * label_smoothing
        else:
            smoothed = target
        main_loss = base(preds, smoothed)
        if aux_preds is None or aux_weight <= 0:
            return main_loss
        aux_loss = 0.0
        count = 0
        for name, aux_out in aux_preds.items():
            aux_loss = aux_loss + base(aux_out, smoothed)
            count += 1
        if count > 0:
            aux_loss = aux_loss / count
        return (1.0 - aux_weight) * main_loss + aux_weight * aux_loss

    return criterion


def build_lr_scheduler(optimizer, config: AICTConfig, total_steps_per_epoch: int):
    total_epochs = int(config.train.epochs)
    grad_accum = max(int(config.train.gradient_accumulation_steps), 1)
    total_steps = total_epochs * total_steps_per_epoch // grad_accum
    scheduler_type = str(config.train.lr_scheduler_type).lower()

    if config.train.warmup_ratio is not None:
        warmup_steps = max(1, int(total_steps * float(config.train.warmup_ratio)))
    else:
        warmup_steps = max(1, int(config.train.warmup_epochs) * total_steps_per_epoch // grad_accum)

    min_lr = float(config.train.learning_rate) * float(config.train.min_lr_ratio)

    if scheduler_type in {"cosine", "cosine_with_warmup"}:
        cosine_steps = max(total_steps - warmup_steps, 1)
        cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=min_lr)
        if warmup_steps <= 0:
            return cosine, None
        warmup = LinearLR(
            optimizer,
            start_factor=1e-4 / max(float(config.train.learning_rate), 1e-8),
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        milestones = [warmup_steps]
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=milestones)
        return scheduler, warmup_steps
    if scheduler_type == "linear":
        total = max(total_steps, 1)
        linear = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=float(config.train.min_lr_ratio),
            total_iters=total,
        )
        return linear, None
    return None, None


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler,
    config: AICTConfig,
    scheduler=None,
    scheduler_step_each_batch: bool = True,
    epoch: int = 0,
):
    model.train()
    losses = []
    autocast_enabled = bool(config.train.mixed_precision and device.type == "cuda")
    grad_accum = max(int(config.train.gradient_accumulation_steps), 1)
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc=f"train epoch={epoch + 1}", leave=False)
    for step_idx, batch in enumerate(pbar):
        batch = move_batch_to_device(batch, device)
        do_backward = (step_idx + 1) % grad_accum == 0 or step_idx + 1 == len(loader)
        with (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if autocast_enabled
            else nullcontext()
        ):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                image=batch["image"],
                audio_inputs=batch["audio_inputs"],
                tabular=batch["tabular"],
            )
            if isinstance(outputs, tuple):
                preds, aux_preds = outputs
            else:
                preds = outputs
                aux_preds = None
            loss = criterion(preds, batch["target"], aux_preds)
            loss_to_backward = loss / grad_accum
        if scaler.is_enabled():
            scaler.scale(loss_to_backward).backward()
            if do_backward:
                if config.train.max_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.train.max_grad_norm))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None and scheduler_step_each_batch:
                    scheduler.step()
        else:
            loss_to_backward.backward()
            if do_backward:
                if config.train.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.train.max_grad_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None and scheduler_step_each_batch:
                    scheduler.step()
        losses.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")
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
            if isinstance(preds, tuple):
                preds = preds[0]
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
        is_train=True,
    )
    val_dataset = AICTDataset(
        splits.val_df,
        config,
        splits.tabular_columns,
        tabular_weights=indicator_weights,
        is_train=False,
    )
    train_loader = build_dataloader(train_dataset, config, shuffle=True)
    val_loader = build_dataloader(val_dataset, config, shuffle=False)

    model = MultiModalEvaluator(config, tabular_dim=len(splits.tabular_columns)).to(device)
    maybe_freeze_backbones(model, config)
    criterion = build_criterion(config)
    param_groups = build_param_groups(model, config)
    optimizer = AdamW(param_groups)
    scaler = GradScaler("cuda", enabled=bool(config.train.mixed_precision and device.type == "cuda"))

    steps_per_epoch = len(train_loader)
    scheduler, warmup_steps = build_lr_scheduler(optimizer, config, steps_per_epoch)

    best_metrics = None
    best_state = None
    epochs_without_improvement = 0
    for epoch in range(config.train.epochs):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            config,
            scheduler=scheduler,
            scheduler_step_each_batch=True,
            epoch=epoch,
        )
        metrics = evaluate(model, val_loader, criterion, device, config)
        metrics["train_loss"] = train_loss
        metrics["epoch"] = epoch + 1
        lr_current = float(optimizer.param_groups[0]["lr"])
        metrics["lr"] = lr_current
        print(f"epoch={epoch + 1} lr={lr_current:.2e} metrics={metrics}")
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
