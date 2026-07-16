from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import AICTConfig
from .model import MultiModalEvaluator


def _group_dimension_weights(tabular_columns: list[str], weights: np.ndarray) -> dict[str, float]:
    names = list(tabular_columns)
    values = np.asarray(weights, dtype=float).reshape(-1)
    if len(names) != values.shape[0]:
        return {}

    def in_group(col: str, keywords: list[str]) -> bool:
        c = col.lower()
        return any(k.lower() in c for k in keywords)

    groups = {
        "技术赋能效能": ["tech", "技术", "empower"],
        "游客感知体验": ["experience", "体验", "visitor", "engagement"],
        "文化价值传播": ["cultural", "文化", "heritage", "value"],
        "经济社会增值": ["economic", "econ", "社会", "增值", "gain", "social"],
    }

    sums: dict[str, float] = {k: 0.0 for k in groups}
    for col, w in zip(names, values):
        matched = False
        for group_name, keywords in groups.items():
            if in_group(col, keywords):
                sums[group_name] += float(w)
                matched = True
                break
        if not matched:
            sums.setdefault("其他", 0.0)
            sums["其他"] += float(w)

    total = float(sum(sums.values()))
    if total <= 0:
        return {}
    return {k: float(v / total) for k, v in sums.items()}


@torch.no_grad()
def summarize_attention(
    model: MultiModalEvaluator,
    loader,
    device: torch.device,
) -> dict:
    model.eval()
    modality_names = list(model.modality_names)
    gate_sum = None
    gate_count = 0
    attn_sums: list[dict[str, np.ndarray]] = []
    attn_count = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        preds, info = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            image=batch["image"],
            audio=batch["audio"],
            tabular=batch["tabular"],
            return_attention=True,
        )
        del preds
        batch_size = int(batch["input_ids"].size(0))
        gates = info.get("gates")
        if gates is not None:
            g = gates.detach().cpu().numpy()
            gate_sum = g.sum(axis=0) if gate_sum is None else gate_sum + g.sum(axis=0)
            gate_count += batch_size

        attentions = info.get("attentions") or []
        if not attn_sums:
            attn_sums = [
                {name: np.zeros(len(modality_names), dtype=float) for name in modality_names}
                for _ in attentions
            ]

        for layer_idx, layer_attn in enumerate(attentions):
            if layer_attn is None:
                continue
            for key in modality_names:
                raw = layer_attn[key]
                w = raw.mean(dim=(1, 2)).detach().cpu().numpy()
                attn_sums[layer_idx][key] += w.sum(axis=0)
        attn_count += batch_size

    out: dict = {}
    if gate_sum is not None and gate_count > 0:
        out["modality_gates_mean"] = {
            name: float(gate_sum[idx] / gate_count) for idx, name in enumerate(modality_names)
        }

    if attn_sums and attn_count > 0:
        layers = []
        for layer in attn_sums:
            query_summary = {}
            for query_name in modality_names:
                query_summary[f"{query_name}_query"] = {
                    f"to_{target_name}": float(layer[query_name][target_idx] / attn_count)
                    for target_idx, target_name in enumerate(modality_names)
                }
            layers.append(query_summary)
        out["cross_attention_mean"] = layers
    return out


def write_diagnostic_report(
    config: AICTConfig,
    metrics: dict,
    tabular_columns: list[str],
    indicator_weights: np.ndarray,
    indicator_alpha: float,
    attention_summary: dict,
    shap_path: Path,
) -> dict:
    output_dir = Path(config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shap_top = []
    if shap_path.exists():
        shap_df = pd.read_csv(shap_path)
        shap_df = shap_df.sort_values("mean_abs_shap", ascending=False)
        shap_top = (
            shap_df.head(int(config.report.top_k_features))
            .to_dict(orient="records")
        )

    indicator_map = dict(zip(tabular_columns, np.asarray(indicator_weights, dtype=float).tolist()))
    report = {
        "metrics": metrics,
        "indicator_weight_alpha": float(indicator_alpha),
        "indicator_weights": indicator_map,
        "dimension_weight_share": _group_dimension_weights(tabular_columns, indicator_weights),
        "top_features_by_shap": shap_top,
        "fusion_attention": attention_summary,
        "denoise": {
            "enabled": bool(config.train.denoise_enabled),
            "method": config.train.denoise_method,
            "group_column": config.train.denoise_group_column,
            "sort_column": config.train.denoise_sort_column,
        },
    }

    json_path = output_dir / "report.json"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    if config.report.write_markdown:
        md_path = output_dir / "report.md"
        lines = []
        lines.append("# AI+文旅成效诊断报告")
        lines.append("")
        lines.append("## 模型效果")
        for k in ("loss", "mae", "rmse", "r2", "train_loss", "epoch"):
            if k in metrics:
                lines.append(f"- {k}: {metrics[k]}")
        lines.append("")
        lines.append("## 指标权重（GRA+CV）")
        lines.append(f"- alpha: {float(indicator_alpha)}")
        dim_share = report.get("dimension_weight_share") or {}
        if dim_share:
            lines.append("")
            lines.append("### 一级指标权重占比")
            for k, v in dim_share.items():
                lines.append(f"- {k}: {v:.4f}")
        lines.append("")
        lines.append("### 结构化指标权重")
        for name, w in sorted(indicator_map.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- {name}: {w:.6f}")
        if shap_top:
            lines.append("")
            lines.append("## 关键驱动因素（SHAP 代理模型）")
            for row in shap_top:
                lines.append(f"- {row['feature']}: {float(row['mean_abs_shap']):.6f}")
        if attention_summary:
            lines.append("")
            lines.append("## 跨模态注意力概览")
            gates = attention_summary.get("modality_gates_mean")
            if gates:
                lines.append("- 动态模态权重（均值）")
                gate_text = " ".join(f"{name}={value:.4f}" for name, value in gates.items())
                lines.append(f"  - {gate_text}")
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return report
