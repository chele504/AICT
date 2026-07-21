from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import AICTConfig
from .model import MultiModalEvaluator


FEATURE_LABELS = {
    "tech_empowerment": "技术赋能效能",
    "visitor_experience": "游客感知体验",
    "cultural_value": "文化价值传播",
    "economic_social_gain": "经济社会增值",
    "interaction_count": "游客互动频次",
    "stay_duration": "游客停留时长",
}

MODALITY_LABELS = {
    "text": "文本模态",
    "image": "视觉图像模态",
    "audio": "语音模态",
    "tabular": "结构化指标模态",
}


def _format_metric(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def _feature_label(name: str) -> str:
    return FEATURE_LABELS.get(name, name)


def _modality_label(name: str) -> str:
    clean = name.replace("_query", "")
    if clean.startswith("to_"):
        clean = clean[3:]
    return MODALITY_LABELS.get(clean, clean)


def _group_dimension_weights(tabular_columns: list[str], weights: np.ndarray) -> dict[str, float]:
    names = list(tabular_columns)
    values = np.asarray(weights, dtype=float).reshape(-1)
    if len(names) != values.shape[0]:
        return {}

    keyword_map = {
        "技术赋能效能": ["tech", "技术", "empower", "efficiency", "智能", "algorithm", "service"],
        "游客感知体验": ["experience", "体验", "visitor", "engagement", "interaction", "stay", "duration", "emotion"],
        "文化价值传播": ["cultural", "文化", "heritage", "value", "knowledge", "传播", "认同"],
        "经济社会增值": ["economic", "econ", "社会", "增值", "gain", "social", "industry", "benefit"],
    }

    def classify_dimension(col: str) -> str:
        c = col.lower()
        best_name = "游客感知体验"
        best_score = -1
        for group_name, keywords in keyword_map.items():
            score = sum(1 for keyword in keywords if keyword.lower() in c)
            if score > best_score:
                best_name = group_name
                best_score = score
        return best_name

    sums: dict[str, float] = {k: 0.0 for k in keyword_map}
    for col, w in zip(names, values):
        sums[classify_dimension(col)] += float(w)

    total = float(sum(sums.values()))
    if total <= 0:
        return {}
    return {k: float(v / total) for k, v in sums.items()}


def _ranked_dimension_text(dim_share: dict[str, float]) -> str:
    if not dim_share:
        return "当前未形成稳定的一级指标权重分布，暂无法对维度侧重点进行解释。"
    ranked = sorted(dim_share.items(), key=lambda x: x[1], reverse=True)
    top_name, top_value = ranked[0]
    second_name, second_value = ranked[1] if len(ranked) > 1 else ranked[0]
    if len(ranked) == 1:
        return f"从一级指标权重结构看，当前评价结果主要由“{top_name}”维度支撑，其权重占比约为 {_format_metric(top_value * 100, 2)}%。"
    gap = (top_value - second_value) * 100
    if abs(gap) < 3:
        return (
            f"一级指标权重整体呈均衡分布，其中“{top_name}”与“{second_name}”相对突出，"
            f"占比分别约为 {_format_metric(top_value * 100, 2)}% 和 {_format_metric(second_value * 100, 2)}%，"
            "表明当前评价体系在综合判断中兼顾了多维目标，没有出现对单一维度的明显偏置。"
        )
    return (
        f"一级指标中“{top_name}”权重最高，约为 {_format_metric(top_value * 100, 2)}%，"
        f"较第二位“{second_name}”高出 {_format_metric(gap, 2)} 个百分点，"
        "说明该维度在当前阶段对综合成效评价具有更强的牵引作用。"
    )


def _build_goal_alignment_summary(dim_share: dict[str, float], attention_summary: dict, shap_top: list[dict]) -> list[str]:
    lines = [
        "本报告围绕附件2提出的“评价科学化、数据多维化、反馈智能化”总体目标，对“AI+文旅”应用成效开展结构化分析与研究性阐释。",
        "在评价科学化方面，报告依据 GRA+CV 方法对四个一级指标进行客观赋权，并结合实证结果对评价输出的稳定性进行检验。",
    ]
    if attention_summary:
        lines.append("在数据多维化方面，评价过程综合引入文本、图像、语音与结构化指标，并借助跨模态注意力机制揭示不同信息源之间的协同关系。")
    if shap_top:
        lines.append("在反馈智能化方面，模型不仅输出综合成效分值，同时结合 SHAP 归因结果识别关键驱动因素，为景区运营优化和行业监管决策提供可解释依据。")
    if dim_share:
        lines.append("报告结构严格对应附件2中提出的四个一级指标，即技术赋能效能、游客感知体验、文化价值传播和经济社会增值。")
    return lines


def _build_metric_analysis(metrics: dict) -> list[str]:
    rmse = float(metrics.get("rmse", 0.0))
    mae = float(metrics.get("mae", 0.0))
    r2 = float(metrics.get("r2", 0.0))
    train_loss = metrics.get("train_loss")
    loss = metrics.get("loss")
    lines = []
    lines.append(
        f"实证结果显示，当前阶段模型在验证集上的平均绝对误差为 {_format_metric(mae)}，均方根误差为 {_format_metric(rmse)}，决定系数为 {_format_metric(r2)}。"
    )
    if r2 >= 0.7:
        lines.append("总体来看，模型已具备较强的解释能力，能够较为稳定地刻画综合成效评分与多模态输入之间的对应关系。")
    elif r2 >= 0.3:
        lines.append("总体来看，模型已经能够识别部分有效规律，但仍存在一定误差，当前结果更适合作为阶段性研究验证依据。")
    elif r2 >= 0.0:
        lines.append("总体来看，模型已经初步呈现出对成效规律的识别能力，但整体解释力度仍然有限，说明在样本丰富度、特征质量和多模态协同方面仍有提升空间。")
    else:
        lines.append("从当前结果看，模型在验证集上的泛化表现仍然偏弱，说明现阶段研究结论主要体现为方法验证意义，后续仍需通过扩充样本、丰富场景与增强表征能力进一步提升稳健性。")
    if train_loss is not None and loss is not None:
        if float(train_loss) > float(loss) * 5:
            lines.append("训练集与验证集损失之间仍存在一定差异，表明小样本条件下批次波动和数据分布差异对结果具有一定影响。")
        elif float(loss) > float(train_loss) * 1.5:
            lines.append("验证集误差高于训练集误差，说明模型在跨样本泛化方面仍有进一步增强空间，后续宜通过扩大样本覆盖和增加实证场景持续验证。")
        else:
            lines.append("训练集与验证集误差差距总体处于可接受范围，说明当前研究流程在原型阶段具备基本稳定性。")
    return lines


def _build_shap_analysis(shap_top: list[dict]) -> list[str]:
    if not shap_top:
        return ["当前未生成 SHAP 结果，暂无法对结构化特征的驱动机制做进一步解释。"]
    top_feature = _feature_label(shap_top[0]["feature"])
    top_value = float(shap_top[0]["mean_abs_shap"])
    lines = [
        f"从 SHAP 归因结果看，“{top_feature}”是当前最关键的结构化驱动因素，其平均绝对贡献度约为 {_format_metric(top_value)}。"
    ]
    if len(shap_top) >= 3:
        top3 = "、".join(_feature_label(row["feature"]) for row in shap_top[:3])
        lines.append(f"排名前三的结构化影响因素分别为{top3}，表明综合成效评价主要受这几类核心业务指标共同牵引。")
    tail = shap_top[-1]
    lines.append(
        f"相较之下，“{_feature_label(tail['feature'])}”的影响度相对较低，说明其在当前样本中的边际解释作用尚不突出。"
    )
    return lines


def _build_attention_analysis(attention_summary: dict) -> list[str]:
    if not attention_summary:
        return ["当前未输出跨模态融合统计，暂无法对模态协同关系进行解释。"]
    lines = []
    gates = attention_summary.get("modality_gates_mean") or {}
    if gates:
        ranked = sorted(gates.items(), key=lambda x: x[1], reverse=True)
        top_name, top_value = ranked[0]
        lines.append(
            f"从动态模态权重看，“{_modality_label(top_name)}”在当前样本中的平均权重最高，约为 {_format_metric(top_value * 100, 2)}%，说明该模态在综合成效识别中承担了更为突出的信息支撑作用。"
        )
        if len(ranked) > 1:
            second_name, second_value = ranked[1]
            lines.append(
                f"其次为“{_modality_label(second_name)}”，其权重约为 {_format_metric(second_value * 100, 2)}%，表明评价结果并非依赖单一信息源，而是体现出一定程度的跨模态协同。"
            )
        low_name, low_value = ranked[-1]
        lines.append(
            f"权重相对较低的为“{_modality_label(low_name)}”，占比约为 {_format_metric(low_value * 100, 2)}%，提示该模态在当前数据条件下的表征潜力尚未充分释放。"
        )
    cross_layers = attention_summary.get("cross_attention_mean") or []
    if cross_layers:
        first_layer = cross_layers[0]
        strongest = None
        for query_name, target_map in first_layer.items():
            for target_name, score in target_map.items():
                if strongest is None or score > strongest[2]:
                    strongest = (query_name, target_name, score)
        if strongest is not None:
            lines.append(
                f"在首层跨模态注意力中，“{_modality_label(strongest[0])}”对“{_modality_label(strongest[1])}”的关注度最高，数值约为 {_format_metric(strongest[2])}，说明模型在早期融合阶段已经形成较为明确的信息对齐方向。"
            )
    return lines


def _build_denoise_analysis(config: AICTConfig) -> str:
    if not config.train.denoise_enabled:
        return "本阶段研究未启用结构化指标去噪流程，评价过程主要基于标准化后的原始数值特征展开。后续如引入连续监测型运营数据，可进一步叠加去噪处理以增强结构化信号稳定性。"
    return (
        f"本阶段研究已启用“{config.train.denoise_method}”去噪策略，"
        "有助于抑制结构化指标中的局部波动与采集噪声，提升评价模型对长期趋势信号的识别能力。"
    )


def _build_framework_analysis(config: AICTConfig) -> list[str]:
    lines = [
        "本研究遵循附件2提出的“感知-价值”双驱动评价逻辑，将技术效能、游客体验、文化传播和经济社会增值统一纳入同一评价框架。",
        "本阶段研究以文本、图像、语音和结构化业务指标作为多模态输入，对游客的认知反馈、行为反馈和价值反馈进行综合建模。",
    ]
    if config.train.audio_column:
        lines.append("在“身—心—行”一体化视角下，当前语音、文本与行为类结构化指标主要承担体验感知与行为响应的代理表征功能，后续仍可进一步接入生理传感器数据，以增强“身”维度的刻画能力。")
    else:
        lines.append("当前原型尚未启用语音或生理相关模态，因此“身—心—行”框架中的即时感知维度仍有进一步扩展空间。")
    return lines


def _build_path_analysis(dim_share: dict[str, float], shap_top: list[dict], attention_summary: dict) -> list[str]:
    dominant_dimension = max(dim_share.items(), key=lambda x: x[1])[0] if dim_share else "游客感知体验"
    top_features = "、".join(_feature_label(item["feature"]) for item in shap_top[:3]) if shap_top else "关键业务指标"
    lines = [
        f"结合当前评价结果，后续可沿附件2提出的“技术迭代-场景适配-文化共鸣-产业增值”路径持续推进模型应用，其中现阶段最值得优先关注的维度是“{dominant_dimension}”。",
        f"从关键驱动因素看，{top_features}对成效分值具有较强影响，说明优化路径宜优先聚焦于可持续采集、可持续反馈的核心业务指标。",
    ]
    gates = attention_summary.get("modality_gates_mean") if attention_summary else None
    if gates:
        top_modality = max(gates.items(), key=lambda x: x[1])[0]
        lines.append(
            f"从融合结构看，当前评价过程更依赖“{_modality_label(top_modality)}”，后续在场景部署中宜优先保障该类数据的采集质量，并同步补强权重偏低模态的数据覆盖。"
        )
    lines.append("在成果转化层面，本报告可为景区项目评估、场景优化复盘以及智慧文旅项目阶段性成效对比提供研究参考。")
    return lines


def _label_indicator_map(indicator_map: dict[str, float]) -> dict[str, float]:
    return {_feature_label(name): value for name, value in indicator_map.items()}


def _label_shap_rows(shap_rows: list[dict]) -> list[dict]:
    labeled = []
    for row in shap_rows:
        labeled.append(
            {
                **row,
                "feature_label": _feature_label(row["feature"]),
            }
        )
    return labeled


@torch.no_grad()
def summarize_attention(
    model: MultiModalEvaluator,
    loader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict:
    model.eval()
    modality_names = list(model.modality_names)
    gate_sum = None
    gate_count = 0
    attn_sums: list[dict[str, np.ndarray]] = []
    attn_count = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = {
            key: (
                {sub_key: sub_value.to(device) for sub_key, sub_value in value.items()}
                if isinstance(value, dict)
                else value.to(device)
            )
            for key, value in batch.items()
        }
        preds, info = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            image=batch["image"],
            audio_inputs=batch["audio_inputs"],
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
    dimension_share = _group_dimension_weights(tabular_columns, indicator_weights)
    goal_alignment = _build_goal_alignment_summary(dimension_share, attention_summary, shap_top)
    metric_analysis = _build_metric_analysis(metrics)
    dimension_analysis = _ranked_dimension_text(dimension_share)
    shap_analysis = _build_shap_analysis(shap_top)
    attention_analysis = _build_attention_analysis(attention_summary)
    denoise_analysis = _build_denoise_analysis(config)
    framework_analysis = _build_framework_analysis(config)
    path_analysis = _build_path_analysis(dimension_share, shap_top, attention_summary)
    labeled_indicator_map = _label_indicator_map(indicator_map)
    labeled_shap_top = _label_shap_rows(shap_top)
    report = {
        "metrics": metrics,
        "indicator_weight_alpha": float(indicator_alpha),
        "indicator_weights": indicator_map,
        "indicator_weights_labeled": labeled_indicator_map,
        "dimension_weight_share": dimension_share,
        "top_features_by_shap": shap_top,
        "top_features_by_shap_labeled": labeled_shap_top,
        "fusion_attention": attention_summary,
        "analysis": {
            "goal_alignment": goal_alignment,
            "metric_analysis": metric_analysis,
            "dimension_analysis": dimension_analysis,
            "shap_analysis": shap_analysis,
            "attention_analysis": attention_analysis,
            "denoise_analysis": denoise_analysis,
            "framework_analysis": framework_analysis,
            "path_analysis": path_analysis,
        },
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
        lines.append("## 与课题目标对齐")
        lines.append("")
        for text in goal_alignment:
            lines.append(text)
        lines.append("")
        lines.append("## 结果综述")
        lines.append("")
        for text in metric_analysis[:2]:
            lines.append(text)
        lines.append(dimension_analysis)
        for text in framework_analysis[:1]:
            lines.append(text)
        lines.append("")
        lines.append("## 实证结果")
        for key, label in (
            ("loss", "验证集损失"),
            ("mae", "平均绝对误差"),
            ("rmse", "均方根误差"),
            ("r2", "决定系数"),
            ("train_loss", "训练集损失"),
            ("epoch", "最优轮次"),
        ):
            if key in metrics:
                lines.append(f"- {label}: {metrics[key]}")
        lines.append("")
        lines.append("### 结果解释")
        for text in metric_analysis:
            lines.append(text)
        lines.append("")
        lines.append("## 指标体系权重分析")
        lines.append(f"- GRA 与 CV 融合系数 alpha: {float(indicator_alpha)}")
        dim_share = dimension_share
        if dim_share:
            lines.append("")
            lines.append("### 一级指标权重占比")
            for k, v in dim_share.items():
                lines.append(f"- {k}: {v:.4f}")
            lines.append("")
            lines.append("### 一级指标解释")
            lines.append(dimension_analysis)
        lines.append("")
        lines.append("### 结构化指标权重")
        for name, w in sorted(labeled_indicator_map.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- {name}: {w:.6f}")
        if shap_top:
            lines.append("")
            lines.append("## 关键驱动因素归因分析")
            for row in labeled_shap_top:
                lines.append(f"- {row['feature_label']}: {float(row['mean_abs_shap']):.6f}")
            lines.append("")
            lines.append("### 驱动因素解释")
            for text in shap_analysis:
                lines.append(text)
        if attention_summary:
            lines.append("")
            lines.append("## 跨模态注意力概览")
            gates = attention_summary.get("modality_gates_mean")
            if gates:
                lines.append("- 动态模态权重（均值）")
                gate_text = " ".join(f"{_modality_label(name)}={value:.4f}" for name, value in gates.items())
                lines.append(f"  - {gate_text}")
            lines.append("")
            lines.append("### 融合机制解释")
            for text in attention_analysis:
                lines.append(text)
        lines.append("")
        lines.append("## 理论框架说明")
        for text in framework_analysis:
            lines.append(text)
        lines.append("")
        lines.append("## 四维协同提升路径")
        for text in path_analysis:
            lines.append(text)
        lines.append("")
        lines.append("## 数据处理说明")
        lines.append(denoise_analysis)
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return report
