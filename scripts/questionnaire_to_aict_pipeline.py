from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_python_binary() -> str:
    return sys.executable


def run_step(step_name: str, command: list[str], cwd: Path, extra_env: dict[str, str] | None = None) -> int:
    print(f"\n{'=' * 60}")
    print(f"[STEP] {step_name}")
    print(f"{' '.join(command)}")
    print("=" * 60, flush=True)
    env = os.environ.copy()
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    completed = subprocess.run(command, cwd=str(cwd), env=env)
    code = int(completed.returncode)
    if code != 0:
        print(f"[ERROR] {step_name} 失败，退出码={code}", file=sys.stderr)
    else:
        print(f"[OK] {step_name} 完成")
    return code


def step_process_questionnaire(args: argparse.Namespace) -> int:
    command = [
        resolve_python_binary(),
        str(Path("scripts") / "process_questionnaire.py"),
        "--input", str(args.input),
        "--output-dir", str(args.questionnaire_output_dir),
    ]
    if args.questionnaire_config:
        command.extend(["--config", str(args.questionnaire_config)])
    if args.enable_ai:
        command.append("--enable-ai")
    return run_step("问卷流水线：题项评分 + 质量检查 + 结构化拆分", command, Path.cwd())


def step_train_model(args: argparse.Namespace, train_csv: Path) -> int:
    model_config = args.model_config or str(Path("configs") / "questionnaire_model.yaml")
    command = [
        resolve_python_binary(),
        "-m",
        "src.aict_eval.train",
        "--data", str(train_csv),
        "--config", str(model_config),
    ]
    return run_step(
        "训练 AICT 多模态成效评价模型（问卷版：文本+结构化）",
        command,
        Path.cwd(),
    )


def step_infer_test(args: argparse.Namespace, test_csv: Path, model_dir: Path, prediction_csv: Path) -> int:
    model_config = args.model_config or str(Path("configs") / "questionnaire_model.yaml")
    command = [
        resolve_python_binary(),
        "-m",
        "src.aict_eval.infer",
        "--data", str(test_csv),
        "--config", str(model_config),
        "--model-dir", str(model_dir),
        "--output", str(prediction_csv),
    ]
    return run_step(
        "独立推理：在问卷测试集上生成 predicted_score 预测结果",
        command,
        Path.cwd(),
    )


def build_summary(
    args: argparse.Namespace,
    questionnaire_output_dir: Path,
    model_dir: Path,
    prediction_csv: Path,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "pipeline": "questionnaire_to_aict_model_v2",
        "inputs": {
            "input_questionnaire_csv": str(Path(args.input).resolve()),
            "questionnaire_config": str(Path(args.questionnaire_config).resolve()) if args.questionnaire_config else None,
            "model_config": str(Path(args.model_config).resolve()) if args.model_config else str((Path("configs") / "questionnaire_model.yaml").resolve()),
            "enable_ai": bool(args.enable_ai),
        },
        "outputs": {
            "questionnaire_analysis_csv": str((questionnaire_output_dir / "questionnaire_analysis.csv").resolve()),
            "model_ready_aict_dataset_csv": str((questionnaire_output_dir / "aict_dataset.csv").resolve()),
            "train_dataset_csv": str((questionnaire_output_dir / "train_dataset.csv").resolve()),
            "test_dataset_csv": str((questionnaire_output_dir / "test_dataset.csv").resolve()),
            "review_queue_csv": str((questionnaire_output_dir / "review_queue.csv").resolve()),
            "column_mapping_json": str((questionnaire_output_dir / "column_mapping.json").resolve()),
            "analysis_report_json": str((questionnaire_output_dir / "analysis_report.json").resolve()),
            "model_weights_pt": str((model_dir / "multimodal_evaluator.pt").resolve()),
            "indicator_weights_json": str((model_dir / "indicator_weights.json").resolve()),
            "metrics_json": str((model_dir / "metrics.json").resolve()),
            "preprocess_artifacts_json": str((model_dir / "preprocess_artifacts.json").resolve()),
            "shap_feature_importance_csv": str((model_dir / "shap_feature_importance.csv").resolve()),
            "report_json": str((model_dir / "report.json").resolve()),
            "report_md": str((model_dir / "report.md").resolve()),
            "prediction_on_test_csv": str(prediction_csv.resolve()),
        },
    }
    metrics_path = model_dir / "metrics.json"
    if metrics_path.exists():
        summary["evaluation_metrics"] = load_json(metrics_path)
    report_path = questionnaire_output_dir / "analysis_report.json"
    if report_path.exists():
        q_report = load_json(report_path)
        summary["questionnaire_summary"] = {
            "total_records": q_report.get("total_records"),
            "valid_records": q_report.get("valid_records"),
            "review_records": q_report.get("review_records"),
            "invalid_records": q_report.get("invalid_records"),
            "target_score": q_report.get("target_score") or q_report.get("summary"),
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="问卷数据一键对接 AICT：问卷处理 → 生成AICT数据集 → 模型训练 → 测试集预测",
    )
    parser.add_argument("--input", required=True, help="问卷星原始 CSV 或已有 scored_full.csv")
    parser.add_argument(
        "--questionnaire-output-dir",
        default=str(Path("outputs") / "questionnaire_pipeline"),
        help="问卷流水线输出目录（默认：outputs/questionnaire_pipeline）",
    )
    parser.add_argument(
        "--questionnaire-config",
        default=None,
        help="问卷流水线 JSON 配置（不指定用内置默认）",
    )
    parser.add_argument(
        "--model-config",
        default=None,
        help="AICT 训练 YAML 配置（不指定则用 configs/questionnaire_model.yaml）",
    )
    parser.add_argument(
        "--enable-ai",
        action="store_true",
        help="调用 DeepSeek 分析开放题；需设置 DEEPSEEK_API_KEY",
    )
    parser.add_argument(
        "--skip-process",
        action="store_true",
        help="跳过问卷处理步骤（直接用现有 --questionnaire-output-dir 中的结果训练）",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="跳过模型训练步骤（仅执行问卷流水线 + 输出数据）",
    )
    parser.add_argument(
        "--skip-infer",
        action="store_true",
        help="跳过测试集推理步骤",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()
    questionnaire_output_dir = Path(args.questionnaire_output_dir).resolve()

    if not args.skip_process:
        code = step_process_questionnaire(args)
        if code != 0:
            raise SystemExit(code)

    train_csv = questionnaire_output_dir / "train_dataset.csv"
    test_csv = questionnaire_output_dir / "test_dataset.csv"
    model_config_path = Path(args.model_config) if args.model_config else Path("configs") / "questionnaire_model.yaml"
    model_dir = Path(cwd / "outputs_questionnaire").resolve()
    if model_config_path.exists():
        try:
            import yaml
            with open(model_config_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            train_section = raw.get("train") or {}
            if train_section.get("output_dir"):
                model_dir = Path(train_section["output_dir"]).resolve()
        except Exception:
            pass

    prediction_csv = questionnaire_output_dir / "test_predictions.csv"

    if not args.skip_train:
        if not train_csv.exists():
            print(f"[ERROR] 训练集文件不存在: {train_csv}", file=sys.stderr)
            print("[HINT] 你可能需要先运行 --skip-process=false 或指定已存在的 --questionnaire-output-dir。")
            raise SystemExit(2)
        code = step_train_model(args, train_csv)
        if code != 0:
            raise SystemExit(code)

    if not args.skip_infer:
        if not test_csv.exists():
            print(f"[ERROR] 测试集文件不存在: {test_csv}", file=sys.stderr)
            raise SystemExit(2)
        if not (model_dir / "preprocess_artifacts.json").exists():
            print(f"[ERROR] 模型目录缺少 preprocess_artifacts.json: {model_dir}", file=sys.stderr)
            print("[HINT] 你需要先完成训练步骤，再执行推理。")
            raise SystemExit(2)
        code = step_infer_test(args, test_csv, model_dir, prediction_csv)
        if code != 0:
            raise SystemExit(code)

    summary = build_summary(args, questionnaire_output_dir, model_dir, prediction_csv)
    summary_path = questionnaire_output_dir / "pipeline_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 问卷对接 AICT 流水线全部完成。汇总见: {summary_path}")

    report_html = questionnaire_output_dir / "report.html"
    report_cmd = [
        resolve_python_binary(),
        str(Path("scripts") / "build_report_html.py"),
        "--output-dir", str(questionnaire_output_dir),
        "--model-dir", str(model_dir),
        "--questionnaire-report", str((questionnaire_output_dir / "analysis_report.json").resolve()),
        "--aict-dataset", str((questionnaire_output_dir / "aict_dataset.csv").resolve()),
        "--train-csv", str(train_csv.resolve()),
        "--test-csv", str(test_csv.resolve()),
        "--output-html", str(report_html.resolve()),
        "--title", "AI+文旅多模态融合成效评价报告（问卷自动生成）",
        "--version", "AICT v2.0",
    ]
    code = run_step("生成可视化静态 HTML 报告（可直接浏览器打开）", report_cmd, Path.cwd())
    if code == 0 and report_html.exists():
        summary["outputs"]["visualization_report_html"] = str(report_html.resolve())
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[OK] 可视化报告: {report_html}")


if __name__ == "__main__":
    main()
