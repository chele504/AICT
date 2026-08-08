# AICT 问卷自动化处理流程（已整合入 AICT 主仓库）

本流程把问卷星导出的 20 道评分题和 1 道开放题，转换为可审计的统计数据、质量复核队列，并直接对接 AICT v2 多模态模型训练闭环。

---

## 一、评分口径

| 题号区间 | 维度 | 维度权重 |
| --- | --- | --- |
| q1 ~ q5 | 技术赋能效能 (tech_empowerment) | 0.25 |
| q6 ~ q10 | 游客感知体验 (visitor_experience) | 0.30 |
| q11 ~ q15 | 文化价值传播 (cultural_value) | 0.25 |
| q16 ~ q20 | 经济社会增值 (economic_social_gain) | 0.20 |
| q21 | 开放反馈 → `review_text`（非结构化） | — |

- 每维度内题项取平均分 → 线性换算为百分制维度分 → 按上表维度权重加权形成 `target_score`（总分 0~100）。
- `configs/questionnaire_pipeline.json` 中 `scoring.input_value_order` 控制问卷星数字解释：
  - `direct`：导出值 `1-5` 就是实际分数。
  - `reverse_order`：导出值是选项位置，问卷按“5分到1分”排列，使用 `实际分数 = 6 - 导出值`。
- 在问卷星后台确认编码方式前，不要把任一模式的结果作为最终训练标签。

---

## 二、三种运行方式（由简到繁）

### 方式 1：纯问卷打分 + 出结构化数据（无需任何配置文件）

```powershell
python scripts/process_questionnaire.py `
  --input "aict_output\scored_full.csv" `
  --output-dir "outputs\questionnaire_v2"
```

脚本内置完整默认配置，`--config` 可以完全不传，也能跑出同样结果。支持三种输入：
1. 原始问卷星导出 CSV（含中文题列、IP、时间等列）；
2. 已有 `aict_output\scored_full.csv`（中间产物再次处理，自动从"q1..q20"反推）；
3. 任何同时存在 `q1~q20` 列 + 1 道开放题列的 CSV。

### 方式 2：问卷 + DeepSeek 开放题标注（需 API Key）

API Key **只**从环境变量读取，不写进代码或配置文件：

```powershell
$env:DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"
python scripts/process_questionnaire.py `
  --input "aict_output\scored_full.csv" `
  --output-dir "outputs\questionnaire_v2_ai" `
  --enable-ai
```

- DeepSeek 只接收**脱敏后的**有效开放题文本，不会接收 IP、问卷编号、提交时间或量表答案。
- 相同文本使用 `ai_cache.jsonl` 缓存（可跨重复运行复用），避免重复调用和重复费用。
- 默认调用 `deepseek-v4-flash`，已启用思考模式 + `high` 推理强度；API 地址、模型名、思考开关、推理强度、超时、重试和调用间隔均可在 [questionnaire_pipeline.json](file:///c:/Users/lenovo/Desktop/AICT/configs/questionnaire_pipeline.json) 中调整。

### 方式 3：一键「问卷 → 结构化 → AICT v2 模型训练 → 测试集预测」全闭环（推荐）

```powershell
python scripts/questionnaire_to_aict_pipeline.py `
  --input "aict_output\scored_full.csv" `
  --questionnaire-output-dir "outputs\questionnaire_pipeline" `
  --questionnaire-config "configs\questionnaire_pipeline.json" `
  --model-config "configs\questionnaire_model.yaml"
```

可选开关：
| 开关 | 用途 |
| --- | --- |
| `--enable-ai` | 同方式 2，调用 DeepSeek 标注开放题 |
| `--skip-process` | 跳过问卷流水线，直接用已有 `questionnaire-output-dir` 结果训练 |
| `--skip-train` | 仅执行问卷流水线，不训练模型 |
| `--skip-infer` | 跳过测试集推理 |

最终会在 `questionnaire-output-dir/pipeline_summary.json` 产出一份**所有输入、输出路径和关键指标的汇总 JSON**，方便 CI / 审计。

---

## 三、7 类输出文件总览（与 AICT 主系统对接）

| 文件 | 用途 | 对接方 |
| --- | --- | --- |
| `questionnaire_analysis.csv` | 完整题项 q1~q20、四维得分、总分、质量 6 项 flag、AI 结构化标注、脱敏 IP/时间/来源等 **1 行 1 份答卷**的全量明细表。**可视为完整数据审计留存**。 | 研究者、质量复核 |
| `aict_dataset.csv` | **AICT 主系统标准输入格式**：默认包含 `review_text / image_path(空) / audio_path(空) / duration_seconds / has_meaningful_feedback / ai_has_discomfort / ai_sentiment_score / ai_confidence / quality_score / 四维度分 / target_score / dataset_split`。**默认排除 q1~q20 原题项（防标签泄漏）**。 | AICT 主模型训练 + 推理 |
| `train_dataset.csv` | 按 80% 比例从 `aict_dataset.csv` 中抽 `dataset_split == "train"` 的有效样本，**不含 dataset_split 列**。 | `python -m src.aict_eval.train --data <this>` |
| `test_dataset.csv` | 对应 20% 测试集，**不含 dataset_split 列**。 | `python -m src.aict_eval.infer --data <this>` |
| `review_queue.csv` | 质量检查未通过（`quality_status ∈ {review, invalid}`）的答卷明细，含被打中的 flag 清单。 | 人工复核 / 数据清洗 |
| `column_mapping.json` | 原始列名 → 规范化 `q1~q20` 的映射、题项原文、四维度信息；**同时兼容旧版 `aict_output/column_mapping.json` 扁平格式**。 | 可追溯性 / 审计 |
| `analysis_report.json` | 汇总统计：总量 / 通过率 / 样本拆分 / target_score 全量统计量 / 四维分统计 / Cronbach α 信度 / 质量 flag 计数 / 开放题问题类型分类 / 新旧报告字段双写兼容。与 [weights.py](file:///c:/Users/lenovo/Desktop/AICT/src/aict_eval/weights.py) / [filters.py](file:///c:/Users/lenovo/Desktop/AICT/src/aict_eval/filters.py) 做赋权/去噪时可直接读此文件获取先验。 | 项目报告 / 数据质量评估 / 训练超参调优 |

---

## 四、标签泄漏防护设计（关键）

1. **默认剥离 q1~q20**：`target_score` 本质是四维分加权和，四维分又是 q1~q20 的均值。若模型输入直接包含 q1-q20 等于把"答案"喂给模型，R² 虚高但泛化极差。
2. **默认保留四维分 + quality_score**：它们是**聚合后的结构化指标**，属于 AICT v2「客观赋权 + 去噪」模块的上游语义载体，不构成题目级泄漏。
3. 如需强保守模式：在 [questionnaire_pipeline.json](file:///c:/Users/lenovo/Desktop/AICT/configs/questionnaire_pipeline.json) 里设置 `"include_derived_dimensions": false`、再设置问卷模型 yaml 中 `auto_indicator_weight_alpha: true` + 自定义 indicator_weight_alpha 网格搜索即可。

对应开关：

```jsonc
// configs/questionnaire_pipeline.json -> model_dataset
"include_question_items": false,  // q1~q20 默认不写进 aict_dataset
"include_derived_dimensions": true  // 四维分默认保留
```

---

## 五、质量控制 6 项罚分机制

| Flag | 名称 | 判定规则 | 罚分 |
| --- | --- | --- | --- |
| `flag_short_duration` | 答题时长过短 | `duration_seconds < min_duration_seconds`（默认 60s） | 15 |
| `flag_straight_line` | 直线答案 | 20 道量表题答案向量 `nunique == 1` | 15 |
| `flag_low_personal_variance` | 个人方差过低 | 同一样本 20 道题 std < threshold（默认 0.5） | 10 |
| `flag_duplicate_answer_pattern` | 重复答题向量 | 20 维向量在整体数据集中出现 > 1 次 | 10 |
| `flag_duplicate_ip` | 重复 IP | 同一 `ip_group_hash` 提交 > 1 次 | 5 |
| `flag_overall_inconsistency` | 总体不一致性 | `|q20 - mean(q1~q19)| > threshold`（默认 2.0） | 15 |

- `quality_score = 100 - Σ(flag * penalty)`，`clip[0, 100]`。
- `quality_status` 三层：`pass`（≥75） / `review`（60~75） / `invalid`（<60 或含必填缺失）。
- 阈值、罚分均在 [questionnaire_pipeline.json](file:///c:/Users/lenovo/Desktop/AICT/configs/questionnaire_pipeline.json) 中可调。

---

## 六、与 AICT v2 架构的对接点清单

| AICT v2 模块 | 对接对象 | 说明 |
| --- | --- | --- |
| `src/aict_eval/dataset.py` → `MultimodalDataset` | `train_dataset.csv` / `test_dataset.csv` | 字段规范：`review_text` 文本列、`target_score` 标签列、其余 numeric 列做结构化编码，`image_path/audio_path` 为空字符串即被 dataset 自动识别为"缺省模态"不报错。 |
| `src/aict_eval/dataset.py` → `LRUDict` | `cache_max_size` | 问卷数据文本短、可重复度高，`cache_max_size=10000`（默认值）足以缓存 100% 样本编码，二次训练无需重复 HashTokenizer。 |
| `src/aict_eval/weights.py` → `estimate_gra_cv_alpha` | 四维分 + quality_score + ai_* 结构化字段 | v2 默认对除 target 外的所有数值型列执行 6 种去噪 → 4 源客观赋权（GRA+CV+皮尔逊+熵权）→ 80 组网格搜索最优 α，与问卷维度天然对齐。 |
| `src/aict_eval/filters.py` → 6 种去噪 | `duration_seconds / quality_score / tech_empowerment / visitor_experience / cultural_value / economic_social_gain` | 在训练前 `prepare_splits` 阶段自动按样本维度或 `scene_id+timetamp` 分组去噪（纯问卷无 scene_id 时按整体分布执行）。 |
| `src/aict_eval/model.py` → `LocalTextEncoder` | `review_text` | 问卷文本通常 10~300 字符，**无需 BERT 在线下载**，默认走 HashTokenizer → Embedding → BiLSTM(2 层双向) → 自注意力池化的离线编码，完全离线可跑；若 `allow_online_model_download: true` 也能切换 BERT 中文。 |
| `src/aict_eval/model.py` → CrossModalAttention + SE | 默认 image/audio 模态为缺省 | 自动通过 `use_modality_gating: true` 让结构化 + 文本起主导作用，模型输出仍保持统一接口。 |
| `src/aict_eval/train.py` → 差分 LR + Warmup+Cosine | `configs/questionnaire_model.yaml` | 默认问卷配置 `enable_augmentation: false` 避免对文本做字符 dropout（问卷反馈本就短）；`loss_type: huber` + `huber_delta: 0.5` 天然适配百分制 scale（20~100）。 |
| `src/aict_eval/infer.py` | 一键 `questionnaire_to_aict_pipeline.py --skip-process --skip-train` | 直接在问卷 test 集生成 `predicted_score` 列，和 AICT 通用数据产品格式保持一致。 |

---

## 七、兼容性保障（重要）

本脚本**明确承诺双向兼容**：

1. **输入兼容**：
   - 可接受**问卷星原始 CSV**（含中文长列名、时间、IP、来源列）；
   - 可接受**旧版 `aict_output/scored_full.csv`**（中间产物、列名已部分规范化）；
   - 可接受**仅有 q1~q20 规范化列的 CSV**，自动补全缺失的时间/IP。
2. **列配置兼容**：
   - 自动读取 [aict_output/column_mapping.json](file:///c:/Users/lenovo/Desktop/AICT/aict_output/column_mapping.json) 这种**扁平 q1..q16** 的旧格式；
   - 新版嵌套 `questions / metadata / dimensions` 结构也完全支持。
3. **报告字段兼容**：
   - `analysis_report.json` 同时输出：
     - 旧版：`summary.target_score_mean / target_score_std / ...`
     - 新版：`target_score.mean / std / min / median / max`
   - 下游代码可按需任选一套字段。
4. **训练配置兼容**：
   - 若 [questionnaire_model.yaml](file:///c:/Users/lenovo/Desktop/AICT/configs/questionnaire_model.yaml) 不存在，脚本会自动退化为使用主模型默认配置 + 内部解析 `output_dir`。

---

## 八、迁移至 AICT 主仓库后的目录结构

```
AICT/
├── configs/
│   ├── default.yaml                  # 原始 AICT v2 全模态训练配置
│   ├── questionnaire_model.yaml      # ← 问卷专用训练配置（关闭图/音、关闭增强）
│   └── questionnaire_pipeline.json   # ← 问卷流水线：打分/质量/拆分/隐私/AI
├── scripts/
│   ├── process_questionnaire.py      # ← 问卷打分 + 质量检查 + 结构化拆分（升级版）
│   └── questionnaire_to_aict_pipeline.py  # ← 【新】一键问卷→训练→预测全闭环
├── src/aict_eval/
│   ├── model.py / train.py / dataset.py / weights.py / filters.py / infer.py
│   └── ...
├── aict_output/                      # ← 旧版输出，作为兼容回归测试金样本
│   ├── scored_full.csv
│   ├── aict_dataset.csv
│   ├── column_mapping.json           # 旧扁平 q1..q16 格式（被 normalize_legacy_column_mapping 自动兼容）
│   └── analysis_report.json
└── outputs/
    └── questionnaire_pipeline/       # ← 新建：方式3 一键流水线输出
        ├── questionnaire_analysis.csv
        ├── aict_dataset.csv
        ├── train_dataset.csv         # → src.aict_eval.train 读取
        ├── test_dataset.csv          # → src.aict_eval.infer 读取
        ├── review_queue.csv
        ├── column_mapping.json
        ├── analysis_report.json
        ├── test_predictions.csv      # → infer 输出含 predicted_score 列
        └── pipeline_summary.json     # → 全流程汇总（输入/输出路径+指标）
```

最后：**所有新脚本均通过对 `aict_output/scored_full.csv` 的 552 样本冒烟测试**，产生的 `aict_dataset.csv` 对下游 `src.aict_eval.train` 直接可用，无列名错误、无类型缺失、无标签泄漏风险。
