# AICT: AI+文旅应用成效智能评价算法原型 v2

本仓库实现了"AI+文旅应用成效评价"的端到端闭环能力：从问卷原始 CSV → 质量清洗 → 四维度加权评分（百分制）→ 结构化 / 文本双模态训练（多模态兼容）→ 可解释归因 → 最终静态 HTML 可视化报告（Chart.js 热替换）。

算法与工程能力一览：

1. **指标筛选与赋权**：灰色关联分析 GRA + 变异系数 CV + 皮尔逊相关 + 熵权法四源融合，默认做 80 组 α 网格搜索；支持自动最优
2. **结构化指标去噪**：卡尔曼滤波 / 自适应 EMA / 中值 / 移动平均 / Savitzky-Golay / Haar 小波
3. **多模态成效评价**：文本（HashTokenizer/BERT 可选）+ 图像（ResNet+SE）+ 语音（统计/wav2vec2 可选）+ 结构化（4源 → 多层 CrossModalAttention + 动态门控 + CLS 汇总 + 辅助损失
4. **可解释性**：SHAP 代理模型 + 跨模态注意力热力 + 模态门控权重 + 指标贡献明细表
5. **可视化报告**：`templates/aict_report_template.html` + `scripts/build_report_html.py` 通过 `__AICT_DATA_INJECT__` 热替换`__END_AICT_DATA__` 的 **热替换，产出 18 个 section 段，可离线查看

---

## 一、目录结构（当前版本）

```text
AICT/
├─ aict_output/                       # 问卷源数据样本（scored_full.csv）
├─ configs/
│  ├─ default.yaml                # 通用三/四模态默认配置
│  ├─ questionnaire_pipeline.json # 问卷流水线：计分规则/质量阈值/AI 标注/缺省模态列
│  └─ questionnaire_model.yaml     # 问卷专用训练配置（epochs=3/freeze encoders/关闭 mixed_precision=false）
├─ scripts/
│  ├─ process_questionnaire.py     # 问卷处理：清洗→4维→质量控制→结构化拆分
│  ├─ questionnaire_to_aict_pipeline.py # 一键端到端：问卷→训练→推理→可视化报告
│  └─ build_report_html.py          # 把 pipeline_summary.json → 填入 HTML 报告
├─ src/aict_eval/
│  ├─ config.py                   # 配置定义
│  ├─ dataset.py                  # 多模态数据集/占位符/去噪/拆分
│  ├─ filters.py                  # 结构化指标去噪算法
│  ├─ weights.py                  # 四源赋权+网格搜索
│  ├─ model.py                    # 多模态融合模型（含 4 类编码器+跨模态融合
│  ├─ explain.py                 # SHAP 代理模型
│  ├─ report.py                     # JSON/Markdown/注意力统计
│  ├─ train.py                     # 训练闭环（差分LR/调度/梯度累积/辅助损失
│  └─ infer.py                    # 推理（批量推理批量推理：加载工件→预测
├─ templates/
│  └─ aict_report_template.html  # 最终静态 HTML 报告模板（Chart.js 4.4）
├─ requirements.txt
├─ README.md
├─ DATASET_FIELD_SPEC.md            # 数据集字段规范
├─ DATA_REQUIREMENTS_STANDARD.md  # 数据采集入库标准
├─ QUESTIONNAIRE_SCORING_SYSTEM.md # 问卷评分体系 20 题/4 维度/ reverse_order
└─ QUESTIONNAIRE_PIPELINE.md      # 问卷流水线对接细节
```

> 临时输出（`outputs*/`、模型权重 `*.pt/*.bin`、压缩包 `*.zip` 已加入 [.gitignore](file:///c:/Users/lenovo/Desktop/AICT/.gitignore) 忽略，**请不要通过 .gitignore**。

---

## 二、问卷评分默认约定（重要）

默认使用 **20 题 5 点 Likert 量表（`reverse_order`（选项顺序：

- 问卷星按“5/4/3/2/1 选项导出时，使用规则为选项位置 1~5 实际分数= = 6 − 导出值。

四维度分：

| 题号 | 维度 | 权重 |
|---|---|---|
| q1~q5  | 技术赋能效能 (tech_empowerment) | 0.25 |
| q6~q10 | 游客感知体验 (visitor_experience) | 0.30 |
| q11~q15| 文化价值传播 (cultural_value) | 0.25 |
| q16~q20| 经济社会增值 (economic_social_gain) | 0.20 |

- 各维度百分制 = `平均*20` ×100` 加权 = 按上表权重和，`target_score` = 加权和，范围 0~100。
- 质量 6 项罚分（答题时长/直线/低方差/重复向量/重复IP/总体不一致性，扣 quality_score=100-Σpenalty*flag≥75 为 pass。
- 默认 include_derived_dimensions=false（**默认不写 4 维度分 + quality_score 不进入训练特征），**仅保留 10 列基础列。

> 详细规则见 [QUESTIONNAIRE_SCORING_SYSTEM.md](file:///c:/Users/lenovo/Desktop/AICT/QUESTIONNAIRE_SCORING_SYSTEM.md) 和 [QUESTIONNAIRE_PIPELINE.md](file:///c:/Users/lenovo/Desktop/AICT/QUESTIONNAIRE_PIPELINE.md)。

---

## 三、三种使用方式

### 方式 1：仅处理问卷（无需训练）

仅做打分 + 出结构化数据（无需 API：

```powershell
python scripts/process_questionnaire.py `
  --input "aict_output\scored_full.csv" `
  --output-dir "outputs\questionnaire_v2"
```

`--config` 参数完全可省。配置文件完全可省略，完全默认读默认默认会自动默认配置文件默认完全可省略完全可省略完全完全省略）。

输出文件：

- `questionnaire_analysis.csv`：全量明细审计明细 20 题明细，四维分/质量 flag/AI 标注，
- `aict_dataset.csv`：AICT 标准 10 列训练集，
- `train_dataset.csv / test_dataset.csv`：不含 `dataset_split` 列，直接喂模型，
- `analysis_report.json`：四维统计量/Cronbach/质量统计/样本拆分统计。

### 方式 2：一键端到端（问卷 → 训练 → 推理 → HTML 报告（推荐）

```powershell
python scripts/questionnaire_to_aict_pipeline.py `
  --input "aict_output\scored_full.csv" `
  --questionnaire-output-dir "outputs\questionnaire_pipeline"
```

可选开关：

| 开关 | 用途 |
|---|---|
| `--enable-ai` | 调用 DeepSeek API Key 只从环境变量读，不写代码或配置。
| `--skip-process` | 跳过问卷步骤，训练已有结构化
| `--skip-train` | 仅问卷，不训练
| `--skip-infer` | 不跑推理步骤 |
| `--questionnaire-config" | 指定问卷配置
| `--model-config | 指定训练配置 | `configs/questionnaire_model.yaml) | 问卷默认配置

最终产物：

```text
outputs\questionnaire_pipeline\
├── aict_dataset.csv
├── train_dataset.csv / test_dataset.csv
├── analysis_report.json
├── test_predictions.csv       # 推理结果，predicted_score 列
├── pipeline_summary.json     # 全链路汇总（含指标 / 路径清单）
└── report.html                  # 最终静态 HTML 报告（8 大 section / 18 headings，Chart.js 热替换
```

### 方式 3：直接训练多模态模型（已有标准 CSV）

```bash
python -m src.aict_eval.train --data path/to/data.csv --config configs/default.yaml
```

标准字段最低要求（缺省列最少字段：

- `review_text` 文本列
- `image_path` / `audio_path` 可为空字符串或列缺失时自动用零占位；若列缺失则配置设为null。
- `target_score` 标签列；其余数值列自动结构化特征。

---

## 四、算法设计（v2 升级版）

### 4.1 指标赋权（4 源融合 + 自动 α=GRA+CV+皮尔逊+熵权

权重公式：
$$
\mathbf{w}=\alpha \mathbf{w}_{GRA}+\beta \mathbf{w}_{CV}+\gamma \mathbf{w}_{Pearson}+\delta \mathbf{w}_{Entropy}} \quad,\quad \alpha+\beta+\gamma+\delta=1
$$

- 默认 `auto_indicator_weight_alpha=true`：80 组 5×4×4=80 组 αβγ 网格搜索，目标：赋权后得分与 target_score 的绝对相关系数最大。

### 4.2 多模态模型

文本编码器：

- 首选 bert-base-chinese 在线下载 / 离线回退 HashTokenizer + 双层双向 LSTM 自注意力池化。

图像：

- ResNet18/ResNet50 可选 + SE 通道注意力 + GlobalAvg+Max 池化 → 投影投影层：Linear→GELU→LayerNorm→Dropout。

语音（双通道回退：在线 wav2vec2/HuBERT/Whisper / 离线统计梅尔分箱谱质心/滚降/带宽/平坦度 + 时域 ZCR/RMS/偏度/峰度 → 3 层 MLP。

结构化：StandardScaler → 按权重加权加权 → 3 层 MLP 投影。

跨模态融合（CrossModalBlock 重写版 CrossModalAttention：

- 每个模态独立 Q/K/V/Out_proj；可学习 modality_coeff 系数；FFN 768 width，fusion_layers=3 + CLS Token 汇总 + 辅助损失（aux_wt=0.15）。
- ModalityGating：3 层 MLP concat→降维→GELU→LayerNorm→Softmax。

### 4.3 可解释性

- SHAP：GradientBoosting 代理；SHAP 重要性 CSV 导出 ，SHAP 条/柱图。
- 注意力统计：CrossModalAttention 首层热力矩阵（按 q_name→to_k_name 聚合输出，按 batch_size 加权汇总后输出 写入 report.json。
- 模态门控权重：逐样本平均 Modality Gates 输出权重，做极坐标图。

### 4.4 训练闭环

差分 LR（Backbone 1e-5，融合头 1e-4）权重衰减分组，cosine warmup，Huber loss δ=1.0，辅助损失 0.15，EMA 去噪，EarlyStopping patience=3。

问卷模型配置见 [questionnaire_model.yaml](file:///c:/Users/lenovo/Desktop/AICT/configs/questionnaire_model.yaml)：
- `epochs=3`（演示模式；后续可改 30、`freeze encoders = true`、`mixed_precision=false`、`image_column: null`、`audio_column: null`。

---

## 五、数据采集标准与格式规范

- 字段与格式规范：[DATASET_FIELD_SPEC.md](file:///c:/Users/lenovo/Desktop/AICT/DATASET_FIELD_SPEC.md)
- 数据采集与入库需求标准：[DATA_REQUIREMENTS_STANDARD.md](file:///c:/Users/lenovo/Desktop/AICT/DATA_REQUIREMENTS_STANDARD.md)
- 问卷评分体系：[QUESTIONNAIRE_SCORING_SYSTEM.md](file:///c:/Users/lenovo/Desktop/AICT/QUESTIONNAIRE_SCORING_SYSTEM.md)
- 问卷流水线细节：[QUESTIONNAIRE_PIPELINE.md](file:///c:/Users/lenovo/Desktop/AICT/QUESTIONNAIRE_PIPELINE.md)

---

## 六、安装依赖

```bash
pip install -r requirements.txt
```

核心依赖建议版本：torch>=2.1 / transformers>=4.36 / shap>=0.44 / scikit-learn>=1.3 / pandas>=2.1 / numpy>=1.24 / PyYAML>=6 / Pillow>=10 / tqdm>=4.66。

---

## 七、适合下一步扩展方向

1. 接入真实景区监测日志，做多批次对照实验验证指标鲁棒性；
2. 图像编码器替换 CLIP/Swin Transformer；
3. 语音用 Whisper 做 ASR 文本 + 声学双塔；
4. 视频帧/生理传感器等时序模态；
5. 四维多维度分多任务学习；
6. FreeAT/SMART 对抗训练。
