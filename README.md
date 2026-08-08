# AICT: AI+文旅应用成效智能评价算法原型

这个原型围绕课题"基于多模态数据融合的 AI+文旅应用成效智能评价模型研究"实现了以下核心能力：

1. **指标筛选与赋权**：`灰色关联分析 GRA + 变异系数 CV + 皮尔逊相关 + 熵权法`（4源融合，支持网格搜索最优加权系数 α）
2. **结构化指标去噪**：`卡尔曼滤波 / 自适应 EMA / 中值滤波 / 移动平均 / Savitzky-Golay 多项式平滑 / Haar 小波软阈值`
3. **多模态成效评价**：`中文文本编码 + 图像编码（ResNet18/50 + SE 通道注意力） + 语音编码（wav2vec2/HuBERT/Whisper/增强统计特征） + 结构化指标融合（多层跨模态注意力 + 动态门控 + CLS Token 汇总 + 各模态辅助监督）`
4. **可解释性分析与诊断报告**：`SHAP + 成效诊断报告（JSON/Markdown）`

---

## 目录结构

```text
AICT/
├─ configs/default.yaml
├─ examples/
│  ├─ generate_demo_data.py
│  ├─ dataset_template.csv
│  ├─ demo_dataset.csv
│  ├─ demo_dataset_audio.csv
│  └─ demo_audio/               # 40 条示例 WAV 音频
├─ requirements.txt
├─ DATASET_FIELD_SPEC.md        # 数据集字段与格式规范
├─ DATA_REQUIREMENTS_STANDARD.md # 数据采集与入库需求标准
├─ QUESTIONNAIRE_SCORING_SYSTEM.md # 问卷评分与标签计算体系
└─ src/aict_eval/
   ├─ config.py                 # 配置定义（模型/训练/解释/报告）
   ├─ dataset.py                # 多模态数据集 + 数据增强 + LRU 缓存
   ├─ filters.py                # 结构化指标去噪算法集合
   ├─ explain.py                # SHAP 代理模型训练 + 重要性导出
   ├─ model.py                  # 多模态评测网络（含 4 类编码器 + 跨模态融合）
   ├─ report.py                 # 诊断报告生成（JSON/Markdown）
   ├─ train.py                  # 训练闭环（差分 LR + 调度 + 梯度累积 + 辅助损失）
   ├─ weights.py                # 客观赋权算法（GRA/CV/皮尔逊/熵 + 融合 + 网格搜索）
   └─ infer.py                  # 独立推理脚本（加载训练工件批量预测）
```

---

## 算法设计（v2 升级版）

### 1. 指标赋权（四源融合 + 自动 α 搜索）

> 代码入口：[weights.py](file:///c:/Users/lenovo/Desktop/AICT/src/aict_eval/weights.py)

将 4 种主流客观赋权法通过凸组合融合，最终权重公式：

$$
\mathbf{w} = \alpha \cdot \mathbf{w}_{\text{GRA}} + \beta \cdot \mathbf{w}_{\text{CV}} + \gamma \cdot \mathbf{w}_{\text{Pearson}} + \delta \cdot \mathbf{w}_{\text{Entropy}}
\quad;\quad \alpha+\beta+\gamma+\delta=1
$$

| 方法 | 说明 | 擅长 |
|------|------|------|
| 灰色关联分析 GRA | 度量曲线几何相似性 | 小样本 + 非线性相关 |
| 变异系数 CV | `std/abs(mean)` | 区分度大的指标自动加权 |
| 皮尔逊相关 | 线性相关绝对值 | 捕捉与目标的线性依赖 |
| 熵权法 | 根据信息熵反向分配 | 信息量大的指标加权 |

- 默认 `auto_indicator_weight_alpha: true` 时，会对 **α/β/γ 进行 5×4×4=80 组网格搜索**，以"赋权综合得分"与真实 `target_score` 的绝对相关系数为目标函数，自动选择最优点。

### 2. 多模态模型（增强版）

> 代码入口：[model.py](file:///c:/Users/lenovo/Desktop/AICT/src/aict_eval/model.py)

#### 2.1 文本编码器

- 首选：`bert-base-chinese`（支持在线下载 + 本地离线缓存）；`<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> + 均值池化 残差相加`作为句向量。
- 回退：`LocalTextEncoder`（**升级版**）= `Embedding` → `双层双向 LSTM` → `自注意力池化` + `均值池化 残差` → `LayerNorm + Dropout`。

#### 2.2 图像编码器

- Backbone 可选 `ResNet18` / `ResNet50`（通过 `model.image_model_name` 切换）；
- **新增** [SEBlock](file:///c:/Users/lenovo/Desktop/AICT/src/aict_eval/model.py#L18-L33) 通道注意力，对 feature map 做通道重标定；
- 池化升级为：`GlobalAvgPool + GlobalMaxPool` 双通道相加；
- 投影层：`Linear → GELU → LayerNorm → Dropout`。

#### 2.3 语音编码器（双通道回退机制）

- **首选（online）**：`AutoFeatureExtractor + AutoModel`，支持 `wav2vec2 / HuBERT / Whisper encoder`。
- **增强离线回退 StatsAudioEncoder**（统计编码器升级）：
  - STFT → 64 维梅尔分箱，在每个分箱上计算：`log/mag 的均值/方差/最大值`；
  - 全局频谱特征：`谱质心 / 谱滚降 / 谱带宽 / 谱平坦度`；
  - 全局时域波形特征：`过零率 ZCR / RMS 能量 / 偏度 / 峰度`；
  - 最终通过 3 层 MLP（GELU + LayerNorm + Dropout）投影。

#### 2.4 结构化编码器

- `StandardScaler` → `Indicator Weights 逐列加权` → 3 层 MLP（GELU + LayerNorm）投影。
- 隐藏层从 `64` → `128`（`tabular_hidden_size` 默认 128）。

#### 2.5 跨模态融合（重写版 CrossModalBlock）

- 每个模态独立维护 `Q_proj / K_proj / V_proj / Out_proj`，而非共用 `nn.MultiheadAttention`；
- **新增** 可学习 `modality_coeff ∈ R^{M×M}`，每个目标模态对 KV 源模态的贡献权重可训练；
- FFN：`Linear → GELU → Dropout → Linear → Dropout`（更宽的 `fusion_ffn_size = 768`）；
- 默认 `fusion_layers = 3`（原 2）；
- **新增** `<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> 可学习 Token`，经 `MultiheadAttention` 对融合后多模态 token 做注意力聚合，最终与各模态 token 拼接回归；
- **新增** `辅助回归头 aux_regressor`：训练期对每模态特征做独立回归，与主预测加权训练。

#### 2.6 模态门控 ModalityGating

- 加深为 3 层 MLP（`concat → 降维 → GELU → LayerNorm → 进一步降维 → Softmax`），输出 M 维权重对各模态 feature 做逐元素缩放。

---

### 3. 可解释性

> 代码入口：[explain.py](file:///c:/Users/lenovo/Desktop/AICT/src/aict_eval/explain.py)

- 训练完成后，基于结构化指标拟合 `GradientBoostingRegressor` 代理模型；
- 使用 `SHAP`（`summary_plot/bar`）输出影响成效分值的关键指标排序；
- 结果导出 CSV，供报告模块生成《关键影响因子分析》章节。

---

### 4. 结构化指标去噪（6 种算法 + 分组/排序）

> 代码入口：[filters.py](file:///c:/Users/lenovo/Desktop/AICT/src/aict_eval/filters.py)

| 方法名（denoise_method） | 原理 | 适用场景 |
|---|---|---|
| `kalman` | 标量卡尔曼滤波（过程/测量方差可调） | 受传感器噪声干扰的时序数据 |
| `adaptive_ema` / `ema` | 自适应 EMA：根据局部窗口 std 动态调整 α | 业务指标渐进波动 |
| `median` / `medfilt` | 中值滤波（窗口可调） | 存在孤立尖峰/离群点 |
| `moving_average` / `ma` | 窗口平均平滑 | 简单降噪 |
| `savgol` / `sg` / `savitzky_golay` | 局部多项式最小二乘拟合（边缘也拟合） | 需保留曲线形状的平滑 |
| `wavelet` / `haar` / `dwt` | Haar 小波 L 层分解 + MAD 自适应软/硬阈值 | 高频信号与噪声分离 |

- 可选参数 `denoise_group_column + denoise_sort_column`，实现**同评价对象按时序分组去噪**（最符合"文旅项目长期监测"场景）。

---

### 5. 训练闭环（全面增强）

> 代码入口：[train.py](file:///c:/Users/lenovo/Desktop/AICT/src/aict_eval/train.py)

| 能力 | 说明 | 默认 |
|------|------|------|
| **差分学习率** | Backbone（文本/图像/语音主干）2e-5，其余头/融合网络 2e-4 | 开启 |
| **权重衰减分组** | Norm 层 & 偏置不加 WD，其余 1e-4 | 开启 |
| **LR 调度器** | `LinearLR warmup + CosineAnnealingLR`（或纯 Linear），支持按 `warmup_epochs` 或 `warmup_ratio`，最小 LR = `learning_rate × 0.05` | cosine_with_warmup |
| **梯度累积** | `gradient_accumulation_steps` 步再 step/scheduler，可模拟任意大 batch | 1（可改 2/4/8） |
| **混合精度** | `torch.amp` + `GradScaler`（CUDA 环境自动生效） | 开启 |
| **损失函数** | 可选 `huber / mae / mse` + 标签平滑 + `auxiliary_loss_weight × 平均辅助损失` | huber, δ=0.5, aux_wt=0.15 |
| **早停** | 以 RMSE 为主指标，`min_delta=1e-4`，`patience=5` | 开启 |
| **Backbone 冻结** | 三模态主干可独立冻结，小数据集 / 微调场景加速收敛 | 全部可训 |
| **分层抽样** | 若配置 `train.scene_column`，划分 train/val 时按场景标签 stratify | 关闭（可配） |
| **数据增强**（训练集） | 图像：RandomCrop + Flip + ColorJitter；文本：字符 dropout；音频：噪声 + 音量 + 随机切片 | 开启 |
| **LRU 缓存** | `OrderedDict LRU` 缓存预处理输入，最大条目 = `cache_max_size=10000`，防止 OOM | 开启 |

---

## 安装依赖

```bash
pip install -r requirements.txt
```

核心依赖版本建议：

```
torch>=2.1.0
torchvision>=0.16.0
transformers>=4.36.0
shap>=0.44.0
scikit-learn>=1.3.0
pandas>=2.1
numpy>=1.24
PyYAML>=6.0
Pillow>=10.0
tqdm>=4.66
```

---

## 生成示例数据（支持音频）

```bash
python examples/generate_demo_data.py
```

- 默认生成 `examples/demo_dataset.csv`（**无音频**三模态）和 `examples/demo_dataset_audio.csv`（**四模态 + 40 条示例 WAV** 在 `examples/demo_audio/`）。

---

## 训练模型

### 四模态（推荐，带语音）

```bash
python -m src.aict_eval.train --data examples/demo_dataset_audio.csv --config configs/default.yaml
```

### 三模态（无语音，兼容旧数据）

```yaml
# configs/default.yaml 中修改
train:
  audio_column: null     # 关闭语音列
```

首次运行会自动从 HuggingFace / TorchVision 下载预训练权重：

- `bert-base-chinese`（文本）
- `ResNet18` 或 `ResNet50`（图像）
- `facebook/wav2vec2-base-960h`（语音，可选）

若环境无法访问外网，将 `model.allow_online_model_download` 改为 `false`，代码会**自动走离线回退模式**，不阻塞训练：

- 文本 → 本地 `HashTokenizer + BiLSTM`；
- 图像 → `ResNet18/50` 随机初始化（或使用已有本地缓存）；
- 语音 → 增强版 `StatsAudioEncoder`（梅尔分箱 + 谱/时域统计）。

---

## 常用配置片段

### 选择编码器 / Backbone

```yaml
model:
  image_model_name: "resnet50"            # 可选：resnet18 / resnet50
  audio_backbone_type: "wav2vec2"         # 可选：stats / wav2vec2 / hubert / whisper
  audio_model_name: "facebook/wav2vec2-base-960h"
  text_model_name: "bert-base-chinese"

  # 融合网络规模升级
  tabular_hidden_size: 128                # 结构化编码器隐藏层
  fusion_hidden_size: 256                 # 融合投影维度
  num_attention_heads: 4                  # 融合注意力头
  fusion_layers: 3                        # 跨模态注意力层数
  fusion_ffn_size: 768                    # 融合 FFN 宽度
  use_modality_gating: true               # 模态门控
  use_auxiliary_loss: true                # 辅助回归头
  auxiliary_loss_weight: 0.15             # 辅助损失权重
  dropout: 0.1
  allow_online_model_download: true
```

### 训练超参与稳定性

```yaml
train:
  batch_size: 4
  epochs: 10
  learning_rate: 0.0002                   # 上层网络 LR
  backbone_learning_rate: 0.00002         # 预训练主干 LR（差分）
  weight_decay: 0.0001
  max_grad_norm: 1.0
  gradient_accumulation_steps: 1          # 等效 batch 可做到 batch_size×N
  lr_scheduler_type: "cosine_with_warmup" # 或 linear
  warmup_epochs: 2
  # warmup_ratio: 0.06                    # 与 warmup_epochs 二选一
  min_lr_ratio: 0.05

  loss_type: "huber"                      # 或 mae / mse
  huber_delta: 0.5
  label_smoothing: 0.0                    # 回归向均值收缩，防止过拟合
  enable_augmentation: true               # 训练期三模态数据增强

  mixed_precision: true
  early_stopping_patience: 5
  early_stopping_min_delta: 0.0001

  # 防 OOM LRU 缓存
  cache_preprocessed_inputs: true
  cache_max_size: 10000

  # 小数据集可选冻结某些主干加速 & 防过拟合
  freeze_text_encoder: false
  freeze_image_encoder: false
  freeze_audio_encoder: false

  # 场景分层抽样（若 CSV 有该列）
  scene_column: null                      # 例如 "scene_type"
```

### 去噪与分组去噪

```yaml
train:
  denoise_enabled: true
  denoise_method: "savgol"                # kalman / adaptive_ema / median / ma / savgol / wavelet
  denoise_group_column: "scene_id"        # 按场景分组去噪
  denoise_sort_column: "timestamp"        # 分组内排序列

  # kalman
  kalman_process_variance: 0.0001
  kalman_measurement_variance: 0.01

  # adaptive_ema
  ema_alpha: 0.25
  ema_min_alpha: 0.05
  ema_max_alpha: 0.6
  ema_window: 5

  # savitzky-golay
  sg_window_length: 5
  sg_polyorder: 2

  # haar wavelet
  wavelet_level: 2
  wavelet_mode: "soft"                    # soft / hard

  # median
  median_window: 5
```

### 指标赋权 & 自动 α 搜索

```yaml
train:
  auto_indicator_weight_alpha: true       # 开启 80 组网格搜索
  indicator_weight_alpha: 0.5             # 关闭自动搜索时使用此值
```

### 性能调优（大数据集）

```yaml
train:
  dataloader_num_workers: 2               # 建议 CPU 核数/2
  dataloader_pin_memory: true
  dataloader_persistent_workers: true
  dataloader_prefetch_factor: 2

  batch_size: 8
  gradient_accumulation_steps: 2          # 等效 batch = 16（显存友好）
```

---

## 真实课题数据替换方式

将真实数据整理成 CSV，字段约定参考 [DATASET_FIELD_SPEC.md](file:///c:/Users/lenovo/Desktop/AICT/DATASET_FIELD_SPEC.md)。**最低要求**：

- `review_text`：游客评论、访谈转写、问卷开放题反馈等
- `image_path`：场景图像路径（本地）
- `audio_path`：语音文件路径（WAV；关闭语音模式时可省略）
- `target_score`：专家综合分 / 问卷总分 / 体系总分
- **其余数值列自动作为结构化特征**，例如：
  - `tech_empowerment` 技术赋能效能
  - `visitor_experience` 游客感知体验
  - `cultural_value` 文化价值传播
  - `economic_social_gain` 经济社会增值
  - `interaction_count` 互动次数
  - `stay_duration` 停留时长
  - 其他：`ai_guide_usage_rate / qa_success_rate / revisit_intention_score / device_response_time / complaint_rate` 等

> 采集标准与数据需求详细说明见 [DATA_REQUIREMENTS_STANDARD.md](file:///c:/Users/lenovo/Desktop/AICT/DATA_REQUIREMENTS_STANDARD.md)，问卷评分与标签换算见 [QUESTIONNAIRE_SCORING_SYSTEM.md](file:///c:/Users/lenovo/Desktop/AICT/QUESTIONNAIRE_SCORING_SYSTEM.md)。

---

## 问卷流水线对接（一键从问卷星 CSV → AICT v2 训练闭环）

针对「问卷星 20 题量表 + 1 道开放题」的典型 AI+文旅 成效评价问卷场景，主仓库已经内置完整脚本，**无需额外子目录**，直接使用以下两种入口即可。
详见独立文档 [QUESTIONNAIRE_PIPELINE.md](file:///c:/Users/lenovo/Desktop/AICT/QUESTIONNAIRE_PIPELINE.md)。

### 快速使用：仅跑问卷打分与结构化拆分（无需模型训练）

```powershell
python scripts/process_questionnaire.py `
  --input "aict_output\scored_full.csv" `
  --output-dir "outputs\questionnaire_v2"
```

内置默认配置，`--config` 完全可省略；同时兼容：
- 问卷星原始导出 CSV（中文长题列名）
- 旧版 `aict_output/scored_full.csv` 中间产物
- 旧版扁平 `aict_output/column_mapping.json`
- 旧版 `aict_output/analysis_report.json` 字段名（报告同时输出新旧两套字段）

### 推荐：一键全闭环（问卷 → 结构化 → AICT v2 模型训练 → 测试集预测）

```powershell
python scripts/questionnaire_to_aict_pipeline.py `
  --input "aict_output\scored_full.csv" `
  --questionnaire-output-dir "outputs\questionnaire_pipeline" `
  --questionnaire-config "configs\questionnaire_pipeline.json" `
  --model-config "configs\questionnaire_model.yaml" `
  --enable-ai   # 可选：需设置 $env:DEEPSEEK_API_KEY
```

最终在 `outputs\questionnaire_pipeline\pipeline_summary.json` 汇总所有输入路径、输出路径和训练指标，便于 CI 审计。

### 问卷流水线核心产物与主系统对接字段

| 问卷输出文件 | 下游 AICT v2 对接对象 | 关键字段对齐 |
| --- | --- | --- |
| `aict_dataset.csv` | `src/aict_eval/dataset.py` 的 `MultimodalDataset` | 严格对齐 README "最低要求" 字段：`review_text / image_path / audio_path / target_score`；额外追加四维分 + `quality_score` + AI 分析 4 列，共 15 列（数值部分自动走 6 种去噪 + 4 源客观赋权 + 80 组 α 网格搜索）。 |
| `train_dataset.csv` | `python -m src.aict_eval.train --data <this>` | 不含 `dataset_split` 列，直接符合 README "训练数据 CSV 格式"约定。 |
| `test_dataset.csv` | `python -m src.aict_eval.infer --data <this>` | 同上。 |
| `analysis_report.json` | `src/aict_eval/weights.py` + `filters.py` 先验参考 | 输出四维分 Cronbach α / 质量罚分统计量 / target_score 全量分位数，供调参决策。 |

### 标签泄漏防护策略（默认安全）

- **默认排除 q1~q20 题项本身**（防止 20 道题直接线性解出 target_score）；
- **默认保留四维分 + quality_score + ai_* 结构化特征**（聚合级特征不泄漏）；
- 若需要更保守，在 [questionnaire_pipeline.json](file:///c:/Users/lenovo/Desktop/AICT/configs/questionnaire_pipeline.json) 中设置 `include_derived_dimensions: false`。

---

## 训练输出

输出目录默认为 `outputs/`（可在配置中修改）：

| 文件 | 说明 |
|---|---|
| `multimodal_evaluator.pt` | 最终多模态模型权重（PyTorch state_dict） |
| `indicator_weights.json` | GRA+CV+皮尔逊+熵 四源融合后的结构化指标权重 |
| `metrics.json` | 验证集指标：`loss / mae / rmse / r2 / train_loss / lr / epoch` |
| `preprocess_artifacts.json` | 结构化列名 + StandardScaler 的 mean/scale（保证推理预处理一致） |
| `shap_feature_importance.csv` | SHAP 逐样本特征重要性（含 feature_value / shap_value / sample_id） |
| `report.json` | 成效诊断报告（权重表 / SHAP TopK / 跨模态注意力统计 / 去噪配置等） |
| `report.md` | Markdown 版成效诊断报告，便于直接粘贴到课题材料 |

---

## 独立推理

训练完成后，加载训练工件对新 CSV 做批量预测：

```bash
python -m src.aict_eval.infer \
  --data examples/demo_dataset_audio.csv \
  --config configs/default.yaml \
  --model-dir outputs \
  --output outputs/predictions.csv
```

输出 CSV 中会包含每行样本的 `predicted_score`，可直接对比 `target_score` 做后验分析。

---

## 适合下一步扩展的方向

1. 接入真实评论语料与景区监测日志，构建"分层抽样 + 多批次对照实验"验证指标鲁棒性。
2. 将图像编码器替换为 `CLIP` / `Swin Transformer`，进一步利用图文强对齐先验。
3. 升级语音编码器为 `Whisper large-v3 / Paraformer` 做 ASR → 文本特征，再与原始声学特征双塔融合。
4. 引入视频帧序列、生理传感器（心率/皮电/眼动）等时序模态，升级 `CrossModalBlock` 为时序注意力。
5. 将回归头扩展为四维分项评分输出（四个一级指标各出一个头），支持"综合分 + 分项分"联合训练与多任务学习。
6. 引入对抗训练（FreeAT / SMART）提升文本编码器抗噪声能力，适配真实评论中的错别字与口语化表达。
