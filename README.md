# AICT: AI+文旅应用成效智能评价算法原型

这个原型围绕课题“基于多模态数据融合的 AI+文旅应用成效智能评价模型研究”实现了以下核心能力：

1. 指标筛选与赋权：`GRA + CV`
2. 结构化指标去噪：`卡尔曼滤波 / 自适应 EMA（可选）`
3. 多模态成效评价：`中文文本编码 + 图像编码 + 语音编码 + 结构化指标融合（跨模态注意力 + 动态门控）`
4. 可解释性分析与诊断报告：`SHAP + 成效诊断报告（JSON/Markdown）`

## 目录结构

```text
AICT/
├─ configs/default.yaml
├─ examples/generate_demo_data.py
├─ requirements.txt
└─ src/aict_eval/
   ├─ config.py
   ├─ dataset.py
   ├─ filters.py
   ├─ explain.py
   ├─ model.py
   ├─ report.py
   ├─ train.py
   └─ weights.py
```

## 算法设计

### 1. 指标赋权

- 使用灰色关联分析计算指标与目标成效之间的关联强度
- 使用变异系数法计算指标客观权重
- 将两者融合，得到四维与多项二级指标的综合权重

### 2. 多模态模型

- 文本模态：`bert-base-chinese`
- 图像模态：`torchvision` 预训练 `ResNet18`
- 语音模态：`WAV` 波形经频谱统计编码后映射到统一融合空间
- 结构化模态：游客停留时长、互动次数、技术效能、文化传播等数值指标
- 融合方式：将文本、图像、语音、结构化特征投影到同一隐空间后，通过多层跨模态注意力做特征对齐，并使用动态门控对模态权重进行自适应调整，再进行回归输出
- 支持在线下载预训练模型；若下载失败，则自动回退到轻量哈希分词器和本地文本编码器，保证算法在无外网环境也能运行

### 3. 可解释性

- 训练完成后，基于结构化指标拟合一个代理模型
- 使用 `SHAP` 输出影响成效分值的关键指标排序
- 适合直接转成课题报告中的“影响因子分析”和“优化建议”

### 4. 结构化指标去噪（可选）

- 支持对结构化数值指标进行去噪预处理（卡尔曼滤波 / 自适应 EMA）
- 可通过配置启用；并可选按 `group_column + sort_column` 做“同一对象的时序去噪”

## 安装依赖

```bash
pip install -r requirements.txt
```

## 生成示例数据

```bash
python examples/generate_demo_data.py
```

## 训练模型

```bash
python -m src.aict_eval.train --data examples/demo_dataset.csv --config configs/default.yaml
```

首次运行会自动从网上下载预训练模型权重，包括：

- `bert-base-chinese`
- `ResNet18` 官方预训练参数

当前默认配置已启用在线下载；如果当前环境无法访问外网，代码会自动走离线回退模式，不阻塞训练。需要强制关闭在线下载时，将 `configs/default.yaml` 中的 `allow_online_model_download` 改为 `false`。

可选能力可在 `configs/default.yaml` 中开启，例如：

```yaml
train:
  denoise_enabled: true
  denoise_method: "kalman"  # 或 "adaptive_ema"
  auto_indicator_weight_alpha: true
```

## 真实课题数据替换方式

将真实数据整理成 CSV，并至少包含以下字段：

- `review_text`：游客评论、访谈文本或问卷主观反馈
- `image_path`：场景图像、监控抽帧图或展区图像路径
- `audio_path`：语音讲解录音、游客语音反馈或现场采集的 `WAV` 音频路径
- `target_score`：专家评分、人工综合评价分或问卷总分
- 其余数值列：自动作为结构化指标输入，例如：
  - `tech_empowerment`
  - `visitor_experience`
  - `cultural_value`
  - `economic_social_gain`
  - `interaction_count`
  - `stay_duration`

## 训练输出

输出目录默认为 `outputs/`，包括：

- `multimodal_evaluator.pt`：模型权重
- `indicator_weights.json`：GRA+CV 指标权重
- `metrics.json`：验证指标
- `shap_feature_importance.csv`：SHAP 重要性排序
- `report.json`：成效诊断报告（指标权重、SHAP 关键特征、跨模态注意力统计、去噪配置等）
- `report.md`：成效诊断报告（便于直接粘贴到课题材料）

如需兼容旧版无语音数据集，可将 `configs/default.yaml` 中的 `train.audio_column` 设为 `null`，模型会自动退回文本 + 图像 + 结构化三模态。

## 适合下一步扩展的方向

1. 接入真实评论语料与景区日志
2. 将图像编码器替换为 `CLIP` 或 `Swin Transformer`
3. 接入视频帧序列与生理传感器时序数据
4. 将输出扩展为四维分项评分，而不只是总分回归
