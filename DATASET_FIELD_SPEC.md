# AICT 数据集字段说明文档

本文档用于说明 AICT 多模态成效评价模型所需的数据集字段、数据格式与采集要求。

当前模型支持以下四类输入模态：

- 文本模态：游客评论、访谈转写、问卷开放题反馈
- 图像模态：景区/展区照片、活动现场图像、监控抽帧
- 语音模态：讲解录音、游客语音反馈、现场语音采集
- 结构化模态：互动次数、停留时长、技术赋能指标、文化传播指标等数值特征

每一行数据代表一个样本。一个样本通常对应某个景区、博物馆、展区、活动场景或某个时间窗口下的一次综合观测。

## 1. 推荐字段清单

| 字段名 | 类型 | 是否必填 | 说明 |
| --- | --- | --- | --- |
| `review_text` | string | 是 | 文本内容，作为文本模态输入；允许空字符串，空值时走 HashTokenizer 占位。 |
| `image_path` | string | 若 `train.image_column != null` 必填 | 图像文件本地路径。若问卷场景无图像，可将配置设为 `train.image_column: null`（问卷默认就是），此时 CSV 中该列可写固定字符串 `"PLACEHOLDER_IMAGE"`，也可以完全不写该列。 |
| `audio_path` | string | 若 `train.audio_column != null` 必填 | 音频文件本地路径。同 image_column；问卷默认场景：列值 `"PLACEHOLDER_AUDIO"` 或直接省略不提供列。 |
| `target_score` | float | 是 | 样本监督标签，即综合成效评分，百分制 0~100 推荐。 |
| 其他数值列 | int / float | 建议提供 | 自动作为结构化指标输入；**自动剔除整列 NaN / std<=0 / 非有限列**，避免 StandardScaler 除 0 生成 NaN 权重。 |

说明：

- 若当前阶段没有图像 / 语音数据，可将配置文件中的 `train.image_column: null` / `train.audio_column: null`，此时 `image_path/audio_path` 列可完全省略，代码自动走零占位 tensor（3×224×224 + 梅尔统计占位）。不会再抛 FileNotFoundError / TypeError。
- 除 `review_text`、`image_path`、`audio_path`、`target_score` 外，其余所有数值型字段会自动识别为结构化特征；非数值列（如 scene_type、dataset_split）会被自动跳过，不会作为模型输入，但可用于分层抽样、分组去噪等。

## 2. 核心字段详细说明

### 2.1 `review_text`

- 类型：`string`
- 含义：反映游客感知、文化理解、服务体验、技术接受度等内容的文本信息
- 数据来源建议：
  - 游客评论
  - 访谈记录转写
  - 问卷开放题
  - 平台评价文本
- 填写要求：
  - 一条样本对应一段文本
  - 尽量避免完全为空
  - 尽量保留原始语义，不建议过度人工改写

示例：

```text
数字导览讲解清晰，互动体验自然，文化内容更容易理解。
```

### 2.2 `image_path`

- 类型：`string`（允许列值为 `"PLACEHOLDER_IMAGE"`，表示"问卷专用：当前样本无图像；当 `train.image_column: null` 时，列可完全省略，此时 dataset.py 会自动返回 `zeros(3,224,224)` 占位，不会抛 FileNotFoundError。
- 含义：与该样本对应的图像文件路径（可选）
- 数据来源建议：
  - 景区现场照片
  - 展馆展陈照片
  - 游客活动照片
  - 监控抽帧图像
- 填写要求：
  - 当列存在时，路径需真实存在；否则走占位符模式
  - 当前实现按单张图像读取
  - 建议使用清晰、内容相关的图片

示例：

```text
C:\Users\lenovo\Desktop\AICT\examples\demo_images\scene_000.png
PLACEHOLDER_IMAGE
```

### 2.3 `audio_path`

- 类型：`string`（允许值 `"PLACEHOLDER_AUDIO"`，表示问卷专用：样本无语音文件；当 `train.audio_column: null` 时，列可完全省略，自动走梅尔统计占位。
- 含义：与该样本对应的语音文件路径（可选）
- 数据来源建议：
  - 游客语音反馈
  - 导览讲解录音
  - 问答交互音频
  - 现场环境中的有效语音片段
- 填写要求：
  - 当列存在时，路径需真实存在；否则走占位符模式
  - 当前代码建议使用 `WAV` 格式
  - 建议一条样本对应一段主要语音内容
  - 若存在长音频，建议按场景切分后再入库

示例：

```text
C:\Users\lenovo\Desktop\AICT\examples\demo_audio\scene_000.wav
PLACEHOLDER_AUDIO
```

### 2.4 `target_score`

- 类型：`float`
- 含义：模型训练的监督标签，表示该样本的综合成效水平
- 推荐来源：
  - 专家打分
  - 多位评审平均分
  - 问卷综合评分
  - 现有评价体系计算后的总分
- 填写要求：
  - 每条样本必须有标签
  - 同一批数据应采用统一评分标准
  - 推荐使用连续值，如 `0-100`

示例：

```text
75.30
```

## 3. 结构化指标字段建议

除核心字段外，建议补充能够反映文旅应用成效的数值型指标。以下字段只是示例，可根据课题实际替换。

| 字段名 | 类型 | 一级维度建议 | 含义说明 |
| --- | --- | --- | --- |
| `tech_empowerment` | float | 技术赋能效能 | AI 技术对服务、管理、讲解、推荐等环节的支持程度 |
| `visitor_experience` | float | 游客感知体验 | 游客满意度、沉浸感、易用性等综合表现 |
| `cultural_value` | float | 文化价值传播 | 文化理解、知识获取、内容传播效果 |
| `economic_social_gain` | float | 经济社会增值 | 消费带动、社会传播、品牌影响等收益表现 |
| `interaction_count` | int | 游客感知体验 | 游客与系统/设备交互次数 |
| `stay_duration` | float | 游客感知体验 | 游客停留时长，单位可自定义但需统一 |

还可以继续扩展以下字段：

- `ai_guide_usage_rate`
- `qa_success_rate`
- `revisit_intention_score`
- `content_share_count`
- `device_response_time`
- `complaint_rate`
- `cultural_topic_hit_rate`

要求：

- 必须是数值型
- 缺失值要提前处理
- 同一字段的量纲和单位要统一

> **关于去噪**：结构化指标可启用模型内置去噪（`train.denoise_enabled: true`），支持 6 种算法：`卡尔曼滤波 / 自适应 EMA / 中值滤波 / 移动平均 / Savitzky-Golay 多项式平滑 / Haar 小波软阈值`。若数据是"同一对象在多个时间窗口"的观测序列（如同一景区多天指标），可同时设置 `denoise_group_column`（如对象 ID）+ `denoise_sort_column`（如时间戳），实现分组时序去噪，更符合文旅项目长期监测场景。

> **关于指标赋权**：所有结构化指标会通过 `灰色关联分析 GRA + 变异系数 CV + 皮尔逊相关 + 熵权法` 四源融合自动计算客观权重（`auto_indicator_weight_alpha: true` 时还会做 5×4×4=80 组网格搜索），因此**不需要人工对指标做手动加权**，但建议保证各指标同向化（数值越大越好）并统一量纲。

## 4. 一条样本应如何对应现实对象

建议采用以下任一组织方式：

### 方式一：按单次游客体验组织

一条样本对应一位游客在一次参观过程中的综合反馈。

适用场景：

- 小规模调研
- 问卷与访谈结合采集
- 单次导览服务评估

### 方式二：按场景/展区组织

一条样本对应某个展区、景点或服务场景在某个时间窗口内的综合状态。

适用场景：

- 景区日常监测
- 博物馆展区成效评估
- 多源数据汇总分析

### 方式三：按活动组织

一条样本对应一场活动、一次主题展演或一个数字文旅项目的综合表现。

适用场景：

- 节庆活动
- 文旅融合项目
- 主题线路或专题展览评价

## 5. 采集建议

### 文本数据

- 保留原始反馈文本
- 尽量避免全是模板化句子
- 可以记录评论时间、来源平台等辅助信息，但这些字段若为非数值型默认不会进入结构化特征
- **离线回退机制**：若无法下载 BERT 等预训练模型，代码会自动回退到本地 `HashTokenizer + 双层双向 BiLSTM + 自注意力池化` 的轻量中文编码器，保证在无外网环境也能训练

### 图像数据

- 图像内容要与样本场景对应
- 尽量避免模糊、纯黑、纯白、无效图
- 文件建议统一放在固定目录下，便于批量管理

### 语音数据

- 建议使用清晰可辨的普通话讲解、游客语音反馈或问答音频
- 当前版本优先支持 `WAV`
- 建议控制单条音频时长在几秒到几十秒内，后续可按需要进一步升级编码方式
- **离线回退机制**：若无法下载 wav2vec2/HuBERT/Whisper 预训练模型，代码会自动回退到增强版统计编码（STFT → 64 维梅尔分箱统计 + 谱质心/谱滚降/谱带宽/谱平坦度 + ZCR/RMS/偏度/峰度 等时域特征 → 3 层 MLP 投影），不阻塞训练

### 结构化数据

- 优先选取与你的四个一级指标相关的量化特征
- 先做字段统一、单位统一、缺失值清理
- 如同一对象存在时间序列数据，可后续启用去噪功能
- **建议按"越大越好"做同向化处理**，便于客观赋权的解释性（GRA/皮尔逊/熵权法对方向不敏感，但可视化报告中"权重高即影响大"的语义更自然）
- **强烈建议保留 scene_id / timestamp 等辅助列**：用于 `scene_column` 分层抽样 train/val，以及 `denoise_group_column + denoise_sort_column` 做分组时序去噪（即使不是数值列也可以保留在 CSV 中，模型会自动跳过非数值列作为特征）

### 标签数据

- 标签标准必须统一
- 若采用多人评分，建议保留原始评分并计算平均分或一致性指标
- 若没有现成标签，可先构建人工评价规则生成 `target_score`

## 6. 推荐 CSV 表头模板

```csv
review_text,image_path,audio_path,tech_empowerment,visitor_experience,cultural_value,economic_social_gain,interaction_count,stay_duration,target_score
```

这也是当前示例文件 [examples/demo_dataset_audio.csv](file:///c:/Users/lenovo/Desktop/AICT/examples/demo_dataset_audio.csv) 使用的字段格式。

## 7. 示例数据行

```csv
数字导览讲解清晰，文化故事更容易理解，沉浸感很强。,C:\data\images\scene_001.png,C:\data\audio\scene_001.wav,75.77,62.51,77.45,92.46,27,49.15,75.30
```

## 8. 与当前代码的对应关系

- 文本列配置：`train.text_column`
- 图像列配置：`train.image_column`
- 语音列配置：`train.audio_column`
- 标签列配置：`train.target_column`

当前默认配置文件见 [configs/default.yaml](file:///c:/Users/lenovo/Desktop/AICT/configs/default.yaml)。

## 9. 最低可用数据要求（两种版本：三模态 / 问卷纯文本+结构化）

### 9.1 完整三 / 四模态版本（若有图像或语音）

至少需要：

- 1 列文本 `review_text`
- 1 列图像路径 `image_path`（或 `train.image_column: null` 直接关闭图像列）
- 1 列语音路径 `audio_path`（或 `train.audio_column: null` 直接关闭语音列）
- 1 列目标分数 `target_score`
- 至少 1 列数值型结构化指标
- （推荐）1 列 scene 标签用于 train/val 分层抽样，1 列 timestamp 用于时序去噪（若你有同一对象的时间序列观测）

### 9.2 问卷场景：纯文本 + 结构化双模态（推荐，与默认 questionnaire_model.yaml 对齐）

问卷场景下**不需要任何图像/语音文件**，CSV 仅需以下 10 列（即 `process_questionnaire.py` 输出 `aict_dataset.csv` 默认列）：

```csv
review_text,image_path,audio_path,duration_seconds,has_meaningful_feedback,ai_has_discomfort,ai_sentiment_score,ai_confidence,target_score,dataset_split
```

- `image_path` / `audio_path` 值固定为 `"PLACEHOLDER_IMAGE"` 与 `"PLACEHOLDER_AUDIO"`（或**直接不提供这两列**，并在模型配置中写：
  ```yaml
  train:
    image_column: null
    audio_column: null
  ```
  此时 dataset.py 自动返回 224×224 0 矩阵 + 梅尔统计占位，不再强要求文件存在。
- 结构化特征自动识别出 5 列：`duration_seconds / has_meaningful_feedback / ai_has_discomfort / ai_sentiment_score / ai_confidence`；**已默认剔除四维度分与 quality_score，防止标签泄漏**。

> 若暂时没有图像或语音数据，可关闭对应列配置，直接使用文本 + 结构化双模态即可跑通端到端流水线与 HTML 报告。

## 10. 面向课题的推荐采集方案

结合“文化和旅游数智化研究”课题，推荐优先采集以下内容：

- 文本：游客评论、访谈转写、问卷开放反馈
- 图像：景区现场图、展陈图、互动设备使用现场图
- 语音：讲解录音、游客口述评价、智能问答语音片段
- 结构化指标：
  - 停留时长
  - 互动次数
  - AI 导览使用率
  - 问答成功率
  - 文化内容触达率
  - 二次传播量
  - 消费转化指标
  - 满意度评分
- 标签：
  - 专家综合评价分
  - 问卷总分
  - 人工建立的成效综合分

## 11. 备注

- 当前实现对语音采用轻量方式接入，适合先完成多模态建模与课题验证
- 若后续你要做更强的语音语义建模，可以进一步升级为 `Wav2Vec2`、`HuBERT` 或音频 Transformer 编码器
- 若你准备采集真实课题数据，建议先按本文档做一版标准化数据模板，再批量填充
