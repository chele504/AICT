# AICT 问卷自动化处理流程

本流程把问卷星导出的 20 道评分题和 1 道开放题，转换为可审计的统计数据、质量复核队列和 AICT 模型数据。

## 评分口径

- `q1-q5`：技术赋能效能
- `q6-q10`：游客感知体验
- `q11-q15`：文化价值传播
- `q16-q20`：经济社会增值
- `q21`：开放反馈，进入 `review_text`

各维度按题项平均分换算为百分制，再按 `0.25 / 0.30 / 0.25 / 0.20` 加权形成 `target_score`。

`configs/questionnaire_pipeline.json` 中的 `scoring.input_value_order` 控制问卷星数字解释：

- `direct`：导出值 `1-5` 就是实际分数。
- `reverse_order`：导出值是选项位置，问卷按“5分到1分”排列，使用 `实际分数 = 6 - 导出值`。

在问卷星后台确认编码方式前，不要把任一模式的结果作为最终训练标签。

## 基础运行

在项目根目录运行：

```powershell
python scripts/process_questionnaire.py `
  --input "C:\path\to\questionnaire.csv" `
  --output-dir "outputs\questionnaire_v1"
```

不配置 API Key 时，评分、质量检查、数据拆分和本地关键词分析仍会完整执行。

## 启用 DeepSeek

API Key 只从环境变量读取，不写进代码或配置文件：

```powershell
$env:DEEPSEEK_API_KEY="你的API Key"
python scripts/process_questionnaire.py `
  --input "C:\path\to\questionnaire.csv" `
  --output-dir "outputs\questionnaire_v1" `
  --enable-ai
```

DeepSeek 只接收脱敏后的有效开放题文本，不会接收 IP、问卷编号、提交时间或量表答案。相同文本会使用 `ai_cache.jsonl` 缓存，避免重复调用和重复费用。

默认调用 `deepseek-v4-flash`。当前配置启用思考模式并使用 `high` 推理强度；API 地址、模型名、思考开关、推理强度、超时、重试和调用间隔均可在 `configs/questionnaire_pipeline.json` 中调整。

## 输出文件

| 文件 | 用途 |
| --- | --- |
| `questionnaire_analysis.csv` | 完整题项、四维得分、总分、质量标记和文本标注 |
| `aict_dataset.csv` | 默认排除题项和派生维度，降低标签泄漏风险 |
| `train_dataset.csv` | 有效样本训练集 |
| `test_dataset.csv` | 有效样本测试集 |
| `review_queue.csv` | 需要人工复核的问卷 |
| `column_mapping.json` | 原始列到 `q1-q20` 的映射和评分口径 |
| `analysis_report.json` | 样本、分数、质量、信度和开放题汇总 |

## 质量标记

脚本会标记答题时间过短、全选同项、个人答案方差过低、重复答案向量、重复 IP、总体题与前 19 题明显矛盾等情况。标记只表示需要复核，不代表自动认定问卷造假。

## 模型使用注意

若 `target_score` 是由 `q1-q20` 和四个维度计算得到，模型输入中不应再次包含这些字段，否则等于把答案直接交给模型。默认配置因此将它们保留在统计文件中，但从 `aict_dataset.csv` 中排除。

当前问卷数据没有逐份对应的图片和音频，可使用问卷专用配置关闭这两个模态：

```powershell
python -m src.aict_eval.train `
  --data "C:\path\to\outputs\train_dataset.csv" `
  --config configs/questionnaire_model.yaml
```
