# AICT 问卷处理模块交接说明

## 本次交付

该模块用于把 20 道 Likert 评分题和第 21 道开放反馈题自动转换为：

- 四个一级维度与综合成效分
- 问卷质量标记和人工复核队列
- 训练集、测试集和 AICT 模型输入 CSV
- DeepSeek 开放文本结构化标注
- 可审计的 JSON 分析报告

## 运行环境

- Python 3.10+
- `pandas`
- `numpy`
- AICT 主项目训练还需要 `requirements.txt` 中的依赖

## 首次试跑

```powershell
python scripts/process_questionnaire.py `
  --input "问卷导出.csv" `
  --output-dir "outputs/questionnaire_v1"
```

启用 DeepSeek：

```powershell
$env:DEEPSEEK_API_KEY="在本机配置，不要写入仓库"
python scripts/process_questionnaire.py `
  --input "问卷导出.csv" `
  --output-dir "outputs/questionnaire_v1" `
  --enable-ai
```

## 合并前必须确认

问卷选项按“5分、4分、3分、2分、1分”排列，但现有二次产物只保留数字 `1-5`。需从问卷星原始导出或后台设置确认：

- 若数字就是实际分值，使用 `input_value_order: direct`。
- 若数字代表选项位置，使用 `input_value_order: reverse_order`。

当前 DeepSeek 文本结果与量表分数呈反向关系，较支持 `reverse_order`，但不能在缺少原始导出的情况下直接定案。

## 数据安全

- API Key 仅从 `DEEPSEEK_API_KEY` 环境变量读取。
- 请求只发送脱敏后的有效开放题文本。
- 不向 DeepSeek 发送 IP、问卷编号、提交时间和量表答案。
- 压缩包和 Git 提交不得包含原始问卷、处理结果、缓存或 API Key。

## 推荐 Git 流程

1. 从最新 `main` 创建 `codex/questionnaire-pipeline` 分支。
2. 只提交脚本、配置、说明文档和必要的图片可选兼容代码。
3. 提交 Pull Request，请学长确认评分编码后再合并。
4. 原始问卷和处理结果通过团队约定的受控存储交付，不上传公开仓库。
