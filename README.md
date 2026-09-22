# Sleep AutoML Agent

面向一维生理信号与睡眠监测场景的 Agent 学习项目。

第一版聚焦 Apple Watch 加速度数据和 PSG 睡眠标签，完成 Wake/Sleep 二分类的最小闭环：

1. 读取加速度 TXT 与睡眠标签 TXT。
2. 按 30 秒 epoch 对齐数据。
3. 提取窗口级加速度特征。
4. 训练传统机器学习二分类模型。
5. 输出指标、报告与版本学习记录。
6. 后续逐步加入 DeepSeek、LangChain、LangGraph 与 C/C++ 部署探索。

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.experiment_runner --config configs/experiment.yaml
```

## v0.2.0 DeepSeek Report Demo

`v0.2.0` 在 `v0.1.0` 的实验结果基础上接入 DeepSeek，用于完成三个学习目标：

1. 调用 DeepSeek 完成简单问答。
2. 读取已有特征表和指标，生成数据分析与实验建议。
3. 输出 Markdown 中间稿和 Word 实验报告。

```powershell
pip install -r requirements.txt
python -m src.experiment_runner --config configs/experiment.yaml
python -m src.v020_report_runner --config configs/experiment.yaml --question "请用一句话解释 Wake/Sleep 二分类实验的目标"
```

如果只想验证本地流程，不调用 DeepSeek：

```powershell
python -m src.v020_report_runner --config configs/experiment.yaml --question "请用一句话解释 Wake/Sleep 二分类实验的目标" --offline
```

## Project Layout

```text
configs/       实验配置
data/          本地数据与数据说明
src/           纯 Python 机器学习流水线
agents/        DeepSeek / LangChain / LangGraph 相关 Agent
docs/          需求、技术路线、学习记录
experiments/   每个版本或实验的配置和结果
reports/       自动生成的实验报告
tests/         基础测试
```

## Version Plan

- `v0.1.0`: 纯 Python 最小闭环，完成 Wake/Sleep baseline。
- `v0.2.0`: 接入 DeepSeek，生成数据分析和实验报告。
- `v0.3.0`: 使用 LangChain 封装工具调用。
- `v0.4.0`: 使用 LangGraph 编排完整工作流。
- `v1.0.0`: 形成可演示的睡眠信号 AutoML Agent 原型。

## Python Module Flow

```text
experiment_runner.py
  -> data_loader.py
  -> dataset_builder.py
       -> data_loader.py
       -> label_loader.py
       -> windowing.py
       -> feature_extraction.py
  -> model_train.py
  -> model_eval.py
  -> report_generator.py
```

`v0.2.0` 的 DeepSeek 报告流程：

```text
v020_report_runner.py
  -> 读取 configs/experiment.yaml
  -> 读取 experiments/v0.1.0_baseline/metrics.json
  -> dataset_summary.py
       -> 优先读取 experiments/v0.1.0_baseline/features.csv
       -> 若不存在，则调用 dataset_builder.py 重新构建
  -> agents/deepseek_client.py
  -> agents/data_analysis_agent.py
  -> agents/experiment_advisor_agent.py
  -> agents/result_report_agent.py
  -> docx_report.py
  -> reports/v0.2.0_deepseek_experiment_report.md
  -> reports/v0.2.0_deepseek_experiment_report.docx
```

## Python Modules

#### Baseline Pipeline

`src/data_loader.py`

负责发现数据文件和读取原始 TXT。它会查找成对的 `*_acceleration.txt` 和 `*_labeled_sleep.txt`，并读取成 DataFrame。

`src/label_loader.py`

负责把 PSG 睡眠分期映射成二分类标签：

```text
0 -> Wake
1/2/3/5 -> Sleep
```

`src/windowing.py`

负责按 30 秒标签 epoch 切分加速度窗口。比如标签从 `time=300` 开始，就取 `[300, 330)` 秒内的加速度数据。

`src/feature_extraction.py`

负责从每个 30 秒窗口提取特征，例如均值、标准差、最大值、最小值、RMS、能量、三轴合加速度模长等。

`src/dataset_builder.py`

负责把“读取数据、标签映射、窗口切分、特征提取”组合起来，最终生成模型训练用的窗口级特征表。

`src/model_train.py`

负责模型训练。目前支持：

```text
random_forest
logistic_regression
```

默认按被试切分训练集和测试集，这比随机切分更接近真实使用场景。

`src/model_eval.py`

负责计算模型评估指标，包括 Accuracy、Macro F1、Sleep F1、混淆矩阵和分类报告。

`src/report_generator.py`

负责根据评估结果生成 Markdown 实验报告。

`src/dataset_summary.py`

负责把窗口级 `features.csv` 压缩成适合喂给 LLM 的结构化摘要，例如被试数、窗口数、Wake/Sleep 分布、特征列和每个被试窗口数。它的定位是“事实统计层”，不要让 LLM 直接猜这些数。

`src/docx_report.py`

负责把 Markdown 报告转换为 `.docx`。当前只支持学习 Demo 所需的标题、列表、段落和代码块；后续如果报告格式更复杂，可以替换为更完整的 Markdown 到 Word 渲染方案。

`src/v020_report_runner.py`

`v0.2.0` 的命令行入口。它不重新训练模型，而是读取 `v0.1.0` 生成的 `features.csv` 和 `metrics.json`，调用 DeepSeek 完成问答、数据分析、实验建议和报告正文生成，最后输出 Markdown 与 Word 报告。

`src/__init__.py`

标记 `src` 是一个 Python 包，并记录当前版本号。

### Agent Modules

`agents/deepseek_client.py`

DeepSeek API 客户端的最小封装。它会从 `.env` 读取 `DEEPSEEK_API_KEY`，默认模型为 `deepseek-flash`，并保留 OpenAI-compatible 的 messages 调用格式，方便后续切换 OpenAI 或迁移到 LangChain。

`agents/data_analysis_agent.py`

生成“数据分析 Agent”的提示词，用于分析数据规模、类别分布、风险点。

`agents/experiment_advisor_agent.py`

生成“实验建议 Agent”的提示词，用于根据指标建议下一轮实验。

`agents/result_report_agent.py`

生成“报告 Agent”的提示词，用于根据配置和指标写实验总结。

## Direct API vs LangChain

当前 `v0.2.0` 先直接调用 DeepSeek 的 OpenAI-compatible API。这样你能先看清楚最基础的请求结构：`messages`、`model`、`temperature`、`headers` 和返回结果。

`v0.3.0` 再切到 LangChain 更合适。LangChain 会把模型调用、PromptTemplate、工具调用、结构化输出、重试和链式组合封装起来，适合做更复杂的 Agent，但对新手来说一上来就用会把“API 原理”和“框架抽象”混在一起。

### Tests

`tests/test_smoke.py`

当前只有一个最小测试，验证睡眠分期到 Wake/Sleep 二分类的映射是否正确。
