# 技术路线

## 阶段 1：纯 Python 最小闭环

目标：不用复杂 Agent，先跑通 Wake/Sleep 二分类。

模块：

- `data_loader.py`: 发现被试、读取加速度与标签。
- `windowing.py`: 按 30 秒标签切分窗口。
- `feature_extraction.py`: 提取窗口级时域特征。
- `dataset_builder.py`: 生成特征矩阵和标签。
- `model_train.py`: 训练 baseline 模型。
- `model_eval.py`: 输出分类指标。
- `report_generator.py`: 生成 Markdown 报告。
- `experiment_runner.py`: 串联完整流程。

## 阶段 2：DeepSeek Agent

目标：接入 DeepSeek，让 Agent 做分析和报告，不直接控制底层算法执行。

Agent：

- `data_analysis_agent.py`
- `experiment_advisor_agent.py`
- `result_report_agent.py`

新增工程模块：

- `dataset_summary.py`: 将 `features.csv` 压缩为适合 LLM 使用的数据摘要。
- `docx_report.py`: 将 Markdown 报告转为 Word 文档。
- `v020_report_runner.py`: 串联配置、指标、数据摘要、DeepSeek 调用和报告输出。

## 阶段 3：LangChain

目标：用 LangChain 管理 LLM 调用、工具调用、Prompt 模板和结构化输出。

## 阶段 4：LangGraph

目标：把完整机器学习流程编排成可追踪、可恢复、可扩展的图工作流。

## 阶段 5：部署探索

目标：调研并验证从 Python baseline 到 C/C++ 或边缘部署的路径。

候选方向：

- ONNX
- m2cgen
- Treelite
- 手写简单模型 C 推理
- Python/C 结果一致性验证
