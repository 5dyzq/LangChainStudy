# 版本记录模板

## v0.1.0 - Baseline Pipeline

### 功能

- [ ] 读取加速度 TXT 数据。
- [ ] 读取 30 秒睡眠标签。
- [ ] 构建 Wake/Sleep 二分类标签。
- [ ] 提取窗口级特征。
- [ ] 训练 baseline 模型。
- [ ] 输出评估报告。

### 实验结果

- 数据：
- 模型：
- 指标：
- 结论：

### 学到的知识

- 

### 问题与下一步

- 

## v0.2.0 - DeepSeek Report Demo

### 功能

- [x] 从 `.env` 读取 DeepSeek API Key。
- [x] 默认使用 `deepseek-flash`。
- [x] 保留 OpenAI-compatible messages 调用格式，方便后续切换 OpenAI 或接入 LangChain。
- [x] 从 `features.csv` 生成数据摘要。
- [x] 根据 `metrics.json` 生成实验建议 Prompt。
- [x] 生成 Markdown 报告和 Word 报告。
- [x] 提供 `--offline` 模式，用于不消耗 API 的本地流程验证。

### 学到的知识

- LLM 不适合直接读取大体量原始信号文件，应该先由 Python 做确定性统计。
- 数据分析报告更适合采用“事实统计由代码生成，解释和建议由 LLM 生成”的分工。
- 直接 API 调用适合学习底层原理，LangChain 更适合下一阶段管理工具调用和结构化输出。

### 下一步

- 用 LangChain 的 `PromptTemplate` 管理 prompt。
- 把数据摘要、指标读取、图表生成封装成 LangChain tools。
- 让模型输出固定 JSON，再由代码渲染 Markdown 和 Word。
