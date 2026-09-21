from __future__ import annotations

"""数据分析 Agent 的 Prompt 构造。

学习提示：
- 当前文件还不是完整 Agent，只负责构造 messages。
- 在 LangChain 中，这个函数的输出可以交给 ChatModel.invoke(...)。
- 在 LangGraph 中，这个函数适合作为“数据分析节点”的一部分：
  节点从 state 中读取 dataset_summary，调用 LLM，然后把 analysis 写回 state。

建议输入的数据概况 dataset_summary 可以包含：
- 被试数量
- 窗口数量
- Wake/Sleep 类别分布
- 每个窗口的样本点数量统计
- 缺失窗口数量
- 采样率和 epoch 秒数
"""


def build_data_analysis_prompt(dataset_summary: dict) -> list[dict]:
    """构造用于数据概况分析的 LLM 消息。

    参数：
        dataset_summary:
            数据集概况字典，由普通 Python 数据分析代码生成，而不是由 LLM 猜测。
            推荐字段示例：
            {
                "subject_count": 31,
                "window_count": 28000,
                "label_distribution": {"wake": 6000, "sleep": 22000},
                "sampling_rate_hz": 50,
                "epoch_seconds": 30
            }

    返回：
        Chat Completions / LangChain messages 兼容的消息列表。
        当前格式是 list[dict]，后续也可以转换成 LangChain 的
        SystemMessage 和 HumanMessage。
    """
    return [
        {
            "role": "system",
            # system 消息用于设定模型角色和回答边界。
            "content": "你是机器学习数据分析助手，擅长一维生理信号和睡眠监测数据。",
        },
        {
            "role": "user",
            # user 消息放入结构化数据概况，让模型基于事实做分析。
            "content": f"请分析以下睡眠二分类数据概况，并指出主要风险和下一步建议：\n{dataset_summary}",
        },
    ]
