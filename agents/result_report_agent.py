from __future__ import annotations

"""实验报告 Agent 的 Prompt 构造。

学习提示：
- 当前函数负责把实验配置和评估指标组织成可喂给 LLM 的 messages。
- 在 LangChain 中，后续可以加入 PromptTemplate、结构化输出解析器等能力。
- 在 LangGraph 中，它适合作为最后的“报告生成节点”：
  从 state 中读取 config 和 metrics，输出 report_text。

和 src.report_generator 的区别：
- src.report_generator 是确定性的 Markdown 生成器，不依赖大模型。
- 本文件面向 Agent 报告，适合生成自然语言解释、问题分析和下一步建议。
"""


def build_result_report_prompt(
    config: dict,
    metrics: dict,
    dataset_summary: dict | None = None,
    data_analysis: str | None = None,
    experiment_advice: str | None = None,
) -> list[dict]:
    """构造用于生成自然语言实验总结的 LLM 消息。

    参数：
        config:
            本次实验配置，通常来自 configs/experiment.yaml。
            推荐包含：
            - 数据路径、采样率、epoch 秒数
            - 标签映射方式
            - 特征开关
            - 模型名称、切分方式、随机种子

        metrics:
            本次实验指标，通常来自 experiments/*/metrics.json。
            推荐包含 accuracy、f1_macro、confusion_matrix、classification_report。

    返回：
        Chat Completions / LangChain messages 兼容的消息列表。

    后续扩展：
        如果希望 LLM 直接输出 Markdown，可在 user prompt 中明确要求标题结构；
        如果希望输出 JSON，可在 LangChain 中配合结构化输出解析器。
    """
    return [
        {
            "role": "system",
            # system 消息要求模型关注可复现性和工程表达。
            "content": "你是实验报告撰写助手，请用中文和清晰、可复现的工程语言总结实验。",
        },
        {
            "role": "user",
            # 同时传入 config 和 metrics，让报告能说明“怎么做”和“结果如何”。
            "content": (
                "请基于以下材料生成一份 Markdown 实验报告正文，风格偏工程记录。\n"
                "必须包含这些二级标题：实验目的、数据来源、方法、结果、图表建议、讨论、结论、局限性、后续工作。\n"
                "请不要编造不存在的数据，只能基于输入材料做分析。\n\n"
                f"实验配置：\n{config}\n\n"
                f"数据摘要：\n{dataset_summary}\n\n"
                f"实验指标：\n{metrics}\n\n"
                f"数据分析：\n{data_analysis}\n\n"
                f"实验建议：\n{experiment_advice}"
            ),
        },
    ]
