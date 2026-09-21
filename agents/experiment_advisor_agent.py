from __future__ import annotations

"""实验建议 Agent 的 Prompt 构造。

学习提示：
- 当前函数适合放在模型评估之后，用于解释指标并建议下一轮实验。
- 在 LangChain 中，可将该函数与 DeepSeekClient 或 ChatModel 组合成 Runnable。
- 在 LangGraph 中，它适合作为“实验建议节点”：
  输入 metrics，输出 advice，并写回流程 state。

这个 Agent 不应该直接修改代码或自动调参；第一版只让它给建议，
这样更容易控制实验过程，也更适合学习 Agent 的边界设计。
"""


def build_experiment_advice_prompt(metrics: dict) -> list[dict]:
    """构造用于下一轮实验建议的 LLM 消息。

    参数：
        metrics:
            模型评估指标字典，通常来自 src.model_eval.evaluate_classifier。
            推荐包含：
            - accuracy: 准确率
            - f1_macro: 宏平均 F1
            - f1_binary_sleep: Sleep 类 F1
            - confusion_matrix: 混淆矩阵
            - classification_report: Wake/Sleep 的 precision、recall、f1-score

    返回：
        Chat Completions / LangChain messages 兼容的消息列表。

    使用方式：
        messages = build_experiment_advice_prompt(metrics)
        advice = DeepSeekClient().chat(messages)
    """
    return [
        {
            "role": "system",
            # 让模型从 AutoML 顾问视角回答，而不是泛泛总结指标。
            "content": "你是 AutoML 实验顾问，关注评估可靠性、类别不平衡和特征改进。",
        },
        {
            "role": "user",
            # metrics 应来自真实代码输出，避免让模型凭空推断实验结果。
            "content": f"请根据以下 Wake/Sleep 二分类结果提出下一轮实验建议：\n{metrics}",
        },
    ]
