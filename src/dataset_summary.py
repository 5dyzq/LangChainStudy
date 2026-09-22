from __future__ import annotations

"""生成供 LLM 使用的数据集摘要。

本模块只做确定性统计，不让大模型直接读取大体量原始数据。这样做有两个好处：

1. 控制 token 数量，避免把几十 MB 的 TXT 塞进提示词。
2. 保证报告中的基础事实来自 Python 计算，而不是 LLM 猜测。
"""

from pathlib import Path

import pandas as pd


def summarize_feature_dataset(dataset: pd.DataFrame, sampling_rate_hz: int, epoch_seconds: int) -> dict:
    """从窗口级特征表生成数据摘要。

    Args:
        dataset: pandas特征数据表
        sampling_rate_hz: 原始信号采样频率(Hz)
        epoch_seconds: 每个数据窗口(epoch)的时长，单位秒

    Returns:
        dict: 数据集各项统计信息摘要
    """
    # 如果传入的数据集是空表格，直接抛出异常终止
    if dataset.empty:
        raise ValueError("dataset is empty")

    # 统计label列各类别的数量，按标签排序，转为字典
    label_counts = dataset["label"].value_counts().sort_index().to_dict()

    # 按被试subject_id分组，统计每个被试的数据窗口行数，并按数量降序排列
    subject_counts = dataset.groupby("subject_id").size().sort_values(ascending=False)

    # 筛选特征列：排除id、时间、标签列，其余全部作为模型特征
    feature_columns = [
        column
        for column in dataset.columns
        if column not in {"subject_id", "epoch_start", "label"}
    ]

    # 组装摘要信息并返回
    return {
        "subject_count": int(dataset["subject_id"].nunique()), # 被试总人数（去重计数）
        "window_count": int(len(dataset)), # 数据窗口总行数
        "label_distribution": {
            "wake": int(label_counts.get(0, 0)),  # label=0 清醒样本数量
            "sleep": int(label_counts.get(1, 0)), # label=1 睡眠样本数量
        },
        "wake_ratio": round(float((dataset["label"] == 0).mean()), 4), # 清醒样本占比，保留4位小数
        "sleep_ratio": round(float((dataset["label"] == 1).mean()), 4), # 睡眠样本占比，保留4位小数
        "sampling_rate_hz": int(sampling_rate_hz), # 采样频率
        "epoch_seconds": int(epoch_seconds), # 每个窗口时长(秒)
        "feature_count": len(feature_columns), # 特征总数量
        "feature_columns": feature_columns, # 特征名称列表
        "top_subject_window_counts": { # 取前10个窗口最多的被试，id:窗口数量
            str(subject_id): int(count)
            for subject_id, count in subject_counts.head(10).items()
        },
        "epoch_start_range": {
            "min": float(dataset["epoch_start"].min()), # 窗口起始时间最小值
            "max": float(dataset["epoch_start"].max()), # 窗口起始时间最大值
        },
    }


def summarize_feature_csv(path: str | Path, sampling_rate_hz: int, epoch_seconds: int) -> dict:
    """读取已保存的 features.csv 文件并生成摘要。

        Args:
            path: csv文件路径
            sampling_rate_hz: 原始信号采样频率(Hz)
            epoch_seconds: 每个数据窗口时长(秒)

        Returns:
            dict: 数据集统计摘要
        """
    dataset = pd.read_csv(path) # 读取csv文件到DataFrame
    return summarize_feature_dataset(dataset, sampling_rate_hz, epoch_seconds) # 调用上面函数生成摘要
