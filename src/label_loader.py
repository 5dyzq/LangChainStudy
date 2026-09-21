from __future__ import annotations

"""将 PSG 睡眠分期映射为第一版 MVP 的二分类标签。"""

import pandas as pd


def map_sleep_stage_to_binary(
    labels: pd.DataFrame,
    wake_stages: list[int] | tuple[int, ...] = (0,),
    sleep_stages: list[int] | tuple[int, ...] = (1, 2, 3, 5),
) -> pd.DataFrame:
    """将 PSG 分期映射为 Wake/Sleep 二分类标签。"""
    mapped = labels.copy()

    # 第一版聚焦 Wake/Sleep；后续多分类睡眠分期可复用本模块并替换映射。
    stage_to_binary = {stage: 0 for stage in wake_stages}
    stage_to_binary.update({stage: 1 for stage in sleep_stages})
    mapped["label"] = mapped["stage"].map(stage_to_binary)

    # 对未知分期不做猜测，直接丢弃。
    return mapped.dropna(subset=["label"]).assign(label=lambda df: df["label"].astype(int))
