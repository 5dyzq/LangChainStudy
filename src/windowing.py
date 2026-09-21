from __future__ import annotations

"""将高频加速度样本对齐到 30 秒睡眠标签 epoch。"""

from collections.abc import Iterator

import pandas as pd


def iter_labeled_windows(
    acceleration: pd.DataFrame,
    labels: pd.DataFrame,
    epoch_seconds: int = 30,
) -> Iterator[tuple[float, int, pd.DataFrame]]:
    """逐个返回与标签 epoch 对齐的加速度窗口。"""
    for row in labels.itertuples(index=False):
        start = float(row.time)
        end = start + epoch_seconds

        # 使用半开区间 [start, end)，避免相邻 epoch 重叠。
        window = acceleration[(acceleration["time"] >= start) & (acceleration["time"] < end)]
        if not window.empty:
            yield start, int(row.label), window
