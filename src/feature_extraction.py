from __future__ import annotations

"""加速度窗口的特征提取。

基线版本有意从简单、可解释的时域特征开始。这样在加入频域特征或
深度学习模型之前，更容易理解早期模型的问题。
"""

import numpy as np
import pandas as pd


def add_magnitude(window: pd.DataFrame) -> pd.DataFrame:
    """增加三轴合加速度模长 sqrt(x^2 + y^2 + z^2)。"""
    enriched = window.copy()
    enriched["magnitude"] = np.sqrt(enriched["x"] ** 2 + enriched["y"] ** 2 + enriched["z"] ** 2)
    return enriched


def extract_time_domain_features(window: pd.DataFrame) -> dict[str, float]:
    """从一个加速度 epoch 中提取简单基线特征。"""
    enriched = add_magnitude(window)
    features: dict[str, float] = {"sample_count": float(len(enriched))}

    for axis in ["x", "y", "z", "magnitude"]:
        values = enriched[axis].to_numpy(dtype=float)

        # 第一版特征保持精简：中心趋势、离散程度、范围和能量。
        features[f"{axis}_mean"] = float(np.mean(values))
        features[f"{axis}_std"] = float(np.std(values))
        features[f"{axis}_min"] = float(np.min(values))
        features[f"{axis}_max"] = float(np.max(values))
        features[f"{axis}_median"] = float(np.median(values))
        features[f"{axis}_rms"] = float(np.sqrt(np.mean(values**2)))
        features[f"{axis}_energy"] = float(np.sum(values**2) / max(len(values), 1))

    return features
