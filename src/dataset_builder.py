from __future__ import annotations

"""构建窗口级机器学习数据表。

输出表中每一行代表一个带标签的睡眠 epoch。保留 subject_id 和
epoch_start 等元数据，方便切分、追踪和错误分析。
"""

from pathlib import Path

import pandas as pd

from .data_loader import SubjectFiles, read_acceleration, read_sleep_labels
from .feature_extraction import extract_time_domain_features
from .label_loader import map_sleep_stage_to_binary
from .windowing import iter_labeled_windows


def build_subject_dataset(
    subject_files: SubjectFiles,
    epoch_seconds: int = 30,
) -> pd.DataFrame:
    """为单个被试构建窗口级特征行。"""
    acceleration = read_acceleration(subject_files.acceleration_path)
    labels = read_sleep_labels(subject_files.label_path)
    labels = map_sleep_stage_to_binary(labels)

    rows: list[dict[str, float | int | str]] = []
    for start, label, window in iter_labeled_windows(acceleration, labels, epoch_seconds):
        features = extract_time_domain_features(window)

        # 将元数据和特征放在一起，便于报告追踪到具体被试和 epoch。
        features["subject_id"] = subject_files.subject_id
        features["epoch_start"] = start
        features["label"] = label
        rows.append(features)

    return pd.DataFrame(rows)


def build_dataset(subjects: list[SubjectFiles], epoch_seconds: int = 30) -> pd.DataFrame:
    """为多个被试构建特征数据集。"""
    frames = [build_subject_dataset(subject, epoch_seconds) for subject in subjects]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def save_dataset(dataset: pd.DataFrame, output_path: str | Path) -> None:
    """保存生成的特征表，便于复现实验。"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output, index=False)
