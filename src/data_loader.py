from __future__ import annotations

"""读取配对的加速度文件和睡眠标签文件。

原始数据按被试组织：每个被试应有一个加速度文件和一个 PSG 标签文件。
本模块只负责文件发现和原始 TXT 解析；标签映射和窗口切分放在独立模块中。
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SubjectFiles:
    """一个被试对应的加速度文件和睡眠标签文件路径。"""

    subject_id: str
    acceleration_path: Path
    label_path: Path


def discover_subjects(
    root_dir: str | Path,
    acceleration_suffix: str = "_acceleration.txt",
    label_suffix: str = "_labeled_sleep.txt",
) -> list[SubjectFiles]:
    """查找同时具备加速度文件和睡眠标签文件的被试。"""
    root = Path(root_dir)
    subjects: list[SubjectFiles] = []

    # 使用文件名前缀作为被试 id，例如 1066528_acceleration.txt。
    for acceleration_path in sorted(root.glob(f"*{acceleration_suffix}")):
        subject_id = acceleration_path.name[: -len(acceleration_suffix)]
        label_path = root / f"{subject_id}{label_suffix}"
        if label_path.exists():
            subjects.append(
                SubjectFiles(
                    subject_id=subject_id,
                    acceleration_path=acceleration_path,
                    label_path=label_path,
                )
            )

    return subjects


def read_acceleration(path: str | Path) -> pd.DataFrame:
    """读取 Apple Watch 加速度 TXT，列名为 time、x、y、z。"""
    # sep=r"\s+" 支持一个或多个空格/Tab 分隔。
    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["time", "x", "y", "z"],
        dtype={"time": "float64", "x": "float64", "y": "float64", "z": "float64"},
    )


def read_sleep_labels(path: str | Path) -> pd.DataFrame:
    """读取 PSG 睡眠标签 TXT，列名为 time、stage。"""
    # 标签是 epoch 级标注；time 表示从 PSG 开始后的秒数。
    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["time", "stage"],
        dtype={"time": "float64", "stage": "int64"},
    )
