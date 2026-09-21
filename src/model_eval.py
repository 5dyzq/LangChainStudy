from __future__ import annotations

"""用适合不均衡睡眠数据的指标评估分类器。"""

import json
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def evaluate_classifier(model, X_test, y_test) -> dict:
    """计算基线分类指标。"""
    y_pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        # Macro F1 可避免多数类掩盖少数类表现差的问题。
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "f1_binary_sleep": float(f1_score(y_test, y_pred, pos_label=1)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["Wake", "Sleep"],
            output_dict=True,
            zero_division=0,
        ),
    }


def save_metrics(metrics: dict, output_path: str | Path) -> None:
    """将指标保存为 UTF-8 JSON，供报告和后续 Agent 分析使用。"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
