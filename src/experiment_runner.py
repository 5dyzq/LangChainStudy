from __future__ import annotations

"""基线实验的命令行入口。"""

import argparse
from pathlib import Path

import yaml

from .data_loader import discover_subjects
from .dataset_builder import build_dataset, save_dataset
from .model_eval import evaluate_classifier, save_metrics
from .model_train import train_test_model
from .report_generator import generate_markdown_report


def run_experiment(config_path: str | Path) -> None:
    """按配置运行一次从原始 TXT 到报告的 Wake/Sleep 实验。"""
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    data_config = config["data"]
    output_config = config["output"]
    training_config = config["training"]

    subjects = discover_subjects(
        data_config["root_dir"],
        acceleration_suffix=data_config["acceleration_suffix"],
        label_suffix=data_config["label_suffix"],
    )

    selected_subject_ids = set(data_config.get("subject_ids") or [])
    if selected_subject_ids:
        # 可选的被试过滤，用于早期小规模调试。
        subjects = [subject for subject in subjects if subject.subject_id in selected_subject_ids]

    if not subjects:
        raise RuntimeError("No paired acceleration/label files found.")

    dataset = build_dataset(subjects, epoch_seconds=int(data_config["epoch_seconds"]))
    if dataset.empty:
        raise RuntimeError("No labeled windows were created from the input data.")

    experiment_dir = Path(output_config["experiment_dir"])

    # 保存特征表，后续可检查本次训练实际使用的数据。
    save_dataset(dataset, experiment_dir / "features.csv")

    model, X_test, y_test = train_test_model(
        dataset,
        model_name=training_config["model"],
        test_size=float(training_config["test_size"]),
        random_state=int(training_config["random_state"]),
        group_split_by_subject=bool(training_config.get("group_split_by_subject", True)),
    )
    metrics = evaluate_classifier(model, X_test, y_test)
    save_metrics(metrics, experiment_dir / "metrics.json")
    generate_markdown_report(metrics, output_config["report_path"])


def main() -> None:
    """解析命令行参数并启动实验。"""
    parser = argparse.ArgumentParser(description="Run Wake/Sleep baseline experiment.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()
