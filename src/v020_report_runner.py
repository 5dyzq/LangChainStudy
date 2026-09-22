from __future__ import annotations

"""v0.2.0 DeepSeek 实验报告命令行入口。
功能：读取实验配置、指标、数据集摘要，调用DeepSeek大模型生成实验报告，输出Markdown和docx文档
支持离线占位模式（不调用API），也支持传入自定义问题做简单问答
"""

import argparse
import json
from pathlib import Path

import yaml

from agents.data_analysis_agent import build_data_analysis_prompt
from agents.deepseek_client import DeepSeekClient
from agents.experiment_advisor_agent import build_experiment_advice_prompt
from agents.result_report_agent import build_result_report_prompt

from .dataset_builder import build_dataset, save_dataset
from .dataset_summary import summarize_feature_csv, summarize_feature_dataset
from .data_loader import discover_subjects
from .docx_report import markdown_to_docx


def _load_json(path: str | Path) -> dict:
    """读取json文件并返回字典"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_text(path: str | Path, text: str) -> None:
    """将文本写入文件，自动创建不存在的父文件夹"""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def _load_or_build_summary(config: dict, experiment_dir: Path) -> dict:
    """
        加载或生成数据集摘要：
        如果features.csv已经存在，直接读取并统计摘要；
        如果不存在，则自动发现被试文件、构建数据集、保存features.csv，再生成摘要
    """
    data_config = config["data"]
    features_path = experiment_dir / "features.csv"

    # 特征文件已存在，直接基于csv生成摘要
    if features_path.exists():
        return summarize_feature_csv(
            features_path,
            sampling_rate_hz=int(data_config["sampling_rate_hz"]),
            epoch_seconds=int(data_config["epoch_seconds"]),
        )

    # 特征文件不存在，需要从原始加速度、标签文件构建数据集
    subjects = discover_subjects(
        data_config["root_dir"],
        acceleration_suffix=data_config["acceleration_suffix"],
        label_suffix=data_config["label_suffix"],
    )
    # 读取配置中指定需要筛选的被试id列表
    selected_subject_ids = set(data_config.get("subject_ids") or [])
    # 如果配置指定了被试，过滤只保留选中的被试
    if selected_subject_ids:
        subjects = [subject for subject in subjects if subject.subject_id in selected_subject_ids]
    # 没有找到任何成对的加速度+标签文件，抛出异常
    if not subjects:
        raise RuntimeError("No paired acceleration/label files found.")

    # 构建窗口级特征数据集
    dataset = build_dataset(subjects, epoch_seconds=int(data_config["epoch_seconds"]))
    # 保存数据集到features.csv
    save_dataset(dataset, features_path)
    # 基于构建好的数据集生成统计摘要并返回
    return summarize_feature_dataset(
        dataset,
        sampling_rate_hz=int(data_config["sampling_rate_hz"]),
        epoch_seconds=int(data_config["epoch_seconds"]),
    )


def _offline_answer(title: str) -> str:
    """离线模式占位文本，不调用DeepSeek API，用于调试报告排版"""
    return f"{title}：离线模式占位文本。请去掉 --offline 后调用 DeepSeek 生成正式内容。"


def generate_v020_report(
    config_path: str | Path,
    question: str | None = None,
    offline: bool = False,
) -> dict[str, Path | str | None]:
    """
    读取已有实验结果，调用 DeepSeek 生成 Markdown 和 docx 报告。
    :param config_path: yaml配置文件路径
    :param question: 可选，用户自定义简单问答问题
    :param offline: 是否开启离线调试模式，True时不请求大模型
    :return: 字典，包含markdown路径、docx路径、问答结果
    """
    # 读取yaml实验配置
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    output_config = config["output"]
    experiment_dir = Path(output_config["experiment_dir"])
    metrics_path = experiment_dir / "metrics.json"

    # 检查指标文件是否存在，指标里存放模型评估结果
    if not metrics_path.exists():
        raise RuntimeError(f"Missing metrics file: {metrics_path}")

    # 读取指标文件、数据集摘要（不存在会自动构建）
    metrics = _load_json(metrics_path)
    dataset_summary = _load_or_build_summary(config, experiment_dir)

    # 定义输出报告路径
    markdown_path = Path("reports/v0.2.0_deepseek_experiment_report.md")
    docx_path = Path("reports/v0.2.0_deepseek_experiment_report.docx")

    if offline:
        # 离线模式：全部使用占位文字，不调用API
        simple_answer = _offline_answer("简单问答") if question else None
        data_analysis = _offline_answer("数据分析")
        experiment_advice = _offline_answer("实验建议")
        report_markdown = _offline_answer("实验报告")
    else:
        # 在线模式：初始化DeepSeek客户端，调用大模型
        client = DeepSeekClient()
        # 可选的简单问答模块
        simple_answer = client.chat(
            [
                {"role": "system", "content": "你是睡眠监测和生理信号分析学习助手。"},
                {"role": "user", "content": question},
            ],
            temperature=0.2,
        ) if question else None
        # 调用数据分析Agent，传入数据集摘要
        data_analysis = client.chat(build_data_analysis_prompt(dataset_summary), temperature=0.2)
        # 调用实验建议Agent，传入评估指标metrics
        experiment_advice = client.chat(build_experiment_advice_prompt(metrics), temperature=0.2)
        # 调用主报告Agent，整合所有内容生成完整实验报告正文
        report_markdown = client.chat(
            build_result_report_prompt(
                config=config,
                metrics=metrics,
                dataset_summary=dataset_summary,
                data_analysis=data_analysis,
                experiment_advice=experiment_advice,
            ),
            temperature=0.2,
        )
    # 拼接完整Markdown报告文本，把问答、数据集摘要、分析、建议、正文整合在一起
    full_markdown = f"""# v0.2.0 DeepSeek 睡眠实验报告

## 简单问答

{simple_answer or "本次未传入 --question，因此跳过简单问答。"}

## 数据摘要

```json
{json.dumps(dataset_summary, ensure_ascii=False, indent=2)}
```

## DeepSeek 数据分析

{data_analysis}

## DeepSeek 实验建议

{experiment_advice}

## 实验报告正文

{report_markdown}
"""
    _write_text(markdown_path, full_markdown)
    markdown_to_docx(full_markdown, docx_path)

    return {
        "markdown_path": markdown_path,
        "docx_path": docx_path,
        "simple_answer": simple_answer,
    }


def main() -> None:
    # 创建命令行参数解析器，description是脚本的功能说明
    parser = argparse.ArgumentParser(description="Generate v0.2.0 DeepSeek experiment report.")
    # 添加--config参数，指定配置文件路径；不传入时默认使用 configs/experiment.yaml
    parser.add_argument("--config", default="configs/experiment.yaml")
    # 添加--question可选参数，用来传用户想要向DeepSeek提问的问题，默认为空
    parser.add_argument("--question", default=None, help="Optional simple DeepSeek question.")
    # 添加--offline开关参数，action="store_true"代表只要命令带上这个参数，值就变成True；离线模式不调用大模型API
    parser.add_argument("--offline", action="store_true", help="Do not call DeepSeek; generate placeholder output.")
    # 解析命令行传入的所有参数，结果存到args对象里
    args = parser.parse_args()

    # 调用报告生成主函数，传入解析出来的配置路径、问题、离线标记
    result = generate_v020_report(args.config, question=args.question, offline=args.offline)
    # 打印生成的Markdown报告路径
    print(f"Markdown report: {result['markdown_path']}")
    # 打印生成的Word(docx)报告路径
    print(f"Word report: {result['docx_path']}")
    # 如果存在问答结果，就打印问答内容
    if result["simple_answer"]:
        print("Simple answer:")
        print(result["simple_answer"])


if __name__ == "__main__":
    main()
