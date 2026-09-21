from __future__ import annotations

"""训练 Wake/Sleep 基线分类器。"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def split_features_labels(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """将模型特征与标签、元数据分离。"""
    drop_columns = ["label", "subject_id", "epoch_start"]
    feature_columns = [col for col in dataset.columns if col not in drop_columns]
    return dataset[feature_columns], dataset["label"]


def create_model(model_name: str, random_state: int = 42):
    """创建一个支持的基线分类器。"""
    if model_name == "logistic_regression":
        # 逻辑回归需要特征标准化；class_weight 用于缓解 Wake/Sleep 类别不均衡。
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )
    if model_name == "random_forest":
        # 随机森林是较强的表格基线模型，且不要求特征提前标准化。
        return RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def train_test_model(
    dataset: pd.DataFrame,
    model_name: str = "random_forest",
    test_size: float = 0.2,
    random_state: int = 42,
    group_split_by_subject: bool = True,
):
    """切分数据、训练指定模型，并返回测试集。"""
    X, y = split_features_labels(dataset)

    if group_split_by_subject and "subject_id" in dataset.columns and dataset["subject_id"].nunique() > 1:
        # 按被试切分能更真实地估计模型在新用户上的表现。
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_index, test_index = next(splitter.split(X, y, groups=dataset["subject_id"]))
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    else:
        # 仅有单个被试的小 demo 使用普通随机切分作为兜底。
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

    model = create_model(model_name, random_state=random_state)
    model.fit(X_train, y_train)
    return model, X_test, y_test
