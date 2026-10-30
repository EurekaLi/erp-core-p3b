"""固定分组划分上的LR增量对照；smoke不计算外层测试成绩，full产生完整OOF。"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from threadpoolctl import threadpool_limits

from erp_core_p3b.cbramod_audit import sha256_file
from erp_core_p3b.classification_data import load_classification_dataset
from erp_core_p3b.classification_io import (
    canonical_hash,
    new_artifact_run,
    resolve_path,
    verify_artifact,
    write_json,
)

METRICS = [
    "balanced_accuracy",
    "roc_auc",
    "average_precision",
    "target_precision",
    "target_recall",
    "target_f1",
]


def binary_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    """两类排序指标缺类时缺失；无预测阳性时precision/F1固定为0，不隐藏告警。"""
    y, p = np.asarray(labels), np.asarray(scores, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("评价标签/分数维度或类别错误")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not 0 <= threshold <= 1:
        raise ValueError("预测分数/阈值必须在[0,1]")
    tn, fp, fn, tp = map(int, confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel())
    recall = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    precision = tp / (tp + fp) if tp + fp else 0.0
    both = len(np.unique(y)) == 2
    return {
        "balanced_accuracy": (recall + specificity) / 2 if both else np.nan,
        "roc_auc": float(roc_auc_score(y, p)) if both else np.nan,
        "average_precision": float(average_precision_score(y, p)) if both else np.nan,
        "target_precision": precision,
        "target_recall": recall,
        "target_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "specificity": specificity,
        "accuracy": (tp + tn) / len(y),
        "trials": len(y),
        "target_trials": tp + fn,
        "standard_trials": tn + fp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def fit_pipeline(x: np.ndarray, y: np.ndarray, config: dict[str, Any]) -> Pipeline:
    """fit仅接收该折训练部分；验证/测试数组不作为函数参数，避免误拟合。"""
    if set(np.unique(y)) != {0, 1} or not np.isfinite(x).all():
        raise ValueError("训练数据缺类或非有限")
    if config["class_weight"] != "balanced" or config["solver"] != "lbfgs":
        raise ValueError("本轮协议固定balanced类别权重与lbfgs求解器")
    if not np.isfinite(config["C"]) or config["C"] <= 0 or int(config["max_iter"]) < 1:
        raise ValueError("LR正则化或迭代参数无效")
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=float(config["C"]),
                    class_weight=config["class_weight"],
                    solver=config["solver"],
                    max_iter=int(config["max_iter"]),
                    random_state=int(config["random_state"]),
                ),
            ),
        ]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(x, y)
    if any(issubclass(warning.category, ConvergenceWarning) for warning in caught):
        raise RuntimeError("LR未收敛；先诊断，禁止静默保留该折成绩")
    return pipeline


def bootstrap_mean(values: np.ndarray, config: dict[str, Any]) -> tuple[float, float, float]:
    """按受试者重采样，CI条件于本次固定划分/已拟合模型，不包含重新训练不确定性。"""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    iterations, confidence = int(config["bootstrap_iterations"]), float(config["confidence_level"])
    if iterations < 1 or not 0 < confidence < 1:
        raise ValueError("bootstrap参数无效")
    rng = np.random.default_rng(int(config["random_state"]))
    indices = rng.integers(0, len(values), size=(iterations, len(values)))
    means = values[indices].mean(axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(means, [alpha, 1 - alpha])
    return float(values.mean()), float(low), float(high)


def subject_metrics(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for (experiment, fold, subject), group in predictions.groupby(
        ["experiment", "fold", "subject_id"]
    ):
        rows.append(
            {
                "experiment": experiment,
                "fold": int(fold),
                "subject_id": subject,
                **binary_metrics(group["label"].to_numpy(), group["score"].to_numpy(), threshold),
            }
        )
    return pd.DataFrame(rows)


def summarize_subjects(
    metrics: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries, comparisons = [], []
    experiments = metrics["experiment"].drop_duplicates().tolist()
    for experiment, frame in metrics.groupby("experiment"):
        if frame["subject_id"].duplicated().any():
            raise ValueError("主分析不能把同一个受试者多次计入等权汇总")
        for metric in METRICS:
            mean, low, high = bootstrap_mean(frame[metric].to_numpy(), config)
            summaries.append(
                {
                    "experiment": experiment,
                    "metric": metric,
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "valid_subjects": int(frame[metric].notna().sum()),
                }
            )
    for index, first in enumerate(experiments):
        for second in experiments[index + 1 :]:
            paired = metrics[metrics["experiment"] == first].merge(
                metrics[metrics["experiment"] == second],
                on="subject_id",
                suffixes=("_first", "_second"),
                validate="one_to_one",
            )
            for metric in METRICS:
                differences = (
                    paired[f"{metric}_second"].to_numpy() - paired[f"{metric}_first"].to_numpy()
                )
                mean, low, high = bootstrap_mean(differences, config)
                comparisons.append(
                    {
                        "first": first,
                        "second": second,
                        "metric": metric,
                        "mean_second_minus_first": mean,
                        "ci_low": low,
                        "ci_high": high,
                        "valid_pairs": int(np.isfinite(differences).sum()),
                        "multiple_test_adjusted": False,
                    }
                )
    return pd.DataFrame(summaries), pd.DataFrame(comparisons)


def _prediction_frame(
    group: pd.DataFrame,
    pipeline: Pipeline,
    columns: list[str],
    experiment: str,
    fold: int,
    threshold: float,
) -> pd.DataFrame:
    positive = list(pipeline.named_steps["lr"].classes_).index(1)
    scores = pipeline.predict_proba(group[columns].to_numpy())[:, positive]
    frame = group[["trial_id", "subject_id", "label"]].copy()
    frame["experiment"], frame["fold"] = experiment, fold
    frame["score"], frame["prediction"] = scores, (scores >= threshold).astype(int)
    return frame


def _plots(output: Path, subjects: pd.DataFrame, pooled: pd.DataFrame) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir()
    fig, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    pivot = subjects.pivot(index="subject_id", columns="experiment", values="balanced_accuracy")
    for experiment in pivot:
        axis.plot(
            np.arange(len(pivot)),
            pivot[experiment],
            "o-",
            markersize=3,
            linewidth=0.7,
            label=experiment,
        )
    axis.axhline(0.5, color="grey", linestyle="--", label="Standard-only BA")
    axis.set_xticks(np.arange(len(pivot)), pivot.index, rotation=90, fontsize=8)
    axis.set(
        ylabel="Balanced accuracy",
        title="Subject-independent OOF | equal subject weights",
        ylim=(-0.02, 1.02),
    )
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.savefig(figures / "subject_balanced_accuracy.png", dpi=140)
    plt.close(fig)
    for _, row in pooled.iterrows():
        fig, axis = plt.subplots(figsize=(5, 4), constrained_layout=True)
        matrix = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
        axis.imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            color = "white" if value > matrix.max() / 2 else "black"
            axis.text(j, i, str(int(value)), ha="center", va="center", color=color)
        axis.set_xticks([0, 1], ["Standard", "Target"])
        axis.set_yticks([0, 1], ["Standard", "Target"])
        axis.set(
            xlabel="Predicted", ylabel="True", title=f"{row['experiment']} | pooled OOF counts"
        )
        fig.savefig(figures / f"{row['experiment']}_confusion.png", dpi=140)
        plt.close(fig)


def train_baseline(
    config_path: Path,
    dataset_dir: Path,
    feature_dir: Path,
    root: Path,
    *,
    mode: str = "smoke",
    run_name: str | None = None,
    output: Path | None = None,
) -> Path:
    if mode not in ("smoke", "full"):
        raise ValueError("mode必须是smoke或full")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    trials, dataset, splits = load_classification_dataset(dataset_dir)
    feature_summary = verify_artifact(feature_dir, "compact_features")
    if (
        not feature_summary["complete_dataset"]
        or feature_summary["dataset_hash"] != dataset["dataset_hash"]
    ):
        raise ValueError("LR需要相同版本的全体特征；pilot特征禁止当正式输入")
    features = pd.read_csv(feature_dir / "features.csv")
    dictionary = json.loads((feature_dir / "feature_dictionary.json").read_text(encoding="utf-8"))
    if canonical_hash(dictionary) != feature_summary["feature_definition_hash"]:
        raise ValueError("特征字典hash改变")
    for column in ("trial_id", "subject_id", "label"):
        if not np.array_equal(features[column].to_numpy(), trials[column].to_numpy()):
            raise ValueError(f"特征表{column}与共享清单顺序/身份不同")
    names = [item["name"] for item in dictionary]
    if any(not item["name"].startswith(item["family"] + "_") for item in dictionary):
        raise ValueError("特征白名单中出现元数据或不符合特征族命名的列")
    if len(set(names)) != len(names) or set(features) != {
        "trial_id",
        "subject_id",
        "label",
        *names,
    }:
        raise ValueError("特征白名单与实际列不同，禁止隐式使用元数据")
    if not np.isfinite(features[names].to_numpy()).all():
        raise ValueError("全体特征含NaN/Inf，不自动补值或删trial")
    name = run_name or f"{mode}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("run-name仅允许字母、数字、下划线、连字符")
    output = (
        resolve_path(root, output)
        if output is not None
        else resolve_path(root, config["outputs"]["directory"]) / name
    )
    with new_artifact_run(root, output, "logistic_baseline") as summary:
        write_json(output / "config_snapshot.json", config)
        write_json(output / "subject_splits.json", splits)
        write_json(output / "feature_dictionary.json", dictionary)
        summary.update(
            mode=mode,
            dataset_hash=dataset["dataset_hash"],
            split_hash=canonical_hash(splits),
            feature_definition_hash=feature_summary["feature_definition_hash"],
            config_sha256=sha256_file(config_path),
            external_test_evaluated=mode == "full",
        )
        experiments = (
            {"erp": config["experiments"]["erp"]} if mode == "smoke" else config["experiments"]
        )
        folds = splits["folds"][:1] if mode == "smoke" else splits["folds"]
        validations, tests, logs, coefficients = [], [], [], []
        (output / "models").mkdir()
        threshold = float(config["model"]["prediction_threshold"])
        if not 0 < threshold < 1:
            raise ValueError("预测阈值须在(0,1)")
        with threadpool_limits(limits=1):
            for experiment, families in experiments.items():
                if not set(families).issubset({item["family"] for item in dictionary}):
                    raise ValueError("实验包含未知特征族")
                columns = [item["name"] for item in dictionary if item["family"] in families]
                if not columns:
                    raise ValueError("实验没有特征")
                for fold in folds:
                    print(f"Fitting {experiment} fold {fold['fold']} ...", flush=True)
                    groups = {
                        role: features[features["trial_id"].isin(ids)]
                        for role, ids in fold["trial_ids"].items()
                    }
                    train = groups["train"]
                    pipeline = fit_pipeline(
                        train[columns].to_numpy(), train["label"].to_numpy(), config["model"]
                    )
                    validations.append(
                        _prediction_frame(
                            groups["validation"],
                            pipeline,
                            columns,
                            experiment,
                            fold["fold"],
                            threshold,
                        )
                    )
                    if mode == "full":
                        tests.append(
                            _prediction_frame(
                                groups["test"],
                                pipeline,
                                columns,
                                experiment,
                                fold["fold"],
                                threshold,
                            )
                        )
                    joblib.dump(
                        pipeline, output / "models" / f"{experiment}_fold-{fold['fold']}.joblib"
                    )
                    weights = compute_class_weight(
                        "balanced", classes=np.array([0, 1]), y=train["label"].to_numpy()
                    )
                    lr, scaler = pipeline.named_steps["lr"], pipeline.named_steps["scaler"]
                    logs.append(
                        {
                            "experiment": experiment,
                            "fold": fold["fold"],
                            "feature_count": len(columns),
                            "train_trials": len(train),
                            "train_subjects": train["subject_id"].nunique(),
                            "validation_trials": len(groups["validation"]),
                            "test_trials": len(groups["test"]),
                            "weight_standard": float(weights[0]),
                            "weight_target": float(weights[1]),
                            "iterations": int(lr.n_iter_.max()),
                            "scaler_fit_scope": "train_only",
                        }
                    )
                    for column, coefficient, mean, scale in zip(
                        columns, lr.coef_[0], scaler.mean_, scaler.scale_, strict=True
                    ):
                        coefficients.append(
                            {
                                "experiment": experiment,
                                "fold": fold["fold"],
                                "feature": column,
                                "coefficient_standardized": float(coefficient),
                                "training_mean": float(mean),
                                "training_scale": float(scale),
                                "intercept": float(lr.intercept_[0]),
                            }
                        )
        validation = pd.concat(validations, ignore_index=True)
        validation.to_csv(output / "validation_predictions.csv", index=False)
        subject_metrics(validation, threshold).to_csv(
            output / "validation_subject_metrics.csv", index=False
        )
        pd.DataFrame(logs).to_csv(output / "fit_audit.csv", index=False)
        pd.DataFrame(coefficients).to_csv(output / "coefficients.csv", index=False)
        lines = [
            "# 紧凑特征Logistic Regression",
            "",
            f"模式：{mode}。",
            "",
            "只在共同训练受试者上拟合Scaler/LR，验证受试者不参与拟合。",
            f"类别权重基于训练标签；C={config['model']['C']}、阈值={threshold}，不依据外层测试调参。",
            "",
        ]
        if mode == "full":
            predictions = pd.concat(tests, ignore_index=True)
            for experiment, group in predictions.groupby("experiment"):
                if group["trial_id"].duplicated().any() or set(group["trial_id"]) != set(
                    trials["trial_id"]
                ):
                    raise ValueError(f"{experiment}: OOF未恰好覆盖全部trial一次")
            predictions.to_csv(output / "test_predictions.csv", index=False)
            subjects = subject_metrics(predictions, threshold)
            subjects.to_csv(output / "test_subject_metrics.csv", index=False)
            macro, paired = summarize_subjects(subjects, config["statistics"])
            macro.to_csv(output / "subject_macro_ci.csv", index=False)
            paired.to_csv(output / "paired_feature_comparisons.csv", index=False)
            pooled = pd.DataFrame(
                [
                    {
                        "experiment": experiment,
                        **binary_metrics(
                            group["label"].to_numpy(), group["score"].to_numpy(), threshold
                        ),
                    }
                    for experiment, group in predictions.groupby("experiment")
                ]
            )
            pooled.to_csv(output / "pooled_test_metrics.csv", index=False)
            pd.DataFrame(
                [binary_metrics(trials["label"].to_numpy(), np.zeros(len(trials)), threshold)]
            ).to_csv(output / "standard_only_reference.csv", index=False)
            _plots(output, subjects, pooled)
            lines.extend(
                [
                    "主结果为受试者等权平均；pooled结果按trial合并，二者不可混称。",
                    "CI按受试者bootstrap，条件于固定模型/划分，不涵盖重新训练的不确定性。",
                    "成对差值CI是描述性比较，未进行多重检验校正，不据此宣称因果或临床效用。",
                    "",
                    f"|特征组|受试者平均BA|{config['statistics']['confidence_level'] * 100:g}%"
                    "受试者bootstrap CI|",
                    "|---|---:|---:|",
                ]
            )
            for _, row in macro[macro["metric"] == "balanced_accuracy"].iterrows():
                lines.append(
                    f"|{row['experiment']}|{row['mean']:.4f}|"
                    f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]|"
                )
            lines.extend(["", "![逐人BA](figures/subject_balanced_accuracy.png)"])
            summary.update(
                test_subject_count=subjects["subject_id"].nunique(),
                test_predictions=len(predictions),
            )
        else:
            lines.append(
                "本轮只训练一折ERP并评估验证受试者，没有计算外层测试成绩，不能作为正式性能结论。"
            )
        lines.extend(
            [
                "",
                "LR系数是标准化特征的模型权重，不是脑源强度；相关特征会重新分配系数。",
                "这些是离线受试者独立结果，上游ICA使用每人整段无标签记录，不是严格在线部署。",
                "完整参数、数据版本和拟合审计见同目录JSON/CSV；不加载来源不明的joblib模型。",
            ]
        )
        (output / "results_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        summary.update(
            fitted_models=len(logs),
            experiments=list(experiments),
            threshold=threshold,
            hyperparameter_search=False,
            pca_fitted=False,
        )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path, default=Path("configs/modeling.yaml"))
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("data/processed/classification/v1")
    )
    parser.add_argument(
        "--features-dir", type=Path, default=Path("data/processed/features/compact_v1")
    )
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--run-name")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        output = train_baseline(
            resolve_path(args.project_root, args.config),
            resolve_path(args.project_root, args.dataset_dir),
            resolve_path(args.project_root, args.features_dir),
            args.project_root,
            mode=args.mode,
            run_name=args.run_name,
            output=args.output_dir,
        )
    except Exception as error:
        print(f"LR training failed: {error}", file=sys.stderr)
        return 1
    print(f"Logistic baseline: {output}")
    return 0
