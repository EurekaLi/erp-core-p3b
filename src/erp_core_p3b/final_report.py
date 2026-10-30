"""核验共同OOF身份后对照六模型；受试者统计和误差复核均不改变原训练或QC裁决。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from erp_core_p3b.cbramod_audit import sha256_file
from erp_core_p3b.classification_data import load_classification_dataset
from erp_core_p3b.classification_io import (
    canonical_hash,
    new_artifact_run,
    resolve_path,
    verify_artifact,
    write_json,
)
from erp_core_p3b.logistic_baseline import binary_metrics, bootstrap_mean, subject_metrics

METRICS = (
    "balanced_accuracy",
    "roc_auc",
    "average_precision",
    "target_precision",
    "target_recall",
    "specificity",
    "target_f1",
    "accuracy",
)
DL_NAMES = ("eegnet", "cbramod_linear", "cbramod_finetune")


def validate_oof(
    frame: pd.DataFrame,
    trials: pd.DataFrame,
    splits: dict[str, Any],
    experiments: Sequence[str],
    threshold: float,
) -> pd.DataFrame:
    """每模型每trial一次，并逐行验证标签、受试者和outer test折；行顺序无关。

    仅有两目录hash相同仍不足以证明CSV未把validation误标为OOF，因此显式核对归属。
    """
    required = {"trial_id", "subject_id", "label", "score", "experiment", "fold"}
    if not required.issubset(frame) or frame[list(required)].isna().any().any():
        raise ValueError("OOF列缺失或存在缺失值")
    if set(frame["experiment"]) != set(experiments) or not experiments:
        raise ValueError("OOF模型名单不符合配置")
    expected = trials[["trial_id", "subject_id", "label"]].copy()
    fold_map = {
        trial: fold["fold"] for fold in splits["folds"] for trial in fold["trial_ids"]["test"]
    }
    expected["fold"] = expected["trial_id"].map(fold_map)
    expected = expected.set_index("trial_id")
    checked = []
    for experiment in experiments:
        group = frame[frame["experiment"] == experiment]
        if group["trial_id"].duplicated().any() or set(group["trial_id"]) != set(expected.index):
            raise ValueError(f"{experiment}: OOF缺少trial、身份重复或含额外trial")
        aligned = group.set_index("trial_id").loc[expected.index].copy()
        for column in ("subject_id", "label", "fold"):
            if not np.array_equal(aligned[column], expected[column]):
                raise ValueError(f"{experiment}: {column}与共同清单/outer test不一致")
        binary_metrics(aligned["label"].to_numpy(), aligned["score"].to_numpy(), threshold)
        prediction = (aligned["score"] >= threshold).astype(int)
        if "prediction" in aligned and not np.array_equal(aligned["prediction"], prediction):
            raise ValueError("OOF已有prediction与固定阈值不一致")
        aligned["prediction"] = prediction
        checked.append(aligned.reset_index())
    return pd.concat(checked, ignore_index=True)


def compare_subjects(
    metrics: pd.DataFrame,
    names: Sequence[str],
    contrasts: list[list[str]],
    statistics: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """CI按受试者重采样；配对差值保留同一人的两模型关联，不按独立均值CI相减。"""
    rows, paired = [], []
    subject_set = set(metrics["subject_id"])
    for name in names:
        frame = metrics[metrics["experiment"] == name].sort_values("subject_id")
        if frame["subject_id"].duplicated().any() or set(frame["subject_id"]) != subject_set:
            raise ValueError("受试者汇总重复或某模型漏人")
        for metric in METRICS:
            if not np.isfinite(frame[metric]).all():
                raise ValueError(f"{name}: {metric}存在未定义值，不能静默删人")
            mean, low, high = bootstrap_mean(frame[metric].to_numpy(), statistics)
            rows.append(
                {
                    "experiment": name,
                    "metric": metric,
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "subjects": len(frame),
                }
            )
    for reference, candidate in contrasts:
        if reference not in names or candidate not in names or reference == candidate:
            raise ValueError("配对比较模型名称无效")
        left = metrics[metrics["experiment"] == reference].set_index("subject_id")
        right = metrics[metrics["experiment"] == candidate].set_index("subject_id").loc[left.index]
        for metric in METRICS:
            delta = right[metric].to_numpy() - left[metric].to_numpy()
            mean, low, high = bootstrap_mean(delta, statistics)
            tied = np.isclose(delta, 0, rtol=0, atol=1e-12)
            paired.append(
                {
                    "reference": reference,
                    "candidate": candidate,
                    "metric": metric,
                    "mean_candidate_minus_reference": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "candidate_higher_subjects": int(((delta > 0) & ~tied).sum()),
                    "candidate_lower_subjects": int(((delta < 0) & ~tied).sum()),
                    "tied_subjects": int(tied.sum()),
                    "pairs": len(delta),
                    "interpretation": "conditional_bootstrap_descriptive_unadjusted",
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(paired)


def trial_errors(
    predictions: pd.DataFrame, trials: pd.DataFrame, names: Sequence[str], threshold: float
) -> pd.DataFrame:
    """通过trial_id连接原刺激身份；保留错误类型，避免只用总错误率掩盖Target漏检。"""
    errors = trials.copy().set_index("trial_id")
    for name in names:
        aligned = (
            predictions[predictions["experiment"] == name].set_index("trial_id").loc[errors.index]
        )
        errors[f"{name}__score"] = aligned["score"]
        errors[f"{name}__prediction"] = (aligned["score"] >= threshold).astype(int)
        errors[f"{name}__error"] = (errors[f"{name}__prediction"] != errors["label"]).astype(int)
    nn_names = [name for name in DL_NAMES if name in names]
    errors["nn_error_models"] = errors[[f"{name}__error" for name in nn_names]].sum(axis=1)
    errors["all_model_errors"] = errors[[f"{name}__error" for name in names]].sum(axis=1)
    errors["dl_prediction_disagreement"] = (
        errors[[f"{name}__prediction" for name in nn_names]].nunique(axis=1) > 1
    )
    return errors.reset_index()


def _subject_review(metrics, evidence, errors, names):
    if evidence["subject_id"].duplicated().any() or set(evidence["subject_id"]) != set(
        metrics["subject_id"]
    ):
        raise ValueError("QC/ERP证据表不完整或身份重复")
    review = evidence.set_index("subject_id").copy()
    for name in names:
        frame = metrics[metrics["experiment"] == name].set_index("subject_id").loc[review.index]
        for column in (
            "fold",
            "balanced_accuracy",
            "roc_auc",
            "average_precision",
            "target_recall",
            "specificity",
            "target_f1",
            "fn",
            "fp",
            "tn",
            "tp",
        ):
            review[f"{name}__{column}"] = frame[column]
    nn_names = [name for name in DL_NAMES if name in names]
    review["nn_mean_ba"] = review[[f"{name}__balanced_accuracy" for name in nn_names]].mean(axis=1)
    review["nn_mean_ba_rank_ascending"] = review["nn_mean_ba"].rank(method="min").astype(int)
    # 20仅用于解释低Target计数下的粒度；不改变QC，不产生新的受试者排除规则。
    review["low_target_count_descriptive"] = review["target"] < 20
    for subject, group in errors.groupby("subject_id"):
        review.loc[subject, "dl_disagreement_trials"] = int(
            group["dl_prediction_disagreement"].sum()
        )
        common = group["nn_error_models"] == len(nn_names)
        review.loc[subject, "all_nn_wrong_trials"] = int(common.sum())
        review.loc[subject, "all_nn_missed_targets"] = int((common & group["label"].eq(1)).sum())
        review.loc[subject, "all_nn_false_positives"] = int((common & group["label"].eq(0)).sum())
    review["finetune_minus_eegnet_ba"] = (
        review["cbramod_finetune__balanced_accuracy"] - review["eegnet__balanced_accuracy"]
    )
    review["finetune_minus_lr_erp_ba"] = (
        review["cbramod_finetune__balanced_accuracy"] - review["lr_erp__balanced_accuracy"]
    )
    return review.reset_index().sort_values("subject_id")


def _fit_review(dl_directory: Path, manifest, splits, predictions, summary):
    audit = pd.read_csv(dl_directory / "fit_audit.csv")
    expected = {(name, fold["fold"]) for name in summary["models"] for fold in splits["folds"]}
    if (
        audit.duplicated(["experiment", "fold"]).any()
        or set(zip(audit["experiment"], audit["fold"], strict=True)) != expected
    ):
        raise ValueError("DL拟合记录不恰好覆盖三模型五折")
    rows = []
    for row in audit.to_dict("records"):
        name, fold = row["experiment"], int(row["fold"])
        history = pd.read_csv(dl_directory / f"{name}_fold-{fold:02d}/history.csv")
        if not np.isfinite(history.to_numpy()).all() or history["epoch"].tolist() != list(
            range(1, int(row["epochs_run"]) + 1)
        ):
            raise ValueError("训练历史含非有限值或epoch不连续")
        best = history[history["epoch"] == row["best_epoch"]]
        if len(best) != 1 or not np.isclose(
            best.iloc[0]["validation_subject_macro_ba"], row["best_validation_subject_macro_ba"]
        ):
            raise ValueError("best epoch与历史不一致")
        fold_info = next(item for item in splits["folds"] if item["fold"] == fold)
        for part in ("train", "validation"):
            if int(row[f"{part}_trials"]) != len(fold_info["trial_ids"][part]):
                raise ValueError("训练/验证trial计数改变")
        test = predictions[(predictions["experiment"] == name) & (predictions["fold"] == fold)]
        macro = float(subject_metrics(test, 0.5)["balanced_accuracy"].mean())
        rows.append(
            {
                **row,
                "outer_test_subject_macro_ba": macro,
                "validation_minus_test_ba": row["best_validation_subject_macro_ba"] - macro,
                "train_loss_at_best": float(best.iloc[0]["train_weighted_loss"]),
                "validation_loss_at_best": float(best.iloc[0]["validation_weighted_loss"]),
                "last_validation_loss": float(history.iloc[-1]["validation_weighted_loss"]),
                "last_train_loss": float(history.iloc[-1]["train_weighted_loss"]),
                "parameter_count_verified_source": "fit_audit",
                "test_trials": len(test),
                "dataset_subjects": manifest["subject_count"],
            }
        )
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str]], digits: int = 4) -> str:
    def cell(value):
        if pd.isna(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{value:.{digits}f}"
        return str(value).replace("|", "/")

    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell(row[key]) for key, _ in columns) + " |"
        for row in frame.to_dict("records")
    )
    return "\n".join(lines)


def _plots(output, macro, paired, metrics, pooled, review, dl_directory, names, labels):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = output / "figures"
    directory.mkdir()
    colours = {name: plt.get_cmap("tab10")(index) for index, name in enumerate(names)}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        ("balanced_accuracy", "roc_auc", "average_precision"),
        ("Subject-macro BA", "Subject-macro ROC-AUC", "Subject-macro AP"),
        strict=True,
    ):
        frame = macro[macro["metric"] == metric].set_index("experiment").loc[names]
        means = frame["mean"].to_numpy()
        axis.errorbar(
            means,
            np.arange(len(names)),
            xerr=np.vstack([means - frame["ci_low"], frame["ci_high"] - means]),
            fmt="o",
            capsize=4,
        )
        axis.set_yticks(np.arange(len(names)), [labels[name] for name in names])
        axis.invert_yaxis()
        axis.set(title=title + " (95% conditional CI)", xlabel="Score", xlim=(0, 1))
        axis.grid(axis="x", alpha=0.2)
    fig.savefig(directory / "model_comparison.png", dpi=160)
    plt.close(fig)
    frame = paired[paired["metric"] == "balanced_accuracy"]
    fig, axis = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    means = frame["mean_candidate_minus_reference"].to_numpy()
    axis.errorbar(
        means,
        np.arange(len(frame)),
        xerr=np.vstack([means - frame["ci_low"], frame["ci_high"] - means]),
        fmt="o",
        capsize=4,
    )
    axis.set_yticks(
        np.arange(len(frame)),
        [f"{labels[r.candidate]} - {labels[r.reference]}" for r in frame.itertuples()],
    )
    axis.axvline(0, color="black", linestyle="--")
    axis.invert_yaxis()
    axis.set(
        title="Paired subject BA differences | descriptive conditional 95% CI",
        xlabel="Candidate - reference",
    )
    fig.savefig(directory / "paired_ba_differences.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 12), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        ("balanced_accuracy", "target_recall", "specificity"),
        ("Balanced accuracy", "Target recall", "Standard specificity"),
        strict=True,
    ):
        pivot = metrics.pivot(index="subject_id", columns="experiment", values=metric).loc[
            review["subject_id"], names
        ]
        plotted = axis.imshow(pivot, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        axis.set_xticks(
            np.arange(len(names)),
            [labels[name] for name in names],
            rotation=65,
            ha="right",
            fontsize=8,
        )
        axis.set_yticks(np.arange(len(pivot)), pivot.index, fontsize=8)
        axis.set_title(title)
    fig.colorbar(plotted, ax=axes, shrink=0.5, label="OOF subject score")
    fig.savefig(directory / "subject_metric_heatmaps.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    for axis, first in zip(axes, ("lr_erp", "eegnet"), strict=True):
        x, y = review[f"{first}__balanced_accuracy"], review["cbramod_finetune__balanced_accuracy"]
        axis.scatter(x, y, s=25, alpha=0.75)
        axis.plot([0, 1], [0, 1], "--", color="grey")
        for row in review[review["low_target_count_descriptive"]].itertuples():
            index = review.index[review["subject_id"] == row.subject_id][0]
            axis.annotate(row.subject_id, (x.loc[index], y.loc[index]), fontsize=8)
        axis.set(
            xlabel=labels[first] + " BA", ylabel="CBraMod finetune BA", xlim=(0.3, 1), ylim=(0.3, 1)
        )
    fig.savefig(directory / "paired_subject_scatter.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for axis, name in zip(axes.flat, names, strict=True):
        row = pooled[pooled["experiment"] == name].iloc[0]
        matrix = np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int)
        norm = matrix / matrix.sum(axis=1, keepdims=True)
        axis.imshow(norm, vmin=0, vmax=1, cmap="Blues")
        for i in range(2):
            for j in range(2):
                axis.text(
                    j,
                    i,
                    f"{matrix[i, j]}\n{norm[i, j]:.1%}",
                    ha="center",
                    va="center",
                    color="white" if norm[i, j] > 0.6 else "black",
                )
        axis.set(title=labels[name], xlabel="Predicted", ylabel="True")
        axis.set_xticks([0, 1], ["Standard", "Target"])
        axis.set_yticks([0, 1], ["Standard", "Target"])
    fig.savefig(directory / "confusion_matrices.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 5, figsize=(17, 9), constrained_layout=True)
    for row, name in enumerate(DL_NAMES):
        for col in range(5):
            history = pd.read_csv(dl_directory / f"{name}_fold-{col + 1:02d}/history.csv")
            axis = axes[row, col]
            axis.plot(history["epoch"], history["validation_subject_macro_ba"], color=colours[name])
            best_epoch = pd.read_json(
                dl_directory / f"{name}_fold-{col + 1:02d}/fit_audit.json", typ="series"
            )["best_epoch"]
            axis.axvline(best_epoch, linestyle="--", color="grey")
            axis.set(
                title=f"{labels[name]} / fold {col + 1}",
                ylim=(0.4, 0.9),
                xlabel="Epoch",
                ylabel="Validation BA",
            )
    fig.savefig(directory / "validation_learning_curves.png", dpi=140)
    plt.close(fig)
    context = [
        ("erp_measurements__difference_pz_mean_uv", "Pz Target - Standard (uV)"),
        ("preprocessing_summary__epoch_rejection_fraction", "Amplitude rejection fraction"),
        ("target", "Retained Target trials"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for axis, (column, label) in zip(axes, context, strict=True):
        if column in review:
            for name in ("eegnet", "cbramod_finetune"):
                axis.scatter(
                    review[column],
                    review[f"{name}__balanced_accuracy"],
                    label=labels[name],
                    s=20,
                    alpha=0.6,
                )
        axis.set(xlabel=label, ylabel="OOF subject BA", ylim=(0.3, 1))
    axes[0].legend(fontsize=8)
    fig.suptitle("Exploratory context only: association does not establish cause")
    fig.savefig(directory / "error_context.png", dpi=150)
    plt.close(fig)


def _subject_reports(output, review, metrics, names, labels, errors):
    directory = output / "subjects"
    directory.mkdir()
    for row in review.to_dict("records"):
        subject = row["subject_id"]
        frame = (
            metrics[metrics["subject_id"] == subject]
            .set_index("experiment")
            .loc[names]
            .reset_index()
        )
        frame["model"] = frame["experiment"].map(labels)
        text = [
            f"# {subject}：外层测试误差复核",
            "",
            f"该人属于outer test折{int(frame.iloc[0]['fold'])}，"
            f"保留Target {int(row['target'])}、Standard {int(row['standard'])}。",
            "原有QC裁决保持不变；本页仅解释已生成OOF错误。",
            "",
            _markdown_table(
                frame,
                [
                    ("model", "模型"),
                    ("balanced_accuracy", "BA"),
                    ("roc_auc", "ROC-AUC"),
                    ("average_precision", "AP"),
                    ("target_recall", "Target召回"),
                    ("specificity", "Standard特异度"),
                    ("fn", "漏检FN"),
                    ("fp", "误报FP"),
                ],
            ),
            "",
        ]
        for column, label in (
            ("preprocessing_summary__manual_bad_channels", "插值/坏道登记"),
            ("preprocessing_summary__ica_excluded_count", "ICA排除成分数"),
            ("preprocessing_summary__epoch_rejection_fraction", "振幅剔除比例"),
            ("erp_measurements__difference_pz_mean_uv", "Pz 300–600ms条件差（µV）"),
        ):
            if column in row:
                value = row[column]
                text.append(
                    f"- {label}：{'无登记/缺失（参照源表）' if pd.isna(value) else value}。"
                )
        text += [
            "",
            f"微调−EEGNet BA：{row['finetune_minus_eegnet_ba']:+.4f}；"
            f"微调−LR ERP：{row['finetune_minus_lr_erp_ba']:+.4f}。",
            f"三神经网络预测不一致trial：{int(row['dl_disagreement_trials'])}；共同漏检Target：{int(row['all_nn_missed_targets'])}；共同误报Standard：{int(row['all_nn_false_positives'])}。",
        ]
        if row["low_target_count_descriptive"]:
            text += [
                "",
                "Target数量较少：每漏检一个Target会使召回率改变1/N_target，BA改变1/(2*N_target)。",
                f"本人的单个Target对BA影响约{1 / (2 * row['target']):.4f}，"
                "不能把小幅模型差值当稳定个体优势。",
            ]
        text += [
            "",
            "复核方法：先看FN/FP是否集中在一类；再看ERP条件差和原QC记录。",
            "共同错误的trial_id、原trial_index、epoch_index和各模型分数见trial_error_audit.csv。",
            "若要看原波形，可按source_file与epoch_index定位FIF；此报告未自动判定残余伪迹或改变排除规则。",
        ]
        common = errors[(errors["subject_id"] == subject) & (errors["nn_error_models"] == 3)]
        if len(common):
            text += [
                "",
                "前三个共同错误定位（按原清单顺序，不按波形质量排序）：",
                "",
                _markdown_table(
                    common.head(3),
                    [
                        ("trial_id", "trial_id"),
                        ("label", "真实标签"),
                        ("trial_index", "原trial序号"),
                        ("epoch_index", "FIF epoch索引"),
                    ],
                ),
            ]
        (directory / f"{subject}.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def _project_report(output, manifest, config, macro, paired, pooled, review, fit_review, erp):
    names = list(config["models"])
    labels = {name: item["label"] for name, item in config["models"].items()}
    overview = pd.DataFrame({"experiment": names, "model": [labels[name] for name in names]})
    for metric in METRICS:
        frame = macro[macro["metric"] == metric].set_index("experiment")
        overview[metric] = [frame.loc[name, "mean"] for name in names]
        if metric == "balanced_accuracy":
            overview["ba_ci"] = [
                f"[{frame.loc[name, 'ci_low']:.4f}, {frame.loc[name, 'ci_high']:.4f}]"
                for name in names
            ]
    overview.to_csv(output / "model_overview.csv", index=False)
    pair_table = paired[paired["metric"] == "balanced_accuracy"].copy()
    pair_table["comparison"] = (
        pair_table["candidate"].map(labels) + " − " + pair_table["reference"].map(labels)
    )
    nn = overview.set_index("experiment")
    ft_diff = pair_table[
        (pair_table["reference"] == "eegnet") & (pair_table["candidate"] == "cbramod_finetune")
    ].iloc[0]
    text = [
        "# ERP CORE P3b：信号处理与跨受试者分类最终报告",
        "",
        "## 1. 本轮数据支持的结论",
        "",
        f"共同裁决版：{manifest['subject_count']}名受试者、{manifest['trial_count']}个trial。三组LR和三种神经网络均完成固定五折。",
        f"EEGNet的受试者等权BA为{nn.loc['eegnet', 'balanced_accuracy']:.4f}；"
        f"CBraMod微调为{nn.loc['cbramod_finetune', 'balanced_accuracy']:.4f}；"
        f"冻结表示为{nn.loc['cbramod_linear', 'balanced_accuracy']:.4f}。",
        f"微调−EEGNet的配对BA差为{ft_diff.mean_candidate_minus_reference:+.4f}，"
        f"条件bootstrap 95% CI [{ft_diff.ci_low:.4f}, {ft_diff.ci_high:.4f}]。",
        "均值排名是当前一次实验的描述；若配对区间跨0，数据不足以支持两模型存在稳定方向的差异，也不等价于已证明等效。",
        "",
        "## 2. 项目完成了什么",
        "",
        "ERP CORE P3原始40人→结构/事件验证→波形、PSD和行为QC→滤波、平均参考、坏道插值、ICA→"
        "trial分段和200µV峰峰值QC→人工裁决后39人→经典P3b→紧凑特征/LR→"
        "EEGNet与CBraMod→共同OOF误差分析。",
        "sub-035/PO3和sub-039/Oz经过坏道登记；sub-030按既有QC裁决排除。Target和Standard标签来自刺激/行为配对，不凭文件行序猜标签。",
        "完整长窗Morlet/ITC支线仍有保护支持候选待复核；本项目已经完成的短窗DWT特征不等于完整长窗TFR。",
        "",
        "## 3. 经典ERP证据与分类问题的区别",
        "",
    ]
    if erp is not None:
        main = erp[
            (erp["analysis_role"] == "primary") & (erp["test"] == "primary_paired_t_test")
        ].iloc[0]
        text += [
            f"预先定义Pz 300–600ms：Target均值{main.target_mean:.3f}µV，"
            f"Standard {main.standard_mean:.3f}µV，条件差{main.mean_difference:.3f}µV，"
            f"95% CI [{main.ci_low:.3f}, {main.ci_high:.3f}]µV，"
            f"t({main.degrees_of_freedom:g})={main.statistic:.3f}，"
            f"p={main.p_value:.3g}，dz={main.effect_size:.3f}。"
        ]
    text += [
        "ERP先在每人条件内平均，统计单位是受试者；分类每trial预测且测试受试者未出现在该折训练中。",
        "群体平均出现清晰P3b，不意味着每条trial易分类；单trial噪声、条件内变异和受试者差异仍影响预测。",
        "",
        "## 4. 共同评价协议",
        "",
        "三组LR分别使用12列ERP、24列ERP+功率、40列ERP+功率+DWT；StandardScaler与LR只在train拟合。",
        "EEGNet使用256Hz、257点原生30通道EEG；CBraMod使用200Hz、200点、单patch，固定100µV缩放。",
        "保持既有0.1–30Hz、ERP电压baseline和最终平均参考；适配使用零点锚定重采样及短窗反射边缘支持。",
        "outer StratifiedGroupKFold五折，inner按受试者留出验证。"
        "每模型每trial恰好一次outer test预测。",
        "类别权重仅用该折train；阈值固定0.5。神经网络按验证受试者macro BA早停，"
        "每折重新初始化或载入原预训练权重。",
        "DL基础训练seed=2026，每折seed=2026+fold；划分seed=42。单训练种子设置，未量化跨种子训练方差。",
        "",
        "## 5. 六模型正式对照",
        "",
        _markdown_table(
            overview,
            [
                ("model", "模型"),
                ("balanced_accuracy", "macro BA"),
                ("ba_ci", "95% CI"),
                ("roc_auc", "macro ROC-AUC"),
                ("average_precision", "macro AP"),
                ("target_recall", "Target召回"),
                ("specificity", "特异度"),
                ("target_f1", "Target F1"),
            ],
        ),
        "",
        "![模型均值与区间](figures/model_comparison.png)",
        "",
        "macro先逐人计算再等权平均；pooled把全部trial合并，trial多的人贡献更大，仅作辅助。ROC/AP的pooled还混合不同折模型的分数刻度。",
        "AP使用average precision，其无信息参照取决于Target比例。"
        "Target少时仅报告总体accuracy容易误导。",
        "",
        "## 6. 受试者配对差值",
        "",
        _markdown_table(
            pair_table,
            [
                ("comparison", "candidate−reference"),
                ("mean_candidate_minus_reference", "平均BA差"),
                ("ci_low", "CI下界"),
                ("ci_high", "CI上界"),
                ("candidate_higher_subjects", "提高人数"),
                ("candidate_lower_subjects", "降低人数"),
                ("tied_subjects", "同分人数"),
            ],
        ),
        "",
        "![配对差值](figures/paired_ba_differences.png)",
        "",
        "每次重采样保留同一受试者两个模型的配对；不能把两条独立均值CI的端点简单相减。",
        "bootstrap 2000次、95%区间，条件于本次固定划分和拟合模型；"
        "不覆盖重新训练不确定性和交叉验证训练集重叠的全部依赖。",
        "本轮对照在OOF结果生成后整理，为描述性比较，未做多重比较校正、未运行标签置换，不作确认性显著性或机会水平检验声明。",
        "",
        "## 7. 漏检与误报",
        "",
    ]
    error_table = pooled.copy()
    error_table["model"] = error_table["experiment"].map(labels)
    text += [
        _markdown_table(
            error_table,
            [
                ("model", "模型"),
                ("tp", "Target检出TP"),
                ("fn", "漏检FN"),
                ("fp", "Standard误报FP"),
                ("tn", "正确Standard TN"),
                ("target_precision", "pooled precision"),
                ("target_recall", "pooled recall"),
            ],
        ),
        "",
        "![混淆矩阵](figures/confusion_matrices.png)",
        "",
        "图中每行按真实类别归一化；上行右格是Standard误报率，下行左格是Target漏检率。",
        "召回提高可能伴随误报增多，应同时看特异度、precision和AP。加权训练分数未做概率校准，不能直接解释为疾病概率。",
        "",
        "## 8. 逐受试者复核",
        "",
        "39人都有独立subjects/sub-XXX.md，包含6模型指标、FN/FP、既有QC/ICA/ERP证据、三神经网络共同错误定位。",
        "subject_review.csv包含全部个体，trial_error_audit.csv保留原trial_id、trial_index、epoch_index、source_file和各模型分数。",
        "![个体指标热图](figures/subject_metric_heatmaps.png)",
        "",
        "热图按受试者编号固定排序；三张图分别为BA、Target召回和Standard特异度。亮/暗说明预测差异，不单独证明伪迹或生理原因。",
        "",
        "三神经网络平均BA最低的5人（只用于解释错误，不作为新增排除名单）：",
        "",
        _markdown_table(
            review.nsmallest(5, "nn_mean_ba"),
            [
                ("subject_id", "受试者"),
                ("target", "Target数"),
                ("eegnet__balanced_accuracy", "EEGNet BA"),
                ("cbramod_linear__balanced_accuracy", "冻结BA"),
                ("cbramod_finetune__balanced_accuracy", "微调BA"),
                ("all_nn_missed_targets", "共同漏检"),
                ("all_nn_false_positives", "共同误报"),
            ],
        ),
        "",
        "Target<20者单独标注：小样本下每个FN对召回的影响为1/N_target，对BA为1/(2*N_target)，个体差异容易呈离散跳变。",
        "![配对个体散点](figures/paired_subject_scatter.png)",
        "",
        "点在对角线上方表示该人微调BA更高；图中标注低Target计数者。关注收益是否广泛及有没有严重退化的个体。",
        "![错误与已有信号证据](figures/error_context.png)",
        "",
        "这些散点仅连接已有ERP效应/剔除比例/Target计数与测试BA，不能证明因果；ERP证据由该人全部保留trial计算，仅用于事后解释，没有进入训练选择。",
        "",
        "## 9. 训练与计算开销",
        "",
        "![十五次验证曲线](figures/validation_learning_curves.png)",
        "",
        "虚线是验证选出的最佳epoch；测试组没有参与选checkpoint。fit_review.csv给出验证−测试BA和最佳/最后loss，解释过拟合。",
    ]
    costs = (
        fit_review.groupby("experiment")
        .agg(
            fits=("fold", "size"),
            total_fit_seconds=("elapsed_seconds", "sum"),
            best_epoch_min=("best_epoch", "min"),
            best_epoch_max=("best_epoch", "max"),
            trainable_parameters=("trainable_parameter_count", "max"),
            peak_cuda_mib=("peak_cuda_allocated_bytes", lambda s: s.max() / 1024**2),
        )
        .reset_index()
    )
    costs["model"] = costs["experiment"].map(labels)
    costs.to_csv(output / "compute_summary.csv", index=False)
    text += [
        "",
        _markdown_table(
            costs,
            [
                ("model", "模型"),
                ("fits", "拟合数"),
                ("total_fit_seconds", "拟合总秒数"),
                ("trainable_parameters", "可训练参数"),
                ("peak_cuda_mib", "峰值分配MiB"),
                ("best_epoch_min", "最佳epoch最小"),
                ("best_epoch_max", "最大"),
            ],
            digits=2,
        ),
        "",
        "拟合秒数来自fit_audit，包含该函数的验证与检查点处理，"
        "不等于进程总耗时或纯GPU kernel时间；显存是PyTorch分配峰值。",
        "",
        "## 10. 限制与结果该如何使用",
        "",
        "1. EEGNet和CBraMod输入网格不同，模型对照同时包含重采样等适配因素；"
        "linear与finetune的输入/分类头一致，更直接反映更新编码器的增量。",
        "2. CBraMod只有一个时间patch，不能据此验证多patch时间关系的优势；"
        "ERP滤波和baseline与预训练分布存在差异。",
        "3. 预处理包括每名受试者整段记录的无监督ICA/参考/QC，"
        "评价适用于离线已获得记录的受试者分组场景；严格在线/完全未知个体预处理尚未验证。",
        "4. 39人为单数据集、小样本；没有外部队列、跨任务、跨设备验证。"
        "模型解码Target/Standard，不能推断临床诊断性能。",
        "5. 当前只有一个训练种子设置；共享训练集的CV与条件bootstrap不能概括所有重新训练不确定性。",
        "6. 观察OOF后继续调参、挑种子或删难例会使当前OOF变成开发信息。"
        "以后扩展须明确探索性并另设外部/未使用验证。",
        "7. 前期08视觉状态pending作为来源记录保留；12已记录接受短窗限制及新增提示0。"
        "命令行接受和详细人工复核日志各有独立用途。",
        "",
        "## 11. 复现与产物阅读顺序",
        "",
        "在erp-core-p3b-dl环境按09→10→11和08→12→13运行已有入口，配置见docs/13、docs/14。",
        "本报告仅消费已完成产物：",
        "",
        "```powershell",
        "python scripts/14_build_final_report.py",
        "```",
        "",
        "重跑需新输出目录：",
        "",
        "```powershell",
        "python scripts/14_build_final_report.py --output-dir results/final_report/v2",
        "```",
        "",
        "优先阅读model_overview.csv和paired_comparisons.csv，再看热图和39人报告；定位trial使用trial_error_audit.csv。",
        "source_provenance.json保存上游摘要hash、数据/划分身份和ERP来源；run_summary.json保存本报告配置与产物hash。",
        "",
        "## 12. 简历和面试表述",
        "",
        "基于ERP CORE P3数据构建可复现EEG流水线，完成40人数据验证与QC、"
        "滤波/重参考/坏道插值/ICA及39人P3b统计；在7207个trial上采用固定受试者分组五折，"
        "对照紧凑ERP/功率/小波特征LR、EEGNet与CBraMod冻结/微调，"
        f"EEGNet受试者macro BA={nn.loc['eegnet', 'balanced_accuracy']:.3f}、"
        f"macro ROC-AUC={nn.loc['eegnet', 'roc_auc']:.3f}，提供配对置信区间与逐受试者误差审计。",
        "面试说明重点：为何trial随机切分会泄漏个体信息；群体ERP显著和单trial分类的区别；"
        "ICA与PCA职责不同；为何Foundation Model在小数据上不必优于轻量模型。",
        "本次基础闭环已完成；外部验证、多种子、长窗TFR和解释性实验是后续扩展，不在简历中写成已经完成。",
    ]
    (output / "project_final_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def build_report(config_path: Path, root: Path, *, output: Path | None = None) -> Path:
    """从已有正式产物生成新版本报告，不读原生EEG、不加载模型或重新训练。"""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = {key: resolve_path(root, value) for key, value in config["inputs"].items()}
    trials, manifest, splits = load_classification_dataset(paths["classification_directory"])
    dl = verify_artifact(paths["dl_directory"], "deep_learning_training")
    lr = verify_artifact(paths["lr_directory"], "logistic_baseline")
    if (
        dl.get("mode") != "full"
        or not dl.get("test_evaluated")
        or lr.get("mode") != "full"
        or not lr.get("external_test_evaluated")
    ):
        raise ValueError("必须使用正式full产物，smoke不能生成最终测试报告")
    split_hash = canonical_hash(splits)
    for label, summary, path in (
        ("dl", dl, paths["dl_directory"]),
        ("lr", lr, paths["lr_directory"]),
    ):
        if (
            summary["dataset_hash"] != manifest["dataset_hash"]
            or summary["split_hash"] != split_hash
        ):
            raise ValueError(f"{label}数据/划分版本不一致")
        if (
            canonical_hash(json.loads((path / "subject_splits.json").read_text(encoding="utf-8")))
            != split_hash
        ):
            raise ValueError("训练目录实际划分文件与共同版本不同")
    dl_config = json.loads(
        (paths["dl_directory"] / "config_snapshot.json").read_text(encoding="utf-8")
    )
    lr_config = json.loads(
        (paths["lr_directory"] / "config_snapshot.json").read_text(encoding="utf-8")
    )
    threshold = float(config["prediction_threshold"])
    if (
        threshold != 0.5
        or dl_config["config"]["training"]["prediction_threshold"] != threshold
        or lr_config["model"]["prediction_threshold"] != threshold
    ):
        raise ValueError("正式报告固定沿用训练时0.5阈值，不允许测试后改阈值")
    model_map = config["models"]
    if not set(DL_NAMES).issubset(model_map) or "lr_erp" not in model_map:
        raise ValueError("报告要求三神经网络和固定LR ERP参考")
    frames = []
    for source, filename in (("dl", "oof_predictions.csv"), ("lr", "test_predictions.csv")):
        selected = {
            item["experiment"]: name for name, item in model_map.items() if item["source"] == source
        }
        if len(selected) != sum(item["source"] == source for item in model_map.values()):
            raise ValueError("同一来源模型重复映射")
        frame = validate_oof(
            pd.read_csv(paths[f"{source}_directory"] / filename),
            trials,
            splits,
            list(selected),
            threshold,
        )
        frame["experiment"] = frame["experiment"].map(selected)
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    names = list(model_map)
    metrics = subject_metrics(predictions, threshold)
    macro, paired = compare_subjects(metrics, names, config["contrasts"], config["statistics"])
    pooled = pd.DataFrame(
        [
            {
                "experiment": name,
                **binary_metrics(group["label"].to_numpy(), group["score"].to_numpy(), threshold),
            }
            for name, group in predictions.groupby("experiment")
        ]
    )
    errors = trial_errors(predictions, trials, names, threshold)
    evidence = pd.read_csv(paths["classification_directory"] / "subject_evidence_summary.csv")
    review = _subject_review(metrics, evidence, errors, names)
    fit_review = _fit_review(
        paths["dl_directory"],
        manifest,
        splits,
        predictions[predictions["experiment"].isin(DL_NAMES)],
        dl,
    )
    erp_path = paths["erp_statistics"]
    erp = pd.read_csv(erp_path) if erp_path.exists() else None
    output = resolve_path(root, output or config["outputs"]["directory"])
    with new_artifact_run(root, output, "final_report") as summary:
        write_json(output / "config_snapshot.json", config)
        provenance = {
            "inputs": {key: str(path) for key, path in paths.items()},
            "dataset_hash": manifest["dataset_hash"],
            "split_hash": split_hash,
            "dl_input_protocol_hash": dl["input_protocol_hash"],
            "source_summary_sha256": {
                key: sha256_file(paths[key] / "run_summary.json")
                for key in ("classification_directory", "dl_directory", "lr_directory")
            },
            "erp_statistics_sha256": sha256_file(erp_path) if erp is not None else None,
            "dl_seed": dl["seed"],
            "dl_device": dl["device"],
            "dl_gpu_name": dl["gpu_name"],
            "dl_torch_version": dl["torch_version"],
            "dl_deterministic": dl["deterministic"],
        }
        write_json(output / "source_provenance.json", provenance)
        for filename, frame in (
            ("subject_metrics.csv", metrics),
            ("subject_macro_ci.csv", macro),
            ("paired_comparisons.csv", paired),
            ("pooled_metrics.csv", pooled),
            ("trial_error_audit.csv", errors),
            ("subject_review.csv", review),
            ("fit_review.csv", fit_review),
        ):
            frame.to_csv(output / filename, index=False)
        references = subject_metrics(
            trials.assign(experiment="standard_only", fold=1, score=0.0), threshold
        )
        references.to_csv(output / "standard_only_subject_reference.csv", index=False)
        _plots(
            output,
            macro,
            paired,
            metrics,
            pooled,
            review,
            paths["dl_directory"],
            names,
            {name: item["label"] for name, item in model_map.items()},
        )
        _subject_reports(
            output,
            review,
            metrics,
            names,
            {name: item["label"] for name, item in model_map.items()},
            errors,
        )
        _project_report(output, manifest, config, macro, paired, pooled, review, fit_review, erp)
        summary.update(
            dataset_hash=manifest["dataset_hash"],
            split_hash=split_hash,
            models=names,
            subjects=manifest["subject_count"],
            trials=manifest["trial_count"],
            predictions_verified=len(predictions),
            threshold=threshold,
            refitted_models=0,
            new_subject_exclusions=0,
            bootstrap_conditional=True,
            multiple_comparison_adjustment=False,
            source_provenance=provenance,
        )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path, default=Path("configs/final_report.yaml"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        output = build_report(
            resolve_path(args.project_root, args.config), args.project_root, output=args.output_dir
        )
    except Exception as error:
        print(f"Final report failed: {error}", file=sys.stderr)
        return 1
    print(f"Final report: {output}")
    return 0
