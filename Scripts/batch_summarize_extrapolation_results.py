"""
Batch summary script for extrapolation experiment results.

The script scans output/extrapolation_results, reads per-model final test
metrics plus representative-fold metrics, and exports a unified summary layout
for easier cross-model / cross-mode comparison.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from _raw_prediction_stats import read_prediction_csv, resolve_prediction_columns, summarize_optuna_predictions


plt.style.use("seaborn-v0_8-white")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["xtick.labelsize"] = 13
plt.rcParams["ytick.labelsize"] = 13
plt.rcParams["legend.fontsize"] = 12
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["figure.dpi"] = 300


MODEL_MAP = {
    "catboost_results": "CatBoost",
    "lightgbm_results": "LightGBM",
    "mlp_results": "MLP",
    "sklearn_rf_results": "RF",
    "xgboost_results": "XGB",
}

MODEL_ORDER = ["CatBoost", "LightGBM", "MLP", "RF", "XGB"]
SUMMARY_TABLES_DIRNAME = "00_summary_tables"
CASES_DIRNAME = "01_alloy_cases"
R2_LABEL = "R\u00b2"


def safe_name(text: str) -> str:
    return (
        text.replace("(", "")
        .replace(")", "")
        .replace("%", "pct")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def copy_tree_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def rename_export_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    renamed = renamed.rename(
        columns={
            "target": "property",
            "representative_train_r2": "global_mean_train_r2",
            "representative_val_r2": "global_mean_val_r2",
            "representative_test_r2": "global_mean_test_r2",
            "representative_test_r2_std": "global_mean_test_r2_std",
            "representative_train_rmse": "global_mean_train_rmse",
            "representative_val_rmse": "global_mean_val_rmse",
            "representative_test_rmse": "global_mean_test_rmse",
            "representative_test_rmse_std": "global_mean_test_rmse_std",
            "representative_train_mae": "global_mean_train_mae",
            "representative_val_mae": "global_mean_val_mae",
            "representative_test_mae": "global_mean_test_mae",
            "representative_test_mae_std": "global_mean_test_mae_std",
            "representative_predictions_file": "global_mean_predictions_file",
            "representative_plot_file": "global_mean_plot_file",
            "representative_trial_id": "global_mean_trial_id",
            "representative_fold": "global_mean_fold",
            "representative_closest_mae": "global_mean_closest_mae",
        }
    )
    return renamed


def extract_final_metrics(metrics: Dict) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    pattern = re.compile(r"^final_model_evaluation_(train|val|test)_(.+)_(r2|rmse|mae)$")

    for key, value in metrics.items():
        match = pattern.match(key)
        if not match:
            continue
        split, target, metric_name = match.groups()
        target_metrics = results.setdefault(target, {})
        target_metrics[f"final_{split}_{metric_name}"] = value

    return results


def extract_representative_metrics(metrics: Dict) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    pattern = re.compile(
        r"^closest_to_global_mean_trial_fold_(train|val|test)_(.+)_(r2|rmse|mae)$"
    )

    for key, value in metrics.items():
        match = pattern.match(key)
        if not match:
            continue
        split, target, metric_name = match.groups()
        target_metrics = results.setdefault(target, {})
        target_metrics[f"representative_{split}_{metric_name}"] = value

    return results


def extract_closest_to_mean_metrics(metrics: Dict) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    pattern = re.compile(r"^closest_to_mean_evaluation_(train|val|test)_(.+)_(r2|rmse|mae)$")

    for key, value in metrics.items():
        match = pattern.match(key)
        if not match:
            continue
        split, target, metric_name = match.groups()
        target_metrics = results.setdefault(target, {})
        target_metrics[f"closest_mean_{split}_{metric_name}"] = value

    return results


def extract_closest_fold_info(info: Dict) -> Dict[str, float]:
    result: Dict[str, float] = {}
    if not info:
        return result

    if "closest_fold_num" in info:
        result["closest_fold_num"] = info["closest_fold_num"]
    if "fold_test_r2" in info:
        result["closest_fold_test_r2"] = info["fold_test_r2"]
    if "mean_test_r2" in info:
        result["closest_fold_mean_test_r2"] = info["mean_test_r2"]

    per_target = info.get("test_r2_per_target", {})
    for target, value in per_target.items():
        result[f"{target}__closest_fold_test_r2"] = value

    return result


def style_comparison_axes(ax, metric_label: str, grid_alpha: float = 0.3) -> None:
    if metric_label == "R2":
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax.tick_params(axis="both", which="major", direction="in", length=6, width=1.2, top=True, right=True)
    ax.tick_params(axis="both", which="minor", direction="in", length=3, width=1.0, top=True, right=True)
    ax.grid(axis="y", alpha=grid_alpha, linestyle="--")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


def plot_metric_comparison(
    case_df: pd.DataFrame,
    output_path: Path,
    case_label: str,
    value_col: str,
    metric_label: str,
    legend_label: str,
) -> None:
    if case_df.empty:
        return

    ordered_df = case_df.copy()
    ordered_df["model"] = pd.Categorical(ordered_df["model"], categories=MODEL_ORDER, ordered=True)
    ordered_df = ordered_df.sort_values("model").dropna(subset=[value_col])
    if ordered_df.empty:
        return

    x = np.arange(len(ordered_df))
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.bar(
        x,
        ordered_df[value_col],
        width=0.55,
        color="#ffb703",
        edgecolor="black",
        linewidth=1.2,
        label=legend_label,
    )

    ax.set_xlabel("Predictive models", fontweight="bold")
    ax.set_ylabel(R2_LABEL if metric_label == "R2" else metric_label, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered_df["model"])
    style_comparison_axes(ax, metric_label)
    ax.legend(frameon=True, edgecolor="black")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_three_metric_comparison(
    case_df: pd.DataFrame,
    output_path: Path,
    case_label: str,
    metric_label: str,
    final_col: str,
    closest_mean_col: str,
    global_mean_col: str,
) -> None:
    if case_df.empty:
        return

    ordered_df = case_df.copy()
    ordered_df["model"] = pd.Categorical(ordered_df["model"], categories=MODEL_ORDER, ordered=True)
    ordered_df = ordered_df.sort_values("model").dropna(subset=[final_col])
    if ordered_df.empty:
        return

    x = np.arange(len(ordered_df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.bar(
        x - width,
        ordered_df[final_col],
        width=width,
        color="#8ecae6",
        edgecolor="black",
        linewidth=1.2,
        label=f"Final extrapolation test {metric_label}",
    )
    ax.bar(
        x,
        ordered_df[closest_mean_col],
        width=width,
        color="#ffb703",
        edgecolor="black",
        linewidth=1.2,
        label=f"Best-trial representative {metric_label}",
    )
    ax.bar(
        x + width,
        ordered_df[global_mean_col],
        width=width,
        color="#90be6d",
        edgecolor="black",
        linewidth=1.2,
        label=f"Global representative {metric_label}",
    )

    ax.set_xlabel("Predictive models", fontweight="bold")
    ax.set_ylabel(R2_LABEL if metric_label == "R2" else metric_label, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered_df["model"])
    style_comparison_axes(ax, metric_label)
    ax.legend(frameon=True, edgecolor="black")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_dataset_property_summary(
    dataset_df: pd.DataFrame,
    output_path: Path,
    value_col: str,
    metric_label: str = "R2",
    err_col: str | None = None,
) -> None:
    plot_df = dataset_df[["model", "target", value_col]].dropna().copy()
    if plot_df.empty:
        return

    if err_col and err_col in dataset_df.columns:
        plot_df = plot_df.merge(dataset_df[["model", "target", err_col]], on=["model", "target"], how="left")
    else:
        err_col = None

    properties = sorted(plot_df["target"].unique())
    model_positions = np.arange(len(MODEL_ORDER))
    width = 0.8 / max(len(properties), 1)
    colors = ["#f4a698", "#9bbfe0", "#c8d5b9", "#d4a5d6", "#f6d186"]

    fig, ax = plt.subplots(figsize=(10, 8))
    for idx, prop in enumerate(properties):
        prop_df = plot_df[plot_df["target"] == prop].copy()
        prop_df["model"] = pd.Categorical(prop_df["model"], categories=MODEL_ORDER, ordered=True)
        prop_df = prop_df.sort_values("model").set_index("model").reindex(MODEL_ORDER).reset_index()
        y_values = prop_df[value_col].tolist()
        offset = (idx - (len(properties) - 1) / 2) * width
        ax.bar(
            model_positions + offset,
            y_values,
            yerr=prop_df[err_col].fillna(0).tolist() if err_col else None,
            width=width,
            capsize=3,
            color=colors[idx % len(colors)],
            edgecolor="black",
            linewidth=1.1,
            error_kw={"elinewidth": 1.2, "ecolor": "black"},
            label=prop,
        )

    ax.set_xlabel("Predictive models", fontweight="bold")
    ax.set_ylabel(R2_LABEL if metric_label == "R2" else metric_label, fontweight="bold")
    ax.set_xticks(model_positions)
    ax.set_xticklabels(MODEL_ORDER)
    if metric_label == "R2" and (plot_df[value_col] >= 0).all():
        ax.set_ylim(0.0, 1.0)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis="both", which="major", direction="in", length=6, width=1.2, top=True, right=True)
    ax.tick_params(axis="both", which="minor", direction="in", length=3, width=1.0, top=True, right=True)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.legend(frameon=True, edgecolor="black")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_dataset_comparison_summary(dataset_df: pd.DataFrame, comparisons_dir: Path) -> None:
    comparison_cols = [
        "target",
        "model",
        "final_test_r2",
        "final_test_mae",
        "final_test_rmse",
        "closest_mean_test_r2",
        "closest_mean_test_r2_std",
        "closest_mean_test_mae",
        "closest_mean_test_mae_std",
        "closest_mean_test_rmse",
        "closest_mean_test_rmse_std",
        "representative_test_r2",
        "representative_test_r2_std",
        "representative_test_mae",
        "representative_test_mae_std",
        "representative_test_rmse",
        "representative_test_rmse_std",
    ]
    available_cols = [col for col in comparison_cols if col in dataset_df.columns]
    export_df = dataset_df[available_cols].sort_values(["target", "model"]).reset_index(drop=True).rename(
        columns={
            "target": "property",
            "representative_test_r2": "global_mean_test_r2",
            "representative_test_r2_std": "global_mean_test_r2_std",
            "representative_test_mae": "global_mean_test_mae",
            "representative_test_mae_std": "global_mean_test_mae_std",
            "representative_test_rmse": "global_mean_test_rmse",
            "representative_test_rmse_std": "global_mean_test_rmse_std",
        }
    )
    save_csv(export_df, comparisons_dir / "dataset_metric_summary.csv")


def plot_diagonal_chart(file_path: Path, property_name: str, model_name: str, output_path: Path) -> None:
    try:
        df = read_prediction_csv(file_path)
    except Exception as exc:
        print(f"[WARN] Failed to read predictions file {file_path}: {exc}")
        return

    if df is None:
        return

    dataset_col, actual_col, pred_col = resolve_prediction_columns(df, property_name)
    if actual_col is None or pred_col is None:
        return

    dataset_config = {
        "Train": {"color": "#4CAF50", "marker": "o", "label": "Training Set", "alpha": 0.5, "s": 60},
        "Validation": {"color": "#2196F3", "marker": "s", "label": "Validation Set", "alpha": 0.6, "s": 70},
        "Test": {"color": "#FF5722", "marker": "^", "label": "Test Set", "alpha": 0.7, "s": 80},
    }
    dataset_aliases = {
        "Train": {"train", "training"},
        "Validation": {"validation", "valid", "val"},
        "Test": {"test", "testing", "extrapolationtest", "extrapolation_test", "extrapolation test"},
    }

    plt.figure(figsize=(8, 8))
    has_dataset_col = dataset_col is not None
    min_val = float("inf")
    max_val = float("-inf")
    metrics_text: List[str] = []
    plotted_any = False

    if has_dataset_col:
        dataset_series = (
            df[dataset_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "", regex=False)
            .str.replace("_", "", regex=False)
        )
        for dataset_key, config in dataset_config.items():
            matching_rows = pd.Series([False] * len(df))
            for alias in dataset_aliases[dataset_key]:
                normalized_alias = alias.replace(" ", "").replace("_", "")
                matching_rows |= dataset_series == normalized_alias

            valid_df = df.loc[matching_rows, [actual_col, pred_col]].dropna()
            if valid_df.empty:
                continue

            plt.scatter(
                valid_df[actual_col],
                valid_df[pred_col],
                alpha=config["alpha"],
                s=config["s"],
                edgecolors="black",
                linewidth=0.8,
                c=config["color"],
                marker=config["marker"],
                label=config["label"],
            )

            min_val = min(min_val, valid_df[actual_col].min(), valid_df[pred_col].min())
            max_val = max(max_val, valid_df[actual_col].max(), valid_df[pred_col].max())
            plotted_any = True

            y_true = valid_df[actual_col].values
            y_pred = valid_df[pred_col].values
            metrics_text.append(
                f"{config['label']}:\n  R² = {r2_score(y_true, y_pred):.4f}, "
                f"MAE = {mean_absolute_error(y_true, y_pred):.2f}, "
                f"RMSE = {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}"
            )

    if not plotted_any:
        plt.close()
        return

    margin = (max_val - min_val) * 0.05 if max_val > min_val else 1.0
    plt.plot(
        [min_val - margin, max_val + margin],
        [min_val - margin, max_val + margin],
        "r--",
        linewidth=2.5,
        label="Perfect Prediction (y=x)",
    )
    plt.xlabel(f"Experimental {property_name}", fontsize=16, fontweight="bold")
    plt.ylabel(f"Predicted {property_name}", fontsize=16, fontweight="bold")
    plt.title(f"{model_name} - {property_name}", fontsize=18, fontweight="bold")
    plt.legend(loc="upper left", fontsize=12, frameon=True, edgecolor="black", fancybox=False)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.axis("equal")
    plt.xlim(min_val - margin, max_val + margin)
    plt.ylim(min_val - margin, max_val + margin)

    if metrics_text:
        plt.text(
            0.98,
            0.02,
            "\n\n".join(metrics_text),
            transform=plt.gca().transAxes,
            fontsize=11,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black", linewidth=1.5),
            family="monospace",
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def find_model_comparison_dirs(base_dir: Path) -> List[Path]:
    return sorted(
        [
            path
            for path in base_dir.rglob("model_comparison")
            if path.is_dir() and path.parent.name != "all_traditional_extrapolation_summary"
        ]
    )


def process_case(model_comparison_dir: Path, base_dir: Path) -> List[Dict]:
    relative_parts = model_comparison_dir.relative_to(base_dir).parts
    case_parts = relative_parts[:-1]
    rows: List[Dict] = []

    for model_dir in sorted(model_comparison_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = MODEL_MAP.get(model_dir.name)
        if model_name is None:
            continue

        final_metrics = extract_final_metrics(load_json(model_dir / "final_model_evaluation_metrics.json"))
        raw_summaries = summarize_optuna_predictions(
            model_dir / "predictions" / "optuna_trials",
            selection_metric="mae",
            global_prefix="representative",
            mean_prefix="closest_mean",
        )

        targets = sorted(set(final_metrics) | set(raw_summaries))
        for target in targets:
            row = {
                "case_id": "__".join(case_parts),
                "case_path": str(model_comparison_dir),
                "alloy_family": case_parts[0] if len(case_parts) > 0 else "",
                "dataset_name": case_parts[1] if len(case_parts) > 1 else "",
                "target": target,
                "feature_mode": case_parts[3] if len(case_parts) > 3 else "",
                "model": model_name,
                "model_dir": str(model_dir),
            }
            row.update(final_metrics.get(target, {}))
            row.update(raw_summaries.get(target, {}))

            row["final_predictions_file"] = str(model_dir / "predictions" / "all_predictions.csv")
            row.setdefault("representative_predictions_file", "")
            row.setdefault("closest_mean_predictions_file", "")
            row["final_plot_file"] = str(model_dir / "plots" / "final_model_evaluation_all_sets_comparison.png")
            row["representative_plot_file"] = ""
            row["closest_mean_plot_file"] = ""
            rows.append(row)

    return rows


def build_best_rows(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.assign(
            final_test_r2=df["final_test_r2"].fillna(-np.inf),
            closest_mean_test_r2=df["closest_mean_test_r2"].fillna(-np.inf),
            representative_test_r2=df["representative_test_r2"].fillna(-np.inf),
        )
        .sort_values(
            ["target", "final_test_r2", "closest_mean_test_r2", "representative_test_r2"],
            ascending=[True, False, False, False],
            na_position="last",
        )
        .groupby("target", group_keys=False)
        .head(1)
        .reset_index(drop=True)
    )


def select_mode_row(property_df: pd.DataFrame, mode: str, selection_metric: str) -> pd.Series | None:
    if property_df.empty:
        return None

    if mode == "final":
        value_col = "final_test_r2"
        ascending = False
        tie_breakers = ["final_test_mae", "closest_mean_test_mae", "representative_test_mae"]
        tie_order = [True, True, True]
    elif mode == "closest_mean":
        value_col = "closest_mean_test_mae" if selection_metric == "mae" else "closest_mean_test_r2"
        ascending = selection_metric == "mae"
        tie_breakers = ["closest_mean_test_r2", "final_test_r2", "representative_test_r2"]
        tie_order = [False, False, False]
    else:
        value_col = "representative_test_mae" if selection_metric == "mae" else "representative_test_r2"
        ascending = selection_metric == "mae"
        tie_breakers = ["representative_test_r2", "final_test_r2", "closest_mean_test_r2"]
        tie_order = [False, False, False]

    if value_col not in property_df.columns:
        return None

    working_df = property_df.dropna(subset=[value_col]).copy()
    if working_df.empty:
        return None

    for column in tie_breakers:
        if column not in working_df.columns:
            working_df[column] = np.nan

    sort_cols = [value_col] + tie_breakers + ["model"]
    ascending_list = [ascending] + tie_order + [True]
    return working_df.sort_values(sort_cols, ascending=ascending_list, na_position="last").iloc[0]


def build_mode_selection_rows(property_df: pd.DataFrame, mean_selection_metric: str) -> pd.DataFrame:
    rows: List[Dict] = []
    for mode in ["final", "closest_mean", "global_mean"]:
        selected = select_mode_row(
            property_df,
            "representative" if mode == "global_mean" else mode,
            "r2" if mode == "final" else mean_selection_metric,
        )
        if selected is None:
            continue
        row = selected.to_dict()
        row["mode"] = mode
        row["selection_metric"] = "R2" if mode == "final" else mean_selection_metric.upper()
        rows.append(row)
    return pd.DataFrame(rows)


def build_combined_predictions(best_rows: pd.DataFrame) -> pd.DataFrame | None:
    combined_df = None

    for _, row in best_rows.iterrows():
        target = str(row["target"])
        mean_pred_path = Path(row["representative_predictions_file"])
        if not mean_pred_path.exists():
            continue

        temp_df = pd.read_csv(mean_pred_path)
        if "ID" not in temp_df.columns:
            temp_df = temp_df.copy()
            temp_df["ID"] = range(len(temp_df))

        cols_to_keep = ["ID"]
        if "Dataset" in temp_df.columns:
            cols_to_keep.append("Dataset")
        actual_col = f"{target}_Actual"
        pred_col = f"{target}_Predicted"
        if actual_col in temp_df.columns:
            cols_to_keep.append(actual_col)
        if pred_col in temp_df.columns:
            cols_to_keep.append(pred_col)
        temp_df = temp_df[cols_to_keep].copy()

        if combined_df is None:
            combined_df = temp_df.copy()
            combined_df["_original_index"] = range(len(combined_df))
            continue

        combined_df = pd.merge(combined_df, temp_df, on="ID", how="outer", suffixes=("", "_extra"))
        if "Dataset_extra" in combined_df.columns:
            combined_df["Dataset"] = combined_df["Dataset"].fillna(combined_df["Dataset_extra"])
            combined_df.drop(columns=["Dataset_extra"], inplace=True)

    if combined_df is not None and "Dataset" in combined_df.columns:
        dataset_order_map = {
            "train": 0,
            "training": 0,
            "validation": 1,
            "val": 1,
            "valid": 1,
            "test": 2,
        }
        combined_df["_sort_key"] = combined_df["Dataset"].astype(str).str.lower().map(dataset_order_map).fillna(3)
        sort_cols = ["_sort_key", "_original_index"] if "_original_index" in combined_df.columns else ["_sort_key"]
        combined_df = combined_df.sort_values(by=sort_cols).drop(
            columns=["_sort_key", "_original_index"], errors="ignore"
        )

    return combined_df


def export_case_views(summary_df: pd.DataFrame, summary_root: Path) -> None:
    cases_root = summary_root / CASES_DIRNAME
    mean_selection_metric = "mae"

    for alloy_name, alloy_df in summary_df.groupby("alloy_family"):
        for dataset_name, dataset_df in alloy_df.groupby("dataset_name"):
            dataset_dir = cases_root / alloy_name / dataset_name
            save_csv(
                rename_export_columns(dataset_df.sort_values(["target", "model"]).reset_index(drop=True)),
                dataset_dir / "dataset_model_summary.csv",
            )
            dataset_comparisons_dir = dataset_dir / "comparisons"
            export_dataset_comparison_summary(dataset_df, dataset_comparisons_dir)
            plot_dataset_property_summary(
                dataset_df,
                dataset_comparisons_dir / "final_test_r2_summary.png",
                "final_test_r2",
            )
            plot_dataset_property_summary(
                dataset_df,
                dataset_comparisons_dir / "final_test_mae_summary.png",
                "final_test_mae",
                "MAE",
            )
            plot_dataset_property_summary(
                dataset_df,
                dataset_comparisons_dir / "final_test_rmse_summary.png",
                "final_test_rmse",
                "RMSE",
            )
            plot_dataset_property_summary(
                dataset_df,
                dataset_comparisons_dir / "closest_mean_r2_summary.png",
                "closest_mean_test_r2",
                "R2",
                "closest_mean_test_r2_std",
            )
            plot_dataset_property_summary(
                dataset_df,
                dataset_comparisons_dir / "closest_mean_mae_summary.png",
                "closest_mean_test_mae",
                "MAE",
                "closest_mean_test_mae_std",
            )
            plot_dataset_property_summary(
                dataset_df,
                dataset_comparisons_dir / "closest_mean_rmse_summary.png",
                "closest_mean_test_rmse",
                "RMSE",
                "closest_mean_test_rmse_std",
            )
            plot_dataset_property_summary(
                dataset_df,
                dataset_comparisons_dir / "global_mean_r2_summary.png",
                "representative_test_r2",
                "R2",
                "representative_test_r2_std",
            )
            plot_dataset_property_summary(
                dataset_df,
                dataset_comparisons_dir / "global_mean_mae_summary.png",
                "representative_test_mae",
                "MAE",
                "representative_test_mae_std",
            )
            plot_dataset_property_summary(
                dataset_df,
                dataset_comparisons_dir / "global_mean_rmse_summary.png",
                "representative_test_rmse",
                "RMSE",
                "representative_test_rmse_std",
            )

            summary_rows: List[Dict] = []

            for target, target_df in dataset_df.groupby("target"):
                mode_rows = build_mode_selection_rows(target_df, mean_selection_metric)
                if mode_rows.empty:
                    continue

                final_row = mode_rows[mode_rows["mode"] == "final"].iloc[0]
                closest_row = mode_rows[mode_rows["mode"] == "closest_mean"].iloc[0]
                global_row = mode_rows[mode_rows["mode"] == "global_mean"].iloc[0]
                safe_target = safe_name(target)
                case_dir = dataset_dir / safe_target
                comparisons_dir = case_dir / "comparisons"
                artifacts_dir = case_dir / "selected_model_artifacts"
                source_root = case_dir / "selected_model_source"
                case_label = f"{alloy_name} / {dataset_name} / {target} / {final_row['feature_mode']}"

                save_csv(
                    rename_export_columns(target_df.sort_values(["model"]).reset_index(drop=True)),
                    case_dir / "case_model_summary.csv",
                )
                save_csv(rename_export_columns(mode_rows), case_dir / "best_model_summary.csv")

                plot_metric_comparison(
                    target_df,
                    comparisons_dir / "final_test_r2_by_model.png",
                    case_label,
                    "final_test_r2",
                    "R2",
                    str(target),
                )
                plot_metric_comparison(
                    target_df,
                    comparisons_dir / "final_test_mae_by_model.png",
                    case_label,
                    "final_test_mae",
                    "MAE",
                    str(target),
                )
                plot_metric_comparison(
                    target_df,
                    comparisons_dir / "closest_mean_r2_by_model.png",
                    case_label,
                    "closest_mean_test_r2",
                    "R2",
                    str(target),
                )
                plot_metric_comparison(
                    target_df,
                    comparisons_dir / "global_mean_r2_by_model.png",
                    case_label,
                    "representative_test_r2",
                    "R2",
                    str(target),
                )
                plot_metric_comparison(
                    target_df,
                    comparisons_dir / "closest_mean_mae_by_model.png",
                    case_label,
                    "closest_mean_test_mae",
                    "MAE",
                    str(target),
                )
                plot_metric_comparison(
                    target_df,
                    comparisons_dir / "global_mean_mae_by_model.png",
                    case_label,
                    "representative_test_mae",
                    "MAE",
                    str(target),
                )
                plot_three_metric_comparison(
                    target_df,
                    comparisons_dir / "three_modes_r2_by_model.png",
                    case_label,
                    "R2",
                    "final_test_r2",
                    "closest_mean_test_r2",
                    "representative_test_r2",
                )
                plot_three_metric_comparison(
                    target_df,
                    comparisons_dir / "three_modes_mae_by_model.png",
                    case_label,
                    "MAE",
                    "final_test_mae",
                    "closest_mean_test_mae",
                    "representative_test_mae",
                )

                copy_if_exists(Path(final_row["final_predictions_file"]), artifacts_dir / "final_predictions.csv")
                copy_if_exists(
                    Path(closest_row["closest_mean_predictions_file"]),
                    artifacts_dir / "closest_mean_predictions.csv",
                )
                copy_if_exists(
                    Path(global_row["representative_predictions_file"]),
                    artifacts_dir / "global_mean_predictions.csv",
                )
                plot_diagonal_chart(
                    Path(final_row["final_predictions_file"]),
                    str(target),
                    str(final_row["model"]),
                    artifacts_dir / "final_diagnostic.png",
                )
                plot_diagonal_chart(
                    Path(closest_row["closest_mean_predictions_file"]),
                    str(target),
                    str(closest_row["model"]),
                    artifacts_dir / "closest_mean_diagnostic.png",
                )
                plot_diagonal_chart(
                    Path(global_row["representative_predictions_file"]),
                    str(target),
                    str(global_row["model"]),
                    artifacts_dir / "global_mean_diagnostic.png",
                )
                copy_tree_if_exists(
                    Path(final_row["model_dir"]),
                    source_root / "final" / safe_name(str(final_row["model"])),
                )
                copy_tree_if_exists(
                    Path(closest_row["model_dir"]),
                    source_root / "closest_mean" / safe_name(str(closest_row["model"])),
                )
                copy_tree_if_exists(
                    Path(global_row["model_dir"]),
                    source_root / "global_mean" / safe_name(str(global_row["model"])),
                )

                summary_rows.append(
                    {
                        "property": target,
                        "final_best_model": final_row.get("model"),
                        "closest_mean_best_model": closest_row.get("model"),
                        "global_mean_best_model": global_row.get("model"),
                        "closest_mean_selection_metric": mean_selection_metric.upper(),
                        "global_mean_selection_metric": mean_selection_metric.upper(),
                        "final_test_r2": final_row.get("final_test_r2"),
                        "closest_mean_test_r2": closest_row.get("closest_mean_test_r2"),
                        "closest_mean_test_mae": closest_row.get("closest_mean_test_mae"),
                        "global_mean_test_r2": global_row.get("representative_test_r2"),
                        "global_mean_test_mae": global_row.get("representative_test_mae"),
                        "case_dir": str(case_dir),
                        "selected_model_dir": str(source_root),
                    }
                )

            if summary_rows:
                save_csv(pd.DataFrame(summary_rows), dataset_dir / "best_models_summary.csv")

            combined_df = build_combined_predictions(build_best_rows(dataset_df))
            if combined_df is not None:
                save_csv(combined_df, dataset_dir / "combined_predictions.csv")


def create_global_exports(summary_df: pd.DataFrame, summary_root: Path) -> None:
    summary_tables_dir = summary_root / SUMMARY_TABLES_DIRNAME
    export_df = rename_export_columns(summary_df)
    save_csv(export_df, summary_tables_dir / "all_extrapolation_model_summary.csv")

    best_rows = (
        summary_df.assign(
            final_test_r2=summary_df["final_test_r2"].fillna(-np.inf),
            closest_mean_test_r2=summary_df["closest_mean_test_r2"].fillna(-np.inf),
            representative_test_r2=summary_df["representative_test_r2"].fillna(-np.inf),
        )
        .sort_values(
            ["case_id", "target", "final_test_r2", "closest_mean_test_r2", "representative_test_r2"],
            ascending=[True, True, False, False, False],
            na_position="last",
        )
        .groupby(["case_id", "target"], group_keys=False)
        .head(1)
        .reset_index(drop=True)
    )
    best_export_df = rename_export_columns(best_rows)
    save_csv(best_export_df, summary_tables_dir / "all_best_extrapolation_models.csv")

    pivot_df = best_rows.pivot_table(
        index=["alloy_family", "dataset_name", "feature_mode"],
        columns="target",
        values="final_test_r2",
        aggfunc="first",
    )
    if not pivot_df.empty:
        pivot_df = pivot_df.reset_index().rename(columns={"feature_mode": "mode"})
        save_csv(pivot_df, summary_tables_dir / "best_model_final_test_r2_pivot.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch summarize extrapolation experiment results.")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=r"output\extrapolation_results",
        help="Base directory containing extrapolation result folders.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    summary_root = base_dir / "all_traditional_extrapolation_summary"
    if summary_root.exists():
        shutil.rmtree(summary_root)
    summary_root.mkdir(parents=True, exist_ok=True)

    model_comparison_dirs = find_model_comparison_dirs(base_dir)
    if not model_comparison_dirs:
        print(f"No model_comparison directories found under: {base_dir}")
        return

    print("=" * 80)
    print("Starting extrapolation results batch summary")
    print("=" * 80)
    print(f"Search root: {base_dir}")
    print(f"Found {len(model_comparison_dirs)} model_comparison directories\n")

    all_rows: List[Dict] = []
    for index, model_comparison_dir in enumerate(model_comparison_dirs, start=1):
        print(f"[{index}/{len(model_comparison_dirs)}] Processing: {model_comparison_dir}")
        case_rows = process_case(model_comparison_dir, base_dir)
        if case_rows:
            all_rows.extend(case_rows)

    if not all_rows:
        print("No extrapolation metrics were collected.")
        return

    summary_df = pd.DataFrame(all_rows)
    create_global_exports(summary_df, summary_root)
    export_case_views(summary_df, summary_root)

    print("\n" + "=" * 80)
    print("Completed extrapolation summary")
    print("=" * 80)
    print(f"Cases processed: {summary_df['case_id'].nunique()}")
    print(f"Targets summarized: {summary_df[['case_id', 'target']].drop_duplicates().shape[0]}")
    print(f"Models summarized: {len(summary_df)}")
    print(f"Summary directory: {summary_root}")


if __name__ == "__main__":
    main()
