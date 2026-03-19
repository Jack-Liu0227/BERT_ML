"""
Batch summary script for BERT extrapolation experiment results.

The script scans output/extrapolation_results for SciBERT, MatSciBERT, and
SteelBERT result folders, collects final / closest-to-mean / global-mean
metrics, and exports a unified summary layout for easier comparison.
"""

from __future__ import annotations

import argparse
import json
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
    "scibert": "SciBERT",
    "matscibert": "MatSciBERT",
    "steelbert": "SteelBERT",
}

MODEL_ORDER = ["MatSciBERT", "SciBERT", "SteelBERT"]
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
    return df.copy().rename(
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


def flatten_eval_summary(summary: Dict, prefix: str) -> Dict[str, float]:
    result: Dict[str, float] = {}
    split_map = {
        "train_set_metrics": "train",
        "validation_set_metrics": "val",
        "test_set_metrics": "test",
    }

    for payload_key, split_name in split_map.items():
        split_payload = summary.get(payload_key, {})
        if not isinstance(split_payload, dict):
            continue
        for metric_key, value in split_payload.items():
            if not isinstance(value, (int, float)):
                continue
            suffix = None
            for candidate in ("r2", "rmse", "mae"):
                if metric_key.endswith(f"_{candidate}"):
                    suffix = candidate
                    break
            if suffix is None:
                continue
            result[f"{prefix}_{split_name}_{suffix}"] = value

    return result


def extract_fold_info(info: Dict, prefix: str) -> Dict[str, float]:
    result: Dict[str, float] = {}
    if not info:
        return result

    if "closest_fold" in info:
        result[f"{prefix}_closest_fold"] = info["closest_fold"]
    if "fold_val_r2" in info:
        result[f"{prefix}_closest_fold_val_r2"] = info["fold_val_r2"]
    if "mean_val_r2" in info:
        result[f"{prefix}_mean_val_r2"] = info["mean_val_r2"]

    return result


def resolve_existing_path(*candidates: Path) -> str:
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


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
    if case_df.empty or value_col not in case_df.columns:
        return

    ordered_df = case_df.copy()
    ordered_df["model"] = pd.Categorical(ordered_df["model"], categories=MODEL_ORDER, ordered=True)
    ordered_df = ordered_df.sort_values("model").dropna(subset=[value_col])
    if ordered_df.empty:
        return

    x = np.arange(len(ordered_df))
    fig, ax = plt.subplots(figsize=(9.5, 6.8))
    ax.bar(
        x,
        ordered_df[value_col],
        width=0.55,
        color="#ffb703",
        edgecolor="black",
        linewidth=1.2,
        label=legend_label,
    )

    ax.set_xlabel("BERT models", fontweight="bold")
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
    required = [final_col, closest_mean_col, global_mean_col]
    if case_df.empty or any(col not in case_df.columns for col in required):
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

    ax.set_xlabel("BERT models", fontweight="bold")
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

    fig, ax = plt.subplots(figsize=(9.6, 7.6))
    for idx, prop in enumerate(properties):
        prop_df = plot_df[plot_df["target"] == prop].copy()
        prop_df["model"] = pd.Categorical(prop_df["model"], categories=MODEL_ORDER, ordered=True)
        prop_df = prop_df.sort_values("model").set_index("model").reindex(MODEL_ORDER).reset_index()
        offset = (idx - (len(properties) - 1) / 2) * width
        ax.bar(
            model_positions + offset,
            prop_df[value_col].tolist(),
            yerr=prop_df[err_col].fillna(0).tolist() if err_col else None,
            width=width,
            capsize=3,
            color=colors[idx % len(colors)],
            edgecolor="black",
            linewidth=1.1,
            error_kw={"elinewidth": 1.2, "ecolor": "black"},
            label=prop,
        )

    ax.set_xlabel("BERT models", fontweight="bold")
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
    if dataset_col == "set":
        dataset_aliases = {
            "Train": {"train", "training"},
            "Validation": {"validation", "valid", "val"},
            "Test": {"test", "testing", "extrapolationtest", "extrapolation_test", "extrapolation test"},
        }

    plt.figure(figsize=(8, 8))
    min_val = float("inf")
    max_val = float("-inf")
    metrics_text: List[str] = []
    plotted_any = False

    if dataset_col is not None:
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


def find_bert_model_dirs(base_dir: Path) -> List[Path]:
    model_names = set(MODEL_MAP)
    return sorted(
        [
            path
            for path in base_dir.rglob("*")
            if path.is_dir()
            and path.name in model_names
            and "all_extrapolation_summary" not in path.parts
            and "all_bert_extrapolation_summary" not in path.parts
        ]
    )


def process_bert_case(model_dir: Path, base_dir: Path) -> Dict | None:
    try:
        relative_parts = model_dir.relative_to(base_dir).parts
        alloy_family, dataset_name, target = relative_parts[:3]
    except Exception:
        print(f"[WARN] Unexpected BERT path layout: {model_dir}")
        return None

    model_name = MODEL_MAP.get(model_dir.name)
    if model_name is None:
        return None

    final_metrics = flatten_eval_summary(
        load_json(model_dir / "best_model_best_model_evaluation_evaluation_summary.json"),
        "final",
    )
    raw_summaries = summarize_optuna_predictions(
        model_dir / "predictions" / "optuna_trials",
        selection_metric="mae",
        global_prefix="representative",
        mean_prefix="closest_mean",
    )

    row = {
        "case_id": "__".join(relative_parts[:3]),
        "case_path": str(model_dir),
        "alloy_family": alloy_family,
        "dataset_name": dataset_name,
        "target": target,
        "feature_mode": "bert",
        "model": model_name,
        "model_dir": str(model_dir),
        "final_predictions_file": resolve_existing_path(model_dir / "predictions" / "best_model_all_predictions.csv"),
        "representative_predictions_file": "",
        "closest_mean_predictions_file": "",
        "final_plot_file": resolve_existing_path(model_dir / "plots" / "best_model_all_sets_comparison.png"),
        "representative_plot_file": "",
        "closest_mean_plot_file": "",
    }
    row.update(final_metrics)
    row.update(raw_summaries.get(target, {}))

    if not any(key.startswith("final_") for key in row):
        print(f"[WARN] Missing final metrics: {model_dir}")
        return None

    return row


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
        combined_df = combined_df.sort_values(by=["_sort_key", "_original_index"]).drop(
            columns=["_sort_key", "_original_index"],
            errors="ignore",
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
                case_label = f"{alloy_name} / {dataset_name} / {target}"

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
                plot_three_metric_comparison(
                    target_df,
                    comparisons_dir / "three_modes_r2_by_model.png",
                    case_label,
                    "R2",
                    "final_test_r2",
                    "closest_mean_test_r2",
                    "representative_test_r2",
                )

                copy_if_exists(Path(final_row["final_predictions_file"]), artifacts_dir / "final_predictions.csv")
                copy_if_exists(Path(closest_row["closest_mean_predictions_file"]), artifacts_dir / "closest_mean_predictions.csv")
                copy_if_exists(Path(global_row["representative_predictions_file"]), artifacts_dir / "global_mean_predictions.csv")
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
                copy_tree_if_exists(Path(final_row["model_dir"]), source_root / "final" / safe_name(str(final_row["model"])))
                copy_tree_if_exists(Path(closest_row["model_dir"]), source_root / "closest_mean" / safe_name(str(closest_row["model"])))
                copy_tree_if_exists(Path(global_row["model_dir"]), source_root / "global_mean" / safe_name(str(global_row["model"])))

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
    save_csv(rename_export_columns(summary_df), summary_tables_dir / "all_bert_extrapolation_model_summary.csv")

    best_rows = (
        summary_df.assign(
            final_test_r2=summary_df["final_test_r2"].fillna(-np.inf),
            closest_mean_test_r2=summary_df["closest_mean_test_r2"].fillna(-np.inf),
            representative_test_r2=summary_df["representative_test_r2"].fillna(-np.inf),
        )
        .sort_values(
            ["case_id", "final_test_r2", "closest_mean_test_r2", "representative_test_r2"],
            ascending=[True, False, False, False],
            na_position="last",
        )
        .groupby("case_id", group_keys=False)
        .head(1)
        .reset_index(drop=True)
    )
    save_csv(rename_export_columns(best_rows), summary_tables_dir / "all_best_bert_extrapolation_models.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch summarize BERT extrapolation experiment results.")
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

    summary_root = base_dir / "all_bert_extrapolation_summary"
    if summary_root.exists():
        shutil.rmtree(summary_root)
    summary_root.mkdir(parents=True, exist_ok=True)

    bert_model_dirs = find_bert_model_dirs(base_dir)
    if not bert_model_dirs:
        print(f"No BERT model directories found under: {base_dir}")
        return

    print("=" * 80)
    print("Starting BERT extrapolation results batch summary")
    print("=" * 80)
    print(f"Search root: {base_dir}")
    print(f"Found {len(bert_model_dirs)} BERT model directories\n")

    rows: List[Dict] = []
    for index, model_dir in enumerate(bert_model_dirs, start=1):
        print(f"[{index}/{len(bert_model_dirs)}] Processing: {model_dir}")
        row = process_bert_case(model_dir, base_dir)
        if row is not None:
            rows.append(row)

    if not rows:
        print("No BERT extrapolation metrics were collected.")
        return

    summary_df = pd.DataFrame(rows)
    create_global_exports(summary_df, summary_root)
    export_case_views(summary_df, summary_root)

    print("\n" + "=" * 80)
    print("Completed BERT extrapolation summary")
    print("=" * 80)
    print(f"Cases processed: {summary_df['case_id'].nunique()}")
    print(f"Models summarized: {len(summary_df)}")
    print(f"Summary directory: {summary_root}")


if __name__ == "__main__":
    main()
