"""
Batch summarize traditional model comparison results.

This script is the traditional-model counterpart to the extrapolation summary
scripts. It scans `output/new_results_withuncertainty/*/*/tradition/model_comparison`,
extracts final / closest-to-mean / global-mean metrics, and exports a unified
summary layout with tables, comparison plots, and copied best-model artifacts.
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

from _raw_prediction_stats import (
    collect_optuna_test_metrics,
    infer_best_trial_id_from_reference_csv,
    read_prediction_csv,
    resolve_prediction_columns,
    summarize_optuna_predictions,
)


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
COLORS = ["#f4a698", "#9bbfe0", "#c8d5b9", "#d4a5d6", "#f6d186"]
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


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_json(payload: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def to_jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if pd.isna(value):
        return None
    return value


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


def extract_metrics(metrics: Dict, prefix: str, output_prefix: str) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    pattern = re.compile(rf"^{prefix}_(train|val|test)_(.+)_(r2|rmse|mae)$")
    for key, value in metrics.items():
        match = pattern.match(key)
        if not match:
            continue
        split, property_name, metric_name = match.groups()
        results.setdefault(property_name, {})[f"{output_prefix}_{split}_{metric_name}"] = value
    return results


def extract_final_test_r2_std(cv_summary_path: Path) -> Dict[str, float]:
    payload = load_json(cv_summary_path)
    std_map: Dict[str, float] = {}
    for property_name, metrics in payload.items():
        if not isinstance(metrics, dict):
            continue
        std_value = metrics.get("r2_std")
        if isinstance(std_value, (int, float)):
            std_map[str(property_name)] = float(std_value)
    return std_map


def extract_test_metric_mean_std_from_cv(csv_path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    if not csv_path.exists():
        return {}

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}

    stats_map: Dict[str, Dict[str, Dict[str, float]]] = {}
    pattern = re.compile(r"^(.*)_(r2|rmse|mae)$")
    for col in df.columns:
        match = pattern.match(col)
        if not match:
            continue
        property_name, metric_name = match.groups()
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue
        stats_map.setdefault(property_name, {})[metric_name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
        }

    return stats_map


def extract_fold_info_test_metric_std(fold_info_path: Path, metric_name: str) -> float | None:
    payload = load_json(fold_info_path)
    values_key = f"all_fold_test_{metric_name}"
    all_fold_values = payload.get(values_key)
    if not isinstance(all_fold_values, list):
        return None

    values = pd.to_numeric(pd.Series(all_fold_values), errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.std())


def plot_dataset_property_summary(
    dataset_df: pd.DataFrame,
    value_col: str,
    output_path: Path,
    metric_label: str = "R2",
    err_col: str | None = None,
) -> None:
    plot_df = dataset_df[["model", "property", value_col]].dropna().copy()
    if plot_df.empty:
        return

    if err_col and err_col in dataset_df.columns:
        plot_df = plot_df.merge(
            dataset_df[["model", "property", err_col]],
            on=["model", "property"],
            how="left",
        )
    else:
        err_col = None

    properties = sorted(plot_df["property"].unique())
    x = np.arange(len(MODEL_ORDER))
    width = 0.8 / max(len(properties), 1)

    fig, ax = plt.subplots(figsize=(10, 8))
    for idx, property_name in enumerate(properties):
        prop_df = plot_df[plot_df["property"] == property_name].copy()
        prop_df["model"] = pd.Categorical(prop_df["model"], categories=MODEL_ORDER, ordered=True)
        prop_df = prop_df.sort_values("model").set_index("model").reindex(MODEL_ORDER).reset_index()
        offset = (idx - (len(properties) - 1) / 2) * width
        yerr_values = None
        if err_col:
            yerr_values = prop_df[err_col].fillna(0).tolist()
        ax.bar(
            x + offset,
            prop_df[value_col].tolist(),
            yerr=yerr_values,
            width=width,
            capsize=3,
            color=COLORS[idx % len(COLORS)],
            edgecolor="black",
            linewidth=1.1,
            error_kw={"elinewidth": 1.2, "ecolor": "black"},
            label=property_name,
        )

    ax.set_xlabel("Predictive models", fontweight="bold")
    ax.set_ylabel(R2_LABEL if metric_label == "R2" else metric_label, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER)
    if metric_label == "R2":
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
        "property",
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
        "global_mean_test_r2",
        "global_mean_test_r2_std",
        "global_mean_test_mae",
        "global_mean_test_mae_std",
        "global_mean_test_rmse",
        "global_mean_test_rmse_std",
    ]
    available_cols = [col for col in comparison_cols if col in dataset_df.columns]
    save_csv(
        dataset_df[available_cols].sort_values(["property", "model"]).reset_index(drop=True),
        comparisons_dir / "dataset_metric_summary.csv",
    )


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
    else:
        valid_df = df[[actual_col, pred_col]].dropna()
        if not valid_df.empty:
            plt.scatter(
                valid_df[actual_col],
                valid_df[pred_col],
                alpha=0.6,
                s=80,
                edgecolors="black",
                linewidth=0.8,
                c="#4CAF50",
                label="All Data",
            )
            min_val = min(valid_df[actual_col].min(), valid_df[pred_col].min())
            max_val = max(valid_df[actual_col].max(), valid_df[pred_col].max())
            plotted_any = True

            y_true = valid_df[actual_col].values
            y_pred = valid_df[pred_col].values
            metrics_text.append(
                f"All Data:\n  R² = {r2_score(y_true, y_pred):.4f}, "
                f"MAE = {mean_absolute_error(y_true, y_pred):.2f}, "
                f"RMSE = {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}"
            )

    if not plotted_any or min_val == float("inf"):
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

    property_label = property_name.replace("_", " ")
    plt.xlabel(f"Experimental {property_label}", fontsize=16, fontweight="bold")
    plt.ylabel(f"Predicted {property_label}", fontsize=16, fontweight="bold")
    plt.title(f"{model_name} - {property_label}", fontsize=18, fontweight="bold")
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


def sync_mode_outputs_into_source_copy(
    source_model_dir: Path,
    mode: str,
    row: pd.Series,
    property_name: str,
) -> None:
    if mode == "closest_mean":
        legacy_dir = source_model_dir / "closest_to_mean_evaluation"
        predictions_dst = legacy_dir / "predictions" / "closest_to_mean_evaluation_all_predictions.csv"
        plot_dst = legacy_dir / "plots" / "closest_to_mean_evaluation_all_sets_comparison.png"
        info_dst = legacy_dir / "closest_to_mean_fold_info.json"
        metric_prefix = "closest_mean"
    elif mode == "global_mean":
        legacy_dir = source_model_dir / "closest_to_global_mean_trial_fold"
        predictions_dst = legacy_dir / "predictions" / "closest_to_global_mean_trial_fold_all_predictions.csv"
        plot_dst = legacy_dir / "plots" / "closest_to_global_mean_trial_fold_all_sets_comparison.png"
        info_dst = legacy_dir / "closest_to_global_mean_fold_info.json"
        metric_prefix = "global_mean"
    else:
        return

    predictions_file = row.get(f"{metric_prefix}_predictions_file")
    if not predictions_file:
        return

    predictions_src = Path(str(predictions_file))
    copy_if_exists(predictions_src, predictions_dst)
    plot_diagonal_chart(predictions_src, property_name, str(row.get("model")), plot_dst)

    payload = {
        "property": to_jsonable(property_name),
        "selection_metric": to_jsonable(row.get("selection_metric")),
        "selected_trial_id": to_jsonable(row.get(f"{metric_prefix}_trial_id")),
        "selected_fold": to_jsonable(row.get(f"{metric_prefix}_fold")),
        "target_test_r2_mean": to_jsonable(row.get(f"{metric_prefix}_test_r2")),
        "target_test_r2_std": to_jsonable(row.get(f"{metric_prefix}_test_r2_std")),
        "target_test_mae_mean": to_jsonable(row.get(f"{metric_prefix}_test_mae")),
        "target_test_mae_std": to_jsonable(row.get(f"{metric_prefix}_test_mae_std")),
        "target_test_rmse_mean": to_jsonable(row.get(f"{metric_prefix}_test_rmse")),
        "target_test_rmse_std": to_jsonable(row.get(f"{metric_prefix}_test_rmse_std")),
        "selected_test_r2": to_jsonable(row.get(f"{metric_prefix}_closest_r2")),
    }
    save_json(payload, info_dst)


def plot_case_by_model(case_df: pd.DataFrame, value_col: str, output_path: Path, legend_label: str) -> None:
    plot_df = case_df.copy()
    plot_df["model"] = pd.Categorical(plot_df["model"], categories=MODEL_ORDER, ordered=True)
    plot_df = plot_df.sort_values("model").dropna(subset=[value_col])
    if plot_df.empty:
        return

    x = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.bar(
        x,
        plot_df[value_col],
        width=0.55,
        color="#ffb703",
        edgecolor="black",
        linewidth=1.2,
        label=legend_label,
    )
    ax.set_xlabel("Predictive models", fontweight="bold")
    ax.set_ylabel(R2_LABEL, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["model"])
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis="both", which="major", direction="in", length=6, width=1.2, top=True, right=True)
    ax.tick_params(axis="both", which="minor", direction="in", length=3, width=1.0, top=True, right=True)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.legend(frameon=True, edgecolor="black")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_three_modes(case_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = case_df.copy()
    plot_df["model"] = pd.Categorical(plot_df["model"], categories=MODEL_ORDER, ordered=True)
    plot_df = plot_df.sort_values("model").dropna(subset=["final_test_r2"])
    if plot_df.empty:
        return

    x = np.arange(len(plot_df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.bar(
        x - width,
        plot_df["final_test_r2"],
        width=width,
        color="#8ecae6",
        edgecolor="black",
        linewidth=1.2,
        label=f"Final {R2_LABEL}",
    )
    ax.bar(
        x,
        plot_df["closest_mean_test_r2"],
        width=width,
        color="#ffb703",
        edgecolor="black",
        linewidth=1.2,
        label=f"Best-trial mean {R2_LABEL}",
    )
    ax.bar(
        x + width,
        plot_df["global_mean_test_r2"],
        width=width,
        color="#90be6d",
        edgecolor="black",
        linewidth=1.2,
        label=f"All-trial mean {R2_LABEL}",
    )
    ax.set_xlabel("Predictive models", fontweight="bold")
    ax.set_ylabel(R2_LABEL, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["model"])
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis="both", which="major", direction="in", length=6, width=1.2, top=True, right=True)
    ax.tick_params(axis="both", which="minor", direction="in", length=3, width=1.0, top=True, right=True)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.legend(frameon=True, edgecolor="black")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def find_model_comparison_dirs(base_dir: Path) -> List[Path]:
    return sorted(
        [
            path
            for path in base_dir.rglob("model_comparison")
            if path.is_dir() and path.parent.name == "tradition"
        ]
    )


def collect_rows(model_comparison_dir: Path, base_dir: Path) -> List[Dict]:
    try:
        alloy_family, dataset_name, feature_mode = model_comparison_dir.relative_to(base_dir).parts[:3]
    except Exception:
        print(f"[WARN] Unexpected traditional path layout: {model_comparison_dir}")
        return []

    rows: List[Dict] = []
    for model_dir in sorted(model_comparison_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = MODEL_MAP.get(model_dir.name)
        if model_name is None:
            continue

        final_metrics = extract_metrics(
            load_json(model_dir / "final_model_evaluation_metrics.json"),
            "final_model_evaluation",
            "final",
        )
        optuna_trials_dir = model_dir / "predictions" / "optuna_trials"
        metrics_df = collect_optuna_test_metrics(optuna_trials_dir)
        preferred_best_trial_id = infer_best_trial_id_from_reference_csv(
            metrics_df,
            model_dir / "best_trial_test_mean_cv_results.csv",
        )
        raw_summaries = summarize_optuna_predictions(
            optuna_trials_dir,
            selection_metric="r2",
            global_prefix="global_mean",
            mean_prefix="closest_mean",
            preferred_best_trial_id=preferred_best_trial_id,
        )

        properties = sorted(set(final_metrics) | set(raw_summaries))
        for property_name in properties:
            row = {
                "case_id": f"{alloy_family}__{dataset_name}__{property_name}__tradition",
                "case_path": str(model_comparison_dir),
                "alloy_family": alloy_family,
                "dataset_name": dataset_name,
                "property": property_name,
                "feature_mode": feature_mode,
                "model": model_name,
                "model_dir": str(model_dir),
                "final_predictions_file": str(model_dir / "predictions" / "all_predictions.csv"),
                "closest_mean_predictions_file": "",
                "global_mean_predictions_file": "",
                "final_plot_file": str(model_dir / "plots" / "final_model_evaluation_all_sets_comparison.png"),
                "closest_mean_plot_file": "",
                "global_mean_plot_file": "",
            }
            row.update(final_metrics.get(property_name, {}))
            row.update(raw_summaries.get(property_name, {}))
            rows.append(row)

    return rows


def build_best_rows(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.assign(
            final_test_r2=df["final_test_r2"].fillna(-np.inf),
            closest_mean_test_r2=df["closest_mean_test_r2"].fillna(-np.inf),
            global_mean_test_r2=df["global_mean_test_r2"].fillna(-np.inf),
        )
        .sort_values(
            ["case_id", "final_test_r2", "closest_mean_test_r2", "global_mean_test_r2"],
            ascending=[True, False, False, False],
            na_position="last",
        )
        .groupby("case_id", group_keys=False)
        .head(1)
        .reset_index(drop=True)
    )


def select_mode_row(property_df: pd.DataFrame, mode: str, selection_metric: str) -> pd.Series | None:
    if property_df.empty:
        return None

    if mode == "final":
        value_col = "final_test_r2"
        ascending = False
        tie_breakers = ["final_test_mae", "closest_mean_test_mae", "global_mean_test_mae"]
        tie_order = [True, True, True]
    elif mode == "closest_mean":
        value_col = "closest_mean_test_mae" if selection_metric == "mae" else "closest_mean_test_r2"
        ascending = selection_metric == "mae"
        tie_breakers = ["closest_mean_test_r2", "final_test_r2", "global_mean_test_r2"]
        tie_order = [False, False, False]
    else:
        value_col = "global_mean_test_mae" if selection_metric == "mae" else "global_mean_test_r2"
        ascending = selection_metric == "mae"
        tie_breakers = ["global_mean_test_r2", "final_test_r2", "closest_mean_test_r2"]
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
            mode,
            "r2" if mode == "final" else mean_selection_metric,
        )
        if selected is None:
            continue
        row = selected.to_dict()
        row["mode"] = mode
        row["selection_metric"] = "R2" if mode == "final" else mean_selection_metric.upper()
        rows.append(row)
    return pd.DataFrame(rows)


def export_tables(summary_df: pd.DataFrame, summary_root: Path) -> pd.DataFrame:
    tables_dir = summary_root / SUMMARY_TABLES_DIRNAME
    save_csv(summary_df, tables_dir / "all_extrapolation_model_summary.csv")
    best_df = build_best_rows(summary_df)
    save_csv(best_df, tables_dir / "all_best_extrapolation_models.csv")

    pivot_df = best_df.pivot_table(
        index=["alloy_family", "dataset_name", "feature_mode"],
        columns="property",
        values="final_test_r2",
        aggfunc="first",
    )
    if not pivot_df.empty:
        pivot_df = pivot_df.reset_index().rename(columns={"feature_mode": "mode"})
        save_csv(pivot_df, tables_dir / "best_model_final_test_r2_pivot.csv")

    return best_df


def export_cases(summary_df: pd.DataFrame, best_df: pd.DataFrame, summary_root: Path) -> None:
    cases_root = summary_root / CASES_DIRNAME
    mean_selection_metric = "r2"

    for (alloy_family, dataset_name), dataset_df in summary_df.groupby(["alloy_family", "dataset_name"]):
        dataset_dir = cases_root / alloy_family / dataset_name
        save_csv(
            dataset_df.sort_values(["property", "model"]).reset_index(drop=True),
            dataset_dir / "dataset_model_summary.csv",
        )

        dataset_best = best_df[
            (best_df["alloy_family"] == alloy_family) & (best_df["dataset_name"] == dataset_name)
        ].reset_index(drop=True)
        save_csv(dataset_best, dataset_dir / "best_models_summary.csv")
        comparisons_dir = dataset_dir / "comparisons"
        export_dataset_comparison_summary(dataset_df, comparisons_dir)

        plot_dataset_property_summary(
            dataset_df,
            "final_test_r2",
            comparisons_dir / "final_test_r2_summary.png",
        )
        plot_dataset_property_summary(
            dataset_df,
            "final_test_mae",
            comparisons_dir / "final_test_mae_summary.png",
            metric_label="MAE",
        )
        plot_dataset_property_summary(
            dataset_df,
            "final_test_rmse",
            comparisons_dir / "final_test_rmse_summary.png",
            metric_label="RMSE",
        )
        plot_dataset_property_summary(
            dataset_df,
            "closest_mean_test_r2",
            comparisons_dir / "closest_mean_r2_summary.png",
            metric_label="R2",
            err_col="closest_mean_test_r2_std",
        )
        plot_dataset_property_summary(
            dataset_df,
            "closest_mean_test_mae",
            comparisons_dir / "closest_mean_mae_summary.png",
            metric_label="MAE",
            err_col="closest_mean_test_mae_std",
        )
        plot_dataset_property_summary(
            dataset_df,
            "closest_mean_test_rmse",
            comparisons_dir / "closest_mean_rmse_summary.png",
            metric_label="RMSE",
            err_col="closest_mean_test_rmse_std",
        )
        plot_dataset_property_summary(
            dataset_df,
            "global_mean_test_r2",
            comparisons_dir / "global_mean_r2_summary.png",
            metric_label="R2",
            err_col="global_mean_test_r2_std",
        )
        plot_dataset_property_summary(
            dataset_df,
            "global_mean_test_mae",
            comparisons_dir / "global_mean_mae_summary.png",
            metric_label="MAE",
            err_col="global_mean_test_mae_std",
        )
        plot_dataset_property_summary(
            dataset_df,
            "global_mean_test_rmse",
            comparisons_dir / "global_mean_rmse_summary.png",
            metric_label="RMSE",
            err_col="global_mean_test_rmse_std",
        )

        for property_name, property_df in dataset_df.groupby("property"):
            case_dir = dataset_dir / safe_name(property_name)
            save_csv(
                property_df.sort_values(["model"]).reset_index(drop=True),
                case_dir / "case_model_summary.csv",
            )
            mode_rows = build_mode_selection_rows(property_df, mean_selection_metric)
            if not mode_rows.empty:
                save_csv(mode_rows, case_dir / "best_model_summary.csv")

            plot_case_by_model(
                property_df,
                "final_test_r2",
                case_dir / "comparisons" / "final_test_r2_by_model.png",
                property_name,
            )
            plot_case_by_model(
                property_df,
                "closest_mean_test_r2",
                case_dir / "comparisons" / "closest_mean_r2_by_model.png",
                property_name,
            )
            plot_case_by_model(
                property_df,
                "global_mean_test_r2",
                case_dir / "comparisons" / "global_mean_r2_by_model.png",
                property_name,
            )
            plot_three_modes(property_df, case_dir / "comparisons" / "three_modes_r2_by_model.png")

            if mode_rows.empty:
                continue

            final_row = mode_rows[mode_rows["mode"] == "final"].iloc[0]
            closest_row = mode_rows[mode_rows["mode"] == "closest_mean"].iloc[0]
            global_row = mode_rows[mode_rows["mode"] == "global_mean"].iloc[0]
            source_root = case_dir / "selected_model_source"
            artifacts_dir = case_dir / "selected_model_artifacts"

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
            sync_mode_outputs_into_source_copy(
                source_root / "closest_mean" / safe_name(str(closest_row["model"])),
                "closest_mean",
                closest_row,
                property_name,
            )
            sync_mode_outputs_into_source_copy(
                source_root / "global_mean" / safe_name(str(global_row["model"])),
                "global_mean",
                global_row,
                property_name,
            )
            copy_if_exists(Path(final_row["final_predictions_file"]), artifacts_dir / "final_predictions.csv")
            copy_if_exists(
                Path(closest_row["closest_mean_predictions_file"]),
                artifacts_dir / "closest_mean_predictions.csv",
            )
            copy_if_exists(
                Path(global_row["global_mean_predictions_file"]),
                artifacts_dir / "global_mean_predictions.csv",
            )
            plot_diagonal_chart(
                Path(final_row["final_predictions_file"]),
                str(property_name),
                str(final_row["model"]),
                artifacts_dir / "final_diagnostic.png",
            )
            plot_diagonal_chart(
                Path(closest_row["closest_mean_predictions_file"]),
                str(property_name),
                str(closest_row["model"]),
                artifacts_dir / "closest_mean_diagnostic.png",
            )
            plot_diagonal_chart(
                Path(global_row["global_mean_predictions_file"]),
                str(property_name),
                str(global_row["model"]),
                artifacts_dir / "global_mean_diagnostic.png",
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch summarize traditional model comparison results.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("output/new_results_withuncertainty"),
        help="Base directory containing traditional model comparison results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional summary output directory. Defaults to <base-dir>/all_alloys_traditional_models_summary.",
    )
    args = parser.parse_args()

    base_dir = args.base_dir
    summary_root = args.output_dir or (base_dir / "all_alloys_traditional_models_summary")
    if summary_root.exists():
        shutil.rmtree(summary_root)
    summary_root.mkdir(parents=True, exist_ok=True)

    model_comparison_dirs = find_model_comparison_dirs(base_dir)
    if not model_comparison_dirs:
        print(f"No tradition/model_comparison directories found under: {base_dir}")
        return

    rows: List[Dict] = []
    for path in model_comparison_dirs:
        rows.extend(collect_rows(path, base_dir))

    if not rows:
        print("No traditional model metrics were collected.")
        return

    summary_df = pd.DataFrame(rows)
    best_df = export_tables(summary_df, summary_root)
    export_cases(summary_df, best_df, summary_root)
    print(f"Traditional summary exported to: {summary_root}")


if __name__ == "__main__":
    main()
