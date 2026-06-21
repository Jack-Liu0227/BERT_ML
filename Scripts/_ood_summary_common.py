from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, MultipleLocator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from _raw_prediction_stats import read_prediction_csv, resolve_prediction_columns


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


OOD_METHOD_MAP = {
    "random_cv_baseline": "RandomCV",
    "target_extrapolation": "Extrapolation",
    "loco": "LOCO",
    "loco_k5": "LOCO",
    "sparse_x_cluster": "SparseXcluster",
    "sparse_x_cluster_k5": "SparseXcluster",
    "sparse_x_single": "SparseXsingle",
    "sparse_x_single_k5": "SparseXsingle",
    "sparse_y_cluster": "SparseYcluster",
    "sparse_y_cluster_k5": "SparseYcluster",
    "sparse_y_single": "SparseYsingle",
    "sparse_y_single_k5": "SparseYsingle",
    "hybrid_extrapolation_loco": "HybridHigh20+LOCO",
    "hybrid_extrapolation_loco_k5": "HybridHigh20+LOCO",
    "hybrid_extrapolation_random_cv": "HybridHigh20+RandCV",
    "hybrid_extrapolation_random_cv_k5": "HybridHigh20+RandCV",
    "hybrid_extrapolation_sparse_x_cluster": "HybridHigh20+SparseXcluster",
    "hybrid_extrapolation_sparse_x_cluster_k5": "HybridHigh20+SparseXcluster",
    "hybrid_extrapolation_sparse_x_single": "HybridHigh20+SparseXsingle",
    "hybrid_extrapolation_sparse_x_single_k5": "HybridHigh20+SparseXsingle",
    "hybrid_extrapolation_sparse_y_cluster": "HybridHigh20+SparseYcluster",
    "hybrid_extrapolation_sparse_y_cluster_k5": "HybridHigh20+SparseYcluster",
    "hybrid_extrapolation_sparse_y_single": "HybridHigh20+SparseYsingle",
    "hybrid_extrapolation_sparse_y_single_k5": "HybridHigh20+SparseYsingle",
}
OOD_METHOD_ORDER = [
    "RandomCV",
    "Extrapolation",
    "LOCO",
    "SparseXcluster",
    "SparseXsingle",
    "SparseYcluster",
    "SparseYsingle",
    "HybridHigh20+RandCV",
    "HybridHigh20+LOCO",
    "HybridHigh20+SparseXcluster",
    "HybridHigh20+SparseXsingle",
    "HybridHigh20+SparseYcluster",
    "HybridHigh20+SparseYsingle",
]
HYBRID_TEST_SET_COLUMN_PREFIXES = {
    "test_extrapolation_high20": "summary_test_extrapolation_high20",
    "test_inner_ood": "summary_test_inner_ood",
}
HYBRID_TEST_SET_METRICS = ("r2", "mae", "rmse", "n_samples")
SUMMARY_TABLES_DIRNAME = "00_summary_tables"
CASES_DIRNAME = "01_alloy_cases"
OOD_SUMMARY_DIRNAME = "02_ood_method_summary"
R2_LABEL = "R²"
FAMILY_RANK_KEYS = ["alloy_family", "dataset_name", "property", "ood_method", "model_family"]
TEST_DATASET_ALIASES = {
    "test",
    "testing",
    "oodtest",
    "ood",
    "oodtesting",
    "extrapolationtest",
    "extrapolation_test",
    "extrapolation test",
    "testextrapolationhigh20",
    "test_extrapolation_high20",
    "testinnerood",
    "test_inner_ood",
}
CASE_LEVEL_PREDICTION_PATTERNS = [
    "predictions/all_predictions.csv",
    "predictions/best_model_all_predictions.csv",
    "closest_to_global_mean_trial_fold/all_predictions.csv",
    "closest_to_global_mean_trial_fold/predictions/*.csv",
    "closest_to_mean_evaluation/all_predictions.csv",
    "closest_to_mean_evaluation/predictions/*.csv",
    "closest_to_mean_predictions/all_predictions.csv",
    "closest_to_mean_predictions/predictions/*.csv",
    "closest_to_global_mean_predictions/all_predictions.csv",
    "closest_to_global_mean_predictions/predictions/*.csv",
]


def safe_name(text: str) -> str:
    return (
        str(text)
        .replace("(", "")
        .replace(")", "")
        .replace("%", "pct")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def normalize_alloy_family_name(text: str) -> str:
    normalized = str(text).strip()
    if normalized == "HEA_half":
        return "HEA"
    if normalized in {"MatbenchSteels", "Matbench Steel", "matbench_steels", "matbench_steel"}:
        return "Matbench Steel"
    return normalized


def save_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def _read_json_if_exists(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def flatten_hybrid_test_set_metrics(payload: dict | None, *, include_std: bool = True) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {}
    test_set_metrics = payload.get("test_set_metrics")
    if not isinstance(test_set_metrics, dict):
        return {}

    flattened: dict[str, object] = {}
    for test_set_name, column_prefix in HYBRID_TEST_SET_COLUMN_PREFIXES.items():
        metrics = test_set_metrics.get(test_set_name)
        if not isinstance(metrics, dict):
            continue
        for metric in HYBRID_TEST_SET_METRICS:
            value = metrics.get(metric)
            flattened[f"{column_prefix}_{metric}"] = value
            if include_std and metric != "n_samples":
                flattened[f"{column_prefix}_{metric}_std"] = 0.0
    return flattened


def load_hybrid_test_set_metrics(metrics_path: Path, *, payload_key: str | None = None) -> dict[str, object]:
    payload = _read_json_if_exists(metrics_path)
    if payload_key is not None:
        nested = payload.get(payload_key, {})
        payload = nested if isinstance(nested, dict) else {}
    return flatten_hybrid_test_set_metrics(payload)


def aggregate_hybrid_test_set_metrics(payloads: Iterable[dict]) -> dict[str, object]:
    values: dict[tuple[str, str], list[float]] = {}
    sample_counts: dict[str, float] = {}
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        test_set_metrics = payload.get("test_set_metrics")
        if not isinstance(test_set_metrics, dict):
            continue
        for test_set_name, column_prefix in HYBRID_TEST_SET_COLUMN_PREFIXES.items():
            metrics = test_set_metrics.get(test_set_name)
            if not isinstance(metrics, dict):
                continue
            for metric in ("r2", "mae", "rmse"):
                value = pd.to_numeric(pd.Series([metrics.get(metric)]), errors="coerce").iloc[0]
                if pd.notna(value):
                    values.setdefault((column_prefix, metric), []).append(float(value))
            count = pd.to_numeric(pd.Series([metrics.get("n_samples")]), errors="coerce").iloc[0]
            if pd.notna(count):
                sample_counts[column_prefix] = sample_counts.get(column_prefix, 0.0) + float(count)

    flattened: dict[str, object] = {}
    for (column_prefix, metric), metric_values in values.items():
        if not metric_values:
            continue
        series = pd.Series(metric_values, dtype="float64")
        flattened[f"{column_prefix}_{metric}"] = float(series.mean())
        std_value = series.std()
        flattened[f"{column_prefix}_{metric}_std"] = 0.0 if pd.isna(std_value) else float(std_value)
    for column_prefix, count in sample_counts.items():
        flattened[f"{column_prefix}_n_samples"] = int(count)
    return flattened


def normalize_ood_method(raw_method: str) -> str:
    normalized = str(raw_method).strip().lower()
    if normalized not in OOD_METHOD_MAP:
        raise ValueError(f"Unsupported OOD method: {raw_method}")
    return OOD_METHOD_MAP[normalized]


def ensure_canonical_summary_schema(summary_df: pd.DataFrame) -> pd.DataFrame:
    df = summary_df.copy()

    def first_available(columns: list[str], default: object = pd.NA) -> pd.Series:
        for column in columns:
            if column in df.columns:
                return df[column]
        return pd.Series([default] * len(df), index=df.index, dtype="object")

    canonical_map: dict[str, list[str]] = {
        "summary_test_r2": ["summary_test_r2", "test_r2", "final_test_r2"],
        "summary_test_r2_std": ["summary_test_r2_std"],
        "summary_test_mae": ["summary_test_mae", "test_mae", "final_test_mae"],
        "summary_test_mae_std": ["summary_test_mae_std"],
        "summary_test_rmse": ["summary_test_rmse", "test_rmse", "final_test_rmse"],
        "summary_test_rmse_std": ["summary_test_rmse_std"],
        "trial_count": ["trial_count"],
        "fold_count": ["fold_count"],
        "representative_selection_mode": ["representative_selection_mode", "selection_mode"],
        "representative_trial_id": ["representative_trial_id", "selected_trial_id"],
        "representative_fold": ["representative_fold", "selected_fold"],
        "representative_test_r2": ["representative_test_r2", "test_r2", "final_test_r2"],
        "representative_test_mae": ["representative_test_mae", "test_mae", "final_test_mae"],
        "representative_test_rmse": ["representative_test_rmse", "test_rmse", "final_test_rmse"],
        "representative_predictions_file": [
            "representative_predictions_file",
            "predictions_file",
            "final_predictions_file",
        ],
        "representative_plot_file": ["representative_plot_file", "plot_file", "final_plot_file"],
        "family_best_metric": ["family_best_metric"],
        "family_rank_score": ["family_rank_score", "summary_test_r2", "test_r2", "final_test_r2"],
        "rank_within_family": ["rank_within_family"],
        "is_family_best": ["is_family_best"],
    }
    for column_prefix in HYBRID_TEST_SET_COLUMN_PREFIXES.values():
        for metric in HYBRID_TEST_SET_METRICS:
            column = f"{column_prefix}_{metric}"
            canonical_map[column] = [column]
            if metric != "n_samples":
                std_column = f"{column_prefix}_{metric}_std"
                canonical_map[std_column] = [std_column]

    for column, candidates in canonical_map.items():
        df[column] = first_available(candidates)

    if "model_dir" not in df.columns:
        df["model_dir"] = pd.NA
    if "source_dir" not in df.columns:
        df["source_dir"] = pd.NA

    numeric_defaults = {
        "summary_test_r2_std": 0.0,
        "summary_test_mae_std": 0.0,
        "summary_test_rmse_std": 0.0,
        "trial_count": 1,
        "fold_count": 1,
        "family_rank_score": np.nan,
        "rank_within_family": np.nan,
        "representative_fold": np.nan,
        "representative_test_r2": np.nan,
        "representative_test_mae": np.nan,
        "representative_test_rmse": np.nan,
    }
    for column_prefix in HYBRID_TEST_SET_COLUMN_PREFIXES.values():
        for metric in HYBRID_TEST_SET_METRICS:
            numeric_defaults[f"{column_prefix}_{metric}"] = np.nan
            if metric != "n_samples":
                numeric_defaults[f"{column_prefix}_{metric}_std"] = 0.0
    for column, default in numeric_defaults.items():
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if default is not np.nan:
            df[column] = df[column].fillna(default)

    text_defaults = {
        "representative_selection_mode": "single_run",
        "family_best_metric": "summary_test_r2",
    }
    for column, default in text_defaults.items():
        df[column] = df[column].astype("object").where(df[column].notna(), default)

    df["is_family_best"] = df["is_family_best"].where(df["is_family_best"].notna(), False).astype(bool)
    return df


def annotate_family_ranks(summary_df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_canonical_summary_schema(summary_df)
    if df.empty:
        return df

    sort_df = df.copy()
    sort_df["_sort_summary_test_r2"] = pd.to_numeric(sort_df["summary_test_r2"], errors="coerce").fillna(-np.inf)
    sort_df["_sort_summary_test_r2_std"] = pd.to_numeric(sort_df["summary_test_r2_std"], errors="coerce").fillna(np.inf)
    sort_df["_sort_summary_test_mae"] = pd.to_numeric(sort_df["summary_test_mae"], errors="coerce").fillna(np.inf)

    sort_df = sort_df.sort_values(
        FAMILY_RANK_KEYS + ["_sort_summary_test_r2", "_sort_summary_test_r2_std", "_sort_summary_test_mae", "model"],
        ascending=[True, True, True, True, True, False, True, True, True],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)

    sort_df["rank_within_family"] = sort_df.groupby(FAMILY_RANK_KEYS, sort=False).cumcount() + 1
    sort_df["is_family_best"] = sort_df["rank_within_family"].eq(1)
    sort_df["family_best_metric"] = "summary_test_r2"
    sort_df["family_rank_score"] = pd.to_numeric(sort_df["summary_test_r2"], errors="coerce")
    return sort_df.drop(columns=["_sort_summary_test_r2", "_sort_summary_test_r2_std", "_sort_summary_test_mae"])


def style_axes(ax, metric_label: str) -> None:
    if metric_label == "R2":
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.tick_params(axis="both", which="major", direction="in", length=6, width=1.2, top=True, right=True)
    ax.tick_params(axis="both", which="minor", direction="in", length=3, width=1.0, top=True, right=True)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


def plot_case_metric(
    case_df: pd.DataFrame,
    output_path: Path,
    metric_col: str,
    metric_label: str,
    err_col: str | None = None,
) -> None:
    if case_df.empty or metric_col not in case_df.columns:
        return

    has_display_label = "display_label" in case_df.columns and case_df["display_label"].notna().any()
    include_family_in_label = (
        not has_display_label
        and "model_family" in case_df.columns
        and case_df["model_family"].astype(str).nunique() > 1
    )
    base_columns = ["model", "ood_method", metric_col]
    if "model_family" in case_df.columns:
        base_columns.insert(0, "model_family")
    if "display_label" in case_df.columns:
        base_columns.append("display_label")
    if err_col and err_col in case_df.columns:
        base_columns.append(err_col)

    working_df = case_df[base_columns].copy()
    if has_display_label:
        working_df["model_key"] = working_df["display_label"].astype(str).str.strip()
    elif include_family_in_label:
        working_df["model_key"] = (
            working_df["model_family"].astype(str).str.strip() + ": " + working_df["model"].astype(str).str.strip()
        )
    else:
        working_df["model_key"] = working_df["model"].astype(str).str.strip()

    working_df[metric_col] = pd.to_numeric(working_df[metric_col], errors="coerce")
    working_df = working_df.dropna(subset=[metric_col])
    if working_df.empty:
        return

    model_order = working_df["model_key"].drop_duplicates().tolist()
    method_order = [method for method in OOD_METHOD_ORDER if method in set(working_df["ood_method"].astype(str))]
    if not method_order:
        return

    pivot_df = working_df.pivot_table(
        index="ood_method",
        columns="model_key",
        values=metric_col,
        aggfunc="first",
        observed=False,
    ).reindex(index=method_order, columns=model_order)

    pivot_err_df = None
    if err_col and err_col in working_df.columns:
        working_df[err_col] = pd.to_numeric(working_df[err_col], errors="coerce").fillna(0.0)
        pivot_err_df = working_df.pivot_table(
            index="ood_method",
            columns="model_key",
            values=err_col,
            aggfunc="first",
            observed=False,
        ).reindex(index=method_order, columns=model_order).fillna(0.0)

    x = np.arange(len(method_order))
    width = 0.8 / max(len(model_order), 1)
    cmap = plt.get_cmap("tab20")
    colors = [cmap(idx % cmap.N) for idx in range(len(model_order))]

    fig, ax = plt.subplots(figsize=(11, 7))
    for idx, model_name in enumerate(model_order):
        offset = (idx - (len(model_order) - 1) / 2) * width
        yerr = None
        if pivot_err_df is not None:
            yerr = pivot_err_df[model_name].to_numpy()
        ax.bar(
            x + offset,
            pivot_df[model_name].tolist(),
            width=width,
            color=colors[idx % len(colors)],
            edgecolor="black",
            linewidth=1.1,
            label=model_name,
            yerr=yerr,
            capsize=3 if yerr is not None else 0,
        )

    ax.set_xlabel("OOD methods", fontweight="bold")
    ax.set_ylabel(R2_LABEL if metric_label == "R2" else metric_label, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(method_order)
    if metric_label == "R2":
        valid_values = pd.to_numeric(working_df[metric_col], errors="coerce").dropna()
        if not valid_values.empty and (valid_values >= 0).all():
            ax.set_ylim(0.0, 1.0)
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    style_axes(ax, metric_label)
    legend_ncol = min(max(1, int(np.ceil(len(model_order) / 3))), max(len(model_order), 1))
    ax.legend(frameon=True, edgecolor="black", ncol=legend_ncol, loc="upper center", bbox_to_anchor=(0.5, 1.03))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_diagonal_chart(file_path: Path, property_name: str, output_path: Path) -> None:
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
        "Train": {"color": "#5B9BD5", "marker": "o", "label": "Train", "alpha": 0.65, "s": 55},
        "Validation": {"color": "#70AD47", "marker": "s", "label": "Validation", "alpha": 0.70, "s": 60},
        "Test": {"color": "#FF4D4F", "marker": "s", "label": "Test", "alpha": 0.80, "s": 60},
    }
    dataset_aliases = {
        "Train": {"train", "training"},
        "Validation": {"validation", "valid", "val"},
        "Test": TEST_DATASET_ALIASES,
    }

    plt.figure(figsize=(8, 8))
    has_dataset_col = dataset_col is not None
    min_val = float("inf")
    max_val = float("-inf")
    metrics_text: list[str] = []
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
                edgecolors="white",
                linewidth=0.6,
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
                f"{config['label']}: {R2_LABEL}={r2_score(y_true, y_pred):.3f}, "
                f"RMSE={np.sqrt(mean_squared_error(y_true, y_pred)):.3f}, "
                f"MAE={mean_absolute_error(y_true, y_pred):.3f}"
            )
    else:
        valid_df = df[[actual_col, pred_col]].dropna()
        if not valid_df.empty:
            plt.scatter(
                valid_df[actual_col],
                valid_df[pred_col],
                alpha=0.65,
                s=60,
                edgecolors="white",
                linewidth=0.6,
                c="#5B9BD5",
                label="All Data",
            )
            min_val = min(valid_df[actual_col].min(), valid_df[pred_col].min())
            max_val = max(valid_df[actual_col].max(), valid_df[pred_col].max())
            plotted_any = True

            y_true = valid_df[actual_col].values
            y_pred = valid_df[pred_col].values
            metrics_text.append(
                f"All Data: {R2_LABEL}={r2_score(y_true, y_pred):.3f}, "
                f"RMSE={np.sqrt(mean_squared_error(y_true, y_pred)):.3f}, "
                f"MAE={mean_absolute_error(y_true, y_pred):.3f}"
            )

    if not plotted_any:
        plt.close()
        return

    padding = max((max_val - min_val) * 0.05, 1.0)
    min_plot = min_val - padding
    max_plot = max_val + padding
    plt.plot([min_plot, max_plot], [min_plot, max_plot], "k--", linewidth=2, label="Ideal")
    plt.xlim(min_plot, max_plot)
    plt.ylim(min_plot, max_plot)

    plt.title(str(property_name), fontsize=22, fontweight="bold", pad=12)
    plt.xlabel("True Values", fontsize=20, fontweight="bold")
    plt.ylabel("Predicted Values", fontsize=20, fontweight="bold")
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(loc="upper left", frameon=True, edgecolor="black")

    if metrics_text:
        plt.text(
            0.98,
            0.02,
            "\n".join(metrics_text),
            transform=plt.gca().transAxes,
            fontsize=9,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.92, edgecolor="black", linewidth=1.0),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def build_case_wide_summary(case_df: pd.DataFrame) -> pd.DataFrame:
    canonical_df = annotate_family_ranks(case_df)
    fields = [
        "summary_test_r2",
        "summary_test_r2_std",
        "summary_test_mae",
        "summary_test_mae_std",
        "summary_test_rmse",
        "summary_test_rmse_std",
        "representative_trial_id",
        "representative_fold",
        "representative_test_r2",
        "representative_test_mae",
        "representative_test_rmse",
        "summary_test_extrapolation_high20_r2",
        "summary_test_extrapolation_high20_r2_std",
        "summary_test_extrapolation_high20_mae",
        "summary_test_extrapolation_high20_mae_std",
        "summary_test_extrapolation_high20_rmse",
        "summary_test_extrapolation_high20_rmse_std",
        "summary_test_extrapolation_high20_n_samples",
        "summary_test_inner_ood_r2",
        "summary_test_inner_ood_r2_std",
        "summary_test_inner_ood_mae",
        "summary_test_inner_ood_mae_std",
        "summary_test_inner_ood_rmse",
        "summary_test_inner_ood_rmse_std",
        "summary_test_inner_ood_n_samples",
        "rank_within_family",
        "is_family_best",
    ]
    rows: list[dict] = []
    for model_name in sorted(canonical_df["model"].astype(str).unique()):
        model_df = canonical_df[canonical_df["model"].astype(str) == model_name].copy()
        row: dict[str, object] = {"model": model_name}
        for method_label in OOD_METHOD_ORDER:
            method_df = model_df[model_df["ood_method"].astype(str) == method_label]
            if method_df.empty:
                for field_name in fields:
                    row[f"{method_label}_{field_name}"] = pd.NA
                continue
            selected = method_df.iloc[0]
            for field_name in fields:
                row[f"{method_label}_{field_name}"] = selected.get(field_name, pd.NA)
        rows.append(row)
    return pd.DataFrame(rows)


def _normalize_dataset_values(dataset_series: pd.Series) -> pd.Series:
    return (
        dataset_series.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "", regex=False)
        .str.replace("_", "", regex=False)
    )


def extract_prediction_frames(
    file_path: Path,
    property_name: str,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, str | None, str | None, str | None]:
    try:
        df = read_prediction_csv(file_path)
    except Exception as exc:
        print(f"[WARN] Failed to read predictions file {file_path}: {exc}")
        return None, None, None, None, None

    if df is None:
        return None, None, None, None, None

    dataset_col, actual_col, pred_col = resolve_prediction_columns(df, property_name)
    if actual_col is None or pred_col is None:
        return df, None, dataset_col, actual_col, pred_col

    if dataset_col is None:
        return df, df[[actual_col, pred_col]].dropna().copy(), dataset_col, actual_col, pred_col

    dataset_series = _normalize_dataset_values(df[dataset_col])
    test_df = df.loc[dataset_series.isin(TEST_DATASET_ALIASES)].copy()
    if test_df.empty:
        return df, None, dataset_col, actual_col, pred_col
    return df, test_df, dataset_col, actual_col, pred_col


def _compute_subset_metrics(df: pd.DataFrame, actual_col: str, pred_col: str) -> dict[str, float] | None:
    valid_df = df[[actual_col, pred_col]].dropna()
    if valid_df.empty:
        return None
    y_true = valid_df[actual_col].to_numpy()
    y_pred = valid_df[pred_col].to_numpy()
    return {
        "test_r2": float(r2_score(y_true, y_pred)),
        "test_mae": float(mean_absolute_error(y_true, y_pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "test_row_count": int(len(valid_df)),
    }


def _resolve_id_column(df: pd.DataFrame) -> str | None:
    for candidate in ["ID", "id"]:
        if candidate in df.columns:
            return candidate
    return None


def _normalize_id_series(series: pd.Series) -> pd.Series:
    raw_series = series.copy()
    numeric = pd.to_numeric(raw_series, errors="coerce")
    normalized = raw_series.astype(str).str.strip()
    numeric_mask = numeric.notna()
    if numeric_mask.any():
        normalized.loc[numeric_mask] = numeric.loc[numeric_mask].map(
            lambda value: str(int(value)) if float(value).is_integer() else str(value)
        )
    return normalized


def _find_expected_test_split_file(model_dir: Path) -> Path | None:
    candidate_dirs: list[Path] = []
    search_paths = [model_dir]
    search_paths.extend(model_dir.parents[:4])
    seen: set[Path] = set()

    for base_path in search_paths:
        split_dir = base_path / "split_data"
        if split_dir.is_dir():
            resolved = split_dir.resolve()
            if resolved not in seen:
                seen.add(resolved)
                candidate_dirs.append(split_dir)

    for split_dir in candidate_dirs:
        test_files = sorted(split_dir.glob("test*.csv"))
        if len(test_files) == 1:
            return test_files[0]
        if test_files:
            prioritized = sorted(
                test_files,
                key=lambda path: (
                    0 if path.name.lower() == "test.csv" else 1,
                    path.name.lower(),
                ),
            )
            return prioritized[0]
    return None


def _load_expected_test_split(model_dir: Path) -> tuple[Path | None, pd.DataFrame | None, set[str]]:
    split_file = _find_expected_test_split_file(model_dir)
    if split_file is None:
        return None, None, set()

    try:
        split_df = pd.read_csv(split_file, low_memory=False)
    except Exception as exc:
        print(f"[WARN] Failed to read split-data file {split_file}: {exc}")
        return split_file, None, set()

    id_col = _resolve_id_column(split_df)
    if id_col is None:
        return split_file, split_df, set()

    expected_ids = set(_normalize_id_series(split_df[id_col]).dropna().tolist())
    return split_file, split_df, expected_ids


def _score_id_match(test_df: pd.DataFrame, expected_ids: set[str]) -> dict[str, object]:
    id_col = _resolve_id_column(test_df)
    if id_col is None:
        return {
            "id_col": None,
            "candidate_ids": set(),
            "id_overlap_count": 0,
            "id_only_in_candidate": 0,
            "id_only_in_split": len(expected_ids),
            "id_exact_match": False,
        }

    candidate_ids = set(_normalize_id_series(test_df[id_col]).dropna().tolist())
    overlap = len(candidate_ids & expected_ids)
    only_candidate = len(candidate_ids - expected_ids)
    only_split = len(expected_ids - candidate_ids)
    return {
        "id_col": id_col,
        "candidate_ids": candidate_ids,
        "id_overlap_count": overlap,
        "id_only_in_candidate": only_candidate,
        "id_only_in_split": only_split,
        "id_exact_match": overlap == len(expected_ids) and overlap == len(candidate_ids),
    }


def _align_test_df_to_split_ids(test_df: pd.DataFrame, expected_split_df: pd.DataFrame | None) -> pd.DataFrame:
    if expected_split_df is None:
        return test_df.copy()

    split_id_col = _resolve_id_column(expected_split_df)
    pred_id_col = _resolve_id_column(test_df)
    if split_id_col is None or pred_id_col is None:
        return test_df.copy()

    split_ids = _normalize_id_series(expected_split_df[split_id_col]).rename("__normalized_id__")
    split_order_df = pd.DataFrame({"__normalized_id__": split_ids})
    aligned_test_df = test_df.copy()
    aligned_test_df["__normalized_id__"] = _normalize_id_series(aligned_test_df[pred_id_col])
    merged_df = split_order_df.merge(aligned_test_df, on="__normalized_id__", how="left", sort=False)
    merged_df = merged_df.drop(columns=["__normalized_id__"])
    return merged_df


SUBSET_LABELING_AUDIT_COLUMNS = [
    "alloy_family",
    "dataset_name",
    "property",
    "ood_method",
    "model_family",
    "model",
    "model_dir",
    "source_dir",
    "subset_name",
    "summary_prefix",
    "status",
    "source_mode",
    "source_prediction_file",
    "split_file",
    "prediction_id_column",
    "split_id_column",
    "actual_column",
    "predicted_column",
    "expected_id_count",
    "matched_prediction_count",
    "missing_id_count",
    "metric_row_count",
]


def _nonempty_normalized_ids(series: pd.Series) -> list[str]:
    values: list[str] = []
    for value in _normalize_id_series(series).tolist():
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "<na>"}:
            continue
        values.append(text)
    return values


def _find_hybrid_test_set_files(model_dir: Path) -> dict[str, Path]:
    candidate_dirs: list[Path] = []
    seen: set[Path] = set()
    search_paths = [model_dir]
    search_paths.extend(model_dir.parents[:6])

    for base_path in search_paths:
        test_sets_dir = base_path / "split_data" / "test_sets"
        if not test_sets_dir.is_dir():
            continue
        resolved = test_sets_dir.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        candidate_dirs.append(test_sets_dir)

    subset_files: dict[str, Path] = {}
    for subset_name in HYBRID_TEST_SET_COLUMN_PREFIXES:
        for test_sets_dir in candidate_dirs:
            candidate = test_sets_dir / f"{subset_name}.csv"
            if candidate.exists():
                subset_files[subset_name] = candidate
                break
    return subset_files


def _read_csv_or_none(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        print(f"[WARN] Failed to read CSV {path}: {exc}")
        return None


def _hybrid_subset_prediction_candidates(model_dir: Path, subset_name: str) -> list[Path]:
    patterns = [
        f"predictions/test_sets/{subset_name}_predictions.csv",
        f"predictions/test_sets/{subset_name}.csv",
        "predictions/all_predictions.csv",
        "predictions/best_model_all_predictions.csv",
        "predictions/test_predictions.csv",
        "predictions/best_model_test_predictions.csv",
    ]
    candidates: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for candidate in model_dir.glob(pattern):
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(candidate)
    return candidates


def _subset_source_mode(model_dir: Path, candidate_file: Path, subset_name: str) -> str:
    try:
        relative_text = "/".join(candidate_file.relative_to(model_dir).parts).lower()
    except ValueError:
        relative_text = str(candidate_file).lower()
    if f"test_sets/{subset_name}" in relative_text:
        return "subset_prediction_file"
    if relative_text.endswith("predictions/all_predictions.csv"):
        return "all_predictions_id_match"
    if relative_text.endswith("predictions/best_model_all_predictions.csv"):
        return "best_model_all_predictions_id_match"
    if "test_predictions" in relative_text:
        return "test_predictions_id_match"
    return "prediction_id_match"


def _align_prediction_rows_to_split_ids(
    prediction_df: pd.DataFrame,
    split_df: pd.DataFrame,
) -> tuple[pd.DataFrame | None, dict[str, object]]:
    split_id_col = _resolve_id_column(split_df)
    pred_id_col = _resolve_id_column(prediction_df)

    if split_id_col is None or pred_id_col is None:
        return None, {
            "prediction_id_column": pred_id_col,
            "split_id_column": split_id_col,
            "expected_id_count": np.nan,
            "matched_prediction_count": 0,
            "missing_id_count": np.nan,
        }

    expected_ids = _nonempty_normalized_ids(split_df[split_id_col])
    expected_id_set = set(expected_ids)
    if not expected_id_set:
        return None, {
            "prediction_id_column": pred_id_col,
            "split_id_column": split_id_col,
            "expected_id_count": 0,
            "matched_prediction_count": 0,
            "missing_id_count": 0,
        }

    candidate_df = prediction_df.copy()
    candidate_df["__normalized_id__"] = _normalize_id_series(candidate_df[pred_id_col])
    candidate_df = candidate_df[candidate_df["__normalized_id__"].isin(expected_id_set)].copy()
    matched_ids = set(_nonempty_normalized_ids(candidate_df["__normalized_id__"])) if not candidate_df.empty else set()
    candidate_df = candidate_df.drop_duplicates(subset=["__normalized_id__"], keep="first")

    split_order_df = pd.DataFrame({"__normalized_id__": list(dict.fromkeys(expected_ids))})
    aligned_df = split_order_df.merge(candidate_df, on="__normalized_id__", how="left", sort=False)
    aligned_df = aligned_df.drop(columns=["__normalized_id__"])

    return aligned_df, {
        "prediction_id_column": pred_id_col,
        "split_id_column": split_id_col,
        "expected_id_count": int(len(expected_id_set)),
        "matched_prediction_count": int(len(matched_ids)),
        "missing_id_count": int(len(expected_id_set - matched_ids)),
    }


def _is_metric_missing(value: object) -> bool:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return pd.isna(numeric)


def has_complete_hybrid_test_set_metrics(values: dict[str, object] | pd.Series) -> bool:
    for column_prefix in HYBRID_TEST_SET_COLUMN_PREFIXES.values():
        for metric in HYBRID_TEST_SET_METRICS:
            if _is_metric_missing(values.get(f"{column_prefix}_{metric}", pd.NA)):
                return False
    return True


def load_hybrid_prediction_subset_metrics(
    model_dir: Path,
    property_name: str,
    *,
    context: dict[str, object] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Recover hybrid subset metrics from existing prediction rows and split IDs."""

    model_dir = Path(model_dir)
    context = dict(context or {})
    metrics_by_subset: dict[str, object] = {}
    audit_rows: list[dict[str, object]] = []
    subset_files = _find_hybrid_test_set_files(model_dir)

    for subset_name, summary_prefix in HYBRID_TEST_SET_COLUMN_PREFIXES.items():
        audit_base = {
            **context,
            "subset_name": subset_name,
            "summary_prefix": summary_prefix,
            "model_dir": str(model_dir),
            "source_prediction_file": pd.NA,
            "split_file": pd.NA,
            "prediction_id_column": pd.NA,
            "split_id_column": pd.NA,
            "actual_column": pd.NA,
            "predicted_column": pd.NA,
            "expected_id_count": np.nan,
            "matched_prediction_count": 0,
            "missing_id_count": np.nan,
            "metric_row_count": 0,
        }

        split_file = subset_files.get(subset_name)
        if split_file is None:
            audit_rows.append({**audit_base, "status": "missing_split_file", "source_mode": pd.NA})
            continue

        split_df = _read_csv_or_none(split_file)
        if split_df is None:
            audit_rows.append(
                {
                    **audit_base,
                    "status": "split_read_failed",
                    "source_mode": pd.NA,
                    "split_file": str(split_file),
                }
            )
            continue

        chosen_audit: dict[str, object] | None = None
        chosen_metrics: dict[str, float] | None = None
        for prediction_file in _hybrid_subset_prediction_candidates(model_dir, subset_name):
            prediction_df = read_prediction_csv(prediction_file)
            if prediction_df is None:
                continue

            _, actual_col, pred_col = resolve_prediction_columns(prediction_df, property_name)
            aligned_df, id_audit = _align_prediction_rows_to_split_ids(prediction_df, split_df)
            candidate_audit = {
                **audit_base,
                **id_audit,
                "status": "no_matching_prediction_rows",
                "source_mode": _subset_source_mode(model_dir, prediction_file, subset_name),
                "source_prediction_file": str(prediction_file),
                "split_file": str(split_file),
                "actual_column": actual_col or pd.NA,
                "predicted_column": pred_col or pd.NA,
            }

            if actual_col is None or pred_col is None or aligned_df is None:
                chosen_audit = candidate_audit
                continue

            candidate_metrics = _compute_subset_metrics(aligned_df, actual_col, pred_col)
            if candidate_metrics is None:
                chosen_audit = candidate_audit
                continue

            candidate_audit["metric_row_count"] = candidate_metrics["test_row_count"]
            candidate_audit["status"] = (
                "complete"
                if int(candidate_audit.get("missing_id_count", 0) or 0) == 0
                and int(candidate_audit.get("matched_prediction_count", 0) or 0)
                == int(candidate_audit.get("expected_id_count", 0) or 0)
                else "partial_id_match"
            )
            chosen_audit = candidate_audit
            chosen_metrics = candidate_metrics
            break

        if chosen_metrics is not None:
            metrics_by_subset[f"{summary_prefix}_r2"] = chosen_metrics["test_r2"]
            metrics_by_subset[f"{summary_prefix}_r2_std"] = 0.0
            metrics_by_subset[f"{summary_prefix}_mae"] = chosen_metrics["test_mae"]
            metrics_by_subset[f"{summary_prefix}_mae_std"] = 0.0
            metrics_by_subset[f"{summary_prefix}_rmse"] = chosen_metrics["test_rmse"]
            metrics_by_subset[f"{summary_prefix}_rmse_std"] = 0.0
            metrics_by_subset[f"{summary_prefix}_n_samples"] = int(chosen_metrics["test_row_count"])
            audit_rows.append(chosen_audit or {**audit_base, "status": "complete"})
        else:
            audit_rows.append(
                chosen_audit
                or {
                    **audit_base,
                    "status": "missing_prediction_file",
                    "source_mode": pd.NA,
                    "split_file": str(split_file),
                }
            )

    return metrics_by_subset, audit_rows


def fill_missing_hybrid_test_set_metrics(
    row: dict[str, object],
    fallback_metrics: dict[str, object],
) -> None:
    for key, value in fallback_metrics.items():
        if key not in row or _is_metric_missing(row.get(key, pd.NA)):
            row[key] = value


def save_subset_labeling_audit(audit_rows: list[dict[str, object]], summary_root: Path) -> None:
    audit_df = pd.DataFrame(audit_rows)
    if audit_df.empty:
        audit_df = pd.DataFrame(columns=SUBSET_LABELING_AUDIT_COLUMNS)
    else:
        for column in SUBSET_LABELING_AUDIT_COLUMNS:
            if column not in audit_df.columns:
                audit_df[column] = pd.NA
        sort_columns = [
            "alloy_family",
            "dataset_name",
            "property",
            "ood_method",
            "model_family",
            "model",
            "subset_name",
        ]
        audit_df = (
            audit_df.reindex(columns=SUBSET_LABELING_AUDIT_COLUMNS)
            .sort_values(sort_columns, na_position="last")
            .reset_index(drop=True)
        )
    save_csv(audit_df, summary_root / SUMMARY_TABLES_DIRNAME / "subset_labeling_audit.csv")


def _artifact_source_label(model_dir: Path, candidate_file: Path) -> str:
    try:
        relative_parts = candidate_file.relative_to(model_dir).parts
    except ValueError:
        return "external_predictions_file"

    relative_text = "/".join(relative_parts).lower()
    if "closest_to_global_mean_trial_fold" in relative_text:
        return "closest_to_global_mean_trial_fold"
    if "closest_to_global_mean_predictions" in relative_text:
        return "closest_to_global_mean_predictions"
    if "closest_to_mean_evaluation" in relative_text:
        return "closest_to_mean_evaluation"
    if "closest_to_mean_predictions" in relative_text:
        return "closest_to_mean_predictions"
    if relative_text.endswith("predictions/best_model_all_predictions.csv"):
        return "best_model_predictions"
    if relative_text.endswith("predictions/all_predictions.csv"):
        return "model_root_predictions"
    return "case_level_predictions"


def _collect_case_level_prediction_candidates(model_dir: Path) -> list[Path]:
    if not model_dir.exists():
        return []

    candidates: list[Path] = []
    seen: set[Path] = set()
    for pattern in CASE_LEVEL_PREDICTION_PATTERNS:
        for candidate in model_dir.glob(pattern):
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(candidate)
    return candidates


def resolve_case_level_artifact(row: pd.Series) -> dict[str, object] | None:
    property_name = str(row.get("property", "") or "")
    loco_fold_details = _load_loco_outer_fold_best_details(row)
    if str(row.get("ood_method", "") or "") in {"LOCO", "RandomCV", "HybridHigh20+LOCO", "HybridHigh20+RandCV"} and loco_fold_details:
        outer_df = pd.DataFrame(loco_fold_details)
        if not outer_df.empty and {"outer_test_r2", "outer_test_mae", "outer_test_rmse"}.issubset(outer_df.columns):
            outer_df["outer_test_r2"] = pd.to_numeric(outer_df["outer_test_r2"], errors="coerce")
            outer_df["outer_test_mae"] = pd.to_numeric(outer_df["outer_test_mae"], errors="coerce")
            outer_df["outer_test_rmse"] = pd.to_numeric(outer_df["outer_test_rmse"], errors="coerce")
            outer_df["outer_test_row_count"] = pd.to_numeric(outer_df.get("outer_test_row_count"), errors="coerce")
            valid_outer_df = outer_df.dropna(subset=["outer_test_r2", "outer_test_mae", "outer_test_rmse"]).copy()
            if not valid_outer_df.empty:
                summary_test_mae = pd.to_numeric(pd.Series([row.get("summary_test_mae")]), errors="coerce").iloc[0]
                valid_outer_df["_distance_to_summary_test_mae"] = (
                    valid_outer_df["outer_test_mae"] - summary_test_mae
                ).abs()
                valid_outer_df["_sort_outer_fold"] = pd.to_numeric(
                    valid_outer_df.get("outer_fold_index"),
                    errors="coerce",
                ).fillna(np.inf)
                valid_outer_df = valid_outer_df.sort_values(
                    ["_distance_to_summary_test_mae", "outer_test_mae", "outer_test_r2", "_sort_outer_fold"],
                    ascending=[True, True, False, True],
                    na_position="last",
                    kind="stable",
                ).reset_index(drop=True)
                representative_outer_row = valid_outer_df.iloc[0]
                source_mode = (
                    "loco_outer_fold_aggregate"
                    if str(row.get("ood_method", "") or "") == "LOCO"
                    else "randomcv_outer_fold_aggregate"
                )
                return {
                    "predictions_file": representative_outer_row.get("outer_predictions_file", row.get("representative_predictions_file", pd.NA)),
                    "source_mode": source_mode,
                    "expected_split_file": pd.NA,
                    "expected_split_count": np.nan,
                    "test_r2": float(valid_outer_df["outer_test_r2"].mean()),
                    "test_mae": float(valid_outer_df["outer_test_mae"].mean()),
                    "test_rmse": float(valid_outer_df["outer_test_rmse"].mean()),
                    "test_row_count": int(valid_outer_df["outer_test_row_count"].fillna(0).sum()),
                    "distance_to_summary_test_r2": abs(
                        float(valid_outer_df["outer_test_r2"].mean())
                        - pd.to_numeric(pd.Series([row.get("summary_test_r2")]), errors="coerce").iloc[0]
                    ),
                }

    tabpfn_loco_fold_details = _load_tabpfn_loco_fold_details(row)
    if (
        str(row.get("ood_method", "") or "") in {"LOCO", "RandomCV", "HybridHigh20+LOCO", "HybridHigh20+RandCV"}
        and str(row.get("model_family", "") or "") == "TabPFN"
        and tabpfn_loco_fold_details
    ):
        fold_df = pd.DataFrame(tabpfn_loco_fold_details)
        if not fold_df.empty and {"test_r2", "test_mae", "test_rmse"}.issubset(fold_df.columns):
            fold_df["test_r2"] = pd.to_numeric(fold_df["test_r2"], errors="coerce")
            fold_df["test_mae"] = pd.to_numeric(fold_df["test_mae"], errors="coerce")
            fold_df["test_rmse"] = pd.to_numeric(fold_df["test_rmse"], errors="coerce")
            valid_fold_df = fold_df.dropna(subset=["test_r2", "test_mae", "test_rmse"]).copy()
            if not valid_fold_df.empty:
                summary_test_r2 = pd.to_numeric(pd.Series([row.get("summary_test_r2")]), errors="coerce").iloc[0]
                representative_predictions_file = str(row.get("representative_predictions_file", "") or "").strip()
                predictions_file = representative_predictions_file
                if not predictions_file:
                    representative_df = valid_fold_df.assign(
                        _distance_to_summary_test_r2=(valid_fold_df["test_r2"] - summary_test_r2).abs(),
                        _sort_fold=pd.to_numeric(valid_fold_df.get("fold_index"), errors="coerce").fillna(np.inf),
                    ).sort_values(
                        ["_distance_to_summary_test_r2", "test_r2", "_sort_fold", "predictions_file"],
                        ascending=[True, False, True, True],
                        na_position="last",
                        kind="stable",
                    )
                    predictions_file = str(representative_df.iloc[0].get("predictions_file", "") or "")
                source_mode = (
                    "tabpfn_loco_fold_aggregate"
                    if str(row.get("ood_method", "") or "") == "LOCO"
                    else "tabpfn_randomcv_fold_aggregate"
                )
                return {
                    "predictions_file": predictions_file or pd.NA,
                    "source_mode": source_mode,
                    "expected_split_file": pd.NA,
                    "expected_split_count": np.nan,
                    "test_r2": float(valid_fold_df["test_r2"].mean()),
                    "test_mae": float(valid_fold_df["test_mae"].mean()),
                    "test_rmse": float(valid_fold_df["test_rmse"].mean()),
                    "test_row_count": np.nan,
                    "distance_to_summary_test_r2": abs(
                        float(valid_fold_df["test_r2"].mean()) - summary_test_r2
                    ),
                }

    model_dir_text = str(row.get("model_dir", "") or "").strip()
    model_dir = Path(model_dir_text) if model_dir_text else None
    summary_test_r2 = pd.to_numeric(pd.Series([row.get("summary_test_r2")]), errors="coerce").iloc[0]

    expected_split_file: Path | None = None
    expected_split_df: pd.DataFrame | None = None
    expected_ids: set[str] = set()
    if model_dir is not None:
        expected_split_file, expected_split_df, expected_ids = _load_expected_test_split(model_dir)

    candidate_records: list[dict[str, object]] = []
    if model_dir is not None:
        for candidate_file in _collect_case_level_prediction_candidates(model_dir):
            full_df, test_df, _, actual_col, pred_col = extract_prediction_frames(candidate_file, property_name)
            if full_df is None or test_df is None or actual_col is None or pred_col is None:
                continue
            metrics = _compute_subset_metrics(test_df, actual_col, pred_col)
            if metrics is None:
                continue
            id_match = _score_id_match(test_df, expected_ids) if expected_ids else {}
            candidate_records.append(
                {
                    "predictions_file": str(candidate_file),
                    "source_mode": _artifact_source_label(model_dir, candidate_file),
                    "expected_split_file": str(expected_split_file) if expected_split_file is not None else pd.NA,
                    "expected_split_count": len(expected_ids) if expected_ids else pd.NA,
                    **id_match,
                    **metrics,
                }
            )

    if not candidate_records:
        fallback_file_text = str(row.get("representative_predictions_file", "") or "").strip()
        if not fallback_file_text:
            return None
        fallback_file = Path(fallback_file_text)
        full_df, test_df, _, actual_col, pred_col = extract_prediction_frames(fallback_file, property_name)
        if full_df is None or actual_col is None or pred_col is None:
            return None
        metric_source_df = test_df if test_df is not None else full_df
        metrics = _compute_subset_metrics(metric_source_df, actual_col, pred_col)
        if metrics is None:
            return None
        return {
            "predictions_file": str(fallback_file),
            "source_mode": "representative_fold_predictions",
            "expected_split_file": str(expected_split_file) if expected_split_file is not None else pd.NA,
            "expected_split_count": len(expected_ids) if expected_ids else pd.NA,
            **metrics,
        }

    ranked_candidates = pd.DataFrame(candidate_records)
    if expected_ids:
        ranked_candidates["_sort_exact_match"] = ranked_candidates["id_exact_match"].fillna(False).astype(int)
        ranked_candidates["_sort_overlap"] = pd.to_numeric(ranked_candidates["id_overlap_count"], errors="coerce").fillna(-1)
        ranked_candidates["_sort_only_split"] = pd.to_numeric(ranked_candidates["id_only_in_split"], errors="coerce").fillna(np.inf)
        ranked_candidates["_sort_only_candidate"] = pd.to_numeric(ranked_candidates["id_only_in_candidate"], errors="coerce").fillna(np.inf)
        ranked_candidates = ranked_candidates.sort_values(
            [
                "_sort_exact_match",
                "_sort_overlap",
                "_sort_only_split",
                "_sort_only_candidate",
                "predictions_file",
            ],
            ascending=[False, False, True, True, True],
            na_position="last",
            kind="stable",
        ).reset_index(drop=True)
        top_row = ranked_candidates.iloc[0]
        has_positive_match = bool(top_row.get("_sort_exact_match", 0)) or float(top_row.get("_sort_overlap", 0)) > 0
        if not has_positive_match:
            ranked_candidates["_distance_to_summary_test_r2"] = (
                pd.to_numeric(ranked_candidates["test_r2"], errors="coerce") - summary_test_r2
            ).abs()
            ranked_candidates = ranked_candidates.sort_values(
                ["_distance_to_summary_test_r2", "test_r2", "test_mae", "predictions_file"],
                ascending=[True, False, True, True],
                na_position="last",
                kind="stable",
            ).reset_index(drop=True)
    else:
        ranked_candidates["_distance_to_summary_test_r2"] = (
            pd.to_numeric(ranked_candidates["test_r2"], errors="coerce") - summary_test_r2
        ).abs()
        ranked_candidates = ranked_candidates.sort_values(
            ["_distance_to_summary_test_r2", "test_r2", "test_mae", "predictions_file"],
            ascending=[True, False, True, True],
            na_position="last",
            kind="stable",
        ).reset_index(drop=True)
    selected_series = ranked_candidates.iloc[0].copy()
    for helper_col in [
        "_distance_to_summary_test_r2",
        "_sort_exact_match",
        "_sort_overlap",
        "_sort_only_split",
        "_sort_only_candidate",
        "candidate_ids",
    ]:
        if helper_col in selected_series.index:
            selected_series = selected_series.drop(labels=[helper_col])
    selected = selected_series.to_dict()
    selected["distance_to_summary_test_r2"] = (
        float(ranked_candidates.iloc[0]["_distance_to_summary_test_r2"])
        if "_distance_to_summary_test_r2" in ranked_candidates.columns and pd.notna(ranked_candidates.iloc[0]["_distance_to_summary_test_r2"])
        else np.nan
    )
    return selected


def enrich_summary_with_artifact_and_plot_metrics(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df

    working_df = summary_df.copy()
    for column in [
        "artifact_selection_mode",
        "artifact_predictions_file",
        "artifact_expected_split_file",
    ]:
        if column not in working_df.columns:
            working_df[column] = pd.NA
    for column in [
        "artifact_test_r2",
        "artifact_test_mae",
        "artifact_test_rmse",
        "artifact_test_row_count",
        "plot_test_r2",
        "plot_test_mae",
        "plot_test_rmse",
    ]:
        if column not in working_df.columns:
            working_df[column] = pd.NA

    for idx, row in working_df.iterrows():
        artifact_info = resolve_case_level_artifact(row)
        if not artifact_info:
            continue
        working_df.at[idx, "artifact_selection_mode"] = artifact_info.get("source_mode", pd.NA)
        working_df.at[idx, "artifact_predictions_file"] = artifact_info.get("predictions_file", pd.NA)
        working_df.at[idx, "artifact_expected_split_file"] = artifact_info.get("expected_split_file", pd.NA)
        working_df.at[idx, "artifact_test_r2"] = artifact_info.get("test_r2", pd.NA)
        working_df.at[idx, "artifact_test_mae"] = artifact_info.get("test_mae", pd.NA)
        working_df.at[idx, "artifact_test_rmse"] = artifact_info.get("test_rmse", pd.NA)
        working_df.at[idx, "artifact_test_row_count"] = artifact_info.get("test_row_count", pd.NA)

    for metric in ["r2", "mae", "rmse"]:
        plot_col = f"plot_test_{metric}"
        artifact_col = f"artifact_test_{metric}"
        representative_col = f"representative_test_{metric}"
        summary_col = f"summary_test_{metric}"
        plot_series = pd.to_numeric(working_df[plot_col], errors="coerce")
        artifact_series = pd.to_numeric(working_df[artifact_col], errors="coerce")
        representative_series = pd.to_numeric(working_df[representative_col], errors="coerce")
        summary_series = pd.to_numeric(working_df[summary_col], errors="coerce")
        plot_series = plot_series.where(plot_series.notna(), artifact_series)
        plot_series = plot_series.where(plot_series.notna(), representative_series)
        plot_series = plot_series.where(plot_series.notna(), summary_series)
        working_df[plot_col] = plot_series

    return working_df


def align_summary_metrics_to_artifact(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df

    working_df = enrich_summary_with_artifact_and_plot_metrics(summary_df)

    summary_mae = pd.to_numeric(working_df.get("summary_test_mae"), errors="coerce")
    artifact_mae = pd.to_numeric(working_df.get("artifact_test_mae"), errors="coerce")
    summary_r2 = pd.to_numeric(working_df.get("summary_test_r2"), errors="coerce")
    artifact_r2 = pd.to_numeric(working_df.get("artifact_test_r2"), errors="coerce")
    summary_rmse = pd.to_numeric(working_df.get("summary_test_rmse"), errors="coerce")
    artifact_rmse = pd.to_numeric(working_df.get("artifact_test_rmse"), errors="coerce")

    changed_mask = (
        artifact_mae.notna()
        & artifact_r2.notna()
        & artifact_rmse.notna()
        & (
            (summary_mae - artifact_mae).abs().fillna(0.0).gt(1e-9)
            | (summary_r2 - artifact_r2).abs().fillna(0.0).gt(1e-9)
            | (summary_rmse - artifact_rmse).abs().fillna(0.0).gt(1e-9)
        )
    )

    for metric in ["r2", "mae", "rmse"]:
        summary_col = f"summary_test_{metric}"
        artifact_col = f"artifact_test_{metric}"
        working_df[summary_col] = pd.to_numeric(working_df[artifact_col], errors="coerce").where(
            pd.to_numeric(working_df[artifact_col], errors="coerce").notna(),
            pd.to_numeric(working_df[summary_col], errors="coerce"),
        )

    for std_col in ["summary_test_r2_std", "summary_test_mae_std", "summary_test_rmse_std"]:
        working_df[std_col] = pd.to_numeric(working_df[std_col], errors="coerce")
        working_df.loc[changed_mask, std_col] = 0.0
        working_df[std_col] = working_df[std_col].fillna(0.0)

    for count_col in ["trial_count", "fold_count"]:
        working_df[count_col] = pd.to_numeric(working_df[count_col], errors="coerce")
        working_df.loc[changed_mask, count_col] = 1

    working_df.loc[changed_mask, "representative_selection_mode"] = "case_level_oodtest"
    working_df.loc[changed_mask, "representative_trial_id"] = pd.NA
    working_df.loc[changed_mask, "representative_fold"] = pd.NA
    for metric in ["r2", "mae", "rmse"]:
        rep_col = f"representative_test_{metric}"
        artifact_col = f"artifact_test_{metric}"
        working_df.loc[changed_mask, rep_col] = pd.to_numeric(working_df.loc[changed_mask, artifact_col], errors="coerce")
    if "artifact_predictions_file" in working_df.columns:
        working_df.loc[changed_mask, "representative_predictions_file"] = working_df.loc[
            changed_mask, "artifact_predictions_file"
        ]

    for metric in ["r2", "mae", "rmse"]:
        plot_col = f"plot_test_{metric}"
        working_df[plot_col] = pd.to_numeric(working_df[plot_col], errors="coerce").where(
            pd.to_numeric(working_df[plot_col], errors="coerce").notna(),
            pd.to_numeric(working_df[f"summary_test_{metric}"], errors="coerce"),
        )

    return working_df


def _load_loco_outer_fold_best_details(row: pd.Series) -> list[dict[str, object]]:
    raw_details = row.get("loco_outer_fold_best_details_json", pd.NA)
    if pd.isna(raw_details):
        return []

    text = str(raw_details).strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []

    if not isinstance(parsed, list):
        return []

    details: list[dict[str, object]] = []
    for item in parsed:
        if isinstance(item, dict):
            details.append(item)
    details.sort(
        key=lambda item: (
            np.inf if pd.isna(item.get("outer_fold_index")) else int(item.get("outer_fold_index")),
            str(item.get("selected_trial_id", "")),
        )
    )
    return details


def _load_tabpfn_loco_fold_details(row: pd.Series) -> list[dict[str, object]]:
    raw_details = row.get("tabpfn_loco_fold_details_json", pd.NA)
    if pd.isna(raw_details):
        return []

    text = str(raw_details).strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []

    if not isinstance(parsed, list):
        return []

    details: list[dict[str, object]] = []
    for item in parsed:
        if isinstance(item, dict):
            details.append(item)
    details.sort(
        key=lambda item: (
            np.inf if pd.isna(item.get("fold_index")) else int(item.get("fold_index")),
            str(item.get("predictions_file", "")),
        )
    )
    return details


def _build_loco_outer_fold_export_row(base_row: pd.Series, fold_detail: dict[str, object]) -> pd.Series:
    fold_row = base_row.copy()
    fold_row["model_dir"] = fold_detail.get("model_dir", fold_row.get("model_dir", pd.NA))
    fold_row["trial_count"] = 1
    fold_row["fold_count"] = fold_detail.get("selected_trial_fold_count", fold_row.get("fold_count", pd.NA))
    fold_row["summary_test_r2"] = fold_detail.get("outer_test_r2", fold_row.get("summary_test_r2", pd.NA))
    fold_row["summary_test_r2_std"] = 0.0
    fold_row["summary_test_mae"] = fold_detail.get("outer_test_mae", fold_row.get("summary_test_mae", pd.NA))
    fold_row["summary_test_mae_std"] = 0.0
    fold_row["summary_test_rmse"] = fold_detail.get("outer_test_rmse", fold_row.get("summary_test_rmse", pd.NA))
    fold_row["summary_test_rmse_std"] = 0.0
    fold_row["representative_selection_mode"] = "loco_outer_fold_oodtest"
    fold_row["representative_trial_id"] = fold_detail.get(
        "selected_trial_id",
        fold_row.get("representative_trial_id", pd.NA),
    )
    fold_row["representative_fold"] = fold_detail.get(
        "outer_fold_index",
        fold_row.get("representative_fold", pd.NA),
    )
    fold_row["representative_test_r2"] = fold_detail.get(
        "outer_test_r2",
        fold_row.get("representative_test_r2", pd.NA),
    )
    fold_row["representative_test_mae"] = fold_detail.get(
        "outer_test_mae",
        fold_row.get("representative_test_mae", pd.NA),
    )
    fold_row["representative_test_rmse"] = fold_detail.get(
        "outer_test_rmse",
        fold_row.get("representative_test_rmse", pd.NA),
    )
    fold_row["representative_predictions_file"] = fold_detail.get(
        "outer_predictions_file",
        fold_row.get("representative_predictions_file", pd.NA),
    )
    fold_row["loco_outer_fold_index"] = fold_detail.get("outer_fold_index", pd.NA)
    fold_row["loco_selected_trial_num"] = fold_detail.get("selected_trial_num", pd.NA)
    fold_row["loco_selected_trial_fold_count"] = fold_detail.get("selected_trial_fold_count", pd.NA)
    return fold_row


def export_selected_artifacts(row: pd.Series, case_root: Path, *, method_dir: Path | None = None) -> None:
    method_dir = method_dir or (case_root / safe_name(str(row["ood_method"])) / safe_name(str(row["model"])))
    artifacts_dir = method_dir / "selected_model_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    artifact_info = resolve_case_level_artifact(row)
    property_name = str(row.get("property", ""))

    # 1) Always export the actual OOD split-aligned test set for manual row-wise comparison.
    test_predictions_file = Path(str(artifact_info["predictions_file"])) if artifact_info is not None else Path("")
    expected_split_df = None
    export_test_df = None
    if artifact_info is not None and pd.notna(artifact_info.get("expected_split_file")):
        expected_split_path = Path(str(artifact_info.get("expected_split_file")))
        if expected_split_path.exists():
            try:
                expected_split_df = pd.read_csv(expected_split_path, low_memory=False)
            except Exception as exc:
                print(f"[WARN] Failed to read expected split file {expected_split_path}: {exc}")
    if test_predictions_file.exists():
        _, test_df, _, _, _ = extract_prediction_frames(test_predictions_file, property_name)
        if test_df is not None and not test_df.empty:
            export_test_df = _align_test_df_to_split_ids(test_df, expected_split_df)
            save_csv(export_test_df, method_dir / "test_oodmethod.csv")

    # 2) selected_predictions.csv should correspond to the exported alloy-case result
    # from the original experiment root (i.e. the actual OOD test split for this model).
    # representative_* fields remain available in the summary, but the selected artifact
    # directory should stay traceable to output/ood_results/... directly.
    representative_predictions_file = Path(str(row.get("representative_predictions_file", "") or ""))
    selected_source_mode = (
        str(artifact_info.get("source_mode", "") or "").strip()
        if artifact_info is not None
        else str(row.get("representative_selection_mode", "") or "").strip()
    )
    selected_source_file = test_predictions_file if test_predictions_file.exists() else representative_predictions_file

    selected_predictions_output = artifacts_dir / "selected_predictions.csv"
    selected_metrics: dict[str, object] = {
        "selected_source_mode": pd.NA,
        "selected_source_predictions_file": pd.NA,
        "selected_test_r2": pd.NA,
        "selected_test_mae": pd.NA,
        "selected_test_rmse": pd.NA,
        "selected_test_row_count": pd.NA,
    }

    export_selected_df = None
    actual_col = None
    pred_col = None
    metric_source_df = None

    if export_test_df is not None and not export_test_df.empty:
        export_selected_df = export_test_df.copy()
        _, _, _, actual_col, pred_col = extract_prediction_frames(selected_source_file, property_name)
        metric_source_df = export_selected_df
    elif selected_source_file.exists():
        full_df, selected_test_df, _, actual_col, pred_col = extract_prediction_frames(selected_source_file, property_name)
        export_selected_df = selected_test_df if selected_test_df is not None and not selected_test_df.empty else full_df
        metric_source_df = export_selected_df

    if export_selected_df is not None and not export_selected_df.empty:
        save_csv(export_selected_df, selected_predictions_output)
        plot_diagonal_chart(selected_predictions_output, property_name, artifacts_dir / "selected_plot.png")
        if actual_col is not None and pred_col is not None:
            metrics = _compute_subset_metrics(metric_source_df, actual_col, pred_col)
            if metrics is not None:
                selected_metrics.update(
                    {
                        "selected_source_mode": selected_source_mode or pd.NA,
                        "selected_source_predictions_file": str(selected_source_file),
                        "selected_test_r2": metrics.get("test_r2"),
                        "selected_test_mae": metrics.get("test_mae"),
                        "selected_test_rmse": metrics.get("test_rmse"),
                        "selected_test_row_count": metrics.get("test_row_count"),
                    }
                )
    else:
        plot_file = Path(str(row.get("representative_plot_file", "") or ""))
        if plot_file.exists():
            try:
                shutil.copy2(plot_file, artifacts_dir / "selected_plot.png")
            except PermissionError:
                pass

    model_dir = Path(str(row.get("model_dir", "") or ""))
    if model_dir.exists():
        (method_dir / "model_source_path.txt").write_text(str(model_dir.resolve()), encoding="utf-8")
        for filename in [
            "final_model_evaluation_metrics.json",
            "best_model_best_model_evaluation_evaluation_summary.json",
            "metrics_summary.json",
            "pipeline_manifest.json",
            "ood_manifest.json",
            "cv_avg_metrics.json",
            "optuna_best_params.json",
        ]:
            candidate = model_dir / filename
            if candidate.exists():
                try:
                    shutil.copy2(candidate, artifacts_dir / filename)
                except PermissionError:
                    pass

    selected_row = row.to_dict()
    selected_row.update(selected_metrics)
    if artifact_info is not None:
        selected_row.update(
            {
                "artifact_selection_mode": artifact_info.get("source_mode"),
                "artifact_predictions_file": artifact_info.get("predictions_file"),
                "artifact_expected_split_file": artifact_info.get("expected_split_file"),
                "artifact_expected_split_count": artifact_info.get("expected_split_count"),
                "artifact_id_exact_match": artifact_info.get("id_exact_match"),
                "artifact_id_overlap_count": artifact_info.get("id_overlap_count"),
                "artifact_id_only_in_candidate": artifact_info.get("id_only_in_candidate"),
                "artifact_id_only_in_split": artifact_info.get("id_only_in_split"),
                "artifact_test_r2": artifact_info.get("test_r2"),
                "artifact_test_mae": artifact_info.get("test_mae"),
                "artifact_test_rmse": artifact_info.get("test_rmse"),
                "artifact_test_row_count": artifact_info.get("test_row_count"),
                "artifact_distance_to_summary_test_r2": artifact_info.get("distance_to_summary_test_r2"),
            }
        )
    save_csv(pd.DataFrame([selected_row]), method_dir / "selected_result_summary.csv")


def export_loco_outer_fold_best_artifacts(row: pd.Series, case_root: Path) -> None:
    if str(row.get("ood_method", "")) != "LOCO":
        return

    fold_details = _load_loco_outer_fold_best_details(row)
    if not fold_details:
        return

    method_root = case_root / safe_name(str(row["ood_method"]))
    model_root = method_root / safe_name(str(row["model"]))
    model_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    for fold_detail in fold_details:
        outer_fold_index = fold_detail.get("outer_fold_index")
        if pd.isna(outer_fold_index) or outer_fold_index is None:
            continue

        fold_index = int(outer_fold_index)
        fold_row = _build_loco_outer_fold_export_row(row, fold_detail)
        summary_rows.append(
            {
                "alloy_family": fold_row.get("alloy_family", pd.NA),
                "dataset_name": fold_row.get("dataset_name", pd.NA),
                "property": fold_row.get("property", pd.NA),
                "ood_method": fold_row.get("ood_method", pd.NA),
                "model": fold_row.get("model", pd.NA),
                "outer_fold_index": fold_index,
                "selected_trial_id": fold_detail.get("selected_trial_id", pd.NA),
                "selected_trial_num": fold_detail.get("selected_trial_num", pd.NA),
                "selected_trial_fold_count": fold_detail.get("selected_trial_fold_count", pd.NA),
                "selected_mean_test_r2": fold_detail.get("selected_mean_test_r2", pd.NA),
                "selected_std_test_r2": fold_detail.get("selected_std_test_r2", pd.NA),
                "selected_mean_test_mae": fold_detail.get("selected_mean_test_mae", pd.NA),
                "selected_std_test_mae": fold_detail.get("selected_std_test_mae", pd.NA),
                "selected_mean_test_rmse": fold_detail.get("selected_mean_test_rmse", pd.NA),
                "selected_std_test_rmse": fold_detail.get("selected_std_test_rmse", pd.NA),
                "selected_inner_predictions_file": fold_detail.get("selected_inner_predictions_file", pd.NA),
                "outer_predictions_file": fold_detail.get("outer_predictions_file", pd.NA),
                "outer_test_r2": fold_detail.get("outer_test_r2", pd.NA),
                "outer_test_mae": fold_detail.get("outer_test_mae", pd.NA),
                "outer_test_rmse": fold_detail.get("outer_test_rmse", pd.NA),
                "outer_test_row_count": fold_detail.get("outer_test_row_count", pd.NA),
                "model_dir": fold_detail.get("model_dir", pd.NA),
            }
        )
        export_selected_artifacts(
            fold_row,
            case_root,
            method_dir=method_root / f"fold_{fold_index}" / safe_name(str(row["model"])),
        )

    if summary_rows:
        save_csv(pd.DataFrame(summary_rows), model_root / "loco_outer_fold_best_summary.csv")


def export_case_outputs(summary_df: pd.DataFrame, summary_root: Path) -> None:
    canonical_df = annotate_family_ranks(summary_df)
    cases_root = summary_root / CASES_DIRNAME
    ood_summary_root = summary_root / OOD_SUMMARY_DIRNAME
    grouped = canonical_df.groupby(["alloy_family", "dataset_name", "property"], sort=True)

    for alloy_family, dataset_name, property_name in grouped.groups:
        case_df = (
            grouped.get_group((alloy_family, dataset_name, property_name))
            .sort_values(["ood_method", "rank_within_family", "model"])
            .reset_index(drop=True)
        )
        case_root = cases_root / alloy_family / dataset_name / safe_name(property_name)
        ood_root = ood_summary_root / alloy_family / dataset_name / safe_name(property_name)

        save_csv(case_df, case_root / "case_model_summary.csv")
        save_csv(build_case_wide_summary(case_df), ood_root / "ood_method_metric_summary.csv")
        plot_case_metric(case_df, ood_root / "ood_method_r2_summary.png", "summary_test_r2", "R2", err_col="summary_test_r2_std")
        plot_case_metric(case_df, ood_root / "ood_method_mae_summary.png", "summary_test_mae", "MAE", err_col="summary_test_mae_std")
        plot_case_metric(case_df, ood_root / "ood_method_rmse_summary.png", "summary_test_rmse", "RMSE", err_col="summary_test_rmse_std")

        for _, row in case_df.iterrows():
            export_selected_artifacts(row, case_root)
            export_loco_outer_fold_best_artifacts(row, case_root)


def create_global_exports(summary_df: pd.DataFrame, summary_root: Path, output_filename: str) -> None:
    summary_df = summary_df.copy()
    if "alloy_family" in summary_df.columns:
        summary_df["alloy_family"] = summary_df["alloy_family"].map(normalize_alloy_family_name)
    summary_tables_dir = summary_root / SUMMARY_TABLES_DIRNAME
    save_csv(
        summary_df.sort_values(["alloy_family", "dataset_name", "property", "ood_method", "model"]).reset_index(drop=True),
        summary_tables_dir / output_filename,
    )


def reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def iter_experiment_dirs(base_dir: Path, prefix: str) -> Iterable[Path]:
    return sorted(path for path in base_dir.iterdir() if path.is_dir() and path.name.startswith(prefix))
