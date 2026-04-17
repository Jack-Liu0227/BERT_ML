from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

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
    "target_extrapolation": "Extrapolation",
    "loco": "LOCO",
    "sparse_x_cluster": "SparseXcluster",
    "sparse_x_single": "SparseXsingle",
    "sparse_y_cluster": "SparseYcluster",
    "sparse_y_single": "SparseYsingle",
}
OOD_METHOD_ORDER = [
    "Extrapolation",
    "LOCO",
    "SparseXcluster",
    "SparseXsingle",
    "SparseYcluster",
    "SparseYsingle",
]
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
    return normalized


def save_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


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


def export_selected_artifacts(row: pd.Series, case_root: Path) -> None:
    method_dir = case_root / safe_name(str(row["ood_method"])) / safe_name(str(row["model"]))
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
