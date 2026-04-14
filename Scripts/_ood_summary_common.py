from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, MultipleLocator


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

    working_df = case_df[["model", "ood_method", metric_col] + ([err_col] if err_col and err_col in case_df.columns else [])].copy()
    working_df[metric_col] = pd.to_numeric(working_df[metric_col], errors="coerce")
    working_df = working_df.dropna(subset=[metric_col])
    if working_df.empty:
        return

    model_order = sorted(working_df["model"].astype(str).unique())
    method_order = [method for method in OOD_METHOD_ORDER if method in set(working_df["ood_method"].astype(str))]
    if not method_order:
        return

    pivot_df = working_df.pivot_table(
        index="ood_method",
        columns="model",
        values=metric_col,
        aggfunc="first",
        observed=False,
    ).reindex(index=method_order, columns=model_order)

    pivot_err_df = None
    if err_col and err_col in working_df.columns:
        working_df[err_col] = pd.to_numeric(working_df[err_col], errors="coerce").fillna(0.0)
        pivot_err_df = working_df.pivot_table(
            index="ood_method",
            columns="model",
            values=err_col,
            aggfunc="first",
            observed=False,
        ).reindex(index=method_order, columns=model_order).fillna(0.0)

    x = np.arange(len(method_order))
    width = 0.8 / max(len(model_order), 1)
    colors = ["#f4a698", "#9bbfe0", "#c8d5b9", "#d4a5d6", "#f6d186", "#84a59d", "#e5989b", "#6d597a"]

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
    ax.legend(frameon=True, edgecolor="black")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


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


def export_selected_artifacts(row: pd.Series, case_root: Path) -> None:
    method_dir = case_root / safe_name(str(row["ood_method"])) / safe_name(str(row["model"]))
    artifacts_dir = method_dir / "selected_model_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    predictions_file = Path(str(row.get("representative_predictions_file", "") or ""))
    if predictions_file.exists():
        try:
            shutil.copy2(predictions_file, artifacts_dir / "selected_predictions.csv")
        except PermissionError:
            pass

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

    save_csv(pd.DataFrame([row.to_dict()]), method_dir / "selected_result_summary.csv")


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
