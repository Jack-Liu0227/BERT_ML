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


def plot_dataset_property_summary(dataset_df: pd.DataFrame, value_col: str, output_path: Path) -> None:
    plot_df = dataset_df[["model", "property", value_col]].dropna().copy()
    if plot_df.empty:
        return

    properties = sorted(plot_df["property"].unique())
    x = np.arange(len(MODEL_ORDER))
    width = 0.8 / max(len(properties), 1)

    fig, ax = plt.subplots(figsize=(10, 8))
    for idx, property_name in enumerate(properties):
        prop_df = plot_df[plot_df["property"] == property_name].copy()
        prop_df["model"] = pd.Categorical(prop_df["model"], categories=MODEL_ORDER, ordered=True)
        prop_df = prop_df.sort_values("model").set_index("model").reindex(MODEL_ORDER).reset_index()
        offset = (idx - (len(properties) - 1) / 2) * width
        ax.bar(
            x + offset,
            prop_df[value_col].tolist(),
            width=width,
            color=COLORS[idx % len(COLORS)],
            edgecolor="black",
            linewidth=1.1,
            label=property_name,
        )

    ax.set_xlabel("Predictive models", fontweight="bold")
    ax.set_ylabel(R2_LABEL, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis="both", which="both", direction="in")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.legend(frameon=True, edgecolor="black")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


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
    ax.tick_params(axis="both", which="both", direction="in")
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
        label=f"Best-trial representative {R2_LABEL}",
    )
    ax.bar(
        x + width,
        plot_df["global_mean_test_r2"],
        width=width,
        color="#90be6d",
        edgecolor="black",
        linewidth=1.2,
        label=f"Global representative {R2_LABEL}",
    )
    ax.set_xlabel("Predictive models", fontweight="bold")
    ax.set_ylabel(R2_LABEL, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["model"])
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis="both", which="both", direction="in")
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
        global_metrics = extract_metrics(
            load_json(
                model_dir
                / "closest_to_global_mean_trial_fold"
                / "closest_to_global_mean_trial_fold_metrics.json"
            ),
            "closest_to_global_mean_trial_fold",
            "global_mean",
        )
        closest_metrics = extract_metrics(
            load_json(model_dir / "closest_to_mean_evaluation" / "closest_to_mean_evaluation_metrics.json"),
            "closest_to_mean_evaluation",
            "closest_mean",
        )

        properties = sorted(set(final_metrics) | set(global_metrics) | set(closest_metrics))
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
                "closest_mean_predictions_file": str(
                    model_dir
                    / "closest_to_mean_evaluation"
                    / "predictions"
                    / "closest_to_mean_evaluation_all_predictions.csv"
                ),
                "global_mean_predictions_file": str(
                    model_dir
                    / "closest_to_global_mean_trial_fold"
                    / "predictions"
                    / "closest_to_global_mean_trial_fold_all_predictions.csv"
                ),
                "final_plot_file": str(model_dir / "plots" / "final_model_evaluation_all_sets_comparison.png"),
                "closest_mean_plot_file": str(
                    model_dir
                    / "closest_to_mean_evaluation"
                    / "plots"
                    / "closest_to_mean_evaluation_all_sets_comparison.png"
                ),
                "global_mean_plot_file": str(
                    model_dir
                    / "closest_to_global_mean_trial_fold"
                    / "plots"
                    / "closest_to_global_mean_trial_fold_all_sets_comparison.png"
                ),
            }
            row.update(final_metrics.get(property_name, {}))
            row.update(closest_metrics.get(property_name, {}))
            row.update(global_metrics.get(property_name, {}))
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

        plot_dataset_property_summary(dataset_df, "final_test_r2", dataset_dir / "comparisons" / "final_test_r2_summary.png")
        plot_dataset_property_summary(
            dataset_df,
            "closest_mean_test_r2",
            dataset_dir / "comparisons" / "closest_mean_r2_summary.png",
        )
        plot_dataset_property_summary(
            dataset_df,
            "global_mean_test_r2",
            dataset_dir / "comparisons" / "global_mean_r2_summary.png",
        )

        for property_name, property_df in dataset_df.groupby("property"):
            case_dir = dataset_dir / safe_name(property_name)
            save_csv(
                property_df.sort_values(["model"]).reset_index(drop=True),
                case_dir / "case_model_summary.csv",
            )
            best_property = dataset_best[dataset_best["property"] == property_name].reset_index(drop=True)
            if not best_property.empty:
                save_csv(best_property, case_dir / "best_model_summary.csv")

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

            if best_property.empty:
                continue

            best_row = best_property.iloc[0]
            model_copy_dir = case_dir / "selected_model_source" / safe_name(str(best_row["model"]))
            artifacts_dir = case_dir / "selected_model_artifacts"

            copy_tree_if_exists(Path(best_row["model_dir"]), model_copy_dir)
            copy_if_exists(Path(best_row["final_predictions_file"]), artifacts_dir / "final_predictions.csv")
            copy_if_exists(
                Path(best_row["closest_mean_predictions_file"]),
                artifacts_dir / "closest_mean_predictions.csv",
            )
            copy_if_exists(
                Path(best_row["global_mean_predictions_file"]),
                artifacts_dir / "global_mean_predictions.csv",
            )
            copy_if_exists(Path(best_row["final_plot_file"]), artifacts_dir / "final_diagnostic.png")
            copy_if_exists(
                Path(best_row["closest_mean_plot_file"]),
                artifacts_dir / "closest_mean_diagnostic.png",
            )
            copy_if_exists(
                Path(best_row["global_mean_plot_file"]),
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
