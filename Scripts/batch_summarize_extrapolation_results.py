"""
Batch summary script for extrapolation experiment results.

The script scans output/extrapolation_results, reads per-model final test metrics
and global-mean CV metrics, selects the best extrapolation model for each case,
and writes compact summaries plus comparison plots.
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


def extract_final_metrics(metrics: Dict) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    pattern = re.compile(
        r"^final_model_evaluation_(train|val|test)_(.+)_(r2|rmse|mae)$"
    )

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
    pattern = re.compile(
        r"^closest_to_mean_evaluation_(train|val|test)_(.+)_(r2|rmse|mae)$"
    )

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


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


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
    ordered_df["model"] = pd.Categorical(
        ordered_df["model"], categories=MODEL_ORDER, ordered=True
    )
    ordered_df = ordered_df.sort_values("model").dropna(subset=[value_col])
    if ordered_df.empty:
        return

    x = np.arange(len(ordered_df))
    width = 0.55

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.bar(
        x,
        ordered_df[value_col],
        width=width,
        color="#ffb703",
        edgecolor="black",
        linewidth=1.2,
        label=legend_label,
    )

    ax.set_xlabel("Models", fontweight="bold")
    ax.set_ylabel(metric_label, fontweight="bold")
    ax.set_title(case_label, fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered_df["model"])
    if metric_label == "R2":
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax.tick_params(axis="both", which="both", direction="in")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
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
    ordered_df["model"] = pd.Categorical(
        ordered_df["model"], categories=MODEL_ORDER, ordered=True
    )
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

    ax.set_xlabel("Models", fontweight="bold")
    ax.set_ylabel(metric_label, fontweight="bold")
    ax.set_title(case_label, fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered_df["model"])
    if metric_label == "R2":
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.25))
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
            if path.is_dir() and path.parent.name != "all_extrapolation_summary"
        ]
    )


def process_case(model_comparison_dir: Path, summary_root: Path, base_dir: Path) -> List[Dict]:
    relative_parts = model_comparison_dir.relative_to(base_dir).parts
    case_parts = relative_parts[:-1]
    rows: List[Dict] = []

    for model_dir in sorted(model_comparison_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = MODEL_MAP.get(model_dir.name)
        if model_name is None:
            continue

        final_metrics = extract_final_metrics(
            load_json(model_dir / "final_model_evaluation_metrics.json")
        )
        representative_metrics = extract_representative_metrics(
            load_json(
                model_dir
                / "closest_to_global_mean_trial_fold"
                / "closest_to_global_mean_trial_fold_metrics.json"
            )
        )
        closest_mean_metrics = extract_closest_to_mean_metrics(
            load_json(
                model_dir
                / "closest_to_mean_evaluation"
                / "closest_to_mean_evaluation_metrics.json"
            )
        )
        closest_info = extract_closest_fold_info(
            load_json(
                model_dir
                / "closest_to_global_mean_trial_fold"
                / "closest_to_global_mean_fold_info.json"
            )
        )

        targets = sorted(
            set(final_metrics) | set(representative_metrics) | set(closest_mean_metrics)
        )
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
            row.update(representative_metrics.get(target, {}))
            row.update(closest_mean_metrics.get(target, {}))

            if "closest_fold_num" in closest_info:
                row["closest_fold_num"] = closest_info["closest_fold_num"]
            target_key = f"{target}__closest_fold_test_r2"
            if target_key in closest_info:
                row["closest_fold_test_r2"] = closest_info[target_key]
            elif "closest_fold_test_r2" in closest_info:
                row["closest_fold_test_r2"] = closest_info["closest_fold_test_r2"]
            if "closest_fold_mean_test_r2" in closest_info:
                row["closest_fold_mean_test_r2"] = closest_info[
                    "closest_fold_mean_test_r2"
                ]

            row["final_predictions_file"] = str(
                model_dir / "predictions" / "all_predictions.csv"
            )
            row["representative_predictions_file"] = str(
                model_dir
                / "closest_to_global_mean_trial_fold"
                / "predictions"
                / "closest_to_global_mean_trial_fold_all_predictions.csv"
            )
            row["closest_mean_predictions_file"] = str(
                model_dir
                / "closest_to_mean_evaluation"
                / "predictions"
                / "closest_to_mean_evaluation_all_predictions.csv"
            )
            row["final_plot_file"] = str(
                model_dir / "plots" / "final_model_evaluation_all_sets_comparison.png"
            )
            row["representative_plot_file"] = str(
                model_dir
                / "closest_to_global_mean_trial_fold"
                / "plots"
                / "closest_to_global_mean_trial_fold_all_sets_comparison.png"
            )
            row["closest_mean_plot_file"] = str(
                model_dir
                / "closest_to_mean_evaluation"
                / "plots"
                / "closest_to_mean_evaluation_all_sets_comparison.png"
            )
            rows.append(row)

    return rows


def export_alloy_level_views(summary_df: pd.DataFrame, summary_root: Path) -> None:
    for alloy_name, alloy_df in summary_df.groupby("alloy_family"):
        alloy_dir = summary_root / alloy_name
        alloy_dir.mkdir(parents=True, exist_ok=True)
        for dataset_name, dataset_df in alloy_df.groupby("dataset_name"):
            dataset_df.to_csv(
                alloy_dir / f"{dataset_name}_extrapolation_model_summary.csv",
                index=False,
                encoding="utf-8-sig",
            )

            best_rows = (
                dataset_df.assign(
                    final_test_r2=dataset_df["final_test_r2"].fillna(-np.inf),
                    closest_mean_test_r2=dataset_df["closest_mean_test_r2"].fillna(-np.inf),
                    representative_test_r2=dataset_df["representative_test_r2"].fillna(-np.inf),
                )
                .sort_values(
                    ["target", "final_test_r2", "closest_mean_test_r2", "representative_test_r2"],
                    ascending=[True, False, False, False],
                    na_position="last",
                )
                .groupby(["target"], group_keys=False)
                .head(1)
                .reset_index(drop=True)
            )

            summary_rows = []
            combined_df = None

            for _, row in best_rows.iterrows():
                target = str(row["target"])
                best_model = str(row["model"])
                safe_target = safe_name(target)
                target_df = dataset_df[dataset_df["target"] == target].copy()
                case_label = f"{alloy_name} / {dataset_name} / {target} / {row['feature_mode']} | {target}"

                plot_metric_comparison(
                    target_df,
                    alloy_dir / f"{dataset_name}_model_comparison_{safe_target}_FinalTest_R2.png",
                    case_label,
                    "final_test_r2",
                    "R2",
                    "Final extrapolation test R2",
                )
                plot_metric_comparison(
                    target_df,
                    alloy_dir / f"{dataset_name}_model_comparison_{safe_target}_FinalTest_MAE.png",
                    case_label,
                    "final_test_mae",
                    "MAE",
                    "Final extrapolation test MAE",
                )

                plot_metric_comparison(
                    target_df,
                    alloy_dir / f"{dataset_name}_model_comparison_{safe_target}_BestTrialMean_R2.png",
                    case_label,
                    "closest_mean_test_r2",
                    "R2",
                    "Best-trial representative R2",
                )
                plot_metric_comparison(
                    target_df,
                    alloy_dir / f"{dataset_name}_model_comparison_{safe_target}_GlobalMean_R2.png",
                    case_label,
                    "representative_test_r2",
                    "R2",
                    "Global representative R2",
                )
                plot_metric_comparison(
                    target_df,
                    alloy_dir / f"{dataset_name}_model_comparison_{safe_target}_BestTrialMean_MAE.png",
                    case_label,
                    "closest_mean_test_mae",
                    "MAE",
                    "Best-trial representative MAE",
                )
                plot_metric_comparison(
                    target_df,
                    alloy_dir / f"{dataset_name}_model_comparison_{safe_target}_GlobalMean_MAE.png",
                    case_label,
                    "representative_test_mae",
                    "MAE",
                    "Global representative MAE",
                )
                plot_three_metric_comparison(
                    target_df,
                    alloy_dir / f"{dataset_name}_model_comparison_{safe_target}_ThreeModes_R2.png",
                    case_label,
                    "R2",
                    "final_test_r2",
                    "closest_mean_test_r2",
                    "representative_test_r2",
                )
                plot_three_metric_comparison(
                    target_df,
                    alloy_dir / f"{dataset_name}_model_comparison_{safe_target}_ThreeModes_MAE.png",
                    case_label,
                    "MAE",
                    "final_test_mae",
                    "closest_mean_test_mae",
                    "representative_test_mae",
                )

                summary_rows.append(
                    {
                        "property": target,
                        "best_model": best_model,
                        "representative_r2": row.get("closest_mean_test_r2"),
                        "file_path": row.get("closest_mean_predictions_file", ""),
                        "all_trials_representative_r2": row.get("representative_test_r2"),
                        "all_trials_file_path": row.get("representative_predictions_file", ""),
                    }
                )

                copy_if_exists(
                    Path(row["closest_mean_predictions_file"]),
                    alloy_dir / f"{dataset_name}_best_model_{best_model}_{safe_target}.csv",
                )
                copy_if_exists(
                    Path(row["final_predictions_file"]),
                    alloy_dir / f"{dataset_name}_final_model_evaluation_test_{best_model}_{safe_target}.csv",
                )
                copy_if_exists(
                    Path(row["closest_mean_plot_file"]),
                    alloy_dir / f"{dataset_name}_{best_model}_best_model_diagonal_{safe_target}.png",
                )
                copy_if_exists(
                    Path(row["representative_plot_file"]),
                    alloy_dir / f"{dataset_name}_{best_model}_mean_model_diagonal_{safe_target}.png",
                )
                copy_if_exists(
                    Path(row["final_plot_file"]),
                    alloy_dir / f"{dataset_name}_{best_model}_final_model_evaluation_test_{safe_target}.png",
                )

                mean_pred_path = Path(row["representative_predictions_file"])
                if mean_pred_path.exists():
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
                        if "ID" in combined_df.columns:
                            combined_df["_original_index"] = range(len(combined_df))
                    else:
                        combined_df = pd.merge(
                            combined_df,
                            temp_df,
                            on="ID",
                            how="outer",
                            suffixes=("", "_extra"),
                        )
                        if "Dataset_extra" in combined_df.columns:
                            combined_df["Dataset"] = combined_df["Dataset"].fillna(
                                combined_df["Dataset_extra"]
                            )
                            combined_df.drop(columns=["Dataset_extra"], inplace=True)

            if summary_rows:
                pd.DataFrame(summary_rows).to_csv(
                    alloy_dir / f"{dataset_name}_best_models_summary.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

            if combined_df is not None:
                if "Dataset" in combined_df.columns:
                    dataset_order_map = {
                        "train": 0,
                        "training": 0,
                        "validation": 1,
                        "val": 1,
                        "valid": 1,
                        "test": 2,
                    }
                    combined_df["_sort_key"] = (
                        combined_df["Dataset"].astype(str).str.lower().map(dataset_order_map).fillna(3)
                    )
                    sort_cols = (
                        ["_sort_key", "_original_index"]
                        if "_original_index" in combined_df.columns
                        else ["_sort_key"]
                    )
                    combined_df = combined_df.sort_values(by=sort_cols).drop(
                        columns=["_sort_key", "_original_index"], errors="ignore"
                    )

                combined_df.to_csv(
                    alloy_dir / f"{dataset_name}_combined_predictions.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

        for child in alloy_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)


def create_global_exports(summary_df: pd.DataFrame, summary_root: Path) -> None:
    summary_df.to_csv(
        summary_root / "ALL_EXTRAPOLATION_MODEL_SUMMARY.csv",
        index=False,
        encoding="utf-8-sig",
    )

    best_rows = (
        summary_df.assign(
            final_test_r2=summary_df["final_test_r2"].fillna(-np.inf),
            closest_mean_test_r2=summary_df["closest_mean_test_r2"].fillna(-np.inf),
            representative_test_r2=summary_df["representative_test_r2"].fillna(-np.inf)
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
    best_rows.to_csv(
        summary_root / "ALL_BEST_EXTRAPOLATION_MODELS.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pivot_df = best_rows.pivot_table(
        index=["alloy_family", "dataset_name", "feature_mode"],
        columns="target",
        values="final_test_r2",
        aggfunc="first",
    )
    if not pivot_df.empty:
        pivot_df = pivot_df.reset_index()
        pivot_df.to_csv(
            summary_root / "BEST_MODEL_FINAL_TEST_R2_PIVOT.csv",
            index=False,
            encoding="utf-8-sig",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch summarize extrapolation experiment results."
    )
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

    summary_root = base_dir / "all_extrapolation_summary"
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
        case_rows = process_case(model_comparison_dir, summary_root, base_dir)
        if case_rows:
            all_rows.extend(case_rows)

    if not all_rows:
        print("No extrapolation metrics were collected.")
        return

    summary_df = pd.DataFrame(all_rows)
    create_global_exports(summary_df, summary_root)
    export_alloy_level_views(summary_df, summary_root)

    print("\n" + "=" * 80)
    print("Completed extrapolation summary")
    print("=" * 80)
    print(f"Cases processed: {summary_df['case_id'].nunique()}")
    print(f"Targets summarized: {summary_df[['case_id', 'target']].drop_duplicates().shape[0]}")
    print(f"Models summarized: {len(summary_df)}")
    print(f"Summary directory: {summary_root}")


if __name__ == "__main__":
    main()
