"""
Batch summary script for BERT extrapolation experiment results.

The script scans output/extrapolation_results for SciBERT, MatSciBERT, and
SteelBERT result folders, collects final / closest-to-mean / global-mean
metrics, writes summary CSV files, and exports per-alloy best-model views.
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
                tail = f"_{candidate}"
                if metric_key.endswith(tail):
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


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def resolve_existing_path(*candidates: Path) -> str:
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


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
    ordered_df["model"] = pd.Categorical(
        ordered_df["model"], categories=MODEL_ORDER, ordered=True
    )
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

    ax.set_xlabel("BERT Models", fontweight="bold")
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
    required = [final_col, closest_mean_col, global_mean_col]
    if case_df.empty or any(col not in case_df.columns for col in required):
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

    ax.set_xlabel("BERT Models", fontweight="bold")
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
    representative_metrics = flatten_eval_summary(
        load_json(
            model_dir
            / "closest_to_global_mean_predictions"
            / "closest_to_global_mean_predictions_closest_to_global_mean_predictions_evaluation_evaluation_summary.json"
        ),
        "representative",
    )
    closest_mean_metrics = flatten_eval_summary(
        load_json(
            model_dir
            / "closest_to_mean_predictions"
            / "closest_to_mean_predictions_closest_to_mean_predictions_evaluation_evaluation_summary.json"
        ),
        "closest_mean",
    )
    representative_info = extract_fold_info(
        load_json(model_dir / "closest_to_global_mean_predictions" / "closest_to_mean_fold_info.json"),
        "representative",
    )
    closest_mean_info = extract_fold_info(
        load_json(model_dir / "closest_to_mean_predictions" / "closest_to_mean_fold_info.json"),
        "closest_mean",
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
        "final_predictions_file": resolve_existing_path(
            model_dir / "predictions" / "best_model_all_predictions.csv"
        ),
        "representative_predictions_file": resolve_existing_path(
            model_dir
            / "closest_to_global_mean_predictions"
            / "predictions"
            / "closest_to_global_mean_predictions_all_predictions.csv",
            model_dir / "closest_to_global_mean_predictions" / "all_predictions.csv",
        ),
        "closest_mean_predictions_file": resolve_existing_path(
            model_dir
            / "closest_to_mean_predictions"
            / "predictions"
            / "closest_to_mean_predictions_all_predictions.csv",
            model_dir / "closest_to_mean_predictions" / "all_predictions.csv",
        ),
        "final_plot_file": resolve_existing_path(
            model_dir / "plots" / "best_model_all_sets_comparison.png"
        ),
        "representative_plot_file": resolve_existing_path(
            model_dir
            / "closest_to_global_mean_predictions"
            / "plots"
            / "closest_to_global_mean_predictions_all_sets_comparison.png"
        ),
        "closest_mean_plot_file": resolve_existing_path(
            model_dir
            / "closest_to_mean_predictions"
            / "plots"
            / "closest_to_mean_predictions_all_sets_comparison.png"
        ),
    }
    row.update(final_metrics)
    row.update(representative_metrics)
    row.update(closest_mean_metrics)
    row.update(representative_info)
    row.update(closest_mean_info)

    if not any(key.startswith("final_") for key in row):
        print(f"[WARN] Missing final metrics: {model_dir}")
        return None

    return row


def export_alloy_level_views(summary_df: pd.DataFrame, summary_root: Path) -> None:
    for alloy_name, alloy_df in summary_df.groupby("alloy_family"):
        alloy_dir = summary_root / alloy_name
        alloy_dir.mkdir(parents=True, exist_ok=True)

        for dataset_name, dataset_df in alloy_df.groupby("dataset_name"):
            dataset_df.to_csv(
                alloy_dir / f"{dataset_name}_bert_extrapolation_model_summary.csv",
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
                .groupby("target", group_keys=False)
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
                case_label = f"{alloy_name} / {dataset_name} / {target}"

                plot_metric_comparison(
                    target_df,
                    alloy_dir / f"{dataset_name}_bert_comparison_{safe_target}_FinalTest_R2.png",
                    case_label,
                    "final_test_r2",
                    "R2",
                    "Final extrapolation test R2",
                )
                plot_metric_comparison(
                    target_df,
                    alloy_dir / f"{dataset_name}_bert_comparison_{safe_target}_FinalTest_MAE.png",
                    case_label,
                    "final_test_mae",
                    "MAE",
                    "Final extrapolation test MAE",
                )
                plot_metric_comparison(
                    target_df,
                    alloy_dir / f"{dataset_name}_bert_comparison_{safe_target}_BestTrialMean_R2.png",
                    case_label,
                    "closest_mean_test_r2",
                    "R2",
                    "Best-trial representative R2",
                )
                plot_metric_comparison(
                    target_df,
                    alloy_dir / f"{dataset_name}_bert_comparison_{safe_target}_GlobalMean_R2.png",
                    case_label,
                    "representative_test_r2",
                    "R2",
                    "Global representative R2",
                )
                plot_three_metric_comparison(
                    target_df,
                    alloy_dir / f"{dataset_name}_bert_comparison_{safe_target}_ThreeModes_R2.png",
                    case_label,
                    "R2",
                    "final_test_r2",
                    "closest_mean_test_r2",
                    "representative_test_r2",
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
                    alloy_dir / f"{dataset_name}_best_bert_model_{best_model}_{safe_target}.csv",
                )
                copy_if_exists(
                    Path(row["final_predictions_file"]),
                    alloy_dir / f"{dataset_name}_final_bert_evaluation_{best_model}_{safe_target}.csv",
                )
                copy_if_exists(
                    Path(row["closest_mean_plot_file"]),
                    alloy_dir / f"{dataset_name}_{best_model}_bert_best_model_diagonal_{safe_target}.png",
                )
                copy_if_exists(
                    Path(row["representative_plot_file"]),
                    alloy_dir / f"{dataset_name}_{best_model}_bert_mean_model_diagonal_{safe_target}.png",
                )
                copy_if_exists(
                    Path(row["final_plot_file"]),
                    alloy_dir / f"{dataset_name}_{best_model}_bert_final_model_evaluation_{safe_target}.png",
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
                    alloy_dir / f"{dataset_name}_best_bert_models_summary.csv",
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
                    combined_df = combined_df.sort_values(
                        by=["_sort_key", "_original_index"]
                    ).drop(columns=["_sort_key", "_original_index"], errors="ignore")

                combined_df.to_csv(
                    alloy_dir / f"{dataset_name}_bert_combined_predictions.csv",
                    index=False,
                    encoding="utf-8-sig",
                )


def create_global_exports(summary_df: pd.DataFrame, summary_root: Path) -> None:
    summary_df.to_csv(
        summary_root / "ALL_BERT_EXTRAPOLATION_MODEL_SUMMARY.csv",
        index=False,
        encoding="utf-8-sig",
    )

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
    best_rows.to_csv(
        summary_root / "ALL_BEST_BERT_EXTRAPOLATION_MODELS.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch summarize BERT extrapolation experiment results."
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
    export_alloy_level_views(summary_df, summary_root)

    print("\n" + "=" * 80)
    print("Completed BERT extrapolation summary")
    print("=" * 80)
    print(f"Cases processed: {summary_df['case_id'].nunique()}")
    print(f"Models summarized: {len(summary_df)}")
    print(f"Summary directory: {summary_root}")


if __name__ == "__main__":
    main()
