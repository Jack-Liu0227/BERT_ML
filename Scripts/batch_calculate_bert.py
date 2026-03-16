"""
Batch script to summarize BERT global-mean representative folds.

The script scans Optuna trial predictions for SciBERT, MatSciBERT, and
SteelBERT models, computes per-property global-mean R2 statistics, and exports
a unified summary layout with per-property model comparisons.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


plt.style.use("seaborn-v0_8-white")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 14
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["figure.dpi"] = 300


MODEL_ORDER = ["matscibert", "scibert", "steelbert"]
MODEL_DISPLAY = {
    "matscibert": "MatSciBERT",
    "scibert": "SciBERT",
    "steelbert": "SteelBERT",
}
SUMMARY_TABLES_DIRNAME = "00_summary_tables"
CASES_DIRNAME = "01_alloy_cases"
CSV_ENCODINGS = ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"]


def safe_name(text: str) -> str:
    return (
        text.replace("(", "")
        .replace(")", "")
        .replace("%", "pct")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


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


def read_prediction_csv(file_path: Path) -> Optional[pd.DataFrame]:
    header_columns: Optional[List[str]] = None
    header_encoding: Optional[str] = None
    last_error: Optional[Exception] = None

    for encoding in CSV_ENCODINGS:
        try:
            header_columns = pd.read_csv(file_path, nrows=0, encoding=encoding).columns.tolist()
            header_encoding = encoding
            break
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
            last_error = exc

    if header_columns is None:
        print(f"[WARN] Failed to read header from {file_path}: {last_error}")
        return None

    usecols = [
        column
        for column in header_columns
        if column == "Dataset" or column.endswith("_Actual") or column.endswith("_Predicted")
    ]
    if not usecols:
        usecols = header_columns

    ordered_encodings = [header_encoding] + [encoding for encoding in CSV_ENCODINGS if encoding != header_encoding]
    for encoding in ordered_encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding, usecols=usecols, low_memory=False)
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
            last_error = exc

        try:
            return pd.read_csv(
                file_path,
                encoding=encoding,
                usecols=usecols,
                engine="python",
                on_bad_lines="skip",
            )
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
            last_error = exc

    print(f"[WARN] Failed to parse {file_path}: {last_error}")
    return None


def rename_export_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy().rename(
        columns={
            "Alloy": "alloy_family",
            "Dataset_Name": "dataset_name",
            "Model": "model",
            "Model_Display": "model_display",
            "Property": "property",
            "Global_Mean_R2": "global_mean_r2",
            "Global_Std_R2": "global_std_r2",
            "Closest_Trial": "closest_trial",
            "Closest_Fold": "closest_fold",
            "Closest_R2": "closest_r2",
            "Representative_File": "global_mean_predictions_file",
            "Source_Model_Dir": "source_model_dir",
        }
    )
    if "feature_mode" not in renamed.columns:
        renamed["feature_mode"] = "bert_global_mean"
    if "case_id" not in renamed.columns:
        renamed["case_id"] = (
            renamed["alloy_family"].astype(str)
            + "__"
            + renamed["dataset_name"].astype(str)
            + "__"
            + renamed["property"].astype(str)
        )
    if "model_dir" not in renamed.columns and "source_model_dir" in renamed.columns:
        renamed["model_dir"] = renamed["source_model_dir"]
    if "case_path" not in renamed.columns and "model_dir" in renamed.columns:
        renamed["case_path"] = renamed["model_dir"]
    return renamed


def plot_property_model_comparison(case_df: pd.DataFrame, output_path: Path, case_label: str) -> None:
    if case_df.empty:
        return

    ordered_df = case_df.copy()
    ordered_df["Model"] = pd.Categorical(ordered_df["Model"], categories=MODEL_ORDER, ordered=True)
    ordered_df = ordered_df.sort_values("Model").dropna(subset=["Global_Mean_R2"])
    if ordered_df.empty:
        return

    x = np.arange(len(ordered_df))
    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    ax.bar(
        x,
        ordered_df["Global_Mean_R2"],
        yerr=ordered_df["Global_Std_R2"],
        width=0.55,
        capsize=4,
        color="#ffb703",
        edgecolor="black",
        linewidth=1.2,
        error_kw={"elinewidth": 1.5, "ecolor": "black"},
    )

    ax.set_xlabel("BERT models", fontweight="bold")
    ax.set_ylabel("R²", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY.get(name, name) for name in ordered_df["Model"]])
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis="both", which="both", direction="in")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_dataset_property_summary(dataset_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = dataset_df[["Model", "Property", "Global_Mean_R2", "Global_Std_R2"]].dropna(subset=["Global_Mean_R2"]).copy()
    if plot_df.empty:
        return

    properties = sorted(plot_df["Property"].unique())
    model_positions = np.arange(len(MODEL_ORDER))
    width = 0.8 / max(len(properties), 1)
    colors = ["#f4a698", "#9bbfe0", "#c8d5b9", "#d4a5d6", "#f6d186"]

    fig, ax = plt.subplots(figsize=(9.8, 7.8))
    for idx, prop in enumerate(properties):
        prop_df = plot_df[plot_df["Property"] == prop].copy()
        prop_df["Model"] = pd.Categorical(prop_df["Model"], categories=MODEL_ORDER, ordered=True)
        prop_df = prop_df.sort_values("Model").set_index("Model").reindex(MODEL_ORDER).reset_index()
        offset = (idx - (len(properties) - 1) / 2) * width
        ax.bar(
            model_positions + offset,
            prop_df["Global_Mean_R2"].tolist(),
            yerr=prop_df["Global_Std_R2"].fillna(0).tolist(),
            width=width,
            capsize=3,
            color=colors[idx % len(colors)],
            edgecolor="black",
            linewidth=1.1,
            error_kw={"elinewidth": 1.2, "ecolor": "black"},
            label=prop,
        )

    ax.set_xlabel("BERT models", fontweight="bold")
    ax.set_ylabel("R²", fontweight="bold")
    ax.set_xticks(model_positions)
    ax.set_xticklabels([MODEL_DISPLAY.get(name, name) for name in MODEL_ORDER])
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


def plot_diagonal_chart(file_path: Path, property_name: str, model_name: str, output_path: Path) -> None:
    try:
        df = read_prediction_csv(file_path)
    except Exception as exc:
        print(f"[WARN] Failed to read predictions file {file_path}: {exc}")
        return
    if df is None:
        return

    actual_col = f"{property_name}_Actual"
    pred_col = f"{property_name}_Predicted"
    if actual_col not in df.columns or pred_col not in df.columns:
        return

    dataset_config = {
        "Train": {"color": "#4CAF50", "marker": "o", "label": "Training Set", "alpha": 0.5, "s": 60},
        "Validation": {"color": "#2196F3", "marker": "s", "label": "Validation Set", "alpha": 0.6, "s": 70},
        "Test": {"color": "#FF5722", "marker": "^", "label": "Test Set", "alpha": 0.7, "s": 80},
    }
    dataset_aliases = {
        "Train": ["Train", "train", "Training", "training"],
        "Validation": ["Validation", "Valid", "Val", "validation", "valid", "val"],
        "Test": ["Test", "test", "Testing", "testing"],
    }

    plt.figure(figsize=(8, 8))
    has_dataset_col = "Dataset" in df.columns
    min_val = float("inf")
    max_val = float("-inf")
    metrics_text: List[str] = []
    plotted_any = False

    if has_dataset_col:
        for dataset_key, config in dataset_config.items():
            matching_rows = pd.Series([False] * len(df))
            for alias in dataset_aliases[dataset_key]:
                matching_rows |= df["Dataset"] == alias

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


def collect_model_rows(model_dir: Path, alloy_name: str, dataset_name: str) -> List[Dict]:
    optuna_trials_dir = model_dir / "predictions" / "optuna_trials"
    if not optuna_trials_dir.exists():
        return []

    model_type = model_dir.name
    trial_data = []
    trial_folders = sorted(
        [d for d in optuna_trials_dir.iterdir() if d.is_dir() and d.name.startswith("trial_")],
        key=lambda path: int(path.name.split("_")[1]),
    )

    for trial_dir in trial_folders:
        fold_dirs = sorted(
            [d for d in trial_dir.iterdir() if d.is_dir() and d.name.startswith("fold_")],
            key=lambda path: int(path.name.split("_")[1]),
        )
        for fold_dir in fold_dirs:
            prediction_file = fold_dir / "all_predictions.csv"
            if not prediction_file.exists():
                continue

            try:
                df_pred = read_prediction_csv(prediction_file)
            except Exception as exc:
                print(f"[WARN] Failed to read {prediction_file}: {exc}")
                continue
            if df_pred is None:
                continue

            if "Dataset" not in df_pred.columns:
                continue

            df_test = df_pred[df_pred["Dataset"].isin(["Test", "test", "Testing", "testing"])].copy()
            if df_test.empty:
                continue

            data_point = {
                "trial_id": trial_dir.name,
                "fold": int(fold_dir.name.split("_")[1]),
                "file_path": str(prediction_file),
            }
            actual_cols = [col for col in df_test.columns if col.endswith("_Actual")]
            for actual_col in actual_cols:
                prop = actual_col.replace("_Actual", "")
                pred_col = f"{prop}_Predicted"
                if pred_col not in df_test.columns:
                    continue
                valid_rows = df_test[[actual_col, pred_col]].dropna()
                if not valid_rows.empty:
                    data_point[prop] = r2_score(valid_rows[actual_col], valid_rows[pred_col])

            if len(data_point) > 3:
                trial_data.append(data_point)

    if not trial_data:
        return []

    all_df = pd.DataFrame(trial_data)
    properties = [col for col in all_df.columns if col not in ["trial_id", "fold", "file_path"]]
    rows: List[Dict] = []

    for prop in properties:
        target_mean = all_df[prop].mean()
        target_std = all_df[prop].std()
        all_df[f"{prop}_dist"] = (all_df[prop] - target_mean).abs()
        closest_row = all_df.sort_values(by=f"{prop}_dist").iloc[0]

        rows.append(
            {
                "Alloy": alloy_name,
                "Dataset_Name": dataset_name,
                "Model": model_type,
                "Model_Display": MODEL_DISPLAY.get(model_type, model_type),
                "Property": prop,
                "Global_Mean_R2": target_mean,
                "Global_Std_R2": target_std,
                "Closest_Trial": closest_row["trial_id"],
                "Closest_Fold": closest_row["fold"],
                "Closest_R2": closest_row[prop],
                "Representative_File": closest_row["file_path"],
                "Source_Model_Dir": str(model_dir),
            }
        )

    return rows


def build_best_rows(dataset_df: pd.DataFrame) -> pd.DataFrame:
    return (
        dataset_df.assign(
            Global_Mean_R2=dataset_df["Global_Mean_R2"].fillna(-np.inf),
            Closest_R2=dataset_df["Closest_R2"].fillna(-np.inf),
        )
        .sort_values(["Property", "Global_Mean_R2", "Closest_R2"], ascending=[True, False, False], na_position="last")
        .groupby("Property", group_keys=False)
        .head(1)
        .reset_index(drop=True)
    )


def export_case_views(summary_df: pd.DataFrame, summary_root: Path) -> None:
    cases_root = summary_root / CASES_DIRNAME

    for alloy_name, alloy_df in summary_df.groupby("Alloy"):
        for dataset_name, dataset_df in alloy_df.groupby("Dataset_Name"):
            dataset_dir = cases_root / alloy_name / dataset_name
            dataset_export_df = rename_export_columns(
                dataset_df.sort_values(["Property", "Model"]).reset_index(drop=True)
            )
            save_csv(dataset_export_df, dataset_dir / "dataset_model_summary.csv")
            plot_dataset_property_summary(dataset_df, dataset_dir / "comparisons" / "global_mean_r2_summary.png")

            best_rows = build_best_rows(dataset_df)
            save_csv(rename_export_columns(best_rows), dataset_dir / "best_models_summary.csv")

            for property_name, property_df in dataset_df.groupby("Property"):
                case_dir = dataset_dir / safe_name(str(property_name))
                comparisons_dir = case_dir / "comparisons"
                artifacts_dir = case_dir / "selected_model_artifacts"

                save_csv(
                    rename_export_columns(property_df.sort_values(["Model"]).reset_index(drop=True)),
                    case_dir / "case_model_summary.csv",
                )
                plot_property_model_comparison(
                    property_df,
                    comparisons_dir / "global_mean_r2_by_model.png",
                    f"{alloy_name} / {dataset_name} / {property_name}",
                )

                best_row = best_rows[best_rows["Property"] == property_name]
                if best_row.empty:
                    continue
                best_record = best_row.iloc[0]
                best_model = str(best_record["Model"])
                model_copy_dir = case_dir / "selected_model_source" / safe_name(best_model)

                save_csv(rename_export_columns(pd.DataFrame([best_record])), case_dir / "best_model_summary.csv")
                copy_if_exists(Path(best_record["Representative_File"]), artifacts_dir / "global_mean_predictions.csv")
                plot_diagonal_chart(
                    Path(best_record["Representative_File"]),
                    str(property_name),
                    str(best_record["Model_Display"]),
                    artifacts_dir / "global_mean_diagnostic.png",
                )
                copy_tree_if_exists(Path(best_record["Source_Model_Dir"]), model_copy_dir)



def create_global_exports(summary_df: pd.DataFrame, summary_root: Path) -> None:
    summary_tables_dir = summary_root / SUMMARY_TABLES_DIRNAME
    export_df = rename_export_columns(summary_df)
    save_csv(export_df, summary_tables_dir / "all_bert_extrapolation_model_summary.csv")

    best_rows = (
        summary_df.assign(
            Global_Mean_R2=summary_df["Global_Mean_R2"].fillna(-np.inf),
            Closest_R2=summary_df["Closest_R2"].fillna(-np.inf),
        )
        .sort_values(
            ["Alloy", "Dataset_Name", "Property", "Global_Mean_R2", "Closest_R2"],
            ascending=[True, True, True, False, False],
            na_position="last",
        )
        .groupby(["Alloy", "Dataset_Name", "Property"], group_keys=False)
        .head(1)
        .reset_index(drop=True)
    )
    best_export_df = rename_export_columns(best_rows)
    save_csv(best_export_df, summary_tables_dir / "all_best_bert_extrapolation_models.csv")

    pivot_df = best_export_df.pivot_table(
        index=["alloy_family", "dataset_name", "feature_mode"],
        columns="property",
        values="global_mean_r2",
        aggfunc="first",
    )
    if not pivot_df.empty:
        save_csv(pivot_df.reset_index(), summary_tables_dir / "best_model_global_mean_r2_pivot.csv")


def iter_model_dirs(base_path: Path) -> List[tuple[str, str, Path]]:
    model_entries: List[tuple[str, str, Path]] = []
    skip_dirs = {"all_alloys_best_models_summary", "all_bert_global_mean_summary"}

    for alloy_dir in base_path.iterdir():
        if not alloy_dir.is_dir() or alloy_dir.name in skip_dirs:
            continue
        for dataset_dir in alloy_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            for model_dir in dataset_dir.iterdir():
                if model_dir.is_dir() and model_dir.name in MODEL_ORDER:
                    model_entries.append((alloy_dir.name, dataset_dir.name, model_dir))

    return sorted(model_entries, key=lambda item: (item[0], item[1], item[2].name))


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch process BERT model global means.")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="output/new_results_withuncertainty",
        help="Base directory of results",
    )
    args = parser.parse_args()

    base_path = Path(args.base_dir)
    summary_root = base_path / "all_bert_global_mean_summary"
    if summary_root.exists():
        shutil.rmtree(summary_root)
    summary_root.mkdir(parents=True, exist_ok=True)

    print(f"Starting batch BERT global-mean analysis in {base_path}...")

    all_rows: List[Dict] = []
    model_entries = iter_model_dirs(base_path)
    for alloy_name, dataset_name, model_dir in model_entries:
        print(f"Processing {alloy_name}/{dataset_name}/{model_dir.name}...")
        all_rows.extend(collect_model_rows(model_dir, alloy_name, dataset_name))

    if not all_rows:
        print("No BERT trial results found.")
        return

    summary_df = pd.DataFrame(all_rows)
    create_global_exports(summary_df, summary_root)
    export_case_views(summary_df, summary_root)

    print(f"\nCompleted! Summary saved to {summary_root}")


if __name__ == "__main__":
    main()
