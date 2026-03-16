"""
Batch summary script for TabPFN extrapolation experiment results.

The script scans output/extrapolation_results_tabpfn, reads per-case metrics,
writes summary CSV files, keeps the overview HTML plots, and exports a unified
case layout with train/test comparisons plus copied model folders.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
import plotly.express as px
import plotly.io as pio


pio.templates.default = "plotly_white"
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


METRICS = ["mae", "rmse", "r2"]
SUMMARY_TABLES_DIRNAME = "00_summary_tables"
CASES_DIRNAME = "01_alloy_cases"


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
        raise FileNotFoundError(f"Missing JSON file: {path}")
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


def find_metric_files(base_dir: Path) -> List[Path]:
    return sorted(base_dir.rglob("metrics_summary.json"))


def extract_case_row(metrics_path: Path, base_dir: Path) -> Dict | None:
    try:
        payload = load_json(metrics_path)
    except Exception as exc:
        print(f"[WARN] Failed to read {metrics_path}: {exc}")
        return None

    try:
        relative_parts = metrics_path.relative_to(base_dir).parts
        alloy_type = relative_parts[0]
        dataset_name = relative_parts[1]
        target_col = relative_parts[2]
    except Exception:
        print(f"[WARN] Unexpected metrics path layout: {metrics_path}")
        return None

    train_metrics = payload.get("train", {})
    test_metrics = payload.get("extrapolation_test", {})
    split_summary = payload.get("split_summary", {})
    model_dir = metrics_path.parent.parent

    required = [
        train_metrics.get("mae"),
        train_metrics.get("rmse"),
        train_metrics.get("r2"),
        test_metrics.get("mae"),
        test_metrics.get("rmse"),
        test_metrics.get("r2"),
    ]
    if any(value is None for value in required):
        print(f"[WARN] Incomplete metrics payload, skipped: {metrics_path}")
        return None

    return {
        "alloy_type": payload.get("alloy_type", alloy_type),
        "dataset_name": dataset_name,
        "target_col": payload.get("target_col", target_col),
        "model": "TabPFN",
        "raw_row_count": payload.get("raw_row_count"),
        "train_n_samples": train_metrics.get("n_samples"),
        "train_mae": train_metrics.get("mae"),
        "train_rmse": train_metrics.get("rmse"),
        "train_r2": train_metrics.get("r2"),
        "test_n_samples": test_metrics.get("n_samples"),
        "test_mae": test_metrics.get("mae"),
        "test_rmse": test_metrics.get("rmse"),
        "test_r2": test_metrics.get("r2"),
        "train_target_min": split_summary.get("train_target_min"),
        "train_target_max": split_summary.get("train_target_max"),
        "test_target_min": split_summary.get("test_target_min"),
        "test_target_max": split_summary.get("test_target_max"),
        "model_dir": str(model_dir),
        "case_path": str(model_dir),
        "metrics_path": str(metrics_path),
        "predictions_file": str(model_dir / "predictions" / "all_predictions.csv"),
        "plot_file": str(model_dir / "plots" / f"{safe_name(target_col)}_predictions.png"),
        "split_summary_file": str(model_dir / "split_data" / "split_summary.json"),
    }


def rows_to_dataframe(rows: List[Dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["target_col", "alloy_type", "dataset_name"]).reset_index(drop=True)


def build_split_df(summary_df: pd.DataFrame, split: str) -> pd.DataFrame:
    prefix = "train" if split == "train" else "test"
    split_df = pd.DataFrame(
        {
            "alloy_type": summary_df["alloy_type"],
            "dataset_name": summary_df["dataset_name"],
            "target_col": summary_df["target_col"],
            "model": summary_df["model"],
            "n_samples": summary_df[f"{prefix}_n_samples"],
            "mae": summary_df[f"{prefix}_mae"],
            "rmse": summary_df[f"{prefix}_rmse"],
            "r2": summary_df[f"{prefix}_r2"],
            "raw_row_count": summary_df["raw_row_count"],
            "train_target_min": summary_df["train_target_min"],
            "train_target_max": summary_df["train_target_max"],
            "test_target_min": summary_df["test_target_min"],
            "test_target_max": summary_df["test_target_max"],
            "case_path": summary_df["case_path"],
            "metrics_path": summary_df["metrics_path"],
            "model_dir": summary_df["model_dir"],
            "predictions_file": summary_df["predictions_file"],
            "plot_file": summary_df["plot_file"],
            "split_summary_file": summary_df["split_summary_file"],
        }
    )
    return split_df.sort_values(["target_col", "alloy_type", "dataset_name"]).reset_index(drop=True)


def prepare_plot_df(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    ascending = metric in {"mae", "rmse"}
    plot_df = df.copy()
    plot_df["label"] = plot_df["alloy_type"] + " - " + plot_df["dataset_name"]
    return plot_df.sort_values(metric, ascending=ascending).reset_index(drop=True)


def build_overview_figure(df: pd.DataFrame, split: str, metric: str):
    if df.empty:
        return None

    overview_df = prepare_plot_df(df, metric)
    overview_df["label"] = (
        overview_df["target_col"] + " | " + overview_df["alloy_type"] + " - " + overview_df["dataset_name"]
    )

    fig = px.bar(
        overview_df,
        x=metric,
        y="label",
        color="target_col",
        orientation="h",
        hover_name="label",
        hover_data={
            metric: ":.6f",
            "target_col": True,
            "alloy_type": True,
            "dataset_name": True,
            "n_samples": True,
            "raw_row_count": True,
            "case_path": True,
            "label": False,
        },
        category_orders={"label": overview_df["label"].tolist()},
        title=f"TabPFN Extrapolation | All Targets | {split.capitalize()} | {metric.upper()}",
    )
    fig.update_traces(marker_line_color="black", marker_line_width=1.0)
    fig.update_layout(
        font=dict(family="Times New Roman", size=14),
        xaxis_title=metric.upper(),
        yaxis_title="Target | Alloy - Dataset",
        legend_title_text="Target",
        height=max(500, 140 * len(overview_df)),
        width=1400,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.15)", zeroline=(metric == "r2"), zerolinecolor="black")
    fig.update_yaxes(showgrid=False, automargin=True)
    return fig


def plot_overview(df: pd.DataFrame, split: str, metric: str, output_path: Path):
    fig = build_overview_figure(df, split, metric)
    if fig is None:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")
    return fig


def write_overview_dashboard(figures: List[tuple[str, str, object]], output_path: Path) -> None:
    sections: List[str] = []
    for split_name, metric, fig in figures:
        chart_html = pio.to_html(fig, full_html=False, include_plotlyjs=False)
        sections.append(
            f"""
            <section class="card">
              <h2>{split_name.capitalize()} | {metric.upper()}</h2>
              <div class="chart">{chart_html}</div>
            </section>
            """
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TabPFN Extrapolation Overview Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      margin: 0;
      font-family: "Times New Roman", serif;
      background: #f6f4ef;
      color: #1f2933;
    }}
    .page {{
      max-width: 1600px;
      margin: 0 auto;
      padding: 24px;
    }}
    .hero {{
      background: linear-gradient(135deg, #fffaf0, #e8f1f8);
      border: 1px solid #d9e2ec;
      border-radius: 18px;
      padding: 24px 28px;
      margin-bottom: 24px;
      box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    }}
    .hero h1 {{
      margin: 0 0 8px 0;
      font-size: 32px;
    }}
    .hero p {{
      margin: 0;
      font-size: 16px;
      line-height: 1.6;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(720px, 1fr));
      gap: 20px;
    }}
    .card {{
      background: #ffffff;
      border: 1px solid #d9e2ec;
      border-radius: 18px;
      padding: 16px;
      box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
    }}
    .card h2 {{
      margin: 0 0 12px 0;
      font-size: 22px;
    }}
    .chart > div {{
      width: 100%;
    }}
    @media (max-width: 900px) {{
      .page {{
        padding: 14px;
      }}
      .grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <header class="hero">
      <h1>TabPFN Extrapolation Overview</h1>
      <p>Interactive dashboard for all-target train/test MAE, RMSE, and R2 comparisons. Hover to inspect each dataset case and use the legend to filter targets.</p>
    </header>
    <main class="grid">
      {''.join(sections)}
    </main>
  </div>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def generate_overview_plots(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path) -> None:
    overview_dir = output_dir / "plots" / "overview"
    figures: List[tuple[str, str, object]] = []
    for split_name, split_df in [("train", train_df), ("test", test_df)]:
        for metric in METRICS:
            fig = plot_overview(
                df=split_df,
                split=split_name,
                metric=metric,
                output_path=overview_dir / f"all_targets_{split_name}_{metric}.html",
            )
            if fig is not None:
                figures.append((split_name, metric, fig))
    if figures:
        write_overview_dashboard(figures, overview_dir / "overview_dashboard.html")


def clean_plot_outputs(output_dir: Path) -> None:
    plots_dir = output_dir / "plots"
    if plots_dir.exists():
        for path in plots_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".png", ".html"}:
                path.unlink()


def plot_two_mode_comparison(
    row: pd.Series,
    output_path: Path,
    metric_key: str,
    ylabel: str,
) -> None:
    values = [row[f"train_{metric_key}"], row[f"test_{metric_key}"]]
    labels = ["Train", "Test"]
    colors = ["#8ecae6", "#ffb703"]

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.bar(labels, values, color=colors, edgecolor="black", linewidth=1.2, width=0.55)
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_title(f"{row['alloy_type']} / {row['dataset_name']} / {row['target_col']}", fontweight="bold", pad=14)
    if ylabel == "R2":
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax.tick_params(axis="both", which="both", direction="in")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_combined_predictions(summary_df: pd.DataFrame) -> pd.DataFrame | None:
    combined_df = None

    for _, row in summary_df.iterrows():
        predictions_path = Path(row["predictions_file"])
        if not predictions_path.exists():
            continue

        temp_df = pd.read_csv(predictions_path)
        if "ID" not in temp_df.columns:
            temp_df = temp_df.copy()
            temp_df["ID"] = range(len(temp_df))

        cols_to_keep = ["ID"]
        if "Dataset" in temp_df.columns:
            cols_to_keep.append("Dataset")
        actual_col = f"{row['target_col']}_Actual"
        pred_col = f"{row['target_col']}_Predicted"
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


def export_case_views(summary_df: pd.DataFrame, output_dir: Path) -> None:
    cases_root = output_dir / CASES_DIRNAME

    for alloy_name, alloy_df in summary_df.groupby("alloy_type"):
        for dataset_name, dataset_df in alloy_df.groupby("dataset_name"):
            dataset_dir = cases_root / alloy_name / dataset_name
            save_csv(
                dataset_df.sort_values(["target_col"]).reset_index(drop=True),
                dataset_dir / "dataset_tabpfn_summary.csv",
            )

            for _, row in dataset_df.sort_values("target_col").iterrows():
                safe_target = safe_name(str(row["target_col"]))
                case_dir = dataset_dir / safe_target
                comparisons_dir = case_dir / "comparisons"
                artifacts_dir = case_dir / "selected_model_artifacts"
                model_copy_dir = case_dir / "selected_model_source" / "TabPFN"

                save_csv(pd.DataFrame([row]), case_dir / "case_model_summary.csv")
                plot_two_mode_comparison(row, comparisons_dir / "two_modes_r2_comparison.png", "r2", "R2")
                plot_two_mode_comparison(row, comparisons_dir / "two_modes_mae_comparison.png", "mae", "MAE")

                copy_if_exists(Path(row["predictions_file"]), artifacts_dir / "all_predictions.csv")
                copy_if_exists(Path(row["plot_file"]), artifacts_dir / "prediction_plot.png")
                copy_if_exists(Path(row["split_summary_file"]), artifacts_dir / "split_summary.json")
                copy_tree_if_exists(Path(row["model_dir"]), model_copy_dir)

            combined_df = build_combined_predictions(dataset_df)
            if combined_df is not None:
                save_csv(combined_df, dataset_dir / "combined_predictions.csv")


def summarize(base_dir: Path, output_dir: Path, skip_plots: bool) -> None:
    metric_files = find_metric_files(base_dir)
    print(f"[INFO] Found {len(metric_files)} metrics_summary.json files under {base_dir}")

    rows: List[Dict] = []
    for metrics_path in metric_files:
        row = extract_case_row(metrics_path, base_dir)
        if row is not None:
            rows.append(row)

    summary_df = rows_to_dataframe(rows)
    if summary_df.empty:
        print("[WARN] No valid TabPFN extrapolation metrics were found.")
        return

    train_df = build_split_df(summary_df, "train")
    test_df = build_split_df(summary_df, "test")

    summary_tables_dir = output_dir / SUMMARY_TABLES_DIRNAME
    save_csv(train_df, summary_tables_dir / "tabpfn_extrapolation_train_summary.csv")
    save_csv(test_df, summary_tables_dir / "tabpfn_extrapolation_test_summary.csv")

    by_target_dir = summary_tables_dir / "by_target"
    for target_col, group_df in train_df.groupby("target_col", sort=True):
        safe_target = safe_name(target_col)
        save_csv(group_df, by_target_dir / f"{safe_target}_train_summary.csv")
    for target_col, group_df in test_df.groupby("target_col", sort=True):
        safe_target = safe_name(target_col)
        save_csv(group_df, by_target_dir / f"{safe_target}_test_summary.csv")

    export_case_views(summary_df, output_dir)

    if not skip_plots:
        clean_plot_outputs(output_dir)
        generate_overview_plots(train_df, test_df, output_dir)

    print(f"[INFO] Parsed {len(summary_df)} valid cases")
    print(f"[INFO] Train summary saved to {summary_tables_dir / 'tabpfn_extrapolation_train_summary.csv'}")
    print(f"[INFO] Test summary saved to {summary_tables_dir / 'tabpfn_extrapolation_test_summary.csv'}")
    if not skip_plots:
        print(f"[INFO] Interactive overview directory: {output_dir / 'plots' / 'overview'}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize TabPFN extrapolation results and generate plots")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("output/extrapolation_results_tabpfn"),
        help="Base directory containing per-case TabPFN extrapolation outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for summary CSV files and plots. Defaults to <base-dir>/all_tabpfn_extrapolation_summary.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Only generate summary CSV files without overview HTML plots.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    base_dir = args.base_dir
    output_dir = args.output_dir or (base_dir / "all_tabpfn_extrapolation_summary")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summarize(base_dir=base_dir, output_dir=output_dir, skip_plots=args.skip_plots)


if __name__ == "__main__":
    main()
