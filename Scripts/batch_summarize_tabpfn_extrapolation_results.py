"""
Batch summary script for TabPFN extrapolation experiment results.

The script scans output/extrapolation_results_tabpfn, reads per-case metrics,
writes compact summary CSV files, and generates comparison plots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "plotly_white"

METRICS = ["mae", "rmse", "r2"]


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
        "case_path": str(metrics_path.parent.parent),
        "metrics_path": str(metrics_path),
    }


def rows_to_dataframe(rows: List[Dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values(["target_col", "alloy_type", "dataset_name"]).reset_index(drop=True)


def build_split_df(summary_df: pd.DataFrame, split: str) -> pd.DataFrame:
    prefix = "train" if split == "train" else "test"
    split_df = pd.DataFrame(
        {
            "alloy_type": summary_df["alloy_type"],
            "dataset_name": summary_df["dataset_name"],
            "target_col": summary_df["target_col"],
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
        }
    )
    return split_df.sort_values(["target_col", "alloy_type", "dataset_name"]).reset_index(drop=True)


def save_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def prepare_plot_df(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    ascending = metric in {"mae", "rmse"}
    plot_df = df.copy()
    plot_df["label"] = plot_df["alloy_type"] + " - " + plot_df["dataset_name"]
    return plot_df.sort_values(metric, ascending=ascending).reset_index(drop=True)


def build_overview_figure(df: pd.DataFrame, split: str, metric: str):
    if df.empty:
        return None

    overview_df = prepare_plot_df(df, metric)
    overview_df["label"] = overview_df["target_col"] + " | " + overview_df["alloy_type"] + " - " + overview_df["dataset_name"]
    overview_df["hover_text"] = (
        "Target: " + overview_df["target_col"]
        + "<br>Alloy: " + overview_df["alloy_type"]
        + "<br>Dataset: " + overview_df["dataset_name"]
        + "<br>Samples: " + overview_df["n_samples"].astype(str)
        + "<br>Raw rows: " + overview_df["raw_row_count"].astype(str)
        + "<br>Case: " + overview_df["case_path"]
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
            "hover_text": False,
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


def generate_by_target_outputs(
    split_df: pd.DataFrame,
    split: str,
    output_dir: Path,
) -> Dict[str, int]:
    by_target_dir = output_dir / "by_target"
    counts: Dict[str, int] = {}

    for target_col, group_df in split_df.groupby("target_col", sort=True):
        safe_target = safe_name(target_col)
        counts[target_col] = len(group_df)
        save_csv(group_df, by_target_dir / f"{safe_target}_{split}_summary.csv")

    return counts


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


def clean_plot_outputs(output_dir: Path) -> None:
    plots_dir = output_dir / "plots"
    by_target_plot_dir = plots_dir / "by_target"
    if by_target_plot_dir.exists():
        shutil.rmtree(by_target_plot_dir)

    overview_dir = plots_dir / "overview"
    if overview_dir.exists():
        for path in overview_dir.iterdir():
            if path.is_file() and path.suffix.lower() in {".png", ".html"}:
                path.unlink()


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

    save_csv(train_df, output_dir / "TABPFN_EXTRAPOLATION_TRAIN_SUMMARY.csv")
    save_csv(test_df, output_dir / "TABPFN_EXTRAPOLATION_TEST_SUMMARY.csv")

    train_counts = generate_by_target_outputs(train_df, "train", output_dir)
    test_counts = generate_by_target_outputs(test_df, "test", output_dir)

    if not skip_plots:
        clean_plot_outputs(output_dir)
        generate_overview_plots(train_df, test_df, output_dir)

    print(f"[INFO] Parsed {len(summary_df)} valid cases")
    for target_col in sorted(set(train_counts) | set(test_counts)):
        print(
            f"[INFO] Target {target_col}: "
            f"train_cases={train_counts.get(target_col, 0)}, "
            f"test_cases={test_counts.get(target_col, 0)}"
        )
    print(f"[INFO] Train summary saved to {output_dir / 'TABPFN_EXTRAPOLATION_TRAIN_SUMMARY.csv'}")
    print(f"[INFO] Test summary saved to {output_dir / 'TABPFN_EXTRAPOLATION_TEST_SUMMARY.csv'}")
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
        help="Only generate summary CSV files without plot PNGs.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    base_dir = args.base_dir
    output_dir = args.output_dir or (base_dir / "all_tabpfn_extrapolation_summary")
    summarize(base_dir=base_dir, output_dir=output_dir, skip_plots=args.skip_plots)


if __name__ == "__main__":
    main()
# python Scripts/batch_summarize_tabpfn_extrapolation_results.py --base-dir "output/extrapolation_results_tabpfn"
