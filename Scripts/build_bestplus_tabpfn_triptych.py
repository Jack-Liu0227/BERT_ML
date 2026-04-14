from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


METHOD_ORDER = [
    "Extrapolation",
    "LOCO",
    "SparseXcluster",
    "SparseXsingle",
    "SparseYcluster",
    "SparseYsingle",
]
SERIES_ORDER = [
    "BERT-best",
    "Traditional-best",
    "TabPFN-2.5-Plus-Numeric",
    "TabPFN-2.5-Plus-Text",
]
PALETTE = {
    "BERT-best": "#1b9e77",
    "Traditional-best": "#7570b3",
    "TabPFN-2.5-Plus-Numeric": "#d95f02",
    "TabPFN-2.5-Plus-Text": "#e7298a",
}
TASK_KEYS = ["alloy_family", "dataset_name", "property"]
CASE_KEYS = TASK_KEYS + ["ood_method"]


def safe_name(text: str) -> str:
    return (
        str(text)
        .replace("(", "")
        .replace(")", "")
        .replace("%", "pct")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "")
    )


def _coalesce_column(
    df: pd.DataFrame,
    target: str,
    candidates: list[str],
    default: object = pd.NA,
) -> None:
    if target in df.columns:
        series = df[target]
    else:
        series = pd.Series(default, index=df.index, dtype="object")

    for candidate in candidates:
        if candidate not in df.columns:
            continue
        series = series.where(series.notna(), df[candidate])

    df[target] = series


def canonicalize_summary_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    working_df = df.copy()
    if "model_family" not in working_df.columns:
        working_df["model_family"] = pd.NA

    family_series = working_df["model_family"].astype(str)
    tabpfn_mask = family_series.eq("TabPFN")

    for column in TASK_KEYS + ["ood_method", "model", "model_family"]:
        if column not in working_df.columns:
            working_df[column] = pd.NA

    _coalesce_column(working_df, "summary_test_r2", ["final_test_r2", "test_r2"])
    _coalesce_column(working_df, "summary_test_r2_std", [], default=pd.NA)
    _coalesce_column(working_df, "summary_test_mae", ["final_test_mae", "test_mae"])
    _coalesce_column(working_df, "summary_test_mae_std", [], default=pd.NA)
    _coalesce_column(working_df, "summary_test_rmse", ["final_test_rmse", "test_rmse"])
    _coalesce_column(working_df, "summary_test_rmse_std", [], default=pd.NA)
    _coalesce_column(working_df, "trial_count", [], default=pd.NA)
    _coalesce_column(working_df, "fold_count", [], default=pd.NA)
    _coalesce_column(working_df, "representative_selection_mode", ["selection_mode"], default=pd.NA)
    _coalesce_column(working_df, "representative_trial_id", ["selected_trial_id"])
    _coalesce_column(working_df, "representative_fold", ["selected_fold"])
    _coalesce_column(working_df, "representative_test_r2", ["test_r2", "final_test_r2"])
    _coalesce_column(working_df, "representative_test_mae", ["test_mae", "final_test_mae"])
    _coalesce_column(working_df, "representative_test_rmse", ["test_rmse", "final_test_rmse"])
    _coalesce_column(working_df, "representative_predictions_file", ["predictions_file"])
    _coalesce_column(working_df, "representative_plot_file", ["plot_file"])

    for column in [
        "summary_test_r2",
        "summary_test_r2_std",
        "summary_test_mae",
        "summary_test_mae_std",
        "summary_test_rmse",
        "summary_test_rmse_std",
        "trial_count",
        "fold_count",
        "representative_fold",
        "representative_test_r2",
        "representative_test_mae",
        "representative_test_rmse",
    ]:
        working_df[column] = pd.to_numeric(working_df[column], errors="coerce")

    for std_col in ["summary_test_r2_std", "summary_test_mae_std", "summary_test_rmse_std"]:
        working_df.loc[tabpfn_mask & working_df[std_col].isna(), std_col] = 0.0
    for count_col in ["trial_count", "fold_count"]:
        working_df.loc[tabpfn_mask & working_df[count_col].isna(), count_col] = 1
    working_df.loc[tabpfn_mask & working_df["representative_selection_mode"].isna(), "representative_selection_mode"] = "single_run"
    return working_df


def select_best_family_rows(df: pd.DataFrame, family_name: str, aggregate_label: str) -> pd.DataFrame:
    family_df = df[df["model_family"] == family_name].copy()
    if family_df.empty:
        return family_df

    family_df["_sort_r2"] = pd.to_numeric(family_df["summary_test_r2"], errors="coerce").fillna(-np.inf)
    family_df["_sort_r2_std"] = pd.to_numeric(family_df["summary_test_r2_std"], errors="coerce").fillna(np.inf)
    family_df["_sort_mae"] = pd.to_numeric(family_df["summary_test_mae"], errors="coerce").fillna(np.inf)
    family_df = family_df.sort_values(
        CASE_KEYS + ["_sort_r2", "_sort_r2_std", "_sort_mae", "model"],
        ascending=[True, True, True, True, False, True, True, True],
        kind="mergesort",
        na_position="last",
    )
    selected = family_df.groupby(CASE_KEYS, as_index=False, observed=True).head(1).copy()
    selected = selected.drop(columns=["_sort_r2", "_sort_r2_std", "_sort_mae"])
    selected["aggregate_label"] = aggregate_label
    return selected


def build_selected_rows(summary_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    if df.empty:
        return df

    df = canonicalize_summary_schema(df)

    bert_best = select_best_family_rows(df, "BERT", "BERT-best")
    traditional_best = select_best_family_rows(df, "Traditional", "Traditional-best")

    tab_df = df[df["model_family"] == "TabPFN"].copy()
    if not tab_df.empty:
        tab_df["aggregate_label"] = tab_df["model"]

    selected = pd.concat([bert_best, traditional_best, tab_df], ignore_index=True)
    if selected.empty:
        return selected

    selected["ood_method"] = pd.Categorical(selected["ood_method"], categories=METHOD_ORDER, ordered=True)
    selected["aggregate_label"] = pd.Categorical(
        selected["aggregate_label"], categories=SERIES_ORDER, ordered=True
    )
    selected["is_negative_r2"] = pd.to_numeric(selected["summary_test_r2"], errors="coerce") < 0
    selected["gap_to_best_among_4"] = (
        selected.groupby(CASE_KEYS, observed=True)["summary_test_r2"].transform("max") - selected["summary_test_r2"]
    )
    selected["is_win"] = selected["gap_to_best_among_4"].eq(0)
    return selected.sort_values(CASE_KEYS + ["aggregate_label"]).reset_index(drop=True)


def summarise_task(task_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        task_df.groupby(["ood_method", "aggregate_label"], observed=True)
        .agg(
            median_summary_test_r2=("summary_test_r2", "median"),
            q1_summary_test_r2=("summary_test_r2", lambda s: s.quantile(0.25)),
            q3_summary_test_r2=("summary_test_r2", lambda s: s.quantile(0.75)),
            win_rate=("is_win", "mean"),
            negative_r2_rate=("is_negative_r2", "mean"),
            representative_model=("model", lambda s: s.iloc[0]),
            representative_summary_test_r2=("summary_test_r2", lambda s: s.iloc[0]),
            representative_summary_test_r2_std=("summary_test_r2_std", lambda s: s.iloc[0]),
            representative_summary_test_mae=("summary_test_mae", lambda s: s.iloc[0]),
            n=("summary_test_r2", "size"),
        )
        .reset_index()
    )
    summary["win_rate"] *= 100
    summary["negative_r2_rate"] *= 100
    return summary


def plot_triptych(summary: pd.DataFrame, title: str, output_path: Path) -> None:
    sns.set_theme(style="whitegrid", font="DejaVu Sans")

    fig = plt.figure(figsize=(20, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.8, 1], height_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[:, 0])
    x = np.arange(len(METHOD_ORDER))
    offsets = {
        "BERT-best": -0.27,
        "Traditional-best": -0.09,
        "TabPFN-2.5-Plus-Numeric": 0.09,
        "TabPFN-2.5-Plus-Text": 0.27,
    }

    for label in SERIES_ORDER:
        sub = summary[summary["aggregate_label"] == label].sort_values("ood_method")
        if sub.empty:
            continue
        y = sub["median_summary_test_r2"].to_numpy()
        yerr = np.vstack(
            [
                y - sub["q1_summary_test_r2"].to_numpy(),
                sub["q3_summary_test_r2"].to_numpy() - y,
            ]
        )
        ax1.errorbar(
            x + offsets[label],
            y,
            yerr=yerr,
            fmt="o-",
            capsize=4,
            lw=2.2,
            markersize=7,
            color=PALETTE[label],
            label=label,
        )

    ax1.axhline(0, color="black", lw=1, alpha=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(METHOD_ORDER, rotation=20)
    ax1.set_xlabel("OOD method")
    ax1.set_ylabel("Summary test R2 (median with IQR)")
    ax1.set_title("Best-of-family comparison using model-level summary_test_r2")
    ax1.legend(title="Series", loc="lower right", frameon=True)
    ax1.text(
        0.02,
        0.97,
        "BERT-best / Traditional-best are chosen by summary_test_r2 desc, summary_test_r2_std asc, "
        "summary_test_mae asc, then model asc. TabPFN variants stay as independent single-run baselines.",
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=10,
    )

    ax2 = fig.add_subplot(gs[0, 1])
    win_pivot = (
        summary.pivot(index="aggregate_label", columns="ood_method", values="win_rate")
        .reindex(index=SERIES_ORDER, columns=METHOD_ORDER)
    )
    sns.heatmap(
        win_pivot,
        annot=True,
        fmt=".0f",
        cmap="YlGnBu",
        cbar_kws={"label": "Win rate among 4 series (%)"},
        ax=ax2,
        linewidths=0.5,
        linecolor="white",
        vmin=0,
        vmax=100,
    )
    ax2.set_title("Task win rate")
    ax2.set_xlabel("OOD method")
    ax2.set_ylabel("")
    ax2.tick_params(axis="x", rotation=20)

    ax3 = fig.add_subplot(gs[1, 1])
    neg_pivot = (
        summary.pivot(index="aggregate_label", columns="ood_method", values="negative_r2_rate")
        .reindex(index=SERIES_ORDER, columns=METHOD_ORDER)
    )
    sns.heatmap(
        neg_pivot,
        annot=True,
        fmt=".0f",
        cmap="Reds",
        cbar_kws={"label": "Negative R2 rate (%)"},
        ax=ax3,
        linewidths=0.5,
        linecolor="white",
        vmin=0,
        vmax=100,
    )
    ax3.set_title("Instability signal")
    ax3.set_xlabel("OOD method")
    ax3.set_ylabel("")
    ax3.tick_params(axis="x", rotation=20)

    fig.suptitle(title, fontsize=18, y=1.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_outputs(selected: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_dir / "selected_rows.csv", index=False, encoding="utf-8-sig")

    all_summaries: list[pd.DataFrame] = []
    tasks = (
        selected[TASK_KEYS]
        .drop_duplicates()
        .sort_values(["alloy_family", "property"])
        .to_dict("records")
    )

    for task in tasks:
        task_df = selected[
            (selected["alloy_family"] == task["alloy_family"])
            & (selected["dataset_name"] == task["dataset_name"])
            & (selected["property"] == task["property"])
        ].copy()
        summary = summarise_task(task_df)
        summary["alloy_family"] = task["alloy_family"]
        summary["dataset_name"] = task["dataset_name"]
        summary["property"] = task["property"]
        all_summaries.append(summary)

        stem = "__".join(
            [
                safe_name(task["alloy_family"]),
                safe_name(task["dataset_name"]),
                safe_name(task["property"]),
                "bestplus_tabpfn_triptych",
            ]
        )
        summary.to_csv(output_dir / f"{stem}.csv", index=False, encoding="utf-8-sig")
        plot_triptych(
            summary,
            title=f"OOD summary for {task['alloy_family']} | {task['dataset_name']} | {task['property']}",
            output_path=output_dir / f"{stem}.png",
        )

    if all_summaries:
        pd.concat(all_summaries, ignore_index=True).to_csv(
            output_dir / "all_tasks_bestplus_tabpfn_triptych_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-task OOD triptych figures using family-best baselines and both TabPFN variants."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("output/ood_summary_reports/Combined/data/all_model_families_ood_summary.csv"),
        help="Combined OOD summary CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/ood_summary_reports/Combined/figure/per_task_bestplus_tabpfn"),
        help="Directory for per-task triptych outputs.",
    )
    args = parser.parse_args()

    if not args.summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {args.summary_csv}")

    selected = build_selected_rows(args.summary_csv)
    if selected.empty:
        print(f"No rows available in summary CSV: {args.summary_csv}")
        return

    build_outputs(selected, args.output_dir)
    print(f"Per-task triptych outputs written to: {args.output_dir}")
    print(f"Selected rows: {len(selected)}")


if __name__ == "__main__":
    main()
