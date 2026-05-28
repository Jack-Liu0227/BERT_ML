from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from _bestplus_tabpfn_triptych_config import default_config_path, load_triptych_config
from _bestplus_tabpfn_triptych_plotting import plot_triptych, prepare_task_table


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


def _direction_to_ascending(direction: str) -> bool:
    normalized = str(direction).strip().lower()
    if normalized in {"asc", "ascending", "low", "lower", "min", "minimum"}:
        return True
    if normalized in {"desc", "descending", "high", "higher", "max", "maximum"}:
        return False
    raise ValueError(f"Unsupported sort direction: {direction}")


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
    _coalesce_column(working_df, "artifact_test_r2", [])
    _coalesce_column(working_df, "artifact_test_mae", [])
    _coalesce_column(working_df, "artifact_test_rmse", [])
    _coalesce_column(working_df, "plot_test_r2", ["artifact_test_r2", "summary_test_r2"])
    _coalesce_column(working_df, "plot_test_mae", ["artifact_test_mae", "summary_test_mae"])
    _coalesce_column(working_df, "plot_test_rmse", ["artifact_test_rmse", "summary_test_rmse"])

    numeric_columns = [
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
        "artifact_test_r2",
        "artifact_test_mae",
        "artifact_test_rmse",
        "plot_test_r2",
        "plot_test_mae",
        "plot_test_rmse",
    ]
    for column in numeric_columns:
        working_df[column] = pd.to_numeric(working_df[column], errors="coerce")

    for std_col in ["summary_test_r2_std", "summary_test_mae_std", "summary_test_rmse_std"]:
        working_df.loc[tabpfn_mask & working_df[std_col].isna(), std_col] = 0.0
    for count_col in ["trial_count", "fold_count"]:
        working_df.loc[tabpfn_mask & working_df[count_col].isna(), count_col] = 1
    working_df.loc[
        tabpfn_mask & working_df["representative_selection_mode"].isna(),
        "representative_selection_mode",
    ] = "single_run"

    loco_summary_mask = working_df["ood_method"].astype(str).eq("LOCO") & family_series.isin({"Traditional", "BERT"})
    for metric in ["r2", "mae", "rmse"]:
        summary_col = f"summary_test_{metric}"
        plot_col = f"plot_test_{metric}"
        working_df.loc[loco_summary_mask, plot_col] = pd.to_numeric(
            working_df.loc[loco_summary_mask, summary_col],
            errors="coerce",
        )
    return working_df


def select_best_family_rows(
    df: pd.DataFrame,
    family_name: str,
    aggregate_label: str,
    config: dict,
) -> pd.DataFrame:
    family_df = df[df["model_family"] == family_name].copy()
    if family_df.empty:
        return family_df

    selection_cfg = config["selection"]
    best_metric = selection_cfg["family_best_metric"]
    best_metric_direction = selection_cfg.get("family_best_metric_direction", "desc")
    std_metric = selection_cfg["family_best_std_metric"]
    std_metric_direction = selection_cfg.get("family_best_std_metric_direction", "asc")
    tiebreak_metric = selection_cfg.get("family_best_tiebreak_metric")
    tiebreak_metric_direction = selection_cfg.get("family_best_tiebreak_metric_direction", "asc")

    family_df["_sort_best_metric"] = pd.to_numeric(family_df[best_metric], errors="coerce")
    if _direction_to_ascending(best_metric_direction):
        family_df["_sort_best_metric"] = family_df["_sort_best_metric"].fillna(np.inf)
    else:
        family_df["_sort_best_metric"] = family_df["_sort_best_metric"].fillna(-np.inf)

    family_df["_sort_std_metric"] = pd.to_numeric(family_df[std_metric], errors="coerce")
    if _direction_to_ascending(std_metric_direction):
        family_df["_sort_std_metric"] = family_df["_sort_std_metric"].fillna(np.inf)
    else:
        family_df["_sort_std_metric"] = family_df["_sort_std_metric"].fillna(-np.inf)

    sort_columns = CASE_KEYS + ["_sort_best_metric", "_sort_std_metric"]
    ascending = [True, True, True, True, _direction_to_ascending(best_metric_direction), _direction_to_ascending(std_metric_direction)]

    if tiebreak_metric:
        family_df["_sort_tiebreak_metric"] = pd.to_numeric(family_df[tiebreak_metric], errors="coerce")
        if _direction_to_ascending(tiebreak_metric_direction):
            family_df["_sort_tiebreak_metric"] = family_df["_sort_tiebreak_metric"].fillna(np.inf)
        else:
            family_df["_sort_tiebreak_metric"] = family_df["_sort_tiebreak_metric"].fillna(-np.inf)
        sort_columns.append("_sort_tiebreak_metric")
        ascending.append(_direction_to_ascending(tiebreak_metric_direction))

    family_df = family_df.sort_values(
        sort_columns + ["model"],
        ascending=ascending + [True],
        kind="mergesort",
        na_position="last",
    )
    selected = family_df.groupby(CASE_KEYS, as_index=False, observed=True).head(1).copy()
    selected = selected.drop(columns=[col for col in ["_sort_best_metric", "_sort_std_metric", "_sort_tiebreak_metric"] if col in selected.columns])
    selected["aggregate_label"] = aggregate_label
    return selected


def select_best_aggregate_rows(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Keep one deterministic row per task/OOD/aggregate series label.

    BERT and Traditional rows are already reduced to one family-best row per
    task/OOD case. Standalone sources (for example external LLM/TabPFN runs) can
    still contain repeated labels for the same case when multiple source
    directories map to the same display label. The plotting code needs a single
    value for each (ood_method, aggregate_label) cell, so resolve such repeats
    with the same metric priority used for family-best selection.
    """
    if df.empty:
        return df

    duplicate_keys = TASK_KEYS + ["ood_method", "aggregate_label"]
    working_df = df.copy()
    selection_cfg = config["selection"]

    sort_columns = duplicate_keys.copy()
    ascending = [True] * len(sort_columns)

    def add_numeric_sort_column(config_key: str, direction_key: str, temp_column: str) -> None:
        metric = selection_cfg.get(config_key)
        if not metric or metric not in working_df.columns:
            return
        direction = selection_cfg.get(direction_key, "asc")
        sort_series = pd.to_numeric(working_df[metric], errors="coerce")
        if _direction_to_ascending(direction):
            sort_series = sort_series.fillna(np.inf)
        else:
            sort_series = sort_series.fillna(-np.inf)
        working_df[temp_column] = sort_series
        sort_columns.append(temp_column)
        ascending.append(_direction_to_ascending(direction))

    add_numeric_sort_column("family_best_metric", "family_best_metric_direction", "_sort_best_metric")
    add_numeric_sort_column("family_best_std_metric", "family_best_std_metric_direction", "_sort_std_metric")
    add_numeric_sort_column("family_best_tiebreak_metric", "family_best_tiebreak_metric_direction", "_sort_tiebreak_metric")

    deterministic_tie_columns = [
        column
        for column in [
            "model_family",
            "model",
            "display_label",
            "model_dir",
            "source_dir",
            "representative_predictions_file",
            "artifact_predictions_file",
        ]
        if column in working_df.columns
    ]
    sort_columns.extend(deterministic_tie_columns)
    ascending.extend([True] * len(deterministic_tie_columns))

    selected = (
        working_df.sort_values(
            sort_columns,
            ascending=ascending,
            kind="mergesort",
            na_position="last",
        )
        .drop_duplicates(duplicate_keys, keep="first")
        .copy()
    )
    cleanup_columns = ["_sort_best_metric", "_sort_std_metric", "_sort_tiebreak_metric"]
    return selected.drop(columns=[col for col in cleanup_columns if col in selected.columns])


def build_selected_rows(summary_csv: Path, config: dict) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    if df.empty:
        return df

    df = canonicalize_summary_schema(df)

    bert_best = select_best_family_rows(df, "BERT", "BERT-best", config)
    traditional_best = select_best_family_rows(df, "Traditional", "Traditional-best", config)

    standalone_mask = ~df["model_family"].astype(str).isin({"BERT", "Traditional"})
    standalone_df = df[standalone_mask].copy()
    if not standalone_df.empty:
        standalone_df["aggregate_label"] = (
            standalone_df["display_label"]
            .where(standalone_df["display_label"].notna(), standalone_df["model"])
            .astype(str)
        )

    selected = pd.concat([bert_best, traditional_best, standalone_df], ignore_index=True)
    if selected.empty:
        return selected
    selected = select_best_aggregate_rows(selected, config)

    configured_series_order = [str(label) for label in config["series_order"]]
    active_series_order = [
        label
        for label in configured_series_order
        if label in set(selected["aggregate_label"].astype(str))
    ]
    unconfigured_labels = [
        label for label in selected["aggregate_label"].astype(str).drop_duplicates().tolist() if label not in active_series_order
    ]
    active_series_order.extend(sorted(unconfigured_labels))
    selected["active_series_order"] = ",".join(active_series_order)

    selected["ood_method"] = pd.Categorical(selected["ood_method"], categories=config["method_order"], ordered=True)
    selected["aggregate_label"] = pd.Categorical(
        selected["aggregate_label"], categories=active_series_order, ordered=True
    )
    selected["summary_test_r2"] = pd.to_numeric(selected["summary_test_r2"], errors="coerce")
    selected["summary_test_r2_std"] = pd.to_numeric(selected["summary_test_r2_std"], errors="coerce").fillna(0.0)
    selected["summary_test_mae"] = pd.to_numeric(selected["summary_test_mae"], errors="coerce")
    selected["plot_test_r2"] = pd.to_numeric(selected["plot_test_r2"], errors="coerce")
    selected["plot_test_mae"] = pd.to_numeric(selected["plot_test_mae"], errors="coerce")
    selected["plot_test_rmse"] = pd.to_numeric(selected["plot_test_rmse"], errors="coerce")
    negative_flag_metric = config["display"]["negative_flag_metric"]
    selected["negative_r2_flag"] = pd.to_numeric(selected[negative_flag_metric], errors="coerce") < 0
    return selected.sort_values(CASE_KEYS + ["aggregate_label"]).reset_index(drop=True)


def build_overall_task_rank_table(task_table: pd.DataFrame) -> pd.DataFrame:
    if task_table.empty:
        return pd.DataFrame()

    summary_df = (
        task_table.groupby("aggregate_label", dropna=False, observed=True)
        .agg(
            overall_method_count=("ood_method", "nunique"),
            overall_mean_rank_in_method=("rank_in_method", "mean"),
            overall_mean_plot_test_mae=("plot_test_mae", "mean"),
            overall_worst_plot_test_mae=("plot_test_mae", "max"),
        )
        .reset_index()
    )

    summary_df = summary_df.sort_values(
        [
            "overall_mean_rank_in_method",
            "overall_mean_plot_test_mae",
            "overall_worst_plot_test_mae",
            "aggregate_label",
        ],
        ascending=[True, True, True, True],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)
    summary_df["overall_rank_across_ood_methods"] = np.arange(1, len(summary_df) + 1, dtype=int)
    summary_df["is_overall_best_across_ood_methods"] = summary_df["overall_rank_across_ood_methods"].eq(1)
    summary_df["overall_best_criterion"] = "mean(rank_in_method by plot_test_mae)"
    overall_best_label = str(summary_df.loc[summary_df["overall_rank_across_ood_methods"].eq(1), "aggregate_label"].iloc[0])
    summary_df["overall_best_label"] = overall_best_label
    return summary_df


def build_outputs(selected: pd.DataFrame, output_dir: Path, config: dict) -> None:
    output_cfg = config["output"]
    output_formats = [str(fmt).lstrip(".") for fmt in output_cfg.get("formats", ["png"])]
    if output_cfg.get("clean_output_dir", True) and output_dir.exists():
        shutil.rmtree(output_dir)

    data_dir = output_dir / output_cfg["data_subdir"]
    figure_dir = output_dir / output_cfg["figure_subdir"]
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    selected.to_csv(data_dir / "selected_rows.csv", index=False, encoding="utf-8-sig")

    all_summaries: list[pd.DataFrame] = []
    all_overall_rank_summaries: list[pd.DataFrame] = []
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
        task_table = prepare_task_table(task_df, config)
        overall_rank_table = build_overall_task_rank_table(task_table)
        if not overall_rank_table.empty:
            overall_rank_table["alloy_family"] = task["alloy_family"]
            overall_rank_table["dataset_name"] = task["dataset_name"]
            overall_rank_table["property"] = task["property"]
            all_overall_rank_summaries.append(overall_rank_table.copy())
            task_table = task_table.merge(
                overall_rank_table,
                on="aggregate_label",
                how="left",
                validate="many_to_one",
            )
        task_table["alloy_family"] = task["alloy_family"]
        task_table["dataset_name"] = task["dataset_name"]
        task_table["property"] = task["property"]
        all_summaries.append(task_table)

        stem = "__".join(
            [
                safe_name(task["alloy_family"]),
                safe_name(task["dataset_name"]),
                safe_name(task["property"]),
                "bestplus_tabpfn_triptych",
            ]
        )
        task_table.to_csv(data_dir / f"{stem}.csv", index=False, encoding="utf-8-sig")
        if not overall_rank_table.empty:
            overall_rank_table.to_csv(
                data_dir / f"{stem.replace('__bestplus_tabpfn_triptych', '__bestplus_tabpfn_overall_rank')}.csv",
                index=False,
                encoding="utf-8-sig",
            )
        for fmt in output_formats:
            plot_triptych(
                task_df,
                title=output_cfg["title_template"].format(**task),
                output_path=figure_dir / f"{stem}.{fmt}",
                config=config,
            )

    if all_summaries:
        pd.concat(all_summaries, ignore_index=True).to_csv(
            data_dir / "all_tasks_bestplus_tabpfn_triptych_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if all_overall_rank_summaries:
        pd.concat(all_overall_rank_summaries, ignore_index=True).to_csv(
            data_dir / "all_tasks_bestplus_tabpfn_overall_rank.csv",
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
        "--config",
        type=Path,
        default=default_config_path(),
        help="Triptych YAML or JSON config file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/ood_summary_reports/Combined/figure/per_task_bestplus_tabpfn"),
        help="Root directory for per-task triptych outputs.",
    )
    args = parser.parse_args()

    if not args.summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {args.summary_csv}")

    config = load_triptych_config(args.config)
    selected = build_selected_rows(args.summary_csv, config)
    if selected.empty:
        print(f"No rows available in summary CSV: {args.summary_csv}")
        return

    build_outputs(selected, args.output_dir, config)
    print(f"Per-task triptych outputs written to: {args.output_dir}")
    print(f"Selected rows: {len(selected)}")
    print(f"Config used: {args.config}")


if __name__ == "__main__":
    main()
