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
    std_metric = selection_cfg["family_best_std_metric"]
    mae_metric = selection_cfg["family_best_mae_metric"]

    family_df["_sort_r2"] = pd.to_numeric(family_df[best_metric], errors="coerce").fillna(-np.inf)
    family_df["_sort_r2_std"] = pd.to_numeric(family_df[std_metric], errors="coerce").fillna(np.inf)
    family_df["_sort_mae"] = pd.to_numeric(family_df[mae_metric], errors="coerce").fillna(np.inf)
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
        help="Triptych JSON config file.",
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
