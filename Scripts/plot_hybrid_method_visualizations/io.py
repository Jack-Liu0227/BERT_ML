from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_HYBRID_ROOT, DEFAULT_OUTPUT_DIR, DEFAULT_PURE_ROOT, DEFAULT_TRIPTYCH_ROOT, RAW_SET_SUBSETS, SUBSET_LABELS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build method-aware Hybrid Set A/B W-severity visualizations.")
    parser.add_argument("--hybrid-root", default=str(DEFAULT_HYBRID_ROOT), help="Root with subset/csv/w_error_samples_long.csv files.")
    parser.add_argument("--pure-root", default=str(DEFAULT_PURE_ROOT), help="Optional pure OOD w_error_relationship root.")
    parser.add_argument("--triptych-root", default=str(DEFAULT_TRIPTYCH_ROOT), help="Optional triptych root used to audit/align Set A/B MAE values.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--case-contains", default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    parser.add_argument("--overview-only", action="store_true", help="Only generate overview figures.")
    parser.add_argument("--task-figures-only", action="store_true", help="Only generate per-task figures.")
    parser.add_argument("--skip-report", action="store_true", help="Skip Markdown report generation.")
    return parser.parse_args(argv)


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False)


def write_csv(frame: pd.DataFrame, path: Path, columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    result = frame.copy()
    if columns is not None:
        for column in columns:
            if column not in result.columns:
                result[column] = np.nan
        result = result[columns]
    result.to_csv(path, index=False, encoding="utf-8-sig")


def clean_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def safe_filename(text: object) -> str:
    value = clean_text(text)
    value = re.sub(r"[^\w.\-]+", "_", value, flags=re.UNICODE)
    value = re.sub(r"_+", "_", value).strip("._")
    return value or "unknown"


def normalize_id(value: object) -> str:
    text = clean_text(value)
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return text


def source_split_subset_map(source_split_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for subset in RAW_SET_SUBSETS:
        path = source_split_dir / "test_sets" / f"{subset}.csv"
        if not path.exists():
            continue
        try:
            frame = read_csv(path)
        except Exception:
            continue
        if "ID" not in frame.columns:
            continue
        for value in frame["ID"].tolist():
            sample_id = normalize_id(value)
            if sample_id:
                mapping[sample_id] = subset
    return mapping


def relabel_hybrid_rows_from_source_splits(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "source_split_dir" not in frame.columns or "ID" not in frame.columns:
        return frame.copy()
    result = frame.copy()
    maps: dict[str, dict[str, str]] = {}
    corrected: list[str] = []
    corrected_any = False
    for _, row in result.iterrows():
        raw_subset = clean_text(row.get("raw_subset") or row.get("test_set"))
        source_split_dir = clean_text(row.get("source_split_dir"))
        if not source_split_dir:
            corrected.append(raw_subset)
            continue
        if source_split_dir not in maps:
            maps[source_split_dir] = source_split_subset_map(Path(source_split_dir))
        strict_subset = maps[source_split_dir].get(normalize_id(row.get("ID")))
        if strict_subset in RAW_SET_SUBSETS:
            corrected.append(strict_subset)
            corrected_any = corrected_any or strict_subset != raw_subset
        else:
            corrected.append(raw_subset)
    if corrected_any:
        result["raw_subset"] = corrected
        result["test_set"] = corrected
    return result


def case_filter(frame: pd.DataFrame, case_contains: str | None) -> pd.DataFrame:
    if not case_contains or frame.empty:
        return frame.copy()
    needle = case_contains.lower()
    hay = (
        frame.get("task_id", pd.Series("", index=frame.index)).fillna("").astype(str)
        + " "
        + frame.get("task_key", pd.Series("", index=frame.index)).fillna("").astype(str)
        + " "
        + frame.get("alloy_family", pd.Series("", index=frame.index)).fillna("").astype(str)
        + " "
        + frame.get("property", pd.Series("", index=frame.index)).fillna("").astype(str)
    ).str.lower()
    return frame[hay.str.contains(needle, regex=False)].copy()


def load_hybrid_long_tables(hybrid_root: Path, case_contains: str | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for subset in RAW_SET_SUBSETS:
        path = hybrid_root / subset / "csv" / "w_error_samples_long.csv"
        if path.exists():
            frame = read_csv(path)
            frame["test_set"] = subset
            frame["raw_subset"] = subset
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No hybrid w_error_samples_long.csv files found under {hybrid_root}")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = relabel_hybrid_rows_from_source_splits(combined)
    combined = combined[combined.get("test_set", "").astype(str).str.lower().isin(["test_extrapolation_high20", "test_inner_ood"])].copy()
    return case_filter(combined, case_contains)


def load_pure_long_table(pure_root: Path | None, case_contains: str | None = None) -> pd.DataFrame:
    if pure_root is None:
        return pd.DataFrame()
    path = pure_root / "csv" / "w_error_samples_long.csv"
    if not path.exists():
        return pd.DataFrame()
    return case_filter(read_csv(path), case_contains)


FIGURE_TITLES = {
    "figure_1_severity_decomposition": "Figure 1. Set A/B W Severity Decomposition",
    "figure_2_error_dumbbell": "Figure 2. A/B Model Error Dumbbell",
    "figure_3_representation_severity_sensitivity": "Figure 3. X/Z Representation Severity-Sensitivity Map",
    "figure_4_target_ceiling_loco": "Figure 4a. Y-Space Target-Ceiling Map (LOCO)",
    "figure_4_target_ceiling_by_method": "Figure 4b. Y-Space Target-Ceiling Map by Method",
    "figure_5_model_method_delta_heatmap": "Figure 5. Model x Method A/B Delta Heatmap",
    "figure_6_hybrid_failure_disentanglement": "Figure 6. Hybrid Failure Disentanglement",
    "figure_7_method_diagnostic_contrast": "Figure 7. Method-Level Diagnostic Contrast",
    "figure_8_yz_ood_map": "Figure 8. Hybrid Set A/B Wy-Wz OOD Map",
}


FIGURE_CAPTIONS = {
    "figure_1_severity_decomposition": "Fold-weighted W_p90 distributions compare Set A and Set B across X/Y/Z spaces and methods.",
    "figure_2_error_dumbbell": "Model dumbbells compare Set A and Set B MAE within each method using fold-weighted task or overview summaries.",
    "figure_3_representation_severity_sensitivity": "Severity-sensitivity scatter plots keep X-space and Z-space separate; points show W_p90 versus relative-error slope.",
    "figure_4_target_ceiling_loco": "Set A Y-space W_p90 is compared with ceiling bias under the LOCO hybrid setting; negative bias means systematic underprediction.",
    "figure_4_target_ceiling_by_method": "Target-ceiling scatter plots compare Set A Y-space W_p90 with ceiling bias across methods and model families.",
    "figure_5_model_method_delta_heatmap": "Delta metrics summarize Set A minus Set B behavior for each model-method pair; diverging colors emphasize direction and magnitude.",
    "figure_6_hybrid_failure_disentanglement": "Hybrid-only Set A/B summaries compare MAE and X/Y/Z W_p90 for LOCO and RandCV, showing whether hybrid splits separate target-space and representation-space failures.",
    "figure_7_method_diagnostic_contrast": "Method-level W_p90 contrasts show what kind of X/Y/Z OOD stress each method creates in Set A and Set B.",
    "figure_8_yz_ood_map": "Set A/top20 and Set B/inner are overlaid on one Wy-Wz coordinate system; marker shape encodes the set and color encodes the hybrid method.",
}


def relative_figure_path(output_dir: Path, figure_path: object) -> str:
    path = Path(clean_text(figure_path))
    try:
        return path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def first_png_for(manifest: pd.DataFrame, scope: str, task_id: str, figure_id: str) -> str | None:
    if manifest.empty:
        return None
    panel = manifest[
        manifest["scope"].astype(str).eq(scope)
        & manifest["task_id"].astype(str).eq(task_id)
        & manifest["figure_id"].astype(str).eq(figure_id)
    ].copy()
    if panel.empty:
        return None
    png = panel[panel["format"].astype(str).eq("png")]
    row = (png if not png.empty else panel).iloc[0]
    return clean_text(row["figure"])


def metric_range_text(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame.columns:
        return "not available"
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return "not available"
    return f"{values.min():.3g} to {values.max():.3g}"


def analysis_for_figure(figure_id: str, scope: str, task_id: str, summary: pd.DataFrame, delta: pd.DataFrame, pure_hybrid: pd.DataFrame) -> str:
    prefix = "Across all tasks" if scope == "overview" else f"For task `{task_id}`"
    if figure_id == "figure_1_severity_decomposition":
        if scope == "by_task":
            return f"{prefix}, this compact by-task plot keeps X/Y/Z as separate panels. Within each space, methods are ordered by mean W_p90 from low to high, and the legend distinguishes Set A from Set B."
        return f"{prefix}, this plot compares fold-level Set A and Set B severity. Y-space W_p90 ranges from {metric_range_text(summary[summary['space'].eq('Y-space')], 'W_p90')}, while X/Z panels show representation-side pressure."
    if figure_id == "figure_2_error_dumbbell":
        panel = summary[summary["space"].eq("Y-space")]
        if scope == "by_task":
            return f"{prefix}, the dumbbells show whether each model has higher MAE on Set A or Set B under each method. This mirrors the overview design but uses only this task's fold-weighted records."
        return f"{prefix}, the dumbbells summarize whether MAE is larger on Set A or Set B for the same model and method. MAE ranges from {metric_range_text(panel, 'MAE')}."
    if figure_id == "figure_3_representation_severity_sensitivity":
        panel = summary[summary["space"].isin(["X-space", "Z-space"])]
        if scope == "by_task":
            return f"{prefix}, this overview-like scatter layout keeps X-space and Z-space separate for each method. Right-up points indicate higher representation severity and stronger error sensitivity."
        return f"{prefix}, points in the upper-right indicate models whose relative error rises quickly as representation OOD severity increases. Slope ranges from {metric_range_text(panel, 'slope')}."
    if figure_id.startswith("figure_4"):
        panel = summary[(summary["subset"].eq("Set A")) & (summary["space"].eq("Y-space"))]
        if scope == "by_task":
            return f"{prefix}, this target-ceiling scatter uses Set A Y-space W_p90 and ceiling bias. More negative points indicate stronger systematic underprediction of high-performance samples."
        return f"{prefix}, negative ceiling bias marks systematic underprediction of high-performance Set A samples. GPT/GPT-5.4-labelled model families are shown as model-family points when present in the source data."
    if figure_id == "figure_5_model_method_delta_heatmap":
        return f"{prefix}, red/blue cells encode the direction of Set A minus Set B deltas, so this is the compact view for deciding whether a model-method pair leans toward target-ceiling or representation-disconnection failure."
    if figure_id == "figure_6_hybrid_failure_disentanglement":
        return f"{prefix}, this Hybrid-only panel compares Set A and Set B directly. The upper panel shows error response, while the lower panel separates X/Y/Z W severity to show which failure source dominates."
    if figure_id == "figure_7_method_diagnostic_contrast":
        return f"{prefix}, this figure isolates method as the stress generator: differences among LOCO, RandCV, SX, and SY show which method creates X/Y/Z severity in each subset."
    if figure_id == "figure_8_yz_ood_map":
        return f"{prefix}, this Wy-Wz map uses W_mean from Y-space on the x-axis and W_mean from Z-space on the y-axis. Set A/top20 and Set B/inner remain separately aggregated, then are overlaid with distinct markers."
    return f"{prefix}, this diagnostic is generated from fold-weighted summary tables."


def append_figure_section(
    lines: list[str],
    output_dir: Path,
    manifest: pd.DataFrame,
    scope: str,
    task_id: str,
    figure_id: str,
    summary: pd.DataFrame,
    delta: pd.DataFrame,
    pure_hybrid: pd.DataFrame,
) -> None:
    figure_path = first_png_for(manifest, scope, task_id, figure_id)
    if figure_path is None:
        return
    rel_path = relative_figure_path(output_dir, figure_path)
    lines.extend(
        [
            f"### {FIGURE_TITLES.get(figure_id, figure_id)}",
            "",
            f"![{figure_id}]({rel_path})",
            "",
            f"**Caption.** {FIGURE_CAPTIONS.get(figure_id, '')}",
            "",
            analysis_for_figure(figure_id, scope, task_id, summary, delta, pure_hybrid),
            "",
        ]
    )


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    delta: pd.DataFrame,
    pure_hybrid: pd.DataFrame,
    manifest: pd.DataFrame,
    audit: pd.DataFrame | None = None,
) -> None:
    figure_order = list(FIGURE_TITLES)
    lines = [
        "# Hybrid Method-Aware Visualization Report",
        "",
        "## Statistical Principle",
        "",
        "All Set A and Set B analyses use a **fold-weighted** aggregation. Set A Top20 samples may appear in multiple folds, but each occurrence has a different training distribution; therefore W, prediction, error, and sensitivity are fold-specific and are retained as fold-level records.",
        "",
        "- Set A means `test_extrapolation_high20`.",
        "- Set B means `test_inner_ood`.",
        "- Raw subset names are retained only for provenance; displayed figures and derived folders use Set A/B labels.",
        "- Figure 1-7 use audited Set A/B master tables.",
        "",
        f"- Hybrid summary rows: {len(summary)}",
        f"- A/B delta rows: {len(delta)}",
        f"- Hybrid-only Set A/B rows: {len(pure_hybrid)}",
        f"- Figure files: {len(manifest)}",
        "",
        "## Source Audit",
        "",
    ]
    if audit is None or audit.empty:
        lines.extend(["No triptych-aligned source audit table was available.", ""])
    else:
        verdict_counts = audit["verdict"].fillna("unknown").astype(str).value_counts().to_dict()
        lines.extend(["Set A/B model MAE is audited against triptych-aligned subset CSV files when available.", ""])
        for verdict, count in sorted(verdict_counts.items()):
            lines.append(f"- `{verdict}`: {count}")
        problem = audit[~audit["verdict"].astype(str).eq("match")].copy()
        if not problem.empty:
            lines.extend(["", "Non-matching or missing audit rows are listed in `csv/set_ab_source_audit.csv`; first rows:"])
            preview_cols = ["case", "model", "method", "set_label", "verdict", "diff_master_triptych"]
            for _, row in problem[preview_cols].head(10).iterrows():
                lines.append(f"- {row['case']} / {row['model']} / {row['method']} / {row['set_label']}: {row['verdict']} ({row['diff_master_triptych']})")
        lines.append("")
    lines.extend(
        [
        "## How To Read Figure 5-7",
        "",
        "- Figure 5: red/blue colors represent the direction of Set A minus Set B differences; strong signed values identify whether the failure is more target-ceiling-like or representation-disconnection-like.",
        "- Figure 6: compare Hybrid Set A and Hybrid Set B within LOCO/RandCV; MAE shows the error response, while X/Y/Z W_p90 shows whether the split separates target-space and representation-space failure sources.",
        "- Figure 7: compare methods as OOD stress generators; the X/Y/Z colors show which representation or target space each method stresses.",
        "",
        "## Overview Figures",
        "",
        "These figures are global summaries across all available tasks/properties. They are useful for finding broad patterns; task-specific conclusions should be checked in the by-task section.",
        "",
        ]
    )
    for figure_id in figure_order:
        append_figure_section(lines, output_dir, manifest, "overview", "all", figure_id, summary, delta, pure_hybrid)

    lines.extend(["## By-Task Figures", ""])
    task_ids = sorted(summary["case"].dropna().astype(str).unique().tolist()) if not summary.empty else []
    for task_id in task_ids:
        task_summary = summary[summary["case"].astype(str).eq(task_id)].copy()
        task_delta = delta[delta["case"].astype(str).eq(task_id)].copy() if not delta.empty else delta.copy()
        task_pure_hybrid = pure_hybrid[pure_hybrid["case"].astype(str).eq(task_id)].copy() if not pure_hybrid.empty else pure_hybrid.copy()
        label_parts = task_summary[["alloy_family", "dataset_name", "property"]].drop_duplicates().astype(str).head(1)
        label = ""
        if not label_parts.empty:
            row = label_parts.iloc[0]
            label = f" ({row['alloy_family']} / {row['dataset_name']} / {row['property']})"
        lines.extend([f"### Task: {task_id}{label}", ""])
        for figure_id in figure_order:
            append_figure_section(lines, output_dir, manifest, "by_task", task_id, figure_id, task_summary, task_delta, task_pure_hybrid)

    lines.extend(
        [
            "## Output Tables",
            "",
            "- `csv/set_ab_master_sample_long.csv`",
            "- `csv/set_ab_model_summary.csv`",
            "- `csv/set_ab_space_summary_long.csv`",
            "- `csv/set_ab_family_best_summary.csv`",
            "- `csv/set_ab_source_audit.csv`",
            "- `csv/set_ab_triptych_aligned_model_review.csv`",
            "- `csv/hybrid_set_ab_yz_ood_map_summary.csv`",
            "- `csv/hybrid_visualization_summary_long.csv`",
            "- `csv/hybrid_ab_delta_summary.csv`",
            "- `csv/pure_vs_hybrid_summary.csv`",
            "- `csv/figure_manifest.csv`",
        ]
    )
    (output_dir / "hybrid_method_visualization_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
