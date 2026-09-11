from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
from PIL import Image, ImageStat


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scripts.maintenance.restructure_hybrid_results import (
    canonical_hybrid_method,
    canonical_task_id,
    copy_png,
    entry,
    prepare_published_hybrid_frame,
    write_catalog,
    write_csv,
)


SCOPES = ("combined", "test_extrapolation_high20", "test_inner_ood")
AXES = {
    "Wx": {"space": "X-space", "source_name": "wx"},
    "Wy": {"space": "Y-space", "source_name": "wy"},
    "Wz": {"space": "Z-space", "source_name": "wz"},
}
TASK_KEY_TO_SOURCE_ID = {
    "Al__aluminum__UTS_MPa": "Al-UTS",
    "HEA__hea__El": "HEA-El",
    "HEA__hea__UTS_MPa": "HEA-UTS",
    "MatbenchSteels__matbench_steels_ood__YS_MPa": "Matbench Steel-YS",
    "Steel__steel__El": "Steel-El",
    "Steel__steel__UTS_MPa": "Steel-UTS",
    "Ti__titanium__El": "Ti-El",
    "Ti__titanium__UTS_MPa": "Ti-UTS",
}
EXPECTED_SOURCE_TASKS = frozenset(TASK_KEY_TO_SOURCE_ID.values())
EXPECTED_METHODS = frozenset({"RandCV", "LOCO", "SX-cls", "SX-sgl", "SY-cls", "SY-sgl"})
ANALYSIS = "confidence_ood_relationship"
MODEL = "GPT-5.4"

OVERVIEW_SPECS = (
    ("confidence_distribution", "supp_confidence_{axis}_sorted_by_task", "progression", "Confidence distribution"),
    ("confidence_share", "supp_confidence_share_{axis}_sorted_by_task", "progression", "Confidence share trend"),
    ("relative_error", "supp_relative_error_{axis}_sorted_by_task", "error", "Relative-error trend"),
    ("relative_error_bars", "supp_relative_error_{axis}_sorted_by_task_bars", "error", "Relative-error bars"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synchronize compact Hybrid confidence PNG/CSV pairs and rebuild the local viewer."
    )
    parser.add_argument("--hybrid-root", type=Path, required=True)
    parser.add_argument("--confidence-source", type=Path, required=True)
    parser.add_argument("--wx-inventory-root", type=Path, required=True)
    parser.add_argument("--confidence-figure-script", type=Path, required=True)
    return parser.parse_args()


def scope_root(root: Path, scope: str) -> Path:
    return root if scope == "combined" else root / scope


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_figure_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("hybrid_confidence_npj_builder", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import confidence figure builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_source_scope(base: Path, scope: str) -> None:
    validation_path = base / "validation_summary.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8-sig"))
    if validation.get("scope") != scope:
        raise ValueError(f"Confidence validation scope mismatch: {validation_path}")
    if validation.get("passed") is not True or int(validation.get("checks_passed", 0)) != 170:
        raise ValueError(f"Confidence source is not fully validated: {validation_path}")


def aggregate_wx(inventory: pd.DataFrame) -> pd.DataFrame:
    required = {"task_key", "model", "method", "fold_id", "wx", "test_size", "wx_mode"}
    missing = sorted(required - set(inventory.columns))
    if missing:
        raise ValueError(f"Wx inventory is missing columns: {missing}")

    rows = inventory[
        inventory["model"].astype(str).eq(MODEL)
        & inventory["wx_mode"].astype(str).eq("wx_sample_score")
    ].copy()
    rows["task_id"] = rows["task_key"].map(TASK_KEY_TO_SOURCE_ID)
    rows["method"] = rows["method"].map(canonical_hybrid_method)
    if rows["task_id"].isna().any() or rows["method"].isna().any():
        unknown_tasks = sorted(rows.loc[rows["task_id"].isna(), "task_key"].astype(str).unique())
        unknown_methods = sorted(rows.loc[rows["method"].isna(), "method"].astype(str).unique())
        raise ValueError(f"Unknown Wx task/method values: tasks={unknown_tasks}, methods={unknown_methods}")

    identity = ["task_id", "method", "fold_id"]
    if rows.duplicated(identity).any():
        duplicates = rows.loc[rows.duplicated(identity, keep=False), identity].head(10).to_dict("records")
        raise ValueError(f"Duplicate GPT-5.4 Wx fold rows: {duplicates}")
    rows["wx"] = pd.to_numeric(rows["wx"], errors="coerce")
    rows["test_size"] = pd.to_numeric(rows["test_size"], errors="coerce")
    if rows[["wx", "test_size"]].isna().any().any() or not rows["test_size"].gt(0).all():
        raise ValueError("Wx inventory contains invalid wx/test_size values")

    rows["weighted_wx"] = rows["wx"] * rows["test_size"]
    grouped = (
        rows.groupby(["task_id", "method"], as_index=False)
        .agg(
            weighted_wx=("weighted_wx", "sum"),
            wx_severity_test_n=("test_size", "sum"),
            wx_fold_count=("fold_id", "nunique"),
        )
    )
    grouped["Wx"] = grouped["weighted_wx"] / grouped["wx_severity_test_n"]
    baseline = grouped[grouped["method"].eq("RandCV")][["task_id", "Wx"]].rename(
        columns={"Wx": "RandCV_Wx"}
    )
    grouped = grouped.merge(baseline, on="task_id", how="left", validate="many_to_one")
    grouped["Wx_RandCV_ratio"] = grouped["Wx"] / grouped["RandCV_Wx"]
    grouped = grouped.drop(columns=["weighted_wx"])

    if set(grouped["task_id"]) != EXPECTED_SOURCE_TASKS:
        raise ValueError(f"Unexpected Wx tasks: {sorted(grouped['task_id'].unique())}")
    if set(grouped["method"]) != EXPECTED_METHODS:
        raise ValueError(f"Unexpected Wx methods: {sorted(grouped['method'].unique())}")
    if len(grouped) != len(EXPECTED_SOURCE_TASKS) * len(EXPECTED_METHODS):
        raise ValueError(f"Expected 48 Wx task-method rows, found {len(grouped)}")
    if grouped["Wx_RandCV_ratio"].isna().any():
        raise ValueError("Wx severity ratio contains missing values")
    rand_ratios = grouped.loc[grouped["method"].eq("RandCV"), "Wx_RandCV_ratio"]
    if not rand_ratios.map(lambda value: math.isclose(float(value), 1.0, abs_tol=1e-10)).all():
        raise ValueError("Wx RandCV baseline ratios are not one")
    return grouped.sort_values(["task_id", "method"], kind="stable").reset_index(drop=True)


def build_wx_tables(confidence_base: Path, inventory_base: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    confidence = pd.read_csv(confidence_base / "csv" / "confidence_by_task_method.csv", low_memory=False)
    confidence["method"] = confidence["method"].map(canonical_hybrid_method)
    wx = aggregate_wx(pd.read_csv(inventory_base / "csv" / "w_source_inventory.csv", low_memory=False))
    merged = confidence.merge(wx, on=["task_id", "method"], how="inner", validate="one_to_one")
    if len(merged) != 48:
        raise ValueError(f"Expected 48 confidence/Wx rows, found {len(merged)}")
    validate_wx_severity_counts(merged)

    merged["axis"] = "Wx"
    merged["axis_ratio"] = merged["Wx_RandCV_ratio"]
    merged["log2_axis_ratio"] = merged["axis_ratio"].map(math.log2)
    merged["axis_sort_order"] = (
        merged.groupby("task_id")["axis_ratio"].rank(method="first", ascending=True).astype(int)
    )
    progression_columns = [
        "axis", "task_id", "alloy", "property", "axis_sort_order", "method", "ood_method",
        "n_test", "axis_ratio", "log2_axis_ratio", "high_count", "medium_count", "low_count",
        "high_pct", "medium_pct", "low_pct", "Wx", "RandCV_Wx", "Wx_RandCV_ratio",
        "Wy_RandCV_ratio", "Wz_RandCV_ratio", "combined_severity", "wx_fold_count",
        "wx_severity_test_n",
    ]
    progression = merged[progression_columns].sort_values(
        ["task_id", "axis_ratio", "method"], kind="stable"
    ).reset_index(drop=True)

    error = pd.read_csv(
        confidence_base / "csv" / "wy_sorted_confidence_relative_error_summary_outlier_removed_100pct.csv",
        low_memory=False,
    )
    error["method"] = error["method"].map(canonical_hybrid_method)
    error = error.drop(columns=["axis", "axis_sort_order", "axis_ratio"], errors="ignore")
    error = error.merge(
        wx[["task_id", "method", "Wx", "RandCV_Wx", "Wx_RandCV_ratio", "wx_fold_count"]],
        on=["task_id", "method"],
        how="inner",
        validate="one_to_one",
    )
    error["axis"] = "Wx"
    error["axis_ratio"] = error["Wx_RandCV_ratio"]
    error["axis_sort_order"] = (
        error.groupby("task_id")["axis_ratio"].rank(method="first", ascending=True).astype(int)
    )
    first = ["axis", "task_id", "alloy", "property", "axis_sort_order", "method", "ood_method", "n_test", "axis_ratio"]
    error = error[first + [column for column in error.columns if column not in first]].sort_values(
        ["task_id", "axis_ratio", "method"], kind="stable"
    ).reset_index(drop=True)
    return progression, error


def validate_wx_severity_counts(merged: pd.DataFrame) -> None:
    """Check Wx against the existing split-based Wy/Wz sample count."""
    required = {"severity_test_n", "wx_severity_test_n"}
    missing = sorted(required - set(merged.columns))
    if missing:
        raise ValueError(f"Missing severity sample-count columns: {missing}")
    existing = pd.to_numeric(merged["severity_test_n"], errors="coerce")
    wx = pd.to_numeric(merged["wx_severity_test_n"], errors="coerce")
    if existing.isna().any() or wx.isna().any() or not existing.eq(wx).all():
        raise ValueError("Wy/Wz and Wx severity test-set sizes differ")


def load_axis_tables(base: Path, axis: str, inventory_base: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if axis == "Wx":
        return build_wx_tables(base, inventory_base)
    name = axis.lower()
    progression = pd.read_csv(base / "csv" / f"{name}_sorted_progression_by_task.csv", low_memory=False)
    error = pd.read_csv(
        base / "csv" / f"{name}_sorted_confidence_relative_error_summary_outlier_removed_100pct.csv",
        low_memory=False,
    )
    return progression, error


def render_wx_figures(
    figure_module: ModuleType,
    progression: pd.DataFrame,
    error: pd.DataFrame,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    outputs.extend(figure_module.build_supp_confidence_figure(progression, "Wx", output_dir))
    outputs.extend(figure_module.build_supp_confidence_share_trend_figure(progression, "Wx", output_dir))
    outputs.extend(figure_module.build_supp_relative_error_figure(error, "Wx", output_dir))
    outputs.extend(figure_module.build_supp_relative_error_bars_figure(error, "Wx", output_dir))
    task_outputs, _ = figure_module.build_all_task_4panel_figures(progression, error, "Wx", output_dir)
    outputs.extend(task_outputs)
    if len(outputs) != 12:
        raise ValueError(f"Expected 12 Wx figures, rendered {len(outputs)}")
    validation = figure_module.validate_outputs(outputs)
    if validation.get("passed") is not True:
        raise ValueError(f"Wx figure validation failed: {validation}")
    return output_dir


def matched_task_table(progression: pd.DataFrame, error: pd.DataFrame, task_id: str) -> pd.DataFrame:
    keys = ["axis", "task_id", "alloy", "property", "axis_sort_order", "method", "ood_method", "n_test", "axis_ratio"]
    extra_error_columns = [column for column in error.columns if column not in keys and column not in progression.columns]
    merged = progression.merge(
        error[keys + extra_error_columns],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    task = merged[merged["task_id"].astype(str).eq(str(task_id))].copy()
    if len(task) != 6 or set(task["method"].map(canonical_hybrid_method)) != EXPECTED_METHODS:
        raise ValueError(f"Expected six methods for {task_id}, found {len(task)}")
    return prepare_published_hybrid_frame(task)


def stage_scope(
    *,
    scope: str,
    confidence_source: Path,
    wx_inventory_root: Path,
    stage_root: Path,
    wx_render_root: Path,
    figure_module: ModuleType,
    catalog: list[dict[str, Any]],
) -> None:
    source_base = scope_root(confidence_source, scope)
    inventory_base = wx_inventory_root / scope
    validate_source_scope(source_base, scope)
    source_figure_root = source_base / "figures" / "npj"
    target_root = stage_root / ANALYSIS / scope

    tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for axis in AXES:
        tables[axis] = load_axis_tables(source_base, axis, inventory_base)
    wx_figure_root = render_wx_figures(
        figure_module,
        tables["Wx"][0],
        tables["Wx"][1],
        wx_render_root / scope,
    )

    for axis, metadata in AXES.items():
        progression, error = tables[axis]
        axis_lower = metadata["source_name"]
        figure_source_root = wx_figure_root if axis == "Wx" else source_figure_root
        for _, stem_template, table_name, title_label in OVERVIEW_SPECS:
            stem = stem_template.format(axis=axis_lower)
            source_png = figure_source_root / f"{stem}.png"
            figure_out = target_root / "figure" / "overview" / axis / f"{stem}.png"
            csv_out = target_root / "csv" / "overview" / axis / f"{stem}.csv"
            copy_png(source_png, figure_out)
            frame = progression if table_name == "progression" else error
            count = write_csv(prepare_published_hybrid_frame(frame), csv_out)
            catalog.append(
                entry(
                    analysis=ANALYSIS,
                    scope=scope,
                    category="overview",
                    model=MODEL,
                    rep_space=metadata["space"],
                    title=f"{scope} | {axis} | {title_label}",
                    figure=figure_out,
                    data=csv_out,
                    output_root=stage_root,
                    row_count=count,
                )
            )

        source_tasks = sorted(set(progression["task_id"].astype(str)))
        if set(source_tasks) != EXPECTED_SOURCE_TASKS:
            raise ValueError(f"Unexpected confidence tasks for {scope}/{axis}: {source_tasks}")
        for source_task in source_tasks:
            canonical_task = canonical_task_id(source_task)
            stem = f"{canonical_task}_confidence_ood_{axis_lower}_4panel"
            source_png = figure_source_root / "by_task" / axis_lower / f"{stem}.png"
            figure_out = target_root / "figure" / "by_task" / axis / canonical_task / f"{stem}.png"
            csv_out = target_root / "csv" / "by_task" / axis / canonical_task / f"{stem}.csv"
            copy_png(source_png, figure_out)
            count = write_csv(matched_task_table(progression, error, source_task), csv_out)
            catalog.append(
                entry(
                    analysis=ANALYSIS,
                    scope=scope,
                    category="by_task",
                    task_id=canonical_task,
                    model=MODEL,
                    rep_space=metadata["space"],
                    title=f"{canonical_task} | {axis} | Confidence–OOD four-panel",
                    figure=figure_out,
                    data=csv_out,
                    output_root=stage_root,
                    row_count=count,
                )
            )


def validate_staged_results(stage_root: Path, catalog: list[dict[str, Any]]) -> dict[str, Any]:
    confidence_root = stage_root / ANALYSIS
    pngs = sorted(confidence_root.rglob("*.png"))
    csvs = sorted(confidence_root.rglob("*.csv"))
    errors: list[str] = []
    if len(catalog) != 108 or len(pngs) != 108 or len(csvs) != 108:
        errors.append(f"Expected 108 entries/PNGs/CSVs, found {len(catalog)}/{len(pngs)}/{len(csvs)}")
    for item in catalog:
        figure = stage_root / item["figure_path"]
        data = stage_root / item["csv_path"]
        if not figure.is_file() or figure.suffix.lower() != ".png":
            errors.append(f"Missing/non-PNG figure: {item['figure_path']}")
            continue
        if not data.is_file() or data.suffix.lower() != ".csv":
            errors.append(f"Missing CSV: {item['csv_path']}")
            continue
        frame = pd.read_csv(data, low_memory=False)
        if len(frame) != int(item["row_count"]):
            errors.append(f"Row mismatch: {item['csv_path']}")
        expected_rows = 48 if item["category"] == "overview" else 6
        if len(frame) != expected_rows:
            errors.append(f"Unexpected rows: {item['csv_path']}={len(frame)}")
        with Image.open(figure) as image:
            dpi = image.info.get("dpi", (0, 0))
            if image.mode != "RGB" or min(dpi) < 590:
                errors.append(f"Image mode/DPI mismatch: {item['figure_path']} mode={image.mode} dpi={dpi}")
            grayscale = image.convert("L")
            if ImageStat.Stat(grayscale).var[0] <= 0:
                errors.append(f"Blank image: {item['figure_path']}")
    counts = pd.DataFrame(catalog).groupby(["scope", "rep_space"]).size().to_dict()
    for scope in SCOPES:
        for space in ("X-space", "Y-space", "Z-space"):
            if counts.get((scope, space)) != 12:
                errors.append(f"Expected 12 entries for {scope}/{space}, found {counts.get((scope, space))}")
    forbidden = [path for path in confidence_root.rglob("*") if path.is_file() and path.suffix.lower() not in {".png", ".csv"}]
    if forbidden:
        errors.append(f"Forbidden confidence file types: {forbidden[:5]}")
    if errors:
        raise RuntimeError("Confidence staging validation failed:\n" + "\n".join(errors[:50]))
    return {"entries": len(catalog), "pngs": len(pngs), "csvs": len(csvs)}


def verify_copied_source_hashes(confidence_source: Path, hybrid_root: Path) -> int:
    checked = 0
    for scope in SCOPES:
        source_base = scope_root(confidence_source, scope) / "figures" / "npj"
        target_base = hybrid_root / ANALYSIS / scope / "figure"
        for axis in ("Wy", "Wz"):
            axis_lower = axis.lower()
            for _, stem_template, _, _ in OVERVIEW_SPECS:
                stem = stem_template.format(axis=axis_lower)
                source = source_base / f"{stem}.png"
                target = target_base / "overview" / axis / f"{stem}.png"
                if sha256(source) != sha256(target):
                    raise ValueError(f"Copied source hash mismatch: {target}")
                checked += 1
            for source in sorted((source_base / "by_task" / axis_lower).glob("*.png")):
                task = source.stem.split("_confidence_ood_", 1)[0]
                target = target_base / "by_task" / axis / task / source.name
                if sha256(source) != sha256(target):
                    raise ValueError(f"Copied source hash mismatch: {target}")
                checked += 1
    if checked != 72:
        raise ValueError(f"Expected 72 copied Wy/Wz hashes, checked {checked}")
    return checked


def rebuild_catalog_and_viewer(hybrid_root: Path, new_entries: list[dict[str, Any]]) -> None:
    current = pd.read_csv(hybrid_root / "figure_catalog.csv", low_memory=False)
    keep = current[~current["analysis"].astype(str).eq(ANALYSIS)].drop(columns=["figure_id"], errors="ignore")
    combined = pd.concat([keep, pd.DataFrame(new_entries)], ignore_index=True, sort=False)
    write_catalog(combined.where(pd.notna(combined), "").to_dict(orient="records"), hybrid_root)
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "Scripts" / "maintenance" / "build_hybrid_data_viewer.py"),
            "--hybrid-root",
            str(hybrid_root),
            "--viewer-root",
            str(hybrid_root),
            "--image-root",
            "./",
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def main() -> None:
    args = parse_args()
    hybrid_root = args.hybrid_root.resolve()
    confidence_source = args.confidence_source.resolve()
    wx_inventory_root = args.wx_inventory_root.resolve()
    figure_script = args.confidence_figure_script.resolve()
    for path in (hybrid_root, confidence_source, wx_inventory_root, figure_script):
        if not path.exists():
            raise FileNotFoundError(path)

    figure_module = load_figure_module(figure_script)
    figure_module.configure_matplotlib()
    new_entries: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix=".confidence-build-", dir=hybrid_root) as temporary:
        stage_root = Path(temporary)
        wx_render_root = stage_root / "_wx_render"
        for scope in SCOPES:
            stage_scope(
                scope=scope,
                confidence_source=confidence_source,
                wx_inventory_root=wx_inventory_root,
                stage_root=stage_root,
                wx_render_root=wx_render_root,
                figure_module=figure_module,
                catalog=new_entries,
            )
        shutil.rmtree(wx_render_root)
        stage_report = validate_staged_results(stage_root, new_entries)
        destination = hybrid_root / ANALYSIS
        if destination.exists():
            shutil.rmtree(destination)
        (stage_root / ANALYSIS).replace(destination)

    rebuild_catalog_and_viewer(hybrid_root, new_entries)
    hash_count = verify_copied_source_hashes(confidence_source, hybrid_root)
    print(
        json.dumps(
            {
                **stage_report,
                "hybrid_root": str(hybrid_root),
                "confidence_entries": len(new_entries),
                "source_hashes_verified": hash_count,
                "catalog_entries": len(pd.read_csv(hybrid_root / "figure_catalog.csv", low_memory=False)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
