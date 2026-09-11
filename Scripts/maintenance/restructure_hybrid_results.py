from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable

import pandas as pd


SCOPES = ("combined", "test_extrapolation_high20", "test_inner_ood")
EXCLUDED_TASK_IDS = frozenset({"HEA-YS", "Steel-YS"})
EXCLUDED_TASK_KEYS = frozenset({"HEA__hea__YS_MPa", "Steel__steel__YS_MPa"})

PANEL_TASKS = {
    "ood_al_aluminum_utsmpa": "Al-UTS",
    "ood_hea_hea_elpct": "HEA-El",
    "ood_hea_hea_utsmpa": "HEA-UTS",
    "ood_matbench_steel_matbench_steels_ood_yieldstrength": "Matbench_Steel-YS",
    "ood_steel_steel_elpct": "Steel-El",
    "ood_steel_steel_utsmpa": "Steel-UTS",
    "ood_ti_titanium_elpct": "Ti-El",
    "ood_ti_titanium_utsmpa": "Ti-UTS",
}

TRIPTYCH_TASKS = {
    "ood_al_aluminum_utsmpa": "Al-UTS",
    "ood_hea_hea_elpct": "HEA-El",
    "ood_hea_hea_utsmpa": "HEA-UTS",
    "ood_matbench_steel_yield_strength": "Matbench_Steel-YS",
    "ood_steel_steel_elpct": "Steel-El",
    "ood_steel_steel_utsmpa": "Steel-UTS",
    "ood_ti_titanium_elpct": "Ti-El",
    "ood_ti_titanium_utsmpa": "Ti-UTS",
}

SCATTER_COLUMNS = [
    "scope",
    "task_id",
    "model",
    "model_family",
    "method_short",
    "fold_id",
    "ID",
    "sample_order",
    "test_set",
    "space",
    "sample_w_contribution",
    "abs_error",
    "relative_error_pct",
]

PHASE_META_COLUMNS = [
    "scope",
    "task_id",
    "method_short",
    "model",
    "model_family",
    "fold_id",
    "ID",
    "sample_order",
    "test_set",
    "relative_error_pct",
]

PANEL_COLUMNS = [
    "task",
    "method",
    "method_short",
    "display_name",
    "plot_group_label",
    "__group_value__",
    "projection_x",
    "projection_y",
    "x",
    "y",
    "role",
    "test_set",
    "test_set_label",
    "fold_id",
    "row_identity",
    "panel_group_source",
]

COLUMN_DESCRIPTIONS = {
    "scope": ("Data scope used to generate the figure.", ""),
    "task_id": ("Canonical alloy-property task identifier.", ""),
    "task": ("Human-readable alloy-property task.", ""),
    "model": ("Model displayed in the figure.", ""),
    "model_family": ("Model-family grouping used for panel ordering.", ""),
    "method": ("Full evaluation protocol or method name.", ""),
    "method_short": ("Short protocol label used in the figure legend.", ""),
    "space": ("Representation space for the W-error panel.", ""),
    "sample_w_contribution": ("Per-sample W contribution plotted on the x axis.", "W"),
    "abs_error": ("Absolute prediction error plotted in the MAE panels.", "target unit"),
    "relative_error_pct": ("Absolute relative prediction error.", "%"),
    "X_space_w": ("Per-sample W contribution in X-space.", "W"),
    "Y_space_w": ("Per-sample W contribution in Y-space.", "W"),
    "Z_space_w": ("Per-sample W contribution in Z-space.", "W"),
    "Ry": ("Y-space severity relative to the plotted baseline.", "ratio"),
    "Rz": ("Z-space severity relative to the plotted baseline.", "ratio"),
    "Wy": ("Y-space split severity.", "W"),
    "Wz": ("Z-space split severity.", "W"),
    "Wx": ("GPT-5.4 X-space split severity, averaged across folds by test-set size.", "W"),
    "RandCV_Wx": ("Task-specific RandCV baseline for X-space split severity.", "W"),
    "Wx_RandCV_ratio": ("X-space split severity relative to the task RandCV baseline.", "ratio"),
    "axis": ("Wasserstein space used to order the confidence figure.", ""),
    "axis_ratio": ("Selected-space severity relative to the task RandCV baseline.", "ratio"),
    "log2_axis_ratio": ("Base-2 logarithm of the selected-space severity ratio.", "log2 ratio"),
    "axis_sort_order": ("Within-task order by selected-space severity ratio.", "rank"),
    "high_count": ("Number of high-confidence predictions.", "samples"),
    "medium_count": ("Number of medium-confidence predictions.", "samples"),
    "low_count": ("Number of low-confidence predictions.", "samples"),
    "high_pct": ("Fraction of predictions labelled high confidence.", "fraction"),
    "medium_pct": ("Fraction of predictions labelled medium confidence.", "fraction"),
    "low_pct": ("Fraction of predictions labelled low confidence.", "fraction"),
    "outlier_rule": ("Relative-error filtering rule used by confidence error panels.", ""),
    "projection_x": ("Displayed t-SNE horizontal coordinate.", ""),
    "projection_y": ("Displayed t-SNE vertical coordinate.", ""),
    "plot_group_label": ("Panel highlight group shown in the figure.", ""),
    "source_table": ("Source panel table represented by this row.", ""),
    "figure_path": ("PNG path relative to the hybrid result root.", ""),
    "csv_path": ("Matching CSV path relative to the hybrid result root.", ""),
    "row_count": ("Number of rows in the matching CSV.", "rows"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the compact Hybrid PNG/CSV result hierarchy.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def ensure_clean_output(path: Path) -> None:
    if path.exists():
        if any(path.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {path}")
    else:
        path.mkdir(parents=True)


def write_csv(frame: pd.DataFrame, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10g")
    return int(len(frame))


def copy_png(source: Path, destination: Path) -> None:
    if source.suffix.lower() != ".png":
        raise ValueError(f"Only PNG images may be copied: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def rel_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def nonempty_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def canonical_task_id(value: Any) -> str:
    text = nonempty_text(value)
    return {"Matbench Steel-YS": "Matbench_Steel-YS"}.get(text, text)


def canonical_hybrid_method(value: Any) -> str:
    raw = nonempty_text(value)
    key = raw.lower().replace("-", "_").replace(" ", "_").replace("+", "_")
    key = key.replace("randcv", "random_cv")
    for prefix in ("hybridhigh20_", "hybrid_high20_", "hybrid_extrapolation_"):
        if key.startswith(prefix):
            key = key[len(prefix) :]
    mapping = {
        "random_cv": "RandCV",
        "random_cv_k5": "RandCV",
        "loco": "LOCO",
        "loco_k5": "LOCO",
        "sparse_x_cluster": "SX-cls",
        "sparsexcluster": "SX-cls",
        "sx_cls": "SX-cls",
        "sparse_x_single": "SX-sgl",
        "sparsexsingle": "SX-sgl",
        "sx_sgl": "SX-sgl",
        "sparse_y_cluster": "SY-cls",
        "sparseycluster": "SY-cls",
        "sy_cls": "SY-cls",
        "sparse_y_single": "SY-sgl",
        "sparseysingle": "SY-sgl",
        "sy_sgl": "SY-sgl",
    }
    return mapping.get(key, raw)


def prepare_published_hybrid_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "task_id" in result.columns:
        result["task_id"] = result["task_id"].map(canonical_task_id)
        result = result[~result["task_id"].isin(EXCLUDED_TASK_IDS)].copy()
    if "task" in result.columns:
        result["task"] = result["task"].map(canonical_task_id)
        result = result[~result["task"].isin(EXCLUDED_TASK_IDS)].copy()
    if "task_key" in result.columns:
        result = result[~result["task_key"].astype(str).isin(EXCLUDED_TASK_KEYS)].copy()
    if "method_short" in result.columns:
        result["method_short"] = result["method_short"].map(canonical_hybrid_method)
    if "method" in result.columns:
        result["method"] = result["method"].map(canonical_hybrid_method)
    for column in result.select_dtypes(include=["object", "string"]).columns:
        values = result[column].astype("string")
        values = values.str.replace(
            "hybrid_extrapolation_random_cv_k5",
            "hybrid_extrapolation_random_cv",
            regex=False,
        )
        values = values.mask(values.str.lower().eq("random_cv_k5"), "RandCV")
        result[column] = values
    return result.reset_index(drop=True)


def entry(
    *,
    analysis: str,
    scope: str,
    category: str,
    title: str,
    figure: Path,
    data: Path,
    output_root: Path,
    row_count: int,
    task_id: str = "",
    method: str = "",
    model: str = "",
    rep_space: str = "",
) -> dict[str, Any]:
    return {
        "analysis": analysis,
        "scope": scope,
        "category": category,
        "task_id": task_id,
        "method": method,
        "model": model,
        "rep_space": rep_space,
        "title": title,
        "figure_path": rel_posix(figure, output_root),
        "csv_path": rel_posix(data, output_root),
        "row_count": row_count,
    }


def panel_task_from_stem(stem: str) -> str:
    key = stem.removesuffix("_panel_distribution")
    if key not in PANEL_TASKS:
        raise KeyError(f"Unknown panel task stem: {stem}")
    return PANEL_TASKS[key]


def triptych_task_from_stem(stem: str) -> str:
    key = stem.removesuffix("_subset_comparison").removesuffix("_triptych_abcde")
    if key not in TRIPTYCH_TASKS:
        raise KeyError(f"Unknown triptych task stem: {stem}")
    return TRIPTYCH_TASKS[key]


def compact_panel_distribution(source: Path, output: Path, catalog: list[dict[str, Any]]) -> None:
    base = source / "panel_distribution"
    for image in sorted((base / "figure").glob("*.png")):
        task_id = panel_task_from_stem(image.stem)
        points_path = base / "csv" / f"{image.stem}_points.csv"
        frame = pd.read_csv(points_path, low_memory=False)
        columns = [column for column in PANEL_COLUMNS if column in frame.columns]
        compact = frame[columns].copy()
        figure_out = output / "panel_distribution" / "combined" / "figure" / task_id / "panel_distribution.png"
        csv_out = output / "panel_distribution" / "combined" / "csv" / task_id / "panel_distribution.csv"
        copy_png(image, figure_out)
        count = write_csv(compact, csv_out)
        catalog.append(
            entry(
                analysis="panel_distribution",
                scope="combined",
                category="panel_distribution",
                task_id=task_id,
                title=f"{task_id} | Panel distribution",
                figure=figure_out,
                data=csv_out,
                output_root=output,
                row_count=count,
            )
        )


def combine_triptych_tables(paths: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(paths):
        frame = pd.read_csv(path, low_memory=False)
        frame.insert(0, "source_table", path.stem)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No matching triptych tables were found.")
    return pd.concat(frames, ignore_index=True, sort=False)


def compact_triptych(source: Path, output: Path, catalog: list[dict[str, Any]]) -> None:
    root = source / "triptych"
    specs = [
        ("combined", "main", root / "figure", root / "csv"),
        ("combined", "subset_comparison", root / "subset_comparison" / "figure", root / "subset_comparison" / "csv"),
        (
            "test_extrapolation_high20",
            "main",
            root / "test_extrapolation_high20" / "figure",
            root / "test_extrapolation_high20" / "csv",
        ),
        ("test_inner_ood", "main", root / "test_inner_ood" / "figure", root / "test_inner_ood" / "csv"),
    ]
    for scope, category, image_dir, csv_dir in specs:
        for image in sorted(image_dir.glob("*.png")):
            task_id = triptych_task_from_stem(image.stem)
            sources = [path for path in csv_dir.glob(f"{image.stem}_*.csv") if "manifest" not in path.name]
            compact = combine_triptych_tables(sources)
            figure_out = output / "triptych" / scope / "figure" / category / task_id / "triptych.png"
            csv_out = output / "triptych" / scope / "csv" / category / task_id / "triptych.csv"
            copy_png(image, figure_out)
            count = write_csv(compact, csv_out)
            catalog.append(
                entry(
                    analysis="triptych",
                    scope=scope,
                    category=category,
                    task_id=task_id,
                    title=f"{task_id} | Triptych{'' if category == 'main' else ' subset comparison'}",
                    figure=figure_out,
                    data=csv_out,
                    output_root=output,
                    row_count=count,
                )
            )


def filter_positive_yz(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["Ry"] = pd.to_numeric(work["Ry"], errors="coerce")
    work["Rz"] = pd.to_numeric(work["Rz"], errors="coerce")
    return work[work["Ry"].gt(0) & work["Rz"].gt(0)].copy()


def compact_yz_severity(source: Path, output: Path, catalog: list[dict[str, Any]]) -> None:
    w_root = source / "w_error_relationship"
    for scope in SCOPES:
        yz_root = w_root / scope / "yz_severity_map"
        image_root = yz_root / "figures" / "yz_severity_map"
        csv_root = yz_root / "csv"
        main_summary = filter_positive_yz(pd.read_csv(csv_root / "hybrid_yz_severity_summary.csv", low_memory=False))
        set_ab_path = csv_root / "hybrid_set_ab_yz_severity_summary.csv"
        set_ab_summary = filter_positive_yz(pd.read_csv(set_ab_path, low_memory=False)) if set_ab_path.exists() else pd.DataFrame()
        main_summary["task_id"] = main_summary["task_id"].map(canonical_task_id)
        if not set_ab_summary.empty:
            set_ab_summary["task_id"] = set_ab_summary["task_id"].map(canonical_task_id)
        for image in sorted(image_root.rglob("*.png")):
            rel = image.relative_to(image_root)
            figure_out = output / "yz_severity_map" / scope / "figure" / rel
            csv_out = output / "yz_severity_map" / scope / "csv" / rel.with_suffix(".csv")
            category = rel.parts[0]
            task_id = ""
            method = "all_methods"
            title = f"{scope} | {task_id or image.stem} | YZ severity map"
            if category == "by_task":
                task_id = rel.parts[1]
                compact = main_summary[main_summary["task_id"].astype(str).eq(task_id)].copy()
                title = f"{scope} | {task_id} | YZ severity map"
            else:
                stem = image.stem
                if "setA_setB" in stem:
                    compact = set_ab_summary.copy()
                    category = "set_ab_comparison"
                else:
                    compact = main_summary.copy()
                if "core_protocols" in stem:
                    compact = compact[compact["method_short"].astype(str).isin(["RandCV", "LOCO"])].copy()
                    method = "RandCV+LOCO"
                if category == "set_ab_comparison":
                    protocol_label = "Core protocols" if method == "RandCV+LOCO" else "All methods"
                    title = f"{scope} | Set A vs Set B | B-RandCV denominator | {protocol_label}"
            if compact.empty:
                raise ValueError(f"No YZ data matched {image}")
            copy_png(image, figure_out)
            count = write_csv(compact, csv_out)
            catalog.append(
                entry(
                    analysis="yz_severity_map",
                    scope=scope,
                    category=category,
                    task_id=task_id,
                    method=method,
                    rep_space="Y/Z-space",
                    title=title,
                    figure=figure_out,
                    data=csv_out,
                    output_root=output,
                    row_count=count,
                )
            )


def manifest_relative_figure(value: str) -> Path:
    path = PureWindowsPath(str(value))
    parts = list(path.parts)
    lowered = [part.lower() for part in parts]
    indices = [index for index, part in enumerate(lowered) if part == "figures"]
    if not indices:
        raise ValueError(f"Manifest path has no figures segment: {value}")
    # The staging root itself may live below a parent directory named "Figures".
    # The final occurrence is the managed figure directory inside the W-error result.
    index = indices[-1]
    return Path(*parts[index + 1 :])


def compact_w_error(
    source: Path,
    output: Path,
    catalog: list[dict[str, Any]],
    *,
    scope_roots: dict[str, Path] | None = None,
) -> None:
    root = source / "w_error_relationship"
    for scope in SCOPES:
        scope_root = Path(scope_roots[scope]) if scope_roots and scope in scope_roots else root / scope
        manifest = prepare_published_hybrid_frame(
            pd.read_csv(scope_root / "csv" / "figure_manifest.csv", low_memory=False)
        )
        manifest = manifest[
            manifest["format"].astype(str).str.lower().eq("png")
            & manifest["figure_group"].astype(str).isin(["by_task_model", "by_method", "phase_diagram"])
        ].copy()

        pending: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for _, row in manifest.iterrows():
            source_rel = manifest_relative_figure(nonempty_text(row["figure"]))
            source_image = scope_root / "figures" / source_rel
            if not source_image.exists():
                raise FileNotFoundError(f"Manifest PNG is missing: {source_image}")
            target_rel = Path(*[("phase_diagram" if part == "phase_diagrams" else part) for part in source_rel.parts])
            figure_out = output / "w_error_relationship" / scope / "figure" / target_rel
            csv_out = output / "w_error_relationship" / scope / "csv" / target_rel.with_suffix(".csv")
            copy_png(source_image, figure_out)
            group = nonempty_text(row["figure_group"])
            task_id = nonempty_text(row["task_id"])
            model = nonempty_text(row["model"])
            method = nonempty_text(row["method_short"])
            rep_space = nonempty_text(row["rep_space"])
            if group == "by_task_model":
                key = (group, task_id, model)
            elif group == "by_method":
                key = (group, method, task_id, model)
            else:
                key = (group, task_id, method)
            pending.setdefault(key, []).append(
                {
                    "scope": scope,
                    "group": group,
                    "task_id": task_id,
                    "model": model,
                    "method": method,
                    "rep_space": rep_space,
                    "figure_out": figure_out,
                    "csv_out": csv_out,
                }
            )

        samples_path = scope_root / "csv" / "w_error_samples_long.csv"
        sample_header = pd.read_csv(samples_path, nrows=0).columns.tolist()
        sample_columns = [column for column in SCATTER_COLUMNS if column in sample_header]
        samples = pd.read_csv(samples_path, usecols=sample_columns, low_memory=False)
        samples = prepare_published_hybrid_frame(samples)
        samples["task_id"] = samples["task_id"].astype(str)
        samples["model"] = samples["model"].astype(str)
        samples["method_short"] = samples["method_short"].astype(str)

        written_keys: set[tuple[str, ...]] = set()
        for (task_id, model), group_frame in samples.groupby(["task_id", "model"], dropna=False, sort=False):
            key = ("by_task_model", str(task_id), str(model))
            for item in pending.get(key, []):
                compact_frame = group_frame[SCATTER_COLUMNS].copy()
                compact_frame["task_id"] = compact_frame["task_id"].map(canonical_task_id)
                count = write_csv(compact_frame, item["csv_out"])
                catalog.append(
                    entry(
                        analysis="w_error_relationship",
                        scope=scope,
                        category="by_task_model",
                        task_id=canonical_task_id(item["task_id"]),
                        model=item["model"],
                        title=f"{item['task_id']} | {item['model']} | W-error",
                        figure=item["figure_out"],
                        data=item["csv_out"],
                        output_root=output,
                        row_count=count,
                    )
                )
                written_keys.add(key)

        method_samples = samples[samples["method_short"].notna() & samples["method_short"].ne("nan")].copy()
        for (method, task_id, model), group_frame in method_samples.groupby(
            ["method_short", "task_id", "model"], dropna=False, sort=False
        ):
            key = ("by_method", str(method), str(task_id), str(model))
            for item in pending.get(key, []):
                compact_frame = group_frame[SCATTER_COLUMNS].copy()
                compact_frame["task_id"] = compact_frame["task_id"].map(canonical_task_id)
                count = write_csv(compact_frame, item["csv_out"])
                catalog.append(
                    entry(
                        analysis="w_error_relationship",
                        scope=scope,
                        category="by_method",
                        task_id=canonical_task_id(item["task_id"]),
                        method=item["method"],
                        model=item["model"],
                        title=f"{item['task_id']} | {item['model']} | {item['method']}",
                        figure=item["figure_out"],
                        data=item["csv_out"],
                        output_root=output,
                        row_count=count,
                    )
                )
                written_keys.add(key)
        del method_samples
        del samples

        phase_path = scope_root / "csv" / "phase_diagram_samples_wide.csv"
        phase_header = pd.read_csv(phase_path, nrows=0).columns.tolist()
        phase_columns = [column for column in [*PHASE_META_COLUMNS, "X_space_w", "Y_space_w", "Z_space_w"] if column in phase_header]
        phase = pd.read_csv(phase_path, usecols=phase_columns, low_memory=False)
        phase = prepare_published_hybrid_frame(phase)
        phase["task_id"] = phase["task_id"].astype(str)
        phase["method_short"] = phase["method_short"].astype(str)
        for (task_id, method), group_frame in phase.groupby(["task_id", "method_short"], dropna=False, sort=False):
            key = ("phase_diagram", str(task_id), str(method))
            for item in pending.get(key, []):
                rep_column = "X_space_w" if item["rep_space"] == "X-space" else "Z_space_w"
                columns = [*PHASE_META_COLUMNS, rep_column, "Y_space_w"]
                columns = list(dict.fromkeys(columns))
                compact = group_frame[columns].copy()
                compact["task_id"] = compact["task_id"].map(canonical_task_id)
                compact = compact[
                    pd.to_numeric(compact[rep_column], errors="coerce").notna()
                    & pd.to_numeric(compact["Y_space_w"], errors="coerce").notna()
                ]
                count = write_csv(compact, item["csv_out"])
                catalog.append(
                    entry(
                        analysis="w_error_relationship",
                        scope=scope,
                        category="phase_diagram",
                        task_id=canonical_task_id(item["task_id"]),
                        method=item["method"],
                        model="all_models",
                        rep_space=item["rep_space"],
                        title=f"{item['task_id']} | {item['method']} | {item['rep_space']} vs Y-space",
                        figure=item["figure_out"],
                        data=item["csv_out"],
                        output_root=output,
                        row_count=count,
                    )
                )
                written_keys.add(key)
        del phase

        expected_keys = set(pending)
        missing = sorted(expected_keys - written_keys)
        if missing:
            raise ValueError(f"No matching W-error rows for {scope}: {missing[:10]} (count={len(missing)})")

        summary_out = output / "w_error_relationship" / scope / "csv" / "summary"
        for filename in ("w_error_correlations.csv", "w_error_join_coverage.csv", "prediction_source_inventory.csv"):
            frame = prepare_published_hybrid_frame(
                pd.read_csv(scope_root / "csv" / filename, low_memory=False)
            )
            if filename == "prediction_source_inventory.csv":
                frame = frame.drop(columns=[column for column in ("prediction_file",) if column in frame.columns])
            write_csv(frame, summary_out / filename)


def write_catalog(catalog: list[dict[str, Any]], output: Path) -> None:
    frame = pd.DataFrame(catalog)
    frame = frame.sort_values(
        ["analysis", "scope", "category", "task_id", "method", "model", "rep_space", "figure_path"],
        kind="stable",
    ).reset_index(drop=True)
    frame.insert(0, "figure_id", [f"hybrid_{index + 1:05d}" for index in range(len(frame))])
    write_csv(frame, output / "figure_catalog.csv")

    columns: set[str] = set()
    for csv_path in sorted(output.rglob("*.csv")):
        if csv_path.name in {"figure_catalog.csv", "data_dictionary.csv"}:
            continue
        try:
            columns.update(pd.read_csv(csv_path, nrows=0).columns.tolist())
        except pd.errors.EmptyDataError:
            continue
    dictionary_rows = []
    for column in sorted(columns):
        description, unit = COLUMN_DESCRIPTIONS.get(column, ("Figure-specific data field retained from the plotting input.", ""))
        dictionary_rows.append({"column": column, "description": description, "unit": unit})
    write_csv(pd.DataFrame(dictionary_rows), output / "data_dictionary.csv")


def validate(output: Path) -> dict[str, Any]:
    catalog = pd.read_csv(output / "figure_catalog.csv", low_memory=False)
    errors: list[str] = []
    for row in catalog.itertuples(index=False):
        figure = output / row.figure_path
        data = output / row.csv_path
        if not figure.exists() or figure.suffix.lower() != ".png":
            errors.append(f"Missing/non-PNG figure: {row.figure_path}")
        if not data.exists() or data.suffix.lower() != ".csv":
            errors.append(f"Missing CSV: {row.csv_path}")
            continue
        actual_rows = sum(1 for _ in data.open("r", encoding="utf-8-sig", errors="replace")) - 1
        if actual_rows != int(row.row_count):
            errors.append(f"Row-count mismatch: {row.csv_path}: catalog={row.row_count}, actual={actual_rows}")
        if actual_rows <= 0:
            errors.append(f"Empty figure CSV: {row.csv_path}")

        analysis = nonempty_text(row.analysis)
        category = nonempty_text(row.category)
        expected_task = nonempty_text(row.task_id)
        expected_method = nonempty_text(row.method)
        expected_model = nonempty_text(row.model)
        header = pd.read_csv(data, nrows=0).columns.tolist()
        check_columns = [column for column in ("task_id", "method_short", "model", "source_table") if column in header]
        metadata = pd.read_csv(data, usecols=check_columns, low_memory=False) if check_columns else pd.DataFrame()

        if analysis == "panel_distribution":
            for required in ("projection_x", "projection_y", "method"):
                if required not in header:
                    errors.append(f"Panel CSV missing plotted column {required}: {row.csv_path}")
        elif analysis == "triptych":
            if "source_table" not in metadata.columns:
                errors.append(f"Triptych CSV has no source_table: {row.csv_path}")
            else:
                source_names = metadata["source_table"].dropna().astype(str).unique().tolist()
                task_keys = [key for key, value in TRIPTYCH_TASKS.items() if value == expected_task]
                if not task_keys or any(not any(name.startswith(key) for key in task_keys) for name in source_names):
                    errors.append(f"Triptych CSV contains a different task: {row.csv_path}")
        elif analysis == "yz_severity_map":
            if category == "by_task" and "task_id" in metadata.columns:
                values = set(metadata["task_id"].dropna().astype(str))
                if values != {expected_task}:
                    errors.append(f"YZ task mismatch: {row.csv_path}: {sorted(values)} != {expected_task}")
            if "core_protocols" in Path(row.figure_path).stem and "method_short" in metadata.columns:
                values = set(metadata["method_short"].dropna().astype(str))
                if not values or not values.issubset({"RandCV", "LOCO"}):
                    errors.append(f"YZ core-protocol mismatch: {row.csv_path}: {sorted(values)}")
        elif analysis == "w_error_relationship":
            if "task_id" in metadata.columns:
                values = set(metadata["task_id"].dropna().astype(str))
                if values != {expected_task}:
                    errors.append(f"W-error task mismatch: {row.csv_path}: {sorted(values)} != {expected_task}")
            if category in {"by_task_model", "by_method"} and "model" in metadata.columns:
                values = set(metadata["model"].dropna().astype(str))
                if values != {expected_model}:
                    errors.append(f"W-error model mismatch: {row.csv_path}: {sorted(values)} != {expected_model}")
            if category in {"by_method", "phase_diagram"} and "method_short" in metadata.columns:
                values = set(metadata["method_short"].dropna().astype(str))
                if values != {expected_method}:
                    errors.append(f"W-error method mismatch: {row.csv_path}: {sorted(values)} != {expected_method}")
            if category == "phase_diagram":
                rep_column = "X_space_w" if nonempty_text(row.rep_space) == "X-space" else "Z_space_w"
                if rep_column not in header or "Y_space_w" not in header:
                    errors.append(f"Phase CSV missing plotted axes: {row.csv_path}")

    pngs = [path for path in output.rglob("*.png")]
    forbidden = [path for path in output.rglob("*") if path.is_file() and path.suffix.lower() not in {".png", ".csv"}]
    catalog_figures = set(catalog["figure_path"].astype(str))
    actual_figures = {rel_posix(path, output) for path in pngs}
    missing_catalog = sorted(actual_figures - catalog_figures)
    missing_files = sorted(catalog_figures - actual_figures)
    if missing_catalog:
        errors.append(f"PNG files missing from catalog: {missing_catalog[:10]}")
    if missing_files:
        errors.append(f"Catalog figures missing from disk: {missing_files[:10]}")
    if forbidden:
        errors.append(f"Forbidden file types: {[rel_posix(path, output) for path in forbidden[:10]]}")
    if errors:
        raise RuntimeError("Validation failed:\n" + "\n".join(errors[:50]))
    return {
        "figure_count": int(len(pngs)),
        "catalog_count": int(len(catalog)),
        "csv_count": int(len(list(output.rglob("*.csv")))),
        "total_size_bytes": int(sum(path.stat().st_size for path in output.rglob("*") if path.is_file())),
    }


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if args.validate_only:
        result = validate(output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    ensure_clean_output(output)

    catalog: list[dict[str, Any]] = []
    compact_panel_distribution(source, output, catalog)
    compact_triptych(source, output, catalog)
    compact_yz_severity(source, output, catalog)
    # W-error is maintained as an independent compressed-data explorer and is
    # intentionally excluded from the static PNG/CSV catalog.
    write_catalog(catalog, output)
    result = validate(output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
