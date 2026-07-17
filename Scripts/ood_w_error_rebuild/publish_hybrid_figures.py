from __future__ import annotations

import argparse
import shutil
import json
from pathlib import Path

import numpy as np
import pandas as pd

from Scripts.analysis import export_three_space_w_ood_report as three_space
from Scripts.figures import plot_w_error_relationships as plotter

from .common import REPO_ROOT, frame_join_ids, normalize_id, read_csv, write_csv, write_json

DEFAULT_FULL_ROOT = REPO_ROOT / "output" / "w_error_relationship_full"
DEFAULT_EMBEDDING_DATA_DIR = (
    Path("D:/XJTU")
    / "已完成论文数据汇总"
    / "Fewshot"
    / "预处理汇总数据"
    / "OOD"
    / "embedding_data"
)
DEFAULT_HYBRID_FINAL_ROOT = (
    Path("D:/XJTU")
    / "已完成论文数据汇总"
    / "Fewshot"
    / "预处理汇总数据"
    / "最终结果图"
    / "OOD HYBIRD"
    / "w_error_relationship"
)


def hybrid_csv_dir(full_root: Path, subset: str) -> Path:
    return full_root / "OOD_HYBIRD" / subset / "csv"


def ood_csv_dir(full_root: Path) -> Path:
    return full_root / "OOD" / "csv"


def paths_match(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve()


def clean_managed_figure_outputs(output_dir: Path) -> None:
    for child in ["figures", "yz_severity_map"]:
        path = output_dir / child
        if path.exists():
            shutil.rmtree(path)


def clean_yz_severity_output(output_dir: Path) -> None:
    path = output_dir / "yz_severity_map"
    if path.exists():
        shutil.rmtree(path)


def property_to_target_col(value: object) -> str:
    text = str(value)
    if "UTS" in text.upper():
        return "UTS(MPa)"
    if "YS" in text.upper() or "yield" in text.lower():
        return "YS(MPa)"
    if text.lower().startswith("el") or "elong" in text.lower():
        return "El(%)"
    return text


def target_col_for_source(inventory_row: pd.Series, fallback_property: object) -> str:
    summary_file = Path(str(inventory_row.get("split_summary_file", "")))
    if summary_file.exists():
        try:
            payload = json.loads(summary_file.read_text(encoding="utf-8-sig"))
            target = str(payload.get("split_target_col") or payload.get("target_column") or "").strip()
            if target:
                return target
        except Exception:
            pass
    return property_to_target_col(fallback_property)


def ensure_plot_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["scope"] = result.get("ood_scope", "hybrid")
    result["method_short"] = result["method"].map(plotter.normalize_hybrid_method)
    result["model_family"] = result.get("model_family", result["model"])
    result["task_id"] = [
        plotter.task_id_from_values(row["alloy_family"], row["property"])
        for _, row in result.iterrows()
    ]
    result["sample_order"] = (
        result.groupby(["source_split_dir", "test_set"], dropna=False, sort=False).cumcount()
    )
    result["join_id"] = result["ID"].map(normalize_id)
    return result


def x_space_joined(samples: pd.DataFrame) -> pd.DataFrame:
    x = samples.copy()
    x["space"] = "X-space"
    x["sample_w_contribution"] = pd.to_numeric(x["wx_sample_score"], errors="coerce")
    x["sample_w_mass_contribution"] = pd.to_numeric(x["wx_mass_contribution"], errors="coerce")
    x["split_w"] = pd.to_numeric(x["wx"], errors="coerce")
    x["target_value"] = pd.to_numeric(x["target_value"], errors="coerce")
    return x


def inventory_by_source(w_inventory: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    rows: dict[tuple[str, str], pd.Series] = {}
    for _, row in w_inventory.iterrows():
        rows[(str(row["source_split_dir"]), str(row["test_set"]))] = row
    return rows


def load_source_frames(row: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_file = Path(str(row["train_file"]))
    test_file = Path(str(row["test_file"]))
    return read_csv(train_file), read_csv(test_file)


def z_arrays_for_source(
    train: pd.DataFrame,
    test: pd.DataFrame,
    background: pd.DataFrame | None,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    if background is None:
        return None, None, "missing_embedding_file"
    if "__id_key__" in background.columns and "ID" in train.columns and "ID" in test.columns:
        train_keys = pd.DataFrame({"__id_key__": pd.to_numeric(train["ID"], errors="coerce")})
        test_keys = pd.DataFrame({"__id_key__": pd.to_numeric(test["ID"], errors="coerce")})
        train_z = train_keys.merge(background, on="__id_key__", how="left", sort=False)
        test_z = test_keys.merge(background, on="__id_key__", how="left", sort=False)
    elif (
        "__source_index_key__" in background.columns
        and "__source_index__" in train.columns
        and "__source_index__" in test.columns
    ):
        train_keys = pd.DataFrame({"__source_index_key__": pd.to_numeric(train["__source_index__"], errors="coerce")})
        test_keys = pd.DataFrame({"__source_index_key__": pd.to_numeric(test["__source_index__"], errors="coerce")})
        train_z = train_keys.merge(background, on="__source_index_key__", how="left", sort=False)
        test_z = test_keys.merge(background, on="__source_index_key__", how="left", sort=False)
    else:
        return None, None, "missing_embedding_join_key"
    if train_z[["x", "y"]].isna().any().any() or test_z[["x", "y"]].isna().any().any():
        return None, None, "missing_embedding_rows"
    return (*three_space.standardized_z_arrays(train_z, test_z), "ok")


def yz_rows_for_source(
    source_samples: pd.DataFrame,
    inventory_row: pd.Series,
    backgrounds: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    first = source_samples.iloc[0]
    train, test = load_source_frames(inventory_row)
    target_col = target_col_for_source(inventory_row, first["property"])
    test = test.copy()
    test["join_id"] = frame_join_ids(test)

    ordered_ids = source_samples[["ID", "join_id"]].drop_duplicates("join_id", keep="first")
    base = ordered_ids.merge(test, on="join_id", how="left", suffixes=("", "_test"), sort=False)

    rows: list[pd.DataFrame] = []
    y_train, y_test = three_space.standardized_target_arrays(train, base, target_col)
    y_split_w, y_scores, y_mass = three_space.one_dimensional_w1_sample_contributions(y_train, y_test)
    y = ordered_ids.copy()
    y["source_split_dir"] = first["source_split_dir"]
    y["test_set"] = first["test_set"]
    y["space"] = "Y-space"
    y["sample_w_contribution"] = y_scores
    y["sample_w_mass_contribution"] = y_mass
    y["split_w"] = y_split_w
    y["w_status"] = "ok" if np.isfinite(y_split_w) else "invalid_target_distribution"
    rows.append(y)

    case_key = three_space.case_key_from_values(first["alloy_family"], first["dataset_name"], first["property"])
    z_train, z_test, z_status = z_arrays_for_source(train, base, backgrounds.get(case_key))
    z = ordered_ids.copy()
    z["source_split_dir"] = first["source_split_dir"]
    z["test_set"] = first["test_set"]
    z["space"] = "Z-space"
    if z_train is None or z_test is None:
        z["sample_w_contribution"] = np.nan
        z["sample_w_mass_contribution"] = np.nan
        z["split_w"] = np.nan
        z["w_status"] = z_status
    else:
        z_split_w, z_scores, z_mass = three_space.nd_w1_sample_contributions(z_train, z_test)
        z["sample_w_contribution"] = z_scores
        z["sample_w_mass_contribution"] = z_mass
        z["split_w"] = z_split_w
        z["w_status"] = "ok" if np.isfinite(z_split_w) else "ot_failed"
    rows.append(z)
    return pd.concat(rows, ignore_index=True, sort=False)


def yz_space_joined(samples: pd.DataFrame, w_inventory: pd.DataFrame, embedding_data_dir: Path) -> pd.DataFrame:
    backgrounds = three_space.load_embedding_backgrounds(embedding_data_dir)
    inventory = inventory_by_source(w_inventory)
    yz_source_rows: list[pd.DataFrame] = []
    for key, group in samples.groupby(["source_split_dir", "test_set"], dropna=False, sort=False):
        inventory_row = inventory.get((str(key[0]), str(key[1])))
        if inventory_row is None:
            continue
        yz_source_rows.append(yz_rows_for_source(group, inventory_row, backgrounds))
    yz_values = pd.concat(yz_source_rows, ignore_index=True, sort=False) if yz_source_rows else pd.DataFrame()
    if yz_values.empty:
        return pd.DataFrame(columns=samples.columns)
    meta_cols = [column for column in samples.columns if column not in {"space", "sample_w_contribution", "sample_w_mass_contribution", "split_w"}]
    merged = samples[meta_cols].merge(
        yz_values,
        on=["source_split_dir", "test_set", "join_id"],
        how="inner",
        suffixes=("", "_w"),
        sort=False,
    )
    if "ID_w" in merged.columns:
        merged = merged.drop(columns=["ID_w"])
    return merged


def build_three_space_joined(samples: pd.DataFrame, w_inventory: pd.DataFrame, embedding_data_dir: Path) -> pd.DataFrame:
    samples = ensure_plot_metadata(samples)
    x = x_space_joined(samples)
    yz = yz_space_joined(samples, w_inventory, embedding_data_dir)
    joined = pd.concat([x, yz], ignore_index=True, sort=False)
    for column in plotter.JOINED_OUTPUT_COLUMNS:
        if column not in joined.columns:
            joined[column] = np.nan
    return joined[plotter.JOINED_OUTPUT_COLUMNS].copy()


def write_publish_tables(
    *,
    input_csv_dir: Path,
    output_dir: Path,
    joined: pd.DataFrame,
    phase_wide: pd.DataFrame,
    figures: pd.DataFrame,
) -> None:
    output_csv_dir = output_dir / "csv"
    if paths_match(input_csv_dir, output_csv_dir):
        write_csv(joined, output_csv_dir / "three_space_w_error_samples_long.csv")
    else:
        write_csv(joined, output_csv_dir / "w_error_samples_long.csv")
    write_csv(phase_wide, output_csv_dir / "phase_diagram_samples_wide.csv")
    write_csv(figures, output_csv_dir / "figure_manifest.csv")


def build_and_write_figures(
    *,
    samples: pd.DataFrame,
    w_inventory: pd.DataFrame,
    input_csv_dir: Path,
    output_dir: Path,
    embedding_data_dir: Path,
    formats: list[str],
    dpi: int,
    include_hybrid_yz_severity: bool,
    include_standard_yz_severity: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clean_managed_figure_outputs(output_dir)
    joined = build_three_space_joined(samples, w_inventory, embedding_data_dir)
    phase_wide = plotter.build_phase_diagram_samples_wide(joined.drop(columns=["__w_row_id"], errors="ignore"))
    figures = pd.concat(
        [
            plotter.plot_task_model_figures(joined, output_dir, formats, dpi),
            plotter.plot_method_figures(joined, output_dir, formats, dpi),
            plotter.plot_phase_diagram_figures(phase_wide, output_dir, formats, dpi),
        ],
        ignore_index=True,
        sort=False,
    )
    if include_hybrid_yz_severity:
        yz_figures = plotter.write_hybrid_yz_severity_outputs(joined, output_dir, formats, dpi)
        figures = pd.concat([figures, yz_figures], ignore_index=True, sort=False)
    if include_standard_yz_severity:
        yz_figures = plotter.write_standard_yz_severity_outputs(joined, output_dir, formats, dpi)
        figures = pd.concat([figures, yz_figures], ignore_index=True, sort=False)
    write_publish_tables(
        input_csv_dir=input_csv_dir,
        output_dir=output_dir,
        joined=joined,
        phase_wide=phase_wide,
        figures=figures,
    )
    return joined, phase_wide, figures


def read_subset_three_space_joined(
    *,
    full_root: Path,
    subset: str,
    embedding_data_dir: Path,
    max_rows: int | None = None,
) -> pd.DataFrame:
    csv_dir = hybrid_csv_dir(full_root, subset)
    samples = read_csv(csv_dir / "w_error_samples_long.csv")
    if max_rows is not None:
        samples = samples.head(max_rows).copy()
    w_inventory = read_csv(csv_dir / "w_source_inventory.csv")
    return build_three_space_joined(samples, w_inventory, embedding_data_dir)


def publish_ood(
    *,
    full_root: Path,
    output_dir: Path,
    embedding_data_dir: Path,
    formats: list[str],
    dpi: int,
    max_rows: int | None = None,
) -> dict[str, object]:
    csv_dir = ood_csv_dir(full_root)
    samples = read_csv(csv_dir / "w_error_samples_long.csv")
    if max_rows is not None:
        samples = samples.head(max_rows).copy()
    w_inventory = read_csv(csv_dir / "w_source_inventory.csv")
    joined, _phase_wide, figures = build_and_write_figures(
        samples=samples,
        w_inventory=w_inventory,
        input_csv_dir=csv_dir,
        output_dir=output_dir,
        embedding_data_dir=embedding_data_dir,
        formats=formats,
        dpi=dpi,
        include_hybrid_yz_severity=False,
        include_standard_yz_severity=True,
    )
    audit = {
        "status": "ok",
        "scope": "ood",
        "input_rows": int(len(samples)),
        "three_space_rows": int(len(joined)),
        "figures": int(len(figures)),
        "models": sorted(str(value) for value in joined["model"].dropna().unique()),
        "spaces": sorted(str(value) for value in joined["space"].dropna().unique()),
        "embedding_data_dir": str(embedding_data_dir),
    }
    write_json(audit, output_dir / "csv" / "publish_audit_summary.json")
    return audit


def publish_ood_yz_severity(
    *,
    full_root: Path,
    output_dir: Path,
    embedding_data_dir: Path,
    formats: list[str],
    dpi: int,
    max_rows: int | None = None,
) -> dict[str, object]:
    clean_yz_severity_output(output_dir)
    csv_dir = ood_csv_dir(full_root)
    samples = read_csv(csv_dir / "w_error_samples_long.csv")
    if max_rows is not None:
        samples = samples.head(max_rows).copy()
    w_inventory = read_csv(csv_dir / "w_source_inventory.csv")
    joined = build_three_space_joined(samples, w_inventory, embedding_data_dir)
    figures = plotter.write_standard_yz_severity_outputs(joined, output_dir, formats, dpi)
    audit = {
        "status": "ok",
        "scope": "ood",
        "mode": "yz_severity_only",
        "input_rows": int(len(samples)),
        "three_space_rows": int(len(joined)),
        "yz_severity_figures": int(len(figures)),
        "spaces": sorted(str(value) for value in joined["space"].dropna().unique()),
        "embedding_data_dir": str(embedding_data_dir),
    }
    write_json(audit, output_dir / "yz_severity_map" / "csv" / "publish_yz_severity_audit_summary.json")
    return audit


def publish_hybrid_subset(
    *,
    full_root: Path,
    output_root: Path,
    subset: str,
    embedding_data_dir: Path,
    formats: list[str],
    dpi: int,
    max_rows: int | None = None,
) -> dict[str, object]:
    csv_dir = hybrid_csv_dir(full_root, subset)
    samples = read_csv(csv_dir / "w_error_samples_long.csv")
    if max_rows is not None:
        samples = samples.head(max_rows).copy()
    w_inventory = read_csv(csv_dir / "w_source_inventory.csv")
    output_dir = output_root / subset
    joined, _phase_wide, figures = build_and_write_figures(
        samples=samples,
        w_inventory=w_inventory,
        input_csv_dir=csv_dir,
        output_dir=output_dir,
        embedding_data_dir=embedding_data_dir,
        formats=formats,
        dpi=dpi,
        include_hybrid_yz_severity=True,
    )
    set_ab_figures = pd.DataFrame(columns=plotter.FIGURE_MANIFEST_COLUMNS)
    if subset == "combined":
        set_a_joined = read_subset_three_space_joined(
            full_root=full_root,
            subset="test_extrapolation_high20",
            embedding_data_dir=embedding_data_dir,
            max_rows=max_rows,
        )
        set_b_joined = read_subset_three_space_joined(
            full_root=full_root,
            subset="test_inner_ood",
            embedding_data_dir=embedding_data_dir,
            max_rows=max_rows,
        )
        set_ab_figures = plotter.write_hybrid_set_ab_yz_random80_outputs(
            set_a_joined,
            set_b_joined,
            output_dir / "yz_severity_map",
            formats,
            dpi,
            random80_w_table=set_b_joined,
        )
        figures = pd.concat([figures, set_ab_figures], ignore_index=True, sort=False)
        write_csv(figures, output_dir / "csv" / "figure_manifest.csv")
    audit = {
        "status": "ok",
        "subset": subset,
        "input_rows": int(len(samples)),
        "three_space_rows": int(len(joined)),
        "figures": int(len(figures)),
        "set_ab_yz_figures": int(len(set_ab_figures)),
        "models": sorted(str(value) for value in joined["model"].dropna().unique()),
        "spaces": sorted(str(value) for value in joined["space"].dropna().unique()),
        "embedding_data_dir": str(embedding_data_dir),
    }
    write_json(audit, output_dir / "csv" / "publish_audit_summary.json")
    return audit


def publish_hybrid_yz_severity(
    *,
    full_root: Path,
    output_root: Path,
    subset: str,
    embedding_data_dir: Path,
    formats: list[str],
    dpi: int,
    max_rows: int | None = None,
) -> dict[str, object]:
    output_dir = output_root / subset
    clean_yz_severity_output(output_dir)
    combined_joined = read_subset_three_space_joined(
        full_root=full_root,
        subset=subset,
        embedding_data_dir=embedding_data_dir,
        max_rows=max_rows,
    )
    figures = plotter.write_hybrid_yz_severity_outputs(combined_joined, output_dir, formats, dpi)
    set_ab_figures = pd.DataFrame(columns=plotter.FIGURE_MANIFEST_COLUMNS)
    if subset == "combined":
        set_a_joined = read_subset_three_space_joined(
            full_root=full_root,
            subset="test_extrapolation_high20",
            embedding_data_dir=embedding_data_dir,
            max_rows=max_rows,
        )
        set_b_joined = read_subset_three_space_joined(
            full_root=full_root,
            subset="test_inner_ood",
            embedding_data_dir=embedding_data_dir,
            max_rows=max_rows,
        )
        set_ab_figures = plotter.write_hybrid_set_ab_yz_random80_outputs(
            set_a_joined,
            set_b_joined,
            output_dir / "yz_severity_map",
            formats,
            dpi,
            random80_w_table=set_b_joined,
        )
        figures = pd.concat([figures, set_ab_figures], ignore_index=True, sort=False)
    audit = {
        "status": "ok",
        "scope": "hybrid",
        "subset": subset,
        "mode": "yz_severity_only",
        "combined_three_space_rows": int(len(combined_joined)),
        "yz_severity_figures": int(len(figures)),
        "set_ab_yz_figures": int(len(set_ab_figures)),
        "spaces": sorted(str(value) for value in combined_joined["space"].dropna().unique()),
        "embedding_data_dir": str(embedding_data_dir),
    }
    write_json(audit, output_dir / "yz_severity_map" / "csv" / "publish_yz_severity_audit_summary.json")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish OOD/Hybrid W-error figures from result-local full CSVs.")
    parser.add_argument("--scope", choices=["ood", "hybrid"], default="hybrid")
    parser.add_argument("--full-root", type=Path, default=DEFAULT_FULL_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--subset", choices=["combined"], default="combined")
    parser.add_argument("--embedding-data-dir", type=Path, default=DEFAULT_EMBEDDING_DATA_DIR)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--only-yz-severity", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scope == "ood":
        if args.only_yz_severity:
            audit = publish_ood_yz_severity(
                full_root=args.full_root,
                output_dir=args.output_root or plotter.DEFAULT_OOD_OUTPUT,
                embedding_data_dir=args.embedding_data_dir,
                formats=args.formats,
                dpi=args.dpi,
                max_rows=args.max_rows,
            )
        else:
            audit = publish_ood(
                full_root=args.full_root,
                output_dir=args.output_root or plotter.DEFAULT_OOD_OUTPUT,
                embedding_data_dir=args.embedding_data_dir,
                formats=args.formats,
                dpi=args.dpi,
                max_rows=args.max_rows,
            )
    else:
        if args.only_yz_severity:
            audit = publish_hybrid_yz_severity(
                full_root=args.full_root,
                output_root=args.output_root or DEFAULT_HYBRID_FINAL_ROOT,
                subset=args.subset,
                embedding_data_dir=args.embedding_data_dir,
                formats=args.formats,
                dpi=args.dpi,
                max_rows=args.max_rows,
            )
        else:
            audit = publish_hybrid_subset(
                full_root=args.full_root,
                output_root=args.output_root or DEFAULT_HYBRID_FINAL_ROOT,
                subset=args.subset,
                embedding_data_dir=args.embedding_data_dir,
                formats=args.formats,
                dpi=args.dpi,
                max_rows=args.max_rows,
            )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
