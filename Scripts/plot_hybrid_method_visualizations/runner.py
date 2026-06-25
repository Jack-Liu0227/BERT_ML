from __future__ import annotations

import sys
from pathlib import Path

from .config import (
    DELTA_COLUMNS,
    MANIFEST_COLUMNS,
    PURE_HYBRID_COLUMNS,
    SET_AB_AUDIT_COLUMNS,
    SET_AB_FAMILY_BEST_COLUMNS,
    SET_AB_MASTER_COLUMNS,
    SET_AB_MODEL_SUMMARY_COLUMNS,
    SUMMARY_COLUMNS,
    TRIPTYCH_MODEL_REVIEW_COLUMNS,
    YZ_OOD_MAP_COLUMNS,
)
from .figures import (
    generate_overview_figures,
    generate_task_figures,
)
from .io import load_hybrid_long_tables, load_pure_long_table, parse_args, write_csv, write_report
from .summaries import (
    build_ab_delta_summary,
    build_hybrid_summary_long,
    build_pure_vs_hybrid_summary,
    build_set_ab_family_best_summary,
    build_set_ab_master_sample_long,
    build_set_ab_model_summary,
    build_set_ab_source_audit,
    build_set_ab_space_summary_long,
    build_triptych_model_review,
    build_yz_ood_map_summary,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def remove_stale_managed_outputs(output_dir: Path) -> None:
    for path in (output_dir / "figures").glob("figure_*.*"):
        if path.is_file():
            path.unlink()
    for path in (output_dir / "figures").rglob("figure_6_pure_vs_hybrid_disentanglement.*"):
        if path.is_file():
            path.unlink()
    for path in [
        output_dir / "csv" / "canonical_family_model_ab_summary_review.csv",
        output_dir / "csv" / "canonical_family_best_ab_summary_review.csv",
    ]:
        if path.is_file():
            path.unlink()


def run_hybrid_visualizations(
    *,
    hybrid_root: Path,
    output_dir: Path,
    pure_root: Path | None,
    formats: list[str],
    dpi: int,
    triptych_root: Path | None = None,
    case_contains: str | None = None,
    overview_only: bool = False,
    task_figures_only: bool = False,
    skip_report: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_stale_managed_outputs(output_dir)
    hybrid_frame = load_hybrid_long_tables(hybrid_root, case_contains=case_contains)
    pure_frame = load_pure_long_table(pure_root, case_contains=case_contains)
    master = build_set_ab_master_sample_long(hybrid_frame)
    triptych_review = build_triptych_model_review(triptych_root, case_contains=case_contains)
    model_summary = build_set_ab_model_summary(master, triptych_review)
    summary = build_set_ab_space_summary_long(master, model_summary)
    old_summary = build_hybrid_summary_long(hybrid_frame)
    family_best = build_set_ab_family_best_summary(model_summary)
    audit = build_set_ab_source_audit(model_summary, triptych_review, old_summary)
    delta = build_ab_delta_summary(summary)
    yz_summary = build_yz_ood_map_summary(summary)
    pure_hybrid = build_pure_vs_hybrid_summary(summary, pure_frame)
    write_csv(master, output_dir / "csv" / "set_ab_master_sample_long.csv", SET_AB_MASTER_COLUMNS)
    write_csv(model_summary, output_dir / "csv" / "set_ab_model_summary.csv", SET_AB_MODEL_SUMMARY_COLUMNS)
    write_csv(summary, output_dir / "csv" / "set_ab_space_summary_long.csv", SUMMARY_COLUMNS)
    write_csv(triptych_review, output_dir / "csv" / "set_ab_triptych_aligned_model_review.csv", TRIPTYCH_MODEL_REVIEW_COLUMNS)
    write_csv(family_best, output_dir / "csv" / "set_ab_family_best_summary.csv", SET_AB_FAMILY_BEST_COLUMNS)
    write_csv(audit, output_dir / "csv" / "set_ab_source_audit.csv", SET_AB_AUDIT_COLUMNS)
    for set_label, group in master.groupby("set_label", dropna=False):
        subset_fs = str(group["subset_fs"].dropna().iloc[0]) if "subset_fs" in group.columns and group["subset_fs"].notna().any() else str(set_label).replace(" ", "_")
        set_dir = output_dir / "data" / subset_fs
        write_csv(group, set_dir / "set_ab_master_sample_long.csv", SET_AB_MASTER_COLUMNS)
        set_model = model_summary[model_summary["set_label"].astype(str).eq(str(set_label))]
        write_csv(set_model, set_dir / "set_ab_model_summary.csv", SET_AB_MODEL_SUMMARY_COLUMNS)
        set_space = summary[summary["subset"].astype(str).eq(str(set_label))]
        write_csv(set_space, set_dir / "set_ab_space_summary_long.csv", SUMMARY_COLUMNS)
        set_yz = yz_summary[yz_summary["subset"].astype(str).eq(str(set_label))]
        write_csv(set_yz, set_dir / "hybrid_yz_ood_map_summary.csv", YZ_OOD_MAP_COLUMNS)
    write_csv(summary, output_dir / "csv" / "hybrid_visualization_summary_long.csv", SUMMARY_COLUMNS)
    write_csv(delta, output_dir / "csv" / "hybrid_ab_delta_summary.csv", DELTA_COLUMNS)
    write_csv(yz_summary, output_dir / "csv" / "hybrid_set_ab_yz_ood_map_summary.csv", YZ_OOD_MAP_COLUMNS)
    write_csv(pure_hybrid, output_dir / "csv" / "pure_vs_hybrid_summary.csv", PURE_HYBRID_COLUMNS)

    manifest_rows: list[dict[str, object]] = []
    if not task_figures_only:
        manifest_rows.extend(generate_overview_figures(summary, delta, pure_hybrid, yz_summary, output_dir, formats, dpi))
    if not overview_only:
        manifest_rows.extend(generate_task_figures(summary, delta, pure_hybrid, yz_summary, output_dir, formats, dpi))
    import pandas as pd

    manifest = pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS)
    write_csv(manifest, output_dir / "csv" / "figure_manifest.csv", MANIFEST_COLUMNS)
    if not skip_report:
        write_report(output_dir, summary, delta, pure_hybrid, manifest, audit=audit)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_hybrid_visualizations(
        hybrid_root=Path(args.hybrid_root),
        pure_root=Path(args.pure_root) if args.pure_root else None,
        triptych_root=Path(args.triptych_root) if args.triptych_root else None,
        output_dir=Path(args.output_dir),
        formats=list(args.formats),
        dpi=int(args.dpi),
        case_contains=args.case_contains,
        overview_only=bool(args.overview_only),
        task_figures_only=bool(args.task_figures_only),
        skip_report=bool(args.skip_report),
    )
    print(f"Saved hybrid method-aware visualizations to: {args.output_dir}")
