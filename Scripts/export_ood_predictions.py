"""
Export OOD prediction CSV files from extrapolation summary folders.

The script scans Traditional, BERT, and TabPFN summary directories, copies the
selected prediction CSV for each alloy case into OOD/<SourceFamily>/..., appends
source metadata columns, and writes a unified export manifest.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


CASES_DIRNAME = "01_alloy_cases"


@dataclass(frozen=True)
class SourceConfig:
    family: str
    summary_dir: Path
    expected_filename: str
    export_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export OOD prediction CSV files with source metadata."
    )
    parser.add_argument(
        "--mode",
        default="closest_mean",
        help=(
            "Prediction export mode for Traditional/BERT. "
            "Supported: final, closest_mean, global_mean. "
            "Aliases: mean -> closest_mean, global -> global_mean."
        ),
    )
    parser.add_argument(
        "--traditional-summary",
        type=Path,
        default=Path("output/extrapolation_results/all_traditional_extrapolation_summary"),
        help="Traditional extrapolation summary directory.",
    )
    parser.add_argument(
        "--bert-summary",
        type=Path,
        default=Path("output/extrapolation_results/all_bert_extrapolation_summary"),
        help="BERT extrapolation summary directory.",
    )
    parser.add_argument(
        "--tabpfn-summary",
        type=Path,
        default=Path("output/extrapolation_results_tabpfn/all_tabpfn_extrapolation_summary"),
        help="TabPFN extrapolation summary directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"D:\XJTU\已完成论文数据汇总\Fewshot\预处理汇总数据\OOD"),
        help="Root directory for exported OOD files.",
    )
    return parser.parse_args()


def normalize_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    alias_map = {
        "final": "final",
        "closest_mean": "closest_mean",
        "mean": "closest_mean",
        "global_mean": "global_mean",
        "global": "global_mean",
    }
    if normalized not in alias_map:
        valid_modes = ", ".join(["final", "closest_mean", "global_mean"])
        raise ValueError(f"Unsupported mode: {mode}. Valid modes: {valid_modes}")
    return alias_map[normalized]


def resolve_expected_filename(family: str, mode: str) -> tuple[str, str]:
    if family == "TabPFN":
        return "all_predictions.csv", "all_predictions"

    filename_map = {
        "final": "final_predictions.csv",
        "closest_mean": "closest_mean_predictions.csv",
        "global_mean": "global_mean_predictions.csv",
    }
    return filename_map[mode], mode


def iter_artifact_dirs(summary_dir: Path) -> Iterable[Path]:
    cases_root = summary_dir / CASES_DIRNAME
    if not cases_root.exists():
        return []
    return sorted(cases_root.rglob("selected_model_artifacts"))


def extract_case_parts(summary_dir: Path, artifacts_dir: Path) -> tuple[str, str, str]:
    relative_parts = artifacts_dir.relative_to(summary_dir / CASES_DIRNAME).parts
    if len(relative_parts) < 4:
        raise ValueError(f"Unexpected case layout: {artifacts_dir}")
    alloy_type, dataset_name, target = relative_parts[0], relative_parts[1], relative_parts[2]
    return alloy_type, dataset_name, target


def export_case_csv(
    config: SourceConfig,
    artifacts_dir: Path,
    output_dir: Path,
) -> dict | None:
    source_file = artifacts_dir / config.expected_filename
    if not source_file.exists():
        print(f"[WARN] Missing source file, skipped: {source_file}")
        return None

    try:
        alloy_type, dataset_name, target = extract_case_parts(config.summary_dir, artifacts_dir)
    except ValueError as exc:
        print(f"[WARN] {exc}")
        return None

    try:
        df = pd.read_csv(source_file)
    except Exception as exc:
        print(f"[WARN] Failed to read CSV, skipped: {source_file} ({exc})")
        return None

    export_df = df.copy()
    export_df["source_family"] = config.family
    export_df["alloy_type"] = alloy_type
    export_df["dataset_name"] = dataset_name
    export_df["target"] = target
    export_df["export_mode"] = config.export_mode
    export_df["source_file"] = str(source_file.resolve())

    export_file = output_dir / config.family / alloy_type / dataset_name / target / config.expected_filename
    export_file.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(export_file, index=False, encoding="utf-8-sig")

    return {
        "source_family": config.family,
        "alloy_type": alloy_type,
        "dataset_name": dataset_name,
        "target": target,
        "export_mode": config.export_mode,
        "source_file": str(source_file.resolve()),
        "export_file": str(export_file.resolve()),
    }


def export_source(config: SourceConfig, output_dir: Path) -> list[dict]:
    print(f"[INFO] Scanning {config.family}: {config.summary_dir}")
    if not config.summary_dir.exists():
        print(f"[WARN] Summary directory does not exist, skipped: {config.summary_dir}")
        return []

    manifest_rows: list[dict] = []
    artifacts_dirs = list(iter_artifact_dirs(config.summary_dir))
    if not artifacts_dirs:
        print(f"[WARN] No selected_model_artifacts directories found in: {config.summary_dir}")
        return []

    for artifacts_dir in artifacts_dirs:
        row = export_case_csv(config, artifacts_dir, output_dir)
        if row is not None:
            manifest_rows.append(row)

    print(f"[INFO] Exported {len(manifest_rows)} case(s) for {config.family}")
    return manifest_rows


def main() -> None:
    args = parse_args()
    mode = normalize_mode(args.mode)

    traditional_filename, traditional_mode = resolve_expected_filename("Traditional", mode)
    bert_filename, bert_mode = resolve_expected_filename("BERT", mode)
    tabpfn_filename, tabpfn_mode = resolve_expected_filename("TabPFN", mode)

    configs = [
        SourceConfig(
            family="Traditional",
            summary_dir=args.traditional_summary,
            expected_filename=traditional_filename,
            export_mode=traditional_mode,
        ),
        SourceConfig(
            family="BERT",
            summary_dir=args.bert_summary,
            expected_filename=bert_filename,
            export_mode=bert_mode,
        ),
        SourceConfig(
            family="TabPFN",
            summary_dir=args.tabpfn_summary,
            expected_filename=tabpfn_filename,
            export_mode=tabpfn_mode,
        ),
    ]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_manifest_rows: list[dict] = []
    for config in configs:
        all_manifest_rows.extend(export_source(config, output_dir))

    manifest_path = output_dir / "ood_export_manifest.csv"
    manifest_df = pd.DataFrame(all_manifest_rows)
    if not manifest_df.empty:
        manifest_df = manifest_df.sort_values(
            ["source_family", "alloy_type", "dataset_name", "target", "export_mode"]
        ).reset_index(drop=True)
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    print(f"[INFO] Total exported case(s): {len(all_manifest_rows)}")
    print(f"[INFO] Requested mode: {mode}")
    print("[INFO] TabPFN export mode is fixed to all_predictions")
    for family in ("Traditional", "BERT", "TabPFN"):
        family_count = sum(row["source_family"] == family for row in all_manifest_rows)
        print(f"[INFO] {family}: {family_count} case(s)")
    print(f"[INFO] Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
# python Scripts/export_ood_predictions.py --mode final
# python Scripts/export_ood_predictions.py --mode closest_mean
# python Scripts/export_ood_predictions.py --mode global_mean
