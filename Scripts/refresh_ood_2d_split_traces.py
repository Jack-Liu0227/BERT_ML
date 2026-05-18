from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_processing.strength_ood_common import PreparedFold, PreparedSplit, save_prepared_split  # noqa: E402
from src.data_processing.strength_ood_registry import create_ood_processor  # noqa: E402
from src.pipelines.batch_configs_ood import ALLOY_CONFIGS_OOD, OOD_METHODS  # noqa: E402


METHOD_LAYOUT = [
    "random_cv_baseline",
    "loco",
    "target_extrapolation",
    "sparse_x_single",
    "sparse_y_single",
    "sparse_x_cluster",
    "sparse_y_cluster",
]


def dataset_name(raw_data: str) -> str:
    name = Path(raw_data).stem
    for suffix in ["_Processed_cleaned", "_with_ID", "_withID", "_cleaned", "_processed", "_Processed"]:
        name = name.replace(suffix, "")
    return name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate only split_data/trace artifacts needed by Scripts/plot_ood_2d_maps.py. "
            "This does not retrain any OOD model."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=REPO_ROOT / "output" / "ood_results",
        help="OOD result root containing experiment1_all_ml_models_* directories.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "output" / "ood_2d_maps" / "experiment1_all_ml_models" / "csv" / "trace_refresh_manifest.csv",
        help="CSV manifest recording refreshed split trace directories.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List planned trace writes without modifying files.")
    parser.add_argument(
        "--exclude-standalone-steel-ys",
        action="store_true",
        help="Skip Steel/steel/YS(MPa). Final paper plots exclude it, but default refreshes it too.",
    )
    return parser.parse_args()


def split_kwargs(method: str, alloy_cfg: dict[str, Any]) -> dict[str, Any]:
    processing_cols = alloy_cfg.get("processing_cols") or []
    return {
        "test_ratio": 0.2,
        "test_size": 0.2,
        "extrapolation_side": "low_to_high",
        "sparse_candidate_pool_size": 500,
        "sparse_cluster_count": 5,
        "sparse_samples_per_cluster": 1,
        "sparse_kde_bandwidth": None,
        "sparse_neighbors_per_seed": 5,
        "loco_cluster_count": 5,
        "baseline_num_folds": 5,
        "processing_cols": processing_cols,
    }


def result_method_root(base_dir: Path, method: str, alloy: str, dataset: str, target: str) -> Path:
    meta = OOD_METHODS[method]
    config_dir = f"experiment1_all_ml_models_{meta['config_suffix']}"
    # Historical BERT_ML OOD map discovery uses HEA_half for target extrapolation
    # and HEA for the other six methods.  Keep that layout so the existing
    # plot_ood_2d_maps.py can rebuild all cases without introducing duplicate
    # HEA/HEA_half panel filenames downstream.
    alloy_dir = "HEA_half" if method == "target_extrapolation" and alloy == "HEA" else alloy
    return base_dir / config_dir / alloy_dir / dataset / target / str(meta["result_dir_suffix"]) / "tradition"


def remove_existing_split_data(root: Path) -> None:
    split_dir = root / "split_data"
    if split_dir.exists():
        shutil.rmtree(split_dir)


def write_prepared(prepared: PreparedSplit | list[PreparedFold], root: Path, *, dry_run: bool) -> list[Path]:
    written: list[Path] = []
    if isinstance(prepared, PreparedSplit):
        split_dir = root / "split_data"
        if not dry_run:
            remove_existing_split_data(root)
            save_prepared_split(prepared, split_dir)
        written.append(split_dir)
        return written

    folds_root = root / "folds"
    for fold in prepared:
        fold_root = folds_root / f"fold_{fold.fold_index}"
        split_dir = fold_root / "split_data"
        if not dry_run:
            remove_existing_split_data(fold_root)
            save_prepared_split(fold.split, split_dir)
        written.append(split_dir)
    return written


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir if args.base_dir.is_absolute() else REPO_ROOT / args.base_dir
    rows: list[dict[str, str]] = []
    started_at = datetime.now().isoformat(timespec="seconds")

    for alloy, alloy_cfg in ALLOY_CONFIGS_OOD.items():
        if alloy == "HEA_corrosion":
            continue
        raw_path = REPO_ROOT / alloy_cfg["raw_data"]
        df = pd.read_csv(raw_path)
        dataset = dataset_name(alloy_cfg["raw_data"])
        targets = alloy_cfg.get("targets") or []
        for target in targets:
            if args.exclude_standalone_steel_ys and alloy == "Steel" and target == "YS(MPa)":
                continue
            for method in METHOD_LAYOUT:
                processor = create_ood_processor(
                    method,
                    str(raw_path),
                    random_state=42,
                    processing_cols=alloy_cfg.get("processing_cols") or [],
                )
                prepared = processor.prepare(
                    df=df,
                    target_col=target,
                    **split_kwargs(method, alloy_cfg),
                )
                root = result_method_root(base_dir, method, alloy, dataset, target)
                written_dirs = write_prepared(prepared, root, dry_run=args.dry_run)
                rows.append(
                    {
                        "started_at": started_at,
                        "dry_run": str(bool(args.dry_run)),
                        "method": method,
                        "alloy": alloy,
                        "dataset": dataset,
                        "target": target,
                        "result_root": str(root),
                        "split_dirs": " | ".join(str(path) for path in written_dirs),
                        "split_dir_count": str(len(written_dirs)),
                    }
                )
                print(f"[{'DRY' if args.dry_run else 'OK'}] {method}: {alloy}/{dataset}/{target} -> {len(written_dirs)} split dir(s)")

    if not args.dry_run:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "started_at",
                    "dry_run",
                    "method",
                    "alloy",
                    "dataset",
                    "target",
                    "result_root",
                    "split_dirs",
                    "split_dir_count",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        metadata_path = args.manifest.with_suffix(".json")
        metadata_path.write_text(
            json.dumps(
                {
                    "started_at": started_at,
                    "completed_at": datetime.now().isoformat(timespec="seconds"),
                    "python": sys.executable,
                    "base_dir": str(base_dir),
                    "row_count": len(rows),
                    "note": "Only split_data/trace artifacts were regenerated; no ML model training was run.",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
