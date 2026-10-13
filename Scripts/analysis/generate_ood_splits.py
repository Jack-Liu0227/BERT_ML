"""Generate canonical OOD split caches without running any training.

The OOD pipelines cache every split under::

    <split_cache_dir>/<alloy_type>/<dataset_name>/<target>/<method>/<hash>/split_manifest.json

That cache is shared by every model family (Traditional ML / BERT-NN / LLM-Prop)
for a given ``(alloy_type, dataset, target, method)`` tuple.  Generating the
splits up front lets the subsequent training runs use
``--require_existing_splits`` and removes any risk of two parallel training
workers racing on the same split-cache directory.

This module reuses the exact same code path as the training pipeline
(:func:`src.pipelines.ood_pipeline.prepare_pipeline_splits`) and the exact same
method parameters as the batch configs, so the cache keys are guaranteed to
match what training later looks up.

Usage (dry run first)::

    python -m Scripts.analysis.generate_ood_splits --alloys Al
    python -m Scripts.analysis.generate_ood_splits --alloys Al --execute
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.pipelines.batch_configs_ood import (  # noqa: E402
    BATCH_CONFIGS_OOD,
    COMMON_OOD_DEFAULTS,
    OOD_METHODS,
    get_alloy_config_ood,
    list_available_alloys_ood,
)
from src.pipelines.ood_pipeline import (  # noqa: E402
    _resolve_split_cache_dir,
    prepare_pipeline_splits,
)

# The split is independent of the model family, so we borrow the parameter set
# from the traditional-ML experiment template for every method.
SPLIT_TEMPLATE_NAME = "experiment1_all_ml_models"

# Mirrors the argparse defaults in src.pipelines.ood_pipeline so that split
# preparation never trips over a missing attribute for unrelated methods.
SPLIT_DEFAULTS: Dict[str, Any] = {
    "test_size": COMMON_OOD_DEFAULTS["test_size"],
    "outer_test_size": COMMON_OOD_DEFAULTS["outer_test_size"],
    "extrapolation_side": "low_to_high",
    "sparse_candidate_pool_size": 500,
    "sparse_cluster_count": 5,
    "sparse_samples_per_cluster": 1,
    "sparse_kde_bandwidth": None,
    "sparse_neighbors_per_seed": 5,
    "loco_cluster_count": 5,
    "baseline_num_folds": COMMON_OOD_DEFAULTS["baseline_num_folds"],
}

NON_HYBRID_METHODS: List[str] = [
    name for name, meta in OOD_METHODS.items() if not meta.get("is_hybrid")
]

METHOD_SUFFIX: Dict[str, str] = {
    name: meta["config_suffix"] for name, meta in OOD_METHODS.items()
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate canonical OOD split caches (no training). "
            "Defaults to a dry run; pass --execute to actually write the cache."
        )
    )
    parser.add_argument(
        "--alloys",
        nargs="*",
        default=None,
        help="Alloy/dataset keys to generate splits for (default: every OOD alloy).",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help="Optional target columns to restrict to (default: all configured targets).",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Optional OOD method names to restrict to (default: all non-hybrid methods).",
    )
    parser.add_argument(
        "--split_cache_dir",
        type=str,
        default=str(COMMON_OOD_DEFAULTS["split_cache_dir"]),
        help="Split cache root (default: output/ood_splits).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Actually create the split caches. Without this flag only a dry run is printed.",
    )
    return parser.parse_args()


def _resolve_methods(requested: List[str] | None) -> List[str]:
    if not requested:
        return list(NON_HYBRID_METHODS)
    unknown = [name for name in requested if name not in OOD_METHODS]
    if unknown:
        raise SystemExit(
            f"Unknown OOD method(s): {unknown}. "
            f"Known: {sorted(OOD_METHODS)}"
        )
    hybrid = [name for name in requested if OOD_METHODS[name].get("is_hybrid")]
    if hybrid:
        raise SystemExit(
            f"Hybrid methods are not supported by this helper: {hybrid}"
        )
    return list(requested)


def _resolve_alloys(requested: List[str] | None) -> List[str]:
    if not requested:
        excluded = set(COMMON_OOD_DEFAULTS.get("exclude_alloys", []))
        return [a for a in list_available_alloys_ood() if a not in excluded]
    return list(requested)


def _build_split_args(
    alloy_type: str,
    target_column: str,
    method_name: str,
    split_cache_dir: str,
) -> argparse.Namespace:
    alloy_config = get_alloy_config_ood(alloy_type)
    config_name = f"{SPLIT_TEMPLATE_NAME}_{METHOD_SUFFIX[method_name]}"
    template_config = BATCH_CONFIGS_OOD[config_name]

    namespace_kwargs: Dict[str, Any] = {
        **SPLIT_DEFAULTS,
        **template_config,
        "alloy_type": alloy_type,
        "data_file": alloy_config["raw_data"],
        "target_column": target_column,
        "processing_cols": list(alloy_config.get("processing_cols") or []),
        "processing_text_column": alloy_config.get("processing_text_column"),
        "split_cache_dir": split_cache_dir,
        "batch_config_name": config_name,
        "require_existing_splits": False,
    }
    return argparse.Namespace(**namespace_kwargs)


def _manifest_path(args: argparse.Namespace) -> Path:
    return _resolve_split_cache_dir(args) / "split_manifest.json"


def main() -> None:
    parsed = parse_args()
    alloys = _resolve_alloys(parsed.alloys)
    methods = _resolve_methods(parsed.methods)
    target_filter = set(parsed.targets) if parsed.targets else None

    mode = "EXECUTE" if parsed.execute else "DRY-RUN"
    print(f"[{mode}] split cache root: {parsed.split_cache_dir}")
    print(f"[{mode}] alloys: {alloys}")
    print(f"[{mode}] methods: {methods}")
    print()

    created = 0
    existed = 0
    total = 0
    failures: List[str] = []

    for alloy_type in alloys:
        alloy_config = get_alloy_config_ood(alloy_type)
        targets = list(alloy_config.get("targets") or [])
        if target_filter is not None:
            targets = [t for t in targets if t in target_filter]
        for target_column in targets:
            for method_name in methods:
                total += 1
                args = _build_split_args(
                    alloy_type, target_column, method_name, parsed.split_cache_dir
                )
                manifest_path = _manifest_path(args)
                label = f"{alloy_type} | {target_column} | {method_name}"
                if manifest_path.exists():
                    existed += 1
                    print(f"  [cached ] {label}\n            -> {manifest_path}")
                    continue
                if not parsed.execute:
                    print(f"  [todo   ] {label}\n            -> {manifest_path}")
                    continue
                try:
                    _prepared, manifest = prepare_pipeline_splits(args)
                    payload = json.dumps(manifest, ensure_ascii=False)[:200]
                    created += 1
                    print(f"  [created] {label}\n            -> {manifest_path}")
                    print(f"            manifest(head): {payload}")
                except Exception as exc:  # noqa: BLE001 - report and continue
                    failures.append(f"{label}: {exc!r}")
                    print(f"  [FAILED ] {label}\n            -> {exc!r}")

    print()
    print(
        f"[{mode}] summary: total={total} created={created} "
        f"already_cached={existed} failed={len(failures)}"
    )
    if failures:
        print("Failures:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
