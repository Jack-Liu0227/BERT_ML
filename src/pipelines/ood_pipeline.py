"""
Unified OOD pipeline entrypoint with single-split and LOCO fold support.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from src.data_processing.ood_split_cache import (
    build_split_cache_dir,
    load_cached_split_manifest,
    save_multi_fold_split_cache,
    save_single_split_cache,
)
from src.data_processing.strength_ood_common import PreparedFold, PreparedSplit, save_json, save_prepared_split
from src.data_processing.strength_ood_registry import create_ood_processor, get_supported_split_strategies
from src.feature_engineering.feature_processor import FeatureProcessor
from src.feature_engineering.utils import set_seed
from src.LLMProp.ood_adapter import convert_split_to_llmprop_csv
from src.LLMProp.trainer import LLMPropTrainingConfig, run_llmprop_ood_training
from src.pipelines.batch_configs_ood import get_ood_method_meta
from src.pipelines.ood_train_eval import run_ood_training

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_SPLIT_STRATEGIES = get_supported_split_strategies()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="End-to-end OOD pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--target_column", type=str, required=True)
    parser.add_argument("--alloy_type", type=str, default=None)
    parser.add_argument("--processing_cols", type=str, nargs="*", default=[])
    parser.add_argument("--processing_text_column", type=str, default=None)

    parser.add_argument("--use_composition_feature", type=str2bool, default=False)
    parser.add_argument("--use_element_embedding", type=str2bool, default=False)
    parser.add_argument("--use_process_embedding", type=str2bool, default=False)
    parser.add_argument("--use_temperature", type=str2bool, default=False)
    parser.add_argument(
        "--embedding_type",
        type=str,
        required=True,
        choices=["tradition", "scibert", "steelbert", "matscibert"],
    )
    parser.add_argument("--use_llmprop", action="store_true", default=False)
    parser.add_argument("--split_cache_dir", type=str, default="output/ood_splits")
    parser.add_argument("--llmprop_epochs", type=int, default=200)
    parser.add_argument("--llmprop_batch_size", type=int, default=64)
    parser.add_argument("--llmprop_lr", type=float, default=1e-3)
    parser.add_argument("--llmprop_max_len", type=int, default=888)
    parser.add_argument("--llmprop_dropout", type=float, default=0.2)
    parser.add_argument("--llmprop_pooling", type=str, choices=["cls", "mean"], default="cls")
    parser.add_argument(
        "--llmprop_tokenizer",
        type=str,
        default="models/llmprop/tokenizers/t5_tokenizer_trained_on_modified_part_of_C4_and_textedge",
    )
    parser.add_argument("--llmprop_base_model", type=str, default="models/llmprop/google_t5_v1_1_small")
    parser.add_argument("--llmprop_valid_ratio", type=float, default=0.2)

    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        choices=["xgboost", "sklearn_rf", "mlp", "lightgbm", "catboost"],
    )
    parser.add_argument("--use_nn", action="store_true")

    parser.add_argument("--cross_validate", action="store_true", default=False)
    parser.add_argument("--num_folds", type=int, default=9)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--outer_test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--use_optuna", action="store_true", default=False)
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--mlp_max_iter", type=int, default=300)

    parser.add_argument("--evaluate_after_train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run_shap_analysis", action="store_true", default=False)

    parser.add_argument(
        "--split_strategy",
        type=str,
        default="target_extrapolation",
        choices=SUPPORTED_SPLIT_STRATEGIES,
    )
    parser.add_argument(
        "--extrapolation_side",
        type=str,
        default="low_to_high",
        choices=["low_to_high", "high_to_low"],
    )
    parser.add_argument("--sparse_candidate_pool_size", type=int, default=500)
    parser.add_argument("--sparse_cluster_count", type=int, default=5)
    parser.add_argument("--sparse_samples_per_cluster", type=int, default=1)
    parser.add_argument("--sparse_kde_bandwidth", type=float, default=None)
    parser.add_argument("--sparse_neighbors_per_seed", type=int, default=5)
    parser.add_argument("--loco_cluster_count", type=int, default=5)
    parser.add_argument("--baseline_num_folds", type=int, default=5)
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if args.use_llmprop:
        if args.use_nn or args.models:
            raise ValueError("Cannot specify --use_llmprop with --use_nn or --models")
    elif args.use_nn and args.models:
        raise ValueError("Cannot specify both --use_nn and --models")
    if not args.use_llmprop and not args.use_nn and not args.models:
        raise ValueError("Must specify either --use_nn or --models")
    if not Path(args.data_file).exists():
        raise FileNotFoundError(f"Data file not found: {args.data_file}")

    if args.split_strategy.endswith("_single"):
        if args.sparse_candidate_pool_size <= 0:
            raise ValueError("sparse_candidate_pool_size must be a positive integer")
        if args.sparse_cluster_count <= 0:
            raise ValueError("sparse_cluster_count must be a positive integer")
        if args.sparse_samples_per_cluster <= 0:
            raise ValueError("sparse_samples_per_cluster must be a positive integer")

    if args.split_strategy.endswith("_cluster"):
        if args.sparse_candidate_pool_size <= 0:
            raise ValueError("sparse_candidate_pool_size must be a positive integer")
        if args.sparse_cluster_count <= 0:
            raise ValueError("sparse_cluster_count must be a positive integer")
        if args.sparse_neighbors_per_seed <= 0:
            raise ValueError("sparse_neighbors_per_seed must be a positive integer")

    if args.split_strategy.startswith("hybrid_extrapolation_"):
        if not 0 < args.outer_test_size < 1:
            raise ValueError("outer_test_size must be between 0 and 1")
    if args.split_strategy in {"loco", "hybrid_extrapolation_loco"} and args.loco_cluster_count <= 0:
        raise ValueError("loco_cluster_count must be a positive integer")
    if args.split_strategy in {"random_cv_baseline", "hybrid_extrapolation_random_cv"} and args.baseline_num_folds <= 1:
        raise ValueError("baseline_num_folds must be greater than 1")


def _build_feature_dir(base_dir: Path, split_name: str, embedding_type: str) -> Path:
    return base_dir / split_name / embedding_type


def _run_feature_generation(
    data_file: str,
    feature_dir: Path,
    args: argparse.Namespace,
) -> str:
    feature_dir.mkdir(parents=True, exist_ok=True)
    log_dir = feature_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    processor = FeatureProcessor(
        data_path=data_file,
        use_process_embedding=args.use_process_embedding,
        use_element_embedding=args.use_element_embedding,
        use_composition_feature=args.use_composition_feature,
        standardize_features=False,
        use_temperature=args.use_temperature,
        feature_dir=str(feature_dir),
        log_dir=str(log_dir),
        target_columns=[args.target_column],
        model_name="steelbert" if args.embedding_type == "tradition" else args.embedding_type,
        other_features_name=args.processing_cols if args.processing_cols else None,
        processing_text_column=args.processing_text_column or "Processing_Description",
    )
    processor.process()
    return str(feature_dir / "features_with_id.csv")


def _feature_key(split_label: str) -> str:
    return split_label.replace(".csv", "")


def _clone_args_with_result_dir(args: argparse.Namespace, result_dir: Path) -> argparse.Namespace:
    cloned = argparse.Namespace(**vars(args))
    cloned.result_dir = str(result_dir)
    return cloned


def _build_method_params(args: argparse.Namespace) -> Dict[str, Any]:
    params: Dict[str, Any] = {"split_strategy": args.split_strategy}
    if args.split_strategy != "random_cv_baseline":
        params["test_size"] = args.test_size
    if args.split_strategy.startswith("hybrid_extrapolation_"):
        params["outer_test_size"] = args.outer_test_size
        params["inner_strategy"] = args.split_strategy.removeprefix("hybrid_extrapolation_")
    if args.split_strategy == "target_extrapolation":
        params["extrapolation_side"] = args.extrapolation_side
    if args.split_strategy.endswith("_single"):
        params.update(
            {
                "sparse_candidate_pool_size": args.sparse_candidate_pool_size,
                "sparse_cluster_count": args.sparse_cluster_count,
                "sparse_samples_per_cluster": args.sparse_samples_per_cluster,
                "sparse_kde_bandwidth": args.sparse_kde_bandwidth,
            }
        )
    if args.split_strategy.endswith("_cluster"):
        params.update(
            {
                "sparse_candidate_pool_size": args.sparse_candidate_pool_size,
                "sparse_cluster_count": args.sparse_cluster_count,
                "sparse_neighbors_per_seed": args.sparse_neighbors_per_seed,
            }
        )
    if args.split_strategy in {"loco", "hybrid_extrapolation_loco"}:
        params["loco_cluster_count"] = args.loco_cluster_count
    if args.split_strategy in {"random_cv_baseline", "hybrid_extrapolation_random_cv"}:
        params["baseline_num_folds"] = args.baseline_num_folds
    return params


def _resolve_split_cache_dir(args: argparse.Namespace) -> Path:
    return build_split_cache_dir(
        split_cache_root=args.split_cache_dir,
        alloy_type=args.alloy_type,
        data_file=args.data_file,
        target_column=args.target_column,
        method_name=args.split_strategy,
        method_params=_build_method_params(args),
        random_state=args.random_state,
    )


def _get_or_create_split_manifest(
    args: argparse.Namespace,
    prepared_result: PreparedSplit | List[PreparedFold],
) -> Dict[str, Any]:
    cache_dir = _resolve_split_cache_dir(args)
    method_params = _build_method_params(args)
    cached = load_cached_split_manifest(
        cache_dir,
        data_file=args.data_file,
        target_column=args.target_column,
        method_name=args.split_strategy,
        method_params=method_params,
        random_state=args.random_state,
    )
    if cached is not None:
        return cached

    if isinstance(prepared_result, PreparedSplit):
        return save_single_split_cache(
            cache_dir,
            prepared_result,
            data_file=args.data_file,
            target_column=args.target_column,
            method_name=args.split_strategy,
            method_params=method_params,
            random_state=args.random_state,
            alloy_type=args.alloy_type,
        )
    return save_multi_fold_split_cache(
        cache_dir,
        prepared_result,
        data_file=args.data_file,
        target_column=args.target_column,
        method_name=args.split_strategy,
        method_params=method_params,
        random_state=args.random_state,
        alloy_type=args.alloy_type,
    )


def _copy_cached_single_split(cache_entry: Dict[str, Any], run_root: Path) -> Dict[str, Any]:
    split_dir = run_root / "split_data"
    split_dir.mkdir(parents=True, exist_ok=True)
    train_src = Path(cache_entry["train_file"])
    test_src = Path(cache_entry["test_file"])
    summary_src = Path(cache_entry["summary_file"])
    train_dst = split_dir / train_src.name
    test_dst = split_dir / test_src.name
    summary_dst = split_dir / "split_summary.json"
    import shutil

    shutil.copy2(train_src, train_dst)
    shutil.copy2(test_src, test_dst)
    if summary_src.exists():
        shutil.copy2(summary_src, summary_dst)
    test_set_files: Dict[str, str] = {}
    for test_set_name, raw_path in cache_entry.get("test_set_files", {}).items():
        test_set_src = Path(raw_path)
        test_sets_dir = split_dir / "test_sets"
        test_sets_dir.mkdir(parents=True, exist_ok=True)
        test_set_dst = test_sets_dir / test_set_src.name
        shutil.copy2(test_set_src, test_set_dst)
        test_set_files[str(test_set_name)] = str(test_set_dst)
    artifacts = {
        "train_file": str(train_dst),
        "test_file": str(test_dst),
        "combined_test_file": str(test_dst),
        "summary_file": str(summary_dst),
        "canonical_train_file": str(train_src),
        "canonical_test_file": str(test_src),
    }
    if test_set_files:
        artifacts["test_set_files"] = test_set_files
    return artifacts


def _write_pipeline_manifest(
    args: argparse.Namespace,
    result_dir: Path,
    split_artifacts: Dict[str, str],
    feature_files: Dict[str, str],
    extra: Dict[str, Any] | None = None,
) -> None:
    manifest: Dict[str, Any] = {
        "data_file": args.data_file,
        "target_column": args.target_column,
        "method_params": _build_method_params(args),
        "feature_files": feature_files,
        "split_artifacts": split_artifacts,
    }
    if extra:
        manifest.update(extra)
    save_json(result_dir / "pipeline_manifest.json", manifest)


def _run_llmprop_single_split_flow(
    args: argparse.Namespace,
    split_entry: Dict[str, Any],
    run_root: Path,
    canonical_manifest: Dict[str, Any],
    fold_index: int | None = None,
    extra_manifest: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    split_artifacts = _copy_cached_single_split(split_entry, run_root)
    llmprop_data_dir = run_root / "llmprop_data"
    train_llmprop_csv = llmprop_data_dir / "train_llmprop.csv"
    test_llmprop_csv = llmprop_data_dir / "test_llmprop.csv"
    train_conversion = convert_split_to_llmprop_csv(
        split_artifacts["train_file"],
        train_llmprop_csv,
        args.target_column,
        processing_text_column=args.processing_text_column or "Processing_Description",
    )
    test_conversion = convert_split_to_llmprop_csv(
        split_artifacts["test_file"],
        test_llmprop_csv,
        args.target_column,
        processing_text_column=args.processing_text_column or "Processing_Description",
    )
    test_set_conversions: Dict[str, Any] = {}
    test_set_llmprop_csvs: Dict[str, str] = {}
    for test_set_name, test_set_csv in split_artifacts.get("test_set_files", {}).items():
        output_csv = llmprop_data_dir / "test_sets" / f"{test_set_name}_llmprop.csv"
        test_set_conversions[test_set_name] = convert_split_to_llmprop_csv(
            test_set_csv,
            output_csv,
            args.target_column,
            processing_text_column=args.processing_text_column or "Processing_Description",
        )
        test_set_llmprop_csvs[test_set_name] = str(output_csv)
    training_manifest = run_llmprop_ood_training(
        LLMPropTrainingConfig(
            result_dir=str(run_root),
            train_csv=str(train_llmprop_csv),
            test_csv=str(test_llmprop_csv),
            test_set_csvs=test_set_llmprop_csvs,
            target_column=args.target_column,
            base_model_path=args.llmprop_base_model,
            tokenizer_path=args.llmprop_tokenizer,
            epochs=args.llmprop_epochs,
            batch_size=args.llmprop_batch_size,
            learning_rate=args.llmprop_lr,
            max_len=args.llmprop_max_len,
            dropout=args.llmprop_dropout,
            pooling=args.llmprop_pooling,
            valid_ratio=args.llmprop_valid_ratio,
            random_state=args.random_state,
            split_manifest=canonical_manifest,
            fold_index=fold_index,
            ood_method=args.split_strategy,
            use_optuna=args.use_optuna,
            n_trials=args.n_trials,
        )
    )
    feature_files = {
        "train_llmprop": str(train_llmprop_csv),
        "test_llmprop": str(test_llmprop_csv),
    }
    for test_set_name, output_csv in test_set_llmprop_csvs.items():
        feature_files[f"test_set_{test_set_name}_llmprop"] = output_csv
    _write_pipeline_manifest(
        args=args,
        result_dir=run_root,
        split_artifacts=split_artifacts,
        feature_files=feature_files,
        extra={
            "canonical_split_manifest": canonical_manifest,
            "canonical_split_entry": split_entry,
            "llmprop_conversion": {"train": train_conversion, "test": test_conversion},
            "llmprop_test_set_conversions": test_set_conversions,
            "llmprop_manifest": training_manifest,
            **(extra_manifest or {}),
        },
    )
    return {
        "run_root": str(run_root),
        "split_artifacts": split_artifacts,
        "feature_files": feature_files,
        "llmprop_manifest": training_manifest,
    }


def _run_llmprop_flow(
    args: argparse.Namespace,
    prepared_result: PreparedSplit | List[PreparedFold],
    result_root: Path,
    canonical_manifest: Dict[str, Any],
    manifest_file_name: str,
) -> None:
    if not canonical_manifest.get("is_multi_fold"):
        split_entry = canonical_manifest["split"]
        prepared_split = prepared_result if isinstance(prepared_result, PreparedSplit) else prepared_result[0].split
        _run_llmprop_single_split_flow(
            args=args,
            split_entry=split_entry,
            run_root=result_root,
            canonical_manifest=canonical_manifest,
            extra_manifest={"split_summary": prepared_split.summary},
        )
        return

    fold_entries: List[Dict[str, Any]] = []
    for fold_entry in canonical_manifest.get("folds", []):
        fold_index = int(fold_entry["fold_index"])
        fold_root = result_root / "folds" / f"fold_{fold_index}"
        result = _run_llmprop_single_split_flow(
            args=args,
            split_entry=fold_entry,
            run_root=fold_root,
            canonical_manifest=canonical_manifest,
            fold_index=fold_index,
            extra_manifest={
                "fold_index": fold_index,
                "held_out_cluster_id": fold_entry.get("held_out_cluster_id"),
                "metadata": fold_entry.get("metadata", {}),
            },
        )
        fold_entries.append(
            {
                "fold_index": fold_index,
                "held_out_cluster_id": fold_entry.get("held_out_cluster_id"),
                "metadata": fold_entry.get("metadata", {}),
                **result,
            }
        )
    save_json(
        result_root / manifest_file_name,
        {
            "data_file": args.data_file,
            "target_column": args.target_column,
            "method_params": _build_method_params(args),
            "canonical_split_manifest": canonical_manifest,
            "fold_count": len(fold_entries),
            "folds": fold_entries,
        },
    )


def _run_single_split_flow(
    args: argparse.Namespace,
    prepared_split: PreparedSplit,
    run_root: Path,
    extra_manifest: Dict[str, Any] | None = None,
    cached_split_entry: Dict[str, Any] | None = None,
    canonical_manifest: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    split_dir = run_root / "split_data"
    if cached_split_entry is not None:
        split_artifacts = _copy_cached_single_split(cached_split_entry, run_root)
    else:
        split_artifacts = save_prepared_split(prepared_split, split_dir)

    features_root = run_root / "features"
    train_feature_file = _run_feature_generation(
        split_artifacts["train_file"],
        _build_feature_dir(features_root, prepared_split.train_label, args.embedding_type),
        args,
    )
    test_feature_file = _run_feature_generation(
        split_artifacts["test_file"],
        _build_feature_dir(features_root, prepared_split.test_label, args.embedding_type),
        args,
    )
    test_set_feature_files: Dict[str, str] = {}
    for test_set_name, test_set_csv in split_artifacts.get("test_set_files", {}).items():
        test_set_feature_files[test_set_name] = _run_feature_generation(
            test_set_csv,
            _build_feature_dir(features_root, test_set_name, args.embedding_type),
            args,
        )

    training_args = _clone_args_with_result_dir(args, run_root)
    run_ood_training(
        training_args,
        train_feature_file=train_feature_file,
        test_feature_file=test_feature_file,
        test_set_feature_files=test_set_feature_files,
    )

    feature_files = {
        _feature_key(prepared_split.train_label): train_feature_file,
        _feature_key(prepared_split.test_label): test_feature_file,
    }
    for test_set_name, test_set_feature_file in test_set_feature_files.items():
        feature_files[f"test_set_{test_set_name}"] = test_set_feature_file
    _write_pipeline_manifest(
        args=args,
        result_dir=run_root,
        split_artifacts=split_artifacts,
        feature_files=feature_files,
        extra={
            "split_summary": prepared_split.summary,
            "canonical_split_manifest": canonical_manifest,
            "canonical_split_entry": cached_split_entry,
            **(extra_manifest or {}),
        },
    )
    return {
        "run_root": str(run_root),
        "split_artifacts": split_artifacts,
        "feature_files": feature_files,
        "split_summary": prepared_split.summary,
    }


def _run_multi_fold_flow(
    args: argparse.Namespace,
    folds: List[PreparedFold],
    result_root: Path,
    manifest_file_name: str,
    canonical_manifest: Dict[str, Any] | None = None,
) -> None:
    fold_entries: List[Dict[str, Any]] = []
    cached_folds = {int(fold["fold_index"]): fold for fold in (canonical_manifest or {}).get("folds", [])}
    for prepared_fold in folds:
        fold_root = result_root / "folds" / f"fold_{prepared_fold.fold_index}"
        cached_entry = cached_folds.get(int(prepared_fold.fold_index))
        fold_result = _run_single_split_flow(
            args=args,
            prepared_split=prepared_fold.split,
            run_root=fold_root,
            cached_split_entry=cached_entry,
            canonical_manifest=canonical_manifest,
            extra_manifest={
                "fold_index": prepared_fold.fold_index,
                "held_out_cluster_id": prepared_fold.held_out_cluster_id,
            },
        )
        fold_entries.append(
            {
                "fold_index": prepared_fold.fold_index,
                "held_out_cluster_id": prepared_fold.held_out_cluster_id,
                "metadata": prepared_fold.metadata,
                **fold_result,
            }
        )

    multi_fold_manifest = {
        "data_file": args.data_file,
        "target_column": args.target_column,
        "method_params": _build_method_params(args),
        "canonical_split_manifest": canonical_manifest,
        "fold_count": len(fold_entries),
        "folds": fold_entries,
    }
    save_json(result_root / manifest_file_name, multi_fold_manifest)


def main() -> None:
    parser = create_argument_parser()
    args = parser.parse_args()

    validate_arguments(args)
    set_seed(args.random_state)

    processor = create_ood_processor(
        args.split_strategy,
        args.data_file,
        random_state=args.random_state,
        processing_cols=args.processing_cols,
    )
    raw_df = processor.load_data()
    prepared_result = processor.prepare(
        df=raw_df,
        target_col=args.target_column,
        test_ratio=args.test_size,
        outer_test_size=args.outer_test_size,
        extrapolation_side=args.extrapolation_side,
        sparse_candidate_pool_size=args.sparse_candidate_pool_size,
        sparse_cluster_count=args.sparse_cluster_count,
        sparse_samples_per_cluster=args.sparse_samples_per_cluster,
        sparse_kde_bandwidth=args.sparse_kde_bandwidth,
        sparse_neighbors_per_seed=args.sparse_neighbors_per_seed,
        loco_cluster_count=args.loco_cluster_count,
        baseline_num_folds=args.baseline_num_folds,
        processing_cols=args.processing_cols,
    )

    result_root = Path(args.result_dir)
    result_root.mkdir(parents=True, exist_ok=True)
    method_meta = get_ood_method_meta(args.split_strategy)
    canonical_manifest = _get_or_create_split_manifest(args, prepared_result)

    if args.use_llmprop:
        _run_llmprop_flow(
            args=args,
            prepared_result=prepared_result,
            result_root=result_root,
            canonical_manifest=canonical_manifest,
            manifest_file_name=str(method_meta["summary_file_name"]),
        )
        logger.info("OOD LLM-Prop pipeline completed")
        return

    if isinstance(prepared_result, PreparedSplit):
        _run_single_split_flow(
            args=args,
            prepared_split=prepared_result,
            run_root=result_root,
            cached_split_entry=canonical_manifest["split"],
            canonical_manifest=canonical_manifest,
        )
    else:
        _run_multi_fold_flow(
            args=args,
            folds=prepared_result,
            result_root=result_root,
            manifest_file_name=str(method_meta["summary_file_name"]),
            canonical_manifest=canonical_manifest,
        )

    logger.info("OOD pipeline completed")


if __name__ == "__main__":
    main()
