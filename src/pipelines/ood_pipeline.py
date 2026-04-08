"""
Unified OOD pipeline entrypoint with single-split and LOCO fold support.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from src.data_processing.strength_ood_common import PreparedFold, PreparedSplit, save_json, save_prepared_split
from src.data_processing.strength_ood_registry import create_ood_processor, get_supported_split_strategies
from src.feature_engineering.feature_processor import FeatureProcessor
from src.feature_engineering.utils import set_seed
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
    parser.add_argument("--sparse_cluster_count", type=int, default=50)
    parser.add_argument("--sparse_samples_per_cluster", type=int, default=1)
    parser.add_argument("--sparse_kde_bandwidth", type=float, default=None)
    parser.add_argument("--sparse_neighbors_per_seed", type=int, default=5)
    parser.add_argument("--loco_cluster_count", type=int, default=5)
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if args.use_nn and args.models:
        raise ValueError("Cannot specify both --use_nn and --models")
    if not args.use_nn and not args.models:
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

    if args.split_strategy == "loco" and args.loco_cluster_count <= 0:
        raise ValueError("loco_cluster_count must be a positive integer")


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
    params: Dict[str, Any] = {
        "split_strategy": args.split_strategy,
        "test_size": args.test_size,
    }
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
    if args.split_strategy == "loco":
        params["loco_cluster_count"] = args.loco_cluster_count
    return params


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


def _run_single_split_flow(
    args: argparse.Namespace,
    prepared_split: PreparedSplit,
    run_root: Path,
    extra_manifest: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    split_dir = run_root / "split_data"
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

    training_args = _clone_args_with_result_dir(args, run_root)
    run_ood_training(
        training_args,
        train_feature_file=train_feature_file,
        test_feature_file=test_feature_file,
    )

    feature_files = {
        _feature_key(prepared_split.train_label): train_feature_file,
        _feature_key(prepared_split.test_label): test_feature_file,
    }
    _write_pipeline_manifest(
        args=args,
        result_dir=run_root,
        split_artifacts=split_artifacts,
        feature_files=feature_files,
        extra={
            "split_summary": prepared_split.summary,
            **(extra_manifest or {}),
        },
    )
    return {
        "run_root": str(run_root),
        "split_artifacts": split_artifacts,
        "feature_files": feature_files,
        "split_summary": prepared_split.summary,
    }


def _run_loco_flow(args: argparse.Namespace, folds: List[PreparedFold], result_root: Path) -> None:
    fold_entries: List[Dict[str, Any]] = []
    for prepared_fold in folds:
        fold_root = result_root / "folds" / f"fold_{prepared_fold.fold_index}"
        fold_result = _run_single_split_flow(
            args=args,
            prepared_split=prepared_fold.split,
            run_root=fold_root,
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

    loco_manifest = {
        "data_file": args.data_file,
        "target_column": args.target_column,
        "method_params": _build_method_params(args),
        "fold_count": len(fold_entries),
        "folds": fold_entries,
    }
    save_json(result_root / "loco_manifest.json", loco_manifest)


def main() -> None:
    parser = create_argument_parser()
    args = parser.parse_args()

    validate_arguments(args)
    set_seed(args.random_state)

    processor = create_ood_processor(args.split_strategy, args.data_file, random_state=args.random_state)
    raw_df = processor.load_data()
    prepared_result = processor.prepare(
        df=raw_df,
        target_col=args.target_column,
        test_ratio=args.test_size,
        extrapolation_side=args.extrapolation_side,
        sparse_candidate_pool_size=args.sparse_candidate_pool_size,
        sparse_cluster_count=args.sparse_cluster_count,
        sparse_samples_per_cluster=args.sparse_samples_per_cluster,
        sparse_kde_bandwidth=args.sparse_kde_bandwidth,
        sparse_neighbors_per_seed=args.sparse_neighbors_per_seed,
        loco_cluster_count=args.loco_cluster_count,
    )

    result_root = Path(args.result_dir)
    result_root.mkdir(parents=True, exist_ok=True)

    if isinstance(prepared_result, PreparedSplit):
        _run_single_split_flow(args=args, prepared_split=prepared_result, run_root=result_root)
    else:
        _run_loco_flow(args=args, folds=prepared_result, result_root=result_root)

    logger.info("OOD pipeline completed")


if __name__ == "__main__":
    main()
