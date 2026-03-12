"""
Independent low-strength-train / high-strength-test extrapolation pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

from src.data_processing.strength_extrapolation_data_processor import StrengthExtrapolationDataProcessor
from src.feature_engineering.feature_processor import FeatureProcessor
from src.feature_engineering.utils import set_seed
from src.pipelines.extrapolation_train_eval import run_extrapolation_training

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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
        description="End-to-end extrapolation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--target_column", type=str, required=True)
    parser.add_argument("--target_columns", type=str, nargs="+", required=True)
    parser.add_argument("--processing_cols", type=str, nargs="*", default=[])
    parser.add_argument("--processing_text_column", type=str, default=None)
    parser.add_argument("--alloy_type", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)

    parser.add_argument("--use_composition_feature", type=str2bool, default=False)
    parser.add_argument("--use_element_embedding", type=str2bool, default=False)
    parser.add_argument("--use_process_embedding", type=str2bool, default=False)
    parser.add_argument("--use_temperature", type=str2bool, default=False)
    parser.add_argument("--embedding_type", type=str, required=True, choices=["tradition", "scibert", "steelbert", "matscibert"])

    parser.add_argument("--models", type=str, nargs="*", default=None, choices=["xgboost", "sklearn_rf", "mlp", "lightgbm", "catboost"])
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

    parser.add_argument("--evaluate_after_train", action="store_true", default=True)
    parser.add_argument("--run_shap_analysis", action="store_true", default=False)

    parser.add_argument("--split_strategy", type=str, default="target_extrapolation")
    parser.add_argument("--split_target_col", type=str, default=None)
    parser.add_argument("--extrapolation_side", type=str, default="low_to_high", choices=["low_to_high", "high_to_low"])
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if args.use_nn and args.models:
        raise ValueError("Cannot specify both --use_nn and --models")
    if not args.use_nn and not args.models:
        raise ValueError("Must specify either --use_nn or --models")
    if args.split_strategy != "target_extrapolation":
        raise ValueError("This pipeline only supports split_strategy='target_extrapolation'")
    if args.target_columns != [args.target_column]:
        raise ValueError("This extrapolation pipeline only supports a single target_column per run")
    if args.split_target_col and args.split_target_col != args.target_column:
        raise ValueError("split_target_col must equal target_column in the extrapolation pipeline")
    if not Path(args.data_file).exists():
        raise FileNotFoundError(f"Data file not found: {args.data_file}")


def _build_feature_dir(base_dir: str, split_name: str, args: argparse.Namespace) -> Path:
    return Path(base_dir) / split_name / args.embedding_type


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
        target_columns=args.target_columns,
        model_name="steelbert" if args.embedding_type == "tradition" else args.embedding_type,
        other_features_name=args.processing_cols if args.processing_cols else None,
        processing_text_column=args.processing_text_column or "Processing_Description",
    )
    processor.process()
    return str(feature_dir / "features_with_id.csv")


def _save_pipeline_manifest(args: argparse.Namespace, result_dir: Path, split_artifacts: Dict[str, str], feature_files: Dict[str, str]) -> None:
    manifest = {
        "data_file": args.data_file,
        "target_column": args.target_column,
        "split_strategy": args.split_strategy,
        "split_target_col": args.target_column,
        "extrapolation_side": args.extrapolation_side,
        "test_size": args.test_size,
        "feature_files": feature_files,
        "split_artifacts": split_artifacts,
    }
    (result_dir / "pipeline_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _infer_dataset_name(args: argparse.Namespace) -> str:
    if args.dataset_name:
        return args.dataset_name
    return Path(args.data_file).stem


def _infer_alloy_type(args: argparse.Namespace) -> str:
    if args.alloy_type:
        return args.alloy_type
    return Path(args.data_file).parent.name


def main() -> None:
    parser = create_argument_parser()
    args = parser.parse_args()

    validate_arguments(args)
    set_seed(args.random_state)

    split_target_col = args.target_column
    processor = StrengthExtrapolationDataProcessor(args.data_file, random_state=args.random_state)
    raw_df = processor.load_data()

    split_dir = Path(args.result_dir) / "split_data"
    train_df, test_df, summary = processor.split_low_high(
        df=raw_df,
        target_col=split_target_col,
        test_ratio=args.test_size,
        extrapolation_side=args.extrapolation_side,
    )
    split_artifacts = processor.save_split_artifacts(train_df, test_df, summary, str(split_dir))

    features_root = Path("Features_extrapolation") / _infer_alloy_type(args) / _infer_dataset_name(args) / args.target_column
    train_feature_file = _run_feature_generation(split_artifacts["train_file"], _build_feature_dir(str(features_root), "train_low", args), args)
    test_feature_file = _run_feature_generation(split_artifacts["test_file"], _build_feature_dir(str(features_root), "test_high", args), args)

    run_extrapolation_training(args, train_feature_file=train_feature_file, test_feature_file=test_feature_file)

    _save_pipeline_manifest(
        args=args,
        result_dir=Path(args.result_dir),
        split_artifacts=split_artifacts,
        feature_files={"train_low": train_feature_file, "test_high": test_feature_file},
    )
    logger.info("Extrapolation pipeline completed")


if __name__ == "__main__":
    main()
