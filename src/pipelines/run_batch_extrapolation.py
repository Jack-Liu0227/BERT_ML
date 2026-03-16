"""
Independent batch runner for extrapolation experiments.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.feature_engineering.utils import set_seed
from src.pipelines.batch_configs_extrapolation import (
    ALLOY_CONFIGS_EXTRAPOLATION,
    BATCH_CONFIGS_EXTRAPOLATION,
    get_alloy_config_extrapolation,
    list_available_alloys_extrapolation,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class ProgressManager:
    def __init__(self, progress_file: str = ".batch_progress_extrapolation.json") -> None:
        self.progress_file = Path(progress_file)
        self.progress_data = self._load_progress()

    def _load_progress(self) -> Dict[str, Any]:
        if self.progress_file.exists():
            try:
                return json.loads(self.progress_file.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning(f"Failed to load progress file: {exc}")
        return {}

    def _save_progress(self) -> None:
        self.progress_file.write_text(
            json.dumps(self.progress_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def get_config_progress(self, config_name: str) -> Dict[str, str]:
        return self.progress_data.get(config_name, {})

    def is_task_completed(self, config_name: str, task_key: str) -> bool:
        return self.get_config_progress(config_name).get(task_key) == "success"

    def update_task_status(self, config_name: str, task_key: str, status: str) -> None:
        self.progress_data.setdefault(config_name, {})[task_key] = status
        self._save_progress()

    def clear_progress(self, config_name: Optional[str] = None) -> None:
        if config_name is None:
            self.progress_data = {}
        else:
            self.progress_data.pop(config_name, None)
        self._save_progress()

    def show_progress(self, config_name: Optional[str] = None) -> None:
        items = self.progress_data if config_name is None else {config_name: self.progress_data.get(config_name, {})}
        if not items or all(not value for value in items.values()):
            logger.info("No extrapolation progress records")
            return
        logger.info("=" * 100)
        logger.info("Extrapolation Task Progress")
        logger.info("=" * 100)
        for cfg_name, tasks in items.items():
            if not tasks:
                continue
            success = sum(1 for status in tasks.values() if status == "success")
            failed = sum(1 for status in tasks.values() if status == "failed")
            total = len(tasks)
            logger.info(f"{cfg_name}: success={success}, failed={failed}, total={total}")
            for task_key, status in sorted(tasks.items()):
                logger.info(f"  - {task_key}: {status}")


def make_task_key(alloy_type: str, target_column: str, model_name: Optional[str] = None) -> str:
    if model_name:
        return f"{alloy_type}_{target_column}_{model_name}"
    return f"{alloy_type}_{target_column}"


def _dataset_name_from_config(raw_data: str) -> str:
    dataset_name = Path(raw_data).stem
    for suffix in ["_Processed_cleaned", "_with_ID", "_withID", "_cleaned", "_processed", "_Processed"]:
        dataset_name = dataset_name.replace(suffix, "")
    return dataset_name


def build_command(alloy_type: str, alloy_config: Dict[str, Any], args: Any, model_name: Optional[str] = None) -> List[str]:
    dataset_name = _dataset_name_from_config(alloy_config["raw_data"])
    target_column = alloy_config["target_column"]
    result_dir = Path("output") / "extrapolation_results" / alloy_type / dataset_name / target_column / args.embedding_type

    cmd = [
        sys.executable,
        "-m",
        "src.pipelines.end_to_end_extrapolation_pipeline",
        "--data_file",
        alloy_config["raw_data"],
        "--result_dir",
        str(result_dir),
        "--target_column",
        target_column,
        "--target_columns",
        target_column,
        "--embedding_type",
        args.embedding_type,
        "--alloy_type",
        alloy_type,
        "--dataset_name",
        dataset_name,
        "--split_strategy",
        args.split_strategy,
        "--split_target_col",
        target_column,
        "--extrapolation_side",
        args.extrapolation_side,
        "--test_size",
        str(args.test_size),
        "--random_state",
        str(args.random_state),
    ]

    processing_cols = alloy_config.get("processing_cols") or []
    if processing_cols:
        cmd.extend(["--processing_cols", *processing_cols])

    processing_text_column = alloy_config.get("processing_text_column")
    if processing_text_column:
        cmd.extend(["--processing_text_column", processing_text_column])

    if args.use_composition_feature:
        cmd.extend(["--use_composition_feature", "True"])
    if args.use_element_embedding:
        cmd.extend(["--use_element_embedding", "True"])
    if args.use_process_embedding and processing_text_column:
        cmd.extend(["--use_process_embedding", "True"])
    if args.use_temperature:
        cmd.extend(["--use_temperature", "True"])

    if args.use_nn:
        cmd.append("--use_nn")
        cmd.extend(["--epochs", str(args.epochs), "--patience", str(args.patience), "--batch_size", str(args.batch_size)])
    elif model_name:
        cmd.extend(["--models", model_name, "--mlp_max_iter", str(args.mlp_max_iter)])
    else:
        cmd.extend(["--models", *args.models, "--mlp_max_iter", str(args.mlp_max_iter)])

    if args.cross_validate:
        cmd.extend(["--cross_validate", "--num_folds", str(args.num_folds)])
    if args.use_optuna:
        cmd.extend(["--use_optuna", "--n_trials", str(args.n_trials)])
    if args.evaluate_after_train:
        cmd.append("--evaluate_after_train")
    if args.run_shap_analysis:
        cmd.append("--run_shap_analysis")

    return cmd


def format_command_for_display(cmd: List[str]) -> str:
    parts = []
    for part in cmd:
        if any(char in part for char in [" ", "(", ")", "%", "/", "\\", "℃"]):
            parts.append(f'"{part}"')
        else:
            parts.append(part)
    return " ".join(parts)


def run_single_task(
    config_name: str,
    alloy_type: str,
    alloy_config: Dict[str, Any],
    args: Any,
    progress_manager: ProgressManager,
    dry_run: bool,
    model_name: Optional[str] = None,
) -> str:
    task_key = make_task_key(alloy_type, alloy_config["target_column"], model_name)
    cmd = build_command(alloy_type, alloy_config, args, model_name)
    logger.info(f"[RUN] {task_key}")
    logger.info(format_command_for_display(cmd))

    if dry_run:
        return "dry_run"

    progress_manager.update_task_status(config_name, task_key, "running")
    result = subprocess.run(cmd, text=True)
    status = "success" if result.returncode == 0 else "failed"
    progress_manager.update_task_status(config_name, task_key, status)
    return status


def resolve_alloy_types(config: Dict[str, Any]) -> List[str]:
    alloy_types = config.get("alloy_types")
    if alloy_types is None:
        alloy_types = list_available_alloys_extrapolation()
    excluded = set(config.get("exclude_alloys", []))
    return [alloy for alloy in alloy_types if alloy not in excluded]


def run_batch_config(
    config_name: str,
    config: Dict[str, Any],
    dry_run: bool,
    progress_manager: ProgressManager,
    resume: bool,
) -> Dict[str, str]:
    args = argparse.Namespace(**config)
    if not hasattr(args, "random_state"):
        args.random_state = 42

    results: Dict[str, str] = {}
    for alloy_type in resolve_alloy_types(config):
        alloy_base_config = get_alloy_config_extrapolation(alloy_type)
        targets = alloy_base_config.get("targets") or []
        if not targets:
            raise ValueError(f"No targets configured for extrapolation alloy: {alloy_type}")

        for target_column in targets:
            alloy_config = alloy_base_config.copy()
            alloy_config["target_column"] = target_column

            if args.use_nn:
                task_key = make_task_key(alloy_type, target_column)
                if resume and progress_manager.is_task_completed(config_name, task_key):
                    results[task_key] = "skipped"
                    continue
                results[task_key] = run_single_task(
                    config_name,
                    alloy_type,
                    alloy_config,
                    args,
                    progress_manager,
                    dry_run,
                )
                continue

            for model_name in args.models:
                task_key = make_task_key(alloy_type, target_column, model_name)
                if resume and progress_manager.is_task_completed(config_name, task_key):
                    results[task_key] = "skipped"
                    continue
                results[task_key] = run_single_task(
                    config_name,
                    alloy_type,
                    alloy_config,
                    args,
                    progress_manager,
                    dry_run,
                    model_name,
                )

    return results


def list_configs() -> None:
    logger.info("Available extrapolation alloy configs:")
    for alloy_name, alloy_cfg in ALLOY_CONFIGS_EXTRAPOLATION.items():
        logger.info(f"  - {alloy_name}: targets={alloy_cfg['targets']}, data={alloy_cfg['raw_data']}")
    logger.info("Available extrapolation batch configs:")
    for config_name, config in BATCH_CONFIGS_EXTRAPOLATION.items():
        logger.info(f"  - {config_name}: {config['description']}")


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Independent extrapolation batch runner")
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument("--list", action="store_true")
    mode_group.add_argument("--config", nargs="+")
    mode_group.add_argument("--all", action="store_true")

    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--show_progress", action="store_true")
    parser.add_argument("--clear_progress", type=str, nargs="?", const="__all__", metavar="CONFIG")
    return parser


def main() -> None:
    parser = create_argument_parser()
    args = parser.parse_args()

    progress_manager = ProgressManager()
    if args.show_progress:
        progress_manager.show_progress()
        return
    if args.clear_progress is not None:
        progress_manager.clear_progress(None if args.clear_progress == "__all__" else args.clear_progress)
        return

    if not any([args.list, args.config, args.all]):
        parser.error("one of --list, --config, --all, --show_progress, or --clear_progress is required")

    if args.list:
        list_configs()
        return

    run_configs = args.config if args.config else list(BATCH_CONFIGS_EXTRAPOLATION.keys())
    invalid = [name for name in run_configs if name not in BATCH_CONFIGS_EXTRAPOLATION]
    if invalid:
        raise ValueError(f"Invalid extrapolation batch configs: {', '.join(invalid)}")

    set_seed(42)
    started_at = datetime.now()
    all_results: Dict[str, Dict[str, str]] = {}

    for config_name in run_configs:
        logger.info("=" * 100)
        logger.info(f"Running extrapolation batch config: {config_name}")
        logger.info("=" * 100)
        config = BATCH_CONFIGS_EXTRAPOLATION[config_name]
        all_results[config_name] = run_batch_config(
            config_name=config_name,
            config=config,
            dry_run=args.dry_run,
            progress_manager=progress_manager,
            resume=args.resume,
        )

    logger.info("=" * 100)
    logger.info("Extrapolation batch summary")
    logger.info("=" * 100)
    for config_name, results in all_results.items():
        logger.info(config_name)
        for task_key, status in results.items():
            logger.info(f"  - {task_key}: {status}")
    logger.info(f"Elapsed: {datetime.now() - started_at}")


if __name__ == "__main__":
    main()
# python -m src.pipelines.run_batch_extrapolation --config experiment1_all_ml_models_extrapolation --dry_run --resume
# python -m src.pipelines.run_batch_extrapolation --config experiment1_all_ml_models_extrapolation
# conda activate llm
# python -m src.pipelines.run_batch_extrapolation --config experiment2a_all_nn_scibert_extrapolation experiment2b_all_nn_steelbert_extrapolation experiment2c_all_nn_matscibert_extrapolation
