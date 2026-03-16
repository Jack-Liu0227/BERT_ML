"""
Independent batch runner for TabPFN low-to-high extrapolation experiments.
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

try:
    from .tabpfn_extrapolation_configs import (
        TABPFN_EXTRAPOLATION_BATCH_CONFIGS,
        get_all_tabpfn_extrapolation_alloys,
        get_tabpfn_extrapolation_config,
    )
except ImportError:  # pragma: no cover
    from tabpfn_extrapolation_configs import (
        TABPFN_EXTRAPOLATION_BATCH_CONFIGS,
        get_all_tabpfn_extrapolation_alloys,
        get_tabpfn_extrapolation_config,
    )


logger = logging.getLogger(__name__)


def configure_logging(log_file: Optional[Path] = None) -> None:
    formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s")
    handlers: list[logging.Handler] = []

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in handlers:
        logger.addHandler(handler)


class ProgressManager:
    def __init__(self, progress_file: str = ".batch_progress_tabpfn_extrapolation.json") -> None:
        self.progress_file = Path(progress_file)
        self.progress_data = self._load_progress()

    def _load_progress(self) -> Dict[str, Any]:
        if self.progress_file.exists():
            try:
                return json.loads(self.progress_file.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save(self) -> None:
        self.progress_file.write_text(
            json.dumps(self.progress_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def is_task_completed(self, config_name: str, task_key: str) -> bool:
        return self.progress_data.get(config_name, {}).get(task_key) == "success"

    def update_task_status(self, config_name: str, task_key: str, status: str) -> None:
        self.progress_data.setdefault(config_name, {})[task_key] = status
        self._save()

    def show_progress(self, config_name: Optional[str] = None) -> None:
        items = self.progress_data if config_name is None else {config_name: self.progress_data.get(config_name, {})}
        if not items or all(not value for value in items.values()):
            logger.info("No TabPFN extrapolation progress records")
            return
        logger.info("=" * 100)
        logger.info("TabPFN Extrapolation Task Progress")
        logger.info("=" * 100)
        for cfg_name, tasks in items.items():
            if not tasks:
                continue
            success = sum(1 for status in tasks.values() if status == "success")
            failed = sum(1 for status in tasks.values() if status == "failed")
            total = len(tasks)
            logger.info("%s: success=%d, failed=%d, total=%d", cfg_name, success, failed, total)
            for task_key, status in sorted(tasks.items()):
                logger.info("  - %s: %s", task_key, status)

    def clear_progress(self, config_name: Optional[str] = None) -> None:
        if config_name is None:
            self.progress_data = {}
        else:
            self.progress_data.pop(config_name, None)
        self._save()


def make_task_key(alloy_type: str, target_col: str) -> str:
    return f"{alloy_type}_{target_col}"


def format_command(cmd: List[str]) -> str:
    formatted = []
    for part in cmd:
        if any(ch in part for ch in [" ", "(", ")", "%", "/", "\\"]):
            formatted.append(f'"{part}"')
        else:
            formatted.append(part)
    return " ".join(formatted)


def resolve_alloy_types(config: Dict[str, Any]) -> List[str]:
    alloy_types = config.get("alloy_types")
    if alloy_types is None:
        alloy_types = get_all_tabpfn_extrapolation_alloys()
    excluded = set(config.get("exclude_alloys", []))
    return [alloy for alloy in alloy_types if alloy not in excluded]


def build_command(
    alloy_type: str,
    target_col: str,
    batch_config: Dict[str, Any],
    base_path: str,
) -> List[str]:
    return [
        sys.executable,
        "-m",
        "src.TabPFN.train_tabpfn_extrapolation",
        "--alloy_type",
        alloy_type,
        "--target_col",
        target_col,
        "--base_path",
        base_path,
        "--output_root",
        batch_config["output_root"],
        "--test_size",
        str(batch_config["test_size"]),
        "--random_state",
        str(batch_config["random_state"]),
        "--extrapolation_side",
        batch_config["extrapolation_side"],
    ]


def run_single_task(
    config_name: str,
    alloy_type: str,
    target_col: str,
    batch_config: Dict[str, Any],
    progress_manager: ProgressManager,
    dry_run: bool,
    base_path: str,
) -> str:
    task_key = make_task_key(alloy_type, target_col)
    cmd = build_command(alloy_type, target_col, batch_config, base_path)
    logger.info("[RUN] %s", task_key)
    logger.info("Command: %s", format_command(cmd))

    if dry_run:
        return "dry_run"

    progress_manager.update_task_status(config_name, task_key, "running")
    result = subprocess.run(cmd, text=True)
    status = "success" if result.returncode == 0 else "failed"
    progress_manager.update_task_status(config_name, task_key, status)
    logger.info("[DONE] %s -> %s", task_key, status)
    return status


def run_batch_config(
    config_name: str,
    batch_config: Dict[str, Any],
    progress_manager: ProgressManager,
    dry_run: bool,
    resume: bool,
    base_path: str,
) -> Dict[str, str]:
    results: Dict[str, str] = {}
    alloy_types = resolve_alloy_types(batch_config)
    total_targets = 0
    for alloy_type in alloy_types:
        total_targets += len(get_tabpfn_extrapolation_config(alloy_type)["targets"])
    logger.info("Resolved %d alloys and %d single-target runs", len(alloy_types), total_targets)

    for alloy_type in alloy_types:
        alloy_config = get_tabpfn_extrapolation_config(alloy_type)
        logger.info("Processing alloy %s with targets %s", alloy_type, alloy_config["targets"])
        for target_col in alloy_config["targets"]:
            task_key = make_task_key(alloy_type, target_col)
            if resume and progress_manager.is_task_completed(config_name, task_key):
                results[task_key] = "skipped"
                logger.info("[SKIP] %s already completed", task_key)
                continue
            results[task_key] = run_single_task(
                config_name=config_name,
                alloy_type=alloy_type,
                target_col=target_col,
                batch_config=batch_config,
                progress_manager=progress_manager,
                dry_run=dry_run,
                base_path=base_path,
            )
    return results


def list_configs() -> None:
    logger.info("Available TabPFN extrapolation alloy configs:")
    for alloy_type in get_all_tabpfn_extrapolation_alloys():
        alloy_config = get_tabpfn_extrapolation_config(alloy_type)
        logger.info("  - %s: targets=%s, data=%s", alloy_type, alloy_config["targets"], alloy_config["raw_data"])
    logger.info("Available TabPFN extrapolation batch configs:")
    for config_name, config in TABPFN_EXTRAPOLATION_BATCH_CONFIGS.items():
        logger.info("  - %s: %s", config_name, config["description"])


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch runner for TabPFN low-to-high extrapolation")
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument("--list", action="store_true")
    mode_group.add_argument("--config", nargs="+")
    mode_group.add_argument("--all", action="store_true")

    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--show_progress", action="store_true")
    parser.add_argument("--clear_progress", type=str, nargs="?", const="__all__", metavar="CONFIG")
    parser.add_argument("--base_path", default=str(Path(__file__).resolve().parents[2]), type=str)
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    batch_log_file = Path(args.base_path) / "output" / "extrapolation_results_tabpfn" / "batch_logs" / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    configure_logging(batch_log_file)

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

    run_configs = args.config if args.config else list(TABPFN_EXTRAPOLATION_BATCH_CONFIGS.keys())
    invalid = [name for name in run_configs if name not in TABPFN_EXTRAPOLATION_BATCH_CONFIGS]
    if invalid:
        raise ValueError(f"Invalid TabPFN extrapolation batch configs: {', '.join(invalid)}")

    started_at = datetime.now()
    all_results: Dict[str, Dict[str, str]] = {}
    for config_name in run_configs:
        logger.info("=" * 100)
        logger.info("Running TabPFN extrapolation batch config: %s", config_name)
        logger.info("=" * 100)
        all_results[config_name] = run_batch_config(
            config_name=config_name,
            batch_config=TABPFN_EXTRAPOLATION_BATCH_CONFIGS[config_name],
            progress_manager=progress_manager,
            dry_run=args.dry_run,
            resume=args.resume,
            base_path=args.base_path,
        )

    logger.info("=" * 100)
    logger.info("TabPFN extrapolation batch summary")
    logger.info("=" * 100)
    for config_name, results in all_results.items():
        logger.info("%s", config_name)
        for task_key, status in results.items():
            logger.info("  - %s: %s", task_key, status)
    logger.info("Elapsed: %s", datetime.now() - started_at)


if __name__ == "__main__":
    main()
