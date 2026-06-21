"""
Independent batch runner for TabPFN OOD experiments.
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
    from .model_factory import get_tabpfn_runtime_config
    from .train_tabpfn_ood import get_ood_method_output_root, get_ood_output_root
    from .tabpfn_ood_configs import (
        TABPFN_OOD_BATCH_CONFIGS,
        get_all_tabpfn_ood_alloys,
        get_tabpfn_ood_config,
    )
except ImportError:  # pragma: no cover
    from model_factory import get_tabpfn_runtime_config
    from train_tabpfn_ood import get_ood_method_output_root, get_ood_output_root
    from tabpfn_ood_configs import (
        TABPFN_OOD_BATCH_CONFIGS,
        get_all_tabpfn_ood_alloys,
        get_tabpfn_ood_config,
    )


logger = logging.getLogger(__name__)

HYBRID_STRATEGY_PREFIX = "hybrid_extrapolation_"
HYBRID_REQUIRED_TEST_SET_NAMES = ("test_extrapolation_high20", "test_inner_ood")


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
    def __init__(self, progress_file: str = ".batch_progress_tabpfn_ood.json") -> None:
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
            logger.info("No TabPFN OOD progress records")
            return
        logger.info("=" * 100)
        logger.info("TabPFN OOD Task Progress")
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


def is_hybrid_batch_config(batch_config: Dict[str, Any]) -> bool:
    return str(batch_config.get("split_strategy", "")).startswith(HYBRID_STRATEGY_PREFIX)


def resolve_batch_output_root(
    batch_config: Dict[str, Any],
    base_path: str,
    backend: str,
    model_version: Optional[str],
    feature_mode: str | None,
) -> Optional[str]:
    output_root = batch_config.get("output_root")
    if not output_root:
        return None
    if "{" not in str(output_root):
        return str(output_root)
    runtime_info = get_tabpfn_runtime_config(
        base_path=base_path,
        backend=backend,
        preferred_model_version=model_version,
        feature_mode=feature_mode,
    )
    return str(output_root).format(**runtime_info)


def _load_json_object(path: Path) -> Dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _has_required_hybrid_test_set_metrics(metrics_payload: Dict[str, Any]) -> bool:
    test_set_metrics = metrics_payload.get("test_set_metrics")
    if not isinstance(test_set_metrics, dict):
        return False
    for test_set_name in HYBRID_REQUIRED_TEST_SET_NAMES:
        metrics = test_set_metrics.get(test_set_name)
        if not isinstance(metrics, dict) or not metrics:
            return False
        if not {"mae", "rmse", "r2", "n_samples"}.intersection(metrics):
            return False
    return True


def _metrics_payload_complete(metrics_payload: Dict[str, Any], require_hybrid_test_sets: bool) -> bool:
    combined_metrics = metrics_payload.get("ood_test") or metrics_payload.get("test")
    if not isinstance(combined_metrics, dict) or not combined_metrics:
        return False
    if require_hybrid_test_sets and not _has_required_hybrid_test_set_metrics(metrics_payload):
        return False
    return True


def task_artifacts_complete(
    alloy_type: str,
    target_col: str,
    batch_config: Dict[str, Any],
    base_path: str,
    backend: str,
    model_version: Optional[str],
    feature_mode: str | None,
) -> bool:
    split_strategy = str(batch_config["split_strategy"])
    runtime_info = get_tabpfn_runtime_config(
        base_path=base_path,
        backend=backend,
        preferred_model_version=model_version,
        feature_mode=feature_mode,
    )
    output_root = resolve_batch_output_root(batch_config, base_path, backend, model_version, feature_mode)
    alloy_config = get_tabpfn_ood_config(
        alloy_type,
        backend=backend,
        feature_mode=feature_mode,
        base_path=base_path,
    )
    result_dir = (
        get_ood_method_output_root(base_path, runtime_info, split_strategy, output_root)
        / alloy_type
        / Path(alloy_config["raw_data"]).stem
        / target_col
    )
    if not result_dir.exists():
        return False
    require_hybrid_test_sets = is_hybrid_batch_config(batch_config)
    if split_strategy in {"loco", "random_cv_baseline", "hybrid_extrapolation_loco", "hybrid_extrapolation_random_cv"}:
        manifest_name = "loco_manifest.json" if split_strategy == "loco" else f"{split_strategy}_manifest.json"
        manifest = _load_json_object(result_dir / manifest_name)
        metrics_payload = manifest.get("aggregated_metrics", {})
        return isinstance(metrics_payload, dict) and _metrics_payload_complete(
            metrics_payload,
            require_hybrid_test_sets,
        )
    return _metrics_payload_complete(
        _load_json_object(result_dir / "metrics" / "metrics_summary.json"),
        require_hybrid_test_sets,
    )


def resolve_alloy_types(
    config: Dict[str, Any],
    backend: str,
    feature_mode: str | None,
    base_path: str,
) -> List[str]:
    alloy_types = config.get("alloy_types")
    if alloy_types is None:
        alloy_types = get_all_tabpfn_ood_alloys(
            backend=backend,
            feature_mode=feature_mode,
            base_path=base_path,
        )
    excluded = set(config.get("exclude_alloys", []))
    return [alloy for alloy in alloy_types if alloy not in excluded]


def build_command(
    alloy_type: str,
    target_col: str,
    batch_config: Dict[str, Any],
    base_path: str,
    backend: str,
    model_version: Optional[str],
    feature_mode: str | None,
) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "src.TabPFN.train_tabpfn_ood",
        "--alloy_type",
        alloy_type,
        "--target_col",
        target_col,
        "--backend",
        backend,
        "--base_path",
        base_path,
        "--test_size",
        str(batch_config["test_size"]),
        "--random_state",
        str(batch_config["random_state"]),
        "--split_strategy",
        batch_config["split_strategy"],
    ]
    if batch_config.get("extrapolation_side"):
        command.extend(["--extrapolation_side", batch_config["extrapolation_side"]])
    if batch_config.get("sparse_candidate_pool_size") is not None:
        command.extend(["--sparse_candidate_pool_size", str(batch_config["sparse_candidate_pool_size"])])
    if batch_config.get("sparse_cluster_count") is not None:
        command.extend(["--sparse_cluster_count", str(batch_config["sparse_cluster_count"])])
    if batch_config.get("sparse_samples_per_cluster") is not None:
        command.extend(["--sparse_samples_per_cluster", str(batch_config["sparse_samples_per_cluster"])])
    if batch_config.get("sparse_kde_bandwidth") is not None:
        command.extend(["--sparse_kde_bandwidth", str(batch_config["sparse_kde_bandwidth"])])
    if batch_config.get("sparse_neighbors_per_seed") is not None:
        command.extend(["--sparse_neighbors_per_seed", str(batch_config["sparse_neighbors_per_seed"])])
    if batch_config.get("loco_cluster_count") is not None:
        command.extend(["--loco_cluster_count", str(batch_config["loco_cluster_count"])])
    if batch_config.get("baseline_num_folds") is not None:
        command.extend(["--baseline_num_folds", str(batch_config["baseline_num_folds"])])
    if batch_config.get("outer_test_size") is not None:
        command.extend(["--outer_test_size", str(batch_config["outer_test_size"])])
    if model_version:
        command.extend(["--model_version", model_version])
    if feature_mode:
        command.extend(["--feature_mode", feature_mode])
    output_root = resolve_batch_output_root(batch_config, base_path, backend, model_version, feature_mode)
    if output_root:
        command.extend(["--output_root", output_root])
    return command


def run_single_task(
    config_name: str,
    alloy_type: str,
    target_col: str,
    batch_config: Dict[str, Any],
    progress_manager: ProgressManager,
    dry_run: bool,
    base_path: str,
    backend: str,
    model_version: Optional[str],
    feature_mode: str | None,
) -> str:
    task_key = make_task_key(alloy_type, target_col)
    cmd = build_command(
        alloy_type,
        target_col,
        batch_config,
        base_path,
        backend,
        model_version,
        feature_mode,
    )
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
    backend: str,
    model_version: Optional[str],
    feature_mode: str | None,
) -> Dict[str, str]:
    results: Dict[str, str] = {}
    alloy_types = resolve_alloy_types(
        batch_config,
        backend=backend,
        feature_mode=feature_mode,
        base_path=base_path,
    )
    total_targets = 0
    for alloy_type in alloy_types:
        total_targets += len(
            get_tabpfn_ood_config(
                alloy_type,
                backend=backend,
                feature_mode=feature_mode,
                base_path=base_path,
            )["targets"]
        )
    logger.info("Resolved %d alloys and %d single-target runs", len(alloy_types), total_targets)

    for alloy_type in alloy_types:
        alloy_config = get_tabpfn_ood_config(
            alloy_type,
            backend=backend,
            feature_mode=feature_mode,
            base_path=base_path,
        )
        logger.info("Processing alloy %s with targets %s", alloy_type, alloy_config["targets"])
        for target_col in alloy_config["targets"]:
            task_key = make_task_key(alloy_type, target_col)
            if resume and progress_manager.is_task_completed(config_name, task_key):
                if not is_hybrid_batch_config(batch_config) or task_artifacts_complete(
                    alloy_type=alloy_type,
                    target_col=target_col,
                    batch_config=batch_config,
                    base_path=base_path,
                    backend=backend,
                    model_version=model_version,
                    feature_mode=feature_mode,
                ):
                    results[task_key] = "skipped"
                    logger.info("[SKIP] %s already completed", task_key)
                    continue
                logger.info("[RERUN] %s progress=success but hybrid metrics are incomplete", task_key)
            if resume and not progress_manager.is_task_completed(config_name, task_key) and task_artifacts_complete(
                alloy_type=alloy_type,
                target_col=target_col,
                batch_config=batch_config,
                base_path=base_path,
                backend=backend,
                model_version=model_version,
                feature_mode=feature_mode,
            ):
                results[task_key] = "skipped"
                logger.info("[SKIP] %s existing output artifacts", task_key)
                if not dry_run:
                    progress_manager.update_task_status(config_name, task_key, "success")
                continue
            results[task_key] = run_single_task(
                config_name=config_name,
                alloy_type=alloy_type,
                target_col=target_col,
                batch_config=batch_config,
                progress_manager=progress_manager,
                dry_run=dry_run,
                base_path=base_path,
                backend=backend,
                model_version=model_version,
                feature_mode=feature_mode,
            )
    return results


def list_configs(backend: str, feature_mode: str | None, base_path: str) -> None:
    logger.info("Available TabPFN OOD alloy configs:")
    for alloy_type in get_all_tabpfn_ood_alloys(
        backend=backend,
        feature_mode=feature_mode,
        base_path=base_path,
    ):
        alloy_config = get_tabpfn_ood_config(
            alloy_type,
            backend=backend,
            feature_mode=feature_mode,
            base_path=base_path,
        )
        logger.info("  - %s: targets=%s, data=%s", alloy_type, alloy_config["targets"], alloy_config["raw_data"])
    logger.info("Available TabPFN OOD batch configs:")
    for config_name, config in TABPFN_OOD_BATCH_CONFIGS.items():
        logger.info("  - %s: %s", config_name, config["description"])


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch runner for TabPFN OOD experiments",
        allow_abbrev=False,
    )
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument("--list", action="store_true")
    mode_group.add_argument("--config", nargs="+")
    mode_group.add_argument("--all", action="store_true")

    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--show_progress", action="store_true")
    parser.add_argument("--clear_progress", type=str, nargs="?", const="__all__", metavar="CONFIG")
    parser.add_argument("--backend", choices=["auto", "api", "local"], default="auto")
    parser.add_argument("--model_version", choices=["latest", "v2", "v2.5", "v2.6"], default=None)
    parser.add_argument("--feature_mode", choices=["numeric", "text"], default=None)
    parser.add_argument("--base_path", default=str(Path(__file__).resolve().parents[2]), type=str)
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    runtime_info = get_tabpfn_runtime_config(
        base_path=args.base_path,
        backend=args.backend,
        preferred_model_version=args.model_version,
        feature_mode=args.feature_mode,
    )
    batch_log_file = (
        get_ood_output_root(
            args.base_path,
            runtime_info,
        )
        / "batch_logs"
        / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
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
        list_configs(args.backend, args.feature_mode, args.base_path)
        return

    run_configs = args.config if args.config else [
        name for name in TABPFN_OOD_BATCH_CONFIGS.keys() if name != "tabpfn_all_ood"
    ]
    invalid = [name for name in run_configs if name not in TABPFN_OOD_BATCH_CONFIGS]
    if invalid:
        raise ValueError(f"Invalid TabPFN OOD batch configs: {', '.join(invalid)}")

    started_at = datetime.now()
    all_results: Dict[str, Dict[str, str]] = {}
    for config_name in run_configs:
        logger.info("=" * 100)
        logger.info("Running TabPFN OOD batch config: %s", config_name)
        logger.info("=" * 100)
        all_results[config_name] = run_batch_config(
            config_name=config_name,
            batch_config=TABPFN_OOD_BATCH_CONFIGS[config_name],
            progress_manager=progress_manager,
            dry_run=args.dry_run,
            resume=args.resume,
            base_path=args.base_path,
            backend=args.backend,
            model_version=args.model_version,
            feature_mode=args.feature_mode,
        )

    logger.info("=" * 100)
    logger.info("TabPFN OOD batch summary")
    logger.info("=" * 100)
    for config_name, results in all_results.items():
        logger.info("%s", config_name)
        for task_key, status in results.items():
            logger.info("  - %s: %s", task_key, status)
    logger.info("Elapsed: %s", datetime.now() - started_at)


if __name__ == "__main__":
    main()
# python -m src.TabPFN.run_tabpfn_ood_batch --config tabpfn_all_random_cv_baseline --backend api --feature_mode text
# python -m src.TabPFN.run_tabpfn_ood_batch --config tabpfn_all_random_cv_baseline --backend api --feature_mode numeric
