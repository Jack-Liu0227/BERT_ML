"""
Unified batch runner for OOD experiments.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.feature_engineering.utils import set_seed
from src.pipelines.batch_configs_ood import (
    ALLOY_CONFIGS_OOD,
    BATCH_CONFIGS_OOD,
    OOD_METHODS,
    get_alloy_config_ood,
    get_ood_method_meta,
    list_available_alloys_ood,
    list_available_ood_methods,
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

HYBRID_STRATEGY_PREFIX = "hybrid_extrapolation_"
HYBRID_REQUIRED_TEST_SET_NAMES = ("test_extrapolation_high20", "test_inner_ood")


class ProgressManager:
    def __init__(self, progress_file: str = ".batch_progress_ood.json") -> None:
        self.progress_file = Path(progress_file)
        self.lock_file = self.progress_file.with_name(f"{self.progress_file.name}.lock")
        self._thread_lock = threading.RLock()
        self.progress_data = self._load_progress()

    def _load_progress(self) -> Dict[str, Any]:
        if self.progress_file.exists():
            try:
                return json.loads(self.progress_file.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning(f"Failed to load progress file: {exc}")
        return {}

    @contextlib.contextmanager
    def _file_lock(self, timeout: float = 120.0, poll_interval: float = 0.1):
        """Best-effort cross-process lock for progress JSON updates.

        This keeps `--jobs > 1` from corrupting the progress file when multiple
        worker threads finish at nearly the same time. It also coordinates with
        any other *new* runner process using the same ProgressManager.
        """
        start = time.monotonic()
        fd: Optional[int] = None
        while True:
            try:
                fd = os.open(str(self.lock_file), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(fd, f"{os.getpid()} {time.time()}\n".encode("utf-8"))
                break
            except FileExistsError:
                # Recover from a stale lock left by a crashed process.
                try:
                    if time.time() - self.lock_file.stat().st_mtime > timeout:
                        self.lock_file.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
                if time.monotonic() - start > timeout:
                    raise TimeoutError(f"Timed out waiting for progress lock: {self.lock_file}")
                time.sleep(poll_interval)
        try:
            yield
        finally:
            if fd is not None:
                os.close(fd)
            try:
                self.lock_file.unlink(missing_ok=True)
            except OSError:
                logger.warning("Failed to remove progress lock file: %s", self.lock_file)

    def _save_progress_unlocked(self) -> None:
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.progress_data, indent=2, ensure_ascii=False)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(self.progress_file.parent),
            delete=False,
            prefix=f".{self.progress_file.name}.",
            suffix=".tmp",
        ) as tmp:
            tmp.write(payload)
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, self.progress_file)

    def _save_progress(self) -> None:
        with self._thread_lock:
            with self._file_lock():
                self._save_progress_unlocked()

    def get_config_progress(self, config_name: str) -> Dict[str, str]:
        with self._thread_lock:
            return self.progress_data.get(config_name, {}).copy()

    def is_task_completed(self, config_name: str, task_key: str) -> bool:
        return self.get_config_progress(config_name).get(task_key) == "success"

    def update_task_status(self, config_name: str, task_key: str, status: str) -> None:
        with self._thread_lock:
            with self._file_lock():
                # Reload inside the lock so concurrent workers don't overwrite
                # each other's recently-finished task statuses.
                self.progress_data = self._load_progress()
                self.progress_data.setdefault(config_name, {})[task_key] = status
                self._save_progress_unlocked()

    def clear_progress(self, config_name: Optional[str] = None) -> None:
        with self._thread_lock:
            with self._file_lock():
                self.progress_data = self._load_progress()
                if config_name is None:
                    self.progress_data = {}
                else:
                    self.progress_data.pop(config_name, None)
                self._save_progress_unlocked()

    def show_progress(self, config_name: Optional[str] = None) -> None:
        with self._thread_lock:
            self.progress_data = self._load_progress()
        items = self.progress_data if config_name is None else {config_name: self.progress_data.get(config_name, {})}
        if not items or all(not value for value in items.values()):
            logger.info("No OOD progress records")
            return
        logger.info("=" * 100)
        logger.info("OOD Task Progress")
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


def is_llmprop_config(args: Any) -> bool:
    return bool(getattr(args, "use_llmprop", False))


def is_hybrid_config(args: Any, method_meta: Dict[str, Any]) -> bool:
    split_strategy = str(getattr(args, "split_strategy", ""))
    return split_strategy.startswith(HYBRID_STRATEGY_PREFIX) or bool(method_meta.get("is_hybrid"))


def _dataset_name_from_config(raw_data: str) -> str:
    dataset_name = Path(raw_data).stem
    for suffix in ["_Processed_cleaned", "_with_ID", "_withID", "_cleaned", "_processed", "_Processed"]:
        dataset_name = dataset_name.replace(suffix, "")
    return dataset_name


def build_result_dir(
    config_name: str,
    alloy_type: str,
    alloy_config: Dict[str, Any],
    args: Any,
    method_meta: Dict[str, Any],
) -> Path:
    dataset_name = _dataset_name_from_config(alloy_config["raw_data"])
    target_column = alloy_config["target_column"]
    terminal_dir = "llmprop" if is_llmprop_config(args) else args.embedding_type
    result_base_dir = Path(getattr(args, "result_base_dir", "output/ood_results"))
    return (
        result_base_dir
        / config_name
        / alloy_type
        / dataset_name
        / target_column
        / method_meta["result_dir_suffix"]
        / terminal_dir
    )


def _append_method_specific_args(cmd: List[str], args: Any, method_meta: Dict[str, Any]) -> None:
    cli_args = set(method_meta.get("cli_args", []))
    if "outer_test_size" in cli_args:
        cmd.extend(["--outer_test_size", str(args.outer_test_size)])
    if "extrapolation_side" in cli_args:
        cmd.extend(["--extrapolation_side", args.extrapolation_side])
    if "sparse_candidate_pool_size" in cli_args:
        cmd.extend(["--sparse_candidate_pool_size", str(args.sparse_candidate_pool_size)])
    if "sparse_cluster_count" in cli_args:
        cmd.extend(["--sparse_cluster_count", str(args.sparse_cluster_count)])
    if "sparse_samples_per_cluster" in cli_args:
        cmd.extend(["--sparse_samples_per_cluster", str(args.sparse_samples_per_cluster)])
    if "sparse_kde_bandwidth" in cli_args and getattr(args, "sparse_kde_bandwidth", None) is not None:
        cmd.extend(["--sparse_kde_bandwidth", str(args.sparse_kde_bandwidth)])
    if "sparse_neighbors_per_seed" in cli_args:
        cmd.extend(["--sparse_neighbors_per_seed", str(args.sparse_neighbors_per_seed)])
    if "loco_cluster_count" in cli_args:
        cmd.extend(["--loco_cluster_count", str(args.loco_cluster_count)])
    if "baseline_num_folds" in cli_args:
        cmd.extend(["--baseline_num_folds", str(args.baseline_num_folds)])


def build_command(
    config_name: str,
    alloy_type: str,
    alloy_config: Dict[str, Any],
    args: Any,
    method_meta: Dict[str, Any],
    model_name: Optional[str] = None,
) -> List[str]:
    target_column = alloy_config["target_column"]
    result_dir = build_result_dir(config_name, alloy_type, alloy_config, args, method_meta)

    cmd = [
        sys.executable,
        "-m",
        "src.pipelines.ood_pipeline",
        "--data_file",
        alloy_config["raw_data"],
        "--result_dir",
        str(result_dir),
        "--target_column",
        target_column,
        "--alloy_type",
        alloy_type,
        "--embedding_type",
        args.embedding_type,
        "--split_strategy",
        args.split_strategy,
        "--test_size",
        str(args.test_size),
        "--random_state",
        str(args.random_state),
        "--split_cache_dir",
        str(getattr(args, "split_cache_dir", "output/ood_splits")),
    ]

    _append_method_specific_args(cmd, args, method_meta)

    processing_cols = alloy_config.get("processing_cols") or []
    if processing_cols:
        cmd.extend(["--processing_cols", *processing_cols])
    else:
        fallback_processing_cols = getattr(args, "processing_cols", None) or []
        if fallback_processing_cols:
            cmd.extend(["--processing_cols", *fallback_processing_cols])

    processing_text_column = alloy_config.get("processing_text_column")
    if processing_text_column:
        cmd.extend(["--processing_text_column", processing_text_column])

    if is_llmprop_config(args):
        cmd.append("--use_llmprop")
        cmd.extend(
            [
                "--llmprop_epochs",
                str(args.llmprop_epochs),
                "--llmprop_batch_size",
                str(args.llmprop_batch_size),
                "--llmprop_lr",
                str(args.llmprop_lr),
                "--llmprop_max_len",
                str(args.llmprop_max_len),
                "--llmprop_dropout",
                str(args.llmprop_dropout),
                "--llmprop_pooling",
                str(args.llmprop_pooling),
                "--llmprop_tokenizer",
                str(args.llmprop_tokenizer),
                "--llmprop_base_model",
                str(args.llmprop_base_model),
                "--llmprop_valid_ratio",
                str(args.llmprop_valid_ratio),
            ]
        )
        if args.use_optuna:
            cmd.extend(["--use_optuna", "--n_trials", str(args.n_trials)])
        return cmd

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
    cmd.append("--evaluate_after_train" if args.evaluate_after_train else "--no-evaluate_after_train")
    if args.run_shap_analysis:
        cmd.append("--run_shap_analysis")

    return cmd


def format_command_for_display(cmd: List[str]) -> str:
    parts = []
    for part in cmd:
        if any(char in part for char in [" ", "(", ")", "%", "/", "\\"]):
            parts.append(f'"{part}"')
        else:
            parts.append(part)
    return " ".join(parts)


def _json_manifest_matches(path: Path, target_column: str) -> bool:
    """Return True when a completion manifest exists and matches the target.

    The batch progress file is the authoritative resume state, but older runs
    may have generated full outputs before they were recorded there.  In that
    case we use the per-task OOD manifest as a conservative completion marker:
    it is written at the end of training/evaluation, after predictions/metrics
    have been emitted.
    """
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    recorded_target = payload.get("target_column")
    return recorded_target in (None, target_column)


def _load_json_object(path: Path) -> Dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _has_combined_metrics(payload: Dict[str, Any]) -> bool:
    combined_summary = payload.get("combined_evaluation_summary")
    if isinstance(combined_summary, dict):
        return bool(combined_summary)
    return any(key != "test_set_metrics" and value is not None for key, value in payload.items())


def _has_required_hybrid_test_set_metrics(payload: Dict[str, Any]) -> bool:
    test_set_metrics = payload.get("test_set_metrics")
    if not isinstance(test_set_metrics, dict):
        return False
    for test_set_name in HYBRID_REQUIRED_TEST_SET_NAMES:
        metrics = test_set_metrics.get(test_set_name)
        if not isinstance(metrics, dict) or not metrics:
            return False
        metric_keys = {"mae", "rmse", "r2", "test_mae", "test_rmse", "test_r2", "n_samples"}
        if not metric_keys.intersection(metrics):
            return False
    return True


def _metrics_file_complete(path: Path, require_hybrid_test_sets: bool) -> bool:
    payload = _load_json_object(path)
    if not payload or not _has_combined_metrics(payload):
        return False
    if require_hybrid_test_sets and not _has_required_hybrid_test_set_metrics(payload):
        return False
    return True


def _ml_model_artifacts_complete(
    run_root: Path,
    target_column: str,
    model_name: str,
    require_hybrid_test_sets: bool = False,
) -> bool:
    model_dir = run_root / "model_comparison" / f"{model_name}_results"
    if not _json_manifest_matches(model_dir / "ood_manifest.json", target_column):
        return False
    # Evaluation metrics are produced before the OOD manifest.  Keep the check
    # lenient because older runs used slightly different metric filenames.
    metric_candidates = [
        model_dir / "final_model_evaluation_metrics.json",
        model_dir / "final_evaluation_metrics.json",
        model_dir / "cv_avg_metrics.json",
    ]
    if require_hybrid_test_sets:
        return _metrics_file_complete(model_dir / "final_evaluation_metrics.json", True)
    return any(path.exists() and path.stat().st_size > 0 for path in metric_candidates)


def _nn_artifacts_complete(
    run_root: Path,
    target_column: str,
    require_hybrid_test_sets: bool = False,
) -> bool:
    if not _json_manifest_matches(run_root / "ood_manifest.json", target_column):
        return False
    checkpoint_candidates = [
        run_root / "checkpoints" / "best_model.pt",
        run_root / "checkpoints" / "best_model_best.pt",
    ]
    if not any(path.exists() and path.stat().st_size > 0 for path in checkpoint_candidates):
        return False
    if require_hybrid_test_sets:
        return _metrics_file_complete(run_root / "final_evaluation_metrics.json", True)
    metric_candidates = [
        run_root / "final_evaluation_metrics.json",
        run_root / "best_model_best_model_evaluation_evaluation_summary.json",
        run_root / "cv_avg_metrics.json",
    ]
    return any(path.exists() and path.stat().st_size > 0 for path in metric_candidates)


def _llmprop_artifacts_complete(
    run_root: Path,
    target_column: str,
    require_hybrid_test_sets: bool = False,
) -> bool:
    manifest_path = run_root / "llmprop_manifest.json"
    metrics_path = run_root / "final_evaluation_metrics.json"
    checkpoint_path = run_root / "checkpoints" / "best_model.pt"
    predictions_path = run_root / "predictions" / "test_predictions.csv"
    if not all(path.exists() and path.stat().st_size > 0 for path in [manifest_path, metrics_path, checkpoint_path, predictions_path]):
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if payload.get("target_column") != target_column:
        return False
    return _metrics_file_complete(metrics_path, require_hybrid_test_sets)


def _load_fold_roots(result_dir: Path, method_meta: Dict[str, Any]) -> List[Path]:
    manifest_path = result_dir / str(method_meta.get("summary_file_name", ""))
    if not manifest_path.exists() or manifest_path.stat().st_size == 0:
        return []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    fold_entries = payload.get("folds") or []
    fold_roots: List[Path] = []
    for entry in fold_entries:
        run_root = entry.get("run_root")
        if run_root:
            fold_roots.append(Path(run_root))
        elif "fold_index" in entry:
            fold_roots.append(result_dir / "folds" / f"fold_{entry['fold_index']}")

    if fold_roots:
        return fold_roots

    fold_count = payload.get("fold_count")
    if isinstance(fold_count, int) and fold_count > 0:
        return [result_dir / "folds" / f"fold_{idx}" for idx in range(fold_count)]
    return []


def task_artifacts_complete(
    config_name: str,
    alloy_type: str,
    alloy_config: Dict[str, Any],
    args: Any,
    method_meta: Dict[str, Any],
    model_name: Optional[str] = None,
) -> bool:
    """Infer completion from existing OOD output files.

    This is intentionally only used when ``--resume`` is enabled.  It lets a
    resumed batch skip legacy finished tasks whose status was never written to
    ``.batch_progress_ood.json`` while still allowing a normal run without
    ``--resume`` to overwrite/recompute outputs.
    """
    result_dir = build_result_dir(config_name, alloy_type, alloy_config, args, method_meta)
    target_column = alloy_config["target_column"]
    require_hybrid_test_sets = is_hybrid_config(args, method_meta)
    if not result_dir.exists():
        return False

    if method_meta.get("is_multi_fold"):
        fold_roots = _load_fold_roots(result_dir, method_meta)
        if not fold_roots:
            return False
        if is_llmprop_config(args):
            return all(
                _llmprop_artifacts_complete(fold_root, target_column, require_hybrid_test_sets)
                for fold_root in fold_roots
            )
        if model_name:
            return all(
                _ml_model_artifacts_complete(fold_root, target_column, model_name, require_hybrid_test_sets)
                for fold_root in fold_roots
            )
        return all(_nn_artifacts_complete(fold_root, target_column, require_hybrid_test_sets) for fold_root in fold_roots)

    if is_llmprop_config(args):
        return _llmprop_artifacts_complete(result_dir, target_column, require_hybrid_test_sets)
    if model_name:
        return _ml_model_artifacts_complete(result_dir, target_column, model_name, require_hybrid_test_sets)
    return _nn_artifacts_complete(result_dir, target_column, require_hybrid_test_sets)


def should_skip_completed_task(
    config_name: str,
    task_key: str,
    alloy_type: str,
    alloy_config: Dict[str, Any],
    args: Any,
    method_meta: Dict[str, Any],
    progress_manager: ProgressManager,
    dry_run: bool,
    model_name: Optional[str] = None,
) -> bool:
    if progress_manager.is_task_completed(config_name, task_key):
        if is_hybrid_config(args, method_meta):
            if task_artifacts_complete(config_name, alloy_type, alloy_config, args, method_meta, model_name):
                logger.info(f"[SKIP] {config_name} / {task_key} (progress=success, hybrid artifacts complete)")
                return True
            logger.info(f"[RERUN] {config_name} / {task_key} (progress=success but hybrid metrics are incomplete)")
            return False
        logger.info(f"[SKIP] {config_name} / {task_key} (progress=success)")
        return True
    if task_artifacts_complete(config_name, alloy_type, alloy_config, args, method_meta, model_name):
        logger.info(f"[SKIP] {config_name} / {task_key} (existing output artifacts)")
        if not dry_run:
            progress_manager.update_task_status(config_name, task_key, "success")
        return True
    return False


def run_single_task(
    config_name: str,
    alloy_type: str,
    alloy_config: Dict[str, Any],
    args: Any,
    progress_manager: ProgressManager,
    dry_run: bool,
    method_meta: Dict[str, Any],
    model_name: Optional[str] = None,
) -> str:
    effective_model_name = "llmprop" if is_llmprop_config(args) else model_name
    task_key = make_task_key(alloy_type, alloy_config["target_column"], effective_model_name)
    cmd = build_command(config_name, alloy_type, alloy_config, args, method_meta, model_name)
    logger.info(f"[RUN] {config_name} / {task_key}")
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
        alloy_types = list_available_alloys_ood()
    excluded = set(config.get("exclude_alloys", []))
    return [alloy for alloy in alloy_types if alloy not in excluded]


def run_batch_config(
    config_name: str,
    config: Dict[str, Any],
    dry_run: bool,
    progress_manager: ProgressManager,
    resume: bool,
    jobs: int = 1,
) -> Dict[str, str]:
    args = argparse.Namespace(**config)
    method_meta = get_ood_method_meta(args.ood_method)
    results: Dict[str, str] = {}
    pending_tasks: List[Dict[str, Any]] = []

    for alloy_type in resolve_alloy_types(config):
        alloy_base_config = get_alloy_config_ood(alloy_type)
        targets = alloy_base_config.get("targets") or []
        if not targets:
            raise ValueError(f"No targets configured for OOD alloy: {alloy_type}")
        target_filter = config.get("target_columns") or []
        if target_filter:
            allowed_targets = set(target_filter)
            targets = [target for target in targets if target in allowed_targets]
            if not targets:
                logger.warning(
                    "No matching OOD targets for alloy=%s under config=%s; requested targets=%s",
                    alloy_type,
                    config_name,
                    list(target_filter),
                )
                continue

        for target_column in targets:
            alloy_config = alloy_base_config.copy()
            alloy_config["target_column"] = target_column

            if is_llmprop_config(args):
                task_key = make_task_key(alloy_type, target_column, "llmprop")
                if resume and should_skip_completed_task(
                    config_name=config_name,
                    task_key=task_key,
                    alloy_type=alloy_type,
                    alloy_config=alloy_config,
                    args=args,
                    method_meta=method_meta,
                    progress_manager=progress_manager,
                    dry_run=dry_run,
                ):
                    results[task_key] = "skipped"
                    continue
                pending_tasks.append(
                    {
                        "config_name": config_name,
                        "alloy_type": alloy_type,
                        "alloy_config": alloy_config,
                        "args": args,
                        "progress_manager": progress_manager,
                        "dry_run": dry_run,
                        "method_meta": method_meta,
                    }
                )
                continue

            if args.use_nn:
                task_key = make_task_key(alloy_type, target_column)
                if resume and should_skip_completed_task(
                    config_name=config_name,
                    task_key=task_key,
                    alloy_type=alloy_type,
                    alloy_config=alloy_config,
                    args=args,
                    method_meta=method_meta,
                    progress_manager=progress_manager,
                    dry_run=dry_run,
                ):
                    results[task_key] = "skipped"
                    continue
                pending_tasks.append(
                    {
                        "config_name": config_name,
                        "alloy_type": alloy_type,
                        "alloy_config": alloy_config,
                        "args": args,
                        "progress_manager": progress_manager,
                        "dry_run": dry_run,
                        "method_meta": method_meta,
                    }
                )
                continue

            for model_name in args.models:
                task_key = make_task_key(alloy_type, target_column, model_name)
                if resume and should_skip_completed_task(
                    config_name=config_name,
                    task_key=task_key,
                    alloy_type=alloy_type,
                    alloy_config=alloy_config,
                    args=args,
                    method_meta=method_meta,
                    progress_manager=progress_manager,
                    dry_run=dry_run,
                    model_name=model_name,
                ):
                    results[task_key] = "skipped"
                    continue
                pending_tasks.append(
                    {
                        "config_name": config_name,
                        "alloy_type": alloy_type,
                        "alloy_config": alloy_config,
                        "args": args,
                        "progress_manager": progress_manager,
                        "dry_run": dry_run,
                        "method_meta": method_meta,
                        "model_name": model_name,
                    }
                )

    jobs = max(1, int(jobs))
    if not pending_tasks:
        return results
    if jobs == 1:
        for task in pending_tasks:
            model_name = "llmprop" if is_llmprop_config(task["args"]) else task.get("model_name")
            task_key = make_task_key(task["alloy_type"], task["alloy_config"]["target_column"], model_name)
            results[task_key] = run_single_task(**task)
        return results

    logger.info("Running %s pending tasks from %s with jobs=%s", len(pending_tasks), config_name, jobs)
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        future_to_task = {}
        for task in pending_tasks:
            model_name = "llmprop" if is_llmprop_config(task["args"]) else task.get("model_name")
            task_key = make_task_key(task["alloy_type"], task["alloy_config"]["target_column"], model_name)
            future_to_task[executor.submit(run_single_task, **task)] = task_key
        for future in as_completed(future_to_task):
            task_key = future_to_task[future]
            try:
                results[task_key] = future.result()
            except Exception as exc:
                logger.exception("Task crashed before status update: %s", task_key)
                progress_manager.update_task_status(config_name, task_key, "failed")
                results[task_key] = f"failed: {exc}"
    return results


def list_configs() -> None:
    logger.info("Available OOD alloy configs:")
    for alloy_name, alloy_cfg in ALLOY_CONFIGS_OOD.items():
        logger.info(f"  - {alloy_name}: targets={alloy_cfg['targets']}, data={alloy_cfg['raw_data']}")

    logger.info("Available OOD methods:")
    for method_name in list_available_ood_methods():
        method_meta = OOD_METHODS[method_name]
        logger.info(
            "  - %s: multi_fold=%s, summary=%s, result_suffix=%s",
            method_name,
            method_meta["is_multi_fold"],
            method_meta["summary_file_name"],
            method_meta["result_dir_suffix"],
        )

    logger.info("Available OOD batch configs:")
    for config_name, config in BATCH_CONFIGS_OOD.items():
        logger.info(f"  - {config_name}: {config['description']}")


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified OOD batch runner")
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument("--list", action="store_true")
    mode_group.add_argument("--config", nargs="+")
    mode_group.add_argument("--all", action="store_true")

    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--alloys",
        nargs="+",
        help="Optional alloy/dataset names to run, e.g. --alloys MatbenchSteels. Overrides each batch config's alloy_types.",
    )
    parser.add_argument(
        "--processing_cols",
        nargs="*",
        default=[],
        help="Optional numeric processing columns for OOD X-space construction when the alloy config does not define them.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        help='Optional target columns to run, e.g. --targets "UTS(MPa)" "El(%%)". Filters each selected alloy.',
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of OOD tasks to run in parallel per batch config. Use 1 for sequential execution.",
    )
    parser.add_argument(
        "--use_optuna",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override selected configs to enable/disable Optuna. Defaults to each config value.",
    )
    parser.add_argument("--n_trials", type=int, default=None, help="Override Optuna trial count.")
    parser.add_argument("--llmprop_epochs", type=int, default=None)
    parser.add_argument("--llmprop_batch_size", type=int, default=None)
    parser.add_argument("--llmprop_lr", type=float, default=None)
    parser.add_argument("--llmprop_max_len", type=int, default=None)
    parser.add_argument("--llmprop_dropout", type=float, default=None)
    parser.add_argument("--llmprop_pooling", choices=["cls", "mean"], default=None)
    parser.add_argument("--llmprop_tokenizer", type=str, default=None)
    parser.add_argument("--llmprop_base_model", type=str, default=None)
    parser.add_argument("--llmprop_valid_ratio", type=float, default=None)
    parser.add_argument("--split_cache_dir", type=str, default=None)
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

    run_configs = args.config if args.config else list(BATCH_CONFIGS_OOD.keys())
    invalid = [name for name in run_configs if name not in BATCH_CONFIGS_OOD]
    if invalid:
        raise ValueError(f"Invalid OOD batch configs: {', '.join(invalid)}")

    set_seed(42)
    started_at = datetime.now()
    all_results: Dict[str, Dict[str, str]] = {}

    for config_name in run_configs:
        logger.info("=" * 100)
        logger.info(f"Running OOD batch config: {config_name}")
        logger.info("=" * 100)
        config = BATCH_CONFIGS_OOD[config_name]
        if args.alloys:
            config = {**config, "alloy_types": args.alloys}
        if args.processing_cols:
            config = {**config, "processing_cols": args.processing_cols}
        if args.targets:
            config = {**config, "target_columns": args.targets}
        override_keys = [
            "use_optuna",
            "n_trials",
            "llmprop_epochs",
            "llmprop_batch_size",
            "llmprop_lr",
            "llmprop_max_len",
            "llmprop_dropout",
            "llmprop_pooling",
            "llmprop_tokenizer",
            "llmprop_base_model",
            "llmprop_valid_ratio",
            "split_cache_dir",
        ]
        overrides = {key: getattr(args, key) for key in override_keys if getattr(args, key) is not None}
        if overrides:
            config = {**config, **overrides}
        all_results[config_name] = run_batch_config(
            config_name=config_name,
            config=config,
            dry_run=args.dry_run,
            progress_manager=progress_manager,
            resume=args.resume,
            jobs=args.jobs,
        )

    logger.info("=" * 100)
    logger.info("OOD batch summary")
    logger.info("=" * 100)
    for config_name, results in all_results.items():
        logger.info(config_name)
        for task_key, status in results.items():
            logger.info(f"  - {task_key}: {status}")
    logger.info(f"Elapsed: {datetime.now() - started_at}")


if __name__ == "__main__":
    main()
# python -m src.pipelines.run_batch_ood --config experiment2a_all_nn_scibert_random_cv_baseline experiment2b_all_nn_steelbert_random_cv_baseline experiment2c_all_nn_matscibert_random_cv_baseline
# python -m src.pipelines.run_batch_ood --config experiment1_all_ml_models_random_cv_baseline
