"""End-to-end execution and aggregation for fixed training-size OOD splits."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.calibration.parameter_registry import FrozenParameterRegistry
from src.calibration.parameter_selection import build_parameter_hash
from src.data_processing.strength_ood_common import save_json
from src.pipelines.batch_configs_ood import ALLOY_CONFIGS_OOD

from .artifacts import (
    canonical_json_hash,
    cleanup_training_artifacts,
    sha256_file,
    validate_completed_artifacts,
    write_canonical_predictions,
)
from .core import (
    LOCO_TEST_NAME,
    TOP20_TEST_NAME,
    FixedTrainingSizeSplit,
    TrainingSubsetPlan,
    TrainingSizeOODConfig,
    build_fixed_split,
    build_training_subset_plan,
    iter_training_run_keys,
    materialize_training_split,
    validate_run_metrics_payload,
)
from .model_adapters import TaskRuntimeContext, get_model_adapter


class TrainingSizeOODRunner:
    """Run fixed-parameter training-size sweeps for one or more model specs."""

    def __init__(
        self,
        config: TrainingSizeOODConfig,
        frozen_parameters_path: str | Path,
        output_root: str | Path,
        source_config_path: str | Path | None = None,
    ) -> None:
        self.config = config
        self.registry = FrozenParameterRegistry(frozen_parameters_path)
        self.frozen_parameters_path = Path(frozen_parameters_path)
        self.source_config_path = Path(source_config_path) if source_config_path is not None else None
        self.output_root = Path(output_root)
        self._prepared_tasks: dict[str, FixedTrainingSizeSplit] = {}
        self._historical_evidence: dict[tuple[str, str, str], dict[str, Any]] = {}

    def prepare_task(self, task_id: str) -> FixedTrainingSizeSplit:
        fixed = build_fixed_split(self.config, task_id)
        split_dir = self.output_root / "split_manifests"
        split_dir.mkdir(parents=True, exist_ok=True)
        save_json(split_dir / f"{_safe_name(task_id)}.json", fixed.to_manifest())
        self._prepared_tasks[task_id] = fixed
        return fixed

    def preflight_tasks(self, task_ids: Iterable[str]) -> pd.DataFrame:
        task_list = list(dict.fromkeys(str(task_id) for task_id in task_ids))
        if not task_list:
            raise ValueError("preflight requires at least one task")
        self._ensure_output_layout()
        self._lock_snapshot()
        audit_rows: list[dict[str, Any]] = []
        inventory_rows: list[dict[str, Any]] = []
        parameter_entry_count = 0
        for task_id in task_list:
            fixed = self.prepare_task(task_id)
            frozen_entries = self._entries_for_task(task_id)
            if not frozen_entries:
                raise ValueError(f"No frozen parameter entries found for task={task_id}")
            for entry in frozen_entries:
                validate_frozen_entry(entry, snapshot_id=self.registry.snapshot_id)
            parameter_entry_count += len(frozen_entries)
            fixed_manifest = fixed.to_manifest()
            subsets = build_training_subset_plan(fixed, self.config)
            save_json(
                self.output_root / "subset_plans" / f"{_safe_name(task_id)}.json",
                {
                    "schema_version": 1,
                    "task_id": task_id,
                    "snapshot_id": self.registry.snapshot_id,
                    "split_hashes": fixed_manifest["hashes"],
                    "subsets": [item.to_manifest() for item in subsets],
                },
            )
            for entry in frozen_entries:
                for subset in subsets:
                    inventory_rows.append(self._run_identity(fixed, entry, subset))
            audit_rows.append(
                {
                    "task_id": task_id,
                    "frozen_task_id": str(frozen_entries[0]["task_id"]),
                    "source_hybrid_manifest": str(fixed.selected_loco.manifest_path),
                    "held_out_cluster_id": int(fixed.selected_loco.held_out_cluster_id),
                    "selected_loco_fold_index": int(fixed.selected_loco.fold_index),
                    "random_cv_fold_index": int(fixed.random_cv_fold_index),
                    "top20_test_size": int(len(fixed.top20_test)),
                    "loco_test_size": int(len(fixed.loco_test)),
                    "random_cv_test_size": int(len(fixed.random_cv_test)),
                    "training_pool_size": int(len(fixed.training_pool)),
                    "parameter_entry_count": int(len(frozen_entries)),
                    "manifest_row_ids_verified": bool(
                        fixed.source_validation.get("manifest_row_ids_verified")
                    ),
                    "loco_origin_verified": bool(
                        fixed.source_validation.get("loco_from_post_top20_remaining")
                    ),
                    **{f"{name}_hash": value for name, value in fixed_manifest["hashes"].items()},
                }
            )
        audit = pd.DataFrame(audit_rows)
        audit.to_csv(self.output_root / "csv" / "split_audit.csv", index=False)
        inventory = pd.DataFrame(inventory_rows)
        inventory.to_csv(self.output_root / "csv" / "run_inventory.csv", index=False)
        run_keys = list(iter_training_run_keys(self.config))
        save_json(
            self.output_root / "experiment_manifest.json",
            {
                "schema_version": 2,
                "stage": "preflight",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "output_root": str(self.output_root.resolve()),
                "snapshot_id": self.registry.snapshot_id,
                "frozen_parameter_file": str(self.frozen_parameters_path.resolve()),
                "locked_frozen_parameter_file": str(
                    (self.output_root / "locked_inputs" / "frozen_model_params_v1.yaml").resolve()
                ),
                "locked_source_config": str(
                    (self.output_root / "locked_inputs" / "figure4_config.yaml").resolve()
                )
                if self.source_config_path is not None
                else "",
                "runtime_environment_file": str(
                    (self.output_root / "locked_inputs" / "runtime_environment.json").resolve()
                ),
                "runtime_environment_hash": sha256_file(
                    self.output_root / "locked_inputs" / "runtime_environment.json"
                ),
                "prediction_schema_version": 1,
                "model_weights_saved": False,
                "task_count": len(task_list),
                "parameter_entry_count": int(parameter_entry_count),
                "runs_per_model": len(run_keys),
                "expected_run_count": int(len(inventory)),
                "expected_ratio_row_count": int(len(inventory) * 2),
                "training_fractions": [float(value) for value in self.config.training_fractions],
                "repeats": int(self.config.repeats),
                "full_fraction_repeats": int(self.config.full_fraction_repeats),
                "split_seed": int(self.config.random_state),
                "random_cv_fold_index": int(self.config.random_cv_fold_index or 0),
            },
        )
        return audit

    def _ensure_output_layout(self) -> None:
        for name in [
            "locked_inputs",
            "split_manifests",
            "subset_plans",
            "cache",
            "runs",
            "csv",
            "figures",
        ]:
            (self.output_root / name).mkdir(parents=True, exist_ok=True)

    def validate_locked_inputs(self) -> None:
        if not (self.output_root / "experiment_manifest.json").exists():
            raise FileNotFoundError(
                f"Training-size experiment has not been preflighted: {self.output_root}"
            )
        self._lock_snapshot()

    def _lock_snapshot(self) -> None:
        manifest_path = self.output_root / "experiment_manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if str(existing.get("snapshot_id", "")) != self.registry.snapshot_id:
                raise ValueError(
                    "Experiment snapshot mismatch: "
                    f"existing={existing.get('snapshot_id')} current={self.registry.snapshot_id}"
                )
        locked_path = self.output_root / "locked_inputs" / "frozen_model_params_v1.yaml"
        source_bytes = self.frozen_parameters_path.read_bytes()
        if locked_path.exists() and locked_path.read_bytes() != source_bytes:
            raise ValueError("Locked frozen-parameter snapshot differs from the current snapshot")
        if not locked_path.exists():
            shutil.copy2(self.frozen_parameters_path, locked_path)
        if self.source_config_path is not None:
            locked_config = self.output_root / "locked_inputs" / "figure4_config.yaml"
            config_bytes = self.source_config_path.read_bytes()
            if locked_config.exists() and locked_config.read_bytes() != config_bytes:
                raise ValueError("Locked Figure 4 config differs from the current source config")
            if not locked_config.exists():
                shutil.copy2(self.source_config_path, locked_config)
        runtime_path = self.output_root / "locked_inputs" / "runtime_environment.json"
        if not runtime_path.exists():
            _atomic_save_json(runtime_path, _runtime_environment_payload())

    def _entries_for_task(self, task_id: str) -> list[dict[str, Any]]:
        aliases = {str(task_id)}
        if str(task_id).startswith("MatbenchSteels__"):
            aliases.add(str(task_id).replace("MatbenchSteels__", "Matbench_Steel__", 1))
        return [entry for entry in self.registry.iter_entries() if str(entry.get("task_id")) in aliases]

    def run_tasks(
        self,
        task_ids: Iterable[str],
        *,
        families: Iterable[str] | None = None,
        models: Iterable[str] | None = None,
        run_keys: Iterable[tuple[float, int]] | None = None,
        resume: bool = True,
        continue_on_error: bool = True,
        jobs: int = 1,
    ) -> pd.DataFrame:
        if jobs < 1:
            raise ValueError("jobs must be positive")
        self._ensure_output_layout()
        self._lock_snapshot()
        family_filter = set(str(value) for value in families or [])
        model_filter = set(str(value) for value in models or [])
        requested_keys = list(run_keys) if run_keys is not None else list(iter_training_run_keys(self.config))
        statuses: list[dict[str, Any]] = []
        for task_id in list(dict.fromkeys(str(value) for value in task_ids)):
            fixed = self._prepared_tasks.get(task_id) or build_fixed_split(self.config, task_id)
            plan_by_key = {
                (item.training_fraction, item.repeat_id): item
                for item in self._load_subset_plan(task_id)
            }
            entries = self._entries_for_task(task_id)
            for entry in entries:
                family = str(entry["model_family"])
                model = str(entry["model"])
                if family_filter and family not in family_filter:
                    continue
                if model_filter and model not in model_filter:
                    continue
                validate_frozen_entry(entry, snapshot_id=self.registry.snapshot_id)
                context = self._task_runtime_context(fixed, entry)
                adapter = get_model_adapter(family)
                feature_cache = None
                if hasattr(adapter, "prepare_cache"):
                    feature_cache = adapter.prepare_cache(
                        context,
                        model,
                        dict(entry["effective_parameters"]),
                        fixed,
                    )
                work_items: list[tuple[TrainingSubsetPlan, Path]] = []
                for fraction, repeat_id in requested_keys:
                    key = (float(fraction), int(repeat_id))
                    if key not in plan_by_key:
                        raise KeyError(f"No frozen subset plan for task={task_id}, key={key}")
                    subset = plan_by_key[key]
                    run_dir = (
                        self.output_root
                        / "runs"
                        / _safe_name(task_id)
                        / _safe_name(family)
                        / _safe_name(model)
                        / f"size_{int(round(float(fraction) * 100)):03d}"
                        / f"repeat_{int(repeat_id):02d}"
                    )
                    work_items.append((subset, run_dir))
                if jobs > 1 and family == "Traditional" and len(work_items) > 1:
                    with ThreadPoolExecutor(max_workers=jobs) as executor:
                        futures = [
                            executor.submit(
                                self._execute_run,
                                fixed=fixed,
                                entry=entry,
                                context=context,
                                model=model,
                                adapter=adapter,
                                feature_cache=feature_cache,
                                subset=subset,
                                run_dir=run_dir,
                                resume=resume,
                                continue_on_error=continue_on_error,
                            )
                            for subset, run_dir in work_items
                        ]
                        for future in as_completed(futures):
                            statuses.append(future.result())
                else:
                    for subset, run_dir in work_items:
                        statuses.append(
                            self._execute_run(
                                fixed=fixed,
                                entry=entry,
                                context=context,
                                model=model,
                                adapter=adapter,
                                feature_cache=feature_cache,
                                subset=subset,
                                run_dir=run_dir,
                                resume=resume,
                                continue_on_error=continue_on_error,
                            )
                        )
        status = pd.DataFrame(statuses)
        status.to_csv(self.output_root / "csv" / "run_status.csv", index=False)
        return status

    def _execute_run(
        self,
        *,
        fixed: FixedTrainingSizeSplit,
        entry: dict[str, Any],
        context: TaskRuntimeContext,
        model: str,
        adapter: Any,
        feature_cache: Any,
        subset: TrainingSubsetPlan,
        run_dir: Path,
        resume: bool,
        continue_on_error: bool,
    ) -> dict[str, Any]:
        identity = self._run_identity(fixed, entry, subset)
        if resume and _run_is_complete(run_dir, identity):
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            return {
                **identity,
                "status": "skipped",
                "run_dir": str(run_dir),
                "predictions_sha256": manifest.get("predictions_sha256", ""),
                "metrics_sha256": manifest.get("metrics_sha256", ""),
            }
        if resume:
            recovered = self._recover_failed_run(
                fixed=fixed,
                adapter=adapter,
                subset=subset,
                run_dir=run_dir,
                identity=identity,
            )
            if recovered is not None:
                return recovered
        self._reset_run_dir(run_dir)
        _atomic_save_json(
            run_dir / "run_manifest.json",
            {
                **identity,
                "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        try:
            prepared_split = materialize_training_split(fixed, subset)
            metrics = adapter.run(
                context=context,
                model=model,
                prepared_split=prepared_split,
                effective_parameters=dict(entry["effective_parameters"]),
                result_dir=run_dir,
                manifest_metadata={
                    "training_size_ood": identity,
                    "parameter_snapshot_id": self.registry.snapshot_id,
                    "parameter_hash": str(entry["parameter_hash"]),
                    "frozen_parameter_file": str(self.frozen_parameters_path.resolve()),
                },
                feature_cache=feature_cache,
            )
            return self._finalize_completed_run(
                prepared_split=prepared_split,
                metrics=metrics,
                run_dir=run_dir,
                identity=identity,
            )
        except Exception as exc:
            cleanup_error = ""
            try:
                cleanup_training_artifacts(run_dir)
            except Exception as cleanup_exc:
                cleanup_error = str(cleanup_exc)
            _atomic_save_json(
                run_dir / "run_manifest.json",
                {
                    **identity,
                    "status": "failed",
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "cleanup_error": cleanup_error,
                },
            )
            if not continue_on_error:
                raise
            return {
                **identity,
                "status": "failed",
                "run_dir": str(run_dir),
                "error": str(exc),
            }

    def _recover_failed_run(
        self,
        *,
        fixed: FixedTrainingSizeSplit,
        adapter: Any,
        subset: TrainingSubsetPlan,
        run_dir: Path,
        identity: dict[str, Any],
    ) -> dict[str, Any] | None:
        loader = getattr(adapter, "load_existing_metrics", None)
        manifest_path = run_dir / "run_manifest.json"
        if not callable(loader) or not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if manifest.get("status") != "failed" or not _identity_matches(manifest, identity):
            return None
        try:
            prepared_split = materialize_training_split(fixed, subset)
            metrics = loader(run_dir)
            return self._finalize_completed_run(
                prepared_split=prepared_split,
                metrics=metrics,
                run_dir=run_dir,
                identity=identity,
                recovered_from_existing_outputs=True,
            )
        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
            return None

    def _finalize_completed_run(
        self,
        *,
        prepared_split: Any,
        metrics: Any,
        run_dir: Path,
        identity: dict[str, Any],
        recovered_from_existing_outputs: bool = False,
    ) -> dict[str, Any]:
        missing_sets = {TOP20_TEST_NAME, LOCO_TEST_NAME}.difference(metrics.test_set_mae)
        if missing_sets:
            raise ValueError(f"Adapter metrics are missing test sets: {sorted(missing_sets)}")
        metrics_payload = {
            "random_cv_mae": float(metrics.random_cv_mae),
            "test_set_metrics": {
                name: {"mae": float(value)} for name, value in metrics.test_set_mae.items()
            },
            "source_metrics_file": metrics.metrics_file,
        }
        validate_run_metrics_payload(metrics_payload)
        predictions_path = write_canonical_predictions(
            run_dir,
            prepared_split,
            random_cv_mae=metrics.random_cv_mae,
            test_set_mae=metrics.test_set_mae,
        )
        predictions_sha256 = sha256_file(predictions_path)
        metrics_payload.update(
            {
                "predictions_file": str(predictions_path.resolve()),
                "predictions_sha256": predictions_sha256,
            }
        )
        _atomic_save_json(run_dir / "run_metrics.json", metrics_payload)
        removed_artifacts = cleanup_training_artifacts(run_dir)
        metrics_sha256 = sha256_file(run_dir / "run_metrics.json")
        completed_at = datetime.now(timezone.utc).isoformat()
        artifact_identity = {
            "predictions_file": str(predictions_path.resolve()),
            "predictions_sha256": predictions_sha256,
            "metrics_sha256": metrics_sha256,
            "removed_training_artifacts": removed_artifacts,
        }
        if recovered_from_existing_outputs:
            artifact_identity["recovered_from_existing_outputs"] = True
        _atomic_save_json(
            run_dir / "run_manifest.json",
            {
                **identity,
                **artifact_identity,
                "status": "complete",
                "completed_at": completed_at,
                "metrics_file": str((run_dir / "run_metrics.json").resolve()),
            },
        )
        _atomic_save_json(
            run_dir / "completion_marker.json",
            {
                **identity,
                **artifact_identity,
                "status": "complete",
                "completed_at": completed_at,
            },
        )
        return {
            **identity,
            **artifact_identity,
            "status": "complete",
            "run_dir": str(run_dir),
        }

    def _reset_run_dir(self, run_dir: Path) -> None:
        runs_root = (self.output_root / "runs").resolve()
        resolved = run_dir.resolve()
        resolved.relative_to(runs_root)
        if resolved.exists():
            shutil.rmtree(resolved)
        resolved.mkdir(parents=True, exist_ok=True)

    def _load_subset_plan(self, task_id: str) -> list[TrainingSubsetPlan]:
        path = self.output_root / "subset_plans" / f"{_safe_name(task_id)}.json"
        if not path.exists():
            self.preflight_tasks([task_id])
        payload = json.loads(path.read_text(encoding="utf-8"))
        if str(payload.get("snapshot_id", "")) != self.registry.snapshot_id:
            raise ValueError(f"Subset plan snapshot mismatch for task={task_id}")
        return [
            TrainingSubsetPlan(
                training_fraction=float(item["training_fraction"]),
                repeat_id=int(item["repeat_id"]),
                sampling_seed=int(item["sampling_seed"]),
                train_ids=list(item["train_ids"]),
                subset_hash=str(item["subset_hash"]),
                train_size=int(item["train_size"]),
            )
            for item in payload.get("subsets", [])
        ]

    def _task_runtime_context(
        self,
        fixed: FixedTrainingSizeSplit,
        entry: dict[str, Any],
    ) -> TaskRuntimeContext:
        row = fixed.selected_loco.proximity_row
        alloy_type = str(row.get("alloy_family") or fixed.selected_loco.task_id.split("__", 1)[0])
        if alloy_type == "Matbench Steel":
            alloy_type = "MatbenchSteels"
        if alloy_type not in ALLOY_CONFIGS_OOD:
            raise ValueError(f"No OOD alloy configuration for task={fixed.selected_loco.task_id}: {alloy_type}")
        alloy_config = ALLOY_CONFIGS_OOD[alloy_type]
        return TaskRuntimeContext(
            task_id=fixed.selected_loco.task_id,
            frozen_task_id=str(entry["task_id"]),
            alloy_type=alloy_type,
            dataset_name=str(row.get("dataset_name") or Path(str(alloy_config["raw_data"])).stem),
            target_col=fixed.target_col,
            data_file=Path(str(alloy_config["raw_data"])),
            processing_cols=tuple(str(value) for value in alloy_config.get("processing_cols", [])),
            processing_text_column=alloy_config.get("processing_text_column"),
            model_seed=_frozen_model_seed(entry),
            cache_root=self.output_root / "cache",
        )

    def _run_identity(
        self,
        fixed: FixedTrainingSizeSplit,
        entry: dict[str, Any],
        subset: TrainingSubsetPlan,
    ) -> dict[str, Any]:
        split_hashes = fixed.to_manifest()["hashes"]
        entry_key = (
            str(entry["task_id"]),
            str(entry["model_family"]),
            str(entry["model"]),
        )
        if entry_key not in self._historical_evidence:
            self._historical_evidence[entry_key] = historical_split_evidence(entry)
        return {
            "task_id": fixed.selected_loco.task_id,
            "frozen_parameter_task_id": str(entry["task_id"]),
            "model_family": str(entry["model_family"]),
            "model": str(entry["model"]),
            "training_fraction": float(subset.training_fraction),
            "repeat_id": int(subset.repeat_id),
            "sampling_seed": int(subset.sampling_seed),
            "snapshot_id": self.registry.snapshot_id,
            "parameter_hash": str(entry["parameter_hash"]),
            "subset_hash": subset.subset_hash,
            "random_cv_test_hash": split_hashes["random_cv_test"],
            "top20_test_hash": split_hashes["top20_test"],
            "loco_test_hash": split_hashes["loco_test"],
            "training_pool_hash": split_hashes["training_pool"],
            **self._historical_evidence[entry_key],
        }

def resolve_frozen_task_id(
    registry: FrozenParameterRegistry,
    task_id: str,
    model_family: str,
    model: str,
) -> str:
    candidates = [str(task_id)]
    if str(task_id).startswith("MatbenchSteels__"):
        candidates.append(str(task_id).replace("MatbenchSteels__", "Matbench_Steel__", 1))
    for candidate in candidates:
        try:
            registry.get(candidate, "", "", model_family, model)
        except KeyError:
            continue
        return candidate
    raise KeyError(f"No frozen task alias for {task_id}/{model_family}/{model}")


def validate_frozen_entry(entry: dict[str, Any], *, snapshot_id: str) -> None:
    effective = entry.get("effective_parameters")
    if not isinstance(effective, dict) or not effective:
        raise ValueError("effective_parameters are missing")
    selection_method = str(entry.get("selection_method", ""))
    if not selection_method:
        raise ValueError("selection_method is missing")
    if str(entry.get("parameter_snapshot_id", "")) != str(snapshot_id):
        raise ValueError("parameter_snapshot_id does not match the registry")
    if selection_method == "fixed_runtime":
        hash_payload = effective
        hash_label = "effective_parameters"
    elif selection_method == "existing_randomcv_validation_only":
        search = entry.get("search_parameters")
        if not isinstance(search, dict) or not search:
            raise ValueError("search_parameters are missing for selected parameters")
        hash_payload = search
        hash_label = "search_parameters"
    else:
        raise ValueError(f"unsupported selection_method={selection_method}")
    expected_hash = build_parameter_hash(hash_payload)
    if str(entry.get("parameter_hash", "")) != expected_hash:
        raise ValueError(f"parameter_hash does not match {hash_label}")


def historical_split_evidence(entry: dict[str, Any]) -> dict[str, Any]:
    default = {
        "historical_split_hash_status": "unavailable",
        "historical_train_split_hash": "",
        "historical_test_split_hash": "",
        "historical_split_manifest": "",
    }
    direct_train = str(entry.get("train_split_hash", "") or "")
    direct_test = str(entry.get("test_split_hash", "") or "")
    if direct_train or direct_test:
        return {
            **default,
            "historical_split_hash_status": "available",
            "historical_train_split_hash": direct_train,
            "historical_test_split_hash": direct_test,
        }
    for source in _flatten_source_paths(entry.get("source_paths")):
        path = Path(source)
        start = path if path.is_dir() else path.parent
        for parent in [start, *list(start.parents)[:10]]:
            for name in ["pipeline_manifest.json", "llmprop_manifest.json", "split_manifest.json"]:
                manifest_path = parent / name
                if not manifest_path.exists():
                    continue
                try:
                    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                train_hash = str(_find_nested_value(payload, "train_split_hash") or "")
                test_hash = str(_find_nested_value(payload, "test_split_hash") or "")
                if train_hash or test_hash:
                    return {
                        **default,
                        "historical_split_hash_status": "available",
                        "historical_train_split_hash": train_hash,
                        "historical_test_split_hash": test_hash,
                        "historical_split_manifest": str(manifest_path.resolve()),
                    }
    return default


def _flatten_source_paths(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        paths: list[str] = []
        for item in value.values():
            paths.extend(_flatten_source_paths(item))
        return paths
    if isinstance(value, (list, tuple)):
        paths = []
        for item in value:
            paths.extend(_flatten_source_paths(item))
        return paths
    text_value = str(value).strip()
    if text_value.startswith("[") or text_value.startswith("{"):
        try:
            return _flatten_source_paths(json.loads(text_value))
        except json.JSONDecodeError:
            pass
    return [text_value] if text_value else []


def _find_nested_value(payload: Any, key: str) -> Any:
    if isinstance(payload, dict):
        if key in payload:
            return payload[key]
        for value in payload.values():
            found = _find_nested_value(value, key)
            if found not in (None, ""):
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find_nested_value(value, key)
            if found not in (None, ""):
                return found
    return None


def _safe_name(value: str) -> str:
    return "".join("_" if character in '<>:"/\\|?*' else character for character in str(value))


def _atomic_save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    temporary.replace(path)


def _run_is_complete(run_dir: Path, expected_identity: dict[str, Any]) -> bool:
    manifest_path = run_dir / "run_manifest.json"
    metrics_path = run_dir / "run_metrics.json"
    marker_path = run_dir / "completion_marker.json"
    if not manifest_path.exists() or not metrics_path.exists() or not marker_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("status") != "complete" or marker.get("status") != "complete":
        return False
    if not _identity_matches(manifest, expected_identity) or not _identity_matches(
        marker,
        expected_identity,
    ):
        return False
    try:
        validate_run_metrics_payload(metrics)
    except ValueError:
        return False
    return validate_completed_artifacts(run_dir, manifest, marker)


def _identity_matches(candidate: dict[str, Any], expected_identity: dict[str, Any]) -> bool:
    return all(candidate.get(key) == value for key, value in expected_identity.items())


def _runtime_environment_payload() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ["numpy", "pandas", "scikit-learn", "torch", "transformers"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "unavailable"
    git_revision = "unavailable"
    git_dirty: bool | str = "unavailable"
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        git_revision = revision.stdout.strip()
        git_dirty = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    cuda: dict[str, Any] = {"available": False}
    try:
        import torch

        cuda = {
            "available": bool(torch.cuda.is_available()),
            "torch_cuda_version": str(torch.version.cuda or "unavailable"),
            "device_name": (
                str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "unavailable"
            ),
        }
    except ImportError:
        pass
    payload = {
        "schema_version": 1,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "git_revision": git_revision,
        "git_dirty": git_dirty,
        "packages": packages,
        "cuda": cuda,
    }
    return {**payload, "environment_hash": canonical_json_hash(payload)}


def _frozen_model_seed(entry: dict[str, Any]) -> int:
    effective = dict(entry.get("effective_parameters", {}))
    candidates = [
        effective.get("random_state"),
        dict(effective.get("args", {})).get("random_state"),
        dict(effective.get("model_params", {})).get("random_state"),
        dict(effective.get("model_params", {})).get("random_seed"),
    ]
    for value in candidates:
        if value is not None:
            return int(value)
    return 42
