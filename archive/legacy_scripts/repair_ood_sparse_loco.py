"""
Repair sparse/LOCO OOD outputs for traditional ML and TabPFN.

The script detects missing or invalid target-level runs, backs up existing
outputs, clears batch progress entries, reruns the affected tasks, and writes a
repair manifest.  It is intentionally conservative: any suspicious target run
is moved aside as a unit before rerun so old split artifacts cannot be mixed
with new model outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.TabPFN.model_factory import get_tabpfn_runtime_config  # noqa: E402
from src.TabPFN.ood_data_processor import TabPFNOODDataProcessor  # noqa: E402
from src.TabPFN.tabpfn_ood_configs import get_tabpfn_ood_config  # noqa: E402
from src.data_processing.strength_ood_registry import create_ood_processor  # noqa: E402
from src.pipelines.batch_configs_ood import (  # noqa: E402
    ALLOY_CONFIGS_OOD,
    BATCH_CONFIGS_OOD,
    get_ood_method_meta,
)
from src.pipelines.run_batch_ood import build_result_dir as build_ml_result_dir  # noqa: E402
from src.pipelines.run_batch_ood import build_command as build_ml_command  # noqa: E402


PERFORMANCE_COLUMNS = {"El(%)", "UTS(MPa)", "YS(MPa)", "yield strength"}
OOD_METHODS = [
    "sparse_x_single",
    "sparse_y_single",
    "sparse_x_cluster",
    "sparse_y_cluster",
    "loco",
]
ML_CONFIGS = {
    "sparse_x_single": "experiment1_all_ml_models_sparse_x_single",
    "sparse_y_single": "experiment1_all_ml_models_sparse_y_single",
    "sparse_x_cluster": "experiment1_all_ml_models_sparse_x_cluster",
    "sparse_y_cluster": "experiment1_all_ml_models_sparse_y_cluster",
    "loco": "experiment1_all_ml_models_loco",
}
TABPFN_CONFIGS = {
    "sparse_x_single": "tabpfn_all_sparse_x_single",
    "sparse_y_single": "tabpfn_all_sparse_y_single",
    "sparse_x_cluster": "tabpfn_all_sparse_x_cluster",
    "sparse_y_cluster": "tabpfn_all_sparse_y_cluster",
    "loco": "tabpfn_all_loco",
}
METHOD_DIRS = {
    "sparse_x_single": "sparse_x_single_k5",
    "sparse_y_single": "sparse_y_single_k5",
    "sparse_x_cluster": "sparse_x_cluster_k5",
    "sparse_y_cluster": "sparse_y_cluster_k5",
    "loco": "loco_k5",
}
TRAIN_TEST_FILES = {
    "sparse_x_single": ("train_inlier.csv", "test_sparse_x_single.csv"),
    "sparse_y_single": ("train_inlier.csv", "test_sparse_y_single.csv"),
    "sparse_x_cluster": ("train_inlier.csv", "test_sparse_x_cluster.csv"),
    "sparse_y_cluster": ("train_inlier.csv", "test_sparse_y_cluster.csv"),
    "loco": ("train.csv", "test.csv"),
}
ML_MODELS = ["xgboost", "sklearn_rf", "lightgbm", "mlp", "catboost"]


@dataclass
class Issue:
    kind: str
    method: str
    alloy: str
    target: str
    result_dir: Path
    reasons: List[str] = field(default_factory=list)
    runtime: str | None = None
    backend: str | None = None
    feature_mode: str | None = None

    @property
    def key(self) -> str:
        prefix = self.kind
        if self.runtime:
            prefix = f"{prefix}:{self.runtime}"
        return f"{prefix}:{self.method}:{self.alloy}:{self.target}"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


def read_json(path: Path) -> Dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._=-" else "_" for ch in value).strip("_") or "item"


def dataset_name(raw_data: str) -> str:
    name = Path(raw_data).stem
    for suffix in ["_Processed_cleaned", "_with_ID", "_withID", "_cleaned", "_processed", "_Processed"]:
        name = name.replace(suffix, "")
    return name


def task_key(alloy: str, target: str, model: str | None = None) -> str:
    return f"{alloy}_{target}_{model}" if model else f"{alloy}_{target}"


def clear_progress_entries(progress_file: Path, config_names: Iterable[str], task_keys: Iterable[str] | None = None) -> None:
    if not progress_file.exists():
        return
    payload = read_json(progress_file) or {}
    keys = set(task_keys or [])
    changed = False
    for config_name in config_names:
        if config_name not in payload:
            continue
        if not keys:
            payload.pop(config_name, None)
            changed = True
            continue
        before = dict(payload.get(config_name, {}))
        for key in keys:
            payload.get(config_name, {}).pop(key, None)
        if before != payload.get(config_name, {}):
            changed = True
        if not payload.get(config_name):
            payload.pop(config_name, None)
    if changed:
        write_json(progress_file, payload)


def set_progress_entry(progress_file: Path, config_name: str, key: str, status: str) -> None:
    payload = read_json(progress_file) or {}
    payload.setdefault(config_name, {})[key] = status
    write_json(progress_file, payload)


def backup_path_for(path: Path, backup_root: Path) -> Path:
    try:
        relative = path.resolve().relative_to(REPO_ROOT.resolve())
    except Exception:
        relative = Path(path.name)
    candidate = backup_root / relative
    if not candidate.exists():
        return candidate
    stamp = datetime.now().strftime("%H%M%S_%f")
    return candidate.with_name(f"{candidate.name}__{stamp}")


def backup_existing(path: Path, backup_root: Path, execute: bool) -> str | None:
    if not path.exists():
        return None
    destination = backup_path_for(path, backup_root)
    if execute:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(destination))
    return str(destination)


def expected_split_ids_from_prepared(prepared: Any) -> List[Dict[str, Any]]:
    folds = prepared if isinstance(prepared, list) else [prepared]
    expected: List[Dict[str, Any]] = []
    for idx, item in enumerate(folds):
        split = item.split if hasattr(item, "split") else item
        fold_index = getattr(item, "fold_index", idx)
        if "ID" not in split.train_df.columns or "ID" not in split.test_df.columns:
            raise ValueError("Expected split data to contain ID column")
        expected.append(
            {
                "fold_index": int(fold_index),
                "train_ids": [int(x) for x in split.train_df["ID"].tolist()],
                "test_ids": [int(x) for x in split.test_df["ID"].tolist()],
            }
        )
    return expected


def expected_ml_split(method: str, alloy: str, target: str) -> List[Dict[str, Any]]:
    alloy_cfg = ALLOY_CONFIGS_OOD[alloy]
    raw_df = pd.read_csv(REPO_ROOT / alloy_cfg["raw_data"])
    processor = create_ood_processor(
        method,
        str(REPO_ROOT / alloy_cfg["raw_data"]),
        random_state=42,
        processing_cols=alloy_cfg.get("processing_cols") or [],
    )
    prepared = processor.prepare(
        df=raw_df,
        target_col=target,
        test_ratio=0.2,
        extrapolation_side="low_to_high",
        sparse_candidate_pool_size=500,
        sparse_cluster_count=5,
        sparse_samples_per_cluster=1,
        sparse_kde_bandwidth=None,
        sparse_neighbors_per_seed=5,
        loco_cluster_count=5,
        baseline_num_folds=5,
        processing_cols=alloy_cfg.get("processing_cols") or [],
    )
    return expected_split_ids_from_prepared(prepared)


def expected_tabpfn_split(method: str, alloy: str, target: str, backend: str, feature_mode: str) -> List[Dict[str, Any]]:
    cfg = get_tabpfn_ood_config(alloy, backend=backend, feature_mode=feature_mode, base_path=str(REPO_ROOT))
    processor = TabPFNOODDataProcessor(cfg, base_path=str(REPO_ROOT))
    processor.load_data()
    frame = processor.prepare_feature_frame(target, drop_na=True)
    prepared = processor.prepare_ood_result(
        frame,
        target,
        method,
        test_size=0.2,
        extrapolation_side="low_to_high",
        sparse_candidate_pool_size=500,
        sparse_cluster_count=5,
        sparse_samples_per_cluster=1,
        sparse_kde_bandwidth=None,
        sparse_neighbors_per_seed=5,
        loco_cluster_count=5,
        baseline_num_folds=5,
    )
    return expected_split_ids_from_prepared(prepared)


def normalize_split_ids(splits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in splits:
        normalized.append(
            {
                "fold_index": int(item["fold_index"]),
                "train_ids": sorted(int(x) for x in item["train_ids"]),
                "test_ids": sorted(int(x) for x in item["test_ids"]),
            }
        )
    return sorted(normalized, key=lambda x: x["fold_index"])


def load_actual_split_ids(run_root: Path, method: str) -> List[Dict[str, Any]]:
    train_file, test_file = TRAIN_TEST_FILES[method]
    if method == "loco":
        fold_dirs = sorted((run_root / "folds").glob("fold_*"), key=lambda p: int(p.name.split("_")[-1]))
        actual = []
        for fold_dir in fold_dirs:
            idx = int(fold_dir.name.split("_")[-1])
            split_dir = fold_dir / "split_data"
            actual.append(
                {
                    "fold_index": idx,
                    "train_ids": [int(x) for x in pd.read_csv(split_dir / train_file)["ID"].tolist()],
                    "test_ids": [int(x) for x in pd.read_csv(split_dir / test_file)["ID"].tolist()],
                }
            )
        return actual
    split_dir = run_root / "split_data"
    return [
        {
            "fold_index": 0,
            "train_ids": [int(x) for x in pd.read_csv(split_dir / train_file)["ID"].tolist()],
            "test_ids": [int(x) for x in pd.read_csv(split_dir / test_file)["ID"].tolist()],
        }
    ]


def compare_split_ids(expected: List[Dict[str, Any]], actual: List[Dict[str, Any]]) -> List[str]:
    reasons: List[str] = []
    expected = normalize_split_ids(expected)
    actual = normalize_split_ids(actual)
    if len(expected) != len(actual):
        return [f"fold_count_mismatch expected={len(expected)} actual={len(actual)}"]
    for exp, act in zip(expected, actual):
        if exp["fold_index"] != act["fold_index"]:
            reasons.append(f"fold_index_mismatch expected={exp['fold_index']} actual={act['fold_index']}")
        if exp["train_ids"] != act["train_ids"]:
            reasons.append(f"train_id_mismatch fold={exp['fold_index']}")
        if exp["test_ids"] != act["test_ids"]:
            reasons.append(f"test_id_mismatch fold={exp['fold_index']}")
    return reasons


def trace_paths(run_root: Path, method: str) -> List[Path]:
    if method == "loco":
        return sorted((run_root / "folds").glob("fold_*/split_data/trace/x_space_features.parquet"))
    return [run_root / "split_data" / "trace" / "x_space_features.parquet"]


def validate_trace_columns(run_root: Path, method: str, target: str) -> List[str]:
    reasons: List[str] = []
    paths = trace_paths(run_root, method)
    if not paths:
        return ["missing_trace_x_space_features"]
    for path in paths:
        if not path.exists():
            reasons.append(f"missing_trace:{rel(path)}")
            continue
        try:
            cols = [col for col in pd.read_parquet(path).columns if col in PERFORMANCE_COLUMNS]
        except Exception as exc:
            reasons.append(f"unreadable_trace:{rel(path)}:{exc}")
            continue
        if cols != [target]:
            reasons.append(f"bad_trace_performance_columns:{rel(path)}:{cols}")
    return reasons


def validate_ml_artifacts(run_root: Path, method: str, target: str) -> List[str]:
    reasons: List[str] = []
    if method == "loco":
        manifest = run_root / "loco_manifest.json"
        if not manifest.exists():
            reasons.append("missing_loco_manifest")
        fold_roots = sorted((run_root / "folds").glob("fold_*"), key=lambda p: int(p.name.split("_")[-1]))
        if not fold_roots:
            reasons.append("missing_loco_folds")
    else:
        fold_roots = [run_root]
        if not (run_root / "pipeline_manifest.json").exists():
            reasons.append("missing_pipeline_manifest")
    for fold_root in fold_roots:
        for model in ML_MODELS:
            model_dir = fold_root / "model_comparison" / f"{model}_results"
            manifest = model_dir / "ood_manifest.json"
            if not manifest.exists():
                reasons.append(f"missing_model_manifest:{rel(manifest)}")
                continue
            payload = read_json(manifest) or {}
            recorded = payload.get("target_column")
            if recorded not in (None, target):
                reasons.append(f"model_manifest_target_mismatch:{rel(manifest)}:{recorded}")
            metrics = [
                model_dir / "final_model_evaluation_metrics.json",
                model_dir / "final_evaluation_metrics.json",
                model_dir / "cv_avg_metrics.json",
            ]
            if not any(p.exists() and p.stat().st_size > 0 for p in metrics):
                reasons.append(f"missing_model_metrics:{rel(model_dir)}")
    return reasons


def validate_tabpfn_artifacts(run_root: Path, method: str) -> List[str]:
    reasons: List[str] = []
    if method == "loco":
        manifest = run_root / "loco_manifest.json"
        if not manifest.exists():
            reasons.append("missing_loco_manifest")
        fold_roots = sorted((run_root / "folds").glob("fold_*"), key=lambda p: int(p.name.split("_")[-1]))
        if not fold_roots:
            reasons.append("missing_loco_folds")
    else:
        fold_roots = [run_root]
    for fold_root in fold_roots:
        required = [
            fold_root / "pipeline_manifest.json",
            fold_root / "metrics" / "metrics_summary.json",
            fold_root / "predictions" / "all_predictions.csv",
        ]
        for path in required:
            if not path.exists() or path.stat().st_size == 0:
                reasons.append(f"missing_tabpfn_artifact:{rel(path)}")
    return reasons


def ml_result_dir(method: str, alloy: str, target: str) -> Path:
    config_name = ML_CONFIGS[method]
    cfg = BATCH_CONFIGS_OOD[config_name]
    args = argparse.Namespace(**cfg)
    alloy_cfg = ALLOY_CONFIGS_OOD[alloy].copy()
    alloy_cfg["target_column"] = target
    return REPO_ROOT / build_ml_result_dir(
        config_name,
        alloy,
        alloy_cfg,
        args,
        get_ood_method_meta(method),
    )


def tabpfn_result_dir(method: str, alloy: str, target: str, runtime_info: Dict[str, Any]) -> Path:
    cfg = get_tabpfn_ood_config(
        alloy,
        backend=runtime_info["resolved_backend"],
        feature_mode=runtime_info["feature_mode"],
        base_path=str(REPO_ROOT),
    )
    return (
        REPO_ROOT
        / "output"
        / f"ood_results_{runtime_info['model_run_dirname']}"
        / METHOD_DIRS[method]
        / alloy
        / dataset_name(cfg["raw_data"])
        / target
    )


def detect_ml_issues() -> List[Issue]:
    issues: List[Issue] = []
    for method in OOD_METHODS:
        for alloy, alloy_cfg in ALLOY_CONFIGS_OOD.items():
            for target in alloy_cfg["targets"]:
                result_dir = ml_result_dir(method, alloy, target)
                reasons: List[str] = []
                if not result_dir.exists():
                    reasons.append("missing_result_dir")
                else:
                    try:
                        expected = expected_ml_split(method, alloy, target)
                        actual = load_actual_split_ids(result_dir, method)
                        reasons.extend(compare_split_ids(expected, actual))
                    except Exception as exc:
                        reasons.append(f"split_validation_error:{exc}")
                    reasons.extend(validate_trace_columns(result_dir, method, target))
                    reasons.extend(validate_ml_artifacts(result_dir, method, target))
                if reasons:
                    issues.append(Issue("ml", method, alloy, target, result_dir, reasons))
    return issues


def available_tabpfn_runtimes(include_api: bool) -> List[Dict[str, str]]:
    runtimes = [{"backend": "local", "feature_mode": "numeric"}]
    if include_api:
        runtimes.extend(
            [
                {"backend": "api", "feature_mode": "numeric"},
                {"backend": "api", "feature_mode": "text"},
            ]
        )
    return runtimes


def detect_tabpfn_issues(include_api: bool) -> List[Issue]:
    issues: List[Issue] = []
    for runtime in available_tabpfn_runtimes(include_api):
        try:
            runtime_info = get_tabpfn_runtime_config(
                base_path=REPO_ROOT,
                backend=runtime["backend"],
                feature_mode=runtime["feature_mode"],
            )
        except Exception as exc:
            print(f"[WARN] Skipping TabPFN runtime {runtime}: {exc}")
            continue
        runtime_name = runtime_info["model_run_dirname"]
        for method in OOD_METHODS:
            for alloy in ALLOY_CONFIGS_OOD:
                try:
                    cfg = get_tabpfn_ood_config(
                        alloy,
                        backend=runtime["backend"],
                        feature_mode=runtime["feature_mode"],
                        base_path=str(REPO_ROOT),
                    )
                except Exception as exc:
                    print(f"[WARN] Skipping {runtime_name} {alloy}: {exc}")
                    continue
                for target in cfg["targets"]:
                    result_dir = tabpfn_result_dir(method, alloy, target, runtime_info)
                    reasons: List[str] = []
                    if not result_dir.exists():
                        reasons.append("missing_result_dir")
                    else:
                        try:
                            expected = expected_ml_split(method, alloy, target)
                            actual = load_actual_split_ids(result_dir, method)
                            reasons.extend(compare_split_ids(expected, actual))
                        except Exception as exc:
                            reasons.append(f"split_validation_error:{exc}")
                        reasons.extend(validate_trace_columns(result_dir, method, target))
                        reasons.extend(validate_tabpfn_artifacts(result_dir, method))
                    if reasons:
                        issues.append(
                            Issue(
                                "tabpfn",
                                method,
                                alloy,
                                target,
                                result_dir,
                                reasons,
                                runtime=runtime_name,
                                backend=runtime["backend"],
                                feature_mode=runtime["feature_mode"],
                            )
                        )
    return issues


def run_command(
    cmd: Sequence[str],
    execute: bool,
    cwd: Path = REPO_ROOT,
    *,
    log_path: Path | None = None,
) -> Dict[str, Any]:
    display = " ".join(f'"{part}"' if any(ch in str(part) for ch in " ()%\\") else str(part) for part in cmd)
    print(f"[CMD] {display}")
    if not execute:
        return {
            "command": list(cmd),
            "returncode": None,
            "dry_run": True,
            "log_path": rel(log_path) if log_path else None,
        }
    started = datetime.now()
    started_monotonic = time.monotonic()
    log_path = log_path or (REPO_ROOT / "output" / "ood_repair_command.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        log_file.write(f"# started_at={started.isoformat()}\n")
        log_file.write(f"# cwd={cwd}\n")
        log_file.write(f"# command={display}\n\n")
        log_file.flush()
        result = subprocess.run(
            list(cmd),
            cwd=str(cwd),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        finished = datetime.now()
        log_file.write(
            f"\n# finished_at={finished.isoformat()} returncode={result.returncode} "
            f"elapsed_seconds={time.monotonic() - started_monotonic:.1f}\n"
        )
    return {
        "command": list(cmd),
        "returncode": result.returncode,
        "dry_run": False,
        "log_path": rel(log_path),
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "elapsed_seconds": round(time.monotonic() - started_monotonic, 3),
    }


def rerun_ml(
    issues: List[Issue],
    execute: bool,
    jobs: int,
    backup_root: Path,
    manifest: Dict[str, Any] | None = None,
    manifest_path: Path | None = None,
) -> List[Dict[str, Any]]:
    commands: List[Dict[str, Any]] = []
    log_dir = backup_root / "logs" / "ml"
    for issue in sorted(issues, key=lambda x: (OOD_METHODS.index(x.method), x.alloy, x.target)):
        method = issue.method
        config_name = ML_CONFIGS[method]
        cfg = BATCH_CONFIGS_OOD[config_name]
        args = argparse.Namespace(**cfg)
        alloy_cfg = ALLOY_CONFIGS_OOD[issue.alloy].copy()
        alloy_cfg["target_column"] = issue.target
        cmd = build_ml_command(
            config_name,
            issue.alloy,
            alloy_cfg,
            args,
            get_ood_method_meta(method),
            model_name=None,
        )
        log_path = log_dir / f"{safe_name(config_name)}__{safe_name(issue.alloy)}__{safe_name(issue.target)}.log"
        command_record = {
            "kind": "ml_target",
            "method": method,
            "alloy": issue.alloy,
            "target": issue.target,
            "models": ML_MODELS,
            "jobs_requested": jobs,
            "status": "running" if execute else "dry_run",
            "log_path": rel(log_path),
        }
        if manifest is not None:
            manifest.setdefault("commands", []).append(command_record)
            if manifest_path is not None:
                write_json(manifest_path, manifest)
        result = run_command(cmd, execute=execute, log_path=log_path)
        command_record.update(result)
        command_record["status"] = "success" if result.get("returncode") == 0 else "failed"
        commands.append(command_record)
        if execute:
            status = "success" if result.get("returncode") == 0 else "failed"
            for model in ML_MODELS:
                set_progress_entry(
                    REPO_ROOT / ".batch_progress_ood.json",
                    config_name,
                    task_key(issue.alloy, issue.target, model),
                    status,
                )
        if manifest is not None and manifest_path is not None:
            write_json(manifest_path, manifest)
    return commands


def rerun_tabpfn(
    issues: List[Issue],
    execute: bool,
    backup_root: Path,
    manifest: Dict[str, Any] | None = None,
    manifest_path: Path | None = None,
) -> List[Dict[str, Any]]:
    commands: List[Dict[str, Any]] = []
    log_dir = backup_root / "logs" / "tabpfn"
    issue_map: Dict[tuple[str, str, str], List[Issue]] = {}
    for issue in issues:
        key = (issue.backend or "local", issue.feature_mode or "numeric", issue.method)
        issue_map.setdefault(key, []).append(issue)

    for (backend, feature_mode, method), grouped in sorted(issue_map.items()):
        for issue in sorted(grouped, key=lambda x: (x.alloy, x.target)):
            cmd = [
                sys.executable,
                "-m",
                "src.TabPFN.train_tabpfn_ood",
                "--alloy_type",
                issue.alloy,
                "--target_col",
                issue.target,
                "--backend",
                backend,
                "--feature_mode",
                feature_mode,
                "--base_path",
                str(REPO_ROOT),
                "--test_size",
                "0.2",
                "--random_state",
                "42",
                "--split_strategy",
                method,
                "--extrapolation_side",
                "low_to_high",
                "--sparse_candidate_pool_size",
                "500",
                "--sparse_cluster_count",
                "5",
                "--sparse_samples_per_cluster",
                "1",
                "--sparse_neighbors_per_seed",
                "5",
                "--loco_cluster_count",
                "5",
                "--baseline_num_folds",
                "5",
            ]
            log_path = (
                log_dir
                / f"{safe_name(issue.runtime or backend)}__{safe_name(method)}__{safe_name(issue.alloy)}__{safe_name(issue.target)}.log"
            )
            command_record = {
                "kind": "tabpfn_target",
                "runtime": issue.runtime,
                "backend": backend,
                "feature_mode": feature_mode,
                "method": method,
                "alloy": issue.alloy,
                "target": issue.target,
                "status": "running" if execute else "dry_run",
                "log_path": rel(log_path),
            }
            if manifest is not None:
                manifest.setdefault("commands", []).append(command_record)
                if manifest_path is not None:
                    write_json(manifest_path, manifest)
            result = run_command(cmd, execute=execute, log_path=log_path)
            command_record.update(result)
            command_record["status"] = "success" if result.get("returncode") == 0 else "failed"
            commands.append(command_record)
            if execute:
                set_progress_entry(
                    REPO_ROOT / ".batch_progress_tabpfn_ood.json",
                    TABPFN_CONFIGS[method],
                    task_key(issue.alloy, issue.target),
                    "success" if result.get("returncode") == 0 else "failed",
                )
            if manifest is not None and manifest_path is not None:
                write_json(manifest_path, manifest)
    return commands


def issue_to_dict(issue: Issue) -> Dict[str, Any]:
    return {
        "kind": issue.kind,
        "method": issue.method,
        "alloy": issue.alloy,
        "target": issue.target,
        "runtime": issue.runtime,
        "backend": issue.backend,
        "feature_mode": issue.feature_mode,
        "result_dir": rel(issue.result_dir),
        "reasons": issue.reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair OOD sparse/LOCO ML and TabPFN outputs")
    parser.add_argument("--execute", action="store_true", help="Actually backup and rerun. Without this, dry-run only.")
    parser.add_argument("--include_ml", action="store_true", help="Check and repair traditional ML outputs.")
    parser.add_argument("--include_tabpfn", action="store_true", help="Check and repair TabPFN outputs.")
    parser.add_argument("--allow_api", action="store_true", help="Include TabPFN API numeric/text runtimes.")
    parser.add_argument("--jobs", type=int, default=1, help="Traditional ML batch jobs.")
    parser.add_argument("--verify_only", action="store_true", help="Only detect and report issues; do not backup/rerun.")
    args = parser.parse_args()

    if not args.include_ml and not args.include_tabpfn:
        parser.error("Specify --include_ml and/or --include_tabpfn")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = REPO_ROOT / "output" / "ood_repair_backups" / f"repair_{timestamp}"
    manifest_path = backup_root / "repair_manifest.json"

    print(f"[INFO] repo={REPO_ROOT}")
    print(f"[INFO] backup_root={backup_root}")
    print("[INFO] detecting issues...")

    ml_issues = detect_ml_issues() if args.include_ml else []
    tabpfn_issues = detect_tabpfn_issues(args.allow_api) if args.include_tabpfn else []
    all_issues = ml_issues + tabpfn_issues
    print(f"[INFO] detected ml_issues={len(ml_issues)} tabpfn_issues={len(tabpfn_issues)} total={len(all_issues)}")
    for issue in all_issues[:50]:
        print(f"[ISSUE] {issue.key} -> {', '.join(issue.reasons[:4])}")
    if len(all_issues) > 50:
        print(f"[INFO] ... {len(all_issues) - 50} more issues")

    manifest: Dict[str, Any] = {
        "started_at": timestamp,
        "execute": args.execute,
        "include_ml": args.include_ml,
        "include_tabpfn": args.include_tabpfn,
        "allow_api": args.allow_api,
        "jobs": args.jobs,
        "backup_root": rel(backup_root),
        "initial_issues": [issue_to_dict(issue) for issue in all_issues],
        "backups": [],
        "commands": [],
        "post_verify_issues": [],
    }

    if args.verify_only:
        backup_root.mkdir(parents=True, exist_ok=True)
        write_json(manifest_path, manifest)
        print(f"[INFO] verify-only manifest written to {manifest_path}")
        return 1 if all_issues else 0

    backup_root.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)

    if all_issues:
        progress_backup_dir = backup_root / "progress_files"
        if args.execute:
            progress_backup_dir.mkdir(parents=True, exist_ok=True)
            for progress_name in [".batch_progress_ood.json", ".batch_progress_tabpfn_ood.json"]:
                progress_path = REPO_ROOT / progress_name
                if progress_path.exists():
                    shutil.copy2(progress_path, progress_backup_dir / progress_name)

    for issue in ml_issues:
        source = issue.result_dir
        backup_dest = backup_existing(source, backup_root, execute=args.execute)
        if backup_dest:
            manifest["backups"].append(
                {
                    "kind": "ml_target",
                    "issue": issue_to_dict(issue),
                    "method": issue.method,
                    "alloy": issue.alloy,
                    "target": issue.target,
                    "source": rel(source),
                    "destination": rel(Path(backup_dest)),
                }
            )
            write_json(manifest_path, manifest)

    for issue in tabpfn_issues:
        backup_dest = backup_existing(issue.result_dir, backup_root, execute=args.execute)
        if backup_dest:
            manifest["backups"].append(
                {
                    "kind": "tabpfn_target",
                    "issue": issue_to_dict(issue),
                    "source": rel(issue.result_dir),
                    "destination": rel(Path(backup_dest)),
                }
            )
            write_json(manifest_path, manifest)

    if args.execute:
        if ml_issues:
            ml_progress_path = REPO_ROOT / ".batch_progress_ood.json"
            for method in sorted({issue.method for issue in ml_issues}, key=OOD_METHODS.index):
                affected_keys = [
                    task_key(issue.alloy, issue.target, model)
                    for issue in ml_issues
                    if issue.method == method
                    for model in ML_MODELS
                ]
                clear_progress_entries(ml_progress_path, [ML_CONFIGS[method]], affected_keys)

        if tabpfn_issues:
            tab_progress_path = REPO_ROOT / ".batch_progress_tabpfn_ood.json"
            for method in sorted({issue.method for issue in tabpfn_issues}, key=OOD_METHODS.index):
                affected_keys = [
                    task_key(issue.alloy, issue.target)
                    for issue in tabpfn_issues
                    if issue.method == method
                ]
                clear_progress_entries(tab_progress_path, [TABPFN_CONFIGS[method]], affected_keys)

    if ml_issues:
        rerun_ml(
            ml_issues,
            execute=args.execute,
            jobs=args.jobs,
            backup_root=backup_root,
            manifest=manifest,
            manifest_path=manifest_path,
        )
    if tabpfn_issues:
        rerun_tabpfn(
            tabpfn_issues,
            execute=args.execute,
            backup_root=backup_root,
            manifest=manifest,
            manifest_path=manifest_path,
        )

    failed_commands = [cmd for cmd in manifest["commands"] if cmd.get("returncode") not in (0, None)]
    if failed_commands:
        print(f"[WARN] {len(failed_commands)} commands failed; skipping final verification for failed outputs")

    print("[INFO] post-run verification...")
    post_ml = detect_ml_issues() if args.include_ml else []
    post_tab = detect_tabpfn_issues(args.allow_api) if args.include_tabpfn else []
    post = post_ml + post_tab
    manifest["post_verify_issues"] = [issue_to_dict(issue) for issue in post]
    manifest["finished_at"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest["failed_commands"] = failed_commands
    backup_root.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)

    print(f"[INFO] manifest written to {manifest_path}")
    print(f"[INFO] post_verify ml_issues={len(post_ml)} tabpfn_issues={len(post_tab)} total={len(post)}")
    for issue in post[:50]:
        print(f"[POST-ISSUE] {issue.key} -> {', '.join(issue.reasons[:4])}")
    if failed_commands or post:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
