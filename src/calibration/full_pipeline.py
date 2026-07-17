from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
from typing import Any

import numpy as np
import pandas as pd

from src.data_processing.canonical_split_registry import (
    CanonicalSplitEntry,
    CanonicalSplitRegistry,
)

from .parameter_registry import FrozenParameterRegistry
from .adapters import SAMPLE_COLUMNS, UnitIdentity


PROTOCOL_DIR_NAMES = {
    "target_extrapolation": "Extrapolation",
    "random_cv_baseline": "RandomCV",
    "loco": "LOCO",
    "sparse_x_single": "SparseXsingle",
    "sparse_x_cluster": "SparseXcluster",
    "sparse_y_single": "SparseYsingle",
    "sparse_y_cluster": "SparseYcluster",
}


@dataclass(frozen=True)
class CalibrationUnitContext:
    identity: UnitIdentity
    workspace_root: Path
    output_root: Path
    cache_root: Path
    split_entry: CanonicalSplitEntry
    frozen_entry: dict[str, Any]
    traditional_seeds: tuple[int, ...]
    mc_passes: int

    @property
    def workspace(self) -> Path:
        return self.workspace_root

    @property
    def seeds(self) -> tuple[int, ...]:
        return self.traditional_seeds

    @property
    def mc_seeds(self) -> tuple[int, ...]:
        return tuple(range(42, 42 + int(self.mc_passes)))

    @property
    def split(self) -> CanonicalSplitEntry:
        return self.split_entry

    @property
    def frozen(self) -> dict[str, Any]:
        return self.frozen_entry


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_component(value: object) -> str:
    text = re.sub(r'[<>:"/\\|?*]+', "_", str(value)).strip(" .")
    if not text:
        raise ValueError(f"Empty calibration path component: {value!r}")
    return text


def _in_scope(task_id: str, scope: str) -> bool:
    normalized = scope.strip().lower()
    if normalized == "all":
        return True
    if normalized == "ti":
        return task_id.startswith("Ti__")
    raise ValueError(f"Unsupported calibration scope: {scope}")


def _split_record(entry: CanonicalSplitEntry) -> dict[str, Any]:
    identity = entry.identity_record()
    return {
        "split_key": "||".join((entry.task_id, entry.protocol, entry.split_id)),
        "task_id": entry.task_id,
        "alloy_type": entry.alloy_type,
        "dataset_name": entry.dataset_name,
        "target_column": entry.target_column,
        "canonical_protocol": entry.protocol,
        "protocol": PROTOCOL_DIR_NAMES[entry.protocol],
        "split_id": entry.split_id,
        "train_file": str(entry.train_file.resolve()),
        "test_file": str(entry.test_file.resolve()),
        "summary_file": str(entry.summary_file.resolve()),
        "split_manifest_file": str(entry.manifest_file.resolve()),
        "train_count": int(identity["train_count"]),
        "test_count": int(identity["test_count"]),
        "train_identity_hash": str(identity["train_hash"]),
        "test_identity_hash": str(identity["test_hash"]),
        "train_file_sha256": _file_sha256(entry.train_file),
        "test_file_sha256": _file_sha256(entry.test_file),
        "random_state": int(entry.random_state),
    }


def build_canonical_split_inventory(
    registry: CanonicalSplitRegistry,
    *,
    scope: str = "all",
) -> pd.DataFrame:
    rows = [
        _split_record(entry)
        for entry in registry.entries()
        if _in_scope(entry.task_id, scope)
    ]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["task_id", "canonical_protocol", "split_id"],
        kind="stable",
    ).reset_index(drop=True)


def build_calibration_run_inventory(
    split_registry: CanonicalSplitRegistry,
    parameter_registry: FrozenParameterRegistry,
    *,
    output_root: str | Path,
    scope: str = "all",
) -> pd.DataFrame:
    output = Path(output_root).resolve()
    frozen_by_task: dict[str, list[dict[str, Any]]] = {}
    for frozen in parameter_registry.iter_entries():
        task_id = str(frozen.get("task_id", ""))
        if _in_scope(task_id, scope):
            frozen_by_task.setdefault(task_id, []).append(frozen)

    rows: list[dict[str, Any]] = []
    for split in split_registry.entries():
        if not _in_scope(split.task_id, scope):
            continue
        split_record = _split_record(split)
        for frozen in sorted(
            frozen_by_task.get(split.task_id, []),
            key=lambda item: (str(item.get("model_family", "")), str(item.get("model", ""))),
        ):
            model = str(frozen["model"])
            protocol_dir = str(split_record["protocol"])
            result_dir = (
                output
                / "results"
                / _safe_component(split.task_id)
                / _safe_component(protocol_dir)
                / _safe_component(split.split_id)
                / _safe_component(model)
            )
            rows.append(
                {
                    **split_record,
                    "run_key": "||".join(
                        (split.task_id, model, split.protocol, split.split_id)
                    ),
                    "model_family": str(frozen["model_family"]),
                    "model": model,
                    "parameter_hash": str(frozen.get("parameter_hash", "")),
                    "parameter_snapshot_id": parameter_registry.snapshot_id,
                    "frozen_parameter_file": str(parameter_registry.path.resolve()),
                    "result_dir": str(result_dir),
                }
            )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).sort_values(
        ["task_id", "canonical_protocol", "split_id", "model_family", "model"],
        kind="stable",
    )
    if not frame["run_key"].is_unique:
        raise ValueError("Calibration run inventory contains duplicate run keys")
    return frame.reset_index(drop=True)


def _atomic_csv_write(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    _replace_with_retry(temporary, path)


def _replace_with_retry(source: Path, destination: Path) -> None:
    delays = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6)
    for attempt, delay in enumerate(delays):
        try:
            source.replace(destination)
            return
        except PermissionError:
            if attempt == len(delays) - 1:
                raise
            time.sleep(delay)


def _promote_ready_directory(
    *,
    result_dir: Path,
    ready_dir: Path,
    backup_dir: Path,
) -> None:
    promoted = False
    try:
        _replace_with_retry(result_dir, backup_dir)
        _replace_with_retry(ready_dir, result_dir)
        promoted = True
    except Exception:
        if not result_dir.exists() and backup_dir.exists():
            _replace_with_retry(backup_dir, result_dir)
        raise
    finally:
        if promoted and backup_dir.exists():
            shutil.rmtree(backup_dir)


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _replace_with_retry(temporary, path)


def _sample_metrics(sample: pd.DataFrame) -> dict[str, Any]:
    y_true = pd.to_numeric(sample["y_true"], errors="raise").to_numpy(float)
    y_pred = pd.to_numeric(sample["y_pred"], errors="raise").to_numpy(float)
    residual = y_pred - y_true
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    nonzero = y_true != 0
    return {
        "n_samples": int(len(sample)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1.0 - np.sum(residual**2) / denominator)
        if denominator > 0
        else float("nan"),
        "mape": float(np.mean(np.abs(residual[nonzero] / y_true[nonzero])) * 100)
        if np.any(nonzero)
        else float("nan"),
    }


def _validate_unit_frames(
    *,
    members: pd.DataFrame,
    sample: pd.DataFrame,
    split_entry: CanonicalSplitEntry,
    identity: UnitIdentity,
    expected_member_count: int,
) -> None:
    missing_sample = sorted(set(SAMPLE_COLUMNS).difference(sample.columns))
    if missing_sample:
        raise ValueError(f"sample_uncertainty.csv is missing columns: {missing_sample}")
    canonical = split_entry.load_test()
    canonical_ids = canonical["ID"].astype(str).tolist()
    sample_ids = sample["sample_id"].astype(str).tolist()
    if sample_ids != canonical_ids:
        raise ValueError("Sample prediction ID order differs from canonical test.")
    canonical_true = pd.to_numeric(
        canonical[split_entry.target_column],
        errors="raise",
    ).to_numpy(float)
    sample_true = pd.to_numeric(sample["y_true"], errors="raise").to_numpy(float)
    if not np.allclose(sample_true, canonical_true, rtol=1e-10, atol=1e-12):
        raise ValueError("Sample y_true differs from canonical test.")
    numeric = sample[["y_true", "y_pred", "abs_error", "uncertainty_score"]].apply(
        pd.to_numeric,
        errors="raise",
    )
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("Sample predictions contain NaN/Inf values.")
    if (numeric["uncertainty_score"] < 0).any():
        raise ValueError("Sample uncertainty must be nonnegative.")
    if not np.allclose(
        numeric["abs_error"],
        np.abs(numeric["y_pred"] - numeric["y_true"]),
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("Sample abs_error is inconsistent with y_true/y_pred.")
    if set(sample["parameter_snapshot_id"].astype(str)) != {
        identity.parameter_snapshot_id
    }:
        raise ValueError("Sample snapshot ID differs from the active frozen snapshot.")
    if set(sample["parameter_hash"].astype(str)) != {identity.parameter_hash}:
        raise ValueError("Sample parameter hash differs from frozen v1.")
    if set(pd.to_numeric(sample["member_count"], errors="raise").astype(int)) != {
        expected_member_count
    }:
        raise ValueError("Sample member_count differs from the adapter contract.")

    member_ids = members["sample_id"].astype(str)
    observed_order = list(dict.fromkeys(member_ids.tolist()))
    if observed_order != canonical_ids:
        raise ValueError("Member prediction ID order differs from canonical test.")
    counts = members.groupby(member_ids, sort=False).size().to_numpy()
    if not np.all(counts == expected_member_count):
        raise ValueError("Member prediction count differs across canonical samples.")


def _expected_member_count(
    model_family: str,
    *,
    traditional_seeds: tuple[int, ...],
    mc_passes: int,
) -> int:
    if model_family == "Traditional":
        return len(traditional_seeds)
    if model_family in {"BERT", "LLMProp"}:
        return int(mc_passes)
    if model_family == "TabPFN":
        return 1
    raise ValueError(f"Unsupported calibration model family: {model_family}")


def _unit_file_hashes(result_dir: Path, names: list[str]) -> dict[str, str]:
    return {name: _file_sha256(result_dir / name) for name in names}


def _workspace_from_output_root(output_root: Path) -> Path:
    if output_root.name.lower() != "calibration" or output_root.parent.name.lower() != "output":
        raise ValueError(f"Calibration output root has an unexpected layout: {output_root}")
    return output_root.parent.parent.resolve()


def execute_calibration_unit(
    *,
    run_row: pd.Series,
    split_entry: CanonicalSplitEntry,
    frozen_entry: dict[str, Any],
    parameter_registry: FrozenParameterRegistry,
    output_root: str | Path,
    attempt_number: int,
    adapter_runner: Any,
    traditional_seeds: tuple[int, ...] = tuple(range(42, 52)),
    mc_passes: int = 30,
) -> dict[str, Any]:
    output = Path(output_root).resolve()
    result_dir = Path(str(run_row["result_dir"])).resolve()
    try:
        result_dir.relative_to(output / "results")
    except ValueError as exc:
        raise ValueError(f"Calibration result directory escapes output/results: {result_dir}") from exc
    identity = UnitIdentity(
        task_id=str(run_row["task_id"]),
        protocol=str(run_row["protocol"]),
        split_id=str(run_row["split_id"]),
        model_family=str(run_row["model_family"]),
        model=str(run_row["model"]),
        parameter_snapshot_id=parameter_registry.snapshot_id,
        parameter_hash=str(frozen_entry["parameter_hash"]),
    )
    if str(run_row["parameter_snapshot_id"]) != identity.parameter_snapshot_id:
        raise ValueError("Run inventory snapshot differs from frozen v1.")
    if str(run_row["parameter_hash"]) != identity.parameter_hash:
        raise ValueError("Run inventory parameter hash differs from frozen v1.")

    result_dir.mkdir(parents=True, exist_ok=True)
    attempt_dir = result_dir / f".attempt_{int(attempt_number)}"
    if attempt_dir.exists():
        if attempt_dir.resolve().parent != result_dir:
            raise ValueError(f"Unsafe calibration attempt path: {attempt_dir}")
        shutil.rmtree(attempt_dir)
    attempt_dir.mkdir(parents=True)
    context = CalibrationUnitContext(
        identity=identity,
        workspace_root=_workspace_from_output_root(output),
        output_root=output,
        cache_root=output / "cache",
        split_entry=split_entry,
        frozen_entry=dict(frozen_entry),
        traditional_seeds=tuple(int(value) for value in traditional_seeds),
        mc_passes=int(mc_passes),
    )
    adapter_result = adapter_runner(context, attempt_dir)
    members = adapter_result.members.copy()
    sample = adapter_result.sample.copy()
    for name, value in identity.metadata().items():
        members[name] = value
        sample[name] = value
    quantiles = getattr(adapter_result, "quantiles", None)
    expected_members = _expected_member_count(
        identity.model_family,
        traditional_seeds=context.traditional_seeds,
        mc_passes=context.mc_passes,
    )
    _validate_unit_frames(
        members=members,
        sample=sample,
        split_entry=split_entry,
        identity=identity,
        expected_member_count=expected_members,
    )

    _atomic_csv_write(attempt_dir / "member_predictions.csv", members)
    _atomic_csv_write(attempt_dir / "sample_uncertainty.csv", sample)
    contract_files = ["member_predictions.csv", "sample_uncertainty.csv"]
    if quantiles is not None:
        _atomic_csv_write(attempt_dir / "predictive_quantiles.csv", quantiles)
        contract_files.append("predictive_quantiles.csv")
    metrics = _sample_metrics(sample)
    _atomic_json_write(attempt_dir / "metrics.json", metrics)
    contract_files.append("metrics.json")
    split_identity = split_entry.identity_record()
    manifest = {
        "schema_version": 1,
        "run_key": str(run_row["run_key"]),
        **identity.metadata(),
        "canonical_protocol": str(run_row["canonical_protocol"]),
        "parameter_snapshot_id": identity.parameter_snapshot_id,
        "parameter_hash": identity.parameter_hash,
        "frozen_parameter_file": str(parameter_registry.path.resolve()),
        "canonical_train_file": str(split_entry.train_file.resolve()),
        "canonical_test_file": str(split_entry.test_file.resolve()),
        "canonical_train_identity_hash": split_identity["train_hash"],
        "canonical_test_identity_hash": split_identity["test_hash"],
        "member_count": expected_members,
        "uncertainty_method": str(sample["uncertainty_method"].iloc[0]),
        "adapter_metadata": dict(getattr(adapter_result, "metadata", {}) or {}),
        "metrics": metrics,
        "files": _unit_file_hashes(attempt_dir, contract_files),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "status": "success",
    }
    _atomic_json_write(attempt_dir / "manifest.json", manifest)
    contract_files.append("manifest.json")

    ready_dir = result_dir.parent / f".{_safe_component(identity.model)}.ready_{int(attempt_number)}"
    backup_dir = result_dir.parent / f".{_safe_component(identity.model)}.backup_{int(attempt_number)}"
    for staging in (ready_dir, backup_dir):
        if staging.exists():
            raise FileExistsError(f"Calibration promotion staging path already exists: {staging}")
    ready_dir.mkdir()
    for name in contract_files:
        (attempt_dir / name).replace(ready_dir / name)
    logs_dir = attempt_dir / "logs"
    if logs_dir.exists():
        logs_dir.replace(ready_dir / "logs")
    if attempt_dir.exists():
        shutil.rmtree(attempt_dir)
    _promote_ready_directory(
        result_dir=result_dir,
        ready_dir=ready_dir,
        backup_dir=backup_dir,
    )
    return {
        "run_key": str(run_row["run_key"]),
        "run_status": "success",
        "attempt_count": int(attempt_number),
        "result_dir": str(result_dir),
    }


def calibration_unit_is_complete(
    run_row: pd.Series,
    split_entry: CanonicalSplitEntry,
    frozen_entry: dict[str, Any],
    snapshot_id: str,
) -> bool:
    result_dir = Path(str(run_row["result_dir"]))
    manifest_path = result_dir / "manifest.json"
    required = [
        result_dir / "member_predictions.csv",
        result_dir / "sample_uncertainty.csv",
        result_dir / "metrics.json",
        manifest_path,
    ]
    if not all(path.exists() for path in required):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "success":
            return False
        if manifest.get("run_key") != str(run_row["run_key"]):
            return False
        if manifest.get("parameter_snapshot_id") != str(snapshot_id):
            return False
        if manifest.get("parameter_hash") != str(frozen_entry["parameter_hash"]):
            return False
        split_identity = split_entry.identity_record()
        if manifest.get("canonical_train_identity_hash") != split_identity["train_hash"]:
            return False
        if manifest.get("canonical_test_identity_hash") != split_identity["test_hash"]:
            return False
        for name, expected_hash in dict(manifest.get("files", {})).items():
            path = result_dir / name
            if not path.exists() or _file_sha256(path) != str(expected_hash):
                return False
        identity = UnitIdentity(
            task_id=str(run_row["task_id"]),
            protocol=str(run_row["protocol"]),
            split_id=str(run_row["split_id"]),
            model_family=str(run_row["model_family"]),
            model=str(run_row["model"]),
            parameter_snapshot_id=str(snapshot_id),
            parameter_hash=str(frozen_entry["parameter_hash"]),
        )
        _validate_unit_frames(
            members=pd.read_csv(result_dir / "member_predictions.csv"),
            sample=pd.read_csv(result_dir / "sample_uncertainty.csv"),
            split_entry=split_entry,
            identity=identity,
            expected_member_count=int(manifest["member_count"]),
        )
    except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return True


def _archive_completed_ready_directory(
    *,
    output_root: Path,
    run_row: pd.Series,
    split_entry: CanonicalSplitEntry,
    frozen_entry: dict[str, Any],
    snapshot_id: str,
    attempt_number: int,
) -> str:
    output = Path(output_root).resolve()
    result_dir = Path(str(run_row["result_dir"])).resolve()
    output_results = (output / "results").resolve()
    try:
        result_dir.relative_to(output_results)
    except ValueError as exc:
        raise ValueError(
            f"Ready-directory result path escapes output/results: {result_dir}"
        ) from exc

    ready_dir = result_dir.parent / (
        f".{_safe_component(str(run_row['model']))}.ready_{int(attempt_number)}"
    )
    if not ready_dir.exists():
        return ""
    if ready_dir.resolve().parent != result_dir.parent.resolve():
        raise ValueError(f"Unsafe calibration ready path: {ready_dir}")

    ready_row = run_row.copy()
    ready_row["result_dir"] = str(ready_dir)
    if not calibration_unit_is_complete(
        ready_row,
        split_entry,
        frozen_entry,
        snapshot_id,
    ):
        raise ValueError(
            f"Calibration ready directory failed validation and was preserved: {ready_dir}"
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    safe_key = _safe_component(str(run_row["run_key"]))
    destination = (
        output
        / "failed_attempts"
        / stamp
        / safe_key
        / f"promotion_ready_attempt_{int(attempt_number)}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    _replace_with_retry(ready_dir, destination)
    return str(destination)


PROTOCOL_AGGREGATE_GROUPS = ["task_id", "protocol", "model_family", "model"]
TIER_ORDER = ("High", "Medium", "Low")


def _assign_protocol_tiers(sample: pd.DataFrame) -> pd.DataFrame:
    output = sample.copy()
    output["confidence_tier"] = pd.NA
    output["_stable_index"] = np.arange(len(output))
    for _, group in output.groupby(
        PROTOCOL_AGGREGATE_GROUPS,
        sort=False,
        dropna=False,
    ):
        ordered = group.assign(
            _uncertainty=pd.to_numeric(group["uncertainty_score"], errors="raise"),
            _split=group["split_id"].astype(str),
            _sample=group["sample_id"].astype(str),
        ).sort_values(
            ["_uncertainty", "_split", "_sample", "_stable_index"],
            kind="mergesort",
        )
        for tier, positions in zip(TIER_ORDER, np.array_split(np.arange(len(ordered)), 3)):
            output.loc[ordered.iloc[positions].index, "confidence_tier"] = tier
    return output.drop(columns=["_stable_index"])


def _mean_ci95(values: pd.Series) -> tuple[float, float, float, float, float]:
    numeric = pd.to_numeric(values, errors="raise")
    mean = float(numeric.mean())
    sd = float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0
    se = sd / np.sqrt(len(numeric))
    return mean, sd, se, mean - 1.96 * se, mean + 1.96 * se


def _protocol_tier_summary(sample: pd.DataFrame) -> pd.DataFrame:
    overall = (
        sample.groupby(PROTOCOL_AGGREGATE_GROUPS, dropna=False)["abs_error"]
        .mean()
        .rename("overall_mae")
    )
    rows: list[dict[str, Any]] = []
    for keys, group in sample.groupby(
        PROTOCOL_AGGREGATE_GROUPS + ["confidence_tier"],
        sort=False,
        dropna=False,
    ):
        mean, sd, se, low, high = _mean_ci95(group["abs_error"])
        base = keys[:-1]
        denominator = float(overall.loc[base])
        rows.append(
            {
                **dict(zip(PROTOCOL_AGGREGATE_GROUPS, base)),
                "confidence_tier": str(keys[-1]),
                "n": int(len(group)),
                "mae": mean,
                "sd": sd,
                "se": se,
                "ci95_low": low,
                "ci95_high": high,
                "overall_mae": denominator,
                "normalized_mae": mean / denominator if denominator else np.nan,
                "normalized_ci95_low": low / denominator if denominator else np.nan,
                "normalized_ci95_high": high / denominator if denominator else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _protocol_uncertainty_metrics(sample: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in sample.groupby(
        PROTOCOL_AGGREGATE_GROUPS,
        sort=False,
        dropna=False,
    ):
        numeric = group[["uncertainty_score", "abs_error"]].apply(
            pd.to_numeric,
            errors="raise",
        )
        correlation = (
            float(numeric.corr(method="spearman").iloc[0, 1])
            if len(numeric) > 1
            else np.nan
        )
        rows.append(
            {
                **dict(zip(PROTOCOL_AGGREGATE_GROUPS, keys)),
                "sample_n": int(len(numeric)),
                "spearman_uncertainty_error": correlation,
            }
        )
    return pd.DataFrame(rows)


def aggregate_full_results(output_root: str | Path) -> dict[str, Any]:
    output = Path(output_root).resolve()
    inventory_path = output / "audit" / "calibration_run_inventory.csv"
    if not inventory_path.exists():
        raise FileNotFoundError(f"Calibration run inventory not found: {inventory_path}")
    inventory = pd.read_csv(inventory_path, keep_default_na=False)
    frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []
    for _, row in inventory.iterrows():
        sample_path = Path(str(row["result_dir"])) / "sample_uncertainty.csv"
        completed = sample_path.exists()
        coverage_rows.append(
            {
                "task_id": row["task_id"],
                "protocol": row["protocol"],
                "model_family": row["model_family"],
                "model": row["model"],
                "split_id": row["split_id"],
                "completed": int(completed),
            }
        )
        if completed:
            frames.append(pd.read_csv(sample_path))
    if not frames:
        raise ValueError("No completed calibration sample files are available for aggregation.")
    raw = pd.concat(frames, ignore_index=True, sort=False)
    calibrated = _assign_protocol_tiers(raw)

    protocol_uncertainty: list[pd.DataFrame] = []
    protocol_calibration: list[pd.DataFrame] = []
    protocol_tiers: list[pd.DataFrame] = []
    protocol_metrics: list[pd.DataFrame] = []
    for (task_id, protocol), group in calibrated.groupby(
        ["task_id", "protocol"],
        sort=False,
        dropna=False,
    ):
        raw_group = raw[
            raw["task_id"].astype(str).eq(str(task_id))
            & raw["protocol"].astype(str).eq(str(protocol))
        ].copy()
        protocol_root = output / "results" / str(task_id) / str(protocol)
        aggregate_root = protocol_root / "aggregate"
        tier = _protocol_tier_summary(group)
        continuous = _protocol_uncertainty_metrics(group)
        _atomic_csv_write(aggregate_root / "sample_level_uncertainty.csv", raw_group)
        _atomic_csv_write(aggregate_root / "sample_level_calibration.csv", group)
        _atomic_csv_write(aggregate_root / "tier_error_summary.csv", tier)
        _atomic_csv_write(aggregate_root / "uncertainty_error_metrics.csv", continuous)
        _atomic_json_write(
            protocol_root / "protocol_manifest.json",
            {
                "schema_version": 1,
                "task_id": str(task_id),
                "protocol": str(protocol),
                "split_count": int(group["split_id"].nunique()),
                "model_count": int(group[["model_family", "model"]].drop_duplicates().shape[0]),
                "sample_count": int(len(group)),
                "tier_summary_count": int(len(tier)),
                "uncertainty_metric_count": int(len(continuous)),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        protocol_uncertainty.append(raw_group)
        protocol_calibration.append(group)
        protocol_tiers.append(tier)
        protocol_metrics.append(continuous)

    coverage = pd.DataFrame(coverage_rows)
    coverage_summary = (
        coverage.groupby(PROTOCOL_AGGREGATE_GROUPS, dropna=False)
        .agg(expected_units=("split_id", "size"), completed_units=("completed", "sum"))
        .reset_index()
    )
    coverage_summary["missing_units"] = (
        coverage_summary["expected_units"] - coverage_summary["completed_units"]
    )
    aggregate_root = output / "aggregate"
    all_uncertainty = pd.concat(protocol_uncertainty, ignore_index=True, sort=False)
    all_calibration = pd.concat(protocol_calibration, ignore_index=True, sort=False)
    all_tiers = pd.concat(protocol_tiers, ignore_index=True, sort=False)
    all_metrics = pd.concat(protocol_metrics, ignore_index=True, sort=False)
    _atomic_csv_write(aggregate_root / "all_sample_level_uncertainty.csv", all_uncertainty)
    _atomic_csv_write(aggregate_root / "all_sample_level_calibration.csv", all_calibration)
    _atomic_csv_write(aggregate_root / "all_tier_error_summary.csv", all_tiers)
    _atomic_csv_write(aggregate_root / "all_uncertainty_error_metrics.csv", all_metrics)
    _atomic_csv_write(aggregate_root / "all_coverage_summary.csv", coverage_summary)
    return {
        "sample_count": int(len(all_calibration)),
        "tier_summary_count": int(len(all_tiers)),
        "uncertainty_metric_count": int(len(all_metrics)),
        "coverage_count": int(len(coverage_summary)),
    }


def _set_status_row(status: pd.DataFrame, run_key: str, **values: Any) -> None:
    mask = status["run_key"].astype(str).eq(str(run_key))
    if int(mask.sum()) != 1:
        raise ValueError(f"Calibration run status row is not unique: {run_key}")
    for name, value in values.items():
        status.loc[mask, name] = value


def _archive_failed_attempt(
    *,
    output_root: Path,
    result_dir: Path,
    attempt_number: int,
    run_key: str,
) -> str:
    output_results = (output_root / "results").resolve()
    resolved_result = result_dir.resolve()
    try:
        resolved_result.relative_to(output_results)
    except ValueError as exc:
        raise ValueError(
            f"Failed-attempt result directory escapes output/results: {resolved_result}"
        ) from exc
    attempt_dir = result_dir / f".attempt_{int(attempt_number)}"
    if not attempt_dir.exists():
        return ""
    if attempt_dir.resolve().parent != result_dir.resolve():
        raise ValueError(f"Unsafe failed calibration attempt path: {attempt_dir}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    safe_key = _safe_component(run_key)
    destination = output_root / "failed_attempts" / stamp / safe_key
    destination.parent.mkdir(parents=True, exist_ok=True)
    _replace_with_retry(attempt_dir, destination)
    return str(destination)


def run_full_units(
    *,
    output_root: str | Path,
    split_root: str | Path,
    snapshot_path: str | Path,
    scope: str,
    resume: bool = True,
    jobs: int = 1,
    adapter_runner: Any | None = None,
    traditional_seeds: tuple[int, ...] = tuple(range(42, 52)),
    mc_passes: int = 30,
    run_keys: set[str] | None = None,
    families: set[str] | None = None,
    models: set[str] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    if jobs != 1:
        raise ValueError("Full calibration currently requires --jobs 1.")
    output = Path(output_root).resolve()
    splits = Path(split_root).resolve()
    snapshot = Path(snapshot_path).resolve()
    manifest_path = output / "manifest.json"
    inventory_path = output / "audit" / "calibration_run_inventory.csv"
    status_path = output / "status" / "run_status.csv"
    for required in (manifest_path, inventory_path, status_path, snapshot):
        if not required.exists():
            raise FileNotFoundError(required)
    root_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry = FrozenParameterRegistry(snapshot)
    if str(root_manifest.get("scope")) != str(scope):
        raise ValueError(
            f"Calibration index scope mismatch: indexed={root_manifest.get('scope')} requested={scope}"
        )
    if str(root_manifest.get("snapshot_id")) != registry.snapshot_id:
        raise ValueError("Calibration root manifest snapshot differs from frozen v1.")
    if str(root_manifest.get("frozen_parameter_file_sha256")) != _file_sha256(snapshot):
        raise ValueError("Frozen v1 file changed after full-index; reset and rebuild are required.")
    if Path(str(root_manifest.get("split_root", ""))).resolve() != splits:
        raise ValueError("Calibration root manifest split root differs from the requested split root.")

    split_registry = CanonicalSplitRegistry.from_root(splits)
    inventory = pd.read_csv(inventory_path, keep_default_na=False)
    status = pd.read_csv(status_path, keep_default_na=False)
    selected = inventory.copy()
    if run_keys:
        selected = selected[selected["run_key"].astype(str).isin(run_keys)]
    if families:
        selected = selected[selected["model_family"].astype(str).isin(families)]
    if models:
        selected = selected[selected["model"].astype(str).isin(models)]
    if limit is not None:
        selected = selected.head(int(limit))
    if adapter_runner is None:
        from .adapters import run_adapter

        adapter_runner = run_adapter

    for _, row in selected.iterrows():
        run_key = str(row["run_key"])
        split_entry = split_registry.require(
            str(row["task_id"]),
            str(row["canonical_protocol"]),
            str(row["split_id"]),
        )
        frozen_entry = registry.get(
            str(row["task_id"]),
            "",
            "",
            str(row["model_family"]),
            str(row["model"]),
        )
        status_mask = status["run_key"].astype(str).eq(run_key)
        if int(status_mask.sum()) != 1:
            raise ValueError(f"Run status is missing or duplicated: {run_key}")
        previous_attempts = int(
            pd.to_numeric(status.loc[status_mask, "attempt_count"], errors="coerce")
            .fillna(0)
            .iloc[0]
        )
        if resume and calibration_unit_is_complete(
            row,
            split_entry,
            frozen_entry,
            registry.snapshot_id,
        ):
            _set_status_row(
                status,
                run_key,
                run_status="success",
                last_error="",
            )
            _atomic_csv_write(status_path, status)
            continue

        for failed_attempt in range(1, previous_attempts + 1):
            _archive_completed_ready_directory(
                output_root=output,
                run_row=row,
                split_entry=split_entry,
                frozen_entry=frozen_entry,
                snapshot_id=registry.snapshot_id,
                attempt_number=failed_attempt,
            )

        attempt_number = previous_attempts + 1
        started_at = datetime.now(timezone.utc).isoformat()
        _set_status_row(
            status,
            run_key,
            run_status="running",
            attempt_count=attempt_number,
            started_at=started_at,
            finished_at="",
            last_error="",
        )
        _atomic_csv_write(status_path, status)
        try:
            execute_calibration_unit(
                run_row=row,
                split_entry=split_entry,
                frozen_entry=frozen_entry,
                parameter_registry=registry,
                output_root=output,
                attempt_number=attempt_number,
                adapter_runner=adapter_runner,
                traditional_seeds=traditional_seeds,
                mc_passes=mc_passes,
            )
        except Exception as exc:
            archived = _archive_failed_attempt(
                output_root=output,
                result_dir=Path(str(row["result_dir"])).resolve(),
                attempt_number=attempt_number,
                run_key=run_key,
            )
            message = f"{type(exc).__name__}: {exc}"
            if archived:
                message = f"{message}; failed_attempt={archived}"
            _set_status_row(
                status,
                run_key,
                run_status="failed",
                finished_at=datetime.now(timezone.utc).isoformat(),
                last_error=message,
            )
            _atomic_csv_write(status_path, status)
            continue
        _set_status_row(
            status,
            run_key,
            run_status="success",
            finished_at=datetime.now(timezone.utc).isoformat(),
            last_error="",
        )
        _atomic_csv_write(status_path, status)
    return status
