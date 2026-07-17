"""Reusable split planning for the Figure 4 training-size experiment.

The LOCO partition itself is deliberately *not* implemented here.  It is read
from the existing ``hybrid_extrapolation_loco`` split cache produced by the
data-processing package.  This preserves the established order: top-20% is
removed first, then LOCO is run only on the remaining rows.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.data_processing.strength_ood_common import PreparedSplit
from src.data_processing.canonical_split_registry import sample_identity_hash
from src.data_processing.strength_random_cv_baseline_data_processor import (
    StrengthRandomCVBaselineDataProcessor,
)


FAILURE_AXIS_TOP20 = "top20_property_extreme"
FAILURE_AXIS_LOCO = "top_tail_distant_loco"
TOP20_TEST_NAME = "ood_top20"
LOCO_TEST_NAME = "ood_loco_farthest"


@dataclass(frozen=True)
class TrainingSizeOODConfig:
    proximity_csv: Path
    random_state: int = 20260707
    random_cv_folds: int = 5
    random_cv_fold_index: int | None = 2
    training_fractions: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)
    repeats: int = 10
    full_fraction_repeats: int = 1

    def __post_init__(self) -> None:
        if self.random_cv_folds < 2:
            raise ValueError("random_cv_folds must be at least 2")
        if self.repeats < 1:
            raise ValueError("repeats must be positive")
        if self.full_fraction_repeats != 1:
            raise ValueError("full_fraction_repeats must be exactly 1")
        if not self.training_fractions or any(not 0 < value <= 1 for value in self.training_fractions):
            raise ValueError("training_fractions must be within (0, 1]")


@dataclass(frozen=True)
class SelectedLocoFold:
    task_id: str
    target_col: str
    manifest_path: Path
    held_out_cluster_id: int
    fold_index: int
    proximity_row: dict[str, Any]
    fold_entry: dict[str, Any]


@dataclass(frozen=True)
class TrainingSubsetPlan:
    training_fraction: float
    repeat_id: int
    sampling_seed: int
    train_ids: list[Any]
    subset_hash: str
    train_size: int

    def to_manifest(self) -> dict[str, Any]:
        return {
            "training_fraction": float(self.training_fraction),
            "repeat_id": int(self.repeat_id),
            "sampling_seed": int(self.sampling_seed),
            "train_ids": [_json_value(value) for value in self.train_ids],
            "subset_hash": self.subset_hash,
            "train_size": int(self.train_size),
        }


@dataclass
class FixedTrainingSizeSplit:
    selected_loco: SelectedLocoFold
    target_col: str
    top20_test: pd.DataFrame
    loco_test: pd.DataFrame
    random_cv_test: pd.DataFrame
    training_pool: pd.DataFrame
    random_cv_fold_index: int
    random_state: int
    source_validation: dict[str, Any] = field(default_factory=dict)

    def sample_training_split(self, training_fraction: float, repeat_id: int) -> PreparedSplit:
        if not 0 < training_fraction <= 1:
            raise ValueError("training_fraction must be within (0, 1]")
        if repeat_id < 0:
            raise ValueError("repeat_id must be non-negative")
        sample_size = max(1, int(round(len(self.training_pool) * float(training_fraction))))
        sample_size = min(sample_size, len(self.training_pool))
        seed = _sample_seed(self.random_state, repeat_id)
        positions = np.random.default_rng(seed).permutation(len(self.training_pool))[:sample_size]
        train_df = self.training_pool.iloc[positions].copy()
        train_df = train_df.sort_index(kind="stable").reset_index(drop=True)
        subset_hash = _frame_identity_hash(train_df, self.target_col)
        split = PreparedSplit(
            split_strategy="training_size_hybrid_ood",
            target_col=self.target_col,
            train_df=train_df,
            test_df=self.random_cv_test.copy(),
            test_sets={
                TOP20_TEST_NAME: self.top20_test.copy(),
                LOCO_TEST_NAME: self.loco_test.copy(),
            },
            summary={
                "split_strategy": "training_size_hybrid_ood",
                "training_fraction": float(training_fraction),
                "repeat_id": int(repeat_id),
                "sampling_seed": int(seed),
                "subset_hash": subset_hash,
                "training_pool_size": int(len(self.training_pool)),
                "train_size": int(len(train_df)),
                "random_cv_test_size": int(len(self.random_cv_test)),
                "top20_test_size": int(len(self.top20_test)),
                "loco_test_size": int(len(self.loco_test)),
                "held_out_cluster_id": int(self.selected_loco.held_out_cluster_id),
                "selected_loco_fold_index": int(self.selected_loco.fold_index),
                "source_hybrid_manifest": str(self.selected_loco.manifest_path),
            },
            train_label="train",
            test_label="random_cv",
        )
        validate_prepared_split(split)
        return split

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "task_id": self.selected_loco.task_id,
            "target_column": self.target_col,
            "source_hybrid_manifest": str(self.selected_loco.manifest_path),
            "held_out_cluster_id": self.selected_loco.held_out_cluster_id,
            "selected_loco_fold_index": self.selected_loco.fold_index,
            "random_cv_fold_index": self.random_cv_fold_index,
            "random_state": self.random_state,
            "proximity_row": self.selected_loco.proximity_row,
            "source_validation": self.source_validation,
            "sizes": {
                "top20_test": len(self.top20_test),
                "loco_test": len(self.loco_test),
                "random_cv_test": len(self.random_cv_test),
                "training_pool": len(self.training_pool),
            },
            "identities": {
                "top20_test": _row_keys(self.top20_test),
                "loco_test": _row_keys(self.loco_test),
                "random_cv_test": _row_keys(self.random_cv_test),
                "training_pool": _row_keys(self.training_pool),
            },
            "hashes": {
                "top20_test": _frame_identity_hash(self.top20_test, self.target_col),
                "loco_test": _frame_identity_hash(self.loco_test, self.target_col),
                "random_cv_test": _frame_identity_hash(self.random_cv_test, self.target_col),
                "training_pool": _frame_identity_hash(self.training_pool, self.target_col),
            },
        }


def select_farthest_loco_fold(proximity_csv: str | Path, task_id: str) -> SelectedLocoFold:
    """Resolve the farthest cached hybrid LOCO fold for one task.

    The proximity analysis already evaluates distances from the fixed top-20%
    outer set to clusters produced from the remaining data.  Selecting its
    largest rank therefore cannot accidentally select a cluster from top-20%.
    """

    proximity_path = Path(proximity_csv)
    data = pd.read_csv(proximity_path)
    task_column = "task_id" if "task_id" in data.columns else "case_key"
    required = {task_column, "threshold_pct", "distance_rank", "cluster_id", "manifest_path"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Proximity CSV is missing columns: {sorted(missing)}")
    rows = data[
        data[task_column].astype(str).eq(str(task_id))
        & pd.to_numeric(data["threshold_pct"], errors="coerce").eq(20)
    ].copy()
    if rows.empty:
        raise KeyError(f"No threshold_pct=20 proximity rows for task '{task_id}'")
    rows["distance_rank"] = pd.to_numeric(rows["distance_rank"], errors="coerce")
    chosen = rows.sort_values(["distance_rank", "cluster_id"], ascending=[False, True], kind="stable").iloc[0]
    manifest_path = _resolve_path(chosen["manifest_path"], proximity_path.parent)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cluster_id = int(chosen["cluster_id"])
    fold_entry = next(
        (fold for fold in manifest.get("folds", []) if int(fold.get("held_out_cluster_id", -1)) == cluster_id),
        None,
    )
    if fold_entry is None:
        raise ValueError(f"Cluster {cluster_id} is absent from hybrid manifest: {manifest_path}")
    target_col = str(chosen.get("target_col") or manifest.get("target_column") or "")
    if not target_col:
        raise ValueError(f"Cannot determine target column for task '{task_id}'")
    return SelectedLocoFold(
        task_id=str(task_id),
        target_col=target_col,
        manifest_path=manifest_path,
        held_out_cluster_id=cluster_id,
        fold_index=int(fold_entry.get("fold_index")),
        proximity_row={str(key): _json_value(value) for key, value in chosen.to_dict().items()},
        fold_entry=fold_entry,
    )


def build_fixed_split(config: TrainingSizeOODConfig, task_id: str) -> FixedTrainingSizeSplit:
    selected = select_farthest_loco_fold(config.proximity_csv, task_id)
    manifest = json.loads(selected.manifest_path.read_text(encoding="utf-8"))
    test_files = selected.fold_entry.get("test_set_files", {})
    if "test_extrapolation_high20" not in test_files or "test_inner_ood" not in test_files:
        raise ValueError("Hybrid fold is missing top-20% or inner OOD test artifacts")
    top20_test = pd.read_csv(_resolve_path(test_files["test_extrapolation_high20"], selected.manifest_path.parent))
    loco_test = pd.read_csv(_resolve_path(test_files["test_inner_ood"], selected.manifest_path.parent))
    candidate_df = pd.read_csv(_resolve_path(selected.fold_entry["train_file"], selected.manifest_path.parent))
    source_validation = _validate_hybrid_source(
        selected=selected,
        manifest=manifest,
        top20_test=top20_test,
        loco_test=loco_test,
        candidate_df=candidate_df,
    )
    processor = StrengthRandomCVBaselineDataProcessor(
        input_file=str(selected.fold_entry["train_file"]),
        random_state=config.random_state,
    )
    random_folds = processor.prepare(
        candidate_df,
        selected.target_col,
        baseline_num_folds=config.random_cv_folds,
    )
    if not random_folds:
        raise ValueError("Random-CV processor returned no folds")
    fold_index = config.random_cv_fold_index
    if fold_index is None:
        fold_index = config.random_state % len(random_folds)
    if not 0 <= int(fold_index) < len(random_folds):
        raise ValueError("random_cv_fold_index is outside the generated folds")
    random_split = random_folds[int(fold_index)].split
    fixed = FixedTrainingSizeSplit(
        selected_loco=selected,
        target_col=selected.target_col,
        top20_test=top20_test.reset_index(drop=True),
        loco_test=loco_test.reset_index(drop=True),
        random_cv_test=random_split.test_df.reset_index(drop=True),
        training_pool=random_split.train_df.reset_index(drop=True),
        random_cv_fold_index=int(fold_index),
        random_state=config.random_state,
        source_validation=source_validation,
    )
    validate_fixed_split(fixed)
    return fixed


def validate_fixed_split(fixed: FixedTrainingSizeSplit) -> None:
    named_frames = {
        TOP20_TEST_NAME: fixed.top20_test,
        LOCO_TEST_NAME: fixed.loco_test,
        "random_cv": fixed.random_cv_test,
        "training_pool": fixed.training_pool,
    }
    for name, frame in named_frames.items():
        if frame.empty:
            raise ValueError(f"Fixed split contains an empty partition: {name}")
    _assert_pairwise_disjoint(named_frames)


def validate_prepared_split(split: PreparedSplit) -> None:
    named_frames = {"train": split.train_df, "random_cv": split.test_df, **split.test_sets}
    _assert_pairwise_disjoint(named_frames)


def iter_training_run_keys(config: TrainingSizeOODConfig) -> Iterable[tuple[float, int]]:
    for fraction in config.training_fractions:
        repeat_count = config.full_fraction_repeats if float(fraction) == 1.0 else config.repeats
        for repeat_id in range(repeat_count):
            yield float(fraction), int(repeat_id)


def build_training_subset_plan(
    fixed: FixedTrainingSizeSplit,
    config: TrainingSizeOODConfig,
) -> list[TrainingSubsetPlan]:
    if "ID" not in fixed.training_pool.columns:
        raise ValueError("training_pool must contain an ID column to freeze subset plans")
    if fixed.training_pool["ID"].duplicated().any():
        raise ValueError("training_pool IDs must be unique")
    plan: list[TrainingSubsetPlan] = []
    for fraction, repeat_id in iter_training_run_keys(config):
        split = fixed.sample_training_split(fraction, repeat_id)
        plan.append(
            TrainingSubsetPlan(
                training_fraction=fraction,
                repeat_id=repeat_id,
                sampling_seed=int(split.summary["sampling_seed"]),
                train_ids=split.train_df["ID"].tolist(),
                subset_hash=str(split.summary["subset_hash"]),
                train_size=int(len(split.train_df)),
            )
        )
    return plan


def materialize_training_split(
    fixed: FixedTrainingSizeSplit,
    plan: TrainingSubsetPlan,
) -> PreparedSplit:
    pool = fixed.training_pool.set_index("ID", drop=False)
    missing = [value for value in plan.train_ids if value not in pool.index]
    if missing:
        raise ValueError(f"Frozen subset IDs are absent from the training pool: {missing[:5]}")
    train_df = pool.loc[plan.train_ids].reset_index(drop=True)
    actual_hash = _frame_identity_hash(train_df, fixed.target_col)
    if actual_hash != plan.subset_hash:
        raise ValueError("Frozen subset hash does not match the materialized training rows")
    split = PreparedSplit(
        split_strategy="training_size_hybrid_ood",
        target_col=fixed.target_col,
        train_df=train_df,
        test_df=fixed.random_cv_test.copy(),
        test_sets={
            TOP20_TEST_NAME: fixed.top20_test.copy(),
            LOCO_TEST_NAME: fixed.loco_test.copy(),
        },
        summary={
            "split_strategy": "training_size_hybrid_ood",
            "training_fraction": float(plan.training_fraction),
            "repeat_id": int(plan.repeat_id),
            "sampling_seed": int(plan.sampling_seed),
            "subset_hash": plan.subset_hash,
            "training_pool_size": int(len(fixed.training_pool)),
            "train_size": int(len(train_df)),
            "random_cv_test_size": int(len(fixed.random_cv_test)),
            "top20_test_size": int(len(fixed.top20_test)),
            "loco_test_size": int(len(fixed.loco_test)),
            "held_out_cluster_id": int(fixed.selected_loco.held_out_cluster_id),
            "selected_loco_fold_index": int(fixed.selected_loco.fold_index),
            "source_hybrid_manifest": str(fixed.selected_loco.manifest_path),
        },
        train_label="train",
        test_label="random_cv",
    )
    validate_prepared_split(split)
    return split


def build_degradation_rows(
    *,
    task_id: str,
    model_family: str,
    model: str,
    training_fraction: float,
    repeat_id: int,
    random_cv_mae: float,
    top20_mae: float,
    loco_mae: float,
) -> list[dict[str, Any]]:
    if random_cv_mae <= 0:
        raise ValueError("random_cv_mae must be positive to calculate degradation ratios")
    common = {
        "task_id": task_id,
        "model_family": model_family,
        "model": model,
        "training_fraction": float(training_fraction),
        "repeat_id": int(repeat_id),
        "random_cv_mae": float(random_cv_mae),
    }
    return [
        {
            **common,
            "failure_axis": FAILURE_AXIS_TOP20,
            "ood_mae": float(top20_mae),
            "degradation_ratio": float(top20_mae / random_cv_mae),
        },
        {
            **common,
            "failure_axis": FAILURE_AXIS_LOCO,
            "ood_mae": float(loco_mae),
            "degradation_ratio": float(loco_mae / random_cv_mae),
        },
    ]


def validate_run_metrics_payload(payload: dict[str, Any]) -> None:
    try:
        random_cv_mae = float(payload["random_cv_mae"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Run metrics are missing a numeric Random-CV MAE") from exc
    if not np.isfinite(random_cv_mae) or random_cv_mae <= 0:
        raise ValueError("Run metrics contain an invalid Random-CV MAE")
    test_sets = payload.get("test_set_metrics")
    if not isinstance(test_sets, dict):
        raise ValueError("Run metrics are missing test_set_metrics")
    for test_name in (TOP20_TEST_NAME, LOCO_TEST_NAME):
        values = test_sets.get(test_name)
        if not isinstance(values, dict) or "mae" not in values:
            raise ValueError(f"Run metrics are missing {test_name} MAE")
        try:
            mae = float(values["mae"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Run metrics contain a non-numeric {test_name} MAE") from exc
        if not np.isfinite(mae) or mae < 0:
            raise ValueError(f"Run metrics contain an invalid {test_name} MAE")


def _resolve_path(value: Any, base_dir: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    repo_candidate = Path.cwd() / path
    if repo_candidate.exists():
        return repo_candidate
    return base_dir / path


def _row_keys(frame: pd.DataFrame) -> list[str]:
    if "ID" in frame.columns:
        return [f"ID:{value}" for value in frame["ID"].astype(str).tolist()]
    records = frame.where(pd.notna(frame), None).to_dict(orient="records")
    return [json.dumps(record, sort_keys=True, default=str, separators=(",", ":")) for record in records]


def _frame_identity_hash(frame: pd.DataFrame, target_col: str) -> str:
    try:
        return sample_identity_hash(frame, target_col)
    except (TypeError, ValueError):
        payload = json.dumps(_row_keys(frame), ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _assert_pairwise_disjoint(named_frames: dict[str, pd.DataFrame]) -> None:
    key_sets = {name: set(_row_keys(frame)) for name, frame in named_frames.items()}
    names = list(key_sets)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            overlap = key_sets[left_name].intersection(key_sets[right_name])
            if overlap:
                raise ValueError(f"Split partitions overlap: {left_name} and {right_name} ({len(overlap)} rows)")


def _validate_hybrid_source(
    *,
    selected: SelectedLocoFold,
    manifest: dict[str, Any],
    top20_test: pd.DataFrame,
    loco_test: pd.DataFrame,
    candidate_df: pd.DataFrame,
) -> dict[str, Any]:
    if str(manifest.get("split_strategy", "")) != "hybrid_extrapolation_loco":
        raise ValueError("Source manifest is not a hybrid_extrapolation_loco split")
    summary = dict(selected.fold_entry.get("split_summary", {}))
    method_params = dict(manifest.get("method_params", {}))
    outer_strategy = str(summary.get("outer_strategy") or method_params.get("outer_strategy") or "")
    inner_strategy = str(summary.get("inner_strategy") or method_params.get("inner_strategy") or "")
    if outer_strategy != "target_extrapolation_high20":
        raise ValueError(f"Unexpected hybrid outer strategy: {outer_strategy}")
    if inner_strategy != "loco":
        raise ValueError(f"Unexpected hybrid inner strategy: {inner_strategy}")
    summary_cluster = summary.get("held_out_cluster_id")
    if summary_cluster is not None and int(summary_cluster) != selected.held_out_cluster_id:
        raise ValueError("Hybrid fold summary cluster ID does not match the selected proximity row")

    identities = dict(selected.fold_entry.get("test_set_identities", {}))
    _validate_manifest_ids(
        top20_test,
        identities.get("test_extrapolation_high20"),
        "test_extrapolation_high20",
    )
    _validate_manifest_ids(
        loco_test,
        identities.get("test_inner_ood"),
        "test_inner_ood",
    )
    _validate_manifest_ids(candidate_df, selected.fold_entry.get("train_identity"), "train")

    _validate_expected_size(selected.proximity_row.get("top_tail_size"), len(top20_test), "top-tail")
    _validate_expected_size(selected.proximity_row.get("cluster_size"), len(loco_test), "LOCO cluster")
    _validate_expected_size(summary.get("outer_test_size"), len(top20_test), "outer test")
    _validate_expected_size(summary.get("inner_test_size"), len(loco_test), "inner LOCO test")
    _validate_expected_size(summary.get("train_size"), len(candidate_df), "candidate train")
    total_size = len(top20_test) + len(loco_test) + len(candidate_df)
    _validate_expected_size(summary.get("total_size"), total_size, "hybrid total")
    return {
        "manifest_split_strategy": "hybrid_extrapolation_loco",
        "outer_strategy": outer_strategy,
        "inner_strategy": inner_strategy,
        "cluster_id_matches_proximity": True,
        "manifest_row_ids_verified": True,
        "loco_from_post_top20_remaining": True,
    }


def _validate_manifest_ids(
    frame: pd.DataFrame,
    identity: Any,
    label: str,
) -> None:
    if "ID" not in frame.columns:
        raise ValueError(f"{label} artifact has no ID column")
    if not isinstance(identity, dict) or "ids" not in identity:
        raise ValueError(f"Hybrid manifest is missing {label} row IDs")
    expected_ids = _parse_manifest_ids(identity["ids"])
    actual_ids = [_normalized_id(value) for value in frame["ID"].tolist()]
    if expected_ids != actual_ids:
        raise ValueError(f"{label} manifest IDs do not match the cached artifact")
    if identity.get("row_count") is not None and int(identity["row_count"]) != len(frame):
        raise ValueError(f"{label} manifest row_count does not match the cached artifact")


def _parse_manifest_ids(value: Any) -> list[str]:
    if isinstance(value, str):
        values = value.replace(",", " ").split()
    elif isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        values = list(value)
    else:
        raise ValueError("Hybrid manifest row IDs must be a list or whitespace-separated string")
    return [_normalized_id(item) for item in values]


def _normalized_id(value: Any) -> str:
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value).strip()


def _validate_expected_size(expected: Any, actual: int, label: str) -> None:
    if expected is None or (isinstance(expected, float) and np.isnan(expected)):
        return
    if int(expected) != int(actual):
        raise ValueError(f"{label} size does not match the proximity/hybrid manifest")


def _sample_seed(base_seed: int, repeat_id: int) -> int:
    return int(np.random.SeedSequence([int(base_seed), int(repeat_id)]).generate_state(1)[0])


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value
