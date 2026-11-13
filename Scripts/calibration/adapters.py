from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
import pandas as pd


if TYPE_CHECKING:
    from src.data_processing.canonical_split_registry import CanonicalSplitEntry
    from src.training_size_ood.model_adapters import (
        StandardFeatureCache,
        TaskRuntimeContext,
    )


MEMBER_COLUMNS = [
    "task_id",
    "protocol",
    "split_id",
    "model_family",
    "model",
    "sample_id",
    "member_id",
    "y_true",
    "y_pred",
    "source_path",
]

SAMPLE_COLUMNS = [
    "task_id",
    "protocol",
    "split_id",
    "model_family",
    "model",
    "sample_id",
    "y_true",
    "y_pred",
    "abs_error",
    "uncertainty_score",
    "confidence_tier",
    "parameter_snapshot_id",
    "parameter_hash",
    "source_path",
    "member_count",
    "uncertainty_method",
]

TABPFN_QUANTILE_COLUMNS = [
    "ID",
    "y_true",
    "y_pred_mean",
    "q10",
    "q50",
    "q90",
    "uncertainty_score",
]

TRADITIONAL_MEMBER_SEEDS = tuple(range(42, 52))
MC_DROPOUT_SEEDS = tuple(range(42, 72))
REQUIRED_MC_PASSES = 30
SOURCE_TRUTH_RTOL = 1e-6
SOURCE_TRUTH_ATOL = 1e-6
PREDICTION_TRUTH_ATOL = 1e-4


@dataclass(frozen=True)
class AdapterResult:
    members: pd.DataFrame
    sample: pd.DataFrame
    quantiles: pd.DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, init=False)
class CalibrationAdapterContext:
    workspace: Path
    output_root: Path
    cache_root: Path
    split_entry: CanonicalSplitEntry
    frozen_entry: dict[str, Any]
    seeds: tuple[int, ...]
    mc_passes: int
    mc_seeds: tuple[int, ...]

    def __init__(
        self,
        workspace: str | Path | None = None,
        output_root: str | Path | None = None,
        cache_root: str | Path | None = None,
        split_entry: CanonicalSplitEntry | None = None,
        frozen_entry: Mapping[str, Any] | None = None,
        seeds: Sequence[int] = TRADITIONAL_MEMBER_SEEDS,
        mc_passes: int = REQUIRED_MC_PASSES,
        mc_seeds: Sequence[int] | None = None,
        *,
        workspace_root: str | Path | None = None,
        output: str | Path | None = None,
        cache: str | Path | None = None,
        split: CanonicalSplitEntry | None = None,
        frozen: Mapping[str, Any] | None = None,
        ensemble_seeds: Sequence[int] | None = None,
    ) -> None:
        resolved_workspace = workspace if workspace is not None else workspace_root
        resolved_output = output_root if output_root is not None else output
        resolved_cache = cache_root if cache_root is not None else cache
        resolved_split = split_entry if split_entry is not None else split
        resolved_frozen = frozen_entry if frozen_entry is not None else frozen
        resolved_seeds = seeds if ensemble_seeds is None else ensemble_seeds
        if resolved_workspace is None:
            raise TypeError("CalibrationAdapterContext requires a workspace path.")
        if resolved_output is None:
            raise TypeError("CalibrationAdapterContext requires an output path.")
        if resolved_cache is None:
            raise TypeError("CalibrationAdapterContext requires a cache path.")
        if resolved_split is None:
            raise TypeError("CalibrationAdapterContext requires a CanonicalSplitEntry.")
        if resolved_frozen is None:
            raise TypeError("CalibrationAdapterContext requires a frozen entry.")
        pass_count = int(mc_passes)
        resolved_mc_seeds = (
            tuple(int(value) for value in mc_seeds)
            if mc_seeds is not None
            else tuple(range(42, 42 + pass_count))
        )
        object.__setattr__(self, "workspace", Path(resolved_workspace).resolve())
        object.__setattr__(self, "output_root", Path(resolved_output).resolve())
        object.__setattr__(self, "cache_root", Path(resolved_cache).resolve())
        object.__setattr__(self, "split_entry", resolved_split)
        object.__setattr__(self, "frozen_entry", dict(resolved_frozen))
        object.__setattr__(self, "seeds", tuple(int(value) for value in resolved_seeds))
        object.__setattr__(self, "mc_passes", pass_count)
        object.__setattr__(self, "mc_seeds", resolved_mc_seeds)

    @property
    def workspace_root(self) -> Path:
        return self.workspace

    @property
    def split(self) -> CanonicalSplitEntry:
        return self.split_entry

    @property
    def frozen(self) -> dict[str, Any]:
        return self.frozen_entry

    @property
    def ensemble_seeds(self) -> tuple[int, ...]:
        return self.seeds


@dataclass(frozen=True)
class UnitIdentity:
    task_id: str
    protocol: str
    split_id: str
    model_family: str
    model: str
    parameter_snapshot_id: str
    parameter_hash: str

    def metadata(self) -> dict[str, str]:
        return {
            "task_id": self.task_id,
            "protocol": self.protocol,
            "split_id": self.split_id,
            "model_family": self.model_family,
            "model": self.model,
        }


def _with_identity(frame: pd.DataFrame, identity: UnitIdentity) -> pd.DataFrame:
    output = frame.copy()
    for name, value in identity.metadata().items():
        output[name] = value
    return output


def aggregate_member_predictions(
    members: pd.DataFrame,
    *,
    identity: UnitIdentity,
    uncertainty_method: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"sample_id", "member_id", "y_true", "y_pred", "source_path"}
    missing = sorted(required.difference(members.columns))
    if missing:
        raise ValueError(f"Member predictions are missing columns: {missing}")

    normalized = _with_identity(members, identity)
    normalized["y_true"] = pd.to_numeric(normalized["y_true"], errors="raise")
    normalized["y_pred"] = pd.to_numeric(normalized["y_pred"], errors="raise")
    if not np.isfinite(normalized[["y_true", "y_pred"]].to_numpy(dtype=float)).all():
        raise ValueError("Member predictions contain NaN/Inf values.")

    rows: list[dict[str, Any]] = []
    for sample_id, group in normalized.groupby("sample_id", sort=False, dropna=False):
        truth = group["y_true"].to_numpy(dtype=float)
        predictions = group["y_pred"].to_numpy(dtype=float)
        if len(predictions) < 2:
            raise ValueError("At least two member predictions are required for sample SD.")
        if not np.allclose(truth, truth[0], rtol=1e-10, atol=1e-12):
            raise ValueError("Ensemble member y_true values do not match.")
        rows.append(
            {
                **identity.metadata(),
                "sample_id": sample_id,
                "y_true": float(truth[0]),
                "y_pred": float(np.mean(predictions)),
                "abs_error": float(abs(np.mean(predictions) - truth[0])),
                "uncertainty_score": float(np.std(predictions, ddof=1)),
                "confidence_tier": pd.NA,
                "parameter_snapshot_id": identity.parameter_snapshot_id,
                "parameter_hash": identity.parameter_hash,
                "source_path": ";".join(
                    dict.fromkeys(group["source_path"].astype(str).tolist())
                ),
                "member_count": int(len(predictions)),
                "uncertainty_method": uncertainty_method,
            }
        )
    return normalized[MEMBER_COLUMNS], pd.DataFrame(rows, columns=SAMPLE_COLUMNS)


def load_tabpfn_quantiles(
    source_path: str | Path,
    *,
    canonical_test: pd.DataFrame,
    target_column: str,
    identity: UnitIdentity,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = Path(source_path)
    quantiles = pd.read_csv(source)
    if list(quantiles.columns) != TABPFN_QUANTILE_COLUMNS:
        raise ValueError(
            "TabPFN predictive quantiles have an unexpected schema: "
            f"{quantiles.columns.tolist()}"
        )
    if "ID" not in canonical_test.columns or target_column not in canonical_test.columns:
        raise ValueError("Canonical test is missing ID or target column.")

    source_ids = quantiles["ID"].astype(str).tolist()
    canonical_ids = canonical_test["ID"].astype(str).tolist()
    if source_ids != canonical_ids:
        raise ValueError("TabPFN predictive quantile ID order differs from canonical test.")
    source_true = pd.to_numeric(quantiles["y_true"], errors="raise").to_numpy(float)
    canonical_true = pd.to_numeric(
        canonical_test[target_column],
        errors="raise",
    ).to_numpy(float)
    if not np.allclose(
        source_true,
        canonical_true,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("TabPFN predictive quantile y_true differs from canonical test.")

    numeric = quantiles[TABPFN_QUANTILE_COLUMNS[1:]].apply(
        pd.to_numeric,
        errors="raise",
    )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("TabPFN predictive quantiles contain NaN/Inf values.")
    if np.any(numeric["q10"] > numeric["q50"]) or np.any(
        numeric["q50"] > numeric["q90"]
    ):
        raise ValueError("TabPFN predictive quantiles are not ordered q10 <= q50 <= q90.")
    expected_uncertainty = numeric["q90"] - numeric["q10"]
    if not np.allclose(
        numeric["uncertainty_score"],
        expected_uncertainty,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("TabPFN uncertainty_score must equal q90 - q10.")

    members = pd.DataFrame(
        {
            "sample_id": quantiles["ID"],
            "member_id": "predictive_mean",
            "y_true": source_true,
            "y_pred": numeric["y_pred_mean"].to_numpy(float),
            "source_path": str(source.resolve()),
        }
    )
    members = _with_identity(members, identity)[MEMBER_COLUMNS]
    sample = pd.DataFrame(
        {
            **identity.metadata(),
            "sample_id": quantiles["ID"],
            "y_true": source_true,
            "y_pred": numeric["y_pred_mean"].to_numpy(float),
            "abs_error": np.abs(numeric["y_pred_mean"].to_numpy(float) - source_true),
            "uncertainty_score": numeric["uncertainty_score"].to_numpy(float),
            "confidence_tier": pd.NA,
            "parameter_snapshot_id": identity.parameter_snapshot_id,
            "parameter_hash": identity.parameter_hash,
            "source_path": str(source.resolve()),
            "member_count": 1,
            "uncertainty_method": "predictive_q90_minus_q10",
        }
    )
    return quantiles[TABPFN_QUANTILE_COLUMNS], members, sample[SAMPLE_COLUMNS]


def run_adapter(
    context: CalibrationAdapterContext,
    attempt_dir: str | Path,
) -> AdapterResult:
    family = str(context.frozen_entry.get("model_family", "")).strip()
    if family == "Traditional":
        return _run_traditional_adapter(context, Path(attempt_dir))
    if family == "BERT":
        return _run_bert_adapter(context, Path(attempt_dir))
    if family == "LLMProp":
        return _run_llmprop_adapter(context, Path(attempt_dir))
    if family == "TabPFN":
        return _run_tabpfn_adapter(context)
    raise ValueError(f"Unsupported calibration model family: {family}")


def _run_traditional_adapter(
    context: CalibrationAdapterContext,
    attempt_dir: Path,
) -> AdapterResult:
    seeds = tuple(context.seeds)
    if seeds != TRADITIONAL_MEMBER_SEEDS:
        raise ValueError(
            "Traditional calibration requires exactly seeds 42..51 in order."
        )
    identity = _identity_from_context(context)
    model = identity.model
    effective_parameters = _effective_parameters(context)
    prepared_split = context.split_entry.load_prepared_split()
    canonical_test = context.split_entry.load_test()
    attempt_dir.mkdir(parents=True, exist_ok=True)

    adapter = _new_standard_adapter("Traditional")
    runtime_context = _build_task_runtime_context(context, model_seed=seeds[0])
    feature_cache = _prepare_standard_feature_cache(
        adapter=adapter,
        context=context,
        runtime_context=runtime_context,
        model=model,
        effective_parameters=effective_parameters,
        prepared_split=prepared_split,
    )

    member_frames: list[pd.DataFrame] = []
    metric_paths: list[str] = []
    for member_number, seed in enumerate(seeds):
        result_dir = attempt_dir / "members" / f"seed_{seed}"
        seeded_parameters = _traditional_parameters_for_seed(
            effective_parameters,
            model=model,
            seed=seed,
        )
        metrics = adapter.run(
            context=replace(runtime_context, model_seed=seed),
            model=model,
            prepared_split=prepared_split,
            effective_parameters=seeded_parameters,
            result_dir=result_dir,
            manifest_metadata=_manifest_metadata(context, member_seed=seed),
            feature_cache=feature_cache,
        )
        metric_path = str(getattr(metrics, "metrics_file", "") or "")
        if metric_path:
            metric_paths.append(metric_path)
        member = _extract_canonical_ood_predictions(
            result_dir,
            canonical_test=canonical_test,
            target_column=context.split_entry.target_column,
        )
        member["member_id"] = f"seed_{seed}"
        member_frames.append(
            member[["sample_id", "member_id", "y_true", "y_pred", "source_path"]]
        )

    raw_members = pd.concat(member_frames, ignore_index=True)
    _validate_generated_members(
        raw_members,
        canonical_test=canonical_test,
        target_column=context.split_entry.target_column,
        expected_member_count=len(seeds),
    )
    members, sample = aggregate_member_predictions(
        raw_members,
        identity=identity,
        uncertainty_method="ensemble_sd_ddof1_10_members",
    )
    return AdapterResult(
        members=members,
        sample=sample,
        metadata={
            **identity.metadata(),
            "member_count": len(seeds),
            "member_seeds": list(seeds),
            "training_runs": len(seeds),
            "uncertainty_method": "ensemble_sd_ddof1_10_members",
            "feature_cache_manifest": _cache_manifest_path(feature_cache),
            "metrics_files": metric_paths,
        },
    )


def _run_bert_adapter(
    context: CalibrationAdapterContext,
    attempt_dir: Path,
) -> AdapterResult:
    seeds = _required_mc_seeds(context, family="BERT")
    identity = _identity_from_context(context)
    effective_parameters = _effective_parameters(context)
    prepared_split = context.split_entry.load_prepared_split()
    canonical_test = context.split_entry.load_test()
    result_dir = attempt_dir / "training"
    result_dir.mkdir(parents=True, exist_ok=True)

    adapter = _new_standard_adapter("BERT")
    runtime_context = _build_task_runtime_context(
        context,
        model_seed=_frozen_model_seed(effective_parameters),
    )
    feature_cache = _prepare_standard_feature_cache(
        adapter=adapter,
        context=context,
        runtime_context=runtime_context,
        model=identity.model,
        effective_parameters=effective_parameters,
        prepared_split=prepared_split,
    )
    metrics = adapter.run(
        context=runtime_context,
        model=identity.model,
        prepared_split=prepared_split,
        effective_parameters=effective_parameters,
        result_dir=result_dir,
        manifest_metadata=_manifest_metadata(context),
        feature_cache=feature_cache,
    )
    deterministic = _extract_canonical_ood_predictions(
        result_dir,
        canonical_test=canonical_test,
        target_column=context.split_entry.target_column,
    )
    checkpoint_path = _find_single_checkpoint(result_dir)
    checkpoint_sha256 = _file_sha256(checkpoint_path)
    raw_members = _run_bert_mc_dropout(
        adapter=adapter,
        runtime_context=runtime_context,
        model=identity.model,
        effective_parameters=effective_parameters,
        feature_cache=feature_cache,
        checkpoint_path=checkpoint_path,
        canonical_test=canonical_test,
        target_column=context.split_entry.target_column,
        seeds=seeds,
    )
    _validate_generated_members(
        raw_members,
        canonical_test=canonical_test,
        target_column=context.split_entry.target_column,
        expected_member_count=len(seeds),
    )
    raw_members["source_path"] = (
        f"{checkpoint_path.resolve()}#sha256={checkpoint_sha256}"
    )
    members, sample = aggregate_member_predictions(
        raw_members,
        identity=identity,
        uncertainty_method="mc_dropout_sd_ddof1_30_passes",
    )
    checkpoint_path.unlink()
    return AdapterResult(
        members=members,
        sample=sample,
        metadata={
            **identity.metadata(),
            "member_count": len(seeds),
            "member_seeds": list(seeds),
            "training_runs": 1,
            "uncertainty_method": "mc_dropout_sd_ddof1_30_passes",
            "feature_cache_manifest": _cache_manifest_path(feature_cache),
            "deterministic_prediction_source": _single_source_path(deterministic),
            "metrics_file": str(getattr(metrics, "metrics_file", "") or ""),
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_deleted": True,
        },
    )


def _run_llmprop_adapter(
    context: CalibrationAdapterContext,
    attempt_dir: Path,
) -> AdapterResult:
    seeds = _required_mc_seeds(context, family="LLMProp")
    identity = _identity_from_context(context)
    effective_parameters = _effective_parameters(context)
    prepared_split = context.split_entry.load_prepared_split()
    canonical_test = context.split_entry.load_test()
    result_dir = attempt_dir / "training"
    result_dir.mkdir(parents=True, exist_ok=True)

    adapter = _new_llmprop_adapter()
    runtime_context = _build_task_runtime_context(
        context,
        model_seed=_frozen_model_seed(effective_parameters),
    )
    metrics = adapter.run(
        context=runtime_context,
        model=identity.model,
        prepared_split=prepared_split,
        effective_parameters=effective_parameters,
        result_dir=result_dir,
        manifest_metadata=_manifest_metadata(context),
        feature_cache=None,
    )
    deterministic = _extract_canonical_ood_predictions(
        result_dir,
        canonical_test=canonical_test,
        target_column=context.split_entry.target_column,
    )
    checkpoint_path = _find_single_checkpoint(result_dir)
    checkpoint_sha256 = _file_sha256(checkpoint_path)
    raw_members = _run_llmprop_mc_dropout(
        result_dir=result_dir,
        checkpoint_path=checkpoint_path,
        effective_parameters=effective_parameters,
        canonical_test=canonical_test,
        target_column=context.split_entry.target_column,
        seeds=seeds,
    )
    _validate_generated_members(
        raw_members,
        canonical_test=canonical_test,
        target_column=context.split_entry.target_column,
        expected_member_count=len(seeds),
    )
    raw_members["source_path"] = (
        f"{checkpoint_path.resolve()}#sha256={checkpoint_sha256}"
    )
    members, sample = aggregate_member_predictions(
        raw_members,
        identity=identity,
        uncertainty_method="mc_dropout_sd_ddof1_30_passes",
    )
    checkpoint_path.unlink()
    return AdapterResult(
        members=members,
        sample=sample,
        metadata={
            **identity.metadata(),
            "member_count": len(seeds),
            "member_seeds": list(seeds),
            "training_runs": 1,
            "uncertainty_method": "mc_dropout_sd_ddof1_30_passes",
            "deterministic_prediction_source": _single_source_path(deterministic),
            "metrics_file": str(getattr(metrics, "metrics_file", "") or ""),
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_deleted": True,
        },
    )


def _run_tabpfn_adapter(context: CalibrationAdapterContext) -> AdapterResult:
    identity = _identity_from_context(context)
    source_path = _locate_tabpfn_quantiles(context)
    quantiles, members, sample = load_tabpfn_quantiles(
        source_path,
        canonical_test=context.split_entry.load_test(),
        target_column=context.split_entry.target_column,
        identity=identity,
    )
    runtime = _tabpfn_runtime_metadata(source_path)
    return AdapterResult(
        members=members,
        sample=sample,
        quantiles=quantiles,
        metadata={
            **identity.metadata(),
            "member_count": 1,
            "training_runs": 0,
            "uncertainty_method": "predictive_q90_minus_q10",
            "source_path": str(source_path.resolve()),
            **runtime,
        },
    )


def _new_standard_adapter(model_family: str) -> Any:
    from src.training_size_ood.model_adapters import StandardPipelineAdapter

    return StandardPipelineAdapter(model_family)


def _new_llmprop_adapter() -> Any:
    from src.training_size_ood.model_adapters import LLMPropAdapter

    return LLMPropAdapter()


def _build_task_runtime_context(
    context: CalibrationAdapterContext,
    *,
    model_seed: int,
) -> TaskRuntimeContext:
    from src.pipelines.batch_configs_ood import ALLOY_CONFIGS_OOD
    from src.training_size_ood.model_adapters import TaskRuntimeContext

    split = context.split_entry
    alloy_key = _alloy_config_key(split.alloy_type, ALLOY_CONFIGS_OOD)
    alloy_config = ALLOY_CONFIGS_OOD[alloy_key]
    raw_data = Path(str(alloy_config["raw_data"]))
    data_file = raw_data if raw_data.is_absolute() else context.workspace / raw_data
    return TaskRuntimeContext(
        task_id=split.task_id,
        frozen_task_id=str(context.frozen_entry.get("task_id", split.task_id)),
        alloy_type=alloy_key,
        dataset_name=split.dataset_name,
        target_col=split.target_column,
        data_file=data_file.resolve(),
        processing_cols=tuple(
            str(value) for value in alloy_config.get("processing_cols", [])
        ),
        processing_text_column=alloy_config.get("processing_text_column"),
        model_seed=int(model_seed),
        cache_root=context.cache_root,
    )


def _alloy_config_key(
    alloy_type: object,
    configs: Mapping[str, Any],
) -> str:
    raw = str(alloy_type).strip()
    normalized = re.sub(r"[^a-z0-9]+", "", raw.lower())
    if normalized in {"matbenchsteel", "matbenchsteels"}:
        candidate = "MatbenchSteels"
    else:
        matches = [
            key
            for key in configs
            if re.sub(r"[^a-z0-9]+", "", str(key).lower()) == normalized
        ]
        candidate = matches[0] if len(matches) == 1 else raw
    if candidate not in configs:
        raise ValueError(f"No ALLOY_CONFIGS_OOD entry for alloy type: {alloy_type}")
    return candidate


def _prepare_standard_feature_cache(
    *,
    adapter: Any,
    context: CalibrationAdapterContext,
    runtime_context: TaskRuntimeContext,
    model: str,
    effective_parameters: dict[str, Any],
    prepared_split: Any,
) -> StandardFeatureCache:
    from src.pipelines.ood_pipeline import prepare_feature_files
    from src.training_size_ood.model_adapters import StandardFeatureCache

    family = str(context.frozen_entry["model_family"])
    args = adapter.build_args(
        context=runtime_context,
        model=model,
        effective_parameters=effective_parameters,
        result_dir=context.cache_root,
    )
    feature_config = {
        "model_family": family,
        "embedding_type": str(args.embedding_type),
        "use_composition_feature": bool(args.use_composition_feature),
        "use_element_embedding": bool(args.use_element_embedding),
        "use_process_embedding": bool(args.use_process_embedding),
        "use_temperature": bool(args.use_temperature),
        "processing_cols": list(args.processing_cols or []),
        "processing_text_column": args.processing_text_column,
    }
    feature_hash = hashlib.sha256(
        json.dumps(
            feature_config,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    cache_dir = (
        context.cache_root
        / "features"
        / _safe_component(context.split_entry.task_id)
        / _safe_component(context.split_entry.protocol)
        / _safe_component(context.split_entry.split_id)
        / feature_hash
    )
    manifest_path = cache_dir / "cache_manifest.json"
    identity = {
        "task_id": context.split_entry.task_id,
        "protocol": context.split_entry.protocol,
        "split_id": context.split_entry.split_id,
        "train_file_sha256": _file_sha256(context.split_entry.train_file),
        "test_file_sha256": _file_sha256(context.split_entry.test_file),
        "feature_hash": feature_hash,
        "feature_config": feature_config,
    }
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            files = dict(payload.get("feature_files", {}))
            candidate = _standard_cache_from_files(manifest_path, files)
            if payload.get("identity") == identity and _standard_cache_files_exist(candidate):
                return candidate
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            pass

    generated = prepare_feature_files(args, prepared_split, cache_dir)
    cache = _standard_cache_from_files(manifest_path, generated)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "identity": identity,
                "feature_files": {
                    "train": str(cache.full_train_file),
                    "test": str(cache.test_file),
                    "test_sets": {
                        name: str(path)
                        for name, path in cache.test_set_files.items()
                    },
                },
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    return cache


def _standard_cache_from_files(
    manifest_path: Path,
    files: Mapping[str, Any],
) -> StandardFeatureCache:
    from src.training_size_ood.model_adapters import StandardFeatureCache

    return StandardFeatureCache(
        full_train_file=Path(str(files["train"])).resolve(),
        test_file=Path(str(files["test"])).resolve(),
        test_set_files={
            str(name): Path(str(path)).resolve()
            for name, path in dict(files.get("test_sets", {})).items()
        },
        cache_manifest=manifest_path.resolve(),
    )


def _standard_cache_files_exist(cache: StandardFeatureCache) -> bool:
    return all(
        path.exists()
        for path in [
            cache.full_train_file,
            cache.test_file,
            *cache.test_set_files.values(),
        ]
    )


def _extract_canonical_ood_predictions(
    result_dir: str | Path,
    *,
    canonical_test: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    root = Path(result_dir)
    candidates = sorted(root.rglob("*all_predictions.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No all_predictions.csv source found below adapter result: {root}"
        )
    matches: list[pd.DataFrame] = []
    errors: list[Exception] = []
    for path in candidates:
        try:
            matches.append(
                _canonical_prediction_rows(
                    path,
                    canonical_test=canonical_test,
                    target_column=target_column,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(exc)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Multiple canonical OODTest prediction sources found below {root}: "
            f"{len(matches)}"
        )
    if len(errors) == 1:
        raise errors[0]
    messages = "; ".join(str(error) for error in errors[:3])
    raise ValueError(
        f"No prediction source below {root} matches the canonical OOD test. {messages}"
    )


def _canonical_prediction_rows(
    source_path: str | Path,
    *,
    canonical_test: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    source = Path(source_path)
    frame = pd.read_csv(source)
    if "ID" not in frame.columns:
        raise ValueError(f"Prediction source is missing ID: {source}")
    if "Dataset" in frame.columns:
        label_column = "Dataset"
    elif "set" in frame.columns:
        label_column = "set"
    else:
        raise ValueError(f"Prediction source is missing an OODTest label column: {source}")
    labels = frame[label_column].map(_normalized_dataset_label)
    ood = frame.loc[labels.eq("oodtest")].reset_index(drop=True)
    if ood.empty:
        raise ValueError(f"Prediction source has no canonical OODTest rows: {source}")

    sanitized_target = _sanitize_target_name(target_column)
    column_pairs = (
        (f"{target_column}_Actual", f"{target_column}_Predicted"),
        (f"True_{sanitized_target}", f"Pred_{sanitized_target}"),
        ("y_true", "y_pred"),
    )
    actual_column = predicted_column = ""
    for actual, predicted in column_pairs:
        if actual in ood.columns and predicted in ood.columns:
            actual_column, predicted_column = actual, predicted
            break
    if not actual_column:
        raise ValueError(
            f"Prediction source has no target prediction columns for {target_column}: {source}"
        )

    source_ids = ood["ID"].astype(str).tolist()
    canonical_ids, canonical_true = _canonical_arrays(canonical_test, target_column)
    if source_ids != [str(value) for value in canonical_ids]:
        raise ValueError("Source prediction ID order differs from canonical test.")
    source_true = pd.to_numeric(ood[actual_column], errors="raise").to_numpy(float)
    source_pred = pd.to_numeric(ood[predicted_column], errors="raise").to_numpy(float)
    if not np.isfinite(np.column_stack([source_true, source_pred])).all():
        raise ValueError("Source predictions contain NaN/Inf values.")
    if not np.allclose(
        source_true,
        canonical_true,
        rtol=SOURCE_TRUTH_RTOL,
        atol=PREDICTION_TRUTH_ATOL,
    ):
        raise ValueError("Source prediction y_true differs from canonical test.")
    return pd.DataFrame(
        {
            "sample_id": canonical_ids,
            "y_true": canonical_true,
            "y_pred": source_pred,
            "source_path": str(source.resolve()),
        }
    )


def _validate_generated_members(
    members: pd.DataFrame,
    *,
    canonical_test: pd.DataFrame,
    target_column: str,
    expected_member_count: int,
) -> None:
    required = {"sample_id", "member_id", "y_true", "y_pred", "source_path"}
    missing = sorted(required.difference(members.columns))
    if missing:
        raise ValueError(f"Generated member predictions are missing columns: {missing}")
    member_ids = members["member_id"].drop_duplicates().tolist()
    if len(member_ids) != int(expected_member_count):
        raise ValueError(
            "Generated member prediction count differs from the required count: "
            f"expected={expected_member_count}, found={len(member_ids)}"
        )
    canonical_ids, canonical_true = _canonical_arrays(canonical_test, target_column)
    canonical_id_strings = [str(value) for value in canonical_ids]
    for member_id, group in members.groupby("member_id", sort=False, dropna=False):
        if group["sample_id"].astype(str).tolist() != canonical_id_strings:
            raise ValueError(
                f"Generated member ID order differs from canonical test: {member_id}"
            )
        source_true = pd.to_numeric(group["y_true"], errors="raise").to_numpy(float)
        source_pred = pd.to_numeric(group["y_pred"], errors="raise").to_numpy(float)
        if not np.isfinite(np.column_stack([source_true, source_pred])).all():
            raise ValueError("Generated member predictions contain NaN/Inf values.")
        if not np.allclose(
            source_true,
            canonical_true,
            rtol=SOURCE_TRUTH_RTOL,
            atol=SOURCE_TRUTH_ATOL,
        ):
            raise ValueError(
                f"Generated member y_true differs from canonical test: {member_id}"
            )


def _run_bert_mc_dropout(
    *,
    adapter: Any,
    runtime_context: TaskRuntimeContext,
    model: str,
    effective_parameters: dict[str, Any],
    feature_cache: StandardFeatureCache,
    checkpoint_path: Path,
    canonical_test: pd.DataFrame,
    target_column: str,
    seeds: Sequence[int],
) -> pd.DataFrame:
    import joblib
    import torch

    from src.models.base.alloys_nn import AlloyNN
    from src.pipelines.ood_train_eval import (
        apply_frozen_nn_parameters,
        build_nn_args,
        load_ood_feature_split,
    )

    base_args = adapter.build_args(
        context=runtime_context,
        model=model,
        effective_parameters=effective_parameters,
        result_dir=checkpoint_path.parents[1],
    )
    nn_args = build_nn_args(base_args)
    apply_frozen_nn_parameters(nn_args, effective_parameters)
    _, raw_test, feature_names = load_ood_feature_split(
        train_feature_file=str(feature_cache.full_train_file),
        test_feature_file=str(feature_cache.test_file),
        target_columns=[target_column],
        args=nn_args,
        pipeline_kind="nn",
    )
    _validate_loaded_test_data(
        raw_test,
        canonical_test=canonical_test,
        target_column=target_column,
        source_name="BERT cached features",
    )

    scaler_x = joblib.load(checkpoint_path.parents[1] / "scaler_X.pkl")
    scaler_y = joblib.load(checkpoint_path.parents[1] / "scaler_y.pkl")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=nn_args.device or "cpu",
        weights_only=False,
    )
    model_config = checkpoint.get("config")
    if not isinstance(model_config, dict):
        raise ValueError("BERT checkpoint is missing its model config.")
    frozen_config = effective_parameters.get("model_config")
    if isinstance(frozen_config, dict) and model_config != frozen_config:
        raise ValueError("BERT checkpoint model config differs from frozen parameters.")
    inference_model = AlloyNN(
        column_names=feature_names,
        output_dim=1,
        feature_type_config=model_config["feature_type_config"],
        feature_hidden_dims=model_config["feature_hidden_dims"],
        hidden_dims=model_config["hidden_dims"],
        dropout_rate=model_config["dropout_rate"],
    )
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("BERT checkpoint is missing model_state_dict.")
    normalized_state = OrderedDict(
        (
            str(name)[7:] if str(name).startswith("module.") else str(name),
            value,
        )
        for name, value in state_dict.items()
    )
    inference_model.load_state_dict(normalized_state)
    device = torch.device(nn_args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    inference_model.to(device)
    x_test = scaler_x.transform(raw_test["X"]).astype("float32")
    x_tensor = torch.from_numpy(x_test).to(device)
    canonical_ids, canonical_true = _canonical_arrays(canonical_test, target_column)
    frames: list[pd.DataFrame] = []
    with torch.no_grad():
        for member_number, seed in enumerate(seeds):
            _seed_torch(torch, int(seed))
            _enable_mc_dropout(inference_model)
            scaled = inference_model(x_tensor).detach().cpu().numpy()
            if scaled.ndim == 1:
                scaled = scaled.reshape(-1, 1)
            predictions = scaler_y.inverse_transform(scaled).reshape(-1)
            frames.append(
                pd.DataFrame(
                    {
                        "sample_id": canonical_ids,
                        "member_id": f"mc_{member_number:02d}_seed_{seed}",
                        "y_true": canonical_true,
                        "y_pred": predictions,
                        "source_path": str(checkpoint_path.resolve()),
                    }
                )
            )
    del inference_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.concat(frames, ignore_index=True)


def _run_llmprop_mc_dropout(
    *,
    result_dir: Path,
    checkpoint_path: Path,
    effective_parameters: dict[str, Any],
    canonical_test: pd.DataFrame,
    target_column: str,
    seeds: Sequence[int],
) -> pd.DataFrame:
    import torch

    from src.LLMProp.dataset import create_dataloader
    from src.LLMProp.trainer import (
        LLMPropTrainingConfig,
        _load_model,
        _load_tokenizer,
    )

    manifest_path = result_dir / "llmprop_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"LLMProp manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_csv = _resolve_runtime_path(manifest.get("train_csv"), result_dir)
    test_csv = _resolve_runtime_path(manifest.get("test_csv"), result_dir)
    normalizer_value = effective_parameters.get("normalizer", "z_norm")
    if isinstance(normalizer_value, dict):
        normalizer_type = str(normalizer_value.get("type", "z_norm"))
    else:
        normalizer_type = str(normalizer_value)
    config = LLMPropTrainingConfig(
        result_dir=str(result_dir),
        train_csv=str(train_csv),
        test_csv=str(test_csv),
        target_column=target_column,
        base_model_path=str(effective_parameters["base_model_path"]),
        tokenizer_path=str(effective_parameters["tokenizer_path"]),
        epochs=int(effective_parameters.get("epochs", 50)),
        batch_size=int(effective_parameters["batch_size"]),
        learning_rate=float(effective_parameters["learning_rate"]),
        max_len=int(effective_parameters["max_len"]),
        dropout=float(effective_parameters["dropout"]),
        pooling=str(effective_parameters["pooling"]),
        valid_ratio=float(effective_parameters.get("valid_ratio", 0.2)),
        random_state=int(effective_parameters.get("random_state", 42)),
        device=effective_parameters.get("device"),
        normalizer=normalizer_type,
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=config.device or "cpu",
        weights_only=False,
    )
    _validate_llmprop_checkpoint(effective_parameters, checkpoint)
    test_frame = pd.read_csv(test_csv)
    _validate_canonical_frame(
        test_frame,
        canonical_test=canonical_test,
        target_column=target_column,
        source_name="LLMProp converted test data",
    )

    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer, _, _ = _load_tokenizer(config)
    inference_model = _load_model(config, len(tokenizer)).to(device)
    inference_model.load_state_dict(checkpoint["model_state_dict"])
    dataloader = create_dataloader(
        tokenizer,
        test_frame,
        config.max_len,
        config.batch_size,
        target_column,
        pooling=config.pooling,
        shuffle=False,
    )
    labels_mean, labels_std = _llmprop_checkpoint_normalizer(
        effective_parameters,
        checkpoint,
    )
    safe_std = labels_std if abs(labels_std) > 1e-12 else 1.0
    canonical_ids, canonical_true = _canonical_arrays(canonical_test, target_column)
    frames: list[pd.DataFrame] = []
    with torch.no_grad():
        for member_number, seed in enumerate(seeds):
            _seed_torch(torch, int(seed))
            _enable_mc_dropout(inference_model)
            predictions: list[float] = []
            for batch in dataloader:
                input_ids, attention_masks = (
                    tensor.to(device) for tensor in batch[:2]
                )
                _, outputs = inference_model(input_ids, attention_masks)
                denormalized = (
                    outputs.detach().cpu().numpy().reshape(-1) * safe_std
                    + labels_mean
                )
                predictions.extend(denormalized.tolist())
            frames.append(
                pd.DataFrame(
                    {
                        "sample_id": canonical_ids,
                        "member_id": f"mc_{member_number:02d}_seed_{seed}",
                        "y_true": canonical_true,
                        "y_pred": np.asarray(predictions, dtype=float),
                        "source_path": str(checkpoint_path.resolve()),
                    }
                )
            )
    del inference_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.concat(frames, ignore_index=True)


def _validate_llmprop_checkpoint(
    effective_parameters: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> None:
    checkpoint_config = checkpoint.get("config", {})
    if not isinstance(checkpoint_config, Mapping):
        raise ValueError("LLMProp checkpoint is missing its config.")
    for key in (
        "target_column",
        "base_model_path",
        "tokenizer_path",
        "batch_size",
        "learning_rate",
        "max_len",
        "dropout",
        "pooling",
    ):
        if (
            key in effective_parameters
            and key in checkpoint_config
            and not _values_match(effective_parameters[key], checkpoint_config[key])
        ):
            raise ValueError(
                f"LLMProp checkpoint config differs from frozen parameters: {key}"
            )
    if "model_state_dict" not in checkpoint:
        raise ValueError("LLMProp checkpoint is missing model_state_dict.")


def _llmprop_checkpoint_normalizer(
    effective_parameters: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> tuple[float, float]:
    del effective_parameters
    if "labels_mean" not in checkpoint or "labels_std" not in checkpoint:
        raise ValueError("LLMProp checkpoint is missing retrained label normalization.")
    labels_mean = float(checkpoint["labels_mean"])
    labels_std = float(checkpoint["labels_std"])
    if not np.isfinite([labels_mean, labels_std]).all():
        raise ValueError("LLMProp checkpoint label normalization contains NaN/Inf.")
    return labels_mean, labels_std


def tabpfn_source_artifacts(
    *,
    workspace_root: str | Path,
    model: str,
    alloy_type: str,
    dataset_name: str,
    target_column: str,
    canonical_protocol: str,
    split_id: str,
) -> tuple[Path, Path]:
    allowed_models = {
        "TabPFN-2.5-Plus-Numeric",
        "TabPFN-2.5-Plus-Text",
    }
    if model not in allowed_models:
        raise ValueError(f"Unsupported full-calibration TabPFN model: {model}")
    workspace = Path(workspace_root).resolve()
    root = (workspace / "output" / f"ood_results_{model}").resolve()
    alloy_dir = (
        "MatbenchSteels"
        if re.sub(r"[^a-z0-9]+", "", str(alloy_type).lower())
        in {"matbenchsteel", "matbenchsteels"}
        else str(alloy_type)
    )
    protocol_dir = {
        "target_extrapolation": "target_extrapolation",
        "random_cv_baseline": "random_cv_baseline",
        "loco": "loco_k5",
        "sparse_x_single": "sparse_x_single_k5",
        "sparse_x_cluster": "sparse_x_cluster_k5",
        "sparse_y_single": "sparse_y_single_k5",
        "sparse_y_cluster": "sparse_y_cluster_k5",
    }.get(str(canonical_protocol))
    if protocol_dir is None:
        raise ValueError(f"Unsupported canonical TabPFN protocol: {canonical_protocol}")
    result_dir = (
        root
        / protocol_dir
        / str(alloy_dir)
        / str(dataset_name)
        / str(target_column)
    )
    if split_id == "main":
        run_root = result_dir
    elif re.fullmatch(r"fold_\d+", str(split_id)):
        run_root = (
            result_dir
            / "folds"
            / str(split_id)
        )
    else:
        raise ValueError(f"Unsupported canonical TabPFN split ID: {split_id}")
    run_root = run_root.resolve()
    source_path = (run_root / "predictions" / "predictive_quantiles.csv").resolve()
    manifest_path = (run_root / "pipeline_manifest.json").resolve()
    for candidate in (source_path, manifest_path):
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError("TabPFN result resolved outside its approved output root.") from exc
    return source_path, manifest_path


def _locate_tabpfn_quantiles(context: CalibrationAdapterContext) -> Path:
    split = context.split_entry
    source_path, _ = tabpfn_source_artifacts(
        workspace_root=context.workspace,
        model=str(context.frozen_entry.get("model", "")),
        alloy_type=split.alloy_type,
        dataset_name=split.dataset_name,
        target_column=split.target_column,
        canonical_protocol=split.protocol,
        split_id=split.split_id,
    )
    if not source_path.is_file():
        raise FileNotFoundError(
            "Existing TabPFN predictive quantiles were not found for canonical "
            f"task/protocol/split: {split.task_id}/{split.protocol}/{split.split_id}: "
            f"{source_path}"
        )
    return source_path


def _tabpfn_runtime_metadata(source_path: Path) -> dict[str, str]:
    manifest_path = source_path.parent.parent / "pipeline_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"TabPFN pipeline manifest is missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_info = dict(payload.get("model_info", {}) or {})
    requested_backend = str(
        payload.get("requested_backend") or model_info.get("requested_backend") or ""
    )
    resolved_backend = str(
        payload.get("resolved_backend") or model_info.get("resolved_backend") or ""
    )
    requested_model_path = str(model_info.get("requested_model_path") or "")
    effective_model_version = str(model_info.get("effective_model_version") or "")
    if requested_backend != "api" or resolved_backend != "api":
        raise ValueError("TabPFN OOD source is not an explicit API runtime.")
    if requested_model_path != "v2.5_default":
        raise ValueError("TabPFN OOD source is not pinned to v2.5_default.")
    if effective_model_version != "api_2.5_plus":
        raise ValueError("TabPFN OOD source is not API 2.5 Plus.")
    return {
        "requested_backend": requested_backend,
        "resolved_backend": resolved_backend,
        "requested_model_path": requested_model_path,
        "effective_model_version": effective_model_version,
        "source_manifest": str(manifest_path.resolve()),
    }


def _identity_from_context(context: CalibrationAdapterContext) -> UnitIdentity:
    split = context.split_entry
    frozen = context.frozen_entry
    frozen_task = str(frozen.get("task_id", ""))
    if _task_alias(frozen_task) != _task_alias(split.task_id):
        raise ValueError(
            "Frozen parameter task differs from the canonical split task: "
            f"{frozen_task} != {split.task_id}"
        )
    snapshot_id = str(
        frozen.get("parameter_snapshot_id") or frozen.get("snapshot_id") or ""
    )
    parameter_hash = str(frozen.get("parameter_hash", ""))
    if not snapshot_id:
        raise ValueError("Frozen entry is missing parameter_snapshot_id.")
    if not parameter_hash:
        raise ValueError("Frozen entry is missing parameter_hash.")
    return UnitIdentity(
        task_id=split.task_id,
        protocol=split.protocol,
        split_id=split.split_id,
        model_family=str(frozen["model_family"]),
        model=str(frozen["model"]),
        parameter_snapshot_id=snapshot_id,
        parameter_hash=parameter_hash,
    )


def _task_alias(value: object) -> str:
    text = str(value)
    return re.sub(
        r"^(?:MatbenchSteels|Matbench[ _]Steel)__",
        "Matbench_Steel__",
        text,
        flags=re.IGNORECASE,
    )


def _effective_parameters(context: CalibrationAdapterContext) -> dict[str, Any]:
    value = context.frozen_entry.get("effective_parameters", {})
    if not isinstance(value, Mapping):
        raise ValueError("Frozen effective_parameters must be a mapping.")
    return deepcopy(dict(value))


def _traditional_parameters_for_seed(
    effective_parameters: Mapping[str, Any],
    *,
    model: str,
    seed: int,
) -> dict[str, Any]:
    seeded = deepcopy(dict(effective_parameters))
    args = dict(seeded.get("args", {}) or {})
    args["random_state"] = int(seed)
    seeded["args"] = args
    model_parameters = dict(seeded.get("model_params", {}) or {})
    seed_key = "random_seed" if model == "CatBoost" else "random_state"
    model_parameters[seed_key] = int(seed)
    seeded["model_params"] = model_parameters
    return seeded


def _frozen_model_seed(effective_parameters: Mapping[str, Any]) -> int:
    args = effective_parameters.get("args", {})
    if isinstance(args, Mapping) and "random_state" in args:
        return int(args["random_state"])
    if "random_state" in effective_parameters:
        return int(effective_parameters["random_state"])
    model_parameters = effective_parameters.get("model_params", {})
    if isinstance(model_parameters, Mapping):
        if "random_state" in model_parameters:
            return int(model_parameters["random_state"])
        if "random_seed" in model_parameters:
            return int(model_parameters["random_seed"])
    return 42


def _required_mc_seeds(
    context: CalibrationAdapterContext,
    *,
    family: str,
) -> tuple[int, ...]:
    if context.mc_passes != REQUIRED_MC_PASSES:
        raise ValueError(f"{family} calibration requires exactly 30 MC-dropout passes.")
    seeds = tuple(context.mc_seeds)
    if len(seeds) != REQUIRED_MC_PASSES or len(set(seeds)) != REQUIRED_MC_PASSES:
        raise ValueError(
            f"{family} calibration requires exactly 30 distinct MC-dropout seeds."
        )
    if seeds != MC_DROPOUT_SEEDS:
        raise ValueError(
            f"{family} calibration requires the fixed MC-dropout seed schedule 42..71."
        )
    return seeds


def _manifest_metadata(
    context: CalibrationAdapterContext,
    *,
    member_seed: int | None = None,
) -> dict[str, Any]:
    split = context.split_entry
    payload = {
        "calibration": {
            "task_id": split.task_id,
            "protocol": split.protocol,
            "split_id": split.split_id,
            "model_family": str(context.frozen_entry["model_family"]),
            "model": str(context.frozen_entry["model"]),
            "parameter_snapshot_id": str(
                context.frozen_entry.get("parameter_snapshot_id", "")
            ),
            "parameter_hash": str(context.frozen_entry.get("parameter_hash", "")),
        }
    }
    if member_seed is not None:
        payload["calibration"]["member_seed"] = int(member_seed)
    return payload


def _find_single_checkpoint(result_dir: str | Path) -> Path:
    root = Path(result_dir)
    expected = root / "checkpoints" / "best_model.pt"
    if expected.is_file():
        return expected.resolve()
    candidates = sorted(root.rglob("best_model.pt"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one best_model.pt below {root}, found {len(candidates)}"
        )
    return candidates[0].resolve()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_arrays(
    canonical_test: pd.DataFrame,
    target_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    if "ID" not in canonical_test.columns or target_column not in canonical_test.columns:
        raise ValueError("Canonical test is missing ID or target column.")
    ids = canonical_test["ID"].to_numpy(copy=True)
    if pd.Series(ids).duplicated().any():
        raise ValueError("Canonical test contains duplicate IDs.")
    y_true = pd.to_numeric(
        canonical_test[target_column],
        errors="raise",
    ).to_numpy(float)
    if not np.isfinite(y_true).all():
        raise ValueError("Canonical test y_true contains NaN/Inf values.")
    return ids, y_true


def _validate_loaded_test_data(
    data: Mapping[str, Any],
    *,
    canonical_test: pd.DataFrame,
    target_column: str,
    source_name: str,
) -> None:
    if "ids" not in data or "y" not in data:
        raise ValueError(f"{source_name} is missing canonical IDs or y_true.")
    canonical_ids, canonical_true = _canonical_arrays(canonical_test, target_column)
    source_ids = np.asarray(data["ids"]).reshape(-1)
    source_true = np.asarray(data["y"], dtype=float).reshape(-1)
    if [str(value) for value in source_ids] != [str(value) for value in canonical_ids]:
        raise ValueError(f"{source_name} ID order differs from canonical test.")
    if not np.allclose(
        source_true,
        canonical_true,
        rtol=SOURCE_TRUTH_RTOL,
        atol=SOURCE_TRUTH_ATOL,
    ):
        raise ValueError(f"{source_name} y_true differs from canonical test.")


def _validate_canonical_frame(
    frame: pd.DataFrame,
    *,
    canonical_test: pd.DataFrame,
    target_column: str,
    source_name: str,
) -> None:
    if "ID" not in frame.columns or target_column not in frame.columns:
        raise ValueError(f"{source_name} is missing ID or target column.")
    canonical_ids, canonical_true = _canonical_arrays(canonical_test, target_column)
    if frame["ID"].astype(str).tolist() != [str(value) for value in canonical_ids]:
        raise ValueError(f"{source_name} ID order differs from canonical test.")
    source_true = pd.to_numeric(frame[target_column], errors="raise").to_numpy(float)
    if not np.allclose(
        source_true,
        canonical_true,
        rtol=SOURCE_TRUTH_RTOL,
        atol=SOURCE_TRUTH_ATOL,
    ):
        raise ValueError(f"{source_name} y_true differs from canonical test.")


def _enable_mc_dropout(model: Any) -> None:
    import torch.nn as nn

    model.eval()
    dropout_types = (
        nn.Dropout,
        nn.Dropout1d,
        nn.Dropout2d,
        nn.Dropout3d,
        nn.AlphaDropout,
        nn.FeatureAlphaDropout,
    )
    for module in model.modules():
        if isinstance(module, dropout_types):
            module.train()


def _seed_torch(torch_module: Any, seed: int) -> None:
    torch_module.manual_seed(int(seed))
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(int(seed))


def _values_match(left: Any, right: Any) -> bool:
    try:
        return bool(np.isclose(float(left), float(right), rtol=1e-10, atol=1e-12))
    except (TypeError, ValueError):
        return left == right


def _resolve_runtime_path(value: Any, result_dir: Path) -> Path:
    if value is None or not str(value).strip():
        raise ValueError("LLMProp manifest is missing a required data path.")
    path = Path(str(value))
    if path.is_absolute():
        return path.resolve()
    candidates = [result_dir / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _sanitize_target_name(value: str) -> str:
    return (
        str(value)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("%", "percent")
    )


def _normalized_dataset_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _safe_component(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    if not text:
        raise ValueError(f"Empty calibration cache path component: {value!r}")
    return text


def _cache_manifest_path(feature_cache: Any) -> str:
    value = getattr(feature_cache, "cache_manifest", "")
    return str(Path(value).resolve()) if value else ""


def _single_source_path(frame: pd.DataFrame) -> str:
    values = frame["source_path"].astype(str).drop_duplicates().tolist()
    if len(values) != 1:
        raise ValueError(f"Expected one prediction source path, found {len(values)}")
    return values[0]
