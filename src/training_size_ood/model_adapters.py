"""Model-family adapters for the fixed-split training-size experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from src.data_processing.strength_ood_common import PreparedSplit


@dataclass(frozen=True)
class TaskRuntimeContext:
    task_id: str
    frozen_task_id: str
    alloy_type: str
    dataset_name: str
    target_col: str
    data_file: Path
    processing_cols: tuple[str, ...]
    processing_text_column: str | None
    model_seed: int
    cache_root: Path


@dataclass(frozen=True)
class RunMetrics:
    random_cv_mae: float
    test_set_mae: dict[str, float]
    metrics_file: str


@dataclass(frozen=True)
class StandardFeatureCache:
    full_train_file: Path
    test_file: Path
    test_set_files: dict[str, Path]
    cache_manifest: Path

    def materialize(self, prepared_split: PreparedSplit, result_dir: Path) -> dict[str, Any]:
        full_train = pd.read_csv(self.full_train_file)
        if "ID" not in full_train.columns or "ID" not in prepared_split.train_df.columns:
            raise ValueError("Feature cache materialization requires ID columns")
        if full_train["ID"].duplicated().any():
            raise ValueError("Cached training features contain duplicate IDs")
        indexed = full_train.set_index("ID", drop=False)
        train_ids = prepared_split.train_df["ID"].tolist()
        missing = [value for value in train_ids if value not in indexed.index]
        if missing:
            raise ValueError(f"Cached features are missing training IDs: {missing[:5]}")
        output_path = result_dir / "cached_features" / "train_features.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        indexed.loc[train_ids].reset_index(drop=True).to_csv(output_path, index=False)
        return {
            "train": str(output_path),
            "test": str(self.test_file),
            "test_sets": {name: str(path) for name, path in self.test_set_files.items()},
        }


@dataclass(frozen=True)
class TabPFNInputCache:
    full_train_file: Path
    test_file: Path
    test_set_files: dict[str, Path]
    cache_manifest: Path

    def materialize(self, prepared_split: PreparedSplit) -> PreparedSplit:
        full_train = pd.read_csv(self.full_train_file)
        if "ID" not in full_train.columns or "ID" not in prepared_split.train_df.columns:
            raise ValueError("TabPFN cache materialization requires ID columns")
        if full_train["ID"].duplicated().any():
            raise ValueError("Cached TabPFN training inputs contain duplicate IDs")
        indexed = full_train.set_index("ID", drop=False)
        train_ids = prepared_split.train_df["ID"].tolist()
        missing = [value for value in train_ids if value not in indexed.index]
        if missing:
            raise ValueError(f"Cached TabPFN inputs are missing training IDs: {missing[:5]}")
        return PreparedSplit(
            split_strategy=prepared_split.split_strategy,
            target_col=prepared_split.target_col,
            train_df=indexed.loc[train_ids].reset_index(drop=True),
            test_df=pd.read_csv(self.test_file),
            test_sets={name: pd.read_csv(path) for name, path in self.test_set_files.items()},
            summary=dict(prepared_split.summary),
            train_label=prepared_split.train_label,
            test_label=prepared_split.test_label,
        )


@dataclass(frozen=True)
class LLMPropInputCache:
    full_train_file: Path
    test_file: Path
    test_set_files: dict[str, Path]
    cache_manifest: Path

    def materialize(self, prepared_split: PreparedSplit, result_dir: Path) -> dict[str, Any]:
        full_train = pd.read_csv(self.full_train_file)
        if "ID" not in full_train.columns or "ID" not in prepared_split.train_df.columns:
            raise ValueError("LLMProp cache materialization requires ID columns")
        if full_train["ID"].duplicated().any():
            raise ValueError("Cached LLMProp training inputs contain duplicate IDs")
        indexed = full_train.set_index("ID", drop=False)
        train_ids = prepared_split.train_df["ID"].tolist()
        missing = [value for value in train_ids if value not in indexed.index]
        if missing:
            raise ValueError(f"Cached LLMProp inputs are missing training IDs: {missing[:5]}")
        output_path = result_dir / "llmprop_data" / "train_llmprop.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        indexed.loc[train_ids].reset_index(drop=True).to_csv(output_path, index=False)
        return {
            "train": str(output_path),
            "test": str(self.test_file),
            "test_sets": {name: str(path) for name, path in self.test_set_files.items()},
        }


class ModelAdapter(Protocol):
    def run(
        self,
        *,
        context: TaskRuntimeContext,
        model: str,
        prepared_split: PreparedSplit,
        effective_parameters: dict[str, Any],
        result_dir: Path,
        manifest_metadata: dict[str, Any],
        feature_cache: StandardFeatureCache | TabPFNInputCache | LLMPropInputCache | None = None,
    ) -> RunMetrics: ...


class StandardPipelineAdapter:
    def __init__(self, model_family: str):
        if model_family not in {"Traditional", "BERT"}:
            raise ValueError(f"Unsupported standard pipeline family: {model_family}")
        self.model_family = model_family

    def build_args(
        self,
        *,
        context: TaskRuntimeContext,
        model: str,
        effective_parameters: dict[str, Any],
        result_dir: Path,
    ) -> argparse.Namespace:
        from src.pipelines.ood_pipeline import create_argument_parser

        embedding_type = "tradition" if self.model_family == "Traditional" else _bert_embedding_type(model)
        parser = create_argument_parser()
        args = parser.parse_args(
            [
                "--data_file",
                str(context.data_file),
                "--result_dir",
                str(result_dir),
                "--target_column",
                context.target_col,
                "--alloy_type",
                context.alloy_type,
                "--embedding_type",
                embedding_type,
            ]
        )
        args.processing_cols = list(context.processing_cols)
        args.processing_text_column = context.processing_text_column
        for key, value in dict(effective_parameters.get("args", {})).items():
            if key not in {
                "data_file",
                "result_dir",
                "target_column",
                "target_columns",
                "use_optuna",
                "n_trials",
                "cross_validate",
            }:
                setattr(args, key, value)
        args.result_dir = str(result_dir)
        args.random_state = int(
            dict(effective_parameters.get("args", {})).get("random_state", context.model_seed)
        )
        args.use_optuna = False
        args.n_trials = 0
        args.cross_validate = False
        args.evaluate_after_train = True
        args.run_shap_analysis = False
        if self.model_family == "Traditional":
            args.use_nn = False
            args.models = [_traditional_pipeline_model(model)]
            args.use_composition_feature = True
            args.use_element_embedding = False
            args.use_process_embedding = False
        else:
            args.use_nn = True
            args.models = None
            args.embedding_type = embedding_type
            args.use_composition_feature = False
            args.use_element_embedding = True
            args.use_process_embedding = bool(context.processing_text_column)
        return args

    def prepare_cache(
        self,
        context: TaskRuntimeContext,
        model: str,
        effective_parameters: dict[str, Any],
        fixed_split: Any,
    ) -> StandardFeatureCache:
        from src.pipelines.ood_pipeline import prepare_feature_files

        args = self.build_args(
            context=context,
            model=model,
            effective_parameters=effective_parameters,
            result_dir=context.cache_root,
        )
        cache_dir = (
            context.cache_root
            / "features"
            / _safe_component(context.task_id)
            / _safe_component(args.embedding_type)
        )
        manifest_path = cache_dir / "cache_manifest.json"
        identity = {
            "task_id": context.task_id,
            "embedding_type": args.embedding_type,
            "split_hashes": fixed_split.to_manifest()["hashes"],
            "feature_config_hash": _feature_config_hash(args),
        }
        if manifest_path.exists():
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                files = payload.get("feature_files", {})
                candidate = _standard_cache_from_payload(manifest_path, files)
                if payload.get("identity") == identity and _standard_cache_exists(candidate):
                    return candidate
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                pass
        full_split = PreparedSplit(
            split_strategy="training_size_feature_cache",
            target_col=fixed_split.target_col,
            train_df=fixed_split.training_pool.copy(),
            test_df=fixed_split.random_cv_test.copy(),
            test_sets={
                "ood_top20": fixed_split.top20_test.copy(),
                "ood_loco_farthest": fixed_split.loco_test.copy(),
            },
            summary={"split_hashes": identity["split_hashes"]},
            train_label="training_pool",
            test_label="random_cv",
        )
        generated = prepare_feature_files(args, full_split, cache_dir)
        payload = {
            "identity": identity,
            "feature_files": generated,
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        return _standard_cache_from_payload(manifest_path, generated)

    def run(
        self,
        *,
        context: TaskRuntimeContext,
        model: str,
        prepared_split: PreparedSplit,
        effective_parameters: dict[str, Any],
        result_dir: Path,
        manifest_metadata: dict[str, Any],
        feature_cache: StandardFeatureCache | None = None,
    ) -> RunMetrics:
        from src.pipelines.ood_pipeline import run_prepared_split

        args = self.build_args(
            context=context,
            model=model,
            effective_parameters=effective_parameters,
            result_dir=result_dir,
        )
        run_prepared_split(
            args=args,
            prepared_split=prepared_split,
            result_dir=result_dir,
            frozen_parameters=effective_parameters,
            extra_manifest=manifest_metadata,
            precomputed_feature_files=(
                feature_cache.materialize(prepared_split, result_dir)
                if feature_cache is not None
                else None
            ),
        )
        return _load_standard_metrics(result_dir)

    def load_existing_metrics(self, result_dir: Path) -> RunMetrics:
        """Read metrics from a completed standard-pipeline evaluation without retraining."""
        return _load_standard_metrics(result_dir)


class TabPFNAdapter:
    @staticmethod
    def build_trainer_kwargs(
        context: TaskRuntimeContext,
        effective_parameters: dict[str, Any],
    ) -> dict[str, Any]:
        model_info = dict(effective_parameters.get("model_info", {}))
        model_version = (
            model_info.get("preferred_model_version")
            or model_info.get("effective_model_version")
            or model_info.get("requested_model_version")
        )
        return {
            "alloy_type": context.alloy_type,
            "target_col": context.target_col,
            "base_path": ".",
            "random_state": int(effective_parameters.get("random_state", context.model_seed)),
            "split_strategy": "training_size_hybrid_ood",
            "backend": str(
                effective_parameters.get("requested_backend")
                or model_info.get("requested_backend")
                or model_info.get("backend")
                or "auto"
            ),
            "model_version": str(model_version) if model_version else None,
            "feature_mode": str(
                effective_parameters.get("feature_mode")
                or model_info.get("feature_mode")
                or model_info.get("requested_feature_mode")
            ),
        }

    def prepare_cache(
        self,
        context: TaskRuntimeContext,
        model: str,
        effective_parameters: dict[str, Any],
        fixed_split: Any,
    ) -> TabPFNInputCache:
        feature_cols = [str(value) for value in effective_parameters.get("feature_cols", [])]
        if not feature_cols:
            raise ValueError("Frozen TabPFN parameters must define feature_cols")
        feature_mode = self.build_trainer_kwargs(context, effective_parameters)["feature_mode"]
        cache_dir = (
            context.cache_root
            / "tabpfn"
            / _safe_component(context.task_id)
            / _safe_component(feature_mode)
        )
        manifest_path = cache_dir / "cache_manifest.json"
        identity = {
            "task_id": context.task_id,
            "feature_mode": feature_mode,
            "feature_columns": feature_cols,
            "target_column": context.target_col,
            "split_hashes": fixed_split.to_manifest()["hashes"],
        }
        if manifest_path.exists():
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                candidate = _tabpfn_cache_from_payload(manifest_path, payload.get("input_files", {}))
                if payload.get("identity") == identity and _tabpfn_cache_exists(candidate):
                    return candidate
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                pass
        columns = list(dict.fromkeys(["ID", *feature_cols, context.target_col]))

        def select(frame: pd.DataFrame, label: str) -> pd.DataFrame:
            missing = [column for column in columns if column not in frame.columns]
            if missing:
                raise ValueError(f"TabPFN {label} cache source is missing columns: {missing}")
            return frame[columns].copy()

        cache_dir.mkdir(parents=True, exist_ok=True)
        train_path = cache_dir / "training_pool.csv"
        test_path = cache_dir / "random_cv.csv"
        select(fixed_split.training_pool, "training_pool").to_csv(train_path, index=False)
        select(fixed_split.random_cv_test, "random_cv").to_csv(test_path, index=False)
        test_set_paths: dict[str, str] = {}
        for name, frame in {
            "ood_top20": fixed_split.top20_test,
            "ood_loco_farthest": fixed_split.loco_test,
        }.items():
            path = cache_dir / f"{name}.csv"
            select(frame, name).to_csv(path, index=False)
            test_set_paths[name] = str(path)
        inputs = {
            "train": str(train_path),
            "test": str(test_path),
            "test_sets": test_set_paths,
        }
        manifest_path.write_text(
            json.dumps({"identity": identity, "input_files": inputs}, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        return _tabpfn_cache_from_payload(manifest_path, inputs)

    def run(
        self,
        *,
        context: TaskRuntimeContext,
        model: str,
        prepared_split: PreparedSplit,
        effective_parameters: dict[str, Any],
        result_dir: Path,
        manifest_metadata: dict[str, Any],
        feature_cache: TabPFNInputCache | None = None,
    ) -> RunMetrics:
        from src.TabPFN.train_tabpfn_ood import TabPFNOODTrainer

        trainer = TabPFNOODTrainer(**self.build_trainer_kwargs(context, effective_parameters))
        cached_split = (
            feature_cache.materialize(prepared_split)
            if isinstance(feature_cache, TabPFNInputCache)
            else prepared_split
        )
        result = trainer.run_canonical_split(
            cached_split,
            result_dir=result_dir,
            feature_cols=list(effective_parameters.get("feature_cols", [])),
            align_predictions=False,
            fold_metadata=manifest_metadata,
        )
        test_metrics = dict(result.get("test_set_metrics", {}))
        return RunMetrics(
            random_cv_mae=float(result[trainer.test_result_key]["mae"]),
            test_set_mae={name: float(values["mae"]) for name, values in test_metrics.items()},
            metrics_file=str(result_dir / "metrics" / "metrics_summary.json"),
        )


class LLMPropAdapter:
    def prepare_cache(
        self,
        context: TaskRuntimeContext,
        model: str,
        effective_parameters: dict[str, Any],
        fixed_split: Any,
    ) -> LLMPropInputCache:
        from src.LLMProp.ood_adapter import convert_split_to_llmprop_csv
        from src.data_processing.strength_ood_common import save_prepared_split

        cache_dir = context.cache_root / "llmprop" / _safe_component(context.task_id)
        manifest_path = cache_dir / "cache_manifest.json"
        identity = {
            "task_id": context.task_id,
            "split_hashes": fixed_split.to_manifest()["hashes"],
            "target_column": context.target_col,
            "processing_text_column": context.processing_text_column,
        }
        if manifest_path.exists():
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                candidate = _llmprop_cache_from_payload(manifest_path, payload.get("input_files", {}))
                if payload.get("identity") == identity and _llmprop_cache_exists(candidate):
                    return candidate
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                pass
        full_split = PreparedSplit(
            split_strategy="training_size_llmprop_cache",
            target_col=fixed_split.target_col,
            train_df=fixed_split.training_pool.copy(),
            test_df=fixed_split.random_cv_test.copy(),
            test_sets={
                "ood_top20": fixed_split.top20_test.copy(),
                "ood_loco_farthest": fixed_split.loco_test.copy(),
            },
            summary={"split_hashes": identity["split_hashes"]},
            train_label="training_pool",
            test_label="random_cv",
        )
        artifacts = save_prepared_split(full_split, cache_dir / "split_data")
        data_dir = cache_dir / "converted"
        train_output = data_dir / "training_pool_llmprop.csv"
        test_output = data_dir / "random_cv_llmprop.csv"
        convert_split_to_llmprop_csv(
            artifacts["train_file"],
            train_output,
            context.target_col,
            processing_text_column=context.processing_text_column,
        )
        convert_split_to_llmprop_csv(
            artifacts["test_file"],
            test_output,
            context.target_col,
            processing_text_column=context.processing_text_column,
        )
        test_set_outputs: dict[str, str] = {}
        for name, source_path in dict(artifacts.get("test_set_files", {})).items():
            output_path = data_dir / f"{name}_llmprop.csv"
            convert_split_to_llmprop_csv(
                source_path,
                output_path,
                context.target_col,
                processing_text_column=context.processing_text_column,
            )
            test_set_outputs[name] = str(output_path)
        inputs = {
            "train": str(train_output),
            "test": str(test_output),
            "test_sets": test_set_outputs,
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps({"identity": identity, "input_files": inputs}, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        return _llmprop_cache_from_payload(manifest_path, inputs)

    @staticmethod
    def build_config_kwargs(
        *,
        context: TaskRuntimeContext,
        effective_parameters: dict[str, Any],
        result_dir: Path,
        train_csv: Path,
        test_csv: Path,
        test_set_csvs: dict[str, str],
        split_manifest: dict[str, Any],
    ) -> dict[str, Any]:
        from src.LLMProp.trainer import LLMPropTrainingConfig

        allowed = {item.name for item in fields(LLMPropTrainingConfig)}
        blocked = {
            "result_dir",
            "train_csv",
            "test_csv",
            "test_set_csvs",
            "target_column",
            "split_manifest",
            "fold_index",
            "ood_method",
            "use_optuna",
            "n_trials",
        }
        kwargs = {
            key: value
            for key, value in effective_parameters.items()
            if key in allowed and key not in blocked and key != "normalizer"
        }
        normalizer = effective_parameters.get("normalizer", "z_norm")
        if isinstance(normalizer, dict):
            normalizer = normalizer.get("type", "z_norm")
        kwargs.update(
            {
                "result_dir": str(result_dir),
                "train_csv": str(train_csv),
                "test_csv": str(test_csv),
                "test_set_csvs": dict(test_set_csvs),
                "target_column": context.target_col,
                "random_state": int(effective_parameters.get("random_state", context.model_seed)),
                "split_manifest": split_manifest,
                "fold_index": None,
                "ood_method": "training_size_hybrid_ood",
                "normalizer": str(normalizer),
                "use_optuna": False,
                "n_trials": 0,
            }
        )
        return kwargs

    def run(
        self,
        *,
        context: TaskRuntimeContext,
        model: str,
        prepared_split: PreparedSplit,
        effective_parameters: dict[str, Any],
        result_dir: Path,
        manifest_metadata: dict[str, Any],
        feature_cache: StandardFeatureCache | LLMPropInputCache | None = None,
    ) -> RunMetrics:
        from src.LLMProp.ood_adapter import convert_split_to_llmprop_csv
        from src.LLMProp.trainer import LLMPropTrainingConfig, run_llmprop_ood_training
        from src.data_processing.strength_ood_common import save_prepared_split

        split_artifacts = save_prepared_split(prepared_split, result_dir / "split_data")
        if isinstance(feature_cache, LLMPropInputCache):
            converted = feature_cache.materialize(prepared_split, result_dir)
            train_csv = Path(converted["train"])
            test_csv = Path(converted["test"])
            test_set_csvs = dict(converted["test_sets"])
        else:
            data_dir = result_dir / "llmprop_data"
            train_csv = data_dir / "train_llmprop.csv"
            test_csv = data_dir / "random_cv_llmprop.csv"
            convert_split_to_llmprop_csv(
                split_artifacts["train_file"],
                train_csv,
                context.target_col,
                processing_text_column=context.processing_text_column,
            )
            convert_split_to_llmprop_csv(
                split_artifacts["test_file"],
                test_csv,
                context.target_col,
                processing_text_column=context.processing_text_column,
            )
            test_set_csvs = {}
            for name, source_path in dict(split_artifacts.get("test_set_files", {})).items():
                output_path = data_dir / f"{name}_llmprop.csv"
                convert_split_to_llmprop_csv(
                    source_path,
                    output_path,
                    context.target_col,
                    processing_text_column=context.processing_text_column,
                )
                test_set_csvs[str(name)] = str(output_path)
        split_manifest = {**manifest_metadata, "split_summary": prepared_split.summary}
        config = LLMPropTrainingConfig(
            **self.build_config_kwargs(
                context=context,
                effective_parameters=effective_parameters,
                result_dir=result_dir,
                train_csv=train_csv,
                test_csv=test_csv,
                test_set_csvs=test_set_csvs,
                split_manifest=split_manifest,
            )
        )
        result = run_llmprop_ood_training(config)
        metrics = dict(result["metrics"])
        return RunMetrics(
            random_cv_mae=_llmprop_test_mae(metrics, context.target_col),
            test_set_mae={
                name: float(values["mae"])
                for name, values in dict(metrics.get("test_set_metrics", {})).items()
            },
            metrics_file=str(result_dir / "final_evaluation_metrics.json"),
        )


def get_model_adapter(model_family: str) -> ModelAdapter:
    if model_family in {"Traditional", "BERT"}:
        return StandardPipelineAdapter(model_family)
    if model_family == "TabPFN":
        return TabPFNAdapter()
    if model_family == "LLMProp":
        return LLMPropAdapter()
    raise ValueError(f"Unsupported training-size model family: {model_family}")


def _traditional_pipeline_model(model: str) -> str:
    mapping = {
        "RF": "sklearn_rf",
        "XGB": "xgboost",
        "MLP": "mlp",
        "CatBoost": "catboost",
        "LightGBM": "lightgbm",
    }
    if model not in mapping:
        raise ValueError(f"No pipeline-model mapping for Traditional model: {model}")
    return mapping[model]


def _bert_embedding_type(model: str) -> str:
    mapping = {"SciBERT": "scibert", "SteelBERT": "steelbert", "MatSciBERT": "matscibert"}
    if model not in mapping:
        raise ValueError(f"No embedding mapping for BERT model: {model}")
    return mapping[model]


def _load_standard_metrics(result_dir: Path) -> RunMetrics:
    candidates = sorted(result_dir.rglob("final_evaluation_metrics.json"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one final_evaluation_metrics.json below {result_dir}, found {len(candidates)}"
        )
    payload = json.loads(candidates[0].read_text(encoding="utf-8"))
    random_cv_mae = _find_test_mae(payload)
    test_sets = dict(payload.get("test_set_metrics", {}))
    combined = dict(payload.get("combined_evaluation_summary", {}))
    if not test_sets:
        test_sets = dict(combined.get("test_set_metrics", {}))
    return RunMetrics(
        random_cv_mae=random_cv_mae,
        test_set_mae=_named_test_set_mae(test_sets),
        metrics_file=str(candidates[0]),
    )


def _named_test_set_mae(test_sets: dict[str, Any]) -> dict[str, float]:
    if not test_sets or not all(isinstance(values, dict) for values in test_sets.values()):
        return {}
    return {
        str(name): float(values["mae"])
        for name, values in test_sets.items()
    }


def _find_test_mae(payload: dict[str, Any]) -> float:
    for key, value in payload.items():
        if _is_random_cv_test_mae_key(key):
            return float(value)
    combined = dict(payload.get("combined_evaluation_summary", {}))
    for key, value in combined.items():
        if _is_random_cv_test_mae_key(key):
            return float(value)
    test_metrics = combined.get("test")
    if isinstance(test_metrics, dict) and "mae" in test_metrics:
        return float(test_metrics["mae"])
    for nested in (
        payload.get("test_set_metrics"),
        combined.get("test_set_metrics"),
    ):
        if not isinstance(nested, dict):
            continue
        for key, value in nested.items():
            if _is_random_cv_test_mae_key(key):
                return float(value)
        if "mae" in nested:
            return float(nested["mae"])
    raise KeyError("Cannot locate Random-CV test MAE in final metrics")


def _llmprop_test_mae(metrics: dict[str, Any], target_column: str) -> float:
    legacy_key = f"{target_column}_test_mae"
    if legacy_key in metrics:
        return float(metrics[legacy_key])
    return _find_test_mae(metrics)


def _is_random_cv_test_mae_key(key: Any) -> bool:
    normalized = str(key).lower()
    if not normalized.endswith("_mae"):
        return False
    if normalized.startswith("ood_") or "_ood_" in normalized:
        return False
    return normalized.startswith("test_") or "_test_" in normalized or normalized.endswith("_test_mae")


def _safe_component(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in str(value))


def _feature_config_hash(args: argparse.Namespace) -> str:
    payload = {
        "embedding_type": args.embedding_type,
        "use_composition_feature": bool(args.use_composition_feature),
        "use_element_embedding": bool(args.use_element_embedding),
        "use_process_embedding": bool(args.use_process_embedding),
        "use_temperature": bool(args.use_temperature),
        "processing_cols": list(args.processing_cols or []),
        "processing_text_column": args.processing_text_column,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _standard_cache_from_payload(
    manifest_path: Path,
    files: dict[str, Any],
) -> StandardFeatureCache:
    return StandardFeatureCache(
        full_train_file=Path(str(files["train"])),
        test_file=Path(str(files["test"])),
        test_set_files={name: Path(str(path)) for name, path in dict(files.get("test_sets", {})).items()},
        cache_manifest=manifest_path,
    )


def _standard_cache_exists(cache: StandardFeatureCache) -> bool:
    return all(
        path.exists()
        for path in [cache.full_train_file, cache.test_file, *cache.test_set_files.values()]
    )


def _llmprop_cache_from_payload(
    manifest_path: Path,
    files: dict[str, Any],
) -> LLMPropInputCache:
    return LLMPropInputCache(
        full_train_file=Path(str(files["train"])),
        test_file=Path(str(files["test"])),
        test_set_files={name: Path(str(path)) for name, path in dict(files.get("test_sets", {})).items()},
        cache_manifest=manifest_path,
    )


def _llmprop_cache_exists(cache: LLMPropInputCache) -> bool:
    return all(
        path.exists()
        for path in [cache.full_train_file, cache.test_file, *cache.test_set_files.values()]
    )


def _tabpfn_cache_from_payload(
    manifest_path: Path,
    files: dict[str, Any],
) -> TabPFNInputCache:
    return TabPFNInputCache(
        full_train_file=Path(str(files["train"])),
        test_file=Path(str(files["test"])),
        test_set_files={name: Path(str(path)) for name, path in dict(files.get("test_sets", {})).items()},
        cache_manifest=manifest_path,
    )


def _tabpfn_cache_exists(cache: TabPFNInputCache) -> bool:
    return all(
        path.exists()
        for path in [cache.full_train_file, cache.test_file, *cache.test_set_files.values()]
    )
