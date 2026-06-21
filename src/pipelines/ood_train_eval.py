from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.models.evaluators import AlloysEvaluator, BaseEvaluator, EvaluatorFactory
from src.models.evaluators.ml_evaluator import MLEvaluator
from src.models.evaluators.shap_analyzer import MLShapAnalyzer
from src.models.ml_train_eval_pipeline.pipeline import MLTrainingPipeline, convert_numpy_types
from src.models.nn_train_eval_pipeline.pipeline import TrainingPipeline as NNTrainingPipeline
from src.models.trainers import TrainerFactory


OOD_INNER_VAL_RATIO = 0.2


def _load_feature_dataframe(feature_file: str) -> pd.DataFrame:
    if not Path(feature_file).exists():
        raise FileNotFoundError(f"Feature file not found: {feature_file}")
    return pd.read_csv(feature_file)


def _resolve_target_columns(args: Any) -> List[str]:
    target_column = getattr(args, "target_column", None)
    target_columns = getattr(args, "target_columns", None)

    if target_column is not None:
        if target_columns is not None and list(target_columns) != [target_column]:
            raise ValueError("OOD training only supports a single target_column per run")
        return [target_column]

    if target_columns is None:
        raise ValueError("OOD training requires target_column")

    resolved = list(target_columns)
    if len(resolved) != 1:
        raise ValueError("OOD training only supports a single target_column per run")
    return resolved


def _select_ml_feature_columns(
    df: pd.DataFrame,
    target_columns: List[str],
    processing_cols: Optional[List[str]],
    use_composition_feature: bool,
    use_temperature: bool,
    other_features_name: Optional[List[str]],
) -> List[str]:
    feature_cols = [col for col in df.columns if col not in target_columns and col != "ID"]
    composition_cols = [col for col in feature_cols if "(wt%)" in col or "(at%)" in col]
    other_cols: List[str] = []
    if other_features_name and other_features_name != ["None"]:
        for name in other_features_name:
            stripped = name.strip()
            if stripped:
                other_cols.extend([col for col in feature_cols if stripped in col])

    selected_cols: List[str] = []
    if use_composition_feature:
        selected_cols.extend(composition_cols)
    if other_cols:
        selected_cols.extend(other_cols)
    if processing_cols:
        selected_cols.extend([col for col in processing_cols if col in feature_cols])
    if use_temperature:
        selected_cols.extend([col for col in feature_cols if "temperature" in col.lower()])

    selected_cols = list(dict.fromkeys(selected_cols))
    if not selected_cols:
        raise ValueError("No ML features selected for OOD training")
    return selected_cols


def _select_nn_feature_columns(
    df: pd.DataFrame,
    target_columns: List[str],
    args: Any,
) -> List[str]:
    feature_cols = [col for col in df.columns if col not in target_columns and col != "ID"]

    ele_emb_cols = [col for col in feature_cols if "ele_emb" in col]
    proc_emb_cols = [col for col in feature_cols if "proc_emb" in col]
    joint_emb_cols = [col for col in feature_cols if "joint_emb" in col]
    composition_cols = [col for col in feature_cols if "(wt%)" in col or "(at%)" in col]
    feature1_cols = [col for col in feature_cols if "feature1" in col.lower()]
    feature2_cols = [col for col in feature_cols if "feature2" in col.lower()]
    temperature_cols = [col for col in feature_cols if "temperature" in col.lower()]
    other_cols: List[str] = []

    if args.other_features_name and args.other_features_name != ["None"]:
        for name in args.other_features_name:
            stripped = name.strip()
            if stripped:
                other_cols.extend([col for col in feature_cols if stripped in col])

    selected_cols: List[str] = []
    if args.use_element_embedding:
        selected_cols.extend(ele_emb_cols)
    if args.use_process_embedding:
        selected_cols.extend(proc_emb_cols)
    if getattr(args, "use_joint_composition_process_embedding", False):
        selected_cols.extend(joint_emb_cols)
    if args.use_composition_feature:
        selected_cols.extend(composition_cols)
    if getattr(args, "use_feature1", False):
        selected_cols.extend(feature1_cols)
    if getattr(args, "use_feature2", False):
        selected_cols.extend(feature2_cols)
    if args.use_temperature:
        selected_cols.extend(temperature_cols)
    if other_cols:
        selected_cols.extend(other_cols)

    selected_cols = list(dict.fromkeys(selected_cols))
    if not selected_cols:
        raise ValueError("No NN features selected for OOD training")
    return selected_cols


def _clone_split_data(data: Dict[str, Any]) -> Dict[str, Any]:
    cloned: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            cloned[key] = value.copy()
        elif isinstance(value, list):
            cloned[key] = list(value)
        else:
            cloned[key] = value
    return cloned


def _build_split_data(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    ids: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    split: Dict[str, Any] = {
        "X": X.astype(np.float32, copy=False),
        "y": y.astype(np.float32, copy=False),
        "feature_names": list(feature_names),
    }
    if ids is not None:
        split["ids"] = np.asarray(ids)
    return split


def _fit_ood_scalers(train_data: Dict[str, Any]) -> Tuple[StandardScaler, StandardScaler]:
    scaler_X = StandardScaler().fit(train_data["X"])
    scaler_y = StandardScaler().fit(train_data["y"])
    return scaler_X, scaler_y


def _transform_split_data(
    data: Dict[str, Any],
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
) -> Dict[str, Any]:
    ids = np.asarray(data["ids"]) if "ids" in data else None
    return _build_split_data(
        scaler_X.transform(data["X"]),
        scaler_y.transform(data["y"]),
        data["feature_names"],
        ids,
    )


def _save_scalers(result_dir: str, scaler_X: StandardScaler, scaler_y: StandardScaler) -> None:
    result_path = Path(result_dir)
    result_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler_X, result_path / "scaler_X.pkl")
    joblib.dump(scaler_y, result_path / "scaler_y.pkl")


def _split_inner_train_val(
    data: Dict[str, Any],
    random_state: int,
    val_ratio: float = OOD_INNER_VAL_RATIO,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if "ids" in data:
        X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
            data["X"],
            data["y"],
            data["ids"],
            test_size=val_ratio,
            random_state=random_state,
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            data["X"],
            data["y"],
            test_size=val_ratio,
            random_state=random_state,
        )
        ids_train = ids_val = None

    return (
        _build_split_data(X_train, y_train, data["feature_names"], ids_train),
        _build_split_data(X_val, y_val, data["feature_names"], ids_val),
    )


def _feature_frame_to_split_data(
    df: pd.DataFrame,
    selected_cols: List[str],
    target_columns: List[str],
) -> Dict[str, np.ndarray]:
    missing = [col for col in selected_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Feature file is missing selected columns: {missing}")
    split_data: Dict[str, np.ndarray] = {
        "X": df[selected_cols].fillna(0).values.astype("float32"),
        "y": df[target_columns].values.astype("float32"),
        "feature_names": selected_cols,
    }
    if "ID" in df.columns:
        split_data["ids"] = df["ID"].values
    return split_data


def load_ood_feature_split(
    train_feature_file: str,
    test_feature_file: str,
    target_columns: List[str],
    args: Any,
    pipeline_kind: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    train_df = _load_feature_dataframe(train_feature_file)
    test_df = _load_feature_dataframe(test_feature_file)

    if pipeline_kind == "ml":
        selected_cols = _select_ml_feature_columns(
            df=train_df,
            target_columns=target_columns,
            processing_cols=args.processing_cols,
            use_composition_feature=args.use_composition_feature,
            use_temperature=args.use_temperature,
            other_features_name=args.other_features_name,
        )
    elif pipeline_kind == "nn":
        selected_cols = _select_nn_feature_columns(train_df, target_columns, args)
    else:
        raise ValueError(f"Unsupported pipeline kind: {pipeline_kind}")

    train_val_data = _feature_frame_to_split_data(train_df, selected_cols, target_columns)
    test_data = _feature_frame_to_split_data(test_df, selected_cols, target_columns)
    return train_val_data, test_data, selected_cols


def load_ood_test_set_feature_splits(
    test_set_feature_files: Dict[str, str] | None,
    target_columns: List[str],
    selected_cols: List[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    result: Dict[str, Dict[str, np.ndarray]] = {}
    for test_set_name, feature_file in (test_set_feature_files or {}).items():
        test_set_df = _load_feature_dataframe(feature_file)
        result[str(test_set_name)] = _feature_frame_to_split_data(test_set_df, selected_cols, target_columns)
    return result


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(np.abs(y_true) < 1e-12, np.nan, np.abs(y_true))
    values = np.abs((y_true - y_pred) / denom)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite) * 100.0)


def _plain_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true_2d = y_true.reshape(-1, 1) if y_true.ndim == 1 else y_true
    y_pred_2d = y_pred.reshape(-1, 1) if y_pred.ndim == 1 else y_pred
    return {
        "mae": float(mean_absolute_error(y_true_2d, y_pred_2d)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_2d, y_pred_2d))),
        "r2": float(r2_score(y_true_2d, y_pred_2d)),
        "mape": _safe_mape(y_true_2d, y_pred_2d),
        "n_samples": int(len(y_true_2d)),
    }


def _save_test_set_predictions(
    result_dir: str,
    test_set_name: str,
    data: Dict[str, Any],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_columns: List[str],
) -> str:
    y_true_2d = y_true.reshape(-1, 1) if y_true.ndim == 1 else y_true
    y_pred_2d = y_pred.reshape(-1, 1) if y_pred.ndim == 1 else y_pred
    out = pd.DataFrame()
    if "ids" in data:
        out["ID"] = data["ids"]
    out["Dataset"] = test_set_name
    for i, target_name in enumerate(target_columns):
        out[f"{target_name}_Actual"] = y_true_2d[:, i]
        out[f"{target_name}_Predicted"] = y_pred_2d[:, i]
    predictions_dir = Path(result_dir) / "predictions" / "test_sets"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    output_path = predictions_dir / f"{test_set_name}_predictions.csv"
    out.to_csv(output_path, index=False)
    return str(output_path)


def _rewrite_prediction_labels(result_dir: str) -> None:
    predictions_dir = Path(result_dir) / "predictions"
    if not predictions_dir.exists():
        return

    for csv_path in predictions_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        changed = False
        if "Dataset" in df.columns:
            updated = df["Dataset"].replace({"Test": "OODTest", "ExtrapolationTest": "OODTest"})
            if not updated.equals(df["Dataset"]):
                df["Dataset"] = updated
                changed = True
        if "set" in df.columns:
            updated = df["set"].replace({"test": "OODTest"})
            if not updated.equals(df["set"]):
                df["set"] = updated
                changed = True
        if changed:
            df.to_csv(csv_path, index=False)


def _write_ood_manifest(
    result_dir: str,
    train_feature_file: str,
    test_feature_file: str,
    target_column: str,
    test_set_feature_files: Dict[str, str] | None = None,
) -> None:
    manifest = {
        "train_feature_file": train_feature_file,
        "test_feature_file": test_feature_file,
        "combined_test_feature_file": test_feature_file,
        "test_set_feature_files": dict(test_set_feature_files or {}),
        "target_column": target_column,
        "test_dataset_label": "OODTest",
    }
    payload = json.dumps(manifest, indent=2, ensure_ascii=False)
    Path(result_dir, "ood_manifest.json").write_text(payload, encoding="utf-8")


class OODMLTrainingPipeline(MLTrainingPipeline):
    def __init__(
        self,
        args: Any,
        train_feature_file: str,
        test_feature_file: str,
        test_set_feature_files: Dict[str, str] | None = None,
    ):
        super().__init__(args)
        self._train_feature_file = train_feature_file
        self._test_feature_file = test_feature_file
        self._test_set_feature_files = dict(test_set_feature_files or {})
        self.raw_outer_train_data: Dict[str, Any] | None = None
        self.raw_outer_test_data: Dict[str, Any] | None = None
        self.raw_test_set_data: Dict[str, Dict[str, Any]] = {}
        self.outer_train_data: Dict[str, Any] | None = None
        self.outer_test_data: Dict[str, Any] | None = None
        self.test_set_data: Dict[str, Dict[str, Any]] = {}
        self.inner_train_data: Dict[str, Any] | None = None
        self.inner_val_data: Dict[str, Any] | None = None
        self._last_model_params: Dict[str, Any] | None = None

    def _prepare_data(self) -> None:
        target_columns = _resolve_target_columns(self.args)
        raw_outer_train_data, raw_outer_test_data, self.feature_names = load_ood_feature_split(
            train_feature_file=self._train_feature_file,
            test_feature_file=self._test_feature_file,
            target_columns=target_columns,
            args=self.args,
            pipeline_kind="ml",
        )
        raw_inner_train_data, raw_inner_val_data = _split_inner_train_val(
            raw_outer_train_data,
            random_state=self.args.random_state,
        )
        self.raw_outer_train_data = _clone_split_data(raw_outer_train_data)
        self.raw_outer_test_data = _clone_split_data(raw_outer_test_data)
        self.raw_test_set_data = {
            name: _clone_split_data(data)
            for name, data in load_ood_test_set_feature_splits(
                self._test_set_feature_files,
                target_columns=target_columns,
                selected_cols=self.feature_names,
            ).items()
        }

        self.scaler_X, self.scaler_y = _fit_ood_scalers(raw_inner_train_data)
        self.inner_train_data = _transform_split_data(raw_inner_train_data, self.scaler_X, self.scaler_y)
        self.inner_val_data = _transform_split_data(raw_inner_val_data, self.scaler_X, self.scaler_y)
        self.outer_train_data = _transform_split_data(raw_outer_train_data, self.scaler_X, self.scaler_y)
        self.outer_test_data = _transform_split_data(raw_outer_test_data, self.scaler_X, self.scaler_y)
        self.test_set_data = {
            name: _transform_split_data(data, self.scaler_X, self.scaler_y)
            for name, data in self.raw_test_set_data.items()
        }
        self.train_val_data = _clone_split_data(self.outer_train_data)
        self.test_data = _clone_split_data(self.outer_test_data)

        _save_scalers(self.args.result_dir, self.scaler_X, self.scaler_y)

    def _train_final_model(self, model_params: Dict[str, Any] | None = None) -> None:
        if self.inner_train_data is None or self.inner_val_data is None:
            raise ValueError("OOD inner train/validation split is not initialized")
        self._last_model_params = None if model_params is None else dict(model_params)

        saved_train_val = self.train_val_data
        saved_test = self.test_data
        self.train_val_data = self.inner_train_data
        self.test_data = self.inner_val_data
        try:
            super()._train_final_model(model_params)
        finally:
            self.train_val_data = saved_train_val
            self.test_data = saved_test

    def _prepare_final_outer_refit_data(self) -> None:
        if self.raw_outer_train_data is None or self.raw_outer_test_data is None:
            raise ValueError("Raw OOD outer split is not initialized")
        self.scaler_X, self.scaler_y = _fit_ood_scalers(self.raw_outer_train_data)
        self.outer_train_data = _transform_split_data(self.raw_outer_train_data, self.scaler_X, self.scaler_y)
        self.outer_test_data = _transform_split_data(self.raw_outer_test_data, self.scaler_X, self.scaler_y)
        self.test_set_data = {
            name: _transform_split_data(data, self.scaler_X, self.scaler_y)
            for name, data in self.raw_test_set_data.items()
        }
        _save_scalers(self.args.result_dir, self.scaler_X, self.scaler_y)

    def _retrain_best_model_on_outer_train(self) -> None:
        if self.outer_train_data is None:
            raise ValueError("Final outer-train refit data is not initialized")

        saved_train_val = self.train_val_data
        saved_test = self.test_data
        self.train_val_data = self.outer_train_data
        self.test_data = self.outer_train_data
        try:
            MLTrainingPipeline._train_final_model(self, self._last_model_params)
        finally:
            self.train_val_data = saved_train_val
            self.test_data = saved_test

    def _save_predictions(self) -> None:
        if (
            self.best_model is None
            or self.outer_train_data is None
            or self.outer_test_data is None
        ):
            raise ValueError("OOD ML prediction state is not initialized")

        def ensure_2d(arr: np.ndarray) -> np.ndarray:
            return arr.reshape(-1, 1) if arr.ndim == 1 else arr

        y_train_pred = ensure_2d(self.best_model.predict(self.outer_train_data["X"]))
        y_test_pred = ensure_2d(self.best_model.predict(self.outer_test_data["X"]))

        y_train_true = self.scaler_y.inverse_transform(ensure_2d(self.outer_train_data["y"]))
        y_test_true = self.scaler_y.inverse_transform(ensure_2d(self.outer_test_data["y"]))
        y_train_pred = self.scaler_y.inverse_transform(y_train_pred)
        y_test_pred = self.scaler_y.inverse_transform(y_test_pred)

        train_df = pd.DataFrame()
        test_df = pd.DataFrame()

        if "ids" in self.outer_train_data:
            train_df["ID"] = self.outer_train_data["ids"]
        if "ids" in self.outer_test_data:
            test_df["ID"] = self.outer_test_data["ids"]

        train_df["Dataset"] = "Train"
        test_df["Dataset"] = "OODTest"

        for i, target_name in enumerate(self.args.target_columns):
            train_df[f"{target_name}_Actual"] = y_train_true[:, i]
            train_df[f"{target_name}_Predicted"] = y_train_pred[:, i]
            test_df[f"{target_name}_Actual"] = y_test_true[:, i]
            test_df[f"{target_name}_Predicted"] = y_test_pred[:, i]

        predictions_dir = Path(self.args.result_dir) / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        pd.concat([train_df, test_df], ignore_index=True).to_csv(
            predictions_dir / "all_predictions.csv",
            index=False,
        )
        train_df.to_csv(predictions_dir / "train_predictions.csv", index=False)
        val_predictions_path = predictions_dir / "val_predictions.csv"
        if val_predictions_path.exists():
            val_predictions_path.unlink()
        test_df.to_csv(predictions_dir / "test_predictions.csv", index=False)

        for test_set_name, test_set_data in self.test_set_data.items():
            y_test_set_pred = ensure_2d(self.best_model.predict(test_set_data["X"]))
            y_test_set_true = self.scaler_y.inverse_transform(ensure_2d(test_set_data["y"]))
            y_test_set_pred = self.scaler_y.inverse_transform(y_test_set_pred)
            _save_test_set_predictions(
                result_dir=self.args.result_dir,
                test_set_name=test_set_name,
                data=test_set_data,
                y_true=y_test_set_true,
                y_pred=y_test_set_pred,
                target_columns=self.args.target_columns,
            )

    def _evaluate_best_model(self) -> None:
        if (
            self.best_model is None
            or self.outer_train_data is None
            or self.outer_test_data is None
        ):
            raise ValueError("OOD ML evaluation state is not initialized")

        evaluator = MLEvaluator(
            result_dir=self.args.result_dir,
            model_name="final_model_evaluation",
            target_names=self.args.target_columns,
        )
        metrics = evaluator.evaluate(
            model=self.best_model,
            train_data=self.outer_train_data,
            val_data=None,
            test_data=self.outer_test_data,
            scaler_y=self.scaler_y,
            save_predictions=False,
        )
        final_metrics = {
            key: convert_numpy_types(value)
            for key, value in metrics.items()
            if "train" not in key
        }
        test_set_metrics: Dict[str, Any] = {}
        for test_set_name, test_set_data in self.test_set_data.items():
            y_pred = self.best_model.predict(test_set_data["X"])
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)
            y_true = test_set_data["y"].reshape(-1, 1) if test_set_data["y"].ndim == 1 else test_set_data["y"]
            y_true_unscaled = self.scaler_y.inverse_transform(y_true)
            y_pred_unscaled = self.scaler_y.inverse_transform(y_pred)
            test_set_metrics[test_set_name] = _plain_regression_metrics(y_true_unscaled, y_pred_unscaled)
        if test_set_metrics:
            final_metrics["test_set_metrics"] = convert_numpy_types(test_set_metrics)
        Path(self.args.result_dir, "final_evaluation_metrics.json").write_text(
            json.dumps(final_metrics, indent=4),
            encoding="utf-8",
        )
        self._save_predictions()

        if self.args.run_shap_analysis:
            shap_result_dir = Path(self.args.result_dir) / "plots" / "SHAP_Analysis"
            shap_result_dir.mkdir(parents=True, exist_ok=True)
            analyzer = MLShapAnalyzer(
                model=self.best_model,
                feature_names=self.feature_names,
                target_names=self.args.target_columns,
                result_dir=str(shap_result_dir),
            )
            analyzer.analyze(self.outer_test_data["X"])

    def run(self) -> None:
        self._prepare_data()
        if (
            self.outer_train_data is None
            or self.outer_test_data is None
            or self.inner_train_data is None
            or self.inner_val_data is None
        ):
            raise ValueError("OOD outer/inner split state is not initialized")

        evaluate_after_train = self.args.evaluate_after_train
        saved_train = self.train_val_data
        saved_test = self.test_data
        self.args.evaluate_after_train = False
        self.train_val_data = self.inner_train_data
        self.test_data = self.inner_val_data
        try:
            if self.args.use_optuna:
                self._run_optuna()
            else:
                self._run_standard_training()
        finally:
            self.args.evaluate_after_train = evaluate_after_train
            self.train_val_data = saved_train
            self.test_data = saved_test

        self._prepare_final_outer_refit_data()
        self._retrain_best_model_on_outer_train()
        self.train_val_data = self.outer_train_data
        self.test_data = self.outer_test_data
        if evaluate_after_train and self.best_model is not None:
            self._evaluate_best_model()

        _rewrite_prediction_labels(self.args.result_dir)
        _write_ood_manifest(
            result_dir=self.args.result_dir,
            train_feature_file=self._train_feature_file,
            test_feature_file=self._test_feature_file,
            target_column=_resolve_target_columns(self.args)[0],
            test_set_feature_files=self._test_set_feature_files,
        )


class OODNNTrainingPipeline(NNTrainingPipeline):
    def __init__(
        self,
        args: Any,
        train_feature_file: str,
        test_feature_file: str,
        test_set_feature_files: Dict[str, str] | None = None,
    ):
        super().__init__(args)
        self._train_feature_file = train_feature_file
        self._test_feature_file = test_feature_file
        self._test_set_feature_files = dict(test_set_feature_files or {})
        self.raw_outer_train_data: Dict[str, Any] | None = None
        self.raw_outer_test_data: Dict[str, Any] | None = None
        self.raw_test_set_data: Dict[str, Dict[str, Any]] = {}
        self.inner_train_data: Dict[str, Any] | None = None
        self.outer_train_data: Dict[str, Any] | None = None
        self.outer_test_data: Dict[str, Any] | None = None
        self.test_set_data: Dict[str, Dict[str, Any]] = {}
        self.inner_val_data: Dict[str, Any] | None = None
        self.final_refit_train_data: Dict[str, Any] | None = None
        self.final_refit_val_data: Dict[str, Any] | None = None
        self._suppress_intermediate_evaluation = False

    def _prepare_data(self) -> None:
        target_columns = _resolve_target_columns(self.args)
        raw_outer_train_data, raw_outer_test_data, self.feature_names = load_ood_feature_split(
            train_feature_file=self._train_feature_file,
            test_feature_file=self._test_feature_file,
            target_columns=target_columns,
            args=self.args,
            pipeline_kind="nn",
        )
        raw_inner_train_data, raw_inner_val_data = _split_inner_train_val(
            raw_outer_train_data,
            random_state=self.args.random_state,
        )
        self.raw_outer_train_data = _clone_split_data(raw_outer_train_data)
        self.raw_outer_test_data = _clone_split_data(raw_outer_test_data)
        self.raw_test_set_data = {
            name: _clone_split_data(data)
            for name, data in load_ood_test_set_feature_splits(
                self._test_set_feature_files,
                target_columns=target_columns,
                selected_cols=self.feature_names,
            ).items()
        }

        self.scaler_X, self.scaler_y = _fit_ood_scalers(raw_inner_train_data)
        self.inner_train_data = _transform_split_data(raw_inner_train_data, self.scaler_X, self.scaler_y)
        self.inner_val_data = _transform_split_data(raw_inner_val_data, self.scaler_X, self.scaler_y)
        self.outer_train_data = _transform_split_data(raw_outer_train_data, self.scaler_X, self.scaler_y)
        self.outer_test_data = _transform_split_data(raw_outer_test_data, self.scaler_X, self.scaler_y)
        self.test_set_data = {
            name: _transform_split_data(data, self.scaler_X, self.scaler_y)
            for name, data in self.raw_test_set_data.items()
        }
        self.train_val_data = _clone_split_data(self.inner_train_data)
        self.test_data = _clone_split_data(self.outer_test_data)

        self.eval_train_X = self.inner_train_data["X"]
        self.eval_train_y = self.inner_train_data["y"]
        self.eval_val_X = self.inner_val_data["X"]
        self.eval_val_y = self.inner_val_data["y"]
        if "ids" in self.inner_train_data:
            self.eval_train_ids = np.asarray(self.inner_train_data["ids"])
        if "ids" in self.inner_val_data:
            self.eval_val_ids = np.asarray(self.inner_val_data["ids"])

        _save_scalers(self.args.result_dir, self.scaler_X, self.scaler_y)

    def _load_selected_model_config(self) -> Dict[str, Any] | None:
        if not self.best_model_path or not Path(self.best_model_path).exists():
            return None
        loaded_object = torch.load(self.best_model_path, map_location="cpu", weights_only=False)
        return loaded_object.get("config")

    def _prepare_final_outer_refit_data(self) -> None:
        if self.raw_outer_train_data is None or self.raw_outer_test_data is None:
            raise ValueError("Raw OOD outer split is not initialized")

        raw_final_train_data, raw_final_val_data = _split_inner_train_val(
            self.raw_outer_train_data,
            random_state=self.args.random_state,
        )

        self.scaler_X, self.scaler_y = _fit_ood_scalers(self.raw_outer_train_data)
        self.final_refit_train_data = _transform_split_data(raw_final_train_data, self.scaler_X, self.scaler_y)
        self.final_refit_val_data = _transform_split_data(raw_final_val_data, self.scaler_X, self.scaler_y)
        self.outer_train_data = _transform_split_data(self.raw_outer_train_data, self.scaler_X, self.scaler_y)
        self.outer_test_data = _transform_split_data(self.raw_outer_test_data, self.scaler_X, self.scaler_y)
        self.test_set_data = {
            name: _transform_split_data(data, self.scaler_X, self.scaler_y)
            for name, data in self.raw_test_set_data.items()
        }

        self.train_val_data = _clone_split_data(self.outer_train_data)
        self.test_data = _clone_split_data(self.outer_test_data)
        self.eval_train_X = self.final_refit_train_data["X"]
        self.eval_train_y = self.final_refit_train_data["y"]
        self.eval_val_X = self.final_refit_val_data["X"]
        self.eval_val_y = self.final_refit_val_data["y"]
        if "ids" in self.final_refit_train_data:
            self.eval_train_ids = np.asarray(self.final_refit_train_data["ids"])
        elif hasattr(self, "eval_train_ids"):
            delattr(self, "eval_train_ids")
        if "ids" in self.final_refit_val_data:
            self.eval_val_ids = np.asarray(self.final_refit_val_data["ids"])
        elif hasattr(self, "eval_val_ids"):
            delattr(self, "eval_val_ids")

        _save_scalers(self.args.result_dir, self.scaler_X, self.scaler_y)

    def _retrain_best_model_on_outer_train(self) -> None:
        if self.final_refit_train_data is None or self.final_refit_val_data is None:
            raise ValueError("Final OOD NN refit split is not initialized")

        selected_model_config = self._load_selected_model_config()
        model, model_config = self._create_model(config=selected_model_config)

        final_trainer = TrainerFactory.create_trainer(
            model_type=self.args.model_type,
            model=model,
            result_dir=self.args.result_dir,
            model_name="best_model",
            target_names=self.args.target_columns,
            train_data=self.final_refit_train_data,
            val_data=self.final_refit_val_data,
            training_params=self._get_training_params(),
        )
        history, _, best_state_dict = final_trainer.train(num_epochs=self.args.epochs)

        if history:
            loss_dir = Path(self.args.result_dir) / "loss"
            loss_dir.mkdir(parents=True, exist_ok=True)

            history_plot_path = loss_dir / "training_curves.png"
            BaseEvaluator.plot_training_curves(
                history=history,
                save_path=str(history_plot_path),
                title="Training Curves",
            )

            history_df = pd.DataFrame(history)
            history_df.to_csv(loss_dir / "training_history.csv", index=False)

        if best_state_dict:
            self._save_best_model(best_state_dict, model_config)
        else:
            raise ValueError("Final OOD NN refit did not produce a best model state")

    def _evaluate_best_model(self) -> None:
        if self._suppress_intermediate_evaluation:
            return
        if self.outer_train_data is None or self.outer_test_data is None:
            raise ValueError("Final OOD NN evaluation data is not initialized")
        if not self.best_model_path or not Path(self.best_model_path).exists():
            raise ValueError("Best OOD NN checkpoint is missing")

        loaded_object = torch.load(
            self.best_model_path,
            map_location=self.args.device or "cpu",
            weights_only=False,
        )
        model_config = loaded_object["config"]
        model_state_dict = loaded_object["model_state_dict"]

        eval_model, _ = self._create_model(config=model_config)
        self._load_best_model_weights(eval_model, model_state_dict)

        eval_train_data: Dict[str, Any] = {
            "X": self.outer_train_data["X"],
            "y": self.outer_train_data["y"],
        }
        if "ids" in self.outer_train_data:
            eval_train_data["ids"] = self.outer_train_data["ids"]

        evaluator = EvaluatorFactory.create_evaluator(
            "alloys",
            result_dir=self.args.result_dir,
            model_name="best_model_evaluation",
            target_names=self.args.target_columns,
            target_scaler=self.scaler_y,
        )
        assert isinstance(evaluator, AlloysEvaluator)
        combined_summary = evaluator.evaluate_model(
            model=eval_model,
            train_data=eval_train_data,
            test_data=self.outer_test_data,
            val_data=None,
            save_prefix="best_model_",
            feature_names=self.feature_names,
        )
        final_metrics_path = Path(self.args.result_dir) / "final_evaluation_metrics.json"
        final_metrics_payload: Dict[str, Any] = {"combined_evaluation_summary": convert_numpy_types(combined_summary)}
        if self.test_set_data:
            test_set_metrics: Dict[str, Any] = {}
            for test_set_name, test_set_data in self.test_set_data.items():
                y_true, y_pred = evaluator._predict_and_unscale(eval_model, test_set_data, data_is_scaled=True)
                if y_true is None or y_pred is None:
                    continue
                test_set_metrics[test_set_name] = _plain_regression_metrics(y_true, y_pred)
                _save_test_set_predictions(
                    result_dir=self.args.result_dir,
                    test_set_name=test_set_name,
                    data=test_set_data,
                    y_true=y_true,
                    y_pred=y_pred,
                    target_columns=self.args.target_columns,
                )
            if test_set_metrics:
                final_metrics_payload["test_set_metrics"] = convert_numpy_types(test_set_metrics)
        final_metrics_path.write_text(json.dumps(final_metrics_payload, indent=4), encoding="utf-8")

        model_structure_path = Path(self.args.result_dir) / "model_structure.txt"
        model_structure_path.write_text(str(eval_model), encoding="utf-8")

    def run(self) -> None:
        self._prepare_data()
        if (
            self.inner_train_data is None
            or self.outer_test_data is None
            or self.inner_val_data is None
        ):
            raise ValueError("OOD outer/inner split state is not initialized")

        evaluate_after_train = self.args.evaluate_after_train
        saved_train = self.train_val_data
        saved_test = self.test_data
        self.args.evaluate_after_train = False
        self._suppress_intermediate_evaluation = True
        self.train_val_data = self.inner_train_data
        self.test_data = self.inner_val_data
        try:
            if self.args.use_optuna:
                self._run_optuna()
            else:
                self._run_standard_training()
        finally:
            self._suppress_intermediate_evaluation = False
            self.args.evaluate_after_train = evaluate_after_train
            self.train_val_data = saved_train
            self.test_data = saved_test

        self._prepare_final_outer_refit_data()
        self._retrain_best_model_on_outer_train()
        self.test_data = self.outer_test_data
        if evaluate_after_train:
            self._evaluate_best_model()

        _rewrite_prediction_labels(self.args.result_dir)
        _write_ood_manifest(
            result_dir=self.args.result_dir,
            train_feature_file=self._train_feature_file,
            test_feature_file=self._test_feature_file,
            target_column=_resolve_target_columns(self.args)[0],
            test_set_feature_files=self._test_set_feature_files,
        )


def build_ml_args(base_args: Any) -> SimpleNamespace:
    target_columns = _resolve_target_columns(base_args)
    return SimpleNamespace(
        data_file=base_args.data_file,
        result_dir=base_args.result_dir,
        model_type=None,
        target_column=target_columns[0],
        target_columns=target_columns,
        processing_cols=base_args.processing_cols or [],
        use_composition_feature=base_args.use_composition_feature,
        use_temperature=base_args.use_temperature,
        other_features_name=base_args.processing_cols if base_args.processing_cols else None,
        test_size=base_args.test_size,
        random_state=base_args.random_state,
        evaluate_after_train=base_args.evaluate_after_train,
        run_shap_analysis=base_args.run_shap_analysis,
        cross_validate=base_args.cross_validate,
        num_folds=base_args.num_folds,
        use_optuna=base_args.use_optuna,
        n_trials=base_args.n_trials,
        study_name="ml_hyperparameter_optimization",
        mlp_max_iter=getattr(base_args, "mlp_max_iter", 300),
        optuna_n_trials=base_args.n_trials,
        optuna_timeout=None,
    )


def build_nn_args(base_args: Any) -> SimpleNamespace:
    target_columns = _resolve_target_columns(base_args)
    return SimpleNamespace(
        data_file=base_args.data_file,
        result_dir=base_args.result_dir,
        model_type="nn",
        emb_hidden_dim=256,
        feature1_hidden_dim=256,
        feature2_hidden_dim=256,
        other_features_hidden_dim=0,
        hidden_dims=[256, 128],
        dropout_rate=0.2,
        epochs=base_args.epochs,
        batch_size=base_args.batch_size,
        learning_rate=0.001,
        weight_decay=1e-4,
        patience=base_args.patience,
        use_lr_scheduler=False,
        lr_scheduler_patience=10,
        lr_scheduler_factor=0.5,
        target_column=target_columns[0],
        target_columns=target_columns,
        use_process_embedding=base_args.use_process_embedding,
        use_joint_composition_process_embedding=False,
        use_element_embedding=base_args.use_element_embedding,
        use_composition_feature=base_args.use_composition_feature,
        use_feature1=False,
        use_feature2=False,
        other_features_name=base_args.processing_cols if base_args.processing_cols else None,
        use_temperature=base_args.use_temperature,
        test_size=base_args.test_size,
        random_state=base_args.random_state,
        use_optuna=base_args.use_optuna,
        n_trials=base_args.n_trials,
        study_name="nn_hyperparameter_optimization",
        evaluate_after_train=base_args.evaluate_after_train,
        cross_validate=base_args.cross_validate,
        num_folds=base_args.num_folds,
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
        use_multi_gpu=False,
    )


def run_ood_training(
    base_args: Any,
    train_feature_file: str,
    test_feature_file: str,
    test_set_feature_files: Dict[str, str] | None = None,
) -> None:
    if getattr(base_args, "use_nn", False):
        nn_args = build_nn_args(base_args)
        OODNNTrainingPipeline(nn_args, train_feature_file, test_feature_file, test_set_feature_files).run()
        return

    if not getattr(base_args, "models", None):
        raise ValueError("Traditional OOD training requires at least one model")

    for model_name in base_args.models:
        model_result_dir = str(Path(base_args.result_dir) / "model_comparison" / f"{model_name}_results")
        ml_args = build_ml_args(base_args)
        ml_args.model_type = model_name
        ml_args.result_dir = model_result_dir
        OODMLTrainingPipeline(ml_args, train_feature_file, test_feature_file, test_set_feature_files).run()
