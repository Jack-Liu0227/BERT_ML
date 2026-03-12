from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.models.ml_train_eval_pipeline.pipeline import MLTrainingPipeline
from src.models.nn_train_eval_pipeline.pipeline import TrainingPipeline as NNTrainingPipeline


def _load_feature_dataframe(feature_file: str) -> pd.DataFrame:
    if not Path(feature_file).exists():
        raise FileNotFoundError(f"Feature file not found: {feature_file}")
    return pd.read_csv(feature_file)


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
        raise ValueError("No ML features selected for extrapolation training")
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
        raise ValueError("No NN features selected for extrapolation training")
    return selected_cols


def load_extrapolation_feature_split(
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

    missing_in_test = [col for col in selected_cols if col not in test_df.columns]
    if missing_in_test:
        raise ValueError(f"Test feature file is missing selected columns: {missing_in_test}")

    train_val_data: Dict[str, np.ndarray] = {
        "X": train_df[selected_cols].fillna(0).values.astype("float32"),
        "y": train_df[target_columns].values.astype("float32"),
        "feature_names": selected_cols,
    }
    test_data: Dict[str, np.ndarray] = {
        "X": test_df[selected_cols].fillna(0).values.astype("float32"),
        "y": test_df[target_columns].values.astype("float32"),
        "feature_names": selected_cols,
    }

    if "ID" in train_df.columns:
        train_val_data["ids"] = train_df["ID"].values
    if "ID" in test_df.columns:
        test_data["ids"] = test_df["ID"].values

    return train_val_data, test_data, selected_cols


def _rewrite_dataset_labels(result_dir: str) -> None:
    result_path = Path(result_dir)
    for csv_path in result_path.rglob("*.csv"):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if "Dataset" not in df.columns:
            continue
        updated = df["Dataset"].replace({"Test": "ExtrapolationTest"})
        if updated.equals(df["Dataset"]):
            continue
        df["Dataset"] = updated
        df.to_csv(csv_path, index=False)


def _write_extrapolation_manifest(
    result_dir: str,
    train_feature_file: str,
    test_feature_file: str,
    target_column: str,
) -> None:
    manifest = {
        "train_feature_file": train_feature_file,
        "test_feature_file": test_feature_file,
        "target_column": target_column,
        "test_dataset_label": "ExtrapolationTest",
    }
    Path(result_dir, "extrapolation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


class ExtrapolationMLTrainingPipeline(MLTrainingPipeline):
    def __init__(self, args: Any, train_feature_file: str, test_feature_file: str):
        super().__init__(args)
        self._train_feature_file = train_feature_file
        self._test_feature_file = test_feature_file

    def _prepare_data(self) -> None:
        self.train_val_data, self.test_data, self.feature_names = load_extrapolation_feature_split(
            train_feature_file=self._train_feature_file,
            test_feature_file=self._test_feature_file,
            target_columns=self.args.target_columns,
            args=self.args,
            pipeline_kind="ml",
        )

        self.scaler_X = StandardScaler().fit(self.train_val_data["X"])
        self.train_val_data["X"] = self.scaler_X.transform(self.train_val_data["X"]).astype(np.float32)
        self.test_data["X"] = self.scaler_X.transform(self.test_data["X"]).astype(np.float32)

        self.scaler_y = StandardScaler().fit(self.train_val_data["y"])
        self.train_val_data["y"] = self.scaler_y.transform(self.train_val_data["y"]).astype(np.float32)
        self.test_data["y"] = self.scaler_y.transform(self.test_data["y"]).astype(np.float32)

        Path(self.args.result_dir).mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler_X, Path(self.args.result_dir, "scaler_X.pkl"))
        joblib.dump(self.scaler_y, Path(self.args.result_dir, "scaler_y.pkl"))

    def run(self) -> None:
        super().run()
        _rewrite_dataset_labels(self.args.result_dir)
        _write_extrapolation_manifest(
            result_dir=self.args.result_dir,
            train_feature_file=self._train_feature_file,
            test_feature_file=self._test_feature_file,
            target_column=self.args.target_columns[0],
        )


class ExtrapolationNNTrainingPipeline(NNTrainingPipeline):
    def __init__(self, args: Any, train_feature_file: str, test_feature_file: str):
        super().__init__(args)
        self._train_feature_file = train_feature_file
        self._test_feature_file = test_feature_file

    def _prepare_data(self) -> None:
        self.train_val_data, self.test_data, self.feature_names = load_extrapolation_feature_split(
            train_feature_file=self._train_feature_file,
            test_feature_file=self._test_feature_file,
            target_columns=self.args.target_columns,
            args=self.args,
            pipeline_kind="nn",
        )

        self.eval_train_X, self.eval_val_X, self.eval_train_y, self.eval_val_y = train_test_split(
            self.train_val_data["X"],
            self.train_val_data["y"],
            test_size=0.25,
            random_state=self.args.random_state,
        )

        if "ids" in self.train_val_data:
            self.eval_train_ids, self.eval_val_ids = train_test_split(
                self.train_val_data["ids"],
                test_size=0.25,
                random_state=self.args.random_state,
            )

        self.scaler_X = StandardScaler().fit(self.train_val_data["X"])
        self.train_val_data["X"] = self.scaler_X.transform(self.train_val_data["X"]).astype(np.float32)
        self.test_data["X"] = self.scaler_X.transform(self.test_data["X"]).astype(np.float32)
        self.eval_train_X = self.scaler_X.transform(self.eval_train_X).astype(np.float32)
        self.eval_val_X = self.scaler_X.transform(self.eval_val_X).astype(np.float32)

        self.scaler_y = StandardScaler().fit(self.train_val_data["y"])
        self.train_val_data["y"] = self.scaler_y.transform(self.train_val_data["y"]).astype(np.float32)
        self.test_data["y"] = self.scaler_y.transform(self.test_data["y"]).astype(np.float32)
        self.eval_train_y = self.scaler_y.transform(self.eval_train_y).astype(np.float32)
        self.eval_val_y = self.scaler_y.transform(self.eval_val_y).astype(np.float32)

        Path(self.args.result_dir).mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler_X, Path(self.args.result_dir, "scaler_X.pkl"))
        joblib.dump(self.scaler_y, Path(self.args.result_dir, "scaler_y.pkl"))

    def run(self) -> None:
        super().run()
        _rewrite_dataset_labels(self.args.result_dir)
        _write_extrapolation_manifest(
            result_dir=self.args.result_dir,
            train_feature_file=self._train_feature_file,
            test_feature_file=self._test_feature_file,
            target_column=self.args.target_columns[0],
        )


def build_ml_args(base_args: Any) -> SimpleNamespace:
    return SimpleNamespace(
        data_file=base_args.data_file,
        result_dir=base_args.result_dir,
        model_type=None,
        target_columns=base_args.target_columns,
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
        target_columns=base_args.target_columns,
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


def run_extrapolation_training(
    base_args: Any,
    train_feature_file: str,
    test_feature_file: str,
) -> None:
    if getattr(base_args, "use_nn", False):
        nn_args = build_nn_args(base_args)
        ExtrapolationNNTrainingPipeline(nn_args, train_feature_file, test_feature_file).run()
        return

    if not getattr(base_args, "models", None):
        raise ValueError("Traditional extrapolation training requires at least one model")

    for model_name in base_args.models:
        model_result_dir = str(Path(base_args.result_dir) / "model_comparison" / f"{model_name}_results")
        ml_args = build_ml_args(base_args)
        ml_args.model_type = model_name
        ml_args.result_dir = model_result_dir
        ExtrapolationMLTrainingPipeline(ml_args, train_feature_file, test_feature_file).run()
