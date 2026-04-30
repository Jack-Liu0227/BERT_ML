from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


CSV_ENCODINGS = ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"]
TEST_LABELS = {"test", "testing", "extrapolationtest"}
CASE_LEVEL_TEST_LABELS = {
    "test",
    "testing",
    "oodtest",
    "ood",
    "oodtesting",
    "extrapolationtest",
    "extrapolation_test",
    "extrapolation test",
}
LOCO_OUTER_PREDICTION_PATTERNS = [
    "predictions/all_predictions.csv",
    "predictions/best_model_all_predictions.csv",
    "predictions/test_predictions.csv",
    "predictions/best_model_test_predictions.csv",
]


def read_prediction_csv(file_path: Path) -> Optional[pd.DataFrame]:
    header_columns: Optional[List[str]] = None
    header_encoding: Optional[str] = None
    last_error: Optional[Exception] = None

    for encoding in CSV_ENCODINGS:
        try:
            header_columns = pd.read_csv(file_path, nrows=0, encoding=encoding).columns.tolist()
            header_encoding = encoding
            break
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
            last_error = exc

    if header_columns is None:
        print(f"[WARN] Failed to read header from {file_path}: {last_error}")
        return None

    usecols = [
        column
        for column in header_columns
        if column in {"Dataset", "set", "ID", "id"}
        or column.endswith("_Actual")
        or column.endswith("_Predicted")
        or column.startswith("True_")
        or column.startswith("Pred_")
    ]
    if not usecols:
        usecols = header_columns

    ordered_encodings = [header_encoding] + [encoding for encoding in CSV_ENCODINGS if encoding != header_encoding]
    for encoding in ordered_encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding, usecols=usecols, low_memory=False)
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
            last_error = exc

        try:
            return pd.read_csv(
                file_path,
                encoding=encoding,
                usecols=usecols,
                engine="python",
                on_bad_lines="skip",
            )
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
            last_error = exc

    print(f"[WARN] Failed to parse {file_path}: {last_error}")
    return None


def collect_optuna_test_metrics(optuna_trials_dir: Path) -> pd.DataFrame:
    if not optuna_trials_dir.exists():
        return pd.DataFrame()

    rows: List[Dict] = []
    trial_dirs = sorted(
        [path for path in optuna_trials_dir.rglob("trial_*") if path.is_dir() and path.name.startswith("trial_")],
        key=lambda path: (
            str(path.parent),
            int(path.name.split("_")[1]) if path.name.split("_")[1].isdigit() else 0,
        ),
    )

    for trial_dir in trial_dirs:
        fold_dirs = sorted(
            [path for path in trial_dir.iterdir() if path.is_dir() and path.name.startswith("fold_")],
            key=lambda path: int(path.name.split("_")[1]),
        )
        for fold_dir in fold_dirs:
            prediction_file = fold_dir / "all_predictions.csv"
            if not prediction_file.exists():
                continue

            df = read_prediction_csv(prediction_file)
            if df is None or "Dataset" not in df.columns:
                continue

            dataset_labels = df["Dataset"].astype(str).str.lower()
            df_test = df[dataset_labels.isin(TEST_LABELS)].copy()
            if df_test.empty:
                continue

            for actual_col in [col for col in df_test.columns if col.endswith("_Actual")]:
                property_name = actual_col[:-7]
                pred_col = f"{property_name}_Predicted"
                if pred_col not in df_test.columns:
                    continue

                valid_df = df_test[[actual_col, pred_col]].dropna()
                if valid_df.empty:
                    continue

                y_true = valid_df[actual_col]
                y_pred = valid_df[pred_col]
                rows.append(
                    {
                        "trial_id": trial_dir.name,
                        "trial_num": int(trial_dir.name.split("_")[1]),
                        "fold": int(fold_dir.name.split("_")[1]),
                        "property": property_name,
                        "r2": float(r2_score(y_true, y_pred)),
                        "mae": float(mean_absolute_error(y_true, y_pred)),
                        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                        "predictions_file": str(prediction_file),
                    }
                )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def collect_optuna_test_metrics_from_dirs(optuna_trials_dirs: Sequence[Path]) -> pd.DataFrame:
    candidate_dirs = [Path(path) for path in optuna_trials_dirs if Path(path).exists()]
    if not candidate_dirs:
        return pd.DataFrame()
    if len(candidate_dirs) == 1:
        return collect_optuna_test_metrics(candidate_dirs[0])

    metrics_frames: List[pd.DataFrame] = []
    trial_num_offset = 0
    for index, optuna_dir in enumerate(candidate_dirs):
        metrics_df = collect_optuna_test_metrics(optuna_dir)
        if metrics_df.empty:
            continue

        metrics_df = metrics_df.copy()
        metrics_df["trial_id"] = metrics_df["trial_id"].astype(str).map(lambda value: f"group{index}_{value}")
        metrics_df["trial_num"] = metrics_df["trial_num"].astype(int) + trial_num_offset
        trial_num_offset = int(metrics_df["trial_num"].max()) + 1000
        metrics_frames.append(metrics_df)

    if not metrics_frames:
        return pd.DataFrame()
    return pd.concat(metrics_frames, ignore_index=True)


def build_trial_level_test_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()

    grouped = (
        metrics_df.groupby(["property", "trial_id", "trial_num"], as_index=False)
        .agg(
            trial_mean_test_r2=("r2", "mean"),
            trial_mean_test_mae=("mae", "mean"),
            trial_mean_test_rmse=("rmse", "mean"),
            trial_fold_count=("fold", "count"),
        )
        .sort_values(["property", "trial_num", "trial_id"], ascending=[True, True, True], na_position="last")
        .reset_index(drop=True)
    )
    return grouped


def _std_or_zero(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    std_value = numeric.std()
    if pd.isna(std_value):
        return 0.0
    return float(std_value)


def infer_model_dir_from_predictions_file(predictions_file: str | Path) -> Optional[str]:
    path = Path(str(predictions_file))
    try:
        return str(path.parents[4])
    except IndexError:
        return None


def _extract_outer_fold_index(path: Path) -> Optional[int]:
    for candidate in (path, *path.parents):
        name = str(candidate.name)
        if not name.startswith("fold_"):
            continue
        suffix = name.split("_", 1)[1]
        if suffix.isdigit():
            return int(suffix)
    return None


def build_trial_level_test_metrics_detailed(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()

    grouped = (
        metrics_df.groupby(["property", "trial_id", "trial_num"], as_index=False)
        .agg(
            trial_mean_test_r2=("r2", "mean"),
            trial_std_test_r2=("r2", _std_or_zero),
            trial_mean_test_mae=("mae", "mean"),
            trial_std_test_mae=("mae", _std_or_zero),
            trial_mean_test_rmse=("rmse", "mean"),
            trial_std_test_rmse=("rmse", _std_or_zero),
            trial_fold_count=("fold", "count"),
        )
        .sort_values(["property", "trial_num", "trial_id"], ascending=[True, True, True], na_position="last")
        .reset_index(drop=True)
    )
    return grouped


def _select_best_trial_row(trial_df: pd.DataFrame) -> Optional[pd.Series]:
    if trial_df.empty:
        return None

    ranked_df = trial_df.copy()
    ranked_df["_sort_mean_mae"] = pd.to_numeric(ranked_df["trial_mean_test_mae"], errors="coerce").fillna(np.inf)
    ranked_df["_sort_std_mae"] = pd.to_numeric(ranked_df["trial_std_test_mae"], errors="coerce").fillna(np.inf)
    ranked_df["_sort_mean_r2"] = pd.to_numeric(ranked_df["trial_mean_test_r2"], errors="coerce").fillna(-np.inf)
    ranked_df["_sort_trial_num"] = pd.to_numeric(ranked_df["trial_num"], errors="coerce").fillna(np.inf)
    ranked_df = ranked_df.sort_values(
        ["_sort_mean_mae", "_sort_std_mae", "_sort_mean_r2", "_sort_trial_num", "trial_id"],
        ascending=[True, True, False, True, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    selected = ranked_df.iloc[0].copy()
    for helper_col in ["_sort_mean_mae", "_sort_std_mae", "_sort_mean_r2", "_sort_trial_num"]:
        if helper_col in selected.index:
            selected = selected.drop(labels=[helper_col])
    return selected


def _select_inner_fold_predictions_for_trial(
    trial_metrics_df: pd.DataFrame,
    *,
    target_mean_mae: float,
) -> Optional[pd.Series]:
    if trial_metrics_df.empty:
        return None

    ranked_df = trial_metrics_df.copy()
    ranked_df["_distance_to_trial_mean_mae"] = (
        pd.to_numeric(ranked_df["mae"], errors="coerce") - float(target_mean_mae)
    ).abs()
    ranked_df = ranked_df.sort_values(
        ["_distance_to_trial_mean_mae", "mae", "r2", "fold", "predictions_file"],
        ascending=[True, True, False, True, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    selected = ranked_df.iloc[0].copy()
    if "_distance_to_trial_mean_mae" in selected.index:
        selected = selected.drop(labels=["_distance_to_trial_mean_mae"])
    return selected


def _clean_optional_scalar(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _normalize_dataset_values(dataset_series: pd.Series) -> pd.Series:
    return (
        dataset_series.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "", regex=False)
        .str.replace("_", "", regex=False)
    )


def _compute_prediction_metrics(
    df: pd.DataFrame,
    *,
    actual_col: str,
    pred_col: str,
) -> Optional[dict[str, float]]:
    valid_df = df[[actual_col, pred_col]].dropna()
    if valid_df.empty:
        return None

    y_true = valid_df[actual_col].to_numpy()
    y_pred = valid_df[pred_col].to_numpy()
    return {
        "test_r2": float(r2_score(y_true, y_pred)),
        "test_mae": float(mean_absolute_error(y_true, y_pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "test_row_count": int(len(valid_df)),
    }


def _collect_outer_prediction_candidates(model_dir: Path) -> list[Path]:
    if not model_dir.exists():
        return []

    candidates: list[Path] = []
    seen: set[Path] = set()
    for pattern in LOCO_OUTER_PREDICTION_PATTERNS:
        for candidate in model_dir.glob(pattern):
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(candidate)
    return candidates


def load_case_level_test_metrics(
    model_dir: Path,
    property_name: str,
) -> Optional[dict[str, object]]:
    for candidate_file in _collect_outer_prediction_candidates(model_dir):
        df = read_prediction_csv(candidate_file)
        if df is None:
            continue

        dataset_col, actual_col, pred_col = resolve_prediction_columns(df, property_name)
        if actual_col is None or pred_col is None:
            continue

        metric_df: Optional[pd.DataFrame] = None
        if dataset_col is not None:
            normalized_dataset = _normalize_dataset_values(df[dataset_col])
            normalized_labels = {
                label.replace(" ", "").replace("_", "").lower() for label in CASE_LEVEL_TEST_LABELS
            }
            metric_df = df.loc[normalized_dataset.isin(normalized_labels)].copy()
            if metric_df.empty and "test_predictions" in candidate_file.name.lower():
                metric_df = df.copy()
        else:
            metric_df = df.copy()

        if metric_df is None or metric_df.empty:
            continue

        metrics = _compute_prediction_metrics(metric_df, actual_col=actual_col, pred_col=pred_col)
        if metrics is None:
            continue

        return {
            "predictions_file": str(candidate_file),
            **metrics,
        }

    return None


def summarize_outer_fold_final_test_metrics(
    outer_model_dirs: Sequence[Path],
    property_name: str | None = None,
    *,
    include_selected_trial_details: bool = True,
) -> Dict[str, Dict[str, object]]:
    candidate_model_dirs = [Path(path) for path in outer_model_dirs if Path(path).exists()]
    if not candidate_model_dirs:
        return {}

    property_records: dict[str, list[dict[str, object]]] = {}
    for model_dir in sorted(
        candidate_model_dirs,
        key=lambda path: (
            np.inf if _extract_outer_fold_index(path) is None else _extract_outer_fold_index(path),
            str(path),
        ),
    ):
        optuna_trials_dir = model_dir / "predictions" / "optuna_trials"
        if not optuna_trials_dir.exists():
            continue

        metrics_df = collect_optuna_test_metrics(optuna_trials_dir)
        if metrics_df.empty:
            continue

        working_df = metrics_df.copy()
        if property_name is not None:
            working_df = working_df[working_df["property"].astype(str) == str(property_name)].copy()
        if working_df.empty:
            continue

        outer_fold_index = _extract_outer_fold_index(model_dir)
        for current_property, property_df in working_df.groupby("property", sort=True):
            best_trial = None
            representative_inner_row = None
            if include_selected_trial_details:
                trial_df = build_trial_level_test_metrics_detailed(property_df)
                if trial_df.empty:
                    continue

                best_trial = _select_best_trial_row(trial_df)
                if best_trial is None:
                    continue

                selected_trial_df = property_df[property_df["trial_id"].astype(str) == str(best_trial["trial_id"])].copy()
                representative_inner_row = _select_inner_fold_predictions_for_trial(
                    selected_trial_df,
                    target_mean_mae=float(best_trial["trial_mean_test_mae"]),
                )
            outer_test_metrics = load_case_level_test_metrics(model_dir, str(current_property))
            if outer_test_metrics is None:
                continue
            property_records.setdefault(str(current_property), []).append(
                {
                    "outer_fold_index": outer_fold_index,
                    "model_dir": str(model_dir),
                    "selected_trial_id": (
                        str(best_trial["trial_id"])
                        if best_trial is not None and pd.notna(best_trial.get("trial_id"))
                        else None
                    ),
                    "selected_trial_num": (
                        int(best_trial["trial_num"])
                        if best_trial is not None and pd.notna(best_trial.get("trial_num"))
                        else None
                    ),
                    "selected_trial_fold_count": (
                        int(best_trial["trial_fold_count"])
                        if best_trial is not None and pd.notna(best_trial.get("trial_fold_count"))
                        else None
                    ),
                    "selected_mean_test_r2": (
                        float(best_trial["trial_mean_test_r2"])
                        if best_trial is not None and pd.notna(best_trial.get("trial_mean_test_r2"))
                        else None
                    ),
                    "selected_std_test_r2": (
                        float(best_trial["trial_std_test_r2"])
                        if best_trial is not None and pd.notna(best_trial.get("trial_std_test_r2"))
                        else None
                    ),
                    "selected_mean_test_mae": (
                        float(best_trial["trial_mean_test_mae"])
                        if best_trial is not None and pd.notna(best_trial.get("trial_mean_test_mae"))
                        else None
                    ),
                    "selected_std_test_mae": (
                        float(best_trial["trial_std_test_mae"])
                        if best_trial is not None and pd.notna(best_trial.get("trial_std_test_mae"))
                        else None
                    ),
                    "selected_mean_test_rmse": (
                        float(best_trial["trial_mean_test_rmse"])
                        if best_trial is not None and pd.notna(best_trial.get("trial_mean_test_rmse"))
                        else None
                    ),
                    "selected_std_test_rmse": (
                        float(best_trial["trial_std_test_rmse"])
                        if best_trial is not None and pd.notna(best_trial.get("trial_std_test_rmse"))
                        else None
                    ),
                    "selected_inner_predictions_file": (
                        str(representative_inner_row["predictions_file"])
                        if representative_inner_row is not None
                        else None
                    ),
                    "outer_predictions_file": outer_test_metrics["predictions_file"],
                    "outer_test_r2": float(outer_test_metrics["test_r2"]),
                    "outer_test_mae": float(outer_test_metrics["test_mae"]),
                    "outer_test_rmse": float(outer_test_metrics["test_rmse"]),
                    "outer_test_row_count": int(outer_test_metrics["test_row_count"]),
                }
            )

    summaries: Dict[str, Dict[str, object]] = {}
    for current_property, outer_rows in sorted(property_records.items()):
        outer_df = pd.DataFrame(outer_rows)
        if outer_df.empty:
            continue

        outer_test_r2 = pd.to_numeric(outer_df["outer_test_r2"], errors="coerce")
        outer_test_mae = pd.to_numeric(outer_df["outer_test_mae"], errors="coerce")
        outer_test_rmse = pd.to_numeric(outer_df["outer_test_rmse"], errors="coerce")
        valid_outer_mask = outer_test_r2.notna() & outer_test_mae.notna() & outer_test_rmse.notna()
        outer_df = outer_df.loc[valid_outer_mask].reset_index(drop=True)
        if outer_df.empty:
            continue

        summary_test_r2 = float(pd.to_numeric(outer_df["outer_test_r2"], errors="coerce").mean())
        summary_test_mae = float(pd.to_numeric(outer_df["outer_test_mae"], errors="coerce").mean())
        summary_test_rmse = float(pd.to_numeric(outer_df["outer_test_rmse"], errors="coerce").mean())

        representative_df = outer_df.assign(
            _distance_to_summary_test_mae=(
                pd.to_numeric(outer_df["outer_test_mae"], errors="coerce") - summary_test_mae
            ).abs(),
            _sort_outer_fold=pd.to_numeric(outer_df["outer_fold_index"], errors="coerce").fillna(np.inf),
        ).sort_values(
            ["_distance_to_summary_test_mae", "outer_test_mae", "outer_test_r2", "_sort_outer_fold"],
            ascending=[True, True, False, True],
            na_position="last",
            kind="mergesort",
        )
        representative_row = representative_df.iloc[0]

        representative_fold = representative_row["outer_fold_index"]
        if pd.notna(representative_fold):
            representative_fold = int(representative_fold)
        else:
            representative_fold = pd.NA

        loco_outer_fold_best_details: list[dict[str, object]] = []
        ordered_outer_df = outer_df.assign(
            _sort_outer_fold=pd.to_numeric(outer_df["outer_fold_index"], errors="coerce").fillna(np.inf),
        ).sort_values(
            ["_sort_outer_fold", "outer_test_mae", "outer_test_r2", "selected_trial_id"],
            ascending=[True, True, False, True],
            na_position="last",
            kind="mergesort",
        )
        for _, outer_row in ordered_outer_df.iterrows():
            loco_outer_fold_best_details.append(
                {
                    "outer_fold_index": _clean_optional_scalar(outer_row.get("outer_fold_index")),
                    "model_dir": _clean_optional_scalar(outer_row.get("model_dir")),
                    "selected_trial_id": _clean_optional_scalar(outer_row.get("selected_trial_id")),
                    "selected_trial_num": _clean_optional_scalar(outer_row.get("selected_trial_num")),
                    "selected_trial_fold_count": _clean_optional_scalar(outer_row.get("selected_trial_fold_count")),
                    "selected_mean_test_r2": _clean_optional_scalar(outer_row.get("selected_mean_test_r2")),
                    "selected_std_test_r2": _clean_optional_scalar(outer_row.get("selected_std_test_r2")),
                    "selected_mean_test_mae": _clean_optional_scalar(outer_row.get("selected_mean_test_mae")),
                    "selected_std_test_mae": _clean_optional_scalar(outer_row.get("selected_std_test_mae")),
                    "selected_mean_test_rmse": _clean_optional_scalar(outer_row.get("selected_mean_test_rmse")),
                    "selected_std_test_rmse": _clean_optional_scalar(outer_row.get("selected_std_test_rmse")),
                    "selected_inner_predictions_file": _clean_optional_scalar(
                        outer_row.get("selected_inner_predictions_file")
                    ),
                    "outer_predictions_file": _clean_optional_scalar(outer_row.get("outer_predictions_file")),
                    "outer_test_r2": _clean_optional_scalar(outer_row.get("outer_test_r2")),
                    "outer_test_mae": _clean_optional_scalar(outer_row.get("outer_test_mae")),
                    "outer_test_rmse": _clean_optional_scalar(outer_row.get("outer_test_rmse")),
                    "outer_test_row_count": _clean_optional_scalar(outer_row.get("outer_test_row_count")),
                }
            )

        summaries[str(current_property)] = {
            "trial_count": int(len(outer_df)),
            "fold_count": int(len(outer_df)),
            "summary_test_r2": summary_test_r2,
            "summary_test_r2_std": _std_or_zero(outer_df["outer_test_r2"]),
            "summary_test_mae": summary_test_mae,
            "summary_test_mae_std": _std_or_zero(outer_df["outer_test_mae"]),
            "summary_test_rmse": summary_test_rmse,
            "summary_test_rmse_std": _std_or_zero(outer_df["outer_test_rmse"]),
            "representative_selection_mode": "closest_summary_test_mae_outer_fold_oodtest",
            "representative_trial_id": str(representative_row["selected_trial_id"]),
            "representative_fold": representative_fold,
            "representative_test_r2": float(representative_row["outer_test_r2"]),
            "representative_test_mae": float(representative_row["outer_test_mae"]),
            "representative_test_rmse": float(representative_row["outer_test_rmse"]),
            "representative_predictions_file": representative_row["outer_predictions_file"],
            "representative_model_dir": str(representative_row["model_dir"]),
            "loco_outer_fold_best_details": loco_outer_fold_best_details,
        }

    return summaries


def summarize_loco_outer_fold_best_trials(
    outer_model_dirs: Sequence[Path],
    property_name: str | None = None,
) -> Dict[str, Dict[str, object]]:
    return summarize_outer_fold_final_test_metrics(
        outer_model_dirs,
        property_name=property_name,
        include_selected_trial_details=True,
    )


def summarize_optuna_model_trials_from_metrics(
    metrics_df: pd.DataFrame,
    property_name: str | None = None,
) -> Dict[str, Dict[str, object]]:
    if metrics_df.empty:
        return {}

    working_df = metrics_df.copy()
    if property_name is not None:
        working_df = working_df[working_df["property"].astype(str) == str(property_name)].copy()
    if working_df.empty:
        return {}

    summaries: Dict[str, Dict[str, object]] = {}
    for current_property, property_df in working_df.groupby("property", sort=True):
        trial_df = build_trial_level_test_metrics(property_df)
        if trial_df.empty:
            continue

        summary_test_r2 = float(trial_df["trial_mean_test_r2"].mean())
        summary_test_mae = float(trial_df["trial_mean_test_mae"].mean())
        summary_test_rmse = float(trial_df["trial_mean_test_rmse"].mean())

        representative_df = property_df.assign(
            _distance_to_summary_test_r2=(pd.to_numeric(property_df["r2"], errors="coerce") - summary_test_r2).abs()
        ).sort_values(
            ["_distance_to_summary_test_r2", "r2", "trial_num", "fold"],
            ascending=[True, False, True, True],
            na_position="last",
        )
        representative_row = representative_df.iloc[0]
        representative_predictions_file = str(representative_row["predictions_file"])
        representative_model_dir = infer_model_dir_from_predictions_file(representative_predictions_file)

        summaries[str(current_property)] = {
            "trial_count": int(trial_df["trial_id"].nunique()),
            "fold_count": int(len(property_df)),
            "summary_test_r2": summary_test_r2,
            "summary_test_r2_std": _std_or_zero(trial_df["trial_mean_test_r2"]),
            "summary_test_mae": summary_test_mae,
            "summary_test_mae_std": _std_or_zero(trial_df["trial_mean_test_mae"]),
            "summary_test_rmse": summary_test_rmse,
            "summary_test_rmse_std": _std_or_zero(trial_df["trial_mean_test_rmse"]),
            "representative_selection_mode": "closest_summary_test_r2_fold",
            "representative_trial_id": str(representative_row["trial_id"]),
            "representative_fold": int(representative_row["fold"]),
            "representative_test_r2": float(representative_row["r2"]),
            "representative_test_mae": float(representative_row["mae"]),
            "representative_test_rmse": float(representative_row["rmse"]),
            "representative_predictions_file": representative_predictions_file,
            "representative_model_dir": representative_model_dir,
        }

    return summaries


def summarize_optuna_model_trials(
    optuna_trials_dir: Path,
    property_name: str | None = None,
) -> Dict[str, Dict[str, object]]:
    metrics_df = collect_optuna_test_metrics(optuna_trials_dir)
    return summarize_optuna_model_trials_from_metrics(metrics_df, property_name=property_name)


def summarize_optuna_model_trials_from_dirs(
    optuna_trials_dirs: Sequence[Path],
    property_name: str | None = None,
) -> Dict[str, Dict[str, object]]:
    metrics_df = collect_optuna_test_metrics_from_dirs(optuna_trials_dirs)
    return summarize_optuna_model_trials_from_metrics(metrics_df, property_name=property_name)


def _property_tokens(property_name: str) -> List[str]:
    raw = str(property_name)
    tokens: List[str] = []

    def add(token: str) -> None:
        if token and token not in tokens:
            tokens.append(token)

    add(raw)
    add(raw.replace("(", "").replace(")", "").replace("/", "").replace("\\", "").replace(" ", ""))
    add(
        raw.replace("%", "percent")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "")
        .replace("\\", "")
        .replace(" ", "")
    )
    add(
        raw.replace("%", "pct")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "")
        .replace("\\", "")
        .replace(" ", "")
    )
    return tokens


def resolve_prediction_columns(df: pd.DataFrame, property_name: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    dataset_col = None
    for candidate in ["Dataset", "set"]:
        if candidate in df.columns:
            dataset_col = candidate
            break

    actual_col = None
    pred_col = None
    for token in _property_tokens(property_name):
        for actual_candidate, pred_candidate in [
            (f"{token}_Actual", f"{token}_Predicted"),
            (f"True_{token}", f"Pred_{token}"),
            (f"Actual_{token}", f"Predicted_{token}"),
        ]:
            if actual_candidate in df.columns and pred_candidate in df.columns:
                actual_col = actual_candidate
                pred_col = pred_candidate
                return dataset_col, actual_col, pred_col

    return dataset_col, actual_col, pred_col


def _best_trial_id(property_df: pd.DataFrame, metric: str) -> Optional[str]:
    if property_df.empty or metric not in property_df.columns:
        return None

    ascending = metric in {"mae", "rmse"}
    trial_means = (
        property_df.groupby("trial_id", as_index=False)[metric]
        .mean()
        .sort_values([metric, "trial_id"], ascending=[ascending, True], na_position="last")
    )
    if trial_means.empty:
        return None
    return str(trial_means.iloc[0]["trial_id"])


def infer_best_trial_id_from_reference_csv(metrics_df: pd.DataFrame, reference_csv: Path) -> Optional[str]:
    if metrics_df.empty or not reference_csv.exists():
        return None

    try:
        reference_df = pd.read_csv(reference_csv)
    except Exception:
        return None

    if reference_df.empty:
        return None

    wide_df = metrics_df.pivot_table(
        index=["trial_id", "fold"],
        columns="property",
        values=["r2", "mae", "rmse"],
        aggfunc="first",
    )
    if wide_df.empty:
        return None

    wide_df.columns = [f"{property_name}_{metric_name}" for metric_name, property_name in wide_df.columns]
    wide_df = wide_df.reset_index()

    common_metric_cols = [col for col in reference_df.columns if col in wide_df.columns]
    if not common_metric_cols:
        return None

    reference_values = reference_df[common_metric_cols].apply(pd.to_numeric, errors="coerce")
    if reference_values.empty:
        return None

    best_trial_id = None
    best_score = None
    expected_rows = len(reference_values)

    for trial_id, trial_df in wide_df.groupby("trial_id", sort=False):
        if len(trial_df) != expected_rows:
            continue

        trial_values = trial_df[common_metric_cols].apply(pd.to_numeric, errors="coerce")
        if trial_values.isna().all().all():
            continue

        ref_sorted = reference_values.sort_values(common_metric_cols, na_position="last").reset_index(drop=True)
        trial_sorted = trial_values.sort_values(common_metric_cols, na_position="last").reset_index(drop=True)
        diff_score = (trial_sorted - ref_sorted).abs().sum().sum()

        if pd.isna(diff_score):
            continue
        if best_score is None or diff_score < best_score:
            best_score = float(diff_score)
            best_trial_id = str(trial_id)

    return best_trial_id


def _mode_summary(mode_df: pd.DataFrame, selection_metric: str, prefix: str) -> Dict[str, object]:
    if mode_df.empty:
        return {}

    summary: Dict[str, object] = {}
    for metric in ["r2", "mae", "rmse"]:
        if metric not in mode_df.columns:
            continue
        values = pd.to_numeric(mode_df[metric], errors="coerce").dropna()
        if values.empty:
            continue
        summary[f"{prefix}_test_{metric}"] = float(values.mean())
        summary[f"{prefix}_test_{metric}_std"] = float(values.std())

    if selection_metric not in mode_df.columns or f"{prefix}_test_{selection_metric}" not in summary:
        return summary

    target_mean = float(summary[f"{prefix}_test_{selection_metric}"])
    ranked_df = mode_df.assign(_dist=(mode_df[selection_metric] - target_mean).abs()).sort_values(
        ["_dist", selection_metric, "trial_num", "fold"],
        ascending=[True, selection_metric in {"mae", "rmse"}, True, True],
        na_position="last",
    )
    closest_row = ranked_df.iloc[0]
    summary[f"{prefix}_trial_id"] = closest_row["trial_id"]
    summary[f"{prefix}_fold"] = int(closest_row["fold"])
    summary[f"{prefix}_predictions_file"] = str(closest_row["predictions_file"])
    summary[f"{prefix}_closest_{selection_metric}"] = float(closest_row[selection_metric])
    return summary


def summarize_optuna_predictions(
    optuna_trials_dir: Path,
    selection_metric: str,
    global_prefix: str,
    mean_prefix: str,
    preferred_best_trial_id: Optional[str] = None,
) -> Dict[str, Dict[str, object]]:
    metrics_df = collect_optuna_test_metrics(optuna_trials_dir)
    if metrics_df.empty:
        return {}

    summaries: Dict[str, Dict[str, object]] = {}
    for property_name, property_df in metrics_df.groupby("property", sort=True):
        property_summary: Dict[str, object] = {}

        property_summary.update(_mode_summary(property_df, selection_metric, global_prefix))

        best_trial_id = preferred_best_trial_id
        if best_trial_id is not None and best_trial_id not in set(property_df["trial_id"].astype(str)):
            best_trial_id = None
        if best_trial_id is None:
            best_trial_id = _best_trial_id(property_df, selection_metric)
        if best_trial_id is not None:
            best_trial_df = property_df[property_df["trial_id"] == best_trial_id].copy()
            property_summary[mean_prefix + "_best_trial_id"] = best_trial_id
            property_summary.update(_mode_summary(best_trial_df, selection_metric, mean_prefix))

        summaries[str(property_name)] = property_summary

    return summaries
