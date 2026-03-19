from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


CSV_ENCODINGS = ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"]
TEST_LABELS = {"test", "testing", "extrapolationtest"}


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
        if column in {"Dataset", "set"}
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
        [path for path in optuna_trials_dir.iterdir() if path.is_dir() and path.name.startswith("trial_")],
        key=lambda path: int(path.name.split("_")[1]),
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
