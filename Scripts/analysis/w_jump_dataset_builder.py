from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


SAMPLE_KEY_COLUMN = "__sample_key__"
STANDARDIZATION_MODE = "fixed_full_data"


def _finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _threshold_count(total_rows: int, fraction: float) -> int:
    requested = int(round(int(total_rows) * float(fraction)))
    return max(1, min(requested, int(total_rows) - 1))


def _format_identity(value: Any) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric) and math.isfinite(float(numeric)):
        if float(numeric).is_integer():
            return str(int(numeric))
        return format(float(numeric), ".15g")
    return str(value).strip()


def prepare_analysis_frame(frame: pd.DataFrame, target_col: str) -> pd.DataFrame:
    if target_col not in frame.columns:
        raise ValueError(f"target column not found: {target_col}")
    work = frame.copy()
    work["__input_order__"] = np.arange(len(work), dtype=int)
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna(subset=[target_col]).copy()
    if len(work) < 2:
        raise ValueError("at least two rows with finite target values are required")

    if "__source_index__" not in work.columns:
        work["__source_index__"] = work["__input_order__"]
    source = pd.to_numeric(work["__source_index__"], errors="coerce")
    source = source.fillna(work["__input_order__"]).astype(int)
    work["__source_index__"] = source

    if SAMPLE_KEY_COLUMN in work.columns:
        keys = work[SAMPLE_KEY_COLUMN].astype(str)
    elif "ID" in work.columns and work["ID"].notna().all() and work["ID"].astype(str).is_unique:
        keys = work["ID"].map(lambda value: f"id:{_format_identity(value)}")
    else:
        keys = pd.Series(
            [f"source:{int(source_value)}:row:{int(row_value)}" for source_value, row_value in zip(source, work["__input_order__"])],
            index=work.index,
            dtype=str,
        )
    if keys.duplicated().any():
        duplicates = keys[keys.duplicated(keep=False)].head(10).tolist()
        raise ValueError(f"sample identity is not unique: {duplicates}")
    work[SAMPLE_KEY_COLUMN] = keys.to_numpy(dtype=str)
    return work.drop(columns=["__input_order__"]).reset_index(drop=True)


def prepare_ranked_frame(frame: pd.DataFrame, target_col: str) -> pd.DataFrame:
    work = prepare_analysis_frame(frame, target_col)
    work["__rank_input_order__"] = np.arange(len(work), dtype=int)
    source = pd.to_numeric(work["__source_index__"], errors="coerce").fillna(work["__rank_input_order__"])
    if "ID" in work.columns:
        rank_id = pd.to_numeric(work["ID"], errors="coerce").fillna(work["__rank_input_order__"])
    else:
        rank_id = work["__rank_input_order__"]
    work["__rank_source__"] = source.astype(float)
    work["__rank_id__"] = pd.Series(rank_id, index=work.index).astype(float)
    work = work.sort_values(
        [target_col, "__rank_source__", "__rank_id__", "__rank_input_order__"],
        ascending=[False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    work["target_rank"] = np.arange(1, len(work) + 1, dtype=int)
    grouped = work.groupby(target_col, sort=False, dropna=False)
    work["target_tie_size"] = grouped[target_col].transform("size").astype(int)
    work["tie_position"] = grouped.cumcount().astype(int) + 1
    work["tie_group_start_rank"] = grouped["target_rank"].transform("min").astype(int)
    work["tie_group_end_rank"] = grouped["target_rank"].transform("max").astype(int)
    work["tie_group_id"] = pd.factorize(work[target_col], sort=False)[0] + 1
    work["is_tied_target"] = work["target_tie_size"].gt(1)
    return work.drop(columns=["__rank_input_order__", "__rank_source__", "__rank_id__"])


def _numeric_matrix(frame: pd.DataFrame, feature_columns: Iterable[str]) -> np.ndarray:
    columns = list(feature_columns)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"X-space feature columns missing from frame: {missing}")
    return pd.DataFrame(
        {column: pd.to_numeric(frame[column], errors="coerce") for column in columns}
    ).to_numpy(dtype=float)


@dataclass(frozen=True)
class FixedWReference:
    feature_columns: tuple[str, ...]
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: float
    y_scale: float
    projections: np.ndarray
    random_state: int
    imputer_fill_value: float = 0.0

    @property
    def reference_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(json.dumps(list(self.feature_columns), separators=(",", ":")).encode("utf-8"))
        for array in (self.x_mean, self.x_scale, self.projections):
            digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
        digest.update(np.asarray([self.y_mean, self.y_scale, self.random_state], dtype=np.float64).tobytes())
        return digest.hexdigest()

    def transform_x(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = _numeric_matrix(frame, self.feature_columns)
        imputed = np.where(np.isnan(matrix), float(self.imputer_fill_value), matrix)
        return (imputed - self.x_mean) / self.x_scale

    def transform_y(self, frame: pd.DataFrame, target_col: str) -> np.ndarray:
        values = pd.to_numeric(frame[target_col], errors="coerce").dropna().to_numpy(dtype=float)
        return (values - float(self.y_mean)) / float(self.y_scale)

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            feature_columns=np.asarray(self.feature_columns, dtype=str),
            x_mean=np.asarray(self.x_mean, dtype=float),
            x_scale=np.asarray(self.x_scale, dtype=float),
            y_mean=np.asarray([self.y_mean], dtype=float),
            y_scale=np.asarray([self.y_scale], dtype=float),
            projections=np.asarray(self.projections, dtype=float),
            random_state=np.asarray([self.random_state], dtype=int),
            imputer_fill_value=np.asarray([self.imputer_fill_value], dtype=float),
            reference_hash=np.asarray([self.reference_hash], dtype=str),
        )


def fit_fixed_w_reference(
    frame: pd.DataFrame,
    target_col: str,
    feature_columns: Iterable[str],
    sliced_projections: int = 128,
    random_state: int = 42,
) -> FixedWReference:
    work = prepare_analysis_frame(frame, target_col)
    columns = tuple(str(column) for column in feature_columns)
    if not columns:
        raise ValueError("feature_columns must not be empty")
    if int(sliced_projections) < 1:
        raise ValueError("sliced_projections must be positive")

    imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    scaler = StandardScaler()
    matrix = _numeric_matrix(work, columns)
    imputed = imputer.fit_transform(matrix)
    scaler.fit(imputed)
    x_scale = np.asarray(scaler.scale_, dtype=float)
    x_scale = np.where(np.isfinite(x_scale) & (x_scale > 1e-12), x_scale, 1.0)

    y = pd.to_numeric(work[target_col], errors="coerce").dropna().to_numpy(dtype=float)
    y_mean = float(np.mean(y))
    y_scale = float(np.std(y, ddof=0))
    if not math.isfinite(y_scale) or y_scale <= 1e-12:
        y_scale = 1.0

    rng = np.random.default_rng(int(random_state))
    projections = rng.normal(size=(int(sliced_projections), len(columns)))
    norms = np.linalg.norm(projections, axis=1, keepdims=True)
    projections = projections / np.where(norms <= 1e-12, 1.0, norms)
    return FixedWReference(
        feature_columns=columns,
        x_mean=np.asarray(scaler.mean_, dtype=float),
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
        projections=np.asarray(projections, dtype=float),
        random_state=int(random_state),
    )


def _sliced_wasserstein(left: np.ndarray, right: np.ndarray, projections: np.ndarray) -> float:
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("X arrays must be two-dimensional")
    if left.shape[1] != right.shape[1] or left.shape[1] != projections.shape[1]:
        raise ValueError("X arrays and projection directions must share the same feature count")
    distances = [
        wasserstein_distance(left @ direction, right @ direction)
        for direction in projections
    ]
    return _finite_float(np.mean(distances))


def compute_fixed_distance(
    left: pd.DataFrame,
    right: pd.DataFrame,
    target_col: str,
    reference: FixedWReference,
) -> dict[str, Any]:
    if left.empty or right.empty:
        raise ValueError("both distributions must contain at least one sample")
    left_x = reference.transform_x(left)
    right_x = reference.transform_x(right)
    left_y = reference.transform_y(left, target_col)
    right_y = reference.transform_y(right, target_col)
    return {
        "Wx": _sliced_wasserstein(left_x, right_x, reference.projections),
        "Wy": _finite_float(wasserstein_distance(left_y, right_y)),
        "left_size": int(len(left)),
        "right_size": int(len(right)),
        "reference_hash": reference.reference_hash,
        "standardization_mode": STANDARDIZATION_MODE,
        "x_distance_method": "sliced_wasserstein_fixed_standardized_x",
        "x_sliced_projections": int(len(reference.projections)),
        "y_distance_method": "wasserstein_1_fixed_standardized_y",
    }


def _sample_metadata(row: pd.Series, target_col: str) -> dict[str, Any]:
    return {
        SAMPLE_KEY_COLUMN: str(row[SAMPLE_KEY_COLUMN]),
        "ID": row.get("ID", np.nan),
        "entering_source_index": row.get("__source_index__", np.nan),
        "entering_target_value": _finite_float(row[target_col]),
        "target_rank": int(row["target_rank"]),
        "target_tie_size": int(row["target_tie_size"]),
        "tie_position": int(row["tie_position"]),
        "tie_group_id": int(row["tie_group_id"]),
        "tie_group_start_rank": int(row["tie_group_start_rank"]),
        "tie_group_end_rank": int(row["tie_group_end_rank"]),
        "is_tied_target": bool(row["is_tied_target"]),
    }


def compute_samplewise_fixed_curve(
    frame: pd.DataFrame,
    target_col: str,
    reference: FixedWReference,
    start_fraction: float = 0.05,
    stop_fraction: float = 0.95,
    metadata: dict[str, Any] | None = None,
    policy: str = "baseline_full_data",
) -> pd.DataFrame:
    if not 0 < float(start_fraction) < float(stop_fraction) < 1:
        raise ValueError("fractions must satisfy 0 < start < stop < 1")
    ranked = prepare_ranked_frame(frame, target_col)
    total = int(len(ranked))
    lower = _threshold_count(total, start_fraction)
    upper = _threshold_count(total, stop_fraction)
    first = max(1, lower - 1)
    previous_wx = float("nan")
    previous_wy = float("nan")
    rows: list[dict[str, Any]] = []
    base = dict(metadata or {})
    for top_size in range(first, upper + 1):
        metrics = compute_fixed_distance(
            ranked.iloc[:top_size],
            ranked.iloc[top_size:],
            target_col,
            reference,
        )
        wx = float(metrics["Wx"])
        wy = float(metrics["Wy"])
        delta_wx = wx - previous_wx if math.isfinite(previous_wx) else float("nan")
        delta_wy = wy - previous_wy if math.isfinite(previous_wy) else float("nan")
        relative = (
            delta_wx / previous_wx
            if math.isfinite(delta_wx) and math.isfinite(previous_wx) and abs(previous_wx) > 1e-12
            else float("nan")
        )
        entering = ranked.iloc[top_size - 1]
        rows.append(
            {
                **base,
                "policy": str(policy),
                "target_col": target_col,
                "total_valid_samples": total,
                "top_tail_size": int(top_size),
                "background_size": int(total - top_size),
                "tail_fraction": float(top_size / total),
                "tail_percent": float(100.0 * top_size / total),
                "Wx": wx,
                "Wy": wy,
                "delta_Wx": _finite_float(delta_wx),
                "relative_delta_Wx": _finite_float(relative),
                "delta_Wy": _finite_float(delta_wy),
                **metrics,
                **_sample_metadata(entering, target_col),
            }
        )
        previous_wx = wx
        previous_wy = wy
    result = pd.DataFrame(rows)
    return result[result["top_tail_size"].ge(lower)].reset_index(drop=True)


@dataclass(frozen=True)
class JumpDetectionResult:
    curve: pd.DataFrame
    candidates: pd.DataFrame
    drivers: pd.DataFrame
    tie_groups: pd.DataFrame
    ranked_frame: pd.DataFrame


def _relative_change(delta: float, baseline: float) -> float:
    if not math.isfinite(delta) or not math.isfinite(baseline) or abs(baseline) <= 1e-12:
        return float("nan")
    return float(delta / baseline)


def _meets_jump_rule(delta: float, relative: float, minimum_abs: float, minimum_rel: float) -> bool:
    return bool(
        math.isfinite(delta)
        and math.isfinite(relative)
        and delta >= float(minimum_abs)
        and relative >= float(minimum_rel)
    )


def detect_jump_drivers(
    frame: pd.DataFrame,
    target_col: str,
    reference: FixedWReference,
    start_fraction: float = 0.05,
    stop_fraction: float = 0.95,
    min_abs_delta_wx: float = 0.15,
    min_rel_delta_wx: float = 0.30,
    metadata: dict[str, Any] | None = None,
) -> JumpDetectionResult:
    ranked = prepare_ranked_frame(frame, target_col)
    total = int(len(ranked))
    lower_visible = _threshold_count(total, start_fraction)
    upper_visible = _threshold_count(total, stop_fraction)
    curve = compute_samplewise_fixed_curve(
        ranked,
        target_col=target_col,
        reference=reference,
        start_fraction=start_fraction,
        stop_fraction=stop_fraction,
        metadata=metadata,
    )
    base = dict(metadata or {})
    candidate_rows: list[dict[str, Any]] = []
    tie_rows: list[dict[str, Any]] = []

    for _, group in ranked.groupby("tie_group_id", sort=False):
        start_rank = int(group["tie_group_start_rank"].iloc[0])
        end_rank = int(group["tie_group_end_rank"].iloc[0])
        tie_size = int(group["target_tie_size"].iloc[0])
        if start_rank < 2 or end_rank < lower_visible or end_rank > upper_visible:
            continue
        lower_size = start_rank - 1
        upper_size = end_rank
        lower_metrics = compute_fixed_distance(
            ranked.iloc[:lower_size], ranked.iloc[lower_size:], target_col, reference
        )
        upper_metrics = compute_fixed_distance(
            ranked.iloc[:upper_size], ranked.iloc[upper_size:], target_col, reference
        )
        delta_wx = float(upper_metrics["Wx"] - lower_metrics["Wx"])
        relative_delta = _relative_change(delta_wx, float(lower_metrics["Wx"]))
        if not _meets_jump_rule(
            delta_wx,
            relative_delta,
            min_abs_delta_wx,
            min_rel_delta_wx,
        ):
            continue

        confirmed_count = 0
        member_keys: list[str] = []
        top_at_upper = ranked.iloc[:upper_size]
        background_at_upper = ranked.iloc[upper_size:]
        for _, member in group.iterrows():
            sample_key = str(member[SAMPLE_KEY_COLUMN])
            member_keys.append(sample_key)
            top_without = top_at_upper[top_at_upper[SAMPLE_KEY_COLUMN].ne(sample_key)]
            without_metrics = compute_fixed_distance(
                top_without,
                background_at_upper,
                target_col,
                reference,
            )
            reduction = float(upper_metrics["Wx"] - without_metrics["Wx"])
            relative_reduction = _relative_change(reduction, float(without_metrics["Wx"]))
            confirmed = _meets_jump_rule(
                reduction,
                relative_reduction,
                min_abs_delta_wx,
                min_rel_delta_wx,
            )
            confirmed_count += int(confirmed)
            candidate_rows.append(
                {
                    **base,
                    **_sample_metadata(member, target_col),
                    "candidate_type": "tie_group" if tie_size > 1 else "singleton",
                    "lower_top_tail_size": lower_size,
                    "upper_top_tail_size": upper_size,
                    "lower_Wx": float(lower_metrics["Wx"]),
                    "upper_Wx": float(upper_metrics["Wx"]),
                    "delta_Wx": delta_wx,
                    "relative_delta_Wx": relative_delta,
                    "lower_Wy": float(lower_metrics["Wy"]),
                    "upper_Wy": float(upper_metrics["Wy"]),
                    "delta_Wy": float(upper_metrics["Wy"] - lower_metrics["Wy"]),
                    "Wx_without_sample": float(without_metrics["Wx"]),
                    "loo_reduction_Wx": reduction,
                    "loo_relative_reduction_Wx": relative_reduction,
                    "is_confirmed_driver": bool(confirmed),
                    "driver_action": "drop" if confirmed else "report_only",
                    "reference_hash": reference.reference_hash,
                    "standardization_mode": STANDARDIZATION_MODE,
                }
            )
        if tie_size > 1:
            tie_rows.append(
                {
                    **base,
                    "tie_group_id": int(group["tie_group_id"].iloc[0]),
                    "target_value": _finite_float(group[target_col].iloc[0]),
                    "member_count": tie_size,
                    "member_sample_keys": ";".join(member_keys),
                    "lower_top_tail_size": lower_size,
                    "upper_top_tail_size": upper_size,
                    "lower_Wx": float(lower_metrics["Wx"]),
                    "upper_Wx": float(upper_metrics["Wx"]),
                    "delta_Wx": delta_wx,
                    "relative_delta_Wx": relative_delta,
                    "confirmed_driver_count": confirmed_count,
                    "tie_status": "confirmed_members" if confirmed_count else "report_only",
                    "reference_hash": reference.reference_hash,
                }
            )

    candidates = pd.DataFrame(candidate_rows)
    if candidates.empty:
        drivers = pd.DataFrame(columns=[*candidates.columns, SAMPLE_KEY_COLUMN])
    else:
        candidates = candidates.sort_values(
            ["upper_top_tail_size", SAMPLE_KEY_COLUMN], kind="stable"
        ).reset_index(drop=True)
        drivers = candidates[candidates["is_confirmed_driver"].eq(True)].reset_index(drop=True)
    return JumpDetectionResult(
        curve=curve,
        candidates=candidates,
        drivers=drivers,
        tie_groups=pd.DataFrame(tie_rows),
        ranked_frame=ranked,
    )


@dataclass(frozen=True)
class PreparedJumpDataset:
    full_data: pd.DataFrame
    train_baseline: pd.DataFrame
    train_drop_jump_drivers: pd.DataFrame
    test_fixed: pd.DataFrame
    split_assignments: pd.DataFrame
    driver_sample_keys: tuple[str, ...]
    test_fraction: float
    random_state: int
    target_strata: int


def _assign_target_strata(ranked: pd.DataFrame, target_strata: int) -> pd.Series:
    count = int(len(ranked))
    strata = np.floor(np.arange(count, dtype=float) * int(target_strata) / count).astype(int)
    return pd.Series(np.clip(strata, 0, int(target_strata) - 1), index=ranked.index, dtype=int)


def prepare_jump_dataset(
    frame: pd.DataFrame,
    target_col: str,
    driver_sample_keys: Iterable[str],
    test_fraction: float = 0.20,
    random_state: int = 42,
    target_strata: int = 20,
) -> PreparedJumpDataset:
    if not 0 < float(test_fraction) < 1:
        raise ValueError("test_fraction must be between zero and one")
    full = prepare_analysis_frame(frame, target_col)
    driver_keys = tuple(sorted({str(value) for value in driver_sample_keys}))
    full_keys = set(full[SAMPLE_KEY_COLUMN].astype(str))
    missing_drivers = sorted(set(driver_keys) - full_keys)
    if missing_drivers:
        raise ValueError(f"driver sample keys are missing from full data: {missing_drivers[:10]}")

    non_driver = full[~full[SAMPLE_KEY_COLUMN].isin(driver_keys)].copy()
    test_count = int(round(len(full) * float(test_fraction)))
    test_count = max(1, min(test_count, len(full) - 1))
    if len(non_driver) <= test_count:
        raise ValueError("non-driver pool is too small for the requested fixed test set")

    strata_count = int(target_strata)
    if strata_count < 2:
        raise ValueError("target_strata must be at least two")
    if test_count < strata_count or (len(non_driver) - test_count) < strata_count:
        raise ValueError("requested 20-stratum split is not feasible for the available samples")

    ranked_non_driver = prepare_ranked_frame(non_driver, target_col)
    ranked_non_driver["__test_stratum__"] = _assign_target_strata(ranked_non_driver, strata_count)
    if ranked_non_driver.groupby("__test_stratum__").size().min() < 2:
        raise ValueError("each target stratum must contain at least two non-driver samples")

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_count,
        random_state=int(random_state),
    )
    indices = np.arange(len(ranked_non_driver), dtype=int)
    _, test_idx = next(splitter.split(indices, ranked_non_driver["__test_stratum__"]))
    test_keys = set(ranked_non_driver.iloc[test_idx][SAMPLE_KEY_COLUMN].astype(str))

    test_fixed = full[full[SAMPLE_KEY_COLUMN].isin(test_keys)].copy().reset_index(drop=True)
    train_baseline = full[~full[SAMPLE_KEY_COLUMN].isin(test_keys)].copy().reset_index(drop=True)
    train_drop = train_baseline[~train_baseline[SAMPLE_KEY_COLUMN].isin(driver_keys)].copy().reset_index(drop=True)

    stratum_by_key = ranked_non_driver.set_index(SAMPLE_KEY_COLUMN)["__test_stratum__"].to_dict()
    assignments = full[[SAMPLE_KEY_COLUMN]].copy()
    assignments["is_confirmed_driver"] = assignments[SAMPLE_KEY_COLUMN].isin(driver_keys)
    assignments["target_stratum"] = assignments[SAMPLE_KEY_COLUMN].map(stratum_by_key).astype("Int64")
    assignments["split_role"] = np.where(
        assignments[SAMPLE_KEY_COLUMN].isin(test_keys),
        "test_fixed",
        np.where(assignments["is_confirmed_driver"], "train_driver", "train_regular"),
    )

    baseline_keys = set(train_baseline[SAMPLE_KEY_COLUMN].astype(str))
    clean_keys = set(train_drop[SAMPLE_KEY_COLUMN].astype(str))
    fixed_test_keys = set(test_fixed[SAMPLE_KEY_COLUMN].astype(str))
    if fixed_test_keys & set(driver_keys):
        raise AssertionError("fixed test set contains confirmed jump drivers")
    if baseline_keys & fixed_test_keys:
        raise AssertionError("baseline train and fixed test overlap")
    if baseline_keys | fixed_test_keys != full_keys:
        raise AssertionError("baseline train and fixed test do not reconstruct full data")
    if clean_keys != baseline_keys - set(driver_keys):
        raise AssertionError("cleaned train is not baseline train minus drivers")
    if clean_keys | fixed_test_keys != full_keys - set(driver_keys):
        raise AssertionError("cleaned train and test do not reconstruct non-driver data")

    return PreparedJumpDataset(
        full_data=full,
        train_baseline=train_baseline,
        train_drop_jump_drivers=train_drop,
        test_fixed=test_fixed,
        split_assignments=assignments,
        driver_sample_keys=driver_keys,
        test_fraction=float(test_fraction),
        random_state=int(random_state),
        target_strata=strata_count,
    )
