from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler


ID_LIKE_PATTERN = re.compile(r"^(id|__row_id__|.*_id)$", re.IGNORECASE)
HELPER_COLUMNS = {"__row_id__", "__original_order__", "__source_index__"}
METADATA_COLUMNS = {"reference_id", "references", "number", "dois"}
PERFORMANCE_COLUMNS = {"El(%)", "UTS(MPa)", "YS(MPa)", "yield strength"}
COMPOSITION_SUFFIXES = ("(wt%)", "(at%)")
PROCESS_REGEXES = (
    re.compile(r"^st\d+$", re.IGNORECASE),
    re.compile(r"^time\d+$", re.IGNORECASE),
)
PROCESS_KEYWORDS = (
    "temp",
    "temperature",
    "time",
    "aging",
    "ageing",
    "anneal",
    "solution",
    "deformation",
    "recrystalize",
    "recrystallize",
    "thermo-mechanical",
)
X_SPACE_FEATURE_POLICY = "composition_plus_numeric_processing"


@dataclass
class TraceArtifacts:
    x_space_features: pd.DataFrame
    projection_2d: pd.DataFrame
    row_assignments: pd.DataFrame
    candidate_pool: pd.DataFrame
    selected_test_rows: pd.DataFrame
    cluster_assignments: pd.DataFrame
    density_scores: pd.DataFrame
    y_density_curve: pd.DataFrame | None = None
    neighbor_map: Dict[str, Any] | None = None
    cluster_size_table: pd.DataFrame | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparedSplit:
    split_strategy: str
    target_col: str
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    summary: Dict[str, Any]
    train_label: str = "train"
    test_label: str = "test"
    trace: TraceArtifacts | None = None


@dataclass
class PreparedFold:
    fold_index: int
    held_out_cluster_id: int
    split: PreparedSplit
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class XSpaceContext:
    feature_names: List[str]
    feature_roles: Dict[str, str]
    excluded_property_columns: List[str]
    feature_policy: str
    scaled_features: pd.DataFrame
    matrix: np.ndarray
    projection: pd.DataFrame
    row_id_to_index: Dict[int, int]


@dataclass
class DensityContext:
    x_space: XSpaceContext
    density_scores: pd.DataFrame
    selection_space: str
    selection_density: np.ndarray
    selection_bandwidth: float
    x_density_bandwidth: float
    y_density_bandwidth: float
    y_density_curve: pd.DataFrame | None = None


class StrengthOODProcessorBase:
    split_strategy = ""
    train_label = "train"
    test_label = "test"

    def __init__(
        self,
        input_file: str,
        random_state: int = 42,
        processing_cols: Sequence[str] | None = None,
    ) -> None:
        self.input_file = input_file
        self.random_state = random_state
        self.processing_cols = list(processing_cols or [])

    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.input_file)

    def prepare(self, df: pd.DataFrame, target_col: str, **kwargs: Any) -> PreparedSplit | List[PreparedFold]:
        raise NotImplementedError

    def _prepare_work_df(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        return prepare_work_dataframe(df, target_col)


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_ready(v) for v in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return json_ready(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return json_ready(value.to_dict())
    return _to_python_scalar(value)


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(json_ready(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def ensure_positive_int(name: str, value: int) -> int:
    if int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def resolve_test_cap(total_rows: int, test_ratio: float) -> int:
    if not 0 < test_ratio < 1:
        raise ValueError("test_size must be between 0 and 1")
    requested = int(round(total_rows * float(test_ratio)))
    return max(1, min(requested, total_rows - 1))


def prepare_work_dataframe(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    if target_col not in df.columns:
        raise ValueError(f"split target column '{target_col}' not found in dataset")

    work_df = df.copy()
    work_df["__source_index__"] = np.arange(len(work_df), dtype=int)
    work_df[target_col] = pd.to_numeric(work_df[target_col], errors="coerce")
    work_df = work_df.dropna(subset=[target_col]).reset_index(drop=True)
    if len(work_df) < 2:
        raise ValueError("at least two valid rows are required for OOD splitting")

    work_df["__row_id__"] = np.arange(len(work_df), dtype=int)
    work_df["__original_order__"] = np.arange(len(work_df), dtype=int)
    return work_df


def drop_helper_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[col for col in df.columns if col in HELPER_COLUMNS], errors="ignore")


def resolve_split_labels(split_strategy: str, extrapolation_side: str | None = None) -> tuple[str, str]:
    if split_strategy == "target_extrapolation":
        if extrapolation_side == "high_to_low":
            return "train_high", "test_low"
        return "train_low", "test_high"
    if split_strategy == "sparse_x_single":
        return "train_inlier", "test_sparse_x_single"
    if split_strategy == "sparse_y_single":
        return "train_inlier", "test_sparse_y_single"
    if split_strategy == "sparse_x_cluster":
        return "train_inlier", "test_sparse_x_cluster"
    if split_strategy == "sparse_y_cluster":
        return "train_inlier", "test_sparse_y_cluster"
    if split_strategy == "loco":
        return "train", "test"
    if split_strategy == "random_cv_baseline":
        return "train", "test"
    return "train", "test"


def _build_summary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    split_strategy: str,
    train_label: str,
    test_label: str,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    train_target = pd.to_numeric(train_df[target_col], errors="coerce")
    test_target = pd.to_numeric(test_df[target_col], errors="coerce")
    total_size = len(train_df) + len(test_df)
    summary: Dict[str, Any] = {
        "split_target_col": target_col,
        "split_strategy": split_strategy,
        "train_label": train_label,
        "test_label": test_label,
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "total_size": int(total_size),
        "test_ratio": float(len(test_df) / total_size),
        "train_target_min": float(train_target.min()),
        "train_target_max": float(train_target.max()),
        "test_target_min": float(test_target.min()),
        "test_target_max": float(test_target.max()),
    }
    if extra:
        summary.update(json_ready(extra))
    return summary


def _ordered_subset(work_df: pd.DataFrame, row_ids: Iterable[int]) -> pd.DataFrame:
    ordered_ids = {int(row_id) for row_id in row_ids}
    subset = work_df[work_df["__row_id__"].isin(ordered_ids)].copy()
    return subset.sort_values("__original_order__", kind="stable").reset_index(drop=True)


def make_prepared_split(
    work_df: pd.DataFrame,
    target_col: str,
    split_strategy: str,
    train_row_ids: Sequence[int],
    test_row_ids: Sequence[int],
    trace: TraceArtifacts,
    extra_summary: Dict[str, Any] | None = None,
    extrapolation_side: str | None = None,
    train_label: str | None = None,
    test_label: str | None = None,
) -> PreparedSplit:
    if len(test_row_ids) == 0:
        raise ValueError("OOD split produced an empty test set")
    if len(train_row_ids) == 0:
        raise ValueError("OOD split produced an empty train set")

    resolved_train_label, resolved_test_label = resolve_split_labels(split_strategy, extrapolation_side)
    final_train_label = train_label or resolved_train_label
    final_test_label = test_label or resolved_test_label

    train_df = drop_helper_columns(_ordered_subset(work_df, train_row_ids)).reset_index(drop=True)
    test_df = drop_helper_columns(_ordered_subset(work_df, test_row_ids)).reset_index(drop=True)
    x_space_summary = {
        "x_space_feature_policy": trace.metadata.get("x_space_feature_policy"),
        "x_space_feature_columns": trace.metadata.get("x_space_feature_columns"),
        "x_space_feature_roles": trace.metadata.get("x_space_feature_roles"),
        "excluded_property_columns": trace.metadata.get("excluded_property_columns"),
    }
    merged_extra = {**x_space_summary, **(extra_summary or {})}
    summary = _build_summary(
        train_df=train_df,
        test_df=test_df,
        target_col=target_col,
        split_strategy=split_strategy,
        train_label=final_train_label,
        test_label=final_test_label,
        extra=merged_extra,
    )
    return PreparedSplit(
        split_strategy=split_strategy,
        target_col=target_col,
        train_df=train_df,
        test_df=test_df,
        summary=summary,
        train_label=final_train_label,
        test_label=final_test_label,
        trace=trace,
    )


def _is_id_like_column(column_name: str) -> bool:
    return bool(ID_LIKE_PATTERN.match(column_name))


def _is_metadata_column(column_name: str) -> bool:
    return column_name.strip().lower() in METADATA_COLUMNS


def _is_property_column(column_name: str, target_col: str) -> bool:
    raw = str(column_name).strip()
    return raw == target_col or raw in PERFORMANCE_COLUMNS


def _is_composition_column(column_name: str) -> bool:
    return str(column_name).strip().endswith(COMPOSITION_SUFFIXES)


def _is_processing_column(column_name: str, explicit_processing_cols: set[str] | None = None) -> bool:
    raw = str(column_name).strip()
    normalized = raw.lower()
    if explicit_processing_cols and raw in explicit_processing_cols:
        return True
    if normalized == "cr(%)":
        return True
    if any(pattern.match(raw) for pattern in PROCESS_REGEXES):
        return True
    return any(keyword in normalized for keyword in PROCESS_KEYWORDS)


def select_numeric_x_space_columns(
    work_df: pd.DataFrame,
    target_col: str,
    processing_cols: Sequence[str] | None = None,
) -> tuple[List[str], Dict[str, str], List[str]]:
    numeric_columns = work_df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    excluded = {target_col, *HELPER_COLUMNS}
    explicit_processing_cols = {str(col).strip() for col in (processing_cols or []) if str(col).strip()}

    selected: List[str] = []
    roles: Dict[str, str] = {}
    excluded_property_columns: List[str] = []
    for column in numeric_columns:
        if column in excluded or _is_id_like_column(column) or _is_metadata_column(column):
            continue
        if _is_property_column(column, target_col):
            excluded_property_columns.append(column)
            continue
        if _is_composition_column(column):
            selected.append(column)
            roles[column] = "composition"
        elif _is_processing_column(column, explicit_processing_cols):
            selected.append(column)
            roles[column] = "process"

    return selected, roles, excluded_property_columns


def build_numeric_x_space(
    work_df: pd.DataFrame,
    target_col: str,
    random_state: int,
    processing_cols: Sequence[str] | None = None,
) -> XSpaceContext:
    numeric_frame = work_df.select_dtypes(include=[np.number, "bool"]).copy()
    candidate_cols, feature_roles, excluded_property_columns = select_numeric_x_space_columns(
        work_df,
        target_col=target_col,
        processing_cols=processing_cols,
    )
    if not candidate_cols:
        raise ValueError(
            "No numeric composition or processing columns are available for OOD X-space "
            f"construction with policy '{X_SPACE_FEATURE_POLICY}'"
        )

    usable = numeric_frame[candidate_cols].apply(pd.to_numeric, errors="coerce")
    usable = usable.dropna(axis=1, how="all")
    usable = usable.loc[:, usable.nunique(dropna=True) > 1]
    feature_roles = {column: feature_roles[column] for column in usable.columns}
    if usable.empty:
        raise ValueError(
            "No usable numeric composition or processing columns remain after dropping "
            "empty/constant OOD X-space columns"
        )

    usable = usable.fillna(usable.median())
    scaler = StandardScaler()
    matrix = scaler.fit_transform(usable.values.astype(float))
    projection = project_to_2d(matrix, random_state=random_state)

    scaled_features = pd.DataFrame(matrix, columns=usable.columns, index=work_df.index)
    scaled_features.insert(0, "__row_id__", work_df["__row_id__"].to_numpy())
    scaled_features.insert(1, "__source_index__", work_df["__source_index__"].to_numpy())
    scaled_features.insert(2, target_col, work_df[target_col].to_numpy())

    projection_df = pd.DataFrame(
        {
            "__row_id__": work_df["__row_id__"].to_numpy(),
            "__source_index__": work_df["__source_index__"].to_numpy(),
            target_col: work_df[target_col].to_numpy(),
            "projection_x": projection[:, 0],
            "projection_y": projection[:, 1],
        }
    )
    return XSpaceContext(
        feature_names=list(usable.columns),
        feature_roles=feature_roles,
        excluded_property_columns=excluded_property_columns,
        feature_policy=X_SPACE_FEATURE_POLICY,
        scaled_features=scaled_features,
        matrix=matrix,
        projection=projection_df,
        row_id_to_index={int(row_id): idx for idx, row_id in enumerate(work_df["__row_id__"].tolist())},
    )


def project_to_2d(matrix: np.ndarray, random_state: int) -> np.ndarray:
    row_count, feature_count = matrix.shape
    if row_count < 3:
        first = matrix[:, 0] if feature_count >= 1 else np.arange(row_count, dtype=float)
        second = matrix[:, 1] if feature_count >= 2 else np.zeros(row_count, dtype=float)
        return np.column_stack([first, second])
    perplexity = min(30, row_count - 1, max(2, row_count // 3))
    try:
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        )
        return tsne.fit_transform(matrix)
    except Exception:
        first = matrix[:, 0] if feature_count >= 1 else np.arange(row_count, dtype=float)
        second = matrix[:, 1] if feature_count >= 2 else np.zeros(row_count, dtype=float)
        return np.column_stack([first, second])


def estimate_kde_bandwidth(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if len(array) < 2:
        return 1.0

    scale_candidates = np.std(array, axis=0, ddof=1)
    scale_candidates = scale_candidates[np.isfinite(scale_candidates) & (scale_candidates > 0)]
    scale = float(np.median(scale_candidates)) if len(scale_candidates) else 1.0
    bandwidth = scale * (len(array) ** (-1.0 / (array.shape[1] + 4)))
    return float(max(bandwidth, 1e-3))


def compute_kde_density(values: np.ndarray, bandwidth: float | None = None) -> tuple[np.ndarray, float]:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)

    actual_bandwidth = float(bandwidth) if bandwidth is not None else estimate_kde_bandwidth(array)
    kde = KernelDensity(kernel="gaussian", bandwidth=actual_bandwidth)
    kde.fit(array)
    density = np.exp(kde.score_samples(array))
    return density, actual_bandwidth


def build_y_density_curve(target_values: np.ndarray, bandwidth: float, target_col: str) -> pd.DataFrame:
    if len(target_values) == 0:
        return pd.DataFrame(columns=[target_col, "density"])
    grid = np.linspace(float(np.min(target_values)), float(np.max(target_values)), num=256)
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(target_values.reshape(-1, 1))
    density = np.exp(kde.score_samples(grid.reshape(-1, 1)))
    return pd.DataFrame({target_col: grid, "density": density})


def build_density_context(
    work_df: pd.DataFrame,
    target_col: str,
    random_state: int,
    selection_space: str,
    kde_bandwidth: float | None = None,
    include_y_curve: bool = False,
    processing_cols: Sequence[str] | None = None,
) -> DensityContext:
    if selection_space not in {"x", "y"}:
        raise ValueError("selection_space must be 'x' or 'y'")

    x_space = build_numeric_x_space(
        work_df,
        target_col=target_col,
        random_state=random_state,
        processing_cols=processing_cols,
    )
    x_density, x_bandwidth = compute_kde_density(
        x_space.projection[["projection_x", "projection_y"]].to_numpy(),
        bandwidth=kde_bandwidth if selection_space == "x" else None,
    )
    y_density, y_bandwidth = compute_kde_density(
        work_df[target_col].to_numpy(dtype=float),
        bandwidth=kde_bandwidth if selection_space == "y" else None,
    )
    selection_density = x_density if selection_space == "x" else y_density

    density_scores = x_space.projection.copy()
    density_scores["x_density"] = x_density
    density_scores["y_density"] = y_density
    density_scores["selection_density"] = selection_density
    density_scores["selection_space"] = selection_space

    y_density_curve = None
    if include_y_curve:
        y_density_curve = build_y_density_curve(
            work_df[target_col].to_numpy(dtype=float),
            bandwidth=y_bandwidth,
            target_col=target_col,
        )

    return DensityContext(
        x_space=x_space,
        density_scores=density_scores,
        selection_space=selection_space,
        selection_density=selection_density,
        selection_bandwidth=float(x_bandwidth if selection_space == "x" else y_bandwidth),
        x_density_bandwidth=float(x_bandwidth),
        y_density_bandwidth=float(y_bandwidth),
        y_density_curve=y_density_curve,
    )


def enrich_rows_for_trace(
    work_df: pd.DataFrame,
    target_col: str,
    density_scores: pd.DataFrame,
    row_ids: Sequence[int] | None = None,
) -> pd.DataFrame:
    base = density_scores.copy()
    if row_ids is not None:
        base = base[base["__row_id__"].isin([int(row_id) for row_id in row_ids])].copy()

    raw_columns = [
        col
        for col in work_df.columns
        if col not in HELPER_COLUMNS and col != target_col
    ]
    raw_slice = work_df[["__row_id__", *raw_columns]].copy()
    return base.merge(raw_slice, on="__row_id__", how="left")


def select_low_density_candidates(
    work_df: pd.DataFrame,
    target_col: str,
    density_context: DensityContext,
    candidate_pool_size: int,
) -> tuple[pd.DataFrame, int]:
    candidate_pool_size = ensure_positive_int("sparse_candidate_pool_size", candidate_pool_size)
    actual_size = min(candidate_pool_size, len(work_df))
    candidates = density_context.density_scores.sort_values(
        by=["selection_density", target_col, "__row_id__"],
        ascending=[True, True, True],
        kind="stable",
    ).head(actual_size)
    return enrich_rows_for_trace(work_df, target_col, candidates), actual_size


def _matrix_for_row_ids(matrix: np.ndarray, row_ids: Sequence[int], row_id_to_index: Dict[int, int]) -> np.ndarray:
    positions = [row_id_to_index[int(row_id)] for row_id in row_ids]
    return matrix[positions]


def assign_clusters(
    frame: pd.DataFrame,
    representation: np.ndarray,
    requested_cluster_count: int,
    random_state: int,
    samples_per_cluster: int | None = None,
) -> tuple[pd.DataFrame, int]:
    if frame.empty:
        result = frame.copy()
        result["cluster_id"] = pd.Series(dtype=int)
        return result, 0

    requested_cluster_count = ensure_positive_int("sparse_cluster_count", requested_cluster_count)
    cluster_count = min(requested_cluster_count, len(frame))
    if samples_per_cluster is not None:
        samples_per_cluster = ensure_positive_int("sparse_samples_per_cluster", samples_per_cluster)
        cluster_count = min(cluster_count, max(1, len(frame) // samples_per_cluster))

    cluster_count = max(1, cluster_count)
    if cluster_count == 1:
        labels = np.zeros(len(frame), dtype=int)
    else:
        model = KMeans(n_clusters=cluster_count, random_state=random_state, n_init=10)
        labels = model.fit_predict(representation)

    clustered = frame.copy().reset_index(drop=True)
    clustered["cluster_id"] = labels
    return clustered, int(cluster_count)


def build_row_assignments(
    work_df: pd.DataFrame,
    target_col: str,
    test_row_ids: Sequence[int],
    train_label: str,
    test_label: str,
    candidate_row_ids: Sequence[int] | None = None,
) -> pd.DataFrame:
    test_id_set = {int(row_id) for row_id in test_row_ids}
    candidate_id_set = {int(row_id) for row_id in (candidate_row_ids or [])}
    row_assignments = work_df[["__row_id__", "__source_index__", target_col]].copy()
    row_assignments["split_role"] = np.where(
        row_assignments["__row_id__"].isin(test_id_set),
        test_label,
        train_label,
    )
    row_assignments["is_test"] = row_assignments["__row_id__"].isin(test_id_set)
    row_assignments["is_candidate"] = row_assignments["__row_id__"].isin(candidate_id_set)
    return row_assignments


def build_trace_artifacts(
    work_df: pd.DataFrame,
    target_col: str,
    density_context: DensityContext,
    split_strategy: str,
    train_label: str,
    test_label: str,
    test_row_ids: Sequence[int],
    candidate_pool: pd.DataFrame | None = None,
    selected_test_rows: pd.DataFrame | None = None,
    cluster_assignments: pd.DataFrame | None = None,
    neighbor_map: Dict[str, Any] | None = None,
    cluster_size_table: pd.DataFrame | None = None,
    metadata: Dict[str, Any] | None = None,
) -> TraceArtifacts:
    candidate_pool = candidate_pool if candidate_pool is not None else pd.DataFrame()
    selected_test_rows = (
        selected_test_rows
        if selected_test_rows is not None
        else enrich_rows_for_trace(work_df, target_col, density_context.density_scores, test_row_ids)
    )
    cluster_assignments = cluster_assignments if cluster_assignments is not None else pd.DataFrame()
    candidate_ids = candidate_pool["__row_id__"].tolist() if "__row_id__" in candidate_pool.columns else []

    row_assignments = build_row_assignments(
        work_df=work_df,
        target_col=target_col,
        test_row_ids=test_row_ids,
        train_label=train_label,
        test_label=test_label,
        candidate_row_ids=candidate_ids,
    )
    projection_2d = density_context.density_scores.merge(
        row_assignments[["__row_id__", "split_role", "is_candidate", "is_test"]],
        on="__row_id__",
        how="left",
    )
    return TraceArtifacts(
        x_space_features=density_context.x_space.scaled_features.copy(),
        projection_2d=projection_2d,
        row_assignments=row_assignments,
        candidate_pool=candidate_pool.copy(),
        selected_test_rows=selected_test_rows.copy(),
        cluster_assignments=cluster_assignments.copy(),
        density_scores=density_context.density_scores.copy(),
        y_density_curve=density_context.y_density_curve.copy() if density_context.y_density_curve is not None else None,
        neighbor_map=json_ready(neighbor_map) if neighbor_map is not None else None,
        cluster_size_table=cluster_size_table.copy() if cluster_size_table is not None else None,
        metadata={
            "split_strategy": split_strategy,
            "selection_space": density_context.selection_space,
            "x_space_feature_policy": density_context.x_space.feature_policy,
            "x_space_feature_columns": list(density_context.x_space.feature_names),
            "x_space_feature_roles": dict(density_context.x_space.feature_roles),
            "excluded_property_columns": list(density_context.x_space.excluded_property_columns),
            **(metadata or {}),
        },
    )


def prepare_target_extrapolation_split(
    df: pd.DataFrame,
    target_col: str,
    test_ratio: float,
    extrapolation_side: str,
    random_state: int,
    processing_cols: Sequence[str] | None = None,
) -> PreparedSplit:
    if extrapolation_side not in {"low_to_high", "high_to_low"}:
        raise ValueError("extrapolation_side must be one of ['low_to_high', 'high_to_low']")

    work_df = prepare_work_dataframe(df, target_col)
    density_context = build_density_context(
        work_df=work_df,
        target_col=target_col,
        random_state=random_state,
        selection_space="y",
        include_y_curve=False,
        processing_cols=processing_cols,
    )
    ascending = extrapolation_side == "low_to_high"
    ordered = work_df.sort_values(by=target_col, ascending=ascending, kind="stable").reset_index(drop=True)
    train_count = len(work_df) - resolve_test_cap(len(work_df), test_ratio)
    train_count = max(1, min(train_count, len(work_df) - 1))
    ordered_test_ids = ordered.iloc[train_count:]["__row_id__"].tolist()
    ordered_train_ids = ordered.iloc[:train_count]["__row_id__"].tolist()

    train_label, test_label = resolve_split_labels("target_extrapolation", extrapolation_side)
    selected_test_rows = enrich_rows_for_trace(work_df, target_col, density_context.density_scores, ordered_test_ids)
    trace = build_trace_artifacts(
        work_df=work_df,
        target_col=target_col,
        density_context=density_context,
        split_strategy="target_extrapolation",
        train_label=train_label,
        test_label=test_label,
        test_row_ids=ordered_test_ids,
        selected_test_rows=selected_test_rows,
        metadata={"extrapolation_side": extrapolation_side},
    )
    return make_prepared_split(
        work_df=work_df,
        target_col=target_col,
        split_strategy="target_extrapolation",
        train_row_ids=ordered_train_ids,
        test_row_ids=ordered_test_ids,
        trace=trace,
        extrapolation_side=extrapolation_side,
        extra_summary={
            "extrapolation_side": extrapolation_side,
            "selection_space": "y",
            "x_density_bandwidth": density_context.x_density_bandwidth,
            "y_density_bandwidth": density_context.y_density_bandwidth,
        },
    )


def prepare_sparse_single_split(
    df: pd.DataFrame,
    target_col: str,
    split_strategy: str,
    density_space: str,
    test_ratio: float,
    candidate_pool_size: int,
    cluster_count: int,
    samples_per_cluster: int,
    random_state: int,
    kde_bandwidth: float | None = None,
    processing_cols: Sequence[str] | None = None,
) -> PreparedSplit:
    work_df = prepare_work_dataframe(df, target_col)
    samples_per_cluster = ensure_positive_int("sparse_samples_per_cluster", samples_per_cluster)
    density_context = build_density_context(
        work_df=work_df,
        target_col=target_col,
        random_state=random_state,
        selection_space=density_space,
        kde_bandwidth=kde_bandwidth,
        include_y_curve=density_space == "y",
        processing_cols=processing_cols,
    )
    candidate_pool, actual_candidate_pool_size = select_low_density_candidates(
        work_df=work_df,
        target_col=target_col,
        density_context=density_context,
        candidate_pool_size=candidate_pool_size,
    )

    representation = _matrix_for_row_ids(
        density_context.x_space.projection[["projection_x", "projection_y"]].to_numpy(),
        candidate_pool["__row_id__"].tolist(),
        density_context.x_space.row_id_to_index,
    )

    candidate_pool, actual_cluster_count = assign_clusters(
        frame=candidate_pool,
        representation=representation,
        requested_cluster_count=cluster_count,
        random_state=random_state,
        samples_per_cluster=samples_per_cluster,
    )
    selected_parts: List[pd.DataFrame] = []
    for _, cluster_frame in candidate_pool.groupby("cluster_id", sort=True):
        selected_parts.append(
            cluster_frame.sort_values(
                by=["selection_density", target_col, "__row_id__"],
                ascending=[True, True, True],
                kind="stable",
            ).head(samples_per_cluster)
        )
    selected_test_rows = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()

    test_cap = resolve_test_cap(len(work_df), test_ratio)
    if len(selected_test_rows) > test_cap:
        selected_test_rows = selected_test_rows.sort_values(
            by=["selection_density", target_col, "__row_id__"],
            ascending=[True, True, True],
            kind="stable",
        ).head(test_cap).reset_index(drop=True)

    selected_test_ids = selected_test_rows["__row_id__"].tolist()
    selected_set = {int(row_id) for row_id in selected_test_ids}
    train_ids = [int(row_id) for row_id in work_df["__row_id__"].tolist() if row_id not in selected_set]
    train_label, test_label = resolve_split_labels(split_strategy)

    cluster_assignments = candidate_pool[["__row_id__", "cluster_id"]].copy()
    cluster_assignments["cluster_role"] = "candidate_cluster"
    trace = build_trace_artifacts(
        work_df=work_df,
        target_col=target_col,
        density_context=density_context,
        split_strategy=split_strategy,
        train_label=train_label,
        test_label=test_label,
        test_row_ids=selected_test_ids,
        candidate_pool=candidate_pool,
        selected_test_rows=selected_test_rows,
        cluster_assignments=cluster_assignments,
        metadata={
            "candidate_pool_size_actual": actual_candidate_pool_size,
            "cluster_count_actual": actual_cluster_count,
            "samples_per_cluster": samples_per_cluster,
        },
    )
    return make_prepared_split(
        work_df=work_df,
        target_col=target_col,
        split_strategy=split_strategy,
        train_row_ids=train_ids,
        test_row_ids=selected_test_ids,
        trace=trace,
        extra_summary={
            "selection_space": density_space,
            "cluster_space": "projection_2d",
            "sparse_candidate_pool_size_requested": int(candidate_pool_size),
            "sparse_candidate_pool_size_actual": int(actual_candidate_pool_size),
            "sparse_cluster_count_requested": int(cluster_count),
            "sparse_cluster_count_actual": int(actual_cluster_count),
            "sparse_samples_per_cluster": int(samples_per_cluster),
            "sparse_kde_bandwidth": float(density_context.selection_bandwidth),
        },
    )


def prepare_sparse_cluster_split(
    df: pd.DataFrame,
    target_col: str,
    split_strategy: str,
    density_space: str,
    test_ratio: float,
    candidate_pool_size: int,
    cluster_count: int,
    neighbors_per_seed: int,
    random_state: int,
    processing_cols: Sequence[str] | None = None,
) -> PreparedSplit:
    work_df = prepare_work_dataframe(df, target_col)
    neighbors_per_seed = ensure_positive_int("sparse_neighbors_per_seed", neighbors_per_seed)
    density_context = build_density_context(
        work_df=work_df,
        target_col=target_col,
        random_state=random_state,
        selection_space=density_space,
        include_y_curve=density_space == "y",
        processing_cols=processing_cols,
    )
    candidate_pool, actual_candidate_pool_size = select_low_density_candidates(
        work_df=work_df,
        target_col=target_col,
        density_context=density_context,
        candidate_pool_size=candidate_pool_size,
    )
    candidate_projection = _matrix_for_row_ids(
        density_context.x_space.projection[["projection_x", "projection_y"]].to_numpy(),
        candidate_pool["__row_id__"].tolist(),
        density_context.x_space.row_id_to_index,
    )
    candidate_pool, actual_cluster_count = assign_clusters(
        frame=candidate_pool,
        representation=candidate_projection,
        requested_cluster_count=cluster_count,
        random_state=random_state,
    )
    seed_rows = (
        candidate_pool.sort_values(
            by=["selection_density", target_col, "__row_id__"],
            ascending=[True, True, True],
            kind="stable",
        )
        .groupby("cluster_id", sort=True, as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    test_cap = resolve_test_cap(len(work_df), test_ratio)
    selected_parts: List[pd.DataFrame] = []
    neighbor_map: Dict[str, Any] = {}
    seen: set[int] = set()
    full_matrix = density_context.x_space.matrix
    trace_enriched = enrich_rows_for_trace(work_df, target_col, density_context.density_scores)

    for order_idx, seed_row in seed_rows.iterrows():
        seed_row_id = int(seed_row["__row_id__"])
        seed_index = density_context.x_space.row_id_to_index[seed_row_id]
        distances = np.linalg.norm(full_matrix - full_matrix[seed_index], axis=1)
        ranked_indices = np.argsort(distances)
        proposed_ids = [int(work_df.iloc[idx]["__row_id__"]) for idx in ranked_indices[: neighbors_per_seed + 1]]
        actual_ids: List[int] = []
        for row_id in proposed_ids:
            if len(seen) >= test_cap:
                break
            if row_id in seen:
                continue
            seen.add(row_id)
            actual_ids.append(row_id)

        selected_frame = trace_enriched[trace_enriched["__row_id__"].isin(actual_ids)].copy()
        selected_frame["seed_row_id"] = seed_row_id
        selected_frame["cluster_id"] = int(seed_row["cluster_id"])
        selected_frame["selection_order"] = order_idx
        selected_frame["distance_to_seed"] = selected_frame["__row_id__"].map(
            {
                int(work_df.iloc[idx]["__row_id__"]): float(distances[idx])
                for idx in ranked_indices[: neighbors_per_seed + 1]
            }
        )
        if not selected_frame.empty:
            selected_parts.append(selected_frame)

        neighbor_map[str(seed_row_id)] = {
            "seed_row_id": seed_row_id,
            "cluster_id": int(seed_row["cluster_id"]),
            "candidate_neighbor_row_ids": proposed_ids,
            "selected_neighbor_row_ids": actual_ids,
        }

    selected_test_rows = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    selected_test_ids = selected_test_rows["__row_id__"].tolist()
    selected_set = {int(row_id) for row_id in selected_test_ids}
    train_ids = [int(row_id) for row_id in work_df["__row_id__"].tolist() if row_id not in selected_set]
    train_label, test_label = resolve_split_labels(split_strategy)

    cluster_assignments = candidate_pool[["__row_id__", "cluster_id"]].copy()
    cluster_assignments["cluster_role"] = "candidate_cluster"
    if not selected_test_rows.empty:
        expansion_assignments = selected_test_rows[["__row_id__", "cluster_id", "seed_row_id", "distance_to_seed"]].copy()
        expansion_assignments["cluster_role"] = "neighbor_expansion"
        cluster_assignments = pd.concat([cluster_assignments, expansion_assignments], ignore_index=True, sort=False)

    trace = build_trace_artifacts(
        work_df=work_df,
        target_col=target_col,
        density_context=density_context,
        split_strategy=split_strategy,
        train_label=train_label,
        test_label=test_label,
        test_row_ids=selected_test_ids,
        candidate_pool=candidate_pool,
        selected_test_rows=selected_test_rows,
        cluster_assignments=cluster_assignments,
        neighbor_map=neighbor_map,
        metadata={
            "candidate_pool_size_actual": actual_candidate_pool_size,
            "cluster_count_actual": actual_cluster_count,
            "neighbors_per_seed": neighbors_per_seed,
        },
    )
    return make_prepared_split(
        work_df=work_df,
        target_col=target_col,
        split_strategy=split_strategy,
        train_row_ids=train_ids,
        test_row_ids=selected_test_ids,
        trace=trace,
        extra_summary={
            "selection_space": density_space,
            "cluster_space": "projection_2d",
            "neighbor_space": "x_feature_euclidean",
            "sparse_candidate_pool_size_requested": int(candidate_pool_size),
            "sparse_candidate_pool_size_actual": int(actual_candidate_pool_size),
            "sparse_cluster_count_requested": int(cluster_count),
            "sparse_cluster_count_actual": int(actual_cluster_count),
            "sparse_neighbors_per_seed": int(neighbors_per_seed),
            "sparse_kde_bandwidth": float(density_context.selection_bandwidth),
        },
    )


def prepare_loco_folds(
    df: pd.DataFrame,
    target_col: str,
    cluster_count: int,
    random_state: int,
    processing_cols: Sequence[str] | None = None,
) -> List[PreparedFold]:
    work_df = prepare_work_dataframe(df, target_col)
    cluster_count = ensure_positive_int("loco_cluster_count", cluster_count)
    density_context = build_density_context(
        work_df=work_df,
        target_col=target_col,
        random_state=random_state,
        selection_space="x",
        include_y_curve=False,
        processing_cols=processing_cols,
    )
    projection_matrix = density_context.x_space.matrix
    effective_cluster_count = min(cluster_count, len(work_df))
    if effective_cluster_count == 1:
        labels = np.zeros(len(work_df), dtype=int)
    else:
        model = KMeans(n_clusters=effective_cluster_count, random_state=random_state, n_init=10)
        labels = model.fit_predict(projection_matrix)

    cluster_assignments = pd.DataFrame(
        {
            "__row_id__": work_df["__row_id__"].to_numpy(),
            "cluster_id": labels,
            "cluster_role": "loco_cluster",
        }
    )
    cluster_size_table = (
        cluster_assignments.groupby("cluster_id", as_index=False)
        .size()
        .rename(columns={"size": "cluster_size"})
        .sort_values("cluster_id", kind="stable")
        .reset_index(drop=True)
    )

    folds: List[PreparedFold] = []
    train_label, test_label = resolve_split_labels("loco")
    for fold_index, cluster_id in enumerate(cluster_size_table["cluster_id"].tolist()):
        test_ids = cluster_assignments.loc[cluster_assignments["cluster_id"] == cluster_id, "__row_id__"].tolist()
        train_ids = cluster_assignments.loc[cluster_assignments["cluster_id"] != cluster_id, "__row_id__"].tolist()
        selected_test_rows = enrich_rows_for_trace(work_df, target_col, density_context.density_scores, test_ids)
        selected_test_rows["held_out_cluster_id"] = int(cluster_id)
        trace = build_trace_artifacts(
            work_df=work_df,
            target_col=target_col,
            density_context=density_context,
            split_strategy="loco",
            train_label=train_label,
            test_label=test_label,
            test_row_ids=test_ids,
            candidate_pool=selected_test_rows,
            selected_test_rows=selected_test_rows,
            cluster_assignments=cluster_assignments,
            cluster_size_table=cluster_size_table,
            metadata={"fold_index": fold_index, "held_out_cluster_id": int(cluster_id)},
        )
        split = make_prepared_split(
            work_df=work_df,
            target_col=target_col,
            split_strategy="loco",
            train_row_ids=train_ids,
            test_row_ids=test_ids,
            trace=trace,
            train_label=train_label,
            test_label=test_label,
            extra_summary={
                "selection_space": "x",
                "loco_cluster_count_requested": int(cluster_count),
                "loco_cluster_count_actual": int(effective_cluster_count),
                "held_out_cluster_id": int(cluster_id),
                "fold_index": int(fold_index),
            },
        )
        folds.append(
            PreparedFold(
                fold_index=fold_index,
                held_out_cluster_id=int(cluster_id),
                split=split,
                metadata={
                    "fold_index": int(fold_index),
                    "held_out_cluster_id": int(cluster_id),
                    "cluster_size": int(len(test_ids)),
                },
            )
        )
    return folds


def prepare_random_cv_baseline_folds(
    df: pd.DataFrame,
    target_col: str,
    num_folds: int,
    random_state: int,
    processing_cols: Sequence[str] | None = None,
) -> List[PreparedFold]:
    work_df = prepare_work_dataframe(df, target_col)
    num_folds = ensure_positive_int("baseline_num_folds", num_folds)
    if num_folds < 2:
        raise ValueError("baseline_num_folds must be at least 2")
    if num_folds > len(work_df):
        raise ValueError("baseline_num_folds cannot exceed the number of valid rows")

    density_context = build_density_context(
        work_df=work_df,
        target_col=target_col,
        random_state=random_state,
        selection_space="x",
        include_y_curve=False,
        processing_cols=processing_cols,
    )

    folds: List[PreparedFold] = []
    train_label, test_label = resolve_split_labels("random_cv_baseline")
    splitter = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)

    row_ids = work_df["__row_id__"].to_numpy()
    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(row_ids)):
        train_ids = row_ids[train_idx].tolist()
        test_ids = row_ids[test_idx].tolist()
        selected_test_rows = enrich_rows_for_trace(work_df, target_col, density_context.density_scores, test_ids)
        selected_test_rows["outer_fold_index"] = int(fold_index)
        selected_test_rows["split_mode"] = "random_cv"
        trace = build_trace_artifacts(
            work_df=work_df,
            target_col=target_col,
            density_context=density_context,
            split_strategy="random_cv_baseline",
            train_label=train_label,
            test_label=test_label,
            test_row_ids=test_ids,
            candidate_pool=selected_test_rows,
            selected_test_rows=selected_test_rows,
            metadata={
                "fold_index": int(fold_index),
                "outer_fold_index": int(fold_index),
                "outer_fold_count": int(num_folds),
                "split_mode": "random_cv",
            },
        )
        split = make_prepared_split(
            work_df=work_df,
            target_col=target_col,
            split_strategy="random_cv_baseline",
            train_row_ids=train_ids,
            test_row_ids=test_ids,
            trace=trace,
            train_label=train_label,
            test_label=test_label,
            extra_summary={
                "selection_space": "x",
                "outer_fold_count": int(num_folds),
                "outer_fold_index": int(fold_index),
                "fold_index": int(fold_index),
                "split_mode": "random_cv",
            },
        )
        folds.append(
            PreparedFold(
                fold_index=fold_index,
                held_out_cluster_id=-1,
                split=split,
                metadata={
                    "fold_index": int(fold_index),
                    "outer_fold_index": int(fold_index),
                    "outer_fold_count": int(num_folds),
                    "split_mode": "random_cv",
                    "test_size": int(len(test_ids)),
                    "train_size": int(len(train_ids)),
                },
            )
        )
    return folds


def _write_table(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8")


def _make_empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _plot_projection_categories(
    projection_df: pd.DataFrame,
    output_path: Path,
    category_col: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    if projection_df.empty or category_col not in projection_df.columns:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        categories = list(pd.Series(projection_df[category_col]).fillna("unassigned").unique())
        cmap = plt.cm.get_cmap("tab20", max(len(categories), 1))
        for idx, category in enumerate(categories):
            mask = projection_df[category_col].fillna("unassigned") == category
            ax.scatter(
                projection_df.loc[mask, "projection_x"],
                projection_df.loc[mask, "projection_y"],
                s=20,
                alpha=0.8,
                label=str(category),
                color=cmap(idx),
            )
        if len(categories) <= 12:
            ax.legend(loc="best", fontsize=8)
    ax.set_xlabel("projection_x")
    ax.set_ylabel("projection_y")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_projection_density(
    projection_df: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    if projection_df.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        scatter = ax.scatter(
            projection_df["projection_x"],
            projection_df["projection_y"],
            c=projection_df["selection_density"],
            cmap="viridis",
            s=24,
            alpha=0.9,
        )
        fig.colorbar(scatter, ax=ax, label="selection_density")
    ax.set_xlabel("projection_x")
    ax.set_ylabel("projection_y")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_projection_highlight(
    projection_df: pd.DataFrame,
    highlighted_ids: Sequence[int],
    output_path: Path,
    title: str,
    highlight_label: str,
) -> None:
    highlighted_ids = [int(row_id) for row_id in highlighted_ids]
    fig, ax = plt.subplots(figsize=(8, 6))
    if projection_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        ax.scatter(
            projection_df["projection_x"],
            projection_df["projection_y"],
            s=16,
            alpha=0.25,
            color="lightgray",
            label="all_rows",
        )
        if highlighted_ids:
            highlighted = projection_df[projection_df["__row_id__"].isin(highlighted_ids)]
            ax.scatter(
                highlighted["projection_x"],
                highlighted["projection_y"],
                s=28,
                alpha=0.95,
                color="tab:red",
                label=highlight_label,
            )
            ax.legend(loc="best", fontsize=8)
    ax.set_xlabel("projection_x")
    ax.set_ylabel("projection_y")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_y_density_curve(y_density_curve: pd.DataFrame, target_col: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    if y_density_curve.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        ax.plot(y_density_curve[target_col], y_density_curve["density"], color="tab:blue")
    ax.set_xlabel(target_col)
    ax.set_ylabel("density")
    ax.set_title("Target Density Curve")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_cluster_sizes(cluster_size_table: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    if cluster_size_table.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        ax.bar(cluster_size_table["cluster_id"].astype(str), cluster_size_table["cluster_size"], color="tab:green")
    ax.set_xlabel("cluster_id")
    ax.set_ylabel("cluster_size")
    ax.set_title("LOCO Cluster Sizes")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_trace_artifacts(trace: TraceArtifacts, output_dir: str | Path, target_col: str) -> Dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files: Dict[str, str] = {}

    x_space_path = output_path / "x_space_features.parquet"
    trace.x_space_features.to_parquet(x_space_path, index=False)
    files["x_space_features"] = str(x_space_path)

    projection_path = output_path / "projection_2d.csv"
    _write_table(projection_path, trace.projection_2d)
    files["projection_2d"] = str(projection_path)

    row_assignments_path = output_path / "row_assignments.csv"
    _write_table(row_assignments_path, trace.row_assignments)
    files["row_assignments"] = str(row_assignments_path)

    candidate_pool_path = output_path / "candidate_pool.csv"
    _write_table(
        candidate_pool_path,
        trace.candidate_pool if not trace.candidate_pool.empty else _make_empty_frame(["__row_id__"]),
    )
    files["candidate_pool"] = str(candidate_pool_path)

    selected_test_rows_path = output_path / "selected_test_rows.csv"
    _write_table(selected_test_rows_path, trace.selected_test_rows)
    files["selected_test_rows"] = str(selected_test_rows_path)

    cluster_assignments_path = output_path / "cluster_assignments.csv"
    _write_table(
        cluster_assignments_path,
        trace.cluster_assignments if not trace.cluster_assignments.empty else _make_empty_frame(["__row_id__", "cluster_id"]),
    )
    files["cluster_assignments"] = str(cluster_assignments_path)

    density_scores_path = output_path / "density_scores.csv"
    _write_table(density_scores_path, trace.density_scores)
    files["density_scores"] = str(density_scores_path)

    if trace.y_density_curve is not None:
        y_curve_csv_path = output_path / "y_density_curve.csv"
        _write_table(y_curve_csv_path, trace.y_density_curve)
        files["y_density_curve"] = str(y_curve_csv_path)

        y_curve_png_path = output_path / "y_density_curve.png"
        _plot_y_density_curve(trace.y_density_curve, target_col, y_curve_png_path)
        files["y_density_curve_plot"] = str(y_curve_png_path)

    if trace.neighbor_map is not None:
        neighbor_map_path = output_path / "neighbor_map.json"
        save_json(neighbor_map_path, trace.neighbor_map)
        files["neighbor_map"] = str(neighbor_map_path)

    overview_png = output_path / "projection_overview.png"
    _plot_projection_categories(trace.projection_2d, overview_png, "split_role", "Projection Overview")
    files["projection_overview"] = str(overview_png)

    by_density_png = output_path / "projection_by_density.png"
    _plot_projection_density(trace.projection_2d, by_density_png, "Projection by Density")
    files["projection_by_density"] = str(by_density_png)

    candidates_png = output_path / "projection_candidates.png"
    candidate_ids = trace.candidate_pool["__row_id__"].tolist() if "__row_id__" in trace.candidate_pool.columns else []
    _plot_projection_highlight(trace.projection_2d, candidate_ids, candidates_png, "Projection Candidates", "candidate_pool")
    files["projection_candidates"] = str(candidates_png)

    test_selection_png = output_path / "projection_test_selection.png"
    _plot_projection_highlight(
        trace.projection_2d,
        trace.selected_test_rows["__row_id__"].tolist(),
        test_selection_png,
        "Projection Test Selection",
        "selected_test",
    )
    files["projection_test_selection"] = str(test_selection_png)

    clusters_png = output_path / "projection_clusters.png"
    cluster_projection = trace.projection_2d.copy()
    if not trace.cluster_assignments.empty and "__row_id__" in trace.cluster_assignments.columns:
        cluster_projection = cluster_projection.merge(
            trace.cluster_assignments.drop_duplicates("__row_id__")[["__row_id__", "cluster_id"]],
            on="__row_id__",
            how="left",
        )
        cluster_col = "cluster_id"
    else:
        cluster_col = "split_role"
    _plot_projection_categories(cluster_projection, clusters_png, cluster_col, "Projection Clusters")
    files["projection_clusters"] = str(clusters_png)

    if trace.cluster_size_table is not None:
        cluster_size_png = output_path / "cluster_size_bar.png"
        _plot_cluster_sizes(trace.cluster_size_table, cluster_size_png)
        files["cluster_size_bar"] = str(cluster_size_png)

    manifest_path = output_path / "split_manifest.json"
    save_json(
        manifest_path,
        {
            "metadata": trace.metadata,
            "files": files,
        },
    )
    files["split_manifest"] = str(manifest_path)
    return files


def save_prepared_split(prepared_split: PreparedSplit, output_dir: str | Path) -> Dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_path = output_path / f"{prepared_split.train_label}.csv"
    test_path = output_path / f"{prepared_split.test_label}.csv"
    summary_path = output_path / "split_summary.json"

    prepared_split.train_df.to_csv(train_path, index=False, encoding="utf-8")
    prepared_split.test_df.to_csv(test_path, index=False, encoding="utf-8")
    save_json(summary_path, prepared_split.summary)

    trace_files: Dict[str, str] = {}
    if prepared_split.trace is not None:
        trace_files = write_trace_artifacts(
            trace=prepared_split.trace,
            output_dir=output_path / "trace",
            target_col=prepared_split.target_col,
        )

    artifacts = {
        "train_file": str(train_path),
        "test_file": str(test_path),
        "summary_file": str(summary_path),
    }
    if trace_files:
        artifacts["trace_dir"] = str(output_path / "trace")
        artifacts["trace_manifest"] = trace_files["split_manifest"]
    return artifacts

    perplexity = min(30, row_count - 1, max(2, row_count // 3))
    try:
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        )
        return tsne.fit_transform(matrix)
    except Exception:
        first = matrix[:, 0] if feature_count >= 1 else np.arange(row_count, dtype=float)
        second = matrix[:, 1] if feature_count >= 2 else np.zeros(row_count, dtype=float)
        return np.column_stack([first, second])
