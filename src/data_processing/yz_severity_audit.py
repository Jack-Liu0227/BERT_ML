"""Utilities for auditing joint Y/Z OOD severity measurements.

The functions in this module are deliberately independent of the plotting
pipeline.  They operate on pandas data frames so the audit can be reproduced
from the exported sample-level long table without model- or sample-count
weighting leaking into split-level statistics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Sequence
import re

import numpy as np
import pandas as pd

from src.data_processing.compute_wx_mixed_ot import solve_exact_ot


_SPLIT_KEYS = [
    "task_key",
    "task_id",
    "alloy_family",
    "dataset_name",
    "property",
    "method_short",
    "source_split_dir",
    "fold_id",
    "space",
]

_DRIVER_SAFE_METADATA = {
    "scope",
    "method",
    "target_value",
    "true_value",
    "test_set",
    "join_mode",
    "wx_mode",
}


def _is_missing_scalar(value: object) -> bool:
    """Return whether a scalar-like value is missing without pd.NA ambiguity."""

    if value is None or value is pd.NA:
        return True
    try:
        missing = pd.isna(value)
        return bool(missing) if np.ndim(missing) == 0 else False
    except (TypeError, ValueError):
        return False


def _slug(value: object) -> str:
    """Return a stable lower-case identifier for a source/model label."""

    text = "" if _is_missing_scalar(value) else str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_") or "unknown"


def classify_source(
    source_split_dir: object, model: object = None
) -> dict[str, object]:
    """Classify an exported split into its experiment/source family.

    Path markers take precedence over the model label because a single split
    may have been evaluated by several downstream regressors.
    """

    path = "" if _is_missing_scalar(source_split_dir) else str(source_split_dir)
    path = path.replace("\\", "/")
    path_segments = [_slug(segment) for segment in path.split("/") if segment]
    model_slug = _slug(model)

    def path_has(marker: str) -> bool:
        return any(
            segment == marker or marker in segment.split("_")
            for segment in path_segments
        )

    def path_contains(marker: str) -> bool:
        return any(marker in segment for segment in path_segments)

    # Only normalized path segments are inspected here.  In particular, a
    # workspace directory such as ``BERT_ML`` is not itself a model source.
    if path_has("tradition") or path_has("traditional"):
        family, name = "traditional", "traditional"
    elif path_contains("fewshot") or path_contains("few_shot"):
        family = "fewshot"
        name = model_slug if model_slug != "unknown" else "fewshot"
    elif path_contains("tabpfn"):
        family = "tabpfn"
        if path_contains("numeric"):
            name = "tabpfn_numeric"
        elif path_contains("text"):
            name = "tabpfn_text"
        else:
            name = "tabpfn"
    elif path_has("matscibert"):
        family, name = "bert", "matscibert"
    elif path_has("steelbert"):
        family, name = "bert", "steelbert"
    elif path_has("scibert"):
        family, name = "bert", "scibert"
    elif path_has("llmprop"):
        family, name = "llmprop", "llmprop"
    elif "fewshot" in model_slug or "few_shot" in model_slug:
        family, name = "fewshot", model_slug
    elif "tabpfn" in model_slug:
        family = "tabpfn"
        name = (
            "tabpfn_numeric"
            if "numeric" in model_slug
            else ("tabpfn_text" if "text" in model_slug else "tabpfn")
        )
    elif model_slug in {"matscibert", "steelbert", "scibert"}:
        family, name = "bert", model_slug
    elif model_slug == "llmprop":
        family, name = "llmprop", "llmprop"
    else:
        family = "other"
        name = model_slug if model_slug != "unknown" else _slug(Path(path).parent.name)

    return {
        "source_family": family,
        "source_name": name,
        "is_canonical": family == "traditional",
    }


def collapse_split_rows(rows: pd.DataFrame, *, atol: float = 1e-12) -> pd.DataFrame:
    """Collapse repeated model/sample rows to one record per split and space.

    All finite ``split_w`` values within a split-space must agree.  A conflict
    indicates that rows which do not represent the same split were combined
    upstream and is therefore raised rather than silently averaged.
    """

    required = {"source_split_dir", "fold_id", "space", "split_w"}
    missing = required.difference(rows.columns)
    if missing:
        raise KeyError(f"missing required columns: {sorted(missing)}")

    group_keys = [column for column in _SPLIT_KEYS if column in rows.columns]
    records: list[dict[str, object]] = []
    for key, group in rows.groupby(group_keys, dropna=False, sort=False):
        key_values = key if isinstance(key, tuple) else (key,)
        record = dict(zip(group_keys, key_values))
        split_values = pd.to_numeric(group["split_w"], errors="coerce").dropna()
        if split_values.empty:
            split_w = np.nan
            split_range = np.nan
        else:
            split_w = float(split_values.iloc[0])
            split_range = float(split_values.max() - split_values.min())
            if not np.isclose(split_range, 0.0, atol=atol, rtol=atol):
                label = ", ".join(f"{k}={record[k]!r}" for k in group_keys)
                raise ValueError(f"conflicting split_w values for {label}")

        path_source = classify_source(record.get("source_split_dir"), None)
        source = path_source
        if path_source["source_family"] == "other" and "model" in group:
            model_sources = [
                classify_source(record.get("source_split_dir"), model)
                for model in group["model"].drop_duplicates().tolist()
            ]
            source_names = {item["source_name"] for item in model_sources}
            if len(source_names) == 1:
                source = model_sources[0]
            elif len(source_names) > 1:
                source = {
                    "source_family": "other",
                    "source_name": "ambiguous_models",
                    "is_canonical": False,
                }
        record.update(source)
        record["split_variant"] = _infer_split_variant(record)
        record["split_w"] = split_w
        record["model_count"] = (
            int(group["model"].nunique(dropna=True)) if "model" in group else 0
        )
        record["sample_count"] = (
            int(group["ID"].nunique(dropna=True)) if "ID" in group else len(group)
        )
        record["raw_row_count"] = int(len(group))
        record["split_w_range"] = split_range
        records.append(record)

    return pd.DataFrame.from_records(records)


def _infer_split_variant(record: dict[str, object]) -> str:
    method = _slug(record.get("method_short"))
    source_path = record.get("source_split_dir", "")
    path = "" if _is_missing_scalar(source_path) else str(source_path)
    path_segments = [
        _slug(segment) for segment in path.replace("\\", "/").split("/") if segment
    ]
    if method == "loco":
        if "loco_k5" in path_segments:
            return "loco_k5"
        if "loco" in path_segments:
            return "loco"
    if method in {"randcv", "rand_cv", "loco", "extra", "extrapolation"}:
        return method
    parent = Path(path).parent.name
    return _slug(parent) if parent else method


def build_source_severity_audit(
    collapsed: pd.DataFrame,
    *,
    ry_threshold: float = 8.0,
    rz_threshold: float = 5.0,
) -> pd.DataFrame:
    """Build source/fold-level Wy, Wz, Ry and Rz audit records.

    RandCV baselines are averaged over folds separately for each task and
    source, then matched only to protocols from that same source.
    """

    frame = collapsed.copy()
    for column in ("source_family", "source_name", "is_canonical"):
        if column not in frame:
            classified = frame.apply(
                lambda row: classify_source(
                    row.get("source_split_dir"), row.get("model")
                ),
                axis=1,
            )
            frame[column] = [item[column] for item in classified]

    identity_candidates = [
        "task_key",
        "task_id",
        "alloy_family",
        "dataset_name",
        "property",
        "source_family",
        "source_name",
        "is_canonical",
    ]
    source_keys = [column for column in identity_candidates if column in frame.columns]
    fold_keys = source_keys + [
        column
        for column in ("method_short", "fold_id", "source_split_dir", "split_variant")
        if column in frame.columns
    ]

    fold_audit = (
        frame.groupby(fold_keys + ["space"], dropna=False, sort=False)["split_w"]
        .first()
        .unstack("space")
        .reset_index()
        .rename(columns={"Y-space": "Wy", "Z-space": "Wz"})
    )
    for column in ("Wy", "Wz"):
        if column not in fold_audit:
            fold_audit[column] = np.nan

    protocol_keys = source_keys + [
        column
        for column in ("method_short", "split_variant")
        if column in fold_audit.columns
    ]
    protocol = (
        fold_audit.groupby(protocol_keys, dropna=False, sort=False)
        .agg(
            protocol_Wy=("Wy", "mean"),
            protocol_Wz=("Wz", "mean"),
            protocol_fold_count=("fold_id", "nunique")
            if "fold_id" in fold_audit
            else ("Wy", "size"),
        )
        .reset_index()
    )

    baseline_rows = fold_audit[
        fold_audit["method_short"].astype(str).str.casefold().eq("randcv")
    ]
    baseline = (
        baseline_rows.groupby(source_keys, dropna=False)[["Wy", "Wz"]]
        .mean()
        .reset_index()
        .rename(columns={"Wy": "Wy_random", "Wz": "Wz_random"})
    )
    protocol = protocol.merge(
        baseline, on=source_keys, how="left", validate="many_to_one"
    )
    audit = fold_audit.merge(
        protocol, on=protocol_keys, how="left", validate="many_to_one"
    )
    for column in ("Wy_random", "Wz_random"):
        if column not in audit:
            audit[column] = np.nan

    baseline_missing = audit[["Wy_random", "Wz_random"]].isna().any(axis=1)
    baseline_nonpositive = (audit[["Wy_random", "Wz_random"]] <= 0).any(axis=1)
    audit["baseline_status"] = np.select(
        [baseline_missing, baseline_nonpositive],
        ["missing", "nonpositive"],
        default="ok",
    )
    audit["Ry"] = _safe_ratio(audit["Wy"], audit["Wy_random"])
    audit["Rz"] = _safe_ratio(audit["Wz"], audit["Wz_random"])
    audit["protocol_Ry"] = _safe_ratio(audit["protocol_Wy"], audit["Wy_random"])
    audit["protocol_Rz"] = _safe_ratio(audit["protocol_Wz"], audit["Wz_random"])
    audit["protocol_high_Ry"] = audit["protocol_Ry"] > ry_threshold
    audit["protocol_high_Rz"] = audit["protocol_Rz"] > rz_threshold
    audit["protocol_joint_high"] = audit["protocol_high_Ry"] & audit["protocol_high_Rz"]
    return audit


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = pd.to_numeric(numerator, errors="coerce") / pd.to_numeric(
        denominator, errors="coerce"
    )
    return result.where(pd.to_numeric(denominator, errors="coerce") > 0)


def legacy_severity_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the plotting pipeline's mixed-source, sample-weighted summary.

    This compatibility view deliberately averages the repeated sample/model
    rows directly.  It exists to verify the published 63-row export; new audit
    results should instead use :func:`collapse_split_rows`.
    """

    required = {"method_short", "space", "split_w"}
    missing = required.difference(rows.columns)
    if missing:
        raise KeyError(f"missing required columns: {sorted(missing)}")

    identity_candidates = [
        "task_key",
        "task_id",
        "alloy_family",
        "dataset_name",
        "property",
    ]
    identity = [column for column in identity_candidates if column in rows.columns]
    fold_keys = identity + ["method_short", "space"]
    if "fold_id" in rows.columns:
        fold_keys.append("fold_id")
    fold_w = (
        rows.groupby(fold_keys, dropna=False, sort=False)["split_w"]
        .mean()
        .reset_index()
    )
    grouped = (
        fold_w.groupby(identity + ["method_short", "space"], dropna=False, sort=False)[
            "split_w"
        ]
        .mean()
        .unstack("space")
        .reset_index()
        .rename(columns={"Y-space": "Wy", "Z-space": "Wz"})
    )
    for column in ("Wy", "Wz"):
        if column not in grouped:
            grouped[column] = np.nan

    random = grouped[grouped["method_short"].astype(str).str.casefold().eq("randcv")][
        identity + ["Wy", "Wz"]
    ].rename(columns={"Wy": "Wy_random", "Wz": "Wz_random"})
    summary = grouped.merge(random, on=identity, how="left", validate="many_to_one")
    summary["Ry"] = _safe_ratio(summary["Wy"], summary["Wy_random"])
    summary["Rz"] = _safe_ratio(summary["Wz"], summary["Wz_random"])
    valid = (
        summary["Wy"].notna()
        & summary["Wz"].notna()
        & pd.to_numeric(summary["Wy_random"], errors="coerce").gt(0)
        & pd.to_numeric(summary["Wz_random"], errors="coerce").gt(0)
    )
    return summary.loc[valid].reset_index(drop=True)


def parse_feature_columns(value: object, frame: pd.DataFrame) -> list[str]:
    """Parse a semicolon-separated inventory field and retain real columns."""

    if _is_missing_scalar(value):
        return []
    requested = [column.strip() for column in str(value).split(";") if column.strip()]
    return list(
        dict.fromkeys(column for column in requested if column in frame.columns)
    )


def validate_split_integrity(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target_col: str,
    feature_columns: Sequence[str],
    id_col: str = "ID",
    expect_target_extrapolation: bool = False,
) -> dict[str, object]:
    """Report leakage, duplication, target-range and traceability checks."""

    checked_columns = [
        column for column in feature_columns if column in train and column in test
    ]
    if target_col not in train or target_col not in test:
        raise KeyError(f"target column is missing: {target_col}")

    train_ids = train[id_col] if id_col in train else pd.Series(dtype=object)
    test_ids = test[id_col] if id_col in test else pd.Series(dtype=object)
    overlap = set(train_ids.dropna().tolist()).intersection(test_ids.dropna().tolist())
    duplicate_subset = checked_columns + [target_col]
    exact_record_overlap = (
        train[duplicate_subset]
        .drop_duplicates()
        .merge(
            test[duplicate_subset].drop_duplicates(),
            on=duplicate_subset,
            how="inner",
        )
    )
    train_target = pd.to_numeric(train[target_col], errors="coerce")
    test_target = pd.to_numeric(test[target_col], errors="coerce")
    train_min, train_max = train_target.min(), train_target.max()
    test_min, test_max = test_target.min(), test_target.max()
    ordered = bool(
        expect_target_extrapolation
        and pd.notna(train_max)
        and pd.notna(test_min)
        and float(train_max) < float(test_min)
    )
    return {
        "train_row_count": int(len(train)),
        "test_row_count": int(len(test)),
        "train_unique_id_count": int(train_ids.nunique(dropna=True)),
        "test_unique_id_count": int(test_ids.nunique(dropna=True)),
        "train_duplicate_id_count": int(
            train_ids.dropna().duplicated(keep="first").sum()
        ),
        "test_duplicate_id_count": int(
            test_ids.dropna().duplicated(keep="first").sum()
        ),
        "train_test_id_overlap_count": int(len(overlap)),
        "train_test_id_overlap": ";".join(sorted(map(str, overlap))),
        "train_test_exact_record_overlap_count": int(len(exact_record_overlap)),
        "train_exact_duplicate_count": int(
            train.duplicated(duplicate_subset, keep="first").sum()
        ),
        "test_exact_duplicate_count": int(
            test.duplicated(duplicate_subset, keep="first").sum()
        ),
        "train_target_min": float(train_min) if pd.notna(train_min) else np.nan,
        "train_target_max": float(train_max) if pd.notna(train_max) else np.nan,
        "test_target_min": float(test_min) if pd.notna(test_min) else np.nan,
        "test_target_max": float(test_max) if pd.notna(test_max) else np.nan,
        "extra_train_max_lt_test_min": ordered,
        "target_extrapolation_expected": bool(expect_target_extrapolation),
        "feature_column_count": int(len(checked_columns)),
    }


def prepare_representation(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    representation: str,
    pca_components: int = 20,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Prepare standardized raw, PCA or deterministic two-dimensional t-SNE data."""

    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    columns = [
        column for column in feature_columns if column in train and column in test
    ]
    if not columns:
        raise ValueError("no shared feature columns")
    train_numeric = (
        train[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    )
    test_numeric = (
        test[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    )
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_imputed = imputer.fit_transform(train_numeric)
    test_imputed = imputer.transform(test_numeric)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_imputed)
    test_scaled = scaler.transform(test_imputed)

    name = str(representation).strip().casefold()
    metadata: dict[str, object] = {
        "representation": name,
        "feature_count": len(columns),
        "feature_columns": list(columns),
        "imputation_strategy": "median",
        "random_state": int(random_state),
    }
    if name in {"raw", "standardized_raw", "standardized"}:
        metadata["effective_components"] = int(train_scaled.shape[1])
        return train_scaled, test_scaled, metadata

    if name in {"pca", "pca20"}:
        if pca_components <= 0:
            raise ValueError("pca_components must be positive")
        effective = min(
            int(pca_components), train_scaled.shape[1], len(train_scaled) - 1
        )
        if effective < 1:
            raise ValueError("PCA requires at least two training samples")
        pca = PCA(n_components=effective, random_state=random_state)
        metadata["effective_components"] = effective
        return pca.fit_transform(train_scaled), pca.transform(test_scaled), metadata

    if name in {"tsne", "t-sne"}:
        combined = np.vstack([train_scaled, test_scaled])
        if len(combined) < 3:
            raise ValueError("t-SNE requires at least three combined samples")
        # t-SNE has no transform method, so this diagnostic proxy is fitted
        # jointly and then separated back into the original train/test rows.
        perplexity = min(30.0, max(1.0, (len(combined) - 1) / 3.0))
        embedded = TSNE(
            n_components=2,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
        ).fit_transform(combined)
        metadata.update({"effective_components": 2, "perplexity": float(perplexity)})
        return embedded[: len(train)], embedded[len(train) :], metadata

    raise ValueError(f"unsupported representation: {representation}")


def prepare_task_global_representations(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    pca_components: int = 20,
    minimum_relevant_train_size: int,
    tsne_seeds: Sequence[int] = (0, 1, 2, 3, 4),
) -> dict[tuple[str, int | None], tuple[np.ndarray, dict[str, object]]]:
    """Fit fixed task-global diagnostic geometries for permutation audits."""

    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    columns = [column for column in feature_columns if column in frame]
    if not columns:
        raise ValueError("no feature columns are present in the task master table")
    if len(frame) < 3:
        raise ValueError("task-global representations require at least three samples")
    if pca_components <= 0:
        raise ValueError("pca_components must be positive")
    minimum_relevant_train_size = int(minimum_relevant_train_size)
    if minimum_relevant_train_size < 2 or minimum_relevant_train_size > len(frame):
        raise ValueError("minimum_relevant_train_size must be between 2 and n_samples")

    numeric = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    imputed = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(
        numeric
    )
    standardized = StandardScaler().fit_transform(imputed)
    common: dict[str, object] = {
        "feature_count": int(len(columns)),
        "feature_columns": list(columns),
        "imputation_strategy": "median",
        "preprocessing_scope": "task_global",
    }
    results: dict[tuple[str, int | None], tuple[np.ndarray, dict[str, object]]] = {}
    results[("standardized_raw", None)] = (
        standardized,
        {
            **common,
            "representation": "standardized_raw",
            "representation_seed": None,
            "effective_dimension": int(standardized.shape[1]),
            "transductive_proxy": False,
        },
    )

    effective = min(
        int(pca_components),
        int(standardized.shape[1]),
        minimum_relevant_train_size - 1,
    )
    if effective < 1:
        raise ValueError("PCA requires at least one effective component")
    pca_values = PCA(n_components=effective, random_state=0).fit_transform(standardized)
    results[("pca20", None)] = (
        pca_values,
        {
            **common,
            "representation": "pca20",
            "representation_seed": None,
            "effective_dimension": int(effective),
            "transductive_proxy": False,
        },
    )

    perplexity = min(30.0, max(1.0, (len(frame) - 1) / 3.0))
    for seed in dict.fromkeys(int(value) for value in tsne_seeds):
        embedded = TSNE(
            n_components=2,
            random_state=seed,
            init="pca" if standardized.shape[1] >= 2 else "random",
            learning_rate="auto",
            perplexity=perplexity,
        ).fit_transform(standardized)
        embedded = StandardScaler().fit_transform(embedded)
        results[("tsne_2d_proxy", seed)] = (
            embedded,
            {
                **common,
                "representation": "tsne_2d_proxy",
                "representation_seed": seed,
                "effective_dimension": 2,
                "perplexity": float(perplexity),
                "transductive_proxy": True,
            },
        )
    return results


def standardized_target_w1(
    train: pd.DataFrame, test: pd.DataFrame, target_col: str
) -> float:
    """Compute target W1 after scaling both samples by the training target."""

    train_values = (
        pd.to_numeric(train[target_col], errors="coerce").dropna().to_numpy(dtype=float)
    )
    test_values = (
        pd.to_numeric(test[target_col], errors="coerce").dropna().to_numpy(dtype=float)
    )
    if train_values.size == 0 or test_values.size == 0:
        return np.nan
    if not np.isfinite(train_values).all() or not np.isfinite(test_values).all():
        return np.nan
    center = float(np.mean(train_values))
    scale = float(np.std(train_values, ddof=1)) if train_values.size > 1 else 0.0
    if not np.isfinite(scale) or scale <= 0:
        return np.nan
    return _empirical_wasserstein_1d(
        (train_values - center) / scale, (test_values - center) / scale
    )


def standardized_target_w1_for_masks(
    values: np.ndarray,
    test_masks: np.ndarray,
) -> np.ndarray:
    """Compute exact train-standardized target W1 for arbitrary mask batches."""

    target = np.asarray(values, dtype=float)
    if target.ndim != 1 or target.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    masks = np.asarray(test_masks)
    if masks.ndim < 2 or masks.shape[-1] != target.size:
        raise ValueError("test_masks must end with the n_samples dimension")
    if masks.dtype != np.bool_:
        raise TypeError("test_masks must be boolean")

    leading_shape = masks.shape[:-1]
    flat_masks = masks.reshape(-1, target.size)
    test_counts = flat_masks.sum(axis=1)
    if np.any(test_counts <= 0) or np.any(test_counts >= target.size):
        raise ValueError("each mask must define non-empty train and test partitions")

    result = np.full(len(flat_masks), np.nan, dtype=float)
    for index, mask in enumerate(flat_masks):
        train = target[~mask]
        test = target[mask]
        if (
            train.size < 2
            or not np.isfinite(train).all()
            or not np.isfinite(test).all()
        ):
            continue
        center = float(np.mean(train))
        scale = float(np.std(train, ddof=1))
        if not np.isfinite(scale) or scale <= 0:
            continue
        result[index] = _empirical_wasserstein_1d(
            (train - center) / scale,
            (test - center) / scale,
        )
    return result.reshape(leading_shape)


def summarize_empirical_null(
    observed: float, null_values: np.ndarray
) -> dict[str, float | int]:
    """Summarize a finite empirical null distribution and its upper tail."""

    null = np.asarray(null_values, dtype=float).ravel()
    null = null[np.isfinite(null)]
    if null.size == 0:
        return {
            "null_iterations": 0,
            "null_mean": np.nan,
            "null_median": np.nan,
            "null_std": np.nan,
            "null_q025": np.nan,
            "null_q05": np.nan,
            "null_q25": np.nan,
            "null_q75": np.nan,
            "null_q95": np.nan,
            "null_q975": np.nan,
            "empirical_p_upper": np.nan,
            "severity_vs_null_median": np.nan,
        }
    median = float(np.median(null))
    quantiles = np.quantile(null, [0.025, 0.05, 0.25, 0.75, 0.95, 0.975])
    observed_is_finite = np.isfinite(observed)
    return {
        "null_iterations": int(null.size),
        "null_mean": float(np.mean(null)),
        "null_median": median,
        "null_std": float(np.std(null, ddof=1)) if null.size > 1 else 0.0,
        "null_q025": float(quantiles[0]),
        "null_q05": float(quantiles[1]),
        "null_q25": float(quantiles[2]),
        "null_q75": float(quantiles[3]),
        "null_q95": float(quantiles[4]),
        "null_q975": float(quantiles[5]),
        "empirical_p_upper": (
            float((1 + np.count_nonzero(null >= observed)) / (null.size + 1))
            if observed_is_finite
            else np.nan
        ),
        "severity_vs_null_median": (
            float(observed / median) if observed_is_finite and median > 0 else np.nan
        ),
    }


def composition_signature(row: pd.Series, composition_columns: Sequence[str]) -> str:
    """Build a stable six-decimal signature from non-zero composition values."""

    parts: list[str] = []
    for column in sorted(set(composition_columns)):
        if column not in row:
            continue
        value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
        if pd.isna(value) or not np.isfinite(value) or np.isclose(float(value), 0.0):
            continue
        parts.append(f"{column}={float(value):.6f}")
    return "|".join(parts)


def _normalise_audit_id(value: object) -> str:
    """Normalize integer-like IDs without changing genuinely textual IDs."""

    if _is_missing_scalar(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        if float(value).is_integer():
            return str(int(value))
        return format(float(value), ".15g")
    text = str(value).strip()
    if re.fullmatch(r"[+-]?\d+\.0+", text):
        return str(int(float(text)))
    return text


def prepare_legacy_embedding_coordinates(
    embedding: pd.DataFrame,
    master_ids: Sequence[object],
) -> tuple[np.ndarray, dict[str, object]]:
    """Audit and align legacy ``x/y`` embedding coordinates to master IDs.

    Identical duplicate coordinate rows are harmless and collapse to one ID.
    Conflicting coordinates for one normalized ID invalidate the whole task,
    because choosing either coordinate would make the exact OT irreproducible.
    """

    id_column = (
        "ID"
        if "ID" in embedding
        else ("__source_index__" if "__source_index__" in embedding else "")
    )
    audit: dict[str, object] = {
        "embedding_status": "ok",
        "embedding_id_column": id_column,
        "embedding_raw_row_count": int(len(embedding)),
        "embedding_invalid_row_count": 0,
        "embedding_duplicate_id_count": 0,
        "embedding_duplicate_row_count": 0,
        "embedding_duplicate_coordinate_conflict_id_count": 0,
        "embedding_master_join_missing_count": 0,
    }
    if not id_column or not {"x", "y"}.issubset(embedding.columns):
        audit["embedding_status"] = "missing_embedding_columns"
        return np.empty((0, 2), dtype=float), audit

    working = embedding[[id_column, "x", "y"]].copy()
    working["_audit_id"] = working[id_column].map(_normalise_audit_id)
    working["x"] = pd.to_numeric(working["x"], errors="coerce")
    working["y"] = pd.to_numeric(working["y"], errors="coerce")
    valid = (
        working["_audit_id"].ne("")
        & np.isfinite(working["x"])
        & np.isfinite(working["y"])
    )
    audit["embedding_invalid_row_count"] = int((~valid).sum())
    working = working.loc[valid, ["_audit_id", "x", "y"]].copy()
    audit["embedding_duplicate_row_count"] = int(
        working["_audit_id"].duplicated(keep="first").sum()
    )
    audit["embedding_duplicate_id_count"] = int(
        (working["_audit_id"].value_counts() > 1).sum()
    )
    coordinate_counts = working.groupby("_audit_id", sort=False)[["x", "y"]].apply(
        lambda group: len(group.drop_duplicates())
    )
    conflict_count = int((coordinate_counts > 1).sum())
    audit["embedding_duplicate_coordinate_conflict_id_count"] = conflict_count
    if conflict_count:
        audit["embedding_status"] = "duplicate_coordinate_conflict"
        return np.empty((0, 2), dtype=float), audit

    deduplicated = working.drop_duplicates("_audit_id", keep="first").set_index(
        "_audit_id"
    )
    normalized_master = [_normalise_audit_id(value) for value in master_ids]
    missing_count = sum(value not in deduplicated.index for value in normalized_master)
    audit["embedding_master_join_missing_count"] = int(missing_count)
    aligned = deduplicated.reindex(normalized_master)[["x", "y"]].to_numpy(dtype=float)
    if missing_count:
        audit["embedding_status"] = "master_join_missing"
    return aligned, audit


def exact_standardized_2d_ot(
    coordinates: np.ndarray,
    test_mask: np.ndarray,
    *,
    solver: Callable[[np.ndarray], object] | None = None,
) -> dict[str, object]:
    """Compute exact equal-mass 2D OT after split-train standardization."""

    values = np.asarray(coordinates, dtype=float)
    mask = np.asarray(test_mask)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("coordinates must have shape (n_samples, 2)")
    if mask.dtype != np.bool_ or mask.ndim != 1 or len(mask) != len(values):
        raise ValueError("test_mask must be a boolean vector matching coordinates")
    train = values[~mask]
    test = values[mask]
    base = {
        "metric_value": np.nan,
        "cost_matrix": np.empty((len(train), len(test)), dtype=float),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "solver_status": "not_run",
    }
    if len(test) == 0:
        return {**base, "status": "skipped_empty_test"}
    if len(train) < 2 or not np.isfinite(train).all() or not np.isfinite(test).all():
        return {**base, "status": "invalid_coordinates_or_train_size"}
    center = np.mean(train, axis=0)
    scale = np.std(train, axis=0, ddof=1)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        return {**base, "status": "invalid_train_scale"}
    train_scaled = (train - center) / scale
    test_scaled = (test - center) / scale
    costs = np.linalg.norm(train_scaled[:, None, :] - test_scaled[None, :, :], axis=2)
    result = (solver or solve_exact_ot)(costs)
    metric = float(getattr(result, "wx"))
    return {
        **base,
        "metric_value": metric,
        "cost_matrix": costs,
        "solver_status": "ok",
        "status": "ok_degenerate_n1" if len(test) == 1 else "ok",
    }


def solve_exact_ot_deletions(
    cost_matrix: np.ndarray,
    removal_sets: dict[object, Sequence[int]],
    *,
    solver: Callable[[np.ndarray], object] | None = None,
) -> dict[object, dict[str, object]]:
    """Solve independent test-column deletions, caching identical removals."""

    costs = np.asarray(cost_matrix, dtype=float)
    if costs.ndim != 2 or costs.shape[0] == 0 or costs.shape[1] == 0:
        raise ValueError("cost_matrix must be a non-empty two-dimensional array")
    if not np.isfinite(costs).all():
        raise ValueError("cost_matrix contains non-finite values")
    cache: dict[tuple[int, ...], dict[str, object]] = {}
    output: dict[object, dict[str, object]] = {}
    for label, raw_removed in removal_sets.items():
        removed = tuple(sorted({int(index) for index in raw_removed}))
        if any(index < 0 or index >= costs.shape[1] for index in removed):
            raise IndexError("removed test index is out of range")
        if removed not in cache:
            keep = np.ones(costs.shape[1], dtype=bool)
            keep[list(removed)] = False
            remaining = int(keep.sum())
            if remaining == 0:
                result = {
                    "status": "skipped_empty_test",
                    "metric_value": np.nan,
                    "removed_count": len(removed),
                    "remaining_test_count": 0,
                }
            else:
                solved = (solver or solve_exact_ot)(costs[:, keep])
                result = {
                    "status": "ok_degenerate_n1" if remaining == 1 else "ok",
                    "metric_value": float(getattr(solved, "wx")),
                    "removed_count": len(removed),
                    "remaining_test_count": remaining,
                }
            cache[removed] = result
        output[label] = dict(cache[removed])
    return output


def compute_joint_overlap_stats(
    summary: pd.DataFrame,
    *,
    ry_threshold: float = 8.0,
    rz_threshold: float = 5.0,
) -> dict[str, float | int]:
    """Summarize normalized overlap and the separate raw-distance rule."""

    all_nonbaseline = summary[
        ~summary["method_short"].astype(str).str.casefold().eq("randcv")
    ]
    ry_numeric = pd.to_numeric(all_nonbaseline["Ry"], errors="coerce")
    rz_numeric = pd.to_numeric(all_nonbaseline["Rz"], errors="coerce")
    valid = np.isfinite(ry_numeric) & np.isfinite(rz_numeric)
    high_ry = ry_numeric.loc[valid] > ry_threshold
    high_rz = rz_numeric.loc[valid] > rz_threshold
    count = int(valid.sum())
    high_ry_count = int(high_ry.sum())
    high_rz_count = int(high_rz.sum())
    raw_joint = (pd.to_numeric(all_nonbaseline["Wy"], errors="coerce") > 1.0) & (
        pd.to_numeric(all_nonbaseline["Wz"], errors="coerce") > 1.0
    )
    return {
        "nonbaseline_count": count,
        "excluded_invalid_count": int((~valid).sum()),
        "high_ry_count": high_ry_count,
        "high_rz_count": high_rz_count,
        "joint_high_count": int((high_ry & high_rz).sum()),
        "expected_joint_under_independence": (
            float(high_ry_count * high_rz_count / count) if count else np.nan
        ),
        "raw_wy_wz_gt_one_count": int(raw_joint.sum()),
    }


def threshold_grid(
    summary: pd.DataFrame,
    *,
    ry_thresholds: Iterable[float],
    rz_thresholds: Iterable[float],
) -> pd.DataFrame:
    """Count normalized joint-high points over a strict-threshold grid."""

    nonbaseline = summary[
        ~summary["method_short"].astype(str).str.casefold().eq("randcv")
    ]
    ry = pd.to_numeric(nonbaseline["Ry"], errors="coerce")
    rz = pd.to_numeric(nonbaseline["Rz"], errors="coerce")
    records = []
    for ry_threshold in ry_thresholds:
        for rz_threshold in rz_thresholds:
            high_ry = ry > float(ry_threshold)
            high_rz = rz > float(rz_threshold)
            records.append(
                {
                    "Ry_threshold": float(ry_threshold),
                    "Rz_threshold": float(rz_threshold),
                    "high_Ry_count": int(high_ry.sum()),
                    "high_Rz_count": int(high_rz.sum()),
                    "joint_high_count": int((high_ry & high_rz).sum()),
                }
            )
    return pd.DataFrame.from_records(records)


def rank_sample_drivers(rows: pd.DataFrame, *, top_n: int = 10) -> pd.DataFrame:
    """Rank sample mass contributions after removing repeated model rows."""

    if top_n < 0:
        raise ValueError("top_n must be non-negative")
    contribution = "sample_w_mass_contribution"
    if contribution not in rows:
        raise KeyError(f"missing required column: {contribution}")

    working = rows.copy()
    working[contribution] = pd.to_numeric(working[contribution], errors="coerce")
    working = working[np.isfinite(working[contribution])].copy()

    split_keys = [column for column in _SPLIT_KEYS if column in working.columns]
    sample_keys = split_keys + (["ID"] if "ID" in rows.columns else [])
    metadata = [column for column in working.columns if column in _DRIVER_SAFE_METADATA]
    if not working.empty and metadata:
        grouped = working.groupby(sample_keys, dropna=False, sort=False)
        for column in metadata:
            conflicts = grouped[column].nunique(dropna=False) > 1
            if conflicts.any():
                raise ValueError(f"conflicting invariant metadata: {column}")
    aggregations: dict[str, object] = {contribution: "mean"}
    if "sample_w_contribution" in working:
        aggregations["sample_w_contribution"] = "mean"
    if "split_w" in working:
        aggregations["split_w"] = "first"
    aggregations.update({column: "first" for column in metadata})

    ranked = (
        working.groupby(sample_keys, dropna=False, sort=False)
        .agg(aggregations)
        .reset_index()
    )
    counts = (
        working.groupby(sample_keys, dropna=False)
        .size()
        .rename("raw_row_count")
        .to_frame()
    )
    if "model" in working:
        counts["model_count"] = working.groupby(sample_keys, dropna=False)[
            "model"
        ].nunique(dropna=True)
    else:
        counts["model_count"] = 0
    ranked = ranked.merge(
        counts.reset_index(), on=sample_keys, how="left", validate="one_to_one"
    )

    rank_keys = [column for column in split_keys if column != "space"] + (
        ["space"] if "space" in split_keys else []
    )
    ranked["rank"] = (
        ranked.groupby(rank_keys, dropna=False)[contribution]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return (
        ranked[ranked["rank"] <= top_n]
        .sort_values(rank_keys + ["rank"])
        .reset_index(drop=True)
    )


def fixed_projection_sliced_wasserstein(
    train: np.ndarray,
    test: np.ndarray,
    *,
    projections: int = 128,
    random_state: int | np.random.Generator | None = 0,
) -> float:
    """Compute deterministic sliced 1-Wasserstein distance projections."""

    left = _as_2d_float(train, "train")
    right = _as_2d_float(test, "test")
    if left.shape[1] != right.shape[1]:
        raise ValueError("train and test must have the same feature dimension")
    if projections <= 0:
        raise ValueError("projections must be positive")
    if left.shape == right.shape and np.array_equal(left, right):
        return 0.0

    directions = _projection_directions(left.shape[1], projections, random_state)
    distances = [
        _empirical_wasserstein_1d(left @ direction, right @ direction)
        for direction in directions
    ]
    return float(np.mean(distances))


def fixed_projection_swd_for_masks(
    values: np.ndarray,
    test_masks: np.ndarray,
    *,
    projections: int = 128,
    random_state: int | np.random.Generator | None = 0,
    batch_size: int = 32,
) -> np.ndarray:
    """Evaluate fixed-projection SWD for many train/test masks efficiently.

    Projected values are sorted once globally for each direction.  Every mask
    then reuses those orders and the adjacent projected gaps to integrate the
    empirical CDF difference without sorting either partition again.
    """

    array = _as_2d_float(values, "values")
    masks = np.asarray(test_masks)
    if masks.ndim == 1:
        masks = masks[None, :]
    if masks.ndim != 2 or masks.shape[1] != len(array):
        raise ValueError("test_masks must have shape (n_masks, n_samples)")
    if masks.dtype != np.bool_:
        raise TypeError("test_masks must be boolean")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    test_counts = masks.sum(axis=1).astype(int)
    if np.any(test_counts <= 0) or np.any(test_counts >= len(array)):
        raise ValueError("each mask must define non-empty train and test partitions")

    directions = _projection_directions(array.shape[1], projections, random_state)
    projected = array @ directions.T
    order = np.argsort(projected, axis=0, kind="mergesort")
    sorted_projected = np.take_along_axis(projected, order, axis=0)
    gaps = np.diff(sorted_projected, axis=0).T
    order_by_projection = order.T
    ranks = np.arange(1, len(array), dtype=float)[None, None, :]

    result = np.empty(len(masks), dtype=float)
    for start in range(0, len(masks), int(batch_size)):
        stop = min(start + int(batch_size), len(masks))
        batch = masks[start:stop]
        ordered_test = np.take(batch, order_by_projection, axis=1)
        cumulative_test = np.cumsum(ordered_test, axis=2, dtype=float)[:, :, :-1]
        cumulative_train = ranks - cumulative_test
        n_test = test_counts[start:stop, None, None]
        n_train = len(array) - n_test
        direction_distances = np.sum(
            np.abs(cumulative_test / n_test - cumulative_train / n_train)
            * gaps[None, :, :],
            axis=2,
        )
        result[start:stop] = np.mean(direction_distances, axis=1)
    return result


def _projection_directions(
    dimension: int,
    projections: int,
    random_state: int | np.random.Generator | None,
) -> np.ndarray:
    if projections <= 0:
        raise ValueError("projections must be positive")
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    directions = rng.normal(size=(int(projections), int(dimension)))
    norms = np.linalg.norm(directions, axis=1)
    while np.any(norms == 0):
        directions[norms == 0] = rng.normal(
            size=(int((norms == 0).sum()), int(dimension))
        )
        norms = np.linalg.norm(directions, axis=1)
    return directions / norms[:, None]


def matched_partition_masks(
    n_samples: int,
    test_sizes: Sequence[int],
    iterations: int,
    random_state: int | np.random.Generator | None,
) -> np.ndarray:
    """Draw reproducible disjoint test folds with exactly matched sizes."""

    n_samples = int(n_samples)
    sizes = np.asarray(list(test_sizes), dtype=int)
    iterations = int(iterations)
    if n_samples < 2:
        raise ValueError("n_samples must be at least two")
    if sizes.size == 0 or np.any(sizes <= 0) or np.any(sizes >= n_samples):
        raise ValueError("test_sizes must contain values between 1 and n_samples - 1")
    if int(sizes.sum()) > n_samples:
        raise ValueError("test_sizes must sum to no more than n_samples")
    if iterations < 0:
        raise ValueError("iterations must be non-negative")

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    masks = np.zeros((iterations, len(sizes), n_samples), dtype=bool)
    boundaries = np.concatenate(([0], np.cumsum(sizes)))
    for iteration in range(iterations):
        permutation = rng.permutation(n_samples)
        for fold_index, (start, stop) in enumerate(
            zip(boundaries[:-1], boundaries[1:])
        ):
            masks[iteration, fold_index, permutation[int(start) : int(stop)]] = True
    return masks


def _as_2d_float(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 1D or 2D array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _empirical_wasserstein_1d(left: np.ndarray, right: np.ndarray) -> float:
    """Exact W1 distance between equally weighted one-dimensional samples."""

    left = np.sort(np.asarray(left, dtype=float))
    right = np.sort(np.asarray(right, dtype=float))
    values = np.concatenate((left, right))
    values.sort()
    if len(values) < 2:
        return float(abs(left[0] - right[0]))
    deltas = np.diff(values)
    left_cdf = np.searchsorted(left, values[:-1], side="right") / left.size
    right_cdf = np.searchsorted(right, values[:-1], side="right") / right.size
    return float(np.sum(np.abs(left_cdf - right_cdf) * deltas))


def empirical_random_distribution(
    values: np.ndarray,
    *,
    test_sizes: Sequence[int],
    distance_fn: Callable[[np.ndarray, np.ndarray], float],
    iterations: int = 1000,
    random_state: int | np.random.Generator | None = 0,
) -> np.ndarray:
    """Generate an empirical distance baseline from random train/test splits."""

    array = _as_2d_float(values, "values")
    sizes = np.asarray(list(test_sizes), dtype=int)
    if sizes.size == 0 or np.any(sizes <= 0) or np.any(sizes >= len(array)):
        raise ValueError("test_sizes must contain values between 1 and n_samples - 1")
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    result = np.empty(int(iterations), dtype=float)
    for index in range(int(iterations)):
        fold_distances = []
        for test_size in sizes:
            permutation = rng.permutation(len(array))
            test_indices = permutation[: int(test_size)]
            train_indices = permutation[int(test_size) :]
            fold_distances.append(
                float(distance_fn(array[train_indices], array[test_indices]))
            )
        result[index] = float(np.mean(fold_distances))
    return result


def write_audit_artifacts(
    output_dir: str | Path,
    *,
    severity_source_audit: pd.DataFrame,
    sample_drivers: pd.DataFrame,
    metric_robustness: pd.DataFrame,
    report_text: str,
) -> dict[str, Path]:
    """Write the four stable audit artifacts and return their paths."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "severity_source_audit": directory / "severity_source_audit.csv",
        "joint_high_sample_drivers": directory / "joint_high_sample_drivers.csv",
        "severity_metric_robustness": directory / "severity_metric_robustness.csv",
        "severity_joint_high_audit": directory / "severity_joint_high_audit.md",
    }
    severity_source_audit.to_csv(
        paths["severity_source_audit"], index=False, encoding="utf-8-sig"
    )
    sample_drivers.to_csv(
        paths["joint_high_sample_drivers"], index=False, encoding="utf-8-sig"
    )
    metric_robustness.to_csv(
        paths["severity_metric_robustness"], index=False, encoding="utf-8-sig"
    )
    paths["severity_joint_high_audit"].write_text(report_text, encoding="utf-8")
    return paths


__all__ = [
    "build_source_severity_audit",
    "classify_source",
    "collapse_split_rows",
    "composition_signature",
    "compute_joint_overlap_stats",
    "empirical_random_distribution",
    "exact_standardized_2d_ot",
    "fixed_projection_swd_for_masks",
    "fixed_projection_sliced_wasserstein",
    "legacy_severity_summary",
    "matched_partition_masks",
    "parse_feature_columns",
    "prepare_representation",
    "prepare_legacy_embedding_coordinates",
    "prepare_task_global_representations",
    "rank_sample_drivers",
    "standardized_target_w1",
    "standardized_target_w1_for_masks",
    "solve_exact_ot_deletions",
    "summarize_empirical_null",
    "threshold_grid",
    "validate_split_integrity",
    "write_audit_artifacts",
]
