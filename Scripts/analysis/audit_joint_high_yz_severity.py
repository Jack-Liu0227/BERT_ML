"""Audit joint-high target/composition OOD severity conditions.

The audit intentionally lives outside the plotting pipeline.  The large
sample-level export is consumed in two streaming passes: the first pass
reproduces the published mixed-source summary and constructs one record per
physical split/space; the second pass reads only canonical joint-high splits
to identify sample drivers.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_processing.yz_severity_audit import (  # noqa: E402
    build_source_severity_audit,
    classify_source,
    composition_signature,
    compute_joint_overlap_stats,
    exact_standardized_2d_ot,
    fixed_projection_swd_for_masks,
    legacy_severity_summary,
    matched_partition_masks,
    parse_feature_columns,
    prepare_task_global_representations,
    prepare_legacy_embedding_coordinates,
    rank_sample_drivers,
    standardized_target_w1_for_masks,
    standardized_target_w1,
    solve_exact_ot_deletions,
    summarize_empirical_null,
    threshold_grid,
    validate_split_integrity,
    write_audit_artifacts,
)
from Scripts.analysis.export_three_space_w_ood_report import (  # noqa: E402
    YZ_DATASET_TO_CASE,
)


DEFAULT_SAMPLES_LONG = Path(
    "output/w_error_relationship_full/OOD/csv/three_space_w_error_samples_long.csv"
)
DEFAULT_SEVERITY_SUMMARY = Path(
    "output/w_error_relationship_full/OOD/yz_severity_map/csv/ood_yz_severity_summary.csv"
)
DEFAULT_SOURCE_INVENTORY = Path(
    "output/w_error_relationship_full/OOD/csv/w_source_inventory.csv"
)
DEFAULT_EMBEDDING_DATA_DIR = (
    Path("D:/XJTU")
    / "已完成论文数据汇总"
    / "Fewshot"
    / "预处理汇总数据"
    / "OOD"
    / "embedding_data"
)
DEFAULT_OUTPUT_ROOT = Path("output/w_error_relationship_full/OOD/yz_severity_map/audit")

_IDENTITY_COLUMNS = [
    "task_key",
    "task_id",
    "alloy_family",
    "dataset_name",
    "property",
]
_SPLIT_COLUMNS = _IDENTITY_COLUMNS + [
    "method_short",
    "source_split_dir",
    "fold_id",
    "space",
]
_FIRST_PASS_COLUMNS = _SPLIT_COLUMNS + ["scope", "model", "ID", "split_w"]
_DRIVER_COLUMNS = [
    "scope",
    *_IDENTITY_COLUMNS,
    "method_short",
    "method",
    "model",
    "fold_id",
    "ID",
    "test_set",
    "space",
    "sample_w_contribution",
    "sample_w_mass_contribution",
    "split_w",
    "target_value",
    "true_value",
    "source_split_dir",
    "join_mode",
    "wx_mode",
]
_INVENTORY_OUTPUT_FIELDS = (
    "ood_scope",
    "method",
    "test_set",
    "wx_mode",
    "split_source",
    "train_file",
    "test_file",
    "split_summary_file",
    "train_size",
    "test_size",
    "feature_count",
    "feature_columns",
    "inventory_models",
    "inventory_model_count",
    "inventory_train_file_variant_count",
    "inventory_test_file_variant_count",
    "inventory_feature_columns_variant_count",
)
_CHUNK_SIZE = 100_000


def build_parser() -> argparse.ArgumentParser:
    """Build the production command-line interface."""

    parser = argparse.ArgumentParser(
        description="核查联合高 Ry/Rz 条件的来源、样本驱动和指标敏感性。"
    )
    parser.add_argument("--samples-long", default=str(DEFAULT_SAMPLES_LONG))
    parser.add_argument("--severity-summary", default=str(DEFAULT_SEVERITY_SUMMARY))
    parser.add_argument("--source-inventory", default=str(DEFAULT_SOURCE_INVENTORY))
    parser.add_argument("--embedding-data-dir", default=str(DEFAULT_EMBEDDING_DATA_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--ry-threshold", type=float, default=8.0)
    parser.add_argument("--rz-threshold", type=float, default=5.0)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--random-splits", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=20260715)
    parser.add_argument("--pca-components", type=int, default=20)
    parser.add_argument("--tsne-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--sliced-projections", type=int, default=128)
    parser.add_argument("--canonical-source", default="traditional")
    parser.add_argument("--legacy-exclude-task", default="Matbench Steel-YS")
    parser.add_argument("--max-exact-deletion-sets-per-split", type=int, default=40)
    parser.add_argument("--no-robustness", action="store_true")
    return parser


def _read_csv_flexible(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    source = Path(path)
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"):
        try:
            return pd.read_csv(source, encoding=encoding, low_memory=False, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(source, low_memory=False, **kwargs)


def _detect_csv_encoding(path: str | Path) -> tuple[str, list[str]]:
    source = Path(path)
    last_error: UnicodeDecodeError | None = None
    with source.open("rb") as handle:
        sample = handle.read(1024 * 1024)
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"):
        try:
            sample.decode(encoding, errors="strict")
            header = pd.read_csv(
                source,
                encoding=encoding,
                low_memory=False,
                nrows=0,
            )
            return encoding, list(header.columns)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise ValueError(f"unable to detect CSV encoding: {source}")


def _csv_columns(path: str | Path) -> list[str]:
    return _detect_csv_encoding(path)[1]


def _stream_csv(
    path: str | Path,
    *,
    requested_columns: Iterable[str],
    chunksize: int = _CHUNK_SIZE,
) -> Iterator[pd.DataFrame]:
    encoding, available = _detect_csv_encoding(path)
    selected = [column for column in requested_columns if column in available]
    for required in ("task_key", "method_short", "space", "split_w"):
        if required not in selected:
            raise KeyError(f"missing required long-table column: {required}")
    yield from pd.read_csv(
        Path(path),
        encoding=encoding,
        low_memory=False,
        usecols=selected,
        chunksize=chunksize,
    )


def _missing(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _key_value(value: object) -> object:
    if _missing(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def _normalise_path_key(value: object) -> str:
    """Return a comparison-only path key without resolving the filesystem."""

    if _missing(value):
        return ""
    text = str(value).strip().replace("\\", "/")
    while text.endswith("/"):
        text = text[:-1]
    return text.casefold()


def _normalise_id(value: object) -> str:
    """Normalize integer-like identifiers while preserving textual IDs."""

    if _missing(value):
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


def _method_slug(value: object) -> str:
    text = "" if _missing(value) else str(value).strip().casefold()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_") or "unknown"


def _split_variant(method_short: object, source_split_dir: object) -> str:
    method = _method_slug(method_short)
    path = "" if _missing(source_split_dir) else str(source_split_dir)
    segments = {
        _method_slug(segment)
        for segment in path.replace("\\", "/").split("/")
        if segment
    }
    if method == "loco":
        if "loco_k5" in segments:
            return "loco_k5"
        return "loco"
    if method in {"randcv", "rand_cv"}:
        return "randcv"
    if method in {"extra", "extrapolation"}:
        return "extra"
    return method


def _classify_accumulated_source(
    source_split_dir: object,
    models: Iterable[str],
    canonical_source: str,
) -> dict[str, object]:
    source = classify_source(source_split_dir, None)
    if source["source_family"] == "other":
        model_sources = [classify_source(source_split_dir, model) for model in models]
        names = {item["source_name"] for item in model_sources}
        if len(names) == 1:
            source = model_sources[0]
        elif len(names) > 1:
            source = {
                "source_family": "other",
                "source_name": "ambiguous_models",
                "is_canonical": False,
            }
    requested = str(canonical_source).strip().casefold()
    source["is_canonical"] = bool(
        str(source["source_name"]).casefold() == requested
        or str(source["source_family"]).casefold() == requested
    )
    return source


def _stream_first_pass(
    samples_long: Path,
    *,
    task_keys: set[str],
    canonical_source: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return legacy fold means and model/sample-deduplicated split records."""

    legacy_sum_count: dict[tuple[object, ...], list[float | int]] = {}
    split_accumulators: dict[tuple[object, ...], dict[str, object]] = {}
    legacy_columns = _IDENTITY_COLUMNS + ["method_short", "fold_id", "space"]

    for chunk in _stream_csv(samples_long, requested_columns=_FIRST_PASS_COLUMNS):
        if "scope" in chunk:
            chunk = chunk[chunk["scope"].astype(str).str.casefold().eq("ood")]
        chunk = chunk[
            chunk["space"].isin(["Y-space", "Z-space"])
            & chunk["task_key"].astype(str).isin(task_keys)
        ].copy()
        if chunk.empty:
            continue
        for column in _IDENTITY_COLUMNS + [
            "fold_id",
            "source_split_dir",
            "model",
            "ID",
        ]:
            if column not in chunk:
                chunk[column] = pd.NA
        chunk["_split_w_numeric"] = pd.to_numeric(chunk["split_w"], errors="coerce")

        legacy_grouped = chunk.groupby(legacy_columns, dropna=False, sort=False)[
            "_split_w_numeric"
        ].agg(["sum", "count"])
        for raw_key, row in legacy_grouped.iterrows():
            values = raw_key if isinstance(raw_key, tuple) else (raw_key,)
            key = tuple(_key_value(value) for value in values)
            state = legacy_sum_count.setdefault(key, [0.0, 0])
            state[0] = float(state[0]) + float(row["sum"])
            state[1] = int(state[1]) + int(row["count"])

        for raw_key, group in chunk.groupby(_SPLIT_COLUMNS, dropna=False, sort=False):
            values = raw_key if isinstance(raw_key, tuple) else (raw_key,)
            key = tuple(_key_value(value) for value in values)
            state = split_accumulators.get(key)
            if state is None:
                state = {
                    "first": np.nan,
                    "first_finite": np.nan,
                    "minimum": np.nan,
                    "maximum": np.nan,
                    "nonmissing_count": 0,
                    "raw_row_count": 0,
                    "models": set(),
                    "ids": set(),
                }
                split_accumulators[key] = state
            numeric = group["_split_w_numeric"]
            if int(state["raw_row_count"]) == 0 and len(numeric):
                state["first"] = (
                    float(numeric.iloc[0]) if pd.notna(numeric.iloc[0]) else np.nan
                )
            finite = numeric[np.isfinite(numeric)]
            if not finite.empty:
                if _missing(state["first_finite"]):
                    state["first_finite"] = float(finite.iloc[0])
                local_min = float(finite.min())
                local_max = float(finite.max())
                state["minimum"] = (
                    local_min
                    if _missing(state["minimum"])
                    else min(float(state["minimum"]), local_min)
                )
                state["maximum"] = (
                    local_max
                    if _missing(state["maximum"])
                    else max(float(state["maximum"]), local_max)
                )
                state["nonmissing_count"] = int(state["nonmissing_count"]) + len(finite)
            state["raw_row_count"] = int(state["raw_row_count"]) + len(group)
            state["models"].update(
                str(value)
                for value in group["model"].dropna().drop_duplicates().tolist()
            )
            state["ids"].update(
                _normalise_id(value)
                for value in group["ID"].dropna().drop_duplicates().tolist()
                if _normalise_id(value)
            )

    legacy_records: list[dict[str, object]] = []
    for key, (total, count) in legacy_sum_count.items():
        record = dict(zip(legacy_columns, key))
        record["split_w"] = float(total) / int(count) if count else np.nan
        record["legacy_raw_row_count"] = int(count)
        legacy_records.append(record)

    collapsed_records: list[dict[str, object]] = []
    for key, state in split_accumulators.items():
        record = dict(zip(_SPLIT_COLUMNS, key))
        minimum = state["minimum"]
        maximum = state["maximum"]
        if int(state["nonmissing_count"]) == 0:
            status = "missing"
            split_range = np.nan
        else:
            split_range = float(maximum) - float(minimum)
            status = (
                "ok"
                if np.isclose(split_range, 0.0, atol=1e-12, rtol=1e-12)
                else "conflict"
            )
        if status == "conflict":
            label = ", ".join(
                f"{column}={record.get(column)!r}"
                for column in (
                    "task_key",
                    "method_short",
                    "source_split_dir",
                    "fold_id",
                    "space",
                )
            )
            raise ValueError(
                "conflicting finite split_w values for "
                f"{label}: min={minimum!r}, max={maximum!r}"
            )
        source = _classify_accumulated_source(
            record.get("source_split_dir"), state["models"], canonical_source
        )
        record.update(source)
        record.update(
            {
                "split_variant": _split_variant(
                    record.get("method_short"), record.get("source_split_dir")
                ),
                "split_w": state["first_finite"],
                "split_w_first": state["first"],
                "split_w_min": minimum,
                "split_w_max": maximum,
                "split_w_range": split_range,
                "split_w_status": status,
                "split_w_nonmissing_count": int(state["nonmissing_count"]),
                "model_count": len(state["models"]),
                "sample_count": len(state["ids"]),
                "raw_row_count": int(state["raw_row_count"]),
            }
        )
        collapsed_records.append(record)

    return pd.DataFrame.from_records(legacy_records), pd.DataFrame.from_records(
        collapsed_records
    )


def _prepare_legacy_rows(
    legacy_folds: pd.DataFrame,
    published: pd.DataFrame,
    *,
    ry_threshold: float,
    rz_threshold: float,
) -> pd.DataFrame:
    reproduced = legacy_severity_summary(legacy_folds)
    fold_counts = (
        legacy_folds.groupby(
            _IDENTITY_COLUMNS + ["method_short", "space"], dropna=False
        )["fold_id"]
        .nunique(dropna=True)
        .unstack("space")
        .reset_index()
        .rename(
            columns={
                "Y-space": "reproduced_n_folds_y",
                "Z-space": "reproduced_n_folds_z",
            }
        )
    )
    reproduced = reproduced.merge(
        fold_counts,
        on=_IDENTITY_COLUMNS + ["method_short"],
        how="left",
        validate="one_to_one",
    )

    keys = _IDENTITY_COLUMNS + ["method_short"]
    published_columns = keys + [
        column
        for column in (
            "Wy",
            "Wz",
            "Wy_random",
            "Wz_random",
            "Ry",
            "Rz",
            "n_folds_y",
            "n_folds_z",
        )
        if column in published
    ]
    comparison = published[published_columns].merge(
        reproduced,
        on=keys,
        how="outer",
        suffixes=("_published", ""),
        indicator=True,
        validate="one_to_one",
    )
    residual_columns: list[str] = []
    for metric in ("Wy", "Wz", "Wy_random", "Wz_random", "Ry", "Rz"):
        published_name = f"{metric}_published"
        if published_name not in comparison:
            comparison[published_name] = np.nan
        residual = f"reproduction_{metric}_residual"
        comparison[residual] = pd.to_numeric(
            comparison.get(metric), errors="coerce"
        ) - pd.to_numeric(comparison[published_name], errors="coerce")
        residual_columns.append(residual)
    comparison["published_reproduction_max_abs_residual"] = (
        comparison[residual_columns].abs().max(axis=1)
    )
    complete = comparison[residual_columns].notna().all(axis=1)
    comparison["published_reproduction_status"] = np.select(
        [
            comparison["_merge"].eq("left_only"),
            comparison["_merge"].eq("right_only"),
            ~complete,
        ],
        ["missing_reproduction", "missing_published", "incomplete"],
        default=np.where(
            comparison["published_reproduction_max_abs_residual"] <= 1e-10,
            "exact",
            "mismatch",
        ),
    )
    comparison["audit_view"] = "legacy_reproduction"
    comparison["source_family"] = "mixed_sources"
    comparison["source_name"] = "mixed_sources"
    comparison["is_canonical"] = False
    comparison["source_split_dir"] = "mixed_sources"
    comparison["fold_id"] = "protocol"
    comparison["split_variant"] = comparison["method_short"].map(_method_slug)
    comparison["protocol_Wy"] = comparison["Wy"]
    comparison["protocol_Wz"] = comparison["Wz"]
    comparison["protocol_Ry"] = comparison["Ry"]
    comparison["protocol_Rz"] = comparison["Rz"]
    comparison["protocol_fold_count"] = comparison[
        [
            column
            for column in ("reproduced_n_folds_y", "reproduced_n_folds_z")
            if column in comparison
        ]
    ].max(axis=1)
    comparison["baseline_status"] = np.select(
        [
            comparison[["Wy_random", "Wz_random"]].isna().any(axis=1),
            (comparison[["Wy_random", "Wz_random"]] <= 0).any(axis=1),
        ],
        ["missing", "nonpositive"],
        default="ok",
    )
    comparison["protocol_high_Ry"] = comparison["protocol_Ry"] > ry_threshold
    comparison["protocol_high_Rz"] = comparison["protocol_Rz"] > rz_threshold
    comparison["protocol_joint_high"] = (
        comparison["protocol_high_Ry"] & comparison["protocol_high_Rz"]
    )
    comparison.drop(columns=["_merge"], inplace=True)
    return comparison


def _normalise_method_identity(value: object) -> str:
    slug = _method_slug(value)
    aliases = {
        "randcv": "randcv",
        "rand_cv": "randcv",
        "randomcv": "randcv",
        "random_cv": "randcv",
        "random_cv_baseline": "randcv",
        "extra": "extra",
        "extrapolation": "extra",
        "target_extrapolation": "extra",
        "loco": "loco",
        "sx_sgl": "sx_sgl",
        "sparsexsingle": "sx_sgl",
        "sparse_x_single": "sx_sgl",
        "sx_cls": "sx_cls",
        "sparsexcluster": "sx_cls",
        "sparse_x_cluster": "sx_cls",
        "sy_sgl": "sy_sgl",
        "sparseysingle": "sy_sgl",
        "sparse_y_single": "sy_sgl",
        "sy_cls": "sy_cls",
        "sparseycluster": "sy_cls",
        "sparse_y_cluster": "sy_cls",
    }
    return aliases.get(slug, slug)


def _inventory_field_key(field: str, value: object) -> str:
    if _missing(value):
        return "<missing>"
    if field in {"train_file", "test_file"}:
        return _normalise_path_key(value)
    if field == "method":
        return _normalise_method_identity(value)
    if field == "fold_id":
        return _method_slug(value)
    if field == "feature_columns":
        return ";".join(
            column.strip() for column in str(value).split(";") if column.strip()
        )
    return str(value).strip()


def _inventory_maps(
    inventory: pd.DataFrame,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    if "source_split_dir" not in inventory:
        raise KeyError("source inventory is missing source_split_dir")
    frame = inventory.copy()
    frame["_exact_path"] = frame["source_split_dir"].map(
        lambda value: "" if _missing(value) else str(value)
    )
    frame["_normalised_path"] = frame["source_split_dir"].map(_normalise_path_key)

    defining_fields = [
        field
        for field in ("train_file", "test_file", "feature_columns", "method", "fold_id")
        if field in frame
    ]
    for path_key, group in frame[frame["_normalised_path"].ne("")].groupby(
        "_normalised_path", sort=False, dropna=False
    ):
        conflicts = [
            field
            for field in defining_fields
            if group[field]
            .map(lambda value: _inventory_field_key(field, value))
            .nunique(dropna=False)
            > 1
        ]
        if conflicts:
            raise ValueError(
                "conflicting inventory variants for "
                f"source_split_dir={path_key!r}: {', '.join(conflicts)}"
            )

    def records_by(column: str) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for key, group in frame[frame[column].ne("")].groupby(
            column, sort=False, dropna=False
        ):
            first = group.iloc[0].to_dict()
            if "model" in group:
                models = sorted({str(value) for value in group["model"].dropna()})
                first["inventory_models"] = ";".join(models)
                first["inventory_model_count"] = len(models)
            for field in ("train_file", "test_file", "feature_columns"):
                if field in group:
                    unique = (
                        group[field]
                        .map(lambda value: _inventory_field_key(field, value))
                        .drop_duplicates()
                    )
                    first[f"inventory_{field}_variant_count"] = int(len(unique))
            result[str(key)] = first
        return result

    return records_by("_exact_path"), records_by("_normalised_path")


def _lookup_inventory(
    source_split_dir: object,
    exact: dict[str, dict[str, object]],
    normalised: dict[str, dict[str, object]],
    *,
    method_short: object = None,
    fold_id: object = None,
) -> tuple[dict[str, object] | None, str]:
    raw = "" if _missing(source_split_dir) else str(source_split_dir)
    item: dict[str, object] | None = None
    mode = "missing"
    if raw and raw in exact:
        item, mode = exact[raw], "exact"
    else:
        key = _normalise_path_key(source_split_dir)
        if key and key in normalised:
            item, mode = normalised[key], "normalised"
    if item is None:
        return None, mode

    mismatches: list[str] = []
    if not _missing(method_short) and not _missing(item.get("method")):
        expected_method = _normalise_method_identity(method_short)
        inventory_method = _normalise_method_identity(item.get("method"))
        if expected_method != inventory_method:
            mismatches.append(
                f"method_short={method_short!r} != inventory method={item.get('method')!r}"
            )
    if not _missing(fold_id) and not _missing(item.get("fold_id")):
        expected_fold = _method_slug(fold_id)
        inventory_fold = _method_slug(item.get("fold_id"))
        if expected_fold != inventory_fold:
            mismatches.append(
                f"fold_id={fold_id!r} != inventory fold_id={item.get('fold_id')!r}"
            )
    if mismatches:
        raise ValueError(
            "inventory split identity mismatch for "
            f"source_split_dir={source_split_dir!r}: {'; '.join(mismatches)}"
        )
    return item, mode


def _enrich_source_audit(
    source_audit: pd.DataFrame,
    collapsed: pd.DataFrame,
    inventory_exact: dict[str, dict[str, object]],
    inventory_normalised: dict[str, dict[str, object]],
) -> pd.DataFrame:
    audit = source_audit.copy()
    merge_keys = [
        column
        for column in (
            *_IDENTITY_COLUMNS,
            "source_family",
            "source_name",
            "is_canonical",
            "method_short",
            "fold_id",
            "source_split_dir",
            "split_variant",
        )
        if column in audit and column in collapsed
    ]
    metadata_records: list[dict[str, object]] = []
    for key, group in collapsed.groupby(merge_keys, dropna=False, sort=False):
        values = key if isinstance(key, tuple) else (key,)
        record = dict(zip(merge_keys, values))
        record["model_count"] = int(group["model_count"].max())
        record["sample_count"] = int(group["sample_count"].max())
        record["raw_row_count"] = int(group["raw_row_count"].sum())
        for space, prefix in (("Y-space", "y"), ("Z-space", "z")):
            subset = group[group["space"].eq(space)]
            if subset.empty:
                continue
            row = subset.iloc[0]
            for column in (
                "split_w_first",
                "split_w_min",
                "split_w_max",
                "split_w_range",
                "split_w_status",
                "split_w_nonmissing_count",
                "model_count",
                "sample_count",
                "raw_row_count",
            ):
                record[f"{prefix}_{column}"] = row.get(column)
        metadata_records.append(record)
    if metadata_records:
        audit = audit.merge(
            pd.DataFrame.from_records(metadata_records),
            on=merge_keys,
            how="left",
            validate="one_to_one",
        )

    inventory_records: list[dict[str, object]] = []
    for row in audit[["source_split_dir", "method_short", "fold_id"]].itertuples(
        index=False
    ):
        item, mode = _lookup_inventory(
            row.source_split_dir,
            inventory_exact,
            inventory_normalised,
            method_short=row.method_short,
            fold_id=row.fold_id,
        )
        record = {"inventory_join_mode": mode}
        if item is not None:
            record.update(
                {field: item.get(field) for field in _INVENTORY_OUTPUT_FIELDS}
            )
        inventory_records.append(record)
    if inventory_records:
        inventory_frame = pd.DataFrame.from_records(
            inventory_records, index=audit.index
        )
        for column in inventory_frame:
            if column in audit:
                audit[f"inventory_{column}"] = inventory_frame[column]
            else:
                audit[column] = inventory_frame[column]
    return audit


def _inventory_method_short(value: object) -> str | None:
    return {
        "randcv": "RandCV",
        "extra": "Extra.",
        "loco": "LOCO",
        "sx_sgl": "SX-sgl",
        "sx_cls": "SX-cls",
        "sy_sgl": "SY-sgl",
        "sy_cls": "SY-cls",
    }.get(_normalise_method_identity(value))


def _published_task_identities(published: pd.DataFrame) -> dict[str, dict[str, object]]:
    identities = published[_IDENTITY_COLUMNS].drop_duplicates()
    ambiguous = identities[identities.duplicated("task_key", keep=False)]
    if not ambiguous.empty:
        keys = sorted(ambiguous["task_key"].astype(str).unique())
        raise ValueError(f"ambiguous published task identities: {keys}")
    return {str(row["task_key"]): row.to_dict() for _, row in identities.iterrows()}


def _path_matches_published_identity(
    source_split_dir: object,
    identity: dict[str, object],
) -> bool:
    if _missing(source_split_dir):
        return False
    segments = [
        _method_slug(segment)
        for segment in str(source_split_dir).replace("\\", "/").split("/")
        if segment
    ]
    expected = [
        _method_slug(identity[column])
        for column in ("alloy_family", "dataset_name", "property")
    ]
    return any(
        segments[index : index + len(expected)] == expected
        for index in range(len(segments) - len(expected) + 1)
    )


def _map_inventory_task_identity(
    inventory_task_key: object,
    source_split_dir: object,
    published_identities: dict[str, dict[str, object]],
) -> tuple[dict[str, object] | None, str]:
    if _missing(inventory_task_key):
        return None, "missing"
    task_key = str(inventory_task_key)
    if task_key in published_identities:
        return published_identities[task_key], "exact"
    path_matches = [
        identity
        for identity in published_identities.values()
        if _path_matches_published_identity(source_split_dir, identity)
    ]
    if len(path_matches) == 1:
        return path_matches[0], "physical_path_identity"
    return None, "ambiguous_physical_path" if path_matches else "unmapped"


def _append_inventory_only_source_rows(
    source_audit: pd.DataFrame,
    *,
    inventory: pd.DataFrame,
    published: pd.DataFrame,
    inventory_exact: dict[str, dict[str, object]],
    inventory_normalised: dict[str, dict[str, object]],
    canonical_source: str,
) -> pd.DataFrame:
    if inventory.empty:
        return source_audit
    published_identities = _published_task_identities(published)
    mapped_records: list[dict[str, object]] = []
    for _, row in inventory.iterrows():
        identity, mapping_status = _map_inventory_task_identity(
            row.get("task_key"), row.get("source_split_dir"), published_identities
        )
        method_short = _inventory_method_short(row.get("method"))
        path_key = _normalise_path_key(row.get("source_split_dir"))
        if identity is None or method_short is None or not path_key:
            continue
        mapped_records.append(
            {
                **row.to_dict(),
                "_published_task_key": identity["task_key"],
                "_task_mapping_status": mapping_status,
                "_method_short": method_short,
                "_path_key": path_key,
            }
        )
    if not mapped_records:
        return source_audit
    mapped = pd.DataFrame.from_records(mapped_records)

    key_columns = [
        "task_key",
        "source_split_dir",
        "fold_id",
        "method_short",
        "split_variant",
    ]

    def audit_key(record: dict[str, object] | pd.Series) -> tuple[str, ...]:
        return (
            str(record["task_key"]),
            _normalise_path_key(record["source_split_dir"]),
            _method_slug(record["fold_id"]),
            _normalise_method_identity(record["method_short"]),
            _method_slug(record["split_variant"]),
        )

    existing_keys = {audit_key(row) for _, row in source_audit[key_columns].iterrows()}
    inventory_only_records: list[dict[str, object]] = []
    for path_key, group in mapped.groupby("_path_key", sort=False, dropna=False):
        task_keys = {
            str(value) for value in group["_published_task_key"].dropna().unique()
        }
        method_shorts = {
            str(value) for value in group["_method_short"].dropna().unique()
        }
        fold_ids = {
            _method_slug(value): value for value in group["fold_id"].dropna().unique()
        }
        if len(task_keys) != 1 or len(method_shorts) != 1 or len(fold_ids) != 1:
            continue
        task_key = next(iter(task_keys))
        method_short = next(iter(method_shorts))
        fold_id = next(iter(fold_ids.values()))
        source_path = group["source_split_dir"].iloc[0]
        split_variant = _split_variant(method_short, source_path)
        candidate_key = (
            task_key,
            str(path_key),
            _method_slug(fold_id),
            _normalise_method_identity(method_short),
            _method_slug(split_variant),
        )
        if candidate_key in existing_keys:
            continue
        identity = published_identities[task_key]
        models = [
            str(value) for value in group.get("model", pd.Series(dtype=object)).dropna()
        ]
        source = _classify_accumulated_source(source_path, models, canonical_source)
        inventory_item, _ = _lookup_inventory(
            source_path,
            inventory_exact,
            inventory_normalised,
            method_short=method_short,
            fold_id=fold_id,
        )
        record: dict[str, object] = {
            **identity,
            "method_short": method_short,
            "fold_id": fold_id,
            "source_split_dir": source_path,
            "split_variant": split_variant,
            **source,
            "baseline_status": "missing",
            "inventory_join_mode": "inventory_only",
            "inventory_task_key": ";".join(
                sorted({str(value) for value in group["task_key"].dropna().unique()})
            ),
            "inventory_task_mapping_status": ";".join(
                sorted({str(value) for value in group["_task_mapping_status"].unique()})
            ),
            "y_split_w_status": "inventory_only_missing_yz",
            "z_split_w_status": "inventory_only_missing_yz",
            "protocol_fold_count": 0,
            "model_count": 0,
            "sample_count": 0,
            "raw_row_count": 0,
            "protocol_high_Ry": False,
            "protocol_high_Rz": False,
            "protocol_joint_high": False,
        }
        for column in (
            "Wy",
            "Wz",
            "Ry",
            "Rz",
            "Wy_random",
            "Wz_random",
            "protocol_Wy",
            "protocol_Wz",
            "protocol_Ry",
            "protocol_Rz",
            "y_split_w_first",
            "y_split_w_min",
            "y_split_w_max",
            "y_split_w_range",
            "z_split_w_first",
            "z_split_w_min",
            "z_split_w_max",
            "z_split_w_range",
        ):
            record[column] = np.nan
        for column in (
            "y_split_w_nonmissing_count",
            "y_model_count",
            "y_sample_count",
            "y_raw_row_count",
            "z_split_w_nonmissing_count",
            "z_model_count",
            "z_sample_count",
            "z_raw_row_count",
        ):
            record[column] = 0
        if inventory_item is not None:
            record.update(
                {field: inventory_item.get(field) for field in _INVENTORY_OUTPUT_FIELDS}
            )
        inventory_only_records.append(record)
        existing_keys.add(candidate_key)

    if not inventory_only_records:
        return source_audit
    combined = pd.concat(
        [source_audit, pd.DataFrame.from_records(inventory_only_records)],
        ignore_index=True,
        sort=False,
    )
    duplicate_mask = combined.duplicated(key_columns, keep=False)
    if duplicate_mask.any():
        duplicates = combined.loc[duplicate_mask, key_columns].to_dict("records")
        raise ValueError(
            f"duplicate source audit split identity after inventory append: {duplicates}"
        )
    return combined


def _joint_protocol_rows(source_audit: pd.DataFrame) -> pd.DataFrame:
    protocol_keys = [
        column
        for column in (
            *_IDENTITY_COLUMNS,
            "source_family",
            "source_name",
            "method_short",
            "split_variant",
        )
        if column in source_audit
    ]
    return (
        source_audit[
            source_audit["is_canonical"].astype(bool)
            & source_audit["protocol_joint_high"].astype(bool)
            & ~source_audit["method_short"].astype(str).str.casefold().eq("randcv")
        ]
        .drop_duplicates(protocol_keys)
        .reset_index(drop=True)
    )


def _joint_physical_splits(
    source_audit: pd.DataFrame, protocols: pd.DataFrame
) -> pd.DataFrame:
    if protocols.empty:
        return source_audit.iloc[0:0].copy()
    protocol_keys = [
        column
        for column in (
            *_IDENTITY_COLUMNS,
            "source_family",
            "source_name",
            "method_short",
            "split_variant",
        )
        if column in protocols and column in source_audit
    ]
    selected = source_audit.merge(
        protocols[protocol_keys].drop_duplicates(),
        on=protocol_keys,
        how="inner",
        validate="many_to_one",
    )
    return selected.drop_duplicates(
        [
            column
            for column in (
                "task_key",
                "source_split_dir",
                "fold_id",
                "method_short",
                "split_variant",
            )
            if column in selected
        ]
    )


def _stream_joint_driver_rows(
    samples_long: Path, physical_splits: pd.DataFrame
) -> pd.DataFrame:
    if physical_splits.empty:
        return pd.DataFrame(columns=_DRIVER_COLUMNS)
    physical_splits = physical_splits.copy()
    if "split_variant" not in physical_splits:
        physical_splits["split_variant"] = [
            _split_variant(method, path)
            for method, path in zip(
                physical_splits["method_short"],
                physical_splits["source_split_dir"],
            )
        ]
    selector_columns = [
        "task_key",
        "source_split_dir",
        "method_short",
        "fold_id",
        "split_variant",
    ]
    missing = [column for column in selector_columns if column not in physical_splits]
    if missing:
        raise KeyError(f"physical split selector is missing columns: {missing}")
    selectors = {
        (
            str(row.task_key),
            _normalise_path_key(row.source_split_dir),
            _method_slug(row.method_short),
            _method_slug(row.fold_id),
            _method_slug(row.split_variant),
        )
        for row in physical_splits[selector_columns].itertuples(index=False)
        if _normalise_path_key(row.source_split_dir)
    }
    selector_index = pd.MultiIndex.from_tuples(selectors, names=selector_columns)
    task_keys = set(physical_splits["task_key"].dropna().astype(str))
    pieces: list[pd.DataFrame] = []
    for chunk in _stream_csv(samples_long, requested_columns=_DRIVER_COLUMNS):
        if "scope" in chunk:
            chunk = chunk[chunk["scope"].astype(str).str.casefold().eq("ood")]
        chunk = chunk[
            chunk["space"].isin(["Y-space", "Z-space"])
            & chunk["task_key"].astype(str).isin(task_keys)
        ].copy()
        if chunk.empty:
            continue
        chunk_index = pd.MultiIndex.from_arrays(
            [
                chunk["task_key"].astype(str),
                chunk["source_split_dir"].map(_normalise_path_key),
                chunk["method_short"].map(_method_slug),
                chunk["fold_id"].map(_method_slug),
                pd.Series(
                    [
                        _split_variant(method, path)
                        for method, path in zip(
                            chunk["method_short"],
                            chunk["source_split_dir"],
                        )
                    ],
                    index=chunk.index,
                ).map(_method_slug),
            ],
            names=selector_columns,
        )
        selector_mask = chunk_index.isin(selector_index)
        if selector_mask.any():
            pieces.append(chunk.loc[selector_mask].copy())
    if not pieces:
        return pd.DataFrame(columns=_DRIVER_COLUMNS)
    return pd.concat(pieces, ignore_index=True, sort=False)


def _resolve_input_file(value: object, inventory_parent: Path) -> Path | None:
    if _missing(value):
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return inventory_parent / path


def _stable_seed(random_state: int, *parts: object) -> int:
    """Derive a process-independent NumPy seed from semantic identifiers."""

    payload = "\x1f".join([str(int(random_state)), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _relevant_robustness_splits(
    source_audit: pd.DataFrame,
    protocols: pd.DataFrame,
) -> pd.DataFrame:
    """Select joint physical splits and their canonical RandCV folds."""

    physical = _joint_physical_splits(source_audit, protocols)
    if protocols.empty:
        return physical
    task_keys = set(protocols["task_key"].dropna().astype(str))
    baseline = source_audit[
        source_audit["is_canonical"].astype(bool)
        & source_audit["task_key"].astype(str).isin(task_keys)
        & source_audit["method_short"].map(_normalise_method_identity).eq("randcv")
    ].copy()
    relevant = pd.concat([baseline, physical], ignore_index=True, sort=False)
    split_keys = [
        column
        for column in (
            "task_key",
            "method_short",
            "fold_id",
            "source_split_dir",
            "split_variant",
        )
        if column in relevant
    ]
    return relevant.drop_duplicates(split_keys).reset_index(drop=True)


def _values_agree(left: object, right: object) -> bool:
    if _missing(left) and _missing(right):
        return True
    if _missing(left) or _missing(right):
        return False
    left_number = pd.to_numeric(pd.Series([left]), errors="coerce").iloc[0]
    right_number = pd.to_numeric(pd.Series([right]), errors="coerce").iloc[0]
    if pd.notna(left_number) and pd.notna(right_number):
        return bool(
            np.isclose(float(left_number), float(right_number), rtol=1e-12, atol=1e-12)
        )
    return str(left).strip() == str(right).strip()


def _build_task_robustness_bundles(
    relevant_splits: pd.DataFrame,
    *,
    inventory_parent: Path,
) -> dict[str, dict[str, object]]:
    """Load canonical splits and validate a shared complement-consistent master."""

    bundles: dict[str, dict[str, object]] = {}
    for task_key, task_rows in relevant_splits.groupby(
        "task_key", sort=True, dropna=False
    ):
        ordering = task_rows.assign(
            _method_order=task_rows["method_short"]
            .map(_normalise_method_identity)
            .map(lambda value: 0 if value == "randcv" else 1),
            _fold_order=task_rows["fold_id"].map(_method_slug),
            _path_order=task_rows["source_split_dir"].map(_normalise_path_key),
        ).sort_values(["_method_order", "_fold_order", "_path_order"], kind="mergesort")
        split_records: list[dict[str, object]] = []
        master: pd.DataFrame | None = None
        master_ids: set[str] = set()
        canonical_features: list[str] | None = None
        target_values = (
            ordering["property"].dropna().astype(str).drop_duplicates().tolist()
        )
        if len(target_values) != 1:
            raise ValueError(
                f"task={task_key!r} has ambiguous target columns: {target_values}"
            )
        target_col = target_values[0]

        for row in ordering.itertuples(index=False):
            train_file = _resolve_input_file(
                getattr(row, "train_file", None), inventory_parent
            )
            test_file = _resolve_input_file(
                getattr(row, "test_file", None), inventory_parent
            )
            if train_file is None or test_file is None:
                raise ValueError(
                    f"robustness split files are missing for task={task_key!r}, "
                    f"method={getattr(row, 'method_short', None)!r}, fold={getattr(row, 'fold_id', None)!r}"
                )
            train = _read_csv_flexible(train_file)
            test = _read_csv_flexible(test_file)
            if "ID" not in train or "ID" not in test:
                raise KeyError(
                    f"robustness split is missing ID: {train_file} / {test_file}"
                )
            if target_col not in train or target_col not in test:
                raise KeyError(
                    f"robustness split is missing target column {target_col!r}"
                )

            feature_columns = parse_feature_columns(
                getattr(row, "feature_columns", None), train
            )
            feature_columns = [column for column in feature_columns if column in test]
            leaked_targets = [
                column
                for column in feature_columns
                if str(column).casefold() == target_col.casefold()
            ]
            if leaked_targets:
                raise ValueError(
                    "target leakage in inventory feature_columns for "
                    f"task={task_key!r}: target column {target_col!r} is listed as a feature"
                )
            if not feature_columns:
                raise ValueError(
                    f"no shared inventory feature columns for task={task_key!r}"
                )
            if canonical_features is None:
                canonical_features = feature_columns
            elif feature_columns != canonical_features:
                raise ValueError(
                    f"conflicting feature column definitions for task={task_key!r}: "
                    f"{canonical_features!r} != {feature_columns!r}"
                )

            train_ids = train["ID"].map(_normalise_id)
            test_ids = test["ID"].map(_normalise_id)
            if train_ids.eq("").any() or test_ids.eq("").any():
                raise ValueError(
                    f"missing normalized ID in robustness split for task={task_key!r}"
                )
            if train_ids.duplicated().any() or test_ids.duplicated().any():
                raise ValueError(
                    f"duplicate ID in robustness split for task={task_key!r}"
                )
            overlap = set(train_ids).intersection(test_ids)
            if overlap:
                raise ValueError(
                    f"train/test ID overlap for task={task_key!r}: {sorted(overlap)[:5]}"
                )

            selected_columns = ["ID", target_col, *feature_columns]
            combined = pd.concat(
                [train[selected_columns], test[selected_columns]],
                ignore_index=True,
            )
            combined.insert(0, "_audit_id", combined["ID"].map(_normalise_id))
            combined = combined.sort_values("_audit_id", kind="mergesort").reset_index(
                drop=True
            )
            current_ids = set(combined["_audit_id"])
            if master is None:
                master = combined
                master_ids = current_ids
            else:
                missing_ids = sorted(master_ids.difference(current_ids))
                extra_ids = sorted(current_ids.difference(master_ids))
                if missing_ids or extra_ids:
                    raise ValueError(
                        f"split ID complement mismatch for task={task_key!r}: "
                        f"missing IDs={missing_ids[:5]}, extra IDs={extra_ids[:5]}"
                    )
                current = combined.set_index("_audit_id")
                reference = master.set_index("_audit_id")
                for sample_id in sorted(master_ids):
                    conflicts = [
                        column
                        for column in [target_col, *feature_columns]
                        if not _values_agree(
                            reference.at[sample_id, column],
                            current.at[sample_id, column],
                        )
                    ]
                    if conflicts:
                        raise ValueError(
                            "conflicting feature/target values for "
                            f"task={task_key!r}, ID={sample_id!r}, columns={conflicts}"
                        )

            master_order = {
                value: index for index, value in enumerate(master["_audit_id"])
            }
            test_mask = np.zeros(len(master), dtype=bool)
            test_mask[[master_order[value] for value in test_ids]] = True
            split_records.append(
                {
                    "task_key": str(task_key),
                    "task_id": getattr(row, "task_id", task_key),
                    "alloy_family": getattr(row, "alloy_family", ""),
                    "dataset_name": getattr(row, "dataset_name", ""),
                    "property": target_col,
                    "method_short": getattr(row, "method_short"),
                    "split_variant": getattr(row, "split_variant", ""),
                    "fold_id": getattr(row, "fold_id"),
                    "source_split_dir": getattr(row, "source_split_dir"),
                    "train_file": train_file,
                    "test_file": test_file,
                    "published_Wy": getattr(row, "Wy", np.nan),
                    "published_Wz": getattr(row, "Wz", np.nan),
                    "test_mask": test_mask,
                    "n_train": int((~test_mask).sum()),
                    "n_test": int(test_mask.sum()),
                }
            )

        assert master is not None and canonical_features is not None
        bundles[str(task_key)] = {
            "master": master,
            "feature_columns": canonical_features,
            "target_col": target_col,
            "splits": split_records,
            "minimum_relevant_train_size": min(
                record["n_train"] for record in split_records
            ),
        }
    return bundles


def _test_row_metadata(
    record: dict[str, object], test_row: pd.Series, target_col: str
) -> None:
    for column, value in test_row.items():
        if column == "ID":
            record["ID"] = _normalise_id(value)
        elif column not in record or column == target_col:
            record[column] = value
        else:
            record[f"test_{column}"] = value


def _fallback_value(row: pd.Series, names: Sequence[str]) -> tuple[str, str]:
    lower_lookup = {str(column).casefold(): str(column) for column in row.index}
    for requested in names:
        column = lower_lookup.get(requested.casefold())
        if column is None:
            continue
        value = row.get(column)
        if not _missing(value) and str(value).strip():
            return str(value).strip(), column
    return "", ""


def _process_metadata(row: pd.Series) -> str:
    markers = (
        "process",
        "temp",
        "time",
        "aging",
        "anneal",
        "recrystal",
        "deformation",
        "cold",
        "quench",
        "solution",
        "hom_",
    )
    selected: list[str] = []
    for column, value in row.items():
        name = str(column).casefold()
        if not any(marker in name for marker in markers) and not re.fullmatch(
            r"st\d+", name
        ):
            continue
        if _missing(value) or not str(value).strip():
            continue
        selected.append(f"{column}={value}")
    return "; ".join(selected)


def _material_group_frame(test: pd.DataFrame) -> pd.DataFrame:
    """Assign every test row to a reproducible composition/reference group."""

    composition_columns = [
        str(column)
        for column in test.columns
        if re.search(r"\((?:wt|at)%\)", str(column), flags=re.IGNORECASE)
    ]
    records: list[dict[str, object]] = []
    for _, row in test.iterrows():
        sample_id = _normalise_id(row.get("ID"))
        signature = composition_signature(row, composition_columns)
        reference, reference_column = _fallback_value(
            row,
            ("References", "DOIs", "reference_id"),
        )
        if signature:
            key = f"composition:{signature}"
            basis = "composition_signature"
        elif reference:
            key = f"reference:{reference}"
            basis = reference_column
        else:
            key = f"id:{sample_id}"
            basis = "ID"
        records.append(
            {
                "_audit_id": sample_id,
                "composition_signature": signature,
                "material_group_key": key,
                "material_group_basis": basis,
            }
        )
    groups = pd.DataFrame.from_records(records, index=test.index)
    if groups.empty:
        groups["material_group_size"] = pd.Series(dtype=int)
    else:
        groups["material_group_size"] = groups["material_group_key"].map(
            groups["material_group_key"].value_counts()
        )
    return groups


def _attach_driver_metadata(
    ranked: pd.DataFrame,
    *,
    inventory_exact: dict[str, dict[str, object]],
    inventory_normalised: dict[str, dict[str, object]],
    inventory_parent: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if ranked.empty:
        return ranked.copy(), pd.DataFrame()
    output_records: list[dict[str, object]] = []
    integrity_records: list[dict[str, object]] = []
    groupers = [
        ranked["task_key"].astype(str),
        ranked["source_split_dir"].map(_normalise_path_key),
        ranked["method_short"].map(_method_slug),
        ranked["fold_id"].map(_method_slug),
    ]

    for _, driver_group in ranked.groupby(groupers, sort=False, dropna=False):
        source_path = driver_group["source_split_dir"].iloc[0]
        inventory, inventory_mode = _lookup_inventory(
            source_path,
            inventory_exact,
            inventory_normalised,
            method_short=driver_group["method_short"].iloc[0],
            fold_id=driver_group["fold_id"].iloc[0],
        )
        base_error = ""
        train = pd.DataFrame()
        test = pd.DataFrame()
        feature_columns: list[str] = []
        integrity: dict[str, object] = {}
        train_file: Path | None = None
        test_file: Path | None = None
        if inventory is None:
            base_error = "inventory_missing"
        else:
            train_file = _resolve_input_file(
                inventory.get("train_file"), inventory_parent
            )
            test_file = _resolve_input_file(
                inventory.get("test_file"), inventory_parent
            )
            if train_file is None or test_file is None:
                base_error = "split_file_path_missing"
            else:
                try:
                    train = _read_csv_flexible(train_file)
                    test = _read_csv_flexible(test_file)
                    requested = parse_feature_columns(
                        inventory.get("feature_columns"), train
                    )
                    feature_columns = [column for column in requested if column in test]
                    target_col = str(driver_group["property"].iloc[0])
                    is_extra = (
                        driver_group["method_short"]
                        .astype(str)
                        .str.casefold()
                        .eq("extra.")
                        .any()
                    )
                    integrity = validate_split_integrity(
                        train,
                        test,
                        target_col=target_col,
                        feature_columns=feature_columns,
                        expect_target_extrapolation=bool(is_extra),
                    )
                except (
                    FileNotFoundError,
                    OSError,
                    UnicodeError,
                    KeyError,
                    ValueError,
                ) as exc:
                    base_error = f"split_read_or_validation_error: {exc}"

        integrity_record = {
            "task_key": driver_group["task_key"].iloc[0],
            "task_id": driver_group["task_id"].iloc[0],
            "method_short": driver_group["method_short"].iloc[0],
            "fold_id": driver_group["fold_id"].iloc[0],
            "source_split_dir": source_path,
            "inventory_join_mode": inventory_mode,
            "driver_metadata_status": base_error or "ok",
            **integrity,
        }

        test_index: dict[str, list[int]] = defaultdict(list)
        if not test.empty and "ID" in test:
            for index, value in test["ID"].items():
                key = _normalise_id(value)
                if key:
                    test_index[key].append(index)
        material_groups = (
            _material_group_frame(test) if not test.empty else pd.DataFrame()
        )
        join_statuses: list[str] = []
        for _, driver in driver_group.iterrows():
            record = driver.to_dict()
            id_key = _normalise_id(driver.get("ID"))
            record["ID"] = id_key
            record["inventory_join_mode"] = inventory_mode
            record["driver_metadata_status"] = base_error or "ok"
            record["train_file"] = str(train_file) if train_file is not None else ""
            record["test_file"] = str(test_file) if test_file is not None else ""
            record["feature_columns"] = ";".join(feature_columns)
            record["feature_count"] = len(feature_columns)
            record.update(integrity)
            matches = test_index.get(id_key, [])
            record["test_id_match_count"] = len(matches)
            if matches:
                test_row = test.loc[matches[0]]
                target_col = str(driver.get("property"))
                _test_row_metadata(record, test_row, target_col)
                group_row = material_groups.loc[matches[0]]
                record["composition_signature"] = group_row["composition_signature"]
                record["material_group_key"] = group_row["material_group_key"]
                record["material_group"] = group_row["material_group_key"]
                record["material_group_basis"] = group_row["material_group_basis"]
                record["material_group_size"] = int(group_row["material_group_size"])
                record["process_metadata"] = _process_metadata(test_row)
                if target_col in test_row:
                    record["test_target_value"] = test_row[target_col]
                record["driver_test_join_status"] = (
                    "unique" if len(matches) == 1 else "duplicate_id"
                )
            else:
                record["composition_signature"] = ""
                record["material_group_key"] = f"id:{id_key}"
                record["material_group"] = f"id:{id_key}"
                record["material_group_basis"] = "ID"
                record["material_group_size"] = 0
                record["process_metadata"] = ""
                record["driver_test_join_status"] = "id_not_found"
            join_statuses.append(str(record["driver_test_join_status"]))
            output_records.append(record)
        unique_join_count = sum(status == "unique" for status in join_statuses)
        if join_statuses and unique_join_count == len(join_statuses):
            split_join_status = "unique"
        elif unique_join_count:
            split_join_status = "partial"
        else:
            split_join_status = "failed"
        integrity_record.update(
            {
                "driver_test_join_status": split_join_status,
                "driver_join_row_count": len(join_statuses),
                "driver_unique_join_count": unique_join_count,
                "driver_nonunique_join_count": len(join_statuses) - unique_join_count,
            }
        )
        integrity_records.append(integrity_record)
    return pd.DataFrame.from_records(output_records), pd.DataFrame.from_records(
        integrity_records
    )


def _legacy_subset(summary: pd.DataFrame, excluded_task: str) -> pd.DataFrame:
    mask = pd.Series(False, index=summary.index)
    for column in ("task_id", "task_key"):
        if column in summary:
            mask |= summary[column].astype(str).eq(excluded_task)
    return summary.loc[~mask].copy()


def _metric_contract(
    published: pd.DataFrame,
    *,
    ry_threshold: float,
    rz_threshold: float,
    legacy_exclude_task: str,
    robustness_run: bool,
) -> pd.DataFrame:
    scopes = {
        "current_9_task": published,
        "legacy_8_task": _legacy_subset(published, legacy_exclude_task),
    }
    records: list[dict[str, object]] = []
    ry_values = sorted({8.0, 9.0, 10.0, float(ry_threshold)})
    rz_values = sorted({5.0, 5.5, 6.0, 7.0, float(rz_threshold)})
    for scope, summary in scopes.items():
        records.append(
            {
                "record_type": "overlap_stats",
                "scope": scope,
                "Ry_threshold": float(ry_threshold),
                "Rz_threshold": float(rz_threshold),
                "high_dimensional_robustness_run": bool(robustness_run),
                **compute_joint_overlap_stats(
                    summary, ry_threshold=ry_threshold, rz_threshold=rz_threshold
                ),
            }
        )
        grid = threshold_grid(
            summary,
            ry_thresholds=ry_values,
            rz_thresholds=rz_values,
        )
        for row in grid.to_dict("records"):
            row = {
                "grid_high_Ry_count": row.pop("high_Ry_count"),
                "grid_high_Rz_count": row.pop("high_Rz_count"),
                "grid_joint_high_count": row.pop("joint_high_count"),
                **row,
            }
            records.append(
                {
                    "record_type": "threshold_grid",
                    "scope": scope,
                    "high_dimensional_robustness_run": bool(robustness_run),
                    **row,
                }
            )
    return pd.DataFrame.from_records(records)


def _safe_scalar_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return np.nan
    return float(numerator / denominator)


def _robustness_base_record(
    identity: dict[str, object],
    *,
    record_type: str,
    method_short: object,
    split_variant: object,
    fold_id: object,
    axis: str,
    representation: str,
    representation_seed: int | None,
    metric_name: str,
    metric_value: float,
    n_train: int,
    n_test: int,
    protocol_fold_count: int,
    randcv_baseline_value: float,
    severity_ratio: float,
    threshold: float,
    projection_count: int,
    effective_dimension: int,
    preprocessing_scope: str,
    transductive_proxy: bool,
    random_state: int,
    projection_random_state: int | None = None,
) -> dict[str, object]:
    return {
        "record_type": record_type,
        "scope": "canonical_metric_robustness",
        **identity,
        "method_short": method_short,
        "split_variant": split_variant,
        "fold_id": fold_id,
        "axis": axis,
        "representation": representation,
        "representation_seed": representation_seed,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "n_train": int(n_train),
        "n_test": int(n_test),
        "protocol_fold_count": int(protocol_fold_count),
        "randcv_baseline_value": randcv_baseline_value,
        "severity_ratio": severity_ratio,
        "Ry_variant": severity_ratio if axis == "Y" else np.nan,
        "Rz_variant": severity_ratio if axis == "Z" else np.nan,
        "threshold": float(threshold),
        "is_high": bool(np.isfinite(severity_ratio) and severity_ratio > threshold),
        "projection_count": int(projection_count),
        "effective_dimension": int(effective_dimension),
        "preprocessing_scope": preprocessing_scope,
        "transductive_proxy": bool(transductive_proxy),
        "random_state": int(random_state),
        "projection_random_state": projection_random_state,
        "null_status": "not_applicable",
        "status": "ok" if np.isfinite(metric_value) else "nonfinite_metric",
    }


def _split_context_key(split: dict[str, object] | pd.Series) -> tuple[str, ...]:
    getter = split.get
    return (
        str(getter("task_key", "")),
        _normalise_path_key(getter("source_split_dir", "")),
        _method_slug(getter("method_short", "")),
        _method_slug(getter("fold_id", "")),
        _method_slug(getter("split_variant", "")),
    )


def _embedding_dataset_key(identity: dict[str, object]) -> str | None:
    expected = tuple(
        str(identity.get(column, "")).strip()
        for column in ("alloy_family", "dataset_name", "property")
    )
    for dataset_key, case_values in YZ_DATASET_TO_CASE.items():
        if expected == tuple(str(value).strip() for value in case_values):
            return dataset_key
    return None


def _exact_proxy_record(
    identity: dict[str, object],
    *,
    level: str,
    method_short: object,
    split_variant: object,
    fold_id: object,
    metric_value: float,
    published_value: float,
    n_train: int,
    n_test: int,
    protocol_fold_count: int,
    baseline_value: float,
    severity_ratio: float,
    rz_threshold: float,
    random_state: int,
    status: str,
    solver_status: str,
    embedding_file: Path | None,
    embedding_dataset_key: str | None,
    embedding_audit: dict[str, object],
) -> dict[str, object]:
    record = _robustness_base_record(
        identity,
        record_type="current_proxy_reproduction",
        method_short=method_short,
        split_variant=split_variant,
        fold_id=fold_id,
        axis="Z",
        representation="legacy_tsne_2d_proxy",
        representation_seed=None,
        metric_name="exact_2d_w1",
        metric_value=metric_value,
        n_train=n_train,
        n_test=n_test,
        protocol_fold_count=protocol_fold_count,
        randcv_baseline_value=baseline_value,
        severity_ratio=severity_ratio,
        threshold=rz_threshold,
        projection_count=0,
        effective_dimension=2,
        preprocessing_scope="split_train_scale",
        transductive_proxy=True,
        random_state=random_state,
    )
    residual = (
        float(metric_value - published_value)
        if np.isfinite(metric_value) and np.isfinite(published_value)
        else np.nan
    )
    record.update(
        {
            "current_proxy_level": level,
            "published_metric_value": published_value,
            "recomputed_metric_value": metric_value,
            "current_proxy_residual": residual,
            "current_proxy_abs_residual": (
                abs(residual) if np.isfinite(residual) else np.nan
            ),
            "current_proxy_relative_residual": (
                abs(residual) / abs(published_value)
                if np.isfinite(residual)
                and np.isfinite(published_value)
                and published_value != 0
                else np.nan
            ),
            "embedding_dataset_key": embedding_dataset_key or "",
            "embedding_file": str(embedding_file) if embedding_file is not None else "",
            "embedding_join_status": embedding_audit.get("embedding_status", "unknown"),
            "solver_status": solver_status,
            "null_status": "not_run_exact_ot_prohibitive",
            "status": status,
            **embedding_audit,
        }
    )
    return record


def _run_exact_current_proxy_audit(
    bundles: dict[str, dict[str, object]],
    protocols: pd.DataFrame,
    *,
    embedding_data_dir: Path,
    ry_threshold: float,
    rz_threshold: float,
    random_state: int,
) -> tuple[pd.DataFrame, dict[tuple[str, ...], dict[str, object]]]:
    """Recompute the legacy split-scaled 2D proxy and retain deletion caches."""

    records: list[dict[str, object]] = []
    contexts: dict[tuple[str, ...], dict[str, object]] = {}
    for task_key, bundle in bundles.items():
        master = bundle["master"]
        assert isinstance(master, pd.DataFrame)
        splits = list(bundle["splits"])
        if not splits:
            continue
        target_col = str(bundle["target_col"])
        identity = {
            key: splits[0].get(key, "")
            for key in (
                "task_key",
                "task_id",
                "alloy_family",
                "dataset_name",
                "property",
            )
        }
        dataset_key = _embedding_dataset_key(identity)
        embedding_file = (
            embedding_data_dir / f"{dataset_key}__all_panel_test_points.csv"
            if dataset_key is not None
            else None
        )
        if dataset_key is None:
            embedding_audit: dict[str, object] = {
                "embedding_status": "missing_embedding_mapping",
                "embedding_id_column": "",
                "embedding_raw_row_count": 0,
                "embedding_invalid_row_count": 0,
                "embedding_duplicate_id_count": 0,
                "embedding_duplicate_row_count": 0,
                "embedding_duplicate_coordinate_conflict_id_count": 0,
                "embedding_master_join_missing_count": len(master),
            }
            coordinates = np.empty((0, 2), dtype=float)
        elif embedding_file is None or not embedding_file.exists():
            embedding_audit = {
                "embedding_status": "missing_embedding_file",
                "embedding_id_column": "",
                "embedding_raw_row_count": 0,
                "embedding_invalid_row_count": 0,
                "embedding_duplicate_id_count": 0,
                "embedding_duplicate_row_count": 0,
                "embedding_duplicate_coordinate_conflict_id_count": 0,
                "embedding_master_join_missing_count": len(master),
            }
            coordinates = np.empty((0, 2), dtype=float)
        else:
            try:
                embedding = _read_csv_flexible(embedding_file)
                coordinates, embedding_audit = prepare_legacy_embedding_coordinates(
                    embedding,
                    master["_audit_id"].tolist(),
                )
            except (OSError, UnicodeError, KeyError, ValueError) as exc:
                embedding_audit = {
                    "embedding_status": f"embedding_read_error: {exc}",
                    "embedding_id_column": "",
                    "embedding_raw_row_count": 0,
                    "embedding_invalid_row_count": 0,
                    "embedding_duplicate_id_count": 0,
                    "embedding_duplicate_row_count": 0,
                    "embedding_duplicate_coordinate_conflict_id_count": 0,
                    "embedding_master_join_missing_count": len(master),
                }
                coordinates = np.empty((0, 2), dtype=float)

        task_contexts: list[dict[str, object]] = []
        for split in splits:
            mask = np.asarray(split["test_mask"], dtype=bool)
            wy_before = standardized_target_w1(
                master.loc[~mask], master.loc[mask], target_col
            )
            if embedding_audit["embedding_status"] == "ok":
                try:
                    exact = exact_standardized_2d_ot(coordinates, mask)
                except (RuntimeError, ValueError) as exc:
                    exact = {
                        "metric_value": np.nan,
                        "cost_matrix": np.empty((int((~mask).sum()), int(mask.sum()))),
                        "status": f"exact_ot_error: {exc}",
                        "solver_status": "error",
                    }
            else:
                exact = {
                    "metric_value": np.nan,
                    "cost_matrix": np.empty((int((~mask).sum()), int(mask.sum()))),
                    "status": str(embedding_audit["embedding_status"]),
                    "solver_status": "not_run",
                }
            context = {
                **split,
                "target_col": target_col,
                "master": master,
                "test_ids": master.loc[mask, "_audit_id"].astype(str).tolist(),
                "Wy_fold_before": float(wy_before),
                "Wz_fold_before": float(exact["metric_value"]),
                "z_cost_matrix": exact["cost_matrix"],
                "exact_status": exact["status"],
                "solver_status": exact["solver_status"],
                "embedding_audit": dict(embedding_audit),
                "embedding_dataset_key": dataset_key,
                "embedding_file": embedding_file,
            }
            contexts[_split_context_key(context)] = context
            task_contexts.append(context)

        baseline_contexts = [
            context
            for context in task_contexts
            if _normalise_method_identity(context["method_short"]) == "randcv"
        ]
        baseline_contexts.sort(key=lambda item: _method_slug(item["fold_id"]))
        baseline_wy_values = np.asarray(
            [context["Wy_fold_before"] for context in baseline_contexts], dtype=float
        )
        baseline_wz_values = np.asarray(
            [context["Wz_fold_before"] for context in baseline_contexts], dtype=float
        )
        wy_baseline = (
            float(np.mean(baseline_wy_values))
            if len(baseline_contexts) == 5 and np.isfinite(baseline_wy_values).all()
            else np.nan
        )
        wz_baseline = (
            float(np.mean(baseline_wz_values))
            if len(baseline_contexts) == 5 and np.isfinite(baseline_wz_values).all()
            else np.nan
        )
        for context in task_contexts:
            context["Wy_random_exact"] = wy_baseline
            context["Wz_random_exact"] = wz_baseline

        for context in task_contexts:
            wz = float(context["Wz_fold_before"])
            records.append(
                _exact_proxy_record(
                    identity,
                    level="fold",
                    method_short=context["method_short"],
                    split_variant=context["split_variant"],
                    fold_id=context["fold_id"],
                    metric_value=wz,
                    published_value=float(context.get("published_Wz", np.nan)),
                    n_train=int(context["n_train"]),
                    n_test=int(context["n_test"]),
                    protocol_fold_count=(
                        5
                        if _normalise_method_identity(context["method_short"])
                        in {"randcv", "loco"}
                        else 1
                    ),
                    baseline_value=wz_baseline,
                    severity_ratio=_safe_scalar_ratio(wz, wz_baseline),
                    rz_threshold=rz_threshold,
                    random_state=random_state,
                    status=str(context["exact_status"]),
                    solver_status=str(context["solver_status"]),
                    embedding_file=embedding_file,
                    embedding_dataset_key=dataset_key,
                    embedding_audit=embedding_audit,
                )
            )

        if baseline_contexts:
            published_baseline = float(
                np.mean(
                    [
                        float(item.get("published_Wz", np.nan))
                        for item in baseline_contexts
                    ]
                )
            )
            baseline_summary_status = (
                "ok"
                if np.isfinite(wz_baseline)
                else (
                    str(embedding_audit["embedding_status"])
                    if embedding_audit["embedding_status"] != "ok"
                    else "incomplete_exact_fold_metrics"
                )
            )
            records.append(
                _exact_proxy_record(
                    identity,
                    level="randcv_baseline",
                    method_short="RandCV",
                    split_variant=baseline_contexts[0]["split_variant"],
                    fold_id="protocol_mean",
                    metric_value=wz_baseline,
                    published_value=published_baseline,
                    n_train=int(
                        np.mean([item["n_train"] for item in baseline_contexts])
                    ),
                    n_test=int(np.mean([item["n_test"] for item in baseline_contexts])),
                    protocol_fold_count=len(baseline_contexts),
                    baseline_value=wz_baseline,
                    severity_ratio=_safe_scalar_ratio(wz_baseline, wz_baseline),
                    rz_threshold=rz_threshold,
                    random_state=random_state,
                    status=baseline_summary_status,
                    solver_status="fold_mean",
                    embedding_file=embedding_file,
                    embedding_dataset_key=dataset_key,
                    embedding_audit=embedding_audit,
                )
            )

        task_protocols = protocols[protocols["task_key"].astype(str).eq(str(task_key))]
        for _, protocol in task_protocols.iterrows():
            method_key = _normalise_method_identity(protocol["method_short"])
            variant_key = _method_slug(protocol.get("split_variant"))
            physical = [
                context
                for context in task_contexts
                if _normalise_method_identity(context["method_short"]) == method_key
                and _method_slug(context.get("split_variant")) == variant_key
            ]
            physical.sort(key=lambda item: _method_slug(item["fold_id"]))
            if not physical:
                continue
            wy_values = np.asarray(
                [context["Wy_fold_before"] for context in physical], dtype=float
            )
            wz_values = np.asarray(
                [context["Wz_fold_before"] for context in physical], dtype=float
            )
            wy_protocol = (
                float(np.mean(wy_values)) if np.isfinite(wy_values).all() else np.nan
            )
            wz_protocol = (
                float(np.mean(wz_values)) if np.isfinite(wz_values).all() else np.nan
            )
            for context in physical:
                context["protocol_Wy_before_exact"] = wy_protocol
                context["protocol_Wz_before_exact"] = wz_protocol
                context["protocol_fold_count_exact"] = len(physical)
            protocol_summary_status = (
                "ok"
                if np.isfinite(wz_protocol)
                else (
                    str(embedding_audit["embedding_status"])
                    if embedding_audit["embedding_status"] != "ok"
                    else "incomplete_exact_fold_metrics"
                )
            )
            records.append(
                _exact_proxy_record(
                    identity,
                    level="protocol",
                    method_short=protocol["method_short"],
                    split_variant=protocol["split_variant"],
                    fold_id="protocol_mean",
                    metric_value=wz_protocol,
                    published_value=float(protocol.get("protocol_Wz", np.nan)),
                    n_train=int(np.mean([item["n_train"] for item in physical])),
                    n_test=int(np.mean([item["n_test"] for item in physical])),
                    protocol_fold_count=len(physical),
                    baseline_value=wz_baseline,
                    severity_ratio=_safe_scalar_ratio(wz_protocol, wz_baseline),
                    rz_threshold=rz_threshold,
                    random_state=random_state,
                    status=protocol_summary_status,
                    solver_status="fold_mean",
                    embedding_file=embedding_file,
                    embedding_dataset_key=dataset_key,
                    embedding_audit=embedding_audit,
                )
            )

    return pd.DataFrame.from_records(records), contexts


def _fold_equal_protocol_update(
    protocol_before: float,
    fold_before: float,
    fold_after: float,
    fold_count: int,
) -> float:
    if (
        fold_count <= 0
        or not np.isfinite(protocol_before)
        or not np.isfinite(fold_before)
        or not np.isfinite(fold_after)
    ):
        return np.nan
    return float((fold_count * protocol_before - fold_before + fold_after) / fold_count)


def _joint_high_flag(
    ry: float,
    rz: float,
    *,
    ry_threshold: float,
    rz_threshold: float,
) -> object:
    if not np.isfinite(ry) or not np.isfinite(rz):
        return pd.NA
    return bool(ry > ry_threshold and rz > rz_threshold)


def _sensitivity_fields(
    prefix: str,
    result: dict[str, object],
    context: dict[str, object],
    *,
    ry_threshold: float,
    rz_threshold: float,
) -> dict[str, object]:
    wy_fold_before = float(context.get("Wy_fold_before", np.nan))
    wz_fold_before = float(context.get("Wz_fold_before", np.nan))
    wy_fold_after = float(result.get("Wy_fold_after", np.nan))
    wz_fold_after = float(result.get("Wz_fold_after", np.nan))
    fold_count = int(context.get("protocol_fold_count_exact", 0) or 0)
    wy_protocol_before = float(context.get("protocol_Wy_before_exact", np.nan))
    wz_protocol_before = float(context.get("protocol_Wz_before_exact", np.nan))
    wy_protocol_after = _fold_equal_protocol_update(
        wy_protocol_before,
        wy_fold_before,
        wy_fold_after,
        fold_count,
    )
    wz_protocol_after = _fold_equal_protocol_update(
        wz_protocol_before,
        wz_fold_before,
        wz_fold_after,
        fold_count,
    )
    wy_baseline = float(context.get("Wy_random_exact", np.nan))
    wz_baseline = float(context.get("Wz_random_exact", np.nan))
    ry_before = _safe_scalar_ratio(wy_protocol_before, wy_baseline)
    rz_before = _safe_scalar_ratio(wz_protocol_before, wz_baseline)
    ry_after = _safe_scalar_ratio(wy_protocol_after, wy_baseline)
    rz_after = _safe_scalar_ratio(wz_protocol_after, wz_baseline)
    return {
        f"{prefix}_removed_count": result.get("removed_count", 0),
        f"{prefix}_remaining_test_count": result.get("remaining_test_count", np.nan),
        f"{prefix}_sensitivity_status": result.get("status", "not_run"),
        f"{prefix}_y_sensitivity_status": result.get("y_status", "not_run"),
        f"{prefix}_z_sensitivity_status": result.get("z_status", "not_run"),
        f"{prefix}_Wy_fold_before": wy_fold_before,
        f"{prefix}_Wy_fold_after": wy_fold_after,
        f"{prefix}_Wz_fold_before": wz_fold_before,
        f"{prefix}_Wz_fold_after": wz_fold_after,
        f"{prefix}_Wy_protocol_before": wy_protocol_before,
        f"{prefix}_Wy_protocol_after": wy_protocol_after,
        f"{prefix}_Wz_protocol_before": wz_protocol_before,
        f"{prefix}_Wz_protocol_after": wz_protocol_after,
        f"{prefix}_Ry_before": ry_before,
        f"{prefix}_Ry_after": ry_after,
        f"{prefix}_Rz_before": rz_before,
        f"{prefix}_Rz_after": rz_after,
        f"{prefix}_joint_high_before": _joint_high_flag(
            ry_before,
            rz_before,
            ry_threshold=ry_threshold,
            rz_threshold=rz_threshold,
        ),
        f"{prefix}_joint_high_after": _joint_high_flag(
            ry_after,
            rz_after,
            ry_threshold=ry_threshold,
            rz_threshold=rz_threshold,
        ),
    }


def _prioritize_deletion_sets(
    candidate_sets: dict[tuple[str, str], tuple[int, ...] | None],
    candidate_ranks: dict[tuple[str, str], float],
    *,
    limit: int,
) -> list[tuple[int, ...]]:
    """Order unique removal sets by best driver rank and stable tie-breaks."""

    removal_best_rank: dict[tuple[int, ...], float] = {}
    for candidate, removal in candidate_sets.items():
        if removal is None:
            continue
        rank = float(candidate_ranks.get(candidate, np.inf))
        removal_best_rank[removal] = min(
            rank,
            removal_best_rank.get(removal, np.inf),
        )
    ordered = sorted(
        removal_best_rank,
        key=lambda removal: (
            removal_best_rank[removal],
            len(removal),
            removal,
        ),
    )
    return ordered[: max(int(limit), 0)]


def _attach_exact_deletion_sensitivity(
    drivers: pd.DataFrame,
    contexts: dict[tuple[str, ...], dict[str, object]],
    *,
    ry_threshold: float,
    rz_threshold: float,
    max_exact_deletion_sets_per_split: int = 40,
) -> pd.DataFrame:
    """Attach independent sample/group deletion effects to every driver row."""

    if drivers.empty:
        return drivers.copy()
    output: list[pd.DataFrame] = []
    group_columns = [
        "task_key",
        "source_split_dir",
        "method_short",
        "fold_id",
        "split_variant",
    ]
    working = drivers.copy()
    if "split_variant" not in working:
        working["split_variant"] = [
            _split_variant(method, path)
            for method, path in zip(
                working["method_short"], working["source_split_dir"]
            )
        ]
    for _, group in working.groupby(group_columns, dropna=False, sort=False):
        context_key = _split_context_key(group.iloc[0])
        context = contexts.get(context_key)
        if context is None:
            missing = group.copy()
            for prefix in ("sample", "group"):
                missing = missing.assign(
                    **_sensitivity_fields(
                        prefix,
                        {
                            "status": "missing_exact_split_context",
                            "y_status": "not_run",
                            "z_status": "not_run",
                        },
                        {},
                        ry_threshold=ry_threshold,
                        rz_threshold=rz_threshold,
                    )
                )
            output.append(missing)
            continue

        test_ids = [str(value) for value in context["test_ids"]]
        id_to_index = {value: index for index, value in enumerate(test_ids)}
        test_file = context.get("test_file")
        try:
            test = _read_csv_flexible(Path(test_file)) if test_file else pd.DataFrame()
            material = _material_group_frame(test)
            material_by_id = dict(
                zip(
                    material["_audit_id"].astype(str),
                    material["material_group_key"].astype(str),
                )
            )
            group_keys = [
                material_by_id.get(value, f"id:{value}") for value in test_ids
            ]
        except (OSError, UnicodeError, KeyError, ValueError):
            group_keys = [f"id:{value}" for value in test_ids]

        sample_candidates = sorted(
            {
                _normalise_id(value)
                for value in group["ID"].tolist()
                if _normalise_id(value)
            }
        )
        group_candidates = sorted(
            {
                str(value)
                for value in group.get("material_group_key", pd.Series(dtype=object))
                if not _missing(value) and str(value)
            }
        )
        candidate_sets: dict[tuple[str, str], tuple[int, ...] | None] = {}
        for sample_id in sample_candidates:
            candidate_sets[("sample", sample_id)] = (
                (id_to_index[sample_id],) if sample_id in id_to_index else None
            )
        for material_key in group_candidates:
            indices = tuple(
                index for index, value in enumerate(group_keys) if value == material_key
            )
            candidate_sets[("group", material_key)] = indices or None

        rank_source = (
            group["rank"]
            if "rank" in group
            else pd.Series(np.nan, index=group.index, dtype=float)
        )
        numeric_rank = pd.to_numeric(rank_source, errors="coerce")
        candidate_ranks: dict[tuple[str, str], float] = {}
        for sample_id in sample_candidates:
            matches = group["ID"].map(_normalise_id).eq(sample_id)
            ranks = numeric_rank.loc[matches].dropna()
            candidate_ranks[("sample", sample_id)] = (
                float(ranks.min()) if not ranks.empty else np.inf
            )
        for material_key in group_candidates:
            matches = group["material_group_key"].astype(str).eq(material_key)
            ranks = numeric_rank.loc[matches].dropna()
            candidate_ranks[("group", material_key)] = (
                float(ranks.min()) if not ranks.empty else np.inf
            )
        unique_set_count = len(
            {removal for removal in candidate_sets.values() if removal is not None}
        )
        prioritized_sets = _prioritize_deletion_sets(
            candidate_sets,
            candidate_ranks,
            limit=unique_set_count,
        )
        accepted_sets = set(prioritized_sets[:max_exact_deletion_sets_per_split])
        target_col = str(context["target_col"])
        master = context["master"]
        assert isinstance(master, pd.DataFrame)
        mask = np.asarray(context["test_mask"], dtype=bool)
        train = master.loc[~mask]
        full_test = master.loc[mask].reset_index(drop=True)
        y_cache: dict[tuple[int, ...], dict[str, object]] = {}
        for removal in accepted_sets:
            keep = np.ones(len(full_test), dtype=bool)
            keep[list(removal)] = False
            remaining = int(keep.sum())
            if remaining == 0:
                y_cache[removal] = {
                    "metric_value": np.nan,
                    "status": "skipped_empty_test",
                    "removed_count": len(removal),
                    "remaining_test_count": 0,
                }
            else:
                value = standardized_target_w1(
                    train,
                    full_test.loc[keep],
                    target_col,
                )
                y_cache[removal] = {
                    "metric_value": value,
                    "status": "ok_degenerate_n1" if remaining == 1 else "ok",
                    "removed_count": len(removal),
                    "remaining_test_count": remaining,
                }

        z_available = (
            str(context.get("exact_status", "")).startswith("ok")
            and isinstance(context.get("z_cost_matrix"), np.ndarray)
            and np.asarray(context["z_cost_matrix"]).size > 0
        )
        z_cache = (
            solve_exact_ot_deletions(
                np.asarray(context["z_cost_matrix"], dtype=float),
                {removal: removal for removal in accepted_sets},
            )
            if z_available and accepted_sets
            else {}
        )
        candidate_results: dict[tuple[str, str], dict[str, object]] = {}
        for candidate, removal in candidate_sets.items():
            if removal is None:
                kind = candidate[0]
                candidate_results[candidate] = {
                    "status": f"{kind}_not_found",
                    "y_status": "not_run",
                    "z_status": "not_run",
                    "removed_count": 0,
                    "remaining_test_count": len(full_test),
                    "Wy_fold_after": np.nan,
                    "Wz_fold_after": np.nan,
                }
                continue
            if removal not in accepted_sets:
                candidate_results[candidate] = {
                    "status": "skipped_safety_cap",
                    "y_status": "skipped_safety_cap",
                    "z_status": "skipped_safety_cap",
                    "removed_count": len(removal),
                    "remaining_test_count": len(full_test) - len(removal),
                    "Wy_fold_after": np.nan,
                    "Wz_fold_after": np.nan,
                }
                continue
            y_result = y_cache[removal]
            if z_available:
                z_result = z_cache[removal]
                z_status = str(z_result["status"])
                combined_status = z_status
                wz_after = float(z_result["metric_value"])
            else:
                z_status = str(context.get("exact_status", "missing_embedding"))
                combined_status = f"partial_missing_z:{z_status}"
                wz_after = np.nan
            candidate_results[candidate] = {
                "status": combined_status,
                "y_status": y_result["status"],
                "z_status": z_status,
                "removed_count": y_result["removed_count"],
                "remaining_test_count": y_result["remaining_test_count"],
                "Wy_fold_after": float(y_result["metric_value"]),
                "Wz_fold_after": wz_after,
            }

        enriched_records: list[dict[str, object]] = []
        for _, driver in group.iterrows():
            record = driver.to_dict()
            sample_id = _normalise_id(driver.get("ID"))
            material_key = str(driver.get("material_group_key", ""))
            record.update(
                _sensitivity_fields(
                    "sample",
                    candidate_results.get(
                        ("sample", sample_id),
                        {
                            "status": "sample_not_found",
                            "y_status": "not_run",
                            "z_status": "not_run",
                        },
                    ),
                    context,
                    ry_threshold=ry_threshold,
                    rz_threshold=rz_threshold,
                )
            )
            record.update(
                _sensitivity_fields(
                    "group",
                    candidate_results.get(
                        ("group", material_key),
                        {
                            "status": "group_not_found",
                            "y_status": "not_run",
                            "z_status": "not_run",
                        },
                    ),
                    context,
                    ry_threshold=ry_threshold,
                    rz_threshold=rz_threshold,
                )
            )
            record["exact_deletion_unique_set_count"] = unique_set_count
            record["exact_deletion_executed_set_count"] = len(accepted_sets)
            record["exact_deletion_safety_cap"] = max_exact_deletion_sets_per_split
            enriched_records.append(record)
        output.append(pd.DataFrame.from_records(enriched_records))
    return pd.concat(output, ignore_index=True, sort=False)


def _summarize_tsne_seed_rows(
    seed_rows: pd.DataFrame,
    *,
    rz_threshold: float,
) -> dict[str, object]:
    """Summarize t-SNE seed ratios using a strict, tie-excluding majority."""

    ratios = (
        pd.to_numeric(seed_rows["severity_ratio"], errors="coerce").dropna().to_numpy()
    )
    seed_row_count = int(len(seed_rows))
    seed_count = int(ratios.size)
    majority_required = seed_count // 2 + 1 if seed_count else 0
    high_count = int(np.count_nonzero(ratios > rz_threshold))
    evidence_status = "ok" if seed_count >= 3 else "insufficient_seeds"
    return {
        "tsne_seed_row_count": seed_row_count,
        "tsne_seed_count": seed_count,
        "tsne_ratio_min": float(np.min(ratios)) if ratios.size else np.nan,
        "tsne_ratio_median": float(np.median(ratios)) if ratios.size else np.nan,
        "tsne_ratio_max": float(np.max(ratios)) if ratios.size else np.nan,
        "high_seed_count": high_count,
        "high_seed_fraction": float(high_count / seed_count) if seed_count else np.nan,
        "majority_required": majority_required,
        "seed_evidence_status": evidence_status,
        "majority_high": bool(
            evidence_status == "ok" and high_count >= majority_required
        ),
    }


def _attach_protocol_null_summary(
    record: dict[str, object],
    *,
    observed: float,
    null_values: np.ndarray,
    requested_iterations: int,
) -> dict[str, object]:
    """Attach empirical-null statistics without hiding incomplete iterations."""

    values = np.asarray(null_values, dtype=float).ravel()
    finite_count = int(np.isfinite(values).sum())
    requested_iterations = int(requested_iterations)
    if finite_count == 0:
        null_status = "invalid_null"
    elif len(values) == requested_iterations and finite_count == requested_iterations:
        null_status = "ok"
    else:
        null_status = "partial_nonfinite_null"
    updated = dict(record)
    updated.update(
        {
            "requested_null_iterations": requested_iterations,
            "null_status": null_status,
            **summarize_empirical_null(observed, values),
        }
    )
    if updated.get("status") == "ok":
        updated["status"] = null_status
    return updated


def _run_metric_robustness(
    bundles: dict[str, dict[str, object]],
    protocols: pd.DataFrame,
    *,
    ry_threshold: float,
    rz_threshold: float,
    random_splits: int,
    random_state: int,
    pca_components: int,
    tsne_seeds: Sequence[int],
    sliced_projections: int,
) -> pd.DataFrame:
    """Compute target and task-global representation robustness records."""

    if random_splits <= 0:
        raise ValueError("--random-splits must be positive when robustness is enabled")
    if sliced_projections <= 0:
        raise ValueError("--sliced-projections must be positive")
    records: list[dict[str, object]] = []

    for task_key, bundle in bundles.items():
        master = bundle["master"]
        assert isinstance(master, pd.DataFrame)
        target_col = str(bundle["target_col"])
        target = pd.to_numeric(master[target_col], errors="coerce").to_numpy(
            dtype=float
        )
        if not np.isfinite(target).all():
            raise ValueError(f"task={task_key!r} has non-finite target values")
        split_records = list(bundle["splits"])
        baseline_splits = [
            split
            for split in split_records
            if _normalise_method_identity(split["method_short"]) == "randcv"
        ]
        if len(baseline_splits) != 5:
            raise ValueError(
                f"task={task_key!r} requires exactly five canonical RandCV folds; "
                f"found {len(baseline_splits)}"
            )
        baseline_splits.sort(key=lambda split: _method_slug(split["fold_id"]))
        baseline_masks = np.stack([split["test_mask"] for split in baseline_splits])
        identity = {
            key: baseline_splits[0].get(key, "")
            for key in (
                "task_key",
                "task_id",
                "alloy_family",
                "dataset_name",
                "property",
            )
        }

        target_baseline_folds = standardized_target_w1_for_masks(target, baseline_masks)
        target_baseline = float(np.mean(target_baseline_folds))
        for split, value in zip(baseline_splits, target_baseline_folds):
            ratio = _safe_scalar_ratio(float(value), target_baseline)
            records.append(
                _robustness_base_record(
                    identity,
                    record_type="randcv_fold",
                    method_short=split["method_short"],
                    split_variant=split["split_variant"],
                    fold_id=split["fold_id"],
                    axis="Y",
                    representation="target",
                    representation_seed=None,
                    metric_name="standardized_w1",
                    metric_value=float(value),
                    n_train=int(split["n_train"]),
                    n_test=int(split["n_test"]),
                    protocol_fold_count=5,
                    randcv_baseline_value=target_baseline,
                    severity_ratio=ratio,
                    threshold=ry_threshold,
                    projection_count=0,
                    effective_dimension=1,
                    preprocessing_scope="split_train_scale",
                    transductive_proxy=False,
                    random_state=random_state,
                )
            )
        records.append(
            _robustness_base_record(
                identity,
                record_type="randcv_baseline",
                method_short="RandCV",
                split_variant=baseline_splits[0]["split_variant"],
                fold_id="protocol_mean",
                axis="Y",
                representation="target",
                representation_seed=None,
                metric_name="standardized_w1",
                metric_value=target_baseline,
                n_train=int(np.mean([split["n_train"] for split in baseline_splits])),
                n_test=int(np.mean([split["n_test"] for split in baseline_splits])),
                protocol_fold_count=5,
                randcv_baseline_value=target_baseline,
                severity_ratio=1.0,
                threshold=ry_threshold,
                projection_count=0,
                effective_dimension=1,
                preprocessing_scope="split_train_scale",
                transductive_proxy=False,
                random_state=random_state,
            )
        )

        representations = prepare_task_global_representations(
            master,
            feature_columns=bundle["feature_columns"],
            pca_components=pca_components,
            minimum_relevant_train_size=int(bundle["minimum_relevant_train_size"]),
            tsne_seeds=tsne_seeds,
        )
        representation_baselines: dict[
            tuple[str, int | None], tuple[np.ndarray, float, dict[str, object], int]
        ] = {}
        for key, (values, metadata) in representations.items():
            representation, representation_seed = key
            direction_family = (
                "tsne_2d_proxy" if representation == "tsne_2d_proxy" else representation
            )
            projection_seed = _stable_seed(
                random_state, task_key, direction_family, "projection_directions"
            )
            baseline_values = fixed_projection_swd_for_masks(
                values,
                baseline_masks,
                projections=sliced_projections,
                random_state=projection_seed,
            )
            baseline_value = float(np.mean(baseline_values))
            representation_baselines[key] = (
                baseline_values,
                baseline_value,
                metadata,
                projection_seed,
            )
            for split, value in zip(baseline_splits, baseline_values):
                ratio = _safe_scalar_ratio(float(value), baseline_value)
                records.append(
                    _robustness_base_record(
                        identity,
                        record_type="randcv_fold",
                        method_short=split["method_short"],
                        split_variant=split["split_variant"],
                        fold_id=split["fold_id"],
                        axis="Z",
                        representation=representation,
                        representation_seed=representation_seed,
                        metric_name="fixed_projection_swd",
                        metric_value=float(value),
                        n_train=int(split["n_train"]),
                        n_test=int(split["n_test"]),
                        protocol_fold_count=5,
                        randcv_baseline_value=baseline_value,
                        severity_ratio=ratio,
                        threshold=rz_threshold,
                        projection_count=sliced_projections,
                        effective_dimension=int(metadata["effective_dimension"]),
                        preprocessing_scope=str(metadata["preprocessing_scope"]),
                        transductive_proxy=bool(metadata["transductive_proxy"]),
                        random_state=random_state,
                        projection_random_state=projection_seed,
                    )
                )
            records.append(
                _robustness_base_record(
                    identity,
                    record_type="randcv_baseline",
                    method_short="RandCV",
                    split_variant=baseline_splits[0]["split_variant"],
                    fold_id="protocol_mean",
                    axis="Z",
                    representation=representation,
                    representation_seed=representation_seed,
                    metric_name="fixed_projection_swd",
                    metric_value=baseline_value,
                    n_train=int(
                        np.mean([split["n_train"] for split in baseline_splits])
                    ),
                    n_test=int(np.mean([split["n_test"] for split in baseline_splits])),
                    protocol_fold_count=5,
                    randcv_baseline_value=baseline_value,
                    severity_ratio=1.0,
                    threshold=rz_threshold,
                    projection_count=sliced_projections,
                    effective_dimension=int(metadata["effective_dimension"]),
                    preprocessing_scope=str(metadata["preprocessing_scope"]),
                    transductive_proxy=bool(metadata["transductive_proxy"]),
                    random_state=random_state,
                    projection_random_state=projection_seed,
                )
            )

        task_protocols = protocols[protocols["task_key"].astype(str).eq(str(task_key))]
        for _, protocol in task_protocols.iterrows():
            method_key = _normalise_method_identity(protocol["method_short"])
            variant_key = _method_slug(protocol.get("split_variant"))
            physical = [
                split
                for split in split_records
                if _normalise_method_identity(split["method_short"]) == method_key
                and _method_slug(split.get("split_variant")) == variant_key
            ]
            physical.sort(key=lambda split: _method_slug(split["fold_id"]))
            if not physical:
                raise ValueError(
                    f"no canonical physical splits for task={task_key!r}, method={protocol['method_short']!r}"
                )
            observed_masks = np.stack([split["test_mask"] for split in physical])
            if method_key == "loco" and not np.all(observed_masks.sum(axis=0) == 1):
                raise ValueError(
                    f"LOCO folds must be disjoint and cover the task master for task={task_key!r}"
                )
            test_sizes = [int(split["n_test"]) for split in physical]
            mask_seed = _stable_seed(
                random_state, task_key, method_key, variant_key, "matched_masks"
            )
            null_masks = matched_partition_masks(
                len(master),
                test_sizes,
                iterations=random_splits,
                random_state=mask_seed,
            )
            null_design = (
                "matched_disjoint_covering_loco_partition"
                if method_key == "loco"
                else "matched_test_size_partition"
            )

            observed_target_folds = standardized_target_w1_for_masks(
                target, observed_masks
            )
            observed_target = float(np.mean(observed_target_folds))
            target_ratio = _safe_scalar_ratio(observed_target, target_baseline)
            target_null_folds = standardized_target_w1_for_masks(target, null_masks)
            target_null = np.mean(target_null_folds, axis=1)
            for split, value in zip(physical, observed_target_folds):
                records.append(
                    _robustness_base_record(
                        identity,
                        record_type="observed_fold",
                        method_short=protocol["method_short"],
                        split_variant=protocol["split_variant"],
                        fold_id=split["fold_id"],
                        axis="Y",
                        representation="target",
                        representation_seed=None,
                        metric_name="standardized_w1",
                        metric_value=float(value),
                        n_train=int(split["n_train"]),
                        n_test=int(split["n_test"]),
                        protocol_fold_count=len(physical),
                        randcv_baseline_value=target_baseline,
                        severity_ratio=_safe_scalar_ratio(
                            float(value), target_baseline
                        ),
                        threshold=ry_threshold,
                        projection_count=0,
                        effective_dimension=1,
                        preprocessing_scope="split_train_scale",
                        transductive_proxy=False,
                        random_state=random_state,
                    )
                )
            target_protocol_record = _robustness_base_record(
                identity,
                record_type="observed_protocol",
                method_short=protocol["method_short"],
                split_variant=protocol["split_variant"],
                fold_id="protocol_mean",
                axis="Y",
                representation="target",
                representation_seed=None,
                metric_name="standardized_w1",
                metric_value=observed_target,
                n_train=int(np.mean([split["n_train"] for split in physical])),
                n_test=int(np.mean(test_sizes)),
                protocol_fold_count=len(physical),
                randcv_baseline_value=target_baseline,
                severity_ratio=target_ratio,
                threshold=ry_threshold,
                projection_count=0,
                effective_dimension=1,
                preprocessing_scope="split_train_scale",
                transductive_proxy=False,
                random_state=random_state,
            )
            target_protocol_record.update(
                {
                    "null_design": null_design,
                    "null_test_sizes": ";".join(map(str, test_sizes)),
                    "matched_mask_random_state": mask_seed,
                }
            )
            target_protocol_record = _attach_protocol_null_summary(
                target_protocol_record,
                observed=observed_target,
                null_values=target_null,
                requested_iterations=random_splits,
            )
            records.append(target_protocol_record)

            for key, (values, _) in representations.items():
                representation, representation_seed = key
                baseline_values, baseline_value, metadata, projection_seed = (
                    representation_baselines[key]
                )
                observed_values = fixed_projection_swd_for_masks(
                    values,
                    observed_masks,
                    projections=sliced_projections,
                    random_state=projection_seed,
                )
                observed_protocol = float(np.mean(observed_values))
                severity_ratio = _safe_scalar_ratio(observed_protocol, baseline_value)
                flat_null_masks = null_masks.reshape(-1, len(master))
                null_fold_values = fixed_projection_swd_for_masks(
                    values,
                    flat_null_masks,
                    projections=sliced_projections,
                    random_state=projection_seed,
                ).reshape(random_splits, len(physical))
                null_values = np.mean(null_fold_values, axis=1)
                for split, value in zip(physical, observed_values):
                    records.append(
                        _robustness_base_record(
                            identity,
                            record_type="observed_fold",
                            method_short=protocol["method_short"],
                            split_variant=protocol["split_variant"],
                            fold_id=split["fold_id"],
                            axis="Z",
                            representation=representation,
                            representation_seed=representation_seed,
                            metric_name="fixed_projection_swd",
                            metric_value=float(value),
                            n_train=int(split["n_train"]),
                            n_test=int(split["n_test"]),
                            protocol_fold_count=len(physical),
                            randcv_baseline_value=baseline_value,
                            severity_ratio=_safe_scalar_ratio(
                                float(value), baseline_value
                            ),
                            threshold=rz_threshold,
                            projection_count=sliced_projections,
                            effective_dimension=int(metadata["effective_dimension"]),
                            preprocessing_scope=str(metadata["preprocessing_scope"]),
                            transductive_proxy=bool(metadata["transductive_proxy"]),
                            random_state=random_state,
                            projection_random_state=projection_seed,
                        )
                    )
                protocol_record = _robustness_base_record(
                    identity,
                    record_type="observed_protocol",
                    method_short=protocol["method_short"],
                    split_variant=protocol["split_variant"],
                    fold_id="protocol_mean",
                    axis="Z",
                    representation=representation,
                    representation_seed=representation_seed,
                    metric_name="fixed_projection_swd",
                    metric_value=observed_protocol,
                    n_train=int(np.mean([split["n_train"] for split in physical])),
                    n_test=int(np.mean(test_sizes)),
                    protocol_fold_count=len(physical),
                    randcv_baseline_value=baseline_value,
                    severity_ratio=severity_ratio,
                    threshold=rz_threshold,
                    projection_count=sliced_projections,
                    effective_dimension=int(metadata["effective_dimension"]),
                    preprocessing_scope=str(metadata["preprocessing_scope"]),
                    transductive_proxy=bool(metadata["transductive_proxy"]),
                    random_state=random_state,
                    projection_random_state=projection_seed,
                )
                protocol_record.update(
                    {
                        "null_design": null_design,
                        "null_test_sizes": ";".join(map(str, test_sizes)),
                        "matched_mask_random_state": mask_seed,
                    }
                )
                protocol_record = _attach_protocol_null_summary(
                    protocol_record,
                    observed=observed_protocol,
                    null_values=null_values,
                    requested_iterations=random_splits,
                )
                records.append(protocol_record)

    frame = pd.DataFrame.from_records(records)
    protocol_tsne = frame[
        frame["record_type"].eq("observed_protocol")
        & frame["axis"].eq("Z")
        & frame["representation"].eq("tsne_2d_proxy")
    ]
    summary_keys = [
        column
        for column in (*_IDENTITY_COLUMNS, "method_short", "split_variant")
        if column in protocol_tsne
    ]
    summaries: list[dict[str, object]] = []
    for key, group in protocol_tsne.groupby(summary_keys, dropna=False, sort=False):
        values = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(summary_keys, values))
        seed_summary = _summarize_tsne_seed_rows(group, rz_threshold=rz_threshold)
        median_ratio = seed_summary["tsne_ratio_median"]
        majority_high = bool(seed_summary["majority_high"])
        summaries.append(
            {
                "record_type": "tsne_seed_summary",
                "scope": "canonical_metric_robustness",
                **identity,
                "fold_id": "seed_summary",
                "axis": "Z",
                "representation": "tsne_2d_proxy",
                "representation_seed": "summary",
                "metric_name": "severity_ratio_across_tsne_seeds",
                "metric_value": median_ratio,
                "severity_ratio": median_ratio,
                "Rz_variant": median_ratio,
                "threshold": float(rz_threshold),
                **seed_summary,
                "is_high": majority_high,
                "projection_count": int(sliced_projections),
                "effective_dimension": 2,
                "preprocessing_scope": "task_global",
                "transductive_proxy": True,
                "random_state": int(random_state),
                "null_status": "summarized_from_seed_rows",
                "status": "ok" if pd.notna(median_ratio) else "missing_seed_rows",
            }
        )
    if summaries:
        frame = pd.concat(
            [frame, pd.DataFrame.from_records(summaries)], ignore_index=True, sort=False
        )
    return frame


def _format_number(value: object, digits: int = 2) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "NA" if pd.isna(number) else f"{float(number):.{digits}f}"


def _robustness_report_lines(
    metric_contract: pd.DataFrame,
    *,
    ry_threshold: float,
    rz_threshold: float,
) -> list[str]:
    observed = metric_contract[metric_contract["record_type"].eq("observed_protocol")]
    target = observed[
        observed["axis"].eq("Y") & observed["representation"].eq("target")
    ]
    lines = [
        "### Task-global representation robustness",
        "",
        "| Task / protocol | Ry_variant | raw Rz | PCA Rz | t-SNE high seeds | Conclusion |",
        "|---|---:|---:|---:|---:|---|",
    ]
    if target.empty:
        lines.append("| none | NA | NA | NA | NA | robustness inputs unavailable |")
        return lines

    for target_row in target.itertuples(index=False):
        task_key = str(getattr(target_row, "task_key"))
        task_id = str(getattr(target_row, "task_id", task_key))
        method = str(getattr(target_row, "method_short"))
        split_variant = getattr(target_row, "split_variant", "")
        variant_key = _method_slug(split_variant)
        matches = observed[
            observed["task_key"].astype(str).eq(task_key)
            & observed["method_short"].astype(str).eq(method)
            & observed["split_variant"].map(_method_slug).eq(variant_key)
            & observed["axis"].eq("Z")
        ]

        def ratio_for(representation: str) -> float:
            values = pd.to_numeric(
                matches.loc[
                    matches["representation"].eq(representation), "severity_ratio"
                ],
                errors="coerce",
            ).dropna()
            return float(values.iloc[0]) if not values.empty else np.nan

        raw_ratio = ratio_for("standardized_raw")
        pca_ratio = ratio_for("pca20")
        seed_summary = metric_contract[
            metric_contract["record_type"].eq("tsne_seed_summary")
            & metric_contract["task_key"].astype(str).eq(task_key)
            & metric_contract["method_short"].astype(str).eq(method)
            & metric_contract["split_variant"].map(_method_slug).eq(variant_key)
        ]
        if seed_summary.empty:
            high_seed_count = 0
            seed_count = 0
            majority_high = False
            seed_evidence_status = "insufficient_seeds"
        else:
            high_seed_count = int(seed_summary["high_seed_count"].iloc[0])
            seed_count = int(seed_summary["tsne_seed_count"].iloc[0])
            majority_high = bool(seed_summary["majority_high"].iloc[0])
            seed_evidence_status = str(seed_summary["seed_evidence_status"].iloc[0])
        ry_value = float(getattr(target_row, "severity_ratio"))
        robust = bool(
            np.isfinite(ry_value)
            and ry_value > ry_threshold
            and np.isfinite(raw_ratio)
            and raw_ratio > rz_threshold
            and np.isfinite(pca_ratio)
            and pca_ratio > rz_threshold
            and majority_high
        )
        if seed_evidence_status != "ok":
            conclusion = "insufficient t-SNE seed evidence"
        elif robust:
            conclusion = "robust coupled composition-property shift"
        else:
            conclusion = "projection/threshold sensitive"
        method_label = method
        if variant_key not in {
            "unknown",
            _method_slug(method),
            _normalise_method_identity(method),
        }:
            method_label = f"{method} [{split_variant}]"
        lines.append(
            f"| {task_id} / {method_label} | {_format_number(ry_value)} | "
            f"{_format_number(raw_ratio)} | {_format_number(pca_ratio)} | "
            f"{high_seed_count}/{seed_count} | {conclusion} |"
        )
    return lines


def _deletion_coverage(
    group: pd.DataFrame,
    *,
    prefix: str,
) -> dict[str, int]:
    """Count unique candidates with complete finite Y/Z deletion evidence."""

    if prefix == "sample":
        candidate_keys = group["ID"].map(_normalise_id)
    else:
        candidate_keys = group["material_group_key"].astype(str)
    working = group.assign(_candidate_key=candidate_keys)
    working = working[working["_candidate_key"].astype(str).ne("")]
    complete_count = 0
    crossing_count = 0
    ok_statuses = {"ok", "ok_degenerate_n1"}
    numeric_columns = [
        f"{prefix}_Ry_before",
        f"{prefix}_Rz_before",
        f"{prefix}_Ry_after",
        f"{prefix}_Rz_after",
    ]
    for _, candidate_rows in working.groupby(
        "_candidate_key", dropna=False, sort=False
    ):
        candidate_complete = False
        candidate_crosses = False
        for _, row in candidate_rows.iterrows():
            statuses = [
                str(row.get(f"{prefix}_sensitivity_status", "")),
                str(row.get(f"{prefix}_y_sensitivity_status", "")),
                str(row.get(f"{prefix}_z_sensitivity_status", "")),
            ]
            metrics = pd.to_numeric(
                pd.Series([row.get(column, np.nan) for column in numeric_columns]),
                errors="coerce",
            ).to_numpy(dtype=float)
            before = row.get(f"{prefix}_joint_high_before", pd.NA)
            after = row.get(f"{prefix}_joint_high_after", pd.NA)
            complete = (
                all(status in ok_statuses for status in statuses)
                and np.isfinite(metrics).all()
                and not _missing(before)
                and not _missing(after)
            )
            if not complete:
                continue
            candidate_complete = True
            candidate_crosses |= bool(before) and not bool(after)
        complete_count += int(candidate_complete)
        crossing_count += int(candidate_crosses)
    return {
        "total": int(working["_candidate_key"].nunique(dropna=True)),
        "complete": complete_count,
        "crossing": crossing_count,
    }


def _exact_proxy_report_lines(
    metric_contract: pd.DataFrame,
    drivers: pd.DataFrame,
) -> list[str]:
    exact = metric_contract[
        metric_contract["record_type"].eq("current_proxy_reproduction")
    ]
    lines = [
        "### Exact current proxy reproduction and deletion sensitivity",
        "",
    ]
    if exact.empty:
        lines.append("- Exact current proxy reproduction was not run.")
        return lines
    folds = exact[exact["current_proxy_level"].eq("fold")]
    finite_residuals = pd.to_numeric(
        folds.get("current_proxy_abs_residual"), errors="coerce"
    ).dropna()
    max_residual = (
        float(finite_residuals.max()) if not finite_residuals.empty else np.nan
    )
    max_residual_text = f"{max_residual:.6g}" if np.isfinite(max_residual) else "NA"
    status_counts = folds["status"].astype(str).value_counts().to_dict()
    task_embedding = folds.drop_duplicates("task_key")
    duplicate_count = int(
        pd.to_numeric(
            task_embedding.get("embedding_duplicate_id_count"), errors="coerce"
        )
        .fillna(0)
        .sum()
    )
    duplicate_row_count = int(
        pd.to_numeric(
            task_embedding.get("embedding_duplicate_row_count"), errors="coerce"
        )
        .fillna(0)
        .sum()
    )
    conflict_count = int(
        pd.to_numeric(
            task_embedding.get("embedding_duplicate_coordinate_conflict_id_count"),
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )
    missing_join_count = int(
        pd.to_numeric(
            task_embedding.get("embedding_master_join_missing_count"),
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )
    lines.append(
        "- Exact split-scaled 2D OT fold reproduction: "
        f"rows={len(folds)}, status={status_counts}, "
        f"max absolute residual={max_residual_text}; "
        f"duplicate embedding IDs={duplicate_count} "
        f"(extra identical rows={duplicate_row_count}), "
        f"coordinate-conflict IDs={conflict_count}, "
        f"missing master joins={missing_join_count}."
    )
    lines.append(
        "- The exact LP path is used only for observed/current-proxy values and "
        "independent deletions. The empirical permutation null remains fixed-projection "
        "SWD (`null_status=not_run_exact_ot_prohibitive`)."
    )
    if drivers.empty:
        lines.append(
            "- No joint-high driver rows were available for deletion sensitivity."
        )
        return lines

    overlap_groupers = [
        column for column in ("task_id", "method_short", "fold_id") if column in drivers
    ]
    for key, group in drivers.groupby(overlap_groupers, dropna=False, sort=False):
        values = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(overlap_groupers, values))
        y_ids = set(group.loc[group["space"].eq("Y-space"), "ID"].map(_normalise_id))
        z_ids = set(group.loc[group["space"].eq("Z-space"), "ID"].map(_normalise_id))
        overlap = len(y_ids.intersection(z_ids))
        rank_source = (
            group["rank"]
            if "rank" in group
            else pd.Series(np.nan, index=group.index, dtype=float)
        )
        ranks = pd.to_numeric(rank_source, errors="coerce").dropna()
        if ranks.empty:
            overlap_label = "top-ranked Y/Z ID overlap"
            constrained_label = "This top-ranked overlap"
        else:
            audited_top_n = int(ranks.max())
            overlap_label = f"top-{audited_top_n} Y/Z ID overlap"
            constrained_label = f"This top-{audited_top_n} overlap"
        lines.append(
            f"- {identity.get('task_id', '')} / {identity.get('method_short', '')} / "
            f"{identity.get('fold_id', '')}: {overlap_label}={overlap} "
            f"(Y={len(y_ids)}, Z={len(z_ids)})."
        )
        if (
            str(identity.get("task_id")) == "Steel-El"
            and _normalise_method_identity(identity.get("method_short")) == "loco"
            and _method_slug(identity.get("fold_id")) in {"fold_2", "fold_4"}
        ):
            test_counts = pd.to_numeric(
                group.get("test_row_count"), errors="coerce"
            ).dropna()
            if not test_counts.empty:
                lines.append(
                    f"  - {constrained_label} is constrained by the small full test set "
                    f"(n_test={int(test_counts.iloc[0])})."
                )

    protocol_groupers = [
        column for column in ("task_id", "method_short") if column in drivers
    ]
    for key, group in drivers.groupby(protocol_groupers, dropna=False, sort=False):
        values = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(protocol_groupers, values))

        sample_coverage = _deletion_coverage(group, prefix="sample")
        group_coverage = _deletion_coverage(group, prefix="group")
        complete_total = sample_coverage["complete"] + group_coverage["complete"]
        crossing_total = sample_coverage["crossing"] + group_coverage["crossing"]
        if complete_total == 0:
            interpretation = "inconclusive: no complete Y+Z deletion result"
        elif crossing_total:
            interpretation = "threshold-sensitive within audited independent deletions"
        else:
            interpretation = (
                "no audited independent top-driver deletion crossed; "
                "this does not establish a broad distribution shift"
            )
        lines.append(
            f"- {identity.get('task_id', '')} / {identity.get('method_short', '')}: "
            f"sample deletion coverage={sample_coverage['complete']}/{sample_coverage['total']}, "
            f"material-group deletion coverage={group_coverage['complete']}/{group_coverage['total']}, "
            f"sample crossings={sample_coverage['crossing']}, "
            f"material-group crossings={group_coverage['crossing']}; "
            f"{interpretation}."
        )
    return lines


def _physical_explanation_lines(
    canonical_protocols: pd.DataFrame,
    integrity: pd.DataFrame,
) -> list[str]:
    known = {
        ("Al-UTS", "extra"): "高强尾部集中于高 Zn-Mg-Cu-Zr 的 7xxx 铝合金。",
        ("Ti-UTS", "extra"): "高强尾部对应不同 β/近 β Ti 成分及不同时效路线。",
        ("Steel-El", "extra"): (
            "高延性尾部主要涉及 TWIP、奥氏体及双相不锈钢；若 "
            "train_max=test_min，只能称高端边界相接，不能称严格分离。"
        ),
        ("Steel-El", "loco"): (
            "留出成分簇可同时跨越不同钢种和延性机制，应逐 fold 追踪，"
            "特别关注 fold_2/fold_4 的小测试集。"
        ),
    }
    lines: list[str] = []
    for row in canonical_protocols.itertuples(index=False):
        task_id = str(getattr(row, "task_id", getattr(row, "task_key", "unknown")))
        method_short = str(getattr(row, "method_short", "unknown"))
        key = (task_id, _normalise_method_identity(method_short))
        label = f"{task_id} / {method_short}"
        expected_value = pd.to_numeric(
            pd.Series([getattr(row, "protocol_fold_count", 1)]), errors="coerce"
        ).iloc[0]
        expected_folds = max(int(expected_value), 1) if pd.notna(expected_value) else 1
        successful_folds: set[str] = set()
        required = {
            "task_id",
            "method_short",
            "fold_id",
            "driver_metadata_status",
            "driver_test_join_status",
        }
        if not integrity.empty and required.issubset(integrity.columns):
            method_keys = integrity["method_short"].map(_normalise_method_identity)
            matches = integrity[
                integrity["task_id"].astype(str).eq(task_id)
                & method_keys.eq(key[1])
                & integrity["driver_metadata_status"]
                .astype(str)
                .str.casefold()
                .eq("ok")
                & integrity["driver_test_join_status"]
                .astype(str)
                .str.casefold()
                .eq("unique")
            ]
            successful_folds = {
                _method_slug(value) for value in matches["fold_id"].dropna().tolist()
            }
        successful_count = len(successful_folds)
        if successful_count < expected_folds:
            lines.append(
                f"- {label}：样本 driver/metadata 证据仅覆盖 "
                f"{successful_count}/{expected_folds} 个预期物理 fold；证据不完整，"
                "不对具体材料机制作断言。"
            )
        elif key in known:
            lines.append(f"- {label}：初步线索：{known[key]}")
        else:
            lines.append(
                f"- {label}：已完成样本与 metadata 回查，但暂无任务特定的物理机制解释；"
                "需结合贡献样本的成分、工艺和参考文献另行判读。"
            )
    return lines or ["- 当前没有 canonical 联合高值协议。"]


def _build_report(
    *,
    published: pd.DataFrame,
    legacy_rows: pd.DataFrame,
    source_audit: pd.DataFrame,
    metric_contract: pd.DataFrame,
    integrity: pd.DataFrame,
    drivers: pd.DataFrame,
    ry_threshold: float,
    rz_threshold: float,
    legacy_exclude_task: str,
    canonical_source: str,
    robustness_run: bool,
) -> str:
    current_stats = compute_joint_overlap_stats(
        published, ry_threshold=ry_threshold, rz_threshold=rz_threshold
    )
    legacy_summary = _legacy_subset(published, legacy_exclude_task)
    legacy_stats = compute_joint_overlap_stats(
        legacy_summary, ry_threshold=ry_threshold, rz_threshold=rz_threshold
    )
    nonbaseline = published[
        ~published["method_short"].astype(str).str.casefold().eq("randcv")
    ].copy()
    joint = nonbaseline[
        pd.to_numeric(nonbaseline["Ry"], errors="coerce").gt(ry_threshold)
        & pd.to_numeric(nonbaseline["Rz"], errors="coerce").gt(rz_threshold)
    ]
    max_residual = pd.to_numeric(
        legacy_rows.get("published_reproduction_max_abs_residual"), errors="coerce"
    ).max()
    canonical_protocols = _joint_protocol_rows(source_audit)
    inventory_only_mask = (
        source_audit.get("y_split_w_status", pd.Series("", index=source_audit.index))
        .astype(str)
        .eq("inventory_only_missing_yz")
    )
    inventory_only = source_audit.loc[inventory_only_mask].drop_duplicates(
        [
            column
            for column in (
                "task_key",
                "source_split_dir",
                "fold_id",
                "method_short",
                "split_variant",
            )
            if column in source_audit
        ]
    )

    lines = [
        "# 联合高 Ry/Rz 数据审计",
        "",
        "## 口径与结论",
        "",
        (
            "图中绘制的是归一化严重度 `Ry=Wy/Wy_random` 与 "
            "`Rz=Wz/Wz_random`，不是原始 `Wy/Wz`。两套判据必须分开报告。"
        ),
        "",
        (
            f"当前 9 任务口径有 {current_stats['nonbaseline_count']} 个非 RandCV 条件："
            f"`Ry>{ry_threshold:g}` 有 {current_stats['high_ry_count']} 个，"
            f"`Rz>{rz_threshold:g}` 有 {current_stats['high_rz_count']} 个，"
            f"联合高值 {current_stats['joint_high_count']} 个。若两轴独立，期望重叠为 "
            f"{_format_number(current_stats['expected_joint_under_independence'])} 个。"
        ),
        (
            f"原始距离规则 `Wy>1 且 Wz>1` 得到 "
            f"{current_stats['raw_wy_wz_gt_one_count']} 个点；它不是上述归一化联合高值规则。"
        ),
        (
            f"论文旧口径另行排除 `{legacy_exclude_task}`，有 "
            f"{legacy_stats['nonbaseline_count']} 个非基准条件、"
            f"{legacy_stats['joint_high_count']} 个联合高值；禁止与当前口径混合统计。"
        ),
        "",
        "当前阈值下的已发表坐标：",
        "",
    ]
    if joint.empty:
        lines.append("- 无。")
    else:
        for row in joint.itertuples(index=False):
            lines.append(
                f"- {getattr(row, 'task_id')} / {getattr(row, 'method_short')}: "
                f"Ry={_format_number(getattr(row, 'Ry'))}, "
                f"Rz={_format_number(getattr(row, 'Rz'))}"
            )

    lines.extend(
        [
            "",
            "canonical source-specific 坐标（与上述 mixed-source 已发表坐标分开）：",
            "",
        ]
    )
    if canonical_protocols.empty:
        lines.append("- 无。")
    else:
        for row in canonical_protocols.itertuples(index=False):
            lines.append(
                f"- {getattr(row, 'task_id')} / {getattr(row, 'method_short')}: "
                f"Ry={_format_number(getattr(row, 'protocol_Ry'))}, "
                f"Rz={_format_number(getattr(row, 'protocol_Rz'))}, "
                f"source={getattr(row, 'source_name', canonical_source)}"
            )

    lines.extend(
        [
            "",
            "## 去重与复现方法",
            "",
            (
                "大体量样本长表采用两遍流式读取。第一遍按 "
                "`task + protocol + source_split_dir + fold + space` 形成一个 split-space "
                "记录，分别保存 split_w 的 first/min/max、模型数和样本数，从而去除模型与"
                "测试样本行数造成的隐式加权。"
            ),
            (
                f"canonical 结果使用 `{canonical_source}`，source 由物理路径判定；inventory "
                "先按原始路径精确匹配，再使用仅斜杠统一、去尾斜杠和 casefold 的后备键，"
                "不调用 Path.resolve，也不按 task_key 关联。"
            ),
            (
                "`legacy_reproduction` 保留原绘图流程的混合 source 行级平均，再对 fold 等权平均。"
                f"与已发表表的最大绝对公式残差为 {_format_number(max_residual, 3)}。"
            ),
            (
                f"canonical source 中按协议去重后识别到 {len(canonical_protocols)} 个联合高值条件。"
            ),
        ]
    )
    lines.append(
        f"inventory-only missing Y/Z splits: {len(inventory_only)} 个；"
        "这些行仅保留来源与 split 元数据，不生成伪造的 Wy/Wz/Ry/Rz。"
    )
    for row in inventory_only.itertuples(index=False):
        lines.append(
            f"- {getattr(row, 'task_id')} / {getattr(row, 'method_short')} / "
            f"{getattr(row, 'fold_id')}: {getattr(row, 'source_split_dir')}"
        )
    lines.extend(["", "## 样本完整性", ""])
    if integrity.empty:
        lines.append("- 没有可回查的 canonical 联合高值 split。")
    else:
        for row in integrity.itertuples(index=False):
            task_id = getattr(row, "task_id", "")
            method = getattr(row, "method_short", "")
            fold = getattr(row, "fold_id", "")
            status = getattr(row, "driver_metadata_status", "")
            overlap = getattr(row, "train_test_id_overlap_count", np.nan)
            train_dup = getattr(row, "train_duplicate_id_count", np.nan)
            test_dup = getattr(row, "test_duplicate_id_count", np.nan)
            train_exact_dup = getattr(row, "train_exact_duplicate_count", np.nan)
            test_exact_dup = getattr(row, "test_exact_duplicate_count", np.nan)
            exact = getattr(row, "train_test_exact_record_overlap_count", np.nan)
            train_max = getattr(row, "train_target_max", np.nan)
            test_min = getattr(row, "test_target_min", np.nan)
            target_note = ""
            if str(method).casefold() == "extra.":
                if (
                    pd.notna(train_max)
                    and pd.notna(test_min)
                    and float(train_max) < float(test_min)
                ):
                    target_note = "；Extra 为严格训练高端之外"
                elif (
                    pd.notna(train_max)
                    and pd.notna(test_min)
                    and np.isclose(float(train_max), float(test_min))
                ):
                    target_note = (
                        "；Extra 在 train_max=test_min 处边界相接，并非严格分离"
                    )
                else:
                    target_note = "；Extra 不是严格训练高端之外"
            lines.append(
                f"- {task_id} / {method} / {fold}: status={status}，"
                f"train-test ID 重叠={_format_number(overlap, 0)}，"
                f"train/test 重复 ID={_format_number(train_dup, 0)}/{_format_number(test_dup, 0)}，"
                "train/test 特征+目标重复记录="
                f"{_format_number(train_exact_dup, 0)}/{_format_number(test_exact_dup, 0)}，"
                f"跨 split 完全相同特征+目标记录={_format_number(exact, 0)}{target_note}。"
            )
        steel_loco = integrity[
            integrity["task_id"].astype(str).eq("Steel-El")
            & integrity["method_short"].astype(str).str.casefold().eq("loco")
            & integrity["fold_id"].astype(str).isin(["fold_2", "fold_4"])
        ]
        if not steel_loco.empty:
            details = ", ".join(
                f"{row.fold_id}: n_test={int(row.test_row_count)}"
                for row in steel_loco.itertuples(index=False)
                if pd.notna(getattr(row, "test_row_count", np.nan))
            )
            lines.append(f"- Steel-El / LOCO 的小测试折需单独解释：{details}。")

    lines.extend(
        [
            "",
            "## 指标与阈值敏感性",
            "",
            (
                "当前 `Wz` 是 composition/process 特征二维 t-SNE 坐标上的 OT 距离，"
                "只应视作诊断性代理，而不是高维潜在空间距离。"
            ),
        ]
    )
    if robustness_run:
        lines.extend(
            [
                "- Robustness metrics use fixed task-global diagnostic geometries. The t-SNE "
                "2D proxy is explicitly transductive; raw/PCA are task-global diagnostics so "
                "permutation labels remain exchangeable without refitting geometry.",
                "- Representation empirical nulls use cached fixed-projection sliced "
                "Wasserstein distances. Exact legacy 2D OT/LP nulls were not run, and no SWD "
                "null is relabeled as an exact OT result.",
                "",
            ]
        )
        lines.extend(_exact_proxy_report_lines(metric_contract, drivers))
        lines.append("")
        lines.extend(
            _robustness_report_lines(
                metric_contract,
                ry_threshold=ry_threshold,
                rz_threshold=rz_threshold,
            )
        )
    else:
        lines.append(
            "- 本次使用 `--no-robustness`，高维原始特征、PCA20、t-SNE 多随机种子和 1,000 次随机划分验证待后续运行。"
        )
    current_grid = metric_contract[
        metric_contract["record_type"].eq("threshold_grid")
        & metric_contract["scope"].eq("current_9_task")
    ]
    lines.append("- 阈值网格（严格大于）显示联合点数：")
    for row in current_grid.itertuples(index=False):
        if (row.Ry_threshold, row.Rz_threshold) in {
            (8.0, 5.0),
            (8.0, 5.5),
            (9.0, 5.0),
            (10.0, 6.0),
        }:
            lines.append(
                f"  - Ry>{row.Ry_threshold:g}, Rz>{row.Rz_threshold:g}: "
                f"{int(row.grid_joint_high_count)}"
            )

    lines.extend(["", "## canonical 联合高值协议的初步物理解释", ""])
    lines.extend(_physical_explanation_lines(canonical_protocols, integrity))
    lines.extend(
        [
            "",
            "这些解释只针对本次实际识别且已有样本证据的协议；只有在标准化原始特征、PCA 和多数 t-SNE 种子下仍保持联合高值时，才可进一步解释为 composition–property coupled shift。",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit and write its four stable artifacts."""

    args = build_parser().parse_args(argv)
    if args.top_n < 0:
        raise ValueError("--top-n must be non-negative")
    if args.max_exact_deletion_sets_per_split <= 0:
        raise ValueError("--max-exact-deletion-sets-per-split must be positive")
    samples_long = Path(args.samples_long)
    severity_summary = Path(args.severity_summary)
    source_inventory = Path(args.source_inventory)
    output_root = Path(args.output_root)

    published = _read_csv_flexible(severity_summary)
    task_keys = set(published["task_key"].dropna().astype(str))
    legacy_folds, collapsed = _stream_first_pass(
        samples_long,
        task_keys=task_keys,
        canonical_source=args.canonical_source,
    )
    legacy_rows = _prepare_legacy_rows(
        legacy_folds,
        published,
        ry_threshold=args.ry_threshold,
        rz_threshold=args.rz_threshold,
    )

    inventory = _read_csv_flexible(source_inventory)
    inventory_exact, inventory_normalised = _inventory_maps(inventory)
    source_audit = build_source_severity_audit(
        collapsed,
        ry_threshold=args.ry_threshold,
        rz_threshold=args.rz_threshold,
    )
    source_audit = _enrich_source_audit(
        source_audit,
        collapsed,
        inventory_exact,
        inventory_normalised,
    )
    source_audit = _append_inventory_only_source_rows(
        source_audit,
        inventory=inventory,
        published=published,
        inventory_exact=inventory_exact,
        inventory_normalised=inventory_normalised,
        canonical_source=args.canonical_source,
    )
    source_audit["audit_view"] = np.where(
        source_audit["is_canonical"].astype(bool),
        "canonical_deduplicated",
        "source_deduplicated",
    )

    protocols = _joint_protocol_rows(source_audit)
    physical_splits = _joint_physical_splits(source_audit, protocols)
    driver_rows = _stream_joint_driver_rows(samples_long, physical_splits)
    if driver_rows.empty:
        ranked = pd.DataFrame(columns=_DRIVER_COLUMNS + ["rank", "model_count"])
    else:
        ranked = rank_sample_drivers(driver_rows, top_n=args.top_n)
    drivers, integrity = _attach_driver_metadata(
        ranked,
        inventory_exact=inventory_exact,
        inventory_normalised=inventory_normalised,
        inventory_parent=source_inventory.parent,
    )

    metric_contract = _metric_contract(
        published,
        ry_threshold=args.ry_threshold,
        rz_threshold=args.rz_threshold,
        legacy_exclude_task=args.legacy_exclude_task,
        robustness_run=not args.no_robustness,
    )
    if not args.no_robustness:
        relevant_splits = _relevant_robustness_splits(source_audit, protocols)
        bundles = _build_task_robustness_bundles(
            relevant_splits,
            inventory_parent=source_inventory.parent,
        )
        robustness_rows = _run_metric_robustness(
            bundles,
            protocols,
            ry_threshold=args.ry_threshold,
            rz_threshold=args.rz_threshold,
            random_splits=args.random_splits,
            random_state=args.random_state,
            pca_components=args.pca_components,
            tsne_seeds=args.tsne_seeds,
            sliced_projections=args.sliced_projections,
        )
        metric_contract = pd.concat(
            [metric_contract, robustness_rows],
            ignore_index=True,
            sort=False,
        )
        exact_proxy_rows, exact_contexts = _run_exact_current_proxy_audit(
            bundles,
            protocols,
            embedding_data_dir=Path(args.embedding_data_dir),
            ry_threshold=args.ry_threshold,
            rz_threshold=args.rz_threshold,
            random_state=args.random_state,
        )
        metric_contract = pd.DataFrame.from_records(
            [
                *metric_contract.to_dict("records"),
                *exact_proxy_rows.to_dict("records"),
            ]
        )
        drivers = _attach_exact_deletion_sensitivity(
            drivers,
            exact_contexts,
            ry_threshold=args.ry_threshold,
            rz_threshold=args.rz_threshold,
            max_exact_deletion_sets_per_split=(args.max_exact_deletion_sets_per_split),
        )
    report = _build_report(
        published=published,
        legacy_rows=legacy_rows,
        source_audit=source_audit,
        metric_contract=metric_contract,
        integrity=integrity,
        drivers=drivers,
        ry_threshold=args.ry_threshold,
        rz_threshold=args.rz_threshold,
        legacy_exclude_task=args.legacy_exclude_task,
        canonical_source=args.canonical_source,
        robustness_run=not args.no_robustness,
    )

    severity_output = pd.concat(
        [legacy_rows, source_audit], ignore_index=True, sort=False
    )
    paths = write_audit_artifacts(
        output_root,
        severity_source_audit=severity_output,
        sample_drivers=drivers,
        metric_robustness=metric_contract,
        report_text=report,
    )
    max_residual = pd.to_numeric(
        legacy_rows.get("published_reproduction_max_abs_residual"), errors="coerce"
    ).max()
    print(
        "audit complete: "
        f"collapsed_split_spaces={len(collapsed)}, "
        f"canonical_joint_protocols={len(protocols)}, "
        f"max_published_residual={max_residual:.3g}"
    )
    for path in paths.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
