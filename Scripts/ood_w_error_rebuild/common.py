from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
WX_MODE = "wx_sample_score"
SPLIT_SOURCE = "result_local"
FORBIDDEN_TOKENS = ("wx_v2", "sliced_wasserstein")
HYBRID_SUBSETS = ("combined", "test_extrapolation_high20", "test_inner_ood")

TARGET_COLUMNS = {
    "YS(MPa)",
    "UTS(MPa)",
    "El(%)",
    "yield_strength",
    "yield strength",
    "ultimate_tensile_strength",
}
NON_FEATURE_COLUMNS = {
    "ID",
    "Number",
    "Processing_Description",
    "__row_id__",
    "__test_index__",
    "__source_index__",
    "sample_index",
    "set",
}
JOIN_ID_COLUMNS = ("ID", "__row_id__", "__test_index__", "sample_index", "composition")


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def clean_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def normalize_id(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    text = str(value).strip()
    if text.endswith(".0"):
        prefix = text[:-2]
        if prefix.lstrip("-").isdigit():
            return prefix
    return text


def join_id_column(frame: pd.DataFrame) -> str | None:
    for column in JOIN_ID_COLUMNS:
        if column in frame.columns:
            return column
    return None


def frame_join_ids(frame: pd.DataFrame) -> list[str]:
    column = join_id_column(frame)
    if column is None:
        return [""] * len(frame)
    return [normalize_id(value) for value in frame[column].tolist()]


def slugify(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", clean_text(value))
    return text.strip("_") or "unknown"


def property_short(property_name: object) -> str:
    text = clean_text(property_name)
    if "UTS" in text.upper() or "ultimate" in text.lower():
        return "UTS_MPa"
    if "YS" in text.upper() or "yield" in text.lower():
        return "YS_MPa"
    if text.lower().startswith("el") or "elong" in text.lower():
        return "El"
    return slugify(text)


def task_key(family: object, dataset: object, property_name: object) -> str:
    return f"{clean_text(family)}__{clean_text(dataset)}__{property_short(property_name)}"


def task_id(family: object, property_name: object) -> str:
    return f"{clean_text(family)}_{property_short(property_name)}"


def normalize_model(value: object) -> str:
    text = clean_text(value)
    lower = text.lower()
    mapping = {
        "steelbert": "SteelBERT",
        "scibert": "SciBERT",
        "matscibert": "MatSciBERT",
        "llmprop": "LLMProp",
        "sklearn_rf_results": "RF",
        "rf_results": "RF",
        "random_forest_results": "RF",
        "xgboost_results": "XGBoost",
        "lightgbm_results": "LightGBM",
        "catboost_results": "CatBoost",
        "mlp_results": "MLP",
    }
    if lower in mapping:
        return mapping[lower]
    if lower.endswith("_results"):
        return text[: -len("_results")]
    if "tabpfn-2.5-plus-text" in lower:
        return "TabPFN-2.5-Plus-Text"
    if "tabpfn-2.5-plus-numeric" in lower:
        return "TabPFN-2.5-Plus-Numeric"
    if "tabpfn-v2-numeric" in lower:
        return "TabPFN-V2-Numeric"
    return text


def normalize_method(method_dir: object, *, scope: str) -> str:
    text = clean_text(method_dir).lower()
    for prefix in ("hybridhigh20_", "hybrid_high20_", "hybrid_extrapolation_"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    mapping = {
        "loco": "LOCO",
        "loco_k5": "LOCO",
        "random_cv": "RandCV" if scope == "hybrid" else "RandomCV",
        "random_cv_baseline": "RandomCV",
        "target_extrapolation": "Extrapolation",
        "sparse_x_single": "SparseXsingle",
        "sparse_x_single_k5": "SparseXsingle",
        "sparse_x_cluster": "SparseXcluster",
        "sparse_x_cluster_k5": "SparseXcluster",
        "sparse_y_single": "SparseYsingle",
        "sparse_y_single_k5": "SparseYsingle",
        "sparse_y_cluster": "SparseYcluster",
        "sparse_y_cluster_k5": "SparseYcluster",
    }
    return mapping.get(text, clean_text(method_dir))


def source_uses_global_ood_splits(path: Path | str) -> bool:
    normalized = str(path).replace("/", "\\").lower()
    return "\\output\\ood_splits\\" in normalized


def infer_x_feature_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target_col: str = "",
    configured: list[str] | None = None,
) -> list[str]:
    if configured:
        columns = [str(column) for column in configured if str(column) in train_df.columns and str(column) in test_df.columns]
        if columns:
            return columns

    excluded = set(NON_FEATURE_COLUMNS) | set(TARGET_COLUMNS)
    if target_col:
        excluded.add(target_col)
    feature_columns: list[str] = []
    for column in train_df.columns:
        if column in excluded or column not in test_df.columns:
            continue
        train_numeric = pd.to_numeric(train_df[column], errors="coerce")
        test_numeric = pd.to_numeric(test_df[column], errors="coerce")
        if train_numeric.notna().any() or test_numeric.notna().any():
            feature_columns.append(str(column))
    if not feature_columns:
        raise ValueError("No X-space feature columns inferred from result-local train/test files.")
    return feature_columns
