from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_processing.strength_ood_common import (  # noqa: E402
    HYBRID_INNER_TEST_SET_NAME,
    HYBRID_OUTER_TEST_SET_NAME,
    select_numeric_x_space_columns,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


DEFAULT_SPLITS_ROOT = Path("output") / "ood_splits"
DEFAULT_OUTPUT_PARENT = Path("output") / "top_tail_loco_proximity"
DEFAULT_OUTPUT_ROOT = Path("output") / "top_tail_loco_proximity" / "xspace"
DEFAULT_CONFIG_PATH = Path("configs") / "analysis" / "top_tail_loco_proximity_outliers.yaml"
DEFAULT_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
CASE_COLUMNS = ["alloy_family", "dataset_name", "property"]


@dataclass(frozen=True)
class HybridClusterPayload:
    manifest_path: Path
    split_root: Path
    alloy_family: str
    dataset_name: str
    property_name: str
    target_col: str
    threshold_pct: int
    source_mode: str
    top_tail: pd.DataFrame
    background: pd.DataFrame
    cluster_frames: dict[int, pd.DataFrame]
    feature_columns: list[str]
    data_file: str
    split_id: str


@dataclass(frozen=True)
class TaskSpec:
    alloy_family: str
    dataset_name: str
    property_name: str
    target_col: str
    data_file: Path
    feature_columns: list[str]
    cluster_count: int
    manifest_path: Path | None = None
    split_id: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze X-space proximity between high-property tails and LOCO clusters."
    )
    parser.add_argument("--splits-root", default=str(DEFAULT_SPLITS_ROOT))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--outlier-policy", default="none")
    parser.add_argument("--thresholds", nargs="+", type=float, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sliced-projections", type=int, default=None)
    parser.add_argument("--case-contains", default=None)
    parser.add_argument("--include-matbench", action="store_true", help="Deprecated no-op; Matbench Steel is included by default.")
    parser.add_argument("--exclude-matbench", action="store_true", help="Exclude Matbench Steel and run only the 9 strength tasks.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_analysis_config(path: Path | str | None) -> dict[str, Any]:
    if not path:
        return {"defaults": {}, "outlier_sets": {}, "policies": {"none": {"use_sets": [], "output_suffix": "xspace"}}}
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"analysis config not found: {config_path}")
    if config_path.suffix.lower() == ".json":
        return read_json(config_path)
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError("PyYAML is required to read YAML configs; install pyyaml or pass a JSON config.") from exc
    content = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return content if isinstance(content, dict) else {}


def resolve_outlier_policy(config: dict[str, Any], policy_name: str) -> dict[str, Any]:
    policies = dict(config.get("policies") or {})
    outlier_sets = dict(config.get("outlier_sets") or {})
    if policy_name not in policies:
        available = ", ".join(sorted(policies)) or "<none>"
        raise ValueError(f"unknown outlier policy: {policy_name}. Available policies: {available}")
    policy = dict(policies[policy_name] or {})
    resolved_sets: list[dict[str, Any]] = []
    for set_name in list(policy.get("use_sets") or []):
        if set_name not in outlier_sets:
            raise ValueError(f"outlier policy {policy_name} references unknown set: {set_name}")
        current = dict(outlier_sets[set_name] or {})
        current["set_name"] = str(set_name)
        current["ids"] = [int(value) for value in list(current.get("ids") or [])]
        resolved_sets.append(current)
    policy["name"] = policy_name
    policy["sets"] = resolved_sets
    policy["has_drop"] = any(str(item.get("action", "")).lower() == "drop" for item in resolved_sets)
    policy["has_flag"] = any(str(item.get("action", "")).lower() == "flag" for item in resolved_sets)
    return policy


def apply_outlier_policy(
    frame: pd.DataFrame,
    alloy_family: str,
    property_name: str,
    policy: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    result = frame.copy()
    manifest_rows: list[dict[str, Any]] = []
    if "ID" not in result.columns:
        return result, manifest_rows
    ids = pd.to_numeric(result["ID"], errors="coerce")
    for item in list(policy.get("sets") or []):
        if str(item.get("alloy_family", "")) != str(alloy_family):
            continue
        item_property = item.get("property")
        if item_property is not None and str(item_property) != str(property_name):
            continue
        action = str(item.get("action", "")).lower()
        target_ids = set(int(value) for value in list(item.get("ids") or []))
        matched = ids.isin(target_ids).fillna(False)
        matched_ids = sorted(int(value) for value in ids[matched].dropna().unique().tolist())
        if action == "drop":
            result = result.loc[~matched].copy()
            ids = pd.to_numeric(result["ID"], errors="coerce")
        elif action == "flag":
            flag_column = str(item.get("flag_column") or f"is_{item.get('set_name', 'outlier')}")
            if flag_column not in result.columns:
                result[flag_column] = False
            result.loc[matched, flag_column] = True
        else:
            raise ValueError(f"unsupported outlier action for {item.get('set_name')}: {action}")
        for sample_id in matched_ids:
            manifest_rows.append(
                {
                    "outlier_policy": str(policy.get("name", "")),
                    "set_name": str(item.get("set_name", "")),
                    "action": action,
                    "alloy_family": alloy_family,
                    "property": property_name,
                    "ID": int(sample_id),
                    "flag_column": str(item.get("flag_column", "")),
                    "reason": str(item.get("reason", "")),
                }
            )
    return result.reset_index(drop=True), manifest_rows


def flag_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("is_")]


def slugify(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", text, flags=re.UNICODE)
    return text.strip("._") or "unknown"


def case_key(alloy_family: object, dataset_name: object, property_name: object) -> str:
    return "__".join(slugify(value) for value in (alloy_family, dataset_name, property_name))


def threshold_to_pct(threshold: float) -> int:
    return int(round(float(threshold) * 100))


def threshold_count(total_rows: int, threshold: float) -> int:
    requested = int(round(int(total_rows) * float(threshold)))
    return max(1, min(requested, int(total_rows) - 1))


def finite_float(value: Any) -> float:
    try:
        result = float(value)
    except Exception:
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def select_top_tail(frame: pd.DataFrame, target_col: str, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < float(threshold) < 1:
        raise ValueError("threshold must be between 0 and 1")
    if target_col not in frame.columns:
        raise ValueError(f"target column not found: {target_col}")
    work = frame.copy()
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna(subset=[target_col]).reset_index(drop=True)
    if len(work) < 2:
        raise ValueError("at least two rows with finite target values are required")
    top_count = threshold_count(len(work), threshold)
    ranked = work.sort_values(target_col, ascending=False, kind="mergesort")
    top_index = ranked.head(top_count).index
    top_tail = work.loc[top_index].copy()
    background = work.drop(index=top_index).copy()
    return top_tail.reset_index(drop=True), background.reset_index(drop=True)


def _numeric_feature_frame(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({column: pd.to_numeric(frame.get(column), errors="coerce") for column in feature_columns})


def _scaled_frames(
    top_tail: pd.DataFrame,
    cluster_frames: dict[int, pd.DataFrame],
    feature_columns: list[str],
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    background = pd.concat(list(cluster_frames.values()), ignore_index=True)
    if background.empty or top_tail.empty:
        raise ValueError("top-tail and background cluster frames must be non-empty")
    if not feature_columns:
        raise ValueError("feature_columns must not be empty")
    imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    scaler = StandardScaler()
    background_x = _numeric_feature_frame(background, feature_columns)
    imputer.fit(background_x)
    scaler.fit(imputer.transform(background_x))
    top_scaled = scaler.transform(imputer.transform(_numeric_feature_frame(top_tail, feature_columns)))
    cluster_scaled = {
        int(cluster_id): scaler.transform(imputer.transform(_numeric_feature_frame(cluster_df, feature_columns)))
        for cluster_id, cluster_df in cluster_frames.items()
    }
    return top_scaled, cluster_scaled


def _background_scaled_x(
    top_tail: pd.DataFrame,
    background: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    if background.empty or top_tail.empty:
        raise ValueError("top-tail and background frames must be non-empty")
    if not feature_columns:
        raise ValueError("feature_columns must not be empty")
    imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    scaler = StandardScaler()
    background_x = _numeric_feature_frame(background, feature_columns)
    imputer.fit(background_x)
    scaler.fit(imputer.transform(background_x))
    background_scaled = scaler.transform(imputer.transform(background_x))
    top_scaled = scaler.transform(imputer.transform(_numeric_feature_frame(top_tail, feature_columns)))
    return top_scaled, background_scaled


def _standardized_target_values(
    top_tail: pd.DataFrame,
    background: pd.DataFrame,
    target_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    background_y = pd.to_numeric(background[target_col], errors="coerce").dropna().to_numpy(dtype=float)
    top_y = pd.to_numeric(top_tail[target_col], errors="coerce").dropna().to_numpy(dtype=float)
    if len(background_y) == 0 or len(top_y) == 0:
        raise ValueError("top-tail and background target values must be finite")
    mean = float(np.mean(background_y))
    std = float(np.std(background_y, ddof=0))
    if not math.isfinite(std) or std <= 1e-12:
        std = 1.0
    return (top_y - mean) / std, (background_y - mean) / std


def _sliced_wasserstein_distance(
    background_x: np.ndarray,
    top_x: np.ndarray,
    projections: int,
    random_state: int,
) -> float:
    if background_x.ndim != 2 or top_x.ndim != 2:
        raise ValueError("X arrays must be two-dimensional")
    if background_x.shape[1] != top_x.shape[1]:
        raise ValueError("background and top-tail X arrays must have the same feature count")
    feature_count = int(background_x.shape[1])
    if feature_count == 0:
        raise ValueError("X arrays must contain at least one feature")
    if feature_count == 1:
        return finite_float(wasserstein_distance(background_x[:, 0], top_x[:, 0]))

    rng = np.random.default_rng(random_state)
    projection_count = max(1, int(projections))
    directions = rng.normal(size=(projection_count, feature_count))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.where(norms <= 1e-12, 1.0, norms)
    distances = [
        wasserstein_distance(background_x @ direction, top_x @ direction)
        for direction in directions
    ]
    return finite_float(np.mean(distances))


def compute_space_shift_metrics(
    top_tail: pd.DataFrame,
    background: pd.DataFrame,
    feature_columns: list[str],
    target_col: str,
    metadata: dict[str, Any],
    sliced_projections: int = 128,
    random_state: int = 42,
) -> dict[str, Any]:
    top_y, background_y = _standardized_target_values(top_tail, background, target_col)
    top_x, background_x = _background_scaled_x(top_tail, background, feature_columns)
    return {
        **metadata,
        "top_tail_size": int(len(top_tail)),
        "background_size": int(len(background)),
        "feature_count": int(len(feature_columns)),
        "Wy": finite_float(wasserstein_distance(background_y, top_y)),
        "Wx": _sliced_wasserstein_distance(background_x, top_x, sliced_projections, random_state),
        "x_distance_method": "sliced_wasserstein_standardized_x",
        "x_sliced_projections": int(max(1, sliced_projections)),
        "y_distance_method": "wasserstein_1_standardized_y",
    }


def _id_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in ["ID", "__row_id__", "__source_index__"] if column in frame.columns]


def _sample_passthrough_columns(frame: pd.DataFrame) -> list[str]:
    return [*_id_columns(frame), *flag_columns(frame)]


def compute_cluster_proximity(
    top_tail: pd.DataFrame,
    cluster_frames: dict[int, pd.DataFrame],
    feature_columns: list[str],
    metadata: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    top_scaled, cluster_scaled = _scaled_frames(top_tail, cluster_frames, feature_columns)
    cluster_ids = sorted(cluster_scaled)
    centroids = np.vstack([cluster_scaled[cluster_id].mean(axis=0) for cluster_id in cluster_ids])
    distances = cdist(top_scaled, centroids, metric="euclidean")

    sample_rows: list[dict[str, Any]] = []
    passthrough_cols = _sample_passthrough_columns(top_tail)
    for sample_index, (_, source_row) in enumerate(top_tail.reset_index(drop=True).iterrows()):
        order = np.argsort(distances[sample_index], kind="stable")
        for rank_index, centroid_index in enumerate(order, start=1):
            cluster_id = int(cluster_ids[int(centroid_index)])
            row = {
                **metadata,
                "cluster_id": cluster_id,
                "distance_rank": int(rank_index),
                "distance_to_centroid": finite_float(distances[sample_index, centroid_index]),
            }
            for column in passthrough_cols:
                row[column] = source_row.get(column, np.nan)
            row["target_value"] = source_row.get(metadata.get("target_col", metadata.get("property", "")), np.nan)
            sample_rows.append(row)

    sample_level = pd.DataFrame(sample_rows)
    summary_rows: list[dict[str, Any]] = []
    for cluster_index, cluster_id in enumerate(cluster_ids):
        cluster_distances = distances[:, cluster_index]
        cluster_frame = cluster_frames[cluster_id]
        summary_rows.append(
            {
                **metadata,
                "cluster_id": int(cluster_id),
                "cluster_size": int(len(cluster_frame)),
                "top_tail_size": int(len(top_tail)),
                "feature_count": int(len(feature_columns)),
                "mean_distance_to_top_tail": finite_float(np.mean(cluster_distances)),
                "median_distance_to_top_tail": finite_float(np.median(cluster_distances)),
                "min_distance_to_top_tail": finite_float(np.min(cluster_distances)),
                "max_distance_to_top_tail": finite_float(np.max(cluster_distances)),
            }
        )
    cluster_summary = pd.DataFrame(summary_rows).sort_values(
        ["mean_distance_to_top_tail", "cluster_id"],
        kind="stable",
    )
    cluster_summary["distance_rank"] = np.arange(1, len(cluster_summary) + 1, dtype=int)
    score = concentration_score(cluster_summary)
    nearest = cluster_summary.iloc[0]
    second_distance = (
        finite_float(cluster_summary.iloc[1]["mean_distance_to_top_tail"])
        if len(cluster_summary) > 1
        else float("nan")
    )
    cluster_summary["nearest_cluster_id"] = int(nearest["cluster_id"])
    cluster_summary["nearest_cluster_mean_distance"] = finite_float(nearest["mean_distance_to_top_tail"])
    cluster_summary["second_nearest_cluster_mean_distance"] = second_distance
    cluster_summary["concentration_score"] = score
    return cluster_summary.reset_index(drop=True), sample_level.reset_index(drop=True)


def concentration_score(cluster_summary: pd.DataFrame) -> float:
    values = (
        pd.to_numeric(cluster_summary["mean_distance_to_top_tail"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values(kind="stable")
        .to_numpy(dtype=float)
    )
    if len(values) < 2 or values[1] <= 1e-12:
        return float("nan")
    return float(1.0 - (values[0] / values[1]))


def _resolve_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidate = (REPO_ROOT / path).resolve()
    if candidate.exists():
        return candidate
    return (base_dir / path).resolve()


def _case_from_manifest_path(manifest_path: Path) -> tuple[str, str, str, str]:
    parts = manifest_path.resolve().parts
    try:
        idx = parts.index("ood_splits")
        return str(parts[idx + 1]), str(parts[idx + 2]), str(parts[idx + 3]), str(parts[idx + 5])
    except Exception:
        parent = manifest_path.parent
        return "", "", "", parent.name


def build_cluster_frames_from_hybrid_manifest(manifest_path: Path) -> HybridClusterPayload:
    manifest = read_json(manifest_path)
    split_root = manifest_path.parent
    alloy_family, dataset_name, property_name, split_id = _case_from_manifest_path(manifest_path)
    alloy_family = str(manifest.get("alloy_type") or alloy_family)
    target_col = str(manifest.get("target_column") or property_name)
    folds = list(manifest.get("folds") or [])
    if not folds:
        raise ValueError(f"hybrid manifest has no folds: {manifest_path}")

    top_tail: pd.DataFrame | None = None
    background_parts: list[pd.DataFrame] = []
    cluster_frames: dict[int, pd.DataFrame] = {}
    feature_columns: list[str] = []
    for fold in folds:
        test_sets = dict(fold.get("test_set_files") or {})
        top_path_text = str(test_sets.get(HYBRID_OUTER_TEST_SET_NAME) or "")
        inner_path_text = str(test_sets.get(HYBRID_INNER_TEST_SET_NAME) or "")
        if not top_path_text or not inner_path_text:
            raise ValueError(f"fold is missing explicit hybrid test set files: {manifest_path}")
        top_path = _resolve_path(top_path_text, split_root)
        inner_path = _resolve_path(inner_path_text, split_root)
        summary_path = _resolve_path(str(fold.get("summary_file")), split_root)
        summary = read_json(summary_path)
        if not feature_columns:
            feature_columns = [str(column) for column in summary.get("x_space_feature_columns", []) if str(column)]
            target_col = str(summary.get("split_target_col") or target_col)
        current_top = read_csv(top_path)
        if top_tail is None:
            top_tail = current_top
        else:
            left_ids = top_tail["ID"].astype(str).tolist() if "ID" in top_tail.columns else []
            right_ids = current_top["ID"].astype(str).tolist() if "ID" in current_top.columns else []
            if left_ids and right_ids and left_ids != right_ids:
                raise ValueError(f"inconsistent top20 test set across folds: {manifest_path}")
        cluster_id = int(fold.get("held_out_cluster_id", fold.get("fold_index", len(cluster_frames))))
        cluster_df = read_csv(inner_path)
        cluster_frames[cluster_id] = cluster_df
        background_parts.append(cluster_df)

    if top_tail is None:
        raise ValueError(f"could not load top-tail rows from: {manifest_path}")
    background = pd.concat(background_parts, ignore_index=True)
    return HybridClusterPayload(
        manifest_path=manifest_path,
        split_root=split_root,
        alloy_family=alloy_family,
        dataset_name=dataset_name,
        property_name=property_name,
        target_col=target_col,
        threshold_pct=20,
        source_mode="existing_hybrid_extrapolation_loco",
        top_tail=top_tail.reset_index(drop=True),
        background=background.reset_index(drop=True),
        cluster_frames=cluster_frames,
        feature_columns=feature_columns,
        data_file=str(manifest.get("data_file") or ""),
        split_id=split_id,
    )


def discover_hybrid_manifests(
    splits_root: Path,
    case_contains: str | None = None,
    limit: int | None = None,
    include_matbench: bool = True,
) -> list[Path]:
    manifests = sorted(
        path
        for path in splits_root.rglob("split_manifest.json")
        if path.parent.parent.name == "hybrid_extrapolation_loco"
    )
    if not include_matbench:
        manifests = [path for path in manifests if "MatbenchSteels" not in path.parts]
    if case_contains:
        needle = case_contains.lower()
        manifests = [path for path in manifests if needle in str(path).lower()]
    if limit is not None:
        manifests = manifests[: int(limit)]
    return manifests


def task_spec_from_payload(payload: HybridClusterPayload) -> TaskSpec:
    summary = read_json(_resolve_path(str(read_json(payload.manifest_path)["folds"][0]["summary_file"]), payload.split_root))
    data_file = _resolve_path(payload.data_file, REPO_ROOT)
    return TaskSpec(
        alloy_family=payload.alloy_family,
        dataset_name=payload.dataset_name,
        property_name=payload.property_name,
        target_col=payload.target_col,
        data_file=data_file,
        feature_columns=payload.feature_columns,
        cluster_count=int(summary.get("loco_cluster_count_actual") or len(payload.cluster_frames)),
        manifest_path=payload.manifest_path,
        split_id=payload.split_id,
    )


def _temporary_cluster_frames(
    background: pd.DataFrame,
    feature_columns: list[str],
    cluster_count: int,
    random_state: int,
) -> dict[int, pd.DataFrame]:
    cluster_count = max(1, min(int(cluster_count), len(background)))
    imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    scaler = StandardScaler()
    x = _numeric_feature_frame(background, feature_columns)
    scaled = scaler.fit_transform(imputer.fit_transform(x))
    if cluster_count == 1:
        labels = np.zeros(len(background), dtype=int)
    else:
        labels = KMeans(n_clusters=cluster_count, random_state=random_state, n_init=10).fit_predict(scaled)
    result: dict[int, pd.DataFrame] = {}
    working = background.copy()
    working["__temporary_cluster_id__"] = labels
    for cluster_id, cluster_frame in working.groupby("__temporary_cluster_id__", sort=True):
        result[int(cluster_id)] = cluster_frame.drop(columns=["__temporary_cluster_id__"]).reset_index(drop=True)
    return result


def load_task_dataframe(spec: TaskSpec) -> pd.DataFrame:
    frame = read_csv(spec.data_file)
    frame[spec.target_col] = pd.to_numeric(frame[spec.target_col], errors="coerce")
    frame = frame.dropna(subset=[spec.target_col]).reset_index(drop=True)
    return frame


def fallback_feature_columns(frame: pd.DataFrame, target_col: str, configured: list[str]) -> list[str]:
    present = [column for column in configured if column in frame.columns]
    if present:
        return present
    selected, _, _ = select_numeric_x_space_columns(frame, target_col)
    return selected


def run_actual_threshold(
    spec: TaskSpec,
    threshold: float,
    random_state: int,
    existing_payload: HybridClusterPayload | None = None,
    task_frame: pd.DataFrame | None = None,
    policy: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    threshold_pct = threshold_to_pct(threshold)
    metadata = {
        "alloy_family": spec.alloy_family,
        "dataset_name": spec.dataset_name,
        "property": spec.property_name,
        "target_col": spec.target_col,
        "threshold": float(threshold),
        "threshold_pct": threshold_pct,
        "case_key": case_key(spec.alloy_family, spec.dataset_name, spec.property_name),
        "split_id": spec.split_id,
    }
    force_temporary = bool((policy or {}).get("has_drop"))
    if threshold_pct == 20 and existing_payload is not None and not force_temporary:
        top_tail = existing_payload.top_tail.copy()
        cluster_frames = {int(cluster_id): cluster_df.copy() for cluster_id, cluster_df in existing_payload.cluster_frames.items()}
        feature_columns = existing_payload.feature_columns
        if policy is not None:
            top_tail, _ = apply_outlier_policy(top_tail, spec.alloy_family, spec.property_name, policy)
            cluster_frames = {
                int(cluster_id): apply_outlier_policy(cluster_df, spec.alloy_family, spec.property_name, policy)[0]
                for cluster_id, cluster_df in cluster_frames.items()
            }
        metadata.update(
            {
                "source_mode": existing_payload.source_mode,
                "manifest_path": str(existing_payload.manifest_path),
            }
        )
    else:
        frame = task_frame.copy() if task_frame is not None else load_task_dataframe(spec)
        feature_columns = fallback_feature_columns(frame, spec.target_col, spec.feature_columns)
        top_tail, background = select_top_tail(frame, spec.target_col, threshold)
        cluster_frames = _temporary_cluster_frames(background, feature_columns, spec.cluster_count, random_state)
        metadata.update({"source_mode": "temporary_threshold_loco_kmeans", "manifest_path": str(spec.manifest_path or "")})
    return compute_cluster_proximity(top_tail, cluster_frames, feature_columns, metadata)


def run_space_shift_threshold(
    spec: TaskSpec,
    threshold: float,
    random_state: int,
    sliced_projections: int,
    existing_payload: HybridClusterPayload | None = None,
    task_frame: pd.DataFrame | None = None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    threshold_pct = threshold_to_pct(threshold)
    metadata = {
        "alloy_family": spec.alloy_family,
        "dataset_name": spec.dataset_name,
        "property": spec.property_name,
        "target_col": spec.target_col,
        "threshold": float(threshold),
        "threshold_pct": threshold_pct,
        "case_key": case_key(spec.alloy_family, spec.dataset_name, spec.property_name),
        "split_id": spec.split_id,
    }
    force_temporary = bool((policy or {}).get("has_drop"))
    if threshold_pct == 20 and existing_payload is not None and not force_temporary:
        top_tail = existing_payload.top_tail.copy()
        background = existing_payload.background.copy()
        feature_columns = existing_payload.feature_columns
        if policy is not None:
            top_tail, _ = apply_outlier_policy(top_tail, spec.alloy_family, spec.property_name, policy)
            background, _ = apply_outlier_policy(background, spec.alloy_family, spec.property_name, policy)
        metadata.update(
            {
                "source_mode": existing_payload.source_mode,
                "manifest_path": str(existing_payload.manifest_path),
            }
        )
    else:
        frame = task_frame.copy() if task_frame is not None else load_task_dataframe(spec)
        feature_columns = fallback_feature_columns(frame, spec.target_col, spec.feature_columns)
        top_tail, background = select_top_tail(frame, spec.target_col, threshold)
        metadata.update({"source_mode": "temporary_threshold_top_tail", "manifest_path": str(spec.manifest_path or "")})
    return compute_space_shift_metrics(
        top_tail=top_tail,
        background=background,
        feature_columns=feature_columns,
        target_col=spec.target_col,
        metadata=metadata,
        sliced_projections=sliced_projections,
        random_state=random_state,
    )


def summarize_threshold(cluster_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = [*CASE_COLUMNS, "case_key", "threshold", "threshold_pct", "source_mode"]
    for keys, group in cluster_summary.groupby(group_cols, sort=True, dropna=False):
        values = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        ordered = group.sort_values("distance_rank", kind="stable")
        nearest = ordered.iloc[0]
        rows.append(
            {
                **values,
                "top_tail_size": int(nearest["top_tail_size"]),
                "cluster_count": int(group["cluster_id"].nunique()),
                "nearest_cluster_id": int(nearest["cluster_id"]),
                "nearest_cluster_size": int(nearest["cluster_size"]),
                "nearest_cluster_mean_distance": finite_float(nearest["mean_distance_to_top_tail"]),
                "second_nearest_cluster_mean_distance": finite_float(nearest["second_nearest_cluster_mean_distance"]),
                "concentration_score": finite_float(nearest["concentration_score"]),
                "manifest_path": str(nearest.get("manifest_path", "")),
            }
        )
    return pd.DataFrame(rows)


def build_manifest_rows(
    specs: list[TaskSpec],
    output_root: Path,
    outlier_policy: str,
    config_path: Path | None,
    outlier_counts: dict[str, dict[str, int]] | None = None,
) -> pd.DataFrame:
    outlier_counts = outlier_counts or {}
    return pd.DataFrame(
        [
            {
                "alloy_family": spec.alloy_family,
                "dataset_name": spec.dataset_name,
                "property": spec.property_name,
                "case_key": case_key(spec.alloy_family, spec.dataset_name, spec.property_name),
                "target_col": spec.target_col,
                "data_file": str(spec.data_file),
                "hybrid_manifest_path": str(spec.manifest_path or ""),
                "split_id": spec.split_id,
                "feature_count": int(len(spec.feature_columns)),
                "cluster_count": int(spec.cluster_count),
                "output_root": str(output_root),
                "outlier_policy": outlier_policy,
                "config_path": str(config_path or ""),
                "dropped_sample_count": int(outlier_counts.get(case_key(spec.alloy_family, spec.dataset_name, spec.property_name), {}).get("drop", 0)),
                "flagged_sample_count": int(outlier_counts.get(case_key(spec.alloy_family, spec.dataset_name, spec.property_name), {}).get("flag", 0)),
            }
            for spec in specs
        ]
    )


def save_figure(fig: plt.Figure, output_base: Path, formats: Iterable[str], dpi: int) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_base.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def clear_existing_figures(output_dir: Path, formats: Iterable[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        for path in output_dir.glob(f"*.{fmt}"):
            path.unlink()


def plot_top20(cluster_top20: pd.DataFrame, output_root: Path, formats: list[str], dpi: int) -> None:
    figure_dir = output_root / "figures" / "top20_cluster_distance_by_task"
    clear_existing_figures(figure_dir, formats)
    increase_color = "#d62728"
    for _, group in cluster_top20.groupby(CASE_COLUMNS, sort=True):
        ordered = group.sort_values("distance_rank", kind="stable")
        label = f"{ordered.iloc[0]['alloy_family']} / {ordered.iloc[0]['property']}"
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        x = np.arange(len(ordered), dtype=float)
        distances = pd.to_numeric(ordered["mean_distance_to_top_tail"], errors="coerce").to_numpy(dtype=float)
        min_distance = float(np.nanmin(distances))
        ax.plot(
            x,
            distances,
            marker="o",
            linewidth=1.8,
            color="#4c78a8",
        )
        ax.margins(y=0.12)
        ax.set_xticks(x)
        ax.set_xticklabels([f"cluster {cluster_id}" for cluster_id in ordered["cluster_id"].astype(str)])
        ax.set_xlabel("LOCO cluster ranked by proximity")
        ax.set_ylabel("Mean standardized X-space Euclidean distance\nto cluster centroid")
        ax.set_title(label)
        ax.grid(axis="y", color="#e6e6e6")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if math.isfinite(min_distance) and min_distance > 1e-12:
            increase_rates = ((distances - min_distance) / min_distance) * 100.0
            for x_value, distance, increase_rate in zip(x, distances, increase_rates, strict=True):
                if not (math.isfinite(distance) and math.isfinite(increase_rate)):
                    continue
                ax.annotate(
                    f"{increase_rate:.1f}%",
                    xy=(x_value, distance),
                    xytext=(0, 9),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    color=increase_color,
                    fontsize=9,
                )
            secondary = ax.secondary_yaxis(
                "right",
                functions=(
                    lambda distance, baseline=min_distance: ((distance - baseline) / baseline) * 100.0,
                    lambda increase, baseline=min_distance: baseline * (1.0 + (increase / 100.0)),
                ),
            )
            secondary.set_ylabel("Increase relative to nearest cluster (%)")
            secondary.spines["top"].set_visible(False)
            secondary.spines["right"].set_color(increase_color)
            secondary.tick_params(axis="y", colors=increase_color)
            secondary.yaxis.label.set_color(increase_color)
        out = figure_dir / case_key(
            ordered.iloc[0]["alloy_family"],
            ordered.iloc[0]["dataset_name"],
            ordered.iloc[0]["property"],
        )
        save_figure(fig, out, formats, dpi)


def plot_space_shift_sweep(space_shift: pd.DataFrame, output_root: Path, formats: list[str], dpi: int) -> None:
    if space_shift.empty:
        return
    figure_dir = output_root / "figures" / "threshold_sweep_space_shift_by_task"
    clear_existing_figures(figure_dir, formats)
    for _, group in space_shift.groupby(CASE_COLUMNS, sort=True):
        work = group.sort_values("threshold_pct", kind="stable")
        x = work["top_tail_size"].to_numpy(dtype=float)
        labels = [f"{int(row.top_tail_size)}\n({int(row.threshold_pct)}%)" for row in work.itertuples()]
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.plot(x, work["Wy"], marker="o", linewidth=1.8, color="#d95f02", label="Wy")
        ax.plot(x, work["Wx"], marker="s", linewidth=1.8, color="#4c78a8", label="Wx")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Top-tail sample count")
        ax.set_ylabel("Wasserstein distance")
        ax.set_title(f"{work.iloc[0]['alloy_family']} / {work.iloc[0]['property']}")
        ax.legend(frameon=False)
        ax.grid(axis="y", color="#e6e6e6")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        out = figure_dir / case_key(
            work.iloc[0]["alloy_family"],
            work.iloc[0]["dataset_name"],
            work.iloc[0]["property"],
        )
        save_figure(fig, out, formats, dpi)


def write_report(
    output_root: Path,
    top20: pd.DataFrame,
    sweep_summary: pd.DataFrame,
    space_shift: pd.DataFrame,
    outlier_policy: str,
    outlier_manifest: pd.DataFrame,
) -> None:
    report_dir = output_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    focus = sweep_summary[
        sweep_summary["case_key"].isin(
            {
                "Ti__titanium__UTS_MPa",
                "Al__aluminum__UTS_MPa",
                "Steel__steel__El",
                "Steel__steel__El_",
            }
        )
    ].copy()
    lines = [
        "# X-space top-tail LOCO proximity report",
        "",
        "This analysis uses X-space only: composition plus numeric processing features.",
        "",
        f"This run used outlier policy `{outlier_policy}`.",
        "",
        "## Files",
        "",
        "- `csv/cluster_proximity_top20.csv`",
        "- `csv/cluster_proximity_top20_sample_level.csv`",
        "- `csv/threshold_sweep_summary.csv`",
        "- `csv/threshold_sweep_sample_level.csv`",
        "- `csv/threshold_sweep_space_shift.csv`",
        "- `csv/analysis_manifest.csv`",
        "- `csv/outlier_handling_manifest.csv`",
        "- `csv/analysis_config_resolved.json`",
        "",
        "## Summary",
        "",
        f"- Top-20 cluster rows: {len(top20)}",
        f"- Threshold summary rows: {len(sweep_summary)}",
        f"- Threshold space-shift rows: {len(space_shift)}",
        f"- Outlier handling rows: {len(outlier_manifest)}",
        f"- Top-20 figures: `figures/top20_cluster_distance_by_task/`",
        f"- Threshold space-shift figures: `figures/threshold_sweep_space_shift_by_task/`",
        "",
        "## Top-20 cluster proximity metric",
        "",
        "- The top-20 figure y-axis is the mean standardized X-space Euclidean distance from top-tail samples to each LOCO cluster centroid.",
        "- X-space means composition plus numeric processing features, with imputer/scaler fit on the remaining background samples only.",
        "- Clusters are ordered by proximity rank, from nearest to farthest.",
        "",
        "## Threshold space-shift metrics",
        "",
        "- `Wy`: Wasserstein-1 distance between background and top-tail target values after standardizing the target with background-only mean/std.",
        "- `Wx`: sliced Wasserstein distance between background and top-tail X-space features after fitting imputer/scaler on background only.",
        "- Figures are written to `figures/threshold_sweep_space_shift_by_task/` and use top-tail sample count on the x-axis.",
        "",
    ]
    if not focus.empty:
        lines.extend(["## Focus tasks", ""])
        for _, row in focus.sort_values(["case_key", "threshold_pct"]).iterrows():
            lines.append(
                f"- {row['alloy_family']} / {row['property']} @ {row['threshold_pct']}%: "
                f"nearest cluster {row['nearest_cluster_id']}, concentration {row['concentration_score']:.4g}"
            )
    (report_dir / "xspace_top_tail_loco_proximity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve() if args.config else None
    config = load_analysis_config(config_path)
    policy = resolve_outlier_policy(config, str(args.outlier_policy))
    defaults = dict(config.get("defaults") or {})
    thresholds = list(args.thresholds) if args.thresholds is not None else list(defaults.get("thresholds") or DEFAULT_THRESHOLDS)
    sliced_projections = int(args.sliced_projections if args.sliced_projections is not None else defaults.get("sliced_projections", 128))
    include_matbench = bool(defaults.get("include_matbench", True)) and not bool(args.exclude_matbench)
    output_root = Path(args.output_root) if args.output_root else DEFAULT_OUTPUT_PARENT / str(policy.get("output_suffix") or "xspace")
    csv_dir = output_root / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    payloads = [
        build_cluster_frames_from_hybrid_manifest(path)
        for path in discover_hybrid_manifests(
            Path(args.splits_root),
            args.case_contains,
            args.limit,
            include_matbench=include_matbench,
        )
    ]
    if not payloads:
        raise SystemExit(f"No hybrid_extrapolation_loco manifests found under {args.splits_root}")
    specs = [task_spec_from_payload(payload) for payload in payloads]
    payload_by_case = {case_key(p.alloy_family, p.dataset_name, p.property_name): p for p in payloads}
    task_frames: dict[str, pd.DataFrame] = {}
    outlier_rows: list[dict[str, Any]] = []
    outlier_counts: dict[str, dict[str, int]] = {}
    for spec in specs:
        current_key = case_key(spec.alloy_family, spec.dataset_name, spec.property_name)
        processed_frame, current_rows = apply_outlier_policy(
            load_task_dataframe(spec),
            spec.alloy_family,
            spec.property_name,
            policy,
        )
        task_frames[current_key] = processed_frame
        for row in current_rows:
            row["dataset_name"] = spec.dataset_name
            row["case_key"] = current_key
        outlier_rows.extend(current_rows)
        outlier_counts[current_key] = {
            action: len({int(row["ID"]) for row in current_rows if row.get("action") == action})
            for action in ("drop", "flag")
        }

    cluster_parts: list[pd.DataFrame] = []
    sample_parts: list[pd.DataFrame] = []
    space_shift_rows: list[dict[str, Any]] = []
    for spec in specs:
        current_key = case_key(spec.alloy_family, spec.dataset_name, spec.property_name)
        existing_payload = payload_by_case.get(current_key)
        for threshold in thresholds:
            cluster_summary, sample_level = run_actual_threshold(
                spec,
                float(threshold),
                int(args.random_state),
                existing_payload=existing_payload,
                task_frame=task_frames[current_key],
                policy=policy,
            )
            cluster_parts.append(cluster_summary)
            sample_parts.append(sample_level)
            space_shift_rows.append(
                run_space_shift_threshold(
                    spec,
                    float(threshold),
                    int(args.random_state),
                    sliced_projections,
                    existing_payload=existing_payload,
                    task_frame=task_frames[current_key],
                    policy=policy,
                )
            )

    cluster_all = pd.concat(cluster_parts, ignore_index=True)
    sample_all = pd.concat(sample_parts, ignore_index=True)
    space_shift = pd.DataFrame(space_shift_rows)
    sweep_summary = summarize_threshold(cluster_all)
    if not space_shift.empty:
        merge_cols = [*CASE_COLUMNS, "case_key", "threshold", "threshold_pct"]
        space_cols = [
            *merge_cols,
            "background_size",
            "feature_count",
            "Wy",
            "Wx",
            "x_distance_method",
            "x_sliced_projections",
            "y_distance_method",
        ]
        sweep_summary = sweep_summary.merge(space_shift[space_cols], on=merge_cols, how="left")
    top20 = cluster_all[cluster_all["threshold_pct"].eq(20)].copy()
    top20_sample = sample_all[sample_all["threshold_pct"].eq(20)].copy()

    write_csv(top20, csv_dir / "cluster_proximity_top20.csv")
    write_csv(top20_sample, csv_dir / "cluster_proximity_top20_sample_level.csv")
    write_csv(sweep_summary, csv_dir / "threshold_sweep_summary.csv")
    write_csv(sample_all, csv_dir / "threshold_sweep_sample_level.csv")
    write_csv(space_shift, csv_dir / "threshold_sweep_space_shift.csv")
    write_csv(
        build_manifest_rows(specs, output_root, str(policy["name"]), config_path, outlier_counts),
        csv_dir / "analysis_manifest.csv",
    )
    outlier_manifest_columns = [
        "outlier_policy",
        "set_name",
        "action",
        "alloy_family",
        "dataset_name",
        "property",
        "case_key",
        "ID",
        "flag_column",
        "reason",
    ]
    write_csv(pd.DataFrame(outlier_rows, columns=outlier_manifest_columns), csv_dir / "outlier_handling_manifest.csv")
    write_json(
        {
            "config_path": str(config_path or ""),
            "outlier_policy": policy,
            "thresholds": thresholds,
            "include_matbench": include_matbench,
            "sliced_projections": sliced_projections,
            "output_root": str(output_root),
        },
        csv_dir / "analysis_config_resolved.json",
    )

    if not args.no_figures:
        plot_top20(top20, output_root, list(args.formats), int(args.dpi))
        plot_space_shift_sweep(space_shift, output_root, list(args.formats), int(args.dpi))
    outlier_manifest = pd.DataFrame(outlier_rows, columns=outlier_manifest_columns)
    write_report(output_root, top20, sweep_summary, space_shift, str(policy["name"]), outlier_manifest)

    print(f"Tasks: {len(specs)}")
    print(f"Outlier policy: {policy['name']}")
    print(f"Threshold summary rows: {len(sweep_summary)}")
    print(f"Threshold space-shift rows: {len(space_shift)}")
    print(f"Outlier handling rows: {len(outlier_manifest)}")
    print(f"Output root: {output_root}")


def main() -> None:
    run_analysis(parse_args())


if __name__ == "__main__":
    main()
