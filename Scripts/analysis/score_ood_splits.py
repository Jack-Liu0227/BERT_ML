from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from scipy.stats import wasserstein_distance
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


HELPER_COLUMNS = {"__row_id__", "__source_index__"}
DEFAULT_SPLITS_ROOT = Path("output") / "ood_splits"
DEFAULT_OUTPUT_ROOT = Path("output") / "ood_xspace_scores"


@dataclass(frozen=True)
class SplitEntry:
    summary_path: Path
    split_dir: Path
    output_dir: Path
    alloy_family: str
    dataset_name: str
    property_name: str
    split_strategy: str
    split_id: str
    fold_id: str


@dataclass
class ScorePayload:
    sample_scores: pd.DataFrame
    train_scores: pd.DataFrame
    summary_row: dict[str, Any]
    warnings: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score existing OOD train/test splits in composition+processing X space. "
            "The script fits all scoring models on train rows only, then assigns "
            "per-test-sample OOD scores and split-level distribution distances."
        )
    )
    parser.add_argument("--splits-root", default=str(DEFAULT_SPLITS_ROOT), help="Root containing canonical OOD splits.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Directory for scoring outputs.")
    parser.add_argument("--method", action="append", default=None, help="Filter by split strategy; can be repeated.")
    parser.add_argument("--case-contains", default=None, help="Only score split paths containing this substring.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of splits/folds to score.")
    parser.add_argument("--knn-k", type=int, default=5, help="k for kNN distance scoring.")
    parser.add_argument("--kde-bandwidth", type=float, default=None, help="Optional Gaussian KDE bandwidth in scaled X space.")
    parser.add_argument("--sliced-projections", type=int, default=128, help="Projection count for sliced Wasserstein.")
    parser.add_argument("--metric-sample-size", type=int, default=1000, help="Sample cap for MMD and energy distance.")
    parser.add_argument(
        "--impute-strategy",
        default="zero",
        choices=["zero", "median", "mean"],
        help=(
            "How to fill missing X-space entries before train-fitted standard scaling. "
            "Use zero when missing/blank composition or processing entries mean physical absence."
        ),
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true", help="Skip per-split PNG figures.")
    parser.add_argument("--strict", action="store_true", help="Fail immediately on the first split error.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def finite_float(value: Any) -> float:
    try:
        result = float(value)
    except Exception:
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def discover_splits(
    splits_root: Path,
    output_root: Path,
    methods: set[str] | None = None,
    case_contains: str | None = None,
    limit: int | None = None,
) -> list[SplitEntry]:
    entries: list[SplitEntry] = []
    root = splits_root.resolve()
    output_base = output_root.resolve()
    for summary_path in sorted(root.rglob("split_summary.json")):
        if summary_path.parent.name != "split_data":
            continue
        rel_parts = summary_path.relative_to(root).parts
        if len(rel_parts) < 7:
            continue
        alloy_family, dataset_name, property_name, split_strategy, split_id = rel_parts[:5]
        if methods and split_strategy not in methods:
            continue
        rel_text = str(summary_path.relative_to(root))
        if case_contains and case_contains not in rel_text:
            continue

        fold_id = ""
        if len(rel_parts) >= 9 and rel_parts[5] == "folds":
            fold_id = rel_parts[6]
        split_dir = summary_path.parent
        output_rel = split_dir.relative_to(root)
        if output_rel.name == "split_data":
            output_rel = output_rel.parent
        entries.append(
            SplitEntry(
                summary_path=summary_path,
                split_dir=split_dir,
                output_dir=output_base / output_rel,
                alloy_family=alloy_family,
                dataset_name=dataset_name,
                property_name=property_name,
                split_strategy=split_strategy,
                split_id=split_id,
                fold_id=fold_id,
            )
        )
        if limit is not None and len(entries) >= limit:
            break
    return entries


def resolve_split_files(split_dir: Path, summary: dict[str, Any]) -> tuple[Path, Path]:
    train_label = str(summary.get("train_label") or "train")
    test_label = str(summary.get("test_label") or "test")
    train_file = split_dir / f"{train_label}.csv"
    test_file = split_dir / f"{test_label}.csv"
    if train_file.exists() and test_file.exists():
        return train_file, test_file

    csv_files = sorted(path for path in split_dir.glob("*.csv") if path.is_file())
    train_candidates = [path for path in csv_files if path.stem.lower().startswith("train")]
    test_candidates = [path for path in csv_files if path.stem.lower().startswith("test")]
    if train_candidates and test_candidates:
        return train_candidates[0], test_candidates[0]
    raise FileNotFoundError(f"Could not resolve train/test CSV files in {split_dir}")


def resolve_feature_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    summary: dict[str, Any],
    target_col: str,
    warnings: list[str],
) -> list[str]:
    configured = [str(col) for col in summary.get("x_space_feature_columns", []) if str(col)]
    if configured:
        missing_train = [col for col in configured if col not in train_df.columns]
        missing_test = [col for col in configured if col not in test_df.columns]
        if missing_train:
            warnings.append(f"missing train feature columns: {missing_train[:10]}")
        if missing_test:
            warnings.append(f"missing test feature columns: {missing_test[:10]}")
        return configured

    warnings.append("split_summary.json has no x_space_feature_columns; falling back to numeric non-helper columns")
    excluded = {target_col, *HELPER_COLUMNS}
    numeric_cols = train_df.select_dtypes(include=[np.number, "bool"]).columns
    return [str(col) for col in numeric_cols if str(col) not in excluded]


def ensure_numeric_feature_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    for column in columns:
        if column in frame.columns:
            result[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            result[column] = np.nan
    return result


def drop_unusable_train_columns(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    warnings: list[str],
    keep_all_missing: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if keep_all_missing:
        return train_x.copy(), test_x.copy()
    usable = [col for col in train_x.columns if not train_x[col].isna().all()]
    dropped = [col for col in train_x.columns if col not in usable]
    if dropped:
        warnings.append(f"dropped all-missing train feature columns: {dropped[:10]}")
    return train_x[usable].copy(), test_x[usable].copy()


def make_imputer(strategy: str) -> SimpleImputer:
    if strategy == "zero":
        return SimpleImputer(strategy="constant", fill_value=0.0)
    return SimpleImputer(strategy=strategy)


def estimate_kde_bandwidth(values: np.ndarray) -> float:
    if len(values) < 2:
        return 1.0
    scale_candidates = np.std(values, axis=0, ddof=1)
    scale_candidates = scale_candidates[np.isfinite(scale_candidates) & (scale_candidates > 0)]
    scale = float(np.median(scale_candidates)) if len(scale_candidates) else 1.0
    bandwidth = scale * (len(values) ** (-1.0 / (values.shape[1] + 4)))
    return float(max(bandwidth, 1e-3))


def knn_scores(reference: np.ndarray, query: np.ndarray, k: int, exclude_self: bool) -> tuple[np.ndarray, np.ndarray]:
    if len(reference) == 0 or len(query) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    if exclude_self and len(reference) > 1:
        n_neighbors = min(max(2, k + 1), len(reference))
        distances, _ = NearestNeighbors(n_neighbors=n_neighbors).fit(reference).kneighbors(query)
        distances = distances[:, 1:]
    else:
        n_neighbors = min(max(1, k), len(reference))
        distances, _ = NearestNeighbors(n_neighbors=n_neighbors).fit(reference).kneighbors(query)
    nearest = distances[:, 0] if distances.shape[1] else np.zeros(len(query), dtype=float)
    mean_distance = distances.mean(axis=1) if distances.shape[1] else np.zeros(len(query), dtype=float)
    return nearest, mean_distance


def mahalanobis_scores(train_scaled: np.ndarray, query_scaled: np.ndarray) -> np.ndarray:
    if len(query_scaled) == 0:
        return np.array([], dtype=float)
    if len(train_scaled) < 2:
        center = train_scaled[0] if len(train_scaled) else np.zeros(query_scaled.shape[1], dtype=float)
        return np.linalg.norm(query_scaled - center, axis=1)
    center = np.mean(train_scaled, axis=0)
    covariance = LedoitWolf().fit(train_scaled).covariance_
    inv_covariance = np.linalg.pinv(covariance)
    delta = query_scaled - center
    squared = np.einsum("ij,jk,ik->i", delta, inv_covariance, delta)
    return np.sqrt(np.maximum(squared, 0.0))


def kde_neg_log_density(train_scaled: np.ndarray, query_scaled: np.ndarray, bandwidth: float | None) -> tuple[np.ndarray, float]:
    if len(query_scaled) == 0:
        return np.array([], dtype=float), float("nan")
    actual_bandwidth = float(bandwidth) if bandwidth is not None else estimate_kde_bandwidth(train_scaled)
    kde = KernelDensity(kernel="gaussian", bandwidth=actual_bandwidth)
    kde.fit(train_scaled)
    return -kde.score_samples(query_scaled), actual_bandwidth


def positive_robust_z(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    finite_reference = reference[np.isfinite(reference)]
    if len(finite_reference) == 0:
        return np.zeros_like(values, dtype=float)
    center = float(np.median(finite_reference))
    q25, q75 = np.percentile(finite_reference, [25, 75])
    scale = float(q75 - q25)
    if not math.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(finite_reference))
    if not math.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return np.maximum((values - center) / scale, 0.0)


def percentile_against_reference(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    finite_reference = np.sort(reference[np.isfinite(reference)])
    if len(finite_reference) == 0:
        return np.full(len(values), np.nan)
    result = []
    for value in values:
        if not math.isfinite(float(value)):
            result.append(float("nan"))
            continue
        left = np.searchsorted(finite_reference, value, side="left")
        right = np.searchsorted(finite_reference, value, side="right")
        result.append(((left + right) / 2.0) / len(finite_reference) * 100.0)
    return np.asarray(result, dtype=float)


def downsample_rows(array: np.ndarray, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    if len(array) <= max_rows:
        return array
    indices = rng.choice(len(array), size=max_rows, replace=False)
    return array[np.sort(indices)]


def random_unit_directions(dim: int, projection_count: int, rng: np.random.Generator) -> np.ndarray:
    projection_count = max(1, int(projection_count))
    directions = rng.normal(size=(projection_count, dim))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    return directions / np.maximum(norms, 1e-12)


def _one_dimensional_wasserstein_test_sample_scores(
    train_values: np.ndarray,
    test_values: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    train_count = len(train_values)
    test_count = len(test_values)
    if train_count == 0 or test_count == 0:
        return float("nan"), np.full(test_count, np.nan), np.full(test_count, np.nan)

    train_order = np.argsort(train_values, kind="stable")
    test_order = np.argsort(test_values, kind="stable")
    train_sorted = train_values[train_order]
    test_sorted = test_values[test_order]
    train_weight = 1.0 / train_count
    test_weight = 1.0 / test_count

    sample_contribution = np.zeros(test_count, dtype=float)
    total_distance = 0.0
    train_index = 0
    test_index = 0
    train_mass = train_weight
    test_mass = test_weight
    eps = 1e-15

    while train_index < train_count and test_index < test_count:
        moved_mass = min(train_mass, test_mass)
        distance = abs(float(train_sorted[train_index]) - float(test_sorted[test_index]))
        contribution = moved_mass * distance
        total_distance += contribution
        sample_contribution[int(test_order[test_index])] += contribution

        train_mass -= moved_mass
        test_mass -= moved_mass
        if train_mass <= eps:
            train_index += 1
            train_mass = train_weight
        if test_mass <= eps:
            test_index += 1
            test_mass = test_weight

    # sample_score has the same unit as W1. Its mean over test samples equals
    # the one-dimensional W1 distance for this projection.
    sample_score = sample_contribution / test_weight
    return float(total_distance), sample_score, sample_contribution


def sliced_wasserstein_distance_and_sample_scores(
    train_scaled: np.ndarray,
    test_scaled: np.ndarray,
    directions: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    if len(train_scaled) == 0 or len(test_scaled) == 0:
        return float("nan"), np.full(len(test_scaled), np.nan), np.full(len(test_scaled), np.nan)

    distances: list[float] = []
    sample_scores = np.zeros(len(test_scaled), dtype=float)
    sample_contributions = np.zeros(len(test_scaled), dtype=float)
    for direction in directions:
        train_projection = train_scaled @ direction
        test_projection = test_scaled @ direction
        distance, direction_sample_scores, direction_sample_contributions = _one_dimensional_wasserstein_test_sample_scores(
            train_projection,
            test_projection,
        )
        distances.append(distance)
        sample_scores += direction_sample_scores
        sample_contributions += direction_sample_contributions

    projection_count = len(directions)
    return (
        float(np.mean(distances)),
        sample_scores / projection_count,
        sample_contributions / projection_count,
    )


def energy_distance_multivariate(train_scaled: np.ndarray, test_scaled: np.ndarray, max_rows: int, rng: np.random.Generator) -> float:
    if len(train_scaled) == 0 or len(test_scaled) == 0:
        return float("nan")
    x = downsample_rows(train_scaled, max_rows, rng)
    y = downsample_rows(test_scaled, max_rows, rng)
    xy = cdist(x, y, metric="euclidean").mean()
    xx = cdist(x, x, metric="euclidean").mean()
    yy = cdist(y, y, metric="euclidean").mean()
    return float(max(0.0, 2.0 * xy - xx - yy))


def rbf_mmd(train_scaled: np.ndarray, test_scaled: np.ndarray, max_rows: int, rng: np.random.Generator) -> tuple[float, float]:
    if len(train_scaled) == 0 or len(test_scaled) == 0:
        return float("nan"), float("nan")
    x = downsample_rows(train_scaled, max_rows, rng)
    y = downsample_rows(test_scaled, max_rows, rng)
    combined = np.vstack([x, y])
    if len(combined) < 2:
        return 0.0, 1.0
    squared_distances = pdist(combined, metric="sqeuclidean")
    positive_distances = squared_distances[squared_distances > 1e-12]
    sigma_sq = float(np.median(positive_distances)) if len(positive_distances) else 1.0
    if not math.isfinite(sigma_sq) or sigma_sq <= 1e-12:
        sigma_sq = 1.0
    gamma = 1.0 / (2.0 * sigma_sq)
    kxx = np.exp(-gamma * cdist(x, x, metric="sqeuclidean")).mean()
    kyy = np.exp(-gamma * cdist(y, y, metric="sqeuclidean")).mean()
    kxy = np.exp(-gamma * cdist(x, y, metric="sqeuclidean")).mean()
    return float(max(0.0, kxx + kyy - 2.0 * kxy)), float(math.sqrt(sigma_sq))


def top_contributing_features(
    raw_values: np.ndarray,
    scaled_values: np.ndarray,
    feature_names: list[str],
    train_min: np.ndarray,
    train_max: np.ndarray,
    top_n: int = 5,
) -> tuple[str, str, int]:
    outside = (raw_values < train_min - 1e-12) | (raw_values > train_max + 1e-12)
    ranking = np.lexsort((-np.abs(scaled_values), ~outside))
    items: list[dict[str, Any]] = []
    for idx in ranking[:top_n]:
        items.append(
            {
                "feature": feature_names[int(idx)],
                "value": finite_float(raw_values[int(idx)]),
                "abs_train_z": finite_float(abs(scaled_values[int(idx)])),
                "train_min": finite_float(train_min[int(idx)]),
                "train_max": finite_float(train_max[int(idx)]),
                "outside_train_range": bool(outside[int(idx)]),
            }
        )
    outside_features = [feature_names[int(idx)] for idx in np.where(outside)[0]]
    return (
        json.dumps(items, ensure_ascii=False),
        json.dumps(outside_features, ensure_ascii=False),
        int(np.sum(outside)),
    )


def projection_2d(train_scaled: np.ndarray, test_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if train_scaled.shape[1] >= 2 and len(train_scaled) >= 2:
        pca = PCA(n_components=2, random_state=0)
        train_projection = pca.fit_transform(train_scaled)
        test_projection = pca.transform(test_scaled)
        return train_projection, test_projection
    train_projection = np.zeros((len(train_scaled), 2), dtype=float)
    test_projection = np.zeros((len(test_scaled), 2), dtype=float)
    if train_scaled.shape[1] >= 1:
        train_projection[:, 0] = train_scaled[:, 0]
        test_projection[:, 0] = test_scaled[:, 0]
    return train_projection, test_projection


def plot_outputs(output_dir: Path, train_scores: pd.DataFrame, sample_scores: pd.DataFrame, train_projection: np.ndarray, test_projection: np.ndarray) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(train_scores["ood_score"], bins=30, alpha=0.65, label="train self-score", color="#4c78a8")
    ax.hist(sample_scores["ood_score"], bins=30, alpha=0.65, label="test score", color="#f58518")
    ax.set_xlabel("OOD score")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "score_distribution.png", dpi=220)
    plt.close(fig)

    ranked = sample_scores.sort_values("ood_score", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(np.arange(1, len(ranked) + 1), ranked["ood_score"], marker="o", linewidth=1.2, markersize=3)
    ax.set_xlabel("test sample rank")
    ax.set_ylabel("OOD score")
    fig.tight_layout()
    fig.savefig(figure_dir / "test_score_rank.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(train_projection[:, 0], train_projection[:, 1], s=14, alpha=0.35, color="#7f7f7f", label="train")
    scatter = ax.scatter(
        test_projection[:, 0],
        test_projection[:, 1],
        c=sample_scores["ood_score"],
        s=34,
        alpha=0.9,
        cmap="viridis",
        label="test",
    )
    fig.colorbar(scatter, ax=ax, label="OOD score")
    ax.set_xlabel("PCA-1")
    ax.set_ylabel("PCA-2")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(figure_dir / "projection_ood_score.png", dpi=220)
    plt.close(fig)


def build_method_comparison(summary_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "test_ood_score_median",
        "test_ood_score_mean",
        "test_ood_score_top10pct_mean",
        "sliced_wasserstein",
        "mmd_rbf",
        "energy_distance",
        "test_ood_percentile_median",
    ]
    rows: list[dict[str, Any]] = []
    for method, method_df in summary_df.groupby("split_strategy", sort=True):
        row: dict[str, Any] = {
            "split_strategy": method,
            "split_count": int(len(method_df)),
            "case_count": int(
                method_df[["alloy_family", "dataset_name", "property"]]
                .drop_duplicates()
                .shape[0]
            ),
        }
        for metric_col in metric_cols:
            values = pd.to_numeric(method_df[metric_col], errors="coerce")
            row[f"{metric_col}_median"] = finite_float(values.median())
            row[f"{metric_col}_mean"] = finite_float(values.mean())
        rows.append(row)

    comparison = pd.DataFrame(rows)
    if comparison.empty:
        return comparison

    baseline = comparison.loc[comparison["split_strategy"] == "random_cv_baseline"]
    if not baseline.empty:
        baseline_row = baseline.iloc[0]
        for metric_col in metric_cols:
            baseline_value = finite_float(baseline_row.get(f"{metric_col}_median"))
            ratio_col = f"{metric_col}_median_vs_random_cv_ratio"
            delta_col = f"{metric_col}_median_minus_random_cv"
            comparison[delta_col] = comparison[f"{metric_col}_median"] - baseline_value
            if math.isfinite(baseline_value) and abs(baseline_value) > 1e-12:
                comparison[ratio_col] = comparison[f"{metric_col}_median"] / baseline_value
            else:
                comparison[ratio_col] = np.nan

    order = {
        "random_cv_baseline": 0,
        "target_extrapolation": 1,
        "sparse_y_single": 2,
        "sparse_y_cluster": 3,
        "sparse_x_single": 4,
        "sparse_x_cluster": 5,
        "loco": 6,
    }
    comparison["_sort_order"] = comparison["split_strategy"].map(order).fillna(999)
    comparison = comparison.sort_values(["_sort_order", "split_strategy"], kind="stable").drop(columns=["_sort_order"])
    return comparison


def plot_method_comparison(comparison: pd.DataFrame, output_root: Path) -> None:
    if comparison.empty or "test_ood_score_median_median" not in comparison.columns:
        return
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_df = comparison.copy()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(plot_df["split_strategy"], plot_df["test_ood_score_median_median"], color="#4c78a8")
    ax.set_xlabel("OOD split strategy")
    ax.set_ylabel("median split-level test OOD score")
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    fig.savefig(figure_dir / "method_median_ood_score.png", dpi=220)
    plt.close(fig)


def build_score_payload(entry: SplitEntry, args: argparse.Namespace) -> ScorePayload:
    warnings: list[str] = []
    summary = read_json(entry.summary_path)
    target_col = str(summary.get("split_target_col") or summary.get("target_column") or entry.property_name)
    train_file, test_file = resolve_split_files(entry.split_dir, summary)
    train_df = read_csv(train_file)
    test_df = read_csv(test_file)
    if train_df.empty or test_df.empty:
        raise ValueError("train or test split is empty")

    feature_columns = resolve_feature_columns(train_df, test_df, summary, target_col, warnings)
    train_x_raw = ensure_numeric_feature_frame(train_df, feature_columns)
    test_x_raw = ensure_numeric_feature_frame(test_df, feature_columns)
    impute_strategy = str(getattr(args, "impute_strategy", "zero")).lower()
    train_x_raw, test_x_raw = drop_unusable_train_columns(
        train_x_raw,
        test_x_raw,
        warnings,
        keep_all_missing=(impute_strategy == "zero"),
    )
    if train_x_raw.empty:
        raise ValueError("no usable X-space feature columns remain")

    imputer = make_imputer(impute_strategy)
    scaler = StandardScaler()
    train_imputed = imputer.fit_transform(train_x_raw)
    test_imputed = imputer.transform(test_x_raw)
    train_scaled = scaler.fit_transform(train_imputed)
    test_scaled = scaler.transform(test_imputed)
    feature_names = list(train_x_raw.columns)

    k = max(1, int(args.knn_k))
    train_nearest, train_knn_mean = knn_scores(train_scaled, train_scaled, k=k, exclude_self=True)
    test_nearest, test_knn_mean = knn_scores(train_scaled, test_scaled, k=k, exclude_self=False)
    train_mahal = mahalanobis_scores(train_scaled, train_scaled)
    test_mahal = mahalanobis_scores(train_scaled, test_scaled)
    train_kde_nll, kde_bandwidth = kde_neg_log_density(train_scaled, train_scaled, args.kde_bandwidth)
    test_kde_nll, _ = kde_neg_log_density(train_scaled, test_scaled, args.kde_bandwidth)

    train_knn_z = positive_robust_z(train_knn_mean, train_knn_mean)
    test_knn_z = positive_robust_z(test_knn_mean, train_knn_mean)
    train_mahal_z = positive_robust_z(train_mahal, train_mahal)
    test_mahal_z = positive_robust_z(test_mahal, train_mahal)
    train_kde_z = positive_robust_z(train_kde_nll, train_kde_nll)
    test_kde_z = positive_robust_z(test_kde_nll, train_kde_nll)
    train_ood_score = np.vstack([train_knn_z, train_mahal_z, train_kde_z]).mean(axis=0)
    test_ood_score = np.vstack([test_knn_z, test_mahal_z, test_kde_z]).mean(axis=0)
    test_percentile = percentile_against_reference(test_ood_score, train_ood_score)

    train_min = np.nanmin(train_imputed, axis=0)
    train_max = np.nanmax(train_imputed, axis=0)
    contribution_payloads = [
        top_contributing_features(test_imputed[idx], test_scaled[idx], feature_names, train_min, train_max)
        for idx in range(len(test_df))
    ]
    rng = np.random.default_rng(int(args.random_state))
    directions = random_unit_directions(train_scaled.shape[1], args.sliced_projections, rng)
    sliced_wasserstein, test_wasserstein_score, test_wasserstein_mass_contribution = (
        sliced_wasserstein_distance_and_sample_scores(train_scaled, test_scaled, directions)
    )

    sample_scores = pd.DataFrame(
        {
            "alloy_family": entry.alloy_family,
            "dataset_name": entry.dataset_name,
            "property": entry.property_name,
            "split_strategy": entry.split_strategy,
            "split_id": entry.split_id,
            "fold_id": entry.fold_id,
            "source_split_dir": str(entry.split_dir),
            "__row_id__": test_df["__row_id__"].to_numpy() if "__row_id__" in test_df.columns else np.arange(len(test_df)),
            "__source_index__": test_df["__source_index__"].to_numpy()
            if "__source_index__" in test_df.columns
            else np.arange(len(test_df)),
            "target_col": target_col,
            "target_value": pd.to_numeric(test_df[target_col], errors="coerce").to_numpy()
            if target_col in test_df.columns
            else np.nan,
            "nearest_train_distance": test_nearest,
            "knn_mean_distance": test_knn_mean,
            "mahalanobis_distance": test_mahal,
            "kde_neg_log_density": test_kde_nll,
            "knn_ood_z": test_knn_z,
            "mahalanobis_ood_z": test_mahal_z,
            "kde_ood_z": test_kde_z,
            "ood_score": test_ood_score,
            "sliced_wasserstein_sample_score": test_wasserstein_score,
            "sliced_wasserstein_mass_contribution": test_wasserstein_mass_contribution,
            "ood_percentile_vs_train": test_percentile,
            "range_violation_count": [item[2] for item in contribution_payloads],
            "range_violation_features": [item[1] for item in contribution_payloads],
            "top_contributing_features": [item[0] for item in contribution_payloads],
        }
    )
    sample_scores["ood_rank_desc"] = sample_scores["ood_score"].rank(method="first", ascending=False).astype(int)
    sample_scores["sliced_wasserstein_rank_desc"] = (
        sample_scores["sliced_wasserstein_sample_score"].rank(method="first", ascending=False).astype(int)
    )
    if "ID" in test_df.columns:
        sample_scores.insert(10, "ID", test_df["ID"].to_numpy())

    train_scores = pd.DataFrame(
        {
            "__row_id__": train_df["__row_id__"].to_numpy() if "__row_id__" in train_df.columns else np.arange(len(train_df)),
            "__source_index__": train_df["__source_index__"].to_numpy()
            if "__source_index__" in train_df.columns
            else np.arange(len(train_df)),
            "nearest_train_distance": train_nearest,
            "knn_mean_distance": train_knn_mean,
            "mahalanobis_distance": train_mahal,
            "kde_neg_log_density": train_kde_nll,
            "knn_ood_z": train_knn_z,
            "mahalanobis_ood_z": train_mahal_z,
            "kde_ood_z": train_kde_z,
            "ood_score": train_ood_score,
        }
    )

    energy_distance = energy_distance_multivariate(train_scaled, test_scaled, args.metric_sample_size, rng)
    mmd_value, mmd_sigma = rbf_mmd(train_scaled, test_scaled, args.metric_sample_size, rng)
    top_n = max(1, int(math.ceil(len(test_ood_score) * 0.1)))
    top_scores = np.sort(test_ood_score)[-top_n:]
    summary_row = {
        "alloy_family": entry.alloy_family,
        "dataset_name": entry.dataset_name,
        "property": entry.property_name,
        "target_col": target_col,
        "split_strategy": entry.split_strategy,
        "split_id": entry.split_id,
        "fold_id": entry.fold_id,
        "source_split_dir": str(entry.split_dir),
        "output_dir": str(entry.output_dir),
        "train_file": str(train_file),
        "test_file": str(test_file),
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "feature_count": int(len(feature_names)),
        "impute_strategy": impute_strategy,
        "kde_bandwidth": finite_float(kde_bandwidth),
        "sliced_wasserstein": finite_float(sliced_wasserstein),
        "mmd_rbf": finite_float(mmd_value),
        "mmd_rbf_sigma": finite_float(mmd_sigma),
        "energy_distance": finite_float(energy_distance),
        "train_ood_score_median": finite_float(np.median(train_ood_score)),
        "train_ood_score_95pct": finite_float(np.percentile(train_ood_score, 95)),
        "test_ood_score_mean": finite_float(np.mean(test_ood_score)),
        "test_ood_score_median": finite_float(np.median(test_ood_score)),
        "test_ood_score_max": finite_float(np.max(test_ood_score)),
        "test_ood_score_top10pct_mean": finite_float(np.mean(top_scores)),
        "test_ood_percentile_median": finite_float(np.median(test_percentile)),
        "test_ood_percentile_95pct": finite_float(np.percentile(test_percentile, 95)),
        "mean_range_violation_count": finite_float(sample_scores["range_violation_count"].mean()),
        "max_range_violation_count": int(sample_scores["range_violation_count"].max()),
        "warnings": " | ".join(warnings),
    }
    return ScorePayload(sample_scores=sample_scores, train_scores=train_scores, summary_row=summary_row, warnings=warnings)


def save_split_outputs(entry: SplitEntry, payload: ScorePayload, no_plots: bool, impute_strategy: str = "zero") -> None:
    entry.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(payload.sample_scores, entry.output_dir / "ood_sample_scores.csv")
    write_csv(payload.train_scores, entry.output_dir / "ood_train_self_scores.csv")
    write_csv(pd.DataFrame([payload.summary_row]), entry.output_dir / "ood_split_summary.csv")
    if not no_plots:
        train_scores = payload.train_scores
        sample_scores = payload.sample_scores
        summary = read_json(entry.summary_path)
        target_col = str(summary.get("split_target_col") or summary.get("target_column") or entry.property_name)
        train_file, test_file = resolve_split_files(entry.split_dir, summary)
        train_df = read_csv(train_file)
        test_df = read_csv(test_file)
        feature_columns = resolve_feature_columns(train_df, test_df, summary, target_col, [])
        train_x_raw, test_x_raw = drop_unusable_train_columns(
            ensure_numeric_feature_frame(train_df, feature_columns),
            ensure_numeric_feature_frame(test_df, feature_columns),
            [],
            keep_all_missing=(impute_strategy == "zero"),
        )
        imputer = make_imputer(impute_strategy)
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(imputer.fit_transform(train_x_raw))
        test_scaled = scaler.transform(imputer.transform(test_x_raw))
        train_projection, test_projection = projection_2d(train_scaled, test_scaled)
        plot_outputs(entry.output_dir, train_scores, sample_scores, train_projection, test_projection)


def score_entries(entries: Iterable[SplitEntry], args: argparse.Namespace) -> tuple[list[pd.DataFrame], list[dict[str, Any]], list[dict[str, Any]]]:
    sample_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    entries_list = list(entries)
    for index, entry in enumerate(entries_list, start=1):
        print(f"[{index}/{len(entries_list)}] scoring {entry.summary_path}")
        try:
            payload = build_score_payload(entry, args)
            save_split_outputs(
                entry,
                payload,
                no_plots=bool(args.no_plots),
                impute_strategy=str(getattr(args, "impute_strategy", "zero")).lower(),
            )
            sample_frames.append(payload.sample_scores)
            summary_rows.append(payload.summary_row)
        except Exception as exc:
            failure = {
                "summary_path": str(entry.summary_path),
                "split_dir": str(entry.split_dir),
                "split_strategy": entry.split_strategy,
                "fold_id": entry.fold_id,
                "error": repr(exc),
            }
            failure_rows.append(failure)
            print(f"  ERROR: {failure['error']}", file=sys.stderr)
            if args.strict:
                raise
    return sample_frames, summary_rows, failure_rows


def main() -> None:
    args = parse_args()
    splits_root = Path(args.splits_root)
    output_root = Path(args.output_root)
    methods = set(args.method) if args.method else None
    entries = discover_splits(
        splits_root=splits_root,
        output_root=output_root,
        methods=methods,
        case_contains=args.case_contains,
        limit=args.limit,
    )
    if not entries:
        raise SystemExit(f"No split_summary.json files found under {splits_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    sample_frames, summary_rows, failure_rows = score_entries(entries, args)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        write_csv(summary_df, output_root / "ood_split_summary.csv")
        method_comparison = build_method_comparison(summary_df)
        write_csv(method_comparison, output_root / "ood_method_comparison.csv")
        if not args.no_plots:
            plot_method_comparison(method_comparison, output_root)
    if sample_frames:
        write_csv(pd.concat(sample_frames, ignore_index=True), output_root / "all_ood_sample_scores.csv")
    if failure_rows:
        write_csv(pd.DataFrame(failure_rows), output_root / "ood_scoring_failures.csv")
    else:
        failure_path = output_root / "ood_scoring_failures.csv"
        if failure_path.exists():
            failure_path.unlink()

    print(f"Scored splits: {len(summary_rows)}")
    print(f"Failures: {len(failure_rows)}")
    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()
