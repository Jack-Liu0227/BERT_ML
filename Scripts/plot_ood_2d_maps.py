from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DIR = PROJECT_ROOT / "output" / "ood_results"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "ood_2d_maps"


METHOD_LAYOUT = [
    "random_cv_baseline",
    "loco",
    "target_extrapolation",
    "sparse_x_single",
    "sparse_y_single",
    "sparse_x_cluster",
    "sparse_y_cluster",
]

METHOD_ALIASES = {
    "random_cv_baseline": "RandomCV",
    "loco": "LOCO",
    "target_extrapolation": "Extra",
    "sparse_x_single": "SparseX-single",
    "sparse_y_single": "SparseY-single",
    "sparse_x_cluster": "SparseX-cluster",
    "sparse_y_cluster": "SparseY-cluster",
}

METHOD_DIR_SUFFIX = {
    "random_cv_baseline": "random_cv_baseline",
    "loco": "loco",
    "target_extrapolation": "extrapolation",
    "sparse_x_single": "sparse_x_single",
    "sparse_y_single": "sparse_y_single",
    "sparse_x_cluster": "sparse_x_cluster",
    "sparse_y_cluster": "sparse_y_cluster",
}

ALLOY_FAMILY_PATH_ALIASES = {
    "HEA_half": ("HEA_half", "HEA"),
    "HEA": ("HEA", "HEA_half"),
}

SINGLE_SPLIT_TEST_FILES = {
    "target_extrapolation": "test_high.csv",
    "sparse_x_single": "test_sparse_x_single.csv",
    "sparse_y_single": "test_sparse_y_single.csv",
    "sparse_x_cluster": "test_sparse_x_cluster.csv",
    "sparse_y_cluster": "test_sparse_y_cluster.csv",
}

FAMILY_ALIASES = {
    "tradition": "tradition",
    "traditional": "tradition",
}

DEFAULT_COLORS = [
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AB",
]

PERFORMANCE_COLUMNS = {"El(%)", "UTS(MPa)", "YS(MPa)"}
EXPLICIT_METADATA_COLUMNS = {"Number"}
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


@dataclass(frozen=True)
class CaseSpec:
    alloy_family: str
    dataset: str
    target: str


@dataclass(frozen=True)
class MethodTracePaths:
    method: str
    feature_path: Path
    metadata_path: Path
    selection_paths: tuple[Path, ...]


@dataclass
class PreparedMethodData:
    method: str
    display_name: str
    test_points: pd.DataFrame
    group_label: str
    unique_group_count: int
    point_count: int


@dataclass(frozen=True)
class EmbeddingFeatureSpec:
    column_name: str
    role: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render shared-embedding 7-panel OOD 2D maps."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Root directory containing OOD result folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for shared-embedding plots.",
    )
    parser.add_argument(
        "--experiment-prefix",
        default="experiment1_all_ml_models",
        help="Experiment prefix before method suffixes.",
    )
    parser.add_argument(
        "--family",
        default="tradition",
        help="Model-family subdirectory to read. v1 supports tradition only.",
    )
    parser.add_argument("--alloy-family", help="Alloy family filter for a single case or batch scan.")
    parser.add_argument("--dataset", help="Dataset filter for a single case or batch scan.")
    parser.add_argument("--target", help="Target/property filter for a single case or batch scan.")
    parser.add_argument(
        "--all-cases",
        action="store_true",
        help="Scan all cases under the experiment prefix instead of a single case.",
    )
    parser.add_argument(
        "--embed-method",
        choices=["tsne", "umap"],
        default="tsne",
        help="Shared embedding method.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="Requested t-SNE perplexity; automatically capped to a legal value.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for shared embedding.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="Figure DPI.",
    )
    parser.add_argument(
        "--formats",
        default="png",
        help="Comma-separated output formats.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs instead of skipping completed cases.",
    )
    return parser.parse_args()


def normalize_family(family: str) -> str:
    normalized = str(family).strip().lower()
    if normalized not in FAMILY_ALIASES:
        supported = ", ".join(sorted(set(FAMILY_ALIASES.values())))
        raise ValueError(f"Unsupported --family={family!r}. v1 supports: {supported}")
    return FAMILY_ALIASES[normalized]


def parse_formats(raw_formats: str) -> list[str]:
    formats = []
    for fmt in str(raw_formats).split(","):
        normalized = fmt.strip().lower()
        if not normalized:
            continue
        if normalized not in {"png", "pdf"}:
            raise ValueError(f"Unsupported output format: {fmt}")
        formats.append(normalized)
    if not formats:
        raise ValueError("At least one output format is required.")
    return formats


def safe_name(text: str) -> str:
    return (
        str(text)
        .replace("(", "")
        .replace(")", "")
        .replace("%", "pct")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "")
    )


def is_id_like_column(column_name: str) -> bool:
    normalized = str(column_name).strip().lower()
    return normalized == "id" or normalized.endswith("_id")


def classify_embedding_role(column_name: str) -> str | None:
    raw = str(column_name).strip()
    normalized = raw.lower()

    if raw in PERFORMANCE_COLUMNS or raw in EXPLICIT_METADATA_COLUMNS:
        return None
    if raw.endswith(COMPOSITION_SUFFIXES):
        return "composition"
    if normalized == "cr(%)":
        return "process"
    if any(pattern.match(raw) for pattern in PROCESS_REGEXES):
        return "process"
    if any(keyword in normalized for keyword in PROCESS_KEYWORDS):
        return "process"
    return None


def build_embedding_feature_specs(
    df: pd.DataFrame,
    target_column: str,
) -> list[EmbeddingFeatureSpec]:
    excluded = {"__row_id__", "__source_index__", target_column}
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    feature_specs: list[EmbeddingFeatureSpec] = []
    unknown_columns: list[str] = []
    for column in numeric_columns:
        if column in excluded or is_id_like_column(column):
            continue

        role = classify_embedding_role(column)
        if role is None:
            if column in PERFORMANCE_COLUMNS or column in EXPLICIT_METADATA_COLUMNS:
                continue
            unknown_columns.append(column)
            continue
        feature_specs.append(EmbeddingFeatureSpec(column_name=column, role=role))

    if unknown_columns:
        joined = ", ".join(unknown_columns)
        raise ValueError(
            "Unclassified numeric columns remained after filtering for embedding-only "
            f"composition/process features: {joined}"
        )
    if not feature_specs:
        raise ValueError("No composition/process feature columns remained for shared embedding.")
    return feature_specs


def method_root(base_dir: Path, experiment_prefix: str, method: str) -> Path:
    suffix = METHOD_DIR_SUFFIX[method]
    return base_dir / f"{experiment_prefix}_{suffix}"


def resolve_alloy_family_dir_name(root: Path, alloy_family: str) -> str:
    candidates = ALLOY_FAMILY_PATH_ALIASES.get(alloy_family, (alloy_family,))
    for candidate in candidates:
        if (root / candidate).exists():
            return candidate
    return alloy_family


def single_split_root(root: Path, case: CaseSpec, family: str, method: str) -> Path:
    if method == "target_extrapolation":
        method_dir = "target_extrapolation"
    else:
        method_dir = method
    alloy_dir_name = resolve_alloy_family_dir_name(root, case.alloy_family)
    return root / alloy_dir_name / case.dataset / case.target / method_dir / family


def fold_split_root(root: Path, case: CaseSpec, family: str, method: str) -> Path:
    method_dir = "random_cv_baseline" if method == "random_cv_baseline" else "loco"
    alloy_dir_name = resolve_alloy_family_dir_name(root, case.alloy_family)
    return root / alloy_dir_name / case.dataset / case.target / method_dir / family / "folds"


def discover_cases(
    base_dir: Path,
    experiment_prefix: str,
    family: str,
    *,
    alloy_family: str | None = None,
    dataset: str | None = None,
    target: str | None = None,
) -> list[CaseSpec]:
    canonical_root = method_root(base_dir, experiment_prefix, "target_extrapolation")
    cases: list[CaseSpec] = []
    if not canonical_root.exists():
        return cases

    for alloy_dir in sorted(path for path in canonical_root.iterdir() if path.is_dir()):
        if alloy_family and alloy_dir.name != alloy_family:
            continue
        for dataset_dir in sorted(path for path in alloy_dir.iterdir() if path.is_dir()):
            if dataset and dataset_dir.name != dataset:
                continue
            for target_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
                if target and target_dir.name != target:
                    continue
                family_dir = target_dir / "target_extrapolation" / family
                if family_dir.exists():
                    cases.append(
                        CaseSpec(
                            alloy_family=alloy_dir.name,
                            dataset=dataset_dir.name,
                            target=target_dir.name,
                        )
                    )
    return cases


def resolve_single_case(args: argparse.Namespace, family: str) -> list[CaseSpec]:
    if args.all_cases:
        return discover_cases(
            args.base_dir,
            args.experiment_prefix,
            family,
            alloy_family=args.alloy_family,
            dataset=args.dataset,
            target=args.target,
        )

    missing = [
        flag
        for flag, value in (
            ("--alloy-family", args.alloy_family),
            ("--dataset", args.dataset),
            ("--target", args.target),
        )
        if not value
    ]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            f"Single-case mode requires {missing_text}. "
            f"Use --all-cases to scan automatically."
        )
    return [CaseSpec(args.alloy_family, args.dataset, args.target)]


def build_method_trace_paths(
    base_dir: Path,
    experiment_prefix: str,
    family: str,
    case: CaseSpec,
) -> dict[str, MethodTracePaths]:
    mapping: dict[str, MethodTracePaths] = {}
    for method in METHOD_LAYOUT:
        root = method_root(base_dir, experiment_prefix, method)
        if method in {"random_cv_baseline", "loco"}:
            folds_root = fold_split_root(root, case, family, method)
            selection_paths = []
            fold_dirs = sorted(path for path in folds_root.iterdir() if path.is_dir()) if folds_root.exists() else []
            for fold_dir in fold_dirs:
                selection_path = fold_dir / "split_data" / "trace" / "selected_test_rows.csv"
                if selection_path.exists():
                    selection_paths.append(selection_path)
            feature_path = (
                fold_dirs[0] / "split_data" / "trace" / "x_space_features.parquet"
                if fold_dirs
                else Path()
            )
            metadata_path = (
                fold_dirs[0] / "split_data" / "split_summary.json"
                if fold_dirs
                else Path()
            )
            mapping[method] = MethodTracePaths(
                method=method,
                feature_path=feature_path,
                metadata_path=metadata_path,
                selection_paths=tuple(selection_paths),
            )
        else:
            split_root = single_split_root(root, case, family, method)
            mapping[method] = MethodTracePaths(
                method=method,
                feature_path=split_root / "split_data" / "trace" / "x_space_features.parquet",
                metadata_path=split_root / "split_data" / "split_summary.json",
                selection_paths=(split_root / "split_data" / "trace" / "selected_test_rows.csv",),
            )
    return mapping


def ensure_case_assets(paths: dict[str, MethodTracePaths]) -> None:
    missing: list[str] = []
    for method, trace_paths in paths.items():
        if not trace_paths.feature_path.exists():
            missing.append(f"{method}: missing feature parquet -> {trace_paths.feature_path}")
        if not trace_paths.metadata_path.exists():
            missing.append(f"{method}: missing split_summary.json -> {trace_paths.metadata_path}")
        if not trace_paths.selection_paths:
            missing.append(f"{method}: no selected_test_rows.csv found")
        else:
            for path in trace_paths.selection_paths:
                if not path.exists():
                    missing.append(f"{method}: missing selection file -> {path}")
    if missing:
        raise FileNotFoundError("; ".join(missing))


def feature_columns_for_embedding(
    df: pd.DataFrame,
    target_column: str,
    feature_specs: list[EmbeddingFeatureSpec] | None = None,
) -> list[str]:
    resolved_specs = feature_specs or build_embedding_feature_specs(df, target_column)
    feature_columns = [spec.column_name for spec in resolved_specs]
    if not feature_columns:
        raise ValueError("No numeric feature columns remained for composition/process embedding.")
    return feature_columns


def sort_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "__row_id__" not in df.columns:
        raise ValueError("Feature parquet is missing __row_id__.")
    return df.sort_values("__row_id__").reset_index(drop=True)


def validate_feature_alignment(
    canonical_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    *,
    method: str,
) -> None:
    canonical_sorted = sort_feature_frame(canonical_df)
    candidate_sorted = sort_feature_frame(candidate_df)

    canonical_ids = canonical_sorted["__row_id__"].tolist()
    candidate_ids = candidate_sorted["__row_id__"].tolist()
    if canonical_ids != candidate_ids:
        raise ValueError(f"{method}: __row_id__ sequence does not match canonical source.")

    if list(canonical_sorted.columns) != list(candidate_sorted.columns):
        raise ValueError(f"{method}: x_space_features.parquet column layout differs from canonical source.")

    for column in canonical_sorted.columns:
        left = canonical_sorted[column]
        right = candidate_sorted[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            if not np.allclose(
                left.to_numpy(dtype=float),
                right.to_numpy(dtype=float),
                equal_nan=True,
                rtol=1e-6,
                atol=1e-8,
            ):
                raise ValueError(f"{method}: numeric feature column mismatch detected in {column}.")
        else:
            if not left.astype(str).equals(right.astype(str)):
                raise ValueError(f"{method}: non-numeric feature column mismatch detected in {column}.")


def resolve_tsne_perplexity(requested: float, sample_count: int) -> float:
    if sample_count <= 1:
        raise ValueError("At least two samples are required to compute a shared 2D embedding.")
    legal_max = max(1.0, float(sample_count - 1))
    perplexity = min(float(requested), legal_max)
    return max(1.0, perplexity)


def compute_shared_embedding(
    features_df: pd.DataFrame,
    *,
    target_column: str,
    feature_specs: list[EmbeddingFeatureSpec] | None = None,
    embed_method: str,
    perplexity: float,
    random_state: int,
) -> pd.DataFrame:
    working_df = sort_feature_frame(features_df)
    feature_columns = feature_columns_for_embedding(
        working_df,
        target_column,
        feature_specs=feature_specs,
    )
    feature_matrix = working_df[feature_columns].to_numpy(dtype=float)
    feature_matrix = StandardScaler().fit_transform(feature_matrix)

    pca_components = min(20, feature_matrix.shape[1], max(1, feature_matrix.shape[0] - 1))
    reduced_matrix = PCA(n_components=pca_components, random_state=random_state).fit_transform(feature_matrix)

    if embed_method == "tsne":
        effective_perplexity = resolve_tsne_perplexity(perplexity, reduced_matrix.shape[0])
        embedder = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=effective_perplexity,
            random_state=random_state,
        )
        embedding = embedder.fit_transform(reduced_matrix)
    elif embed_method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise ImportError(
                "UMAP support requires umap-learn. Install it or use --embed-method tsne."
            ) from exc
        embedder = umap.UMAP(
            n_components=2,
            init="spectral",
            random_state=random_state,
        )
        embedding = embedder.fit_transform(reduced_matrix)
    else:
        raise ValueError(f"Unsupported embed method: {embed_method}")

    shared_embedding = pd.DataFrame(
        {
            "__row_id__": working_df["__row_id__"].to_numpy(),
            "x": embedding[:, 0],
            "y": embedding[:, 1],
        }
    )
    return shared_embedding


def fold_key_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("fold_"):
            return part
    return path.stem


def prepare_method_data(
    method: str,
    trace_paths: MethodTracePaths,
    shared_embedding: pd.DataFrame,
) -> PreparedMethodData:
    selection_frames: list[pd.DataFrame] = []
    for selection_path in trace_paths.selection_paths:
        selection_df = pd.read_csv(selection_path)
        if "__row_id__" not in selection_df.columns:
            raise ValueError(f"{method}: selection file missing __row_id__ -> {selection_path}")

        fold_key = fold_key_from_path(selection_path)
        selection_df["__fold_key__"] = fold_key

        if method == "random_cv_baseline":
            group_column = "outer_fold_index"
            if group_column not in selection_df.columns:
                selection_df[group_column] = selection_df["__fold_key__"].str.replace("fold_", "", regex=False)
        elif method == "loco":
            group_column = "held_out_cluster_id"
            if group_column not in selection_df.columns:
                group_column = "fold_index"
                selection_df[group_column] = selection_df["__fold_key__"].str.replace("fold_", "", regex=False)
        elif method in {"sparse_x_single", "sparse_y_single"}:
            if "cluster_id" in selection_df.columns:
                group_column = "cluster_id"
            elif "seed_row_id" in selection_df.columns:
                group_column = "seed_row_id"
            else:
                group_column = "__single_group__"
                selection_df[group_column] = "test"
        elif method in {"sparse_x_cluster", "sparse_y_cluster"}:
            if "cluster_id" in selection_df.columns:
                group_column = "cluster_id"
            else:
                group_column = "__cluster_group__"
                selection_df[group_column] = "test"
        else:
            group_column = "__extrapolation_group__"
            selection_df[group_column] = "test"

        selection_df["__group_value__"] = selection_df[group_column].astype(str)
        selection_frames.append(selection_df)

    if not selection_frames:
        raise ValueError(f"{method}: no test selection frames were loaded.")

    combined = pd.concat(selection_frames, ignore_index=True)
    combined = combined.merge(shared_embedding, on="__row_id__", how="left", validate="many_to_one")
    if combined[["x", "y"]].isna().any().any():
        raise ValueError(f"{method}: failed to align test rows onto shared embedding.")

    return PreparedMethodData(
        method=method,
        display_name=METHOD_ALIASES[method],
        test_points=combined,
        group_label=group_column,
        unique_group_count=int(combined["__group_value__"].nunique(dropna=False)),
        point_count=int(len(combined)),
    )


def palette_for_groups(values: Iterable[str]) -> dict[str, str]:
    unique_values = [str(value) for value in pd.Index(values).astype(str).drop_duplicates()]
    if not unique_values:
        return {}
    cmap = plt.get_cmap("tab20")
    if len(unique_values) <= len(DEFAULT_COLORS):
        colors = [DEFAULT_COLORS[index % len(DEFAULT_COLORS)] for index in range(len(unique_values))]
    else:
        colors = [cmap(index / max(1, len(unique_values) - 1)) for index in range(len(unique_values))]
    return dict(zip(unique_values, colors))


def method_note_text(prepared: list[PreparedMethodData]) -> str:
    lines = [
        "Shared 2D embedding",
        "Grey: all samples",
        "Colour +: test samples",
        "",
    ]
    for item in prepared:
        if item.method == "random_cv_baseline":
            detail = "colour by outer fold"
        elif item.method == "loco":
            detail = "colour by held-out cluster"
        elif item.method in {"sparse_x_single", "sparse_y_single"}:
            detail = "colour by cluster/seed"
        elif item.method in {"sparse_x_cluster", "sparse_y_cluster"}:
            detail = "colour by cluster"
        else:
            detail = "single test group"
        lines.append(f"{item.display_name}: n={item.point_count}, {detail}")
    return "\n".join(lines)


def ensure_output_dirs(case_output_dir: Path, formats: Iterable[str]) -> tuple[Path, dict[str, Path]]:
    csv_dir = case_output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    format_dirs: dict[str, Path] = {}
    for fmt in formats:
        format_dir = case_output_dir / fmt
        format_dir.mkdir(parents=True, exist_ok=True)
        format_dirs[fmt] = format_dir
    return csv_dir, format_dirs


def save_case_csv_artifacts(
    *,
    canonical_df: pd.DataFrame,
    shared_embedding: pd.DataFrame,
    embedding_feature_specs: list[EmbeddingFeatureSpec],
    prepared_methods: list[PreparedMethodData],
    method_paths: dict[str, MethodTracePaths],
    csv_dir: Path,
) -> dict[str, Path]:
    written_paths: dict[str, Path] = {}

    background_points = (
        sort_feature_frame(canonical_df)
        .merge(shared_embedding, on="__row_id__", how="left", validate="one_to_one")
    )
    background_path = csv_dir / "background_points.csv"
    background_points.to_csv(background_path, index=False, encoding="utf-8-sig")
    written_paths["background_points"] = background_path

    shared_embedding_path = csv_dir / "shared_embedding.csv"
    shared_embedding.sort_values("__row_id__").to_csv(shared_embedding_path, index=False, encoding="utf-8-sig")
    written_paths["shared_embedding"] = shared_embedding_path

    embedding_feature_columns = pd.DataFrame(
        [
            {
                "column_order": index,
                "column_name": spec.column_name,
                "role": spec.role,
            }
            for index, spec in enumerate(embedding_feature_specs, start=1)
        ]
    )
    embedding_feature_columns_path = csv_dir / "embedding_feature_columns.csv"
    embedding_feature_columns.to_csv(embedding_feature_columns_path, index=False, encoding="utf-8-sig")
    written_paths["embedding_feature_columns"] = embedding_feature_columns_path

    panel_summary = pd.DataFrame(
        [
            {
                "method": item.method,
                "display_name": item.display_name,
                "group_label": item.group_label,
                "unique_group_count": item.unique_group_count,
                "point_count": item.point_count,
            }
            for item in prepared_methods
        ]
    )
    panel_summary_path = csv_dir / "panel_summary.csv"
    panel_summary.to_csv(panel_summary_path, index=False, encoding="utf-8-sig")
    written_paths["panel_summary"] = panel_summary_path

    source_manifest_rows: list[dict[str, object]] = []
    all_panel_frames: list[pd.DataFrame] = []
    for item in prepared_methods:
        method_df = item.test_points.copy()
        method_df.insert(0, "method", item.method)
        method_df.insert(1, "display_name", item.display_name)
        method_df.insert(2, "group_label", item.group_label)
        all_panel_frames.append(method_df)

        method_path = csv_dir / f"{item.method}_test_points.csv"
        method_df.to_csv(method_path, index=False, encoding="utf-8-sig")
        written_paths[f"{item.method}_test_points"] = method_path

        trace_paths = method_paths[item.method]
        source_manifest_rows.append(
            {
                "method": item.method,
                "display_name": item.display_name,
                "feature_path": str(trace_paths.feature_path.resolve()),
                "metadata_path": str(trace_paths.metadata_path.resolve()),
                "selection_paths": " | ".join(str(path.resolve()) for path in trace_paths.selection_paths),
            }
        )

    all_panel_test_points = pd.concat(all_panel_frames, ignore_index=True)
    all_panel_path = csv_dir / "all_panel_test_points.csv"
    all_panel_test_points.to_csv(all_panel_path, index=False, encoding="utf-8-sig")
    written_paths["all_panel_test_points"] = all_panel_path

    source_manifest = pd.DataFrame(source_manifest_rows)
    source_manifest_path = csv_dir / "source_manifest.csv"
    source_manifest.to_csv(source_manifest_path, index=False, encoding="utf-8-sig")
    written_paths["source_manifest"] = source_manifest_path

    return written_paths


def plot_case(
    *,
    case: CaseSpec,
    shared_embedding: pd.DataFrame,
    prepared_methods: list[PreparedMethodData],
    format_dirs: dict[str, Path],
    embed_method: str,
    dpi: int,
    formats: list[str],
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(17.2, 8.8), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    x_margin = (shared_embedding["x"].max() - shared_embedding["x"].min()) * 0.04
    y_margin = (shared_embedding["y"].max() - shared_embedding["y"].min()) * 0.04
    x_limits = (shared_embedding["x"].min() - x_margin, shared_embedding["x"].max() + x_margin)
    y_limits = (shared_embedding["y"].min() - y_margin, shared_embedding["y"].max() + y_margin)

    for index, method_data in enumerate(prepared_methods):
        ax = axes_flat[index]
        ax.scatter(
            shared_embedding["x"],
            shared_embedding["y"],
            s=9,
            c="#D3D3D3",
            alpha=0.65,
            linewidths=0.0,
            marker="o",
            zorder=1,
        )

        palette = palette_for_groups(method_data.test_points["__group_value__"])
        for group_value, group_df in method_data.test_points.groupby("__group_value__", sort=True):
            ax.scatter(
                group_df["x"],
                group_df["y"],
                s=70,
                c=[palette[str(group_value)]],
                marker="+",
                linewidths=1.3,
                alpha=0.95,
                zorder=2,
            )

        ax.set_title(method_data.display_name, fontsize=12.5)
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_xlabel("Shared 2D-1", fontsize=10)
        ax.set_ylabel("Shared 2D-2", fontsize=10)
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(False)

    legend_ax = axes_flat[-1]
    legend_ax.axis("off")
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=6, markerfacecolor="#D3D3D3", markeredgecolor="#D3D3D3", alpha=0.8, label="All samples"),
        Line2D([0], [0], marker="+", linestyle="", markersize=10, color="#4E79A7", markeredgewidth=1.4, label="Test samples"),
    ]
    legend_ax.legend(handles=legend_handles, loc="upper left", frameon=True, fontsize=10)
    legend_ax.text(
        0.02,
        0.72,
        method_note_text(prepared_methods),
        ha="left",
        va="top",
        fontsize=9.3,
        linespacing=1.35,
        family="monospace",
        transform=legend_ax.transAxes,
    )

    title = f"{case.alloy_family} | {case.dataset} | {case.target} | shared {embed_method.upper()} map"
    fig.suptitle(title, fontsize=15, y=0.985)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965], pad=0.9)

    stem = f"ood_2d_map_{embed_method.lower()}"
    for output_format in formats:
        output_path = format_dirs[output_format] / f"{stem}.{output_format}"
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")

    plt.close(fig)


def render_case(
    *,
    case: CaseSpec,
    base_dir: Path,
    output_root: Path,
    experiment_prefix: str,
    family: str,
    embed_method: str,
    perplexity: float,
    random_state: int,
    dpi: int,
    formats: list[str],
    overwrite: bool,
) -> dict[str, object]:
    output_dir = output_root / experiment_prefix / case.alloy_family / case.dataset / case.target
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir, format_dirs = ensure_output_dirs(output_dir, formats)
    shared_embedding_path = csv_dir / "shared_embedding.csv"
    embedding_feature_columns_path = csv_dir / "embedding_feature_columns.csv"
    plot_paths = [format_dirs[fmt] / f"ood_2d_map_{embed_method}.{fmt}" for fmt in formats]

    if (
        not overwrite
        and shared_embedding_path.exists()
        and embedding_feature_columns_path.exists()
        and all(path.exists() for path in plot_paths)
    ):
        print(f"[SKIP] {case.alloy_family}/{case.dataset}/{case.target}: outputs already exist.")
        return {
            "alloy_family": case.alloy_family,
            "dataset": case.dataset,
            "target": case.target,
            "status": "skipped",
            "output_dir": str(output_dir.resolve()),
            "message": "outputs already existed",
        }

    method_paths = build_method_trace_paths(base_dir, experiment_prefix, family, case)
    ensure_case_assets(method_paths)

    canonical_method = "target_extrapolation"
    canonical_df = pd.read_parquet(method_paths[canonical_method].feature_path)
    target_column = case.target
    embedding_feature_specs = build_embedding_feature_specs(canonical_df, target_column)

    for method, trace_paths in method_paths.items():
        candidate_df = pd.read_parquet(trace_paths.feature_path)
        validate_feature_alignment(canonical_df, candidate_df, method=method)

    shared_embedding = compute_shared_embedding(
        canonical_df,
        target_column=target_column,
        feature_specs=embedding_feature_specs,
        embed_method=embed_method,
        perplexity=perplexity,
        random_state=random_state,
    )

    prepared_methods = [
        prepare_method_data(method, method_paths[method], shared_embedding)
        for method in METHOD_LAYOUT
    ]
    written_paths = save_case_csv_artifacts(
        canonical_df=canonical_df,
        shared_embedding=shared_embedding,
        embedding_feature_specs=embedding_feature_specs,
        prepared_methods=prepared_methods,
        method_paths=method_paths,
        csv_dir=csv_dir,
    )
    plot_case(
        case=case,
        shared_embedding=shared_embedding,
        prepared_methods=prepared_methods,
        format_dirs=format_dirs,
        embed_method=embed_method,
        dpi=dpi,
        formats=formats,
    )

    count_map = {item.display_name: item.point_count for item in prepared_methods}
    print(
        "[OK] "
        f"{case.alloy_family}/{case.dataset}/{case.target}: "
        + ", ".join(f"{name}={count}" for name, count in count_map.items())
    )
    return {
        "alloy_family": case.alloy_family,
        "dataset": case.dataset,
        "target": case.target,
        "status": "ok",
        "output_dir": str(output_dir.resolve()),
        "csv_dir": str(csv_dir.resolve()),
        "png_dir": str((output_dir / "png").resolve()),
        "shared_embedding": str(shared_embedding_path.resolve()),
        "embedding_feature_columns": str(written_paths["embedding_feature_columns"].resolve()),
        "background_points": str(written_paths["background_points"].resolve()),
        "all_panel_test_points": str(written_paths["all_panel_test_points"].resolve()),
        "panel_summary": str(written_paths["panel_summary"].resolve()),
        "random_cv_count": count_map.get("RandomCV", 0),
        "loco_count": count_map.get("LOCO", 0),
        "extrapolation_count": count_map.get("Extra", 0),
        "sparse_x_single_count": count_map.get("SparseX-single", 0),
        "sparse_y_single_count": count_map.get("SparseY-single", 0),
        "sparse_x_cluster_count": count_map.get("SparseX-cluster", 0),
        "sparse_y_cluster_count": count_map.get("SparseY-cluster", 0),
        "message": "",
    }


def write_manifest(
    *,
    records: list[dict[str, object]],
    output_root: Path,
    experiment_prefix: str,
) -> Path:
    manifest_dir = output_root / experiment_prefix / "csv"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "render_manifest.csv"
    pd.DataFrame(records).to_csv(manifest_path, index=False, encoding="utf-8-sig")
    return manifest_path


def main() -> None:
    args = parse_args()
    family = normalize_family(args.family)
    formats = parse_formats(args.formats)
    cases = resolve_single_case(args, family)
    if not cases:
        raise FileNotFoundError("No matching cases were found for the provided filters.")

    records: list[dict[str, object]] = []
    for case in cases:
        try:
            record = render_case(
                case=case,
                base_dir=args.base_dir,
                output_root=args.output_root,
                experiment_prefix=args.experiment_prefix,
                family=family,
                embed_method=args.embed_method,
                perplexity=args.perplexity,
                random_state=args.random_state,
                dpi=args.dpi,
                formats=formats,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            output_dir = args.output_root / args.experiment_prefix / case.alloy_family / case.dataset / case.target
            print(f"[ERROR] {case.alloy_family}/{case.dataset}/{case.target}: {exc}")
            record = {
                "alloy_family": case.alloy_family,
                "dataset": case.dataset,
                "target": case.target,
                "status": "error",
                "output_dir": str(output_dir.resolve()),
                "message": str(exc),
            }
        records.append(record)

    manifest_path = write_manifest(
        records=records,
        output_root=args.output_root,
        experiment_prefix=args.experiment_prefix,
    )
    ok_count = sum(1 for record in records if record.get("status") == "ok")
    error_count = sum(1 for record in records if record.get("status") == "error")
    skipped_count = sum(1 for record in records if record.get("status") == "skipped")
    print(
        f"[DONE] total={len(records)}, ok={ok_count}, skipped={skipped_count}, error={error_count}, "
        f"manifest={manifest_path.resolve()}"
    )


if __name__ == "__main__":
    main()
