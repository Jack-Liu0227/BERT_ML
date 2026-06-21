from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


DEFAULT_SPLIT_SUMMARY = Path("output") / "ood_xspace_scores" / "ood_split_summary.csv"
DEFAULT_OUTPUT_ROOT = Path("output") / "ood_xspace_scores" / "wx_v2_mixed_ot"
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
SPLIT_SUMMARY_COLUMNS = [
    "alloy_family",
    "dataset_name",
    "property",
    "method",
    "split_id",
    "fold_id",
    "source_split_dir",
    "train_file",
    "test_file",
    "train_size",
    "test_size",
    "feature_count",
    "wx_v2",
    "wx_v2_solver",
    "wx_v2_status",
    "wx_v2_warning",
]
SAMPLE_COLUMNS = [
    "alloy_family",
    "dataset_name",
    "property",
    "method",
    "split_id",
    "fold_id",
    "source_split_dir",
    "ID",
    "__row_id__",
    "__source_index__",
    "target_col",
    "target_value",
    "wx_v2_sample_score",
    "wx_v2_mass_contribution",
    "wx_v2_rank_desc",
]
FEATURE_COLUMNS = [
    "alloy_family",
    "dataset_name",
    "property",
    "method",
    "split_id",
    "fold_id",
    "source_split_dir",
    "ID",
    "__row_id__",
    "__source_index__",
    "target_col",
    "target_value",
    "feature",
    "feature_role",
    "train_value_matched_mean",
    "test_value",
    "feature_scale",
    "wx_v2_feature_score",
    "wx_v2_feature_fraction",
    "presence_mismatch_fraction",
]
FAILURE_COLUMNS = [
    "alloy_family",
    "dataset_name",
    "property",
    "method",
    "split_id",
    "fold_id",
    "source_split_dir",
    "error",
]


@dataclass
class MixedDistanceComponents:
    cost_matrix: np.ndarray
    feature_distances: dict[str, np.ndarray]
    feature_presence_mismatches: dict[str, np.ndarray]
    feature_scales: dict[str, float]


@dataclass
class ExactOTResult:
    wx_v2: float
    plan: np.ndarray
    sample_scores: np.ndarray
    sample_mass_contributions: np.ndarray
    solver_message: str


@dataclass
class WxV2Result:
    split_row: dict[str, Any]
    sample_scores: pd.DataFrame
    feature_contributions: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute parallel Wx_v2 scores with bounded mixed feature distances "
            "and exact optimal transport."
        )
    )
    parser.add_argument("--split-summary", default=str(DEFAULT_SPLIT_SUMMARY))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--case-contains", default=None, help="Filter by text in case path or source split directory.")
    parser.add_argument("--method", action="append", default=None, help="Filter by split method; can be repeated.")
    parser.add_argument("--fold-id", default=None, help="Filter by fold id, for example fold_3.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of matching splits to score.")
    parser.add_argument("--strict", action="store_true", help="Raise immediately on the first split failure.")
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value)
    return "" if text.lower() == "nan" else text


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path)


def write_csv(frame: pd.DataFrame, path: Path, columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    if columns is not None:
        for column in columns:
            if column not in output.columns:
                output[column] = np.nan
        output = output[columns + [column for column in output.columns if column not in columns]]
    output.to_csv(path, index=False, encoding="utf-8-sig")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def positive_train_scale(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    positive = numeric[np.isfinite(numeric) & (numeric > 0)].astype(float)
    if positive.empty:
        return 1.0

    q25, q75 = np.percentile(positive.to_numpy(dtype=float), [25, 75])
    scale = float(q75 - q25)
    if math.isfinite(scale) and scale > 1e-12:
        return scale

    value_range = float(positive.max() - positive.min())
    if math.isfinite(value_range) and value_range > 1e-12:
        return value_range

    median = float(positive.median())
    fallback = max(abs(median), 1.0)
    return fallback if math.isfinite(fallback) and fallback > 0 else 1.0


def numeric_feature_frame(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    for column in feature_columns:
        if column in frame.columns:
            result[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            result[column] = np.nan
    return result


def feature_role(feature: str, configured_roles: dict[str, Any] | None = None) -> str:
    if configured_roles and feature in configured_roles and clean_text(configured_roles[feature]):
        return clean_text(configured_roles[feature])
    raw = str(feature).strip()
    normalized = raw.lower()
    if raw.endswith(COMPOSITION_SUFFIXES):
        return "composition"
    if normalized == "cr(%)":
        return "process"
    if any(pattern.match(raw) for pattern in PROCESS_REGEXES):
        return "process"
    if any(keyword in normalized for keyword in PROCESS_KEYWORDS):
        return "process"
    return "x_feature"


def build_mixed_distance_components(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    feature_columns: list[str],
) -> MixedDistanceComponents:
    if not feature_columns:
        raise ValueError("No X-space feature columns available for Wx_v2.")
    if train_x.empty or test_x.empty:
        raise ValueError("Train or test feature frame is empty.")

    train = numeric_feature_frame(train_x, feature_columns)
    test = numeric_feature_frame(test_x, feature_columns)
    train_count = len(train)
    test_count = len(test)
    feature_count = len(feature_columns)

    cost_matrix = np.zeros((train_count, test_count), dtype=float)
    feature_distances: dict[str, np.ndarray] = {}
    feature_presence_mismatches: dict[str, np.ndarray] = {}
    feature_scales: dict[str, float] = {}

    for feature in feature_columns:
        train_values = pd.to_numeric(train[feature], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        test_values = pd.to_numeric(test[feature], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        train_present = train_values > 0
        test_present = test_values > 0
        scale = positive_train_scale(pd.Series(train_values))

        train_present_matrix = train_present[:, None]
        test_present_matrix = test_present[None, :]
        both_absent = ~train_present_matrix & ~test_present_matrix
        both_present = train_present_matrix & test_present_matrix
        presence_mismatch = train_present_matrix ^ test_present_matrix

        distances = np.zeros((train_count, test_count), dtype=float)
        distances[presence_mismatch] = 1.0
        if both_present.any():
            value_distance = np.abs(train_values[:, None] - test_values[None, :]) / scale
            distances[both_present] = np.minimum(value_distance[both_present], 1.0)
        distances[both_absent] = 0.0

        feature_distances[feature] = distances
        feature_presence_mismatches[feature] = presence_mismatch.astype(float)
        feature_scales[feature] = float(scale)
        cost_matrix += distances / feature_count

    if not np.isfinite(cost_matrix).all():
        raise ValueError("Wx_v2 cost matrix contains non-finite values.")
    return MixedDistanceComponents(
        cost_matrix=cost_matrix,
        feature_distances=feature_distances,
        feature_presence_mismatches=feature_presence_mismatches,
        feature_scales=feature_scales,
    )


def solve_exact_ot(cost_matrix: np.ndarray) -> ExactOTResult:
    costs = np.asarray(cost_matrix, dtype=float)
    if costs.ndim != 2 or costs.size == 0:
        raise ValueError("Cost matrix must be a non-empty 2D array.")
    if not np.isfinite(costs).all():
        raise ValueError("Cost matrix contains non-finite values.")

    train_count, test_count = costs.shape
    flat_costs = costs.ravel()
    row_constraints = sparse.block_diag((np.ones((1, test_count)),) * train_count, format="csr")
    col_constraints = sparse.hstack((sparse.eye(test_count, format="csr"),) * train_count, format="csr")
    constraints = sparse.vstack((row_constraints, col_constraints), format="csr")
    masses = np.concatenate(
        [np.full(train_count, 1.0 / train_count), np.full(test_count, 1.0 / test_count)]
    )

    result = linprog(flat_costs, A_eq=constraints, b_eq=masses, bounds=(0, None), method="highs")
    if not result.success:
        raise RuntimeError(f"Exact OT failed: {result.message}")

    plan = np.asarray(result.x, dtype=float).reshape(train_count, test_count)
    sample_mass_contributions = (plan * costs).sum(axis=0)
    sample_mass_contributions[np.abs(sample_mass_contributions) < 1e-15] = 0.0
    sample_scores = sample_mass_contributions * test_count
    wx_v2 = float(result.fun)
    return ExactOTResult(
        wx_v2=wx_v2,
        plan=plan,
        sample_scores=sample_scores,
        sample_mass_contributions=sample_mass_contributions,
        solver_message=str(result.message),
    )


def base_sample_frame(test_df: pd.DataFrame, target_col: str, metadata: dict[str, Any]) -> pd.DataFrame:
    row_count = len(test_df)
    result = pd.DataFrame(
        {
            "alloy_family": metadata.get("alloy_family", ""),
            "dataset_name": metadata.get("dataset_name", ""),
            "property": metadata.get("property", ""),
            "method": metadata.get("method", metadata.get("split_strategy", "")),
            "split_id": metadata.get("split_id", ""),
            "fold_id": metadata.get("fold_id", ""),
            "source_split_dir": metadata.get("source_split_dir", ""),
            "ID": test_df["ID"].to_numpy() if "ID" in test_df.columns else np.full(row_count, np.nan),
            "__row_id__": test_df["__row_id__"].to_numpy() if "__row_id__" in test_df.columns else np.arange(row_count),
            "__source_index__": test_df["__source_index__"].to_numpy()
            if "__source_index__" in test_df.columns
            else np.full(row_count, np.nan),
            "target_col": target_col,
            "target_value": pd.to_numeric(test_df[target_col], errors="coerce").to_numpy()
            if target_col in test_df.columns
            else np.full(row_count, np.nan),
        }
    )
    return result


def build_feature_contributions(
    test_df: pd.DataFrame,
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    target_col: str,
    metadata: dict[str, Any],
    feature_columns: list[str],
    configured_roles: dict[str, Any],
    components: MixedDistanceComponents,
    ot_result: ExactOTResult,
) -> pd.DataFrame:
    base = base_sample_frame(test_df, target_col, metadata)
    test_count = len(test_df)
    feature_count = len(feature_columns)
    sample_scores = ot_result.sample_scores
    rows: list[pd.DataFrame] = []

    for feature in feature_columns:
        distances = components.feature_distances[feature]
        mismatch = components.feature_presence_mismatches[feature]
        feature_mass = (ot_result.plan * (distances / feature_count)).sum(axis=0)
        feature_score = feature_mass * test_count
        feature_fraction = np.divide(
            feature_score,
            sample_scores,
            out=np.zeros_like(feature_score, dtype=float),
            where=np.abs(sample_scores) > 1e-15,
        )

        train_values = pd.to_numeric(train_x[feature], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        test_values = pd.to_numeric(test_x[feature], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        matched_train_mean = (ot_result.plan * train_values[:, None]).sum(axis=0) * test_count
        presence_mismatch_fraction = (ot_result.plan * mismatch).sum(axis=0) * test_count

        frame = base.copy()
        frame["feature"] = feature
        frame["feature_role"] = feature_role(feature, configured_roles)
        frame["train_value_matched_mean"] = matched_train_mean
        frame["test_value"] = test_values
        frame["feature_scale"] = components.feature_scales[feature]
        frame["wx_v2_feature_score"] = feature_score
        frame["wx_v2_feature_fraction"] = feature_fraction
        frame["presence_mismatch_fraction"] = presence_mismatch_fraction
        rows.append(frame)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=FEATURE_COLUMNS)


def compute_wx_v2_for_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    metadata: dict[str, Any] | None = None,
    target_col: str = "",
    configured_roles: dict[str, Any] | None = None,
) -> WxV2Result:
    metadata = dict(metadata or {})
    configured_roles = dict(configured_roles or {})
    train_x = numeric_feature_frame(train_df, feature_columns)
    test_x = numeric_feature_frame(test_df, feature_columns)
    components = build_mixed_distance_components(train_x, test_x, feature_columns)
    ot_result = solve_exact_ot(components.cost_matrix)

    sample_scores = base_sample_frame(test_df, target_col, metadata)
    sample_scores["wx_v2_sample_score"] = ot_result.sample_scores
    sample_scores["wx_v2_mass_contribution"] = ot_result.sample_mass_contributions
    sample_scores["wx_v2_rank_desc"] = (
        sample_scores["wx_v2_sample_score"].rank(method="first", ascending=False).astype(int)
    )

    feature_contributions = build_feature_contributions(
        test_df=test_df,
        train_x=train_x,
        test_x=test_x,
        target_col=target_col,
        metadata=metadata,
        feature_columns=feature_columns,
        configured_roles=configured_roles,
        components=components,
        ot_result=ot_result,
    )

    split_row = {
        "alloy_family": metadata.get("alloy_family", ""),
        "dataset_name": metadata.get("dataset_name", ""),
        "property": metadata.get("property", ""),
        "method": metadata.get("method", metadata.get("split_strategy", "")),
        "split_id": metadata.get("split_id", ""),
        "fold_id": metadata.get("fold_id", ""),
        "source_split_dir": metadata.get("source_split_dir", ""),
        "train_file": metadata.get("train_file", ""),
        "test_file": metadata.get("test_file", ""),
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "feature_count": int(len(feature_columns)),
        "wx_v2": float(ot_result.wx_v2),
        "wx_v2_solver": "scipy_linprog_highs_exact",
        "wx_v2_status": "ok",
        "wx_v2_warning": metadata.get("wx_v2_warning", ""),
    }
    return WxV2Result(split_row=split_row, sample_scores=sample_scores, feature_contributions=feature_contributions)


def load_split_metadata(row: pd.Series) -> tuple[dict[str, Any], list[str], dict[str, Any], str, Path, Path, Path | None]:
    source_split_dir = Path(str(row["source_split_dir"]))
    split_summary_path = source_split_dir / "split_summary.json"
    if not split_summary_path.exists():
        raise FileNotFoundError(f"Missing split summary JSON: {split_summary_path}")
    split_summary = read_json(split_summary_path)
    feature_columns = [str(column) for column in split_summary.get("x_space_feature_columns", []) if str(column)]
    if not feature_columns:
        raise ValueError(f"No x_space_feature_columns in {split_summary_path}")

    target_col = clean_text(row.get("target_col")) or clean_text(
        split_summary.get("split_target_col") or split_summary.get("target_column") or row.get("property")
    )
    train_file = Path(clean_text(row.get("train_file")))
    test_file = Path(clean_text(row.get("test_file")))
    if not train_file.exists():
        train_label = clean_text(split_summary.get("train_label")) or "train"
        train_file = source_split_dir / f"{train_label}.csv"
    if not test_file.exists():
        test_label = clean_text(split_summary.get("test_label")) or "test"
        test_file = source_split_dir / f"{test_label}.csv"
    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(f"Could not resolve train/test files for {source_split_dir}")

    method = clean_text(row.get("split_strategy")) or clean_text(row.get("method"))
    metadata = {
        "alloy_family": clean_text(row.get("alloy_family")),
        "dataset_name": clean_text(row.get("dataset_name")),
        "property": clean_text(row.get("property")),
        "method": method,
        "split_strategy": method,
        "split_id": clean_text(row.get("split_id")),
        "fold_id": clean_text(row.get("fold_id")),
        "source_split_dir": str(source_split_dir),
        "train_file": str(train_file),
        "test_file": str(test_file),
    }
    configured_roles = split_summary.get("x_space_feature_roles") or {}
    output_dir_text = clean_text(row.get("output_dir"))
    output_dir = Path(output_dir_text) if output_dir_text else None
    return metadata, feature_columns, configured_roles, target_col, train_file, test_file, output_dir


def row_matches(row: pd.Series, args: argparse.Namespace) -> bool:
    method = clean_text(row.get("split_strategy")) or clean_text(row.get("method"))
    if args.method and method not in set(args.method):
        return False
    if args.fold_id is not None and clean_text(row.get("fold_id")) != args.fold_id:
        return False
    if args.case_contains:
        case_text = "\\".join(
            [
                clean_text(row.get("alloy_family")),
                clean_text(row.get("dataset_name")),
                clean_text(row.get("property")),
                method,
                clean_text(row.get("fold_id")),
                clean_text(row.get("source_split_dir")),
            ]
        )
        needles = {
            args.case_contains.lower(),
            args.case_contains.replace("/", "\\").lower(),
            args.case_contains.replace("\\", "/").lower(),
        }
        haystacks = {
            case_text.lower(),
            case_text.replace("/", "\\").lower(),
            case_text.replace("\\", "/").lower(),
        }
        if not any(needle in haystack for needle in needles for haystack in haystacks):
            return False
    return True


def iter_matching_rows(split_summary: pd.DataFrame, args: argparse.Namespace) -> Iterable[pd.Series]:
    count = 0
    for _, row in split_summary.iterrows():
        if not row_matches(row, args):
            continue
        yield row
        count += 1
        if args.limit is not None and count >= args.limit:
            break


def failure_row_from(row: pd.Series, error: Exception) -> dict[str, Any]:
    return {
        "alloy_family": clean_text(row.get("alloy_family")),
        "dataset_name": clean_text(row.get("dataset_name")),
        "property": clean_text(row.get("property")),
        "method": clean_text(row.get("split_strategy")) or clean_text(row.get("method")),
        "split_id": clean_text(row.get("split_id")),
        "fold_id": clean_text(row.get("fold_id")),
        "source_split_dir": clean_text(row.get("source_split_dir")),
        "error": repr(error),
    }


def compute_one_split(row: pd.Series) -> tuple[WxV2Result, Path | None]:
    metadata, feature_columns, configured_roles, target_col, train_file, test_file, output_dir = load_split_metadata(row)
    train_df = read_csv(train_file)
    test_df = read_csv(test_file)
    if train_df.empty or test_df.empty:
        raise ValueError("Train or test split is empty.")
    missing = sorted(set(feature_columns) - (set(train_df.columns) | set(test_df.columns)))
    if missing:
        metadata["wx_v2_warning"] = f"feature columns missing from both train and test: {missing[:10]}"
    result = compute_wx_v2_for_frames(
        train_df=train_df,
        test_df=test_df,
        feature_columns=feature_columns,
        metadata=metadata,
        target_col=target_col,
        configured_roles=configured_roles,
    )
    return result, output_dir


def write_split_outputs(result: WxV2Result, output_dir: Path | None) -> None:
    if output_dir is None:
        return
    write_csv(pd.DataFrame([result.split_row]), output_dir / "ood_split_summary_wx_v2.csv", SPLIT_SUMMARY_COLUMNS)
    write_csv(result.sample_scores, output_dir / "ood_sample_scores_wx_v2.csv", SAMPLE_COLUMNS)
    write_csv(result.feature_contributions, output_dir / "ood_feature_contributions_wx_v2.csv", FEATURE_COLUMNS)


def main() -> None:
    args = parse_args()
    split_summary_path = Path(args.split_summary)
    output_root = Path(args.output_root)
    if not split_summary_path.exists():
        raise SystemExit(f"Missing split summary CSV: {split_summary_path}")

    output_root.mkdir(parents=True, exist_ok=True)
    split_summary = read_csv(split_summary_path)
    if "source_split_dir" not in split_summary.columns:
        raise SystemExit(f"{split_summary_path} is missing required column: source_split_dir")

    matching_rows = list(iter_matching_rows(split_summary, args))
    if not matching_rows:
        raise SystemExit("No matching splits found.")

    split_rows: list[dict[str, Any]] = []
    sample_frames: list[pd.DataFrame] = []
    feature_frames: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []

    for index, row in enumerate(matching_rows, start=1):
        method = clean_text(row.get("split_strategy")) or clean_text(row.get("method"))
        fold = clean_text(row.get("fold_id"))
        print(f"[{index}/{len(matching_rows)}] Wx_v2 {row.get('alloy_family')} / {row.get('dataset_name')} / {row.get('property')} / {method} / {fold}")
        try:
            result, split_output_dir = compute_one_split(row)
            write_split_outputs(result, split_output_dir)
            split_rows.append(result.split_row)
            sample_frames.append(result.sample_scores)
            feature_frames.append(result.feature_contributions)
        except Exception as exc:
            failures.append(failure_row_from(row, exc))
            print(f"  ERROR: {exc!r}", file=sys.stderr)
            if args.strict:
                write_csv(pd.DataFrame(failures), output_root / "wx_v2_failures.csv", FAILURE_COLUMNS)
                raise

    write_csv(pd.DataFrame(split_rows), output_root / "wx_v2_split_summary.csv", SPLIT_SUMMARY_COLUMNS)
    write_csv(
        pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame(columns=SAMPLE_COLUMNS),
        output_root / "all_wx_v2_sample_scores.csv",
        SAMPLE_COLUMNS,
    )
    write_csv(
        pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame(columns=FEATURE_COLUMNS),
        output_root / "all_wx_v2_feature_contributions.csv",
        FEATURE_COLUMNS,
    )
    write_csv(pd.DataFrame(failures), output_root / "wx_v2_failures.csv", FAILURE_COLUMNS)

    print(f"Scored splits: {len(split_rows)}")
    print(f"Failures: {len(failures)}")
    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()
