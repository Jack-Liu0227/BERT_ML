import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "Scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "Scripts"))

from Scripts import export_three_space_w_ood_report as three_space
from _raw_prediction_stats import read_prediction_csv, resolve_prediction_columns
from src.data_processing.compute_wx_v2_mixed_ot import compute_one_split

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


CASE_COLUMNS = ["alloy_family", "dataset_name", "property"]
SPACE_ORDER = ["X-space", "Y-space", "Z-space"]
HYBRID_SUBSETS = ("combined", "test_extrapolation_high20", "test_inner_ood")
STRATEGY_DIR_PATTERN = re.compile(
    r"^(?:"
    r"loco(?:_k5)?|"
    r"random_cv(?:_baseline|_k5)?|"
    r"target_extrapolation|"
    r"sparse_[xy]_(?:single|cluster)(?:_k5)?|"
    r"hybrid_extrapolation_(?:loco|random_cv|sparse_[xy]_(?:single|cluster))(?:_k5)?"
    r")$"
)
TEST_LABELS = {
    "test",
    "testing",
    "ood",
    "oodtest",
    "oodtesting",
    "extrapolationtest",
    "extrapolation_test",
    "test_extrapolation_high20",
    "testextrapolationhigh20",
    "test_hybrid_combined",
    "testhybridcombined",
    "test_inner_ood",
    "testinnerood",
}
METHOD_COLORS = {
    "RandCV": "#666666",
    "Extra.": "#4c78a8",
    "LOCO": "#b279a2",
    "SX-sgl": "#f58518",
    "SX-cls": "#e45756",
    "SY-sgl": "#72b7b2",
    "SY-cls": "#54a24b",
}
METHOD_ORDER = ["RandCV", "Extra.", "LOCO", "SX-sgl", "SX-cls", "SY-sgl", "SY-cls"]
HYBRID_METHOD_DIR_BY_SHORT = {
    "RandCV": "hybrid_extrapolation_random_cv",
    "LOCO": "hybrid_extrapolation_loco",
    "SX-sgl": "hybrid_extrapolation_sparse_x_single",
    "SX-cls": "hybrid_extrapolation_sparse_x_cluster",
    "SY-sgl": "hybrid_extrapolation_sparse_y_single",
    "SY-cls": "hybrid_extrapolation_sparse_y_cluster",
}
JOINED_OUTPUT_COLUMNS = [
    "scope",
    "task_key",
    "task_id",
    "alloy_family",
    "dataset_name",
    "property",
    "method_short",
    "method",
    "model",
    "model_family",
    "fold_id",
    "ID",
    "sample_order",
    "test_set",
    "space",
    "sample_w_contribution",
    "sample_w_mass_contribution",
    "split_w",
    "target_value",
    "true_value",
    "predicted_value",
    "signed_error",
    "abs_error",
    "relative_error_pct",
    "prediction_file",
    "source_split_dir",
    "join_mode",
]
COVERAGE_COLUMNS = [
    "scope",
    "task_key",
    "method_short",
    "model",
    "fold_id",
    "w_rows",
    "error_rows",
    "matched_rows",
    "unmatched_w_rows",
]
CORRELATION_COLUMNS = [
    "scope",
    "task_key",
    "task_id",
    "model",
    "method_short",
    "space",
    "error_metric",
    "n",
    "pearson_r",
    "spearman_r",
]
PREDICTION_INVENTORY_COLUMNS = [
    "scope",
    "task_key",
    "task_id",
    "method_short",
    "model",
    "fold_id",
    "prediction_file",
    "exists",
    "error_rows",
    "source_mode",
]
FIGURE_MANIFEST_COLUMNS = ["figure_group", "task_id", "model", "method_short", "rep_space", "figure", "format"]
STANDARD_YZ_SEVERITY_COLUMNS = [
    "task_key",
    "task_id",
    "alloy_family",
    "dataset_name",
    "property",
    "method_short",
    "Wy",
    "Wz",
    "Wy_random",
    "Wz_random",
    "Ry",
    "Rz",
    "n_folds_y",
    "n_folds_z",
]
PHASE_DIAGRAM_W_COLUMNS = ["X_space_w", "Y_space_w", "Z_space_w"]
PHASE_DIAGRAM_REP_SPACES = {
    "zy": ("Z-space", "Z_space_w", "zy_phase_diagram"),
    "xy": ("X-space", "X_space_w", "xy_phase_diagram"),
}
TASK_ID_TO_CASE = {
    "Al-UTS": ("Al", "aluminum", "UTS(MPa)"),
    "HEA-El": ("HEA", "hea", "El(%)"),
    "HEA-UTS": ("HEA", "hea", "UTS(MPa)"),
    "HEA-YS": ("HEA", "hea", "YS(MPa)"),
    "Steel-El": ("Steel", "steel", "El(%)"),
    "Steel-UTS": ("Steel", "steel", "UTS(MPa)"),
    "Steel-YS": ("Steel", "steel", "YS(MPa)"),
    "Ti-El": ("Ti", "titanium", "El(%)"),
    "Ti-UTS": ("Ti", "titanium", "UTS(MPa)"),
    "Matbench Steel-YS": ("MatbenchSteels", "matbench_steels_ood", "yield strength"),
    "Matbench_Steel-YS": ("MatbenchSteels", "matbench_steels_ood", "yield strength"),
}


def final_results_root() -> Path:
    return (
        Path("D:/XJTU")
        / "\u5df2\u5b8c\u6210\u8bba\u6587\u6570\u636e\u6c47\u603b"
        / "Fewshot"
        / "\u9884\u5904\u7406\u6c47\u603b\u6570\u636e"
        / "\u6700\u7ec8\u7ed3\u679c\u56fe"
    )


DEFAULT_STANDARD_W = Path("output") / "ood_xspace_scores" / "sample_w_reports_v2" / "three_space_sample_w_values.csv"
DEFAULT_HYBRID_CACHE = Path("output") / "ood_xspace_scores" / "hybrid_w_error_cache"
DEFAULT_EMBEDDING_DIR = three_space.DEFAULT_EMBEDDING_DATA_DIR
DEFAULT_OOD_OUTPUT = final_results_root() / "OOD" / "w_error_relationship"
DEFAULT_HYBRID_OUTPUT = final_results_root() / "OOD HYBIRD" / "w_error_relationship"
GPT_PREDICTION_ROOT = Path("D:/XJTU/ImportantFile/auto-design-alloy/fewshot-guided/output/ood/k5")


@dataclass(frozen=True)
class PredictionSource:
    scope: str
    task_key: str
    task_id: str
    alloy_family: str
    dataset_name: str
    property: str
    method_short: str
    method_raw: str
    model: str
    model_family: str
    prediction_file: Path
    fold_id: str
    model_dir: Path | None = None
    expected_split_file: Path | None = None
    source_mode: str = ""


def parse_max_relative_error_pct(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    try:
        threshold = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--max-relative-error-pct must be a non-negative number or none") from exc
    if threshold < 0:
        raise argparse.ArgumentTypeError("--max-relative-error-pct must be non-negative")
    return threshold


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot per-sample W versus per-sample prediction errors.")
    parser.add_argument("--scope", choices=["ood", "hybrid", "both"], default="both")
    parser.add_argument("--standard-w-table", default=str(DEFAULT_STANDARD_W))
    parser.add_argument("--hybrid-cache-root", default=str(DEFAULT_HYBRID_CACHE))
    parser.add_argument("--embedding-data-dir", default=str(DEFAULT_EMBEDDING_DIR))
    parser.add_argument("--ood-output", default=str(DEFAULT_OOD_OUTPUT))
    parser.add_argument("--hybrid-output", default=str(DEFAULT_HYBRID_OUTPUT))
    parser.add_argument("--force-recompute-hybrid-w", action="store_true")
    parser.add_argument("--case-contains", default=None, help="Optional task/path filter for smoke runs.")
    parser.add_argument("--max-sources", type=int, default=None, help="Optional prediction source limit for smoke runs.")
    parser.add_argument("--max-relative-error-pct", type=parse_max_relative_error_pct, default=None)
    parser.add_argument("--phase-diagram-spaces", choices=["both", "zy", "xy", "none"], default="both")
    parser.add_argument("--phase-diagram-color-percentile", type=float, default=95.0)
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Write joined CSV/summary artifacts but skip all figure rendering.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    return parser.parse_args(argv)


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def normalize_family(value: object) -> str:
    text = clean_text(value)
    if text == "HEA_half":
        return "HEA"
    if text in {"Matbench Steel", "Matbench_Steel", "matbench_steel", "matbench_steels"}:
        return "MatbenchSteels"
    return text


def clean_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def slugify(value: object) -> str:
    text = clean_text(value)
    text = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", text, flags=re.UNICODE)
    return text.strip("._") or "unknown"


def case_key_from_values(alloy_family: object, dataset_name: object, property_name: object) -> str:
    return "__".join(slugify(value) for value in (normalize_family(alloy_family), dataset_name, property_name))


def case_filter_matches(case_contains: str | None, *values: object) -> bool:
    if not case_contains:
        return True
    needle = case_contains.lower()
    return any(needle in clean_text(value).lower() for value in values)


def property_short(property_name: object) -> str:
    text = clean_text(property_name)
    if text.lower() == "yield strength":
        return "YS"
    text = text.replace("(MPa)", "").replace("(%)", "").replace("%", "")
    text = text.replace("yieldstrength", "YS").replace("yield strength", "YS")
    return slugify(text).replace("_", "-")


def task_id_from_values(alloy_family: object, property_name: object) -> str:
    family = normalize_family(alloy_family)
    if family == "MatbenchSteels":
        family = "Matbench Steel"
    return f"{family}-{property_short(property_name)}"


def normalize_id(value: object) -> str:
    text = clean_text(value)
    if not text:
        return ""
    number = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    if pd.notna(number) and float(number).is_integer():
        return str(int(number))
    return text


def normalize_standard_method(value: object) -> str:
    raw = clean_text(value)
    key = raw.lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "random_cv": "RandCV",
        "randomcv": "RandCV",
        "random_cv_baseline": "RandCV",
        "target_extrapolation": "Extra.",
        "extrapolation": "Extra.",
        "extra.": "Extra.",
        "extra": "Extra.",
        "loco": "LOCO",
        "loco_k5": "LOCO",
        "sparse_x_single": "SX-sgl",
        "sparsexsingle": "SX-sgl",
        "sx_sgl": "SX-sgl",
        "sx-sgl": "SX-sgl",
        "sparse_x_cluster": "SX-cls",
        "sparsexcluster": "SX-cls",
        "sx_cls": "SX-cls",
        "sx-cls": "SX-cls",
        "sparse_y_single": "SY-sgl",
        "sparseysingle": "SY-sgl",
        "sy_sgl": "SY-sgl",
        "sy-sgl": "SY-sgl",
        "sparse_y_cluster": "SY-cls",
        "sparseycluster": "SY-cls",
        "sy_cls": "SY-cls",
        "sy-cls": "SY-cls",
    }
    return mapping.get(key, raw)


def normalize_hybrid_method(value: object) -> str:
    raw = clean_text(value)
    key = raw.lower().replace("-", "_").replace(" ", "_").replace("+", "_")
    key = key.replace("randcv", "random_cv")
    for prefix in ("hybridhigh20_", "hybrid_high20_", "hybrid_extrapolation_"):
        if key.startswith(prefix):
            key = key[len(prefix) :]
    return normalize_standard_method(key)


def compute_error_columns(frame: pd.DataFrame, *, true_col: str, pred_col: str) -> pd.DataFrame:
    result = frame.copy()
    true_values = pd.to_numeric(result[true_col], errors="coerce")
    predicted_values = pd.to_numeric(result[pred_col], errors="coerce")
    result["true_value"] = true_values
    result["predicted_value"] = predicted_values
    result["signed_error"] = predicted_values - true_values
    result["abs_error"] = result["signed_error"].abs()
    denominator = true_values.abs()
    result["relative_error_pct"] = np.where(denominator > 0, result["abs_error"] / denominator * 100.0, np.nan)
    return result


def filter_errors_by_relative_error(frame: pd.DataFrame, max_relative_error_pct: float | None) -> pd.DataFrame:
    if max_relative_error_pct is None or frame.empty or "relative_error_pct" not in frame.columns:
        return frame.copy()
    relative_error = pd.to_numeric(frame["relative_error_pct"], errors="coerce")
    return frame[relative_error.isna() | relative_error.le(max_relative_error_pct)].copy()


def filter_hybrid_subset(frame: pd.DataFrame, subset: str) -> pd.DataFrame:
    if subset in {"combined", "test_hybrid_combined"}:
        return frame.copy()
    if "test_set" not in frame.columns:
        return frame.iloc[0:0].copy()
    return frame[frame["test_set"].astype(str).str.lower().eq(subset.lower())].copy()


def prepare_w_table(frame: pd.DataFrame, *, scope: str, hybrid_subset: str | None = None) -> pd.DataFrame:
    result = frame.copy()
    if "case_key" not in result.columns:
        result["case_key"] = [
            case_key_from_values(row["alloy_family"], row["dataset_name"], row["property"])
            for _, row in result.iterrows()
        ]
    result["scope"] = scope
    result["task_key"] = result["case_key"].astype(str)
    result["alloy_family"] = result["alloy_family"].map(normalize_family)
    result["task_id"] = [task_id_from_values(row["alloy_family"], row["property"]) for _, row in result.iterrows()]
    normalizer = normalize_hybrid_method if scope == "hybrid" else normalize_standard_method
    result["method_short"] = result["method"].map(normalizer)
    result["fold_id"] = result.get("fold_id", "").fillna("").astype(str)
    result["join_id"] = result.get("ID", pd.Series([""] * len(result))).map(normalize_id)
    if "__row_id__" in result.columns:
        source_order = pd.to_numeric(result["__row_id__"], errors="coerce")
        fallback_order = result.groupby(["task_key", "method_short", "fold_id", "space"], sort=False).cumcount()
        result["sample_order"] = source_order.where(source_order.notna(), fallback_order)
    else:
        result["sample_order"] = result.groupby(["task_key", "method_short", "fold_id", "space"], sort=False).cumcount()
    if scope == "hybrid":
        result = annotate_hybrid_test_sets(result)
        if hybrid_subset is not None:
            result = filter_hybrid_subset(result, hybrid_subset)
    else:
        result["test_set"] = "ood_test"
    required = [
        "scope",
        "task_key",
        "task_id",
        *CASE_COLUMNS,
        "method",
        "method_short",
        "fold_id",
        "join_id",
        "ID",
        "sample_order",
        "test_set",
        "space",
        "source_split_dir",
        "sample_w_contribution",
        "sample_w_mass_contribution",
        "split_w",
        "target_value",
    ]
    for column in required:
        if column not in result.columns:
            result[column] = np.nan
    return result[required].copy()


def annotate_hybrid_test_sets(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if result.empty:
        result["test_set"] = []
        return result
    result["test_set"] = "test_hybrid_combined"
    maps: dict[str, dict[str, str]] = {}
    for source_split_dir in result["source_split_dir"].dropna().astype(str).unique():
        maps[source_split_dir] = hybrid_id_to_subset_map(Path(source_split_dir))
    labels: list[str] = []
    for _, row in result.iterrows():
        label = maps.get(str(row.get("source_split_dir")), {}).get(normalize_id(row.get("ID")), "test_hybrid_combined")
        labels.append(label)
    result["test_set"] = labels
    return result


def hybrid_id_to_subset_map(source_split_dir: Path) -> dict[str, str]:
    summary_path = source_split_dir / "split_summary.json"
    combined_path = source_split_dir / "test_hybrid_combined.csv"
    if not summary_path.exists() or not combined_path.exists():
        return {}
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        combined = read_csv(combined_path)
    except Exception:
        return {}
    if "ID" not in combined.columns:
        return {}
    explicit_mapping: dict[str, str] = {}
    for subset in ("test_extrapolation_high20", "test_inner_ood"):
        subset_path = source_split_dir / "test_sets" / f"{subset}.csv"
        if not subset_path.exists():
            continue
        try:
            subset_frame = read_csv(subset_path)
        except Exception:
            continue
        if "ID" not in subset_frame.columns:
            continue
        for sample_id in subset_frame["ID"].tolist():
            explicit_mapping[normalize_id(sample_id)] = subset
    if explicit_mapping:
        for sample_id in combined["ID"].tolist():
            key = normalize_id(sample_id)
            if key and key not in explicit_mapping:
                explicit_mapping[key] = "test_hybrid_combined"
        return explicit_mapping
    outer_n = int(payload.get("outer_test_size", 0) or 0)
    inner_n = int(payload.get("inner_test_size", 0) or 0)
    mapping: dict[str, str] = {}
    for idx, sample_id in enumerate(combined["ID"].tolist()):
        if idx < outer_n:
            label = "test_extrapolation_high20"
        elif idx < outer_n + inner_n:
            label = "test_inner_ood"
        else:
            label = "test_hybrid_combined"
        mapping[normalize_id(sample_id)] = label
    return mapping


def join_w_and_errors(w_values: pd.DataFrame, errors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if w_values.empty or errors.empty:
        return pd.DataFrame(columns=[*JOINED_OUTPUT_COLUMNS, "__w_row_id"]), pd.DataFrame(columns=COVERAGE_COLUMNS)
    w = w_values.copy().reset_index(drop=True)
    e = errors.copy().reset_index(drop=True)
    w["__w_row_id"] = np.arange(len(w))
    e["__error_row_id"] = np.arange(len(e))
    for frame in (w, e):
        frame["join_id"] = frame.get("join_id", frame.get("ID", "")).map(normalize_id)
        frame["fold_id"] = frame.get("fold_id", "").fillna("").astype(str)
        frame["sample_order"] = pd.to_numeric(frame.get("sample_order", np.nan), errors="coerce")
    key_cols = ["scope", "task_key", "method_short", "fold_id"]
    id_w = w[w["join_id"].astype(str).ne("")]
    id_e = e[e["join_id"].astype(str).ne("")]
    joined = id_w.merge(id_e, on=[*key_cols, "join_id"], how="inner", suffixes=("_w", ""))
    joined["join_mode"] = "fold_id_id"
    order_e = e[e["join_id"].astype(str).eq("")]
    if not order_e.empty:
        order_frames: list[pd.DataFrame] = []
        group_cols = [*key_cols, "model"]
        for keys, error_group in order_e.groupby(group_cols, dropna=False, sort=False):
            values = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
            w_mask = pd.Series(True, index=w.index)
            for column in key_cols:
                w_mask &= w[column].astype(str).eq(str(values[column]))
            already_matched: set[object] = set()
            if not joined.empty:
                joined_mask = pd.Series(True, index=joined.index)
                for column in key_cols:
                    joined_mask &= joined[column].astype(str).eq(str(values[column]))
                joined_mask &= joined["model"].astype(str).eq(str(values["model"]))
                already_matched = set(joined.loc[joined_mask, "__w_row_id"].tolist())
            candidate_w = w[w_mask & ~w["__w_row_id"].isin(already_matched)]
            if candidate_w.empty:
                continue
            order_frames.append(
                candidate_w.merge(error_group, on=[*key_cols, "sample_order"], how="inner", suffixes=("_w", ""))
            )
        if order_frames:
            order_joined = pd.concat(order_frames, ignore_index=True, sort=False)
            order_joined["join_mode"] = "fold_id_sample_order"
            joined = pd.concat([joined, order_joined], ignore_index=True, sort=False)
    if joined.empty:
        coverage = build_join_coverage(w, e, joined)
        return joined, coverage
    joined["space"] = joined.get("space_w", joined.get("space"))
    joined["ID"] = joined.get("ID_w", joined.get("ID"))
    joined["target_value"] = joined.get("target_value_w", joined.get("target_value"))
    keep_cols = [*JOINED_OUTPUT_COLUMNS, "__w_row_id"]
    for column in keep_cols:
        if column not in joined.columns:
            joined[column] = np.nan
    coverage = build_join_coverage(w, e, joined)
    return joined[keep_cols].copy(), coverage


def build_join_coverage(w: pd.DataFrame, e: pd.DataFrame, joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    key_cols = ["scope", "task_key", "method_short"]
    for keys, error_group in e.groupby([*key_cols, "model"], dropna=False, sort=True):
        values = dict(zip([*key_cols, "model"], keys if isinstance(keys, tuple) else (keys,)))
        mask = pd.Series(True, index=w.index)
        for column in key_cols:
            mask &= w[column].astype(str).eq(str(values[column]))
        w_group = w[mask]
        if joined.empty:
            matched_w = 0
        else:
            joined_mask = pd.Series(True, index=joined.index)
            for column in key_cols:
                joined_mask &= joined[column].astype(str).eq(str(values[column]))
            joined_mask &= joined["model"].astype(str).eq(str(values["model"]))
            matched_w = int(joined.loc[joined_mask, "__w_row_id"].nunique())
        rows.append(
            {
                **values,
                "fold_id": "all",
                "w_rows": int(len(w_group)),
                "error_rows": int(len(error_group)),
                "matched_rows": matched_w,
                "unmatched_w_rows": int(max(len(w_group) - matched_w, 0)),
            }
        )
    return pd.DataFrame(rows, columns=COVERAGE_COLUMNS)


def resolve_path(path_text: object) -> Path | None:
    text = clean_text(path_text)
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def resolve_confidence_prediction_path(row: pd.Series, prediction_file: Path) -> Path:
    if prediction_file.exists():
        return prediction_file
    path_parts_lower = [part.lower() for part in prediction_file.parts]
    if "gpt-5.4" not in path_parts_lower:
        return prediction_file
    root = clean_text(row.get("root"))
    task_dir = clean_text(row.get("task_dir"))
    if not root or not task_dir:
        return prediction_file
    gpt_index = path_parts_lower.index("gpt-5.4")
    provider = prediction_file.parts[gpt_index - 1] if gpt_index > 0 else ""
    fold = clean_text(row.get("fold"))
    candidate = GPT_PREDICTION_ROOT / root / task_dir
    if fold:
        candidate = candidate / fold
    if provider:
        candidate = candidate / provider
    candidate = candidate / "gpt-5.4" / "predictions.csv"
    if candidate.exists():
        return candidate
    search_root = GPT_PREDICTION_ROOT / root / task_dir
    if search_root.exists():
        matches = sorted(search_root.rglob("gpt-5.4/predictions.csv"))
        if len(matches) == 1:
            return matches[0]
    return prediction_file


def confidence_hybrid_method_dir(row: pd.Series, method_short: str) -> str:
    root = clean_text(row.get("root"))
    match = re.match(r"^strength_ood_(hybrid_extrapolation_.+?)_no_analysis$", root)
    if match:
        return match.group(1)
    return HYBRID_METHOD_DIR_BY_SHORT.get(method_short, "")


def resolve_confidence_hybrid_split_file(
    row: pd.Series,
    *,
    family: str,
    dataset: str,
    prop: str,
    method_short: str,
    fold_id: str,
) -> Path | None:
    method_dir = confidence_hybrid_method_dir(row, method_short)
    if not method_dir:
        return None
    split_root = REPO_ROOT / "output" / "ood_splits" / normalize_family(family) / dataset / prop / method_dir
    if not split_root.exists():
        return None
    if fold_id:
        matches = sorted(split_root.glob(f"*/folds/{fold_id}/split_data/test_hybrid_combined.csv"))
    else:
        matches = sorted(split_root.glob("*/split_data/test_hybrid_combined.csv"))
    matches = [path for path in matches if (path.parent / "split_summary.json").exists()]
    return matches[0] if matches else None


def parse_json_list(text: object) -> list[dict[str, object]]:
    raw = clean_text(text)
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except Exception:
        return []
    return payload if isinstance(payload, list) else []


def parse_fold_detail_prediction_sources(row: pd.Series, detail_column: str) -> list[dict[str, object]]:
    details: list[dict[str, object]] = []
    artifact_path = resolve_path(row.get("artifact_predictions_file"))
    artifact_strategy = infer_strategy_dir(artifact_path)
    for detail in parse_json_list(row.get(detail_column)):
        prediction_file = resolve_path(detail.get("outer_predictions_file") or detail.get("predictions_file"))
        if prediction_file is None:
            continue
        if artifact_strategy:
            detail_strategy = infer_strategy_dir(prediction_file)
            if detail_strategy and detail_strategy != artifact_strategy:
                continue
        fold_index = detail.get("outer_fold_index", detail.get("fold_index"))
        fold_id = f"fold_{int(fold_index)}" if pd.notna(fold_index) else infer_fold_id(prediction_file)
        details.append(
            {
                "prediction_file": prediction_file,
                "fold_id": fold_id,
                "model_dir": resolve_path(detail.get("model_dir")),
            }
        )
    return details


def infer_strategy_dir(path: Path | None) -> str:
    if path is None:
        return ""
    parts = [part.lower() for part in path.parts]
    for part in parts:
        if STRATEGY_DIR_PATTERN.match(part):
            return part
    return ""


def collect_summary_prediction_sources(scope: str, *, case_contains: str | None = None) -> list[PredictionSource]:
    root = REPO_ROOT / "output" / ("ood_summary_reports_hybrid" if scope == "hybrid" else "ood_summary_reports")
    sources: list[PredictionSource] = []
    for table_path in sorted(root.glob("*/00_summary_tables/*.csv")):
        table = read_csv(table_path)
        for _, row in table.iterrows():
            family = normalize_family(row.get("alloy_family"))
            dataset = clean_text(row.get("dataset_name"))
            prop = clean_text(row.get("property"))
            task_key = case_key_from_values(family, dataset, prop)
            if not case_filter_matches(case_contains, task_key, table_path):
                continue
            method_raw = clean_text(row.get("ood_method") or row.get("method"))
            method_short = normalize_hybrid_method(method_raw) if scope == "hybrid" else normalize_standard_method(method_raw)
            model = clean_text(row.get("model")) or clean_text(row.get("model_family")) or table_path.parent.parent.name
            model_family = clean_text(row.get("model_family")) or table_path.parent.parent.name
            base = {
                "scope": scope,
                "task_key": task_key,
                "task_id": task_id_from_values(family, prop),
                "alloy_family": family,
                "dataset_name": dataset,
                "property": prop,
                "method_short": method_short,
                "method_raw": method_raw,
                "model": model,
                "model_family": model_family,
            }
            detail_sources: list[dict[str, object]] = []
            for detail_column in ("loco_outer_fold_best_details_json", "tabpfn_loco_fold_details_json"):
                detail_sources.extend(parse_fold_detail_prediction_sources(row, detail_column))
            for detail in detail_sources:
                prediction_file = detail.get("prediction_file")
                if prediction_file is None:
                    continue
                sources.append(
                    PredictionSource(
                        **base,
                        prediction_file=prediction_file,
                        fold_id=clean_text(detail.get("fold_id")),
                        model_dir=detail.get("model_dir"),
                        source_mode="outer_fold_details",
                    )
                )
            if detail_sources:
                continue
            for column, mode in [
                ("artifact_predictions_file", "artifact_predictions_file"),
                ("representative_predictions_file", "representative_predictions_file"),
                ("selected_source_predictions_file", "selected_source_predictions_file"),
            ]:
                prediction_file = resolve_path(row.get(column))
                if prediction_file is None:
                    continue
                sources.append(
                    PredictionSource(
                        **base,
                        prediction_file=prediction_file,
                        fold_id=infer_fold_id(prediction_file),
                        model_dir=resolve_path(row.get("model_dir")),
                        expected_split_file=resolve_path(row.get("artifact_expected_split_file")),
                        source_mode=mode,
                    )
                )
    return deduplicate_sources(sources)


def collect_confidence_prediction_sources(scope: str, *, case_contains: str | None = None) -> list[PredictionSource]:
    base_dir = final_results_root() / ("OOD HYBIRD" if scope == "hybrid" else "OOD")
    table_path = base_dir / "confidence_ood_relationship" / "csv" / "confidence_by_prediction_file.csv"
    if not table_path.exists():
        return []
    table = read_csv(table_path)
    sources: list[PredictionSource] = []
    for _, row in table.iterrows():
        task_id = clean_text(row.get("task_id"))
        case_values = TASK_ID_TO_CASE.get(task_id)
        if case_values is None:
            continue
        family, dataset, prop = case_values
        task_key = case_key_from_values(family, dataset, prop)
        if not case_filter_matches(case_contains, task_key, task_id, table_path):
            continue
        method_raw = clean_text(row.get("ood_method") or row.get("method"))
        method_short = normalize_hybrid_method(method_raw) if scope == "hybrid" else normalize_standard_method(method_raw)
        prediction_file = resolve_path(row.get("prediction_path"))
        if prediction_file is None:
            continue
        prediction_file = resolve_confidence_prediction_path(row, prediction_file)
        fold_id = clean_text(row.get("fold")) or infer_fold_id(prediction_file)
        expected_split_file = None
        if scope == "hybrid":
            expected_split_file = resolve_confidence_hybrid_split_file(
                row,
                family=family,
                dataset=dataset,
                prop=prop,
                method_short=method_short,
                fold_id=fold_id,
            )
        sources.append(
            PredictionSource(
                scope=scope,
                task_key=task_key,
                task_id=task_id,
                alloy_family=family,
                dataset_name=dataset,
                property=prop,
                method_short=method_short,
                method_raw=method_raw,
                model="GPT-5.4",
                model_family="LLM",
                prediction_file=prediction_file,
                fold_id=fold_id,
                expected_split_file=expected_split_file,
                source_mode="confidence_prediction_file",
            )
        )
    return deduplicate_sources(sources)


def deduplicate_sources(sources: list[PredictionSource]) -> list[PredictionSource]:
    seen: set[tuple[str, str, str, str, str, str]] = set()
    unique: list[PredictionSource] = []
    for source in sources:
        key = (
            source.scope,
            source.task_key,
            source.method_short,
            source.model,
            source.fold_id,
            str(source.prediction_file),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(source)
    return unique


def infer_fold_id(path: Path | None) -> str:
    if path is None:
        return ""
    for candidate in (path, *path.parents):
        match = re.match(r"fold_(\d+)$", candidate.name)
        if match:
            return f"fold_{match.group(1)}"
    return ""


def prediction_errors_from_source(source: PredictionSource, *, hybrid_subset: str | None = None) -> pd.DataFrame:
    if not source.prediction_file.exists():
        return pd.DataFrame()
    frame = read_prediction_csv(source.prediction_file)
    if frame is None or frame.empty:
        return pd.DataFrame()
    dataset_col, actual_col, pred_col = resolve_prediction_columns(frame, source.property)
    if actual_col is None or pred_col is None:
        return pd.DataFrame()
    work = frame.copy()
    if dataset_col is not None:
        normalized = work[dataset_col].astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False)
        work = work[normalized.isin(TEST_LABELS)].copy()
    if work.empty:
        return pd.DataFrame()
    id_col = "ID" if "ID" in work.columns else "id" if "id" in work.columns else None
    work = work.reset_index(drop=True)
    work["sample_order"] = np.arange(len(work))
    if id_col is not None:
        work["ID"] = work[id_col]
        work["join_id"] = work[id_col].map(normalize_id)
    else:
        work["ID"] = np.nan
        work["join_id"] = ""
    if source.scope == "hybrid":
        work["test_set"] = classify_hybrid_prediction_rows(source, work)
        if hybrid_subset is not None:
            work = filter_hybrid_subset(work, hybrid_subset)
    else:
        work["test_set"] = "ood_test"
    if work.empty:
        return pd.DataFrame()
    work = compute_error_columns(work, true_col=actual_col, pred_col=pred_col)
    result = pd.DataFrame(
        {
            "scope": source.scope,
            "task_key": source.task_key,
            "task_id": source.task_id,
            "alloy_family": source.alloy_family,
            "dataset_name": source.dataset_name,
            "property": source.property,
            "method_short": source.method_short,
            "method": source.method_raw,
            "model": source.model,
            "model_family": source.model_family,
            "fold_id": source.fold_id,
            "ID": work["ID"].to_numpy(),
            "join_id": work["join_id"].to_numpy(),
            "sample_order": work["sample_order"].to_numpy(),
            "test_set": work["test_set"].to_numpy(),
            "true_value": work["true_value"].to_numpy(),
            "predicted_value": work["predicted_value"].to_numpy(),
            "signed_error": work["signed_error"].to_numpy(),
            "abs_error": work["abs_error"].to_numpy(),
            "relative_error_pct": work["relative_error_pct"].to_numpy(),
            "prediction_file": str(source.prediction_file),
            "source_mode": source.source_mode,
        }
    )
    return result


def classify_hybrid_prediction_rows(source: PredictionSource, frame: pd.DataFrame) -> list[str]:
    mapping = hybrid_map_from_source(source)
    if mapping and "join_id" in frame.columns:
        return [mapping.get(normalize_id(value), "test_hybrid_combined") for value in frame["join_id"].tolist()]
    return ["test_hybrid_combined"] * len(frame)


def hybrid_map_from_source(source: PredictionSource) -> dict[str, str]:
    candidates: list[Path] = []
    if source.expected_split_file is not None:
        candidates.append(source.expected_split_file.parent)
    if source.model_dir is not None:
        candidates.append(source.model_dir / "split_data")
    for parent in source.prediction_file.parents:
        split_data = parent / "split_data"
        candidates.append(split_data)
    for candidate in candidates:
        mapping = hybrid_id_to_subset_map(candidate)
        if mapping:
            return mapping
    return {}


def collect_prediction_errors(scope: str, *, hybrid_subset: str | None = None, case_contains: str | None = None, max_sources: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    sources = collect_summary_prediction_sources(scope, case_contains=case_contains)
    sources.extend(collect_confidence_prediction_sources(scope, case_contains=case_contains))
    sources = deduplicate_sources(sources)
    if max_sources is not None:
        sources = sources[:max_sources]
    frames: list[pd.DataFrame] = []
    inventory_rows: list[dict[str, object]] = []
    for index, source in enumerate(sources, start=1):
        print(f"[{index}/{len(sources)}] read predictions {source.scope} {source.task_id} {source.method_short} {source.model} {source.fold_id}")
        errors = prediction_errors_from_source(source, hybrid_subset=hybrid_subset)
        inventory_rows.append(
            {
                "scope": source.scope,
                "task_key": source.task_key,
                "task_id": source.task_id,
                "method_short": source.method_short,
                "model": source.model,
                "fold_id": source.fold_id,
                "prediction_file": str(source.prediction_file),
                "exists": source.prediction_file.exists(),
                "error_rows": int(len(errors)),
                "source_mode": source.source_mode,
            }
        )
        if not errors.empty:
            frames.append(errors)
    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return combined, pd.DataFrame(inventory_rows, columns=PREDICTION_INVENTORY_COLUMNS)


def build_hybrid_split_summary(subset: str, cache_root: Path, *, case_contains: str | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    split_paths = sorted((REPO_ROOT / "output" / "ood_splits").rglob("split_summary.json"))
    for summary_path in split_paths:
        source_split_dir = summary_path.parent
        if "hybrid_extrapolation" not in str(source_split_dir):
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        combined_test = source_split_dir / "test_hybrid_combined.csv"
        train_file = source_split_dir / "train.csv"
        if not train_file.exists() or not combined_test.exists():
            continue
        test_file = combined_test
        if subset != "combined":
            test_file = write_hybrid_subset_test_file(source_split_dir, payload, subset, cache_root)
            if test_file is None:
                continue
        parts = source_split_dir.parts
        try:
            idx = parts.index("ood_splits")
            family, dataset, prop, method, split_id = parts[idx + 1 : idx + 6]
        except Exception:
            family = clean_text(payload.get("alloy_family"))
            dataset = clean_text(payload.get("dataset_name"))
            prop = clean_text(payload.get("split_target_col"))
            method = clean_text(payload.get("split_strategy"))
            split_id = source_split_dir.parent.name
        task_key = case_key_from_values(family, dataset, prop)
        task_id = task_id_from_values(family, prop)
        if not case_filter_matches(case_contains, task_key, task_id, source_split_dir):
            continue
        fold_id = infer_fold_id(source_split_dir)
        test_df = read_csv(test_file)
        train_df = read_csv(train_file)
        rows.append(
            {
                "alloy_family": normalize_family(family),
                "dataset_name": dataset,
                "property": prop,
                "target_col": clean_text(payload.get("split_target_col")) or prop,
                "split_strategy": clean_text(payload.get("split_strategy")) or method,
                "method": clean_text(payload.get("split_strategy")) or method,
                "split_id": split_id,
                "fold_id": fold_id,
                "source_split_dir": str(source_split_dir),
                "output_dir": "",
                "train_file": str(train_file),
                "test_file": str(test_file),
                "train_size": int(len(train_df)),
                "test_size": int(len(test_df)),
                "feature_count": int(len(payload.get("x_space_feature_columns", []))),
            }
        )
    return pd.DataFrame(rows)


def write_hybrid_subset_test_file(source_split_dir: Path, payload: dict[str, object], subset: str, cache_root: Path) -> Path | None:
    combined_path = source_split_dir / "test_hybrid_combined.csv"
    if not combined_path.exists():
        return None
    if subset == "test_extrapolation_high20":
        explicit_path = source_split_dir / "test_sets" / "test_extrapolation_high20.csv"
    elif subset == "test_inner_ood":
        explicit_path = source_split_dir / "test_sets" / "test_inner_ood.csv"
    else:
        return None
    if explicit_path.exists():
        child = read_csv(explicit_path)
    else:
        combined = read_csv(combined_path)
        outer_n = int(payload.get("outer_test_size", 0) or 0)
        inner_n = int(payload.get("inner_test_size", 0) or 0)
        if subset == "test_extrapolation_high20":
            child = combined.iloc[:outer_n].copy()
        else:
            child = combined.iloc[outer_n : outer_n + inner_n].copy()
    safe_rel = "_".join(slugify(part) for part in source_split_dir.parts[-8:])
    output_path = cache_root / "subset_test_files" / subset / f"{safe_rel}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    child.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def ensure_hybrid_w_table(subset: str, cache_root: Path, embedding_data_dir: Path, *, force: bool = False, case_contains: str | None = None) -> Path:
    subset_root = cache_root / subset
    output_path = subset_root / "three_space_sample_w_values.csv"
    if output_path.exists() and not force:
        return output_path
    subset_root.mkdir(parents=True, exist_ok=True)
    split_summary = build_hybrid_split_summary(subset, subset_root, case_contains=case_contains)
    if split_summary.empty:
        raise RuntimeError(f"No hybrid split rows found for subset {subset}.")
    split_rows: list[dict[str, object]] = []
    sample_frames: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []
    for index, row in split_summary.iterrows():
        print(f"[{index + 1}/{len(split_summary)}] compute hybrid Wx_v2 {row['alloy_family']} {row['property']} {row['method']} {row['fold_id']}")
        try:
            result, _ = compute_one_split(row)
            split_rows.append(result.split_row)
            sample_frames.append(result.sample_scores)
        except Exception as exc:
            failures.append({**row.to_dict(), "error": repr(exc)})
            print(f"  WARN: {exc!r}", file=sys.stderr)
    wx_split = pd.DataFrame(split_rows)
    wx_samples = pd.concat(sample_frames, ignore_index=True, sort=False) if sample_frames else pd.DataFrame()
    write_csv(split_summary, subset_root / "hybrid_split_summary.csv")
    write_csv(wx_split, subset_root / "wx_v2_split_summary.csv")
    write_csv(wx_samples, subset_root / "all_wx_v2_sample_scores.csv")
    write_csv(pd.DataFrame(failures), subset_root / "wx_v2_failures.csv")
    if wx_split.empty or wx_samples.empty:
        raise RuntimeError(f"No hybrid Wx_v2 rows were computed for subset {subset}.")
    sample_values = wx_samples.rename(
        columns={
            "wx_v2_sample_score": "sample_w_contribution",
            "wx_v2_mass_contribution": "sample_w_mass_contribution",
            "wx_v2_rank_desc": "sample_w_rank_desc",
        }
    ).copy()
    sample_values["ood_score"] = sample_values["sample_w_contribution"]
    sample_values["ood_percentile_vs_train"] = np.nan
    split_for_three = split_summary.merge(
        wx_split[["source_split_dir", "wx_v2"]],
        on="source_split_dir",
        how="left",
    )
    split_for_three["sliced_wasserstein"] = pd.to_numeric(split_for_three["wx_v2"], errors="coerce")
    split_for_three_path = subset_root / "hybrid_split_summary_for_three_space.csv"
    write_csv(split_for_three, split_for_three_path)
    three_space_values = three_space.build_three_space_sample_values(sample_values, split_for_three_path, embedding_data_dir)
    write_csv(three_space_values, output_path)
    return output_path


def build_correlations(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if joined.empty:
        return pd.DataFrame(columns=CORRELATION_COLUMNS)
    group_cols = ["scope", "task_key", "task_id", "model", "method_short", "space"]
    for keys, group in joined.groupby(group_cols, dropna=False, sort=True):
        row_base = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        x = pd.to_numeric(group["sample_w_contribution"], errors="coerce")
        for metric in ["abs_error", "relative_error_pct"]:
            y = pd.to_numeric(group[metric], errors="coerce")
            valid = x.notna() & y.notna()
            x_valid = x[valid].to_numpy(dtype=float)
            y_valid = y[valid].to_numpy(dtype=float)
            finite = np.isfinite(x_valid) & np.isfinite(y_valid)
            if int(finite.sum()) >= 2 and np.unique(x_valid[finite]).size >= 2 and np.unique(y_valid[finite]).size >= 2:
                finite_x = pd.Series(x_valid[finite])
                finite_y = pd.Series(y_valid[finite])
                pearson = float(finite_x.corr(finite_y, method="pearson"))
                spearman = float(finite_x.corr(finite_y, method="spearman"))
            else:
                pearson = np.nan
                spearman = np.nan
            rows.append({**row_base, "error_metric": metric, "n": int(finite.sum()), "pearson_r": pearson, "spearman_r": spearman})
    return pd.DataFrame(rows, columns=CORRELATION_COLUMNS)


def safe_filename(text: object) -> str:
    return slugify(text).replace("__", "_")


def hybrid_cache_run_root(cache_root: Path, case_contains: str | None) -> Path:
    if not case_contains:
        return cache_root / "full"
    return cache_root / "case_filters" / safe_filename(case_contains)


def ordered_methods(methods: Iterable[object]) -> list[str]:
    method_text = [str(method) for method in pd.Series(list(methods)).dropna().unique()]
    return sorted(method_text, key=lambda item: METHOD_ORDER.index(item) if item in METHOD_ORDER else 999)


def build_standard_yz_severity_summary(w_table: pd.DataFrame) -> pd.DataFrame:
    if w_table.empty:
        return pd.DataFrame(columns=STANDARD_YZ_SEVERITY_COLUMNS)
    required = ["task_key", "task_id", *CASE_COLUMNS, "method_short", "fold_id", "space", "split_w"]
    missing = [column for column in required if column not in w_table.columns]
    if missing:
        raise ValueError(f"Standard OOD W table is missing Wy-Wz severity columns: {missing}")
    work = w_table.copy()
    work["method_short"] = work["method_short"].map(normalize_standard_method)
    work = work[work["space"].isin(["Y-space", "Z-space"])].copy()
    if work.empty:
        return pd.DataFrame(columns=STANDARD_YZ_SEVERITY_COLUMNS)
    work["split_w"] = pd.to_numeric(work["split_w"], errors="coerce")
    work = work[work["split_w"].notna()].copy()
    fold_cols = ["task_key", "task_id", *CASE_COLUMNS, "method_short", "space", "fold_id"]
    fold_w = work.groupby(fold_cols, dropna=False, sort=False)["split_w"].mean().reset_index()
    group_cols = ["task_key", "task_id", *CASE_COLUMNS, "method_short", "space"]
    grouped = (
        fold_w.groupby(group_cols, dropna=False, sort=False)
        .agg(split_w_mean=("split_w", "mean"), n_folds=("fold_id", "nunique"))
        .reset_index()
    )
    pivot = grouped.pivot_table(
        index=["task_key", "task_id", *CASE_COLUMNS, "method_short"],
        columns="space",
        values=["split_w_mean", "n_folds"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}__{space}" for metric, space in pivot.columns]
    result = pivot.reset_index().rename(
        columns={
            "split_w_mean__Y-space": "Wy",
            "split_w_mean__Z-space": "Wz",
            "n_folds__Y-space": "n_folds_y",
            "n_folds__Z-space": "n_folds_z",
        }
    )
    for column in ["Wy", "Wz", "n_folds_y", "n_folds_z"]:
        if column not in result.columns:
            result[column] = np.nan
    random_base = result[result["method_short"].eq("RandCV")][["task_key", "Wy", "Wz"]].rename(
        columns={"Wy": "Wy_random", "Wz": "Wz_random"}
    )
    result = result.merge(random_base, on="task_key", how="left")
    for column in ["Wy", "Wz", "Wy_random", "Wz_random"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    valid = result["Wy"].notna() & result["Wz"].notna() & result["Wy_random"].gt(0) & result["Wz_random"].gt(0)
    result = result[valid].copy()
    if result.empty:
        return pd.DataFrame(columns=STANDARD_YZ_SEVERITY_COLUMNS)
    result["Ry"] = result["Wy"] / result["Wy_random"]
    result["Rz"] = result["Wz"] / result["Wz_random"]
    result["n_folds_y"] = pd.to_numeric(result["n_folds_y"], errors="coerce").fillna(0).astype(int)
    result["n_folds_z"] = pd.to_numeric(result["n_folds_z"], errors="coerce").fillna(0).astype(int)
    result = result[STANDARD_YZ_SEVERITY_COLUMNS].sort_values(["task_id", "method_short"], key=lambda series: series.map(str), kind="stable")
    method_rank = {method: index for index, method in enumerate(METHOD_ORDER)}
    result["_method_rank"] = result["method_short"].map(method_rank).fillna(999).astype(int)
    result = result.sort_values(["task_id", "_method_rank", "method_short"], kind="stable").drop(columns="_method_rank")
    return result.reset_index(drop=True)


def fit_linear_trend_line(x: Iterable[object], y: Iterable[object]) -> tuple[np.ndarray, np.ndarray] | None:
    x_values = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    if int(valid.sum()) < 2:
        return None
    x_valid = x_values[valid]
    y_valid = y_values[valid]
    if np.unique(x_valid).size < 2:
        return None
    slope, intercept = np.polyfit(x_valid, y_valid, 1)
    line_x = np.array([float(np.min(x_valid)), float(np.max(x_valid))])
    line_y = slope * line_x + intercept
    if not np.all(np.isfinite(line_y)):
        return None
    return line_x, line_y


def draw_method_points_and_trend(ax: plt.Axes, method_group: pd.DataFrame, metric: str, method: str, *, show_label: bool) -> None:
    x = pd.to_numeric(method_group["sample_w_contribution"], errors="coerce")
    y = pd.to_numeric(method_group[metric], errors="coerce")
    x_values = x.to_numpy(dtype=float)
    y_values = y.to_numpy(dtype=float)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    if not valid.any():
        return
    color = METHOD_COLORS.get(str(method), "#333333")
    ax.scatter(
        x_values[valid],
        y_values[valid],
        s=18,
        alpha=0.62,
        color=color,
        label=str(method) if show_label else "_nolegend_",
        edgecolors="none",
    )
    trend = fit_linear_trend_line(x, y)
    if trend is None:
        return
    line_x, line_y = trend
    ax.plot(line_x, line_y, color=color, linewidth=1.8, alpha=0.9, label="_nolegend_")


def style_error_axis(ax: plt.Axes, space: str, metric: str) -> None:
    ax.set_title(f"{space} / {'MAE' if metric == 'abs_error' else 'Relative error'}", loc="left", fontsize=11)
    ax.set_xlabel("W")
    ax.set_ylabel("sample MAE" if metric == "abs_error" else "relative error (%)")
    ax.grid(True, color="#e6e6e6", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def phase_diagram_space_keys(spaces: str | Iterable[str]) -> list[str]:
    if isinstance(spaces, str):
        if spaces == "both":
            return ["zy", "xy"]
        if spaces == "none":
            return []
        return [spaces]
    result: list[str] = []
    for space in spaces:
        if space == "both":
            result.extend(["zy", "xy"])
        elif space != "none":
            result.append(str(space))
    return [space for space in result if space in PHASE_DIAGRAM_REP_SPACES]


def build_phase_diagram_samples_wide(joined: pd.DataFrame) -> pd.DataFrame:
    key_cols = [
        "scope",
        "task_key",
        "task_id",
        "alloy_family",
        "dataset_name",
        "property",
        "method_short",
        "method",
        "model",
        "model_family",
        "fold_id",
        "ID",
        "sample_order",
        "test_set",
    ]
    value_cols = [
        "target_value",
        "true_value",
        "predicted_value",
        "signed_error",
        "abs_error",
        "relative_error_pct",
        "prediction_file",
        "source_split_dir",
        "join_mode",
    ]
    output_cols = [*key_cols, *PHASE_DIAGRAM_W_COLUMNS, *value_cols]
    if joined.empty:
        return pd.DataFrame(columns=output_cols)
    required = [*key_cols, "space", "sample_w_contribution"]
    missing = [column for column in required if column not in joined.columns]
    if missing:
        raise ValueError(f"Joined W/error table is missing phase diagram columns: {missing}")
    work = joined.copy()
    for column in [*key_cols, *value_cols]:
        if column not in work.columns:
            work[column] = np.nan
    work["sample_w_contribution"] = pd.to_numeric(work["sample_w_contribution"], errors="coerce")
    pivot = (
        work.groupby([*key_cols, "space"], dropna=False, sort=False)["sample_w_contribution"]
        .first()
        .unstack("space")
        .reset_index()
    )
    rename_map = {"X-space": "X_space_w", "Y-space": "Y_space_w", "Z-space": "Z_space_w"}
    pivot = pivot.rename(columns=rename_map)
    for column in PHASE_DIAGRAM_W_COLUMNS:
        if column not in pivot.columns:
            pivot[column] = np.nan
    meta = work.groupby(key_cols, dropna=False, sort=False)[value_cols].first().reset_index()
    wide = pivot.merge(meta, on=key_cols, how="left")
    for column in output_cols:
        if column not in wide.columns:
            wide[column] = np.nan
    return wide[output_cols].copy()


def phase_diagram_points(wide: pd.DataFrame, rep_space: str) -> pd.DataFrame:
    rep_col = rep_space.replace("-", "_").replace(" ", "_") + "_w"
    if rep_space in {"X-space", "Y-space", "Z-space"}:
        rep_col = rep_space[0] + "_space_w"
    required = [rep_col, "Y_space_w"]
    if wide.empty or any(column not in wide.columns for column in required):
        return pd.DataFrame(columns=wide.columns)
    result = wide.copy()
    result[rep_col] = pd.to_numeric(result[rep_col], errors="coerce")
    result["Y_space_w"] = pd.to_numeric(result["Y_space_w"], errors="coerce")
    valid = result[rep_col].notna() & result["Y_space_w"].notna()
    return result[valid].copy()


def phase_color_limit(values: pd.Series, percentile: float = 95.0) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    numeric = numeric[np.isfinite(numeric)]
    if numeric.empty:
        return 1.0
    percentile = min(max(float(percentile), 0.0), 100.0)
    try:
        value = float(np.nanpercentile(numeric.to_numpy(dtype=float), percentile, method="nearest"))
    except TypeError:
        value = float(np.nanpercentile(numeric.to_numpy(dtype=float), percentile, interpolation="nearest"))
    if not math.isfinite(value) or value <= 0:
        maximum = float(numeric.max())
        return maximum if math.isfinite(maximum) and maximum > 0 else 1.0
    return value


def ordered_models(frame: pd.DataFrame) -> list[str]:
    if frame.empty or "model" not in frame.columns:
        return []
    work = frame[["model", "model_family"]].drop_duplicates().copy()
    work["model_family"] = work["model_family"].fillna("").astype(str)
    work["model"] = work["model"].fillna("").astype(str)
    return work.sort_values(["model_family", "model"], kind="stable")["model"].tolist()


def phase_axis_label(rep_space: str) -> str:
    if rep_space == "Z-space":
        return "W_rep (Z-space)"
    if rep_space == "X-space":
        return "W_rep (X-space)"
    return f"W_rep ({rep_space})"


def severity_multiplier_ticks(values: pd.Series) -> list[float]:
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite[np.isfinite(finite) & finite.gt(0)]
    candidates = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    if finite.empty:
        return [1.0, 2.0, 5.0]
    upper = max(5.0, float(finite.max()) * 1.15)
    lower = min(1.0, float(finite.min()) * 0.85)
    return [tick for tick in candidates if lower <= tick <= upper]


def severity_axis_limits(values: pd.Series) -> tuple[float, float]:
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite[np.isfinite(finite) & finite.gt(0)]
    if finite.empty:
        return 0.8, 5.0
    lower = min(0.8, float(finite.min()) * 0.85)
    upper = max(1.2, float(finite.max()) * 1.15)
    return max(lower, 1e-3), upper


def plot_standard_yz_severity_figures(
    summary: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> pd.DataFrame:
    manifest_rows: list[dict[str, object]] = []
    if summary.empty:
        return pd.DataFrame(columns=FIGURE_MANIFEST_COLUMNS)
    work = summary.copy()
    work["Ry"] = pd.to_numeric(work["Ry"], errors="coerce")
    work["Rz"] = pd.to_numeric(work["Rz"], errors="coerce")
    work = work[work["Ry"].gt(0) & work["Rz"].gt(0)].copy()
    if work.empty:
        return pd.DataFrame(columns=FIGURE_MANIFEST_COLUMNS)
    for task_id, group in work.groupby("task_id", dropna=False, sort=True):
        task_id_text = clean_text(task_id) or "unknown_task"
        group = group.copy()
        group["_method_order"] = group["method_short"].map({method: index for index, method in enumerate(METHOD_ORDER)}).fillna(999)
        group = group.sort_values(["_method_order", "method_short"], kind="stable")
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        for _, row in group.iterrows():
            method = clean_text(row["method_short"])
            ax.scatter(
                row["Ry"],
                row["Rz"],
                s=90 if method == "RandCV" else 70,
                color=METHOD_COLORS.get(method, "#333333"),
                edgecolors="#333333" if method == "RandCV" else "#ffffff",
                linewidths=1.0 if method == "RandCV" else 0.5,
                alpha=0.86,
                label=method,
                zorder=3,
            )
            ax.annotate(
                method,
                (row["Ry"], row["Rz"]),
                xytext=(6, 5),
                textcoords="offset points",
                fontsize=8,
                color="#333333",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        x_ticks = severity_multiplier_ticks(group["Ry"])
        y_ticks = severity_multiplier_ticks(group["Rz"])
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels([f"{tick:g}x" for tick in x_ticks])
        ax.set_yticklabels([f"{tick:g}x" for tick in y_ticks])
        ax.set_xlim(*severity_axis_limits(group["Ry"]))
        ax.set_ylim(*severity_axis_limits(group["Rz"]))
        ax.axvline(1.0, color="#999999", linewidth=0.8, linestyle="--", zorder=1)
        ax.axhline(1.0, color="#999999", linewidth=0.8, linestyle="--", zorder=1)
        ax.set_xlabel("R_y = Wy / Wy_random")
        ax.set_ylabel("R_z = Wz / Wz_random")
        ax.set_title(f"{task_id_text} pure OOD Wy-Wz severity map", loc="left", fontsize=13, fontweight="bold")
        ax.grid(True, which="both", color="#e6e6e6", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="best", frameon=False, fontsize=8)
        fig.tight_layout()
        base = output_dir / "figures" / "yz_severity_map" / "by_task" / safe_filename(task_id_text) / "yz_severity_map"
        base.parent.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            path = base.parent / f"{base.name}.{fmt}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            manifest_rows.append(
                {
                    "figure_group": "yz_severity_map",
                    "task_id": task_id_text,
                    "model": "all_methods",
                    "method_short": "all_methods",
                    "rep_space": "Y/Z-space",
                    "figure": str(path),
                    "format": fmt,
                }
            )
        plt.close(fig)
    return pd.DataFrame(manifest_rows, columns=FIGURE_MANIFEST_COLUMNS)


def task_marker_map(tasks: Iterable[object]) -> dict[str, str]:
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8"]
    task_text = [clean_text(task) or "unknown_task" for task in pd.Series(list(tasks)).dropna().unique()]
    task_text = sorted(task_text)
    return {task: markers[index % len(markers)] for index, task in enumerate(task_text)}


def overview_label_rows(summary: pd.DataFrame, *, per_method: int = 2) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    work = summary.copy()
    work["Ry"] = pd.to_numeric(work["Ry"], errors="coerce")
    work["Rz"] = pd.to_numeric(work["Rz"], errors="coerce")
    work = work[work["Ry"].gt(0) & work["Rz"].gt(0) & ~work["method_short"].astype(str).eq("RandCV")].copy()
    if work.empty:
        return work
    rows: list[pd.Series] = []
    for method in ordered_methods(work["method_short"]):
        group = work[work["method_short"].astype(str).eq(method)].copy()
        if group.empty:
            continue
        candidates = [
            group.sort_values(["Ry", "task_id"], ascending=[False, True], kind="stable").head(1),
            group.sort_values(["Rz", "task_id"], ascending=[False, True], kind="stable").head(1),
        ]
        selected = pd.concat(candidates, ignore_index=False).drop_duplicates(subset=["task_id", "method_short"], keep="first")
        if len(selected) > per_method:
            selected = selected.assign(_score=selected[["Ry", "Rz"]].max(axis=1)).sort_values("_score", ascending=False, kind="stable").head(per_method)
            selected = selected.drop(columns="_score")
        rows.extend([row for _, row in selected.iterrows()])
    if not rows:
        return work.iloc[0:0].copy()
    return pd.DataFrame(rows).reset_index(drop=True)


CORE_YZ_PROTOCOLS = ("RandCV", "Extra.", "LOCO")


def core_protocol_yz_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    work = summary.copy()
    work["method_short"] = work["method_short"].map(normalize_standard_method)
    return work[work["method_short"].isin(CORE_YZ_PROTOCOLS)].copy().reset_index(drop=True)


def plot_standard_yz_severity_overview_to_file(
    summary: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
    dpi: int,
    *,
    figure_stem: str,
    figure_group: str,
    manifest_task_id: str,
    method_short: str,
    title: str,
) -> pd.DataFrame:
    manifest_rows: list[dict[str, object]] = []
    if summary.empty:
        return pd.DataFrame(columns=FIGURE_MANIFEST_COLUMNS)
    work = summary.copy()
    work["Ry"] = pd.to_numeric(work["Ry"], errors="coerce")
    work["Rz"] = pd.to_numeric(work["Rz"], errors="coerce")
    work = work[work["Ry"].gt(0) & work["Rz"].gt(0)].copy()
    if work.empty:
        return pd.DataFrame(columns=FIGURE_MANIFEST_COLUMNS)

    marker_by_task = task_marker_map(work["task_id"])
    fig, ax = plt.subplots(figsize=(10.5, 7.4))
    method_handles: dict[str, object] = {}
    task_handles: dict[str, object] = {}
    label_keys = {
        (clean_text(row["task_id"]) or "unknown_task", clean_text(row["method_short"]))
        for _, row in overview_label_rows(work).iterrows()
    }
    for _, row in work.iterrows():
        method = clean_text(row["method_short"])
        point_task_id = clean_text(row["task_id"]) or "unknown_task"
        marker = marker_by_task.get(point_task_id, "o")
        color = METHOD_COLORS.get(method, "#333333")
        point = ax.scatter(
            row["Ry"],
            row["Rz"],
            s=74 if method == "RandCV" else 58,
            color=color,
            marker=marker,
            alpha=0.78,
            edgecolors="#333333" if method == "RandCV" else "#ffffff",
            linewidths=0.8 if method == "RandCV" else 0.45,
            zorder=3,
        )
        method_handles.setdefault(method, point)
        task_handles.setdefault(
            point_task_id,
            Line2D([0], [0], marker=marker, linestyle="None", color="#555555", markerfacecolor="#555555", label=point_task_id),
        )
        if (point_task_id, method) in label_keys:
            ax.annotate(
                f"{point_task_id} {method}",
                (row["Ry"], row["Rz"]),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=7.5,
                color="#333333",
                arrowprops={"arrowstyle": "-", "color": "#777777", "linewidth": 0.5, "shrinkA": 0, "shrinkB": 3},
                bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.76},
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    x_ticks = severity_multiplier_ticks(work["Ry"])
    y_ticks = severity_multiplier_ticks(work["Rz"])
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([f"{tick:g}x" for tick in x_ticks])
    ax.set_yticklabels([f"{tick:g}x" for tick in y_ticks])
    ax.set_xlim(*severity_axis_limits(work["Ry"]))
    ax.set_ylim(*severity_axis_limits(work["Rz"]))
    ax.axvline(1.0, color="#999999", linewidth=0.8, linestyle="--", zorder=1)
    ax.axhline(1.0, color="#999999", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xlabel("R_y = Wy / Wy_random")
    ax.set_ylabel("R_z = Wz / Wz_random")
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, which="both", color="#e6e6e6", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ordered_method_labels = ordered_methods(method_handles.keys())
    method_legend = ax.legend(
        [method_handles[label] for label in ordered_method_labels if label in method_handles],
        [label for label in ordered_method_labels if label in method_handles],
        loc="upper left",
        frameon=False,
        fontsize=8,
        title="Protocol",
    )
    ax.add_artist(method_legend)
    task_labels = sorted(task_handles)
    ax.legend(
        [task_handles[label] for label in task_labels],
        task_labels,
        loc="lower right",
        frameon=False,
        fontsize=7,
        title="Task",
        ncol=2 if len(task_labels) > 6 else 1,
    )
    fig.tight_layout(rect=(0, 0, 0.98, 1))

    base = output_dir / "figures" / "yz_severity_map" / "overview" / figure_stem
    base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = base.parent / f"{base.name}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        manifest_rows.append(
                {
                    "figure_group": figure_group,
                    "task_id": manifest_task_id,
                    "model": "all_methods",
                    "method_short": method_short,
                    "rep_space": "Y/Z-space",
                    "figure": str(path),
                    "format": fmt,
            }
        )
    plt.close(fig)
    return pd.DataFrame(manifest_rows, columns=FIGURE_MANIFEST_COLUMNS)


def plot_standard_yz_severity_overview(
    summary: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> pd.DataFrame:
    return plot_standard_yz_severity_overview_to_file(
        summary,
        output_dir,
        formats,
        dpi,
        figure_stem="yz_severity_map_all_tasks",
        figure_group="yz_severity_map_overview",
        manifest_task_id="all_tasks",
        method_short="all_methods",
        title="All-task pure OOD Wy-Wz severity map",
    )


def plot_standard_yz_core_protocol_overview(
    summary: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> pd.DataFrame:
    return plot_standard_yz_severity_overview_to_file(
        core_protocol_yz_summary(summary),
        output_dir,
        formats,
        dpi,
        figure_stem="yz_severity_map_core_protocols",
        figure_group="yz_severity_map_core_protocols",
        manifest_task_id="all_tasks_core_protocols",
        method_short="RandCV+Extra.+LOCO",
        title="All-task pure OOD Wy-Wz severity map - core protocols",
    )


def plot_phase_diagram_figures(
    wide: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
    dpi: int,
    spaces: str | Iterable[str] = "both",
    color_percentile: float = 95.0,
) -> pd.DataFrame:
    manifest_rows: list[dict[str, object]] = []
    if wide.empty:
        return pd.DataFrame(columns=FIGURE_MANIFEST_COLUMNS)
    for space_key in phase_diagram_space_keys(spaces):
        rep_space, rep_col, figure_stem = PHASE_DIAGRAM_REP_SPACES[space_key]
        points = phase_diagram_points(wide, rep_space)
        if points.empty:
            continue
        for (task_id, method), group in points.groupby(["task_id", "method_short"], dropna=False, sort=True):
            group = group.copy()
            models = ordered_models(group)
            if not models:
                continue
            ncols = min(3, len(models))
            nrows = int(math.ceil(len(models) / ncols))
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(4.2 * ncols, 3.6 * nrows),
                squeeze=False,
                constrained_layout=True,
            )
            fig.suptitle(f"{task_id} | {method} | {rep_space} vs Y-space", fontsize=14, fontweight="bold")
            vmax = phase_color_limit(group["relative_error_pct"], color_percentile)
            norm = Normalize(vmin=0.0, vmax=vmax)
            cmap = plt.get_cmap("viridis")
            scatter = None
            axes_flat = list(axes.ravel())
            for ax, model in zip(axes_flat, models):
                model_group = group[group["model"].astype(str).eq(str(model))]
                x = pd.to_numeric(model_group[rep_col], errors="coerce")
                y = pd.to_numeric(model_group["Y_space_w"], errors="coerce")
                color_values = pd.to_numeric(model_group["relative_error_pct"], errors="coerce").clip(upper=vmax)
                scatter = ax.scatter(
                    x,
                    y,
                    c=color_values,
                    cmap=cmap,
                    norm=norm,
                    s=18,
                    alpha=0.72,
                    edgecolors="none",
                    rasterized=True,
                )
                ax.set_title(str(model), loc="left", fontsize=10)
                ax.set_xlabel(phase_axis_label(rep_space))
                ax.set_ylabel("W_target (Y-space)")
                ax.grid(True, color="#e6e6e6", linewidth=0.7)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            for ax in axes_flat[len(models) :]:
                ax.set_visible(False)
            if scatter is not None:
                colorbar = fig.colorbar(scatter, ax=axes_flat[: len(models)], fraction=0.035, pad=0.02)
                colorbar.set_label(f"relative error (%) clipped at p{color_percentile:g}")
            base = (
                output_dir
                / "figures"
                / "phase_diagrams"
                / "by_task_method"
                / safe_filename(task_id)
                / safe_filename(method)
                / figure_stem
            )
            base.parent.mkdir(parents=True, exist_ok=True)
            for fmt in formats:
                path = base.parent / f"{base.name}.{fmt}"
                fig.savefig(path, dpi=dpi, bbox_inches="tight")
                manifest_rows.append(
                    {
                        "figure_group": "phase_diagram",
                        "task_id": task_id,
                        "model": "all_models",
                        "method_short": method,
                        "rep_space": rep_space,
                        "figure": str(path),
                        "format": fmt,
                    }
                )
            plt.close(fig)
    return pd.DataFrame(manifest_rows, columns=FIGURE_MANIFEST_COLUMNS)


def plot_task_model_figures(joined: pd.DataFrame, output_dir: Path, formats: Iterable[str], dpi: int) -> pd.DataFrame:
    manifest_rows: list[dict[str, object]] = []
    if joined.empty:
        return pd.DataFrame(columns=FIGURE_MANIFEST_COLUMNS)
    for (task_id, model), group in joined.groupby(["task_id", "model"], dropna=False, sort=True):
        fig, axes = plt.subplots(3, 2, figsize=(12.5, 11.0), sharex=False)
        fig.suptitle(f"{task_id} | {model}", fontsize=15, fontweight="bold")
        for row_idx, space in enumerate(SPACE_ORDER):
            space_group = group[group["space"].astype(str).eq(space)]
            for col_idx, metric in enumerate(["abs_error", "relative_error_pct"]):
                ax = axes[row_idx, col_idx]
                for method in ordered_methods(space_group["method_short"]):
                    method_group = space_group[space_group["method_short"].eq(method)]
                    draw_method_points_and_trend(ax, method_group, metric, method, show_label=True)
                style_error_axis(ax, space, metric)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 7), frameon=False)
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        base = output_dir / "figures" / "by_task_model" / safe_filename(task_id) / f"{safe_filename(model)}_w_error_scatter"
        base.parent.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            path = base.parent / f"{base.name}.{fmt}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            manifest_rows.append(
                {
                    "figure_group": "by_task_model",
                    "task_id": task_id,
                    "model": model,
                    "method_short": np.nan,
                    "rep_space": np.nan,
                    "figure": str(path),
                    "format": fmt,
                }
            )
        plt.close(fig)
    return pd.DataFrame(manifest_rows, columns=FIGURE_MANIFEST_COLUMNS)


def plot_method_figures(joined: pd.DataFrame, output_dir: Path, formats: Iterable[str], dpi: int) -> pd.DataFrame:
    manifest_rows: list[dict[str, object]] = []
    if joined.empty or "method_short" not in joined.columns:
        return pd.DataFrame(columns=FIGURE_MANIFEST_COLUMNS)
    method_joined = joined[joined["method_short"].notna()].copy()
    if method_joined.empty:
        return pd.DataFrame(columns=FIGURE_MANIFEST_COLUMNS)
    for (method, task_id, model), group in method_joined.groupby(["method_short", "task_id", "model"], dropna=False, sort=True):
        method = str(method)
        fig, axes = plt.subplots(3, 2, figsize=(12.5, 11.0), sharex=False)
        fig.suptitle(f"{task_id} | {model} | {method}", fontsize=15, fontweight="bold")
        for row_idx, space in enumerate(SPACE_ORDER):
            space_group = group[group["space"].astype(str).eq(space)]
            for col_idx, metric in enumerate(["abs_error", "relative_error_pct"]):
                ax = axes[row_idx, col_idx]
                draw_method_points_and_trend(ax, space_group, metric, method, show_label=True)
                style_error_axis(ax, space, metric)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=1, frameon=False)
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        base = (
            output_dir
            / "figures"
            / "by_method"
            / safe_filename(method)
            / safe_filename(task_id)
            / f"{safe_filename(model)}_w_error_scatter"
        )
        base.parent.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            path = base.parent / f"{base.name}.{fmt}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            manifest_rows.append(
                {
                    "figure_group": "by_method",
                    "task_id": task_id,
                    "model": model,
                    "method_short": method,
                    "rep_space": np.nan,
                    "figure": str(path),
                    "format": fmt,
                }
            )
        plt.close(fig)
    return pd.DataFrame(manifest_rows, columns=FIGURE_MANIFEST_COLUMNS)


def write_report(output_dir: Path, joined: pd.DataFrame, coverage: pd.DataFrame, inventory: pd.DataFrame, figures: pd.DataFrame) -> None:
    lines = [
        "# W-error relationship analysis",
        "",
        f"- Joined sample rows: {len(joined)}",
        f"- Figure files: {len(figures)}",
        f"- Prediction sources scanned: {len(inventory)}",
        "",
        "## Join coverage",
        "",
    ]
    if coverage.empty:
        lines.append("No join coverage rows were produced.")
    else:
        total_w = int(pd.to_numeric(coverage["w_rows"], errors="coerce").fillna(0).sum())
        matched = int(pd.to_numeric(coverage["matched_rows"], errors="coerce").fillna(0).sum())
        lines.append(f"- Total W rows considered across model groups: {total_w}")
        lines.append(f"- Matched W rows across model groups: {matched}")
        weak = coverage[pd.to_numeric(coverage["matched_rows"], errors="coerce").fillna(0).eq(0)]
        lines.append(f"- Zero-match model groups: {len(weak)}")
    lines.extend(["", "## Files", ""])
    lines.append("- `csv/w_error_samples_long.csv`")
    lines.append("- `csv/w_error_correlations.csv`")
    lines.append("- `csv/w_error_join_coverage.csv`")
    lines.append("- `csv/phase_diagram_samples_wide.csv`")
    lines.append("- `csv/prediction_source_inventory.csv`")
    lines.append("- `figures/by_task_model/`")
    lines.append("- `figures/by_method/`")
    lines.append("- `figures/phase_diagrams/by_task_method/`")
    (output_dir / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one_output(
    w_table: pd.DataFrame,
    errors: pd.DataFrame,
    inventory: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    dpi: int,
    max_relative_error_pct: float | None = None,
    phase_diagram_spaces: str | Iterable[str] = "both",
    phase_diagram_color_percentile: float = 95.0,
    standard_yz_severity: pd.DataFrame | None = None,
    csv_only: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    errors = filter_errors_by_relative_error(errors, max_relative_error_pct)
    joined, coverage = join_w_and_errors(w_table, errors)
    correlations = build_correlations(joined)
    phase_wide = build_phase_diagram_samples_wide(joined.drop(columns=["__w_row_id"], errors="ignore"))
    if csv_only:
        figures = pd.DataFrame(columns=FIGURE_MANIFEST_COLUMNS)
    else:
        figures = pd.concat(
            [
                plot_task_model_figures(joined, output_dir, formats, dpi),
                plot_method_figures(joined, output_dir, formats, dpi),
                plot_phase_diagram_figures(
                    phase_wide,
                    output_dir,
                    formats,
                    dpi,
                    spaces=phase_diagram_spaces,
                    color_percentile=phase_diagram_color_percentile,
                ),
                plot_standard_yz_severity_figures(
                    standard_yz_severity if standard_yz_severity is not None else pd.DataFrame(),
                    output_dir,
                    formats,
                    dpi,
                ),
                plot_standard_yz_severity_overview(
                    standard_yz_severity if standard_yz_severity is not None else pd.DataFrame(),
                    output_dir,
                    formats,
                    dpi,
                ),
            ],
            ignore_index=True,
        )
    write_csv(joined.drop(columns=["__w_row_id"], errors="ignore"), output_dir / "csv" / "w_error_samples_long.csv")
    write_csv(correlations, output_dir / "csv" / "w_error_correlations.csv")
    write_csv(coverage, output_dir / "csv" / "w_error_join_coverage.csv")
    write_csv(phase_wide, output_dir / "csv" / "phase_diagram_samples_wide.csv")
    if standard_yz_severity is not None:
        write_csv(standard_yz_severity, output_dir / "csv" / "standard_yz_severity_summary.csv")
    write_csv(inventory, output_dir / "csv" / "prediction_source_inventory.csv")
    write_csv(figures, output_dir / "csv" / "figure_manifest.csv")
    write_report(output_dir, joined, coverage, inventory, figures)


def run_standard(args: argparse.Namespace) -> None:
    w_path = Path(args.standard_w_table)
    if not w_path.exists():
        raise FileNotFoundError(f"Missing standard OOD W table: {w_path}")
    w_table = prepare_w_table(read_csv(w_path), scope="ood")
    yz_severity = build_standard_yz_severity_summary(w_table)
    errors, inventory = collect_prediction_errors("ood", case_contains=args.case_contains, max_sources=args.max_sources)
    run_one_output(
        w_table,
        errors,
        inventory,
        Path(args.ood_output),
        args.formats,
        args.dpi,
        args.max_relative_error_pct,
        args.phase_diagram_spaces,
        args.phase_diagram_color_percentile,
        standard_yz_severity=yz_severity,
        csv_only=args.csv_only,
    )


def run_hybrid(args: argparse.Namespace) -> None:
    cache_root = hybrid_cache_run_root(Path(args.hybrid_cache_root), args.case_contains)
    embedding_dir = Path(args.embedding_data_dir)
    for subset in HYBRID_SUBSETS:
        w_path = ensure_hybrid_w_table(
            subset,
            cache_root,
            embedding_dir,
            force=args.force_recompute_hybrid_w,
            case_contains=args.case_contains,
        )
        w_table = prepare_w_table(read_csv(w_path), scope="hybrid", hybrid_subset=subset)
        errors, inventory = collect_prediction_errors(
            "hybrid",
            hybrid_subset=subset,
            case_contains=args.case_contains,
            max_sources=args.max_sources,
        )
        run_one_output(
            w_table,
            errors,
            inventory,
            Path(args.hybrid_output) / subset,
            args.formats,
            args.dpi,
            args.max_relative_error_pct,
            args.phase_diagram_spaces,
            args.phase_diagram_color_percentile,
            csv_only=args.csv_only,
        )


def main() -> None:
    args = parse_args()
    if args.scope in {"ood", "both"}:
        run_standard(args)
    if args.scope in {"hybrid", "both"}:
        run_hybrid(args)


if __name__ == "__main__":
    main()
