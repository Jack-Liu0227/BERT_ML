from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from .common import (
    REPO_ROOT,
    clean_text,
    normalize_method,
    normalize_model,
    property_short,
    read_json,
    source_uses_global_ood_splits,
    task_id,
    task_key,
)

DEFAULT_FEWSHOT_GPT54_ROOT = REPO_ROOT.parent / "fewshot-guided" / "output" / "ood" / "k5"

ALLOWED_TABPFN_OOD_ROOTS = {
    "ood_results_tabpfn-2.5-plus-numeric",
    "ood_results_tabpfn-2.5-plus-text",
}
ALLOWED_TABPFN_HYBRID_ROOTS = {
    "ood_hybrid_results_tabpfn-2.5-plus-numeric",
    "ood_hybrid_results_tabpfn-2.5-plus-text",
}
ALLOWED_TABPFN_OOD_METHOD_DIRS = {
    "loco_k5",
    "random_cv_baseline",
    "target_extrapolation",
    "sparse_x_single_k5",
    "sparse_x_cluster_k5",
    "sparse_y_single_k5",
    "sparse_y_cluster_k5",
}


@dataclass(frozen=True)
class ResultSource:
    scope: str
    result_root: Path
    prediction_file: Path
    fold_dir: Path
    split_dir: Path
    split_summary_file: Path
    train_file: Path
    combined_test_file: Path
    family: str
    dataset: str
    property_name: str
    method_dir: str
    method: str
    model: str
    fold_id: str
    task_key: str
    task_id: str
    source_kind: str = "bert_ml"


def default_roots(scope: str) -> list[Path]:
    pattern = "ood_hybrid_results*" if scope == "hybrid" else "ood_results*"
    return sorted(
        path
        for path in (REPO_ROOT / "output").glob(pattern)
        if path.is_dir() and is_allowed_tabpfn_result_root(path, scope=scope)
    )


def tabpfn_result_root_name(path: Path) -> str | None:
    for part in path.parts:
        lower = part.lower()
        if lower.startswith(("ood_results_tabpfn", "ood_hybrid_results_tabpfn")):
            return lower
    return None


def is_allowed_tabpfn_result_root(path: Path, *, scope: str) -> bool:
    tabpfn_root = tabpfn_result_root_name(path)
    if tabpfn_root is None:
        return True
    if scope == "hybrid":
        return tabpfn_root in ALLOWED_TABPFN_HYBRID_ROOTS
    return tabpfn_root in ALLOWED_TABPFN_OOD_ROOTS


def is_allowed_tabpfn_source(source: "ResultSource") -> bool:
    tabpfn_root = tabpfn_result_root_name(source.result_root)
    if tabpfn_root is None:
        return True
    if not is_allowed_tabpfn_result_root(source.result_root, scope=source.scope):
        return False
    if source.scope == "ood":
        return source.method_dir.lower() in ALLOWED_TABPFN_OOD_METHOD_DIRS
    return True


def is_formal_prediction_file(path: Path) -> bool:
    parts_lower = [part.lower() for part in path.parts]
    if "optuna_trials" in parts_lower or any(part.startswith("trial_") for part in parts_lower):
        return False
    if path.name not in {"best_model_all_predictions.csv", "all_predictions.csv"}:
        return False
    if path.parent.name.lower() != "predictions":
        return False
    return True


def find_fold_dir(prediction_file: Path) -> Path | None:
    for parent in prediction_file.parents:
        if (parent / "split_data" / "split_summary.json").exists():
            return parent
    return None


def _metadata_from_relative_parts(scope: str, root: Path, fold_dir: Path, split_summary: dict[str, object]) -> tuple[str, str, str, str, str]:
    rel_parts = list(fold_dir.relative_to(root).parts)
    if "folds" in rel_parts:
        fold_index = rel_parts.index("folds")
        before = rel_parts[:fold_index]
    else:
        before = rel_parts

    method_like = (
        "loco",
        "loco_k5",
        "random_cv",
        "random_cv_baseline",
        "target_extrapolation",
        "sparse_x",
        "sparse_y",
        "hybrid_extrapolation",
    )

    if before and before[0].lower().startswith("experiment"):
        family = before[1] if len(before) > 1 else ""
        dataset = before[2] if len(before) > 2 else ""
        property_name = before[3] if len(before) > 3 else clean_text(split_summary.get("split_target_col"))
        method_dir = before[4] if len(before) > 4 else clean_text(split_summary.get("split_strategy"))
        model_hint = before[5] if len(before) > 5 else ""
    elif before and before[0].lower().startswith(method_like) and len(before) >= 4:
        method_dir = before[0]
        family = before[1]
        dataset = before[2]
        property_name = before[3]
        model_hint = root.name
    elif len(before) >= 4:
        family = before[0]
        dataset = before[1]
        property_name = before[2]
        method_dir = before[3]
        model_hint = before[4] if len(before) > 4 else ""
    elif len(before) == 1 and len(root.parts) >= 4:
        family = root.parts[-4]
        dataset = root.parts[-3]
        property_name = root.parts[-2]
        method_dir = root.parts[-1]
        model_hint = before[0]
    else:
        family = clean_text(split_summary.get("alloy_family"))
        dataset = clean_text(split_summary.get("dataset_name"))
        property_name = clean_text(split_summary.get("split_target_col"))
        method_dir = clean_text(split_summary.get("split_strategy"))
        model_hint = ""

    return family, dataset, property_name, method_dir, model_hint


def infer_model(root: Path, prediction_file: Path, model_hint: str) -> str:
    parts = list(prediction_file.parts)
    lower_parts = [part.lower() for part in parts]
    if "model_comparison" in lower_parts:
        index = lower_parts.index("model_comparison")
        if index + 1 < len(parts):
            return normalize_model(parts[index + 1])
    if root.name.lower().startswith(("ood_results_tabpfn", "ood_hybrid_results_tabpfn")):
        return normalize_model(root.name.replace("ood_results_", "").replace("ood_hybrid_results_", ""))
    return normalize_model(model_hint)


def make_source(scope: str, root: Path, prediction_file: Path) -> ResultSource | None:
    fold_dir = find_fold_dir(prediction_file)
    if fold_dir is None:
        return None
    split_dir = fold_dir / "split_data"
    if source_uses_global_ood_splits(split_dir):
        return None
    split_summary_file = split_dir / "split_summary.json"
    split_summary = read_json(split_summary_file)
    train_label = clean_text(split_summary.get("train_label")) or "train"
    test_label = clean_text(split_summary.get("test_label")) or ("test_hybrid_combined" if scope == "hybrid" else "test")
    train_file = split_dir / f"{train_label}.csv"
    combined_test_file = split_dir / f"{test_label}.csv"
    if not train_file.exists() or not combined_test_file.exists():
        return None

    family, dataset, property_name, method_dir, model_hint = _metadata_from_relative_parts(
        scope, root, fold_dir, split_summary
    )
    model = infer_model(root, prediction_file, model_hint)
    method = normalize_method(method_dir or split_summary.get("split_strategy"), scope=scope)
    property_name = property_name or clean_text(split_summary.get("split_target_col"))
    fold_id = fold_dir.name
    if fold_dir.parent.name.lower() != "folds":
        fold_id = "all"
    return ResultSource(
        scope=scope,
        result_root=root,
        prediction_file=prediction_file,
        fold_dir=fold_dir,
        split_dir=split_dir,
        split_summary_file=split_summary_file,
        train_file=train_file,
        combined_test_file=combined_test_file,
        family=family,
        dataset=dataset,
        property_name=property_name,
        method_dir=method_dir,
        method=method,
        model=model,
        fold_id=fold_id,
        task_key=task_key(family, dataset, property_name),
        task_id=task_id(family, property_name),
    )


FEWSHOT_TASK_MAP = {
    "al": ("Al", "aluminum"),
    "hea": ("HEA", "hea"),
    "steel": ("Steel", "steel"),
    "ti": ("Ti", "titanium"),
}

FEWSHOT_PROPERTY_MAP = {
    "el": "El(%)",
    "uts": "UTS(MPa)",
    "ys": "YS(MPa)",
}


def _fewshot_scope_from_root_dir(root_dir: str) -> str:
    return "hybrid" if "hybrid_extrapolation" in root_dir.lower() else "ood"


def _fewshot_fold_id(prediction_file: Path) -> str:
    for parent in prediction_file.parents:
        if parent.name.lower().startswith("fold_"):
            return parent.name
    return "all"


def _fewshot_summary_file(case_dir: Path) -> Path | None:
    matches = sorted(case_dir.glob("*_split_summary.json"))
    if not matches:
        return None
    return matches[0]


def _fewshot_task_metadata(task_dir: str, summary: dict[str, object]) -> tuple[str, str, str]:
    parts = task_dir.lower().split("_")
    if parts[:2] == ["matbench", "steel"]:
        return "MatbenchSteels", "matbench_steels_ood", FEWSHOT_PROPERTY_MAP.get(parts[2], clean_text(summary.get("split_target_col")))
    family_key = parts[0] if parts else ""
    property_key = parts[1] if len(parts) > 1 else ""
    family, dataset = FEWSHOT_TASK_MAP.get(family_key, (family_key, family_key))
    property_name = FEWSHOT_PROPERTY_MAP.get(property_key, clean_text(summary.get("split_target_col")))
    return family, dataset, property_name


def _fewshot_case_parts(root: Path, prediction_file: Path) -> tuple[str, str] | None:
    rel_parts = prediction_file.relative_to(root).parts
    if len(rel_parts) < 5:
        return None
    return rel_parts[0], rel_parts[1]


def make_fewshot_gpt54_source(root: Path, prediction_file: Path, *, scope: str) -> ResultSource | None:
    if prediction_file.name != "predictions.csv":
        return None
    if prediction_file.parent.name.lower() != "gpt-5.4":
        return None
    case_parts = _fewshot_case_parts(root, prediction_file)
    if case_parts is None:
        return None
    root_dir, task_dir = case_parts
    source_scope = _fewshot_scope_from_root_dir(root_dir)
    if source_scope != scope:
        return None

    case_dir = prediction_file.parent
    split_summary_file = _fewshot_summary_file(case_dir)
    if split_summary_file is None:
        return None
    split_summary = read_json(split_summary_file)
    train_file = case_dir / "train_set.csv"
    combined_test_file = case_dir / "test_set.csv"
    if not train_file.exists() or not combined_test_file.exists():
        return None
    family, dataset, property_name = _fewshot_task_metadata(task_dir, split_summary)
    method_dir = clean_text(split_summary.get("split_strategy")) or root_dir
    method = normalize_method(method_dir, scope=scope)
    return ResultSource(
        scope=scope,
        result_root=root,
        prediction_file=prediction_file,
        fold_dir=case_dir,
        split_dir=case_dir,
        split_summary_file=split_summary_file,
        train_file=train_file,
        combined_test_file=combined_test_file,
        family=family,
        dataset=dataset,
        property_name=property_name,
        method_dir=method_dir,
        method=method,
        model="GPT-5.4",
        fold_id=_fewshot_fold_id(prediction_file),
        task_key=task_key(family, dataset, property_name),
        task_id=task_id(family, property_name),
        source_kind="fewshot_gpt54",
    )


def discover_sources(
    *,
    scope: str,
    roots: list[Path] | None = None,
    case_contains: str | None = None,
    limit: int | None = None,
) -> list[ResultSource]:
    roots = default_roots(scope) if roots is None else roots
    sources: list[ResultSource] = []
    seen: set[Path] = set()
    for root in roots:
        if not is_allowed_tabpfn_result_root(root, scope=scope):
            continue
        prediction_files = iter_formal_prediction_files(root)
        for prediction_file in prediction_files:
            if prediction_file in seen or not is_formal_prediction_file(prediction_file):
                continue
            seen.add(prediction_file)
            source = make_source(scope, root, prediction_file)
            if source is None:
                continue
            if not is_allowed_tabpfn_source(source):
                continue
            haystack = " ".join(
                [
                    str(source.prediction_file),
                    source.task_key,
                    source.model,
                    source.method,
                    source.fold_id,
                ]
            ).lower()
            if case_contains and case_contains.lower() not in haystack:
                continue
            sources.append(source)
            if limit is not None and len(sources) >= limit:
                return sources
    return sources


def discover_fewshot_gpt54_sources(
    root: Path = DEFAULT_FEWSHOT_GPT54_ROOT,
    *,
    scope: str,
    case_contains: str | None = None,
    limit: int | None = None,
) -> list[ResultSource]:
    root = Path(root)
    if not root.exists():
        return []
    sources: list[ResultSource] = []
    for prediction_file in sorted(root.rglob("gpt-5.4/predictions.csv")):
        source = make_fewshot_gpt54_source(root, prediction_file, scope=scope)
        if source is None:
            continue
        haystack = " ".join(
            [
                str(source.prediction_file),
                source.task_key,
                source.model,
                source.method,
                source.fold_id,
            ]
        ).lower()
        if case_contains and case_contains.lower() not in haystack:
            continue
        sources.append(source)
        if limit is not None and len(sources) >= limit:
            return sources
    return sources


def iter_formal_prediction_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if dirname.lower() != "optuna_trials" and not dirname.lower().startswith("trial_")
        ]
        current = Path(dirpath)
        if current.name.lower() != "predictions":
            continue
        for filename in ("best_model_all_predictions.csv", "all_predictions.csv"):
            if filename in filenames:
                yield current / filename
