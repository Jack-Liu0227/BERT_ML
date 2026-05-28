from __future__ import annotations

import argparse
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SUMMARY_ROOT = Path(
    r"D:\XJTU\ImportantFile\auto-design-alloy\BERT_ML\output\new_results_withuncertainty\all_alloys_traditional_models_summary\01_alloy_cases"
)
DEFAULT_PRIOR_ROOT = Path(
    r"D:\XJTU\ImportantFile\auto-design-alloy\fewshot-guided\PriorModel"
)


@dataclass(frozen=True)
class AlloyConfig:
    alloy_dirname: str
    dataset_dirname: str
    prior_dirname: str


ALLOY_CONFIGS = [
    AlloyConfig("Ti", "titanium", "Ti"),
    AlloyConfig("HEA_half", "hea", "HEA_half"),
    AlloyConfig("HEA_corrosion", "Pitting_potential_data_xiongjie", "HEA_corrosion"),
    AlloyConfig("Al", "aluminum", "Al"),
]

MODE_MAP = {
    "final": "final",
    "mean": "closest_mean",
    "global": "global_mean",
}

MODEL_KEY_MAP = {
    "CatBoost": "catboost",
    "XGB": "xgboost",
    "LightGBM": "lightgbm",
    "RF": "rf",
    "MLP": "mlp",
}

DIAGNOSTIC_MAP = {
    "final": "final_diagnostic.png",
    "closest_mean": "closest_mean_diagnostic.png",
    "global_mean": "global_mean_diagnostic.png",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy selected traditional models into fewshot-guided PriorModel folders."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=sorted(MODE_MAP),
        help="Copy mode: final, mean, or global.",
    )
    parser.add_argument(
        "--summary-root",
        type=Path,
        default=DEFAULT_SUMMARY_ROOT,
        help="Root of all_alloys_traditional_models_summary/01_alloy_cases.",
    )
    parser.add_argument(
        "--prior-root",
        type=Path,
        default=DEFAULT_PRIOR_ROOT,
        help="Root of fewshot-guided/PriorModel.",
    )
    return parser.parse_args()


def find_single_model_dir(mode_dir: Path) -> Path:
    model_dirs = sorted(path for path in mode_dir.iterdir() if path.is_dir())
    if len(model_dirs) != 1:
        raise RuntimeError(
            f"Expected exactly one model directory under {mode_dir}, found {len(model_dirs)}."
        )
    return model_dirs[0]


def resolve_model_key(model_name: str) -> str:
    if model_name not in MODEL_KEY_MAP:
        raise RuntimeError(f"Unsupported model name: {model_name}")
    return MODEL_KEY_MAP[model_name]


def find_existing_file(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "None of the expected files exist: " + ", ".join(str(path) for path in candidates)
    )


def resolve_model_file(mode: str, model_dir: Path, model_name: str) -> Path:
    if mode == "final":
        candidates = [model_dir / "final_best_model.pkl"]
        if model_name == "XGB":
            candidates.insert(0, model_dir / "final_best_model.json")
        return find_existing_file(candidates)

    if mode == "closest_mean":
        eval_dir = model_dir / "closest_to_mean_evaluation"
        candidates = [eval_dir / "closest_to_mean_model.pkl"]
        if model_name == "XGB":
            candidates.insert(0, eval_dir / "closest_to_mean_model.json")
        return find_existing_file(candidates)

    if mode == "global_mean":
        eval_dir = model_dir / "closest_to_global_mean_trial_fold"
        candidates = [eval_dir / "closest_to_mean_model.pkl"]
        if model_name == "XGB":
            candidates.insert(0, eval_dir / "closest_to_mean_model.json")
        return find_existing_file(candidates)

    raise RuntimeError(f"Unsupported normalized mode: {mode}")


def resolve_scaler_file(model_dir: Path, name: str) -> Path:
    scaler_path = model_dir / name
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler file: {scaler_path}")
    return scaler_path


def resolve_diagnostic_file(property_dir: Path, mode: str) -> Path:
    diagnostic_path = property_dir / "selected_model_artifacts" / DIAGNOSTIC_MAP[mode]
    if not diagnostic_path.exists():
        raise FileNotFoundError(f"Missing diagnostic file: {diagnostic_path}")
    return diagnostic_path


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for _ in range(3):
        try:
            shutil.copy2(src, dst)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.5)
    if last_error is not None:
        raise last_error


def iter_property_dirs(dataset_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in dataset_dir.iterdir()
        if path.is_dir() and path.name != "comparisons"
    )


def process_property(
    property_dir: Path,
    prior_dir: Path,
    normalized_mode: str,
) -> tuple[str, str]:
    mode_dir = property_dir / "selected_model_source" / normalized_mode
    if not mode_dir.exists():
        raise FileNotFoundError(f"Missing mode directory: {mode_dir}")

    model_dir = find_single_model_dir(mode_dir)
    model_name = model_dir.name
    model_key = resolve_model_key(model_name)
    property_name = property_dir.name

    model_file = resolve_model_file(normalized_mode, model_dir, model_name)
    scaler_x = resolve_scaler_file(model_dir, "scaler_X.pkl")
    scaler_y = resolve_scaler_file(model_dir, "scaler_y.pkl")
    diagnostic = resolve_diagnostic_file(property_dir, normalized_mode)

    model_dst = prior_dir / f"{property_name}_{model_key}_model{model_file.suffix}"
    scaler_x_dst = prior_dir / f"{property_name}_{model_key}_scaler_X.pkl"
    scaler_y_dst = prior_dir / f"{property_name}_{model_key}_scaler_y.pkl"
    diagnostic_dst = prior_dir / "plots" / f"{property_name}_{model_key}.png"

    copy_file(model_file, model_dst)
    copy_file(scaler_x, scaler_x_dst)
    copy_file(scaler_y, scaler_y_dst)
    copy_file(diagnostic, diagnostic_dst)

    print(f"[INFO] {property_name}: {model_name}")
    print(f"       model      {model_file} -> {model_dst}")
    print(f"       scaler_X   {scaler_x} -> {scaler_x_dst}")
    print(f"       scaler_y   {scaler_y} -> {scaler_y_dst}")
    print(f"       diagnostic {diagnostic} -> {diagnostic_dst}")
    return property_name, model_name


def main() -> int:
    args = parse_args()
    normalized_mode = MODE_MAP[args.mode]
    summary_root = args.summary_root
    prior_root = args.prior_root

    if not summary_root.exists():
        print(f"[ERROR] Summary root does not exist: {summary_root}", file=sys.stderr)
        return 1

    copied = 0
    warnings = 0

    print(f"[INFO] Requested mode: {args.mode}")
    print(f"[INFO] Normalized mode: {normalized_mode}")
    print(f"[INFO] Summary root: {summary_root}")
    print(f"[INFO] Prior root: {prior_root}")

    for config in ALLOY_CONFIGS:
        dataset_dir = summary_root / config.alloy_dirname / config.dataset_dirname
        prior_dir = prior_root / config.prior_dirname
        print(f"[INFO] Processing {config.alloy_dirname}/{config.dataset_dirname}")

        if not dataset_dir.exists():
            print(f"[WARN] Dataset directory does not exist: {dataset_dir}")
            warnings += 1
            continue

        property_dirs = iter_property_dirs(dataset_dir)
        if not property_dirs:
            print(f"[WARN] No property directories found: {dataset_dir}")
            warnings += 1
            continue

        for property_dir in property_dirs:
            try:
                process_property(property_dir, prior_dir, normalized_mode)
                copied += 1
            except Exception as exc:
                print(f"[WARN] Failed for {property_dir}: {exc}")
                warnings += 1

    print(f"[INFO] Completed mode {normalized_mode}")
    print(f"[INFO] Properties copied: {copied}")
    print(f"[INFO] Warnings: {warnings}")
    return 0 if copied > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
