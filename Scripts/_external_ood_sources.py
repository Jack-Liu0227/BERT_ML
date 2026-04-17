from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(pd.Series(values, dtype="float64").std())


def _match_case_metadata(case_name: str, case_map: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    normalized = str(case_name).strip().lower()
    for prefix, metadata in case_map.items():
        if normalized.startswith(str(prefix).strip().lower()):
            return metadata
    return None


def _metric_triplet_from_metrics_json(metrics_path: Path, property_name: str) -> tuple[float, float, float]:
    payload = _read_json(metrics_path)
    metric_block = payload.get(property_name, payload)
    mae = float(metric_block["mae"])
    rmse = float(metric_block["rmse"])
    r2 = float(metric_block["r2"])
    return r2, mae, rmse


def _build_single_run_row(
    *,
    method_name: str,
    case_metadata: dict[str, Any],
    model_dir: Path,
    model_family: str,
    model_name: str,
    display_label: str,
    source_root: Path,
) -> dict[str, Any]:
    property_name = str(case_metadata["property"])
    metrics_path = model_dir / "metrics.json"
    predictions_path = model_dir / "predictions.csv"
    test_set_path = model_dir / "test_set.csv"
    r2, mae, rmse = _metric_triplet_from_metrics_json(metrics_path, property_name)
    return {
        "alloy_family": case_metadata["alloy_family"],
        "dataset_name": case_metadata["dataset_name"],
        "property": property_name,
        "ood_method": method_name,
        "model_family": model_family,
        "model": model_name,
        "display_label": display_label,
        "model_dir": str(model_dir.resolve()),
        "source_dir": str(source_root.resolve()),
        "trial_count": 1,
        "fold_count": 1,
        "summary_test_r2": r2,
        "summary_test_r2_std": 0.0,
        "summary_test_mae": mae,
        "summary_test_mae_std": 0.0,
        "summary_test_rmse": rmse,
        "summary_test_rmse_std": 0.0,
        "representative_selection_mode": "external_single_run",
        "representative_trial_id": pd.NA,
        "representative_fold": pd.NA,
        "representative_test_r2": r2,
        "representative_test_mae": mae,
        "representative_test_rmse": rmse,
        "representative_predictions_file": str(predictions_path.resolve()) if predictions_path.exists() else pd.NA,
        "representative_plot_file": str((model_dir / f"diagonal_{property_name}.png").resolve())
        if (model_dir / f"diagonal_{property_name}.png").exists()
        else pd.NA,
        "artifact_selection_mode": "external_single_run",
        "artifact_predictions_file": str(predictions_path.resolve()) if predictions_path.exists() else pd.NA,
        "artifact_expected_split_file": str(test_set_path.resolve()) if test_set_path.exists() else pd.NA,
        "artifact_test_r2": r2,
        "artifact_test_mae": mae,
        "artifact_test_rmse": rmse,
        "artifact_test_row_count": int(pd.read_csv(test_set_path).shape[0]) if test_set_path.exists() else np.nan,
        "plot_test_r2": r2,
        "plot_test_mae": mae,
        "plot_test_rmse": rmse,
        "family_best_metric": "summary_test_r2",
        "family_rank_score": r2,
        "rank_within_family": 1,
        "is_family_best": True,
        "source_family_dir": str(source_root.resolve()),
    }


def _build_fold_aggregated_row(
    *,
    method_name: str,
    case_metadata: dict[str, Any],
    case_dir: Path,
    provider_name: str,
    model_dir_name: str,
    model_family: str,
    model_name: str,
    display_label: str,
    source_root: Path,
) -> dict[str, Any] | None:
    property_name = str(case_metadata["property"])
    fold_records: list[dict[str, Any]] = []
    for fold_dir in sorted(case_dir.glob("fold_*")):
        model_dir = fold_dir / provider_name / model_dir_name
        metrics_path = model_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        r2, mae, rmse = _metric_triplet_from_metrics_json(metrics_path, property_name)
        fold_index = pd.to_numeric(pd.Series([fold_dir.name.replace("fold_", "")]), errors="coerce").iloc[0]
        fold_records.append(
            {
                "fold_dir": fold_dir,
                "model_dir": model_dir,
                "fold_index": float(fold_index) if pd.notna(fold_index) else np.nan,
                "r2": r2,
                "mae": mae,
                "rmse": rmse,
                "predictions_file": model_dir / "predictions.csv",
                "test_set_file": model_dir / "test_set.csv",
                "plot_file": model_dir / f"diagonal_{property_name}.png",
            }
        )

    if not fold_records:
        return None

    fold_df = pd.DataFrame(fold_records)
    summary_r2 = float(fold_df["r2"].mean())
    summary_mae = float(fold_df["mae"].mean())
    summary_rmse = float(fold_df["rmse"].mean())
    representative = (
        fold_df.assign(
            _distance=(fold_df["r2"] - summary_r2).abs(),
            _fold_sort=pd.to_numeric(fold_df["fold_index"], errors="coerce").fillna(np.inf),
        )
        .sort_values(["_distance", "r2", "_fold_sort"], ascending=[True, False, True], kind="mergesort")
        .iloc[0]
    )
    representative_test_set = Path(str(representative["test_set_file"]))
    return {
        "alloy_family": case_metadata["alloy_family"],
        "dataset_name": case_metadata["dataset_name"],
        "property": property_name,
        "ood_method": method_name,
        "model_family": model_family,
        "model": model_name,
        "display_label": display_label,
        "model_dir": str(case_dir.resolve()),
        "source_dir": str(source_root.resolve()),
        "trial_count": 1,
        "fold_count": int(len(fold_df)),
        "summary_test_r2": summary_r2,
        "summary_test_r2_std": _sample_std(fold_df["r2"].tolist()),
        "summary_test_mae": summary_mae,
        "summary_test_mae_std": _sample_std(fold_df["mae"].tolist()),
        "summary_test_rmse": summary_rmse,
        "summary_test_rmse_std": _sample_std(fold_df["rmse"].tolist()),
        "representative_selection_mode": "external_fold_closest_to_mean_r2",
        "representative_trial_id": pd.NA,
        "representative_fold": representative["fold_index"],
        "representative_test_r2": float(representative["r2"]),
        "representative_test_mae": float(representative["mae"]),
        "representative_test_rmse": float(representative["rmse"]),
        "representative_predictions_file": str(Path(str(representative["predictions_file"])).resolve())
        if Path(str(representative["predictions_file"])).exists()
        else pd.NA,
        "representative_plot_file": str(Path(str(representative["plot_file"])).resolve())
        if Path(str(representative["plot_file"])).exists()
        else pd.NA,
        "artifact_selection_mode": "external_fold_aggregated_summary",
        "artifact_predictions_file": pd.NA,
        "artifact_expected_split_file": str(representative_test_set.resolve()) if representative_test_set.exists() else pd.NA,
        "artifact_test_r2": summary_r2,
        "artifact_test_mae": summary_mae,
        "artifact_test_rmse": summary_rmse,
        "artifact_test_row_count": np.nan,
        "plot_test_r2": summary_r2,
        "plot_test_mae": summary_mae,
        "plot_test_rmse": summary_rmse,
        "family_best_metric": "summary_test_r2",
        "family_rank_score": summary_r2,
        "rank_within_family": 1,
        "is_family_best": True,
        "source_family_dir": str(source_root.resolve()),
    }


def _load_fewshot_guided_source(source_cfg: dict[str, Any]) -> pd.DataFrame:
    root_dir = Path(str(source_cfg["root_dir"]))
    if not root_dir.exists():
        return pd.DataFrame()

    method_map = {str(k): str(v) for k, v in source_cfg["method_map"].items()}
    case_map = {str(k): v for k, v in source_cfg["case_map"].items()}
    models = list(source_cfg.get("models", []))
    if not models:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for method_dir_name, method_name in method_map.items():
        method_root = root_dir / method_dir_name
        if not method_root.exists():
            continue
        for case_dir in sorted(method_root.iterdir()):
            if not case_dir.is_dir():
                continue
            case_metadata = _match_case_metadata(case_dir.name, case_map)
            if case_metadata is None:
                continue
            for model_cfg in models:
                provider_name = str(model_cfg.get("provider", source_cfg.get("default_provider", ""))).strip()
                model_dir_name = str(model_cfg["model_dir"])
                model_name = str(model_cfg.get("model_name", model_dir_name))
                display_label = str(model_cfg.get("display_label", model_name))
                model_family = str(model_cfg.get("model_family", source_cfg.get("family_name", "External")))
                direct_model_dir = case_dir / provider_name / model_dir_name if provider_name else case_dir / model_dir_name
                row: dict[str, Any] | None = None
                if direct_model_dir.exists() and (direct_model_dir / "metrics.json").exists():
                    row = _build_single_run_row(
                        method_name=method_name,
                        case_metadata=case_metadata,
                        model_dir=direct_model_dir,
                        model_family=model_family,
                        model_name=model_name,
                        display_label=display_label,
                        source_root=root_dir,
                    )
                elif any(case_dir.glob("fold_*")):
                    row = _build_fold_aggregated_row(
                        method_name=method_name,
                        case_metadata=case_metadata,
                        case_dir=case_dir,
                        provider_name=provider_name,
                        model_dir_name=model_dir_name,
                        model_family=model_family,
                        model_name=model_name,
                        display_label=display_label,
                        source_root=root_dir,
                    )
                if row is not None:
                    rows.append(row)
    return pd.DataFrame(rows)


SOURCE_LOADERS = {
    "fewshot_guided_no_analysis": _load_fewshot_guided_source,
}


def load_external_ood_sources(config_path: Path | None) -> pd.DataFrame:
    if config_path is None or not config_path.exists():
        return pd.DataFrame()

    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    rows: list[pd.DataFrame] = []
    for source_cfg in payload.get("sources", []):
        if not source_cfg or not source_cfg.get("enabled", True):
            continue
        source_type = str(source_cfg.get("type", "")).strip()
        loader = SOURCE_LOADERS.get(source_type)
        if loader is None:
            raise ValueError(f"Unsupported external OOD source type: {source_type}")
        df = loader(source_cfg)
        if not df.empty:
            rows.append(df)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)
