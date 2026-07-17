from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_processing.canonical_split_registry import CanonicalSplitRegistry

from .full_pipeline import (
    aggregate_full_results,
    build_calibration_run_inventory,
    build_canonical_split_inventory,
    run_full_units,
)
from .full_verification import verify_full_results
from .full_plotting import build_full_calibration_figures
from .gpt_plotting import build_gpt_calibration_figures
from .adapters import tabpfn_source_artifacts
from .metrics import run_calibration
from .output_reset import reset_calibration_output
from .parameter_audit import write_parameter_audit
from .parameter_evaluation import run_parameter_evaluation_queue
from .parameter_registry import FrozenParameterRegistry, freeze_parameters
from .parameter_selection import (
    write_existing_randomcv_parameter_selection,
    write_randomcv_parameter_selection,
)
from .schemas import default_config_path, load_config
from .snapshot_repair import repair_v1_matbench_snapshot
from .ti_pilot import (
    build_uncertainty_source_index,
    run_ti_calibration,
    run_ti_pilot,
    run_ti_uncertainty,
)
from .uncertainty import run_uncertainty


STAGES = (
    "audit-parameters",
    "freeze-parameters",
    "select-best-parameters",
    "evaluate-missing-parameters",
    "uncertainty",
    "calibration",
    "repair-v1-matbench",
    "ti-index",
    "ti-uncertainty",
    "ti-calibration",
    "ti-pilot",
    "reset-output",
    "full-index",
    "full-run",
    "full-aggregate",
    "full-plot",
    "full-plot-gpt",
    "full-verify",
    "all",
)

FULL_INDEX_SCHEMA_VERSION = 1
DIRECTORY_LAYOUT_VERSION = 1
RUN_STATUS_ID_COLUMNS = [
    "run_key",
    "task_id",
    "canonical_protocol",
    "protocol",
    "split_id",
    "model_family",
    "model",
    "parameter_snapshot_id",
    "result_dir",
]


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Non-GPT model calibration workflow")
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--config", default=str(default_config_path()))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--snapshot-version", default="v1")
    parser.add_argument("--execute", action="store_true", default=False)
    parser.add_argument("--scope", choices=("ti", "all"), default="all")
    parser.add_argument("--split-root", default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--run-key", action="append", default=[])
    parser.add_argument("--full-families", default="")
    parser.add_argument("--models", default="")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ti-adapters", default="reuse,llmprop,tabpfn")
    parser.add_argument("--mc-passes", type=int, default=30)
    parser.add_argument("--model-families", default="Traditional,BERT,LLMProp")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def _snapshot_path(cfg, version: str) -> Path:
    return cfg.output_root / "parameter_freeze" / f"frozen_model_params_{version}.yaml"


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_csv_write(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _resolve_split_root(cfg, value: str | Path | None) -> Path:
    if value is None:
        return (cfg.project_root / "output" / "ood_splits").resolve()
    return cfg.resolve(value).resolve()


def _initial_run_status(run_inventory: pd.DataFrame) -> pd.DataFrame:
    status = run_inventory.reindex(columns=RUN_STATUS_ID_COLUMNS).copy()
    status["run_status"] = "pending"
    status["attempt_count"] = 0
    status["started_at"] = ""
    status["finished_at"] = ""
    status["last_error"] = ""
    return status


def _protected_file_inventory(
    split_inventory: pd.DataFrame,
    snapshot_path: Path,
    *,
    workspace_root: str | Path,
    run_inventory: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "protected_kind": "frozen_parameters",
            "task_id": "",
            "canonical_protocol": "",
            "split_id": "",
            "path": str(snapshot_path.resolve()),
            "sha256": _file_sha256(snapshot_path),
        }
    ]
    columns = {
        "train_file": "canonical_train",
        "test_file": "canonical_test",
        "summary_file": "canonical_summary",
        "split_manifest_file": "canonical_manifest",
    }
    for _, split in split_inventory.iterrows():
        for path_column, kind in columns.items():
            candidate = Path(str(split[path_column])).resolve()
            rows.append(
                {
                    "protected_kind": kind,
                    "task_id": str(split["task_id"]),
                    "canonical_protocol": str(split["canonical_protocol"]),
                    "split_id": str(split["split_id"]),
                    "path": str(candidate),
                    "sha256": _file_sha256(candidate),
                }
            )
        if run_inventory is None:
            tabpfn_models = (
                "TabPFN-2.5-Plus-Numeric",
                "TabPFN-2.5-Plus-Text",
            )
        else:
            matches = run_inventory[
                run_inventory["task_id"].astype(str).eq(str(split["task_id"]))
                & run_inventory["canonical_protocol"]
                .astype(str)
                .eq(str(split["canonical_protocol"]))
                & run_inventory["split_id"].astype(str).eq(str(split["split_id"]))
                & run_inventory["model_family"].astype(str).eq("TabPFN")
            ]
            tabpfn_models = tuple(sorted(matches["model"].astype(str).unique()))
        for model in tabpfn_models:
            quantile_path, manifest_path = tabpfn_source_artifacts(
                workspace_root=workspace_root,
                model=model,
                alloy_type=str(split["alloy_type"]),
                dataset_name=str(split["dataset_name"]),
                target_column=str(split["target_column"]),
                canonical_protocol=str(split["canonical_protocol"]),
                split_id=str(split["split_id"]),
            )
            for protected_kind, candidate in (
                ("tabpfn_predictive_quantiles", quantile_path),
                ("tabpfn_pipeline_manifest", manifest_path),
            ):
                if not candidate.is_file():
                    raise FileNotFoundError(
                        f"Protected TabPFN API 2.5 source is missing: {candidate}"
                    )
                rows.append(
                    {
                        "protected_kind": protected_kind,
                        "task_id": str(split["task_id"]),
                        "canonical_protocol": str(split["canonical_protocol"]),
                        "split_id": str(split["split_id"]),
                        "model": model,
                        "path": str(candidate),
                        "sha256": _file_sha256(candidate),
                    }
                )
    return pd.DataFrame(rows).drop_duplicates(["protected_kind", "path"]).reset_index(drop=True)


def _write_full_index(cfg, args: argparse.Namespace) -> dict[str, Any]:
    if args.snapshot_version != "v1":
        raise ValueError("Full calibration only supports frozen v1.")
    split_root = _resolve_split_root(cfg, args.split_root)
    if not split_root.exists() or not split_root.is_dir():
        raise FileNotFoundError(f"Canonical split root does not exist: {split_root}")

    snapshot_path = _snapshot_path(cfg, args.snapshot_version).resolve()
    if not snapshot_path.exists() or not snapshot_path.is_file():
        raise FileNotFoundError(f"Frozen parameter snapshot not found: {snapshot_path}")

    split_registry = CanonicalSplitRegistry.from_root(split_root)
    parameter_registry = FrozenParameterRegistry(snapshot_path)
    if not parameter_registry.is_task_model_registry:
        raise ValueError(
            "The active frozen snapshot uses the legacy protocol/split-level schema. "
            "Complete --stage select-best-parameters before building the full index."
        )

    split_inventory = build_canonical_split_inventory(
        split_registry,
        scope=args.scope,
    )
    run_inventory = build_calibration_run_inventory(
        split_registry,
        parameter_registry,
        output_root=cfg.output_root,
        scope=args.scope,
    )
    run_status = _initial_run_status(run_inventory)
    protected_files = _protected_file_inventory(
        split_inventory,
        snapshot_path,
        workspace_root=cfg.project_root,
        run_inventory=run_inventory,
    )

    audit_root = cfg.output_root / "audit"
    status_root = cfg.output_root / "status"
    _atomic_csv_write(audit_root / "canonical_split_inventory.csv", split_inventory)
    _atomic_csv_write(audit_root / "calibration_run_inventory.csv", run_inventory)
    _atomic_csv_write(audit_root / "protected_file_hashes.csv", protected_files)
    _atomic_csv_write(status_root / "run_status.csv", run_status)

    manifest = {
        "schema_version": FULL_INDEX_SCHEMA_VERSION,
        "stage": "full-index",
        "scope": args.scope,
        "snapshot_id": parameter_registry.snapshot_id,
        "frozen_parameter_file": str(snapshot_path),
        "frozen_parameter_file_sha256": _file_sha256(snapshot_path),
        "split_root": str(split_root),
        "counts": {
            "splits": len(split_inventory),
            "runs": len(run_inventory),
        },
        "directory_layout_version": DIRECTORY_LAYOUT_VERSION,
        "protected_file_count": int(len(protected_files)),
    }
    manifest_path = cfg.output_root / "manifest.json"
    _atomic_json_write(manifest_path, manifest)
    return {
        "split_count": len(split_inventory),
        "run_count": len(run_inventory),
        "manifest_path": str(manifest_path),
    }


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    cfg = load_config(args.config, args.output_root)

    if args.stage == "reset-output":
        result = reset_calibration_output(cfg, execute=args.execute)
        print(
            "Calibration output reset: "
            f"executed={result['executed']}, "
            f"planned={result['planned_count']}, "
            f"deleted={result['deleted_count']} -> {result['manifest_path']}"
        )
        return 0

    cfg.output_root.mkdir(parents=True, exist_ok=True)

    if args.stage == "full-index":
        result = _write_full_index(cfg, args)
        print(
            "Full calibration index: "
            f"splits={result['split_count']}, runs={result['run_count']} "
            f"-> {result['manifest_path']}"
        )
        return 0

    if args.stage in {"full-run", "full-verify"} and args.snapshot_version != "v1":
        raise ValueError(f"{args.stage} only supports frozen v1.")

    if args.stage == "full-run":
        split_root = _resolve_split_root(cfg, args.split_root)
        snapshot_path = _snapshot_path(cfg, "v1").resolve()
        settings = dict(cfg.raw.get("full_pipeline", {}) or {})
        seeds = tuple(
            int(value)
            for value in settings.get("traditional_seeds", list(range(42, 52)))
        )
        families = {
            value.strip() for value in args.full_families.split(",") if value.strip()
        }
        models = {value.strip() for value in args.models.split(",") if value.strip()}
        status = run_full_units(
            output_root=cfg.output_root,
            split_root=split_root,
            snapshot_path=snapshot_path,
            scope=args.scope,
            resume=args.resume,
            jobs=args.jobs,
            traditional_seeds=seeds,
            mc_passes=args.mc_passes,
            run_keys=set(args.run_key) or None,
            families=families or None,
            models=models or None,
            limit=args.limit,
        )
        counts = status["run_status"].value_counts().to_dict() if not status.empty else {}
        print(f"Full calibration run: {counts} -> {cfg.output_root / 'status' / 'run_status.csv'}")
        return 1 if int(counts.get("failed", 0)) > 0 else 0

    if args.stage == "full-aggregate":
        result = aggregate_full_results(cfg.output_root)
        print(
            "Full calibration aggregate: "
            f"samples={result['sample_count']}, "
            f"tiers={result['tier_summary_count']}, "
            f"metrics={result['uncertainty_metric_count']}"
        )
        return 0

    if args.stage == "full-plot":
        result = build_full_calibration_figures(
            cfg.output_root,
            scope=args.scope,
            settings=dict(cfg.raw.get("full_plot", {}) or {}),
        )
        print(
            "Full calibration figures: "
            f"logical={result['logical_figure_count']}, "
            f"files={result['output_file_count']} "
            f"-> {result['output_dir']}"
        )
        return 0

    if args.stage == "full-plot-gpt":
        result = build_gpt_calibration_figures(
            cfg.output_root,
            project_root=cfg.project_root,
            scope=args.scope,
            settings=dict(cfg.raw.get("full_plot_gpt", {}) or {}),
        )
        print(
            "GPT-integrated calibration figures: "
            f"logical={result['logical_figure_count']}, "
            f"files={result['output_file_count']} "
            f"-> {result['output_dir']}"
        )
        return 0

    if args.stage == "full-verify":
        result = verify_full_results(
            output_root=cfg.output_root,
            split_root=_resolve_split_root(cfg, args.split_root),
            snapshot_path=_snapshot_path(cfg, "v1").resolve(),
            scope=args.scope,
            strict_counts=True,
        )
        print(
            "Full calibration verification: "
            f"{result['verification_status']}; verified={result['verified_unit_count']}"
        )
        return 0

    if args.stage in {"audit-parameters", "all"}:
        audit = write_parameter_audit(cfg)
        print(f"Parameter audit: {len(audit)} rows -> {cfg.output_root / 'parameter_freeze'}")
    else:
        audit = None

    if args.stage == "freeze-parameters":
        snapshot_path = freeze_parameters(cfg, audit=audit, version=args.snapshot_version)
        print(f"Frozen parameters: {snapshot_path}")
    else:
        snapshot_path = _snapshot_path(cfg, args.snapshot_version)

    if args.stage in {"select-best-parameters", "all"}:
        if audit is None:
            audit_path = cfg.output_root / "parameter_freeze" / "parameter_source_audit.csv"
            audit = pd.read_csv(audit_path) if audit_path.exists() else write_parameter_audit(cfg)
        selection_mode = str(
            cfg.raw.get("parameter_selection", {}).get("mode", "existing_results_only")
        )
        if selection_mode == "existing_results_only":
            selection = write_existing_randomcv_parameter_selection(cfg, audit)
        elif selection_mode == "strict_five_fold":
            selection = write_randomcv_parameter_selection(cfg, audit)
        else:
            raise ValueError(f"Unsupported parameter_selection.mode: {selection_mode}")
        missing_score_count = selection.get(
            "missing_parameter_fold_score_count",
            selection.get("missing_existing_trial_score_count", 0),
        )
        print(
            "RandomCV parameter selection: "
            f"{selection['selection_status']}; "
            f"selected={selection['selected_tuned_entry_count']}, "
            f"fixed={selection['fixed_runtime_entry_count']}, "
            f"missing_scores={missing_score_count}"
        )
        if selection["selection_status"] != "complete":
            return 2
        snapshot_path = Path(selection["snapshot_path"])

    if args.stage == "evaluate-missing-parameters":
        families = {value.strip() for value in args.model_families.split(",") if value.strip()}
        status = run_parameter_evaluation_queue(
            cfg,
            families=families,
            limit=args.limit,
            resume=args.resume,
        )
        counts = status["evaluation_status"].value_counts().to_dict() if not status.empty else {}
        print(f"Parameter evaluations: {counts} -> {cfg.output_root / 'parameter_evaluations'}")

    snapshot_id = ""
    registry = None
    if snapshot_path.exists():
        registry = FrozenParameterRegistry(snapshot_path)
        snapshot_id = registry.snapshot_id

    parameter_consuming_stages = {
        "uncertainty",
        "calibration",
        "ti-index",
        "ti-uncertainty",
        "ti-calibration",
        "ti-pilot",
        "all",
    }
    if args.stage in parameter_consuming_stages and registry is not None and not registry.is_task_model_registry:
        raise ValueError(
            "The active frozen v1 uses the legacy protocol/split-level schema. "
            "Complete --stage select-best-parameters before running calibration."
        )

    if args.stage in {"uncertainty", "all"}:
        sample = run_uncertainty(cfg, parameter_snapshot_id=snapshot_id)
        print(f"Uncertainty table: {len(sample)} rows -> {cfg.output_root / 'uncertainty'}")

    if args.stage in {"calibration", "all"}:
        sample = run_calibration(cfg)
        print(f"Calibration table: {len(sample)} rows -> {cfg.output_root / 'calibration'}")

    if args.stage == "repair-v1-matbench":
        if args.snapshot_version != "v1":
            raise ValueError("The Matbench repair stage only supports snapshot version v1.")
        result = repair_v1_matbench_snapshot(cfg)
        print(
            "Repaired frozen v1: "
            f"{result['old_entry_count']} -> {result['new_entry_count']} entries; "
            f"snapshot {result['old_snapshot_id']} -> {result['new_snapshot_id']}"
        )

    ti_stages = {"ti-index", "ti-uncertainty", "ti-calibration", "ti-pilot"}
    if args.stage in ti_stages:
        if not snapshot_path.exists():
            raise FileNotFoundError(f"Frozen parameter snapshot not found: {snapshot_path}")
        adapters = tuple(value.strip() for value in args.ti_adapters.split(",") if value.strip())
        pilot_root = cfg.output_root / "pilots" / "ti"
        if args.stage == "ti-index":
            index = build_uncertainty_source_index(FrozenParameterRegistry(snapshot_path), pilot_root)
            print(f"Ti source index: {len(index)} rows -> {pilot_root / 'audit'}")
        elif args.stage == "ti-uncertainty":
            sample = run_ti_uncertainty(
                cfg,
                snapshot_path,
                resume=args.resume,
                adapters=adapters,
                mc_passes=args.mc_passes,
            )
            print(f"Ti uncertainty table: {len(sample)} rows -> {pilot_root}")
        elif args.stage == "ti-calibration":
            sample = run_ti_calibration(cfg)
            print(f"Ti calibration table: {len(sample)} rows -> {pilot_root}")
        else:
            sample = run_ti_pilot(
                cfg,
                snapshot_path,
                resume=args.resume,
                adapters=adapters,
                mc_passes=args.mc_passes,
            )
            print(f"Ti pilot calibration table: {len(sample)} rows -> {pilot_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
