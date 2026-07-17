"""Command-line entrypoint for the Figure 4 training-size OOD experiment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
import yaml

from .aggregation import aggregate_experiment, scan_run_status
from .core import TrainingSizeOODConfig, iter_training_run_keys
from .runner import TrainingSizeOODRunner


DEFAULT_CONFIG = Path("configs/figures/ood_figures_3_4.config.yaml")
EXPECTED_TASK_COUNT = 10
FIGURE4_MODEL_SPECS = frozenset(
    {
        ("Traditional", "RF"),
        ("Traditional", "XGB"),
        ("Traditional", "LightGBM"),
        ("Traditional", "MLP"),
        ("Traditional", "CatBoost"),
        ("BERT", "SciBERT"),
        ("BERT", "SteelBERT"),
        ("BERT", "MatSciBERT"),
        ("TabPFN", "TabPFN-2.5-Plus-Numeric"),
        ("TabPFN", "TabPFN-2.5-Plus-Text"),
        ("LLMProp", "LLM-Prop"),
    }
)


@dataclass(frozen=True)
class ExperimentSettings:
    config_path: Path
    output_root: Path
    frozen_parameters_path: Path
    training_config: TrainingSizeOODConfig
    bootstrap_iterations: int


def load_settings(config_path: str | Path = DEFAULT_CONFIG) -> ExperimentSettings:
    path = Path(config_path).resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    figure4 = dict(payload.get("figure4") or {})
    proximity_csv = _resolve_path(
        figure4.get(
            "proximity_csv",
            "output/top_tail_loco_proximity/xspace/csv/cluster_proximity_top20.csv",
        )
    )
    frozen_parameters = _resolve_path(
        figure4.get(
            "frozen_parameters_path",
            "output/calibration/parameter_freeze/frozen_model_params_v1.yaml",
        )
    )
    output_root = _resolve_path(
        figure4.get("training_output_root", "output/training_size_ood")
    )
    config = TrainingSizeOODConfig(
        proximity_csv=proximity_csv,
        random_state=int(figure4.get("random_seed", 20260707)),
        random_cv_folds=int(figure4.get("random_cv_folds", 5)),
        random_cv_fold_index=int(figure4.get("random_cv_fold_index", 2)),
        training_fractions=tuple(
            float(value) for value in figure4.get("training_fractions", [0.2, 0.4, 0.6, 0.8, 1.0])
        ),
        repeats=int(figure4.get("repeats", 10)),
        full_fraction_repeats=int(figure4.get("full_fraction_repeats", 1)),
    )
    return ExperimentSettings(
        config_path=path,
        output_root=output_root,
        frozen_parameters_path=frozen_parameters,
        training_config=config,
        bootstrap_iterations=int(figure4.get("bootstrap_iterations", 10_000)),
    )


def discover_task_ids(proximity_csv: str | Path) -> list[str]:
    data = pd.read_csv(proximity_csv)
    task_column = "task_id" if "task_id" in data.columns else "case_key"
    if task_column not in data.columns or "threshold_pct" not in data.columns:
        raise ValueError("Proximity CSV must contain task/case and threshold_pct columns")
    selected = data[pd.to_numeric(data["threshold_pct"], errors="coerce").eq(20)]
    return sorted(selected[task_column].dropna().astype(str).unique().tolist())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Figure 4 training-size OOD experiment")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["preflight", "smoke", "run", "status", "aggregate", "figure"],
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--families", nargs="*")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--bootstrap-iterations", type=int, default=None)
    return parser


def validate_preflight_contract(
    audit: pd.DataFrame,
    inventory: pd.DataFrame,
    *,
    expected_task_ids: Sequence[str],
    expected_model_specs: set[tuple[str, str]] | frozenset[tuple[str, str]],
    expected_runs_per_model: int,
    expected_task_count: int,
) -> None:
    tasks = list(dict.fromkeys(str(value) for value in expected_task_ids))
    if len(tasks) != expected_task_count:
        raise ValueError(
            f"Figure 4 task coverage mismatch: expected={expected_task_count}, found={len(tasks)}"
        )
    audit_tasks = set(audit.get("task_id", pd.Series(dtype=str)).astype(str))
    if audit_tasks != set(tasks):
        raise ValueError("Figure 4 preflight audit does not cover the discovered task set")
    required_columns = {
        "task_id",
        "model_family",
        "model",
        "training_fraction",
        "repeat_id",
    }
    missing = required_columns.difference(inventory.columns)
    if missing:
        raise ValueError(f"Figure 4 run inventory is missing columns: {sorted(missing)}")
    run_key_columns = ["task_id", "model_family", "model", "training_fraction", "repeat_id"]
    if inventory.duplicated(run_key_columns).any():
        raise ValueError("Figure 4 run inventory contains duplicate run keys")
    for task_id in tasks:
        task_inventory = inventory[inventory["task_id"].astype(str).eq(task_id)]
        actual_specs = set(
            task_inventory[["model_family", "model"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        if actual_specs != set(expected_model_specs):
            missing_specs = sorted(set(expected_model_specs).difference(actual_specs))
            extra_specs = sorted(actual_specs.difference(expected_model_specs))
            raise ValueError(
                f"Figure 4 model coverage mismatch for task={task_id}: "
                f"missing={missing_specs}, extra={extra_specs}"
            )
        counts = task_inventory.groupby(["model_family", "model"], dropna=False).size()
        if not counts.eq(expected_runs_per_model).all():
            raise ValueError(f"Figure 4 run-key coverage mismatch for task={task_id}")
    expected_parameter_entries = expected_task_count * len(expected_model_specs)
    parameter_entries = int(pd.to_numeric(audit["parameter_entry_count"], errors="raise").sum())
    if parameter_entries != expected_parameter_entries:
        raise ValueError(
            "Figure 4 frozen-parameter coverage mismatch: "
            f"expected={expected_parameter_entries}, found={parameter_entries}"
        )
    expected_runs = expected_parameter_entries * expected_runs_per_model
    if len(inventory) != expected_runs:
        raise ValueError(
            f"Figure 4 run inventory mismatch: expected={expected_runs}, found={len(inventory)}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.jobs < 1:
        raise ValueError("jobs must be positive")
    settings = load_settings(args.config)
    if args.stage == "figure":
        from Scripts.figures.ood_figures_3_4.config import load_config as load_figure_config
        from Scripts.figures.ood_figures_3_4.figure4 import run_figure4

        result = run_figure4(load_figure_config(settings.config_path))
        print(f"Figure 4 complete: rows={result.row_count}, output={result.output_dir}")
        return 0
    all_tasks = discover_task_ids(settings.training_config.proximity_csv)
    tasks = list(args.tasks or all_tasks)
    unknown_tasks = sorted(set(tasks).difference(all_tasks))
    if unknown_tasks:
        raise ValueError(f"Unknown Figure 4 task filters: {unknown_tasks}")
    runner = TrainingSizeOODRunner(
        settings.training_config,
        frozen_parameters_path=settings.frozen_parameters_path,
        output_root=settings.output_root,
        source_config_path=settings.config_path,
    )
    if args.stage == "preflight":
        audit = runner.preflight_tasks(all_tasks)
        inventory = pd.read_csv(settings.output_root / "csv" / "run_inventory.csv")
        validate_preflight_contract(
            audit,
            inventory,
            expected_task_ids=all_tasks,
            expected_model_specs=FIGURE4_MODEL_SPECS,
            expected_runs_per_model=len(list(iter_training_run_keys(settings.training_config))),
            expected_task_count=EXPECTED_TASK_COUNT,
        )
        print(f"Preflight complete: tasks={len(audit)}, output={settings.output_root}")
        return 0
    if args.stage == "smoke":
        audit = runner.preflight_tasks(all_tasks)
        inventory = pd.read_csv(settings.output_root / "csv" / "run_inventory.csv")
        validate_preflight_contract(
            audit,
            inventory,
            expected_task_ids=all_tasks,
            expected_model_specs=FIGURE4_MODEL_SPECS,
            expected_runs_per_model=len(list(iter_training_run_keys(settings.training_config))),
            expected_task_count=EXPECTED_TASK_COUNT,
        )
        smoke_task = next(
            (task_id for task_id in all_tasks if task_id.startswith("Al__") and "UTS" in task_id),
            None,
        )
        if smoke_task is None:
            raise ValueError("Smoke stage requires the Al UTS task")
        status = runner.run_tasks(
            [smoke_task],
            models=["RF", "SciBERT", "TabPFN-2.5-Plus-Numeric", "TabPFN-2.5-Plus-Text", "LLM-Prop"],
            run_keys=[(0.2, 0), (1.0, 0)],
            resume=not args.no_resume,
            jobs=args.jobs,
        )
        print(status["status"].value_counts().to_string())
        return 0
    if args.stage == "run":
        audit = runner.preflight_tasks(all_tasks)
        inventory = pd.read_csv(settings.output_root / "csv" / "run_inventory.csv")
        validate_preflight_contract(
            audit,
            inventory,
            expected_task_ids=all_tasks,
            expected_model_specs=FIGURE4_MODEL_SPECS,
            expected_runs_per_model=len(list(iter_training_run_keys(settings.training_config))),
            expected_task_count=EXPECTED_TASK_COUNT,
        )
        status = runner.run_tasks(
            tasks,
            families=args.families,
            models=args.models,
            resume=not args.no_resume,
            jobs=args.jobs,
        )
        print(status["status"].value_counts().to_string())
        return 0
    runner.validate_locked_inputs()
    if args.stage == "status":
        status = scan_run_status(settings.output_root)
        csv_dir = settings.output_root / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        status.to_csv(csv_dir / "run_status.csv", index=False)
        print(status["status"].value_counts(dropna=False).to_string() if not status.empty else "No runs")
        return 0
    iterations = args.bootstrap_iterations or settings.bootstrap_iterations
    result = aggregate_experiment(
        settings.output_root,
        bootstrap_iterations=iterations,
        random_seed=settings.training_config.random_state,
        require_complete=not args.allow_partial,
    )
    print(result["coverage"].to_string(index=False))
    return 0


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
