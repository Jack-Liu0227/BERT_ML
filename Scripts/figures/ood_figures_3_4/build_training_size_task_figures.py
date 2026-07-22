from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scripts.figures.ood_figures_3_4.training_size_task_figures import (  # noqa: E402
    DISPLAY_METHOD_ORDER,
    prepare_display_data,
    render_training_size_task_outputs,
)
from Scripts.figures.ood_figures_3_4.training_size_gpt54_adapter import (  # noqa: E402
    load_gpt54_training_size_rows,
)


def parse_args(
    argv: list[str] | None = None,
    *,
    default_output_dir: Path = Path("output/training_size_ood/figures"),
    description: str = (
        "Build eight-task MAE, degradation-ratio, and Random-CV Figure 4 outputs."
    ),
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=description
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("output/training_size_ood/csv/degradation_ratio_by_repeat.csv"),
    )
    parser.add_argument(
        "--panel-a-drawio",
        type=Path,
        default=Path(
            "output/training_size_ood/figures/sources/figure4_panel_a_source.drawio"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
    )
    parser.add_argument(
        "--gpt-root",
        type=Path,
        default=None,
        help="Optional few-shot training-size output root containing GPT-5.4 results.",
    )
    parser.add_argument("--gpt-model", default="gpt-5.4")
    parser.add_argument(
        "--allow-partial-gpt",
        action="store_true",
        help="Include sampled GPT results when prediction counts do not cover full test sets.",
    )
    parser.add_argument(
        "--allow-incomplete-gpt-coverage",
        action="store_true",
        help="Plot available full-test GPT points even when some task/subset keys are missing.",
    )
    parser.add_argument("--drawio-executable", type=Path, default=None)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260707)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args(argv)


def append_gpt_rows_if_complete(
    selected: pd.DataFrame,
    gpt_rows: pd.DataFrame,
    *,
    allow_incomplete_coverage: bool = False,
) -> tuple[pd.DataFrame, bool, str]:
    """Append GPT rows only when every displayed task/fraction/repeat key is covered."""
    if gpt_rows.empty:
        return selected.copy(), False, "no eligible GPT-5.4 rows"
    keys = ["task_id", "training_fraction", "repeat_id"]
    expected = {
        (str(row[0]), float(row[1]), int(row[2]))
        for row in selected[keys].drop_duplicates().itertuples(index=False, name=None)
    }
    axis_counts = gpt_rows.groupby(keys, dropna=False)["failure_axis"].nunique()
    actual = {
        (str(index[0]), float(index[1]), int(index[2]))
        for index, count in axis_counts.items()
        if int(count) == 2
    }
    missing = expected.difference(actual)
    extra = actual.difference(expected)
    invalid_axis_keys = int(axis_counts.ne(2).sum())
    if allow_incomplete_coverage and not invalid_axis_keys:
        available = gpt_rows[
            gpt_rows.apply(
                lambda row: (
                    str(row["task_id"]),
                    float(row["training_fraction"]),
                    int(row["repeat_id"]),
                )
                in expected,
                axis=1,
            )
        ].copy()
        if not available.empty:
            return (
                pd.concat([selected, available], ignore_index=True, sort=False),
                True,
                f"partial GPT-5.4 coverage: {len(actual)}/{len(expected)} run keys",
            )
    if missing or extra or invalid_axis_keys:
        reason = (
            f"incomplete GPT-5.4 coverage: missing run keys={len(missing)}, "
            f"extra run keys={len(extra)}, invalid axis keys={invalid_axis_keys}"
        )
        return selected.copy(), False, reason
    combined = pd.concat([selected, gpt_rows], ignore_index=True, sort=False)
    return combined, True, "complete GPT-5.4 coverage"


def run_build(
    args: argparse.Namespace,
    *,
    excluded_display_methods: tuple[str, ...] = (),
) -> dict[str, object]:
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Training-size input CSV is missing: {args.input_csv}")
    raw = pd.read_csv(args.input_csv)
    selected = prepare_display_data(
        raw,
        excluded_display_methods=excluded_display_methods,
    )
    task_count = int(selected["task_id"].nunique())
    if task_count != 8:
        raise ValueError(f"Expected eight display tasks after filtering, found {task_count}")
    method_order = [
        method for method in DISPLAY_METHOD_ORDER if method not in excluded_display_methods
    ]
    gpt_result: dict[str, object] = {
        "gpt_root": None,
        "gpt_included": False,
        "gpt_reason": "--gpt-root was not provided",
        "gpt_normalized_rows": 0,
        "gpt_audit_rows": 0,
    }
    if args.gpt_root is not None:
        gpt_rows, gpt_audit = load_gpt54_training_size_rows(
            args.gpt_root,
            allow_partial=args.allow_partial_gpt,
            model_name=args.gpt_model,
        )
        data_dir = args.output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        gpt_rows.to_csv(data_dir / "gpt54_degradation_ratio_by_repeat.csv", index=False)
        gpt_audit.to_csv(data_dir / "gpt54_coverage_audit.csv", index=False)
        selected, gpt_included, gpt_reason = append_gpt_rows_if_complete(
            selected,
            gpt_rows,
            allow_incomplete_coverage=(
                args.allow_incomplete_gpt_coverage or args.allow_partial_gpt
            ),
        )
        if gpt_included:
            method_order.append("GPT-5.4")
        gpt_result = {
            "gpt_root": str(args.gpt_root.resolve()),
            "gpt_included": gpt_included,
            "gpt_reason": gpt_reason,
            "gpt_normalized_rows": int(len(gpt_rows)),
            "gpt_audit_rows": int(len(gpt_audit)),
            "gpt_normalized_csv": str(
                (data_dir / "gpt54_degradation_ratio_by_repeat.csv").resolve()
            ),
            "gpt_audit_csv": str((data_dir / "gpt54_coverage_audit.csv").resolve()),
        }
    result = render_training_size_task_outputs(
        selected,
        output_dir=args.output_dir,
        panel_a_drawio=args.panel_a_drawio,
        drawio_executable=args.drawio_executable,
        clean=args.clean,
        bootstrap_iterations=args.bootstrap_iterations,
        random_seed=args.random_seed,
        dpi=args.dpi,
        display_methods=method_order,
        allow_missing_methods=(
            bool(args.gpt_root)
            and bool(gpt_result["gpt_included"])
            and (
                args.allow_incomplete_gpt_coverage or args.allow_partial_gpt
            )
        ),
    )
    result.update(gpt_result)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_build(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
