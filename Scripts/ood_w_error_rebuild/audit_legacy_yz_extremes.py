from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .common import REPO_ROOT, write_csv, write_json


DEFAULT_SUMMARY = (
    REPO_ROOT
    / "output"
    / "w_error_relationship_full"
    / "OOD"
    / "yz_severity_map"
    / "csv"
    / "ood_yz_severity_summary.csv"
)
DEFAULT_REFERENCE_DIR = (
    Path("D:/XJTU")
    / "\u5df2\u5b8c\u6210\u8bba\u6587\u6570\u636e\u6c47\u603b"
    / "Fewshot"
    / "\u9884\u5904\u7406\u6c47\u603b\u6570\u636e"
    / "\u6700\u7ec8\u7ed3\u679c\u56fe"
    / "OOD"
    / "per_task_extreme_ood_model_analysis"
    / "publication_tables"
    / "final_summary_excels"
    / "excel"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "output"
    / "w_error_relationship_full"
    / "OOD"
    / "yz_severity_map"
    / "audit_plan_checks"
)
EXCLUDED_TASKS = {"HEA-YS", "Steel-YS"}


def _extreme_rows(summary: pd.DataFrame, primary: str) -> pd.DataFrame:
    work = summary[~summary["task_id"].astype(str).isin(EXCLUDED_TASKS)].copy()
    work[primary] = pd.to_numeric(work[primary], errors="raise")
    return work.loc[work.groupby("task_id", sort=False)[primary].idxmax()].copy()


def _compare_one(
    summary: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    table_name: str,
    primary: str,
    secondary: str,
    method_column: str,
    primary_column: str,
    secondary_column: str,
) -> pd.DataFrame:
    actual = _extreme_rows(summary, primary)[
        ["task_id", "method_short", primary, secondary]
    ].rename(
        columns={
            "task_id": "Task",
            "method_short": "actual_method",
            primary: "actual_primary",
            secondary: "actual_secondary",
        }
    )
    expected = reference[
        ["Task", method_column, primary_column, secondary_column]
    ].rename(
        columns={
            method_column: "expected_method",
            primary_column: "expected_primary",
            secondary_column: "expected_secondary",
        }
    )
    expected = expected[~expected["Task"].astype(str).isin(EXCLUDED_TASKS)].copy()
    compared = expected.merge(actual, on="Task", how="outer", validate="one_to_one")
    compared.insert(0, "reference_table", table_name)
    compared["method_match"] = compared["expected_method"].eq(compared["actual_method"])
    compared["primary_round2_match"] = compared["expected_primary"].round(2).eq(
        compared["actual_primary"].round(2)
    )
    compared["secondary_round2_match"] = compared["expected_secondary"].round(2).eq(
        compared["actual_secondary"].round(2)
    )
    compared["status"] = compared[
        ["method_match", "primary_round2_match", "secondary_round2_match"]
    ].all(axis=1).map({True: "ok", False: "mismatch"})
    return compared


def compare_extreme_tables(
    summary: pd.DataFrame,
    table_a: pd.DataFrame,
    table_b: pd.DataFrame,
) -> pd.DataFrame:
    return pd.concat(
        [
            _compare_one(
                summary,
                table_a,
                table_name="Table_A_Wy",
                primary="Ry",
                secondary="Rz",
                method_column="Wy-extreme method",
                primary_column="Wy/RandCV",
                secondary_column="Wz/RandCV at same method",
            ),
            _compare_one(
                summary,
                table_b,
                table_name="Table_B_Wz",
                primary="Rz",
                secondary="Ry",
                method_column="Wz-extreme method",
                primary_column="Wz/RandCV",
                secondary_column="Wy/RandCV at same method",
            ),
        ],
        ignore_index=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit regenerated Pure OOD Y/Z extremes against legacy publication Tables A/B."
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.summary, encoding="utf-8-sig")
    table_a = pd.read_excel(args.reference_dir / "Table_A_Wy_extreme_summary.xlsx")
    table_b = pd.read_excel(args.reference_dir / "Table_B_Wz_extreme_summary.xlsx")
    comparison = compare_extreme_tables(summary, table_a, table_b)
    write_csv(comparison, args.output_dir / "table_ab_extreme_match.csv")
    ok = bool(comparison["status"].eq("ok").all())
    audit = {
        "status": "ok" if ok else "failure",
        "comparison_rows": int(len(comparison)),
        "matched_rows": int(comparison["status"].eq("ok").sum()),
        "summary": str(args.summary),
        "reference_dir": str(args.reference_dir),
    }
    write_json(audit, args.output_dir / "table_ab_extreme_match.json")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
