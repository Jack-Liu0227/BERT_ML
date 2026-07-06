from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .common import HYBRID_SUBSETS, REPO_ROOT, write_csv, write_json
from .join_audit import build_audit_summary, process_source
from .predictions import prediction_inventory
from .sources import DEFAULT_FEWSHOT_GPT54_ROOT, discover_fewshot_gpt54_sources, discover_sources
from .splits import load_train_and_test, test_id_set

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "w_error_relationship_full"


def output_dir_for(output_root: Path, *, scope: str, subset: str) -> Path:
    if scope == "ood":
        return output_root / "OOD" / "csv"
    return output_root / "OOD_HYBIRD" / subset / "csv"


def rebuild_scope(
    *,
    scope: str,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    roots: list[Path] | None = None,
    fewshot_root: Path | None = None,
    include_fewshot_gpt54: bool = False,
    case_contains: str | None = None,
    limit_sources: int | None = None,
    hybrid_subset: str = "combined",
    strict: bool = False,
) -> dict[str, object]:
    source_scope = "hybrid" if scope == "hybrid" else "ood"
    sources = discover_sources(scope=source_scope, roots=roots, case_contains=case_contains, limit=limit_sources)
    if include_fewshot_gpt54:
        remaining = None if limit_sources is None else max(limit_sources - len(sources), 0)
        if remaining is None or remaining > 0:
            sources.extend(
                discover_fewshot_gpt54_sources(
                    fewshot_root or DEFAULT_FEWSHOT_GPT54_ROOT,
                    scope=source_scope,
                    case_contains=case_contains,
                    limit=remaining,
                )
            )
    joined_frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, object]] = []
    w_inventory_rows: list[dict[str, object]] = []
    prediction_inventory_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    wx_cache: dict[tuple[str, str], tuple] = {}
    seen_w_inventory: set[tuple[str, str]] = set()

    for index, source in enumerate(sources, start=1):
        try:
            _, test, _, test_set = load_train_and_test(source, hybrid_subset=hybrid_subset)
            allowed_ids = test_id_set(test)
            prediction_inventory_rows.append(prediction_inventory(source, allowed_ids=allowed_ids, test_set=test_set))
            joined, coverage, w_inventory = process_source(source, hybrid_subset=hybrid_subset, wx_cache=wx_cache)
            joined_frames.append(joined)
            coverage_rows.append(coverage)
            w_key = (str(source.split_dir), test_set)
            if w_key not in seen_w_inventory:
                w_inventory_rows.append(w_inventory)
                seen_w_inventory.add(w_key)
        except Exception as exc:
            failure = {
                "source_index": index,
                "prediction_file": str(source.prediction_file),
                "source_split_dir": str(source.split_dir),
                "error": repr(exc),
            }
            failures.append(failure)
            if strict:
                raise

    joined_all = pd.concat(joined_frames, ignore_index=True, sort=False) if joined_frames else pd.DataFrame()
    coverage = pd.DataFrame(coverage_rows)
    w_inventory = pd.DataFrame(w_inventory_rows)
    pred_inventory = pd.DataFrame(prediction_inventory_rows)
    audit = build_audit_summary(
        joined_all,
        coverage,
        w_inventory,
        pred_inventory,
        scope=source_scope,
        test_set=hybrid_subset if source_scope == "hybrid" else "ood_test",
    )
    if failures:
        audit["status"] = "failed"
        audit.setdefault("failures", []).append(f"Source failures: {len(failures)}")
        audit["source_failures"] = failures

    out_dir = output_dir_for(output_root, scope=scope, subset=hybrid_subset)
    write_csv(joined_all, out_dir / "w_error_samples_long.csv")
    write_csv(pred_inventory, out_dir / "prediction_source_inventory.csv")
    write_csv(w_inventory, out_dir / "w_source_inventory.csv")
    write_csv(coverage, out_dir / "w_error_join_coverage.csv")
    write_json(audit, out_dir / "audit_summary.json")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild result-local OOD/Hybrid Wx-error tables.")
    parser.add_argument("--scope", choices=["ood", "hybrid", "both"], default="both")
    parser.add_argument("--hybrid-subset", choices=[*HYBRID_SUBSETS, "all"], default="all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--result-root", type=Path, action="append", default=None)
    parser.add_argument("--include-fewshot-gpt54", action="store_true")
    parser.add_argument("--fewshot-root", type=Path, default=DEFAULT_FEWSHOT_GPT54_ROOT)
    parser.add_argument("--case-contains", default=None)
    parser.add_argument("--limit-sources", type=int, default=None)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audits: list[dict[str, object]] = []
    if args.scope in {"ood", "both"}:
        audits.append(
            rebuild_scope(
                scope="ood",
                output_root=args.output_root,
                roots=args.result_root,
                fewshot_root=args.fewshot_root,
                include_fewshot_gpt54=args.include_fewshot_gpt54,
                case_contains=args.case_contains,
                limit_sources=args.limit_sources,
                strict=args.strict,
            )
        )
    if args.scope in {"hybrid", "both"}:
        subsets = HYBRID_SUBSETS if args.hybrid_subset == "all" else (args.hybrid_subset,)
        for subset in subsets:
            audits.append(
                rebuild_scope(
                    scope="hybrid",
                    output_root=args.output_root,
                    roots=args.result_root,
                    fewshot_root=args.fewshot_root,
                    include_fewshot_gpt54=args.include_fewshot_gpt54,
                    case_contains=args.case_contains,
                    limit_sources=args.limit_sources,
                    hybrid_subset=subset,
                    strict=args.strict,
                )
            )
    print(json.dumps(audits, ensure_ascii=False, indent=2))
    if any(audit.get("status") != "ok" for audit in audits):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
