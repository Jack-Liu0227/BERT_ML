from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scripts.maintenance.restructure_hybrid_results import (
    compact_w_error,
    ensure_clean_output,
    validate,
    write_catalog,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact Hybrid W-error PNG/CSV hierarchy from per-scope source roots."
    )
    parser.add_argument("--combined-root", type=Path, required=True)
    parser.add_argument("--set-a-root", type=Path, required=True)
    parser.add_argument("--set-b-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    ensure_clean_output(output)
    catalog: list[dict[str, object]] = []
    compact_w_error(
        Path("."),
        output,
        catalog,
        scope_roots={
            "combined": args.combined_root.resolve(),
            "test_extrapolation_high20": args.set_a_root.resolve(),
            "test_inner_ood": args.set_b_root.resolve(),
        },
    )
    write_catalog(catalog, output)
    print(json.dumps(validate(output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
