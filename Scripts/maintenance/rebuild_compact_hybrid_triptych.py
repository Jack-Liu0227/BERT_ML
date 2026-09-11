from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scripts.maintenance.restructure_hybrid_results import (
    compact_triptych,
    ensure_clean_output,
    validate,
    write_catalog,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and validate the compact Hybrid Triptych PNG/CSV hierarchy."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    ensure_clean_output(output)
    catalog: list[dict[str, object]] = []
    compact_triptych(source, output, catalog)
    write_catalog(catalog, output)
    print(json.dumps(validate(output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
