from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scripts.maintenance.restructure_hybrid_results import write_catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh the global Hybrid catalog from replacement analysis catalogs."
    )
    parser.add_argument("--hybrid-root", type=Path, required=True)
    parser.add_argument("--triptych-catalog", type=Path, required=True)
    return parser.parse_args()


def read_without_id(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    return frame.drop(columns=["figure_id"], errors="ignore")


def main() -> None:
    args = parse_args()
    root = args.hybrid_root.resolve()
    current = pd.read_csv(root / "figure_catalog.csv", low_memory=False)
    keep = current[~current["analysis"].isin({"w_error_relationship", "triptych"})].copy()
    keep = keep.drop(columns=["figure_id"], errors="ignore")
    combined = pd.concat(
        [
            keep,
            read_without_id(args.triptych_catalog.resolve()),
        ],
        ignore_index=True,
        sort=False,
    )
    if combined["figure_path"].duplicated().any() or combined["csv_path"].duplicated().any():
        raise ValueError("Duplicate figure_path or csv_path in refreshed Hybrid catalog")
    write_catalog(combined.where(pd.notna(combined), "").to_dict(orient="records"), root)
    refreshed = pd.read_csv(root / "figure_catalog.csv", low_memory=False)
    print(
        json.dumps(
            {
                "catalog_count": int(len(refreshed)),
                "analysis_counts": refreshed["analysis"].value_counts().sort_index().to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
