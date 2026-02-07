"""
Align prediction CSV row order to a reference CSV by ID.

Typical use: make TabPFN prediction CSV have the same ID order as another model's
`all_predictions.csv`, so you can compare columns row-by-row safely.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .prediction_alignment import align_df_to_reference_id_order
except ImportError:  # pragma: no cover
    from prediction_alignment import align_df_to_reference_id_order

def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"Failed to read CSV: {path}\n{exc}") from exc


def align_by_id_order(
    input_csv: Path,
    reference_csv: Path,
    output_csv: Path,
    id_col: str = "ID",
) -> None:
    df_in = _read_csv(input_csv)
    df_ref = _read_csv(reference_csv)

    if id_col not in df_in.columns:
        raise SystemExit(f"Input CSV missing '{id_col}' column: {input_csv}")
    if id_col not in df_ref.columns:
        raise SystemExit(f"Reference CSV missing '{id_col}' column: {reference_csv}")

    try:
        aligned = align_df_to_reference_id_order(df_in, df_ref, id_col=id_col)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(output_csv, index=False)


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Align prediction CSV row order to a reference CSV by ID.")
    parser.add_argument("--input", required=True, type=Path, help="Input CSV to reorder (e.g., TabPFN predictions).")
    parser.add_argument("--reference", required=True, type=Path, help="Reference CSV that provides ID order.")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV path.")
    parser.add_argument("--id-col", default="ID", help="ID column name (default: ID).")
    args = parser.parse_args()

    align_by_id_order(args.input, args.reference, args.output, id_col=args.id_col)


if __name__ == "__main__":  # pragma: no cover
    main()
