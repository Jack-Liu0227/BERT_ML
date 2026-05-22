from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd

from src.data_processing.strength_ood_common import canonical_row_hash


NO_PROCESSING_DESCRIPTION = "No processing description."


def _composition_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if str(col).endswith(("(wt%)", "(at%)"))]


def _format_composition_from_columns(row: pd.Series, columns: Iterable[str]) -> str:
    parts: list[str] = []
    for col in columns:
        value = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
        if pd.notna(value) and float(value) > 0:
            element = str(col).split("(")[0].strip()
            suffix = str(col)[str(col).find("(") :] if "(" in str(col) else ""
            parts.append(f"{element}{float(value):.6g}{suffix}")
    return "Composition: " + ", ".join(parts) + "." if parts else "Composition: unknown."


def build_llmprop_descriptions(
    df: pd.DataFrame,
    processing_text_column: str | None = "Processing_Description",
) -> pd.Series:
    if "composition" in df.columns:
        base = "Composition: " + df["composition"].fillna("").astype(str).str.strip()
        base = base.str.replace(r"Composition:\s*$", "Composition: unknown", regex=True) + "."
    else:
        comp_cols = _composition_columns(df)
        base = df.apply(lambda row: _format_composition_from_columns(row, comp_cols), axis=1)

    if processing_text_column and processing_text_column in df.columns:
        processing = df[processing_text_column].fillna("").astype(str).str.strip()
        processing = processing.where(processing.ne(""), NO_PROCESSING_DESCRIPTION)
    else:
        processing = pd.Series([NO_PROCESSING_DESCRIPTION] * len(df), index=df.index)

    return base.astype(str) + " Processing: " + processing.astype(str)


def convert_split_to_llmprop_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    target_column: str,
    processing_text_column: str | None = "Processing_Description",
) -> Dict[str, Any]:
    df = pd.read_csv(input_csv)
    if target_column not in df.columns:
        raise ValueError(f"Target column not found in split CSV: {target_column}")

    out = pd.DataFrame()
    out["ID"] = df["ID"] if "ID" in df.columns else range(1, len(df) + 1)
    out["description"] = build_llmprop_descriptions(df, processing_text_column=processing_text_column)
    out[target_column] = pd.to_numeric(df[target_column], errors="coerce")
    out = out.dropna(subset=[target_column]).reset_index(drop=True)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8")
    return {
        "input_csv": str(input_csv),
        "output_csv": str(output_path),
        "row_count": int(len(out)),
        "row_hash": canonical_row_hash(out),
        "target_column": target_column,
        "description_policy": "composition_or_composition_columns_plus_processing",
        "processing_text_column": processing_text_column,
    }
