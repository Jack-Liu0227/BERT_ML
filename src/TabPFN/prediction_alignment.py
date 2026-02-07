"""
Prediction CSV alignment helpers.

Used to make exported predictions have deterministic, comparable row order.
"""

from __future__ import annotations

from typing import Iterable, List

import pandas as pd


def stable_unique(values: Iterable) -> List:
    seen = set()
    out = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def align_df_to_reference_id_order(
    df: pd.DataFrame,
    reference_df: pd.DataFrame,
    id_col: str = "ID",
) -> pd.DataFrame:
    if id_col not in df.columns:
        raise ValueError(f"df missing '{id_col}' column")
    if id_col not in reference_df.columns:
        raise ValueError(f"reference_df missing '{id_col}' column")

    ref_ids = stable_unique(reference_df[id_col].tolist())
    indexed = df.set_index(id_col, drop=False)

    missing = [x for x in ref_ids if x not in indexed.index]
    if missing:
        raise ValueError(
            f"df is missing {len(missing)} IDs present in reference (showing up to 20): {missing[:20]}"
        )

    return indexed.loc[ref_ids].reset_index(drop=True)

