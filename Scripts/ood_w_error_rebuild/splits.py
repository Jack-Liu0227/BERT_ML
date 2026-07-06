from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import clean_text, frame_join_ids, read_csv, read_json
from .sources import ResultSource


def load_train_and_test(source: ResultSource, *, hybrid_subset: str = "combined") -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], str]:
    summary = read_json(source.split_summary_file)
    train = read_csv(source.train_file)
    if source.scope != "hybrid":
        return train, read_csv(source.combined_test_file), summary, "ood_test"

    combined = read_csv(source.combined_test_file)
    if hybrid_subset in {"combined", "test_hybrid_combined"}:
        return train, combined, summary, "test_hybrid_combined"

    explicit = source.split_dir / "test_sets" / f"{hybrid_subset}.csv"
    if explicit.exists():
        return train, read_csv(explicit), summary, hybrid_subset

    outer_n = int(summary.get("outer_test_size", 0) or 0)
    inner_n = int(summary.get("inner_test_size", 0) or 0)
    if hybrid_subset == "test_extrapolation_high20":
        return train, combined.iloc[:outer_n].copy(), summary, hybrid_subset
    if hybrid_subset == "test_inner_ood":
        return train, combined.iloc[outer_n : outer_n + inner_n].copy(), summary, hybrid_subset
    raise ValueError(f"Unsupported hybrid subset: {hybrid_subset}")


def test_id_set(test: pd.DataFrame) -> set[str]:
    return {value for value in frame_join_ids(test) if value}


def configured_feature_columns(summary: dict[str, object]) -> list[str]:
    raw = summary.get("x_space_feature_columns")
    if isinstance(raw, list):
        return [clean_text(value) for value in raw if clean_text(value)]
    return []
