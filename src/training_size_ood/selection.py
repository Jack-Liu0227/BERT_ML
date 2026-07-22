from __future__ import annotations

from typing import Sequence
import pandas as pd


SELECTION_COLUMNS = [
    "task_id",
    "training_fraction",
    "model_family",
    "selected_model",
    "candidate_count",
    "selection_random_cv_mae_mean",
    "selection_random_cv_mae_sd",
    "selection_repeat_count",
    "selection_rule",
]

REPRESENTATIVE_FAMILIES = ("Traditional", "BERT")
DISPLAY_METHODS = (
    "Traditional-best",
    "BERT-best",
    "TabPFN Numeric",
    "TabPFN Text",
    "LLM-Prop",
)


def select_representative_models(
    data: pd.DataFrame,
    *,
    families: Sequence[str] = REPRESENTATIVE_FAMILIES,
    criterion: str = "random_cv_mae",
    group_by: Sequence[str] = ("task_id", "training_fraction"),
) -> pd.DataFrame:
    """Select one Traditional/BERT model per task and training fraction.

    Random-CV MAE is measured once per run and is duplicated across the two
    failure axes in the source table. The duplicate axis rows are collapsed
    before averaging repeats. OOD columns are intentionally not read here.
    """
    required = {
        "task_id",
        "model_family",
        "model",
        "training_fraction",
        "repeat_id",
        criterion,
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Representative selection is missing columns: {sorted(missing)}")
    if criterion != "random_cv_mae":
        raise ValueError("Representative model selection only supports random_cv_mae")
    if tuple(group_by) != ("task_id", "training_fraction"):
        raise ValueError("Representative selection group_by must be task_id and training_fraction")

    columns = [*group_by, "model_family", "model", "repeat_id", criterion]
    candidates = data[data["model_family"].isin(families)][columns].copy()
    if candidates.empty:
        return pd.DataFrame(columns=SELECTION_COLUMNS)

    run_key = [*group_by, "model_family", "model", "repeat_id"]
    duplicate_scores = candidates.groupby(run_key, dropna=False)[criterion].nunique(dropna=False)
    if duplicate_scores.gt(1).any():
        raise ValueError("Random-CV MAE differs between failure axes for one model run")
    candidates = candidates.drop_duplicates(run_key)

    score_keys = [*group_by, "model_family", "model"]
    scores = (
        candidates.groupby(score_keys, dropna=False)[criterion]
        .agg(
            selection_random_cv_mae_mean="mean",
            selection_random_cv_mae_sd=lambda values: float(values.std(ddof=1))
            if len(values) > 1
            else 0.0,
            selection_repeat_count="nunique",
        )
        .reset_index()
    )
    scores["selection_random_cv_mae_sd"] = scores["selection_random_cv_mae_sd"].fillna(0.0)
    candidate_counts = scores.groupby([*group_by, "model_family"], dropna=False)["model"].nunique()
    scores = scores.merge(
        candidate_counts.rename("candidate_count").reset_index(),
        on=[*group_by, "model_family"],
        how="left",
    )
    scores = scores.sort_values(
        [*group_by, "model_family", "selection_random_cv_mae_mean", "selection_random_cv_mae_sd", "model"],
        kind="mergesort",
    )
    selected = scores.groupby([*group_by, "model_family"], dropna=False, as_index=False).head(1).copy()
    selected = selected.rename(columns={"model": "selected_model"})
    selected["selection_rule"] = (
        "lowest mean Random-CV MAE across repeats; tie-break by SD then model name"
    )
    return selected[SELECTION_COLUMNS].reset_index(drop=True)


def apply_representative_selection(
    data: pd.DataFrame,
    selection: pd.DataFrame,
    *,
    families: Sequence[str] = REPRESENTATIVE_FAMILIES,
) -> pd.DataFrame:
    """Filter family rows to selected models and attach display method labels."""
    required_selection = set(SELECTION_COLUMNS)
    missing = required_selection.difference(selection.columns)
    if missing:
        raise ValueError(f"Selection table is missing columns: {sorted(missing)}")
    key = ["task_id", "training_fraction", "model_family"]
    selected_lookup = selection[key + ["selected_model"]]
    family_rows = data[data["model_family"].isin(families)].merge(
        selected_lookup,
        on=key,
        how="left",
        validate="many_to_one",
    )
    if family_rows["selected_model"].isna().any():
        raise ValueError("Representative selection is missing a task/fraction/family key")
    family_rows = family_rows[family_rows["model"].eq(family_rows["selected_model"])].copy()
    family_rows = family_rows.drop(columns=["selected_model"])

    other_rows = data[~data["model_family"].isin(families)].copy()
    result = pd.concat([family_rows, other_rows], ignore_index=True, sort=False)
    result["display_method"] = result.apply(_display_method, axis=1)
    return result


def _display_method(row: pd.Series) -> str:
    family = str(row["model_family"])
    model = str(row["model"])
    if family in REPRESENTATIVE_FAMILIES:
        return f"{family}-best"
    if family == "TabPFN" and "Text" in model:
        return "TabPFN Text"
    if family == "TabPFN":
        return "TabPFN Numeric"
    if family == "LLMProp":
        return "LLM-Prop"
    return f"{family}: {model}"


def validate_display_methods(data: pd.DataFrame) -> None:
    observed = set(data.get("display_method", pd.Series(dtype=str)).dropna().astype(str))
    unexpected = observed.difference(DISPLAY_METHODS)
    if unexpected:
        raise ValueError(f"Unexpected Figure 4 display methods: {sorted(unexpected)}")
    missing = set(DISPLAY_METHODS).difference(observed)
    if missing:
        raise ValueError(f"Figure 4 display methods are missing: {sorted(missing)}")
