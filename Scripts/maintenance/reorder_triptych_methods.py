from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(r"D:\XJTU\Research\PHD\ood\Figures\hybrid\triptych")
DEFAULT_BUILDER = Path(
    r"D:\XJTU\ImportantFile\auto-design-alloy\fewshot-guided\scripts\build_ood_hybird_results.py"
)

HYBRID_METHOD_COLUMNS = [
    "HybridHigh20+RandCV",
    "HybridHigh20+LOCO",
    "HybridHigh20+SparseXcluster",
    "HybridHigh20+SparseXsingle",
    "HybridHigh20+SparseYcluster",
    "HybridHigh20+SparseYsingle",
]
PURE_OOD_METHOD_COLUMNS = [
    "HybridHigh20+RandCV",
    "Extra.",
    "HybridHigh20+LOCO",
    "HybridHigh20+SparseXcluster",
    "HybridHigh20+SparseXsingle",
    "HybridHigh20+SparseYcluster",
    "HybridHigh20+SparseYsingle",
]
SCOPE_CATEGORIES = (
    ("combined", "main"),
    ("combined", "subset_comparison"),
    ("test_extrapolation_high20", "main"),
    ("test_inner_ood", "main"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reorder compact Hybrid Triptych CSV method columns and redraw matching PNGs without recomputing values."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--builder", type=Path, default=DEFAULT_BUILDER)
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def load_builder(path: Path):
    spec = importlib.util.spec_from_file_location("ood_hybird_triptych_builder", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load builder: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def task_for_folder(builder, folder_name: str):
    normalized = folder_name.replace("_", " ")
    for task in builder.TASKS:
        if task.display_label == normalized:
            return task
    raise KeyError(f"No builder task matches folder: {folder_name}")


def reorder_method_columns(frame: pd.DataFrame) -> pd.DataFrame:
    method_set = set(PURE_OOD_METHOD_COLUMNS)
    positions = [index for index, column in enumerate(frame.columns) if column in method_set]
    if not positions:
        return frame.copy()
    first = min(positions)
    non_methods = [column for column in frame.columns if column not in method_set]
    methods = [column for column in PURE_OOD_METHOD_COLUMNS if column in frame.columns]
    before = [column for column in non_methods if frame.columns.get_loc(column) < first]
    after = [column for column in non_methods if column not in before]
    return frame.loc[:, [*before, *methods, *after]].copy()


def remove_unplotted_subset_placeholders(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not {"source_table", "test_subset", "ood_method", "value"}.issubset(frame.columns):
        return frame.copy(), frame.iloc[0:0].copy()
    long_rows = frame["source_table"].fillna("").astype(str).str.endswith("_long")
    removable = (
        long_rows
        & frame["test_subset"].fillna("").astype(str).ne("pure_ood")
        & frame["ood_method"].fillna("").astype(str).eq("Extra.")
    )
    removed = frame[removable].copy()
    if not removed.empty and pd.to_numeric(removed["value"], errors="coerce").notna().any():
        raise AssertionError("Refusing to remove non-empty Extra. rows from Hybrid-only subset panels")
    return frame[~removable].reset_index(drop=True), removed


def source_table(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
    names = frame["source_table"].fillna("").astype(str)
    rows = frame[names.str.endswith(suffix)].copy()
    if rows.empty:
        raise KeyError(f"Missing source table ending with {suffix}")
    return rows


def indexed_metric_table(frame: pd.DataFrame, suffix: str, index_column: str, columns: list[str]) -> pd.DataFrame:
    rows = source_table(frame, suffix)
    table = rows.set_index(index_column).reindex(columns=columns)
    table.index.name = None
    return table.apply(pd.to_numeric, errors="coerce")


def save_png(builder, fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    builder.plt.close(fig)


def draw_main(builder, task, frame: pd.DataFrame, path: Path) -> None:
    methods = HYBRID_METHOD_COLUMNS
    stem = task.triptych_stem
    mae = indexed_metric_table(frame, f"{stem}_mae", "aggregate_label", methods).reindex(
        index=builder.AGGREGATE_LABELS
    )
    delta = indexed_metric_table(frame, f"{stem}_delta_vs_trad", "aggregate_label", methods).reindex(
        index=builder.AGGREGATE_LABELS
    )
    rank = indexed_metric_table(frame, f"{stem}_rank", "aggregate_label", methods).reindex(
        index=builder.AGGREGATE_LABELS
    )
    traditional = indexed_metric_table(frame, f"{stem}_traditional_models_mae", "model", methods)
    bert = indexed_metric_table(frame, f"{stem}_bert_models_mae", "model", methods).reindex(
        index=builder.BERT_MODEL_ORDER
    )

    builder.apply_reference_rcparams(11.0)
    fig = builder.plt.figure(
        figsize=builder.TRIPTYCH_FIGURE["size_inches"],
        dpi=int(builder.TRIPTYCH_FIGURE["dpi"]),
        constrained_layout=True,
    )
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=builder.TRIPTYCH_FIGURE["panel_width_ratios"],
        height_ratios=builder.TRIPTYCH_FIGURE["panel_height_ratios"],
    )
    top_axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    bottom_grid = grid[1, :].subgridspec(1, 2, wspace=0.18)
    bottom_axes = [fig.add_subplot(bottom_grid[0, index]) for index in range(2)]
    family_ymax = builder.combined_family_ymax([traditional, bert])

    builder.draw_aggregate_bar_panel(top_axes[0], mae)
    builder.draw_heatmap(
        top_axes[1],
        delta,
        "Delta MAE vs Trad. (%)",
        cmap=builder.matplotlib.colors.LinearSegmentedColormap.from_list(
            "npg_delta", builder.HEATMAP_STYLE["delta_colors"], N=int(builder.HEATMAP_STYLE["cmap_n"])
        ),
    )
    builder.draw_heatmap(
        top_axes[2],
        rank,
        "Rank by MAE",
        cmap=builder.matplotlib.colors.LinearSegmentedColormap.from_list(
            "npg_rank", builder.HEATMAP_STYLE["rank_colors"], N=int(builder.HEATMAP_STYLE["cmap_n"])
        ),
    )
    builder.draw_model_family_panel(bottom_axes[0], traditional, "Traditional", family_ymax)
    builder.draw_model_family_panel(bottom_axes[1], bert, "BERT", family_ymax)
    builder.add_panel_labels([*top_axes, *bottom_axes], ["a", "b", "c", "d", "e"])
    save_png(builder, fig, path)


def subset_wide_table(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
    rows = source_table(frame, suffix)
    keep = [
        "task",
        "test_subset",
        "test_subset_label",
        "aggregate_label",
        *PURE_OOD_METHOD_COLUMNS,
    ]
    return rows.reindex(columns=keep).copy()


def draw_subset_comparison(builder, task, frame: pd.DataFrame, path: Path) -> None:
    stem = f"{task.triptych_stem}_subset_comparison"
    metric_tables = {
        "mae": subset_wide_table(frame, f"{stem}_mae"),
        "delta": subset_wide_table(frame, f"{stem}_delta_vs_trad"),
        "rank": subset_wide_table(frame, f"{stem}_rank"),
    }
    mae_tables = {
        subset: builder._panel_table_from_wide(metric_tables["mae"], subset)
        for subset in builder.SUBSET_COMPARISON_SCOPES
    }
    delta_tables = {
        subset: builder._panel_table_from_wide(metric_tables["delta"], subset)
        for subset in builder.SUBSET_COMPARISON_SCOPES
    }
    rank_tables = {
        subset: builder._panel_table_from_wide(metric_tables["rank"], subset)
        for subset in builder.SUBSET_COMPARISON_SCOPES
    }
    mae_ymax = builder.combined_family_ymax(mae_tables.values()) * (
        1.0 + float(builder.BOTTOM_PANEL.get("ylim_padding_ratio", 0.18))
    )
    delta_min, delta_max = builder._finite_min_max(delta_tables.values())
    if delta_min is not None and delta_max is not None:
        delta_abs = max(abs(delta_min), abs(delta_max), 1.0)
        delta_min, delta_max = -delta_abs, delta_abs

    builder.apply_reference_rcparams(11.0)
    fig = builder.plt.figure(
        figsize=builder.SUBSET_COMPARISON_FIGURE["size_inches"],
        dpi=int(builder.SUBSET_COMPARISON_FIGURE["dpi"]),
        constrained_layout=True,
    )
    grid = fig.add_gridspec(
        len(builder.SUBSET_COMPARISON_SCOPES),
        3,
        width_ratios=builder.SUBSET_COMPARISON_FIGURE["panel_width_ratios"],
        wspace=float(builder.SUBSET_COMPARISON_FIGURE.get("wspace", 0.26)),
        hspace=float(builder.SUBSET_COMPARISON_FIGURE.get("hspace", 0.36)),
    )
    axes = []
    delta_cmap = builder.matplotlib.colors.LinearSegmentedColormap.from_list(
        "npg_subset_delta",
        builder.HEATMAP_STYLE["delta_colors"],
        N=int(builder.HEATMAP_STYLE["cmap_n"]),
    )
    rank_cmap = builder.matplotlib.colors.LinearSegmentedColormap.from_list(
        "npg_subset_rank",
        builder.HEATMAP_STYLE["rank_colors"],
        N=int(builder.HEATMAP_STYLE["cmap_n"]),
    )
    for row_index, subset in enumerate(builder.SUBSET_COMPARISON_SCOPES):
        method_labels = builder.subset_comparison_method_axis_labels(subset)
        subset_label = builder.subset_comparison_label(subset)
        row_axes = [fig.add_subplot(grid[row_index, column_index]) for column_index in range(3)]
        axes.extend(row_axes)
        builder.draw_aggregate_bar_panel(row_axes[0], mae_tables[subset], method_labels=method_labels)
        row_axes[0].set_ylabel(f"{subset_label}\nMAE")
        row_axes[0].set_ylim(0.0, max(mae_ymax, 1.0))
        builder.draw_heatmap(
            row_axes[1],
            delta_tables[subset],
            "Delta MAE vs Trad. (%)",
            cmap=delta_cmap,
            vmin=delta_min,
            vmax=delta_max,
            method_labels=method_labels,
        )
        builder.draw_heatmap(
            row_axes[2],
            rank_tables[subset],
            "Rank by MAE",
            cmap=rank_cmap,
            vmin=1.0,
            vmax=float(len(builder.AGGREGATE_LABELS)),
            method_labels=method_labels,
        )
    builder.add_panel_labels(axes, list("abcdefghi"))
    fig.suptitle(
        f"{task.display_label}: Pure OOD vs High20 target vs Inner OOD",
        fontsize=builder.STYLE.get("axes_title_size", 13.0) + 1.5,
    )
    save_png(builder, fig, path)


def assert_values_unchanged(before: pd.DataFrame, csv_path: Path) -> None:
    after = pd.read_csv(csv_path, low_memory=False)
    if set(before.columns) != set(after.columns):
        raise AssertionError(f"Column set changed: {csv_path}")
    pd.testing.assert_frame_equal(
        before,
        after.reindex(columns=before.columns),
        check_dtype=False,
        check_exact=True,
        check_names=True,
    )


def update_catalog_row_counts(root: Path, row_counts: dict[str, int]) -> None:
    catalog_path = root.parent / "figure_catalog.csv"
    catalog = pd.read_csv(catalog_path, low_memory=False)
    for csv_path, row_count in row_counts.items():
        mask = catalog["csv_path"].astype(str).eq(csv_path)
        if int(mask.sum()) != 1:
            raise AssertionError(f"Expected one catalog row for {csv_path}, found {int(mask.sum())}")
        catalog.loc[mask, "row_count"] = row_count
    catalog.to_csv(catalog_path, index=False, encoding="utf-8-sig", float_format="%.10g")


def expected_method_columns(category: str) -> list[str]:
    return PURE_OOD_METHOD_COLUMNS if category == "subset_comparison" else HYBRID_METHOD_COLUMNS


def validate_method_order(frame: pd.DataFrame, category: str, csv_path: Path) -> None:
    observed = [column for column in frame.columns if column in PURE_OOD_METHOD_COLUMNS]
    expected = [column for column in expected_method_columns(category) if column in frame.columns]
    if observed != expected:
        raise AssertionError(f"Method order mismatch in {csv_path}: {observed}")


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    builder = load_builder(args.builder.resolve())
    csv_paths = sorted(root.glob("*/csv/*/*/triptych.csv"))
    if len(csv_paths) != 32:
        raise ValueError(f"Expected 32 compact Triptych CSV files, found {len(csv_paths)} in {root}")

    figures = 0
    removed_placeholders = 0
    row_counts: dict[str, int] = {}
    for csv_path in csv_paths:
        scope = csv_path.relative_to(root).parts[0]
        category = csv_path.relative_to(root).parts[2]
        task_folder = csv_path.parent.name
        if (scope, category) not in SCOPE_CATEGORIES:
            raise ValueError(f"Unexpected Triptych location: {csv_path}")
        before = pd.read_csv(csv_path, low_memory=False)
        normalized, removed = remove_unplotted_subset_placeholders(before)
        removed_placeholders += len(removed)
        ordered = reorder_method_columns(normalized)
        ordered.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.10g")
        assert_values_unchanged(normalized, csv_path)
        validate_method_order(pd.read_csv(csv_path, nrows=0), category, csv_path)
        relative_csv = csv_path.relative_to(root.parent).as_posix()
        row_counts[relative_csv] = len(ordered)

        if not args.skip_figures:
            task = task_for_folder(builder, task_folder)
            figure_path = root / scope / "figure" / category / task_folder / "triptych.png"
            if category == "subset_comparison":
                draw_subset_comparison(builder, task, ordered, figure_path)
            else:
                draw_main(builder, task, ordered, figure_path)
            figures += 1

    update_catalog_row_counts(root, row_counts)

    png_paths = sorted(root.glob("*/figure/*/*/triptych.png"))
    if len(png_paths) != 32 or (not args.skip_figures and figures != 32):
        raise AssertionError(f"Expected 32 Triptych PNG files, found {len(png_paths)}; redrawn={figures}")
    if any(path.stat().st_size <= 0 for path in png_paths):
        raise AssertionError("One or more Triptych PNG files are empty")
    print(f"Reordered 32 CSV files in {root}; redrew {figures} PNG files")
    print(f"Removed {removed_placeholders} empty Extra. placeholder rows from Hybrid-only subset panels")
    print("Hybrid order: RandCV -> LOCO -> SX-cls -> SX-sgl -> SY-cls -> SY-sgl")
    print("Pure OOD order: RandCV -> Extra. -> LOCO -> SX-cls -> SX-sgl -> SY-cls -> SY-sgl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
