from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


PACKAGE_DIR = Path(__file__).with_suffix("")
PACKAGE_INIT = PACKAGE_DIR / "__init__.py"
SPEC = importlib.util.spec_from_file_location(
    "_plot_hybrid_method_visualizations_pkg",
    PACKAGE_INIT,
    submodule_search_locations=[str(PACKAGE_DIR)],
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

DEFAULT_HYBRID_ROOT = MODULE.DEFAULT_HYBRID_ROOT
DEFAULT_OUTPUT_DIR = MODULE.DEFAULT_OUTPUT_DIR
DEFAULT_PURE_ROOT = MODULE.DEFAULT_PURE_ROOT
DEFAULT_TRIPTYCH_ROOT = MODULE.DEFAULT_TRIPTYCH_ROOT
DELTA_COLUMNS = MODULE.DELTA_COLUMNS
MANIFEST_COLUMNS = MODULE.MANIFEST_COLUMNS
METHOD_ORDER = MODULE.METHOD_ORDER
MODEL_FAMILY_COLORS = MODULE.MODEL_FAMILY_COLORS
PURE_HYBRID_COLUMNS = MODULE.PURE_HYBRID_COLUMNS
REPRESENTATION_SPACES = MODULE.REPRESENTATION_SPACES
SPACE_ORDER = MODULE.SPACE_ORDER
SUBSET_LABELS = MODULE.SUBSET_LABELS
SUBSET_MARKERS = MODULE.SUBSET_MARKERS
SUMMARY_COLUMNS = MODULE.SUMMARY_COLUMNS
YZ_OOD_MAP_COLUMNS = MODULE.YZ_OOD_MAP_COLUMNS
build_ab_delta_summary = MODULE.build_ab_delta_summary
build_hybrid_summary_long = MODULE.build_hybrid_summary_long
build_pure_summary = MODULE.build_pure_summary
build_pure_vs_hybrid_summary = MODULE.build_pure_vs_hybrid_summary
build_set_ab_family_best_summary = MODULE.build_set_ab_family_best_summary
build_set_ab_master_sample_long = MODULE.build_set_ab_master_sample_long
build_set_ab_model_summary = MODULE.build_set_ab_model_summary
build_set_ab_source_audit = MODULE.build_set_ab_source_audit
build_set_ab_space_summary_long = MODULE.build_set_ab_space_summary_long
build_triptych_model_review = MODULE.build_triptych_model_review
build_yz_ood_map_summary = MODULE.build_yz_ood_map_summary
case_filter = MODULE.case_filter
clean_text = MODULE.clean_text
deduplicate_group_for_subset = MODULE.deduplicate_group_for_subset
figure_1_severity_decomposition = MODULE.figure_1_severity_decomposition
figure_1_severity_decomposition_task_compact = MODULE.figure_1_severity_decomposition_task_compact
figure_2_error_dumbbell = MODULE.figure_2_error_dumbbell
figure_2_error_dumbbell_task_compact = MODULE.figure_2_error_dumbbell_task_compact
figure_3_representation_severity_sensitivity = MODULE.figure_3_representation_severity_sensitivity
figure_3_representation_severity_sensitivity_task_compact = MODULE.figure_3_representation_severity_sensitivity_task_compact
figure_4_target_ceiling = MODULE.figure_4_target_ceiling
figure_4_target_ceiling_task_compact = MODULE.figure_4_target_ceiling_task_compact
figure_5_delta_heatmap = MODULE.figure_5_delta_heatmap
figure_6_hybrid_failure_disentanglement = MODULE.figure_6_hybrid_failure_disentanglement
figure_6_pure_vs_hybrid = MODULE.figure_6_pure_vs_hybrid
figure_7_method_contrast = MODULE.figure_7_method_contrast
figure_8_yz_ood_map = MODULE.figure_8_yz_ood_map
fit_slope = MODULE.fit_slope
load_hybrid_long_tables = MODULE.load_hybrid_long_tables
load_pure_long_table = MODULE.load_pure_long_table
main = MODULE.main
method_sort_key = MODULE.method_sort_key
model_family_for = MODULE.model_family_for
normalize_method = MODULE.normalize_method
normalize_subset = MODULE.normalize_subset
parse_args = MODULE.parse_args
r2_score_safe = MODULE.r2_score_safe
read_csv = MODULE.read_csv
run_hybrid_visualizations = MODULE.run_hybrid_visualizations
safe_filename = MODULE.safe_filename
summarize_one_group = MODULE.summarize_one_group
write_csv = MODULE.write_csv
write_report = MODULE.write_report

__all__ = [
    "DEFAULT_HYBRID_ROOT",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_PURE_ROOT",
    "DEFAULT_TRIPTYCH_ROOT",
    "DELTA_COLUMNS",
    "MANIFEST_COLUMNS",
    "METHOD_ORDER",
    "MODEL_FAMILY_COLORS",
    "PURE_HYBRID_COLUMNS",
    "REPRESENTATION_SPACES",
    "SPACE_ORDER",
    "SUBSET_LABELS",
    "SUBSET_MARKERS",
    "SUMMARY_COLUMNS",
    "YZ_OOD_MAP_COLUMNS",
    "build_ab_delta_summary",
    "build_hybrid_summary_long",
    "build_pure_summary",
    "build_pure_vs_hybrid_summary",
    "build_set_ab_family_best_summary",
    "build_set_ab_master_sample_long",
    "build_set_ab_model_summary",
    "build_set_ab_source_audit",
    "build_set_ab_space_summary_long",
    "build_triptych_model_review",
    "build_yz_ood_map_summary",
    "case_filter",
    "clean_text",
    "deduplicate_group_for_subset",
    "figure_1_severity_decomposition",
    "figure_1_severity_decomposition_task_compact",
    "figure_2_error_dumbbell",
    "figure_2_error_dumbbell_task_compact",
    "figure_3_representation_severity_sensitivity",
    "figure_3_representation_severity_sensitivity_task_compact",
    "figure_4_target_ceiling",
    "figure_4_target_ceiling_task_compact",
    "figure_5_delta_heatmap",
    "figure_6_hybrid_failure_disentanglement",
    "figure_6_pure_vs_hybrid",
    "figure_7_method_contrast",
    "figure_8_yz_ood_map",
    "fit_slope",
    "load_hybrid_long_tables",
    "load_pure_long_table",
    "main",
    "method_sort_key",
    "model_family_for",
    "normalize_method",
    "normalize_subset",
    "parse_args",
    "r2_score_safe",
    "read_csv",
    "run_hybrid_visualizations",
    "safe_filename",
    "summarize_one_group",
    "write_csv",
    "write_report",
]


if __name__ == "__main__":
    main()
