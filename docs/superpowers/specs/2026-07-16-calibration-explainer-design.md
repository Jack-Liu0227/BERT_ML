# Calibration Explainer Design

## Purpose

Build a standalone, scroll-based HTML explanation of the GPT-integrated calibration workflow for readers with no machine-learning background. The page must explain where the data came from, how parameters and representative models were selected, how each method produced confidence, how metrics were calculated, and how the final figures were verified.

## Audience

The primary reader understands neither OOD evaluation nor calibration. Every section therefore has three synchronized layers:

1. Plain explanation: everyday language and concrete examples.
2. Technical detail: exact algorithms, formulas, counts, seeds, and selection rules.
3. File evidence: source paths, downloadable CSV/YAML/JSON evidence, hashes, and verification state.

## Information Architecture

The page is a single vertical narrative with a sticky section navigator and a sticky three-mode segmented control. Sections are:

1. What calibration asks.
2. Data and task scope.
3. Hyperparameter selection and frozen v1.
4. Seven OOD protocols.
5. Prediction and uncertainty generation.
6. GPT integration and common-task filtering.
7. Metric calculations and equal-weight aggregation.
8. Results and interpretation.
9. Verification and reproducibility evidence.

The parameter section includes a task/model browser for all 110 frozen entries. The representative-model section includes a task/protocol browser for the 112 BERT-best and Traditional-best selections. The calculation section includes a small interactive absolute-error and normalized-MAE calculator.

## Data Contract

The generator reads only verified artifacts:

- `output/calibration/parameter_freeze/frozen_model_params_v1.yaml`
- `output/calibration/figures/final_with_gpt/figure_manifest.json`
- `output/calibration/figures/final_with_gpt/data/representative_model_selection.csv`
- `output/calibration/figures/final_with_gpt/data/method_tier_summary.csv`
- `output/calibration/figures/final_with_gpt/data/method_protocol_summary.csv`
- `output/calibration/figures/final_with_gpt/data/coverage_audit.csv`
- `output/calibration/figures/final_with_gpt/data/integrated_sample_level.csv`

Generation refuses to publish unless the snapshot is v1-compatible, verification passed, frozen entries equal 110, representative selections equal 112, integrated rows equal 55,386, and all six methods contain 9,231 samples.

## Delivery

The generated package lives at `output/calibration/explainer/` and opens directly from `index.html` without a server or external network dependency. It contains copied summary evidence and figure assets, but never modifies the verified source package. A generated manifest records source and output SHA256 values.

The previous HyperFrames package is removed only after the new explainer passes automated and browser QA.

## Visual Direction

Use a restrained scientific editorial layout: cool white canvas, charcoal text, electric blue structure, orange warnings, green verified states, and the existing method colors. Sections are full-width bands rather than floating cards. Repeated protocols and methods may use compact cards with at most 8px radius. Typography is dense enough for evidence but remains readable on a projector and mobile screen.
