# Training-Size GPT-5.4 Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reusable GPT-5.4 result ingestion to both BERT-inclusive and no-BERT Figure 4 task outputs.

**Architecture:** A focused adapter normalizes the few-shot directory into the existing per-repeat metric schema and emits an audit. The existing renderer becomes method-list driven, while two thin CLI entry points select the BERT-inclusive or no-BERT method set.

**Tech Stack:** Python, pandas, matplotlib, pytest, JSON/CSV filesystem artifacts.

---

### Task 1: Specify GPT-5.4 normalization

**Files:**
- Create: `tests/test_training_size_gpt54_adapter.py`
- Create: `Scripts/figures/ood_figures_3_4/training_size_gpt54_adapter.py`

- [ ] Write failing tests for axis mapping, metric extraction, full-test coverage checks, incomplete triples, and partial-preview mode.
- [ ] Run `python -m pytest -q tests/test_training_size_gpt54_adapter.py` and confirm failure because the adapter does not exist.
- [ ] Implement directory scanning, normalized rows, and coverage audit.
- [ ] Re-run the adapter tests and confirm they pass.

### Task 2: Make the renderer method-list driven

**Files:**
- Modify: `tests/test_training_size_task_figures.py`
- Modify: `Scripts/figures/ood_figures_3_4/training_size_task_figures.py`

- [ ] Write failing tests showing a no-BERT method list and an appended GPT method can render.
- [ ] Run the focused tests and confirm failure from the fixed five-method globals.
- [ ] Add explicit method-order/style parameters while preserving current defaults.
- [ ] Re-run the focused tests and confirm both default and dynamic paths pass.

### Task 3: Extend both CLI entry points

**Files:**
- Modify: `tests/test_training_size_task_figures.py`
- Modify: `Scripts/figures/ood_figures_3_4/build_training_size_task_figures.py`
- Create: `Scripts/figures/ood_figures_3_4/build_training_size_no_bert_figures.py`

- [ ] Write failing CLI tests for `--gpt-root`, `--allow-partial-gpt`, audit output, and the `figures_no_bert` default.
- [ ] Run the focused CLI tests and confirm the arguments or module are missing.
- [ ] Implement shared build orchestration and both CLI defaults.
- [ ] Re-run the CLI tests and confirm they pass.

### Task 4: Verify output contracts

**Files:**
- Modify: `tests/test_training_size_task_figures.py`

- [ ] Test that BERT-inclusive output preserves `BERT-best`, no-BERT output excludes it, and GPT is appended only when eligible.
- [ ] Run all Figure 4 plotting, selection, explainer, and adapter tests.
- [ ] Run Ruff on all changed Python files.
- [ ] Run both scripts with zero bootstrap iterations and existing Panel A to verify output directories without retraining models.
