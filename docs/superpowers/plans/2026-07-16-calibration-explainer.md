# Calibration Explainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate and deliver a standalone interactive HTML explanation of the full calibration workflow for zero-background readers.

**Architecture:** A Python builder validates the verified calibration artifacts, creates a compact JavaScript data payload, copies immutable evidence and figures, and atomically publishes a static package. A dependency-free HTML template renders the three explanation layers, parameter and representative-model browsers, calculation widgets, results, and evidence downloads.

**Tech Stack:** Python standard library, PyYAML, HTML5, CSS, vanilla JavaScript, pytest, Playwright.

---

### Task 1: Define the artifact contract

**Files:**
- Create: `tests/test_calibration_explainer.py`
- Create: `src/calibration/explainer.py`

- [ ] Test that the payload contains snapshot, verification, scope, parameter, selection, method, result, and evidence data.
- [ ] Test exact source invariants: 110 frozen entries, 112 representative selections, 55,386 integrated rows, and 9,231 rows per method.
- [ ] Test atomic publication and copied evidence files.

### Task 2: Build the standalone template

**Files:**
- Create: `src/calibration/templates/calibration_explainer.html`

- [ ] Add the sticky section navigation and three-mode segmented control.
- [ ] Add all nine narrative sections with plain, technical, and evidence layers.
- [ ] Add parameter and representative-selection browsers driven by `data/pipeline_data.js`.
- [ ] Add the calculation widget, figure tabs, lightbox, and evidence downloads.
- [ ] Keep the page fully functional from a `file:///` URL without external assets.

### Task 3: Generate the delivery package

**Files:**
- Generate: `output/calibration/explainer/index.html`
- Generate: `output/calibration/explainer/data/pipeline_data.js`
- Generate: `output/calibration/explainer/assets/*.png`
- Generate: `output/calibration/explainer/evidence/*`
- Generate: `output/calibration/explainer/artifact_manifest.json`

- [ ] Run the builder with `C:\Users\HK\miniconda3\envs\llm\python.exe`.
- [ ] Confirm source hashes remain unchanged.
- [ ] Confirm the output package contains no external network references.

### Task 4: Verify and replace the old delivery

**Files:**
- Verify: `output/calibration/explainer/index.html`
- Remove after successful QA: `output/calibration/slides/calibration_method_process/`

- [ ] Run focused pytest coverage.
- [ ] Open the page in Chromium at desktop and mobile sizes.
- [ ] Verify all modes, navigators, selectors, calculator, figure tabs, and downloads.
- [ ] Check screenshots for overflow, overlap, blank images, and unreadable text.
- [ ] Stop the old HyperFrames preview process and remove only its exact delivery directory.
