from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


FROZEN_REL = Path("output/calibration/parameter_freeze/frozen_model_params_v1.yaml")
FINAL_REL = Path("output/calibration/figures/final_with_gpt")
TEMPLATE_REL = Path("src/calibration/templates/calibration_explainer.html")
ENGLISH_TEXT_REL = Path("src/calibration/templates/calibration_explainer_english.js")

_CJK_PATTERN = re.compile(r"[\u3400-\u9fff]")
_TRANSLATION_PAIR_PATTERN = re.compile(
    r'^\s*(\["(?:\\.|[^"\\])*",\s*"(?:\\.|[^"\\])*"\]),?\s*$',
    re.MULTILINE,
)
_ENGLISH_TEMPLATE_REPLACEMENTS = {
    '<html lang="zh-CN">': '<html lang="en">',
    (
        "GPT 有 ${data.gpt.complete_three_tier_cases}/${data.gpt.case_total} "
        "个完整三层案例、${data.gpt.high_medium_paired_cases}/${data.gpt.case_total} "
        "个 High/Medium、${data.gpt.medium_low_paired_cases}/${data.gpt.case_total} "
        "个 Medium/Low、${data.gpt.high_low_paired_cases}/${data.gpt.case_total} "
        "个 High/Low 可配对案例，以及 ${data.gpt.valid_spearman_cases}/"
        "${data.gpt.case_total} 个有效 tier-Spearman 案例。"
    ): (
        "GPT has ${data.gpt.complete_three_tier_cases}/${data.gpt.case_total} "
        "cases with all three tiers, ${data.gpt.high_medium_paired_cases}/"
        "${data.gpt.case_total} paired High/Medium cases, "
        "${data.gpt.medium_low_paired_cases}/${data.gpt.case_total} paired "
        "Medium/Low cases, ${data.gpt.high_low_paired_cases}/"
        "${data.gpt.case_total} paired High/Low cases, and "
        "${data.gpt.valid_spearman_cases}/${data.gpt.case_total} cases with a "
        "valid tier-Spearman correlation."
    ),
    "${task} calibration 按协议对比图": "${task} calibration comparison by protocol",
    "${task} 相邻置信层级诊断图": "${task} adjacent confidence-tier diagnostics",
    (
        "当前展示任务 ${task}：逐协议比较六种方法的 High、Medium、Low "
        "normalized MAE。"
    ): (
        "Task shown: ${task}. The figure compares High, Medium, and Low "
        "normalized MAE across all six methods within each protocol."
    ),
    (
        "当前展示任务 ${task}：左图为 Medium-minus-High，右图为 "
        "Low-minus-Medium，逐协议比较六种方法；所有逐任务图统一使用 ±4 色标。"
    ): (
        "Task shown: ${task}. The left panel is Medium-minus-High and the "
        "right panel is Low-minus-Medium across six methods and seven protocols; "
        "all task-level figures use a shared ±4 color range."
    ),
}

METHOD_ORDER = (
    "Traditional-best",
    "BERT-best",
    "LLM-Prop",
    "TabPFN-2.5-Plus-Numeric",
    "TabPFN-2.5-Plus-Text",
    "GPT-5.4",
)

EVIDENCE_SOURCES = {
    "frozen_model_params_v1.yaml": FROZEN_REL,
    "representative_model_selection.csv": FINAL_REL
    / "data/representative_model_selection.csv",
    "method_tier_summary.csv": FINAL_REL / "data/method_tier_summary.csv",
    "method_protocol_summary.csv": FINAL_REL
    / "data/method_protocol_summary.csv",
    "task_protocol_method_summary.csv": FINAL_REL
    / "data/task_protocol_method_summary.csv",
    "coverage_audit.csv": FINAL_REL / "data/coverage_audit.csv",
    "figure_manifest.json": FINAL_REL / "figure_manifest.json",
}

FIGURE_SOURCES = {
    "calibration_main.png": FINAL_REL / "overview/calibration_main.png",
    "calibration_by_protocol.png": FINAL_REL
    / "overview/calibration_by_protocol.png",
    "calibration_by_method.png": FINAL_REL / "overview/calibration_by_method.png",
    "calibration_adjacent_tier_diagnostics.png": FINAL_REL
    / "overview/calibration_adjacent_tier_diagnostics.png",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _int(value: object) -> int | None:
    numeric = _float(value)
    return int(numeric) if numeric is not None else None


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Calibration explainer inputs are missing: {missing}")


def _load_english_translations(path: Path) -> dict[str, str]:
    source = path.read_text(encoding="utf-8")
    pairs = [
        json.loads(match.group(1))
        for match in _TRANSLATION_PAIR_PATTERN.finditer(source)
    ]
    if not pairs:
        raise ValueError(f"No English translations were found in {path}")
    translations = {str(chinese): str(english) for chinese, english in pairs}
    if any(_CJK_PATTERN.search(value) for value in translations.values()):
        raise ValueError("English translation values must not contain CJK characters.")
    return translations


def _english_template(
    template_text: str,
    translations: dict[str, str],
) -> str:
    replacements = {**translations, **_ENGLISH_TEMPLATE_REPLACEMENTS}
    translated = template_text
    for source, target in sorted(
        replacements.items(), key=lambda item: len(item[0]), reverse=True
    ):
        translated = translated.replace(source, target)
    translated = translated.replace(
        "data/pipeline_data.js", "data/pipeline_data_en.js"
    )
    matches = sorted(set(_CJK_PATTERN.findall(translated)))
    if matches:
        lines = [
            f"{line_number}: {line.strip()}"
            for line_number, line in enumerate(translated.splitlines(), 1)
            if _CJK_PATTERN.search(line)
        ]
        raise ValueError(
            "English explainer template still contains CJK text: "
            + " | ".join(lines[:10])
        )
    return translated


def _english_payload(value: Any, translations: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: _english_payload(item, translations) for key, item in value.items()}
    if isinstance(value, list):
        return [_english_payload(item, translations) for item in value]
    if isinstance(value, str):
        return translations.get(value, value)
    return value


def _parameter_entries(frozen: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for source in frozen.get("entries", []):
        source_paths = source.get("source_paths", {}) or {}
        entries.append(
            {
                "task_id": str(source.get("task_id", "")),
                "model_family": str(source.get("model_family", "")),
                "model": str(source.get("model", "")),
                "parameter_hash": str(source.get("parameter_hash", "")),
                "selection_method": str(source.get("selection_method", "")),
                "selection_metric": str(source.get("selection_metric", "")),
                "mean_mae": _float(source.get("mean_mae")),
                "sd_mae": _float(source.get("sd_mae")),
                "mean_r2": _float(source.get("mean_r2")),
                "observed_fold_count": _int(source.get("observed_fold_count")),
                "observed_fold_ids": list(source.get("observed_fold_ids", []) or []),
                "representative_outer_fold_id": str(
                    source.get("representative_outer_fold_id", "")
                ),
                "representative_trial_number": _int(
                    source.get("representative_trial_number")
                ),
                "search_parameters": _json_safe(
                    source.get("search_parameters", {}) or {}
                ),
                "effective_parameters": _json_safe(
                    source.get("effective_parameters", {}) or {}
                ),
                "source_fold_ids": sorted(str(key) for key in source_paths),
            }
        )
    return entries


def _representatives(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        output.append(
            {
                "task_id": row["task_id"],
                "protocol": row["protocol"],
                "method": row["method"],
                "model_family": row["model_family"],
                "selected_model": row["selected_model"],
                "plot_test_mae": _float(row.get("plot_test_mae")),
                "summary_test_mae_std": _float(row.get("summary_test_mae_std")),
                "summary_test_r2": _float(row.get("summary_test_r2")),
            }
        )
    return output


def _result_rows(rows: list[dict[str, str]], numeric: tuple[str, ...]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        converted: dict[str, Any] = dict(row)
        for key in numeric:
            converted[key] = _float(row.get(key))
        output.append(converted)
    return output


def _sample_inventory(path: Path) -> tuple[int, dict[str, int], list[dict[str, Any]]]:
    counts: Counter[str] = Counter()
    examples: dict[str, dict[str, Any]] = {}
    total = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            total += 1
            method = str(row.get("method", ""))
            counts[method] += 1
            if method not in examples:
                examples[method] = {
                    "task_id": row.get("task_id", ""),
                    "protocol": row.get("protocol", ""),
                    "method": method,
                    "selected_model": row.get("selected_model", ""),
                    "sample_id": row.get("sample_id", ""),
                    "y_true": _float(row.get("y_true")),
                    "y_pred": _float(row.get("y_pred")),
                    "abs_error": _float(row.get("abs_error")),
                    "uncertainty_score": _float(row.get("uncertainty_score")),
                    "confidence_tier": row.get("confidence_tier", ""),
                    "confidence_source": row.get("confidence_source", ""),
                    "member_count": _int(row.get("member_count")),
                    "uncertainty_method": row.get("uncertainty_method", ""),
                }
    ordered_examples = [examples[method] for method in METHOD_ORDER]
    return total, dict(counts), ordered_examples


def build_payload(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    frozen_path = root / FROZEN_REL
    final_root = root / FINAL_REL
    manifest_path = final_root / "figure_manifest.json"
    selection_path = final_root / "data/representative_model_selection.csv"
    tier_path = final_root / "data/method_tier_summary.csv"
    protocol_path = final_root / "data/method_protocol_summary.csv"
    coverage_path = final_root / "data/coverage_audit.csv"
    integrated_path = final_root / "data/integrated_sample_level.csv"

    _require_files(
        (
            frozen_path,
            manifest_path,
            selection_path,
            tier_path,
            protocol_path,
            coverage_path,
            integrated_path,
        )
    )

    frozen = yaml.safe_load(frozen_path.read_text(encoding="utf-8")) or {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection_rows = _read_csv(selection_path)
    tier_rows = _read_csv(tier_path)
    protocol_rows = _read_csv(protocol_path)
    coverage_rows = _read_csv(coverage_path)
    sample_total, sample_counts, sample_examples = _sample_inventory(integrated_path)
    parameter_entries = _parameter_entries(frozen)
    representatives = _representatives(selection_rows)

    if str(frozen.get("version")) != "v1":
        raise ValueError("Calibration explainer requires frozen parameter version v1.")
    if manifest.get("source_verification_status") != "passed":
        raise ValueError("Calibration explainer requires a passed full verification report.")
    if frozen.get("snapshot_id") != manifest.get("snapshot_id"):
        raise ValueError("Frozen v1 and final figure snapshot IDs do not match.")
    if int(manifest.get("logical_figure_count", 0)) != 20:
        raise ValueError("Calibration explainer requires 20 logical GPT figures.")
    if int(manifest.get("figure_output_count", 0)) != 60:
        raise ValueError("Calibration explainer requires 60 GPT figure files.")
    if len(parameter_entries) != 110:
        raise ValueError(f"Expected 110 frozen parameter entries, found {len(parameter_entries)}.")
    if len(representatives) != 112:
        raise ValueError(f"Expected 112 representative selections, found {len(representatives)}.")
    if sample_total != 55386:
        raise ValueError(f"Expected 55,386 integrated samples, found {sample_total}.")
    expected_sample_counts = {method: 9231 for method in METHOD_ORDER}
    if sample_counts != expected_sample_counts:
        raise ValueError(
            "Integrated method counts changed: "
            f"expected={expected_sample_counts} observed={sample_counts}"
        )

    gpt_cases = [
        row
        for row in coverage_rows
        if row.get("method") == "GPT-5.4"
        and row.get("record_type", "case") == "case"
        and row.get("coverage_status", "included") == "included"
    ]
    gpt_valid_spearman = sum(_bool(row.get("tier_spearman_valid")) for row in gpt_cases)

    overall_tiers = _result_rows(
        [row for row in tier_rows if row.get("protocol") == "All"],
        (
            "normalized_mae_mean",
            "bootstrap_ci95_low",
            "bootstrap_ci95_high",
            "observation_count",
        ),
    )
    protocol_summary = _result_rows(
        protocol_rows,
        (
            "discrimination_mean",
            "discrimination_valid_count",
            "discrimination_total_count",
            "medium_minus_high_mean",
            "medium_minus_high_valid_count",
            "medium_minus_high_total_count",
            "low_minus_medium_mean",
            "low_minus_medium_valid_count",
            "low_minus_medium_total_count",
            "tier_spearman_mean",
            "tier_spearman_valid_count",
            "tier_spearman_total_count",
            "tier_spearman_valid_percent",
        ),
    )

    parameter_counts = Counter(entry["model_family"] for entry in parameter_entries)
    model_counts = Counter(entry["model"] for entry in parameter_entries)
    representative_counts = Counter(
        f'{row["method"]}:{row["selected_model"]}' for row in representatives
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "snapshot": {
            "version": str(frozen.get("version")),
            "id": str(frozen.get("snapshot_id")),
            "generated_at": str(frozen.get("generated_at")),
            "selection_mode": str(frozen.get("selection_mode")),
            "selection_metric": str(frozen.get("selection_metric")),
            "verification_status": str(manifest.get("source_verification_status")),
            "verified_unit_count": int(manifest.get("verified_unit_count", 0)),
            "protected_hashes": manifest.get("protected_source_hashes", {}),
        },
        "scope": {
            "all_task_count": 10,
            "common_tasks": list(manifest.get("common_tasks", [])),
            "excluded_tasks": dict(manifest.get("excluded_tasks", {})),
            "protocols": list(manifest.get("protocols", [])),
            "methods": list(manifest.get("methods", [])),
            "common_case_count": len(manifest.get("common_tasks", []))
            * len(manifest.get("protocols", [])),
            "task_protocol_method_count": int(
                manifest.get("summary_row_counts", {}).get(
                    "task_protocol_method_summary", 0
                )
            ),
            "integrated_sample_count": sample_total,
            "sample_count_by_method": sample_counts,
            "logical_figure_count": int(manifest.get("logical_figure_count", 0)),
            "figure_output_count": int(manifest.get("figure_output_count", 0)),
        },
        "parameters": {
            "entry_count": len(parameter_entries),
            "model_family_counts": dict(sorted(parameter_counts.items())),
            "model_counts": dict(sorted(model_counts.items())),
            "entries": parameter_entries,
        },
        "representatives": representatives,
        "representative_composition": dict(sorted(representative_counts.items())),
        "uncertainty_methods": [
            {
                "method": "Traditional-best",
                "source": "10 个独立随机种子训练成员",
                "settings": "Seeds 42-51；预测均值作为 y_pred；样本标准差使用 ddof=1",
                "tier_rule": "在 task-protocol-model 内稳定排序并按样本秩近似等分三组",
            },
            {
                "method": "BERT-best",
                "source": "同一 checkpoint 的 30 次随机 MC-dropout 预测",
                "settings": "Seeds 42-71；推理时保持 dropout 开启；样本标准差 ddof=1",
                "tier_rule": "在 task-protocol-model 内稳定排序并按样本秩近似等分三组",
            },
            {
                "method": "LLM-Prop",
                "source": "同一 checkpoint 的 30 次随机 MC-dropout 预测",
                "settings": "Seeds 42-71；推理时保持 dropout 开启；样本标准差 ddof=1",
                "tier_rule": "在 task-protocol-model 内稳定排序并按样本秩近似等分三组",
            },
            {
                "method": "TabPFN-2.5-Plus-Numeric",
                "source": "API 预测分位数",
                "settings": "Uncertainty = q90 - q10，即 80% 预测区间宽度",
                "tier_rule": "在 task-protocol-model 内稳定排序并按样本秩近似等分三组",
            },
            {
                "method": "TabPFN-2.5-Plus-Text",
                "source": "API 预测分位数",
                "settings": "Uncertainty = q90 - q10，即 80% 预测区间宽度",
                "tier_rule": "在 task-protocol-model 内稳定排序并按样本秩近似等分三组",
            },
            {
                "method": "GPT-5.4",
                "source": "原始语义置信度",
                "settings": "直接读取 prediction record 中的 High、Medium 或 Low",
                "tier_rule": "不重新分箱，也不填补缺失置信层",
            },
        ],
        "gpt": {
            "sources_discovered": int(manifest.get("gpt_source_count_discovered", 0)),
            "sources_included": int(manifest.get("gpt_source_count_included", 0)),
            "complete_three_tier_cases": int(
                manifest.get("gpt_complete_three_tier_cases", 0)
            ),
            "high_medium_paired_cases": int(
                manifest.get("gpt_high_medium_paired_cases", 0)
            ),
            "medium_low_paired_cases": int(
                manifest.get("gpt_medium_low_paired_cases", 0)
            ),
            "high_low_paired_cases": int(
                manifest.get("gpt_high_low_paired_cases", 0)
            ),
            "valid_spearman_cases": gpt_valid_spearman,
            "case_total": len(gpt_cases),
        },
        "statistics": dict(manifest.get("statistics", {})),
        "results": {
            "overall_tiers": overall_tiers,
            "protocol_summary": protocol_summary,
        },
        "sample_examples": sample_examples,
        "task_figures": {
            task_id: f"assets/by_task/{task_id}.png"
            for task_id in manifest.get("common_tasks", [])
        },
        "task_adjacent_figures": {
            task_id: f"assets/by_task_adjacent/{task_id}.png"
            for task_id in manifest.get("common_tasks", [])
        },
        "evidence": [
            {
                "name": name,
                "source_path": str((root / relative).resolve()),
                "download_path": f"evidence/{name}",
                "sha256": _sha256(root / relative),
            }
            for name, relative in EVIDENCE_SOURCES.items()
        ],
    }
    return _json_safe(payload)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _assert_publish_target(project_root: Path, output_dir: Path) -> None:
    resolved_root = project_root.resolve()
    resolved_output = output_dir.resolve()
    if resolved_output == resolved_root or resolved_output.parent == resolved_output:
        raise ValueError(f"Unsafe explainer output directory: {resolved_output}")
    if resolved_output.name.lower() in {"output", "calibration", "figures"}:
        raise ValueError(f"Explainer output is too broad: {resolved_output}")


def build_explainer(
    project_root: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> Path:
    root = Path(project_root).resolve()
    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (root / "output/calibration/explainer").resolve()
    )
    _assert_publish_target(root, destination)
    template_path = root / TEMPLATE_REL
    english_text_path = root / ENGLISH_TEXT_REL
    payload = build_payload(root)
    source_paths = [
        template_path,
        english_text_path,
        root / FROZEN_REL,
        root / FINAL_REL / "figure_manifest.json",
    ]
    source_paths.extend(root / relative for relative in EVIDENCE_SOURCES.values())
    source_paths.extend(root / relative for relative in FIGURE_SOURCES.values())
    source_paths.append(root / FINAL_REL / "data/integrated_sample_level.csv")
    task_figure_sources = {
        task_id: root
        / FINAL_REL
        / "by_task"
        / task_id
        / "calibration_by_protocol.png"
        for task_id in payload["scope"]["common_tasks"]
    }
    task_adjacent_figure_sources = {
        task_id: root
        / FINAL_REL
        / "by_task"
        / task_id
        / "calibration_adjacent_tier_diagnostics.png"
        for task_id in payload["scope"]["common_tasks"]
    }
    source_paths.extend(task_figure_sources.values())
    source_paths.extend(task_adjacent_figure_sources.values())
    _require_files(source_paths)
    source_hashes_before = {str(path.resolve()): _sha256(path) for path in source_paths}

    ready = destination.parent / f".{destination.name}.ready"
    if ready.exists():
        if ready.parent.resolve() != destination.parent.resolve():
            raise ValueError(f"Unsafe ready directory: {ready}")
        shutil.rmtree(ready)
    (ready / "assets").mkdir(parents=True)
    (ready / "assets" / "by_task").mkdir(parents=True)
    (ready / "assets" / "by_task_adjacent").mkdir(parents=True)
    (ready / "data").mkdir(parents=True)
    (ready / "evidence").mkdir(parents=True)

    template_text = template_path.read_text(encoding="utf-8")
    english_translations = _load_english_translations(english_text_path)
    english_template_text = _english_template(template_text, english_translations)
    english_payload = _english_payload(payload, english_translations)
    _write_text(ready / "index.html", template_text)
    _write_text(ready / "index_en.html", english_template_text)
    for name, relative in FIGURE_SOURCES.items():
        shutil.copy2(root / relative, ready / "assets" / name)
    for task_id, source_path in task_figure_sources.items():
        shutil.copy2(source_path, ready / "assets" / "by_task" / f"{task_id}.png")
    for task_id, source_path in task_adjacent_figure_sources.items():
        shutil.copy2(
            source_path,
            ready / "assets" / "by_task_adjacent" / f"{task_id}.png",
        )
    for name, relative in EVIDENCE_SOURCES.items():
        shutil.copy2(root / relative, ready / "evidence" / name)

    data_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    _write_text(
        ready / "data/pipeline_data.js",
        f"window.CALIBRATION_EXPLAINER_DATA = {data_json};\n",
    )
    english_data_json = json.dumps(
        english_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if _CJK_PATTERN.search(english_data_json):
        raise ValueError("English explainer data still contains CJK text.")
    _write_text(
        ready / "data/pipeline_data_en.js",
        f"window.CALIBRATION_EXPLAINER_DATA = {english_data_json};\n",
    )
    _write_text(
        ready / "README.md",
        "# Calibration Explainer\n\n"
        "- Chinese: open `index.html`\n"
        "- English: open `index_en.html`\n\n"
        "Both pages are standalone, share the same verified data and figure assets, "
        "and require no network connection.\n",
    )

    source_hashes_after = {str(path.resolve()): _sha256(path) for path in source_paths}
    if source_hashes_after != source_hashes_before:
        raise ValueError("Calibration source files changed while building the explainer.")

    output_paths = sorted(path for path in ready.rglob("*") if path.is_file())
    output_hashes = {
        str(path.relative_to(ready)).replace("\\", "/"): _sha256(path)
        for path in output_paths
    }
    artifact_manifest = {
        "schema_version": 1,
        "stage": "calibration-explainer",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_snapshot_id": payload["snapshot"]["id"],
        "source_verification_status": payload["snapshot"]["verification_status"],
        "source_sha256": source_hashes_before,
        "output_file_count": len(output_paths) + 1,
        "output_sha256": output_hashes,
    }
    _write_text(
        ready / "artifact_manifest.json",
        json.dumps(artifact_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )

    if destination.exists():
        archive_root = destination.parent / "explainer_archive"
        archive_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive_path = archive_root / timestamp
        counter = 1
        while archive_path.exists():
            archive_path = archive_root / f"{timestamp}_{counter}"
            counter += 1
        os.replace(destination, archive_path)
    os.replace(ready, destination)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the calibration explainer HTML.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    output = build_explainer(args.project_root, output_dir=args.output_dir)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
