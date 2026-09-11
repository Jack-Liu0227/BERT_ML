from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scripts.figures.ood_figures_3_4.training_size_task_figures import (  # noqa: E402
    DISPLAY_METHOD_ORDER,
    prepare_display_data,
    render_combined_task_curve_figures,
    summarize_task_metric,
)
from Scripts.figures.ood_figures_3_4.compose_supplementary_figure4 import (  # noqa: E402
    DEFAULT_TASK_ORDER,
    compose_figure as compose_supplementary_figure,
)
from Scripts.figures.ood_figures_3_4.figure4_overview import (  # noqa: E402
    build_overview_from_sources,
)

DEFAULT_SOURCE = REPO_ROOT / "output" / "training_size_ood"
DEFAULT_OUTPUT = Path(r"D:\XJTU\Research\PHD\ood\Figures\figure4-data")
DEFAULT_GPT_ROOT = Path(
    r"D:\XJTU\ImportantFile\auto-design-alloy\fewshot-guided\output\ood\k5\strength_ood_training_size_ood_no_analysis"
)
WORKBOOK_BUILDER = Path(__file__).with_name("build_figure4_workbooks.mjs")
BUNDLED_DEPENDENCIES = (
    Path.home() / ".cache" / "codex-runtimes" / "codex-primary-runtime" / "dependencies"
)


def _copy_pngs(source: Path, target: Path) -> list[str]:
    target.mkdir(parents=True, exist_ok=True)
    for stale in target.glob("*.png"):
        stale.unlink()
    copied: list[str] = []
    for path in sorted(source.glob("*.png")):
        shutil.copy2(path, target / path.name)
        copied.append(path.name)
    return copied


def _dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _table_payload(frame: pd.DataFrame, *, key: str, label: str) -> dict[str, Any]:
    return {
        "key": key,
        "filename": label,
        "columns": [str(column) for column in frame.columns],
        "rows": _dataframe_records(frame),
    }


def _workbook_sheet(name: str, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "name": name,
        "columns": [str(column) for column in frame.columns],
        "rows": _dataframe_records(frame),
    }


def _artifact_runtime_paths() -> tuple[Path, Path]:
    node = Path(
        os.environ.get(
            "CODEX_BUNDLED_NODE",
            BUNDLED_DEPENDENCIES / "node" / "bin" / "node.exe",
        )
    )
    node_modules = Path(
        os.environ.get(
            "CODEX_BUNDLED_NODE_MODULES",
            BUNDLED_DEPENDENCIES / "node" / "node_modules",
        )
    )
    if not node.exists() or not node_modules.exists():
        raise FileNotFoundError(
            "The bundled spreadsheet runtime is unavailable. Expected Node.js at "
            f"{node} and packages at {node_modules}."
        )
    return node.resolve(), node_modules.resolve()


def _write_figure_workbooks(specs: list[dict[str, Any]]) -> None:
    node, node_modules = _artifact_runtime_paths()
    if not WORKBOOK_BUILDER.exists():
        raise FileNotFoundError(f"Workbook builder is missing: {WORKBOOK_BUILDER}")
    with tempfile.TemporaryDirectory(prefix="figure4-workbooks-") as temp_dir:
        runtime = Path(temp_dir).resolve()
        builder = runtime / WORKBOOK_BUILDER.name
        manifest = runtime / "manifest.json"
        junction = runtime / "node_modules"
        shutil.copy2(WORKBOOK_BUILDER, builder)
        manifest.write_text(
            json.dumps(
                {"verify_render": True, "workbooks": specs},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        if os.name == "nt":
            linked = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(junction), str(node_modules)],
                capture_output=True,
                text=True,
                check=False,
            )
            if linked.returncode != 0:
                raise RuntimeError(
                    "Failed to prepare the bundled spreadsheet runtime: "
                    + (linked.stderr or linked.stdout).strip()
                )
        else:
            junction.symlink_to(node_modules, target_is_directory=True)
        try:
            completed = subprocess.run(
                [str(node), str(builder), str(manifest)],
                cwd=runtime,
                capture_output=True,
                text=True,
                check=False,
                timeout=300,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "Figure 4 workbook export failed: "
                    + (completed.stderr or completed.stdout).strip()
                )
        finally:
            if junction.exists():
                junction.rmdir()


def _image_matched_curve_tables(
    source: Path,
    figures: Path,
    *,
    bootstrap_iterations: int,
    random_seed: int,
    cache_dir: Path,
    recompute_bootstrap: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    raw = pd.read_csv(source / "csv" / "degradation_ratio_by_repeat.csv")
    data = prepare_display_data(raw)
    gpt_path = figures / "data" / "gpt54_degradation_ratio_by_repeat.csv"
    if gpt_path.exists():
        data = pd.concat([data, pd.read_csv(gpt_path)], ignore_index=True, sort=False)
    hash_columns = [
        "task_id",
        "display_method",
        "failure_axis",
        "training_fraction",
        "repeat_id",
        "random_cv_mae",
        "ood_mae",
        "degradation_ratio",
    ]
    hashed = data[hash_columns].sort_values(hash_columns[:5]).reset_index(drop=True)
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(hashed, index=False).values.tobytes())
    digest.update(f"{bootstrap_iterations}:{random_seed}:task-curves-v1".encode("ascii"))
    input_sha256 = digest.hexdigest()
    manifest_path = cache_dir / "manifest.json"
    degradation_path = cache_dir / "degradation.csv"
    absolute_path = cache_dir / "absolute_mae.csv"
    if not recompute_bootstrap and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        if (
            manifest.get("input_sha256") == input_sha256
            and manifest.get("bootstrap_iterations") == bootstrap_iterations
            and manifest.get("random_seed") == random_seed
            and degradation_path.exists()
            and absolute_path.exists()
        ):
            return (
                pd.read_csv(degradation_path),
                pd.read_csv(absolute_path),
                "hit",
            )
    degradation = summarize_task_metric(
        data,
        metric_column="degradation_ratio",
        output_column="mean_degradation_ratio",
        bootstrap_iterations=bootstrap_iterations,
        random_seed=random_seed,
    )
    absolute_mae = summarize_task_metric(
        data,
        metric_column="ood_mae",
        output_column="mean_ood_mae",
        bootstrap_iterations=bootstrap_iterations,
        random_seed=random_seed,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    degradation_temp = degradation_path.with_suffix(".csv.tmp")
    absolute_temp = absolute_path.with_suffix(".csv.tmp")
    manifest_temp = manifest_path.with_suffix(".json.tmp")
    degradation.to_csv(degradation_temp, index=False)
    absolute_mae.to_csv(absolute_temp, index=False)
    manifest_temp.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "input_sha256": input_sha256,
                "bootstrap_iterations": bootstrap_iterations,
                "random_seed": random_seed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(degradation_temp, degradation_path)
    os.replace(absolute_temp, absolute_path)
    os.replace(manifest_temp, manifest_path)
    return degradation, absolute_mae, (
        "forced_recompute" if recompute_bootstrap else "miss"
    )


def _build_viewer_directory(
    source_root: str | Path,
    output_root: str | Path,
    *,
    gpt_root: str | Path = DEFAULT_GPT_ROOT,
    bootstrap_iterations: int = 10_000,
    random_seed: int = 20260707,
    cache_overview_dir: str | Path | None = None,
    recompute_bootstrap: bool = False,
) -> dict[str, Any]:
    source = Path(source_root).resolve()
    output = Path(output_root).resolve()
    figures = source / "figures"
    if not (source / "csv").exists() or not figures.exists():
        raise FileNotFoundError(f"Figure 4 source root is incomplete: {source}")

    output.mkdir(parents=True, exist_ok=True)
    data_dir = output / "data"
    no_bert_dir = output / "no_bert"
    if no_bert_dir.exists():
        shutil.rmtree(no_bert_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    overview_build_dir = output / ".overview_build"
    if overview_build_dir.exists():
        shutil.rmtree(overview_build_dir)
    overview_dir = output / "overview"
    if overview_dir.exists():
        shutil.rmtree(overview_dir)
    overview_dir.mkdir(parents=True, exist_ok=True)
    overview_result = build_overview_from_sources(
        source / "csv" / "degradation_ratio_by_repeat.csv",
        Path(gpt_root).resolve(),
        overview_build_dir,
        bootstrap_iterations=bootstrap_iterations,
        random_seed=random_seed,
        cache_dir=cache_overview_dir,
        recompute_bootstrap=recompute_bootstrap,
    )
    for path in sorted(overview_build_dir.glob("*.png")):
        shutil.copy2(path, overview_dir / path.name)

    task_cache_dir = (
        Path(cache_overview_dir).resolve().parent / "task_curves"
        if cache_overview_dir is not None
        else output.with_name(f"{output.name}.__cache__") / "task_curves"
    )
    (
        image_degradation,
        image_absolute_mae,
        task_curve_cache_status,
    ) = _image_matched_curve_tables(
        source,
        figures,
        bootstrap_iterations=bootstrap_iterations,
        random_seed=random_seed,
        cache_dir=task_cache_dir,
        recompute_bootstrap=recompute_bootstrap,
    )
    overview_degradation = pd.read_csv(
        overview_build_dir / "figure4_six_method_summary.csv"
    )
    overview_degradation_tasks = pd.read_csv(
        overview_build_dir / "figure4_six_method_task_repeat_means.csv"
    )
    overview_absolute = pd.read_csv(
        overview_build_dir / "figure4_six_method_absolute_mae_summary.csv"
    )
    overview_absolute_tasks = pd.read_csv(
        overview_build_dir / "figure4_six_method_absolute_mae_task_repeat_means.csv"
    )
    random_summary = pd.read_csv(
        figures / "random_cv_mae" / "random_cv_mae_all_tasks.csv"
    )
    tables = [
        _table_payload(
            overview_degradation,
            key="overview_degradation",
            label="Overview degradation summary",
        ),
        _table_payload(
            overview_absolute,
            key="overview_absolute_mae",
            label="Overview absolute-MAE summary",
        ),
        _table_payload(
            image_degradation,
            key="task_degradation",
            label="Task-level degradation curves",
        ),
        _table_payload(
            image_absolute_mae,
            key="task_absolute_mae",
            label="Task-level absolute-MAE curves",
        ),
        _table_payload(
            random_summary,
            key="random_cv_mae",
            label="Random-CV MAE overview",
        ),
    ]

    available_methods = set(image_degradation["display_method"].astype(str))
    combined_methods = [
        method
        for method in [*DISPLAY_METHOD_ORDER, "GPT-5.4"]
        if method in available_methods
    ]
    combined_paths = render_combined_task_curve_figures(
        image_degradation,
        image_absolute_mae,
        output_dir=output / "combined",
        display_methods=combined_methods,
    )
    combined = [path.name for path in combined_paths]
    supplementary_inputs = [
        output / "combined" / f"{task_id}.png" for task_id in DEFAULT_TASK_ORDER
    ]
    if all(path.is_file() for path in supplementary_inputs):
        compose_supplementary_figure(
            output / "combined",
            output / "combined" / "supplementary",
            dpi=600,
            write_pdf=False,
        )
    random_source = figures / "random_cv_mae" / "random_cv_mae_all_tasks.png"
    random_target = output / "random_cv_mae"
    random_target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(random_source, random_target / random_source.name)

    workbook_specs: list[dict[str, Any]] = [
        {
            "path": str(
                overview_dir / "figure4_performance_degradation_ratio.xlsx"
            ),
            "sheets": [
                _workbook_sheet("Summary", overview_degradation),
                _workbook_sheet("Task means", overview_degradation_tasks),
            ],
        },
        {
            "path": str(overview_dir / "figure4_absolute_ood_mae_overview.xlsx"),
            "sheets": [
                _workbook_sheet("Summary", overview_absolute),
                _workbook_sheet("Task means", overview_absolute_tasks),
            ],
        },
        {
            "path": str(overview_dir / "figure4_all_overviews.xlsx"),
            "sheets": [
                _workbook_sheet("Degradation summary", overview_degradation),
                _workbook_sheet("Degradation task means", overview_degradation_tasks),
                _workbook_sheet("Absolute MAE summary", overview_absolute),
                _workbook_sheet("Absolute MAE task means", overview_absolute_tasks),
            ],
        },
    ]
    for filename in combined:
        task_id = filename.removesuffix(".png")
        workbook_specs.append(
            {
                "path": str((output / "combined" / filename).with_suffix(".xlsx")),
                "sheets": [
                    _workbook_sheet(
                        "Degradation",
                        image_degradation[image_degradation["task_id"].eq(task_id)],
                    ),
                    _workbook_sheet(
                        "Absolute MAE",
                        image_absolute_mae[image_absolute_mae["task_id"].eq(task_id)],
                    ),
                ],
            }
        )
    workbook_specs.append(
        {
            "path": str(random_target / "random_cv_mae_all_tasks.xlsx"),
            "sheets": [_workbook_sheet("Random-CV MAE", random_summary)],
        }
    )
    _write_figure_workbooks(workbook_specs)

    image_groups = [
        {
            "key": "with_bert",
            "label": "With BERT",
            "groups": [
                {
                    "key": "overview",
                    "label": "Six-method overview",
                    "images": [
                        {
                            "name": "figure4_performance_degradation_ratio",
                            "src": "overview/figure4_performance_degradation_ratio.png",
                            "data": "overview/figure4_performance_degradation_ratio.xlsx",
                            "table_keys": ["overview_degradation"],
                        },
                        {
                            "name": "figure4_absolute_ood_mae_overview",
                            "src": "overview/figure4_absolute_ood_mae_overview.png",
                            "data": "overview/figure4_absolute_ood_mae_overview.xlsx",
                            "table_keys": ["overview_absolute_mae"],
                        },
                        {
                            "name": "figure4_all_overviews",
                            "src": "overview/figure4_all_overviews.png",
                            "data": "overview/figure4_all_overviews.xlsx",
                            "table_keys": [
                                "overview_degradation",
                                "overview_absolute_mae",
                            ],
                        },
                    ],
                },
                {
                    "key": "combined",
                    "label": "Combined task overview",
                    "images": [
                        {
                            "name": path.removesuffix(".png"),
                            "src": f"combined/{path}",
                            "data": f"combined/{path.removesuffix('.png')}.xlsx",
                            "table_keys": ["task_degradation", "task_absolute_mae"],
                        }
                        for path in combined
                    ],
                },
                {
                    "key": "random_cv",
                    "label": "Random-CV overview",
                    "images": [
                        {
                            "name": "random_cv_mae_all_tasks",
                            "src": f"random_cv_mae/{random_source.name}",
                            "data": "random_cv_mae/random_cv_mae_all_tasks.xlsx",
                            "table_keys": ["random_cv_mae"],
                        }
                    ],
                },
            ],
        }
    ]

    payload = {
        "tables": tables,
        "image_groups": image_groups,
        "workbook_count": len(workbook_specs),
        "overview_process": _overview_process_html(),
    }
    (data_dir / "viewer_data.js").write_text(
        "window.FIGURE4_VIEWER_DATA = "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
        newline="\n",
    )
    (output / "index.html").write_text(_page(), encoding="utf-8", newline="\n")
    if cache_overview_dir is not None:
        cache_target = Path(cache_overview_dir).resolve()
        if cache_target.exists():
            shutil.rmtree(cache_target)
        shutil.copytree(overview_build_dir, cache_target)
    manifest_overview = dict(overview_result)
    manifest_overview["output_dir"] = "overview"
    manifest_overview["cache_manifest"] = (
        str(Path(cache_overview_dir).resolve() / "figure4_overview_cache.json")
        if cache_overview_dir is not None
        else None
    )
    manifest_overview["images"] = {
        key: f"overview/{Path(path).name}"
        for key, path in overview_result["images"].items()
    }
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "source_root": str(source),
                "table_count": len(tables),
                "workbook_count": len(workbook_specs),
                "image_group_count": len(image_groups),
                "task_curve_cache_status": task_curve_cache_status,
                "overview": manifest_overview,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    shutil.rmtree(overview_build_dir)
    return {
        "output": str(output),
        "tables": len(tables),
        "workbooks": len(workbook_specs),
        "image_variants": len(image_groups),
        "task_curve_cache_status": task_curve_cache_status,
        "overview": overview_result,
    }


def _remove_stale_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _publish_staged_viewer(staging: Path, output: Path) -> None:
    backup = output.with_name(f"{output.name}.__backup__")
    _remove_stale_directory(backup)
    had_output = output.exists()
    if had_output:
        output.rename(backup)
    try:
        staging.rename(output)
    except Exception:
        if had_output and backup.exists() and not output.exists():
            backup.rename(output)
        raise
    if backup.exists():
        try:
            shutil.rmtree(backup)
        except OSError:
            pass


def _finalize_result_paths(result: dict[str, Any], output: Path) -> dict[str, Any]:
    finalized = dict(result)
    finalized["output"] = str(output)
    overview = dict(finalized["overview"])
    overview["output_dir"] = str(output / "overview")
    overview["cache_manifest"] = str(
        output.with_name(f"{output.name}.__cache__")
        / "overview"
        / "figure4_overview_cache.json"
    )
    overview["images"] = {
        key: str(output / "overview" / Path(path).name)
        for key, path in overview["images"].items()
    }
    finalized["overview"] = overview
    return finalized


def export_viewer(
    source_root: str | Path,
    output_root: str | Path,
    *,
    gpt_root: str | Path = DEFAULT_GPT_ROOT,
    bootstrap_iterations: int = 10_000,
    random_seed: int = 20260707,
    recompute_bootstrap: bool = False,
) -> dict[str, Any]:
    output = Path(output_root).resolve()
    staging = output.with_name(f"{output.name}.__staging__")
    _remove_stale_directory(staging)
    cache_overview_dir = (
        output.with_name(f"{output.name}.__cache__") / "overview"
    ).resolve()
    legacy_cache = output / "overview"
    if not cache_overview_dir.exists() and legacy_cache.exists():
        cache_overview_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(legacy_cache, cache_overview_dir)
    try:
        result = _build_viewer_directory(
            source_root,
            staging,
            gpt_root=gpt_root,
            bootstrap_iterations=bootstrap_iterations,
            random_seed=random_seed,
            cache_overview_dir=cache_overview_dir,
            recompute_bootstrap=recompute_bootstrap,
        )
        _publish_staged_viewer(staging, output)
    except Exception:
        _remove_stale_directory(staging)
        raise
    return _finalize_result_paths(result, output)


def _overview_process_html() -> str:
    return """
      <h3>Calculation process</h3>
      <p><code>degradation ratio = OOD MAE / Random-CV MAE</code></p>
      <ol>
        <li>Calculate one degradation ratio for each task, repeat, training fraction, method, and OOD axis.</li>
        <li>Average repeat-level ratios within each task.</li>
        <li>Average the task means with equal task weight to obtain each overview point.</li>
        <li>For each of 10,000 bootstrap iterations, resample tasks and then resample repeats within each selected task.</li>
        <li>Report the 2.5th and 97.5th percentiles of the bootstrap means as the 95% confidence interval.</li>
      </ol>
      <p>The 20-80% fractions contain ten sampling repeats. The 100% fraction uses the complete training pool and has one evaluation. All six displayed methods use the same eight tasks.</p>
      <p>Saved bootstrap summaries are reused when the selected data hash, task set, seed, and iteration count are unchanged. Bootstrap is recomputed only after a data/statistical-setting change or when explicitly forced.</p>
      <p><strong>Absolute-MAE note:</strong> the absolute overview averages native-unit OOD MAE across heterogeneous property tasks. It is useful for global inspection, while the degradation ratio is the preferred cross-task comparison.</p>
      <div class="process-links">
        <a href="overview/figure4_performance_degradation_ratio.xlsx" download>Degradation data</a>
        <a href="overview/figure4_absolute_ood_mae_overview.xlsx" download>Absolute-MAE data</a>
        <a href="overview/figure4_all_overviews.xlsx" download>Combined overview data</a>
      </div>
    """.strip()


def _legacy_page() -> str:
    return '''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" href="data:,">
  <title>Figure 4 Data</title>
  <style>
    :root { --paper:#fff; --wash:#f4f6f7; --surface:#fff; --ink:#171a1d; --muted:#687078; --line:#d8dde1; --blue:#0072b2; --green:#087f78; --teal-soft:#e5f3f1; }
    * { box-sizing:border-box; }
    body { margin:0; color:var(--ink); background:var(--wash); font-family:Aptos,"Segoe UI","Microsoft YaHei",sans-serif; font-size:14px; line-height:1.45; }
    button,select,input,a { font:inherit; }
    header { position:sticky; top:0; z-index:10; display:flex; align-items:center; justify-content:space-between; gap:16px; min-height:66px; padding:10px clamp(18px,4vw,52px); color:#fff; background:#20272b; border-bottom:3px solid var(--green); }
    h1 { margin:0; font:600 23px Georgia,"Times New Roman",serif; }
    .counts { display:flex; gap:18px; } .count-box { text-align:right; } .count-box strong { display:block; color:#fff; font-size:16px; } .count-box span { color:#aeb9be; font-size:11px; text-transform:uppercase; }
    .tabs { display:inline-grid; grid-auto-flow:column; gap:2px; margin:18px 0 14px; padding:3px; border:1px solid var(--line); border-radius:6px; background:#e7ebed; }
    .tab { min-height:35px; padding:0 14px; border:0; border-radius:4px; color:#40484e; background:transparent; cursor:pointer; font-weight:650; }
    .tab.active { color:#fff; background:var(--green); }
    main { max-width:1660px; margin:auto; padding:0 clamp(18px,4vw,52px) 40px; }
    .panel { display:none; } .panel.active { display:block; }
    .toolbar,.image-toolbar { display:flex; flex-wrap:wrap; align-items:end; gap:12px; margin-bottom:16px; padding:16px; background:#eef1f2; border:1px solid var(--line); border-radius:6px 6px 0 0; }
    label { display:grid; gap:5px; color:var(--muted); font-size:12px; font-weight:650; }
    select,input { min-width:230px; padding:9px 10px; border:1px solid #bfc9cf; border-radius:4px; background:#fff; color:var(--ink); }
    input { min-width:300px; }
    .count { margin-left:auto; color:var(--muted); font-size:13px; }
    .table-wrap { max-height:620px; overflow:auto; border:1px solid var(--line); background:#fff; }
    table { width:max-content; min-width:100%; border-collapse:collapse; font-size:12px; }
    th,td { padding:8px 9px; border-bottom:1px solid #e2e6e8; text-align:left; white-space:nowrap; font-variant-numeric:tabular-nums; }
    th { position:sticky; top:0; z-index:1; background:#30383d; color:#fff; font-size:11px; }
    tbody tr:nth-child(even) { background:#f7f8f8; } tbody tr:hover { background:var(--teal-soft); }
    .pager { display:flex; align-items:center; gap:10px; margin-top:14px; }
    .pager button { min-width:72px; padding:8px 10px; border:1px solid #c5d0d6; border-radius:4px; background:#fff; cursor:pointer; }
    .pager button:disabled { opacity:.45; cursor:default; }
    .gallery { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }
    .gallery.single { grid-template-columns:minmax(0,1fr); max-width:1550px; }
    .card { min-width:0; overflow:hidden; border:1px solid var(--line); border-radius:6px; background:#fff; }
    .card-head { display:flex; justify-content:space-between; gap:10px; padding:10px 12px; border-bottom:1px solid var(--line); font-size:13px; font-weight:650; }
    .card-head a { color:var(--blue); text-decoration:none; font-size:12px; font-weight:500; }
    .frame { display:grid; place-items:center; min-height:250px; aspect-ratio:1.587; padding:7px; }
    .single .frame { aspect-ratio:2/1; }
    img { display:block; width:100%; height:100%; object-fit:contain; cursor:zoom-in; }
    dialog { width:min(96vw,1850px); max-width:none; padding:0; border:1px solid #bfc9cf; background:#fff; }
    dialog::backdrop { background:rgba(18,24,30,.74); }
    .dialog-head { display:flex; justify-content:space-between; align-items:center; padding:10px 14px; border-bottom:1px solid var(--line); }
    .close { border:0; background:transparent; font-size:22px; cursor:pointer; }
    .dialog-image { max-height:calc(100vh - 75px); object-fit:contain; }
    @media(max-width:900px) { header { align-items:flex-start; flex-direction:column; } input { min-width:230px; } .count { margin-left:0; } .gallery { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <header><h1>Figure 4 Data</h1><div class="counts"><div class="count-box"><strong id="imageCount"></strong><span>PNG</span></div><div class="count-box"><strong id="csvCount"></strong><span>CSV</span></div></div></header>
  <main>
    <nav class="tabs"><button class="tab active" data-panel="data">Data</button><button class="tab" data-panel="images">Images</button></nav>
    <section class="panel active" id="data">
      <div class="toolbar"><label>CSV<select id="tableSelect"></select></label><label>Filter<input id="filter" type="search" autocomplete="off"></label><span class="count" id="count"></span></div>
      <div class="table-wrap"><table><thead id="thead"></thead><tbody id="tbody"></tbody></table></div>
      <div class="pager"><button id="previous">Previous</button><span id="page"></span><button id="next">Next</button><a id="download" href="" download>Download CSV</a></div>
    </section>
    <section class="panel" id="images">
      <div class="image-toolbar"><label>Figures<select id="imageSelect"></select></label></div>
      <div class="gallery" id="gallery"></div>
    </section>
  </main>
  <dialog id="viewer"><div class="dialog-head"><strong id="viewerTitle"></strong><button class="close" id="close">x</button></div><img class="dialog-image" id="viewerImage" alt=""></dialog>
  <script src="data/viewer_data.js"></script>
  <script>
    const bundle = window.FIGURE4_VIEWER_DATA;
    const byId = (id) => document.getElementById(id);
    const pageSize = 100;
    let tableKey = bundle.tables[0]?.key || '';
    let pageIndex = 0;
    let imageKey = 'degradation';
    const safe = (value) => String(value ?? '');
    function table() { return bundle.tables.find((item) => item.key === tableKey); }
    function fill(select, values, valueKey='key', labelKey='key') { select.innerHTML = values.map((item) => `<option value="${safe(item[valueKey])}">${safe(item[labelKey])}</option>`).join(''); }
    function renderTableSelect() { fill(byId('tableSelect'), bundle.tables, 'key', 'filename'); byId('tableSelect').value = tableKey; }
    function filteredRows() { const query = byId('filter').value.trim().toLowerCase(); const current = table(); return !query ? current.rows : current.rows.filter((row) => current.columns.some((column) => safe(row[column]).toLowerCase().includes(query))); }
    function renderTable() { const current = table(); const rows = filteredRows(); const totalPages = Math.max(1, Math.ceil(rows.length / pageSize)); pageIndex = Math.min(pageIndex, totalPages - 1); const visible = rows.slice(pageIndex * pageSize, (pageIndex + 1) * pageSize); byId('thead').innerHTML = `<tr>${current.columns.map((column) => `<th>${column}</th>`).join('')}</tr>`; byId('tbody').innerHTML = visible.map((row) => `<tr>${current.columns.map((column) => `<td>${safe(row[column])}</td>`).join('')}</tr>`).join(''); byId('count').textContent = `${rows.length} rows`; byId('page').textContent = `${pageIndex + 1} / ${totalPages}`; byId('previous').disabled = pageIndex === 0; byId('next').disabled = pageIndex >= totalPages - 1; byId('download').href = current.filename; }
    function imageGroups() { return bundle.image_groups[0].groups; }
    function renderImageControls() { const groups = imageGroups(); fill(byId('imageSelect'), groups, 'key', 'label'); if (!groups.some((item) => item.key === imageKey)) imageKey = groups[0].key; byId('imageSelect').value = imageKey; }
    function renderImages() { const group = imageGroups().find((item) => item.key === imageKey); const gallery = byId('gallery'); gallery.classList.toggle('single', group.key === 'random_cv'); gallery.innerHTML = group.images.map((image) => `<article class="card"><div class="card-head"><span>${image.name}</span><a href="${image.src}" target="_blank">Open PNG</a></div><div class="frame"><img src="${image.src}" alt="${image.name}" data-title="${image.name}"></div></article>`).join(''); gallery.querySelectorAll('img').forEach((image) => image.onclick = () => { byId('viewerTitle').textContent = image.dataset.title; byId('viewerImage').src = image.src; byId('viewerImage').alt = image.alt; byId('viewer').showModal(); }); }
    document.querySelectorAll('.tab').forEach((button) => button.onclick = () => { document.querySelectorAll('.tab').forEach((item) => item.classList.toggle('active', item === button)); document.querySelectorAll('.panel').forEach((panel) => panel.classList.toggle('active', panel.id === button.dataset.panel)); });
    byId('tableSelect').onchange = () => { tableKey = byId('tableSelect').value; pageIndex = 0; renderTable(); }; byId('filter').oninput = () => { pageIndex = 0; renderTable(); }; byId('previous').onclick = () => { pageIndex--; renderTable(); }; byId('next').onclick = () => { pageIndex++; renderTable(); };
    byId('imageSelect').onchange = () => { imageKey = byId('imageSelect').value; renderImages(); }; byId('close').onclick = () => byId('viewer').close(); byId('viewer').onclick = (event) => { if (event.target === byId('viewer')) byId('viewer').close(); };
    byId('imageCount').textContent = imageGroups().flatMap((group) => group.images).length; byId('csvCount').textContent = bundle.tables.length;
    renderTableSelect(); renderTable(); renderImageControls(); renderImages();
  </script>
</body>
</html>
'''


def _page() -> str:
    return '''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" href="data:,">
  <title>Figure 4 data viewer</title>
  <style>
    :root { --ink:#171a1d; --muted:#687078; --line:#d8dde1; --paper:#fff; --wash:#f4f6f7; --teal:#087f78; --teal-soft:#e5f3f1; }
    * { box-sizing:border-box; }
    body { margin:0; color:var(--ink); background:var(--wash); font-family:Aptos,"Segoe UI","Microsoft YaHei",sans-serif; font-size:14px; line-height:1.45; }
    button,select,input,a { font:inherit; letter-spacing:0; }
    .topbar { position:sticky; top:0; z-index:20; display:flex; align-items:center; justify-content:space-between; gap:20px; min-height:66px; padding:10px 24px; color:#fff; background:#20272b; border-bottom:3px solid var(--teal); }
    .brand h1 { margin:0; font-family:Georgia,"Times New Roman",serif; font-size:22px; font-weight:600; }
    .counts { display:flex; gap:18px; } .count { text-align:right; } .count strong { display:block; color:#fff; font-size:16px; } .count span { color:#aeb9be; font-size:11px; text-transform:uppercase; }
    .shell { max-width:1660px; margin:0 auto; padding:18px 24px 34px; }
    .modebar { display:flex; align-items:center; justify-content:space-between; gap:16px; margin-bottom:14px; }
    .segmented { display:inline-grid; grid-auto-flow:column; gap:2px; padding:3px; border:1px solid var(--line); border-radius:6px; background:#e7ebed; overflow-x:auto; }
    .segmented button { min-width:126px; padding:7px 12px; border:0; border-radius:4px; color:#40484e; background:transparent; cursor:pointer; }
    .segmented button[aria-selected="true"] { color:#fff; background:var(--teal); }
    .workspace { display:grid; grid-template-columns:minmax(230px,280px) minmax(0,1fr); min-height:720px; background:var(--paper); border:1px solid var(--line); border-radius:6px; overflow:hidden; }
    .sidebar { padding:18px; background:#eef1f2; border-right:1px solid var(--line); }
    .field { margin-bottom:16px; } .field[hidden] { display:none; }
    .field label { display:block; margin-bottom:5px; color:#596168; font-size:11px; font-weight:700; text-transform:uppercase; }
    select,input[type="search"] { width:100%; min-height:36px; padding:7px 9px; border:1px solid #bfc7cc; border-radius:4px; color:var(--ink); background:#fff; outline:none; }
    select:focus,input[type="search"]:focus { border-color:var(--teal); box-shadow:0 0 0 2px #cce9e6; }
    .sidebar-rule { height:1px; margin:18px 0; background:var(--line); }
    .download { display:inline-flex; align-items:center; justify-content:center; width:100%; min-height:34px; padding:6px 10px; border:1px solid #bcc5ca; border-radius:4px; color:#263238; background:#fff; text-decoration:none; }
    .content { min-width:0; padding:20px 22px 26px; }
    .figure-head { display:flex; align-items:flex-start; justify-content:space-between; gap:18px; margin-bottom:12px; }
    .figure-head h2 { margin:0; font-size:19px; font-weight:650; } .figure-head p { margin:3px 0 0; color:var(--muted); font-size:12px; overflow-wrap:anywhere; }
    .action { display:inline-flex; align-items:center; justify-content:center; min-height:34px; padding:6px 10px; border:1px solid #bcc5ca; border-radius:4px; color:#263238; background:#fff; text-decoration:none; }
    .figure-stage { display:grid; place-items:center; min-height:420px; padding:12px; background:#fafbfb; border:1px solid var(--line); border-radius:4px; overflow:hidden; }
    .figure-stage img { display:block; max-width:100%; max-height:780px; width:auto; height:auto; object-fit:contain; cursor:zoom-in; }
    .data-head { display:flex; align-items:end; justify-content:space-between; gap:14px; margin-top:22px; padding-top:18px; border-top:2px solid #20272b; }
    .data-head h3 { margin:0; font-size:16px; } .data-head p { margin:2px 0 0; color:var(--muted); font-size:12px; }
    .filters { display:grid; grid-template-columns:minmax(180px,1fr) repeat(3,minmax(130px,160px)); gap:8px; width:min(760px,100%); }
    .process { margin-top:18px; padding:14px 16px; border:1px solid var(--line); border-left:3px solid var(--teal); background:#f5f8f8; }
    .process[hidden] { display:none; } .process h3 { margin:0 0 8px; font-size:15px; } .process p { margin:6px 0; color:#4d565b; font-size:12px; } .process ol { margin:8px 0 8px 20px; padding:0; color:#4d565b; font-size:12px; } .process-links { display:flex; flex-wrap:wrap; gap:8px 16px; margin-top:10px; } .process-links a { color:#087f78; font-size:12px; }
    .table-wrap { max-height:520px; margin-top:10px; overflow:auto; border:1px solid var(--line); }
    table { width:100%; border-collapse:collapse; white-space:nowrap; font-size:12px; font-variant-numeric:tabular-nums; }
    th,td { padding:7px 9px; border-bottom:1px solid #e2e6e8; text-align:right; } th:first-child,td:first-child { text-align:left; }
    th { position:sticky; top:0; z-index:2; color:#fff; background:#30383d; font-size:11px; font-weight:650; }
    tbody tr:nth-child(even) { background:#f7f8f8; } tbody tr:hover { background:var(--teal-soft); }
    dialog { width:min(96vw,1800px); max-width:none; height:min(94vh,1200px); padding:0; border:0; border-radius:6px; background:#fff; box-shadow:0 24px 70px rgba(0,0,0,.35); }
    dialog::backdrop { background:rgba(17,22,25,.82); } .dialog-bar { display:flex; align-items:center; justify-content:space-between; padding:10px 14px; color:#fff; background:#20272b; }
    .close { width:34px; height:34px; border:0; border-radius:4px; color:#fff; background:#465158; font-size:20px; cursor:pointer; }
    .dialog-image { display:grid; place-items:center; height:calc(100% - 54px); overflow:auto; padding:12px; } .dialog-image img { max-width:none; width:auto; min-width:min(100%,1000px); height:auto; }
    @media(max-width:900px) { .topbar { align-items:flex-start; padding:10px 14px; } .counts { gap:10px; } .shell { padding:12px; } .modebar { align-items:stretch; flex-direction:column; } .workspace { grid-template-columns:1fr; } .sidebar { border-right:0; border-bottom:1px solid var(--line); } .content { padding:16px 12px 22px; } .figure-head,.data-head { align-items:stretch; flex-direction:column; } .filters { grid-template-columns:1fr; width:100%; } .figure-stage { min-height:250px; padding:4px; } }
  </style>
</head>
<body>
  <header class="topbar"><div class="brand"><h1>Figure 4 data</h1></div><div class="counts"><div class="count"><strong id="imageCount"></strong><span>PNG</span></div><div class="count"><strong id="workbookCount"></strong><span>XLSX</span></div><div class="count"><strong id="taskCount"></strong><span>Tasks</span></div></div></header>
  <main class="shell">
    <div class="modebar"><div class="segmented" role="tablist" aria-label="Image category"><button data-category="overview" aria-selected="true">Overview</button><button data-category="combined" aria-selected="false">Combined</button><button data-category="random_cv" aria-selected="false">Random-CV</button></div></div>
    <section class="workspace">
      <aside class="sidebar"><div class="field" id="overviewImageField"><label for="overviewImageSelect">Overview figure</label><select id="overviewImageSelect"></select></div><div class="field" id="taskField"><label for="taskSelect">Task</label><select id="taskSelect"></select></div><div class="field"><label for="tableSelect">Displayed data</label><select id="tableSelect"></select></div><div class="sidebar-rule"></div><a class="download" id="workbookLink" download>Download Excel</a></aside>
      <div class="content"><div class="figure-head"><div><h2 id="figureTitle"></h2><p id="figurePath"></p></div><a class="action" id="imageLink" target="_blank">Open PNG</a></div><div class="figure-stage"><img id="figureImage" alt=""></div><section class="process" id="overviewProcess" hidden></section><div class="data-head"><div><h3 id="tableTitle"></h3><p><span id="visibleRows"></span> rows · <span id="workbookPath"></span></p></div><div class="filters"><input id="searchInput" type="search" placeholder="Search table"><select id="methodFilter"></select><select id="axisFilter"></select><select id="fractionFilter"></select></div></div><div class="table-wrap"><table><thead id="tableHead"></thead><tbody id="tableBody"></tbody></table></div></div>
    </section>
  </main>
  <dialog id="imageDialog"><div class="dialog-bar"><strong id="dialogTitle"></strong><button class="close" id="closeDialog">x</button></div><div class="dialog-image"><img id="dialogImage" alt=""></div></dialog>
  <script src="data/viewer_data.js"></script>
  <script>
    const DATA = window.FIGURE4_VIEWER_DATA;
    const byId = (id) => document.getElementById(id);
    const groups = DATA.image_groups[0].groups;
    const defaults = { overview:'overview_degradation', combined:'task_degradation', random_cv:'random_cv_mae' };
    const state = { category:'overview', overviewImage:'', task:'', tableKey:'' };
    const text = (value) => String(value ?? '');
    function currentGroup() { return groups.find((group) => group.key === state.category); }
    function tasks() { return groups.find((group) => group.key === 'combined').images.map((image) => image.name); }
    function currentImage() { const images=currentGroup().images; if (state.category==='overview') return images.find((image) => image.name===state.overviewImage) || images[0]; if (state.category==='random_cv') return images[0]; return images.find((image) => image.name===state.task) || images[0]; }
    function currentTable() { return DATA.tables.find((table) => table.key === state.tableKey); }
    function fill(select, values, value, label=value) { select.innerHTML=values.map((item) => `<option value="${text(item[value])}">${text(item[label])}</option>`).join(''); }
    function setCategory(category) { state.category=category; state.tableKey=''; byId('searchInput').value=''; byId('methodFilter').value=''; byId('axisFilter').value=''; byId('fractionFilter').value=''; document.querySelectorAll('.segmented button').forEach((button) => button.setAttribute('aria-selected',String(button.dataset.category===category))); render(); }
    function renderSelectors() { const byTask=state.category==='combined'; const overview=state.category==='overview'; byId('taskField').hidden=!byTask; byId('overviewImageField').hidden=!overview; fill(byId('taskSelect'),tasks().map((name) => ({key:name,label:name})),'key','label'); if (!state.task) state.task=tasks()[0]; byId('taskSelect').value=state.task; const overviewImages=groups.find((group) => group.key==='overview').images; fill(byId('overviewImageSelect'),overviewImages.map((image) => ({key:image.name,label:image.name})),'key','label'); if (!state.overviewImage) state.overviewImage=overviewImages[0].name; byId('overviewImageSelect').value=state.overviewImage; const allowedKeys=currentImage().table_keys || [defaults[state.category]]; const allowedTables=allowedKeys.map((key) => DATA.tables.find((table) => table.key===key)).filter(Boolean); if (!allowedKeys.includes(state.tableKey)) state.tableKey=allowedKeys[0]; fill(byId('tableSelect'),allowedTables,'key','filename'); byId('tableSelect').value=state.tableKey; }
    function methodKey(table) { return table.columns.includes('method') ? 'method' : table.columns.includes('display_method') ? 'display_method' : table.columns.includes('model') ? 'model' : table.columns.includes('model_family') ? 'model_family' : ''; }
    function valuesFor(table,column) { return table.columns.includes(column) ? [...new Set(table.rows.map((row) => text(row[column])).filter(Boolean))] : []; }
    function optionHtml(values,label) { return [`<option value="">${label}</option>`,...values.map((value) => `<option value="${text(value)}">${text(value)}</option>`)].join(''); }
    function renderTableFilters(table) { const key=methodKey(table); const methods=key ? [...new Set(table.rows.map((row) => text(row[key])).filter(Boolean))].sort() : []; const axes=valuesFor(table,'failure_axis').sort(); const fractions=valuesFor(table,'training_fraction').sort((a,b) => Number(a)-Number(b)); const oldMethod=byId('methodFilter').value; const oldAxis=byId('axisFilter').value; const oldFraction=byId('fractionFilter').value; byId('methodFilter').innerHTML=optionHtml(methods,'All methods'); byId('axisFilter').innerHTML=optionHtml(axes,'All axes'); byId('fractionFilter').innerHTML=optionHtml(fractions,'All fractions'); byId('methodFilter').value=methods.includes(oldMethod) ? oldMethod : ''; byId('axisFilter').value=axes.includes(oldAxis) ? oldAxis : ''; byId('fractionFilter').value=fractions.includes(oldFraction) ? oldFraction : ''; }
    function filteredRows() { const table=currentTable(); const query=byId('searchInput').value.trim().toLowerCase(); const method=byId('methodFilter').value; const axis=byId('axisFilter').value; const fraction=byId('fractionFilter').value; const key=methodKey(table); return table.rows.filter((row) => { if (state.category==='combined' && table.columns.includes('task_id') && text(row.task_id)!==state.task) return false; if (method && key && text(row[key])!==method) return false; if (axis && table.columns.includes('failure_axis') && text(row.failure_axis)!==axis) return false; if (fraction && table.columns.includes('training_fraction') && text(row.training_fraction)!==fraction) return false; return !query || table.columns.some((column) => text(row[column]).toLowerCase().includes(query)); }); }
    function renderTable() { const table=currentTable(); const image=currentImage(); renderTableFilters(table); const rows=filteredRows(); byId('tableTitle').textContent=table.filename; byId('workbookPath').textContent=image.data; byId('visibleRows').textContent=rows.length; byId('workbookLink').href=image.data; byId('tableHead').innerHTML=`<tr>${table.columns.map((column) => `<th>${column}</th>`).join('')}</tr>`; byId('tableBody').innerHTML=rows.map((row) => `<tr>${table.columns.map((column) => `<td>${text(row[column])}</td>`).join('')}</tr>`).join(''); }
    function renderFigure() { const image=currentImage(); const overviewTitles={figure4_performance_degradation_ratio:'Degradation-ratio overview',figure4_absolute_ood_mae_overview:'Absolute OOD MAE overview',figure4_all_overviews:'Combined four-panel overview'}; byId('figureTitle').textContent=state.category==='overview' ? overviewTitles[image.name] : state.category==='random_cv' ? 'Random-CV overview' : `${state.task} · ${currentGroup().label}`; byId('figurePath').textContent=image.src; byId('figureImage').src=image.src; byId('figureImage').alt=byId('figureTitle').textContent; byId('imageLink').href=image.src; byId('overviewProcess').hidden=state.category!=='overview'; byId('overviewProcess').innerHTML=state.category==='overview' ? DATA.overview_process : ''; }
    function render() { renderSelectors(); renderFigure(); renderTable(); }
    document.querySelectorAll('.segmented button').forEach((button) => button.onclick=() => setCategory(button.dataset.category));
    byId('overviewImageSelect').onchange=(event) => { state.overviewImage=event.target.value; state.tableKey=''; render(); };
    byId('taskSelect').onchange=(event) => { state.task=event.target.value; state.tableKey=''; render(); };
    byId('tableSelect').onchange=(event) => { state.tableKey=event.target.value; renderTable(); };
    byId('searchInput').oninput=renderTable; byId('methodFilter').onchange=renderTable; byId('axisFilter').onchange=renderTable; byId('fractionFilter').onchange=renderTable;
    byId('figureImage').onclick=() => { byId('dialogTitle').textContent=byId('figureTitle').textContent; byId('dialogImage').src=currentImage().src; byId('imageDialog').showModal(); };
    byId('closeDialog').onclick=() => byId('imageDialog').close(); byId('imageDialog').onclick=(event) => { if (event.target===byId('imageDialog')) byId('imageDialog').close(); };
    byId('imageCount').textContent=groups.flatMap((group) => group.images).length; byId('workbookCount').textContent=DATA.workbook_count; byId('taskCount').textContent=tasks().length; state.task=tasks()[0]; state.tableKey=defaults.overview; render();
  </script>
</body>
</html>
'''


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a static Figure 4 data viewer.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpt-root", type=Path, default=DEFAULT_GPT_ROOT)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260707)
    parser.add_argument(
        "--recompute-bootstrap",
        action="store_true",
        help="Ignore a valid overview cache and recompute all bootstrap summaries.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(
        json.dumps(
            export_viewer(
                args.source_root,
                args.output_root,
                gpt_root=args.gpt_root,
                bootstrap_iterations=args.bootstrap_iterations,
                random_seed=args.random_seed,
                recompute_bootstrap=args.recompute_bootstrap,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
