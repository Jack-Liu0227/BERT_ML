from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any


DISPLAY_METHODS = [
    "Traditional-best",
    "BERT-best",
    "TabPFN Numeric",
    "TabPFN Text",
    "LLM-Prop",
]

EVIDENCE_FILES = [
    "degradation_ratio_by_repeat.csv",
    "selected_degradation_ratio_by_repeat.csv",
    "representative_model_selection.csv",
    "selected_method_curve_summary.csv",
    "selected_method_sensitivity_summary.csv",
    "coverage_report.csv",
]


def build_figure4_explainer(
    output_root: str | Path,
    *,
    output_dir: str | Path | None = None,
    excluded_tasks: tuple[str, ...] = (),
) -> Path:
    """Build bilingual Figure 4 data explorers with live-linked PNG galleries."""
    root = Path(output_root).resolve()
    destination = Path(output_dir).resolve() if output_dir else root / "explainer"
    csv_dir = root / "csv"
    figures_dir = root / "figures"
    raw_path = csv_dir / "degradation_ratio_by_repeat.csv"
    if not raw_path.exists():
        raise FileNotFoundError("Figure 4 degradation_ratio_by_repeat.csv is missing")

    raw_rows = _read_csv_records(raw_path)
    selected_path = csv_dir / "selected_degradation_ratio_by_repeat.csv"
    selected_rows = _read_csv_records(selected_path if selected_path.exists() else raw_path)
    selection_rows = _read_csv_records(csv_dir / "representative_model_selection.csv")
    curve_rows = _read_csv_records(csv_dir / "selected_method_curve_summary.csv")
    sensitivity_rows = _read_csv_records(csv_dir / "selected_method_sensitivity_summary.csv")
    excluded = set(excluded_tasks)
    if excluded:
        selected_rows = [row for row in selected_rows if str(row.get("task_id", "")) not in excluded]
        selection_rows = [row for row in selection_rows if str(row.get("task_id", "")) not in excluded]
    tasks = sorted({str(row.get("task_id", "")) for row in selected_rows if row.get("task_id")})
    fractions = sorted(
        {
            float(row["training_fraction"])
            for row in selected_rows
            if _is_number(row.get("training_fraction"))
        }
    )
    axes = sorted(
        {str(row.get("failure_axis", "")) for row in selected_rows if row.get("failure_axis")}
    )
    payload = _json_safe(
        {
            "metadata": {
                "stage": "figure4-explainer",
                "run_count": 4510,
                "ratio_row_count": len(raw_rows),
                "task_count": len(tasks),
                "display_methods": DISPLAY_METHODS,
                "tasks": tasks,
                "training_fractions": fractions,
                "failure_axes": axes,
                "selection_rule": "Lowest mean Random-CV MAE per task and training fraction; ties use SD then model name.",
            },
            "selection": selection_rows,
            "results": selected_rows,
            "curve_summary": curve_rows,
            "sensitivity_summary": sensitivity_rows,
        }
    )

    ready = destination.parent / f".{destination.name}.ready"
    if ready.exists():
        shutil.rmtree(ready)
    (ready / "data").mkdir(parents=True)
    (ready / "evidence").mkdir(parents=True)
    source_hashes: dict[str, str] = {}
    for name in EVIDENCE_FILES:
        source = csv_dir / name
        if source.exists():
            target = ready / "evidence" / name
            shutil.copy2(source, target)
            source_hashes[str(source)] = _sha256(source)
    for directory in [
        figures_dir / "figure4_absolute_mae_tasks",
        figures_dir / "figure4_ratio_tasks",
        figures_dir / "random_cv_mae",
    ]:
        if directory.exists():
            for source in sorted(directory.glob("*.png")):
                source_hashes[str(source)] = _sha256(source)

    data_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    _write(ready / "data" / "figure4_data.js", f"window.FIGURE4_DATA = {data_json};\n")
    _write(ready / "data" / "figure4_data_en.js", f"window.FIGURE4_DATA = {data_json};\n")
    _write(ready / "index.html", _page(language="zh", data_script="data/figure4_data.js"))
    _write(ready / "index_en.html", _page(language="en", data_script="data/figure4_data_en.js"))
    _write(
        ready / "README.md",
        "# Figure 4 Explainer\n\n"
        "- Chinese: open `index.html`\n"
        "- English: open `index_en.html`\n\n"
        "The pages link directly to the formal Figure 4 PNG outputs.\n",
    )
    output_hashes = {
        str(path.relative_to(ready)).replace("\\", "/"): _sha256(path)
        for path in sorted(path for path in ready.rglob("*") if path.is_file())
    }
    manifest = {
        "schema_version": 2,
        "stage": "figure4-explainer",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_hashes,
        "output_sha256": output_hashes,
        "output_file_count": len(output_hashes) + 1,
    }
    _write(
        ready / "artifact_manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    if destination.exists():
        shutil.rmtree(destination)
    ready.rename(destination)
    return destination


def _page(*, language: str, data_script: str) -> str:
    chinese = language == "zh"
    labels = {
        "title": "Figure 4 训练规模结果浏览器" if chinese else "Figure 4 Training-Size Results",
        "subtitle": "按任务、训练比例和失效轴审查代表模型结果。" if chinese else "Inspect representative-model results by task, training fraction, and failure axis.",
        "runs": "运行次数" if chinese else "Runs",
        "records": "Ratio 记录" if chinese else "Ratio records",
        "tasks": "展示任务" if chinese else "Displayed tasks",
        "methods": "展示方法" if chinese else "Displayed methods",
        "details": "数据明细" if chinese else "Data explorer",
        "task": "任务" if chinese else "Task",
        "fraction": "训练比例" if chinese else "Training fraction",
        "axis": "失效轴" if chinese else "Failure axis",
        "method": "方法" if chinese else "Method",
        "model": "实际模型" if chinese else "Selected model",
        "random": "Random-CV MAE",
        "ood": "OOD MAE",
        "ratio": "Degradation ratio",
        "repeats": "重复数" if chinese else "Repeats",
        "focused": "当前任务图像" if chinese else "Selected-task figures",
        "mae": "绝对 MAE" if chinese else "Absolute MAE",
        "ratio_fig": "退化比" if chinese else "Degradation Ratio",
        "galleries": "图像总览" if chinese else "Figure galleries",
        "random_fig": "Random-CV 总图" if chinese else "Random-CV overview",
        "open_png": "打开 PNG" if chinese else "Open PNG",
        "evidence": "数据与哈希证据" if chinese else "Evidence downloads",
        "raw_data": "完整 ratio 数据" if chinese else "Full degradation-ratio data",
        "selection_data": "代表模型选择表" if chinese else "Representative-model selection",
        "curve_data": "曲线汇总" if chinese else "Selected curve summary",
        "sensitivity_data": "敏感度汇总" if chinese else "Sensitivity summary",
        "manifest": "输出哈希 manifest" if chinese else "Artifact hash manifest",
        "close": "关闭" if chinese else "Close",
        "no_data": "没有可显示的数据。" if chinese else "No data available.",
    }
    template = r'''<!doctype html>
<html lang="__LANG__">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" href="data:,">
  <title>__TITLE__</title>
  <style>
    :root { color-scheme: light; --paper:#f3f4f1; --surface:#fff; --ink:#18212a; --muted:#65717c; --line:#d5dce1; --blue:#356f9f; --green:#238b7e; --shadow:0 8px 24px rgba(23,34,44,.08); }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--paper); color:var(--ink); font-family:Aptos,"Segoe UI",sans-serif; letter-spacing:0; }
    button,select,a { font:inherit; }
    header { padding:28px clamp(18px,4vw,56px) 24px; background:var(--surface); border-bottom:1px solid var(--line); }
    h1 { margin:0 0 8px; font-family:Georgia,"Times New Roman",serif; font-size:clamp(27px,3vw,42px); font-weight:600; }
    h2 { margin:0 0 16px; font-size:20px; }
    .subtitle { color:var(--muted); }
    main { max-width:1800px; margin:auto; padding:24px clamp(18px,4vw,56px) 64px; }
    .summary { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin-bottom:20px; }
    .metric { padding:16px; background:var(--surface); border:1px solid var(--line); border-radius:6px; }
    .metric strong { display:block; margin-bottom:4px; font-size:25px; }
    .metric span { color:var(--muted); font-size:13px; }
    section { margin-top:20px; padding:20px; background:var(--surface); border:1px solid var(--line); border-radius:6px; box-shadow:var(--shadow); }
    .controls { display:flex; flex-wrap:wrap; gap:12px; margin-bottom:18px; }
    label { display:grid; gap:5px; color:var(--muted); font-size:13px; font-weight:650; }
    select { min-width:200px; padding:9px 10px; border:1px solid #b8c2ca; border-radius:4px; background:#fff; color:var(--ink); }
    .table-wrap { overflow:auto; }
    table { width:100%; border-collapse:collapse; font-size:13px; }
    th,td { padding:9px 8px; border-bottom:1px solid #e4e8eb; text-align:left; white-space:nowrap; }
    th { background:#f6f8f9; color:#4f5d68; }
    .focused-grid,.gallery { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }
    .gallery.single { grid-template-columns:minmax(0,1fr); }
    .figure-card { min-width:0; overflow:hidden; border:1px solid var(--line); border-radius:6px; background:#fff; }
    .card-head { display:flex; align-items:center; justify-content:space-between; gap:12px; min-height:46px; padding:10px 13px; border-bottom:1px solid var(--line); }
    .card-title { margin:0; font-size:14px; overflow-wrap:anywhere; }
    .open-link { color:var(--blue); text-decoration:none; font-size:12px; white-space:nowrap; }
    .open-link:hover { text-decoration:underline; }
    .figure-frame { display:grid; place-items:center; min-height:250px; aspect-ratio:1.587; padding:7px; background:#fff; }
    .single .figure-frame { aspect-ratio:2/1; }
    .figure-image { display:block; width:100%; height:100%; object-fit:contain; cursor:zoom-in; }
    .tabs { display:flex; gap:4px; margin:-4px 0 18px; border-bottom:1px solid var(--line); overflow-x:auto; }
    .tab { min-height:44px; padding:0 14px; border:0; border-bottom:3px solid transparent; background:transparent; color:var(--muted); cursor:pointer; font-weight:650; }
    .tab[aria-selected="true"] { color:var(--ink); border-bottom-color:var(--blue); }
    .tab-count { margin-left:6px; font-weight:500; }
    .selection-note { margin-top:12px; color:var(--muted); font-size:13px; }
    .downloads { display:flex; flex-wrap:wrap; gap:9px; }
    .downloads a { padding:8px 10px; border:1px solid #c6d0d7; border-radius:4px; color:var(--blue); background:#f8fafb; text-decoration:none; }
    dialog { width:min(96vw,1800px); max-width:none; padding:0; border:1px solid #bcc5cc; border-radius:6px; background:#fff; box-shadow:0 24px 80px rgba(20,28,36,.24); }
    dialog::backdrop { background:rgba(18,24,30,.74); }
    .dialog-head { display:flex; align-items:center; justify-content:space-between; padding:10px 14px; border-bottom:1px solid var(--line); }
    .close-button { width:36px; height:36px; border:0; background:transparent; font-size:25px; cursor:pointer; }
    .dialog-image { display:block; width:100%; max-height:calc(100vh - 90px); object-fit:contain; background:#fff; }
    @media(max-width:900px) { .summary { grid-template-columns:repeat(2,minmax(0,1fr)); } .focused-grid,.gallery { grid-template-columns:1fr; } section { padding:15px; } }
  </style>
</head>
<body>
  <header><h1>__TITLE__</h1><div class="subtitle">__SUBTITLE__</div></header>
  <main>
    <div class="summary">
      <div class="metric"><strong id="runCount">-</strong><span>__RUNS__</span></div>
      <div class="metric"><strong id="rowCount">-</strong><span>__RECORDS__</span></div>
      <div class="metric"><strong id="taskCount">-</strong><span>__TASKS__</span></div>
      <div class="metric"><strong id="methodCount">-</strong><span>__METHODS__</span></div>
    </div>
    <section>
      <h2>__DETAILS__</h2>
      <div class="controls">
        <label>__TASK__<select id="task"></select></label>
        <label>__FRACTION__<select id="fraction"></select></label>
        <label>__AXIS__<select id="axis"></select></label>
      </div>
      <div class="table-wrap"><table><thead><tr><th>__METHOD__</th><th>__MODEL__</th><th>__RANDOM__</th><th>__OOD__</th><th>__RATIO__</th><th>__REPEATS__</th></tr></thead><tbody id="rows"></tbody></table></div>
      <div class="selection-note" id="selectionDetails"></div>
    </section>
    <section>
      <h2>__FOCUSED__</h2>
      <div class="focused-grid">
        <article class="figure-card"><div class="card-head"><h3 class="card-title">__MAE__</h3><a class="open-link" id="focusedMaeLink" target="_blank">__OPEN_PNG__</a></div><div class="figure-frame"><img class="figure-image live-image" id="focusedMae" alt="__MAE__"></div></article>
        <article class="figure-card"><div class="card-head"><h3 class="card-title">__RATIO_FIG__</h3><a class="open-link" id="focusedRatioLink" target="_blank">__OPEN_PNG__</a></div><div class="figure-frame"><img class="figure-image live-image" id="focusedRatio" alt="__RATIO_FIG__"></div></article>
      </div>
    </section>
    <section>
      <h2>__GALLERIES__</h2>
      <div class="tabs" id="galleryTabs"></div>
      <div class="gallery" id="gallery"></div>
    </section>
    <section><h2>__EVIDENCE__</h2><div class="downloads">
      <a href="evidence/degradation_ratio_by_repeat.csv">__RAW_DATA__</a>
      <a href="evidence/representative_model_selection.csv">__SELECTION_DATA__</a>
      <a href="evidence/selected_method_curve_summary.csv">__CURVE_DATA__</a>
      <a href="evidence/selected_method_sensitivity_summary.csv">__SENSITIVITY_DATA__</a>
      <a href="artifact_manifest.json">__MANIFEST__</a>
    </div></section>
  </main>
  <dialog id="viewer"><div class="dialog-head"><strong id="viewerTitle"></strong><button class="close-button" id="closeViewer" aria-label="__CLOSE__">×</button></div><img class="dialog-image" id="viewerImage" alt="Expanded figure"></dialog>
  <script src="__DATA_SCRIPT__"></script>
  <script>
    const data = window.FIGURE4_DATA;
    const text = __LABELS__;
    const methods = data.metadata.display_methods;
    const visibleImages = new Set();
    let galleryKey = 'mae';
    const byId = (id) => document.getElementById(id);
    const safeName = (value) => String(value).replace(/[^A-Za-z0-9._-]/g, '_');
    const number = (value) => value === null || value === undefined || Number.isNaN(Number(value)) ? '-' : Number(value).toFixed(4);
    const figurePath = (kind, task) => kind === 'mae' ? `../figures/figure4_absolute_mae_tasks/${safeName(task)}.png` : `../figures/figure4_ratio_tasks/${safeName(task)}.png`;
    const versioned = (source) => `${source}${source.includes('?') ? '&' : '?'}v=${Date.now()}`;
    const observer = new IntersectionObserver((entries) => entries.forEach((entry) => { if (entry.isIntersecting) { visibleImages.add(entry.target); refreshImage(entry.target); } else { visibleImages.delete(entry.target); } }), { rootMargin: '220px' });
    function refreshImage(image) { if (!image?.dataset.source || image.dataset.loading === 'true') return; image.dataset.loading = 'true'; const candidate = new Image(); candidate.onload = () => { image.src = candidate.src; image.dataset.loading = 'false'; }; candidate.onerror = () => { image.dataset.loading = 'false'; }; candidate.src = versioned(image.dataset.source); }
    function attachImage(image, source, title) { image.dataset.source = source; image.alt = title; observer.observe(image); image.onclick = () => { byId('viewerTitle').textContent = title; byId('viewerImage').src = versioned(source); byId('viewer').showModal(); }; refreshImage(image); }
    function fill(id, values) { byId(id).innerHTML = values.map((value) => `<option value="${String(value)}">${String(value)}</option>`).join(''); }
    function renderDetails() {
      const task = byId('task').value;
      const fraction = Number(byId('fraction').value);
      const axis = byId('axis').value;
      const rows = data.results.filter((row) => row.task_id === task && Number(row.training_fraction) === fraction && row.failure_axis === axis);
      byId('rows').innerHTML = methods.map((method) => { const values = rows.filter((row) => row.display_method === method); if (!values.length) return `<tr><td>${method}</td><td colspan="5">-</td></tr>`; const models = [...new Set(values.map((row) => row.model))].join(', '); const mean = (key) => values.reduce((sum, row) => sum + Number(row[key] || 0), 0) / values.length; return `<tr><td>${method}</td><td>${models}</td><td>${number(mean('random_cv_mae'))}</td><td>${number(mean('ood_mae'))}</td><td>${number(mean('degradation_ratio'))}</td><td>${values.length}</td></tr>`; }).join('');
      const selected = data.selection.filter((row) => row.task_id === task && Number(row.training_fraction) === fraction);
      byId('selectionDetails').textContent = selected.map((row) => `${row.model_family}: ${row.selected_model || '-'} (${number(row.selection_random_cv_mae_mean)})`).join(' · ');
      const mae = figurePath('mae', task); const ratio = figurePath('ratio', task);
      byId('focusedMaeLink').href = mae; byId('focusedRatioLink').href = ratio;
      attachImage(byId('focusedMae'), mae, `${text.mae}: ${task}`); attachImage(byId('focusedRatio'), ratio, `${text.ratio_fig}: ${task}`);
    }
    function galleryGroups() { return [
      { key:'mae', label:text.mae, images:data.metadata.tasks.map((task) => ({ name:task, src:figurePath('mae', task) })) },
      { key:'ratio', label:text.ratio_fig, images:data.metadata.tasks.map((task) => ({ name:task, src:figurePath('ratio', task) })) },
      { key:'random', label:text.random_fig, images:[{ name:text.random_fig, src:'../figures/random_cv_mae/random_cv_mae_all_tasks.png' }] }
    ]; }
    function renderGalleryTabs() { byId('galleryTabs').innerHTML = galleryGroups().map((group) => `<button class="tab" data-key="${group.key}" aria-selected="${group.key === galleryKey}">${group.label}<span class="tab-count">${group.images.length}</span></button>`).join(''); byId('galleryTabs').querySelectorAll('.tab').forEach((tab) => tab.onclick = () => { galleryKey = tab.dataset.key; renderGalleryTabs(); renderGallery(); }); }
    function renderGallery() { const group = galleryGroups().find((item) => item.key === galleryKey); const gallery = byId('gallery'); gallery.classList.toggle('single', galleryKey === 'random'); gallery.innerHTML = group.images.map((image) => `<article class="figure-card"><div class="card-head"><h3 class="card-title">${image.name}</h3><a class="open-link" href="${image.src}" target="_blank">${text.open_png}</a></div><div class="figure-frame"><img class="figure-image live-image" data-source="${image.src}" alt="${image.name}"></div></article>`).join(''); gallery.querySelectorAll('.live-image').forEach((image) => attachImage(image, image.dataset.source, image.alt)); }
    fill('task', data.metadata.tasks); fill('fraction', data.metadata.training_fractions); fill('axis', data.metadata.failure_axes);
    byId('runCount').textContent = data.metadata.run_count; byId('rowCount').textContent = data.metadata.ratio_row_count; byId('taskCount').textContent = data.metadata.task_count; byId('methodCount').textContent = methods.length;
    ['task','fraction','axis'].forEach((id) => byId(id).addEventListener('change', renderDetails));
    byId('closeViewer').onclick = () => byId('viewer').close(); byId('viewer').onclick = (event) => { if (event.target === byId('viewer')) byId('viewer').close(); };
    setInterval(() => visibleImages.forEach(refreshImage), 5000);
    if (data.metadata.tasks.length) { renderDetails(); renderGalleryTabs(); renderGallery(); }
  </script>
</body>
</html>'''
    replacements = {
        "__LANG__": "zh-CN" if chinese else "en",
        "__DATA_SCRIPT__": data_script,
        "__LABELS__": json.dumps(labels, ensure_ascii=False, sort_keys=True),
    }
    replacements.update({f"__{key.upper()}__": str(value) for key, value in labels.items()})
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template


def _read_csv_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{key: _coerce(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def _coerce(value: str | None) -> Any:
    if value is None or value == "":
        return value
    try:
        number = float(value)
    except ValueError:
        return value
    if math.isfinite(number) and number.is_integer() and not value.startswith("0"):
        return int(number)
    return number if math.isfinite(number) else None


def _is_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
