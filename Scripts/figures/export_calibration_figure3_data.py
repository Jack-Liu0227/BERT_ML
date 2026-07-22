from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image, ImageStat


OVERVIEW_IMAGES = (
    "calibration_main.png",
    "calibration_by_protocol.png",
    "calibration_by_method.png",
    "calibration_adjacent_tier_diagnostics.png",
)
TASK_IMAGES = (
    "calibration_by_protocol.png",
    "calibration_adjacent_tier_diagnostics.png",
)
PROTOCOL_ORDER = (
    "RandomCV",
    "LOCO",
    "Extrapolation",
    "SparseXsingle",
    "SparseXcluster",
    "SparseYsingle",
    "SparseYcluster",
)
METHOD_ORDER = (
    "Traditional-best",
    "BERT-best",
    "LLM-Prop",
    "TabPFN-2.5-Plus-Numeric",
    "TabPFN-2.5-Plus-Text",
    "GPT-5.4",
)
TIER_ORDER = ("High", "Medium", "Low")
TASK_LABELS = {
    "Al__aluminum__UTS_MPa": "Al UTS",
    "HEA__hea__El": "HEA elongation",
    "HEA__hea__UTS_MPa": "HEA UTS",
    "Matbench_Steel__matbench_steels_ood__yield_strength": "Matbench Steel yield strength",
    "Steel__steel__El": "Steel elongation",
    "Steel__steel__UTS_MPa": "Steel UTS",
    "Ti__titanium__El": "Ti elongation",
    "Ti__titanium__UTS_MPa": "Ti UTS",
}


def _require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing delivery sources: " + "; ".join(missing))


def _sort_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    categories = {
        "protocol": PROTOCOL_ORDER,
        "method": METHOD_ORDER,
        "confidence_tier": TIER_ORDER,
    }
    sort_columns: list[str] = []
    for column, order in categories.items():
        if column in result.columns:
            result[column] = pd.Categorical(result[column], categories=order, ordered=True)
            sort_columns.append(column)
    if sort_columns:
        result = result.sort_values(sort_columns, kind="stable")
    for column in sort_columns:
        result[column] = result[column].astype("object")
    return result.reset_index(drop=True)


def _write_csv(frame: pd.DataFrame, path: Path, columns: list[str]) -> dict[str, object]:
    output = _sort_frame(frame.loc[:, columns])
    output.to_csv(path, index=False, encoding="utf-8-sig")
    records = json.loads(output.to_json(orient="records", force_ascii=False))
    return {
        "file": path.as_posix(),
        "columns": columns,
        "rows": records,
        "row_count": len(output),
    }


def _relative_table(table: dict[str, object], root: Path) -> dict[str, object]:
    result = dict(table)
    result["file"] = Path(str(result["file"])).relative_to(root).as_posix()
    return result


def _validate_png(path: Path) -> None:
    with Image.open(path) as image:
        image.load()
        if image.format != "PNG":
            raise ValueError(f"Not a PNG image: {path}")
        if image.width < 800 or image.height < 500:
            raise ValueError(f"Image is unexpectedly small: {path} ({image.size})")
        variance = float(ImageStat.Stat(image.convert("L")).var[0])
        if variance < 10.0:
            raise ValueError(f"Image appears blank: {path} (variance={variance})")


def _html_template(payload: dict[str, object]) -> str:
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    template = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Figure 3 data viewer</title>
  <style>
    :root {
      --ink: #171a1d;
      --muted: #687078;
      --line: #d8dde1;
      --paper: #ffffff;
      --wash: #f4f6f7;
      --teal: #087f78;
      --teal-soft: #e5f3f1;
      --orange: #d55e00;
      --blue: #0072b2;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      color: var(--ink);
      background: var(--wash);
      font-family: Aptos, "Segoe UI", "Microsoft YaHei", sans-serif;
      font-size: 14px;
      line-height: 1.45;
      letter-spacing: 0;
    }
    button, select, input { font: inherit; letter-spacing: 0; }
    button, select, input, a { -webkit-tap-highlight-color: transparent; }
    .topbar {
      position: sticky;
      top: 0;
      z-index: 20;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
      min-height: 66px;
      padding: 10px 24px;
      color: #fff;
      background: #20272b;
      border-bottom: 3px solid var(--teal);
    }
    .brand { min-width: 0; }
    .brand h1 {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      font-size: 22px;
      font-weight: 600;
      letter-spacing: 0;
    }
    .brand p { margin: 2px 0 0; color: #bfc8cc; font-size: 12px; }
    .counts { display: flex; gap: 18px; flex: 0 0 auto; }
    .count { text-align: right; }
    .count strong { display: block; font-size: 16px; color: #fff; }
    .count span { color: #aeb9be; font-size: 11px; text-transform: uppercase; }
    .shell { max-width: 1660px; margin: 0 auto; padding: 18px 24px 34px; }
    .modebar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 14px;
    }
    .segmented {
      display: inline-grid;
      grid-auto-flow: column;
      gap: 2px;
      padding: 3px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #e7ebed;
    }
    .segmented button {
      min-width: 108px;
      padding: 7px 12px;
      border: 0;
      border-radius: 4px;
      color: #40484e;
      background: transparent;
      cursor: pointer;
    }
    .segmented button[aria-selected="true"] {
      color: #fff;
      background: var(--teal);
    }
    .source-note { color: var(--muted); font-size: 12px; text-align: right; }
    .workspace {
      display: grid;
      grid-template-columns: minmax(230px, 280px) minmax(0, 1fr);
      min-height: 720px;
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
    }
    .sidebar {
      padding: 18px;
      background: #eef1f2;
      border-right: 1px solid var(--line);
    }
    .field { margin-bottom: 16px; }
    .field label {
      display: block;
      margin-bottom: 5px;
      color: #596168;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
    }
    select, input[type="search"] {
      width: 100%;
      min-height: 36px;
      padding: 7px 9px;
      border: 1px solid #bfc7cc;
      border-radius: 4px;
      color: var(--ink);
      background: #fff;
      outline: none;
    }
    select:focus, input[type="search"]:focus { border-color: var(--teal); box-shadow: 0 0 0 2px #cce9e6; }
    .task-field[hidden] { display: none; }
    .sidebar-rule { height: 1px; margin: 18px 0; background: var(--line); }
    .metric-key { display: grid; gap: 9px; }
    .metric-key div { display: grid; grid-template-columns: 9px 1fr; gap: 8px; align-items: start; color: #596168; font-size: 12px; }
    .dot { width: 8px; height: 8px; margin-top: 4px; border-radius: 50%; background: var(--teal); }
    .dot.orange { background: var(--orange); }
    .dot.blue { background: var(--blue); }
    .content { min-width: 0; padding: 20px 22px 26px; }
    .figure-head {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 12px;
    }
    .figure-head h2 { margin: 0; font-size: 19px; font-weight: 650; }
    .figure-head p { margin: 3px 0 0; color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
    .actions { display: flex; gap: 7px; flex: 0 0 auto; }
    .action {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 34px;
      padding: 6px 10px;
      border: 1px solid #bcc5ca;
      border-radius: 4px;
      color: #263238;
      background: #fff;
      text-decoration: none;
      cursor: pointer;
    }
    .action:hover { border-color: var(--teal); color: var(--teal); }
    .figure-stage {
      display: grid;
      place-items: center;
      min-height: 420px;
      padding: 12px;
      background: #fafbfb;
      border: 1px solid var(--line);
      border-radius: 4px;
      overflow: hidden;
    }
    .figure-stage img {
      display: block;
      max-width: 100%;
      max-height: 780px;
      width: auto;
      height: auto;
      object-fit: contain;
      cursor: zoom-in;
    }
    .data-head {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 14px;
      margin-top: 22px;
      padding-top: 18px;
      border-top: 2px solid #20272b;
    }
    .data-head h3 { margin: 0; font-size: 16px; }
    .data-head p { margin: 2px 0 0; color: var(--muted); font-size: 12px; }
    .filters { display: grid; grid-template-columns: minmax(180px, 1fr) 140px 170px; gap: 8px; width: min(660px, 100%); }
    .table-wrap { margin-top: 10px; max-height: 520px; overflow: auto; border: 1px solid var(--line); }
    table { width: 100%; border-collapse: collapse; white-space: nowrap; font-variant-numeric: tabular-nums; }
    th, td { padding: 7px 9px; border-bottom: 1px solid #e2e6e8; text-align: right; }
    th {
      position: sticky;
      top: 0;
      z-index: 2;
      color: #fff;
      background: #30383d;
      font-size: 11px;
      font-weight: 650;
    }
    th:first-child, td:first-child { text-align: left; }
    tbody tr:nth-child(even) { background: #f7f8f8; }
    tbody tr:hover { background: var(--teal-soft); }
    td.na { color: #8a9298; font-style: italic; }
    dialog {
      width: min(96vw, 1800px);
      max-width: none;
      height: min(94vh, 1200px);
      padding: 0;
      border: 0;
      border-radius: 6px;
      background: #fff;
      box-shadow: 0 24px 70px rgba(0,0,0,.35);
    }
    dialog::backdrop { background: rgba(17, 22, 25, .82); }
    .dialog-bar { display: flex; align-items: center; justify-content: space-between; padding: 10px 14px; color: #fff; background: #20272b; }
    .dialog-bar strong { overflow-wrap: anywhere; }
    .icon-button { width: 34px; height: 34px; border: 0; border-radius: 4px; color: #fff; background: #465158; font-size: 22px; cursor: pointer; }
    .dialog-image { display: grid; place-items: center; height: calc(100% - 54px); overflow: auto; padding: 12px; }
    .dialog-image img { max-width: none; width: auto; min-width: min(100%, 1000px); height: auto; }
    @media (max-width: 900px) {
      .topbar { align-items: flex-start; padding: 10px 14px; }
      .counts { gap: 10px; }
      .shell { padding: 12px; }
      .modebar { align-items: stretch; flex-direction: column; }
      .source-note { text-align: left; }
      .workspace { grid-template-columns: 1fr; }
      .sidebar { border-right: 0; border-bottom: 1px solid var(--line); }
      .content { padding: 16px 12px 22px; }
      .figure-head, .data-head { align-items: stretch; flex-direction: column; }
      .filters { grid-template-columns: 1fr; }
      .figure-stage { min-height: 250px; padding: 4px; }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="brand">
      <h1>Figure 3 · Calibration data</h1>
      <p>PNG figures and corresponding numerical summaries</p>
    </div>
    <div class="counts">
      <div class="count"><strong id="imageCount">20</strong><span>PNG</span></div>
      <div class="count"><strong id="csvCount">22</strong><span>CSV</span></div>
      <div class="count"><strong>8</strong><span>Tasks</span></div>
    </div>
  </header>
  <main class="shell">
    <div class="modebar">
      <div class="segmented" role="tablist" aria-label="数据范围">
        <button id="overviewMode" type="button" aria-selected="true">Overview</button>
        <button id="taskMode" type="button" aria-selected="false">By task</button>
      </div>
      <div class="source-note" id="generatedAt"></div>
    </div>
    <section class="workspace">
      <aside class="sidebar">
        <div class="field task-field" id="taskField" hidden>
          <label for="taskSelect">Task</label>
          <select id="taskSelect"></select>
        </div>
        <div class="field">
          <label for="figureSelect">Figure</label>
          <select id="figureSelect"></select>
        </div>
        <div class="field" id="tableField">
          <label for="tableSelect">Data table</label>
          <select id="tableSelect"></select>
        </div>
        <div class="sidebar-rule"></div>
        <div class="metric-key">
          <div><span class="dot"></span><span>Normalized MAE is computed within each task–protocol–method case.</span></div>
          <div><span class="dot orange"></span><span>Heatmap summaries use equal task weights and retain valid-count fields.</span></div>
          <div><span class="dot blue"></span><span>Missing GPT confidence tiers remain missing and are never filled with zero.</span></div>
        </div>
      </aside>
      <div class="content">
        <div class="figure-head">
          <div>
            <h2 id="figureTitle"></h2>
            <p id="figurePath"></p>
          </div>
          <div class="actions">
            <a class="action" id="imageLink" target="_blank" rel="noopener">↗ PNG</a>
            <a class="action" id="csvLink" download>↓ CSV</a>
          </div>
        </div>
        <div class="figure-stage">
          <img id="figureImage" alt="" />
        </div>
        <div class="data-head">
          <div>
            <h3 id="tableTitle"></h3>
            <p><span id="visibleRows">0</span> rows · <span id="csvPath"></span></p>
          </div>
          <div class="filters">
            <input id="searchInput" type="search" placeholder="Search table" />
            <select id="protocolFilter" aria-label="Filter protocol"></select>
            <select id="methodFilter" aria-label="Filter method"></select>
          </div>
        </div>
        <div class="table-wrap">
          <table>
            <thead id="tableHead"></thead>
            <tbody id="tableBody"></tbody>
          </table>
        </div>
      </div>
    </section>
  </main>
  <dialog id="imageDialog">
    <div class="dialog-bar"><strong id="dialogTitle"></strong><button class="icon-button" id="closeDialog" type="button" aria-label="Close">×</button></div>
    <div class="dialog-image"><img id="dialogImage" alt="" /></div>
  </dialog>
  <script>
    const DATA = __DATA__;
    const COLUMN_LABELS = {
      task_id: 'Task', protocol: 'Protocol', method: 'Method', selected_model: 'Selected model',
      confidence_tier: 'Tier', normalized_mae_mean: 'Mean NMAE', bootstrap_ci95_low: 'CI 2.5%',
      bootstrap_ci95_high: 'CI 97.5%', observation_count: 'Valid cases', discrimination_mean: 'Low − High',
      discrimination_valid_count: 'Valid', discrimination_total_count: 'Total', tier_spearman_mean: 'Spearman ρ',
      tier_spearman_valid_count: 'Valid', tier_spearman_total_count: 'Total', tier_spearman_valid_percent: 'Valid %',
      medium_minus_high_mean: 'Medium − High', medium_minus_high_valid_count: 'Valid M−H',
      medium_minus_high_total_count: 'Total M−H', low_minus_medium_mean: 'Low − Medium',
      low_minus_medium_valid_count: 'Valid L−M', low_minus_medium_total_count: 'Total L−M',
      High: 'High NMAE', Medium: 'Medium NMAE', Low: 'Low NMAE', sample_n: 'Samples',
      confidence_source: 'Confidence source', complete_three_tiers: 'All tiers', medium_minus_high: 'Medium − High',
      low_minus_medium: 'Low − Medium', has_high_medium_pair: 'High/Medium', has_medium_low_pair: 'Medium/Low'
    };
    const state = { mode: 'overview', task: DATA.tasks[0].id, figure: 0, table: 0 };
    const byId = (id) => document.getElementById(id);
    const format = (value) => {
      if (value === null || value === undefined || value === '') return 'NA';
      if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toLocaleString('en-US', { maximumFractionDigits: 5 });
      if (typeof value === 'boolean') return value ? 'Yes' : 'No';
      return String(value);
    };
    const currentTask = () => DATA.tasks.find((task) => task.id === state.task) || DATA.tasks[0];
    const figureList = () => state.mode === 'overview' ? DATA.overview : currentTask().figures;
    const currentFigure = () => figureList()[state.figure] || figureList()[0];
    const currentTable = () => currentFigure().tables[state.table] || currentFigure().tables[0];

    function fillSelect(select, items, selectedIndex = 0) {
      select.innerHTML = items.map((item, index) => `<option value="${index}" ${index === selectedIndex ? 'selected' : ''}>${item.label}</option>`).join('');
    }
    function uniqueValues(rows, column) {
      return [...new Set(rows.map((row) => row[column]).filter((value) => value !== null && value !== undefined && value !== ''))];
    }
    function fillFilter(select, label, values) {
      select.innerHTML = `<option value="">${label}</option>` + values.map((value) => `<option value="${String(value).replaceAll('&','&amp;').replaceAll('"','&quot;')}">${value}</option>`).join('');
    }
    function renderSelectors() {
      byId('taskField').hidden = state.mode !== 'task';
      byId('taskSelect').innerHTML = DATA.tasks.map((task) => `<option value="${task.id}" ${task.id === state.task ? 'selected' : ''}>${task.label}</option>`).join('');
      fillSelect(byId('figureSelect'), figureList(), state.figure);
      fillSelect(byId('tableSelect'), currentFigure().tables, state.table);
    }
    function filteredRows(table) {
      const query = byId('searchInput').value.trim().toLowerCase();
      const protocol = byId('protocolFilter').value;
      const method = byId('methodFilter').value;
      return table.rows.filter((row) => {
        if (protocol && String(row.protocol) !== protocol) return false;
        if (method && String(row.method) !== method) return false;
        if (!query) return true;
        return table.columns.some((column) => format(row[column]).toLowerCase().includes(query));
      });
    }
    function renderTableBody() {
      const table = currentTable();
      const rows = filteredRows(table);
      byId('visibleRows').textContent = rows.length;
      byId('tableBody').innerHTML = rows.map((row) => `<tr>${table.columns.map((column) => {
        const value = row[column];
        const isNA = value === null || value === undefined || value === '';
        return `<td class="${isNA ? 'na' : ''}">${format(value)}</td>`;
      }).join('')}</tr>`).join('');
    }
    function render() {
      renderSelectors();
      const figure = currentFigure();
      const table = currentTable();
      byId('figureTitle').textContent = state.mode === 'task' ? `${currentTask().label} · ${figure.label}` : figure.label;
      byId('figurePath').textContent = figure.image;
      byId('figureImage').src = figure.image;
      byId('figureImage').alt = byId('figureTitle').textContent;
      byId('imageLink').href = figure.image;
      byId('csvLink').href = table.file;
      byId('tableTitle').textContent = table.label;
      byId('csvPath').textContent = table.file;
      fillFilter(byId('protocolFilter'), 'All protocols', uniqueValues(table.rows, 'protocol'));
      fillFilter(byId('methodFilter'), 'All methods', uniqueValues(table.rows, 'method'));
      byId('tableHead').innerHTML = `<tr>${table.columns.map((column) => `<th>${COLUMN_LABELS[column] || column}</th>`).join('')}</tr>`;
      byId('searchInput').value = '';
      renderTableBody();
    }
    function setMode(mode) {
      state.mode = mode;
      state.figure = 0;
      state.table = 0;
      byId('overviewMode').setAttribute('aria-selected', String(mode === 'overview'));
      byId('taskMode').setAttribute('aria-selected', String(mode === 'task'));
      render();
    }
    byId('overviewMode').addEventListener('click', () => setMode('overview'));
    byId('taskMode').addEventListener('click', () => setMode('task'));
    byId('taskSelect').addEventListener('change', (event) => { state.task = event.target.value; state.figure = 0; state.table = 0; render(); });
    byId('figureSelect').addEventListener('change', (event) => { state.figure = Number(event.target.value); state.table = 0; render(); });
    byId('tableSelect').addEventListener('change', (event) => { state.table = Number(event.target.value); render(); });
    byId('searchInput').addEventListener('input', renderTableBody);
    byId('protocolFilter').addEventListener('change', renderTableBody);
    byId('methodFilter').addEventListener('change', renderTableBody);
    byId('figureImage').addEventListener('click', () => {
      byId('dialogTitle').textContent = byId('figureTitle').textContent;
      byId('dialogImage').src = currentFigure().image;
      byId('dialogImage').alt = byId('figureTitle').textContent;
      byId('imageDialog').showModal();
    });
    byId('closeDialog').addEventListener('click', () => byId('imageDialog').close());
    byId('imageDialog').addEventListener('click', (event) => { if (event.target === byId('imageDialog')) byId('imageDialog').close(); });
    byId('generatedAt').textContent = `Generated ${DATA.generated_at} · standalone offline viewer`;
    byId('imageCount').textContent = DATA.image_count;
    byId('csvCount').textContent = DATA.csv_count;
    render();
  </script>
</body>
</html>
'''
    return template.replace("__DATA__", data_json)


def build_delivery(source_root: Path, target: Path) -> tuple[Path | None, dict[str, int]]:
    source_root = source_root.resolve()
    target = target.resolve()
    if target.name != "figure3-data":
        raise ValueError(f"Refusing unexpected target name: {target}")
    source_data = source_root / "data"
    required_data = [
        source_data / "method_tier_summary.csv",
        source_data / "method_protocol_summary.csv",
        source_data / "task_protocol_method_summary.csv",
    ]
    overview_sources = [source_root / "overview" / name for name in OVERVIEW_IMAGES]
    task_root = source_root / "by_task"
    task_dirs = sorted(path for path in task_root.iterdir() if path.is_dir())
    if [path.name for path in task_dirs] != sorted(TASK_LABELS):
        raise ValueError("The source task directory set is not the expected eight-task scope.")
    task_sources = [task_dir / name for task_dir in task_dirs for name in TASK_IMAGES]
    _require_files(required_data + overview_sources + task_sources)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ready = target.parent / f".{target.name}.ready_{stamp}"
    if ready.exists():
        raise FileExistsError(ready)
    ready.mkdir(parents=True)
    archive_path: Path | None = None
    try:
        overview_dir = ready / "overview"
        by_task_dir = ready / "by_task"
        overview_dir.mkdir()
        by_task_dir.mkdir()
        for source in overview_sources:
            shutil.copy2(source, overview_dir / source.name)
        for task_dir in task_dirs:
            destination = by_task_dir / task_dir.name
            destination.mkdir()
            for name in TASK_IMAGES:
                shutil.copy2(task_dir / name, destination / name)

        tier = pd.read_csv(required_data[0])
        protocol = pd.read_csv(required_data[1])
        task = pd.read_csv(required_data[2])
        overview_payload: list[dict[str, object]] = []

        main_tables = [
            ("Panel A · Tier curves", tier[tier["protocol"].eq("All")], "calibration_main_panel_a_tiers.csv", [
                "method", "confidence_tier", "normalized_mae_mean", "bootstrap_ci95_low", "bootstrap_ci95_high", "observation_count"
            ]),
            ("Panel B · Low − High discrimination", protocol, "calibration_main_panel_b_discrimination.csv", [
                "method", "protocol", "discrimination_mean", "discrimination_valid_count", "discrimination_total_count"
            ]),
            ("Panel C · Tier-risk Spearman", protocol, "calibration_main_panel_c_spearman.csv", [
                "method", "protocol", "tier_spearman_mean", "tier_spearman_valid_count", "tier_spearman_total_count", "tier_spearman_valid_percent"
            ]),
        ]
        main_output_tables = []
        for label, frame, filename, columns in main_tables:
            table_info = _write_csv(frame, overview_dir / filename, columns)
            table_info["label"] = label
            main_output_tables.append(_relative_table(table_info, ready))
        overview_payload.append({
            "label": "Calibration main · A–C",
            "image": "overview/calibration_main.png",
            "tables": main_output_tables,
        })

        by_protocol_info = _write_csv(
            tier[~tier["protocol"].eq("All")],
            overview_dir / "calibration_by_protocol.csv",
            ["protocol", "method", "confidence_tier", "normalized_mae_mean", "bootstrap_ci95_low", "bootstrap_ci95_high", "observation_count"],
        )
        by_protocol_info["label"] = "Tier curves by protocol"
        overview_payload.append({
            "label": "Calibration by protocol",
            "image": "overview/calibration_by_protocol.png",
            "tables": [_relative_table(by_protocol_info, ready)],
        })

        by_method_info = _write_csv(
            protocol,
            overview_dir / "calibration_by_method.csv",
            list(protocol.columns),
        )
        by_method_info["label"] = "Discrimination and Spearman by method"
        overview_payload.append({
            "label": "Calibration by method",
            "image": "overview/calibration_by_method.png",
            "tables": [_relative_table(by_method_info, ready)],
        })

        adjacent_columns = [
            "method", "protocol", "medium_minus_high_mean", "medium_minus_high_valid_count",
            "medium_minus_high_total_count", "low_minus_medium_mean", "low_minus_medium_valid_count",
            "low_minus_medium_total_count",
        ]
        adjacent_info = _write_csv(
            protocol,
            overview_dir / "calibration_adjacent_tier_diagnostics.csv",
            adjacent_columns,
        )
        adjacent_info["label"] = "Adjacent-tier differences"
        overview_payload.append({
            "label": "Adjacent-tier diagnostics",
            "image": "overview/calibration_adjacent_tier_diagnostics.png",
            "tables": [_relative_table(adjacent_info, ready)],
        })

        task_payload = []
        for task_dir in task_dirs:
            task_id = task_dir.name
            frame = task[task["task_id"].astype(str).eq(task_id)]
            destination = by_task_dir / task_id
            curve_info = _write_csv(
                frame,
                destination / "calibration_by_protocol.csv",
                ["task_id", "protocol", "method", "selected_model", "High", "Medium", "Low", "sample_n", "confidence_source", "complete_three_tiers"],
            )
            curve_info["label"] = "High / Medium / Low NMAE"
            adjacent_task_info = _write_csv(
                frame,
                destination / "calibration_adjacent_tier_diagnostics.csv",
                ["task_id", "protocol", "method", "selected_model", "medium_minus_high", "low_minus_medium", "has_high_medium_pair", "has_medium_low_pair"],
            )
            adjacent_task_info["label"] = "Adjacent-tier differences"
            task_payload.append({
                "id": task_id,
                "label": TASK_LABELS[task_id],
                "figures": [
                    {
                        "label": "Calibration by protocol",
                        "image": f"by_task/{task_id}/calibration_by_protocol.png",
                        "tables": [_relative_table(curve_info, ready)],
                    },
                    {
                        "label": "Adjacent-tier diagnostics",
                        "image": f"by_task/{task_id}/calibration_adjacent_tier_diagnostics.png",
                        "tables": [_relative_table(adjacent_task_info, ready)],
                    },
                ],
            })

        png_files = sorted(ready.rglob("*.png"))
        csv_files = sorted(ready.rglob("*.csv"))
        if len(png_files) != 20 or len(csv_files) != 22:
            raise ValueError(f"Delivery count mismatch: PNG={len(png_files)}, CSV={len(csv_files)}")
        for path in png_files:
            _validate_png(path)
        html_payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "image_count": len(png_files),
            "csv_count": len(csv_files),
            "overview": overview_payload,
            "tasks": task_payload,
        }
        (ready / "index.html").write_text(_html_template(html_payload), encoding="utf-8", newline="\n")
        unexpected = [path for path in ready.rglob("*") if path.is_file() and path.suffix.lower() not in {".png", ".csv", ".html"}]
        if unexpected:
            raise ValueError(f"Unexpected delivery file types: {unexpected}")

        if target.exists():
            archive_root = target.parent / f"{target.name}_archive"
            archive_root.mkdir(exist_ok=True)
            archive_path = archive_root / stamp
            counter = 1
            while archive_path.exists():
                archive_path = archive_root / f"{stamp}_{counter}"
                counter += 1
            os.replace(target, archive_path)
        os.replace(ready, target)
        return archive_path, {"png": len(png_files), "csv": len(csv_files), "html": 1}
    except Exception:
        shutil.rmtree(ready, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the PNG/CSV Figure 3 calibration delivery.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    args = parser.parse_args()
    archive, counts = build_delivery(args.source, args.target)
    print(json.dumps({"target": str(args.target.resolve()), "archive": str(archive) if archive else None, **counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
