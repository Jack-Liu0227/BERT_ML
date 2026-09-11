from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path


DEFAULT_ROOT = Path(r"D:\XJTU\Research\PHD\ood\Figures\hybrid\triptych")

SCOPE_LABELS = {
    "combined": "Combined",
    "test_extrapolation_high20": "Extrapolation high20",
    "test_inner_ood": "Inner OOD",
}

VIEW_LABELS = {
    "main": "Main triptych",
    "subset_comparison": "Subset comparison",
}

TASK_ORDER = {
    task: index
    for index, task in enumerate(
        (
            "Al-UTS",
            "HEA-El",
            "HEA-UTS",
            "Matbench_Steel-YS",
            "Steel-El",
            "Steel-UTS",
            "Ti-El",
            "Ti-UTS",
        )
    )
}


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" href="data:,">
  <title>Hybrid Triptych data atlas</title>
  <style>
    :root {
      --ink:#152127;
      --muted:#66747a;
      --line:#d2dadd;
      --paper:#fffdf8;
      --canvas:#ecefed;
      --nav:#172229;
      --accent:#c55a20;
      --accent-deep:#913c14;
      --accent-soft:#fae8dc;
      --blue:#315e78;
      --shadow:0 18px 55px rgba(31,42,47,.10);
    }
    * { box-sizing:border-box; }
    html { scroll-behavior:smooth; }
    body {
      margin:0;
      min-width:320px;
      color:var(--ink);
      background:
        linear-gradient(rgba(255,255,255,.42),rgba(255,255,255,.42)),
        repeating-linear-gradient(0deg,transparent 0,transparent 27px,rgba(59,78,86,.045) 28px),
        var(--canvas);
      font-family:Bahnschrift,Aptos,"Microsoft YaHei",sans-serif;
      font-size:14px;
      line-height:1.45;
    }
    button,select,input,a { font:inherit; }
    button,a { -webkit-tap-highlight-color:transparent; }
    .masthead {
      position:relative;
      overflow:hidden;
      color:#fff;
      background:var(--nav);
      border-bottom:4px solid var(--accent);
    }
    .masthead::after {
      content:"";
      position:absolute;
      right:-120px;
      top:-175px;
      width:520px;
      height:520px;
      border:1px solid rgba(255,255,255,.11);
      border-radius:50%;
      box-shadow:0 0 0 62px rgba(255,255,255,.025),0 0 0 126px rgba(255,255,255,.018);
      pointer-events:none;
    }
    .masthead-inner {
      position:relative;
      z-index:1;
      display:grid;
      grid-template-columns:minmax(0,1fr) auto;
      align-items:end;
      gap:34px;
      max-width:1780px;
      margin:0 auto;
      padding:25px 28px 22px;
    }
    .eyebrow { margin:0 0 5px; color:#ed9d70; font-size:10px; font-weight:700; letter-spacing:.18em; text-transform:uppercase; }
    h1 { margin:0; font-family:Cambria,Georgia,serif; font-size:clamp(27px,3vw,43px); font-weight:600; letter-spacing:-.025em; }
    .subtitle { max-width:760px; margin:6px 0 0; color:#b8c3c8; font-size:12px; }
    .stats { display:flex; gap:30px; padding:5px 0; }
    .stat { min-width:68px; text-align:right; }
    .stat strong { display:block; color:#fff; font-family:Cambria,Georgia,serif; font-size:25px; font-weight:600; line-height:1; }
    .stat span { display:block; margin-top:4px; color:#98a7ad; font-size:9px; letter-spacing:.1em; text-transform:uppercase; }
    .shell { max-width:1780px; margin:0 auto; padding:20px 28px 42px; }
    .workspace {
      display:grid;
      grid-template-columns:286px minmax(0,1fr);
      min-height:820px;
      overflow:hidden;
      background:var(--paper);
      border:1px solid #cbd3d6;
      box-shadow:var(--shadow);
    }
    .sidebar { min-width:0; padding:20px 18px 24px; background:#e7ebea; border-right:1px solid var(--line); }
    .sidebar-kicker { margin:0 0 14px; color:var(--accent-deep); font-size:10px; font-weight:700; letter-spacing:.13em; text-transform:uppercase; }
    .field { margin-bottom:15px; }
    label { display:block; margin-bottom:5px; color:#536168; font-size:10px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
    select,input[type="search"] {
      width:100%;
      min-height:39px;
      padding:8px 10px;
      color:var(--ink);
      background:#fff;
      border:1px solid #b9c4c8;
      border-radius:3px;
      outline:none;
    }
    select:focus,input[type="search"]:focus { border-color:var(--accent); box-shadow:0 0 0 3px rgba(197,90,32,.13); }
    .rule { height:1px; margin:18px 0; background:#cbd3d6; }
    .dataset-label { display:flex; align-items:center; justify-content:space-between; gap:10px; margin-bottom:8px; }
    .dataset-label h2 { margin:0; font-family:Cambria,Georgia,serif; font-size:16px; font-weight:600; }
    .dataset-label span { color:var(--muted); font-size:10px; }
    .dataset-list { display:grid; gap:5px; }
    .dataset-button {
      position:relative;
      width:100%;
      padding:9px 10px 9px 15px;
      color:#47555b;
      background:rgba(255,255,255,.45);
      border:1px solid transparent;
      text-align:left;
      cursor:pointer;
    }
    .dataset-button::before { content:""; position:absolute; left:6px; top:50%; width:3px; height:17px; background:#a4afb3; transform:translateY(-50%); }
    .dataset-button:hover { color:var(--accent-deep); background:#fff; border-color:#d5dcde; }
    .dataset-button[aria-current="true"] { color:#fff; background:var(--accent); border-color:var(--accent); }
    .dataset-button[aria-current="true"]::before { background:#fff; }
    .sidebar-note { margin:18px 0 0; padding:12px; color:#5d696f; background:#f6f7f5; border-left:3px solid var(--blue); font-size:11px; }
    .sidebar-note strong { display:block; margin-bottom:3px; color:var(--ink); }
    .content { min-width:0; padding:25px 27px 32px; }
    .record-head { display:flex; align-items:flex-start; justify-content:space-between; gap:20px; margin-bottom:14px; }
    .breadcrumb { margin:0 0 4px; color:var(--accent-deep); font-size:10px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
    .record-head h2 { margin:0; font-family:Cambria,Georgia,serif; font-size:24px; font-weight:600; }
    .paths { margin:5px 0 0; color:var(--muted); font-family:"Cascadia Mono",Consolas,monospace; font-size:10px; overflow-wrap:anywhere; }
    .chips { display:flex; flex-wrap:wrap; gap:5px; margin-top:9px; }
    .chip { padding:3px 8px; color:#526067; background:#f1f3f2; border:1px solid #d6dddf; border-radius:999px; font-size:10px; }
    .actions { display:flex; flex-wrap:wrap; justify-content:flex-end; gap:7px; flex:0 0 auto; }
    .action {
      display:inline-flex;
      align-items:center;
      justify-content:center;
      min-height:35px;
      padding:7px 11px;
      color:#334047;
      background:#fff;
      border:1px solid #b9c3c7;
      border-radius:3px;
      text-decoration:none;
      cursor:pointer;
    }
    .action:hover { color:var(--accent-deep); border-color:var(--accent); }
    .action.primary { color:#fff; background:var(--accent); border-color:var(--accent); }
    .action.primary:hover { background:var(--accent-deep); }
    .figure-card { position:relative; padding:12px; background:#f5f4ef; border:1px solid var(--line); }
    .figure-card::before { content:"TRIPTYCH / PNG"; position:absolute; z-index:1; left:20px; top:18px; padding:4px 7px; color:#fff; background:rgba(23,34,41,.78); font-size:9px; letter-spacing:.12em; }
    .figure-card img { display:block; width:auto; max-width:100%; max-height:860px; margin:auto; object-fit:contain; cursor:zoom-in; }
    .data-section { margin-top:25px; padding-top:20px; border-top:3px solid var(--nav); }
    .data-head { display:flex; align-items:end; justify-content:space-between; gap:20px; }
    .data-head h3 { margin:0; font-family:Cambria,Georgia,serif; font-size:20px; font-weight:600; }
    .data-head p { margin:3px 0 0; color:var(--muted); font-size:11px; }
    .table-tools { display:grid; grid-template-columns:minmax(210px,340px) 105px; gap:8px; width:min(520px,100%); }
    .table-status { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-top:10px; color:var(--muted); font-size:11px; }
    .pager { display:flex; align-items:center; gap:6px; }
    .pager button { min-width:31px; height:29px; color:#36434a; background:#fff; border:1px solid #c2cbce; cursor:pointer; }
    .pager button:hover:not(:disabled) { color:var(--accent-deep); border-color:var(--accent); }
    .pager button:disabled { opacity:.38; cursor:not-allowed; }
    .table-wrap { max-height:610px; margin-top:7px; overflow:auto; background:#fff; border:1px solid var(--line); }
    table { width:100%; border-collapse:separate; border-spacing:0; white-space:nowrap; font-size:11px; font-variant-numeric:tabular-nums; }
    th,td { padding:7px 9px; border-right:1px solid #edf0f0; border-bottom:1px solid #e1e6e7; text-align:left; }
    th { position:sticky; z-index:3; top:0; color:#fff; background:#303c42; font-size:10px; font-weight:650; }
    th:first-child { left:0; z-index:4; }
    td:first-child { position:sticky; left:0; z-index:1; background:inherit; }
    tbody tr:nth-child(even) { background:#f6f8f7; }
    tbody tr:hover { background:var(--accent-soft); }
    td.numeric { text-align:right; }
    .empty { padding:34px; color:var(--muted); text-align:center; }
    dialog { width:min(97vw,1900px); max-width:none; height:min(95vh,1260px); padding:0; overflow:hidden; background:#12191d; border:0; box-shadow:0 26px 90px rgba(0,0,0,.55); }
    dialog::backdrop { background:rgba(8,12,14,.88); }
    .lightbox { position:relative; display:grid; width:100%; height:100%; place-items:center; padding:40px; }
    .lightbox img { max-width:100%; max-height:100%; object-fit:contain; }
    .lightbox button { position:absolute; right:13px; top:12px; width:38px; height:38px; color:#fff; background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.35); border-radius:50%; cursor:pointer; }
    .footer { max-width:1780px; margin:0 auto; padding:0 28px 30px; color:#6b777c; font-size:10px; }
    @media (max-width:900px) {
      .masthead-inner { grid-template-columns:1fr; }
      .stats { justify-content:flex-start; }
      .stat { text-align:left; }
      .shell { padding:12px; }
      .workspace { grid-template-columns:1fr; }
      .sidebar { border-right:0; border-bottom:1px solid var(--line); }
      .dataset-list { grid-template-columns:repeat(2,minmax(0,1fr)); }
      .content { padding:19px 15px 25px; }
      .record-head,.data-head { align-items:stretch; flex-direction:column; }
      .actions { justify-content:flex-start; }
      .table-tools { width:100%; }
    }
    @media (max-width:560px) {
      .masthead-inner { padding:20px 16px; }
      .stats { gap:16px; }
      .dataset-list { grid-template-columns:1fr; }
      .table-tools { grid-template-columns:1fr; }
      .table-status { align-items:flex-start; flex-direction:column; }
    }
    @media (prefers-reduced-motion:reduce) { html { scroll-behavior:auto; } }
  </style>
</head>
<body>
  <header class="masthead">
    <div class="masthead-inner">
      <div>
        <p class="eyebrow">Hybrid OOD / Result archive</p>
        <h1>Triptych data atlas</h1>
        <p class="subtitle">浏览 Triptych 文件夹内全部结果。每张 PNG 与其同范围、同视图、同任务的 CSV 一一对应，图像在上，原始数据在下。</p>
      </div>
      <div class="stats" aria-label="数据统计">
        <div class="stat"><strong id="pairCount">32</strong><span>PNG–CSV pairs</span></div>
        <div class="stat"><strong id="scopeCount">3</strong><span>Scopes</span></div>
        <div class="stat"><strong id="taskCount">8</strong><span>Tasks</span></div>
      </div>
    </div>
  </header>

  <main class="shell">
    <section class="workspace">
      <aside class="sidebar">
        <p class="sidebar-kicker">Data navigation</p>
        <div class="field">
          <label for="scopeSelect">数据范围 / Scope</label>
          <select id="scopeSelect"></select>
        </div>
        <div class="field">
          <label for="viewSelect">结果视图 / View</label>
          <select id="viewSelect"></select>
        </div>
        <div class="rule"></div>
        <div class="dataset-label">
          <h2>任务数据</h2>
          <span id="visibleCount"></span>
        </div>
        <div id="datasetList" class="dataset-list"></div>
        <p class="sidebar-note"><strong>文件匹配规则</strong>当前 PNG 和 CSV 由同一个 scope / view / task 组合生成；表格数据已嵌入本页面，因此可直接通过 file:// 查看。</p>
      </aside>

      <article class="content">
        <header class="record-head">
          <div>
            <p id="breadcrumb" class="breadcrumb"></p>
            <h2 id="recordTitle"></h2>
            <p id="recordPaths" class="paths"></p>
            <div id="chips" class="chips"></div>
          </div>
          <div class="actions">
            <button id="prevButton" class="action" type="button">← 上一组</button>
            <button id="nextButton" class="action" type="button">下一组 →</button>
            <a id="pngLink" class="action" download>下载 PNG</a>
            <a id="csvLink" class="action primary" download>下载 CSV</a>
          </div>
        </header>

        <div class="figure-card">
          <img id="figureImage" alt="Triptych figure">
        </div>

        <section class="data-section">
          <div class="data-head">
            <div>
              <h3>Matched CSV data</h3>
              <p id="csvMeta"></p>
            </div>
            <div class="table-tools">
              <input id="tableSearch" type="search" placeholder="搜索当前 CSV 的任意字段…" aria-label="搜索 CSV">
              <select id="pageSize" aria-label="每页行数">
                <option value="25">25 行</option>
                <option value="50">50 行</option>
                <option value="100">100 行</option>
                <option value="all">全部</option>
              </select>
            </div>
          </div>
          <div class="table-status">
            <span id="rowStatus"></span>
            <div class="pager">
              <button id="pagePrev" type="button" aria-label="上一页">‹</button>
              <span id="pageLabel"></span>
              <button id="pageNext" type="button" aria-label="下一页">›</button>
            </div>
          </div>
          <div class="table-wrap">
            <table>
              <thead id="tableHead"></thead>
              <tbody id="tableBody"></tbody>
            </table>
          </div>
        </section>
      </article>
    </section>
  </main>

  <p class="footer">Generated __GENERATED_AT__ · Source: D:\XJTU\Research\PHD\ood\Figures\hybrid\triptych</p>

  <dialog id="imageDialog">
    <div class="lightbox">
      <button id="closeDialog" type="button" aria-label="关闭">×</button>
      <img id="dialogImage" alt="Triptych enlarged view">
    </div>
  </dialog>

  <script id="atlas-data" type="application/json">__DATA_JSON__</script>
  <script>
    (() => {
      const datasets = JSON.parse(document.getElementById('atlas-data').textContent);
      const byId = new Map(datasets.map(item => [item.id, item]));
      const scopeOrder = [...new Set(datasets.map(item => item.scope))];
      const state = { currentId: datasets[0]?.id || '', query: '', page: 1, pageSize: 25 };

      const el = Object.fromEntries([
        'pairCount','scopeCount','taskCount','scopeSelect','viewSelect','visibleCount','datasetList',
        'breadcrumb','recordTitle','recordPaths','chips','prevButton','nextButton','pngLink','csvLink',
        'figureImage','csvMeta','tableSearch','pageSize','rowStatus','pageLabel','pagePrev','pageNext',
        'tableHead','tableBody','imageDialog','dialogImage','closeDialog'
      ].map(id => [id, document.getElementById(id)]));

      const unique = values => [...new Set(values)];
      const current = () => byId.get(state.currentId) || datasets[0];
      const isNumeric = value => value !== '' && /^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$/.test(value);
      const filteredForNav = () => datasets.filter(item => item.scope === el.scopeSelect.value && item.view === el.viewSelect.value);

      function setOptions(select, options, selected) {
        select.replaceChildren(...options.map(option => {
          const node = document.createElement('option');
          node.value = option.value;
          node.textContent = option.label;
          node.selected = option.value === selected;
          return node;
        }));
      }

      function readHash() {
        const raw = location.hash.replace(/^#/, '');
        const params = new URLSearchParams(raw);
        const id = params.get('id');
        if (id && byId.has(id)) state.currentId = id;
      }

      function writeHash() {
        const next = `id=${encodeURIComponent(state.currentId)}`;
        if (location.hash.slice(1) !== next) history.replaceState(null, '', `#${next}`);
      }

      function syncSelectors() {
        const item = current();
        setOptions(el.scopeSelect, scopeOrder.map(scope => ({
          value: scope,
          label: datasets.find(entry => entry.scope === scope).scopeLabel
        })), item.scope);
        const views = unique(datasets.filter(entry => entry.scope === item.scope).map(entry => entry.view));
        setOptions(el.viewSelect, views.map(view => ({
          value: view,
          label: datasets.find(entry => entry.scope === item.scope && entry.view === view).viewLabel
        })), item.view);
      }

      function renderDatasetList() {
        const visible = filteredForNav();
        el.visibleCount.textContent = `${visible.length} 组`;
        el.datasetList.replaceChildren(...visible.map(item => {
          const button = document.createElement('button');
          button.type = 'button';
          button.className = 'dataset-button';
          button.textContent = item.task;
          button.setAttribute('aria-current', String(item.id === state.currentId));
          button.addEventListener('click', () => selectDataset(item.id));
          return button;
        }));
      }

      function renderRecord() {
        const item = current();
        if (!item) return;
        el.breadcrumb.textContent = `${item.scopeLabel} / ${item.viewLabel}`;
        el.recordTitle.textContent = item.task;
        el.recordPaths.textContent = `PNG  ${item.image}   ·   CSV  ${item.csv}`;
        el.chips.replaceChildren(...[
          item.scopeLabel,
          item.viewLabel,
          `${item.rows.length} rows`,
          `${item.columns.length} columns`
        ].map(text => {
          const chip = document.createElement('span');
          chip.className = 'chip';
          chip.textContent = text;
          return chip;
        }));
        el.figureImage.src = encodeURI(item.image);
        el.figureImage.alt = `${item.task} — ${item.scopeLabel} — ${item.viewLabel}`;
        el.pngLink.href = encodeURI(item.image);
        el.pngLink.download = `${item.scope}_${item.view}_${item.task}_triptych.png`;
        el.csvLink.href = encodeURI(item.csv);
        el.csvLink.download = `${item.scope}_${item.view}_${item.task}_triptych.csv`;
        el.csvMeta.textContent = `${item.csv} · ${item.csvBytes.toLocaleString()} bytes`;
        renderTable();
      }

      function renderTable() {
        const item = current();
        const query = state.query.trim().toLocaleLowerCase();
        const rows = query ? item.rows.filter(row => row.some(value => value.toLocaleLowerCase().includes(query))) : item.rows;
        const totalPages = state.pageSize === 'all' ? 1 : Math.max(1, Math.ceil(rows.length / state.pageSize));
        state.page = Math.min(Math.max(1, state.page), totalPages);
        const start = state.pageSize === 'all' ? 0 : (state.page - 1) * state.pageSize;
        const end = state.pageSize === 'all' ? rows.length : Math.min(rows.length, start + state.pageSize);
        const shown = rows.slice(start, end);

        const headerRow = document.createElement('tr');
        item.columns.forEach(column => {
          const th = document.createElement('th');
          th.textContent = column;
          th.title = column;
          headerRow.appendChild(th);
        });
        el.tableHead.replaceChildren(headerRow);

        if (!shown.length) {
          const row = document.createElement('tr');
          const cell = document.createElement('td');
          cell.colSpan = item.columns.length;
          cell.className = 'empty';
          cell.textContent = '没有匹配的数据行';
          row.appendChild(cell);
          el.tableBody.replaceChildren(row);
        } else {
          el.tableBody.replaceChildren(...shown.map(values => {
            const row = document.createElement('tr');
            values.forEach(value => {
              const cell = document.createElement('td');
              cell.textContent = value;
              cell.title = value;
              if (isNumeric(value)) cell.classList.add('numeric');
              row.appendChild(cell);
            });
            return row;
          }));
        }

        el.rowStatus.textContent = rows.length
          ? `显示 ${start + 1}–${end} / ${rows.length} 行${query ? `（原始 ${item.rows.length} 行）` : ''}`
          : `0 / ${item.rows.length} 行`;
        el.pageLabel.textContent = `${state.page} / ${totalPages}`;
        el.pagePrev.disabled = state.page <= 1;
        el.pageNext.disabled = state.page >= totalPages;
      }

      function selectDataset(id) {
        if (!byId.has(id)) return;
        state.currentId = id;
        state.query = '';
        state.page = 1;
        el.tableSearch.value = '';
        syncSelectors();
        renderDatasetList();
        renderRecord();
        writeHash();
      }

      function move(direction) {
        const index = datasets.findIndex(item => item.id === state.currentId);
        const next = (index + direction + datasets.length) % datasets.length;
        selectDataset(datasets[next].id);
        window.scrollTo({ top: 0, behavior: 'smooth' });
      }

      el.scopeSelect.addEventListener('change', () => {
        const first = datasets.find(item => item.scope === el.scopeSelect.value);
        if (first) selectDataset(first.id);
      });
      el.viewSelect.addEventListener('change', () => {
        const first = datasets.find(item => item.scope === el.scopeSelect.value && item.view === el.viewSelect.value);
        if (first) selectDataset(first.id);
      });
      el.prevButton.addEventListener('click', () => move(-1));
      el.nextButton.addEventListener('click', () => move(1));
      el.tableSearch.addEventListener('input', event => {
        state.query = event.target.value;
        state.page = 1;
        renderTable();
      });
      el.pageSize.addEventListener('change', event => {
        state.pageSize = event.target.value === 'all' ? 'all' : Number(event.target.value);
        state.page = 1;
        renderTable();
      });
      el.pagePrev.addEventListener('click', () => { state.page -= 1; renderTable(); });
      el.pageNext.addEventListener('click', () => { state.page += 1; renderTable(); });
      el.figureImage.addEventListener('click', () => {
        el.dialogImage.src = el.figureImage.src;
        el.imageDialog.showModal();
      });
      el.closeDialog.addEventListener('click', () => el.imageDialog.close());
      el.imageDialog.addEventListener('click', event => { if (event.target === el.imageDialog) el.imageDialog.close(); });
      window.addEventListener('keydown', event => {
        if (el.imageDialog.open) return;
        if (event.key === 'ArrowLeft') move(-1);
        if (event.key === 'ArrowRight') move(1);
      });
      window.addEventListener('hashchange', () => { readHash(); selectDataset(state.currentId); });

      el.pairCount.textContent = datasets.length;
      el.scopeCount.textContent = unique(datasets.map(item => item.scope)).length;
      el.taskCount.textContent = unique(datasets.map(item => item.task)).length;
      readHash();
      selectDataset(state.currentId);
    })();
  </script>
</body>
</html>
'''


def read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    columns = rows[0]
    data_rows = rows[1:]
    for line_number, row in enumerate(data_rows, start=2):
        if len(row) != len(columns):
            raise ValueError(
                f"CSV width mismatch at {path}:{line_number}: "
                f"expected {len(columns)}, got {len(row)}"
            )
    return columns, data_rows


def normalized_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def build_manifest(root: Path) -> list[dict[str, object]]:
    png_paths = sorted(root.glob("*/figure/*/*/triptych.png"))
    csv_paths = sorted(root.glob("*/csv/*/*/triptych.csv"))
    if not png_paths:
        raise FileNotFoundError(f"No Triptych PNG files found below {root}")

    png_relatives = {path.relative_to(root).as_posix() for path in png_paths}
    csv_relatives = {path.relative_to(root).as_posix() for path in csv_paths}
    manifest: list[dict[str, object]] = []

    for image_path in png_paths:
        relative = image_path.relative_to(root)
        scope, figure_dir, view, task, filename = relative.parts
        if figure_dir != "figure" or filename != "triptych.png":
            raise ValueError(f"Unexpected Triptych image path: {relative}")
        if scope not in SCOPE_LABELS:
            raise ValueError(f"Unknown scope in {relative}: {scope}")
        if view not in VIEW_LABELS:
            raise ValueError(f"Unknown view in {relative}: {view}")

        csv_relative = Path(scope, "csv", view, task, "triptych.csv")
        csv_path = root / csv_relative
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing matched CSV for {relative}: {csv_relative}")

        columns, rows = read_csv(csv_path)
        if "task" in columns:
            task_index = columns.index("task")
            expected_task = normalized_label(task)
            bad_tasks = sorted(
                {
                    row[task_index]
                    for row in rows
                    if normalized_label(row[task_index]) != expected_task
                }
            )
            if bad_tasks:
                raise ValueError(f"Task mismatch in {csv_relative}: {bad_tasks}")

        manifest.append(
            {
                "id": f"{scope}__{view}__{task}",
                "scope": scope,
                "scopeLabel": SCOPE_LABELS[scope],
                "view": view,
                "viewLabel": VIEW_LABELS[view],
                "task": task,
                "image": relative.as_posix(),
                "csv": csv_relative.as_posix(),
                "imageBytes": image_path.stat().st_size,
                "csvBytes": csv_path.stat().st_size,
                "columns": columns,
                "rows": rows,
            }
        )

    expected_csv = {
        Path(item["csv"]).as_posix()  # type: ignore[arg-type]
        for item in manifest
    }
    orphan_csv = csv_relatives - expected_csv
    orphan_png = png_relatives - {str(item["image"]) for item in manifest}
    if orphan_csv or orphan_png:
        raise ValueError(
            "Unpaired Triptych files detected: "
            f"orphan CSV={sorted(orphan_csv)}, orphan PNG={sorted(orphan_png)}"
        )

    scope_rank = {scope: index for index, scope in enumerate(SCOPE_LABELS)}
    view_rank = {view: index for index, view in enumerate(VIEW_LABELS)}
    manifest.sort(
        key=lambda item: (
            scope_rank[str(item["scope"])],
            view_rank[str(item["view"])],
            TASK_ORDER.get(str(item["task"]), 999),
            str(item["task"]),
        )
    )
    return manifest


def write_viewer(root: Path) -> Path:
    manifest = build_manifest(root)
    data_json = json.dumps(manifest, ensure_ascii=False, separators=(",", ":"))
    data_json = (
        data_json.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    html = HTML_TEMPLATE.replace("__DATA_JSON__", data_json).replace(
        "__GENERATED_AT__", datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    )
    output = root / "index.html"
    output.write_text(html, encoding="utf-8", newline="\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a standalone file:// Triptych PNG–CSV viewer."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    output = write_viewer(root)
    manifest = build_manifest(root)
    print(f"Wrote {output}")
    print(f"Embedded {len(manifest)} strict PNG–CSV pairs")
    print(f"HTML size: {output.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
