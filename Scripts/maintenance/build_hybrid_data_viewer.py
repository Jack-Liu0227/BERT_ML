from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" href="data:,">
  <title>Hybrid OOD data viewer</title>
  <style>
    :root {
      --ink:#172025; --muted:#66737a; --line:#d8e0e3; --paper:#fff;
      --wash:#f1f4f4; --nav:#1d282d; --accent:#087f78; --accent-soft:#e2f1ef;
      --panel:#38598a; --triptych:#c45b18; --yz:#087f78; --werror:#9d3d4e;
    }
    * { box-sizing:border-box; }
    html { scroll-behavior:smooth; }
    body { margin:0; color:var(--ink); background:var(--wash); font-family:Aptos,"Segoe UI","Microsoft YaHei",sans-serif; font-size:14px; line-height:1.45; }
    button,select,input,a { font:inherit; }
    button,select,input,a { -webkit-tap-highlight-color:transparent; }
    .topbar { position:sticky; top:0; z-index:30; display:flex; align-items:center; justify-content:space-between; gap:24px; min-height:70px; padding:11px 26px; color:#fff; background:var(--nav); border-bottom:3px solid var(--accent); }
    .brand { min-width:0; }
    .brand h1 { margin:0; font-family:Georgia,"Times New Roman",serif; font-size:23px; font-weight:600; letter-spacing:.01em; }
    .brand p { margin:2px 0 0; color:#b8c5ca; font-size:12px; }
    .counts { display:flex; gap:20px; flex:0 0 auto; }
    .count { text-align:right; }
    .count strong { display:block; font-size:16px; color:#fff; }
    .count span { color:#aeb9be; font-size:10px; letter-spacing:.08em; text-transform:uppercase; }
    .shell { max-width:1780px; margin:0 auto; padding:18px 24px 36px; }
    .category-bar { display:flex; align-items:center; justify-content:space-between; gap:16px; margin-bottom:14px; }
    .categories { display:flex; gap:5px; min-width:0; overflow-x:auto; padding:3px; border:1px solid var(--line); border-radius:7px; background:#e4e9eb; }
    .categories button { position:relative; min-width:165px; padding:8px 13px 8px 17px; border:0; border-radius:5px; color:#3e4b51; background:transparent; cursor:pointer; white-space:nowrap; }
    .categories button::before { content:""; position:absolute; left:7px; top:50%; width:4px; height:16px; border-radius:4px; background:var(--cat-color,#687078); transform:translateY(-50%); opacity:.7; }
    .categories button[aria-selected="true"] { color:#fff; background:var(--cat-color,#087f78); }
    .categories button[aria-selected="true"]::before { background:#fff; opacity:.9; }
    .source-note { color:var(--muted); font-size:12px; text-align:right; }
    .workspace { display:grid; grid-template-columns:minmax(245px,300px) minmax(0,1fr); min-height:760px; background:var(--paper); border:1px solid var(--line); border-radius:7px; overflow:hidden; box-shadow:0 10px 30px rgba(30,45,52,.05); }
    .sidebar { padding:19px; background:#edf1f2; border-right:1px solid var(--line); }
    .sidebar-title { margin:0 0 17px; font-family:Georgia,"Times New Roman",serif; font-size:16px; font-weight:600; }
    .field { margin-bottom:14px; }
    .field[hidden] { display:none; }
    .field label { display:flex; justify-content:space-between; gap:8px; margin-bottom:5px; color:#57646a; font-size:10px; font-weight:750; letter-spacing:.07em; text-transform:uppercase; }
    .field label span { color:#8a969b; font-weight:500; letter-spacing:0; text-transform:none; }
    select,input[type="search"] { width:100%; min-height:37px; padding:7px 9px; border:1px solid #bdc8cd; border-radius:4px; color:var(--ink); background:#fff; outline:none; }
    select:focus,input[type="search"]:focus { border-color:var(--accent); box-shadow:0 0 0 2px color-mix(in srgb,var(--accent) 22%,transparent); }
    .sidebar-rule { height:1px; margin:18px 0; background:var(--line); }
    .pair-card { padding:12px; border-left:3px solid var(--accent); background:#fff; color:#536168; font-size:12px; }
    .pair-card strong { display:block; margin-bottom:4px; color:var(--ink); font-size:12px; }
    .pair-card code { display:block; margin-top:4px; color:#647177; font-family:"Cascadia Mono",Consolas,monospace; font-size:10px; overflow-wrap:anywhere; }
    .content { min-width:0; padding:21px 23px 27px; }
    .figure-head { display:flex; align-items:flex-start; justify-content:space-between; gap:18px; margin-bottom:12px; }
    .figure-head h2 { margin:0; font-family:Georgia,"Times New Roman",serif; font-size:20px; font-weight:600; }
    .figure-head p { margin:4px 0 0; color:var(--muted); font-size:11px; overflow-wrap:anywhere; }
    .chips { display:flex; flex-wrap:wrap; gap:5px; margin-top:8px; }
    .chip { padding:3px 7px; border:1px solid var(--line); border-radius:999px; color:#4f5d63; background:#f6f8f8; font-size:10px; }
    .actions { display:flex; gap:7px; flex:0 0 auto; }
    .action { display:inline-flex; align-items:center; justify-content:center; min-height:34px; padding:6px 10px; border:1px solid #bbc6cb; border-radius:4px; color:#263238; background:#fff; text-decoration:none; cursor:pointer; }
    .action:hover { border-color:var(--accent); color:var(--accent); }
    .figure-stage { position:relative; display:grid; place-items:center; min-height:440px; padding:12px; background:#fafbfb; border:1px solid var(--line); border-radius:4px; overflow:hidden; }
    .figure-stage::after { content:"PNG"; position:absolute; right:10px; bottom:8px; color:#9aa5aa; font-size:9px; letter-spacing:.12em; }
    .figure-stage img { display:block; max-width:100%; max-height:800px; width:auto; height:auto; object-fit:contain; cursor:zoom-in; }
    .data-head { display:flex; align-items:end; justify-content:space-between; gap:15px; margin-top:23px; padding-top:18px; border-top:2px solid var(--nav); }
    .data-head h3 { margin:0; font-family:Georgia,"Times New Roman",serif; font-size:17px; font-weight:600; }
    .data-head p { margin:3px 0 0; color:var(--muted); font-size:11px; }
    .filters { display:grid; grid-template-columns:minmax(210px,1fr) 108px; gap:8px; width:min(480px,100%); }
    .table-status { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-top:9px; color:var(--muted); font-size:11px; }
    .pager { display:flex; align-items:center; gap:6px; }
    .pager button { min-width:31px; height:29px; padding:3px 7px; border:1px solid #c4cdd1; border-radius:4px; color:#334047; background:#fff; cursor:pointer; }
    .pager button:disabled { cursor:not-allowed; opacity:.42; }
    .table-wrap { position:relative; max-height:540px; margin-top:7px; overflow:auto; border:1px solid var(--line); background:#fff; }
    table { width:100%; border-collapse:collapse; white-space:nowrap; font-size:11px; font-variant-numeric:tabular-nums; }
    th,td { padding:7px 9px; border-bottom:1px solid #e3e8ea; text-align:right; }
    th:first-child,td:first-child { text-align:left; }
    th { position:sticky; top:0; z-index:2; color:#fff; background:#303b40; font-size:10px; font-weight:650; }
    tbody tr:nth-child(even) { background:#f7f9f9; }
    tbody tr:hover { background:var(--accent-soft); }
    td.na { color:#939da1; font-style:italic; }
    .loading { display:grid; place-items:center; min-height:180px; color:var(--muted); }
    .loading::before { content:""; width:22px; height:22px; margin-right:8px; border:2px solid #cad4d8; border-top-color:var(--accent); border-radius:50%; animation:spin .75s linear infinite; }
    @keyframes spin { to { transform:rotate(360deg); } }
    dialog { width:min(96vw,1850px); max-width:none; height:min(94vh,1220px); padding:0; border:0; border-radius:6px; background:#fff; box-shadow:0 24px 70px rgba(0,0,0,.35); }
    dialog::backdrop { background:rgba(15,21,24,.84); }
    .dialog-bar { display:flex; align-items:center; justify-content:space-between; padding:10px 14px; color:#fff; background:var(--nav); }
    .dialog-bar strong { overflow-wrap:anywhere; }
    .close { width:34px; height:34px; border:0; border-radius:4px; color:#fff; background:#47545a; font-size:20px; cursor:pointer; }
    .dialog-image { display:grid; place-items:center; height:calc(100% - 54px); overflow:auto; padding:12px; }
    .dialog-image img { max-width:none; width:auto; min-width:min(100%,1000px); height:auto; }
    @media(max-width:960px) {
      .topbar { align-items:flex-start; padding:10px 14px; }
      .counts { gap:10px; }
      .shell { padding:12px; }
      .category-bar { align-items:stretch; flex-direction:column; }
      .source-note { text-align:left; }
      .workspace { grid-template-columns:1fr; }
      .sidebar { border-right:0; border-bottom:1px solid var(--line); }
      .content { padding:16px 12px 22px; }
      .figure-head,.data-head { align-items:stretch; flex-direction:column; }
      .filters { grid-template-columns:1fr; }
      .figure-stage { min-height:270px; padding:4px; }
      .actions { align-self:flex-start; }
    }
    @media(max-width:520px) {
      .topbar { gap:8px; }
      .brand h1 { font-size:20px; }
      .brand p { max-width:155px; font-size:10px; }
      .counts { gap:7px; }
      .count strong { font-size:14px; }
      .count span { font-size:8px; letter-spacing:.04em; }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="brand"><h1>Hybrid OOD data</h1><p>PNG figure above · matched CSV below</p></div>
    <div class="counts"><div class="count"><strong id="imageCount">0</strong><span>PNG</span></div><div class="count"><strong id="csvCount">0</strong><span>Paired CSV</span></div><div class="count"><strong id="taskCount">0</strong><span>Tasks</span></div></div>
  </header>
  <main class="shell">
    <div class="category-bar"><div class="categories" id="categoryTabs" role="tablist"></div><div class="source-note">Source: hybrid/figure_catalog.csv</div></div>
    <section class="workspace">
      <aside class="sidebar">
        <h2 class="sidebar-title" id="sidebarTitle">Browse results</h2>
        <div class="field" id="scopeField"><label for="scopeSelect">Scope <span id="scopeCount"></span></label><select id="scopeSelect"></select></div>
        <div class="field" id="viewField"><label for="viewSelect">View <span id="viewCount"></span></label><select id="viewSelect"></select></div>
        <div class="field" id="taskField"><label for="taskSelect">Task <span id="taskOptionCount"></span></label><select id="taskSelect"></select></div>
        <div class="field" id="methodField"><label for="methodSelect">Method <span id="methodCount"></span></label><select id="methodSelect"></select></div>
        <div class="field" id="modelField"><label for="modelSelect">Model <span id="modelCount"></span></label><select id="modelSelect"></select></div>
        <div class="field" id="spaceField"><label for="spaceSelect">Space <span id="spaceCount"></span></label><select id="spaceSelect"></select></div>
        <div class="field" id="figureField"><label for="figureSelect">Figure <span id="figureCount"></span></label><select id="figureSelect"></select></div>
        <div class="sidebar-rule"></div>
        <div class="pair-card"><strong>Strict PNG–CSV pairing</strong><span id="pairRows">0 rows</span><code id="pairPath"></code></div>
      </aside>
      <div class="content">
        <div class="figure-head">
          <div><h2 id="figureTitle"></h2><p id="figurePath"></p><div class="chips" id="metaChips"></div></div>
          <div class="actions"><a class="action" id="imageLink" target="_blank" rel="noopener">Open PNG</a><a class="action" id="csvLink" download>Download CSV</a></div>
        </div>
        <div class="figure-stage"><img id="figureImage" alt=""></div>
        <div class="data-head">
          <div><h3 id="tableTitle"></h3><p><span id="matchedRows">0</span> matched rows · <span id="csvPath"></span></p></div>
          <div class="filters"><input id="searchInput" type="search" placeholder="Search current CSV"><select id="pageSizeSelect"><option>100</option><option selected>250</option><option>500</option><option>1000</option></select></div>
        </div>
        <div class="table-status"><span id="visibleStatus">Loading data…</span><div class="pager"><button id="prevPage" type="button">‹</button><span id="pageLabel">1 / 1</span><button id="nextPage" type="button">›</button></div></div>
        <div class="table-wrap" id="tableWrap"><div class="loading">Loading matched CSV</div></div>
      </div>
    </section>
  </main>
  <dialog id="imageDialog"><div class="dialog-bar"><strong id="dialogTitle"></strong><button class="close" id="closeDialog" type="button" aria-label="Close">×</button></div><div class="dialog-image"><img id="dialogImage" alt=""></div></dialog>
  <script src="__CATALOG_SRC__"></script>
  <script>
    const CATALOG=window.HYBRID_VIEWER_CATALOG;
    window.HYBRID_TABLE_DATA=window.HYBRID_TABLE_DATA||{};
    const byId=(id)=>document.getElementById(id); const text=(value)=>String(value??'');
    const ANALYSES={panel_distribution:{label:'Panel distribution',color:'#38598a'},triptych:{label:'Triptych',color:'#c45b18'},yz_severity_map:{label:'YZ severity map',color:'#087f78'},w_error_relationship:{label:'W-error relationship',color:'#9d3d4e'}};
    const LABELS={combined:'Combined',test_extrapolation_high20:'Extrapolation high20',test_inner_ood:'Inner OOD',main:'Main',subset_comparison:'Subset comparison',panel_distribution:'Panel distribution',overview:'Overview',set_ab_comparison:'Set A vs Set B (B-RandCV baseline)',by_task:'By task',by_task_model:'By task & model',by_method:'By method',phase_diagram:'Phase diagram',all_methods:'All methods',all_models:'All models','RandCV+LOCO':'Core protocols (RandCV + LOCO)'};
    const filterOrder=['scope','category','task_id','method','model','rep_space'];
    const state={analysis:CATALOG.entries[0].analysis,scope:'',category:'',task_id:'',method:'',model:'',rep_space:'',figure_id:'',page:1,pageSize:250,query:'',loadToken:0};
    let currentTable={columns:[],data:[]}; let filteredIndexes=[];
    function label(value){return LABELS[value]||value||'All';}
    function unique(entries,key){return [...new Set(entries.map((item)=>text(item[key])).filter(Boolean))].sort((a,b)=>a.localeCompare(b,undefined,{numeric:true}));}
    function setOptions(select,values,current){select.innerHTML=''; values.forEach((value)=>{const option=document.createElement('option');option.value=value;option.textContent=label(value);select.append(option);}); if(values.includes(current))select.value=current; else if(values.length)select.value=values[0]; return select.value;}
    function baseEntries(){return CATALOG.entries.filter((item)=>item.analysis===state.analysis);}
    function cascade(startIndex=0){
      let entries=baseEntries();
      filterOrder.forEach((key,index)=>{
        const select=byId(({scope:'scopeSelect',category:'viewSelect',task_id:'taskSelect',method:'methodSelect',model:'modelSelect',rep_space:'spaceSelect'})[key]);
        const field=byId(({scope:'scopeField',category:'viewField',task_id:'taskField',method:'methodField',model:'modelField',rep_space:'spaceField'})[key]);
        let values=unique(entries,key);
        const globalSetAb=key==='category'&&state.analysis==='yz_severity_map'&&baseEntries().some((item)=>item.category==='set_ab_comparison');
        if(globalSetAb&&!values.includes('set_ab_comparison'))values=[...values,'set_ab_comparison'].sort((a,b)=>a.localeCompare(b,undefined,{numeric:true}));
        field.hidden=values.length===0;
        if(values.length){
          state[key]=setOptions(select,values,index<startIndex?state[key]:'');
          if(key==='category'&&state.analysis==='yz_severity_map'&&state[key]==='set_ab_comparison')entries=baseEntries().filter((item)=>item.category==='set_ab_comparison');
          else entries=entries.filter((item)=>text(item[key])===state[key]);
        } else state[key]='';
        const counter=field.querySelector('label span'); if(counter)counter.textContent=values.length?values.length:'';
      });
      const figures=entries.sort((a,b)=>a.title.localeCompare(b.title,undefined,{numeric:true}));
      byId('figureCount').textContent=figures.length;
      byId('figureSelect').innerHTML=''; figures.forEach((item)=>{const option=document.createElement('option'); option.value=item.figure_id; option.textContent=item.title; byId('figureSelect').append(option);});
      if(figures.some((item)=>item.figure_id===state.figure_id))byId('figureSelect').value=state.figure_id; else if(figures.length){state.figure_id=figures[0].figure_id;byId('figureSelect').value=state.figure_id;}
      renderEntry();
    }
    function buildTabs(){const tabs=byId('categoryTabs');tabs.innerHTML='';Object.entries(ANALYSES).forEach(([key,meta])=>{if(!CATALOG.entries.some((item)=>item.analysis===key))return;const button=document.createElement('button');button.type='button';button.dataset.analysis=key;button.textContent=meta.label;button.style.setProperty('--cat-color',meta.color);button.setAttribute('aria-selected',String(key===state.analysis));button.onclick=()=>{state.analysis=key;state.figure_id='';document.documentElement.style.setProperty('--accent',meta.color);document.documentElement.style.setProperty('--accent-soft',meta.color+'1c');tabs.querySelectorAll('button').forEach((node)=>node.setAttribute('aria-selected',String(node===button)));cascade(0);};tabs.append(button);});}
    function currentEntry(){return CATALOG.entries.find((item)=>item.figure_id===state.figure_id);}
    function chips(entry){const values=[entry.scope,entry.category,entry.task_id,entry.method,entry.model,entry.rep_space].filter(Boolean);byId('metaChips').innerHTML='';values.forEach((value)=>{const span=document.createElement('span');span.className='chip';span.textContent=label(value);byId('metaChips').append(span);});}
    function renderEntry(){const entry=currentEntry();if(!entry)return;state.page=1;state.query='';byId('searchInput').value='';byId('figureTitle').textContent=entry.title;byId('figurePath').textContent=entry.figure_path;byId('figureImage').src=CATALOG.image_root+entry.figure_path;byId('figureImage').alt=entry.title;byId('imageLink').href=CATALOG.image_root+entry.figure_path;byId('csvLink').href=CATALOG.image_root+entry.csv_path;byId('tableTitle').textContent=entry.csv_path.split('/').pop();byId('csvPath').textContent=entry.csv_path;byId('pairPath').textContent=entry.csv_path;byId('pairRows').textContent=entry.row_count+' rows';byId('matchedRows').textContent=entry.row_count;chips(entry);loadTable(entry);}
    function loadTable(entry){const token=++state.loadToken;byId('tableWrap').innerHTML='<div class="loading">Loading matched CSV</div>';if(window.HYBRID_TABLE_DATA[entry.figure_id]){setTable(window.HYBRID_TABLE_DATA[entry.figure_id]);return;}const existing=document.querySelector(`script[data-figure-id="${entry.figure_id}"]`);if(existing)existing.remove();const script=document.createElement('script');script.src=entry.data_script;script.dataset.figureId=entry.figure_id;script.onload=()=>{if(token!==state.loadToken)return;const table=window.HYBRID_TABLE_DATA[entry.figure_id];if(!table)showTableError('Data module loaded without table data.');else setTable(table);};script.onerror=()=>{if(token===state.loadToken)showTableError('Unable to load the matched CSV data module.');};document.body.append(script);}
    function showTableError(message){byId('tableWrap').innerHTML='';const div=document.createElement('div');div.className='loading';div.textContent=message;byId('tableWrap').append(div);}
    function setTable(table){currentTable=table;applySearch();}
    function applySearch(){const q=state.query.trim().toLowerCase();filteredIndexes=[];currentTable.data.forEach((row,index)=>{if(!q||row.some((value)=>text(value).toLowerCase().includes(q)))filteredIndexes.push(index);});const maxPage=Math.max(1,Math.ceil(filteredIndexes.length/state.pageSize));state.page=Math.min(state.page,maxPage);renderTable();}
    function formatValue(value){if(value===null||value===undefined||value==='')return 'NA';if(typeof value==='number')return Number.isFinite(value)?Number(value.toPrecision(9)).toString():'NA';return text(value);}
    function renderTable(){const maxPage=Math.max(1,Math.ceil(filteredIndexes.length/state.pageSize));const start=(state.page-1)*state.pageSize;const indexes=filteredIndexes.slice(start,start+state.pageSize);const table=document.createElement('table');const thead=document.createElement('thead');const hr=document.createElement('tr');currentTable.columns.forEach((column)=>{const th=document.createElement('th');th.textContent=column;hr.append(th);});thead.append(hr);table.append(thead);const tbody=document.createElement('tbody');indexes.forEach((index)=>{const tr=document.createElement('tr');currentTable.data[index].forEach((value)=>{const td=document.createElement('td');const formatted=formatValue(value);td.textContent=formatted;if(formatted==='NA')td.className='na';tr.append(td);});tbody.append(tr);});table.append(tbody);byId('tableWrap').innerHTML='';byId('tableWrap').append(table);byId('visibleStatus').textContent=`${filteredIndexes.length} matching rows · showing ${indexes.length}`;byId('pageLabel').textContent=`${state.page} / ${maxPage}`;byId('prevPage').disabled=state.page<=1;byId('nextPage').disabled=state.page>=maxPage;}
    function bindCascade(){const mapping=[['scopeSelect',0],['viewSelect',1],['taskSelect',2],['methodSelect',3],['modelSelect',4],['spaceSelect',5]];mapping.forEach(([id,index])=>{byId(id).onchange=(event)=>{state[filterOrder[index]]=event.target.value;filterOrder.slice(index+1).forEach((key)=>state[key]='');state.figure_id='';cascade(index+1);};});byId('figureSelect').onchange=(event)=>{state.figure_id=event.target.value;renderEntry();};}
    byId('searchInput').oninput=(event)=>{state.query=event.target.value;state.page=1;applySearch();};byId('pageSizeSelect').onchange=(event)=>{state.pageSize=Number(event.target.value);state.page=1;renderTable();};byId('prevPage').onclick=()=>{if(state.page>1){state.page--;renderTable();}};byId('nextPage').onclick=()=>{const max=Math.max(1,Math.ceil(filteredIndexes.length/state.pageSize));if(state.page<max){state.page++;renderTable();}};
    byId('figureImage').onclick=()=>{const entry=currentEntry();byId('dialogTitle').textContent=entry.title;byId('dialogImage').src=CATALOG.image_root+entry.figure_path;byId('imageDialog').showModal();};byId('closeDialog').onclick=()=>byId('imageDialog').close();byId('imageDialog').onclick=(event)=>{if(event.target===byId('imageDialog'))byId('imageDialog').close();};
    byId('imageCount').textContent=CATALOG.entries.length;byId('csvCount').textContent=CATALOG.entries.length;byId('taskCount').textContent=unique(CATALOG.entries,'task_id').length;buildTabs();bindCascade();document.documentElement.style.setProperty('--accent',ANALYSES[state.analysis].color);cascade(0);
  </script>
</body>
</html>
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local-file Hybrid PNG/CSV viewer.")
    parser.add_argument("--hybrid-root", type=Path, required=True)
    parser.add_argument("--viewer-root", type=Path, required=True)
    parser.add_argument("--data-dir", default="data/hybrid")
    parser.add_argument("--catalog-name", default="catalog.js")
    parser.add_argument("--index-name", default="index.html")
    parser.add_argument("--image-root", default="../hybrid/")
    parser.add_argument("--skip-data", action="store_true")
    return parser.parse_args()


def js_table_payload(frame: pd.DataFrame) -> str:
    return frame.to_json(orient="split", index=False, force_ascii=False, double_precision=10)


def main() -> None:
    args = parse_args()
    hybrid_root = args.hybrid_root.resolve()
    viewer_root = args.viewer_root.resolve()
    data_dir = viewer_root / args.data_dir
    catalog_path = data_dir / args.catalog_name
    index_path = viewer_root / args.index_name
    viewer_root.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    catalog = pd.read_csv(hybrid_root / "figure_catalog.csv", low_memory=False)
    entries = catalog.where(pd.notna(catalog), "").to_dict(orient="records")
    for item in entries:
        item["data_script"] = f"{args.data_dir.replace('\\', '/')}/{item['figure_id']}.js"

    if not args.skip_data:
        for index, item in enumerate(entries, start=1):
            csv_path = hybrid_root / item["csv_path"]
            frame = pd.read_csv(csv_path, low_memory=False)
            payload = js_table_payload(frame)
            module_path = data_dir / f"{item['figure_id']}.js"
            module_path.write_text(
                f"window.HYBRID_TABLE_DATA=window.HYBRID_TABLE_DATA||{{}};window.HYBRID_TABLE_DATA[{json.dumps(item['figure_id'])}]={payload};",
                encoding="utf-8",
            )
            if index % 250 == 0:
                print(f"viewer data: {index}/{len(entries)}")

    catalog_payload = {
        "image_root": args.image_root,
        "entries": entries,
    }
    catalog_path.write_text(
        "window.HYBRID_VIEWER_CATALOG=" + json.dumps(catalog_payload, ensure_ascii=False, separators=(",", ":")) + ";",
        encoding="utf-8",
    )
    catalog_src = f"{args.data_dir.replace('\\', '/')}/{args.catalog_name}"
    index_path.write_text(HTML_TEMPLATE.replace("__CATALOG_SRC__", catalog_src), encoding="utf-8")
    result = {
        "index": str(index_path),
        "catalog": str(catalog_path),
        "entries": len(entries),
        "data_modules": len(list(data_dir.glob("hybrid_*.js"))),
        "data_size_bytes": sum(path.stat().st_size for path in data_dir.glob("*.js")),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
