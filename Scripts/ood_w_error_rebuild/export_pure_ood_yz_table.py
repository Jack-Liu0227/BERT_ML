from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    REPO_ROOT
    / "output"
    / "w_error_relationship_full"
    / "OOD"
    / "yz_severity_map"
    / "csv"
    / "ood_yz_severity_summary.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "output"
    / "w_error_relationship_full"
    / "OOD"
    / "yz_severity_map"
    / "tables"
    / "pure_ood_yz_table_preview.html"
)

TASK_ORDER = [
    "Ti-UTS",
    "Ti-El",
    "Al-UTS",
    "HEA-UTS",
    "HEA-El",
    "Steel-UTS",
    "Steel-El",
    "Matbench Steel-YS",
]
METHOD_ORDER = ["RandCV", "Extra.", "LOCO", "SX-sgl", "SX-cls", "SY-sgl", "SY-cls"]
METHOD_COLORS = {
    "RandCV": "#666666",
    "Extra.": "#2F6DB0",
    "LOCO": "#D62728",
    "SX-sgl": "#F28E2B",
    "SX-cls": "#C65D00",
    "SY-sgl": "#2A9D8F",
    "SY-cls": "#147D75",
}
EXCLUDED_TASKS = {"HEA-YS", "Steel-YS"}


def load_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"task_id", "method_short", "Wy", "Wz", "Ry", "Rz"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    frame = frame[~frame["task_id"].astype(str).isin(EXCLUDED_TASKS)].copy()
    frame = frame[frame["task_id"].isin(TASK_ORDER) & frame["method_short"].isin(METHOD_ORDER)].copy()
    for column in ["Wy", "Wz", "Ry", "Rz"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    frame["_task_order"] = frame["task_id"].map({value: index for index, value in enumerate(TASK_ORDER)})
    frame["_method_order"] = frame["method_short"].map({value: index for index, value in enumerate(METHOD_ORDER)})
    frame = frame.sort_values(["_task_order", "_method_order"], kind="stable").reset_index(drop=True)

    if len(frame) != len(TASK_ORDER) * len(METHOD_ORDER):
        raise ValueError(f"Expected 56 task-method rows, found {len(frame)}")
    counts = frame.groupby("task_id")["method_short"].nunique().to_dict()
    if any(counts.get(task) != len(METHOD_ORDER) for task in TASK_ORDER):
        raise ValueError(f"Each task must contain seven protocols: {counts}")
    return frame


def build_html(frame: pd.DataFrame) -> str:
    pearson_r, pearson_p = pearsonr(frame["Wy"], frame["Wz"])
    body_rows: list[str] = []
    for task in TASK_ORDER:
        group = frame[frame["task_id"].eq(task)]
        for row_index, (_, row) in enumerate(group.iterrows()):
            cells: list[str] = []
            if row_index == 0:
                cells.append(f'<th class="task" rowspan="{len(group)}">{html.escape(task)}</th>')
            method = str(row["method_short"])
            color = METHOD_COLORS[method]
            cells.extend(
                [
                    '<td class="protocol">'
                    f'<span class="swatch" style="background:{color}"></span>'
                    f'{html.escape(method)}</td>',
                    f'<td class="numeric">{float(row["Ry"]):.3f}</td>',
                    f'<td class="numeric">{float(row["Rz"]):.3f}</td>',
                ]
            )
            row_class = " baseline" if method == "RandCV" else ""
            body_rows.append(f'<tr class="data-row{row_class}">{"".join(cells)}</tr>')

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pure OOD Y-Z severity data</title>
  <style>
    :root {{ color-scheme: light; }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: #f3f4f6;
      color: #171717;
      font-family: Arial, Helvetica, sans-serif;
      font-size: 14px;
      line-height: 1.35;
    }}
    main {{
      width: min(760px, calc(100% - 32px));
      margin: 28px auto;
      padding: 30px 36px 26px;
      background: #ffffff;
      border: 1px solid #d7d7d7;
    }}
    h1 {{ margin: 0 0 5px; text-align: center; font-size: 20px; font-weight: 700; }}
    .subtitle {{ margin: 0 0 20px; text-align: center; color: #555555; font-size: 12px; }}
    table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
    col.task-col {{ width: 28%; }}
    col.protocol-col {{ width: 32%; }}
    col.value-col {{ width: 20%; }}
    thead th {{
      padding: 8px 10px;
      border-top: 1.6px solid #1f1f1f;
      border-bottom: 1.1px solid #1f1f1f;
      text-align: left;
      font-size: 12px;
      font-weight: 700;
    }}
    thead th.numeric {{ text-align: right; }}
    tbody th, tbody td {{ padding: 6px 10px; border-bottom: 1px solid #dddddd; }}
    tbody th.task {{
      vertical-align: middle;
      text-align: left;
      font-size: 12px;
      font-weight: 700;
      background: #f7f7f7;
      border-top: 1.1px solid #777777;
    }}
    tbody tr.baseline td {{ background: #fafafa; }}
    tbody tr:last-child td, tbody tr:last-child th {{ border-bottom: 1.6px solid #1f1f1f; }}
    td.protocol {{ white-space: nowrap; }}
    td.numeric {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .swatch {{
      display: inline-block;
      width: 8px;
      height: 8px;
      margin-right: 8px;
      border-radius: 50%;
      vertical-align: 1px;
    }}
    .notes {{ margin-top: 14px; font-size: 11px; color: #444444; }}
    .notes p {{ margin: 3px 0; }}
    @media print {{
      @page {{ size: A4 portrait; margin: 16mm 18mm; }}
      body {{ background: white; font-size: 9pt; }}
      main {{ width: auto; margin: 0; padding: 0; border: 0; }}
      h1 {{ font-size: 12pt; }}
      .subtitle {{ font-size: 8pt; }}
      thead {{ display: table-header-group; }}
      tbody {{ break-inside: auto; }}
      tbody th, tbody td {{ padding: 3.5pt 6pt; }}
      tr {{ break-inside: avoid; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Pure OOD Y-Z Severity Data</h1>
    <p class="subtitle">All data points shown in the Pure OOD overview</p>
    <table aria-label="Pure OOD Y-Z severity data">
      <colgroup>
        <col class="task-col">
        <col class="protocol-col">
        <col class="value-col">
        <col class="value-col">
      </colgroup>
      <thead>
        <tr><th>Task</th><th>Protocol</th><th class="numeric">R<sub>y</sub></th><th class="numeric">R<sub>z</sub></th></tr>
      </thead>
      <tbody>
        {''.join(body_rows)}
      </tbody>
    </table>
    <section class="notes">
      <p><strong>Definition:</strong> R<sub>y</sub> = W<sub>y</sub> / W<sub>y</sub>(RandCV); R<sub>z</sub> = W<sub>z</sub> / W<sub>z</sub>(RandCV).</p>
      <p><strong>Raw W correlation:</strong> Pearson r = {pearson_r:.3f}, p = {pearson_p:.4f}, n = {len(frame)}.</p>
      <p>Values are rounded to three decimal places for presentation.</p>
    </section>
  </main>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a Word-ready Pure OOD Y-Z HTML table preview.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_table(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_html(frame), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
