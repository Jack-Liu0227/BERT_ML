import fs from "node:fs/promises";
import path from "node:path";

import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


function columnLabel(index) {
  let value = index + 1;
  let label = "";
  while (value > 0) {
    const remainder = (value - 1) % 26;
    label = String.fromCharCode(65 + remainder) + label;
    value = Math.floor((value - 1) / 26);
  }
  return label;
}


function numberFormatFor(column) {
  if (column === "training_fraction") return "0%";
  if (column.endsWith("_count") || column === "repeat_id") return "#,##0";
  if (
    column.includes("mae") ||
    column.includes("ratio") ||
    column.includes("ci95") ||
    column.includes("sensitivity")
  ) {
    return "0.0000";
  }
  return null;
}


function widthFor(column) {
  if (column === "task_id") return 46;
  if (column === "failure_axis") return 28;
  if (column.includes("method") || column.includes("model")) return 24;
  if (column.includes("selection_rule")) return 38;
  return 18;
}


function safeTableName(workbookIndex, sheetIndex) {
  return `Figure4Data_${workbookIndex + 1}_${sheetIndex + 1}`;
}


async function buildWorkbook(spec, workbookIndex, verifyRender) {
  const workbook = Workbook.create();
  for (let sheetIndex = 0; sheetIndex < spec.sheets.length; sheetIndex += 1) {
    const sheetSpec = spec.sheets[sheetIndex];
    const sheet = workbook.worksheets.add(sheetSpec.name);
    sheet.showGridLines = false;
    const columns = sheetSpec.columns;
    const rows = sheetSpec.rows;
    const matrix = [columns, ...rows.map((row) => columns.map((column) => row[column] ?? null))];
    const lastColumn = columnLabel(columns.length - 1);
    const lastRow = matrix.length;
    const usedRange = sheet.getRange(`A1:${lastColumn}${lastRow}`);
    usedRange.values = matrix;
    sheet.getRange(`A1:${lastColumn}1`).format = {
      fill: "#30383D",
      font: { bold: true, color: "#FFFFFF" },
      rowHeight: 24,
      verticalAlignment: "center",
    };
    if (lastRow > 1) {
      const body = sheet.getRange(`A2:${lastColumn}${lastRow}`);
      body.format = {
        font: { color: "#171A1D" },
        borders: {
          insideHorizontal: { style: "thin", color: "#E2E6E8" },
        },
      };
    }
    for (let columnIndex = 0; columnIndex < columns.length; columnIndex += 1) {
      const column = columns[columnIndex];
      const range = sheet.getRange(
        `${columnLabel(columnIndex)}1:${columnLabel(columnIndex)}${lastRow}`,
      );
      range.format.columnWidth = widthFor(column);
      const numberFormat = numberFormatFor(column);
      if (numberFormat && lastRow > 1) {
        sheet
          .getRange(
            `${columnLabel(columnIndex)}2:${columnLabel(columnIndex)}${lastRow}`,
          )
          .setNumberFormat(numberFormat);
      }
    }
    sheet.freezePanes.freezeRows(1);
    const table = sheet.tables.add(
      `A1:${lastColumn}${lastRow}`,
      true,
      safeTableName(workbookIndex, sheetIndex),
    );
    table.style = "TableStyleMedium2";
    table.showBandedRows = true;
    table.showFilterButton = true;
  }

  if (verifyRender) {
    for (const sheetSpec of spec.sheets) {
      const preview = await workbook.render({
        sheetName: sheetSpec.name,
        autoCrop: "all",
        scale: 0.5,
        format: "png",
      });
      const bytes = new Uint8Array(await preview.arrayBuffer());
      if (bytes.length < 1000) {
        throw new Error(`Rendered sheet is unexpectedly small: ${spec.path} / ${sheetSpec.name}`);
      }
    }
  }

  const output = await SpreadsheetFile.exportXlsx(workbook);
  await fs.mkdir(path.dirname(spec.path), { recursive: true });
  await output.save(spec.path);
  await fs.rm(`${spec.path}.inspect.ndjson`, { force: true });
}


const manifestPath = process.argv[2];
if (!manifestPath) {
  throw new Error("Usage: node build_figure4_workbooks.mjs <manifest.json>");
}
const manifest = JSON.parse(await fs.readFile(manifestPath, "utf8"));
for (let index = 0; index < manifest.workbooks.length; index += 1) {
  await buildWorkbook(manifest.workbooks[index], index, Boolean(manifest.verify_render));
}
process.stdout.write(
  `${JSON.stringify({ workbook_count: manifest.workbooks.length })}\n`,
);
