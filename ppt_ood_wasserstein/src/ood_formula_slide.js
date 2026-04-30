import {
  Presentation,
  PresentationFile,
  row,
  column,
  grid,
  panel,
  image,
  text,
  rule,
  fill,
  hug,
  fixed,
  fr,
} from "@oai/artifact-tool";
import { writeFile } from "node:fs/promises";
import { readFileSync } from "node:fs";

const W = 1920;
const H = 1080;
const navy = "#0B1220";
const slate = "#334155";
const muted = "#64748B";
const blue = "#2563EB";
const amber = "#D97706";
const green = "#059669";
const bg = "#F8FAFC";
const line = "#CBD5E1";

const titleStyle = { fontSize: 54, bold: true, color: navy, fontFace: "Aptos Display" };
const subStyle = { fontSize: 24, color: slate, fontFace: "Aptos" };
const labelStyle = { fontSize: 19, color: muted, bold: true, fontFace: "Aptos" };
const formulaStyle = { fontSize: 28, color: navy, fontFace: "Cambria Math" };
const formulaSmallStyle = { fontSize: 18, color: muted, fontFace: "Cambria Math" };
const bodyStyle = { fontSize: 21, color: slate, fontFace: "Aptos" };
const smallStyle = { fontSize: 16, color: muted, fontFace: "Aptos" };

function pngDataUrl(path) {
  return `data:image/png;base64,${readFileSync(path).toString("base64")}`;
}

function formulaImage(path, height, name) {
  return image({
    name,
    dataUrl: pngDataUrl(path),
    contentType: "image/png",
    width: fill,
    height: fixed(height),
    fit: "contain",
    alt: name,
  });
}

function metricBlock(name, accent, symbol, formulaPath, meaning, unit) {
  return panel(
    {
      name,
      width: fill,
      height: fill,
      padding: { x: 26, y: 22 },
      background: "#FFFFFF",
      border: { color: line, weight: 1 },
      borderRadius: 20,
    },
    column({ width: fill, height: fill, gap: 12 }, [
      row({ width: fill, height: hug, gap: 12, align: "center" }, [
        text(symbol, { width: fill, height: hug, style: { fontSize: 34, bold: true, color: accent, fontFace: "Aptos Display" } }),
      ]),
      formulaImage(formulaPath, 46, `${name}-formula`),
      rule({ width: fill, stroke: "#E2E8F0", weight: 1 }),
      text(meaning, { width: fill, height: hug, style: bodyStyle }),
      text(unit, { width: fill, height: hug, style: smallStyle }),
    ])
  );
}

const presentation = Presentation.create({ slideSize: { width: W, height: H } });
const slide = presentation.slides.add();

slide.compose(
  column(
    { name: "root", width: fill, height: fill, padding: { x: 72, y: 54 }, gap: 26, background: bg },
    [
      column({ name: "header", width: fill, height: hug, gap: 10 }, [
        text("How each OOD-severity table cell is computed", {
          name: "title",
          width: fill,
          height: hug,
          style: titleStyle,
        }),
        text("Cell entry = test-set size; Wasserstein shift in property space / representation space", {
          name: "subtitle",
          width: fill,
          height: hug,
          style: subStyle,
        }),
      ]),

      panel(
        {
          name: "cell-format-band",
          width: fill,
          height: fixed(132),
          padding: { x: 34, y: 22 },
          background: "#EAF2FF",
          border: { color: "#BFDBFE", weight: 1 },
          borderRadius: 26,
        },
        row({ width: fill, height: fill, gap: 28, align: "center" }, [
          column({ width: fixed(560), height: hug, gap: 5 }, [
            text("Cell format", { width: fill, height: hug, style: { ...labelStyle, color: blue } }),
            image({
              name: "cell-format-formula",
              dataUrl: pngDataUrl("scratch/formulas/cell.png"),
              contentType: "image/png",
              width: fill,
              height: fixed(54),
              fit: "contain",
              alt: "n test; W1(y) divided by W1(z)",
            }),
          ]),
          text("Read as: number of OOD test samples; target-property distribution shift divided by 2D material-representation shift.", {
            width: fill,
            height: hug,
            style: { fontSize: 27, color: slate, fontFace: "Aptos" },
          }),
        ])
      ),

      grid(
        { name: "metric-grid", width: fill, height: fixed(378), columns: [fr(1), fr(1), fr(1)], rows: [fr(1)], columnGap: 22 },
        [
          metricBlock(
            "n-test-block",
            blue,
            "nₜₑₛₜ",
            "scratch/formulas/ntest.png",
            "Counts how many samples are held out under the OOD split protocol.",
            "Symbol: Dₜₑₛₜ = held-out OOD test set."
          ),
          metricBlock(
            "w1-y-block",
            amber,
            "W₁(y)",
            "scratch/formulas/w1y.png",
            "Measures how far the test target-property distribution is from the training target-property distribution.",
            "Physical meaning: property extrapolation severity, e.g., UTS or El shift."
          ),
          metricBlock(
            "w1-z-block",
            green,
            "W₁(z)",
            "scratch/formulas/w1z.png",
            "Measures how far the test sample cloud is from the training cloud in 2D t-SNE representation space.",
            "Physical meaning: material-feature / chemistry-process space shift."
          ),
        ]
      ),

      grid(
        { name: "bottom", width: fill, height: fill, columns: [fr(1.12), fr(0.88)], rows: [fr(1)], columnGap: 28 },
        [
          panel(
            { name: "w1-definition", width: fill, height: fill, padding: { x: 30, y: 24 }, background: "#FFFFFF", border: { color: line, weight: 1 }, borderRadius: 22 },
            column({ width: fill, height: fill, gap: 8 }, [
              text("Wasserstein-1 distance: minimum transport cost", { width: fill, height: hug, style: { fontSize: 27, bold: true, color: navy, fontFace: "Aptos Display" } }),
              formulaImage("scratch/formulas/w1def.png", 54, "wasserstein-definition-formula"),
              text("Intuition: the average distance needed to ‘move’ one empirical distribution into another.", { width: fill, height: hug, style: bodyStyle }),
              text("Empirical distributions:", { width: fill, height: hug, style: { fontSize: 18, color: muted, fontFace: "Aptos", bold: true } }),
              formulaImage("scratch/formulas/empirical.png", 36, "empirical-distribution-formula"),
            ])
          ),
          panel(
            { name: "loco", width: fill, height: fill, padding: { x: 28, y: 24 }, background: "#FFFBEB", border: { color: "#FDE68A", weight: 1 }, borderRadius: 22 },
            column({ width: fill, height: fill, gap: 10 }, [
              text("LOCO aggregation", { width: fill, height: hug, style: { fontSize: 27, bold: true, color: "#78350F", fontFace: "Aptos Display" } }),
              text("When multiple groups are held out:", { width: fill, height: hug, style: { fontSize: 20, color: "#92400E", fontFace: "Aptos" } }),
              formulaImage("scratch/formulas/loco.png", 46, "loco-formula"),
              text("same weighted average for W₁(z)", { width: fill, height: hug, style: { fontSize: 21, color: navy, fontFace: "Aptos" } }),
              text("k indexes the held-out group/fold; larger held-out groups contribute more.", { width: fill, height: hug, style: { fontSize: 18, color: "#92400E", fontFace: "Aptos" } }),
            ])
          ),
        ]
      ),

      text("Interpretation: larger W₁(y) = stronger property shift; larger W₁(z) = stronger material-representation shift.", {
        width: fill,
        height: hug,
        style: { fontSize: 18, color: muted, italic: true, fontFace: "Aptos" },
      }),
    ]
  ),
  { frame: { left: 0, top: 0, width: W, height: H }, baseUnit: 8 }
);

const pptxBlob = await PresentationFile.exportPptx(presentation);
await pptxBlob.save("output/ood_wasserstein_formula_slide_formula_fixed.pptx");

const layoutBlob = await slide.export({ format: "layout" });
await writeFile("scratch/ood_wasserstein_formula_slide.layout.json", Buffer.from(await layoutBlob.arrayBuffer()));

const pngBlob = await slide.export({ format: "png" });
await writeFile("scratch/ood_wasserstein_formula_slide.png", Buffer.from(await pngBlob.arrayBuffer()));

console.log("Exported output/ood_wasserstein_formula_slide_formula_fixed.pptx");
console.log("Rendered scratch/ood_wasserstein_formula_slide.png");
console.log("Wrote scratch/ood_wasserstein_formula_slide.layout.json");
