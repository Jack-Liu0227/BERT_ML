import path from "node:path";

const C = {
  ink: "#172033",
  muted: "#5C6B82",
  paper: "#FBFCFE",
  panel: "#FFFFFF",
  faint: "#E6ECF3",
  soft: "#F4F7FB",
  blue: "#2F6FED",
  teal: "#14A3A1",
  orange: "#E76F2E",
  purple: "#7C3AED",
  green: "#2F9D66",
  gray: "#9AA7B8",
};

function addText(slide, ctx, text, x, y, w, h, options = {}) {
  return ctx.addText(slide, {
    text,
    x,
    y,
    w,
    h,
    fontSize: options.size ?? 18,
    color: options.color ?? C.ink,
    bold: options.bold ?? false,
    typeface: options.face ?? ctx.fonts.body,
    align: options.align ?? "left",
    valign: options.valign ?? "top",
    fill: options.fill ?? { color: "transparent", transparency: 100 },
    line: options.line ?? { fill: { color: "transparent", transparency: 100 }, width: 0 },
    insets: options.insets ?? { left: 0, right: 0, top: 0, bottom: 0 },
  });
}

function addBox(slide, ctx, x, y, w, h, fill = C.panel, stroke = C.faint, geometry = "roundRect") {
  return ctx.addShape(slide, {
    geometry,
    x,
    y,
    w,
    h,
    fill: { color: fill },
    line: { fill: { color: stroke }, width: 1 },
  });
}

function addLine(slide, x1, y1, x2, y2, color = C.gray, width = 2) {
  return slide.shapes.add({
    geometry: "line",
    position: { x: x1, y: y1, w: x2 - x1, h: y2 - y1 },
    fill: { color: "transparent", transparency: 100 },
    line: { fill: { color }, width },
  });
}

async function addFormulaImage(slide, ctx, filename, x, y, w, h, alt) {
  return ctx.addImage(slide, {
    path: path.join(ctx.assetDir, "formulas", filename),
    x,
    y,
    w,
    h,
    fit: "contain",
    alt: alt ?? filename,
  });
}

async function addStep(slide, ctx, index, label, formulaImage, x, y, w, color) {
  addBox(slide, ctx, x, y, w, 132, C.panel, C.faint);
  addBox(slide, ctx, x + 14, y + 14, 30, 30, color, color);
  addText(slide, ctx, String(index), x + 14, y + 20, 30, 16, {
    size: 12,
    bold: true,
    color: "#FFFFFF",
    align: "center",
  });
  addText(slide, ctx, label, x + 54, y + 13, w - 68, 22, { size: 15, bold: true });
  addBox(slide, ctx, x + 18, y + 55, w - 36, 58, C.soft, "#DDE5EF");
  await addFormulaImage(slide, ctx, formulaImage, x + 24, y + 61, w - 48, 46, label);
}

async function addFormulaPanel(slide, ctx, title, formulaImage, note, x, y, w, h, color) {
  addBox(slide, ctx, x, y, w, h, C.panel, C.faint);
  addText(slide, ctx, title, x + 18, y + 15, w - 36, 22, { size: 15, bold: true, color });
  addBox(slide, ctx, x + 18, y + 48, w - 36, 44, C.soft, "#DDE5EF");
  await addFormulaImage(slide, ctx, formulaImage, x + 30, y + 53, w - 60, 34, title);
  addText(slide, ctx, note, x + 18, y + 103, w - 36, h - 112, { size: 11.5, color: C.muted });
}

export async function addSlide(presentation, ctx) {
  const slide = presentation.slides.add();
  ctx.addShape(slide, {
    geometry: "rect",
    x: 0,
    y: 0,
    w: 1280,
    h: 720,
    fill: { color: C.paper },
    line: { fill: { color: "transparent", transparency: 100 }, width: 0 },
  });
  ctx.addShape(slide, {
    geometry: "rect",
    x: 0,
    y: 0,
    w: 1280,
    h: 8,
    fill: { color: C.blue },
    line: { fill: { color: C.blue }, width: 0 },
  });

  addText(slide, ctx, "X-space sample-level sliced Wasserstein contribution", 44, 30, 780, 36, {
    size: 26,
    bold: true,
    face: ctx.fonts.title,
  });
  addText(
    slide,
    ctx,
    "Goal: rank which test samples contribute most to the composition + processing input-space OOD shift.",
    46,
    68,
    920,
    24,
    { size: 13.5, color: C.muted },
  );
  addBox(slide, ctx, 976, 30, 120, 32, "#FFFFFF", C.green);
  addText(slide, ctx, "train-only fit", 986, 38, 100, 16, { size: 11.5, bold: true, color: C.green, align: "center" });
  addBox(slide, ctx, 1112, 30, 116, 32, "#FFFFFF", C.orange);
  addText(slide, ctx, "no leakage", 1124, 38, 92, 16, { size: 11.5, bold: true, color: C.orange, align: "center" });

  const topY = 118;
  const stepW = 216;
  const stepGap = 24;
  const xs = [44, 44 + stepW + stepGap, 44 + 2 * (stepW + stepGap), 44 + 3 * (stepW + stepGap), 44 + 4 * (stepW + stepGap)];
  await addStep(slide, ctx, 1, "Select X features", "step_vector.png", xs[0], topY, stepW, C.blue);
  await addStep(slide, ctx, 2, "Zero-fill and scale", "step_scale.png", xs[1], topY, stepW, C.teal);
  await addStep(slide, ctx, 3, "Draw directions", "step_theta.png", xs[2], topY, stepW, C.orange);
  await addStep(slide, ctx, 4, "Project to 1D", "step_projection.png", xs[3], topY, stepW, C.purple);
  await addStep(slide, ctx, 5, "Average sample W", "step_sample_w.png", xs[4], topY, stepW, C.green);
  for (let i = 0; i < 4; i += 1) {
    addLine(slide, xs[i] + stepW + 5, topY + 59, xs[i + 1] - 8, topY + 59, "#AAB4C3", 2);
    addText(slide, ctx, ">", xs[i] + stepW + 12, topY + 48, 16, 18, { size: 17, bold: true, color: "#AAB4C3" });
  }

  addText(slide, ctx, "Mathematical definition", 44, 276, 360, 24, { size: 18, bold: true });
  addText(slide, ctx, "All statistics below are fitted on the training split only.", 286, 281, 420, 18, {
    size: 11.5,
    color: C.muted,
  });

  await addFormulaPanel(
    slide,
    ctx,
    "1. Zero-coded absence",
    "missing.png",
    "Zeros in composition and processing columns are kept as physical absence; rare blanks are filled with 0.",
    44,
    318,
    384,
    126,
    C.teal,
  );
  await addFormulaPanel(
    slide,
    ctx,
    "2. Standard scaling",
    "scaling.png",
    "Standard scaler puts composition and processing variables on comparable units.",
    448,
    318,
    384,
    126,
    C.blue,
  );
  await addFormulaPanel(
    slide,
    ctx,
    "3. One-dimensional matching",
    "matching.png",
    "For each random direction, projected train and test values are sorted and matched by W1 transport.",
    852,
    318,
    384,
    126,
    C.purple,
  );

  addBox(slide, ctx, 44, 472, 780, 108, "#FFFFFF", C.faint);
  addText(slide, ctx, "Final sample-level contribution", 66, 492, 280, 22, { size: 16, bold: true, color: C.green });
  addBox(slide, ctx, 66, 522, 700, 46, C.soft, "#DDE5EF");
  await addFormulaImage(slide, ctx, "final.png", 78, 527, 676, 36, "Final sample-level contribution");

  addBox(slide, ctx, 852, 472, 384, 108, "#FFFFFF", C.faint);
  addText(slide, ctx, "Projection details", 874, 489, 180, 22, { size: 15.5, bold: true, color: C.orange });
  addBox(slide, ctx, 874, 512, 330, 40, C.soft, "#DDE5EF");
  await addFormulaImage(slide, ctx, "projection_sum.png", 884, 517, 310, 30, "Projection dot product");
  addText(
    slide,
    ctx,
    "Use the same direction for train and test. The dot product converts each d-dimensional X vector into one scalar coordinate for W1 matching.",
    874,
    556,
    330,
    28,
    { size: 10.5, color: C.ink },
  );

  addBox(slide, ctx, 44, 602, 1192, 70, C.soft, "#DDE5EF");
  addText(slide, ctx, "Reporting language", 66, 615, 160, 18, { size: 13.2, bold: true, color: C.blue });
  addText(
    slide,
    ctx,
    "Use \"sample-level contribution to sliced Wasserstein distance\", not \"single-sample Wasserstein distance\". Vector length d is determined by the usable X-space feature columns.",
    226,
    615,
    970,
    22,
    { size: 11.5, color: C.ink },
  );
  addText(slide, ctx, "Imputation note", 66, 645, 160, 18, { size: 13.2, bold: true, color: C.teal });
  addText(
    slide,
    ctx,
    "Raw 0 is used because it has physical meaning here: no element or no process. Only rare NaNs/blanks are filled with 0 before train-fitted standard scaling.",
    226,
    645,
    970,
    20,
    { size: 11.5, color: C.ink },
  );

  addText(slide, ctx, "BERT_ML OOD analysis", 1048, 682, 188, 18, { size: 11.5, color: C.muted, align: "right" });
  return slide;
}
