from __future__ import annotations

from pathlib import Path
import re
import tempfile
import zipfile

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPARISON_ROOT = REPO_ROOT / "output" / "top_tail_loco_proximity" / "sensitivity_comparison"
DEFAULT_OUTPUT_NAME = "top_tail_loco_sensitivity_methods_and_figure_notes.docx"


FIGURE_NOTES = [
    (
        "HEA__hea__UTS_MPa",
        "HEA UTS",
        "Main interval: 35% to 40%. The comparison tests whether the Wx jump is mainly caused by samples newly entering the top tail.",
    ),
    (
        "HEA__hea__YS_MPa",
        "HEA YS",
        "Main interval: 20% to 25%. The comparison checks whether removing the new entrants reduces the Wx jump.",
    ),
    (
        "Steel__steel__UTS_MPa",
        "Steel UTS",
        "Main intervals: 10% to 15% and 30% to 35%. Later high-threshold instability is reported but not removed by default.",
    ),
    (
        "MatbenchSteels__matbench_steels_ood__yield_strength",
        "Matbench Steel yield strength",
        "The top 5% has no previous threshold, so the most extreme top-5% X-space samples are used for the removal sensitivity check.",
    ),
]


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else pd.DataFrame()


def set_run_font(run, size: float | None = None, bold: bool | None = None, color: str | None = None) -> None:
    run.font.name = "Calibri"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)


def add_page_number(paragraph) -> None:
    run = paragraph.add_run("Page ")
    set_run_font(run, 9)
    field_run = paragraph.add_run()
    set_run_font(field_run, 9)
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    fld_text = OxmlElement("w:t")
    fld_text.text = "1"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    field_run._r.append(fld_begin)
    field_run._r.append(instr)
    field_run._r.append(fld_sep)
    field_run._r.append(fld_text)
    field_run._r.append(fld_end)


def add_paragraph(doc: Document, text: str = "", style: str | None = None):
    paragraph = doc.add_paragraph(style=style)
    if text:
        run = paragraph.add_run(text)
        set_run_font(run, 10.5)
    paragraph.paragraph_format.space_after = Pt(5)
    paragraph.paragraph_format.line_spacing = 1.12
    return paragraph


def add_heading(doc: Document, text: str, level: int = 1):
    paragraph = doc.add_heading(level=level)
    run = paragraph.add_run(text)
    set_run_font(run, 16 if level == 1 else 12.5, True, "1F4E79" if level == 1 else "000000")
    paragraph.paragraph_format.space_before = Pt(10 if level == 1 else 7)
    paragraph.paragraph_format.space_after = Pt(4)
    return paragraph


def add_bullet(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph(style="List Bullet")
    run = paragraph.add_run(text)
    set_run_font(run, 10.5)
    paragraph.paragraph_format.space_after = Pt(2)


def add_caption(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(text)
    set_run_font(run, 8.8, False, "555555")
    paragraph.paragraph_format.space_after = Pt(8)


def driver_count(drivers: pd.DataFrame, case_key: str) -> int:
    if drivers.empty:
        return 0
    work = drivers[
        drivers["case_key"].eq(case_key)
        & drivers["jump_status"].eq("removable")
        & drivers["driver_action"].eq("drop")
    ]
    return int(work["ID"].nunique()) if not work.empty else 0


def jump_summary(jumps: pd.DataFrame, case_key: str) -> str:
    if jumps.empty:
        return "No jump table was found."
    work = jumps[jumps["case_key"].eq(case_key)].copy()
    if work.empty:
        return "No significant jump was detected for this task."
    parts: list[str] = []
    for row in work.itertuples(index=False):
        parts.append(
            f"{int(row.lower_threshold_pct)}%->{int(row.upper_threshold_pct)}% "
            f"(delta Wx={row.delta_Wx:.3g}, rel={row.relative_delta_Wx:.3g}, {row.jump_status})"
        )
    return "; ".join(parts)


def build_docx(comparison_root: Path = DEFAULT_COMPARISON_ROOT) -> Path:
    figure_dir = comparison_root / "figures"
    csv_dir = comparison_root / "csv"
    output_path = figure_dir / DEFAULT_OUTPUT_NAME
    jumps = read_csv_if_exists(csv_dir / "detected_wx_jumps_5_95.csv")
    drivers = read_csv_if_exists(csv_dir / "detected_jump_driver_ids.csv")

    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.6)
    section.bottom_margin = Inches(0.6)
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_page_number(footer)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Top-tail LOCO Sensitivity Comparison")
    set_run_font(run, 20, True, "1F4E79")
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("Methods and figure notes")
    set_run_font(run, 12.5, False, "555555")

    add_heading(doc, "Purpose", 1)
    add_paragraph(
        doc,
        "These figures test whether abrupt increases in Wx are driven by a small set of newly entering top-tail samples. All removals are in-memory only; the raw data are unchanged.",
    )

    add_heading(doc, "Definitions", 1)
    add_bullet(doc, "Top tail: the highest-property t% samples at each threshold t.")
    add_bullet(doc, "Background: all remaining samples.")
    add_bullet(doc, "X-space: composition plus numeric processing features. For Steel, processing text is not included in Wx.")
    add_bullet(doc, "Y-space: the target property, such as UTS, YS, or yield strength.")

    add_heading(doc, "Metrics", 1)
    add_paragraph(doc, "Wy: standardize y using the background mean and standard deviation, then compute Wasserstein-1 distance between background_y* and top_tail_y*.")
    add_paragraph(doc, "Wx: fit the imputer and scaler on background X only, transform both groups, then compute sliced Wasserstein distance in standardized X-space using 128 projections.")

    add_heading(doc, "Jump-driver removal", 1)
    add_paragraph(
        doc,
        "Adjacent thresholds from 5% to 95% are scanned. A jump is detected when delta Wx >= 0.15 and relative delta >= 30%. If the upper threshold is <=50%, new entrants are removed in the auto-drop curve. If the upper threshold is >=55%, the jump is reported but not removed by default.",
    )

    add_heading(doc, "How to read the figures", 1)
    add_bullet(doc, "Black line: baseline.")
    add_bullet(doc, "Orange line: auto-drop jump drivers.")
    add_bullet(doc, "Top panel: full Wx range.")
    add_bullet(doc, "Bottom panel: zoomed low-Wx range.")
    add_bullet(doc, "Triangles in the zoom panel mark values above the zoom range.")
    add_bullet(doc, "Red bands mark the original jump intervals of interest.")

    add_heading(doc, "Figure notes", 1)
    for key, title_text, note in FIGURE_NOTES:
        add_heading(doc, title_text, 2)
        add_paragraph(doc, note)
        add_paragraph(doc, "Detected jumps: " + jump_summary(jumps, key))
        add_paragraph(doc, f"Removed unique sample IDs: {driver_count(drivers, key)}.")
        image_path = figure_dir / f"{key}.png"
        if image_path.exists():
            doc.add_picture(str(image_path), width=Inches(6.15))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_caption(doc, f"{title_text}: baseline versus auto-drop jump drivers.")
        else:
            add_paragraph(doc, f"Missing figure: {image_path}")

    add_heading(doc, "Output files", 1)
    add_bullet(doc, "figures/*.png: four main comparison figures.")
    add_bullet(doc, "csv/jump_driver_removal_wx.csv: data used for the figures.")
    add_bullet(doc, "csv/detected_wx_jumps_5_95.csv: detected jump intervals.")
    add_bullet(doc, "csv/detected_jump_driver_ids.csv: sample IDs associated with detected jumps.")

    for style_name in ["Normal", "List Bullet"]:
        style = doc.styles[style_name]
        style.font.name = "Calibri"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
        style.font.size = Pt(10.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)
    scrub_docx_xml_to_ascii(output_path)
    return output_path


def scrub_docx_xml_to_ascii(path: Path) -> None:
    """Remove non-English/non-ASCII text from OOXML metadata and theme parts."""
    xml_suffixes = (".xml", ".rels")
    typeface_pattern = re.compile(r'(typeface|w:name|w:ascii|w:hAnsi|w:eastAsia|w:cs)="([^"]*[^\x00-\x7F][^"]*)"')

    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx", dir=path.parent) as tmp_file:
        tmp_path = Path(tmp_file.name)

    with zipfile.ZipFile(path, "r") as src, zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            if info.filename.endswith(xml_suffixes):
                text = data.decode("utf-8", errors="ignore")
                text = typeface_pattern.sub(lambda match: f'{match.group(1)}="Calibri"', text)
                text = text.encode("ascii", errors="ignore").decode("ascii")
                data = text.encode("utf-8")
            dst.writestr(info, data)

    tmp_path.replace(path)


if __name__ == "__main__":
    print(build_docx().resolve())
