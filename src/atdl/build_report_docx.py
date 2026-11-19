"""Render the generated final Markdown report as a formatted DOCX."""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "reports/FINAL_EXPERIMENT_REPORT.md"
OUTPUT = ROOT / "reports/FINAL_EXPERIMENT_REPORT.docx"


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=90, start=100, bottom=90, end=100) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for edge, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{edge}"))
        if node is None:
            node = OxmlElement(f"w:{edge}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table) -> None:
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = OxmlElement(f"w:{edge}")
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "5")
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), "D9E0E8")
        borders.append(element)
    tbl_pr.append(borders)


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)


def set_run_font(run, size=10, bold=None, color="1F2937", name="Aptos") -> None:
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor.from_string(color)
    if bold is not None:
        run.bold = bold


def add_inline(paragraph, text: str, size=10, color="1F2937") -> None:
    # Handle bold, inline code, and Markdown links without leaving syntax markers.
    pattern = re.compile(r"(\*\*.*?\*\*|(?<!\*)\*[^*]+\*(?!\*)|`[^`]+`|\[[^]]+\]\(https?://[^)]+\))")
    cursor = 0
    for match in pattern.finditer(text):
        if match.start() > cursor:
            run = paragraph.add_run(text[cursor:match.start()])
            set_run_font(run, size, color=color)
        token = match.group(0)
        if token.startswith("**"):
            run = paragraph.add_run(token[2:-2])
            set_run_font(run, size, bold=True, color=color)
        elif token.startswith("*"):
            run = paragraph.add_run(token[1:-1])
            set_run_font(run, size, color=color)
            run.italic = True
        elif token.startswith("`"):
            run = paragraph.add_run(token[1:-1])
            set_run_font(run, size - 0.3, color="334155", name="Aptos Mono")
        else:
            label, url = re.match(r"\[([^]]+)\]\((https?://[^)]+)\)", token).groups()
            run = paragraph.add_run(f"{label} ({url})")
            set_run_font(run, size, color="1D4ED8")
        cursor = match.end()
    if cursor < len(text):
        run = paragraph.add_run(text[cursor:])
        set_run_font(run, size, color=color)


def parse_table(lines: list[str]) -> list[list[str]]:
    output = []
    for line in lines:
        if re.fullmatch(r"\|?[\s:|-]+\|?", line.strip()):
            continue
        output.append([value.strip() for value in line.strip().strip("|").split("|")])
    return output


def add_table(document, rows: list[list[str]]) -> None:
    if not rows:
        return
    cols = max(map(len, rows))
    table = document.add_table(rows=len(rows), cols=cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    table.style = "Table Grid"
    table.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
    set_repeat_table_header(table.rows[0])
    for ri, values in enumerate(rows):
        row = table.rows[ri]
        for ci in range(cols):
            cell = row.cells[ci]
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)
            cell.text = ""
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_before = Pt(1)
            paragraph.paragraph_format.space_after = Pt(1)
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if ci == 0 else WD_ALIGN_PARAGRAPH.CENTER
            value = values[ci] if ci < len(values) else ""
            add_inline(paragraph, value, size=7.7 if cols >= 8 else (8.2 if cols >= 6 else 9.0), color="FFFFFF" if ri == 0 else "1F2937")
            if ri == 0:
                set_cell_shading(cell, "1F3B64")
                for run in paragraph.runs:
                    run.bold = True
            elif ri % 2 == 0:
                set_cell_shading(cell, "F3F6FA")
    set_table_borders(table)
    document.add_paragraph().paragraph_format.space_after = Pt(2)


def add_page_field(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("Page ")
    set_run_font(run, 8, color="64748B")
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    paragraph._p.append(fld)


def build() -> Path:
    markdown = SOURCE.read_text(encoding="utf-8").splitlines()
    document = Document()
    section = document.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.68)
    section.bottom_margin = Inches(0.68)
    section.left_margin = Inches(0.78)
    section.right_margin = Inches(0.78)
    section.header_distance = Inches(0.35)
    section.footer_distance = Inches(0.35)

    styles = document.styles
    normal = styles["Normal"]
    normal.font.name = "Aptos"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Aptos")
    normal.font.size = Pt(10.2)
    normal.font.color.rgb = RGBColor.from_string("1F2937")
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.12
    for name, size in (("Heading 1", 17), ("Heading 2", 13.5), ("Heading 3", 11.5)):
        style = styles[name]
        style.font.name = "Aptos Display"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Aptos Display")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_before = Pt(11)
        style.paragraph_format.space_after = Pt(4)
        style.paragraph_format.keep_with_next = True

    header = section.header.paragraphs[0]
    add_inline(header, "ATDL  |  NEU-DET steel surface defect detection", size=8, color="64748B")
    add_page_field(section.footer.paragraphs[0])

    i = 0
    first_title = True
    while i < len(markdown):
        line = markdown[i].strip()
        if not line or line == "---":
            i += 1
            continue
        if line.startswith("|"):
            table_lines = []
            while i < len(markdown) and markdown[i].strip().startswith("|"):
                table_lines.append(markdown[i].strip())
                i += 1
            add_table(document, parse_table(table_lines))
            continue
        image_match = re.match(r"!\[([^]]*)\]\(([^)]+)\)", line)
        if image_match:
            alt, relative = image_match.groups()
            image_path = (SOURCE.parent / relative).resolve()
            if image_path.is_file():
                paragraph = document.add_paragraph()
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                paragraph.paragraph_format.space_before = Pt(4)
                paragraph.paragraph_format.space_after = Pt(2)
                run = paragraph.add_run()
                run.add_picture(str(image_path), width=Inches(6.55))
                caption = document.add_paragraph()
                caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
                caption.paragraph_format.space_after = Pt(8)
                r = caption.add_run(alt)
                set_run_font(r, 8.5, color="64748B")
            i += 1
            continue
        heading = re.match(r"^(#{1,3})\s+(.*)$", line)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2)
            if level == 1 and first_title:
                p = document.add_paragraph(style="Title")
                p.paragraph_format.space_after = Pt(8)
                run = p.add_run(title)
                set_run_font(run, 23, bold=True, color="000000", name="Aptos Display")
                first_title = False
            else:
                p = document.add_paragraph(style=f"Heading {min(level,3)}")
                add_inline(p, title, size={1:17,2:13.5,3:11.5}[min(level,3)], color="000000")
            i += 1
            continue
        if line.startswith("``` ") or line == "```":
            i += 1
            block = []
            while i < len(markdown) and markdown[i].strip() != "```":
                block.append(markdown[i])
                i += 1
            p = document.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.15)
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(7)
            for j, content in enumerate(block):
                if j:
                    p.add_run().add_break()
                run = p.add_run(content)
                set_run_font(run, 8.5, color="334155", name="Aptos Mono")
            i += 1
            continue
        if line.startswith("- "):
            p = document.add_paragraph(style="List Bullet")
            p.paragraph_format.space_after = Pt(3)
            add_inline(p, line[2:], size=9.8)
            i += 1
            continue
        numbered = re.match(r"^\d+\.\s+(.*)$", line)
        if numbered:
            p = document.add_paragraph(style="List Number")
            p.paragraph_format.space_after = Pt(3)
            add_inline(p, numbered.group(1), size=9.8)
            i += 1
            continue
        p = document.add_paragraph()
        p.paragraph_format.widow_control = True
        add_inline(p, line, size=10.2)
        i += 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build())
