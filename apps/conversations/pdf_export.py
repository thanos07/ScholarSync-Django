import re
from html import escape
from io import BytesIO

from django.utils import timezone
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    CondPageBreak,
    HRFlowable,
    ListFlowable,
    ListItem,
    LongTable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PLUM = colors.HexColor("#5A304D")
STONE = colors.HexColor("#F3F1F2")
INK = colors.HexColor("#292327")
MUTED = colors.HexColor("#6B6268")
BORDER = colors.HexColor("#D8D1D6")
FORMULA_BG = colors.HexColor("#F7F5F6")

SUBSCRIPT_MAP = {
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "ₐ": "a", "ₑ": "e", "ₕ": "h", "ᵢ": "i", "ⱼ": "j",
    "ₖ": "k", "ₗ": "l", "ₘ": "m", "ₙ": "n", "ₒ": "o",
    "ₚ": "p", "ᵣ": "r", "ₛ": "s", "ₜ": "t", "ᵤ": "u",
    "ᵥ": "v", "ₓ": "x",
}
SUPERSCRIPT_MAP = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-", "⁼": "=", "⁽": "(", "⁾": ")",
    "ᵀ": "T", "ᵃ": "a", "ᵇ": "b", "ᶜ": "c", "ᵈ": "d",
    "ᵉ": "e", "ᶠ": "f", "ᵍ": "g", "ʰ": "h", "ⁱ": "i",
    "ʲ": "j", "ᵏ": "k", "ˡ": "l", "ᵐ": "m", "ⁿ": "n",
    "ᵒ": "o", "ᵖ": "p", "ʳ": "r", "ˢ": "s", "ᵗ": "t",
    "ᵘ": "u", "ᵛ": "v", "ʷ": "w", "ˣ": "x", "ʸ": "y", "ᶻ": "z",
}


def _replace_unicode_scripts(text):
    """Convert Unicode sub/superscripts into PDF-safe ASCII notation."""
    output = []
    index = 0
    while index < len(text):
        char = text[index]
        mapping = SUBSCRIPT_MAP if char in SUBSCRIPT_MAP else SUPERSCRIPT_MAP if char in SUPERSCRIPT_MAP else None
        if mapping is None:
            output.append(char)
            index += 1
            continue
        source_map = SUBSCRIPT_MAP if char in SUBSCRIPT_MAP else SUPERSCRIPT_MAP
        marker = "_" if source_map is SUBSCRIPT_MAP else "^"
        values = []
        while index < len(text) and text[index] in source_map:
            values.append(source_map[text[index]])
            index += 1
        output.append(f"{marker}({''.join(values)})")
    return ''.join(output)


def _pdf_safe_text(value):
    raw = str(value or "")
    # Common model notation sometimes arrives as Unicode subscript letters.
    raw = (
        raw.replace("dₘₒdₑₗ", "d_model")
        .replace("dₘₒdₑₗ", "d_model")
        .replace("QKᵀ", "QK^T")
    )
    text = _replace_unicode_scripts(raw)
    replacements = {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00ad": "",
        "\u2192": "->",
        "\u2190": "<-",
        "\u00d7": "x",
        "\u2022": "-",
        "\u00a0": " ",
        "\u221e": "infinity",
        "\u2211": "sum",
        "\u2208": "in",
        "\u03b5": "epsilon",
        "\u03b2": "beta",
        "\u03b1": "alpha",
        "\u03c3": "sigma",
        "\u221a": "sqrt",
        "\u00b7": "*",
        "\u2264": "<=",
        "\u2265": ">=",
        "\u2248": "~=",
        "\u2260": "!=",
        "\u221d": "proportional to",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _latex_to_ascii(value):
    """Convert common research-paper LaTeX into readable, font-safe text."""
    text = _pdf_safe_text(value).strip()
    text = text.replace("\\[", "").replace("\\]", "")
    text = text.replace("\\(", "").replace("\\)", "")
    text = text.replace("$$", "").replace("$", "")

    # Resolve common braced commands repeatedly so nested simple expressions
    # such as \frac{QK^{T}}{\sqrt{d_k}} remain readable in built-in PDF fonts.
    for _ in range(6):
        before = text
        text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", text)
        text = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", text)
        text = re.sub(r"\\(?:operatorname|mathrm|text)\{([^{}]+)\}", r"\1", text)
        text = re.sub(r"_\{([^{}]+)\}", r"_(\1)", text)
        text = re.sub(r"\^\{([^{}]+)\}", r"^(\1)", text)
        if text == before:
            break

    commands = {
        r"\operatorname": "",
        r"\mathrm": "",
        r"\text": "",
        r"\softmax": "softmax",
        r"\max": "max",
        r"\min": "min",
        r"\sin": "sin",
        r"\cos": "cos",
        r"\exp": "exp",
        r"\log": "log",
        r"\sum": "sum",
        r"\cdot": " * ",
        r"\times": " x ",
        r"\ldots": "...",
        r"\dots": "...",
        r"\infty": "infinity",
        r"\beta": "beta",
        r"\alpha": "alpha",
        r"\epsilon": "epsilon",
        r"\varepsilon": "epsilon",
        r"\sigma": "sigma",
        r"\top": "T",
        r"\mathbf": "",
        r"\mathit": "",
        r"\mathcal": "",
        r"\left": "",
        r"\right": "",
        r"\,": " ",
        r"\;": " ",
        r"\!": "",
        r"\\": " ",
    }
    for command, replacement in commands.items():
        text = text.replace(command, replacement)

    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"_\(([A-Za-z0-9]+)\)", r"_\1", text)
    text = re.sub(r"\^\(([A-Za-z0-9]+)\)", r"^\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _inline_markup(value):
    """Convert a safe subset of Markdown and inline math to ReportLab markup."""
    raw = _pdf_safe_text(value)
    protected = []

    def stash(content, font="Courier"):
        token = f"ZZSCHOLARFORMULA{len(protected)}ZZ"
        protected.append((token, f"<font name='{font}'>{escape(content)}</font>"))
        return token

    # Protect complete math/code spans first. Display math can occasionally
    # arrive inside a Markdown table cell; convert it to readable PDF-safe
    # notation instead of printing the raw $$...$$ source.
    raw = re.sub(r"\$\$([\s\S]+?)\$\$", lambda match: stash(_latex_to_ascii(match.group(1))), raw)
    raw = re.sub(r"\\\[([\s\S]+?)\\\]", lambda match: stash(_latex_to_ascii(match.group(1))), raw)
    raw = re.sub(r"\\\((.+?)\\\)", lambda match: stash(_latex_to_ascii(match.group(0))), raw)
    raw = re.sub(r"(?<!\$)\$([^$\n]+?)\$(?!\$)", lambda match: stash(_latex_to_ascii(match.group(0))), raw)
    raw = re.sub(r"`([^`]+)`", lambda match: stash(match.group(1)), raw)

    # Protect PDF-safe sub/superscript notation that appears in normal prose.
    raw = re.sub(
        r"\b([A-Za-z][A-Za-z0-9]*)_([A-Za-z0-9]+)\b",
        lambda match: stash(f"{match.group(1)}_{match.group(2)}"),
        raw,
    )
    raw = re.sub(
        r"([A-Za-z0-9])_\(([^()]+)\)",
        lambda match: stash(f"{match.group(1)}_({match.group(2)})"),
        raw,
    )
    raw = re.sub(
        r"([A-Za-z0-9])\^\(([^()]+)\)",
        lambda match: stash(f"{match.group(1)}^({match.group(2)})"),
        raw,
    )

    text = escape(raw)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)
    text = re.sub(r"(^|[^*])\*([^*\n]+?)\*(?!\*)", r"\1<i>\2</i>", text)
    text = re.sub(r"(^|[^_])_([^_\n]+?)_(?!_)", r"\1<i>\2</i>", text)

    for token, markup in protected:
        text = text.replace(escape(token), markup).replace(token, markup)
    return text


def _table_cells(line):
    return [
        cell.strip()
        for cell in line.strip().strip("|").split("|")
    ]


def _is_table_divider(line):
    cells = _table_cells(line)
    return len(cells) > 1 and all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells)


def _markdown_flowables(value, styles, available_width):
    lines = str(value or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    flowables = []
    index = 0

    while index < len(lines):
        raw = lines[index]
        line = raw.strip()
        if not line:
            flowables.append(Spacer(1, 3))
            index += 1
            continue

        # Display formulas can span several Markdown lines.
        if line.startswith("$$") or line.startswith("\\["):
            dollar = line.startswith("$$")
            opening = "$$" if dollar else "\\["
            closing = "$$" if dollar else "\\]"
            formula = line[len(opening):]
            closed = formula.endswith(closing) and len(formula) > len(closing)
            if closed:
                formula = formula[:-len(closing)]
            while not closed and index + 1 < len(lines):
                index += 1
                next_line = lines[index].strip()
                if next_line.endswith(closing):
                    formula += " " + next_line[:-len(closing)]
                    closed = True
                else:
                    formula += " " + next_line
            safe_formula = escape(_latex_to_ascii(formula))
            flowables.append(Paragraph(safe_formula, styles["Formula"]))
            flowables.append(Spacer(1, 5))
            index += 1
            continue

        # Standard Markdown pipe table.
        next_line = lines[index + 1].strip() if index + 1 < len(lines) else ""
        if "|" in line and _is_table_divider(next_line):
            headers = _table_cells(line)
            rows = []
            index += 2
            while index < len(lines):
                row_line = lines[index].strip()
                if not row_line or "|" not in row_line:
                    break
                rows.append(_table_cells(row_line))
                index += 1

            column_count = max(1, len(headers))
            normalized_headers = [header.strip().lower() for header in headers]
            formula_table = "formula" in normalized_headers and "symbols" in normalized_headers
            if formula_table and column_count == 4:
                col_widths = [available_width * 0.07, available_width * 0.45, available_width * 0.37, available_width * 0.11]
            else:
                col_widths = [available_width / column_count] * column_count
            table_data = [
                [Paragraph(_inline_markup(cell), styles["TableHeader"]) for cell in headers]
            ]
            for row in rows:
                table_data.append(
                    [
                        Paragraph(
                            _inline_markup(row[cell_index] if cell_index < len(row) else ""),
                            styles["TableCell"],
                        )
                        for cell_index in range(column_count)
                    ]
                )

            table = LongTable(
                table_data,
                colWidths=col_widths,
                repeatRows=1,
                hAlign="LEFT",
                splitByRow=1,
            )
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), STONE),
                        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
                        ("GRID", (0, 0), (-1, -1), 0.45, BORDER),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 5),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FBFAFB")]),
                    ]
                )
            )
            flowables.extend([table, Spacer(1, 8)])
            continue

        heading = re.match(r"^(#{1,4})\s+(.+)$", line)
        if heading:
            level = min(4, len(heading.group(1)))
            flowables.append(Paragraph(_inline_markup(heading.group(2)), styles[f"MarkdownH{level}"]))
            index += 1
            continue

        list_match = re.match(r"^(?:[-*]|\u2022)\s+(.+)$", line)
        numbered_match = re.match(r"^(\d+)[.)]\s+(.+)$", line)
        if list_match or numbered_match:
            ordered = bool(numbered_match)
            contents = []
            start_number = int(numbered_match.group(1)) if numbered_match else 1
            while index < len(lines):
                current = lines[index].strip()
                match = re.match(r"^(\d+)[.)]\s+(.+)$", current) if ordered else re.match(r"^(?:[-*]|\u2022)\s+(.+)$", current)
                if not match:
                    break
                contents.append(match.group(2) if ordered else match.group(1))
                index += 1

            # Models sometimes place a single ``1.`` before an ordinary
            # paragraph under every heading. Rendering that as a numbered list
            # produces a distracting column of repeated ones in the PDF.
            if ordered and len(contents) == 1:
                flowables.append(Paragraph(_inline_markup(contents[0]), styles["Body"]))
                continue

            items = [
                ListItem(Paragraph(_inline_markup(content), styles["Body"]), leftIndent=12)
                for content in contents
            ]
            flowables.append(
                ListFlowable(
                    items,
                    bulletType="1" if ordered else "bullet",
                    start=str(start_number),
                    leftIndent=18,
                    bulletFontName="Helvetica",
                    bulletFontSize=8.5,
                    spaceAfter=6,
                )
            )
            continue

        if line.startswith(">"):
            flowables.append(Paragraph(_inline_markup(line.lstrip("> ")), styles["Quote"]))
            index += 1
            continue

        flowables.append(Paragraph(_inline_markup(line), styles["Body"]))
        index += 1

    return flowables


def _format_generated(value):
    value = value or timezone.now()
    if timezone.is_naive(value):
        value = timezone.make_aware(value, timezone.get_current_timezone())
    local_value = timezone.localtime(value)
    zone = local_value.tzname() or "UTC"
    return f"{local_value.strftime('%d %B %Y, %I:%M %p')} {zone}"


def _draw_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawCentredString(A4[0] / 2, 10 * mm, f"ScholarSync - Page {doc.page}")
    canvas.restoreState()


def build_conversation_pdf(conversation, messages):
    output = BytesIO()
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(name="Brand", parent=styles["Title"], textColor=PLUM, fontSize=22, leading=26, alignment=TA_CENTER, spaceAfter=4))
    styles.add(ParagraphStyle(name="Meta", parent=styles["Normal"], textColor=MUTED, fontSize=9, leading=13, alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="Role", parent=styles["Normal"], fontName="Helvetica-Bold", textColor=PLUM, fontSize=10.5, leading=13, spaceBefore=10, spaceAfter=7, keepWithNext=True))
    styles.add(ParagraphStyle(name="UserMessage", parent=styles["BodyText"], textColor=INK, fontSize=10, leading=15, backColor=STONE, borderColor=BORDER, borderWidth=0.7, borderPadding=10, spaceAfter=8, splitLongWords=True))
    styles.add(ParagraphStyle(name="Body", parent=styles["BodyText"], textColor=INK, fontSize=9.6, leading=14.4, spaceAfter=6, splitLongWords=True))
    styles.add(ParagraphStyle(name="Formula", parent=styles["Code"], fontName="Courier", textColor=INK, fontSize=8.6, leading=12.5, backColor=FORMULA_BG, borderColor=BORDER, borderWidth=0.6, borderPadding=8, leftIndent=3, rightIndent=3, alignment=TA_LEFT, splitLongWords=True))
    styles.add(ParagraphStyle(name="Quote", parent=styles["Body"], textColor=MUTED, leftIndent=10, borderColor=PLUM, borderWidth=1.3, borderPadding=7, backColor=STONE))
    styles.add(ParagraphStyle(name="SourceHeading", parent=styles["Normal"], fontName="Helvetica-Bold", textColor=PLUM, fontSize=9.2, leading=12, spaceBefore=7, spaceAfter=4, keepWithNext=True))
    styles.add(ParagraphStyle(name="SourceLine", parent=styles["Normal"], textColor=MUTED, fontSize=8.3, leading=11.5, leftIndent=8, spaceAfter=2.5))
    styles.add(ParagraphStyle(name="TableHeader", parent=styles["Normal"], fontName="Helvetica-Bold", textColor=PLUM, fontSize=7.1, leading=9.4, splitLongWords=True))
    styles.add(ParagraphStyle(name="TableCell", parent=styles["Normal"], textColor=INK, fontSize=6.9, leading=9.2, splitLongWords=True))
    for level, size in ((1, 14), (2, 12.2), (3, 11), (4, 10.2)):
        styles.add(ParagraphStyle(name=f"MarkdownH{level}", parent=styles["Normal"], fontName="Helvetica-Bold", textColor=INK, fontSize=size, leading=size + 3, spaceBefore=8, spaceAfter=5, keepWithNext=True))

    doc = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=conversation.title,
        allowSplitting=True,
    )
    available_width = A4[0] - doc.leftMargin - doc.rightMargin

    story = [
        Paragraph("ScholarSync", styles["Brand"]),
        Paragraph("Evidence-grounded research conversation", styles["Meta"]),
        Spacer(1, 10),
    ]
    meta = [
        ["Conversation", escape(_pdf_safe_text(conversation.title))],
        ["Workspace", escape(_pdf_safe_text(conversation.workspace.name))],
        ["Generated", _format_generated(getattr(conversation, "updated_at", None))],
    ]
    meta_table = Table(meta, colWidths=[30 * mm, 125 * mm])
    meta_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), STONE),
                ("TEXTCOLOR", (0, 0), (-1, -1), INK),
                ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("PADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.extend([meta_table, Spacer(1, 14)])

    for message in messages:
        role = "User" if message.role == "USER" else "ScholarSync"
        story.append(CondPageBreak(25 * mm))
        story.append(Paragraph(role, styles["Role"]))

        if role == "User":
            story.append(Paragraph(_inline_markup(message.content), styles["UserMessage"]))
        else:
            story.extend(_markdown_flowables(message.content, styles, available_width))

        citations = list(message.citations.select_related("document").all())
        if citations:
            story.append(Paragraph("Sources", styles["SourceHeading"]))
            for citation in citations:
                story.append(
                    Paragraph(
                        f"[{citation.citation_number}] "
                        f"{escape(_pdf_safe_text(citation.document.display_title))}, "
                        f"page {citation.page_number}",
                        styles["SourceLine"],
                    )
                )

        story.extend([
            Spacer(1, 7),
            HRFlowable(width="100%", thickness=0.4, color=BORDER),
            Spacer(1, 5),
        ])

    story.extend([
        Spacer(1, 10),
        Paragraph(
            "This report was generated from selected documents. "
            "Check AI-generated interpretations against the cited sources.",
            styles["Meta"],
        ),
    ])
    doc.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)
    return output.getvalue()
