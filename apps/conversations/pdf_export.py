import re
from html import escape
from io import BytesIO
from pathlib import Path

import reportlab
from django.utils import timezone
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
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

# ScholarSync editorial PDF palette.
PLUM = colors.HexColor("#641328")
PLUM_DARK = colors.HexColor("#48101E")

STONE = colors.HexColor("#FBF2E9")
SURFACE = colors.HexColor("#FFFDF9")

ROSE = colors.HexColor("#F2DFE4")
ROSE_SOFT = colors.HexColor("#FBF0F3")

GOLD = colors.HexColor("#BD8717")
GOLD_SOFT = colors.HexColor("#F4E5BF")

INK = colors.HexColor("#332725")
MUTED = colors.HexColor("#786D66")
BORDER = colors.HexColor("#E3CFBD")

SUCCESS_BG = colors.HexColor("#E4EFE5")
SUCCESS = colors.HexColor("#3D6547")

PARTIAL_BG = colors.HexColor("#F6E8BF")
PARTIAL = colors.HexColor("#846116")

REVIEW_BG = colors.HexColor("#F4DDDD")
REVIEW = colors.HexColor("#8E3E47")

FORMULA_BG = colors.HexColor("#FBF2E9")


def _register_formula_font():
    name = "ScholarSyncFormula"
    try:
        pdfmetrics.getFont(name)
        return name
    except KeyError:
        pass
    try:
        font_path = Path(reportlab.__file__).resolve().parent / "fonts" / "Vera.ttf"
        if font_path.exists():
            pdfmetrics.registerFont(TTFont(name, str(font_path)))
            return name
    except Exception:
        pass
    return "Helvetica"


FORMULA_FONT = _register_formula_font()

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


def _verification_presentation(status):
    """Return label and colors for a citation verification status."""
    status = str(status or "").upper()

    mapping = {
        "SUPPORTED": ("Strong evidence match", SUCCESS_BG, "#3D6547"),
        "PARTIAL": ("Partial evidence match", PARTIAL_BG, "#846116"),
        "REVIEW": ("Needs review", REVIEW_BG, "#8E3E47"),
        "VALID": ("Supported", SUCCESS_BG, "#3D6547"),
        "UNCHECKED": ("Not checked", STONE, "#786D66"),
    }

    return mapping.get(
        status,
        ("Not checked", STONE, "#786D66"),
    )


def _truncate_pdf_excerpt(value, limit=300):
    """Keep evidence cards compact without changing stored evidence."""
    text = _pdf_safe_text(value).strip()

    if len(text) <= limit:
        return text

    return text[: limit - 3].rstrip() + "..."


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




def _read_latex_group(text, start):
    # Return (group_content, next_index) for a balanced {...} group.
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    index = start
    content_start = start + 1

    while index < len(text):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:index], index + 1
        index += 1
    return None


def _expand_latex_fractions(text):
    # Convert \frac{...}{...} using balanced braces, including nested scripts.
    value = str(text or "")
    output = []
    index = 0
    marker = r"\frac"

    while index < len(value):
        if not value.startswith(marker, index):
            output.append(value[index])
            index += 1
            continue

        cursor = index + len(marker)
        while cursor < len(value) and value[cursor].isspace():
            cursor += 1

        numerator = _read_latex_group(value, cursor)
        if numerator is None:
            output.append(marker)
            index += len(marker)
            continue

        numerator_text, cursor = numerator
        while cursor < len(value) and value[cursor].isspace():
            cursor += 1

        denominator = _read_latex_group(value, cursor)
        if denominator is None:
            output.append(marker)
            index += len(marker)
            continue

        denominator_text, next_index = denominator
        numerator_text = _expand_latex_fractions(numerator_text)
        denominator_text = _expand_latex_fractions(denominator_text)
        output.append(f"({numerator_text}) / ({denominator_text})")
        index = next_index

    return "".join(output)


def _split_latex_environment_rows(body):
    """Split a LaTeX matrix/cases body on real ``\\`` row separators."""
    rows = []
    for row in re.split(r"\\\\", str(body or "")):
        row = re.sub(r"\s+", " ", row).strip()
        if row:
            rows.append(row)
    return rows


def _expand_latex_environments(text):
    """Convert unsupported LaTeX layout environments to readable PDF text.

    Browser KaTeX can render environments such as ``bmatrix`` and ``cases``.
    ReportLab Paragraph cannot, so preserve their visual row structure as
    multiline text before the normal LaTeX-to-ReportLab conversion runs.
    """
    value = str(text or "")

    matrix_re = re.compile(
        r"\\begin\{(?P<env>bmatrix|pmatrix|matrix)\}"
        r"(?P<body>.*?)"
        r"\\end\{(?P=env)\}",
        re.S,
    )

    def matrix_repl(match):
        env = match.group("env")
        rows = []

        for row in _split_latex_environment_rows(match.group("body")):
            cells = [cell.strip() for cell in row.split("&")]
            rows.append(" | ".join(cells))

        if not rows:
            return ""

        if env == "bmatrix":
            rows[0] = "[ " + rows[0]
            rows[-1] = rows[-1] + " ]"
        elif env == "pmatrix":
            rows[0] = "( " + rows[0]
            rows[-1] = rows[-1] + " )"

        return "\n".join(rows)

    value = matrix_re.sub(matrix_repl, value)

    cases_re = re.compile(
        r"\\begin\{cases\}(?P<body>.*?)\\end\{cases\}",
        re.S,
    )

    def cases_repl(match):
        rows = []

        for row in _split_latex_environment_rows(match.group("body")):
            parts = row.split("&", 1)
            expression = parts[0].strip().rstrip(",")

            if len(parts) == 2:
                condition = parts[1].strip().lstrip(",")
                rows.append(
                    f"{expression}, if {condition}"
                    if condition
                    else expression
                )
            else:
                rows.append(expression)

        if not rows:
            return ""

        # Use sentinels because the generic cleanup later removes LaTeX braces.
        rows[0] = "ZZSCHOLARSYNCCASEOPENZZ " + rows[0]
        rows[-1] = rows[-1] + " ZZSCHOLARSYNCCASECLOSEZZ"

        return "\n".join(rows)

    return cases_re.sub(cases_repl, value)


def _latex_to_pdf_markup(value):
    """Convert formula LaTeX into readable ReportLab Paragraph markup."""
    text = str(value or "").strip()
    text = text.replace(r"\[", "").replace(r"\]", "")
    text = text.replace(r"\(", "").replace(r"\)", "")
    text = text.replace("$$", "").replace("$", "")

    # ReportLab does not understand LaTeX layout environments. Expand them
    # before command replacement so matrices and piecewise cases stay readable.
    text = _expand_latex_environments(text)

    # Collapse accidental doubled command slashes after matrix/cases rows have
    # already been converted to real line breaks.
    text = re.sub(r"\\\\(?=[A-Za-z])", lambda _match: "\\", text)
    text = _expand_latex_fractions(text)

    for _ in range(8):
        before = text
        text = re.sub(
            r"\\(?:operatorname|mathrm|text)\{([^{}]*)\}",
            r"\1",
            text,
        )
        if text == before:
            break

    # Font-safe representation for simple accent commands.
    text = re.sub(
        r"\\bar\s*([A-Za-z](?:_\{[^{}]+\}|_[A-Za-z0-9]+)?)",
        r"bar(\1)",
        text,
    )
    text = re.sub(
        r"\\overline\{([^{}]+)\}",
        r"bar(\1)",
        text,
    )

    replacements = {
        r"\lambda": "lambda",
        r"\Lambda": "Lambda",
        r"\xi": "xi",
        r"\max": "max",
        r"\min": "min",
        r"\sin": "sin",
        r"\cos": "cos",
        r"\sum": "∑",
        r"\prod": "∏",
        r"\sqrt": "√",
        r"\times": "×",
        r"\cdot": "·",
        r"\in": " in ",
        r"\leq": "≤",
        r"\geq": "≥",
        r"\neq": "≠",
        r"\approx": "≈",
        r"\infty": "∞",
        r"\alpha": "α",
        r"\beta": "β",
        r"\gamma": "γ",
        r"\theta": "θ",
        r"\sigma": "σ",
        r"\mu": "μ",
        r"\pm": "+/-",
        r"\vdots": ":",
        r"\ddots": "...",
        r"\cdots": "…",
        r"\ldots": "…",
        r"\dots": "…",
        r"\qquad": " ",
        r"\quad": " ",
        r"\left": "",
        r"\right": "",
        r"\,": " ",
        r"\;": " ",
        r"\!": "",
    }

    for source, target in replacements.items():
        text = text.replace(source, target)

    markup = escape(text)
    markup = re.sub(r"_\{([^{}]+)\}", r"<sub>\1</sub>", markup)
    markup = re.sub(r"\^\{([^{}]+)\}", r"<super>\1</super>", markup)
    markup = re.sub(r"_([A-Za-z0-9]+)", r"<sub>\1</sub>", markup)
    markup = re.sub(r"\^([A-Za-z0-9+\-]+)", r"<super>\1</super>", markup)

    # Remove remaining LaTeX grouping braces, but restore the visual brace used
    # for a piecewise ``cases`` environment.
    markup = markup.replace("{", "").replace("}", "")
    markup = markup.replace("ZZSCHOLARSYNCCASEOPENZZ", "{")
    markup = markup.replace("ZZSCHOLARSYNCCASECLOSEZZ", "}")

    # Preserve environment rows as ReportLab line breaks.
    markup = re.sub(r"[ \t]+", " ", markup)
    markup = re.sub(r"\s*\n\s*", "<br/>", markup)
    return markup.strip()


def _inline_markup(value):
    """Convert a safe subset of Markdown and inline math to ReportLab markup."""
    raw = _pdf_safe_text(value)
    protected = []

    def stash(content="", font=FORMULA_FONT, markup=None):
        token = f"ZZSCHOLARFORMULA{len(protected)}ZZ"
        rendered = markup if markup is not None else escape(content)
        protected.append((token, f"<font name='{font}'>{rendered}</font>"))
        return token

    # Protect math before Markdown emphasis. Use the same balanced-brace PDF
    # converter for inline and display math.
    raw = re.sub(
        r"\$\$([\s\S]+?)\$\$",
        lambda match: stash(markup=_latex_to_pdf_markup(match.group(1))),
        raw,
    )
    raw = re.sub(
        r"\\\[([\s\S]+?)\\\]",
        lambda match: stash(markup=_latex_to_pdf_markup(match.group(1))),
        raw,
    )
    raw = re.sub(
        r"\\\((.+?)\\\)",
        lambda match: stash(markup=_latex_to_pdf_markup(match.group(1))),
        raw,
    )
    raw = re.sub(
        r"(?<!\$)\$([^$\n]+?)\$(?!\$)",
        lambda match: stash(markup=_latex_to_pdf_markup(match.group(1))),
        raw,
    )
    raw = re.sub(
        r"`([^`]+)`",
        lambda match: stash(match.group(1), font="Courier"),
        raw,
    )

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

    # Normal **bold**.
    text = re.sub(
        r"\*\*(.+?)\*\*",
        r"<b>\1</b>",
        text,
    )

    # __bold__ only when underscores act as Markdown delimiters, not as
    # separators inside filenames / identifiers.
    text = re.sub(
        r"(^|[\s([{>])__([^_\n]+?)__(?=$|[\s.,!?;:)\]}>])",
        r"\1<b>\2</b>",
        text,
    )

    # Normal *italic*.
    text = re.sub(
        r"(^|[^*])\*([^*\n]+?)\*(?!\*)",
        r"\1<i>\2</i>",
        text,
    )

    # _italic_ only at natural text boundaries. This preserves filenames such
    # as 44_Songkhla, Thailand_GIS_AHP_2019.
    text = re.sub(
        r"(^|[\s([{>])_([^_\n]+?)_(?=$|[\s.,!?;:)\]}>])",
        r"\1<i>\2</i>",
        text,
    )

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

        # Markdown thematic break. Do not print literal "---" in exports.
        if re.fullmatch(r"(?:-{3,}|\*{3,}|_{3,})", line):
            flowables.append(
                HRFlowable(
                    width="100%",
                    thickness=0.4,
                    color=BORDER,
                    spaceBefore=2,
                    spaceAfter=6,
                )
            )
            index += 1
            continue

        # Comparison-grounding note: render it as a compact evidence callout
        # instead of leaving it as an isolated italic paragraph. Generated
        # comparison notes are sometimes wrapped in Markdown emphasis, so
        # normalize only those outer markers before checking the prefix.
        note_candidate = re.sub(r"^[*_]+|[*_]+$", "", line).strip()
        if note_candidate.lower().startswith(
            "this comparison uses only the retrieved passages"
        ):
            flowables.append(
                Paragraph(
                    _inline_markup(note_candidate),
                    styles["EvidenceNote"],
                )
            )
            flowables.append(Spacer(1, 6))
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
            safe_formula = _latex_to_pdf_markup(formula)
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
            normalized_headers = [
                header.strip().lower()
                for header in headers
            ]

            formula_table = (
                "formula" in normalized_headers
                and "symbols" in normalized_headers
            )

            comparison_table = (
                column_count >= 3
                and normalized_headers
                and normalized_headers[0] == "aspect"
            )

            if formula_table and column_count == 4:
                col_widths = [
                    available_width * 0.07,
                    available_width * 0.45,
                    available_width * 0.37,
                    available_width * 0.11,
                ]
            elif comparison_table:
                aspect_ratio = 0.18 if column_count <= 3 else 0.16
                aspect_width = available_width * aspect_ratio
                remaining_width = available_width - aspect_width
                paper_width = remaining_width / (column_count - 1)
                col_widths = [
                    aspect_width,
                    *[paper_width for _ in range(column_count - 1)],
                ]
            else:
                col_widths = [
                    available_width / column_count
                    for _ in range(column_count)
                ]

            if comparison_table:
                header_row = [
                    Paragraph(
                        _inline_markup(headers[0]),
                        styles["TableHeaderInverse"],
                    )
                ]
                header_row.extend(
                    Paragraph(
                        _inline_markup(cell),
                        styles["TableHeader"],
                    )
                    for cell in headers[1:]
                )
            else:
                header_row = [
                    Paragraph(
                        _inline_markup(cell),
                        styles["TableHeader"],
                    )
                    for cell in headers
                ]

            table_data = [header_row]

            for row in rows:
                rendered_row = []

                for cell_index in range(column_count):
                    value = (
                        row[cell_index]
                        if cell_index < len(row)
                        else ""
                    )
                    style_name = (
                        "AspectCell"
                        if comparison_table and cell_index == 0
                        else "TableCell"
                    )
                    rendered_row.append(
                        Paragraph(
                            _inline_markup(value),
                            styles[style_name],
                        )
                    )

                table_data.append(rendered_row)

            table = LongTable(
                table_data,
                colWidths=col_widths,
                repeatRows=1,
                hAlign="LEFT",
                splitByRow=1,
            )

            table_commands = [
                ("TEXTCOLOR", (0, 0), (-1, -1), INK),
                ("GRID", (0, 0), (-1, -1), 0.45, BORDER),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]

            if comparison_table:
                table_commands.extend(
                    [
                        ("BACKGROUND", (0, 0), (0, 0), PLUM),
                        ("BACKGROUND", (1, 0), (-1, 0), ROSE),
                        ("BACKGROUND", (0, 1), (0, -1), ROSE_SOFT),
                        (
                            "ROWBACKGROUNDS",
                            (1, 1),
                            (-1, -1),
                            [SURFACE, colors.HexColor("#FFF9F3")],
                        ),
                    ]
                )
            else:
                table_commands.extend(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), ROSE),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [SURFACE, colors.HexColor("#FFF9F3")],
                        ),
                    ]
                )

            table.setStyle(TableStyle(table_commands))
            flowables.extend([table, Spacer(1, 10)])
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
            list_kwargs = {
                "bulletType": "1" if ordered else "bullet",
                "leftIndent": 18,
                "bulletFontName": "Helvetica",
                "bulletFontSize": 8.5,
                "spaceAfter": 6,
            }
            if ordered:
                list_kwargs["start"] = str(start_number)

            flowables.append(
                ListFlowable(
                    items,
                    **list_kwargs,
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

    page_width, page_height = doc.pagesize
    left = 18 * mm
    right = page_width - (18 * mm)

    if doc.page > 1:
        canvas.setFont("Helvetica-Bold", 7.5)
        canvas.setFillColor(PLUM)
        canvas.drawString(left, page_height - (10 * mm), "ScholarSync")

        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(
            right,
            page_height - (10 * mm),
            "Research conversation",
        )

    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.4)
    canvas.line(left, 13 * mm, right, 13 * mm)

    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(left, 8 * mm, "ScholarSync - Read. Ask. Cite.")
    canvas.drawRightString(right, 8 * mm, f"Page {doc.page}")

    canvas.restoreState()


def build_conversation_pdf(conversation, messages):
    output = BytesIO()
    styles = getSampleStyleSheet()

    # Brand / report hierarchy
    styles.add(
        ParagraphStyle(
            name="Brand",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            textColor=PLUM_DARK,
            fontSize=20,
            leading=22,
            spaceAfter=1,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Tagline",
            parent=styles["Normal"],
            textColor=MUTED,
            fontSize=8,
            leading=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="HeaderKicker",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            textColor=GOLD,
            fontSize=7.2,
            leading=9,
            alignment=TA_RIGHT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ConversationEyebrow",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            textColor=PLUM,
            fontSize=7.3,
            leading=9,
            spaceBefore=3,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ConversationTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            textColor=PLUM_DARK,
            fontSize=18,
            leading=22,
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Meta",
            parent=styles["Normal"],
            textColor=MUTED,
            fontSize=8.3,
            leading=11,
        )
    )

    # Conversation content
    styles.add(
        ParagraphStyle(
            name="Role",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            textColor=PLUM,
            fontSize=7.5,
            leading=9,
            spaceBefore=8,
            spaceAfter=6,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="UserMessage",
            parent=styles["BodyText"],
            textColor=INK,
            fontSize=9.7,
            leading=14.5,
            spaceAfter=0,
            splitLongWords=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            textColor=INK,
            fontSize=9.4,
            leading=14.1,
            spaceAfter=6,
            splitLongWords=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Formula",
            parent=styles["BodyText"],
            fontName=FORMULA_FONT,
            textColor=INK,
            fontSize=10.2,
            leading=15.2,
            backColor=FORMULA_BG,
            borderColor=BORDER,
            borderWidth=0.6,
            borderPadding=9,
            leftIndent=3,
            rightIndent=3,
            alignment=TA_CENTER,
            splitLongWords=True,
            spaceBefore=3,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Quote",
            parent=styles["Body"],
            textColor=MUTED,
            leftIndent=10,
            borderColor=GOLD,
            borderWidth=1.3,
            borderPadding=7,
            backColor=STONE,
        )
    )
    styles.add(
        ParagraphStyle(
            name="EvidenceNote",
            parent=styles["Normal"],
            textColor=MUTED,
            fontSize=8.3,
            leading=11.8,
            backColor=GOLD_SOFT,
            borderColor=GOLD,
            borderWidth=0.7,
            borderPadding=8,
            spaceBefore=2,
            spaceAfter=4,
        )
    )

    # Tables
    styles.add(
        ParagraphStyle(
            name="TableHeader",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            textColor=PLUM_DARK,
            fontSize=8.5,
            leading=11,
            splitLongWords=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableHeaderInverse",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            textColor=colors.white,
            fontSize=8.5,
            leading=11,
            splitLongWords=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="AspectCell",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            textColor=PLUM_DARK,
            fontSize=8.4,
            leading=11.2,
            splitLongWords=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableCell",
            parent=styles["Normal"],
            textColor=INK,
            fontSize=8.2,
            leading=11.5,
            splitLongWords=True,
        )
    )

    # Evidence
    styles.add(
        ParagraphStyle(
            name="SourceHeading",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            textColor=PLUM,
            fontSize=7.5,
            leading=10,
            spaceBefore=8,
            spaceAfter=2,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="EvidenceSubtitle",
            parent=styles["Normal"],
            textColor=MUTED,
            fontSize=7.7,
            leading=10,
            spaceAfter=7,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SourceTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            textColor=PLUM_DARK,
            fontSize=8.5,
            leading=11,
            spaceAfter=2,
            splitLongWords=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SourceMeta",
            parent=styles["Normal"],
            textColor=MUTED,
            fontSize=7.6,
            leading=9.8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SourceExcerpt",
            parent=styles["Normal"],
            textColor=INK,
            fontSize=7.9,
            leading=11.1,
            splitLongWords=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="VerificationPill",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=6.9,
            leading=8.6,
            alignment=TA_CENTER,
        )
    )

    for level, size in ((1, 14), (2, 12.5), (3, 11.2), (4, 10.3)):
        styles.add(
            ParagraphStyle(
                name=f"MarkdownH{level}",
                parent=styles["Normal"],
                fontName="Helvetica-Bold",
                textColor=PLUM_DARK,
                fontSize=size,
                leading=size + 3,
                spaceBefore=8,
                spaceAfter=5,
                keepWithNext=True,
            )
        )

    doc = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=conversation.title,
        author="ScholarSync",
        subject="Evidence-grounded research conversation",
        allowSplitting=True,
    )
    available_width = A4[0] - doc.leftMargin - doc.rightMargin

    header_left = [
        Paragraph("ScholarSync", styles["Brand"]),
        Paragraph("Read. Ask. Cite.", styles["Tagline"]),
    ]
    header = Table(
        [
            [
                header_left,
                Paragraph(
                    "EVIDENCE-GROUNDED RESEARCH",
                    styles["HeaderKicker"],
                ),
            ]
        ],
        colWidths=[available_width * 0.67, available_width * 0.33],
    )
    header.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )

    document_count = conversation.workspace.documents.count()
    paper_word = "paper" if document_count == 1 else "papers"

    story = [
        header,
        Spacer(1, 5),
        HRFlowable(
            width="100%",
            thickness=0.8,
            color=GOLD,
            spaceAfter=10,
        ),
        Paragraph("RESEARCH CONVERSATION", styles["ConversationEyebrow"]),
        Paragraph(
            escape(_pdf_safe_text(conversation.title)),
            styles["ConversationTitle"],
        ),
        Paragraph(
            (
                f"{escape(_pdf_safe_text(conversation.workspace.name))}"
                f"  |  "
                f"{_format_generated(getattr(conversation, 'updated_at', None))}"
                f"  |  "
                f"{document_count} {paper_word}"
            ),
            styles["Meta"],
        ),
        Spacer(1, 12),
    ]

    for message in messages:
        is_user = message.role == "USER"
        role_label = "YOU" if is_user else "SCHOLARSYNC"

        story.append(CondPageBreak(28 * mm))
        story.append(Paragraph(role_label, styles["Role"]))

        if is_user:
            user_box = Table(
                [
                    [
                        Paragraph(
                            _inline_markup(message.content),
                            styles["UserMessage"],
                        )
                    ]
                ],
                colWidths=[available_width],
            )
            user_box.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), ROSE),
                        ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
                        ("LEFTPADDING", (0, 0), (-1, -1), 10),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                        ("TOPPADDING", (0, 0), (-1, -1), 9),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                    ]
                )
            )
            story.append(user_box)
        else:
            story.extend(
                _markdown_flowables(
                    message.content,
                    styles,
                    available_width,
                )
            )

        citations = list(
            message.citations
            .select_related("document")
            .all()
        )

        if citations:
            story.append(Spacer(1, 4))
            story.append(Paragraph("EVIDENCE USED", styles["SourceHeading"]))
            story.append(
                Paragraph(
                    "Retrieved passages supporting this response",
                    styles["EvidenceSubtitle"],
                )
            )

            for citation in citations:
                (
                    verification_label,
                    verification_background,
                    verification_text_color,
                ) = _verification_presentation(citation.verification_status)

                title = (
                    f"[{citation.citation_number}] "
                    f"{escape(_pdf_safe_text(citation.document.display_title))}"
                )
                source_meta = (
                    f"Page {citation.page_number}"
                    f"  |  retrieval score {citation.retrieval_score:.3f}"
                )
                excerpt = _truncate_pdf_excerpt(citation.quoted_passage)

                verification = Paragraph(
                    (
                        f"<font color='{verification_text_color}'>"
                        f"<b>{escape(verification_label)}</b>"
                        f"</font>"
                    ),
                    styles["VerificationPill"],
                )

                badge_width = 39 * mm
                card_inner_width = available_width - 16
                meta_width = card_inner_width - badge_width

                badge = Table(
                    [[verification]],
                    colWidths=[badge_width],
                    hAlign="RIGHT",
                )
                badge.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, -1), verification_background),
                            ("BOX", (0, 0), (-1, -1), 0.35, verification_background),
                            ("LEFTPADDING", (0, 0), (-1, -1), 6),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                            ("TOPPADDING", (0, 0), (-1, -1), 4),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ]
                    )
                )

                meta_row = Table(
                    [
                        [
                            Paragraph(
                                escape(source_meta),
                                styles["SourceMeta"],
                            ),
                            badge,
                        ]
                    ],
                    colWidths=[meta_width, badge_width],
                    hAlign="LEFT",
                )
                meta_row.setStyle(
                    TableStyle(
                        [
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 0),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                            ("TOPPADDING", (0, 0), (-1, -1), 0),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                        ]
                    )
                )

                source_content = [
                    Paragraph(title, styles["SourceTitle"]),
                    meta_row,
                    Spacer(1, 4),
                    Paragraph(
                        escape(excerpt),
                        styles["SourceExcerpt"],
                    ),
                ]

                evidence_card = Table(
                    [[source_content]],
                    colWidths=[available_width],
                )
                evidence_card.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, -1), SURFACE),
                            ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 8),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                            ("TOPPADDING", (0, 0), (-1, -1), 7),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                        ]
                    )
                )

                story.append(evidence_card)
                story.append(Spacer(1, 6))

        story.extend(
            [
                Spacer(1, 6),
                HRFlowable(
                    width="100%",
                    thickness=0.4,
                    color=BORDER,
                ),
                Spacer(1, 5),
            ]
        )

    story.extend(
        [
            Spacer(1, 7),
            Paragraph(
                (
                    "Generated from uploaded research papers. "
                    "Verify AI-generated interpretations against the cited source pages."
                ),
                styles["Meta"],
            ),
        ]
    )

    doc.build(
        story,
        onFirstPage=_draw_footer,
        onLaterPages=_draw_footer,
    )
    return output.getvalue()
