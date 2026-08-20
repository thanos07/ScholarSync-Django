import base64
import logging
import re
import time

import fitz
import httpx
from django.conf import settings


logger = logging.getLogger(__name__)


# A real visual-equation appendix page should explicitly contain:
# 1. "table"
# 2. "appendix"
# 3. "formula" or "equation"
#
# This prevents ordinary body pages such as:
# "The formulas ... are in Appendix A"
# from triggering vision extraction just because they also contain a figure.
_TABLE_APPENDIX_RE = re.compile(
    r"\btable\b[\s\S]{0,180}\bappendix\b"
    r"|\bappendix\b[\s\S]{0,180}\btable\b",
    re.I,
)

_FORMULA_LANGUAGE_RE = re.compile(
    r"\b(?:equations?|formulas?)\b",
    re.I,
)

_STEP_LINE_RE = re.compile(
    r"^\s*STEP\s+\d+\s*:\s*(.+?)\s*$",
    re.I,
)


def _looks_like_formula_image_page(page_text):
    """
    Return True only for pages that look like actual appendix
    formula/equation-table pages.

    Example accepted:
        "The table presented in Appendix A uses the formulas..."

    Example rejected:
        "The formulas used to calculate the AHP criteria are in Appendix A."
    """
    text = str(page_text or "")

    has_table_appendix = bool(_TABLE_APPENDIX_RE.search(text))
    has_formula_language = bool(_FORMULA_LANGUAGE_RE.search(text))

    return has_table_appendix and has_formula_language


def _significant_image_rects(page, min_area_ratio=0.12):
    """
    Return large embedded-image regions on a PDF page.

    Small icons/logos are ignored. This is intentionally conservative
    because formula tables in the diagnosed Brazil paper occupy a
    substantial part of the page.
    """
    page_area = max(
        1.0,
        float(page.rect.width * page.rect.height),
    )

    rects = []
    seen = set()

    for info in page.get_image_info(xrefs=True):
        bbox = info.get("bbox")

        if not bbox:
            continue

        rect = fitz.Rect(bbox)

        area_ratio = (
            float(rect.width * rect.height)
            / page_area
        )

        if area_ratio < min_area_ratio:
            continue

        key = tuple(
            round(value, 1)
            for value in rect
        )

        if key in seen:
            continue

        seen.add(key)
        rects.append(rect)

    return rects


def _vision_prompt():
    """
    Prompt used only for one-time visual equation transcription
    during document indexing.
    """
    return (
        "This image is an appendix formula/equation table from an uploaded "
        "research paper. "
        "Read ONLY mathematical equations visibly printed in the table's "
        "EQUATION column. "
        "Transcribe every equation in top-to-bottom row order. "
        "Do not use outside knowledge, do not repair missing symbols, and "
        "do not invent or replace formulas with textbook versions. "
        "If an equation is unreadable, write UNCERTAIN for that row. "
        "Output plain text only, one line per row, formatted exactly as "
        "STEP 1: equation, STEP 2: equation, and so on."
    )


def _clean_transcription(content):
    """
    Keep only equation-looking STEP rows returned by the vision model.

    This provides a small safety layer against prose or unsupported
    model-generated material.
    """
    lines = []

    normalized = (
        str(content or "")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )

    for raw_line in normalized.split("\n"):
        match = _STEP_LINE_RE.match(raw_line)

        if not match:
            continue

        equation = match.group(1).strip()

        if not equation:
            continue

        if equation.upper() == "UNCERTAIN":
            continue

        looks_mathematical = (
            "=" in equation
            or any(
                symbol in equation
                for symbol in (
                    "∑",
                    "Σ",
                    "√",
                    "∏",
                    "λ",
                    "ξ",
                    "/",
                    "^",
                )
            )
        )

        if not looks_mathematical:
            continue

        lines.append(
            f"STEP {len(lines) + 1}: {equation}"
        )

    return lines


def _message_text(data):
    """
    Extract assistant text from a Groq-compatible
    chat-completion response.
    """
    choices = data.get("choices") or []

    if not choices:
        return ""

    message = (
        (choices[0] or {}).get("message")
        or {}
    )

    content = message.get("content")

    if not isinstance(content, str):
        return ""

    return content.strip()


def _retry_after_seconds(response):
    """
    Return the provider-requested delay for a 429 response.

    Groq commonly returns text such as:
        "Please try again in 5.805s."

    Prefer Retry-After when available, otherwise parse the response body.
    """
    headers = getattr(response, "headers", {}) or {}

    retry_after = headers.get("retry-after")

    if retry_after:
        try:
            return max(
                0.5,
                min(float(retry_after), 60.0),
            )
        except (TypeError, ValueError):
            pass

    body = str(
        getattr(response, "text", "")
        or ""
    )

    match = re.search(
        r"try again in\s+([0-9.]+)s",
        body,
        re.I,
    )

    if match:
        try:
            return max(
                0.5,
                min(float(match.group(1)), 60.0),
            )
        except (TypeError, ValueError):
            pass

    return 6.0

def _transcribe_image_png(png_bytes):
    """
    Transcribe equations from one rendered image region.

    Vision is used only during indexing.

    A Groq 429 is retried once after respecting the provider's
    requested reset delay. Other failures remain non-fatal.
    """
    api_key = str(
        getattr(
            settings,
            "GROQ_API_KEY",
            "",
        )
        or ""
    ).strip()

    if not api_key:
        logger.info(
            "Skipping visual equation extraction because "
            "GROQ_API_KEY is not configured."
        )
        return []

    model = (
        str(
            getattr(
                settings,
                "GROQ_VISION_MODEL",
                "qwen/qwen3.6-27b",
            )
            or ""
        ).strip()
        or "qwen/qwen3.6-27b"
    )

    encoded = base64.b64encode(
        png_bytes
    ).decode("ascii")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": _vision_prompt(),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                "data:image/png;base64,"
                                + encoded
                            )
                        },
                    },
                ],
            }
        ],
        "reasoning_effort": "none",
        "temperature": 0.1,

        # Our measured appendix transcriptions used:
        # AHP    ~167 completion tokens
        # TOPSIS ~255 completion tokens
        # MAUT    ~98 completion tokens
        #
        # 500 leaves comfortable room while reducing TPM pressure.
        "max_completion_tokens": 500,
    }

    max_attempts = 2

    for attempt in range(1, max_attempts + 1):
        try:
            response = httpx.post(
                (
                    "https://api.groq.com/"
                    "openai/v1/chat/completions"
                ),
                headers={
                    "Authorization": (
                        f"Bearer {api_key}"
                    ),
                    "Content-Type": (
                        "application/json"
                    ),
                },
                json=payload,
                timeout=90,
            )

            if response.status_code == 429:
                if attempt >= max_attempts:
                    logger.warning(
                        "Visual formula transcription "
                        "rate limit remained after retry "
                        "(model=%s).",
                        model,
                    )
                    return []

                wait_seconds = (
                    _retry_after_seconds(response)
                    + 0.75
                )

                logger.warning(
                    "Visual formula transcription "
                    "rate limited; retrying in %.2fs "
                    "(attempt=%s/%s, model=%s).",
                    wait_seconds,
                    attempt,
                    max_attempts,
                    model,
                )

                time.sleep(wait_seconds)
                continue

            response.raise_for_status()

            content = _message_text(
                response.json()
            )

            lines = _clean_transcription(
                content
            )

            if not lines:
                logger.warning(
                    "Visual formula transcription "
                    "returned no usable equation rows "
                    "(model=%s).",
                    model,
                )

            return lines

        except httpx.HTTPStatusError as exc:
            body = ""

            try:
                body = exc.response.text[:700]
            except Exception:
                pass

            logger.warning(
                "Visual formula transcription failed "
                "(status=%s, model=%s): %s",
                getattr(
                    exc.response,
                    "status_code",
                    "?",
                ),
                model,
                body,
            )

            return []

        except httpx.RequestError as exc:
            logger.warning(
                "Visual formula transcription "
                "network failure: %s",
                exc,
            )

            return []

        except Exception:
            logger.exception(
                "Unexpected visual formula "
                "transcription failure"
            )

            return []

    return []


def extract_visual_equation_chunks(
    page,
    page_number,
    page_text,
):
    """
    Recover equation chunks from large raster formula tables.

    Vision extraction runs only when:

    1. The selectable text indicates an actual appendix table.
    2. Formula/equation language is present.
    3. A sufficiently large embedded image exists.

    Normal PDF text indexing remains independent of this function.
    """
    if not _looks_like_formula_image_page(
        page_text
    ):
        return []

    rects = _significant_image_rects(
        page
    )

    if not rects:
        return []

    recovered = []
    seen = set()

    for rect in rects:
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(
                2.5,
                2.5,
            ),
            clip=rect,
            alpha=False,
        )

        png_bytes = pixmap.tobytes(
            "png"
        )

        lines = _transcribe_image_png(
            png_bytes
        )

        unique_lines = []

        for line in lines:
            key = re.sub(
                r"\s+",
                " ",
                line,
            ).strip().lower()

            if key in seen:
                continue

            seen.add(key)
            unique_lines.append(
                line
            )

        if not unique_lines:
            continue

        content = "\n".join(
            [
                (
                    "[Visual equation extraction "
                    f"- page {page_number}]"
                ),
                (
                    "The following equations were "
                    "transcribed from an embedded "
                    "appendix image in this uploaded PDF. "
                    "Treat them as visual transcription "
                    "evidence; do not replace them with "
                    "outside textbook formulas."
                ),
                *unique_lines,
            ]
        )

        recovered.append(
            content
        )

    return recovered