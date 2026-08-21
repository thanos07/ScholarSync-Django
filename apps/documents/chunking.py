import re


SENTENCE_BOUNDARY_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z0-9(])"
)


def clean_text(text: str) -> str:
    """
    Normalize PDF-extracted text without rewriting
    or inventing document content.
    """
    text = (text or "").replace("\x00", " ")
    text = text.replace("\u00ad", "")

    # Collapse repeated spaces/tabs while preserving line structure.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces surrounding line breaks.
    text = re.sub(r" *\n *", "\n", text)

    # Collapse excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _split_long_block(
    block: str,
    target_chars: int,
):
    """
    Split PDF pages that contain no useful
    blank-line paragraph breaks.
    """
    block = block.strip()

    if len(block) <= target_chars:
        return [block] if block else []

    sentences = [
        part.strip()
        for part in SENTENCE_BOUNDARY_RE.split(block)
        if part.strip()
    ]

    if len(sentences) == 1:
        # Last-resort word split for extraction output
        # with no useful punctuation.
        words = block.split()

        pieces = []
        current = []

        for word in words:
            candidate = " ".join([*current, word])

            if current and len(candidate) > target_chars:
                pieces.append(" ".join(current))
                current = [word]
            else:
                current.append(word)

        if current:
            pieces.append(" ".join(current))

        return pieces

    pieces = []
    current = ""

    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()

        if current and len(candidate) > target_chars:
            pieces.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        pieces.append(current)

    return pieces


def _overlap_tail(
    text: str,
    overlap_chars: int,
):
    """
    Return a safe overlap from the end of a chunk.

    The overlap must never begin in the middle of a token.

    For example, a fixed character slice could otherwise
    transform:

        "solar farms"

    into:

        "ar farms"

    in the next chunk.

    Prefer a sentence boundary when available. Otherwise,
    move the overlap start backward until the complete token
    at the character boundary is preserved.
    """
    if overlap_chars <= 0:
        return ""

    if len(text) <= overlap_chars:
        return text.strip()

    start = len(text) - overlap_chars

    # A raw character boundary may land inside a word/token.
    #
    # Example:
    #
    #     sol|ar farms
    #
    # Move backward until the beginning of that token:
    #
    #     |solar farms
    #
    if (
        start > 0
        and start < len(text)
        and not text[start - 1].isspace()
        and not text[start].isspace()
    ):
        while (
            start > 0
            and not text[start - 1].isspace()
        ):
            start -= 1

    tail = text[start:]

    # When possible, start the overlap after a complete
    # sentence instead of in arbitrary prose.
    boundary = re.search(
        r"[.!?]\s+",
        tail,
    )

    if boundary:
        tail = tail[boundary.end():]

    return tail.strip()


def chunk_page(
    text: str,
    target_chars: int = 1450,
    overlap_chars: int = 180,
):
    """
    Create compact page-local chunks while preserving
    equations and surrounding research context.
    """
    text = clean_text(text)

    if not text:
        return []

    raw_blocks = [
        part.strip()
        for part in text.split("\n\n")
        if part.strip()
    ]

    blocks = []

    for block in raw_blocks:
        blocks.extend(
            _split_long_block(
                block,
                target_chars,
            )
        )

    chunks = []
    current = ""

    for block in blocks:
        candidate = (
            f"{current}\n\n{block}".strip()
        )

        if (
            current
            and len(candidate) > target_chars
        ):
            chunks.append(current)

            overlap = _overlap_tail(
                current,
                overlap_chars,
            )

            current = (
                f"{overlap}\n\n{block}".strip()
                if overlap
                else block
            )

            if len(current) > target_chars * 1.35:
                # A pathological PDF extraction block
                # should not create an excessively large chunk.
                split_current = _split_long_block(
                    current,
                    target_chars,
                )

                chunks.extend(
                    split_current[:-1]
                )

                current = split_current[-1]

        else:
            current = candidate

    if current:
        chunks.append(current)

    # Remove accidental duplicate chunks caused by
    # repeated PDF headers/footers.
    unique = []
    seen = set()

    for chunk in chunks:
        key = re.sub(
            r"\s+",
            " ",
            chunk,
        ).strip()

        if not key or key in seen:
            continue

        seen.add(key)
        unique.append(chunk)

    return unique