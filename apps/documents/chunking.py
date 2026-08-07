import re


SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


def clean_text(text: str) -> str:
    text = (text or "").replace("\x00", " ")
    text = text.replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_long_block(block: str, target_chars: int):
    """Split PDF pages that contain no useful blank-line paragraph breaks."""
    block = block.strip()
    if len(block) <= target_chars:
        return [block] if block else []

    sentences = [part.strip() for part in SENTENCE_BOUNDARY_RE.split(block) if part.strip()]
    if len(sentences) == 1:
        # Last-resort word split for extraction output with no punctuation.
        words = block.split()
        pieces, current = [], []
        for word in words:
            if current and len(" ".join([*current, word])) > target_chars:
                pieces.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            pieces.append(" ".join(current))
        return pieces

    pieces, current = [], ""
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


def _overlap_tail(text: str, overlap_chars: int):
    if overlap_chars <= 0 or len(text) <= overlap_chars:
        return text
    tail = text[-overlap_chars:]
    boundary = re.search(r"[.!?]\s+", tail)
    return tail[boundary.end():].strip() if boundary else tail.strip()


def chunk_page(text: str, target_chars: int = 1450, overlap_chars: int = 180):
    """Create compact page-local chunks while preserving equations and context."""
    text = clean_text(text)
    if not text:
        return []

    raw_blocks = [part.strip() for part in text.split("\n\n") if part.strip()]
    blocks = []
    for block in raw_blocks:
        blocks.extend(_split_long_block(block, target_chars))

    chunks, current = [], ""
    for block in blocks:
        candidate = f"{current}\n\n{block}".strip()
        if current and len(candidate) > target_chars:
            chunks.append(current)
            overlap = _overlap_tail(current, overlap_chars)
            current = f"{overlap}\n\n{block}".strip() if overlap else block
            if len(current) > target_chars * 1.35:
                # A pathological extraction block should not create a huge chunk.
                split_current = _split_long_block(current, target_chars)
                chunks.extend(split_current[:-1])
                current = split_current[-1]
        else:
            current = candidate

    if current:
        chunks.append(current)

    # Remove accidental duplicate chunks caused by repeated PDF headers/footers.
    unique = []
    seen = set()
    for chunk in chunks:
        key = re.sub(r"\s+", " ", chunk).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(chunk)
    return unique
