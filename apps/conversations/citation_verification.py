import re

from apps.retrieval.lexical import STOPWORDS, tokens


_CITATION_RE = re.compile(r"\[(\d+)\]")
_NUMBER_RE = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?(?:\s*%)?")
_MARKDOWN_RE = re.compile(r"[*_`#>]")

STATUS_LABELS = {
    "SUPPORTED": "Strong evidence match",
    "PARTIAL": "Partial evidence match",
    "REVIEW": "Needs review",
    "UNCHECKED": "Not checked",
    "VALID": "Supported",
}


def _strip_citations(text):
    return _CITATION_RE.sub("", str(text or ""))


def _clean_claim_text(text):
    value = _strip_citations(text)
    value = _MARKDOWN_RE.sub("", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" |:-\t\n")


def _citation_only_source_line(line):
    """Return True when a line only groups source markers, not a factual claim."""
    value = str(line or "").strip()

    # Models often emit the label in Markdown, e.g.
    # "**Sources:** [1], [4], [5]". Strip formatting before classification so
    # a grouped source list is not mistaken for a factual claim.
    value = re.sub(r"[*_`#>]", "", value)
    value = re.sub(r"\s+", " ", value).strip()

    return bool(
        re.fullmatch(
            r"(?:sources?|citations?|evidence)\s*:?\s*"
            r"(?:\[\d+\]\s*[,;]?\s*)+",
            value,
            flags=re.I,
        )
    )


def _claim_units_for_citation(answer, citation_number):
    marker = f"[{citation_number}]"
    units = []

    for raw_line in str(answer or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if marker not in line:
            continue

        # A trailing line such as "Sources: [1], [2], [3]" does not map any
        # individual claim to an individual source. Treat it as unmapped rather
        # than falsely labelling the citation as unsupported.
        if _citation_only_source_line(line):
            continue

        # Comparison/table answers can cite several papers on one row. Verify
        # each citation only against the table cell where that marker appears.
        if "|" in line:
            for cell in line.strip("|").split("|"):
                if marker in cell:
                    claim = _clean_claim_text(cell)
                    if claim:
                        units.append(claim)
            continue

        # For prose/list answers the citation usually sits at the end of the
        # factual sentence or bullet. Keep the complete cited line.
        claim = _clean_claim_text(line)
        if claim:
            units.append(claim)

    return units


def _meaningful_terms(text):
    return {
        term
        for term in tokens(str(text or ""))
        if term not in STOPWORDS and len(term) > 2 and not term.isdigit()
    }


def _numbers(text):
    return {
        re.sub(r"\s+", "", match.group(0))
        for match in _NUMBER_RE.finditer(_strip_citations(text))
    }


def _claim_support_score(claim, passage):
    claim_terms = _meaningful_terms(claim)
    passage_terms = _meaningful_terms(passage)
    if not claim_terms or not passage_terms:
        return 0.0, False

    overlap = len(claim_terms & passage_terms)
    coverage = overlap / max(1, len(claim_terms))
    precision = overlap / max(1, min(len(passage_terms), 40))
    score = (0.82 * coverage) + (0.18 * precision)

    normalized_claim = " ".join(tokens(claim))
    normalized_passage = " ".join(tokens(passage))
    if normalized_claim and normalized_claim in normalized_passage:
        score = max(score, 0.92)

    claim_numbers = _numbers(claim)
    passage_numbers = _numbers(passage)
    numeric_mismatch = bool(claim_numbers and not claim_numbers.issubset(passage_numbers))
    if numeric_mismatch:
        score *= 0.45

    return min(score, 1.0), numeric_mismatch


def verify_citation_support(answer, citation_number, passage):
    """Return a conservative claim-to-passage support status.

    This is a deterministic evidence-match check, not a truth guarantee. It
    verifies that the claim text carrying [n] has meaningful lexical/numeric
    support in the passage saved for citation n.
    """
    claims = _claim_units_for_citation(answer, citation_number)
    if not claims:
        return "UNCHECKED"

    best_score = 0.0
    best_numeric_mismatch = False
    for claim in claims:
        score, numeric_mismatch = _claim_support_score(claim, passage)
        if score > best_score:
            best_score = score
            best_numeric_mismatch = numeric_mismatch

    if best_score >= 0.42 and not best_numeric_mismatch:
        return "SUPPORTED"
    if best_score >= 0.20:
        return "PARTIAL"
    return "REVIEW"


def verification_label(status):
    return STATUS_LABELS.get(str(status or "").upper(), "Not checked")
