import math
import re
from collections import Counter
from dataclasses import dataclass

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]{0,}")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for",
    "from", "how", "in", "is", "it", "of", "on", "or", "that", "the", "their",
    "these", "this", "to", "was", "were", "what", "when", "where", "which", "why",
    "with", "work", "works", "explain", "paper", "papers", "please", "tell", "me",
}

TOKEN_ALIASES = {
    "transformers": "transformer",
    "networks": "network",
    "models": "model",
    "architectures": "architecture",
    "convolutions": "convolution",
    "encoders": "encoder",
    "decoders": "decoder",
    "mechanisms": "mechanism",
    "dependencies": "dependency",
    "positions": "position",
    "layers": "layer",
    "heads": "head",
    "queries": "query",
    "keys": "key",
    "values": "value",
    "rnns": "rnn",
    "cnns": "cnn",
}


def _normalize_token(token):
    token = token.lower().strip("-_")
    if not token:
        return ""
    token = TOKEN_ALIASES.get(token, token)
    if token.endswith("ies") and len(token) > 5:
        token = token[:-3] + "y"
    elif token.endswith("s") and len(token) > 5 and not token.endswith(("ss", "us", "is")):
        token = token[:-1]
    return TOKEN_ALIASES.get(token, token)


def tokens(text, *, remove_stopwords=False):
    values = [_normalize_token(token) for token in TOKEN_RE.findall(text or "")]
    values = [token for token in values if token]
    if remove_stopwords:
        values = [token for token in values if token not in STOPWORDS]
    return values


@dataclass
class SearchHit:
    item: object
    score: float


def _title_for(item):
    document = getattr(item, "document", None)
    if document is not None:
        return getattr(document, "display_title", "") or ""
    return getattr(item, "source", "") or ""


def bm25_search(query, items, text_getter=lambda x: x.content, limit=6):
    """BM25 retrieval with light stemming, phrase matching, and title boosts."""
    query_terms = tokens(query, remove_stopwords=True)
    if not query_terms:
        query_terms = tokens(query)
    if not query_terms:
        return []

    item_list = list(items)
    raw_docs = [text_getter(item) or "" for item in item_list]
    docs = [tokens(text) for text in raw_docs]
    n = len(docs)
    if n == 0:
        return []

    avg_len = sum(map(len, docs)) / n or 1
    df = Counter(term for doc in docs for term in set(doc))
    query_counter = Counter(query_terms)
    normalized_query = " ".join(query_terms)
    phrase_candidates = [
        " ".join(query_terms[index:index + size])
        for size in (4, 3, 2)
        for index in range(0, max(0, len(query_terms) - size + 1))
    ]

    scored = []
    for item, doc, raw_text in zip(item_list, docs, raw_docs):
        tf = Counter(doc)
        score = 0.0
        matched_terms = 0

        for term, query_frequency in query_counter.items():
            if term not in tf:
                continue
            matched_terms += 1
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            frequency = tf[term]
            denominator = frequency + 1.5 * (1 - 0.75 + 0.75 * len(doc) / avg_len)
            score += query_frequency * idf * (frequency * 2.5) / denominator

        if score <= 0:
            continue

        lower_text = " ".join(tokens(raw_text))
        lower_title = " ".join(tokens(_title_for(item)))
        coverage = matched_terms / max(1, len(query_counter))
        score *= 1.0 + (0.7 * coverage)

        for phrase in phrase_candidates:
            if phrase and phrase in lower_text:
                score += 1.25
            if phrase and phrase in lower_title:
                score += 2.0

        if normalized_query and normalized_query in lower_text:
            score += 1.8
        if normalized_query and normalized_query in lower_title:
            score += 2.6
        if any(term in lower_title for term in query_terms):
            score += 0.7

        # Penalize bibliography-only passages, which often contain matching paper
        # titles but do not explain the requested concept.
        lowered_raw = raw_text.lower()
        if lowered_raw.count(" et al.") >= 5 or lowered_raw.count("arxiv") >= 3:
            score *= 0.55

        scored.append(SearchHit(item=item, score=score))

    return sorted(scored, key=lambda hit: hit.score, reverse=True)[:limit]
