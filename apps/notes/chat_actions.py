import re

from apps.retrieval.lexical import tokens


NOTE_REQUEST_PATTERNS = (
    re.compile(
        r"^(?:please\s+)?(?:create|make|add)\s+(?:this\s+|that\s+|a\s+)?note\b",
        re.I,
    ),
    re.compile(
        r"^(?:please\s+)?save\s+(?:this|that|it)\s+as\s+(?:a\s+)?note\b",
        re.I,
    ),
    re.compile(
        r"^(?:please\s+)?save\s+(?:a\s+)?note\b",
        re.I,
    ),
    re.compile(
        r"^(?:please\s+)?(?:note|save)\s+(?:this|that)\b",
        re.I,
    ),
    re.compile(
        r"^(?:can|could|would)\s+you\s+(?:please\s+)?"
        r"(?:create|make|add|save)\s+(?:this\s+|that\s+|a\s+)?note\b",
        re.I,
    ),
    re.compile(
        r"^(?:please\s+)?add\s+(?:this|that)\s+to\s+(?:my\s+)?notes\b",
        re.I,
    ),
)

_GENERIC_DOCUMENT_TERMS = {
    "paper",
    "pdf",
    "study",
    "research",
    "article",
    "document",
    "method",
    "methods",
    "methodology",
    "approach",
    "workflow",
    "gis",
    "ahp",
    "fahp",
    "topsis",
    "mcdm",
    "solar",
    "energy",
    "site",
    "selection",
}

_GENERIC_NOTE_TERMS = {
    "create",
    "make",
    "add",
    "save",
    "note",
    "notes",
    "this",
    "that",
    "it",
    "a",
    "an",
    "the",
    "from",
    "of",
    "my",
    "as",
    "to",
    "please",
    "can",
    "could",
    "would",
    "you",
    "discussion",
    "conversation",
    "answer",
    "response",
    "above",
    "previous",
}


def _normalize(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def is_chat_note_request(question):
    """Return True only for direct note-creation commands, not help questions."""
    value = str(question or "").strip()
    if not value:
        return False
    return any(pattern.search(value) for pattern in NOTE_REQUEST_PATTERNS)


def _document_title(document):
    return str(
        getattr(document, "display_title", "")
        or getattr(document, "original_filename", "")
        or "Document"
    )


def _document_terms(document):
    return {
        term
        for term in tokens(_document_title(document))
        if len(term) >= 3 and term not in _GENERIC_DOCUMENT_TERMS
    }


def _explicit_document(question, documents):
    query_terms = set(tokens(question))
    matches = []
    for document in documents:
        distinctive = _document_terms(document)
        overlap = query_terms & distinctive
        if not overlap:
            continue
        score = sum(2.0 + min(len(term), 10) / 10 for term in overlap)
        matches.append((score, document))

    if not matches:
        return None

    matches.sort(key=lambda row: row[0], reverse=True)
    if len(matches) > 1 and matches[0][0] == matches[1][0]:
        return None
    return matches[0][1]


def _note_subject_terms(question):
    return {
        term
        for term in tokens(question)
        if len(term) >= 3 and term not in _GENERIC_NOTE_TERMS
    }


def _topic_label(question):
    normalized = f" {_normalize(question)} "
    if any(term in normalized for term in (" methodology ", " methods ", " method ", " approach ", " workflow ")):
        return "Methodology"
    if any(term in normalized for term in (" limitation ", " limitations ", " drawback ", " drawbacks ")):
        return "Limitations"
    if any(term in normalized for term in (" formula ", " formulas ", " equation ", " equations ")):
        return "Formulas"
    if any(term in normalized for term in (" finding ", " findings ", " result ", " results ")):
        return "Findings"
    if any(term in normalized for term in (" summary ", " overview ")):
        return "Summary"
    return "Research note"


def _note_title(question, document):
    topic = _topic_label(question)
    if document is not None:
        title = f"{_document_title(document)} — {topic}"
    else:
        title = topic
    return title[:180]


def _candidate_messages(conversation, question, documents):
    messages = list(
        conversation.messages.prefetch_related("citations__document").order_by("created_at")
    )
    if not messages:
        return []

    explicit_document = _explicit_document(question, documents)
    explicit_id = str(getattr(explicit_document, "id", "")) if explicit_document else ""
    subject_terms = _note_subject_terms(question)
    topic = _topic_label(question).lower()

    prior_user = ""
    candidates = []
    total = max(1, len(messages))

    for index, message in enumerate(messages):
        role = str(getattr(message, "role", "") or "").upper()
        content = str(getattr(message, "content", "") or "").strip()

        if role == "USER":
            prior_user = content
            continue
        if role != "ASSISTANT" or not content:
            continue

        # Never use an answer produced for an earlier note command as the
        # source of a new note. This also excludes note-action confirmations.
        # Otherwise, retrying "create a note..." can select the previous failed
        # note attempt instead of the actual research discussion the user named.
        if is_chat_note_request(prior_user) or str(
            getattr(message, "model_name", "") or ""
        ).lower() == "note-action":
            continue

        citations = list(message.citations.all())
        cited_document_ids = {
            str(getattr(citation, "document_id", "") or "")
            for citation in citations
            if getattr(citation, "document_id", None)
        }

        # If the user explicitly named a document, never select an answer that
        # is positively grounded in a different document.
        if explicit_id and cited_document_ids and explicit_id not in cited_document_ids:
            continue

        basis = f"{prior_user} {content}".strip()
        basis_terms = set(tokens(basis))
        overlap = len(subject_terms & basis_terms)

        score = float(overlap) * 3.0
        score += (index + 1) / total * 0.35

        if citations:
            score += 1.0

        if explicit_id:
            if cited_document_ids == {explicit_id}:
                score += 8.0
            elif explicit_id in cited_document_ids:
                score += 3.0
            elif _document_terms(explicit_document) & basis_terms:
                score += 1.0

        lower_basis = basis.lower()
        if topic == "methodology" and any(
            term in lower_basis
            for term in ("methodology", "method", "fahp", "gis", "workflow", "pairwise")
        ):
            score += 2.0
        elif topic == "limitations" and any(
            term in lower_basis for term in ("limitation", "limitations", "drawback")
        ):
            score += 2.0
        elif topic == "formulas" and any(
            term in lower_basis for term in ("formula", "equation", "calculation")
        ):
            score += 2.0
        elif topic == "findings" and any(
            term in lower_basis for term in ("finding", "findings", "result", "results")
        ):
            score += 2.0

        candidates.append(
            {
                "score": score,
                "message": message,
                "prior_user": prior_user,
                "citations": citations,
                "explicit_document": explicit_document,
            }
        )

    candidates.sort(key=lambda row: row["score"], reverse=True)
    return candidates


def build_chat_note_draft(conversation, question, documents):
    """Build a deterministic note draft from an earlier assistant discussion.

    This function does not mutate the database. The caller can create the Note
    inside the same transaction used for the conversation messages.
    """
    candidates = _candidate_messages(conversation, question, documents)
    if not candidates:
        return None

    selected = candidates[0]
    message = selected["message"]
    citations = selected["citations"]
    explicit_document = selected["explicit_document"]

    cited_documents = {}
    for citation in citations:
        document = getattr(citation, "document", None)
        if document is not None:
            cited_documents[str(document.id)] = document

    document = None
    if explicit_document is not None:
        document = explicit_document
    elif len(cited_documents) == 1:
        document = next(iter(cited_documents.values()))

    source_pages = []
    if document is not None:
        for citation in citations:
            if str(getattr(citation, "document_id", "")) != str(document.id):
                continue
            page = int(getattr(citation, "page_number", 0) or 0)
            if page > 0 and page not in source_pages:
                source_pages.append(page)
        source_pages.sort()

    page_number = source_pages[0] if len(source_pages) == 1 else None

    content = str(message.content).strip()
    if document is not None and source_pages:
        page_label = "page" if len(source_pages) == 1 else "pages"
        page_text = ", ".join(str(page) for page in source_pages)
        content = (
            f"{content}\n\n"
            f"Source: {_document_title(document)}, {page_label} {page_text}."
        )

    return {
        "title": _note_title(question, document),
        "content": content,
        "document": document,
        "page_number": page_number,
        "source_pages": source_pages,
        "tags": f"chat,{_topic_label(question).lower().replace(' ', '-')}",
    }


def chat_note_confirmation(draft):
    title = draft["title"]
    document = draft.get("document")
    source_pages = draft.get("source_pages") or []

    sentence = f"Saved note **{title}**."
    if document is not None:
        sentence += f" It is linked to **{_document_title(document)}**"
        if source_pages:
            label = "page" if len(source_pages) == 1 else "pages"
            pages = ", ".join(str(page) for page in source_pages)
            sentence += f" ({label} {pages})"
        sentence += "."
    return sentence
