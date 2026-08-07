import re
from collections import defaultdict

from apps.retrieval.lexical import SearchHit, bm25_search, tokens
from .generator import generate_answer


REFERENTIAL_TERMS = {
    "it", "its", "this", "that", "they", "them", "their", "these",
    "those", "former", "latter", "above", "previous", "same", "here",
}

ACKNOWLEDGEMENTS = {
    "thanks", "thank you", "thanks a lot", "ok thanks", "okay thanks",
    "ok thank you", "okay thank you", "got it", "understood", "ok", "okay",
    "great", "awesome", "nice", "perfect", "cool", "good", "sounds good",
    "excellent", "amazing", "wonderful", "fantastic", "very good",
}

CAPABILITY_QUESTIONS = {
    "what do you do", "what can you do", "what are you", "who are you",
    "what is scholarsync", "how can you help", "help", "show capabilities",
    "what else could you do", "what else can you do", "what else do you do",
    "what more can you do", "what other things can you do",
}

# Words that often occur in research-paper filenames but are not useful as a
# unique document identifier. This lets a query such as "Tamil Nadu paper" or
# "Egypt paper" resolve to the intended uploaded PDF without hard-coding a
# fixed list of countries or filenames.
GENERIC_TITLE_TERMS = {
    "paper", "pdf", "study", "research", "article", "journal", "conference",
    "solar", "photovoltaic", "energy", "power", "site", "sites", "suitability",
    "mapping", "analysis", "method", "methods", "model", "models", "approach",
    "gis", "ahp", "topsis", "mcdm", "mcda", "fuzzy", "boolean", "weighted",
    "ranking", "rank", "farm", "farms", "plant", "plants", "framework",
}

INSUFFICIENT_ANSWER = (
    "I could not find enough relevant evidence in your uploaded documents to answer that question."
)


def _normalize_question(question):
    return re.sub(r"[^a-z0-9]+", " ", str(question or "").lower()).strip()


def _document_title(document):
    return str(getattr(document, "display_title", "") or getattr(document, "original_filename", "") or "Document")


def _document_status(document):
    return str(getattr(document, "processing_status", "") or "").upper()


def _document_id(document):
    return str(getattr(document, "id", ""))


def _workspace_documents(chunks, documents=None):
    if documents is not None:
        values = list(documents)
    else:
        values = []
        seen = set()
        for chunk in chunks:
            document = getattr(chunk, "document", None)
            if document is None:
                continue
            key = _document_id(document) or _document_title(document)
            if key in seen:
                continue
            seen.add(key)
            values.append(document)
    return values


def _inventory_answer(documents):
    documents = list(documents)
    count = len(documents)
    noun = "PDF" if count == 1 else "PDFs"
    if not documents:
        return "There are no uploaded PDFs in this workspace yet."
    lines = [f"You have **{count} uploaded {noun}** in this workspace:"]
    for index, document in enumerate(documents, 1):
        status = _document_status(document)
        suffix = f" — {status.title()}" if status and status != "READY" else ""
        lines.append(f"{index}. **{_document_title(document)}**{suffix}")
    return "\n".join(lines)


def _is_inventory_question(question):
    normalized = f" {_normalize_question(question)} "
    patterns = (
        " how many files ", " how many pdf ", " how many pdfs ",
        " how many papers ", " what files ", " which files ",
        " what pdfs ", " which pdfs ", " list files ", " list pdfs ",
        " uploaded files ", " uploaded pdfs ", " uploaded papers ",
        " files i have uploaded ", " pdfs i have uploaded ",
        " papers i have uploaded ", " documents i have uploaded ",
    )
    return any(pattern in normalized for pattern in patterns)


def _social_answer(question, documents):
    normalized = _normalize_question(question)
    document_count = len(documents)

    if normalized in {"hi", "hello", "hey", "hi scholarsync", "hello scholarsync"}:
        noun = "document" if document_count == 1 else "documents"
        return (
            f"Hello! I can answer questions from the {document_count} uploaded {noun}, "
            "summarize the research, compare documents, extract supported formulas, "
            "build evidence tables, and show page-level citations."
        )

    if normalized in CAPABILITY_QUESTIONS or re.fullmatch(
        r"what\s+(?:else|more|other things)\s+(?:can|could|do)\s+you\s+do",
        normalized,
    ):
        return (
            "I am ScholarSync, an evidence-grounded assistant for your private workspace. "
            "I can summarize uploaded PDFs, answer document-specific questions, compare methods and results, "
            "extract formulas that are explicitly present, create Markdown tables, show page citations, "
            "open the cited PDF pages, copy responses, and export the conversation as a PDF."
        )

    if normalized in ACKNOWLEDGEMENTS or re.fullmatch(
        r"(?:(?:ok|okay|great|perfect|awesome|nice|cool|excellent|amazing)\s+)?"
        r"(?:thanks|thank you)(?:\s+scholarsync)?",
        normalized,
    ):
        return "You’re welcome! Ask another question whenever you’re ready."

    return None


def _intent_for(question):
    normalized = f" {_normalize_question(question)} "

    if any(term in normalized for term in (
        " formula ", " formulas ", " equation ", " equations ",
        " mathematical ", " mathematics ", " calculation ", " derive ",
    )):
        return "formula"

    if any(term in normalized for term in (
        " table ", " tabulate ", " comparison table ", " put in table ",
    )):
        return "table"

    if any(term in normalized for term in (
        " compare ", " comparison ", " difference ", " differences ",
        " versus ", " vs ", " contrast ", " across documents ",
        " across papers ",
    )):
        return "comparison"

    if any(term in normalized for term in (
        " summarize ", " summary ", " overview ", " key points ",
        " main points ", " tell me about the paper ", " about this pdf ",
        " about the document ", " what these pdfs are all about ",
        " what are these pdfs about ", " what are the pdfs about ",
        " what these papers are all about ", " what are these papers about ",
    )):
        return "summary"

    return "general"


def _history_rows(history):
    rows = []
    for row in history or []:
        role = str(row.get("role", "")).upper()
        content = str(row.get("content", "")).strip()
        if role in {"USER", "ASSISTANT"} and content:
            rows.append({"role": role, "content": content})
    return rows


def _latest_meaningful_user_question(history):
    for row in reversed(_history_rows(history)):
        if row["role"] != "USER":
            continue
        normalized = _normalize_question(row["content"])
        if normalized in ACKNOWLEDGEMENTS:
            continue
        return row["content"].strip()
    return ""


def _contextual_query(question, history, intent):
    """Resolve a terse follow-up using only the latest meaningful user turn.

    Using three or more older turns can leak a previously mentioned country or
    paper into a later document-specific request. One prior user turn is enough
    for references such as "give me the formulas" or "put this in a table".
    """
    question_terms = set(tokens(question))
    short_follow_up = len([term for term in tokens(question) if len(term) > 1]) <= 7
    continuation_intent = intent in {"formula", "table", "summary", "comparison"}
    needs_context = bool(question_terms & REFERENTIAL_TERMS) or (
        short_follow_up and continuation_intent
    )
    if not needs_context:
        return question

    prior = _latest_meaningful_user_question(history)
    if not prior:
        return question
    return f"{prior} {question}".strip()


def _title_keywords(document):
    raw_tokens = [term for term in _normalize_question(_document_title(document)).split() if not term.isdigit()]
    return [
        term for term in raw_tokens
        if len(term) >= 3 and term not in GENERIC_TITLE_TERMS
    ]


def _match_documents(text, documents):
    """Return uploaded documents explicitly named in `text`.

    Matching is based on distinctive filename/title words. A query containing
    "Tamil Nadu", "Egypt", "Brazil", "Khuzestan", etc. therefore hard-scopes
    retrieval to the matching uploaded document instead of letting BM25 mix
    evidence from other papers.
    """
    normalized = _normalize_question(text)
    padded = f" {normalized} "
    query_terms = set(normalized.split())
    matches = []

    for document in documents:
        title = _document_title(document)
        title_normalized = _normalize_question(title)
        keywords = _title_keywords(document)
        score = 0

        if title_normalized and f" {title_normalized} " in padded:
            score += 50

        # Consecutive title-keyword phrases (e.g. "tamil nadu") are strong.
        for size in (3, 2):
            for index in range(0, max(0, len(keywords) - size + 1)):
                phrase = " ".join(keywords[index:index + size])
                if phrase and f" {phrase} " in padded:
                    score += 12 * size

        overlap = query_terms.intersection(keywords)
        score += sum(3 + min(len(term), 10) / 10 for term in overlap)

        if score > 0:
            matches.append((score, document))

    # Keep every explicitly named document, not only the best one, so queries
    # such as "compare Tamil Nadu and Egypt" stay scoped to exactly those two.
    matches.sort(key=lambda row: row[0], reverse=True)
    return [document for _score, document in matches]


def _asks_for_all_documents(question):
    normalized = f" {_normalize_question(question)} "
    return any(term in normalized for term in (
        " all papers ", " all the papers ", " all pdfs ", " all the pdfs ",
        " all documents ", " all the documents ", " all 3 ", " all the 3 ", " all three ",
        " these pdfs ", " these papers ", " these documents ",
        " pdfs are all about ", " papers are all about ",
    ))


def _intent_queries(question, contextual_question, intent):
    queries = [contextual_question]
    if question.strip() != contextual_question.strip():
        queries.append(question)

    if intent == "formula":
        queries.extend([
            f"{contextual_question} formula equation mathematical expression",
            f"{contextual_question} weight index ratio matrix sum product consistency",
            f"{contextual_question} calculated computed where denotes appendix equation",
        ])
    elif intent == "table":
        queries.extend([
            f"{contextual_question} objective methodology data criteria results conclusion",
            f"{contextual_question} key information findings thresholds dataset resolution",
        ])
    elif intent == "comparison":
        queries.extend([
            f"{contextual_question} method approach data results findings limitations",
            f"{contextual_question} similarities differences contribution performance",
        ])
    elif intent == "summary":
        queries.extend([
            f"{contextual_question} abstract objective methodology results conclusion",
            f"{contextual_question} contribution findings limitations",
        ])
    else:
        queries.append(f"{contextual_question} definition method result")

    unique = []
    seen = set()
    for query in queries:
        key = " ".join(tokens(query))
        if key and key not in seen:
            seen.add(key)
            unique.append(query)
    return unique


def _formula_signal(content):
    content = str(content or "")
    # Some scientific PDFs extract the equals sign as the legacy glyph "¼"
    # (or the full-width equals sign). Normalize only for formula detection so
    # retrieval scoring sees these as equations without altering stored text.
    formula_content = content.replace("¼", "=").replace("＝", "=")
    lower = formula_content.lower()
    score = 0.0
    # Numeric min/max cells from data tables are values, not mathematical
    # formulas. Do not let a page full of ranges outrank real equations.
    scalar_assignments = len(re.findall(
        r"(?im)^\s*(?:min(?:imum)?|max(?:imum)?|mean|average|range)\s*=\s*[-+]?\d+(?:\.\d+)?",
        formula_content,
    ))
    # Equations extracted from PDFs are often imperfect; accept common ASCII,
    # Unicode and LaTeX forms as a strong signal.
    if re.search(r"(?:^|\s)[A-Za-zλΛ][A-Za-z0-9_()\-]{0,40}\s*=", formula_content):
        score += 1.25
    if re.search(r"\b(?:CI|CR|RI|PV|W|E|A|x|y|w_i|r_i)\b[^\n]{0,50}=", formula_content, re.I):
        score += 0.55
    if any(symbol in formula_content for symbol in ("∑", "Σ", "√", "≤", "≥", "∏", "λ", "∑")):
        score += 0.75
    if any(term in lower for term in (
        "equation", "formula", "calculated as", "computed as", "defined as",
        "where", "consistency ratio", "consistency index", "weighted sum",
        "appendix a", "appendix b", "appendix c", "eq.", "equation (",
    )):
        score += 0.45
    if any(term in formula_content for term in ("\\frac", "\\sum", "\\sqrt", "^", "_")):
        score += 0.45
    if scalar_assignments >= 3 and not any(term in lower for term in (
        "consistency index", "consistency ratio", "equation", "formula",
        "weighted sum", "appendix", "lambda", "eigenvalue",
    )):
        score *= 0.18
    return score


def _summary_signal(item):
    content = str(getattr(item, "content", "") or "").lower()
    heading = str(getattr(item, "section_heading", "") or "").lower()
    page = int(getattr(item, "page_number", 0) or 0)
    score = 0.0
    if page == 1:
        score += 0.18
    if any(term in f"{heading} {content[:350]}" for term in (
        "abstract", "introduction", "methodology", "results", "conclusion",
        "discussion", "objective", "contribution",
    )):
        score += 0.24
    return score


def _is_reference_heavy(content):
    lower = str(content or "").lower()
    return lower.count(" et al.") >= 5 or lower.count("doi") >= 5 or lower.count("references") >= 2


def _fused_search(queries, chunks, intent, limit=10):
    chunks = list(chunks)
    by_id = {}

    for query_index, query in enumerate(queries):
        hits = bm25_search(
            query,
            chunks,
            text_getter=lambda item: (
                f"{getattr(getattr(item, 'document', None), 'display_title', '')} "
                f"{getattr(item, 'section_heading', '')} {item.content}"
            ),
            limit=min(30, max(14, limit * 3)),
        )
        query_weight = 1.35 if query_index == 0 else 1.0
        for rank, hit in enumerate(hits, 1):
            score = query_weight / (18 + rank)
            score += min(float(hit.score), 25.0) * 0.0025
            if intent == "formula":
                score += _formula_signal(hit.item.content)
            elif intent in {"summary", "table", "comparison"}:
                score += _summary_signal(hit.item)
            if _is_reference_heavy(hit.item.content):
                score *= 0.55

            key = str(hit.item.id)
            if key not in by_id:
                by_id[key] = SearchHit(item=hit.item, score=score)
            else:
                by_id[key].score += score

    # For formula requests, independently scan the scoped document for formula
    # signals. This catches equations whose surrounding wording does not share
    # lexical terms with the user's query.
    if intent == "formula":
        for item in chunks:
            signal = _formula_signal(item.content)
            if signal <= 0:
                continue
            key = str(item.id)
            bonus = 0.55 + signal
            if key not in by_id:
                by_id[key] = SearchHit(item=item, score=bonus)
            else:
                by_id[key].score += bonus

    ranked = sorted(by_id.values(), key=lambda hit: hit.score, reverse=True)

    selected = []
    seen_terms = []
    for hit in ranked:
        terms = set(tokens(hit.item.content, remove_stopwords=True))
        if any(
            len(terms & prior) / max(1, len(terms | prior)) > 0.82
            for prior in seen_terms
        ):
            continue
        selected.append(hit)
        seen_terms.append(terms)
        if len(selected) >= limit:
            break
    return selected



def _formula_core_hits(scoped_chunks, limit):
    """Return the strongest equation-bearing chunks with page diversity.

    Formula questions need a document-wide scan, not only lexical BM25 hits.
    This helper is deliberately document-agnostic: callers have already
    restricted ``scoped_chunks`` to the requested paper(s).
    """
    candidates = []
    for chunk in list(scoped_chunks):
        content = str(getattr(chunk, "content", "") or "")
        signal = _formula_signal(content)
        if signal <= 0:
            continue

        normalized = content.replace("¼", "=").replace("＝", "=")
        lower = normalized.lower()
        score = float(signal)

        # Prefer passages that actually contain an equation over passages that
        # merely say an equation/formula exists elsewhere in the paper.
        explicit_equation = bool(
            re.search(r"(?:^|\s)[A-Za-zλΛ][A-Za-z0-9_()\-]{0,50}\s*=", normalized)
            or re.search(r"\b(?:CI|CR|RI|PV|E|W|A|x|y|w_i|r_i)\b[^\n]{0,70}=", normalized, re.I)
            or any(symbol in normalized for symbol in ("∑", "Σ", "√", "∏", "λ"))
            or any(command in normalized for command in ("\\frac", "\\sum", "\\sqrt"))
        )
        if explicit_equation:
            score += 1.35

        if any(term in lower for term in (
            "consistency index", "consistency ratio", "eigenvalue",
            "weighted sum", "suitability index", "reciprocity",
        )):
            score += 0.55

        # Appendix/formula references are useful context but should not outrank
        # an explicit equation-bearing chunk.
        if not explicit_equation and any(term in lower for term in (
            "formula", "equation", "appendix a", "appendix b", "appendix c",
        )):
            score -= 0.20

        candidates.append(SearchHit(item=chunk, score=max(score, 0.01)))

    candidates.sort(key=lambda hit: hit.score, reverse=True)
    if not candidates:
        return []

    # First take the strongest hit from each page so CI/CR on one page does not
    # crowd out a suitability-index equation on another page. Then fill any
    # remaining slots by score.
    selected = []
    seen_ids = set()
    seen_pages = set()
    for hit in candidates:
        page_key = (
            str(getattr(hit.item, "document_id", "")),
            int(getattr(hit.item, "page_number", 0) or 0),
        )
        if page_key in seen_pages:
            continue
        selected.append(hit)
        seen_ids.add(str(hit.item.id))
        seen_pages.add(page_key)
        if len(selected) >= limit:
            return selected

    for hit in candidates:
        key = str(hit.item.id)
        if key in seen_ids:
            continue
        selected.append(hit)
        seen_ids.add(key)
        if len(selected) >= limit:
            break
    return selected

def _formula_neighbor_hits(selected, scoped_chunks, limit):
    """Add nearby appendix/equation chunks from the same paper.

    PDF extraction often splits an equation from the sentence that says
    "the formulas are in Appendix A". Pulling adjacent page/chunk content keeps
    the formula and its explanation together without crossing documents.
    """
    if not selected:
        return []
    scoped_chunks = list(scoped_chunks)
    by_doc = defaultdict(list)
    for chunk in scoped_chunks:
        by_doc[str(chunk.document_id)].append(chunk)

    output = []
    seen = set()
    for hit in selected:
        key = str(hit.item.id)
        if key not in seen:
            output.append(hit)
            seen.add(key)
        document_chunks = by_doc[str(hit.item.document_id)]
        page = int(getattr(hit.item, "page_number", 0) or 0)
        chunk_index = int(getattr(hit.item, "chunk_index", 0) or 0)
        neighbors = []
        for chunk in document_chunks:
            if str(chunk.id) in seen:
                continue
            neighbor_page = int(getattr(chunk, "page_number", 0) or 0)
            neighbor_index = int(getattr(chunk, "chunk_index", 0) or 0)
            close = (
                (neighbor_page == page and abs(neighbor_index - chunk_index) <= 1)
                or abs(neighbor_page - page) == 1
            )
            if not close:
                continue
            signal = _formula_signal(chunk.content)
            if signal > 0 or any(term in str(chunk.content).lower() for term in ("appendix", "equation", "formula")):
                neighbors.append(SearchHit(item=chunk, score=float(hit.score) * 0.65 + signal))
        neighbors.sort(key=lambda row: row.score, reverse=True)
        for neighbor in neighbors[:2]:
            if str(neighbor.item.id) not in seen:
                output.append(neighbor)
                seen.add(str(neighbor.item.id))
        if len(output) >= limit:
            break
    return output[:limit]




def _merge_unique_hits(*groups, limit):
    output = []
    seen = set()
    for group in groups:
        for hit in group:
            key = str(hit.item.id)
            if key in seen:
                continue
            output.append(hit)
            seen.add(key)
            if len(output) >= limit:
                return output
    return output

def _round_robin_document_hits(document_hits, limit, per_document):
    selected = []
    round_index = 0
    doc_ids = list(document_hits.keys())
    while len(selected) < limit:
        added = False
        for doc_id in doc_ids:
            hits = document_hits[doc_id]
            if round_index < min(per_document, len(hits)):
                selected.append(hits[round_index])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        round_index += 1
    return selected


def _retrieve(question, chunks, history, intent, target_documents=None):
    chunk_list = list(chunks)
    contextual = _contextual_query(question, history, intent)
    queries = _intent_queries(question, contextual, intent)

    target_ids = {_document_id(document) for document in (target_documents or []) if _document_id(document)}
    if target_ids:
        scoped_chunks = [chunk for chunk in chunk_list if str(chunk.document_id) in target_ids]
    else:
        scoped_chunks = chunk_list

    limit = 14 if intent == "formula" else 12 if intent in {"table", "comparison", "summary"} else 9

    # Multi-document tasks must retrieve independently from each requested PDF.
    # Global BM25 can otherwise omit one entire document even when it is loaded.
    if intent in {"comparison", "summary", "table"} or (intent == "formula" and len(target_ids) > 1):
        by_document = defaultdict(list)
        for chunk in scoped_chunks:
            by_document[str(chunk.document_id)].append(chunk)
        document_hits = {}
        per_document_limit = 5 if intent == "formula" else 4
        for doc_id, doc_chunks in by_document.items():
            title = _document_title(doc_chunks[0].document) if doc_chunks else ""
            doc_queries = [f"{query} {title}" for query in queries]
            hits = _fused_search(doc_queries, doc_chunks, intent, limit=max(per_document_limit * 2, 8))
            if intent == "formula":
                core = _formula_core_hits(doc_chunks, limit=max(per_document_limit * 2, 8))
                lexical_with_neighbors = _formula_neighbor_hits(
                    hits, doc_chunks, limit=max(per_document_limit * 2, 8)
                )
                # Equation-bearing chunks come first. Lexical results and
                # neighboring appendix text supplement them rather than
                # displacing them.
                hits = _merge_unique_hits(
                    core, lexical_with_neighbors, hits,
                    limit=max(per_document_limit * 2, 8),
                )
            document_hits[doc_id] = hits
        return _round_robin_document_hits(document_hits, limit=limit, per_document=per_document_limit)

    ranked = _fused_search(queries, scoped_chunks, intent, limit=max(limit * 2, 18))
    if intent == "formula":
        core = _formula_core_hits(scoped_chunks, limit=limit)
        lexical_with_neighbors = _formula_neighbor_hits(ranked, scoped_chunks, limit=limit)
        ranked = _merge_unique_hits(core, lexical_with_neighbors, ranked, limit=limit)
    return ranked[:limit]


def answer_workspace_question(question, chunks, history=None, documents=None):
    chunk_list = list(chunks)
    history = _history_rows(history)
    workspace_documents = _workspace_documents(chunk_list, documents=documents)

    if _is_inventory_question(question):
        return {
            "answer": _inventory_answer(workspace_documents),
            "confidence": "high",
            "model": "workspace-router",
            "hits": [],
            "intent": "inventory",
        }

    social = _social_answer(question, workspace_documents)
    if social is not None:
        return {
            "answer": social,
            "confidence": "high",
            "model": "conversation-router",
            "hits": [],
            "intent": "conversation",
        }

    if not workspace_documents:
        return {
            "answer": "Upload at least one PDF before asking a research question.",
            "confidence": "low",
            "model": "workspace-router",
            "hits": [],
            "intent": "general",
        }

    intent = _intent_for(question)
    contextual = _contextual_query(question, history, intent)

    # Explicit names in the current question always win over conversation
    # context. Context is used only when the current turn is genuinely terse.
    target_documents = _match_documents(question, workspace_documents)
    if not target_documents and contextual != question:
        target_documents = _match_documents(contextual, workspace_documents)

    if _asks_for_all_documents(question):
        target_documents = list(workspace_documents)
    elif intent == "comparison" and not target_documents and len(workspace_documents) > 1:
        target_documents = list(workspace_documents)
    elif intent in {"summary", "table"} and not target_documents and len(workspace_documents) > 1:
        # A first-turn request such as "what are these PDFs about?" means all
        # workspace documents. A follow-up table/summary usually resolves via
        # contextual document matching above.
        target_documents = list(workspace_documents)

    # If the user named a real uploaded paper whose indexing is not ready, say
    # that instead of silently borrowing evidence from another document.
    not_ready = [document for document in target_documents if _document_status(document) not in {"", "READY"}]
    if not_ready:
        names = ", ".join(f"**{_document_title(document)}**" for document in not_ready)
        return {
            "answer": f"{names} is still being processed, so I cannot ground an answer in it yet.",
            "confidence": "low",
            "model": "workspace-router",
            "hits": [],
            "intent": intent,
        }

    hits = _retrieve(
        question,
        chunk_list,
        history,
        intent,
        target_documents=target_documents,
    )
    if not hits:
        if target_documents:
            names = ", ".join(f"**{_document_title(document)}**" for document in target_documents)
            answer = (
                f"I could not find supporting extracted text in {names} for that request. "
                "I will not substitute evidence from another uploaded paper."
            )
        else:
            answer = INSUFFICIENT_ANSWER
        return {
            "answer": answer,
            "confidence": "low",
            "model": "retrieval-only",
            "hits": [],
            "intent": intent,
        }

    answer_mode = {
        "formula": "workspace-formula",
        "table": "workspace-table",
        "comparison": "workspace-comparison",
        "summary": "workspace-summary",
        "general": "workspace-general",
    }[intent]

    answer, confidence, model = generate_answer(
        question,
        hits,
        conversation_context=history[-6:],
        answer_mode=answer_mode,
    )
    return {
        "answer": answer,
        "confidence": confidence,
        "model": model,
        "hits": hits,
        "intent": intent,
    }
