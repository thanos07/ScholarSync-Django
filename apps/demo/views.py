import re
import uuid
from types import SimpleNamespace

from django.conf import settings
from django.core.cache import cache
from django.http import FileResponse, Http404, HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils import timezone

from apps.conversations.citation_verification import (
    verification_label,
    verify_citation_support,
)
from apps.conversations.pdf_export import build_conversation_pdf

from .services import (
    answer_demo,
    get_demo_document,
    load_demo_documents,
)


DEMO_TTL_SECONDS = 2 * 60 * 60
CITATION_RE = re.compile(r"\[(\d+)\]")


def _history_key(request):
    demo_id = request.session.get("demo_id")

    if not demo_id:
        demo_id = uuid.uuid4().hex
        request.session["demo_id"] = demo_id

    return f"scholarsync-demo:{demo_id}"


def _get_history(request):
    return cache.get(
        _history_key(request),
        [],
    )


def _set_history(request, history):
    cache.set(
        _history_key(request),
        history,
        timeout=DEMO_TTL_SECONDS,
    )


def _latest_demo_citations(history):
    """
    Return citations from the latest assistant answer only.

    This keeps the right-hand evidence viewer aligned with
    the private workspace behaviour.
    """
    for row in reversed(history):
        if row.get("role") == "ASSISTANT":
            return row.get("citations") or []

    return []


def _is_ajax(request):
    return (
        request.headers.get("x-requested-with")
        == "XMLHttpRequest"
    )


def _citations_for_result(result):
    """
    Build citation payloads only for evidence explicitly
    cited by the generated answer.

    Do not attach arbitrary retrieval hits to an uncited
    answer, because that can make unrelated evidence appear
    to support a claim.
    """
    hits = result.get("hits") or []
    answer = result.get("answer") or ""

    cited_numbers = []

    for match in CITATION_RE.finditer(answer):
        number = int(match.group(1))

        if (
            1 <= number <= len(hits)
            and number not in cited_numbers
        ):
            cited_numbers.append(number)

    citations = []

    for number in cited_numbers:
        hit = hits[number - 1]
        item = hit.item

        verification_status = (
            verify_citation_support(
                answer,
                number,
                item.content,
            )
        )

        citations.append(
            {
                "number": number,
                "document_id": item.document_id,
                "source": item.source,
                "page": item.page,
                "excerpt": item.content[:520],
                "score": round(
                    float(hit.score),
                    3,
                ),
                "verification_status": (
                    verification_status
                ),
                "verification_label": (
                    verification_label(
                        verification_status
                    )
                ),
                "paper_url": (
                    f"{reverse('demo-paper', args=[item.document_id])}"
                    f"#page={item.page}"
                ),
            }
        )

    return citations


def demo_home(request):
    history = _get_history(request)

    if request.method == "POST":
        question = (
            request.POST
            .get("question", "")
            .strip()
        )

        if not question:
            if _is_ajax(request):
                return JsonResponse(
                    {
                        "ok": False,
                        "error": "Please enter a question.",
                    },
                    status=400,
                )

            return redirect("demo-home")

        try:
            result = answer_demo(
                question,
                history=history,
            )

            citations = _citations_for_result(
                result
            )

            assistant_row = {
                "role": "ASSISTANT",
                "content": result["answer"],
                "citations": citations,
                "model": result["model"],
                "confidence": result["confidence"],
            }

            history.extend(
                [
                    {
                        "role": "USER",
                        "content": question,
                        "citations": [],
                    },
                    assistant_row,
                ]
            )

            # Keep the demo history bounded.
            history = history[-20:]

            _set_history(
                request,
                history,
            )

            if _is_ajax(request):
                return JsonResponse(
                    {
                        "ok": True,
                        "question": question,
                        "answer": assistant_row[
                            "content"
                        ],
                        "citations": citations,
                        "model": assistant_row[
                            "model"
                        ],
                        "confidence": assistant_row[
                            "confidence"
                        ],
                    }
                )

            return redirect(
                f"{reverse('demo-home')}?scroll=latest"
            )

        except Exception:
            # Full exception remains available in
            # the Django development-server terminal.
            if _is_ajax(request):
                return JsonResponse(
                    {
                        "ok": False,
                        "error": (
                            "ScholarSync could not complete "
                            "this request. Check the server "
                            "terminal for details."
                        ),
                    },
                    status=500,
                )

            raise

    return render(
        request,
        "demo/home.html",
        {
            "history": history,
            "documents": load_demo_documents(),
            "latest_citations": (
                _latest_demo_citations(
                    history
                )
            ),
        },
    )


def demo_clear(request):
    cache.delete(
        _history_key(request)
    )

    return redirect(
        "demo-home"
    )


def demo_paper(
    request,
    document_id,
):
    document = get_demo_document(
        document_id
    )

    if document is None:
        raise Http404(
            "Demo paper was not found."
        )

    path = (
        settings.DEMO_DATA_DIR
        / "papers"
        / document.filename
    )

    if not path.exists():
        return HttpResponse(
            "Demo paper is unavailable.",
            status=404,
        )

    return FileResponse(
        path.open("rb"),
        content_type="application/pdf",
        filename=document.filename,
    )


def demo_export_pdf(request):
    """
    Adapt the cached demo conversation to the same interface
    expected by the production conversation PDF exporter.

    The demo intentionally does not persist Conversation,
    Message or Citation database rows.
    """
    history = _get_history(request)

    demo_documents = load_demo_documents()

    # build_conversation_pdf() expects:
    #
    # conversation.workspace.documents.count()
    #
    # The public demo uses an in-memory/JSON-backed library,
    # therefore expose the equivalent interface here.
    document_manager = SimpleNamespace(
        count=lambda: len(
            demo_documents
        )
    )

    conversation = SimpleNamespace(
        title="ScholarSync Public Demo",
        workspace=SimpleNamespace(
            name=(
                "Curated attention "
                "research library"
            ),
            documents=document_manager,
        ),
        updated_at=timezone.now(),
    )

    messages = []

    for row in history:
        citations = []

        for citation in row.get(
            "citations",
            [],
        ):
            citations.append(
                SimpleNamespace(
                    citation_number=(
                        citation["number"]
                    ),

                    document=SimpleNamespace(
                        display_title=(
                            citation["source"]
                        )
                    ),

                    page_number=(
                        citation["page"]
                    ),

                    # Required by the redesigned
                    # evidence-card PDF output.
                    quoted_passage=(
                        citation.get(
                            "excerpt",
                            "",
                        )
                    ),

                    retrieval_score=float(
                        citation.get(
                            "score",
                            0,
                        )
                        or 0
                    ),

                    verification_status=(
                        citation.get(
                            "verification_status",
                            "UNCHECKED",
                        )
                    ),
                )
            )

        manager = SimpleNamespace(
            select_related=(
                lambda *args,
                _citations=citations:
                SimpleNamespace(
                    all=lambda: _citations
                )
            )
        )

        messages.append(
            SimpleNamespace(
                role=row["role"],
                content=row["content"],
                citations=manager,
            )
        )

    pdf = build_conversation_pdf(
        conversation,
        messages,
    )

    response = HttpResponse(
        pdf,
        content_type="application/pdf",
    )

    response["Content-Disposition"] = (
        'attachment; '
        'filename="scholarsync-demo-conversation.pdf"'
    )

    return response