import logging
import re
import time
from urllib.parse import urlencode
from uuid import UUID

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone

from apps.notes.chat_actions import (
    build_chat_note_draft,
    chat_note_confirmation,
    is_chat_note_request,
)
from apps.notes.models import Note
from apps.rag.orchestrator import answer_workspace_question
from apps.workspaces.models import Workspace
from .citation_verification import verification_label, verify_citation_support
from .forms import QuestionForm
from .models import Citation, Conversation, Message
from .pdf_export import build_conversation_pdf


logger = logging.getLogger(__name__)

CITATION_RE = re.compile(r"\[(\d+)\]")
INSUFFICIENT_RE = re.compile(
    r"(?:could not find enough|does not contain|do not contain|not present in|"
    r"missing from|insufficient evidence|no relevant evidence)",
    re.I,
)


def _is_ajax(request):
    return request.headers.get("x-requested-with") == "XMLHttpRequest"


def _conversation_history(conversation, limit=8):
    rows = list(
        conversation.messages.order_by("-created_at").values("role", "content")[:limit]
    )
    rows.reverse()
    return rows


def _selected_hit_numbers(answer, hits):
    numbers = []
    for match in CITATION_RE.finditer(answer or ""):
        number = int(match.group(1))
        if 1 <= number <= len(hits) and number not in numbers:
            numbers.append(number)

    # Only show evidence that the answer explicitly cited. Attaching the first
    # three retrieved chunks to an uncited answer can make an Egypt passage look
    # like support for a Tamil Nadu claim (or vice versa).
    return numbers


def _citation_payload(number, hit, answer):
    chunk = hit.item
    verification_status = verify_citation_support(
        answer,
        number,
        chunk.content,
    )
    return {
        "number": number,
        "document_id": str(chunk.document_id),
        "source": chunk.document.display_title,
        "page": chunk.page_number,
        "excerpt": chunk.content[:520],
        "score": round(float(hit.score), 3),
        "verification_status": verification_status,
        "verification_label": verification_label(verification_status),
        "paper_url": f"{reverse('document-view', args=[chunk.document_id])}#page={chunk.page_number}",
    }


def _citations_for_result(result):
    hits = result.get("hits") or []
    answer = result.get("answer", "")
    numbers = _selected_hit_numbers(answer, hits)
    return [
        (
            number,
            hits[number - 1],
            _citation_payload(number, hits[number - 1], answer),
        )
        for number in numbers
    ]


@login_required
def conversation_new(request, workspace_id):
    workspace = get_object_or_404(Workspace, id=workspace_id, owner=request.user)
    conversation = Conversation.objects.create(workspace=workspace, owner=request.user)
    return redirect("conversation-detail", conversation_id=conversation.id)


@login_required
def conversation_compare_new(request, workspace_id):
    workspace = get_object_or_404(Workspace, id=workspace_id, owner=request.user)
    if request.method != "POST":
        return redirect("workspace-detail", workspace_id=workspace.id)

    selected_ids = []
    for value in request.POST.getlist("documents"):
        try:
            selected_ids.append(UUID(str(value)))
        except (TypeError, ValueError, AttributeError):
            continue

    documents = list(
        workspace.documents.filter(
            owner=request.user,
            processing_status="READY",
            id__in=selected_ids,
        ).order_by("uploaded_at")
    )

    if len(documents) < 2:
        messages.error(request, "Select at least two ready papers to compare.")
        return redirect("workspace-detail", workspace_id=workspace.id)

    quoted_titles = "; ".join(f'"{document.display_title}"' for document in documents)
    question = (
        f"Compare these uploaded papers: {quoted_titles}. "
        "Compare their objective, methodology, data or criteria, main findings, and limitations. "
        "Keep the papers distinct and cite every factual comparison."
    )
    conversation = Conversation.objects.create(
        workspace=workspace,
        owner=request.user,
        title="Paper comparison",
    )
    query = urlencode({"question": question})
    return redirect(f"{reverse('conversation-detail', args=[conversation.id])}?{query}")


@login_required
def conversation_detail(request, conversation_id):
    conversation = get_object_or_404(
        Conversation.objects.select_related("workspace"),
        id=conversation_id,
        owner=request.user,
    )
    initial_question = ""
    if request.method == "GET" and not conversation.messages.exists():
        initial_question = request.GET.get("question", "").strip()[:2000]
    form = QuestionForm(
        request.POST or None,
        initial={"question": initial_question} if initial_question else None,
    )

    if request.method == "POST":
        if not form.is_valid():
            if _is_ajax(request):
                error = form.errors.get("question", ["Please enter a valid question."])[0]
                return JsonResponse({"ok": False, "error": error}, status=400)
        else:
            question = form.cleaned_data["question"].strip()
            history = _conversation_history(conversation)
            documents = list(
                conversation.workspace.documents.filter(owner=request.user).order_by("uploaded_at")
            )
            chunks = conversation.workspace.chunks.filter(
                owner=request.user,
                document__processing_status="READY",
            ).select_related("document")

            started = time.perf_counter()
            note_draft = None
            try:
                if is_chat_note_request(question):
                    note_draft = build_chat_note_draft(
                        conversation,
                        question,
                        documents,
                    )
                    if note_draft is None:
                        result = {
                            "answer": (
                                "I could not find an earlier research answer in this "
                                "conversation to save as a note. Ask a research question "
                                "first, then say `save this as a note`."
                            ),
                            "confidence": "low",
                            "model": "note-action",
                            "hits": [],
                            "intent": "note",
                        }
                    else:
                        result = {
                            "answer": chat_note_confirmation(note_draft),
                            "confidence": "high",
                            "model": "note-action",
                            "hits": [],
                            "intent": "note",
                        }
                    citation_rows = []
                else:
                    result = answer_workspace_question(
                        question,
                        chunks,
                        history=history,
                        documents=documents,
                    )
                    citation_rows = _citations_for_result(result)
            except Exception:
                logger.exception("Private workspace question failed")
                if _is_ajax(request):
                    return JsonResponse(
                        {
                            "ok": False,
                            "error": "ScholarSync could not complete this request. Check the server terminal for details.",
                        },
                        status=500,
                    )
                raise

            with transaction.atomic():
                is_first_message = not conversation.messages.exists()
                Message.objects.create(
                    conversation=conversation,
                    role=Message.Role.USER,
                    content=question,
                )
                assistant = Message.objects.create(
                    conversation=conversation,
                    role=Message.Role.ASSISTANT,
                    content=result["answer"],
                    model_name=result["model"],
                    confidence=result["confidence"],
                    response_time_ms=int((time.perf_counter() - started) * 1000),
                )

                if note_draft is not None:
                    Note.objects.create(
                        owner=request.user,
                        workspace=conversation.workspace,
                        document=note_draft["document"],
                        page_number=note_draft["page_number"],
                        title=note_draft["title"],
                        content=note_draft["content"],
                        tags=note_draft["tags"],
                    )

                for number, hit, _payload in citation_rows:
                    chunk = hit.item
                    Citation.objects.create(
                        message=assistant,
                        chunk=chunk,
                        document=chunk.document,
                        page_number=chunk.page_number,
                        citation_number=number,
                        quoted_passage=chunk.content[:700],
                        retrieval_score=float(hit.score),
                        verification_status=_payload["verification_status"],
                    )

                update_fields = ["updated_at"]
                conversation.updated_at = timezone.now()
                if is_first_message:
                    conversation.title = question[:90]
                    update_fields.append("title")
                conversation.save(update_fields=update_fields)

            payloads = [payload for _number, _hit, payload in citation_rows]
            if _is_ajax(request):
                return JsonResponse(
                    {
                        "ok": True,
                        "question": question,
                        "answer": result["answer"],
                        "citations": payloads,
                        "model": result["model"],
                        "confidence": result["confidence"],
                    }
                )
            return redirect("conversation-detail", conversation_id=conversation.id)

    messages_list = list(
        conversation.messages.prefetch_related("citations__document").all()
    )
    latest_citations = []
    for message in reversed(messages_list):
        if message.role == Message.Role.ASSISTANT:
            latest_citations = list(message.citations.all())
            break

    return render(
        request,
        "conversations/detail.html",
        {
            "conversation": conversation,
            "messages_list": messages_list,
            "latest_citations": latest_citations,
            "form": form,
        },
    )


@login_required
def export_pdf(request, conversation_id):
    conversation = get_object_or_404(
        Conversation.objects.select_related("workspace"),
        id=conversation_id,
        owner=request.user,
    )
    pdf = build_conversation_pdf(
        conversation,
        conversation.messages.prefetch_related("citations__document"),
    )
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", conversation.title).strip("-")[:80]
    safe = safe or "scholarsync-conversation"
    response = HttpResponse(pdf, content_type="application/pdf")
    response["Content-Disposition"] = f'attachment; filename="{safe}.pdf"'
    return response
