import re
import time
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from apps.rag.orchestrator import answer_workspace_question
from apps.workspaces.models import Workspace
from .forms import QuestionForm
from .models import Citation, Conversation, Message
from .pdf_export import build_conversation_pdf

@login_required
def conversation_new(request, workspace_id):
    workspace = get_object_or_404(Workspace, id=workspace_id, owner=request.user)
    conversation = Conversation.objects.create(workspace=workspace, owner=request.user)
    return redirect("conversation-detail", conversation_id=conversation.id)

@login_required
def conversation_detail(request, conversation_id):
    conversation = get_object_or_404(Conversation.objects.select_related("workspace"), id=conversation_id, owner=request.user)
    form = QuestionForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        question = form.cleaned_data["question"].strip()
        if not conversation.messages.exists():
            conversation.title = question[:90]
            conversation.save(update_fields=["title", "updated_at"])
        Message.objects.create(conversation=conversation, role=Message.Role.USER, content=question)
        started = time.perf_counter()
        result = answer_workspace_question(question, conversation.workspace.chunks.filter(owner=request.user).select_related("document"))
        assistant = Message.objects.create(
            conversation=conversation, role=Message.Role.ASSISTANT, content=result["answer"],
            model_name=result["model"], confidence=result["confidence"],
            response_time_ms=int((time.perf_counter()-started)*1000),
        )
        for number, hit in enumerate(result["hits"], 1):
            chunk = hit.item
            Citation.objects.create(message=assistant, chunk=chunk, document=chunk.document,
                page_number=chunk.page_number, citation_number=number,
                quoted_passage=chunk.content[:700], retrieval_score=hit.score)
        return redirect("conversation-detail", conversation_id=conversation.id)
    return render(request, "conversations/detail.html", {"conversation": conversation, "messages_list": conversation.messages.prefetch_related("citations__document"), "form": form})

@login_required
def export_pdf(request, conversation_id):
    conversation = get_object_or_404(Conversation.objects.select_related("workspace"), id=conversation_id, owner=request.user)
    pdf = build_conversation_pdf(conversation, conversation.messages.prefetch_related("citations__document"))
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", conversation.title).strip("-")[:80] or "scholarsync-conversation"
    response = HttpResponse(pdf, content_type="application/pdf")
    response["Content-Disposition"] = f'attachment; filename="{safe}.pdf"'
    return response
