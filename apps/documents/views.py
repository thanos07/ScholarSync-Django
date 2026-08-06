from io import BytesIO
from pathlib import Path
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.exceptions import ValidationError
from django.http import FileResponse, Http404
from django.shortcuts import get_object_or_404, redirect, render
from apps.workspaces.models import Workspace
from .forms import DocumentUploadForm
from .models import Document
from .services import index_document, remove_document
from .storage import get_document_storage
from .validators import validate_pdf

@login_required
def upload_document(request, workspace_id):
    workspace = get_object_or_404(Workspace, id=workspace_id, owner=request.user)
    form = DocumentUploadForm(request.POST or None, request.FILES or None)
    if request.method == "POST" and form.is_valid():
        if request.user.documents.count() >= settings.MAX_DOCUMENTS_PER_USER:
            form.add_error("pdf", "Your portfolio account has reached its document limit.")
        else:
            uploaded = form.cleaned_data["pdf"]
            try:
                result = validate_pdf(uploaded)
                if Document.objects.filter(owner=request.user, file_hash=result["sha256"]).exists():
                    form.add_error("pdf", "This document is already in your workspace.")
                else:
                    document = Document.objects.create(
                        owner=request.user, workspace=workspace,
                        original_filename=Path(uploaded.name).name,
                        display_title=Path(uploaded.name).stem,
                        storage_path="pending", file_hash=result["sha256"],
                        file_size=len(result["bytes"]), page_count=result["page_count"],
                    )
                    path = f"users/{request.user.id}/workspaces/{workspace.id}/documents/{document.id}.pdf"
                    get_document_storage().save(path, result["bytes"])
                    document.storage_path = path
                    document.save(update_fields=["storage_path"])
                    index_document(document, result["bytes"])
                    messages.success(request, "PDF uploaded and indexed.")
                    return redirect("workspace-detail", workspace_id=workspace.id)
            except ValidationError as exc:
                form.add_error("pdf", exc)
            except Exception:
                form.add_error("pdf", "The document could not be processed. Please try another PDF.")
    return render(request, "documents/upload.html", {"form": form, "workspace": workspace})

@login_required
def view_document(request, document_id):
    document = get_object_or_404(Document, id=document_id, owner=request.user)
    try:
        data = get_document_storage().read(document.storage_path)
    except Exception as exc:
        raise Http404("Document file unavailable") from exc
    return FileResponse(BytesIO(data), content_type="application/pdf", filename=document.original_filename)

@login_required
def delete_document(request, document_id):
    document = get_object_or_404(Document, id=document_id, owner=request.user)
    workspace_id = document.workspace_id
    if request.method == "POST":
        remove_document(document)
        messages.success(request, "Document deleted with its indexed chunks.")
    return redirect("workspace-detail", workspace_id=workspace_id)
