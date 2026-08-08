from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render
from .forms import WorkspaceForm
from .models import Workspace

@login_required
def workspace_list(request):
    return render(request, "workspaces/list.html", {"workspaces": Workspace.objects.filter(owner=request.user)})

@login_required
def workspace_create(request):
    form = WorkspaceForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        workspace = form.save(commit=False)
        workspace.owner = request.user
        workspace.save()
        return redirect("workspace-detail", workspace_id=workspace.id)
    return render(request, "workspaces/form.html", {"form": form})

@login_required
def workspace_detail(request, workspace_id):
    workspace = get_object_or_404(Workspace, id=workspace_id, owner=request.user)
    documents = list(workspace.documents.order_by("-uploaded_at"))
    ready_documents = [
        document for document in documents if document.processing_status == "READY"
    ]
    return render(request, "workspaces/detail.html", {
        "workspace": workspace,
        "documents": documents,
        "ready_documents": ready_documents,
        "conversations": workspace.conversations.order_by("-updated_at")[:8],
    })
