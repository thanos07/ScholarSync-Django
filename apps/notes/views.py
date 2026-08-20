from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render

from .forms import NoteForm


@login_required
def note_list(request):
    form = NoteForm(request.POST or None, owner=request.user)

    if request.method == "POST" and form.is_valid():
        note = form.save(commit=False)
        note.owner = request.user
        note.save()
        messages.success(request, "Note saved.")
        return redirect("note-list")

    notes = request.user.notes.select_related("workspace", "document").order_by("-updated_at")
    return render(
        request,
        "notes/list.html",
        {
            "notes": notes,
            "form": form,
        },
    )
