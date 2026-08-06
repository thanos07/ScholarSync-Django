from django.contrib.auth.decorators import login_required
from django.shortcuts import render
@login_required
def note_list(request):
    return render(request, "notes/list.html", {"notes": request.user.notes.select_related("workspace", "document")})
