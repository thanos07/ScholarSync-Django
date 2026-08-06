from django.contrib.auth import login
from django.shortcuts import redirect, render
from .forms import RegistrationForm

def home(request):
    return render(request, "core/home.html")

def register(request):
    if request.user.is_authenticated:
        return redirect("workspace-list")
    form = RegistrationForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        user = form.save()
        login(request, user)
        return redirect("workspace-list")
    return render(request, "registration/register.html", {"form": form})

def health(request):
    from django.http import JsonResponse
    return JsonResponse({"status": "ok", "service": "scholarsync"})
