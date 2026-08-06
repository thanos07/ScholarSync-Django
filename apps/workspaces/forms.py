from django import forms
from .models import Workspace
class WorkspaceForm(forms.ModelForm):
    class Meta:
        model = Workspace
        fields = ["name", "description"]
        widgets = {"description": forms.Textarea(attrs={"rows": 3})}
