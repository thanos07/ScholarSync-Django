from django import forms

from apps.documents.models import Document
from apps.workspaces.models import Workspace

from .models import Note


class NoteForm(forms.ModelForm):
    class Meta:
        model = Note
        fields = ["workspace", "document", "page_number", "title", "content", "tags"]
        widgets = {
            "page_number": forms.NumberInput(attrs={"min": 1, "placeholder": "Optional page"}),
            "title": forms.TextInput(attrs={"placeholder": "Short note title"}),
            "content": forms.Textarea(
                attrs={
                    "rows": 6,
                    "placeholder": "Write the research note you want to keep...",
                }
            ),
            "tags": forms.TextInput(
                attrs={"placeholder": "Optional tags, comma separated"}
            ),
        }

    def __init__(self, *args, owner=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.owner = owner

        self.fields["workspace"].queryset = Workspace.objects.none()
        self.fields["document"].queryset = Document.objects.none()
        self.fields["document"].required = False
        self.fields["page_number"].required = False

        if owner is not None:
            self.fields["workspace"].queryset = Workspace.objects.filter(owner=owner)
            self.fields["document"].queryset = (
                Document.objects.filter(owner=owner)
                .select_related("workspace")
                .order_by("workspace__name", "display_title")
            )
            self.fields["document"].label_from_instance = (
                lambda document: f"{document.display_title} — {document.workspace.name}"
            )

    def clean(self):
        cleaned = super().clean()
        workspace = cleaned.get("workspace")
        document = cleaned.get("document")
        page_number = cleaned.get("page_number")

        if self.owner is not None:
            if workspace is not None and workspace.owner_id != self.owner.id:
                self.add_error("workspace", "Choose one of your own workspaces.")
            if document is not None and document.owner_id != self.owner.id:
                self.add_error("document", "Choose one of your own documents.")

        if document is not None and workspace is not None:
            if document.workspace_id != workspace.id:
                self.add_error(
                    "document",
                    "The selected document must belong to the selected workspace.",
                )

        if page_number and document is None:
            self.add_error(
                "page_number",
                "Choose a document before attaching a page number.",
            )
        elif (
            page_number
            and document is not None
            and document.page_count
            and page_number > document.page_count
        ):
            self.add_error(
                "page_number",
                f"This document has {document.page_count} pages.",
            )

        return cleaned
