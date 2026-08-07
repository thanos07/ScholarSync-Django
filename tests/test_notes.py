from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from apps.documents.models import Document
from apps.notes.models import Note
from apps.workspaces.models import Workspace


@override_settings(
    STORAGES={
        "default": {
            "BACKEND": "django.core.files.storage.FileSystemStorage",
        },
        "staticfiles": {
            "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage",
        },
    }
)
class NoteCreationTests(TestCase):
    def setUp(self):
        user_model = get_user_model()
        self.user = user_model.objects.create_user(
            username="note-owner",
            password="test-pass-123",
        )
        self.other_user = user_model.objects.create_user(
            username="other-note-owner",
            password="test-pass-123",
        )

        self.workspace = Workspace.objects.create(
            owner=self.user,
            name="Solar Research",
        )
        self.other_workspace = Workspace.objects.create(
            owner=self.other_user,
            name="Other Workspace",
        )
        self.second_workspace = Workspace.objects.create(
            owner=self.user,
            name="Second Workspace",
        )

        self.document = Document.objects.create(
            owner=self.user,
            workspace=self.workspace,
            original_filename="egypt.pdf",
            display_title="Egypt GIS AHP",
            storage_path="documents/egypt.pdf",
            file_hash="a" * 64,
            file_size=100,
            page_count=12,
            processing_status=Document.Status.READY,
        )
        self.second_document = Document.objects.create(
            owner=self.user,
            workspace=self.second_workspace,
            original_filename="brazil.pdf",
            display_title="Brazil AHP TOPSIS",
            storage_path="documents/brazil.pdf",
            file_hash="b" * 64,
            file_size=100,
            page_count=9,
            processing_status=Document.Status.READY,
        )

    def test_note_list_requires_login(self):
        response = self.client.get(reverse("note-list"))
        self.assertEqual(response.status_code, 302)

    def test_user_can_create_page_linked_note(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("note-list"),
            {
                "workspace": self.workspace.id,
                "document": self.document.id,
                "page_number": 6,
                "title": "Consistency formula",
                "content": "Recheck the CI and CR definitions before writing.",
                "tags": "AHP,formula",
            },
        )

        self.assertRedirects(response, reverse("note-list"))
        note = Note.objects.get()
        self.assertEqual(note.owner, self.user)
        self.assertEqual(note.workspace, self.workspace)
        self.assertEqual(note.document, self.document)
        self.assertEqual(note.page_number, 6)

    def test_user_cannot_use_another_users_workspace(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("note-list"),
            {
                "workspace": self.other_workspace.id,
                "title": "Should fail",
                "content": "This workspace is not owned by the signed-in user.",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(Note.objects.exists())
        self.assertIn("workspace", response.context["form"].errors)

    def test_document_must_match_selected_workspace(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("note-list"),
            {
                "workspace": self.workspace.id,
                "document": self.second_document.id,
                "title": "Mismatched document",
                "content": "This document belongs to another workspace.",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(Note.objects.exists())
        self.assertIn("document", response.context["form"].errors)

    def test_page_number_cannot_exceed_document_page_count(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("note-list"),
            {
                "workspace": self.workspace.id,
                "document": self.document.id,
                "page_number": 99,
                "title": "Invalid page",
                "content": "The page should be rejected.",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(Note.objects.exists())
        self.assertIn("page_number", response.context["form"].errors)

    def test_note_list_is_owner_scoped(self):
        own_note = Note.objects.create(
            owner=self.user,
            workspace=self.workspace,
            title="My note",
            content="Visible to the owner.",
        )
        Note.objects.create(
            owner=self.other_user,
            workspace=self.other_workspace,
            title="Private other note",
            content="Must not be visible.",
        )

        self.client.force_login(self.user)
        response = self.client.get(reverse("note-list"))

        self.assertContains(response, own_note.title)
        self.assertNotContains(response, "Private other note")
