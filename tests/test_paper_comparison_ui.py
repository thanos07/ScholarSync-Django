from urllib.parse import parse_qs, urlparse

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from apps.conversations.models import Conversation
from apps.documents.models import Document
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
class PaperComparisonUiTests(TestCase):
    def setUp(self):
        user_model = get_user_model()
        self.user = user_model.objects.create_user(
            username="compare-owner",
            password="test-pass-123",
        )
        self.other_user = user_model.objects.create_user(
            username="other-compare-owner",
            password="test-pass-123",
        )
        self.workspace = Workspace.objects.create(
            owner=self.user,
            name="Solar Research",
        )
        self.other_workspace = Workspace.objects.create(
            owner=self.other_user,
            name="Other Research",
        )

        self.egypt = self._document(
            self.user,
            self.workspace,
            "Egypt GIS AHP",
            "a" * 64,
        )
        self.brazil = self._document(
            self.user,
            self.workspace,
            "Brazil AHP TOPSIS",
            "b" * 64,
        )
        self.other_document = self._document(
            self.other_user,
            self.other_workspace,
            "Private Other Paper",
            "c" * 64,
        )

    def _document(self, owner, workspace, title, file_hash, status=Document.Status.READY):
        return Document.objects.create(
            owner=owner,
            workspace=workspace,
            original_filename=f"{title}.pdf",
            display_title=title,
            storage_path=f"documents/{file_hash[:8]}.pdf",
            file_hash=file_hash,
            file_size=100,
            page_count=10,
            processing_status=status,
        )

    def test_comparison_creation_requires_login(self):
        response = self.client.post(
            reverse("conversation-compare-new", args=[self.workspace.id]),
            {"documents": [self.egypt.id, self.brazil.id]},
        )
        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse("login"), response.url)

    def test_at_least_two_ready_documents_are_required(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("conversation-compare-new", args=[self.workspace.id]),
            {"documents": [self.egypt.id]},
        )

        self.assertRedirects(
            response,
            reverse("workspace-detail", args=[self.workspace.id]),
        )
        self.assertFalse(Conversation.objects.exists())

    def test_cross_workspace_document_is_not_accepted(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("conversation-compare-new", args=[self.workspace.id]),
            {"documents": [self.egypt.id, self.other_document.id]},
        )

        self.assertRedirects(
            response,
            reverse("workspace-detail", args=[self.workspace.id]),
        )
        self.assertFalse(Conversation.objects.exists())

    def test_valid_selection_creates_prefilled_comparison_conversation(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("conversation-compare-new", args=[self.workspace.id]),
            {"documents": [self.egypt.id, self.brazil.id]},
        )

        self.assertEqual(response.status_code, 302)
        conversation = Conversation.objects.get()
        self.assertEqual(conversation.owner, self.user)
        self.assertEqual(conversation.workspace, self.workspace)
        self.assertEqual(conversation.title, "Paper comparison")

        query = parse_qs(urlparse(response.url).query)
        question = query["question"][0]
        self.assertIn("Egypt GIS AHP", question)
        self.assertIn("Brazil AHP TOPSIS", question)
        self.assertIn("objective", question)
        self.assertIn("limitations", question)

    def test_prefill_query_populates_empty_conversation_composer(self):
        self.client.force_login(self.user)
        conversation = Conversation.objects.create(
            owner=self.user,
            workspace=self.workspace,
            title="Paper comparison",
        )
        question = "Compare Egypt GIS AHP and Brazil AHP TOPSIS."

        response = self.client.get(
            reverse("conversation-detail", args=[conversation.id]),
            {"question": question},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.context["form"].initial["question"],
            question,
        )
