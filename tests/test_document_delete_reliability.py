from unittest.mock import Mock, patch

from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase, override_settings

from apps.documents.models import Document
from apps.documents.services import remove_document
from apps.documents.storage import SupabasePrivateStorage
from apps.workspaces.models import Workspace


@override_settings(
    SUPABASE_URL="https://example.supabase.co",
    SUPABASE_SERVICE_ROLE_KEY="test-service-role",
    SUPABASE_STORAGE_BUCKET="research-pdfs",
)
class SupabaseDeleteRequestTests(SimpleTestCase):
    @patch("apps.documents.storage.httpx.request")
    def test_delete_uses_request_with_json_body(self, request_mock):
        response = Mock()
        response.raise_for_status.return_value = None
        request_mock.return_value = response

        storage = SupabasePrivateStorage()

        path = (
            "users/1/workspaces/2/documents/"
            "example-document.pdf"
        )

        storage.delete(path)

        request_mock.assert_called_once_with(
            "DELETE",
            "https://example.supabase.co/storage/v1/object/research-pdfs",
            headers={
                "Authorization": "Bearer test-service-role",
                "apikey": "test-service-role",
                "Content-Type": "application/json",
            },
            json={"prefixes": [path]},
            timeout=30,
        )

        response.raise_for_status.assert_called_once_with()

    @patch("apps.documents.storage.httpx.request")
    def test_delete_skips_pending_storage_path(self, request_mock):
        storage = SupabasePrivateStorage()

        storage.delete("pending")

        request_mock.assert_not_called()


class RemoveDocumentReliabilityTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="delete-test-user",
            password="test-password",
        )

        self.workspace = Workspace.objects.create(
            owner=self.user,
            name="Delete Test Workspace",
        )

    def _create_document(self, *, status=Document.Status.READY):
        return Document.objects.create(
            owner=self.user,
            workspace=self.workspace,
            original_filename="paper.pdf",
            display_title="paper",
            storage_path=(
                f"users/{self.user.id}/workspaces/"
                f"{self.workspace.id}/documents/paper.pdf"
            ),
            file_hash="a" * 64,
            file_size=100,
            page_count=1,
            processing_status=status,
        )

    @patch("apps.documents.services.get_document_storage")
    def test_failed_remote_delete_does_not_leave_document_deleting(
        self,
        get_storage_mock,
    ):
        document = self._create_document()

        storage = Mock()
        storage.delete.side_effect = RuntimeError(
            "Supabase delete failed"
        )
        get_storage_mock.return_value = storage

        with self.assertRaisesRegex(
            RuntimeError,
            "Supabase delete failed",
        ):
            remove_document(document)

        document.refresh_from_db()

        self.assertEqual(
            document.processing_status,
            Document.Status.READY,
        )
        self.assertIn(
            "Supabase delete failed",
            document.processing_error,
        )

    @patch("apps.documents.services.get_document_storage")
    def test_successful_remote_delete_removes_database_row(
        self,
        get_storage_mock,
    ):
        document = self._create_document()
        document_id = document.id
        storage_path = document.storage_path

        storage = Mock()
        get_storage_mock.return_value = storage

        remove_document(document)

        storage.delete.assert_called_once_with(storage_path)

        self.assertFalse(
            Document.objects.filter(id=document_id).exists()
        )