from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.conversations.models import Citation, Conversation, Message
from apps.documents.models import Document, DocumentChunk
from apps.notes.chat_actions import is_chat_note_request
from apps.notes.models import Note
from apps.workspaces.models import Workspace


class ChatNoteCreationTests(TestCase):
    def setUp(self):
        user_model = get_user_model()
        self.user = user_model.objects.create_user(
            username="chat-note-owner",
            password="test-pass-123",
        )
        self.workspace = Workspace.objects.create(
            owner=self.user,
            name="Solar Research",
        )
        self.document = Document.objects.create(
            owner=self.user,
            workspace=self.workspace,
            original_filename="15_Turkey_AHP.pdf",
            display_title="15_Turkey_AHP",
            storage_path="documents/turkey.pdf",
            file_hash="c" * 64,
            file_size=100,
            page_count=18,
            processing_status=Document.Status.READY,
        )
        self.conversation = Conversation.objects.create(
            owner=self.user,
            workspace=self.workspace,
            title="Turkey methodology",
        )

        self.method_user = Message.objects.create(
            conversation=self.conversation,
            role=Message.Role.USER,
            content="What methods are used in the Turkey AHP paper?",
        )
        self.method_answer = Message.objects.create(
            conversation=self.conversation,
            role=Message.Role.ASSISTANT,
            content=(
                "## Methodology supported by the paper\n\n"
                "The Turkey AHP study uses GIS with fuzzy analytic hierarchy "
                "process (FAHP), fuzzy pairwise comparisons, and defuzzification "
                "to derive criteria weights. [1][2]"
            ),
            model_name="test-model",
            confidence="high",
        )

        chunk_page_2 = DocumentChunk.objects.create(
            owner=self.user,
            workspace=self.workspace,
            document=self.document,
            page_number=2,
            chunk_index=0,
            section_heading="Methodology",
            content="The study uses GIS and FAHP for site suitability analysis.",
            content_hash="d" * 64,
        )
        chunk_page_13 = DocumentChunk.objects.create(
            owner=self.user,
            workspace=self.workspace,
            document=self.document,
            page_number=13,
            chunk_index=0,
            section_heading="Methodology",
            content="Fuzzy pairwise comparisons are defuzzified to obtain weights.",
            content_hash="e" * 64,
        )

        Citation.objects.create(
            message=self.method_answer,
            chunk=chunk_page_2,
            document=self.document,
            page_number=2,
            citation_number=1,
            quoted_passage=chunk_page_2.content,
            retrieval_score=1.0,
            verification_status="VALID",
        )
        Citation.objects.create(
            message=self.method_answer,
            chunk=chunk_page_13,
            document=self.document,
            page_number=13,
            citation_number=2,
            quoted_passage=chunk_page_13.content,
            retrieval_score=0.9,
            verification_status="VALID",
        )

        Message.objects.create(
            conversation=self.conversation,
            role=Message.Role.USER,
            content="What are the main limitations of this approach?",
        )
        Message.objects.create(
            conversation=self.conversation,
            role=Message.Role.ASSISTANT,
            content=(
                "The provided excerpts do not explicitly discuss limitations "
                "of the Turkey GIS-based FAHP approach."
            ),
            model_name="test-model",
            confidence="low",
        )

        self.client.force_login(self.user)

    def test_direct_note_commands_are_detected_but_help_question_is_not(self):
        self.assertTrue(
            is_chat_note_request(
                "create a note from the Turkey methodology discussion"
            )
        )
        self.assertTrue(is_chat_note_request("save this as a note"))
        self.assertTrue(is_chat_note_request("Could you please create a note from this?"))
        self.assertFalse(is_chat_note_request("How do I create a note?"))

    @patch("apps.conversations.views.answer_workspace_question")
    def test_chat_note_uses_matching_earlier_methodology_discussion(
        self,
        answer_workspace_question_mock,
    ):
        response = self.client.post(
            reverse("conversation-detail", args=[self.conversation.id]),
            {"question": "create a note from the Turkey methodology discussion"},
            HTTP_X_REQUESTED_WITH="XMLHttpRequest",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["model"], "note-action")
        self.assertIn("Saved note", payload["answer"])
        answer_workspace_question_mock.assert_not_called()

        note = Note.objects.get()
        self.assertEqual(note.owner, self.user)
        self.assertEqual(note.workspace, self.workspace)
        self.assertEqual(note.document, self.document)
        self.assertIsNone(note.page_number)
        self.assertIn("15_Turkey_AHP", note.title)
        self.assertIn("Methodology", note.title)
        self.assertIn("FAHP", note.content)
        self.assertIn("GIS", note.content)
        self.assertIn("Source: 15_Turkey_AHP, pages 2, 13.", note.content)
        self.assertNotIn("do not explicitly discuss limitations", note.content)
        self.assertIn("chat,methodology", note.tags)

    @patch("apps.conversations.views.answer_workspace_question")
    def test_retry_ignores_answer_from_previous_failed_note_attempt(
        self,
        answer_workspace_question_mock,
    ):
        Message.objects.create(
            conversation=self.conversation,
            role=Message.Role.USER,
            content="create a note from the Turkey methodology discussion",
        )
        failed_answer = Message.objects.create(
            conversation=self.conversation,
            role=Message.Role.ASSISTANT,
            content=(
                "Methodology supported by the paper\n\n"
                "Accessed July 27, 2022 Karabulut AI, Yazici-Karabulut B, "
                "Derin P, Yesilnacar MI, Cullu MA (2022) Landfill siting "
                "for municipal solid waste using remote sensing and "
                "geographic information system integrated analytic hierarchy "
                "process and simple additive weighting methods. [8]"
            ),
            model_name="test-model",
            confidence="high",
        )
        page_17_chunk = DocumentChunk.objects.create(
            owner=self.user,
            workspace=self.workspace,
            document=self.document,
            page_number=17,
            chunk_index=0,
            section_heading="References",
            content=(
                "Karabulut AI et al. (2022) Landfill siting using remote "
                "sensing, GIS, AHP and simple additive weighting methods."
            ),
            content_hash="f" * 64,
        )
        Citation.objects.create(
            message=failed_answer,
            chunk=page_17_chunk,
            document=self.document,
            page_number=17,
            citation_number=8,
            quoted_passage=page_17_chunk.content,
            retrieval_score=1.0,
            verification_status="VALID",
        )

        response = self.client.post(
            reverse("conversation-detail", args=[self.conversation.id]),
            {"question": "create a note from the Turkey methodology discussion"},
            HTTP_X_REQUESTED_WITH="XMLHttpRequest",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["model"], "note-action")
        answer_workspace_question_mock.assert_not_called()

        note = Note.objects.get()
        self.assertIn("FAHP", note.content)
        self.assertIn("GIS", note.content)
        self.assertIn("Source: 15_Turkey_AHP, pages 2, 13.", note.content)
        self.assertNotIn("Accessed July 27, 2022", note.content)
        self.assertNotEqual(note.page_number, 17)

    @patch("apps.conversations.views.answer_workspace_question")
    def test_note_command_without_prior_answer_does_not_create_empty_note(
        self,
        answer_workspace_question_mock,
    ):
        empty_conversation = Conversation.objects.create(
            owner=self.user,
            workspace=self.workspace,
            title="Empty",
        )

        response = self.client.post(
            reverse("conversation-detail", args=[empty_conversation.id]),
            {"question": "save this as a note"},
            HTTP_X_REQUESTED_WITH="XMLHttpRequest",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["model"], "note-action")
        self.assertIn("could not find an earlier research answer", payload["answer"].lower())
        self.assertEqual(Note.objects.count(), 0)
        answer_workspace_question_mock.assert_not_called()

    @patch("apps.conversations.views.answer_workspace_question")
    def test_note_help_question_remains_a_normal_research_request(
        self,
        answer_workspace_question_mock,
    ):
        answer_workspace_question_mock.return_value = {
            "answer": "Use the Notes page to create a note.",
            "confidence": "high",
            "model": "test-model",
            "hits": [],
            "intent": "general",
        }

        response = self.client.post(
            reverse("conversation-detail", args=[self.conversation.id]),
            {"question": "How do I create a note?"},
            HTTP_X_REQUESTED_WITH="XMLHttpRequest",
        )

        self.assertEqual(response.status_code, 200)
        answer_workspace_question_mock.assert_called_once()
        self.assertEqual(Note.objects.count(), 0)
