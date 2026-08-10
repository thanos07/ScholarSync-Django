from types import SimpleNamespace
from unittest.mock import patch

from django.core.cache import cache
from django.test import SimpleTestCase, override_settings

from apps.rag.generator import (
    _grounded_answer_cache_key,
    _request_payload,
    generate_answer,
)


class GroundedAnswerStabilityTests(SimpleTestCase):
    def setUp(self):
        cache.clear()
        document = SimpleNamespace(id="doc-brazil", display_title="Brazil")
        item = SimpleNamespace(
            id="chunk-1",
            document_id="doc-brazil",
            document=document,
            page_number=3,
            content=(
                "The analysis uses GIS with AHP for weighting and TOPSIS "
                "for ranking alternatives."
            ),
        )
        self.hits = [SimpleNamespace(item=item, score=1.0)]

    @override_settings(GROQ_MODEL="test-model")
    def test_same_question_and_evidence_have_same_cache_key(self):
        first = _grounded_answer_cache_key(
            "What methodology does the Brazil paper use?",
            self.hits,
            "workspace-general",
            conversation_context=[],
        )
        second = _grounded_answer_cache_key(
            "What methodology does the Brazil paper use?",
            self.hits,
            "workspace-general",
            conversation_context=[],
        )
        self.assertEqual(first, second)
        self.assertTrue(first)

    @override_settings(GROQ_MODEL="test-model")
    def test_different_question_changes_cache_key(self):
        first = _grounded_answer_cache_key(
            "What methodology does the Brazil paper use?",
            self.hits,
            "workspace-general",
            conversation_context=[],
        )
        second = _grounded_answer_cache_key(
            "What are the Brazil paper findings?",
            self.hits,
            "workspace-general",
            conversation_context=[],
        )
        self.assertNotEqual(first, second)

    @override_settings(GROQ_MODEL="test-model")
    def test_workspace_general_payload_is_deterministic(self):
        payload = _request_payload(
            "What methodology does the Brazil paper use?",
            self.hits,
            [],
            "workspace-general",
            structured=True,
        )
        self.assertEqual(payload["temperature"], 0.0)

    @override_settings(GROQ_API_KEY="fake", GROQ_MODEL="test-model")
    def test_cached_workspace_general_answer_skips_api(self):
        key = _grounded_answer_cache_key(
            "What methodology does the Brazil paper use?",
            self.hits,
            "workspace-general",
            conversation_context=[],
        )
        cache.set(
            key,
            {
                "answer": "Brazil uses GIS, AHP, and TOPSIS. [1]",
                "confidence": "high",
                "model": "test-model-cached",
            },
            timeout=120,
        )

        with patch("apps.rag.generator._call_groq") as call:
            answer, confidence, model = generate_answer(
                "What methodology does the Brazil paper use?",
                self.hits,
                conversation_context=[],
                answer_mode="workspace-general",
            )

        call.assert_not_called()
        self.assertEqual(answer, "Brazil uses GIS, AHP, and TOPSIS. [1]")
        self.assertEqual(confidence, "high")
        self.assertEqual(model, "test-model-cached")
