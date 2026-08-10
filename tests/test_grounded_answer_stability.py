import json

from types import SimpleNamespace
from unittest.mock import patch

import httpx

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
    @override_settings(GROQ_API_KEY="fake", GROQ_MODEL="test-model")
    def test_429_stops_after_first_model_attempt_and_uses_fallback(self):
        request = httpx.Request(
            "POST",
            "https://api.groq.com/openai/v1/chat/completions",
        )
        response = httpx.Response(
            429,
            request=request,
            headers={"retry-after": "20"},
            json={
                "error": {
                    "message": "Rate limit reached",
                    "type": "tokens",
                    "code": "rate_limit_exceeded",
                }
            },
        )
        error = httpx.HTTPStatusError(
            "429 Too Many Requests",
            request=request,
            response=response,
        )

        with (
            patch("apps.rag.generator._call_groq", side_effect=error) as call,
            patch("apps.rag.generator.time.sleep") as sleep,
        ):
            answer, confidence, model = generate_answer(
                "What methodology does the Brazil paper use?",
                self.hits,
                conversation_context=[],
                answer_mode="workspace-general",
            )

        self.assertEqual(call.call_count, 1)
        sleep.assert_not_called()
        self.assertEqual(model, "retrieval-only")
        self.assertEqual(confidence, "medium")
        self.assertIn("Methodology supported by the paper", answer)

    @override_settings(GROQ_API_KEY="fake", GROQ_MODEL="test-model")
    def test_transient_503_still_retries(self):
        request = httpx.Request(
            "POST",
            "https://api.groq.com/openai/v1/chat/completions",
        )
        response = httpx.Response(
            503,
            request=request,
            json={"error": {"message": "temporarily unavailable"}},
        )
        error = httpx.HTTPStatusError(
            "503 Service Unavailable",
            request=request,
            response=response,
        )

        with (
            patch("apps.rag.generator._call_groq", side_effect=error) as call,
            patch("apps.rag.generator.time.sleep"),
        ):
            _answer, _confidence, model = generate_answer(
                "What methodology does the Brazil paper use?",
                self.hits,
                conversation_context=[],
                answer_mode="workspace-general",
            )

        self.assertEqual(call.call_count, 3)
        self.assertEqual(model, "retrieval-only")
    @override_settings(GROQ_MODEL="test-model")
    def test_workspace_general_payload_uses_bounded_token_budget(self):
        long_text = "method evidence " * 500
        document = SimpleNamespace(id="doc-long", display_title="Long Paper")
        hits = []
        for index in range(10):
            item = SimpleNamespace(
                id=f"chunk-{index}",
                document_id="doc-long",
                document=document,
                page_number=index + 1,
                content=long_text,
            )
            hits.append(SimpleNamespace(item=item, score=1.0))

        payload = _request_payload(
            "What methodology does this paper use?",
            hits,
            [],
            "workspace-general",
            structured=True,
        )

        self.assertEqual(payload["max_completion_tokens"], 900)
        prompt = payload["messages"][1]["content"]
        evidence = prompt.split("Evidence:\n", 1)[1].split(
            "\n\nCurrent question:",
            1,
        )[0]
        self.assertLessEqual(len(evidence), 6500)

    @override_settings(GROQ_MODEL="test-model")
    def test_formula_payload_keeps_full_completion_budget(self):
        payload = _request_payload(
            "Give me the formulas.",
            self.hits,
            [],
            "workspace-formula",
            structured=True,
        )

        self.assertEqual(payload["max_completion_tokens"], 2200)
    @override_settings(GROQ_API_KEY="fake", GROQ_MODEL="test-model")
    def test_workspace_general_citation_miss_does_not_regenerate_full_answer(self):
        response = httpx.Response(
            200,
            headers={"x-request-id": "req-no-citations"},
        )
        data = {
            "id": "resp-no-citations",
            "model": "test-model",
        }
        content = json.dumps(
            {
                "answer_markdown": (
                    "The paper uses GIS with AHP weighting and TOPSIS ranking."
                ),
                "used_sources": [],
                "evidence_sufficient": True,
            }
        )

        with (
            patch(
                "apps.rag.generator._call_groq",
                return_value=(response, data, content),
            ) as call,
            patch("apps.rag.generator.time.sleep") as sleep,
        ):
            answer, confidence, model = generate_answer(
                "What methodology does the Brazil paper use?",
                self.hits,
                conversation_context=[],
                answer_mode="workspace-general",
            )

        self.assertEqual(call.call_count, 1)
        sleep.assert_not_called()
        self.assertEqual(model, "retrieval-only")
        self.assertEqual(confidence, "medium")
        self.assertIn("Methodology supported by the paper", answer)
