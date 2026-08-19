import json

from types import SimpleNamespace
from unittest.mock import patch

import httpx

from django.core.cache import cache
from django.test import SimpleTestCase, override_settings

from apps.rag.generator import (
    _clean_answer_markdown,
    _fold_trailing_citation_paragraph,
    _grounded_answer_cache_key,
    _parse_model_content,
    _request_payload,
    _response_schema,
    _should_strip_model_source_appendix,
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
    def test_structured_schema_uses_fixed_boolean_source_slots(self):
        schema = _response_schema(4)
        used_sources = (
            schema["json_schema"]["schema"]["properties"]["used_sources"]
        )
        self.assertEqual(used_sources["type"], "object")
        self.assertEqual(
            list(used_sources["properties"]),
            ["source_1", "source_2", "source_3", "source_4"],
        )
        self.assertEqual(
            used_sources["required"],
            ["source_1", "source_2", "source_3", "source_4"],
        )
        self.assertFalse(used_sources["additionalProperties"])
    @override_settings(GROQ_MODEL="test-model")
    def test_structured_prompt_separates_source_ids_from_bibliography_numbers(self):
        payload = _request_payload(
            "What methodology does the Brazil paper use?",
            self.hits,
            [],
            "workspace-general",
            structured=True,
        )

        prompt = payload["messages"][1]["content"]
        self.assertIn("Valid ScholarSync SOURCE ids for this request: 1.", prompt)
        self.assertIn("Bibliography/reference numbers", prompt)
        self.assertIn("used_sources is a boolean object", prompt)

        used_sources = (
            payload["response_format"]["json_schema"]["schema"]["properties"]
            ["used_sources"]
        )
        self.assertEqual(list(used_sources["properties"]), ["source_1"])
        self.assertEqual(used_sources["required"], ["source_1"])

    def test_clean_answer_strips_partially_bold_evidence_sufficient_label(self):
        answer = (
            "The retrieved Turkey passages do not explicitly discuss limitations."
            "\n\n**Evidence sufficient:** false"
        )

        cleaned = _clean_answer_markdown(
            answer,
            strip_source_appendix=True,
        )

        self.assertEqual(
            cleaned,
            "The retrieved Turkey passages do not explicitly discuss limitations.",
        )
    def test_parse_model_content_reads_boolean_source_slots(self):
        content = json.dumps(
            {
                "answer_markdown": "Grounded answer [2][4]",
                "used_sources": {
                    "source_1": False,
                    "source_2": True,
                    "source_3": False,
                    "source_4": True,
                },
                "evidence_sufficient": True,
            }
        )

        answer, used, sufficient = _parse_model_content(content)

        self.assertEqual(answer, "Grounded answer [2][4]")
        self.assertEqual(used, [2, 4])
        self.assertTrue(sufficient)

    def test_parse_model_content_keeps_legacy_list_compatibility(self):
        content = json.dumps(
            {
                "answer_markdown": "Legacy answer",
                "used_sources": [2, 4],
                "evidence_sufficient": True,
            }
        )

        _answer, used, _sufficient = _parse_model_content(content)

        self.assertEqual(used, [2, 4])
    def test_clean_answer_strips_trailing_model_sources_appendix(self):
        answer = (
            "The Brazil study uses GIS-MCDM with AHP and TOPSIS. [2]\n\n"
            "## Sources\n"
            "- AHP weighting and TOPSIS ranking are described here. [5][6]\n"
            "- The overall GIS-MCDM workflow is described here. [2]"
        )

        cleaned = _clean_answer_markdown(
            answer,
            strip_source_appendix=True,
        )

        self.assertEqual(
            cleaned,
            "The Brazil study uses GIS-MCDM with AHP and TOPSIS. [2]",
        )

    def test_clean_answer_does_not_strip_data_sources_heading(self):
        answer = (
            "## Data sources\n"
            "The study uses spatial criteria from the supplied datasets. [1]"
        )

        cleaned = _clean_answer_markdown(
            answer,
            strip_source_appendix=True,
        )

        self.assertEqual(cleaned, answer)

    def test_source_listing_questions_preserve_model_source_content(self):
        self.assertFalse(
            _should_strip_model_source_appendix(
                "What sources and references are discussed in the Brazil paper?",
                "workspace-general",
            )
        )
        self.assertFalse(
            _should_strip_model_source_appendix(
                "What are the other studies mentioned in the Brazil paper?",
                "workspace-general",
            )
        )
        self.assertTrue(
            _should_strip_model_source_appendix(
                "What methodology does the Brazil paper use?",
                "workspace-general",
            )
        )

    @override_settings(GROQ_MODEL="test-model")
    def test_workspace_general_prompt_forbids_duplicate_source_appendix(self):
        payload = _request_payload(
            "What methodology does the Brazil paper use?",
            self.hits,
            [],
            "workspace-general",
            structured=True,
        )

        prompt = payload["messages"][1]["content"]
        self.assertIn(
            "Do not append a separate Sources, References",
            prompt,
        )
        self.assertIn(
            "ScholarSync renders the source list separately",
            prompt,
        )
    def test_clean_answer_strips_source_appendix_before_citation_normalization(self):
        answer = (
            "The Brazil study uses GIS-MCDM with AHP and TOPSIS.\n\n"
            "## Sources\n"
            "1. AHP weighting and TOPSIS ranking are described here. [SOURCE 5][SOURCE 6]\n"
            "1. The overall GIS-MCDM workflow is described here. [SOURCE 2]"
        )

        cleaned = _clean_answer_markdown(
            answer,
            strip_source_appendix=True,
        )

        self.assertEqual(
            cleaned,
            "The Brazil study uses GIS-MCDM with AHP and TOPSIS.",
        )

    def test_clean_answer_strips_grouped_and_bracket_source_markers(self):
        grouped = (
            "Grounded answer.\n\n"
            "Sources\n"
            "- Supporting summary [2, 5]"
        )
        bracketed = (
            "Grounded answer.\n\n"
            "References\n"
            "- Supporting summary 【5】"
        )

        self.assertEqual(
            _clean_answer_markdown(grouped, strip_source_appendix=True),
            "Grounded answer.",
        )
        self.assertEqual(
            _clean_answer_markdown(bracketed, strip_source_appendix=True),
            "Grounded answer.",
        )
    def test_clean_answer_strips_bold_sources_appendix(self):
        answer = (
            "The Brazil study uses GIS-MCDM with AHP and TOPSIS.\n\n"
            "**Sources**\n"
            "- AHP weighting and TOPSIS ranking are described here. [5][6]\n"
            "- The overall GIS-MCDM workflow is described here. [2]"
        )

        cleaned = _clean_answer_markdown(
            answer,
            strip_source_appendix=True,
        )

        self.assertEqual(
            cleaned,
            "The Brazil study uses GIS-MCDM with AHP and TOPSIS.",
        )
    @override_settings(GROQ_API_KEY="fake", GROQ_MODEL="test-model")
    def test_workspace_general_used_sources_attach_to_answer_not_standalone_line(self):
        response = httpx.Response(
            200,
            headers={"x-request-id": "req-valid-used-sources"},
        )
        data = {
            "id": "resp-valid-used-sources",
            "model": "test-model",
        }
        content = json.dumps(
            {
                "answer_markdown": (
                    "The Brazil study uses GIS-MCDM with AHP weighting and "
                    "TOPSIS ranking."
                ),
                "used_sources": {
                    "source_1": True,
                },
                "evidence_sufficient": True,
            }
        )

        with patch(
            "apps.rag.generator._call_groq",
            return_value=(response, data, content),
        ) as call:
            answer, confidence, model = generate_answer(
                "What methodology does the Brazil paper use?",
                self.hits,
                conversation_context=[],
                answer_mode="workspace-general",
            )

        self.assertEqual(call.call_count, 1)
        self.assertEqual(
            answer,
            "The Brazil study uses GIS-MCDM with AHP weighting and TOPSIS ranking. [1]",
        )
        self.assertNotIn("\\n\\n[1]", answer)
        self.assertEqual(confidence, "high")
        self.assertEqual(model, "test-model")
    def test_fold_trailing_citation_only_paragraph_into_prose(self):
        answer = (
            "The Brazil study uses GIS-MCDM with AHP weighting and TOPSIS "
            "ranking.\n\n[5] [6]"
        )

        folded = _fold_trailing_citation_paragraph(answer)

        self.assertEqual(
            folded,
            "The Brazil study uses GIS-MCDM with AHP weighting and TOPSIS ranking. [5][6]",
        )

    @override_settings(GROQ_API_KEY="fake", GROQ_MODEL="test-model")
    def test_workspace_general_model_citation_only_tail_is_folded(self):
        response = httpx.Response(
            200,
            headers={"x-request-id": "req-citation-tail"},
        )
        data = {
            "id": "resp-citation-tail",
            "model": "test-model",
        }
        content = json.dumps(
            {
                "answer_markdown": (
                    "The Brazil study uses GIS-MCDM with AHP weighting and "
                    "TOPSIS ranking.\n\n[1]"
                ),
                "used_sources": {
                    "source_1": True,
                },
                "evidence_sufficient": True,
            }
        )

        with patch(
            "apps.rag.generator._call_groq",
            return_value=(response, data, content),
        ) as call:
            answer, confidence, model = generate_answer(
                "What methodology does the Brazil paper use?",
                self.hits,
                conversation_context=[],
                answer_mode="workspace-general",
            )

        self.assertEqual(call.call_count, 1)
        self.assertEqual(
            answer,
            "The Brazil study uses GIS-MCDM with AHP weighting and TOPSIS ranking. [1]",
        )
        self.assertNotIn("\n\n[1]", answer)
        self.assertEqual(confidence, "high")
        self.assertEqual(model, "test-model")
