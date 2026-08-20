import json
from types import SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase, override_settings

from apps.rag.generator import (
    _applied_methodology_answer_is_safe,
    _is_applied_methodology_question,
    _workspace_methodology_fallback,
    generate_answer,
)
from apps.retrieval.lexical import SearchHit


def make_hit(document_id, title, content, page=1, section_heading=""):
    document = SimpleNamespace(
        id=document_id,
        display_title=title,
    )
    chunk = SimpleNamespace(
        id=f"chunk-{document_id}-{page}-{abs(hash(content))}",
        document=document,
        document_id=document_id,
        page_number=page,
        section_heading=section_heading,
        content=content,
    )
    return SearchHit(item=chunk, score=1.0)


class WorkspaceMethodologyAttributionTests(SimpleTestCase):
    def setUp(self):
        self.hits = [
            make_hit(
                "turkey-doc",
                "15_Turkey_AHP",
                (
                    "In this study, suitable solar power plant areas in Safranbolu "
                    "District are determined by using fuzzy Analytic Hierarchy "
                    "Process (FAHP) together with Geographic Information Systems "
                    "(GIS). The study constructs raster criteria layers and applies "
                    "fuzzy pairwise comparisons to derive criterion weights."
                ),
                page=2,
                section_heading="Methodology",
            ),
            make_hit(
                "turkey-doc",
                "15_Turkey_AHP",
                (
                    "TOPSIS is used as an objective weighting method in the literature. "
                    "SMART is mentioned as a subjective weighting method. Entropy "
                    "Weight Method is cited as an objective weighting approach, and "
                    "the Delphi method is referenced as an expert-consensus technique."
                ),
                page=10,
                section_heading="Literature Review",
            ),
        ]

    def test_applied_methodology_question_is_detected(self):
        self.assertTrue(
            _is_applied_methodology_question(
                "What methods are used in the Turkey AHP paper?"
            )
        )
        self.assertTrue(
            _is_applied_methodology_question(
                "Explain the methodology of this study."
            )
        )
        self.assertFalse(
            _is_applied_methodology_question(
                "What methods are mentioned in the literature review?"
            )
        )

    def test_methodology_fallback_excludes_literature_only_methods(self):
        answer = _workspace_methodology_fallback(self.hits)

        self.assertIn("FAHP", answer)
        self.assertIn("GIS", answer)
        self.assertNotIn("TOPSIS", answer)
        self.assertNotIn("SMART", answer)
        self.assertNotIn("Entropy", answer)
        self.assertNotIn("Delphi", answer)

    @override_settings(
        GROQ_API_KEY="test-key",
        GROQ_MODEL="test-model",
    )
    @patch("apps.rag.generator._call_groq")
    def test_background_methods_in_model_answer_trigger_safe_fallback(
        self,
        call_groq_mock,
    ):
        contaminated = {
            "answer_markdown": (
                "The study uses FAHP with GIS. [1]\n\n"
                "TOPSIS is used as an objective weighting method in the literature, "
                "SMART is mentioned as a subjective weighting method, and Delphi "
                "is referenced as an expert-consensus method. [2]"
            ),
            "used_sources": {
                "source_1": True,
                "source_2": True,
            },
            "evidence_sufficient": True,
        }

        response = SimpleNamespace(
            headers={"x-request-id": "method-attribution-test"}
        )
        data = {
            "id": "method-attribution-test",
            "model": "test-model",
        }
        call_groq_mock.return_value = (
            response,
            data,
            json.dumps(contaminated),
        )

        self.assertFalse(
            _applied_methodology_answer_is_safe(
                contaminated["answer_markdown"],
                self.hits,
            )
        )

        answer, confidence, model = generate_answer(
            "What methods are used in the Turkey AHP paper?",
            self.hits,
            answer_mode="workspace-general",
        )

        # Preserve the normal model/cache path; reject only the unsafe answer.
        self.assertEqual(call_groq_mock.call_count, 1)
        self.assertIn("FAHP", answer)
        self.assertIn("GIS", answer)
        self.assertNotIn("TOPSIS", answer)
        self.assertNotIn("SMART", answer)
        self.assertNotIn("Entropy", answer)
        self.assertNotIn("Delphi", answer)
        self.assertEqual(confidence, "medium")
        self.assertEqual(model, "retrieval-only")
