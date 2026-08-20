import json
from types import SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase, override_settings

from apps.rag.generator import (
    _workspace_summary_covers_all_documents,
    generate_answer,
)
from apps.retrieval.lexical import SearchHit


def make_hit(document_id, title, content, page=1):
    document = SimpleNamespace(
        id=document_id,
        display_title=title,
    )

    chunk = SimpleNamespace(
        id=f"chunk-{document_id}-{page}",
        document=document,
        document_id=document_id,
        page_number=page,
        content=content,
    )

    return SearchHit(
        item=chunk,
        score=1.0,
    )


class WorkspaceSummaryCoverageTests(SimpleTestCase):
    def setUp(self):
        self.hits = [
            make_hit(
                "egypt-doc",
                "16_Egypt_GIS_AHP",
                (
                    "This study aims to identify suitable photovoltaic sites "
                    "in Egypt using GIS and AHP for solar-energy planning."
                ),
            ),
            make_hit(
                "turkey-doc",
                "15_Turkey_AHP",
                (
                    "This study aims to determine suitable solar power plant "
                    "locations in Turkey using fuzzy AHP together with GIS."
                ),
            ),
            make_hit(
                "brazil-doc",
                "40_Brazil_AHP_TOPSIS_2020",
                (
                    "This paper applies AHP with TOPSIS and MAUT to rank "
                    "alternative photovoltaic installation areas in Brazil."
                ),
            ),
        ]

    def test_summary_requires_citation_from_every_document(self):
        egypt_only = (
            "The Egypt study evaluates photovoltaic site suitability "
            "using GIS and AHP. [1]"
        )

        complete = (
            "Egypt uses GIS and AHP. [1]\n\n"
            "Turkey uses fuzzy AHP with GIS. [2]\n\n"
            "Brazil combines AHP with TOPSIS and MAUT. [3]"
        )

        self.assertFalse(
            _workspace_summary_covers_all_documents(
                egypt_only,
                self.hits,
            )
        )

        self.assertTrue(
            _workspace_summary_covers_all_documents(
                complete,
                self.hits,
            )
        )

    @override_settings(
        GROQ_API_KEY="test-key",
        GROQ_MODEL="test-model",
    )
    @patch("apps.rag.generator._call_groq")
    def test_egypt_only_model_summary_falls_back_to_all_documents(
        self,
        call_groq_mock,
    ):
        model_answer = {
            "answer_markdown": (
                "## Egypt\n\n"
                "The Egypt study evaluates photovoltaic site suitability "
                "using GIS and AHP methods for national solar planning. [1]"
            ),
            "used_sources": {
                "source_1": True,
                "source_2": False,
                "source_3": False,
            },
            "evidence_sufficient": True,
        }

        response = SimpleNamespace(
            headers={
                "x-request-id": "summary-coverage-test",
            }
        )

        data = {
            "id": "summary-coverage-test",
            "model": "test-model",
        }

        call_groq_mock.return_value = (
            response,
            data,
            json.dumps(model_answer),
        )

        answer, confidence, model = generate_answer(
            "Summarize the uploaded research papers.",
            self.hits,
            answer_mode="workspace-summary",
        )

        # The incomplete Egypt-only model answer must not be accepted.
        self.assertNotEqual(
            answer,
            model_answer["answer_markdown"],
        )

        # The deterministic fallback must represent every uploaded PDF.
        self.assertIn("16_Egypt_GIS_AHP", answer)
        self.assertIn("15_Turkey_AHP", answer)
        self.assertIn("40_Brazil_AHP_TOPSIS_2020", answer)

        # It should fail over immediately instead of wasting more Groq calls.
        self.assertEqual(
            call_groq_mock.call_count,
            1,
        )

        self.assertEqual(
            confidence,
            "medium",
        )
        self.assertEqual(
            model,
            "retrieval-only",
        )