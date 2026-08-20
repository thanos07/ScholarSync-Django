from types import SimpleNamespace

from django.test import SimpleTestCase

from apps.rag.orchestrator import (
    _contextual_query,
    _intent_queries,
    _needs_recent_document_context,
    _recent_history_documents,
)


class SectionFollowupRoutingTests(SimpleTestCase):
    def _doc(self, doc_id, title):
        return SimpleNamespace(
            id=doc_id,
            display_title=title,
            original_filename=title + ".pdf",
            processing_status="READY",
        )

    def test_check_and_give_then_uses_previous_question_context(self):
        history = [
            {
                "role": "USER",
                "content": "What methodology does the Brazil paper use?",
            },
            {"role": "ASSISTANT", "content": "A grounded methodology answer."},
        ]
        contextual = _contextual_query(
            "check and give then",
            history,
            "general",
        )

        self.assertIn("What methodology does the Brazil paper use?", contextual)
        self.assertIn("check and give then", contextual)

    def test_section_2_then_is_treated_as_contextual_followup(self):
        history = [
            {
                "role": "USER",
                "content": "What methodology does the Brazil paper use?",
            }
        ]
        contextual = _contextual_query("what is section 2 then", history, "general")
        self.assertIn("Brazil paper", contextual)
        self.assertTrue(_needs_recent_document_context("what is section 2 then"))

    def test_short_followup_reuses_latest_named_document(self):
        brazil = self._doc("brazil", "3.40_Brazil_AHP_TOPSIS_2020")
        egypt = self._doc("egypt", "2.16_Egypt_GIS_AHP")
        history = [
            {
                "role": "USER",
                "content": "What methodology does the Brazil paper use?",
            },
            {"role": "ASSISTANT", "content": "A grounded answer."},
            {"role": "USER", "content": "what is section 2 then"},
        ]

        matches = _recent_history_documents(history, [brazil, egypt])
        self.assertEqual(matches, [brazil])

    def test_studies_mentioned_routes_to_state_of_art_queries(self):
        queries = _intent_queries(
            "what are the studies mentioned in the brazil paper",
            "what are the studies mentioned in the brazil paper",
            "general",
        )
        joined = " ".join(queries).lower()

        self.assertIn("previous studies", joined)
        self.assertIn("tanzania", joined)
        self.assertIn("topsis", joined)
