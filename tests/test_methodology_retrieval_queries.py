from django.test import SimpleTestCase

from apps.rag.orchestrator import _intent_queries


class MethodologyRetrievalQueryTests(SimpleTestCase):
    def test_methodology_question_expands_to_method_specific_queries(self):
        queries = _intent_queries(
            "What methodology does the Brazil paper use?",
            "What methodology does the Brazil paper use?",
            "general",
        )
        joined = " ".join(queries).lower()

        self.assertIn("workflow", joined)
        self.assertIn("ahp", joined)
        self.assertIn("topsis", joined)
        self.assertIn("sensitivity", joined)
