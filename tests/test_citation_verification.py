from types import SimpleNamespace

from django.test import SimpleTestCase

from apps.conversations.citation_verification import (
    verification_label,
    verify_citation_support,
)
from apps.conversations.views import _citations_for_result


class CitationVerificationTests(SimpleTestCase):
    def test_strong_claim_passage_match_is_supported(self):
        answer = "The Brazil workflow uses AHP for weighting and TOPSIS for ranking. [2]"
        passage = (
            "The analysis used AHP methods for weighting the factors and TOPSIS "
            "for ranking the alternatives in gvSIG."
        )
        self.assertEqual(
            verify_citation_support(answer, 2, passage),
            "SUPPORTED",
        )

    def test_numeric_mismatch_is_not_marked_supported(self):
        answer = "The highly suitable area is 50 GW. [1]"
        passage = "The moderately suitable area could generate around 40 GW."
        self.assertNotEqual(
            verify_citation_support(answer, 1, passage),
            "SUPPORTED",
        )

    def test_unrelated_passage_requires_review(self):
        answer = "The study uses TOPSIS to rank 453 alternatives. [4]"
        passage = "Solar radiation is an important resource for photovoltaic generation."
        self.assertEqual(
            verify_citation_support(answer, 4, passage),
            "REVIEW",
        )

    def test_table_cells_are_verified_independently(self):
        answer = (
            "| Methodology | Egypt uses GIS and AHP. [1] | "
            "Brazil uses AHP and TOPSIS. [2] |"
        )
        egypt = "The method combines GIS processing with AHP weighting."
        brazil = "AHP was used for weighting and TOPSIS for ranking alternatives."

        self.assertEqual(
            verify_citation_support(answer, 1, egypt),
            "SUPPORTED",
        )
        self.assertEqual(
            verify_citation_support(answer, 2, brazil),
            "SUPPORTED",
        )

    def test_missing_inline_marker_is_unchecked(self):
        self.assertEqual(
            verify_citation_support(
                "A grounded answer without this source marker.",
                3,
                "Relevant passage text.",
            ),
            "UNCHECKED",
        )

    def test_citation_payload_exposes_verification_status(self):
        document = SimpleNamespace(
            id="11111111-1111-1111-1111-111111111111",
            display_title="Brazil paper",
        )
        chunk = SimpleNamespace(
            document_id="11111111-1111-1111-1111-111111111111",
            document=document,
            page_number=3,
            content="AHP was used for weighting and TOPSIS for ranking alternatives.",
        )
        hit = SimpleNamespace(item=chunk, score=0.9)
        result = {
            "answer": "Brazil uses AHP for weighting and TOPSIS for ranking. [1]",
            "hits": [hit],
        }

        rows = _citations_for_result(result)
        payload = rows[0][2]

        self.assertEqual(payload["verification_status"], "SUPPORTED")
        self.assertEqual(payload["verification_label"], "Strong evidence match")

    def test_label_for_existing_valid_status_remains_readable(self):
        self.assertEqual(verification_label("VALID"), "Supported")
    def test_grouped_sources_line_is_unchecked_not_review(self):
        answer = (
            "The study uses GIS, AHP, and weighted overlay for site suitability.\n"
            "Sources: [1], [2], [3], [5]"
        )
        passage = "The ArcGIS weighted overlay tool combines the input layers."

        self.assertEqual(
            verify_citation_support(answer, 1, passage),
            "UNCHECKED",
        )

    def test_inline_claim_still_gets_verified(self):
        answer = "The analysis uses AHP for weighting and TOPSIS for ranking. [1]"
        passage = "AHP was used for weighting and TOPSIS for ranking alternatives."

        self.assertEqual(
            verify_citation_support(answer, 1, passage),
            "SUPPORTED",
        )
    def test_bold_grouped_sources_line_is_unchecked(self):
        answer = (
            "The Brazil study uses GIS-MCDM with AHP and TOPSIS.\n"
            "**Sources:** [1], [4], [5]"
        )
        passage = "AHP was used for weighting and TOPSIS for ranking alternatives."

        self.assertEqual(
            verify_citation_support(answer, 1, passage),
            "UNCHECKED",
        )
