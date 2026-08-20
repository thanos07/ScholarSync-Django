from types import SimpleNamespace

from django.test import SimpleTestCase

from apps.rag.generator import _workspace_methodology_fallback


class WorkspaceMethodologyFallbackTests(SimpleTestCase):
    def _hit(self, source_id, page, content, heading=""):
        document = SimpleNamespace(
            id="brazil",
            display_title="3.40_Brazil_AHP_TOPSIS_2020",
        )
        item = SimpleNamespace(
            id=source_id,
            document_id="brazil",
            document=document,
            page_number=page,
            section_heading=heading,
            content=content,
        )
        return SimpleNamespace(item=item, score=1.0)

    def test_methodology_fallback_rejects_background_brazil_sentences(self):
        hits = [
            self._hit(
                "c1",
                3,
                "Research for localization of thermosolar power plants in the northeastern region of Brazil presented a macro-spatial approach based on Geoprocessing, MCDM, and AHP.",
            ),
            self._hit(
                "c2",
                2,
                "Overall, there are high expectations about the potential demand for solar energy in Brazil.",
            ),
            self._hit(
                "c3",
                12,
                "The analysis of the areas was processed by the gvSIG software, using AHP methods for weighting the factors and TOPSIS for ranking the alternatives.",
                heading="Methodology",
            ),
            self._hit(
                "c4",
                12,
                "The ranking was also evaluated using MAUT as a sensitivity analysis to compare the robustness of the alternatives.",
                heading="Results and sensitivity analysis",
            ),
        ]

        answer = _workspace_methodology_fallback(hits)

        self.assertIn("gvSIG", answer)
        self.assertIn("AHP", answer)
        self.assertIn("TOPSIS", answer)
        self.assertIn("MAUT", answer)
        self.assertNotIn("high expectations", answer)
        self.assertNotIn("Research for localization", answer)

    def test_methodology_fallback_keeps_inline_source_numbers(self):
        hits = [
            self._hit(
                "c1",
                12,
                "The analysis was processed using GIS, with AHP for weighting and TOPSIS for ranking the alternatives.",
                heading="Methodology",
            ),
        ]

        answer = _workspace_methodology_fallback(hits)

        self.assertIn("[1]", answer)

    def test_methodology_fallback_refuses_unrelated_evidence(self):
        hits = [
            self._hit(
                "c1",
                2,
                "Overall, there are high expectations about the potential demand for solar energy in Brazil.",
            ),
            self._hit(
                "c2",
                2,
                "There are a few photovoltaic plants in commercial operation in Brazil.",
            ),
        ]

        answer = _workspace_methodology_fallback(hits)

        self.assertIn("could not find a sufficiently specific methodology passage", answer)
        self.assertNotIn("high expectations", answer)
    def test_methodology_fallback_prefers_complementary_method_evidence(self):
        hits = [
            self._hit(
                "c1",
                3,
                "The analysis was processed by gvSIG using AHP for weighting the factors and TOPSIS for ranking the alternatives.",
                heading="Methodology",
            ),
            self._hit(
                "c2",
                12,
                "The sensitivity test used MAUT instead of TOPSIS and compared the ranking with the original AHP-TOPSIS methodology.",
                heading="Sensitivity analysis",
            ),
            self._hit(
                "c3",
                12,
                "The MAUT ranking was similar to the ranking produced by the proposed methodology.",
                heading="Sensitivity analysis",
            ),
        ]

        answer = _workspace_methodology_fallback(hits)

        self.assertIn("gvSIG", answer)
        self.assertIn("AHP", answer)
        self.assertIn("TOPSIS", answer)
        self.assertIn("MAUT", answer)
        self.assertIn("[1]", answer)
        self.assertIn("[2]", answer)
