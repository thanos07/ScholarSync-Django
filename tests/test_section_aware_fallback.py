from types import SimpleNamespace

from django.test import SimpleTestCase

from apps.rag.generator import (
    _workspace_methodology_fallback,
    _workspace_state_of_art_fallback,
)


class SectionAwareFallbackTests(SimpleTestCase):
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

    def test_methodology_fallback_rejects_section_2_organizational_sentence(self):
        hits = [
            self._hit(
                "c1",
                2,
                "Section 2 presents the state of the art of studies that used MCDM-GIS.",
            ),
            self._hit(
                "c2",
                3,
                "After obtaining all the necessary information, these data are processed by the gvSIG software and evaluated through the TOPSIS method, generating a ranking of the best areas.",
                heading="Materials and methods",
            ),
        ]

        answer = _workspace_methodology_fallback(hits)
        self.assertIn("gvSIG", answer)
        self.assertNotIn("Section 2 presents", answer)

    def test_state_of_art_fallback_lists_prior_studies(self):
        hits = [
            self._hit(
                "c1",
                2,
                "Section 2 presents the state of the art of studies that used MCDM-GIS. Many previous studies about solar energy have formulated the problem as a multi-criteria decision-making problem in combination with GIS.",
            ),
            self._hit(
                "c2",
                3,
                "Aly, Jensen, and Pedersen used GIS-MCDM, specifically the AHP method to investigate the spatial adequacy for large-scale solar energy in Tanzania.",
            ),
            self._hit(
                "c3",
                3,
                "Sanchez-Lozano et al. analyzed the combination of GIS-MCDM in Cartagena, Spain, using AHP to weight criteria and TOPSIS to evaluate alternatives.",
            ),
            self._hit(
                "c4",
                9,
                "The areas considered as inadequate represent those areas with restrictions removed in the previous step.",
            ),
        ]

        answer = _workspace_state_of_art_fallback(
            "what are the studies mentioned in the brazil paper",
            hits,
        )

        self.assertIn("Section 2", answer)
        self.assertIn("Prior studies mentioned", answer)
        self.assertIn("Tanzania", answer)
        self.assertIn("Cartagena", answer)
        self.assertNotIn("areas considered as inadequate", answer.lower())
