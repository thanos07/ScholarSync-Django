from types import SimpleNamespace
from unittest.mock import patch
from django.test import SimpleTestCase
from apps.rag import generator
from apps.rag.generator import _is_state_of_art_question

class StateOfArtRuntimeRoutingTests(SimpleTestCase):
    def _hit(self):
        document = SimpleNamespace(id="brazil", display_title="3.40_Brazil_AHP_TOPSIS_2020")
        item = SimpleNamespace(
            id="c1", document_id="brazil", document=document, page_number=3,
            section_heading="State of the art",
            content="Aly, Jensen, and Pedersen used GIS-MCDM, specifically AHP, for solar-energy siting in Tanzania.",
        )
        return SimpleNamespace(item=item, score=1.0)

    def test_natural_studies_mentioned_phrase_is_detected(self):
        self.assertTrue(_is_state_of_art_question(
            "what are the other studies mentioned in the brazil paper, like what are the other papers doing"
        ))

    def test_workspace_general_routes_to_state_of_art_fallback(self):
        q = "what are the studies mentioned in the brazil paper"
        with patch.object(generator, "_workspace_state_of_art_fallback", return_value="STATE-OF-ART") as fn:
            result = generator._extractive_answer(q, [self._hit()], answer_mode="workspace-general")
        self.assertEqual(result, "STATE-OF-ART")
        fn.assert_called_once()
