from types import SimpleNamespace

from django.test import SimpleTestCase

from apps.rag.generator import _workspace_comparison_integrity_is_safe


def _hit(document_id, title, page, content, heading=""):
    document = SimpleNamespace(id=document_id, display_title=title)
    item = SimpleNamespace(
        document=document,
        document_id=document_id,
        page_number=page,
        section_heading=heading,
        content=content,
    )
    return SimpleNamespace(item=item, score=1.0)


class WorkspaceComparisonIntegrityTests(SimpleTestCase):
    def setUp(self):
        self.hits = [
            _hit(
                "turkey",
                "15_Turkey_AHP",
                5,
                (
                    "The study uses GIS and fuzzy AHP to combine environmental "
                    "and technical criteria for solar site selection."
                ),
                "Methodology",
            ),
            _hit(
                "turkey",
                "15_Turkey_AHP",
                10,
                (
                    "There are many criteria-weighting methods in the literature. "
                    "These weighting methods divide the criteria weights into "
                    "subjective, objective, and mixed methods."
                ),
                "Background",
            ),
            _hit(
                "egypt",
                "16_Egypt_GIS_AHP",
                5,
                (
                    "Due to lack of reliable forest and groundwater-distance data, "
                    "those factors were not included in the study."
                ),
                "Data",
            ),
        ]

    def test_rejects_inferred_turkey_limitation_and_uncited_key_differences(self):
        answer = """| Dimension | 15_Turkey_AHP | 16_Egypt_GIS_AHP |
|---|---|---|
| Objective | Solar siting in Turkey [1] | PV siting in Egypt [3] |
| Methodology | GIS + fuzzy AHP [1] | GIS + AHP [3] |
| Data / criteria | Environmental and technical criteria [1] | Spatial criteria [3] |
| Main findings | Suitable sites identified [1] | Suitable sites identified [3] |
| Limitations | AHP weights are subjective [2] | Forest and groundwater data were not included [3] |

### Key Differences
- Turkey uses fuzzy AHP, whereas Egypt uses AHP.
"""
        self.assertFalse(
            _workspace_comparison_integrity_is_safe(answer, self.hits)
        )

    def test_rejects_limitation_cited_from_the_other_document(self):
        answer = """| Dimension | 15_Turkey_AHP | 16_Egypt_GIS_AHP |
|---|---|---|
| Objective | Solar siting in Turkey [1] | PV siting in Egypt [3] |
| Methodology | GIS + fuzzy AHP [1] | GIS + AHP [3] |
| Data / criteria | Environmental and technical criteria [1] | Spatial criteria [3] |
| Main findings | Suitable sites identified [1] | Suitable sites identified [3] |
| Limitations | Forest and groundwater data were not included [3] | Forest and groundwater data were not included [3] |
"""
        self.assertFalse(
            _workspace_comparison_integrity_is_safe(answer, self.hits)
        )

    def test_accepts_explicit_limitation_and_cross_document_cited_difference(self):
        answer = """| Dimension | 15_Turkey_AHP | 16_Egypt_GIS_AHP |
|---|---|---|
| Objective | Solar siting in Turkey [1] | PV siting in Egypt [3] |
| Methodology | GIS + fuzzy AHP [1] | GIS + AHP [3] |
| Data / criteria | Environmental and technical criteria [1] | Spatial criteria [3] |
| Main findings | Suitable sites identified [1] | Suitable sites identified [3] |
| Limitations | Not explicitly stated in the supplied evidence | Forest and groundwater data were not included [3] |

### Key Differences
- Turkey uses fuzzy AHP, while Egypt uses AHP [1][3].
"""
        self.assertTrue(
            _workspace_comparison_integrity_is_safe(answer, self.hits)
        )

    def test_key_differences_section_is_optional(self):
        answer = """| Dimension | 15_Turkey_AHP | 16_Egypt_GIS_AHP |
|---|---|---|
| Objective | Solar siting in Turkey [1] | PV siting in Egypt [3] |
| Methodology | GIS + fuzzy AHP [1] | GIS + AHP [3] |
| Data / criteria | Environmental and technical criteria [1] | Spatial criteria [3] |
| Main findings | Suitable sites identified [1] | Suitable sites identified [3] |
| Limitations | Not explicitly stated in the supplied evidence | Forest and groundwater data were not included [3] |
"""
        self.assertTrue(
            _workspace_comparison_integrity_is_safe(answer, self.hits)
        )
