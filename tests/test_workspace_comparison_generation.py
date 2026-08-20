from types import SimpleNamespace

from django.test import SimpleTestCase

from apps.rag.generator import (
    _valid_workspace_comparison,
    _workspace_comparison_fallback,
)


class WorkspaceComparisonGenerationTests(SimpleTestCase):
    def _hit(self, document_id, title, page, content):
        document = SimpleNamespace(id=document_id, display_title=title)
        item = SimpleNamespace(
            id=f"{document_id}-{page}",
            document_id=document_id,
            document=document,
            page_number=page,
            content=content,
        )
        return SimpleNamespace(item=item, score=1.0)

    def test_workspace_fallback_is_a_real_comparison_matrix(self):
        hits = [
            self._hit(
                "egypt",
                "2.16_Egypt_GIS_AHP",
                1,
                "The study aims to identify suitable photovoltaic farm locations in Egypt. "
                "The methodology integrates GIS with the analytical hierarchy process (AHP). "
                "Solar radiation, temperature, slope, roads, and urban proximity are used as criteria.",
            ),
            self._hit(
                "egypt",
                "2.16_Egypt_GIS_AHP",
                10,
                "The results identify the east southwestern region as among the most suitable areas for PV farms.",
            ),
            self._hit(
                "brazil",
                "3.40_Brazil_AHP_TOPSIS_2020",
                2,
                "The objective is sustainable energy site planning using multicriteria analysis. "
                "The method combines AHP and TOPSIS to evaluate decision criteria and rank alternatives.",
            ),
            self._hit(
                "brazil",
                "3.40_Brazil_AHP_TOPSIS_2020",
                8,
                "The results provide a ranked set of alternatives from the evaluated criteria.",
            ),
        ]

        answer = _workspace_comparison_fallback(hits)

        self.assertIn("| Aspect |", answer)
        self.assertIn("2.16_Egypt_GIS_AHP", answer)
        self.assertIn("3.40_Brazil_AHP_TOPSIS_2020", answer)
        self.assertIn("| Objective |", answer)
        self.assertIn("| Methodology |", answer)
        self.assertIn("| Data / criteria |", answer)
        self.assertIn("| Main findings |", answer)
        self.assertIn("| Limitations |", answer)
        self.assertTrue(_valid_workspace_comparison(answer))

    def test_workspace_fallback_keeps_source_numbers_with_their_document(self):
        hits = [
            self._hit(
                "egypt",
                "Egypt",
                1,
                "The objective is to identify suitable PV sites in Egypt using GIS and AHP methodology.",
            ),
            self._hit(
                "brazil",
                "Brazil",
                2,
                "The objective is to rank renewable-energy alternatives using AHP and TOPSIS methodology.",
            ),
        ]

        answer = _workspace_comparison_fallback(hits)

        self.assertIn("Egypt", answer)
        self.assertIn("Brazil", answer)
        self.assertIn("[1]", answer)
        self.assertIn("[2]", answer)

    def test_old_excerpt_fallback_is_not_accepted_as_workspace_comparison(self):
        old_answer = (
            "I could not produce a full synthesized model comparison, so here is "
            "a document-balanced grounded fallback:\n\n"
            "- **Egypt:** one excerpt [1]\n"
            "- **Brazil:** another excerpt [2]"
        )
        self.assertFalse(_valid_workspace_comparison(old_answer))

    def test_structured_comparison_table_is_accepted(self):
        answer = (
            "| Aspect | Egypt | Brazil |\n"
            "|---|---|---|\n"
            "| Objective | Identify sites [1] | Rank alternatives [2] |\n"
            "| Methodology | GIS-AHP [3] | AHP-TOPSIS [4] |\n"
            "| Main findings | Suitable areas [5] | Ranked alternatives [6] |"
        )
        self.assertTrue(_valid_workspace_comparison(answer))
    def test_background_prose_is_not_mislabeled_as_objective(self):
        hits = [
            self._hit(
                "egypt",
                "Egypt",
                4,
                "The economic factors have not included because most of the investigated areas in this research are desert.",
            ),
            self._hit(
                "brazil",
                "Brazil",
                2,
                "Multi-Criteria Decision-Making Methods have been widely used to evaluate conflicting criteria and determine the best choice between alternatives.",
            ),
        ]

        answer = _workspace_comparison_fallback(hits)
        objective_row = next(
            line for line in answer.splitlines() if line.startswith("| Objective |")
        )

        self.assertEqual(
            objective_row.count("Not explicitly stated in the supplied evidence"),
            2,
        )

    def test_explicit_omission_can_be_used_as_limitation(self):
        hits = [
            self._hit(
                "egypt",
                "Egypt",
                4,
                "Economic factors such as construction cost and land cost have not included in the analysis.",
            ),
            self._hit(
                "brazil",
                "Brazil",
                2,
                "The methodology combines GIS, AHP, and TOPSIS for photovoltaic site ranking.",
            ),
        ]

        answer = _workspace_comparison_fallback(hits)
        limitation_row = next(
            line for line in answer.splitlines() if line.startswith("| Limitations |")
        )

        self.assertIn("construction cost", limitation_row)
        self.assertIn("[1]", limitation_row)
    def test_objective_rejects_proposed_area_and_section_noise(self):
        hits = [
            self._hit(
                "egypt",
                "Egypt",
                1,
                "Since the strategic plan 2052 of the country includes different proposed areas to be developed, as shown in Fig. 3.",
            ),
            self._hit(
                "brazil",
                "Brazil",
                2,
                "Finally, Section 5 presents the conclusions found by the application of the proposed modeling.",
            ),
        ]

        answer = _workspace_comparison_fallback(hits)
        objective_row = next(
            line for line in answer.splitlines() if line.startswith("| Objective |")
        )

        self.assertEqual(
            objective_row.count("Not explicitly stated in the supplied evidence"),
            2,
        )

    def test_objective_accepts_explicit_paper_goal_language(self):
        hits = [
            self._hit(
                "egypt",
                "Egypt",
                1,
                "This study aims to identify the most suitable photovoltaic farm locations in Egypt using GIS and AHP.",
            ),
            self._hit(
                "brazil",
                "Brazil",
                1,
                "The paper proposes a model for locating large-scale photovoltaic plants by combining GIS, AHP, and TOPSIS.",
            ),
        ]

        answer = _workspace_comparison_fallback(hits)
        objective_row = next(
            line for line in answer.splitlines() if line.startswith("| Objective |")
        )

        self.assertIn("aims to identify", objective_row)
        self.assertIn("proposes a model", objective_row)
        self.assertIn("[1]", objective_row)
        self.assertIn("[2]", objective_row)
    def test_policy_goal_is_not_used_as_paper_objective(self):
        hits = [
            self._hit(
                "tamil",
                "Tamil",
                2,
                "The government of Tamil Nadu accepted the Tamil Nadu solar energy policy in 2012 with aim of reducing carbon emission and achieving 3000 MW by 2015.",
            ),
            self._hit(
                "brazil",
                "Brazil",
                1,
                "The objective of this study is to propose a model capable of indicating the best location for large-scale photovoltaic plants.",
            ),
        ]
        answer = _workspace_comparison_fallback(hits)
        objective_row = next(
            line for line in answer.splitlines() if line.startswith("| Objective |")
        )
        self.assertIn("Not explicitly stated in the supplied evidence", objective_row)
        self.assertIn("objective of this study", objective_row.lower())

    def test_bibliographic_header_is_not_used_as_methodology(self):
        hits = [
            self._hit(
                "tamil",
                "Tamil",
                1,
                "C (February 2024) 105(1):81-99 https://doi.org/10.1007/s40032-023-01001-3 ORIGINAL CONTRIBUTION Geographical Information System (GIS)-Based Solar Photovoltaic Farm Site Suitability Using Multi-criteria Approach (MCA) in Southern Tamilnadu, India.",
            ),
            self._hit(
                "brazil",
                "Brazil",
                1,
                "The analysis of the areas was processed by the gvSIG software, using AHP methods for weighting the factors and TOPSIS for ranking the alternatives.",
            ),
        ]
        answer = _workspace_comparison_fallback(hits)
        methodology_row = next(
            line for line in answer.splitlines() if line.startswith("| Methodology |")
        )
        self.assertNotIn("doi.org", methodology_row)
        self.assertNotIn("ORIGINAL CONTRIBUTION", methodology_row)
        self.assertIn("Not explicitly stated in the supplied evidence", methodology_row)
        self.assertIn("gvSIG", methodology_row)

    def test_contact_affiliation_tail_is_removed_from_data_cell(self):
        hits = [
            self._hit(
                "tamil",
                "Tamil",
                1,
                "The annual Global Horizontal Irradiation (GHI) interpolated map derived from NREL data, as * Edison Gundabattini edison.g@vit.ac.in Department of Thermal and Energy Engineering, School of Mechanical Engineering, Vellore Institute of Technology.",
            ),
            self._hit(
                "brazil",
                "Brazil",
                1,
                "The criteria include climatic factors, protected areas, slope, road distance, and substation distance.",
            ),
        ]
        answer = _workspace_comparison_fallback(hits)
        data_row = next(
            line for line in answer.splitlines() if line.startswith("| Data / criteria |")
        )
        self.assertIn("Global Horizontal Irradiation", data_row)
        self.assertNotIn("@vit.ac.in", data_row)
        self.assertNotIn("Department of", data_row)

    def test_global_background_capacity_stat_is_not_main_finding(self):
        hits = [
            self._hit(
                "tamil",
                "Tamil",
                1,
                "Considering the moderately suitable area, the state has potential to generate around 50 GW of electricity on average.",
            ),
            self._hit(
                "brazil",
                "Brazil",
                1,
                "The world generation capacity for renewable sources increased from 755.1 GW in 2000 to 2350.4 GW in 2018.",
            ),
        ]
        answer = _workspace_comparison_fallback(hits)
        findings_row = next(
            line for line in answer.splitlines() if line.startswith("| Main findings |")
        )
        self.assertIn("50 GW", findings_row)
        self.assertNotIn("2350.4 GW", findings_row)
        self.assertIn("Not explicitly stated in the supplied evidence", findings_row)
