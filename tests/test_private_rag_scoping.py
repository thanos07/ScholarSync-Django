from types import SimpleNamespace

from django.test import SimpleTestCase

from apps.rag.orchestrator import (
    _asks_for_all_documents,
    _contextual_query,
    _match_documents,
    _formula_core_hits,
    _retrieve,
    _should_route_inventory,
)


class PrivateWorkspaceScopingTests(SimpleTestCase):
    def setUp(self):
        self.documents = [
            SimpleNamespace(id="tamil", display_title="6.Tamil nadu", processing_status="READY"),
            SimpleNamespace(id="egypt", display_title="2.16_Egypt_GIS_AHP", processing_status="READY"),
            SimpleNamespace(id="brazil", display_title="3.40_Brazil_AHP_TOPSIS_2020", processing_status="READY"),
        ]

    def titles(self, question):
        return [doc.display_title for doc in _match_documents(question, self.documents)]

    def test_formula_request_scopes_to_tamil_nadu(self):
        self.assertEqual(
            self.titles("give me the formula list of tamil nadu paper"),
            ["6.Tamil nadu"],
        )

    def test_formula_request_scopes_to_egypt(self):
        self.assertEqual(
            self.titles("give me the formula list of egypt paper"),
            ["2.16_Egypt_GIS_AHP"],
        )

    def test_comparison_can_name_two_documents(self):
        self.assertEqual(
            set(self.titles("compare tamil nadu and egypt")),
            {"6.Tamil nadu", "2.16_Egypt_GIS_AHP"},
        )

    def test_all_three_phrase_selects_workspace_scope(self):
        self.assertTrue(_asks_for_all_documents("compare all the 3 papers"))
        self.assertTrue(_asks_for_all_documents("compare all three papers"))

    def test_comparison_prompt_is_not_misrouted_as_inventory(self):
        question = (
            'Compare these uploaded papers: "2.16_Egypt_GIS_AHP"; '
            '"3.40_Brazil_AHP_TOPSIS_2020". Compare their objective, '
            'methodology, data or criteria, main findings, and limitations.'
        )
        self.assertFalse(_should_route_inventory(question))

    def test_plain_uploaded_pdf_inventory_question_still_routes_to_inventory(self):
        self.assertTrue(_should_route_inventory("which uploaded PDFs do I have?"))

    def test_formula_followup_uses_only_latest_meaningful_user_turn(self):
        history = [
            {"role": "USER", "content": "compare brazil and tamil nadu"},
            {"role": "ASSISTANT", "content": "comparison"},
            {"role": "USER", "content": "tell me about egypt"},
            {"role": "ASSISTANT", "content": "egypt answer"},
        ]
        contextual = _contextual_query("give me the formulas used here", history, "formula")
        self.assertIn("egypt", contextual.lower())
        self.assertNotIn("brazil", contextual.lower())
        self.assertNotIn("tamil", contextual.lower())

from apps.rag.generator import (
    _formula_answer_covers_evidence,
    _clean_answer_markdown,
    _formula_answer_cache_key,
    _formula_fallback,
    _normalize_formula_answer_markdown,
    _plain_equation_candidates,
)
from apps.conversations.pdf_export import _inline_markup, _latex_to_pdf_markup
from apps.rag.orchestrator import _formula_signal


class FormulaEvidenceTests(SimpleTestCase):
    def test_scalar_min_max_values_are_not_formulas(self):
        text = "Min = 1400\nMax = 2500\nMin = 11.9\nMax = 28.4"
        self.assertEqual(_plain_equation_candidates(text), [])

    def test_named_equations_are_detected(self):
        text = (
            "Consistency Index CI = (lambda_max - m)/(m-1) where m is the number of criteria.\n"
            "Consistency Ratio CR = (CI)/(RI) where RI is the random consistency index.\n"
            "PVLandSuitabilityIndex = sum_(i=1)^n w_i R_i where n is the number of criteria."
        )
        candidates = "\n".join(_plain_equation_candidates(text))
        self.assertIn("CI =", candidates)
        self.assertIn("CR =", candidates)
        self.assertIn("PVLandSuitabilityIndex =", candidates)

    def test_pdf_equals_glyph_counts_as_formula_signal(self):
        self.assertGreater(_formula_signal("reciprocity Eij ¼ 1/Eji"), 0)


class FormulaRuntimeRegressionTests(SimpleTestCase):
    def _chunk(self, ident, page, content):
        document = SimpleNamespace(id="egypt", display_title="2.16_Egypt_GIS_AHP")
        return SimpleNamespace(
            id=ident, document_id="egypt", document=document, page_number=page,
            chunk_index=0, section_heading="", content=content,
        )

    def test_formula_core_hits_exists_and_prefers_equations(self):
        chunks = [
            self._chunk("p5", 5, "reciprocity Eij ¼ 1/Eji where i and j denote criteria"),
            self._chunk("p6", 6, "Consistency Index CI = (lambda_max - m)/(m-1). Consistency Ratio CR = CI/RI."),
            self._chunk("p8", 8, "PVLandSuitabilityIndex = sum_(i=1)^n w_i R_i where w_i is the criterion weight."),
        ]
        hits = _formula_core_hits(chunks, limit=8)
        self.assertGreaterEqual(len(hits), 3)
        self.assertEqual({hit.item.page_number for hit in hits}, {5, 6, 8})

    def test_formula_retrieve_path_does_not_raise_name_error(self):
        chunks = [
            self._chunk("p5", 5, "reciprocity Eij ¼ 1/Eji where i and j denote criteria"),
            self._chunk("p6", 6, "Consistency Index CI = (lambda_max - m)/(m-1). Consistency Ratio CR = CI/RI."),
            self._chunk("p8", 8, "PVLandSuitabilityIndex = sum_(i=1)^n w_i R_i where w_i is the criterion weight."),
        ]
        hits = _retrieve(
            "give me the formula list of egypt paper",
            chunks,
            history=[],
            intent="formula",
            target_documents=[chunks[0].document],
        )
        self.assertTrue(hits)
        self.assertTrue(all(str(hit.item.document_id) == "egypt" for hit in hits))


class FormulaPresentationRegressionTests(SimpleTestCase):
    def _hit(self, ident, page, content):
        document = SimpleNamespace(id="egypt", display_title="2.16_Egypt_GIS_AHP")
        item = SimpleNamespace(
            id=ident,
            document_id="egypt",
            document=document,
            page_number=page,
            content_hash=f"hash-{ident}",
            content=content,
        )
        return SimpleNamespace(item=item, score=1.0)

    def test_meta_section_and_html_subscripts_are_removed(self):
        answer = (
            "## Formula\n\n$$CI=(\\lambda_{max}-m)/(m-1)$$\n\n"
            "λ<sub>max</sub> is the largest eigenvalue. [1]\n\n"
            "## Evidence Sufficient\nThe above equations are supported."
        )
        cleaned = _clean_answer_markdown(answer)
        self.assertIn(r"$\lambda_{max}$", cleaned)
        self.assertNotIn("<sub>", cleaned)
        self.assertNotIn("Evidence Sufficient", cleaned)
        self.assertNotIn("The above equations are supported", cleaned)

    def test_corrupted_ocr_is_not_dumped_as_formula(self):
        hits = [
            self._hit(
                "p6",
                6,
                "Consistency Index equation CI = kmax ■ m m ■ 1 ð1Þ and Consistency Ratio equation CR = CI RI ð2Þ",
            )
        ]
        answer = _formula_fallback(hits)
        self.assertNotIn("kmax", answer)
        self.assertNotIn("■", answer)
        self.assertIn("too corrupted", answer)

    def test_formula_cache_key_depends_on_exact_evidence_order(self):
        first = self._hit("p6", 6, "CI = (lambda_max-m)/(m-1)")
        second = self._hit("p8", 8, "PVLandSuitabilityIndex = sum_(i=1)^n w_i R_i")
        key_a = _formula_answer_cache_key([first, second], "workspace-formula")
        key_b = _formula_answer_cache_key([first, second], "workspace-formula")
        key_c = _formula_answer_cache_key([second, first], "workspace-formula")
        self.assertEqual(key_a, key_b)
        self.assertNotEqual(key_a, key_c)


    def test_evidence_used_section_is_removed(self):
        answer = (
            "## Formulas\n\n$$CI = \\frac{\\lambda_{max}-m}{m-1}$$\n\n"
            "The equation is defined in the paper. [1]\n\n"
            "## Evidence Used\n- Source 1: CI equation."
        )
        cleaned = _clean_answer_markdown(answer)
        self.assertNotIn("Evidence Used", cleaned)
        self.assertNotIn("Source 1: CI equation", cleaned)

    def test_plain_formula_lines_are_wrapped_for_katex(self):
        answer = (
            "### 1. Consistency Index\n"
            "CI=(\\lambda_max-m)/(m-1)\n"
            "\\lambda_max is the largest eigenvalue. [1]\n"
            "PVLandSuitabilityIndex=sum_(i=1)^n w_i R_i\n"
        )
        normalized = _normalize_formula_answer_markdown(answer)
        self.assertIn(r"CI = \frac{\lambda_{max}-m}{m-1}", normalized)
        self.assertIn(r"\sum_{i=1}^{n}", normalized)
        self.assertIn(r"$\lambda_{max}$", normalized)

    def test_pdf_math_markup_removes_raw_latex(self):
        markup = _latex_to_pdf_markup(r"CI = \\frac{\\lambda_{max}-m}{m-1}")
        self.assertIn("lambda<sub>max</sub>", markup)
        self.assertNotIn(r"\\lambda", markup)

    def test_pv_index_nested_wrappers_are_canonicalized(self):
        answer = (
            r"$$\text{\mathrm{\mathrm{PVLandSuitabilityIndex}}} "
            r"= \sum_{i=1}^{n} w_i R_i$$"
        )
        normalized = _normalize_formula_answer_markdown(answer)
        self.assertIn(
            r"\mathrm{PVLandSuitabilityIndex} = \sum_{i=1}^{n} w_i R_i",
            normalized,
        )
        self.assertNotIn(r"\text{", normalized)
        self.assertNotIn(r"\mathrm{\mathrm", normalized)

    def test_pdf_fraction_parser_handles_nested_subscript_braces(self):
        ci = _latex_to_pdf_markup(
            r"CI = \frac{\lambda_{max}-m}{m-1}"
        )
        reciprocity = _latex_to_pdf_markup(
            r"E_{ij} = \frac{1}{E_{ji}}"
        )
        self.assertIn("lambda<sub>max</sub>", ci)
        self.assertIn("(lambda<sub>max</sub>-m) / (m-1)", ci)
        self.assertNotIn(r"\frac", ci)
        self.assertIn("E<sub>ij</sub>", reciprocity)
        self.assertIn("(1) / (E<sub>ji</sub>)", reciprocity)
        self.assertNotIn(r"\frac", reciprocity)

    def test_pdf_inline_math_uses_math_markup_not_raw_tex(self):
        markup = _inline_markup(
            r"$\lambda_{max}$, $w_i$, $R_i$, and $E_{ij}$"
        )
        self.assertIn("lambda<sub>max</sub>", markup)
        self.assertIn("w<sub>i</sub>", markup)
        self.assertIn("R<sub>i</sub>", markup)
        self.assertIn("E<sub>ij</sub>", markup)
        self.assertNotIn(r"\lambda", markup)

class FormulaCompletenessAndPdfRegressionTests(SimpleTestCase):
    def _hit(self, ident, page, content):
        document = SimpleNamespace(
            id="egypt",
            display_title="2.16_Egypt_GIS_AHP",
        )
        item = SimpleNamespace(
            id=ident,
            document_id="egypt",
            document=document,
            page_number=page,
            content_hash=f"hash-{ident}",
            content=content,
        )
        return SimpleNamespace(item=item, score=1.0)

    def test_formula_answer_must_cover_every_verified_named_equation(self):
        hits = [
            self._hit(
                "p5",
                5,
                "The pair-wise matrix has reciprocity Eij = 1/Eji.",
            ),
            self._hit(
                "p6",
                6,
                "Consistency Index CI = (lambda_max-m)/(m-1). "
                "Consistency Ratio CR = (CI)/(RI).",
            ),
            self._hit(
                "p8",
                8,
                "PVLandSuitabilityIndex = sum_(i=1)^n w_i R_i.",
            ),
        ]

        incomplete = (
            r"$$CI = \frac{\lambda_{max}-m}{m-1}$$" "\n"
            r"$$CR = \frac{CI}{RI}$$" "\n"
            r"$$\mathrm{PVLandSuitabilityIndex} = \sum_{i=1}^{n} w_i R_i$$"
        )
        complete = incomplete + "\n" + r"$$E_{ij} = \frac{1}{E_{ji}}$$"

        self.assertFalse(_formula_answer_covers_evidence(incomplete, hits))
        self.assertTrue(_formula_answer_covers_evidence(complete, hits))

    def test_pdf_lambda_uses_readable_ascii_with_subscript(self):
        markup = _latex_to_pdf_markup(
            r"CI = \frac{\lambda_{max}-m}{m-1}"
        )
        self.assertIn("lambda<sub>max</sub>", markup)
        self.assertNotIn(r"\lambda", markup)
        self.assertNotIn("λ", markup)
