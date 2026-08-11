from django.test import SimpleTestCase
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import HRFlowable, ListFlowable

from apps.conversations.pdf_export import (
    _latex_to_pdf_markup,
    _markdown_flowables,
)


class PdfFormulaRenderingTests(SimpleTestCase):
    def test_matrix_uses_font_safe_multiline_fallback(self):
        raw = (
            r"A = \begin{bmatrix}"
            r"1 & \dots & a_{1n} \\ "
            r"\vdots & \ddots & \vdots \\ "
            r"1/a_{1n} & \dots & 1"
            r"\end{bmatrix}"
        )

        rendered = _latex_to_pdf_markup(raw)

        self.assertNotIn(r"\begin{bmatrix}", rendered)
        self.assertNotIn(r"\end{bmatrix}", rendered)
        self.assertNotIn(r"\vdots", rendered)
        self.assertNotIn(r"\ddots", rendered)
        self.assertNotIn("⋮", rendered)
        self.assertNotIn("⋱", rendered)
        self.assertIn("<br/>", rendered)
        self.assertIn(":", rendered)
        self.assertIn("...", rendered)
        self.assertIn("a<sub>1n</sub>", rendered)

    def test_cases_use_readable_ascii_membership(self):
        raw = (
            r"p_j^+ = \begin{cases}"
            r"\max_i p_{ij}, & j \in J_1 \\ "
            r"\min_i p_{ij}, & j \in J_2"
            r"\end{cases}"
        )

        rendered = _latex_to_pdf_markup(raw)

        self.assertNotIn(r"\begin{cases}", rendered)
        self.assertNotIn(r"\end{cases}", rendered)
        self.assertNotIn(r"\in", rendered)
        self.assertNotIn("∈", rendered)
        self.assertIn("<br/>", rendered)
        self.assertIn("j in J<sub>1</sub>", rendered)
        self.assertIn("j in J<sub>2</sub>", rendered)
        self.assertIn("p<sub>j</sub><super>+</super>", rendered)

    def test_xi_and_pm_are_font_safe(self):
        rendered = _latex_to_pdf_markup(
            r"\xi_i = \bar d_i^- / (\bar d_i^+ + \bar d_i^-), \quad \pm 1"
        )

        self.assertNotIn(r"\xi", rendered)
        self.assertNotIn("ξ", rendered)
        self.assertNotIn(r"\pm", rendered)
        self.assertIn("xi<sub>i</sub>", rendered)
        self.assertIn("+/-", rendered)

    def test_distance_formula_removes_layout_commands(self):
        raw = (
            r"d_i^+ = \sqrt{\sum_{j=1}^{n} "
            r"w_j (p_j^+ - p_{ij})^2} "
            r"\quad i=1,\dots,m"
        )

        rendered = _latex_to_pdf_markup(raw)

        self.assertNotIn(r"\sqrt", rendered)
        self.assertNotIn(r"\sum", rendered)
        self.assertNotIn(r"\quad", rendered)
        self.assertNotIn(r"\dots", rendered)
        self.assertIn("√", rendered)
        self.assertIn("∑", rendered)
        self.assertIn("…", rendered)

    def test_markdown_separator_becomes_rule(self):
        flowables = _markdown_flowables("---", {}, 400)

        self.assertEqual(len(flowables), 1)
        self.assertIsInstance(flowables[0], HRFlowable)

    def test_unordered_list_does_not_start_at_one(self):
        styles = {
            "Body": ParagraphStyle("Body"),
        }

        flowables = _markdown_flowables(
            "- first item\n- second item",
            styles,
            400,
        )

        self.assertEqual(len(flowables), 1)
        self.assertIsInstance(flowables[0], ListFlowable)
        self.assertEqual(flowables[0]._bulletType, "bullet")
        self.assertEqual(flowables[0]._start, "bulletchar")
