from django.test import SimpleTestCase

from apps.rag.generator import _normalize_formula_latex


class FormulaLatexNormalizationTests(SimpleTestCase):
    def test_repairs_visual_topsis_cases_expression(self):
        raw = (
            r"p_j^+ = \begin{cases}"
            r"max_i p_{ij}, & jin J_1\ "
            r"min_i p_{ij}, & jin J_2"
            r"end{cases}"
        )

        expected = (
            r"p_j^+ = \begin{cases}"
            r"\max_i p_{ij}, & j \in J_1 \\ "
            r"\min_i p_{ij}, & j \in J_2"
            r"\end{cases}"
        )

        self.assertEqual(
            _normalize_formula_latex(raw),
            expected,
        )

    def test_repairs_visual_topsis_distance_expression(self):
        raw = (
            r"d_i^+ = "
            r"sqrt{sum_{j=1}^{n} "
            r"w_j,(p_j^+ - p_{ij})^2}"
            r"quad i=1,dots,m"
        )

        expected = (
            r"d_i^+ = "
            r"\sqrt{\sum_{j=1}^{n} "
            r"w_j (p_j^+ - p_{ij})^2} "
            r"\quad i=1,\dots,m"
        )

        self.assertEqual(
            _normalize_formula_latex(raw),
            expected,
        )

    def test_repairs_visual_ahp_matrix_expression(self):
        raw = (
            r"A = \begin{bmatrix}"
            r"1 & dots & a_{1n}\ "
            r"vdots & ddots & vdots\ "
            r"1/a_{1n} & dots & 1"
            r"end{bmatrix}"
        )

        expected = (
            r"A = \begin{bmatrix}"
            r"1 & \dots & a_{1n} \\ "
            r"\vdots & \ddots & \vdots \\ "
            r"1/a_{1n} & \dots & 1"
            r"\end{bmatrix}"
        )

        self.assertEqual(
            _normalize_formula_latex(raw),
            expected,
        )

    def test_restores_common_missing_tex_commands(self):
        self.assertEqual(
            _normalize_formula_latex(
                r"a_i = prod_{i=1}^{n} a_{in}"
            ),
            r"a_i = \prod_{i=1}^{n} a_{in}",
        )

        self.assertEqual(
            _normalize_formula_latex(
                r"w_i = sqrt[n]{a_i}"
            ),
            r"w_i = \sqrt[n]{a_i}",
        )

    def test_does_not_damage_already_valid_formula(self):
        formula = (
            r"CI = "
            r"\frac{\lambda_{max} - n}{n - 1}"
        )

        self.assertEqual(
            _normalize_formula_latex(formula),
            formula,
        )