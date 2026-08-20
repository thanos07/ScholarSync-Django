from unittest.mock import Mock, patch

from django.test import SimpleTestCase, override_settings

from apps.documents.visual_equations import (
    _clean_transcription,
    _looks_like_formula_image_page,
    _transcribe_image_png,
)


class VisualEquationExtractionTests(SimpleTestCase):
    def test_formula_page_heuristic_requires_appendix_formula_language(self):
        # Actual appendix-style page: should trigger.
        self.assertTrue(
            _looks_like_formula_image_page(
                "The table presented in Appendix A uses the formulas "
                "used in applying the method."
            )
        )

        # Another actual appendix-style page: should trigger.
        self.assertTrue(
            _looks_like_formula_image_page(
                "Appendix C. The table presented here uses the equations "
                "used in applying the method."
            )
        )

        # Page-7-style text:
        # this points the reader TO Appendix A but is not itself
        # the appendix equation table.
        self.assertFalse(
            _looks_like_formula_image_page(
                "The formulas used to calculate the weighting of "
                "the AHP criteria is in Appendix A."
            )
        )

        # Ordinary methodology text must not trigger vision.
        self.assertFalse(
            _looks_like_formula_image_page(
                "The methodology ranks photovoltaic alternatives "
                "using TOPSIS."
            )
        )

    def test_clean_transcription_keeps_equation_rows_only(self):
        content = (
            "STEP 1: A = [1 ... a1n]\n"
            "STEP 2: explanation only\n"
            "STEP 3: CI = (lambda_max - n) / (n - 1)\n"
            "STEP 4: UNCERTAIN\n"
        )

        self.assertEqual(
            _clean_transcription(content),
            [
                "STEP 1: A = [1 ... a1n]",
                "STEP 2: CI = (lambda_max - n) / (n - 1)",
            ],
        )

    @override_settings(
        GROQ_API_KEY="fake-key",
        GROQ_VISION_MODEL="qwen/qwen3.6-27b",
    )
    @patch("apps.documents.visual_equations.httpx.post")
    def test_transcribe_image_uses_plain_vision_output(self, post):
        response = Mock()

        response.status_code = 200
        response.raise_for_status.return_value = None

        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "STEP 1: CI = (lambda_max - n) / (n - 1)\n"
                            "STEP 2: CR = CI / RCI"
                        )
                    }
                }
            ]
        }

        post.return_value = response

        rows = _transcribe_image_png(
            b"fake-png"
        )

        self.assertEqual(
            rows,
            [
                "STEP 1: CI = (lambda_max - n) / (n - 1)",
                "STEP 2: CR = CI / RCI",
            ],
        )

        self.assertEqual(
            post.call_count,
            1,
        )

        payload = post.call_args.kwargs["json"]

        self.assertEqual(
            payload["model"],
            "qwen/qwen3.6-27b",
        )

        self.assertEqual(
            payload["reasoning_effort"],
            "none",
        )

        # Formula transcription deliberately uses plain text.
        # Qwen vision previously returned JSON validation errors
        # when response_format=json was requested.
        self.assertNotIn(
            "response_format",
            payload,
        )

        # Keep the completion allowance small enough for the
        # Groq free-tier TPM budget.
        self.assertEqual(
            payload["max_completion_tokens"],
            500,
        )

    @override_settings(
        GROQ_API_KEY="fake-key",
        GROQ_VISION_MODEL="qwen/qwen3.6-27b",
    )
    @patch(
        "apps.documents.visual_equations.time.sleep"
    )
    @patch(
        "apps.documents.visual_equations.httpx.post"
    )
    def test_transcribe_image_retries_once_after_429(
        self,
        post,
        sleep,
    ):
        # First request hits the Groq TPM limit.
        rate_limited = Mock()

        rate_limited.status_code = 429
        rate_limited.headers = {}

        rate_limited.text = (
            '{"error":{"message":'
            '"Rate limit reached. '
            'Please try again in 5.5s."}}'
        )

        # Second request succeeds after waiting.
        success = Mock()

        success.status_code = 200
        success.raise_for_status.return_value = None

        success.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "STEP 1: "
                            "CI = (lambda_max - n) / (n - 1)"
                        )
                    }
                }
            ]
        }

        post.side_effect = [
            rate_limited,
            success,
        ]

        rows = _transcribe_image_png(
            b"fake-png"
        )

        self.assertEqual(
            rows,
            [
                (
                    "STEP 1: "
                    "CI = (lambda_max - n) / (n - 1)"
                )
            ],
        )

        # Exactly one retry should occur.
        self.assertEqual(
            post.call_count,
            2,
        )

        # The implementation should respect the provider's
        # requested wait period before retrying.
        sleep.assert_called_once()

        waited = sleep.call_args.args[0]

        # Provider requested 5.5 seconds and implementation
        # adds a small safety buffer.
        self.assertGreaterEqual(
            waited,
            6.0,
        )

        # The same reduced request budget should be used
        # for the retry.
        payload = post.call_args.kwargs["json"]

        self.assertEqual(
            payload["max_completion_tokens"],
            500,
        )

        self.assertEqual(
            payload["model"],
            "qwen/qwen3.6-27b",
        )

    @override_settings(
        GROQ_API_KEY="fake-key",
        GROQ_VISION_MODEL="qwen/qwen3.6-27b",
    )
    @patch(
        "apps.documents.visual_equations.time.sleep"
    )
    @patch(
        "apps.documents.visual_equations.httpx.post"
    )
    def test_transcribe_image_stops_after_second_429(
        self,
        post,
        sleep,
    ):
        """
        A persistent rate limit must not create an endless retry loop.

        Visual equation extraction is optional enrichment, so after
        one retry it should safely return an empty result.
        """
        first_429 = Mock()
        first_429.status_code = 429
        first_429.headers = {}
        first_429.text = (
            '{"error":{"message":'
            '"Please try again in 2.0s."}}'
        )

        second_429 = Mock()
        second_429.status_code = 429
        second_429.headers = {}
        second_429.text = (
            '{"error":{"message":'
            '"Please try again in 3.0s."}}'
        )

        post.side_effect = [
            first_429,
            second_429,
        ]

        rows = _transcribe_image_png(
            b"fake-png"
        )

        self.assertEqual(
            rows,
            [],
        )

        self.assertEqual(
            post.call_count,
            2,
        )

        # Sleep only happens between attempt 1 and attempt 2.
        sleep.assert_called_once()