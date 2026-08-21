from django.test import SimpleTestCase

from apps.documents.chunking import (
    _overlap_tail,
    chunk_page,
)


class ChunkingTests(SimpleTestCase):

    def test_empty_page_returns_no_chunks(self):
        self.assertEqual(
            chunk_page("  \n"),
            [],
        )

    def test_long_page_splits(self):
        text = "\n\n".join(
            [
                "paragraph " + ("evidence " * 200)
                for _ in range(4)
            ]
        )

        chunks = chunk_page(
            text,
            target_chars=1000,
            overlap_chars=100,
        )

        self.assertGreater(
            len(chunks),
            1,
        )

    def test_overlap_tail_does_not_start_mid_word(self):
        """
        Regression test for corruption such as:

            solar farms
                 ↓
            ar farms

        caused by a fixed character overlap beginning
        inside the word "solar".
        """
        text = "prefix solar farms Thailand"

        overlap = _overlap_tail(
            text,
            overlap_chars=17,
        )

        self.assertEqual(
            overlap,
            "solar farms Thailand",
        )

    def test_overlap_tail_returns_empty_when_disabled(self):
        text = "research evidence from uploaded papers"

        self.assertEqual(
            _overlap_tail(
                text,
                overlap_chars=0,
            ),
            "",
        )

    def test_overlap_tail_keeps_short_text_intact(self):
        text = "solar farms Thailand"

        overlap = _overlap_tail(
            text,
            overlap_chars=100,
        )

        self.assertEqual(
            overlap,
            text,
        )

    def test_overlap_prefers_sentence_boundary(self):
        text = (
            "Earlier research context. "
            "Solar farms are evaluated using GIS and AHP."
        )

        overlap = _overlap_tail(
            text,
            overlap_chars=55,
        )

        self.assertEqual(
            overlap,
            "Solar farms are evaluated using GIS and AHP.",
        )