from django.test import SimpleTestCase

from apps.documents.chunking import chunk_page


class ChunkingTests(SimpleTestCase):
    def test_empty_page_returns_no_chunks(self):
        self.assertEqual(chunk_page("  \n"), [])

    def test_long_page_splits(self):
        text = "\n\n".join(
            ["paragraph " + ("evidence " * 200) for _ in range(4)]
        )
        chunks = chunk_page(text, target_chars=1000, overlap_chars=100)
        self.assertGreater(len(chunks), 1)
