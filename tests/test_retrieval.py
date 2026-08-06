from types import SimpleNamespace
from django.test import SimpleTestCase
from apps.retrieval.lexical import bm25_search
class RetrievalTests(SimpleTestCase):
    def test_exact_term_ranks_first(self):
        docs=[SimpleNamespace(content="TLS fingerprinting with JA3"),SimpleNamespace(content="BGP route selection")]
        hits=bm25_search("JA3 fingerprinting",docs)
        self.assertIs(hits[0].item, docs[0])
