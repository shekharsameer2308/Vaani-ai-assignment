"""Unit tests for the paper corpus ingestion and multi-setting retrieval."""

import unittest
from retrieval.chunk import chunk_pages
from retrieval.index import get_or_build_index
from retrieval.search import search


class TestRetrievalModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.bm25, cls.chunks = get_or_build_index(piece_tokens=200, overlap=40)

    def test_corpus_loaded(self):
        """Verify all three arXiv papers are represented in the passages."""
        paper_ids = {c.paper_id for c in self.chunks}
        self.assertIn("1706.03762v7", paper_ids)
        self.assertIn("2410.00037v1", paper_ids)
        self.assertIn("2507.07955v2", paper_ids)
        self.assertGreater(len(self.chunks), 20)

    def test_citation_structure(self):
        """Verify citation strings follow the required exact passage format."""
        for c in self.chunks[:5]:
            self.assertTrue(c.citation.startswith("["))
            self.assertTrue(c.citation.endswith("]"))
            self.assertIn("arXiv:", c.citation)
            self.assertIn("p.", c.citation)

    def test_retrieval_attention_is_all_you_need(self):
        """Test query over Attention paper."""
        query = "What is the formula for scaled dot product attention and how is it computed?"
        results = search(query, k=3, piece_tokens=200, overlap=40)
        self.assertGreater(len(results), 0)
        top = results[0].chunk
        self.assertEqual(top.paper_id, "1706.03762v7")

    def test_retrieval_moshi(self):
        """Test query over Moshi paper."""
        query = "What is Inner Monologue in Moshi and what do PAD and EPAD tokens do?"
        results = search(query, k=3, piece_tokens=200, overlap=40)
        self.assertGreater(len(results), 0)
        top = results[0].chunk
        self.assertEqual(top.paper_id, "2410.00037v1")

    def test_retrieval_dynamic_chunking(self):
        """Test query over Dynamic Chunking paper."""
        query = "How does the ratio loss in dynamic chunking guide the compression ratio?"
        results = search(query, k=3, piece_tokens=200, overlap=40)
        self.assertGreater(len(results), 0)
        top = results[0].chunk
        self.assertEqual(top.paper_id, "2507.07955v2")

    def test_out_of_domain_refusal_detection(self):
        """Test that absence threshold drops completely unrelated questions."""
        out_of_domain = "What is the gravitational wave signature of binary neutron star mergers?"
        results = search(out_of_domain, k=3, piece_tokens=200, overlap=40, absence_threshold=30.0)
        self.assertEqual(len(results), 0, "Out-of-domain question should be dropped by absence threshold.")

    def test_settings_depths(self):
        """Test that retrieval respects configured k."""
        q = "Transformer architecture layers"
        res_3 = search(q, k=3, piece_tokens=200, overlap=40)
        res_5 = search(q, k=5, piece_tokens=200, overlap=40)
        self.assertEqual(len(res_3), 3)
        self.assertEqual(len(res_5), 5)


if __name__ == "__main__":
    unittest.main()
