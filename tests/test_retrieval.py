"""Unit tests for the paper corpus ingestion and multi-setting retrieval."""

import unittest
from src.retrieval.settings import RETRIEVAL_SETTINGS
from src.retrieval.retriever import get_retriever_for_setting, PaperRetriever


class TestRetrievalModule(unittest.TestCase):

    def setUp(self):
        self.retriever_c = get_retriever_for_setting("setting_c_500_k10_rerank")
        self.retriever_a = get_retriever_for_setting("setting_a_250_k3_norerank")
        self.retriever_b = get_retriever_for_setting("setting_b_500_k5_norerank")

    def test_corpus_loaded(self):
        """Verify all three arXiv papers are represented in the passages."""
        paper_ids = {p.paper_id for p in self.retriever_c.passages}
        self.assertIn("1706.03762v7", paper_ids)
        self.assertIn("2410.00037v1", paper_ids)
        self.assertIn("2507.07955v2", paper_ids)
        self.assertGreater(len(self.retriever_c.passages), 20)

    def test_citation_structure(self):
        """Verify citation strings follow the required exact passage format."""
        for p in self.retriever_c.passages[:5]:
            self.assertTrue(p.citation.startswith("["))
            self.assertTrue(p.citation.endswith("]"))
            self.assertIn("arXiv:", p.citation)
            self.assertIn("p.", p.citation)
            self.assertIn("Sec.", p.citation)

    def test_retrieval_attention_is_all_you_need(self):
        """Test query over Attention paper."""
        query = "What is the formula for scaled dot product attention and how is it computed?"
        results = self.retriever_c.retrieve(query)
        self.assertGreater(len(results), 0)
        top = results[0]
        self.assertEqual(top["paper_id"], "1706.03762v7")
        self.assertIn("3.2.1", top["section"])
        self.assertTrue(self.retriever_c.is_answerable(results))

    def test_retrieval_moshi(self):
        """Test query over Moshi paper."""
        query = "What is Inner Monologue in Moshi and what do PAD and EPAD tokens do?"
        results = self.retriever_c.retrieve(query)
        self.assertGreater(len(results), 0)
        top = results[0]
        self.assertEqual(top["paper_id"], "2410.00037v1")
        self.assertTrue(self.retriever_c.is_answerable(results))

    def test_retrieval_dynamic_chunking(self):
        """Test query over Dynamic Chunking paper."""
        query = "How does the ratio loss in dynamic chunking guide the compression ratio?"
        results = self.retriever_c.retrieve(query)
        self.assertGreater(len(results), 0)
        top = results[0]
        self.assertEqual(top["paper_id"], "2507.07955v2")
        self.assertTrue(self.retriever_c.is_answerable(results))

    def test_out_of_domain_refusal_detection(self):
        """Test that completely unrelated questions are deemed unanswerable."""
        out_of_domain = "What is the gravitational wave signature of binary neutron star mergers?"
        results = self.retriever_c.retrieve(out_of_domain)
        is_ans = self.retriever_c.is_answerable(results)
        self.assertFalse(is_ans, "Out-of-domain question should be deemed unanswerable.")

    def test_settings_depths(self):
        """Test that each setting respects its configured top_k."""
        q = "Transformer architecture layers"
        res_a = self.retriever_a.retrieve(q)
        res_b = self.retriever_b.retrieve(q)
        res_c = self.retriever_c.retrieve(q)
        self.assertEqual(len(res_a), 3)
        self.assertEqual(len(res_b), 5)
        self.assertEqual(len(res_c), 10)


if __name__ == "__main__":
    unittest.main()

