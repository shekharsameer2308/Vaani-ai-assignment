"""Retrieval setting configurations for parameter sweeps."""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class RetrievalSetting:
    name: str
    chunk_size_words: int
    overlap_words: int
    top_k: int
    use_reranking: bool
    description: str


# The 3 distinct retrieval settings required for parameter sweep
RETRIEVAL_SETTINGS: Dict[str, RetrievalSetting] = {
    "setting_a_250_k3_norerank": RetrievalSetting(
        name="setting_a_250_k3_norerank",
        chunk_size_words=250,
        overlap_words=40,
        top_k=3,
        use_reranking=False,
        description="Small chunks (250 words), top-3 passages, BM25 only without re-ranking",
    ),
    "setting_b_500_k5_norerank": RetrievalSetting(
        name="setting_b_500_k5_norerank",
        chunk_size_words=500,
        overlap_words=75,
        top_k=5,
        use_reranking=False,
        description="Medium chunks (500 words), top-5 passages, BM25 only without re-ranking",
    ),
    "setting_c_500_k10_rerank": RetrievalSetting(
        name="setting_c_500_k10_rerank",
        chunk_size_words=500,
        overlap_words=75,
        top_k=10,
        use_reranking=True,
        description="Medium chunks (500 words), fetch top-10, apply lexical-semantic cross reranking",
    ),
}

