"""Core multi-setting RAG Retriever over arXiv paper passages."""

import math
import re
from typing import List, Dict, Any, Optional, Tuple
from rank_bm25 import BM25Okapi

from src.ingest.chunker import Passage, build_passages_for_setting
from src.retrieval.settings import RetrievalSetting, RETRIEVAL_SETTINGS


STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with", "by",
    "from", "of", "about", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "what", "which", "who", "whom",
    "this", "that", "these", "those", "how", "why", "where", "when", "can", "could",
    "should", "would", "does", "we", "our", "you", "your", "it", "its", "as"
}


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens."""
    return re.findall(r"\w+", text.lower())


def get_informative_tokens(tokens: List[str]) -> List[str]:
    """Filter out stopwords to get key search terms."""
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


class PaperRetriever:
    """Configurable retriever supporting BM25 search, metadata filtering, and re-ranking."""

    def __init__(self, setting: RetrievalSetting, parsed_dir: str = "data/parsed"):
        self.setting = setting
        self.parsed_dir = parsed_dir
        self.passages: List[Passage] = build_passages_for_setting(
            parsed_dir=parsed_dir,
            chunk_size_words=setting.chunk_size_words,
            overlap_words=setting.overlap_words,
        )
        self.corpus_tokens = [tokenize(p.text) for p in self.passages]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def _rerank(self, query: str, candidate_indices: List[int], bm25_scores: List[float]) -> List[Tuple[int, float]]:
        """Lexical and semantic cross re-ranker."""
        query_tokens = tokenize(query)
        info_query_tokens = get_informative_tokens(query_tokens)
        query_set = set(info_query_tokens) if info_query_tokens else set(query_tokens)
        query_lower = query.lower()

        scores_list = [float(s) for s in bm25_scores]
        max_bm25 = max(scores_list) if scores_list and max(scores_list) > 0 else 1.0

        scored_candidates = []
        for idx in candidate_indices:
            passage = self.passages[idx]
            raw_bm25 = float(bm25_scores[idx])
            norm_bm25 = raw_bm25 / max_bm25 if max_bm25 > 0 else 0.0

            p_text_lower = passage.text.lower()
            p_tokens = self.corpus_tokens[idx]
            p_set = set(p_tokens)

            # Informative term coverage
            matched_terms = query_set.intersection(p_set)
            term_coverage = len(matched_terms) / max(1, len(query_set))

            # Exact sequence match bonus
            exact_phrase_bonus = 0.0
            if len(info_query_tokens) >= 2:
                for n in (3, 2):
                    for i in range(len(info_query_tokens) - n + 1):
                        ngram = " ".join(info_query_tokens[i : i + n])
                        if ngram in p_text_lower:
                            exact_phrase_bonus += 0.20 * n

            # Section and title match bonus
            section_tokens = set(tokenize(passage.section + " " + passage.paper_title))
            sec_overlap = len(query_set.intersection(section_tokens)) / max(1, len(query_set))

            # Combine and heavily scale by term coverage so out-of-domain queries drop
            base_score = (0.45 * norm_bm25) + (0.30 * term_coverage) + (0.15 * min(1.0, exact_phrase_bonus)) + (0.10 * sec_overlap)
            final_score = base_score * (term_coverage ** 0.8)

            scored_candidates.append((idx, final_score))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates

    def retrieve(
        self,
        query: str,
        paper_filter: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve the most relevant passages for a query."""
        k = top_k or self.setting.top_k
        query_tokens = tokenize(query)
        info_query_tokens = get_informative_tokens(query_tokens)
        query_set = set(info_query_tokens) if info_query_tokens else set(query_tokens)

        if not query_tokens:
            return []

        raw_scores = self.bm25.get_scores(query_tokens)

        # Filter indices by paper_id if specified
        if paper_filter:
            eligible_indices = [
                i for i, p in enumerate(self.passages) if p.paper_id == paper_filter
            ]
        else:
            eligible_indices = list(range(len(self.passages)))

        if not eligible_indices:
            return []

        if self.setting.use_reranking:
            initial_k = min(len(eligible_indices), max(k * 3, 15))
            top_candidates = sorted(eligible_indices, key=lambda i: float(raw_scores[i]), reverse=True)[:initial_k]
            ranked = self._rerank(query, top_candidates, raw_scores)[:k]
        else:
            top_candidates = sorted(eligible_indices, key=lambda i: float(raw_scores[i]), reverse=True)[:k]
            cand_scores = [float(raw_scores[i]) for i in top_candidates]
            max_s = max(cand_scores) if cand_scores and max(cand_scores) > 0 else 1.0
            ranked = []
            for i in top_candidates:
                p_set = set(self.corpus_tokens[i])
                coverage = len(query_set.intersection(p_set)) / max(1, len(query_set))
                score = (float(raw_scores[i]) / max_s) * (coverage ** 0.8)
                ranked.append((i, score))

        results = []
        for idx, score in ranked:
            p = self.passages[idx]
            p_set = set(self.corpus_tokens[idx])
            matched_terms = list(query_set.intersection(p_set))
            coverage = len(matched_terms) / max(1, len(query_set))

            results.append({
                "passage_id": p.passage_id,
                "paper_id": p.paper_id,
                "paper_title": p.paper_title,
                "version": p.version,
                "page_number": p.page_number,
                "section": p.section,
                "text": p.text,
                "citation": p.citation,
                "relevance_score": round(float(score), 4),
                "query_coverage": round(float(coverage), 4),
                "matched_terms": matched_terms,
            })
        return results

    def is_answerable(self, results: List[Dict[str, Any]], threshold: float = 0.35, min_coverage: float = 0.40) -> bool:
        """Determines whether the retrieved results possess sufficient confidence to ground an answer."""
        if not results:
            return False
        top = results[0]
        # Must pass both minimum relevance score and minimum term coverage
        return top["relevance_score"] >= threshold and top["query_coverage"] >= min_coverage


def get_retriever_for_setting(setting_name: str) -> PaperRetriever:
    """Factory helper to instantiate a retriever by setting name."""
    if setting_name not in RETRIEVAL_SETTINGS:
        raise ValueError(f"Unknown setting '{setting_name}'. Available: {list(RETRIEVAL_SETTINGS.keys())}")
    return PaperRetriever(RETRIEVAL_SETTINGS[setting_name])
