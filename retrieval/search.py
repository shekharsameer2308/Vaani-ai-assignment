"""Search the paper index using BM25 with absence gating and CLI interface."""

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from retrieval.chunk import Chunk
from retrieval.index import get_or_build_index, tokenize_text


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    rank: int


def search(
    query: str,
    k: int = 4,
    piece_tokens: int = 200,
    overlap: int = 40,
    absence_threshold: Optional[float] = None,  # Parameter: default None per brief. Do not set a value.
    rerank: str = "none",                       # Parameter: "none" for now, TODO for later.
) -> List[SearchResult]:
    """Retrieve top-k chunks from the cached BM25 index."""
    bm25, chunks = get_or_build_index(piece_tokens=piece_tokens, overlap=overlap)

    query_tokens = tokenize_text(query)
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)

    results: List[SearchResult] = []
    for rank, idx in enumerate(ranked_indices[:k], start=1):
        score_val = float(scores[idx])

        # Absence threshold check: if threshold provided and score below, drop
        if absence_threshold is not None and score_val < absence_threshold:
            continue

        results.append(SearchResult(chunk=chunks[idx], score=round(score_val, 4), rank=rank))

    # TODO: Implement re-ranking when rerank != "none"
    if rerank != "none":
        pass

    return results


# Functional alias for retrieve
retrieve = search


def main():
    parser = argparse.ArgumentParser(description="Search arXiv research paper index")
    parser.add_argument("query", type=str, help="Question or query string")
    parser.add_argument("--piece", type=int, default=200, help="Chunk piece size in whitespace tokens")
    parser.add_argument("--overlap", type=int, default=40, help="Chunk overlap in whitespace tokens")
    parser.add_argument("--k", type=int, default=4, help="Number of chunks to fetch")
    parser.add_argument("--absence-threshold", type=float, default=None, help="Absence threshold (default None)")
    parser.add_argument("--rerank", type=str, default="none", help="Reranking method (default 'none')")

    args = parser.parse_args()
    res = search(
        query=args.query,
        k=args.k,
        piece_tokens=args.piece,
        overlap=args.overlap,
        absence_threshold=args.absence_threshold,
        rerank=args.rerank,
    )

    print(f"\nQuery: '{args.query}' (found {len(res)} chunks)\n")
    for r in res:
        print(f"[{r.rank}] Score: {r.score} | ID: {r.chunk.chunk_id}")
        print(f"    Citation: {r.chunk.citation}")
        print(f"    Snippet: {r.chunk.text[:220]}...")
        print()


if __name__ == "__main__":
    main()
