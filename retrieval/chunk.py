"""Chunk pages from corpus/pages.jsonl using whitespace token counting, skipping references."""

from dataclasses import dataclass, asdict
import json
import os
from typing import Dict, List, Optional

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "corpus", "pages.jsonl")


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    paper_title: str
    page_start: int
    page_end: int
    text: str
    token_count: int
    citation: str

    def to_dict(self) -> Dict[str, any]:
        return asdict(self)


def chunk_pages(
    corpus_path: str = CORPUS_PATH,
    piece_tokens: int = 200,  # Parameter: chunk target size
    overlap: int = 40,        # Parameter: chunk overlap
) -> List[Chunk]:
    """Sliding-window chunker using whitespace token counting, skipping reference pages.
    
    Per design decision: Whitespace splitting (.split()) is used for all chunking.
    No subword or BPE tokenizer (e.g. tiktoken) is used here.
    """
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}. Run corpus/extract.py first.")

    # Load pages grouped by paper
    pages_by_paper: Dict[str, List[Dict]] = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            p = json.loads(line)
            pages_by_paper.setdefault(p["paper_id"], []).append(p)

    chunks: List[Chunk] = []
    global_chunk_idx = 1

    for paper_id, pages in pages_by_paper.items():
        paper_title = pages[0]["title"]

        # Aggregate word tokens along with page numbers for pages that are NOT references
        token_entries = []  # List of (word_str, page_number)
        for page in pages:
            if page.get("is_references", False):
                # Skip bibliography / reference pages per requirement
                continue
            # Whitespace token counting
            words = page["text"].split()
            page_num = page["page"]
            for w in words:
                token_entries.append((w, page_num))

        if not token_entries:
            continue

        step = max(1, piece_tokens - overlap)
        i = 0
        while i < len(token_entries):
            slice_entries = token_entries[i : i + piece_tokens]
            chunk_words = [w for w, _ in slice_entries]
            page_start = slice_entries[0][1]
            page_end = slice_entries[-1][1]

            chunk_text = " ".join(chunk_words)
            chunk_id = f"dc-{global_chunk_idx:04d}"
            global_chunk_idx += 1

            page_str = f"p. {page_start}" if page_start == page_end else f"pp. {page_start}-{page_end}"
            citation = f"[{paper_title} (arXiv:{paper_id}), {page_str}]"

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    paper_id=paper_id,
                    paper_title=paper_title,
                    page_start=page_start,
                    page_end=page_end,
                    text=chunk_text,
                    token_count=len(chunk_words),
                    citation=citation,
                )
            )

            if i + piece_tokens >= len(token_entries):
                break
            i += step

    return chunks


if __name__ == "__main__":
    c = chunk_pages(piece_tokens=200, overlap=40)
    print(f"Generated {len(c)} chunks with piece_tokens=200, overlap=40")
    print(f"First chunk: {c[0].chunk_id} ({c[0].citation}): {c[0].text[:120]}...")
