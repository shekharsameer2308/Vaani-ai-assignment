"""Build, save, and load BM25 index with sidecar parameter metadata."""

from dataclasses import asdict
import json
import os
import pickle
import re
from typing import Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi
from retrieval.chunk import Chunk, chunk_pages

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
INDEX_PATH = os.path.join(DATA_DIR, "index.pkl")
METADATA_PATH = os.path.join(DATA_DIR, "index_meta.json")


def tokenize_text(text: str) -> List[str]:
    """Basic lowercased alphanumeric tokenizer for BM25 matching."""
    return re.findall(r"\w+", text.lower())


def save_index(
    bm25: BM25Okapi,
    chunks: List[Chunk],
    piece_tokens: int,
    overlap: int,
    index_path: str = INDEX_PATH,
    meta_path: str = METADATA_PATH,
):
    """Serialize BM25 index, chunks, and sidecar parameter metadata."""
    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)

    with open(index_path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    meta = {
        "piece_tokens": piece_tokens,
        "overlap": overlap,
        "total_chunks": len(chunks),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_index(
    index_path: str = INDEX_PATH,
    meta_path: str = METADATA_PATH,
) -> Optional[Tuple[BM25Okapi, List[Chunk], Dict[str, any]]]:
    """Load BM25 index and chunks if existing."""
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        return None

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        with open(index_path, "rb") as f:
            data = pickle.load(f)
        return data["bm25"], data["chunks"], meta
    except Exception:
        return None


def get_or_build_index(
    piece_tokens: int = 200,
    overlap: int = 40,
    force_rebuild: bool = False,
    index_path: str = INDEX_PATH,
    meta_path: str = METADATA_PATH,
) -> Tuple[BM25Okapi, List[Chunk]]:
    """Return index from cache if parameters match, otherwise build and save once."""
    if not force_rebuild:
        loaded = load_index(index_path, meta_path)
        if loaded is not None:
            bm25, chunks, meta = loaded
            if meta.get("piece_tokens") == piece_tokens and meta.get("overlap") == overlap:
                return bm25, chunks

    # Build new index
    chunks = chunk_pages(piece_tokens=piece_tokens, overlap=overlap)
    corpus_tokens = [tokenize_text(c.text) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)

    save_index(bm25, chunks, piece_tokens, overlap, index_path, meta_path)
    return bm25, chunks


load_or_build_index = get_or_build_index


if __name__ == "__main__":
    bm25, chunks = get_or_build_index(piece_tokens=200, overlap=40)
    print(f"Index built and cached with {len(chunks)} chunks.")
