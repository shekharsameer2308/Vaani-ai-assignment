"""Chunking module to convert parsed pages into structured passage chunks."""

from dataclasses import dataclass, asdict
import json
import os
import re
from typing import List, Dict, Any


@dataclass
class Passage:
    passage_id: str
    paper_id: str
    paper_title: str
    version: str
    page_number: int
    section: str
    text: str
    citation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def chunk_text_by_words(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Sliding window chunking over words while preserving sentence boundaries where possible."""
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [text.strip()]

    chunks = []
    step = max(1, chunk_size - overlap)
    i = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunk_str = " ".join(chunk_words).strip()
        if chunk_str:
            chunks.append(chunk_str)
        if i + chunk_size >= len(words):
            break
        i += step
    return chunks


def build_passages_for_setting(
    parsed_dir: str = "data/parsed",
    chunk_size_words: int = 250,
    overlap_words: int = 40,
) -> List[Passage]:
    """Generate structured passages from all parsed paper files for given chunk parameters."""
    passages: List[Passage] = []
    if not os.path.exists(parsed_dir):
        raise FileNotFoundError(f"Directory {parsed_dir} does not exist.")

    files = [f for f in os.listdir(parsed_dir) if f.endswith(".json")]
    files.sort()

    for filename in files:
        filepath = os.path.join(parsed_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        paper_id = data["arxiv_id"]
        paper_title = data["title"]
        version = data.get("version", "")

        for page in data["pages"]:
            page_num = page["page_number"]
            section = page.get("section", "General")
            page_text = page["text"]

            text_chunks = chunk_text_by_words(page_text, chunk_size_words, overlap_words)
            for idx, chunk in enumerate(text_chunks):
                passage_id = f"{paper_id}_p{page_num}_c{idx}"
                citation = f"[{paper_title} (arXiv:{paper_id}), p. {page_num}, Sec. '{section}']"
                passages.append(
                    Passage(
                        passage_id=passage_id,
                        paper_id=paper_id,
                        paper_title=paper_title,
                        version=version,
                        page_number=page_num,
                        section=section,
                        text=chunk,
                        citation=citation,
                    )
                )

    return passages

