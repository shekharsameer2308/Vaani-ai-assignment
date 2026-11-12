"""Extract text from PDFs into corpus/pages.jsonl using pypdf."""

import json
import os
import re

import pypdf

PAPERS_DIR = os.path.join(os.path.dirname(__file__), "..", "papers")
OUTPUT = os.path.join(os.path.dirname(__file__), "pages.jsonl")

PAPERS = [
    {
        "id": "1706.03762v7",
        "title": "Attention Is All You Need",
        "filename": "attention_v7.pdf",
    },
    {
        "id": "2410.00037v1",
        "title": "Moshi: a speech-text foundation model for real-time dialogue",
        "filename": "moshi_v1.pdf",
    },
    {
        "id": "2507.07955v2",
        "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "filename": "dynamic_chunking_v2.pdf",
    },
]

def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to single spaces, strip leading/trailing."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def is_references_heading(line: str) -> bool:
    """Check if a line marks the beginning of the References section."""
    cleaned = line.strip().lower()
    return cleaned in ("references", "references and notes", "bibliography") or cleaned.startswith("references ") or cleaned == "7 references" or "references\n" in cleaned


def is_appendix_heading(line: str) -> bool:
    """Check if a line marks the beginning of an Appendix section."""
    cleaned = line.strip().lower()
    return (
        cleaned.startswith("appendix")
        or cleaned.startswith("supplementary")
        or cleaned.startswith("attention visualizations")
        or re.match(r"^[a-d]\s+[a-z]", cleaned) is not None
    )


def extract_paper(paper_meta):
    """Extract all pages from one PDF, return list of page records."""
    path = os.path.join(PAPERS_DIR, paper_meta["filename"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF not found: {path}")

    reader = pypdf.PdfReader(path)
    pages = []
    in_references = False

    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        text = normalize_whitespace(raw)
        page_num = i + 1

        lines = [l.strip() for l in text.split("\n") if l.strip()]
        first_few = lines[:5] if lines else []

        # Check if an appendix has started, which exits references
        if in_references and any(is_appendix_heading(l) for l in first_few):
            in_references = False

        # Check if references heading appears on this page
        if not in_references and any(is_references_heading(l) for l in lines):
            in_references = True

        pages.append({
            "paper_id": paper_meta["id"],
            "title": paper_meta["title"],
            "page": page_num,
            "text": text,
            "is_references": in_references,
            "char_count": len(text),
        })

    return pages


def main():
    all_pages = []
    for paper in PAPERS:
        pages = extract_paper(paper)
        total = len(pages)
        ref_count = sum(1 for p in pages if p["is_references"])
        short_pages = [p for p in pages if p["char_count"] < 50]

        print(f"{paper['id']}: {total} pages ({ref_count} reference pages)")
        print(f"  First 300 chars of page 1:")
        print(f"  {pages[0]['text'][:300]}")
        if short_pages:
            print(f"  Short pages (<50 chars): {[p['page'] for p in short_pages]}")
        print()

        all_pages.extend(pages)

    os.makedirs(os.path.dirname(OUTPUT) or ".", exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for page in all_pages:
            f.write(json.dumps(page, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_pages)} pages to {OUTPUT}")


if __name__ == "__main__":
    main()

