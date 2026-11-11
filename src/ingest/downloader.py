"""arXiv paper downloader for exact paper versions."""

import logging
import os
import time
from typing import Dict, Optional
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PAPER_REGISTRY: Dict[str, Dict[str, str]] = {
    "1706.03762v7": {
        "title": "Attention Is All You Need",
        "version": "v7",
        "url": "https://arxiv.org/pdf/1706.03762v7.pdf",
        "filename": "1706.03762v7.pdf",
    },
    "2410.00037v1": {
        "title": "Moshi: a speech-text foundation model for real-time dialogue",
        "version": "v1",
        "url": "https://arxiv.org/pdf/2410.00037v1.pdf",
        "filename": "2410.00037v1.pdf",
    },
    "2507.07955v2": {
        "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "version": "v2",
        "url": "https://arxiv.org/pdf/2507.07955v2.pdf",
        "filename": "2507.07955v2.pdf",
    },
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36 (TheRelayRAG/1.0)"
    ),
    "Accept": "application/pdf,*/*",
}


def download_paper(paper_id: str, output_dir: str = "data/raw_papers", max_retries: int = 3) -> str:
    """Download an exact arXiv version with retries and validation."""
    if paper_id not in PAPER_REGISTRY:
        raise ValueError(f"Unknown paper ID '{paper_id}'. Available: {list(PAPER_REGISTRY.keys())}")

    meta = PAPER_REGISTRY[paper_id]
    os.makedirs(output_dir, exist_ok=True)
    target_path = os.path.join(output_dir, meta["filename"])

    if os.path.exists(target_path) and os.path.getsize(target_path) > 10000:
        logger.info(f"Paper {paper_id} already cached at {target_path} ({os.path.getsize(target_path)} bytes)")
        return target_path

    url = meta["url"]
    logger.info(f"Downloading {paper_id} ({meta['title']}) from {url}...")

    session = requests.Session()
    session.headers.update(HEADERS)

    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, stream=True, timeout=60, allow_redirects=True)
            response.raise_for_status()

            # Verify it's a PDF or contains %PDF header
            first_chunk = response.raw.read(1024)
            if b"%PDF" not in first_chunk and b"PDF" not in first_chunk:
                raise ValueError(f"Downloaded content from {url} is not a valid PDF (got: {first_chunk[:50]}...)")

            with open(target_path, "wb") as f:
                f.write(first_chunk)
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)

            file_size = os.path.getsize(target_path)
            logger.info(f"Successfully downloaded {paper_id} to {target_path} ({file_size} bytes)")
            return target_path

        except Exception as e:
            logger.warning(f"Attempt {attempt}/{max_retries} failed for {paper_id}: {e}")
            if attempt < max_retries:
                time.sleep(2 * attempt)
            else:
                raise RuntimeError(f"Failed to download {paper_id} after {max_retries} attempts.") from e

    return target_path


def download_all_papers(output_dir: str = "data/raw_papers") -> Dict[str, str]:
    """Download all 3 mandatory papers."""
    results = {}
    for paper_id in PAPER_REGISTRY:
        path = download_paper(paper_id, output_dir=output_dir)
        results[paper_id] = path
    return results


if __name__ == "__main__":
    download_all_papers()

